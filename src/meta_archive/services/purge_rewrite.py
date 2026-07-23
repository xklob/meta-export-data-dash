from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import orjson
from sqlalchemy import select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from meta_archive.models import (
    ActivityRecord,
    AttachmentOccurrence,
    Blob,
    ConversationSegment,
    Message,
    SourceChunk,
    SourceFile,
    SourceRecord,
)
from meta_archive.paths import PRIVATE_FILE_MODE, ArchiveLayout
from meta_archive.services.hashing import sha256_file
from meta_archive.services.identifiers import canonical_fingerprint

TOMBSTONE_VERSION = 1
TOMBSTONE = {"meta_archive_purged": {"version": TOMBSTONE_VERSION}}
BINARY_TOMBSTONE_BYTES = b'{"meta_archive_purged":{"kind":"binary","version":1}}\n'
_HASH_BLOCK_SIZE = 8 * 1024 * 1024


class RawSourceRewriteError(RuntimeError):
    """A raw-source rewrite cannot be proven complete and safe."""


def _safe_workspace_path(layout: ArchiveLayout, path: Path, *, label: str) -> Path:
    root = layout.root.resolve()
    try:
        path.relative_to(layout.root)
    except ValueError as exc:
        raise RawSourceRewriteError(f"{label} is not lexically inside the archive") from exc
    resolved = path.resolve(strict=False)
    if resolved == root or root not in resolved.parents:
        raise RawSourceRewriteError(f"{label} escapes the archive")
    cursor = path.parent
    while cursor != layout.root:
        if cursor.is_symlink():
            raise RawSourceRewriteError(f"{label} traverses a symlinked directory")
        if cursor.parent == cursor:
            raise RawSourceRewriteError(f"{label} has no archive-root ancestor")
        cursor = cursor.parent
    if layout.root.is_symlink():
        raise RawSourceRewriteError("Archive root must not be a symlink during hard purge")
    return path


@dataclass(frozen=True, slots=True)
class FingerprintUpdate:
    source_record_id: str
    message_id: str | None
    activity_id: str | None
    old_fingerprint: str
    new_fingerprint: str
    new_activity_payload: Any | None = None


@dataclass(frozen=True, slots=True)
class RawSourceRewrite:
    source_file_id: int
    source: str
    chunk: str
    relative_path: str
    physical_path: Path | None
    mode: int
    mtime_ns: int
    media_type: str | None
    old_sha256: str
    old_size_bytes: int
    new_sha256: str
    new_size_bytes: int
    new_bytes: bytes
    binary: bool
    pointers: tuple[str, ...]
    selected_source_record_ids: tuple[str, ...]
    fingerprint_updates: tuple[FingerprintUpdate, ...]

    def revision_record(self) -> dict[str, Any]:
        return {
            "source_file_id": self.source_file_id,
            "source": self.source,
            "old_sha256": self.old_sha256,
            "old_size_bytes": self.old_size_bytes,
            "new_sha256": self.new_sha256,
            "new_size_bytes": self.new_size_bytes,
            "binary": self.binary,
            "pointers": list(self.pointers),
            "selected_source_record_ids": list(self.selected_source_record_ids),
            "fingerprint_updates": [
                {
                    "source_record_id": item.source_record_id,
                    "activity_id": item.activity_id,
                    "old_fingerprint": item.old_fingerprint,
                    "new_fingerprint": item.new_fingerprint,
                }
                for item in self.fingerprint_updates
            ],
        }


@dataclass(frozen=True, slots=True)
class RawRewritePlan:
    rewrites: tuple[RawSourceRewrite, ...]
    original_manifest_rows: tuple[dict[str, Any], ...]
    relocation_manifest_rows: tuple[dict[str, Any], ...]
    obsolete_blob_ids: tuple[str, ...]

    @property
    def selected_source_record_ids(self) -> set[str]:
        return {
            source_record_id
            for rewrite in self.rewrites
            for source_record_id in rewrite.selected_source_record_ids
        }

    def revision_records(self) -> list[dict[str, Any]]:
        return [rewrite.revision_record() for rewrite in self.rewrites]


def _validate_regular_file_identity(
    path: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    label: str,
) -> None:
    """Hash one non-symlinked inode and prove the pathname still names that inode."""

    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError as exc:
        raise RawSourceRewriteError(f"{label} is missing") from exc
    except OSError as exc:
        raise RawSourceRewriteError(f"{label} cannot be opened safely") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RawSourceRewriteError(f"{label} is not a regular file")
        if before.st_size != expected_size:
            raise RawSourceRewriteError(f"{label} size changed after inventory")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, _HASH_BLOCK_SIZE):
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise RawSourceRewriteError(f"{label} changed while it was being validated")
        if digest.hexdigest() != expected_sha256:
            raise RawSourceRewriteError(f"{label} SHA-256 changed after inventory")
        try:
            current = path.lstat()
        except FileNotFoundError as exc:
            raise RawSourceRewriteError(f"{label} disappeared during validation") from exc
        except OSError as exc:
            raise RawSourceRewriteError(f"{label} cannot be inspected safely") from exc
        if not stat.S_ISREG(current.st_mode):
            raise RawSourceRewriteError(f"{label} became a non-regular file")
        if (current.st_dev, current.st_ino) != (after.st_dev, after.st_ino):
            raise RawSourceRewriteError(f"{label} was replaced during validation")
    finally:
        os.close(descriptor)


def validate_raw_rewrite_physical_source(
    layout: ArchiveLayout,
    rewrite: RawSourceRewrite,
) -> None:
    """Validate the live physical source identity recorded in a rewrite plan."""

    if rewrite.physical_path is None:
        return
    path = _safe_workspace_path(
        layout,
        rewrite.physical_path,
        label="physical raw source path",
    )
    _validate_regular_file_identity(
        path,
        expected_size=rewrite.old_size_bytes,
        expected_sha256=rewrite.old_sha256,
        label=f"Physical raw source {rewrite.source}",
    )


def validate_raw_rewrite_physical_sources(
    layout: ArchiveLayout,
    plan: RawRewritePlan,
) -> None:
    """Validate every live physical source in a persisted or freshly built plan."""

    for rewrite in plan.rewrites:
        validate_raw_rewrite_physical_source(layout, rewrite)


def is_purge_tombstone(value: object) -> bool:
    return value == TOMBSTONE


def _safe_relative(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise RawSourceRewriteError(f"Manifest field {field!r} must be a string")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or "\x00" in value or value in {"", "."}:
        raise RawSourceRewriteError(f"Unsafe {field} path in source manifest")
    return pure.as_posix()


def _load_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise RawSourceRewriteError(f"Required {label} is missing or unsafe")
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise RawSourceRewriteError(f"{label} line {line_number} is not a JSON object")
                rows.append(row)
    except (OSError, json.JSONDecodeError) as exc:
        raise RawSourceRewriteError(f"Required {label} is invalid") from exc
    return rows


def _load_complete_manifests(
    layout: ArchiveLayout,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    originals = _load_jsonl(
        _safe_workspace_path(
            layout, layout.original_files_manifest, label="original-files manifest"
        ),
        label="original-files manifest",
    )
    relocations = _load_jsonl(
        _safe_workspace_path(layout, layout.relocation_manifest, label="relocation manifest"),
        label="relocation manifest",
    )
    original_by_source: dict[str, dict[str, Any]] = {}
    for row in originals:
        chunk = _safe_relative(row.get("chunk"), field="chunk")
        if "/" in chunk:
            raise RawSourceRewriteError("Source manifest chunk is not an exact root entry")
        relative = _safe_relative(row.get("relative_path"), field="relative_path")
        source = f"{chunk}/{relative}"
        digest = row.get("sha256")
        if (
            source in original_by_source
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not isinstance(row.get("size_bytes"), int)
            or not isinstance(row.get("mode"), int)
            or not isinstance(row.get("mtime_ns"), int)
        ):
            raise RawSourceRewriteError(f"Invalid original manifest record for {source}")
        original_by_source[source] = row

    relocation_by_source: dict[str, dict[str, Any]] = {}
    for row in relocations:
        source = _safe_relative(row.get("source"), field="source")
        if source in relocation_by_source:
            raise RawSourceRewriteError(f"Duplicate relocation coordinate: {source}")
        relocation_by_source[source] = row
    if original_by_source.keys() != relocation_by_source.keys():
        raise RawSourceRewriteError("Source manifests do not contain the same coordinates")
    for source, original in original_by_source.items():
        relocation = relocation_by_source[source]
        digest = str(original["sha256"])
        expected_blob = layout.blob_path(digest).relative_to(layout.root).as_posix()
        for key in ("chunk", "relative_path", "sha256", "size_bytes", "mode", "mtime_ns"):
            if relocation.get(key) != original.get(key):
                raise RawSourceRewriteError(f"Source manifest metadata differs at {source}: {key}")
        if relocation.get("blob") != expected_blob:
            raise RawSourceRewriteError(f"Relocation blob mapping is invalid at {source}")
    return originals, relocations, original_by_source


def _unescape_pointer_token(token: str) -> str:
    result: list[str] = []
    index = 0
    while index < len(token):
        if token[index] != "~":
            result.append(token[index])
            index += 1
            continue
        if index + 1 >= len(token) or token[index + 1] not in {"0", "1"}:
            raise RawSourceRewriteError("Catalog contains an invalid JSON pointer escape")
        result.append("~" if token[index + 1] == "0" else "/")
        index += 2
    return "".join(result)


def _escape_pointer_token(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def _pointer_tokens(pointer: str) -> tuple[str, ...]:
    if pointer == "":
        return ()
    if not pointer.startswith("/"):
        raise RawSourceRewriteError(f"Catalog contains an invalid JSON pointer: {pointer!r}")
    return tuple(_unescape_pointer_token(token) for token in pointer[1:].split("/"))


def _resolve_pointer(payload: Any, pointer: str) -> Any:
    current = payload
    for token in _pointer_tokens(pointer):
        if isinstance(current, list):
            if not token.isdigit() or (len(token) > 1 and token.startswith("0")):
                raise RawSourceRewriteError(
                    f"JSON pointer does not address an array index: {pointer}"
                )
            index = int(token)
            if index >= len(current):
                raise RawSourceRewriteError(f"JSON pointer is outside its source array: {pointer}")
            current = current[index]
        elif isinstance(current, dict):
            if token not in current:
                raise RawSourceRewriteError(f"JSON pointer is absent from its source: {pointer}")
            current = current[token]
        else:
            raise RawSourceRewriteError(f"JSON pointer traverses a scalar value: {pointer}")
    return current


def _replace_pointer(payload: Any, pointer: str) -> Any:
    tokens = _pointer_tokens(pointer)
    if not tokens:
        return json.loads(json.dumps(TOMBSTONE))
    parent = payload
    for token in tokens[:-1]:
        if isinstance(parent, list):
            if not token.isdigit() or int(token) >= len(parent):
                raise RawSourceRewriteError(f"JSON pointer is outside its source array: {pointer}")
            parent = parent[int(token)]
        elif isinstance(parent, dict) and token in parent:
            parent = parent[token]
        else:
            raise RawSourceRewriteError(f"JSON pointer is absent from its source: {pointer}")
    final = tokens[-1]
    replacement = json.loads(json.dumps(TOMBSTONE))
    if isinstance(parent, list):
        if not final.isdigit() or int(final) >= len(parent):
            raise RawSourceRewriteError(f"JSON pointer is outside its source array: {pointer}")
        parent[int(final)] = replacement
    elif isinstance(parent, dict) and final in parent:
        parent[final] = replacement
    else:
        raise RawSourceRewriteError(f"JSON pointer is absent from its source: {pointer}")
    return payload


def _canonical_value_sha256(value: Any) -> str:
    return hashlib.sha256(orjson.dumps(value, option=orjson.OPT_SORT_KEYS)).hexdigest()


def _minimal_pointers(pointers: set[str]) -> tuple[str, ...]:
    ordered = sorted(pointers, key=lambda value: (len(_pointer_tokens(value)), value))
    result: list[str] = []
    for pointer in ordered:
        tokens = _pointer_tokens(pointer)
        if any(
            tokens[: len(_pointer_tokens(parent))] == _pointer_tokens(parent) for parent in result
        ):
            continue
        result.append(pointer)
    return tuple(result)


def _attachment_pointer(
    attachment: AttachmentOccurrence,
    messages: Mapping[str, Message],
) -> str:
    metadata_pointer = attachment.metadata_json.get("json_pointer")
    if isinstance(metadata_pointer, str):
        return metadata_pointer
    if attachment.kind == "thread_image":
        return "/image"
    if attachment.message_id is None:
        raise RawSourceRewriteError(
            f"Attachment {attachment.id} has no raw JSON pointer provenance"
        )
    message = messages.get(attachment.message_id)
    if message is None:
        raise RawSourceRewriteError(f"Attachment {attachment.id} has no parent message")
    field = {
        "photo": "photos",
        "video": "videos",
        "audio": "audio_files",
        "file": "files",
        "gif": "gifs",
        "support_file": "attachments",
    }.get(attachment.kind)
    if attachment.kind == "sticker":
        return f"{message.json_pointer}/sticker"
    if field is None:
        raise RawSourceRewriteError(
            f"Attachment {attachment.id} has an unsupported raw JSON kind: {attachment.kind}"
        )
    return f"{message.json_pointer}/{field}/{attachment.ordinal}"


def _fingerprint(
    source_record: SourceRecord,
    raw_value: Any,
    activity: ActivityRecord | None,
) -> str | None:
    if source_record.record_type == "message":
        return canonical_fingerprint("facebook-message", raw_value)
    if source_record.record_type == "support_message":
        return canonical_fingerprint("support-message", raw_value)
    if source_record.record_type.startswith("activity:") and activity is not None:
        return canonical_fingerprint(
            source_record.record_type,
            {
                "category": activity.category,
                "subtype": activity.subtype,
                "payload": raw_value,
            },
        )
    return None


def _read_verified_json(layout: ArchiveLayout, source_file: SourceFile) -> Any:
    digest = source_file.sha256
    if digest is None:
        raise RawSourceRewriteError(f"Source file {source_file.id} has no inventory digest")
    blob_path = _safe_workspace_path(
        layout, layout.blob_path(digest), label="catalogued source blob"
    )
    try:
        if blob_path.is_symlink() or not blob_path.is_file():
            raise RawSourceRewriteError(f"Source blob is missing for source file {source_file.id}")
        if blob_path.stat().st_size != source_file.size_bytes or sha256_file(blob_path) != digest:
            raise RawSourceRewriteError(f"Source blob failed digest validation: {source_file.id}")
        return orjson.loads(blob_path.read_bytes())
    except orjson.JSONDecodeError as exc:
        raise RawSourceRewriteError(
            f"Source JSON is invalid for source file {source_file.id}"
        ) from exc


def build_raw_rewrite_plan(
    layout: ArchiveLayout,
    session: Session,
    selected: Mapping[Any, set[str]],
    *,
    binary_blob_ids: set[str] | None = None,
    binary_source_file_ids: set[int] | None = None,
) -> RawRewritePlan:
    """Build and validate tombstone rewrites without mutating the catalog or workspace."""

    pointers_by_file: dict[int, set[str]] = {}
    binary_file_ids: set[int] = set()
    selected_source_records: dict[int, set[str]] = {}
    selected_messages = (
        list(session.scalars(select(Message).where(Message.id.in_(selected["message"]))))
        if selected["message"]
        else []
    )
    messages_by_id = {row.id: row for row in selected_messages}
    # Attachment-only selection needs its retained parent message to derive the raw pointer.
    attachment_rows = (
        list(
            session.scalars(
                select(AttachmentOccurrence).where(
                    AttachmentOccurrence.id.in_(selected["attachment"])
                )
            )
        )
        if selected["attachment"]
        else []
    )
    parent_message_ids = {
        row.message_id
        for row in attachment_rows
        if row.message_id is not None and row.message_id not in messages_by_id
    }
    if parent_message_ids:
        messages_by_id.update(
            {
                row.id: row
                for row in session.scalars(
                    select(Message).where(Message.id.in_(parent_message_ids))
                )
            }
        )
    for message in selected_messages:
        pointers_by_file.setdefault(message.source_file_id, set()).add(message.json_pointer)
        selected_source_records.setdefault(message.source_file_id, set()).add(
            message.source_record_id
        )
    selected_activities = (
        list(
            session.scalars(
                select(ActivityRecord).where(ActivityRecord.id.in_(selected["activity"]))
            )
        )
        if selected["activity"]
        else []
    )
    for activity in selected_activities:
        pointers_by_file.setdefault(activity.source_file_id, set()).add(activity.json_pointer)
        selected_source_records.setdefault(activity.source_file_id, set()).add(
            activity.source_record_id
        )
    for attachment in attachment_rows:
        if attachment.source_file_id is None:
            continue
        pointers_by_file.setdefault(attachment.source_file_id, set()).add(
            _attachment_pointer(attachment, messages_by_id)
        )

    if selected["conversation"]:
        segments = session.scalars(
            select(ConversationSegment).where(
                ConversationSegment.conversation_id.in_(selected["conversation"])
            )
        )
        for segment in segments:
            containers = segment.metadata_json.get("containers")
            if not isinstance(containers, dict) or not containers:
                raise RawSourceRewriteError(
                    f"Conversation segment has no source-container provenance: {segment.id}"
                )
            pointer = ""
            if segment.source_category == "support":
                prefix = "support/"
                if not segment.thread_path.startswith(prefix):
                    raise RawSourceRewriteError(
                        f"Support segment has an invalid thread path: {segment.id}"
                    )
                opaque_key = segment.thread_path[len(prefix) :]
                pointer = f"/support_messages/{_escape_pointer_token(opaque_key)}"
            for source_file_key in containers:
                try:
                    source_file_id = int(source_file_key)
                except (TypeError, ValueError) as exc:
                    raise RawSourceRewriteError(
                        f"Conversation segment has invalid source provenance: {segment.id}"
                    ) from exc
                pointers_by_file.setdefault(source_file_id, set()).add(pointer)

    if binary_blob_ids:
        binary_statement = select(SourceFile).where(SourceFile.sha256.in_(binary_blob_ids))
        if binary_source_file_ids is not None:
            binary_statement = binary_statement.where(SourceFile.id.in_(binary_source_file_ids))
        for binary_source in session.scalars(binary_statement):
            pointers_by_file.setdefault(binary_source.id, set())
            binary_file_ids.add(binary_source.id)

    if not pointers_by_file:
        return RawRewritePlan((), (), (), ())

    originals, relocations, original_by_source = _load_complete_manifests(layout)
    source_files = {
        row.id: row
        for row in session.scalars(select(SourceFile).where(SourceFile.id.in_(pointers_by_file)))
    }
    chunks = {
        row.id: row
        for row in session.scalars(
            select(SourceChunk).where(
                SourceChunk.id.in_({row.chunk_id for row in source_files.values()})
            )
        )
    }
    all_records: dict[int, list[SourceRecord]] = {}
    for record in session.scalars(
        select(SourceRecord).where(SourceRecord.source_file_id.in_(pointers_by_file))
    ):
        all_records.setdefault(record.source_file_id, []).append(record)
    activities_by_source_record = {
        row.source_record_id: row
        for row in session.scalars(
            select(ActivityRecord).where(ActivityRecord.source_file_id.in_(pointers_by_file))
        )
    }
    messages_by_source_record = {
        row.source_record_id: row
        for row in session.scalars(
            select(Message).where(Message.source_file_id.in_(pointers_by_file))
        )
    }

    rewrites: list[RawSourceRewrite] = []
    for source_file_id, raw_pointers in sorted(pointers_by_file.items()):
        source_file = source_files.get(source_file_id)
        if source_file is None:
            raise RawSourceRewriteError(f"Raw source file is missing: {source_file_id}")
        chunk = chunks.get(source_file.chunk_id)
        if chunk is None:
            raise RawSourceRewriteError(f"Raw source chunk is missing: {source_file_id}")
        source = f"{chunk.original_path}/{source_file.relative_path}"
        manifest_row = original_by_source.get(source)
        if manifest_row is None:
            raise RawSourceRewriteError(f"Raw source is absent from complete manifests: {source}")
        if any(
            manifest_row.get(key) != expected
            for key, expected in (
                ("sha256", source_file.sha256),
                ("size_bytes", source_file.size_bytes),
                ("mode", source_file.mode),
                ("mtime_ns", source_file.mtime_ns),
            )
        ):
            raise RawSourceRewriteError(f"Catalog and source manifests disagree at {source}")
        selected_record_ids = selected_source_records.get(source_file_id, set())
        fingerprint_updates: list[FingerprintUpdate] = []
        if source_file_id in binary_file_ids:
            old_blob = _safe_workspace_path(
                layout,
                layout.blob_path(str(source_file.sha256)),
                label="catalogued binary source blob",
            )
            if (
                old_blob.is_symlink()
                or not old_blob.is_file()
                or old_blob.stat().st_size != source_file.size_bytes
                or sha256_file(old_blob) != source_file.sha256
            ):
                raise RawSourceRewriteError(
                    f"Binary source blob failed digest validation: {source_file.id}"
                )
            if raw_pointers or all_records.get(source_file_id):
                raise RawSourceRewriteError(
                    f"A binary purge target also contains normalized JSON records: {source}"
                )
            pointers: tuple[str, ...] = ()
            new_bytes = BINARY_TOMBSTONE_BYTES
        else:
            payload = _read_verified_json(layout, source_file)
            # orjson returns a fresh mutable JSON graph, so the original semantic hashes can be
            # captured before applying replacements in-place.
            original_payload = orjson.loads(orjson.dumps(payload))
            pointers = _minimal_pointers(raw_pointers)
            for pointer in pointers:
                _resolve_pointer(payload, pointer)
                payload = _replace_pointer(payload, pointer)

            for record in all_records.get(source_file_id, []):
                old_value = _resolve_pointer(original_payload, record.json_pointer)
                old_fingerprint = _fingerprint(
                    record, old_value, activities_by_source_record.get(record.id)
                )
                if old_fingerprint is not None and old_fingerprint != record.logical_fingerprint:
                    raise RawSourceRewriteError(
                        f"Stored source fingerprint does not match raw JSON: {record.id}"
                    )
                if record.id in selected_record_ids:
                    continue
                new_value = _resolve_pointer(payload, record.json_pointer)
                if is_purge_tombstone(new_value):
                    raise RawSourceRewriteError(
                        f"A retained source record would be tombstoned: {record.id}"
                    )
                if _canonical_value_sha256(old_value) == _canonical_value_sha256(new_value):
                    continue
                new_fingerprint = _fingerprint(
                    record, new_value, activities_by_source_record.get(record.id)
                )
                if old_fingerprint is None or new_fingerprint is None:
                    raise RawSourceRewriteError(
                        "A retained source record changed without a fingerprint adapter: "
                        f"{record.id}"
                    )
                linked_message = messages_by_source_record.get(record.id)
                linked_activity = activities_by_source_record.get(record.id)
                fingerprint_updates.append(
                    FingerprintUpdate(
                        source_record_id=record.id,
                        message_id=linked_message.id if linked_message is not None else None,
                        activity_id=linked_activity.id if linked_activity is not None else None,
                        old_fingerprint=old_fingerprint,
                        new_fingerprint=new_fingerprint,
                        new_activity_payload=(new_value if linked_activity is not None else None),
                    )
                )
            new_bytes = orjson.dumps(
                payload,
                option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE | orjson.OPT_SORT_KEYS,
            )
        new_sha256 = hashlib.sha256(new_bytes).hexdigest()
        physical = _safe_workspace_path(
            layout,
            layout.root / chunk.original_path / source_file.relative_path,
            label="physical raw source path",
        )
        if physical.exists() or physical.is_symlink():
            try:
                physical_stat = physical.lstat()
            except OSError as exc:
                raise RawSourceRewriteError(f"Cannot inspect raw source path: {source}") from exc
            if not stat.S_ISREG(physical_stat.st_mode):
                raise RawSourceRewriteError(f"Raw source path is not a regular file: {source}")
            _validate_regular_file_identity(
                physical,
                expected_size=source_file.size_bytes,
                expected_sha256=str(source_file.sha256),
                label=f"Physical raw source {source}",
            )
            physical_path: Path | None = physical
        else:
            physical_path = None
        rewrites.append(
            RawSourceRewrite(
                source_file_id=source_file.id,
                source=source,
                chunk=chunk.original_path,
                relative_path=source_file.relative_path,
                physical_path=physical_path,
                mode=source_file.mode,
                mtime_ns=source_file.mtime_ns,
                media_type=source_file.media_type,
                old_sha256=str(source_file.sha256),
                old_size_bytes=source_file.size_bytes,
                new_sha256=new_sha256,
                new_size_bytes=len(new_bytes),
                new_bytes=new_bytes,
                binary=source_file_id in binary_file_ids,
                pointers=pointers,
                selected_source_record_ids=tuple(sorted(selected_record_ids)),
                fingerprint_updates=tuple(
                    sorted(fingerprint_updates, key=lambda item: item.source_record_id)
                ),
            )
        )

    rewritten_ids = {rewrite.source_file_id for rewrite in rewrites}
    obsolete: list[str] = []
    for old_sha256 in sorted({rewrite.old_sha256 for rewrite in rewrites}):
        other_source_ref = session.scalar(
            select(SourceFile.id)
            .where(SourceFile.sha256 == old_sha256, SourceFile.id.not_in(rewritten_ids))
            .limit(1)
        )
        retained_attachment = session.scalar(
            select(AttachmentOccurrence.id)
            .where(
                AttachmentOccurrence.blob_sha256 == old_sha256,
                AttachmentOccurrence.id.not_in(selected["attachment"]),
            )
            .limit(1)
        )
        if other_source_ref is None and retained_attachment is None:
            obsolete.append(old_sha256)

    replacements = {rewrite.source: rewrite for rewrite in rewrites}
    new_originals: list[dict[str, Any]] = []
    for original in originals:
        source = f"{original['chunk']}/{original['relative_path']}"
        rewrite = replacements.get(source)
        row = dict(original)
        if rewrite is not None:
            row.update(sha256=rewrite.new_sha256, size_bytes=rewrite.new_size_bytes)
        new_originals.append(row)
    new_relocations: list[dict[str, Any]] = []
    for relocation in relocations:
        rewrite = replacements.get(str(relocation["source"]))
        row = dict(relocation)
        if rewrite is not None:
            row.update(
                sha256=rewrite.new_sha256,
                size_bytes=rewrite.new_size_bytes,
                blob=layout.blob_path(rewrite.new_sha256).relative_to(layout.root).as_posix(),
                operation="hard-purge-tombstone-v1",
            )
        new_relocations.append(row)
    return RawRewritePlan(
        rewrites=tuple(rewrites),
        original_manifest_rows=tuple(new_originals),
        relocation_manifest_rows=tuple(new_relocations),
        obsolete_blob_ids=tuple(obsolete),
    )


def apply_raw_rewrite_catalog(
    session: Session, layout: ArchiveLayout, plan: RawRewritePlan
) -> None:
    for rewrite in plan.rewrites:
        session.execute(
            sqlite_insert(Blob)
            .values(
                sha256=rewrite.new_sha256,
                size_bytes=rewrite.new_size_bytes,
                media_type=(
                    "application/x-meta-archive-purged"
                    if rewrite.binary
                    else rewrite.media_type or "application/json"
                ),
                extension=None if rewrite.binary else ".json",
                relative_path=layout.blob_path(rewrite.new_sha256)
                .relative_to(layout.root)
                .as_posix(),
                state="materialized",
            )
            .on_conflict_do_update(
                index_elements=[Blob.sha256],
                set_={
                    "size_bytes": rewrite.new_size_bytes,
                    "relative_path": layout.blob_path(rewrite.new_sha256)
                    .relative_to(layout.root)
                    .as_posix(),
                    "state": "materialized",
                },
            )
        )
        source_result = session.execute(
            update(SourceFile)
            .where(
                SourceFile.id == rewrite.source_file_id,
                SourceFile.sha256 == rewrite.old_sha256,
                SourceFile.size_bytes == rewrite.old_size_bytes,
            )
            .values(
                sha256=rewrite.new_sha256,
                size_bytes=rewrite.new_size_bytes,
                imported_sha256=rewrite.new_sha256,
                media_type=(
                    "application/x-meta-archive-purged" if rewrite.binary else rewrite.media_type
                ),
                json_status="purged" if rewrite.binary else "parsed",
                scan_status="discovered",
            )
        )
        if getattr(source_result, "rowcount", -1) != 1:
            raise RawSourceRewriteError(
                f"Raw source catalog row changed during purge: {rewrite.source_file_id}"
            )
        for fingerprint in rewrite.fingerprint_updates:
            fingerprint_result = session.execute(
                update(SourceRecord)
                .where(
                    SourceRecord.id == fingerprint.source_record_id,
                    SourceRecord.logical_fingerprint == fingerprint.old_fingerprint,
                )
                .values(logical_fingerprint=fingerprint.new_fingerprint, duplicate_of_id=None)
            )
            if getattr(fingerprint_result, "rowcount", -1) != 1:
                raise RawSourceRewriteError(
                    "Retained source fingerprint changed during purge: "
                    f"{fingerprint.source_record_id}"
                )
            session.execute(
                update(SourceRecord)
                .where(SourceRecord.duplicate_of_id == fingerprint.source_record_id)
                .values(duplicate_of_id=None)
            )
            if fingerprint.message_id is not None:
                message_result = session.execute(
                    update(Message)
                    .where(
                        Message.id == fingerprint.message_id,
                        Message.logical_fingerprint == fingerprint.old_fingerprint,
                    )
                    .values(
                        logical_fingerprint=fingerprint.new_fingerprint,
                        duplicate_of_id=None,
                    )
                )
                if getattr(message_result, "rowcount", -1) != 1:
                    raise RawSourceRewriteError(
                        "Retained message fingerprint changed during purge: "
                        f"{fingerprint.message_id}"
                    )
                session.execute(
                    update(Message)
                    .where(Message.duplicate_of_id == fingerprint.message_id)
                    .values(duplicate_of_id=None)
                )
            if fingerprint.activity_id is not None:
                activity = session.scalar(
                    select(ActivityRecord).where(ActivityRecord.id == fingerprint.activity_id)
                )
                if activity is None:
                    raise RawSourceRewriteError(
                        f"Retained activity disappeared during purge: {fingerprint.activity_id}"
                    )
                extra_json = dict(activity.extra_json)
                extra_json["raw_payload"] = fingerprint.new_activity_payload
                activity.extra_json = extra_json


def _write_bytes_atomic(path: Path, payload: bytes, *, mode: int, mtime_ns: int | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode & 0o7777)
        if mtime_ns is not None:
            os.utime(temporary, ns=(mtime_ns, mtime_ns), follow_symlinks=False)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _jsonl_bytes(rows: tuple[dict[str, Any], ...]) -> bytes:
    return b"".join(
        orjson.dumps(row, option=orjson.OPT_SORT_KEYS | orjson.OPT_APPEND_NEWLINE) for row in rows
    )


def _publication_validation_path(
    layout: ArchiveLayout,
    rewrite: RawSourceRewrite,
) -> Path | None:
    """Resolve the old source either live or in the caller's durable rollback staging tree."""

    if rewrite.physical_path is None:
        return None
    physical = _safe_workspace_path(
        layout,
        rewrite.physical_path,
        label="physical raw source path",
    )
    if physical.exists() or physical.is_symlink():
        return physical

    try:
        relative = physical.relative_to(layout.root)
    except ValueError as exc:
        raise RawSourceRewriteError("Physical raw source escapes the archive") from exc
    staging_root = layout.trash / "purge-staging"
    if not staging_root.exists() and not staging_root.is_symlink():
        raise RawSourceRewriteError(f"Physical raw source {rewrite.source} is missing")
    try:
        staging_stat = staging_root.lstat()
    except OSError as exc:
        raise RawSourceRewriteError("Purge staging root cannot be inspected safely") from exc
    if not stat.S_ISDIR(staging_stat.st_mode):
        raise RawSourceRewriteError("Purge staging root is not a safe directory")

    candidates: list[Path] = []
    try:
        operation_roots = list(staging_root.iterdir())
    except OSError as exc:
        raise RawSourceRewriteError("Purge staging root cannot be read safely") from exc
    for operation_root in operation_roots:
        candidate = operation_root / relative
        if not candidate.exists() and not candidate.is_symlink():
            continue
        candidates.append(
            _safe_workspace_path(
                layout,
                candidate,
                label="staged physical raw source path",
            )
        )
    if not candidates:
        raise RawSourceRewriteError(f"Physical raw source {rewrite.source} is missing")
    if len(candidates) != 1:
        raise RawSourceRewriteError(
            f"Physical raw source {rewrite.source} has ambiguous rollback staging"
        )
    return candidates[0]


def _validate_publication_source(
    layout: ArchiveLayout,
    rewrite: RawSourceRewrite,
) -> None:
    validation_path = _publication_validation_path(layout, rewrite)
    if validation_path is None:
        return
    _validate_regular_file_identity(
        validation_path,
        expected_size=rewrite.old_size_bytes,
        expected_sha256=rewrite.old_sha256,
        label=f"Physical raw source {rewrite.source}",
    )


def publish_raw_rewrite_files(layout: ArchiveLayout, plan: RawRewritePlan) -> set[Path]:
    """Publish prevalidated bytes after callers have staged every overwritten live path."""

    if not plan.rewrites:
        return set()
    # Validate every source before any publication, then validate each source again immediately
    # before replacing its live pathname. The second pass closes the plan-to-publish window; the
    # caller's source-publisher lock excludes cooperative writers during this loop.
    for rewrite in plan.rewrites:
        _validate_publication_source(layout, rewrite)
    created: set[Path] = set()
    try:
        for rewrite in plan.rewrites:
            blob_path = _safe_workspace_path(
                layout,
                layout.blob_path(rewrite.new_sha256),
                label="new source blob path",
            )
            if blob_path.exists() or blob_path.is_symlink():
                if blob_path.is_symlink() or not blob_path.is_file():
                    raise RawSourceRewriteError("New source blob path is unsafe")
                if (
                    blob_path.stat().st_size != rewrite.new_size_bytes
                    or sha256_file(blob_path) != rewrite.new_sha256
                ):
                    raise RawSourceRewriteError("Existing new source blob has incorrect bytes")
            else:
                _write_bytes_atomic(blob_path, rewrite.new_bytes, mode=0o444, mtime_ns=None)
                created.add(blob_path)
            if rewrite.physical_path is not None:
                _validate_publication_source(layout, rewrite)
                _write_bytes_atomic(
                    rewrite.physical_path,
                    rewrite.new_bytes,
                    mode=rewrite.mode,
                    mtime_ns=rewrite.mtime_ns,
                )
        _write_bytes_atomic(
            _safe_workspace_path(
                layout, layout.original_files_manifest, label="original-files manifest"
            ),
            _jsonl_bytes(plan.original_manifest_rows),
            mode=PRIVATE_FILE_MODE,
            mtime_ns=None,
        )
        _write_bytes_atomic(
            _safe_workspace_path(layout, layout.relocation_manifest, label="relocation manifest"),
            _jsonl_bytes(plan.relocation_manifest_rows),
            mode=PRIVATE_FILE_MODE,
            mtime_ns=None,
        )
        return created
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise


def raw_rewrite_live_targets(layout: ArchiveLayout, plan: RawRewritePlan) -> set[Path]:
    if not plan.rewrites:
        return set()
    targets: set[Path] = {layout.original_files_manifest, layout.relocation_manifest}
    targets.update(
        rewrite.physical_path for rewrite in plan.rewrites if rewrite.physical_path is not None
    )
    targets.update(layout.blob_path(digest) for digest in plan.obsolete_blob_ids)
    return {path for path in targets if path.exists() or path.is_symlink()}
