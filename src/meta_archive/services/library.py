from __future__ import annotations

import errno
import fcntl
import json
import os
import re
import shutil
import stat
import tempfile
import uuid
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, TextIO

from sqlalchemy import Engine, delete, func, select
from sqlalchemy.orm import Session

from meta_archive.models import (
    ActivityRecord,
    AttachmentCandidate,
    AttachmentOccurrence,
    Blob,
    Conversation,
    ConversationSegment,
    CoverageArea,
    DeletionOperation,
    Message,
    MessageReaction,
    Participant,
    SegmentParticipant,
    SourceChunk,
    SourceFile,
)
from meta_archive.paths import (
    ArchiveLayout,
    create_directory_beneath,
    open_directory_beneath,
    remove_directory_tree_beneath,
    replace_entry_beneath,
    validate_directory_tree_beneath,
)
from meta_archive.services.deletion_lock_lease import DeletionLockLease
from meta_archive.services.hashing import sha256_file
from meta_archive.services.identifiers import stable_id
from meta_archive.services.message_importer import redact_sensitive_message_metadata
from meta_archive.services.source_import_lock import (
    SourceImportPublicationLease,
    assert_source_import_publication_allowed,
    source_import_publication_lock,
)
from meta_archive.services.text import safe_filesystem_label

ProgressCallback = Callable[[str, int, int | None], None]

LIBRARY_FORMAT_VERSION = 1
UNLINKED_GENERATOR = "filesystem-library"
UNLINKED_GENERATOR_VERSION = 1
_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_EXTENSION_PATTERN = re.compile(r"\.[a-z0-9]{1,15}\Z")
_UNLINKED_DOCUMENT_EXTENSIONS = frozenset(
    {
        ".7z",
        ".doc",
        ".docx",
        ".pdf",
        ".ppt",
        ".pptx",
        ".rar",
        ".tar",
        ".xls",
        ".xlsx",
        ".zip",
    }
)


class LibraryBuildError(RuntimeError):
    """The catalog cannot be represented as a safe filesystem library."""


class LibraryBuildBusyError(LibraryBuildError):
    """Another process currently owns the filesystem-library publication lock."""


@dataclass(slots=True)
class LibraryBuildSummary:
    conversation_count: int = 0
    message_count: int = 0
    transcript_file_count: int = 0
    attachment_occurrence_count: int = 0
    conversation_attachment_link_count: int = 0
    media_view_link_count: int = 0
    unlinked_media_count: int = 0
    activity_record_count: int = 0
    hardlink_count: int = 0
    copy_count: int = 0


def _ensure_library_lock_directory(layout: ArchiveLayout) -> None:
    try:
        layout.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        root_stat = layout.root.lstat()
        if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
            raise LibraryBuildError(f"Archive root is not a real directory: {layout.root}")
        os.chmod(layout.root, 0o700, follow_symlinks=False)
        create_directory_beneath(layout.catalog, layout.root, mode=0o700)
    except OSError as exc:
        raise LibraryBuildError("Could not prepare the filesystem-library lock directory") from exc


@contextmanager
def library_build_lock(
    layout: ArchiveLayout,
    *,
    blocking: bool = True,
) -> Iterator[None]:
    """Serialize publication and recover validated remnants before yielding to the caller.

    Destructive workflows acquire the source/import lock immediately after their deletion lock,
    then this library lock before worker locks or a SQLite writer transaction. They call
    :func:`build_filesystem_library` with both ``source_lock`` and ``lock_held=True``. Passing
    ``blocking=False`` makes those workflows fail fast instead of deadlocking against an active
    builder. Normal standalone builds block and serialize by default.
    """

    _ensure_library_lock_directory(layout)
    catalog_descriptor = open_directory_beneath(layout.catalog, layout.root)
    descriptor: int | None = None
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | int(getattr(os, "O_CLOEXEC", 0))
        | int(getattr(os, "O_NOFOLLOW", 0))
        | int(getattr(os, "O_NONBLOCK", 0))
    )
    try:
        descriptor = os.open(
            "library-build.lock",
            flags,
            0o600,
            dir_fd=catalog_descriptor,
        )
        lock_stat = os.fstat(descriptor)
        if not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_nlink != 1:
            os.close(descriptor)
            descriptor = None
            raise LibraryBuildError(
                "The filesystem-library lock is not a private single-link regular file"
            )
        os.fchmod(descriptor, 0o600)
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise LibraryBuildError("Could not securely open the filesystem-library lock") from exc
    finally:
        os.close(catalog_descriptor)

    assert descriptor is not None
    with os.fdopen(descriptor, "r+", encoding="utf-8") as handle:
        operation = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(handle.fileno(), operation)
        except BlockingIOError as exc:
            raise LibraryBuildBusyError(
                "Another filesystem-library build or publication is active"
            ) from exc
        try:
            handle.seek(0)
            handle.truncate()
            json.dump(
                {
                    "operation": "filesystem-library-publication",
                    "pid": os.getpid(),
                    "started_at": datetime.now(UTC).isoformat(),
                },
                handle,
                sort_keys=True,
            )
            handle.flush()
            os.fsync(handle.fileno())
            _recover_library_build_state_locked(layout)
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True, slots=True)
class _SourceCoordinate:
    id: int
    chunk_key: str
    chunk_path: str
    relative_path: str
    normalized_path: str
    media_type: str | None
    extension: str | None
    sha256: str | None
    size_bytes: int


@dataclass(frozen=True, slots=True)
class _BlobMetadata:
    sha256: str
    size_bytes: int
    media_type: str | None
    extension: str | None


def _json_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_ndjson(path: Path, records: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(_json_value(record))
            handle.write("\n")
            count += 1
    return count


def _source_manifest_metadata(layout: ArchiveLayout) -> dict[str, dict[str, Any]]:
    manifests = {
        "original_files": layout.original_files_manifest,
        "original_directories": layout.original_directories_manifest,
        "relocation": layout.relocation_manifest,
    }
    result: dict[str, dict[str, Any]] = {}
    for key, path in manifests.items():
        if path.is_symlink() or not path.is_file():
            raise LibraryBuildError(
                f"Canonical source manifest is missing or unsafe: {path}. "
                "Complete the hashed scan and blob-store consolidation first."
            )
        with path.open("rb") as handle:
            line_count = sum(1 for _line in handle)
        result[key] = {
            "path": path.relative_to(layout.root).as_posix(),
            "sha256": sha256_file(path),
            "line_count": line_count,
            "size_bytes": path.stat(follow_symlinks=False).st_size,
        }
    return result


def _utc_iso(timestamp_ms: int | None) -> str | None:
    if timestamp_ms is None:
        return None
    try:
        return datetime.fromtimestamp(timestamp_ms / 1000, UTC).isoformat().replace("+00:00", "Z")
    except (OSError, OverflowError, ValueError):
        return None


def _month_bucket(timestamp_ms: int | None) -> str:
    value = _utc_iso(timestamp_ms)
    return value[:7] if value is not None else "unknown-date"


def _date_parts(timestamp_ms: int | None) -> tuple[str, ...]:
    value = _utc_iso(timestamp_ms)
    if value is None:
        return ("unknown-date",)
    return (value[:4], value[5:7])


def _conversation_bucket(conversation_type: str) -> str:
    if conversation_type == "direct":
        return "direct"
    if conversation_type == "group":
        return "groups"
    if conversation_type == "system":
        return "system"
    return "unknown"


def _conversation_directory_name(conversation: Conversation) -> str:
    title = safe_filesystem_label(conversation.title_display, fallback="untitled", limit=72)
    return f"{title}--{conversation.id}"


def conversation_library_relative_root(conversation: Conversation) -> Path:
    """Return the deterministic readable-library root for one catalog conversation."""

    return (
        Path("conversations")
        / _conversation_bucket(conversation.conversation_type)
        / _conversation_directory_name(conversation)
    )


def _safe_extension(*values: str | None) -> str:
    for value in values:
        if not value:
            continue
        candidate = value.casefold()
        if not candidate.startswith("."):
            candidate = f".{candidate}"
        if _EXTENSION_PATTERN.fullmatch(candidate):
            return candidate
    return ""


def _filename_hint(
    attachment: AttachmentOccurrence, source: _SourceCoordinate | SourceFile | None
) -> str:
    if attachment.filename:
        return PurePosixPath(attachment.filename.replace("\\", "/")).name
    if source is not None:
        return PurePosixPath(source.relative_path).name
    if attachment.source_uri:
        return PurePosixPath(attachment.source_uri.split("?", 1)[0].replace("\\", "/")).name
    return attachment.kind


def _attachment_filename(
    attachment: AttachmentOccurrence,
    source: _SourceCoordinate | SourceFile | None,
    blob: _BlobMetadata | Blob | None,
) -> str:
    hint = _filename_hint(attachment, source)
    return _attachment_filename_from_values(
        attachment,
        hint=hint,
        source_extension=source.extension if source else None,
        blob_extension=blob.extension if blob else None,
    )


def _attachment_filename_from_values(
    attachment: AttachmentOccurrence,
    *,
    hint: str,
    source_extension: str | None,
    blob_extension: str | None,
) -> str:
    hint_path = PurePosixPath(hint)
    stem = safe_filesystem_label(hint_path.stem, fallback=attachment.kind, limit=72)
    extension = _safe_extension(
        source_extension,
        blob_extension,
        hint_path.suffix,
    )
    return f"{stem}--{attachment.id}{extension}"


def _attachment_bucket(attachment: AttachmentOccurrence) -> str:
    if attachment.availability == "unlinked":
        return "unlinked"
    media_type = (attachment.media_type or "").casefold()
    kind = attachment.kind.casefold()
    if media_type.startswith("video/") or "video" in kind:
        return "videos"
    if media_type.startswith("audio/") or kind in {"audio", "audio_file", "voice_message"}:
        return "audio"
    if media_type.startswith("image/") or kind in {
        "gif",
        "photo",
        "sticker",
        "thread_image",
    }:
        return "photos"
    return "files"


def _unlinked_kind(media_type: str | None) -> str:
    value = (media_type or "").casefold()
    if value.startswith("image/"):
        return "photo"
    if value.startswith("video/"):
        return "video"
    if value.startswith("audio/"):
        return "audio"
    return "file"


def _is_browsable_unlinked_source(source: SourceFile) -> bool:
    media_type = (source.media_type or "").casefold()
    return (
        media_type.startswith(("image/", "video/", "audio/"))
        or (source.extension or "").casefold() in _UNLINKED_DOCUMENT_EXTENSIONS
    )


def _is_generated_unlinked(attachment: AttachmentOccurrence) -> bool:
    return (
        attachment.availability == "unlinked"
        and attachment.metadata_json.get("generated_by") == UNLINKED_GENERATOR
    )


def _source_coordinates(session: Session) -> dict[int, _SourceCoordinate]:
    rows = session.execute(
        select(SourceFile, SourceChunk.chunk_key, SourceChunk.original_path)
        .join(SourceChunk, SourceChunk.id == SourceFile.chunk_id)
        .where(SourceFile.scan_status == "discovered")
        .order_by(SourceChunk.ordinal, SourceFile.relative_path, SourceFile.id)
    )
    return {
        source.id: _SourceCoordinate(
            id=source.id,
            chunk_key=chunk_key,
            chunk_path=chunk_path,
            relative_path=source.relative_path,
            normalized_path=source.normalized_path,
            media_type=source.media_type,
            extension=source.extension,
            sha256=source.sha256,
            size_bytes=source.size_bytes,
        )
        for source, chunk_key, chunk_path in rows
    }


def _source_record(source: _SourceCoordinate | None) -> dict[str, Any] | None:
    if source is None:
        return None
    return {
        "source_file_id": source.id,
        "chunk_key": source.chunk_key,
        "chunk_path": source.chunk_path,
        "relative_path": source.relative_path,
        "normalized_path": source.normalized_path,
        "sha256": source.sha256,
    }


def _reconcile_unlinked_media_session(session: Session) -> int:
    # A savepoint keeps this helper composable inside the hard-purge transaction while the public
    # wrapper continues to own its normal top-level transaction.
    with session.begin_nested():
        all_occurrences = list(session.scalars(select(AttachmentOccurrence)))
        generated = {row.id: row for row in all_occurrences if _is_generated_unlinked(row)}
        referenced_source_ids = {
            row.resolved_source_file_id
            for row in all_occurrences
            if not _is_generated_unlinked(row) and row.resolved_source_file_id is not None
        }
        referenced_source_ids.update(session.scalars(select(AttachmentCandidate.source_file_id)))

        source_rows = session.execute(
            select(SourceFile, SourceChunk.chunk_key, SourceChunk.original_path)
            .join(SourceChunk, SourceChunk.id == SourceFile.chunk_id)
            .where(SourceFile.scan_status == "discovered")
            .where(SourceFile.json_status == "not_json")
            .order_by(SourceChunk.ordinal, SourceFile.relative_path, SourceFile.id)
        ).all()
        unhashed = [
            source.id
            for source, _chunk_key, _chunk_path in source_rows
            if _is_browsable_unlinked_source(source) and not source.sha256
        ]
        if unhashed:
            raise LibraryBuildError(
                "Cannot build the library before every non-JSON source file is hashed "
                f"({len(unhashed)} catalog rows are unhashed)."
            )

        grouped_sources: defaultdict[str, list[tuple[SourceFile, str, str]]] = defaultdict(list)
        for source, chunk_key, chunk_path in source_rows:
            if _is_browsable_unlinked_source(source):
                grouped_sources[source.normalized_path].append((source, chunk_key, chunk_path))

        target_ids: set[str] = set()
        for normalized_path, copies in grouped_sources.items():
            if any(source.id in referenced_source_ids for source, _key, _path in copies):
                continue
            digests = {source.sha256 for source, _key, _path in copies}
            if None in digests or len(digests) != 1:
                raise LibraryBuildError(
                    "Unlinked normalized-path copies have conflicting or missing bytes: "
                    f"{normalized_path}"
                )
            source, chunk_key, chunk_path = copies[0]
            assert source.sha256 is not None
            occurrence_id = stable_id("unlinked-media", normalized_path)
            target_ids.add(occurrence_id)
            metadata = {
                "generated_by": UNLINKED_GENERATOR,
                "generator_version": UNLINKED_GENERATOR_VERSION,
                "origin": "unreferenced_catalogued_source_file",
                "date_status": "unknown",
                "source_copy_count": len(copies),
                "source_copies": [
                    {
                        "source_file_id": copy.id,
                        "chunk_key": copy_chunk_key,
                        "chunk_path": copy_chunk_path,
                        "relative_path": copy.relative_path,
                    }
                    for copy, copy_chunk_key, copy_chunk_path in copies
                ],
                "provenance": {
                    "source_file_id": source.id,
                    "chunk_key": chunk_key,
                    "chunk_path": chunk_path,
                    "relative_path": source.relative_path,
                    "normalized_path": source.normalized_path,
                },
            }
            row = session.get(AttachmentOccurrence, occurrence_id)
            if row is not None and not _is_generated_unlinked(row):
                raise LibraryBuildError(
                    f"Generated unlinked-media ID collides with an importer occurrence: {row.id}"
                )
            if row is not None and row.deleted_at is not None:
                # A logical deletion owns the exact quarantined row image until restore or purge.
                # Library reconciliation must neither resurrect nor rewrite that sealed state.
                continue
            if row is None:
                row = AttachmentOccurrence(
                    id=occurrence_id,
                    message_id=None,
                    activity_record_id=None,
                    conversation_id=None,
                    source_file_id=source.id,
                    resolved_source_file_id=source.id,
                    blob_sha256=source.sha256,
                    kind=_unlinked_kind(source.media_type),
                    ordinal=0,
                    source_uri=source.relative_path,
                    filename=PurePosixPath(source.relative_path).name,
                    media_type=source.media_type,
                    size_bytes=source.size_bytes,
                    timestamp_ms=None,
                    availability="unlinked",
                    metadata_json=metadata,
                )
                session.add(row)
            else:
                row.message_id = None
                row.activity_record_id = None
                row.conversation_id = None
                row.source_file_id = source.id
                row.resolved_source_file_id = source.id
                row.blob_sha256 = source.sha256
                row.kind = _unlinked_kind(source.media_type)
                row.ordinal = 0
                row.source_uri = source.relative_path
                row.filename = PurePosixPath(source.relative_path).name
                row.media_type = source.media_type
                row.size_bytes = source.size_bytes
                row.timestamp_ms = None
                row.availability = "unlinked"
                row.metadata_json = metadata

        stale_ids = sorted(set(generated) - target_ids)
        if stale_ids:
            session.execute(
                delete(AttachmentOccurrence).where(
                    AttachmentOccurrence.id.in_(stale_ids),
                    AttachmentOccurrence.deleted_at.is_(None),
                )
            )
    return len(target_ids)


def reconcile_unlinked_media(layout: ArchiveLayout, engine: Engine) -> int:
    """Materialize first-class catalog rows for unreferenced non-JSON source files.

    Importer-owned occurrences and candidates define references. Generated rows never count as
    references themselves, so a second reconciliation is a true idempotency check. Rows for files
    that stop being unlinked are removed without touching importer-owned occurrences.
    """

    with source_import_publication_lock(layout) as source_lock:
        assert_source_import_publication_allowed(
            layout,
            engine,
            lease=source_lock,
            publisher="Unlinked-media catalog reconciliation",
        )
        with Session(engine) as session, session.begin():
            return _reconcile_unlinked_media_session(session)


class _BlobMaterializer:
    def __init__(self, layout: ArchiveLayout, blobs: Mapping[str, _BlobMetadata]) -> None:
        self.layout = layout
        self.blobs = blobs
        self.validated: set[str] = set()
        self.hardlinks = 0
        self.copies = 0

    def _validated_source(self, digest: str) -> tuple[Path, _BlobMetadata]:
        if not _DIGEST_PATTERN.fullmatch(digest):
            raise LibraryBuildError(f"Catalog contains an invalid SHA-256 value: {digest!r}")
        metadata = self.blobs.get(digest)
        if metadata is None:
            raise LibraryBuildError(f"Attachment references absent blob catalog row {digest}")
        source = self.layout.blob_path(digest)
        if source.is_symlink() or not source.is_file():
            raise LibraryBuildError(f"Catalogued blob file is missing or unsafe: {source}")
        if digest not in self.validated:
            stat = source.stat(follow_symlinks=False)
            if stat.st_size != metadata.size_bytes:
                raise LibraryBuildError(
                    f"Catalogued blob size mismatch for {digest}: "
                    f"expected {metadata.size_bytes}, got {stat.st_size}"
                )
            actual = sha256_file(source)
            if actual != digest:
                raise LibraryBuildError(
                    f"Catalogued blob digest mismatch: expected {digest}, got {actual}"
                )
            self.validated.add(digest)
        return source, metadata

    def link(self, digest: str, destination: Path) -> None:
        source, _metadata = self._validated_source(digest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            raise LibraryBuildError(f"Generated library path collides: {destination}")
        try:
            os.link(source, destination, follow_symlinks=False)
            self.hardlinks += 1
        except OSError as exc:
            fallback_errors = {errno.EXDEV, errno.EPERM, errno.EMLINK}
            if hasattr(errno, "EOPNOTSUPP"):
                fallback_errors.add(errno.EOPNOTSUPP)
            if exc.errno not in fallback_errors:
                raise
            shutil.copyfile(source, destination, follow_symlinks=False)
            if sha256_file(destination) != digest:
                destination.unlink(missing_ok=True)
                raise LibraryBuildError(
                    f"Copied library file failed digest verification: {destination}"
                ) from exc
            self.copies += 1


def _segment_record(
    segment: ConversationSegment,
    participants: Sequence[Participant],
) -> dict[str, Any]:
    return {
        "segment_id": segment.id,
        "sequence": segment.sequence,
        "thread_path": segment.thread_path,
        "title_raw": segment.title_raw,
        "title_display": segment.title_display,
        "source_category": segment.source_category,
        "source_chunk_count": segment.source_chunk_count,
        "is_still_participant": segment.is_still_participant,
        "first_activity_ms": segment.first_activity_ms,
        "first_activity_utc": _utc_iso(segment.first_activity_ms),
        "last_activity_ms": segment.last_activity_ms,
        "last_activity_utc": _utc_iso(segment.last_activity_ms),
        "completeness": segment.completeness,
        "participants": [
            {
                "participant_id": participant.id,
                "name_raw": participant.name_raw,
                "name_display": participant.name_display,
                "is_owner": participant.is_owner,
            }
            for participant in participants
        ],
        "source_metadata": segment.metadata_json,
    }


def _attachment_record(
    attachment: AttachmentOccurrence,
    sources: Mapping[int, _SourceCoordinate],
    *,
    library_path: str | None,
) -> dict[str, Any]:
    source = sources.get(attachment.source_file_id) if attachment.source_file_id else None
    resolved = (
        sources.get(attachment.resolved_source_file_id)
        if attachment.resolved_source_file_id
        else None
    )
    return {
        "record_type": "attachment_occurrence",
        "attachment_id": attachment.id,
        "message_id": attachment.message_id,
        "activity_record_id": attachment.activity_record_id,
        "conversation_id": attachment.conversation_id,
        "kind": attachment.kind,
        "ordinal": attachment.ordinal,
        "source_uri": attachment.source_uri,
        "filename": attachment.filename,
        "media_type": attachment.media_type,
        "size_bytes": attachment.size_bytes,
        "timestamp_ms": attachment.timestamp_ms,
        "timestamp_utc": _utc_iso(attachment.timestamp_ms),
        "date_status": "known" if _utc_iso(attachment.timestamp_ms) else "unknown",
        "availability": attachment.availability,
        "blob_sha256": attachment.blob_sha256,
        "source": _source_record(source),
        "resolved_source": _source_record(resolved),
        "library_path": library_path,
        "metadata": attachment.metadata_json,
    }


def _md_inline(value: str) -> str:
    return re.sub(r"([\\`*_{}\[\]()<>#+.!|~-])", r"\\\1", value).replace("\n", " ")


def _indented_block(value: str) -> str:
    lines = value.splitlines() or [""]
    return "\n".join(f"    {line}" for line in lines)


def _message_record(
    message: Message,
    source: _SourceCoordinate | None,
    reactions: Sequence[MessageReaction],
    attachments: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    safe_extra, legacy_omitted = redact_sensitive_message_metadata(message.extra_json)
    _unused_gated, gated_field_names = redact_sensitive_message_metadata(message.sensitive_json)
    omitted = set(legacy_omitted) | set(gated_field_names)
    if message.has_sensitive_ip:
        omitted.add("ip")
    if message.has_payment:
        omitted.add("payment_info")
    return {
        "record_type": "message",
        "message_id": message.id,
        "source_record_id": message.source_record_id,
        "conversation_id": message.conversation_id,
        "segment_id": message.segment_id,
        "source": _source_record(source),
        "json_pointer": message.json_pointer,
        "chunk_number": message.chunk_number,
        "source_array_index": message.source_array_index,
        "timestamp_ms": message.timestamp_ms,
        "timestamp_utc": _utc_iso(message.timestamp_ms),
        "sender_raw": message.sender_raw,
        "sender_display": message.sender_display,
        "sender_is_owner": message.sender_is_owner,
        "content_raw": message.content_raw,
        "content_display": message.content_display,
        "message_kind": message.message_kind,
        "call_duration_seconds": message.call_duration_seconds,
        "is_missed_call": message.is_missed_call,
        "has_payment": message.has_payment,
        "has_sensitive_ip": message.has_sensitive_ip,
        "is_unsent": message.is_unsent,
        "is_unavailable_body": message.is_unavailable_body,
        "transcription_raw": message.transcription_raw,
        "transcription_display": message.transcription_display,
        "logical_fingerprint": message.logical_fingerprint,
        "duplicate_of_id": message.duplicate_of_id,
        "extra": safe_extra,
        "sensitive_fields_omitted": sorted(omitted),
        "reactions": [
            {
                "reaction_id": reaction.id,
                "ordinal": reaction.ordinal,
                "actor_raw": reaction.actor_raw,
                "actor_display": reaction.actor_display,
                "reaction_raw": reaction.reaction_raw,
                "reaction_display": reaction.reaction_display,
                "timestamp_ms": reaction.timestamp_ms,
                "timestamp_utc": _utc_iso(reaction.timestamp_ms),
                "extra": reaction.extra_json,
            }
            for reaction in reactions
        ],
        "attachments": list(attachments),
    }


def _markdown_message(record: Mapping[str, Any]) -> str:
    message_id = str(record["message_id"])
    timestamp = str(record.get("timestamp_utc") or "Unknown date")
    sender = _md_inline(str(record["sender_display"]))
    result = [
        f'<a id="message-{message_id}"></a>',
        f"### {_md_inline(timestamp)} — {sender}",
        "",
        f"- Message ID: `{message_id}`",
        f"- Segment ID: `{record['segment_id']}`",
        f"- Source record ID: `{record['source_record_id']}`",
        f"- JSON pointer: `{record['json_pointer']}`",
        "",
    ]
    content = record.get("content_display")
    if not isinstance(content, str):
        if record.get("is_unsent"):
            content = "[Message was unsent]"
        elif record.get("is_unavailable_body"):
            content = "[Message body unavailable in this export]"
        elif record.get("message_kind") == "call":
            duration = record.get("call_duration_seconds")
            content = f"[Call{f' — {duration} seconds' if duration is not None else ''}]"
        else:
            content = "[No text body]"
    result.extend((_indented_block(content), ""))
    transcription = record.get("transcription_display")
    if isinstance(transcription, str):
        result.extend(("Transcription:", "", _indented_block(transcription), ""))
    attachments = record.get("attachments")
    if isinstance(attachments, list) and attachments:
        result.append("Attachments:")
        result.append("")
        for attachment in attachments:
            if not isinstance(attachment, dict):
                continue
            label = _md_inline(str(attachment.get("kind") or "file"))
            local_path = attachment.get("conversation_relative_path")
            if isinstance(local_path, str):
                result.append(f"- [{label}](../{local_path})")
            else:
                availability = _md_inline(str(attachment.get("availability") or "missing"))
                result.append(f"- {label} — {availability}")
        result.append("")
    if record.get("sensitive_fields_omitted"):
        result.extend(("Sensitive source fields are omitted from this readable transcript.", ""))
    return "\n".join(result)


@dataclass(slots=True)
class _TranscriptHandles:
    ndjson: TextIO
    markdown: TextIO


def _open_transcript_handles(
    transcript_root: Path,
    month: str,
    conversation: Conversation,
) -> _TranscriptHandles:
    transcript_root.mkdir(parents=True, exist_ok=True)
    ndjson = (transcript_root / f"{month}.ndjson").open("w", encoding="utf-8", newline="\n")
    markdown = (transcript_root / f"{month}.md").open("w", encoding="utf-8", newline="\n")
    markdown.write(f"# {_md_inline(conversation.title_display)} — {_md_inline(month)}\n\n")
    markdown.write(f"Conversation ID: `{conversation.id}`\n\n")
    markdown.write(
        "Message bodies are rendered as inert code blocks. Stable message anchors use "
        "`message-<message-id>`.\n\n"
    )
    return _TranscriptHandles(ndjson=ndjson, markdown=markdown)


def _build_conversation(
    root: Path,
    session: Session,
    conversation: Conversation,
    sources: Mapping[int, _SourceCoordinate],
    blobs: Mapping[str, _BlobMetadata],
    materializer: _BlobMaterializer,
) -> tuple[dict[str, Any], int, int, int]:
    bucket = _conversation_bucket(conversation.conversation_type)
    relative_root = conversation_library_relative_root(conversation)
    conversation_root = root / relative_root
    conversation_root.mkdir(parents=True, exist_ok=True)

    segments = list(
        session.scalars(
            select(ConversationSegment)
            .where(ConversationSegment.conversation_id == conversation.id)
            .order_by(ConversationSegment.sequence, ConversationSegment.thread_path)
        )
    )
    participants_by_segment: defaultdict[str, list[Participant]] = defaultdict(list)
    if segments:
        participant_rows = session.execute(
            select(SegmentParticipant.segment_id, Participant)
            .join(Participant, Participant.id == SegmentParticipant.participant_id)
            .where(SegmentParticipant.segment_id.in_([segment.id for segment in segments]))
            .order_by(SegmentParticipant.segment_id, SegmentParticipant.ordinal)
        )
        for segment_id, participant in participant_rows:
            participants_by_segment[segment_id].append(participant)
    segment_records = {
        segment.id: _segment_record(segment, participants_by_segment[segment.id])
        for segment in segments
    }

    attachments = list(
        session.scalars(
            select(AttachmentOccurrence)
            .where(AttachmentOccurrence.conversation_id == conversation.id)
            .where(AttachmentOccurrence.deleted_at.is_(None))
            .order_by(
                AttachmentOccurrence.timestamp_ms,
                AttachmentOccurrence.message_id,
                AttachmentOccurrence.kind,
                AttachmentOccurrence.ordinal,
                AttachmentOccurrence.id,
            )
        )
    )
    attachments_by_message: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    attachment_records: list[dict[str, Any]] = []
    linked_attachment_count = 0
    for attachment in attachments:
        resolved_source = (
            sources.get(attachment.resolved_source_file_id)
            if attachment.resolved_source_file_id
            else None
        )
        blob = blobs.get(attachment.blob_sha256) if attachment.blob_sha256 else None
        library_path: str | None = None
        conversation_relative_path: str | None = None
        if attachment.availability == "resolved":
            if attachment.blob_sha256 is None:
                raise LibraryBuildError(f"Resolved attachment {attachment.id} has no blob digest")
            filename = _attachment_filename(attachment, resolved_source, blob)
            destination_relative = (
                Path("attachments")
                / safe_filesystem_label(attachment.kind, fallback="file", limit=40)
                / _month_bucket(attachment.timestamp_ms)
                / filename
            )
            materializer.link(attachment.blob_sha256, conversation_root / destination_relative)
            library_path = (relative_root / destination_relative).as_posix()
            conversation_relative_path = destination_relative.as_posix()
            linked_attachment_count += 1
        record = _attachment_record(
            attachment,
            sources,
            library_path=library_path,
        )
        record["conversation_relative_path"] = conversation_relative_path
        attachment_records.append(record)
        if attachment.message_id is not None:
            attachments_by_message[attachment.message_id].append(record)
    _write_ndjson(conversation_root / "attachments" / "manifest.ndjson", attachment_records)

    reactions_by_message: defaultdict[str, list[MessageReaction]] = defaultdict(list)
    reaction_rows = session.scalars(
        select(MessageReaction)
        .join(Message, Message.id == MessageReaction.message_id)
        .where(Message.conversation_id == conversation.id)
        .where(Message.deleted_at.is_(None))
        .order_by(MessageReaction.message_id, MessageReaction.ordinal, MessageReaction.id)
    )
    for reaction in reaction_rows:
        reactions_by_message[reaction.message_id].append(reaction)

    handles: dict[str, _TranscriptHandles] = {}
    active_segment_by_month: dict[str, str] = {}
    message_count = 0
    try:
        messages = session.scalars(
            select(Message)
            .where(Message.conversation_id == conversation.id)
            .where(Message.deleted_at.is_(None))
            .order_by(Message.timestamp_ms, Message.id)
        ).yield_per(2_000)
        for message in messages:
            month = _month_bucket(message.timestamp_ms)
            handles_for_month = handles.get(month)
            if handles_for_month is None:
                handles_for_month = _open_transcript_handles(
                    conversation_root / "transcripts", month, conversation
                )
                handles[month] = handles_for_month
            if active_segment_by_month.get(month) != message.segment_id:
                segment_record = segment_records.get(message.segment_id)
                if segment_record is None:
                    raise LibraryBuildError(
                        f"Message {message.id} references absent conversation segment "
                        f"{message.segment_id}"
                    )
                boundary = {
                    "record_type": "segment_boundary",
                    "conversation_id": conversation.id,
                    **segment_record,
                }
                handles_for_month.ndjson.write(_json_value(boundary) + "\n")
                handles_for_month.markdown.write(
                    "## Segment: "
                    f"{_md_inline(str(segment_record['title_display']))}\n\n"
                    f"- Segment ID: `{message.segment_id}`\n"
                    f"- Thread path: `{segment_record['thread_path']}`\n"
                    f"- Source category: `{segment_record['source_category']}`\n\n"
                )
                active_segment_by_month[month] = message.segment_id
            record = _message_record(
                message,
                sources.get(message.source_file_id),
                reactions_by_message[message.id],
                attachments_by_message[message.id],
            )
            handles_for_month.ndjson.write(_json_value(record) + "\n")
            handles_for_month.markdown.write(_markdown_message(record) + "\n")
            message_count += 1
    finally:
        for item in handles.values():
            item.ndjson.close()
            item.markdown.close()

    manifest = {
        "format_version": LIBRARY_FORMAT_VERSION,
        "conversation_id": conversation.id,
        "conversation_key": conversation.conversation_key,
        "conversation_type": conversation.conversation_type,
        "taxonomy": bucket,
        "title_raw": conversation.title_raw,
        "title_display": conversation.title_display,
        "primary_category": conversation.primary_category,
        "participant_count": conversation.participant_count,
        "first_activity_ms": conversation.first_activity_ms,
        "first_activity_utc": _utc_iso(conversation.first_activity_ms),
        "last_activity_ms": conversation.last_activity_ms,
        "last_activity_utc": _utc_iso(conversation.last_activity_ms),
        "message_count": message_count,
        "sent_count": conversation.sent_count,
        "received_count": conversation.received_count,
        "attachment_occurrence_count": len(attachments),
        "unique_attachment_count": conversation.unique_attachment_count,
        "attachment_bytes": conversation.attachment_bytes,
        "active_days": conversation.active_days,
        "completeness": conversation.completeness,
        "segments": [segment_records[segment.id] for segment in segments],
        "library_path": relative_root.as_posix(),
    }
    _write_json(conversation_root / "manifest.json", manifest)
    return manifest, message_count, len(handles) * 2, linked_attachment_count


def _media_owner_label(
    attachment: AttachmentOccurrence,
    conversations: Mapping[str, Conversation],
) -> str:
    if attachment.conversation_id and attachment.conversation_id in conversations:
        conversation = conversations[attachment.conversation_id]
        return safe_filesystem_label(conversation.title_display, fallback="conversation", limit=36)
    if attachment.activity_record_id:
        return f"activity-{attachment.activity_record_id[:12]}"
    return "unlinked"


def unlinked_media_library_path(
    attachment: AttachmentOccurrence,
    *,
    source_relative_path: str | None,
    source_extension: str | None,
    blob_extension: str | None = None,
) -> str | None:
    """Return the safe archive-relative path used by the readable media library."""

    if attachment.availability != "unlinked":
        return None
    source_hint = (
        PurePosixPath(source_relative_path).name if source_relative_path is not None else None
    )
    hint = (
        PurePosixPath(attachment.filename.replace("\\", "/")).name
        if attachment.filename
        else source_hint
        or (
            PurePosixPath(attachment.source_uri.split("?", 1)[0].replace("\\", "/")).name
            if attachment.source_uri
            else attachment.kind
        )
    )
    filename = _attachment_filename_from_values(
        attachment,
        hint=hint,
        source_extension=source_extension,
        blob_extension=blob_extension,
    )
    return (
        Path("library")
        / "media"
        / "unlinked"
        / Path(*_date_parts(attachment.timestamp_ms))
        / f"unlinked--{filename}"
    ).as_posix()


def _build_media_views(
    root: Path,
    session: Session,
    sources: Mapping[int, _SourceCoordinate],
    blobs: Mapping[str, _BlobMetadata],
    conversations: Mapping[str, Conversation],
    materializer: _BlobMaterializer,
) -> tuple[int, int]:
    media_root = root / "media"
    for bucket in ("photos", "videos", "audio", "files", "unlinked"):
        (media_root / bucket).mkdir(parents=True, exist_ok=True)
    records_by_bucket: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    all_records: list[dict[str, Any]] = []
    link_count = 0
    attachments = session.scalars(
        select(AttachmentOccurrence)
        .where(AttachmentOccurrence.deleted_at.is_(None))
        .order_by(
            AttachmentOccurrence.timestamp_ms,
            AttachmentOccurrence.conversation_id,
            AttachmentOccurrence.activity_record_id,
            AttachmentOccurrence.id,
        )
    )
    for attachment in attachments:
        bucket = _attachment_bucket(attachment)
        resolved_source = (
            sources.get(attachment.resolved_source_file_id)
            if attachment.resolved_source_file_id
            else None
        )
        blob = blobs.get(attachment.blob_sha256) if attachment.blob_sha256 else None
        library_path: str | None = None
        if attachment.availability in {"resolved", "unlinked"}:
            if attachment.blob_sha256 is None:
                raise LibraryBuildError(
                    f"Available attachment {attachment.id} has no catalogued blob digest"
                )
            filename = _attachment_filename(attachment, resolved_source, blob)
            if attachment.availability == "unlinked":
                archive_relative = unlinked_media_library_path(
                    attachment,
                    source_relative_path=(
                        resolved_source.relative_path if resolved_source else None
                    ),
                    source_extension=resolved_source.extension if resolved_source else None,
                    blob_extension=blob.extension if blob else None,
                )
                assert archive_relative is not None
                destination_relative = Path(archive_relative).relative_to("library")
            else:
                destination_relative = (
                    Path("media")
                    / bucket
                    / Path(*_date_parts(attachment.timestamp_ms))
                    / f"{_media_owner_label(attachment, conversations)}--{filename}"
                )
            materializer.link(attachment.blob_sha256, root / destination_relative)
            library_path = destination_relative.as_posix()
            link_count += 1
        record = _attachment_record(attachment, sources, library_path=library_path)
        record["media_bucket"] = bucket
        all_records.append(record)
        records_by_bucket[bucket].append(record)

    _write_ndjson(media_root / "manifest.ndjson", all_records)
    for bucket in ("photos", "videos", "audio", "files", "unlinked"):
        _write_ndjson(media_root / bucket / "manifest.ndjson", records_by_bucket[bucket])
    return len(all_records), link_count


def _coverage_record(area: CoverageArea) -> dict[str, Any]:
    return {
        "key": area.key,
        "label": area.label,
        "status": area.status,
        "raw_count": area.raw_count,
        "logical_count": area.logical_count,
        "description": area.description,
        "source_paths": area.source_paths_json,
    }


def _activity_record(
    activity: ActivityRecord,
    source: _SourceCoordinate | None,
    attachment_ids: Sequence[str],
) -> dict[str, Any]:
    return {
        "record_type": "activity",
        "activity_id": activity.id,
        "source_record_id": activity.source_record_id,
        "parent_id": activity.parent_id,
        "source": _source_record(source),
        "json_pointer": activity.json_pointer,
        "category": activity.category,
        "subtype": activity.subtype,
        "source_fbid": activity.source_fbid,
        "timestamp_ms": activity.timestamp_ms,
        "timestamp_utc": _utc_iso(activity.timestamp_ms),
        "date_status": "known" if _utc_iso(activity.timestamp_ms) else "unknown",
        "title_raw": activity.title_raw,
        "title_display": activity.title_display,
        "content_raw": activity.content_raw,
        "content_display": activity.content_display,
        "actor_raw": activity.actor_raw,
        "actor_display": activity.actor_display,
        "target_uri": activity.target_uri,
        "availability": activity.availability,
        "attachment_ids": list(attachment_ids),
        "extra": activity.extra_json,
    }


def _activity_views(activity: ActivityRecord) -> set[str]:
    category = activity.category.casefold()
    subtype = activity.subtype.casefold()
    views: set[str] = set()
    if category == "posts" or any(token in subtype for token in ("post", "album", "check_in")):
        views.add("posts")
    if "comment" in subtype:
        views.add("comments")
    if "event" in category or "event" in subtype:
        views.add("events")
    if any(
        token in category or token in subtype
        for token in ("friend", "follower", "contact", "block", "people", "person")
    ):
        views.add("people")
    return views


def _coverage_views(area: CoverageArea) -> set[str]:
    key = area.key.casefold()
    views: set[str] = set()
    if key in {"posts", "general_posts", "albums"} or "post" in key or "album" in key:
        views.add("posts")
    if "comment" in key:
        views.add("comments")
    if "event" in key:
        views.add("events")
    if any(token in key for token in ("friend", "follower", "contact", "block", "people")):
        views.add("people")
    return views


def _build_activity_views(
    root: Path,
    session: Session,
    sources: Mapping[int, _SourceCoordinate],
) -> int:
    view_names = ("posts", "comments", "events", "people")
    for view in ("activity", *view_names):
        (root / view).mkdir(parents=True, exist_ok=True)
    attachment_ids: defaultdict[str, list[str]] = defaultdict(list)
    for activity_id, attachment_id in session.execute(
        select(AttachmentOccurrence.activity_record_id, AttachmentOccurrence.id)
        .where(AttachmentOccurrence.activity_record_id.is_not(None))
        .where(AttachmentOccurrence.deleted_at.is_(None))
        .order_by(AttachmentOccurrence.activity_record_id, AttachmentOccurrence.ordinal)
    ):
        assert activity_id is not None
        attachment_ids[activity_id].append(attachment_id)

    records_by_view: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    all_records: list[dict[str, Any]] = []
    activities = session.scalars(
        select(ActivityRecord)
        .where(ActivityRecord.deleted_at.is_(None))
        .order_by(ActivityRecord.timestamp_ms, ActivityRecord.category, ActivityRecord.id)
    )
    for activity in activities:
        record = _activity_record(
            activity,
            sources.get(activity.source_file_id),
            attachment_ids[activity.id],
        )
        all_records.append(record)
        for view in _activity_views(activity):
            records_by_view[view].append(record)
    _write_ndjson(root / "activity" / "records.ndjson", all_records)

    coverage_by_view: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    all_coverage: list[dict[str, Any]] = []
    for area in session.scalars(select(CoverageArea).order_by(CoverageArea.key)):
        record = _coverage_record(area)
        all_coverage.append(record)
        for view in _coverage_views(area):
            coverage_by_view[view].append(record)
    _write_json(root / "activity" / "coverage.json", all_coverage)
    for view in view_names:
        _write_ndjson(root / view / "records.ndjson", records_by_view[view])
        _write_json(root / view / "coverage.json", coverage_by_view[view])
    return len(all_records)


def _validated_library_tree(layout: ArchiveLayout, path: Path, *, label: str) -> None:
    try:
        relative = path.relative_to(layout.root)
    except ValueError as exc:
        raise LibraryBuildError(f"{label} is outside the archive root: {path}") from exc
    if len(relative.parts) != 1 or relative.parts[0] != path.name:
        raise LibraryBuildError(f"{label} is not a direct archive-root child: {path}")
    try:
        path_stat = path.lstat()
    except FileNotFoundError as exc:
        raise LibraryBuildError(f"{label} disappeared during validation: {path}") from exc
    if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISDIR(path_stat.st_mode):
        raise LibraryBuildError(f"{label} is not a real directory: {path}")


def _remove_validated_library_tree(layout: ArchiveLayout, path: Path, *, label: str) -> None:
    _validated_library_tree(layout, path, label=label)
    remove_directory_tree_beneath(path, layout.root, missing_ok=False)


def _stale_library_trees(layout: ArchiveLayout, prefix: str) -> list[Path]:
    paths = sorted(layout.root.glob(f"{prefix}*"), key=lambda path: path.name)
    for path in paths:
        _validated_library_tree(layout, path, label=f"Stale {prefix} tree")
        validate_directory_tree_beneath(path, layout.root)
    return paths


def _recover_library_build_state_locked(layout: ArchiveLayout) -> None:
    """Recover only crash remnants after the caller has excluded every active builder."""

    stale_builds = _stale_library_trees(layout, ".library-build-")
    previous_trees = _stale_library_trees(layout, ".library-previous-")
    try:
        library_stat = layout.library.lstat()
    except FileNotFoundError:
        library_stat = None
    if library_stat is not None and (
        stat.S_ISLNK(library_stat.st_mode) or not stat.S_ISDIR(library_stat.st_mode)
    ):
        raise LibraryBuildError(f"Filesystem library is not a real directory: {layout.library}")
    if library_stat is None and len(previous_trees) > 1:
        raise LibraryBuildError(
            "Filesystem library is missing and multiple previous trees require manual review"
        )

    if library_stat is None and len(previous_trees) == 1:
        validate_directory_tree_beneath(previous_trees[0], layout.root)
        replace_entry_beneath(
            previous_trees[0],
            layout.library,
            layout.root,
            expected_kind="directory",
        )
        previous_trees = []
    elif library_stat is not None:
        for previous in previous_trees:
            _remove_validated_library_tree(
                layout,
                previous,
                label="Stale previous filesystem library",
            )

    for staged in stale_builds:
        _remove_validated_library_tree(
            layout,
            staged,
            label="Stale staged filesystem library",
        )


def recover_library_build_state(
    layout: ArchiveLayout,
    *,
    lock_held: bool = False,
) -> None:
    """Recover validated crash remnants without racing an active library builder."""

    if not lock_held:
        with library_build_lock(layout):
            pass
        return
    _recover_library_build_state_locked(layout)


def _replace_library_tree(layout: ArchiveLayout, staged: Path) -> None:
    destination = layout.library
    if destination.is_symlink():
        raise LibraryBuildError(f"Refusing to replace a symlinked library directory: {destination}")
    backup = layout.root / f".library-previous-{uuid.uuid4().hex}"
    had_previous = destination.exists()
    try:
        if had_previous:
            os.replace(destination, backup)
        os.replace(staged, destination)
    except BaseException:
        if had_previous and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    if backup.exists():
        _remove_validated_library_tree(
            layout,
            backup,
            label="Previous filesystem library",
        )


def build_filesystem_library(
    layout: ArchiveLayout,
    engine: Engine,
    *,
    progress: ProgressCallback | None = None,
    catalog_session: Session | None = None,
    lock_held: bool = False,
    source_lock: SourceImportPublicationLease | None = None,
    deletion_lock_lease: DeletionLockLease | None = None,
    deletion_operation_id: str | None = None,
) -> LibraryBuildSummary:
    """Rebuild the deterministic human-readable library from catalog rows and verified blobs.

    The source chunk directories are never read. A complete tree is staged beside ``library`` and
    promoted only after every manifest and link has been produced successfully.
    """

    if not lock_held:
        if (
            source_lock is not None
            or deletion_lock_lease is not None
            or deletion_operation_id is not None
        ):
            raise LibraryBuildError("Explicit publication leases require held lock mode")
        with (
            source_import_publication_lock(layout) as acquired_source_lock,
            library_build_lock(layout),
        ):
            return build_filesystem_library(
                layout,
                engine,
                progress=progress,
                catalog_session=catalog_session,
                lock_held=True,
                source_lock=acquired_source_lock,
            )
    if source_lock is None:
        raise LibraryBuildError("Held library builds require a live source/publication lease")
    source_lock.assert_held(layout)
    if (deletion_lock_lease is None) != (deletion_operation_id is None):
        raise LibraryBuildError("Deletion-scoped library authorization is incomplete")
    if deletion_lock_lease is None:
        assert_source_import_publication_allowed(
            layout,
            engine,
            lease=source_lock,
            publisher="Filesystem library publication",
        )
    else:
        if catalog_session is None:
            raise LibraryBuildError(
                "Deletion-scoped library publication requires the held catalog transaction"
            )
        deletion_lock_lease.assert_held(layout)
        operation = catalog_session.get(DeletionOperation, deletion_operation_id)
        if operation is None or operation.status not in {"committed", "normalized_purged"}:
            raise LibraryBuildError(
                "Deletion-scoped library authorization does not match an active purge"
            )
    _recover_library_build_state_locked(layout)
    layout.ensure_runtime_directories()
    source_manifests = _source_manifest_metadata(layout)
    if progress:
        progress("Reconciling unlinked catalog media", 0, None)
    if catalog_session is None:
        with Session(engine) as reconciliation_session, reconciliation_session.begin():
            unlinked_count = _reconcile_unlinked_media_session(reconciliation_session)
    else:
        unlinked_count = _reconcile_unlinked_media_session(catalog_session)
        catalog_session.flush()

    staged = Path(tempfile.mkdtemp(prefix=".library-build-", dir=layout.root))
    staged.chmod(0o700)
    summary = LibraryBuildSummary(unlinked_media_count=unlinked_count)
    try:
        session_context = (
            Session(engine) if catalog_session is None else nullcontext(catalog_session)
        )
        with session_context as session:
            sources = _source_coordinates(session)
            blobs = {
                blob.sha256: _BlobMetadata(
                    sha256=blob.sha256,
                    size_bytes=blob.size_bytes,
                    media_type=blob.media_type,
                    extension=blob.extension,
                )
                for blob in session.scalars(select(Blob).order_by(Blob.sha256))
            }
            materializer = _BlobMaterializer(layout, blobs)
            conversations = list(
                session.scalars(
                    select(Conversation)
                    .where(Conversation.deleted_at.is_(None))
                    .order_by(
                        Conversation.conversation_type,
                        Conversation.title_search,
                        Conversation.id,
                    )
                )
            )
            conversation_map = {conversation.id: conversation for conversation in conversations}
            for bucket in ("direct", "groups", "unknown", "system"):
                (staged / "conversations" / bucket).mkdir(parents=True, exist_ok=True)
            conversation_manifests: list[dict[str, Any]] = []
            total = len(conversations)
            for index, conversation in enumerate(conversations, start=1):
                manifest, message_count, transcript_count, attachment_link_count = (
                    _build_conversation(
                        staged,
                        session,
                        conversation,
                        sources,
                        blobs,
                        materializer,
                    )
                )
                conversation_manifests.append(manifest)
                summary.message_count += message_count
                summary.transcript_file_count += transcript_count
                summary.conversation_attachment_link_count += attachment_link_count
                if progress and (index % 25 == 0 or index == total):
                    progress("Building conversation library", index, total)
            summary.conversation_count = len(conversations)
            _write_ndjson(staged / "conversations" / "manifest.ndjson", conversation_manifests)

            summary.attachment_occurrence_count, summary.media_view_link_count = _build_media_views(
                staged,
                session,
                sources,
                blobs,
                conversation_map,
                materializer,
            )
            summary.unlinked_media_count = (
                session.scalar(
                    select(func.count(AttachmentOccurrence.id))
                    .where(AttachmentOccurrence.availability == "unlinked")
                    .where(AttachmentOccurrence.deleted_at.is_(None))
                )
                or 0
            )
            summary.activity_record_count = _build_activity_views(staged, session, sources)
            summary.hardlink_count = materializer.hardlinks
            summary.copy_count = materializer.copies

        _write_json(
            staged / "manifest.json",
            {
                "format_version": LIBRARY_FORMAT_VERSION,
                "counts": asdict(summary),
                "taxonomy": {
                    "conversations": ["direct", "groups", "unknown", "system"],
                    "media": ["photos", "videos", "audio", "files", "unlinked"],
                    "activity": ["posts", "comments", "events", "people"],
                },
                "source_manifests": source_manifests,
                "timestamp_policy": "UTC; unknown dates remain null and use unknown-date paths",
                "blob_policy": (
                    "Library files are hardlinks when supported and verified copies otherwise; "
                    "treat them as read-only."
                ),
            },
        )
        (staged / "README.md").write_text(
            "# Facebook archive library\n\n"
            "This tree is generated deterministically from `catalog/facebook.sqlite` and "
            "`blobs/sha256`. Treat linked media files as read-only; rebuild the library after "
            "catalog changes. NDJSON manifests preserve stable IDs and source provenance. "
            "Unknown dates are never inferred from filesystem metadata.\n",
            encoding="utf-8",
            newline="\n",
        )
        _replace_library_tree(layout, staged)
        if progress:
            progress(
                "Filesystem library complete",
                summary.conversation_count,
                summary.conversation_count,
            )
        if deletion_lock_lease is not None:
            deletion_lock_lease.assert_held(layout)
        source_lock.assert_held(layout)
        return summary
    except BaseException:
        if staged.exists():
            _remove_validated_library_tree(
                layout,
                staged,
                label="Failed staged filesystem library",
            )
        raise


# A concise service alias for CLI and orchestration integration.
build_library = build_filesystem_library
