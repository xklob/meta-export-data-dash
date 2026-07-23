from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from meta_archive.models import Conversation, ConversationSegment
from meta_archive.paths import (
    ArchiveLayout,
    UnsafeArchivePathError,
    archive_entry_stat_beneath,
    open_directory_beneath,
    open_regular_file_beneath,
    regular_file_stat_beneath,
    replace_entry_beneath,
    unlink_regular_file_beneath,
)
from meta_archive.services.library import (
    library_build_lock,
)
from meta_archive.services.message_importer import redact_sensitive_message_metadata

_CONVERSATION_ID_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_MAX_TRANSCRIPT_LINE_BYTES = 64 * 1024 * 1024
_TEMP_PREFIX = ".support-library-scrub-"
_TEMP_SUFFIX = ".tmp"


class SupportLibraryScrubError(RuntimeError):
    """A targeted support-transcript repair could not complete safely."""


@dataclass(slots=True)
class SupportLibraryScrubSummary:
    support_conversation_count: int = 0
    conversation_directories_found: int = 0
    transcript_files_scanned: int = 0
    transcript_records_scanned: int = 0
    message_records_scanned: int = 0
    message_records_changed: int = 0
    sensitive_fields_removed: int = 0
    transcript_files_rewritten: int = 0
    transcript_files_unchanged: int = 0
    stale_temp_files_removed: int = 0


@dataclass(frozen=True, slots=True)
class _FileIdentity:
    device: int
    inode: int
    size: int
    mtime_ns: int
    links: int


@dataclass(frozen=True, slots=True)
class _StagedTranscript:
    target: Path
    temporary: Path
    target_identity: _FileIdentity
    temporary_identity: _FileIdentity


@dataclass(frozen=True, slots=True)
class _SupportConversation:
    conversation: Conversation
    segment_ids: frozenset[str]
    known_segment_ids: frozenset[str]


def _identity(value: os.stat_result) -> _FileIdentity:
    return _FileIdentity(
        device=value.st_dev,
        inode=value.st_ino,
        size=value.st_size,
        mtime_ns=value.st_mtime_ns,
        links=value.st_nlink,
    )


def _support_conversations(session: Session) -> list[_SupportConversation]:
    grouped: dict[str, tuple[Conversation, set[str]]] = {}
    rows = session.execute(
        select(Conversation, ConversationSegment.id)
        .join(
            ConversationSegment,
            ConversationSegment.conversation_id == Conversation.id,
        )
        .where(ConversationSegment.source_category == "support")
        .order_by(Conversation.id, ConversationSegment.id)
    )
    for conversation, segment_id in rows:
        if _CONVERSATION_ID_PATTERN.fullmatch(conversation.id) is None:
            raise SupportLibraryScrubError(
                "A support conversation has an unsafe catalog identifier"
            )
        grouped.setdefault(conversation.id, (conversation, set()))[1].add(segment_id)
    known: dict[str, set[str]] = {conversation_id: set() for conversation_id in grouped}
    if known:
        all_segment_rows = session.execute(
            select(ConversationSegment.conversation_id, ConversationSegment.id)
            .where(ConversationSegment.conversation_id.in_(known))
            .order_by(ConversationSegment.conversation_id, ConversationSegment.id)
        )
        for conversation_id, segment_id in all_segment_rows:
            known[conversation_id].add(segment_id)
    return [
        _SupportConversation(
            conversation=conversation,
            segment_ids=frozenset(segment_ids),
            known_segment_ids=frozenset(known[conversation.id]),
        )
        for conversation, segment_ids in grouped.values()
    ]


def _find_support_conversation_roots(
    layout: ArchiveLayout,
    conversations: list[_SupportConversation],
) -> dict[str, Path]:
    """Locate existing library roots by stable ID, including pre-rename directory names."""

    if not conversations:
        return {}
    wanted = {item.conversation.id for item in conversations}
    conversations_root = layout.library / "conversations"
    try:
        root_stat = archive_entry_stat_beneath(conversations_root, layout.root)
    except UnsafeArchivePathError as exc:
        raise SupportLibraryScrubError("The conversation library path is unsafe") from exc
    if root_stat is None:
        raise SupportLibraryScrubError("The conversation library is unexpectedly absent")
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise SupportLibraryScrubError("The conversation library is not a real directory")

    found: dict[str, Path] = {}
    for bucket in ("direct", "groups", "system", "unknown"):
        bucket_root = conversations_root / bucket
        try:
            bucket_stat = archive_entry_stat_beneath(bucket_root, layout.root)
        except UnsafeArchivePathError as exc:
            raise SupportLibraryScrubError("A conversation-library bucket is unsafe") from exc
        if bucket_stat is None:
            continue
        if stat.S_ISLNK(bucket_stat.st_mode) or not stat.S_ISDIR(bucket_stat.st_mode):
            raise SupportLibraryScrubError("A conversation-library bucket is not a real directory")

        descriptor: int | None = None
        try:
            descriptor = open_directory_beneath(bucket_root, layout.root)
            with os.scandir(descriptor) as iterator:
                entries = list(iterator)
            for entry in entries:
                _separator, marker, candidate_id = entry.name.rpartition("--")
                if not marker or candidate_id not in wanted:
                    continue
                entry_stat = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(entry_stat.st_mode) or not stat.S_ISDIR(entry_stat.st_mode):
                    raise SupportLibraryScrubError("A support conversation-library entry is unsafe")
                if candidate_id in found:
                    raise SupportLibraryScrubError(
                        "A support conversation has duplicate filesystem-library roots"
                    )
                found[candidate_id] = bucket_root / entry.name
        except (OSError, UnsafeArchivePathError) as exc:
            raise SupportLibraryScrubError(
                "A conversation-library bucket changed unsafely"
            ) from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)

    required = {
        item.conversation.id for item in conversations if item.conversation.deleted_at is None
    }
    if not required.issubset(found):
        raise SupportLibraryScrubError(
            "One or more support conversations are absent from the filesystem library"
        )
    return found


def _transcript_files(
    layout: ArchiveLayout,
    conversation_root: Path,
    summary: SupportLibraryScrubSummary,
) -> list[Path]:
    transcript_root = conversation_root / "transcripts"
    try:
        root_stat = archive_entry_stat_beneath(transcript_root, layout.root)
    except UnsafeArchivePathError as exc:
        raise SupportLibraryScrubError("A support transcript path is unsafe") from exc
    if root_stat is None:
        return []
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise SupportLibraryScrubError("A support transcript root is not a real directory")

    descriptor: int | None = None
    try:
        descriptor = open_directory_beneath(transcript_root, layout.root)
        with os.scandir(descriptor) as iterator:
            entries = sorted(iterator, key=lambda entry: entry.name)
        targets: list[Path] = []
        stale_temporaries: list[Path] = []
        for entry in entries:
            entry_stat = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(entry_stat.st_mode) or not stat.S_ISREG(entry_stat.st_mode):
                raise SupportLibraryScrubError(
                    "A support transcript directory contains an unsafe entry"
                )
            if entry_stat.st_nlink != 1:
                raise SupportLibraryScrubError(
                    "A support transcript directory contains a hard-linked entry"
                )
            path = transcript_root / entry.name
            if entry.name.startswith(_TEMP_PREFIX) and entry.name.endswith(_TEMP_SUFFIX):
                stale_temporaries.append(path)
            elif entry.name.endswith(".ndjson"):
                targets.append(path)
        for temporary in stale_temporaries:
            unlink_regular_file_beneath(temporary, layout.root, missing_ok=False)
            summary.stale_temp_files_removed += 1
    except (OSError, UnsafeArchivePathError) as exc:
        raise SupportLibraryScrubError("A support transcript directory changed unsafely") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)

    summary.conversation_directories_found += 1
    return targets


def _open_temporary_sibling(layout: ArchiveLayout, target: Path) -> tuple[Path, BinaryIO]:
    descriptor: int | None = None
    temporary_descriptor: int | None = None
    temporary_name: str | None = None
    try:
        descriptor = open_directory_beneath(target.parent, layout.root)
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | int(getattr(os, "O_CLOEXEC", 0))
            | int(getattr(os, "O_NOFOLLOW", 0))
        )
        for _attempt in range(16):
            temporary_name = f"{_TEMP_PREFIX}{uuid.uuid4().hex}{_TEMP_SUFFIX}"
            try:
                temporary_descriptor = os.open(
                    temporary_name,
                    flags,
                    0o600,
                    dir_fd=descriptor,
                )
                break
            except FileExistsError:
                temporary_name = None
        if temporary_descriptor is None or temporary_name is None:
            raise SupportLibraryScrubError("Could not allocate a private transcript repair file")
        os.fchmod(temporary_descriptor, 0o600)
        temporary_stat = os.fstat(temporary_descriptor)
        if (
            not stat.S_ISREG(temporary_stat.st_mode)
            or temporary_stat.st_nlink != 1
            or stat.S_IMODE(temporary_stat.st_mode) != 0o600
        ):
            raise SupportLibraryScrubError("A transcript repair file is not private and regular")
        os.fsync(descriptor)
        handle = os.fdopen(temporary_descriptor, "wb", buffering=0)
        temporary_descriptor = None
        return target.parent / temporary_name, handle
    except (OSError, UnsafeArchivePathError) as exc:
        raise SupportLibraryScrubError("Could not create a private transcript repair file") from exc
    finally:
        if temporary_descriptor is not None:
            os.close(temporary_descriptor)
        if descriptor is not None:
            os.close(descriptor)


def _repair_record(
    raw: bytes,
    summary: SupportLibraryScrubSummary,
    support_segment_ids: frozenset[str],
    known_segment_ids: frozenset[str],
) -> tuple[bytes, bool]:
    if not raw.endswith(b"\n"):
        raise SupportLibraryScrubError("A support transcript record is not newline terminated")
    try:
        record = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SupportLibraryScrubError("A support transcript contains malformed JSON") from exc
    if not isinstance(record, dict):
        raise SupportLibraryScrubError("A support transcript record is not a JSON object")
    record_type = record.get("record_type")
    if record_type == "segment_boundary":
        return raw, False
    if record_type != "message":
        raise SupportLibraryScrubError("A support transcript contains an unexpected record type")

    segment_id = record.get("segment_id")
    if not isinstance(segment_id, str):
        raise SupportLibraryScrubError("A support message record has an unexpected schema")
    if segment_id not in known_segment_ids:
        raise SupportLibraryScrubError("A support message references an unknown segment")
    if segment_id not in support_segment_ids:
        return raw, False

    extra = record.get("extra")
    omitted = record.get("sensitive_fields_omitted")
    if (
        not isinstance(extra, dict)
        or not isinstance(omitted, list)
        or not all(isinstance(value, str) for value in omitted)
    ):
        raise SupportLibraryScrubError("A support message record has an unexpected schema")
    safe_extra, removed = redact_sensitive_message_metadata(extra)
    safe_omitted = sorted(set(omitted) | set(removed))
    changed = safe_extra != extra or safe_omitted != omitted
    record["extra"] = safe_extra
    record["sensitive_fields_omitted"] = safe_omitted
    summary.message_records_scanned += 1
    summary.sensitive_fields_removed += len(removed)
    if changed:
        summary.message_records_changed += 1
        return (
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n",
            True,
        )
    return raw, False


def _write_all(handle: BinaryIO, value: bytes) -> None:
    remaining = memoryview(value)
    while remaining:
        written = handle.write(remaining)
        if written is None or written <= 0:
            raise SupportLibraryScrubError("A transcript repair file could not be written")
        remaining = remaining[written:]


def _stage_transcript(
    layout: ArchiveLayout,
    target: Path,
    summary: SupportLibraryScrubSummary,
    support_segment_ids: frozenset[str],
    known_segment_ids: frozenset[str],
) -> _StagedTranscript | None:
    temporary: Path | None = None
    temporary_handle: BinaryIO | None = None
    try:
        source_handle = open_regular_file_beneath(target, layout.root)
        with source_handle:
            source_stat = os.fstat(source_handle.fileno())
            if source_stat.st_nlink != 1:
                raise SupportLibraryScrubError("A support transcript is hard-linked")
            target_identity = _identity(source_stat)
            temporary, temporary_handle = _open_temporary_sibling(layout, target)
            original_digest = hashlib.sha256()
            repaired_digest = hashlib.sha256()
            record_count = 0
            while True:
                raw = source_handle.readline(_MAX_TRANSCRIPT_LINE_BYTES + 1)
                if not raw:
                    break
                if len(raw) > _MAX_TRANSCRIPT_LINE_BYTES:
                    raise SupportLibraryScrubError("A support transcript record exceeds the limit")
                original_digest.update(raw)
                encoded, _changed = _repair_record(
                    raw,
                    summary,
                    support_segment_ids,
                    known_segment_ids,
                )
                repaired_digest.update(encoded)
                _write_all(temporary_handle, encoded)
                record_count += 1
            if record_count == 0:
                raise SupportLibraryScrubError("A support transcript is unexpectedly empty")
            source_after = os.fstat(source_handle.fileno())
            if _identity(source_after) != target_identity:
                raise SupportLibraryScrubError("A support transcript changed while being read")
            temporary_handle.flush()
            os.fsync(temporary_handle.fileno())
            temporary_identity = _identity(os.fstat(temporary_handle.fileno()))
            temporary_handle.close()
            temporary_handle = None

        summary.transcript_records_scanned += record_count
        summary.transcript_files_scanned += 1
        needs_rewrite = (
            original_digest.digest() != repaired_digest.digest()
            or stat.S_IMODE(source_stat.st_mode) != 0o600
        )
        if not needs_rewrite:
            assert temporary is not None
            unlink_regular_file_beneath(temporary, layout.root, missing_ok=False)
            summary.transcript_files_unchanged += 1
            return None
        assert temporary is not None
        staged = _StagedTranscript(
            target=target,
            temporary=temporary,
            target_identity=target_identity,
            temporary_identity=temporary_identity,
        )
        temporary = None
        return staged
    except (OSError, UnsafeArchivePathError) as exc:
        raise SupportLibraryScrubError("A support transcript changed unsafely") from exc
    finally:
        if temporary_handle is not None:
            temporary_handle.close()
        if temporary is not None:
            with suppress(UnsafeArchivePathError):
                unlink_regular_file_beneath(temporary, layout.root, missing_ok=True)


def _publish_transcript(layout: ArchiveLayout, staged: _StagedTranscript) -> None:
    try:
        current = regular_file_stat_beneath(staged.target, layout.root)
        if _identity(current) != staged.target_identity or current.st_nlink != 1:
            raise SupportLibraryScrubError("A support transcript changed before publication")
        temporary = regular_file_stat_beneath(staged.temporary, layout.root)
        if (
            _identity(temporary) != staged.temporary_identity
            or temporary.st_nlink != 1
            or stat.S_IMODE(temporary.st_mode) != 0o600
            or not stat.S_ISREG(temporary.st_mode)
        ):
            raise SupportLibraryScrubError("A transcript repair file changed before publication")
        replace_entry_beneath(
            staged.temporary,
            staged.target,
            layout.root,
            expected_kind="file",
            replace_existing=True,
        )
        published = regular_file_stat_beneath(staged.target, layout.root)
        if (
            _identity(published) != staged.temporary_identity
            or published.st_nlink != 1
            or stat.S_IMODE(published.st_mode) != 0o600
        ):
            raise SupportLibraryScrubError("A repaired support transcript is not owner-only")
    except (OSError, UnsafeArchivePathError) as exc:
        raise SupportLibraryScrubError(
            "A support transcript could not be published safely"
        ) from exc


def scrub_support_library(
    layout: ArchiveLayout,
    engine: Engine,
) -> SupportLibraryScrubSummary:
    """Redact legacy structured secrets from only existing support transcript NDJSON files."""

    summary = SupportLibraryScrubSummary()
    staged_files: list[_StagedTranscript] = []
    seen_targets: set[Path] = set()
    with library_build_lock(layout):
        try:
            library_stat = archive_entry_stat_beneath(layout.library, layout.root)
        except UnsafeArchivePathError as exc:
            raise SupportLibraryScrubError("The filesystem library path is unsafe") from exc
        if library_stat is None:
            return summary
        if stat.S_ISLNK(library_stat.st_mode) or not stat.S_ISDIR(library_stat.st_mode):
            raise SupportLibraryScrubError("The filesystem library is not a real directory")

        with Session(engine) as session:
            conversations = _support_conversations(session)
            summary.support_conversation_count = len(conversations)
            conversation_roots = _find_support_conversation_roots(layout, conversations)
            try:
                for item in conversations:
                    conversation_root = conversation_roots.get(item.conversation.id)
                    if conversation_root is None:
                        continue
                    for target in _transcript_files(layout, conversation_root, summary):
                        if target in seen_targets:
                            raise SupportLibraryScrubError(
                                "A support transcript target is duplicated"
                            )
                        seen_targets.add(target)
                        staged = _stage_transcript(
                            layout,
                            target,
                            summary,
                            item.segment_ids,
                            item.known_segment_ids,
                        )
                        if staged is not None:
                            staged_files.append(staged)
                for staged in staged_files:
                    _publish_transcript(layout, staged)
                    summary.transcript_files_rewritten += 1
            finally:
                for staged in staged_files:
                    with suppress(UnsafeArchivePathError):
                        unlink_regular_file_beneath(
                            staged.temporary,
                            layout.root,
                            missing_ok=True,
                        )
    return summary
