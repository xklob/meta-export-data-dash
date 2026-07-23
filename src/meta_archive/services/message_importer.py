from __future__ import annotations

import uuid
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, cast
from urllib.parse import unquote, urlsplit

import orjson
from sqlalchemy import Engine, and_, delete, func, or_, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from meta_archive import __version__
from meta_archive.models import (
    ArchiveSetting,
    AttachmentCandidate,
    AttachmentOccurrence,
    Conversation,
    ConversationSegment,
    ConversationSegmentOverride,
    CoverageArea,
    ImportRun,
    IntegrityAnomaly,
    Job,
    Message,
    MessageReaction,
    Participant,
    SegmentParticipant,
    SourceChunk,
    SourceFile,
    SourceRecord,
    Star,
)
from meta_archive.paths import ArchiveLayout
from meta_archive.services.hashing import sha256_file
from meta_archive.services.identifiers import canonical_fingerprint, stable_id
from meta_archive.services.purge_rewrite import is_purge_tombstone
from meta_archive.services.semantic_state import invalidate_ready_semantic_indexes
from meta_archive.services.source_import_lock import (
    SourceImportPublicationLease,
    assert_source_import_publication_allowed,
    source_import_publication_lock,
)
from meta_archive.services.stars import (
    StarTargetKey,
    StarTargetType,
    delete_stars_for_targets,
    resolve_star_targets,
)
from meta_archive.services.text import NormalizedText, normalize_text

ProgressCallback = Callable[[str, int, int | None], None]

MESSAGE_PARSER_VERSION = f"{__version__}:messages-v1"
DEFAULT_OWNER_NAME = "Archive Owner"
_STAR_RECONCILIATION_VERSION = 1
_MESSAGE_IMPORT_STAR_TYPES = frozenset(
    {"message", "conversation", "attachment", "messenger_contact"}
)
_MESSAGE_GLOBAL_STAR_TYPES = frozenset({"message", "conversation", "messenger_contact"})

THREAD_CATEGORIES = frozenset(
    {
        "archived_threads",
        "e2ee_cutover",
        "filtered_threads",
        "inbox",
        "legacy_threads",
    }
)
ARRAY_ATTACHMENT_FIELDS: tuple[tuple[str, str], ...] = (
    ("photos", "photo"),
    ("videos", "video"),
    ("audio_files", "audio"),
    ("files", "file"),
    ("gifs", "gif"),
)
MESSAGE_MODELED_FIELDS = frozenset(
    {
        "sender_name",
        "timestamp_ms",
        "content",
        "reactions",
        "photos",
        "videos",
        "audio_files",
        "files",
        "gifs",
        "sticker",
        "call_duration",
        "missed",
        "is_unsent",
        "ip",
        "payment_info",
        "transcription",
    }
)
# These names identify structured message metadata that belongs behind the advanced-details
# privacy gate.  Normalize separators and case before matching so export variants such as
# ``ipAddress`` and ``payment-information`` are handled without broad substring rules that
# would accidentally classify ordinary keys such as ``shipping`` or ``participant``.
SENSITIVE_MESSAGE_FIELD_KEYS = frozenset(
    {
        "accountcredentials",
        "bankaccount",
        "bankaccountdetails",
        "bankaccountinfo",
        "bankaccountnumber",
        "billing",
        "billingdetails",
        "billinginfo",
        "billinginformation",
        "cardcredentials",
        "carddetails",
        "cardinfo",
        "cardinformation",
        "cardnumber",
        "clientip",
        "creditcard",
        "creditcarddetails",
        "creditcardinfo",
        "cvv",
        "cvc",
        "iban",
        "ip",
        "ipaddress",
        "payment",
        "paymentdetails",
        "paymentinfo",
        "paymentinformation",
        "paymentmethod",
        "routingnumber",
        "sourceip",
    }
)
_IP_MESSAGE_FIELD_KEYS = frozenset({"clientip", "ip", "ipaddress", "sourceip"})
IMPORTER_ANOMALY_CODES = frozenset(
    {
        "invalid_message_json",
        "unsupported_message_envelope",
        "malformed_message_record",
        "duplicate_message_record",
        "missing_leading_message_chunk",
        "missing_internal_message_chunk",
        "unresolved_attachment",
        "unsafe_attachment_uri",
    }
)


@dataclass(slots=True)
class _SensitivePartition:
    public: Any
    sensitive: Any | None
    field_names: set[str]
    has_sensitive: bool


def _sensitive_message_field_kind(key: str) -> str | None:
    normalized = "".join(character for character in key.casefold() if character.isalnum())
    if normalized not in SENSITIVE_MESSAGE_FIELD_KEYS:
        return None
    return "ip" if normalized in _IP_MESSAGE_FIELD_KEYS else "payment"


def _partition_sensitive_message_value(value: Any) -> _SensitivePartition:
    if isinstance(value, Mapping):
        public: dict[str, Any] = {}
        sensitive: dict[str, Any] = {}
        field_names: set[str] = set()
        for raw_key, item in value.items():
            key = str(raw_key)
            if _sensitive_message_field_kind(key) is not None:
                sensitive[key] = item
                field_names.add(key)
                continue
            child = _partition_sensitive_message_value(item)
            public[key] = child.public
            field_names.update(child.field_names)
            if child.has_sensitive:
                sensitive[key] = child.sensitive
        return _SensitivePartition(public, sensitive or None, field_names, bool(sensitive))

    if isinstance(value, list):
        public_items: list[Any] = []
        sensitive_items: list[Any] = []
        field_names = set()
        has_sensitive = False
        for item in value:
            child = _partition_sensitive_message_value(item)
            public_items.append(child.public)
            sensitive_items.append(child.sensitive if child.has_sensitive else None)
            field_names.update(child.field_names)
            has_sensitive = has_sensitive or child.has_sensitive
        return _SensitivePartition(
            public_items,
            sensitive_items if has_sensitive else None,
            field_names,
            has_sensitive,
        )

    return _SensitivePartition(value, None, set(), False)


def partition_sensitive_message_metadata(
    metadata: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], frozenset[str]]:
    """Split structured message metadata into public and gated JSON trees."""

    partition = _partition_sensitive_message_value(metadata)
    public = partition.public if isinstance(partition.public, dict) else {}
    sensitive = partition.sensitive if isinstance(partition.sensitive, dict) else {}
    return public, sensitive, frozenset(partition.field_names)


def redact_sensitive_message_metadata(
    metadata: Mapping[str, Any],
) -> tuple[dict[str, Any], frozenset[str]]:
    """Return library-safe metadata and the structured field names that were removed."""

    public, _sensitive, field_names = partition_sensitive_message_metadata(metadata)
    return public, field_names


@dataclass(slots=True)
class MessageImportSummary:
    candidate_file_count: int = 0
    imported_file_count: int = 0
    skipped_file_count: int = 0
    invalid_file_count: int = 0
    unsupported_file_count: int = 0
    normal_container_count: int = 0
    support_container_count: int = 0
    normal_segment_count: int = 0
    support_thread_count: int = 0
    conversation_count: int = 0
    media_only_conversation_count: int = 0
    normal_message_count: int = 0
    support_message_count: int = 0
    participant_occurrence_count: int = 0
    reaction_count: int = 0
    attachment_count: int = 0
    thread_image_count: int = 0
    resolved_attachment_count: int = 0
    missing_attachment_count: int = 0
    external_attachment_count: int = 0
    cross_chunk_attachment_count: int = 0
    duplicate_message_count: int = 0
    missing_leading_segment_count: int = 0
    internal_gap_segment_count: int = 0


@dataclass(frozen=True, slots=True)
class _SourceFileRef:
    id: int
    chunk_id: int
    chunk_key: str
    chunk_path: str
    chunk_ordinal: int
    relative_path: str
    normalized_path: str
    size_bytes: int
    mtime_ns: int
    media_type: str | None
    sha256: str | None
    json_status: str
    imported_parser_version: str | None
    imported_sha256: str | None


@dataclass(frozen=True, slots=True)
class _AttachmentResolution:
    availability: str
    resolution_kind: str
    selected: _SourceFileRef | None
    candidates: tuple[_SourceFileRef, ...]
    normalized_uri: str | None
    reason: str | None = None


def _json_pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _source_record_id(source: _SourceFileRef, json_pointer: str) -> str:
    return stable_id("source-record", source.chunk_key, source.relative_path, json_pointer)


def _message_id(source: _SourceFileRef, json_pointer: str) -> str:
    return stable_id("message", source.chunk_key, source.relative_path, json_pointer)


def _attachment_id(source: _SourceFileRef, json_pointer: str) -> str:
    return stable_id("attachment", source.chunk_key, source.relative_path, json_pointer)


def _path_after_messages(relative_path: str) -> tuple[str, ...] | None:
    parts = PurePosixPath(relative_path).parts
    try:
        position = parts.index("messages")
    except ValueError:
        return None
    return parts[position + 1 :]


def _candidate_kind(relative_path: str) -> str | None:
    tail = _path_after_messages(relative_path)
    if not tail:
        return None
    if tail == ("support_messages.json",):
        return "support"
    if tail[0] in THREAD_CATEGORIES and relative_path.casefold().endswith(".json"):
        return "thread"
    return None


def _source_category(relative_path: str) -> str:
    tail = _path_after_messages(relative_path)
    if tail and tail[0] in THREAD_CATEGORIES:
        return tail[0]
    return "unknown"


def _chunk_number(relative_path: str) -> int:
    stem = PurePosixPath(relative_path).stem
    for prefix in ("message_",):
        if stem.startswith(prefix) and stem[len(prefix) :].isdigit():
            return int(stem[len(prefix) :])
    suffix = stem.rsplit("_", 1)[-1]
    return int(suffix) if suffix.isdigit() else 1


def _load_source_refs(engine: Engine) -> list[_SourceFileRef]:
    with Session(engine) as session:
        rows = session.execute(
            select(
                SourceFile.id,
                SourceFile.chunk_id,
                SourceChunk.chunk_key,
                SourceChunk.original_path,
                SourceChunk.ordinal,
                SourceFile.relative_path,
                SourceFile.normalized_path,
                SourceFile.size_bytes,
                SourceFile.mtime_ns,
                SourceFile.media_type,
                SourceFile.sha256,
                SourceFile.json_status,
                SourceFile.imported_parser_version,
                SourceFile.imported_sha256,
            )
            .join(SourceChunk, SourceChunk.id == SourceFile.chunk_id)
            .where(SourceFile.scan_status == "discovered")
            .order_by(SourceChunk.ordinal, SourceFile.relative_path, SourceFile.id)
        )
        return [_SourceFileRef(*row) for row in rows]


def _physical_source_path(
    layout: ArchiveLayout,
    source: _SourceFileRef,
    *,
    validated_blobs: set[str] | None = None,
) -> Path:
    archive_root = layout.root.resolve()
    chunk_root = (archive_root / source.chunk_path).resolve()
    if not chunk_root.is_relative_to(archive_root):
        raise ValueError(f"Source chunk escapes the archive root: {source.chunk_path}")
    candidate = (chunk_root / source.relative_path).resolve()
    if not candidate.is_relative_to(chunk_root):
        raise ValueError(f"Source file escapes its chunk: {source.relative_path}")
    if candidate.is_file():
        return candidate
    if source.sha256 is None:
        raise FileNotFoundError(candidate)
    blob_root = layout.blobs.resolve()
    blob = layout.blob_path(source.sha256).resolve()
    if not blob.is_relative_to(blob_root) or not blob.is_file():
        raise FileNotFoundError(candidate)
    if validated_blobs is None or source.sha256 not in validated_blobs:
        actual_digest = sha256_file(blob)
        if actual_digest != source.sha256:
            raise ValueError(
                f"Catalogued blob digest mismatch: expected {source.sha256}, got {actual_digest}"
            )
        if validated_blobs is not None:
            validated_blobs.add(source.sha256)
    return blob


def _current_source_digest(path: Path, source: _SourceFileRef) -> str:
    """Reuse an inventory digest only while the original stat coordinate still matches."""

    if source.sha256 is not None:
        try:
            stat = path.stat(follow_symlinks=False)
        except OSError:
            stat = None
        if (
            stat is not None
            and stat.st_size == source.size_bytes
            and stat.st_mtime_ns == source.mtime_ns
        ):
            return source.sha256
        if path.name == source.sha256 and len(source.sha256) == 64:
            # _physical_source_path already validated canonical fallback blobs.
            return source.sha256
    return sha256_file(path)


def _normal_envelope_error(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return "thread payload is not a JSON object"
    required_types: dict[str, type[object]] = {
        "title": str,
        "participants": list,
        "messages": list,
        "thread_path": str,
        "is_still_participant": bool,
        "magic_words": list,
    }
    for key, expected_type in required_types.items():
        if key not in payload:
            return f"missing required envelope field {key!r}"
        if not isinstance(payload[key], expected_type):
            return f"envelope field {key!r} has an unsupported type"
    for index, participant in enumerate(payload["participants"]):
        if not isinstance(participant, dict) or not isinstance(participant.get("name"), str):
            return f"participant {index} does not contain a string name"
    return None


def _support_envelope_error(payload: Any) -> str | None:
    if not isinstance(payload, dict) or not isinstance(payload.get("support_messages"), dict):
        return "support payload does not contain a support_messages object"
    return None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _seconds_to_ms(value: Any) -> int | None:
    parsed = _as_int(value)
    if parsed is None:
        return None
    return parsed if abs(parsed) >= 100_000_000_000 else parsed * 1000


def _optional_text(value: Any) -> NormalizedText | None:
    return normalize_text(value) if isinstance(value, str) else None


def _conversation_type(participants: Sequence[Mapping[str, Any]]) -> str:
    count = len(participants)
    if count == 2:
        return "direct"
    if count > 2:
        return "group"
    if count == 1:
        return "solo"
    return "unknown"


def _direct_counterpart(
    participants: Sequence[Mapping[str, Any]],
    *,
    owner_display: str,
    fallback_title: NormalizedText,
) -> NormalizedText:
    names = [normalize_text(str(participant["name"])) for participant in participants]
    non_owner = [name for name in names if name.display != owner_display]
    if len(non_owner) == 1:
        return non_owner[0]
    # Some legacy two-person envelopes omit the owner (and a few list the owner twice), so the
    # source title is the only available provisional counterpart label in those cases.
    return fallback_title


def _conversation_key(
    *, conversation_type: str, conversation_title: NormalizedText, thread_path: str
) -> str:
    if conversation_type == "direct":
        # This is deliberately exact-display grouping. Case, spelling, and repaired display
        # differences remain separate until a user explicitly merges them.
        return f"direct:{conversation_title.display}"
    return f"segment:{thread_path}"


def _ensure_conversation(
    session: Session,
    *,
    conversation_key: str,
    title: NormalizedText,
    conversation_type: str,
    source_category: str,
) -> Conversation:
    conversation_id = stable_id("conversation", conversation_key)
    row = session.get(Conversation, conversation_id)
    if row is None:
        row = Conversation(
            id=conversation_id,
            conversation_key=conversation_key,
            title_raw=title.raw,
            title_display=title.display,
            title_search=title.search,
            conversation_type=conversation_type,
            primary_category=source_category,
        )
        session.add(row)
    else:
        row.title_raw = title.raw
        row.title_display = title.display
        row.title_search = title.search
        row.conversation_type = conversation_type
    return row


def _merge_container_metadata(
    metadata: Mapping[str, Any],
    *,
    source: _SourceFileRef,
    chunk_number: int,
    envelope: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(metadata)
    containers_value = result.get("containers", {})
    containers = dict(containers_value) if isinstance(containers_value, dict) else {}
    containers[str(source.id)] = {
        "chunk_number": chunk_number,
        "source_chunk_id": source.chunk_id,
        "source_chunk_key": source.chunk_key,
        "relative_path": source.relative_path,
    }
    result["containers"] = containers
    if "joinable_mode" in envelope:
        result["joinable_mode"] = envelope["joinable_mode"]
    if "magic_words" in envelope:
        result["magic_words"] = envelope["magic_words"]
    return result


def _ensure_segment(
    session: Session,
    *,
    conversation: Conversation,
    source: _SourceFileRef,
    thread_path: str,
    title: NormalizedText,
    source_category: str,
    chunk_number: int,
    is_still_participant: bool | None,
    envelope: Mapping[str, Any],
) -> ConversationSegment:
    segment_id = stable_id("conversation-segment", thread_path)
    row = session.get(ConversationSegment, segment_id)
    if row is None:
        row = ConversationSegment(
            id=segment_id,
            conversation_id=conversation.id,
            thread_path=thread_path,
            title_raw=title.raw,
            title_display=title.display,
            title_search=title.search,
            source_category=source_category,
            is_still_participant=is_still_participant,
            metadata_json={},
        )
        session.add(row)
    override = session.get(ConversationSegmentOverride, segment_id)
    assigned_conversation_id = conversation.id
    if override is not None:
        corrected_conversation = session.get(Conversation, override.conversation_id)
        if corrected_conversation is None or corrected_conversation.deleted_at is not None:
            raise RuntimeError(f"Conversation segment override has no active target: {segment_id}")
        assigned_conversation_id = corrected_conversation.id
    row.conversation_id = assigned_conversation_id
    row.title_raw = title.raw
    row.title_display = title.display
    row.title_search = title.search
    row.source_category = source_category
    row.is_still_participant = is_still_participant
    row.metadata_json = _merge_container_metadata(
        row.metadata_json,
        source=source,
        chunk_number=chunk_number,
        envelope=envelope,
    )
    return row


def _replace_segment_participants(
    session: Session,
    *,
    segment_id: str,
    participants: Sequence[Mapping[str, Any]],
    owner_display: str,
) -> None:
    session.execute(delete(SegmentParticipant).where(SegmentParticipant.segment_id == segment_id))
    for ordinal, participant in enumerate(participants):
        name = normalize_text(str(participant["name"]))
        participant_id = stable_id("segment-participant", segment_id, ordinal)
        row = session.get(Participant, participant_id)
        if row is None:
            row = Participant(
                id=participant_id,
                name_raw=name.raw,
                name_display=name.display,
                name_search=name.search,
                is_owner=name.display == owner_display,
            )
            session.add(row)
        else:
            row.name_raw = name.raw
            row.name_display = name.display
            row.name_search = name.search
            row.is_owner = name.display == owner_display
        session.add(
            SegmentParticipant(
                segment_id=segment_id,
                participant_id=participant_id,
                ordinal=ordinal,
            )
        )


def _safe_local_uri(uri: str) -> tuple[str | None, str | None]:
    if not uri:
        return None, "empty URI"
    parsed = urlsplit(uri)
    if parsed.scheme or parsed.netloc:
        return None, "external URI"
    decoded = unquote(parsed.path).replace("\\", "/")
    while decoded.startswith("./"):
        decoded = decoded[2:]
    if not decoded or decoded.startswith("/") or "\x00" in decoded:
        return None, "absolute, empty, or NUL-containing URI"
    parts = PurePosixPath(decoded).parts
    if any(part in {"", ".", ".."} for part in parts):
        return None, "URI contains an unsafe path component"
    if parts and len(parts[0]) == 2 and parts[0][1] == ":":
        return None, "URI contains a drive-qualified path"
    return PurePosixPath(*parts).as_posix(), None


def _build_attachment_index(
    layout: ArchiveLayout,
    sources: Iterable[_SourceFileRef],
    *,
    validated_blobs: set[str] | None = None,
) -> dict[str, tuple[_SourceFileRef, ...]]:
    grouped: defaultdict[str, list[_SourceFileRef]] = defaultdict(list)
    for source in sources:
        normalized, error = _safe_local_uri(source.normalized_path)
        if error is not None or normalized is None:
            continue
        try:
            _physical_source_path(layout, source, validated_blobs=validated_blobs)
        except (OSError, ValueError):
            continue
        grouped[normalized].append(source)
    return {
        key: tuple(sorted(values, key=lambda item: (item.chunk_ordinal, item.id)))
        for key, values in grouped.items()
    }


def _resolve_attachment(
    uri: Any,
    *,
    source: _SourceFileRef,
    attachment_index: Mapping[str, tuple[_SourceFileRef, ...]],
) -> _AttachmentResolution:
    if not isinstance(uri, str):
        return _AttachmentResolution("missing", "missing", None, (), None, "URI is not text")
    parsed = urlsplit(uri)
    if parsed.scheme.casefold() in {"http", "https"} and parsed.netloc:
        return _AttachmentResolution("external", "external", None, (), None)
    normalized, error = _safe_local_uri(uri)
    if normalized is None:
        resolution_kind = "unsafe" if error != "external URI" else "external"
        availability = "external" if resolution_kind == "external" else "missing"
        return _AttachmentResolution(availability, resolution_kind, None, (), None, error)
    candidates = attachment_index.get(normalized, ())
    if not candidates:
        return _AttachmentResolution("missing", "missing", None, (), normalized)
    same_chunk = tuple(item for item in candidates if item.chunk_id == source.chunk_id)
    selected = same_chunk[0] if same_chunk else candidates[0]
    resolution_kind = "local" if same_chunk else "cross_chunk"
    return _AttachmentResolution("resolved", resolution_kind, selected, candidates, normalized)


def _attachment_filename(uri: Any) -> str | None:
    if not isinstance(uri, str) or not uri:
        return None
    path = unquote(urlsplit(uri).path).replace("\\", "/")
    name = PurePosixPath(path).name
    return name or None


def _record_anomaly(
    session: Session,
    *,
    anomaly_id: str,
    code: str,
    severity: str,
    summary: str,
    detail: Mapping[str, Any],
    source_file_id: int | None = None,
    source_record_id: str | None = None,
    segment_id: str | None = None,
) -> None:
    session.execute(
        sqlite_insert(IntegrityAnomaly)
        .values(
            id=anomaly_id,
            code=code,
            severity=severity,
            summary=summary,
            detail_json=dict(detail),
            source_file_id=source_file_id,
            source_record_id=source_record_id,
            conversation_segment_id=segment_id,
            resolved=False,
        )
        .on_conflict_do_update(
            index_elements=[IntegrityAnomaly.id],
            set_={
                "code": code,
                "severity": severity,
                "summary": summary,
                "detail_json": dict(detail),
                "source_file_id": source_file_id,
                "source_record_id": source_record_id,
                "conversation_segment_id": segment_id,
                "resolved": False,
            },
        )
    )


def _add_attachment(
    session: Session,
    *,
    source: _SourceFileRef,
    conversation_id: str,
    message_id: str | None,
    source_record_id: str | None,
    json_pointer: str,
    kind: str,
    ordinal: int,
    raw_attachment: Any,
    fallback_timestamp_ms: int | None,
    attachment_index: Mapping[str, tuple[_SourceFileRef, ...]],
    candidate_rows: list[AttachmentCandidate],
) -> None:
    if isinstance(raw_attachment, dict):
        uri = raw_attachment.get("uri")
        if uri is None:
            uri = raw_attachment.get("url")
        timestamp_ms = _seconds_to_ms(raw_attachment.get("creation_timestamp"))
        metadata = {
            key: value
            for key, value in raw_attachment.items()
            if key not in {"uri", "url", "creation_timestamp"}
        }
    else:
        uri = raw_attachment if isinstance(raw_attachment, str) else None
        timestamp_ms = None
        metadata = {"raw_attachment": raw_attachment}

    resolution = _resolve_attachment(
        uri,
        source=source,
        attachment_index=attachment_index,
    )
    selected = resolution.selected
    metadata["resolution_kind"] = resolution.resolution_kind
    if resolution.normalized_uri is not None:
        metadata["normalized_uri"] = resolution.normalized_uri
    if resolution.reason is not None:
        metadata["resolution_reason"] = resolution.reason
    attachment_id = _attachment_id(source, json_pointer)
    session.add(
        AttachmentOccurrence(
            id=attachment_id,
            message_id=message_id,
            conversation_id=conversation_id,
            source_file_id=source.id,
            resolved_source_file_id=selected.id if selected else None,
            blob_sha256=selected.sha256 if selected else None,
            kind=kind,
            ordinal=ordinal,
            source_uri=uri if isinstance(uri, str) else None,
            filename=_attachment_filename(uri),
            media_type=selected.media_type if selected else None,
            size_bytes=selected.size_bytes if selected else None,
            timestamp_ms=timestamp_ms if timestamp_ms is not None else fallback_timestamp_ms,
            availability=resolution.availability,
            metadata_json=metadata,
        )
    )
    for candidate_ordinal, candidate in enumerate(resolution.candidates):
        candidate_rows.append(
            AttachmentCandidate(
                attachment_id=attachment_id,
                source_file_id=candidate.id,
                ordinal=candidate_ordinal,
                match_kind="same_chunk" if candidate.chunk_id == source.chunk_id else "cross_chunk",
                selected=selected is not None and candidate.id == selected.id,
            )
        )
    if resolution.availability == "missing":
        code = (
            "unsafe_attachment_uri"
            if resolution.resolution_kind == "unsafe"
            else "unresolved_attachment"
        )
        _record_anomaly(
            session,
            anomaly_id=stable_id(code, attachment_id),
            code=code,
            severity="warning",
            summary=(
                "Attachment URI was rejected as unsafe."
                if code == "unsafe_attachment_uri"
                else "Attachment URI does not resolve to a catalogued source file."
            ),
            detail={
                "attachment_id": attachment_id,
                "source_uri": uri if isinstance(uri, str) else None,
                "resolution_reason": resolution.reason,
            },
            source_file_id=source.id,
            source_record_id=None,
        )


def _message_kind(payload: Mapping[str, Any], *, has_attachments: bool) -> str:
    if payload.get("is_unsent") is True:
        return "unsent"
    if _as_int(payload.get("call_duration")) is not None or "missed" in payload:
        return "call"
    if "payment_info" in payload:
        return "payment"
    if has_attachments:
        return "attachment"
    if "share" in payload:
        return "share"
    if isinstance(payload.get("content"), str):
        return "text"
    if "transcription" in payload:
        return "transcription"
    return "unavailable"


def _has_attachment_payload(payload: Mapping[str, Any]) -> bool:
    if isinstance(payload.get("sticker"), dict):
        return True
    return any(
        isinstance(payload.get(field), list) and bool(payload[field])
        for field, _ in ARRAY_ATTACHMENT_FIELDS
    )


def _transcription_text(payload: Mapping[str, Any]) -> NormalizedText | None:
    transcription = payload.get("transcription")
    if isinstance(transcription, dict):
        return _optional_text(transcription.get("text"))
    return _optional_text(transcription)


def _message_has_required_coordinate_fields(payload: Mapping[str, Any], *, support: bool) -> bool:
    sender_key = "from" if support else "sender_name"
    sender = payload.get(sender_key)
    timestamp = (
        _seconds_to_ms(payload.get("timestamp"))
        if support
        else _as_int(payload.get("timestamp_ms"))
    )
    return isinstance(sender, str) and timestamp is not None


def _prepare_message_source_records(
    session: Session,
    *,
    run_id: str,
    source: _SourceFileRef,
    messages: Sequence[Any],
    pointer_prefix: str,
    support: bool,
) -> None:
    """Insert a file's source coordinates before dependent Message rows.

    The models intentionally do not expose ORM relationships. Flushing this compact parent batch
    once per envelope avoids both ambiguous unit-of-work ordering and a flush per message.
    """

    for source_array_index, raw_message in enumerate(messages):
        if is_purge_tombstone(raw_message):
            continue
        if not isinstance(raw_message, dict) or not _message_has_required_coordinate_fields(
            raw_message, support=support
        ):
            continue
        json_pointer = f"{pointer_prefix}/{source_array_index}"
        session.add(
            SourceRecord(
                id=_source_record_id(source, json_pointer),
                source_file_id=source.id,
                json_pointer=json_pointer,
                record_type="support_message" if support else "message",
                logical_fingerprint=canonical_fingerprint(
                    "support-message" if support else "facebook-message", raw_message
                ),
                import_run_id=run_id,
            )
        )
    session.flush()


def _import_reactions(
    session: Session,
    *,
    message_id: str,
    reactions: Any,
) -> None:
    if not isinstance(reactions, list):
        return
    for ordinal, reaction in enumerate(reactions):
        if not isinstance(reaction, dict):
            continue
        actor = _optional_text(reaction.get("actor"))
        value = _optional_text(reaction.get("reaction"))
        if actor is None or value is None:
            continue
        session.add(
            MessageReaction(
                id=stable_id("message-reaction", message_id, ordinal),
                message_id=message_id,
                actor_raw=actor.raw,
                actor_display=actor.display,
                reaction_raw=value.raw,
                reaction_display=value.display,
                ordinal=ordinal,
                timestamp_ms=_seconds_to_ms(reaction.get("timestamp")),
                extra_json={
                    key: item
                    for key, item in reaction.items()
                    if key not in {"actor", "reaction", "timestamp"}
                },
            )
        )


def _import_message_attachments(
    session: Session,
    *,
    source: _SourceFileRef,
    conversation_id: str,
    message_id: str,
    source_record_id: str,
    message_pointer: str,
    timestamp_ms: int,
    payload: Mapping[str, Any],
    attachment_index: Mapping[str, tuple[_SourceFileRef, ...]],
    candidate_rows: list[AttachmentCandidate],
) -> None:
    for field, kind in ARRAY_ATTACHMENT_FIELDS:
        values = payload.get(field)
        if not isinstance(values, list):
            continue
        for ordinal, attachment in enumerate(values):
            _add_attachment(
                session,
                source=source,
                conversation_id=conversation_id,
                message_id=message_id,
                source_record_id=source_record_id,
                json_pointer=f"{message_pointer}/{field}/{ordinal}",
                kind=kind,
                ordinal=ordinal,
                raw_attachment=attachment,
                fallback_timestamp_ms=timestamp_ms,
                attachment_index=attachment_index,
                candidate_rows=candidate_rows,
            )
    sticker = payload.get("sticker")
    if isinstance(sticker, dict):
        _add_attachment(
            session,
            source=source,
            conversation_id=conversation_id,
            message_id=message_id,
            source_record_id=source_record_id,
            json_pointer=f"{message_pointer}/sticker",
            kind="sticker",
            ordinal=0,
            raw_attachment=sticker,
            fallback_timestamp_ms=timestamp_ms,
            attachment_index=attachment_index,
            candidate_rows=candidate_rows,
        )


def _add_message(
    session: Session,
    *,
    source: _SourceFileRef,
    conversation_id: str,
    segment_id: str,
    chunk_number: int,
    source_array_index: int,
    json_pointer: str,
    payload: Mapping[str, Any],
    owner_display: str,
    support: bool = False,
) -> bool:
    sender_key = "from" if support else "sender_name"
    content_key = "message" if support else "content"
    sender = _optional_text(payload.get(sender_key))
    timestamp_value = (
        _seconds_to_ms(payload.get("timestamp"))
        if support
        else _as_int(payload.get("timestamp_ms"))
    )
    if sender is None or timestamp_value is None:
        _record_anomaly(
            session,
            anomaly_id=stable_id("malformed-message", source.id, json_pointer),
            code="malformed_message_record",
            severity="warning",
            summary="A message record is missing a usable sender or timestamp.",
            detail={"json_pointer": json_pointer, "support": support},
            source_file_id=source.id,
        )
        return False

    content = _optional_text(payload.get(content_key))
    transcription = _transcription_text(payload)
    source_record_id = _source_record_id(source, json_pointer)
    message_id = _message_id(source, json_pointer)
    fingerprint = canonical_fingerprint(
        "support-message" if support else "facebook-message", payload
    )
    has_attachments = (
        bool(payload.get("attachments")) if support else _has_attachment_payload(payload)
    )
    if support:
        metadata_json = {
            key: value
            for key, value in payload.items()
            if key not in {"from", "message", "timestamp", "attachments"}
        }
        message_kind = "attachment" if has_attachments else "text" if content else "unavailable"
    else:
        metadata_json = {
            key: value
            for key, value in payload.items()
            if key not in MESSAGE_MODELED_FIELDS or _sensitive_message_field_kind(key) is not None
        }
        message_kind = _message_kind(payload, has_attachments=has_attachments)
    extra_json, sensitive_json, sensitive_field_names = partition_sensitive_message_metadata(
        metadata_json
    )
    sensitive_field_kinds = {
        kind
        for key in sensitive_field_names
        if (kind := _sensitive_message_field_kind(key)) is not None
    }

    session.add(
        Message(
            id=message_id,
            source_record_id=source_record_id,
            source_file_id=source.id,
            json_pointer=json_pointer,
            chunk_number=chunk_number,
            source_array_index=source_array_index,
            conversation_id=conversation_id,
            segment_id=segment_id,
            timestamp_ms=timestamp_value,
            sender_raw=sender.raw,
            sender_display=sender.display,
            sender_search=sender.search,
            sender_is_owner=sender.display == owner_display,
            content_raw=content.raw if content else None,
            content_display=content.display if content else None,
            content_search=content.search if content else None,
            message_kind=message_kind,
            call_duration_seconds=_as_int(payload.get("call_duration")),
            is_missed_call=(
                payload.get("missed") if isinstance(payload.get("missed"), bool) else None
            ),
            has_payment="payment" in sensitive_field_kinds,
            has_sensitive_ip="ip" in sensitive_field_kinds,
            has_transcription=transcription is not None,
            transcription_raw=transcription.raw if transcription else None,
            transcription_display=transcription.display if transcription else None,
            transcription_search=transcription.search if transcription else None,
            is_unsent=payload.get("is_unsent") is True,
            is_unavailable_body=message_kind == "unavailable",
            logical_fingerprint=fingerprint,
            extra_json=extra_json,
            sensitive_json=sensitive_json,
        )
    )
    return True


def _import_message_dependents(
    session: Session,
    *,
    source: _SourceFileRef,
    conversation_id: str,
    json_pointer: str,
    payload: Mapping[str, Any],
    attachment_index: Mapping[str, tuple[_SourceFileRef, ...]],
    candidate_rows: list[AttachmentCandidate],
    support: bool = False,
) -> None:
    message_id = _message_id(source, json_pointer)
    source_record_id = _source_record_id(source, json_pointer)
    timestamp_value = (
        _seconds_to_ms(payload.get("timestamp"))
        if support
        else _as_int(payload.get("timestamp_ms"))
    )
    assert timestamp_value is not None
    if not support:
        _import_reactions(session, message_id=message_id, reactions=payload.get("reactions"))
        _import_message_attachments(
            session,
            source=source,
            conversation_id=conversation_id,
            message_id=message_id,
            source_record_id=source_record_id,
            message_pointer=json_pointer,
            timestamp_ms=timestamp_value,
            payload=payload,
            attachment_index=attachment_index,
            candidate_rows=candidate_rows,
        )
        return
    attachments = payload.get("attachments")
    if isinstance(attachments, list):
        for ordinal, attachment in enumerate(attachments):
            _add_attachment(
                session,
                source=source,
                conversation_id=conversation_id,
                message_id=message_id,
                source_record_id=source_record_id,
                json_pointer=f"{json_pointer}/attachments/{ordinal}",
                kind="support_file",
                ordinal=ordinal,
                raw_attachment=attachment,
                fallback_timestamp_ms=timestamp_value,
                attachment_index=attachment_index,
                candidate_rows=candidate_rows,
            )


def _delete_previous_file_import(session: Session, source_file_id: int) -> None:
    session.execute(delete(Message).where(Message.source_file_id == source_file_id))
    session.execute(
        delete(AttachmentOccurrence)
        .where(AttachmentOccurrence.source_file_id == source_file_id)
        .where(AttachmentOccurrence.message_id.is_(None))
        .where(AttachmentOccurrence.activity_record_id.is_(None))
    )
    session.execute(
        delete(SourceRecord)
        .where(SourceRecord.source_file_id == source_file_id)
        .where(SourceRecord.record_type.in_(("message", "support_message")))
    )


def _detach_source_container(session: Session, source_file_id: int) -> None:
    source_key = str(source_file_id)
    json_path = f'$.containers."{source_key}"'
    for segment in session.scalars(
        select(ConversationSegment).where(
            func.json_type(ConversationSegment.metadata_json, json_path).is_not(None)
        )
    ).all():
        containers_value = segment.metadata_json.get("containers")
        if not isinstance(containers_value, dict) or source_key not in containers_value:
            continue
        containers = dict(containers_value)
        containers.pop(source_key, None)
        metadata = dict(segment.metadata_json)
        metadata["containers"] = containers
        segment.metadata_json = metadata


def _import_thread_file(
    session: Session,
    *,
    run_id: str,
    source: _SourceFileRef,
    payload: Mapping[str, Any],
    owner_display: str,
    attachment_index: Mapping[str, tuple[_SourceFileRef, ...]],
) -> None:
    participants = payload["participants"]
    assert isinstance(participants, list)
    participant_maps = [item for item in participants if isinstance(item, dict)]
    title = normalize_text(str(payload["title"]))
    thread_path = str(payload["thread_path"])
    conversation_type = _conversation_type(participant_maps)
    conversation_title = (
        _direct_counterpart(
            participant_maps,
            owner_display=owner_display,
            fallback_title=title,
        )
        if conversation_type == "direct"
        else title
    )
    source_category = _source_category(source.relative_path)
    chunk_number = _chunk_number(source.relative_path)
    key = _conversation_key(
        conversation_type=conversation_type,
        conversation_title=conversation_title,
        thread_path=thread_path,
    )
    conversation = _ensure_conversation(
        session,
        conversation_key=key,
        title=conversation_title,
        conversation_type=conversation_type,
        source_category=source_category,
    )
    segment = _ensure_segment(
        session,
        conversation=conversation,
        source=source,
        thread_path=thread_path,
        title=title,
        source_category=source_category,
        chunk_number=chunk_number,
        is_still_participant=bool(payload["is_still_participant"]),
        envelope=payload,
    )
    assigned_conversation_id = segment.conversation_id
    _replace_segment_participants(
        session,
        segment_id=segment.id,
        participants=participant_maps,
        owner_display=owner_display,
    )
    messages = payload["messages"]
    assert isinstance(messages, list)
    session.flush()
    _prepare_message_source_records(
        session,
        run_id=run_id,
        source=source,
        messages=messages,
        pointer_prefix="/messages",
        support=False,
    )
    candidate_rows: list[AttachmentCandidate] = []
    for source_array_index, raw_message in enumerate(messages):
        pointer = f"/messages/{source_array_index}"
        if is_purge_tombstone(raw_message):
            continue
        if not isinstance(raw_message, dict):
            _record_anomaly(
                session,
                anomaly_id=stable_id("malformed-message", source.id, pointer),
                code="malformed_message_record",
                severity="warning",
                summary="A message array member is not a JSON object.",
                detail={"json_pointer": pointer},
                source_file_id=source.id,
            )
            continue
        _add_message(
            session,
            source=source,
            conversation_id=assigned_conversation_id,
            segment_id=segment.id,
            chunk_number=chunk_number,
            source_array_index=source_array_index,
            json_pointer=pointer,
            payload=raw_message,
            owner_display=owner_display,
        )
    session.flush()
    for source_array_index, raw_message in enumerate(messages):
        if is_purge_tombstone(raw_message):
            continue
        if not isinstance(raw_message, dict) or not _message_has_required_coordinate_fields(
            raw_message, support=False
        ):
            continue
        pointer = f"/messages/{source_array_index}"
        _import_message_dependents(
            session,
            source=source,
            conversation_id=assigned_conversation_id,
            json_pointer=pointer,
            payload=raw_message,
            attachment_index=attachment_index,
            candidate_rows=candidate_rows,
        )

    image = payload.get("image")
    if isinstance(image, dict):
        _add_attachment(
            session,
            source=source,
            conversation_id=assigned_conversation_id,
            message_id=None,
            source_record_id=None,
            json_pointer="/image",
            kind="thread_image",
            ordinal=0,
            raw_attachment=image,
            fallback_timestamp_ms=None,
            attachment_index=attachment_index,
            candidate_rows=candidate_rows,
        )
    session.flush()
    session.add_all(candidate_rows)


def _support_participants(messages: Sequence[Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_message in messages:
        if not isinstance(raw_message, dict):
            continue
        for key in ("from", "to"):
            value = raw_message.get(key)
            if isinstance(value, str) and value not in seen:
                seen.add(value)
                result.append({"name": value})
    return result


def _import_support_file(
    session: Session,
    *,
    run_id: str,
    source: _SourceFileRef,
    payload: Mapping[str, Any],
    owner_display: str,
    attachment_index: Mapping[str, tuple[_SourceFileRef, ...]],
) -> None:
    support_threads = payload["support_messages"]
    assert isinstance(support_threads, dict)
    candidate_rows: list[AttachmentCandidate] = []
    for thread_ordinal, (opaque_key, raw_thread) in enumerate(support_threads.items()):
        if is_purge_tombstone(raw_thread):
            continue
        if not isinstance(raw_thread, dict) or not isinstance(raw_thread.get("messages"), list):
            pointer = f"/support_messages/{_json_pointer_token(str(opaque_key))}"
            _record_anomaly(
                session,
                anomaly_id=stable_id("malformed-support-thread", source.id, pointer),
                code="unsupported_message_envelope",
                severity="warning",
                summary="A support thread does not contain a message array.",
                detail={"json_pointer": pointer},
                source_file_id=source.id,
            )
            continue
        messages = raw_thread["messages"]
        subject_value = raw_thread.get("subject")
        subject = normalize_text(subject_value if isinstance(subject_value, str) else "Support")
        thread_path = f"support/{opaque_key}"
        conversation_key = f"support:{opaque_key}"
        conversation = _ensure_conversation(
            session,
            conversation_key=conversation_key,
            title=subject,
            conversation_type="system",
            source_category="support",
        )
        support_envelope: dict[str, Any] = {
            "support_timestamp": raw_thread.get("timestamp"),
            "thread_ordinal": thread_ordinal,
        }
        segment = _ensure_segment(
            session,
            conversation=conversation,
            source=source,
            thread_path=thread_path,
            title=subject,
            source_category="support",
            chunk_number=1,
            is_still_participant=None,
            envelope=support_envelope,
        )
        assigned_conversation_id = segment.conversation_id
        _replace_segment_participants(
            session,
            segment_id=segment.id,
            participants=_support_participants(messages),
            owner_display=owner_display,
        )
        pointer_prefix = f"/support_messages/{_json_pointer_token(str(opaque_key))}/messages"
        session.flush()
        _prepare_message_source_records(
            session,
            run_id=run_id,
            source=source,
            messages=messages,
            pointer_prefix=pointer_prefix,
            support=True,
        )
        for source_array_index, raw_message in enumerate(messages):
            pointer = f"{pointer_prefix}/{source_array_index}"
            if is_purge_tombstone(raw_message):
                continue
            if not isinstance(raw_message, dict):
                _record_anomaly(
                    session,
                    anomaly_id=stable_id("malformed-message", source.id, pointer),
                    code="malformed_message_record",
                    severity="warning",
                    summary="A support message array member is not a JSON object.",
                    detail={"json_pointer": pointer},
                    source_file_id=source.id,
                )
                continue
            _add_message(
                session,
                source=source,
                conversation_id=assigned_conversation_id,
                segment_id=segment.id,
                chunk_number=1,
                source_array_index=source_array_index,
                json_pointer=pointer,
                payload=raw_message,
                owner_display=owner_display,
                support=True,
            )
        session.flush()
        for source_array_index, raw_message in enumerate(messages):
            if is_purge_tombstone(raw_message):
                continue
            if not isinstance(raw_message, dict) or not _message_has_required_coordinate_fields(
                raw_message, support=True
            ):
                continue
            pointer = f"{pointer_prefix}/{source_array_index}"
            _import_message_dependents(
                session,
                source=source,
                conversation_id=assigned_conversation_id,
                json_pointer=pointer,
                payload=raw_message,
                attachment_index=attachment_index,
                candidate_rows=candidate_rows,
                support=True,
            )
    session.flush()
    session.add_all(candidate_rows)


def _mark_file_problem(
    engine: Engine,
    *,
    source: _SourceFileRef,
    status: str,
    code: str,
    message: str,
) -> None:
    with Session(engine) as session, session.begin():
        # The catalog must never present rows from an older byte sequence as if they were backed
        # by the newly inventoried invalid/unsupported source. Remove that file's last normalized
        # representation while preserving other files in the same segment/conversation.
        _resolve_previous_file_anomalies(session, source.id)
        _delete_previous_file_import(session, source.id)
        _detach_source_container(session, source.id)
        session.execute(
            update(SourceFile)
            .where(SourceFile.id == source.id)
            .values(json_status=status, json_error=message)
        )
        _record_anomaly(
            session,
            anomaly_id=stable_id(code, source.id),
            code=code,
            severity="warning",
            summary=message,
            detail={"relative_path": source.relative_path},
            source_file_id=source.id,
        )


def _remove_empty_segments(session: Session) -> None:
    message_segment_ids = set(session.scalars(select(Message.segment_id).distinct()))
    for segment in session.scalars(select(ConversationSegment)).all():
        containers = segment.metadata_json.get("containers")
        has_containers = isinstance(containers, dict) and bool(containers)
        if segment.id not in message_segment_ids and not has_containers:
            session.delete(segment)


def _resolve_previous_file_anomalies(session: Session, source_file_id: int) -> None:
    session.execute(
        update(IntegrityAnomaly)
        .where(IntegrityAnomaly.source_file_id == source_file_id)
        .where(IntegrityAnomaly.code.in_(IMPORTER_ANOMALY_CODES))
        .values(resolved=True)
    )


def _reconcile_duplicate_messages(session: Session) -> int:
    session.execute(update(Message).values(duplicate_of_id=None))
    session.execute(
        update(SourceRecord)
        .where(SourceRecord.record_type.in_(("message", "support_message")))
        .values(duplicate_of_id=None)
    )
    session.execute(
        update(IntegrityAnomaly)
        .where(IntegrityAnomaly.code == "duplicate_message_record")
        .values(resolved=True)
    )
    duplicate_keys = session.execute(
        select(Message.segment_id, Message.logical_fingerprint)
        .group_by(Message.segment_id, Message.logical_fingerprint)
        .having(func.count(Message.pk) > 1)
    ).all()
    duplicate_count = 0
    for segment_id, fingerprint in duplicate_keys:
        rows = session.execute(
            select(
                Message.id,
                Message.source_record_id,
                Message.source_file_id,
                Message.json_pointer,
            )
            .where(Message.segment_id == segment_id)
            .where(Message.logical_fingerprint == fingerprint)
            .order_by(Message.source_file_id, Message.json_pointer, Message.id)
        ).all()
        canonical_message_id, canonical_record_id, _source_file_id, _pointer = rows[0]
        for message_id, source_record_id, source_file_id, pointer in rows[1:]:
            duplicate_count += 1
            session.execute(
                update(Message)
                .where(Message.id == message_id)
                .values(duplicate_of_id=canonical_message_id)
            )
            session.execute(
                update(SourceRecord)
                .where(SourceRecord.id == source_record_id)
                .values(duplicate_of_id=canonical_record_id)
            )
            _record_anomaly(
                session,
                anomaly_id=stable_id("duplicate-message", source_record_id),
                code="duplicate_message_record",
                severity="info",
                summary="This source occurrence duplicates an earlier message record.",
                detail={
                    "canonical_message_id": canonical_message_id,
                    "duplicate_message_id": message_id,
                    "json_pointer": pointer,
                },
                source_file_id=source_file_id,
                source_record_id=source_record_id,
            )
    return duplicate_count


def _container_chunk_numbers(segment: ConversationSegment) -> list[int]:
    containers = segment.metadata_json.get("containers")
    if not isinstance(containers, dict):
        return []
    numbers: set[int] = set()
    for value in containers.values():
        if isinstance(value, dict):
            parsed = _as_int(value.get("chunk_number"))
            if parsed is not None and parsed > 0:
                numbers.add(parsed)
    return sorted(numbers)


def _reconcile_segments(session: Session) -> tuple[int, int]:
    session.execute(
        update(IntegrityAnomaly)
        .where(
            IntegrityAnomaly.code.in_(
                ("missing_leading_message_chunk", "missing_internal_message_chunk")
            )
        )
        .values(resolved=True)
    )
    for segment in session.scalars(select(ConversationSegment)).all():
        containers = segment.metadata_json.get("containers")
        if not isinstance(containers, dict) or not containers:
            session.delete(segment)
    session.flush()
    message_stats = {
        segment_id: (count, first_ms, last_ms)
        for segment_id, count, first_ms, last_ms in session.execute(
            select(
                Message.segment_id,
                func.count(Message.pk),
                func.min(Message.timestamp_ms),
                func.max(Message.timestamp_ms),
            ).group_by(Message.segment_id)
        )
    }
    leading_count = 0
    internal_count = 0
    for segment in session.scalars(select(ConversationSegment)).all():
        _count, first_ms, last_ms = message_stats.get(segment.id, (0, None, None))
        segment.first_activity_ms = first_ms
        segment.last_activity_ms = last_ms
        chunk_numbers = _container_chunk_numbers(segment)
        segment.source_chunk_count = len(chunk_numbers)
        missing_leading: list[int] = []
        missing_internal: list[int] = []
        if segment.source_category != "support" and chunk_numbers:
            minimum = chunk_numbers[0]
            maximum = chunk_numbers[-1]
            if minimum > 1:
                missing_leading = list(range(1, minimum))
            missing_internal = [
                number for number in range(minimum + 1, maximum) if number not in chunk_numbers
            ]
        if missing_leading:
            leading_count += 1
            _record_anomaly(
                session,
                anomaly_id=stable_id("missing-leading-chunk", segment.id),
                code="missing_leading_message_chunk",
                severity="warning",
                summary="Conversation export starts after message_1.json.",
                detail={"missing_chunk_numbers": missing_leading, "present": chunk_numbers},
                segment_id=segment.id,
            )
        if missing_internal:
            internal_count += 1
            _record_anomaly(
                session,
                anomaly_id=stable_id("missing-internal-chunk", segment.id),
                code="missing_internal_message_chunk",
                severity="warning",
                summary="Conversation export has a gap between message chunks.",
                detail={"missing_chunk_numbers": missing_internal, "present": chunk_numbers},
                segment_id=segment.id,
            )
        segment.completeness = "incomplete" if missing_leading or missing_internal else "complete"
    return leading_count, internal_count


def _reconcile_conversation_aggregates(session: Session) -> None:
    participant_counts = {
        segment_id: count
        for segment_id, count in session.execute(
            select(SegmentParticipant.segment_id, func.count(SegmentParticipant.id)).group_by(
                SegmentParticipant.segment_id
            )
        )
    }
    message_stats = {
        conversation_id: (count, sent, first_ms, last_ms, active_days)
        for conversation_id, count, sent, first_ms, last_ms, active_days in session.execute(
            select(
                Message.conversation_id,
                func.count(Message.pk),
                func.sum(func.iif(Message.sender_is_owner.is_(True), 1, 0)),
                func.min(Message.timestamp_ms),
                func.max(Message.timestamp_ms),
                func.count(func.distinct(func.date(Message.timestamp_ms / 1000.0, "unixepoch"))),
            ).group_by(Message.conversation_id)
        )
    }
    attachment_stats = {
        conversation_id: (count, unique_count, byte_count)
        for conversation_id, count, unique_count, byte_count in session.execute(
            select(
                AttachmentOccurrence.conversation_id,
                func.count(AttachmentOccurrence.id),
                func.count(func.distinct(AttachmentOccurrence.blob_sha256)),
                func.coalesce(func.sum(AttachmentOccurrence.size_bytes), 0),
            )
            .where(AttachmentOccurrence.deleted_at.is_(None))
            .group_by(AttachmentOccurrence.conversation_id)
        )
        if conversation_id is not None
    }
    segments_by_conversation: defaultdict[str, list[ConversationSegment]] = defaultdict(list)
    for segment in session.scalars(
        select(ConversationSegment).order_by(
            ConversationSegment.first_activity_ms, ConversationSegment.thread_path
        )
    ).all():
        segments_by_conversation[segment.conversation_id].append(segment)

    for conversation in session.scalars(select(Conversation)).all():
        segments = segments_by_conversation.get(conversation.id, [])
        for sequence, segment in enumerate(segments):
            segment.sequence = sequence
        count, sent, first_ms, last_ms, active_days = message_stats.get(
            conversation.id, (0, 0, None, None, 0)
        )
        attachment_count, unique_count, attachment_bytes = attachment_stats.get(
            conversation.id, (0, 0, 0)
        )
        conversation.message_count = int(count or 0)
        conversation.sent_count = int(sent or 0)
        conversation.received_count = conversation.message_count - conversation.sent_count
        conversation.first_activity_ms = first_ms
        conversation.last_activity_ms = last_ms
        conversation.active_days = int(active_days or 0)
        conversation.attachment_count = int(attachment_count or 0)
        conversation.unique_attachment_count = int(unique_count or 0)
        conversation.attachment_bytes = int(attachment_bytes or 0)
        conversation.participant_count = max(
            (participant_counts.get(segment.id, 0) for segment in segments), default=0
        )
        conversation.completeness = (
            "incomplete"
            if any(segment.completeness == "incomplete" for segment in segments)
            else "complete"
        )
        if segments:
            conversation.primary_category = segments[0].source_category

    orphan_conversation_ids = session.scalars(
        select(Conversation.id).where(
            ~Conversation.id.in_(select(ConversationSegment.conversation_id))
        )
    ).all()
    if orphan_conversation_ids:
        session.execute(delete(Conversation).where(Conversation.id.in_(orphan_conversation_ids)))
    session.execute(
        delete(Participant).where(~Participant.id.in_(select(SegmentParticipant.participant_id)))
    )


def _message_import_owned_star_targets(session: Session) -> set[StarTargetKey]:
    """Snapshot starred identities whose authoritative representation this importer replaces."""

    targets: set[StarTargetKey] = {
        ("message", target_id)
        for target_id in session.scalars(
            select(Star.target_id)
            .join(Message, Message.id == Star.target_id)
            .where(Star.target_type == "message")
        )
    }
    targets.update(
        ("conversation", target_id)
        for target_id in session.scalars(
            select(Star.target_id)
            .join(Conversation, Conversation.id == Star.target_id)
            .where(Star.target_type == "conversation")
        )
    )
    targets.update(
        ("attachment", target_id)
        for target_id in session.scalars(
            select(Star.target_id)
            .join(AttachmentOccurrence, AttachmentOccurrence.id == Star.target_id)
            .where(
                Star.target_type == "attachment",
                AttachmentOccurrence.activity_record_id.is_(None),
                or_(
                    AttachmentOccurrence.message_id.is_not(None),
                    AttachmentOccurrence.conversation_id.is_not(None),
                ),
            )
        )
    )
    contact_targets: set[StarTargetKey] = {
        ("messenger_contact", target_id)
        for target_id in session.scalars(
            select(Star.target_id).where(Star.target_type == "messenger_contact")
        )
    }
    contact_resolutions = resolve_star_targets(
        session,
        contact_targets,
        include_payload=False,
    )
    targets.update(
        key for key, resolution in contact_resolutions.items() if resolution.state != "missing"
    )
    return targets


def _pending_star_reconciliation_checkpoint(
    targets: set[StarTargetKey],
) -> dict[str, object]:
    return {
        "version": _STAR_RECONCILIATION_VERSION,
        "state": "pending",
        "targets": [
            {"target_type": target_type, "target_id": target_id}
            for target_type, target_id in sorted(targets)
        ],
    }


def _pending_message_import_star_targets(session: Session) -> set[StarTargetKey]:
    """Recover exact ownership snapshots from interrupted or incomplete message imports."""

    targets: set[StarTargetKey] = set()
    checkpoints = session.scalars(
        select(ImportRun.checkpoint_json).where(ImportRun.kind == "messages")
    )
    for checkpoint in checkpoints:
        raw = checkpoint.get("star_reconciliation")
        if raw is None:
            continue
        if (
            not isinstance(raw, dict)
            or raw.get("version") != _STAR_RECONCILIATION_VERSION
            or raw.get("state") not in {"pending", "completed"}
        ):
            raise ValueError("Message import star-reconciliation checkpoint is invalid")
        if raw["state"] == "completed":
            continue
        rows = raw.get("targets")
        if not isinstance(rows, list):
            raise ValueError("Message import star-reconciliation checkpoint is invalid")
        for row in rows:
            if (
                not isinstance(row, dict)
                or set(row) != {"target_type", "target_id"}
                or row.get("target_type") not in _MESSAGE_IMPORT_STAR_TYPES
                or not isinstance(row.get("target_id"), str)
                or not row["target_id"]
            ):
                raise ValueError("Message import star-reconciliation checkpoint is invalid")
            targets.add(
                (
                    cast(StarTargetType, row["target_type"]),
                    row["target_id"],
                )
            )
    return targets


def _prune_missing_message_import_star_targets(
    session: Session,
    current_targets: set[StarTargetKey],
) -> tuple[int, int]:
    targets = current_targets | _pending_message_import_star_targets(session)
    targets.update(
        (cast(StarTargetType, target_type), target_id)
        for target_type, target_id in session.execute(
            select(Star.target_type, Star.target_id).where(
                Star.target_type.in_(_MESSAGE_GLOBAL_STAR_TYPES)
            )
        )
    )
    resolutions = resolve_star_targets(session, targets, include_payload=False)
    deleted = delete_stars_for_targets(
        session,
        (key for key, resolution in resolutions.items() if resolution.state == "missing"),
    )
    return deleted, len(targets)


def _upsert_coverage(
    session: Session,
    *,
    key: str,
    label: str,
    status: str,
    raw_count: int,
    logical_count: int,
    description: str,
    source_paths: Mapping[str, Any],
) -> None:
    session.execute(
        sqlite_insert(CoverageArea)
        .values(
            key=key,
            label=label,
            status=status,
            raw_count=raw_count,
            logical_count=logical_count,
            description=description,
            source_paths_json=dict(source_paths),
        )
        .on_conflict_do_update(
            index_elements=[CoverageArea.key],
            set_={
                "label": label,
                "status": status,
                "raw_count": raw_count,
                "logical_count": logical_count,
                "description": description,
                "source_paths_json": dict(source_paths),
            },
        )
    )


def _refresh_summary_and_coverage(
    session: Session,
    summary: MessageImportSummary,
) -> None:
    support_filter = ConversationSegment.source_category == "support"
    summary.normal_segment_count = (
        session.scalar(select(func.count(ConversationSegment.id)).where(~support_filter)) or 0
    )
    summary.support_thread_count = (
        session.scalar(select(func.count(ConversationSegment.id)).where(support_filter)) or 0
    )
    summary.conversation_count = session.scalar(select(func.count(Conversation.id))) or 0
    summary.media_only_conversation_count = (
        session.scalar(
            select(func.count(Conversation.id)).where(
                Conversation.message_count == 0,
                Conversation.attachment_count > 0,
                Conversation.deleted_at.is_(None),
            )
        )
        or 0
    )
    summary.normal_message_count = (
        session.scalar(
            select(func.count(Message.pk))
            .join(ConversationSegment, ConversationSegment.id == Message.segment_id)
            .where(ConversationSegment.source_category != "support")
        )
        or 0
    )
    summary.support_message_count = (
        session.scalar(
            select(func.count(Message.pk))
            .join(ConversationSegment, ConversationSegment.id == Message.segment_id)
            .where(support_filter)
        )
        or 0
    )
    summary.participant_occurrence_count = (
        session.scalar(select(func.count(SegmentParticipant.id))) or 0
    )
    summary.reaction_count = session.scalar(select(func.count(MessageReaction.id))) or 0
    summary.attachment_count = (
        session.scalar(
            select(func.count(AttachmentOccurrence.id)).where(
                AttachmentOccurrence.message_id.is_not(None)
            )
        )
        or 0
    )
    summary.thread_image_count = (
        session.scalar(
            select(func.count(AttachmentOccurrence.id)).where(
                AttachmentOccurrence.kind == "thread_image"
            )
        )
        or 0
    )
    message_attachment_scope = AttachmentOccurrence.activity_record_id.is_(
        None
    ) & AttachmentOccurrence.conversation_id.is_not(None)
    summary.resolved_attachment_count = (
        session.scalar(
            select(func.count(AttachmentOccurrence.id))
            .where(message_attachment_scope)
            .where(AttachmentOccurrence.availability == "resolved")
        )
        or 0
    )
    summary.missing_attachment_count = (
        session.scalar(
            select(func.count(AttachmentOccurrence.id))
            .where(message_attachment_scope)
            .where(AttachmentOccurrence.availability == "missing")
        )
        or 0
    )
    summary.external_attachment_count = (
        session.scalar(
            select(func.count(AttachmentOccurrence.id))
            .where(message_attachment_scope)
            .where(AttachmentOccurrence.availability == "external")
        )
        or 0
    )
    summary.cross_chunk_attachment_count = (
        session.scalar(
            select(func.count(AttachmentOccurrence.id))
            .where(message_attachment_scope)
            .where(
                AttachmentOccurrence.metadata_json["resolution_kind"].as_string() == "cross_chunk"
            )
        )
        or 0
    )
    total_messages = summary.normal_message_count + summary.support_message_count
    logical_messages = total_messages - summary.duplicate_message_count
    total_attachments = summary.attachment_count + summary.thread_image_count
    categories = {
        category: count
        for category, count in session.execute(
            select(
                ConversationSegment.source_category, func.count(ConversationSegment.id)
            ).group_by(ConversationSegment.source_category)
        )
    }
    _upsert_coverage(
        session,
        key="messages",
        label="Messages",
        status="available" if total_messages else "unavailable",
        raw_count=total_messages,
        logical_count=logical_messages,
        description=(
            "Every imported message remains a source occurrence; exact duplicate records are "
            "linked rather than discarded."
        ),
        source_paths={
            "categories": categories,
            "normal_containers": summary.normal_container_count,
            "media_only_conversations": summary.media_only_conversation_count,
        },
    )
    _upsert_coverage(
        session,
        key="message_attachments",
        label="Message attachments",
        status="available" if total_attachments else "unavailable",
        raw_count=total_attachments,
        logical_count=summary.resolved_attachment_count,
        description=(
            "Includes message attachments and conversation images, including missing placeholders."
        ),
        source_paths={
            "missing": summary.missing_attachment_count,
            "external": summary.external_attachment_count,
            "cross_chunk": summary.cross_chunk_attachment_count,
        },
    )
    _upsert_coverage(
        session,
        key="message_reactions",
        label="Message reactions",
        status="available" if summary.reaction_count else "unavailable",
        raw_count=summary.reaction_count,
        logical_count=summary.reaction_count,
        description="Messenger reactions, preserving source order and optional timestamps.",
        source_paths={"area": "your_facebook_activity/messages"},
    )
    _upsert_coverage(
        session,
        key="support_messages",
        label="Support messages",
        status="available" if summary.support_thread_count else "unavailable",
        raw_count=summary.support_message_count,
        logical_count=summary.support_thread_count,
        description=(
            "Support message occurrences and their independently browsable support threads."
        ),
        source_paths={"file": "your_facebook_activity/messages/support_messages.json"},
    )


def _owner_classification_requires_reimport(session: Session, owner_display: str) -> bool:
    message_mismatch = session.scalar(
        select(Message.pk)
        .where(
            or_(
                and_(
                    Message.sender_display == owner_display,
                    Message.sender_is_owner.is_(False),
                ),
                and_(
                    Message.sender_display != owner_display,
                    Message.sender_is_owner.is_(True),
                ),
            )
        )
        .limit(1)
    )
    if message_mismatch is not None:
        return True
    participant_mismatch = session.scalar(
        select(Participant.id)
        .where(
            or_(
                and_(
                    Participant.name_display == owner_display,
                    Participant.is_owner.is_(False),
                ),
                and_(
                    Participant.name_display != owner_display,
                    Participant.is_owner.is_(True),
                ),
            )
        )
        .limit(1)
    )
    return participant_mismatch is not None


def _resolve_owner_identity(engine: Engine, explicit_owner_name: str | None) -> tuple[str, bool]:
    """Resolve and, when explicit, persist the archive owner's display identity.

    Existing archives keep their catalogued setting. A CLI-provided identity takes precedence and
    source reparsing remains forced on every retry while retained ``sender_is_owner``
    classifications disagree. This lets interrupted reclassification resume safely, while a
    correctly classified legacy catalog can adopt the new private setting without invalidating
    derived data. New archives without a setting use a deliberately neutral fallback.
    """

    with Session(engine) as session, session.begin():
        setting = session.get(ArchiveSetting, "owner_identity")
        stored_value = setting.value_json.get("value") if setting is not None else None
        stored_display = (
            normalize_text(stored_value).display.strip()
            if isinstance(stored_value, str) and stored_value.strip()
            else None
        )

        if explicit_owner_name is None:
            owner_display = stored_display or DEFAULT_OWNER_NAME
            requires_reimport = (
                _owner_classification_requires_reimport(session, owner_display)
                if stored_display is not None
                else False
            )
            return owner_display, requires_reimport

        owner_display = normalize_text(explicit_owner_name).display.strip()
        if not owner_display:
            raise ValueError("Owner identity must contain at least one visible character")
        requires_reimport = _owner_classification_requires_reimport(session, owner_display)
        if setting is None:
            session.add(ArchiveSetting(key="owner_identity", value_json={"value": owner_display}))
        else:
            setting.value_json = {"value": owner_display}
        return owner_display, requires_reimport


def import_messages_locked(
    layout: ArchiveLayout,
    engine: Engine,
    *,
    lock: SourceImportPublicationLease,
    owner_name: str | None = None,
    force: bool = False,
    progress: ProgressCallback | None = None,
) -> MessageImportSummary:
    """Import messages while reusing a validated outer source/publication lease.

    The source archive is read-only. A file is skipped only when both its content digest and this
    parser version match the successful prior import. Re-importing a file replaces only records
    whose source coordinate belongs to that file, then global aliases, gaps, aggregates, and
    coverage are deterministically reconciled.
    """

    assert_source_import_publication_allowed(
        layout,
        engine,
        lease=lock,
        publisher="Message source import",
    )
    layout.ensure_runtime_directories()
    owner_display, owner_changed = _resolve_owner_identity(engine, owner_name)
    force = force or owner_changed
    sources = _load_source_refs(engine)
    candidates = [source for source in sources if _candidate_kind(source.relative_path)]
    candidates.sort(
        key=lambda item: (
            item.chunk_ordinal,
            _source_category(item.relative_path),
            PurePosixPath(item.relative_path).parent.as_posix(),
            _chunk_number(item.relative_path),
            item.relative_path,
        )
    )
    validated_blobs: set[str] = set()
    attachment_index = _build_attachment_index(layout, sources, validated_blobs=validated_blobs)
    run_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())
    now = datetime.now(UTC)
    summary = MessageImportSummary(candidate_file_count=len(candidates))
    with Session(engine) as session:
        owned_star_targets = _message_import_owned_star_targets(session)

    with Session(engine) as session, session.begin():
        session.add(
            ImportRun(
                id=run_id,
                kind="messages",
                status="running",
                parser_version=MESSAGE_PARSER_VERSION,
                counters_json={},
                checkpoint_json={
                    "star_reconciliation": _pending_star_reconciliation_checkpoint(
                        owned_star_targets
                    )
                },
            )
        )
        session.add(
            Job(
                id=job_id,
                kind="messages",
                status="running",
                stage="Discovering message containers",
                progress_total=len(candidates),
                started_at=now,
                counters_json={},
                checkpoint_json={},
            )
        )

    try:
        for position, source in enumerate(candidates, start=1):
            kind = _candidate_kind(source.relative_path)
            assert kind is not None
            try:
                source_path = _physical_source_path(layout, source, validated_blobs=validated_blobs)
                digest = _current_source_digest(source_path, source)
                unchanged = (
                    not force
                    and source.imported_parser_version == MESSAGE_PARSER_VERSION
                    and source.imported_sha256 == digest
                    and source.json_status == "parsed"
                )
                if unchanged:
                    summary.skipped_file_count += 1
                    if kind == "support":
                        summary.support_container_count += 1
                    else:
                        summary.normal_container_count += 1
                else:
                    payload = orjson.loads(source_path.read_bytes())
                    if is_purge_tombstone(payload):
                        with Session(engine) as session, session.begin():
                            _resolve_previous_file_anomalies(session, source.id)
                            _detach_source_container(session, source.id)
                            _delete_previous_file_import(session, source.id)
                            session.execute(
                                update(SourceFile)
                                .where(SourceFile.id == source.id)
                                .values(
                                    json_status="parsed",
                                    json_error=None,
                                    imported_parser_version=MESSAGE_PARSER_VERSION,
                                    imported_sha256=digest,
                                )
                            )
                        summary.imported_file_count += 1
                    else:
                        envelope_error = (
                            _support_envelope_error(payload)
                            if kind == "support"
                            else _normal_envelope_error(payload)
                        )
                        if envelope_error is not None:
                            summary.unsupported_file_count += 1
                            _mark_file_problem(
                                engine,
                                source=source,
                                status="unsupported",
                                code="unsupported_message_envelope",
                                message=envelope_error,
                            )
                        else:
                            assert isinstance(payload, dict)
                            with Session(engine) as session, session.begin():
                                _resolve_previous_file_anomalies(session, source.id)
                                _detach_source_container(session, source.id)
                                _delete_previous_file_import(session, source.id)
                                if kind == "support":
                                    _import_support_file(
                                        session,
                                        run_id=run_id,
                                        source=source,
                                        payload=payload,
                                        owner_display=owner_display,
                                        attachment_index=attachment_index,
                                    )
                                    summary.support_container_count += 1
                                else:
                                    _import_thread_file(
                                        session,
                                        run_id=run_id,
                                        source=source,
                                        payload=payload,
                                        owner_display=owner_display,
                                        attachment_index=attachment_index,
                                    )
                                    summary.normal_container_count += 1
                                session.execute(
                                    update(SourceFile)
                                    .where(SourceFile.id == source.id)
                                    .values(
                                        json_status="parsed",
                                        json_error=None,
                                        imported_parser_version=MESSAGE_PARSER_VERSION,
                                        imported_sha256=digest,
                                    )
                                )
                                session.execute(
                                    update(Job)
                                    .where(Job.id == job_id)
                                    .values(
                                        progress_current=position,
                                        stage=f"Imported {source.relative_path}",
                                        checkpoint_json={
                                            "source_file_id": source.id,
                                            "relative_path": source.relative_path,
                                        },
                                    )
                                )
                            summary.imported_file_count += 1
            except (OSError, ValueError, orjson.JSONDecodeError) as exc:
                summary.invalid_file_count += 1
                _mark_file_problem(
                    engine,
                    source=source,
                    status="invalid",
                    code="invalid_message_json",
                    message=f"Message container could not be imported: {exc}",
                )

            if progress is not None:
                progress("Importing message containers", position, len(candidates))

        with Session(engine) as session, session.begin():
            summary.duplicate_message_count = _reconcile_duplicate_messages(session)
            _remove_empty_segments(session)
            (
                summary.missing_leading_segment_count,
                summary.internal_gap_segment_count,
            ) = _reconcile_segments(session)
            _reconcile_conversation_aggregates(session)
            _refresh_summary_and_coverage(session, summary)
            finished = datetime.now(UTC)
            status = (
                "completed_with_errors"
                if summary.invalid_file_count or summary.unsupported_file_count
                else "completed"
            )
            reconciliation_checkpoint: dict[str, object]
            if status == "completed":
                stars_pruned, star_candidate_count = _prune_missing_message_import_star_targets(
                    session,
                    owned_star_targets,
                )
                reconciliation_checkpoint = {
                    "version": _STAR_RECONCILIATION_VERSION,
                    "state": "completed",
                    "candidate_count": star_candidate_count,
                    "pruned_count": stars_pruned,
                }
            else:
                reconciliation_checkpoint = _pending_star_reconciliation_checkpoint(
                    owned_star_targets
                )
            counters = asdict(summary)
            session.execute(
                update(ImportRun)
                .where(ImportRun.id == run_id)
                .values(
                    status=status,
                    finished_at=finished,
                    counters_json=counters,
                    checkpoint_json={
                        "processed_files": len(candidates),
                        "star_reconciliation": reconciliation_checkpoint,
                    },
                )
            )
            session.execute(
                update(Job)
                .where(Job.id == job_id)
                .values(
                    status=status,
                    finished_at=finished,
                    progress_current=len(candidates),
                    stage="Message import complete",
                    counters_json=counters,
                )
            )
        if progress is not None:
            progress("Message import complete", len(candidates), len(candidates))
        if summary.imported_file_count:
            # Imported source rows change the population represented by any prior full-corpus
            # classification, even before a replacement semantic index is built.
            from meta_archive.services.conflict_aggregates import invalidate_conflict_aggregates

            with Session(engine) as session, session.begin():
                invalidate_ready_semantic_indexes(
                    session,
                    reason="Searchable message source data changed",
                    source_kind="message_import",
                    source_id=run_id,
                )
            invalidate_conflict_aggregates(
                engine,
                reason="Message source data changed; rebuild semantic and conflict aggregates",
            )
        return summary
    except BaseException as exc:
        with Session(engine) as session, session.begin():
            finished = datetime.now(UTC)
            session.execute(
                update(ImportRun)
                .where(ImportRun.id == run_id)
                .values(status="failed", finished_at=finished, error_text=str(exc))
            )
            session.execute(
                update(Job)
                .where(Job.id == job_id)
                .values(status="failed", finished_at=finished, error_text=str(exc))
            )
        raise


def import_messages(
    layout: ArchiveLayout,
    engine: Engine,
    *,
    owner_name: str | None = None,
    force: bool = False,
    progress: ProgressCallback | None = None,
) -> MessageImportSummary:
    """Import Messenger and support-message payloads beneath one source lock."""

    with source_import_publication_lock(layout) as lock:
        return import_messages_locked(
            layout,
            engine,
            lock=lock,
            owner_name=owner_name,
            force=force,
            progress=progress,
        )
