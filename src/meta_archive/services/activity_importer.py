from __future__ import annotations

import re
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from urllib.parse import unquote, urlsplit

import orjson
from sqlalchemy import Engine, delete, func, or_, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from meta_archive import __version__
from meta_archive.models import (
    ActivityRecord,
    AttachmentCandidate,
    AttachmentOccurrence,
    CoverageArea,
    ImportRun,
    IntegrityAnomaly,
    Job,
    SourceChunk,
    SourceFile,
    SourceRecord,
)
from meta_archive.paths import ArchiveLayout
from meta_archive.services.identifiers import canonical_fingerprint, stable_id
from meta_archive.services.purge_rewrite import is_purge_tombstone
from meta_archive.services.semantic_state import invalidate_ready_semantic_indexes
from meta_archive.services.source_import_lock import (
    SourceImportPublicationLease,
    assert_source_import_publication_allowed,
    source_import_publication_lock,
)
from meta_archive.services.text import normalize_text

ProgressCallback = Callable[[str, int, int | None], None]
JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
SchemaKind = Literal[
    "album",
    "dating",
    "generic",
    "group_comments",
    "group_membership",
    "message",
    "purged",
    "reactions",
    "support_messages",
]

_THREAD_DIRECTORIES = frozenset(
    {
        "archived_threads",
        "e2ee_cutover",
        "filtered_threads",
        "inbox",
        "legacy_threads",
        "message_requests",
    }
)
_EXTERNAL_SCHEMES = frozenset({"http", "https", "ftp", "data"})
_PREFERRED_TITLE_LABELS = (
    "title",
    "name",
    "product title",
    "product name",
    "group name",
    "message",
)
_ACTOR_LABELS = frozenset({"actor", "author", "seller name", "inviter"})
_TIMESTAMP_KEYS = (
    "timestamp",
    "creation_timestamp",
    "timestamp_value",
    "last_modified_timestamp",
    "recently_active_date",
)


@dataclass(slots=True)
class ActivityImportSummary:
    files_considered: int = 0
    files_imported: int = 0
    files_skipped: int = 0
    files_unsupported: int = 0
    files_invalid: int = 0
    schema_skipped: int = 0
    records_imported: int = 0
    duplicate_records: int = 0
    attachments_imported: int = 0
    attachments_resolved: int = 0
    attachments_missing: int = 0
    attachments_external: int = 0


@dataclass(frozen=True, slots=True)
class _SourceFileRef:
    id: int
    chunk_id: int
    chunk_key: str
    chunk_path: str
    relative_path: str
    normalized_path: str
    sha256: str | None
    json_status: str
    imported_parser_version: str | None
    imported_sha256: str | None

    def physical_path(self, layout: ArchiveLayout) -> Path:
        archive_root = layout.root.resolve()
        chunk_root = (archive_root / self.chunk_path).resolve()
        if chunk_root != archive_root and archive_root not in chunk_root.parents:
            raise ActivitySourceIntegrityError(
                f"source chunk escapes the archive root: {self.chunk_path}"
            )
        candidate = (chunk_root / self.relative_path).resolve()
        if candidate == chunk_root or chunk_root not in candidate.parents:
            raise ActivitySourceIntegrityError(
                f"source coordinate escapes its chunk: {self.relative_path}"
            )
        return candidate


@dataclass(frozen=True, slots=True)
class _MediaSource:
    id: int
    chunk_id: int
    normalized_path: str
    sha256: str | None
    size_bytes: int
    media_type: str | None


@dataclass(frozen=True, slots=True)
class _AttachmentPayload:
    json_pointer: str
    raw_payload: dict[str, JsonValue]
    source_uri: str
    timestamp_ms: int | None


@dataclass(frozen=True, slots=True)
class _ParsedActivity:
    json_pointer: str
    payload: dict[str, JsonValue]
    category: str
    subtype: str
    parent_pointer: str | None = None
    source_fbid: str | None = None
    timestamp_ms: int | None = None
    title_raw: str | None = None
    content_raw: str | None = None
    actor_raw: str | None = None
    target_uri: str | None = None
    availability: str = "available"
    attachments: tuple[_AttachmentPayload, ...] = ()


class ActivitySchemaError(ValueError):
    """A JSON document is valid JSON but is not a supported activity schema."""


class ActivitySourceIntegrityError(RuntimeError):
    """A catalogued source blob no longer matches its inventory digest."""


def _json_pointer_join(base: str, token: str | int) -> str:
    escaped = str(token).replace("~", "~0").replace("/", "~1")
    return f"{base}/{escaped}"


def _as_object(value: object) -> dict[str, JsonValue] | None:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        return None
    return value


def _as_objects(value: object) -> list[dict[str, JsonValue]]:
    if not isinstance(value, list):
        return []
    return [
        item
        for entry in value
        if not is_purge_tombstone(entry)
        for item in (_as_object(entry),)
        if item is not None
    ]


def _text(value: object) -> str | None:
    if isinstance(value, str):
        return value
    return None


def _source_fbid(value: object) -> str | None:
    if isinstance(value, str):
        return value[:100]
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)[:100]
    return None


def _timestamp_ms(value: object) -> int | None:
    """Normalize Facebook epoch seconds while accepting already-normalized milliseconds.

    Facebook uses zero as an unknown/not-recorded sentinel in several activity exports. Negative,
    non-numeric, boolean, NaN, and infinite values are likewise not valid archive timestamps.
    """

    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    if value <= 0 or value != value or value in (float("inf"), float("-inf")):
        return None
    integer = int(value)
    return integer if integer >= 100_000_000_000 else integer * 1000


def _first_timestamp(record: Mapping[str, object]) -> int | None:
    for key in _TIMESTAMP_KEYS:
        normalized = _timestamp_ms(record.get(key))
        if normalized is not None:
            return normalized
    for entry in _as_objects(record.get("label_values")):
        normalized = _timestamp_ms(entry.get("timestamp_value"))
        if normalized is not None:
            return normalized
    return None


def _humanize(value: str) -> str:
    return re.sub(r"[_-]+", " ", value).strip().capitalize()


def _relative_category(relative_path: str) -> str:
    parts = PurePosixPath(relative_path).parts
    if len(parts) >= 2 and parts[0] == "your_facebook_activity":
        source = parts[1]
    else:
        source = parts[0] if parts else "other"
    return {
        "facebook_marketplace": "marketplace",
        "messages": "messenger_metadata",
        "your_problem_reports": "support_cases",
    }.get(source, source)


def _relative_subtype(relative_path: str) -> str:
    stem = PurePosixPath(relative_path).stem.casefold()
    return re.sub(r"[^a-z0-9]+", "_", stem).strip("_")[:80] or "activity"


def _scalar_text(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    return None


def _label_lines(values: object, *, prefix: str = "") -> list[str]:
    lines: list[str] = []
    for item in _as_objects(values):
        label = _text(item.get("label")) or _text(item.get("title")) or "Value"
        qualified = f"{prefix} / {label}" if prefix else label
        scalar = _scalar_text(item.get("value"))
        if scalar is not None:
            lines.append(f"{qualified}: {scalar}")
        title = _text(item.get("title"))
        if title is not None and title != label:
            lines.append(f"{qualified}: {title}")
        for nested_key in ("dict", "vec"):
            lines.extend(_label_lines(item.get(nested_key), prefix=qualified))
        for media in _as_objects(item.get("media")):
            for key in ("title", "description"):
                media_text = _text(media.get(key))
                if media_text:
                    lines.append(f"{qualified} / {_humanize(key)}: {media_text}")
    return lines


def _label_value(record: Mapping[str, object], names: Iterable[str]) -> str | None:
    wanted = {name.casefold() for name in names}
    for item in _as_objects(record.get("label_values")):
        label = (_text(item.get("label")) or "").casefold()
        if label not in wanted:
            continue
        for key in ("value", "title"):
            value = _scalar_text(item.get(key))
            if value:
                return value
    return None


def _first_target_uri(value: object) -> str | None:
    if isinstance(value, dict):
        for key in ("href", "url"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                return candidate
        for child in value.values():
            candidate = _first_target_uri(child)
            if candidate is not None:
                return candidate
    elif isinstance(value, list):
        for child in value:
            candidate = _first_target_uri(child)
            if candidate is not None:
                return candidate
    return None


def _iter_media_objects(value: object, pointer: str) -> Iterator[tuple[str, dict[str, JsonValue]]]:
    if isinstance(value, dict):
        if isinstance(value.get("uri"), str):
            media = _as_object(value)
            if media is not None:
                yield pointer, media
        for key, child in value.items():
            yield from _iter_media_objects(child, _json_pointer_join(pointer, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_media_objects(child, _json_pointer_join(pointer, index))


def _attachments(value: object, pointer: str) -> tuple[_AttachmentPayload, ...]:
    attachments: list[_AttachmentPayload] = []
    for media_pointer, media in _iter_media_objects(value, pointer):
        uri = _text(media.get("uri"))
        if uri is None:
            continue
        attachments.append(
            _AttachmentPayload(
                json_pointer=media_pointer,
                raw_payload=media,
                source_uri=uri,
                timestamp_ms=_first_timestamp(media),
            )
        )
    return tuple(attachments)


def _detect_schema(payload: object) -> SchemaKind | None:
    if is_purge_tombstone(payload):
        return "purged"
    root = _as_object(payload)
    if root is not None:
        if isinstance(root.get("messages"), list) and (
            isinstance(root.get("participants"), list) or isinstance(root.get("thread_path"), str)
        ):
            return "message"
        if isinstance(root.get("support_messages"), list | dict):
            return "support_messages"
        if isinstance(root.get("group_comments_v2"), list):
            return "group_comments"
        if isinstance(root.get("groups_joined_v2"), list):
            return "group_membership"
        if isinstance(root.get("dating_your_activity_v2"), dict):
            return "dating"
        if isinstance(root.get("photos"), list) and isinstance(root.get("cover_photo"), dict):
            return "album"
        if isinstance(root.get("label_values"), list) and (
            isinstance(root.get("media"), list)
            or "fbid" in root
            or "timestamp" in root
            or "title" in root
        ):
            return "generic"
        return None

    if not isinstance(payload, list):
        return None
    if not payload:
        return "generic"
    records = _as_objects(payload)
    if len(records) != sum(not is_purge_tombstone(record) for record in payload):
        return None
    if not records:
        return "generic"
    if any(
        isinstance(data.get("reaction"), dict)
        for record in records
        for data in _as_objects(record.get("data"))
    ):
        return "reactions"
    if all(
        isinstance(record.get("label_values"), list)
        and any(key in record for key in ("fbid", "media", "timestamp", "title"))
        for record in records
    ):
        return "generic"
    return None


def _generic_record(
    record: dict[str, JsonValue],
    *,
    pointer: str,
    relative_path: str,
) -> _ParsedActivity:
    title = _text(record.get("title")) or _label_value(record, _PREFERRED_TITLE_LABELS)
    if title is None:
        title = _humanize(PurePosixPath(relative_path).stem)
    content_lines = _label_lines(record.get("label_values"))
    content = "\n".join(content_lines) or None
    actor = _text(record.get("actor")) or _text(record.get("author"))
    if actor is None:
        actor = _label_value(record, _ACTOR_LABELS)
    return _ParsedActivity(
        json_pointer=pointer,
        payload=record,
        category=_relative_category(relative_path),
        subtype=_relative_subtype(relative_path),
        source_fbid=_source_fbid(record.get("fbid")),
        timestamp_ms=_first_timestamp(record),
        title_raw=title,
        content_raw=content,
        actor_raw=actor,
        target_uri=_first_target_uri(record),
        attachments=_attachments(record, pointer),
    )


def _adapt_generic(payload: object, relative_path: str) -> list[_ParsedActivity]:
    root = _as_object(payload)
    if root is not None:
        return [_generic_record(root, pointer="", relative_path=relative_path)]
    return [
        _generic_record(record, pointer=f"/{index}", relative_path=relative_path)
        for index, record in enumerate(_as_objects(payload))
    ]


def _adapt_reactions(payload: object) -> list[_ParsedActivity]:
    result: list[_ParsedActivity] = []
    for index, record in enumerate(_as_objects(payload)):
        reaction: dict[str, JsonValue] | None = None
        for data in _as_objects(record.get("data")):
            reaction = _as_object(data.get("reaction"))
            if reaction is not None:
                break
        if reaction is None:
            raise ActivitySchemaError(f"reaction record {index} has no reaction payload")
        result.append(
            _ParsedActivity(
                json_pointer=f"/{index}",
                payload=record,
                category="comments_and_reactions",
                subtype="reaction",
                timestamp_ms=_timestamp_ms(record.get("timestamp")),
                title_raw=_text(record.get("title")),
                content_raw=_scalar_text(reaction.get("reaction")),
                actor_raw=_text(reaction.get("actor")),
            )
        )
    return result


def _adapt_group_comments(payload: object) -> list[_ParsedActivity]:
    root = _as_object(payload)
    if root is None:
        raise ActivitySchemaError("group comments payload is not an object")
    result: list[_ParsedActivity] = []
    for index, record in enumerate(_as_objects(root.get("group_comments_v2"))):
        comment: dict[str, JsonValue] | None = None
        for data in _as_objects(record.get("data")):
            comment = _as_object(data.get("comment"))
            if comment is not None:
                break
        pointer = f"/group_comments_v2/{index}"
        title = _text(record.get("title")) or (
            _text(comment.get("group")) if comment is not None else None
        )
        timestamp = (
            _timestamp_ms(comment.get("timestamp")) if comment is not None else None
        ) or _timestamp_ms(record.get("timestamp"))
        result.append(
            _ParsedActivity(
                json_pointer=pointer,
                payload=record,
                category="groups",
                subtype="group_comment",
                timestamp_ms=timestamp,
                title_raw=title,
                content_raw=_text(comment.get("comment")) if comment is not None else None,
                actor_raw=_text(comment.get("author")) if comment is not None else None,
                target_uri=_first_target_uri(record),
                availability="available" if comment is not None else "unavailable_body",
                attachments=_attachments(record, pointer),
            )
        )
    return result


def _adapt_group_membership(payload: object) -> list[_ParsedActivity]:
    root = _as_object(payload)
    if root is None:
        raise ActivitySchemaError("group membership payload is not an object")
    result: list[_ParsedActivity] = []
    for index, record in enumerate(_as_objects(root.get("groups_joined_v2"))):
        names = [
            name
            for data in _as_objects(record.get("data"))
            if (name := _text(data.get("name"))) is not None
        ]
        result.append(
            _ParsedActivity(
                json_pointer=f"/groups_joined_v2/{index}",
                payload=record,
                category="groups",
                subtype="group_membership",
                timestamp_ms=_timestamp_ms(record.get("timestamp")),
                title_raw=_text(record.get("title")) or (names[0] if names else None),
                content_raw="\n".join(names) or None,
            )
        )
    return result


def _adapt_dating(payload: object) -> list[_ParsedActivity]:
    root = _as_object(payload)
    activity = _as_object(root.get("dating_your_activity_v2")) if root is not None else None
    if activity is None:
        raise ActivitySchemaError("dating activity payload is missing")
    lines = [
        f"{_humanize(key)}: {text}"
        for key, value in activity.items()
        if (text := _scalar_text(value)) is not None
    ]
    return [
        _ParsedActivity(
            json_pointer="/dating_your_activity_v2",
            payload=activity,
            category="dating",
            subtype="dating_activity",
            timestamp_ms=_timestamp_ms(activity.get("recently_active_date")),
            title_raw="Your dating activity",
            content_raw="\n".join(lines) or None,
        )
    ]


def _album_child(
    payload: dict[str, JsonValue],
    *,
    pointer: str,
    parent_pointer: str,
    subtype: str,
    fallback_title: str,
) -> _ParsedActivity:
    title = _text(payload.get("title")) or _text(payload.get("description")) or fallback_title
    return _ParsedActivity(
        json_pointer=pointer,
        payload=payload,
        category="posts",
        subtype=subtype,
        parent_pointer=parent_pointer,
        timestamp_ms=_first_timestamp(payload),
        title_raw=title,
        content_raw=_text(payload.get("description")),
        attachments=_attachments(payload, pointer),
    )


def _adapt_album(payload: object) -> list[_ParsedActivity]:
    root = _as_object(payload)
    if root is None:
        raise ActivitySchemaError("album payload is not an object")
    album_name = _text(root.get("name")) or "Untitled album"
    result = [
        _ParsedActivity(
            json_pointer="",
            payload=root,
            category="posts",
            subtype="album",
            timestamp_ms=_timestamp_ms(root.get("last_modified_timestamp")),
            title_raw=album_name,
            content_raw=_text(root.get("description")),
        )
    ]
    for index, photo in enumerate(_as_objects(root.get("photos"))):
        result.append(
            _album_child(
                photo,
                pointer=f"/photos/{index}",
                parent_pointer="",
                subtype="album_photo",
                fallback_title=album_name,
            )
        )
    cover = _as_object(root.get("cover_photo"))
    if cover is not None:
        result.append(
            _album_child(
                cover,
                pointer="/cover_photo",
                parent_pointer="",
                subtype="album_cover_photo",
                fallback_title=f"{album_name} cover photo",
            )
        )
    return result


def _adapt_payload(payload: object, relative_path: str) -> tuple[SchemaKind, list[_ParsedActivity]]:
    schema = _detect_schema(payload)
    if schema is None:
        raise ActivitySchemaError("unsupported JSON activity schema")
    if schema in {"message", "purged", "support_messages"}:
        return schema, []
    adapters: dict[SchemaKind, Callable[[object], list[_ParsedActivity]]] = {
        "album": _adapt_album,
        "dating": _adapt_dating,
        "group_comments": _adapt_group_comments,
        "group_membership": _adapt_group_membership,
        "reactions": _adapt_reactions,
        "generic": lambda value: _adapt_generic(value, relative_path),
    }
    return schema, adapters[schema](payload)


def _is_prefiltered_message_file(relative_path: str) -> bool:
    parts = PurePosixPath(relative_path).parts
    if PurePosixPath(relative_path).name == "support_messages.json":
        return True
    try:
        messages_index = parts.index("messages")
    except ValueError:
        return False
    return len(parts) > messages_index + 1 and parts[messages_index + 1] in _THREAD_DIRECTORIES


def _load_source_files(session: Session) -> list[_SourceFileRef]:
    rows = session.execute(
        select(SourceFile, SourceChunk.original_path, SourceChunk.chunk_key)
        .join(SourceChunk, SourceChunk.id == SourceFile.chunk_id)
        .where(SourceFile.scan_status == "discovered")
        .where(SourceFile.extension == ".json")
        .order_by(SourceChunk.ordinal, SourceFile.relative_path)
    )
    return [
        _SourceFileRef(
            id=source.id,
            chunk_id=source.chunk_id,
            chunk_key=chunk_key,
            chunk_path=chunk_path,
            relative_path=source.relative_path,
            normalized_path=source.normalized_path,
            sha256=source.sha256,
            json_status=source.json_status,
            imported_parser_version=source.imported_parser_version,
            imported_sha256=source.imported_sha256,
        )
        for source, chunk_path, chunk_key in rows
        if not _is_prefiltered_message_file(source.relative_path)
    ]


def _read_source_bytes(layout: ArchiveLayout, source: _SourceFileRef) -> bytes:
    physical_path = source.physical_path(layout)
    if physical_path.is_file():
        return physical_path.read_bytes()
    if source.sha256 is None:
        raise FileNotFoundError(
            f"source entry is absent and has no catalogued digest: {source.relative_path}"
        )
    if re.fullmatch(r"[0-9a-f]{64}", source.sha256) is None:
        raise ActivitySourceIntegrityError(
            f"source has an invalid catalogued digest: {source.relative_path}"
        )
    blob_root = layout.blobs.resolve()
    blob_path = layout.blob_path(source.sha256).resolve()
    if blob_root not in blob_path.parents:
        raise ActivitySourceIntegrityError(
            f"catalogued blob path escapes the blob store: {source.relative_path}"
        )
    if not blob_path.is_file():
        raise FileNotFoundError(
            f"source entry and catalogued blob are absent: {source.relative_path}"
        )
    content = blob_path.read_bytes()
    actual_digest = sha256(content).hexdigest()
    if actual_digest != source.sha256:
        raise ActivitySourceIntegrityError(
            "catalogued source blob digest mismatch for "
            f"{source.relative_path}: expected {source.sha256}, found {actual_digest}"
        )
    return content


def _media_index(session: Session) -> dict[str, list[_MediaSource]]:
    index: dict[str, list[_MediaSource]] = {}
    for row in session.execute(
        select(
            SourceFile.id,
            SourceFile.chunk_id,
            SourceFile.normalized_path,
            SourceFile.sha256,
            SourceFile.size_bytes,
            SourceFile.media_type,
        ).where(SourceFile.scan_status == "discovered")
    ):
        source = _MediaSource(*row)
        index.setdefault(source.normalized_path, []).append(source)
    for candidates in index.values():
        candidates.sort(key=lambda item: item.id)
    return index


def _normalized_local_uri(uri: str) -> str | None:
    if "\x00" in uri or "\\" in uri:
        return None
    try:
        split = urlsplit(uri)
    except ValueError:
        return None
    if split.scheme or split.netloc or split.query or split.fragment or split.path.startswith("/"):
        return None
    decoded = unquote(split.path)
    if (
        not decoded
        or "\x00" in decoded
        or "\\" in decoded
        or decoded.startswith("/")
        or re.match(r"^[A-Za-z]:", decoded) is not None
    ):
        return None
    raw_parts = decoded.split("/")
    if any(part in {".", ".."} for part in raw_parts):
        return None
    return PurePosixPath(*raw_parts).as_posix()


def _is_external_uri(uri: str) -> bool:
    try:
        split = urlsplit(uri)
    except ValueError:
        return False
    return bool(split.netloc) or split.scheme.casefold() in _EXTERNAL_SCHEMES


def _attachment_kind(uri: str, media_type: str | None) -> str:
    lowered = (media_type or "").casefold()
    suffix = PurePosixPath(urlsplit(uri).path).suffix.casefold()
    if lowered == "image/gif" or suffix == ".gif":
        return "gif"
    if lowered.startswith("image/") or suffix in {
        ".bmp",
        ".heic",
        ".jpeg",
        ".jpg",
        ".png",
        ".webp",
    }:
        return "photo"
    if lowered.startswith("video/") or suffix in {".m4v", ".mov", ".mp4", ".webm"}:
        return "video"
    if lowered.startswith("audio/") or suffix in {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".wav"}:
        return "audio"
    if suffix in {
        ".doc",
        ".docx",
        ".odt",
        ".pdf",
        ".ppt",
        ".pptx",
        ".rtf",
        ".txt",
        ".xls",
        ".xlsx",
    }:
        return "document"
    return "file"


def _normalized_fields(raw: str | None) -> tuple[str | None, str | None, str | None]:
    if raw is None:
        return None, None, None
    normalized = normalize_text(raw)
    return normalized.raw, normalized.display, normalized.search


def _delete_previous_file_records(session: Session, source_file_id: int) -> None:
    # A reparse retires missing-media findings from the previous representation. Any still-missing
    # occurrence is deterministically re-opened while the replacement rows are inserted.
    session.execute(
        update(IntegrityAnomaly)
        .where(IntegrityAnomaly.source_file_id == source_file_id)
        .where(IntegrityAnomaly.code == "missing_activity_attachment")
        .values(resolved=True)
    )
    record_ids = select(ActivityRecord.id).where(ActivityRecord.source_file_id == source_file_id)
    session.execute(
        delete(AttachmentOccurrence).where(AttachmentOccurrence.activity_record_id.in_(record_ids))
    )
    session.execute(delete(ActivityRecord).where(ActivityRecord.source_file_id == source_file_id))
    session.execute(
        delete(SourceRecord)
        .where(SourceRecord.source_file_id == source_file_id)
        .where(SourceRecord.record_type.like("activity:%"))
    )


def _first_fingerprints(session: Session) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    rows = session.execute(
        select(SourceRecord.record_type, SourceRecord.logical_fingerprint, SourceRecord.id)
        .where(SourceRecord.record_type.like("activity:%"))
        .where(SourceRecord.logical_fingerprint.is_not(None))
        .order_by(SourceRecord.id)
    )
    for record_type, fingerprint, source_record_id in rows:
        if fingerprint is not None:
            result.setdefault((record_type, fingerprint), source_record_id)
    return result


def _upsert_anomaly(
    session: Session,
    *,
    anomaly_id: str,
    code: str,
    severity: str,
    summary: str,
    source_file_id: int,
    source_record_id: str | None = None,
    detail: dict[str, Any] | None = None,
    resolved: bool = False,
) -> None:
    values: dict[str, Any] = {
        "id": anomaly_id,
        "code": code,
        "severity": severity,
        "summary": summary,
        "source_file_id": source_file_id,
        "source_record_id": source_record_id,
        "detail_json": detail or {},
        "resolved": resolved,
    }
    session.execute(
        sqlite_insert(IntegrityAnomaly)
        .values(**values)
        .on_conflict_do_update(
            index_elements=[IntegrityAnomaly.id],
            set_={key: value for key, value in values.items() if key != "id"},
        )
    )


def _resolve_old_parse_anomalies(session: Session, source_file_id: int) -> None:
    session.execute(
        update(IntegrityAnomaly)
        .where(IntegrityAnomaly.source_file_id == source_file_id)
        .where(IntegrityAnomaly.code.in_({"invalid_activity_json", "unsupported_activity_schema"}))
        .values(resolved=True)
    )


def _insert_attachment(
    session: Session,
    *,
    source: _SourceFileRef,
    activity: _ParsedActivity,
    activity_id: str,
    source_record_id: str,
    attachment: _AttachmentPayload,
    ordinal: int,
    media_index: Mapping[str, Sequence[_MediaSource]],
    summary: ActivityImportSummary,
) -> None:
    local_path = _normalized_local_uri(attachment.source_uri)
    candidates = list(media_index.get(local_path, ())) if local_path is not None else []
    candidates.sort(key=lambda item: (item.chunk_id != source.chunk_id, item.id))
    selected = candidates[0] if candidates else None
    external = _is_external_uri(attachment.source_uri)
    availability = "resolved" if selected is not None else "external" if external else "missing"
    occurrence_id = stable_id(
        "activity-attachment",
        source.chunk_key,
        source.relative_path,
        activity.json_pointer,
        attachment.json_pointer,
        ordinal,
    )
    media_type = selected.media_type if selected is not None else None
    timestamp_ms = attachment.timestamp_ms or activity.timestamp_ms
    session.add(
        AttachmentOccurrence(
            id=occurrence_id,
            activity_record_id=activity_id,
            source_file_id=source.id,
            resolved_source_file_id=selected.id if selected is not None else None,
            blob_sha256=selected.sha256 if selected is not None else None,
            kind=_attachment_kind(attachment.source_uri, media_type),
            ordinal=ordinal,
            source_uri=attachment.source_uri,
            filename=PurePosixPath(urlsplit(attachment.source_uri).path).name or None,
            media_type=media_type,
            size_bytes=selected.size_bytes if selected is not None else None,
            timestamp_ms=timestamp_ms,
            availability=availability,
            metadata_json={
                "json_pointer": attachment.json_pointer,
                "raw_payload": attachment.raw_payload,
                "normalized_local_path": local_path,
            },
        )
    )
    # AttachmentCandidate also uses scalar IDs rather than an ORM relationship.
    session.flush()
    for candidate_ordinal, candidate in enumerate(candidates):
        session.add(
            AttachmentCandidate(
                attachment_id=occurrence_id,
                source_file_id=candidate.id,
                ordinal=candidate_ordinal,
                match_kind="same_chunk" if candidate.chunk_id == source.chunk_id else "cross_chunk",
                selected=candidate.id == selected.id if selected is not None else False,
            )
        )
    summary.attachments_imported += 1
    if availability == "resolved":
        summary.attachments_resolved += 1
    elif availability == "external":
        summary.attachments_external += 1
    else:
        summary.attachments_missing += 1
        _upsert_anomaly(
            session,
            anomaly_id=stable_id("missing-activity-attachment", occurrence_id),
            code="missing_activity_attachment",
            severity="warning",
            summary="An activity attachment references a local file absent from this export.",
            source_file_id=source.id,
            source_record_id=source_record_id,
            detail={
                "activity_id": activity_id,
                "attachment_id": occurrence_id,
                "source_uri": attachment.source_uri,
                "json_pointer": attachment.json_pointer,
            },
        )


def _insert_file_records(
    session: Session,
    *,
    source: _SourceFileRef,
    records: Sequence[_ParsedActivity],
    run_id: str,
    media_index: Mapping[str, Sequence[_MediaSource]],
    summary: ActivityImportSummary,
) -> None:
    _delete_previous_file_records(session, source.id)
    fingerprints = _first_fingerprints(session)
    activity_ids = {
        record.json_pointer: stable_id(
            "activity-record", source.chunk_key, source.relative_path, record.json_pointer
        )
        for record in records
    }
    for activity in records:
        activity_id = activity_ids[activity.json_pointer]
        source_record_id = stable_id(
            "source-record", source.chunk_key, source.relative_path, activity.json_pointer
        )
        record_type = f"activity:{activity.subtype}"
        fingerprint = canonical_fingerprint(
            record_type,
            {
                "category": activity.category,
                "subtype": activity.subtype,
                "payload": activity.payload,
            },
        )
        fingerprint_key = (record_type, fingerprint)
        duplicate_of = fingerprints.get(fingerprint_key)
        if duplicate_of is not None:
            summary.duplicate_records += 1
        else:
            fingerprints[fingerprint_key] = source_record_id
        session.add(
            SourceRecord(
                id=source_record_id,
                source_file_id=source.id,
                json_pointer=activity.json_pointer,
                record_type=record_type,
                logical_fingerprint=fingerprint,
                duplicate_of_id=duplicate_of,
                import_run_id=run_id,
            )
        )
        # The ORM models intentionally expose scalar provenance IDs without relationships. Flush
        # the source coordinate first so SQLite can enforce ActivityRecord.source_record_id.
        session.flush()
        title_raw, title_display, title_search = _normalized_fields(activity.title_raw)
        content_raw, content_display, content_search = _normalized_fields(activity.content_raw)
        actor_raw, actor_display, actor_search = _normalized_fields(activity.actor_raw)
        parent_id = (
            activity_ids.get(activity.parent_pointer)
            if activity.parent_pointer is not None
            else None
        )
        if activity.parent_pointer is not None and parent_id is None:
            raise ActivitySchemaError(
                f"activity {activity.json_pointer} references missing parent "
                f"{activity.parent_pointer}"
            )
        session.add(
            ActivityRecord(
                id=activity_id,
                source_record_id=source_record_id,
                parent_id=parent_id,
                source_file_id=source.id,
                json_pointer=activity.json_pointer,
                category=activity.category,
                subtype=activity.subtype,
                source_fbid=activity.source_fbid,
                timestamp_ms=activity.timestamp_ms,
                title_raw=title_raw,
                title_display=title_display,
                title_search=title_search,
                content_raw=content_raw,
                content_display=content_display,
                content_search=content_search,
                actor_raw=actor_raw,
                actor_display=actor_display,
                actor_search=actor_search,
                target_uri=activity.target_uri,
                availability=activity.availability,
                extra_json={
                    "raw_payload": activity.payload,
                    "provenance": {
                        "source_file_id": source.id,
                        "relative_path": source.relative_path,
                        "json_pointer": activity.json_pointer,
                    },
                },
            )
        )
        session.flush()
        for ordinal, attachment in enumerate(activity.attachments):
            _insert_attachment(
                session,
                source=source,
                activity=activity,
                activity_id=activity_id,
                source_record_id=source_record_id,
                attachment=attachment,
                ordinal=ordinal,
                media_index=media_index,
                summary=summary,
            )
        summary.records_imported += 1
    _resolve_old_parse_anomalies(session, source.id)
    session.execute(
        update(SourceFile)
        .where(SourceFile.id == source.id)
        .values(
            json_status="parsed",
            json_error=None,
            imported_parser_version=__version__,
            imported_sha256=source.sha256,
        )
    )


def _status_file_error(
    session: Session,
    *,
    source: _SourceFileRef,
    status: Literal["invalid", "unsupported"],
    code: str,
    summary: str,
    error: str,
) -> None:
    _delete_previous_file_records(session, source.id)
    invalidate_ready_semantic_indexes(
        session,
        reason="Searchable activity source data was removed after an import error",
        source_kind="activity_import_error",
        source_id=str(source.id),
    )
    _upsert_anomaly(
        session,
        anomaly_id=stable_id(code, source.chunk_key, source.relative_path),
        code=code,
        severity="warning",
        summary=summary,
        source_file_id=source.id,
        detail={"relative_path": source.relative_path, "error": error[:1000]},
    )
    session.execute(
        update(SourceFile)
        .where(SourceFile.id == source.id)
        .values(
            json_status=status,
            json_error=error[:4000],
            imported_parser_version=__version__,
            imported_sha256=source.sha256,
        )
    )


def _upsert_coverage(
    session: Session,
    *,
    key: str,
    label: str,
    status: str,
    raw_count: int | None,
    logical_count: int | None,
    description: str,
    paths: Sequence[str],
) -> None:
    values: dict[str, Any] = {
        "key": key,
        "label": label,
        "status": status,
        "raw_count": raw_count,
        "logical_count": logical_count,
        "description": description,
        "source_paths_json": {"paths": sorted(set(paths))},
        "updated_at": datetime.now(UTC),
    }
    session.execute(
        sqlite_insert(CoverageArea)
        .values(**values)
        .on_conflict_do_update(
            index_elements=[CoverageArea.key],
            set_={key_: value for key_, value in values.items() if key_ != "key"},
        )
    )


def _coverage_counts(
    session: Session, *, subtype: str | None = None, category: str | None = None
) -> tuple[int, int, list[str]]:
    statement = (
        select(
            func.count(ActivityRecord.pk),
            func.count(func.distinct(SourceRecord.logical_fingerprint)),
        )
        .join(SourceRecord, SourceRecord.id == ActivityRecord.source_record_id)
        .where(ActivityRecord.deleted_at.is_(None))
    )
    path_statement = (
        select(SourceFile.normalized_path)
        .join(ActivityRecord, ActivityRecord.source_file_id == SourceFile.id)
        .distinct()
    )
    if subtype is not None:
        statement = statement.where(ActivityRecord.subtype == subtype)
        path_statement = path_statement.where(ActivityRecord.subtype == subtype)
    if category is not None:
        statement = statement.where(ActivityRecord.category == category)
        path_statement = path_statement.where(ActivityRecord.category == category)
    raw_count, logical_count = session.execute(statement).one()
    paths = list(session.scalars(path_statement.order_by(SourceFile.normalized_path)))
    return int(raw_count or 0), int(logical_count or 0), paths


def _coverage_counts_for_paths(
    session: Session,
    *,
    include: Sequence[str],
    exclude: Sequence[str] = (),
) -> tuple[int, int, list[str]]:
    """Count source-backed activity rows selected only by normalized export paths.

    Several relationship exports share the generic ``connections`` category, while general
    posts, albums, and check-ins all share ``posts``. Coverage for those user-facing categories
    must therefore follow explicit source paths rather than broad inferred categories.
    """

    normalized_path = func.lower(SourceFile.normalized_path)
    path_filter = or_(*(normalized_path.like(pattern) for pattern in include))
    filters = [ActivityRecord.deleted_at.is_(None), path_filter]
    filters.extend(~normalized_path.like(pattern) for pattern in exclude)
    raw_count, logical_count = session.execute(
        select(
            func.count(ActivityRecord.pk),
            func.count(func.distinct(SourceRecord.logical_fingerprint)),
        )
        .join(SourceRecord, SourceRecord.id == ActivityRecord.source_record_id)
        .join(SourceFile, SourceFile.id == ActivityRecord.source_file_id)
        .where(*filters)
    ).one()
    paths = list(
        session.scalars(
            select(SourceFile.normalized_path)
            .join(ActivityRecord, ActivityRecord.source_file_id == SourceFile.id)
            .where(*filters)
            .distinct()
            .order_by(SourceFile.normalized_path)
        )
    )
    return int(raw_count or 0), int(logical_count or 0), paths


def _refresh_coverage(session: Session) -> None:
    path_backed = (
        (
            "friends",
            "Friends",
            ("%connections/friends/%",),
            (),
            "Accepted-friend and removed-friend records explicitly present in the export.",
            "Accepted-friend history is unavailable in this export; chat contacts are not friends.",
        ),
        (
            "friend_requests",
            "Friend requests",
            ("%connections/friend_requests/%",),
            (),
            "Friend-request records explicitly present in the export.",
            "Friend-request history is unavailable in this export.",
        ),
        (
            "followers",
            "Followers",
            ("%connections/followers%", "%connections/following%"),
            (),
            "Follower or following records explicitly present in the export.",
            "Follower history is unavailable in this export.",
        ),
        (
            "facebook_blocks",
            "Facebook blocks",
            ("%facebook%block%", "%blocked_profiles%"),
            (),
            "Facebook block records explicitly present in the export.",
            "Facebook block history is unavailable in this export.",
        ),
        (
            "messenger_blocks",
            "Messenger blocks",
            ("%messenger%block%", "%blocked_messages%"),
            (),
            "Messenger block records explicitly present in the export.",
            "Messenger block history is unavailable in this export.",
        ),
        (
            "events",
            "Events",
            ("%your_facebook_activity/events/%",),
            (),
            "Event records explicitly present in the export.",
            "The events directory is present but contains no event history files.",
        ),
        (
            "general_posts",
            "General posts",
            ("%your_facebook_activity/posts/%",),
            ("%/album/%", "%check%in%"),
            (
                "General post records explicitly present in the export; album and check-in "
                "records remain separate."
            ),
            "No general wall-post manifest is present; album and check-in records are separate.",
        ),
    )
    general_posts_available = False
    for key, label, include, exclude, available_description, unavailable_description in path_backed:
        raw, logical, paths = _coverage_counts_for_paths(
            session,
            include=include,
            exclude=exclude,
        )
        if key == "general_posts":
            general_posts_available = bool(paths)
        _upsert_coverage(
            session,
            key=key,
            label=label,
            status="partial" if paths else "unavailable",
            raw_count=raw if paths else None,
            logical_count=logical if paths else None,
            description=available_description if paths else unavailable_description,
            paths=paths,
        )

    special = (
        (
            "reactions",
            "Likes and reactions",
            "available",
            "reaction",
            None,
            "Raw source occurrences are preserved; exact duplicates are linked by provenance.",
        ),
        (
            "group_comments",
            "Group comments",
            "available",
            "group_comment",
            None,
            "Comments authored in Facebook groups that are present in this export.",
        ),
        (
            "group_membership",
            "Group membership activity",
            "available",
            "group_membership",
            None,
            "Explicit group-join activity present in this export.",
        ),
        (
            "albums",
            "Albums and album media",
            "available",
            None,
            "posts",
            "Album, photo, and cover-photo source records present in the export.",
        ),
    )
    for key, label, status, subtype, category, description in special:
        raw, logical, coverage_paths = _coverage_counts(session, subtype=subtype, category=category)
        if key == "albums":
            coverage_paths = [path for path in coverage_paths if "/album/" in f"/{path}"]
            ids = select(SourceFile.id).where(SourceFile.normalized_path.in_(coverage_paths))
            raw = (
                session.scalar(
                    select(func.count(ActivityRecord.pk)).where(
                        ActivityRecord.source_file_id.in_(ids)
                    )
                )
                or 0
            )
            logical = (
                session.scalar(
                    select(func.count(func.distinct(SourceRecord.logical_fingerprint)))
                    .join(ActivityRecord, ActivityRecord.source_record_id == SourceRecord.id)
                    .where(ActivityRecord.source_file_id.in_(ids))
                )
                or 0
            )
        _upsert_coverage(
            session,
            key=key,
            label=label,
            status=status if coverage_paths else "unavailable",
            raw_count=int(raw) if coverage_paths else None,
            logical_count=int(logical) if coverage_paths else None,
            description=description,
            paths=coverage_paths,
        )

    posts_raw, posts_logical, post_paths = _coverage_counts(session, category="posts")
    _upsert_coverage(
        session,
        key="posts",
        label="Posts, albums, and check-ins",
        status="partial" if post_paths else "unavailable",
        raw_count=posts_raw if post_paths else None,
        logical_count=posts_logical if post_paths else None,
        description=(
            "Exported post-related records are represented, but post coverage remains partial."
            if general_posts_available
            else (
                "Only exported album and check-in activity is represented; general posts are "
                "absent."
            )
        ),
        paths=post_paths,
    )

    categories = list(
        session.execute(
            select(ActivityRecord.category, func.count(ActivityRecord.pk))
            .where(ActivityRecord.deleted_at.is_(None))
            .group_by(ActivityRecord.category)
            .order_by(ActivityRecord.category)
        )
    )
    for category, _count in categories:
        if category == "posts":
            continue
        raw, logical, category_paths = _coverage_counts(session, category=category)
        _upsert_coverage(
            session,
            key=f"activity_{category}",
            label=_humanize(category),
            status=("partial" if category in {"comments_and_reactions", "groups"} else "available"),
            raw_count=raw,
            logical_count=logical,
            description=f"Activity records available from the exported {category} sources.",
            paths=category_paths,
        )


def import_activities_locked(
    layout: ArchiveLayout,
    engine: Engine,
    *,
    lock: SourceImportPublicationLease,
    force: bool = False,
    progress: ProgressCallback | None = None,
) -> ActivityImportSummary:
    """Import activities while reusing a validated outer source/publication lease.

    A completed source file is skipped when its parser version and source hash still match. Every
    imported file is its own transaction, so an interruption can resume without rolling back prior
    files. Conversation threads and the separate support-message schema are deliberately left for
    the message importer.
    """

    assert_source_import_publication_allowed(
        layout,
        engine,
        lease=lock,
        publisher="Activity source import",
    )
    layout.ensure_runtime_directories()
    run_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())
    started = datetime.now(UTC)
    summary = ActivityImportSummary()
    with Session(engine) as session, session.begin():
        sources = _load_source_files(session)
        media_index = _media_index(session)
        summary.files_considered = len(sources)
        session.add(
            ImportRun(
                id=run_id,
                kind="activity",
                status="running",
                parser_version=__version__,
                counters_json={},
                checkpoint_json={},
            )
        )
        session.add(
            Job(
                id=job_id,
                kind="activity_import",
                status="running",
                progress_total=len(sources),
                stage="Importing activity files",
                started_at=started,
                counters_json={},
                checkpoint_json={},
            )
        )

    try:
        for file_index, source in enumerate(sources, start=1):
            unchanged_import = (
                source.imported_parser_version == __version__
                and source.json_status in {"parsed", "invalid", "unsupported"}
                and (
                    source.sha256 is None
                    or source.imported_sha256 is None
                    or source.imported_sha256 == source.sha256
                )
            )
            if unchanged_import and not force:
                summary.files_skipped += 1
            else:
                try:
                    payload = orjson.loads(_read_source_bytes(layout, source))
                except (OSError, orjson.JSONDecodeError) as exc:
                    with Session(engine) as session, session.begin():
                        _status_file_error(
                            session,
                            source=source,
                            status="invalid",
                            code="invalid_activity_json",
                            summary="An activity source file could not be parsed as JSON.",
                            error=str(exc),
                        )
                    summary.files_invalid += 1
                else:
                    try:
                        schema, records = _adapt_payload(payload, source.relative_path)
                    except ActivitySchemaError as exc:
                        with Session(engine) as session, session.begin():
                            _status_file_error(
                                session,
                                source=source,
                                status="unsupported",
                                code="unsupported_activity_schema",
                                summary=(
                                    "A valid JSON file does not match a supported activity schema."
                                ),
                                error=str(exc),
                            )
                        summary.files_unsupported += 1
                    else:
                        if schema in {"message", "support_messages"}:
                            summary.schema_skipped += 1
                        else:
                            with Session(engine) as session, session.begin():
                                _insert_file_records(
                                    session,
                                    source=source,
                                    records=records,
                                    run_id=run_id,
                                    media_index=media_index,
                                    summary=summary,
                                )
                                invalidate_ready_semantic_indexes(
                                    session,
                                    reason="Searchable activity source data changed",
                                    source_kind="activity_import",
                                    source_id=str(source.id),
                                )
                            summary.files_imported += 1

            with Session(engine) as session, session.begin():
                session.execute(
                    update(Job)
                    .where(Job.id == job_id)
                    .values(
                        progress_current=file_index,
                        stage=f"Processed {source.relative_path}",
                        checkpoint_json={
                            "last_source_file_id": source.id,
                            "last_relative_path": source.relative_path,
                        },
                        counters_json=asdict(summary),
                    )
                )
            if progress:
                progress("Importing activity", file_index, len(sources))

        finished = datetime.now(UTC)
        with Session(engine) as session, session.begin():
            _refresh_coverage(session)
            session.execute(
                update(ImportRun)
                .where(ImportRun.id == run_id)
                .values(
                    status="completed",
                    finished_at=finished,
                    counters_json=asdict(summary),
                    checkpoint_json={"files_processed": len(sources)},
                )
            )
            session.execute(
                update(Job)
                .where(Job.id == job_id)
                .values(
                    status="completed",
                    finished_at=finished,
                    progress_current=len(sources),
                    stage="Activity import complete",
                    counters_json=asdict(summary),
                )
            )
        if progress:
            progress("Activity import complete", len(sources), len(sources))
        return summary
    except BaseException as exc:
        finished = datetime.now(UTC)
        with Session(engine) as session, session.begin():
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


def import_activities(
    layout: ArchiveLayout,
    engine: Engine,
    *,
    force: bool = False,
    progress: ProgressCallback | None = None,
) -> ActivityImportSummary:
    """Import all recognized non-conversation activity JSON beneath one source lock."""

    with source_import_publication_lock(layout) as lock:
        return import_activities_locked(
            layout,
            engine,
            lock=lock,
            force=force,
            progress=progress,
        )


# A singular spelling is convenient for callers and remains an intentionally identical API.
import_activity = import_activities


def inspect_activity_payload(payload: object, relative_path: str) -> dict[str, int | str]:
    """Return privacy-safe adapter counts without writing a catalog.

    This is intentionally small and is useful for schema audits and fixture tests. It never returns
    titles, message/comment text, actors, URIs, or raw payloads.
    """

    schema, records = _adapt_payload(payload, relative_path)
    return {
        "schema": schema,
        "record_count": len(records),
        "attachment_count": sum(len(record.attachments) for record in records),
    }
