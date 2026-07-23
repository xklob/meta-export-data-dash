from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Literal, cast

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import and_, case, func, literal, or_, select
from sqlalchemy.orm import Session, aliased
from sqlalchemy.sql.elements import ColumnElement

from meta_archive.api.aggregate_cache import (
    catalog_aggregate_memo,
    catalog_revision,
)
from meta_archive.api.pagination import (
    InvalidCursor,
    decode_cursor,
    encode_cursor,
    require_cursor_int,
    require_cursor_str,
)
from meta_archive.api.security import require_session
from meta_archive.models import (
    ActivityRecord,
    AttachmentOccurrence,
    Blob,
    Conversation,
    ConversationSegment,
    Message,
    SourceChunk,
    SourceFile,
)
from meta_archive.services.jumps import activity_jump
from meta_archive.services.library import unlinked_media_library_path
from meta_archive.services.media_derivatives import derivatives_by_blob
from meta_archive.services.stars import attach_star_states
from meta_archive.services.text import normalize_text

media_router = APIRouter(prefix="/api/v1/media", dependencies=[Depends(require_session)])
ResolvedSourceFile = aliased(SourceFile, name="resolved_media_source")
ResolvedSourceChunk = aliased(SourceChunk, name="resolved_media_chunk")

MediaView = Literal["occurrences", "unique"]
MediaDirection = Literal["asc", "desc"]

MEDIA_SORT_NAMES = frozenset(
    {"date", "size", "type", "filename", "conversation", "availability", "occurrences"}
)
MEDIA_AVAILABILITY = frozenset({"resolved", "missing", "unlinked", "external"})
MEDIA_TYPES = frozenset({"photo", "video", "audio", "gif", "sticker", "document", "other"})
_MEDIA_CACHE_INITIALIZATION_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class _MediaSummaryAggregate:
    """Private-content-free immutable value retained by the shared aggregate cache."""

    occurrences: int
    unique_files: int
    occurrence_bytes: int
    unique_bytes: int
    availability: tuple[tuple[str, int], ...]
    media_types: tuple[tuple[str, int], ...]

    def payload(self) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
        return (
            {
                "occurrences": self.occurrences,
                "unique_files": self.unique_files,
                "occurrence_bytes": self.occurrence_bytes,
                "unique_bytes": self.unique_bytes,
            },
            {"availability": dict(self.availability), "type": dict(self.media_types)},
        )


@dataclass(frozen=True, slots=True)
class _PageCacheEntry:
    created_monotonic: float
    items: list[dict[str, Any]]
    next_cursor: str | None


@dataclass(frozen=True, slots=True)
class _MediaProjectionRow:
    """Minimal immutable cursor identity/order metadata; never a display payload."""

    id: str
    group_key: str
    occurrence_count: int
    first_timestamp_ms: int | None
    last_timestamp_ms: int | None
    sort_value: str | int | None
    sort_null: int


@dataclass(frozen=True, slots=True)
class _MediaProjection:
    ascending: tuple[_MediaProjectionRow, ...]
    descending: tuple[_MediaProjectionRow, ...]
    accounted_bytes: int

    @property
    def row_count(self) -> int:
        return len(self.ascending)


class _MediaProjectionCache:
    """Revision-bound LRU shared by minimal occurrence and unique cursor projections.

    Scope/revision/sort inputs are hashed, values are deeply immutable, and both retained rows and
    conservatively accounted Python object bytes are capped. In particular, this cache never holds
    filenames, conversation titles, payload dictionaries, or source paths unless one normalized
    string is itself the active sort value.
    """

    def __init__(
        self,
        *,
        max_entries: int = 8,
        max_total_rows: int = 500_000,
        max_total_bytes: int = 128 * 1024 * 1024,
    ) -> None:
        if min(max_entries, max_total_rows, max_total_bytes) < 1:
            raise ValueError("Unique media projection cache bounds must be positive")
        self.max_entries = max_entries
        self.max_total_rows = max_total_rows
        self.max_total_bytes = max_total_bytes
        self._entries: OrderedDict[bytes, _MediaProjection] = OrderedDict()
        self._total_rows = 0
        self._total_bytes = 0
        self._lock = threading.Lock()

    @staticmethod
    def _key(*, namespace: str, scope: str, revision: str, sort: str) -> bytes:
        framed = f"media-projection-v1\0{namespace}\0{scope}\0{revision}\0{sort}".encode(
            "utf-8", errors="surrogatepass"
        )
        return hashlib.sha256(framed).digest()

    def get_or_compute(
        self,
        *,
        namespace: str,
        scope: str,
        revision: str,
        sort: str,
        revision_is_current: Callable[[], bool],
        compute: Callable[[], _MediaProjection],
    ) -> _MediaProjection:
        key = self._key(namespace=namespace, scope=scope, revision=revision, sort=sort)
        with self._lock:
            cached = self._entries.get(key)
            if cached is not None:
                self._entries.move_to_end(key)
        if cached is not None and revision_is_current():
            return cached

        value = compute()
        if not revision_is_current():
            return value
        # An oversized projection remains usable for this one request but is never retained. This
        # fails safe on memory for unusually large archives without changing response semantics.
        if value.row_count > self.max_total_rows or value.accounted_bytes > self.max_total_bytes:
            return value
        with self._lock:
            existing = self._entries.get(key)
            if existing is not None:
                self._entries.move_to_end(key)
                return existing
            while self._entries and (
                len(self._entries) >= self.max_entries
                or self._total_rows + value.row_count > self.max_total_rows
                or self._total_bytes + value.accounted_bytes > self.max_total_bytes
            ):
                _, evicted = self._entries.popitem(last=False)
                self._total_rows -= evicted.row_count
                self._total_bytes -= evicted.accounted_bytes
            self._entries[key] = value
            self._total_rows += value.row_count
            self._total_bytes += value.accounted_bytes
        return value

    @property
    def retained_rows(self) -> int:
        with self._lock:
            return self._total_rows

    @property
    def retained_bytes(self) -> int:
        with self._lock:
            return self._total_bytes

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


class _MediaPageCache:
    """Revision-bound LRU for warm list navigation over immutable catalog pages."""

    def __init__(self, *, ttl_seconds: float = 30.0, max_entries: int = 256) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._entries: OrderedDict[str, _PageCacheEntry] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> tuple[list[dict[str, Any]], str | None] | None:
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if now - entry.created_monotonic > self.ttl_seconds:
                del self._entries[key]
                return None
            self._entries.move_to_end(key)
            return deepcopy(entry.items), entry.next_cursor

    def set(self, key: str, items: list[dict[str, Any]], next_cursor: str | None) -> None:
        with self._lock:
            self._entries[key] = _PageCacheEntry(
                created_monotonic=time.monotonic(),
                items=deepcopy(items),
                next_cursor=next_cursor,
            )
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)


def _page_cache(request: Request) -> _MediaPageCache:
    cache = getattr(request.app.state, "media_page_cache", None)
    if cache is None:
        cache = _MediaPageCache()
        request.app.state.media_page_cache = cache
    return cache


def _projection_cache(request: Request) -> _MediaProjectionCache:
    cache = getattr(request.app.state, "media_projection_cache", None)
    if isinstance(cache, _MediaProjectionCache):
        return cache
    with _MEDIA_CACHE_INITIALIZATION_LOCK:
        cache = getattr(request.app.state, "media_projection_cache", None)
        if isinstance(cache, _MediaProjectionCache):
            return cache
        created = _MediaProjectionCache()
        request.app.state.media_projection_cache = created
        return created


def _session(request: Request) -> Session:
    return Session(request.app.state.engine)


def _scope_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def _csv_values(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))


def _media_type_expression() -> ColumnElement[str]:
    media_type = func.lower(func.coalesce(AttachmentOccurrence.media_type, ""))
    return case(
        (AttachmentOccurrence.kind.in_(("gifs", "gif")), "gif"),
        (AttachmentOccurrence.kind == "sticker", "sticker"),
        # Voice messages can be exported as video/mp4, so explicit source kind wins.
        (AttachmentOccurrence.kind.in_(("audio_files", "audio")), "audio"),
        (AttachmentOccurrence.kind.in_(("videos", "video")), "video"),
        (AttachmentOccurrence.kind.in_(("photos", "photo", "thread_image")), "photo"),
        (media_type.like("audio/%"), "audio"),
        (media_type.like("video/%"), "video"),
        (media_type.like("image/%"), "photo"),
        (
            or_(
                AttachmentOccurrence.kind.in_(("files", "file", "support_file", "document")),
                media_type.in_(("application/pdf", "text/plain")),
            ),
            "document",
        ),
        else_="other",
    )


def _source_category_expression() -> ColumnElement[str]:
    return func.coalesce(
        ConversationSegment.source_category,
        Conversation.primary_category,
        ActivityRecord.category,
        SourceFile.category,
    )


def _media_joins(statement: Any) -> Any:
    return (
        statement.select_from(AttachmentOccurrence)
        .outerjoin(Conversation, Conversation.id == AttachmentOccurrence.conversation_id)
        .outerjoin(Message, Message.id == AttachmentOccurrence.message_id)
        .outerjoin(ConversationSegment, ConversationSegment.id == Message.segment_id)
        .outerjoin(
            ActivityRecord,
            and_(
                ActivityRecord.id == AttachmentOccurrence.activity_record_id,
                ActivityRecord.deleted_at.is_(None),
            ),
        )
        .outerjoin(SourceFile, SourceFile.id == AttachmentOccurrence.source_file_id)
        .outerjoin(
            ResolvedSourceFile,
            ResolvedSourceFile.id == AttachmentOccurrence.resolved_source_file_id,
        )
        .outerjoin(ResolvedSourceChunk, ResolvedSourceChunk.id == ResolvedSourceFile.chunk_id)
        .outerjoin(Blob, Blob.sha256 == AttachmentOccurrence.blob_sha256)
    )


def _media_filters(
    *,
    occurrence_id: str | None,
    q: str | None,
    media_type: str | None,
    kind: str | None,
    availability: str | None,
    conversation_id: str | None,
    conversation_q: str | None,
    conversation_type: str | None,
    source_category: str | None,
    from_ms: int | None,
    to_ms: int | None,
    min_size: int | None,
    max_size: int | None,
) -> tuple[list[ColumnElement[bool]], tuple[str, ...], tuple[str, ...], bool]:
    filters: list[ColumnElement[bool]] = [AttachmentOccurrence.deleted_at.is_(None)]
    if occurrence_id:
        filters.append(AttachmentOccurrence.id == occurrence_id)
    types = _csv_values(media_type)
    invalid_types = set(types) - MEDIA_TYPES
    if invalid_types:
        raise HTTPException(
            status_code=422, detail=f"Unknown media type: {sorted(invalid_types)[0]}"
        )
    if types:
        filters.append(_media_type_expression().in_(types))
    kinds = _csv_values(kind)
    if kinds:
        filters.append(AttachmentOccurrence.kind.in_(kinds))
    availability_values = _csv_values(availability)
    invalid_availability = set(availability_values) - MEDIA_AVAILABILITY
    if invalid_availability:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown media availability: {sorted(invalid_availability)[0]}",
        )
    if availability_values:
        filters.append(AttachmentOccurrence.availability.in_(availability_values))
    if conversation_id:
        filters.append(AttachmentOccurrence.conversation_id == conversation_id)
    if conversation_q:
        normalized_conversation = normalize_text(conversation_q).search
        escaped_conversation = (
            normalized_conversation.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        )
        filters.append(Conversation.title_search.like(f"%{escaped_conversation}%", escape="\\"))
    conversation_types = _csv_values(conversation_type)
    if conversation_types:
        filters.append(Conversation.conversation_type.in_(conversation_types))
    source_categories = _csv_values(source_category)
    if source_categories:
        filters.append(_source_category_expression().in_(source_categories))
    if from_ms is not None:
        filters.append(AttachmentOccurrence.timestamp_ms >= from_ms)
    if to_ms is not None:
        filters.append(AttachmentOccurrence.timestamp_ms <= to_ms)
    if min_size is not None:
        filters.append(AttachmentOccurrence.size_bytes >= min_size)
    if max_size is not None:
        filters.append(AttachmentOccurrence.size_bytes <= max_size)
    if q:
        normalized = normalize_text(q).search
        escaped = normalized.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        filters.append(func.lower(AttachmentOccurrence.filename).like(f"%{escaped}%", escape="\\"))
    needs_summary_joins = bool(conversation_q or conversation_types or source_categories)
    return filters, types, availability_values, needs_summary_joins


def _jump(
    conversation_id: str | None,
    message_id: str | None,
    activity_id: str | None,
    occurrence_id: str,
) -> dict[str, str] | None:
    if conversation_id is not None:
        if message_id is None:
            return {
                "href": f"/conversations/{conversation_id}",
                "conversation_id": conversation_id,
            }
        return {
            "href": (
                f"/conversations/{conversation_id}?around={message_id}"
                f"&media={occurrence_id}#message-{message_id}"
            ),
            "conversation_id": conversation_id,
            "message_id": message_id,
            "fragment": f"message-{message_id}",
        }
    return activity_jump(activity_id) if activity_id is not None else None


def _safe_archive_source_path(chunk_key: str | None, relative_path: str | None) -> str | None:
    if not chunk_key or not relative_path:
        return None
    chunk = PurePosixPath(chunk_key.replace("\\", "/"))
    relative = PurePosixPath(relative_path.replace("\\", "/"))
    if chunk.is_absolute() or relative.is_absolute():
        return None
    if any(part in {"", ".", ".."} for part in (*chunk.parts, *relative.parts)):
        return None
    return (chunk / relative).as_posix()


def _unlinked_provenance(
    row: Any,
    *,
    source_chunk: str | None,
    source_relative_path: str | None,
    source_extension: str | None,
    blob_extension: str | None,
) -> dict[str, Any] | None:
    if (
        row.availability != "unlinked"
        or row.conversation_id is not None
        or row.message_id is not None
        or row.activity_record_id is not None
    ):
        return None
    copy_count_value = row.metadata_json.get("source_copy_count")
    copy_count = (
        copy_count_value
        if isinstance(copy_count_value, int) and not isinstance(copy_count_value, bool)
        else 1
    )
    return {
        "source_file_id": row.resolved_source_file_id or row.source_file_id,
        "source_chunk": source_chunk,
        "source_relative_path": source_relative_path,
        "source_path": _safe_archive_source_path(source_chunk, source_relative_path),
        "source_copy_count": max(1, copy_count),
        "library_path": unlinked_media_library_path(
            row,
            source_relative_path=source_relative_path,
            source_extension=source_extension,
            blob_extension=blob_extension,
        ),
        "explanation": (
            "This catalogued file was present in the export but was not referenced by an "
            "imported message or activity record."
        ),
    }


def _occurrence_payload(
    row: AttachmentOccurrence,
    *,
    media_type: str,
    conversation_title: str | None,
    activity_id: str | None,
    activity_title: str | None,
    activity_category: str | None,
    activity_subtype: str | None,
    source_category: str | None,
    source_chunk: str | None,
    source_relative_path: str | None,
    source_extension: str | None,
    blob_extension: str | None,
) -> dict[str, Any]:
    return {
        "id": row.id,
        "view_key": f"occurrence:{row.id}",
        "type": media_type,
        "kind": row.kind,
        "filename": row.filename,
        "media_type": row.media_type,
        "size_bytes": row.size_bytes,
        "timestamp_ms": row.timestamp_ms,
        "unknown_date": row.timestamp_ms is None,
        "availability": row.availability,
        "conversation": (
            {"id": row.conversation_id, "title": conversation_title}
            if row.conversation_id
            else None
        ),
        "activity": (
            {
                "id": activity_id,
                "title": activity_title,
                "category": activity_category,
                "subtype": activity_subtype,
            }
            if activity_id is not None
            else None
        ),
        "source_category": source_category,
        "message_id": row.message_id,
        "blob_sha256": row.blob_sha256,
        "blob_href": f"/api/v1/blobs/{row.blob_sha256}" if row.blob_sha256 else None,
        "is_external_reference": row.availability == "external",
        "jump": _jump(row.conversation_id, row.message_id, activity_id, row.id),
        "provenance": _unlinked_provenance(
            row,
            source_chunk=source_chunk,
            source_relative_path=source_relative_path,
            source_extension=source_extension,
            blob_extension=blob_extension,
        ),
    }


def _base_summary(
    session: Session,
    *,
    filters: list[ColumnElement[bool]],
    needs_joins: bool,
) -> _MediaSummaryAggregate:
    group_key = case(
        (
            AttachmentOccurrence.blob_sha256.is_not(None),
            literal("blob:") + AttachmentOccurrence.blob_sha256,
        ),
        else_=literal("occurrence:") + AttachmentOccurrence.id,
    )
    type_expression = _media_type_expression()
    aggregate_columns: list[Any] = [
        func.count(AttachmentOccurrence.id),
        func.count(func.distinct(group_key)),
        func.coalesce(func.sum(AttachmentOccurrence.size_bytes), 0),
        *[
            func.coalesce(
                func.sum(case((AttachmentOccurrence.availability == value, 1), else_=0)), 0
            )
            for value in sorted(MEDIA_AVAILABILITY)
        ],
        *[
            func.coalesce(func.sum(case((type_expression == value, 1), else_=0)), 0)
            for value in sorted(MEDIA_TYPES)
        ],
    ]
    totals_statement = select(*aggregate_columns).select_from(AttachmentOccurrence)
    if needs_joins:
        totals_statement = _media_joins(select(*aggregate_columns))
    aggregate_row = session.execute(totals_statement.where(*filters)).one()
    occurrences, unique_files, occurrence_bytes = aggregate_row[:3]
    unique_statement = select(
        group_key.label("group_key"),
        func.max(AttachmentOccurrence.size_bytes).label("size_bytes"),
    ).select_from(AttachmentOccurrence)
    if needs_joins:
        unique_statement = _media_joins(
            select(
                group_key.label("group_key"),
                func.max(AttachmentOccurrence.size_bytes).label("size_bytes"),
            )
        )
    unique_sizes = unique_statement.where(*filters).group_by(group_key).subquery()
    unique_bytes = session.scalar(select(func.coalesce(func.sum(unique_sizes.c.size_bytes), 0)))
    offset = 3
    availability_counts = {
        value: int(aggregate_row[offset + index])
        for index, value in enumerate(sorted(MEDIA_AVAILABILITY))
        if aggregate_row[offset + index]
    }
    offset += len(MEDIA_AVAILABILITY)
    type_counts = {
        value: int(aggregate_row[offset + index])
        for index, value in enumerate(sorted(MEDIA_TYPES))
        if aggregate_row[offset + index]
    }
    return _MediaSummaryAggregate(
        occurrences=int(occurrences),
        unique_files=int(unique_files),
        occurrence_bytes=int(occurrence_bytes),
        unique_bytes=int(unique_bytes or 0),
        availability=tuple(availability_counts.items()),
        media_types=tuple(type_counts.items()),
    )


def _occurrence_page(
    request: Request,
    session: Session,
    *,
    revision: str,
    filters: list[ColumnElement[bool]],
    needs_filter_joins: bool,
    sort: str,
    direction: MediaDirection,
    limit: int,
    cursor: str | None,
    scope: str,
) -> tuple[list[dict[str, Any]], str | None]:
    projection_needs_joins = needs_filter_joins or sort == "conversation"
    sorts: dict[str, Any] = {
        "date": AttachmentOccurrence.timestamp_ms,
        "size": AttachmentOccurrence.size_bytes,
        "type": _media_type_expression(),
        "filename": func.lower(AttachmentOccurrence.filename),
        "conversation": func.lower(Conversation.title_display),
        "availability": AttachmentOccurrence.availability,
        "occurrences": literal(1),
    }
    sort_expression = sorts[sort]
    null_rank = case((sort_expression.is_(None), 1), else_=0)
    projection_columns: list[Any] = [
        AttachmentOccurrence.id.label("id"),
        AttachmentOccurrence.id.label("group_key"),
        literal(1).label("occurrence_count"),
        literal(None).label("first_timestamp_ms"),
        literal(None).label("last_timestamp_ms"),
        sort_expression.label("sort_value"),
        null_rank.label("sort_null"),
    ]
    projection_statement = (
        select(*projection_columns)
        .select_from(AttachmentOccurrence)
        .prefix_with("/* media-occurrence-projection */")
    )
    if projection_needs_joins:
        projection_statement = _media_joins(
            select(*projection_columns).prefix_with("/* media-occurrence-projection */")
        )
    projection_statement = projection_statement.where(*filters)
    projection = _projection_cache(request).get_or_compute(
        namespace="occurrences",
        scope=scope,
        revision=revision,
        sort=sort,
        revision_is_current=lambda: catalog_revision(request, session) == revision,
        compute=lambda: _build_media_projection(session.execute(projection_statement)),
    )
    ordered_rows = projection.ascending if direction == "asc" else projection.descending
    page_start = 0
    if cursor is not None:
        try:
            decoded = decode_cursor(cursor, kind="media")
            if (
                require_cursor_str(decoded, "scope") != scope
                or require_cursor_str(decoded, "view") != "occurrences"
                or require_cursor_str(decoded, "sort") != sort
                or require_cursor_str(decoded, "direction") != direction
            ):
                raise InvalidCursor("Pagination cursor does not match the current media query")
            cursor_id = require_cursor_str(decoded, "id")
            cursor_null = require_cursor_int(decoded, "null")
            cursor_value = decoded.get("value")
            if cursor_null not in {0, 1} or not isinstance(cursor_value, (str, int, type(None))):
                raise InvalidCursor("Invalid pagination cursor")
            page_start = _projection_page_start(
                ordered_rows,
                cursor_null=cursor_null,
                cursor_value=cursor_value,
                cursor_id=cursor_id,
                direction=direction,
            )
        except InvalidCursor as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    bounded_rows = ordered_rows[page_start : page_start + limit + 1]
    has_more = len(bounded_rows) > limit
    page = bounded_rows[:limit]
    page_ids = [row.id for row in page]
    detail_statement = _media_joins(
        select(
            AttachmentOccurrence,
            Conversation.title_display.label("conversation_title"),
            ActivityRecord.id.label("activity_id"),
            ActivityRecord.title_display.label("activity_title"),
            ActivityRecord.category.label("activity_category"),
            ActivityRecord.subtype.label("activity_subtype"),
            _source_category_expression().label("source_category"),
            _media_type_expression().label("media_class"),
            ResolvedSourceChunk.chunk_key.label("source_chunk"),
            ResolvedSourceFile.relative_path.label("source_relative_path"),
            ResolvedSourceFile.extension.label("source_extension"),
            Blob.extension.label("blob_extension"),
        )
    ).where(AttachmentOccurrence.id.in_(page_ids))
    detail_by_id = (
        {detail[0].id: detail for detail in session.execute(detail_statement).all()}
        if page_ids
        else {}
    )
    items = [
        _occurrence_payload(
            detail[0],
            media_type=detail.media_class,
            conversation_title=detail.conversation_title,
            activity_id=detail.activity_id,
            activity_title=detail.activity_title,
            activity_category=detail.activity_category,
            activity_subtype=detail.activity_subtype,
            source_category=detail.source_category,
            source_chunk=detail.source_chunk,
            source_relative_path=detail.source_relative_path,
            source_extension=detail.source_extension,
            blob_extension=detail.blob_extension,
        )
        for row in page
        if (detail := detail_by_id.get(row.id)) is not None
    ]
    next_cursor = None
    if has_more and page:
        last = page[-1]
        next_cursor = encode_cursor(
            "media",
            scope=scope,
            view="occurrences",
            sort=sort,
            direction=direction,
            id=last.group_key,
            null=last.sort_null,
            value=last.sort_value,
        )
    return items, next_cursor


def _build_media_projection(rows: Iterable[Any]) -> _MediaProjection:
    projection_rows = tuple(
        _MediaProjectionRow(
            id=str(row.id),
            group_key=str(row.group_key),
            occurrence_count=int(row.occurrence_count),
            first_timestamp_ms=row.first_timestamp_ms,
            last_timestamp_ms=row.last_timestamp_ms,
            sort_value=row.sort_value,
            sort_null=int(row.sort_null),
        )
        for row in rows
    )
    non_null = [row for row in projection_rows if row.sort_null == 0]
    null = [row for row in projection_rows if row.sort_null == 1]

    def non_null_key(row: _MediaProjectionRow) -> tuple[Any, str]:
        return cast(Any, row.sort_value), row.group_key

    ascending = tuple(
        [*sorted(non_null, key=non_null_key), *sorted(null, key=lambda row: row.group_key)]
    )
    descending = tuple(
        [
            *sorted(non_null, key=non_null_key, reverse=True),
            *sorted(null, key=lambda row: row.group_key, reverse=True),
        ]
    )
    # Account every retained Python container/reference conservatively, even for shared singleton
    # integers/None, plus fixed headroom for the dataclass, digest key, and OrderedDict node.
    accounted_bytes = sys.getsizeof(ascending) + sys.getsizeof(descending) + 2048
    for row in projection_rows:
        accounted_bytes += sys.getsizeof(row)
        accounted_bytes += sum(
            sys.getsizeof(value)
            for value in (
                row.id,
                row.group_key,
                row.occurrence_count,
                row.first_timestamp_ms,
                row.last_timestamp_ms,
                row.sort_value,
                row.sort_null,
            )
        )
    return _MediaProjection(
        ascending=ascending,
        descending=descending,
        accounted_bytes=accounted_bytes,
    )


def _projection_page_start(
    rows: tuple[_MediaProjectionRow, ...],
    *,
    cursor_null: int,
    cursor_value: str | int | None,
    cursor_id: str,
    direction: MediaDirection,
) -> int:
    """Binary-seek to the first cached projection row strictly after an opaque cursor."""

    if cursor_null == 0 and cursor_value is None:
        # Match SQLite's `value > NULL OR value = NULL`: both expressions are unknown, leaving
        # only the subsequent null-rank segment eligible.
        low, high = 0, len(rows)
        while low < high:
            middle = (low + high) // 2
            if rows[middle].sort_null == 0:
                low = middle + 1
            else:
                high = middle
        return low

    def compare(row: _MediaProjectionRow) -> int:
        if row.sort_null != cursor_null:
            return -1 if row.sort_null < cursor_null else 1
        if cursor_null == 0:
            try:
                if row.sort_value != cursor_value:
                    result = -1 if row.sort_value < cursor_value else 1  # type: ignore[operator]
                    return result if direction == "asc" else -result
            except TypeError as exc:
                raise InvalidCursor("Invalid pagination cursor") from exc
        if row.group_key == cursor_id:
            return 0
        result = -1 if row.group_key < cursor_id else 1
        return result if direction == "asc" else -result

    low, high = 0, len(rows)
    while low < high:
        middle = (low + high) // 2
        if compare(rows[middle]) <= 0:
            low = middle + 1
        else:
            high = middle
    return low


def _unique_page(
    request: Request,
    session: Session,
    *,
    revision: str,
    filters: list[ColumnElement[bool]],
    needs_filter_joins: bool,
    sort: str,
    direction: MediaDirection,
    limit: int,
    cursor: str | None,
    scope: str,
) -> tuple[list[dict[str, Any]], str | None]:
    group_key = case(
        (
            AttachmentOccurrence.blob_sha256.is_not(None),
            literal("blob:") + AttachmentOccurrence.blob_sha256,
        ),
        else_=literal("occurrence:") + AttachmentOccurrence.id,
    )
    # Build one narrow grouped projection. The old query evaluated ROW_NUMBER and three other
    # windows across the complete filtered occurrence set for every cursor page. SQLite can group
    # these scalar columns directly, then select the exact representative from only the rows tied
    # at the group's latest non-null timestamp (or all rows when every timestamp is null). This
    # preserves the prior `(timestamp IS NULL, timestamp DESC, id ASC)` representative contract
    # without a full-set window sort. Rich display/source details are still fetched only for the
    # bounded representative page below.
    ranking_needs_joins = needs_filter_joins or sort == "conversation"
    representative_sort_values: dict[str, Any] = {
        "size": AttachmentOccurrence.size_bytes,
        "type": _media_type_expression(),
        "filename": func.lower(AttachmentOccurrence.filename),
        "conversation": func.lower(Conversation.title_display),
        "availability": AttachmentOccurrence.availability,
    }
    occurrence_columns: list[Any] = [
        AttachmentOccurrence.id.label("id"),
        AttachmentOccurrence.timestamp_ms.label("timestamp_ms"),
        group_key.label("group_key"),
        representative_sort_values.get(sort, literal(None)).label("representative_sort_value"),
    ]
    ranking_statement = select(*occurrence_columns).select_from(AttachmentOccurrence)
    if ranking_needs_joins:
        ranking_statement = _media_joins(select(*occurrence_columns))
    filtered = ranking_statement.where(*filters).cte("filtered_media_occurrences")

    grouped = (
        select(
            filtered.c.group_key,
            func.count().label("occurrence_count"),
            func.min(filtered.c.timestamp_ms).label("first_timestamp_ms"),
            func.max(filtered.c.timestamp_ms).label("last_timestamp_ms"),
        )
        .group_by(filtered.c.group_key)
        .cte("grouped_media_occurrences")
    )
    representative_selection = (
        select(
            filtered.c.group_key,
            func.min(filtered.c.id).label("representative_id"),
        )
        .join(grouped, grouped.c.group_key == filtered.c.group_key)
        .where(
            or_(
                and_(
                    grouped.c.last_timestamp_ms.is_not(None),
                    filtered.c.timestamp_ms == grouped.c.last_timestamp_ms,
                ),
                and_(
                    grouped.c.last_timestamp_ms.is_(None),
                    filtered.c.timestamp_ms.is_(None),
                ),
            )
        )
        .group_by(filtered.c.group_key)
        .cte("representative_media_occurrences")
    )
    representatives = (
        select(
            filtered.c.id,
            filtered.c.group_key,
            filtered.c.representative_sort_value,
            grouped.c.occurrence_count,
            grouped.c.first_timestamp_ms,
            grouped.c.last_timestamp_ms,
        )
        .join(
            representative_selection,
            and_(
                representative_selection.c.group_key == filtered.c.group_key,
                representative_selection.c.representative_id == filtered.c.id,
            ),
        )
        .join(grouped, grouped.c.group_key == filtered.c.group_key)
        .cte("unique_media_representatives")
    )
    sorts: dict[str, Any] = {
        "date": representatives.c.last_timestamp_ms,
        "size": representatives.c.representative_sort_value,
        "type": representatives.c.representative_sort_value,
        "filename": representatives.c.representative_sort_value,
        "conversation": representatives.c.representative_sort_value,
        "availability": representatives.c.representative_sort_value,
        "occurrences": representatives.c.occurrence_count,
    }
    sort_expression = sorts[sort]
    null_rank = case((sort_expression.is_(None), 1), else_=0)
    projection_statement = select(
        representatives,
        sort_expression.label("sort_value"),
        null_rank.label("sort_null"),
    )
    projection = _projection_cache(request).get_or_compute(
        namespace="unique",
        scope=scope,
        revision=revision,
        sort=sort,
        revision_is_current=lambda: catalog_revision(request, session) == revision,
        compute=lambda: _build_media_projection(session.execute(projection_statement)),
    )
    ordered_rows = projection.ascending if direction == "asc" else projection.descending
    page_start = 0
    if cursor is not None:
        try:
            decoded = decode_cursor(cursor, kind="media")
            if (
                require_cursor_str(decoded, "scope") != scope
                or require_cursor_str(decoded, "view") != "unique"
                or require_cursor_str(decoded, "sort") != sort
                or require_cursor_str(decoded, "direction") != direction
            ):
                raise InvalidCursor("Pagination cursor does not match the current media query")
            cursor_id = require_cursor_str(decoded, "id")
            cursor_null = require_cursor_int(decoded, "null")
            cursor_value = decoded.get("value")
            if cursor_null not in {0, 1} or not isinstance(cursor_value, (str, int, type(None))):
                raise InvalidCursor("Invalid pagination cursor")
        except InvalidCursor as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            page_start = _projection_page_start(
                ordered_rows,
                cursor_null=cursor_null,
                cursor_value=cursor_value,
                cursor_id=cursor_id,
                direction=direction,
            )
        except InvalidCursor as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    bounded_rows = ordered_rows[page_start : page_start + limit + 1]
    has_more = len(bounded_rows) > limit
    page = bounded_rows[:limit]
    page_representative_ids = [row.id for row in page]
    detail_statement = _media_joins(
        select(
            AttachmentOccurrence,
            Conversation.title_display.label("conversation_title"),
            ActivityRecord.id.label("activity_id"),
            ActivityRecord.title_display.label("activity_title"),
            ActivityRecord.category.label("activity_category"),
            ActivityRecord.subtype.label("activity_subtype"),
            _source_category_expression().label("source_category"),
            _media_type_expression().label("media_class"),
            ResolvedSourceChunk.chunk_key.label("source_chunk"),
            ResolvedSourceFile.relative_path.label("source_relative_path"),
            ResolvedSourceFile.extension.label("source_extension"),
            Blob.extension.label("blob_extension"),
        )
    ).where(AttachmentOccurrence.id.in_(page_representative_ids))
    detail_by_id = (
        {detail[0].id: detail for detail in session.execute(detail_statement).all()}
        if page_representative_ids
        else {}
    )
    items: list[dict[str, Any]] = []
    for row in page:
        detail = detail_by_id.get(row.id)
        if detail is None:
            continue
        occurrence = detail[0]
        items.append(
            {
                "id": occurrence.id,
                "view_key": row.group_key,
                "type": detail.media_class,
                "kind": occurrence.kind,
                "filename": occurrence.filename,
                "media_type": occurrence.media_type,
                "size_bytes": occurrence.size_bytes,
                "timestamp_ms": occurrence.timestamp_ms,
                "first_timestamp_ms": row.first_timestamp_ms,
                "last_timestamp_ms": row.last_timestamp_ms,
                "unknown_date": occurrence.timestamp_ms is None,
                "availability": occurrence.availability,
                "occurrence_count": row.occurrence_count,
                "conversation": (
                    {"id": occurrence.conversation_id, "title": detail.conversation_title}
                    if occurrence.conversation_id
                    else None
                ),
                "activity": (
                    {
                        "id": detail.activity_id,
                        "title": detail.activity_title,
                        "category": detail.activity_category,
                        "subtype": detail.activity_subtype,
                    }
                    if detail.activity_id is not None
                    else None
                ),
                "source_category": detail.source_category,
                "message_id": occurrence.message_id,
                "blob_sha256": occurrence.blob_sha256,
                "blob_href": (
                    f"/api/v1/blobs/{occurrence.blob_sha256}" if occurrence.blob_sha256 else None
                ),
                "occurrences_href": (
                    f"/api/v1/media/blobs/{occurrence.blob_sha256}/occurrences"
                    if occurrence.blob_sha256
                    else None
                ),
                "is_external_reference": occurrence.availability == "external",
                "jump": _jump(
                    occurrence.conversation_id,
                    occurrence.message_id,
                    detail.activity_id,
                    occurrence.id,
                ),
                "provenance": _unlinked_provenance(
                    occurrence,
                    source_chunk=detail.source_chunk,
                    source_relative_path=detail.source_relative_path,
                    source_extension=detail.source_extension,
                    blob_extension=detail.blob_extension,
                ),
            }
        )
    next_cursor = None
    if has_more and page:
        last = page[-1]
        next_cursor = encode_cursor(
            "media",
            scope=scope,
            view="unique",
            sort=sort,
            direction=direction,
            id=last.group_key,
            null=last.sort_null,
            value=last.sort_value,
        )
    return items, next_cursor


@media_router.get("")
def media(
    request: Request,
    view: MediaView = "occurrences",
    occurrence: str | None = Query(default=None, max_length=128),
    q: str | None = Query(default=None, max_length=300),
    type: str | None = Query(default=None, max_length=200),
    kind: str | None = Query(default=None, max_length=300),
    availability: str | None = Query(default=None, max_length=200),
    conversation_id: str | None = Query(default=None, max_length=128),
    conversation_q: str | None = Query(default=None, max_length=200),
    conversation_type: str | None = Query(default=None, max_length=200),
    source_category: str | None = Query(default=None, max_length=500),
    from_ms: int | None = Query(default=None, ge=0),
    to_ms: int | None = Query(default=None, ge=0),
    min_size: int | None = Query(default=None, ge=0),
    max_size: int | None = Query(default=None, ge=0),
    sort: str = Query(default="date"),
    direction: MediaDirection = "desc",
    limit: int = Query(default=60, ge=1, le=250),
    cursor: str | None = Query(default=None, max_length=2048),
) -> dict[str, Any]:
    if sort not in MEDIA_SORT_NAMES:
        raise HTTPException(status_code=422, detail=f"Unknown media sort: {sort}")
    if from_ms is not None and to_ms is not None and from_ms > to_ms:
        raise HTTPException(status_code=422, detail="Invalid date range")
    if min_size is not None and max_size is not None and min_size > max_size:
        raise HTTPException(status_code=422, detail="Invalid size range")
    filters, types, availability_values, needs_summary_joins = _media_filters(
        occurrence_id=occurrence,
        q=q,
        media_type=type,
        kind=kind,
        availability=availability,
        conversation_id=conversation_id,
        conversation_q=conversation_q,
        conversation_type=conversation_type,
        source_category=source_category,
        from_ms=from_ms,
        to_ms=to_ms,
        min_size=min_size,
        max_size=max_size,
    )
    scope = _scope_hash(
        {
            "occurrence": occurrence,
            "q": normalize_text(q).search if q else None,
            "type": types,
            "kind": _csv_values(kind),
            "availability": availability_values,
            "conversation_id": conversation_id,
            "conversation_q": normalize_text(conversation_q).search if conversation_q else None,
            "conversation_type": _csv_values(conversation_type),
            "source_category": _csv_values(source_category),
            "from_ms": from_ms,
            "to_ms": to_ms,
            "min_size": min_size,
            "max_size": max_size,
        }
    )
    with _session(request) as session:
        aggregate_memo = catalog_aggregate_memo(request, session)
        summary_aggregate = aggregate_memo.get_or_compute(
            namespace="media-summary-v1",
            scope=scope,
            compute=lambda: _base_summary(
                session,
                filters=filters,
                needs_joins=needs_summary_joins,
            ),
        )
        summary, facets = summary_aggregate.payload()
        page_cache_key = ":".join(
            (
                scope,
                aggregate_memo.revision,
                view,
                sort,
                direction,
                str(limit),
                cursor or "first",
            )
        )
        cached_page = _page_cache(request).get(page_cache_key)
        if cached_page is not None:
            items, next_cursor = cached_page
        else:
            if view == "occurrences":
                items, next_cursor = _occurrence_page(
                    request,
                    session,
                    revision=aggregate_memo.revision,
                    filters=filters,
                    needs_filter_joins=needs_summary_joins,
                    sort=sort,
                    direction=direction,
                    limit=limit,
                    cursor=cursor,
                    scope=scope,
                )
            else:
                items, next_cursor = _unique_page(
                    request,
                    session,
                    revision=aggregate_memo.revision,
                    filters=filters,
                    needs_filter_joins=needs_summary_joins,
                    sort=sort,
                    direction=direction,
                    limit=limit,
                    cursor=cursor,
                    scope=scope,
                )
            _page_cache(request).set(page_cache_key, items, next_cursor)
        derivative_map = derivatives_by_blob(session, (item.get("blob_sha256") for item in items))
        for item in items:
            blob_sha256 = item.get("blob_sha256")
            item["derivatives"] = (
                derivative_map.get(blob_sha256, []) if isinstance(blob_sha256, str) else []
            )
        attach_star_states(
            session,
            items,
            lambda item: (
                ("blob", str(item["blob_sha256"]))
                if view == "unique" and item.get("blob_sha256")
                else ("attachment", str(item["id"]))
            ),
        )
        aggregate_memo.require_current()
    total = summary["occurrences"] if view == "occurrences" else summary["unique_files"]
    return {
        "items": items,
        "page": {"limit": limit, "total": total, "next_cursor": next_cursor},
        "view": view,
        "sort": {"field": sort, "direction": direction},
        "summary": summary,
        "facets": facets,
    }


@media_router.get("/blobs/{sha256}/occurrences")
def blob_occurrences(
    request: Request,
    sha256: str,
    limit: int = Query(default=100, ge=1, le=250),
    cursor: str | None = Query(default=None, max_length=2048),
) -> dict[str, Any]:
    if len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Media blob not found")
    statement = _media_joins(
        select(
            AttachmentOccurrence,
            Conversation.title_display.label("conversation_title"),
            ActivityRecord.id.label("activity_id"),
            ActivityRecord.title_display.label("activity_title"),
            ActivityRecord.category.label("activity_category"),
            ActivityRecord.subtype.label("activity_subtype"),
            _source_category_expression().label("source_category"),
            _media_type_expression().label("media_class"),
            ResolvedSourceChunk.chunk_key.label("source_chunk"),
            ResolvedSourceFile.relative_path.label("source_relative_path"),
            ResolvedSourceFile.extension.label("source_extension"),
            Blob.extension.label("blob_extension"),
        )
    ).where(
        AttachmentOccurrence.blob_sha256 == sha256,
        AttachmentOccurrence.deleted_at.is_(None),
    )
    if cursor is not None:
        try:
            decoded = decode_cursor(cursor, kind="blob-occurrences")
            if require_cursor_str(decoded, "sha256") != sha256:
                raise InvalidCursor("Pagination cursor belongs to another blob")
            cursor_null = require_cursor_int(decoded, "null")
            timestamp_ms = require_cursor_int(decoded, "timestamp_ms")
            occurrence_id = require_cursor_str(decoded, "id")
            if cursor_null not in {0, 1}:
                raise InvalidCursor("Invalid pagination cursor")
        except InvalidCursor as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        statement = statement.where(
            and_(
                AttachmentOccurrence.timestamp_ms.is_(None),
                AttachmentOccurrence.id < occurrence_id,
            )
            if cursor_null == 1
            else or_(
                AttachmentOccurrence.timestamp_ms < timestamp_ms,
                and_(
                    AttachmentOccurrence.timestamp_ms == timestamp_ms,
                    AttachmentOccurrence.id < occurrence_id,
                ),
                AttachmentOccurrence.timestamp_ms.is_(None),
            )
        )
    with _session(request) as session:
        rows = session.execute(
            statement.order_by(
                AttachmentOccurrence.timestamp_ms.is_(None),
                AttachmentOccurrence.timestamp_ms.desc(),
                AttachmentOccurrence.id.desc(),
            ).limit(limit + 1)
        ).all()
        total = (
            session.scalar(
                select(func.count(AttachmentOccurrence.id)).where(
                    AttachmentOccurrence.blob_sha256 == sha256,
                    AttachmentOccurrence.deleted_at.is_(None),
                )
            )
            or 0
        )
        if total == 0:
            raise HTTPException(status_code=404, detail="Media blob has no retained occurrences")
        has_more = len(rows) > limit
        page = rows[:limit]
        items = [
            _occurrence_payload(
                row[0],
                media_type=row.media_class,
                conversation_title=row.conversation_title,
                activity_id=row.activity_id,
                activity_title=row.activity_title,
                activity_category=row.activity_category,
                activity_subtype=row.activity_subtype,
                source_category=row.source_category,
                source_chunk=row.source_chunk,
                source_relative_path=row.source_relative_path,
                source_extension=row.source_extension,
                blob_extension=row.blob_extension,
            )
            for row in page
        ]
        attach_star_states(
            session,
            items,
            lambda item: ("attachment", str(item["id"])),
        )
    next_cursor = None
    if has_more and page:
        last = page[-1][0]
        next_cursor = encode_cursor(
            "blob-occurrences",
            sha256=sha256,
            null=1 if last.timestamp_ms is None else 0,
            timestamp_ms=last.timestamp_ms or 0,
            id=last.id,
        )
    return {
        "blob_sha256": sha256,
        "items": items,
        "page": {"limit": limit, "total": total, "next_cursor": next_cursor},
    }
