from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import (
    Integer,
    String,
    and_,
    case,
    cast,
    column,
    func,
    literal,
    or_,
    select,
    table,
    text,
    tuple_,
    union_all,
)
from sqlalchemy.sql.elements import ColumnElement

from meta_archive.api.aggregate_cache import catalog_aggregate_memo
from meta_archive.api.pagination import (
    InvalidCursor,
    decode_cursor,
    encode_cursor,
    require_cursor_int,
    require_cursor_str,
)
from meta_archive.api.security import require_session
from meta_archive.api.stage4_common import (
    coverage_payload,
    csv_values,
    escaped_like,
    invalid_cursor,
    scope_hash,
    session_for,
    unavailable_page,
)
from meta_archive.models import (
    ActivityRecord,
    AttachmentOccurrence,
    Conversation,
    ConversationSegment,
    Message,
    SourceChunk,
    SourceFile,
    SourceRecord,
)
from meta_archive.services.jumps import activity_jump as _activity_jump
from meta_archive.services.search import _fts_expression
from meta_archive.services.stars import attach_star_states
from meta_archive.services.text import normalize_classifier_token, normalize_text

activity_router = APIRouter(prefix="/api/v1", dependencies=[Depends(require_session)])

ActivityDirection = Literal["asc", "desc"]
ActivitySort = Literal["date", "title", "actor", "category", "subtype"]
ActivityView = Literal[
    "posts", "comments", "groups", "marketplace", "pages", "check-ins", "support"
]
POST_CONTEXT_VALUES = frozenset(
    {
        "own_wall",
        "others_wall",
        "group",
        "reshared",
        "tagged",
        "others_on_your_wall",
        "unknown",
    }
)
_OPAQUE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_EXPLICIT_ACTIVITY_FIELDS = (
    "activity_type",
    "post_type",
    "comment_type",
    "story_type",
    "timeline_type",
    "wall_type",
    "post_context",
    "comment_context",
    "surface",
)
_POST_CONTEXT_ALIASES: dict[str, frozenset[str]] = {
    "own_wall": frozenset(
        {
            "own_wall",
            "your_wall",
            "your_timeline",
            "timeline_self",
            "posted_by_you",
            "your_posts",
            "posts_by_you",
            "your_wall_posts",
        }
    ),
    "others_wall": frozenset(
        {
            "others_wall",
            "other_wall",
            "another_timeline",
            "posts_on_others_walls",
            "posts_on_other_peoples_timelines",
            "posts_to_other_profiles",
        }
    ),
    "group": frozenset(
        {
            "group",
            "group_post",
            "group_comment",
            "posts_in_groups",
            "comments_in_groups",
        }
    ),
    "reshared": frozenset(
        {
            "reshared",
            "reshare",
            "reshared_post",
            "shared_post",
            "shared_posts",
            "repost",
            "reposts",
        }
    ),
    "tagged": frozenset(
        {
            "tagged",
            "tagged_post",
            "tagged_posts",
            "posts_youre_tagged_in",
            "mentioned",
        }
    ),
    "others_on_your_wall": frozenset(
        {
            "others_on_your_wall",
            "others_on_your_timeline",
            "visitor_post",
            "visitor_posts",
            "others_posts_to_your_timeline",
            "posts_by_others",
        }
    ),
}


class ActivityQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    q: str | None = Field(default=None, max_length=300)
    category: str | None = Field(default=None, max_length=500)
    subtype: str | None = Field(default=None, max_length=500)
    post_context: str | None = Field(default=None, max_length=500)
    availability: str | None = Field(default=None, max_length=200)
    from_ms: int | None = Field(default=None, ge=0)
    to_ms: int | None = Field(default=None, ge=0)
    has_attachments: bool | None = None
    logical_only: bool = False
    sort: ActivitySort = "date"
    direction: ActivityDirection = "desc"
    limit: int = Field(default=50, ge=1, le=250)
    cursor: str | None = Field(default=None, max_length=2048)


_VIEW_COVERAGE: dict[str, tuple[str, str, str, str]] = {
    "posts": (
        "posts",
        "Posts, albums, and check-ins",
        "Only exported post-related records are shown; the general wall-post manifest is absent.",
        "partial",
    ),
    "comments": (
        "group_comments",
        "Comments",
        "Only comments present in exported activity sources can be shown.",
        "partial",
    ),
    "groups": (
        "activity_groups",
        "Groups",
        "No group activity records are available in this export.",
        "partial",
    ),
    "marketplace": (
        "activity_marketplace",
        "Marketplace",
        "No Marketplace activity records are available in this export.",
        "available",
    ),
    "pages": (
        "activity_pages",
        "Pages",
        "No Pages activity records are available in this export.",
        "partial",
    ),
    "check-ins": (
        "check_ins",
        "Check-ins",
        "No explicit check-in records are available in this export.",
        "partial",
    ),
    "support": (
        "activity_support_cases",
        "Support cases",
        "No non-message support-case activity is available in this export.",
        "partial",
    ),
}


def _normalize_activity_token(value: str) -> str:
    return normalize_classifier_token(value)


def _post_context_for_row(row: ActivityRecord) -> str:
    raw = row.extra_json.get("raw_payload")
    if isinstance(raw, dict):
        for field in _EXPLICIT_ACTIVITY_FIELDS:
            value = raw.get(field)
            if not isinstance(value, str) or not value.strip():
                continue
            normalized = _normalize_activity_token(value)
            for post_context, aliases in _POST_CONTEXT_ALIASES.items():
                if normalized in aliases:
                    return post_context
            # An explicit but unfamiliar source value stays unknown. Do not override it using
            # titles, actors, or other inferred text.
            return "unknown"

    subtype = _normalize_activity_token(row.subtype)
    for post_context, aliases in _POST_CONTEXT_ALIASES.items():
        if subtype in aliases:
            return post_context
    if row.category == "groups" and subtype in {"group_comment", "group_post"}:
        return "group"
    return "unknown"


def _post_context_basis_for_row(row: ActivityRecord) -> str:
    raw = row.extra_json.get("raw_payload")
    if isinstance(raw, dict) and any(
        isinstance(raw.get(field), str) and bool(raw[field].strip())
        for field in _EXPLICIT_ACTIVITY_FIELDS
    ):
        return "explicit_source_field"
    subtype = _normalize_activity_token(row.subtype)
    if any(subtype in aliases for aliases in _POST_CONTEXT_ALIASES.values()):
        return "source_subtype"
    if row.category == "groups" and subtype in {"group_comment", "group_post"}:
        return "source_category_and_subtype"
    return "not_explicit_in_source"


def _post_context_expression() -> ColumnElement[str]:
    paths = [f"$.raw_payload.{field}" for field in _EXPLICIT_ACTIVITY_FIELDS]
    # Match the row serializer: use the first nonblank string field, ignoring JSON nulls,
    # numbers, and blank earlier fields rather than letting COALESCE mask a later value.
    raw_value = case(
        *(
            (
                and_(
                    func.json_type(ActivityRecord.extra_json, path) == "text",
                    func.trim(cast(func.json_extract(ActivityRecord.extra_json, path), String))
                    != "",
                ),
                cast(func.json_extract(ActivityRecord.extra_json, path), String),
            )
            for path in paths
        ),
        else_="",
    )
    normalized_raw = func.normalize_classifier_token(raw_value)
    subtype = func.normalize_classifier_token(ActivityRecord.subtype)
    explicit_cases = [
        (normalized_raw.in_(tuple(sorted(aliases))), post_context)
        for post_context, aliases in _POST_CONTEXT_ALIASES.items()
    ]
    subtype_cases = [
        (
            and_(raw_value == "", subtype.in_(tuple(sorted(aliases)))),
            post_context,
        )
        for post_context, aliases in _POST_CONTEXT_ALIASES.items()
    ]
    return case(
        *explicit_cases,
        *subtype_cases,
        (
            and_(
                raw_value == "",
                ActivityRecord.category == "groups",
                subtype.in_(("group_comment", "group_post")),
            ),
            "group",
        ),
        else_="unknown",
    )


def _view_filter(view: str) -> ColumnElement[bool]:
    subtype = func.lower(ActivityRecord.subtype)
    source_path = func.lower(SourceFile.normalized_path)
    if view == "posts":
        return ActivityRecord.category == "posts"
    if view == "comments":
        return subtype.like("%comment%")
    if view == "groups":
        return ActivityRecord.category == "groups"
    if view == "marketplace":
        return ActivityRecord.category == "marketplace"
    if view == "pages":
        return ActivityRecord.category == "pages"
    if view == "check-ins":
        return or_(subtype.like("%check%in%"), source_path.like("%check%in%"))
    if view == "support":
        return ActivityRecord.category.in_(("support", "support_cases"))
    raise HTTPException(status_code=422, detail=f"Unknown activity view: {view}")


def _activity_filters(query: ActivityQuery, *, view: str | None) -> list[ColumnElement[bool]]:
    if query.from_ms is not None and query.to_ms is not None and query.from_ms > query.to_ms:
        raise HTTPException(status_code=422, detail="Invalid activity date range")
    filters: list[ColumnElement[bool]] = [ActivityRecord.deleted_at.is_(None)]
    if view is not None:
        filters.append(_view_filter(view))
    categories = csv_values(query.category)
    if categories:
        filters.append(ActivityRecord.category.in_(categories))
    subtypes = csv_values(query.subtype)
    if subtypes:
        filters.append(ActivityRecord.subtype.in_(subtypes))
    post_contexts = csv_values(query.post_context)
    invalid_post_contexts = set(post_contexts) - POST_CONTEXT_VALUES
    if invalid_post_contexts:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown post context filter: {sorted(invalid_post_contexts)[0]}",
        )
    if post_contexts:
        if view not in {None, "posts", "comments"}:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Typed activity filters are available only for all activity, posts, or comments"
                ),
            )
        filters.append(_post_context_expression().in_(post_contexts))
    availability = csv_values(query.availability)
    if availability:
        filters.append(ActivityRecord.availability.in_(availability))
    if query.from_ms is not None:
        filters.append(ActivityRecord.timestamp_ms >= query.from_ms)
    if query.to_ms is not None:
        filters.append(ActivityRecord.timestamp_ms <= query.to_ms)
    if query.q:
        normalized = escaped_like(normalize_text(query.q).search)
        filters.append(
            or_(
                ActivityRecord.title_search.like(f"%{normalized}%", escape="\\"),
                ActivityRecord.content_search.like(f"%{normalized}%", escape="\\"),
                ActivityRecord.actor_search.like(f"%{normalized}%", escape="\\"),
            )
        )
    attachment_exists = select(AttachmentOccurrence.id).where(
        AttachmentOccurrence.activity_record_id == ActivityRecord.id,
        AttachmentOccurrence.deleted_at.is_(None),
    )
    if query.has_attachments is True:
        filters.append(attachment_exists.exists())
    elif query.has_attachments is False:
        filters.append(~attachment_exists.exists())
    if query.logical_only:
        filters.append(
            ActivityRecord.source_record_id.in_(
                select(SourceRecord.id).where(SourceRecord.duplicate_of_id.is_(None))
            )
        )
    return filters


def _activity_sort(sort: ActivitySort) -> Any:
    return {
        "date": ActivityRecord.timestamp_ms,
        "title": func.coalesce(ActivityRecord.title_search, ActivityRecord.content_search, ""),
        "actor": func.coalesce(ActivityRecord.actor_search, ""),
        "category": ActivityRecord.category,
        "subtype": ActivityRecord.subtype,
    }[sort]


def _seek_condition(
    *,
    null_rank: ColumnElement[int],
    value: ColumnElement[Any],
    cursor_null: int,
    cursor_value: str | int,
    cursor_id: str,
    direction: ActivityDirection,
) -> ColumnElement[bool]:
    id_seek = ActivityRecord.id > cursor_id if direction == "asc" else ActivityRecord.id < cursor_id
    if cursor_null == 1:
        return and_(null_rank == 1, id_seek)
    value_seek = value > cursor_value if direction == "asc" else value < cursor_value
    return or_(
        null_rank > cursor_null,
        and_(null_rank == cursor_null, or_(value_seek, and_(value == cursor_value, id_seek))),
    )


def _activity_payload(
    row: ActivityRecord,
    *,
    chunk_key: str,
    relative_path: str,
    attachment_count: int,
) -> dict[str, Any]:
    post_context = _post_context_for_row(row)
    return {
        "id": row.id,
        "parent_id": row.parent_id,
        "category": row.category,
        "subtype": row.subtype,
        "post_context": post_context,
        "post_context_basis": _post_context_basis_for_row(row),
        "timestamp_ms": row.timestamp_ms,
        "unknown_date": row.timestamp_ms is None,
        "title": row.title_display,
        "content": row.content_display,
        "actor": row.actor_display,
        "availability": row.availability,
        "target_uri": row.target_uri,
        "source_fbid": row.source_fbid,
        "attachment_count": attachment_count,
        "source": {
            "source_file_id": row.source_file_id,
            "source_record_id": row.source_record_id,
            "chunk": chunk_key,
            "relative_path": relative_path,
            "json_pointer": row.json_pointer,
        },
        "jump": _activity_jump(row.id),
    }


def _activity_page(
    request: Request,
    query: ActivityQuery,
    *,
    view: str | None,
) -> dict[str, Any]:
    filters = _activity_filters(query, view=view)
    scope = scope_hash(
        {
            "view": view,
            "q": query.q or "",
            "category": csv_values(query.category),
            "subtype": csv_values(query.subtype),
            "post_context": csv_values(query.post_context),
            "availability": csv_values(query.availability),
            "from_ms": query.from_ms,
            "to_ms": query.to_ms,
            "has_attachments": query.has_attachments,
            "logical_only": query.logical_only,
        }
    )
    sort_value = _activity_sort(query.sort)
    null_rank = case((sort_value.is_(None), 1), else_=0)
    statement = (
        select(ActivityRecord, SourceChunk.chunk_key, SourceFile.relative_path)
        .join(SourceFile, SourceFile.id == ActivityRecord.source_file_id)
        .join(SourceChunk, SourceChunk.id == SourceFile.chunk_id)
        .where(*filters)
    )
    if query.cursor:
        try:
            decoded = decode_cursor(query.cursor, kind="activity")
            if (
                require_cursor_str(decoded, "sort") != query.sort
                or require_cursor_str(decoded, "direction") != query.direction
                or require_cursor_str(decoded, "scope") != scope
            ):
                raise InvalidCursor("Pagination cursor does not match these activity filters")
            cursor_null = require_cursor_int(decoded, "null")
            cursor_value: str | int = (
                require_cursor_int(decoded, "value")
                if query.sort == "date"
                else require_cursor_str(decoded, "value")
            )
            cursor_id = require_cursor_str(decoded, "id")
        except InvalidCursor as exc:
            raise invalid_cursor(str(exc)) from exc
        statement = statement.where(
            _seek_condition(
                null_rank=null_rank,
                value=sort_value,
                cursor_null=cursor_null,
                cursor_value=cursor_value,
                cursor_id=cursor_id,
                direction=query.direction,
            )
        )
    value_order = sort_value.asc() if query.direction == "asc" else sort_value.desc()
    id_order = ActivityRecord.id.asc() if query.direction == "asc" else ActivityRecord.id.desc()
    statement = statement.order_by(null_rank.asc(), value_order, id_order).limit(query.limit + 1)
    requested_post_contexts = csv_values(query.post_context)

    with session_for(request) as session:
        aggregate_memo = catalog_aggregate_memo(request, session)
        source_filters = _activity_filters(ActivityQuery(), view=view)

        def load_source_summary() -> tuple[int, tuple[tuple[str, int], ...]]:
            loaded_source_total = int(
                session.scalar(
                    select(func.count(ActivityRecord.pk))
                    .select_from(ActivityRecord)
                    .join(SourceFile, SourceFile.id == ActivityRecord.source_file_id)
                    .where(*source_filters)
                )
                or 0
            )
            loaded_context_counts = {
                post_context: 0 for post_context in sorted(POST_CONTEXT_VALUES)
            }
            if view in {None, "posts", "comments"}:
                context_expression = _post_context_expression()
                for post_context, count in session.execute(
                    select(context_expression, func.count(ActivityRecord.pk))
                    .select_from(ActivityRecord)
                    .join(SourceFile, SourceFile.id == ActivityRecord.source_file_id)
                    .where(*source_filters)
                    .group_by(context_expression)
                ):
                    loaded_context_counts[str(post_context)] = int(count)
            return loaded_source_total, tuple(sorted(loaded_context_counts.items()))

        source_total, context_count_pairs = aggregate_memo.get_or_compute(
            namespace="activity-source-summary-v1",
            scope=scope_hash({"view": view}),
            compute=load_source_summary,
        )
        context_counts = dict(context_count_pairs)
        total = aggregate_memo.get_or_compute(
            namespace="activity-filter-total-v1",
            scope=scope,
            compute=lambda: int(
                session.scalar(
                    select(func.count(ActivityRecord.pk))
                    .select_from(ActivityRecord)
                    .join(SourceFile, SourceFile.id == ActivityRecord.source_file_id)
                    .where(*filters)
                )
                or 0
            ),
        )
        rows = list(session.execute(statement))
        page_rows = rows[: query.limit]
        activity_ids = [row.id for row, _chunk, _path in page_rows]
        attachment_counts = {
            activity_id: count
            for activity_id, count in session.execute(
                select(
                    AttachmentOccurrence.activity_record_id,
                    func.count(AttachmentOccurrence.id),
                )
                .where(
                    AttachmentOccurrence.activity_record_id.in_(activity_ids),
                    AttachmentOccurrence.deleted_at.is_(None),
                )
                .group_by(AttachmentOccurrence.activity_record_id)
            )
            if activity_id is not None
        }
        items = [
            _activity_payload(
                row,
                chunk_key=chunk,
                relative_path=path,
                attachment_count=attachment_counts.get(row.id, 0),
            )
            for row, chunk, path in page_rows
        ]
        attach_star_states(
            session,
            items,
            lambda item: ("activity", str(item["id"])),
        )
        coverage_components: list[dict[str, Any]] = []
        if view is None:
            coverage = coverage_payload(
                session,
                "activity_all",
                label="Facebook activity",
                unavailable_description="No supported non-message activity is available.",
                available_description=(
                    "Retained non-message Facebook activity records are available; coverage "
                    "remains partial where export categories are absent."
                ),
                actual_count=source_total,
                status_with_data="partial",
            )
        else:
            key, label, description, status_with_data = _VIEW_COVERAGE[view]
            coverage = coverage_payload(
                session,
                key,
                label=label,
                unavailable_description=description,
                actual_count=source_total,
                status_with_data=status_with_data,
            )
            if view == "posts":
                coverage_components.append(
                    coverage_payload(
                        session,
                        "general_posts",
                        label="General posts",
                        unavailable_description=(
                            "No general wall-post manifest is available in this export."
                        ),
                    )
                )
        context_matches = sum(context_counts.get(value, 0) for value in requested_post_contexts)
        if not requested_post_contexts:
            context_status = "summary"
            context_description = (
                "Types classify retained records only from explicit source fields or explicit "
                "source schema categories and subtypes; unknown means the export did not state "
                "a supported type."
            )
        elif source_total == 0 and coverage["status"] == "unavailable":
            context_status = "unavailable"
            context_description = (
                "This activity source is unavailable, so the requested history cannot be counted."
            )
        elif context_matches:
            context_status = "source_backed"
            context_description = (
                "Matching retained records declare the requested context through an explicit "
                "source field, category, or subtype."
            )
        else:
            context_status = "unknown"
            context_description = (
                "No retained source field, category, or subtype declares the requested context. "
                "This does not establish that the history was empty."
            )
        post_context_coverage = {
            "status": context_status,
            "requested": list(requested_post_contexts),
            "counts": context_counts,
            "classification": "source_fields_categories_and_subtypes",
            "description": context_description,
            "absence_is_not_zero_history": context_status in {"unavailable", "unknown"},
        }
        aggregate_memo.require_current()
    next_cursor: str | None = None
    if len(rows) > query.limit and page_rows:
        last = page_rows[-1][0]
        raw_value = (
            last.timestamp_ms
            if query.sort == "date"
            else str(
                {
                    "title": last.title_search or last.content_search or "",
                    "actor": last.actor_search or "",
                    "category": last.category,
                    "subtype": last.subtype,
                }[query.sort]
            )
        )
        next_cursor = encode_cursor(
            "activity",
            sort=query.sort,
            direction=query.direction,
            scope=scope,
            null=1 if raw_value is None else 0,
            value=0 if raw_value is None else raw_value,
            id=last.id,
        )
    return {
        "coverage": coverage,
        "coverage_components": coverage_components,
        "post_context_coverage": post_context_coverage,
        "items": items,
        "page": {
            "total": (None if source_total == 0 and coverage["status"] == "unavailable" else total),
            "limit": query.limit,
            "next_cursor": next_cursor,
            "count_is_unknown": source_total == 0 and coverage["status"] == "unavailable",
        },
        "filters": {"view": view, **query.model_dump(exclude={"cursor"})},
    }


@activity_router.get("/activity")
def activity(
    request: Request,
    query: Annotated[ActivityQuery, Query()],
) -> dict[str, Any]:
    return _activity_page(request, query, view=None)


@activity_router.get("/activity/views/{view}")
def activity_view(
    request: Request,
    view: ActivityView,
    query: Annotated[ActivityQuery, Query()],
) -> dict[str, Any]:
    return _activity_page(request, query, view=view)


@activity_router.get("/activity/records/{activity_id}")
def activity_detail(request: Request, activity_id: str) -> dict[str, Any]:
    with session_for(request) as session:
        result = session.execute(
            select(ActivityRecord, SourceChunk.chunk_key, SourceFile.relative_path)
            .join(SourceFile, SourceFile.id == ActivityRecord.source_file_id)
            .join(SourceChunk, SourceChunk.id == SourceFile.chunk_id)
            .where(ActivityRecord.id == activity_id, ActivityRecord.deleted_at.is_(None))
        ).one_or_none()
        if result is None:
            raise HTTPException(status_code=404, detail="Activity record not found")
        row, chunk, path = result
        attachments = list(
            session.scalars(
                select(AttachmentOccurrence)
                .where(
                    AttachmentOccurrence.activity_record_id == row.id,
                    AttachmentOccurrence.deleted_at.is_(None),
                )
                .order_by(AttachmentOccurrence.ordinal, AttachmentOccurrence.id)
            )
        )
        children = list(
            session.scalars(
                select(ActivityRecord)
                .where(ActivityRecord.parent_id == row.id, ActivityRecord.deleted_at.is_(None))
                .order_by(ActivityRecord.timestamp_ms, ActivityRecord.id)
            )
        )
        record_payload = _activity_payload(
            row,
            chunk_key=chunk,
            relative_path=path,
            attachment_count=len(attachments),
        )
        attachment_payloads = [
            {
                "id": attachment.id,
                "kind": attachment.kind,
                "filename": attachment.filename,
                "media_type": attachment.media_type,
                "size_bytes": attachment.size_bytes,
                "timestamp_ms": attachment.timestamp_ms,
                "availability": attachment.availability,
                "blob_sha256": attachment.blob_sha256,
                "blob_href": (
                    f"/api/v1/blobs/{attachment.blob_sha256}" if attachment.blob_sha256 else None
                ),
            }
            for attachment in attachments
        ]
        child_payloads = [
            {
                "id": child.id,
                "subtype": child.subtype,
                "timestamp_ms": child.timestamp_ms,
                "title": child.title_display,
                "jump": _activity_jump(child.id),
            }
            for child in children
        ]
        attach_star_states(
            session,
            [record_payload, *child_payloads],
            lambda item: ("activity", str(item["id"])),
        )
        attach_star_states(
            session,
            attachment_payloads,
            lambda item: ("attachment", str(item["id"])),
        )
    return {
        "record": record_payload,
        "attachments": attachment_payloads,
        "children": child_payloads,
    }


def _timeline_filters(
    *, q: str | None, from_ms: int | None, to_ms: int | None
) -> tuple[list[ColumnElement[bool]], list[ColumnElement[bool]]]:
    message_filters: list[ColumnElement[bool]] = [Message.deleted_at.is_(None)]
    activity_filters: list[ColumnElement[bool]] = [ActivityRecord.deleted_at.is_(None)]
    if from_ms is not None:
        message_filters.append(Message.timestamp_ms >= from_ms)
        activity_filters.append(ActivityRecord.timestamp_ms >= from_ms)
    if to_ms is not None:
        message_filters.append(Message.timestamp_ms <= to_ms)
        activity_filters.append(ActivityRecord.timestamp_ms <= to_ms)
    if q:
        try:
            fts_query = _fts_expression(q)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        message_fts = table("message_fts", column("rowid", Integer))
        activity_fts = table("activity_fts", column("rowid", Integer))
        message_filters.append(
            Message.pk.in_(
                select(message_fts.c.rowid)
                .where(text("message_fts MATCH :timeline_message_fts_query"))
                .params(
                    timeline_message_fts_query=(f"{{sender_search content_search}} : ({fts_query})")
                )
            )
        )
        activity_filters.append(
            ActivityRecord.pk.in_(
                select(activity_fts.c.rowid)
                .where(text("activity_fts MATCH :timeline_activity_fts_query"))
                .params(
                    timeline_activity_fts_query=(
                        f"{{title_search content_search actor_search}} : ({fts_query})"
                    )
                )
            )
        )
    return message_filters, activity_filters


def _timeline_union(
    *,
    q: str | None,
    entity_types: tuple[str, ...],
    from_ms: int | None,
    to_ms: int | None,
) -> Any:
    message_filters, activity_filters = _timeline_filters(q=q, from_ms=from_ms, to_ms=to_ms)
    selects: list[Any] = []
    if not entity_types or "message" in entity_types:
        selects.append(
            select(
                literal("message").label("entity_type"),
                Message.id.label("entity_id"),
                Message.timestamp_ms.label("timestamp_ms"),
                Message.sender_display.label("title"),
                Message.content_display.label("content"),
                literal("messages").label("category"),
                Message.message_kind.label("subtype"),
                Message.source_file_id.label("source_file_id"),
                Message.json_pointer.label("json_pointer"),
                Message.conversation_id.label("conversation_id"),
            ).where(*message_filters)
        )
    if not entity_types or "activity" in entity_types:
        selects.append(
            select(
                literal("activity").label("entity_type"),
                ActivityRecord.id.label("entity_id"),
                ActivityRecord.timestamp_ms.label("timestamp_ms"),
                ActivityRecord.title_display.label("title"),
                ActivityRecord.content_display.label("content"),
                ActivityRecord.category.label("category"),
                ActivityRecord.subtype.label("subtype"),
                ActivityRecord.source_file_id.label("source_file_id"),
                ActivityRecord.json_pointer.label("json_pointer"),
                literal(None).label("conversation_id"),
            ).where(*activity_filters)
        )
    if not selects:
        raise HTTPException(
            status_code=422, detail="Timeline entity_type must be message or activity"
        )
    return union_all(*selects).subquery("timeline_entries")


def _timeline_payload(row: Any) -> dict[str, Any]:
    identity = f"{row.entity_type}:{row.entity_id}"
    if row.entity_type == "message":
        fragment = f"message-{row.entity_id}"
        jump = {
            "href": f"/conversations/{row.conversation_id}?around={row.entity_id}#{fragment}",
            "fragment": fragment,
            "conversation_id": row.conversation_id,
            "message_id": row.entity_id,
        }
    else:
        jump = _activity_jump(row.entity_id)
    return {
        "id": identity,
        "entity_type": row.entity_type,
        "entity_id": row.entity_id,
        "timestamp_ms": row.timestamp_ms,
        "unknown_date": row.timestamp_ms is None,
        "title": row.title,
        "content": row.content,
        "category": row.category,
        "subtype": row.subtype,
        "source": {
            "source_file_id": row.source_file_id,
            "chunk": row.chunk_key,
            "relative_path": row.relative_path,
            "json_pointer": row.json_pointer,
        },
        "jump": jump,
    }


def _timeline_around_activity_id(value: str) -> str:
    entity_type, separator, entity_id = value.partition(":")
    if entity_type != "activity" or separator != ":" or _OPAQUE_ID.fullmatch(entity_id) is None:
        raise HTTPException(
            status_code=422,
            detail="Timeline around must be activity:<activity-id>",
        )
    return entity_id


def _timeline_branch_identity_seek(
    *,
    entity_type: Literal["message", "activity"],
    entity_id: Any,
    target_identity: str,
    direction: ActivityDirection,
    inclusive: bool,
) -> Any:
    target_type, separator, target_id = target_identity.partition(":")
    if separator == ":" and target_type in {"activity", "message"}:
        if entity_type == target_type:
            if direction == "asc":
                return entity_id >= target_id if inclusive else entity_id > target_id
            return entity_id <= target_id if inclusive else entity_id < target_id
        succeeds = entity_type > target_type if direction == "asc" else entity_type < target_type
        return literal(succeeds)
    identity = literal(f"{entity_type}:") + entity_id
    if direction == "asc":
        return identity >= target_identity if inclusive else identity > target_identity
    return identity <= target_identity if inclusive else identity < target_identity


def _timeline_branch_seek(
    *,
    entity_type: Literal["message", "activity"],
    entity_id: Any,
    timestamp_ms: Any,
    target_timestamp_ms: int | None,
    target_identity: str,
    direction: ActivityDirection,
    inclusive: bool,
    include_unknown_timestamps: bool,
) -> Any:
    identity_seek = _timeline_branch_identity_seek(
        entity_type=entity_type,
        entity_id=entity_id,
        target_identity=target_identity,
        direction=direction,
        inclusive=inclusive,
    )
    if target_timestamp_ms is None:
        if entity_type == "message" or not include_unknown_timestamps:
            return literal(False)
        return and_(timestamp_ms.is_(None), identity_seek)

    target_type, separator, target_id = target_identity.partition(":")
    if separator == ":" and target_type == entity_type:
        values = tuple_(timestamp_ms, entity_id)
        boundary = tuple_(literal(target_timestamp_ms), literal(target_id))
        if direction == "asc":
            dated_seek = values >= boundary if inclusive else values > boundary
        else:
            dated_seek = values <= boundary if inclusive else values < boundary
    elif separator == ":" and target_type in {"activity", "message"}:
        same_timestamp_matches = (
            entity_type > target_type if direction == "asc" else entity_type < target_type
        )
        if direction == "asc":
            dated_seek = (
                timestamp_ms >= target_timestamp_ms
                if same_timestamp_matches
                else timestamp_ms > target_timestamp_ms
            )
        else:
            dated_seek = (
                timestamp_ms <= target_timestamp_ms
                if same_timestamp_matches
                else timestamp_ms < target_timestamp_ms
            )
    else:
        time_seek = (
            timestamp_ms > target_timestamp_ms
            if direction == "asc"
            else timestamp_ms < target_timestamp_ms
        )
        dated_seek = or_(
            time_seek,
            and_(timestamp_ms == target_timestamp_ms, identity_seek),
        )
    if entity_type == "activity" and include_unknown_timestamps:
        return or_(dated_seek, timestamp_ms.is_(None))
    return dated_seek


def _timeline_page_statement(
    *,
    entity_type: Literal["message", "activity"],
    filters: list[ColumnElement[bool]],
    direction: ActivityDirection,
    candidate_limit: int,
    seeks: tuple[tuple[int | None, str, bool], ...],
    include_unknown_timestamps: bool,
) -> Any:
    entity_id: Any
    timestamp_ms: Any
    if entity_type == "message":
        entity_id = Message.id
        timestamp_ms = Message.timestamp_ms
        statement = (
            select(
                literal("message").label("entity_type"),
                Message.id.label("entity_id"),
                Message.timestamp_ms.label("timestamp_ms"),
                Message.sender_display.label("title"),
                Message.content_display.label("content"),
                literal("messages").label("category"),
                Message.message_kind.label("subtype"),
                Message.source_file_id.label("source_file_id"),
                Message.json_pointer.label("json_pointer"),
                Message.conversation_id.label("conversation_id"),
                SourceChunk.chunk_key.label("chunk_key"),
                SourceFile.relative_path.label("relative_path"),
            )
            .join(SourceFile, SourceFile.id == Message.source_file_id)
            .join(SourceChunk, SourceChunk.id == SourceFile.chunk_id)
            .where(*filters)
        )
    else:
        entity_id = ActivityRecord.id
        timestamp_ms = ActivityRecord.timestamp_ms
        statement = (
            select(
                literal("activity").label("entity_type"),
                ActivityRecord.id.label("entity_id"),
                ActivityRecord.timestamp_ms.label("timestamp_ms"),
                ActivityRecord.title_display.label("title"),
                ActivityRecord.content_display.label("content"),
                ActivityRecord.category.label("category"),
                ActivityRecord.subtype.label("subtype"),
                ActivityRecord.source_file_id.label("source_file_id"),
                ActivityRecord.json_pointer.label("json_pointer"),
                literal(None).label("conversation_id"),
                SourceChunk.chunk_key.label("chunk_key"),
                SourceFile.relative_path.label("relative_path"),
            )
            .join(SourceFile, SourceFile.id == ActivityRecord.source_file_id)
            .join(SourceChunk, SourceChunk.id == SourceFile.chunk_id)
            .where(*filters)
        )
    for target_timestamp_ms, target_identity, inclusive in seeks:
        statement = statement.where(
            _timeline_branch_seek(
                entity_type=entity_type,
                entity_id=entity_id,
                timestamp_ms=timestamp_ms,
                target_timestamp_ms=target_timestamp_ms,
                target_identity=target_identity,
                direction=direction,
                inclusive=inclusive,
                include_unknown_timestamps=include_unknown_timestamps,
            )
        )
    timestamp_order = (
        timestamp_ms.asc().nulls_last() if direction == "asc" else timestamp_ms.desc().nulls_last()
    )
    id_order = entity_id.asc() if direction == "asc" else entity_id.desc()
    return statement.order_by(timestamp_order, id_order).limit(candidate_limit)


def _timeline_page_statements(
    *,
    q: str | None,
    entity_types: tuple[str, ...],
    from_ms: int | None,
    to_ms: int | None,
    direction: ActivityDirection,
    candidate_limit: int,
    seeks: tuple[tuple[int | None, str, bool], ...] = (),
) -> list[Any]:
    message_filters, activity_filters = _timeline_filters(q=q, from_ms=from_ms, to_ms=to_ms)
    statements: list[Any] = []
    if not entity_types or "message" in entity_types:
        statements.append(
            _timeline_page_statement(
                entity_type="message",
                filters=message_filters,
                direction=direction,
                candidate_limit=candidate_limit,
                seeks=seeks,
                include_unknown_timestamps=False,
            )
        )
    if not entity_types or "activity" in entity_types:
        statements.append(
            _timeline_page_statement(
                entity_type="activity",
                filters=activity_filters,
                direction=direction,
                candidate_limit=candidate_limit,
                seeks=seeks,
                include_unknown_timestamps=from_ms is None and to_ms is None,
            )
        )
    return statements


def _merge_timeline_rows(rows: list[Any], *, direction: ActivityDirection) -> list[Any]:
    reverse = direction == "desc"
    rows.sort(key=lambda row: f"{row.entity_type}:{row.entity_id}", reverse=reverse)
    rows.sort(
        key=lambda row: row.timestamp_ms if row.timestamp_ms is not None else 0,
        reverse=reverse,
    )
    rows.sort(key=lambda row: row.timestamp_ms is None)
    return rows


@activity_router.get("/timeline")
def timeline(
    request: Request,
    q: str | None = Query(default=None, max_length=300),
    entity_type: str | None = Query(default=None, max_length=100),
    from_ms: int | None = Query(default=None, ge=0),
    to_ms: int | None = Query(default=None, ge=0),
    direction: ActivityDirection = "desc",
    limit: int = Query(default=50, ge=1, le=250),
    cursor: str | None = Query(default=None, max_length=2048),
    around: str | None = Query(default=None, min_length=1, max_length=150),
) -> dict[str, Any]:
    if from_ms is not None and to_ms is not None and from_ms > to_ms:
        raise HTTPException(status_code=422, detail="Invalid timeline date range")
    entity_types = csv_values(entity_type)
    if set(entity_types) - {"message", "activity"}:
        raise HTTPException(status_code=422, detail="Unknown timeline entity type")
    around_activity_id = _timeline_around_activity_id(around) if around is not None else None
    scope = scope_hash(
        {"q": q or "", "entity_types": entity_types, "from_ms": from_ms, "to_ms": to_ms}
    )
    entries = _timeline_union(q=q, entity_types=entity_types, from_ms=from_ms, to_ms=to_ms)
    seeks: list[tuple[int | None, str, bool]] = []
    if cursor:
        try:
            decoded = decode_cursor(cursor, kind="timeline")
            if (
                require_cursor_str(decoded, "direction") != direction
                or require_cursor_str(decoded, "scope") != scope
            ):
                raise InvalidCursor("Pagination cursor does not match these timeline filters")
            cursor_null = require_cursor_int(decoded, "null")
            cursor_time = require_cursor_int(decoded, "timestamp_ms")
            cursor_id = require_cursor_str(decoded, "id")
            if cursor_null not in {0, 1}:
                raise InvalidCursor("Invalid pagination cursor")
        except InvalidCursor as exc:
            raise invalid_cursor(str(exc)) from exc
        seeks.append(
            (
                None if cursor_null == 1 else cursor_time,
                cursor_id,
                False,
            )
        )
    with session_for(request) as session:
        if around_activity_id is not None:
            active_target = session.scalar(
                select(ActivityRecord.id).where(
                    ActivityRecord.id == around_activity_id,
                    ActivityRecord.deleted_at.is_(None),
                )
            )
            if active_target is None:
                raise HTTPException(status_code=404, detail="Timeline activity target not found")
            target = session.execute(
                select(entries.c.timestamp_ms).where(
                    entries.c.entity_type == "activity",
                    entries.c.entity_id == around_activity_id,
                )
            ).one_or_none()
            if target is None:
                raise HTTPException(
                    status_code=422,
                    detail="Timeline activity target does not match the active filters",
                )
            seeks.append(
                (
                    target.timestamp_ms,
                    f"activity:{around_activity_id}",
                    True,
                )
            )
        aggregate_memo = catalog_aggregate_memo(request, session)
        total = aggregate_memo.get_or_compute(
            namespace="timeline-filter-total-v1",
            scope=scope,
            compute=lambda: int(session.scalar(select(func.count()).select_from(entries)) or 0),
        )
        source_counts = aggregate_memo.get_or_compute(
            namespace="timeline-source-summary-v1",
            scope="all-active-messages-and-activities",
            compute=lambda: (
                int(
                    session.scalar(
                        select(func.count(Message.pk)).where(Message.deleted_at.is_(None))
                    )
                    or 0
                ),
                int(
                    session.scalar(
                        select(func.count(ActivityRecord.pk)).where(
                            ActivityRecord.deleted_at.is_(None)
                        )
                    )
                    or 0
                ),
            ),
        )
        source_backed_total = sum(source_counts)
        rows = _merge_timeline_rows(
            [
                row
                for statement in _timeline_page_statements(
                    q=q,
                    entity_types=entity_types,
                    from_ms=from_ms,
                    to_ms=to_ms,
                    direction=direction,
                    candidate_limit=limit + 1,
                    seeks=tuple(seeks),
                )
                for row in session.execute(statement)
            ],
            direction=direction,
        )[: limit + 1]
        page_rows = rows[:limit]
        items = [_timeline_payload(row) for row in page_rows]
        attach_star_states(
            session,
            items,
            lambda item: (
                "message" if item["entity_type"] == "message" else "activity",
                str(item["entity_id"]),
            ),
        )
        aggregate_memo.require_current()
    next_cursor = None
    if len(rows) > limit and page_rows:
        last = page_rows[-1]
        next_cursor = encode_cursor(
            "timeline",
            direction=direction,
            scope=scope,
            null=1 if last.timestamp_ms is None else 0,
            timestamp_ms=last.timestamp_ms or 0,
            id=f"{last.entity_type}:{last.entity_id}",
        )
    return {
        "coverage": {
            "status": "source_backed" if source_backed_total else "unavailable",
            "absence_is_not_zero_history": source_backed_total == 0,
        },
        "around": (
            {
                "requested": around,
                "entity_type": "activity",
                "entity_id": around_activity_id,
                "resolved": True,
            }
            if around_activity_id is not None
            else None
        ),
        "items": items,
        "page": {
            "total": total if source_backed_total else None,
            "limit": limit,
            "next_cursor": next_cursor,
            "count_is_unknown": source_backed_total == 0,
        },
    }


@activity_router.get("/timeline/on-this-day")
def on_this_day(
    request: Request,
    month: int = Query(ge=1, le=12),
    day: int = Query(ge=1, le=31),
    timezone: str = Query(default="America/New_York", max_length=100),
    entity_type: str | None = Query(default=None, max_length=100),
    limit: int = Query(default=200, ge=1, le=500),
    cursor: str | None = Query(default=None, max_length=2048),
) -> dict[str, Any]:
    try:
        zone = ZoneInfo(timezone)
        datetime(2000, month, day)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Invalid timezone or calendar date") from exc
    entity_types = csv_values(entity_type)
    if set(entity_types) - {"message", "activity"}:
        raise HTTPException(status_code=422, detail="Unknown timeline entity type")
    scope = scope_hash(
        {
            "month": month,
            "day": day,
            "timezone": timezone,
            "entity_types": entity_types,
        }
    )
    entries = _timeline_union(q=None, entity_types=entity_types, from_ms=None, to_ms=None)
    with session_for(request) as session:
        bounds: list[int] = []
        if not entity_types or "message" in entity_types:
            message_bounds = session.execute(
                select(func.min(Message.timestamp_ms), func.max(Message.timestamp_ms)).where(
                    Message.deleted_at.is_(None)
                )
            ).one()
            bounds.extend(value for value in message_bounds if value is not None)
        if not entity_types or "activity" in entity_types:
            activity_bounds = session.execute(
                select(
                    func.min(ActivityRecord.timestamp_ms),
                    func.max(ActivityRecord.timestamp_ms),
                ).where(
                    ActivityRecord.deleted_at.is_(None),
                    ActivityRecord.timestamp_ms.is_not(None),
                )
            ).one()
            bounds.extend(value for value in activity_bounds if value is not None)
        local_ranges: list[tuple[int, int]] = []
        if bounds:
            first_year = datetime.fromtimestamp(min(bounds) / 1000, zone).year
            last_year = datetime.fromtimestamp(max(bounds) / 1000, zone).year
            for year in range(first_year, last_year + 1):
                try:
                    start_local = datetime(year, month, day, tzinfo=zone)
                except ValueError:  # February 29 in a non-leap year
                    continue
                end_local = start_local + timedelta(days=1)
                local_ranges.append(
                    (
                        round(start_local.astimezone(UTC).timestamp() * 1000),
                        round(end_local.astimezone(UTC).timestamp() * 1000),
                    )
                )
        if not local_ranges:
            return {
                "month": month,
                "day": day,
                "timezone": timezone,
                "items": [],
                "limit": limit,
                "page": {
                    "total": 0,
                    "limit": limit,
                    "next_cursor": None,
                    "count_is_unknown": False,
                },
                "query_strategy": "indexed_local_day_utc_ranges",
            }
        range_filter = or_(
            *(
                and_(entries.c.timestamp_ms >= start, entries.c.timestamp_ms < end)
                for start, end in local_ranges
            )
        )
        identity = entries.c.entity_type + literal(":") + entries.c.entity_id
        statement = (
            select(
                entries,
                SourceChunk.chunk_key.label("chunk_key"),
                SourceFile.relative_path.label("relative_path"),
            )
            .join(SourceFile, SourceFile.id == entries.c.source_file_id)
            .join(SourceChunk, SourceChunk.id == SourceFile.chunk_id)
            .where(range_filter)
        )
        if cursor:
            try:
                decoded = decode_cursor(cursor, kind="on-this-day")
                if require_cursor_str(decoded, "scope") != scope:
                    raise InvalidCursor("Pagination cursor does not match this On This Day query")
                cursor_time = require_cursor_int(decoded, "timestamp_ms")
                cursor_id = require_cursor_str(decoded, "id")
            except InvalidCursor as exc:
                raise invalid_cursor(str(exc)) from exc
            statement = statement.where(
                or_(
                    entries.c.timestamp_ms < cursor_time,
                    and_(entries.c.timestamp_ms == cursor_time, identity > cursor_id),
                )
            )
        statement = statement.order_by(entries.c.timestamp_ms.desc(), identity.asc()).limit(
            limit + 1
        )
        total = session.scalar(select(func.count()).select_from(entries).where(range_filter)) or 0
        rows = list(session.execute(statement))
        page_rows = rows[:limit]
        items = []
        for row in page_rows:
            local = datetime.fromtimestamp(row.timestamp_ms / 1000, zone)
            item = _timeline_payload(row)
            item["local_date"] = local.date().isoformat()
            item["year"] = local.year
            items.append(item)
        attach_star_states(
            session,
            items,
            lambda item: (
                "message" if item["entity_type"] == "message" else "activity",
                str(item["entity_id"]),
            ),
        )
    next_cursor = None
    if len(rows) > limit and page_rows:
        last = page_rows[-1]
        next_cursor = encode_cursor(
            "on-this-day",
            scope=scope,
            timestamp_ms=last.timestamp_ms,
            id=f"{last.entity_type}:{last.entity_id}",
        )
    return {
        "month": month,
        "day": day,
        "timezone": timezone,
        "items": items,
        "limit": limit,
        "page": {
            "total": total,
            "limit": limit,
            "next_cursor": next_cursor,
            "count_is_unknown": False,
        },
        "query_strategy": "indexed_local_day_utc_ranges",
    }


@activity_router.get("/support")
def support_threads(
    request: Request,
    limit: int = Query(default=100, ge=1, le=250),
    cursor: str | None = Query(default=None, max_length=2_048),
) -> dict[str, Any]:
    cursor_id: str | None = None
    cursor_null = 0
    cursor_value: int | None = None
    if cursor:
        try:
            decoded = decode_cursor(cursor, kind="support-threads")
            if require_cursor_str(decoded, "scope") != "all":
                raise InvalidCursor("Pagination cursor does not match support conversations")
            if require_cursor_int(decoded, "limit") != limit:
                raise InvalidCursor("Pagination cursor does not match the requested page size")
            cursor_id = require_cursor_str(decoded, "id")
            cursor_null = require_cursor_int(decoded, "null")
            raw_value = decoded.get("value")
            if cursor_null not in {0, 1} or not isinstance(raw_value, (int, type(None))):
                raise InvalidCursor("Invalid support-conversation cursor")
            if cursor_null == 0 and raw_value is None:
                raise InvalidCursor("Invalid support-conversation cursor")
            cursor_value = raw_value
        except InvalidCursor as exc:
            raise invalid_cursor(str(exc)) from exc
    null_rank = case((Conversation.last_activity_ms.is_(None), 1), else_=0)
    with session_for(request) as session:
        support_ids = select(ConversationSegment.conversation_id).where(
            ConversationSegment.source_category == "support"
        )
        total = (
            session.scalar(
                select(func.count(Conversation.id)).where(
                    Conversation.id.in_(support_ids), Conversation.deleted_at.is_(None)
                )
            )
            or 0
        )
        coverage = coverage_payload(
            session,
            "support_messages",
            label="Support messages",
            unavailable_description="No support-message history is available in this export.",
            actual_count=total,
            status_with_data="available",
        )
        if total == 0 and coverage["status"] == "unavailable":
            return unavailable_page(coverage, limit=limit)
        statement = select(Conversation).where(
            Conversation.id.in_(support_ids), Conversation.deleted_at.is_(None)
        )
        if cursor_id is not None:
            id_seek = Conversation.id > cursor_id
            if cursor_null == 1:
                seek = and_(null_rank == 1, id_seek)
            else:
                seek = or_(
                    null_rank > cursor_null,
                    and_(
                        null_rank == cursor_null,
                        or_(
                            Conversation.last_activity_ms < cursor_value,
                            and_(Conversation.last_activity_ms == cursor_value, id_seek),
                        ),
                    ),
                )
            statement = statement.where(seek)
        conversations = list(
            session.scalars(
                statement.order_by(
                    null_rank.asc(), Conversation.last_activity_ms.desc(), Conversation.id.asc()
                ).limit(limit + 1)
            )
        )
        page_rows = conversations[:limit]
        items = [
            {
                "id": row.id,
                "title": row.title_display,
                "message_count": row.message_count,
                "first_activity_ms": row.first_activity_ms,
                "last_activity_ms": row.last_activity_ms,
                "href": f"/conversations/{row.id}",
            }
            for row in page_rows
        ]
        attach_star_states(
            session,
            items,
            lambda item: ("conversation", str(item["id"])),
        )
    next_cursor = None
    if len(conversations) > limit and page_rows:
        last = page_rows[-1]
        next_cursor = encode_cursor(
            "support-threads",
            scope="all",
            limit=limit,
            id=last.id,
            null=1 if last.last_activity_ms is None else 0,
            value=last.last_activity_ms,
        )
    return {
        "coverage": coverage,
        "items": items,
        "page": {
            "total": total,
            "limit": limit,
            "next_cursor": next_cursor,
            "count_is_unknown": False,
        },
    }
