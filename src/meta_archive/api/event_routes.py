from __future__ import annotations

import re
from collections import Counter
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import func, or_, select

from meta_archive.api.pagination import (
    InvalidCursor,
    decode_cursor,
    encode_cursor,
    require_cursor_str,
)
from meta_archive.api.security import require_session
from meta_archive.api.stage4_common import (
    coverage_payload,
    escaped_like,
    invalid_cursor,
    scope_hash,
    session_for,
    unavailable_page,
)
from meta_archive.models import ActivityRecord, SourceChunk, SourceFile
from meta_archive.services.stars import attach_star_states
from meta_archive.services.text import normalize_text

event_router = APIRouter(prefix="/api/v1", dependencies=[Depends(require_session)])

EventDirection = Literal["asc", "desc"]
EventSort = Literal["date", "title", "status"]
EVENT_STATUSES = frozenset(
    {"hosted", "created", "went", "interested", "invited", "declined", "cancelled", "unknown"}
)
_OPAQUE_EVENT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def _raw_payload(row: ActivityRecord) -> dict[str, Any]:
    raw = row.extra_json.get("raw_payload")
    return raw if isinstance(raw, dict) else {}


def _event_status(row: ActivityRecord) -> str:
    raw = _raw_payload(row)
    candidates = [
        raw.get("event_response"),
        raw.get("rsvp_status"),
        raw.get("response"),
        raw.get("status"),
        row.subtype,
    ]
    aliases = {
        "going": "went",
        "attended": "went",
        "events_you_attended": "went",
        "events_you_went_to": "went",
        "created_event": "created",
        "events_created_by_you": "created",
        "events_you_created": "created",
        "hosting": "hosted",
        "events_you_hosted": "hosted",
        "maybe": "interested",
        "events_you_were_interested_in": "interested",
        "events_you_were_invited_to": "invited",
        "events_you_declined": "declined",
        "cancelled_events": "cancelled",
        "not_going": "declined",
        "canceled": "cancelled",
    }
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        normalized = candidate.casefold().replace("-", "_").replace(" ", "_")
        normalized = aliases.get(normalized, normalized)
        if normalized in EVENT_STATUSES:
            return normalized
    return "unknown"


def _event_hosted(row: ActivityRecord, status_value: str) -> bool | None:
    raw = _raw_payload(row)
    for key in ("hosted", "is_host", "is_hosted", "created_by_you"):
        value = raw.get(key)
        if isinstance(value, bool):
            return value
    if status_value in {"hosted", "created"}:
        return True
    return None


def _sort_events(
    rows: list[tuple[ActivityRecord, str, str]],
    *,
    sort: EventSort,
    direction: EventDirection,
) -> list[tuple[ActivityRecord, str, str]]:
    reverse = direction == "desc"

    def value(item: tuple[ActivityRecord, str, str]) -> int | str | None:
        row = item[0]
        if sort == "date":
            return row.timestamp_ms
        if sort == "status":
            return _event_status(row)
        return row.title_search or ""

    known = [item for item in rows if value(item) is not None]
    unknown = [item for item in rows if value(item) is None]
    if sort == "date":
        known.sort(key=lambda item: (item[0].timestamp_ms or 0, item[0].id), reverse=reverse)
    elif sort == "status":
        known.sort(key=lambda item: (_event_status(item[0]), item[0].id), reverse=reverse)
    else:
        known.sort(key=lambda item: (item[0].title_search or "", item[0].id), reverse=reverse)
    unknown.sort(key=lambda item: item[0].id, reverse=reverse)
    return known + unknown


@event_router.get("/events")
def events(
    request: Request,
    q: str | None = Query(default=None, max_length=300),
    event_status: str | None = Query(default=None, max_length=100),
    hosted: bool | None = None,
    from_ms: int | None = Query(default=None, ge=0),
    to_ms: int | None = Query(default=None, ge=0),
    sort: EventSort = "date",
    direction: EventDirection = "desc",
    limit: int = Query(default=50, ge=1, le=250),
    cursor: str | None = Query(default=None, max_length=2048),
    around: str | None = Query(default=None, min_length=1, max_length=128),
) -> dict[str, Any]:
    if from_ms is not None and to_ms is not None and from_ms > to_ms:
        raise HTTPException(status_code=422, detail="Invalid event date range")
    statuses = tuple(
        dict.fromkeys(part.strip() for part in (event_status or "").split(",") if part.strip())
    )
    if set(statuses) - EVENT_STATUSES:
        raise HTTPException(status_code=422, detail="Unknown event status")
    if around is not None and _OPAQUE_EVENT_ID.fullmatch(around) is None:
        raise HTTPException(status_code=422, detail="Invalid event around target")
    source_filters = [
        ActivityRecord.deleted_at.is_(None),
        or_(
            ActivityRecord.category == "events",
            func.lower(SourceFile.normalized_path).like("%/events/%"),
        ),
    ]
    filters = list(source_filters)
    if from_ms is not None:
        filters.append(ActivityRecord.timestamp_ms >= from_ms)
    if to_ms is not None:
        filters.append(ActivityRecord.timestamp_ms <= to_ms)
    if q:
        normalized = escaped_like(normalize_text(q).search)
        filters.append(
            or_(
                ActivityRecord.title_search.like(f"%{normalized}%", escape="\\"),
                ActivityRecord.content_search.like(f"%{normalized}%", escape="\\"),
            )
        )
    scope = scope_hash(
        {
            "q": q or "",
            "status": statuses,
            "hosted": hosted,
            "from_ms": from_ms,
            "to_ms": to_ms,
        }
    )
    with session_for(request) as session:
        source_total = (
            session.scalar(
                select(func.count(ActivityRecord.pk))
                .join(SourceFile, SourceFile.id == ActivityRecord.source_file_id)
                .where(*source_filters)
            )
            or 0
        )
        candidates = [
            (row, chunk, path)
            for row, chunk, path in session.execute(
                select(ActivityRecord, SourceChunk.chunk_key, SourceFile.relative_path)
                .join(SourceFile, SourceFile.id == ActivityRecord.source_file_id)
                .join(SourceChunk, SourceChunk.id == SourceFile.chunk_id)
                .where(*filters)
            )
        ]
        if around is not None:
            target_exists = session.scalar(
                select(ActivityRecord.id)
                .join(SourceFile, SourceFile.id == ActivityRecord.source_file_id)
                .where(ActivityRecord.id == around, *source_filters)
            )
            if target_exists is None:
                raise HTTPException(status_code=404, detail="Event target not found")
        filtered = [
            item
            for item in candidates
            if (not statuses or _event_status(item[0]) in statuses)
            and (hosted is None or _event_hosted(item[0], _event_status(item[0])) is hosted)
        ]
        ordered = _sort_events(filtered, sort=sort, direction=direction)
        coverage = coverage_payload(
            session,
            "events",
            label="Events",
            unavailable_description="Event history is unavailable in this export.",
            actual_count=source_total,
            status_with_data="partial",
        )
    if source_total == 0 and coverage["status"] == "unavailable":
        return unavailable_page(coverage, limit=limit) | {
            "summary": {"by_status": None, "hosted": None, "attended": None}
        }

    start = 0
    if cursor:
        try:
            decoded = decode_cursor(cursor, kind="events")
            if (
                require_cursor_str(decoded, "scope") != scope
                or require_cursor_str(decoded, "sort") != sort
                or require_cursor_str(decoded, "direction") != direction
            ):
                raise InvalidCursor("Pagination cursor does not match these event filters")
            cursor_id = require_cursor_str(decoded, "id")
            start = next(index + 1 for index, item in enumerate(ordered) if item[0].id == cursor_id)
        except (InvalidCursor, StopIteration) as exc:
            raise invalid_cursor(str(exc) or "Event cursor no longer exists") from exc
    elif around is not None:
        try:
            start = next(index for index, item in enumerate(ordered) if item[0].id == around)
        except StopIteration as exc:
            raise HTTPException(
                status_code=422,
                detail="Event target does not match the active filters",
            ) from exc
    page_rows = ordered[start : start + limit]
    next_cursor = None
    if start + limit < len(ordered) and page_rows:
        next_cursor = encode_cursor(
            "events",
            scope=scope,
            sort=sort,
            direction=direction,
            id=page_rows[-1][0].id,
        )
    counts = Counter(_event_status(item[0]) for item in filtered)
    items = [
        {
            "id": row.id,
            "title": row.title_display,
            "description": row.content_display,
            "timestamp_ms": row.timestamp_ms,
            "unknown_date": row.timestamp_ms is None,
            "status": _event_status(row),
            "hosted": _event_hosted(row, _event_status(row)),
            "source": {
                "source_file_id": row.source_file_id,
                "source_record_id": row.source_record_id,
                "chunk": chunk,
                "relative_path": path,
                "json_pointer": row.json_pointer,
            },
            "jump": {
                "href": f"/events?around={row.id}#event-{row.id}",
                "fragment": f"event-{row.id}",
            },
        }
        for row, chunk, path in page_rows
    ]
    with session_for(request) as session:
        attach_star_states(
            session,
            items,
            lambda item: ("activity", str(item["id"])),
        )
    return {
        "coverage": coverage,
        "around": (
            {"requested": around, "event_id": around, "resolved": True}
            if around is not None
            else None
        ),
        "items": items,
        "summary": {
            "by_status": dict(sorted(counts.items())),
            "hosted": sum(
                _event_hosted(item[0], _event_status(item[0])) is True for item in filtered
            ),
            # Only an explicit `went` source status is counted as attended.
            "attended": counts.get("went", 0),
        },
        "page": {
            "total": len(filtered),
            "limit": limit,
            "next_cursor": next_cursor,
            "count_is_unknown": False,
        },
    }
