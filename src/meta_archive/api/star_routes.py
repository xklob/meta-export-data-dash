from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, cast

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from meta_archive.api.aggregate_cache import catalog_aggregate_memo
from meta_archive.api.pagination import (
    InvalidCursor,
    decode_cursor,
    encode_cursor,
    require_cursor_int,
    require_cursor_str,
)
from meta_archive.api.security import require_session
from meta_archive.models import Star
from meta_archive.services.deletion_lock_lease import DeletionError, deletion_lock
from meta_archive.services.source_import_lock import (
    SourceImportPublicationLockError,
    source_import_publication_lock,
)
from meta_archive.services.stars import (
    STAR_CATEGORY_TARGET_TYPES,
    STAR_TARGET_TYPES,
    StarTargetKey,
    make_star_target,
    resolve_star_targets,
    serialize_star,
)

StarCategory = Literal["all", "conversations", "messages", "media", "activity", "people"]
StarDirection = Literal["asc", "desc"]
StarTargetPath = Literal[
    "conversation",
    "message",
    "activity",
    "attachment",
    "blob",
    "messenger_contact",
]

star_router = APIRouter(prefix="/api/v1/stars", dependencies=[Depends(require_session)])

_TARGET_CATEGORY = {
    "conversation": "conversations",
    "message": "messages",
    "activity": "activity",
    "attachment": "media",
    "blob": "media",
    "messenger_contact": "people",
}


def _utc_datetime(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _chunks[ItemT](items: list[ItemT], size: int) -> Iterator[list[ItemT]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def _cursor_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise InvalidCursor("Invalid starred-items cursor") from exc
    if parsed.tzinfo is None:
        raise InvalidCursor("Invalid starred-items cursor")
    return parsed.astimezone(UTC)


def _cursor_error(exc: InvalidCursor) -> HTTPException:
    return HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc))


def _mutation_conflict(exc: Exception) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail="The archive catalog is currently being updated; retry the star change",
    )


@star_router.get("")
def starred_items(
    request: Request,
    category: StarCategory = "all",
    direction: StarDirection = "desc",
    limit: int = Query(default=50, ge=1, le=250),
    cursor: str | None = Query(default=None, max_length=2048),
) -> dict[str, Any]:
    order = (
        (Star.starred_at.asc(), Star.id.asc())
        if direction == "asc"
        else (Star.starred_at.desc(), Star.id.desc())
    )
    with Session(request.app.state.engine) as session:
        aggregate_memo = catalog_aggregate_memo(request, session)
        stored = list(session.scalars(select(Star).order_by(*order)))
        active: list[Star] = []
        facets = {
            "all": 0,
            "conversations": 0,
            "messages": 0,
            "media": 0,
            "activity": 0,
            "people": 0,
        }
        for batch in _chunks(stored, 200):
            keys = [
                cast(StarTargetKey, (row.target_type, row.target_id))
                for row in batch
                if row.target_type in STAR_TARGET_TYPES
            ]
            resolutions = resolve_star_targets(session, keys, include_payload=False)
            for row in batch:
                key = cast(StarTargetKey, (row.target_type, row.target_id))
                resolution = resolutions.get(key)
                if resolution is None or resolution.state != "active":
                    continue
                active.append(row)
                facets["all"] += 1
                facets[_TARGET_CATEGORY[row.target_type]] += 1

        selected_types = STAR_CATEGORY_TARGET_TYPES[category]
        selected = [row for row in active if row.target_type in selected_types]
        total = len(selected)
        if cursor is not None:
            try:
                decoded = decode_cursor(cursor, kind="stars")
                if (
                    require_cursor_str(decoded, "category") != category
                    or require_cursor_str(decoded, "direction") != direction
                    or require_cursor_int(decoded, "limit") != limit
                ):
                    raise InvalidCursor(
                        "Pagination cursor does not match these starred-item filters"
                    )
                cursor_at = _cursor_datetime(require_cursor_str(decoded, "at"))
                cursor_id = require_cursor_str(decoded, "id")
            except InvalidCursor as exc:
                raise _cursor_error(exc) from exc
            cursor_key = (cursor_at, cursor_id)
            if direction == "asc":
                selected = [
                    row for row in selected if (_utc_datetime(row.starred_at), row.id) > cursor_key
                ]
            else:
                selected = [
                    row for row in selected if (_utc_datetime(row.starred_at), row.id) < cursor_key
                ]

        page_rows = selected[:limit]
        page_keys = [cast(StarTargetKey, (row.target_type, row.target_id)) for row in page_rows]
        hydrated = resolve_star_targets(session, page_keys)
        items: list[dict[str, Any]] = []
        for row, key in zip(page_rows, page_keys, strict=True):
            resolution = hydrated[key]
            if resolution.state != "active" or resolution.payload is None:
                continue
            item = dict(resolution.payload)
            item["star_target"] = make_star_target(*key)
            item["star"] = serialize_star(row)
            items.append(item)

        next_cursor = None
        if len(selected) > limit and page_rows:
            last = page_rows[-1]
            next_cursor = encode_cursor(
                "stars",
                category=category,
                direction=direction,
                limit=limit,
                at=_utc_datetime(last.starred_at).isoformat(),
                id=last.id,
            )
        aggregate_memo.require_current()

    return {
        "items": items,
        "page": {
            "limit": limit,
            "total": total,
            "next_cursor": next_cursor,
        },
        "category": category,
        "direction": direction,
        "facets": facets,
    }


@star_router.put("/{target_type}/{target_id}")
def put_star(
    request: Request,
    target_type: StarTargetPath,
    target_id: Annotated[
        str,
        Path(
            min_length=1,
            max_length=128,
            pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$",
        ),
    ],
) -> dict[str, Any]:
    key: StarTargetKey = (target_type, target_id)
    try:
        with (
            deletion_lock(request.app.state.layout),
            source_import_publication_lock(request.app.state.layout, blocking=False),
            Session(request.app.state.engine) as session,
            session.begin(),
        ):
            resolution = resolve_star_targets(session, [key], include_payload=False)[key]
            if resolution.state != "active":
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Star target not found",
                )
            row = session.scalar(
                select(Star).where(
                    Star.target_type == target_type,
                    Star.target_id == target_id,
                )
            )
            created = row is None
            if row is None:
                row = Star(
                    id=str(uuid.uuid4()),
                    target_type=target_type,
                    target_id=target_id,
                    starred_at=datetime.now(UTC),
                )
                session.add(row)
                session.flush()
                session.refresh(row)
            response = {
                "star_target": make_star_target(*key),
                "star": serialize_star(row),
                "created": created,
            }
    except (DeletionError, SourceImportPublicationLockError) as exc:
        raise _mutation_conflict(exc) from exc
    return response


@star_router.delete("/{target_type}/{target_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_star(
    request: Request,
    target_type: StarTargetPath,
    target_id: Annotated[
        str,
        Path(
            min_length=1,
            max_length=128,
            pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$",
        ),
    ],
) -> Response:
    try:
        with (
            deletion_lock(request.app.state.layout),
            source_import_publication_lock(request.app.state.layout, blocking=False),
            Session(request.app.state.engine) as session,
            session.begin(),
        ):
            row = session.scalar(
                select(Star).where(
                    Star.target_type == target_type,
                    Star.target_id == target_id,
                )
            )
            if row is not None:
                session.delete(row)
    except (DeletionError, SourceImportPublicationLockError) as exc:
        raise _mutation_conflict(exc) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)
