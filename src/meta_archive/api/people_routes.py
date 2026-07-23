from __future__ import annotations

import hashlib
import re
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import and_, case, func, or_, select
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from meta_archive.api.aggregate_cache import (
    CatalogAggregateMemo,
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
from meta_archive.api.stage4_common import (
    coverage_payload,
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
    Participant,
    SegmentParticipant,
    SourceChunk,
    SourceFile,
)
from meta_archive.services.identifiers import stable_id
from meta_archive.services.stars import attach_star_states
from meta_archive.services.text import normalize_text

people_router = APIRouter(prefix="/api/v1", dependencies=[Depends(require_session)])

PeopleView = Literal[
    "contacts",
    "friends",
    "requests",
    "followers",
    "facebook-blocks",
    "messenger-blocks",
]
ContactSort = Literal[
    "name", "messages", "photos", "videos", "audio", "conversations", "last_activity"
]
RelationshipSort = Literal["name", "date", "status"]
PeopleSort = Literal[
    "name",
    "messages",
    "photos",
    "videos",
    "audio",
    "conversations",
    "last_activity",
    "date",
    "status",
]
Direction = Literal["asc", "desc"]
_CONTACT_SORTS = frozenset(
    {"name", "messages", "photos", "videos", "audio", "conversations", "last_activity"}
)
_RELATIONSHIP_SORTS = frozenset({"name", "date", "status"})
_OPAQUE_RELATIONSHIP_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_PEOPLE_CACHE_INITIALIZATION_LOCK = threading.Lock()
_PEOPLE_PROJECTION_MAX_ENTRIES = 1
_PEOPLE_PROJECTION_MAX_ROWS = 250_000
_PEOPLE_PROJECTION_MAX_BYTES = 64 * 1024 * 1024
_MISSING = object()


@dataclass(frozen=True, slots=True)
class _ConversationLink:
    id: str
    title: str
    conversation_type: str
    message_count: int
    first_activity_ms: int | None
    last_activity_ms: int | None


@dataclass(slots=True)
class _ContactBuilder:
    id: str
    name: str
    name_search: str
    conversations: dict[str, _ConversationLink] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _ContactCounts:
    messages: int
    photos: int
    videos: int
    audio: int

    def payload(self) -> dict[str, int]:
        return {
            "messages": self.messages,
            "photos": self.photos,
            "videos": self.videos,
            "audio": self.audio,
        }


@dataclass(frozen=True, slots=True)
class _ContactProjection:
    id: str
    name: str
    name_search: str
    conversations: tuple[_ConversationLink, ...]
    direct_counts: _ContactCounts
    combined_thread_counts: _ContactCounts
    shared_group_counts: _ContactCounts
    direct_conversation_count: int
    group_conversation_count: int
    first_activity_ms: int | None
    last_activity_ms: int | None

    @property
    def conversation_count(self) -> int:
        return self.direct_conversation_count + self.group_conversation_count


@dataclass(frozen=True, slots=True)
class _ContactProjectionSet:
    contacts: tuple[_ContactProjection, ...]
    row_count: int
    byte_count: int


@dataclass(frozen=True, slots=True)
class _WeightedProjection[ProjectionT]:
    value: ProjectionT
    row_count: int
    byte_count: int


class _RevisionBoundPeopleProjectionCache:
    """Bounded LRU for immutable people projections containing private display data.

    This cache is intentionally separate from ``RevisionBoundAggregateCache``, whose values may
    never contain display content. Cache keys are SHA-256 digests, and explicit aggregate row and
    byte budgets prevent an archive with unusually broad group membership from retaining an
    unbounded private projection in process memory.
    """

    def __init__(
        self,
        *,
        max_entries: int = _PEOPLE_PROJECTION_MAX_ENTRIES,
        max_rows: int = _PEOPLE_PROJECTION_MAX_ROWS,
        max_bytes: int = _PEOPLE_PROJECTION_MAX_BYTES,
    ) -> None:
        if min(max_entries, max_rows, max_bytes) < 1:
            raise ValueError("People projection cache limits must be positive")
        self._max_entries = max_entries
        self._max_rows = max_rows
        self._max_bytes = max_bytes
        self._entries: OrderedDict[bytes, _WeightedProjection[object]] = OrderedDict()
        self._retained_rows = 0
        self._retained_bytes = 0
        self._lock = threading.Lock()

    @staticmethod
    def _key(*, namespace: str, scope: str, revision: str) -> bytes:
        framed = f"{namespace}\0{scope}\0{revision}".encode("utf-8", errors="surrogatepass")
        return hashlib.sha256(framed).digest()

    def get_or_compute[ProjectionT](
        self,
        *,
        namespace: str,
        scope: str,
        revision: str,
        revision_is_current: Callable[[], bool],
        compute: Callable[[], _WeightedProjection[ProjectionT]],
    ) -> ProjectionT:
        key = self._key(namespace=namespace, scope=scope, revision=revision)
        with self._lock:
            cached = self._entries.get(key, _MISSING)
            if cached is not _MISSING:
                self._entries.move_to_end(key)
        if cached is not _MISSING and revision_is_current():
            return cast(_WeightedProjection[ProjectionT], cached).value

        loaded = compute()
        if loaded.row_count < 0 or loaded.byte_count < 0:
            raise ValueError("People projection cache weights cannot be negative")
        if not revision_is_current():
            return loaded.value
        if loaded.row_count > self._max_rows or loaded.byte_count > self._max_bytes:
            return loaded.value

        with self._lock:
            existing = self._entries.get(key, _MISSING)
            if existing is not _MISSING:
                self._entries.move_to_end(key)
                return cast(_WeightedProjection[ProjectionT], existing).value
            retained = cast(_WeightedProjection[object], loaded)
            self._entries[key] = retained
            self._entries.move_to_end(key)
            self._retained_rows += retained.row_count
            self._retained_bytes += retained.byte_count
            while (
                len(self._entries) > self._max_entries
                or self._retained_rows > self._max_rows
                or self._retained_bytes > self._max_bytes
            ):
                _evicted_key, evicted = self._entries.popitem(last=False)
                self._retained_rows -= evicted.row_count
                self._retained_bytes -= evicted.byte_count
        return loaded.value

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    @property
    def retained_rows(self) -> int:
        with self._lock:
            return self._retained_rows

    @property
    def retained_bytes(self) -> int:
        with self._lock:
            return self._retained_bytes


def _attachment_type_expression() -> Any:
    media_type = func.lower(func.coalesce(AttachmentOccurrence.media_type, ""))
    return case(
        (
            AttachmentOccurrence.kind.in_(("gifs", "gif", "sticker")),
            "other",
        ),
        (AttachmentOccurrence.kind.in_(("audio_files", "audio")), "audio"),
        (AttachmentOccurrence.kind.in_(("videos", "video")), "videos"),
        (AttachmentOccurrence.kind.in_(("photos", "photo", "thread_image")), "photos"),
        (media_type.like("audio/%"), "audio"),
        (media_type.like("video/%"), "videos"),
        (media_type.like("image/%"), "photos"),
        else_="other",
    )


def _build_contact_projection(session: Session) -> _ContactProjectionSet:
    contacts: dict[str, _ContactBuilder] = {}
    statement = (
        select(
            Participant.name_search,
            func.min(Participant.name_display),
            Conversation.id,
            Conversation.title_display,
            Conversation.conversation_type,
            Conversation.message_count,
            Conversation.first_activity_ms,
            Conversation.last_activity_ms,
        )
        .join(SegmentParticipant, SegmentParticipant.participant_id == Participant.id)
        .join(ConversationSegment, ConversationSegment.id == SegmentParticipant.segment_id)
        .join(Conversation, Conversation.id == ConversationSegment.conversation_id)
        .where(
            Participant.is_owner.is_(False),
            Conversation.deleted_at.is_(None),
            Conversation.conversation_type.in_(("direct", "group")),
        )
        .group_by(Participant.name_search, Conversation.id)
        .order_by(Participant.name_search, Conversation.id)
    )
    rows = session.execute(statement)
    for (
        name_search,
        name_display,
        conversation_id,
        title,
        conversation_type,
        message_count,
        first_activity_ms,
        last_activity_ms,
    ) in rows:
        contact = contacts.get(name_search)
        if contact is None:
            contact = _ContactBuilder(
                id=stable_id("messenger-contact", name_search),
                name=name_display,
                name_search=name_search,
            )
            contacts[name_search] = contact
        elif name_display < contact.name:
            contact.name = name_display
        contact.conversations[conversation_id] = _ConversationLink(
            id=conversation_id,
            title=title,
            conversation_type=conversation_type,
            message_count=message_count,
            first_activity_ms=first_activity_ms,
            last_activity_ms=last_activity_ms,
        )

    conversation_ids = {
        conversation_id
        for contact in contacts.values()
        for conversation_id in contact.conversations
    }
    attachment_counts: dict[str, dict[str, int]] = {}
    if conversation_ids:
        type_expression = _attachment_type_expression()
        for conversation_id, attachment_type, count in session.execute(
            select(
                AttachmentOccurrence.conversation_id,
                type_expression,
                func.count(AttachmentOccurrence.id),
            )
            .where(
                AttachmentOccurrence.conversation_id.in_(conversation_ids),
                AttachmentOccurrence.message_id.is_not(None),
                AttachmentOccurrence.deleted_at.is_(None),
            )
            .group_by(AttachmentOccurrence.conversation_id, type_expression)
        ):
            if conversation_id is not None:
                attachment_counts.setdefault(conversation_id, {})[attachment_type] = count

    projections: list[_ContactProjection] = []
    for contact in contacts.values():
        links = tuple(
            sorted(
                contact.conversations.values(),
                key=lambda row: (row.conversation_type, -(row.last_activity_ms or -1), row.id),
            )
        )
        direct = tuple(link for link in links if link.conversation_type == "direct")
        groups = tuple(link for link in links if link.conversation_type == "group")

        def attachment_total(selected: tuple[_ConversationLink, ...], kind: str) -> int:
            return sum(attachment_counts.get(link.id, {}).get(kind, 0) for link in selected)

        def messages(selected: tuple[_ConversationLink, ...]) -> int:
            return sum(link.message_count for link in selected)

        projections.append(
            _ContactProjection(
                id=contact.id,
                name=contact.name,
                name_search=contact.name_search,
                conversations=links,
                direct_counts=_ContactCounts(
                    messages=messages(direct),
                    photos=attachment_total(direct, "photos"),
                    videos=attachment_total(direct, "videos"),
                    audio=attachment_total(direct, "audio"),
                ),
                combined_thread_counts=_ContactCounts(
                    messages=messages(links),
                    photos=attachment_total(links, "photos"),
                    videos=attachment_total(links, "videos"),
                    audio=attachment_total(links, "audio"),
                ),
                shared_group_counts=_ContactCounts(
                    messages=messages(groups),
                    photos=attachment_total(groups, "photos"),
                    videos=attachment_total(groups, "videos"),
                    audio=attachment_total(groups, "audio"),
                ),
                direct_conversation_count=len(direct),
                group_conversation_count=len(groups),
                first_activity_ms=min(
                    (
                        link.first_activity_ms
                        for link in links
                        if link.first_activity_ms is not None
                    ),
                    default=None,
                ),
                last_activity_ms=max(
                    (link.last_activity_ms for link in links if link.last_activity_ms is not None),
                    default=None,
                ),
            )
        )

    # This is a deliberately conservative retained-memory estimate, not serialized payload size.
    # It includes fixed object/tuple overhead in addition to the exact UTF-8 string material.
    row_count = len(projections) + sum(len(contact.conversations) for contact in projections)
    byte_count = sum(
        512
        + len(contact.id.encode("utf-8", errors="surrogatepass"))
        + len(contact.name.encode("utf-8", errors="surrogatepass"))
        + len(contact.name_search.encode("utf-8", errors="surrogatepass"))
        + sum(
            256
            + len(link.id.encode("utf-8", errors="surrogatepass"))
            + len(link.title.encode("utf-8", errors="surrogatepass"))
            + len(link.conversation_type.encode("utf-8", errors="surrogatepass"))
            for link in contact.conversations
        )
        for contact in projections
    )
    return _ContactProjectionSet(
        contacts=tuple(projections),
        row_count=row_count,
        byte_count=byte_count,
    )


def _people_projection_cache(request: Request) -> _RevisionBoundPeopleProjectionCache:
    cache = getattr(request.app.state, "people_projection_cache", None)
    if isinstance(cache, _RevisionBoundPeopleProjectionCache):
        return cache
    with _PEOPLE_CACHE_INITIALIZATION_LOCK:
        cache = getattr(request.app.state, "people_projection_cache", None)
        if isinstance(cache, _RevisionBoundPeopleProjectionCache):
            return cache
        created = _RevisionBoundPeopleProjectionCache()
        request.app.state.people_projection_cache = created
        return created


def _contacts_projection(
    request: Request,
    session: Session,
    aggregate_memo: CatalogAggregateMemo,
) -> tuple[_ContactProjection, ...]:
    def load() -> _WeightedProjection[tuple[_ContactProjection, ...]]:
        projection = _build_contact_projection(session)
        return _WeightedProjection(
            value=projection.contacts,
            row_count=projection.row_count,
            byte_count=projection.byte_count,
        )

    return _people_projection_cache(request).get_or_compute(
        namespace="messenger-contact-projection-v1",
        scope="all",
        revision=aggregate_memo.revision,
        revision_is_current=lambda: catalog_revision(request, session) == aggregate_memo.revision,
        compute=load,
    )


def _contact_payload(contact: _ContactProjection) -> dict[str, Any]:
    direct_counts = contact.direct_counts.payload()
    return {
        "id": contact.id,
        "name": contact.name,
        "contact_basis": "messenger_participant_label",
        "is_friend": None,
        "friendship_status": "unknown",
        "counts": direct_counts.copy(),
        "direct_counts": direct_counts,
        "combined_thread_counts": contact.combined_thread_counts.payload(),
        "shared_group_counts": contact.shared_group_counts.payload(),
        "conversation_count": contact.conversation_count,
        "direct_conversation_count": contact.direct_conversation_count,
        "group_conversation_count": contact.group_conversation_count,
        "first_activity_ms": contact.first_activity_ms,
        "last_activity_ms": contact.last_activity_ms,
        "conversations": [
            {
                "id": link.id,
                "title": link.title,
                "type": link.conversation_type,
                "message_count": link.message_count,
                "first_activity_ms": link.first_activity_ms,
                "last_activity_ms": link.last_activity_ms,
                "href": f"/conversations/{link.id}",
            }
            for link in contact.conversations
        ],
        "count_semantics": (
            "Counts and direct_counts describe one-to-one exchanges. Shared-group counts "
            "cover whole shared conversations and are not attributed to this person; "
            "combined_thread_counts is the explicit sum of those two scopes."
        ),
    }


def _sort_contacts(
    items: list[_ContactProjection], sort: ContactSort, direction: Direction
) -> None:
    reverse = direction == "desc"

    def key(item: _ContactProjection) -> tuple[int, Any, str]:
        if sort == "name":
            value: Any = normalize_text(item.name).search
        elif sort == "conversations":
            value = item.conversation_count
        elif sort == "last_activity":
            value = item.last_activity_ms
        else:
            value = getattr(item.direct_counts, sort)
        return (1 if value is None else 0, value if value is not None else 0, item.id)

    if reverse:
        known = [item for item in items if key(item)[0] == 0]
        unknown = [item for item in items if key(item)[0] == 1]
        known.sort(key=key, reverse=True)
        unknown.sort(key=lambda item: item.id, reverse=True)
        items[:] = known + unknown
    else:
        items.sort(key=key)


_RELATIONSHIP_VIEW: dict[str, tuple[str, str, tuple[str, ...], str]] = {
    "friends": (
        "friends",
        "Friends",
        ("%friends%", "%friendship%"),
        "Accepted-friend history is unavailable in this export; contacts are not friends.",
    ),
    "requests": (
        "friend_requests",
        "Friend requests",
        ("%friend%request%",),
        "Friend-request history is unavailable in this export.",
    ),
    "followers": (
        "followers",
        "Followers",
        ("%followers%", "%following%"),
        "Follower history is unavailable in this export.",
    ),
    "facebook-blocks": (
        "facebook_blocks",
        "Facebook blocks",
        ("%facebook%block%", "%blocked_profiles%"),
        "Facebook block history is unavailable in this export.",
    ),
    "messenger-blocks": (
        "messenger_blocks",
        "Messenger blocks",
        ("%messenger%block%", "%blocked_messages%"),
        "Messenger block history is unavailable in this export.",
    ),
}


def _relationship_sort_expression(sort: RelationshipSort) -> Any:
    return {
        "name": func.coalesce(ActivityRecord.title_search, ActivityRecord.actor_search, ""),
        "date": ActivityRecord.timestamp_ms,
        "status": ActivityRecord.subtype,
    }[sort]


def _relationship_seek_condition(
    *,
    null_rank: ColumnElement[int],
    value: ColumnElement[Any],
    cursor_null: int,
    cursor_value: str | int,
    cursor_id: str,
    direction: Direction,
    inclusive: bool = False,
) -> ColumnElement[bool]:
    if direction == "asc":
        id_seek = ActivityRecord.id >= cursor_id if inclusive else ActivityRecord.id > cursor_id
        value_seek = value > cursor_value
    else:
        id_seek = ActivityRecord.id <= cursor_id if inclusive else ActivityRecord.id < cursor_id
        value_seek = value < cursor_value
    if cursor_null == 1:
        return and_(null_rank == 1, id_seek)
    return or_(
        null_rank > cursor_null,
        and_(null_rank == cursor_null, or_(value_seek, and_(value == cursor_value, id_seek))),
    )


def _relationship_record_count(session: Session, filters: list[Any]) -> int:
    return int(
        session.scalar(
            select(func.count(ActivityRecord.pk))
            .join(SourceFile, SourceFile.id == ActivityRecord.source_file_id)
            .where(*filters)
        )
        or 0
    )


def _relationship_page(
    request: Request,
    *,
    view: str,
    q: str | None,
    sort: RelationshipSort,
    direction: Direction,
    limit: int,
    cursor: str | None,
    around: str | None,
) -> dict[str, Any]:
    if around is not None and _OPAQUE_RELATIONSHIP_ID.fullmatch(around) is None:
        raise HTTPException(status_code=422, detail="Invalid relationship around target")
    coverage_key, label, patterns, description = _RELATIONSHIP_VIEW[view]
    path_value = func.lower(SourceFile.normalized_path)
    source_filters = [
        ActivityRecord.deleted_at.is_(None),
        or_(*(path_value.like(pattern) for pattern in patterns)),
    ]
    filters = list(source_filters)
    if q:
        normalized = escaped_like(normalize_text(q).search)
        filters.append(
            or_(
                ActivityRecord.title_search.like(f"%{normalized}%", escape="\\"),
                ActivityRecord.content_search.like(f"%{normalized}%", escape="\\"),
                ActivityRecord.actor_search.like(f"%{normalized}%", escape="\\"),
            )
        )
    scope = scope_hash({"view": view, "q": q or "", "sort": sort, "direction": direction})
    sort_value = _relationship_sort_expression(sort)
    null_rank = case((sort_value.is_(None), 1), else_=0)
    statement = (
        select(ActivityRecord, SourceChunk.chunk_key, SourceFile.relative_path)
        .join(SourceFile, SourceFile.id == ActivityRecord.source_file_id)
        .join(SourceChunk, SourceChunk.id == SourceFile.chunk_id)
        .where(*filters)
    )
    if cursor:
        try:
            decoded = decode_cursor(cursor, kind="relationships")
            if (
                require_cursor_str(decoded, "scope") != scope
                or require_cursor_str(decoded, "sort") != sort
                or require_cursor_str(decoded, "direction") != direction
            ):
                raise InvalidCursor("Pagination cursor does not match these relationship filters")
            cursor_null = require_cursor_int(decoded, "null")
            cursor_value: str | int = (
                require_cursor_int(decoded, "value")
                if sort == "date"
                else require_cursor_str(decoded, "value")
            )
            cursor_id = require_cursor_str(decoded, "id")
        except InvalidCursor as exc:
            raise invalid_cursor(str(exc)) from exc
        statement = statement.where(
            _relationship_seek_condition(
                null_rank=null_rank,
                value=sort_value,
                cursor_null=cursor_null,
                cursor_value=cursor_value,
                cursor_id=cursor_id,
                direction=direction,
            )
        )
    value_order = sort_value.asc() if direction == "asc" else sort_value.desc()
    id_order = ActivityRecord.id.asc() if direction == "asc" else ActivityRecord.id.desc()
    with session_for(request) as session:
        aggregate_memo = catalog_aggregate_memo(request, session)
        source_total = aggregate_memo.get_or_compute(
            namespace="relationship-source-total-v1",
            scope=scope_hash({"view": view}),
            compute=lambda: _relationship_record_count(session, source_filters),
        )
        total = (
            aggregate_memo.get_or_compute(
                namespace="relationship-filter-total-v1",
                scope=scope_hash({"view": view, "q": q}),
                compute=lambda: _relationship_record_count(session, filters),
            )
            if q
            else source_total
        )
        around_target: ActivityRecord | None = None
        if around is not None:
            around_target = session.scalar(
                select(ActivityRecord)
                .join(SourceFile, SourceFile.id == ActivityRecord.source_file_id)
                .where(ActivityRecord.id == around, *source_filters)
            )
            if around_target is None:
                raise HTTPException(status_code=404, detail="Relationship target not found")
            matching_target = session.scalar(
                select(ActivityRecord.id)
                .join(SourceFile, SourceFile.id == ActivityRecord.source_file_id)
                .where(ActivityRecord.id == around, *filters)
            )
            if matching_target is None:
                raise HTTPException(
                    status_code=422,
                    detail="Relationship target does not match the active filters",
                )
            if cursor is None:
                target_value: str | int | None = {
                    "name": around_target.title_search or around_target.actor_search or "",
                    "date": around_target.timestamp_ms,
                    "status": around_target.subtype,
                }[sort]
                statement = statement.where(
                    _relationship_seek_condition(
                        null_rank=null_rank,
                        value=sort_value,
                        cursor_null=1 if target_value is None else 0,
                        cursor_value=0 if target_value is None else target_value,
                        cursor_id=around_target.id,
                        direction=direction,
                        inclusive=True,
                    )
                )
        coverage = coverage_payload(
            session,
            coverage_key,
            label=label,
            unavailable_description=description,
            actual_count=source_total,
            status_with_data="partial",
        )
        if source_total:
            coverage["description"] = (
                f"Exported {label.casefold()} records are available. They remain separate from "
                "Messenger-derived contacts."
            )
        if source_total == 0 and coverage["status"] == "unavailable":
            aggregate_memo.require_current()
            return unavailable_page(coverage, limit=limit) | {
                "authority": "source_records_required",
                "contacts_are_not_included": True,
            }
        rows = list(
            session.execute(
                statement.order_by(null_rank.asc(), value_order, id_order).limit(limit + 1)
            )
        )
        relationship_name_searches = {
            value
            for row, _chunk, _path in rows[:limit]
            for value in (row.title_search or row.actor_search,)
            if value
        }
        contact_projections = (
            _contacts_projection(request, session, aggregate_memo)
            if relationship_name_searches
            else ()
        )
        contact_by_name = {
            normalize_text(contact.name).search: contact
            for contact in contact_projections
            if contact.name and normalize_text(contact.name).search in relationship_name_searches
        }
        aggregate_memo.require_current()
    page_rows = rows[:limit]
    next_cursor = None
    if len(rows) > limit and page_rows:
        last = page_rows[-1][0]
        raw_value: str | int | None = {
            "name": last.title_search or last.actor_search or "",
            "date": last.timestamp_ms,
            "status": last.subtype,
        }[sort]
        next_cursor = encode_cursor(
            "relationships",
            scope=scope,
            sort=sort,
            direction=direction,
            null=1 if raw_value is None else 0,
            value=0 if raw_value is None else raw_value,
            id=last.id,
        )
    items: list[dict[str, Any]] = []
    for row, chunk, path in page_rows:
        name = row.title_display or row.actor_display
        matched_contact = contact_by_name.get(normalize_text(name).search) if name else None
        contact = _contact_payload(matched_contact) if matched_contact is not None else None
        item: dict[str, Any] = {
            "id": row.id,
            "name": name,
            "status": row.subtype,
            "timestamp_ms": row.timestamp_ms,
            "source": {
                "source_file_id": row.source_file_id,
                "chunk": chunk,
                "relative_path": path,
                "json_pointer": row.json_pointer,
            },
            "jump": {
                "href": f"/people?view={view}&around={row.id}#relationship-{row.id}",
                "fragment": f"relationship-{row.id}",
            },
        }
        if contact is not None:
            # The export has participant labels but no stable Facebook person IDs. Exact
            # normalized-label matching is useful for the requested conversation/media links,
            # but it must remain visibly provisional rather than becoming an identity claim.
            item.update(
                {
                    "contact_basis": "exact_normalized_messenger_label_match",
                    "counts": contact["counts"],
                    "direct_counts": contact["direct_counts"],
                    "shared_group_counts": contact["shared_group_counts"],
                    "combined_thread_counts": contact["combined_thread_counts"],
                    "conversation_count": contact["conversation_count"],
                    "direct_conversation_count": contact["direct_conversation_count"],
                    "group_conversation_count": contact["group_conversation_count"],
                    "first_activity_ms": contact["first_activity_ms"],
                    "last_activity_ms": contact["last_activity_ms"],
                    "conversations": contact["conversations"],
                    "count_semantics": (
                        "Conversation statistics are linked provisionally by an exact normalized "
                        "display-label match; this is not a Facebook identity match. The counts "
                        "and direct_counts fields describe the one-to-one history. Shared-group "
                        "counts cover whole group conversations and are not attributed to this "
                        "person; combined-thread counts explicitly sum both scopes."
                    ),
                    "messenger_contact_match": {
                        "contact_id": contact["id"],
                        "match_basis": "exact_normalized_display_label",
                        "authoritative_identity": False,
                    },
                }
            )
        items.append(item)
    with session_for(request) as session:
        attach_star_states(
            session,
            items,
            lambda item: ("activity", str(item["id"])),
        )

    return {
        "coverage": coverage,
        "authority": "exported_relationship_records",
        "contacts_are_not_included": True,
        "messenger_contact_matches_are_provisional": True,
        "around": (
            {"requested": around, "relationship_id": around, "resolved": True}
            if around is not None
            else None
        ),
        "items": items,
        "page": {
            "total": total,
            "limit": limit,
            "next_cursor": next_cursor,
            "count_is_unknown": False,
        },
        "filters": {"view": view, "q": q, "sort": sort, "direction": direction},
    }


def _contacts_page(
    request: Request,
    *,
    q: str | None,
    conversation_type: str | None,
    sort: ContactSort,
    direction: Direction,
    limit: int,
    cursor: str | None,
    around: str | None,
) -> dict[str, Any]:
    if conversation_type not in {None, "direct", "group"}:
        raise HTTPException(status_code=422, detail="Unknown contact conversation type")
    scope = scope_hash(
        {"q": q or "", "conversation_type": conversation_type, "sort": sort, "direction": direction}
    )
    with session_for(request) as session:
        aggregate_memo = catalog_aggregate_memo(request, session)
        all_items = _contacts_projection(request, session, aggregate_memo)
        items = list(all_items)
        if q:
            needle = normalize_text(q).search
            items = [item for item in items if needle in normalize_text(item.name).search]
        if conversation_type == "direct":
            items = [item for item in items if item.direct_conversation_count]
        elif conversation_type == "group":
            items = [item for item in items if item.group_conversation_count]
        _sort_contacts(items, sort, direction)
        coverage = {
            "key": "messenger_contacts",
            "label": "Messenger contacts",
            "status": "derived" if all_items else "unavailable",
            "raw_count": len(all_items) if all_items else None,
            "logical_count": len(all_items) if all_items else None,
            "description": (
                "Contacts are participant labels from direct and group conversations. They are "
                "not authoritative Facebook friends."
            ),
            "absence_is_not_zero_history": not all_items,
        }
        start = 0
        if cursor and around is not None:
            raise HTTPException(status_code=422, detail="Choose a cursor or contact target")
        if cursor:
            try:
                decoded = decode_cursor(cursor, kind="contacts")
                if require_cursor_str(decoded, "scope") != scope:
                    raise InvalidCursor("Pagination cursor does not match these contact filters")
                cursor_id = require_cursor_str(decoded, "id")
                start = next(index + 1 for index, item in enumerate(items) if item.id == cursor_id)
            except (InvalidCursor, StopIteration) as exc:
                raise invalid_cursor(str(exc) or "Contact cursor no longer exists") from exc
        elif around is not None:
            try:
                start = next(index for index, item in enumerate(items) if item.id == around)
            except StopIteration as exc:
                known_contact = any(item.id == around for item in all_items)
                if not known_contact:
                    raise HTTPException(
                        status_code=404,
                        detail="Messenger contact not found",
                    ) from exc
                raise HTTPException(
                    status_code=422,
                    detail="Messenger contact does not match the active filters",
                ) from exc
        page_items = items[start : start + limit]
        next_cursor = None
        if start + limit < len(items) and page_items:
            next_cursor = encode_cursor("contacts", scope=scope, id=page_items[-1].id)
        total = len(items) if all_items else None
        count_is_unknown = not all_items
        payload_items = [_contact_payload(item) for item in page_items]
        attach_star_states(
            session,
            payload_items,
            lambda item: ("messenger_contact", str(item["id"])),
        )
        aggregate_memo.require_current()
    return {
        "coverage": coverage,
        "authority": "messenger_contact_only",
        "around": (
            {"requested": around, "contact_id": around, "resolved": True}
            if around is not None
            else None
        ),
        "items": payload_items,
        "page": {
            "total": total,
            "limit": limit,
            "next_cursor": next_cursor,
            "count_is_unknown": count_is_unknown,
        },
    }


@people_router.get("/people")
def people(
    request: Request,
    view: PeopleView = "contacts",
    q: str | None = Query(default=None, max_length=300),
    conversation_type: str | None = Query(default=None, max_length=20),
    sort: PeopleSort = "name",
    direction: Direction = "asc",
    limit: int = Query(default=100, ge=1, le=250),
    cursor: str | None = Query(default=None, max_length=2048),
    around: str | None = Query(default=None, min_length=1, max_length=128),
) -> dict[str, Any]:
    if view == "contacts":
        if sort not in _CONTACT_SORTS:
            raise HTTPException(status_code=422, detail="Unknown contact sort")
        return _contacts_page(
            request,
            q=q,
            conversation_type=conversation_type,
            sort=cast(ContactSort, sort),
            direction=direction,
            limit=limit,
            cursor=cursor,
            around=around,
        )
    if sort not in _RELATIONSHIP_SORTS:
        raise HTTPException(status_code=422, detail="Unknown relationship sort")
    return _relationship_page(
        request,
        view=view,
        q=q,
        sort=cast(RelationshipSort, sort),
        direction=direction,
        limit=limit,
        cursor=cursor,
        around=around,
    )


@people_router.get("/people/views/{view}")
def people_view(
    request: Request,
    view: PeopleView,
    q: str | None = Query(default=None, max_length=300),
    sort: PeopleSort = "name",
    direction: Direction = "asc",
    limit: int = Query(default=100, ge=1, le=250),
    cursor: str | None = Query(default=None, max_length=2048),
    around: str | None = Query(default=None, min_length=1, max_length=128),
) -> dict[str, Any]:
    return people(
        request,
        view=view,
        q=q,
        conversation_type=None,
        sort=sort,
        direction=direction,
        limit=limit,
        cursor=cursor,
        around=around,
    )


@people_router.get("/people/contacts/{contact_id}")
def contact_detail(request: Request, contact_id: str) -> dict[str, Any]:
    with session_for(request) as session:
        aggregate_memo = catalog_aggregate_memo(request, session)
        contact_projection = next(
            (
                item
                for item in _contacts_projection(request, session, aggregate_memo)
                if item.id == contact_id
            ),
            None,
        )
        contact = _contact_payload(contact_projection) if contact_projection is not None else None
        if contact is not None:
            attach_star_states(
                session,
                [contact],
                lambda item: ("messenger_contact", str(item["id"])),
            )
        aggregate_memo.require_current()
    if contact_projection is None:
        raise HTTPException(status_code=404, detail="Messenger contact not found")
    assert contact is not None
    return {
        "authority": "messenger_contact_only",
        "friendship_history": {
            "status": "unavailable",
            "is_friend": None,
            "description": "No authoritative friend relationship can be inferred from chat data.",
        },
        "contact": contact,
    }
