from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import and_, case, func, or_, select
from sqlalchemy.orm import Session
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
from meta_archive.models import (
    AttachmentOccurrence,
    Conversation,
    ConversationSegment,
    ConversationSegmentOverride,
    Message,
    MessageReaction,
    Participant,
    SegmentParticipant,
)
from meta_archive.services.media_derivatives import derivatives_by_blob
from meta_archive.services.text import normalize_text

conversation_router = APIRouter(
    prefix="/api/v1/conversations", dependencies=[Depends(require_session)]
)

ConversationDirection = Literal["asc", "desc"]
ConversationStart = Literal["latest", "earliest"]


def _attachment_media_type_expression() -> ColumnElement[str]:
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


def _conversation_media_count(media_class: str) -> ColumnElement[int]:
    return (
        select(func.count(AttachmentOccurrence.id))
        .where(
            AttachmentOccurrence.conversation_id == Conversation.id,
            AttachmentOccurrence.deleted_at.is_(None),
            _attachment_media_type_expression() == media_class,
        )
        .correlate(Conversation)
        .scalar_subquery()
    )


CONVERSATION_MEDIA_SORTS: dict[str, ColumnElement[int]] = {
    "photos": _conversation_media_count("photo"),
    "videos": _conversation_media_count("video"),
    "audio": _conversation_media_count("audio"),
}

CONVERSATION_SORTS: dict[str, Any] = {
    "title": Conversation.title_search,
    "type": Conversation.conversation_type,
    "source_category": Conversation.primary_category,
    "participants": Conversation.participant_count,
    "first_activity": Conversation.first_activity_ms,
    "last_activity": Conversation.last_activity_ms,
    "messages": Conversation.message_count,
    "sent": Conversation.sent_count,
    "received": Conversation.received_count,
    "attachments": Conversation.attachment_count,
    "unique_attachments": Conversation.unique_attachment_count,
    "attachment_bytes": Conversation.attachment_bytes,
    "active_days": Conversation.active_days,
    "completeness": Conversation.completeness,
    **CONVERSATION_MEDIA_SORTS,
}


def _session(request: Request) -> Session:
    return Session(request.app.state.engine)


def _not_found(entity: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"{entity} not found")


def _invalid_cursor(exc: InvalidCursor) -> HTTPException:
    return HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc))


def _csv_values(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))


def _scope_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def _conversation_filters(
    *,
    q: str | None,
    conversation_type: str | None,
    source_category: str | None,
    completeness: str | None,
    min_participants: int | None,
    max_participants: int | None,
    min_messages: int | None,
    max_messages: int | None,
    activity_from_ms: int | None,
    activity_to_ms: int | None,
    has_attachments: bool | None,
    min_photos: int | None,
    max_photos: int | None,
    min_videos: int | None,
    max_videos: int | None,
    min_audio: int | None,
    max_audio: int | None,
) -> list[ColumnElement[bool]]:
    filters: list[ColumnElement[bool]] = [Conversation.deleted_at.is_(None)]
    if q:
        normalized = normalize_text(q).search
        escaped = normalized.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        filters.append(Conversation.title_search.like(f"%{escaped}%", escape="\\"))
    types = _csv_values(conversation_type)
    if types:
        filters.append(Conversation.conversation_type.in_(types))
    categories = _csv_values(source_category)
    if categories:
        filters.append(
            Conversation.id.in_(
                select(ConversationSegment.conversation_id).where(
                    ConversationSegment.source_category.in_(categories)
                )
            )
        )
    completeness_values = _csv_values(completeness)
    if completeness_values:
        filters.append(Conversation.completeness.in_(completeness_values))
    if min_participants is not None:
        filters.append(Conversation.participant_count >= min_participants)
    if max_participants is not None:
        filters.append(Conversation.participant_count <= max_participants)
    if min_messages is not None:
        filters.append(Conversation.message_count >= min_messages)
    if max_messages is not None:
        filters.append(Conversation.message_count <= max_messages)
    if activity_from_ms is not None:
        filters.append(Conversation.last_activity_ms >= activity_from_ms)
    if activity_to_ms is not None:
        filters.append(Conversation.first_activity_ms <= activity_to_ms)
    if has_attachments is True:
        filters.append(Conversation.attachment_count > 0)
    elif has_attachments is False:
        filters.append(Conversation.attachment_count == 0)
    for media_class, minimum, maximum in (
        ("photo", min_photos, max_photos),
        ("video", min_videos, max_videos),
        ("audio", min_audio, max_audio),
    ):
        count_expression = _conversation_media_count(media_class)
        if minimum is not None:
            filters.append(count_expression >= minimum)
        if maximum is not None:
            filters.append(count_expression <= maximum)
    return filters


def _conversation_payload(
    row: Conversation,
    *,
    source_categories: list[str],
    segment_count: int,
    media_counts: dict[str, int],
    corrected_segment_count: int = 0,
) -> dict[str, Any]:
    has_messages = row.message_count > 0
    has_attachments = row.attachment_count > 0
    content_kind = (
        "messages_and_media"
        if has_messages and has_attachments
        else "messages_only"
        if has_messages
        else "media_only"
        if has_attachments
        else "metadata_only"
    )
    identity_status = (
        "corrected"
        if row.conversation_type == "direct" and corrected_segment_count
        else "provisional"
        if row.conversation_type == "direct"
        else "source"
    )
    return {
        "id": row.id,
        "title": row.title_display,
        "type": row.conversation_type,
        "primary_category": row.primary_category,
        "source_categories": source_categories,
        "participant_count": row.participant_count,
        "first_activity_ms": row.first_activity_ms,
        "last_activity_ms": row.last_activity_ms,
        "message_counts": {
            "total": row.message_count,
            "sent": row.sent_count,
            "received": row.received_count,
        },
        "content_profile": {
            "kind": content_kind,
            "message_records": "available" if has_messages else "not_present",
            "attachment_records": "available" if has_attachments else "not_present",
        },
        "attachment_counts": {
            "occurrences": row.attachment_count,
            "unique": row.unique_attachment_count,
        },
        "attachment_bytes": row.attachment_bytes,
        "media_counts": media_counts,
        "active_days": row.active_days,
        "completeness": row.completeness,
        "segment_count": segment_count,
        "identity": {
            "status": identity_status,
            "basis": (
                "manual_segment_correction"
                if identity_status == "corrected"
                else "exact_counterpart_display_label"
                if identity_status == "provisional"
                else "source_thread"
            ),
            "corrected_segment_count": corrected_segment_count,
            "can_correct": row.conversation_type == "direct",
        },
        "href": f"/conversations/{row.id}",
    }


def _conversation_page_metadata(
    session: Session, conversation_ids: list[str]
) -> tuple[
    dict[str, list[str]],
    dict[str, int],
    dict[str, dict[str, int]],
    dict[str, int],
]:
    categories: dict[str, list[str]] = defaultdict(list)
    counts: dict[str, int] = {}
    media_counts = {
        conversation_id: {
            "photo": 0,
            "video": 0,
            "audio": 0,
            "gif": 0,
            "sticker": 0,
            "document": 0,
            "other": 0,
        }
        for conversation_id in conversation_ids
    }
    if not conversation_ids:
        return categories, counts, media_counts, {}
    rows = session.execute(
        select(
            ConversationSegment.conversation_id,
            ConversationSegment.source_category,
            func.count(ConversationSegment.id),
        )
        .where(ConversationSegment.conversation_id.in_(conversation_ids))
        .group_by(
            ConversationSegment.conversation_id,
            ConversationSegment.source_category,
        )
        .order_by(
            ConversationSegment.conversation_id,
            ConversationSegment.source_category,
        )
    )
    for conversation_id, source_category, count in rows:
        categories[conversation_id].append(source_category)
        counts[conversation_id] = counts.get(conversation_id, 0) + count
    media_rows = session.execute(
        select(
            AttachmentOccurrence.conversation_id,
            _attachment_media_type_expression().label("media_class"),
            func.count(AttachmentOccurrence.id),
        )
        .where(
            AttachmentOccurrence.conversation_id.in_(conversation_ids),
            AttachmentOccurrence.deleted_at.is_(None),
        )
        .group_by(
            AttachmentOccurrence.conversation_id,
            _attachment_media_type_expression(),
        )
    )
    for conversation_id, media_class, count in media_rows:
        if conversation_id is not None:
            media_counts[conversation_id][media_class] = count
    corrected_counts = {
        conversation_id: int(count)
        for conversation_id, count in session.execute(
            select(
                ConversationSegment.conversation_id,
                func.count(ConversationSegmentOverride.segment_id),
            )
            .join(
                ConversationSegmentOverride,
                ConversationSegmentOverride.segment_id == ConversationSegment.id,
            )
            .where(ConversationSegment.conversation_id.in_(conversation_ids))
            .group_by(ConversationSegment.conversation_id)
        )
    }
    return categories, counts, media_counts, corrected_counts


@conversation_router.get("")
def conversations(
    request: Request,
    q: str | None = Query(default=None, max_length=200),
    conversation_type: str | None = Query(default=None, max_length=200),
    source_category: str | None = Query(default=None, max_length=500),
    completeness: str | None = Query(default=None, max_length=200),
    min_participants: int | None = Query(default=None, ge=0),
    max_participants: int | None = Query(default=None, ge=0),
    min_messages: int | None = Query(default=None, ge=0),
    max_messages: int | None = Query(default=None, ge=0),
    activity_from_ms: int | None = Query(default=None, ge=0),
    activity_to_ms: int | None = Query(default=None, ge=0),
    has_attachments: bool | None = None,
    min_photos: int | None = Query(default=None, ge=0),
    max_photos: int | None = Query(default=None, ge=0),
    min_videos: int | None = Query(default=None, ge=0),
    max_videos: int | None = Query(default=None, ge=0),
    min_audio: int | None = Query(default=None, ge=0),
    max_audio: int | None = Query(default=None, ge=0),
    sort: str = Query(default="last_activity"),
    direction: ConversationDirection = "desc",
    limit: int = Query(default=50, ge=1, le=250),
    cursor: str | None = Query(default=None, max_length=2048),
) -> dict[str, Any]:
    if sort not in CONVERSATION_SORTS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown conversation sort: {sort}",
        )
    if (
        min_participants is not None
        and max_participants is not None
        and min_participants > max_participants
    ):
        raise HTTPException(status_code=422, detail="Invalid participant range")
    if min_messages is not None and max_messages is not None and min_messages > max_messages:
        raise HTTPException(status_code=422, detail="Invalid message range")
    if (
        activity_from_ms is not None
        and activity_to_ms is not None
        and activity_from_ms > activity_to_ms
    ):
        raise HTTPException(status_code=422, detail="Invalid activity range")
    for label, minimum, maximum in (
        ("photo", min_photos, max_photos),
        ("video", min_videos, max_videos),
        ("audio", min_audio, max_audio),
    ):
        if minimum is not None and maximum is not None and minimum > maximum:
            raise HTTPException(status_code=422, detail=f"Invalid {label} range")

    filters = _conversation_filters(
        q=q,
        conversation_type=conversation_type,
        source_category=source_category,
        completeness=completeness,
        min_participants=min_participants,
        max_participants=max_participants,
        min_messages=min_messages,
        max_messages=max_messages,
        activity_from_ms=activity_from_ms,
        activity_to_ms=activity_to_ms,
        has_attachments=has_attachments,
        min_photos=min_photos,
        max_photos=max_photos,
        min_videos=min_videos,
        max_videos=max_videos,
        min_audio=min_audio,
        max_audio=max_audio,
    )
    scope = _scope_hash(
        {
            "q": q or "",
            "conversation_type": _csv_values(conversation_type),
            "source_category": _csv_values(source_category),
            "completeness": _csv_values(completeness),
            "min_participants": min_participants,
            "max_participants": max_participants,
            "min_messages": min_messages,
            "max_messages": max_messages,
            "activity_from_ms": activity_from_ms,
            "activity_to_ms": activity_to_ms,
            "has_attachments": has_attachments,
            "min_photos": min_photos,
            "max_photos": max_photos,
            "min_videos": min_videos,
            "max_videos": max_videos,
            "min_audio": min_audio,
            "max_audio": max_audio,
        }
    )
    sort_expression = CONVERSATION_SORTS[sort]
    null_rank = case((sort_expression.is_(None), 1), else_=0)
    value_order = sort_expression.asc() if direction == "asc" else sort_expression.desc()
    id_order = Conversation.id.asc() if direction == "asc" else Conversation.id.desc()

    statement = select(Conversation, sort_expression.label("sort_value")).where(*filters)
    if cursor is not None:
        try:
            decoded = decode_cursor(cursor, kind="conversations")
            if (
                require_cursor_str(decoded, "sort") != sort
                or require_cursor_str(decoded, "direction") != direction
                or require_cursor_str(decoded, "scope") != scope
            ):
                raise InvalidCursor("Pagination cursor does not match the current query")
            cursor_id = require_cursor_str(decoded, "id")
            cursor_null = require_cursor_int(decoded, "null")
            cursor_value = decoded.get("value")
            if cursor_null not in {0, 1} or not isinstance(cursor_value, (str, int, type(None))):
                raise InvalidCursor("Invalid pagination cursor")
        except InvalidCursor as exc:
            raise _invalid_cursor(exc) from exc
        id_seek = Conversation.id > cursor_id if direction == "asc" else Conversation.id < cursor_id
        if cursor_null == 1:
            seek = and_(null_rank == 1, id_seek)
        else:
            value_seek = (
                sort_expression > cursor_value
                if direction == "asc"
                else sort_expression < cursor_value
            )
            seek = or_(
                null_rank > cursor_null,
                and_(
                    null_rank == cursor_null,
                    or_(value_seek, and_(sort_expression == cursor_value, id_seek)),
                ),
            )
        statement = statement.where(seek)

    with _session(request) as session:
        aggregate_memo = catalog_aggregate_memo(request, session)
        rows = session.execute(
            statement.order_by(null_rank.asc(), value_order, id_order).limit(limit + 1)
        ).all()
        has_more = len(rows) > limit
        page_rows = rows[:limit]
        page = [result[0] for result in page_rows]
        ids = [row.id for row in page]
        categories, segment_counts, media_counts, corrected_counts = _conversation_page_metadata(
            session, ids
        )
        total = aggregate_memo.get_or_compute(
            namespace="conversation-list-total-v1",
            scope=scope,
            compute=lambda: int(
                session.scalar(select(func.count(Conversation.id)).where(*filters)) or 0
            ),
        )
        aggregate_memo.require_current()

    next_cursor: str | None = None
    if has_more and page:
        last = page[-1]
        value = page_rows[-1].sort_value
        next_cursor = encode_cursor(
            "conversations",
            sort=sort,
            direction=direction,
            scope=scope,
            id=last.id,
            null=1 if value is None else 0,
            value=value,
        )
    return {
        "items": [
            _conversation_payload(
                row,
                source_categories=categories.get(row.id, []),
                segment_count=segment_counts.get(row.id, 0),
                media_counts=media_counts[row.id],
                corrected_segment_count=corrected_counts.get(row.id, 0),
            )
            for row in page
        ],
        "page": {"limit": limit, "total": total, "next_cursor": next_cursor},
        "sort": {"field": sort, "direction": direction},
    }


def _participant_rows(session: Session, segment_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not segment_ids:
        return result
    rows = session.execute(
        select(
            SegmentParticipant.segment_id,
            SegmentParticipant.ordinal,
            Participant.id,
            Participant.name_display,
            Participant.is_owner,
        )
        .join(Participant, Participant.id == SegmentParticipant.participant_id)
        .where(SegmentParticipant.segment_id.in_(segment_ids))
        .order_by(SegmentParticipant.segment_id, SegmentParticipant.ordinal)
    )
    for segment_id, ordinal, participant_id, name, is_owner in rows:
        result[segment_id].append(
            {
                "id": participant_id,
                "name": name,
                "is_owner": is_owner,
                "ordinal": ordinal,
            }
        )
    return result


@conversation_router.get("/{conversation_id}")
def conversation_detail(request: Request, conversation_id: str) -> dict[str, Any]:
    with _session(request) as session:
        conversation = session.scalar(
            select(Conversation).where(
                Conversation.id == conversation_id, Conversation.deleted_at.is_(None)
            )
        )
        if conversation is None:
            raise _not_found("Conversation")
        segments = session.scalars(
            select(ConversationSegment)
            .where(ConversationSegment.conversation_id == conversation_id)
            .order_by(
                ConversationSegment.first_activity_ms.is_(None),
                ConversationSegment.first_activity_ms,
                ConversationSegment.sequence,
                ConversationSegment.id,
            )
        ).all()
        segment_ids = [segment.id for segment in segments]
        participants = _participant_rows(session, segment_ids)
        source_categories = sorted({segment.source_category for segment in segments})
        _categories, _segment_counts, media_counts, corrected_counts = _conversation_page_metadata(
            session, [conversation.id]
        )
        overrides = {
            row.segment_id: row
            for row in session.scalars(
                select(ConversationSegmentOverride).where(
                    ConversationSegmentOverride.segment_id.in_(segment_ids)
                )
            )
        }

    return {
        "conversation": _conversation_payload(
            conversation,
            source_categories=source_categories,
            segment_count=len(segments),
            media_counts=media_counts[conversation.id],
            corrected_segment_count=corrected_counts.get(conversation.id, 0),
        ),
        "segments": [
            {
                "id": segment.id,
                "title": segment.title_display,
                "thread_path": segment.thread_path,
                "source_category": segment.source_category,
                "sequence": segment.sequence,
                "source_chunk_count": segment.source_chunk_count,
                "is_still_participant": segment.is_still_participant,
                "first_activity_ms": segment.first_activity_ms,
                "last_activity_ms": segment.last_activity_ms,
                "completeness": segment.completeness,
                "assignment": {
                    "status": "corrected" if segment.id in overrides else "provisional",
                    "correction_id": (
                        overrides[segment.id].correction_id if segment.id in overrides else None
                    ),
                    "base_conversation_id": (
                        overrides[segment.id].base_conversation_id
                        if segment.id in overrides
                        else conversation.id
                    ),
                },
                "participants": participants.get(segment.id, []),
            }
            for segment in segments
        ],
    }


def _message_after(timestamp_ms: int, message_id: str) -> ColumnElement[bool]:
    return or_(
        Message.timestamp_ms > timestamp_ms,
        and_(Message.timestamp_ms == timestamp_ms, Message.id > message_id),
    )


def _message_before(timestamp_ms: int, message_id: str) -> ColumnElement[bool]:
    return or_(
        Message.timestamp_ms < timestamp_ms,
        and_(Message.timestamp_ms == timestamp_ms, Message.id < message_id),
    )


def _message_anchor(session: Session, conversation_id: str, message_id: str) -> Message:
    row = session.scalar(
        select(Message).where(
            Message.id == message_id,
            Message.conversation_id == conversation_id,
            Message.deleted_at.is_(None),
        )
    )
    if row is None:
        raise _not_found("Message anchor")
    return row


def _message_jump(conversation_id: str, message_id: str) -> dict[str, str]:
    return {
        "message_id": message_id,
        "fragment": f"message-{message_id}",
        "href": f"/conversations/{conversation_id}?around={message_id}#message-{message_id}",
    }


def _attachment_payload(
    row: AttachmentOccurrence,
    *,
    derivatives: list[dict[str, object]] | None = None,
) -> dict[str, Any]:
    blob_href = f"/api/v1/blobs/{row.blob_sha256}" if row.blob_sha256 else None
    return {
        "id": row.id,
        "kind": row.kind,
        "filename": row.filename,
        "media_type": row.media_type,
        "size_bytes": row.size_bytes,
        "timestamp_ms": row.timestamp_ms,
        "availability": row.availability,
        "blob_sha256": row.blob_sha256,
        "blob_href": blob_href,
        "is_external_reference": row.availability == "external",
        "derivatives": derivatives or [],
        "jump": (
            _message_jump(row.conversation_id, row.message_id)
            if row.conversation_id and row.message_id
            else None
        ),
    }


def _message_payloads(session: Session, rows: list[Message]) -> list[dict[str, Any]]:
    if not rows:
        return []
    message_ids = [row.id for row in rows]
    reactions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reaction_rows = session.scalars(
        select(MessageReaction)
        .where(MessageReaction.message_id.in_(message_ids))
        .order_by(MessageReaction.message_id, MessageReaction.ordinal, MessageReaction.id)
    )
    for reaction in reaction_rows:
        reactions[reaction.message_id].append(
            {
                "id": reaction.id,
                "actor": reaction.actor_display,
                "reaction": reaction.reaction_display,
                "timestamp_ms": reaction.timestamp_ms,
            }
        )

    attachments: dict[str, list[dict[str, Any]]] = defaultdict(list)
    attachment_rows = list(
        session.scalars(
            select(AttachmentOccurrence)
            .where(
                AttachmentOccurrence.message_id.in_(message_ids),
                AttachmentOccurrence.deleted_at.is_(None),
            )
            .order_by(
                AttachmentOccurrence.message_id,
                AttachmentOccurrence.kind,
                AttachmentOccurrence.ordinal,
                AttachmentOccurrence.id,
            )
        )
    )
    derivative_map = derivatives_by_blob(
        session, (attachment.blob_sha256 for attachment in attachment_rows)
    )
    for attachment in attachment_rows:
        if attachment.message_id is not None:
            attachments[attachment.message_id].append(
                _attachment_payload(
                    attachment,
                    derivatives=(
                        derivative_map.get(attachment.blob_sha256, [])
                        if attachment.blob_sha256 is not None
                        else []
                    ),
                )
            )

    segment_ids = list(dict.fromkeys(row.segment_id for row in rows))
    segments = {
        segment.id: {
            "id": segment.id,
            "title": segment.title_display,
            "source_category": segment.source_category,
            "completeness": segment.completeness,
        }
        for segment in session.scalars(
            select(ConversationSegment).where(ConversationSegment.id.in_(segment_ids))
        )
    }

    return [
        {
            "id": row.id,
            "conversation_id": row.conversation_id,
            "segment": segments.get(row.segment_id),
            "timestamp_ms": row.timestamp_ms,
            "sender": row.sender_display,
            "sender_is_owner": row.sender_is_owner,
            "content": row.content_display,
            "transcription": row.transcription_display,
            "kind": row.message_kind,
            "call": (
                {
                    "duration_seconds": row.call_duration_seconds,
                    "missed": row.is_missed_call,
                }
                if row.message_kind == "call"
                else None
            ),
            "has_payment_details": row.has_payment,
            "has_sensitive_ip": row.has_sensitive_ip,
            "is_unsent": row.is_unsent,
            "is_unavailable_body": row.is_unavailable_body,
            "is_duplicate": row.duplicate_of_id is not None,
            "duplicate_of_id": row.duplicate_of_id,
            "reactions": reactions.get(row.id, []),
            "attachments": attachments.get(row.id, []),
            "anchor": _message_jump(row.conversation_id, row.id),
        }
        for row in rows
    ]


def _has_messages(
    session: Session,
    *,
    conversation_id: str,
    condition: ColumnElement[bool],
) -> bool:
    return (
        session.scalar(
            select(Message.pk)
            .where(
                Message.conversation_id == conversation_id,
                Message.deleted_at.is_(None),
                condition,
            )
            .limit(1)
        )
        is not None
    )


def _message_page_cursors(
    session: Session,
    *,
    conversation_id: str,
    rows: list[Message],
) -> tuple[str | None, str | None]:
    if not rows:
        return None, None
    first = rows[0]
    last = rows[-1]
    has_before = _has_messages(
        session,
        conversation_id=conversation_id,
        condition=_message_before(first.timestamp_ms, first.id),
    )
    has_after = _has_messages(
        session,
        conversation_id=conversation_id,
        condition=_message_after(last.timestamp_ms, last.id),
    )
    before_cursor = (
        encode_cursor(
            "conversation-messages",
            conversation_id=conversation_id,
            side="before",
            timestamp_ms=first.timestamp_ms,
            id=first.id,
        )
        if has_before
        else None
    )
    after_cursor = (
        encode_cursor(
            "conversation-messages",
            conversation_id=conversation_id,
            side="after",
            timestamp_ms=last.timestamp_ms,
            id=last.id,
        )
        if has_after
        else None
    )
    return before_cursor, after_cursor


def _around_messages(
    session: Session,
    *,
    conversation_id: str,
    anchor: Message,
    limit: int,
) -> list[Message]:
    before_limit = limit // 2
    after_limit = limit - before_limit - 1
    common = (
        Message.conversation_id == conversation_id,
        Message.deleted_at.is_(None),
    )
    before = session.scalars(
        select(Message)
        .where(*common, _message_before(anchor.timestamp_ms, anchor.id))
        .order_by(Message.timestamp_ms.desc(), Message.id.desc())
        .limit(before_limit)
    ).all()
    after = session.scalars(
        select(Message)
        .where(*common, _message_after(anchor.timestamp_ms, anchor.id))
        .order_by(Message.timestamp_ms, Message.id)
        .limit(after_limit)
    ).all()
    return [*reversed(before), anchor, *after]


@conversation_router.get("/{conversation_id}/messages")
def conversation_messages(
    request: Request,
    conversation_id: str,
    limit: int = Query(default=100, ge=1, le=250),
    cursor: str | None = Query(default=None, max_length=2048),
    before: str | None = Query(default=None, min_length=1, max_length=128),
    after: str | None = Query(default=None, min_length=1, max_length=128),
    around: str | None = Query(default=None, min_length=1, max_length=128),
    at_ms: int | None = Query(default=None, ge=0),
    start: ConversationStart = "latest",
) -> dict[str, Any]:
    positions = [
        cursor is not None,
        before is not None,
        after is not None,
        around is not None,
        at_ms is not None,
    ]
    if sum(positions) > 1:
        raise HTTPException(status_code=422, detail="Choose only one message position")

    with _session(request) as session:
        aggregate_memo = catalog_aggregate_memo(request, session)
        exists = session.scalar(
            select(Conversation.id).where(
                Conversation.id == conversation_id, Conversation.deleted_at.is_(None)
            )
        )
        if exists is None:
            raise _not_found("Conversation")
        common = (
            Message.conversation_id == conversation_id,
            Message.deleted_at.is_(None),
        )
        anchor_message: Message | None = None
        rows: list[Message]

        if cursor is not None:
            try:
                decoded = decode_cursor(cursor, kind="conversation-messages")
                if require_cursor_str(decoded, "conversation_id") != conversation_id:
                    raise InvalidCursor("Pagination cursor belongs to another conversation")
                side = require_cursor_str(decoded, "side")
                timestamp_ms = require_cursor_int(decoded, "timestamp_ms")
                message_id = require_cursor_str(decoded, "id")
                if side not in {"before", "after"}:
                    raise InvalidCursor("Invalid pagination cursor")
            except InvalidCursor as exc:
                raise _invalid_cursor(exc) from exc
            condition = (
                _message_before(timestamp_ms, message_id)
                if side == "before"
                else _message_after(timestamp_ms, message_id)
            )
            if side == "before":
                rows = list(
                    session.scalars(
                        select(Message)
                        .where(*common, condition)
                        .order_by(Message.timestamp_ms.desc(), Message.id.desc())
                        .limit(limit)
                    ).all()
                )
                rows.reverse()
            else:
                rows = list(
                    session.scalars(
                        select(Message)
                        .where(*common, condition)
                        .order_by(Message.timestamp_ms, Message.id)
                        .limit(limit)
                    ).all()
                )
        elif before is not None:
            anchor_message = _message_anchor(session, conversation_id, before)
            rows = list(
                session.scalars(
                    select(Message)
                    .where(*common, _message_before(anchor_message.timestamp_ms, anchor_message.id))
                    .order_by(Message.timestamp_ms.desc(), Message.id.desc())
                    .limit(limit)
                ).all()
            )
            rows.reverse()
        elif after is not None:
            anchor_message = _message_anchor(session, conversation_id, after)
            rows = list(
                session.scalars(
                    select(Message)
                    .where(*common, _message_after(anchor_message.timestamp_ms, anchor_message.id))
                    .order_by(Message.timestamp_ms, Message.id)
                    .limit(limit)
                ).all()
            )
        elif around is not None:
            anchor_message = _message_anchor(session, conversation_id, around)
            rows = _around_messages(
                session, conversation_id=conversation_id, anchor=anchor_message, limit=limit
            )
        elif at_ms is not None:
            anchor_message = session.scalar(
                select(Message)
                .where(*common, Message.timestamp_ms >= at_ms)
                .order_by(Message.timestamp_ms, Message.id)
                .limit(1)
            )
            if anchor_message is None:
                anchor_message = session.scalar(
                    select(Message)
                    .where(*common)
                    .order_by(Message.timestamp_ms.desc(), Message.id.desc())
                    .limit(1)
                )
            rows = (
                _around_messages(
                    session, conversation_id=conversation_id, anchor=anchor_message, limit=limit
                )
                if anchor_message is not None
                else []
            )
        elif start == "earliest":
            rows = list(
                session.scalars(
                    select(Message)
                    .where(*common)
                    .order_by(Message.timestamp_ms, Message.id)
                    .limit(limit)
                ).all()
            )
        else:
            rows = list(
                session.scalars(
                    select(Message)
                    .where(*common)
                    .order_by(Message.timestamp_ms.desc(), Message.id.desc())
                    .limit(limit)
                ).all()
            )
            rows.reverse()

        before_cursor, after_cursor = _message_page_cursors(
            session, conversation_id=conversation_id, rows=rows
        )
        items = _message_payloads(session, rows)
        total = aggregate_memo.get_or_compute(
            namespace="conversation-message-total-v1",
            scope=_scope_hash({"conversation_id": conversation_id}),
            compute=lambda: int(session.scalar(select(func.count(Message.pk)).where(*common)) or 0),
        )
        aggregate_memo.require_current()

    return {
        "items": items,
        "page": {
            "limit": limit,
            "total": total,
            "before_cursor": before_cursor,
            "after_cursor": after_cursor,
        },
        "focus": _message_jump(conversation_id, anchor_message.id) if anchor_message else None,
    }


@conversation_router.get("/{conversation_id}/messages/date-jump")
def conversation_date_jump(
    request: Request,
    conversation_id: str,
    at_ms: int = Query(ge=0),
) -> dict[str, Any]:
    with _session(request) as session:
        if session.get(Conversation, conversation_id) is None:
            raise _not_found("Conversation")
        common = (
            Message.conversation_id == conversation_id,
            Message.deleted_at.is_(None),
        )
        target = session.scalar(
            select(Message)
            .where(*common, Message.timestamp_ms >= at_ms)
            .order_by(Message.timestamp_ms, Message.id)
            .limit(1)
        )
        if target is None:
            target = session.scalar(
                select(Message)
                .where(*common)
                .order_by(Message.timestamp_ms.desc(), Message.id.desc())
                .limit(1)
            )
    return {
        "requested_timestamp_ms": at_ms,
        "target_timestamp_ms": target.timestamp_ms if target else None,
        "jump": _message_jump(conversation_id, target.id) if target else None,
    }


@conversation_router.get("/{conversation_id}/messages/{message_id}/advanced")
def message_advanced_details(
    request: Request,
    conversation_id: str,
    message_id: str,
) -> dict[str, Any]:
    with _session(request) as session:
        row = _message_anchor(session, conversation_id, message_id)
        sensitive = dict(row.sensitive_json)
    return {
        "message_id": row.id,
        "has_sensitive_details": bool(sensitive),
        "sensitive": sensitive,
        "notice": "Advanced details can contain private network or payment data.",
    }


@conversation_router.get("/{conversation_id}/message-search")
def conversation_message_search(
    request: Request,
    conversation_id: str,
    q: str = Query(min_length=1, max_length=500),
    kind: str | None = Query(default=None, max_length=100),
    sender: str | None = Query(default=None, max_length=200),
    from_ms: int | None = Query(default=None, ge=0),
    to_ms: int | None = Query(default=None, ge=0),
    direction: ConversationDirection = "desc",
    limit: int = Query(default=50, ge=1, le=200),
    cursor: str | None = Query(default=None, max_length=2048),
) -> dict[str, Any]:
    if from_ms is not None and to_ms is not None and from_ms > to_ms:
        raise HTTPException(status_code=422, detail="Invalid date range")
    normalized = normalize_text(q).search
    escaped = normalized.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    pattern = f"%{escaped}%"
    match = or_(
        Message.content_search.like(pattern, escape="\\"),
        Message.transcription_search.like(pattern, escape="\\"),
        Message.sender_search.like(pattern, escape="\\"),
    )
    filters: list[ColumnElement[bool]] = [
        Message.conversation_id == conversation_id,
        Message.deleted_at.is_(None),
        match,
    ]
    kinds = _csv_values(kind)
    if kinds:
        filters.append(Message.message_kind.in_(kinds))
    if sender:
        sender_normalized = normalize_text(sender).search
        filters.append(Message.sender_search == sender_normalized)
    if from_ms is not None:
        filters.append(Message.timestamp_ms >= from_ms)
    if to_ms is not None:
        filters.append(Message.timestamp_ms <= to_ms)
    scope = _scope_hash(
        {
            "conversation_id": conversation_id,
            "q": normalized,
            "kind": kinds,
            "sender": normalize_text(sender).search if sender else None,
            "from_ms": from_ms,
            "to_ms": to_ms,
            "direction": direction,
        }
    )
    statement = select(Message).where(*filters)
    if cursor is not None:
        try:
            decoded = decode_cursor(cursor, kind="conversation-message-search")
            if require_cursor_str(decoded, "scope") != scope:
                raise InvalidCursor("Pagination cursor does not match the current search")
            timestamp_ms = require_cursor_int(decoded, "timestamp_ms")
            message_id = require_cursor_str(decoded, "id")
        except InvalidCursor as exc:
            raise _invalid_cursor(exc) from exc
        statement = statement.where(
            _message_after(timestamp_ms, message_id)
            if direction == "asc"
            else _message_before(timestamp_ms, message_id)
        )
    order = (
        (Message.timestamp_ms.asc(), Message.id.asc())
        if direction == "asc"
        else (Message.timestamp_ms.desc(), Message.id.desc())
    )
    with _session(request) as session:
        aggregate_memo = catalog_aggregate_memo(request, session)
        if (
            session.scalar(select(Conversation.id).where(Conversation.id == conversation_id))
            is None
        ):
            raise _not_found("Conversation")
        rows = list(session.scalars(statement.order_by(*order).limit(limit + 1)).all())
        has_more = len(rows) > limit
        page = rows[:limit]
        items = _message_payloads(session, page)
        total = aggregate_memo.get_or_compute(
            namespace="conversation-message-search-total-v1",
            scope=scope,
            compute=lambda: int(
                session.scalar(select(func.count(Message.pk)).where(*filters)) or 0
            ),
        )
        aggregate_memo.require_current()

    for item, row in zip(items, page, strict=True):
        matched_fields: list[str] = []
        if row.content_search and normalized in row.content_search:
            matched_fields.append("content")
        if row.transcription_search and normalized in row.transcription_search:
            matched_fields.append("transcription")
        if normalized in row.sender_search:
            matched_fields.append("sender")
        item["matched_fields"] = matched_fields
    next_cursor = None
    if has_more and page:
        last = page[-1]
        next_cursor = encode_cursor(
            "conversation-message-search",
            scope=scope,
            timestamp_ms=last.timestamp_ms,
            id=last.id,
        )
    return {
        "items": items,
        "page": {"limit": limit, "total": total, "next_cursor": next_cursor},
        "query": q,
    }
