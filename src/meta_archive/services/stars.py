from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, MutableMapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, cast

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from meta_archive.models import (
    ActivityRecord,
    AttachmentOccurrence,
    Blob,
    Conversation,
    ConversationCorrection,
    ConversationCorrectionSegment,
    ConversationSegment,
    ConversationSegmentOverride,
    Message,
    Participant,
    SegmentParticipant,
    Star,
)
from meta_archive.services.identifiers import stable_id
from meta_archive.services.jumps import activity_jump
from meta_archive.services.media_derivatives import derivatives_by_blob

StarTargetType = Literal[
    "conversation",
    "message",
    "activity",
    "attachment",
    "blob",
    "messenger_contact",
]
StarTargetState = Literal["active", "latent", "missing"]
StarTargetKey = tuple[StarTargetType, str]

STAR_TARGET_TYPES: frozenset[str] = frozenset(
    {
        "conversation",
        "message",
        "activity",
        "attachment",
        "blob",
        "messenger_contact",
    }
)
STAR_CATEGORY_TARGET_TYPES: dict[str, frozenset[str]] = {
    "all": STAR_TARGET_TYPES,
    "conversations": frozenset({"conversation"}),
    "messages": frozenset({"message"}),
    "media": frozenset({"attachment", "blob"}),
    "activity": frozenset({"activity"}),
    "people": frozenset({"messenger_contact"}),
}


@dataclass(frozen=True, slots=True)
class ResolvedStarTarget:
    state: StarTargetState
    payload: dict[str, Any] | None = None


@dataclass(slots=True)
class _ContactData:
    target_id: str
    name: str
    name_search: str
    conversations: dict[str, Conversation]


def _utc_datetime(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def serialize_star(row: Star) -> dict[str, str]:
    return {"id": row.id, "starred_at": _utc_datetime(row.starred_at).isoformat()}


def star_payload(row: Star) -> dict[str, str]:
    """Backward-compatible descriptive alias for ``serialize_star``."""

    return serialize_star(row)


def make_star_target(target_type: StarTargetType, target_id: str) -> dict[str, str]:
    return {"target_type": target_type, "target_id": target_id}


def star_target_payload(target_type: StarTargetType, target_id: str) -> dict[str, str]:
    """Backward-compatible descriptive alias for ``make_star_target``."""

    return make_star_target(target_type, target_id)


def _normalized_keys(targets: Iterable[StarTargetKey]) -> list[StarTargetKey]:
    return list(dict.fromkeys((target_type, target_id) for target_type, target_id in targets))


def star_rows_by_target(
    session: Session, targets: Iterable[StarTargetKey]
) -> dict[StarTargetKey, Star]:
    keys = _normalized_keys(targets)
    if not keys:
        return {}
    grouped: dict[StarTargetType, list[str]] = defaultdict(list)
    for target_type, target_id in keys:
        grouped[target_type].append(target_id)
    loaded: dict[StarTargetKey, Star] = {}
    for target_type, target_ids in grouped.items():
        for offset in range(0, len(target_ids), 500):
            rows = session.scalars(
                select(Star).where(
                    Star.target_type == target_type,
                    Star.target_id.in_(target_ids[offset : offset + 500]),
                )
            )
            loaded.update(
                {cast(StarTargetKey, (row.target_type, row.target_id)): row for row in rows}
            )
    return loaded


def annotate_star_targets[PayloadT: MutableMapping[str, Any]](
    session: Session,
    bindings: Iterable[tuple[PayloadT, StarTargetType, str]],
) -> list[PayloadT]:
    """Attach canonical star fields to payloads with one bounded star query."""

    rows = list(bindings)
    stars = star_rows_by_target(
        session, ((target_type, target_id) for _item, target_type, target_id in rows)
    )
    for item, target_type, target_id in rows:
        key = (target_type, target_id)
        item["star_target"] = make_star_target(target_type, target_id)
        item["star"] = serialize_star(stars[key]) if key in stars else None
    return [item for item, _target_type, _target_id in rows]


def attach_star_states[PayloadT: MutableMapping[str, Any]](
    session: Session,
    items: Sequence[PayloadT],
    target_for_item: Callable[[PayloadT], StarTargetKey],
) -> list[PayloadT]:
    """Annotate a domain page without teaching it how stars are stored."""

    return annotate_star_targets(
        session,
        ((item, *target_for_item(item)) for item in items),
    )


def delete_stars_for_targets(session: Session, targets: Iterable[StarTargetKey]) -> int:
    """Permanently remove exact stars as part of an enclosing purge transaction."""

    keys = _normalized_keys(targets)
    if not keys:
        return 0
    grouped: dict[StarTargetType, list[str]] = defaultdict(list)
    for target_type, target_id in keys:
        grouped[target_type].append(target_id)
    deleted_count = 0
    for target_type, target_ids in grouped.items():
        for offset in range(0, len(target_ids), 500):
            result = session.execute(
                delete(Star).where(
                    Star.target_type == target_type,
                    Star.target_id.in_(target_ids[offset : offset + 500]),
                )
            )
            deleted_count += int(cast(Any, result).rowcount or 0)
    return deleted_count


def _message_jump(conversation_id: str, message_id: str) -> dict[str, str]:
    fragment = f"message-{message_id}"
    return {
        "href": f"/conversations/{conversation_id}?around={message_id}#{fragment}",
        "conversation_id": conversation_id,
        "message_id": message_id,
        "fragment": fragment,
    }


def _attachment_jump(
    row: AttachmentOccurrence,
    *,
    conversation: Conversation | None,
    activity: ActivityRecord | None,
) -> dict[str, str] | None:
    if conversation is not None:
        if row.message_id is None:
            return {
                "href": f"/conversations/{conversation.id}",
                "conversation_id": conversation.id,
            }
        fragment = f"message-{row.message_id}"
        return {
            "href": (
                f"/conversations/{conversation.id}?around={row.message_id}"
                f"&media={row.id}#{fragment}"
            ),
            "conversation_id": conversation.id,
            "message_id": row.message_id,
            "fragment": fragment,
        }
    return activity_jump(activity.id) if activity is not None else None


def _media_class(row: AttachmentOccurrence) -> str:
    kind = row.kind.casefold()
    media_type = (row.media_type or "").casefold()
    if kind in {"gifs", "gif"}:
        return "gif"
    if kind == "sticker":
        return "sticker"
    if kind in {"audio_files", "audio"} or media_type.startswith("audio/"):
        return "audio"
    if kind in {"videos", "video"} or media_type.startswith("video/"):
        return "video"
    if kind in {"photos", "photo", "thread_image"} or media_type.startswith("image/"):
        return "photo"
    if kind in {"files", "file", "support_file", "document"}:
        return "document"
    return "other"


def _attachment_is_active(
    row: AttachmentOccurrence,
    *,
    conversation: Conversation | None,
    message: Message | None,
    activity: ActivityRecord | None,
) -> bool:
    if row.deleted_at is not None:
        return False
    if row.message_id is not None and (message is None or message.deleted_at is not None):
        return False
    if row.activity_record_id is not None and (activity is None or activity.deleted_at is not None):
        return False
    return not (
        (row.conversation_id is not None or message is not None)
        and (conversation is None or conversation.deleted_at is not None)
    )


def _attachment_media_payload(
    row: AttachmentOccurrence,
    *,
    conversation: Conversation | None,
    activity: ActivityRecord | None,
    occurrence_count: int = 1,
) -> dict[str, Any]:
    return {
        "id": row.id,
        "view_key": f"occurrence:{row.id}",
        "type": _media_class(row),
        "kind": row.kind,
        "filename": row.filename,
        "media_type": row.media_type,
        "size_bytes": row.size_bytes,
        "timestamp_ms": row.timestamp_ms,
        "unknown_date": row.timestamp_ms is None,
        "availability": row.availability,
        "conversation": (
            {"id": conversation.id, "title": conversation.title_display}
            if conversation is not None
            else None
        ),
        "activity": (
            {
                "id": activity.id,
                "title": activity.title_display,
                "category": activity.category,
                "subtype": activity.subtype,
            }
            if activity is not None
            else None
        ),
        "message_id": row.message_id,
        "blob_sha256": row.blob_sha256,
        "blob_href": f"/api/v1/blobs/{row.blob_sha256}" if row.blob_sha256 else None,
        "is_external_reference": row.availability == "external",
        "occurrence_count": occurrence_count,
        "jump": _attachment_jump(row, conversation=conversation, activity=activity),
    }


def _common_payload(
    *,
    kind: Literal["conversation", "message", "media", "activity", "person"],
    title: str,
    subtitle: str | None,
    excerpt: str | None,
    timestamp_ms: int | None,
    jump: dict[str, str] | None,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "title": title,
        "subtitle": subtitle,
        "excerpt": excerpt,
        "timestamp_ms": timestamp_ms,
        "jump": jump,
    }


def _conversation_payload(row: Conversation) -> dict[str, Any]:
    payload = _common_payload(
        kind="conversation",
        title=row.title_display,
        subtitle=row.conversation_type.replace("_", " ").title(),
        excerpt=(
            f"{row.message_count:,} messages · {row.attachment_count:,} attachment"
            f"{'' if row.attachment_count == 1 else 's'}"
        ),
        timestamp_ms=row.last_activity_ms,
        jump={"href": f"/conversations/{row.id}", "conversation_id": row.id},
    )
    payload["conversation"] = {
        "id": row.id,
        "type": row.conversation_type,
        "message_count": row.message_count,
        "attachment_count": row.attachment_count,
        "first_activity_ms": row.first_activity_ms,
        "last_activity_ms": row.last_activity_ms,
    }
    return payload


def _message_payload(row: Message, conversation: Conversation) -> dict[str, Any]:
    excerpt = row.content_display or row.transcription_display
    if excerpt is None:
        excerpt = row.message_kind.replace("_", " ").title()
    payload = _common_payload(
        kind="message",
        title=row.sender_display,
        subtitle=conversation.title_display,
        excerpt=excerpt,
        timestamp_ms=row.timestamp_ms,
        jump=_message_jump(conversation.id, row.id),
    )
    payload["sender"] = row.sender_display
    payload["sender_is_owner"] = row.sender_is_owner
    payload["conversation"] = {
        "id": conversation.id,
        "title": conversation.title_display,
        "type": conversation.conversation_type,
    }
    return payload


def _activity_payload(row: ActivityRecord) -> dict[str, Any]:
    title = row.title_display or row.content_display or row.subtype.replace("_", " ").title()
    payload = _common_payload(
        kind="activity",
        title=title,
        subtitle=row.category.replace("_", " ").title(),
        excerpt=row.content_display if row.content_display != title else None,
        timestamp_ms=row.timestamp_ms,
        jump=activity_jump(row.id),
    )
    payload["activity"] = {
        "id": row.id,
        "category": row.category,
        "subtype": row.subtype,
        "actor": row.actor_display,
        "availability": row.availability,
    }
    return payload


def _attachment_payload(
    row: AttachmentOccurrence,
    *,
    conversation: Conversation | None,
    activity: ActivityRecord | None,
) -> dict[str, Any]:
    media = _attachment_media_payload(row, conversation=conversation, activity=activity)
    subtitle = (
        conversation.title_display
        if conversation is not None
        else activity.title_display
        if activity is not None and activity.title_display
        else "Unlinked archive media"
    )
    payload = _common_payload(
        kind="media",
        title=row.filename or _media_class(row).replace("_", " ").title(),
        subtitle=subtitle,
        excerpt=None,
        timestamp_ms=row.timestamp_ms,
        jump=cast(dict[str, str] | None, media["jump"]),
    )
    payload["media_scope"] = "occurrence"
    payload["media"] = media
    return payload


def _blob_payload(
    blob: Blob,
    row: AttachmentOccurrence,
    *,
    conversation: Conversation | None,
    activity: ActivityRecord | None,
    occurrence_count: int,
) -> dict[str, Any]:
    media = _attachment_media_payload(
        row,
        conversation=conversation,
        activity=activity,
        occurrence_count=occurrence_count,
    )
    media.update(
        {
            "id": blob.sha256,
            "view_key": f"blob:{blob.sha256}",
            "size_bytes": blob.size_bytes,
            "media_type": blob.media_type or row.media_type,
            "blob_sha256": blob.sha256,
        }
    )
    subtitle = (
        conversation.title_display
        if conversation is not None
        else activity.title_display
        if activity is not None and activity.title_display
        else "Archive media"
    )
    payload = _common_payload(
        kind="media",
        title=row.filename or f"Unique {_media_class(row)}",
        subtitle=subtitle,
        excerpt=f"{occurrence_count:,} occurrence{'' if occurrence_count == 1 else 's'}",
        timestamp_ms=row.timestamp_ms,
        jump=cast(dict[str, str] | None, media["jump"]),
    )
    payload["media_scope"] = "unique"
    payload["media"] = media
    return payload


def _recoverable_conversation_ids(session: Session, requested: set[str]) -> set[str]:
    if not requested:
        return set()
    recoverable: set[str] = set()
    corrections = session.scalars(
        select(ConversationCorrection).where(ConversationCorrection.status == "active")
    )
    for correction in corrections:
        if correction.action != "merge":
            continue
        snapshots = correction.before_json.get("conversations")
        if not isinstance(snapshots, list) or not snapshots:
            continue
        if not all(_conversation_snapshot_is_restorable(snapshot) for snapshot in snapshots):
            continue
        snapshot_ids = {
            cast(dict[str, Any], snapshot)["id"]
            for snapshot in snapshots
            if cast(dict[str, Any], snapshot)["id"] in requested
        }
        if not snapshot_ids:
            continue
        all_segment_rows = list(
            session.scalars(
                select(ConversationCorrectionSegment)
                .where(
                    ConversationCorrectionSegment.correction_id == correction.id,
                )
                .order_by(ConversationCorrectionSegment.segment_id)
            )
        )
        if not all_segment_rows:
            continue
        globally_revertible = True
        rows_by_before: dict[str, list[ConversationCorrectionSegment]] = defaultdict(list)
        for segment_row in all_segment_rows:
            rows_by_before[segment_row.before_conversation_id].append(segment_row)
            segment = session.get(ConversationSegment, segment_row.segment_id)
            override = session.get(ConversationSegmentOverride, segment_row.segment_id)
            if (
                segment is None
                or override is None
                or override.correction_id != correction.id
                or segment.conversation_id != segment_row.after_conversation_id
            ):
                globally_revertible = False
                break
        if not globally_revertible:
            continue
        for target_id in snapshot_ids:
            target_rows = rows_by_before[target_id]
            if not target_rows:
                continue
            recoverable.add(target_id)
    return recoverable


def _conversation_snapshot_is_restorable(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    required_strings = (
        "id",
        "conversation_key",
        "title_raw",
        "title_display",
        "title_search",
        "conversation_type",
        "primary_category",
    )
    if any(not isinstance(value.get(key), str) for key in required_strings):
        return False
    deleted_at = value.get("deleted_at")
    if deleted_at is not None:
        if not isinstance(deleted_at, str):
            return False
        try:
            datetime.fromisoformat(deleted_at)
        except ValueError:
            return False
    integer_fields = (
        "participant_count",
        "message_count",
        "sent_count",
        "received_count",
        "attachment_count",
        "unique_attachment_count",
        "attachment_bytes",
        "active_days",
    )
    try:
        for key in integer_fields:
            int(value.get(key) or 0)
    except (TypeError, ValueError):
        return False
    return True


def _load_contact_data(session: Session) -> dict[str, _ContactData]:
    contacts: dict[str, _ContactData] = {}
    rows = session.execute(
        select(
            Participant.name_search,
            func.min(Participant.name_display),
            Conversation,
        )
        .join(SegmentParticipant, SegmentParticipant.participant_id == Participant.id)
        .join(ConversationSegment, ConversationSegment.id == SegmentParticipant.segment_id)
        .join(Conversation, Conversation.id == ConversationSegment.conversation_id)
        .where(
            Participant.is_owner.is_(False),
            Conversation.conversation_type.in_(("direct", "group")),
        )
        .group_by(Participant.name_search, Conversation.id)
        .order_by(Participant.name_search, Conversation.id)
    )
    for name_search, name_display, conversation in rows:
        target_id = stable_id("messenger-contact", name_search)
        contact = contacts.get(target_id)
        if contact is None:
            contact = _ContactData(
                target_id=target_id,
                name=name_display,
                name_search=name_search,
                conversations={},
            )
            contacts[target_id] = contact
        elif name_display < contact.name:
            contact.name = name_display
        contact.conversations[conversation.id] = conversation
    return contacts


def messenger_contact_ids_for_conversations(
    session: Session,
    conversation_ids: Iterable[str],
) -> set[str]:
    identifiers = tuple(dict.fromkeys(conversation_ids))
    if not identifiers:
        return set()
    names = session.scalars(
        select(Participant.name_search)
        .join(SegmentParticipant, SegmentParticipant.participant_id == Participant.id)
        .join(ConversationSegment, ConversationSegment.id == SegmentParticipant.segment_id)
        .join(Conversation, Conversation.id == ConversationSegment.conversation_id)
        .where(
            Participant.is_owner.is_(False),
            ConversationSegment.conversation_id.in_(identifiers),
            Conversation.conversation_type.in_(("direct", "group")),
        )
        .distinct()
    )
    return {stable_id("messenger-contact", name) for name in names}


def messenger_contact_ids_eliminated_by_conversation_removal(
    session: Session,
    conversation_ids: Iterable[str],
) -> set[str]:
    """Return contact identities whose final retained conversation is being removed."""

    selected = tuple(dict.fromkeys(conversation_ids))
    if not selected:
        return set()
    candidates = messenger_contact_ids_for_conversations(session, selected)
    if not candidates:
        return set()
    remaining_names = session.scalars(
        select(Participant.name_search)
        .join(SegmentParticipant, SegmentParticipant.participant_id == Participant.id)
        .join(ConversationSegment, ConversationSegment.id == SegmentParticipant.segment_id)
        .join(Conversation, Conversation.id == ConversationSegment.conversation_id)
        .where(
            Participant.is_owner.is_(False),
            ConversationSegment.conversation_id.not_in(selected),
            Conversation.conversation_type.in_(("direct", "group")),
        )
        .distinct()
    )
    remaining = {stable_id("messenger-contact", name) for name in remaining_names}
    return candidates - remaining


def _contact_payload(
    session: Session,
    contact: _ContactData,
    *,
    active_conversations: list[Conversation],
) -> dict[str, Any]:
    direct = [row for row in active_conversations if row.conversation_type == "direct"]
    conversation_ids = [row.id for row in direct]
    media_counts = {"photos": 0, "videos": 0, "audio": 0}
    if conversation_ids:
        attachments = session.scalars(
            select(AttachmentOccurrence).where(
                AttachmentOccurrence.conversation_id.in_(conversation_ids),
                AttachmentOccurrence.message_id.is_not(None),
                AttachmentOccurrence.deleted_at.is_(None),
            )
        )
        for attachment in attachments:
            media_class = _media_class(attachment)
            if media_class == "photo":
                media_counts["photos"] += 1
            elif media_class == "video":
                media_counts["videos"] += 1
            elif media_class == "audio":
                media_counts["audio"] += 1
    counts = {
        "messages": sum(row.message_count for row in direct),
        **media_counts,
    }
    links = [
        {
            "id": row.id,
            "title": row.title_display,
            "type": row.conversation_type,
            "message_count": row.message_count,
            "first_activity_ms": row.first_activity_ms,
            "last_activity_ms": row.last_activity_ms,
            "href": f"/conversations/{row.id}",
        }
        for row in sorted(
            active_conversations,
            key=lambda item: (-(item.last_activity_ms or -1), item.id),
        )
    ]
    last_activity_ms = max(
        (row.last_activity_ms for row in active_conversations if row.last_activity_ms is not None),
        default=None,
    )
    jump = {
        "href": (f"/people?view=contacts&around={contact.target_id}#contact-{contact.target_id}"),
        "contact_id": contact.target_id,
        "fragment": f"contact-{contact.target_id}",
    }
    payload = _common_payload(
        kind="person",
        title=contact.name,
        subtitle="Messenger label",
        excerpt=(
            f"{counts['messages']:,} direct messages · "
            f"{len(active_conversations):,} conversation"
            f"{'' if len(active_conversations) == 1 else 's'}"
        ),
        timestamp_ms=last_activity_ms,
        jump=jump,
    )
    payload["contact"] = {
        "id": contact.target_id,
        "name": contact.name,
        "contact_basis": "messenger_participant_label",
        "counts": counts,
        "conversation_count": len(active_conversations),
        "conversations": links,
    }
    payload["provisional"] = True
    payload["counts"] = counts
    payload["conversations"] = links
    return payload


def resolve_star_targets(
    session: Session,
    targets: Iterable[StarTargetKey],
    *,
    include_payload: bool = True,
) -> dict[StarTargetKey, ResolvedStarTarget]:
    """Resolve exact target identities against the current logical catalog."""

    keys = _normalized_keys(targets)
    if len(keys) > 400:
        resolutions: dict[StarTargetKey, ResolvedStarTarget] = {}
        for offset in range(0, len(keys), 400):
            resolutions.update(
                resolve_star_targets(
                    session,
                    keys[offset : offset + 400],
                    include_payload=include_payload,
                )
            )
        return resolutions
    resolutions = {key: ResolvedStarTarget(state="missing") for key in keys}
    grouped: dict[str, set[str]] = defaultdict(set)
    for target_type, target_id in keys:
        grouped[target_type].add(target_id)

    conversation_ids = grouped["conversation"]
    if conversation_ids:
        conversation_rows = {
            conversation_row.id: conversation_row
            for conversation_row in session.scalars(
                select(Conversation).where(Conversation.id.in_(conversation_ids))
            )
        }
        for target_id, conversation_row in conversation_rows.items():
            state: StarTargetState = "active" if conversation_row.deleted_at is None else "latent"
            resolutions[("conversation", target_id)] = ResolvedStarTarget(
                state=state,
                payload=(
                    _conversation_payload(conversation_row)
                    if include_payload and state == "active"
                    else None
                ),
            )
        missing_ids = conversation_ids - conversation_rows.keys()
        for target_id in _recoverable_conversation_ids(session, missing_ids):
            resolutions[("conversation", target_id)] = ResolvedStarTarget(state="latent")

    message_ids = grouped["message"]
    if message_ids:
        for message_row, conversation in session.execute(
            select(Message, Conversation)
            .outerjoin(Conversation, Conversation.id == Message.conversation_id)
            .where(Message.id.in_(message_ids))
        ):
            active = (
                message_row.deleted_at is None
                and conversation is not None
                and conversation.deleted_at is None
            )
            resolutions[("message", message_row.id)] = ResolvedStarTarget(
                state="active" if active else "latent",
                payload=(
                    _message_payload(message_row, conversation)
                    if include_payload and active and conversation is not None
                    else None
                ),
            )

    activity_ids = grouped["activity"]
    if activity_ids:
        for activity_row in session.scalars(
            select(ActivityRecord).where(ActivityRecord.id.in_(activity_ids))
        ):
            state = "active" if activity_row.deleted_at is None else "latent"
            resolutions[("activity", activity_row.id)] = ResolvedStarTarget(
                state=state,
                payload=(
                    _activity_payload(activity_row)
                    if include_payload and state == "active"
                    else None
                ),
            )

    attachment_ids = grouped["attachment"]
    if attachment_ids:
        statement = (
            select(AttachmentOccurrence, Conversation, Message, ActivityRecord)
            .outerjoin(Message, Message.id == AttachmentOccurrence.message_id)
            .outerjoin(
                Conversation,
                Conversation.id
                == func.coalesce(AttachmentOccurrence.conversation_id, Message.conversation_id),
            )
            .outerjoin(ActivityRecord, ActivityRecord.id == AttachmentOccurrence.activity_record_id)
            .where(AttachmentOccurrence.id.in_(attachment_ids))
        )
        for attachment_row, conversation, message, activity in session.execute(statement):
            active = _attachment_is_active(
                attachment_row,
                conversation=conversation,
                message=message,
                activity=activity,
            )
            resolutions[("attachment", attachment_row.id)] = ResolvedStarTarget(
                state="active" if active else "latent",
                payload=(
                    _attachment_payload(
                        attachment_row,
                        conversation=conversation,
                        activity=activity,
                    )
                    if include_payload and active
                    else None
                ),
            )

    blob_ids = grouped["blob"]
    if blob_ids:
        blobs = {
            blob_row.sha256: blob_row
            for blob_row in session.scalars(select(Blob).where(Blob.sha256.in_(blob_ids)))
        }
        occurrences: dict[
            str,
            list[
                tuple[
                    AttachmentOccurrence,
                    Conversation | None,
                    Message | None,
                    ActivityRecord | None,
                ]
            ],
        ] = defaultdict(list)
        occurrence_statement = (
            select(AttachmentOccurrence, Conversation, Message, ActivityRecord)
            .outerjoin(Message, Message.id == AttachmentOccurrence.message_id)
            .outerjoin(
                Conversation,
                Conversation.id
                == func.coalesce(AttachmentOccurrence.conversation_id, Message.conversation_id),
            )
            .outerjoin(ActivityRecord, ActivityRecord.id == AttachmentOccurrence.activity_record_id)
            .where(AttachmentOccurrence.blob_sha256.in_(blob_ids))
        )
        for occurrence_row, conversation, message, activity in session.execute(
            occurrence_statement
        ):
            if occurrence_row.blob_sha256 is not None and _attachment_is_active(
                occurrence_row,
                conversation=conversation,
                message=message,
                activity=activity,
            ):
                occurrences[occurrence_row.blob_sha256].append(
                    (occurrence_row, conversation, message, activity)
                )
        for target_id, blob in blobs.items():
            active_rows = occurrences[target_id]
            if not active_rows:
                resolutions[("blob", target_id)] = ResolvedStarTarget(state="latent")
                continue
            representative = max(
                active_rows,
                key=lambda value: (
                    value[0].timestamp_ms is not None,
                    value[0].timestamp_ms or 0,
                    value[0].id,
                ),
            )
            occurrence_row, conversation, _message, activity = representative
            resolutions[("blob", target_id)] = ResolvedStarTarget(
                state="active",
                payload=(
                    _blob_payload(
                        blob,
                        occurrence_row,
                        conversation=conversation,
                        activity=activity,
                        occurrence_count=len(active_rows),
                    )
                    if include_payload
                    else None
                ),
            )

    media_payloads = [
        resolution.payload
        for (target_type, _target_id), resolution in resolutions.items()
        if target_type in {"attachment", "blob"}
        and resolution.state == "active"
        and resolution.payload is not None
    ]
    if include_payload and media_payloads:
        blob_sha256s = [
            media["blob_sha256"]
            for payload in media_payloads
            if isinstance((media := payload.get("media")), dict)
            and isinstance(media.get("blob_sha256"), str)
        ]
        derivative_map = derivatives_by_blob(session, blob_sha256s)
        for payload in media_payloads:
            media = payload.get("media")
            if not isinstance(media, dict):
                continue
            blob_sha256 = media.get("blob_sha256")
            media["derivatives"] = (
                derivative_map.get(blob_sha256, []) if isinstance(blob_sha256, str) else []
            )

    contact_ids = grouped["messenger_contact"]
    if contact_ids:
        contacts = _load_contact_data(session)
        for target_id in contact_ids:
            contact = contacts.get(target_id)
            if contact is None:
                continue
            active_conversations = [
                row for row in contact.conversations.values() if row.deleted_at is None
            ]
            active = bool(active_conversations)
            resolutions[("messenger_contact", target_id)] = ResolvedStarTarget(
                state="active" if active else "latent",
                payload=(
                    _contact_payload(
                        session,
                        contact,
                        active_conversations=active_conversations,
                    )
                    if include_payload and active
                    else None
                ),
            )

    return resolutions
