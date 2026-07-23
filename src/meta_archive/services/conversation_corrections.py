from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, case, func, or_, select, update
from sqlalchemy.orm import Session

from meta_archive.models import (
    AttachmentOccurrence,
    ConflictAggregateRun,
    ConflictCalibration,
    ConflictHourlyRollup,
    ConflictWindowScore,
    Conversation,
    ConversationCorrection,
    ConversationCorrectionAudit,
    ConversationCorrectionSegment,
    ConversationSegment,
    ConversationSegmentOverride,
    Message,
    SearchWindow,
    SegmentParticipant,
    SemanticIndexConfig,
)
from meta_archive.paths import ArchiveLayout
from meta_archive.services.deletion import deletion_lock, immediate_session
from meta_archive.services.library import (
    LibraryBuildError,
    build_filesystem_library,
    library_build_lock,
)
from meta_archive.services.search_index import semantic_index_lock
from meta_archive.services.source_import_lock import (
    SourceImportPublicationLockError,
    assert_source_import_publication_allowed,
    source_import_publication_lock,
)
from meta_archive.services.text import normalize_text


class ConversationCorrectionError(RuntimeError):
    pass


class StaleConversationCorrectionError(ConversationCorrectionError):
    pass


class ConversationCorrectionConflictError(ConversationCorrectionError):
    pass


@dataclass(frozen=True, slots=True)
class CorrectionResult:
    correction_id: str
    action: str
    status: str
    identity_revision: str
    redirect_conversation_id: str
    effect: dict[str, Any]


def _rowcount(result: Any) -> int:
    value = getattr(result, "rowcount", 0)
    return int(value) if isinstance(value, int) and value > 0 else 0


def _chunks(values: Sequence[str], size: int = 500) -> Iterable[Sequence[str]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def conversation_identity_revision(session: Session) -> str:
    """Hash the current direct-history identity graph, excluding derived counters."""

    conversations = [
        [
            row.id,
            row.conversation_key,
            row.title_raw,
            row.title_display,
            row.title_search,
            row.conversation_type,
            row.deleted_at.isoformat() if row.deleted_at else None,
        ]
        for row in session.scalars(
            select(Conversation)
            .where(Conversation.conversation_type == "direct")
            .order_by(Conversation.id)
        )
    ]
    segments = [
        [row.id, row.conversation_id, row.thread_path, row.sequence]
        for row in session.scalars(
            select(ConversationSegment)
            .join(Conversation, Conversation.id == ConversationSegment.conversation_id)
            .where(Conversation.conversation_type == "direct")
            .order_by(ConversationSegment.id)
        )
    ]
    overrides = [
        [row.segment_id, row.base_conversation_id, row.conversation_id, row.correction_id]
        for row in session.scalars(
            select(ConversationSegmentOverride).order_by(ConversationSegmentOverride.segment_id)
        )
    ]
    encoded = json.dumps(
        {"conversations": conversations, "segments": segments, "overrides": overrides},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _conversation_snapshot(row: Conversation) -> dict[str, Any]:
    return {
        "id": row.id,
        "conversation_key": row.conversation_key,
        "title_raw": row.title_raw,
        "title_display": row.title_display,
        "title_search": row.title_search,
        "conversation_type": row.conversation_type,
        "primary_category": row.primary_category,
        "participant_count": row.participant_count,
        "first_activity_ms": row.first_activity_ms,
        "last_activity_ms": row.last_activity_ms,
        "message_count": row.message_count,
        "sent_count": row.sent_count,
        "received_count": row.received_count,
        "attachment_count": row.attachment_count,
        "unique_attachment_count": row.unique_attachment_count,
        "attachment_bytes": row.attachment_bytes,
        "active_days": row.active_days,
        "completeness": row.completeness,
        "deleted_at": row.deleted_at.isoformat() if row.deleted_at else None,
    }


def _restore_conversation_snapshot(session: Session, snapshot: dict[str, Any]) -> Conversation:
    identifier = str(snapshot["id"])
    existing = session.get(Conversation, identifier)
    if existing is not None:
        return existing
    deleted_at_value = snapshot.get("deleted_at")
    deleted_at = (
        datetime.fromisoformat(str(deleted_at_value)) if deleted_at_value is not None else None
    )
    row = Conversation(
        id=identifier,
        conversation_key=str(snapshot["conversation_key"]),
        title_raw=str(snapshot["title_raw"]),
        title_display=str(snapshot["title_display"]),
        title_search=str(snapshot["title_search"]),
        conversation_type=str(snapshot["conversation_type"]),
        primary_category=str(snapshot["primary_category"]),
        participant_count=int(snapshot.get("participant_count") or 0),
        first_activity_ms=snapshot.get("first_activity_ms"),
        last_activity_ms=snapshot.get("last_activity_ms"),
        message_count=int(snapshot.get("message_count") or 0),
        sent_count=int(snapshot.get("sent_count") or 0),
        received_count=int(snapshot.get("received_count") or 0),
        attachment_count=int(snapshot.get("attachment_count") or 0),
        unique_attachment_count=int(snapshot.get("unique_attachment_count") or 0),
        attachment_bytes=int(snapshot.get("attachment_bytes") or 0),
        active_days=int(snapshot.get("active_days") or 0),
        completeness=str(snapshot.get("completeness") or "unknown"),
        deleted_at=deleted_at,
    )
    session.add(row)
    session.flush()
    return row


def _container_source_file_ids(segments: Iterable[ConversationSegment]) -> set[int]:
    identifiers: set[int] = set()
    for segment in segments:
        containers = segment.metadata_json.get("containers")
        if not isinstance(containers, dict):
            continue
        for value in containers:
            try:
                identifiers.add(int(value))
            except (TypeError, ValueError):
                continue
    return identifiers


def _apply_segment_assignments(
    session: Session,
    assignments: dict[str, str],
) -> dict[str, Any]:
    effect: dict[str, Any] = {
        "segments": 0,
        "messages": 0,
        "attachments": 0,
        "search_windows": 0,
    }
    grouped: defaultdict[str, list[str]] = defaultdict(list)
    for segment_id, conversation_id in assignments.items():
        grouped[conversation_id].append(segment_id)

    for conversation_id, segment_ids in grouped.items():
        segments = list(
            session.scalars(
                select(ConversationSegment).where(ConversationSegment.id.in_(segment_ids))
            )
        )
        if len(segments) != len(segment_ids):
            raise ConversationCorrectionConflictError("One or more source segments no longer exist")
        source_file_ids = _container_source_file_ids(segments)
        message_ids = select(Message.id).where(Message.segment_id.in_(segment_ids))
        occurrence_filter: Any = AttachmentOccurrence.message_id.in_(message_ids)
        if source_file_ids:
            occurrence_filter = or_(
                occurrence_filter,
                AttachmentOccurrence.source_file_id.in_(source_file_ids),
            )
        occurrence_ids = select(AttachmentOccurrence.id).where(occurrence_filter)

        effect["search_windows"] += _rowcount(
            session.execute(
                update(SearchWindow)
                .where(
                    or_(
                        SearchWindow.segment_id.in_(segment_ids),
                        SearchWindow.attachment_occurrence_id.in_(occurrence_ids),
                    )
                )
                .values(conversation_id=conversation_id)
            )
        )
        effect["attachments"] += _rowcount(
            session.execute(
                update(AttachmentOccurrence)
                .where(occurrence_filter)
                .values(conversation_id=conversation_id)
            )
        )
        effect["messages"] += _rowcount(
            session.execute(
                update(Message)
                .where(Message.segment_id.in_(segment_ids))
                .values(conversation_id=conversation_id)
            )
        )
        for segment in segments:
            if segment.conversation_id != conversation_id:
                segment.conversation_id = conversation_id
                effect["segments"] += 1
    session.flush()
    return effect


def _reconcile_conversations(session: Session, conversation_ids: Iterable[str]) -> None:
    session.flush()
    for conversation_id in sorted(set(conversation_ids)):
        conversation = session.get(Conversation, conversation_id)
        if conversation is None:
            continue
        segments = list(
            session.scalars(
                select(ConversationSegment)
                .where(ConversationSegment.conversation_id == conversation_id)
                .order_by(
                    ConversationSegment.first_activity_ms.is_(None),
                    ConversationSegment.first_activity_ms,
                    ConversationSegment.thread_path,
                )
            )
        )
        if not segments:
            dependent_count = int(
                session.scalar(
                    select(func.count(Message.pk)).where(Message.conversation_id == conversation_id)
                )
                or 0
            ) + int(
                session.scalar(
                    select(func.count(AttachmentOccurrence.id)).where(
                        AttachmentOccurrence.conversation_id == conversation_id
                    )
                )
                or 0
            )
            if dependent_count:
                raise ConversationCorrectionConflictError(
                    "An empty conversation still has dependent records"
                )
            session.delete(conversation)
            continue

        participant_counts = {
            segment_id: count
            for segment_id, count in session.execute(
                select(
                    SegmentParticipant.segment_id,
                    func.count(SegmentParticipant.id),
                )
                .where(SegmentParticipant.segment_id.in_([segment.id for segment in segments]))
                .group_by(SegmentParticipant.segment_id)
            )
        }
        for sequence, segment in enumerate(segments):
            segment.sequence = sequence
            segment.first_activity_ms, segment.last_activity_ms = session.execute(
                select(func.min(Message.timestamp_ms), func.max(Message.timestamp_ms)).where(
                    Message.segment_id == segment.id,
                    Message.deleted_at.is_(None),
                )
            ).one()

        message_count, sent_count, first_ms, last_ms, active_days = session.execute(
            select(
                func.count(Message.pk),
                func.coalesce(func.sum(case((Message.sender_is_owner.is_(True), 1), else_=0)), 0),
                func.min(Message.timestamp_ms),
                func.max(Message.timestamp_ms),
                func.count(func.distinct(func.date(Message.timestamp_ms / 1000.0, "unixepoch"))),
            ).where(
                Message.conversation_id == conversation_id,
                Message.deleted_at.is_(None),
            )
        ).one()
        attachment_count, unique_count, attachment_bytes = session.execute(
            select(
                func.count(AttachmentOccurrence.id),
                func.count(func.distinct(AttachmentOccurrence.blob_sha256)),
                func.coalesce(func.sum(AttachmentOccurrence.size_bytes), 0),
            ).where(
                AttachmentOccurrence.conversation_id == conversation_id,
                AttachmentOccurrence.deleted_at.is_(None),
            )
        ).one()
        conversation.message_count = int(message_count or 0)
        conversation.sent_count = int(sent_count or 0)
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
        conversation.primary_category = segments[0].source_category
    session.flush()


def _invalidate_conversation_derived_state(session: Session, correction_id: str) -> dict[str, int]:
    configs = list(
        session.scalars(
            select(SemanticIndexConfig).where(SemanticIndexConfig.status.in_(("ready", "building")))
        )
    )
    for config in configs:
        audit = dict(config.audit_json)
        audit["stale_reason"] = "conversation_identity_correction"
        audit["stale_correction_id"] = correction_id
        config.audit_json = audit
        config.active = False
        config.status = "stale"
    calibrations = list(
        session.scalars(
            select(ConflictCalibration).where(ConflictCalibration.aggregation_enabled.is_(True))
        )
    )
    for calibration in calibrations:
        provenance = dict(calibration.provenance_json)
        provenance["stale_reason"] = "conversation_identity_correction"
        provenance["stale_correction_id"] = correction_id
        calibration.provenance_json = provenance
        calibration.aggregation_enabled = False
        calibration.status = "stale"
    aggregate_runs = list(
        session.scalars(
            select(ConflictAggregateRun).where(
                ConflictAggregateRun.status.in_(("building", "ready"))
            )
        )
    )
    aggregate_run_ids = [row.id for row in aggregate_runs]
    for run in aggregate_runs:
        provenance = dict(run.provenance_json)
        provenance["stale_reason"] = "conversation_identity_correction"
        provenance["stale_correction_id"] = correction_id
        run.provenance_json = provenance
        run.active = False
        run.status = "stale"
        run.error_text = "Conversation identity changed; rebuild conflict aggregates"
    if aggregate_run_ids:
        session.execute(
            update(ConflictWindowScore)
            .where(ConflictWindowScore.run_id.in_(aggregate_run_ids))
            .values(active=False)
        )
        session.execute(
            update(ConflictHourlyRollup)
            .where(ConflictHourlyRollup.run_id.in_(aggregate_run_ids))
            .values(active=False)
        )
    return {
        "semantic_configs_invalidated": len(configs),
        "conflict_calibrations_invalidated": len(calibrations),
        "conflict_aggregate_runs_invalidated": len(aggregate_runs),
    }


def _record_audit(
    session: Session,
    correction_id: str,
    *,
    action: str,
    detail: dict[str, Any],
) -> str:
    audit_id = str(uuid.uuid4())
    session.add(
        ConversationCorrectionAudit(
            id=audit_id,
            correction_id=correction_id,
            action=action,
            detail_json=detail,
        )
    )
    return audit_id


def _validate_direct_conversations(
    session: Session, conversation_ids: Sequence[str]
) -> dict[str, Conversation]:
    rows = {
        row.id: row
        for row in session.scalars(
            select(Conversation).where(Conversation.id.in_(conversation_ids))
        )
    }
    if len(rows) != len(conversation_ids):
        raise ConversationCorrectionError("One or more conversations do not exist")
    if any(row.deleted_at is not None for row in rows.values()):
        raise ConversationCorrectionConflictError(
            "Quarantined conversations cannot be identity-corrected"
        )
    if any(row.conversation_type != "direct" for row in rows.values()):
        raise ConversationCorrectionError(
            "Manual identity corrections are limited to provisional direct histories"
        )
    return rows


def _stage_library(layout: ArchiveLayout, correction_id: str) -> tuple[bool, Path]:
    backup = layout.root / f".conversation-correction-library-{correction_id}"
    if backup.exists() or backup.is_symlink():
        raise ConversationCorrectionConflictError(
            "A stale conversation-correction library backup requires manual review"
        )
    if layout.library.is_symlink():
        raise ConversationCorrectionConflictError(
            "Refusing to rebuild a symlinked filesystem library"
        )
    present = layout.library.is_dir()
    if present:
        os.replace(layout.library, backup)
    return present, backup


def _remove_tree(path: Path) -> None:
    if path.is_symlink():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _mutate_with_derived_views(
    layout: ArchiveLayout,
    engine: Engine,
    correction_id: str,
    mutation: Callable[[Session, bool], CorrectionResult],
) -> CorrectionResult:
    try:
        with (
            deletion_lock(layout),
            source_import_publication_lock(layout, blocking=False) as source_lock,
            library_build_lock(layout, blocking=False),
            semantic_index_lock(layout),
        ):
            assert_source_import_publication_allowed(
                layout,
                engine,
                lease=source_lock,
                publisher="Conversation correction publication",
            )
            library_present, backup = _stage_library(layout, correction_id)
            try:
                with immediate_session(engine) as session:
                    result = mutation(session, library_present)
                    if library_present:
                        build_filesystem_library(
                            layout,
                            engine,
                            catalog_session=session,
                            lock_held=True,
                            source_lock=source_lock,
                        )
            except BaseException:
                if library_present:
                    _remove_tree(layout.library)
                    if backup.exists():
                        os.replace(backup, layout.library)
                raise
            if backup.exists():
                _remove_tree(backup)
            return result
    except (LibraryBuildError, SourceImportPublicationLockError) as exc:
        raise ConversationCorrectionConflictError(
            "The source catalog or filesystem library is currently unavailable"
        ) from exc


def merge_direct_histories(
    layout: ArchiveLayout,
    engine: Engine,
    *,
    conversation_ids: Sequence[str],
    target_conversation_id: str,
    expected_revision: str,
    reason: str | None = None,
) -> CorrectionResult:
    identifiers = tuple(dict.fromkeys(conversation_ids))
    if len(identifiers) < 2 or target_conversation_id not in identifiers:
        raise ConversationCorrectionError(
            "Merge requires at least two conversations and a target from that selection"
        )
    correction_id = str(uuid.uuid4())

    def mutate(session: Session, library_present: bool) -> CorrectionResult:
        current_revision = conversation_identity_revision(session)
        if current_revision != expected_revision:
            raise StaleConversationCorrectionError(
                "Conversation identities changed; refresh before applying this merge"
            )
        conversations = _validate_direct_conversations(session, identifiers)
        source_ids = [value for value in identifiers if value != target_conversation_id]
        segments = list(
            session.scalars(
                select(ConversationSegment)
                .where(ConversationSegment.conversation_id.in_(source_ids))
                .order_by(ConversationSegment.id)
            )
        )
        if not segments:
            raise ConversationCorrectionConflictError(
                "The selected source histories no longer contain segments"
            )
        before_snapshots = [_conversation_snapshot(conversations[value]) for value in identifiers]
        correction = ConversationCorrection(
            id=correction_id,
            action="merge",
            status="active",
            reason=reason,
            request_json={
                "conversation_ids": list(identifiers),
                "target_conversation_id": target_conversation_id,
            },
            before_json={"conversations": before_snapshots},
            after_json={},
            catalog_revision=current_revision,
            result_revision="pending",
            effect_json={},
        )
        session.add(correction)
        session.flush()
        for segment in segments:
            override = session.get(ConversationSegmentOverride, segment.id)
            session.add(
                ConversationCorrectionSegment(
                    correction_id=correction_id,
                    segment_id=segment.id,
                    thread_path=segment.thread_path,
                    before_conversation_id=segment.conversation_id,
                    after_conversation_id=target_conversation_id,
                    previous_correction_id=override.correction_id if override else None,
                )
            )
            if override is None:
                session.add(
                    ConversationSegmentOverride(
                        segment_id=segment.id,
                        base_conversation_id=segment.conversation_id,
                        conversation_id=target_conversation_id,
                        correction_id=correction_id,
                    )
                )
            else:
                override.conversation_id = target_conversation_id
                override.correction_id = correction_id
                override.updated_at = datetime.now(UTC)

        effect = _apply_segment_assignments(
            session, {segment.id: target_conversation_id for segment in segments}
        )
        _reconcile_conversations(session, identifiers)
        effect.update(_invalidate_conversation_derived_state(session, correction_id))
        effect["library_rebuilt"] = library_present
        session.flush()
        target = session.get(Conversation, target_conversation_id)
        if target is None:
            raise ConversationCorrectionConflictError("The merge target disappeared")
        correction.after_json = {"conversations": [_conversation_snapshot(target)]}
        result_revision = conversation_identity_revision(session)
        correction.result_revision = result_revision
        audit_id = _record_audit(
            session,
            correction_id,
            action="apply",
            detail={
                "action": "merge",
                "before_revision": current_revision,
                "after_revision": result_revision,
                "effect": effect,
            },
        )
        effect["audit_id"] = audit_id
        correction.effect_json = effect
        return CorrectionResult(
            correction_id=correction_id,
            action="merge",
            status="active",
            identity_revision=result_revision,
            redirect_conversation_id=target_conversation_id,
            effect=effect,
        )

    return _mutate_with_derived_views(layout, engine, correction_id, mutate)


def split_direct_history(
    layout: ArchiveLayout,
    engine: Engine,
    *,
    conversation_id: str,
    segment_ids: Sequence[str],
    title: str,
    expected_revision: str,
    reason: str | None = None,
) -> CorrectionResult:
    selected_ids = tuple(dict.fromkeys(segment_ids))
    if not selected_ids:
        raise ConversationCorrectionError("Split requires at least one source segment")
    normalized_title = normalize_text(title)
    if not normalized_title.display:
        raise ConversationCorrectionError("Split title cannot be empty")
    correction_id = str(uuid.uuid4())
    new_conversation_id = hashlib.sha256(
        f"manual-conversation\x00{correction_id}".encode()
    ).hexdigest()

    def mutate(session: Session, library_present: bool) -> CorrectionResult:
        current_revision = conversation_identity_revision(session)
        if current_revision != expected_revision:
            raise StaleConversationCorrectionError(
                "Conversation identities changed; refresh before applying this split"
            )
        source = _validate_direct_conversations(session, [conversation_id])[conversation_id]
        all_segments = list(
            session.scalars(
                select(ConversationSegment)
                .where(ConversationSegment.conversation_id == conversation_id)
                .order_by(ConversationSegment.id)
            )
        )
        by_id = {segment.id: segment for segment in all_segments}
        if any(segment_id not in by_id for segment_id in selected_ids):
            raise ConversationCorrectionError(
                "Every split segment must currently belong to the source conversation"
            )
        if len(selected_ids) >= len(all_segments):
            raise ConversationCorrectionError(
                "A split must leave at least one source segment in the original history"
            )
        first = min(
            (by_id[value] for value in selected_ids),
            key=lambda segment: (
                segment.first_activity_ms is None,
                segment.first_activity_ms or 0,
                segment.thread_path,
            ),
        )
        destination = Conversation(
            id=new_conversation_id,
            conversation_key=f"manual:{correction_id}",
            title_raw=normalized_title.raw,
            title_display=normalized_title.display,
            title_search=normalized_title.search,
            conversation_type="direct",
            primary_category=first.source_category,
            completeness="unknown",
        )
        session.add(destination)
        correction = ConversationCorrection(
            id=correction_id,
            action="split",
            status="active",
            reason=reason,
            request_json={
                "conversation_id": conversation_id,
                "segment_ids": list(selected_ids),
                "title": normalized_title.display,
            },
            before_json={"conversations": [_conversation_snapshot(source)]},
            after_json={},
            catalog_revision=current_revision,
            result_revision="pending",
            effect_json={},
        )
        session.add(correction)
        session.flush()
        selected_segments = [by_id[value] for value in selected_ids]
        for segment in selected_segments:
            override = session.get(ConversationSegmentOverride, segment.id)
            session.add(
                ConversationCorrectionSegment(
                    correction_id=correction_id,
                    segment_id=segment.id,
                    thread_path=segment.thread_path,
                    before_conversation_id=conversation_id,
                    after_conversation_id=new_conversation_id,
                    previous_correction_id=override.correction_id if override else None,
                )
            )
            if override is None:
                session.add(
                    ConversationSegmentOverride(
                        segment_id=segment.id,
                        base_conversation_id=conversation_id,
                        conversation_id=new_conversation_id,
                        correction_id=correction_id,
                    )
                )
            else:
                override.conversation_id = new_conversation_id
                override.correction_id = correction_id
                override.updated_at = datetime.now(UTC)

        effect = _apply_segment_assignments(
            session, {segment.id: new_conversation_id for segment in selected_segments}
        )
        _reconcile_conversations(session, [conversation_id, new_conversation_id])
        effect.update(_invalidate_conversation_derived_state(session, correction_id))
        effect["library_rebuilt"] = library_present
        session.flush()
        source_after = session.get(Conversation, conversation_id)
        destination_after = session.get(Conversation, new_conversation_id)
        if source_after is None or destination_after is None:
            raise ConversationCorrectionConflictError("The split result is incomplete")
        correction.after_json = {
            "conversations": [
                _conversation_snapshot(source_after),
                _conversation_snapshot(destination_after),
            ]
        }
        result_revision = conversation_identity_revision(session)
        correction.result_revision = result_revision
        audit_id = _record_audit(
            session,
            correction_id,
            action="apply",
            detail={
                "action": "split",
                "before_revision": current_revision,
                "after_revision": result_revision,
                "effect": effect,
            },
        )
        effect["audit_id"] = audit_id
        correction.effect_json = effect
        return CorrectionResult(
            correction_id=correction_id,
            action="split",
            status="active",
            identity_revision=result_revision,
            redirect_conversation_id=new_conversation_id,
            effect=effect,
        )

    return _mutate_with_derived_views(layout, engine, correction_id, mutate)


def revert_conversation_correction(
    layout: ArchiveLayout,
    engine: Engine,
    correction_id: str,
    *,
    expected_revision: str,
) -> CorrectionResult:
    def mutate(session: Session, library_present: bool) -> CorrectionResult:
        current_revision = conversation_identity_revision(session)
        if current_revision != expected_revision:
            raise StaleConversationCorrectionError(
                "Conversation identities changed; refresh before reverting this correction"
            )
        correction = session.get(ConversationCorrection, correction_id)
        if correction is None:
            raise ConversationCorrectionError("Conversation correction not found")
        if correction.status != "active":
            raise ConversationCorrectionConflictError("Conversation correction is not active")
        segment_rows = list(
            session.scalars(
                select(ConversationCorrectionSegment)
                .where(ConversationCorrectionSegment.correction_id == correction_id)
                .order_by(ConversationCorrectionSegment.segment_id)
            )
        )
        if not segment_rows:
            raise ConversationCorrectionConflictError("Conversation correction has no audit rows")
        for row in segment_rows:
            segment = session.get(ConversationSegment, row.segment_id)
            override = session.get(ConversationSegmentOverride, row.segment_id)
            if (
                segment is None
                or override is None
                or override.correction_id != correction_id
                or segment.conversation_id != row.after_conversation_id
            ):
                raise ConversationCorrectionConflictError(
                    "A later correction or source change must be reverted first"
                )

        snapshots_value = correction.before_json.get("conversations")
        if not isinstance(snapshots_value, list):
            raise ConversationCorrectionConflictError(
                "Conversation correction lacks restorable snapshots"
            )
        for snapshot in snapshots_value:
            if not isinstance(snapshot, dict):
                raise ConversationCorrectionConflictError(
                    "Conversation correction contains an invalid snapshot"
                )
            _restore_conversation_snapshot(session, snapshot)

        assignments = {row.segment_id: row.before_conversation_id for row in segment_rows}
        effect = _apply_segment_assignments(session, assignments)
        for row in segment_rows:
            override = session.get(ConversationSegmentOverride, row.segment_id)
            assert override is not None
            if row.previous_correction_id is None:
                session.delete(override)
            else:
                override.conversation_id = row.before_conversation_id
                override.correction_id = row.previous_correction_id
                override.updated_at = datetime.now(UTC)
        affected_ids = {
            value
            for row in segment_rows
            for value in (row.before_conversation_id, row.after_conversation_id)
        }
        _reconcile_conversations(session, affected_ids)
        effect.update(_invalidate_conversation_derived_state(session, correction_id))
        effect["library_rebuilt"] = library_present
        correction.status = "reverted"
        correction.reverted_at = datetime.now(UTC)
        session.flush()
        result_revision = conversation_identity_revision(session)
        correction.revert_revision = result_revision
        stored_effect = dict(correction.effect_json)
        audit_id = _record_audit(
            session,
            correction_id,
            action="revert",
            detail={
                "before_revision": current_revision,
                "after_revision": result_revision,
                "effect": effect,
            },
        )
        effect["audit_id"] = audit_id
        stored_effect["revert"] = effect
        correction.effect_json = stored_effect
        redirect = segment_rows[0].before_conversation_id
        return CorrectionResult(
            correction_id=correction_id,
            action=correction.action,
            status="reverted",
            identity_revision=result_revision,
            redirect_conversation_id=redirect,
            effect=effect,
        )

    return _mutate_with_derived_views(layout, engine, correction_id, mutate)


def correction_is_revertible(session: Session, correction_id: str) -> bool:
    rows = list(
        session.scalars(
            select(ConversationCorrectionSegment).where(
                ConversationCorrectionSegment.correction_id == correction_id
            )
        )
    )
    return bool(rows) and all(
        (override := session.get(ConversationSegmentOverride, row.segment_id)) is not None
        and override.correction_id == correction_id
        for row in rows
    )


def correction_payload(session: Session, correction: ConversationCorrection) -> dict[str, Any]:
    segments = list(
        session.scalars(
            select(ConversationCorrectionSegment)
            .where(ConversationCorrectionSegment.correction_id == correction.id)
            .order_by(ConversationCorrectionSegment.segment_id)
        )
    )
    audits = list(
        session.scalars(
            select(ConversationCorrectionAudit)
            .where(ConversationCorrectionAudit.correction_id == correction.id)
            .order_by(ConversationCorrectionAudit.created_at, ConversationCorrectionAudit.id)
        )
    )
    return {
        "id": correction.id,
        "action": correction.action,
        "status": correction.status,
        "reason": correction.reason,
        "request": correction.request_json,
        "before": correction.before_json,
        "after": correction.after_json,
        "catalog_revision": correction.catalog_revision,
        "result_revision": correction.result_revision,
        "revert_revision": correction.revert_revision,
        "effect": correction.effect_json,
        "created_at": correction.created_at,
        "reverted_at": correction.reverted_at,
        "can_revert": correction.status == "active"
        and correction_is_revertible(session, correction.id),
        "segments": [
            {
                "id": row.segment_id,
                "thread_path": row.thread_path,
                "before_conversation_id": row.before_conversation_id,
                "after_conversation_id": row.after_conversation_id,
                "previous_correction_id": row.previous_correction_id,
            }
            for row in segments
        ],
        "audits": [
            {
                "id": row.id,
                "action": row.action,
                "detail": row.detail_json,
                "created_at": row.created_at,
            }
            for row in audits
        ],
    }


def list_conversation_corrections(
    session: Session,
    *,
    conversation_id: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    statement = select(ConversationCorrection).order_by(
        ConversationCorrection.created_at.desc(), ConversationCorrection.id.desc()
    )
    count_statement = select(func.count(ConversationCorrection.id))
    if conversation_id:
        correction_ids = select(ConversationCorrectionSegment.correction_id).where(
            or_(
                ConversationCorrectionSegment.before_conversation_id == conversation_id,
                ConversationCorrectionSegment.after_conversation_id == conversation_id,
            )
        )
        statement = statement.where(ConversationCorrection.id.in_(correction_ids))
        count_statement = count_statement.where(ConversationCorrection.id.in_(correction_ids))
    total = int(session.scalar(count_statement) or 0)
    rows = list(session.scalars(statement.offset(offset).limit(limit)))
    return [correction_payload(session, row) for row in rows], total
