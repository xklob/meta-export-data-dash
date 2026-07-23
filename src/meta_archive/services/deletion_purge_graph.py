from __future__ import annotations

import os
import re
import stat
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from sqlalchemy import delete, exists, or_, select
from sqlalchemy.orm import Session

from meta_archive.models import (
    AttachmentOccurrence,
    AttachmentTextExtraction,
    AudioTranscription,
    AudioTranscriptSegment,
    ConflictCalibration,
    ConflictLabel,
    ConversationCorrection,
    ConversationCorrectionAudit,
    ConversationCorrectionSegment,
    ConversationSegment,
    DeletionItem,
    DeletionOperation,
    Job,
    MediaDerivative,
    Participant,
    RagAudit,
    RelevanceFeedback,
    SearchWindow,
    SearchWindowMessage,
    SegmentParticipant,
)
from meta_archive.paths import ArchiveLayout

ID_CHUNK = 500
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
HEX_PREFIX_PATTERN = re.compile(r"[0-9a-f]{2}")
DERIVATIVE_ROOT_CONTROL_FILES = frozenset({".build.lock", "manifest.json"})


def _chunks(values: Sequence[str], size: int = ID_CHUNK) -> Iterator[Sequence[str]]:
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def _rowcount(result: Any) -> int:
    return int(getattr(result, "rowcount", 0) or 0)


def _scalar_ids_for_chunks(
    session: Session,
    values: Iterable[str],
    statement_factory: Any,
) -> set[str]:
    result: set[str] = set()
    for chunk in _chunks(sorted(set(values))):
        result.update(str(value) for value in session.scalars(statement_factory(chunk)))
    return result


@dataclass(frozen=True, slots=True)
class BlobOwnership:
    """Attachment ownership relevant to physical blob erasure.

    A logically quarantined occurrence remains a recoverable reference. It therefore protects its
    blob exactly like a live occurrence until that occurrence is part of the same hard purge.
    """

    selected_blob_ids: frozenset[str]
    live_retained_blob_ids: frozenset[str]
    recoverable_retained_blob_ids: frozenset[str]

    @property
    def shared_blob_ids(self) -> frozenset[str]:
        return frozenset(self.live_retained_blob_ids | self.recoverable_retained_blob_ids)

    @property
    def physically_exclusive_blob_ids(self) -> frozenset[str]:
        return frozenset(self.selected_blob_ids - self.shared_blob_ids)


def classify_blob_ownership(
    session: Session,
    selected_attachment_ids: Iterable[str],
) -> BlobOwnership:
    """Classify selected blobs without treating another quarantine as disposable."""

    selected_ids = set(selected_attachment_ids)
    selected_blob_ids = _scalar_ids_for_chunks(
        session,
        selected_ids,
        lambda chunk: select(AttachmentOccurrence.blob_sha256).where(
            AttachmentOccurrence.id.in_(chunk),
            AttachmentOccurrence.blob_sha256.is_not(None),
        ),
    )
    live_retained: set[str] = set()
    recoverable_retained: set[str] = set()
    for chunk in _chunks(sorted(selected_blob_ids)):
        rows = session.execute(
            select(
                AttachmentOccurrence.id,
                AttachmentOccurrence.blob_sha256,
                AttachmentOccurrence.deleted_at,
            ).where(AttachmentOccurrence.blob_sha256.in_(chunk))
        )
        for occurrence_id, blob_sha256, deleted_at in rows:
            if str(occurrence_id) in selected_ids or blob_sha256 is None:
                continue
            target = live_retained if deleted_at is None else recoverable_retained
            target.add(str(blob_sha256))
    return BlobOwnership(
        selected_blob_ids=frozenset(selected_blob_ids),
        live_retained_blob_ids=frozenset(live_retained),
        recoverable_retained_blob_ids=frozenset(recoverable_retained),
    )


@dataclass(frozen=True, slots=True)
class DerivedPurgeGraph:
    conversation_segment_ids: frozenset[str]
    conversation_participant_candidate_ids: frozenset[str]
    logical_search_window_ids: frozenset[str]
    physical_search_window_ids: frozenset[str]
    audio_transcription_ids: frozenset[str]
    audio_segment_ids: frozenset[str]
    attachment_extraction_ids: frozenset[str]
    media_derivative_ids: frozenset[str]

    @property
    def search_window_ids(self) -> frozenset[str]:
        return frozenset(self.logical_search_window_ids | self.physical_search_window_ids)


def build_derived_purge_graph(
    session: Session,
    *,
    message_ids: Iterable[str] = (),
    activity_ids: Iterable[str] = (),
    conversation_ids: Iterable[str] = (),
    attachment_ids: Iterable[str] = (),
    physical_blob_ids: Iterable[str] = (),
) -> DerivedPurgeGraph:
    """Resolve every derived row that must be handled before FK cascades run.

    Logical windows copy selected normalized text. Physical windows copy transcript or extraction
    text that can be erased only when no live or recoverably quarantined occurrence retains the
    underlying blob.
    """

    selected_messages = set(message_ids)
    selected_activities = set(activity_ids)
    selected_conversations = set(conversation_ids)
    selected_attachments = set(attachment_ids)
    physical_blobs = set(physical_blob_ids)

    segment_ids = _scalar_ids_for_chunks(
        session,
        selected_conversations,
        lambda chunk: select(ConversationSegment.id).where(
            ConversationSegment.conversation_id.in_(chunk)
        ),
    )
    participant_candidates = _scalar_ids_for_chunks(
        session,
        segment_ids,
        lambda chunk: select(SegmentParticipant.participant_id).where(
            SegmentParticipant.segment_id.in_(chunk)
        ),
    )

    logical_windows: set[str] = set()
    logical_windows.update(
        _scalar_ids_for_chunks(
            session,
            selected_messages,
            lambda chunk: select(SearchWindowMessage.window_id).where(
                SearchWindowMessage.message_id.in_(chunk)
            ),
        )
    )
    for values, statement_factory in (
        (
            selected_messages,
            lambda chunk: select(SearchWindow.id).where(
                or_(
                    SearchWindow.first_message_id.in_(chunk),
                    SearchWindow.last_message_id.in_(chunk),
                )
            ),
        ),
        (
            selected_activities,
            lambda chunk: select(SearchWindow.id).where(SearchWindow.activity_id.in_(chunk)),
        ),
        (
            selected_attachments,
            lambda chunk: select(SearchWindow.id).where(
                SearchWindow.attachment_occurrence_id.in_(chunk)
            ),
        ),
        (
            selected_conversations,
            lambda chunk: select(SearchWindow.id).where(SearchWindow.conversation_id.in_(chunk)),
        ),
        (
            segment_ids,
            lambda chunk: select(SearchWindow.id).where(SearchWindow.segment_id.in_(chunk)),
        ),
    ):
        logical_windows.update(_scalar_ids_for_chunks(session, values, statement_factory))

    transcription_ids = _scalar_ids_for_chunks(
        session,
        physical_blobs,
        lambda chunk: select(AudioTranscription.id).where(
            AudioTranscription.blob_sha256.in_(chunk)
        ),
    )
    audio_segment_ids = _scalar_ids_for_chunks(
        session,
        transcription_ids,
        lambda chunk: select(AudioTranscriptSegment.id).where(
            AudioTranscriptSegment.transcription_id.in_(chunk)
        ),
    )
    extraction_ids = _scalar_ids_for_chunks(
        session,
        physical_blobs,
        lambda chunk: select(AttachmentTextExtraction.id).where(
            AttachmentTextExtraction.blob_sha256.in_(chunk)
        ),
    )
    derivative_ids = _scalar_ids_for_chunks(
        session,
        physical_blobs,
        lambda chunk: select(MediaDerivative.id).where(MediaDerivative.blob_sha256.in_(chunk)),
    )
    physical_windows: set[str] = set()
    physical_windows.update(
        _scalar_ids_for_chunks(
            session,
            audio_segment_ids,
            lambda chunk: select(SearchWindow.id).where(SearchWindow.audio_segment_id.in_(chunk)),
        )
    )
    physical_windows.update(
        _scalar_ids_for_chunks(
            session,
            extraction_ids,
            lambda chunk: select(SearchWindow.id).where(
                SearchWindow.attachment_extraction_id.in_(chunk)
            ),
        )
    )
    return DerivedPurgeGraph(
        conversation_segment_ids=frozenset(segment_ids),
        conversation_participant_candidate_ids=frozenset(participant_candidates),
        logical_search_window_ids=frozenset(logical_windows),
        physical_search_window_ids=frozenset(physical_windows),
        audio_transcription_ids=frozenset(transcription_ids),
        audio_segment_ids=frozenset(audio_segment_ids),
        attachment_extraction_ids=frozenset(extraction_ids),
        media_derivative_ids=frozenset(derivative_ids),
    )


@dataclass(frozen=True, slots=True)
class DerivativeTreeInventory:
    selected_catalogued_paths: frozenset[Path]
    selected_orphan_paths: frozenset[Path]
    missing_selected_catalogued_paths: frozenset[Path]
    global_metadata_paths: frozenset[Path]
    blockers: tuple[str, ...]

    @property
    def selected_paths(self) -> frozenset[Path]:
        return frozenset(self.selected_catalogued_paths | self.selected_orphan_paths)

    @property
    def executable(self) -> bool:
        return not self.blockers


def _safe_catalog_path(layout: ArchiveLayout, relative_path: str) -> Path | None:
    pure = PurePosixPath(relative_path)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
        or "\x00" in relative_path
    ):
        return None
    root = Path(os.path.abspath(layout.root))
    candidate = root.joinpath(*pure.parts)
    if candidate == root or root not in candidate.parents:
        return None
    return candidate


def _canonical_derivative_blob(relative_to_thumbnails: PurePosixPath) -> str | None:
    parts = relative_to_thumbnails.parts
    if len(parts) != 4:
        return None
    _pipeline, prefix, sha256, filename = parts
    if (
        not filename
        or HEX_PREFIX_PATTERN.fullmatch(prefix) is None
        or SHA256_PATTERN.fullmatch(sha256) is None
        or sha256[:2] != prefix
    ):
        return None
    return sha256


def _scan_derivative_tree(
    layout: ArchiveLayout,
) -> tuple[dict[str, Path], set[Path], list[str]]:
    root = Path(os.path.abspath(layout.thumbnails))
    if not root.exists() and not root.is_symlink():
        return {}, set(), []
    blockers: list[str] = []
    try:
        root_stat = root.lstat()
    except OSError:
        return {}, set(), ["derivative_root_unreadable"]
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        return {}, set(), ["derivative_root_unsafe"]

    files: dict[str, Path] = {}
    global_metadata: set[Path] = set()
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            entries = sorted(directory.iterdir(), key=lambda path: path.name)
        except OSError:
            blockers.append("derivative_directory_unreadable")
            continue
        for candidate in entries:
            relative = candidate.relative_to(Path(os.path.abspath(layout.root))).as_posix()
            try:
                candidate_stat = candidate.lstat()
            except OSError:
                blockers.append(f"derivative_entry_unreadable:{relative}")
                continue
            if stat.S_ISLNK(candidate_stat.st_mode):
                blockers.append(f"derivative_symlink:{relative}")
            elif stat.S_ISDIR(candidate_stat.st_mode):
                stack.append(candidate)
            elif stat.S_ISREG(candidate_stat.st_mode):
                files[relative] = candidate
                if candidate.parent == root and candidate.name in DERIVATIVE_ROOT_CONTROL_FILES:
                    global_metadata.add(candidate)
            else:
                blockers.append(f"derivative_special_file:{relative}")
    return files, global_metadata, blockers


def inventory_derivative_tree(
    layout: ArchiveLayout,
    session: Session,
    selected_blob_ids: Iterable[str],
) -> DerivativeTreeInventory:
    """Inventory the complete derivative tree without following links.

    A regular file is attributable when its exact path is catalogued or when it is under the
    deterministic ``pipeline/prefix/sha256/file`` layout. Uncatalogued files for selected blobs
    are returned as purge targets. Unattributed bytes and every link/special file block execution.
    """

    selected_blobs = set(selected_blob_ids)
    invalid_blob_ids = sorted(
        value for value in selected_blobs if not SHA256_PATTERN.fullmatch(value)
    )
    files, global_metadata, blockers = _scan_derivative_tree(layout)
    if invalid_blob_ids:
        blockers.append("selected_blob_identifier_invalid")

    catalog_by_path: dict[str, str] = {}
    actual_paths = sorted(files)
    for chunk in _chunks(actual_paths):
        for relative_path, blob_sha256 in session.execute(
            select(MediaDerivative.relative_path, MediaDerivative.blob_sha256).where(
                MediaDerivative.relative_path.in_(chunk)
            )
        ):
            catalog_by_path[str(relative_path)] = str(blob_sha256)

    selected_catalog_rows: dict[str, str] = {}
    for chunk in _chunks(sorted(selected_blobs)):
        for relative_path, blob_sha256 in session.execute(
            select(MediaDerivative.relative_path, MediaDerivative.blob_sha256).where(
                MediaDerivative.blob_sha256.in_(chunk)
            )
        ):
            selected_catalog_rows[str(relative_path)] = str(blob_sha256)

    selected_catalogued: set[Path] = set()
    selected_orphans: set[Path] = set()
    thumbnails = Path(os.path.abspath(layout.thumbnails))
    root_control_relatives = {
        path.relative_to(Path(os.path.abspath(layout.root))).as_posix() for path in global_metadata
    }
    for relative, path in files.items():
        if relative in root_control_relatives:
            continue
        catalog_blob = catalog_by_path.get(relative)
        if catalog_blob is not None:
            if catalog_blob in selected_blobs:
                selected_catalogued.add(path)
            continue
        try:
            derivative_relative = PurePosixPath(path.relative_to(thumbnails).as_posix())
        except ValueError:
            blockers.append(f"derivative_file_outside_root:{relative}")
            continue
        attributed_blob = _canonical_derivative_blob(derivative_relative)
        if attributed_blob is None:
            blockers.append(f"unattributed_derivative_file:{relative}")
        elif attributed_blob in selected_blobs:
            selected_orphans.add(path)

    missing_catalogued: set[Path] = set()
    for relative, expected_blob in selected_catalog_rows.items():
        candidate = _safe_catalog_path(layout, relative)
        if candidate is None:
            blockers.append("catalogued_derivative_path_unsafe")
            continue
        try:
            candidate.relative_to(thumbnails)
        except ValueError:
            blockers.append(f"catalogued_derivative_outside_root:{relative}")
            continue
        actual_catalog_blob = catalog_by_path.get(relative)
        if actual_catalog_blob is not None and actual_catalog_blob != expected_blob:
            blockers.append(f"catalogued_derivative_owner_mismatch:{relative}")
            continue
        if relative not in files:
            missing_catalogued.add(candidate)

    return DerivativeTreeInventory(
        selected_catalogued_paths=frozenset(selected_catalogued),
        selected_orphan_paths=frozenset(selected_orphans),
        missing_selected_catalogued_paths=frozenset(missing_catalogued),
        global_metadata_paths=frozenset(global_metadata),
        blockers=tuple(sorted(set(blockers))),
    )


@dataclass(frozen=True, slots=True)
class FeedbackCalibrationPurgeResult:
    relevance_feedback_deleted: int
    rag_audits_deleted: int
    selected_conflict_labels: int
    conflict_calibrations_scrubbed: int


def _contains_window_reference(value: Any, references: set[str]) -> bool:
    if isinstance(value, str):
        return value in references
    if isinstance(value, dict):
        return any(
            _contains_window_reference(key, references)
            or _contains_window_reference(item, references)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_window_reference(item, references) for item in value)
    return False


def purge_window_feedback_and_calibrations(
    session: Session,
    window_ids: Iterable[str],
    *,
    invalidated_at: datetime | None = None,
) -> FeedbackCalibrationPurgeResult:
    """Delete soft-linked feedback and scrub calibration products before window deletion."""

    selected = set(window_ids)
    if not selected:
        return FeedbackCalibrationPurgeResult(0, 0, 0, 0)
    references = selected | {f"window:{window_id}" for window_id in selected}
    feedback_deleted = 0
    label_count = 0
    # Each window expands to both a legacy raw ID and the canonical ``window:<id>`` result ID.
    # Keep the resulting bind count below SQLite's classic 999-variable ceiling.
    for chunk in _chunks(sorted(selected), 250):
        result_ids = [value for window_id in chunk for value in (window_id, f"window:{window_id}")]
        feedback_deleted += _rowcount(
            session.execute(
                delete(RelevanceFeedback).where(RelevanceFeedback.result_id.in_(result_ids))
            )
        )
        label_count += len(
            list(
                session.scalars(select(ConflictLabel.id).where(ConflictLabel.window_id.in_(chunk)))
            )
        )

    rag_audits_deleted = 0
    for audit in list(session.scalars(select(RagAudit).order_by(RagAudit.id))):
        if any(
            _contains_window_reference(payload, references)
            for payload in (
                audit.retrieved_ids_json,
                audit.citation_ids_json,
                audit.excerpt_hashes_json,
            )
        ):
            session.delete(audit)
            rag_audits_deleted += 1

    calibrations = list(
        session.scalars(select(ConflictCalibration).order_by(ConflictCalibration.id))
    )
    if label_count:
        selected_calibrations = calibrations
    else:
        selected_calibrations = [
            row
            for row in calibrations
            if any(
                _contains_window_reference(payload, references)
                for payload in (row.metrics_json, row.provenance_json, row.warnings_json)
            )
        ]
    timestamp = (invalidated_at or datetime.now(UTC)).astimezone(UTC).isoformat()
    for calibration in selected_calibrations:
        calibration.status = "stale"
        calibration.sample_count = 0
        calibration.direct_conflict_precision = None
        calibration.macro_f1 = None
        calibration.aggregation_enabled = False
        calibration.metrics_json = {
            "scrubbed": True,
            "reason": "hard_purge_removed_labeled_window",
            "invalidated_at": timestamp,
        }
        calibration.provenance_json = {
            "scrubbed": True,
            "reason": "hard_purge_removed_labeled_window",
            "invalidated_at": timestamp,
        }
        calibration.warnings_json = {
            "hard_purge_scrubbed": True,
            "recalibration_required": True,
        }
        calibration.completed_at = None
    return FeedbackCalibrationPurgeResult(
        relevance_feedback_deleted=feedback_deleted,
        rag_audits_deleted=rag_audits_deleted,
        selected_conflict_labels=label_count,
        conflict_calibrations_scrubbed=len(selected_calibrations),
    )


def gc_conversation_candidate_participants(
    session: Session,
    candidate_participant_ids: Iterable[str],
) -> int:
    """Delete unreferenced participants only when a purged conversation nominated them."""

    deleted = 0
    for chunk in _chunks(sorted(set(candidate_participant_ids))):
        deleted += _rowcount(
            session.execute(
                delete(Participant).where(
                    Participant.id.in_(chunk),
                    ~exists(
                        select(SegmentParticipant.id).where(
                            SegmentParticipant.participant_id == Participant.id
                        )
                    ),
                )
            )
        )
    return deleted


@dataclass(frozen=True, slots=True)
class DeletionOverlapInventory:
    blocking_operation_ids: frozenset[str]
    invalidatable_operation_ids: frozenset[str]
    invalidatable_manifest_paths: frozenset[str]
    invalidatable_item_count: int
    unexpected_operation_ids: frozenset[str]

    @property
    def blockers(self) -> tuple[str, ...]:
        blockers: list[str] = []
        if self.blocking_operation_ids:
            blockers.append("selected_rows_belong_to_another_committed_deletion")
        if self.unexpected_operation_ids:
            blockers.append("selected_rows_belong_to_an_unknown_deletion_state")
        return tuple(blockers)


def inventory_deletion_overlaps(
    session: Session,
    *,
    current_operation_id: str,
    selected_entity_ids: Mapping[str, Iterable[str]],
) -> DeletionOverlapInventory:
    """Find deletion manifests that would otherwise retain hard-purged entity identities."""

    rows_by_operation: dict[str, tuple[str, str | None]] = {}
    item_counts_by_operation: dict[str, int] = {}
    for entity_type, identifiers in selected_entity_ids.items():
        for chunk in _chunks(sorted(set(identifiers))):
            rows = session.execute(
                select(
                    DeletionItem.operation_id,
                    DeletionOperation.status,
                    DeletionOperation.manifest_relative_path,
                )
                .join(
                    DeletionOperation,
                    DeletionOperation.id == DeletionItem.operation_id,
                )
                .where(
                    DeletionItem.operation_id != current_operation_id,
                    DeletionItem.entity_type == entity_type,
                    DeletionItem.entity_id.in_(chunk),
                )
            )
            for operation_id, status, manifest_path in rows:
                normalized_operation_id = str(operation_id)
                rows_by_operation[normalized_operation_id] = (
                    str(status),
                    str(manifest_path) if manifest_path is not None else None,
                )
                item_counts_by_operation[normalized_operation_id] = (
                    item_counts_by_operation.get(normalized_operation_id, 0) + 1
                )

    blocking: set[str] = set()
    invalidatable: set[str] = set()
    unexpected: set[str] = set()
    manifests: set[str] = set()
    for operation_id, (status, manifest_path) in rows_by_operation.items():
        if status in {"committed", "cleanup_pending"}:
            blocking.add(operation_id)
        elif status in {"previewed", "restored"}:
            invalidatable.add(operation_id)
            if manifest_path is not None:
                manifests.add(manifest_path)
        else:
            unexpected.add(operation_id)
    return DeletionOverlapInventory(
        blocking_operation_ids=frozenset(blocking),
        invalidatable_operation_ids=frozenset(invalidatable),
        invalidatable_manifest_paths=frozenset(manifests),
        invalidatable_item_count=sum(
            item_counts_by_operation[operation_id] for operation_id in invalidatable
        ),
        unexpected_operation_ids=frozenset(unexpected),
    )


def invalidate_overlapping_deletion_operations(
    session: Session,
    inventory: DeletionOverlapInventory,
    *,
    invalidated_at: datetime | None = None,
) -> int:
    """Scrub stale previews/restores after their entity rows are authorized for hard purge."""

    if inventory.blockers:
        raise ValueError("Blocking deletion overlaps must be resolved before invalidation")
    operation_ids = set(inventory.invalidatable_operation_ids)
    if not operation_ids:
        return 0
    timestamp = (invalidated_at or datetime.now(UTC)).astimezone(UTC).isoformat()
    operations: list[DeletionOperation] = []
    for chunk in _chunks(sorted(operation_ids)):
        operations.extend(
            session.scalars(select(DeletionOperation).where(DeletionOperation.id.in_(chunk)))
        )
    if {row.id for row in operations} != operation_ids or any(
        row.status not in {"previewed", "restored"} for row in operations
    ):
        raise ValueError("Deletion overlap state changed before invalidation")
    for chunk in _chunks(sorted(operation_ids)):
        session.execute(delete(DeletionItem).where(DeletionItem.operation_id.in_(chunk)))
    for operation in operations:
        operation.status = "invalidated"
        operation.selector_json = {
            "invalidated": True,
            "reason": "overlapped_entity_was_hard_purged",
        }
        operation.impact_json = {"invalidated": True}
        operation.catalog_revision = "0" * 64
        operation.selection_sha256 = "0" * 64
        operation.commit_token_hash = None
        operation.purge_token_hash = None
        operation.purge_revision = None
        operation.purge_plan_json = {
            "invalidated": True,
            "invalidated_at": timestamp,
        }
        operation.manifest_relative_path = None
        operation.item_count = 0
        operation.error_text = "Deletion preview invalidated by a completed hard purge"
    return len(operations)


@dataclass(frozen=True, slots=True)
class ConversationCorrectionPurgeInventory:
    blocking_correction_ids: frozenset[str]
    scrub_correction_ids: frozenset[str]

    @property
    def blockers(self) -> tuple[str, ...]:
        return (
            ("active_conversation_correction_references_selected_rows",)
            if self.blocking_correction_ids
            else ()
        )


def inventory_conversation_corrections(
    session: Session,
    *,
    conversation_ids: Iterable[str],
    segment_ids: Iterable[str],
) -> ConversationCorrectionPurgeInventory:
    """Resolve active blockers and reverted correction history that copies selected metadata."""

    selected_conversations = set(conversation_ids)
    selected_segments = set(segment_ids)
    referenced_correction_ids: set[str] = set()
    conditions: list[Any] = []
    if selected_conversations:
        conditions.extend(
            (
                ConversationCorrectionSegment.before_conversation_id.in_(selected_conversations),
                ConversationCorrectionSegment.after_conversation_id.in_(selected_conversations),
            )
        )
    if selected_segments:
        conditions.append(ConversationCorrectionSegment.segment_id.in_(selected_segments))
    if conditions:
        referenced_correction_ids.update(
            str(value)
            for value in session.scalars(
                select(ConversationCorrectionSegment.correction_id).where(or_(*conditions))
            )
        )

    references = selected_conversations | selected_segments
    corrections = list(
        session.scalars(select(ConversationCorrection).order_by(ConversationCorrection.id))
    )
    for correction in corrections:
        if any(
            _contains_window_reference(payload, references)
            for payload in (
                correction.request_json,
                correction.before_json,
                correction.after_json,
                correction.effect_json,
            )
        ):
            referenced_correction_ids.add(correction.id)
    by_id = {row.id: row for row in corrections}
    blocking = {
        correction_id
        for correction_id in referenced_correction_ids
        if correction_id not in by_id or by_id[correction_id].status != "reverted"
    }
    scrub = {
        correction_id
        for correction_id in referenced_correction_ids
        if correction_id in by_id and by_id[correction_id].status == "reverted"
    }
    return ConversationCorrectionPurgeInventory(
        blocking_correction_ids=frozenset(blocking),
        scrub_correction_ids=frozenset(scrub),
    )


def scrub_reverted_conversation_corrections(
    session: Session,
    inventory: ConversationCorrectionPurgeInventory,
    *,
    invalidated_at: datetime | None = None,
) -> int:
    """Remove paths, titles, snapshots, and audit detail from reverted corrections."""

    if inventory.blockers:
        raise ValueError("Active conversation corrections must be reverted before hard purge")
    correction_ids = set(inventory.scrub_correction_ids)
    if not correction_ids:
        return 0
    timestamp = (invalidated_at or datetime.now(UTC)).astimezone(UTC).isoformat()
    corrections: list[ConversationCorrection] = []
    for chunk in _chunks(sorted(correction_ids)):
        corrections.extend(
            session.scalars(
                select(ConversationCorrection).where(ConversationCorrection.id.in_(chunk))
            )
        )
    if {row.id for row in corrections} != correction_ids or any(
        row.status != "reverted" for row in corrections
    ):
        raise ValueError("Conversation correction state changed before scrubbing")
    for chunk in _chunks(sorted(correction_ids)):
        session.execute(
            delete(ConversationCorrectionAudit).where(
                ConversationCorrectionAudit.correction_id.in_(chunk)
            )
        )
        session.execute(
            delete(ConversationCorrectionSegment).where(
                ConversationCorrectionSegment.correction_id.in_(chunk)
            )
        )
    for correction in corrections:
        correction.status = "purged"
        correction.reason = None
        correction.request_json = {}
        correction.before_json = {}
        correction.after_json = {}
        correction.effect_json = {
            "scrubbed": True,
            "reason": "hard_purge_removed_conversation_history",
            "invalidated_at": timestamp,
        }
    return len(corrections)


@dataclass(frozen=True, slots=True)
class JobCheckpointScrubResult:
    scrubbed_job_ids: frozenset[str]

    @property
    def scrubbed_count(self) -> int:
        return len(self.scrubbed_job_ids)


def scrub_blob_linked_job_checkpoints(
    session: Session,
    blob_ids: Iterable[str],
    *,
    invalidated_at: datetime | None = None,
) -> JobCheckpointScrubResult:
    """Remove resume checkpoints that retain a physically purged blob identity."""

    selected = set(blob_ids)
    if not selected:
        return JobCheckpointScrubResult(frozenset())
    timestamp = (invalidated_at or datetime.now(UTC)).astimezone(UTC).isoformat()
    scrubbed: set[str] = set()
    for job in session.scalars(select(Job).order_by(Job.id)):
        if not any(
            _contains_window_reference(payload, selected)
            for payload in (job.checkpoint_json, job.counters_json)
        ):
            continue
        scrubbed.add(job.id)
        job.status = "stale"
        job.stage = "Checkpoint invalidated by hard purge"
        job.checkpoint_json = {
            "invalidated": True,
            "reason": "hard_purge_removed_blob",
            "invalidated_at": timestamp,
        }
        job.counters_json = {"invalidated": True}
        job.error_text = "A checkpoint blob was hard-purged; restart this job"
        job.finished_at = invalidated_at or datetime.now(UTC)
    return JobCheckpointScrubResult(frozenset(scrubbed))
