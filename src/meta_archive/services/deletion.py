from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import os
import secrets
import stat
import tempfile
import uuid
from collections.abc import Iterable, Iterator, Sequence
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import Engine, case, delete, exists, false, func, or_, select, text, update
from sqlalchemy.orm import Session

from meta_archive.models import (
    ActivityRecord,
    AttachmentOccurrence,
    AttachmentTextExtraction,
    AudioTranscription,
    AudioTranscriptSegment,
    Blob,
    ConflictAggregateRun,
    ConflictCalibration,
    ConflictHourlyRollup,
    ConflictLabel,
    ConflictWindowScore,
    Conversation,
    ConversationSegment,
    ConversationSegmentOverride,
    DeletionAudit,
    DeletionItem,
    DeletionOperation,
    IntegrityAnomaly,
    Job,
    MediaDerivative,
    Message,
    NlpAnnotation,
    NlpHourlyRollup,
    Participant,
    RelevanceFeedback,
    SearchWindow,
    SegmentParticipant,
    SemanticIndexConfig,
    SemanticWindowEmbedding,
    SourceFile,
    SourceRecord,
)
from meta_archive.paths import (
    ArchiveLayout,
    UnsafeArchivePathError,
    archive_entry_stat_beneath,
    create_directory_beneath,
    open_directory_beneath,
    open_regular_file_beneath,
    regular_file_stat_beneath,
    remove_directory_tree_beneath,
    replace_entry_beneath,
    unlink_regular_file_beneath,
)
from meta_archive.services.analytics_cache import analytics_cache_lock
from meta_archive.services.consolidation import (
    ConflictHoldingLease,
    ConflictHoldingLockError,
    conflict_holding_lock,
)
from meta_archive.services.deletion_lock_lease import (
    DeletionError as _DeletionError,
)
from meta_archive.services.deletion_lock_lease import (
    DeletionLockLease,
    _open_private_lock_file,
    _open_private_lock_path,
    _validate_private_lock_file,
)
from meta_archive.services.deletion_lock_lease import (
    deletion_lock as deletion_lock,
)
from meta_archive.services.deletion_purge_graph import (
    build_derived_purge_graph,
    classify_blob_ownership,
    gc_conversation_candidate_participants,
    invalidate_overlapping_deletion_operations,
    inventory_conversation_corrections,
    inventory_deletion_overlaps,
    inventory_derivative_tree,
    purge_window_feedback_and_calibrations,
    scrub_blob_linked_job_checkpoints,
    scrub_reverted_conversation_corrections,
)
from meta_archive.services.deletion_restore_state import (
    DerivedRestoreStateError,
    capture_derived_restore_state,
    restore_derived_restore_state,
    seal_quarantined_derived_state,
)
from meta_archive.services.deletion_secure_publication import (
    PrivateFileSnapshot,
    SecurePublicationError,
    publish_private_file,
    read_private_file,
    recover_private_publication,
)
from meta_archive.services.deletion_upgrade_state import (
    DeletionUpgradeStateError,
    SavedDeletionResources,
    WorkspaceUpgradeStateReceipt,
    load_workspace_upgrade_state,
    publish_workspace_upgrade_state,
)
from meta_archive.services.library import (
    LibraryBuildError,
    build_filesystem_library,
    library_build_lock,
)
from meta_archive.services.media_derivatives import write_status_manifest
from meta_archive.services.purge_rewrite import (
    RawRewritePlan,
    RawSourceRewriteError,
    apply_raw_rewrite_catalog,
    build_raw_rewrite_plan,
    publish_raw_rewrite_files,
    raw_rewrite_live_targets,
    validate_raw_rewrite_physical_sources,
)
from meta_archive.services.semantic_state import (
    SemanticWindowIdentity,
    invalidate_ready_semantic_indexes,
    lance_table_names,
    lance_window_identity,
    semantic_config_window_count,
    semantic_config_window_ids,
    semantic_window_identity,
)
from meta_archive.services.source_import_lock import (
    SourceImportPublicationLease,
    SourceImportPublicationLockError,
    source_import_publication_lock,
)
from meta_archive.services.stars import (
    StarTargetKey,
    StarTargetType,
    delete_stars_for_targets,
    messenger_contact_ids_eliminated_by_conversation_removal,
    star_rows_by_target,
)

EntityType = Literal["message", "activity", "conversation", "attachment"]
TextMode = Literal["exact", "keyword"]
TextVariant = Literal["raw", "display"]
PurgeScope = Literal["normalized", "workspace"]

ENTITY_TYPES = frozenset({"message", "activity", "conversation", "attachment"})
MEDIA_TYPES = frozenset({"photo", "video", "audio", "gif", "sticker", "document", "other"})
AVAILABILITY = frozenset({"resolved", "missing", "unlinked", "external"})
NLP_ENTITY_TYPES: tuple[EntityType, ...] = ("message", "activity")
MANIFEST_VERSION = 2
ID_CHUNK = 500
PURGE_ROLLBACK_JOURNAL_VERSION = 1
PURGE_ROLLBACK_JOURNAL_NAME = ".precommit-rollback.json"

# Preserve the long-standing public import while the lock implementation lives in a cycle-free
# module that other publishers can depend on safely.
DeletionError = _DeletionError


class UnsafeDeletionError(DeletionError):
    pass


class StaleDeletionPreviewError(DeletionError):
    pass


class InvalidDeletionTokenError(DeletionError):
    pass


class HardPurgeBlockedError(DeletionError):
    pass


@dataclass(frozen=True, slots=True)
class DeletionSelector:
    entity_types: tuple[EntityType, ...] = ("message", "activity")
    text: str | None = None
    text_mode: TextMode = "exact"
    case_sensitive: bool = True
    whole_field: bool = False
    text_variant: TextVariant = "display"
    from_date: str | None = None
    to_date: str | None = None
    timezone: str = "America/New_York"
    conversation_ids: tuple[str, ...] = ()
    participant: str | None = None
    media_types: tuple[str, ...] = ()
    availability: tuple[str, ...] = ()
    entity_ids: dict[str, tuple[str, ...]] = field(default_factory=dict)
    allow_all: bool = False

    def validate(self) -> None:
        invalid_entities = set(self.entity_types) - ENTITY_TYPES
        if not self.entity_types or invalid_entities:
            raise UnsafeDeletionError(f"Invalid deletion entity types: {sorted(invalid_entities)}")
        if self.text is not None and not self.text:
            raise UnsafeDeletionError("Deletion text cannot be empty")
        if self.text_mode not in {"exact", "keyword"}:
            raise UnsafeDeletionError(f"Invalid deletion text mode: {self.text_mode}")
        if self.text and self.text_mode == "keyword" and "attachment" in self.entity_types:
            raise UnsafeDeletionError(
                "Keyword text matching is not supported for attachment deletion; "
                "use exact matching for filenames/source URIs or filter attachments "
                "by date, media type, availability, conversation, participant, or explicit IDs"
            )
        if self.text_variant not in {"raw", "display"}:
            raise UnsafeDeletionError(f"Invalid deletion text variant: {self.text_variant}")
        invalid_media = set(self.media_types) - MEDIA_TYPES
        if invalid_media:
            raise UnsafeDeletionError(f"Invalid deletion media types: {sorted(invalid_media)}")
        invalid_availability = set(self.availability) - AVAILABILITY
        if invalid_availability:
            raise UnsafeDeletionError(
                f"Invalid attachment availability: {sorted(invalid_availability)}"
            )
        invalid_id_types = set(self.entity_ids) - ENTITY_TYPES
        if invalid_id_types:
            raise UnsafeDeletionError(f"Invalid explicit entity IDs: {sorted(invalid_id_types)}")
        try:
            parsed_from = date.fromisoformat(self.from_date) if self.from_date else None
            parsed_to = date.fromisoformat(self.to_date) if self.to_date else None
        except ValueError as exc:
            raise UnsafeDeletionError("Deletion dates must use YYYY-MM-DD") from exc
        if parsed_from and parsed_to and parsed_from > parsed_to:
            raise UnsafeDeletionError("Deletion start date must not be after its end date")
        try:
            ZoneInfo(self.timezone)
        except ZoneInfoNotFoundError as exc:
            raise UnsafeDeletionError(f"Unknown deletion timezone: {self.timezone}") from exc
        narrowed = any(
            (
                self.text,
                self.from_date,
                self.to_date,
                self.conversation_ids,
                self.participant,
                self.media_types,
                self.availability,
                any(self.entity_ids.values()),
            )
        )
        if not narrowed and not self.allow_all:
            raise UnsafeDeletionError(
                "Refusing an unbounded deletion preview without explicit allow_all"
            )

    def utc_range_ms(self) -> tuple[int | None, int | None]:
        zone = ZoneInfo(self.timezone)
        start = (
            datetime.combine(date.fromisoformat(self.from_date), time.min, zone)
            if self.from_date
            else None
        )
        end = (
            datetime.combine(date.fromisoformat(self.to_date) + timedelta(days=1), time.min, zone)
            if self.to_date
            else None
        )
        return (
            round(start.astimezone(UTC).timestamp() * 1000) if start else None,
            round(end.astimezone(UTC).timestamp() * 1000) if end else None,
        )

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["entity_types"] = list(self.entity_types)
        payload["conversation_ids"] = list(self.conversation_ids)
        payload["media_types"] = list(self.media_types)
        payload["availability"] = list(self.availability)
        payload["entity_ids"] = {
            key: list(values) for key, values in sorted(self.entity_ids.items())
        }
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> DeletionSelector:
        entity_ids = payload.get("entity_ids") or {}
        return cls(
            entity_types=tuple(payload.get("entity_types") or ()),
            text=payload.get("text"),
            text_mode=payload.get("text_mode", "exact"),
            case_sensitive=bool(payload.get("case_sensitive", True)),
            whole_field=bool(payload.get("whole_field", False)),
            text_variant=payload.get("text_variant", "display"),
            from_date=payload.get("from_date"),
            to_date=payload.get("to_date"),
            timezone=payload.get("timezone", "America/New_York"),
            conversation_ids=tuple(payload.get("conversation_ids") or ()),
            participant=payload.get("participant"),
            media_types=tuple(payload.get("media_types") or ()),
            availability=tuple(payload.get("availability") or ()),
            entity_ids={key: tuple(value) for key, value in entity_ids.items()},
            allow_all=bool(payload.get("allow_all", False)),
        )


@dataclass(frozen=True, slots=True)
class SelectionItem:
    entity_type: EntityType
    entity_id: str
    reason: str
    snapshot_sha256: str
    previous_deleted_at: datetime | None
    snapshot_json: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class PreviewState:
    items: tuple[SelectionItem, ...]
    impact: dict[str, Any]
    catalog_revision: str
    selection_sha256: str


@dataclass(frozen=True, slots=True)
class DeletionPreview:
    operation_id: str
    commit_token: str
    catalog_revision: str
    selection_sha256: str
    impact: dict[str, Any]
    manifest: str


@dataclass(frozen=True, slots=True)
class MutationResult:
    operation_id: str
    status: str
    affected: dict[str, int]
    audit_id: str


@dataclass(frozen=True, slots=True)
class PurgePlan:
    operation_id: str
    scope: PurgeScope
    executable: bool
    blockers: tuple[str, ...]
    purge_token: str
    purge_revision: str
    impact: dict[str, Any]


@dataclass(frozen=True, slots=True)
class VectorDeletionResult:
    deleted_rows: int
    table_identities: dict[str, SemanticWindowIdentity]
    stale_config_count: int = 0
    memberships_removed: int = 0
    store_discarded: bool = False

    @property
    def table_row_counts(self) -> dict[str, int]:
        return {
            table_name: identity.row_count for table_name, identity in self.table_identities.items()
        }


def _chunks(values: Sequence[str], size: int = ID_CHUNK) -> Iterator[Sequence[str]]:
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def _result_rowcount(result: Any) -> int:
    return int(getattr(result, "rowcount", 0) or 0)


def _token_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, datetime):
        normalized = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
        return normalized.isoformat()
    return value


@contextmanager
def vector_store_lock(layout: ArchiveLayout) -> Iterator[None]:
    """Coordinate purge snapshots with the semantic-index builder's on-disk lock."""

    descriptor: int | None = None
    directory_descriptor: int | None = None
    try:
        descriptor, directory_descriptor = _open_private_lock_file(layout, "semantic-index.lock")
    except (OSError, UnsafeArchivePathError) as exc:
        raise HardPurgeBlockedError("The semantic vector index lock file is unsafe") from exc
    try:
        with os.fdopen(descriptor, "r+", encoding="utf-8") as handle:
            descriptor = None
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise HardPurgeBlockedError(
                    "The semantic vector index is currently being rebuilt"
                ) from exc
            try:
                assert directory_descriptor is not None
                _validate_private_lock_file(
                    handle.fileno(), directory_descriptor, "semantic-index.lock"
                )
                handle.seek(0)
                handle.truncate()
                json.dump(
                    {
                        "pid": os.getpid(),
                        "started_at": datetime.now(UTC).isoformat(),
                        "operation": "hard-purge",
                    },
                    handle,
                    sort_keys=True,
                )
                handle.flush()
                os.fsync(handle.fileno())
            except OSError as exc:
                raise HardPurgeBlockedError(
                    "The semantic vector index lock file became unsafe"
                ) from exc
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if directory_descriptor is not None:
            os.close(directory_descriptor)


@contextmanager
def purge_worker_locks(layout: ArchiveLayout) -> Iterator[None]:
    """Freeze every worker that can publish blob- or text-derived state.

    These are the same advisory lock files used by the media, transcription, attachment-text,
    NLP, and conflict-aggregate workers.  Hard purge takes them non-blockingly while already
    holding ``deletion_lock``.  Failing fast avoids deadlocks with an in-flight worker and, more
    importantly, prevents a worker from recreating a derivative after purge inventory but before
    the selected blob or search window is removed.
    """

    lock_specs = (
        (layout.thumbnails / ".build.lock", "media derivative build"),
        (layout.catalog / "audio-transcription.lock", "audio transcription job"),
        (layout.catalog / "attachment-text.lock", "attachment text job"),
        (layout.catalog / "nlp-annotations.lock", "NLP annotation job"),
        (layout.catalog / "conflict-calibration.lock", "conflict calibration job"),
        (layout.catalog / "conflict-aggregates.lock", "conflict aggregate job"),
    )
    with ExitStack() as stack:
        for lock_path, label in lock_specs:
            try:
                descriptor, directory_descriptor = _open_private_lock_path(layout, lock_path)
            except (OSError, UnsafeArchivePathError) as exc:
                raise HardPurgeBlockedError(f"The {label} lock file is unsafe") from exc
            stack.callback(os.close, directory_descriptor)
            stack.callback(os.close, descriptor)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise HardPurgeBlockedError(f"A {label} is currently running") from exc
            stack.callback(fcntl.flock, descriptor, fcntl.LOCK_UN)
            try:
                _validate_private_lock_file(
                    descriptor,
                    directory_descriptor,
                    lock_path.name,
                )
            except OSError as exc:
                raise HardPurgeBlockedError(f"The {label} lock file became unsafe") from exc
        yield


@dataclass(frozen=True)
class PurgeSourcePublisherLeases:
    """Live leases held beneath the deletion lock for composite purge workflows."""

    source_import: SourceImportPublicationLease
    conflict_holding: ConflictHoldingLease


@contextmanager
def purge_source_publisher_locks(layout: ArchiveLayout) -> Iterator[PurgeSourcePublisherLeases]:
    """Freeze source import, filesystem library, and conflict-copy publishers in order."""

    try:
        with (
            source_import_publication_lock(layout, blocking=False) as source_lease,
            library_build_lock(layout, blocking=False),
            conflict_holding_lock(layout, blocking=False) as conflict_lease,
        ):
            yield PurgeSourcePublisherLeases(
                source_import=source_lease,
                conflict_holding=conflict_lease,
            )
    except (
        SourceImportPublicationLockError,
        LibraryBuildError,
        ConflictHoldingLockError,
    ) as exc:
        raise HardPurgeBlockedError(
            "A source import, filesystem library, or conflict-holding publication is running"
        ) from exc


@contextmanager
def immediate_session(engine: Engine) -> Iterator[Session]:
    """Hold SQLite's writer reservation across revision validation and mutation."""

    with engine.connect() as connection:
        connection.exec_driver_sql("BEGIN IMMEDIATE")
        session = Session(bind=connection)
        try:
            yield session
            session.flush()
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            session.close()


def _media_type_expression() -> Any:
    raw_kind = func.lower(AttachmentOccurrence.kind)
    media_type = func.lower(func.coalesce(AttachmentOccurrence.media_type, ""))
    return case(
        (raw_kind.in_(("audio_files", "audio")), "audio"),
        (raw_kind.in_(("videos", "video")), "video"),
        (raw_kind.in_(("gifs", "gif")), "gif"),
        (raw_kind == "sticker", "sticker"),
        (raw_kind.in_(("photos", "photo", "thread_image")), "photo"),
        (media_type.like("audio/%"), "audio"),
        (media_type.like("video/%"), "video"),
        (media_type.like("image/gif%"), "gif"),
        (media_type.like("image/%"), "photo"),
        (
            or_(
                raw_kind.in_(("files", "file", "support_file", "document")),
                media_type.in_(("application/pdf", "text/plain")),
            ),
            "document",
        ),
        else_="other",
    )


def _literal_condition(columns: Sequence[Any], selector: DeletionSelector) -> Any:
    if selector.text is None:
        return None
    if selector.text_mode == "keyword":
        raise AssertionError("Keyword mode uses FTS row IDs instead of literal conditions")
    conditions: list[Any] = []
    for column in columns:
        value: Any = func.coalesce(column, "")
        literal = selector.text
        if not selector.case_sensitive:
            value = func.unicode_casefold(value)
            literal = literal.casefold()
        conditions.append(
            value == literal if selector.whole_field else func.instr(value, literal) > 0
        )
    return or_(*conditions)


def _date_conditions(column: Any, selector: DeletionSelector) -> list[Any]:
    start_ms, end_ms = selector.utc_range_ms()
    conditions: list[Any] = []
    if start_ms is not None:
        conditions.append(column >= start_ms)
    if end_ms is not None:
        conditions.append(column < end_ms)
    return conditions


def _participant_segment_exists(segment_id: Any, participant: str) -> Any:
    normalized = participant.casefold()
    return exists(
        select(SegmentParticipant.id)
        .join(Participant, Participant.id == SegmentParticipant.participant_id)
        .where(
            SegmentParticipant.segment_id == segment_id,
            func.instr(Participant.name_search, normalized) > 0,
        )
    )


def _attachment_filters(selector: DeletionSelector) -> list[Any]:
    filters: list[Any] = []
    if selector.media_types:
        filters.append(_media_type_expression().in_(selector.media_types))
    if selector.availability:
        filters.append(AttachmentOccurrence.availability.in_(selector.availability))
    return filters


def _attachment_participant_condition(participant: str) -> Any:
    normalized = participant.casefold()
    message_match = exists(
        select(Message.id).where(
            Message.id == AttachmentOccurrence.message_id,
            Message.deleted_at.is_(None),
            or_(
                func.instr(Message.sender_search, normalized) > 0,
                _participant_segment_exists(Message.segment_id, normalized),
            ),
        )
    )
    activity_match = exists(
        select(ActivityRecord.id).where(
            ActivityRecord.id == AttachmentOccurrence.activity_record_id,
            ActivityRecord.deleted_at.is_(None),
            func.instr(ActivityRecord.actor_search, normalized) > 0,
        )
    )
    conversation_match = exists(
        select(ConversationSegment.id)
        .join(SegmentParticipant, SegmentParticipant.segment_id == ConversationSegment.id)
        .join(Participant, Participant.id == SegmentParticipant.participant_id)
        .where(
            ConversationSegment.conversation_id == AttachmentOccurrence.conversation_id,
            func.instr(Participant.name_search, normalized) > 0,
        )
    )
    return or_(message_match, activity_match, conversation_match)


def _explicit_id_condition(selector: DeletionSelector, entity_type: EntityType, column: Any) -> Any:
    """Narrow an ID-driven selector without broadening its other entity types."""

    if not selector.entity_ids:
        return None
    identifiers = selector.entity_ids.get(entity_type, ())
    return column.in_(identifiers) if identifiers else False


def _keyword_message_pks(session: Session, query: str) -> list[int]:
    try:
        return [
            int(rowid)
            for rowid in session.scalars(
                text(
                    "SELECT rowid FROM message_fts "
                    "WHERE content_search MATCH :query "
                    "OR transcription_search MATCH :query"
                ).bindparams(query=query)
            )
        ]
    except Exception as exc:
        raise UnsafeDeletionError(f"Invalid message FTS query: {exc}") from exc


def _keyword_activity_pks(session: Session, query: str) -> list[int]:
    try:
        return [
            int(rowid)
            for rowid in session.scalars(
                text(
                    "SELECT rowid FROM activity_fts "
                    "WHERE title_search MATCH :query OR content_search MATCH :query"
                ).bindparams(query=query)
            )
        ]
    except Exception as exc:
        raise UnsafeDeletionError(f"Invalid activity FTS query: {exc}") from exc


def _direct_selection(session: Session, selector: DeletionSelector) -> dict[EntityType, set[str]]:
    selected: dict[EntityType, set[str]] = {
        "message": set(),
        "activity": set(),
        "conversation": set(),
        "attachment": set(),
    }
    attachment_filters = _attachment_filters(selector)
    if "message" in selector.entity_types:
        statement = select(Message.id).where(Message.deleted_at.is_(None))
        if selector.text:
            if selector.text_mode == "keyword":
                pks = _keyword_message_pks(session, selector.text)
                statement = statement.where(Message.pk.in_(pks or [-1]))
            else:
                message_columns: Sequence[Any] = (
                    (Message.content_raw, Message.transcription_raw)
                    if selector.text_variant == "raw"
                    else (Message.content_display, Message.transcription_display)
                )
                statement = statement.where(_literal_condition(message_columns, selector))
        statement = statement.where(*_date_conditions(Message.timestamp_ms, selector))
        if selector.conversation_ids:
            statement = statement.where(Message.conversation_id.in_(selector.conversation_ids))
        if selector.participant:
            normalized = selector.participant.casefold()
            statement = statement.where(
                or_(
                    func.instr(Message.sender_search, normalized) > 0,
                    _participant_segment_exists(Message.segment_id, normalized),
                )
            )
        if attachment_filters:
            statement = statement.where(
                exists(
                    select(AttachmentOccurrence.id).where(
                        AttachmentOccurrence.message_id == Message.id,
                        AttachmentOccurrence.deleted_at.is_(None),
                        *attachment_filters,
                    )
                )
            )
        explicit_condition = _explicit_id_condition(selector, "message", Message.id)
        if explicit_condition is not None:
            statement = statement.where(explicit_condition)
        selected["message"].update(session.scalars(statement))

    if "activity" in selector.entity_types:
        statement = select(ActivityRecord.id).where(ActivityRecord.deleted_at.is_(None))
        if selector.text:
            if selector.text_mode == "keyword":
                pks = _keyword_activity_pks(session, selector.text)
                statement = statement.where(ActivityRecord.pk.in_(pks or [-1]))
            else:
                activity_columns: Sequence[Any] = (
                    (ActivityRecord.title_raw, ActivityRecord.content_raw)
                    if selector.text_variant == "raw"
                    else (ActivityRecord.title_display, ActivityRecord.content_display)
                )
                statement = statement.where(_literal_condition(activity_columns, selector))
        statement = statement.where(*_date_conditions(ActivityRecord.timestamp_ms, selector))
        if selector.conversation_ids:
            statement = statement.where(false())
        if selector.participant:
            statement = statement.where(
                func.instr(ActivityRecord.actor_search, selector.participant.casefold()) > 0
            )
        if attachment_filters:
            statement = statement.where(
                exists(
                    select(AttachmentOccurrence.id).where(
                        AttachmentOccurrence.activity_record_id == ActivityRecord.id,
                        AttachmentOccurrence.deleted_at.is_(None),
                        *attachment_filters,
                    )
                )
            )
        explicit_condition = _explicit_id_condition(selector, "activity", ActivityRecord.id)
        if explicit_condition is not None:
            statement = statement.where(explicit_condition)
        selected["activity"].update(session.scalars(statement))

    if "conversation" in selector.entity_types:
        statement = select(Conversation.id).where(Conversation.deleted_at.is_(None))
        if selector.text:
            if selector.text_mode == "keyword":
                matching_message_pks = _keyword_message_pks(session, selector.text)
                statement = statement.where(
                    exists(
                        select(Message.pk).where(
                            Message.conversation_id == Conversation.id,
                            Message.deleted_at.is_(None),
                            Message.pk.in_(matching_message_pks or [-1]),
                        )
                    )
                )
            else:
                conversation_columns: Sequence[Any] = (
                    (Conversation.title_raw,)
                    if selector.text_variant == "raw"
                    else (Conversation.title_display,)
                )
                conversation_message_columns: Sequence[Any] = (
                    (Message.content_raw, Message.transcription_raw)
                    if selector.text_variant == "raw"
                    else (Message.content_display, Message.transcription_display)
                )
                statement = statement.where(
                    or_(
                        _literal_condition(conversation_columns, selector),
                        exists(
                            select(Message.pk).where(
                                Message.conversation_id == Conversation.id,
                                Message.deleted_at.is_(None),
                                _literal_condition(conversation_message_columns, selector),
                            )
                        ),
                    )
                )
        start_ms, end_ms = selector.utc_range_ms()
        if start_ms is not None:
            statement = statement.where(Conversation.last_activity_ms >= start_ms)
        if end_ms is not None:
            statement = statement.where(Conversation.first_activity_ms < end_ms)
        if selector.conversation_ids:
            statement = statement.where(Conversation.id.in_(selector.conversation_ids))
        if selector.participant:
            normalized = selector.participant.casefold()
            statement = statement.where(
                exists(
                    select(ConversationSegment.id)
                    .join(
                        SegmentParticipant,
                        SegmentParticipant.segment_id == ConversationSegment.id,
                    )
                    .join(Participant, Participant.id == SegmentParticipant.participant_id)
                    .where(
                        ConversationSegment.conversation_id == Conversation.id,
                        func.instr(Participant.name_search, normalized) > 0,
                    )
                )
            )
        if attachment_filters:
            statement = statement.where(
                exists(
                    select(AttachmentOccurrence.id).where(
                        AttachmentOccurrence.conversation_id == Conversation.id,
                        AttachmentOccurrence.deleted_at.is_(None),
                        *attachment_filters,
                    )
                )
            )
        explicit_condition = _explicit_id_condition(selector, "conversation", Conversation.id)
        if explicit_condition is not None:
            statement = statement.where(explicit_condition)
        selected["conversation"].update(session.scalars(statement))

    if "attachment" in selector.entity_types:
        statement = select(AttachmentOccurrence.id).where(
            AttachmentOccurrence.deleted_at.is_(None), *attachment_filters
        )
        if selector.text:
            statement = statement.where(
                _literal_condition(
                    (AttachmentOccurrence.filename, AttachmentOccurrence.source_uri), selector
                )
            )
        statement = statement.where(*_date_conditions(AttachmentOccurrence.timestamp_ms, selector))
        if selector.conversation_ids:
            statement = statement.where(
                AttachmentOccurrence.conversation_id.in_(selector.conversation_ids)
            )
        if selector.participant:
            statement = statement.where(_attachment_participant_condition(selector.participant))
        explicit_condition = _explicit_id_condition(selector, "attachment", AttachmentOccurrence.id)
        if explicit_condition is not None:
            statement = statement.where(explicit_condition)
        selected["attachment"].update(session.scalars(statement))
    return selected


def _expand_selection(
    session: Session, direct: dict[EntityType, set[str]]
) -> tuple[dict[EntityType, set[str]], dict[tuple[EntityType, str], str]]:
    selected = {entity_type: set(values) for entity_type, values in direct.items()}
    reasons = {
        (entity_type, entity_id): "direct"
        for entity_type, values in direct.items()
        for entity_id in values
    }
    conversation_ids = sorted(selected["conversation"])
    for chunk in _chunks(conversation_ids):
        message_ids = list(
            session.scalars(
                select(Message.id).where(
                    Message.conversation_id.in_(chunk), Message.deleted_at.is_(None)
                )
            )
        )
        attachment_ids = list(
            session.scalars(
                select(AttachmentOccurrence.id).where(
                    AttachmentOccurrence.conversation_id.in_(chunk),
                    AttachmentOccurrence.deleted_at.is_(None),
                )
            )
        )
        for entity_id in message_ids:
            selected["message"].add(entity_id)
            reasons.setdefault(("message", entity_id), "conversation_cascade")
        for entity_id in attachment_ids:
            selected["attachment"].add(entity_id)
            reasons.setdefault(("attachment", entity_id), "conversation_cascade")

    for chunk in _chunks(sorted(selected["message"])):
        for entity_id in session.scalars(
            select(AttachmentOccurrence.id).where(
                AttachmentOccurrence.message_id.in_(chunk),
                AttachmentOccurrence.deleted_at.is_(None),
            )
        ):
            selected["attachment"].add(entity_id)
            reasons.setdefault(("attachment", entity_id), "message_cascade")

    frontier = set(selected["activity"])
    while frontier:
        children: set[str] = set()
        for chunk in _chunks(sorted(frontier)):
            children.update(
                session.scalars(
                    select(ActivityRecord.id).where(
                        ActivityRecord.parent_id.in_(chunk), ActivityRecord.deleted_at.is_(None)
                    )
                )
            )
        children -= selected["activity"]
        for entity_id in children:
            selected["activity"].add(entity_id)
            reasons[("activity", entity_id)] = "activity_cascade"
        frontier = children
    for chunk in _chunks(sorted(selected["activity"])):
        for entity_id in session.scalars(
            select(AttachmentOccurrence.id).where(
                AttachmentOccurrence.activity_record_id.in_(chunk),
                AttachmentOccurrence.deleted_at.is_(None),
            )
        ):
            selected["attachment"].add(entity_id)
            reasons.setdefault(("attachment", entity_id), "activity_cascade")
    return selected, reasons


_ENTITY_MODELS: dict[EntityType, Any] = {
    "message": Message,
    "activity": ActivityRecord,
    "conversation": Conversation,
    "attachment": AttachmentOccurrence,
}


def _rows_by_id(
    session: Session, entity_type: EntityType, identifiers: Iterable[str]
) -> dict[str, Any]:
    model = _ENTITY_MODELS[entity_type]
    result: dict[str, Any] = {}
    ordered = sorted(identifiers)
    for chunk in _chunks(ordered):
        rows = session.scalars(select(model).where(model.id.in_(chunk)))
        result.update((row.id, row) for row in rows)
    return result


def _row_snapshot(row: Any) -> tuple[dict[str, Any], str]:
    snapshot = {
        column.name: _json_value(getattr(row, column.name)) for column in row.__table__.columns
    }
    return snapshot, _canonical_sha256(snapshot)


def _selection_items(
    session: Session,
    selected: dict[EntityType, set[str]],
    reasons: dict[tuple[EntityType, str], str],
) -> tuple[SelectionItem, ...]:
    items: list[SelectionItem] = []
    for entity_type in sorted(ENTITY_TYPES):
        typed = cast(EntityType, entity_type)
        rows = _rows_by_id(session, typed, selected[typed])
        if set(rows) != selected[typed]:
            missing = sorted(selected[typed] - set(rows))
            raise StaleDeletionPreviewError(
                f"Selected {typed} rows disappeared while previewing: {missing[:5]}"
            )
        for entity_id, row in sorted(rows.items()):
            snapshot, snapshot_sha256 = _row_snapshot(row)
            items.append(
                SelectionItem(
                    entity_type=typed,
                    entity_id=entity_id,
                    reason=reasons[(typed, entity_id)],
                    snapshot_sha256=snapshot_sha256,
                    previous_deleted_at=row.deleted_at,
                    snapshot_json=snapshot if typed == "conversation" else None,
                )
            )
    return tuple(items)


def _scalar_ids_for_chunks(
    session: Session,
    values: Iterable[str],
    statement_factory: Any,
) -> set[str]:
    result: set[str] = set()
    for chunk in _chunks(sorted(values)):
        result.update(str(value) for value in session.scalars(statement_factory(chunk)))
    return result


def _selection_resources(
    session: Session, selected: dict[EntityType, set[str]]
) -> dict[str, set[str]]:
    message_ids = selected["message"]
    activity_ids = selected["activity"]
    attachment_ids = selected["attachment"]
    attachment_rows = _rows_by_id(session, "attachment", attachment_ids)
    blob_ids = {row.blob_sha256 for row in attachment_rows.values() if row.blob_sha256 is not None}
    source_file_ids: set[int] = set()
    conversation_ids = set(selected["conversation"])
    for entity_type in ("message", "activity"):
        for row in _rows_by_id(session, entity_type, selected[entity_type]).values():
            source_file_ids.add(row.source_file_id)
            if entity_type == "message":
                conversation_ids.add(row.conversation_id)
    for row in attachment_rows.values():
        if row.source_file_id is not None:
            source_file_ids.add(row.source_file_id)
        if row.resolved_source_file_id is not None:
            source_file_ids.add(row.resolved_source_file_id)
        if row.conversation_id is not None:
            conversation_ids.add(row.conversation_id)
    candidate_binary_source_file_ids: set[int] = set()
    source_sha256_by_id: dict[int, str] = {}
    if source_file_ids and blob_ids:
        for source_file_id, sha256 in session.execute(
            select(SourceFile.id, SourceFile.sha256).where(
                SourceFile.id.in_(source_file_ids),
                SourceFile.sha256.in_(blob_ids),
            )
        ):
            if sha256 is not None:
                candidate_binary_source_file_ids.add(int(source_file_id))
                source_sha256_by_id[int(source_file_id)] = str(sha256)

    shared_source_file_ids: set[int] = set()
    live_shared_source_blob_ids: set[str] = set()
    recoverable_shared_source_blob_ids: set[str] = set()
    for chunk in _chunks(sorted(str(value) for value in candidate_binary_source_file_ids)):
        integer_chunk = [int(value) for value in chunk]
        for source_id, resolved_id, deleted_at in session.execute(
            select(
                AttachmentOccurrence.source_file_id,
                AttachmentOccurrence.resolved_source_file_id,
                AttachmentOccurrence.deleted_at,
            ).where(
                AttachmentOccurrence.id.not_in(attachment_ids),
                or_(
                    AttachmentOccurrence.source_file_id.in_(integer_chunk),
                    AttachmentOccurrence.resolved_source_file_id.in_(integer_chunk),
                ),
            )
        ):
            referenced_ids = {
                int(value)
                for value in (source_id, resolved_id)
                if value is not None and int(value) in candidate_binary_source_file_ids
            }
            shared_source_file_ids.update(referenced_ids)
            referenced_blobs = {
                source_sha256_by_id[value]
                for value in referenced_ids
                if value in source_sha256_by_id
            }
            recoverable_shared_source_blob_ids.update(referenced_blobs)
            if deleted_at is None:
                live_shared_source_blob_ids.update(referenced_blobs)
    binary_source_file_ids = {
        str(value) for value in candidate_binary_source_file_ids - shared_source_file_ids
    }
    all_source_blob_refs: dict[str, set[int]] = {blob_id: set() for blob_id in blob_ids}
    for chunk in _chunks(sorted(blob_ids)):
        for source_file_id, sha256 in session.execute(
            select(SourceFile.id, SourceFile.sha256).where(SourceFile.sha256.in_(chunk))
        ):
            if sha256 is not None:
                all_source_blob_refs[str(sha256)].add(int(source_file_id))
    handled_source_file_ids = {int(value) for value in binary_source_file_ids} | (
        shared_source_file_ids
    )
    outside_source_file_ids = {
        source_file_id
        for references in all_source_blob_refs.values()
        for source_file_id in references
        if source_file_id not in handled_source_file_ids
    }
    outside_source_blob_ids = {
        blob_id
        for blob_id, references in all_source_blob_refs.items()
        if references & outside_source_file_ids
    }

    ownership = classify_blob_ownership(session, attachment_ids)
    live_retained_blob_ids = set(ownership.live_retained_blob_ids) | live_shared_source_blob_ids
    recoverably_referenced_blob_ids = (
        set(ownership.shared_blob_ids) | recoverable_shared_source_blob_ids
    )
    exclusive_blob_ids = blob_ids - live_retained_blob_ids
    # Attachment ownership alone cannot prove physical exclusivity. A retained or separately
    # restorable occurrence can reach the same bytes through its source/resolved SourceFile even
    # when its own ``blob_sha256`` is null. Any additional SourceFile row for the bytes is likewise
    # a physical owner and is handled conservatively by the purge blocker below.
    source_protected_blob_ids = recoverable_shared_source_blob_ids | outside_source_blob_ids
    physically_exclusive_blob_ids = set(ownership.physically_exclusive_blob_ids) - (
        source_protected_blob_ids
    )
    logical_graph = build_derived_purge_graph(
        session,
        message_ids=message_ids,
        activity_ids=activity_ids,
        conversation_ids=selected["conversation"],
        attachment_ids=attachment_ids,
        physical_blob_ids=exclusive_blob_ids,
    )
    physical_graph = build_derived_purge_graph(
        session,
        message_ids=message_ids,
        activity_ids=activity_ids,
        conversation_ids=selected["conversation"],
        attachment_ids=attachment_ids,
        physical_blob_ids=physically_exclusive_blob_ids,
    )
    restore_graph = build_derived_purge_graph(
        session,
        message_ids=message_ids,
        activity_ids=activity_ids,
        conversation_ids=selected["conversation"],
        attachment_ids=attachment_ids,
        physical_blob_ids=blob_ids,
    )
    return {
        "blob_ids": blob_ids,
        "shared_blob_ids": live_retained_blob_ids,
        "recoverably_referenced_blob_ids": recoverably_referenced_blob_ids,
        "exclusive_blob_ids": exclusive_blob_ids,
        "physically_exclusive_blob_ids": physically_exclusive_blob_ids,
        "source_file_ids": {str(value) for value in source_file_ids},
        "binary_source_file_ids": binary_source_file_ids,
        "shared_source_file_ids": {str(value) for value in shared_source_file_ids},
        "outside_source_file_ids": {str(value) for value in outside_source_file_ids},
        "outside_source_blob_ids": outside_source_blob_ids,
        "conversation_ids": conversation_ids,
        "search_window_ids": set(logical_graph.search_window_ids),
        "purge_search_window_ids": set(physical_graph.search_window_ids),
        "audio_segment_ids": set(logical_graph.audio_segment_ids),
        "attachment_extraction_ids": set(logical_graph.attachment_extraction_ids),
        "physical_audio_segment_ids": set(physical_graph.audio_segment_ids),
        "physical_attachment_extraction_ids": set(physical_graph.attachment_extraction_ids),
        "restore_audio_segment_ids": set(restore_graph.audio_segment_ids),
        "restore_attachment_extraction_ids": set(restore_graph.attachment_extraction_ids),
        "conversation_participant_ids": set(physical_graph.conversation_participant_candidate_ids),
        "conversation_segment_ids": set(physical_graph.conversation_segment_ids),
    }


def _count_for_chunks(
    session: Session,
    values: Iterable[str],
    statement_factory: Any,
) -> int:
    total = 0
    for chunk in _chunks(sorted(values)):
        total += int(session.scalar(statement_factory(chunk)) or 0)
    return total


def _impact(
    session: Session,
    selected: dict[EntityType, set[str]],
    reasons: dict[tuple[EntityType, str], str],
) -> dict[str, Any]:
    resources = _selection_resources(session, selected)
    attachment_rows = _rows_by_id(session, "attachment", selected["attachment"])
    occurrence_bytes = sum(int(row.size_bytes or 0) for row in attachment_rows.values())
    blob_ids = resources["blob_ids"]
    unique_blob_bytes = 0
    if blob_ids:
        for chunk in _chunks(sorted(blob_ids)):
            unique_blob_bytes += int(
                session.scalar(
                    select(func.coalesce(func.sum(Blob.size_bytes), 0)).where(
                        Blob.sha256.in_(chunk)
                    )
                )
                or 0
            )
    source_file_ids = {int(value) for value in resources["source_file_ids"]}
    raw_source_bytes = 0
    if source_file_ids:
        for chunk in _chunks([str(value) for value in sorted(source_file_ids)]):
            raw_source_bytes += int(
                session.scalar(
                    select(func.coalesce(func.sum(SourceFile.size_bytes), 0)).where(
                        SourceFile.id.in_([int(value) for value in chunk])
                    )
                )
                or 0
            )
    derivative_count = _count_for_chunks(
        session,
        blob_ids,
        lambda chunk: select(func.count(MediaDerivative.id)).where(
            MediaDerivative.blob_sha256.in_(chunk)
        ),
    )
    derivative_bytes = 0
    for chunk in _chunks(sorted(blob_ids)):
        derivative_bytes += int(
            session.scalar(
                select(func.coalesce(func.sum(MediaDerivative.size_bytes), 0)).where(
                    MediaDerivative.blob_sha256.in_(chunk)
                )
            )
            or 0
        )
    transcription_count = _count_for_chunks(
        session,
        blob_ids,
        lambda chunk: select(func.count(AudioTranscription.id)).where(
            AudioTranscription.blob_sha256.in_(chunk)
        ),
    )
    extraction_count = _count_for_chunks(
        session,
        blob_ids,
        lambda chunk: select(func.count(AttachmentTextExtraction.id)).where(
            AttachmentTextExtraction.blob_sha256.in_(chunk)
        ),
    )
    nlp_targets = {
        "message": selected["message"],
        "activity": selected["activity"],
        "audio_segment": resources["audio_segment_ids"],
        "attachment_text": resources["attachment_extraction_ids"],
    }
    nlp_count = 0
    for entity_type, identifiers in nlp_targets.items():
        nlp_count += _count_for_chunks(
            session,
            identifiers,
            lambda chunk, entity_type=entity_type: select(func.count(NlpAnnotation.id)).where(
                NlpAnnotation.target_type == entity_type,
                NlpAnnotation.target_id.in_(chunk),
            ),
        )
    active_rollups = (
        int(
            session.scalar(
                select(func.count(NlpHourlyRollup.id)).where(NlpHourlyRollup.active.is_(True))
            )
            or 0
        )
        if any(selected.values())
        else 0
    )
    direct_counts = {
        entity_type: sum(
            reason == "direct"
            for (reason_type, _entity_id), reason in reasons.items()
            if reason_type == entity_type
        )
        for entity_type in sorted(ENTITY_TYPES)
    }
    selected_counts = {
        entity_type: len(selected[cast(EntityType, entity_type)])
        for entity_type in sorted(ENTITY_TYPES)
    }
    return {
        "selected": selected_counts,
        "direct": direct_counts,
        "cascade_count": sum(selected_counts.values()) - sum(direct_counts.values()),
        "record_count": sum(selected_counts.values()),
        "affected_conversation_count": len(resources["conversation_ids"]),
        "attachments": {
            "occurrences": len(selected["attachment"]),
            "occurrence_bytes": occurrence_bytes,
            "unique_blobs": len(blob_ids),
            "unique_blob_bytes": unique_blob_bytes,
            "shared_blobs": len(resources["shared_blob_ids"]),
            "exclusive_blobs": len(resources["exclusive_blob_ids"]),
        },
        "derived": {
            "thumbnails": derivative_count,
            "thumbnail_bytes": derivative_bytes,
            "audio_transcriptions": transcription_count,
            "text_extractions": extraction_count,
            "nlp_annotations": nlp_count,
            "nlp_hourly_rollups_invalidated": active_rollups,
            "vector_windows": len(resources["search_window_ids"]),
        },
        "raw_sources": {
            "files": len(source_file_ids),
            "bytes": raw_source_bytes,
        },
    }


def _build_preview_state(session: Session, selector: DeletionSelector) -> PreviewState:
    selector.validate()
    direct = _direct_selection(session, selector)
    selected, reasons = _expand_selection(session, direct)
    items = _selection_items(session, selected, reasons)
    impact = _impact(session, selected, reasons)
    selection_payload = [
        {
            "entity_type": item.entity_type,
            "entity_id": item.entity_id,
            "reason": item.reason,
            "snapshot_sha256": item.snapshot_sha256,
        }
        for item in items
    ]
    selection_sha256 = _canonical_sha256(selection_payload)
    catalog_revision = _canonical_sha256(
        {
            "selection": selection_payload,
            "impact": impact,
            "schema_version": session.scalar(text("PRAGMA schema_version")),
        }
    )
    return PreviewState(
        items=items,
        impact=impact,
        catalog_revision=catalog_revision,
        selection_sha256=selection_sha256,
    )


def _manifest_path(layout: ArchiveLayout, operation_id: str) -> Path:
    if len(operation_id) != 36 or any(
        character not in "0123456789abcdef-" for character in operation_id
    ):
        raise DeletionError("Invalid deletion operation identifier")
    return layout.trash / "deletions" / operation_id / "manifest.json"


def _canonical_manifest_relative_path(
    layout: ArchiveLayout,
    operation: DeletionOperation,
) -> str:
    expected = _manifest_path(layout, operation.id).relative_to(layout.root).as_posix()
    if operation.manifest_relative_path != expected:
        raise DeletionError("Deletion manifest path does not match its operation")
    return expected


def _write_manifest(
    layout: ArchiveLayout,
    relative_path: str,
    *,
    operation: DeletionOperation,
    items: Sequence[SelectionItem] | Sequence[DeletionItem],
    publication_phase: Literal["preview", "prepared", "published"] | None = None,
) -> None:
    if relative_path != _canonical_manifest_relative_path(layout, operation):
        raise DeletionError("Deletion manifest path does not match its operation")
    try:
        recover_private_publication(layout.root, relative_path)
        prior = read_private_file(layout.root, relative_path, missing_ok=True)
    except SecurePublicationError as exc:
        raise DeletionError("Deletion manifest publication path is unsafe") from exc
    retained_snapshots: dict[tuple[str, str], dict[str, Any]] = {}
    if publication_phase in {"prepared", "published"}:
        if prior is None:
            raise DeletionError("Deletion restore manifest is missing or invalid")
        try:
            prior_payload = json.loads(prior.payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DeletionError("Deletion restore manifest is missing or invalid") from exc
        if not isinstance(prior_payload, dict):
            raise DeletionError("Deletion restore manifest is missing or invalid")
        stored_items = cast(Sequence[DeletionItem], items)
        _validate_deletion_manifest_payload(
            operation,
            stored_items,
            prior_payload,
            validate_restore_state=publication_phase == "published",
        )
        for prior_item in prior_payload["items"]:
            if (
                isinstance(prior_item, dict)
                and prior_item.get("entity_type") == "conversation"
                and isinstance(prior_item.get("snapshot"), dict)
            ):
                retained_snapshots[
                    (str(prior_item.get("entity_type")), str(prior_item.get("entity_id")))
                ] = prior_item["snapshot"]
    restore_state = operation.purge_plan_json.get("logical_restore_state")
    restore_state_sha256 = (
        _canonical_sha256(restore_state) if isinstance(restore_state, dict) else None
    )
    publication = operation.purge_plan_json.get("manifest_publication")
    publication_action = publication.get("action") if isinstance(publication, dict) else None
    payload = {
        "manifest_version": MANIFEST_VERSION,
        "operation_id": operation.id,
        "status": operation.status,
        "selector": operation.selector_json,
        "impact": operation.impact_json,
        "catalog_revision": operation.catalog_revision,
        "selection_sha256": operation.selection_sha256,
        "created_at": _json_value(operation.created_at),
        "committed_at": _json_value(operation.committed_at),
        "restored_at": _json_value(operation.restored_at),
        "restore_state": restore_state,
        "restore_state_sha256": restore_state_sha256,
        "publication": {
            "action": publication_action,
            "phase": publication_phase
            or (str(publication.get("phase")) if isinstance(publication, dict) else "preview"),
        },
        "items": [
            {
                "entity_type": item.entity_type,
                "entity_id": item.entity_id,
                "reason": item.reason,
                "snapshot_sha256": item.snapshot_sha256,
                "previous_deleted_at": _json_value(item.previous_deleted_at),
                "snapshot": (
                    item.snapshot_json
                    if isinstance(item, SelectionItem)
                    else retained_snapshots.get((item.entity_type, item.entity_id))
                ),
            }
            for item in items
        ],
    }
    encoded = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    try:
        publish_private_file(
            layout.root,
            relative_path,
            encoded,
            expected_existing=(prior if publication_phase in {"prepared", "published"} else None),
        )
    except SecurePublicationError as exc:
        raise DeletionError("Deletion manifest publication failed safely") from exc


def preview_deletion(
    layout: ArchiveLayout,
    engine: Engine,
    selector: DeletionSelector,
) -> DeletionPreview:
    # ``ensure_runtime_directories`` creates every derived workspace tree. Serialize it with
    # purge staging so a preview cannot recreate a just-staged tree before discovering that a
    # destructive operation owns the archive.
    with deletion_lock(layout):
        return _preview_deletion_locked(layout, engine, selector)


def _preview_deletion_locked(
    layout: ArchiveLayout,
    engine: Engine,
    selector: DeletionSelector,
) -> DeletionPreview:
    layout.ensure_runtime_directories()
    with Session(engine) as session:
        state = _build_preview_state(session, selector)
    operation_id = str(uuid.uuid4())
    token = secrets.token_urlsafe(32)
    relative_manifest = _manifest_path(layout, operation_id).relative_to(layout.root).as_posix()
    with Session(engine) as session, session.begin():
        operation = DeletionOperation(
            id=operation_id,
            status="previewed",
            selector_json=selector.to_json(),
            impact_json=state.impact,
            catalog_revision=state.catalog_revision,
            selection_sha256=state.selection_sha256,
            commit_token_hash=_token_hash(token),
            manifest_relative_path=relative_manifest,
            item_count=len(state.items),
        )
        session.add(operation)
        session.flush()
        session.add_all(
            DeletionItem(
                operation_id=operation_id,
                entity_type=item.entity_type,
                entity_id=item.entity_id,
                reason=item.reason,
                snapshot_sha256=item.snapshot_sha256,
                previous_deleted_at=item.previous_deleted_at,
            )
            for item in state.items
        )
    with Session(engine) as session:
        published_operation = session.get(DeletionOperation, operation_id)
        if published_operation is None or published_operation.manifest_relative_path is None:
            raise DeletionError("Deletion preview disappeared before manifest publication")
        _write_manifest(
            layout,
            published_operation.manifest_relative_path,
            operation=published_operation,
            items=state.items,
        )
    return DeletionPreview(
        operation_id=operation_id,
        commit_token=token,
        catalog_revision=state.catalog_revision,
        selection_sha256=state.selection_sha256,
        impact=state.impact,
        manifest=str(layout.root / relative_manifest),
    )


def _stored_items(session: Session, operation_id: str) -> list[DeletionItem]:
    return list(
        session.scalars(
            select(DeletionItem)
            .where(DeletionItem.operation_id == operation_id)
            .order_by(DeletionItem.entity_type, DeletionItem.entity_id)
        )
    )


def _publication_result(
    operation: DeletionOperation,
    *,
    action: Literal["commit", "restore"],
) -> MutationResult:
    publication = operation.purge_plan_json.get("manifest_publication")
    if (
        not isinstance(publication, dict)
        or publication.get("action") != action
        or not isinstance(publication.get("audit_id"), str)
        or not isinstance(publication.get("affected"), dict)
        or any(
            not isinstance(key, str)
            or not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            for key, value in publication["affected"].items()
        )
    ):
        raise DeletionError(f"Deletion {action} publication state is missing or invalid")
    return MutationResult(
        operation.id,
        "committed" if action == "commit" else "restored",
        cast(dict[str, int], publication["affected"]),
        publication["audit_id"],
    )


def _finalize_manifest_publication(
    layout: ArchiveLayout,
    engine: Engine,
    operation_id: str,
    *,
    action: Literal["commit", "restore"],
) -> MutationResult:
    """Idempotently publish a mutation manifest after its SQLite transaction commits."""

    expected_status = "committed" if action == "commit" else "restored"
    with Session(engine) as session:
        operation = session.get(DeletionOperation, operation_id)
        if operation is None or operation.status != expected_status:
            raise DeletionError(f"Deletion {action} did not reach its durable catalog state")
        result = _publication_result(operation, action=action)
        items = _stored_items(session, operation_id)
        if operation.manifest_relative_path is None:
            raise DeletionError(f"Deletion {action} lost its manifest path")
        _validate_deletion_manifest_items(layout, operation, items)
        _write_manifest(
            layout,
            operation.manifest_relative_path,
            operation=operation,
            items=items,
            publication_phase="published",
        )
    with immediate_session(engine) as session:
        operation = session.get(DeletionOperation, operation_id)
        if operation is None or operation.status != expected_status:
            raise StaleDeletionPreviewError(
                f"Deletion {action} changed during manifest publication"
            )
        publication = operation.purge_plan_json.get("manifest_publication")
        if not isinstance(publication, dict) or publication.get("action") != action:
            raise DeletionError(f"Deletion {action} publication state changed unexpectedly")
        operation.purge_plan_json = {
            **operation.purge_plan_json,
            "manifest_publication": {**publication, "phase": "published"},
        }
    return result


def _verify_token(expected_hash: str | None, candidate: str) -> None:
    if expected_hash is None or not hmac.compare_digest(expected_hash, _token_hash(candidate)):
        raise InvalidDeletionTokenError("Invalid or expired deletion confirmation token")


def _reconcile_conversations(session: Session, conversation_ids: Iterable[str]) -> None:
    for conversation_id in sorted(set(conversation_ids)):
        conversation = session.get(Conversation, conversation_id)
        if conversation is None:
            continue
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
        for segment in session.scalars(
            select(ConversationSegment).where(
                ConversationSegment.conversation_id == conversation_id
            )
        ):
            segment.first_activity_ms, segment.last_activity_ms = session.execute(
                select(func.min(Message.timestamp_ms), func.max(Message.timestamp_ms)).where(
                    Message.segment_id == segment.id,
                    Message.deleted_at.is_(None),
                )
            ).one()


def _audit(
    session: Session,
    operation_id: str,
    *,
    action: str,
    outcome: str,
    scope: str | None,
    detail: dict[str, Any],
) -> str:
    audit_id = str(uuid.uuid4())
    session.add(
        DeletionAudit(
            id=audit_id,
            operation_id=operation_id,
            action=action,
            scope=scope,
            outcome=outcome,
            detail_json=detail,
        )
    )
    return audit_id


def _items_as_selected(items: Sequence[DeletionItem]) -> dict[EntityType, set[str]]:
    selected: dict[EntityType, set[str]] = {
        "message": set(),
        "activity": set(),
        "conversation": set(),
        "attachment": set(),
    }
    for item in items:
        if item.entity_type not in ENTITY_TYPES:
            raise DeletionError(f"Operation contains invalid entity type: {item.entity_type}")
        selected[cast(EntityType, item.entity_type)].add(item.entity_id)
    return selected


def _update_deleted_at(
    session: Session,
    selected: dict[EntityType, set[str]],
    value: datetime | None,
    *,
    require_null: bool,
) -> dict[str, int]:
    affected: dict[str, int] = {}
    for entity_type in ("attachment", "message", "activity", "conversation"):
        model = _ENTITY_MODELS[entity_type]
        count = 0
        for chunk in _chunks(sorted(selected[entity_type])):
            statement = update(model).where(model.id.in_(chunk))
            if require_null:
                statement = statement.where(model.deleted_at.is_(None))
            result = session.execute(statement.values(deleted_at=value))
            count += _result_rowcount(result)
        affected[entity_type] = count
    return affected


def _nlp_target_ids(
    selected: dict[EntityType, set[str]], resources: dict[str, set[str]]
) -> dict[str, set[str]]:
    """Return every annotation target made unavailable by this selection.

    Blob-derived targets are included only for exclusive blobs.  A shared attachment occurrence
    must not hide or destroy the transcription/extraction annotations still used by a retained
    occurrence of the same bytes.
    """

    return {
        "message": selected["message"],
        "activity": selected["activity"],
        "audio_segment": resources["audio_segment_ids"],
        "attachment_text": resources["attachment_extraction_ids"],
    }


def _physical_nlp_target_ids(
    selected: dict[EntityType, set[str]], resources: dict[str, set[str]]
) -> dict[str, set[str]]:
    """Return annotation targets that no separately restorable occurrence still needs."""

    return {
        "message": selected["message"],
        "activity": selected["activity"],
        "audio_segment": resources["physical_audio_segment_ids"],
        "attachment_text": resources["physical_attachment_extraction_ids"],
    }


def _passthrough_nlp_target_ids(resources: dict[str, set[str]]) -> dict[str, set[str]]:
    """Capture shared-derived annotations that a later final owner may quarantine.

    These targets remain visible for this operation because another live occurrence still owns
    the same blob. Capturing their exact active state lets a provenance-bound later quarantine
    delegate restoration back to this still-recoverable occurrence.
    """

    return {
        "audio_segment": resources["restore_audio_segment_ids"] - resources["audio_segment_ids"],
        "attachment_text": resources["restore_attachment_extraction_ids"]
        - resources["attachment_extraction_ids"],
    }


def _deactivate_nlp_annotations(
    session: Session,
    selected: dict[EntityType, set[str]],
    resources: dict[str, set[str]],
) -> int:
    count = 0
    for target_type, target_ids in _nlp_target_ids(selected, resources).items():
        for chunk in _chunks(sorted(target_ids)):
            result = session.execute(
                update(NlpAnnotation)
                .where(
                    NlpAnnotation.target_type == target_type,
                    NlpAnnotation.target_id.in_(chunk),
                    NlpAnnotation.active.is_(True),
                )
                .values(active=False)
            )
            count += _result_rowcount(result)
    return count


def _restore_state_annotation_ids(
    restore_state: object,
    *,
    passthrough: bool,
) -> set[str]:
    if not isinstance(restore_state, dict):
        raise DeletionError("Deletion operation contains invalid logical restore state")
    raw_entries = restore_state.get("nlp_annotations")
    if not isinstance(raw_entries, list):
        raise DeletionError("Deletion operation contains invalid NLP restore state")
    identifiers: set[str] = set()
    for raw_entry in raw_entries:
        if (
            not isinstance(raw_entry, dict)
            or not isinstance(raw_entry.get("id"), str)
            or not isinstance(raw_entry.get("passthrough", False), bool)
        ):
            raise DeletionError("Deletion operation contains invalid NLP restore state")
        if bool(raw_entry.get("passthrough", False)) is passthrough:
            identifiers.add(raw_entry["id"])
    return identifiers


def _record_logical_restore_delegations(
    session: Session,
    *,
    deactivating_operation_id: str,
    restore_state: dict[str, Any],
) -> int:
    """Bind a final-owner deactivation to earlier recoverable shared-blob quarantines."""

    deactivated_ids = _restore_state_annotation_ids(restore_state, passthrough=False)
    if not deactivated_ids:
        return 0
    claimed = 0
    prior_operations = session.scalars(
        select(DeletionOperation)
        .where(
            DeletionOperation.status == "committed",
            DeletionOperation.id != deactivating_operation_id,
        )
        .order_by(DeletionOperation.created_at, DeletionOperation.id)
    )
    for prior in prior_operations:
        prior_state = prior.purge_plan_json.get("logical_restore_state")
        passthrough_ids = _restore_state_annotation_ids(prior_state, passthrough=True)
        delegated_ids = sorted(deactivated_ids & passthrough_ids)
        if not delegated_ids:
            continue
        raw_claims = prior.purge_plan_json.get("logical_restore_delegations", [])
        if not isinstance(raw_claims, list) or any(
            not isinstance(value, dict)
            or not isinstance(value.get("annotation_id"), str)
            or not isinstance(value.get("deactivated_by"), str)
            for value in raw_claims
        ):
            raise DeletionError("Deletion operation contains invalid logical restore delegations")
        coordinates = {(value["annotation_id"], value["deactivated_by"]) for value in raw_claims}
        new_claims = list(raw_claims)
        for annotation_id in delegated_ids:
            coordinate = (annotation_id, deactivating_operation_id)
            if coordinate in coordinates:
                continue
            new_claims.append(
                {
                    "annotation_id": annotation_id,
                    "deactivated_by": deactivating_operation_id,
                }
            )
            coordinates.add(coordinate)
            claimed += 1
        prior.purge_plan_json = {
            **prior.purge_plan_json,
            "logical_restore_delegations": sorted(
                new_claims,
                key=lambda value: (value["annotation_id"], value["deactivated_by"]),
            ),
        }
    return claimed


def _validated_logical_restore_delegations(
    session: Session,
    operation: DeletionOperation,
) -> set[str]:
    raw_claims = operation.purge_plan_json.get("logical_restore_delegations", [])
    if not isinstance(raw_claims, list):
        raise DeletionError("Deletion operation contains invalid logical restore delegations")
    claims: dict[str, set[str]] = {}
    for raw_claim in raw_claims:
        if (
            not isinstance(raw_claim, dict)
            or not isinstance(raw_claim.get("annotation_id"), str)
            or not isinstance(raw_claim.get("deactivated_by"), str)
            or raw_claim["deactivated_by"] == operation.id
        ):
            raise DeletionError("Deletion operation contains invalid logical restore delegations")
        claims.setdefault(raw_claim["annotation_id"], set()).add(raw_claim["deactivated_by"])
    source_ids = {source_id for values in claims.values() for source_id in values}
    source_statuses = {
        source_id: status
        for source_id, status in session.execute(
            select(DeletionOperation.id, DeletionOperation.status).where(
                DeletionOperation.id.in_(source_ids)
            )
        )
    }
    if set(source_statuses) != source_ids:
        raise DeletionError("Deletion restore delegation provenance is incomplete")
    allowed_statuses = {
        "committed",
        "cleanup_pending",
        "normalized_purged",
        "purged",
        "restored",
    }
    if any(status not in allowed_statuses for status in source_statuses.values()):
        raise DeletionError("Deletion restore delegation provenance is invalid")
    if any(
        source_statuses[source_id] == "committed"
        for deactivators in claims.values()
        for source_id in deactivators
    ):
        raise DeletionError("A later shared-data quarantine must be restored or hard-purged first")
    purge_statuses = {"cleanup_pending", "normalized_purged", "purged"}
    return {
        annotation_id
        for annotation_id, deactivators in claims.items()
        if any(source_statuses[source_id] in purge_statuses for source_id in deactivators)
    }


def _mark_other_restore_states_for_global_rebuild(
    session: Session,
    *,
    invalidating_operation_id: str,
) -> int:
    """Record that a later hard purge made earlier global snapshots unsafe to republish."""

    updated = 0
    operations = session.scalars(
        select(DeletionOperation).where(
            DeletionOperation.status == "committed",
            DeletionOperation.id != invalidating_operation_id,
        )
    )
    for operation in operations:
        raw = operation.purge_plan_json.get("logical_restore_compatibility")
        if raw is None:
            invalidated_by: list[str] = []
        elif (
            isinstance(raw, dict)
            and raw.get("global_derived") == "rebuild_required"
            and isinstance(raw.get("invalidated_by"), list)
            and all(isinstance(value, str) for value in raw["invalidated_by"])
        ):
            invalidated_by = list(raw["invalidated_by"])
        else:
            raise DeletionError("Deletion operation contains invalid restore compatibility state")
        if invalidating_operation_id in invalidated_by:
            continue
        invalidated_by.append(invalidating_operation_id)
        operation.purge_plan_json = {
            **operation.purge_plan_json,
            "logical_restore_compatibility": {
                "global_derived": "rebuild_required",
                "invalidated_by": sorted(set(invalidated_by)),
            },
        }
        updated += 1
    return updated


def _global_derived_restore_compatibility(
    session: Session,
    operation: DeletionOperation,
) -> tuple[bool, str | None, int]:
    raw = operation.purge_plan_json.get("logical_restore_compatibility")
    if raw is None:
        return True, None, 0
    if (
        not isinstance(raw, dict)
        or raw.get("global_derived") != "rebuild_required"
        or not isinstance(raw.get("invalidated_by"), list)
        or not raw["invalidated_by"]
        or any(not isinstance(value, str) for value in raw["invalidated_by"])
    ):
        raise DeletionError("Deletion operation contains invalid restore compatibility state")
    invalidators = set(raw["invalidated_by"])
    statuses = {
        operation_id: status
        for operation_id, status in session.execute(
            select(DeletionOperation.id, DeletionOperation.status).where(
                DeletionOperation.id.in_(invalidators)
            )
        )
    }
    allowed_statuses = {
        "committed",
        "cleanup_pending",
        "normalized_purged",
        "purged",
        "restored",
    }
    if set(statuses) != invalidators or any(
        status not in allowed_statuses for status in statuses.values()
    ):
        raise DeletionError("Deletion restore compatibility provenance is invalid")
    incompatible_statuses = [status for status in statuses.values() if status != "restored"]
    if not incompatible_statuses:
        return True, None, 0
    reason = (
        "intervening_hard_purge"
        if any(
            status in {"cleanup_pending", "normalized_purged", "purged"}
            for status in incompatible_statuses
        )
        else "later_quarantine"
    )
    return False, reason, len(incompatible_statuses)


def _invalidate_nlp_hourly_rollups(session: Session) -> int:
    """Invalidate aggregate rows; the explicit NLP rebuild recreates authoritative rollups."""

    return _result_rowcount(
        session.execute(
            update(NlpHourlyRollup).where(NlpHourlyRollup.active.is_(True)).values(active=False)
        )
    )


def _invalidate_conflict_aggregate_rows(session: Session, *, reason: str) -> int:
    runs = list(
        session.scalars(
            select(ConflictAggregateRun).where(
                ConflictAggregateRun.status.in_(("building", "ready"))
            )
        )
    )
    run_ids = [row.id for row in runs]
    now = datetime.now(UTC)
    for row in runs:
        row.active = False
        row.status = "stale"
        row.error_text = reason
        row.provenance_json = {
            **(row.provenance_json or {}),
            "invalidated_at": now.isoformat(),
            "invalidation_reason": reason,
        }
    if run_ids:
        session.execute(
            update(ConflictWindowScore)
            .where(ConflictWindowScore.run_id.in_(run_ids))
            .values(active=False)
        )
        session.execute(
            update(ConflictHourlyRollup)
            .where(ConflictHourlyRollup.run_id.in_(run_ids))
            .values(active=False)
        )
    return len(run_ids)


def _capture_logical_restore_state(
    session: Session,
    selected: dict[EntityType, set[str]],
    resources: dict[str, set[str]],
) -> dict[str, Any]:
    """Capture exactly the published derived state invalidated by logical quarantine."""

    annotation_ids: set[str] = set()
    for target_type, target_ids in _nlp_target_ids(selected, resources).items():
        for chunk in _chunks(sorted(target_ids)):
            annotation_ids.update(
                session.scalars(
                    select(NlpAnnotation.id).where(
                        NlpAnnotation.target_type == target_type,
                        NlpAnnotation.target_id.in_(chunk),
                        NlpAnnotation.active.is_(True),
                    )
                )
            )
    conflict_runs = list(
        session.scalars(
            select(ConflictAggregateRun).where(
                ConflictAggregateRun.status.in_(("building", "ready"))
            )
        )
    )
    conflict_run_ids = [row.id for row in conflict_runs]
    topic_jobs = list(session.scalars(select(Job).where(Job.kind == "topic_clusters")))
    return {
        "version": 1,
        "search_window_ids": sorted(resources["search_window_ids"]),
        "nlp_annotation_ids": sorted(annotation_ids),
        "nlp_hourly_rollup_ids": sorted(
            session.scalars(select(NlpHourlyRollup.id).where(NlpHourlyRollup.active.is_(True)))
        ),
        "conflict_runs": [
            {
                "id": row.id,
                "active": row.active,
                "status": row.status,
                "error_text": row.error_text,
                "provenance_json": row.provenance_json,
            }
            for row in conflict_runs
        ],
        "conflict_score_ids": (
            sorted(
                session.scalars(
                    select(ConflictWindowScore.id).where(
                        ConflictWindowScore.run_id.in_(conflict_run_ids),
                        ConflictWindowScore.active.is_(True),
                    )
                )
            )
            if conflict_run_ids
            else []
        ),
        "conflict_rollup_ids": (
            sorted(
                session.scalars(
                    select(ConflictHourlyRollup.id).where(
                        ConflictHourlyRollup.run_id.in_(conflict_run_ids),
                        ConflictHourlyRollup.active.is_(True),
                    )
                )
            )
            if conflict_run_ids
            else []
        ),
        "topic_jobs": [
            {
                "id": row.id,
                "status": row.status,
                "progress_current": row.progress_current,
                "progress_total": row.progress_total,
                "stage": row.stage,
                "checkpoint_json": row.checkpoint_json,
                "counters_json": row.counters_json,
                "error_text": row.error_text,
                "started_at": _json_value(row.started_at),
                "finished_at": _json_value(row.finished_at),
            }
            for row in topic_jobs
        ],
    }


def _parse_optional_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise DeletionError("Logical restore state contains an invalid timestamp")
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise DeletionError("Logical restore state contains an invalid timestamp") from exc


def _restore_logical_derived_state(
    session: Session,
    operation: DeletionOperation,
) -> dict[str, int]:
    raw_state = operation.purge_plan_json.get("logical_restore_state")
    if not isinstance(raw_state, dict) or raw_state.get("version") != 1:
        raise DeletionError("Deletion operation lacks compatible logical restore state")
    list_fields = (
        "search_window_ids",
        "nlp_annotation_ids",
        "nlp_hourly_rollup_ids",
        "conflict_score_ids",
        "conflict_rollup_ids",
    )
    if any(
        not isinstance(raw_state.get(field), list)
        or any(not isinstance(value, str) for value in raw_state[field])
        for field in list_fields
    ):
        raise DeletionError("Deletion operation contains invalid logical restore state")

    search_window_ids = set(raw_state["search_window_ids"])
    existing_windows = _scalar_ids_for_chunks(
        session,
        search_window_ids,
        lambda chunk: select(SearchWindow.id).where(SearchWindow.id.in_(chunk)),
    )
    if existing_windows != search_window_ids:
        raise DeletionError(
            "Search windows changed while rows were quarantined; rebuild search before restore"
        )

    global_state_present = any(
        raw_state.get(field) for field in ("nlp_hourly_rollup_ids", "conflict_runs", "topic_jobs")
    )
    if global_state_present:
        other_committed = session.scalar(
            select(DeletionOperation.id)
            .where(
                DeletionOperation.id != operation.id,
                DeletionOperation.status == "committed",
            )
            .limit(1)
        )
        if other_committed is not None:
            raise DeletionError(
                "Restore derived state is blocked by another committed quarantine; "
                "restore newer quarantines first"
            )

    affected: dict[str, int] = {}
    annotation_ids = set(raw_state["nlp_annotation_ids"])
    existing_annotations = _scalar_ids_for_chunks(
        session,
        annotation_ids,
        lambda chunk: select(NlpAnnotation.id).where(NlpAnnotation.id.in_(chunk)),
    )
    if existing_annotations != annotation_ids:
        raise DeletionError("NLP annotations changed while rows were quarantined")
    affected["nlp_annotations_reactivated"] = 0
    for chunk in _chunks(sorted(annotation_ids)):
        affected["nlp_annotations_reactivated"] += _result_rowcount(
            session.execute(
                update(NlpAnnotation)
                .where(NlpAnnotation.id.in_(chunk), NlpAnnotation.active.is_(False))
                .values(active=True)
            )
        )

    rollup_ids = set(raw_state["nlp_hourly_rollup_ids"])
    existing_rollups = _scalar_ids_for_chunks(
        session,
        rollup_ids,
        lambda chunk: select(NlpHourlyRollup.id).where(NlpHourlyRollup.id.in_(chunk)),
    )
    if existing_rollups != rollup_ids:
        raise DeletionError("NLP analytics changed while rows were quarantined")
    affected["nlp_hourly_rollups_reactivated"] = 0
    for chunk in _chunks(sorted(rollup_ids)):
        affected["nlp_hourly_rollups_reactivated"] += _result_rowcount(
            session.execute(
                update(NlpHourlyRollup)
                .where(NlpHourlyRollup.id.in_(chunk), NlpHourlyRollup.active.is_(False))
                .values(active=True)
            )
        )

    conflict_runs = raw_state.get("conflict_runs")
    topic_jobs = raw_state.get("topic_jobs")
    if not isinstance(conflict_runs, list) or not isinstance(topic_jobs, list):
        raise DeletionError("Deletion operation contains invalid logical restore state")
    affected["conflict_aggregate_runs_restored"] = 0
    for snapshot in conflict_runs:
        if not isinstance(snapshot, dict) or not isinstance(snapshot.get("id"), str):
            raise DeletionError("Deletion operation contains invalid conflict restore state")
        conflict_run = session.get(ConflictAggregateRun, snapshot["id"])
        if conflict_run is None:
            raise DeletionError("Conflict analytics changed while rows were quarantined")
        conflict_run.active = bool(snapshot.get("active"))
        conflict_run.status = str(snapshot.get("status"))
        conflict_run.error_text = snapshot.get("error_text")
        provenance = snapshot.get("provenance_json")
        if not isinstance(provenance, dict):
            raise DeletionError("Deletion operation contains invalid conflict restore state")
        conflict_run.provenance_json = provenance
        affected["conflict_aggregate_runs_restored"] += 1
    for model, id_field, affected_key in (
        (ConflictWindowScore, "conflict_score_ids", "conflict_window_scores_reactivated"),
        (ConflictHourlyRollup, "conflict_rollup_ids", "conflict_hourly_rollups_reactivated"),
    ):
        identifiers = set(raw_state[id_field])
        existing = _scalar_ids_for_chunks(
            session,
            identifiers,
            lambda chunk, model=model: select(model.id).where(model.id.in_(chunk)),
        )
        if existing != identifiers:
            raise DeletionError("Conflict analytics changed while rows were quarantined")
        affected[affected_key] = 0
        for chunk in _chunks(sorted(identifiers)):
            affected[affected_key] += _result_rowcount(
                session.execute(update(model).where(model.id.in_(chunk)).values(active=True))
            )

    affected["topic_cluster_jobs_restored"] = 0
    for snapshot in topic_jobs:
        if not isinstance(snapshot, dict) or not isinstance(snapshot.get("id"), str):
            raise DeletionError("Deletion operation contains invalid topic restore state")
        topic_job = session.get(Job, snapshot["id"])
        if topic_job is None or topic_job.kind != "topic_clusters":
            raise DeletionError("Topic analytics changed while rows were quarantined")
        checkpoint = snapshot.get("checkpoint_json")
        counters = snapshot.get("counters_json")
        if not isinstance(checkpoint, dict) or not isinstance(counters, dict):
            raise DeletionError("Deletion operation contains invalid topic restore state")
        topic_job.status = str(snapshot.get("status"))
        topic_job.progress_current = int(snapshot.get("progress_current", 0))
        progress_total = snapshot.get("progress_total")
        topic_job.progress_total = int(progress_total) if progress_total is not None else None
        topic_job.stage = snapshot.get("stage")
        topic_job.checkpoint_json = checkpoint
        topic_job.counters_json = counters
        topic_job.error_text = snapshot.get("error_text")
        topic_job.started_at = _parse_optional_datetime(snapshot.get("started_at"))
        topic_job.finished_at = _parse_optional_datetime(snapshot.get("finished_at"))
        affected["topic_cluster_jobs_restored"] += 1
    return affected


def commit_deletion(
    layout: ArchiveLayout,
    engine: Engine,
    operation_id: str,
    *,
    commit_token: str,
    catalog_revision: str,
) -> MutationResult:
    with (
        deletion_lock(layout),
        purge_source_publisher_locks(layout) as _publisher_leases,
        purge_worker_locks(layout),
        vector_store_lock(layout),
        analytics_cache_lock(layout, exclusive=True),
    ):
        # A previous hard-purge process may have died after moving live files into durable
        # staging but before SQLite committed. Restore that exact filesystem state before any
        # new quarantine mutation or manifest replay observes the workspace.
        _recover_precommit_purge_staging_locked(layout, engine)
        replay_committed = False
        with Session(engine) as replay_session:
            replay = replay_session.get(DeletionOperation, operation_id)
            if replay is not None and replay.status == "committed":
                _verify_token(replay.commit_token_hash, commit_token)
                if not hmac.compare_digest(replay.catalog_revision, catalog_revision):
                    raise StaleDeletionPreviewError(
                        "Deletion catalog revision does not match preview"
                    )
                replay_committed = True
        if replay_committed:
            return _finalize_manifest_publication(layout, engine, operation_id, action="commit")
        with immediate_session(engine) as session:
            operation = session.get(DeletionOperation, operation_id)
            if operation is None:
                raise DeletionError("Deletion operation not found")
            if operation.status != "previewed":
                raise DeletionError(f"Deletion operation is {operation.status}, not previewed")
            _verify_token(operation.commit_token_hash, commit_token)
            if not hmac.compare_digest(operation.catalog_revision, catalog_revision):
                raise StaleDeletionPreviewError("Deletion catalog revision does not match preview")
            selector = DeletionSelector.from_json(operation.selector_json)
            current = _build_preview_state(session, selector)
            stored_items = _stored_items(session, operation_id)
            expected_items = [
                (item.entity_type, item.entity_id, item.reason, item.snapshot_sha256)
                for item in stored_items
            ]
            current_items = [
                (item.entity_type, item.entity_id, item.reason, item.snapshot_sha256)
                for item in current.items
            ]
            if (
                current.catalog_revision != operation.catalog_revision
                or current.selection_sha256 != operation.selection_sha256
                or current_items != expected_items
            ):
                raise StaleDeletionPreviewError(
                    "Catalog contents changed after preview; create a new deletion preview"
                )
            # Do not quarantine rows unless the independently stored recovery manifest still
            # agrees with the catalog. Conversation snapshots are the only restore inputs that
            # cannot be reconstructed solely from DeletionItem rows, so validate their digests
            # before making the catalog unavailable as well.
            _validate_deletion_manifest_items(layout, operation, stored_items)
            _restore_conversation_snapshots(layout, operation, stored_items)
            selected = _items_as_selected(stored_items)
            resources = _selection_resources(session, selected)
            restore_state = capture_derived_restore_state(
                session,
                nlp_targets=_nlp_target_ids(selected, resources),
                passthrough_nlp_targets=_passthrough_nlp_target_ids(resources),
                search_window_ids=resources["search_window_ids"],
            )
            committed_at = datetime.now(UTC)
            affected = _update_deleted_at(session, selected, committed_at, require_null=True)
            if sum(affected.values()) != operation.item_count:
                raise StaleDeletionPreviewError(
                    "One or more selected records changed before the deletion commit"
                )
            _reconcile_conversations(session, resources["conversation_ids"])
            affected["nlp_annotations_deactivated"] = _deactivate_nlp_annotations(
                session, selected, resources
            )
            affected["nlp_hourly_rollups_invalidated"] = (
                _invalidate_nlp_hourly_rollups(session) if any(selected.values()) else 0
            )
            affected["conflict_aggregate_runs_invalidated"] = (
                _invalidate_conflict_aggregate_rows(
                    session,
                    reason="Archive rows were logically deleted; rebuild conflict aggregates",
                )
                if any(selected.values())
                else 0
            )
            affected["topic_cluster_jobs_invalidated"] = (
                _invalidate_topic_cluster_jobs(
                    session,
                    reason="Archive rows were logically deleted; rebuild topic clusters",
                )
                if any(selected.values())
                else 0
            )
            session.flush()
            restore_state = seal_quarantined_derived_state(session, restore_state)
            affected["logical_restore_delegations"] = _record_logical_restore_delegations(
                session,
                deactivating_operation_id=operation_id,
                restore_state=restore_state,
            )
            affected["prior_restore_states_requiring_global_rebuild"] = (
                _mark_other_restore_states_for_global_rebuild(
                    session,
                    invalidating_operation_id=operation_id,
                )
            )
            operation.status = "committed"
            operation.committed_at = committed_at
            audit_id = _audit(
                session,
                operation_id,
                action="commit",
                outcome="completed",
                scope="logical_quarantine",
                detail={"affected": affected, "catalog_revision": catalog_revision},
            )
            operation.purge_plan_json = {
                **operation.purge_plan_json,
                "logical_restore_state": restore_state,
                "logical_restore_state_sha256": _canonical_sha256(restore_state),
                "manifest_publication": {
                    "action": "commit",
                    "phase": "prepared",
                    "affected": affected,
                    "audit_id": audit_id,
                },
            }
            session.flush()
            if operation.manifest_relative_path is None:
                raise DeletionError("Committed deletion operation lost its manifest path")
            _write_manifest(
                layout,
                operation.manifest_relative_path,
                operation=operation,
                items=stored_items,
                publication_phase="prepared",
            )
        return _finalize_manifest_publication(layout, engine, operation_id, action="commit")


def _restore_conversation_snapshots(
    layout: ArchiveLayout,
    operation: DeletionOperation,
    items: Sequence[DeletionItem],
) -> dict[str, dict[str, Any]]:
    if operation.manifest_relative_path is None:
        raise DeletionError("Restore manifest path is missing")
    relative_path = _canonical_manifest_relative_path(layout, operation)
    _snapshot, payload = _read_deletion_manifest(
        layout,
        relative_path,
        error_message="Restore manifest is missing or invalid",
    )
    _validate_deletion_manifest_payload(operation, items, payload)
    manifest_items = payload.get("items")
    assert isinstance(manifest_items, list)
    by_coordinate = {
        (item.get("entity_type"), item.get("entity_id")): item
        for item in manifest_items
        if isinstance(item, dict)
    }
    snapshots: dict[str, dict[str, Any]] = {}
    for item in items:
        if item.entity_type != "conversation":
            continue
        manifest_item = by_coordinate.get((item.entity_type, item.entity_id))
        snapshot = manifest_item.get("snapshot") if manifest_item else None
        if not isinstance(snapshot, dict):
            raise DeletionError(
                f"Restore manifest lacks the original conversation row {item.entity_id}"
            )
        if _canonical_sha256(snapshot) != item.snapshot_sha256:
            raise DeletionError(
                f"Restore manifest snapshot failed integrity validation for {item.entity_id}"
            )
        snapshots[item.entity_id] = snapshot
    return snapshots


def _read_deletion_manifest(
    layout: ArchiveLayout,
    relative_path: str,
    *,
    error_message: str,
) -> tuple[PrivateFileSnapshot, dict[str, Any]]:
    try:
        snapshot = read_private_file(layout.root, relative_path)
    except SecurePublicationError as exc:
        raise DeletionError(error_message) from exc
    if snapshot is None:  # pragma: no cover - `missing_ok=False` cannot return None.
        raise DeletionError(error_message)
    try:
        payload = json.loads(snapshot.payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DeletionError(error_message) from exc
    if not isinstance(payload, dict):
        raise DeletionError(error_message)
    return snapshot, payload


def _validate_deletion_manifest_payload(
    operation: DeletionOperation,
    items: Sequence[DeletionItem],
    payload: dict[str, Any],
    *,
    validate_restore_state: bool = True,
) -> None:
    if (
        payload.get("manifest_version") != MANIFEST_VERSION
        or payload.get("operation_id") != operation.id
        or payload.get("selection_sha256") != operation.selection_sha256
    ):
        raise DeletionError("Deletion restore manifest identity is invalid")
    manifest_restore_state = payload.get("restore_state")
    manifest_restore_sha256 = payload.get("restore_state_sha256")
    catalog_restore_state = operation.purge_plan_json.get("logical_restore_state")
    catalog_restore_sha256 = operation.purge_plan_json.get("logical_restore_state_sha256")
    if (
        validate_restore_state
        and operation.status in {"committed", "restored"}
        and (catalog_restore_state is not None or manifest_restore_state is not None)
        and (
            not isinstance(catalog_restore_state, dict)
            or not isinstance(manifest_restore_state, dict)
            or not isinstance(catalog_restore_sha256, str)
            or manifest_restore_sha256 != catalog_restore_sha256
            or _canonical_sha256(catalog_restore_state) != catalog_restore_sha256
            or _canonical_sha256(manifest_restore_state) != catalog_restore_sha256
        )
    ):
        raise DeletionError("Deletion restore state failed integrity validation")
    manifest_items = payload.get("items")
    if not isinstance(manifest_items, list):
        raise DeletionError("Deletion restore manifest item list is invalid")
    actual: list[tuple[Any, ...]] = []
    expected_by_coordinate = {(item.entity_type, item.entity_id): item for item in items}
    for manifest_item in manifest_items:
        if not isinstance(manifest_item, dict):
            raise DeletionError("Deletion restore manifest contains an invalid item")
        entity_type = manifest_item.get("entity_type")
        entity_id = manifest_item.get("entity_id")
        item = (
            expected_by_coordinate.get((entity_type, entity_id))
            if isinstance(entity_type, str) and isinstance(entity_id, str)
            else None
        )
        if item is not None and item.entity_type == "conversation":
            snapshot = manifest_item.get("snapshot")
            if (
                not isinstance(snapshot, dict)
                or _canonical_sha256(snapshot) != item.snapshot_sha256
            ):
                raise DeletionError("Deletion restore conversation snapshot is invalid")
        elif item is not None and manifest_item.get("snapshot") is not None:
            raise DeletionError("Deletion restore manifest contains an unexpected row snapshot")
        actual.append(
            (
                manifest_item.get("entity_type"),
                manifest_item.get("entity_id"),
                manifest_item.get("reason"),
                manifest_item.get("snapshot_sha256"),
                manifest_item.get("previous_deleted_at"),
            )
        )
    expected = [
        (
            item.entity_type,
            item.entity_id,
            item.reason,
            item.snapshot_sha256,
            _json_value(item.previous_deleted_at),
        )
        for item in items
    ]
    if actual != expected:
        raise DeletionError("Deletion restore manifest does not match catalog items")


def _validate_deletion_manifest_items(
    layout: ArchiveLayout,
    operation: DeletionOperation,
    items: Sequence[DeletionItem],
) -> None:
    if operation.manifest_relative_path is None:
        raise DeletionError("Deletion restore manifest path is missing")
    relative_path = _canonical_manifest_relative_path(layout, operation)
    _snapshot, payload = _read_deletion_manifest(
        layout,
        relative_path,
        error_message="Deletion restore manifest is missing or invalid",
    )
    _validate_deletion_manifest_payload(operation, items, payload)


def _without_volatile_update_time(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in snapshot.items() if key != "updated_at"}


def restore_deletion(
    layout: ArchiveLayout,
    engine: Engine,
    operation_id: str,
    *,
    confirmation: str,
) -> MutationResult:
    if confirmation != operation_id:
        raise InvalidDeletionTokenError("Restore confirmation must equal the operation ID")
    with (
        deletion_lock(layout),
        purge_source_publisher_locks(layout) as _publisher_leases,
        purge_worker_locks(layout),
        vector_store_lock(layout),
        analytics_cache_lock(layout, exclusive=True),
    ):
        # Direct CLI/service restore does not necessarily pass through API startup recovery.
        # Reconcile any precommit purge staging while the complete publication lock set is held
        # so a restore can never make quarantined rows live beside missing workspace artifacts.
        _recover_precommit_purge_staging_locked(layout, engine)
        replay_restored = False
        with Session(engine) as replay_session:
            replay = replay_session.get(DeletionOperation, operation_id)
            if replay is not None and replay.status == "restored":
                replay_restored = True
        if replay_restored:
            return _finalize_manifest_publication(layout, engine, operation_id, action="restore")
        with immediate_session(engine) as session:
            operation = session.get(DeletionOperation, operation_id)
            if operation is None:
                raise DeletionError("Deletion operation not found")
            if operation.status != "committed":
                raise DeletionError(
                    f"Only committed quarantine can be restored, not {operation.status}"
                )
            items = _stored_items(session, operation_id)
            if len(items) != operation.item_count:
                raise DeletionError("Restore manifest items are incomplete")
            _validate_deletion_manifest_items(layout, operation, items)
            selected = _items_as_selected(items)
            resources = _selection_resources(session, selected)
            conversation_snapshots = _restore_conversation_snapshots(layout, operation, items)
            for item in items:
                row = _rows_by_id(
                    session, cast(EntityType, item.entity_type), [item.entity_id]
                ).get(item.entity_id)
                if row is None:
                    raise DeletionError(
                        f"Cannot restore missing {item.entity_type} row {item.entity_id}"
                    )
                if row.deleted_at is None:
                    raise DeletionError(
                        f"Cannot restore {item.entity_type} row that is no longer quarantined"
                    )
                if item.entity_type != "conversation":
                    snapshot, _digest = _row_snapshot(row)
                    snapshot["deleted_at"] = _json_value(item.previous_deleted_at)
                    if _canonical_sha256(snapshot) != item.snapshot_sha256:
                        raise DeletionError(
                            f"Cannot restore modified {item.entity_type} row {item.entity_id}"
                        )
            affected = _update_deleted_at(session, selected, None, require_null=False)
            _reconcile_conversations(session, resources["conversation_ids"])
            restore_state = operation.purge_plan_json.get("logical_restore_state")
            if not isinstance(restore_state, dict):
                raise DeletionError("Deletion operation lacks logical restore state")
            (
                restore_global_derived,
                rebuild_reason,
                incompatible_operation_count,
            ) = _global_derived_restore_compatibility(session, operation)
            rebuild_summary: dict[str, Any] | None = None
            if not restore_global_derived:
                if rebuild_reason is None or incompatible_operation_count < 1:
                    raise DeletionError("Deletion restore compatibility state is inconsistent")
                rebuild_summary = {
                    "required": True,
                    "reason": rebuild_reason,
                    "product_kinds": [
                        "search_windows",
                        "semantic_index",
                        "nlp_hourly_rollups",
                        "conflict_aggregates",
                        "topic_clusters",
                    ],
                    "invalidated_by_count": incompatible_operation_count,
                }
            try:
                affected.update(
                    restore_derived_restore_state(
                        session,
                        restore_state,
                        delegated_annotation_ids=_validated_logical_restore_delegations(
                            session, operation
                        ),
                        restore_global_derived=restore_global_derived,
                    )
                )
            except DerivedRestoreStateError as exc:
                raise DeletionError(str(exc)) from exc
            if not restore_global_derived:
                affected["semantic_configs_invalidated_for_rebuild"] = (
                    invalidate_ready_semantic_indexes(
                        session,
                        reason=(
                            "A quarantined operation was restored after another hard purge; "
                            "rebuild search windows and semantic indexes"
                        ),
                        source_kind="deletion_restore",
                        source_id=operation.id,
                    )
                )
                affected["new_nlp_rollups_invalidated_for_rebuild"] = (
                    _invalidate_nlp_hourly_rollups(session)
                )
                affected["new_conflict_runs_invalidated_for_rebuild"] = (
                    _invalidate_conflict_aggregate_rows(
                        session,
                        reason=(
                            "A quarantined operation was restored after another hard purge; "
                            "rebuild conflict aggregates"
                        ),
                    )
                )
                affected["new_topic_jobs_invalidated_for_rebuild"] = _invalidate_topic_cluster_jobs(
                    session,
                    reason=(
                        "A quarantined operation was restored after another hard purge; "
                        "rebuild topic clusters"
                    ),
                )
            session.flush()
            session.expire_all()
            for conversation_id, expected_snapshot in conversation_snapshots.items():
                conversation = _rows_by_id(session, "conversation", [conversation_id]).get(
                    conversation_id
                )
                if conversation is None:
                    raise DeletionError(
                        f"Cannot restore missing conversation row {conversation_id}"
                    )
                current_snapshot, _digest = _row_snapshot(conversation)
                if _without_volatile_update_time(current_snapshot) != _without_volatile_update_time(
                    expected_snapshot
                ):
                    raise DeletionError(
                        f"Restored conversation invariants do not match {conversation_id}"
                    )
            operation.status = "restored"
            operation.restored_at = datetime.now(UTC)
            operation.commit_token_hash = None
            operation.purge_token_hash = None
            operation.purge_revision = None
            audit_id = _audit(
                session,
                operation_id,
                action="restore",
                outcome="completed",
                scope="logical_quarantine",
                detail={
                    "affected": affected,
                    **(
                        {"rebuild_required": rebuild_summary} if rebuild_summary is not None else {}
                    ),
                },
            )
            operation.purge_plan_json = {
                **operation.purge_plan_json,
                **(
                    {"restore_rebuild_required": rebuild_summary}
                    if rebuild_summary is not None
                    else {}
                ),
                "manifest_publication": {
                    "action": "restore",
                    "phase": "prepared",
                    "affected": affected,
                    "audit_id": audit_id,
                },
            }
            session.flush()
            if operation.manifest_relative_path is None:
                raise DeletionError("Restored deletion operation lost its manifest path")
            _write_manifest(
                layout,
                operation.manifest_relative_path,
                operation=operation,
                items=items,
                publication_phase="prepared",
            )
        return _finalize_manifest_publication(layout, engine, operation_id, action="restore")


def _dependent_rows_outside_operation(
    session: Session, selected: dict[EntityType, set[str]]
) -> int:
    outside = 0
    for chunk in _chunks(sorted(selected["conversation"])):
        outside += sum(
            message_id not in selected["message"]
            for message_id in session.scalars(
                select(Message.id).where(Message.conversation_id.in_(chunk))
            )
        )
        outside += sum(
            attachment_id not in selected["attachment"]
            for attachment_id in session.scalars(
                select(AttachmentOccurrence.id).where(
                    AttachmentOccurrence.conversation_id.in_(chunk)
                )
            )
        )
    for chunk in _chunks(sorted(selected["message"])):
        outside += sum(
            attachment_id not in selected["attachment"]
            for attachment_id in session.scalars(
                select(AttachmentOccurrence.id).where(AttachmentOccurrence.message_id.in_(chunk))
            )
        )
    for chunk in _chunks(sorted(selected["activity"])):
        outside += sum(
            child_id not in selected["activity"]
            for child_id in session.scalars(
                select(ActivityRecord.id).where(ActivityRecord.parent_id.in_(chunk))
            )
        )
        outside += sum(
            attachment_id not in selected["attachment"]
            for attachment_id in session.scalars(
                select(AttachmentOccurrence.id).where(
                    AttachmentOccurrence.activity_record_id.in_(chunk)
                )
            )
        )
    # Manual merge/split corrections are durable user-authored state. Physically deleting a
    # corrected conversation would either trip the RESTRICT foreign key or silently discard the
    # segment override through a segment cascade, leaving its correction audit impossible to
    # revert. Require the correction to be reverted before planning a purge instead.
    for chunk in _chunks(sorted(selected["conversation"])):
        outside += int(
            session.scalar(
                select(func.count(ConversationSegmentOverride.segment_id)).where(
                    ConversationSegmentOverride.conversation_id.in_(chunk)
                )
            )
            or 0
        )
        outside += int(
            session.scalar(
                select(func.count(ConversationSegmentOverride.segment_id))
                .join(
                    ConversationSegment,
                    ConversationSegment.id == ConversationSegmentOverride.segment_id,
                )
                .where(
                    ConversationSegment.conversation_id.in_(chunk),
                    ConversationSegmentOverride.conversation_id.not_in(chunk),
                )
            )
            or 0
        )
    return outside


def _tree_revision_entries(layout: ArchiveLayout, root: Path) -> list[dict[str, Any]]:
    """Return safe, cheap revision metadata for a workspace tree used by a purge plan."""

    if not root.exists() and not root.is_symlink():
        return []
    if root.is_symlink() or not root.is_dir():
        raise HardPurgeBlockedError(f"Workspace tree is not a safe directory: {root.name}")
    resolved_root = root.resolve()
    archive_root = layout.root.resolve()
    if (
        resolved_root == archive_root
        or archive_root not in resolved_root.parents
        or resolved_root != root.absolute()
    ):
        raise HardPurgeBlockedError(
            f"Workspace tree traverses an unsafe parent directory: {root.name}"
        )
    entries: list[dict[str, Any]] = []
    for candidate in sorted(root.rglob("*")):
        candidate_stat = candidate.lstat()
        if stat.S_ISLNK(candidate_stat.st_mode):
            raise HardPurgeBlockedError(
                f"Workspace tree contains a symlink: {candidate.relative_to(layout.root)}"
            )
        if stat.S_ISDIR(candidate_stat.st_mode):
            continue
        if not stat.S_ISREG(candidate_stat.st_mode):
            raise HardPurgeBlockedError(
                f"Workspace tree contains a non-regular entry: {candidate.relative_to(layout.root)}"
            )
        entries.append(
            {
                "path": candidate.relative_to(layout.root).as_posix(),
                "size_bytes": candidate_stat.st_size,
                "mtime_ns": candidate_stat.st_mtime_ns,
            }
        )
    return entries


def _vector_table_inventory(
    layout: ArchiveLayout, session: Session
) -> tuple[set[str], set[str], dict[str, int], str | None]:
    registered = set(session.scalars(select(SemanticIndexConfig.vector_table)))
    if not layout.vectors.is_dir() or not any(layout.vectors.iterdir()):
        return registered, set(), {}, None
    try:
        import lancedb  # type: ignore[import-untyped]

        database = lancedb.connect(str(layout.vectors))
        available = lance_table_names(database)
        table_counts = {
            table_name: int(database.open_table(table_name).count_rows())
            for table_name in available
        }
    except Exception as exc:
        return registered, set(), {}, f"{type(exc).__name__}: {exc}"
    return registered, available, table_counts, None


def _vector_identity_inventory(
    layout: ArchiveLayout, session: Session
) -> tuple[set[str], set[str], dict[str, SemanticWindowIdentity], str | None]:
    registered = set(session.scalars(select(SemanticIndexConfig.vector_table)))
    if not layout.vectors.is_dir() or not any(layout.vectors.iterdir()):
        return registered, set(), {}, None
    try:
        import lancedb

        database = lancedb.connect(str(layout.vectors))
        available = lance_table_names(database)
        identities = {
            table_name: lance_window_identity(database.open_table(table_name))
            for table_name in available
        }
    except Exception as exc:
        return registered, set(), {}, f"{type(exc).__name__}: {exc}"
    return registered, available, identities, None


def _normalized_hard_purge_star_targets(
    session: Session,
    selected: dict[EntityType, set[str]],
) -> set[StarTargetKey]:
    """Return stars whose normalized targets become permanently unavailable."""

    targets: set[StarTargetKey] = {
        (cast(StarTargetType, entity_type), entity_id)
        for entity_type, entity_ids in selected.items()
        for entity_id in entity_ids
    }
    targets.update(
        ("messenger_contact", contact_id)
        for contact_id in messenger_contact_ids_eliminated_by_conversation_removal(
            session,
            selected["conversation"],
        )
    )
    return targets


def _blob_star_targets(blob_ids: Iterable[str]) -> set[StarTargetKey]:
    return {("blob", blob_id) for blob_id in blob_ids}


def _star_revision_rows(
    session: Session,
    targets: Iterable[StarTargetKey],
) -> list[dict[str, str]]:
    """Bind every affected star to a purge plan without copying target content."""

    rows = star_rows_by_target(session, targets)
    return [
        {
            "id": row.id,
            "target_type": row.target_type,
            "target_id": row.target_id,
            "starred_at": str(_json_value(row.starred_at)),
        }
        for _key, row in sorted(rows.items())
    ]


def _purge_state(
    layout: ArchiveLayout,
    session: Session,
    operation: DeletionOperation,
    items: Sequence[DeletionItem],
    scope: PurgeScope,
) -> tuple[dict[str, Any], tuple[str, ...], str]:
    selected = _items_as_selected(items)
    resources = _selection_resources(session, selected)
    overlap_inventory = inventory_deletion_overlaps(
        session,
        current_operation_id=operation.id,
        selected_entity_ids={str(key): value for key, value in selected.items()},
    )
    correction_inventory = inventory_conversation_corrections(
        session,
        conversation_ids=selected["conversation"],
        segment_ids=resources["conversation_segment_ids"],
    )
    for overlap_operation_id in sorted(overlap_inventory.invalidatable_operation_ids):
        overlap_operation = session.get(DeletionOperation, overlap_operation_id)
        if overlap_operation is None:
            raise HardPurgeBlockedError("An overlapping deletion operation disappeared")
        if overlap_operation.manifest_relative_path is not None:
            try:
                _canonical_manifest_relative_path(layout, overlap_operation)
            except DeletionError as exc:
                raise HardPurgeBlockedError(
                    "An overlapping deletion manifest path is invalid"
                ) from exc
    overlap_manifest_revision: list[dict[str, Any]] = []
    for relative_path in sorted(overlap_inventory.invalidatable_manifest_paths):
        try:
            manifest_snapshot = read_private_file(layout.root, relative_path)
            if manifest_snapshot is None:  # pragma: no cover - `missing_ok=False`.
                raise SecurePublicationError("Overlapping deletion manifest is missing")
        except SecurePublicationError as exc:
            raise HardPurgeBlockedError(
                "An overlapping deletion manifest is missing or unsafe"
            ) from exc
        overlap_manifest_revision.append(
            {
                "path": relative_path,
                "size_bytes": manifest_snapshot.identity.size_bytes,
                "mtime_ns": manifest_snapshot.identity.modified_ns,
                "sha256": manifest_snapshot.identity.sha256,
            }
        )
    deletion_manifest_revision: dict[str, Any] | None = None
    deletion_manifest_error: str | None = None
    if operation.manifest_relative_path is None:
        deletion_manifest_error = "Committed deletion operation has no restore manifest"
    else:
        try:
            manifest_relative_path = _canonical_manifest_relative_path(layout, operation)
            manifest_snapshot, manifest_payload = _read_deletion_manifest(
                layout,
                manifest_relative_path,
                error_message="Deletion restore manifest is missing or invalid",
            )
            _validate_deletion_manifest_payload(operation, items, manifest_payload)
            deletion_manifest_revision = {
                "path": manifest_relative_path,
                "size_bytes": manifest_snapshot.identity.size_bytes,
                "mtime_ns": manifest_snapshot.identity.modified_ns,
                "sha256": manifest_snapshot.identity.sha256,
            }
        except (
            DeletionError,
            HardPurgeBlockedError,
        ) as exc:
            deletion_manifest_error = str(exc)
    raw_rewrite_plan = RawRewritePlan((), (), (), ())
    raw_rewrite_error: str | None = None
    try:
        raw_rewrite_plan = build_raw_rewrite_plan(
            layout,
            session,
            selected,
            binary_blob_ids=resources["blob_ids"],
            binary_source_file_ids={int(value) for value in resources["binary_source_file_ids"]},
        )
    except RawSourceRewriteError as exc:
        raw_rewrite_error = str(exc)
    star_targets = _normalized_hard_purge_star_targets(session, selected)
    if scope == "workspace":
        star_targets.update(
            _blob_star_targets(
                resources["physically_exclusive_blob_ids"] | set(raw_rewrite_plan.obsolete_blob_ids)
            )
        )
    star_revision = _star_revision_rows(session, star_targets)
    missing: list[str] = []
    not_quarantined: list[str] = []
    current_snapshots: list[tuple[str, str, str]] = []
    for entity_type in sorted(ENTITY_TYPES):
        typed = cast(EntityType, entity_type)
        rows = _rows_by_id(session, typed, selected[typed])
        for entity_id in sorted(selected[typed]):
            row = rows.get(entity_id)
            if row is None:
                missing.append(f"{typed}:{entity_id}")
            else:
                if row.deleted_at is None:
                    not_quarantined.append(f"{typed}:{entity_id}")
                _snapshot, snapshot_digest = _row_snapshot(row)
                current_snapshots.append((typed, entity_id, snapshot_digest))
    source_file_ids = {int(value) for value in resources["source_file_ids"]}
    source_blob_refs = 0
    for chunk in _chunks(sorted(resources["physically_exclusive_blob_ids"])):
        source_blob_refs += int(
            session.scalar(select(func.count(SourceFile.id)).where(SourceFile.sha256.in_(chunk)))
            or 0
        )
    candidate_blob_bytes = 0
    derivative_count = 0
    derivative_bytes = 0
    transcription_count = 0
    transcript_segment_count = 0
    extraction_count = 0
    for chunk in _chunks(sorted(resources["physically_exclusive_blob_ids"])):
        candidate_blob_bytes += int(
            session.scalar(
                select(func.coalesce(func.sum(Blob.size_bytes), 0)).where(Blob.sha256.in_(chunk))
            )
            or 0
        )
        derivative_count += int(
            session.scalar(
                select(func.count(MediaDerivative.id)).where(MediaDerivative.blob_sha256.in_(chunk))
            )
            or 0
        )
        derivative_bytes += int(
            session.scalar(
                select(func.coalesce(func.sum(MediaDerivative.size_bytes), 0)).where(
                    MediaDerivative.blob_sha256.in_(chunk)
                )
            )
            or 0
        )
        transcription_count += int(
            session.scalar(
                select(func.count(AudioTranscription.id)).where(
                    AudioTranscription.blob_sha256.in_(chunk)
                )
            )
            or 0
        )
        transcript_segment_count += int(
            session.scalar(
                select(func.count(AudioTranscriptSegment.id))
                .join(
                    AudioTranscription,
                    AudioTranscription.id == AudioTranscriptSegment.transcription_id,
                )
                .where(AudioTranscription.blob_sha256.in_(chunk))
            )
            or 0
        )
        extraction_count += int(
            session.scalar(
                select(func.count(AttachmentTextExtraction.id)).where(
                    AttachmentTextExtraction.blob_sha256.in_(chunk)
                )
            )
            or 0
        )
    nlp_count = 0
    for entity_type, identifiers in _physical_nlp_target_ids(selected, resources).items():
        nlp_count += _count_for_chunks(
            session,
            identifiers,
            lambda chunk, entity_type=entity_type: select(func.count(NlpAnnotation.id)).where(
                NlpAnnotation.target_type == entity_type,
                NlpAnnotation.target_id.in_(chunk),
            ),
        )
    dependent_outside = _dependent_rows_outside_operation(session, selected)
    workspace_files: list[dict[str, Any]] = []
    orphan_derivative_files = 0
    derivative_inventory_blockers: tuple[str, ...] = ()
    filesystem_library_present = False
    library_revision: list[dict[str, Any]] = []
    conflict_holding_revision: list[dict[str, Any]] = []
    # Analytics caches are derived from normalized rows, so both purge scopes remove them.
    analytics_cache_revision: list[dict[str, Any]] = _tree_revision_entries(
        layout, layout.analytics_cache
    )
    if scope == "workspace":
        library_revision = _tree_revision_entries(layout, layout.library)
        filesystem_library_present = bool(library_revision)
        conflict_holding_revision = _tree_revision_entries(layout, layout.conflicts)
        derivative_inventory = inventory_derivative_tree(
            layout,
            session,
            resources["physically_exclusive_blob_ids"],
        )
        derivative_inventory_blockers = derivative_inventory.blockers
        orphan_derivative_files = len(derivative_inventory.selected_orphan_paths)
        workspace_targets = (
            _workspace_targets(layout, session, resources["physically_exclusive_blob_ids"])
            | raw_rewrite_live_targets(layout, raw_rewrite_plan)
            | set(derivative_inventory.selected_paths)
        )
        for target in sorted(workspace_targets):
            try:
                target_stat = regular_file_stat_beneath(target, layout.root)
            except UnsafeArchivePathError as exc:
                raise HardPurgeBlockedError(
                    f"Workspace purge target became unsafe: {target.name}"
                ) from exc
            workspace_files.append(
                {
                    "path": target.relative_to(Path(os.path.abspath(layout.root))).as_posix(),
                    "size_bytes": target_stat.st_size,
                    "mtime_ns": target_stat.st_mtime_ns,
                }
            )
    # Hard purge discards every live Lance fragment, independent of current SearchWindow links.
    # Stale/orphan rows can retain selected plaintext even when the selected entity has no current
    # membership. A complete safe-tree revision binds the discard target to this plan.
    vector_revision = _tree_revision_entries(layout, layout.vectors)
    vector_store_present = layout.vectors.is_dir() and any(layout.vectors.iterdir())
    # Inventory remains useful diagnostic evidence, but inconsistency cannot block erasure now
    # that the whole store is staged and removed rather than selectively rewritten.
    (
        registered_vector_tables,
        available_vector_tables,
        vector_table_counts,
        vector_inventory_error,
    ) = _vector_table_inventory(layout, session)
    unregistered_vector_tables = available_vector_tables - registered_vector_tables
    missing_ready_vector_tables = {
        table_name
        for table_name in session.scalars(
            select(SemanticIndexConfig.vector_table).where(
                SemanticIndexConfig.active.is_(True),
                SemanticIndexConfig.status == "ready",
            )
        )
        if table_name not in available_vector_tables
    }
    vector_catalog_count_mismatches: list[dict[str, Any]] = []
    for config in session.scalars(select(SemanticIndexConfig).order_by(SemanticIndexConfig.id)):
        if config.vector_table not in vector_table_counts:
            continue
        sqlite_count = semantic_config_window_count(session, config.id)
        lance_count = vector_table_counts[config.vector_table]
        if sqlite_count != lance_count:
            vector_catalog_count_mismatches.append(
                {
                    "config_id": config.id,
                    "table": config.vector_table,
                    "sqlite_rows": sqlite_count,
                    "lance_rows": lance_count,
                }
            )
    blockers: list[str] = []
    if missing:
        blockers.append("normalized_rows_missing")
    if not_quarantined:
        blockers.append("normalized_rows_not_quarantined")
    if dependent_outside:
        blockers.append("dependent_rows_belong_to_another_operation")
    blockers.extend(overlap_inventory.blockers)
    blockers.extend(correction_inventory.blockers)
    if deletion_manifest_error is not None:
        blockers.append("deletion_manifest_missing_or_unsafe")
    if raw_rewrite_error is not None:
        blockers.append("raw_source_rewrite_validation_failed")
    if resources["outside_source_file_ids"]:
        blockers.append("outside_source_file_references_require_review")
    blockers.extend(derivative_inventory_blockers)
    impact = {
        "scope": scope,
        "normalized_rows": len(items),
        "search_windows": len(resources["purge_search_window_ids"]),
        "candidate_blobs": len(resources["physically_exclusive_blob_ids"]),
        "logically_hidden_blobs": len(resources["exclusive_blob_ids"]),
        "recoverably_referenced_blobs": len(resources["recoverably_referenced_blob_ids"]),
        "candidate_blob_bytes": candidate_blob_bytes,
        "shared_blobs_preserved": len(resources["shared_blob_ids"]),
        "media_derivatives": derivative_count,
        "orphan_media_derivative_files": orphan_derivative_files,
        "media_derivative_bytes": derivative_bytes,
        "audio_transcriptions": transcription_count,
        "audio_transcript_segments": transcript_segment_count,
        "attachment_text_extractions": extraction_count,
        "nlp_annotations": nlp_count,
        "stars_removed": len(star_revision),
        "raw_source_files": len(source_file_ids),
        "raw_source_files_rewritten": len(raw_rewrite_plan.rewrites),
        "raw_source_obsolete_blobs": len(raw_rewrite_plan.obsolete_blob_ids),
        "raw_source_rewrite_error": raw_rewrite_error,
        "source_blob_references": source_blob_refs,
        "shared_source_files_preserved": len(resources["shared_source_file_ids"]),
        "outside_source_file_references": len(resources["outside_source_file_ids"]),
        "dependent_rows_outside_operation": dependent_outside,
        "overlapping_deletion_operations_invalidated": len(
            overlap_inventory.invalidatable_operation_ids
        ),
        "overlapping_deletion_items_invalidated": overlap_inventory.invalidatable_item_count,
        "reverted_conversation_corrections_scrubbed": len(
            correction_inventory.scrub_correction_ids
        ),
        "workspace_files": len(workspace_files),
        "workspace_bytes": sum(item["size_bytes"] for item in workspace_files),
        "filesystem_library_present": filesystem_library_present,
        "filesystem_library_files": len(library_revision),
        "conflict_holding_files": len(conflict_holding_revision),
        "analytics_cache_files": len(analytics_cache_revision),
        "vector_store_present": vector_store_present,
        "semantic_store_will_be_discarded": vector_store_present,
        "vector_store_files": len(vector_revision),
        "registered_vector_tables": len(registered_vector_tables),
        "unregistered_vector_tables": sorted(unregistered_vector_tables),
        "missing_ready_vector_tables": sorted(missing_ready_vector_tables),
        "vector_catalog_count_mismatches": vector_catalog_count_mismatches,
        "vector_inventory_error": vector_inventory_error,
        "manifest_will_be_removed": not blockers,
        "deletion_manifest_error": deletion_manifest_error,
        "sqlite_compaction": not blockers,
    }
    revision = _canonical_sha256(
        {
            "operation_id": operation.id,
            "operation_status": operation.status,
            "scope": scope,
            "rows": current_snapshots,
            "resources": {key: sorted(value) for key, value in resources.items()},
            "workspace_files": workspace_files,
            "raw_rewrites": raw_rewrite_plan.revision_records(),
            "star_revision": star_revision,
            "library_revision": library_revision,
            "conflict_holding_revision": conflict_holding_revision,
            "analytics_cache_revision": analytics_cache_revision,
            "vector_revision": vector_revision,
            "deletion_manifest_revision": deletion_manifest_revision,
            "overlap_manifest_revision": overlap_manifest_revision,
            "impact": impact,
            "blockers": blockers,
        }
    )
    return impact, tuple(blockers), revision


def _saved_resources_as_sets(resources: SavedDeletionResources) -> dict[str, set[str]]:
    return {key: {str(value) for value in values} for key, values in resources.to_json().items()}


def _source_manifest_content_revision(layout: ArchiveLayout) -> dict[str, dict[str, Any]]:
    revision: dict[str, dict[str, Any]] = {}
    for path in (layout.original_files_manifest, layout.relocation_manifest):
        try:
            with open_regular_file_beneath(path, layout.root) as handle:
                path_stat = os.fstat(handle.fileno())
                digest = hashlib.sha256()
                while chunk := handle.read(1_048_576):
                    digest.update(chunk)
        except (OSError, UnsafeArchivePathError) as exc:
            raise HardPurgeBlockedError("Source manifest is missing or unsafe") from exc
        revision[path.relative_to(Path(os.path.abspath(layout.root))).as_posix()] = {
            "size_bytes": path_stat.st_size,
            "sha256": digest.hexdigest(),
        }
    return revision


def _saved_source_manifest_baseline(
    layout: ArchiveLayout, operation: DeletionOperation
) -> dict[str, dict[str, Any]]:
    raw = operation.purge_plan_json.get("workspace_upgrade_manifest_baseline")
    expected_paths = {
        path.relative_to(Path(os.path.abspath(layout.root))).as_posix()
        for path in (layout.original_files_manifest, layout.relocation_manifest)
    }
    if not isinstance(raw, dict) or set(raw) != expected_paths:
        raise HardPurgeBlockedError("Normalized purge lacks a source-manifest baseline")
    result: dict[str, dict[str, Any]] = {}
    for relative_path, value in raw.items():
        if (
            not isinstance(relative_path, str)
            or not isinstance(value, dict)
            or set(value) != {"size_bytes", "sha256"}
            or isinstance(value.get("size_bytes"), bool)
            or not isinstance(value.get("size_bytes"), int)
            or value["size_bytes"] < 0
            or not isinstance(value.get("sha256"), str)
            or len(value["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in value["sha256"])
        ):
            raise HardPurgeBlockedError("Normalized purge source-manifest baseline is invalid")
        result[relative_path] = {
            "size_bytes": value["size_bytes"],
            "sha256": value["sha256"],
        }
    return result


def _normalized_workspace_purge_state(
    layout: ArchiveLayout,
    session: Session,
    operation: DeletionOperation,
    items: Sequence[DeletionItem],
) -> tuple[dict[str, Any], tuple[str, ...], str]:
    """Plan the workspace half of a previously completed normalized-only purge."""

    receipt_value = operation.purge_plan_json.get("workspace_upgrade_state")
    try:
        receipt = WorkspaceUpgradeStateReceipt.from_json(receipt_value)
        saved = load_workspace_upgrade_state(
            layout,
            layout.root / receipt.relative_path,
            receipt=receipt,
            expected_operation_id=operation.id,
            expected_selection_sha256=operation.selection_sha256,
            expected_purge_revision=receipt.purge_revision,
        )
    except DeletionUpgradeStateError as exc:
        raise HardPurgeBlockedError(
            "Normalized purge workspace-upgrade state is missing or invalid"
        ) from exc
    if operation.manifest_relative_path is None:
        raise HardPurgeBlockedError("Normalized purge selection manifest is missing")
    try:
        _validate_deletion_manifest_items(layout, operation, items)
    except DeletionError as exc:
        raise HardPurgeBlockedError(
            "Normalized purge selection manifest no longer matches the catalog"
        ) from exc
    manifest_path = _safe_workspace_path(layout, operation.manifest_relative_path)

    resources = _saved_resources_as_sets(saved.resources)
    physical_blobs = resources["physically_exclusive_blob_ids"]
    selected = _items_as_selected(items)
    recreated_entity_ids: list[str] = []
    for entity_type in sorted(ENTITY_TYPES):
        typed = cast(EntityType, entity_type)
        recreated_entity_ids.extend(
            f"{typed}:{entity_id}"
            for entity_id in sorted(_rows_by_id(session, typed, selected[typed]))
        )
    recreated_source_record_ids = _scalar_ids_for_chunks(
        session,
        resources["source_record_ids"],
        lambda chunk: select(SourceRecord.id).where(SourceRecord.id.in_(chunk)),
    )
    recreated_search_window_ids = _scalar_ids_for_chunks(
        session,
        resources["purge_search_window_ids"],
        lambda chunk: select(SearchWindow.id).where(SearchWindow.id.in_(chunk)),
    )
    recreated_audio_segment_ids = _scalar_ids_for_chunks(
        session,
        resources["physical_audio_segment_ids"],
        lambda chunk: select(AudioTranscriptSegment.id).where(AudioTranscriptSegment.id.in_(chunk)),
    )
    recreated_extraction_ids = _scalar_ids_for_chunks(
        session,
        resources["physical_attachment_extraction_ids"],
        lambda chunk: select(AttachmentTextExtraction.id).where(
            AttachmentTextExtraction.id.in_(chunk)
        ),
    )
    expected_manifest_baseline: dict[str, dict[str, Any]] = {}
    current_manifest_baseline: dict[str, dict[str, Any]] = {}
    manifest_baseline_changed = False
    if saved.raw_plan.rewrites:
        expected_manifest_baseline = _saved_source_manifest_baseline(layout, operation)
        current_manifest_baseline = _source_manifest_content_revision(layout)
        manifest_baseline_changed = current_manifest_baseline != expected_manifest_baseline
    physical_source_identity_error: str | None = None
    try:
        validate_raw_rewrite_physical_sources(layout, saved.raw_plan)
    except RawSourceRewriteError as exc:
        physical_source_identity_error = str(exc)
    rewrite_source_file_ids = {rewrite.source_file_id for rewrite in saved.raw_plan.rewrites}
    blob_delete_candidates = physical_blobs | set(saved.raw_plan.obsolete_blob_ids)
    blob_star_targets = _blob_star_targets(blob_delete_candidates)
    star_revision = _star_revision_rows(session, blob_star_targets)
    late_source_file_refs: set[str] = set()
    for chunk in _chunks(sorted(blob_delete_candidates)):
        late_source_file_refs.update(
            str(value)
            for value in session.scalars(
                select(SourceFile.id).where(
                    SourceFile.sha256.in_(chunk),
                    SourceFile.id.not_in(rewrite_source_file_ids),
                )
            )
        )
    late_attachment_refs = _scalar_ids_for_chunks(
        session,
        blob_delete_candidates,
        lambda chunk: select(AttachmentOccurrence.id).where(
            AttachmentOccurrence.blob_sha256.in_(chunk)
        ),
    )
    derivative_inventory = inventory_derivative_tree(layout, session, physical_blobs)
    raw_targets = raw_rewrite_live_targets(layout, saved.raw_plan)
    workspace_targets = (
        _workspace_targets(layout, session, physical_blobs)
        | raw_targets
        | set(derivative_inventory.selected_paths)
        | {manifest_path, layout.root / receipt.relative_path}
    )
    workspace_files: list[dict[str, Any]] = []
    for target in sorted(workspace_targets):
        try:
            target_stat = regular_file_stat_beneath(target, layout.root)
        except UnsafeArchivePathError as exc:
            raise HardPurgeBlockedError(
                "Normalized workspace-upgrade target became unsafe"
            ) from exc
        workspace_files.append(
            {
                "path": target.relative_to(Path(os.path.abspath(layout.root))).as_posix(),
                "size_bytes": target_stat.st_size,
                "mtime_ns": target_stat.st_mtime_ns,
            }
        )

    library_revision = _tree_revision_entries(layout, layout.library)
    conflict_revision = _tree_revision_entries(layout, layout.conflicts)
    analytics_revision = _tree_revision_entries(layout, layout.analytics_cache)
    vector_revision = _tree_revision_entries(layout, layout.vectors)
    vector_store_present = layout.vectors.is_dir() and any(layout.vectors.iterdir())
    registered, available, vector_identities, vector_error = _vector_identity_inventory(
        layout, session
    )
    unregistered = available - registered
    missing_ready = {
        table_name
        for table_name in session.scalars(
            select(SemanticIndexConfig.vector_table).where(
                SemanticIndexConfig.active.is_(True),
                SemanticIndexConfig.status == "ready",
            )
        )
        if table_name not in available
    }
    count_mismatches: list[dict[str, Any]] = []
    identity_mismatches: list[dict[str, Any]] = []
    for config in session.scalars(select(SemanticIndexConfig).order_by(SemanticIndexConfig.id)):
        lance_identity = vector_identities.get(config.vector_table)
        if lance_identity is None:
            continue
        sqlite_identity = semantic_window_identity(semantic_config_window_ids(session, config.id))
        if sqlite_identity.row_count != lance_identity.row_count:
            count_mismatches.append(
                {
                    "config_id": config.id,
                    "table": config.vector_table,
                    "sqlite_rows": sqlite_identity.row_count,
                    "lance_rows": lance_identity.row_count,
                }
            )
        if sqlite_identity != lance_identity:
            identity_mismatches.append(
                {
                    "config_id": config.id,
                    "table": config.vector_table,
                    "sqlite_rows": sqlite_identity.row_count,
                    "lance_rows": lance_identity.row_count,
                    "lance_unique_rows": lance_identity.unique_count,
                }
            )
    blockers = list(derivative_inventory.blockers)
    if recreated_entity_ids:
        blockers.append("normalized_rows_recreated_after_normalized_purge")
    if recreated_source_record_ids:
        blockers.append("source_records_recreated_after_normalized_purge")
    if recreated_search_window_ids or recreated_audio_segment_ids or recreated_extraction_ids:
        blockers.append("derived_rows_recreated_after_normalized_purge")
    if manifest_baseline_changed:
        blockers.append("source_manifests_changed_after_normalized_purge")
    if physical_source_identity_error is not None:
        blockers.append("physical_source_changed_after_normalized_purge")
    if late_source_file_refs:
        blockers.append("new_source_file_references_block_workspace_upgrade")
    if late_attachment_refs:
        blockers.append("new_attachment_references_block_workspace_upgrade")
    for chunk in _chunks(sorted(physical_blobs)):
        if (
            session.scalar(
                select(AttachmentOccurrence.id)
                .where(AttachmentOccurrence.blob_sha256.in_(chunk))
                .limit(1)
            )
            is not None
        ):
            blockers.append("physical_blob_gained_a_recoverable_reference")
            break
    source_catalog_revision: list[dict[str, Any]] = []
    for rewrite in saved.raw_plan.rewrites:
        source_file = session.get(SourceFile, rewrite.source_file_id)
        source_catalog_revision.append(
            {
                "id": rewrite.source_file_id,
                "sha256": source_file.sha256 if source_file is not None else None,
                "size_bytes": source_file.size_bytes if source_file is not None else None,
            }
        )
        if (
            source_file is None
            or source_file.sha256 != rewrite.old_sha256
            or source_file.size_bytes != rewrite.old_size_bytes
        ):
            blockers.append("raw_source_catalog_changed_after_normalized_purge")
    impact = {
        "scope": "workspace",
        "normalized_rows": 0,
        "workspace_upgrade": True,
        "candidate_blobs": len(physical_blobs),
        "stars_removed": len(star_revision),
        "search_windows": 0,
        "workspace_files": len(workspace_files),
        "workspace_bytes": sum(row["size_bytes"] for row in workspace_files),
        "filesystem_library_present": bool(library_revision),
        "filesystem_library_files": len(library_revision),
        "conflict_holding_files": len(conflict_revision),
        "analytics_cache_files": len(analytics_revision),
        "vector_store_present": vector_store_present,
        "semantic_store_will_be_discarded": vector_store_present,
        "vector_store_files": len(vector_revision),
        "registered_vector_tables": len(registered),
        "unregistered_vector_tables": sorted(unregistered),
        "missing_ready_vector_tables": sorted(missing_ready),
        "vector_catalog_count_mismatches": count_mismatches,
        "vector_catalog_identity_mismatches": identity_mismatches,
        "vector_inventory_error": vector_error,
        "raw_source_files_rewritten": len(saved.raw_plan.rewrites),
        "raw_source_obsolete_blobs": len(saved.raw_plan.obsolete_blob_ids),
        "recreated_normalized_rows": len(recreated_entity_ids),
        "recreated_source_records": len(recreated_source_record_ids),
        "recreated_derived_rows": (
            len(recreated_search_window_ids)
            + len(recreated_audio_segment_ids)
            + len(recreated_extraction_ids)
        ),
        "source_manifest_baseline_changed": manifest_baseline_changed,
        "physical_source_identity_changed": physical_source_identity_error is not None,
        "late_source_file_references": len(late_source_file_refs),
        "late_attachment_references": len(late_attachment_refs),
        "orphan_media_derivative_files": len(derivative_inventory.selected_orphan_paths),
        "manifest_will_be_removed": not blockers,
        "sqlite_compaction": not blockers,
    }
    revision = _canonical_sha256(
        {
            "operation_id": operation.id,
            "operation_status": operation.status,
            "selection_sha256": operation.selection_sha256,
            "upgrade_receipt": receipt.to_json(),
            "workspace_files": workspace_files,
            "library_revision": library_revision,
            "conflict_holding_revision": conflict_revision,
            "analytics_cache_revision": analytics_revision,
            "vector_revision": vector_revision,
            "source_catalog_revision": source_catalog_revision,
            "expected_manifest_baseline": expected_manifest_baseline,
            "current_manifest_baseline": current_manifest_baseline,
            "recreated_entity_ids": recreated_entity_ids,
            "recreated_source_record_ids": sorted(recreated_source_record_ids),
            "recreated_search_window_ids": sorted(recreated_search_window_ids),
            "recreated_audio_segment_ids": sorted(recreated_audio_segment_ids),
            "recreated_extraction_ids": sorted(recreated_extraction_ids),
            "late_source_file_refs": sorted(late_source_file_refs),
            "late_attachment_refs": sorted(late_attachment_refs),
            "star_revision": star_revision,
            "impact": impact,
            "blockers": blockers,
            "item_count": len(items),
        }
    )
    return impact, tuple(blockers), revision


def plan_hard_purge(
    layout: ArchiveLayout,
    engine: Engine,
    operation_id: str,
    *,
    scope: PurgeScope,
) -> PurgePlan:
    if scope not in {"normalized", "workspace"}:
        raise HardPurgeBlockedError("Unknown hard-purge scope")
    with (
        deletion_lock(layout),
        purge_source_publisher_locks(layout) as _publisher_leases,
        purge_worker_locks(layout),
        vector_store_lock(layout),
        analytics_cache_lock(layout, exclusive=True),
        Session(engine) as session,
    ):
        # Planning must bind to the recovered live workspace, not a transient tree left behind
        # by a purge process that died before its catalog transaction committed.
        _recover_precommit_purge_staging_locked(layout, engine)
        operation = session.get(DeletionOperation, operation_id)
        if operation is None:
            raise DeletionError("Deletion operation not found")
        if operation.status not in {"committed", "normalized_purged"}:
            raise DeletionError(
                "Only committed or normalized-purged quarantine can be planned for hard purge"
            )
        if operation.status == "normalized_purged" and scope != "workspace":
            raise DeletionError("A normalized-purged operation can only be upgraded to workspace")
        items = _stored_items(session, operation_id)
        if operation.status == "normalized_purged":
            impact, blockers, revision = _normalized_workspace_purge_state(
                layout, session, operation, items
            )
        else:
            impact, blockers, revision = _purge_state(layout, session, operation, items, scope)
    token = secrets.token_urlsafe(32)
    with Session(engine) as session, session.begin():
        operation = session.get(DeletionOperation, operation_id)
        if operation is None or operation.status not in {"committed", "normalized_purged"}:
            raise StaleDeletionPreviewError("Deletion operation changed during purge planning")
        operation.purge_token_hash = _token_hash(token)
        operation.purge_revision = revision
        operation.purge_plan_json = {
            **operation.purge_plan_json,
            "scope": scope,
            "impact": impact,
            "blockers": list(blockers),
            "planned_at": datetime.now(UTC).isoformat(),
        }
        _audit(
            session,
            operation_id,
            action="purge_preview",
            outcome="blocked" if blockers else "ready",
            scope=scope,
            detail={"impact": impact, "blockers": list(blockers), "purge_revision": revision},
        )
    return PurgePlan(
        operation_id=operation_id,
        scope=scope,
        executable=not blockers,
        blockers=blockers,
        purge_token=token,
        purge_revision=revision,
        impact=impact,
    )


def _assert_live_vector_store_empty(layout: ArchiveLayout) -> None:
    """Require the live vector path to be absent or an actually empty safe directory."""

    try:
        vector_stat = archive_entry_stat_beneath(layout.vectors, layout.root)
    except UnsafeArchivePathError as exc:
        raise HardPurgeBlockedError("The live semantic vector path is unsafe") from exc
    if vector_stat is None:
        return
    if stat.S_ISLNK(vector_stat.st_mode) or not stat.S_ISDIR(vector_stat.st_mode):
        raise HardPurgeBlockedError("The live semantic vector path is unsafe")
    descriptor: int | None = None
    try:
        descriptor = open_directory_beneath(layout.vectors, layout.root)
        if os.listdir(descriptor):
            raise HardPurgeBlockedError(
                "The live semantic vector tree contains unexpected residual files"
            )
    except UnsafeArchivePathError as exc:
        raise HardPurgeBlockedError("The live semantic vector path is unsafe") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _delete_vectors(
    layout: ArchiveLayout,
    session: Session,
    window_ids: set[str],
    *,
    discard_store: bool,
) -> VectorDeletionResult:
    """Discard the complete semantic store when any indexed private window is purged.

    LanceDB deletion vectors and ordinary compaction do not prove physical erasure: the sole
    current data fragment and ANN index can retain the deleted row's plaintext and vector bytes.
    The execution path therefore stages the entire old vector tree and intentionally does not
    copy it back. Clearing every catalog membership and marking every configuration stale makes
    the empty live store explicit and forces a clean rebuild from retained SearchWindow rows.
    """

    _assert_live_vector_store_empty(layout)

    configs = list(session.scalars(select(SemanticIndexConfig).order_by(SemanticIndexConfig.id)))
    discarded_rows = sum(len(semantic_config_window_ids(session, config.id)) for config in configs)
    memberships_removed = _result_rowcount(session.execute(delete(SemanticWindowEmbedding)))
    session.execute(
        update(SearchWindow)
        .where(
            or_(
                SearchWindow.embedding_config_hash.is_not(None),
                SearchWindow.embedding_updated_at.is_not(None),
            )
        )
        .values(embedding_config_hash=None, embedding_updated_at=None)
    )
    stale_configs = invalidate_ready_semantic_indexes(
        session,
        reason=(
            "Semantic vector files were physically discarded after hard purge; "
            "rebuild the semantic index"
        ),
        source_kind="hard_purge",
        source_id=hashlib.sha256("\n".join(sorted(window_ids)).encode()).hexdigest(),
    )
    for config in configs:
        config.active = False
        config.status = "stale"
        config.window_count = 0
    return VectorDeletionResult(
        deleted_rows=discarded_rows,
        table_identities={},
        stale_config_count=stale_configs,
        memberships_removed=memberships_removed,
        store_discarded=discard_store,
    )


def _invalidate_topic_cluster_jobs(session: Session, *, reason: str) -> int:
    """Remove derived topic text and make every prior result visibly stale.

    Topic terms are copied text-derived artifacts stored in ``jobs.counters_json``. Sanitizing the
    result is required for physical purge, and doing the same on logical quarantine prevents a
    dashboard from continuing to reveal a hidden message through an old cluster label.
    """

    rows = list(session.scalars(select(Job).where(Job.kind == "topic_clusters")))
    now = datetime.now(UTC)
    invalidated = 0
    for row in rows:
        if row.status == "stale" and bool((row.counters_json or {}).get("invalidated")):
            continue
        row.status = "stale"
        row.stage = "Topic clusters invalidated"
        row.counters_json = {
            "invalidated": True,
            "reason": reason,
            "invalidated_at": now.isoformat(),
        }
        row.error_text = reason
        row.finished_at = now
        invalidated += 1
    return invalidated


def _reconcile_semantic_index_configs(
    session: Session, vector_result: VectorDeletionResult
) -> dict[str, int]:
    """Keep advertised semantic corpus sizes aligned with SQLite and LanceDB."""

    updated = 0
    stale = vector_result.stale_config_count
    for config in session.scalars(select(SemanticIndexConfig).order_by(SemanticIndexConfig.id)):
        sqlite_identity = semantic_window_identity(semantic_config_window_ids(session, config.id))
        lance_identity = vector_result.table_identities.get(config.vector_table)
        if lance_identity is not None and lance_identity != sqlite_identity:
            raise HardPurgeBlockedError(
                f"LanceDB table {config.vector_table} identities do not match SQLite membership"
            )
        lance_count = lance_identity.row_count if lance_identity is not None else None
        advertised_count = lance_count if lance_count is not None else sqlite_identity.row_count
        if int(config.window_count) != advertised_count:
            config.window_count = advertised_count
            updated += 1
    return {
        "semantic_configs_updated": updated,
        "semantic_configs_stale": stale,
        "semantic_memberships_removed": vector_result.memberships_removed,
        "semantic_store_discarded": int(vector_result.store_discarded),
    }


def _safe_workspace_path(layout: ArchiveLayout, relative_path: str) -> Path:
    pure = PurePosixPath(relative_path)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
        or "\x00" in relative_path
    ):
        raise HardPurgeBlockedError("Catalog contains an unsafe workspace path")
    root = Path(os.path.abspath(layout.root))
    candidate = root / Path(*pure.parts)
    if candidate == root or root not in candidate.parents:
        raise HardPurgeBlockedError("Workspace purge target escapes the archive root")
    return candidate


def _safe_blob_path(layout: ArchiveLayout, sha256: str) -> Path:
    if len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256):
        raise HardPurgeBlockedError("Catalog contains an unsafe blob identifier")
    return _safe_workspace_path(layout, (Path("blobs") / "sha256" / sha256[:2] / sha256).as_posix())


def _library_paths_for_blobs(layout: ArchiveLayout, blob_ids: set[str]) -> set[Path]:
    targets: set[Path] = set()
    if not blob_ids or not layout.library.is_dir():
        return targets
    for manifest in layout.library.rglob("*.ndjson"):
        if manifest.is_symlink() or not manifest.is_file():
            continue
        with manifest.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict) or record.get("blob_sha256") not in blob_ids:
                    continue
                relative = record.get("library_path")
                if isinstance(relative, str):
                    candidate = _safe_workspace_path(layout, f"library/{relative}")
                    if candidate.exists() or candidate.is_symlink():
                        targets.add(candidate)
    return targets


def _derivative_paths_for_blobs(layout: ArchiveLayout, blob_ids: set[str]) -> set[Path]:
    """Inventory catalogued and orphan derivative files by deterministic blob directory.

    Media derivatives live under ``thumbnails/<pipeline>/<prefix>/<sha256>/``.  Looking at the
    directory rather than only ``media_derivatives`` rows ensures an interrupted worker cannot
    leave an untracked thumbnail, poster, or waveform behind after the source blob is purged.
    Every traversed component is checked without following symlinks.
    """

    targets: set[Path] = set()
    if not blob_ids or not layout.thumbnails.exists():
        return targets
    thumbnails_stat = layout.thumbnails.lstat()
    if stat.S_ISLNK(thumbnails_stat.st_mode) or not stat.S_ISDIR(thumbnails_stat.st_mode):
        raise HardPurgeBlockedError("Media derivative root is not a safe directory")
    for sha256 in sorted(blob_ids):
        _safe_blob_path(layout, sha256)
        for pipeline_root in layout.thumbnails.iterdir():
            pipeline_stat = pipeline_root.lstat()
            if stat.S_ISLNK(pipeline_stat.st_mode) or not stat.S_ISDIR(pipeline_stat.st_mode):
                # The build lock and status manifest are regular files at the root.
                if stat.S_ISREG(pipeline_stat.st_mode):
                    continue
                raise HardPurgeBlockedError("Media derivative root contains an unsafe entry")
            blob_root = pipeline_root / sha256[:2] / sha256
            if not blob_root.exists() and not blob_root.is_symlink():
                continue
            blob_stat = blob_root.lstat()
            if stat.S_ISLNK(blob_stat.st_mode) or not stat.S_ISDIR(blob_stat.st_mode):
                raise HardPurgeBlockedError("Media derivative blob path is not a safe directory")
            for candidate in blob_root.iterdir():
                candidate_stat = candidate.lstat()
                if stat.S_ISLNK(candidate_stat.st_mode) or not stat.S_ISREG(candidate_stat.st_mode):
                    raise HardPurgeBlockedError(
                        "Media derivative blob directory contains an unsafe entry"
                    )
                targets.add(candidate)
    return targets


def _workspace_targets(layout: ArchiveLayout, session: Session, blob_ids: set[str]) -> set[Path]:
    targets = _library_paths_for_blobs(layout, blob_ids)
    targets.update(_derivative_paths_for_blobs(layout, blob_ids))
    if blob_ids and (
        layout.derivative_manifest.exists() or layout.derivative_manifest.is_symlink()
    ):
        targets.add(_safe_workspace_path(layout, "catalog/thumbnails/manifest.json"))
    for sha256 in blob_ids:
        targets.add(_safe_blob_path(layout, sha256))
    for chunk in _chunks(sorted(blob_ids)):
        for relative_path in session.scalars(
            select(MediaDerivative.relative_path).where(MediaDerivative.blob_sha256.in_(chunk))
        ):
            targets.add(_safe_workspace_path(layout, relative_path))
    safe_targets: set[Path] = set()
    for target in targets:
        if not target.exists() and not target.is_symlink():
            continue
        try:
            regular_file_stat_beneath(target, layout.root)
        except UnsafeArchivePathError as exc:
            raise HardPurgeBlockedError(
                f"Workspace purge target is not a regular file: {target}"
            ) from exc
        safe_targets.add(target)
    return safe_targets


def _stage_workspace_files(
    layout: ArchiveLayout,
    operation_id: str,
    targets: set[Path],
    *,
    scope: PurgeScope,
    tree_targets: set[Path],
    rollback_remove_paths: set[Path],
) -> tuple[Path, list[tuple[Path, Path]]]:
    staging = layout.trash / "purge-staging" / operation_id
    try:
        create_directory_beneath(staging, layout.root, mode=0o700, exist_ok=False)
    except UnsafeArchivePathError as exc:
        raise HardPurgeBlockedError(
            "A stale or unsafe purge staging directory requires manual review"
        ) from exc
    try:
        _write_precommit_rollback_journal(
            layout,
            staging,
            operation_id=operation_id,
            scope=scope,
            file_targets=targets,
            tree_targets=tree_targets,
            rollback_remove_paths=rollback_remove_paths,
        )
    except BaseException:
        # No live target has moved yet.  Remove the just-created operation directory so an
        # incomplete journal cannot permanently block startup recovery or the next retry.
        if _purge_entry_stat(layout, staging) is not None:
            _remove_tree_exact(staging, required_parent=layout.trash / "purge-staging")
            _fsync_directory(staging.parent)
        raise
    staged: list[tuple[Path, Path]] = []
    try:
        for source in sorted(targets):
            relative = source.relative_to(Path(os.path.abspath(layout.root)))
            destination = staging / relative
            # Register rollback intent before the rename.  The secure move can complete and then
            # report a verification/fsync error; in that case the staged inode must still be
            # restored rather than erased with an apparently empty staging directory.
            staged.append((source, destination))
            try:
                replace_entry_beneath(
                    source,
                    destination,
                    layout.root,
                    expected_kind="file",
                )
            except UnsafeArchivePathError as exc:
                raise HardPurgeBlockedError(
                    f"Workspace purge target became unsafe: {relative.as_posix()}"
                ) from exc
    except BaseException:
        _restore_and_verify_staged_files(layout, staging, operation_id, staged)
        if staging.exists():
            _remove_tree_exact(staging, required_parent=layout.trash / "purge-staging")
        raise
    return staging, staged


def _stage_workspace_tree(
    layout: ArchiveLayout,
    staging: Path,
    source: Path,
    staged: list[tuple[Path, Path]],
) -> None:
    if not source.exists() and not source.is_symlink():
        return
    _tree_revision_entries(layout, source)
    destination = staging / source.relative_to(Path(os.path.abspath(layout.root)))
    # See `_stage_workspace_files`: post-rename failures still require this pair for rollback.
    staged.append((source, destination))
    try:
        replace_entry_beneath(
            source,
            destination,
            layout.root,
            expected_kind="directory",
        )
    except UnsafeArchivePathError as exc:
        raise HardPurgeBlockedError(f"Workspace purge tree became unsafe: {source.name}") from exc


def _restore_staged_files(layout: ArchiveLayout, staged: Sequence[tuple[Path, Path]]) -> None:
    for original, staged_path in reversed(staged):
        if not staged_path.exists():
            continue
        staged_stat = archive_entry_stat_beneath(staged_path, layout.root)
        expected_kind: Literal["file", "directory"] = (
            "directory" if staged_stat is not None and stat.S_ISDIR(staged_stat.st_mode) else "file"
        )
        try:
            replace_entry_beneath(
                staged_path,
                original,
                layout.root,
                expected_kind=expected_kind,
                replace_existing=expected_kind == "file",
            )
        except UnsafeArchivePathError as exc:
            raise HardPurgeBlockedError(
                f"Purge rollback target became unsafe: {original.name}"
            ) from exc


def _restore_and_verify_staged_files(
    layout: ArchiveLayout,
    staging: Path,
    operation_id: str,
    staged: Sequence[tuple[Path, Path]],
) -> None:
    """Restore an aborted precommit move and prove every journaled inode is live."""

    journal = _read_precommit_rollback_journal(layout, staging, operation_id)
    _restore_staged_files(layout, staged)
    _verify_precommit_targets(layout, staging, journal, restored=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    """Make a promoted replacement tree durable before SQLite records cleanup_pending."""

    if root.is_symlink() or not root.is_dir():
        raise HardPurgeBlockedError(f"Promoted purge tree is unsafe: {root.name}")
    directories = [root]
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise HardPurgeBlockedError(f"Promoted purge tree contains a symlink: {path.name}")
        if path.is_dir():
            directories.append(path)
            continue
        if not path.is_file():
            raise HardPurgeBlockedError(
                f"Promoted purge tree contains a non-regular entry: {path.name}"
            )
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    for directory in reversed(directories):
        _fsync_directory(directory)
    _fsync_directory(root.parent)


def _fsync_file_and_parent(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise HardPurgeBlockedError(f"Promoted purge file is unsafe: {path.name}")
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _workspace_relative(layout: ArchiveLayout, path: Path) -> str:
    root = Path(os.path.abspath(layout.root))
    absolute = Path(os.path.abspath(path))
    if absolute == root or root not in absolute.parents:
        raise HardPurgeBlockedError("Purge rollback path escapes the archive root")
    return absolute.relative_to(root).as_posix()


def _path_identity(layout: ArchiveLayout, path: Path, *, tree: bool) -> dict[str, Any]:
    try:
        value = archive_entry_stat_beneath(path, layout.root)
    except UnsafeArchivePathError as exc:
        raise HardPurgeBlockedError(f"Purge rollback target is unsafe: {path.name}") from exc
    if value is None:
        raise HardPurgeBlockedError(f"Purge rollback target is missing: {path.name}")
    expected_mode = stat.S_ISDIR(value.st_mode) if tree else stat.S_ISREG(value.st_mode)
    if stat.S_ISLNK(value.st_mode) or not expected_mode:
        raise HardPurgeBlockedError(f"Purge rollback target is unsafe: {path.name}")
    return {
        "kind": "tree" if tree else "file",
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "size_bytes": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
    }


def _write_precommit_rollback_journal(
    layout: ArchiveLayout,
    staging: Path,
    *,
    operation_id: str,
    scope: PurgeScope,
    file_targets: set[Path],
    tree_targets: set[Path],
    rollback_remove_paths: set[Path],
) -> None:
    """Publish rollback intent durably before the first live filesystem rename."""

    target_rows = [
        {
            "path": _workspace_relative(layout, path),
            "identity": _path_identity(layout, path, tree=False),
        }
        for path in sorted(file_targets)
    ] + [
        {
            "path": _workspace_relative(layout, path),
            "identity": _path_identity(layout, path, tree=True),
        }
        for path in sorted(tree_targets)
    ]
    document = {
        "version": PURGE_ROLLBACK_JOURNAL_VERSION,
        "operation_id": operation_id,
        "scope": scope,
        "created_at": datetime.now(UTC).isoformat(),
        "targets": target_rows,
        "rollback_remove_paths": [
            _workspace_relative(layout, path) for path in sorted(rollback_remove_paths)
        ],
    }
    journal_path = staging / PURGE_ROLLBACK_JOURNAL_NAME
    descriptor, temporary_name = tempfile.mkstemp(prefix=".rollback-", dir=staging)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, journal_path)
        _fsync_directory(staging)
        _fsync_directory(staging.parent)
        # ``mkdir(parents=True)`` may have created the shared purge-staging directory as well as
        # this operation directory. Persist both directory entries before any live file is moved;
        # otherwise a power loss could make the rollback journal unreachable from ``trash``.
        _fsync_directory(staging.parent.parent)
        # `ensure_runtime_directories` may have linked `trash` without syncing the archive root.
        # Persist that final ancestor before moving the only live copy of any private inode.
        _fsync_directory(layout.root)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _read_precommit_rollback_journal(
    layout: ArchiveLayout, staging: Path, operation_id: str
) -> dict[str, Any]:
    journal_path = staging / PURGE_ROLLBACK_JOURNAL_NAME
    try:
        with open_regular_file_beneath(journal_path, layout.root) as handle:
            value = json.loads(handle.read().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, UnsafeArchivePathError) as exc:
        raise HardPurgeBlockedError("Purge precommit rollback journal is invalid") from exc
    if (
        not isinstance(value, dict)
        or value.get("version") != PURGE_ROLLBACK_JOURNAL_VERSION
        or value.get("operation_id") != operation_id
        or value.get("scope") not in {"normalized", "workspace"}
        or not isinstance(value.get("targets"), list)
        or not isinstance(value.get("rollback_remove_paths"), list)
    ):
        raise HardPurgeBlockedError("Purge precommit rollback journal has invalid metadata")
    # Resolve every path now so malformed or symlink-traversing input fails before mutation.
    for row in value["targets"]:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("path"), str)
            or not isinstance(row.get("identity"), dict)
            or row["identity"].get("kind") not in {"file", "tree"}
        ):
            raise HardPurgeBlockedError("Purge precommit rollback target is invalid")
        _safe_workspace_path(layout, row["path"])
    for relative in value["rollback_remove_paths"]:
        if not isinstance(relative, str):
            raise HardPurgeBlockedError("Purge precommit cleanup path is invalid")
        _safe_workspace_path(layout, relative)
    return value


def _identity_matches(layout: ArchiveLayout, path: Path, expected: dict[str, Any]) -> bool:
    try:
        value = archive_entry_stat_beneath(path, layout.root)
    except UnsafeArchivePathError:
        return False
    if value is None or stat.S_ISLNK(value.st_mode):
        return False
    kind_matches = (
        stat.S_ISDIR(value.st_mode)
        if expected.get("kind") == "tree"
        else stat.S_ISREG(value.st_mode)
    )
    return kind_matches and all(
        int(expected.get(key, -1)) == actual
        for key, actual in (
            ("device", int(value.st_dev)),
            ("inode", int(value.st_ino)),
            ("size_bytes", int(value.st_size)),
            ("mtime_ns", int(value.st_mtime_ns)),
        )
    )


def _staged_path(staging: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or "\x00" in relative:
        raise HardPurgeBlockedError("Purge rollback staging path is unsafe")
    candidate = staging / Path(*pure.parts)
    absolute_staging = Path(os.path.abspath(staging))
    absolute_candidate = Path(os.path.abspath(candidate))
    if absolute_candidate == absolute_staging or absolute_staging not in absolute_candidate.parents:
        raise HardPurgeBlockedError("Purge rollback staging path escapes its operation")
    return absolute_candidate


def _purge_entry_stat(layout: ArchiveLayout, path: Path) -> os.stat_result | None:
    try:
        return archive_entry_stat_beneath(path, layout.root)
    except UnsafeArchivePathError as exc:
        raise HardPurgeBlockedError(f"Purge path has unsafe ancestry: {path.name}") from exc


def _verify_precommit_targets(
    layout: ArchiveLayout,
    staging: Path,
    journal: dict[str, Any],
    *,
    restored: bool,
) -> None:
    for row in journal["targets"]:
        relative = str(row["path"])
        candidate = (
            _safe_workspace_path(layout, relative) if restored else _staged_path(staging, relative)
        )
        if not _identity_matches(layout, candidate, cast(dict[str, Any], row["identity"])):
            phase = "restored" if restored else "staged"
            raise HardPurgeBlockedError(
                f"Purge rollback {phase} target identity does not match: {relative}"
            )


def _recover_precommit_purge_staging_locked(layout: ArchiveLayout, engine: Engine) -> list[str]:
    """Restore filesystem mutations whose SQLite purge transaction never committed."""

    root = layout.trash / "purge-staging"
    root_stat = _purge_entry_stat(layout, root)
    if root_stat is None:
        return []
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise HardPurgeBlockedError("Purge staging root is unsafe")
    recovered: list[str] = []
    for staging in sorted(root.iterdir()):
        if staging.is_symlink() or not staging.is_dir():
            raise HardPurgeBlockedError("Unexpected entry in purge staging root")
        operation_id = staging.name
        with Session(engine) as session:
            operation = session.get(DeletionOperation, operation_id)
            status_value = operation.status if operation is not None else None
        if status_value == "cleanup_pending":
            # The catalog delete committed. The finalizer must erase, never restore, these bytes.
            continue
        if status_value == "purged":
            _remove_tree_exact(staging, required_parent=root)
            _fsync_directory(root)
            continue
        if status_value not in {"committed", "normalized_purged"}:
            raise HardPurgeBlockedError(
                "Purge staging has no matching recoverable or cleanup-pending operation"
            )
        recoverable_status = status_value
        journal = _read_precommit_rollback_journal(layout, staging, operation_id)
        # A replacement directory may exist (rebuilt library, copied/edited Lance tree, or a
        # concurrently recreated analytics cache). Remove it before moving the original inode
        # tree back into place.
        for row in sorted(
            journal["targets"],
            key=lambda item: len(PurePosixPath(str(item["path"])).parts),
            reverse=True,
        ):
            if row["identity"].get("kind") != "tree":
                continue
            staged_path = _staged_path(staging, str(row["path"]))
            staged_stat = _purge_entry_stat(layout, staged_path)
            if staged_stat is None:
                # The process may have died after publishing the journal but before this
                # particular rename. In that phase the original live tree is already correct.
                continue
            original = _safe_workspace_path(layout, str(row["path"]))
            original_stat = _purge_entry_stat(layout, original)
            if original_stat is not None:
                if stat.S_ISLNK(original_stat.st_mode) or not stat.S_ISDIR(original_stat.st_mode):
                    raise HardPurgeBlockedError("Purge rollback tree target became unsafe")
                _remove_tree_exact(original, required_parent=layout.root)
        for relative in journal["rollback_remove_paths"]:
            candidate = _safe_workspace_path(layout, str(relative))
            candidate_stat = _purge_entry_stat(layout, candidate)
            if candidate_stat is None:
                continue
            if stat.S_ISLNK(candidate_stat.st_mode) or not stat.S_ISREG(candidate_stat.st_mode):
                raise HardPurgeBlockedError("Purge rollback cleanup target became unsafe")
            try:
                unlink_regular_file_beneath(candidate, layout.root)
            except UnsafeArchivePathError as exc:
                raise HardPurgeBlockedError("Purge rollback cleanup target became unsafe") from exc
        for row in sorted(
            journal["targets"],
            key=lambda item: len(PurePosixPath(str(item["path"])).parts),
            reverse=True,
        ):
            relative = str(row["path"])
            staged_path = _staged_path(staging, relative)
            staged_stat = _purge_entry_stat(layout, staged_path)
            if staged_stat is None:
                # A crash can happen before every planned rename. An untouched live target must
                # still have the journaled identity; anything else requires manual review.
                live_path = _safe_workspace_path(layout, relative)
                if not _identity_matches(
                    layout,
                    live_path,
                    cast(dict[str, Any], row["identity"]),
                ):
                    raise HardPurgeBlockedError(
                        f"Unstaged purge rollback target changed unexpectedly: {relative}"
                    )
                continue
            if stat.S_ISLNK(staged_stat.st_mode):
                raise HardPurgeBlockedError("Purge rollback staged target became unsafe")
            original = _safe_workspace_path(layout, relative)
            expected_kind: Literal["file", "directory"] = (
                "directory" if row["identity"].get("kind") == "tree" else "file"
            )
            try:
                replace_entry_beneath(
                    staged_path,
                    original,
                    layout.root,
                    expected_kind=expected_kind,
                    replace_existing=expected_kind == "file",
                )
            except UnsafeArchivePathError as exc:
                raise HardPurgeBlockedError(
                    f"Purge rollback target became unsafe: {relative}"
                ) from exc
        _verify_precommit_targets(layout, staging, journal, restored=True)
        _remove_tree_exact(staging, required_parent=root)
        _fsync_directory(root)
        with immediate_session(engine) as session:
            operation = session.get(DeletionOperation, operation_id)
            if operation is None or operation.status != recoverable_status:
                raise StaleDeletionPreviewError(
                    "Purge operation status changed during precommit recovery"
                )
            recovered_at = datetime.now(UTC).isoformat()
            operation.error_text = None
            operation.purge_plan_json = {
                **(operation.purge_plan_json or {}),
                "precommit_recovery": {
                    "completed": True,
                    "completed_at": recovered_at,
                    "journal_version": PURGE_ROLLBACK_JOURNAL_VERSION,
                },
            }
            _audit(
                session,
                operation_id,
                action="purge_precommit_recovery",
                outcome="restored",
                scope=cast(PurgeScope, journal["scope"]),
                detail={
                    "journal_version": PURGE_ROLLBACK_JOURNAL_VERSION,
                    "completed_at": recovered_at,
                },
            )
        recovered.append(operation_id)
    return recovered


def _remove_tree_exact(path: Path, *, required_parent: Path) -> None:
    try:
        remove_directory_tree_beneath(path, required_parent, missing_ok=True)
    except UnsafeArchivePathError as exc:
        raise HardPurgeBlockedError("Refusing to remove an unsafe purge staging path") from exc


def _delete_nlp_rows(
    session: Session,
    selected: dict[EntityType, set[str]],
    resources: dict[str, set[str]],
) -> int:
    count = 0
    for target_type, target_ids in _physical_nlp_target_ids(selected, resources).items():
        for chunk in _chunks(sorted(target_ids)):
            result = session.execute(
                delete(NlpAnnotation).where(
                    NlpAnnotation.target_type == target_type,
                    NlpAnnotation.target_id.in_(chunk),
                )
            )
            count += _result_rowcount(result)
    return count


def _delete_search_feedback_and_retire_calibrations(
    session: Session,
    window_ids: set[str],
) -> dict[str, int]:
    """Remove human judgments whose target is being erased and retire dependent gates."""

    affected = {
        "relevance_feedback": 0,
        "conflict_calibrations_retired": 0,
    }
    if not window_ids:
        return affected
    selected_conflict_labels = 0
    for chunk in _chunks(sorted(window_ids)):
        result_ids = [value for window_id in chunk for value in (window_id, f"window:{window_id}")]
        affected["relevance_feedback"] += _result_rowcount(
            session.execute(
                delete(RelevanceFeedback).where(RelevanceFeedback.result_id.in_(result_ids))
            )
        )
        selected_conflict_labels += int(
            session.scalar(
                select(func.count(ConflictLabel.id)).where(ConflictLabel.window_id.in_(chunk))
            )
            or 0
        )
    if selected_conflict_labels:
        now = datetime.now(UTC)
        calibrations = list(session.scalars(select(ConflictCalibration)))
        for calibration in calibrations:
            calibration.status = "stale"
            calibration.aggregation_enabled = False
            calibration.provenance_json = {
                **(calibration.provenance_json or {}),
                "invalidated_at": now.isoformat(),
                "invalidation_reason": (
                    "A labeled search window was hard-purged; recalibrate from retained labels"
                ),
            }
            calibration.warnings_json = {
                **(calibration.warnings_json or {}),
                "hard_purge_invalidated": True,
            }
        affected["conflict_calibrations_retired"] = len(calibrations)
    return affected


def _garbage_collect_conversation_participants(
    session: Session,
    participant_ids: set[str],
) -> int:
    """Delete only participants formerly owned exclusively by purged conversations."""

    count = 0
    for chunk in _chunks(sorted(participant_ids)):
        count += _result_rowcount(
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
    return count


def _restore_quarantined_fts_rows(
    session: Session,
    selected: dict[EntityType, set[str]],
) -> None:
    """Put quarantined rows back in FTS immediately before their physical delete.

    The frozen catalog triggers remove a row from FTS whenever ``deleted_at`` changes and again
    whenever the normalized row is deleted.  A committed deletion therefore needs this short-lived
    reinsert so the second trigger operation has a matching index entry.  Both the insert and the
    following normalized delete happen in the same immediate transaction.
    """

    specifications: tuple[
        tuple[EntityType, type[Message] | type[ActivityRecord], str, tuple[Any, ...]], ...
    ] = (
        (
            "message",
            Message,
            "INSERT INTO message_fts("
            "rowid, sender_search, content_search, transcription_search"
            ") VALUES (?, ?, ?, ?)",
            (
                Message.pk,
                Message.sender_search,
                Message.content_search,
                Message.transcription_search,
            ),
        ),
        (
            "activity",
            ActivityRecord,
            "INSERT INTO activity_fts("
            "rowid, title_search, content_search, actor_search"
            ") VALUES (?, ?, ?, ?)",
            (
                ActivityRecord.pk,
                ActivityRecord.title_search,
                ActivityRecord.content_search,
                ActivityRecord.actor_search,
            ),
        ),
    )
    connection = session.connection()
    for entity_type, model, insert_sql, columns in specifications:
        for chunk in _chunks(sorted(selected[entity_type])):
            rows = session.execute(
                select(*columns).where(model.id.in_(chunk), model.deleted_at.is_not(None))
            ).all()
            if rows:
                connection.exec_driver_sql(insert_sql, [tuple(row) for row in rows])


def _delete_normalized_rows(
    session: Session,
    selected: dict[EntityType, set[str]],
    resources: dict[str, set[str]],
) -> dict[str, int]:
    star_targets = _normalized_hard_purge_star_targets(session, selected)
    feedback_result = purge_window_feedback_and_calibrations(
        session, resources["purge_search_window_ids"]
    )
    affected: dict[str, int] = {
        "stars": delete_stars_for_targets(session, star_targets),
        "nlp_annotations": _delete_nlp_rows(session, selected, resources),
        "nlp_hourly_rollups": (
            _result_rowcount(session.execute(delete(NlpHourlyRollup)))
            if any(selected.values())
            else 0
        ),
        "relevance_feedback": feedback_result.relevance_feedback_deleted,
        "rag_audits": feedback_result.rag_audits_deleted,
        "conflict_calibrations_scrubbed": feedback_result.conflict_calibrations_scrubbed,
    }
    affected_run_ids: set[str] = set()
    for chunk in _chunks(sorted(resources["purge_search_window_ids"])):
        affected_run_ids.update(
            session.scalars(
                select(ConflictWindowScore.run_id).where(ConflictWindowScore.window_id.in_(chunk))
            )
        )
    affected["conflict_aggregate_runs"] = 0
    for chunk in _chunks(sorted(affected_run_ids)):
        affected["conflict_aggregate_runs"] += _result_rowcount(
            session.execute(delete(ConflictAggregateRun).where(ConflictAggregateRun.id.in_(chunk)))
        )
    if feedback_result.relevance_feedback_deleted:
        for config in session.scalars(select(SemanticIndexConfig)):
            audit = dict(config.audit_json or {})
            audit.pop("semantic_ir_evaluation", None)
            audit["feedback_evaluation"] = {
                "status": "stale",
                "reason": "hard_purge_removed_feedback_target",
            }
            config.audit_json = audit
    for chunk in _chunks(sorted(resources["purge_search_window_ids"])):
        result = session.execute(delete(SearchWindow).where(SearchWindow.id.in_(chunk)))
        affected["search_windows"] = affected.get("search_windows", 0) + _result_rowcount(result)
    for chunk in _chunks(sorted(selected["message"])):
        session.execute(
            update(Message).where(Message.duplicate_of_id.in_(chunk)).values(duplicate_of_id=None)
        )
    _restore_quarantined_fts_rows(session, selected)
    for entity_type in ("attachment", "message", "activity", "conversation"):
        model = _ENTITY_MODELS[entity_type]
        count = 0
        for chunk in _chunks(sorted(selected[entity_type])):
            result = session.execute(delete(model).where(model.id.in_(chunk)))
            count += _result_rowcount(result)
        affected[entity_type] = count
    affected["participants"] = gc_conversation_candidate_participants(
        session, resources["conversation_participant_ids"]
    )
    return affected


def _selected_source_record_ids(session: Session, selected: dict[EntityType, set[str]]) -> set[str]:
    identifiers: set[str] = set()
    for chunk in _chunks(sorted(selected["message"])):
        identifiers.update(
            session.scalars(select(Message.source_record_id).where(Message.id.in_(chunk)))
        )
    for chunk in _chunks(sorted(selected["activity"])):
        identifiers.update(
            session.scalars(
                select(ActivityRecord.source_record_id).where(ActivityRecord.id.in_(chunk))
            )
        )
    return identifiers


def _json_contains_any_reference(value: Any, references: set[str]) -> bool:
    if isinstance(value, str):
        return value in references
    if isinstance(value, dict):
        return any(
            _json_contains_any_reference(key, references)
            or _json_contains_any_reference(item, references)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_json_contains_any_reference(item, references) for item in value)
    return False


def _delete_linked_integrity_anomalies(
    session: Session,
    *,
    selected: dict[EntityType, set[str]],
    resources: dict[str, set[str]],
    source_record_ids: set[str],
) -> int:
    references = set().union(*selected.values())
    references.update(resources["conversation_segment_ids"])
    references.update(resources["blob_ids"])
    references.update(source_record_ids)
    deleted = 0
    for anomaly in session.scalars(select(IntegrityAnomaly).order_by(IntegrityAnomaly.id)):
        if (
            anomaly.source_record_id in source_record_ids
            or anomaly.conversation_segment_id in resources["conversation_segment_ids"]
            or _json_contains_any_reference(anomaly.detail_json, references)
        ):
            session.delete(anomaly)
            deleted += 1
    return deleted


def _delete_source_records(session: Session, identifiers: set[str]) -> int:
    count = 0
    for chunk in _chunks(sorted(identifiers)):
        count += _result_rowcount(
            session.execute(delete(SourceRecord).where(SourceRecord.id.in_(chunk)))
        )
    return count


def _compact_catalog_after_purge(engine: Engine) -> None:
    """Discard SQLite history only after the purge transaction has committed."""

    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
        connection.exec_driver_sql("INSERT INTO message_fts(message_fts) VALUES ('optimize')")
        connection.exec_driver_sql("INSERT INTO activity_fts(activity_fts) VALUES ('optimize')")
        checkpoint = connection.exec_driver_sql("PRAGMA wal_checkpoint(TRUNCATE)").one()
        if int(checkpoint[0]) != 0:
            raise HardPurgeBlockedError("SQLite WAL checkpoint is busy; cleanup will retry")
        connection.exec_driver_sql("VACUUM")
        final_checkpoint = connection.exec_driver_sql("PRAGMA wal_checkpoint(TRUNCATE)").one()
        if int(final_checkpoint[0]) != 0:
            raise HardPurgeBlockedError("SQLite final WAL checkpoint is busy; cleanup will retry")


def _verify_pending_vector_state(
    layout: ArchiveLayout, engine: Engine, journal: dict[str, Any]
) -> None:
    expected_counts_value = journal.get("vector_table_row_counts", {})
    if not isinstance(expected_counts_value, dict) or any(
        not isinstance(key, str) or not isinstance(value, int) or value < 0
        for key, value in expected_counts_value.items()
    ):
        raise HardPurgeBlockedError("Pending purge contains invalid vector verification state")
    expected_counts = cast(dict[str, int], expected_counts_value)
    expected_identities_value = journal.get("vector_table_identities", {})
    if not isinstance(expected_identities_value, dict):
        raise HardPurgeBlockedError("Pending purge contains invalid vector identity state")
    expected_identities: dict[str, SemanticWindowIdentity] = {}
    for table_name, raw_identity in expected_identities_value.items():
        if not isinstance(table_name, str) or not isinstance(raw_identity, dict):
            raise HardPurgeBlockedError("Pending purge contains invalid vector identity state")
        row_count = raw_identity.get("row_count")
        unique_count = raw_identity.get("unique_count")
        identity_sha256 = raw_identity.get("identity_sha256")
        if (
            not isinstance(row_count, int)
            or isinstance(row_count, bool)
            or row_count < 0
            or not isinstance(unique_count, int)
            or isinstance(unique_count, bool)
            or unique_count < 0
            or unique_count != row_count
            or not isinstance(identity_sha256, str)
            or len(identity_sha256) != 64
            or any(character not in "0123456789abcdef" for character in identity_sha256)
        ):
            raise HardPurgeBlockedError("Pending purge contains invalid vector identity state")
        expected_identities[table_name] = SemanticWindowIdentity(
            row_count=row_count,
            unique_count=unique_count,
            identity_sha256=identity_sha256,
        )
    if not journal.get("vector_store_rewritten"):
        if expected_counts or expected_identities:
            raise HardPurgeBlockedError("Pending purge has unexpected vector verification rows")
        return
    _assert_live_vector_store_empty(layout)
    with Session(engine) as session:
        registered, available, actual_identities, error = _vector_identity_inventory(
            layout, session
        )
        sqlite_identities = {
            config.vector_table: semantic_window_identity(
                semantic_config_window_ids(session, config.id)
            )
            for config in session.scalars(
                select(SemanticIndexConfig).order_by(SemanticIndexConfig.id)
            )
        }
    if error is not None:
        raise HardPurgeBlockedError(f"Pending purge vector verification failed: {error}")
    if available - registered:
        raise HardPurgeBlockedError("Pending purge found an unregistered LanceDB table")
    actual_counts = {
        table_name: identity.row_count for table_name, identity in actual_identities.items()
    }
    if actual_counts != expected_counts:
        raise HardPurgeBlockedError(
            "Pending purge LanceDB row counts do not match the committed cleanup journal"
        )
    if expected_identities and actual_identities != expected_identities:
        raise HardPurgeBlockedError(
            "Pending purge LanceDB identities do not match the committed cleanup journal"
        )
    for table_name, actual_identity in actual_identities.items():
        if actual_identity.row_count != actual_identity.unique_count:
            raise HardPurgeBlockedError(
                f"Pending purge LanceDB table {table_name} contains duplicate identities"
            )
        if actual_identity != sqlite_identities.get(table_name):
            raise HardPurgeBlockedError(
                "Pending purge LanceDB identities do not match the committed SQLite catalog"
            )


def _invalidated_manifest_paths_from_journal(
    layout: ArchiveLayout,
    journal: dict[str, Any],
) -> tuple[Path, ...]:
    raw_paths = journal.get("invalidated_manifest_relative_paths", [])
    if not isinstance(raw_paths, list) or any(not isinstance(value, str) for value in raw_paths):
        raise HardPurgeBlockedError("Pending purge contains invalidated manifest paths")
    relative_paths = cast(list[str], raw_paths)
    if relative_paths != sorted(set(relative_paths)):
        raise HardPurgeBlockedError("Pending purge contains invalidated manifest paths")
    result: list[Path] = []
    for relative_path in relative_paths:
        parts = PurePosixPath(relative_path).parts
        if len(parts) != 4 or parts[:2] != ("trash", "deletions") or parts[3] != "manifest.json":
            raise HardPurgeBlockedError("Pending purge contains invalidated manifest paths")
        try:
            expected = _manifest_path(layout, parts[2]).relative_to(layout.root).as_posix()
        except DeletionError as exc:
            raise HardPurgeBlockedError(
                "Pending purge contains invalidated manifest paths"
            ) from exc
        if relative_path != expected:
            raise HardPurgeBlockedError("Pending purge contains invalidated manifest paths")
        result.append(_safe_workspace_path(layout, relative_path))
    return tuple(result)


def _finalize_pending_hard_purge_locked(
    layout: ArchiveLayout, engine: Engine, operation_id: str
) -> MutationResult:
    """Finish an already-authorized hard purge idempotently.

    The normalized delete transaction records all cleanup inputs before returning. A crash at any
    later instruction therefore leaves ``cleanup_pending`` rather than a false ``purged`` state,
    and this routine can safely repeat missing-file removal and SQLite compaction.
    """

    with Session(engine) as session:
        operation = session.get(DeletionOperation, operation_id)
        if operation is None:
            raise DeletionError("Deletion operation not found")
        if operation.status in {"purged", "normalized_purged"}:
            affected = cast(dict[str, int], operation.purge_plan_json.get("affected", {}))
            audit_id = str(operation.purge_plan_json.get("audit_id", ""))
            return MutationResult(operation_id, operation.status, affected, audit_id)
        if operation.status != "cleanup_pending":
            raise DeletionError("Hard purge is not awaiting cleanup")
        journal = dict(operation.purge_plan_json)
        manifest_relative_path = journal.get("manifest_relative_path")
        if manifest_relative_path is not None and not isinstance(manifest_relative_path, str):
            raise HardPurgeBlockedError("Pending purge contains an invalid manifest path")
        try:
            canonical_manifest_relative_path = _canonical_manifest_relative_path(layout, operation)
        except DeletionError as exc:
            raise HardPurgeBlockedError("Pending purge manifest path is invalid") from exc
        if manifest_relative_path != canonical_manifest_relative_path:
            raise HardPurgeBlockedError("Pending purge manifest path is invalid")
        affected = cast(dict[str, int], journal.get("affected", {}))
        selection_sha256 = operation.selection_sha256
        scope_value = journal.get("scope")
        if scope_value not in {"normalized", "workspace"}:
            raise HardPurgeBlockedError("Pending purge contains an invalid scope")
        scope = cast(PurgeScope, scope_value)

    _verify_pending_vector_state(layout, engine, journal)
    staging = layout.trash / "purge-staging" / operation_id
    manifest_path = _safe_workspace_path(layout, canonical_manifest_relative_path)
    invalidated_manifest_paths = _invalidated_manifest_paths_from_journal(layout, journal)
    if scope == "workspace" and _purge_entry_stat(layout, manifest_path) is not None:
        # The authorized manifest inode is already in staging. A live entry appearing after the
        # commit is unbound new data; never unlink it under authority derived from the old inode.
        raise HardPurgeBlockedError("Pending purge restore manifest was unexpectedly recreated")
    if any(_purge_entry_stat(layout, path) is not None for path in invalidated_manifest_paths):
        raise HardPurgeBlockedError("Pending purge invalidated manifest was unexpectedly recreated")
    if _purge_entry_stat(layout, staging) is not None:
        _remove_tree_exact(staging, required_parent=layout.trash / "purge-staging")
    if scope == "workspace" and _purge_entry_stat(layout, manifest_path.parent) is not None:
        with suppress(OSError):
            manifest_path.parent.rmdir()
    if scope == "normalized":
        receipt_value = journal.get("workspace_upgrade_state")
        try:
            receipt = WorkspaceUpgradeStateReceipt.from_json(receipt_value)
            load_workspace_upgrade_state(
                layout,
                layout.root / receipt.relative_path,
                receipt=receipt,
                expected_operation_id=operation_id,
                expected_selection_sha256=selection_sha256,
                expected_purge_revision=receipt.purge_revision,
            )
        except DeletionUpgradeStateError as exc:
            raise HardPurgeBlockedError(
                "Normalized purge workspace-upgrade state is missing or invalid"
            ) from exc
    analytics_stat = _purge_entry_stat(layout, layout.analytics_cache)
    if analytics_stat is not None:
        if stat.S_ISLNK(analytics_stat.st_mode) or not stat.S_ISDIR(analytics_stat.st_mode):
            raise HardPurgeBlockedError("Analytics cache became unsafe during purge cleanup")
        _remove_tree_exact(layout.analytics_cache, required_parent=layout.root)
    _compact_catalog_after_purge(engine)
    if (
        _purge_entry_stat(layout, staging) is not None
        or (scope == "workspace" and _purge_entry_stat(layout, manifest_path) is not None)
        or any(_purge_entry_stat(layout, path) is not None for path in invalidated_manifest_paths)
    ):
        raise HardPurgeBlockedError("Pending purge cleanup could not remove recoverable copies")

    with immediate_session(engine) as session:
        operation = session.get(DeletionOperation, operation_id)
        if operation is None or operation.status != "cleanup_pending":
            raise StaleDeletionPreviewError("Pending purge status changed during finalization")
        completed_at = datetime.now(UTC)
        final_status = "purged" if scope == "workspace" else "normalized_purged"
        operation.status = final_status
        operation.purged_at = completed_at
        operation.purge_token_hash = None
        if scope == "workspace":
            operation.manifest_relative_path = None
        audit_id = _audit(
            session,
            operation_id,
            action="purge_finalize",
            outcome="completed",
            scope=scope,
            detail={"affected": affected, "recovered": bool(journal.get("cleanup_attempted"))},
        )
        operation.purge_plan_json = {
            **journal,
            "completed": True,
            "cleanup_pending": False,
            "completed_at": completed_at.isoformat(),
            "audit_id": audit_id,
        }
        operation.error_text = None
    return MutationResult(operation_id, final_status, affected, audit_id)


def recover_pending_hard_purges(layout: ArchiveLayout, engine: Engine) -> list[MutationResult]:
    """Recover precommit staging and resume cleanup-pending purges at startup."""

    results: list[MutationResult] = []
    with (
        deletion_lock(layout),
        purge_source_publisher_locks(layout) as _publisher_leases,
        purge_worker_locks(layout),
        vector_store_lock(layout),
        analytics_cache_lock(layout, exclusive=True),
    ):
        _recover_precommit_purge_staging_locked(layout, engine)
        with Session(engine) as session:
            operation_ids = list(
                session.scalars(
                    select(DeletionOperation.id)
                    .where(DeletionOperation.status == "cleanup_pending")
                    .order_by(DeletionOperation.created_at)
                )
            )
        for operation_id in operation_ids:
            results.append(_finalize_pending_hard_purge_locked(layout, engine, operation_id))
    return results


def _execute_normalized_workspace_upgrade_locked(
    layout: ArchiveLayout,
    engine: Engine,
    operation_id: str,
    *,
    deletion_lock_lease: DeletionLockLease,
    source_lock: SourceImportPublicationLease,
    purge_token: str,
    purge_revision: str,
) -> MutationResult:
    """Finish full workspace erasure from an immutable normalized-purge handoff."""

    staging: Path | None = None
    staged: list[tuple[Path, Path]] = []
    created_paths: set[Path] = set()
    promoted_trees: set[Path] = set()
    try:
        with immediate_session(engine) as session:
            operation = session.get(DeletionOperation, operation_id)
            if operation is None or operation.status != "normalized_purged":
                raise DeletionError("Only normalized-purged quarantine can be upgraded")
            _verify_token(operation.purge_token_hash, purge_token)
            if operation.purge_revision != purge_revision:
                raise StaleDeletionPreviewError("Hard-purge revision does not match its plan")
            if operation.purge_plan_json.get("scope") != "workspace":
                raise StaleDeletionPreviewError("Hard-purge scope does not match its plan")
            items = _stored_items(session, operation_id)
            impact, blockers, current_revision = _normalized_workspace_purge_state(
                layout, session, operation, items
            )
            if current_revision != purge_revision:
                raise StaleDeletionPreviewError(
                    "Workspace changed after the normalized-purge upgrade was planned"
                )
            if blockers:
                raise HardPurgeBlockedError("Hard purge is blocked: " + ", ".join(sorted(blockers)))
            receipt = WorkspaceUpgradeStateReceipt.from_json(
                operation.purge_plan_json.get("workspace_upgrade_state")
            )
            saved = load_workspace_upgrade_state(
                layout,
                layout.root / receipt.relative_path,
                receipt=receipt,
                expected_operation_id=operation.id,
                expected_selection_sha256=operation.selection_sha256,
                expected_purge_revision=receipt.purge_revision,
            )
            resources = _saved_resources_as_sets(saved.resources)
            physical_blobs = resources["physically_exclusive_blob_ids"]
            derivative_inventory = inventory_derivative_tree(layout, session, physical_blobs)
            if derivative_inventory.blockers:
                raise HardPurgeBlockedError(
                    "Hard purge is blocked: " + ", ".join(sorted(derivative_inventory.blockers))
                )
            if operation.manifest_relative_path is None:
                raise HardPurgeBlockedError("Normalized purge selection manifest is missing")
            manifest_target = _safe_workspace_path(layout, operation.manifest_relative_path)
            state_target = layout.root / receipt.relative_path
            targets = (
                _workspace_targets(layout, session, physical_blobs)
                | raw_rewrite_live_targets(layout, saved.raw_plan)
                | set(derivative_inventory.selected_paths)
                | {manifest_target, state_target}
            )
            library_root = Path(os.path.abspath(layout.library))
            targets = {
                target
                for target in targets
                if library_root != Path(os.path.abspath(target))
                and library_root not in Path(os.path.abspath(target)).parents
            }
            library_was_present = bool(impact["filesystem_library_present"])
            conflicts_were_present = bool(impact["conflict_holding_files"])
            analytics_cache_was_present = bool(impact["analytics_cache_files"])
            vector_was_present = bool(impact["vector_store_present"])
            tree_targets = {
                path
                for path, present in (
                    (layout.library, library_was_present),
                    (layout.conflicts, conflicts_were_present),
                    (layout.analytics_cache, analytics_cache_was_present),
                    (layout.vectors, vector_was_present),
                )
                if present
            }
            rollback_remove_paths = {
                layout.blob_path(rewrite.new_sha256)
                for rewrite in saved.raw_plan.rewrites
                if not layout.blob_path(rewrite.new_sha256).exists()
                and not layout.blob_path(rewrite.new_sha256).is_symlink()
            }
            staging, staged = _stage_workspace_files(
                layout,
                operation_id,
                targets,
                scope="workspace",
                tree_targets=tree_targets,
                rollback_remove_paths=rollback_remove_paths,
            )
            for tree_target in sorted(tree_targets):
                _stage_workspace_tree(layout, staging, tree_target, staged)
            journal = _read_precommit_rollback_journal(layout, staging, operation_id)
            _verify_precommit_targets(layout, staging, journal, restored=False)

            created_paths.update(publish_raw_rewrite_files(layout, saved.raw_plan))
            apply_raw_rewrite_catalog(session, layout, saved.raw_plan)
            vector_result = _delete_vectors(
                layout,
                session,
                set(),
                discard_store=vector_was_present,
            )
            affected: dict[str, int] = {
                "workspace_upgrade": 1,
                "blobs": 0,
            }
            affected.update(_reconcile_semantic_index_configs(session, vector_result))
            blob_ids_to_delete = physical_blobs | set(saved.raw_plan.obsolete_blob_ids)
            affected["stars"] = affected.get("stars", 0) + delete_stars_for_targets(
                session,
                _blob_star_targets(blob_ids_to_delete),
            )
            for chunk in _chunks(sorted(blob_ids_to_delete)):
                affected["blobs"] += _result_rowcount(
                    session.execute(delete(Blob).where(Blob.sha256.in_(chunk)))
                )
            if physical_blobs and any(
                original == layout.derivative_manifest for original, _staged in staged
            ):
                write_status_manifest(layout, engine, catalog_session=session)
            if library_was_present:
                build_filesystem_library(
                    layout,
                    engine,
                    catalog_session=session,
                    lock_held=True,
                    source_lock=source_lock,
                    deletion_lock_lease=deletion_lock_lease,
                    deletion_operation_id=operation.id,
                )
                promoted_trees.add(layout.library)
            durable_files = set(created_paths)
            durable_files.update(
                path for path in raw_rewrite_live_targets(layout, saved.raw_plan) if path.exists()
            )
            if layout.derivative_manifest.is_file():
                durable_files.add(layout.derivative_manifest)
            for durable_file in sorted(durable_files):
                _fsync_file_and_parent(durable_file)
            for promoted_tree in sorted(promoted_trees):
                _fsync_tree(promoted_tree)
            session.execute(delete(DeletionItem).where(DeletionItem.operation_id == operation_id))
            operation.status = "cleanup_pending"
            operation.selector_json = {"purged": True}
            operation.commit_token_hash = None
            operation.purge_plan_json = {
                **operation.purge_plan_json,
                "scope": "workspace",
                "completed": False,
                "cleanup_pending": True,
                "cleanup_attempted": False,
                "impact": impact,
                "affected": affected,
                "manifest_relative_path": operation.manifest_relative_path,
                "staged_workspace_files": len(staged),
                "deleted_vector_rows": vector_result.deleted_rows,
                "vector_table_row_counts": vector_result.table_row_counts,
                "vector_table_identities": {
                    table_name: asdict(identity)
                    for table_name, identity in vector_result.table_identities.items()
                },
                "raw_source_rewrites": len(saved.raw_plan.rewrites),
                "library_rebuilt": library_was_present,
                "conflict_holding_removed": conflicts_were_present,
                "analytics_cache_removed": analytics_cache_was_present,
                "vector_store_rewritten": vector_was_present,
            }
            _audit(
                session,
                operation_id,
                action="purge",
                outcome="cleanup_pending",
                scope="workspace",
                detail=operation.purge_plan_json,
            )
    except BaseException as exc:
        try:
            with Session(engine) as state_session:
                durable_status = state_session.scalar(
                    select(DeletionOperation.status).where(DeletionOperation.id == operation_id)
                )
        except BaseException as state_exc:
            raise exc from state_exc
        if durable_status == "cleanup_pending":
            raise
        for promoted in sorted(promoted_trees, key=lambda value: len(value.parts), reverse=True):
            if promoted.exists():
                _remove_tree_exact(promoted, required_parent=layout.root)
        for created_path in created_paths:
            if _purge_entry_stat(layout, created_path) is not None:
                unlink_regular_file_beneath(created_path, layout.root)
        if staging is not None:
            _restore_and_verify_staged_files(layout, staging, operation_id, staged)
            if _purge_entry_stat(layout, staging) is not None:
                _remove_tree_exact(staging, required_parent=layout.trash / "purge-staging")
        if isinstance(exc, (RawSourceRewriteError, DeletionUpgradeStateError)):
            raise HardPurgeBlockedError(str(exc)) from exc
        raise

    try:
        return _finalize_pending_hard_purge_locked(layout, engine, operation_id)
    except BaseException:
        with immediate_session(engine) as session:
            operation = session.get(DeletionOperation, operation_id)
            if operation is not None and operation.status == "cleanup_pending":
                operation.error_text = "Hard-purge cleanup is pending retry"
                operation.purge_plan_json = {
                    **operation.purge_plan_json,
                    "cleanup_attempted": True,
                }
        raise


def execute_hard_purge(
    layout: ArchiveLayout,
    engine: Engine,
    operation_id: str,
    *,
    scope: PurgeScope,
    purge_token: str,
    purge_revision: str,
    confirmation: str,
) -> MutationResult:
    if confirmation != operation_id:
        raise InvalidDeletionTokenError("Purge confirmation must equal the operation ID")
    with (
        deletion_lock(layout) as deletion_lease,
        purge_source_publisher_locks(layout) as publisher_leases,
        purge_worker_locks(layout),
        vector_store_lock(layout),
        analytics_cache_lock(layout, exclusive=True),
    ):
        # A prior process can die after staging live files but before SQLite commits the physical
        # delete. Recover that durable rollback intent before attempting a fresh execution.
        _recover_precommit_purge_staging_locked(layout, engine)
        with Session(engine) as session:
            other_pending_ids = list(
                session.scalars(
                    select(DeletionOperation.id)
                    .where(
                        DeletionOperation.status == "cleanup_pending",
                        DeletionOperation.id != operation_id,
                    )
                    .order_by(DeletionOperation.created_at)
                )
            )
        for pending_id in other_pending_ids:
            _finalize_pending_hard_purge_locked(layout, engine, pending_id)
        with Session(engine) as session:
            existing = session.get(DeletionOperation, operation_id)
            if existing is not None and existing.status == "cleanup_pending":
                _verify_token(existing.purge_token_hash, purge_token)
                if existing.purge_revision != purge_revision:
                    raise StaleDeletionPreviewError("Hard-purge revision does not match its plan")
                if existing.purge_plan_json.get("scope") != scope:
                    raise StaleDeletionPreviewError("Hard-purge scope does not match its plan")
                return _finalize_pending_hard_purge_locked(layout, engine, operation_id)
            normalized_upgrade = existing is not None and existing.status == "normalized_purged"
        if normalized_upgrade:
            if scope != "workspace":
                raise DeletionError(
                    "A normalized-purged operation can only be upgraded to workspace"
                )
            return _execute_normalized_workspace_upgrade_locked(
                layout,
                engine,
                operation_id,
                deletion_lock_lease=deletion_lease,
                source_lock=publisher_leases.source_import,
                purge_token=purge_token,
                purge_revision=purge_revision,
            )

        staging: Path | None = None
        staged: list[tuple[Path, Path]] = []
        created_paths: set[Path] = set()
        promoted_trees: set[Path] = set()
        try:
            with immediate_session(engine) as session:
                operation = session.get(DeletionOperation, operation_id)
                if operation is None or operation.status != "committed":
                    raise DeletionError("Only committed quarantine can be hard-purged")
                _verify_token(operation.purge_token_hash, purge_token)
                if operation.purge_revision != purge_revision:
                    raise StaleDeletionPreviewError("Hard-purge revision does not match its plan")
                if operation.purge_plan_json.get("scope") != scope:
                    raise StaleDeletionPreviewError("Hard-purge scope does not match its plan")
                items = _stored_items(session, operation_id)
                impact, blockers, current_revision = _purge_state(
                    layout, session, operation, items, scope
                )
                if current_revision != purge_revision:
                    raise StaleDeletionPreviewError(
                        "Workspace or catalog changed after hard-purge planning"
                    )
                if blockers:
                    raise HardPurgeBlockedError(
                        "Hard purge is blocked: " + ", ".join(sorted(blockers))
                    )
                selected = _items_as_selected(items)
                resources = _selection_resources(session, selected)
                overlap_inventory = inventory_deletion_overlaps(
                    session,
                    current_operation_id=operation.id,
                    selected_entity_ids={str(key): value for key, value in selected.items()},
                )
                correction_inventory = inventory_conversation_corrections(
                    session,
                    conversation_ids=selected["conversation"],
                    segment_ids=resources["conversation_segment_ids"],
                )
                if operation.manifest_relative_path is None:
                    raise HardPurgeBlockedError("Deletion restore manifest is missing")
                manifest_relative_path = operation.manifest_relative_path
                manifest_target = _safe_workspace_path(layout, manifest_relative_path)
                source_record_ids = _selected_source_record_ids(session, selected)
                raw_plan = build_raw_rewrite_plan(
                    layout,
                    session,
                    selected,
                    binary_blob_ids=resources["blob_ids"],
                    binary_source_file_ids={
                        int(value) for value in resources["binary_source_file_ids"]
                    },
                )
                workspace_upgrade_manifest_baseline = (
                    _source_manifest_content_revision(layout)
                    if scope == "normalized" and raw_plan.rewrites
                    else {}
                )
                upgrade_path = manifest_target.parent / "workspace-upgrade.json"
                upgrade_receipt: WorkspaceUpgradeStateReceipt | None = None
                targets = {manifest_target} if scope == "workspace" else set()
                targets.update(
                    _safe_workspace_path(layout, relative_path)
                    for relative_path in overlap_inventory.invalidatable_manifest_paths
                )
                if scope == "workspace":
                    targets |= _workspace_targets(
                        layout, session, resources["physically_exclusive_blob_ids"]
                    ) | raw_rewrite_live_targets(layout, raw_plan)
                    library_root = Path(os.path.abspath(layout.library))
                    targets = {
                        target
                        for target in targets
                        if library_root != Path(os.path.abspath(target))
                        and library_root not in Path(os.path.abspath(target)).parents
                    }
                library_was_present = scope == "workspace" and bool(
                    impact["filesystem_library_present"]
                )
                conflicts_were_present = scope == "workspace" and bool(
                    impact["conflict_holding_files"]
                )
                analytics_cache_was_present = bool(impact["analytics_cache_files"])
                vector_was_present = bool(impact["vector_store_present"])
                tree_targets = {
                    path
                    for path, present in (
                        (layout.library, library_was_present),
                        (layout.conflicts, conflicts_were_present),
                        (layout.analytics_cache, analytics_cache_was_present),
                        (layout.vectors, vector_was_present),
                    )
                    if present
                }
                rollback_remove_paths = {
                    layout.blob_path(rewrite.new_sha256)
                    for rewrite in raw_plan.rewrites
                    if not layout.blob_path(rewrite.new_sha256).exists()
                    and not layout.blob_path(rewrite.new_sha256).is_symlink()
                }
                if scope == "normalized" and not upgrade_path.exists():
                    rollback_remove_paths.add(upgrade_path)
                staging, staged = _stage_workspace_files(
                    layout,
                    operation_id,
                    targets,
                    scope=scope,
                    tree_targets=tree_targets,
                    rollback_remove_paths=rollback_remove_paths,
                )
                for tree_target in sorted(tree_targets):
                    _stage_workspace_tree(layout, staging, tree_target, staged)
                journal = _read_precommit_rollback_journal(layout, staging, operation_id)
                _verify_precommit_targets(layout, staging, journal, restored=False)

                if scope == "normalized":
                    saved_resources = SavedDeletionResources.from_selection_resources(
                        resources,
                        source_record_ids=source_record_ids,
                    )
                    if not upgrade_path.exists() and not upgrade_path.is_symlink():
                        # Register the create-only target before publication. The publisher can
                        # durably link the file and still report a post-publication fsync error.
                        created_paths.add(upgrade_path)
                    upgrade_receipt = publish_workspace_upgrade_state(
                        layout,
                        upgrade_path,
                        operation_id=operation_id,
                        selection_sha256=operation.selection_sha256,
                        purge_revision=purge_revision,
                        raw_plan=raw_plan,
                        resources=saved_resources,
                    )

                # Never copy a staged Lance tree back. Lance delete/optimize can leave selected
                # plaintext and vector bytes in the sole current data/index fragments. The old
                # tree stays in purge staging until post-commit cleanup securely removes it;
                # `_delete_vectors` clears memberships and requires a clean semantic rebuild.

                if scope == "workspace":
                    created_paths.update(publish_raw_rewrite_files(layout, raw_plan))
                    apply_raw_rewrite_catalog(session, layout, raw_plan)

                vector_result = _delete_vectors(
                    layout,
                    session,
                    resources["purge_search_window_ids"],
                    discard_store=vector_was_present,
                )
                affected = _delete_normalized_rows(session, selected, resources)
                affected["restore_states_requiring_global_rebuild"] = (
                    _mark_other_restore_states_for_global_rebuild(
                        session,
                        invalidating_operation_id=operation_id,
                    )
                )
                affected["overlapping_deletion_operations_invalidated"] = (
                    invalidate_overlapping_deletion_operations(session, overlap_inventory)
                )
                affected["conversation_corrections_scrubbed"] = (
                    scrub_reverted_conversation_corrections(session, correction_inventory)
                )
                affected["blob_job_checkpoints_scrubbed"] = scrub_blob_linked_job_checkpoints(
                    session,
                    resources["physically_exclusive_blob_ids"],
                ).scrubbed_count
                affected.update(_reconcile_semantic_index_configs(session, vector_result))
                affected["topic_cluster_jobs_invalidated"] = _invalidate_topic_cluster_jobs(
                    session, reason="Archive rows were hard-purged; rebuild topic clusters"
                )
                affected["integrity_anomalies"] = _delete_linked_integrity_anomalies(
                    session,
                    selected=selected,
                    resources=resources,
                    source_record_ids=source_record_ids,
                )
                affected["source_records"] = _delete_source_records(session, source_record_ids)
                if scope == "workspace":
                    blob_ids_to_delete = resources["physically_exclusive_blob_ids"] | set(
                        raw_plan.obsolete_blob_ids
                    )
                    affected["stars"] = affected.get("stars", 0) + delete_stars_for_targets(
                        session,
                        _blob_star_targets(blob_ids_to_delete),
                    )
                    for chunk in _chunks(sorted(blob_ids_to_delete)):
                        result = session.execute(delete(Blob).where(Blob.sha256.in_(chunk)))
                        affected["blobs"] = affected.get("blobs", 0) + _result_rowcount(result)
                    if resources["physically_exclusive_blob_ids"] and any(
                        original == layout.derivative_manifest for original, _staged in staged
                    ):
                        write_status_manifest(layout, engine, catalog_session=session)
                    if library_was_present:
                        build_filesystem_library(
                            layout,
                            engine,
                            catalog_session=session,
                            lock_held=True,
                            source_lock=publisher_leases.source_import,
                            deletion_lock_lease=deletion_lease,
                            deletion_operation_id=operation.id,
                        )
                        promoted_trees.add(layout.library)
                durable_files = set(created_paths)
                if scope == "workspace":
                    durable_files.update(
                        path for path in raw_rewrite_live_targets(layout, raw_plan) if path.exists()
                    )
                    if layout.derivative_manifest.is_file():
                        durable_files.add(layout.derivative_manifest)
                for durable_file in sorted(durable_files):
                    _fsync_file_and_parent(durable_file)
                for promoted_tree in sorted(promoted_trees):
                    _fsync_tree(promoted_tree)
                if scope == "workspace":
                    session.execute(
                        delete(DeletionItem).where(DeletionItem.operation_id == operation_id)
                    )
                operation.status = "cleanup_pending"
                if scope == "workspace":
                    operation.selector_json = {"purged": True}
                operation.commit_token_hash = None
                operation.purge_plan_json = {
                    "scope": scope,
                    "completed": False,
                    "cleanup_pending": True,
                    "cleanup_attempted": False,
                    "impact": impact,
                    "affected": affected,
                    "manifest_relative_path": manifest_relative_path,
                    "staged_workspace_files": len(staged),
                    "deleted_vector_rows": vector_result.deleted_rows,
                    "vector_table_row_counts": vector_result.table_row_counts,
                    "vector_table_identities": {
                        table_name: asdict(identity)
                        for table_name, identity in vector_result.table_identities.items()
                    },
                    "raw_source_rewrites": len(raw_plan.rewrites),
                    "library_rebuilt": library_was_present,
                    "conflict_holding_removed": conflicts_were_present,
                    "analytics_cache_removed": analytics_cache_was_present,
                    "vector_store_rewritten": vector_was_present,
                    "invalidated_manifest_relative_paths": sorted(
                        overlap_inventory.invalidatable_manifest_paths
                    ),
                    "workspace_upgrade_state": (
                        upgrade_receipt.to_json() if upgrade_receipt is not None else None
                    ),
                    "workspace_upgrade_manifest_baseline": (
                        workspace_upgrade_manifest_baseline if scope == "normalized" else None
                    ),
                }
                _audit(
                    session,
                    operation_id,
                    action="purge",
                    outcome="cleanup_pending",
                    scope=scope,
                    detail=operation.purge_plan_json,
                )
        except BaseException as exc:
            # A connection/session teardown could report an error after SQLite accepted COMMIT.
            # Consult the durable state before restoring anything: once cleanup_pending exists,
            # reintroducing staged copies would violate the purge that already committed.
            try:
                with Session(engine) as state_session:
                    durable_status = state_session.scalar(
                        select(DeletionOperation.status).where(DeletionOperation.id == operation_id)
                    )
            except BaseException as state_exc:
                # Leave staging intact when commit state cannot be proven. This is recoverable and
                # safer than copying private artifacts back into a possibly purged workspace.
                raise exc from state_exc
            if durable_status == "cleanup_pending":
                raise
            for promoted in sorted(
                promoted_trees, key=lambda value: len(value.parts), reverse=True
            ):
                if promoted.exists():
                    _remove_tree_exact(promoted, required_parent=layout.root)
            for created_path in created_paths:
                if _purge_entry_stat(layout, created_path) is not None:
                    try:
                        unlink_regular_file_beneath(created_path, layout.root)
                    except UnsafeArchivePathError as path_exc:
                        raise HardPurgeBlockedError(
                            "Purge rollback created-file target became unsafe"
                        ) from path_exc
            if staging is not None:
                _restore_and_verify_staged_files(layout, staging, operation_id, staged)
                if _purge_entry_stat(layout, staging) is not None:
                    _remove_tree_exact(staging, required_parent=layout.trash / "purge-staging")
            if isinstance(exc, RawSourceRewriteError):
                raise HardPurgeBlockedError(str(exc)) from exc
            raise

        # From here onward, restoration would reintroduce private data after the catalog delete
        # committed. Leave the durable cleanup journal in place on error and let startup/retry
        # call the idempotent finalizer.
        try:
            return _finalize_pending_hard_purge_locked(layout, engine, operation_id)
        except BaseException:
            with immediate_session(engine) as session:
                operation = session.get(DeletionOperation, operation_id)
                if operation is not None and operation.status == "cleanup_pending":
                    operation.error_text = "Hard-purge cleanup is pending retry"
                    operation.purge_plan_json = {
                        **operation.purge_plan_json,
                        "cleanup_attempted": True,
                    }
            raise
