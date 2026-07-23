from __future__ import annotations

import threading
import uuid
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Literal, cast

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, status
from sqlalchemy import Engine, and_, exists, func, not_, select, update
from sqlalchemy.orm import Session, defer
from sqlalchemy.sql import Select

from meta_archive.api.derived_publication import (
    guarded_derived_request,
    raise_publication_blocked,
    raise_publication_busy,
    safe_derived_status_refresh,
)
from meta_archive.api.schemas import (
    AttachmentTextJobRequest,
    AudioTranscriptionJobRequest,
    ConflictAggregateJobRequest,
    ConflictCalibrationCreate,
    ConflictLabelCreate,
    NlpAnnotationJobRequest,
)
from meta_archive.api.search_routes import (
    _cached_search_embedder_factory,
    _cached_search_reranker_factory,
)
from meta_archive.api.security import require_session
from meta_archive.models import (
    ActivityRecord,
    AttachmentOccurrence,
    AttachmentTextExtraction,
    AudioTranscription,
    AudioTranscriptSegment,
    ConflictCalibration,
    ConflictLabel,
    Job,
    NlpAnnotation,
    NlpHourlyRollup,
    SearchWindow,
)
from meta_archive.paths import ArchiveLayout
from meta_archive.services.attachment_text import (
    EXTRACTOR_REVISION as ATTACHMENT_EXTRACTOR_REVISION,
)
from meta_archive.services.attachment_text import (
    PIPELINE_VERSION as ATTACHMENT_TEXT_PIPELINE_VERSION,
)
from meta_archive.services.attachment_text import (
    attachment_text_lock_is_held,
    build_attachment_text,
    extraction_limits,
    sandbox_capability,
)
from meta_archive.services.conflict_aggregates import (
    build_conflict_aggregates,
    conflict_aggregate_lock_is_held,
    conflict_aggregate_status,
    invalidate_conflict_aggregates,
)
from meta_archive.services.conflict_review import (
    CONFLICT_SPLIT_ALGORITHM_VERSION,
    ConflictCalibrationIneligibleError,
    ConflictReviewSplit,
    conflict_review_queue,
    conflict_review_status,
    conflict_split_algorithm,
    deterministic_conflict_split,
    next_unlabelled_conflict_window,
)
from meta_archive.services.conflict_scoring import (
    ConflictModelSpec,
    ConflictScorer,
    ModernBertNliConflictScorer,
    conflict_model_status,
    conflict_search,
)
from meta_archive.services.deletion_lock_lease import DeletionError
from meta_archive.services.derived_publication_guard import DerivedPublicationBlockedError
from meta_archive.services.jumps import activity_jump
from meta_archive.services.nlp_annotations import (
    MIN_CONFLICT_LABELS,
    MODEL_SPECS,
    NLP_INFERENCE_CONFIGURATION_SCHEMA,
    NlpTask,
    _canonical_json_sha256,
    build_nlp_annotations,
    conflict_aggregation_status,
    nlp_annotations_lock_is_held,
    record_conflict_calibration,
)
from meta_archive.services.nlp_annotations import (
    PIPELINE_VERSION as NLP_PIPELINE_VERSION,
)
from meta_archive.services.search import SearchFilters, SearchUnavailableError
from meta_archive.services.transcription import (
    MODEL_NAME as TRANSCRIPTION_MODEL_NAME,
)
from meta_archive.services.transcription import (
    MODEL_REVISION as TRANSCRIPTION_MODEL_REVISION,
)
from meta_archive.services.transcription import (
    PIPELINE_VERSION as TRANSCRIPTION_PIPELINE_VERSION,
)
from meta_archive.services.transcription import (
    audio_transcription_lock_is_held,
    build_audio_transcriptions,
)

enrichment_router = APIRouter(prefix="/api/v1/enrichment", dependencies=[Depends(require_session)])

ATTACHMENT_TEXT_PREVIEW_CHARACTERS = 16_000
DEFAULT_ATTACHMENT_EXTRACTOR_NAME = "bubblewrap-prlimit-local-worker"
_ENRICHMENT_REQUEST_STATE_LOCK = threading.Lock()


def _session(request: Request) -> Session:
    return Session(request.app.state.engine)


def _enrichment_request_state(request: Request) -> tuple[threading.Lock, set[str]]:
    with _ENRICHMENT_REQUEST_STATE_LOCK:
        request_lock = getattr(request.app.state, "enrichment_job_request_lock", None)
        if request_lock is None:
            request_lock = threading.Lock()
            request.app.state.enrichment_job_request_lock = request_lock
        dispatched_job_ids = getattr(
            request.app.state,
            "enrichment_dispatched_job_ids",
            None,
        )
        if dispatched_job_ids is None:
            dispatched_job_ids = set()
            request.app.state.enrichment_dispatched_job_ids = dispatched_job_ids
    return request_lock, dispatched_job_ids


def _conflict_aggregate_status(request: Request) -> dict[str, Any]:
    engine = request.app.state.engine
    return safe_derived_status_refresh(
        request.app.state.layout,
        engine,
        publisher="Conflict aggregate status refresh",
        refresh=not bool(getattr(request.app.state, "acceptance_read_only", False)),
        loader=lambda refresh: conflict_aggregate_status(engine, refresh=refresh),
    )


def _active_nlp_annotation_counts_query() -> Select[tuple[str, str | None, int]]:
    return (
        select(NlpAnnotation.task, NlpAnnotation.job_id, func.count())
        .where(NlpAnnotation.active.is_(True), NlpAnnotation.status == "completed")
        .group_by(NlpAnnotation.task, NlpAnnotation.job_id)
    )


def _current_transcription_version(request: Request) -> dict[str, str]:
    adapter = getattr(request.app.state, "audio_transcription_adapter", None)
    return {
        "pipeline_version": TRANSCRIPTION_PIPELINE_VERSION,
        "model_name": str(adapter.model_name) if adapter is not None else TRANSCRIPTION_MODEL_NAME,
        "model_revision": (
            str(adapter.model_revision) if adapter is not None else TRANSCRIPTION_MODEL_REVISION
        ),
    }


def _current_attachment_text_version(request: Request) -> dict[str, str]:
    extractor = getattr(request.app.state, "attachment_text_extractor", None)
    return {
        "pipeline_version": ATTACHMENT_TEXT_PIPELINE_VERSION,
        "extractor_name": (
            str(extractor.identity) if extractor is not None else DEFAULT_ATTACHMENT_EXTRACTOR_NAME
        ),
        "extractor_revision": (
            str(extractor.revision) if extractor is not None else ATTACHMENT_EXTRACTOR_REVISION
        ),
    }


def _mark_background_job_failed(engine: Engine, job_id: str, exc: BaseException) -> None:
    with Session(engine) as session, session.begin():
        session.execute(
            update(Job)
            .where(Job.id == job_id, Job.status.not_in(("completed", "completed_with_errors")))
            .values(
                status="failed",
                stage="Enrichment job failed",
                error_text=f"{type(exc).__name__}: {exc}"[:4_000],
                finished_at=datetime.now(UTC),
            )
        )


def _finish_background_dispatch(dispatch_done: Callable[[], None] | None) -> None:
    if dispatch_done is not None:
        dispatch_done()


def _run_audio_transcription_job(
    *,
    layout: ArchiveLayout,
    engine: Engine,
    job_id: str,
    retry_failed: bool,
    adapter: Any | None,
    dispatch_done: Callable[[], None] | None = None,
) -> None:
    try:
        build_audio_transcriptions(
            layout,
            engine,
            adapter=adapter,
            retry_failed=retry_failed,
            requested_job_id=job_id,
        )
    except (DerivedPublicationBlockedError, DeletionError):
        return
    except BaseException as exc:
        _mark_background_job_failed(engine, job_id, exc)
    finally:
        _finish_background_dispatch(dispatch_done)


def _run_attachment_text_job(
    *,
    layout: ArchiveLayout,
    engine: Engine,
    job_id: str,
    retry_failed: bool,
    extractor: Any | None,
    dispatch_done: Callable[[], None] | None = None,
) -> None:
    try:
        build_attachment_text(
            layout,
            engine,
            extractor=extractor,
            retry_failed=retry_failed,
            requested_job_id=job_id,
        )
    except (DerivedPublicationBlockedError, DeletionError):
        return
    except BaseException as exc:
        _mark_background_job_failed(engine, job_id, exc)
    finally:
        _finish_background_dispatch(dispatch_done)


def _run_nlp_annotation_job(
    *,
    layout: ArchiveLayout,
    engine: Engine,
    job_id: str,
    tasks: tuple[str, ...],
    batch_size: int,
    classifier: Any | None,
    dispatch_done: Callable[[], None] | None = None,
) -> None:
    try:
        build_nlp_annotations(
            layout,
            engine,
            classifier=classifier,
            tasks=cast(Any, tasks),
            batch_size=batch_size,
            requested_job_id=job_id,
        )
    except (DerivedPublicationBlockedError, DeletionError):
        return
    except BaseException as exc:
        _mark_background_job_failed(engine, job_id, exc)
    finally:
        _finish_background_dispatch(dispatch_done)


def _run_conflict_aggregate_job(
    *,
    layout: ArchiveLayout,
    engine: Engine,
    job_id: str,
    batch_size: int,
    scorer_factory: Any,
    dispatch_done: Callable[[], None] | None = None,
) -> None:
    try:
        build_conflict_aggregates(
            layout,
            engine,
            scorer_factory=scorer_factory,
            batch_size=batch_size,
            requested_job_id=job_id,
        )
    except (DerivedPublicationBlockedError, DeletionError):
        return
    except BaseException as exc:
        _mark_background_job_failed(engine, job_id, exc)
    finally:
        _finish_background_dispatch(dispatch_done)


def _queue_enrichment_job(
    request: Request,
    *,
    kind: Literal[
        "audio_transcription", "attachment_text", "nlp_annotations", "conflict_aggregates"
    ],
    stage: str,
    checkpoint: dict[str, Any],
    lock_is_held: bool,
    dispatched_job_ids: set[str],
) -> tuple[str, bool, Callable[[], None]]:
    request_lock, registered_job_ids = _enrichment_request_state(request)
    if registered_job_ids is not dispatched_job_ids:
        raise RuntimeError("Enrichment dispatch registry changed during request")
    conflict_id: str | None = None
    reusable_id: str | None = None
    job_id = str(uuid.uuid4())
    selected_job_id: str | None = None
    try:
        with request_lock, _session(request) as session, session.begin():
            active_jobs = list(
                session.scalars(
                    select(Job)
                    .where(Job.kind == kind, Job.status.in_(("queued", "running")))
                    .order_by(Job.created_at, Job.id)
                )
            )
            for active_job in active_jobs:
                pristine_queued = (
                    active_job.status == "queued"
                    and active_job.stage == stage
                    and active_job.progress_current == 0
                    and active_job.progress_total is None
                    and active_job.checkpoint_json == checkpoint
                    and active_job.counters_json == {}
                    and active_job.error_text is None
                    and active_job.started_at is None
                    and active_job.finished_at is None
                )
                if pristine_queued and active_job.id not in dispatched_job_ids:
                    if reusable_id is not None:
                        conflict_id = active_job.id
                        break
                    reusable_id = active_job.id
                    continue
                if active_job.status == "running" and not lock_is_held:
                    active_job.status = "failed"
                    active_job.stage = "Interrupted enrichment job recovered"
                    active_job.error_text = "Recovered because no process owns the job lock"
                    active_job.finished_at = datetime.now(UTC)
                    continue
                conflict_id = active_job.id
                break
            if conflict_id is None and reusable_id is None:
                session.add(
                    Job(
                        id=job_id,
                        kind=kind,
                        status="queued",
                        stage=stage,
                        progress_current=0,
                        checkpoint_json=checkpoint,
                        counters_json={},
                    )
                )
            if conflict_id is None:
                selected_job_id = reusable_id or job_id
                dispatched_job_ids.add(selected_job_id)
    except BaseException:
        if selected_job_id is not None:
            with request_lock:
                dispatched_job_ids.discard(selected_job_id)
        raise
    if conflict_id is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Enrichment job {conflict_id} is already active",
        )
    assert selected_job_id is not None

    def dispatch_done() -> None:
        with request_lock:
            dispatched_job_ids.discard(selected_job_id)

    return selected_job_id, reusable_id is not None, dispatch_done


def _warning_items(value: Any) -> list[str]:
    """Expose persisted warning envelopes as a stable list contract."""

    if isinstance(value, dict):
        value = value.get("items", [])
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _probability_payload(value: Any) -> tuple[dict[str, float], str | None]:
    """Separate probability labels from the recorded activation metadata."""

    if not isinstance(value, dict):
        return {}, None
    activation = value.get("activation")
    labels = value.get("labels", value)
    if not isinstance(labels, dict):
        return {}, activation if isinstance(activation, str) else None
    probabilities = {
        label: float(probability)
        for label, probability in labels.items()
        if isinstance(label, str) and isinstance(probability, (int, float))
    }
    return probabilities, activation if isinstance(activation, str) else None


def _nlp_model_provenance_status(
    checkpoint: object,
    *,
    expected_tasks: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    task_names = set(expected_tasks or ())
    if isinstance(checkpoint, dict):
        raw_task_names = checkpoint.get("task_inference_configuration_sha256")
        if isinstance(raw_task_names, dict) and expected_tasks is None:
            task_names.update(str(task) for task in raw_task_names)
    result: dict[str, dict[str, Any]] = {
        task: {"provenance_status": "unavailable"} for task in task_names if task in MODEL_SPECS
    }
    if not isinstance(checkpoint, dict):
        return result
    raw_tasks = checkpoint.get("task_inference_configuration_sha256")
    raw_registry = checkpoint.get("inference_configurations")
    if not isinstance(raw_tasks, dict) or not isinstance(raw_registry, dict):
        return result
    for task in result:
        spec = MODEL_SPECS[cast(NlpTask, task)]
        configuration_sha256 = raw_tasks.get(task)
        if not isinstance(configuration_sha256, str) or not configuration_sha256:
            continue
        configuration = raw_registry.get(configuration_sha256)
        if not isinstance(configuration, dict):
            continue
        artifact = configuration.get("model_artifact")
        implementation = configuration.get("implementation")
        implementation_sha256 = configuration.get("implementation_sha256")
        if (
            not isinstance(artifact, dict)
            or not isinstance(implementation, dict)
            or not isinstance(implementation_sha256, str)
        ):
            continue
        try:
            configuration_hash_matches = (
                _canonical_json_sha256(configuration) == configuration_sha256
            )
            implementation_hash_matches = (
                _canonical_json_sha256(implementation) == implementation_sha256
            )
        except (TypeError, ValueError):
            continue
        if (
            not configuration_hash_matches
            or configuration.get("schema") != NLP_INFERENCE_CONFIGURATION_SCHEMA
            or configuration.get("task") != task
            or configuration.get("model_name") != spec.model_name
            or configuration.get("model_revision") != spec.model_revision
            or not implementation_hash_matches
            or implementation.get("pipeline_version") != NLP_PIPELINE_VERSION
            or not isinstance(implementation.get("provider"), str)
            or not implementation.get("provider")
        ):
            continue
        result[task] = {
            "provenance_status": "verified",
            "inference_configuration_sha256": configuration_sha256,
            "model_artifact": artifact,
            "inference_implementation_sha256": implementation_sha256,
            "inference_backend": {
                "provider": implementation.get("provider"),
                "packages": implementation.get("packages", {}),
            },
        }
    return result


def _occurrences_by_blob(
    session: Session, blob_sha256s: list[str]
) -> dict[str, list[dict[str, str | None]]]:
    result: dict[str, list[dict[str, str | None]]] = defaultdict(list)
    if not blob_sha256s:
        return result
    rows = session.execute(
        select(
            AttachmentOccurrence,
            ActivityRecord.id.label("active_activity_id"),
        )
        .outerjoin(
            ActivityRecord,
            and_(
                ActivityRecord.id == AttachmentOccurrence.activity_record_id,
                ActivityRecord.deleted_at.is_(None),
            ),
        )
        .where(
            AttachmentOccurrence.blob_sha256.in_(blob_sha256s),
            AttachmentOccurrence.deleted_at.is_(None),
        )
        .order_by(
            AttachmentOccurrence.blob_sha256,
            AttachmentOccurrence.timestamp_ms,
            AttachmentOccurrence.id,
        )
    )
    for row, active_activity_id in rows:
        if row.blob_sha256 is None:
            continue
        result[row.blob_sha256].append(
            {
                "occurrence_id": row.id,
                "conversation_id": row.conversation_id,
                "message_id": row.message_id,
                "activity_id": active_activity_id,
            }
        )
    return result


def _has_retained_occurrence(blob_sha256: Any) -> Any:
    """Require a blob to remain reachable from at least one retained occurrence."""
    return exists(
        select(AttachmentOccurrence.id).where(
            AttachmentOccurrence.blob_sha256 == blob_sha256,
            AttachmentOccurrence.deleted_at.is_(None),
        )
    )


def _source_jumps(
    occurrences: list[dict[str, str | None]], *, start_ms: int | None = None
) -> list[dict[str, str | int | None]]:
    jumps: list[dict[str, str | int | None]] = []
    time_query = f"&t={start_ms}" if start_ms is not None else ""
    for occurrence in occurrences:
        occurrence_id = occurrence["occurrence_id"]
        conversation_id = occurrence["conversation_id"]
        message_id = occurrence["message_id"]
        activity_id = occurrence["activity_id"]
        href = (
            f"/conversations/{conversation_id}?around={message_id}"
            f"&media={occurrence_id}{time_query}#message-{message_id}"
            if conversation_id and message_id
            else activity_jump(activity_id)["href"]
            if activity_id
            else f"/media?occurrence={occurrence_id}{time_query}"
        )
        jumps.append(
            {
                **occurrence,
                **({"start_ms": start_ms} if start_ms is not None else {}),
                "href": href,
            }
        )
    return jumps


def _cached_conflict_scorer_factory(request: Request) -> Any:
    cache = getattr(request.app.state, "conflict_scorer_cache", None)
    if cache is None:
        cache = {}
        request.app.state.conflict_scorer_cache = cache
    lock = getattr(request.app.state, "conflict_scorer_lock", None)
    if lock is None:
        lock = threading.Lock()
        request.app.state.conflict_scorer_lock = lock
    configured_factory = getattr(request.app.state, "conflict_scorer_factory", None)

    def get_scorer(spec: ConflictModelSpec) -> ConflictScorer:
        key = (spec.model_name, spec.model_revision, spec.quantization, spec.device)
        with lock:
            existing = cache.get(key)
            if existing is not None:
                return cast(ConflictScorer, existing)
            try:
                created = cast(
                    ConflictScorer,
                    configured_factory(spec)
                    if configured_factory is not None
                    else ModernBertNliConflictScorer(
                        request.app.state.layout.models / "conflict" / spec.model_revision,
                        spec,
                        local_files_only=bool(
                            getattr(request.app.state, "acceptance_read_only", False)
                        ),
                    ),
                )
            except Exception as exc:
                request.app.state.conflict_scorer_error = f"{type(exc).__name__}: {exc}"[:1_000]
                raise
            cache[key] = created
            request.app.state.conflict_scorer_error = None
            return created

    return get_scorer


@enrichment_router.get("/status")
def enrichment_status(request: Request) -> dict[str, Any]:
    transcription_version = _current_transcription_version(request)
    extraction_version = _current_attachment_text_version(request)
    current_transcription = and_(
        AudioTranscription.pipeline_version == transcription_version["pipeline_version"],
        AudioTranscription.model_name == transcription_version["model_name"],
        AudioTranscription.model_revision == transcription_version["model_revision"],
    )
    current_extraction = and_(
        AttachmentTextExtraction.pipeline_version == extraction_version["pipeline_version"],
        AttachmentTextExtraction.extractor_name == extraction_version["extractor_name"],
        AttachmentTextExtraction.extractor_revision == extraction_version["extractor_revision"],
    )
    with _session(request) as session:
        transcript_counts = {
            status_value: count
            for status_value, count in session.execute(
                select(AudioTranscription.status, func.count(AudioTranscription.id))
                .where(
                    current_transcription,
                    _has_retained_occurrence(AudioTranscription.blob_sha256),
                )
                .group_by(AudioTranscription.status)
            )
        }
        extraction_counts = {
            status_value: count
            for status_value, count in session.execute(
                select(
                    AttachmentTextExtraction.status,
                    func.count(AttachmentTextExtraction.id),
                )
                .where(
                    current_extraction,
                    _has_retained_occurrence(AttachmentTextExtraction.blob_sha256),
                )
                .group_by(AttachmentTextExtraction.status)
            )
        }
        historical_transcription_count = int(
            session.scalar(
                select(func.count(AudioTranscription.id)).where(
                    not_(current_transcription),
                    _has_retained_occurrence(AudioTranscription.blob_sha256),
                )
            )
            or 0
        )
        historical_extraction_count = int(
            session.scalar(
                select(func.count(AttachmentTextExtraction.id)).where(
                    not_(current_extraction),
                    _has_retained_occurrence(AttachmentTextExtraction.blob_sha256),
                )
            )
            or 0
        )
        annotation_counts: dict[str, int] = {}
        active_nlp_job_ids: set[str] = set()
        active_nlp_job_tasks: dict[str, set[str]] = {}
        for task, job_id, count in session.execute(_active_nlp_annotation_counts_query()):
            annotation_counts[str(task)] = annotation_counts.get(str(task), 0) + int(count)
            if isinstance(job_id, str) and job_id:
                active_nlp_job_ids.add(job_id)
                active_nlp_job_tasks.setdefault(job_id, set()).add(str(task))
        rollup_counts = {
            task: count
            for task, count in session.execute(
                select(NlpHourlyRollup.task, func.count(NlpHourlyRollup.id))
                .where(NlpHourlyRollup.active.is_(True))
                .group_by(NlpHourlyRollup.task)
            )
        }
        annotation_binding_counts: dict[str, dict[str, int]] = {}
        nlp_model_provenance: dict[str, dict[str, Any]] = {}
        active_nlp_jobs = (
            list(
                session.scalars(
                    select(Job)
                    .where(Job.id.in_(active_nlp_job_ids), Job.kind == "nlp_annotations")
                    .order_by(Job.created_at, Job.id)
                )
            )
            if active_nlp_job_ids
            else []
        )
        for active_nlp_job in active_nlp_jobs:
            nlp_model_provenance.update(
                _nlp_model_provenance_status(
                    active_nlp_job.checkpoint_json,
                    expected_tasks=active_nlp_job_tasks.get(active_nlp_job.id, set()),
                )
            )
            raw_counts = active_nlp_job.counters_json.get("provenance_binding_counts")
            if isinstance(raw_counts, dict):
                for task, task_counts in raw_counts.items():
                    if not isinstance(task, str) or not isinstance(task_counts, dict):
                        continue
                    annotation_binding_counts[task] = {
                        str(binding_kind): int(count)
                        for binding_kind, count in task_counts.items()
                        if isinstance(binding_kind, str)
                        and isinstance(count, int)
                        and not isinstance(count, bool)
                        and count >= 0
                    }
        jobs = [
            {
                "id": row.id,
                "kind": row.kind,
                "status": row.status,
                "stage": row.stage,
                "progress_current": row.progress_current,
                "progress_total": row.progress_total,
                "error": row.error_text,
            }
            for row in session.scalars(
                select(Job)
                .where(
                    Job.kind.in_(
                        (
                            "audio_transcription",
                            "attachment_text",
                            "nlp_annotations",
                            "conflict_aggregates",
                        )
                    )
                )
                .order_by(Job.created_at.desc())
                .limit(10)
            )
        ]
    conflict_cache = getattr(request.app.state, "conflict_scorer_cache", {})
    conflict_scorer = next(iter(conflict_cache.values()), None)
    conflict_scorer_error = getattr(request.app.state, "conflict_scorer_error", None)
    return {
        "audio_transcriptions": transcript_counts,
        "attachment_text_extractions": extraction_counts,
        "audio_transcription_selection": {
            "scope": "current",
            "current": transcription_version,
            "current_record_count": sum(transcript_counts.values()),
            "historical_record_count": historical_transcription_count,
        },
        "attachment_text_selection": {
            "scope": "current",
            "current": extraction_version,
            "current_record_count": sum(extraction_counts.values()),
            "historical_record_count": historical_extraction_count,
        },
        "active_nlp_annotations": annotation_counts,
        "active_nlp_hourly_rollups": rollup_counts,
        "nlp_models": {
            task: {
                "model_name": spec.model_name,
                "model_revision": spec.model_revision,
                "language_scope": spec.language_scope,
                "device": spec.device,
                "quantization": spec.quantization,
                "warnings": list(spec.warnings),
                "provenance_status": "unavailable",
                "provenance_binding_counts": (
                    annotation_binding_counts.get(task, {}) if task in annotation_counts else {}
                ),
                **nlp_model_provenance.get(task, {}),
            }
            for task, spec in MODEL_SPECS.items()
        },
        "attachment_sandbox": sandbox_capability(),
        "attachment_limits": extraction_limits(),
        "conflict": conflict_aggregation_status(request.app.state.engine),
        "conflict_aggregates": _conflict_aggregate_status(request),
        "conflict_scoring_model": (
            conflict_scorer.status
            if conflict_scorer is not None
            else conflict_model_status(
                status="unavailable" if conflict_scorer_error else "not_loaded",
                error=conflict_scorer_error,
            )
        ),
        "jobs": jobs,
    }


@enrichment_router.post("/transcriptions/jobs", status_code=status.HTTP_202_ACCEPTED)
def start_audio_transcription_job(
    request: Request,
    payload: AudioTranscriptionJobRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    _, dispatched_job_ids = _enrichment_request_state(request)
    with guarded_derived_request(
        request.app.state.layout,
        request.app.state.engine,
        publisher="Audio transcription request",
    ):
        job_id, reused, dispatch_done = _queue_enrichment_job(
            request,
            kind="audio_transcription",
            stage="Audio transcription queued",
            checkpoint={"retry_failed": payload.retry_failed},
            lock_is_held=audio_transcription_lock_is_held(request.app.state.layout),
            dispatched_job_ids=dispatched_job_ids,
        )
    try:
        background_tasks.add_task(
            _run_audio_transcription_job,
            layout=request.app.state.layout,
            engine=request.app.state.engine,
            job_id=job_id,
            retry_failed=payload.retry_failed,
            adapter=getattr(request.app.state, "audio_transcription_adapter", None),
            dispatch_done=dispatch_done,
        )
    except BaseException:
        dispatch_done()
        raise
    return {
        "job_id": job_id,
        "status": "queued",
        "reused": reused,
        "retry_failed": payload.retry_failed,
        "status_href": "/api/v1/enrichment/status",
    }


@enrichment_router.post("/attachment-text/jobs", status_code=status.HTTP_202_ACCEPTED)
def start_attachment_text_job(
    request: Request,
    payload: AttachmentTextJobRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    _, dispatched_job_ids = _enrichment_request_state(request)
    with guarded_derived_request(
        request.app.state.layout,
        request.app.state.engine,
        publisher="Attachment text request",
    ):
        job_id, reused, dispatch_done = _queue_enrichment_job(
            request,
            kind="attachment_text",
            stage="Attachment text extraction queued",
            checkpoint={"retry_failed": payload.retry_failed},
            lock_is_held=attachment_text_lock_is_held(request.app.state.layout),
            dispatched_job_ids=dispatched_job_ids,
        )
    try:
        background_tasks.add_task(
            _run_attachment_text_job,
            layout=request.app.state.layout,
            engine=request.app.state.engine,
            job_id=job_id,
            retry_failed=payload.retry_failed,
            extractor=getattr(request.app.state, "attachment_text_extractor", None),
            dispatch_done=dispatch_done,
        )
    except BaseException:
        dispatch_done()
        raise
    return {
        "job_id": job_id,
        "status": "queued",
        "reused": reused,
        "retry_failed": payload.retry_failed,
        "status_href": "/api/v1/enrichment/status",
    }


@enrichment_router.post("/nlp/jobs", status_code=status.HTTP_202_ACCEPTED)
def start_nlp_annotation_job(
    request: Request,
    payload: NlpAnnotationJobRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    tasks = tuple(dict.fromkeys(payload.tasks))
    _, dispatched_job_ids = _enrichment_request_state(request)
    with guarded_derived_request(
        request.app.state.layout,
        request.app.state.engine,
        publisher="NLP annotation request",
    ):
        job_id, reused, dispatch_done = _queue_enrichment_job(
            request,
            kind="nlp_annotations",
            stage="Local NLP annotations queued",
            checkpoint={"tasks": list(tasks), "batch_size": payload.batch_size},
            lock_is_held=nlp_annotations_lock_is_held(request.app.state.layout),
            dispatched_job_ids=dispatched_job_ids,
        )
    try:
        background_tasks.add_task(
            _run_nlp_annotation_job,
            layout=request.app.state.layout,
            engine=request.app.state.engine,
            job_id=job_id,
            tasks=tasks,
            batch_size=payload.batch_size,
            classifier=getattr(request.app.state, "nlp_classifier", None),
            dispatch_done=dispatch_done,
        )
    except BaseException:
        dispatch_done()
        raise
    return {
        "job_id": job_id,
        "status": "queued",
        "reused": reused,
        "tasks": list(tasks),
        "status_href": "/api/v1/enrichment/status",
    }


@enrichment_router.post("/conflict/aggregate-jobs", status_code=status.HTTP_202_ACCEPTED)
def start_conflict_aggregate_job(
    request: Request,
    payload: ConflictAggregateJobRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    _, dispatched_job_ids = _enrichment_request_state(request)
    with guarded_derived_request(
        request.app.state.layout,
        request.app.state.engine,
        publisher="Conflict aggregate request",
    ):
        gate = conflict_aggregation_status(request.app.state.engine)
        if not gate["aggregation_enabled"]:
            raise HTTPException(
                status_code=409,
                detail=gate["reason"] or "Conflict calibration gate has not passed",
            )
        aggregate = conflict_aggregate_status(request.app.state.engine)
        if aggregate["available"]:
            return {
                "job_id": aggregate["job"]["id"] if aggregate.get("job") else None,
                "status": "completed",
                "reused": True,
                "status_href": "/api/v1/enrichment/conflict/status",
            }
        job_id, reused, dispatch_done = _queue_enrichment_job(
            request,
            kind="conflict_aggregates",
            stage="Gated full-corpus conflict scoring queued",
            checkpoint={"batch_size": payload.batch_size},
            lock_is_held=conflict_aggregate_lock_is_held(request.app.state.layout),
            dispatched_job_ids=dispatched_job_ids,
        )
    try:
        background_tasks.add_task(
            _run_conflict_aggregate_job,
            layout=request.app.state.layout,
            engine=request.app.state.engine,
            job_id=job_id,
            batch_size=payload.batch_size,
            scorer_factory=_cached_conflict_scorer_factory(request),
            dispatch_done=dispatch_done,
        )
    except BaseException:
        dispatch_done()
        raise
    return {
        "job_id": job_id,
        "status": "queued",
        "reused": reused,
        "batch_size": payload.batch_size,
        "status_href": "/api/v1/enrichment/conflict/status",
    }


@enrichment_router.get("/transcriptions")
def transcriptions(
    request: Request,
    status_filter: Literal["all", "completed", "failed", "processing"] = Query(
        default="all", alias="status"
    ),
    version_scope: Literal["current", "history"] = Query(default="current", alias="version"),
    pipeline_version: str | None = Query(default=None, max_length=40),
    model_revision: str | None = Query(default=None, max_length=80),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    current = _current_transcription_version(request)
    statement = (
        select(AudioTranscription)
        .order_by(AudioTranscription.finished_at.desc(), AudioTranscription.id)
        .where(_has_retained_occurrence(AudioTranscription.blob_sha256))
    )
    selected_pipeline_version: str | None
    selected_model_name: str | None
    selected_model_revision: str | None
    if version_scope == "current":
        incompatible = (
            pipeline_version is not None and pipeline_version != current["pipeline_version"]
        ) or (model_revision is not None and model_revision != current["model_revision"])
        if incompatible:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Exact historical versions require version=history; current always selects "
                    "the configured pipeline and model revision"
                ),
            )
        statement = statement.where(
            AudioTranscription.pipeline_version == current["pipeline_version"],
            AudioTranscription.model_name == current["model_name"],
            AudioTranscription.model_revision == current["model_revision"],
        )
        selected_pipeline_version = current["pipeline_version"]
        selected_model_name = current["model_name"]
        selected_model_revision = current["model_revision"]
    else:
        if pipeline_version is not None:
            statement = statement.where(AudioTranscription.pipeline_version == pipeline_version)
        if model_revision is not None:
            statement = statement.where(AudioTranscription.model_revision == model_revision)
        selected_pipeline_version = pipeline_version
        selected_model_name = None
        selected_model_revision = model_revision
    if status_filter != "all":
        statement = statement.where(AudioTranscription.status == status_filter)
    with _session(request) as session:
        rows = list(session.scalars(statement.offset(offset).limit(limit + 1)))
        page_rows = rows[:limit]
        segments_by_transcription: dict[str, list[AudioTranscriptSegment]] = defaultdict(list)
        if page_rows:
            for segment in session.scalars(
                select(AudioTranscriptSegment)
                .where(AudioTranscriptSegment.transcription_id.in_([row.id for row in page_rows]))
                .order_by(
                    AudioTranscriptSegment.transcription_id,
                    AudioTranscriptSegment.ordinal,
                )
            ):
                segments_by_transcription[segment.transcription_id].append(segment)
        occurrences = _occurrences_by_blob(session, [row.blob_sha256 for row in page_rows])
        items: list[dict[str, Any]] = []
        for row in page_rows:
            items.append(
                {
                    "id": row.id,
                    "blob_sha256": row.blob_sha256,
                    "status": row.status,
                    "pipeline_version": row.pipeline_version,
                    "model_name": row.model_name,
                    "model_revision": row.model_revision,
                    "is_current_version": bool(
                        row.pipeline_version == current["pipeline_version"]
                        and row.model_name == current["model_name"]
                        and row.model_revision == current["model_revision"]
                    ),
                    "device": row.device,
                    "compute_type": row.compute_type,
                    "language": row.language,
                    "language_probability": row.language_probability,
                    "duration_ms": row.duration_ms,
                    "text": row.text_display,
                    "segment_count": row.segment_count,
                    "warnings": _warning_items(row.warnings_json),
                    "provenance": row.provenance_json,
                    "error": row.error_text,
                    "segments": [
                        {
                            "id": segment.id,
                            "ordinal": segment.ordinal,
                            "start_ms": segment.start_ms,
                            "end_ms": segment.end_ms,
                            "text": segment.text_display,
                            "confidence": segment.confidence,
                            "avg_log_probability": segment.avg_log_probability,
                            "no_speech_probability": segment.no_speech_probability,
                            "jumps": _source_jumps(
                                occurrences.get(row.blob_sha256, []),
                                start_ms=segment.start_ms,
                            ),
                        }
                        for segment in segments_by_transcription.get(row.id, [])
                    ],
                }
            )
    return {
        "items": items,
        "selection": {
            "scope": version_scope,
            "pipeline_version": selected_pipeline_version,
            "model_name": selected_model_name,
            "model_revision": selected_model_revision,
            "current": current,
        },
        "page": {
            "limit": limit,
            "offset": offset,
            "next_offset": offset + limit if len(rows) > limit else None,
        },
    }


@enrichment_router.get("/attachment-text")
def attachment_text(
    request: Request,
    status_filter: Literal[
        "all", "completed", "failed", "excluded", "unsupported", "limit_exceeded"
    ] = Query(default="all", alias="status"),
    version_scope: Literal["current", "history"] = Query(default="current", alias="version"),
    pipeline_version: str | None = Query(default=None, max_length=40),
    extractor_revision: str | None = Query(default=None, max_length=80),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    current = _current_attachment_text_version(request)
    statement = (
        select(
            AttachmentTextExtraction,
            func.substr(
                AttachmentTextExtraction.text_display,
                1,
                ATTACHMENT_TEXT_PREVIEW_CHARACTERS + 1,
            ).label("text_preview"),
        )
        .options(
            defer(AttachmentTextExtraction.text_display),
            defer(AttachmentTextExtraction.text_search),
        )
        .where(_has_retained_occurrence(AttachmentTextExtraction.blob_sha256))
        .order_by(AttachmentTextExtraction.finished_at.desc(), AttachmentTextExtraction.id)
    )
    selected_pipeline_version: str | None
    selected_extractor_name: str | None
    selected_extractor_revision: str | None
    if version_scope == "current":
        incompatible = (
            pipeline_version is not None and pipeline_version != current["pipeline_version"]
        ) or (
            extractor_revision is not None and extractor_revision != current["extractor_revision"]
        )
        if incompatible:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Exact historical versions require version=history; current always selects "
                    "the configured pipeline and extractor revision"
                ),
            )
        statement = statement.where(
            AttachmentTextExtraction.pipeline_version == current["pipeline_version"],
            AttachmentTextExtraction.extractor_name == current["extractor_name"],
            AttachmentTextExtraction.extractor_revision == current["extractor_revision"],
        )
        selected_pipeline_version = current["pipeline_version"]
        selected_extractor_name = current["extractor_name"]
        selected_extractor_revision = current["extractor_revision"]
    else:
        if pipeline_version is not None:
            statement = statement.where(
                AttachmentTextExtraction.pipeline_version == pipeline_version
            )
        if extractor_revision is not None:
            statement = statement.where(
                AttachmentTextExtraction.extractor_revision == extractor_revision
            )
        selected_pipeline_version = pipeline_version
        selected_extractor_name = None
        selected_extractor_revision = extractor_revision
    if status_filter != "all":
        statement = statement.where(AttachmentTextExtraction.status == status_filter)
    with _session(request) as session:
        rows = list(session.execute(statement.offset(offset).limit(limit + 1)))
        page_rows = rows[:limit]
        occurrences = _occurrences_by_blob(
            session, [row.blob_sha256 for row, _preview in page_rows]
        )
        items: list[dict[str, Any]] = []
        for row, raw_preview in page_rows:
            preview_with_sentinel = str(raw_preview) if raw_preview is not None else None
            text_preview = (
                preview_with_sentinel[:ATTACHMENT_TEXT_PREVIEW_CHARACTERS]
                if preview_with_sentinel is not None
                else None
            )
            items.append(
                {
                    "id": row.id,
                    "blob_sha256": row.blob_sha256,
                    "status": row.status,
                    "pipeline_version": row.pipeline_version,
                    "extractor_name": row.extractor_name,
                    "extractor_revision": row.extractor_revision,
                    "is_current_version": bool(
                        row.pipeline_version == current["pipeline_version"]
                        and row.extractor_name == current["extractor_name"]
                        and row.extractor_revision == current["extractor_revision"]
                    ),
                    "filename": row.filename,
                    "media_type": row.media_type,
                    "text_preview": text_preview,
                    "preview_character_count": len(text_preview or ""),
                    "preview_truncated": bool(
                        preview_with_sentinel is not None
                        and len(preview_with_sentinel) > ATTACHMENT_TEXT_PREVIEW_CHARACTERS
                    ),
                    "preview_limit": ATTACHMENT_TEXT_PREVIEW_CHARACTERS,
                    "character_count": row.character_count,
                    "truncated": row.truncated,
                    "index_eligible": row.index_eligible,
                    "sandboxed": row.sandboxed,
                    "sandbox_status": row.sandbox_status,
                    "limits": row.limits_json,
                    "warnings": _warning_items(row.warnings_json),
                    "provenance": row.provenance_json,
                    "error": row.error_text,
                    "jumps": _source_jumps(occurrences.get(row.blob_sha256, [])),
                }
            )
    return {
        "items": items,
        "selection": {
            "scope": version_scope,
            "pipeline_version": selected_pipeline_version,
            "extractor_name": selected_extractor_name,
            "extractor_revision": selected_extractor_revision,
            "current": current,
        },
        "page": {
            "limit": limit,
            "offset": offset,
            "next_offset": offset + limit if len(rows) > limit else None,
        },
    }


@enrichment_router.get("/nlp")
def nlp_annotations(
    request: Request,
    task: Literal["all", "sentiment", "emotion", "toxicity"] = "all",
    target_type: str | None = Query(default=None, max_length=32),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    statement = select(NlpAnnotation).where(
        NlpAnnotation.active.is_(True), NlpAnnotation.status == "completed"
    )
    if task != "all":
        statement = statement.where(NlpAnnotation.task == task)
    if target_type:
        statement = statement.where(NlpAnnotation.target_type == target_type)
    statement = statement.order_by(NlpAnnotation.created_at.desc(), NlpAnnotation.id)
    with _session(request) as session:
        rows = list(session.scalars(statement.offset(offset).limit(limit + 1)))
    return {
        "items": [
            {
                "id": row.id,
                "target_type": row.target_type,
                "target_id": row.target_id,
                "task": row.task,
                "model_name": row.model_name,
                "model_revision": row.model_revision,
                "predicted_label": row.predicted_label,
                "confidence": row.confidence,
                "probabilities": _probability_payload(row.probabilities_json)[0],
                "probability_activation": _probability_payload(row.probabilities_json)[1],
                "attribution": row.attribution_json,
                "warnings": _warning_items(row.warnings_json),
                "provenance": row.provenance_json,
            }
            for row in rows[:limit]
        ],
        "page": {
            "limit": limit,
            "offset": offset,
            "next_offset": offset + limit if len(rows) > limit else None,
        },
    }


@enrichment_router.get("/nlp/hourly")
def nlp_hourly_rollups(
    request: Request,
    task: Literal["all", "sentiment", "emotion", "toxicity"] = "all",
    direction: Literal["all", "owner", "other", "activity", "audio", "attachment"] = "all",
    start_ms: int | None = None,
    end_ms: int | None = None,
    limit: int = Query(default=500, ge=1, le=5_000),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    statement = select(NlpHourlyRollup).where(NlpHourlyRollup.active.is_(True))
    if task != "all":
        statement = statement.where(NlpHourlyRollup.task == task)
    if direction != "all":
        statement = statement.where(NlpHourlyRollup.direction == direction)
    if start_ms is not None:
        statement = statement.where(NlpHourlyRollup.hour_start_ms >= start_ms)
    if end_ms is not None:
        statement = statement.where(NlpHourlyRollup.hour_start_ms < end_ms)
    statement = statement.order_by(
        NlpHourlyRollup.hour_start_ms, NlpHourlyRollup.task, NlpHourlyRollup.direction
    )
    with _session(request) as session:
        rows = list(session.scalars(statement.offset(offset).limit(limit + 1)))
    return {
        "items": [
            {
                "id": row.id,
                "task": row.task,
                "hour_start_ms": row.hour_start_ms,
                "timezone": "UTC",
                "direction": row.direction,
                "pipeline_version": row.pipeline_version,
                "model_name": row.model_name,
                "model_revision": row.model_revision,
                "configuration_sha256": row.configuration_sha256,
                "configuration": row.configuration_json,
                "predicted_label": row.predicted_label,
                "sample_count": row.sample_count,
                "confidence_sum": row.confidence_sum,
                "probability_sums": row.probability_sums_json,
                "generated_at": row.generated_at,
            }
            for row in rows[:limit]
        ],
        "page": {
            "limit": limit,
            "offset": offset,
            "next_offset": offset + limit if len(rows) > limit else None,
        },
    }


@enrichment_router.get("/conflict/status")
def conflict_status(request: Request) -> dict[str, Any]:
    return {
        **conflict_aggregation_status(request.app.state.engine),
        "aggregate": _conflict_aggregate_status(request),
    }


@enrichment_router.get("/conflict/search")
def search_conflict_windows(
    request: Request,
    q: str = Query(min_length=1, max_length=2_000),
    label: Literal[
        "direct conflict",
        "calm disagreement",
        "banter",
        "third-party conflict",
        "no conflict",
        "uncertain",
    ] = "direct conflict",
    conversation_id: str | None = Query(default=None, max_length=128),
    participant: str | None = Query(default=None, max_length=200),
    from_ms: int | None = Query(default=None, ge=0),
    to_ms: int | None = Query(default=None, ge=0),
    limit: int = Query(default=25, ge=1, le=100),
) -> dict[str, Any]:
    if from_ms is not None and to_ms is not None and from_ms > to_ms:
        raise HTTPException(status_code=422, detail="Invalid date range")
    try:
        page = conflict_search(
            request.app.state.layout,
            request.app.state.engine,
            q,
            target_label=label,
            filters=SearchFilters(
                entity_type="message",
                conversation_id=conversation_id,
                participant=participant,
                from_ms=from_ms,
                to_ms=to_ms,
            ),
            limit=limit,
            embedder_factory=_cached_search_embedder_factory(request),
            reranker_factory=_cached_search_reranker_factory(request),
            scorer_factory=_cached_conflict_scorer_factory(request),
            acceptance_read_only=bool(getattr(request.app.state, "acceptance_read_only", False)),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SearchUnavailableError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    aggregation = conflict_aggregation_status(request.app.state.engine)
    return {
        "items": page.items,
        "page": {
            "limit": limit,
            "returned": len(page.items),
            "candidate_count": page.total_candidates,
        },
        "query": q,
        "target_label": label,
        "retrieval_config": page.retrieval_config,
        "conflict_model": page.model_status,
        "aggregation": aggregation,
    }


@enrichment_router.get("/conflict/review/status")
def conflict_manual_review_status(request: Request) -> dict[str, Any]:
    return conflict_review_status(
        request.app.state.engine,
        minimum_label_count=MIN_CONFLICT_LABELS,
    )


@enrichment_router.get("/conflict/review/queue")
def conflict_manual_review_queue(
    request: Request,
    label: Literal[
        "unlabelled",
        "direct conflict",
        "calm disagreement",
        "banter",
        "third-party conflict",
        "no conflict",
        "uncertain",
    ] = "unlabelled",
    split: ConflictReviewSplit = "all",
    after: str | None = Query(default=None, max_length=64),
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    return conflict_review_queue(
        request.app.state.engine,
        label=label,
        split=split,
        after=after,
        limit=limit,
        minimum_label_count=MIN_CONFLICT_LABELS,
    )


@enrichment_router.get("/conflict/review/next")
def next_conflict_manual_review_window(
    request: Request,
    split: ConflictReviewSplit = "all",
    after: str | None = Query(default=None, max_length=64),
) -> dict[str, Any]:
    return next_unlabelled_conflict_window(
        request.app.state.engine,
        split=split,
        after=after,
        minimum_label_count=MIN_CONFLICT_LABELS,
    )


@enrichment_router.post("/conflict/labels", status_code=status.HTTP_201_CREATED)
def create_conflict_label(request: Request, payload: ConflictLabelCreate) -> dict[str, Any]:
    with guarded_derived_request(
        request.app.state.layout,
        request.app.state.engine,
        publisher="Conflict label publication",
    ):
        with _session(request) as session, session.begin():
            window = session.get(SearchWindow, payload.window_id)
            if window is None or window.entity_type != "message_window":
                raise HTTPException(status_code=404, detail="Message search window not found")
            if window.conversation_id is None:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Message search window has no conversation for deterministic splitting"
                    ),
                )
            evaluation_split = deterministic_conflict_split(window.conversation_id)
            row = session.scalar(
                select(ConflictLabel)
                .where(ConflictLabel.window_id == payload.window_id)
                .order_by(ConflictLabel.created_at.desc(), ConflictLabel.id.desc())
            )
            updated = row is not None
            if row is None:
                row = ConflictLabel(
                    id=str(uuid.uuid4()),
                    window_id=payload.window_id,
                    label=payload.label,
                    evaluation_split=evaluation_split,
                    notes=payload.notes,
                )
                session.add(row)
            else:
                row.label = payload.label
                row.evaluation_split = evaluation_split
                row.notes = payload.notes
            session.flush()
            result = {
                "id": row.id,
                "window_id": row.window_id,
                "label": row.label,
                "notes": row.notes,
                "evaluation_split": row.evaluation_split,
                "split_assignment": {
                    "version": CONFLICT_SPLIT_ALGORITHM_VERSION,
                    "source": "conversation_id",
                },
                "updated": updated,
                "calibration_invalidated": True,
            }
            session.execute(
                update(ConflictCalibration)
                .where(ConflictCalibration.status == "calibrated")
                .values(
                    status="stale",
                    aggregation_enabled=False,
                )
            )
        invalidate_conflict_aggregates(
            request.app.state.engine,
            reason="Conflict labels changed; held-out calibration must be rerun",
        )
    return result


@enrichment_router.post("/conflict/calibrations", status_code=status.HTTP_201_CREATED)
def create_conflict_calibration(
    request: Request, _payload: ConflictCalibrationCreate
) -> dict[str, Any]:
    try:
        row = record_conflict_calibration(
            request.app.state.layout,
            request.app.state.engine,
            scorer_factory=_cached_conflict_scorer_factory(request),
            require_eligible=True,
        )
    except ConflictCalibrationIneligibleError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "conflict_calibration_ineligible",
                "message": "Conflict calibration requires eligible genuine manual labels.",
                "eligibility": exc.eligibility,
            },
        ) from exc
    except DerivedPublicationBlockedError as exc:
        raise_publication_blocked(exc)
    except DeletionError as exc:
        raise_publication_busy(exc)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(
            status_code=409,
            detail="Held-out conflict calibration could not be completed",
        ) from exc
    return {
        "id": row.id,
        "status": row.status,
        "sample_count": row.sample_count,
        "aggregation_enabled": row.aggregation_enabled,
        "direct_conflict_precision": row.direct_conflict_precision,
        "macro_f1": row.macro_f1,
        "evaluation_split_algorithm": conflict_split_algorithm(),
        "relationship_ranking_enabled": False,
    }
