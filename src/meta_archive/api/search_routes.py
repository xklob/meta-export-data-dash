from __future__ import annotations

import copy
import hashlib
import json
import threading
import uuid
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any, Literal, cast

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, status
from sqlalchemy import Engine, and_, desc, func, or_, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from meta_archive.api.derived_publication import guarded_derived_request
from meta_archive.api.pagination import (
    InvalidCursor,
    decode_cursor,
    encode_cursor,
    require_cursor_int,
    require_cursor_str,
)
from meta_archive.api.schemas import (
    RagQuestion,
    RelevanceFeedbackCreate,
    RelevanceFeedbackUpdate,
    SavedSearchCreate,
    SavedSearchUpdate,
    SemanticIndexRequest,
)
from meta_archive.api.security import require_session
from meta_archive.models import (
    Job,
    RagAudit,
    RelevanceFeedback,
    SavedSearch,
    SemanticIndexConfig,
)
from meta_archive.paths import ArchiveLayout
from meta_archive.services.deletion_lock_lease import DeletionError
from meta_archive.services.derived_publication_guard import DerivedPublicationBlockedError
from meta_archive.services.direct_conflict_ranker import (
    DirectConflictRankerError,
    development_feedback_revision_payload,
    is_direct_conflict_trigger,
)
from meta_archive.services.rag import (
    CloudRagDisabledError,
    RagCitationError,
    answer_with_rag,
)
from meta_archive.services.reranking import (
    DEFAULT_RERANKER_SPEC,
    RerankerSpec,
    TextReranker,
    TransformersCrossEncoderReranker,
    reranker_status_payload,
)
from meta_archive.services.search import (
    SearchFilters,
    SearchMode,
    SearchUnavailableError,
    derived_occurrence_page,
    run_search,
)
from meta_archive.services.search_index import (
    EmbeddingModelSpec,
    TextEmbedder,
    build_semantic_index,
    default_text_embedder,
    held_out_feedback_revision,
    semantic_index_lock_is_held,
)

search_router = APIRouter(prefix="/api/v1", dependencies=[Depends(require_session)])
SEARCH_RESULT_CAP = 1_000
_SEMANTIC_REQUEST_STATE_LOCK = threading.Lock()

_DEVELOPMENT_CALIBRATION_PUBLIC_KEYS = frozenset(
    {
        "benchmark_kind",
        "evaluation_intent",
        "evidence_kind",
        "held_out",
        "required_for_production_acceptance",
        "status",
        "passed",
        "method",
        "ranking_pipeline",
        "query_details_persisted",
        "model",
        "model_metadata",
        "minimums",
        "thresholds",
        "judgment_count",
        "reviewed_judgment_count",
        "eligible_judgment_count",
        "binary_judgment_count",
        "conversation_count",
        "conversation_group_count",
        "fold_count",
        "relevant_count",
        "not_relevant_count",
        "uncertain_count",
        "grouped_cross_validation",
        "literal_query_regression",
        "feedback_revision_stable_during_evaluation",
    }
)
_DEVELOPMENT_CALIBRATION_NESTED_PUBLIC_KEYS = frozenset(
    {
        "name",
        "version",
        "kind",
        "family",
        "classifier",
        "estimator",
        "library",
        "library_version",
        "model_name",
        "model_revision",
        "vector_model_name",
        "vector_model_revision",
        "feature_kind",
        "ranking_pipeline_version",
        "ranking_pipeline",
        "score_kind",
        "ranking_feature_only",
        "parameters",
        "counts",
        "class_weight",
        "random_state",
        "c",
        "loss",
        "penalty",
        "dual",
        "fit_intercept",
        "intercept_scaling",
        "max_iter",
        "tolerance",
        "max_feedback_rows",
        "candidate_count",
        "grouping",
        "group_unit",
        "selection",
        "fold_count",
        "judgment_count",
        "reviewed_judgment_count",
        "eligible_judgment_count",
        "binary_judgment_count",
        "conversation_count",
        "conversation_group_count",
        "relevant_count",
        "not_relevant_count",
        "uncertain_count",
        "reviewed_feedback_rows",
        "binary_training_rows",
        "relevant_rows",
        "not_relevant_rows",
        "uncertain_rows",
        "conversation_groups",
        "minimum",
        "mean",
        "minimum_metrics",
        "mean_metrics",
        "metrics",
        "thresholds",
        "k",
        "average_precision",
        "recall_at_k",
        "recall_at_10",
        "mrr",
        "mrr_at_10",
        "ndcg_at_k",
        "ndcg_at_10",
        "roc_auc",
        "status",
        "passed",
    }
)


def _session(request: Request) -> Session:
    return Session(request.app.state.engine)


def _semantic_request_state(request: Request) -> tuple[threading.Lock, set[str]]:
    with _SEMANTIC_REQUEST_STATE_LOCK:
        request_lock = getattr(request.app.state, "semantic_index_request_lock", None)
        if request_lock is None:
            request_lock = threading.Lock()
            request.app.state.semantic_index_request_lock = request_lock
        dispatched_job_ids = getattr(
            request.app.state,
            "semantic_index_dispatched_job_ids",
            None,
        )
        if dispatched_job_ids is None:
            dispatched_job_ids = set()
            request.app.state.semantic_index_dispatched_job_ids = dispatched_job_ids
    return request_lock, dispatched_job_ids


def _state_value(request: Request, name: str) -> Any | None:
    return getattr(request.app.state, name, None)


def _public_development_calibration_value(value: object) -> object | None:
    if isinstance(value, dict):
        return {
            key: sanitized
            for key, child in value.items()
            if key in _DEVELOPMENT_CALIBRATION_NESTED_PUBLIC_KEYS
            and (sanitized := _public_development_calibration_value(child)) is not None
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    # Per-fold records and other sequences can encode identities or rankings. The public status
    # surface intentionally exposes only aggregate counts and metrics.
    return None


def _public_development_calibration(
    calibration: object,
    current_feedback_revision: dict[str, int | str] | str,
) -> tuple[dict[str, Any] | None, bool]:
    if not isinstance(calibration, dict):
        return None, True
    public = {
        key: sanitized
        for key, value in calibration.items()
        if key in _DEVELOPMENT_CALIBRATION_PUBLIC_KEYS
        and (sanitized := _public_development_calibration_value(value)) is not None
    }
    recorded_revision = calibration.get("feedback_revision")
    revision_matches = bool(
        isinstance(recorded_revision, dict)
        and (
            recorded_revision == current_feedback_revision
            if isinstance(current_feedback_revision, dict)
            else recorded_revision.get("sha256") == current_feedback_revision
        )
    )
    revision_current = bool(
        revision_matches and calibration.get("feedback_revision_stable_during_evaluation") is True
    )
    public["feedback_revision"] = {
        "version": (
            recorded_revision.get("version") if isinstance(recorded_revision, dict) else None
        ),
        "row_count": (
            recorded_revision.get("row_count") if isinstance(recorded_revision, dict) else None
        ),
        "current": revision_current,
    }
    if not revision_current:
        public["status"] = "stale_feedback"
        public["passed"] = False
    return public, revision_current


def _public_semantic_evaluation(
    evaluation: object,
    current_feedback_revision: dict[str, int | str],
    current_development_feedback_revision: dict[str, int | str] | str | None = None,
) -> dict[str, Any] | None:
    if not isinstance(evaluation, dict):
        return None
    public = copy.deepcopy(evaluation)
    automatic = public.get("automatic_known_item")
    if isinstance(automatic, dict):
        automatic.pop("queries", None)
    development = public.get("direct_conflict_development_calibration")
    if development is not None:
        public_development, development_revision_current = _public_development_calibration(
            development,
            current_development_feedback_revision or {},
        )
        public["direct_conflict_development_calibration"] = public_development
        if not development_revision_current:
            public["production_acceptance_passed"] = False
            public["decision"] = (
                "stale; development-calibration feedback changed and requires reevaluation"
            )
    held_out = public.get("held_out_feedback")
    if not isinstance(held_out, dict):
        public["passed"] = False
        public["production_acceptance_passed"] = False
        return public
    held_out.pop("queries", None)
    recorded_revision = held_out.get("feedback_revision")
    revision_current = bool(
        isinstance(recorded_revision, dict)
        and recorded_revision == current_feedback_revision
        and held_out.get("feedback_revision_stable_during_evaluation") is True
    )
    held_out["feedback_revision"] = {
        "version": (
            recorded_revision.get("version") if isinstance(recorded_revision, dict) else None
        ),
        "row_count": (
            recorded_revision.get("row_count") if isinstance(recorded_revision, dict) else None
        ),
        "current": revision_current,
    }
    held_out["feedback_revision_current"] = revision_current
    if not revision_current:
        public["passed"] = False
        public["production_acceptance_passed"] = False
        public["decision"] = "stale; held-out feedback changed and requires reevaluation"
        held_out["status"] = "stale_feedback"
        held_out["passed"] = False
        argument = held_out.get("argument_direct_conflict")
        if isinstance(argument, dict):
            argument["status"] = "stale_feedback"
            argument["passed"] = False
            argument["selection_complete"] = False
    return public


def _public_semantic_audit(
    audit: object,
    current_feedback_revision: dict[str, int | str],
    current_development_feedback_revision: dict[str, int | str] | str | None = None,
) -> dict[str, Any]:
    public = copy.deepcopy(audit) if isinstance(audit, dict) else {}
    for key in ("semantic_ir_evaluation", "accepted_semantic_ir_evaluation"):
        if key in public:
            public[key] = _public_semantic_evaluation(
                public[key],
                current_feedback_revision,
                current_development_feedback_revision,
            )
    return public


def _cached_search_embedder_factory(request: Request) -> Any:
    cache = getattr(request.app.state, "search_embedder_cache", None)
    if cache is None:
        cache = {}
        request.app.state.search_embedder_cache = cache
    lock = getattr(request.app.state, "search_embedder_lock", None)
    if lock is None:
        lock = threading.Lock()
        request.app.state.search_embedder_lock = lock
    configured_factory = _state_value(request, "search_embedder_factory")

    def get_embedder(spec: EmbeddingModelSpec) -> TextEmbedder:
        key = (spec.model_name, spec.model_revision, spec.dimensions)
        with lock:
            existing = cache.get(key)
            if existing is not None:
                return cast(TextEmbedder, existing)
            created = cast(
                TextEmbedder,
                configured_factory(spec)
                if configured_factory is not None
                else default_text_embedder(
                    request.app.state.layout,
                    spec,
                    local_files_only=bool(
                        getattr(request.app.state, "acceptance_read_only", False)
                    ),
                ),
            )
            cache[key] = created
            return created

    return get_embedder


def _cached_search_reranker_factory(request: Request) -> Any:
    cache = getattr(request.app.state, "search_reranker_cache", None)
    if cache is None:
        cache = {}
        request.app.state.search_reranker_cache = cache
    lock = getattr(request.app.state, "search_reranker_lock", None)
    if lock is None:
        lock = threading.Lock()
        request.app.state.search_reranker_lock = lock
    configured_factory = _state_value(request, "search_reranker_factory")

    def get_reranker(spec: RerankerSpec) -> TextReranker:
        key = (spec.model_name, spec.model_revision, spec.device)
        with lock:
            existing = cache.get(key)
            if existing is not None:
                return cast(TextReranker, existing)
            try:
                created = cast(
                    TextReranker,
                    configured_factory(spec)
                    if configured_factory is not None
                    else TransformersCrossEncoderReranker(
                        request.app.state.layout.models / "reranker" / spec.model_revision,
                        spec,
                        local_files_only=bool(
                            getattr(request.app.state, "acceptance_read_only", False)
                        ),
                    ),
                )
            except Exception as exc:
                request.app.state.search_reranker_error = f"{type(exc).__name__}: {exc}"[:1_000]
                raise
            cache[key] = created
            request.app.state.search_reranker_error = None
            return created

    return get_reranker


def _scope_hash(
    query: str,
    mode: str,
    filters: SearchFilters,
    *,
    semantic_config_id: str | None,
    direct_conflict_revision_sha256: str | None = None,
) -> str:
    payload = "\x1f".join(
        (
            query,
            mode,
            filters.entity_type,
            filters.conversation_id or "",
            filters.participant or "",
            filters.source_category or "",
            str(filters.from_ms),
            str(filters.to_ms),
            semantic_config_id or "",
            DEFAULT_RERANKER_SPEC.model_revision,
            direct_conflict_revision_sha256 or "",
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _current_direct_conflict_revision_sha256(request: Request) -> str:
    try:
        return str(development_feedback_revision_payload(request.app.state.engine)["sha256"])
    except DirectConflictRankerError as exc:
        raise HTTPException(
            status_code=409,
            detail="Direct-conflict calibration feedback is unavailable",
        ) from exc


def _run_semantic_index_job(
    *,
    layout: ArchiveLayout,
    engine: Engine,
    job_id: str,
    model: str | None,
    rebuild_windows: bool,
    embedder_factory: Any | None,
    reranker_factory: Any | None,
    dispatch_done: Callable[[], None] | None = None,
) -> None:
    try:
        build_semantic_index(
            layout,
            engine,
            embedder_factory=embedder_factory,
            reranker_factory=reranker_factory,
            model_alias=model,
            rebuild_windows=rebuild_windows,
            job_id=job_id,
        )
    except (DerivedPublicationBlockedError, DeletionError):
        return
    except BaseException as exc:
        with Session(engine) as session, session.begin():
            session.execute(
                update(Job)
                .where(Job.id == job_id, Job.status.in_(("queued", "running")))
                .values(
                    status="failed",
                    stage="Semantic index failed",
                    error_text=str(exc),
                    finished_at=datetime.now(UTC),
                )
            )
    finally:
        if dispatch_done is not None:
            dispatch_done()


@search_router.get("/search")
def search(
    request: Request,
    q: str = Query(min_length=1, max_length=2_000),
    mode: SearchMode = "keyword",
    entity_type: Literal[
        "all", "message", "activity", "audio_transcript", "attachment_text"
    ] = "all",
    conversation_id: str | None = Query(default=None, max_length=128),
    participant: str | None = Query(default=None, max_length=200),
    source_category: str | None = Query(default=None, max_length=80),
    from_ms: int | None = Query(default=None, ge=0),
    to_ms: int | None = Query(default=None, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    cursor: str | None = Query(default=None, max_length=2_048),
) -> dict[str, Any]:
    if from_ms is not None and to_ms is not None and from_ms > to_ms:
        raise HTTPException(status_code=422, detail="Invalid date range")
    filters = SearchFilters(
        entity_type=entity_type,
        conversation_id=conversation_id,
        participant=participant,
        source_category=source_category,
        from_ms=from_ms,
        to_ms=to_ms,
    )
    semantic_config_id: str | None = None
    direct_conflict_revision_sha256: str | None = None
    if mode in {"semantic", "hybrid"}:
        with _session(request) as session:
            semantic_config_id = session.scalar(
                select(SemanticIndexConfig.id).where(
                    SemanticIndexConfig.active.is_(True),
                    SemanticIndexConfig.status == "ready",
                )
            )
        if is_direct_conflict_trigger(q):
            try:
                direct_conflict_revision_sha256 = str(
                    development_feedback_revision_payload(request.app.state.engine)["sha256"]
                )
            except DirectConflictRankerError as exc:
                raise HTTPException(
                    status_code=409,
                    detail="Direct-conflict calibration feedback is unavailable",
                ) from exc
    scope = _scope_hash(
        q,
        mode,
        filters,
        semantic_config_id=semantic_config_id,
        direct_conflict_revision_sha256=direct_conflict_revision_sha256,
    )
    offset = 0
    if cursor:
        try:
            decoded = decode_cursor(cursor, kind="search")
            if require_cursor_str(decoded, "scope") != scope:
                raise InvalidCursor("Pagination cursor does not match the current search")
            offset = require_cursor_int(decoded, "offset")
            if offset < 0:
                raise InvalidCursor("Invalid search cursor")
        except InvalidCursor as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        page = run_search(
            request.app.state.layout,
            request.app.state.engine,
            q,
            mode=mode,
            filters=filters,
            # Hybrid/RRF ranks depend on the candidate pool. Use one fixed result horizon for
            # every cursor page so later requests cannot reorder earlier results as the offset
            # grows. The cap also bounds local vector/FTS work and response-memory usage.
            limit=SEARCH_RESULT_CAP,
            embedder_factory=_cached_search_embedder_factory(request),
            reranker_factory=_cached_search_reranker_factory(request),
            acceptance_read_only=bool(getattr(request.app.state, "acceptance_read_only", False)),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SearchUnavailableError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if mode in {"semantic", "hybrid"} and (
        page.index_config is None or page.index_config.get("id") != semantic_config_id
    ):
        raise HTTPException(
            status_code=409,
            detail="The active semantic index changed during this search; retry the request",
        )
    if direct_conflict_revision_sha256 is not None and (
        _current_direct_conflict_revision_sha256(request) != direct_conflict_revision_sha256
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                "Direct-conflict calibration feedback changed during this search; retry the request"
            ),
        )
    items = page.items[offset : offset + limit]
    next_cursor = (
        encode_cursor("search", scope=scope, offset=offset + limit)
        if len(page.items) > offset + limit
        else None
    )
    return {
        "items": items,
        "page": {
            "limit": limit,
            "returned": len(items),
            "candidate_count": page.total_candidates,
            "result_cap": SEARCH_RESULT_CAP,
            "next_cursor": next_cursor,
        },
        "query": q,
        "mode": mode,
        "filters": asdict(filters),
        "index_config": page.index_config,
    }


@search_router.get("/search/derived-occurrences")
def search_derived_occurrences(
    request: Request,
    entity_type: Literal["audio_transcript", "attachment_text"],
    source_id: str = Query(min_length=1, max_length=128),
    conversation_id: str | None = Query(default=None, max_length=128),
    participant: str | None = Query(default=None, max_length=200),
    source_category: str | None = Query(default=None, max_length=80),
    from_ms: int | None = Query(default=None, ge=0),
    to_ms: int | None = Query(default=None, ge=0),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
) -> dict[str, Any]:
    if from_ms is not None and to_ms is not None and from_ms > to_ms:
        raise HTTPException(status_code=422, detail="Invalid date range")
    try:
        return derived_occurrence_page(
            request.app.state.engine,
            entity_type=entity_type,
            source_id=source_id,
            filters=SearchFilters(
                entity_type=entity_type,
                conversation_id=conversation_id,
                participant=participant,
                source_category=source_category,
                from_ms=from_ms,
                to_ms=to_ms,
            ),
            offset=offset,
            limit=limit,
        )
    except SearchUnavailableError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@search_router.get("/search/status")
def search_status(request: Request) -> dict[str, Any]:
    with _session(request) as session:
        active = session.scalar(
            select(SemanticIndexConfig).where(SemanticIndexConfig.active.is_(True))
        )
        latest_stale = session.scalar(
            select(SemanticIndexConfig)
            .where(SemanticIndexConfig.status == "stale")
            .order_by(
                desc(SemanticIndexConfig.updated_at),
                desc(SemanticIndexConfig.created_at),
                desc(SemanticIndexConfig.id),
            )
            .limit(1)
        )
        latest_job = session.scalar(
            select(Job)
            .where(Job.kind == "semantic_index")
            .order_by(desc(Job.started_at), desc(Job.created_at), desc(Job.id))
            .limit(1)
        )
        latest_config: SemanticIndexConfig | None = None
        if latest_job is not None:
            latest_config_id = latest_job.checkpoint_json.get(
                "config_hash"
            ) or latest_job.counters_json.get("config_hash")
            if isinstance(latest_config_id, str):
                latest_config = session.get(SemanticIndexConfig, latest_config_id)
    current_feedback_revision = held_out_feedback_revision(request.app.state.engine)
    try:
        current_development_feedback_revision = development_feedback_revision_payload(
            request.app.state.engine
        )
    except DirectConflictRankerError:
        current_development_feedback_revision = {}
    active_audit = (
        _public_semantic_audit(
            active.audit_json,
            current_feedback_revision,
            current_development_feedback_revision,
        )
        if active
        else None
    )
    latest_audit = (
        _public_semantic_audit(
            latest_config.audit_json,
            current_feedback_revision,
            current_development_feedback_revision,
        )
        if latest_config
        else None
    )
    active_acceptance = (
        active_audit.get("accepted_semantic_ir_evaluation") if active_audit else None
    )
    latest_evaluation = latest_audit.get("semantic_ir_evaluation") if latest_audit else None
    reranker_cache = getattr(request.app.state, "search_reranker_cache", {})
    reranker = next(iter(reranker_cache.values()), None)
    reranker_error = getattr(request.app.state, "search_reranker_error", None)
    ready = bool(active and active.status == "ready")
    stale_detail = latest_stale.audit_json.get("stale") if latest_stale else None
    return {
        "ready": ready,
        "needs_rebuild": bool(not ready and latest_stale),
        "stale_reason": stale_detail if isinstance(stale_detail, dict) else None,
        "stale_config": (
            {
                "id": latest_stale.id,
                "status": latest_stale.status,
                "model_name": latest_stale.model_name,
                "model_revision": latest_stale.model_revision,
                "window_count": latest_stale.window_count,
            }
            if latest_stale
            else None
        ),
        "active": (
            {
                "id": active.id,
                "status": active.status,
                "model_name": active.model_name,
                "model_revision": active.model_revision,
                "dimensions": active.dimensions,
                "window_version": active.window_version,
                "window_count": active.window_count,
                "audit": active_audit,
            }
            if active
            else None
        ),
        "acceptance": active_acceptance,
        "latest_evaluation": latest_evaluation,
        "latest_candidate": (
            {
                "id": latest_config.id,
                "status": latest_config.status,
                "active": latest_config.active,
                "window_count": latest_config.window_count,
                "evaluation": latest_evaluation,
            }
            if latest_config
            else None
        ),
        "latest_job": (
            {
                "id": latest_job.id,
                "status": latest_job.status,
                "stage": latest_job.stage,
                "progress_current": latest_job.progress_current,
                "progress_total": latest_job.progress_total,
                "counters": latest_job.counters_json,
                "error": latest_job.error_text,
            }
            if latest_job
            else None
        ),
        "reranker": (
            reranker.status
            if reranker is not None
            else reranker_status_payload(
                status="unavailable" if reranker_error else "not_loaded",
                error=reranker_error,
            )
        ),
    }


@search_router.post("/search/index", status_code=status.HTTP_202_ACCEPTED)
def create_semantic_index(
    request: Request,
    payload: SemanticIndexRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    request_lock, dispatched_job_ids = _semantic_request_state(request)
    requested_checkpoint = {
        "requested_model": payload.model or "auto",
        "rebuild_windows": payload.rebuild_windows,
    }
    reused = False
    job_id: str | None = None
    try:
        with (
            guarded_derived_request(
                request.app.state.layout,
                request.app.state.engine,
                publisher="Semantic index request",
            ),
            request_lock,
            _session(request) as session,
            session.begin(),
        ):
            active_jobs = list(
                session.scalars(
                    select(Job)
                    .where(
                        Job.kind == "semantic_index",
                        Job.status.in_(("queued", "running")),
                    )
                    .order_by(Job.created_at)
                )
            )
            lock_held = semantic_index_lock_is_held(request.app.state.layout)
            reusable_id: str | None = None
            for active_job in active_jobs:
                pristine_queued = (
                    active_job.status == "queued"
                    and active_job.stage == "Semantic index queued"
                    and active_job.progress_current == 0
                    and active_job.progress_total is None
                    and active_job.checkpoint_json == requested_checkpoint
                    and active_job.counters_json == {}
                    and active_job.error_text is None
                    and active_job.started_at is None
                    and active_job.finished_at is None
                )
                if pristine_queued and active_job.id not in dispatched_job_ids:
                    if reusable_id is not None:
                        raise HTTPException(
                            status_code=409,
                            detail=f"Semantic index job {active_job.id} is already active",
                        )
                    reusable_id = active_job.id
                    continue
                if active_job.status == "running" and not lock_held:
                    active_job.status = "failed"
                    active_job.stage = "Interrupted semantic index recovered"
                    active_job.error_text = (
                        "Recovered because no process owns the semantic-index build lock"
                    )
                    active_job.finished_at = datetime.now(UTC)
                    continue
                raise HTTPException(
                    status_code=409,
                    detail=f"Semantic index job {active_job.id} is already active",
                )
            if reusable_id is not None:
                job_id = reusable_id
                reused = True
            else:
                job_id = str(uuid.uuid4())
                session.add(
                    Job(
                        id=job_id,
                        kind="semantic_index",
                        status="queued",
                        stage="Semantic index queued",
                        progress_current=0,
                        checkpoint_json=requested_checkpoint,
                        counters_json={},
                    )
                )
            dispatched_job_ids.add(job_id)
    except BaseException:
        if job_id is not None:
            with request_lock:
                dispatched_job_ids.discard(job_id)
        raise
    assert job_id is not None

    def dispatch_done() -> None:
        with request_lock:
            dispatched_job_ids.discard(job_id)

    factory = _state_value(request, "semantic_index_embedder_factory")
    try:
        background_tasks.add_task(
            _run_semantic_index_job,
            layout=request.app.state.layout,
            engine=request.app.state.engine,
            job_id=job_id,
            model=payload.model,
            rebuild_windows=payload.rebuild_windows,
            embedder_factory=factory,
            reranker_factory=_cached_search_reranker_factory(request),
            dispatch_done=dispatch_done,
        )
    except BaseException:
        dispatch_done()
        raise
    return {
        "job_id": job_id,
        "status": "queued",
        "reused": reused,
        "status_href": "/api/v1/search/status",
    }


@search_router.get("/saved-searches")
def saved_searches(request: Request) -> dict[str, Any]:
    with _session(request) as session:
        rows = list(session.scalars(select(SavedSearch).order_by(SavedSearch.name, SavedSearch.id)))
    return {"items": [{"id": row.id, "name": row.name, "query": row.query_json} for row in rows]}


@search_router.post("/saved-searches", status_code=status.HTTP_201_CREATED)
def create_saved_search(request: Request, payload: SavedSearchCreate) -> dict[str, Any]:
    row = SavedSearch(id=str(uuid.uuid4()), name=payload.name.strip(), query_json=payload.query)
    if not row.name:
        raise HTTPException(status_code=422, detail="Saved-search name must not be blank")
    try:
        with _session(request) as session, session.begin():
            session.add(row)
            session.flush()
            result = {"id": row.id, "name": row.name, "query": row.query_json}
    except IntegrityError as exc:
        raise HTTPException(status_code=409, detail="A saved search with this name exists") from exc
    return result


@search_router.patch("/saved-searches/{saved_search_id}")
def update_saved_search(
    request: Request, saved_search_id: str, payload: SavedSearchUpdate
) -> dict[str, Any]:
    try:
        with _session(request) as session, session.begin():
            row = session.get(SavedSearch, saved_search_id)
            if row is None:
                raise HTTPException(status_code=404, detail="Saved search not found")
            if payload.name is not None:
                name = payload.name.strip()
                if not name:
                    raise HTTPException(
                        status_code=422, detail="Saved-search name must not be blank"
                    )
                row.name = name
            if payload.query is not None:
                row.query_json = payload.query
            session.flush()
            result = {"id": row.id, "name": row.name, "query": row.query_json}
    except IntegrityError as exc:
        raise HTTPException(status_code=409, detail="A saved search with this name exists") from exc
    return result


@search_router.delete("/saved-searches/{saved_search_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_saved_search(request: Request, saved_search_id: str) -> None:
    with _session(request) as session, session.begin():
        row = session.get(SavedSearch, saved_search_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Saved search not found")
        session.delete(row)


def _feedback_payload(row: RelevanceFeedback) -> dict[str, Any]:
    return {
        "id": row.id,
        "query_sha256": row.query_sha256,
        "query": row.query_text,
        "search_mode": row.search_mode,
        "filter_scope": row.filter_scope_json,
        "result_id": row.result_id,
        "label": row.label,
        "evaluation_split": row.evaluation_split,
        "evaluation_intent": row.evaluation_intent,
        "notes": row.notes,
        "created_at": row.created_at,
    }


def _feedback_scope(payload: RelevanceFeedbackCreate) -> tuple[dict[str, Any], str]:
    scope = payload.filter_scope.model_dump(exclude_none=True, exclude_defaults=True)
    rendered = json.dumps(scope, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return scope, hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _validate_feedback_evaluation_intent(
    evaluation_intent: str | None,
    *,
    evaluation_split: str | None,
    search_mode: str,
    filter_scope: dict[str, Any],
) -> None:
    if evaluation_intent is None:
        return
    if evaluation_intent == "argument_direct_conflict":
        if evaluation_split == "test" and search_mode in {"semantic", "hybrid"}:
            return
        detail = "Evaluation intent requires test-split semantic or hybrid feedback"
    elif evaluation_intent == "argument_direct_conflict_development":
        if evaluation_split == "validation" and search_mode == "semantic" and filter_scope == {}:
            return
        detail = (
            "Development-calibration intent requires validation-split semantic feedback "
            "with the default filter scope"
        )
    else:  # pragma: no cover - Pydantic rejects unknown intent values at the request boundary
        detail = "Unsupported evaluation intent"
    raise HTTPException(status_code=422, detail=detail)


@search_router.get("/search/feedback")
def list_relevance_feedback(
    request: Request,
    query: str | None = Query(default=None, min_length=1, max_length=2_000),
    search_mode: Literal["all", "keyword", "exact", "semantic", "hybrid"] = "all",
    label: Literal["all", "relevant", "not_relevant", "uncertain"] = "all",
    evaluation_split: Literal["all", "unassigned", "train", "validation", "test"] = "all",
    limit: int = Query(default=50, ge=1, le=200),
    cursor: str | None = Query(default=None, max_length=2_048),
) -> dict[str, Any]:
    query_sha256 = hashlib.sha256(query.encode("utf-8")).hexdigest() if query else None
    scope = hashlib.sha256(
        "\x1f".join((query_sha256 or "", search_mode, label, evaluation_split)).encode("utf-8")
    ).hexdigest()[:20]
    offset = 0
    if cursor:
        try:
            decoded = decode_cursor(cursor, kind="relevance-feedback")
            if require_cursor_str(decoded, "scope") != scope:
                raise InvalidCursor("Pagination cursor does not match the current feedback filters")
            offset = require_cursor_int(decoded, "offset")
            if offset < 0:
                raise InvalidCursor("Invalid feedback cursor")
        except InvalidCursor as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    statement = select(RelevanceFeedback).order_by(
        desc(RelevanceFeedback.created_at), desc(RelevanceFeedback.id)
    )
    if query_sha256:
        statement = statement.where(RelevanceFeedback.query_sha256 == query_sha256)
    if search_mode != "all":
        statement = statement.where(RelevanceFeedback.search_mode == search_mode)
    if label != "all":
        statement = statement.where(RelevanceFeedback.label == label)
    if evaluation_split == "unassigned":
        statement = statement.where(RelevanceFeedback.evaluation_split.is_(None))
    elif evaluation_split != "all":
        statement = statement.where(RelevanceFeedback.evaluation_split == evaluation_split)
    with _session(request) as session:
        rows = list(session.scalars(statement.offset(offset).limit(limit + 1)))
    items = [_feedback_payload(row) for row in rows[:limit]]
    return {
        "items": items,
        "page": {
            "limit": limit,
            "returned": len(items),
            "next_cursor": (
                encode_cursor("relevance-feedback", scope=scope, offset=offset + limit)
                if len(rows) > limit
                else None
            ),
        },
    }


@search_router.post("/search/feedback", status_code=status.HTTP_201_CREATED)
def relevance_feedback(request: Request, payload: RelevanceFeedbackCreate) -> dict[str, Any]:
    query_sha256 = hashlib.sha256(payload.query.encode("utf-8")).hexdigest()
    filter_scope, filter_scope_sha256 = _feedback_scope(payload)
    _validate_feedback_evaluation_intent(
        payload.evaluation_intent,
        evaluation_split=payload.evaluation_split,
        search_mode=payload.search_mode,
        filter_scope=filter_scope,
    )
    row_id = str(uuid.uuid4())
    with (
        guarded_derived_request(
            request.app.state.layout,
            request.app.state.engine,
            publisher="Search relevance feedback publication",
        ),
        _session(request) as session,
        session.begin(),
    ):
        session.execute(
            sqlite_insert(RelevanceFeedback)
            .values(
                id=row_id,
                query_sha256=query_sha256,
                query_text=payload.query,
                search_mode=payload.search_mode,
                filter_scope_sha256=filter_scope_sha256,
                filter_scope_json=filter_scope,
                result_id=payload.result_id,
                label=payload.label,
                evaluation_split=payload.evaluation_split,
                evaluation_intent=payload.evaluation_intent,
                notes=payload.notes,
            )
            .on_conflict_do_update(
                index_elements=[
                    "query_sha256",
                    "search_mode",
                    "filter_scope_sha256",
                    "result_id",
                ],
                set_={
                    "query_text": payload.query,
                    "filter_scope_json": filter_scope,
                    "label": payload.label,
                    "evaluation_split": payload.evaluation_split,
                    "evaluation_intent": payload.evaluation_intent,
                    "notes": payload.notes,
                },
            )
        )
        row = session.scalar(
            select(RelevanceFeedback).where(
                RelevanceFeedback.query_sha256 == query_sha256,
                RelevanceFeedback.search_mode == payload.search_mode,
                RelevanceFeedback.filter_scope_sha256 == filter_scope_sha256,
                RelevanceFeedback.result_id == payload.result_id,
            )
        )
        if row is None:  # pragma: no cover - protected by the unique upsert
            raise RuntimeError("Relevance feedback upsert did not persist a row")
        result = {**_feedback_payload(row), "updated": row.id != row_id}
    return result


@search_router.patch("/search/feedback/{feedback_id}")
def update_relevance_feedback(
    request: Request, feedback_id: str, payload: RelevanceFeedbackUpdate
) -> dict[str, Any]:
    if not payload.model_fields_set:
        raise HTTPException(status_code=422, detail="Provide at least one feedback field to update")
    with (
        guarded_derived_request(
            request.app.state.layout,
            request.app.state.engine,
            publisher="Search relevance feedback publication",
        ),
        _session(request) as session,
        session.begin(),
    ):
        row = session.get(RelevanceFeedback, feedback_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Relevance feedback not found")
        next_split = (
            payload.evaluation_split
            if "evaluation_split" in payload.model_fields_set
            else row.evaluation_split
        )
        next_intent = (
            payload.evaluation_intent
            if "evaluation_intent" in payload.model_fields_set
            else row.evaluation_intent
        )
        _validate_feedback_evaluation_intent(
            next_intent,
            evaluation_split=next_split,
            search_mode=row.search_mode,
            filter_scope=row.filter_scope_json,
        )
        if "label" in payload.model_fields_set:
            if payload.label is None:
                raise HTTPException(status_code=422, detail="Feedback label cannot be null")
            row.label = payload.label
        if "evaluation_split" in payload.model_fields_set:
            row.evaluation_split = payload.evaluation_split
        if "evaluation_intent" in payload.model_fields_set:
            row.evaluation_intent = payload.evaluation_intent
        if "notes" in payload.model_fields_set:
            row.notes = payload.notes
        session.flush()
        result = _feedback_payload(row)
    return result


@search_router.delete("/search/feedback/{feedback_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_relevance_feedback(request: Request, feedback_id: str) -> None:
    with (
        guarded_derived_request(
            request.app.state.layout,
            request.app.state.engine,
            publisher="Search relevance feedback publication",
        ),
        _session(request) as session,
        session.begin(),
    ):
        row = session.get(RelevanceFeedback, feedback_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Relevance feedback not found")
        session.delete(row)


@search_router.get("/rag/audits")
def rag_audits(
    request: Request,
    status_filter: Literal["all", "completed", "failed"] = Query(default="all", alias="status"),
    limit: int = Query(default=50, ge=1, le=200),
    cursor: str | None = Query(default=None, max_length=2_048),
) -> dict[str, Any]:
    cursor_created_at_text: str | None = None
    cursor_id: str | None = None
    if cursor:
        try:
            decoded = decode_cursor(cursor, kind="rag-audits")
            if require_cursor_str(decoded, "status") != status_filter:
                raise InvalidCursor("Pagination cursor does not match the current audit filter")
            if require_cursor_int(decoded, "limit") != limit:
                raise InvalidCursor("Pagination cursor does not match the requested page size")
            cursor_created_at_text = require_cursor_str(decoded, "created_at")
            datetime.fromisoformat(cursor_created_at_text)
            cursor_id = require_cursor_str(decoded, "id")
        except (InvalidCursor, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    audit_created_at = func.datetime(RagAudit.created_at)
    statement = select(RagAudit)
    if status_filter != "all":
        statement = statement.where(RagAudit.status == status_filter)
    if cursor_created_at_text is not None and cursor_id is not None:
        cursor_created_at = func.datetime(cursor_created_at_text)
        statement = statement.where(
            or_(
                audit_created_at < cursor_created_at,
                and_(audit_created_at == cursor_created_at, RagAudit.id < cursor_id),
            )
        )
    with _session(request) as session:
        rows = list(
            session.scalars(
                statement.order_by(desc(audit_created_at), desc(RagAudit.id)).limit(limit + 1)
            )
        )
    page_rows = rows[:limit]
    items = [
        {
            "id": row.id,
            "status": row.status,
            "model": row.model,
            "query_sha256": row.query_sha256,
            "search_mode": row.search_mode,
            "retrieved_ids": row.retrieved_ids_json.get("ids", []),
            "citation_ids": row.citation_ids_json.get("ids", []),
            "remote_response_id": row.remote_response_id,
            "usage": row.usage_json,
            "error": row.error_text,
            "created_at": row.created_at,
        }
        for row in page_rows
    ]
    next_cursor = None
    if len(rows) > limit and page_rows:
        last = page_rows[-1]
        next_cursor = encode_cursor(
            "rag-audits",
            status=status_filter,
            limit=limit,
            created_at=last.created_at.isoformat(),
            id=last.id,
        )
    return {
        "items": items,
        "page": {
            "limit": limit,
            "returned": len(items),
            "next_cursor": next_cursor,
        },
    }


@search_router.post("/rag/answer")
def rag_answer(request: Request, payload: RagQuestion) -> dict[str, Any]:
    if (
        payload.from_ms is not None
        and payload.to_ms is not None
        and payload.from_ms > payload.to_ms
    ):
        raise HTTPException(status_code=422, detail="Invalid date range")
    filters = SearchFilters(
        entity_type=payload.entity_type,
        conversation_id=payload.conversation_id,
        participant=payload.participant,
        source_category=payload.source_category,
        from_ms=payload.from_ms,
        to_ms=payload.to_ms,
    )
    try:
        with guarded_derived_request(
            request.app.state.layout,
            request.app.state.engine,
            publisher="RAG audit publication",
        ):
            return answer_with_rag(
                request.app.state.layout,
                request.app.state.engine,
                payload.question,
                search_mode=payload.search_mode,
                filters=filters,
                top_k=payload.top_k,
                embedder_factory=_cached_search_embedder_factory(request),
                reranker_factory=_cached_search_reranker_factory(request),
                client_factory=_state_value(request, "openai_client_factory"),
            )
    except HTTPException:
        raise
    except (CloudRagDisabledError, SearchUnavailableError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RagCitationError as exc:
        raise HTTPException(
            status_code=502, detail="Cloud RAG response failed citation validation"
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Cloud RAG request failed") from exc
