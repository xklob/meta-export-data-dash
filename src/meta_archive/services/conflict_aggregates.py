from __future__ import annotations

import fcntl
import hashlib
import json
import math
import uuid
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from sqlalchemy import Engine, func, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from meta_archive.models import (
    ConflictAggregateRun,
    ConflictHourlyRollup,
    ConflictWindowScore,
    Job,
    Message,
    SearchWindow,
    SearchWindowMessage,
    SemanticIndexConfig,
)
from meta_archive.paths import ArchiveLayout
from meta_archive.services.conflict_scoring import (
    CONFLICT_LABELS,
    CONFLICT_WARNINGS,
    DEFAULT_CONFLICT_SPEC,
    ConflictScorerFactory,
    ModernBertNliConflictScorer,
)
from meta_archive.services.deletion_lock_lease import deletion_lock
from meta_archive.services.derived_publication_guard import assert_derived_publication_allowed
from meta_archive.services.identifiers import canonical_fingerprint, stable_id
from meta_archive.services.nlp_annotations import conflict_aggregation_status
from meta_archive.services.semantic_state import semantic_window_membership_clause

CONFLICT_AGGREGATE_PIPELINE_VERSION = "conflict-aggregate-v1"
DEFAULT_AGGREGATE_BATCH_SIZE = 32
MAX_AGGREGATE_BATCH_SIZE = 100
AGGREGATE_CANDIDATE_CHUNK_SIZE = 256
MAX_ERROR_LENGTH = 4_000
DIRECT_CONFLICT_LABEL: Literal["direct conflict"] = "direct conflict"
ProgressCallback = Callable[[str, int, int | None], None]

AGGREGATE_WARNINGS = (
    "Conflict classifications are fallible local model estimates, not facts.",
    "The displayed rate is the share of classified text windows whose top label is "
    "direct conflict.",
    "Window overlap and conversational context mean this is not a rate of arguments or events.",
    "No person, conversation, or relationship may be ranked by these values.",
    *CONFLICT_WARNINGS[:2],
)


@dataclass(frozen=True, slots=True)
class ConflictAggregateSummary:
    job_id: str
    run_id: str
    status: str
    window_count: int
    scored: int
    reused: int
    direct_conflict_count: int
    rollup_count: int
    resumed: bool
    configuration_sha256: str


@dataclass(frozen=True, slots=True)
class _BuildContext:
    calibration_id: str
    semantic_config_id: str
    semantic_corpus_revision: str | None
    corpus_revision: str
    source_revision: str
    window_count: int


def conflict_aggregate_lock_is_held(layout: ArchiveLayout) -> bool:
    """Probe the aggregate lock without disturbing its owner."""

    if layout.catalog.is_symlink() or not layout.catalog.is_dir():
        return False
    lock_path = layout.catalog / "conflict-aggregates.lock"
    with lock_path.open("a+b") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    return False


def _live_message_window_statement(config_id: str) -> Any:
    live_message_count = (
        select(func.count(SearchWindowMessage.id))
        .join(Message, Message.id == SearchWindowMessage.message_id)
        .where(
            SearchWindowMessage.window_id == SearchWindow.id,
            Message.deleted_at.is_(None),
        )
        .correlate(SearchWindow)
        .scalar_subquery()
    )
    return select(SearchWindow).where(
        SearchWindow.entity_type == "message_window",
        semantic_window_membership_clause(config_id),
        SearchWindow.turn_count > 0,
        live_message_count == SearchWindow.turn_count,
    )


def _source_revision(session: Session) -> str:
    live_count, maximum_pk, maximum_timestamp = session.execute(
        select(func.count(Message.pk), func.max(Message.pk), func.max(Message.timestamp_ms)).where(
            Message.deleted_at.is_(None)
        )
    ).one()
    deleted_count = session.scalar(
        select(func.count(Message.pk)).where(Message.deleted_at.is_not(None))
    )
    return canonical_fingerprint(
        "conflict-source-revision-v1",
        {
            "live_message_count": int(live_count or 0),
            "deleted_message_count": int(deleted_count or 0),
            "maximum_message_pk": int(maximum_pk or 0),
            "maximum_timestamp_ms": int(maximum_timestamp or 0),
        },
    )


def _context(engine: Engine, *, require_gate: bool) -> _BuildContext:
    gate = conflict_aggregation_status(engine)
    calibration = gate.get("latest_calibration")
    if require_gate and not gate.get("aggregation_enabled"):
        raise RuntimeError(
            gate.get("reason") or "Conflict aggregate calibration gate has not passed"
        )
    if not isinstance(calibration, dict) or not isinstance(calibration.get("id"), str):
        raise RuntimeError("No current conflict calibration is available")
    with Session(engine) as session:
        config = session.scalar(
            select(SemanticIndexConfig)
            .where(
                SemanticIndexConfig.active.is_(True),
                SemanticIndexConfig.status == "ready",
            )
            .order_by(SemanticIndexConfig.completed_at.desc(), SemanticIndexConfig.id.desc())
            .limit(1)
        )
        if config is None:
            raise RuntimeError("A ready semantic index is required before conflict aggregation")
        digest = hashlib.sha256()
        window_count = 0
        rows = session.execute(
            _live_message_window_statement(config.id)
            .with_only_columns(
                SearchWindow.id,
                SearchWindow.first_timestamp_ms,
                SearchWindow.last_timestamp_ms,
                SearchWindow.turn_count,
                SearchWindow.text_display,
                SearchWindow.text_search,
            )
            .order_by(SearchWindow.id)
            .execution_options(yield_per=AGGREGATE_CANDIDATE_CHUNK_SIZE)
        )
        for row in rows:
            digest.update(
                json.dumps(tuple(row), ensure_ascii=False, separators=(",", ":")).encode(
                    "utf-8", errors="surrogatepass"
                )
            )
            digest.update(b"\n")
            window_count += 1
        audit = config.audit_json if isinstance(config.audit_json, dict) else {}
        semantic_corpus_revision = audit.get("corpus_revision")
        return _BuildContext(
            calibration_id=str(calibration["id"]),
            semantic_config_id=config.id,
            semantic_corpus_revision=(
                semantic_corpus_revision if isinstance(semantic_corpus_revision, str) else None
            ),
            corpus_revision=digest.hexdigest(),
            source_revision=_source_revision(session),
            window_count=window_count,
        )


def _configuration(context: _BuildContext) -> tuple[dict[str, Any], str]:
    value = {
        "pipeline_version": CONFLICT_AGGREGATE_PIPELINE_VERSION,
        "model": asdict(DEFAULT_CONFLICT_SPEC),
        "calibration_id": context.calibration_id,
        "semantic_config_id": context.semantic_config_id,
        "semantic_corpus_revision": context.semantic_corpus_revision,
        "corpus_revision": context.corpus_revision,
        "source_revision": context.source_revision,
        "labels": list(CONFLICT_LABELS),
        "prediction": "maximum NLI entailment logit",
        "rollup_bucket": "UTC hour",
        "dimensions": ["time"],
        "relationship_dimensions": [],
    }
    return value, canonical_fingerprint("conflict-aggregate-configuration-v1", value)


def invalidate_conflict_aggregates(engine: Engine, *, reason: str) -> int:
    """Persistently retire published and in-progress conflict aggregate data."""

    now = datetime.now(UTC)
    with Session(engine) as session, session.begin():
        runs = list(
            session.scalars(
                select(ConflictAggregateRun).where(
                    ConflictAggregateRun.status.in_(("building", "ready"))
                )
            )
        )
        run_ids = [row.id for row in runs]
        for row in runs:
            row.active = False
            row.status = "stale"
            row.error_text = reason[:MAX_ERROR_LENGTH]
            row.provenance_json = {
                **(row.provenance_json if isinstance(row.provenance_json, dict) else {}),
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


def refresh_conflict_aggregate_staleness(engine: Engine) -> None:
    """Retire active output when its gate, corpus, source, or model context changed."""

    gate = conflict_aggregation_status(engine)
    calibration = gate.get("latest_calibration")
    with Session(engine) as session:
        active = session.scalar(
            select(ConflictAggregateRun).where(
                ConflictAggregateRun.active.is_(True),
                ConflictAggregateRun.status == "ready",
            )
        )
        config = session.scalar(
            select(SemanticIndexConfig)
            .where(
                SemanticIndexConfig.active.is_(True),
                SemanticIndexConfig.status == "ready",
            )
            .order_by(SemanticIndexConfig.completed_at.desc(), SemanticIndexConfig.id.desc())
            .limit(1)
        )
        source_revision = _source_revision(session)
    if active is None:
        return
    if not gate.get("aggregation_enabled"):
        invalidate_conflict_aggregates(
            engine,
            reason=gate.get("reason") or "Conflict calibration gate is no longer valid",
        )
        return
    mismatches: list[str] = []
    calibration_id = calibration.get("id") if isinstance(calibration, dict) else None
    if active.calibration_id != calibration_id:
        mismatches.append("calibration revision changed")
    if config is None or active.semantic_config_id != config.id:
        mismatches.append("semantic index configuration changed")
    elif isinstance(active.configuration_json, dict):
        recorded_semantic_revision = active.configuration_json.get("semantic_corpus_revision")
        current_semantic_revision = (
            config.audit_json.get("corpus_revision")
            if isinstance(config.audit_json, dict)
            else None
        )
        if recorded_semantic_revision != current_semantic_revision:
            mismatches.append("semantic corpus revision changed")
    if active.source_revision != source_revision:
        mismatches.append("source message data changed")
    if (
        active.model_name != DEFAULT_CONFLICT_SPEC.model_name
        or active.model_revision != DEFAULT_CONFLICT_SPEC.model_revision
    ):
        mismatches.append("conflict model revision changed")
    if mismatches:
        invalidate_conflict_aggregates(engine, reason="; ".join(mismatches))


def conflict_aggregate_status(engine: Engine, *, refresh: bool = True) -> dict[str, Any]:
    if refresh:
        refresh_conflict_aggregate_staleness(engine)
    gate = conflict_aggregation_status(engine)
    with Session(engine) as session:
        run = session.scalar(
            select(ConflictAggregateRun)
            .order_by(ConflictAggregateRun.created_at.desc(), ConflictAggregateRun.id.desc())
            .limit(1)
        )
        job = (
            session.get(Job, run.job_id)
            if run is not None and run.job_id is not None
            else session.scalar(
                select(Job)
                .where(Job.kind == "conflict_aggregates")
                .order_by(Job.created_at.desc())
                .limit(1)
            )
        )
    available = bool(
        gate.get("aggregation_enabled") and run is not None and run.active and run.status == "ready"
    )
    if available:
        status_value = "ready"
        reason = None
    elif job is not None and job.status in {"queued", "running"}:
        status_value = "building"
        reason = "The gated full-corpus conflict job is still running."
    elif run is not None and run.status in {"stale", "failed"}:
        status_value = run.status
        reason = run.error_text
    else:
        status_value = "unavailable"
        reason = gate.get("reason") or "No completed full-corpus conflict aggregate exists."
    return {
        "available": available,
        "status": status_value,
        "reason": reason,
        "gate": gate,
        "run": {
            "id": run.id,
            "status": run.status,
            "active": run.active,
            "pipeline_version": run.pipeline_version,
            "model_name": run.model_name,
            "model_revision": run.model_revision,
            "calibration_id": run.calibration_id,
            "semantic_config_id": run.semantic_config_id,
            "corpus_revision": run.corpus_revision,
            "source_revision": run.source_revision,
            "configuration_sha256": run.configuration_sha256,
            "window_count": run.window_count,
            "scored_count": run.scored_count,
            "direct_conflict_count": run.direct_conflict_count,
            "warnings": list((run.warnings_json or {}).get("items", [])),
            "completed_at": run.completed_at,
        }
        if run is not None
        else None,
        "job": {
            "id": job.id,
            "status": job.status,
            "stage": job.stage,
            "progress_current": job.progress_current,
            "progress_total": job.progress_total,
            "error": job.error_text,
        }
        if job is not None
        else None,
        "warnings": list(AGGREGATE_WARNINGS),
        "relationship_ranking_enabled": False,
        "relationship_dimensions": [],
    }


def _prepare_job(
    engine: Engine,
    *,
    requested_job_id: str | None,
    run_id: str,
    configuration_sha256: str,
    total: int,
) -> tuple[str, bool]:
    now = datetime.now(UTC)
    with Session(engine) as session, session.begin():
        job: Job | None
        for stale in session.scalars(
            select(Job).where(
                Job.kind == "conflict_aggregates",
                Job.status.in_(("queued", "running")),
                Job.id != requested_job_id,
            )
        ):
            stale.status = "failed"
            stale.stage = "Interrupted conflict aggregate job recovered"
            stale.error_text = "Recovered because the prior process lost the aggregate lock"
            stale.finished_at = now
        existing_run = session.get(ConflictAggregateRun, run_id)
        resumed = bool(existing_run is not None and existing_run.scored_count < total)
        if requested_job_id is None:
            previous = session.scalar(
                select(Job)
                .where(
                    Job.kind == "conflict_aggregates",
                    Job.status == "failed",
                )
                .order_by(Job.created_at.desc())
                .limit(1)
            )
            if previous is not None and previous.checkpoint_json.get("run_id") == run_id:
                job = previous
                resumed = True
            else:
                job = Job(id=str(uuid.uuid4()), kind="conflict_aggregates", status="running")
                session.add(job)
        else:
            job = session.get(Job, requested_job_id)
            if job is None:
                job = Job(
                    id=requested_job_id,
                    kind="conflict_aggregates",
                    status="running",
                )
                session.add(job)
        assert job is not None
        job.status = "running"
        job.stage = (
            "Resuming full-corpus conflict scoring" if resumed else "Scoring message windows"
        )
        job.progress_current = existing_run.scored_count if existing_run is not None else 0
        job.progress_total = total
        job.checkpoint_json = {
            "run_id": run_id,
            "configuration_sha256": configuration_sha256,
        }
        job.counters_json = {}
        job.error_text = None
        job.started_at = now
        job.finished_at = None
        return job.id, resumed


def _iter_candidate_chunks(
    engine: Engine, *, config_id: str, chunk_size: int = AGGREGATE_CANDIDATE_CHUNK_SIZE
) -> Iterator[list[tuple[str, int | None, str]]]:
    last_id: str | None = None
    while True:
        statement = (
            _live_message_window_statement(config_id)
            .with_only_columns(
                SearchWindow.id,
                SearchWindow.first_timestamp_ms,
                SearchWindow.text_display,
            )
            .order_by(SearchWindow.id)
            .limit(chunk_size)
        )
        if last_id is not None:
            statement = statement.where(SearchWindow.id > last_id)
        with Session(engine) as session:
            rows = [
                (str(window_id), timestamp_ms, str(text_display))
                for window_id, timestamp_ms, text_display in session.execute(statement)
            ]
        if not rows:
            return
        yield rows
        last_id = rows[-1][0]


def _score_input_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()


def _upsert_score_rows(engine: Engine, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    insert_statement = sqlite_insert(ConflictWindowScore)
    excluded = insert_statement.excluded
    statement = insert_statement.on_conflict_do_update(
        index_elements=["id"],
        set_={
            "input_sha256": excluded.input_sha256,
            "first_timestamp_ms": excluded.first_timestamp_ms,
            "predicted_label": excluded.predicted_label,
            "direct_conflict_score": excluded.direct_conflict_score,
            "scores_json": excluded.scores_json,
            "active": False,
            "provenance_json": excluded.provenance_json,
        },
    )
    with Session(engine) as session, session.begin():
        session.connection().execute(statement, rows)


def _publish_rollups(engine: Engine, *, run_id: str) -> tuple[int, int, int]:
    buckets: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    scored_count = direct_count = 0
    with Session(engine) as session:
        rows = session.execute(
            select(
                ConflictWindowScore.first_timestamp_ms,
                ConflictWindowScore.predicted_label,
            )
            .where(ConflictWindowScore.run_id == run_id)
            .execution_options(yield_per=AGGREGATE_CANDIDATE_CHUNK_SIZE)
        )
        for timestamp_ms, predicted_label in rows:
            scored_count += 1
            is_direct = int(predicted_label == DIRECT_CONFLICT_LABEL)
            direct_count += is_direct
            if timestamp_ms is not None:
                hour_start = int(timestamp_ms) - (int(timestamp_ms) % 3_600_000)
                buckets[hour_start][0] += 1
                buckets[hour_start][1] += is_direct
    now = datetime.now(UTC)
    rows_to_publish = [
        {
            "id": stable_id("conflict-hourly-rollup", run_id, hour),
            "run_id": run_id,
            "hour_start_ms": hour,
            "sample_count": counts[0],
            "direct_conflict_count": counts[1],
            "active": False,
            "generated_at": now,
        }
        for hour, counts in sorted(buckets.items())
    ]
    insert_statement = sqlite_insert(ConflictHourlyRollup)
    excluded = insert_statement.excluded
    upsert = insert_statement.on_conflict_do_update(
        index_elements=["id"],
        set_={
            "sample_count": excluded.sample_count,
            "direct_conflict_count": excluded.direct_conflict_count,
            "active": False,
            "generated_at": excluded.generated_at,
        },
    )
    with Session(engine) as session, session.begin():
        session.query(ConflictHourlyRollup).filter(ConflictHourlyRollup.run_id == run_id).delete(
            synchronize_session=False
        )
        for start in range(0, len(rows_to_publish), AGGREGATE_CANDIDATE_CHUNK_SIZE):
            session.connection().execute(
                upsert, rows_to_publish[start : start + AGGREGATE_CANDIDATE_CHUNK_SIZE]
            )
    return len(rows_to_publish), scored_count, direct_count


def build_conflict_aggregates(
    layout: ArchiveLayout,
    engine: Engine,
    *,
    scorer_factory: ConflictScorerFactory | None = None,
    batch_size: int = DEFAULT_AGGREGATE_BATCH_SIZE,
    progress: ProgressCallback | None = None,
    requested_job_id: str | None = None,
) -> ConflictAggregateSummary:
    """Build and atomically publish gated, resumable, time-only conflict aggregates."""

    if not 1 <= batch_size <= min(MAX_AGGREGATE_BATCH_SIZE, DEFAULT_CONFLICT_SPEC.max_candidates):
        raise ValueError(
            f"Conflict aggregate batch size must be between 1 and {MAX_AGGREGATE_BATCH_SIZE}"
        )
    lock_path = layout.catalog / "conflict-aggregates.lock"
    with ExitStack() as archive_locks:
        archive_locks.enter_context(deletion_lock(layout))
        layout.ensure_runtime_directories()
        lock_handle = archive_locks.enter_context(lock_path.open("a+b"))
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another conflict aggregate job is already running") from exc
        assert_derived_publication_allowed(engine, publisher="Conflict aggregate publication")

        context = _context(engine, require_gate=True)
        configuration, configuration_sha256 = _configuration(context)
        run_id = stable_id("conflict-aggregate-run", configuration_sha256)
        job_id, resumed = _prepare_job(
            engine,
            requested_job_id=requested_job_id,
            run_id=run_id,
            configuration_sha256=configuration_sha256,
            total=context.window_count,
        )
        ready_snapshot: tuple[int, int, int] | None = None
        with Session(engine) as session:
            existing = session.get(ConflictAggregateRun, run_id)
            if existing is not None and existing.active and existing.status == "ready":
                rollup_count = (
                    session.scalar(
                        select(func.count(ConflictHourlyRollup.id)).where(
                            ConflictHourlyRollup.run_id == run_id
                        )
                    )
                    or 0
                )
                ready_snapshot = (
                    existing.window_count,
                    existing.scored_count,
                    existing.direct_conflict_count,
                )
        if ready_snapshot is not None:
            window_count, scored_count, direct_conflict_count = ready_snapshot
            with Session(engine) as session, session.begin():
                job = session.get(Job, job_id)
                if job is not None:
                    job.status = "completed"
                    job.stage = "Current conflict aggregates already available"
                    job.progress_current = scored_count
                    job.progress_total = window_count
                    job.counters_json = {"reused": scored_count, "scored": 0}
                    job.finished_at = datetime.now(UTC)
            return ConflictAggregateSummary(
                job_id=job_id,
                run_id=run_id,
                status="completed",
                window_count=window_count,
                scored=0,
                reused=scored_count,
                direct_conflict_count=direct_conflict_count,
                rollup_count=int(rollup_count),
                resumed=True,
                configuration_sha256=configuration_sha256,
            )

        now = datetime.now(UTC)
        with Session(engine) as session, session.begin():
            run = session.get(ConflictAggregateRun, run_id)
            values = {
                "active": False,
                "status": "building",
                "pipeline_version": CONFLICT_AGGREGATE_PIPELINE_VERSION,
                "model_name": DEFAULT_CONFLICT_SPEC.model_name,
                "model_revision": DEFAULT_CONFLICT_SPEC.model_revision,
                "calibration_id": context.calibration_id,
                "semantic_config_id": context.semantic_config_id,
                "corpus_revision": context.corpus_revision,
                "source_revision": context.source_revision,
                "configuration_sha256": configuration_sha256,
                "configuration_json": configuration,
                "window_count": context.window_count,
                "job_id": job_id,
                "provenance_json": {
                    "semantic_corpus_revision": context.semantic_corpus_revision,
                    "bounded_cpu_batches": True,
                    "batch_size": batch_size,
                    "candidate_chunk_size": AGGREGATE_CANDIDATE_CHUNK_SIZE,
                    "ranking_dimensions": [],
                },
                "warnings_json": {"items": list(AGGREGATE_WARNINGS)},
                "error_text": None,
                "started_at": now,
                "completed_at": None,
            }
            if run is None:
                run = ConflictAggregateRun(
                    id=run_id,
                    scored_count=0,
                    direct_conflict_count=0,
                    **values,
                )
                session.add(run)
            else:
                for key, value in values.items():
                    setattr(run, key, value)

        scored = reused = processed = 0
        try:
            scorer = (
                scorer_factory(DEFAULT_CONFLICT_SPEC)
                if scorer_factory is not None
                else ModernBertNliConflictScorer(
                    layout.models / "conflict" / DEFAULT_CONFLICT_SPEC.model_revision,
                    DEFAULT_CONFLICT_SPEC,
                )
            )
            for chunk in _iter_candidate_chunks(engine, config_id=context.semantic_config_id):
                input_hashes = {
                    window_id: _score_input_sha256(text) for window_id, _ts, text in chunk
                }
                with Session(engine) as session:
                    existing_rows = {
                        row.window_id: row
                        for row in session.scalars(
                            select(ConflictWindowScore).where(
                                ConflictWindowScore.run_id == run_id,
                                ConflictWindowScore.window_id.in_([item[0] for item in chunk]),
                            )
                        )
                    }
                pending: list[tuple[str, int | None, str]] = []
                for item in chunk:
                    existing_score = existing_rows.get(item[0])
                    if (
                        existing_score is not None
                        and existing_score.input_sha256 == input_hashes[item[0]]
                        and set((existing_score.scores_json or {}).keys()) == set(CONFLICT_LABELS)
                    ):
                        reused += 1
                        processed += 1
                    else:
                        pending.append(item)
                for start in range(0, len(pending), batch_size):
                    batch = pending[start : start + batch_size]
                    vectors = scorer.score([text for _window_id, _timestamp, text in batch])
                    if len(vectors) != len(batch):
                        raise RuntimeError("Conflict scorer returned an incomplete aggregate batch")
                    batch_rows: list[dict[str, Any]] = []
                    for (window_id, timestamp_ms, _text), vector in zip(
                        batch, vectors, strict=True
                    ):
                        scores = dict(vector.scores)
                        if set(scores) != set(CONFLICT_LABELS) or any(
                            not isinstance(value, (int, float)) or not math.isfinite(float(value))
                            for value in scores.values()
                        ):
                            raise RuntimeError(
                                "Conflict scorer returned an invalid aggregate vector"
                            )
                        predicted = vector.predicted_label
                        batch_rows.append(
                            {
                                "id": stable_id("conflict-window-score", run_id, window_id),
                                "run_id": run_id,
                                "window_id": window_id,
                                "input_sha256": input_hashes[window_id],
                                "first_timestamp_ms": timestamp_ms,
                                "predicted_label": predicted,
                                "direct_conflict_score": float(scores[DIRECT_CONFLICT_LABEL]),
                                "scores_json": {
                                    label: float(scores[label]) for label in CONFLICT_LABELS
                                },
                                "active": False,
                                "provenance_json": {
                                    "pipeline_version": CONFLICT_AGGREGATE_PIPELINE_VERSION,
                                    "model_name": DEFAULT_CONFLICT_SPEC.model_name,
                                    "model_revision": DEFAULT_CONFLICT_SPEC.model_revision,
                                    "score_kind": DEFAULT_CONFLICT_SPEC.score_kind,
                                    "calibration_id": context.calibration_id,
                                    "model_output_is_ranking_feature": True,
                                },
                            }
                        )
                    # Commit each bounded scorer batch. A process interruption can therefore
                    # resume from immutable input hashes without repeating completed CPU work.
                    _upsert_score_rows(engine, batch_rows)
                    scored += len(batch)
                    processed += len(batch)
                last_window_id = chunk[-1][0]
                with Session(engine) as session, session.begin():
                    run = session.get(ConflictAggregateRun, run_id)
                    job = session.get(Job, job_id)
                    persisted_count = (
                        session.scalar(
                            select(func.count(ConflictWindowScore.id)).where(
                                ConflictWindowScore.run_id == run_id
                            )
                        )
                        or 0
                    )
                    if run is not None:
                        run.scored_count = int(persisted_count)
                    if job is not None:
                        job.progress_current = processed
                        job.checkpoint_json = {
                            "run_id": run_id,
                            "configuration_sha256": configuration_sha256,
                            "last_window_id": last_window_id,
                        }
                        job.counters_json = {"scored": scored, "reused": reused}
                if progress:
                    progress("Scored full-corpus conflict windows", processed, context.window_count)

            current = _context(engine, require_gate=True)
            if current != context:
                raise RuntimeError(
                    "Conflict corpus, source, semantic index, or calibration changed during scoring"
                )
            rollup_count, persisted_count, direct_count = _publish_rollups(engine, run_id=run_id)
            if persisted_count != context.window_count:
                raise RuntimeError("Conflict aggregate output is incomplete and was not published")
            completed_at = datetime.now(UTC)
            with Session(engine) as session, session.begin():
                session.execute(update(ConflictAggregateRun).values(active=False))
                session.execute(update(ConflictWindowScore).values(active=False))
                session.execute(update(ConflictHourlyRollup).values(active=False))
                session.execute(
                    update(ConflictAggregateRun)
                    .where(ConflictAggregateRun.id == run_id)
                    .values(
                        active=True,
                        status="ready",
                        scored_count=persisted_count,
                        direct_conflict_count=direct_count,
                        error_text=None,
                        completed_at=completed_at,
                    )
                )
                session.execute(
                    update(ConflictWindowScore)
                    .where(ConflictWindowScore.run_id == run_id)
                    .values(active=True)
                )
                session.execute(
                    update(ConflictHourlyRollup)
                    .where(ConflictHourlyRollup.run_id == run_id)
                    .values(active=True)
                )
                session.execute(
                    update(Job)
                    .where(Job.id == job_id)
                    .values(
                        status="completed",
                        stage="Gated conflict aggregates ready",
                        progress_current=persisted_count,
                        progress_total=context.window_count,
                        counters_json={
                            "scored": scored,
                            "reused": reused,
                            "direct_conflict_count": direct_count,
                            "rollup_count": rollup_count,
                        },
                        error_text=None,
                        finished_at=completed_at,
                    )
                )
            return ConflictAggregateSummary(
                job_id=job_id,
                run_id=run_id,
                status="completed",
                window_count=context.window_count,
                scored=scored,
                reused=reused,
                direct_conflict_count=direct_count,
                rollup_count=rollup_count,
                resumed=resumed,
                configuration_sha256=configuration_sha256,
            )
        except BaseException as exc:
            with Session(engine) as session, session.begin():
                session.execute(
                    update(ConflictAggregateRun)
                    .where(ConflictAggregateRun.id == run_id)
                    .values(
                        active=False,
                        status="failed",
                        error_text=f"{type(exc).__name__}: {exc}"[:MAX_ERROR_LENGTH],
                    )
                )
                session.execute(
                    update(Job)
                    .where(Job.id == job_id)
                    .values(
                        status="failed",
                        stage="Conflict aggregate job interrupted",
                        error_text=f"{type(exc).__name__}: {exc}"[:MAX_ERROR_LENGTH],
                        finished_at=datetime.now(UTC),
                    )
                )
            raise
