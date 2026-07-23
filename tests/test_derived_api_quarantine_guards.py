from __future__ import annotations

from collections.abc import Callable

import pytest
from fastapi import BackgroundTasks
from fastapi.testclient import TestClient
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

from meta_archive.api.analytics_routes import _run_topic_cluster_job
from meta_archive.api.enrichment_routes import (
    _run_attachment_text_job,
    _run_audio_transcription_job,
    _run_conflict_aggregate_job,
    _run_nlp_annotation_job,
)
from meta_archive.api.main import create_app
from meta_archive.api.search_routes import _run_semantic_index_job
from meta_archive.models import (
    ConflictAggregateRun,
    ConflictCalibration,
    ConflictLabel,
    Conversation,
    DeletionOperation,
    Job,
    RagAudit,
    RelevanceFeedback,
    SearchWindow,
    SemanticIndexConfig,
)
from meta_archive.paths import ArchiveLayout
from meta_archive.services.deletion_lock_lease import deletion_lock


def _add_deletion(engine: Engine, *, status: str) -> str:
    operation_id = f"api-guard-{status}"
    with Session(engine) as session, session.begin():
        session.add(
            DeletionOperation(
                id=operation_id,
                status=status,
                selector_json={"entity_types": ["message"]},
                impact_json={},
                catalog_revision="a" * 64,
                selection_sha256="b" * 64,
                purge_plan_json={},
                item_count=0,
            )
        )
    return operation_id


BLOCKED_POSTS: tuple[tuple[str, dict[str, object]], ...] = (
    ("/api/v1/search/index", {}),
    ("/api/v1/enrichment/transcriptions/jobs", {}),
    ("/api/v1/enrichment/attachment-text/jobs", {}),
    ("/api/v1/enrichment/nlp/jobs", {}),
    ("/api/v1/enrichment/conflict/aggregate-jobs", {}),
    ("/api/v1/enrichment/conflict/calibrations", {}),
    (
        "/api/v1/enrichment/conflict/labels",
        {"window_id": "blocked-window", "label": "direct conflict"},
    ),
    ("/api/v1/analytics/topics/build", {}),
    (
        "/api/v1/search/feedback",
        {
            "query": "fixture",
            "search_mode": "semantic",
            "result_id": "window:blocked",
            "label": "relevant",
        },
    ),
    ("/api/v1/rag/answer", {"question": "fixture question"}),
)


@pytest.mark.parametrize(
    "deletion_status",
    ["committed", "cleanup_pending", "normalized_purged"],
)
def test_derived_api_launches_return_stable_block_without_catalog_mutation(
    archive_layout: ArchiveLayout,
    catalog_engine: Engine,
    deletion_status: str,
) -> None:
    app = create_app(archive_layout.root, security_disabled=True)
    _add_deletion(catalog_engine, status=deletion_status)

    with TestClient(app) as client:
        responses = [client.post(path, json=payload) for path, payload in BLOCKED_POSTS]
        responses.extend(
            [
                client.patch(
                    "/api/v1/search/feedback/blocked-feedback",
                    json={"label": "relevant"},
                ),
                client.delete("/api/v1/search/feedback/blocked-feedback"),
            ]
        )

    assert [response.status_code for response in responses] == [409] * (len(BLOCKED_POSTS) + 2)
    assert [response.json()["detail"] for response in responses] == [
        {
            "code": "derived_publication_blocked",
            "message": (
                "Derived-data changes are unavailable while a deletion quarantine or purge "
                "is outstanding."
            ),
            "deletion_status": deletion_status,
        }
    ] * (len(BLOCKED_POSTS) + 2)
    with Session(catalog_engine) as session:
        assert session.scalar(select(func.count(Job.id))) == 0
        assert session.scalar(select(func.count(ConflictLabel.id))) == 0
        assert session.scalar(select(func.count(RelevanceFeedback.id))) == 0
        assert session.scalar(select(func.count(RagAudit.id))) == 0
        operation = session.get(DeletionOperation, f"api-guard-{deletion_status}")
        assert operation is not None
        assert operation.status == deletion_status


def test_derived_api_lock_contention_uses_stable_nonleaking_busy_response(
    archive_layout: ArchiveLayout,
) -> None:
    app = create_app(archive_layout.root, security_disabled=True)
    with deletion_lock(archive_layout), TestClient(app) as client:
        response = client.post("/api/v1/search/index", json={})

    assert response.status_code == 409
    assert response.json()["detail"] == {
        "code": "derived_publication_busy",
        "message": (
            "Derived-data changes are temporarily unavailable while deletion state is changing."
        ),
    }


def test_conflict_label_and_status_refresh_preserve_quarantined_canaries(
    archive_layout: ArchiveLayout,
    catalog_engine: Engine,
) -> None:
    conversation_id = "guarded-conflict-conversation"
    window_id = "guarded-conflict-window"
    calibration_id = "guarded-conflict-calibration"
    run_id = "guarded-conflict-run"
    with Session(catalog_engine) as session, session.begin():
        session.add(
            Conversation(
                id=conversation_id,
                conversation_key=conversation_id,
                title_raw="Guard fixture",
                title_display="Guard fixture",
                title_search="guard fixture",
                conversation_type="direct",
                primary_category="fixture",
            )
        )
        session.add(
            SearchWindow(
                id=window_id,
                window_version="fixture-v1",
                entity_type="message_window",
                conversation_id=conversation_id,
                turn_count=1,
                approx_token_count=3,
                text_display="Original conflict label canary",
                text_search="original conflict label canary",
                participants_search="",
                source_ids_json={},
                build_generation="fixture-generation",
            )
        )
        session.flush()
        session.add(
            ConflictLabel(
                id="guarded-label",
                window_id=window_id,
                label="banter",
                evaluation_split="train",
                notes="ORIGINAL_LABEL_CANARY",
            )
        )
        session.add(
            ConflictCalibration(
                id=calibration_id,
                pipeline_version="fixture-v1",
                model_name="fixture-model",
                model_revision="fixture-revision",
                status="calibrated",
                sample_count=1,
                aggregation_enabled=True,
                metrics_json={},
                provenance_json={},
                warnings_json={},
            )
        )
        session.flush()
        session.add(
            ConflictAggregateRun(
                id=run_id,
                active=True,
                status="ready",
                pipeline_version="fixture-v1",
                model_name="deliberately-stale-model",
                model_revision="deliberately-stale-revision",
                calibration_id=calibration_id,
                corpus_revision="c" * 64,
                source_revision="d" * 64,
                configuration_sha256="e" * 64,
                configuration_json={},
                provenance_json={"canary": "ORIGINAL_AGGREGATE_CANARY"},
                warnings_json={},
            )
        )
    _add_deletion(catalog_engine, status="committed")
    app = create_app(archive_layout.root, security_disabled=True)

    with TestClient(app) as client:
        labelled = client.post(
            "/api/v1/enrichment/conflict/labels",
            json={
                "window_id": window_id,
                "label": "direct conflict",
                "notes": "MUTATED_LABEL_CANARY",
            },
        )
        enrichment_status = client.get("/api/v1/enrichment/conflict/status")
        analytics_status = client.get("/api/v1/analytics/conflict-rate")

    assert labelled.status_code == 409
    assert labelled.json()["detail"]["code"] == "derived_publication_blocked"
    assert enrichment_status.status_code == 200
    assert analytics_status.status_code == 200
    with Session(catalog_engine) as session:
        label = session.get(ConflictLabel, "guarded-label")
        calibration = session.get(ConflictCalibration, calibration_id)
        run = session.get(ConflictAggregateRun, run_id)
        assert label is not None
        assert (label.label, label.notes) == ("banter", "ORIGINAL_LABEL_CANARY")
        assert calibration is not None
        assert (calibration.status, calibration.aggregation_enabled) == ("calibrated", True)
        assert run is not None
        assert (run.status, run.active, run.error_text) == ("ready", True, None)
        assert run.provenance_json == {"canary": "ORIGINAL_AGGREGATE_CANARY"}


BackgroundRunner = Callable[[ArchiveLayout, Engine, str], None]


def _audio_runner(layout: ArchiveLayout, engine: Engine, job_id: str) -> None:
    _run_audio_transcription_job(
        layout=layout,
        engine=engine,
        job_id=job_id,
        retry_failed=True,
        adapter=None,
    )


def _attachment_runner(layout: ArchiveLayout, engine: Engine, job_id: str) -> None:
    _run_attachment_text_job(
        layout=layout,
        engine=engine,
        job_id=job_id,
        retry_failed=True,
        extractor=None,
    )


def _nlp_runner(layout: ArchiveLayout, engine: Engine, job_id: str) -> None:
    _run_nlp_annotation_job(
        layout=layout,
        engine=engine,
        job_id=job_id,
        tasks=("sentiment",),
        batch_size=1,
        classifier=None,
    )


def _conflict_runner(layout: ArchiveLayout, engine: Engine, job_id: str) -> None:
    _run_conflict_aggregate_job(
        layout=layout,
        engine=engine,
        job_id=job_id,
        batch_size=1,
        scorer_factory=lambda _spec: None,
    )


def _semantic_runner(layout: ArchiveLayout, engine: Engine, job_id: str) -> None:
    _run_semantic_index_job(
        layout=layout,
        engine=engine,
        job_id=job_id,
        model=None,
        rebuild_windows=True,
        embedder_factory=None,
        reranker_factory=None,
    )


def _topic_runner(layout: ArchiveLayout, engine: Engine, job_id: str) -> None:
    _run_topic_cluster_job(
        layout=layout,
        engine=engine,
        job_id=job_id,
        cluster_count=2,
        sample_limit=100,
    )


@pytest.mark.parametrize(
    ("kind", "runner"),
    [
        ("audio_transcription", _audio_runner),
        ("attachment_text", _attachment_runner),
        ("nlp_annotations", _nlp_runner),
        ("conflict_aggregates", _conflict_runner),
        ("semantic_index", _semantic_runner),
        ("topic_clusters", _topic_runner),
    ],
)
@pytest.mark.parametrize("deletion_status", ["committed", "cleanup_pending", "normalized_purged"])
def test_deletion_blocked_background_wrapper_preserves_exact_queued_state(
    archive_layout: ArchiveLayout,
    catalog_engine: Engine,
    kind: str,
    runner: BackgroundRunner,
    deletion_status: str,
) -> None:
    _add_deletion(catalog_engine, status=deletion_status)
    job_id = f"queued-{kind}"
    with Session(catalog_engine) as session, session.begin():
        session.add(
            Job(
                id=job_id,
                kind=kind,
                status="queued",
                stage="ORIGINAL_QUEUED_STAGE",
                progress_current=7,
                progress_total=11,
                checkpoint_json={"canary": "ORIGINAL_CHECKPOINT"},
                counters_json={"canary": "ORIGINAL_COUNTERS"},
            )
        )

    runner(archive_layout, catalog_engine, job_id)

    with Session(catalog_engine) as session:
        job = session.get(Job, job_id)
        assert job is not None
        assert job.status == "queued"
        assert job.stage == "ORIGINAL_QUEUED_STAGE"
        assert (job.progress_current, job.progress_total) == (7, 11)
        assert job.checkpoint_json == {"canary": "ORIGINAL_CHECKPOINT"}
        assert job.counters_json == {"canary": "ORIGINAL_COUNTERS"}
        assert job.error_text is None
        assert job.started_at is None
        assert job.finished_at is None


def test_deletion_lock_contention_preserves_exact_queued_job_state(
    archive_layout: ArchiveLayout,
    catalog_engine: Engine,
) -> None:
    job_id = "queued-during-deletion-lock-contention"
    with Session(catalog_engine) as session, session.begin():
        session.add(
            Job(
                id=job_id,
                kind="audio_transcription",
                status="queued",
                stage="ORIGINAL_QUEUED_STAGE",
                progress_current=0,
                checkpoint_json={"retry_failed": True},
                counters_json={},
            )
        )

    with deletion_lock(archive_layout):
        _audio_runner(archive_layout, catalog_engine, job_id)

    with Session(catalog_engine) as session:
        job = session.get(Job, job_id)
        assert job is not None
        assert (job.status, job.stage, job.error_text) == (
            "queued",
            "ORIGINAL_QUEUED_STAGE",
            None,
        )


class _EmptyTranscriber:
    identity = "empty-api-race-transcriber"
    model_name = "fixture/empty"
    model_revision = "fixture-empty-v1"
    device = "cpu"
    compute_type = "int8"

    def transcribe(self, _source: object) -> None:
        raise AssertionError("An empty archive must not invoke the transcriber")


CapturedTask = tuple[Callable[..., object], tuple[object, ...], dict[str, object]]


def _capture_background_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> list[CapturedTask]:
    captured: list[CapturedTask] = []

    def capture(
        _tasks: BackgroundTasks,
        function: Callable[..., object],
        *args: object,
        **kwargs: object,
    ) -> None:
        captured.append((function, args, kwargs))

    monkeypatch.setattr(BackgroundTasks, "add_task", capture)
    return captured


def test_post_queue_deletion_race_reschedules_same_untouched_job_after_restore(
    archive_layout: ArchiveLayout,
    catalog_engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture_background_tasks(monkeypatch)
    app = create_app(archive_layout.root, security_disabled=True)
    app.state.audio_transcription_adapter = _EmptyTranscriber()
    with TestClient(app) as client:
        queued = client.post("/api/v1/enrichment/transcriptions/jobs", json={})
        assert queued.status_code == 202
        job_id = queued.json()["job_id"]
        assert queued.json()["reused"] is False
        assert len(captured) == 1
        duplicate = client.post("/api/v1/enrichment/transcriptions/jobs", json={})
        assert duplicate.status_code == 409
        assert len(captured) == 1

        _add_deletion(catalog_engine, status="committed")
        function, args, kwargs = captured.pop()
        function(*args, **kwargs)
        with Session(catalog_engine) as session:
            blocked = session.get(Job, job_id)
            assert blocked is not None
            assert (
                blocked.status,
                blocked.stage,
                blocked.checkpoint_json,
                blocked.error_text,
                blocked.started_at,
                blocked.finished_at,
            ) == (
                "queued",
                "Audio transcription queued",
                {"retry_failed": True},
                None,
                None,
                None,
            )

        with Session(catalog_engine) as session, session.begin():
            deletion = session.get(DeletionOperation, "api-guard-committed")
            assert deletion is not None
            deletion.status = "restored"
        retried = client.post("/api/v1/enrichment/transcriptions/jobs", json={})
        assert retried.status_code == 202
        assert retried.json()["job_id"] == job_id
        assert retried.json()["reused"] is True
        assert len(captured) == 1
        function, args, kwargs = captured.pop()
        function(*args, **kwargs)

    with Session(catalog_engine) as session:
        completed = session.get(Job, job_id)
        assert completed is not None
        assert completed.status == "completed"


def test_semantic_and_topic_requests_reschedule_exact_queued_checkpoints(
    archive_layout: ArchiveLayout,
    catalog_engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_id = "queued-topic-semantic-config"
    semantic_job_id = "queued-semantic-race-job"
    topic_job_id = "queued-topic-race-job"
    with Session(catalog_engine) as session, session.begin():
        session.add(
            SemanticIndexConfig(
                id=config_id,
                active=True,
                status="ready",
                backend="fixture",
                backend_version="fixture-v1",
                model_name="fixture-model",
                model_revision="fixture-revision",
                dimensions=3,
                distance_metric="cosine",
                document_prefix="",
                query_prefix="",
                vector_table="fixture-vectors",
                window_version="fixture-v1",
                window_count=0,
                audit_json={},
            )
        )
        session.add_all(
            [
                Job(
                    id=semantic_job_id,
                    kind="semantic_index",
                    status="queued",
                    stage="Semantic index queued",
                    progress_current=0,
                    checkpoint_json={"requested_model": "auto", "rebuild_windows": True},
                    counters_json={},
                ),
                Job(
                    id=topic_job_id,
                    kind="topic_clusters",
                    status="queued",
                    stage="Topic clustering queued",
                    progress_current=0,
                    progress_total=20_000,
                    checkpoint_json={
                        "semantic_config_id": config_id,
                        "sample_limit": 20_000,
                        "requested_clusters": 12,
                    },
                    counters_json={},
                ),
            ]
        )

    captured = _capture_background_tasks(monkeypatch)
    app = create_app(archive_layout.root, security_disabled=True)
    with TestClient(app) as client:
        semantic = client.post("/api/v1/search/index", json={})
        topic = client.post("/api/v1/analytics/topics/build", json={})
        duplicate_semantic = client.post("/api/v1/search/index", json={})
        duplicate_topic = client.post("/api/v1/analytics/topics/build", json={})

    assert semantic.status_code == topic.status_code == 202
    assert duplicate_semantic.status_code == duplicate_topic.status_code == 409
    assert semantic.json()["job_id"] == semantic_job_id
    assert topic.json()["job_id"] == topic_job_id
    assert semantic.json()["reused"] is topic.json()["reused"] is True
    assert len(captured) == 2
    with Session(catalog_engine) as session:
        semantic_job = session.get(Job, semantic_job_id)
        topic_job = session.get(Job, topic_job_id)
        assert semantic_job is not None and semantic_job.status == "queued"
        assert topic_job is not None and topic_job.status == "queued"
