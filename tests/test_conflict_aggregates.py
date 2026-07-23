from __future__ import annotations

import shutil
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from meta_archive.api.main import create_app
from meta_archive.cli import app as cli_app
from meta_archive.models import (
    ConflictAggregateRun,
    ConflictCalibration,
    ConflictHourlyRollup,
    ConflictLabel,
    ConflictWindowScore,
    Job,
    Message,
    SearchWindow,
    SearchWindowMessage,
    SemanticIndexConfig,
)
from meta_archive.paths import ArchiveLayout
from meta_archive.services.conflict_aggregates import (
    build_conflict_aggregates,
    conflict_aggregate_status,
)
from meta_archive.services.conflict_scoring import (
    CONFLICT_LABELS,
    ConflictModelSpec,
    ConflictScoreVector,
)
from meta_archive.services.inventory import scan_archive
from meta_archive.services.message_importer import import_messages
from meta_archive.services.nlp_annotations import (
    CONFLICT_MODEL_NAME,
    CONFLICT_MODEL_REVISION,
    CONFLICT_PIPELINE_VERSION,
)

FIXTURES = Path(__file__).parent / "fixtures"


class SyntheticConflictScorer:
    def __init__(self, spec: ConflictModelSpec, *, fail_on_call: int | None = None) -> None:
        self.spec = spec
        self.fail_on_call = fail_on_call
        self.calls: list[list[str]] = []

    @property
    def status(self) -> dict[str, Any]:
        return {"status": "ready", "fixture": True}

    def score(self, documents: Sequence[str]) -> list[ConflictScoreVector]:
        self.calls.append(list(documents))
        if self.fail_on_call == len(self.calls):
            raise RuntimeError("synthetic interruption")
        return [
            ConflictScoreVector(
                scores={
                    label: 10.0 if label == "direct conflict" else float(-index)
                    for index, label in enumerate(CONFLICT_LABELS, start=1)
                }
            )
            for _document in documents
        ]


def _prepare_gated_corpus(
    layout: ArchiveLayout,
    engine: Engine,
    *,
    target_count: int = 3,
) -> list[str]:
    shutil.copytree(FIXTURES / "messages", layout.root, dirs_exist_ok=True)
    scan_archive(layout, engine, hash_files=True)
    import_messages(layout, engine)
    config_id = "fixture-semantic-config"
    with Session(engine) as session, session.begin():
        messages = list(
            session.scalars(
                select(Message)
                .where(Message.deleted_at.is_(None))
                .order_by(Message.timestamp_ms, Message.id)
                .limit(target_count)
            )
        )
        assert len(messages) == target_count
        session.add(
            SemanticIndexConfig(
                id=config_id,
                active=True,
                status="ready",
                backend="fixture",
                backend_version="fixture-v1",
                model_name="fixture-embedding-model",
                model_revision="fixture-embedding-revision",
                dimensions=3,
                distance_metric="cosine",
                vector_table="fixture_vectors",
                window_version="fixture-window-v1",
                window_count=target_count,
                audit_json={"corpus_revision": "fixture-semantic-corpus"},
                completed_at=datetime.now(UTC),
            )
        )
        target_window_ids: list[str] = []
        for ordinal, message in enumerate(messages):
            window_id = f"aggregate-target-{ordinal}"
            target_window_ids.append(window_id)
            session.add(
                SearchWindow(
                    id=window_id,
                    window_version="fixture-window-v1",
                    entity_type="message_window",
                    conversation_id=message.conversation_id,
                    segment_id=message.segment_id,
                    first_message_id=message.id,
                    last_message_id=message.id,
                    first_timestamp_ms=message.timestamp_ms,
                    last_timestamp_ms=message.timestamp_ms,
                    turn_count=1,
                    approx_token_count=5,
                    text_display=message.content_display or f"fixture conflict text {ordinal}",
                    text_search=message.content_search or f"fixture conflict text {ordinal}",
                    participants_search="",
                    source_ids_json={"message_ids": [message.id]},
                    build_generation="fixture-generation",
                    embedding_config_hash=config_id,
                )
            )
            session.flush()
            session.add(
                SearchWindowMessage(
                    window_id=window_id,
                    message_id=message.id,
                    ordinal=0,
                )
            )
        calibration_labels: list[ConflictLabel] = []
        for index in range(500):
            window_id = f"calibration-window-{index:04d}"
            session.add(
                SearchWindow(
                    id=window_id,
                    window_version="fixture-calibration-v1",
                    entity_type="message_window",
                    turn_count=1,
                    approx_token_count=3,
                    text_display="held-out calibration fixture",
                    text_search="held out calibration fixture",
                    participants_search="",
                    source_ids_json={},
                    build_generation="fixture-calibration-generation",
                )
            )
            calibration_labels.append(
                ConflictLabel(
                    id=f"calibration-label-{index:04d}",
                    window_id=window_id,
                    label=CONFLICT_LABELS[index % len(CONFLICT_LABELS)],
                    evaluation_split=("train", "validation", "test")[index % 3],
                )
            )
        session.flush()
        session.add_all(calibration_labels)
        session.add(
            ConflictCalibration(
                id="fixture-passing-calibration",
                pipeline_version=CONFLICT_PIPELINE_VERSION,
                model_name=CONFLICT_MODEL_NAME,
                model_revision=CONFLICT_MODEL_REVISION,
                status="calibrated",
                sample_count=500,
                direct_conflict_precision=0.9,
                macro_f1=0.8,
                aggregation_enabled=True,
                metrics_json={
                    "evaluation_source": "held_out_test_predictions",
                    "conversation_disjoint": True,
                    "stratified": True,
                },
                provenance_json={"evaluation_splits": {"train": 1, "validation": 1, "test": 1}},
                warnings_json={"items": []},
                completed_at=datetime.now(UTC),
            )
        )
    return target_window_ids


def test_gate_refuses_job_before_scorer_is_created(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> None:
    factory_calls = 0

    def factory(spec: ConflictModelSpec) -> SyntheticConflictScorer:
        nonlocal factory_calls
        factory_calls += 1
        return SyntheticConflictScorer(spec)

    with pytest.raises(RuntimeError, match="calibration"):
        build_conflict_aggregates(
            archive_layout,
            catalog_engine,
            scorer_factory=factory,
        )
    assert factory_calls == 0
    with Session(catalog_engine) as session:
        assert (
            session.scalar(select(func.count(Job.id)).where(Job.kind == "conflict_aggregates")) == 0
        )
        assert session.scalar(select(func.count(ConflictAggregateRun.id))) == 0

    catalog_engine.dispose()
    cli_result = CliRunner().invoke(
        cli_app,
        ["build-conflict-aggregates", "--archive-root", str(archive_layout.root)],
    )
    assert cli_result.exit_code == 2
    assert "calibration" in cli_result.output.casefold()

    app = create_app(archive_layout.root, security_disabled=True)
    app.state.conflict_scorer_factory = factory
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/enrichment/conflict/aggregate-jobs",
            json={"batch_size": 8},
        )
        analytics = client.get("/api/v1/analytics/conflict-rate")
    assert response.status_code == 409
    assert analytics.status_code == 200
    assert analytics.json()["available"] is False
    assert analytics.json()["series"] == []
    assert factory_calls == 0
    with Session(app.state.engine) as session:
        assert (
            session.scalar(select(func.count(Job.id)).where(Job.kind == "conflict_aggregates")) == 0
        )


def test_full_corpus_job_resumes_publishes_and_exposes_time_only_analytics(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> None:
    target_window_ids = _prepare_gated_corpus(archive_layout, catalog_engine)
    with Session(catalog_engine) as session, session.begin():
        for index, window_id in enumerate(target_window_ids):
            window = session.get(SearchWindow, window_id)
            assert window is not None
            window.first_timestamp_ms = 1_672_531_200_000 + index * 86_400_000
            window.last_timestamp_ms = window.first_timestamp_ms
    interrupted = SyntheticConflictScorer(
        ConflictModelSpec(),
        fail_on_call=2,
    )
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        build_conflict_aggregates(
            archive_layout,
            catalog_engine,
            scorer_factory=lambda _spec: interrupted,
            batch_size=1,
        )
    with Session(catalog_engine) as session:
        assert session.scalar(select(func.count(ConflictWindowScore.id))) == 1
        assert (
            session.scalar(
                select(func.count(ConflictWindowScore.id)).where(
                    ConflictWindowScore.active.is_(True)
                )
            )
            == 0
        )
        failed_run = session.scalar(select(ConflictAggregateRun))
        assert failed_run is not None and failed_run.status == "failed"

    resumed_scorer = SyntheticConflictScorer(ConflictModelSpec())
    summary = build_conflict_aggregates(
        archive_layout,
        catalog_engine,
        scorer_factory=lambda _spec: resumed_scorer,
        batch_size=1,
    )
    assert summary.window_count == len(target_window_ids)
    assert summary.reused == 1
    assert summary.scored == len(target_window_ids) - 1
    assert summary.resumed is True
    assert sum(len(batch) for batch in resumed_scorer.calls) == len(target_window_ids) - 1
    with Session(catalog_engine) as session:
        run = session.get(ConflictAggregateRun, summary.run_id)
        assert run is not None and run.active and run.status == "ready"
        assert run.calibration_id == "fixture-passing-calibration"
        assert run.semantic_config_id == "fixture-semantic-config"
        assert run.corpus_revision and run.source_revision and run.configuration_sha256
        scores = list(
            session.scalars(select(ConflictWindowScore).where(ConflictWindowScore.active.is_(True)))
        )
        assert {row.window_id for row in scores} == set(target_window_ids)
        assert all(row.predicted_label == "direct conflict" for row in scores)
        assert session.scalar(
            select(func.sum(ConflictHourlyRollup.sample_count)).where(
                ConflictHourlyRollup.active.is_(True)
            )
        ) == len(target_window_ids)

    no_op_scorer = SyntheticConflictScorer(ConflictModelSpec())
    no_op = build_conflict_aggregates(
        archive_layout,
        catalog_engine,
        scorer_factory=lambda _spec: no_op_scorer,
        batch_size=1,
    )
    assert no_op.reused == len(target_window_ids)
    assert no_op.scored == 0
    assert no_op_scorer.calls == []

    app = create_app(archive_layout.root, security_disabled=True)
    with TestClient(app) as client:
        response = client.get(
            "/api/v1/analytics/conflict-rate",
            params={"bucket": "day", "timezone": "UTC"},
        )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["available"] is True
    assert payload["summary"]["sample_count"] == len(target_window_ids)
    assert payload["summary"]["direct_conflict_window_rate"] == pytest.approx(1.0)
    assert payload["series"]
    assert all(item["sample_count"] > 0 for item in payload["series"])
    assert payload["relationship_ranking_enabled"] is False
    assert payload["relationship_dimensions"] == []
    assert not any(
        forbidden in item
        for item in payload["series"]
        for forbidden in ("person", "participant", "conversation", "relationship")
    )
    with TestClient(app) as client:
        first_page = client.get(
            "/api/v1/analytics/conflict-rate",
            params={"bucket": "day", "timezone": "UTC", "limit": 1},
        )
        assert first_page.status_code == 200, first_page.text
        first_payload = first_page.json()
        cursor = first_payload["page"]["next_cursor"]
        assert first_payload["page"]["total"] is None
        assert first_payload["page"]["count_is_unknown"] is True
        assert first_payload["page"]["returned"] == len(first_payload["series"]) == 1
        assert cursor
        second_page = client.get(
            "/api/v1/analytics/conflict-rate",
            params={"bucket": "day", "timezone": "UTC", "limit": 1, "cursor": cursor},
        )
        assert second_page.status_code == 200, second_page.text
        second_payload = second_page.json()
        assert second_payload["summary"] == first_payload["summary"]
        assert second_payload["series"][0]["period"] != first_payload["series"][0]["period"]
        assert (
            client.get(
                "/api/v1/analytics/conflict-rate",
                params={
                    "bucket": "day",
                    "timezone": "America/New_York",
                    "limit": 1,
                    "cursor": cursor,
                },
            ).status_code
            == 422
        )
        assert (
            client.get(
                "/api/v1/analytics/conflict-rate",
                params={"bucket": "day", "timezone": "UTC", "limit": 2, "cursor": cursor},
            ).status_code
            == 422
        )
    with Session(catalog_engine) as session:
        first_rollup = session.scalar(
            select(ConflictHourlyRollup)
            .where(ConflictHourlyRollup.active.is_(True))
            .order_by(ConflictHourlyRollup.hour_start_ms)
        )
        assert first_rollup is not None
        first_hour_ms = first_rollup.hour_start_ms
        expected_partial_samples = first_rollup.sample_count
    with TestClient(app) as client:
        partial = client.get(
            "/api/v1/analytics/conflict-rate",
            params={
                "bucket": "hour",
                "timezone": "Asia/Kathmandu",
                "from_ms": first_hour_ms + 1,
                "to_ms": first_hour_ms + 2,
            },
        ).json()
    assert partial["summary"]["sample_count"] == expected_partial_samples
    assert partial["provenance"]["partial_boundary_hours_included"] is True
    assert partial["provenance"]["non_whole_hour_timezone_offset_observed"] is True
    assert any("partial boundary hours" in warning for warning in partial["warnings"])


def test_source_change_persistently_stales_published_aggregate(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> None:
    _prepare_gated_corpus(archive_layout, catalog_engine, target_count=2)
    scorer = SyntheticConflictScorer(ConflictModelSpec())
    build_conflict_aggregates(
        archive_layout,
        catalog_engine,
        scorer_factory=lambda _spec: scorer,
        batch_size=2,
    )
    with Session(catalog_engine) as session, session.begin():
        message = session.scalar(select(Message).order_by(Message.pk).limit(1))
        assert message is not None
        message.deleted_at = datetime.now(UTC)

    status = conflict_aggregate_status(catalog_engine)
    assert status["available"] is False
    assert status["status"] == "stale"
    with Session(catalog_engine) as session:
        run = session.scalar(select(ConflictAggregateRun))
        assert run is not None and run.status == "stale" and not run.active
        assert (
            session.scalar(
                select(func.count(ConflictWindowScore.id)).where(
                    ConflictWindowScore.active.is_(True)
                )
            )
            == 0
        )
        assert (
            session.scalar(
                select(func.count(ConflictHourlyRollup.id)).where(
                    ConflictHourlyRollup.active.is_(True)
                )
            )
            == 0
        )
