from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import pytest
from sqlalchemy import Engine, func, select, update
from sqlalchemy.orm import Session

from meta_archive.models import (
    ConflictAggregateRun,
    ConflictHourlyRollup,
    ConflictWindowScore,
    Job,
    NlpAnnotation,
    NlpHourlyRollup,
    SearchWindow,
    SemanticIndexConfig,
    SemanticWindowEmbedding,
)
from meta_archive.services.deletion_restore_state import (
    DerivedRestoreStateError,
    capture_derived_restore_state,
    captured_search_window_ids,
    restore_derived_restore_state,
    seal_quarantined_derived_state,
)

NOW = datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC)
ANNOTATION_ID = "restore-annotation"
ROLLUP_ID = "restore-nlp-rollup"
RUN_ID = "restore-conflict-run"
SCORE_ID = "restore-conflict-score"
CONFLICT_ROLLUP_ID = "restore-conflict-rollup"
TOPIC_JOB_ID = "restore-topic-job"
WINDOW_ID = "a" * 64
CONFIG_ID = "b" * 64
SCALE_ROWS = 1_024


def _annotation(
    identifier: str,
    *,
    active: bool,
    pipeline: str = "fixture-v1",
    target_type: str = "message",
    target_id: str = "message-restore-target",
) -> NlpAnnotation:
    return NlpAnnotation(
        id=identifier,
        target_type=target_type,
        target_id=target_id,
        task="sentiment",
        pipeline_version=pipeline,
        model_name="fixture-model",
        model_revision=pipeline,
        input_sha256=("1" if pipeline == "fixture-v1" else "2") * 64,
        status="completed",
        active=active,
        language="en",
        predicted_label="neutral",
        confidence=0.75,
        probabilities_json={"neutral": 0.75},
        attribution_json={"source": "fixture"},
        warnings_json={},
        provenance_json={"pipeline": pipeline},
    )


def _seed_published_state(engine: Engine) -> None:
    with Session(engine) as session, session.begin():
        session.add(
            SemanticIndexConfig(
                id=CONFIG_ID,
                active=True,
                status="ready",
                backend="fixture",
                backend_version="fixture-v1",
                model_name="fixture-embedding",
                model_revision="fixture-revision",
                dimensions=3,
                distance_metric="cosine",
                document_prefix="",
                query_prefix="",
                vector_table="restore_fixture_vectors",
                window_version="fixture-window-v1",
                window_count=1,
                audit_json={"accepted": True},
                completed_at=NOW,
            )
        )
        session.add(
            SearchWindow(
                id=WINDOW_ID,
                window_version="fixture-window-v1",
                entity_type="attachment_text",
                conversation_id=None,
                segment_id=None,
                activity_id=None,
                audio_segment_id=None,
                attachment_extraction_id=None,
                attachment_occurrence_id=None,
                first_message_id=None,
                last_message_id=None,
                first_timestamp_ms=1_700_000_000_000,
                last_timestamp_ms=1_700_000_000_000,
                turn_count=1,
                approx_token_count=3,
                text_display="restore fixture window",
                text_search="restore fixture window",
                source_category="fixture",
                participants_search="",
                source_ids_json={},
                build_generation="restore-fixture-generation",
                embedding_config_hash=CONFIG_ID,
                embedding_updated_at=NOW,
            )
        )
        session.flush()
        session.add(SemanticWindowEmbedding(config_id=CONFIG_ID, window_id=WINDOW_ID))
        session.add(_annotation(ANNOTATION_ID, active=True))
        session.add(
            NlpHourlyRollup(
                id=ROLLUP_ID,
                task="sentiment",
                hour_start_ms=1_700_000_000_000,
                direction="owner",
                pipeline_version="fixture-v1",
                model_name="fixture-model",
                model_revision="fixture-v1",
                configuration_sha256="3" * 64,
                configuration_json={"fixture": True},
                predicted_label="neutral",
                sample_count=1,
                confidence_sum=0.75,
                probability_sums_json={"neutral": 0.75},
                active=True,
                generated_at=NOW,
            )
        )
        session.add(
            ConflictAggregateRun(
                id=RUN_ID,
                active=True,
                status="ready",
                pipeline_version="fixture-conflict-v1",
                model_name="fixture-conflict-model",
                model_revision="fixture-revision",
                calibration_id=None,
                semantic_config_id=CONFIG_ID,
                corpus_revision="4" * 64,
                source_revision="5" * 64,
                configuration_sha256="6" * 64,
                configuration_json={"fixture": True},
                window_count=1,
                scored_count=1,
                direct_conflict_count=1,
                provenance_json={"published": True},
                warnings_json={},
                started_at=NOW,
                completed_at=NOW,
            )
        )
        session.flush()
        session.add(
            ConflictWindowScore(
                id=SCORE_ID,
                run_id=RUN_ID,
                window_id=WINDOW_ID,
                input_sha256="7" * 64,
                first_timestamp_ms=1_700_000_000_000,
                predicted_label="direct_conflict",
                direct_conflict_score=0.8,
                scores_json={"direct_conflict": 0.8},
                active=True,
                provenance_json={"fixture": True},
                created_at=NOW,
            )
        )
        session.add(
            ConflictHourlyRollup(
                id=CONFLICT_ROLLUP_ID,
                run_id=RUN_ID,
                hour_start_ms=1_700_000_000_000,
                sample_count=1,
                direct_conflict_count=1,
                active=True,
                generated_at=NOW,
            )
        )
        session.add(
            Job(
                id=TOPIC_JOB_ID,
                kind="topic_clusters",
                status="completed",
                progress_current=10,
                progress_total=10,
                stage="Topic clusters ready",
                checkpoint_json={"semantic_config_id": CONFIG_ID},
                counters_json={"result": {"terms": ["private fixture term"]}},
                error_text=None,
                started_at=NOW,
                finished_at=NOW,
            )
        )


def _capture_and_quarantine(session: Session) -> dict[str, object]:
    state = capture_derived_restore_state(
        session,
        nlp_targets={"message": {"message-restore-target"}},
        search_window_ids={WINDOW_ID},
    )
    return _quarantine_and_seal(session, state)


def _quarantine_and_seal(session: Session, state: dict[str, object]) -> dict[str, object]:
    annotation = session.get(NlpAnnotation, ANNOTATION_ID)
    run = session.get(ConflictAggregateRun, RUN_ID)
    topic = session.get(Job, TOPIC_JOB_ID)
    assert annotation and run and topic
    annotation.active = False
    session.execute(update(NlpHourlyRollup).values(active=False))
    run.active = False
    run.status = "stale"
    run.error_text = "Archive rows were logically deleted"
    run.provenance_json = {"invalidated": True}
    session.execute(
        update(ConflictWindowScore).where(ConflictWindowScore.run_id == RUN_ID).values(active=False)
    )
    session.execute(
        update(ConflictHourlyRollup)
        .where(ConflictHourlyRollup.run_id == RUN_ID)
        .values(active=False)
    )
    topic.status = "stale"
    topic.stage = "Topic clusters invalidated"
    topic.counters_json = {"invalidated": True}
    topic.error_text = "Archive rows were logically deleted"
    topic.finished_at = NOW
    session.flush()
    return seal_quarantined_derived_state(session, state)


def _sealed_digest(state: dict[str, object]) -> str:
    payload = {
        key: value for key, value in state.items() if key not in {"capture_sha256", "sealed_sha256"}
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def test_exact_derived_state_round_trip_is_json_safe(catalog_engine: Engine) -> None:
    _seed_published_state(catalog_engine)
    with Session(catalog_engine) as session, session.begin():
        state = _capture_and_quarantine(session)
        state = json.loads(json.dumps(state))
        assert state["sealed"] is True
        assert captured_search_window_ids(state) == (WINDOW_ID,)
        affected = restore_derived_restore_state(session, state)
        assert affected == {
            "nlp_annotations_reactivated": 1,
            "nlp_hourly_rollups_reactivated": 1,
            "conflict_aggregate_runs_restored": 1,
            "conflict_window_scores_reactivated": 1,
            "conflict_hourly_rollups_reactivated": 1,
            "topic_cluster_jobs_restored": 1,
        }

    with Session(catalog_engine) as session:
        assert session.get(NlpAnnotation, ANNOTATION_ID).active is True  # type: ignore[union-attr]
        assert session.get(NlpHourlyRollup, ROLLUP_ID).active is True  # type: ignore[union-attr]
        run = session.get(ConflictAggregateRun, RUN_ID)
        topic = session.get(Job, TOPIC_JOB_ID)
        assert run is not None and run.active is True and run.status == "ready"
        assert run.provenance_json == {"published": True}
        assert topic is not None and topic.status == "completed"
        assert topic.counters_json == {"result": {"terms": ["private fixture term"]}}


def test_post_seal_row_drift_fails_before_reactivation(catalog_engine: Engine) -> None:
    _seed_published_state(catalog_engine)
    with Session(catalog_engine) as session, session.begin():
        state = _capture_and_quarantine(session)
        annotation = session.get(NlpAnnotation, ANNOTATION_ID)
        assert annotation is not None
        annotation.confidence = 0.01
        session.flush()
        with pytest.raises(DerivedRestoreStateError, match="NLP annotations changed"):
            restore_derived_restore_state(session, state)
        assert annotation.active is False
        assert session.get(NlpHourlyRollup, ROLLUP_ID).active is False  # type: ignore[union-attr]


def test_competing_active_annotation_fails_closed(catalog_engine: Engine) -> None:
    _seed_published_state(catalog_engine)
    with Session(catalog_engine) as session, session.begin():
        replacement = _annotation("replacement-annotation", active=False, pipeline="fixture-v2")
        session.add(replacement)
        session.flush()
        state = _capture_and_quarantine(session)
        replacement.active = True
        session.flush()
        with pytest.raises(DerivedRestoreStateError, match="replacement NLP annotation"):
            restore_derived_restore_state(session, state)
        assert session.get(NlpAnnotation, ANNOTATION_ID).active is False  # type: ignore[union-attr]


def test_search_membership_drift_fails_closed(catalog_engine: Engine) -> None:
    _seed_published_state(catalog_engine)
    with Session(catalog_engine) as session, session.begin():
        state = _capture_and_quarantine(session)
        membership = session.get(SemanticWindowEmbedding, (CONFIG_ID, WINDOW_ID))
        assert membership is not None
        session.delete(membership)
        session.flush()
        with pytest.raises(DerivedRestoreStateError, match="Search-window links changed"):
            restore_derived_restore_state(session, state)
        assert session.get(NlpAnnotation, ANNOTATION_ID).active is False  # type: ignore[union-attr]


def test_new_global_rollup_fails_closed(catalog_engine: Engine) -> None:
    _seed_published_state(catalog_engine)
    with Session(catalog_engine) as session, session.begin():
        state = _capture_and_quarantine(session)
        replacement = session.get(NlpHourlyRollup, ROLLUP_ID)
        assert replacement is not None
        replacement.active = True
        session.flush()
        with pytest.raises(DerivedRestoreStateError, match="NLP rollup generation changed"):
            restore_derived_restore_state(session, state)


def test_passthrough_annotation_requires_exact_delegation_after_later_owner_retires_it(
    catalog_engine: Engine,
) -> None:
    shared_target = "shared-derived-audio-segment"
    with Session(catalog_engine) as session, session.begin():
        session.add(
            _annotation(
                ANNOTATION_ID,
                active=True,
                target_type="audio_segment",
                target_id=shared_target,
            )
        )
        session.flush()

        operation_a = capture_derived_restore_state(
            session,
            nlp_targets={},
            passthrough_nlp_targets={"audio_segment": {shared_target}},
        )
        assert operation_a["nlp_annotations"][0]["passthrough"] is True
        operation_a = seal_quarantined_derived_state(session, operation_a)

        operation_b = capture_derived_restore_state(
            session,
            nlp_targets={"audio_segment": {shared_target}},
        )
        annotation = session.get(NlpAnnotation, ANNOTATION_ID)
        assert annotation is not None
        annotation.active = False
        session.flush()
        operation_b = seal_quarantined_derived_state(session, operation_b)
        assert operation_b["nlp_annotations"][0]["passthrough"] is False

        with pytest.raises(DerivedRestoreStateError, match="lacks exact delegated provenance"):
            restore_derived_restore_state(session, operation_a)
        assert annotation.active is False

        affected = restore_derived_restore_state(
            session,
            operation_a,
            delegated_annotation_ids={ANNOTATION_ID},
        )
        assert affected["nlp_annotations_reactivated"] == 1
        assert annotation.active is True


def test_passthrough_delegation_rejects_unknown_ids_and_non_active_field_drift(
    catalog_engine: Engine,
) -> None:
    shared_target = "shared-derived-text-extraction"
    with Session(catalog_engine) as session, session.begin():
        session.add(
            _annotation(
                ANNOTATION_ID,
                active=True,
                target_type="attachment_text",
                target_id=shared_target,
            )
        )
        session.flush()
        state = capture_derived_restore_state(
            session,
            nlp_targets={},
            passthrough_nlp_targets={"attachment_text": {shared_target}},
        )
        state = seal_quarantined_derived_state(session, state)
        annotation = session.get(NlpAnnotation, ANNOTATION_ID)
        assert annotation is not None
        annotation.active = False
        session.flush()

        with pytest.raises(DerivedRestoreStateError, match="not captured passthrough"):
            restore_derived_restore_state(
                session,
                state,
                delegated_annotation_ids={"not-captured"},
            )
        annotation.confidence = 0.01
        session.flush()
        with pytest.raises(DerivedRestoreStateError, match="lacks exact delegated provenance"):
            restore_derived_restore_state(
                session,
                state,
                delegated_annotation_ids={ANNOTATION_ID},
            )
        assert annotation.active is False


def test_v1_annotation_entries_without_passthrough_field_remain_restorable(
    catalog_engine: Engine,
) -> None:
    _seed_published_state(catalog_engine)
    with Session(catalog_engine) as session, session.begin():
        state = _capture_and_quarantine(session)
        raw_entries = state["nlp_annotations"]
        assert isinstance(raw_entries, list)
        for entry in raw_entries:
            assert isinstance(entry, dict)
            entry.pop("passthrough")
        state["sealed_sha256"] = _sealed_digest(state)
        affected = restore_derived_restore_state(session, state)
        assert affected["nlp_annotations_reactivated"] == 1
        annotation = session.get(NlpAnnotation, ANNOTATION_ID)
        assert annotation is not None and annotation.active is True


def test_same_annotation_cannot_be_direct_and_passthrough(catalog_engine: Engine) -> None:
    target_id = "overlapping-restore-target"
    with Session(catalog_engine) as session, session.begin():
        session.add(_annotation(ANNOTATION_ID, active=True, target_id=target_id))
        session.flush()
        with pytest.raises(ValueError, match="both quarantined and passthrough"):
            capture_derived_restore_state(
                session,
                nlp_targets={"message": {target_id}},
                passthrough_nlp_targets={"message": {target_id}},
            )


def test_annotation_only_restore_survives_later_global_hard_purge(catalog_engine: Engine) -> None:
    _seed_published_state(catalog_engine)
    with Session(catalog_engine) as session, session.begin():
        state = _capture_and_quarantine(session)
        for model, identifier in (
            (NlpHourlyRollup, ROLLUP_ID),
            (ConflictWindowScore, SCORE_ID),
            (ConflictHourlyRollup, CONFLICT_ROLLUP_ID),
            (ConflictAggregateRun, RUN_ID),
            (Job, TOPIC_JOB_ID),
            (SearchWindow, WINDOW_ID),
        ):
            row = session.get(model, identifier)
            assert row is not None
            session.delete(row)
        session.flush()

        with pytest.raises(DerivedRestoreStateError, match="changed while rows were quarantined"):
            restore_derived_restore_state(session, state)
        annotation = session.get(NlpAnnotation, ANNOTATION_ID)
        assert annotation is not None and annotation.active is False

        config = session.get(SemanticIndexConfig, CONFIG_ID)
        assert config is not None
        config.window_count = 0
        config.audit_json = {"changed_by_later_purge": True}
        session.flush()
        affected = restore_derived_restore_state(
            session,
            state,
            restore_global_derived=False,
        )
        assert affected == {
            "nlp_annotations_reactivated": 1,
            "nlp_hourly_rollups_reactivated": 0,
            "conflict_aggregate_runs_restored": 0,
            "conflict_window_scores_reactivated": 0,
            "conflict_hourly_rollups_reactivated": 0,
            "topic_cluster_jobs_restored": 0,
            "global_derived_rebuild_required": 1,
        }
        assert annotation.active is True
        assert session.get(NlpHourlyRollup, ROLLUP_ID) is None
        assert session.get(ConflictAggregateRun, RUN_ID) is None
        assert session.get(Job, TOPIC_JOB_ID) is None
        assert session.get(SearchWindow, WINDOW_ID) is None


def test_compact_restore_state_rejects_mixed_or_incomplete_encodings(
    catalog_engine: Engine,
) -> None:
    _seed_published_state(catalog_engine)
    with Session(catalog_engine) as session, session.begin():
        state = _capture_and_quarantine(session)
        state["nlp_hourly_rollups"] = [{"id": ROLLUP_ID}]
        state["sealed_sha256"] = _sealed_digest(state)
        with pytest.raises(DerivedRestoreStateError, match="mixes compact and legacy"):
            restore_derived_restore_state(session, state)

        state["nlp_hourly_rollups"] = []
        conflict_runs = state["conflict_runs"]
        assert isinstance(conflict_runs, list) and isinstance(conflict_runs[0], dict)
        conflict_runs[0].pop("rollup_generation")
        state["sealed_sha256"] = _sealed_digest(state)
        with pytest.raises(DerivedRestoreStateError, match="generation state is incomplete"):
            restore_derived_restore_state(session, state)


def test_compact_generation_state_is_bounded_and_restores_thousands_of_rows(
    catalog_engine: Engine,
) -> None:
    _seed_published_state(catalog_engine)
    total_rows = SCALE_ROWS + 1
    base_timestamp_ms = 1_700_000_000_000
    with Session(catalog_engine) as session, session.begin():
        baseline = capture_derived_restore_state(
            session,
            nlp_targets={"message": {"message-restore-target"}},
            search_window_ids={WINDOW_ID},
        )
        baseline_bytes = len(json.dumps(baseline, sort_keys=True))

        window_rows: list[dict[str, object]] = []
        nlp_rows: list[dict[str, object]] = []
        score_rows: list[dict[str, object]] = []
        conflict_rollup_rows: list[dict[str, object]] = []
        for index in range(1, SCALE_ROWS + 1):
            window_id = f"{index:064x}"
            timestamp_ms = base_timestamp_ms + index * 3_600_000
            text = f"scale fixture {index}"
            window_rows.append(
                {
                    "id": window_id,
                    "window_version": "fixture-window-v1",
                    "entity_type": "attachment_text",
                    "conversation_id": None,
                    "segment_id": None,
                    "activity_id": None,
                    "audio_segment_id": None,
                    "attachment_extraction_id": None,
                    "attachment_occurrence_id": None,
                    "first_message_id": None,
                    "last_message_id": None,
                    "first_timestamp_ms": timestamp_ms,
                    "last_timestamp_ms": timestamp_ms,
                    "turn_count": 1,
                    "approx_token_count": 3,
                    "text_display": text,
                    "text_search": text,
                    "source_category": "fixture",
                    "participants_search": "",
                    "source_ids_json": {},
                    "build_generation": "restore-fixture-generation",
                    "embedding_config_hash": None,
                    "embedding_updated_at": None,
                }
            )
            nlp_rows.append(
                {
                    "id": f"scale-nlp-{index:06d}",
                    "task": "sentiment",
                    "hour_start_ms": timestamp_ms,
                    "direction": "owner",
                    "pipeline_version": "fixture-v1",
                    "model_name": "fixture-model",
                    "model_revision": "fixture-v1",
                    "configuration_sha256": "3" * 64,
                    "configuration_json": {"fixture": True},
                    "predicted_label": "neutral",
                    "sample_count": 1,
                    "confidence_sum": 0.75,
                    "probability_sums_json": {"neutral": 0.75},
                    "active": True,
                    "generated_at": NOW,
                }
            )
            score_rows.append(
                {
                    "id": f"scale-score-{index:06d}",
                    "run_id": RUN_ID,
                    "window_id": window_id,
                    "input_sha256": "7" * 64,
                    "first_timestamp_ms": timestamp_ms,
                    "predicted_label": "direct_conflict",
                    "direct_conflict_score": 0.8,
                    "scores_json": {"direct_conflict": 0.8},
                    "active": True,
                    "provenance_json": {"fixture": True},
                    "created_at": NOW,
                }
            )
            conflict_rollup_rows.append(
                {
                    "id": f"scale-conflict-rollup-{index:06d}",
                    "run_id": RUN_ID,
                    "hour_start_ms": timestamp_ms,
                    "sample_count": 1,
                    "direct_conflict_count": 1,
                    "active": True,
                    "generated_at": NOW,
                }
            )

        session.execute(SearchWindow.__table__.insert(), window_rows)
        session.execute(NlpHourlyRollup.__table__.insert(), nlp_rows)
        session.execute(ConflictWindowScore.__table__.insert(), score_rows)
        session.execute(ConflictHourlyRollup.__table__.insert(), conflict_rollup_rows)
        run = session.get(ConflictAggregateRun, RUN_ID)
        assert run is not None
        run.window_count = total_rows
        run.scored_count = total_rows
        run.direct_conflict_count = total_rows
        session.flush()

        expanded = capture_derived_restore_state(
            session,
            nlp_targets={"message": {"message-restore-target"}},
            search_window_ids={WINDOW_ID},
        )
        generations = expanded["nlp_rollup_generations"]
        conflict_runs = expanded["conflict_runs"]
        assert isinstance(generations, list) and len(generations) == 1
        assert generations[0]["row_count"] == total_rows
        assert expanded["nlp_hourly_rollups"] == []
        assert isinstance(conflict_runs, list) and len(conflict_runs) == 1
        assert conflict_runs[0]["score_generation"]["row_count"] == total_rows
        assert conflict_runs[0]["rollup_generation"]["row_count"] == total_rows
        assert expanded["conflict_scores"] == []
        assert expanded["conflict_rollups"] == []
        assert len(json.dumps(expanded, sort_keys=True)) <= baseline_bytes + 512

        sealed = json.loads(json.dumps(_quarantine_and_seal(session, expanded)))
        affected = restore_derived_restore_state(session, sealed)
        assert affected == {
            "nlp_annotations_reactivated": 1,
            "nlp_hourly_rollups_reactivated": total_rows,
            "conflict_aggregate_runs_restored": 1,
            "conflict_window_scores_reactivated": total_rows,
            "conflict_hourly_rollups_reactivated": total_rows,
            "topic_cluster_jobs_restored": 1,
        }
        for model in (NlpHourlyRollup, ConflictWindowScore, ConflictHourlyRollup):
            assert (
                session.scalar(select(func.count(model.id)).where(model.active.is_(True)))
                == total_rows
            )
