from __future__ import annotations

import hashlib
import json
import stat
from collections.abc import Sequence
from pathlib import Path

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from meta_archive.config import settings
from meta_archive.models import (
    RelevanceFeedback,
    SearchWindow,
    SemanticIndexConfig,
    SemanticWindowEmbedding,
)
from meta_archive.paths import ArchiveLayout
from meta_archive.services import semantic_bakeoff as bakeoff_service
from meta_archive.services.direct_conflict_ranker import DIRECT_CONFLICT_EVALUATION_INTENT
from meta_archive.services.search_index import (
    EmbeddingModelSpec,
    embedding_implementation_identity,
    embedding_model_spec,
)
from meta_archive.services.semantic_bakeoff import (
    CONTROL_MODEL_ALIAS,
    BakeoffError,
    available_model_aliases,
    build_bakeoff_manifest,
    evaluate_bakeoff,
    explicit_model_roster,
    production_embedder_factory,
    write_private_report,
)

PRIVATE_QUERY = "private conflict query"
PRIVATE_PARTICIPANT = "Private Participant"
DEFAULT_SCOPE_SHA256 = hashlib.sha256(b"{}").hexdigest()
ROSTER = (CONTROL_MODEL_ALIAS, "bge-base-en-v1.5")


class _FixtureEmbedder:
    def __init__(self, spec: EmbeddingModelSpec) -> None:
        self.spec = spec

    def _vector(self, text: str) -> list[float]:
        values = [0.0] * self.spec.dimensions
        if "conflict" in text:
            values[0] = 1.0
        elif "peaceful" in text:
            values[1] = 1.0
        else:
            values[2] = 1.0
        return values

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class _ImprovingCandidateEmbedder(_FixtureEmbedder):
    def embed_query(self, text: str) -> list[float]:
        if self.spec.alias == CONTROL_MODEL_ALIAS:
            values = [0.0] * self.spec.dimensions
            values[1] = 1.0
            return values
        return super().embed_query(text)


def _window(window_id: str, text: str, *, ordinal: int) -> SearchWindow:
    return SearchWindow(
        id=window_id,
        window_version="message-window-v1",
        entity_type="message_window",
        conversation_id=None,
        segment_id=None,
        activity_id=None,
        audio_segment_id=None,
        attachment_extraction_id=None,
        attachment_occurrence_id=None,
        first_message_id=None,
        last_message_id=None,
        first_timestamp_ms=1_600_000_000_000 + ordinal,
        last_timestamp_ms=1_600_000_000_000 + ordinal,
        turn_count=1,
        approx_token_count=4,
        text_display=text,
        text_search=text.casefold(),
        source_category="messages",
        participants_search=PRIVATE_PARTICIPANT.casefold(),
        source_ids_json={},
        build_generation="private-bakeoff-fixture",
    )


def _feedback(
    feedback_id: str,
    *,
    result_id: str,
    label: str,
    evaluation_split: str | None,
    evaluation_intent: str | None,
    query: str = PRIVATE_QUERY,
) -> RelevanceFeedback:
    return RelevanceFeedback(
        id=feedback_id,
        query_sha256=hashlib.sha256(query.encode()).hexdigest(),
        query_text=query,
        search_mode="semantic",
        filter_scope_sha256=DEFAULT_SCOPE_SHA256,
        filter_scope_json={},
        result_id=f"window:{result_id}",
        label=label,
        evaluation_split=evaluation_split,
        evaluation_intent=evaluation_intent,
    )


def _seed_bakeoff_catalog(engine: Engine) -> tuple[list[str], list[str]]:
    control = embedding_model_spec(CONTROL_MODEL_ALIAS)
    config_id = "fixture-active-control"
    relevant_ids = [f"private-relevant-{index:02d}" for index in range(12)]
    not_relevant_ids = [f"private-negative-{index:02d}" for index in range(12)]
    distractor_ids = [f"private-distractor-{index:02d}" for index in range(16)]
    with Session(engine) as session, session.begin():
        session.add(
            SemanticIndexConfig(
                id=config_id,
                active=True,
                status="ready",
                backend="fixture",
                backend_version="fixture",
                model_name=control.model_name,
                model_revision=control.model_revision,
                dimensions=control.dimensions,
                distance_metric="cosine",
                document_prefix=control.document_prefix,
                query_prefix=control.query_prefix,
                vector_table="fixture_vectors",
                window_version="message-window-v1",
                window_count=len(relevant_ids) + len(not_relevant_ids) + len(distractor_ids),
                audit_json={"embedding_implementation": embedding_implementation_identity(control)},
            )
        )
        windows = [
            *[
                _window(window_id, f"private conflict material {index}", ordinal=index)
                for index, window_id in enumerate(relevant_ids)
            ],
            *[
                _window(window_id, f"private peaceful material {index}", ordinal=100 + index)
                for index, window_id in enumerate(not_relevant_ids)
            ],
            *[
                _window(window_id, f"private miscellaneous material {index}", ordinal=200 + index)
                for index, window_id in enumerate(distractor_ids)
            ],
        ]
        session.add_all(windows)
        session.flush()
        session.add_all(
            [
                SemanticWindowEmbedding(config_id=config_id, window_id=window.id)
                for window in windows
            ]
        )
        session.add_all(
            [
                *[
                    _feedback(
                        f"private-relevant-feedback-{index:02d}",
                        result_id=window_id,
                        label="relevant",
                        evaluation_split="validation",
                        evaluation_intent=DIRECT_CONFLICT_EVALUATION_INTENT,
                    )
                    for index, window_id in enumerate(relevant_ids)
                ],
                *[
                    _feedback(
                        f"private-negative-feedback-{index:02d}",
                        result_id=window_id,
                        label="not_relevant",
                        evaluation_split="validation",
                        evaluation_intent=DIRECT_CONFLICT_EVALUATION_INTENT,
                    )
                    for index, window_id in enumerate(not_relevant_ids)
                ],
                _feedback(
                    "private-test-feedback",
                    result_id="private-target-that-does-not-exist",
                    label="relevant",
                    evaluation_split="test",
                    evaluation_intent=None,
                    query="private held out query that must never enter development",
                ),
            ]
        )
    return relevant_ids, not_relevant_ids


def test_manifest_is_deterministic_bounded_and_omits_private_material(
    archive_layout: ArchiveLayout,
    catalog_engine: Engine,
) -> None:
    _seed_bakeoff_catalog(catalog_engine)

    first = build_bakeoff_manifest(
        archive_layout,
        model_aliases=ROSTER,
        corpus_cap=30,
        hard_distractor_cap=3,
    )
    second = build_bakeoff_manifest(
        archive_layout,
        model_aliases=ROSTER,
        corpus_cap=30,
        hard_distractor_cap=3,
    )

    assert first.manifest_digest_sha256 == second.manifest_digest_sha256
    summary = first.safe_summary()
    assert summary["manifest"]["query_group_count"] == 1
    assert summary["manifest"]["judgment_count"] == 24
    assert summary["manifest"]["judged_target_count"] == 24
    assert summary["manifest"]["corpus_window_count"] == 30
    assert summary["manifest"]["sample_source_counts"] == {
        "judged": 24,
        "hard_distractor": 3,
        "stable_distractor": 3,
    }
    assert summary["development_only"] is True
    assert summary["activation_permitted"] is False

    rendered = json.dumps(summary, sort_keys=True)
    for private_value in (
        PRIVATE_QUERY,
        PRIVATE_PARTICIPANT,
        "private-relevant",
        "private-negative",
        "private-distractor",
        "private held out query",
    ):
        assert private_value not in rendered


def test_exact_cosine_bakeoff_reports_aggregate_control_comparison_and_grouped_validation(
    archive_layout: ArchiveLayout,
    catalog_engine: Engine,
) -> None:
    _seed_bakeoff_catalog(catalog_engine)
    manifest = build_bakeoff_manifest(
        archive_layout,
        model_aliases=ROSTER,
        corpus_cap=30,
        hard_distractor_cap=3,
    )

    report = evaluate_bakeoff(manifest, embedder_factory=_FixtureEmbedder)

    assert report["status"] == "completed"
    assert report["evaluation"]["kind"] == "bounded_development_model_selection"
    assert report["evaluation"]["activation_permitted"] is False
    assert len(report["evaluation"]["results"]) == 2
    for result in report["evaluation"]["results"]:
        metrics = result["retrieval"]["metrics"]
        assert metrics["recall_at_10"] == pytest.approx(10 / 12)
        assert metrics["recall_at_50"] == 1.0
        assert metrics["recall_at_200"] == 1.0
        assert result["direct_conflict_grouped_validation"]["status"] == "completed"
        assert result["direct_conflict_grouped_validation"]["fold_count"] == 4
    gate = report["evaluation"]["comparisons_to_control"][0]["development_gate"]
    assert gate["metric_deltas"] == {
        "recall_at_10": 0.0,
        "mrr_at_10": 0.0,
        "ndcg_at_10": 0.0,
        "recall_at_50": 0.0,
        "recall_at_200": 0.0,
        "average_precision_at_200": 0.0,
    }
    assert gate["passed"] is False
    assert gate["checks"]["demonstrated_improvement"] is False
    assert report["evaluation"]["development_recommendation"] == {
        "status": "no_candidate_passed",
        "alias": None,
        "keep_active_control": True,
        "basis": (
            "Among candidates passing every explicit development gate, highest Recall@200, "
            "then nDCG@10, then Recall@10 on the bounded exact-cosine sample"
        ),
        "requires_full_corpus_regression_acceptance": True,
        "blind_acceptance_available": False,
        "existing_test_cohort_role": "exposed_overlap_regression_only",
    }

    rendered = json.dumps(report, sort_keys=True)
    assert PRIVATE_QUERY not in rendered
    assert PRIVATE_PARTICIPANT not in rendered
    assert "private-relevant" not in rendered


def test_development_gate_recommends_only_a_candidate_that_beats_control(
    archive_layout: ArchiveLayout,
    catalog_engine: Engine,
) -> None:
    _seed_bakeoff_catalog(catalog_engine)
    manifest = build_bakeoff_manifest(
        archive_layout,
        model_aliases=ROSTER,
        corpus_cap=30,
        hard_distractor_cap=3,
    )

    report = evaluate_bakeoff(
        manifest,
        embedder_factory=_ImprovingCandidateEmbedder,
    )

    comparison = report["evaluation"]["comparisons_to_control"][0]
    assert comparison["development_gate"]["passed"] is True
    assert comparison["development_gate"]["checks"] == {
        "broad_recall_non_regression": True,
        "top_rank_non_regression": True,
        "demonstrated_improvement": True,
        "direct_conflict_query_non_regression": True,
        "direct_conflict_grouped_minimums": True,
    }
    assert report["evaluation"]["development_recommendation"]["status"] == "candidate_passed"
    assert report["evaluation"]["development_recommendation"]["alias"] == "bge-base-en-v1.5"
    assert report["evaluation"]["development_recommendation"]["keep_active_control"] is False


def test_manifest_fails_closed_on_missing_development_target_and_caps(
    archive_layout: ArchiveLayout,
    catalog_engine: Engine,
) -> None:
    _seed_bakeoff_catalog(catalog_engine)
    with Session(catalog_engine) as session, session.begin():
        session.add(
            _feedback(
                "private-missing-development-feedback",
                result_id="private-missing-development-target",
                label="relevant",
                evaluation_split="train",
                evaluation_intent=None,
                query="another private development query",
            )
        )

    with pytest.raises(BakeoffError) as missing:
        build_bakeoff_manifest(
            archive_layout,
            model_aliases=ROSTER,
            corpus_cap=30,
            hard_distractor_cap=3,
        )
    assert missing.value.code == "judged_target_missing_from_active_corpus"
    assert str(missing.value) == "judged_target_missing_from_active_corpus"

    with Session(catalog_engine) as session, session.begin():
        session.query(RelevanceFeedback).filter(
            RelevanceFeedback.id == "private-missing-development-feedback"
        ).delete()

    with pytest.raises(BakeoffError) as corpus_cap:
        build_bakeoff_manifest(
            archive_layout,
            model_aliases=ROSTER,
            corpus_cap=23,
            hard_distractor_cap=3,
        )
    assert corpus_cap.value.code == "judged_targets_exceed_corpus_cap"

    with pytest.raises(BakeoffError) as judgment_cap:
        build_bakeoff_manifest(
            archive_layout,
            model_aliases=ROSTER,
            corpus_cap=30,
            hard_distractor_cap=3,
            judgment_cap_per_query=23,
        )
    assert judgment_cap.value.code == "development_judgment_cap_exceeded"


def test_explicit_roster_requires_e5_control_and_candidate() -> None:
    with pytest.raises(BakeoffError, match="model_roster_required"):
        explicit_model_roster(())
    with pytest.raises(BakeoffError, match="active_e5_control_required"):
        explicit_model_roster(("e5-large-v2",))
    with pytest.raises(BakeoffError, match="candidate_model_required"):
        explicit_model_roster((CONTROL_MODEL_ALIAS,))
    with pytest.raises(BakeoffError, match="model_roster_duplicate"):
        explicit_model_roster((CONTROL_MODEL_ALIAS, CONTROL_MODEL_ALIAS))
    assert explicit_model_roster(ROSTER) == tuple(embedding_model_spec(alias) for alias in ROSTER)
    for candidate in ("e5-base-v2", "e5-small-v2", "nomic-embed-text-v1.5"):
        candidate_roster = (CONTROL_MODEL_ALIAS, candidate)
        assert candidate in available_model_aliases()
        assert explicit_model_roster(candidate_roster) == tuple(
            embedding_model_spec(alias) for alias in candidate_roster
        )


def test_production_factory_applies_fastembed_batch_size_and_fails_if_it_drifts(
    archive_layout: ArchiveLayout,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int, bool]] = []

    def fake_default(
        _layout: ArchiveLayout,
        spec: EmbeddingModelSpec,
        *,
        local_files_only: bool,
    ) -> _FixtureEmbedder:
        calls.append((spec.alias, settings.embedding_batch_size, local_files_only))
        return _FixtureEmbedder(spec)

    monkeypatch.setattr(bakeoff_service, "default_text_embedder", fake_default)
    monkeypatch.setattr(settings, "embedding_batch_size", 128)
    factory = production_embedder_factory(
        archive_layout,
        batch_size=4,
        local_files_only=True,
    )

    embedder = factory(embedding_model_spec("bge-base-en-v1.5"))

    assert embedder.spec.alias == "bge-base-en-v1.5"
    assert calls == [("bge-base-en-v1.5", 4, True)]
    settings.embedding_batch_size = 8
    with pytest.raises(BakeoffError, match="embedding_batch_size_changed"):
        factory(embedding_model_spec("bge-base-en-v1.5"))


def test_aggregate_report_is_owner_only_and_contains_no_private_values(tmp_path: Path) -> None:
    path = tmp_path / "bakeoff.json"
    report = {
        "status": "manifest_ready",
        "manifest": {"sha256": "a" * 64},
        "private_query_details_persisted": False,
    }

    write_private_report(path, report)

    assert json.loads(path.read_text()) == report
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
