from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from meta_archive.api.main import create_app
from meta_archive.api.schemas import RagStructuredAnswer, SemanticIndexRequest
from meta_archive.api.search_routes import _public_semantic_evaluation, _scope_hash
from meta_archive.database import create_catalog_engine, run_migrations
from meta_archive.models import Job, Message, RagAudit, SemanticIndexConfig
from meta_archive.paths import ArchiveLayout
from meta_archive.services import search_index as search_index_service
from meta_archive.services.activity_importer import import_activities
from meta_archive.services.inventory import scan_archive
from meta_archive.services.message_importer import import_messages
from meta_archive.services.reranking import DEFAULT_RERANKER_SPEC, RerankerSpec
from meta_archive.services.search import SearchFilters
from meta_archive.services.search_index import EmbeddingModelSpec, build_semantic_index
from meta_archive.services.semantic_state import invalidate_ready_semantic_indexes

MESSAGE_FIXTURES = Path(__file__).parent / "fixtures" / "messages"
ACTIVITY_FIXTURES = Path(__file__).parent / "fixtures" / "activity"

FIXTURE_SPEC = EmbeddingModelSpec(
    alias="fixture-embedding",
    model_name="fixture/embedding",
    model_revision="fixture-v1",
    dimensions=4,
    document_prefix="passage: ",
    query_prefix="query: ",
    language_scope="Fixture text",
    approximate_download_gb=0,
)


class FixtureEmbedder:
    def __init__(self, spec: EmbeddingModelSpec = FIXTURE_SPEC) -> None:
        self.spec = spec

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = [float(digest[index % len(digest)] + 1) for index in range(self.spec.dimensions)]
        norm = sum(value * value for value in values) ** 0.5
        return [value / norm for value in values]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class FixtureReranker:
    spec = DEFAULT_RERANKER_SPEC

    @property
    def status(self) -> dict[str, object]:
        return {
            "status": "loaded_fixture",
            "model_name": self.spec.model_name,
            "model_revision": self.spec.model_revision,
            "device": "cpu",
            "score_kind": self.spec.score_kind,
            "max_candidates": self.spec.max_candidates,
            "batch_size": self.spec.batch_size,
        }

    def rerank(self, _query: str, documents: Sequence[str]) -> list[float]:
        return [float(index) for index, _document in enumerate(documents)]


def fixture_reranker_factory(_spec: RerankerSpec) -> FixtureReranker:
    return FixtureReranker()


def _assert_search_result_contract(item: dict[str, Any]) -> None:
    assert item["coverage"]["status"] in {"available", "partial", "unavailable"}
    assert isinstance(item["coverage"]["absence_is_not_zero_history"], bool)
    assert "confidence" in item
    assert item["confidence_kind"] in {"not_applicable", "audio_decoder_score"}
    assert item["confidence_calibrated"] is False
    if item["confidence_kind"] == "not_applicable":
        assert item["confidence"] is None
    assert item["availability"]
    assert item["deletion_state"] == "retained"
    assert item["provenance"]["entity_type"] == item["entity_type"]
    assert item["provenance"]["source_id"] == item["source_id"]


def test_semantic_index_request_accepts_and_documents_bge_base_default() -> None:
    request = SemanticIndexRequest(model="bge-base-en-v1.5", rebuild_windows=False)
    schema_text = json.dumps(SemanticIndexRequest.model_json_schema(), sort_keys=True)

    assert request.model == "bge-base-en-v1.5"
    assert request.rebuild_windows is False
    assert "default auto policy selects bge-base-en-v1.5" in schema_text
    assert "manual acceptance override is intentionally unavailable over HTTP" in schema_text

    with pytest.raises(ValidationError, match="extra_forbidden"):
        SemanticIndexRequest.model_validate(
            {
                "model": "bge-base-en-v1.5",
                "rebuild_windows": False,
                "manual_activation_override": True,
            }
        )


def test_public_semantic_evaluation_redacts_development_calibration_details() -> None:
    held_out_revision: dict[str, int | str] = {
        "version": "held-out-feedback-revision-v1",
        "sha256": "held-out-private-digest",
        "row_count": 3,
    }
    development_revision: dict[str, int | str] = {
        "version": "development-feedback-revision-v1",
        "sha256": "development-private-digest",
        "row_count": 214,
    }
    evaluation = {
        "passed": True,
        "production_acceptance_passed": True,
        "automatic_known_item": {"queries": [{"query_text": "private automatic query"}]},
        "held_out_feedback": {
            "queries": [{"query_text": "private held-out query"}],
            "feedback_revision": held_out_revision,
            "feedback_revision_stable_during_evaluation": True,
        },
        "direct_conflict_development_calibration": {
            "benchmark_kind": "literal_direct_conflict_development",
            "evaluation_intent": "argument_direct_conflict_development",
            "evidence_kind": "conversation_grouped_development_calibration",
            "held_out": False,
            "required_for_production_acceptance": True,
            "status": "passed",
            "passed": True,
            "method": "linear_svm",
            "query_details_persisted": False,
            "model_metadata": {
                "version": "linear-svc-direct-conflict-v2",
                "parameters": {
                    "c": 1.0,
                    "class_weight": "balanced",
                    "random_state": 20260722,
                },
                "counts": {
                    "reviewed_feedback_rows": 214,
                    "binary_training_rows": 212,
                    "conversation_groups": 85,
                },
                "score_kind": "linear_svc_decision_function",
                "ranking_feature_only": True,
                "private_training_rows": ["window:private"],
                "private_material_fingerprint_sha256": "private-material-fingerprint",
            },
            "judgment_count": 214,
            "conversation_count": 85,
            "fold_count": 4,
            "relevant_count": 27,
            "not_relevant_count": 185,
            "uncertain_count": 2,
            "grouped_cross_validation": {
                "minimum_metrics": {"recall_at_k": 0.667, "mrr": 1.0},
                "mean_metrics": {"recall_at_k": 0.699, "mrr": 1.0},
                "folds": [{"conversation_ids": ["private-conversation"]}],
            },
            "literal_query_regression": {
                "metrics": {"recall_at_k": 0.75, "mrr": 1.0},
                "rankings": ["window:private"],
            },
            "feedback_revision": development_revision,
            "feedback_revision_stable_during_evaluation": True,
            "query_text": "private development query",
            "query_sha256": "private-query-digest",
            "result_ids": ["window:private"],
            "conversation_ids": ["private-conversation"],
        },
    }

    public = _public_semantic_evaluation(
        evaluation,
        held_out_revision,
        development_revision,
    )

    assert public is not None
    development = public["direct_conflict_development_calibration"]
    assert development["evidence_kind"] == "conversation_grouped_development_calibration"
    assert development["held_out"] is False
    assert development["judgment_count"] == 214
    assert development["model_metadata"] == {
        "version": "linear-svc-direct-conflict-v2",
        "parameters": {
            "c": 1.0,
            "class_weight": "balanced",
            "random_state": 20260722,
        },
        "counts": {
            "reviewed_feedback_rows": 214,
            "binary_training_rows": 212,
            "conversation_groups": 85,
        },
        "score_kind": "linear_svc_decision_function",
        "ranking_feature_only": True,
    }
    assert development["grouped_cross_validation"] == {
        "minimum_metrics": {"recall_at_k": 0.667, "mrr": 1.0},
        "mean_metrics": {"recall_at_k": 0.699, "mrr": 1.0},
    }
    assert development["literal_query_regression"] == {"metrics": {"recall_at_k": 0.75, "mrr": 1.0}}
    assert development["feedback_revision"] == {
        "version": "development-feedback-revision-v1",
        "row_count": 214,
        "current": True,
    }
    rendered = json.dumps(public, sort_keys=True)
    assert "private development query" not in rendered
    assert "private-query-digest" not in rendered
    assert "window:private" not in rendered
    assert "private-conversation" not in rendered
    assert "development-private-digest" not in rendered
    assert "private-material-fingerprint" not in rendered

    evaluation_without_held_out = {**evaluation}
    evaluation_without_held_out.pop("held_out_feedback")
    public_without_held_out = _public_semantic_evaluation(
        evaluation_without_held_out,
        held_out_revision,
        development_revision,
    )
    assert public_without_held_out is not None
    assert "private development query" not in json.dumps(public_without_held_out, sort_keys=True)

    stale = _public_semantic_evaluation(
        evaluation,
        held_out_revision,
        {**development_revision, "row_count": 215},
    )
    assert stale is not None
    assert stale["passed"] is True
    assert stale["production_acceptance_passed"] is False
    assert stale["direct_conflict_development_calibration"]["status"] == "stale_feedback"
    assert stale["direct_conflict_development_calibration"]["feedback_revision"]["current"] is False


def test_semantic_cursor_scope_binds_the_direct_conflict_feedback_revision() -> None:
    filters = SearchFilters()
    first = _scope_hash(
        "argument",
        "semantic",
        filters,
        semantic_config_id="fixture-config",
        direct_conflict_revision_sha256="a" * 64,
    )
    changed = _scope_hash(
        "argument",
        "semantic",
        filters,
        semantic_config_id="fixture-config",
        direct_conflict_revision_sha256="b" * 64,
    )

    assert first != changed


class FakeResponses:
    def __init__(self, citation_id: str | None) -> None:
        self.citation_id = citation_id
        self.calls: list[dict[str, Any]] = []

    def parse(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(
            id="resp_fixture",
            output_parsed=RagStructuredAnswer(
                answer="The fixture excerpts support this answer.",
                citation_ids=[self.citation_id] if self.citation_id is not None else [],
                insufficient_evidence=False,
            ),
            usage=SimpleNamespace(model_dump=lambda: {"input_tokens": 42, "output_tokens": 12}),
        )


class FakeOpenAI:
    def __init__(self, citation_id: str | None) -> None:
        self.responses = FakeResponses(citation_id)


@pytest.fixture
def stage3_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[TestClient]:
    layout = ArchiveLayout(tmp_path / "meta-backup-stage3")
    shutil.copytree(MESSAGE_FIXTURES, layout.root)
    shutil.copytree(ACTIVITY_FIXTURES, layout.root / "activity-package")
    layout.ensure_runtime_directories()
    run_migrations(layout.database)
    engine = create_catalog_engine(layout.database)
    scan_archive(layout, engine, hash_files=True)
    import_messages(layout, engine)
    import_activities(layout, engine)
    monkeypatch.setattr(
        search_index_service,
        "_direct_conflict_development_evaluation",
        lambda *_args, **_kwargs: {
            **search_index_service._direct_conflict_calibration_base(),
            "status": "passed",
            "passed": True,
            "feedback_revision": {
                "version": "fixture-direct-conflict-feedback-v1",
                "sha256": "a" * 64,
                "row_count": 20,
            },
            "feedback_revision_stable_during_evaluation": True,
            "model_metadata": {
                "classifier": "fixture",
                "ranker_version": "fixture-direct-conflict-ranker-v1",
            },
            "private_material_fingerprint_sha256": "b" * 64,
        },
    )
    build_semantic_index(layout, engine, embedder=FixtureEmbedder(), batch_size=4)
    engine.dispose()

    app = create_app(layout.root, security_disabled=True)
    app.state.search_embedder_factory = FixtureEmbedder
    app.state.search_reranker_factory = fixture_reranker_factory
    app.state.semantic_index_embedder_factory = FixtureEmbedder
    with TestClient(app) as client:
        yield client


def test_keyword_exact_semantic_and_hybrid_search_with_scoped_pagination(
    stage3_client: TestClient,
) -> None:
    embedder_factory_calls: list[str] = []

    def counting_factory(spec: EmbeddingModelSpec) -> FixtureEmbedder:
        embedder_factory_calls.append(spec.model_name)
        return FixtureEmbedder(spec)

    stage3_client.app.state.search_embedder_factory = counting_factory
    stage3_client.app.state.search_embedder_cache = {}
    keyword = stage3_client.get(
        "/api/v1/search", params={"q": "fixture", "mode": "keyword", "limit": 2}
    )
    assert keyword.status_code == 200, keyword.text
    keyword_payload = keyword.json()
    assert len(keyword_payload["items"]) == 2
    assert keyword_payload["page"]["next_cursor"]
    assert all(item["jump"]["href"] for item in keyword_payload["items"])
    for item in keyword_payload["items"]:
        _assert_search_result_contract(item)

    next_page = stage3_client.get(
        "/api/v1/search",
        params={
            "q": "fixture",
            "mode": "keyword",
            "limit": 2,
            "cursor": keyword_payload["page"]["next_cursor"],
        },
    )
    assert next_page.status_code == 200
    assert {item["id"] for item in next_page.json()["items"]}.isdisjoint(
        {item["id"] for item in keyword_payload["items"]}
    )
    mismatched = stage3_client.get(
        "/api/v1/search",
        params={
            "q": "different",
            "mode": "keyword",
            "limit": 2,
            "cursor": keyword_payload["page"]["next_cursor"],
        },
    )
    assert mismatched.status_code == 422

    exact = stage3_client.get(
        "/api/v1/search",
        params={
            "q": "A café comment containing only fixture text",
            "mode": "exact",
            "entity_type": "activity",
        },
    )
    assert exact.status_code == 200, exact.text
    assert [item["entity_type"] for item in exact.json()["items"]] == ["activity"]
    assert exact.json()["items"][0]["source_category"] == "groups"
    assert exact.json()["items"][0]["jump"]["href"].startswith("/timeline?around=activity:")
    _assert_search_result_contract(exact.json()["items"][0])

    semantic = stage3_client.get(
        "/api/v1/search", params={"q": "a tense disagreement", "mode": "semantic"}
    )
    assert semantic.status_code == 200, semantic.text
    assert semantic.json()["index_config"]["model_name"] == FIXTURE_SPEC.model_name
    assert semantic.json()["items"]
    assert all(
        item["score_kind"] == DEFAULT_RERANKER_SPEC.score_kind for item in semantic.json()["items"]
    )
    assert all(item["window_id"] for item in semantic.json()["items"])
    assert all(item["window"] is not None for item in semantic.json()["items"])
    for item in semantic.json()["items"]:
        _assert_search_result_contract(item)

    participant_filtered = stage3_client.get(
        "/api/v1/search",
        params={
            "q": "fixture",
            "mode": "semantic",
            "entity_type": "message",
            "participant": "Alex Example",
        },
    )
    assert participant_filtered.status_code == 200, participant_filtered.text
    assert participant_filtered.json()["items"]
    assert all(
        "alex example" in item["participants"] for item in participant_filtered.json()["items"]
    )

    hybrid = stage3_client.get(
        "/api/v1/search", params={"q": "fixture", "mode": "hybrid", "limit": 2}
    )
    assert hybrid.status_code == 200, hybrid.text
    assert hybrid.json()["items"]
    assert all(item["score_kind"] == "rrf" for item in hybrid.json()["items"])
    for item in hybrid.json()["items"]:
        _assert_search_result_contract(item)
    assert embedder_factory_calls == [FIXTURE_SPEC.model_name]
    if hybrid.json()["page"]["next_cursor"]:
        hybrid_next = stage3_client.get(
            "/api/v1/search",
            params={
                "q": "fixture",
                "mode": "hybrid",
                "limit": 2,
                "cursor": hybrid.json()["page"]["next_cursor"],
            },
        )
        assert hybrid_next.status_code == 200
        assert {item["id"] for item in hybrid_next.json()["items"]}.isdisjoint(
            {item["id"] for item in hybrid.json()["items"]}
        )

    status_response = stage3_client.get("/api/v1/search/status")
    assert status_response.status_code == 200
    assert status_response.json()["ready"] is True
    assert status_response.json()["active"]["audit"]["selected_model"]["model_name"] == (
        FIXTURE_SPEC.model_name
    )
    acceptance = status_response.json()["acceptance"]
    assert acceptance["passed"] is True
    public_revision = acceptance["held_out_feedback"]["feedback_revision"]
    assert public_revision == {
        "version": "held-out-feedback-revision-v1",
        "row_count": 0,
        "current": True,
    }
    assert "sha256" not in public_revision
    assert "queries" not in acceptance["automatic_known_item"]
    assert "queries" not in acceptance["held_out_feedback"]
    acceptance_metrics = acceptance["automatic_known_item"]["metrics"]
    assert 0 < acceptance_metrics["query_count"] <= 48
    metric_names = ("k", "recall_at_k", "mrr", "ndcg_at_k")
    assert {key: acceptance_metrics[key] for key in metric_names} == {
        "k": 10,
        "recall_at_k": 1.0,
        "mrr": 1.0,
        "ndcg_at_k": 1.0,
    }
    assert status_response.json()["latest_evaluation"] == acceptance
    assert status_response.json()["latest_candidate"]["status"] == "ready"
    assert status_response.json()["latest_candidate"]["evaluation"] == acceptance
    assert status_response.json()["latest_job"]["counters"]["acceptance_passed"] is True
    assert status_response.json()["reranker"]["status"] == "loaded_fixture"
    assert status_response.json()["reranker"]["model_revision"] == (
        DEFAULT_RERANKER_SPEC.model_revision
    )

    with Session(stage3_client.app.state.engine) as session:
        textless = session.scalar(
            select(Message)
            .where(
                Message.content_display.is_(None),
                Message.transcription_display.is_(None),
                Message.deleted_at.is_(None),
            )
            .order_by(Message.id)
        )
    assert textless is not None
    sender_hybrid = stage3_client.get(
        "/api/v1/search",
        params={"q": textless.sender_display, "mode": "hybrid", "entity_type": "message"},
    )
    assert sender_hybrid.status_code == 200
    assert f"message:{textless.id}" in {item["id"] for item in sender_hybrid.json()["items"]}

    message_window = next(
        item for item in semantic.json()["items"] if item["entity_type"] == "message_window"
    )
    deleted_message_id = message_window["window"]["start_message_id"]
    with Session(stage3_client.app.state.engine) as session, session.begin():
        session.execute(
            update(Message)
            .where(Message.id == deleted_message_id)
            .values(deleted_at=datetime.now(UTC))
        )
    after_delete = stage3_client.get(
        "/api/v1/search", params={"q": "a tense disagreement", "mode": "semantic"}
    )
    assert after_delete.status_code == 200
    assert message_window["id"] not in {item["id"] for item in after_delete.json()["items"]}


def test_search_results_star_the_canonical_source_identity(stage3_client: TestClient) -> None:
    initial = stage3_client.get(
        "/api/v1/search",
        params={"q": "fixture", "mode": "keyword", "limit": 100},
    )
    assert initial.status_code == 200
    source = next(
        item for item in initial.json()["items"] if item["entity_type"] in {"message", "activity"}
    )
    target_type = source["entity_type"]
    target_id = source["source_id"]
    assert source["id"] == f"{target_type}:{target_id}"

    created = stage3_client.put(f"/api/v1/stars/{target_type}/{target_id}")
    assert created.status_code == 200
    refreshed = stage3_client.get(
        "/api/v1/search",
        params={"q": "fixture", "mode": "keyword", "limit": 100},
    )
    assert refreshed.status_code == 200
    starred = next(item for item in refreshed.json()["items"] if item["id"] == source["id"])
    assert starred["star_target"] == {
        "target_type": target_type,
        "target_id": target_id,
    }
    assert starred["star"] == created.json()["star"]


def test_manual_override_argument_uses_cross_encoder_fallback_over_http(
    stage3_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with Session(stage3_client.app.state.engine) as session, session.begin():
        active = session.scalar(
            select(SemanticIndexConfig).where(
                SemanticIndexConfig.active.is_(True),
                SemanticIndexConfig.status == "ready",
            )
        )
        assert active is not None
        audit = json.loads(json.dumps(active.audit_json))
        audit.pop("accepted_semantic_ir_evaluation", None)
        audit["manual_activation_override"] = {
            "version": "semantic-manual-activation-v1",
            "candidate_config_id": active.id,
            "acknowledged_failed_quality_gate": True,
            "source": "cli_only",
            "activation_job_id": "fixture-manual-activation",
        }
        active.audit_json = audit

    def forbid_development_feedback(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("manual fallback must not require development feedback")

    monkeypatch.setattr(
        "meta_archive.api.search_routes.development_feedback_revision_payload",
        forbid_development_feedback,
    )
    monkeypatch.setattr(
        "meta_archive.services.search.development_feedback_revision_payload",
        forbid_development_feedback,
    )

    for mode in ("semantic", "hybrid"):
        response = stage3_client.get(
            "/api/v1/search",
            params={"q": "argument", "mode": mode, "limit": 10},
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["items"]
        fallback = payload["index_config"]["direct_conflict_ranker"]
        assert fallback == {
            "status": "normal_cross_encoder_fallback",
            "specialized_ranker_used": False,
            "reason": "manual_override_without_accepted_direct_conflict_calibration",
            "trigger": "exact_normalized_literal",
            "pipeline": "cosine_retrieval_then_local_cross_encoder",
            "manual_activation_version": "semantic-manual-activation-v1",
            "manual_activation_job_id": "fixture-manual-activation",
        }
        assert all(
            item["ranking_provenance"]["literal_direct_conflict"] == fallback
            for item in payload["items"]
        )


def test_semantic_search_reports_an_unavailable_reranker_without_cosine_fallback(
    stage3_client: TestClient,
) -> None:
    def unavailable_reranker(_spec: RerankerSpec) -> FixtureReranker:
        raise RuntimeError("fixture model is unavailable")

    stage3_client.app.state.search_reranker_factory = unavailable_reranker
    stage3_client.app.state.search_reranker_cache = {}
    response = stage3_client.get("/api/v1/search", params={"q": "fixture", "mode": "semantic"})
    assert response.status_code == 409
    assert "will not silently return cosine-only" in response.json()["detail"]

    status_response = stage3_client.get("/api/v1/search/status")
    assert status_response.status_code == 200
    assert status_response.json()["reranker"]["status"] == "unavailable"
    assert "fixture model is unavailable" in status_response.json()["reranker"]["error"]


def test_stale_semantic_index_status_requires_rebuild(stage3_client: TestClient) -> None:
    with Session(stage3_client.app.state.engine) as session, session.begin():
        invalidated = invalidate_ready_semantic_indexes(
            session,
            reason="Searchable fixture transcript changed",
            source_kind="audio_transcription",
            source_id="fixture-transcription",
        )
    assert invalidated == 1

    status = stage3_client.get("/api/v1/search/status")
    assert status.status_code == 200
    assert status.json()["ready"] is False
    assert status.json()["needs_rebuild"] is True
    assert status.json()["stale_config"]["status"] == "stale"
    assert status.json()["stale_reason"]["source_id"] == "fixture-transcription"
    unavailable = stage3_client.get(
        "/api/v1/search",
        params={"q": "fixture", "mode": "semantic"},
    )
    assert unavailable.status_code == 409


def test_feedback_identity_and_pagination_include_filter_scope(
    stage3_client: TestClient,
) -> None:
    base = {
        "query": "scoped fixture",
        "search_mode": "semantic",
        "result_id": "window:fixture",
        "label": "relevant",
        "evaluation_split": "test",
        "evaluation_intent": "argument_direct_conflict",
    }
    first = stage3_client.post(
        "/api/v1/search/feedback",
        json={
            **base,
            "filter_scope": {
                "entity_type": "message",
                "conversation_id": "conversation-one",
            },
        },
    )
    same_scope = stage3_client.post(
        "/api/v1/search/feedback",
        json={
            **base,
            "label": "uncertain",
            "filter_scope": {
                "entity_type": "message",
                "conversation_id": "conversation-one",
            },
        },
    )
    other_scope = stage3_client.post(
        "/api/v1/search/feedback",
        json={
            **base,
            "filter_scope": {
                "entity_type": "message",
                "conversation_id": "conversation-two",
            },
        },
    )
    assert first.status_code == same_scope.status_code == other_scope.status_code == 201
    assert same_scope.json()["id"] == first.json()["id"]
    assert same_scope.json()["updated"] is True
    assert other_scope.json()["id"] != first.json()["id"]
    assert first.json()["filter_scope"] == {
        "entity_type": "message",
        "conversation_id": "conversation-one",
    }
    assert first.json()["evaluation_intent"] == "argument_direct_conflict"

    page_one = stage3_client.get("/api/v1/search/feedback", params={"limit": 1})
    assert page_one.status_code == 200
    assert len(page_one.json()["items"]) == 1
    assert page_one.json()["page"]["next_cursor"]
    page_two = stage3_client.get(
        "/api/v1/search/feedback",
        params={"limit": 1, "cursor": page_one.json()["page"]["next_cursor"]},
    )
    assert page_two.status_code == 200
    assert len(page_two.json()["items"]) == 1
    assert page_two.json()["items"][0]["id"] != page_one.json()["items"][0]["id"]
    unsafe_scope = stage3_client.post(
        "/api/v1/search/feedback",
        json={
            **base,
            "filter_scope": {
                "entity_type": "message",
                "raw_sql": "conversation_id = '*'",
            },
        },
    )
    assert unsafe_scope.status_code == 422

    invalid_mode = stage3_client.post(
        "/api/v1/search/feedback",
        json={
            **base,
            "search_mode": "keyword",
        },
    )
    assert invalid_mode.status_code == 422
    invalid_split = stage3_client.patch(
        f"/api/v1/search/feedback/{first.json()['id']}",
        json={"evaluation_split": "validation"},
    )
    assert invalid_split.status_code == 422
    cleared_intent = stage3_client.patch(
        f"/api/v1/search/feedback/{first.json()['id']}",
        json={"evaluation_intent": None, "evaluation_split": "validation"},
    )
    assert cleared_intent.status_code == 200
    assert cleared_intent.json()["evaluation_intent"] is None


def test_development_calibration_feedback_requires_validation_semantic_default_scope(
    stage3_client: TestClient,
) -> None:
    base = {
        "query": "development calibration fixture",
        "search_mode": "semantic",
        "result_id": "window:development-calibration",
        "label": "relevant",
        "evaluation_split": "validation",
        "evaluation_intent": "argument_direct_conflict_development",
    }
    created = stage3_client.post(
        "/api/v1/search/feedback",
        json={**base, "filter_scope": {"entity_type": "all"}},
    )
    assert created.status_code == 201, created.text
    assert created.json()["evaluation_intent"] == "argument_direct_conflict_development"
    assert created.json()["filter_scope"] == {}

    invalid_mode = stage3_client.post(
        "/api/v1/search/feedback",
        json={**base, "result_id": "window:development-hybrid", "search_mode": "hybrid"},
    )
    assert invalid_mode.status_code == 422
    invalid_split = stage3_client.post(
        "/api/v1/search/feedback",
        json={**base, "result_id": "window:development-test", "evaluation_split": "test"},
    )
    assert invalid_split.status_code == 422
    invalid_scope = stage3_client.post(
        "/api/v1/search/feedback",
        json={
            **base,
            "result_id": "window:development-scoped",
            "filter_scope": {"entity_type": "message"},
        },
    )
    assert invalid_scope.status_code == 422
    unknown_intent = stage3_client.post(
        "/api/v1/search/feedback",
        json={**base, "evaluation_intent": "private_experiment"},
    )
    assert unknown_intent.status_code == 422

    invalid_patch = stage3_client.patch(
        f"/api/v1/search/feedback/{created.json()['id']}",
        json={"evaluation_split": "test"},
    )
    assert invalid_patch.status_code == 422
    promoted_to_held_out = stage3_client.patch(
        f"/api/v1/search/feedback/{created.json()['id']}",
        json={
            "evaluation_split": "test",
            "evaluation_intent": "argument_direct_conflict",
        },
    )
    assert promoted_to_held_out.status_code == 200, promoted_to_held_out.text
    assert promoted_to_held_out.json()["evaluation_intent"] == "argument_direct_conflict"
    assert promoted_to_held_out.json()["evaluation_split"] == "test"


def test_saved_search_feedback_and_cloud_gated_cited_rag(
    stage3_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = stage3_client.post(
        "/api/v1/saved-searches",
        json={"name": "Fixture search", "query": {"q": "fixture", "mode": "hybrid"}},
    )
    assert created.status_code == 201
    saved_id = created.json()["id"]
    duplicate = stage3_client.post(
        "/api/v1/saved-searches",
        json={"name": "Fixture search", "query": {"q": "other"}},
    )
    assert duplicate.status_code == 409
    updated = stage3_client.patch(
        f"/api/v1/saved-searches/{saved_id}",
        json={"name": "Renamed fixture search"},
    )
    assert updated.status_code == 200
    assert updated.json()["name"] == "Renamed fixture search"
    assert len(stage3_client.get("/api/v1/saved-searches").json()["items"]) == 1

    feedback = stage3_client.post(
        "/api/v1/search/feedback",
        json={
            "query": "fixture",
            "search_mode": "hybrid",
            "result_id": "window:fixture",
            "label": "relevant",
            "evaluation_split": "test",
        },
    )
    assert feedback.status_code == 201
    assert feedback.json()["evaluation_split"] == "test"
    feedback_id = feedback.json()["id"]
    stale_status = stage3_client.get("/api/v1/search/status")
    assert stale_status.status_code == 200
    assert stale_status.json()["acceptance"]["passed"] is False
    assert (
        stale_status.json()["acceptance"]["held_out_feedback"]["feedback_revision_current"] is False
    )
    assert (
        stale_status.json()["active"]["audit"]["accepted_semantic_ir_evaluation"][
            "production_acceptance_passed"
        ]
        is False
    )
    replacement = stage3_client.post(
        "/api/v1/search/feedback",
        json={
            "query": "fixture",
            "search_mode": "hybrid",
            "result_id": "window:fixture",
            "label": "not_relevant",
            "notes": "corrected judgment",
        },
    )
    assert replacement.status_code == 201
    assert replacement.json()["id"] == feedback_id
    assert replacement.json()["updated"] is True
    listed_feedback = stage3_client.get(
        "/api/v1/search/feedback", params={"query": "fixture", "search_mode": "hybrid"}
    )
    assert listed_feedback.status_code == 200
    assert len(listed_feedback.json()["items"]) == 1
    assert listed_feedback.json()["items"][0]["label"] == "not_relevant"
    patched_feedback = stage3_client.patch(
        f"/api/v1/search/feedback/{feedback_id}",
        json={"label": "uncertain", "evaluation_split": "validation"},
    )
    assert patched_feedback.status_code == 200
    assert patched_feedback.json()["label"] == "uncertain"

    disabled = stage3_client.post(
        "/api/v1/rag/answer", json={"question": "What happened in the fixture?"}
    )
    assert disabled.status_code == 409

    monkeypatch.setenv("OPENAI_API_KEY", "fixture-key-never-sent-to-response-input")
    enabled = stage3_client.patch("/api/v1/settings", json={"cloud_enabled": True})
    assert enabled.status_code == 200
    monkeypatch.delenv("OPENAI_API_KEY")
    client_called = False

    def forbidden_client(_key: str) -> FakeOpenAI:
        nonlocal client_called
        client_called = True
        return FakeOpenAI(None)

    stage3_client.app.state.openai_client_factory = forbidden_client
    missing_key = stage3_client.post(
        "/api/v1/rag/answer", json={"question": "What happened in the fixture?"}
    )
    assert missing_key.status_code == 409
    assert client_called is False
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-key-never-sent-to-response-input")
    retrieval = stage3_client.get(
        "/api/v1/search",
        params={
            "q": "What happened in the fixture?",
            "mode": "hybrid",
            "entity_type": "activity",
            "limit": 8,
        },
    ).json()
    citation_id = retrieval["items"][0]["id"]
    fake = FakeOpenAI(citation_id)
    stage3_client.app.state.openai_client_factory = lambda _key: fake

    answered = stage3_client.post(
        "/api/v1/rag/answer",
        json={
            "question": "What happened in the fixture?",
            "entity_type": "activity",
            "top_k": 8,
        },
    )
    assert answered.status_code == 200, answered.text
    assert answered.json()["citation_ids"] == [citation_id]
    assert answered.json()["audit_id"]
    assert {source["entity_type"] for source in answered.json()["sources"]} == {"activity"}
    assert fake.responses.calls[0]["model"] == "gpt-5.6-sol"
    assert fake.responses.calls[0]["store"] is False
    assert fake.responses.calls[0]["tools"] == []
    assert fake.responses.calls[0]["text_format"] is RagStructuredAnswer
    assert fake.responses.calls[0]["safety_identifier"].startswith("mae_")
    assert len(fake.responses.calls[0]["safety_identifier"]) <= 64
    assert "fixture-key-never-sent" not in fake.responses.calls[0]["input"]

    with Session(stage3_client.app.state.engine) as session:
        audit = session.get(RagAudit, answered.json()["audit_id"])
        assert audit is not None
        assert audit.status == "completed"
        assert audit.remote_response_id == "resp_fixture"
        assert audit.citation_ids_json == {"ids": [citation_id]}
        assert audit.usage_json == {"input_tokens": 42, "output_tokens": 12}
        assert audit.query_text == ""
        assert audit.error_text is None

    invalid_fake = FakeOpenAI("window:not-retrieved")
    stage3_client.app.state.openai_client_factory = lambda _key: invalid_fake
    invalid = stage3_client.post(
        "/api/v1/rag/answer", json={"question": "What happened in the fixture?"}
    )
    assert invalid.status_code == 502
    no_citation_fake = FakeOpenAI(None)
    stage3_client.app.state.openai_client_factory = lambda _key: no_citation_fake
    no_citation = stage3_client.post(
        "/api/v1/rag/answer", json={"question": "What happened in the fixture?"}
    )
    assert no_citation.status_code == 502

    private_provider_text = "provider echoed a private archive excerpt and authorization header"

    class EchoingResponses:
        def parse(self, **_kwargs: Any) -> Any:
            raise RuntimeError(private_provider_text)

    stage3_client.app.state.openai_client_factory = lambda _key: SimpleNamespace(
        responses=EchoingResponses()
    )
    provider_failure = stage3_client.post(
        "/api/v1/rag/answer", json={"question": "Private fixture question"}
    )
    assert provider_failure.status_code == 502
    assert provider_failure.json()["detail"] == "Cloud RAG request failed"
    assert private_provider_text not in provider_failure.text

    with Session(stage3_client.app.state.engine) as session:
        failed_audits = list(session.scalars(select(RagAudit).where(RagAudit.status == "failed")))
        assert len(failed_audits) == 3
        assert {audit.error_text for audit in failed_audits} == {
            "RagCitationError",
            "RuntimeError",
        }
        assert all(audit.query_text == "" for audit in failed_audits)
        assert all(private_provider_text not in (audit.error_text or "") for audit in failed_audits)

    audits = stage3_client.get("/api/v1/rag/audits", params={"status": "completed"})
    assert audits.status_code == 200
    assert [item["id"] for item in audits.json()["items"]] == [answered.json()["audit_id"]]
    assert audits.json()["items"][0]["citation_ids"] == [citation_id]
    assert "query" not in audits.json()["items"][0]

    deleted = stage3_client.delete(f"/api/v1/saved-searches/{saved_id}")
    assert deleted.status_code == 204
    deleted_feedback = stage3_client.delete(f"/api/v1/search/feedback/{feedback_id}")
    assert deleted_feedback.status_code == 204
    assert stage3_client.get("/api/v1/search/feedback").json()["items"] == []


def test_rag_audit_cursor_is_keyset_continuous_and_scope_bound(
    stage3_client: TestClient,
) -> None:
    def audit(identifier: str, created_at: datetime) -> RagAudit:
        return RagAudit(
            id=identifier,
            status="completed",
            model="fixture-rag-model",
            query_sha256="a" * 64,
            query_text="",
            search_mode="hybrid",
            retrieved_ids_json={"ids": []},
            citation_ids_json={"ids": []},
            excerpt_hashes_json={"hashes": []},
            usage_json={},
            created_at=created_at,
        )

    with Session(stage3_client.app.state.engine) as session, session.begin():
        session.add_all(
            [
                audit("rag-audit-old-1", datetime(2026, 1, 2, tzinfo=UTC)),
                audit("rag-audit-old-2", datetime(2026, 1, 2, tzinfo=UTC)),
                audit("rag-audit-old-3", datetime(2026, 1, 3, tzinfo=UTC)),
            ]
        )

    first = stage3_client.get("/api/v1/rag/audits", params={"status": "completed", "limit": 1})
    assert first.status_code == 200, first.text
    first_payload = first.json()
    assert [item["id"] for item in first_payload["items"]] == ["rag-audit-old-3"]
    cursor = first_payload["page"]["next_cursor"]
    assert cursor

    with Session(stage3_client.app.state.engine) as session, session.begin():
        session.add(audit("rag-audit-newer-insert", datetime(2026, 1, 4, tzinfo=UTC)))

    seen = ["rag-audit-old-3"]
    while cursor:
        page = stage3_client.get(
            "/api/v1/rag/audits",
            params={"status": "completed", "limit": 1, "cursor": cursor},
        )
        assert page.status_code == 200, page.text
        payload = page.json()
        seen.extend(item["id"] for item in payload["items"])
        cursor = payload["page"]["next_cursor"]
    assert seen == ["rag-audit-old-3", "rag-audit-old-2", "rag-audit-old-1"]
    assert "rag-audit-newer-insert" not in seen

    refreshed = stage3_client.get("/api/v1/rag/audits", params={"status": "completed", "limit": 1})
    assert refreshed.status_code == 200, refreshed.text
    assert refreshed.json()["items"][0]["id"] == "rag-audit-newer-insert"
    assert (
        stage3_client.get(
            "/api/v1/rag/audits",
            params={"status": "failed", "limit": 1, "cursor": first_payload["page"]["next_cursor"]},
        ).status_code
        == 422
    )
    assert (
        stage3_client.get(
            "/api/v1/rag/audits",
            params={
                "status": "completed",
                "limit": 2,
                "cursor": first_payload["page"]["next_cursor"],
            },
        ).status_code
        == 422
    )


def test_lexical_phrase_prefix_case_filters_and_match_centered_excerpt(
    stage3_client: TestClient,
) -> None:
    phrase = stage3_client.get(
        "/api/v1/search",
        params={"q": '"Exact duplicate"', "mode": "keyword", "entity_type": "message"},
    )
    assert phrase.status_code == 200, phrase.text
    assert phrase.json()["items"]
    assert all("Exact duplicate" in item["excerpt"] for item in phrase.json()["items"])
    assert all(item["score"] >= 0 for item in phrase.json()["items"])

    prefix = stage3_client.get(
        "/api/v1/search",
        params={"q": "transcrib*", "mode": "keyword", "entity_type": "message"},
    )
    assert prefix.status_code == 200, prefix.text
    assert any("Transcribed fixture" in item["excerpt"] for item in prefix.json()["items"])

    group_only = stage3_client.get(
        "/api/v1/search",
        params={
            "q": "fixture",
            "mode": "keyword",
            "entity_type": "activity",
            "source_category": "groups",
        },
    )
    assert group_only.status_code == 200
    assert group_only.json()["items"]
    assert {item["source_category"] for item in group_only.json()["items"]} == {"groups"}

    with Session(stage3_client.app.state.engine) as session:
        target = session.scalar(
            select(Message)
            .where(Message.content_display.is_not(None), Message.deleted_at.is_(None))
            .order_by(Message.id)
        )
        assert target is not None
        target_id = target.id
        target_conversation_id = target.conversation_id
        target_timestamp_ms = target.timestamp_ms
    marker = "NEEDLE_LATE_CASE_SENSITIVE"
    long_text = f"{'ordinary prefix text ' * 180}{marker}{' ordinary suffix text' * 180}"
    with Session(stage3_client.app.state.engine) as session, session.begin():
        session.execute(
            update(Message)
            .where(Message.id == target_id)
            .values(
                content_raw=long_text,
                content_display=long_text,
                content_search=long_text.casefold(),
            )
        )

    exact = stage3_client.get(
        "/api/v1/search",
        params={
            "q": marker,
            "mode": "exact",
            "entity_type": "message",
            "conversation_id": target_conversation_id,
            "from_ms": target_timestamp_ms,
            "to_ms": target_timestamp_ms,
        },
    )
    assert exact.status_code == 200, exact.text
    assert [item["source_id"] for item in exact.json()["items"]] == [target_id]
    assert marker in exact.json()["items"][0]["excerpt"]
    assert exact.json()["items"][0]["excerpt"].startswith("… ")

    wrong_case = stage3_client.get(
        "/api/v1/search",
        params={"q": marker.casefold(), "mode": "exact", "entity_type": "message"},
    )
    assert wrong_case.status_code == 200
    assert wrong_case.json()["items"] == []


def test_async_index_route_uses_injected_backend_invalidates_cursor_and_recovers_jobs(
    stage3_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_page = stage3_client.get(
        "/api/v1/search",
        params={"q": "fixture", "mode": "semantic", "limit": 2},
    )
    assert first_page.status_code == 200
    stale_cursor = first_page.json()["page"]["next_cursor"]
    assert stale_cursor

    built_models: list[str] = []

    def recording_factory(spec: EmbeddingModelSpec) -> FixtureEmbedder:
        built_models.append(spec.model_name)
        return FixtureEmbedder(spec)

    def forbid_default_backend(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("the injected index embedder must prevent a model download")

    monkeypatch.setattr(
        "meta_archive.services.search_index.FastEmbedTextEmbedder",
        forbid_default_backend,
    )
    monkeypatch.setattr(
        search_index_service,
        "_active_incumbent_comparison",
        lambda *_args, **_kwargs: search_index_service._semantic_upgrade_metric_comparison(
            {"recall_at_k": 0.73, "mrr": 0.63, "ndcg_at_k": 0.61},
            {"recall_at_k": 0.70, "mrr": 0.60, "ndcg_at_k": 0.58},
        ),
    )
    stage3_client.app.state.semantic_index_embedder_factory = recording_factory
    created = stage3_client.post(
        "/api/v1/search/index",
        json={"model": "bge-small-en-v1.5", "rebuild_windows": False},
    )
    assert created.status_code == 202, created.text
    job_id = created.json()["job_id"]
    assert created.json()["status"] == "queued"
    assert built_models == ["BAAI/bge-small-en-v1.5"]
    with Session(stage3_client.app.state.engine) as session:
        completed = session.get(Job, job_id)
        assert completed is not None
        assert completed.status == "completed"

    old_cursor = stage3_client.get(
        "/api/v1/search",
        params={
            "q": "fixture",
            "mode": "semantic",
            "limit": 2,
            "cursor": stale_cursor,
        },
    )
    assert old_cursor.status_code == 422

    queued_id = "queued-semantic-fixture"
    with Session(stage3_client.app.state.engine) as session, session.begin():
        session.add(
            Job(
                id=queued_id,
                kind="semantic_index",
                status="queued",
                stage="Semantic index queued",
                progress_current=0,
                checkpoint_json={},
                counters_json={},
            )
        )
    conflict = stage3_client.post(
        "/api/v1/search/index",
        json={"model": "bge-small-en-v1.5", "rebuild_windows": False},
    )
    assert conflict.status_code == 409
    with Session(stage3_client.app.state.engine) as session, session.begin():
        queued = session.get(Job, queued_id)
        assert queued is not None
        session.delete(queued)

    dead_id = "dead-semantic-worker"
    with Session(stage3_client.app.state.engine) as session, session.begin():
        session.add(
            Job(
                id=dead_id,
                kind="semantic_index",
                status="running",
                stage="Embedding search windows",
                progress_current=1,
                checkpoint_json={},
                counters_json={},
            )
        )
    recovered = stage3_client.post(
        "/api/v1/search/index",
        json={"model": "bge-small-en-v1.5", "rebuild_windows": False},
    )
    assert recovered.status_code == 202, recovered.text
    with Session(stage3_client.app.state.engine) as session:
        dead = session.get(Job, dead_id)
        replacement = session.get(Job, recovered.json()["job_id"])
        assert dead is not None
        assert dead.status == "failed"
        assert replacement is not None
        assert replacement.status == "completed"
