from __future__ import annotations

import hashlib
import json
import math
import shutil
import sqlite3
from collections.abc import Iterator, Sequence
from pathlib import Path

import lancedb  # type: ignore[import-untyped]
import pytest
from sqlalchemy import Engine, event, func, select, update
from sqlalchemy.orm import Session

from meta_archive.models import (
    Job,
    Message,
    RelevanceFeedback,
    SearchWindow,
    SearchWindowMessage,
    SemanticIndexConfig,
    SemanticWindowEmbedding,
)
from meta_archive.paths import ArchiveLayout
from meta_archive.services import search as search_service
from meta_archive.services import search_index as search_index_service
from meta_archive.services.activity_importer import import_activities
from meta_archive.services.direct_conflict_ranker import (
    DIRECT_CONFLICT_EVALUATION_INTENT,
    DIRECT_CONFLICT_SCORE_KIND,
)
from meta_archive.services.inventory import scan_archive
from meta_archive.services.message_importer import import_messages
from meta_archive.services.reranking import (
    DEFAULT_RERANKER_SPEC,
    RerankerSpec,
    TransformersCrossEncoderReranker,
)
from meta_archive.services.search import (
    SearchFilters,
    SearchUnavailableError,
    hybrid_search,
    semantic_search,
)
from meta_archive.services.search_index import (
    GAP_MS,
    HELD_OUT_THRESHOLDS,
    MAX_APPROX_TOKENS,
    EmbeddingModelSpec,
    SemanticIndexAcceptanceError,
    SemanticIndexBusyError,
    WindowDraft,
    WindowTurn,
    approximate_token_count,
    build_semantic_index,
    deterministic_message_windows,
    queue_semantic_index_job,
    rebuild_search_windows,
    retrieval_metrics,
    search_window_language_audit,
    semantic_index_lock,
)
from meta_archive.services.semantic_state import (
    SemanticTableIntegrityError,
    invalidate_ready_semantic_indexes,
)

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
    spec = FIXTURE_SPEC

    def __init__(self, *, fail_on_call: int | None = None) -> None:
        self.calls = 0
        self.fail_on_call = fail_on_call

    @staticmethod
    def _vector(text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = [float(digest[index] + 1) for index in range(FIXTURE_SPEC.dimensions)]
        norm = sum(value * value for value in values) ** 0.5
        return [value / norm for value in values]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        if self.fail_on_call == self.calls:
            raise RuntimeError("synthetic embedding interruption")
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class FingerprintedFixtureEmbedder(FixtureEmbedder):
    def __init__(self, fingerprint: str, *, fail_on_call: int | None = None) -> None:
        super().__init__(fail_on_call=fail_on_call)
        self.fingerprint = fingerprint

    def artifact_fingerprint(self) -> str:
        return self.fingerprint


class BrokenQueryFixtureEmbedder(FingerprintedFixtureEmbedder):
    def embed_query(self, _text: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]


class HeldOutFailureFixtureEmbedder(FingerprintedFixtureEmbedder):
    def embed_query(self, text: str) -> list[float]:
        if text.startswith("held-out unrelated query"):
            return [1.0, 0.0, 0.0, 0.0]
        return super().embed_query(text)


class NonFiniteDocumentFixtureEmbedder(FixtureEmbedder):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float("nan"), 0.0, 0.0, 0.0] for _text in texts]


class NonFiniteQueryFixtureEmbedder(FixtureEmbedder):
    def embed_query(self, text: str) -> list[float]:
        return [float("nan"), 0.0, 0.0, 0.0]


class DynamicFixtureEmbedder:
    def __init__(self, spec: EmbeddingModelSpec) -> None:
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

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, list[str]]] = []

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

    def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        self.calls.append((query, list(documents)))
        if self.fail:
            raise RuntimeError("synthetic reranker unavailable")
        return [float(index) for index, _document in enumerate(documents)]


class ExactMatchFixtureReranker(FixtureReranker):
    def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        self.calls.append((query, list(documents)))
        return [1.0 if document == query else 0.0 for document in documents]


def fixture_reranker_factory(_spec: RerankerSpec) -> FixtureReranker:
    return FixtureReranker()


@pytest.fixture
def stage3_catalog(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> Iterator[tuple[ArchiveLayout, Engine]]:
    shutil.copytree(MESSAGE_FIXTURES, archive_layout.root, dirs_exist_ok=True)
    shutil.copytree(ACTIVITY_FIXTURES, archive_layout.root / "activity-package")
    scan_archive(archive_layout, catalog_engine, hash_files=True)
    import_messages(archive_layout, catalog_engine)
    import_activities(archive_layout, catalog_engine)
    yield archive_layout, catalog_engine


def _turn(index: int, *, timestamp_ms: int | None = None, text: str | None = None) -> WindowTurn:
    return WindowTurn(
        id=f"message-{index:03}",
        timestamp_ms=timestamp_ms if timestamp_ms is not None else index * 60_000,
        sender_display="Alex" if index % 2 else "Owner",
        sender_search="alex" if index % 2 else "owner",
        text_display=text or f"fixture turn {index}",
        segment_id="segment-fixture",
        source_category="inbox",
    )


def test_retrieval_metrics_report_recall_mrr_and_ndcg() -> None:
    metrics = retrieval_metrics(
        [["a", "noise", "b"], ["noise", "c", "other"]],
        [{"a", "b"}, {"c", "d"}],
        k=3,
    )

    discount_at_two = 1.0 / math.log2(3)
    ideal_two = 1.0 + discount_at_two
    expected_first_ndcg = 1.5 / ideal_two
    expected_second_ndcg = discount_at_two / ideal_two

    assert metrics["query_count"] == 2
    assert metrics["k"] == 3
    assert metrics["recall_at_k"] == pytest.approx(0.75)
    assert metrics["mrr"] == pytest.approx(0.75)
    assert metrics["ndcg_at_k"] == pytest.approx((expected_first_ndcg + expected_second_ndcg) / 2)


def test_literal_ranker_runs_cross_encoder_only_for_exact_margin_ties() -> None:
    class FixtureAudit:
        @staticmethod
        def public_metadata() -> dict[str, object]:
            return {"version": "fixture-direct-ranker"}

    class FixtureDirectRanker:
        audit = FixtureAudit()

        @staticmethod
        def score(_vectors: Sequence[Sequence[float]]) -> list[float]:
            return [3.0, 2.0, 2.0, 1.0]

    reranker = FixtureReranker()
    vector = [1.0, 0.0, 0.0, 0.0]
    ranking = search_service._rank_semantic_candidates(
        "argument",
        [
            ("a", 0.9, vector),
            ("b", 0.8, vector),
            ("c", 0.7, vector),
            ("d", 0.6, vector),
        ],
        {key: f"document {key}" for key in ("a", "b", "c", "d")},
        reranker,
        direct_conflict_ranker=FixtureDirectRanker(),  # type: ignore[arg-type]
    )

    assert [row[0] for row in ranking.rows] == ["a", "c", "b", "d"]
    assert reranker.calls == [("argument", ["document b", "document c"])]
    assert ranking.provenance["cross_encoder_tie_candidate_count"] == 2


def test_known_item_evaluator_bypasses_approximate_vector_index() -> None:
    class FakeVectorQuery:
        def __init__(self) -> None:
            self.bypassed = False

        def distance_type(self, distance: str) -> FakeVectorQuery:
            assert distance == "cosine"
            return self

        def bypass_vector_index(self) -> FakeVectorQuery:
            self.bypassed = True
            return self

        def limit(self, value: int) -> FakeVectorQuery:
            assert value == 10
            return self

        def select(self, columns: list[str]) -> FakeVectorQuery:
            assert columns == ["id", "text", "_distance"]
            return self

        def to_list(self) -> list[dict[str, object]]:
            assert self.bypassed is True
            return [{"id": "window-fixture", "text": "fixture", "_distance": 0.0}]

    query = FakeVectorQuery()

    class FakeTable:
        def search(self, vector: list[float], *, vector_column_name: str) -> FakeVectorQuery:
            assert len(vector) == FIXTURE_SPEC.dimensions
            assert vector_column_name == "vector"
            return query

    rows = search_index_service._retrieved_window_rows(
        FakeTable(),
        FixtureEmbedder(),
        "fixture",
        k=10,
    )

    assert rows == [{"id": "window-fixture", "text": "fixture", "_distance": 0.0}]


def test_message_windows_are_deterministic_bounded_and_provenance_complete() -> None:
    turns = [_turn(index) for index in range(20)]
    first = deterministic_message_windows("conversation-fixture", turns)
    second = deterministic_message_windows("conversation-fixture", turns)

    assert first == second
    assert [len(window.message_ids) for window in first] == [12, 10]
    assert first[0].message_ids[-2:] == first[1].message_ids[:2]
    assert all(window.approx_token_count <= MAX_APPROX_TOKENS for window in first)
    assert first[0].first_timestamp_ms == turns[0].timestamp_ms
    assert first[-1].last_timestamp_ms == turns[-1].timestamp_ms

    separated = turns[:3] + [
        _turn(index + 100, timestamp_ms=turns[2].timestamp_ms + GAP_MS + 1) for index in range(4)
    ]
    split = deterministic_message_windows("conversation-fixture", separated)
    assert [len(window.message_ids) for window in split] == [3, 4]

    long_turn = _turn(999, text=" ".join(f"token-{index}" for index in range(700)))
    long_windows = deterministic_message_windows("conversation-fixture", [long_turn])
    assert len(long_windows) > 1
    assert all(window.message_ids == (long_turn.id,) for window in long_windows)
    assert all(
        approximate_token_count(window.text_display) <= MAX_APPROX_TOKENS for window in long_windows
    )
    assert "token-699" in "\n".join(window.text_display for window in long_windows)
    assert [window.chunk_ordinal for window in long_windows] == list(range(len(long_windows)))

    dense_turns = [
        _turn(
            index,
            text=" ".join([*(f"dense_{index}_{token}" for token in range(88)), f"TAIL_{index}"]),
        )
        for index in range(6)
    ]
    dense_windows = deterministic_message_windows("conversation-fixture", dense_turns)
    rendered_dense = "\n".join(window.text_display for window in dense_windows)
    assert len(dense_windows) >= 2
    assert all(f"TAIL_{index}" in rendered_dense for index in range(6))
    assert all(window.approx_token_count <= MAX_APPROX_TOKENS for window in dense_windows)


def test_window_catalog_covers_messages_and_activities_idempotently(
    stage3_catalog: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = stage3_catalog
    first = rebuild_search_windows(layout, engine, generation="fixture-generation-one")
    with Session(engine) as session:
        ids_one = set(session.scalars(select(SearchWindow.id)))
        associations_one = session.scalar(select(func.count(SearchWindowMessage.id))) or 0
        activity_windows = list(
            session.scalars(select(SearchWindow).where(SearchWindow.entity_type == "activity"))
        )
    second = rebuild_search_windows(layout, engine, generation="fixture-generation-two")
    with Session(engine) as session:
        ids_two = set(session.scalars(select(SearchWindow.id)))
        associations_two = session.scalar(select(func.count(SearchWindowMessage.id))) or 0
        windows = list(session.scalars(select(SearchWindow)))

    assert ids_one == ids_two
    assert first.window_count == second.window_count == len(windows)
    assert first.association_count == second.association_count
    assert associations_one == associations_two
    assert activity_windows
    assert any(window.source_category == "groups" for window in activity_windows)
    assert all("fixture-only" not in window.text_display for window in windows)
    assert all(window.approx_token_count <= MAX_APPROX_TOKENS for window in windows)


def test_window_upsert_batches_below_sqlite_variable_floor(
    stage3_catalog: tuple[ArchiveLayout, Engine],
) -> None:
    _layout, engine = stage3_catalog
    with Session(engine) as session:
        seed = session.execute(
            select(
                Message.id,
                Message.conversation_id,
                Message.segment_id,
                Message.timestamp_ms,
            ).limit(1)
        ).one()
    drafts = [
        WindowDraft(
            id=f"{ordinal + 1:064x}",
            conversation_id=seed.conversation_id,
            segment_id=seed.segment_id,
            source_category="fixture",
            message_ids=(seed.id,),
            first_timestamp_ms=seed.timestamp_ms,
            last_timestamp_ms=seed.timestamp_ms,
            text_display=f"bounded window {ordinal}",
            text_search=f"bounded window {ordinal}",
            participants_search="fixture participant",
            approx_token_count=3,
        )
        for ordinal in range(81)
    ]

    with engine.connect() as connection:
        raw_connection = connection.connection.driver_connection
        prior_limit = raw_connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
        try:
            with Session(bind=connection) as session, session.begin():
                associations = search_index_service._upsert_window_batch(
                    session,
                    drafts=drafts,
                    generation="bounded-variable-fixture",
                )
        finally:
            raw_connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, prior_limit)

    assert associations == len(drafts)
    with Session(engine) as session:
        assert session.scalar(
            select(func.count(SearchWindow.id)).where(
                SearchWindow.build_generation == "bounded-variable-fixture"
            )
        ) == len(drafts)
    assert engine.hide_parameters is True


def test_requested_semantic_job_records_preflight_failure(
    stage3_catalog: tuple[ArchiveLayout, Engine],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, engine = stage3_catalog
    job_id = queue_semantic_index_job(
        engine,
        model_alias=FIXTURE_SPEC.alias,
        rebuild_windows=True,
    )

    def fail_preflight(_engine: Engine, *, generation: str | None = None) -> None:
        del generation
        raise RuntimeError("synthetic safe preflight failure")

    monkeypatch.setattr(search_index_service, "_rebuild_search_windows_locked", fail_preflight)
    with pytest.raises(RuntimeError, match="synthetic safe preflight failure"):
        build_semantic_index(
            layout,
            engine,
            embedder=FixtureEmbedder(),
            job_id=job_id,
        )

    with Session(engine) as session:
        failed = session.get(Job, job_id)
        assert failed is not None
        assert failed.status == "failed"
        assert failed.stage == "Semantic index failed during preflight"
        assert failed.error_text == "RuntimeError: synthetic safe preflight failure"
        assert failed.finished_at is not None


def test_queued_semantic_job_cannot_revive_after_corpus_invalidation(
    stage3_catalog: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = stage3_catalog
    job_id = queue_semantic_index_job(
        engine,
        model_alias=FIXTURE_SPEC.alias,
        rebuild_windows=True,
    )
    with Session(engine) as session, session.begin():
        invalidate_ready_semantic_indexes(
            session,
            reason="synthetic source mutation before worker start",
            source_kind="fixture",
            source_id="fixture-source",
        )

    with pytest.raises(RuntimeError, match="invalidated before it started"):
        build_semantic_index(
            layout,
            engine,
            embedder=FixtureEmbedder(),
            job_id=job_id,
        )

    with Session(engine) as session:
        job = session.get(Job, job_id)
        assert job is not None
        assert job.status == "failed"
        assert job.stage == "Semantic index invalidated by searchable corpus change"


def test_running_semantic_job_cannot_revive_after_preflight_invalidation(
    stage3_catalog: tuple[ArchiveLayout, Engine],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, engine = stage3_catalog
    job_id = queue_semantic_index_job(
        engine,
        model_alias=FIXTURE_SPEC.alias,
        rebuild_windows=True,
    )
    original_rebuild = search_index_service._rebuild_search_windows_locked

    def rebuild_then_invalidate(
        target_engine: Engine,
        *,
        generation: str | None = None,
    ) -> object:
        summary = original_rebuild(target_engine, generation=generation)
        with Session(target_engine) as session, session.begin():
            invalidate_ready_semantic_indexes(
                session,
                reason="synthetic source mutation during preflight",
                source_kind="fixture",
                source_id="fixture-source",
            )
        return summary

    monkeypatch.setattr(
        search_index_service,
        "_rebuild_search_windows_locked",
        rebuild_then_invalidate,
    )
    with pytest.raises(RuntimeError, match="invalidated during preflight"):
        build_semantic_index(
            layout,
            engine,
            embedder=FixtureEmbedder(),
            job_id=job_id,
        )

    with Session(engine) as session:
        job = session.get(Job, job_id)
        assert job is not None
        assert job.status == "failed"
        assert job.stage == "Semantic index invalidated by searchable corpus change"


def test_semantic_index_is_audited_idempotent_and_resumable(
    stage3_catalog: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = stage3_catalog
    interrupted = FixtureEmbedder(fail_on_call=2)
    with pytest.raises(RuntimeError, match="synthetic embedding interruption"):
        build_semantic_index(layout, engine, embedder=interrupted, batch_size=2)

    with Session(engine) as session:
        failed_job = session.scalar(
            select(Job).where(Job.kind == "semantic_index", Job.status == "failed")
        )
        assert failed_job is not None
        completed_before_resume = (
            session.scalar(
                select(func.count(SemanticWindowEmbedding.window_id)).where(
                    SemanticWindowEmbedding.config_id == failed_job.checkpoint_json["config_hash"]
                )
            )
            or 0
        )
    assert completed_before_resume == 2

    pending_statements: list[str] = []

    def capture_pending_sql(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        if (
            "FROM search_windows" in statement
            and "search_windows.embedding_config_hash" in statement
            and "ORDER BY search_windows.id" in statement
            and "LIMIT" in statement
        ):
            pending_statements.append(statement)

    event.listen(engine, "before_cursor_execute", capture_pending_sql)
    try:
        resumed = build_semantic_index(layout, engine, embedder=FixtureEmbedder(), batch_size=2)
    finally:
        event.remove(engine, "before_cursor_execute", capture_pending_sql)
    assert resumed.resumed is True
    assert resumed.job_id == failed_job.id
    assert resumed.embedded_count == resumed.window_count - completed_before_resume
    assert len(pending_statements) > 1
    assert all("search_windows.id > ?" in statement for statement in pending_statements)

    unchanged = build_semantic_index(layout, engine, embedder=FixtureEmbedder(), batch_size=2)
    assert unchanged.config_hash == resumed.config_hash
    assert unchanged.vector_table == resumed.vector_table
    assert unchanged.embedded_count == 0
    assert unchanged.window_count == resumed.window_count

    with Session(engine) as session:
        active = session.scalar(
            select(SemanticIndexConfig).where(SemanticIndexConfig.active.is_(True))
        )
        assert active is not None
        assert active.id == resumed.config_hash
        assert active.model_name == FIXTURE_SPEC.model_name
        assert active.model_revision == FIXTURE_SPEC.model_revision
        assert active.audit_json["window_policy"] == {
            "minimum_turns": 6,
            "maximum_turns": 12,
            "overlap_turns": 2,
            "gap_minutes": 30,
            "approximate_token_cap": 450,
            "version": "message-window-v1",
        }
        assert active.audit_json["embedding_implementation"] == (
            search_index_service.embedding_implementation_identity(FIXTURE_SPEC)
        )
        evaluation = active.audit_json["semantic_ir_evaluation"]
        assert evaluation == active.audit_json["accepted_semantic_ir_evaluation"]
        assert evaluation["passed"] is True
        assert evaluation["candidate"]["query_engine"] == "lancedb-exact-vector-search"
        assert evaluation["automatic_known_item"]["status"] == "passed"
        assert evaluation["automatic_known_item"]["metrics"]["recall_at_k"] == 1.0
        assert evaluation["automatic_known_item"]["metrics"]["mrr"] == 1.0
        assert evaluation["automatic_known_item"]["metrics"]["ndcg_at_k"] == 1.0
        assert evaluation["held_out_feedback"]["status"] == "not_available"
        assert evaluation["automatic_known_item"]["sampling"]["query_details_persisted"] is False
        assert "queries" not in evaluation["automatic_known_item"]
        assert "queries" not in evaluation["held_out_feedback"]

    vector_database = lancedb.connect(str(layout.vectors))
    vector_database.drop_table(unchanged.vector_table)
    with pytest.raises(RuntimeError, match="synthetic embedding interruption"):
        build_semantic_index(
            layout,
            engine,
            embedder=FixtureEmbedder(fail_on_call=1),
            batch_size=2,
        )
    with Session(engine) as session:
        unavailable_repair = session.get(SemanticIndexConfig, unchanged.config_hash)
        assert unavailable_repair is not None
        assert unavailable_repair.active is False
        assert unavailable_repair.status == "failed"
    recovered = build_semantic_index(layout, engine, embedder=FixtureEmbedder(), batch_size=2)
    assert recovered.embedded_count == recovered.window_count
    assert vector_database.open_table(recovered.vector_table).count_rows() == recovered.window_count

    table = vector_database.open_table(recovered.vector_table)
    removed_id = table.search().select(["id"]).limit(1).to_list()[0]["id"]
    table.delete(f"id = '{removed_id}'")
    partial_recovery = build_semantic_index(
        layout, engine, embedder=FixtureEmbedder(), batch_size=2
    )
    assert partial_recovery.embedded_count == partial_recovery.window_count
    assert vector_database.open_table(recovered.vector_table).count_rows() == recovered.window_count

    table = vector_database.open_table(recovered.vector_table)
    duplicate = table.search().limit(1).to_list()[0]
    duplicate.pop("_distance", None)
    table.add([duplicate])
    assert table.count_rows() == recovered.window_count + 1
    duplicate_recovery = build_semantic_index(
        layout, engine, embedder=FixtureEmbedder(), batch_size=2
    )
    assert duplicate_recovery.embedded_count == duplicate_recovery.window_count
    assert vector_database.open_table(recovered.vector_table).count_rows() == recovered.window_count


@pytest.mark.parametrize("corruption", ["zero-vector", "metadata"])
def test_semantic_reuse_fails_closed_on_same_id_lance_content_corruption(
    stage3_catalog: tuple[ArchiveLayout, Engine],
    corruption: str,
) -> None:
    layout, engine = stage3_catalog
    built = build_semantic_index(layout, engine, embedder=FixtureEmbedder(), batch_size=4)
    table = lancedb.connect(str(layout.vectors)).open_table(built.vector_table)
    identifier = str(table.search().select(["id"]).limit(1).to_list()[0]["id"])
    if corruption == "zero-vector":
        table.update(
            f"id = '{identifier}'",
            values={"vector": [0.0] * FIXTURE_SPEC.dimensions},
        )
    else:
        table.update(
            f"id = '{identifier}'",
            values={"entity_type": "corrupted-fixture-metadata"},
        )

    with pytest.raises(
        SemanticTableIntegrityError,
        match="invalid vectors or retrieval metadata",
    ):
        build_semantic_index(
            layout,
            engine,
            embedder=FixtureEmbedder(),
            rebuild_windows=False,
            batch_size=4,
        )

    with Session(engine) as session:
        config = session.get(SemanticIndexConfig, built.config_hash)
        failed_job = session.scalar(
            select(Job)
            .where(Job.kind == "semantic_index", Job.status == "failed")
            .order_by(Job.created_at.desc())
        )
    assert config is not None
    assert config.active is False
    assert config.status == "failed"
    assert "semantic_integrity_failure" in config.audit_json
    assert failed_job is not None


def test_semantic_index_evaluates_held_out_feedback_separately(
    stage3_catalog: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = stage3_catalog
    rebuild_search_windows(layout, engine, generation="held-out-evaluation-fixture")
    with Session(engine) as session:
        windows = list(session.scalars(select(SearchWindow).order_by(SearchWindow.id)))
    distinct_windows: list[SearchWindow] = []
    seen_text: set[str] = set()
    for window in windows:
        if window.text_display in seen_text:
            continue
        distinct_windows.append(window)
        seen_text.add(window.text_display)
        if len(distinct_windows) == 5:
            break
    assert len(distinct_windows) == 5
    with Session(engine) as session, session.begin():
        for index, window in enumerate(distinct_windows):
            scope = {
                "entity_type": (
                    "message" if window.entity_type == "message_window" else window.entity_type
                )
            }
            scope_json = json.dumps(scope, sort_keys=True, separators=(",", ":"))
            session.add(
                RelevanceFeedback(
                    id=f"held-out-feedback-{index}",
                    query_sha256=hashlib.sha256(window.text_display.encode("utf-8")).hexdigest(),
                    query_text=window.text_display,
                    search_mode="semantic" if index % 2 == 0 else "hybrid",
                    filter_scope_sha256=hashlib.sha256(scope_json.encode()).hexdigest(),
                    filter_scope_json=scope,
                    result_id=f"window:{window.id}",
                    label="relevant",
                    evaluation_split="test",
                    evaluation_intent=("argument_direct_conflict" if index == 0 else None),
                )
            )

    held_out_reranker = ExactMatchFixtureReranker()
    summary = build_semantic_index(
        layout,
        engine,
        embedder=FixtureEmbedder(),
        reranker_factory=lambda _spec: held_out_reranker,
        rebuild_windows=False,
        batch_size=4,
    )

    with Session(engine) as session:
        config = session.get(SemanticIndexConfig, summary.config_hash)
        assert config is not None
        held_out = config.audit_json["semantic_ir_evaluation"]["held_out_feedback"]
    assert held_out["benchmark_kind"] == "held_out_user_feedback"
    assert held_out["status"] == "passed"
    assert held_out["used_for_activation"] is True
    assert held_out["metrics"]["query_count"] == 5
    assert held_out["metrics"]["recall_at_k"] == 1.0
    assert held_out["ranking_pipeline"] == "runtime-search-v2"
    assert held_out["reranker"]["model_revision"] == DEFAULT_RERANKER_SPEC.model_revision
    assert len(held_out_reranker.calls) == 5
    assert all(
        len(documents) <= DEFAULT_RERANKER_SPEC.max_candidates
        for _query, documents in held_out_reranker.calls
    )
    assert held_out["query_group_counts_by_mode"] == {"semantic": 3, "hybrid": 2}
    assert held_out["query_details_persisted"] is False
    assert held_out["feedback_revision_stable_during_evaluation"] is True
    assert "queries" not in held_out
    argument_gate = held_out["argument_direct_conflict"]
    assert argument_gate["status"] == "insufficient_judgments"
    assert argument_gate["passed"] is False
    assert argument_gate["query_group_count"] == 1
    assert argument_gate["judgment_count"] == 1
    assert config.audit_json["semantic_ir_evaluation"]["production_acceptance_passed"] is False


def test_tagged_argument_group_requires_substantive_judgments_and_runtime_relevance(
    stage3_catalog: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = stage3_catalog
    rebuild_search_windows(layout, engine, generation="argument-evaluation-fixture")
    with Session(engine) as session:
        windows = list(session.scalars(select(SearchWindow).order_by(SearchWindow.id)))
    assert len(windows) >= 14
    argument_query = "private flagship query fixture"
    argument_targets = windows[:2]
    with Session(engine) as session, session.begin():
        for index, window in enumerate(windows[:10]):
            session.add(
                RelevanceFeedback(
                    id=f"argument-feedback-{index}",
                    query_sha256=hashlib.sha256(argument_query.encode()).hexdigest(),
                    query_text=argument_query,
                    search_mode="semantic",
                    result_id=f"window:{window.id}",
                    label="relevant" if index < 2 else "not_relevant",
                    evaluation_split="test",
                    evaluation_intent="argument_direct_conflict",
                )
            )
        for index, window in enumerate(windows[10:14]):
            session.add(
                RelevanceFeedback(
                    id=f"supporting-feedback-{index}",
                    query_sha256=hashlib.sha256(window.text_display.encode()).hexdigest(),
                    query_text=window.text_display,
                    search_mode="hybrid" if index % 2 else "semantic",
                    result_id=f"window:{window.id}",
                    label="relevant",
                    evaluation_split="test",
                )
            )

    target_texts = {window.text_display for window in argument_targets}

    class ArgumentFixtureReranker(ExactMatchFixtureReranker):
        def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
            self.calls.append((query, list(documents)))
            if query == argument_query:
                return [2.0 if document in target_texts else 0.0 for document in documents]
            return [1.0 if document == query else 0.0 for document in documents]

    summary = build_semantic_index(
        layout,
        engine,
        embedder=FixtureEmbedder(),
        reranker_factory=lambda _spec: ArgumentFixtureReranker(),
        rebuild_windows=False,
        batch_size=4,
    )
    with Session(engine) as session:
        config = session.get(SemanticIndexConfig, summary.config_hash)
        assert config is not None
        evaluation = config.audit_json["semantic_ir_evaluation"]
    argument_gate = evaluation["held_out_feedback"]["argument_direct_conflict"]
    assert evaluation["production_acceptance_passed"] is False
    assert argument_gate["status"] == "passed"
    assert argument_gate["passed"] is True
    assert argument_gate["required_for_production_acceptance"] is False
    assert argument_gate["query_group_count"] == 1
    assert argument_gate["judgment_count"] == 10
    assert argument_gate["relevant_judgment_count"] == 2
    assert argument_gate["not_relevant_judgment_count"] == 8
    assert argument_gate["minimum_judgment_count_per_query_group"] == 10
    assert argument_gate["all_query_groups_meet_judgment_minimums"] is True
    assert argument_gate["all_query_groups_meet_metric_thresholds"] is True
    assert argument_gate["non_window_judgment_count"] == 0
    assert argument_gate["target_not_in_candidate_judgment_count"] == 0
    serialized = json.dumps(evaluation, sort_keys=True)
    assert argument_query not in serialized
    assert "window:" not in serialized
    assert argument_targets[0].id not in serialized

    with Session(engine) as session, session.begin():
        stale_positive = session.get(RelevanceFeedback, "argument-feedback-0")
        non_window_negative = session.get(RelevanceFeedback, "argument-feedback-9")
        assert stale_positive is not None and non_window_negative is not None
        stale_positive.result_id = "window:missing-private-fixture"
        non_window_negative.result_id = "message:non-window-private-fixture"

    rebuilt = build_semantic_index(
        layout,
        engine,
        embedder=FixtureEmbedder(),
        reranker_factory=lambda _spec: ArgumentFixtureReranker(),
        rebuild_windows=False,
        batch_size=4,
    )
    with Session(engine) as session:
        config = session.get(SemanticIndexConfig, rebuilt.config_hash)
        assert config is not None
        reevaluation = config.audit_json["semantic_ir_evaluation"]
    invalid_gate = reevaluation["held_out_feedback"]["argument_direct_conflict"]
    assert reevaluation["production_acceptance_passed"] is False
    assert invalid_gate["status"] == "incomplete_or_inconsistent"
    assert invalid_gate["passed"] is False
    assert invalid_gate["reviewed_judgment_count"] == 10
    assert invalid_gate["judgment_count"] == 8
    assert invalid_gate["relevant_judgment_count"] == 1
    assert invalid_gate["non_window_judgment_count"] == 1
    assert invalid_gate["target_not_in_candidate_judgment_count"] == 1


def test_literal_direct_conflict_calibration_drives_shared_runtime_and_fails_stale(
    stage3_catalog: tuple[ArchiveLayout, Engine],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, engine = stage3_catalog
    rebuild_search_windows(layout, engine, generation="direct-conflict-development-fixture")
    with Session(engine) as session:
        windows = list(session.scalars(select(SearchWindow).order_by(SearchWindow.id)))

    selected: list[SearchWindow] = []
    seen_groups: set[str] = set()
    for window in windows:
        group = window.conversation_id or f"window:{window.id}"
        if group in seen_groups:
            continue
        selected.append(window)
        seen_groups.add(group)
    assert len(selected) >= 20
    selected = selected[:20]
    by_feature = sorted(
        selected,
        key=lambda window: FixtureEmbedder._vector(window.text_display)[0],
    )
    relevant_ids = {window.id for window in by_feature[-10:]}
    default_scope_sha256 = hashlib.sha256(b"{}").hexdigest()
    literal_digest = hashlib.sha256(b"argument").hexdigest()
    literal_regression_note = "fixture-literal-regression"
    monkeypatch.setattr(
        search_index_service,
        "DIRECT_CONFLICT_LITERAL_REGRESSION_NOTES_SHA256",
        hashlib.sha256(literal_regression_note.encode()).hexdigest(),
    )
    monkeypatch.setattr(
        search_index_service,
        "DIRECT_CONFLICT_LITERAL_REGRESSION_RELEVANT_COUNT",
        10,
    )
    monkeypatch.setattr(
        search_index_service,
        "DIRECT_CONFLICT_LITERAL_REGRESSION_NOT_RELEVANT_COUNT",
        10,
    )
    distinct_held_out: list[SearchWindow] = []
    seen_text: set[str] = set()
    for window in windows:
        if window.text_display in seen_text:
            continue
        distinct_held_out.append(window)
        seen_text.add(window.text_display)
        if len(distinct_held_out) == 5:
            break
    assert len(distinct_held_out) == 5

    with Session(engine) as session, session.begin():
        for index, window in enumerate(selected):
            session.add(
                RelevanceFeedback(
                    id=f"direct-development-{index}",
                    query_sha256=literal_digest,
                    query_text="argument",
                    search_mode="semantic",
                    filter_scope_sha256=default_scope_sha256,
                    filter_scope_json={},
                    result_id=f"window:{window.id}",
                    label="relevant" if window.id in relevant_ids else "not_relevant",
                    evaluation_split="validation",
                    evaluation_intent=DIRECT_CONFLICT_EVALUATION_INTENT,
                    notes=literal_regression_note,
                )
            )
        for index, window in enumerate(distinct_held_out):
            session.add(
                RelevanceFeedback(
                    id=f"direct-general-held-out-{index}",
                    query_sha256=hashlib.sha256(window.text_display.encode("utf-8")).hexdigest(),
                    query_text=window.text_display,
                    search_mode="semantic" if index % 2 == 0 else "hybrid",
                    filter_scope_sha256=default_scope_sha256,
                    filter_scope_json={},
                    result_id=f"window:{window.id}",
                    label="relevant",
                    evaluation_split="test",
                )
            )

    summary = build_semantic_index(
        layout,
        engine,
        embedder=FixtureEmbedder(),
        reranker_factory=lambda _spec: ExactMatchFixtureReranker(),
        rebuild_windows=False,
        batch_size=4,
    )
    with Session(engine) as session:
        config = session.get(SemanticIndexConfig, summary.config_hash)
        assert config is not None
        evaluation = config.audit_json["semantic_ir_evaluation"]
    calibration = evaluation["direct_conflict_development_calibration"]
    assert evaluation["production_acceptance_passed"] is True
    assert calibration["status"] == "passed"
    assert calibration["evidence_kind"] == "conversation_grouped_development_calibration"
    assert calibration["held_out"] is False
    assert calibration["judgment_count"] == 20
    assert calibration["binary_judgment_count"] == 20
    assert calibration["relevant_count"] == 10
    assert calibration["not_relevant_count"] == 10
    assert calibration["conversation_count"] == 20
    assert calibration["grouped_cross_validation"]["passed"] is True
    assert calibration["literal_query_regression"]["passed"] is True

    real_derive_ranker = search_service.derive_direct_conflict_ranker
    derive_call_count = 0

    def counted_derive_ranker(*args: object, **kwargs: object) -> object:
        nonlocal derive_call_count
        derive_call_count += 1
        return real_derive_ranker(*args, **kwargs)

    monkeypatch.setattr(
        search_service,
        "derive_direct_conflict_ranker",
        counted_derive_ranker,
    )
    page = semantic_search(
        layout,
        engine,
        "  ARGUMENT  ",
        limit=10,
        embedder_factory=lambda _spec: FixtureEmbedder(),
        reranker_factory=lambda _spec: ExactMatchFixtureReranker(),
    )
    assert page.items
    assert all(item["score_kind"] == DIRECT_CONFLICT_SCORE_KIND for item in page.items)
    assert page.items[0]["ranking_provenance"]["ranking_pipeline"] == "runtime-search-v2"
    assert len({str(item["window_id"]) for item in page.items} & relevant_ids) >= 6
    cached_page = semantic_search(
        layout,
        engine,
        "argument",
        limit=10,
        embedder_factory=lambda _spec: FixtureEmbedder(),
        reranker_factory=lambda _spec: ExactMatchFixtureReranker(),
    )
    assert [item["window_id"] for item in cached_page.items] == [
        item["window_id"] for item in page.items
    ]
    assert derive_call_count == 1

    baseline = semantic_search(
        layout,
        engine,
        "argument!",
        limit=3,
        embedder_factory=lambda _spec: FixtureEmbedder(),
        reranker_factory=lambda _spec: ExactMatchFixtureReranker(),
    )
    assert baseline.items
    assert all(item["score_kind"] == DEFAULT_RERANKER_SPEC.score_kind for item in baseline.items)

    with Session(engine) as session, session.begin():
        relevant = session.scalar(
            select(RelevanceFeedback).where(
                RelevanceFeedback.id.in_(
                    [f"direct-development-{index}" for index in range(len(selected))]
                ),
                RelevanceFeedback.label == "relevant",
            )
        )
        not_relevant = session.scalar(
            select(RelevanceFeedback).where(
                RelevanceFeedback.id.in_(
                    [f"direct-development-{index}" for index in range(len(selected))]
                ),
                RelevanceFeedback.label == "not_relevant",
            )
        )
        assert relevant is not None and not_relevant is not None
        relevant.label = "not_relevant"
        not_relevant.label = "relevant"

    with pytest.raises(SearchUnavailableError, match="stale"):
        semantic_search(
            layout,
            engine,
            "argument",
            limit=3,
            embedder_factory=lambda _spec: FixtureEmbedder(),
            reranker_factory=lambda _spec: ExactMatchFixtureReranker(),
        )
    assert derive_call_count == 2


def test_held_out_feedback_can_reject_an_automatic_benchmark_pass(
    stage3_catalog: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = stage3_catalog
    rebuild_search_windows(layout, engine, generation="held-out-rejection-fixture")
    with Session(engine) as session:
        windows = list(session.scalars(select(SearchWindow).order_by(SearchWindow.id)))
    assert len(windows) > 15
    # The held-out query encoder below returns the first basis vector. Select the five documents
    # with the lowest cosine similarity to it, keeping every judged target outside top 10.
    targets = sorted(
        windows,
        key=lambda window: FixtureEmbedder._vector(window.text_display)[0],
    )[:5]
    with Session(engine) as session, session.begin():
        for index, window in enumerate(targets):
            query_text = f"held-out unrelated query {index}"
            session.add(
                RelevanceFeedback(
                    id=f"held-out-rejection-{index}",
                    query_sha256=hashlib.sha256(query_text.encode("utf-8")).hexdigest(),
                    query_text=query_text,
                    search_mode="semantic",
                    result_id=f"window:{window.id}",
                    label="relevant",
                    evaluation_split="test",
                )
            )

    with pytest.raises(SemanticIndexAcceptanceError) as failure:
        build_semantic_index(
            layout,
            engine,
            embedder=HeldOutFailureFixtureEmbedder("sha256:held-out-failure"),
            reranker_factory=lambda _spec: ExactMatchFixtureReranker(),
            rebuild_windows=False,
            batch_size=4,
        )

    evaluation = failure.value.evaluation
    assert evaluation["automatic_known_item"]["passed"] is True
    assert evaluation["held_out_feedback"]["used_for_activation"] is True
    assert evaluation["held_out_feedback"]["passed"] is False
    assert evaluation["held_out_feedback"]["metrics"]["recall_at_k"] == 0.0
    with Session(engine) as session:
        candidate = session.get(SemanticIndexConfig, failure.value.config_hash)
        assert candidate is not None
        assert candidate.status == "rejected"
        assert candidate.active is False
        assert (
            session.scalar(
                select(func.count(SemanticIndexConfig.id)).where(
                    SemanticIndexConfig.active.is_(True)
                )
            )
            == 0
        )

    target_text_by_query = {
        f"held-out unrelated query {index}": window.text_display
        for index, window in enumerate(targets)
    }

    class TargetAwareFixtureReranker(ExactMatchFixtureReranker):
        def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
            self.calls.append((query, list(documents)))
            target_text = target_text_by_query.get(query)
            return [1.0 if document == target_text else 0.0 for document in documents]

    reranker = TargetAwareFixtureReranker()
    recovered = build_semantic_index(
        layout,
        engine,
        embedder=HeldOutFailureFixtureEmbedder("sha256:held-out-failure"),
        reranker_factory=lambda _spec: reranker,
        rebuild_windows=False,
        batch_size=4,
    )

    assert recovered.embedded_count == 0
    with Session(engine) as session:
        active = session.get(SemanticIndexConfig, recovered.config_hash)
        assert active is not None and active.active is True and active.status == "ready"
        recovered_held_out = active.audit_json["semantic_ir_evaluation"]["held_out_feedback"]
    assert recovered_held_out["passed"] is True
    assert recovered_held_out["metrics"]["recall_at_k"] == 1.0
    assert recovered_held_out["metrics"]["mrr"] >= HELD_OUT_THRESHOLDS["mrr"]
    assert recovered_held_out["metrics"]["ndcg_at_k"] >= HELD_OUT_THRESHOLDS["ndcg_at_k"]
    assert len(reranker.calls) == 5


def test_rejected_candidate_preserves_ready_index_and_resumes_without_reembedding(
    stage3_catalog: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = stage3_catalog
    accepted = build_semantic_index(
        layout,
        engine,
        embedder=FingerprintedFixtureEmbedder("sha256:accepted-artifact"),
        batch_size=4,
    )
    broken = BrokenQueryFixtureEmbedder("sha256:rejected-artifact")

    with pytest.raises(SemanticIndexAcceptanceError) as first_failure:
        build_semantic_index(
            layout,
            engine,
            embedder=broken,
            rebuild_windows=False,
            batch_size=4,
        )
    rejected_hash = first_failure.value.config_hash
    document_calls_after_first_attempt = broken.calls
    first_metrics = first_failure.value.evaluation["automatic_known_item"]["metrics"]

    with Session(engine) as session:
        active = session.scalar(
            select(SemanticIndexConfig).where(SemanticIndexConfig.active.is_(True))
        )
        rejected = session.get(SemanticIndexConfig, rejected_hash)
        failed_job = session.get(Job, first_failure.value.job_id)
        assert active is not None
        assert active.id == accepted.config_hash
        assert active.status == "ready"
        assert rejected is not None
        assert rejected.active is False
        assert rejected.status == "rejected"
        assert rejected.audit_json["semantic_ir_evaluation"]["passed"] is False
        assert failed_job is not None
        assert failed_job.status == "failed"
        assert failed_job.stage == "Semantic index rejected by retrieval quality gate"
        assert (
            session.scalar(
                select(func.count(SemanticWindowEmbedding.window_id)).where(
                    SemanticWindowEmbedding.config_id == accepted.config_hash
                )
            )
            == accepted.window_count
        )
        assert (
            session.scalar(
                select(func.count(SemanticWindowEmbedding.window_id)).where(
                    SemanticWindowEmbedding.config_id == rejected_hash
                )
            )
            == accepted.window_count
        )

    fallback = semantic_search(
        layout,
        engine,
        "fixture",
        limit=10,
        embedder_factory=lambda _spec: FingerprintedFixtureEmbedder("sha256:accepted-artifact"),
        reranker_factory=fixture_reranker_factory,
    )
    assert fallback.items
    assert fallback.index_config is not None
    assert fallback.index_config["id"] == accepted.config_hash

    with pytest.raises(SemanticIndexAcceptanceError) as resumed_failure:
        build_semantic_index(
            layout,
            engine,
            embedder=broken,
            rebuild_windows=False,
            batch_size=4,
        )

    assert resumed_failure.value.job_id == first_failure.value.job_id
    assert broken.calls == document_calls_after_first_attempt
    assert resumed_failure.value.evaluation["automatic_known_item"]["metrics"] == first_metrics
    with Session(engine) as session:
        active = session.scalar(
            select(SemanticIndexConfig).where(SemanticIndexConfig.active.is_(True))
        )
        rejected = session.get(SemanticIndexConfig, rejected_hash)
        assert active is not None and active.id == accepted.config_hash
        assert rejected is not None and rejected.status == "rejected"


def test_language_audit_detects_latin_script_non_english_text(
    stage3_catalog: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = stage3_catalog
    rebuild_search_windows(layout, engine, generation="language-audit-fixture")
    spanish = (
        "Esta conversación de ejemplo está completamente en español y contiene suficientes "
        "palabras naturales para identificar el idioma correctamente."
    )
    with Session(engine) as session, session.begin():
        session.execute(update(SearchWindow).values(text_display=spanish, text_search=spanish))

    audit = search_window_language_audit(engine)

    assert audit["language_counts"]["SPANISH"] > 0
    assert audit["non_english_ratio"] == 1.0
    assert audit["recommended_model"] == "multilingual-e5-small"


def test_auto_model_selection_runs_on_reused_windows(
    stage3_catalog: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = stage3_catalog
    rebuild_search_windows(layout, engine, generation="auto-language-fixture")
    spanish = (
        "Esta conversación está escrita completamente en español para comprobar que la auditoría "
        "de idioma sigue ejecutándose aunque se reutilicen las ventanas existentes."
    )
    with Session(engine) as session, session.begin():
        session.execute(update(SearchWindow).values(text_display=spanish, text_search=spanish))

    selected: list[str] = []

    def factory(spec: EmbeddingModelSpec) -> DynamicFixtureEmbedder:
        selected.append(spec.alias)
        return DynamicFixtureEmbedder(spec)

    summary = build_semantic_index(
        layout,
        engine,
        embedder_factory=factory,
        model_alias="auto",
        rebuild_windows=False,
        batch_size=4,
    )

    assert selected == ["multilingual-e5-small"]
    assert summary.model_name == "intfloat/multilingual-e5-small"
    with Session(engine) as session:
        config = session.get(SemanticIndexConfig, summary.config_hash)
        assert config is not None
        assert config.backend == "lancedb-transformers"
        assert "transformers-" in config.backend_version
        assert "torch-" in config.backend_version
        assert config.audit_json["model_selection"] == "language_audit"
        assert config.audit_json["language_audit"]["recommended_model"] == ("multilingual-e5-small")


def test_default_embedding_backend_uses_transformers_only_for_multilingual_small(
    archive_layout: ArchiveLayout, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected: list[tuple[str, str]] = []

    def fake_transformers(spec: EmbeddingModelSpec, *, cache_dir: str) -> DynamicFixtureEmbedder:
        selected.append(("transformers", cache_dir))
        return DynamicFixtureEmbedder(spec)

    def fake_fastembed(spec: EmbeddingModelSpec, *, cache_dir: str) -> DynamicFixtureEmbedder:
        selected.append(("fastembed", cache_dir))
        return DynamicFixtureEmbedder(spec)

    monkeypatch.setattr(search_index_service, "TransformersTextEmbedder", fake_transformers)
    monkeypatch.setattr(search_index_service, "FastEmbedTextEmbedder", fake_fastembed)

    multilingual = search_index_service.default_text_embedder(
        archive_layout,
        search_index_service.embedding_model_spec("multilingual-e5-small"),
    )
    english = search_index_service.default_text_embedder(
        archive_layout,
        search_index_service.embedding_model_spec("bge-small-en-v1.5"),
    )

    assert multilingual.spec.model_name == "intfloat/multilingual-e5-small"
    assert multilingual.spec.model_revision == "fd1525a9fd15316a2d503bf26ab031a61d056e98"
    assert english.spec.model_name == "BAAI/bge-small-en-v1.5"
    assert selected == [
        ("transformers", str(archive_layout.models / "transformers")),
        ("fastembed", str(archive_layout.models / "fastembed")),
    ]


def test_artifact_fingerprint_versions_index_and_rejects_query_mismatch(
    stage3_catalog: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = stage3_catalog
    first = build_semantic_index(
        layout,
        engine,
        embedder=FingerprintedFixtureEmbedder("sha256:artifact-a"),
        batch_size=4,
    )
    second = build_semantic_index(
        layout,
        engine,
        embedder=FingerprintedFixtureEmbedder("sha256:artifact-b"),
        rebuild_windows=False,
        batch_size=4,
    )

    assert second.config_hash != first.config_hash
    assert second.vector_table != first.vector_table
    with Session(engine) as session:
        active = session.get(SemanticIndexConfig, second.config_hash)
        assert active is not None
        assert active.audit_json["artifact_fingerprint"] == "sha256:artifact-b"

    with pytest.raises(SearchUnavailableError, match="artifact no longer matches"):
        semantic_search(
            layout,
            engine,
            "fixture",
            limit=10,
            embedder_factory=lambda _spec: FingerprintedFixtureEmbedder("sha256:artifact-a"),
        )


def test_activity_reimport_invalidates_every_ready_semantic_config(
    stage3_catalog: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = stage3_catalog
    built = build_semantic_index(layout, engine, embedder=FixtureEmbedder(), batch_size=4)

    summary = import_activities(layout, engine, force=True)
    assert summary.files_imported > 0
    with Session(engine) as session:
        config = session.get(SemanticIndexConfig, built.config_hash)
        assert config is not None
        assert config.active is False
        assert config.status == "stale"
        assert config.audit_json["stale"]["source_kind"] in {
            "activity_import",
            "activity_import_error",
        }


def test_embedding_implementation_identity_versions_index_and_search(
    stage3_catalog: tuple[ArchiveLayout, Engine], monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, engine = stage3_catalog
    built = build_semantic_index(
        layout,
        engine,
        embedder=FingerprintedFixtureEmbedder("sha256:implementation-artifact"),
        batch_size=4,
    )
    original_hash = search_index_service.embedding_config_hash(
        FIXTURE_SPEC,
        corpus_revision="synthetic-revision",
        artifact_fingerprint="sha256:implementation-artifact",
    )
    monkeypatch.setattr(
        search_index_service,
        "EMBEDDING_PIPELINE_VERSION",
        "synthetic-breaking-pipeline-change",
    )
    changed_hash = search_index_service.embedding_config_hash(
        FIXTURE_SPEC,
        corpus_revision="synthetic-revision",
        artifact_fingerprint="sha256:implementation-artifact",
    )
    assert changed_hash != original_hash

    with pytest.raises(SearchUnavailableError, match="implementation no longer matches"):
        semantic_search(
            layout,
            engine,
            "fixture",
            limit=10,
            embedder_factory=lambda _spec: FingerprintedFixtureEmbedder(
                "sha256:implementation-artifact"
            ),
            reranker_factory=fixture_reranker_factory,
        )
    with Session(engine) as session:
        active = session.get(SemanticIndexConfig, built.config_hash)
        assert active is not None and active.active is True


def test_process_lock_blocks_concurrent_build_and_recovers_stale_running_job(
    stage3_catalog: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = stage3_catalog
    initial = build_semantic_index(layout, engine, embedder=FixtureEmbedder(), batch_size=4)

    with (
        semantic_index_lock(layout),
        pytest.raises(SemanticIndexBusyError, match="already running"),
    ):
        build_semantic_index(layout, engine, embedder=FixtureEmbedder(), batch_size=4)

    stale_job_id = "stale-semantic-index-job"
    with Session(engine) as session, session.begin():
        session.add(
            Job(
                id=stale_job_id,
                kind="semantic_index",
                status="running",
                stage="Embedding search windows",
                progress_current=1,
                checkpoint_json={"config_hash": initial.config_hash},
                counters_json={},
            )
        )
    recovered = build_semantic_index(
        layout,
        engine,
        embedder=FixtureEmbedder(),
        rebuild_windows=False,
        batch_size=4,
    )

    assert recovered.job_id == stale_job_id
    assert recovered.resumed is True
    with Session(engine) as session:
        stale = session.get(Job, stale_job_id)
        assert stale is not None
        assert stale.status == "completed"
        assert stale.error_text is None


def test_failed_changed_corpus_never_pairs_old_vectors_with_new_window_text(
    stage3_catalog: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = stage3_catalog
    initial = build_semantic_index(layout, engine, embedder=FixtureEmbedder(), batch_size=4)
    with Session(engine) as session:
        target = session.scalar(
            select(Message)
            .where(Message.content_display.is_not(None), Message.deleted_at.is_(None))
            .order_by(Message.id)
        )
        assert target is not None
        target_window_ids = set(
            session.scalars(
                select(SearchWindowMessage.window_id).where(
                    SearchWindowMessage.message_id == target.id
                )
            )
        )
        assert target_window_ids
    changed_text = "changed corpus marker that must never use the old vector"
    with Session(engine) as session, session.begin():
        session.execute(
            update(Message)
            .where(Message.id == target.id)
            .values(
                content_raw=changed_text,
                content_display=changed_text,
                content_search=changed_text,
            )
        )

    with pytest.raises(RuntimeError, match="synthetic embedding interruption"):
        build_semantic_index(
            layout,
            engine,
            embedder=FixtureEmbedder(fail_on_call=1),
            batch_size=4,
        )

    page = semantic_search(
        layout,
        engine,
        "fixture",
        limit=1_000,
        embedder_factory=lambda _spec: FixtureEmbedder(),
        reranker_factory=fixture_reranker_factory,
    )
    returned_window_ids = {str(item["window_id"]) for item in page.items}
    assert target_window_ids.isdisjoint(returned_window_ids)
    assert page.index_config is not None
    assert page.index_config["id"] == initial.config_hash

    hybrid = hybrid_search(
        layout,
        engine,
        "changed corpus marker",
        limit=1_000,
        embedder_factory=lambda _spec: FixtureEmbedder(),
        reranker_factory=fixture_reranker_factory,
    )
    assert f"message:{target.id}" in {item["id"] for item in hybrid.items}


def test_semantic_filters_are_pushed_down_before_the_candidate_cap(
    stage3_catalog: tuple[ArchiveLayout, Engine], monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, engine = stage3_catalog
    built = build_semantic_index(layout, engine, embedder=FixtureEmbedder(), batch_size=4)
    with Session(engine) as session:
        target = session.scalar(
            select(SearchWindow)
            .where(SearchWindow.participants_search.like("%alex example%"))
            .order_by(SearchWindow.id)
        )
        assert target is not None

    class FakeVectorQuery:
        def __init__(self) -> None:
            self.filter_statement: str | None = None
            self.result_limit = 0
            self.vector_index_bypassed = False

        def distance_type(self, _distance: str) -> FakeVectorQuery:
            return self

        def bypass_vector_index(self) -> FakeVectorQuery:
            self.vector_index_bypassed = True
            return self

        def where(self, statement: str) -> FakeVectorQuery:
            self.filter_statement = statement
            return self

        def limit(self, value: int) -> FakeVectorQuery:
            self.result_limit = value
            return self

        def to_list(self) -> list[dict[str, object]]:
            # The matching row conceptually ranks after 5,000 unfiltered rows. It is available
            # only because the participant predicate was applied before the result limit.
            if self.filter_statement and "participants_search LIKE" in self.filter_statement:
                return [{"id": target.id, "_distance": 0.1}]
            return [
                {"id": f"noise-{index}", "_distance": 0.2} for index in range(self.result_limit)
            ]

    query = FakeVectorQuery()

    class FakeTable:
        def search(self, _vector: list[float], *, vector_column_name: str) -> FakeVectorQuery:
            assert vector_column_name == "vector"
            return query

    class FakeDatabase:
        def list_tables(self, *, limit: int) -> object:
            assert limit == 1_000
            return type("TableNames", (), {"tables": [built.vector_table]})()

        def open_table(self, table_name: str) -> FakeTable:
            assert table_name == built.vector_table
            return FakeTable()

    monkeypatch.setattr(search_service.lancedb, "connect", lambda _path: FakeDatabase())
    page = semantic_search(
        layout,
        engine,
        "fixture",
        filters=SearchFilters(entity_type="message", participant="Alex Example"),
        limit=1,
        embedder_factory=lambda _spec: FixtureEmbedder(),
        reranker_factory=fixture_reranker_factory,
    )

    assert query.result_limit == 100
    assert query.vector_index_bypassed is True
    assert query.filter_statement is not None
    assert "entity_type = 'message_window'" in query.filter_statement
    assert [item["window_id"] for item in page.items] == [target.id]

    escaped = search_service._lance_filter(SearchFilters(participant=r"Literal%_\\Name"))
    assert escaped is not None
    assert r"\%" in escaped
    assert r"\_" in escaped
    assert "ESCAPE" in escaped


def test_semantic_vector_contract_rejects_nonfinite_documents_and_queries(
    stage3_catalog: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = stage3_catalog
    with pytest.raises(RuntimeError, match="invalid vector"):
        build_semantic_index(
            layout,
            engine,
            embedder=NonFiniteDocumentFixtureEmbedder(),
            batch_size=4,
        )

    build_semantic_index(layout, engine, embedder=FixtureEmbedder(), batch_size=4)
    with pytest.raises(SearchUnavailableError, match="vector contract"):
        semantic_search(
            layout,
            engine,
            "fixture",
            embedder_factory=lambda _spec: NonFiniteQueryFixtureEmbedder(),
            reranker_factory=fixture_reranker_factory,
        )


def test_semantic_search_uses_bounded_pinned_local_reranking_and_fails_honestly(
    stage3_catalog: tuple[ArchiveLayout, Engine], tmp_path: Path
) -> None:
    layout, engine = stage3_catalog
    build_semantic_index(layout, engine, embedder=FixtureEmbedder(), batch_size=4)
    lazy = TransformersCrossEncoderReranker(tmp_path / "models")
    assert lazy._model is None
    assert lazy.status["status"] == "ready_for_lazy_load"

    reranker = FixtureReranker()
    page = semantic_search(
        layout,
        engine,
        "fixture",
        limit=5,
        embedder_factory=lambda _spec: FixtureEmbedder(),
        reranker_factory=lambda _spec: reranker,
    )
    assert len(reranker.calls) == 1
    assert len(reranker.calls[0][1]) <= DEFAULT_RERANKER_SPEC.max_candidates
    assert page.items
    assert page.items[0]["score"] == len(reranker.calls[0][1]) - 1
    assert all(item["score_kind"] == "cross_encoder_logit" for item in page.items)
    assert page.index_config is not None
    assert page.index_config["reranker"]["model_revision"] == (DEFAULT_RERANKER_SPEC.model_revision)
    assert page.items[0]["ranking_provenance"]["pipeline"] == (
        "cosine_retrieval_then_local_cross_encoder"
    )

    with pytest.raises(SearchUnavailableError, match="will not silently"):
        semantic_search(
            layout,
            engine,
            "fixture",
            limit=5,
            embedder_factory=lambda _spec: FixtureEmbedder(),
            reranker_factory=lambda _spec: FixtureReranker(fail=True),
        )
