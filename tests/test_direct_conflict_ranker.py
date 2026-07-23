from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import lancedb  # type: ignore[import-untyped]
import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pytest
from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from meta_archive.models import (
    RelevanceFeedback,
    SearchWindow,
    SemanticIndexConfig,
    SemanticWindowEmbedding,
)
from meta_archive.paths import ArchiveLayout
from meta_archive.services import search as search_service
from meta_archive.services.direct_conflict_ranker import (
    DEFAULT_FILTER_SCOPE_SHA256,
    DIRECT_CONFLICT_EVALUATION_INTENT,
    DIRECT_CONFLICT_FEEDBACK_REVISION_VERSION,
    DIRECT_CONFLICT_MAX_FEEDBACK_ROWS,
    DIRECT_CONFLICT_RANKER_VERSION,
    DirectConflictFeedbackRevision,
    DirectConflictRankerError,
    DirectConflictTrainingData,
    DirectConflictTrainingExample,
    DirectConflictVectorRow,
    LanceDirectConflictVectorSource,
    conversation_group_validation_splits,
    derive_direct_conflict_ranker,
    development_feedback_revision_payload,
    direct_conflict_material_fingerprint,
    fit_direct_conflict_ranker,
    is_direct_conflict_trigger,
    public_ranker_audit_metadata,
)


def _unit_vector(index: int, *, relevant: bool, dimensions: int = 4) -> tuple[float, ...]:
    values = np.asarray(
        [
            2.0 if relevant else -2.0,
            ((index % 5) - 2) * 0.13,
            ((index % 3) - 1) * 0.11,
            0.07 + (index % 2) * 0.03,
        ][:dimensions],
        dtype=np.float32,
    )
    values /= np.linalg.norm(values)
    return tuple(float(value) for value in values)


def _training_data(
    *,
    relevant: int = 10,
    not_relevant: int = 10,
    uncertain: int = 2,
    reverse: bool = False,
) -> DirectConflictTrainingData:
    examples: list[DirectConflictTrainingExample] = []
    for index in range(relevant):
        examples.append(
            DirectConflictTrainingExample(
                private_window_id=f"private-positive-{index:03d}",
                private_conversation_group=f"private-group-{index % 5}",
                label="relevant",
                vector=_unit_vector(index, relevant=True),
            )
        )
    for index in range(not_relevant):
        examples.append(
            DirectConflictTrainingExample(
                private_window_id=f"private-negative-{index:03d}",
                private_conversation_group=f"private-group-{index % 5}",
                label="not_relevant",
                vector=_unit_vector(index, relevant=False),
            )
        )
    for index in range(uncertain):
        examples.append(
            DirectConflictTrainingExample(
                private_window_id=f"private-uncertain-{index:03d}",
                private_conversation_group=f"private-uncertain-group-{index}",
                label="uncertain",
                vector=_unit_vector(index + 20, relevant=bool(index % 2)),
            )
        )
    if reverse:
        examples.reverse()
    return DirectConflictTrainingData(
        examples=tuple(examples),
        dimensions=4,
        private_config_material={
            "id": "private-config-identifier",
            "vector_table": "private-vector-table",
            "library_revision": "private-library-revision",
        },
        feedback_revision=DirectConflictFeedbackRevision(
            version=DIRECT_CONFLICT_FEEDBACK_REVISION_VERSION,
            sha256="1" * 64,
            row_count=len(examples),
        ),
    )


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("argument", True),
        (" ARGUMENT\n", True),
        ("\uff41\uff52\uff47\uff55\uff4d\uff45\uff4e\uff54", True),
        ("argument ", True),
        ("an argument", False),
        ("argument now", False),
        ("argu ment", False),
        ("arguments", False),
        ("", False),
    ],
)
def test_trigger_is_exact_after_nfkc_casefold_and_outer_whitespace(
    query: str, expected: bool
) -> None:
    assert is_direct_conflict_trigger(query) is expected


def test_fit_is_deterministic_and_uncertain_rows_only_change_material_fingerprint() -> None:
    first = fit_direct_conflict_ranker(_training_data())
    reordered = fit_direct_conflict_ranker(_training_data(reverse=True))
    probes = [
        _unit_vector(40, relevant=True),
        _unit_vector(41, relevant=False),
    ]

    assert first.audit.version == DIRECT_CONFLICT_RANKER_VERSION
    assert first.audit.counts.reviewed_feedback_rows == 22
    assert first.audit.counts.binary_training_rows == 20
    assert first.audit.counts.relevant_rows == 10
    assert first.audit.counts.not_relevant_rows == 10
    assert first.audit.counts.uncertain_rows == 2
    assert first.audit.counts.conversation_groups == 7
    assert first.score(probes) == reordered.score(probes)
    assert direct_conflict_material_fingerprint(first) == direct_conflict_material_fingerprint(
        reordered
    )

    original = _training_data()
    changed_examples = list(original.examples)
    uncertain_example = changed_examples[-1]
    changed_examples[-1] = DirectConflictTrainingExample(
        private_window_id=uncertain_example.private_window_id,
        private_conversation_group=uncertain_example.private_conversation_group,
        label="uncertain",
        vector=_unit_vector(99, relevant=True),
    )
    changed = fit_direct_conflict_ranker(
        DirectConflictTrainingData(
            examples=tuple(changed_examples),
            dimensions=original.dimensions,
            private_config_material=original.private_config_material,
            feedback_revision=DirectConflictFeedbackRevision(
                version=original.feedback_revision.version,
                sha256="2" * 64,
                row_count=original.feedback_revision.row_count,
            ),
        )
    )
    assert first.score(probes) == changed.score(probes)
    assert first.material_fingerprint != changed.material_fingerprint


def test_validation_is_conversation_disjoint_and_reports_mean_and_minimum() -> None:
    labels = [1] * 10 + [0] * 10
    groups = [f"group-{index % 5}" for index in range(10)] * 2
    splits = conversation_group_validation_splits(labels, groups)
    assert len(splits) == 4
    for training, validation in splits:
        assert {groups[index] for index in training}.isdisjoint(
            {groups[index] for index in validation}
        )
        assert {labels[index] for index in validation} == {0, 1}

    metrics = fit_direct_conflict_ranker(_training_data()).audit.metrics
    assert metrics.fold_count == 4
    for summary in (
        metrics.average_precision,
        metrics.roc_auc,
        metrics.recall_at_10,
        metrics.mrr_at_10,
        metrics.ndcg_at_10,
    ):
        assert 0.0 <= summary.minimum <= summary.mean <= 1.0


def test_validation_fails_when_a_fold_lacks_a_class() -> None:
    labels = [1] * 10 + [0] * 10
    groups = ["all-positive"] * 10 + [f"negative-{index}" for index in range(10)]
    with pytest.raises(DirectConflictRankerError) as failure:
        conversation_group_validation_splits(labels, groups)
    assert failure.value.code == "direct_conflict_validation_fold_class_missing"


@pytest.mark.parametrize(
    "vector",
    [
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0, 0.0),
        (float("nan"), 0.0, 0.0, 1.0),
    ],
)
def test_score_rejects_malformed_nonfinite_or_nonunit_vectors(
    vector: tuple[float, ...],
) -> None:
    ranker = fit_direct_conflict_ranker(_training_data())
    with pytest.raises(DirectConflictRankerError) as failure:
        ranker.score([vector])
    assert failure.value.code == "direct_conflict_vector_invalid"


def test_fit_enforces_minimums_cap_and_unique_targets() -> None:
    insufficient = _training_data(relevant=9, uncertain=0)
    with pytest.raises(DirectConflictRankerError) as minimum_failure:
        fit_direct_conflict_ranker(insufficient)
    assert minimum_failure.value.code == "direct_conflict_feedback_minimum_not_met"

    over_cap = _training_data(relevant=130, not_relevant=127, uncertain=0)
    assert len(over_cap.examples) == DIRECT_CONFLICT_MAX_FEEDBACK_ROWS + 1
    with pytest.raises(DirectConflictRankerError) as cap_failure:
        fit_direct_conflict_ranker(over_cap)
    assert cap_failure.value.code == "direct_conflict_training_material_invalid"

    duplicated = _training_data(uncertain=0)
    duplicate_examples = list(duplicated.examples)
    duplicate_examples[-1] = DirectConflictTrainingExample(
        private_window_id=duplicate_examples[0].private_window_id,
        private_conversation_group=duplicate_examples[-1].private_conversation_group,
        label=duplicate_examples[-1].label,
        vector=duplicate_examples[-1].vector,
    )
    with pytest.raises(DirectConflictRankerError) as duplicate_failure:
        fit_direct_conflict_ranker(
            DirectConflictTrainingData(
                examples=tuple(duplicate_examples),
                dimensions=duplicated.dimensions,
                private_config_material=duplicated.private_config_material,
                feedback_revision=duplicated.feedback_revision,
            )
        )
    assert duplicate_failure.value.code == "direct_conflict_feedback_target_duplicate"


@dataclass
class _FakeVectorSource:
    rows: list[DirectConflictVectorRow]

    def load_vectors(
        self,
        *,
        vector_table: str,
        window_ids: Sequence[str],
        expected_row_count: int,
    ) -> Sequence[DirectConflictVectorRow]:
        del vector_table, window_ids, expected_row_count
        return list(self.rows)


@dataclass(frozen=True)
class _SeededDevelopmentData:
    config_id: str
    window_ids: tuple[str, ...]
    source: _FakeVectorSource


def _seed_development_data(
    engine: Engine,
    *,
    relevant: int = 10,
    not_relevant: int = 10,
    uncertain: int = 2,
    active: bool = True,
    status: str = "ready",
) -> _SeededDevelopmentData:
    labels = ["relevant"] * relevant + ["not_relevant"] * not_relevant + ["uncertain"] * uncertain
    config_id = "private-config-id"
    rows: list[DirectConflictVectorRow] = []
    window_ids: list[str] = []
    with Session(engine) as session, session.begin():
        session.add(
            SemanticIndexConfig(
                id=config_id,
                active=active,
                status=status,
                backend="private-backend",
                backend_version="private-backend-version",
                model_name="private-model",
                model_revision="private-model-revision",
                dimensions=4,
                distance_metric="cosine",
                document_prefix="",
                query_prefix="",
                vector_table="private-vector-table",
                window_version="private-window-version",
                window_count=len(labels),
                audit_json={"corpus_revision": "private-corpus-revision"},
            )
        )
        for index, label in enumerate(labels):
            window_id = f"private-window-{index:03d}"
            window_ids.append(window_id)
            session.add(
                SearchWindow(
                    id=window_id,
                    window_version="private-window-version",
                    entity_type="message_window",
                    turn_count=1,
                    approx_token_count=1,
                    text_display=f"private-content-{index}",
                    text_search=f"private-content-{index}",
                    participants_search="private-participant",
                    source_ids_json={},
                    build_generation="private-generation",
                )
            )
            session.add(SemanticWindowEmbedding(config_id=config_id, window_id=window_id))
            query = f"private-query-content-{index}"
            session.add(
                RelevanceFeedback(
                    id=f"private-feedback-id-{index:03d}",
                    query_sha256=hashlib.sha256(query.encode()).hexdigest(),
                    query_text=query,
                    search_mode="semantic",
                    filter_scope_sha256=DEFAULT_FILTER_SCOPE_SHA256,
                    filter_scope_json={},
                    result_id=f"window:{window_id}",
                    label=label,
                    evaluation_split="validation",
                    evaluation_intent=DIRECT_CONFLICT_EVALUATION_INTENT,
                )
            )
            rows.append(
                DirectConflictVectorRow(
                    window_id=window_id,
                    vector=_unit_vector(index, relevant=label == "relevant"),
                )
            )
    return _SeededDevelopmentData(config_id, tuple(window_ids), _FakeVectorSource(rows))


@pytest.fixture
def live_window_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    def valid_window_statement(
        ids: Sequence[str], _filters: Any, *, config_hash: str | None = None
    ) -> Any:
        del config_hash
        return select(SearchWindow).where(SearchWindow.id.in_(ids))

    monkeypatch.setattr(search_service, "_valid_window_statement", valid_window_statement)


def test_revision_and_derivation_include_uncertain_but_exclude_unrelated_intents(
    catalog_engine: Engine,
    live_window_stub: None,
) -> None:
    seeded = _seed_development_data(catalog_engine)
    first_revision = development_feedback_revision_payload(catalog_engine)
    assert first_revision["version"] == DIRECT_CONFLICT_FEEDBACK_REVISION_VERSION
    assert first_revision["row_count"] == 22
    assert len(str(first_revision["sha256"])) == 64

    with Session(catalog_engine) as session, session.begin():
        session.add(
            RelevanceFeedback(
                id="private-unrelated-feedback",
                query_sha256="3" * 64,
                query_text="private-unrelated-query",
                search_mode="semantic",
                filter_scope_sha256=DEFAULT_FILTER_SCOPE_SHA256,
                filter_scope_json={},
                result_id=f"window:{seeded.window_ids[0]}",
                label="relevant",
                evaluation_split="validation",
                evaluation_intent="different-private-intent",
            )
        )
    assert development_feedback_revision_payload(catalog_engine) == first_revision

    first = derive_direct_conflict_ranker(catalog_engine, seeded.source)
    assert first.audit.counts.reviewed_feedback_rows == 22
    assert first.audit.counts.binary_training_rows == 20
    assert first.audit.counts.uncertain_rows == 2

    with Session(catalog_engine) as session, session.begin():
        uncertain = session.get(RelevanceFeedback, "private-feedback-id-021")
        assert uncertain is not None
        uncertain.query_text = "changed-private-uncertain-query"
    changed_revision = development_feedback_revision_payload(catalog_engine)
    assert changed_revision["sha256"] != first_revision["sha256"]
    changed = derive_direct_conflict_ranker(catalog_engine, seeded.source)
    assert changed.score([_unit_vector(40, relevant=True)]) == first.score(
        [_unit_vector(40, relevant=True)]
    )
    assert changed.material_fingerprint != first.material_fingerprint


def test_tagged_selection_boundary_violation_is_fingerprinted_and_fails_closed(
    catalog_engine: Engine,
    live_window_stub: None,
) -> None:
    seeded = _seed_development_data(catalog_engine)
    before = development_feedback_revision_payload(catalog_engine)
    with Session(catalog_engine) as session, session.begin():
        session.add(
            RelevanceFeedback(
                id="private-invalid-boundary-feedback",
                query_sha256="4" * 64,
                query_text="private-invalid-boundary-query",
                search_mode="hybrid",
                filter_scope_sha256=DEFAULT_FILTER_SCOPE_SHA256,
                filter_scope_json={},
                result_id=f"window:{seeded.window_ids[0]}",
                label="relevant",
                evaluation_split="validation",
                evaluation_intent=DIRECT_CONFLICT_EVALUATION_INTENT,
            )
        )
    after = development_feedback_revision_payload(catalog_engine)
    assert after["row_count"] == int(before["row_count"]) + 1
    assert after["sha256"] != before["sha256"]
    with pytest.raises(DirectConflictRankerError) as failure:
        derive_direct_conflict_ranker(catalog_engine, seeded.source)
    assert failure.value.code == "direct_conflict_feedback_scope_invalid"


def test_candidate_config_can_be_derived_before_activation(
    catalog_engine: Engine,
    live_window_stub: None,
) -> None:
    seeded = _seed_development_data(catalog_engine, active=False, status="building")
    with pytest.raises(DirectConflictRankerError) as runtime_failure:
        derive_direct_conflict_ranker(catalog_engine, seeded.source)
    assert runtime_failure.value.code == "direct_conflict_active_config_invalid"
    candidate = derive_direct_conflict_ranker(
        catalog_engine,
        seeded.source,
        config_id=seeded.config_id,
    )
    assert candidate.audit.counts.binary_training_rows == 20


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("malformed", "direct_conflict_feedback_target_invalid"),
        ("stale", "direct_conflict_feedback_target_stale"),
        ("nonmember", "direct_conflict_feedback_target_nonmember"),
    ],
)
def test_malformed_stale_and_nonmember_targets_fail_closed(
    catalog_engine: Engine,
    live_window_stub: None,
    mutation: str,
    expected_code: str,
) -> None:
    seeded = _seed_development_data(catalog_engine)
    with Session(catalog_engine) as session, session.begin():
        if mutation == "malformed":
            feedback = session.get(RelevanceFeedback, "private-feedback-id-000")
            assert feedback is not None
            feedback.result_id = "message:private-malformed-target"
        elif mutation == "stale":
            feedback = session.get(RelevanceFeedback, "private-feedback-id-000")
            assert feedback is not None
            feedback.result_id = "window:private-stale-target"
        else:
            session.execute(
                delete(SemanticWindowEmbedding).where(
                    SemanticWindowEmbedding.config_id == seeded.config_id,
                    SemanticWindowEmbedding.window_id == seeded.window_ids[0],
                )
            )
    with pytest.raises(DirectConflictRankerError) as failure:
        derive_direct_conflict_ranker(catalog_engine, seeded.source)
    assert failure.value.code == expected_code


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "invalid"])
def test_vector_material_must_be_complete_unique_finite_unit_float32(
    catalog_engine: Engine,
    live_window_stub: None,
    mutation: str,
) -> None:
    seeded = _seed_development_data(catalog_engine)
    if mutation == "missing":
        seeded.source.rows.pop()
        expected_code = "direct_conflict_vector_set_incomplete"
    elif mutation == "duplicate":
        seeded.source.rows.append(seeded.source.rows[0])
        expected_code = "direct_conflict_vector_duplicate"
    else:
        first = seeded.source.rows[0]
        seeded.source.rows[0] = DirectConflictVectorRow(
            window_id=first.window_id,
            vector=(float("inf"), 0.0, 0.0, 1.0),
        )
        expected_code = "direct_conflict_vector_invalid"
    with pytest.raises(DirectConflictRankerError) as failure:
        derive_direct_conflict_ranker(catalog_engine, seeded.source)
    assert failure.value.code == expected_code


def test_lance_vector_source_reads_only_requested_rows_and_checks_table_count(
    archive_layout: ArchiveLayout,
) -> None:
    archive_layout.ensure_runtime_directories()
    database = lancedb.connect(str(archive_layout.vectors))
    database.create_table(
        "ranker_vectors",
        pa.table(
            {
                "id": ["private-vector-a", "private-vector-b"],
                "vector": pa.array(
                    [[1.0, 0.0], [0.0, 1.0]],
                    type=pa.list_(pa.float32(), 2),
                ),
            }
        ),
    )
    source = LanceDirectConflictVectorSource(archive_layout)
    rows = source.load_vectors(
        vector_table="ranker_vectors",
        window_ids=["private-vector-b", "private-vector-a"],
        expected_row_count=2,
    )
    assert {row.window_id for row in rows} == {"private-vector-a", "private-vector-b"}

    with pytest.raises(DirectConflictRankerError) as failure:
        source.load_vectors(
            vector_table="ranker_vectors",
            window_ids=["private-vector-a"],
            expected_row_count=3,
        )
    assert failure.value.code == "direct_conflict_vector_table_count_mismatch"


def test_errors_and_public_metadata_do_not_leak_private_material(
    catalog_engine: Engine,
    live_window_stub: None,
) -> None:
    seeded = _seed_development_data(catalog_engine)
    ranker = derive_direct_conflict_ranker(catalog_engine, seeded.source)
    public = public_ranker_audit_metadata(ranker)
    serialized = json.dumps(public, sort_keys=True)
    for private_value in (
        "private-query-content",
        "private-window",
        "private-feedback-id",
        "private-config-id",
        "private-vector-table",
        "private-content",
        "private-corpus-revision",
    ):
        assert private_value not in serialized
    assert "sha256" not in serialized
    assert "fingerprint" not in serialized
    assert "private" not in serialized

    with Session(catalog_engine) as session, session.begin():
        feedback = session.get(RelevanceFeedback, "private-feedback-id-000")
        assert feedback is not None
        feedback.result_id = "private-secret-malformed-target"
    with pytest.raises(DirectConflictRankerError) as failure:
        derive_direct_conflict_ranker(catalog_engine, seeded.source)
    error_payload = json.dumps(
        {"code": failure.value.code, "counts": failure.value.counts}, sort_keys=True
    )
    assert str(failure.value) == "direct_conflict_feedback_target_invalid"
    assert "private-secret" not in error_payload
