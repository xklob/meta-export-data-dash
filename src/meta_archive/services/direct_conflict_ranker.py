from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import version as package_version
from typing import Any, Literal, Protocol, cast

import lancedb  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (  # type: ignore[import-untyped]
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold  # type: ignore[import-untyped]
from sklearn.svm import LinearSVC  # type: ignore[import-untyped]
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from meta_archive.models import RelevanceFeedback, SearchWindow, SemanticIndexConfig
from meta_archive.paths import ArchiveLayout
from meta_archive.services.semantic_state import (
    lance_table_names,
    semantic_window_membership_clause,
)

DIRECT_CONFLICT_TRIGGER = "argument"
DIRECT_CONFLICT_EVALUATION_INTENT = "argument_direct_conflict_development"
DIRECT_CONFLICT_RANKER_VERSION = "linear-svc-direct-conflict-v2"
DIRECT_CONFLICT_FEEDBACK_REVISION_VERSION = "direct-conflict-development-feedback-v1"
DIRECT_CONFLICT_C = 1.0
DIRECT_CONFLICT_CLASS_WEIGHT = "balanced"
DIRECT_CONFLICT_RANDOM_STATE = 20260722
DIRECT_CONFLICT_MAX_FEEDBACK_ROWS = 256
DIRECT_CONFLICT_MIN_RELEVANT = 10
DIRECT_CONFLICT_MIN_NOT_RELEVANT = 10
DIRECT_CONFLICT_VALIDATION_FOLDS = 4
DIRECT_CONFLICT_RANKING_K = 10
DIRECT_CONFLICT_SCORE_KIND = "linear_svc_decision_function"
DEFAULT_FILTER_SCOPE_SHA256 = hashlib.sha256(b"{}").hexdigest()
_UNIT_NORM_TOLERANCE = 1e-3

DirectConflictFeedbackLabel = Literal["relevant", "not_relevant", "uncertain"]
BinaryDirectConflictLabel = Literal["relevant", "not_relevant"]


class DirectConflictRankerError(RuntimeError):
    """Fail-closed ranker error whose message contains only an aggregate-safe code."""

    def __init__(self, code: str, *, counts: Mapping[str, int] | None = None) -> None:
        self.code = code
        self.counts = dict(counts or {})
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class DirectConflictFeedbackRevision:
    version: str
    sha256: str
    row_count: int

    def private_payload(self) -> dict[str, int | str]:
        return asdict(self)

    def public_payload(self, *, current: bool | None = None) -> dict[str, int | str | bool]:
        payload: dict[str, int | str | bool] = {
            "version": self.version,
            "row_count": self.row_count,
        }
        if current is not None:
            payload["current"] = current
        return payload


@dataclass(frozen=True, slots=True)
class DirectConflictVectorRow:
    window_id: str
    vector: Sequence[float]


class DirectConflictVectorSource(Protocol):
    def load_vectors(
        self,
        *,
        vector_table: str,
        window_ids: Sequence[str],
        expected_row_count: int,
    ) -> Sequence[DirectConflictVectorRow]: ...


@dataclass(frozen=True, slots=True)
class LanceDirectConflictVectorSource:
    layout: ArchiveLayout

    def load_vectors(
        self,
        *,
        vector_table: str,
        window_ids: Sequence[str],
        expected_row_count: int,
    ) -> Sequence[DirectConflictVectorRow]:
        try:
            database = lancedb.connect(str(self.layout.vectors))
            if vector_table not in lance_table_names(database):
                raise DirectConflictRankerError("direct_conflict_vector_table_missing")
            table = database.open_table(vector_table)
            if int(table.count_rows()) != expected_row_count:
                raise DirectConflictRankerError("direct_conflict_vector_table_count_mismatch")
            loaded: list[DirectConflictVectorRow] = []
            for start in range(0, len(window_ids), 64):
                batch = list(window_ids[start : start + 64])
                literals = ",".join("'" + value.replace("'", "''") + "'" for value in batch)
                rows = (
                    table.search()
                    .where(f"id IN ({literals})")
                    .select(["id", "vector"])
                    .limit(len(batch) + 1)
                    .to_list()
                )
                loaded.extend(
                    DirectConflictVectorRow(
                        window_id=str(row.get("id") or ""),
                        vector=cast(Sequence[float], row.get("vector")),
                    )
                    for row in rows
                )
            return loaded
        except DirectConflictRankerError:
            raise
        except Exception:
            raise DirectConflictRankerError("direct_conflict_vector_source_unavailable") from None


@dataclass(frozen=True, slots=True)
class DirectConflictRankerParameters:
    c: float = DIRECT_CONFLICT_C
    class_weight: str = DIRECT_CONFLICT_CLASS_WEIGHT
    random_state: int = DIRECT_CONFLICT_RANDOM_STATE
    loss: str = "squared_hinge"
    penalty: str = "l2"
    dual: str = "auto"
    fit_intercept: bool = True
    intercept_scaling: float = 1.0
    max_iter: int = 1_000
    tolerance: float = 1e-4


DEFAULT_DIRECT_CONFLICT_PARAMETERS = DirectConflictRankerParameters()


@dataclass(frozen=True, slots=True)
class DirectConflictMetricSummary:
    mean: float
    minimum: float


@dataclass(frozen=True, slots=True)
class DirectConflictValidationMetrics:
    fold_count: int
    average_precision: DirectConflictMetricSummary
    roc_auc: DirectConflictMetricSummary
    recall_at_10: DirectConflictMetricSummary
    mrr_at_10: DirectConflictMetricSummary
    ndcg_at_10: DirectConflictMetricSummary

    def metadata(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DirectConflictRankerCounts:
    reviewed_feedback_rows: int
    binary_training_rows: int
    relevant_rows: int
    not_relevant_rows: int
    uncertain_rows: int
    conversation_groups: int


@dataclass(frozen=True, slots=True)
class DirectConflictRankerAudit:
    version: str
    parameters: DirectConflictRankerParameters
    counts: DirectConflictRankerCounts
    metrics: DirectConflictValidationMetrics
    private_feedback_revision: DirectConflictFeedbackRevision
    private_material_fingerprint_sha256: str

    def metadata(self, *, include_private: bool = True) -> dict[str, Any]:
        payload = self.public_metadata()
        if include_private:
            payload.update(
                {
                    "private_feedback_revision": (self.private_feedback_revision.private_payload()),
                    "private_material_fingerprint_sha256": (
                        self.private_material_fingerprint_sha256
                    ),
                }
            )
        return payload

    def public_metadata(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "parameters": asdict(self.parameters),
            "counts": asdict(self.counts),
            "metrics": self.metrics.metadata(),
            "score_kind": DIRECT_CONFLICT_SCORE_KIND,
            "ranking_feature_only": True,
        }


@dataclass(frozen=True, slots=True)
class _SelectedFeedback:
    private_row_id: str
    private_query_sha256: str
    private_query_text_sha256: str
    private_search_mode: str
    private_filter_scope_sha256: str
    private_filter_scope_json: Any
    private_target: str
    label: str
    private_evaluation_split: str | None
    private_evaluation_intent: str | None
    private_notes_sha256: str

    def revision_material(self) -> list[Any]:
        return [
            self.private_row_id,
            self.private_query_sha256,
            self.private_query_text_sha256,
            self.private_search_mode,
            self.private_filter_scope_sha256,
            self.private_filter_scope_json,
            self.private_target,
            self.label,
            self.private_evaluation_split,
            self.private_evaluation_intent,
            self.private_notes_sha256,
        ]


@dataclass(frozen=True, slots=True)
class DirectConflictTrainingExample:
    private_window_id: str
    private_conversation_group: str
    label: DirectConflictFeedbackLabel
    vector: Sequence[float]


@dataclass(frozen=True, slots=True)
class DirectConflictTrainingData:
    examples: tuple[DirectConflictTrainingExample, ...]
    dimensions: int
    private_config_material: Mapping[str, Any]
    feedback_revision: DirectConflictFeedbackRevision


@dataclass(frozen=True, slots=True)
class _CatalogTrainingContext:
    vector_table: str
    expected_row_count: int
    dimensions: int
    private_conversation_groups: Mapping[str, str]
    private_config_material: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class DirectConflictRanker:
    dimensions: int
    audit: DirectConflictRankerAudit
    _classifier: Any

    @property
    def material_fingerprint(self) -> str:
        return self.audit.private_material_fingerprint_sha256

    def score(self, vectors: Sequence[Sequence[float]]) -> list[float]:
        matrix = _validated_vector_matrix(vectors, dimensions=self.dimensions, allow_empty=True)
        if matrix.shape[0] == 0:
            return []
        try:
            result = np.asarray(self._classifier.decision_function(matrix), dtype=np.float64)
        except Exception:
            raise DirectConflictRankerError("direct_conflict_scoring_failed") from None
        if result.shape != (matrix.shape[0],) or not bool(np.isfinite(result).all()):
            raise DirectConflictRankerError("direct_conflict_scoring_failed")
        return [float(value) for value in result]


def is_direct_conflict_trigger(query: str) -> bool:
    return unicodedata.normalize("NFKC", query).casefold().strip() == DIRECT_CONFLICT_TRIGGER


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8", errors="surrogatepass")
    except (TypeError, ValueError):
        raise DirectConflictRankerError("direct_conflict_material_invalid") from None


def _update_framed(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _selected_development_feedback(engine: Engine) -> tuple[_SelectedFeedback, ...]:
    try:
        with Session(engine) as session:
            rows = list(
                session.scalars(
                    select(RelevanceFeedback)
                    .where(
                        RelevanceFeedback.evaluation_intent == DIRECT_CONFLICT_EVALUATION_INTENT,
                    )
                    .order_by(RelevanceFeedback.id)
                )
            )
    except Exception:
        raise DirectConflictRankerError("direct_conflict_feedback_source_unavailable") from None
    selected = tuple(
        sorted(
            (
                _SelectedFeedback(
                    private_row_id=str(row.id),
                    private_query_sha256=str(row.query_sha256),
                    private_query_text_sha256=hashlib.sha256(
                        str(row.query_text).encode("utf-8", errors="surrogatepass")
                    ).hexdigest(),
                    private_search_mode=str(row.search_mode),
                    private_filter_scope_sha256=str(row.filter_scope_sha256),
                    private_filter_scope_json=row.filter_scope_json,
                    private_target=str(row.result_id),
                    label=str(row.label),
                    private_evaluation_split=(
                        str(row.evaluation_split) if row.evaluation_split is not None else None
                    ),
                    private_evaluation_intent=(
                        str(row.evaluation_intent) if row.evaluation_intent is not None else None
                    ),
                    private_notes_sha256=hashlib.sha256(
                        str(row.notes or "").encode("utf-8", errors="surrogatepass")
                    ).hexdigest(),
                )
                for row in rows
            ),
            key=lambda row: _canonical_bytes(row.revision_material()),
        )
    )
    if len(selected) > DIRECT_CONFLICT_MAX_FEEDBACK_ROWS:
        raise DirectConflictRankerError(
            "direct_conflict_feedback_cap_exceeded",
            counts={"selected_feedback_rows": len(selected)},
        )
    return selected


def _revision_for_feedback(
    feedback: Sequence[_SelectedFeedback],
) -> DirectConflictFeedbackRevision:
    digest = hashlib.sha256()
    digest.update(DIRECT_CONFLICT_FEEDBACK_REVISION_VERSION.encode("ascii"))
    for row in feedback:
        _update_framed(digest, _canonical_bytes(row.revision_material()))
    return DirectConflictFeedbackRevision(
        version=DIRECT_CONFLICT_FEEDBACK_REVISION_VERSION,
        sha256=digest.hexdigest(),
        row_count=len(feedback),
    )


def development_feedback_revision_payload(engine: Engine) -> dict[str, int | str]:
    feedback = _selected_development_feedback(engine)
    return _revision_for_feedback(feedback).private_payload()


def development_feedback_revision(engine: Engine) -> str:
    return str(development_feedback_revision_payload(engine)["sha256"])


def public_ranker_audit_metadata(
    audit: DirectConflictRankerAudit | DirectConflictRanker,
) -> dict[str, Any]:
    selected = audit.audit if isinstance(audit, DirectConflictRanker) else audit
    return selected.public_metadata()


def direct_conflict_material_fingerprint(ranker: DirectConflictRanker) -> str:
    return ranker.material_fingerprint


def _validated_vector_matrix(
    vectors: Sequence[Sequence[float]],
    *,
    dimensions: int,
    allow_empty: bool,
) -> NDArray[np.float32]:
    if len(vectors) == 0:
        if not allow_empty:
            raise DirectConflictRankerError("direct_conflict_vector_set_incomplete")
        return np.empty((0, dimensions), dtype=np.float32)
    try:
        matrix = np.asarray(vectors, dtype=np.float32)
    except (TypeError, ValueError, OverflowError):
        raise DirectConflictRankerError("direct_conflict_vector_invalid") from None
    if matrix.shape != (len(vectors), dimensions):
        raise DirectConflictRankerError("direct_conflict_vector_invalid")
    if not bool(np.isfinite(matrix).all()):
        raise DirectConflictRankerError("direct_conflict_vector_invalid")
    norms = np.linalg.norm(matrix, axis=1)
    if not bool(
        np.allclose(
            norms,
            np.ones(len(vectors), dtype=np.float32),
            rtol=_UNIT_NORM_TOLERANCE,
            atol=_UNIT_NORM_TOLERANCE,
        )
    ):
        raise DirectConflictRankerError("direct_conflict_vector_invalid")
    return np.ascontiguousarray(matrix, dtype=np.float32)


def _linear_svc(parameters: DirectConflictRankerParameters) -> Any:
    return LinearSVC(
        C=parameters.c,
        class_weight=parameters.class_weight,
        random_state=parameters.random_state,
        loss=parameters.loss,
        penalty=parameters.penalty,
        dual=parameters.dual,
        fit_intercept=parameters.fit_intercept,
        intercept_scaling=parameters.intercept_scaling,
        max_iter=parameters.max_iter,
        tol=parameters.tolerance,
    )


def _ranking_metrics(labels: NDArray[np.int64], scores: NDArray[np.float64]) -> dict[str, float]:
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    top = ranked[:DIRECT_CONFLICT_RANKING_K]
    positives = int(np.sum(labels))
    recall = float(np.sum(top)) / positives
    positive_ranks = np.flatnonzero(top == 1)
    mrr = 0.0 if positive_ranks.size == 0 else 1.0 / float(positive_ranks[0] + 1)
    discounts = 1.0 / np.log2(np.arange(2, len(top) + 2, dtype=np.float64))
    discounted_gain = float(np.sum(top * discounts))
    ideal_count = min(positives, DIRECT_CONFLICT_RANKING_K)
    ideal_gain = float(np.sum(discounts[:ideal_count]))
    return {
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "recall_at_10": recall,
        "mrr_at_10": mrr,
        "ndcg_at_10": discounted_gain / ideal_gain,
    }


def _validation_metrics(
    matrix: NDArray[np.float32],
    labels: NDArray[np.int64],
    groups: Sequence[str],
    parameters: DirectConflictRankerParameters,
) -> DirectConflictValidationMetrics:
    splits = conversation_group_validation_splits(labels, groups)
    fold_metrics: list[dict[str, float]] = []
    for training_indices_value, validation_indices_value in splits:
        training_indices = np.asarray(training_indices_value, dtype=np.int64)
        validation_indices = np.asarray(validation_indices_value, dtype=np.int64)
        training_labels = labels[training_indices]
        validation_labels = labels[validation_indices]
        classifier = _linear_svc(parameters)
        try:
            classifier.fit(matrix[training_indices], training_labels)
            scores = np.asarray(
                classifier.decision_function(matrix[validation_indices]), dtype=np.float64
            )
        except Exception:
            raise DirectConflictRankerError("direct_conflict_validation_failed") from None
        if scores.shape != validation_labels.shape or not bool(np.isfinite(scores).all()):
            raise DirectConflictRankerError("direct_conflict_validation_failed")
        fold_metrics.append(_ranking_metrics(validation_labels, scores))
    if len(fold_metrics) != DIRECT_CONFLICT_VALIDATION_FOLDS:
        raise DirectConflictRankerError("direct_conflict_validation_unavailable")

    def summary(metric: str) -> DirectConflictMetricSummary:
        values = [fold[metric] for fold in fold_metrics]
        return DirectConflictMetricSummary(
            mean=float(np.mean(values)),
            minimum=float(np.min(values)),
        )

    return DirectConflictValidationMetrics(
        fold_count=len(fold_metrics),
        average_precision=summary("average_precision"),
        roc_auc=summary("roc_auc"),
        recall_at_10=summary("recall_at_10"),
        mrr_at_10=summary("mrr_at_10"),
        ndcg_at_10=summary("ndcg_at_10"),
    )


def conversation_group_validation_splits(
    labels: Sequence[int] | NDArray[np.int64],
    groups: Sequence[str],
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    label_array = np.asarray(labels, dtype=np.int64)
    if label_array.shape != (len(labels),) or len(groups) != len(labels):
        raise DirectConflictRankerError("direct_conflict_validation_unavailable")
    splitter = StratifiedGroupKFold(
        n_splits=DIRECT_CONFLICT_VALIDATION_FOLDS,
        shuffle=True,
        random_state=DIRECT_CONFLICT_RANDOM_STATE,
    )
    try:
        raw_splits = list(
            splitter.split(
                np.zeros((len(labels), 1), dtype=np.float32),
                label_array,
                np.asarray(groups, dtype=object),
            )
        )
    except ValueError:
        raise DirectConflictRankerError("direct_conflict_validation_unavailable") from None
    splits: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for training_indices, validation_indices in raw_splits:
        training_labels = label_array[training_indices]
        validation_labels = label_array[validation_indices]
        if len(np.unique(training_labels)) != 2 or len(np.unique(validation_labels)) != 2:
            raise DirectConflictRankerError("direct_conflict_validation_fold_class_missing")
        splits.append(
            (
                tuple(int(index) for index in training_indices),
                tuple(int(index) for index in validation_indices),
            )
        )
    if len(splits) != DIRECT_CONFLICT_VALIDATION_FOLDS:
        raise DirectConflictRankerError("direct_conflict_validation_unavailable")
    return tuple(splits)


def _material_fingerprint(
    prepared: DirectConflictTrainingData,
    parameters: DirectConflictRankerParameters,
    classifier: Any,
) -> str:
    digest = hashlib.sha256()
    digest.update(DIRECT_CONFLICT_RANKER_VERSION.encode("ascii"))
    _update_framed(digest, _canonical_bytes(prepared.private_config_material))
    _update_framed(
        digest,
        _canonical_bytes(
            {
                "numpy": package_version("numpy"),
                "scikit-learn": package_version("scikit-learn"),
            }
        ),
    )
    _update_framed(digest, _canonical_bytes(asdict(parameters)))
    _update_framed(digest, prepared.feedback_revision.sha256.encode("ascii"))
    for example in prepared.examples:
        _update_framed(digest, example.private_window_id.encode("utf-8", errors="surrogatepass"))
        _update_framed(
            digest,
            example.private_conversation_group.encode("utf-8", errors="surrogatepass"),
        )
        _update_framed(digest, example.label.encode("ascii"))
        vector = np.asarray(example.vector, dtype="<f4")
        _update_framed(digest, vector.tobytes(order="C"))
    coefficients = np.ascontiguousarray(classifier.coef_, dtype="<f8")
    intercept = np.ascontiguousarray(classifier.intercept_, dtype="<f8")
    _update_framed(digest, coefficients.tobytes(order="C"))
    _update_framed(digest, intercept.tobytes(order="C"))
    return digest.hexdigest()


def _catalog_training_context(
    engine: Engine,
    private_targets: Sequence[str],
    *,
    config_id: str | None,
) -> _CatalogTrainingContext:
    try:
        with Session(engine) as session:
            if config_id is None:
                configs = list(
                    session.scalars(
                        select(SemanticIndexConfig)
                        .where(
                            SemanticIndexConfig.active.is_(True),
                            SemanticIndexConfig.status == "ready",
                        )
                        .order_by(SemanticIndexConfig.id)
                    )
                )
                if len(configs) != 1:
                    raise DirectConflictRankerError("direct_conflict_active_config_invalid")
            else:
                selected_config = session.get(SemanticIndexConfig, config_id)
                configs = [selected_config] if selected_config is not None else []
                if not configs or configs[0].status not in {"building", "ready", "rejected"}:
                    raise DirectConflictRankerError("direct_conflict_candidate_config_invalid")
            config = configs[0]
            if config.dimensions <= 0 or config.window_count < len(private_targets):
                raise DirectConflictRankerError("direct_conflict_active_config_invalid")
            windows = {
                window.id: window
                for window in session.scalars(
                    select(SearchWindow).where(SearchWindow.id.in_(private_targets))
                )
            }
            if len(windows) != len(private_targets):
                raise DirectConflictRankerError("direct_conflict_feedback_target_stale")
            member_ids = set(
                session.scalars(
                    select(SearchWindow.id).where(
                        SearchWindow.id.in_(private_targets),
                        semantic_window_membership_clause(config.id),
                    )
                )
            )
            if len(member_ids) != len(private_targets):
                raise DirectConflictRankerError("direct_conflict_feedback_target_nonmember")

            from meta_archive.services.search import SearchFilters, _valid_window_statement

            live_ids = {
                window.id
                for window in session.scalars(
                    _valid_window_statement(
                        private_targets,
                        SearchFilters(),
                        config_hash=config.id,
                    )
                )
            }
            if len(live_ids) != len(private_targets):
                raise DirectConflictRankerError("direct_conflict_feedback_target_not_live")
            return _CatalogTrainingContext(
                vector_table=str(config.vector_table),
                expected_row_count=int(config.window_count),
                dimensions=int(config.dimensions),
                private_conversation_groups={
                    window_id: (windows[window_id].conversation_id or f"window:{window_id}")
                    for window_id in private_targets
                },
                private_config_material={
                    "id": config.id,
                    "backend": config.backend,
                    "backend_version": config.backend_version,
                    "model_name": config.model_name,
                    "model_revision": config.model_revision,
                    "dimensions": config.dimensions,
                    "distance_metric": config.distance_metric,
                    "document_prefix": config.document_prefix,
                    "query_prefix": config.query_prefix,
                    "vector_table": config.vector_table,
                    "window_version": config.window_version,
                    "window_count": config.window_count,
                    "corpus_revision": config.audit_json.get("corpus_revision"),
                },
            )
    except DirectConflictRankerError:
        raise
    except Exception:
        raise DirectConflictRankerError("direct_conflict_catalog_unavailable") from None


def prepare_direct_conflict_training_data(
    engine: Engine,
    vector_source: DirectConflictVectorSource,
    *,
    config_id: str | None = None,
) -> DirectConflictTrainingData:
    feedback = _selected_development_feedback(engine)
    revision = _revision_for_feedback(feedback)
    labels = {"relevant": 0, "not_relevant": 0, "uncertain": 0}
    target_labels: dict[str, str] = {}
    private_targets: list[str] = []
    for row in feedback:
        if (
            row.private_search_mode != "semantic"
            or row.private_filter_scope_sha256 != DEFAULT_FILTER_SCOPE_SHA256
            or not isinstance(row.private_filter_scope_json, dict)
            or bool(row.private_filter_scope_json)
            or row.private_evaluation_split != "validation"
            or row.private_evaluation_intent != DIRECT_CONFLICT_EVALUATION_INTENT
        ):
            raise DirectConflictRankerError("direct_conflict_feedback_scope_invalid")
        if row.label not in labels:
            raise DirectConflictRankerError("direct_conflict_feedback_label_invalid")
        labels[row.label] += 1
        if not row.private_target.startswith("window:"):
            raise DirectConflictRankerError("direct_conflict_feedback_target_invalid")
        private_window_id = row.private_target.removeprefix("window:")
        if not private_window_id:
            raise DirectConflictRankerError("direct_conflict_feedback_target_invalid")
        previous = target_labels.get(private_window_id)
        if previous is not None:
            code = (
                "direct_conflict_feedback_label_conflict"
                if previous != row.label
                else "direct_conflict_feedback_target_duplicate"
            )
            raise DirectConflictRankerError(code)
        target_labels[private_window_id] = row.label
        private_targets.append(private_window_id)
    if (
        labels["relevant"] < DIRECT_CONFLICT_MIN_RELEVANT
        or labels["not_relevant"] < DIRECT_CONFLICT_MIN_NOT_RELEVANT
    ):
        raise DirectConflictRankerError(
            "direct_conflict_feedback_minimum_not_met",
            counts={
                "relevant_rows": labels["relevant"],
                "not_relevant_rows": labels["not_relevant"],
                "uncertain_rows": labels["uncertain"],
            },
        )

    context = _catalog_training_context(engine, private_targets, config_id=config_id)
    try:
        vector_rows = list(
            vector_source.load_vectors(
                vector_table=context.vector_table,
                window_ids=private_targets,
                expected_row_count=context.expected_row_count,
            )
        )
    except DirectConflictRankerError:
        raise
    except Exception:
        raise DirectConflictRankerError("direct_conflict_vector_source_unavailable") from None
    vectors_by_id: dict[str, Sequence[float]] = {}
    for vector_row in vector_rows:
        if vector_row.window_id in vectors_by_id:
            raise DirectConflictRankerError("direct_conflict_vector_duplicate")
        if vector_row.window_id not in target_labels:
            raise DirectConflictRankerError("direct_conflict_vector_set_mismatch")
        vectors_by_id[vector_row.window_id] = vector_row.vector
    if set(vectors_by_id) != set(private_targets):
        raise DirectConflictRankerError("direct_conflict_vector_set_incomplete")
    ordered_targets = sorted(private_targets)
    matrix = _validated_vector_matrix(
        [vectors_by_id[target] for target in ordered_targets],
        dimensions=context.dimensions,
        allow_empty=False,
    )
    examples = tuple(
        DirectConflictTrainingExample(
            private_window_id=target,
            private_conversation_group=context.private_conversation_groups[target],
            label=cast(DirectConflictFeedbackLabel, target_labels[target]),
            vector=matrix[index],
        )
        for index, target in enumerate(ordered_targets)
    )
    return DirectConflictTrainingData(
        examples=examples,
        dimensions=context.dimensions,
        private_config_material=context.private_config_material,
        feedback_revision=revision,
    )


def fit_direct_conflict_ranker(
    prepared: DirectConflictTrainingData,
    *,
    parameters: DirectConflictRankerParameters = DEFAULT_DIRECT_CONFLICT_PARAMETERS,
) -> DirectConflictRanker:
    if prepared.dimensions <= 0 or len(prepared.examples) > DIRECT_CONFLICT_MAX_FEEDBACK_ROWS:
        raise DirectConflictRankerError("direct_conflict_training_material_invalid")
    ordered_examples = tuple(
        sorted(prepared.examples, key=lambda example: example.private_window_id)
    )
    private_targets: set[str] = set()
    label_counts = {"relevant": 0, "not_relevant": 0, "uncertain": 0}
    for example in ordered_examples:
        if (
            not example.private_window_id
            or not example.private_conversation_group
            or example.label not in label_counts
        ):
            raise DirectConflictRankerError("direct_conflict_training_material_invalid")
        if example.private_window_id in private_targets:
            raise DirectConflictRankerError("direct_conflict_feedback_target_duplicate")
        private_targets.add(example.private_window_id)
        label_counts[example.label] += 1
    if (
        label_counts["relevant"] < DIRECT_CONFLICT_MIN_RELEVANT
        or label_counts["not_relevant"] < DIRECT_CONFLICT_MIN_NOT_RELEVANT
        or prepared.feedback_revision.row_count != len(ordered_examples)
    ):
        raise DirectConflictRankerError("direct_conflict_feedback_minimum_not_met")
    all_vectors = _validated_vector_matrix(
        [example.vector for example in ordered_examples],
        dimensions=prepared.dimensions,
        allow_empty=False,
    )
    normalized_examples = tuple(
        DirectConflictTrainingExample(
            private_window_id=example.private_window_id,
            private_conversation_group=example.private_conversation_group,
            label=example.label,
            vector=all_vectors[index],
        )
        for index, example in enumerate(ordered_examples)
    )
    normalized_prepared = DirectConflictTrainingData(
        examples=normalized_examples,
        dimensions=prepared.dimensions,
        private_config_material=prepared.private_config_material,
        feedback_revision=prepared.feedback_revision,
    )
    binary_indices = [
        index for index, example in enumerate(normalized_examples) if example.label != "uncertain"
    ]
    binary_examples = [normalized_examples[index] for index in binary_indices]
    matrix = np.ascontiguousarray(all_vectors[binary_indices], dtype=np.float32)
    labels = np.asarray(
        [1 if example.label == "relevant" else 0 for example in binary_examples],
        dtype=np.int64,
    )
    groups = [example.private_conversation_group for example in binary_examples]
    metrics = _validation_metrics(matrix, labels, groups, parameters)
    classifier = _linear_svc(parameters)
    try:
        classifier.fit(matrix, labels)
    except Exception:
        raise DirectConflictRankerError("direct_conflict_training_failed") from None
    fingerprint = _material_fingerprint(normalized_prepared, parameters, classifier)
    relevant = int(np.sum(labels))
    audit = DirectConflictRankerAudit(
        version=DIRECT_CONFLICT_RANKER_VERSION,
        parameters=parameters,
        counts=DirectConflictRankerCounts(
            reviewed_feedback_rows=len(normalized_examples),
            binary_training_rows=len(binary_examples),
            relevant_rows=relevant,
            not_relevant_rows=len(binary_examples) - relevant,
            uncertain_rows=len(normalized_examples) - len(binary_examples),
            conversation_groups=len(
                {example.private_conversation_group for example in normalized_examples}
            ),
        ),
        metrics=metrics,
        private_feedback_revision=normalized_prepared.feedback_revision,
        private_material_fingerprint_sha256=fingerprint,
    )
    return DirectConflictRanker(
        dimensions=prepared.dimensions,
        audit=audit,
        _classifier=classifier,
    )


def derive_direct_conflict_ranker(
    engine: Engine,
    vector_source: DirectConflictVectorSource,
    *,
    config_id: str | None = None,
    parameters: DirectConflictRankerParameters = DEFAULT_DIRECT_CONFLICT_PARAMETERS,
) -> DirectConflictRanker:
    return fit_direct_conflict_ranker(
        prepare_direct_conflict_training_data(engine, vector_source, config_id=config_id),
        parameters=parameters,
    )
