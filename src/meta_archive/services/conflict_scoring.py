from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Protocol

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from meta_archive.models import SearchWindow
from meta_archive.paths import ArchiveLayout
from meta_archive.services.model_cache import pinned_transformers_snapshot
from meta_archive.services.reranking import RerankerFactory
from meta_archive.services.search import (
    EmbedderFactory,
    SearchFilters,
    SearchUnavailableError,
    semantic_search,
)

CONFLICT_MODEL_NAME = "tasksource/ModernBERT-base-nli"
CONFLICT_MODEL_REVISION = "de4ab7e77845098b7fab7f6ab9d370ddff27b19c"
MAX_CONFLICT_CANDIDATES = 100
CONFLICT_BATCH_SIZE = 16
MAX_SEQUENCE_TOKENS = 512
MAX_DOCUMENT_CHARACTERS = 20_000

ConflictLabel = Literal[
    "direct conflict",
    "calm disagreement",
    "banter",
    "third-party conflict",
    "no conflict",
    "uncertain",
]
CONFLICT_LABELS: tuple[ConflictLabel, ...] = (
    "direct conflict",
    "calm disagreement",
    "banter",
    "third-party conflict",
    "no conflict",
    "uncertain",
)
CONFLICT_HYPOTHESES: dict[ConflictLabel, str] = {
    "direct conflict": (
        "The speakers are directly arguing or expressing hostility toward each other."
    ),
    "calm disagreement": "The speakers disagree with each other while remaining calm and civil.",
    "banter": "The speakers are teasing or joking with each other as friendly banter.",
    "third-party conflict": (
        "The speakers are discussing a conflict involving another person or group."
    ),
    "no conflict": "The conversation does not contain interpersonal conflict or disagreement.",
    "uncertain": (
        "The context is insufficient or ambiguous about whether interpersonal conflict occurs."
    ),
}


@dataclass(frozen=True, slots=True)
class ConflictModelSpec:
    model_name: str = CONFLICT_MODEL_NAME
    model_revision: str = CONFLICT_MODEL_REVISION
    device: str = "cpu"
    quantization: str = "dynamic_int8"
    score_kind: str = "nli_entailment_logit"
    max_candidates: int = MAX_CONFLICT_CANDIDATES
    batch_size: int = CONFLICT_BATCH_SIZE
    max_sequence_tokens: int = MAX_SEQUENCE_TOKENS


DEFAULT_CONFLICT_SPEC = ConflictModelSpec()
CONFLICT_WARNINGS = (
    "Conflict scores are uncalibrated NLI ranking features, not probabilities.",
    "Banter, quotation, missing context, dialect, and repaired text can change the ranking.",
    "Scoring does not enable aggregate charts and must never rank people or relationships.",
)


@dataclass(frozen=True, slots=True)
class ConflictScoreVector:
    scores: dict[ConflictLabel, float]

    @property
    def predicted_label(self) -> ConflictLabel:
        return max(CONFLICT_LABELS, key=lambda label: self.scores[label])


class ConflictScorer(Protocol):
    @property
    def spec(self) -> ConflictModelSpec: ...

    @property
    def status(self) -> dict[str, Any]: ...

    def score(self, documents: Sequence[str]) -> list[ConflictScoreVector]: ...


ConflictScorerFactory = Callable[[ConflictModelSpec], ConflictScorer]


class ConflictScoringUnavailableError(SearchUnavailableError):
    pass


def conflict_model_status(
    *, status: str = "not_loaded", error: str | None = None
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        **asdict(DEFAULT_CONFLICT_SPEC),
        "status": status,
        "lazy": True,
        "labels": list(CONFLICT_LABELS),
        "hypotheses": dict(CONFLICT_HYPOTHESES),
        "warnings": list(CONFLICT_WARNINGS),
        "ranking_feature_only": True,
        "aggregation_gate_bypassed": False,
    }
    if error:
        payload["error"] = error
    return payload


class ModernBertNliConflictScorer:
    """Lazy, CPU dynamic-INT8 ModernBERT NLI scorer for bounded search candidates."""

    def __init__(
        self,
        cache_dir: Path,
        spec: ConflictModelSpec = DEFAULT_CONFLICT_SPEC,
        *,
        local_files_only: bool = False,
    ) -> None:
        self.cache_dir = cache_dir
        self._spec = spec
        self._local_files_only = local_files_only
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._entailment_index: int | None = None
        self._error: str | None = None

    @property
    def spec(self) -> ConflictModelSpec:
        return self._spec

    @property
    def status(self) -> dict[str, Any]:
        if self._error:
            return conflict_model_status(status="unavailable", error=self._error)
        return conflict_model_status(
            status="loaded" if self._model is not None else "ready_for_lazy_load"
        )

    def _load(self) -> tuple[Any, Any, int]:
        if self._tokenizer is None or self._model is None or self._entailment_index is None:
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                model_source: str | Path
                if self._local_files_only:
                    model_source = pinned_transformers_snapshot(
                        self.cache_dir,
                        model_name=self.spec.model_name,
                        model_revision=self.spec.model_revision,
                    )
                    load_options: dict[str, Any] = {"local_files_only": True}
                else:
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    model_source = self.spec.model_name
                    load_options = {
                        "revision": self.spec.model_revision,
                        "cache_dir": self.cache_dir,
                    }
                tokenizer = AutoTokenizer.from_pretrained(
                    model_source,
                    **load_options,
                )
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_source,
                    **load_options,
                )
                model.to("cpu")
                model.eval()
                label_mapping = {
                    int(index): str(label).casefold()
                    for index, label in model.config.id2label.items()
                }
                entailment_indices = [
                    index for index, label in label_mapping.items() if "entail" in label
                ]
                if len(entailment_indices) != 1:
                    raise RuntimeError("Pinned NLI model does not expose one entailment label")
                quantize_dynamic: Callable[..., Any] = torch.ao.quantization.quantize_dynamic
                model = quantize_dynamic(
                    model,
                    {torch.nn.Linear},
                    dtype=torch.qint8,
                )
                model.eval()
                self._tokenizer = tokenizer
                self._model = model
                self._entailment_index = entailment_indices[0]
            except Exception as exc:
                self._error = f"{type(exc).__name__}: {exc}"[:1_000]
                raise RuntimeError(
                    "The pinned local ModernBERT NLI conflict model could not be loaded"
                ) from exc
        return self._tokenizer, self._model, self._entailment_index

    def score(self, documents: Sequence[str]) -> list[ConflictScoreVector]:
        if len(documents) > self.spec.max_candidates:
            raise ValueError(f"Conflict candidate pool exceeds {self.spec.max_candidates} windows")
        if not documents:
            return []
        import torch

        tokenizer, model, entailment_index = self._load()
        pairs = [
            (document[:MAX_DOCUMENT_CHARACTERS], CONFLICT_HYPOTHESES[label])
            for document in documents
            for label in CONFLICT_LABELS
        ]
        raw_scores: list[float] = []
        try:
            for start in range(0, len(pairs), self.spec.batch_size):
                batch = pairs[start : start + self.spec.batch_size]
                encoded = tokenizer(
                    [premise for premise, _hypothesis in batch],
                    [hypothesis for _premise, hypothesis in batch],
                    padding=True,
                    truncation=True,
                    max_length=self.spec.max_sequence_tokens,
                    return_tensors="pt",
                )
                with torch.inference_mode():
                    logits = model(**encoded).logits
                if logits.ndim != 2 or logits.shape[1] <= entailment_index:
                    raise RuntimeError("Conflict NLI model returned an unexpected score shape")
                raw_scores.extend(
                    float(value) for value in logits[:, entailment_index].cpu().tolist()
                )
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"[:1_000]
            raise
        expected_count = len(documents) * len(CONFLICT_LABELS)
        if len(raw_scores) != expected_count or any(
            not math.isfinite(score) for score in raw_scores
        ):
            self._error = "Conflict NLI model returned invalid ranking scores"
            raise RuntimeError(self._error)
        return [
            ConflictScoreVector(
                scores={
                    label: raw_scores[(document_index * len(CONFLICT_LABELS)) + label_index]
                    for label_index, label in enumerate(CONFLICT_LABELS)
                }
            )
            for document_index in range(len(documents))
        ]


@lru_cache(maxsize=4)
def cached_conflict_scorer(cache_dir: str) -> ModernBertNliConflictScorer:
    return ModernBertNliConflictScorer(Path(cache_dir))


@dataclass(frozen=True, slots=True)
class ConflictSearchPage:
    items: list[dict[str, Any]]
    total_candidates: int
    retrieval_config: dict[str, Any] | None
    model_status: dict[str, Any]


_EMPTY_CONFLICT_FILTERS = SearchFilters(entity_type="message")


def conflict_search(
    layout: ArchiveLayout,
    engine: Engine,
    query: str,
    *,
    target_label: ConflictLabel = "direct conflict",
    filters: SearchFilters = _EMPTY_CONFLICT_FILTERS,
    limit: int = 25,
    embedder_factory: EmbedderFactory | None = None,
    reranker_factory: RerankerFactory | None = None,
    scorer_factory: ConflictScorerFactory | None = None,
    acceptance_read_only: bool = False,
) -> ConflictSearchPage:
    if target_label not in CONFLICT_LABELS:
        raise ValueError("Choose one of the six supported conflict labels")
    if limit < 1 or limit > MAX_CONFLICT_CANDIDATES:
        raise ValueError(f"Conflict result limit must be between 1 and {MAX_CONFLICT_CANDIDATES}")
    retrieval_limit = min(max(limit * 4, 25), MAX_CONFLICT_CANDIDATES)
    retrieval = semantic_search(
        layout,
        engine,
        query,
        filters=replace(filters, entity_type="message"),
        limit=retrieval_limit,
        embedder_factory=embedder_factory,
        reranker_factory=reranker_factory,
        acceptance_read_only=acceptance_read_only,
        literal_direct_conflict_ranking=False,
    )
    candidate_items = [
        item for item in retrieval.items if item.get("entity_type") == "message_window"
    ][:MAX_CONFLICT_CANDIDATES]
    if not candidate_items:
        return ConflictSearchPage(
            items=[],
            total_candidates=0,
            retrieval_config=retrieval.index_config,
            model_status=conflict_model_status(status="not_needed_empty_retrieval"),
        )
    window_ids = [str(item["window_id"]) for item in candidate_items]
    with Session(engine) as session:
        windows = {
            row.id: row.text_display
            for row in session.scalars(
                select(SearchWindow).where(
                    SearchWindow.id.in_(window_ids),
                    SearchWindow.entity_type == "message_window",
                )
            )
        }
    candidate_items = [item for item in candidate_items if str(item["window_id"]) in windows]
    documents = [windows[str(item["window_id"])] for item in candidate_items]
    try:
        scorer = (
            scorer_factory(DEFAULT_CONFLICT_SPEC)
            if scorer_factory is not None
            else cached_conflict_scorer(
                str(layout.models / "conflict" / DEFAULT_CONFLICT_SPEC.model_revision)
            )
        )
        if scorer.spec != DEFAULT_CONFLICT_SPEC:
            raise ConflictScoringUnavailableError(
                "The configured conflict scorer does not match the pinned model revision"
            )
        vectors = scorer.score(documents)
    except ConflictScoringUnavailableError:
        raise
    except Exception as exc:
        raise ConflictScoringUnavailableError(
            "Conflict scoring is unavailable; semantic results were not mislabeled as NLI-scored"
        ) from exc
    if len(vectors) != len(candidate_items):
        raise ConflictScoringUnavailableError("Conflict scorer returned the wrong result count")
    scored: list[tuple[float, int, str, dict[str, Any]]] = []
    for semantic_rank, (item, document, vector) in enumerate(
        zip(candidate_items, documents, vectors, strict=True), 1
    ):
        if set(vector.scores) != set(CONFLICT_LABELS) or any(
            not math.isfinite(score) for score in vector.scores.values()
        ):
            raise ConflictScoringUnavailableError(
                "Conflict scorer did not return all six finite ranking features"
            )
        payload = dict(item)
        payload["semantic_rank"] = semantic_rank
        payload["score"] = vector.scores[target_label]
        payload["score_kind"] = DEFAULT_CONFLICT_SPEC.score_kind
        payload["conflict_scoring"] = {
            "target_label": target_label,
            "predicted_label": vector.predicted_label,
            "scores": vector.scores,
            "score_kind": DEFAULT_CONFLICT_SPEC.score_kind,
            "model_name": DEFAULT_CONFLICT_SPEC.model_name,
            "model_revision": DEFAULT_CONFLICT_SPEC.model_revision,
            "quantization": DEFAULT_CONFLICT_SPEC.quantization,
            "ranking_feature_only": True,
            "aggregation_gate_bypassed": False,
            "hypotheses": dict(CONFLICT_HYPOTHESES),
            "provenance": {
                "generated_locally": True,
                "pipeline_version": "conflict-nli-ranking-v1",
                "model_name": DEFAULT_CONFLICT_SPEC.model_name,
                "model_revision": DEFAULT_CONFLICT_SPEC.model_revision,
                "quantization": DEFAULT_CONFLICT_SPEC.quantization,
                "input_character_count": len(document),
                "input_truncated": len(document) > MAX_DOCUMENT_CHARACTERS,
            },
            "warnings": list(CONFLICT_WARNINGS),
        }
        scored.append(
            (
                vector.scores[target_label],
                semantic_rank,
                str(item["window_id"]),
                payload,
            )
        )
    scored.sort(key=lambda row: (-row[0], row[1], row[2]))
    items = [row[3] for row in scored[:limit]]
    for rank, item in enumerate(items, 1):
        item["rank"] = rank
    return ConflictSearchPage(
        items=items,
        total_candidates=len(scored),
        retrieval_config=retrieval.index_config,
        model_status=scorer.status,
    )
