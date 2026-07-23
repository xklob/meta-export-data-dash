from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

from meta_archive.services.model_cache import pinned_transformers_snapshot

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANKER_MODEL_REVISION = "c5ee24cb16019beea0893ab7796b1df96625c6b8"
MAX_RERANK_CANDIDATES = 200
RERANK_BATCH_SIZE = 16
MAX_SEQUENCE_TOKENS = 512


@dataclass(frozen=True, slots=True)
class RerankerSpec:
    model_name: str = RERANKER_MODEL_NAME
    model_revision: str = RERANKER_MODEL_REVISION
    device: str = "cpu"
    score_kind: str = "cross_encoder_logit"
    max_candidates: int = MAX_RERANK_CANDIDATES
    batch_size: int = RERANK_BATCH_SIZE
    max_sequence_tokens: int = MAX_SEQUENCE_TOKENS


DEFAULT_RERANKER_SPEC = RerankerSpec()


class TextReranker(Protocol):
    @property
    def spec(self) -> RerankerSpec: ...

    @property
    def status(self) -> dict[str, Any]: ...

    def rerank(self, query: str, documents: Sequence[str]) -> list[float]: ...


RerankerFactory = Callable[[RerankerSpec], TextReranker]


def reranker_status_payload(
    *, status: str = "not_loaded", error: str | None = None
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        **asdict(DEFAULT_RERANKER_SPEC),
        "status": status,
        "lazy": True,
        "score_warning": (
            "Cross-encoder logits are ranking features, not calibrated probabilities."
        ),
    }
    if error:
        payload["error"] = error
    return payload


class TransformersCrossEncoderReranker:
    """Lazy, revision-pinned local cross-encoder backed by Hugging Face Transformers."""

    def __init__(
        self,
        cache_dir: Path,
        spec: RerankerSpec = DEFAULT_RERANKER_SPEC,
        *,
        local_files_only: bool = False,
    ) -> None:
        self.cache_dir = cache_dir
        self._spec = spec
        self._local_files_only = local_files_only
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._error: str | None = None

    @property
    def spec(self) -> RerankerSpec:
        return self._spec

    @property
    def status(self) -> dict[str, Any]:
        if self._error:
            return reranker_status_payload(status="unavailable", error=self._error)
        return reranker_status_payload(
            status="loaded" if self._model is not None else "ready_for_lazy_load"
        )

    def _load(self) -> tuple[Any, Any]:
        if self._tokenizer is None or self._model is None:
            try:
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
                self._tokenizer = AutoTokenizer.from_pretrained(
                    model_source,
                    **load_options,
                )
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    model_source,
                    **load_options,
                )
                self._model.to("cpu")
                self._model.eval()
            except Exception as exc:
                self._error = f"{type(exc).__name__}: {exc}"[:1_000]
                raise RuntimeError(
                    "The local cross-encoder reranker could not be loaded at its pinned revision"
                ) from exc
        return self._tokenizer, self._model

    def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        if len(documents) > self.spec.max_candidates:
            raise ValueError(
                f"Cross-encoder candidate pool exceeds {self.spec.max_candidates} documents"
            )
        if not documents:
            return []
        import torch

        tokenizer, model = self._load()
        scores: list[float] = []
        try:
            for start in range(0, len(documents), self.spec.batch_size):
                batch = list(documents[start : start + self.spec.batch_size])
                encoded = tokenizer(
                    [query] * len(batch),
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.spec.max_sequence_tokens,
                    return_tensors="pt",
                )
                with torch.inference_mode():
                    logits = model(**encoded).logits
                if logits.ndim != 2 or logits.shape[1] != 1:
                    raise RuntimeError("Cross-encoder returned an unexpected score shape")
                scores.extend(float(value) for value in logits[:, 0].cpu().tolist())
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"[:1_000]
            raise
        if len(scores) != len(documents) or any(not math.isfinite(score) for score in scores):
            raise RuntimeError("Cross-encoder returned invalid ranking scores")
        return scores


@lru_cache(maxsize=4)
def cached_cross_encoder_reranker(cache_dir: str) -> TransformersCrossEncoderReranker:
    """Cache one lazy model instance per archive-local model directory."""

    return TransformersCrossEncoderReranker(Path(cache_dir))
