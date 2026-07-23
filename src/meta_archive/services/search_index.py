from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import stat
import threading
import unicodedata
import uuid
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, cast

import lancedb  # type: ignore[import-untyped]
import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
from lancedb.index import BTree, HnswSq  # type: ignore[import-untyped]
from sqlalchemy import Engine, and_, delete, desc, func, or_, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from meta_archive.config import settings
from meta_archive.models import (
    ActivityRecord,
    AttachmentOccurrence,
    AttachmentTextExtraction,
    AudioTranscription,
    AudioTranscriptSegment,
    ConversationSegment,
    Job,
    Message,
    Participant,
    RelevanceFeedback,
    SearchWindow,
    SearchWindowMessage,
    SegmentParticipant,
    SemanticIndexConfig,
    SemanticWindowEmbedding,
)
from meta_archive.paths import (
    ArchiveLayout,
    UnsafeArchivePathError,
    create_directory_beneath,
    open_directory_beneath,
)
from meta_archive.services.deletion_lock_lease import deletion_lock
from meta_archive.services.derived_publication_guard import (
    DerivedPublicationBlockedError,
    assert_derived_publication_allowed,
)
from meta_archive.services.direct_conflict_ranker import (
    DIRECT_CONFLICT_EVALUATION_INTENT,
    DIRECT_CONFLICT_MIN_NOT_RELEVANT,
    DIRECT_CONFLICT_MIN_RELEVANT,
    DIRECT_CONFLICT_RANKER_VERSION,
    DIRECT_CONFLICT_RANKING_K,
    DIRECT_CONFLICT_SCORE_KIND,
    DIRECT_CONFLICT_TRIGGER,
    DIRECT_CONFLICT_VALIDATION_FOLDS,
    DirectConflictRanker,
    DirectConflictRankerError,
    LanceDirectConflictVectorSource,
    derive_direct_conflict_ranker,
    development_feedback_revision_payload,
)
from meta_archive.services.identifiers import stable_id
from meta_archive.services.model_cache import (
    MODEL_ARTIFACT_FINGERPRINT_SCHEMA,
    pinned_transformers_artifact,
    pinned_transformers_snapshot,
)
from meta_archive.services.reranking import (
    DEFAULT_RERANKER_SPEC,
    MAX_RERANK_CANDIDATES,
    RerankerFactory,
    TextReranker,
    cached_cross_encoder_reranker,
)
from meta_archive.services.semantic_state import (
    SemanticTableIntegrityError,
    lance_table_names,
    validate_semantic_table,
)
from meta_archive.services.text import normalize_text

WINDOW_VERSION = "message-window-v1"
MIN_TURNS = 6
MAX_TURNS = 12
OVERLAP_TURNS = 2
GAP_MS = 30 * 60 * 1000
MAX_APPROX_TOKENS = 450
# Each search-window row currently binds 18 values. Forty rows stay below SQLite's historical
# 999-variable floor, even when one conversation produces tens of thousands of windows;
# associations use executemany separately.
WINDOW_UPSERT_BATCH_SIZE = 40
# Three values per row; 300 remains below SQLite's historical 999-variable floor.
SEMANTIC_MEMBERSHIP_BATCH_SIZE = 300
_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)

# Bump this whenever tokenization, truncation, pooling, normalization, or precision changes in a
# way that can alter vectors. Package/model revisions alone cannot prevent a resumed build from
# mixing vectors produced by two versions of our own embedding implementation.
EMBEDDING_PIPELINE_VERSION = "semantic-text-embedding-v1"
TRANSFORMERS_MAX_SEQUENCE_TOKENS = 512
TRANSFORMERS_POOLING = "attention-mask-mean"
TRANSFORMERS_PRECISION = "float32"
TRANSFORMERS_MICROBATCH_SIZE = 16
TRANSFORMERS_SORT_BUCKET_SIZE = 128
FASTEMBED_PROVIDER = "fastembed"
TRANSFORMERS_PROVIDER = "transformers"
INJECTED_TEXT_EMBEDDER_PROVIDER = "injected-text-embedder-contract"
PROVIDER_DECLARED_POOLING = "provider-declared"
CLS_POOLING = "cls"
INPUT_ORDER_BATCHING = "input-order"
STABLE_TOKEN_LENGTH_BATCHING = "stable-token-length-ascending-restored"
LEGACY_GTE_AUTO_EMBEDDING_MODEL_ALIAS = "gte-modernbert-base"
AUTO_EMBEDDING_MODEL_ALIAS = "bge-base-en-v1.5"
LEGACY_GTE_SEMANTIC_MODEL_SELECTION_POLICY = "english-gte-default-explicit-multilingual-override"
SEMANTIC_MODEL_SELECTION_POLICY = "english-default-explicit-multilingual-override-v1"

PREVIOUS_SEMANTIC_IR_EVALUATOR_VERSION = "semantic-ir-acceptance-v7"
SEMANTIC_IR_EVALUATOR_VERSION = "semantic-ir-acceptance-v8"
SEMANTIC_RANKING_PIPELINE_VERSION = "runtime-search-v2"
SEMANTIC_IR_K = 10
SEMANTIC_CANDIDATE_RECALL_KS = (50, 200)
SEMANTIC_UPGRADE_NON_REGRESSION_TOLERANCE = 0.02
SEMANTIC_UPGRADE_DEMONSTRATED_IMPROVEMENT = 0.02
SEMANTIC_UPGRADE_METRICS = ("recall_at_k", "mrr", "ndcg_at_k")
KNOWN_ITEM_QUERY_CAP = 48
FEEDBACK_QUERY_CAP = 64
FEEDBACK_JUDGMENT_CAP_PER_QUERY = 200
HELD_OUT_FEEDBACK_REVISION_VERSION = "held-out-feedback-revision-v1"
HELD_OUT_MIN_QUERY_COUNT = 5
KNOWN_ITEM_THRESHOLDS = {
    "recall_at_k": 0.95,
    "mrr": 0.90,
    "ndcg_at_k": 0.90,
}
HELD_OUT_THRESHOLDS = {
    "recall_at_k": 0.60,
    "mrr": 0.50,
    "ndcg_at_k": 0.55,
}
ARGUMENT_DIRECT_CONFLICT_INTENT = "argument_direct_conflict"
ARGUMENT_MIN_QUERY_GROUP_COUNT = 1
ARGUMENT_MIN_JUDGMENTS_PER_QUERY = 10
ARGUMENT_MIN_RELEVANT_JUDGMENTS_PER_QUERY = 2
ARGUMENT_MIN_NOT_RELEVANT_JUDGMENTS_PER_QUERY = 2
ARGUMENT_THRESHOLDS = dict(HELD_OUT_THRESHOLDS)
DIRECT_CONFLICT_DEVELOPMENT_THRESHOLDS = {
    "recall_at_10": HELD_OUT_THRESHOLDS["recall_at_k"],
    "mrr_at_10": HELD_OUT_THRESHOLDS["mrr"],
    "ndcg_at_10": HELD_OUT_THRESHOLDS["ndcg_at_k"],
}
DIRECT_CONFLICT_LITERAL_QUERY_SHA256 = hashlib.sha256(
    DIRECT_CONFLICT_TRIGGER.encode("utf-8")
).hexdigest()
DIRECT_CONFLICT_LITERAL_REGRESSION_NOTES_SHA256 = (
    "0b37b5f8b36ec24796a66289a4e2129291c2c25a0445e305b1dddee93a583072"
)
DIRECT_CONFLICT_LITERAL_REGRESSION_REVIEWED_COUNT = 20
DIRECT_CONFLICT_LITERAL_REGRESSION_RELEVANT_COUNT = 12
DIRECT_CONFLICT_LITERAL_REGRESSION_NOT_RELEVANT_COUNT = 8


@dataclass(frozen=True, slots=True)
class EmbeddingModelSpec:
    alias: str
    model_name: str
    model_revision: str
    dimensions: int
    document_prefix: str
    query_prefix: str
    language_scope: str
    approximate_download_gb: float
    provider: str = INJECTED_TEXT_EMBEDDER_PROVIDER
    pooling: str = PROVIDER_DECLARED_POOLING
    max_sequence_tokens: int | None = None
    precision: str = "provider-declared"
    batch_order: str = INPUT_ORDER_BATCHING


@dataclass(frozen=True, slots=True)
class FastEmbedArtifactSpec:
    repository: str
    revision: str
    model_file: str
    required_files: tuple[str, ...]
    custom_pooling: str | None = None
    custom_normalization: bool | None = None
    application_max_tokens: int | None = None


_FASTEMBED_REQUIRED_METADATA = (
    "config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


PINNED_FASTEMBED_ARTIFACTS: dict[str, FastEmbedArtifactSpec] = {
    "bge-base-en-v1.5": FastEmbedArtifactSpec(
        repository="qdrant/bge-base-en-v1.5-onnx-q",
        revision="738cad1c108e2f23649db9e44b2eab988626493b",
        model_file="model_optimized.onnx",
        required_files=(*_FASTEMBED_REQUIRED_METADATA, "model_optimized.onnx"),
    ),
    "mxbai-embed-large-v1": FastEmbedArtifactSpec(
        repository="mixedbread-ai/mxbai-embed-large-v1",
        revision="b33106f585b9ce46904ad7443a3b52b7a63e231c",
        model_file="onnx/model.onnx",
        required_files=(*_FASTEMBED_REQUIRED_METADATA, "onnx/model.onnx"),
    ),
    "nomic-embed-text-v1.5": FastEmbedArtifactSpec(
        repository="nomic-ai/nomic-embed-text-v1.5",
        revision="e9b6763023c676ca8431644204f50c2b100d9aab",
        model_file="onnx/model.onnx",
        required_files=(*_FASTEMBED_REQUIRED_METADATA, "onnx/model.onnx"),
        application_max_tokens=512,
    ),
    "e5-base-v2": FastEmbedArtifactSpec(
        repository="intfloat/e5-base-v2",
        revision="f52bf8ec8c7124536f0efb74aca902b2995e5bcd",
        model_file="onnx/model.onnx",
        required_files=(*_FASTEMBED_REQUIRED_METADATA, "onnx/model.onnx"),
        custom_pooling="MEAN",
        custom_normalization=True,
    ),
    "e5-small-v2": FastEmbedArtifactSpec(
        repository="intfloat/e5-small-v2",
        revision="ffb93f3bd4047442299a41ebb6fa998a38507c52",
        model_file="model.onnx",
        required_files=(*_FASTEMBED_REQUIRED_METADATA, "model.onnx"),
        custom_pooling="MEAN",
        custom_normalization=True,
    ),
    "snowflake-arctic-embed-l": FastEmbedArtifactSpec(
        repository="snowflake/snowflake-arctic-embed-l",
        revision="d8fb21ca8d905d2832ee8b96c894d3298964346b",
        model_file="onnx/model.onnx",
        required_files=(*_FASTEMBED_REQUIRED_METADATA, "onnx/model.onnx"),
    ),
}
_FASTEMBED_CUSTOM_MODEL_LOCK = threading.Lock()
_FASTEMBED_CUSTOM_MODEL_OWNERS: dict[str, object] = {}


MODEL_SPECS: dict[str, EmbeddingModelSpec] = {
    "bge-small-en-v1.5": EmbeddingModelSpec(
        alias="bge-small-en-v1.5",
        model_name="BAAI/bge-small-en-v1.5",
        model_revision="qdrant/bge-small-en-v1.5-onnx-q via fastembed-0.8",
        dimensions=384,
        document_prefix="",
        query_prefix="Represent this sentence for searching relevant passages: ",
        language_scope="English",
        approximate_download_gb=0.067,
        provider=FASTEMBED_PROVIDER,
        pooling=PROVIDER_DECLARED_POOLING,
        max_sequence_tokens=None,
        precision="provider-declared-onnx",
    ),
    "bge-base-en-v1.5": EmbeddingModelSpec(
        alias="bge-base-en-v1.5",
        model_name="BAAI/bge-base-en-v1.5",
        model_revision="738cad1c108e2f23649db9e44b2eab988626493b",
        dimensions=768,
        document_prefix="",
        query_prefix="Represent this sentence for searching relevant passages: ",
        language_scope="English",
        approximate_download_gb=0.22,
        provider=FASTEMBED_PROVIDER,
        pooling=CLS_POOLING,
        max_sequence_tokens=512,
        precision="float16-onnx",
    ),
    "e5-base-v2": EmbeddingModelSpec(
        alias="e5-base-v2",
        model_name="intfloat/e5-base-v2",
        model_revision="f52bf8ec8c7124536f0efb74aca902b2995e5bcd",
        dimensions=768,
        document_prefix="passage: ",
        query_prefix="query: ",
        language_scope="English",
        approximate_download_gb=0.44,
        provider=FASTEMBED_PROVIDER,
        pooling=TRANSFORMERS_POOLING,
        max_sequence_tokens=512,
        precision="float32-onnx",
    ),
    "e5-small-v2": EmbeddingModelSpec(
        alias="e5-small-v2",
        model_name="intfloat/e5-small-v2",
        model_revision="ffb93f3bd4047442299a41ebb6fa998a38507c52",
        dimensions=384,
        document_prefix="passage: ",
        query_prefix="query: ",
        language_scope="English",
        approximate_download_gb=0.134,
        provider=FASTEMBED_PROVIDER,
        pooling=TRANSFORMERS_POOLING,
        max_sequence_tokens=512,
        precision="float32-onnx",
    ),
    "mxbai-embed-large-v1": EmbeddingModelSpec(
        alias="mxbai-embed-large-v1",
        model_name="mixedbread-ai/mxbai-embed-large-v1",
        model_revision="b33106f585b9ce46904ad7443a3b52b7a63e231c",
        dimensions=1024,
        document_prefix="",
        query_prefix="Represent this sentence for searching relevant passages: ",
        language_scope="English",
        approximate_download_gb=1.34,
        provider=FASTEMBED_PROVIDER,
        pooling=CLS_POOLING,
        max_sequence_tokens=512,
        precision="float32-onnx",
    ),
    "multilingual-e5-small": EmbeddingModelSpec(
        alias="multilingual-e5-small",
        model_name="intfloat/multilingual-e5-small",
        model_revision="fd1525a9fd15316a2d503bf26ab031a61d056e98",
        dimensions=384,
        document_prefix="passage: ",
        query_prefix="query: ",
        language_scope="Multilingual, 94 languages",
        approximate_download_gb=0.47,
        provider=TRANSFORMERS_PROVIDER,
        pooling=TRANSFORMERS_POOLING,
        max_sequence_tokens=TRANSFORMERS_MAX_SEQUENCE_TOKENS,
        precision=TRANSFORMERS_PRECISION,
    ),
    "nomic-embed-text-v1.5": EmbeddingModelSpec(
        alias="nomic-embed-text-v1.5",
        model_name="nomic-ai/nomic-embed-text-v1.5",
        model_revision="e9b6763023c676ca8431644204f50c2b100d9aab",
        dimensions=768,
        document_prefix="search_document: ",
        query_prefix="search_query: ",
        language_scope="English",
        approximate_download_gb=0.55,
        provider=FASTEMBED_PROVIDER,
        pooling=TRANSFORMERS_POOLING,
        max_sequence_tokens=512,
        precision="float32-onnx",
    ),
    "gte-modernbert-base": EmbeddingModelSpec(
        alias="gte-modernbert-base",
        model_name="Alibaba-NLP/gte-modernbert-base",
        model_revision="e7f32e3c00f91d699e8c43b53106206bcc72bb22",
        dimensions=768,
        document_prefix="",
        query_prefix="",
        language_scope="English",
        approximate_download_gb=0.288,
        provider=TRANSFORMERS_PROVIDER,
        pooling=CLS_POOLING,
        max_sequence_tokens=TRANSFORMERS_MAX_SEQUENCE_TOKENS,
        precision=TRANSFORMERS_PRECISION,
        batch_order=STABLE_TOKEN_LENGTH_BATCHING,
    ),
    "snowflake-arctic-embed-l": EmbeddingModelSpec(
        alias="snowflake-arctic-embed-l",
        model_name="snowflake/snowflake-arctic-embed-l",
        model_revision="d8fb21ca8d905d2832ee8b96c894d3298964346b",
        dimensions=1024,
        document_prefix="",
        query_prefix="Represent this sentence for searching relevant passages: ",
        language_scope="English",
        approximate_download_gb=1.34,
        provider=FASTEMBED_PROVIDER,
        pooling=CLS_POOLING,
        max_sequence_tokens=512,
        precision="float32-onnx",
    ),
}


class TextEmbedder(Protocol):
    spec: EmbeddingModelSpec

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class TextEmbedderFactory(Protocol):
    def __call__(self, spec: EmbeddingModelSpec) -> TextEmbedder: ...


class SemanticIndexBusyError(RuntimeError):
    pass


class SemanticIndexAcceptanceError(RuntimeError):
    """Raised after a candidate is durably rejected by the retrieval quality gate."""

    def __init__(
        self,
        *,
        job_id: str,
        config_hash: str,
        evaluation: dict[str, Any],
    ) -> None:
        known_item = evaluation["automatic_known_item"]
        metrics = known_item["metrics"]
        k = metrics["k"]
        held_out = evaluation["held_out_feedback"]
        held_out_metrics = held_out["metrics"]
        held_out_detail = (
            f"; held-out Recall@{k}={held_out_metrics['recall_at_k']:.3f}, "
            f"MRR={held_out_metrics['mrr']:.3f}, "
            f"nDCG@{k}={held_out_metrics['ndcg_at_k']:.3f}"
            if held_out["used_for_activation"]
            else ""
        )
        super().__init__(
            "Semantic index candidate failed retrieval acceptance gate: "
            f"known-item Recall@{k}={metrics['recall_at_k']:.3f}, "
            f"MRR={metrics['mrr']:.3f}, nDCG@{k}={metrics['ndcg_at_k']:.3f}"
            f"{held_out_detail}"
        )
        self.job_id = job_id
        self.config_hash = config_hash
        self.evaluation = evaluation


class SemanticIndexManualActivationError(RuntimeError):
    """Raised when an explicit rejected-candidate activation fails closed."""


def _open_private_semantic_lock_file(layout: ArchiveLayout) -> tuple[int, int]:
    """Open the semantic lock without following or mutating an unsafe pathname."""

    if not all(hasattr(os, flag) for flag in ("O_DIRECTORY", "O_NOFOLLOW", "O_CLOEXEC")):
        raise OSError("Secure no-follow lock files are unavailable")
    directory_descriptor: int | None = None
    descriptor: int | None = None
    try:
        create_directory_beneath(layout.catalog, layout.root, mode=0o700)
        directory_descriptor = open_directory_beneath(layout.catalog, layout.root)
        descriptor = os.open(
            "semantic-index.lock",
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
            0o600,
            dir_fd=directory_descriptor,
        )
        descriptor_stat = os.fstat(descriptor)
        path_stat = os.stat(
            "semantic-index.lock", dir_fd=directory_descriptor, follow_symlinks=False
        )
        if (
            not stat.S_ISREG(descriptor_stat.st_mode)
            or descriptor_stat.st_uid != os.geteuid()
            or descriptor_stat.st_nlink != 1
            or (descriptor_stat.st_dev, descriptor_stat.st_ino)
            != (path_stat.st_dev, path_stat.st_ino)
        ):
            raise OSError("Semantic lock is not one owner-owned regular file")
        # A multiply linked inode is rejected before this repair so an external target can never
        # have its permissions changed through the lock pathname.
        os.fchmod(descriptor, 0o600)
        if stat.S_IMODE(os.fstat(descriptor).st_mode) != 0o600:
            raise OSError("Semantic lock permissions are not owner-only")
        return descriptor, directory_descriptor
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        if directory_descriptor is not None:
            os.close(directory_descriptor)
        raise


def _validate_private_semantic_lock_file(descriptor: int, directory_descriptor: int) -> None:
    descriptor_stat = os.fstat(descriptor)
    path_stat = os.stat("semantic-index.lock", dir_fd=directory_descriptor, follow_symlinks=False)
    if (
        not stat.S_ISREG(descriptor_stat.st_mode)
        or descriptor_stat.st_uid != os.geteuid()
        or descriptor_stat.st_nlink != 1
        or stat.S_IMODE(descriptor_stat.st_mode) != 0o600
        or (descriptor_stat.st_dev, descriptor_stat.st_ino) != (path_stat.st_dev, path_stat.st_ino)
    ):
        raise OSError("Semantic lock path changed after it was opened")


def semantic_index_lock_is_held(layout: ArchiveLayout) -> bool:
    """Return whether another process owns the build lock, failing closed on unsafe state."""

    try:
        descriptor, directory_descriptor = _open_private_semantic_lock_file(layout)
    except (OSError, UnsafeArchivePathError):
        return True
    acquired = False
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError:
            return True
        try:
            _validate_private_semantic_lock_file(descriptor, directory_descriptor)
        except OSError:
            return True
        return False
    finally:
        if acquired:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
        os.close(directory_descriptor)


def queue_semantic_index_job(
    engine: Engine,
    *,
    model_alias: str | None,
    rebuild_windows: bool,
) -> str:
    """Create the observable job before any expensive semantic-index preflight work."""

    job_id = str(uuid.uuid4())
    with Session(engine) as session, session.begin():
        session.add(
            Job(
                id=job_id,
                kind="semantic_index",
                status="queued",
                stage="Semantic index queued",
                progress_current=0,
                checkpoint_json={
                    "requested_model": model_alias or "auto",
                    "rebuild_windows": rebuild_windows,
                },
                counters_json={},
            )
        )
    return job_id


@contextmanager
def semantic_index_lock(layout: ArchiveLayout) -> Iterator[None]:
    descriptor: int | None = None
    directory_descriptor: int | None = None
    try:
        descriptor, directory_descriptor = _open_private_semantic_lock_file(layout)
    except (OSError, UnsafeArchivePathError) as exc:
        raise SemanticIndexBusyError("The semantic index lock file is unsafe") from exc
    try:
        with os.fdopen(descriptor, "r+", encoding="utf-8") as handle:
            descriptor = None
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise SemanticIndexBusyError(
                    "Another semantic index build is already running"
                ) from exc
            try:
                assert directory_descriptor is not None
                _validate_private_semantic_lock_file(handle.fileno(), directory_descriptor)
                handle.seek(0)
                handle.truncate()
                json.dump(
                    {"pid": os.getpid(), "started_at": datetime.now(UTC).isoformat()},
                    handle,
                    sort_keys=True,
                )
                handle.flush()
                os.fsync(handle.fileno())
            except OSError as exc:
                raise SemanticIndexBusyError("The semantic index lock file became unsafe") from exc
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if directory_descriptor is not None:
            os.close(directory_descriptor)


def _fastembed_artifact_identity(artifact: FastEmbedArtifactSpec) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "kind": "huggingface_snapshot",
        "repository": artifact.repository,
        "revision": artifact.revision,
        "model_file": artifact.model_file,
        "required_files": list(artifact.required_files),
        "fingerprint_schema": MODEL_ARTIFACT_FINGERPRINT_SCHEMA,
    }
    if artifact.custom_pooling is not None:
        identity["fastembed_custom_model"] = {
            "pooling": artifact.custom_pooling,
            "normalization": artifact.custom_normalization,
        }
    if artifact.application_max_tokens is not None:
        identity["application_max_tokens"] = artifact.application_max_tokens
    return identity


def _validate_fastembed_artifact_declaration(artifact: FastEmbedArtifactSpec) -> None:
    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*",
        artifact.repository,
    ):
        raise ValueError("Pinned FastEmbed artifact repository is invalid")
    if not re.fullmatch(r"[0-9a-f]{40}", artifact.revision):
        raise ValueError("Pinned FastEmbed artifact revision must be a full commit SHA")
    if not artifact.required_files or len(set(artifact.required_files)) != len(
        artifact.required_files
    ):
        raise ValueError("Pinned FastEmbed artifact required files are invalid")
    for raw_path in artifact.required_files:
        path = PurePosixPath(raw_path)
        if (
            not raw_path
            or "\\" in raw_path
            or path.is_absolute()
            or not path.parts
            or path.as_posix() != raw_path
            or ".." in path.parts
        ):
            raise ValueError("Pinned FastEmbed artifact path is unsafe")
    if artifact.model_file not in artifact.required_files:
        raise ValueError("Pinned FastEmbed artifact does not include its model file")
    if artifact.application_max_tokens is not None and artifact.application_max_tokens < 1:
        raise ValueError("Pinned FastEmbed application token limit must be positive")
    custom_values = (artifact.custom_pooling, artifact.custom_normalization)
    if custom_values == (None, None):
        return
    if artifact.custom_pooling != "MEAN" or artifact.custom_normalization is not True:
        raise ValueError("Pinned FastEmbed custom-model postprocessing is unsupported")


def _ensure_fastembed_custom_model(
    text_embedding_class: Any,
    spec: EmbeddingModelSpec,
    artifact: FastEmbedArtifactSpec,
) -> None:
    if artifact.custom_pooling is None:
        return
    from fastembed.common.model_description import ModelSource, PoolingType

    with _FASTEMBED_CUSTOM_MODEL_LOCK:
        candidates = [
            candidate
            for candidate in text_embedding_class.list_supported_models()
            if candidate.get("model") == spec.model_name
        ]
        owner = _FASTEMBED_CUSTOM_MODEL_OWNERS.get(spec.model_name)
        if candidates:
            if owner is not text_embedding_class:
                raise RuntimeError(
                    "FastEmbed custom model alias was registered outside this loader"
                )
            return
        if owner is text_embedding_class:
            raise RuntimeError("FastEmbed custom model registration disappeared")
        text_embedding_class.add_custom_model(
            model=spec.model_name,
            pooling=PoolingType.MEAN,
            normalization=True,
            sources=ModelSource(hf=artifact.repository),
            dim=spec.dimensions,
            model_file=artifact.model_file,
            description=f"English {spec.alias} retrieval embeddings",
            license="mit",
            size_in_gb=spec.approximate_download_gb,
        )
        _FASTEMBED_CUSTOM_MODEL_OWNERS[spec.model_name] = text_embedding_class


def _validate_fastembed_registry_artifact(
    text_embedding_class: Any,
    spec: EmbeddingModelSpec,
    artifact: FastEmbedArtifactSpec,
) -> None:
    candidates = [
        candidate
        for candidate in text_embedding_class.list_supported_models()
        if candidate.get("model") == spec.model_name
    ]
    if len(candidates) != 1:
        raise RuntimeError("Installed FastEmbed does not expose the pinned model alias")
    candidate = candidates[0]
    sources = candidate.get("sources")
    source_repository = sources.get("hf") if isinstance(sources, dict) else None
    if (
        source_repository != artifact.repository
        or candidate.get("model_file") != artifact.model_file
        or candidate.get("dim") != spec.dimensions
    ):
        raise RuntimeError("Installed FastEmbed artifact declaration does not match the pin")


def _pinned_fastembed_artifact_spec(
    spec: EmbeddingModelSpec,
) -> FastEmbedArtifactSpec | None:
    artifact = PINNED_FASTEMBED_ARTIFACTS.get(spec.alias)
    if artifact is None:
        return None
    _validate_fastembed_artifact_declaration(artifact)
    if spec.provider != FASTEMBED_PROVIDER:
        raise ValueError("Pinned FastEmbed artifacts require the FastEmbed provider")
    if spec.model_revision != artifact.revision:
        raise ValueError("Pinned FastEmbed model revision does not match its artifact revision")
    if artifact.custom_pooling == "MEAN" and spec.pooling != TRANSFORMERS_POOLING:
        raise ValueError("Pinned FastEmbed mean-pooling model specification is inconsistent")
    if (
        artifact.application_max_tokens is not None
        and spec.max_sequence_tokens != artifact.application_max_tokens
    ):
        raise ValueError("Pinned FastEmbed application token limit is inconsistent")
    return artifact


def _pinned_fastembed_snapshot(
    spec: EmbeddingModelSpec,
    *,
    cache_dir: Path,
    local_files_only: bool,
) -> Path:
    artifact = _pinned_fastembed_artifact_spec(spec)
    if artifact is None:
        raise ValueError("Embedding model does not declare a pinned FastEmbed artifact")
    if local_files_only:
        return pinned_transformers_snapshot(
            cache_dir,
            model_name=artifact.repository,
            model_revision=artifact.revision,
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download

    downloaded = Path(
        snapshot_download(
            repo_id=artifact.repository,
            revision=artifact.revision,
            cache_dir=cache_dir,
            allow_patterns=list(artifact.required_files),
        )
    )
    snapshot = pinned_transformers_snapshot(
        cache_dir,
        model_name=artifact.repository,
        model_revision=artifact.revision,
    )
    if downloaded.resolve() != snapshot.resolve():
        raise RuntimeError("Pinned FastEmbed download resolved to an unexpected snapshot")
    return snapshot


class FastEmbedTextEmbedder:
    def __init__(
        self,
        spec: EmbeddingModelSpec,
        *,
        cache_dir: str,
        local_files_only: bool = False,
    ) -> None:
        artifact_spec = _pinned_fastembed_artifact_spec(spec)
        if local_files_only and artifact_spec is None:
            # FastEmbed's installed cache manager can create/delete temporary cache entries even
            # when asked for local-only resolution. Acceptance mode must never invoke it.
            raise RuntimeError(
                "Acceptance read-only mode cannot safely initialize a FastEmbed cache"
            )
        from fastembed import TextEmbedding

        self.spec = spec
        if artifact_spec is not None:
            _ensure_fastembed_custom_model(TextEmbedding, spec, artifact_spec)
            _validate_fastembed_registry_artifact(TextEmbedding, spec, artifact_spec)
        model_options: dict[str, Any] = {
            "model_name": spec.model_name,
            "cache_dir": cache_dir,
            "threads": max(1, min(8, settings.embedding_batch_size // 16)),
        }
        pinned_before = None
        snapshot = None
        if artifact_spec is not None:
            cache_path = Path(cache_dir)
            snapshot = _pinned_fastembed_snapshot(
                spec,
                cache_dir=cache_path,
                local_files_only=local_files_only,
            )
            pinned_before = pinned_transformers_artifact(
                cache_path,
                model_name=artifact_spec.repository,
                model_revision=artifact_spec.revision,
            )
            missing = [
                required
                for required in artifact_spec.required_files
                if not (snapshot / required).is_file()
            ]
            if missing:
                raise RuntimeError(
                    "Pinned FastEmbed snapshot is incomplete: " + ", ".join(sorted(missing))
                )
            model_options["specific_model_path"] = str(snapshot)
            # The explicit path already bypasses FastEmbed's cache manager. Keep its own
            # local-only guard enabled as defense in depth for lazy loads and worker processes.
            model_options["local_files_only"] = True
        self._model = TextEmbedding(**model_options)
        model = getattr(self._model, "model", None)
        if artifact_spec is not None and artifact_spec.application_max_tokens is not None:
            tokenizer = getattr(model, "tokenizer", None)
            enable_truncation = getattr(tokenizer, "enable_truncation", None)
            if not callable(enable_truncation):
                raise RuntimeError("FastEmbed did not expose its tokenizer truncation control")
            enable_truncation(max_length=artifact_spec.application_max_tokens)
            truncation = getattr(tokenizer, "truncation", None)
            if (
                not isinstance(truncation, dict)
                or truncation.get("max_length") != artifact_spec.application_max_tokens
            ):
                raise RuntimeError("FastEmbed did not apply the required token limit")
        model_dir = getattr(model, "_model_dir", None)
        self._model_dir = Path(model_dir).resolve() if model_dir else None
        self._artifact_fingerprint: str | None = None
        if artifact_spec is not None:
            assert snapshot is not None
            assert pinned_before is not None
            if self._model_dir != snapshot.resolve():
                raise RuntimeError("FastEmbed did not load the exact pinned snapshot")
            pinned_after = pinned_transformers_artifact(
                Path(cache_dir),
                model_name=artifact_spec.repository,
                model_revision=artifact_spec.revision,
            )
            if pinned_after != pinned_before:
                raise RuntimeError("Pinned FastEmbed snapshot changed while loading")
            self._artifact_fingerprint = f"sha256:{pinned_after.fingerprint_sha256}"

    @staticmethod
    def _normalized(vector: Any) -> list[float]:
        values = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(values))
        if norm:
            values = values / norm
        return cast(list[float], values.tolist())

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        prepared = [f"{self.spec.document_prefix}{text}" for text in texts]
        return [
            self._normalized(vector)
            for vector in self._model.embed(
                prepared,
                batch_size=settings.embedding_batch_size,
                parallel=None,
            )
        ]

    def embed_query(self, text: str) -> list[float]:
        vector = next(iter(self._model.embed([f"{self.spec.query_prefix}{text}"])))
        return self._normalized(vector)

    def artifact_fingerprint(self) -> str:
        if self._artifact_fingerprint is not None:
            return self._artifact_fingerprint
        if self._model_dir is None or not self._model_dir.is_dir():
            raise RuntimeError("FastEmbed did not expose a downloaded model directory")
        digest = hashlib.sha256()
        files = sorted(path for path in self._model_dir.rglob("*") if path.is_file())
        if not files:
            raise RuntimeError("FastEmbed model directory is empty")
        for path in files:
            relative = path.relative_to(self._model_dir).as_posix()
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(path.stat().st_size).encode("ascii"))
            digest.update(b"\0")
            with path.open("rb") as handle:
                while chunk := handle.read(8 * 1024 * 1024):
                    digest.update(chunk)
            digest.update(b"\n")
        self._artifact_fingerprint = f"sha256:{digest.hexdigest()}"
        return self._artifact_fingerprint


class TransformersTextEmbedder:
    """Revision-pinned CPU encoder for models FastEmbed does not package."""

    def __init__(
        self,
        spec: EmbeddingModelSpec,
        *,
        cache_dir: str,
        local_files_only: bool = False,
    ) -> None:
        if spec.provider != TRANSFORMERS_PROVIDER:
            raise ValueError("Transformers embedder requires a Transformers model specification")
        if spec.pooling not in {TRANSFORMERS_POOLING, CLS_POOLING}:
            raise ValueError(f"Unsupported Transformers pooling strategy: {spec.pooling}")
        if spec.max_sequence_tokens is None or spec.max_sequence_tokens < 1:
            raise ValueError("Transformers model specification requires a positive token limit")
        if spec.precision != TRANSFORMERS_PRECISION:
            raise ValueError("Transformers text embeddings currently require float32 precision")
        if spec.batch_order not in {INPUT_ORDER_BATCHING, STABLE_TOKEN_LENGTH_BATCHING}:
            raise ValueError(f"Unsupported Transformers batching order: {spec.batch_order}")
        self.spec = spec
        self._cache_dir = Path(cache_dir)
        model_source: str | Path
        if local_files_only:
            model_source = pinned_transformers_snapshot(
                self._cache_dir,
                model_name=spec.model_name,
                model_revision=spec.model_revision,
            )
            load_options: dict[str, Any] = {"local_files_only": True}
        else:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            model_source = spec.model_name
            load_options = {
                "revision": spec.model_revision,
                "cache_dir": self._cache_dir,
            }
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            **load_options,
        )
        self._model = AutoModel.from_pretrained(
            model_source,
            dtype=torch.float32,
            **load_options,
        )
        self._model.to("cpu")
        self._model.float()
        self._model.eval()
        torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
        self._artifact_fingerprint: str | None = None

    def _token_lengths(self, texts: Sequence[str]) -> list[int]:
        encoded = self._tokenizer(
            list(texts),
            add_special_tokens=True,
            max_length=self.spec.max_sequence_tokens,
            padding=False,
            truncation=True,
            return_length=True,
        )
        raw_lengths = encoded.get("length")
        if raw_lengths is None:
            raise RuntimeError("Tokenizer did not return input lengths for stable batching")
        if hasattr(raw_lengths, "tolist"):
            raw_lengths = raw_lengths.tolist()
        lengths = [int(length) for length in raw_lengths]
        if len(lengths) != len(texts):
            raise RuntimeError("Tokenizer returned an invalid number of input lengths")
        return lengths

    def _embed_ordered(self, texts: Sequence[str]) -> list[list[float]]:
        import torch

        vectors: list[list[float]] = []
        for offset in range(0, len(texts), TRANSFORMERS_MICROBATCH_SIZE):
            batch = list(texts[offset : offset + TRANSFORMERS_MICROBATCH_SIZE])
            encoded = self._tokenizer(
                batch,
                max_length=self.spec.max_sequence_tokens,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            with torch.inference_mode():
                hidden = self._model(**encoded).last_hidden_state
            if self.spec.pooling == CLS_POOLING:
                pooled = hidden[:, 0]
            else:
                attention = encoded["attention_mask"].unsqueeze(-1).bool()
                masked = hidden.masked_fill(~attention, 0.0)
                pooled = masked.sum(dim=1) / attention.sum(dim=1).clamp(min=1)
            normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.extend(cast(list[list[float]], normalized.cpu().tolist()))
        return vectors

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        if self.spec.batch_order == INPUT_ORDER_BATCHING or len(texts) < 2:
            return self._embed_ordered(texts)

        vectors: list[list[float]] = []
        # Fixed 128-text buckets make the emitted vectors invariant between the production
        # outer batch sizes (128 and 512), while length-sorted microbatches sharply reduce
        # padding. Each bucket is restored to caller order before it is returned.
        for offset in range(0, len(texts), TRANSFORMERS_SORT_BUCKET_SIZE):
            bucket = list(texts[offset : offset + TRANSFORMERS_SORT_BUCKET_SIZE])
            lengths = self._token_lengths(bucket)
            order = sorted(range(len(bucket)), key=lengths.__getitem__)
            ordered_vectors = self._embed_ordered([bucket[index] for index in order])
            vectors.extend(
                vector
                for _index, vector in sorted(
                    zip(order, ordered_vectors, strict=True),
                    key=lambda item: item[0],
                )
            )
        return vectors

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed([f"{self.spec.document_prefix}{text}" for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._embed([f"{self.spec.query_prefix}{text}"])[0]

    def artifact_fingerprint(self) -> str:
        if self._artifact_fingerprint is not None:
            return self._artifact_fingerprint
        # Transformers loads by pathname, so callers must keep this archive-local cache
        # quiescent while the model is loaded and fingerprinted. The descriptor-bound helper
        # below rejects unsafe links and concurrent ordinary mutations and hashes exact bytes.
        artifact = pinned_transformers_artifact(
            self._cache_dir,
            model_name=self.spec.model_name,
            model_revision=self.spec.model_revision,
        )
        self._artifact_fingerprint = f"sha256:{artifact.fingerprint_sha256}"
        return self._artifact_fingerprint


def default_text_embedder(
    layout: ArchiveLayout,
    spec: EmbeddingModelSpec,
    *,
    local_files_only: bool = False,
) -> TextEmbedder:
    if spec.provider == TRANSFORMERS_PROVIDER:
        if local_files_only:
            return TransformersTextEmbedder(
                spec,
                cache_dir=str(layout.models / "transformers"),
                local_files_only=True,
            )
        return TransformersTextEmbedder(
            spec,
            cache_dir=str(layout.models / "transformers"),
        )
    if spec.provider not in {FASTEMBED_PROVIDER, INJECTED_TEXT_EMBEDDER_PROVIDER}:
        raise ValueError(f"Unsupported embedding provider: {spec.provider}")
    if local_files_only:
        return FastEmbedTextEmbedder(
            spec,
            cache_dir=str(layout.models / "fastembed"),
            local_files_only=True,
        )
    return FastEmbedTextEmbedder(
        spec,
        cache_dir=str(layout.models / "fastembed"),
    )


def _validated_embedding_batch_size(
    spec: EmbeddingModelSpec,
    requested_batch_size: int | None,
) -> int:
    actual_batch_size = (
        settings.embedding_batch_size if requested_batch_size is None else requested_batch_size
    )
    if actual_batch_size < 1:
        raise ValueError("Semantic embedding batch size must be positive")
    if (
        spec.batch_order == STABLE_TOKEN_LENGTH_BATCHING
        and actual_batch_size % TRANSFORMERS_SORT_BUCKET_SIZE != 0
    ):
        raise ValueError(
            "Length-sorted semantic embedding batch size must be divisible by "
            f"{TRANSFORMERS_SORT_BUCKET_SIZE}"
        )
    return actual_batch_size


@dataclass(frozen=True, slots=True)
class WindowTurn:
    id: str
    timestamp_ms: int
    sender_display: str
    sender_search: str
    text_display: str
    segment_id: str
    source_category: str


@dataclass(frozen=True, slots=True)
class WindowDraft:
    id: str
    conversation_id: str
    segment_id: str
    source_category: str
    message_ids: tuple[str, ...]
    first_timestamp_ms: int
    last_timestamp_ms: int
    text_display: str
    text_search: str
    participants_search: str
    approx_token_count: int
    source_char_start: int | None = None
    source_char_end: int | None = None
    chunk_ordinal: int | None = None
    chunk_count: int | None = None


@dataclass(slots=True)
class WindowBuildSummary:
    generation: str
    message_window_count: int = 0
    activity_window_count: int = 0
    derived_window_count: int = 0
    association_count: int = 0
    short_window_count: int = 0
    non_latin_letter_count: int = 0
    total_letter_count: int = 0

    @property
    def window_count(self) -> int:
        return self.message_window_count + self.activity_window_count + self.derived_window_count

    @property
    def non_latin_ratio(self) -> float:
        if not self.total_letter_count:
            return 0.0
        return self.non_latin_letter_count / self.total_letter_count

    @property
    def recommended_model(self) -> str:
        return AUTO_EMBEDDING_MODEL_ALIAS


@dataclass(frozen=True, slots=True)
class SemanticIndexSummary:
    job_id: str
    config_hash: str
    vector_table: str
    window_count: int
    embedded_count: int
    resumed: bool
    model_name: str


@dataclass(frozen=True, slots=True)
class SemanticManualActivationSummary:
    activation_job_id: str
    config_hash: str
    previous_active_config_id: str
    vector_table: str
    window_count: int
    model_name: str
    invalidated_conflict_run_count: int


@dataclass(frozen=True, slots=True)
class _SemanticEvaluationRanking:
    cosine_ids: tuple[str, ...]
    reranked_ids: tuple[str, ...]


def embedding_model_spec(alias: str | None = None) -> EmbeddingModelSpec:
    selected = alias or settings.embedding_model
    if selected == "auto":
        raise ValueError("The automatic model choice requires a completed language audit")
    try:
        return MODEL_SPECS[selected]
    except KeyError as exc:
        supported = ", ".join(sorted(MODEL_SPECS))
        raise ValueError(
            f"Unknown embedding model {selected!r}; choose one of: {supported}"
        ) from exc


def embedding_config_hash(
    spec: EmbeddingModelSpec,
    *,
    corpus_revision: str = "unbuilt",
    artifact_fingerprint: str | None = None,
) -> str:
    backend, backend_version = embedding_backend(spec)
    payload = {
        "backend": backend,
        "backend_version": backend_version,
        "embedding_implementation": embedding_implementation_identity(spec),
        "model": asdict(spec),
        "window_version": WINDOW_VERSION,
        "distance": "cosine",
        "corpus_revision": corpus_revision,
        "artifact_fingerprint": artifact_fingerprint or f"declared:{spec.model_revision}",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def embedding_implementation_identity(spec: EmbeddingModelSpec) -> dict[str, Any]:
    """Describe every application-owned semantic that can change emitted vectors."""

    common: dict[str, Any] = {
        "pipeline_version": EMBEDDING_PIPELINE_VERSION,
        "document_prefix": spec.document_prefix,
        "query_prefix": spec.query_prefix,
        "normalization": "l2",
    }
    if spec.provider == TRANSFORMERS_PROVIDER:
        if spec.pooling not in {TRANSFORMERS_POOLING, CLS_POOLING}:
            raise ValueError(f"Unsupported Transformers pooling strategy: {spec.pooling}")
        if spec.max_sequence_tokens is None or spec.max_sequence_tokens < 1:
            raise ValueError("Transformers model specification requires a positive token limit")
        if spec.batch_order not in {INPUT_ORDER_BATCHING, STABLE_TOKEN_LENGTH_BATCHING}:
            raise ValueError(f"Unsupported Transformers batching order: {spec.batch_order}")
        identity = {
            **common,
            "provider": TRANSFORMERS_PROVIDER,
            "device": "cpu",
            "precision": spec.precision,
            "pooling": spec.pooling,
            "max_sequence_tokens": spec.max_sequence_tokens,
            "truncation": True,
        }
        if spec.batch_order == STABLE_TOKEN_LENGTH_BATCHING:
            identity["document_batching"] = {
                "microbatch_size": TRANSFORMERS_MICROBATCH_SIZE,
                "sort_bucket_size": TRANSFORMERS_SORT_BUCKET_SIZE,
                "sort_order": STABLE_TOKEN_LENGTH_BATCHING,
                "restore_input_order": True,
            }
        return identity
    if spec.provider == FASTEMBED_PROVIDER:
        fastembed_identity: dict[str, Any] = {
            **common,
            "provider": FASTEMBED_PROVIDER,
            "precision": spec.precision,
            "pooling": spec.pooling,
        }
        artifact = _pinned_fastembed_artifact_spec(spec)
        if artifact is not None:
            fastembed_identity["artifact_source"] = _fastembed_artifact_identity(artifact)
            fastembed_identity["max_sequence_tokens"] = spec.max_sequence_tokens
            fastembed_identity["truncation"] = True
        return fastembed_identity
    if spec.provider == INJECTED_TEXT_EMBEDDER_PROVIDER:
        return {
            **common,
            "provider": INJECTED_TEXT_EMBEDDER_PROVIDER,
            "precision": spec.precision,
            "pooling": spec.pooling,
        }
    raise ValueError(f"Unsupported embedding provider: {spec.provider}")


def embedding_backend(spec: EmbeddingModelSpec) -> tuple[str, str]:
    """Return the recorded implementation identity for an embedding specification."""

    if spec.provider == TRANSFORMERS_PROVIDER:
        return (
            "lancedb-transformers",
            f"lancedb-{version('lancedb')}:transformers-{version('transformers')}:torch-{version('torch')}",
        )
    if spec.provider in {FASTEMBED_PROVIDER, INJECTED_TEXT_EMBEDDER_PROVIDER}:
        return (
            "lancedb-fastembed",
            f"lancedb-{version('lancedb')}:fastembed-{version('fastembed')}",
        )
    raise ValueError(f"Unsupported embedding provider: {spec.provider}")


def embedding_artifact_fingerprint(embedder: TextEmbedder) -> str:
    provider = getattr(embedder, "artifact_fingerprint", None)
    if callable(provider):
        value = provider()
        if not isinstance(value, str) or not value:
            raise RuntimeError("Embedding artifact fingerprint must be a non-empty string")
        return value
    # Injected/test embedders must still produce a stable audit identity without requiring local
    # model files. Production FastEmbed always follows the content-hash branch above.
    return f"declared:{embedder.spec.model_revision}"


def retrieval_metrics(
    rankings: Sequence[Sequence[str]],
    relevant_ids: Sequence[set[str] | frozenset[str]],
    *,
    k: int,
) -> dict[str, int | float]:
    """Compute macro-averaged binary Recall@K, MRR, and nDCG@K."""

    if k < 1:
        raise ValueError("Retrieval metric K must be positive")
    if len(rankings) != len(relevant_ids):
        raise ValueError("Each ranking must have a corresponding relevance set")
    if not rankings:
        return {
            "query_count": 0,
            "k": k,
            "recall_at_k": 0.0,
            "mrr": 0.0,
            "ndcg_at_k": 0.0,
        }

    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    normalized_discounted_gains: list[float] = []
    for ranking, relevant in zip(rankings, relevant_ids, strict=True):
        if not relevant:
            raise ValueError("Retrieval metrics require at least one relevant result per query")
        top_k = list(ranking[:k])
        relevant_ranks = [
            rank for rank, result_id in enumerate(top_k, start=1) if result_id in relevant
        ]
        retrieved_relevant = {result_id for result_id in top_k if result_id in relevant}
        recalls.append(len(retrieved_relevant) / len(relevant))
        reciprocal_ranks.append(1.0 / relevant_ranks[0] if relevant_ranks else 0.0)
        discounted_gain = sum(1.0 / math.log2(rank + 1) for rank in relevant_ranks)
        ideal_gain = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
        normalized_discounted_gains.append(discounted_gain / ideal_gain)

    query_count = len(rankings)
    return {
        "query_count": query_count,
        "k": k,
        "recall_at_k": sum(recalls) / query_count,
        "mrr": sum(reciprocal_ranks) / query_count,
        "ndcg_at_k": sum(normalized_discounted_gains) / query_count,
    }


def _retrieved_window_rows(
    table: Any,
    embedder: TextEmbedder,
    query_text: str,
    *,
    k: int,
    lance_filter: str | None = None,
    include_vectors: bool = False,
) -> list[dict[str, Any]]:
    vector = embedder.embed_query(query_text)
    _validate_embedding_vectors(
        [vector],
        expected_count=1,
        dimensions=embedder.spec.dimensions,
    )
    # The archive-scale acceptance benchmark found that LanceDB's approximate SQ index can
    # omit exact-text equivalents even when their stored vectors are valid.  At the supported
    # corpus size an exact Lance scan remains comfortably inside the semantic-search latency
    # budget, so retrieval quality takes precedence over an unnecessary approximation.
    query = (
        table.search(vector, vector_column_name="vector")
        .distance_type("cosine")
        .bypass_vector_index()
    )
    if lance_filter:
        query = query.where(lance_filter)
    selected_columns = ["id", "text", "_distance"]
    if include_vectors:
        selected_columns.append("vector")
    rows = query.limit(k).select(selected_columns).to_list()
    return [dict(row) for row in rows]


def _retrieved_window_ids(
    table: Any,
    embedder: TextEmbedder,
    query_text: str,
    *,
    k: int,
    lance_filter: str | None = None,
) -> list[str]:
    return [
        str(row["id"])
        for row in _retrieved_window_rows(
            table,
            embedder,
            query_text,
            k=k,
            lance_filter=lance_filter,
        )
    ]


def _known_item_sample(
    engine: Engine,
    *,
    query_cap: int,
) -> list[dict[str, str]]:
    """Select a bounded, stable-ID sample stratified across indexed entity types."""

    with Session(engine) as session:
        entity_types = list(
            session.scalars(
                select(SearchWindow.entity_type)
                .where(SearchWindow.text_display != "")
                .distinct()
                .order_by(SearchWindow.entity_type)
            )
        )
        if not entity_types:
            return []
        per_type = max(1, query_cap // len(entity_types))
        sample: list[dict[str, str]] = []
        selected_ids: list[str] = []
        for entity_type in entity_types:
            rows = session.execute(
                select(SearchWindow.id, SearchWindow.entity_type, SearchWindow.text_display)
                .where(
                    SearchWindow.entity_type == entity_type,
                    SearchWindow.text_display != "",
                )
                .order_by(SearchWindow.id)
                .limit(per_type)
            )
            for window_id, row_entity_type, text_display in rows:
                sample.append(
                    {
                        "window_id": str(window_id),
                        "entity_type": str(row_entity_type),
                        "query_text": str(text_display),
                    }
                )
                selected_ids.append(str(window_id))
        remaining = query_cap - len(sample)
        if remaining > 0:
            statement = select(
                SearchWindow.id,
                SearchWindow.entity_type,
                SearchWindow.text_display,
            ).where(SearchWindow.text_display != "")
            if selected_ids:
                statement = statement.where(SearchWindow.id.not_in(selected_ids))
            rows = session.execute(statement.order_by(SearchWindow.id).limit(remaining))
            for window_id, entity_type, text_display in rows:
                sample.append(
                    {
                        "window_id": str(window_id),
                        "entity_type": str(entity_type),
                        "query_text": str(text_display),
                    }
                )
    return sample[:query_cap]


def _metric_thresholds_passed(
    metrics: dict[str, int | float], thresholds: dict[str, float]
) -> bool:
    return all(float(metrics[name]) >= threshold for name, threshold in thresholds.items())


def held_out_feedback_revision(engine: Engine) -> dict[str, int | str]:
    """Fingerprint every input that can change held-out semantic evaluation.

    The aggregate digest binds cached acceptance evidence to the exact local feedback set without
    persisting query text, result identities, or per-judgment labels in the audit/status payload.
    """

    statement = (
        select(
            RelevanceFeedback.query_sha256,
            RelevanceFeedback.query_text,
            RelevanceFeedback.search_mode,
            RelevanceFeedback.filter_scope_sha256,
            RelevanceFeedback.filter_scope_json,
            RelevanceFeedback.result_id,
            RelevanceFeedback.label,
            RelevanceFeedback.evaluation_split,
            RelevanceFeedback.evaluation_intent,
        )
        .where(
            RelevanceFeedback.evaluation_split == "test",
            RelevanceFeedback.search_mode.in_(("semantic", "hybrid")),
        )
        .order_by(
            RelevanceFeedback.query_sha256,
            RelevanceFeedback.search_mode,
            RelevanceFeedback.filter_scope_sha256,
            RelevanceFeedback.result_id,
        )
    )
    digest = hashlib.sha256()
    digest.update(HELD_OUT_FEEDBACK_REVISION_VERSION.encode("ascii"))
    row_count = 0
    with Session(engine) as session:
        for (
            query_sha256,
            query_text,
            search_mode,
            filter_scope_sha256,
            filter_scope_json,
            result_id,
            label,
            evaluation_split,
            evaluation_intent,
        ) in session.execute(statement):
            record = json.dumps(
                [
                    str(query_sha256),
                    hashlib.sha256(str(query_text).encode("utf-8")).hexdigest(),
                    str(search_mode),
                    str(filter_scope_sha256),
                    filter_scope_json or {},
                    str(result_id),
                    str(label),
                    str(evaluation_split),
                    str(evaluation_intent) if evaluation_intent is not None else None,
                ],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            digest.update(len(record).to_bytes(8, "big"))
            digest.update(record)
            row_count += 1
    return {
        "version": HELD_OUT_FEEDBACK_REVISION_VERSION,
        "sha256": digest.hexdigest(),
        "row_count": row_count,
    }


def _automatic_known_item_evaluation(
    engine: Engine,
    table: Any,
    embedder: TextEmbedder,
) -> dict[str, Any]:
    samples = _known_item_sample(engine, query_cap=KNOWN_ITEM_QUERY_CAP)
    rankings: list[list[str]] = []
    relevance_sets: list[set[str]] = []
    for sample in samples:
        query_text = sample["query_text"]
        query_sha256 = hashlib.sha256(query_text.encode("utf-8")).hexdigest()
        retrieved_rows = _retrieved_window_rows(
            table,
            embedder,
            query_text,
            k=SEMANTIC_IR_K,
        )
        ranking_text_hashes: list[str] = []
        for row in retrieved_rows:
            text_hash = hashlib.sha256(str(row["text"]).encode("utf-8")).hexdigest()
            if text_hash not in ranking_text_hashes:
                ranking_text_hashes.append(text_hash)
        relevant = {query_sha256}
        rankings.append(ranking_text_hashes)
        relevance_sets.append(relevant)
    aggregate = retrieval_metrics(rankings, relevance_sets, k=SEMANTIC_IR_K)
    passed = bool(samples) and _metric_thresholds_passed(aggregate, KNOWN_ITEM_THRESHOLDS)
    return {
        "benchmark_kind": "automatic_known_item",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "used_for_activation": True,
        "method": (
            "Full indexed window text is re-encoded as a query and must retrieve an "
            "exact-text-equivalent window. Equivalent duplicate texts are collapsed by SHA-256. "
            "This is an automatic end-to-end encoder/vector-index sanity benchmark, not a "
            "substitute for human relevance judgments."
        ),
        "sampling": {
            "method": "stable-id-prefix-stratified-by-entity-type",
            "query_cap": KNOWN_ITEM_QUERY_CAP,
            "sampled_query_count": len(samples),
            "query_text_persisted": False,
            "query_details_persisted": False,
        },
        "thresholds": {**KNOWN_ITEM_THRESHOLDS, "k": SEMANTIC_IR_K},
        "metrics": aggregate,
    }


def _held_out_feedback_groups(engine: Engine, *, config_hash: str) -> dict[str, Any]:
    # Import lazily because the runtime search service imports this index module.
    from meta_archive.services.search import SearchFilters, _valid_window_statement

    with Session(engine) as session:
        group_rows = list(
            session.execute(
                select(
                    RelevanceFeedback.query_sha256,
                    RelevanceFeedback.query_text,
                    RelevanceFeedback.search_mode,
                    RelevanceFeedback.filter_scope_sha256,
                    RelevanceFeedback.filter_scope_json,
                    func.max(RelevanceFeedback.evaluation_intent).label("evaluation_intent_hint"),
                )
                .where(
                    RelevanceFeedback.evaluation_split == "test",
                    RelevanceFeedback.search_mode.in_(("semantic", "hybrid")),
                )
                .group_by(
                    RelevanceFeedback.query_sha256,
                    RelevanceFeedback.query_text,
                    RelevanceFeedback.search_mode,
                    RelevanceFeedback.filter_scope_sha256,
                    RelevanceFeedback.filter_scope_json,
                )
                .order_by(
                    desc(func.max(RelevanceFeedback.evaluation_intent)),
                    RelevanceFeedback.query_sha256,
                    RelevanceFeedback.search_mode,
                    RelevanceFeedback.filter_scope_sha256,
                )
                .limit(FEEDBACK_QUERY_CAP + 1)
            )
        )
        selected_groups = group_rows[:FEEDBACK_QUERY_CAP]
        prepared: list[dict[str, Any]] = []
        exclusions = {
            "without_relevant_window_judgment": 0,
            "relevant_target_not_in_candidate": 0,
        }
        any_judgments_truncated = False
        argument_intent = {
            "selected_query_group_count": 0,
            "prepared_query_group_count": 0,
            "inconsistently_tagged_query_group_count": 0,
            "judgments_truncated_query_group_count": 0,
            "query_selection_truncated": bool(
                len(group_rows) > FEEDBACK_QUERY_CAP
                and group_rows[FEEDBACK_QUERY_CAP][-1] == ARGUMENT_DIRECT_CONFLICT_INTENT
            ),
            "without_relevant_window_judgment": 0,
            "relevant_target_not_in_candidate": 0,
            "non_window_judgment_count": 0,
            "target_not_in_candidate_judgment_count": 0,
            "unsupported_label_judgment_count": 0,
        }
        for (
            query_sha256,
            query_text,
            search_mode,
            filter_scope_sha256,
            filter_scope_json,
            _evaluation_intent_hint,
        ) in selected_groups:
            filter_scope = dict(filter_scope_json or {})
            judgments = list(
                session.execute(
                    select(
                        RelevanceFeedback.result_id,
                        RelevanceFeedback.label,
                        RelevanceFeedback.evaluation_intent,
                    )
                    .where(
                        RelevanceFeedback.query_sha256 == query_sha256,
                        RelevanceFeedback.query_text == query_text,
                        RelevanceFeedback.search_mode == search_mode,
                        RelevanceFeedback.filter_scope_sha256 == filter_scope_sha256,
                        RelevanceFeedback.evaluation_split == "test",
                    )
                    .order_by(RelevanceFeedback.result_id)
                    .limit(FEEDBACK_JUDGMENT_CAP_PER_QUERY + 1)
                )
            )
            judgments_were_truncated = len(judgments) > FEEDBACK_JUDGMENT_CAP_PER_QUERY
            if judgments_were_truncated:
                any_judgments_truncated = True
            evaluation_intents: set[str | None] = {
                str(evaluation_intent) if evaluation_intent is not None else None
                for _result_id, _label, evaluation_intent in judgments
            }
            judgments = judgments[:FEEDBACK_JUDGMENT_CAP_PER_QUERY]
            label_counts = {"relevant": 0, "not_relevant": 0, "uncertain": 0}
            eligible_label_counts = {"relevant": 0, "not_relevant": 0, "uncertain": 0}
            relevant_window_ids: set[str] = set()
            judged_window_ids: set[str] = set()
            non_window_judgment_count = 0
            unsupported_label_judgment_count = 0
            for result_id, label, _evaluation_intent in judgments:
                label_text = str(label)
                if label_text in label_counts:
                    label_counts[label_text] += 1
                else:
                    unsupported_label_judgment_count += 1
                result_text = str(result_id)
                if not result_text.startswith("window:"):
                    non_window_judgment_count += 1
                    continue
                window_id = result_text.removeprefix("window:")
                judged_window_ids.add(window_id)
                if label_text == "relevant":
                    relevant_window_ids.add(window_id)
            evaluation_intent = (
                ARGUMENT_DIRECT_CONFLICT_INTENT
                if evaluation_intents == {ARGUMENT_DIRECT_CONFLICT_INTENT}
                else None
            )
            if evaluation_intent == ARGUMENT_DIRECT_CONFLICT_INTENT:
                argument_intent["selected_query_group_count"] += 1
                if judgments_were_truncated:
                    argument_intent["judgments_truncated_query_group_count"] += 1
            elif ARGUMENT_DIRECT_CONFLICT_INTENT in evaluation_intents:
                argument_intent["inconsistently_tagged_query_group_count"] += 1
            candidate_judged_ids = (
                {
                    window.id
                    for window in session.scalars(
                        _valid_window_statement(
                            sorted(judged_window_ids),
                            SearchFilters(**filter_scope),
                            config_hash=config_hash,
                        )
                    )
                }
                if judged_window_ids
                else set()
            )
            target_not_in_candidate_judgment_count = len(judged_window_ids - candidate_judged_ids)
            for result_id, label, _evaluation_intent in judgments:
                result_text = str(result_id)
                label_text = str(label)
                if not result_text.startswith("window:") or label_text not in eligible_label_counts:
                    continue
                if result_text.removeprefix("window:") in candidate_judged_ids:
                    eligible_label_counts[label_text] += 1
            candidate_relevant_ids = relevant_window_ids & candidate_judged_ids
            missing_relevant_window_count = len(relevant_window_ids - candidate_relevant_ids)
            if evaluation_intent == ARGUMENT_DIRECT_CONFLICT_INTENT:
                argument_intent["non_window_judgment_count"] += non_window_judgment_count
                argument_intent["target_not_in_candidate_judgment_count"] += (
                    target_not_in_candidate_judgment_count
                )
                argument_intent["unsupported_label_judgment_count"] += (
                    unsupported_label_judgment_count
                )
            if not relevant_window_ids:
                exclusions["without_relevant_window_judgment"] += 1
                if evaluation_intent == ARGUMENT_DIRECT_CONFLICT_INTENT:
                    argument_intent["without_relevant_window_judgment"] += 1
                continue
            if missing_relevant_window_count:
                exclusions["relevant_target_not_in_candidate"] += 1
                if evaluation_intent == ARGUMENT_DIRECT_CONFLICT_INTENT:
                    argument_intent["relevant_target_not_in_candidate"] += 1
            if not candidate_relevant_ids:
                continue
            if evaluation_intent == ARGUMENT_DIRECT_CONFLICT_INTENT:
                argument_intent["prepared_query_group_count"] += 1
            prepared.append(
                {
                    "query_sha256": str(query_sha256),
                    "query_text": str(query_text),
                    "source_search_mode": str(search_mode),
                    "filter_scope": filter_scope,
                    "filter_scope_sha256": str(filter_scope_sha256),
                    "judgment_count": len(judgments),
                    "label_counts": label_counts,
                    "eligible_judgment_count": sum(eligible_label_counts.values()),
                    "eligible_label_counts": eligible_label_counts,
                    "relevant_window_ids": candidate_relevant_ids,
                    "missing_relevant_window_count": missing_relevant_window_count,
                    "non_window_judgment_count": non_window_judgment_count,
                    "target_not_in_candidate_judgment_count": (
                        target_not_in_candidate_judgment_count
                    ),
                    "unsupported_label_judgment_count": unsupported_label_judgment_count,
                    "evaluation_intent": evaluation_intent,
                }
            )
    return {
        "groups": prepared,
        "selected_query_group_count": len(selected_groups),
        "query_selection_truncated": len(group_rows) > FEEDBACK_QUERY_CAP,
        "judgments_truncated": any_judgments_truncated,
        "exclusions": exclusions,
        "argument_direct_conflict_intent": argument_intent,
    }


def _held_out_reranker(
    layout: ArchiveLayout,
    reranker_factory: RerankerFactory | None,
) -> TextReranker:
    reranker = (
        reranker_factory(DEFAULT_RERANKER_SPEC)
        if reranker_factory is not None
        else cached_cross_encoder_reranker(
            str(layout.models / "reranker" / DEFAULT_RERANKER_SPEC.model_revision)
        )
    )
    if reranker.spec != DEFAULT_RERANKER_SPEC:
        raise RuntimeError("Held-out evaluation received an unexpected reranker specification")
    return reranker


def _reranked_vector_ids(
    engine: Engine,
    table: Any,
    embedder: TextEmbedder,
    reranker: TextReranker,
    query_text: str,
    *,
    filters: Any,
    config_hash: str,
    direct_conflict_ranker: DirectConflictRanker | None = None,
) -> _SemanticEvaluationRanking:
    """Mirror the runtime's exact-cosine candidate pool and local cross-encoder ordering."""

    from meta_archive.services.search import (
        _lance_filter,
        _rank_semantic_candidates,
        _valid_window_statement,
    )

    vector_rows = _retrieved_window_rows(
        table,
        embedder,
        query_text,
        k=MAX_RERANK_CANDIDATES,
        lance_filter=_lance_filter(filters),
        include_vectors=direct_conflict_ranker is not None,
    )
    ordered_ids = [str(row["id"]) for row in vector_rows]
    distances = {str(row["id"]): float(row["_distance"]) for row in vector_rows}
    with Session(engine) as session:
        windows = list(
            session.scalars(_valid_window_statement(ordered_ids, filters, config_hash=config_hash))
        )
    windows_by_id = {window.id: window for window in windows}
    valid_ordered = [window_id for window_id in ordered_ids if window_id in windows_by_id]
    vectors = {str(row["id"]): row.get("vector") for row in vector_rows}
    ranking = _rank_semantic_candidates(
        query_text,
        [
            (
                window_id,
                max(0.0, 1.0 - distances[window_id]),
                cast(Sequence[float], vectors[window_id])
                if vectors[window_id] is not None
                else None,
            )
            for window_id in valid_ordered
        ],
        {window_id: windows_by_id[window_id].text_display for window_id in valid_ordered},
        reranker,
        direct_conflict_ranker=direct_conflict_ranker,
    )
    return _SemanticEvaluationRanking(
        cosine_ids=tuple(valid_ordered),
        reranked_ids=tuple(window_id for window_id, _score, _cross, _cosine in ranking.rows),
    )


def _argument_direct_conflict_evaluation(
    groups: Sequence[dict[str, Any]],
    rankings: Sequence[Sequence[str]],
    relevance_sets: Sequence[set[str]],
    intent_audit: dict[str, Any],
) -> dict[str, Any]:
    selected_indices = [
        index
        for index, group in enumerate(groups)
        if group.get("evaluation_intent") == ARGUMENT_DIRECT_CONFLICT_INTENT
    ]
    selected_rankings = [rankings[index] for index in selected_indices]
    selected_relevance = [relevance_sets[index] for index in selected_indices]
    metrics = retrieval_metrics(
        selected_rankings,
        selected_relevance,
        k=SEMANTIC_IR_K,
    )
    selected_groups = [groups[index] for index in selected_indices]
    judgment_minimums_passed = bool(selected_groups) and all(
        int(group["eligible_judgment_count"]) >= ARGUMENT_MIN_JUDGMENTS_PER_QUERY
        and int(group["eligible_label_counts"]["relevant"])
        >= ARGUMENT_MIN_RELEVANT_JUDGMENTS_PER_QUERY
        and int(group["eligible_label_counts"]["not_relevant"])
        >= ARGUMENT_MIN_NOT_RELEVANT_JUDGMENTS_PER_QUERY
        for group in selected_groups
    )
    per_query_metrics_passed = bool(selected_groups) and all(
        _metric_thresholds_passed(
            retrieval_metrics([ranking], [relevant], k=SEMANTIC_IR_K),
            ARGUMENT_THRESHOLDS,
        )
        for ranking, relevant in zip(
            selected_rankings,
            selected_relevance,
            strict=True,
        )
    )
    selected_query_group_count = int(intent_audit["selected_query_group_count"])
    prepared_query_group_count = int(intent_audit["prepared_query_group_count"])
    selection_complete = bool(
        selected_query_group_count == prepared_query_group_count == len(selected_groups)
        and int(intent_audit["inconsistently_tagged_query_group_count"]) == 0
        and int(intent_audit["judgments_truncated_query_group_count"]) == 0
        and intent_audit["query_selection_truncated"] is False
        and int(intent_audit["without_relevant_window_judgment"]) == 0
        and int(intent_audit["relevant_target_not_in_candidate"]) == 0
        and int(intent_audit["non_window_judgment_count"]) == 0
        and int(intent_audit["target_not_in_candidate_judgment_count"]) == 0
        and int(intent_audit["unsupported_label_judgment_count"]) == 0
    )
    enough_query_groups = len(selected_groups) >= ARGUMENT_MIN_QUERY_GROUP_COUNT
    aggregate_metrics_passed = _metric_thresholds_passed(metrics, ARGUMENT_THRESHOLDS)
    passed = bool(
        enough_query_groups
        and selection_complete
        and judgment_minimums_passed
        and per_query_metrics_passed
        and aggregate_metrics_passed
    )
    if passed:
        status = "passed"
    elif selected_query_group_count == 0:
        status = "not_available"
    elif not selection_complete:
        status = "incomplete_or_inconsistent"
    elif not judgment_minimums_passed:
        status = "insufficient_judgments"
    else:
        status = "failed_metrics"
    return {
        "benchmark_kind": "flagship_argument_direct_conflict",
        "evaluation_intent": ARGUMENT_DIRECT_CONFLICT_INTENT,
        "required_for_production_acceptance": False,
        "status": status,
        "passed": passed,
        "method": (
            "Retain the legacy tagged test-split semantic or hybrid diagnostic independently "
            "through the runtime ranking path. Require explicit relevant and not-relevant "
            "judgments plus per-group and aggregate ranking thresholds. This diagnostic no "
            "longer supplies the production literal-query gate; query text and result identities "
            "are omitted from the aggregate evidence."
        ),
        "minimums": {
            "query_group_count": ARGUMENT_MIN_QUERY_GROUP_COUNT,
            "judgments_per_query_group": ARGUMENT_MIN_JUDGMENTS_PER_QUERY,
            "relevant_judgments_per_query_group": (ARGUMENT_MIN_RELEVANT_JUDGMENTS_PER_QUERY),
            "not_relevant_judgments_per_query_group": (
                ARGUMENT_MIN_NOT_RELEVANT_JUDGMENTS_PER_QUERY
            ),
        },
        "thresholds": {**ARGUMENT_THRESHOLDS, "k": SEMANTIC_IR_K},
        "metrics": metrics,
        "query_group_count": len(selected_groups),
        "reviewed_judgment_count": sum(int(group["judgment_count"]) for group in selected_groups),
        "judgment_count": sum(int(group["eligible_judgment_count"]) for group in selected_groups),
        "relevant_judgment_count": sum(
            int(group["eligible_label_counts"]["relevant"]) for group in selected_groups
        ),
        "not_relevant_judgment_count": sum(
            int(group["eligible_label_counts"]["not_relevant"]) for group in selected_groups
        ),
        "minimum_judgment_count_per_query_group": min(
            (int(group["eligible_judgment_count"]) for group in selected_groups),
            default=0,
        ),
        "minimum_relevant_judgment_count_per_query_group": min(
            (int(group["eligible_label_counts"]["relevant"]) for group in selected_groups),
            default=0,
        ),
        "minimum_not_relevant_judgment_count_per_query_group": min(
            (int(group["eligible_label_counts"]["not_relevant"]) for group in selected_groups),
            default=0,
        ),
        "all_query_groups_meet_judgment_minimums": judgment_minimums_passed,
        "all_query_groups_meet_metric_thresholds": per_query_metrics_passed,
        "selection_complete": selection_complete,
        **intent_audit,
    }


def _direct_conflict_calibration_base() -> dict[str, Any]:
    return {
        "benchmark_kind": "literal_direct_conflict_development",
        "evaluation_intent": DIRECT_CONFLICT_EVALUATION_INTENT,
        "evidence_kind": "conversation_grouped_development_calibration",
        "held_out": False,
        "required_for_production_acceptance": True,
        "method": "linear_svm",
        "ranking_pipeline": SEMANTIC_RANKING_PIPELINE_VERSION,
        "query_details_persisted": False,
        "minimums": {
            "relevant_count": DIRECT_CONFLICT_MIN_RELEVANT,
            "not_relevant_count": DIRECT_CONFLICT_MIN_NOT_RELEVANT,
            "fold_count": DIRECT_CONFLICT_VALIDATION_FOLDS,
        },
        "thresholds": {
            **DIRECT_CONFLICT_DEVELOPMENT_THRESHOLDS,
            "k": DIRECT_CONFLICT_RANKING_K,
        },
    }


def _direct_conflict_development_evaluation(
    layout: ArchiveLayout,
    engine: Engine,
    table: Any,
    embedder: TextEmbedder,
    *,
    config_hash: str,
    reranker_factory: RerankerFactory | None,
) -> dict[str, Any]:
    from meta_archive.services.search import (
        SearchFilters,
        _rank_semantic_candidates,
        _valid_window_statement,
    )

    base = _direct_conflict_calibration_base()
    try:
        revision_before = development_feedback_revision_payload(engine)
        ranker = derive_direct_conflict_ranker(
            engine,
            LanceDirectConflictVectorSource(layout),
            config_id=config_hash,
        )
    except Exception as exc:
        return {
            **base,
            "status": "not_available",
            "passed": False,
            "failure_code": (
                exc.code
                if isinstance(exc, DirectConflictRankerError)
                else "direct_conflict_calibration_unavailable"
            ),
            "feedback_revision_stable_during_evaluation": False,
        }

    counts = ranker.audit.counts
    validation = ranker.audit.metrics
    minimum_metrics = {
        "average_precision": validation.average_precision.minimum,
        "roc_auc": validation.roc_auc.minimum,
        "recall_at_10": validation.recall_at_10.minimum,
        "mrr_at_10": validation.mrr_at_10.minimum,
        "ndcg_at_10": validation.ndcg_at_10.minimum,
    }
    mean_metrics = {
        "average_precision": validation.average_precision.mean,
        "roc_auc": validation.roc_auc.mean,
        "recall_at_10": validation.recall_at_10.mean,
        "mrr_at_10": validation.mrr_at_10.mean,
        "ndcg_at_10": validation.ndcg_at_10.mean,
    }
    grouped_cross_validation_passed = all(
        float(minimum_metrics[name]) >= threshold
        for name, threshold in DIRECT_CONFLICT_DEVELOPMENT_THRESHOLDS.items()
    )

    with Session(engine) as session:
        literal_review_rows = [
            (str(result_id), str(label), str(notes or ""))
            for result_id, label, notes in session.execute(
                select(
                    RelevanceFeedback.result_id,
                    RelevanceFeedback.label,
                    RelevanceFeedback.notes,
                ).where(
                    RelevanceFeedback.evaluation_intent == DIRECT_CONFLICT_EVALUATION_INTENT,
                    RelevanceFeedback.evaluation_split == "validation",
                    RelevanceFeedback.search_mode == "semantic",
                    RelevanceFeedback.query_sha256 == DIRECT_CONFLICT_LITERAL_QUERY_SHA256,
                    RelevanceFeedback.result_id.like("window:%"),
                )
            )
            if hashlib.sha256(str(notes or "").encode("utf-8")).hexdigest()
            == DIRECT_CONFLICT_LITERAL_REGRESSION_NOTES_SHA256
        ]
    literal_label_counts = {
        label: sum(row_label == label for _result_id, row_label, _notes in literal_review_rows)
        for label in ("relevant", "not_relevant", "uncertain")
    }
    literal_cohort_complete = bool(
        len(literal_review_rows) == DIRECT_CONFLICT_LITERAL_REGRESSION_REVIEWED_COUNT
        and literal_label_counts["relevant"] == DIRECT_CONFLICT_LITERAL_REGRESSION_RELEVANT_COUNT
        and literal_label_counts["not_relevant"]
        == DIRECT_CONFLICT_LITERAL_REGRESSION_NOT_RELEVANT_COUNT
        and literal_label_counts["uncertain"] == 0
    )
    literal_relevant_ids = {
        result_id.removeprefix("window:")
        for result_id, label, _notes in literal_review_rows
        if label == "relevant"
    }

    literal_metrics = {
        "query_count": 0,
        "k": DIRECT_CONFLICT_RANKING_K,
        "recall_at_10": 0.0,
        "mrr_at_10": 0.0,
        "ndcg_at_10": 0.0,
    }
    literal_candidate_count = 0
    literal_regression_passed = False
    literal_failure_code: str | None = None
    if literal_relevant_ids:
        try:
            vector_rows = _retrieved_window_rows(
                table,
                embedder,
                DIRECT_CONFLICT_TRIGGER,
                k=MAX_RERANK_CANDIDATES,
                include_vectors=True,
            )
            ordered_ids = [str(row["id"]) for row in vector_rows]
            distances = {str(row["id"]): float(row.get("_distance", 1.0)) for row in vector_rows}
            vectors = {str(row["id"]): row.get("vector") for row in vector_rows}
            with Session(engine) as session:
                windows = list(
                    session.scalars(
                        _valid_window_statement(
                            ordered_ids,
                            SearchFilters(),
                            config_hash=config_hash,
                        )
                    )
                )
            windows_by_id = {window.id: window for window in windows}
            candidates = [
                (
                    window_id,
                    max(0.0, 1.0 - distances[window_id]),
                    cast(Sequence[float], vectors[window_id])
                    if vectors[window_id] is not None
                    else None,
                )
                for window_id in ordered_ids
                if window_id in windows_by_id
            ]
            reranker = _held_out_reranker(layout, reranker_factory)
            ranking = _rank_semantic_candidates(
                DIRECT_CONFLICT_TRIGGER,
                candidates,
                {
                    window_id: windows_by_id[window_id].text_display
                    for window_id, _cosine, _vector in candidates
                },
                reranker,
                direct_conflict_ranker=ranker,
            )
            literal_candidate_count = len(ranking.rows)
            raw_literal_metrics = retrieval_metrics(
                [[window_id for window_id, _score, _cross, _cosine in ranking.rows]],
                [literal_relevant_ids],
                k=DIRECT_CONFLICT_RANKING_K,
            )
            literal_metrics = {
                "query_count": raw_literal_metrics["query_count"],
                "k": raw_literal_metrics["k"],
                "recall_at_10": raw_literal_metrics["recall_at_k"],
                "mrr_at_10": raw_literal_metrics["mrr"],
                "ndcg_at_10": raw_literal_metrics["ndcg_at_k"],
            }
            literal_regression_passed = literal_cohort_complete and all(
                float(literal_metrics[name]) >= threshold
                for name, threshold in DIRECT_CONFLICT_DEVELOPMENT_THRESHOLDS.items()
            )
        except Exception:
            literal_failure_code = "direct_conflict_literal_regression_unavailable"
    else:
        literal_failure_code = "direct_conflict_literal_relevant_set_missing"

    try:
        revision_after = development_feedback_revision_payload(engine)
    except DirectConflictRankerError:
        revision_after = None
    revision_stable = bool(
        revision_after == revision_before
        and ranker.audit.private_feedback_revision.private_payload() == revision_before
    )
    counts_passed = bool(
        counts.relevant_rows >= DIRECT_CONFLICT_MIN_RELEVANT
        and counts.not_relevant_rows >= DIRECT_CONFLICT_MIN_NOT_RELEVANT
        and counts.binary_training_rows == counts.relevant_rows + counts.not_relevant_rows
        and counts.reviewed_feedback_rows == counts.binary_training_rows + counts.uncertain_rows
        and validation.fold_count == DIRECT_CONFLICT_VALIDATION_FOLDS
    )
    passed = bool(
        counts_passed
        and grouped_cross_validation_passed
        and literal_regression_passed
        and revision_stable
    )
    if passed:
        status = "passed"
    elif not revision_stable:
        status = "stale_feedback"
    elif not counts_passed:
        status = "insufficient_judgments"
    elif not grouped_cross_validation_passed:
        status = "failed_grouped_cross_validation"
    else:
        status = "failed_literal_regression"
    return {
        **base,
        "status": status,
        "passed": passed,
        "model_metadata": {
            "classifier": "LinearSVC",
            "library": "scikit-learn",
            "library_version": version("scikit-learn"),
            "ranker_version": DIRECT_CONFLICT_RANKER_VERSION,
            "score_kind": DIRECT_CONFLICT_SCORE_KIND,
            "parameters": asdict(ranker.audit.parameters),
        },
        "judgment_count": counts.reviewed_feedback_rows,
        "reviewed_judgment_count": counts.reviewed_feedback_rows,
        "binary_judgment_count": counts.binary_training_rows,
        "relevant_count": counts.relevant_rows,
        "not_relevant_count": counts.not_relevant_rows,
        "uncertain_count": counts.uncertain_rows,
        "conversation_count": counts.conversation_groups,
        "fold_count": validation.fold_count,
        "grouped_cross_validation": {
            "fold_count": validation.fold_count,
            "minimum_metrics": minimum_metrics,
            "mean_metrics": mean_metrics,
            "passed": grouped_cross_validation_passed,
        },
        "literal_query_regression": {
            "candidate_count": literal_candidate_count,
            "reviewed_judgment_count": len(literal_review_rows),
            "relevant_judgment_count": len(literal_relevant_ids),
            "not_relevant_judgment_count": literal_label_counts["not_relevant"],
            "uncertain_judgment_count": literal_label_counts["uncertain"],
            "cohort_complete": literal_cohort_complete,
            "metrics": literal_metrics,
            "passed": literal_regression_passed,
            "failure_code": literal_failure_code,
        },
        "feedback_revision": revision_before,
        "feedback_revision_stable_during_evaluation": revision_stable,
        "private_material_fingerprint_sha256": (ranker.audit.private_material_fingerprint_sha256),
    }


def _candidate_recall_summary(
    rankings: Sequence[Sequence[str]],
    relevance_sets: Sequence[set[str]],
) -> dict[str, int | float]:
    summary: dict[str, int | float] = {"query_count": len(rankings)}
    for k in SEMANTIC_CANDIDATE_RECALL_KS:
        metrics = retrieval_metrics(rankings, relevance_sets, k=k)
        summary[f"recall_at_{k}"] = float(metrics["recall_at_k"])
    return summary


def _validated_upgrade_metrics(value: object) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    metrics: dict[str, float] = {}
    for name in SEMANTIC_UPGRADE_METRICS:
        metric = value.get(name)
        if (
            not isinstance(metric, (int, float))
            or isinstance(metric, bool)
            or not math.isfinite(float(metric))
        ):
            return None
        metrics[name] = float(metric)
    return metrics


def _semantic_upgrade_metric_comparison(
    candidate_metrics_value: object,
    incumbent_metrics_value: object,
) -> dict[str, Any]:
    candidate_metrics = _validated_upgrade_metrics(candidate_metrics_value)
    incumbent_metrics = _validated_upgrade_metrics(incumbent_metrics_value)
    if candidate_metrics is None or incumbent_metrics is None:
        raise ValueError("Semantic upgrade comparison requires finite retrieval metrics")
    metric_comparisons: dict[str, dict[str, float | bool]] = {}
    for name in SEMANTIC_UPGRADE_METRICS:
        candidate = candidate_metrics[name]
        incumbent = incumbent_metrics[name]
        delta = candidate - incumbent
        regression = incumbent - candidate
        metric_comparisons[name] = {
            "candidate": candidate,
            "incumbent": incumbent,
            "delta": delta,
            "regression_exceeds_tolerance": (
                regression > SEMANTIC_UPGRADE_NON_REGRESSION_TOLERANCE
                and not math.isclose(
                    regression,
                    SEMANTIC_UPGRADE_NON_REGRESSION_TOLERANCE,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ),
            "demonstrated_improvement": (
                delta > SEMANTIC_UPGRADE_DEMONSTRATED_IMPROVEMENT
                or math.isclose(
                    delta,
                    SEMANTIC_UPGRADE_DEMONSTRATED_IMPROVEMENT,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ),
        }
    non_regression_passed = all(
        comparison["regression_exceeds_tolerance"] is False
        for comparison in metric_comparisons.values()
    )
    demonstrated_improvement = any(
        comparison["demonstrated_improvement"] is True for comparison in metric_comparisons.values()
    )
    passed = non_regression_passed and demonstrated_improvement
    status = (
        "passed"
        if passed
        else (
            "failed_regression"
            if not non_regression_passed
            else "failed_no_demonstrated_improvement"
        )
    )
    return {
        "status": status,
        "required_for_activation": True,
        "passed": passed,
        "non_regression_tolerance": SEMANTIC_UPGRADE_NON_REGRESSION_TOLERANCE,
        "demonstrated_improvement_threshold": (SEMANTIC_UPGRADE_DEMONSTRATED_IMPROVEMENT),
        "demonstrated_improvement_required": True,
        "demonstrated_improvement": demonstrated_improvement,
        "metrics": metric_comparisons,
    }


def _not_required_incumbent_comparison(
    status: str,
    *,
    reason_codes: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "status": status,
        "required_for_activation": False,
        "passed": None,
        "reason_codes": list(reason_codes),
        "non_regression_tolerance": SEMANTIC_UPGRADE_NON_REGRESSION_TOLERANCE,
        "demonstrated_improvement_threshold": (SEMANTIC_UPGRADE_DEMONSTRATED_IMPROVEMENT),
        "demonstrated_improvement_required": False,
        "demonstrated_improvement": None,
    }


def _failed_incumbent_comparison(
    *,
    reason_codes: Sequence[str],
) -> dict[str, Any]:
    return {
        "status": "failed_incomparable_incumbent",
        "required_for_activation": True,
        "passed": False,
        "reason_codes": list(reason_codes),
        "non_regression_tolerance": SEMANTIC_UPGRADE_NON_REGRESSION_TOLERANCE,
        "demonstrated_improvement_threshold": (SEMANTIC_UPGRADE_DEMONSTRATED_IMPROVEMENT),
        "demonstrated_improvement_required": False,
        "demonstrated_improvement": None,
    }


def _active_incumbent_comparison(
    engine: Engine,
    *,
    candidate_config_hash: str,
    candidate_corpus_revision: str,
    candidate_held_out: dict[str, Any],
) -> dict[str, Any]:
    with Session(engine) as session:
        incumbents = list(
            session.scalars(
                select(SemanticIndexConfig)
                .where(
                    SemanticIndexConfig.active.is_(True),
                    SemanticIndexConfig.status == "ready",
                    SemanticIndexConfig.id != candidate_config_hash,
                )
                .order_by(SemanticIndexConfig.id)
                .limit(2)
            )
        )
    if not incumbents:
        return _not_required_incumbent_comparison("not_required_no_different_active_incumbent")
    if len(incumbents) != 1:
        return _failed_incumbent_comparison(
            reason_codes=("multiple_active_incumbents",),
        )

    audit = incumbents[0].audit_json
    accepted = audit.get("accepted_semantic_ir_evaluation") if isinstance(audit, dict) else None
    if not isinstance(accepted, dict):
        return _failed_incumbent_comparison(
            reason_codes=("accepted_evidence_missing",),
        )
    incumbent_candidate = accepted.get("candidate")
    incumbent_candidate = incumbent_candidate if isinstance(incumbent_candidate, dict) else {}
    incumbent_held_out = accepted.get("held_out_feedback")
    incumbent_held_out = incumbent_held_out if isinstance(incumbent_held_out, dict) else {}
    candidate_metrics = candidate_held_out.get("metrics")
    incumbent_metrics = incumbent_held_out.get("metrics")
    candidate_query_count = (
        candidate_metrics.get("query_count") if isinstance(candidate_metrics, dict) else None
    )
    incumbent_query_count = (
        incumbent_metrics.get("query_count") if isinstance(incumbent_metrics, dict) else None
    )
    reason_codes: list[str] = []
    if incumbent_candidate.get("corpus_revision") != candidate_corpus_revision:
        reason_codes.append("corpus_revision_mismatch")
    if incumbent_held_out.get("feedback_revision") != candidate_held_out.get("feedback_revision"):
        reason_codes.append("feedback_revision_mismatch")
    if (
        not isinstance(candidate_query_count, int)
        or isinstance(candidate_query_count, bool)
        or not isinstance(incumbent_query_count, int)
        or isinstance(incumbent_query_count, bool)
        or incumbent_query_count != candidate_query_count
    ):
        reason_codes.append("query_count_mismatch")
    if incumbent_held_out.get("ranking_pipeline") != candidate_held_out.get("ranking_pipeline"):
        reason_codes.append("ranking_pipeline_mismatch")
    if incumbent_held_out.get("reranker") != candidate_held_out.get("reranker"):
        reason_codes.append("reranker_spec_mismatch")
    if not (
        incumbent_held_out.get("status") == "passed"
        and incumbent_held_out.get("passed") is True
        and incumbent_held_out.get("used_for_activation") is True
    ):
        reason_codes.append("incumbent_held_out_not_accepted")
    if _validated_upgrade_metrics(incumbent_metrics) is None:
        reason_codes.append("incumbent_metrics_invalid")
    if _validated_upgrade_metrics(candidate_metrics) is None:
        reason_codes.append("candidate_metrics_invalid")
    if reason_codes:
        return _failed_incumbent_comparison(
            reason_codes=reason_codes,
        )
    return _semantic_upgrade_metric_comparison(candidate_metrics, incumbent_metrics)


def _same_config_accepted_evidence_comparison(
    prior_evaluation_value: object,
    current_evaluation: dict[str, Any],
    *,
    config_hash: str,
    corpus_revision: str,
) -> dict[str, Any]:
    reason_codes: list[str] = []
    prior_evaluation = prior_evaluation_value if isinstance(prior_evaluation_value, dict) else {}
    if not prior_evaluation:
        reason_codes.append("prior_accepted_evaluation_missing")
    elif prior_evaluation.get("passed") is not True:
        reason_codes.append("prior_accepted_evaluation_not_passed")
    if current_evaluation.get("passed") is not True:
        reason_codes.append("current_evaluation_not_passed")

    prior_candidate = prior_evaluation.get("candidate")
    prior_candidate = prior_candidate if isinstance(prior_candidate, dict) else {}
    current_candidate = current_evaluation.get("candidate")
    current_candidate = current_candidate if isinstance(current_candidate, dict) else {}
    if (
        prior_candidate.get("config_hash") != config_hash
        or current_candidate.get("config_hash") != config_hash
    ):
        reason_codes.append("candidate_config_hash_mismatch")
    if (
        prior_candidate.get("corpus_revision") != corpus_revision
        or current_candidate.get("corpus_revision") != corpus_revision
    ):
        reason_codes.append("candidate_corpus_revision_mismatch")
    if not isinstance(prior_evaluation.get("evaluator_version"), str) or prior_evaluation.get(
        "evaluator_version"
    ) != current_evaluation.get("evaluator_version"):
        reason_codes.append("evaluator_revision_mismatch")

    prior_held_out = prior_evaluation.get("held_out_feedback")
    prior_held_out = prior_held_out if isinstance(prior_held_out, dict) else {}
    current_held_out = current_evaluation.get("held_out_feedback")
    current_held_out = current_held_out if isinstance(current_held_out, dict) else {}
    if not prior_held_out or not current_held_out:
        reason_codes.append("held_out_evidence_missing")
    if not isinstance(prior_held_out.get("feedback_revision"), dict) or prior_held_out.get(
        "feedback_revision"
    ) != current_held_out.get("feedback_revision"):
        reason_codes.append("held_out_feedback_revision_mismatch")
    if not isinstance(prior_held_out.get("ranking_pipeline"), str) or prior_held_out.get(
        "ranking_pipeline"
    ) != current_held_out.get("ranking_pipeline"):
        reason_codes.append("held_out_ranking_pipeline_mismatch")
    if not isinstance(prior_held_out.get("reranker"), dict) or prior_held_out.get(
        "reranker"
    ) != current_held_out.get("reranker"):
        reason_codes.append("held_out_reranker_mismatch")
    held_out_policy_fields = (
        "benchmark_kind",
        "minimum_query_count_for_activation",
        "query_cap",
        "judgment_cap_per_query",
        "thresholds",
    )
    if any(
        prior_held_out.get(name) != current_held_out.get(name) for name in held_out_policy_fields
    ):
        reason_codes.append("held_out_policy_mismatch")
    if current_held_out.get("status") == "stale_feedback":
        reason_codes.append("held_out_feedback_stale")

    prior_development = prior_evaluation.get("direct_conflict_development_calibration")
    prior_development = prior_development if isinstance(prior_development, dict) else {}
    current_development = current_evaluation.get("direct_conflict_development_calibration")
    current_development = current_development if isinstance(current_development, dict) else {}
    if not prior_development or not current_development:
        reason_codes.append("direct_conflict_evidence_missing")
    if prior_development.get("feedback_revision") != current_development.get("feedback_revision"):
        reason_codes.append("direct_conflict_feedback_revision_mismatch")
    if not isinstance(prior_development.get("ranking_pipeline"), str) or prior_development.get(
        "ranking_pipeline"
    ) != current_development.get("ranking_pipeline"):
        reason_codes.append("direct_conflict_ranking_pipeline_mismatch")
    if prior_development.get("model_metadata") != current_development.get("model_metadata"):
        reason_codes.append("direct_conflict_ranker_mismatch")
    development_policy_fields = (
        "benchmark_kind",
        "evaluation_intent",
        "evidence_kind",
        "method",
        "minimums",
        "thresholds",
    )
    if any(
        prior_development.get(name) != current_development.get(name)
        for name in development_policy_fields
    ):
        reason_codes.append("direct_conflict_policy_mismatch")
    if current_development.get("status") == "stale_feedback":
        reason_codes.append("direct_conflict_feedback_stale")

    # Retrieval metrics are deliberately re-gated rather than required to be byte-identical.
    # Exact config/corpus and evaluator inputs must match, while harmless floating-point drift is
    # allowed only when the current run still passes the same acceptance policy.
    prior_was_production_accepted = prior_evaluation.get("production_acceptance_passed") is True
    if prior_was_production_accepted and not (
        current_evaluation.get("production_acceptance_passed") is True
        and current_held_out.get("feedback_revision_stable_during_evaluation") is True
        and current_development.get("feedback_revision_stable_during_evaluation") is True
    ):
        reason_codes.append("production_acceptance_not_revalidated")

    reason_codes = list(dict.fromkeys(reason_codes))
    passed = not reason_codes
    return {
        "status": "passed_preserved_prior_acceptance" if passed else "failed_stale_acceptance",
        "required_for_activation": True,
        "passed": passed,
        "reason_codes": reason_codes,
        "prior_production_acceptance": prior_was_production_accepted,
        "preserves_prior_accepted_evaluation": passed,
    }


def _held_out_feedback_evaluation(
    layout: ArchiveLayout,
    engine: Engine,
    table: Any,
    embedder: TextEmbedder,
    *,
    config_hash: str,
    reranker_factory: RerankerFactory | None,
) -> dict[str, Any]:
    # Import lazily because the runtime search service imports this index module.
    from meta_archive.services.search import (
        SearchFilters,
        _lexical_window_ranks,
        _valid_window_statement,
        lexical_search,
    )

    feedback_revision = held_out_feedback_revision(engine)
    prepared = _held_out_feedback_groups(engine, config_hash=config_hash)
    groups = prepared.pop("groups")
    rankings: list[list[str]] = []
    cosine_rankings: list[list[str]] = []
    relevance_sets: list[set[str]] = []
    rankings_by_mode: dict[str, list[list[str]]] = {"semantic": [], "hybrid": []}
    cosine_rankings_by_mode: dict[str, list[list[str]]] = {
        "semantic": [],
        "hybrid": [],
    }
    relevance_sets_by_mode: dict[str, list[set[str]]] = {
        "semantic": [],
        "hybrid": [],
    }
    query_group_counts_by_mode = {"semantic": 0, "hybrid": 0}
    reranker = _held_out_reranker(layout, reranker_factory) if groups else None
    for group in groups:
        if reranker is None:  # pragma: no cover - guarded by the nonempty loop
            raise RuntimeError("Held-out reranker was not initialized")
        filters = SearchFilters(**cast(dict[str, Any], group["filter_scope"]))
        semantic_ranking = _reranked_vector_ids(
            engine,
            table,
            embedder,
            reranker,
            str(group["query_text"]),
            filters=filters,
            config_hash=config_hash,
        )
        ranking = list(semantic_ranking.reranked_ids[:SEMANTIC_IR_K])
        source_search_mode = str(group["source_search_mode"])
        if source_search_mode == "hybrid":
            lexical_items = lexical_search(
                engine,
                str(group["query_text"]),
                filters=filters,
                # The HTTP route caps the hybrid lexical pool at 2,000 before RRF.
                limit=2_000,
            )
            with Session(engine) as session:
                lexical_ranking, _result_windows = _lexical_window_ranks(
                    session,
                    lexical_items,
                )
                candidate_ids = {
                    window.id
                    for window in session.scalars(
                        _valid_window_statement(
                            list(dict.fromkeys([*lexical_ranking, *semantic_ranking.reranked_ids])),
                            filters,
                            config_hash=config_hash,
                        )
                    )
                }
            fused: dict[str, float] = {}
            for source_ranking in (lexical_ranking, semantic_ranking.reranked_ids):
                for rank, window_id in enumerate(source_ranking, start=1):
                    if window_id in candidate_ids:
                        fused[window_id] = fused.get(window_id, 0.0) + 1.0 / (60 + rank)
            ranking = [
                window_id
                for window_id, _score in sorted(
                    fused.items(),
                    key=lambda item: (-item[1], item[0]),
                )[:SEMANTIC_IR_K]
            ]
        relevant = cast(set[str], group["relevant_window_ids"])
        rankings.append(ranking)
        cosine_ranking = list(semantic_ranking.cosine_ids)
        cosine_rankings.append(cosine_ranking)
        relevance_sets.append(relevant)
        rankings_by_mode[source_search_mode].append(ranking)
        cosine_rankings_by_mode[source_search_mode].append(cosine_ranking)
        relevance_sets_by_mode[source_search_mode].append(relevant)
        query_group_counts_by_mode[source_search_mode] += 1
    aggregate = retrieval_metrics(rankings, relevance_sets, k=SEMANTIC_IR_K)
    metrics_by_mode = {
        mode: retrieval_metrics(
            rankings_by_mode[mode],
            relevance_sets_by_mode[mode],
            k=SEMANTIC_IR_K,
        )
        for mode in ("semantic", "hybrid")
    }
    cosine_candidate_recall = _candidate_recall_summary(
        cosine_rankings,
        relevance_sets,
    )
    cosine_candidate_recall_by_mode = {
        mode: _candidate_recall_summary(
            cosine_rankings_by_mode[mode],
            relevance_sets_by_mode[mode],
        )
        for mode in ("semantic", "hybrid")
    }
    argument_direct_conflict = _argument_direct_conflict_evaluation(
        groups,
        rankings,
        relevance_sets,
        cast(dict[str, Any], prepared["argument_direct_conflict_intent"]),
    )
    query_count = len(rankings)
    feedback_revision_stable = feedback_revision == held_out_feedback_revision(engine)
    evidence_truncated = bool(
        prepared["query_selection_truncated"] or prepared["judgments_truncated"]
    )
    if not feedback_revision_stable:
        argument_direct_conflict = {
            **argument_direct_conflict,
            "status": "stale_feedback",
            "passed": False,
            "selection_complete": False,
        }
    # Truncation must remain activation-relevant even when the cap itself reduces the selected
    # query count below the ordinary minimum. Otherwise a capped evidence set could silently
    # downgrade itself to an optional benchmark and let the candidate through.
    used_for_activation = query_count >= HELD_OUT_MIN_QUERY_COUNT or evidence_truncated
    passed: bool | None = (
        _metric_thresholds_passed(aggregate, HELD_OUT_THRESHOLDS)
        and feedback_revision_stable
        and not evidence_truncated
        if used_for_activation
        else None
    )
    if not feedback_revision_stable:
        evaluation_status = "stale_feedback"
    elif evidence_truncated:
        evaluation_status = "truncated_evidence"
    elif not query_count:
        evaluation_status = "not_available"
    elif not used_for_activation:
        evaluation_status = "insufficient_sample"
    else:
        evaluation_status = "passed" if passed else "failed"
    return {
        "benchmark_kind": "held_out_user_feedback",
        "status": evaluation_status,
        "passed": passed,
        "used_for_activation": used_for_activation,
        "method": (
            "Re-run the candidate's exact-cosine pool through the pinned local cross-encoder "
            "and the runtime hybrid RRF path against test-split semantic and hybrid judgments. "
            "Train, validation, unassigned, non-window, uncertain-only, and stale targets do "
            "not enter the activation metric. Query text is used locally, while per-query "
            "identities, judgments, rankings, and content are omitted from the persisted audit."
        ),
        "ranking_pipeline": SEMANTIC_RANKING_PIPELINE_VERSION,
        "reranker": asdict(DEFAULT_RERANKER_SPEC),
        "query_details_persisted": False,
        "query_group_counts_by_mode": query_group_counts_by_mode,
        "feedback_revision": feedback_revision,
        "feedback_revision_stable_during_evaluation": feedback_revision_stable,
        "minimum_query_count_for_activation": HELD_OUT_MIN_QUERY_COUNT,
        "query_cap": FEEDBACK_QUERY_CAP,
        "judgment_cap_per_query": FEEDBACK_JUDGMENT_CAP_PER_QUERY,
        "thresholds": {**HELD_OUT_THRESHOLDS, "k": SEMANTIC_IR_K},
        "metrics": aggregate,
        "metrics_by_mode": metrics_by_mode,
        "cosine_candidate_recall": cosine_candidate_recall,
        "cosine_candidate_recall_by_mode": cosine_candidate_recall_by_mode,
        "evidence_truncated": evidence_truncated,
        "argument_direct_conflict": argument_direct_conflict,
        **prepared,
    }


def _semantic_ir_evaluation(
    layout: ArchiveLayout,
    engine: Engine,
    table: Any,
    embedder: TextEmbedder,
    *,
    config_hash: str,
    vector_table: str,
    corpus_revision: str,
    total_windows: int,
    approximate_index_requested: bool,
    reranker_factory: RerankerFactory | None,
) -> dict[str, Any]:
    automatic = _automatic_known_item_evaluation(engine, table, embedder)
    held_out = _held_out_feedback_evaluation(
        layout,
        engine,
        table,
        embedder,
        config_hash=config_hash,
        reranker_factory=reranker_factory,
    )
    active_incumbent_comparison = _active_incumbent_comparison(
        engine,
        candidate_config_hash=config_hash,
        candidate_corpus_revision=corpus_revision,
        candidate_held_out=held_out,
    )
    direct_conflict_development = _direct_conflict_development_evaluation(
        layout,
        engine,
        table,
        embedder,
        config_hash=config_hash,
        reranker_factory=reranker_factory,
    )
    incumbent_comparison_satisfied = bool(
        active_incumbent_comparison["required_for_activation"] is False
        or active_incumbent_comparison["passed"] is True
    )
    passed = (
        bool(automatic["passed"])
        and (not held_out["used_for_activation"] or bool(held_out["passed"]))
        and incumbent_comparison_satisfied
        and direct_conflict_development["passed"] is True
    )
    production_acceptance_passed = bool(
        passed
        and held_out["used_for_activation"] is True
        and held_out["passed"] is True
        and held_out["feedback_revision_stable_during_evaluation"] is True
        and held_out["evidence_truncated"] is False
        and incumbent_comparison_satisfied
        and direct_conflict_development["passed"] is True
    )
    return {
        "evaluator_version": SEMANTIC_IR_EVALUATOR_VERSION,
        "evaluated_at": datetime.now(UTC).isoformat(),
        "passed": passed,
        "production_acceptance_passed": production_acceptance_passed,
        "decision": (
            "accepted"
            if passed
            else "rejected; an existing ready index, when present, remains active"
        ),
        "candidate": {
            "config_hash": config_hash,
            "vector_table": vector_table,
            "corpus_revision": corpus_revision,
            "window_count": total_windows,
            "distance_metric": "cosine",
            "query_engine": "lancedb-exact-vector-search",
            "approximate_index_requested": approximate_index_requested,
        },
        "automatic_known_item": automatic,
        "held_out_feedback": held_out,
        "active_incumbent_comparison": active_incumbent_comparison,
        "direct_conflict_development_calibration": direct_conflict_development,
    }


def approximate_token_count(text: str) -> int:
    return len(_TOKEN_PATTERN.findall(text))


def _text_chunks(text: str, limit: int) -> list[tuple[int, int, str]]:
    matches = list(_TOKEN_PATTERN.finditer(text))
    if len(matches) <= limit:
        return [(0, len(text), text)]
    chunks: list[tuple[int, int, str]] = []
    token_start = 0
    while token_start < len(matches):
        token_end = min(token_start + limit, len(matches))
        char_start = 0 if token_start == 0 else matches[token_start].start()
        char_end = len(text) if token_end == len(matches) else matches[token_end - 1].end()
        chunks.append((char_start, char_end, text[char_start:char_end]))
        token_start = token_end
    return chunks


def _render_turns(turns: Sequence[WindowTurn]) -> str:
    return "\n".join(f"{turn.sender_display}: {turn.text_display}" for turn in turns)


def _window_id(
    conversation_id: str, turns: Sequence[WindowTurn], *, chunk_ordinal: int | None = None
) -> str:
    return stable_id(
        "semantic-window",
        WINDOW_VERSION,
        conversation_id,
        turns[0].segment_id,
        turns[0].id,
        turns[-1].id,
        str(chunk_ordinal) if chunk_ordinal is not None else "full-turn-range",
    )


def _window_draft(
    conversation_id: str,
    turns: Sequence[WindowTurn],
    *,
    source_char_start: int | None = None,
    source_char_end: int | None = None,
    chunk_ordinal: int | None = None,
    chunk_count: int | None = None,
) -> WindowDraft:
    rendered = _render_turns(turns)
    if approximate_token_count(rendered) > MAX_APPROX_TOKENS:
        raise RuntimeError("Search-window policy attempted to omit source text")
    normalized = normalize_text(rendered)
    participants = "\n".join(dict.fromkeys(turn.sender_search for turn in turns))
    return WindowDraft(
        id=_window_id(conversation_id, turns, chunk_ordinal=chunk_ordinal),
        conversation_id=conversation_id,
        segment_id=turns[0].segment_id,
        source_category=turns[0].source_category,
        message_ids=tuple(turn.id for turn in turns),
        first_timestamp_ms=turns[0].timestamp_ms,
        last_timestamp_ms=turns[-1].timestamp_ms,
        text_display=normalized.display,
        text_search=normalized.search,
        participants_search=participants,
        approx_token_count=approximate_token_count(normalized.display),
        source_char_start=source_char_start,
        source_char_end=source_char_end,
        chunk_ordinal=chunk_ordinal,
        chunk_count=chunk_count,
    )


def _time_blocks(turns: Sequence[WindowTurn]) -> Iterator[list[WindowTurn]]:
    block: list[WindowTurn] = []
    previous_timestamp: int | None = None
    for turn in turns:
        if previous_timestamp is not None and turn.timestamp_ms - previous_timestamp > GAP_MS:
            if block:
                yield block
            block = []
        block.append(turn)
        previous_timestamp = turn.timestamp_ms
    if block:
        yield block


def deterministic_message_windows(
    conversation_id: str, turns: Sequence[WindowTurn]
) -> list[WindowDraft]:
    drafts: list[WindowDraft] = []
    for block in _time_blocks(turns):
        regular_block: list[WindowTurn] = []
        for turn in block:
            if approximate_token_count(_render_turns([turn])) > MAX_APPROX_TOKENS:
                drafts.extend(_regular_message_windows(conversation_id, regular_block))
                regular_block = []
                sender_cost = approximate_token_count(f"{turn.sender_display}: ")
                chunks = _text_chunks(turn.text_display, max(1, MAX_APPROX_TOKENS - sender_cost))
                for ordinal, (char_start, char_end, chunk) in enumerate(chunks):
                    drafts.append(
                        _window_draft(
                            conversation_id,
                            [replace(turn, text_display=chunk)],
                            source_char_start=char_start,
                            source_char_end=char_end,
                            chunk_ordinal=ordinal,
                            chunk_count=len(chunks),
                        )
                    )
            else:
                regular_block.append(turn)
        drafts.extend(_regular_message_windows(conversation_id, regular_block))
    return drafts


def _regular_message_windows(
    conversation_id: str, block: Sequence[WindowTurn]
) -> list[WindowDraft]:
    drafts: list[WindowDraft] = []
    if not block:
        return drafts
    start = 0
    while start < len(block):
        end = min(start + MAX_TURNS, len(block))
        while end - start > 1:
            rendered = _render_turns(block[start:end])
            if approximate_token_count(rendered) <= MAX_APPROX_TOKENS:
                break
            end -= 1

        if end < len(block):
            next_size = len(block) - (end - OVERLAP_TURNS)
            if next_size < MIN_TURNS and end - start > MIN_TURNS:
                end -= min(MIN_TURNS - next_size, end - start - MIN_TURNS)

        selected = block[start:end]
        if not selected:
            break
        drafts.append(_window_draft(conversation_id, selected))
        if end >= len(block):
            break
        next_start = end - OVERLAP_TURNS
        start = next_start if next_start > start else end
    return drafts


def _message_text(row: Any) -> str | None:
    parts: list[str] = []
    if row.content_display:
        parts.append(row.content_display)
    if row.transcription_display:
        parts.append(f"[Audio transcript] {row.transcription_display}")
    return "\n".join(parts) if parts else None


def _track_script(summary: WindowBuildSummary, text: str) -> None:
    for character in text:
        if not character.isalpha():
            continue
        summary.total_letter_count += 1
        name = unicodedata.name(character, "")
        if "LATIN" not in name:
            summary.non_latin_letter_count += 1


def _upsert_window_batch(
    session: Session,
    *,
    drafts: Sequence[WindowDraft],
    generation: str,
) -> int:
    if not drafts:
        return 0
    association_count = 0
    for start in range(0, len(drafts), WINDOW_UPSERT_BATCH_SIZE):
        batch = drafts[start : start + WINDOW_UPSERT_BATCH_SIZE]
        values = [
            {
                "id": draft.id,
                "window_version": WINDOW_VERSION,
                "entity_type": "message_window",
                "conversation_id": draft.conversation_id,
                "segment_id": draft.segment_id,
                "activity_id": None,
                "first_message_id": draft.message_ids[0],
                "last_message_id": draft.message_ids[-1],
                "first_timestamp_ms": draft.first_timestamp_ms,
                "last_timestamp_ms": draft.last_timestamp_ms,
                "turn_count": len(draft.message_ids),
                "approx_token_count": draft.approx_token_count,
                "text_display": draft.text_display,
                "text_search": draft.text_search,
                "source_category": draft.source_category,
                "participants_search": draft.participants_search,
                "source_ids_json": {
                    "message_ids": list(draft.message_ids),
                    "source_char_start": draft.source_char_start,
                    "source_char_end": draft.source_char_end,
                    "chunk_ordinal": draft.chunk_ordinal,
                    "chunk_count": draft.chunk_count,
                },
                "build_generation": generation,
            }
            for draft in batch
        ]
        _invalidate_changed_window_memberships(session, values)
        statement = sqlite_insert(SearchWindow).values(values)
        excluded = statement.excluded
        session.execute(
            statement.on_conflict_do_update(
                index_elements=[SearchWindow.id],
                set_={
                    "window_version": excluded.window_version,
                    "entity_type": excluded.entity_type,
                    "conversation_id": excluded.conversation_id,
                    "segment_id": excluded.segment_id,
                    "activity_id": excluded.activity_id,
                    "first_message_id": excluded.first_message_id,
                    "last_message_id": excluded.last_message_id,
                    "first_timestamp_ms": excluded.first_timestamp_ms,
                    "last_timestamp_ms": excluded.last_timestamp_ms,
                    "turn_count": excluded.turn_count,
                    "approx_token_count": excluded.approx_token_count,
                    "text_display": excluded.text_display,
                    "text_search": excluded.text_search,
                    "source_category": excluded.source_category,
                    "participants_search": excluded.participants_search,
                    "source_ids_json": excluded.source_ids_json,
                    "build_generation": excluded.build_generation,
                    "embedding_config_hash": SearchWindow.embedding_config_hash,
                    "embedding_updated_at": SearchWindow.embedding_updated_at,
                    "updated_at": func.current_timestamp(),
                },
            )
        )
        window_ids = [draft.id for draft in batch]
        session.execute(
            delete(SearchWindowMessage).where(SearchWindowMessage.window_id.in_(window_ids))
        )
        associations = [
            {
                "window_id": draft.id,
                "message_id": message_id,
                "ordinal": ordinal,
            }
            for draft in batch
            for ordinal, message_id in enumerate(draft.message_ids)
        ]
        if associations:
            session.execute(sqlite_insert(SearchWindowMessage), associations)
        association_count += len(associations)
    return association_count


_SEMANTIC_WINDOW_IDENTITY_FIELDS = (
    "entity_type",
    "conversation_id",
    "segment_id",
    "activity_id",
    "audio_segment_id",
    "attachment_extraction_id",
    "attachment_occurrence_id",
    "first_message_id",
    "last_message_id",
    "first_timestamp_ms",
    "last_timestamp_ms",
    "turn_count",
    "approx_token_count",
    "text_display",
    "text_search",
    "source_category",
    "participants_search",
    "source_ids_json",
)


def _invalidate_changed_window_memberships(
    session: Session, values: Sequence[dict[str, Any]]
) -> None:
    """Remove every old vector membership when searchable content or metadata changed."""

    if not values:
        return
    incoming = {str(value["id"]): value for value in values}
    existing = {
        row.id: row
        for row in session.scalars(select(SearchWindow).where(SearchWindow.id.in_(incoming)))
    }
    changed_ids = [
        window_id
        for window_id, value in incoming.items()
        if (row := existing.get(window_id)) is not None
        and any(
            getattr(row, field) != value.get(field) for field in _SEMANTIC_WINDOW_IDENTITY_FIELDS
        )
    ]
    if not changed_ids:
        return
    session.execute(
        delete(SemanticWindowEmbedding).where(SemanticWindowEmbedding.window_id.in_(changed_ids))
    )
    session.execute(
        update(SearchWindow)
        .where(SearchWindow.id.in_(changed_ids))
        .values(embedding_config_hash=None, embedding_updated_at=None)
    )


def rebuild_search_windows(
    layout: ArchiveLayout,
    engine: Engine,
    *,
    generation: str | None = None,
) -> WindowBuildSummary:
    """Rebuild searchable windows while serialized with deletion and semantic publication."""

    with deletion_lock(layout), semantic_index_lock(layout):
        assert_derived_publication_allowed(engine, publisher="Search window publication")
        return _rebuild_search_windows_locked(engine, generation=generation)


def _rebuild_search_windows_locked(
    engine: Engine, *, generation: str | None = None
) -> WindowBuildSummary:
    build_generation = generation or str(uuid.uuid4())
    summary = WindowBuildSummary(generation=build_generation)
    with Session(engine) as session:
        conversation_ids = list(
            session.scalars(
                select(Message.conversation_id)
                .where(Message.deleted_at.is_(None))
                .distinct()
                .order_by(Message.conversation_id)
            )
        )

    for conversation_id in conversation_ids:
        with Session(engine) as session:
            rows = session.execute(
                select(
                    Message.id,
                    Message.timestamp_ms,
                    Message.sender_display,
                    Message.sender_search,
                    Message.content_display,
                    Message.transcription_display,
                    Message.segment_id,
                    ConversationSegment.source_category,
                )
                .join(ConversationSegment, ConversationSegment.id == Message.segment_id)
                .where(
                    Message.conversation_id == conversation_id,
                    Message.deleted_at.is_(None),
                )
                .order_by(Message.segment_id, Message.timestamp_ms, Message.id)
            ).all()
            conversation_participants = set(
                session.scalars(
                    select(Participant.name_search)
                    .join(
                        SegmentParticipant,
                        SegmentParticipant.participant_id == Participant.id,
                    )
                    .join(
                        ConversationSegment,
                        ConversationSegment.id == SegmentParticipant.segment_id,
                    )
                    .where(ConversationSegment.conversation_id == conversation_id)
                )
            )
        segment_turns: dict[str, list[WindowTurn]] = {}
        for row in rows:
            text_display = _message_text(row)
            if text_display is None:
                continue
            _track_script(summary, text_display)
            segment_turns.setdefault(row.segment_id, []).append(
                WindowTurn(
                    id=row.id,
                    timestamp_ms=row.timestamp_ms,
                    sender_display=row.sender_display,
                    sender_search=row.sender_search,
                    text_display=text_display,
                    segment_id=row.segment_id,
                    source_category=row.source_category,
                )
            )
        drafts = [
            draft
            for turns in segment_turns.values()
            for draft in deterministic_message_windows(conversation_id, turns)
        ]
        drafts = [
            replace(
                draft,
                participants_search="\n".join(
                    sorted(conversation_participants | set(draft.participants_search.splitlines()))
                ),
            )
            for draft in drafts
        ]
        with Session(engine) as session, session.begin():
            summary.association_count += _upsert_window_batch(
                session, drafts=drafts, generation=build_generation
            )
        summary.message_window_count += len(drafts)
        summary.short_window_count += sum(len(draft.message_ids) < MIN_TURNS for draft in drafts)

    with Session(engine) as session:
        activities = session.scalars(
            select(ActivityRecord)
            .where(ActivityRecord.deleted_at.is_(None))
            .order_by(ActivityRecord.id)
        ).all()
    activity_values: list[dict[str, Any]] = []
    for activity in activities:
        parts = [
            value
            for value in (
                activity.title_display,
                activity.content_display,
                activity.actor_display,
            )
            if value
        ]
        if not parts:
            continue
        display = "\n".join(parts)
        _track_script(summary, display)
        normalized = normalize_text(display)
        chunks = _text_chunks(normalized.display, MAX_APPROX_TOKENS)
        for ordinal, (char_start, char_end, chunk) in enumerate(chunks):
            activity_values.append(
                {
                    "id": stable_id(
                        "semantic-activity-window",
                        WINDOW_VERSION,
                        activity.id,
                        str(ordinal),
                    ),
                    "window_version": WINDOW_VERSION,
                    "entity_type": "activity",
                    "conversation_id": None,
                    "segment_id": None,
                    "activity_id": activity.id,
                    "first_message_id": None,
                    "last_message_id": None,
                    "first_timestamp_ms": activity.timestamp_ms,
                    "last_timestamp_ms": activity.timestamp_ms,
                    "turn_count": 1,
                    "approx_token_count": approximate_token_count(chunk),
                    "text_display": chunk,
                    "text_search": normalize_text(chunk).search,
                    "source_category": activity.category,
                    "participants_search": activity.actor_search or "",
                    "source_ids_json": {
                        "activity_id": activity.id,
                        "source_char_start": char_start,
                        "source_char_end": char_end,
                        "chunk_ordinal": ordinal,
                        "chunk_count": len(chunks),
                    },
                    "build_generation": build_generation,
                }
            )
        if len(activity_values) >= 1000:
            _upsert_activity_windows(engine, activity_values)
            summary.activity_window_count += len(activity_values)
            activity_values = []
    if activity_values:
        _upsert_activity_windows(engine, activity_values)
        summary.activity_window_count += len(activity_values)

    derived_values = _derived_window_values(engine, generation=build_generation)
    for start in range(0, len(derived_values), 1_000):
        _upsert_derived_windows(engine, derived_values[start : start + 1_000])
    summary.derived_window_count = len(derived_values)

    with Session(engine) as session, session.begin():
        session.execute(
            delete(SearchWindow).where(SearchWindow.build_generation != build_generation)
        )
    return summary


def _upsert_activity_windows(engine: Engine, values: list[dict[str, Any]]) -> None:
    with Session(engine) as session, session.begin():
        _invalidate_changed_window_memberships(session, values)
        statement = sqlite_insert(SearchWindow).values(values)
        excluded = statement.excluded
        session.execute(
            statement.on_conflict_do_update(
                index_elements=[SearchWindow.id],
                set_={
                    "window_version": excluded.window_version,
                    "entity_type": excluded.entity_type,
                    "activity_id": excluded.activity_id,
                    "first_timestamp_ms": excluded.first_timestamp_ms,
                    "last_timestamp_ms": excluded.last_timestamp_ms,
                    "approx_token_count": excluded.approx_token_count,
                    "text_display": excluded.text_display,
                    "text_search": excluded.text_search,
                    "source_category": excluded.source_category,
                    "participants_search": excluded.participants_search,
                    "source_ids_json": excluded.source_ids_json,
                    "build_generation": excluded.build_generation,
                    "embedding_config_hash": SearchWindow.embedding_config_hash,
                    "embedding_updated_at": SearchWindow.embedding_updated_at,
                    "updated_at": func.current_timestamp(),
                },
            )
        )


def _derived_participants(engine: Engine) -> dict[str, str]:
    values: dict[str, set[str]] = {}
    with Session(engine) as session:
        rows = session.execute(
            select(ConversationSegment.conversation_id, Participant.name_search)
            .join(
                SegmentParticipant,
                SegmentParticipant.segment_id == ConversationSegment.id,
            )
            .join(Participant, Participant.id == SegmentParticipant.participant_id)
            .order_by(ConversationSegment.conversation_id, Participant.name_search)
        )
        for conversation_id, participant in rows:
            values.setdefault(conversation_id, set()).add(participant)
    return {
        conversation_id: "\n".join(sorted(participants))
        for conversation_id, participants in values.items()
    }


def _derived_window_values(engine: Engine, *, generation: str) -> list[dict[str, Any]]:
    participants = _derived_participants(engine)
    values: list[dict[str, Any]] = []
    with Session(engine) as session:
        audio_rows = session.execute(
            select(
                AudioTranscriptSegment,
                AudioTranscription,
                AttachmentOccurrence,
                Message.segment_id.label("message_segment_id"),
                ConversationSegment.source_category,
                ActivityRecord.id.label("activity_id"),
                ActivityRecord.category.label("activity_category"),
            )
            .join(
                AudioTranscription,
                AudioTranscription.id == AudioTranscriptSegment.transcription_id,
            )
            .join(
                AttachmentOccurrence,
                AttachmentOccurrence.blob_sha256 == AudioTranscription.blob_sha256,
            )
            .outerjoin(Message, Message.id == AttachmentOccurrence.message_id)
            .outerjoin(ConversationSegment, ConversationSegment.id == Message.segment_id)
            .outerjoin(
                ActivityRecord,
                and_(
                    ActivityRecord.id == AttachmentOccurrence.activity_record_id,
                    ActivityRecord.deleted_at.is_(None),
                ),
            )
            .where(
                AudioTranscription.status == "completed",
                AudioTranscription.index_eligible.is_(True),
                AttachmentOccurrence.deleted_at.is_(None),
                or_(AttachmentOccurrence.message_id.is_(None), Message.deleted_at.is_(None)),
            )
            .order_by(AudioTranscriptSegment.id, AttachmentOccurrence.id)
        )
        for (
            segment,
            transcription,
            occurrence,
            message_segment_id,
            source_category,
            activity_id,
            activity_category,
        ) in audio_rows:
            rendered = f"[Audio transcript {segment.start_ms / 1000:.3f}s]\n{segment.text_display}"
            chunks = _text_chunks(rendered, MAX_APPROX_TOKENS)
            for ordinal, (char_start, char_end, chunk) in enumerate(chunks):
                values.append(
                    {
                        "id": stable_id(
                            "semantic-audio-segment",
                            WINDOW_VERSION,
                            segment.id,
                            occurrence.id,
                            str(ordinal),
                        ),
                        "window_version": WINDOW_VERSION,
                        "entity_type": "audio_transcript",
                        "conversation_id": occurrence.conversation_id,
                        "segment_id": message_segment_id,
                        "activity_id": activity_id,
                        "audio_segment_id": segment.id,
                        "attachment_extraction_id": None,
                        "attachment_occurrence_id": occurrence.id,
                        "first_message_id": occurrence.message_id,
                        "last_message_id": occurrence.message_id,
                        "first_timestamp_ms": occurrence.timestamp_ms,
                        "last_timestamp_ms": occurrence.timestamp_ms,
                        "turn_count": 1,
                        "approx_token_count": approximate_token_count(chunk),
                        "text_display": chunk,
                        "text_search": normalize_text(chunk).search,
                        "source_category": source_category or activity_category,
                        "participants_search": participants.get(
                            occurrence.conversation_id or "", ""
                        ),
                        "source_ids_json": {
                            "audio_segment_id": segment.id,
                            "transcription_id": transcription.id,
                            "attachment_occurrence_id": occurrence.id,
                            "source_char_start": char_start,
                            "source_char_end": char_end,
                            "chunk_ordinal": ordinal,
                            "chunk_count": len(chunks),
                            "media_start_ms": segment.start_ms,
                            "media_end_ms": segment.end_ms,
                        },
                        "build_generation": generation,
                    }
                )

        extraction_rows = session.execute(
            select(
                AttachmentTextExtraction,
                AttachmentOccurrence,
                Message.segment_id.label("message_segment_id"),
                ConversationSegment.source_category,
                ActivityRecord.id.label("activity_id"),
                ActivityRecord.category.label("activity_category"),
            )
            .join(
                AttachmentOccurrence,
                AttachmentOccurrence.blob_sha256 == AttachmentTextExtraction.blob_sha256,
            )
            .outerjoin(Message, Message.id == AttachmentOccurrence.message_id)
            .outerjoin(ConversationSegment, ConversationSegment.id == Message.segment_id)
            .outerjoin(
                ActivityRecord,
                and_(
                    ActivityRecord.id == AttachmentOccurrence.activity_record_id,
                    ActivityRecord.deleted_at.is_(None),
                ),
            )
            .where(
                AttachmentTextExtraction.status == "completed",
                AttachmentTextExtraction.index_eligible.is_(True),
                AttachmentTextExtraction.text_display.is_not(None),
                AttachmentOccurrence.deleted_at.is_(None),
                or_(AttachmentOccurrence.message_id.is_(None), Message.deleted_at.is_(None)),
            )
            .order_by(AttachmentTextExtraction.id, AttachmentOccurrence.id)
        )
        for (
            extraction,
            occurrence,
            message_segment_id,
            source_category,
            activity_id,
            activity_category,
        ) in extraction_rows:
            rendered = (
                f"[Attachment text: {extraction.filename or 'file'}]\n{extraction.text_display}"
            )
            chunks = _text_chunks(rendered, MAX_APPROX_TOKENS)
            for ordinal, (char_start, char_end, chunk) in enumerate(chunks):
                values.append(
                    {
                        "id": stable_id(
                            "semantic-attachment-text",
                            WINDOW_VERSION,
                            extraction.id,
                            occurrence.id,
                            str(ordinal),
                        ),
                        "window_version": WINDOW_VERSION,
                        "entity_type": "attachment_text",
                        "conversation_id": occurrence.conversation_id,
                        "segment_id": message_segment_id,
                        "activity_id": activity_id,
                        "audio_segment_id": None,
                        "attachment_extraction_id": extraction.id,
                        "attachment_occurrence_id": occurrence.id,
                        "first_message_id": occurrence.message_id,
                        "last_message_id": occurrence.message_id,
                        "first_timestamp_ms": occurrence.timestamp_ms,
                        "last_timestamp_ms": occurrence.timestamp_ms,
                        "turn_count": 1,
                        "approx_token_count": approximate_token_count(chunk),
                        "text_display": chunk,
                        "text_search": normalize_text(chunk).search,
                        "source_category": source_category or activity_category,
                        "participants_search": participants.get(
                            occurrence.conversation_id or "", ""
                        ),
                        "source_ids_json": {
                            "attachment_extraction_id": extraction.id,
                            "attachment_occurrence_id": occurrence.id,
                            "source_char_start": char_start,
                            "source_char_end": char_end,
                            "chunk_ordinal": ordinal,
                            "chunk_count": len(chunks),
                        },
                        "build_generation": generation,
                    }
                )
    return values


def _upsert_derived_windows(engine: Engine, values: Sequence[dict[str, Any]]) -> None:
    if not values:
        return
    with Session(engine) as session, session.begin():
        _invalidate_changed_window_memberships(session, values)
        statement = sqlite_insert(SearchWindow).values(list(values))
        excluded = statement.excluded
        session.execute(
            statement.on_conflict_do_update(
                index_elements=[SearchWindow.id],
                set_={
                    "window_version": excluded.window_version,
                    "entity_type": excluded.entity_type,
                    "conversation_id": excluded.conversation_id,
                    "segment_id": excluded.segment_id,
                    "activity_id": excluded.activity_id,
                    "audio_segment_id": excluded.audio_segment_id,
                    "attachment_extraction_id": excluded.attachment_extraction_id,
                    "attachment_occurrence_id": excluded.attachment_occurrence_id,
                    "first_message_id": excluded.first_message_id,
                    "last_message_id": excluded.last_message_id,
                    "first_timestamp_ms": excluded.first_timestamp_ms,
                    "last_timestamp_ms": excluded.last_timestamp_ms,
                    "turn_count": excluded.turn_count,
                    "approx_token_count": excluded.approx_token_count,
                    "text_display": excluded.text_display,
                    "text_search": excluded.text_search,
                    "source_category": excluded.source_category,
                    "participants_search": excluded.participants_search,
                    "source_ids_json": excluded.source_ids_json,
                    "build_generation": excluded.build_generation,
                    "embedding_config_hash": SearchWindow.embedding_config_hash,
                    "embedding_updated_at": SearchWindow.embedding_updated_at,
                    "updated_at": func.current_timestamp(),
                },
            )
        )


def _vector_schema(dimensions: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string(), nullable=False),
            pa.field("vector", pa.list_(pa.float32(), dimensions), nullable=False),
            pa.field("entity_type", pa.string(), nullable=False),
            pa.field("conversation_id", pa.string(), nullable=False),
            pa.field("segment_id", pa.string(), nullable=False),
            pa.field("activity_id", pa.string(), nullable=False),
            pa.field("source_category", pa.string(), nullable=False),
            pa.field("participants_search", pa.string(), nullable=False),
            pa.field("first_timestamp_ms", pa.int64(), nullable=True),
            pa.field("last_timestamp_ms", pa.int64(), nullable=True),
            pa.field("jump_id", pa.string(), nullable=False),
            pa.field("text", pa.string(), nullable=False),
            pa.field("build_generation", pa.string(), nullable=False),
        ]
    )


def _vector_row(window: SearchWindow, vector: list[float]) -> dict[str, Any]:
    return {
        "id": window.id,
        "vector": vector,
        "entity_type": window.entity_type,
        "conversation_id": window.conversation_id or "",
        "segment_id": window.segment_id or "",
        "activity_id": window.activity_id or "",
        "source_category": window.source_category or "",
        "participants_search": window.participants_search,
        "first_timestamp_ms": window.first_timestamp_ms,
        "last_timestamp_ms": window.last_timestamp_ms,
        "jump_id": (
            window.first_message_id or window.activity_id or window.attachment_occurrence_id or ""
        ),
        "text": window.text_display,
        "build_generation": window.build_generation,
    }


def _reset_semantic_checkpoint(
    database: Any,
    *,
    vector_table: str,
    dimensions: int,
    engine: Engine,
    config_hash: str,
) -> Any:
    database.drop_table(vector_table, ignore_missing=True)
    table = database.create_table(vector_table, schema=_vector_schema(dimensions))
    with Session(engine) as session, session.begin():
        session.execute(
            delete(SemanticWindowEmbedding).where(SemanticWindowEmbedding.config_id == config_hash)
        )
    return table


def _reconcile_semantic_checkpoint(
    table: Any,
    database: Any,
    *,
    vector_table: str,
    dimensions: int,
    engine: Engine,
    config_hash: str,
) -> tuple[Any, int, str | None]:
    """Validate the Lance prefix and repair only crash-safe missing SQL memberships."""

    validation = validate_semantic_table(
        table,
        engine,
        config_id=config_hash,
        dimensions=dimensions,
        require_complete=False,
        repair_missing_memberships=True,
        raise_on_error=False,
    )
    # A structurally incomplete prefix can arise when a process stops between Lance operations;
    # reset that checkpoint exactly as before. Same-ID vector or retrieval-metadata corruption is
    # qualitatively different: silently rebuilding would hide failed integrity evidence, so fail
    # closed and leave the prior active product's state visible to the outer recovery handler.
    if validation.invalid_vector_rows or validation.metadata_mismatch_rows:
        raise SemanticTableIntegrityError(
            "Semantic checkpoint contains invalid vectors or retrieval metadata "
            f"(invalid_vectors={validation.invalid_vector_rows}, "
            f"metadata_mismatches={validation.metadata_mismatch_rows})"
        )
    if not validation.valid:
        table = _reset_semantic_checkpoint(
            database,
            vector_table=vector_table,
            dimensions=dimensions,
            engine=engine,
            config_hash=config_hash,
        )
        return table, 0, None
    return table, validation.row_count, validation.last_window_id


def _assert_complete_semantic_checkpoint(
    table: Any,
    engine: Engine,
    *,
    config_hash: str,
    dimensions: int,
) -> int:
    validation = validate_semantic_table(
        table,
        engine,
        config_id=config_hash,
        dimensions=dimensions,
        require_complete=True,
    )
    return validation.row_count


def _index_covers_table(table: Any, *, column: str, total_rows: int) -> bool:
    for index in table.list_indices():
        if list(index.columns) != [column]:
            continue
        if int(index.num_indexed_rows) == total_rows and int(index.num_unindexed_rows) == 0:
            return True
    return False


def _validate_embedding_vectors(
    vectors: Sequence[Sequence[float]], *, expected_count: int, dimensions: int
) -> None:
    if len(vectors) != expected_count:
        raise RuntimeError("Embedding backend returned an unexpected vector count")
    for vector in vectors:
        values = np.asarray(vector, dtype=np.float32)
        if values.shape != (dimensions,) or not bool(np.isfinite(values).all()):
            raise RuntimeError("Embedding backend returned an invalid vector shape or value")
        norm = float(np.linalg.norm(values))
        if not math.isclose(norm, 1.0, rel_tol=1e-3, abs_tol=1e-3):
            raise RuntimeError("Embedding backend violated the recorded L2-normalization contract")


def _assert_semantic_candidate_current(
    engine: Engine,
    *,
    config_hash: str,
    job_id: str,
    candidate_was_active_ready: bool,
) -> None:
    with Session(engine) as session:
        _assert_semantic_candidate_current_in_session(
            session,
            config_hash=config_hash,
            job_id=job_id,
            candidate_was_active_ready=candidate_was_active_ready,
        )


def _assert_semantic_candidate_current_in_session(
    session: Session,
    *,
    config_hash: str,
    job_id: str,
    candidate_was_active_ready: bool,
) -> None:
    config = session.get(SemanticIndexConfig, config_hash)
    job = session.get(Job, job_id)
    expected_status = "ready" if candidate_was_active_ready else "building"
    if (
        config is None
        or config.status != expected_status
        or (candidate_was_active_ready and not config.active)
        or job is None
        or job.status != "running"
    ):
        raise RuntimeError(
            "Semantic candidate was invalidated by a concurrent searchable corpus change"
        )


@contextmanager
def _immediate_session(engine: Engine) -> Iterator[Session]:
    """Reserve SQLite's writer slot while validating and publishing an index."""

    with engine.connect() as connection:
        connection.exec_driver_sql("BEGIN IMMEDIATE")
        session = Session(bind=connection)
        try:
            yield session
            session.flush()
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            session.close()


def _prepare_job(
    engine: Engine,
    *,
    config_hash: str,
    model_name: str,
    requested_job_id: str | None = None,
) -> tuple[str, bool]:
    with Session(engine) as session, session.begin():
        now = datetime.now(UTC)
        stale_statement = select(Job).where(
            Job.kind == "semantic_index",
            Job.status == "running",
        )
        if requested_job_id is not None:
            stale_statement = stale_statement.where(Job.id != requested_job_id)
        stale_jobs = list(session.scalars(stale_statement))
        for stale in stale_jobs:
            stale.status = "failed"
            stale.stage = "Interrupted semantic index recovered"
            stale.error_text = "Recovered after the prior process lost the semantic-index lock"
            stale.finished_at = now
        previous = session.scalar(
            select(Job)
            .where(Job.kind == "semantic_index", Job.status == "failed")
            .order_by(Job.started_at.desc(), Job.created_at.desc(), Job.id.desc())
            .limit(1)
        )
        resumable = bool(
            previous is not None and previous.checkpoint_json.get("config_hash") == config_hash
        )
        if requested_job_id is not None:
            requested = session.get(Job, requested_job_id)
            if requested is None or requested.kind != "semantic_index":
                raise RuntimeError("Queued semantic index job was not found")
            if requested.status not in {"queued", "running"}:
                raise RuntimeError("Queued semantic index job was invalidated during preflight")
            requested.status = "running"
            requested.stage = "Resuming semantic index" if resumable else "Embedding search windows"
            requested.error_text = None
            requested.finished_at = None
            requested.started_at = now
            requested.checkpoint_json = {"config_hash": config_hash}
            requested.counters_json = {"model_name": model_name}
            return requested.id, resumable
        if resumable and previous is not None:
            previous.status = "running"
            previous.error_text = None
            previous.finished_at = None
            previous.stage = "Resuming semantic index"
            previous.started_at = now
            return previous.id, True
        job_id = str(uuid.uuid4())
        session.add(
            Job(
                id=job_id,
                kind="semantic_index",
                status="running",
                stage="Building deterministic search windows",
                progress_current=0,
                checkpoint_json={"config_hash": config_hash},
                counters_json={"model_name": model_name},
                started_at=now,
            )
        )
        return job_id, False


def _search_window_corpus_revision_in_session(session: Session) -> str:
    digest = hashlib.sha256()
    rows = session.execute(
        select(
            SearchWindow.id,
            SearchWindow.entity_type,
            SearchWindow.conversation_id,
            SearchWindow.segment_id,
            SearchWindow.activity_id,
            SearchWindow.audio_segment_id,
            SearchWindow.attachment_extraction_id,
            SearchWindow.attachment_occurrence_id,
            SearchWindow.first_message_id,
            SearchWindow.last_message_id,
            SearchWindow.first_timestamp_ms,
            SearchWindow.last_timestamp_ms,
            SearchWindow.source_category,
            SearchWindow.participants_search,
            SearchWindow.text_display,
            SearchWindow.text_search,
        ).order_by(SearchWindow.id)
    )
    for row in rows:
        digest.update(
            json.dumps(tuple(row), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def search_window_corpus_revision(engine: Engine) -> str:
    with Session(engine) as session:
        return _search_window_corpus_revision_in_session(session)


def search_window_language_audit(engine: Engine, *, sample_limit: int = 2_000) -> dict[str, Any]:
    """Run a deterministic, bounded language audit over stable-ID-ordered windows."""

    from lingua import Language, LanguageDetectorBuilder

    with Session(engine) as session:
        texts = list(
            session.scalars(
                select(SearchWindow.text_display)
                .where(func.length(SearchWindow.text_display) >= 20)
                .order_by(SearchWindow.id)
                .limit(sample_limit)
            )
        )
    detector = LanguageDetectorBuilder.from_all_languages().with_low_accuracy_mode().build()
    detected = detector.detect_languages_in_parallel_of(texts) if texts else []
    counts: dict[str, int] = {}
    for language in detected:
        if language is None:
            continue
        counts[language.name] = counts.get(language.name, 0) + 1
    confident_count = sum(counts.values())
    non_english_count = confident_count - counts.get(Language.ENGLISH.name, 0)
    non_english_ratio = non_english_count / confident_count if confident_count else 0.0
    recommended_model = AUTO_EMBEDDING_MODEL_ALIAS
    return {
        "method": "lingua-2.2-low-accuracy-stable-window-sample",
        "sample_limit": sample_limit,
        "sampled_window_count": len(texts),
        "confident_window_count": confident_count,
        "language_counts": dict(sorted(counts.items())),
        "non_english_ratio": non_english_ratio,
        "threshold": 0.05,
        "threshold_used_for_selection": False,
        "selection_policy": SEMANTIC_MODEL_SELECTION_POLICY,
        "recommended_model": recommended_model,
    }


_MANUAL_ACTIVATION_VERSION = "semantic-manual-activation-v1"
_MANUAL_ACTIVATION_REASON = (
    "User-directed CLI activation after acknowledging the rejected candidate's failed "
    "automatic quality gate."
)


def _manual_activation_registered_spec(
    config: SemanticIndexConfig,
) -> EmbeddingModelSpec:
    audit = config.audit_json if isinstance(config.audit_json, dict) else {}
    selected_model = audit.get("selected_model")
    if not isinstance(selected_model, dict):
        raise SemanticIndexManualActivationError(
            "Rejected candidate does not record its selected model specification"
        )
    alias = selected_model.get("alias")
    spec = MODEL_SPECS.get(alias) if isinstance(alias, str) else None
    if spec is None or selected_model != asdict(spec):
        raise SemanticIndexManualActivationError(
            "Rejected candidate model specification is not a current registered model"
        )
    expected_config = {
        "backend": embedding_backend(spec)[0],
        "backend_version": embedding_backend(spec)[1],
        "model_name": spec.model_name,
        "model_revision": spec.model_revision,
        "dimensions": spec.dimensions,
        "distance_metric": "cosine",
        "document_prefix": spec.document_prefix,
        "query_prefix": spec.query_prefix,
        "vector_table": f"text_windows_{config.id[:16]}",
        "window_version": WINDOW_VERSION,
    }
    mismatched = [
        key for key, expected in expected_config.items() if getattr(config, key) != expected
    ]
    if mismatched:
        raise SemanticIndexManualActivationError(
            "Rejected candidate configuration no longer matches the implementation: "
            + ", ".join(sorted(mismatched))
        )
    return spec


def _manual_activation_catalog_preflight(
    session: Session,
    *,
    config_hash: str,
) -> tuple[
    SemanticIndexConfig,
    SemanticIndexConfig,
    EmbeddingModelSpec,
    dict[str, Any],
    str,
    int,
]:
    active_jobs = list(
        session.scalars(
            select(Job)
            .where(Job.status.in_(("queued", "running")))
            .order_by(Job.created_at, Job.id)
        )
    )
    if active_jobs:
        raise SemanticIndexManualActivationError(
            "Manual activation requires every queued or running job to finish first"
        )
    candidate = session.get(SemanticIndexConfig, config_hash)
    if candidate is None:
        raise SemanticIndexManualActivationError(
            f"Rejected semantic candidate {config_hash} was not found"
        )
    if candidate.active or candidate.status != "rejected":
        raise SemanticIndexManualActivationError(
            "Manual activation requires an inactive candidate with status 'rejected'"
        )
    audit = candidate.audit_json if isinstance(candidate.audit_json, dict) else {}
    evaluation = audit.get("semantic_ir_evaluation")
    if not isinstance(evaluation, dict) or not (
        evaluation.get("passed") is False
        and evaluation.get("production_acceptance_passed") is False
        and evaluation.get("evaluator_version")
        in {
            PREVIOUS_SEMANTIC_IR_EVALUATOR_VERSION,
            SEMANTIC_IR_EVALUATOR_VERSION,
        }
    ):
        raise SemanticIndexManualActivationError(
            "Rejected candidate does not retain an explicit failed quality evaluation"
        )
    evaluation_candidate = evaluation.get("candidate")
    corpus_revision = audit.get("corpus_revision")
    if not isinstance(evaluation_candidate, dict) or not isinstance(corpus_revision, str):
        raise SemanticIndexManualActivationError(
            "Rejected candidate does not retain complete corpus evaluation provenance"
        )
    expected_evaluation_candidate = {
        "config_hash": config_hash,
        "vector_table": candidate.vector_table,
        "corpus_revision": corpus_revision,
        "window_count": candidate.window_count,
        "distance_metric": candidate.distance_metric,
    }
    if any(
        evaluation_candidate.get(key) != value
        for key, value in expected_evaluation_candidate.items()
    ):
        raise SemanticIndexManualActivationError(
            "Rejected candidate evaluation does not match its persisted configuration"
        )
    active_configs = list(
        session.scalars(
            select(SemanticIndexConfig).where(
                SemanticIndexConfig.active.is_(True),
                SemanticIndexConfig.status == "ready",
            )
        )
    )
    if len(active_configs) != 1:
        raise SemanticIndexManualActivationError(
            "Manual activation requires exactly one ready active incumbent"
        )
    incumbent = active_configs[0]
    incumbent_audit = incumbent.audit_json if isinstance(incumbent.audit_json, dict) else {}
    if incumbent_audit.get("corpus_revision") != corpus_revision:
        raise SemanticIndexManualActivationError(
            "Rejected candidate and active incumbent do not cover the same corpus revision"
        )
    catalog_window_count = int(session.scalar(select(func.count(SearchWindow.id))) or 0)
    membership_count = int(
        session.scalar(
            select(func.count(SemanticWindowEmbedding.window_id)).where(
                SemanticWindowEmbedding.config_id == config_hash
            )
        )
        or 0
    )
    incumbent_binding_count = int(
        session.scalar(
            select(func.count(SearchWindow.id)).where(
                SearchWindow.embedding_config_hash == incumbent.id
            )
        )
        or 0
    )
    if (
        candidate.window_count != catalog_window_count
        or membership_count != catalog_window_count
        or incumbent_binding_count != catalog_window_count
    ):
        raise SemanticIndexManualActivationError(
            "Rejected candidate, membership, catalog, or incumbent binding counts diverge"
        )
    if candidate.completed_at is None or candidate.job_id is None:
        raise SemanticIndexManualActivationError(
            "Rejected candidate does not record a completed build"
        )
    build_job = session.get(Job, candidate.job_id)
    if (
        build_job is None
        or build_job.kind != "semantic_index"
        or build_job.status != "failed"
        or build_job.checkpoint_json.get("config_hash") != config_hash
        or build_job.counters_json.get("config_hash") != config_hash
        or build_job.counters_json.get("acceptance_passed") is not False
        or build_job.progress_current != catalog_window_count
        or build_job.progress_total != catalog_window_count
    ):
        raise SemanticIndexManualActivationError(
            "Rejected candidate's failed build job is incomplete or inconsistent"
        )
    spec = _manual_activation_registered_spec(candidate)
    implementation = embedding_implementation_identity(spec)
    if audit.get("embedding_implementation") != implementation:
        raise SemanticIndexManualActivationError(
            "Rejected candidate embedding implementation is not current"
        )
    artifact_fingerprint = audit.get("artifact_fingerprint")
    if not isinstance(artifact_fingerprint, str) or not artifact_fingerprint:
        raise SemanticIndexManualActivationError(
            "Rejected candidate model artifact fingerprint is missing"
        )
    expected_hash = embedding_config_hash(
        spec,
        corpus_revision=corpus_revision,
        artifact_fingerprint=artifact_fingerprint,
    )
    if expected_hash != config_hash:
        raise SemanticIndexManualActivationError(
            "Rejected candidate config ID does not match its artifact and corpus identity"
        )
    current_corpus_revision = _search_window_corpus_revision_in_session(session)
    if current_corpus_revision != corpus_revision:
        raise SemanticIndexManualActivationError(
            "Rejected candidate no longer matches the searchable corpus"
        )
    return (
        candidate,
        incumbent,
        spec,
        cast(dict[str, Any], evaluation),
        artifact_fingerprint,
        catalog_window_count,
    )


def _manual_activation_lance_spot_check(
    table: Any,
    engine: Engine,
    *,
    config_hash: str,
    dimensions: int,
    row_count: int,
    sample_limit: int = 64,
) -> dict[str, Any]:
    """Deterministically sample vector and retrieval metadata before manual publication."""

    if row_count < 1:
        raise SemanticIndexManualActivationError(
            "Rejected candidate Lance table does not contain any rows"
        )
    sample_count = min(row_count, sample_limit)
    offsets = (
        [0]
        if sample_count == 1
        else [round(index * (row_count - 1) / (sample_count - 1)) for index in range(sample_count)]
    )
    columns = [
        "id",
        "vector",
        "entity_type",
        "conversation_id",
        "segment_id",
        "activity_id",
        "source_category",
        "participants_search",
        "first_timestamp_ms",
        "last_timestamp_ms",
        "jump_id",
        "text",
        "build_generation",
    ]
    rows = table.take_offsets(offsets).select(columns).to_list()
    if len(rows) != sample_count:
        raise SemanticIndexManualActivationError(
            "Rejected candidate Lance spot check returned an incomplete sample"
        )
    identifiers = [str(row.get("id") or "") for row in rows]
    if any(not identifier for identifier in identifiers) or len(set(identifiers)) != sample_count:
        raise SemanticIndexManualActivationError(
            "Rejected candidate Lance spot check found missing or duplicate IDs"
        )
    try:
        _validate_embedding_vectors(
            [cast(Sequence[float], row.get("vector")) for row in rows],
            expected_count=sample_count,
            dimensions=dimensions,
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise SemanticIndexManualActivationError(
            "Rejected candidate Lance spot check found an invalid vector"
        ) from exc

    with Session(engine) as session:
        windows = {
            window.id: window
            for window in session.scalars(
                select(SearchWindow).where(SearchWindow.id.in_(identifiers))
            )
        }
        memberships = set(
            session.scalars(
                select(SemanticWindowEmbedding.window_id).where(
                    SemanticWindowEmbedding.config_id == config_hash,
                    SemanticWindowEmbedding.window_id.in_(identifiers),
                )
            )
        )
    if set(identifiers) != set(windows) or set(identifiers) != memberships:
        raise SemanticIndexManualActivationError(
            "Rejected candidate Lance spot check diverges from catalog membership"
        )
    metadata_fields = [column for column in columns if column not in {"id", "vector"}]
    for row, identifier in zip(rows, identifiers, strict=True):
        expected = _vector_row(windows[identifier], [])
        if any(row.get(field) != expected[field] for field in metadata_fields):
            raise SemanticIndexManualActivationError(
                "Rejected candidate Lance spot check found retrieval metadata drift"
            )
    return {
        "strategy": "evenly_spaced_physical_offsets_v1",
        "sampled_row_count": sample_count,
        "sample_limit": sample_limit,
        "offsets_sha256": hashlib.sha256(
            json.dumps(offsets, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "ids_memberships_vectors_metadata_matched": True,
    }


def activate_rejected_semantic_candidate(
    layout: ArchiveLayout,
    engine: Engine,
    *,
    config_hash: str,
    acknowledge_failed_quality_gate: bool,
) -> SemanticManualActivationSummary:
    """Explicitly activate one complete rejected candidate without rewriting its evaluation."""

    if not re.fullmatch(r"[0-9a-f]{64}", config_hash):
        raise SemanticIndexManualActivationError(
            "Manual activation requires the exact 64-character lowercase config ID"
        )
    if acknowledge_failed_quality_gate is not True:
        raise SemanticIndexManualActivationError(
            "Pass --acknowledge-failed-quality-gate to confirm this candidate failed "
            "automatic retrieval acceptance"
        )
    with deletion_lock(layout), semantic_index_lock(layout):
        assert_derived_publication_allowed(
            engine,
            publisher="Manual semantic candidate activation",
        )
        with Session(engine) as session:
            (
                candidate,
                incumbent,
                spec,
                failed_evaluation,
                artifact_fingerprint,
                catalog_window_count,
            ) = _manual_activation_catalog_preflight(
                session,
                config_hash=config_hash,
            )
            candidate_vector_table = candidate.vector_table
            candidate_dimensions = candidate.dimensions
            previous_active_config_id = incumbent.id
            original_failed_evaluation = json.loads(json.dumps(failed_evaluation))

        active_embedder = default_text_embedder(layout, spec, local_files_only=True)
        if embedding_artifact_fingerprint(active_embedder) != artifact_fingerprint:
            raise SemanticIndexManualActivationError(
                "Installed model artifact no longer matches the rejected candidate"
            )

        database = lancedb.connect(str(layout.vectors))
        if candidate_vector_table not in lance_table_names(database):
            raise SemanticIndexManualActivationError("Rejected candidate Lance table is missing")
        table = database.open_table(candidate_vector_table)
        lance_row_count = int(table.count_rows())
        vector_field = table.schema.field("vector")
        if (
            lance_row_count != catalog_window_count
            or not pa.types.is_fixed_size_list(vector_field.type)
            or vector_field.type.list_size != candidate_dimensions
        ):
            raise SemanticIndexManualActivationError(
                "Rejected candidate Lance row count or vector dimensions diverge"
            )
        lance_spot_check = _manual_activation_lance_spot_check(
            table,
            engine,
            config_hash=config_hash,
            dimensions=candidate_dimensions,
            row_count=lance_row_count,
        )

        activated_at = datetime.now(UTC)
        activation_job_id = str(uuid.uuid4())
        with _immediate_session(engine) as session:
            (
                locked_candidate,
                locked_incumbent,
                locked_spec,
                locked_failed_evaluation,
                locked_artifact_fingerprint,
                locked_catalog_window_count,
            ) = _manual_activation_catalog_preflight(
                session,
                config_hash=config_hash,
            )
            if (
                locked_incumbent.id != previous_active_config_id
                or locked_spec != spec
                or locked_artifact_fingerprint != artifact_fingerprint
                or locked_catalog_window_count != catalog_window_count
                or locked_failed_evaluation != original_failed_evaluation
            ):
                raise SemanticIndexManualActivationError(
                    "Rejected candidate changed during manual activation preflight"
                )
            locked_audit = (
                locked_candidate.audit_json if isinstance(locked_candidate.audit_json, dict) else {}
            )
            language_audit = locked_audit.get("language_audit")
            current_language_audit = (
                {
                    **language_audit,
                    "recommended_model": AUTO_EMBEDDING_MODEL_ALIAS,
                    "selection_policy": SEMANTIC_MODEL_SELECTION_POLICY,
                }
                if isinstance(language_audit, dict)
                else None
            )
            prior_selection_metadata = {
                key: locked_audit.get(key)
                for key in (
                    "recommended_model",
                    "model_selection",
                    "selection_note",
                    "selection_policy",
                )
                if key in locked_audit
            }
            if isinstance(language_audit, dict):
                prior_selection_metadata["language_audit"] = {
                    "recommended_model": language_audit.get("recommended_model"),
                    "selection_policy": language_audit.get("selection_policy"),
                }
            manual_override = {
                "version": _MANUAL_ACTIVATION_VERSION,
                "activated_at": activated_at.isoformat(),
                "candidate_config_id": config_hash,
                "previous_active_config_id": previous_active_config_id,
                "acknowledged_failed_quality_gate": True,
                "reason": _MANUAL_ACTIVATION_REASON,
                "source": "cli_only",
                "failed_evaluation_preserved": True,
                "source_build_job_id": locked_candidate.job_id,
                "activation_job_id": activation_job_id,
                "prior_selection_metadata": prior_selection_metadata,
                "validation": {
                    "catalog_window_count": catalog_window_count,
                    "membership_count": catalog_window_count,
                    "lance_row_count": lance_row_count,
                    "vector_dimensions": candidate_dimensions,
                    "corpus_revision_matched_incumbent": True,
                    "artifact_fingerprint_matched": True,
                    "embedding_implementation_matched": True,
                    "lance_spot_check": lance_spot_check,
                    "failed_quality_evaluator_version": locked_failed_evaluation[
                        "evaluator_version"
                    ],
                },
                "literal_direct_conflict_runtime": (
                    "normal_cross_encoder_when_specialized_calibration_is_not_accepted"
                ),
            }
            refreshed_audit = {
                **locked_audit,
                "recommended_model": AUTO_EMBEDDING_MODEL_ALIAS,
                "model_selection": "explicit_manual_override",
                "selection_policy": SEMANTIC_MODEL_SELECTION_POLICY,
                "selection_note": (
                    f"The rejected semantic candidate {locked_spec.model_name} was explicitly "
                    "activated by CLI after its failed compatibility evaluation was acknowledged."
                ),
                "manual_activation_override": manual_override,
            }
            if current_language_audit is not None:
                refreshed_audit["language_audit"] = current_language_audit
            if refreshed_audit.get("semantic_ir_evaluation") != original_failed_evaluation:
                raise SemanticIndexManualActivationError(
                    "Manual activation attempted to alter the failed quality evaluation"
                )

            session.execute(update(SemanticIndexConfig).values(active=False))
            session.execute(
                update(SearchWindow).values(
                    embedding_config_hash=config_hash,
                    embedding_updated_at=activated_at,
                )
            )
            locked_candidate.active = True
            locked_candidate.status = "ready"
            locked_candidate.job_id = activation_job_id
            locked_candidate.audit_json = refreshed_audit
            session.add(
                Job(
                    id=activation_job_id,
                    kind="semantic_index",
                    status="completed",
                    stage="Semantic index manually activated",
                    progress_current=catalog_window_count,
                    progress_total=catalog_window_count,
                    checkpoint_json={
                        "config_hash": config_hash,
                        "previous_active_config_id": previous_active_config_id,
                        "source_build_job_id": manual_override["source_build_job_id"],
                    },
                    counters_json={
                        "config_hash": config_hash,
                        "model_name": locked_candidate.model_name,
                        "window_count": catalog_window_count,
                        "manual_activation_override": True,
                        "acceptance_passed": False,
                        "production_acceptance_passed": False,
                    },
                    started_at=activated_at,
                    finished_at=activated_at,
                )
            )

        try:
            from meta_archive.services.conflict_aggregates import (
                invalidate_conflict_aggregates,
            )

            invalidated_conflict_runs = invalidate_conflict_aggregates(
                engine,
                reason=(
                    "Semantic index manually changed after a failed quality gate; "
                    "rebuild conflict aggregates"
                ),
            )
            invalidation_result: dict[str, Any] = {
                "status": "completed",
                "invalidated_run_count": invalidated_conflict_runs,
                "completed_at": datetime.now(UTC).isoformat(),
            }
        except Exception as exc:
            invalidated_conflict_runs = 0
            invalidation_result = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "failed_at": datetime.now(UTC).isoformat(),
            }
        with Session(engine) as session, session.begin():
            activated = session.get(SemanticIndexConfig, config_hash)
            if activated is not None:
                override = activated.audit_json.get("manual_activation_override")
                activated.audit_json = {
                    **activated.audit_json,
                    "manual_activation_override": {
                        **(override if isinstance(override, dict) else {}),
                        "conflict_aggregate_invalidation": invalidation_result,
                    },
                    "conflict_aggregate_invalidation": invalidation_result,
                }
        return SemanticManualActivationSummary(
            activation_job_id=activation_job_id,
            config_hash=config_hash,
            previous_active_config_id=previous_active_config_id,
            vector_table=candidate_vector_table,
            window_count=catalog_window_count,
            model_name=spec.model_name,
            invalidated_conflict_run_count=invalidated_conflict_runs,
        )


def build_semantic_index(
    layout: ArchiveLayout,
    engine: Engine,
    *,
    embedder: TextEmbedder | None = None,
    embedder_factory: TextEmbedderFactory | None = None,
    reranker_factory: RerankerFactory | None = None,
    model_alias: str | None = None,
    rebuild_windows: bool = True,
    batch_size: int | None = None,
    job_id: str | None = None,
) -> SemanticIndexSummary:
    try:
        # Keep the deletion lock for the complete build. A deletion commit cannot capture its
        # exact restore snapshot until this publisher finishes, and a publisher that starts after
        # commit observes the blocking operation while both locks are held.
        with deletion_lock(layout), semantic_index_lock(layout):
            assert_derived_publication_allowed(engine, publisher="Semantic index publication")
            return _build_semantic_index_locked(
                layout,
                engine,
                embedder=embedder,
                embedder_factory=embedder_factory,
                reranker_factory=reranker_factory,
                model_alias=model_alias,
                rebuild_windows=rebuild_windows,
                batch_size=batch_size,
                requested_job_id=job_id,
            )
    except BaseException as exc:
        # Requested jobs exist before window rebuilding, language audit, model acquisition, and
        # fingerprinting. Ensure every ordinary preflight failure is visible; later inner failure
        # handling wins because this update only targets jobs that are still active.
        if job_id is not None and not isinstance(exc, DerivedPublicationBlockedError):
            with Session(engine) as session, session.begin():
                session.execute(
                    update(Job)
                    .where(Job.id == job_id, Job.status.in_(("queued", "running")))
                    .values(
                        status="failed",
                        stage="Semantic index failed during preflight",
                        error_text=f"{type(exc).__name__}: {exc}",
                        finished_at=datetime.now(UTC),
                    )
                )
        raise


def _build_semantic_index_locked(
    layout: ArchiveLayout,
    engine: Engine,
    *,
    embedder: TextEmbedder | None,
    embedder_factory: TextEmbedderFactory | None,
    reranker_factory: RerankerFactory | None,
    model_alias: str | None,
    rebuild_windows: bool,
    batch_size: int | None,
    requested_job_id: str | None,
) -> SemanticIndexSummary:
    if requested_job_id is not None:
        # This update happens only after the process-safe file lock is held. A `running` job
        # without that lock therefore unambiguously represents an interrupted worker and can be
        # recovered by the next request/build.
        with Session(engine) as session, session.begin():
            requested = session.get(Job, requested_job_id)
            if requested is None or requested.kind != "semantic_index":
                raise RuntimeError("Queued semantic index job was not found")
            if requested.status not in {"queued", "running"}:
                raise RuntimeError("Queued semantic index job was invalidated before it started")
            requested.status = "running"
            requested.stage = "Building deterministic search windows"
            requested.started_at = datetime.now(UTC)
            requested.error_text = None
            requested.finished_at = None
    if rebuild_windows:
        window_summary = _rebuild_search_windows_locked(engine)
    else:
        with Session(engine) as session:
            generation = session.scalar(
                select(SearchWindow.build_generation).order_by(SearchWindow.id).limit(1)
            )
        if generation is None:
            window_summary = _rebuild_search_windows_locked(engine)
        else:
            window_summary = WindowBuildSummary(generation=generation)
    corpus_revision = search_window_corpus_revision(engine)
    language_audit = search_window_language_audit(engine)
    if embedder is not None:
        spec = embedder.spec
        model_selection = "injected_embedder"
    else:
        configured_model = model_alias or settings.embedding_model
        selected_alias = (
            AUTO_EMBEDDING_MODEL_ALIAS if configured_model == "auto" else configured_model
        )
        spec = embedding_model_spec(selected_alias)
        model_selection = "automatic_default" if configured_model == "auto" else "explicit_config"
    active_embedder = (
        embedder
        or (embedder_factory(spec) if embedder_factory is not None else None)
        or default_text_embedder(layout, spec)
    )
    artifact_fingerprint = embedding_artifact_fingerprint(active_embedder)
    config_hash = embedding_config_hash(
        spec,
        corpus_revision=corpus_revision,
        artifact_fingerprint=artifact_fingerprint,
    )
    vector_table = f"text_windows_{config_hash[:16]}"
    job_id, resumed = _prepare_job(
        engine,
        config_hash=config_hash,
        model_name=spec.model_name,
        requested_job_id=requested_job_id,
    )
    embedded_count = 0
    candidate_was_active_ready = False
    prior_candidate_audit: dict[str, Any] = {}
    try:
        actual_batch_size = _validated_embedding_batch_size(spec, batch_size)
        audit = {
            "model_candidates": [asdict(candidate) for candidate in MODEL_SPECS.values()],
            "selected_model": asdict(spec),
            "observed_non_latin_letter_ratio": window_summary.non_latin_ratio,
            "language_audit": language_audit,
            "recommended_model": language_audit["recommended_model"],
            "model_selection": model_selection,
            "selection_policy": SEMANTIC_MODEL_SELECTION_POLICY,
            "artifact_fingerprint": artifact_fingerprint,
            "embedding_implementation": embedding_implementation_identity(spec),
            "corpus_revision": corpus_revision,
            "selection_note": (
                "Automatic selection uses the pinned English BGE-base model. The language audit "
                "is informational; other registered models remain explicit candidate choices."
            ),
            "window_policy": {
                "minimum_turns": MIN_TURNS,
                "maximum_turns": MAX_TURNS,
                "overlap_turns": OVERLAP_TURNS,
                "gap_minutes": GAP_MS // 60_000,
                "approximate_token_cap": MAX_APPROX_TOKENS,
                "version": WINDOW_VERSION,
            },
        }
        with Session(engine) as session, session.begin():
            row = session.get(SemanticIndexConfig, config_hash)
            candidate_was_active_ready = bool(
                row is not None and row.active and row.status == "ready"
            )
            prior_candidate_audit = dict(row.audit_json) if row is not None else {}
            backend, backend_version = embedding_backend(spec)
            values = {
                "active": False,
                "status": "building",
                "backend": backend,
                "backend_version": backend_version,
                "model_name": spec.model_name,
                "model_revision": spec.model_revision,
                "dimensions": spec.dimensions,
                "distance_metric": "cosine",
                "document_prefix": spec.document_prefix,
                "query_prefix": spec.query_prefix,
                "vector_table": vector_table,
                "window_version": WINDOW_VERSION,
                "job_id": job_id,
                "audit_json": audit,
                "completed_at": None,
            }
            if row is None:
                session.add(SemanticIndexConfig(id=config_hash, window_count=0, **values))
            else:
                for key, value in values.items():
                    if candidate_was_active_ready and key in {
                        "active",
                        "status",
                        "audit_json",
                        "completed_at",
                    }:
                        continue
                    setattr(row, key, value)

        database = lancedb.connect(str(layout.vectors))
        table_exists = vector_table in lance_table_names(database)
        if table_exists:
            table = database.open_table(vector_table)
            if rebuild_windows and table.count_rows():
                # ``build_generation`` is mark-and-sweep bookkeeping rather than vector input,
                # so an otherwise identical deterministic rebuild keeps the same config hash.
                # Synchronize that one field before strict metadata validation; every retrieval
                # field and text hash remains fail-closed below.
                table.update(values={"build_generation": window_summary.generation})
        else:
            table = database.create_table(vector_table, schema=_vector_schema(spec.dimensions))

        table, preexisting_embedded_count, last_window_id = _reconcile_semantic_checkpoint(
            table,
            database,
            vector_table=vector_table,
            dimensions=spec.dimensions,
            engine=engine,
            config_hash=config_hash,
        )

        with Session(engine) as session:
            total = session.scalar(select(func.count(SearchWindow.id))) or 0
        if candidate_was_active_ready and preexisting_embedded_count != total:
            # The only table supporting this active config is already incomplete/corrupt. Do not
            # advertise an empty or partial repair as ready while it is rebuilt in place.
            with Session(engine) as session, session.begin():
                candidate = session.get(SemanticIndexConfig, config_hash)
                if candidate is not None:
                    candidate.active = False
                    candidate.status = "building"
                    candidate.audit_json = {
                        **prior_candidate_audit,
                        "repair": {
                            "reason": "active_lance_sql_membership_divergence",
                            "started_at": datetime.now(UTC).isoformat(),
                        },
                    }
            candidate_was_active_ready = False
        while True:
            with Session(engine) as session:
                pending_statement = select(SearchWindow)
                if last_window_id is not None:
                    # Keyset pagination prevents every batch from rescanning the already embedded
                    # primary-key prefix. A resumed build may pay for that prefix once, then stays
                    # O(total windows) rather than O(total windows * batch count).
                    pending_statement = pending_statement.where(SearchWindow.id > last_window_id)
                pending = list(
                    session.scalars(
                        pending_statement.order_by(SearchWindow.id).limit(actual_batch_size)
                    )
                )
            if not pending:
                break
            vectors = active_embedder.embed_documents([row.text_display for row in pending])
            _validate_embedding_vectors(
                vectors,
                expected_count=len(pending),
                dimensions=spec.dimensions,
            )
            vector_rows = [
                _vector_row(window, vector) for window, vector in zip(pending, vectors, strict=True)
            ]
            (
                table.merge_insert("id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(vector_rows)
            )
            completed_at = datetime.now(UTC)
            pending_ids = [row.id for row in pending]
            last_window_id = pending[-1].id
            with Session(engine) as session, session.begin():
                for start in range(0, len(pending_ids), SEMANTIC_MEMBERSHIP_BATCH_SIZE):
                    statement = sqlite_insert(SemanticWindowEmbedding).values(
                        [
                            {
                                "config_id": config_hash,
                                "window_id": window_id,
                                "embedded_at": completed_at,
                            }
                            for window_id in pending_ids[
                                start : start + SEMANTIC_MEMBERSHIP_BATCH_SIZE
                            ]
                        ]
                    )
                    session.execute(
                        statement.on_conflict_do_update(
                            index_elements=[
                                SemanticWindowEmbedding.config_id,
                                SemanticWindowEmbedding.window_id,
                            ],
                            set_={"embedded_at": statement.excluded.embedded_at},
                        )
                    )
                embedded_count += len(pending)
                session.execute(
                    update(Job)
                    .where(Job.id == job_id)
                    .values(
                        stage="Embedding search windows",
                        progress_current=preexisting_embedded_count + embedded_count,
                        progress_total=total,
                        checkpoint_json={
                            "config_hash": config_hash,
                            "last_window_id": pending[-1].id,
                            "embedded_this_run": embedded_count,
                            "reused_from_checkpoint": preexisting_embedded_count,
                        },
                    )
                )

        total_windows = _assert_complete_semantic_checkpoint(
            table,
            engine,
            config_hash=config_hash,
            dimensions=spec.dimensions,
        )
        _assert_semantic_candidate_current(
            engine,
            config_hash=config_hash,
            job_id=job_id,
            candidate_was_active_ready=candidate_was_active_ready,
        )
        approximate_index_requested = total_windows >= 256
        if approximate_index_requested:
            if not _index_covers_table(table, column="vector", total_rows=total_windows):
                table.create_index(
                    "vector",
                    config=HnswSq(distance_type="cosine"),
                    replace=True,
                )
            for column in ("id", "entity_type", "conversation_id", "source_category"):
                if not _index_covers_table(table, column=column, total_rows=total_windows):
                    table.create_index(column, config=BTree(), replace=True)

        with Session(engine) as session, session.begin():
            session.execute(
                update(SemanticIndexConfig)
                .where(SemanticIndexConfig.id == config_hash)
                .values(window_count=total_windows)
            )
            session.execute(
                update(Job)
                .where(Job.id == job_id)
                .values(
                    stage="Evaluating semantic retrieval quality",
                    progress_current=total_windows,
                    progress_total=total_windows,
                )
            )
        evaluation = _semantic_ir_evaluation(
            layout,
            engine,
            table,
            active_embedder,
            config_hash=config_hash,
            vector_table=vector_table,
            corpus_revision=corpus_revision,
            total_windows=total_windows,
            approximate_index_requested=approximate_index_requested,
            reranker_factory=reranker_factory,
        )
        preserved_accepted_evaluation: dict[str, Any] | None = None
        if candidate_was_active_ready:
            prior_accepted_evaluation = prior_candidate_audit.get("accepted_semantic_ir_evaluation")
            accepted_evidence_comparison = _same_config_accepted_evidence_comparison(
                prior_accepted_evaluation,
                evaluation,
                config_hash=config_hash,
                corpus_revision=corpus_revision,
            )
            evaluation = {
                **evaluation,
                "same_config_accepted_evidence": accepted_evidence_comparison,
            }
            if accepted_evidence_comparison["passed"] is True:
                preserved_accepted_evaluation = cast(
                    dict[str, Any],
                    prior_accepted_evaluation,
                )
            else:
                evaluation["passed"] = False
                evaluation["production_acceptance_passed"] = False
                evaluation["decision"] = (
                    "rejected; the prior accepted evaluation and active index remain unchanged"
                )
        _assert_semantic_candidate_current(
            engine,
            config_hash=config_hash,
            job_id=job_id,
            candidate_was_active_ready=candidate_was_active_ready,
        )
        if not evaluation["passed"]:
            finished = datetime.now(UTC)
            failed_audit = {
                **(prior_candidate_audit if candidate_was_active_ready else audit),
                "semantic_ir_evaluation": evaluation,
            }
            if candidate_was_active_ready:
                prior_evaluation = prior_candidate_audit.get("semantic_ir_evaluation")
                if (
                    "accepted_semantic_ir_evaluation" not in failed_audit
                    and isinstance(prior_evaluation, dict)
                    and prior_evaluation.get("passed") is True
                ):
                    failed_audit["accepted_semantic_ir_evaluation"] = prior_evaluation
            known_metrics = evaluation["automatic_known_item"]["metrics"]
            held_out = evaluation["held_out_feedback"]
            held_out_metrics = held_out["metrics"]
            with _immediate_session(engine) as session:
                _assert_semantic_candidate_current_in_session(
                    session,
                    config_hash=config_hash,
                    job_id=job_id,
                    candidate_was_active_ready=candidate_was_active_ready,
                )
                session.execute(
                    update(SemanticIndexConfig)
                    .where(SemanticIndexConfig.id == config_hash)
                    .values(
                        active=candidate_was_active_ready,
                        status="ready" if candidate_was_active_ready else "rejected",
                        window_count=total_windows,
                        audit_json=failed_audit,
                        completed_at=finished,
                    )
                )
                session.execute(
                    update(Job)
                    .where(Job.id == job_id)
                    .values(
                        status="failed",
                        stage="Semantic index rejected by retrieval quality gate",
                        progress_current=total_windows,
                        progress_total=total_windows,
                        error_text=(
                            "Candidate did not meet semantic retrieval acceptance thresholds"
                        ),
                        finished_at=finished,
                        counters_json={
                            "config_hash": config_hash,
                            "model_name": spec.model_name,
                            "window_count": total_windows,
                            "embedded_this_run": embedded_count,
                            "resumed": resumed,
                            "acceptance_passed": False,
                            "known_item_recall_at_k": known_metrics["recall_at_k"],
                            "known_item_mrr": known_metrics["mrr"],
                            "known_item_ndcg_at_k": known_metrics["ndcg_at_k"],
                            "held_out_used_for_activation": held_out["used_for_activation"],
                            "held_out_query_count": held_out_metrics["query_count"],
                            "held_out_recall_at_k": held_out_metrics["recall_at_k"],
                            "held_out_mrr": held_out_metrics["mrr"],
                            "held_out_ndcg_at_k": held_out_metrics["ndcg_at_k"],
                        },
                    )
                )
            raise SemanticIndexAcceptanceError(
                job_id=job_id,
                config_hash=config_hash,
                evaluation=evaluation,
            )

        finished = datetime.now(UTC)
        accepted_audit = {
            **audit,
            "semantic_ir_evaluation": evaluation,
            "accepted_semantic_ir_evaluation": (preserved_accepted_evaluation or evaluation),
        }
        with _immediate_session(engine) as session:
            _assert_semantic_candidate_current_in_session(
                session,
                config_hash=config_hash,
                job_id=job_id,
                candidate_was_active_ready=candidate_was_active_ready,
            )
            session.execute(update(SemanticIndexConfig).values(active=False))
            session.execute(
                update(SearchWindow).values(
                    embedding_config_hash=config_hash,
                    embedding_updated_at=finished,
                )
            )
            session.execute(
                update(SemanticIndexConfig)
                .where(SemanticIndexConfig.id == config_hash)
                .values(
                    active=True,
                    status="ready",
                    window_count=total_windows,
                    audit_json=accepted_audit,
                    completed_at=finished,
                )
            )
            session.execute(
                update(Job)
                .where(Job.id == job_id)
                .values(
                    status="completed",
                    stage="Semantic index ready",
                    progress_current=total_windows,
                    progress_total=total_windows,
                    finished_at=finished,
                    counters_json={
                        "config_hash": config_hash,
                        "model_name": spec.model_name,
                        "window_count": total_windows,
                        "embedded_this_run": embedded_count,
                        "resumed": resumed,
                        "acceptance_passed": True,
                        "production_acceptance_passed": evaluation["production_acceptance_passed"],
                        "known_item_recall_at_k": evaluation["automatic_known_item"]["metrics"][
                            "recall_at_k"
                        ],
                        "known_item_mrr": evaluation["automatic_known_item"]["metrics"]["mrr"],
                        "known_item_ndcg_at_k": evaluation["automatic_known_item"]["metrics"][
                            "ndcg_at_k"
                        ],
                        "held_out_used_for_activation": evaluation["held_out_feedback"][
                            "used_for_activation"
                        ],
                        "held_out_query_count": evaluation["held_out_feedback"]["metrics"][
                            "query_count"
                        ],
                        "held_out_recall_at_k": evaluation["held_out_feedback"]["metrics"][
                            "recall_at_k"
                        ],
                        "held_out_mrr": evaluation["held_out_feedback"]["metrics"]["mrr"],
                        "held_out_ndcg_at_k": evaluation["held_out_feedback"]["metrics"][
                            "ndcg_at_k"
                        ],
                        "direct_conflict_calibration_status": evaluation[
                            "direct_conflict_development_calibration"
                        ]["status"],
                        "direct_conflict_calibration_judgment_count": evaluation[
                            "direct_conflict_development_calibration"
                        ].get("judgment_count", 0),
                        "direct_conflict_calibration_conversation_count": evaluation[
                            "direct_conflict_development_calibration"
                        ].get("conversation_count", 0),
                    },
                )
            )
        try:
            from meta_archive.services.conflict_aggregates import (
                invalidate_conflict_aggregates,
            )

            invalidated_conflict_runs = invalidate_conflict_aggregates(
                engine,
                reason=("Semantic index configuration changed; rebuild conflict aggregates"),
            )
            invalidation_result: dict[str, Any] = {
                "status": "completed",
                "invalidated_run_count": invalidated_conflict_runs,
                "completed_at": datetime.now(UTC).isoformat(),
            }
        except Exception as exc:
            # Activation is already durable and must not be rolled back by a downstream cache
            # invalidation failure. The status service independently detects stale aggregates;
            # retain an explicit audit warning so the failure is never silent.
            invalidation_result = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "failed_at": datetime.now(UTC).isoformat(),
            }
        with Session(engine) as session, session.begin():
            activated = session.get(SemanticIndexConfig, config_hash)
            if activated is not None:
                activated.audit_json = {
                    **activated.audit_json,
                    "conflict_aggregate_invalidation": invalidation_result,
                }
        return SemanticIndexSummary(
            job_id=job_id,
            config_hash=config_hash,
            vector_table=vector_table,
            window_count=total_windows,
            embedded_count=embedded_count,
            resumed=resumed,
            model_name=spec.model_name,
        )
    except SemanticIndexAcceptanceError:
        raise
    except BaseException as exc:
        finished = datetime.now(UTC)
        with Session(engine) as session, session.begin():
            session.execute(
                update(Job)
                .where(Job.id == job_id)
                .values(
                    status="failed",
                    stage="Semantic index failed",
                    error_text=str(exc),
                    finished_at=finished,
                )
            )
            candidate = session.get(SemanticIndexConfig, config_hash)
            if candidate is not None:
                if candidate_was_active_ready and isinstance(exc, SemanticTableIntegrityError):
                    candidate.status = "failed"
                    candidate.active = False
                    candidate.audit_json = {
                        **prior_candidate_audit,
                        "semantic_integrity_failure": {
                            "failed_at": finished.isoformat(),
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        },
                    }
                elif candidate_was_active_ready:
                    candidate.status = "ready"
                    candidate.active = True
                    candidate.audit_json = {
                        **prior_candidate_audit,
                        "semantic_ir_last_build_error": {
                            "failed_at": finished.isoformat(),
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        },
                    }
                else:
                    candidate.status = "failed"
                    candidate.active = False
        raise


def iter_window_message_ids(session: Session, window_ids: Iterable[str]) -> dict[str, list[str]]:
    ids = list(window_ids)
    result: dict[str, list[str]] = {window_id: [] for window_id in ids}
    if not ids:
        return result
    rows = session.execute(
        select(
            SearchWindowMessage.window_id,
            SearchWindowMessage.message_id,
        )
        .where(SearchWindowMessage.window_id.in_(ids))
        .order_by(SearchWindowMessage.window_id, SearchWindowMessage.ordinal)
    )
    for window_id, message_id in rows:
        result[window_id].append(message_id)
    return result
