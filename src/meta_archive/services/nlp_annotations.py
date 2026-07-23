from __future__ import annotations

import fcntl
import gc
import hashlib
import json
import math
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from sqlalchemy import (
    Engine,
    String,
    bindparam,
    column,
    func,
    or_,
    select,
    update,
    values,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from meta_archive.models import (
    ActivityRecord,
    AttachmentOccurrence,
    AttachmentTextExtraction,
    AudioTranscription,
    AudioTranscriptSegment,
    ConflictCalibration,
    ConflictLabel,
    Job,
    Message,
    NlpAnnotation,
    NlpHourlyRollup,
    SearchWindow,
)
from meta_archive.paths import ArchiveLayout
from meta_archive.services.conflict_review import (
    CONFLICT_EVALUATION_SPLITS,
    ConflictCalibrationIneligibleError,
    conflict_review_status,
    deterministic_conflict_split,
)
from meta_archive.services.conflict_scoring import (
    CONFLICT_LABELS,
    DEFAULT_CONFLICT_SPEC,
    ConflictScorerFactory,
    ModernBertNliConflictScorer,
)
from meta_archive.services.deletion_lock_lease import deletion_lock
from meta_archive.services.derived_publication_guard import assert_derived_publication_allowed
from meta_archive.services.identifiers import stable_id
from meta_archive.services.model_cache import (
    PinnedTransformersArtifact,
    pinned_transformers_artifact,
)

PIPELINE_VERSION = "derived-nlp-v1"
CONFLICT_PIPELINE_VERSION = "conflict-calibration-v1"
CONFLICT_MODEL_NAME = "tasksource/ModernBERT-base-nli"
CONFLICT_MODEL_REVISION = "de4ab7e77845098b7fab7f6ab9d370ddff27b19c"
MIN_CONFLICT_LABELS = 500
MIN_DIRECT_CONFLICT_PRECISION = 0.75
MIN_MACRO_F1 = 0.70
MAX_TEXT_CHARACTERS = 20_000
NLP_TOKENIZER_MAX_LENGTH = 512
MAX_ERROR_LENGTH = 4_000
NLP_CANDIDATE_CHUNK_SIZE = 512
MAX_NLP_CANDIDATE_CHUNK_SIZE = 900
MAX_NLP_BATCH_SIZE = 32
NLP_ROLLUP_PUBLISH_BATCH_SIZE = 512
NLP_INFERENCE_CONFIGURATION_SCHEMA = "nlp-inference-configuration-v1"
NLP_IMPLEMENTATION_SCHEMA = "transformers-sequence-classification-v1"
NLP_PROVENANCE_KEY = "inference_configuration"
DIRECT_PROVENANCE_BINDING = "direct_prepost_snapshot_verified"
LEGACY_PROVENANCE_BINDING = "legacy_posthoc_cache_verified_at_reuse"
INJECTED_PROVENANCE_BINDING = "injected_classifier_declared_configuration"
VALID_NLP_PROVENANCE_BINDINGS = frozenset(
    {
        DIRECT_PROVENANCE_BINDING,
        LEGACY_PROVENANCE_BINDING,
        INJECTED_PROVENANCE_BINDING,
    }
)

NlpTask = Literal["sentiment", "emotion", "toxicity"]
ProgressCallback = Callable[[str, int, int | None], None]


def nlp_annotations_lock_is_held(layout: ArchiveLayout) -> bool:
    """Probe the process lock without disturbing a running annotation job."""

    if layout.catalog.is_symlink() or not layout.catalog.is_dir():
        return False
    lock_path = layout.catalog / "nlp-annotations.lock"
    with lock_path.open("a+b") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    return False


@dataclass(frozen=True, slots=True)
class NlpModelSpec:
    task: NlpTask
    model_name: str
    model_revision: str
    activation: Literal["softmax", "sigmoid"]
    language_scope: str
    attribution: str
    warnings: tuple[str, ...]
    device: Literal["cpu"] = "cpu"
    quantization: Literal["none"] = "none"


MODEL_SPECS: dict[NlpTask, NlpModelSpec] = {
    "sentiment": NlpModelSpec(
        task="sentiment",
        model_name="cardiffnlp/twitter-roberta-base-sentiment-latest",
        model_revision="3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7",
        activation="softmax",
        language_scope="English social-media text",
        attribution="Cardiff NLP TweetEval-derived RoBERTa sentiment classifier",
        warnings=(
            "Sentiment is model-estimated and may fail on sarcasm, context, or repaired text.",
            "Do not use model output to rank people or relationships.",
        ),
    ),
    "emotion": NlpModelSpec(
        task="emotion",
        model_name="j-hartmann/emotion-english-distilroberta-base",
        model_revision="0e1cd914e3d46199ed785853e12b57304e04178b",
        activation="softmax",
        language_scope="English text",
        attribution="DistilRoBERTa English emotion classifier by Jochen Hartmann",
        warnings=(
            "Emotion labels describe text patterns, not a person's internal mental state.",
            "Do not use model output to rank people or relationships.",
        ),
    ),
    "toxicity": NlpModelSpec(
        task="toxicity",
        model_name="unitary/toxic-bert",
        model_revision="4d6c22e74ba2fdd26bc4f7238f50766b045a0d94",
        activation="sigmoid",
        language_scope="English moderation text",
        attribution="Unitary toxic-comment BERT classifier",
        warnings=(
            "Toxicity is a fallible moderation estimate with dialect and identity-term biases.",
            "Context, quotation, banter, and reclaimed language can produce false positives.",
            "Do not use model output to rank people or relationships.",
        ),
    ),
}


def _model_execution_configurations(tasks: Sequence[NlpTask]) -> dict[str, dict[str, str]]:
    return {
        task: {
            "device": MODEL_SPECS[task].device,
            "quantization": MODEL_SPECS[task].quantization,
        }
        for task in tasks
    }


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    probabilities: dict[str, float]

    @property
    def predicted_label(self) -> str:
        return max(self.probabilities, key=self.probabilities.__getitem__)

    @property
    def confidence(self) -> float:
        return self.probabilities[self.predicted_label]


class NlpClassifier(Protocol):
    def classify(self, spec: NlpModelSpec, texts: Sequence[str]) -> list[ClassificationResult]: ...


@dataclass(frozen=True, slots=True)
class NlpInferenceConfiguration:
    sha256: str
    payload: dict[str, Any]
    direct_binding_kind: str


def _canonical_json_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _distribution_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unavailable"


def _runtime_class_name(value: object) -> str:
    runtime_class = value.__class__
    return f"{runtime_class.__module__}.{runtime_class.__qualname__}"


def _model_precision(model: object) -> str:
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return "provider-declared"
    try:
        parameter = next(iter(parameters()))
    except (StopIteration, TypeError):
        return "provider-declared"
    return str(getattr(parameter, "dtype", "provider-declared"))


def _configuration_from_payload(
    payload: dict[str, Any], *, direct_binding_kind: str
) -> NlpInferenceConfiguration:
    if direct_binding_kind not in VALID_NLP_PROVENANCE_BINDINGS:
        raise RuntimeError("NLP inference configuration has an invalid provenance binding")
    return NlpInferenceConfiguration(
        sha256=_canonical_json_sha256(payload),
        payload=payload,
        direct_binding_kind=direct_binding_kind,
    )


def _transformers_configuration(
    spec: NlpModelSpec,
    *,
    artifact: PinnedTransformersArtifact,
    tokenizer: object,
    model: object,
) -> NlpInferenceConfiguration:
    implementation = {
        "schema": NLP_IMPLEMENTATION_SCHEMA,
        "pipeline_version": PIPELINE_VERSION,
        "provider": "transformers-auto-sequence-classification",
        "packages": {
            package: _distribution_version(package)
            for package in ("transformers", "torch", "tokenizers", "safetensors")
        },
        "tokenizer_class": _runtime_class_name(tokenizer),
        "model_class": _runtime_class_name(model),
        "device": spec.device,
        "precision": _model_precision(model),
        "quantization": spec.quantization,
        "evaluation_mode": True,
        "inference_mode": "torch.inference_mode",
        "activation": spec.activation,
        "input_character_guard": MAX_TEXT_CHARACTERS,
        "tokenizer_padding": True,
        "tokenizer_truncation": True,
        "tokenizer_max_length": NLP_TOKENIZER_MAX_LENGTH,
        "label_mapping": "model.config.id2label casefold with label_<index> fallback",
    }
    implementation_sha256 = _canonical_json_sha256(implementation)
    payload = {
        "schema": NLP_INFERENCE_CONFIGURATION_SCHEMA,
        "task": spec.task,
        "model_name": spec.model_name,
        "model_revision": spec.model_revision,
        "model_artifact": {
            "schema": artifact.schema,
            "sha256": artifact.fingerprint_sha256,
            "file_count": artifact.file_count,
            "total_bytes": artifact.total_bytes,
        },
        "implementation": implementation,
        "implementation_sha256": implementation_sha256,
    }
    return _configuration_from_payload(payload, direct_binding_kind=DIRECT_PROVENANCE_BINDING)


def _injected_classifier_configuration(
    classifier: NlpClassifier, spec: NlpModelSpec
) -> NlpInferenceConfiguration:
    provider = getattr(classifier, "inference_configuration", None)
    if callable(provider):
        supplied = provider(spec)
        if isinstance(supplied, NlpInferenceConfiguration):
            if supplied.sha256 != _canonical_json_sha256(supplied.payload):
                raise RuntimeError("Injected NLP inference configuration hash is invalid")
            if supplied.direct_binding_kind not in VALID_NLP_PROVENANCE_BINDINGS:
                raise RuntimeError(
                    "Injected NLP inference configuration has an invalid provenance binding"
                )
            return supplied
        if not isinstance(supplied, dict):
            raise RuntimeError("Injected NLP inference configuration must be a JSON object")
        payload = dict(supplied)
    else:
        implementation = {
            "schema": "injected-classifier-contract-v1",
            "pipeline_version": PIPELINE_VERSION,
            "provider": "injected-classifier-contract",
            "classifier_class": _runtime_class_name(classifier),
            "device": spec.device,
            "quantization": spec.quantization,
            "activation": spec.activation,
            "input_character_guard": MAX_TEXT_CHARACTERS,
            "tokenizer_max_length": NLP_TOKENIZER_MAX_LENGTH,
        }
        payload = {
            "schema": NLP_INFERENCE_CONFIGURATION_SCHEMA,
            "task": spec.task,
            "model_name": spec.model_name,
            "model_revision": spec.model_revision,
            "model_artifact": {
                "schema": "declared-model-revision-v1",
                "declared_revision": spec.model_revision,
                "byte_verified": False,
            },
            "implementation": implementation,
            "implementation_sha256": _canonical_json_sha256(implementation),
        }
    return _configuration_from_payload(
        payload,
        direct_binding_kind=INJECTED_PROVENANCE_BINDING,
    )


def _empty_corpus_configuration(spec: NlpModelSpec) -> NlpInferenceConfiguration:
    implementation = {
        "schema": "empty-corpus-no-inference-v1",
        "pipeline_version": PIPELINE_VERSION,
        "provider": "no-inference-empty-corpus",
    }
    payload = {
        "schema": NLP_INFERENCE_CONFIGURATION_SCHEMA,
        "task": spec.task,
        "model_name": spec.model_name,
        "model_revision": spec.model_revision,
        "model_artifact": {
            "schema": "unresolved-empty-corpus-v1",
            "declared_revision": spec.model_revision,
            "byte_verified": False,
        },
        "implementation": implementation,
        "implementation_sha256": _canonical_json_sha256(implementation),
    }
    return _configuration_from_payload(
        payload,
        direct_binding_kind=INJECTED_PROVENANCE_BINDING,
    )


class TransformersClassifier:
    """Lazy, revision-pinned CPU classifier cache using well-known Transformers models."""

    def __init__(self, model_cache: Path) -> None:
        self.model_cache = model_cache
        self._models: dict[tuple[str, str], tuple[Any, Any, NlpInferenceConfiguration]] = {}

    @staticmethod
    def _bootstrap_snapshot(spec: NlpModelSpec, location: Path) -> None:
        """Preserve first-run downloads, then discard unverified bootstrap objects."""

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        location.mkdir(parents=True, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(
            spec.model_name,
            revision=spec.model_revision,
            cache_dir=location,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            spec.model_name,
            revision=spec.model_revision,
            cache_dir=location,
        )
        del tokenizer, model
        gc.collect()

    def _load(self, spec: NlpModelSpec) -> tuple[Any, Any, NlpInferenceConfiguration]:
        key = (spec.model_name, spec.model_revision)
        if key not in self._models:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            location = self.model_cache / spec.task / spec.model_revision
            try:
                artifact_before = pinned_transformers_artifact(
                    location,
                    model_name=spec.model_name,
                    model_revision=spec.model_revision,
                )
            except RuntimeError as exc:
                if str(exc) != "The pinned local model snapshot is unavailable":
                    raise
                self._bootstrap_snapshot(spec, location)
                artifact_before = pinned_transformers_artifact(
                    location,
                    model_name=spec.model_name,
                    model_revision=spec.model_revision,
                )
            # Transformers accepts pathnames rather than our verified descriptors. Comparing the
            # complete snapshot before and after loading detects ordinary cache changes while
            # relying on the documented quiescent-private-cache trust assumption for the interval.
            tokenizer = AutoTokenizer.from_pretrained(
                artifact_before.snapshot,
                local_files_only=True,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                artifact_before.snapshot,
                local_files_only=True,
            )
            model.to(spec.device)
            model.eval()
            artifact_after = pinned_transformers_artifact(
                location,
                model_name=spec.model_name,
                model_revision=spec.model_revision,
            )
            if artifact_after != artifact_before:
                raise RuntimeError("The pinned local model snapshot changed during model loading")
            configuration = _transformers_configuration(
                spec,
                artifact=artifact_after,
                tokenizer=tokenizer,
                model=model,
            )
            self._models[key] = (tokenizer, model, configuration)
        return self._models[key]

    def prepare(self, spec: NlpModelSpec) -> None:
        """Load one pinned model before any current annotation rows are retired."""

        self._load(spec)

    def inference_configuration(self, spec: NlpModelSpec) -> NlpInferenceConfiguration:
        return self._load(spec)[2]

    def classify(self, spec: NlpModelSpec, texts: Sequence[str]) -> list[ClassificationResult]:
        import torch

        tokenizer, model, _configuration = self._load(spec)
        encoded = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=NLP_TOKENIZER_MAX_LENGTH,
            return_tensors="pt",
        )
        with torch.inference_mode():
            logits = model(**encoded).logits
            values = (
                torch.softmax(logits, dim=-1)
                if spec.activation == "softmax"
                else torch.sigmoid(logits)
            ).cpu()
        id_to_label = {
            int(index): str(label).casefold() for index, label in model.config.id2label.items()
        }
        results: list[ClassificationResult] = []
        for row in values.tolist():
            probabilities = {
                id_to_label.get(index, f"label_{index}"): float(value)
                for index, value in enumerate(row)
            }
            results.append(ClassificationResult(probabilities=probabilities))
        return results

    def release(self, spec: NlpModelSpec) -> None:
        """Release a completed task's model before loading the next CPU model."""

        self._models.pop((spec.model_name, spec.model_revision), None)


@dataclass(frozen=True, slots=True)
class _TextCandidate:
    target_type: str
    target_id: str
    text: str
    language: str | None
    attribution: dict[str, Any]


@dataclass(frozen=True, slots=True)
class AnnotationSummary:
    job_id: str
    pipeline_version: str
    target_count: int
    task_count: int
    annotated: int
    reused: int
    failed: int
    resumed: bool
    rollup_count: int
    model_specs: dict[str, dict[str, Any]]


def annotation_id(candidate: _TextCandidate, spec: NlpModelSpec, input_sha256: str) -> str:
    return stable_id(
        "nlp-annotation",
        PIPELINE_VERSION,
        candidate.target_type,
        candidate.target_id,
        spec.task,
        spec.model_name,
        spec.model_revision,
        input_sha256,
    )


def _occurrence_attribution_by_blob(
    engine: Engine, blob_sha256s: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """Choose one stable live occurrence for only the blobs in the current bounded page."""

    unique_blobs = tuple(dict.fromkeys(blob_sha256s))
    if not unique_blobs:
        return {}
    by_blob: dict[str, dict[str, Any]] = {}
    with Session(engine) as session:
        rows = session.execute(
            select(
                AttachmentOccurrence.blob_sha256,
                AttachmentOccurrence.id,
                AttachmentOccurrence.conversation_id,
                AttachmentOccurrence.message_id,
                AttachmentOccurrence.timestamp_ms,
            )
            .where(
                AttachmentOccurrence.blob_sha256.in_(unique_blobs),
                AttachmentOccurrence.deleted_at.is_(None),
            )
            .order_by(
                AttachmentOccurrence.blob_sha256,
                AttachmentOccurrence.timestamp_ms.is_(None),
                AttachmentOccurrence.timestamp_ms,
                AttachmentOccurrence.id,
            )
        )
        for blob_sha256, occurrence_id, conversation_id, message_id, timestamp_ms in rows:
            if blob_sha256 is None or blob_sha256 in by_blob:
                continue
            by_blob[blob_sha256] = {
                "attachment_occurrence_id": occurrence_id,
                "conversation_id": conversation_id,
                "message_id": message_id,
                "timestamp_ms": timestamp_ms,
            }
    return by_blob


def _text_candidate_count(engine: Engine) -> int:
    message_has_text = or_(
        Message.content_display != "",
        Message.transcription_display != "",
    )
    activity_has_text = or_(
        ActivityRecord.title_display != "",
        ActivityRecord.content_display != "",
    )
    with Session(engine) as session:
        message_count = session.scalar(
            select(func.count(Message.pk)).where(Message.deleted_at.is_(None), message_has_text)
        )
        activity_count = session.scalar(
            select(func.count(ActivityRecord.pk)).where(
                ActivityRecord.deleted_at.is_(None), activity_has_text
            )
        )
        audio_count = session.scalar(
            select(func.count(AudioTranscriptSegment.id))
            .join(
                AudioTranscription,
                AudioTranscription.id == AudioTranscriptSegment.transcription_id,
            )
            .where(
                AudioTranscription.status == "completed",
                AudioTranscription.index_eligible.is_(True),
                AudioTranscriptSegment.text_display != "",
            )
        )
        attachment_count = session.scalar(
            select(func.count(AttachmentTextExtraction.id)).where(
                AttachmentTextExtraction.status == "completed",
                AttachmentTextExtraction.index_eligible.is_(True),
                AttachmentTextExtraction.text_display.is_not(None),
                AttachmentTextExtraction.text_display != "",
            )
        )
    return sum(
        int(value or 0) for value in (message_count, activity_count, audio_count, attachment_count)
    )


def _iter_text_candidate_chunks(
    engine: Engine, *, chunk_size: int = NLP_CANDIDATE_CHUNK_SIZE
) -> Iterator[list[_TextCandidate]]:
    """Yield deterministic keyset pages without retaining ORM rows or read transactions."""

    if chunk_size < 1:
        raise ValueError("NLP candidate chunk size must be positive")

    last_id: str | None = None
    while True:
        message_statement = (
            select(
                Message.id,
                Message.content_display,
                Message.transcription_display,
                Message.conversation_id,
                Message.sender_is_owner,
                Message.timestamp_ms,
            )
            .where(
                Message.deleted_at.is_(None),
                or_(Message.content_display != "", Message.transcription_display != ""),
            )
            .order_by(Message.id)
            .limit(chunk_size)
        )
        if last_id is not None:
            message_statement = message_statement.where(Message.id > last_id)
        with Session(engine) as session:
            rows = list(session.execute(message_statement))
        if not rows:
            break
        last_id = str(rows[-1].id)
        yield [
            _TextCandidate(
                target_type="message",
                target_id=str(row.id),
                text="\n".join(
                    value for value in (row.content_display, row.transcription_display) if value
                ),
                language=None,
                attribution={
                    "conversation_id": row.conversation_id,
                    "sender_direction": "owner" if row.sender_is_owner else "other",
                    "direction": "owner" if row.sender_is_owner else "other",
                    "timestamp_ms": row.timestamp_ms,
                },
            )
            for row in rows
        ]

    last_id = None
    while True:
        activity_statement = (
            select(
                ActivityRecord.id,
                ActivityRecord.title_display,
                ActivityRecord.content_display,
                ActivityRecord.category,
                ActivityRecord.timestamp_ms,
            )
            .where(
                ActivityRecord.deleted_at.is_(None),
                or_(ActivityRecord.title_display != "", ActivityRecord.content_display != ""),
            )
            .order_by(ActivityRecord.id)
            .limit(chunk_size)
        )
        if last_id is not None:
            activity_statement = activity_statement.where(ActivityRecord.id > last_id)
        with Session(engine) as session:
            rows = list(session.execute(activity_statement))
        if not rows:
            break
        last_id = str(rows[-1].id)
        yield [
            _TextCandidate(
                target_type="activity",
                target_id=str(row.id),
                text="\n".join(
                    value for value in (row.title_display, row.content_display) if value
                ),
                language=None,
                attribution={
                    "category": row.category,
                    "direction": "activity",
                    "timestamp_ms": row.timestamp_ms,
                },
            )
            for row in rows
        ]

    last_id = None
    while True:
        audio_statement = (
            select(
                AudioTranscriptSegment.id,
                AudioTranscriptSegment.transcription_id,
                AudioTranscriptSegment.start_ms,
                AudioTranscriptSegment.end_ms,
                AudioTranscriptSegment.text_display,
                AudioTranscription.blob_sha256,
                AudioTranscription.language,
            )
            .join(
                AudioTranscription,
                AudioTranscription.id == AudioTranscriptSegment.transcription_id,
            )
            .where(
                AudioTranscription.status == "completed",
                AudioTranscription.index_eligible.is_(True),
                AudioTranscriptSegment.text_display != "",
            )
            .order_by(AudioTranscriptSegment.id)
            .limit(chunk_size)
        )
        if last_id is not None:
            audio_statement = audio_statement.where(AudioTranscriptSegment.id > last_id)
        with Session(engine) as session:
            rows = list(session.execute(audio_statement))
        if not rows:
            break
        last_id = str(rows[-1].id)
        occurrences = _occurrence_attribution_by_blob(
            engine, [str(row.blob_sha256) for row in rows]
        )
        yield [
            _TextCandidate(
                target_type="audio_segment",
                target_id=str(row.id),
                text=str(row.text_display),
                language=cast(str | None, row.language),
                attribution={
                    "transcription_id": row.transcription_id,
                    "start_ms": row.start_ms,
                    "end_ms": row.end_ms,
                    "direction": "audio",
                    **occurrences.get(str(row.blob_sha256), {}),
                },
            )
            for row in rows
        ]

    last_id = None
    while True:
        attachment_statement = (
            select(
                AttachmentTextExtraction.id,
                AttachmentTextExtraction.blob_sha256,
                AttachmentTextExtraction.text_display,
            )
            .where(
                AttachmentTextExtraction.status == "completed",
                AttachmentTextExtraction.index_eligible.is_(True),
                AttachmentTextExtraction.text_display.is_not(None),
                AttachmentTextExtraction.text_display != "",
            )
            .order_by(AttachmentTextExtraction.id)
            .limit(chunk_size)
        )
        if last_id is not None:
            attachment_statement = attachment_statement.where(AttachmentTextExtraction.id > last_id)
        with Session(engine) as session:
            rows = list(session.execute(attachment_statement))
        if not rows:
            break
        last_id = str(rows[-1].id)
        occurrences = _occurrence_attribution_by_blob(
            engine, [str(row.blob_sha256) for row in rows]
        )
        yield [
            _TextCandidate(
                target_type="attachment_text",
                target_id=str(row.id),
                text=cast(str, row.text_display),
                language=None,
                attribution={
                    "blob_sha256": row.blob_sha256,
                    "direction": "attachment",
                    **occurrences.get(str(row.blob_sha256), {}),
                },
            )
            for row in rows
        ]


def _prepare_job(
    engine: Engine,
    *,
    total: int,
    tasks: Sequence[NlpTask],
    batch_size: int,
    candidate_chunk_size: int,
    requested_job_id: str | None = None,
) -> tuple[str, bool]:
    revisions = {task: MODEL_SPECS[task].model_revision for task in tasks}
    execution_configurations = _model_execution_configurations(tasks)
    checkpoint = {
        "pipeline_version": PIPELINE_VERSION,
        "model_revisions": revisions,
        "execution_configurations": execution_configurations,
        "batch_size": batch_size,
        "candidate_chunk_size": candidate_chunk_size,
    }
    now = datetime.now(UTC)
    with Session(engine) as session, session.begin():
        if requested_job_id is not None:
            requested = session.get(Job, requested_job_id)
            if requested is None or requested.kind != "nlp_annotations":
                raise RuntimeError("Queued NLP annotation job was not found")
            if requested.status not in {"queued", "running"}:
                raise RuntimeError("NLP annotation job is no longer queued or running")
            resumed = requested.status == "running"
            existing_registry = requested.checkpoint_json.get("inference_configurations")
            existing_task_configurations = requested.checkpoint_json.get(
                "task_inference_configuration_sha256"
            )
            if isinstance(existing_registry, dict):
                checkpoint["inference_configurations"] = existing_registry
            if isinstance(existing_task_configurations, dict):
                checkpoint["task_inference_configuration_sha256"] = existing_task_configurations
            requested.status = "running"
            requested.stage = "Computing local NLP annotations"
            requested.progress_current = 0
            requested.progress_total = total
            requested.checkpoint_json = checkpoint
            requested.counters_json = {}
            requested.error_text = None
            requested.started_at = now
            requested.finished_at = None
            return requested.id, resumed
        previous = session.scalar(
            select(Job)
            .where(Job.kind == "nlp_annotations", Job.status.in_(("running", "failed")))
            .order_by(Job.created_at.desc())
        )
        if (
            previous is not None
            and previous.checkpoint_json.get("pipeline_version") == PIPELINE_VERSION
            and previous.checkpoint_json.get("model_revisions") == revisions
            and previous.checkpoint_json.get("execution_configurations")
            in (None, execution_configurations)
        ):
            existing_registry = previous.checkpoint_json.get("inference_configurations")
            existing_task_configurations = previous.checkpoint_json.get(
                "task_inference_configuration_sha256"
            )
            if isinstance(existing_registry, dict):
                checkpoint["inference_configurations"] = existing_registry
            if isinstance(existing_task_configurations, dict):
                checkpoint["task_inference_configuration_sha256"] = existing_task_configurations
            previous.status = "running"
            previous.stage = "Resuming local NLP annotations"
            previous.progress_total = total
            previous.error_text = None
            previous.finished_at = None
            previous.started_at = now
            previous.checkpoint_json = checkpoint
            return previous.id, True
        job_id = str(uuid.uuid4())
        session.add(
            Job(
                id=job_id,
                kind="nlp_annotations",
                status="running",
                stage="Computing local NLP annotations",
                progress_current=0,
                progress_total=total,
                checkpoint_json=checkpoint,
                counters_json={},
                started_at=now,
            )
        )
        return job_id, False


def _register_job_inference_configuration(
    engine: Engine,
    job_id: str,
    *,
    task: NlpTask,
    configuration: NlpInferenceConfiguration,
) -> None:
    """Publish an append-only full configuration registry before retiring active rows."""

    with Session(engine) as session, session.begin():
        job = session.get(Job, job_id)
        if job is None:
            raise RuntimeError("NLP annotation job was not found")
        checkpoint = dict(job.checkpoint_json or {})
        raw_registry = checkpoint.get("inference_configurations")
        registry = dict(raw_registry) if isinstance(raw_registry, dict) else {}
        existing = registry.get(configuration.sha256)
        if existing is not None and existing != configuration.payload:
            raise RuntimeError("NLP inference configuration hash collision")
        registry[configuration.sha256] = configuration.payload
        raw_tasks = checkpoint.get("task_inference_configuration_sha256")
        task_configurations = dict(raw_tasks) if isinstance(raw_tasks, dict) else {}
        task_configurations[task] = configuration.sha256
        checkpoint["inference_configurations"] = registry
        checkpoint["task_inference_configuration_sha256"] = task_configurations
        job.checkpoint_json = checkpoint


def _require_active_configuration_compatible(
    engine: Engine,
    *,
    task: NlpTask,
    configuration_sha256: str,
) -> None:
    """Fail before deactivation if published rows claim a different verified configuration."""

    provenance_path = f"$.{NLP_PROVENANCE_KEY}"
    provenance_type = func.json_type(NlpAnnotation.provenance_json, provenance_path)
    sha_path = f"{provenance_path}.sha256"
    sha_type = func.json_type(NlpAnnotation.provenance_json, sha_path)
    sha_value = func.json_extract(NlpAnnotation.provenance_json, sha_path)
    binding_path = f"{provenance_path}.binding_kind"
    binding_type = func.json_type(NlpAnnotation.provenance_json, binding_path)
    binding_value = func.json_extract(NlpAnnotation.provenance_json, binding_path)
    with Session(engine) as session:
        incompatible_count = int(
            session.scalar(
                select(func.count(NlpAnnotation.id)).where(
                    NlpAnnotation.task == task,
                    NlpAnnotation.active.is_(True),
                    NlpAnnotation.status == "completed",
                    # A truly absent key is the only legacy-adoption case. Once present, every
                    # component must already be a valid, matching producer commitment.
                    provenance_type.is_not(None),
                    or_(
                        provenance_type != "object",
                        sha_type.is_(None),
                        sha_type != "text",
                        sha_value != configuration_sha256,
                        binding_type.is_(None),
                        binding_type != "text",
                        binding_value.not_in(tuple(sorted(VALID_NLP_PROVENANCE_BINDINGS))),
                    ),
                )
            )
            or 0
        )
    if incompatible_count:
        raise RuntimeError(
            "Active NLP annotations contain malformed or incompatible inference provenance; "
            "an explicit versioned recomputation is required"
        )


def _validate_probabilities(result: ClassificationResult) -> None:
    if not result.probabilities or any(
        not math.isfinite(value) or value < 0 or value > 1
        for value in result.probabilities.values()
    ):
        raise RuntimeError("NLP classifier returned an invalid probability vector")


def _increment_binding_count(
    counts: dict[NlpTask, dict[str, int]],
    *,
    task: NlpTask,
    binding_kind: str,
    amount: int,
) -> None:
    if amount <= 0:
        return
    task_counts = counts.setdefault(task, {})
    task_counts[binding_kind] = task_counts.get(binding_kind, 0) + amount


@dataclass(frozen=True, slots=True)
class _AnnotationWork:
    candidate: _TextCandidate
    record_id: str
    input_sha256: str
    truncated: bool


@dataclass(frozen=True, slots=True)
class _CachedInference:
    annotation_id: str
    result: ClassificationResult
    binding_kind: str


def _reuse_provenance_for_configuration(
    provenance: dict[str, Any],
    *,
    configuration_sha256: str,
) -> dict[str, Any] | None:
    if NLP_PROVENANCE_KEY not in provenance:
        return {
            **provenance,
            NLP_PROVENANCE_KEY: {
                "sha256": configuration_sha256,
                "binding_kind": LEGACY_PROVENANCE_BINDING,
            },
        }
    existing = provenance[NLP_PROVENANCE_KEY]
    if not isinstance(existing, dict):
        return None
    existing_hash = existing.get("sha256")
    existing_binding = existing.get("binding_kind")
    if (
        not isinstance(existing_hash, str)
        or not existing_hash
        or existing_hash != configuration_sha256
        or not isinstance(existing_binding, str)
        or existing_binding not in VALID_NLP_PROVENANCE_BINDINGS
    ):
        return None
    return {
        **provenance,
        NLP_PROVENANCE_KEY: {
            "sha256": configuration_sha256,
            "binding_kind": existing_binding,
        },
    }


def _annotation_work(
    candidates: Sequence[_TextCandidate], spec: NlpModelSpec
) -> list[_AnnotationWork]:
    work: list[_AnnotationWork] = []
    for candidate in candidates:
        input_sha256 = hashlib.sha256(candidate.text.encode("utf-8")).hexdigest()
        work.append(
            _AnnotationWork(
                candidate=candidate,
                record_id=annotation_id(candidate, spec, input_sha256),
                input_sha256=input_sha256,
                truncated=len(candidate.text) > MAX_TEXT_CHARACTERS,
            )
        )
    return work


def _completed_annotation_provenance(
    engine: Engine,
    record_ids: Sequence[str],
    *,
    configuration_sha256: str,
) -> dict[str, dict[str, Any]]:
    if not record_ids:
        return {}
    with Session(engine) as session:
        reusable: dict[str, dict[str, Any]] = {}
        for record_id, provenance in session.execute(
            select(NlpAnnotation.id, NlpAnnotation.provenance_json).where(
                NlpAnnotation.id.in_(record_ids),
                NlpAnnotation.status == "completed",
            )
        ):
            if not isinstance(provenance, dict):
                continue
            updated = _reuse_provenance_for_configuration(
                provenance,
                configuration_sha256=configuration_sha256,
            )
            if updated is not None:
                reusable[str(record_id)] = updated
        return reusable


def _completed_inferences_by_input(
    engine: Engine,
    work: Sequence[_AnnotationWork],
    *,
    spec: NlpModelSpec,
    configuration_sha256: str,
) -> dict[str, _CachedInference]:
    """Reuse exact model inputs without conflating their target-specific attribution rows."""

    input_hashes = tuple(dict.fromkeys(item.input_sha256 for item in work))
    cached: dict[str, _CachedInference] = {}
    for start in range(0, len(input_hashes), 400):
        chunk = input_hashes[start : start + 400]
        with Session(engine) as session:
            # A ubiquitous exact input (for example one short reaction) can have thousands of
            # target-specific annotation rows. Probe one representative per requested hash with
            # an indexed LIMIT 1 scalar subquery, rather than grouping/scanning or transferring
            # every earlier duplicate and drifting toward quadratic work.
            requested_inputs = (
                values(column("input_sha256", String), name="requested_input_values")
                .data([(input_sha256,) for input_sha256 in chunk])
                .cte("requested_inputs")
            )
            representative_id = (
                select(NlpAnnotation.id)
                .where(
                    NlpAnnotation.task == spec.task,
                    NlpAnnotation.pipeline_version == PIPELINE_VERSION,
                    NlpAnnotation.model_name == spec.model_name,
                    NlpAnnotation.model_revision == spec.model_revision,
                    NlpAnnotation.input_sha256 == requested_inputs.c.input_sha256,
                    NlpAnnotation.status == "completed",
                    func.json_extract(
                        NlpAnnotation.provenance_json,
                        f"$.{NLP_PROVENANCE_KEY}.sha256",
                    )
                    == configuration_sha256,
                )
                .limit(1)
                .correlate(requested_inputs)
                .scalar_subquery()
            )
            representatives = select(
                requested_inputs.c.input_sha256,
                representative_id.label("annotation_id"),
            ).subquery()
            rows = session.execute(
                select(
                    NlpAnnotation.id,
                    NlpAnnotation.input_sha256,
                    NlpAnnotation.probabilities_json,
                    NlpAnnotation.provenance_json,
                )
                .join(
                    representatives,
                    representatives.c.annotation_id == NlpAnnotation.id,
                )
                .order_by(NlpAnnotation.input_sha256)
            )
            for annotation_id, input_sha256, payload, provenance in rows:
                if input_sha256 in cached or not isinstance(payload, dict):
                    continue
                labels = payload.get("labels")
                if payload.get("activation") != spec.activation or not isinstance(labels, dict):
                    continue
                probabilities = {
                    str(label): float(value)
                    for label, value in labels.items()
                    if isinstance(label, str)
                    and isinstance(value, (int, float))
                    and not isinstance(value, bool)
                }
                result = ClassificationResult(probabilities=probabilities)
                try:
                    _validate_probabilities(result)
                except RuntimeError:
                    continue
                inference_provenance = (
                    provenance.get(NLP_PROVENANCE_KEY) if isinstance(provenance, dict) else None
                )
                binding_kind = (
                    inference_provenance.get("binding_kind")
                    if isinstance(inference_provenance, dict)
                    else None
                )
                if not isinstance(binding_kind, str) or not binding_kind:
                    continue
                cached[str(input_sha256)] = _CachedInference(
                    annotation_id=str(annotation_id),
                    result=result,
                    binding_kind=binding_kind,
                )
    return cached


def _reactivate_annotations(
    engine: Engine,
    work: Sequence[_AnnotationWork],
    *,
    spec: NlpModelSpec,
    job_id: str,
    provenance_by_id: dict[str, dict[str, Any]],
) -> None:
    if not work:
        return
    statement = (
        update(NlpAnnotation)
        .where(NlpAnnotation.id == bindparam("_record_id"))
        .values(
            active=True,
            attribution_json=bindparam("_attribution_json"),
            provenance_json=bindparam("_provenance_json"),
            job_id=bindparam("_job_id"),
        )
    )
    parameters = [
        {
            "_record_id": item.record_id,
            "_attribution_json": {
                **item.candidate.attribution,
                "model_attribution": spec.attribution,
            },
            "_provenance_json": {
                **provenance_by_id[item.record_id],
                "input_truncated_by_character_guard": bool(
                    provenance_by_id[item.record_id].get("input_truncated", False)
                ),
                "tokenizer_truncation_enabled": True,
                "tokenizer_max_length": NLP_TOKENIZER_MAX_LENGTH,
                "execution": {
                    "device": spec.device,
                    "quantization": spec.quantization,
                },
            },
            "_job_id": job_id,
        }
        for item in work
    ]
    with Session(engine) as session, session.begin():
        session.connection().execute(statement, parameters)


def _upsert_annotations(engine: Engine, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    statement = sqlite_insert(NlpAnnotation)
    excluded = statement.excluded
    statement = statement.on_conflict_do_update(
        index_elements=["id"],
        set_={
            "status": excluded.status,
            "active": excluded.active,
            "language": excluded.language,
            "predicted_label": excluded.predicted_label,
            "confidence": excluded.confidence,
            "probabilities_json": excluded.probabilities_json,
            "attribution_json": excluded.attribution_json,
            "warnings_json": excluded.warnings_json,
            "provenance_json": excluded.provenance_json,
            "error_text": excluded.error_text,
            "job_id": excluded.job_id,
        },
    )
    with Session(engine) as session, session.begin():
        session.connection().execute(statement, list(rows))


def _annotation_row(
    item: _AnnotationWork,
    spec: NlpModelSpec,
    *,
    job_id: str,
    configuration: NlpInferenceConfiguration,
    binding_kind: str,
    result: ClassificationResult | None = None,
    error: BaseException | None = None,
    inference_reuse: dict[str, Any] | None = None,
) -> dict[str, Any]:
    completed = result is not None
    return {
        "id": item.record_id,
        "target_type": item.candidate.target_type,
        "target_id": item.candidate.target_id,
        "task": spec.task,
        "pipeline_version": PIPELINE_VERSION,
        "model_name": spec.model_name,
        "model_revision": spec.model_revision,
        "input_sha256": item.input_sha256,
        "status": "completed" if completed else "failed",
        "active": completed,
        "language": item.candidate.language,
        "predicted_label": result.predicted_label if result is not None else None,
        "confidence": result.confidence if result is not None else None,
        "probabilities_json": (
            {"activation": spec.activation, "labels": result.probabilities}
            if result is not None
            else {}
        ),
        "attribution_json": (
            {
                **item.candidate.attribution,
                "model_attribution": spec.attribution,
            }
            if completed
            else item.candidate.attribution
        ),
        "warnings_json": {"items": list(spec.warnings)},
        "provenance_json": {
            "generated_locally": True,
            "language_scope": spec.language_scope,
            # This compatibility field describes the explicit character guard. Tokenizer-level
            # clipping cannot be known without tokenizing each stored row again, so its configured
            # bound is recorded separately and never implied to be false.
            "input_truncated": item.truncated,
            "input_truncated_by_character_guard": item.truncated,
            "input_character_count": len(item.candidate.text),
            "tokenizer_truncation_enabled": True,
            "tokenizer_max_length": NLP_TOKENIZER_MAX_LENGTH,
            "execution": {
                "device": spec.device,
                "quantization": spec.quantization,
            },
            NLP_PROVENANCE_KEY: {
                "sha256": configuration.sha256,
                "binding_kind": binding_kind,
            },
            **({"inference_reuse": inference_reuse} if inference_reuse else {}),
        },
        "error_text": str(error)[:MAX_ERROR_LENGTH] if error is not None else None,
        "job_id": job_id,
    }


def build_nlp_annotations(
    layout: ArchiveLayout,
    engine: Engine,
    *,
    classifier: NlpClassifier | None = None,
    tasks: Sequence[NlpTask] = ("sentiment", "emotion", "toxicity"),
    batch_size: int = 16,
    progress: ProgressCallback | None = None,
    requested_job_id: str | None = None,
    candidate_chunk_size: int = NLP_CANDIDATE_CHUNK_SIZE,
) -> AnnotationSummary:
    selected_tasks = tuple(dict.fromkeys(tasks))
    if not selected_tasks or any(task not in MODEL_SPECS for task in selected_tasks):
        raise ValueError("Choose at least one supported NLP task")
    if not 1 <= batch_size <= MAX_NLP_BATCH_SIZE:
        raise ValueError(f"NLP batch size must be between 1 and {MAX_NLP_BATCH_SIZE}")
    if not batch_size <= candidate_chunk_size <= MAX_NLP_CANDIDATE_CHUNK_SIZE:
        raise ValueError(
            "NLP candidate chunk size must be at least the classifier batch size and no more "
            f"than {MAX_NLP_CANDIDATE_CHUNK_SIZE}"
        )
    active_classifier = classifier or TransformersClassifier(layout.models / "nlp")
    candidate_count = _text_candidate_count(engine)
    total = candidate_count * len(selected_tasks)
    model_revisions = {task: MODEL_SPECS[task].model_revision for task in selected_tasks}
    execution_configurations = _model_execution_configurations(selected_tasks)
    inference_configurations: dict[NlpTask, NlpInferenceConfiguration] = {}
    provenance_binding_counts: dict[NlpTask, dict[str, int]] = {}
    lock_path = layout.catalog / "nlp-annotations.lock"
    with ExitStack() as archive_locks, lock_path.open("a+b") as lock_handle:
        archive_locks.enter_context(deletion_lock(layout))
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another NLP annotation job is already running") from exc
        assert_derived_publication_allowed(engine, publisher="NLP annotation publication")
        job_id, resumed = _prepare_job(
            engine,
            total=total,
            tasks=selected_tasks,
            batch_size=batch_size,
            candidate_chunk_size=candidate_chunk_size,
            requested_job_id=requested_job_id,
        )
        annotated = reused = failed = processed = 0
        try:
            for task in selected_tasks:
                spec = MODEL_SPECS[task]
                provenance_binding_counts[task] = {}
                if candidate_count and isinstance(active_classifier, TransformersClassifier):
                    # Model download/configuration errors are job-level failures, not one error
                    # per classifier batch. Preparing before deactivation also leaves the last
                    # published annotations active when an offline/missing model cannot load.
                    active_classifier.prepare(spec)
                    inference_configuration = active_classifier.inference_configuration(spec)
                elif isinstance(active_classifier, TransformersClassifier):
                    inference_configuration = _empty_corpus_configuration(spec)
                else:
                    inference_configuration = _injected_classifier_configuration(
                        active_classifier,
                        spec,
                    )
                inference_configurations[task] = inference_configuration
                _register_job_inference_configuration(
                    engine,
                    job_id,
                    task=task,
                    configuration=inference_configuration,
                )
                _require_active_configuration_compatible(
                    engine,
                    task=task,
                    configuration_sha256=inference_configuration.sha256,
                )
                # Active rows always describe the current, live candidate set. This also makes
                # deletions safe: annotations for targets no longer present are retired.
                with Session(engine) as session, session.begin():
                    session.execute(
                        update(NlpAnnotation)
                        .where(NlpAnnotation.task == task, NlpAnnotation.active.is_(True))
                        .values(active=False)
                    )
                for candidates in _iter_text_candidate_chunks(
                    engine, chunk_size=candidate_chunk_size
                ):
                    work = _annotation_work(candidates, spec)
                    completed_provenance = _completed_annotation_provenance(
                        engine,
                        [item.record_id for item in work],
                        configuration_sha256=inference_configuration.sha256,
                    )
                    reusable = [item for item in work if item.record_id in completed_provenance]
                    pending = [item for item in work if item.record_id not in completed_provenance]
                    _reactivate_annotations(
                        engine,
                        reusable,
                        spec=spec,
                        job_id=job_id,
                        provenance_by_id=completed_provenance,
                    )
                    for item in reusable:
                        inference_provenance = completed_provenance[item.record_id][
                            NLP_PROVENANCE_KEY
                        ]
                        _increment_binding_count(
                            provenance_binding_counts,
                            task=task,
                            binding_kind=str(inference_provenance["binding_kind"]),
                            amount=1,
                        )
                    reused += len(reusable)
                    processed += len(reusable)
                    rows_to_upsert: list[dict[str, Any]] = []
                    pending_by_input: dict[str, list[_AnnotationWork]] = {}
                    for item in pending:
                        pending_by_input.setdefault(item.input_sha256, []).append(item)
                    cached_inferences = _completed_inferences_by_input(
                        engine,
                        pending,
                        spec=spec,
                        configuration_sha256=inference_configuration.sha256,
                    )
                    uncached_groups: list[list[_AnnotationWork]] = []
                    for input_sha256, group in pending_by_input.items():
                        cached = cached_inferences.get(input_sha256)
                        if cached is None:
                            uncached_groups.append(group)
                            continue
                        rows_to_upsert.extend(
                            _annotation_row(
                                item,
                                spec,
                                job_id=job_id,
                                configuration=inference_configuration,
                                binding_kind=cached.binding_kind,
                                result=cached.result,
                                inference_reuse={
                                    "source": "completed_annotation",
                                    "source_annotation_id": cached.annotation_id,
                                    "input_sha256": input_sha256,
                                    "inference_configuration_sha256": (
                                        inference_configuration.sha256
                                    ),
                                },
                            )
                            for item in group
                        )
                        annotated += len(group)
                        processed += len(group)
                        _increment_binding_count(
                            provenance_binding_counts,
                            task=task,
                            binding_kind=cached.binding_kind,
                            amount=len(group),
                        )
                    # Length bucketing avoids padding every short social-media message to the
                    # longest unrelated item in its target-ID page.
                    uncached_groups.sort(
                        key=lambda group: (
                            len(group[0].candidate.text[:MAX_TEXT_CHARACTERS]),
                            group[0].record_id,
                        )
                    )
                    for batch_start in range(0, len(uncached_groups), batch_size):
                        group_batch = uncached_groups[batch_start : batch_start + batch_size]
                        classifier_batch = [group[0] for group in group_batch]
                        batch_failed = False
                        try:
                            results = active_classifier.classify(
                                spec,
                                [
                                    item.candidate.text[:MAX_TEXT_CHARACTERS]
                                    for item in classifier_batch
                                ],
                            )
                            if len(results) != len(classifier_batch):
                                raise RuntimeError("NLP classifier returned the wrong result count")
                            rows: list[dict[str, Any]] = []
                            for group, result in zip(group_batch, results, strict=True):
                                _validate_probabilities(result)
                                rows.extend(
                                    _annotation_row(
                                        item,
                                        spec,
                                        job_id=job_id,
                                        configuration=inference_configuration,
                                        binding_kind=(inference_configuration.direct_binding_kind),
                                        result=result,
                                        inference_reuse=(
                                            {
                                                "source": "same_classifier_batch",
                                                "source_annotation_id": group[0].record_id,
                                                "input_sha256": item.input_sha256,
                                                "inference_configuration_sha256": (
                                                    inference_configuration.sha256
                                                ),
                                            }
                                            if item is not group[0]
                                            else None
                                        ),
                                    )
                                    for item in group
                                )
                        except Exception as exc:
                            rows = [
                                _annotation_row(
                                    item,
                                    spec,
                                    job_id=job_id,
                                    configuration=inference_configuration,
                                    binding_kind=inference_configuration.direct_binding_kind,
                                    error=exc,
                                )
                                for group in group_batch
                                for item in group
                            ]
                            batch_failed = True
                        rows_to_upsert.extend(rows)
                        batch_target_count = sum(len(group) for group in group_batch)
                        if batch_failed:
                            failed += batch_target_count
                        else:
                            annotated += batch_target_count
                            _increment_binding_count(
                                provenance_binding_counts,
                                task=task,
                                binding_kind=inference_configuration.direct_binding_kind,
                                amount=batch_target_count,
                            )
                        processed += batch_target_count
                    _upsert_annotations(engine, rows_to_upsert)
                    _update_job(
                        engine,
                        job_id,
                        processed=processed,
                        total=total,
                        task=task,
                        last_target_id=candidates[-1].target_id if candidates else None,
                        annotated=annotated,
                        reused=reused,
                        failed=failed,
                        model_revisions=model_revisions,
                        execution_configurations=execution_configurations,
                        batch_size=batch_size,
                        candidate_chunk_size=candidate_chunk_size,
                        provenance_binding_counts=provenance_binding_counts,
                    )
                    if progress:
                        progress("Computed local NLP annotations", processed, total)
                if isinstance(active_classifier, TransformersClassifier):
                    # Tasks run sequentially and never revisit an earlier model in this job.
                    # Retaining all three revision-pinned models only raises peak RAM before the
                    # disk-backed rollup, which matters on the target 12 GiB workstation.
                    active_classifier.release(spec)
            rollup_count = _rebuild_nlp_hourly_rollups_locked(
                engine,
                tasks=selected_tasks,
                inference_configurations=inference_configurations,
                provenance_binding_counts=provenance_binding_counts,
            )
            final_status = "completed_with_errors" if failed else "completed"
            with Session(engine) as session, session.begin():
                job = session.get(Job, job_id)
                if job is not None:
                    job.status = final_status
                    job.stage = "Local NLP annotations complete"
                    job.error_text = None if not failed else "Some annotations failed"
                    job.finished_at = datetime.now(UTC)
            return AnnotationSummary(
                job_id=job_id,
                pipeline_version=PIPELINE_VERSION,
                target_count=candidate_count,
                task_count=len(selected_tasks),
                annotated=annotated,
                reused=reused,
                failed=failed,
                resumed=resumed,
                rollup_count=rollup_count,
                model_specs={task: asdict(MODEL_SPECS[task]) for task in selected_tasks},
            )
        except BaseException as exc:
            with Session(engine) as session, session.begin():
                session.execute(
                    update(Job)
                    .where(Job.id == job_id)
                    .values(
                        status="failed",
                        stage="Local NLP annotation job interrupted",
                        error_text=str(exc)[:MAX_ERROR_LENGTH],
                        finished_at=datetime.now(UTC),
                    )
                )
            raise


def _update_job(
    engine: Engine,
    job_id: str,
    *,
    processed: int,
    total: int,
    task: NlpTask,
    last_target_id: str | None,
    annotated: int,
    reused: int,
    failed: int,
    model_revisions: dict[NlpTask, str],
    execution_configurations: dict[str, dict[str, str]],
    batch_size: int,
    candidate_chunk_size: int,
    provenance_binding_counts: dict[NlpTask, dict[str, int]],
) -> None:
    with Session(engine) as session, session.begin():
        job = session.get(Job, job_id)
        if job is None:
            raise RuntimeError("NLP annotation job was not found")
        checkpoint = dict(job.checkpoint_json or {})
        checkpoint.update(
            {
                "pipeline_version": PIPELINE_VERSION,
                "model_revisions": model_revisions,
                "execution_configurations": execution_configurations,
                "batch_size": batch_size,
                "candidate_chunk_size": candidate_chunk_size,
                "task": task,
                "last_target_id": last_target_id,
            }
        )
        job.progress_current = processed
        job.progress_total = total
        job.checkpoint_json = checkpoint
        job.counters_json = {
            "annotated": annotated,
            "reused": reused,
            "failed": failed,
            "provenance_binding_counts": provenance_binding_counts,
        }


def _active_annotation_binding_kinds(engine: Engine, task: NlpTask) -> tuple[str, ...]:
    with Session(engine) as session:
        values = list(
            session.scalars(
                select(
                    func.json_extract(
                        NlpAnnotation.provenance_json,
                        f"$.{NLP_PROVENANCE_KEY}.binding_kind",
                    )
                )
                .where(
                    NlpAnnotation.task == task,
                    NlpAnnotation.active.is_(True),
                    NlpAnnotation.status == "completed",
                )
                .distinct()
            )
        )
    if any(not isinstance(value, str) or not value for value in values):
        raise RuntimeError("Active NLP annotations are missing inference provenance")
    return tuple(sorted(cast(str, value) for value in values))


def _rollup_configuration(
    spec: NlpModelSpec,
    *,
    inference_configuration: NlpInferenceConfiguration,
    binding_kinds: Sequence[str],
) -> tuple[dict[str, Any], str]:
    implementation = inference_configuration.payload.get("implementation")
    model_artifact = inference_configuration.payload.get("model_artifact")
    if not isinstance(implementation, dict) or not isinstance(model_artifact, dict):
        raise RuntimeError("NLP inference configuration is incomplete")
    configuration: dict[str, Any] = {
        "activation": spec.activation,
        "language_scope": spec.language_scope,
        "device": spec.device,
        "quantization": spec.quantization,
        "max_text_characters": MAX_TEXT_CHARACTERS,
        "tokenizer_truncation_enabled": True,
        "tokenizer_max_length": NLP_TOKENIZER_MAX_LENGTH,
        "time_bucket": "UTC hour",
        "direction_values": ["owner", "other", "activity", "audio", "attachment"],
        "aggregation": "sample count plus confidence and per-label probability sums",
        "model_artifact": model_artifact,
        "inference_backend": {
            "provider": implementation.get("provider"),
            "packages": implementation.get("packages", {}),
        },
        "inference_implementation_sha256": inference_configuration.payload.get(
            "implementation_sha256"
        ),
        "inference_configuration_sha256": inference_configuration.sha256,
        "annotation_binding_kinds": list(binding_kinds),
    }
    return configuration, _canonical_json_sha256(configuration)


def _rollup_row(
    *,
    task: NlpTask,
    hour_start_ms: int,
    direction: str,
    predicted_label: str,
    sample_count: int,
    confidence_sum: float,
    probability_sums: dict[str, float],
    configuration: dict[str, Any],
    configuration_sha256: str,
    spec: NlpModelSpec,
    generated_at: datetime,
) -> dict[str, Any]:
    return {
        "id": stable_id(
            "nlp-hourly-rollup",
            task,
            hour_start_ms,
            direction,
            PIPELINE_VERSION,
            spec.model_name,
            spec.model_revision,
            configuration_sha256,
            predicted_label,
        ),
        "task": task,
        "hour_start_ms": hour_start_ms,
        "direction": direction,
        "pipeline_version": PIPELINE_VERSION,
        "model_name": spec.model_name,
        "model_revision": spec.model_revision,
        "configuration_sha256": configuration_sha256,
        "configuration_json": configuration,
        "predicted_label": predicted_label,
        "sample_count": sample_count,
        "confidence_sum": confidence_sum,
        "probability_sums_json": {"labels": probability_sums},
        "active": True,
        "generated_at": generated_at,
    }


def rebuild_nlp_hourly_rollups(
    layout: ArchiveLayout,
    engine: Engine,
    *,
    tasks: Sequence[NlpTask] = ("sentiment", "emotion", "toxicity"),
    inference_configurations: dict[NlpTask, NlpInferenceConfiguration] | None = None,
    provenance_binding_counts: dict[NlpTask, dict[str, int]] | None = None,
) -> int:
    """Publish rollups while serialized with deletion and annotation publication."""

    lock_path = layout.catalog / "nlp-annotations.lock"
    with ExitStack() as archive_locks:
        archive_locks.enter_context(deletion_lock(layout))
        layout.ensure_runtime_directories()
        lock_handle = archive_locks.enter_context(lock_path.open("a+b"))
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another NLP annotation job is already running") from exc
        assert_derived_publication_allowed(engine, publisher="NLP rollup publication")
        return _rebuild_nlp_hourly_rollups_locked(
            engine,
            tasks=tasks,
            inference_configurations=inference_configurations,
            provenance_binding_counts=provenance_binding_counts,
        )


def _rebuild_nlp_hourly_rollups_locked(
    engine: Engine,
    *,
    tasks: Sequence[NlpTask] = ("sentiment", "emotion", "toxicity"),
    inference_configurations: dict[NlpTask, NlpInferenceConfiguration] | None = None,
    provenance_binding_counts: dict[NlpTask, dict[str, int]] | None = None,
) -> int:
    """Atomically publish versioned UTC-hour NLP aggregates with bounded Python memory."""
    selected_tasks = tuple(dict.fromkeys(tasks))
    if not selected_tasks or any(task not in MODEL_SPECS for task in selected_tasks):
        raise ValueError("Choose at least one supported NLP task")
    if inference_configurations is None or any(
        task not in inference_configurations for task in selected_tasks
    ):
        raise RuntimeError("NLP rollup publication requires exact inference configurations")

    now = datetime.now(UTC)
    insert_statement = sqlite_insert(NlpHourlyRollup)
    excluded = insert_statement.excluded
    upsert_statement = insert_statement.on_conflict_do_update(
        index_elements=["id"],
        set_={
            "task": excluded.task,
            "hour_start_ms": excluded.hour_start_ms,
            "direction": excluded.direction,
            "pipeline_version": excluded.pipeline_version,
            "model_name": excluded.model_name,
            "model_revision": excluded.model_revision,
            "configuration_sha256": excluded.configuration_sha256,
            "configuration_json": excluded.configuration_json,
            "predicted_label": excluded.predicted_label,
            "sample_count": excluded.sample_count,
            "confidence_sum": excluded.confidence_sum,
            "probability_sums_json": excluded.probability_sums_json,
            "active": True,
            "generated_at": excluded.generated_at,
        },
    )
    configurations = {
        task: (
            *_rollup_configuration(
                MODEL_SPECS[task],
                inference_configuration=inference_configurations[task],
                binding_kinds=(
                    tuple(sorted(provenance_binding_counts.get(task, {})))
                    if provenance_binding_counts is not None
                    else _active_annotation_binding_kinds(engine, task)
                ),
            ),
            MODEL_SPECS[task],
        )
        for task in selected_tasks
    }
    published = 0
    with engine.connect() as connection:
        # The application defaults SQLite temporary data to memory for interactive queries.
        # Rollup staging can be much larger than RAM, so this one connection explicitly spills
        # its temporary aggregate to disk and restores the application default before pooling.
        connection.exec_driver_sql("PRAGMA temp_store=FILE")
        connection.commit()
        try:
            with connection.begin():
                connection.exec_driver_sql(
                    """
                    CREATE TEMP TABLE nlp_rollup_bucket_stage (
                        task TEXT NOT NULL,
                        hour_start_ms INTEGER NOT NULL,
                        direction TEXT NOT NULL,
                        predicted_label TEXT NOT NULL,
                        sample_count INTEGER NOT NULL,
                        confidence_sum REAL NOT NULL,
                        PRIMARY KEY (task, hour_start_ms, direction, predicted_label)
                    ) WITHOUT ROWID
                    """
                )
                connection.exec_driver_sql(
                    """
                    CREATE TEMP TABLE nlp_rollup_probability_stage (
                        task TEXT NOT NULL,
                        hour_start_ms INTEGER NOT NULL,
                        direction TEXT NOT NULL,
                        predicted_label TEXT NOT NULL,
                        probability_label TEXT NOT NULL,
                        probability_sum REAL NOT NULL,
                        PRIMARY KEY (
                            task,
                            hour_start_ms,
                            direction,
                            predicted_label,
                            probability_label
                        )
                    ) WITHOUT ROWID
                    """
                )

                for task in selected_tasks:
                    spec = MODEL_SPECS[task]
                    parameters = {
                        "task": task,
                        "pipeline_version": PIPELINE_VERSION,
                        "model_name": spec.model_name,
                        "model_revision": spec.model_revision,
                    }
                    # JSON extraction and aggregation stay inside SQLite's compiled engine. Its
                    # GROUP BY temporary B-trees use this connection's FILE temp store, avoiding
                    # both an unbounded Python dict and millions of Python-driven staging writes.
                    connection.exec_driver_sql(
                        """
                        WITH eligible AS (
                            SELECT
                                CAST(
                                    json_extract(attribution_json, '$.timestamp_ms') AS INTEGER
                                ) AS timestamp_ms,
                                CAST(
                                    json_extract(attribution_json, '$.direction') AS TEXT
                                ) AS direction,
                                predicted_label,
                                confidence
                            FROM nlp_annotations
                            WHERE task = :task
                              AND pipeline_version = :pipeline_version
                              AND model_name = :model_name
                              AND model_revision = :model_revision
                              AND status = 'completed'
                              AND active = 1
                              AND json_type(
                                  attribution_json, '$.timestamp_ms'
                              ) = 'integer'
                              AND json_extract(attribution_json, '$.direction') IN (
                                  'owner', 'other', 'activity', 'audio', 'attachment'
                              )
                              AND predicted_label IS NOT NULL
                              AND predicted_label != ''
                              AND confidence IS NOT NULL
                        ),
                        normalized AS (
                            SELECT
                                CASE
                                    WHEN timestamp_ms >= 0 THEN
                                        timestamp_ms - (timestamp_ms % 3600000)
                                    ELSE
                                        timestamp_ms - (
                                            ((timestamp_ms % 3600000) + 3600000) % 3600000
                                        )
                                END AS hour_start_ms,
                                direction,
                                predicted_label,
                                confidence
                            FROM eligible
                        )
                        INSERT INTO temp.nlp_rollup_bucket_stage (
                            task,
                            hour_start_ms,
                            direction,
                            predicted_label,
                            sample_count,
                            confidence_sum
                        )
                        SELECT
                            :task,
                            hour_start_ms,
                            direction,
                            predicted_label,
                            COUNT(*),
                            SUM(confidence)
                        FROM normalized
                        GROUP BY hour_start_ms, direction, predicted_label
                        """,
                        parameters,
                    )
                    connection.exec_driver_sql(
                        """
                        WITH eligible AS (
                            SELECT
                                CAST(
                                    json_extract(attribution_json, '$.timestamp_ms') AS INTEGER
                                ) AS timestamp_ms,
                                CAST(
                                    json_extract(attribution_json, '$.direction') AS TEXT
                                ) AS direction,
                                predicted_label,
                                probabilities_json
                            FROM nlp_annotations
                            WHERE task = :task
                              AND pipeline_version = :pipeline_version
                              AND model_name = :model_name
                              AND model_revision = :model_revision
                              AND status = 'completed'
                              AND active = 1
                              AND json_type(
                                  attribution_json, '$.timestamp_ms'
                              ) = 'integer'
                              AND json_extract(attribution_json, '$.direction') IN (
                                  'owner', 'other', 'activity', 'audio', 'attachment'
                              )
                              AND predicted_label IS NOT NULL
                              AND predicted_label != ''
                        ),
                        normalized AS (
                            SELECT
                                CASE
                                    WHEN timestamp_ms >= 0 THEN
                                        timestamp_ms - (timestamp_ms % 3600000)
                                    ELSE
                                        timestamp_ms - (
                                            ((timestamp_ms % 3600000) + 3600000) % 3600000
                                        )
                                END AS hour_start_ms,
                                direction,
                                predicted_label,
                                probabilities_json
                            FROM eligible
                        )
                        INSERT INTO temp.nlp_rollup_probability_stage (
                            task,
                            hour_start_ms,
                            direction,
                            predicted_label,
                            probability_label,
                            probability_sum
                        )
                        SELECT
                            :task,
                            normalized.hour_start_ms,
                            normalized.direction,
                            normalized.predicted_label,
                            CAST(probability.key AS TEXT),
                            SUM(CAST(probability.value AS REAL))
                        FROM normalized
                        CROSS JOIN json_each(
                            normalized.probabilities_json, '$.labels'
                        ) AS probability
                        WHERE probability.type IN ('integer', 'real')
                          AND probability.key IS NOT NULL
                        GROUP BY
                            normalized.hour_start_ms,
                            normalized.direction,
                            normalized.predicted_label,
                            probability.key
                        """,
                        parameters,
                    )

                # Do not retire the currently published product until staging succeeded for all
                # requested tasks. Deactivation and replacement then commit atomically.
                connection.execute(
                    update(NlpHourlyRollup)
                    .where(
                        NlpHourlyRollup.task.in_(selected_tasks),
                        NlpHourlyRollup.active.is_(True),
                    )
                    .values(active=False)
                )
                staged_rows = connection.exec_driver_sql(
                    """
                    SELECT
                        bucket.task,
                        bucket.hour_start_ms,
                        bucket.direction,
                        bucket.predicted_label,
                        bucket.sample_count,
                        bucket.confidence_sum,
                        probability.probability_label,
                        probability.probability_sum
                    FROM temp.nlp_rollup_bucket_stage AS bucket
                    LEFT JOIN temp.nlp_rollup_probability_stage AS probability
                      ON probability.task = bucket.task
                     AND probability.hour_start_ms = bucket.hour_start_ms
                     AND probability.direction = bucket.direction
                     AND probability.predicted_label = bucket.predicted_label
                    ORDER BY
                        bucket.task,
                        bucket.hour_start_ms,
                        bucket.direction,
                        bucket.predicted_label,
                        probability.probability_label
                    """
                )
                publish_buffer: list[dict[str, Any]] = []
                current_key: tuple[str, int, str, str] | None = None
                current_stats: tuple[int, float] | None = None
                current_probabilities: dict[str, float] = {}

                def buffer_current_row() -> None:
                    nonlocal published
                    if current_key is None or current_stats is None:
                        return
                    task_value, hour_start_ms, direction, predicted_label = current_key
                    configuration, configuration_sha256, spec = configurations[
                        cast(NlpTask, task_value)
                    ]
                    publish_buffer.append(
                        _rollup_row(
                            task=cast(NlpTask, task_value),
                            hour_start_ms=hour_start_ms,
                            direction=direction,
                            predicted_label=predicted_label,
                            sample_count=current_stats[0],
                            confidence_sum=current_stats[1],
                            probability_sums=current_probabilities.copy(),
                            configuration=configuration,
                            configuration_sha256=configuration_sha256,
                            spec=spec,
                            generated_at=now,
                        )
                    )
                    published += 1

                for row in staged_rows:
                    row_key = cast(tuple[str, int, str, str], tuple(row[:4]))
                    if current_key is not None and row_key != current_key:
                        buffer_current_row()
                        current_probabilities.clear()
                        if len(publish_buffer) >= NLP_ROLLUP_PUBLISH_BATCH_SIZE:
                            connection.execute(upsert_statement, publish_buffer)
                            publish_buffer.clear()
                    current_key = row_key
                    current_stats = (int(row[4]), float(row[5]))
                    if row[6] is not None:
                        current_probabilities[str(row[6])] = float(row[7])
                buffer_current_row()
                if publish_buffer:
                    connection.execute(upsert_statement, publish_buffer)
                connection.exec_driver_sql("DROP TABLE temp.nlp_rollup_probability_stage")
                connection.exec_driver_sql("DROP TABLE temp.nlp_rollup_bucket_stage")
        finally:
            if connection.in_transaction():
                connection.rollback()
            connection.exec_driver_sql("DROP TABLE IF EXISTS temp.nlp_rollup_probability_stage")
            connection.exec_driver_sql("DROP TABLE IF EXISTS temp.nlp_rollup_bucket_stage")
            connection.commit()
            connection.exec_driver_sql("PRAGMA temp_store=MEMORY")
            connection.commit()
    return published


def conflict_aggregation_status(engine: Engine) -> dict[str, Any]:
    with Session(engine) as session:
        calibration = session.scalar(
            select(ConflictCalibration)
            .where(
                ConflictCalibration.pipeline_version == CONFLICT_PIPELINE_VERSION,
                ConflictCalibration.model_name == CONFLICT_MODEL_NAME,
                ConflictCalibration.model_revision == CONFLICT_MODEL_REVISION,
            )
            .order_by(
                ConflictCalibration.completed_at.desc(), ConflictCalibration.created_at.desc()
            )
            .limit(1)
        )
        label_count = (
            session.scalar(select(func.count(func.distinct(ConflictLabel.window_id)))) or 0
        )
    meets_metrics = bool(
        calibration
        and calibration.status == "calibrated"
        and label_count >= MIN_CONFLICT_LABELS
        and calibration.sample_count >= MIN_CONFLICT_LABELS
        and (calibration.direct_conflict_precision or 0) >= MIN_DIRECT_CONFLICT_PRECISION
        and (calibration.macro_f1 or 0) >= MIN_MACRO_F1
        and calibration.metrics_json.get("evaluation_source") == "held_out_test_predictions"
        and calibration.metrics_json.get("conversation_disjoint") is True
        and calibration.metrics_json.get("stratified") is True
        and all(
            calibration.provenance_json.get("evaluation_splits", {}).get(split, 0) > 0
            for split in ("train", "validation", "test")
        )
    )
    enabled = bool(calibration and calibration.aggregation_enabled and meets_metrics)
    reason = (
        None
        if enabled
        else (
            "No calibration has been recorded for the current conflict model."
            if calibration is None
            else (
                "The latest calibration is stale for the current derived data."
                if calibration.status == "stale"
                else (
                    "The latest held-out calibration did not pass every sample, split, "
                    "stratification, precision, and macro-F1 gate."
                )
            )
        )
    )
    warning = (
        "Conflict aggregates are hidden until 500 stratified, conversation-disjoint labels "
        "and held-out precision/macro-F1 gates pass. Never rank people or relationships."
    )
    return {
        "aggregation_enabled": enabled,
        "status": "enabled" if enabled else "disabled_until_calibrated",
        "sample_count": label_count,
        "label_count": label_count,
        "minimum_label_count": MIN_CONFLICT_LABELS,
        "thresholds": {
            "direct_conflict_precision": MIN_DIRECT_CONFLICT_PRECISION,
            "macro_f1": MIN_MACRO_F1,
        },
        "latest_calibration": {
            "id": calibration.id,
            "status": calibration.status,
            "sample_count": calibration.sample_count,
            "direct_conflict_precision": calibration.direct_conflict_precision,
            "macro_f1": calibration.macro_f1,
            "recorded_enabled": calibration.aggregation_enabled,
            "evaluation_source": calibration.metrics_json.get("evaluation_source"),
            "conversation_disjoint": calibration.metrics_json.get("conversation_disjoint"),
            "stratified": calibration.metrics_json.get("stratified"),
        }
        if calibration
        else None,
        "reason": reason,
        "warning": warning,
        "relationship_ranking_enabled": False,
    }


def _conflict_metrics(
    actual: Sequence[str], predicted: Sequence[str]
) -> tuple[float, float, dict[str, Any]]:
    if len(actual) != len(predicted) or not actual:
        raise RuntimeError("Held-out conflict predictions are incomplete")
    labels = tuple(str(label) for label in CONFLICT_LABELS)
    confusion = {expected: {observed: 0 for observed in labels} for expected in labels}
    for expected, observed in zip(actual, predicted, strict=True):
        if expected not in confusion or observed not in confusion[expected]:
            raise RuntimeError("Held-out conflict predictions contained an unsupported label")
        confusion[expected][observed] += 1
    per_label: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for label in labels:
        true_positive = confusion[label][label]
        false_positive = sum(confusion[other][label] for other in labels if other != label)
        false_negative = sum(confusion[label][other] for other in labels if other != label)
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_label[label] = {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    direct_conflict_precision = float(per_label["direct conflict"]["precision"])
    macro_f1 = sum(f1_values) / len(labels)
    return (
        direct_conflict_precision,
        macro_f1,
        {
            "per_label": per_label,
            "confusion_matrix": confusion,
        },
    )


def record_conflict_calibration(
    layout: ArchiveLayout,
    engine: Engine,
    *,
    scorer_factory: ConflictScorerFactory | None = None,
    require_eligible: bool = False,
) -> ConflictCalibration:
    """Serialize calibration publication against hard purge and other calibrators."""

    lock_path = layout.catalog / "conflict-calibration.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as archive_locks, lock_path.open("a+b") as lock_handle:
        archive_locks.enter_context(deletion_lock(layout))
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another conflict calibration is currently running") from exc
        try:
            assert_derived_publication_allowed(engine, publisher="Conflict calibration publication")
            return _record_conflict_calibration_locked(
                layout,
                engine,
                scorer_factory=scorer_factory,
                require_eligible=require_eligible,
            )
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _record_conflict_calibration_locked(
    layout: ArchiveLayout,
    engine: Engine,
    *,
    scorer_factory: ConflictScorerFactory | None = None,
    require_eligible: bool = False,
) -> ConflictCalibration:
    if require_eligible:
        review = conflict_review_status(engine, minimum_label_count=MIN_CONFLICT_LABELS)
        if not review["validation"]["eligible_for_calibration"]:
            raise ConflictCalibrationIneligibleError(review["validation"])
    with Session(engine) as session:
        raw_labels = list(
            session.execute(
                select(
                    ConflictLabel.window_id,
                    ConflictLabel.label,
                    ConflictLabel.evaluation_split,
                    SearchWindow.conversation_id,
                    SearchWindow.text_display,
                    SearchWindow.entity_type,
                    ConflictLabel.created_at,
                )
                .join(SearchWindow, SearchWindow.id == ConflictLabel.window_id)
                .order_by(ConflictLabel.created_at.desc(), ConflictLabel.id.desc())
            )
        )
    labels_by_window: dict[str, tuple[str, str, str | None, str, str]] = {}
    for (
        window_id,
        label,
        split,
        conversation_id,
        text_display,
        entity_type,
        _created_at,
    ) in raw_labels:
        labels_by_window.setdefault(
            window_id,
            (label, split, conversation_id, text_display, entity_type),
        )
    labelled_windows = [
        (window_id, *label_data) for window_id, label_data in labels_by_window.items()
    ]
    splits = CONFLICT_EVALUATION_SPLITS
    conversations_by_split: dict[str, set[str]] = {split: set() for split in splits}
    label_distribution: dict[str, int] = {label: 0 for label in CONFLICT_LABELS}
    split_label_distribution: dict[str, dict[str, int]] = {
        split: {label: 0 for label in CONFLICT_LABELS} for split in splits
    }
    split_window_counts = {split: 0 for split in splits}
    missing_conversation_count = 0
    invalid_entity_count = 0
    unsupported_label_count = 0
    unsupported_split_count = 0
    split_assignment_mismatch_count = 0
    for (
        _window_id,
        label,
        split,
        conversation_id,
        _text_display,
        entity_type,
    ) in labelled_windows:
        if entity_type != "message_window":
            invalid_entity_count += 1
        if split in conversations_by_split and conversation_id:
            conversations_by_split[split].add(conversation_id)
        elif not conversation_id:
            missing_conversation_count += 1
        if split not in conversations_by_split:
            unsupported_split_count += 1
        if conversation_id and split != deterministic_conflict_split(conversation_id):
            split_assignment_mismatch_count += 1
        if label in label_distribution:
            label_distribution[label] += 1
            if split in split_label_distribution:
                split_label_distribution[split][label] += 1
        else:
            unsupported_label_count += 1
        if split in split_window_counts:
            split_window_counts[split] += 1
    disjoint = all(
        conversations_by_split.get(left, set()).isdisjoint(conversations_by_split.get(right, set()))
        for index, left in enumerate(splits)
        for right in splits[index + 1 :]
    )
    stratified = all(
        split_label_distribution[split][label] > 0 for split in splits for label in CONFLICT_LABELS
    )
    sample_count = len(labelled_windows)
    data_gates_pass = (
        sample_count >= MIN_CONFLICT_LABELS
        and all(conversations_by_split.get(split) for split in splits)
        and disjoint
        and stratified
        and missing_conversation_count == 0
        and invalid_entity_count == 0
        and unsupported_label_count == 0
        and unsupported_split_count == 0
        and split_assignment_mismatch_count == 0
    )
    if require_eligible and not data_gates_pass:
        refreshed = conflict_review_status(engine, minimum_label_count=MIN_CONFLICT_LABELS)
        raise ConflictCalibrationIneligibleError(refreshed["validation"])
    direct_conflict_precision: float | None = None
    macro_f1: float | None = None
    evaluation_metrics: dict[str, Any] = {}
    if data_gates_pass:
        scorer = (
            scorer_factory(DEFAULT_CONFLICT_SPEC)
            if scorer_factory is not None
            else ModernBertNliConflictScorer(
                layout.models / "conflict" / DEFAULT_CONFLICT_SPEC.model_revision,
                DEFAULT_CONFLICT_SPEC,
            )
        )
        held_out = [
            (window_id, label, text_display)
            for (
                window_id,
                label,
                split,
                _conversation_id,
                text_display,
                _entity_type,
            ) in labelled_windows
            if split == "test"
        ]
        predicted: list[str] = []
        for start in range(0, len(held_out), DEFAULT_CONFLICT_SPEC.max_candidates):
            batch = held_out[start : start + DEFAULT_CONFLICT_SPEC.max_candidates]
            vectors = scorer.score([text_display for _window_id, _label, text_display in batch])
            if len(vectors) != len(batch):
                raise RuntimeError("Conflict scorer returned an incomplete held-out batch")
            predicted.extend(vector.predicted_label for vector in vectors)
        direct_conflict_precision, macro_f1, evaluation_metrics = _conflict_metrics(
            [label for _window_id, label, _text_display in held_out], predicted
        )
        evaluation_metrics["held_out_predictions"] = [
            {"window_id": window_id, "expected": label, "predicted": prediction}
            for (window_id, label, _text_display), prediction in zip(
                held_out, predicted, strict=True
            )
        ]
    enabled = bool(
        data_gates_pass
        and direct_conflict_precision is not None
        and macro_f1 is not None
        and direct_conflict_precision >= MIN_DIRECT_CONFLICT_PRECISION
        and macro_f1 >= MIN_MACRO_F1
    )
    row = ConflictCalibration(
        id=str(uuid.uuid4()),
        pipeline_version=CONFLICT_PIPELINE_VERSION,
        model_name=CONFLICT_MODEL_NAME,
        model_revision=CONFLICT_MODEL_REVISION,
        status="calibrated" if enabled else "gates_failed",
        sample_count=sample_count,
        direct_conflict_precision=direct_conflict_precision,
        macro_f1=macro_f1,
        aggregation_enabled=enabled,
        metrics_json={
            **evaluation_metrics,
            "evaluation_source": "held_out_test_predictions"
            if data_gates_pass
            else "not_evaluated_data_gates_failed",
            "conversation_disjoint": disjoint,
            "stratified": stratified,
            "exact_split_assignment": split_assignment_mismatch_count == 0,
            "all_six_labels_present": all(label_distribution[label] for label in CONFLICT_LABELS),
        },
        provenance_json={
            "evaluation_splits": {key: len(value) for key, value in conversations_by_split.items()},
            "split_window_counts": split_window_counts,
            "split_label_distribution": split_label_distribution,
            "label_distribution": label_distribution,
            "unique_window_count": sample_count,
            "missing_conversation_count": missing_conversation_count,
            "invalid_entity_count": invalid_entity_count,
            "unsupported_label_count": unsupported_label_count,
            "unsupported_split_count": unsupported_split_count,
            "split_assignment_mismatch_count": split_assignment_mismatch_count,
            "held_out_prediction_count": split_window_counts["test"] if data_gates_pass else 0,
            "minimum_unique_windows": MIN_CONFLICT_LABELS,
            "model_spec": asdict(DEFAULT_CONFLICT_SPEC),
            "model_output_is_ranking_feature": True,
        },
        warnings_json={
            "items": [
                "Conflict labels are model estimates and must not rank people or relationships."
            ]
        },
        completed_at=datetime.now(UTC),
    )
    with Session(engine) as session, session.begin():
        session.add(row)
        session.flush()
        session.expunge(row)
    # A full-corpus aggregate is revision-bound to one exact held-out calibration.
    # Import locally to avoid a module cycle during service initialization.
    from meta_archive.services.conflict_aggregates import invalidate_conflict_aggregates

    invalidate_conflict_aggregates(
        engine,
        reason="A new held-out calibration was recorded; rebuild conflict aggregates",
    )
    return row
