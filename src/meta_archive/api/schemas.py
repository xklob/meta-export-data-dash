from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

NlpJobTask = Literal["sentiment", "emotion", "toxicity"]
EvaluationIntent = Literal[
    "argument_direct_conflict",
    "argument_direct_conflict_development",
]


def _default_nlp_job_tasks() -> list[NlpJobTask]:
    return ["sentiment", "emotion", "toxicity"]


class SettingsUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timezone: str | None = None
    cloud_enabled: bool | None = None


class SemanticIndexRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: (
        Literal[
            "bge-base-en-v1.5",
            "e5-base-v2",
            "gte-modernbert-base",
            "bge-small-en-v1.5",
            "mxbai-embed-large-v1",
            "multilingual-e5-small",
            "snowflake-arctic-embed-l",
        ]
        | None
    ) = Field(
        default=None,
        description=(
            "Embedding model alias. The default auto policy selects bge-base-en-v1.5. Supplying "
            "an alias requests the normal audited build; manual acceptance override is "
            "intentionally unavailable over HTTP."
        ),
    )
    rebuild_windows: bool = True


class AudioTranscriptionJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    retry_failed: bool = True


class AttachmentTextJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    retry_failed: bool = True


class NlpAnnotationJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tasks: list[NlpJobTask] = Field(
        default_factory=_default_nlp_job_tasks,
        min_length=1,
        max_length=3,
    )
    batch_size: int = Field(default=16, ge=1, le=32)


class ConflictAggregateJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_size: int = Field(default=32, ge=1, le=100)


class SavedSearchCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=160)
    query: dict[str, Any]


class SavedSearchUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=160)
    query: dict[str, Any] | None = None


class SearchFilterScope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_type: Literal["all", "message", "activity", "audio_transcript", "attachment_text"] = (
        "all"
    )
    conversation_id: str | None = Field(default=None, max_length=128)
    participant: str | None = Field(default=None, max_length=200)
    source_category: str | None = Field(default=None, max_length=80)
    from_ms: int | None = Field(default=None, ge=0)
    to_ms: int | None = Field(default=None, ge=0)


class RelevanceFeedbackCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2_000)
    search_mode: Literal["keyword", "exact", "semantic", "hybrid"]
    filter_scope: SearchFilterScope = Field(default_factory=SearchFilterScope)
    result_id: str = Field(min_length=1, max_length=128)
    label: Literal["relevant", "not_relevant", "uncertain"]
    evaluation_split: Literal["train", "validation", "test"] | None = None
    evaluation_intent: EvaluationIntent | None = None
    notes: str | None = Field(default=None, max_length=2_000)


class RelevanceFeedbackUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: Literal["relevant", "not_relevant", "uncertain"] | None = None
    evaluation_split: Literal["train", "validation", "test"] | None = None
    evaluation_intent: EvaluationIntent | None = None
    notes: str | None = Field(default=None, max_length=2_000)


class RagQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2_000)
    search_mode: Literal["semantic", "hybrid"] = "hybrid"
    top_k: int = Field(default=8, ge=1, le=20)
    entity_type: Literal["all", "message", "activity", "audio_transcript", "attachment_text"] = (
        "all"
    )
    conversation_id: str | None = Field(default=None, max_length=128)
    participant: str | None = Field(default=None, max_length=200)
    source_category: str | None = Field(default=None, max_length=80)
    from_ms: int | None = Field(default=None, ge=0)
    to_ms: int | None = Field(default=None, ge=0)


class RagStructuredAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    citation_ids: list[str]
    insufficient_evidence: bool


class ConflictLabelCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window_id: str = Field(min_length=1, max_length=64)
    label: Literal[
        "direct conflict",
        "calm disagreement",
        "banter",
        "third-party conflict",
        "no conflict",
        "uncertain",
    ]
    notes: str | None = Field(default=None, max_length=2_000)


class ConflictCalibrationCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
