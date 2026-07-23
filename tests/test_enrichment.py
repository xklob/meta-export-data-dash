from __future__ import annotations

import copy
import hashlib
import io
import math
import shutil
import stat
import subprocess
import tarfile
import warnings
import zipfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, event, func, select, text
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from meta_archive.api import enrichment_routes
from meta_archive.api.main import create_app
from meta_archive.cli import app as cli_app
from meta_archive.models import (
    AttachmentOccurrence,
    AttachmentTextExtraction,
    AudioTranscription,
    AudioTranscriptSegment,
    Blob,
    ConflictCalibration,
    Conversation,
    Job,
    NlpAnnotation,
    NlpHourlyRollup,
    SearchWindow,
    SemanticIndexConfig,
)
from meta_archive.models import (
    ConflictLabel as ConflictLabelRow,
)
from meta_archive.paths import ArchiveLayout
from meta_archive.services import attachment_worker
from meta_archive.services.activity_importer import import_activities
from meta_archive.services.attachment_text import (
    BubblewrapAttachmentExtractor,
    ExtractionResult,
    build_attachment_text,
    sandbox_capability,
)
from meta_archive.services.conflict_review import deterministic_conflict_split
from meta_archive.services.conflict_scoring import (
    CONFLICT_LABELS,
    DEFAULT_CONFLICT_SPEC,
    ConflictModelSpec,
    ConflictScoreVector,
    ConflictScoringUnavailableError,
    ModernBertNliConflictScorer,
    conflict_search,
)
from meta_archive.services.inventory import scan_archive
from meta_archive.services.message_importer import import_messages
from meta_archive.services.nlp_annotations import (
    DIRECT_PROVENANCE_BINDING,
    INJECTED_PROVENANCE_BINDING,
    LEGACY_PROVENANCE_BINDING,
    MODEL_SPECS,
    NLP_PROVENANCE_KEY,
    ClassificationResult,
    NlpInferenceConfiguration,
    NlpModelSpec,
    TransformersClassifier,
    _canonical_json_sha256,
    _reuse_provenance_for_configuration,
    build_nlp_annotations,
    conflict_aggregation_status,
    record_conflict_calibration,
)
from meta_archive.services.reranking import DEFAULT_RERANKER_SPEC, RerankerSpec
from meta_archive.services.search import (
    SearchFilters,
    derived_occurrence_page,
    lexical_search,
    semantic_search,
)
from meta_archive.services.search_index import EmbeddingModelSpec, build_semantic_index
from meta_archive.services.transcription import (
    FasterWhisperSmallCpuInt8,
    TranscriptResult,
    TranscriptSegmentResult,
    build_audio_transcriptions,
)


def _materialized_attachment(
    layout: ArchiveLayout,
    engine: Engine,
    *,
    content: bytes,
    filename: str,
    media_type: str,
    kind: str,
    occurrences: int = 1,
    timestamp_ms: int | None = 1_704_067_200_123,
) -> str:
    sha256 = hashlib.sha256(content).hexdigest()
    blob_path = layout.blob_path(sha256)
    blob_path.parent.mkdir(parents=True, exist_ok=True)
    blob_path.write_bytes(content)
    blob_path.chmod(0o444)
    with Session(engine) as session, session.begin():
        session.add(
            Blob(
                sha256=sha256,
                size_bytes=len(content),
                media_type=media_type,
                extension=Path(filename).suffix,
                relative_path=blob_path.relative_to(layout.root).as_posix(),
                state="materialized",
            )
        )
        session.flush()
        for ordinal in range(occurrences):
            session.add(
                AttachmentOccurrence(
                    id=hashlib.sha256(f"{sha256}:{ordinal}".encode()).hexdigest(),
                    kind=kind,
                    ordinal=ordinal,
                    filename=filename,
                    media_type=media_type,
                    size_bytes=len(content),
                    timestamp_ms=timestamp_ms,
                    availability="unlinked",
                    blob_sha256=sha256,
                    metadata_json={},
                )
            )
    return sha256


class FakeTranscriber:
    identity = "fake-transcriber"
    model_name = "fixture/whisper-small"
    model_revision = "fixture-whisper-revision"
    device = "cpu"
    compute_type = "int8"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[Path] = []

    def transcribe(self, source: Path) -> TranscriptResult:
        self.calls.append(source)
        if self.fail:
            raise RuntimeError("fixture transcription failure")
        return TranscriptResult(
            segments=(
                TranscriptSegmentResult(
                    start_ms=125,
                    end_ms=1_500,
                    text="Transcript needle first segment",
                    avg_log_probability=-0.1,
                    no_speech_probability=0.02,
                    words=({"word": "Transcript", "start_ms": 125, "end_ms": 600},),
                ),
                TranscriptSegmentResult(
                    start_ms=1_500,
                    end_ms=2_500,
                    text="Second timestamped segment",
                    avg_log_probability=-0.2,
                    no_speech_probability=0.03,
                ),
            ),
            language="en",
            language_probability=0.98,
            duration_ms=2_500,
        )


class FakeExtractor:
    identity = "fake-sandbox-extractor"
    revision = "fixture-extractor-v1"
    sandbox_status: ClassVar[dict[str, object]] = {
        "available": True,
        "status": "available",
    }

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[Path, str, str | None]] = []

    def extract(self, source: Path, *, filename: str, media_type: str | None) -> ExtractionResult:
        self.calls.append((source, filename, media_type))
        if self.fail:
            raise RuntimeError("fixture extraction failure")
        return ExtractionResult(
            status="completed",
            text="Attachment needle safely extracted",
            truncated=False,
            index_eligible=True,
            diagnostic=None,
            archive_entries=1,
            expanded_bytes=32,
        )


def _ready_semantic_config(config_id: str) -> SemanticIndexConfig:
    return SemanticIndexConfig(
        id=config_id,
        active=True,
        status="ready",
        backend="fixture",
        backend_version="fixture-v1",
        model_name="fixture/embedding",
        model_revision="fixture-v1",
        dimensions=4,
        distance_metric="cosine",
        document_prefix="",
        query_prefix="",
        vector_table=f"fixture_vectors_{config_id[:8]}",
        window_version="fixture-window-v1",
        window_count=1,
        audit_json={"fixture": True},
    )


class FakeClassifier:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def classify(self, spec: NlpModelSpec, texts: Sequence[str]) -> list[ClassificationResult]:
        self.calls.append((spec.task, list(texts)))
        probabilities = (
            {"negative": 0.1, "neutral": 0.2, "positive": 0.7}
            if spec.task == "sentiment"
            else {"joy": 0.6, "sadness": 0.4}
            if spec.task == "emotion"
            else {"toxic": 0.25, "insult": 0.1}
        )
        return [ClassificationResult(probabilities=probabilities) for _text in texts]


FIXTURE_EMBEDDING_SPEC = EmbeddingModelSpec(
    alias="fixture-enrichment",
    model_name="fixture/enrichment",
    model_revision="fixture-enrichment-v1",
    dimensions=4,
    document_prefix="",
    query_prefix="",
    language_scope="fixture",
    approximate_download_gb=0,
)


class FakeEmbedder:
    spec = FIXTURE_EMBEDDING_SPEC

    @staticmethod
    def _vector(value: str) -> list[float]:
        digest = hashlib.sha256(value.encode()).digest()
        raw = [float(digest[index] + 1) for index in range(4)]
        norm = sum(value * value for value in raw) ** 0.5
        return [value / norm for value in raw]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(value) for value in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class FakeReranker:
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


def fake_reranker_factory(_spec: RerankerSpec) -> FakeReranker:
    return FakeReranker()


class FakeConflictScorer:
    spec = DEFAULT_CONFLICT_SPEC

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[list[str]] = []

    @property
    def status(self) -> dict[str, object]:
        return {
            "status": "loaded_fixture",
            "model_name": self.spec.model_name,
            "model_revision": self.spec.model_revision,
            "device": "cpu",
            "quantization": self.spec.quantization,
            "score_kind": self.spec.score_kind,
            "max_candidates": self.spec.max_candidates,
            "batch_size": self.spec.batch_size,
            "ranking_feature_only": True,
            "aggregation_gate_bypassed": False,
            "labels": list(CONFLICT_LABELS),
        }

    def score(self, documents: Sequence[str]) -> list[ConflictScoreVector]:
        self.calls.append(list(documents))
        if self.fail:
            raise RuntimeError("fixture NLI scoring unavailable")
        return [
            ConflictScoreVector(
                scores={
                    label: float(index + 2) if label == "direct conflict" else float(-label_index)
                    for label_index, label in enumerate(CONFLICT_LABELS)
                }
            )
            for index, _document in enumerate(documents)
        ]


def fake_conflict_scorer_factory(_spec: ConflictModelSpec) -> FakeConflictScorer:
    return FakeConflictScorer()


class LabelEchoConflictScorer(FakeConflictScorer):
    def score(self, documents: Sequence[str]) -> list[ConflictScoreVector]:
        self.calls.append(list(documents))
        results: list[ConflictScoreVector] = []
        for document in documents:
            expected = document.removeprefix("expected:").strip()
            results.append(
                ConflictScoreVector(
                    scores={
                        label: 10.0 if label == expected else -10.0 for label in CONFLICT_LABELS
                    }
                )
            )
        return results


def _fixture_conversation_for_split(prefix: str, split: str) -> str:
    for ordinal in range(10_000):
        conversation_id = f"{prefix}-{ordinal}"
        if deterministic_conflict_split(conversation_id) == split:
            return conversation_id
    raise AssertionError(f"Could not create a fixture conversation for {split}")


def test_transcription_is_unique_resumable_timestamped_indexed_and_lazy(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> None:
    audio_sha = _materialized_attachment(
        archive_layout,
        catalog_engine,
        content=b"fixture audio bytes",
        filename="voice.m4a",
        media_type="video/mp4",
        kind="audio_files",
        occurrences=25,
    )
    production_adapter = FasterWhisperSmallCpuInt8(archive_layout.models / "fixture")
    assert production_adapter._model is None
    adapter = FakeTranscriber()

    first = build_audio_transcriptions(archive_layout, catalog_engine, adapter=adapter)
    second = build_audio_transcriptions(archive_layout, catalog_engine, adapter=adapter)

    assert first.total == first.transcribed == 1
    assert second.total == second.reused == 1
    assert len(adapter.calls) == 1
    with Session(catalog_engine) as session:
        transcription = session.scalar(select(AudioTranscription))
        segments = list(
            session.scalars(select(AudioTranscriptSegment).order_by(AudioTranscriptSegment.ordinal))
        )
        fts_count = session.scalar(
            text("SELECT count(*) FROM derived_text_fts WHERE derived_text_fts MATCH 'transcript'")
        )
    assert transcription is not None
    assert transcription.blob_sha256 == audio_sha
    assert transcription.segment_count == 2
    assert [segment.start_ms for segment in segments] == [125, 1_500]
    assert segments[0].confidence == pytest.approx(math.exp(-0.1))
    assert fts_count == 1

    results = lexical_search(
        catalog_engine,
        "transcript needle",
        filters=SearchFilters(entity_type="audio_transcript"),
    )
    assert [item["entity_type"] for item in results] == ["audio_transcript"]
    assert len(results[0]["occurrences"]) == 20
    assert results[0]["occurrence_page"] == {
        "limit": 20,
        "offset": 0,
        "returned": 20,
        "total": 25,
        "next_offset": 20,
    }
    assert "t=125" in results[0]["jump"]["href"]
    next_occurrences = derived_occurrence_page(
        catalog_engine,
        entity_type="audio_transcript",
        source_id=results[0]["source_id"],
        filters=SearchFilters(entity_type="audio_transcript"),
        offset=20,
    )
    assert len(next_occurrences["items"]) == 5
    assert next_occurrences["page"]["next_offset"] is None
    app = create_app(archive_layout.root, security_disabled=True)
    with TestClient(app) as client:
        occurrence_response = client.get(
            "/api/v1/search/derived-occurrences",
            params={
                "entity_type": "audio_transcript",
                "source_id": results[0]["source_id"],
                "offset": 20,
                "limit": 20,
            },
        )
    assert occurrence_response.status_code == 200
    assert len(occurrence_response.json()["items"]) == 5


def test_searchable_transcription_and_extraction_mark_ready_semantic_indexes_stale(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> None:
    _materialized_attachment(
        archive_layout,
        catalog_engine,
        content=b"stale audio fixture",
        filename="stale.m4a",
        media_type="audio/mp4",
        kind="audio_files",
    )
    with Session(catalog_engine) as session, session.begin():
        session.add(_ready_semantic_config("a" * 64))

    build_audio_transcriptions(archive_layout, catalog_engine, adapter=FakeTranscriber())

    with Session(catalog_engine) as session:
        audio_config = session.get(SemanticIndexConfig, "a" * 64)
        assert audio_config is not None
        assert (audio_config.active, audio_config.status) == (False, "stale")
        assert audio_config.audit_json["stale"]["source_kind"] == "audio_transcription"

    _materialized_attachment(
        archive_layout,
        catalog_engine,
        content=b"stale document fixture",
        filename="stale.txt",
        media_type="text/plain",
        kind="files",
    )
    with Session(catalog_engine) as session, session.begin():
        session.add(_ready_semantic_config("b" * 64))

    build_attachment_text(archive_layout, catalog_engine, extractor=FakeExtractor())

    with Session(catalog_engine) as session:
        attachment_config = session.get(SemanticIndexConfig, "b" * 64)
        assert attachment_config is not None
        assert (attachment_config.active, attachment_config.status) == (False, "stale")
        assert attachment_config.audit_json["stale"]["source_kind"] == "attachment_text_extraction"


def test_derived_lexical_filters_scope_candidates_before_limit(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> None:
    for index in range(2):
        _materialized_attachment(
            archive_layout,
            catalog_engine,
            content=f"scoped audio fixture {index}".encode(),
            filename=f"scope-{index}.m4a",
            media_type="audio/mp4",
            kind="audio_files",
            timestamp_ms=1_704_067_200_123 + index,
        )
    build_audio_transcriptions(archive_layout, catalog_engine, adapter=FakeTranscriber())

    with Session(catalog_engine) as session, session.begin():
        conversations = [
            Conversation(
                id=f"scope-conversation-{index}",
                conversation_key=f"scope-key-{index}",
                title_raw=f"Scope {index}",
                title_display=f"Scope {index}",
                title_search=f"scope {index}",
                conversation_type="direct",
                primary_category="inbox",
            )
            for index in range(2)
        ]
        session.add_all(conversations)
        session.flush()
        segment_rows = list(
            session.execute(
                select(
                    AudioTranscriptSegment.id,
                    AudioTranscription.blob_sha256,
                )
                .join(
                    AudioTranscription,
                    AudioTranscription.id == AudioTranscriptSegment.transcription_id,
                )
                .where(AudioTranscriptSegment.ordinal == 0)
                .order_by(AudioTranscriptSegment.id)
            )
        )
        assert len(segment_rows) == 2
        for index, (_segment_id, blob_sha256) in enumerate(segment_rows):
            occurrence = session.scalar(
                select(AttachmentOccurrence).where(AttachmentOccurrence.blob_sha256 == blob_sha256)
            )
            assert occurrence is not None
            occurrence.conversation_id = conversations[index].id
        target_segment_id = str(segment_rows[1].id)
        target_conversation_id = conversations[1].id

    filters = SearchFilters(
        entity_type="audio_transcript",
        conversation_id=target_conversation_id,
    )
    exact = lexical_search(
        catalog_engine,
        "Transcript needle",
        filters=filters,
        exact=True,
        limit=1,
    )
    keyword = lexical_search(
        catalog_engine,
        "transcript needle",
        filters=filters,
        limit=1,
    )

    assert [item["source_id"] for item in exact] == [target_segment_id]
    assert [item["source_id"] for item in keyword] == [target_segment_id]
    assert exact[0]["conversation"]["id"] == target_conversation_id

    for exact_mode in (False, True):
        first_order = [
            item["source_id"]
            for item in lexical_search(
                catalog_engine,
                "Transcript needle" if exact_mode else "transcript needle",
                filters=SearchFilters(entity_type="audio_transcript"),
                exact=exact_mode,
                limit=10,
            )
        ]
        second_order = [
            item["source_id"]
            for item in lexical_search(
                catalog_engine,
                "Transcript needle" if exact_mode else "transcript needle",
                filters=SearchFilters(entity_type="audio_transcript"),
                exact=exact_mode,
                limit=10,
            )
        ]
        assert first_order == second_order


def test_transcription_failure_can_be_skipped_then_retried(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> None:
    _materialized_attachment(
        archive_layout,
        catalog_engine,
        content=b"retry fixture audio",
        filename="retry.m4a",
        media_type="audio/mp4",
        kind="audio_files",
    )
    failed_adapter = FakeTranscriber(fail=True)
    assert (
        build_audio_transcriptions(archive_layout, catalog_engine, adapter=failed_adapter).failed
        == 1
    )
    assert (
        build_audio_transcriptions(
            archive_layout,
            catalog_engine,
            adapter=failed_adapter,
            retry_failed=False,
        ).skipped_failed
        == 1
    )
    recovered = build_audio_transcriptions(
        archive_layout, catalog_engine, adapter=FakeTranscriber()
    )
    assert recovered.transcribed == 1
    with Session(catalog_engine) as session:
        row = session.scalar(select(AudioTranscription))
        assert row is not None
        assert row.status == "completed"
        assert row.attempt_count == 2


def test_enrichment_job_endpoints_reuse_checkpointed_workers_and_recover_stale_jobs(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> None:
    _materialized_attachment(
        archive_layout,
        catalog_engine,
        content=b"queued audio fixture",
        filename="queued.m4a",
        media_type="audio/mp4",
        kind="audio_files",
    )
    _materialized_attachment(
        archive_layout,
        catalog_engine,
        content=b"queued document fixture",
        filename="queued.txt",
        media_type="text/plain",
        kind="files",
    )
    app = create_app(archive_layout.root, security_disabled=True)
    app.state.audio_transcription_adapter = FakeTranscriber()
    app.state.attachment_text_extractor = FakeExtractor()
    app.state.nlp_classifier = FakeClassifier()
    with TestClient(app) as client:
        audio = client.post("/api/v1/enrichment/transcriptions/jobs", json={})
        attachment = client.post(
            "/api/v1/enrichment/attachment-text/jobs", json={"retry_failed": False}
        )
        nlp = client.post(
            "/api/v1/enrichment/nlp/jobs",
            json={"tasks": ["sentiment"], "batch_size": 4},
        )
        unsafe_batch = client.post(
            "/api/v1/enrichment/nlp/jobs",
            json={"tasks": ["sentiment"], "batch_size": 33},
        )
        assert audio.status_code == attachment.status_code == nlp.status_code == 202
        assert unsafe_batch.status_code == 422

        with Session(app.state.engine) as session, session.begin():
            session.add(
                Job(
                    id="fixture-still-queued-audio-job",
                    kind="audio_transcription",
                    status="queued",
                    stage="Fixture queued",
                    progress_current=0,
                    checkpoint_json={},
                    counters_json={},
                )
            )
            session.add(
                Job(
                    id="fixture-stale-running-nlp-job",
                    kind="nlp_annotations",
                    status="running",
                    stage="Fixture interrupted",
                    progress_current=0,
                    checkpoint_json={},
                    counters_json={},
                )
            )
        duplicate = client.post("/api/v1/enrichment/transcriptions/jobs", json={})
        recovered = client.post("/api/v1/enrichment/nlp/jobs", json={"tasks": ["sentiment"]})
        status_response = client.get("/api/v1/enrichment/status")

    assert duplicate.status_code == 409
    assert recovered.status_code == 202
    assert status_response.status_code == 200
    sentiment_status = status_response.json()["nlp_models"]["sentiment"]
    assert sentiment_status["device"] == "cpu"
    assert sentiment_status["quantization"] == "none"
    assert len(sentiment_status["inference_configuration_sha256"]) == 64
    assert sentiment_status["model_artifact"]["byte_verified"] is False
    assert sentiment_status["inference_backend"]["provider"] == ("injected-classifier-contract")
    assert set(sentiment_status["provenance_binding_counts"]) == {INJECTED_PROVENANCE_BINDING}
    assert (
        sum(sentiment_status["provenance_binding_counts"].values())
        == (status_response.json()["active_nlp_annotations"]["sentiment"])
    )
    with Session(app.state.engine) as session:
        launched = [
            session.get(Job, response.json()["job_id"])
            for response in (audio, attachment, nlp, recovered)
        ]
        stale = session.get(Job, "fixture-stale-running-nlp-job")
    assert all(job is not None and job.status == "completed" for job in launched)
    assert stale is not None and stale.status == "failed"
    assert stale.stage == "Interrupted enrichment job recovered"
    assert audio.json()["status_href"] == "/api/v1/enrichment/status"
    assert attachment.json()["retry_failed"] is False


def test_enrichment_status_uses_exact_covering_active_annotation_counts(
    archive_layout: ArchiveLayout,
    catalog_engine: Engine,
) -> None:
    with Session(catalog_engine) as session, session.begin():
        session.add_all(
            [
                Job(
                    id="status-count-job-a",
                    kind="nlp_annotations",
                    status="completed",
                    progress_current=0,
                    checkpoint_json={},
                    counters_json={},
                ),
                Job(
                    id="status-count-job-b",
                    kind="nlp_annotations",
                    status="completed",
                    progress_current=0,
                    checkpoint_json={},
                    counters_json={},
                ),
            ]
        )
        rows = (
            ("status-count-a", "sentiment", "status-count-job-a", "completed", True),
            ("status-count-b", "sentiment", "status-count-job-a", "completed", True),
            ("status-count-c", "sentiment", "status-count-job-b", "completed", True),
            ("status-count-d", "emotion", "status-count-job-a", "completed", True),
            ("status-count-inactive", "sentiment", "status-count-job-a", "completed", False),
            ("status-count-failed", "toxicity", "status-count-job-b", "failed", True),
        )
        session.add_all(
            [
                NlpAnnotation(
                    id=identifier,
                    target_type="message",
                    target_id=identifier,
                    task=task,
                    pipeline_version="fixture-v1",
                    model_name="fixture/model",
                    model_revision="fixture-revision",
                    input_sha256=f"{position:064x}",
                    status=status_value,
                    active=active,
                    probabilities_json={},
                    attribution_json={},
                    warnings_json={},
                    provenance_json={},
                    job_id=job_id,
                )
                for position, (identifier, task, job_id, status_value, active) in enumerate(rows, 1)
            ]
        )

    statement = enrichment_routes._active_nlp_annotation_counts_query()
    compiled = statement.compile(
        dialect=catalog_engine.dialect,
        compile_kwargs={"literal_binds": True},
    )
    compiled_sql = str(compiled)
    with catalog_engine.connect() as connection:
        plan = [
            str(row[-1]) for row in connection.execute(text(f"EXPLAIN QUERY PLAN {compiled_sql}"))
        ]
    assert "count(*)" in compiled_sql
    assert any(
        "USING COVERING INDEX ix_nlp_annotations_active_completed_task_job" in step for step in plan
    )
    assert not any("USE TEMP B-TREE FOR GROUP BY" in step for step in plan)

    app = create_app(archive_layout.root, security_disabled=True)
    with TestClient(app) as client:
        response = client.get("/api/v1/enrichment/status")

    assert response.status_code == 200, response.text
    assert response.json()["active_nlp_annotations"] == {"emotion": 1, "sentiment": 3}


def test_enrichment_status_rejects_corrupted_nlp_checkpoint_provenance(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> None:
    _materialized_attachment(
        archive_layout,
        catalog_engine,
        content=b"status provenance fixture",
        filename="status.txt",
        media_type="text/plain",
        kind="files",
    )
    build_attachment_text(archive_layout, catalog_engine, extractor=FakeExtractor())
    built = build_nlp_annotations(
        archive_layout,
        catalog_engine,
        classifier=FakeClassifier(),
        tasks=("sentiment",),
        batch_size=4,
    )
    with Session(catalog_engine) as session:
        job = session.get(Job, built.job_id)
        assert job is not None
        baseline_checkpoint = copy.deepcopy(job.checkpoint_json)
    catalog_engine.dispose()
    app = create_app(archive_layout.root, security_disabled=True)

    def assert_unavailable(payload: dict[str, object]) -> None:
        sentiment = payload["nlp_models"]["sentiment"]  # type: ignore[index]
        assert sentiment["provenance_status"] == "unavailable"
        assert "model_artifact" not in sentiment
        assert "inference_backend" not in sentiment
        assert "inference_configuration_sha256" not in sentiment

    with TestClient(app) as client:
        valid = client.get("/api/v1/enrichment/status")
        assert valid.status_code == 200
        valid_sentiment = valid.json()["nlp_models"]["sentiment"]
        assert valid_sentiment["provenance_status"] == "verified"
        assert len(valid_sentiment["inference_configuration_sha256"]) == 64

        configuration_sha256 = baseline_checkpoint["task_inference_configuration_sha256"][
            "sentiment"
        ]
        uncanonical = copy.deepcopy(baseline_checkpoint)
        uncanonical["inference_configurations"][configuration_sha256]["model_revision"] = (
            "corrupted-without-rehash"
        )
        with Session(app.state.engine) as session, session.begin():
            persisted = session.get(Job, built.job_id)
            assert persisted is not None
            persisted.checkpoint_json = uncanonical
        assert_unavailable(client.get("/api/v1/enrichment/status").json())

        for field, wrong_value in (
            ("task", "emotion"),
            ("model_name", "wrong/model"),
            ("model_revision", "wrong-revision"),
        ):
            wrong_identity = copy.deepcopy(baseline_checkpoint)
            payload = copy.deepcopy(
                wrong_identity["inference_configurations"][configuration_sha256]
            )
            payload[field] = wrong_value
            wrong_sha256 = _canonical_json_sha256(payload)
            wrong_identity["inference_configurations"] = {wrong_sha256: payload}
            wrong_identity["task_inference_configuration_sha256"]["sentiment"] = wrong_sha256
            with Session(app.state.engine) as session, session.begin():
                persisted = session.get(Job, built.job_id)
                assert persisted is not None
                persisted.checkpoint_json = wrong_identity
            assert_unavailable(client.get("/api/v1/enrichment/status").json())


def test_attachment_worker_is_inert_bounded_and_sandboxed(tmp_path: Path) -> None:
    html_source = tmp_path / "fixture.html"
    html_source.write_text("<h1>Safe</h1><script>secret()</script><p>Visible text</p>")
    payload = attachment_worker.run(html_source, "fixture.html", "text/html")
    assert payload["status"] == "completed"
    assert "Visible text" in str(payload["text"])
    assert "secret" not in str(payload["text"])

    credential = tmp_path / "private.pem"
    credential.write_text("-----BEGIN " + "PRIVATE KEY-----\nfixture\n")
    excluded = attachment_worker.run(credential, "private.pem", "text/plain")
    assert excluded["status"] == "excluded"
    assert excluded["index_eligible"] is False

    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w") as archive:
        archive.writestr("../escape.txt", "must not escape")
    traversal = tmp_path / "traversal.zip"
    traversal.write_bytes(archive_bytes.getvalue())
    rejected = attachment_worker.run(traversal, "traversal.zip", "application/zip")
    assert rejected["status"] == "unsupported"
    assert "traversal" in str(rejected["diagnostic"])

    compressed = io.BytesIO()
    with tarfile.open(fileobj=compressed, mode="w:gz") as archive:
        info = tarfile.TarInfo("oversized-ratio.txt")
        payload_bytes = b"0" * (2 * 1024 * 1024)
        info.size = len(payload_bytes)
        archive.addfile(info, io.BytesIO(payload_bytes))
    compression_bomb = tmp_path / "compression-bomb.tgz"
    compression_bomb.write_bytes(compressed.getvalue())
    bounded = attachment_worker.run(compression_bomb, compression_bomb.name, "application/gzip")
    assert bounded["status"] == "limit_exceeded"
    assert "100:1" in str(bounded["diagnostic"])

    malformed_pdf = tmp_path / "malformed.pdf"
    malformed_pdf.write_bytes(b"%PDF-1.4\nbroken fixture without an EOF marker\n")
    malformed = attachment_worker.run(malformed_pdf, malformed_pdf.name, "application/pdf")
    assert malformed["status"] == "unsupported"
    assert malformed["diagnostic"] == "unsupported or invalid PDF container"

    if sandbox_capability()["available"]:
        real = BubblewrapAttachmentExtractor().extract(
            html_source,
            filename="fixture.html",
            media_type="text/html",
        )
        assert real.status == "completed"
        assert real.index_eligible is True


def test_attachment_runner_suppresses_only_the_irrelevant_lancedb_fork_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "fixture.txt"
    source.write_text("fixture", encoding="utf-8")
    extractor = object.__new__(BubblewrapAttachmentExtractor)
    extractor.bwrap = "/usr/bin/bwrap"
    extractor.prlimit = "/usr/bin/prlimit"
    extractor.timeout_seconds = 7

    def fake_run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        warnings.warn(
            "lancedb fork support is experimental: fixture at-fork warning",
            RuntimeWarning,
            stacklevel=2,
        )
        warnings.warn("unrelated attachment warning", UserWarning, stacklevel=2)
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=(
                b'{"status":"completed","text":"fixture","truncated":false,"index_eligible":true}'
            ),
            stderr=b"",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = extractor.extract(source, filename=source.name, media_type="text/plain")

    assert result.status == "completed"
    assert [str(item.message) for item in caught] == ["unrelated attachment warning"]


def test_attachment_runner_bounds_and_hashes_private_child_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "fixture.txt"
    source.write_text("fixture", encoding="utf-8")
    extractor = object.__new__(BubblewrapAttachmentExtractor)
    extractor.bwrap = "/usr/bin/bwrap"
    extractor.prlimit = "/usr/bin/prlimit"
    extractor.timeout_seconds = 7
    private_diagnostic = b"private attachment excerpt that must never enter a durable job row"

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        stderr = kwargs["stderr"]
        assert hasattr(stderr, "write")
        stderr.write(private_diagnostic)  # type: ignore[union-attr]
        return subprocess.CompletedProcess(argv, 7)

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError) as raised:
        extractor.extract(source, filename=source.name, media_type="text/plain")

    message = str(raised.value)
    assert private_diagnostic.decode() not in message
    assert hashlib.sha256(private_diagnostic).hexdigest() in message


def test_attachment_extraction_rejects_a_symlinked_blob_shard(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> None:
    sha256 = _materialized_attachment(
        archive_layout,
        catalog_engine,
        content=b"private symlink fixture",
        filename="notes.txt",
        media_type="text/plain",
        kind="files",
    )
    blob_path = archive_layout.blob_path(sha256)
    relocated = archive_layout.root.parent / "relocated-private-attachment-shard"
    blob_path.parent.rename(relocated)
    blob_path.parent.symlink_to(relocated, target_is_directory=True)
    extractor = FakeExtractor()

    summary = build_attachment_text(archive_layout, catalog_engine, extractor=extractor)

    assert summary.failed == 1
    assert extractor.calls == []


def test_attachment_worker_enforces_archive_depth_entry_size_link_and_filetype_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def zip_payload(entries: list[tuple[str, bytes]]) -> bytes:
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            for name, value in entries:
                archive.writestr(name, value)
        return output.getvalue()

    nested = zip_payload([("deep.txt", b"deep text")])
    for depth in range(4):
        nested = zip_payload([(f"level-{depth}.zip", nested)])
    nested_path = tmp_path / "nested.zip"
    nested_path.write_bytes(nested)
    depth_result = attachment_worker.run(nested_path, nested_path.name, "application/zip")
    assert depth_result["status"] == "limit_exceeded"
    assert "depth" in str(depth_result["diagnostic"])

    link_archive = io.BytesIO()
    with zipfile.ZipFile(link_archive, "w") as archive:
        link = zipfile.ZipInfo("link.txt")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(link, "elsewhere")
    link_path = tmp_path / "link.zip"
    link_path.write_bytes(link_archive.getvalue())
    link_result = attachment_worker.run(link_path, link_path.name, "application/zip")
    assert link_result["status"] == "unsupported"
    assert "link, device, or FIFO" in str(link_result["diagnostic"])

    fifo_archive = io.BytesIO()
    with tarfile.open(fileobj=fifo_archive, mode="w") as archive:
        fifo = tarfile.TarInfo("pipe")
        fifo.type = tarfile.FIFOTYPE
        archive.addfile(fifo)
    fifo_path = tmp_path / "fifo.tar"
    fifo_path.write_bytes(fifo_archive.getvalue())
    fifo_result = attachment_worker.run(fifo_path, fifo_path.name, "application/x-tar")
    assert fifo_result["status"] == "unsupported"
    assert "link, device, or FIFO" in str(fifo_result["diagnostic"])

    with monkeypatch.context() as limits:
        limits.setattr(attachment_worker, "MAX_ARCHIVE_ENTRIES", 2)
        entry_path = tmp_path / "entries.zip"
        entry_path.write_bytes(zip_payload([(f"{index}.txt", b"text") for index in range(3)]))
        entry_result = attachment_worker.run(entry_path, entry_path.name, "application/zip")
        assert entry_result["status"] == "limit_exceeded"
        assert "2 entries" in str(entry_result["diagnostic"])

    with monkeypatch.context() as limits:
        limits.setattr(attachment_worker, "MAX_EXPANDED_BYTES", 8)
        expanded_path = tmp_path / "expanded.zip"
        expanded_path.write_bytes(zip_payload([("large.txt", b"more than eight bytes")]))
        expanded_result = attachment_worker.run(
            expanded_path, expanded_path.name, "application/zip"
        )
        assert expanded_result["status"] == "limit_exceeded"
        assert "expanded data" in str(expanded_result["diagnostic"])

    with monkeypatch.context() as limits:
        limits.setattr(attachment_worker, "MAX_PLAIN_INPUT_BYTES", 4)
        plain_path = tmp_path / "large.txt"
        plain_path.write_text("too large", encoding="utf-8")
        plain_result = attachment_worker.run(plain_path, plain_path.name, "text/plain")
        assert plain_result["status"] == "limit_exceeded"
        assert "plain-text input" in str(plain_result["diagnostic"])

    with monkeypatch.context() as limits:
        limits.setattr(attachment_worker, "MAX_OUTPUT_CHARACTERS", 5)
        output_path = tmp_path / "output.txt"
        output_path.write_text("0123456789", encoding="utf-8")
        output_result = attachment_worker.run(output_path, output_path.name, "text/plain")
        assert output_result["status"] == "completed"
        assert output_result["truncated"] is True
        assert output_result["text"] == "01234"

    for filename, content, expected in (
        ("macro.docm", b"not executed", "macro-bearing"),
        ("program.exe", b"MZfixture", "executable"),
        ("script.txt", b"#!/bin/sh\necho unsafe", "executable"),
    ):
        path = tmp_path / filename
        path.write_bytes(content)
        result = attachment_worker.run(path, filename, None)
        assert result["status"] == "excluded"
        assert expected in str(result["diagnostic"])

    delayed_key = tmp_path / "delayed-key.txt"
    delayed_key.write_bytes(b"x" * 300_000 + b"-----BEGIN " + b"PRIVATE KEY-----\nprivate")
    delayed_key_result = attachment_worker.run(delayed_key, delayed_key.name, "text/plain")
    assert delayed_key_result["status"] == "excluded"
    assert delayed_key_result["index_eligible"] is False

    macro_office = tmp_path / "macro.docx"
    macro_office.write_bytes(
        zip_payload(
            [
                ("word/document.xml", b"<document><p>visible</p></document>"),
                ("word/vbaProject.bin", b"macro"),
            ]
        )
    )
    office_result = attachment_worker.run(
        macro_office,
        macro_office.name,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    assert office_result["status"] == "excluded"
    assert "macro-bearing" in str(office_result["diagnostic"])


def test_attachment_and_nlp_pipelines_are_versioned_indexed_and_conflict_gated(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> None:
    _materialized_attachment(
        archive_layout,
        catalog_engine,
        content=b"fixture document bytes",
        filename="notes.txt",
        media_type="text/plain",
        kind="files",
    )
    extractor = FakeExtractor()
    first_extraction = build_attachment_text(archive_layout, catalog_engine, extractor=extractor)
    second_extraction = build_attachment_text(archive_layout, catalog_engine, extractor=extractor)
    assert first_extraction.extracted == 1
    assert second_extraction.reused == 1
    assert len(extractor.calls) == 1

    classifier = FakeClassifier()
    first = build_nlp_annotations(
        archive_layout,
        catalog_engine,
        classifier=classifier,
        tasks=("sentiment", "emotion", "toxicity"),
        batch_size=4,
    )
    calls_after_first = len(classifier.calls)
    second = build_nlp_annotations(
        archive_layout,
        catalog_engine,
        classifier=classifier,
        tasks=("sentiment", "emotion", "toxicity"),
        batch_size=4,
    )
    assert first.annotated == 3
    assert second.reused == 3
    assert len(classifier.calls) == calls_after_first
    with Session(catalog_engine) as session:
        extraction = session.scalar(select(AttachmentTextExtraction))
        annotations = list(session.scalars(select(NlpAnnotation).order_by(NlpAnnotation.task)))
        assert extraction is not None
        assert extraction.sandboxed is True
        assert len(annotations) == 3
        assert all(
            annotation.active and annotation.status == "completed" for annotation in annotations
        )
        assert all(annotation.probabilities_json["labels"] for annotation in annotations)
        assert all(annotation.model_revision for annotation in annotations)
        assert all(annotation.warnings_json["items"] for annotation in annotations)

    before = conflict_aggregation_status(catalog_engine)
    assert before["aggregation_enabled"] is False
    calibration = record_conflict_calibration(
        archive_layout,
        catalog_engine,
    )
    assert calibration.aggregation_enabled is False
    with Session(catalog_engine) as session:
        stored = session.scalar(select(ConflictCalibration))
        assert stored is not None
        assert stored.status == "gates_failed"

    lexical = lexical_search(
        catalog_engine,
        "attachment needle",
        filters=SearchFilters(entity_type="attachment_text"),
    )
    assert [item["entity_type"] for item in lexical] == ["attachment_text"]

    catalog_engine.dispose()
    app = create_app(archive_layout.root, security_disabled=True)
    with TestClient(app) as client:
        nlp_response = client.get("/api/v1/enrichment/nlp")
    assert nlp_response.status_code == 200
    nlp_item = nlp_response.json()["items"][0]
    assert nlp_item["probability_activation"] in {"sigmoid", "softmax"}
    assert nlp_item["probabilities"]
    assert all(isinstance(value, float) for value in nlp_item["probabilities"].values())
    assert "labels" not in nlp_item["probabilities"]
    assert isinstance(nlp_item["warnings"], list)


def test_nlp_legacy_rows_are_adopted_once_and_remain_all_reuse(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> None:
    _materialized_attachment(
        archive_layout,
        catalog_engine,
        content=b"legacy provenance fixture",
        filename="legacy.txt",
        media_type="text/plain",
        kind="files",
    )
    build_attachment_text(archive_layout, catalog_engine, extractor=FakeExtractor())
    initial_classifier = FakeClassifier()
    initial = build_nlp_annotations(
        archive_layout,
        catalog_engine,
        classifier=initial_classifier,
        tasks=("sentiment",),
        batch_size=4,
    )
    assert initial.annotated == initial.target_count == 1

    with Session(catalog_engine) as session, session.begin():
        rows = list(session.scalars(select(NlpAnnotation)))
        before = {row.id: (row.predicted_label, dict(row.probabilities_json)) for row in rows}
        for row in rows:
            provenance = dict(row.provenance_json)
            assert provenance[NLP_PROVENANCE_KEY]["binding_kind"] == (INJECTED_PROVENANCE_BINDING)
            provenance.pop(NLP_PROVENANCE_KEY)
            row.provenance_json = provenance

    legacy_classifier = FakeClassifier()
    adopted = build_nlp_annotations(
        archive_layout,
        catalog_engine,
        classifier=legacy_classifier,
        tasks=("sentiment",),
        batch_size=4,
    )
    assert adopted.annotated == adopted.failed == 0
    assert adopted.reused == adopted.target_count == 1
    assert legacy_classifier.calls == []

    with Session(catalog_engine) as session:
        adopted_rows = list(session.scalars(select(NlpAnnotation)))
        adopted_job = session.get(Job, adopted.job_id)
        rollup = session.scalar(select(NlpHourlyRollup).where(NlpHourlyRollup.active.is_(True)))
    assert {
        row.id: (row.predicted_label, dict(row.probabilities_json)) for row in adopted_rows
    } == before
    assert all(row.job_id == adopted.job_id for row in adopted_rows)
    assert all(
        row.provenance_json[NLP_PROVENANCE_KEY]["binding_kind"] == LEGACY_PROVENANCE_BINDING
        for row in adopted_rows
    )
    assert adopted_job is not None
    configuration_sha256 = adopted_job.checkpoint_json["task_inference_configuration_sha256"][
        "sentiment"
    ]
    assert configuration_sha256 in adopted_job.checkpoint_json["inference_configurations"]
    assert all(
        row.provenance_json[NLP_PROVENANCE_KEY]["sha256"] == configuration_sha256
        for row in adopted_rows
    )
    assert rollup is not None
    assert rollup.configuration_json["inference_configuration_sha256"] == (configuration_sha256)
    assert rollup.configuration_json["annotation_binding_kinds"] == [LEGACY_PROVENANCE_BINDING]

    repeated = build_nlp_annotations(
        archive_layout,
        catalog_engine,
        classifier=legacy_classifier,
        tasks=("sentiment",),
        batch_size=4,
    )
    assert repeated.reused == repeated.target_count == 1
    assert repeated.annotated == repeated.failed == 0
    assert legacy_classifier.calls == []
    with Session(catalog_engine) as session:
        repeated_row = session.scalar(select(NlpAnnotation))
    assert repeated_row is not None
    assert repeated_row.provenance_json[NLP_PROVENANCE_KEY]["binding_kind"] == (
        LEGACY_PROVENANCE_BINDING
    )


@pytest.mark.parametrize(
    "binding_kind",
    (
        DIRECT_PROVENANCE_BINDING,
        LEGACY_PROVENANCE_BINDING,
        INJECTED_PROVENANCE_BINDING,
    ),
)
def test_nlp_reuse_preserves_only_valid_existing_bindings(binding_kind: str) -> None:
    configuration_sha256 = "a" * 64
    provenance = {
        "fixture": True,
        NLP_PROVENANCE_KEY: {
            "sha256": configuration_sha256,
            "binding_kind": binding_kind,
        },
    }

    assert (
        _reuse_provenance_for_configuration(
            provenance,
            configuration_sha256=configuration_sha256,
        )
        == provenance
    )


def test_nlp_malformed_present_provenance_fails_before_deactivation_or_classification(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> None:
    _materialized_attachment(
        archive_layout,
        catalog_engine,
        content=b"malformed provenance fixture",
        filename="malformed.txt",
        media_type="text/plain",
        kind="files",
    )
    build_attachment_text(archive_layout, catalog_engine, extractor=FakeExtractor())
    build_nlp_annotations(
        archive_layout,
        catalog_engine,
        classifier=FakeClassifier(),
        tasks=("sentiment",),
        batch_size=4,
    )
    with Session(catalog_engine) as session:
        original = session.scalar(select(NlpAnnotation))
        assert original is not None
        original_id = original.id
        valid_provenance = copy.deepcopy(original.provenance_json)
    configuration_sha256 = valid_provenance[NLP_PROVENANCE_KEY]["sha256"]
    valid_binding = valid_provenance[NLP_PROVENANCE_KEY]["binding_kind"]
    malformed_values: tuple[tuple[str, object], ...] = (
        ("present-null", None),
        ("present-non-object", "not-an-object"),
        ("missing-sha", {"binding_kind": valid_binding}),
        ("null-sha", {"sha256": None, "binding_kind": valid_binding}),
        ("non-string-sha", {"sha256": 7, "binding_kind": valid_binding}),
        ("empty-sha", {"sha256": "", "binding_kind": valid_binding}),
        ("mismatched-sha", {"sha256": "f" * 64, "binding_kind": valid_binding}),
        ("missing-binding", {"sha256": configuration_sha256}),
        ("null-binding", {"sha256": configuration_sha256, "binding_kind": None}),
        ("non-string-binding", {"sha256": configuration_sha256, "binding_kind": 7}),
        ("empty-binding", {"sha256": configuration_sha256, "binding_kind": ""}),
        (
            "invalid-binding",
            {"sha256": configuration_sha256, "binding_kind": "invented-binding"},
        ),
    )

    for label, malformed in malformed_values:
        malformed_provenance = copy.deepcopy(valid_provenance)
        malformed_provenance[NLP_PROVENANCE_KEY] = malformed
        with Session(catalog_engine) as session, session.begin():
            row = session.get(NlpAnnotation, original_id)
            assert row is not None
            row.provenance_json = malformed_provenance
            row.active = True
        classifier = FakeClassifier()
        with pytest.raises(RuntimeError, match="malformed or incompatible"):
            build_nlp_annotations(
                archive_layout,
                catalog_engine,
                classifier=classifier,
                tasks=("sentiment",),
                batch_size=4,
            )
        assert classifier.calls == [], label
        with Session(catalog_engine) as session:
            unchanged = session.get(NlpAnnotation, original_id)
            assert unchanged is not None
            assert unchanged.active is True, label
            assert unchanged.provenance_json == malformed_provenance, label


def test_nlp_configuration_mismatch_fails_before_active_rows_are_retired(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> None:
    _materialized_attachment(
        archive_layout,
        catalog_engine,
        content=b"configuration mismatch fixture",
        filename="mismatch.txt",
        media_type="text/plain",
        kind="files",
    )
    build_attachment_text(archive_layout, catalog_engine, extractor=FakeExtractor())
    build_nlp_annotations(
        archive_layout,
        catalog_engine,
        classifier=FakeClassifier(),
        tasks=("sentiment",),
        batch_size=4,
    )
    with Session(catalog_engine) as session:
        before = session.scalar(select(NlpAnnotation))
        assert before is not None
        before_id = before.id
        before_provenance = dict(before.provenance_json)

    class AlternateFakeClassifier(FakeClassifier):
        pass

    alternate = AlternateFakeClassifier()
    with pytest.raises(RuntimeError, match="explicit versioned recomputation"):
        build_nlp_annotations(
            archive_layout,
            catalog_engine,
            classifier=alternate,
            tasks=("sentiment",),
            batch_size=4,
        )
    assert alternate.calls == []
    with Session(catalog_engine) as session:
        after = session.scalar(select(NlpAnnotation))
        failed_job = session.scalar(
            select(Job)
            .where(Job.kind == "nlp_annotations", Job.status == "failed")
            .order_by(Job.created_at.desc())
        )
    assert after is not None and after.id == before_id and after.active is True
    assert after.provenance_json == before_provenance
    assert failed_job is not None


def test_mocked_transformers_legacy_reuse_and_artifact_mismatch_are_fail_safe(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> None:
    _materialized_attachment(
        archive_layout,
        catalog_engine,
        content=b"mocked transformers provenance fixture",
        filename="mocked-transformers.txt",
        media_type="text/plain",
        kind="files",
    )
    build_attachment_text(archive_layout, catalog_engine, extractor=FakeExtractor())
    initial = build_nlp_annotations(
        archive_layout,
        catalog_engine,
        classifier=FakeClassifier(),
        tasks=("sentiment",),
        batch_size=4,
    )
    with Session(catalog_engine) as session, session.begin():
        row = session.scalar(select(NlpAnnotation))
        job = session.get(Job, initial.job_id)
        assert row is not None and job is not None
        row_id = row.id
        provenance = dict(row.provenance_json)
        provenance.pop(NLP_PROVENANCE_KEY)
        row.provenance_json = provenance
        source_sha = job.checkpoint_json["task_inference_configuration_sha256"]["sentiment"]
        source_payload = copy.deepcopy(job.checkpoint_json["inference_configurations"][source_sha])

    def production_configuration(artifact_sha256: str) -> NlpInferenceConfiguration:
        payload = copy.deepcopy(source_payload)
        payload["model_artifact"] = {
            "schema": "hf-logical-snapshot-v1",
            "sha256": artifact_sha256,
            "file_count": 3,
            "total_bytes": 128,
        }
        implementation = payload["implementation"]
        implementation["schema"] = "transformers-sequence-classification-v1"
        implementation["provider"] = "transformers-auto-sequence-classification"
        implementation["packages"] = {
            "transformers": "fixture",
            "torch": "fixture",
            "tokenizers": "fixture",
            "safetensors": "fixture",
        }
        payload["implementation_sha256"] = _canonical_json_sha256(implementation)
        return NlpInferenceConfiguration(
            sha256=_canonical_json_sha256(payload),
            payload=payload,
            direct_binding_kind=DIRECT_PROVENANCE_BINDING,
        )

    class MockedTransformersClassifier(TransformersClassifier):
        def __init__(self, configuration: NlpInferenceConfiguration) -> None:
            self.configuration = configuration
            self.prepare_calls = 0
            self.classify_calls = 0

        def prepare(self, _spec: NlpModelSpec) -> None:
            self.prepare_calls += 1

        def inference_configuration(self, _spec: NlpModelSpec) -> NlpInferenceConfiguration:
            return self.configuration

        def classify(
            self, _spec: NlpModelSpec, _texts: Sequence[str]
        ) -> list[ClassificationResult]:
            self.classify_calls += 1
            raise AssertionError("legacy exact-ID reuse must not classify")

        def release(self, _spec: NlpModelSpec) -> None:
            return None

    adopted_classifier = MockedTransformersClassifier(production_configuration("a" * 64))
    adopted = build_nlp_annotations(
        archive_layout,
        catalog_engine,
        classifier=adopted_classifier,
        tasks=("sentiment",),
        batch_size=4,
    )
    assert adopted.reused == adopted.target_count == 1
    assert adopted.annotated == adopted.failed == 0
    assert adopted_classifier.prepare_calls == 1
    assert adopted_classifier.classify_calls == 0
    with Session(catalog_engine) as session:
        adopted_row = session.get(NlpAnnotation, row_id)
        assert adopted_row is not None
        adopted_provenance = copy.deepcopy(adopted_row.provenance_json)
    assert adopted_provenance[NLP_PROVENANCE_KEY]["binding_kind"] == (LEGACY_PROVENANCE_BINDING)

    mismatch_classifier = MockedTransformersClassifier(production_configuration("b" * 64))
    with pytest.raises(RuntimeError, match="explicit versioned recomputation"):
        build_nlp_annotations(
            archive_layout,
            catalog_engine,
            classifier=mismatch_classifier,
            tasks=("sentiment",),
            batch_size=4,
        )
    assert mismatch_classifier.prepare_calls == 1
    assert mismatch_classifier.classify_calls == 0
    with Session(catalog_engine) as session:
        unchanged = session.get(NlpAnnotation, row_id)
    assert unchanged is not None and unchanged.active is True
    assert unchanged.provenance_json == adopted_provenance


def test_nlp_annotation_pipeline_reuses_exact_model_inputs(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> None:
    duplicate_count = 6
    blobs = [
        Blob(
            sha256=hashlib.sha256(f"duplicate-nlp-{index}".encode()).hexdigest(),
            size_bytes=0,
            media_type="text/plain",
            extension=".txt",
            state="materialized",
        )
        for index in range(duplicate_count)
    ]
    empty_blob = Blob(
        sha256=hashlib.sha256(b"empty-nlp-candidate").hexdigest(),
        size_bytes=0,
        media_type="text/plain",
        extension=".txt",
        state="materialized",
    )
    with Session(catalog_engine) as session, session.begin():
        session.add_all([*blobs, empty_blob])
        session.flush()
        session.add_all(
            [
                AttachmentTextExtraction(
                    id=f"duplicate-nlp-extraction-{index}",
                    blob_sha256=blob.sha256,
                    pipeline_version="attachment-text-v1",
                    status="completed",
                    extractor_name="fixture",
                    extractor_revision="fixture-v1",
                    text_display="The exact same classifier input.",
                    text_search="the exact same classifier input.",
                    character_count=32,
                    truncated=False,
                    index_eligible=True,
                    sandboxed=True,
                    sandbox_status="available",
                    attempt_count=1,
                    limits_json={},
                    provenance_json={},
                    warnings_json={},
                )
                for index, blob in enumerate(blobs)
            ]
        )
        session.add(
            AttachmentTextExtraction(
                id="empty-nlp-extraction",
                blob_sha256=empty_blob.sha256,
                pipeline_version="attachment-text-v1",
                status="completed",
                extractor_name="fixture",
                extractor_revision="fixture-v1",
                text_display="",
                text_search="",
                character_count=0,
                truncated=False,
                index_eligible=True,
                sandboxed=True,
                sandbox_status="available",
                attempt_count=1,
                limits_json={},
                provenance_json={},
                warnings_json={},
            )
        )

    classifier = FakeClassifier()
    first = build_nlp_annotations(
        archive_layout,
        catalog_engine,
        classifier=classifier,
        tasks=("sentiment",),
        batch_size=4,
        candidate_chunk_size=4,
    )
    assert first.annotated == duplicate_count
    assert sum(len(texts) for _task, texts in classifier.calls) == 1
    # Four targets share one inference in the first page. The second page reuses the completed
    # exact-input result from that page without another model call.
    assert len(classifier.calls) == 1

    with Session(catalog_engine) as session:
        annotations = list(session.scalars(select(NlpAnnotation).order_by(NlpAnnotation.id)))
    assert len(annotations) == duplicate_count
    reused = [row for row in annotations if "inference_reuse" in row.provenance_json]
    assert len(reused) == duplicate_count - 1
    assert {row.provenance_json["inference_reuse"]["source"] for row in reused} == {
        "same_classifier_batch",
        "completed_annotation",
    }


def test_nlp_model_preparation_failure_is_job_level_and_preserves_active_product(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> None:
    spec = MODEL_SPECS["sentiment"]
    existing_id = "existing-active-nlp-annotation"
    candidate_blob_sha256 = _materialized_attachment(
        archive_layout,
        catalog_engine,
        content=b"model preparation candidate",
        filename="preparation.txt",
        media_type="text/plain",
        kind="files",
    )
    with Session(catalog_engine) as session, session.begin():
        session.add(
            AttachmentTextExtraction(
                id="model-preparation-candidate",
                blob_sha256=candidate_blob_sha256,
                pipeline_version="attachment-text-v1",
                status="completed",
                extractor_name="fixture",
                extractor_revision="fixture-v1",
                text_display="Model preparation candidate",
                text_search="model preparation candidate",
                character_count=27,
                truncated=False,
                index_eligible=True,
                sandboxed=True,
                sandbox_status="available",
                attempt_count=1,
                limits_json={},
                provenance_json={},
                warnings_json={},
            )
        )
        session.add(
            NlpAnnotation(
                id=existing_id,
                target_type="message",
                target_id="existing-message",
                task="sentiment",
                pipeline_version="derived-nlp-v1",
                model_name=spec.model_name,
                model_revision=spec.model_revision,
                input_sha256="0" * 64,
                status="completed",
                active=True,
                predicted_label="positive",
                confidence=0.7,
                probabilities_json={
                    "activation": "softmax",
                    "labels": {"negative": 0.1, "neutral": 0.2, "positive": 0.7},
                },
                attribution_json={},
                warnings_json={"items": []},
                provenance_json={},
            )
        )

    class FailingPrepareClassifier(TransformersClassifier):
        def prepare(self, _spec: NlpModelSpec) -> None:
            raise RuntimeError("fixture model preparation failure")

        def classify(
            self, _spec: NlpModelSpec, _texts: Sequence[str]
        ) -> list[ClassificationResult]:
            raise AssertionError("classification must not start after preparation fails")

    with pytest.raises(RuntimeError, match="fixture model preparation failure"):
        build_nlp_annotations(
            archive_layout,
            catalog_engine,
            classifier=FailingPrepareClassifier(archive_layout.models / "fixture-nlp"),
            tasks=("sentiment",),
            batch_size=4,
        )

    with Session(catalog_engine) as session:
        existing = session.get(NlpAnnotation, existing_id)
        failed_job = session.scalar(
            select(Job).where(Job.kind == "nlp_annotations").order_by(Job.created_at.desc())
        )
    assert existing is not None and existing.active is True
    assert failed_job is not None
    assert failed_job.status == "failed"
    assert failed_job.stage == "Local NLP annotation job interrupted"


def test_nlp_annotation_pipeline_bounds_candidate_pages_and_batches_database_work(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> None:
    synthetic_count = 1_200
    blobs = [
        Blob(
            sha256=hashlib.sha256(f"synthetic-nlp-{index}".encode()).hexdigest(),
            size_bytes=0,
            media_type="text/plain",
            extension=".txt",
            state="materialized",
        )
        for index in range(synthetic_count)
    ]
    with Session(catalog_engine) as session, session.begin():
        session.add_all(blobs)
        session.flush()
        session.add_all(
            [
                AttachmentTextExtraction(
                    id=f"synthetic-nlp-extraction-{index:04d}",
                    blob_sha256=blob.sha256,
                    pipeline_version="attachment-text-v1",
                    status="completed",
                    extractor_name="fixture",
                    extractor_revision="fixture-v1",
                    text_display=f"Synthetic bounded NLP candidate {index}",
                    text_search=f"synthetic bounded nlp candidate {index}",
                    character_count=36,
                    truncated=False,
                    index_eligible=True,
                    sandboxed=True,
                    sandbox_status="available",
                    attempt_count=1,
                    limits_json={},
                    provenance_json={},
                    warnings_json={},
                )
                for index, blob in enumerate(blobs)
            ]
        )

    statements: list[tuple[str, bool]] = []

    def record_statement(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        executemany: bool,
    ) -> None:
        if "nlp_annotations" in statement:
            statements.append((statement, executemany))

    classifier = FakeClassifier()
    event.listen(catalog_engine, "before_cursor_execute", record_statement)
    try:
        first = build_nlp_annotations(
            archive_layout,
            catalog_engine,
            classifier=classifier,
            tasks=("sentiment",),
            batch_size=23,
            candidate_chunk_size=127,
        )
        calls_after_first = len(classifier.calls)
        second = build_nlp_annotations(
            archive_layout,
            catalog_engine,
            classifier=classifier,
            tasks=("sentiment",),
            batch_size=23,
            candidate_chunk_size=127,
        )
    finally:
        event.remove(catalog_engine, "before_cursor_execute", record_statement)

    assert first.target_count == first.annotated == synthetic_count
    assert second.target_count == second.reused == synthetic_count
    full_chunks, final_chunk = divmod(synthetic_count, 127)
    expected_classifier_calls = (full_chunks * math.ceil(127 / 23)) + math.ceil(final_chunk / 23)
    assert len(classifier.calls) == calls_after_first == expected_classifier_calls
    assert max(len(texts) for _task, texts in classifier.calls) <= 23
    lookup_statements = [
        statement for statement, _many in statements if "nlp_annotations.id IN" in statement
    ]
    assert len(lookup_statements) == 2 * math.ceil(synthetic_count / 127)
    reuse_queries = [
        statement
        for statement, _many in statements
        if "WITH requested_inputs(input_sha256) AS" in statement
    ]
    assert reuse_queries
    assert all("LIMIT ? OFFSET ?" in statement for statement in reuse_queries)
    assert all(
        "GROUP BY nlp_annotations.input_sha256" not in statement for statement in reuse_queries
    )
    # This stays proportional to bounded pages, not candidates. The previous implementation
    # issued several thousand annotation SELECT/UPDATE/INSERT statements for this fixture.
    assert len(statements) < 100
    assert any(
        many and "INSERT INTO nlp_annotations" in statement for statement, many in statements
    )
    assert any(many and "UPDATE nlp_annotations" in statement for statement, many in statements)

    with Session(catalog_engine) as session:
        active_count = session.scalar(
            select(func.count(NlpAnnotation.id)).where(NlpAnnotation.active.is_(True))
        )
        annotation = session.scalar(select(NlpAnnotation).limit(1))
        latest_job = session.scalar(
            select(Job).where(Job.kind == "nlp_annotations").order_by(Job.created_at.desc())
        )
    assert active_count == synthetic_count
    assert annotation is not None
    assert annotation.provenance_json["execution"] == {
        "device": "cpu",
        "quantization": "none",
    }
    assert annotation.provenance_json["input_truncated_by_character_guard"] is False
    assert annotation.provenance_json["tokenizer_truncation_enabled"] is True
    assert annotation.provenance_json["tokenizer_max_length"] == 512
    assert latest_job is not None
    assert latest_job.checkpoint_json["execution_configurations"]["sentiment"] == {
        "device": "cpu",
        "quantization": "none",
    }
    assert latest_job.checkpoint_json["batch_size"] == 23
    assert latest_job.checkpoint_json["candidate_chunk_size"] == 127


def test_conflict_calibration_derives_metrics_from_stratified_held_out_predictions(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> None:
    labels = tuple(CONFLICT_LABELS)
    splits = ("train", "validation", "test")
    conversation_ids = {
        (split, label): _fixture_conversation_for_split(
            f"calibration-conversation-{split_index}-{label_index}", split
        )
        for split_index, split in enumerate(splits)
        for label_index, label in enumerate(labels)
    }
    with Session(catalog_engine) as session, session.begin():
        for split in splits:
            for label in labels:
                conversation_id = conversation_ids[(split, label)]
                session.add(
                    Conversation(
                        id=conversation_id,
                        conversation_key=conversation_id,
                        title_raw=conversation_id,
                        title_display=conversation_id,
                        title_search=conversation_id,
                        conversation_type="direct",
                        primary_category="fixture",
                    )
                )
        session.flush()
        conflict_labels: list[ConflictLabelRow] = []
        for index in range(500):
            split_index = index % len(splits)
            label_index = (index // len(splits)) % len(labels)
            split = splits[split_index]
            label = labels[label_index]
            conversation_id = conversation_ids[(split, label)]
            window_id = f"calibration-window-{index:04d}"
            session.add(
                SearchWindow(
                    id=window_id,
                    window_version="fixture-v1",
                    entity_type="message_window",
                    conversation_id=conversation_id,
                    turn_count=1,
                    approx_token_count=4,
                    text_display=f"expected:{label}",
                    text_search=f"expected {label}",
                    participants_search="",
                    source_ids_json={},
                    build_generation="fixture-generation",
                )
            )
            conflict_labels.append(
                ConflictLabelRow(
                    id=f"calibration-label-{index:04d}",
                    window_id=window_id,
                    label=label,
                    evaluation_split=split,
                )
            )
        session.flush()
        session.add_all(conflict_labels)

    scorer = LabelEchoConflictScorer()
    calibration = record_conflict_calibration(
        archive_layout,
        catalog_engine,
        scorer_factory=lambda _spec: scorer,
    )
    assert calibration.status == "calibrated"
    assert calibration.sample_count == 500
    assert calibration.direct_conflict_precision == pytest.approx(1.0)
    assert calibration.macro_f1 == pytest.approx(1.0)
    assert calibration.aggregation_enabled is True
    assert len(scorer.calls) == 2
    assert all(len(batch) <= DEFAULT_CONFLICT_SPEC.max_candidates for batch in scorer.calls)
    assert calibration.metrics_json["evaluation_source"] == "held_out_test_predictions"
    assert calibration.metrics_json["conversation_disjoint"] is True
    assert calibration.metrics_json["stratified"] is True

    aggregation = conflict_aggregation_status(catalog_engine)
    assert aggregation["status"] == "enabled"
    assert aggregation["sample_count"] == aggregation["label_count"] == 500
    assert aggregation["reason"] is None
    assert aggregation["relationship_ranking_enabled"] is False

    app = create_app(archive_layout.root, security_disabled=True)
    app.state.conflict_scorer_factory = lambda _spec: LabelEchoConflictScorer()
    with TestClient(app) as client:
        caller_metrics = client.post(
            "/api/v1/enrichment/conflict/calibrations",
            json={"direct_conflict_precision": 1.0, "macro_f1": 1.0},
        )
        derived = client.post("/api/v1/enrichment/conflict/calibrations", json={})
    assert caller_metrics.status_code == 422
    assert derived.status_code == 201, derived.text
    assert derived.json()["direct_conflict_precision"] == pytest.approx(1.0)
    assert derived.json()["macro_f1"] == pytest.approx(1.0)

    with Session(catalog_engine) as session, session.begin():
        tampered = session.get(ConflictLabelRow, "calibration-label-0000")
        assert tampered is not None
        assert tampered.evaluation_split == "train"
        tampered.evaluation_split = "validation"
    forbidden_scorer_calls = 0

    def forbidden_scorer(_spec: ConflictModelSpec) -> LabelEchoConflictScorer:
        nonlocal forbidden_scorer_calls
        forbidden_scorer_calls += 1
        raise AssertionError("Split validation must run before scorer loading")

    app.state.conflict_scorer_factory = forbidden_scorer
    app.state.conflict_scorer_cache = {}
    with TestClient(app) as client:
        invalid_status = client.get("/api/v1/enrichment/conflict/review/status")
        invalid_calibration = client.post("/api/v1/enrichment/conflict/calibrations", json={})
    blocker_codes = {blocker["code"] for blocker in invalid_status.json()["validation"]["blockers"]}
    assert {"split_assignment_mismatch", "conversation_split_leakage"} <= blocker_codes
    assert invalid_calibration.status_code == 409
    assert invalid_calibration.json()["detail"]["code"] == "conflict_calibration_ineligible"
    assert forbidden_scorer_calls == 0
    with Session(catalog_engine) as session:
        assert session.scalar(select(func.count(ConflictCalibration.id))) == 2


def test_nlp_hourly_rollups_are_versioned_active_and_cover_all_directions(
    archive_layout: ArchiveLayout,
    catalog_engine: Engine,
) -> None:
    fixtures = Path(__file__).parent / "fixtures"
    shutil.copytree(fixtures / "messages", archive_layout.root, dirs_exist_ok=True)
    shutil.copytree(fixtures / "activity", archive_layout.root / "activity-package")
    scan_archive(archive_layout, catalog_engine, hash_files=True)
    import_messages(archive_layout, catalog_engine)
    import_activities(archive_layout, catalog_engine)
    _materialized_attachment(
        archive_layout,
        catalog_engine,
        content=b"rollup audio fixture",
        filename="rollup.m4a",
        media_type="audio/mp4",
        kind="audio_files",
    )
    _materialized_attachment(
        archive_layout,
        catalog_engine,
        content=b"rollup attachment fixture",
        filename="rollup.txt",
        media_type="text/plain",
        kind="files",
    )
    build_audio_transcriptions(archive_layout, catalog_engine, adapter=FakeTranscriber())
    build_attachment_text(archive_layout, catalog_engine, extractor=FakeExtractor())

    classifier = FakeClassifier()
    first = build_nlp_annotations(
        archive_layout,
        catalog_engine,
        classifier=classifier,
        tasks=("sentiment",),
        batch_size=8,
    )
    second = build_nlp_annotations(
        archive_layout,
        catalog_engine,
        classifier=classifier,
        tasks=("sentiment",),
        batch_size=8,
    )

    with Session(catalog_engine) as session:
        rollups = list(
            session.scalars(
                select(NlpHourlyRollup)
                .where(NlpHourlyRollup.active.is_(True))
                .order_by(NlpHourlyRollup.direction, NlpHourlyRollup.hour_start_ms)
            )
        )
        total_rows = len(list(session.scalars(select(NlpHourlyRollup))))
        expected_samples = len(
            [
                row
                for row in session.scalars(
                    select(NlpAnnotation).where(
                        NlpAnnotation.task == "sentiment",
                        NlpAnnotation.active.is_(True),
                    )
                )
                if isinstance(row.attribution_json.get("timestamp_ms"), int)
            ]
        )
    assert first.rollup_count == len(rollups)
    assert second.rollup_count == len(rollups)
    assert total_rows == len(rollups)
    assert {row.direction for row in rollups} >= {
        "owner",
        "other",
        "activity",
        "audio",
        "attachment",
    }
    assert sum(row.sample_count for row in rollups) == expected_samples
    assert all(row.hour_start_ms % 3_600_000 == 0 for row in rollups)
    assert all(row.pipeline_version == "derived-nlp-v1" for row in rollups)
    assert all(row.configuration_sha256 for row in rollups)
    assert all(row.configuration_json["time_bucket"] == "UTC hour" for row in rollups)
    for row in rollups:
        assert row.predicted_label == "positive"
        assert row.confidence_sum == pytest.approx(0.7 * row.sample_count)
        assert row.probability_sums_json["labels"]["positive"] == pytest.approx(
            0.7 * row.sample_count
        )

    expected_audio_samples = sum(row.sample_count for row in rollups if row.direction == "audio")
    catalog_engine.dispose()
    app = create_app(archive_layout.root, security_disabled=True)
    with TestClient(app) as client:
        response = client.get(
            "/api/v1/enrichment/nlp/hourly",
            params={"task": "sentiment", "direction": "audio"},
        )
        status_response = client.get("/api/v1/enrichment/status")
    assert response.status_code == 200
    assert sum(row["sample_count"] for row in response.json()["items"]) == expected_audio_samples
    assert status_response.json()["active_nlp_hourly_rollups"]["sentiment"] == len(rollups)


def test_derived_text_is_semantically_indexed_and_api_exposes_provenance(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> None:
    _materialized_attachment(
        archive_layout,
        catalog_engine,
        content=b"semantic fixture audio",
        filename="semantic.m4a",
        media_type="audio/mp4",
        kind="audio_files",
    )
    _materialized_attachment(
        archive_layout,
        catalog_engine,
        content=b"semantic fixture document",
        filename="semantic.txt",
        media_type="text/plain",
        kind="files",
    )
    build_audio_transcriptions(archive_layout, catalog_engine, adapter=FakeTranscriber())
    build_attachment_text(archive_layout, catalog_engine, extractor=FakeExtractor())
    built = build_semantic_index(
        archive_layout,
        catalog_engine,
        embedder=FakeEmbedder(),
        batch_size=4,
    )
    assert built.window_count >= 3
    with Session(catalog_engine) as session:
        derived_types = set(
            session.scalars(
                select(SearchWindow.entity_type).where(
                    SearchWindow.entity_type.in_(("audio_transcript", "attachment_text"))
                )
            )
        )
    assert derived_types == {"audio_transcript", "attachment_text"}
    semantic = semantic_search(
        archive_layout,
        catalog_engine,
        "timestamped audio",
        limit=100,
        embedder_factory=lambda _spec: FakeEmbedder(),
        reranker_factory=fake_reranker_factory,
    )
    assert {item["entity_type"] for item in semantic.items} >= {
        "audio_transcript",
        "attachment_text",
    }
    for item in semantic.items:
        assert item["coverage"]["status"] in {"available", "partial"}
        assert "confidence" in item
        assert item["confidence_calibrated"] is False
        assert item["availability"] == "available"
        assert item["deletion_state"] == "retained"
        assert item["provenance"]["source_id"] == item["source_id"]
    audio_result = next(
        item for item in semantic.items if item["entity_type"] == "audio_transcript"
    )
    assert audio_result["confidence_kind"] == "audio_decoder_score"
    assert isinstance(audio_result["confidence"], float)

    catalog_engine.dispose()
    app = create_app(archive_layout.root, security_disabled=True)
    with TestClient(app) as client:
        status_response = client.get("/api/v1/enrichment/status")
        transcripts_response = client.get(
            "/api/v1/enrichment/transcriptions", params={"version": "history"}
        )
        extraction_response = client.get(
            "/api/v1/enrichment/attachment-text", params={"version": "history"}
        )
        nlp_response = client.get("/api/v1/enrichment/nlp")
        nlp_hourly_response = client.get("/api/v1/enrichment/nlp/hourly")
    assert status_response.status_code == 200
    assert status_response.json()["attachment_sandbox"]["network"] == "unshared"
    assert transcripts_response.status_code == 200
    assert transcripts_response.json()["selection"]["scope"] == "history"
    assert transcripts_response.json()["items"][0]["segments"][0]["jumps"]
    assert isinstance(transcripts_response.json()["items"][0]["warnings"], list)
    assert extraction_response.status_code == 200
    assert extraction_response.json()["selection"]["scope"] == "history"
    assert "text" not in extraction_response.json()["items"][0]
    assert extraction_response.json()["items"][0]["text_preview"]
    assert extraction_response.json()["items"][0]["limits"]["cpu_seconds"] == 60
    assert isinstance(extraction_response.json()["items"][0]["warnings"], list)
    assert nlp_response.status_code == 200
    assert nlp_response.json()["items"] == []
    assert nlp_hourly_response.status_code == 200


def test_enrichment_lists_select_current_versions_and_bound_attachment_previews(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> None:
    blob_sha256 = _materialized_attachment(
        archive_layout,
        catalog_engine,
        content=b"versioned enrichment fixture",
        filename="versioned.txt",
        media_type="text/plain",
        kind="files",
    )
    large_text = "x" * 16_125
    with Session(catalog_engine) as session, session.begin():
        session.add_all(
            [
                AudioTranscription(
                    id="current-transcription-fixture",
                    blob_sha256=blob_sha256,
                    pipeline_version="audio-transcription-v1",
                    status="completed",
                    model_name=FakeTranscriber.model_name,
                    model_revision=FakeTranscriber.model_revision,
                    device="cpu",
                    compute_type="int8",
                    text_display="current transcript",
                    segment_count=0,
                ),
                AudioTranscription(
                    id="historical-transcription-fixture",
                    blob_sha256=blob_sha256,
                    pipeline_version="audio-transcription-v0",
                    status="completed",
                    model_name="fixture/whisper-old",
                    model_revision="fixture-whisper-old",
                    device="cpu",
                    compute_type="int8",
                    text_display="historical transcript",
                    segment_count=0,
                ),
                AttachmentTextExtraction(
                    id="current-extraction-fixture",
                    blob_sha256=blob_sha256,
                    pipeline_version="attachment-text-v1",
                    status="completed",
                    extractor_name=FakeExtractor.identity,
                    extractor_revision=FakeExtractor.revision,
                    filename="versioned.txt",
                    media_type="text/plain",
                    text_display=large_text,
                    text_search=large_text,
                    character_count=len(large_text),
                    truncated=False,
                    index_eligible=True,
                    sandboxed=True,
                    sandbox_status="available",
                ),
                AttachmentTextExtraction(
                    id="historical-extraction-fixture",
                    blob_sha256=blob_sha256,
                    pipeline_version="attachment-text-v0",
                    status="completed",
                    extractor_name="fixture-old-extractor",
                    extractor_revision="fixture-extractor-v0",
                    filename="versioned.txt",
                    media_type="text/plain",
                    text_display="historical extraction",
                    text_search="historical extraction",
                    character_count=21,
                    truncated=False,
                    index_eligible=True,
                    sandboxed=True,
                    sandbox_status="available",
                ),
            ]
        )

    app = create_app(archive_layout.root, security_disabled=True)
    app.state.audio_transcription_adapter = FakeTranscriber()
    app.state.attachment_text_extractor = FakeExtractor()
    with TestClient(app) as client:
        current_transcriptions = client.get("/api/v1/enrichment/transcriptions")
        historical_transcriptions = client.get(
            "/api/v1/enrichment/transcriptions", params={"version": "history"}
        )
        exact_historical_transcription = client.get(
            "/api/v1/enrichment/transcriptions",
            params={
                "version": "history",
                "pipeline_version": "audio-transcription-v0",
                "model_revision": "fixture-whisper-old",
            },
        )
        invalid_current_transcription = client.get(
            "/api/v1/enrichment/transcriptions",
            params={"pipeline_version": "audio-transcription-v0"},
        )
        current_extractions = client.get("/api/v1/enrichment/attachment-text")
        historical_extractions = client.get(
            "/api/v1/enrichment/attachment-text", params={"version": "history"}
        )
        exact_historical_extraction = client.get(
            "/api/v1/enrichment/attachment-text",
            params={
                "version": "history",
                "pipeline_version": "attachment-text-v0",
                "extractor_revision": "fixture-extractor-v0",
            },
        )
        status_response = client.get("/api/v1/enrichment/status")

    assert current_transcriptions.status_code == 200
    current_transcription_payload = current_transcriptions.json()
    assert [item["id"] for item in current_transcription_payload["items"]] == [
        "current-transcription-fixture"
    ]
    assert current_transcription_payload["items"][0]["is_current_version"] is True
    assert current_transcription_payload["selection"]["scope"] == "current"
    assert current_transcription_payload["selection"]["current"]["model_revision"] == (
        FakeTranscriber.model_revision
    )
    assert len(historical_transcriptions.json()["items"]) == 2
    assert exact_historical_transcription.json()["items"][0]["id"] == (
        "historical-transcription-fixture"
    )
    assert invalid_current_transcription.status_code == 422

    assert current_extractions.status_code == 200
    extraction_item = current_extractions.json()["items"][0]
    assert extraction_item["id"] == "current-extraction-fixture"
    assert "text" not in extraction_item
    assert len(extraction_item["text_preview"]) == 16_000
    assert extraction_item["preview_character_count"] == 16_000
    assert extraction_item["preview_truncated"] is True
    assert extraction_item["preview_limit"] == 16_000
    assert extraction_item["character_count"] == len(large_text)
    assert extraction_item["truncated"] is False
    assert len(historical_extractions.json()["items"]) == 2
    assert exact_historical_extraction.json()["items"][0]["id"] == ("historical-extraction-fixture")

    status_payload = status_response.json()
    assert status_payload["audio_transcriptions"] == {"completed": 1}
    assert status_payload["attachment_text_extractions"] == {"completed": 1}
    assert status_payload["audio_transcription_selection"]["current"]["model_revision"] == (
        FakeTranscriber.model_revision
    )
    assert status_payload["audio_transcription_selection"]["historical_record_count"] == 1
    assert status_payload["attachment_text_selection"]["current"]["extractor_revision"] == (
        FakeExtractor.revision
    )
    assert status_payload["attachment_text_selection"]["historical_record_count"] == 1


def test_enrichment_surfaces_quarantine_blobs_without_retained_occurrences(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> None:
    retained_sha256 = _materialized_attachment(
        archive_layout,
        catalog_engine,
        content=b"retained enrichment quarantine fixture",
        filename="retained.bin",
        media_type="application/octet-stream",
        kind="files",
    )
    quarantined_sha256 = _materialized_attachment(
        archive_layout,
        catalog_engine,
        content=b"last occurrence enrichment quarantine fixture",
        filename="quarantined.bin",
        media_type="application/octet-stream",
        kind="files",
    )
    shared_sha256 = _materialized_attachment(
        archive_layout,
        catalog_engine,
        content=b"shared enrichment quarantine fixture",
        filename="shared.bin",
        media_type="application/octet-stream",
        kind="files",
        occurrences=2,
    )
    now = datetime.now(UTC)
    labels_by_blob = {
        retained_sha256: "b-retained",
        quarantined_sha256: "a-quarantined",
        shared_sha256: "c-shared",
    }
    occurrence_ids_by_blob: dict[str, list[str]] = {}
    with Session(catalog_engine) as session, session.begin():
        for blob_sha256 in labels_by_blob:
            occurrences = list(
                session.scalars(
                    select(AttachmentOccurrence)
                    .where(AttachmentOccurrence.blob_sha256 == blob_sha256)
                    .order_by(AttachmentOccurrence.id)
                )
            )
            occurrence_ids_by_blob[blob_sha256] = [row.id for row in occurrences]
            if blob_sha256 == quarantined_sha256:
                assert len(occurrences) == 1
                occurrences[0].deleted_at = now
            elif blob_sha256 == shared_sha256:
                assert len(occurrences) == 2
                occurrences[0].deleted_at = now

        for blob_sha256, label in labels_by_blob.items():
            secret = "QUARANTINED_SECRET" if blob_sha256 == quarantined_sha256 else label
            current_transcription_id = f"{label}-tx-current"
            historical_transcription_id = f"{label}-tx-history"
            session.add_all(
                [
                    AudioTranscription(
                        id=current_transcription_id,
                        blob_sha256=blob_sha256,
                        pipeline_version="audio-transcription-v1",
                        status="completed",
                        model_name=FakeTranscriber.model_name,
                        model_revision=FakeTranscriber.model_revision,
                        device="cpu",
                        compute_type="int8",
                        text_display=f"current transcript {secret}",
                        segment_count=1,
                    ),
                    AudioTranscription(
                        id=historical_transcription_id,
                        blob_sha256=blob_sha256,
                        pipeline_version="audio-transcription-v0",
                        status="completed",
                        model_name="fixture/whisper-old",
                        model_revision="fixture-whisper-old",
                        device="cpu",
                        compute_type="int8",
                        text_display=f"historical transcript {secret}",
                        segment_count=1,
                    ),
                    AttachmentTextExtraction(
                        id=f"{label}-text-current",
                        blob_sha256=blob_sha256,
                        pipeline_version="attachment-text-v1",
                        status="completed",
                        extractor_name=FakeExtractor.identity,
                        extractor_revision=FakeExtractor.revision,
                        filename=f"{label}.bin",
                        media_type="application/octet-stream",
                        text_display=f"current extraction {secret}",
                        text_search=f"current extraction {secret}".casefold(),
                        character_count=len(f"current extraction {secret}"),
                        truncated=False,
                        index_eligible=True,
                        sandboxed=True,
                        sandbox_status="available",
                    ),
                    AttachmentTextExtraction(
                        id=f"{label}-text-history",
                        blob_sha256=blob_sha256,
                        pipeline_version="attachment-text-v0",
                        status="completed",
                        extractor_name="fixture-old-extractor",
                        extractor_revision="fixture-extractor-v0",
                        filename=f"{label}.bin",
                        media_type="application/octet-stream",
                        text_display=f"historical extraction {secret}",
                        text_search=f"historical extraction {secret}".casefold(),
                        character_count=len(f"historical extraction {secret}"),
                        truncated=False,
                        index_eligible=True,
                        sandboxed=True,
                        sandbox_status="available",
                    ),
                ]
            )
            session.flush()
            session.add_all(
                [
                    AudioTranscriptSegment(
                        id=f"{label}-segment-current",
                        transcription_id=current_transcription_id,
                        ordinal=0,
                        start_ms=1_000,
                        end_ms=2_000,
                        text_display=f"current segment {secret}",
                        text_search=f"current segment {secret}".casefold(),
                    ),
                    AudioTranscriptSegment(
                        id=f"{label}-segment-history",
                        transcription_id=historical_transcription_id,
                        ordinal=0,
                        start_ms=3_000,
                        end_ms=4_000,
                        text_display=f"historical segment {secret}",
                        text_search=f"historical segment {secret}".casefold(),
                    ),
                ]
            )

    app = create_app(archive_layout.root, security_disabled=True)
    app.state.audio_transcription_adapter = FakeTranscriber()
    app.state.attachment_text_extractor = FakeExtractor()

    def paged_items(
        client: TestClient, path: str, *, version: str, limit: int
    ) -> list[dict[str, object]]:
        offset = 0
        items: list[dict[str, object]] = []
        while True:
            response = client.get(
                path,
                params={"version": version, "limit": limit, "offset": offset},
            )
            assert response.status_code == 200
            payload = response.json()
            assert payload["page"]["limit"] == limit
            assert payload["page"]["offset"] == offset
            items.extend(payload["items"])
            next_offset = payload["page"]["next_offset"]
            if next_offset is None:
                return items
            assert next_offset == offset + limit
            offset = next_offset

    with TestClient(app) as client:
        current_transcriptions = paged_items(
            client,
            "/api/v1/enrichment/transcriptions",
            version="current",
            limit=1,
        )
        historical_transcriptions = paged_items(
            client,
            "/api/v1/enrichment/transcriptions",
            version="history",
            limit=2,
        )
        current_extractions = paged_items(
            client,
            "/api/v1/enrichment/attachment-text",
            version="current",
            limit=1,
        )
        historical_extractions = paged_items(
            client,
            "/api/v1/enrichment/attachment-text",
            version="history",
            limit=2,
        )
        status_payload = client.get("/api/v1/enrichment/status").json()

    assert {item["id"] for item in current_transcriptions} == {
        "b-retained-tx-current",
        "c-shared-tx-current",
    }
    assert {item["id"] for item in historical_transcriptions} == {
        "b-retained-tx-current",
        "b-retained-tx-history",
        "c-shared-tx-current",
        "c-shared-tx-history",
    }
    assert {item["id"] for item in current_extractions} == {
        "b-retained-text-current",
        "c-shared-text-current",
    }
    assert {item["id"] for item in historical_extractions} == {
        "b-retained-text-current",
        "b-retained-text-history",
        "c-shared-text-current",
        "c-shared-text-history",
    }

    all_payloads = (
        current_transcriptions
        + historical_transcriptions
        + current_extractions
        + historical_extractions
    )
    assert "QUARANTINED_SECRET" not in repr(all_payloads)
    assert quarantined_sha256 not in {item["blob_sha256"] for item in all_payloads}

    retained_occurrence_id = occurrence_ids_by_blob[retained_sha256][0]
    shared_deleted_id, shared_retained_id = occurrence_ids_by_blob[shared_sha256]
    for item in historical_transcriptions:
        expected_occurrence_id = (
            retained_occurrence_id if item["blob_sha256"] == retained_sha256 else shared_retained_id
        )
        segments = item["segments"]
        assert isinstance(segments, list) and len(segments) == 1
        assert {jump["occurrence_id"] for jump in segments[0]["jumps"]} == {expected_occurrence_id}
    for item in historical_extractions:
        expected_occurrence_id = (
            retained_occurrence_id if item["blob_sha256"] == retained_sha256 else shared_retained_id
        )
        assert {jump["occurrence_id"] for jump in item["jumps"]} == {expected_occurrence_id}
    assert shared_deleted_id not in repr(all_payloads)

    assert status_payload["audio_transcriptions"] == {"completed": 2}
    assert status_payload["attachment_text_extractions"] == {"completed": 2}
    assert status_payload["audio_transcription_selection"]["current_record_count"] == 2
    assert status_payload["audio_transcription_selection"]["historical_record_count"] == 2
    assert status_payload["attachment_text_selection"]["current_record_count"] == 2
    assert status_payload["attachment_text_selection"]["historical_record_count"] == 2


def test_conflict_review_queue_assigns_server_splits_and_blocks_early_calibration(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> None:
    conversation_id = "manual-review-conversation"
    expected_split = deterministic_conflict_split(conversation_id)
    with Session(catalog_engine) as session, session.begin():
        session.add(
            Conversation(
                id=conversation_id,
                conversation_key=conversation_id,
                title_raw="Manual review",
                title_display="Manual review",
                title_search="manual review",
                conversation_type="direct",
                primary_category="fixture",
            )
        )
        session.add_all(
            [
                SearchWindow(
                    id=f"manual-review-window-{index}",
                    window_version="fixture-v1",
                    entity_type="message_window",
                    conversation_id=conversation_id,
                    first_message_id=f"manual-review-message-{index}",
                    turn_count=1,
                    approx_token_count=4,
                    text_display=f"Review text {index}",
                    text_search=f"review text {index}",
                    participants_search="",
                    source_ids_json={},
                    build_generation="fixture-generation",
                )
                for index in range(2)
            ]
        )

    scorer_calls = 0

    def forbidden_scorer(_spec: ConflictModelSpec) -> FakeConflictScorer:
        nonlocal scorer_calls
        scorer_calls += 1
        raise AssertionError("Ineligible calibration must not load the scorer")

    app = create_app(archive_layout.root, security_disabled=True)
    app.state.conflict_scorer_factory = forbidden_scorer
    with TestClient(app) as client:
        initial_status = client.get("/api/v1/enrichment/conflict/review/status")
        initial_queue = client.get(
            "/api/v1/enrichment/conflict/review/queue",
            params={"split": expected_split, "label": "unlabelled", "limit": 1},
        )
        first_window_id = initial_queue.json()["items"][0]["window_id"]
        caller_split = client.post(
            "/api/v1/enrichment/conflict/labels",
            json={
                "window_id": first_window_id,
                "label": "direct conflict",
                "evaluation_split": "test",
            },
        )
        labelled = client.post(
            "/api/v1/enrichment/conflict/labels",
            json={
                "window_id": first_window_id,
                "label": "calm disagreement",
                "notes": "Explicit six-way manual review",
            },
        )
        labelled_queue = client.get(
            "/api/v1/enrichment/conflict/review/queue",
            params={"split": expected_split, "label": "calm disagreement"},
        )
        next_window = client.get(
            "/api/v1/enrichment/conflict/review/next",
            params={"split": expected_split, "after": first_window_id},
        )
        calibration = client.post("/api/v1/enrichment/conflict/calibrations", json={})
        final_status = client.get("/api/v1/enrichment/conflict/review/status")

    assert initial_status.status_code == 200
    assert initial_status.json()["manual_label_count"] == 0
    assert initial_status.json()["relationship_ranking_enabled"] is False
    assert initial_queue.status_code == 200
    assert initial_queue.json()["page"]["total"] == 2
    assert initial_queue.json()["items"][0]["evaluation_split"] == expected_split
    assert initial_queue.json()["items"][0]["jump"]["href"].startswith(
        f"/conversations/{conversation_id}?around="
    )
    assert caller_split.status_code == 422
    assert labelled.status_code == 201
    assert labelled.json()["evaluation_split"] == expected_split
    assert labelled.json()["notes"] == "Explicit six-way manual review"
    assert labelled.json()["split_assignment"]["source"] == "conversation_id"
    assert labelled_queue.json()["items"][0]["current_manual_label"]["notes"] == (
        "Explicit six-way manual review"
    )
    assert next_window.json()["item"]["window_id"] != first_window_id
    assert next_window.json()["item"]["conversation_id"] == conversation_id
    assert next_window.json()["item"]["evaluation_split"] == expected_split
    assert calibration.status_code == 409
    calibration_detail = calibration.json()["detail"]
    assert calibration_detail["code"] == "conflict_calibration_ineligible"
    assert any(
        blocker["code"] == "insufficient_manual_labels"
        for blocker in calibration_detail["eligibility"]["blockers"]
    )
    assert scorer_calls == 0
    assert final_status.json()["manual_label_count"] == 1
    assert final_status.json()["validation"]["eligible_for_calibration"] is False
    with Session(catalog_engine) as session:
        assert session.scalar(select(func.count(ConflictCalibration.id))) == 0


def test_conflict_search_is_lazy_bounded_six_label_and_never_bypasses_calibration(
    archive_layout: ArchiveLayout, catalog_engine: Engine, tmp_path: Path
) -> None:
    fixtures = Path(__file__).parent / "fixtures"
    shutil.copytree(fixtures / "messages", archive_layout.root, dirs_exist_ok=True)
    scan_archive(archive_layout, catalog_engine, hash_files=True)
    import_messages(archive_layout, catalog_engine)
    build_semantic_index(
        archive_layout,
        catalog_engine,
        embedder=FakeEmbedder(),
        batch_size=4,
    )
    lazy = ModernBertNliConflictScorer(tmp_path / "conflict-model")
    assert lazy._model is None
    assert lazy.status["status"] == "ready_for_lazy_load"

    scorer = FakeConflictScorer()
    page = conflict_search(
        archive_layout,
        catalog_engine,
        "parts of conversations that seem like arguments",
        target_label="direct conflict",
        limit=5,
        embedder_factory=lambda _spec: FakeEmbedder(),
        reranker_factory=fake_reranker_factory,
        scorer_factory=lambda _spec: scorer,
    )
    assert page.items
    assert len(scorer.calls) == 1
    assert len(scorer.calls[0]) <= DEFAULT_CONFLICT_SPEC.max_candidates
    assert all(
        set(item["conflict_scoring"]["scores"]) == set(CONFLICT_LABELS) for item in page.items
    )
    assert all(item["score_kind"] == "nli_entailment_logit" for item in page.items)
    assert all(item["conflict_scoring"]["ranking_feature_only"] is True for item in page.items)
    assert all(
        item["conflict_scoring"]["provenance"]["model_revision"]
        == DEFAULT_CONFLICT_SPEC.model_revision
        for item in page.items
    )
    assert all(
        item["conflict_scoring"]["aggregation_gate_bypassed"] is False for item in page.items
    )
    assert conflict_aggregation_status(catalog_engine)["aggregation_enabled"] is False

    duplicate_window_id = str(page.items[0]["window_id"])
    with Session(catalog_engine) as session, session.begin():
        session.add_all(
            [
                ConflictLabelRow(
                    id="fixture-conflict-label-one",
                    window_id=duplicate_window_id,
                    label="direct conflict",
                    evaluation_split="train",
                ),
                ConflictLabelRow(
                    id="fixture-conflict-label-two",
                    window_id=duplicate_window_id,
                    label="direct conflict",
                    evaluation_split="train",
                ),
            ]
        )
    duplicate_safe_calibration = record_conflict_calibration(
        archive_layout,
        catalog_engine,
    )
    assert duplicate_safe_calibration.sample_count == 1
    assert conflict_aggregation_status(catalog_engine)["label_count"] == 1
    assert duplicate_safe_calibration.aggregation_enabled is False

    with pytest.raises(ConflictScoringUnavailableError, match="were not mislabeled"):
        conflict_search(
            archive_layout,
            catalog_engine,
            "argument",
            embedder_factory=lambda _spec: FakeEmbedder(),
            reranker_factory=fake_reranker_factory,
            scorer_factory=lambda _spec: FakeConflictScorer(fail=True),
        )

    catalog_engine.dispose()
    app = create_app(archive_layout.root, security_disabled=True)
    app.state.search_embedder_factory = lambda _spec: FakeEmbedder()
    app.state.search_reranker_factory = fake_reranker_factory
    app.state.conflict_scorer_factory = fake_conflict_scorer_factory

    def unavailable_conflict_scorer(_spec: ConflictModelSpec) -> FakeConflictScorer:
        raise RuntimeError("fixture conflict model missing")

    with TestClient(app) as client:
        response = client.get(
            "/api/v1/enrichment/conflict/search",
            params={"q": "argument", "label": "direct conflict", "limit": 3},
        )
        status_response = client.get("/api/v1/enrichment/status")
        window_id = next(
            item["window_id"]
            for item in response.json()["items"]
            if item["window_id"] != duplicate_window_id
        )
        first_label_response = client.post(
            "/api/v1/enrichment/conflict/labels",
            json={
                "window_id": window_id,
                "label": "direct conflict",
                "notes": "First manual review",
            },
        )
        updated_label_response = client.post(
            "/api/v1/enrichment/conflict/labels",
            json={
                "window_id": window_id,
                "label": "banter",
                "notes": "Reviewed again with more context",
            },
        )
        app.state.conflict_scorer_factory = unavailable_conflict_scorer
        app.state.conflict_scorer_cache = {}
        unavailable_response = client.get(
            "/api/v1/enrichment/conflict/search",
            params={"q": "argument", "limit": 3},
        )
        unavailable_status = client.get("/api/v1/enrichment/status")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["items"]
    assert payload["conflict_model"]["model_revision"] == DEFAULT_CONFLICT_SPEC.model_revision
    assert payload["conflict_model"]["quantization"] == "dynamic_int8"
    assert payload["aggregation"]["aggregation_enabled"] is False
    assert status_response.json()["conflict_scoring_model"]["status"] == "loaded_fixture"
    assert first_label_response.status_code == 201
    assert first_label_response.json()["updated"] is False
    assigned_split = first_label_response.json()["evaluation_split"]
    assert updated_label_response.status_code == 201
    assert updated_label_response.json()["updated"] is True
    assert updated_label_response.json()["label"] == "banter"
    assert updated_label_response.json()["notes"] == "Reviewed again with more context"
    assert updated_label_response.json()["evaluation_split"] == assigned_split
    assert unavailable_response.status_code == 409
    assert "were not mislabeled" in unavailable_response.json()["detail"]
    assert unavailable_status.json()["conflict_scoring_model"]["status"] == "unavailable"
    assert (
        "fixture conflict model missing"
        in unavailable_status.json()["conflict_scoring_model"]["error"]
    )

    help_result = CliRunner().invoke(cli_app, ["search-conflict", "--help"])
    assert help_result.exit_code == 0
    assert "ModernBERT NLI" in help_result.stdout
