from __future__ import annotations

import fcntl
import math
import os
import re
import uuid
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from sqlalchemy import Engine, and_, delete, select, text, update
from sqlalchemy.orm import Session

from meta_archive.models import (
    ActivityRecord,
    AttachmentOccurrence,
    AudioTranscription,
    AudioTranscriptSegment,
    Blob,
    Job,
)
from meta_archive.paths import ArchiveLayout, UnsafeArchivePathError, regular_file_stat_beneath
from meta_archive.services.deletion_lock_lease import deletion_lock
from meta_archive.services.derived_publication_guard import assert_derived_publication_allowed
from meta_archive.services.hashing import sha256_file
from meta_archive.services.identifiers import stable_id
from meta_archive.services.jumps import activity_jump
from meta_archive.services.semantic_state import invalidate_ready_semantic_indexes
from meta_archive.services.text import normalize_text

PIPELINE_VERSION = "audio-transcription-v1"
MODEL_NAME = "Systran/faster-whisper-small"
MODEL_REVISION = "536b0662742c02347bc0e980a01041f333bce120"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
MAX_ERROR_LENGTH = 4_000
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
AUDIO_EXTENSIONS = {
    ".aac",
    ".amr",
    ".flac",
    ".m4a",
    ".mp3",
    ".oga",
    ".ogg",
    ".opus",
    ".wav",
}
ProgressCallback = Callable[[str, int, int | None], None]


def audio_transcription_lock_is_held(layout: ArchiveLayout) -> bool:
    """Probe the process lock without disturbing a running transcription."""

    if layout.catalog.is_symlink() or not layout.catalog.is_dir():
        return False
    lock_path = layout.catalog / "audio-transcription.lock"
    with lock_path.open("a+b") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    return False


@dataclass(frozen=True, slots=True)
class TranscriptSegmentResult:
    start_ms: int
    end_ms: int
    text: str
    avg_log_probability: float | None = None
    no_speech_probability: float | None = None
    words: tuple[dict[str, Any], ...] = ()

    @property
    def confidence(self) -> float | None:
        if self.avg_log_probability is None:
            return None
        return max(0.0, min(1.0, math.exp(self.avg_log_probability)))


@dataclass(frozen=True, slots=True)
class TranscriptResult:
    segments: tuple[TranscriptSegmentResult, ...]
    language: str | None
    language_probability: float | None
    duration_ms: int | None


class TranscriptionAdapter(Protocol):
    @property
    def identity(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    @property
    def model_revision(self) -> str: ...

    @property
    def device(self) -> str: ...

    @property
    def compute_type(self) -> str: ...

    def transcribe(self, source: Path) -> TranscriptResult: ...


class FasterWhisperSmallCpuInt8:
    """Lazy, revision-pinned faster-whisper adapter.

    Constructing this adapter performs no model import or download. The first actual transcription
    loads the multilingual small model into the archive-local model cache using CPU INT8.
    """

    def __init__(self, model_cache: Path) -> None:
        self.model_cache = model_cache
        self._model: Any | None = None

    @property
    def identity(self) -> str:
        return "faster-whisper-small-cpu-int8"

    @property
    def model_name(self) -> str:
        return MODEL_NAME

    @property
    def model_revision(self) -> str:
        return MODEL_REVISION

    @property
    def device(self) -> str:
        return DEVICE

    @property
    def compute_type(self) -> str:
        return COMPUTE_TYPE

    def _loaded_model(self) -> Any:
        if self._model is None:
            from faster_whisper import WhisperModel  # type: ignore[import-untyped]

            self.model_cache.mkdir(parents=True, exist_ok=True)
            self._model = WhisperModel(
                "small",
                device=DEVICE,
                compute_type=COMPUTE_TYPE,
                cpu_threads=max(1, min(8, os.cpu_count() or 1)),
                num_workers=1,
                download_root=str(self.model_cache),
                revision=MODEL_REVISION,
            )
        return self._model

    def transcribe(self, source: Path) -> TranscriptResult:
        raw_segments, info = self._loaded_model().transcribe(
            str(source),
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        segments: list[TranscriptSegmentResult] = []
        for raw in raw_segments:
            rendered = normalize_text(str(raw.text)).display.strip()
            if not rendered:
                continue
            words = tuple(
                {
                    "start_ms": round(float(word.start) * 1000),
                    "end_ms": round(float(word.end) * 1000),
                    "word": normalize_text(str(word.word)).display,
                    "probability": float(word.probability),
                }
                for word in (raw.words or ())
            )
            segments.append(
                TranscriptSegmentResult(
                    start_ms=round(float(raw.start) * 1000),
                    end_ms=round(float(raw.end) * 1000),
                    text=rendered,
                    avg_log_probability=float(raw.avg_logprob),
                    no_speech_probability=float(raw.no_speech_prob),
                    words=words,
                )
            )
        return TranscriptResult(
            segments=tuple(segments),
            language=str(info.language) if info.language else None,
            language_probability=float(info.language_probability)
            if info.language_probability is not None
            else None,
            duration_ms=round(float(info.duration) * 1000) if info.duration is not None else None,
        )


@dataclass(frozen=True, slots=True)
class TranscriptionSummary:
    job_id: str
    pipeline_version: str
    model_name: str
    total: int
    transcribed: int
    reused: int
    failed: int
    skipped_failed: int
    resumed: bool


@dataclass(frozen=True, slots=True)
class _AudioCandidate:
    blob_sha256: str
    size_bytes: int
    relative_path: str | None
    state: str


def transcription_id(blob_sha256: str, model_revision: str = MODEL_REVISION) -> str:
    return stable_id("audio-transcription", PIPELINE_VERSION, model_revision, blob_sha256)


def segment_id(transcription: str, ordinal: int) -> str:
    return stable_id("audio-transcript-segment", transcription, str(ordinal))


def _is_audio(
    *, kind: str, media_type: str | None, extension: str | None, filename: str | None
) -> bool:
    normalized_extension = (extension or Path(filename or "").suffix).casefold()
    if normalized_extension and not normalized_extension.startswith("."):
        normalized_extension = f".{normalized_extension}"
    return (
        kind.casefold() in {"audio", "audio_files"}
        or (media_type or "").casefold().startswith("audio/")
        or normalized_extension in AUDIO_EXTENSIONS
    )


def _candidates(engine: Engine) -> list[_AudioCandidate]:
    by_blob: dict[str, _AudioCandidate] = {}
    with Session(engine) as session:
        rows = session.execute(
            select(
                Blob.sha256,
                Blob.size_bytes,
                Blob.relative_path,
                Blob.state,
                Blob.media_type,
                Blob.extension,
                AttachmentOccurrence.kind,
                AttachmentOccurrence.media_type,
                AttachmentOccurrence.filename,
            )
            .join(AttachmentOccurrence, AttachmentOccurrence.blob_sha256 == Blob.sha256)
            .where(
                Blob.state == "materialized",
                AttachmentOccurrence.deleted_at.is_(None),
                AttachmentOccurrence.availability.in_(("resolved", "unlinked")),
            )
            .order_by(Blob.sha256, AttachmentOccurrence.id)
        )
        for row in rows:
            if not _is_audio(
                kind=row.kind,
                media_type=row.media_type or row.media_type_1,
                extension=row.extension,
                filename=row.filename,
            ):
                continue
            by_blob.setdefault(
                row.sha256,
                _AudioCandidate(row.sha256, row.size_bytes, row.relative_path, row.state),
            )
    return [by_blob[key] for key in sorted(by_blob)]


def _validated_blob(layout: ArchiveLayout, candidate: _AudioCandidate) -> Path:
    if not SHA256_PATTERN.fullmatch(candidate.blob_sha256):
        raise RuntimeError("Catalog contains an invalid blob identifier")
    source = layout.blob_path(candidate.blob_sha256)
    if candidate.relative_path != source.relative_to(layout.root).as_posix():
        raise RuntimeError("Catalogued blob path does not match its fixed content address")
    try:
        source_stat = regular_file_stat_beneath(source, layout.root)
    except (OSError, UnsafeArchivePathError) as exc:
        raise RuntimeError("Catalogued audio bytes are unavailable") from exc
    if candidate.state != "materialized" or source_stat.st_mode & 0o222:
        raise RuntimeError("Audio input is not an immutable materialized catalog blob")
    if source_stat.st_size != candidate.size_bytes or sha256_file(source) != candidate.blob_sha256:
        raise RuntimeError("Catalogued audio bytes do not match their content address")
    return source


def _prepare_job(
    engine: Engine,
    *,
    total: int,
    adapter: TranscriptionAdapter,
    requested_job_id: str | None = None,
) -> tuple[str, bool]:
    checkpoint = {
        "pipeline_version": PIPELINE_VERSION,
        "model_revision": adapter.model_revision,
    }
    now = datetime.now(UTC)
    with Session(engine) as session, session.begin():
        if requested_job_id is not None:
            requested = session.get(Job, requested_job_id)
            if requested is None or requested.kind != "audio_transcription":
                raise RuntimeError("Queued audio transcription job was not found")
            resumed = requested.status not in {"queued", "running"}
            requested.status = "running"
            requested.stage = "Transcribing unique audio blobs"
            requested.progress_current = 0
            requested.progress_total = total
            requested.checkpoint_json = checkpoint
            requested.counters_json = {"adapter": adapter.identity}
            requested.error_text = None
            requested.started_at = now
            requested.finished_at = None
            return requested.id, resumed
        previous = session.scalar(
            select(Job)
            .where(Job.kind == "audio_transcription", Job.status.in_(("running", "failed")))
            .order_by(Job.created_at.desc())
        )
        if previous is not None and all(
            previous.checkpoint_json.get(key) == value for key, value in checkpoint.items()
        ):
            previous.status = "running"
            previous.stage = "Resuming local audio transcription"
            previous.progress_total = total
            previous.error_text = None
            previous.finished_at = None
            previous.started_at = now
            previous.counters_json = {"adapter": adapter.identity}
            return previous.id, True
        job_id = str(uuid.uuid4())
        session.add(
            Job(
                id=job_id,
                kind="audio_transcription",
                status="running",
                stage="Transcribing unique audio blobs",
                progress_current=0,
                progress_total=total,
                checkpoint_json=checkpoint,
                counters_json={"adapter": adapter.identity},
                started_at=now,
            )
        )
        return job_id, False


def _safe_error(error: BaseException, layout: ArchiveLayout) -> str:
    return str(error).replace(str(layout.root), "<archive>")[:MAX_ERROR_LENGTH]


def _write_result(
    engine: Engine,
    *,
    record_id: str,
    candidate: _AudioCandidate,
    adapter: TranscriptionAdapter,
    result: TranscriptResult,
    job_id: str,
) -> None:
    normalized_segments = [
        (segment, normalize_text(segment.text))
        for segment in result.segments
        if segment.text.strip()
    ]
    joined = "\n".join(normalized.display for _segment, normalized in normalized_segments)
    normalized_joined = normalize_text(joined)
    now = datetime.now(UTC)
    with Session(engine) as session, session.begin():
        row = session.get(AudioTranscription, record_id)
        if row is None:
            raise RuntimeError("Audio transcription checkpoint disappeared")
        old_ids = list(
            session.scalars(
                select(AudioTranscriptSegment.id).where(
                    AudioTranscriptSegment.transcription_id == record_id
                )
            )
        )
        for old_id in old_ids:
            session.execute(
                text(
                    "DELETE FROM derived_text_fts "
                    "WHERE entity_kind = 'audio_segment' AND source_id = :source_id"
                ),
                {"source_id": old_id},
            )
        session.execute(
            delete(AudioTranscriptSegment).where(
                AudioTranscriptSegment.transcription_id == record_id
            )
        )
        for ordinal, (segment, normalized) in enumerate(normalized_segments):
            source_id = segment_id(record_id, ordinal)
            session.add(
                AudioTranscriptSegment(
                    id=source_id,
                    transcription_id=record_id,
                    ordinal=ordinal,
                    start_ms=segment.start_ms,
                    end_ms=segment.end_ms,
                    text_display=normalized.display,
                    text_search=normalized.search,
                    avg_log_probability=segment.avg_log_probability,
                    no_speech_probability=segment.no_speech_probability,
                    confidence=segment.confidence,
                    words_json={"items": list(segment.words)},
                )
            )
            session.execute(
                text(
                    "INSERT INTO derived_text_fts(entity_kind, source_id, text_search) "
                    "VALUES ('audio_segment', :source_id, :text_search)"
                ),
                {"source_id": source_id, "text_search": normalized.search},
            )
        row.status = "completed"
        row.language = result.language
        row.language_probability = result.language_probability
        row.duration_ms = result.duration_ms
        row.text_display = normalized_joined.display
        row.text_search = normalized_joined.search
        row.segment_count = len(normalized_segments)
        row.index_eligible = bool(normalized_segments)
        row.error_text = None
        row.finished_at = now
        row.provenance_json = {
            "blob_sha256": candidate.blob_sha256,
            "model_name": adapter.model_name,
            "model_revision": adapter.model_revision,
            "pipeline_version": PIPELINE_VERSION,
            "generated_locally": True,
        }
        row.warnings_json = {
            "items": [
                "Machine transcription may contain errors; inspect the original audio for context.",
                "Confidence values are model estimates and are not calibrated probabilities.",
            ]
        }
        row.job_id = job_id
        if normalized_segments:
            invalidate_ready_semantic_indexes(
                session,
                reason="Searchable audio transcript text changed",
                source_kind="audio_transcription",
                source_id=record_id,
            )


def build_audio_transcriptions(
    layout: ArchiveLayout,
    engine: Engine,
    *,
    adapter: TranscriptionAdapter | None = None,
    retry_failed: bool = True,
    progress: ProgressCallback | None = None,
    requested_job_id: str | None = None,
) -> TranscriptionSummary:
    """Transcribe each immutable audio blob once with durable per-blob checkpoints."""

    lock_path = layout.catalog / "audio-transcription.lock"
    with ExitStack() as archive_locks:
        archive_locks.enter_context(deletion_lock(layout))
        layout.ensure_runtime_directories()
        lock_handle = archive_locks.enter_context(lock_path.open("a+b"))
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another audio transcription job is already running") from exc
        assert_derived_publication_allowed(engine, publisher="Audio transcription publication")
        active_adapter = adapter or FasterWhisperSmallCpuInt8(
            layout.models / "faster-whisper" / MODEL_REVISION
        )
        candidates = _candidates(engine)
        job_id, resumed = _prepare_job(
            engine,
            total=len(candidates),
            adapter=active_adapter,
            requested_job_id=requested_job_id,
        )
        transcribed = reused = failed = skipped_failed = 0
        try:
            for processed, candidate in enumerate(candidates, start=1):
                record_id = transcription_id(candidate.blob_sha256, active_adapter.model_revision)
                with Session(engine) as session:
                    existing = session.get(AudioTranscription, record_id)
                    if existing is not None and existing.status == "completed":
                        reused += 1
                        _update_job(
                            engine,
                            job_id,
                            processed,
                            len(candidates),
                            candidate,
                            model_revision=active_adapter.model_revision,
                            transcribed=transcribed,
                            reused=reused,
                            failed=failed,
                            skipped_failed=skipped_failed,
                        )
                        continue
                    if existing is not None and existing.status == "failed" and not retry_failed:
                        skipped_failed += 1
                        _update_job(
                            engine,
                            job_id,
                            processed,
                            len(candidates),
                            candidate,
                            model_revision=active_adapter.model_revision,
                            transcribed=transcribed,
                            reused=reused,
                            failed=failed,
                            skipped_failed=skipped_failed,
                        )
                        continue
                    attempt_count = (existing.attempt_count if existing is not None else 0) + 1
                with Session(engine) as session, session.begin():
                    row = session.get(AudioTranscription, record_id)
                    if row is None:
                        row = AudioTranscription(
                            id=record_id,
                            blob_sha256=candidate.blob_sha256,
                            pipeline_version=PIPELINE_VERSION,
                            status="processing",
                            model_name=active_adapter.model_name,
                            model_revision=active_adapter.model_revision,
                            device=active_adapter.device,
                            compute_type=active_adapter.compute_type,
                            segment_count=0,
                            attempt_count=attempt_count,
                            index_eligible=True,
                            provenance_json={},
                            warnings_json={},
                            job_id=job_id,
                            started_at=datetime.now(UTC),
                        )
                        session.add(row)
                    else:
                        row.status = "processing"
                        row.attempt_count = attempt_count
                        row.error_text = None
                        row.job_id = job_id
                        row.started_at = datetime.now(UTC)
                try:
                    source = _validated_blob(layout, candidate)
                    result = active_adapter.transcribe(source)
                    _write_result(
                        engine,
                        record_id=record_id,
                        candidate=candidate,
                        adapter=active_adapter,
                        result=result,
                        job_id=job_id,
                    )
                    transcribed += 1
                except BaseException as exc:
                    failed += 1
                    with Session(engine) as session, session.begin():
                        row = session.get(AudioTranscription, record_id)
                        if row is not None:
                            row.status = "failed"
                            row.index_eligible = False
                            row.error_text = _safe_error(exc, layout)
                            row.finished_at = datetime.now(UTC)
                _update_job(
                    engine,
                    job_id,
                    processed,
                    len(candidates),
                    candidate,
                    model_revision=active_adapter.model_revision,
                    transcribed=transcribed,
                    reused=reused,
                    failed=failed,
                    skipped_failed=skipped_failed,
                )
                if progress:
                    progress("Transcribed audio blobs", processed, len(candidates))
            final_status = "completed_with_errors" if failed or skipped_failed else "completed"
            with Session(engine) as session, session.begin():
                job = session.get(Job, job_id)
                if job is not None:
                    job.status = final_status
                    job.stage = "Audio transcription complete"
                    job.finished_at = datetime.now(UTC)
                    job.error_text = None if final_status == "completed" else "Some audio failed"
            return TranscriptionSummary(
                job_id=job_id,
                pipeline_version=PIPELINE_VERSION,
                model_name=active_adapter.model_name,
                total=len(candidates),
                transcribed=transcribed,
                reused=reused,
                failed=failed,
                skipped_failed=skipped_failed,
                resumed=resumed,
            )
        except BaseException as exc:
            with Session(engine) as session, session.begin():
                session.execute(
                    update(Job)
                    .where(Job.id == job_id)
                    .values(
                        status="failed",
                        stage="Audio transcription interrupted",
                        error_text=_safe_error(exc, layout),
                        finished_at=datetime.now(UTC),
                    )
                )
            raise


def _update_job(
    engine: Engine,
    job_id: str,
    processed: int,
    total: int,
    candidate: _AudioCandidate,
    *,
    model_revision: str,
    transcribed: int,
    reused: int,
    failed: int,
    skipped_failed: int,
) -> None:
    with Session(engine) as session, session.begin():
        session.execute(
            update(Job)
            .where(Job.id == job_id)
            .values(
                progress_current=processed,
                progress_total=total,
                checkpoint_json={
                    "pipeline_version": PIPELINE_VERSION,
                    "model_revision": model_revision,
                    "last_blob_sha256": candidate.blob_sha256,
                },
                counters_json={
                    "transcribed": transcribed,
                    "reused": reused,
                    "failed": failed,
                    "skipped_failed": skipped_failed,
                },
            )
        )


def transcript_occurrence_jumps(
    engine: Engine, blob_sha256: str, *, start_ms: int
) -> list[dict[str, Any]]:
    with Session(engine) as session:
        rows = session.execute(
            select(
                AttachmentOccurrence,
                ActivityRecord.id.label("active_activity_id"),
            )
            .outerjoin(
                ActivityRecord,
                and_(
                    ActivityRecord.id == AttachmentOccurrence.activity_record_id,
                    ActivityRecord.deleted_at.is_(None),
                ),
            )
            .where(
                AttachmentOccurrence.blob_sha256 == blob_sha256,
                AttachmentOccurrence.deleted_at.is_(None),
            )
            .order_by(AttachmentOccurrence.timestamp_ms, AttachmentOccurrence.id)
        ).all()
    return [
        {
            "occurrence_id": occurrence.id,
            "conversation_id": occurrence.conversation_id,
            "message_id": occurrence.message_id,
            "activity_id": active_activity_id,
            "start_ms": start_ms,
            "href": (
                f"/conversations/{occurrence.conversation_id}?around={occurrence.message_id}"
                f"&media={occurrence.id}&t={start_ms}#message-{occurrence.message_id}"
                if occurrence.conversation_id and occurrence.message_id
                else activity_jump(active_activity_id)["href"]
                if active_activity_id
                else f"/media?occurrence={occurrence.id}&t={start_ms}"
            ),
        }
        for occurrence, active_activity_id in rows
    ]
