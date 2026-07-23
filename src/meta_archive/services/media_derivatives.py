from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import resource
import shutil
import stat
import subprocess
import tempfile
import uuid
import warnings
from collections.abc import Callable, Iterable
from contextlib import ExitStack, nullcontext
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol, cast

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from meta_archive.models import AttachmentOccurrence, Blob, Job, MediaDerivative
from meta_archive.paths import ArchiveLayout, UnsafeArchivePathError, regular_file_stat_beneath
from meta_archive.services.deletion_lock_lease import deletion_lock
from meta_archive.services.derived_publication_guard import assert_derived_publication_allowed
from meta_archive.services.hashing import sha256_file

PIPELINE_VERSION = "media-v1"
MAX_OUTPUT_BYTES = 64 * 1024 * 1024
MAX_ERROR_LENGTH = 4_000
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_DURATION_PATTERN = re.compile(
    r"Duration:\s*(?P<hours>\d{2,}):(?P<minutes>\d{2}):(?P<seconds>\d{2}(?:\.\d+)?)"
)

DerivativeKind = Literal["image_thumbnail", "video_poster", "audio_waveform"]
ProgressCallback = Callable[[str, int, int | None], None]


@dataclass(frozen=True, slots=True)
class DerivativeSpec:
    kind: DerivativeKind
    extension: str
    media_type: str


SPECS: dict[DerivativeKind, DerivativeSpec] = {
    "image_thumbnail": DerivativeSpec("image_thumbnail", "jpg", "image/jpeg"),
    "video_poster": DerivativeSpec("video_poster", "jpg", "image/jpeg"),
    "audio_waveform": DerivativeSpec("audio_waveform", "png", "image/png"),
}


@dataclass(frozen=True, slots=True)
class DerivativeResult:
    duration_ms: int | None = None
    width: int | None = None
    height: int | None = None
    metadata: dict[str, str | int | float | bool | None] | None = None


class MediaRunner(Protocol):
    @property
    def identity(self) -> str: ...

    def generate(
        self,
        kind: DerivativeKind,
        source: Path,
        destination: Path,
    ) -> DerivativeResult: ...


class MediaCommandError(RuntimeError):
    pass


def _subprocess_limits(cpu_seconds: int) -> Callable[[], None]:
    def apply_limits() -> None:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
        resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
        resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_OUTPUT_BYTES, MAX_OUTPUT_BYTES))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))

    return apply_limits


class FfmpegMediaRunner:
    """Bounded, argv-only local ffmpeg runner.

    Duration is parsed from ffmpeg's own input report because this workspace's user-local static
    ffmpeg installation does not ship a separate ffprobe binary.
    """

    def __init__(
        self,
        executable: Path | None = None,
        *,
        timeout_seconds: int = 120,
    ) -> None:
        resolved = str(executable) if executable is not None else shutil.which("ffmpeg")
        if resolved is None:
            raise RuntimeError(
                "A user-local ffmpeg executable is required; install it without sudo or pass "
                "an explicit executable path"
            )
        self.executable = Path(resolved).expanduser().resolve()
        if not self.executable.is_file() or not os.access(self.executable, os.X_OK):
            raise RuntimeError("The configured ffmpeg path is not an executable file")
        self.timeout_seconds = timeout_seconds

    @property
    def identity(self) -> str:
        return "ffmpeg"

    def generate(
        self,
        kind: DerivativeKind,
        source: Path,
        destination: Path,
    ) -> DerivativeResult:
        common = [
            str(self.executable),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "info",
            "-max_alloc",
            str(256 * 1024 * 1024),
            "-probesize",
            str(20 * 1024 * 1024),
            "-analyzeduration",
            "10000000",
            "-i",
            str(source),
            "-threads",
            "1",
            "-filter_threads",
            "1",
            "-filter_complex_threads",
            "1",
        ]
        if kind in {"image_thumbnail", "video_poster"}:
            argv = [
                *common,
                "-map",
                "0:v:0",
                "-frames:v",
                "1",
                "-vf",
                "scale=480:480:force_original_aspect_ratio=decrease",
                "-an",
                "-sn",
                "-dn",
                "-map_metadata",
                "-1",
                "-q:v",
                "4",
                "-f",
                "image2",
                "-y",
                str(destination),
            ]
        else:
            argv = [
                *common,
                "-filter_complex",
                "[0:a:0]aformat=channel_layouts=mono,showwavespic=s=800x180:colors=#5266c7[v]",
                "-map",
                "[v]",
                "-frames:v",
                "1",
                "-an",
                "-sn",
                "-dn",
                "-map_metadata",
                "-1",
                "-c:v",
                "png",
                "-f",
                "image2",
                "-y",
                str(destination),
            ]
        try:
            with tempfile.TemporaryFile() as stderr:
                # The CLI also exposes semantic commands and may therefore have LanceDB imported
                # even though this isolated ffmpeg path never uses it. LanceDB's at-fork hook emits
                # one experimental-support warning for every child process, which can turn a real
                # archive run into hundreds of thousands of irrelevant log lines. Suppress only
                # that exact third-party warning; all ffmpeg failures and unrelated warnings remain
                # visible and are handled normally.
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"lancedb fork support is experimental:.*",
                        category=RuntimeWarning,
                    )
                    completed = subprocess.run(
                        argv,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=stderr,
                        check=False,
                        timeout=self.timeout_seconds,
                        env={"LC_ALL": "C", "LANG": "C"},
                        close_fds=True,
                        preexec_fn=_subprocess_limits(max(self.timeout_seconds + 5, 10)),
                    )
                stderr.seek(0, os.SEEK_END)
                length = stderr.tell()
                stderr.seek(max(0, length - 64 * 1024))
                report = stderr.read().decode("utf-8", "replace")
        except subprocess.TimeoutExpired as exc:
            raise MediaCommandError(
                f"{kind} generation exceeded the {self.timeout_seconds}-second limit"
            ) from exc
        if completed.returncode != 0:
            tail = " ".join(report.strip().splitlines()[-3:])
            raise MediaCommandError(
                f"{kind} generation failed with exit code {completed.returncode}: {tail}"
            )
        duration_ms = _duration_ms(report)
        if kind in {"video_poster", "audio_waveform"} and duration_ms is None:
            raise MediaCommandError(f"{kind} generation did not report media duration")
        return DerivativeResult(duration_ms=duration_ms)


@dataclass(frozen=True, slots=True)
class DerivativeSummary:
    job_id: str
    pipeline_version: str
    total: int
    generated: int
    reused: int
    failed: int
    skipped_failed: int
    retried: int
    manifest: str


@dataclass(frozen=True, slots=True)
class _Candidate:
    blob_sha256: str
    size_bytes: int
    relative_path: str | None
    state: str
    kind: DerivativeKind


def derivative_id(blob_sha256: str, kind: DerivativeKind) -> str:
    payload = f"{PIPELINE_VERSION}\0{blob_sha256}\0{kind}".encode()
    return hashlib.sha256(payload).hexdigest()


def _duration_ms(report: str) -> int | None:
    match = _DURATION_PATTERN.search(report)
    if match is None:
        return None
    seconds = (
        int(match.group("hours")) * 3600
        + int(match.group("minutes")) * 60
        + float(match.group("seconds"))
    )
    return round(seconds * 1000)


def _kind_from_metadata(
    media_type: str | None,
    attachment_kind: str,
    *,
    extension: str | None,
    filename: str | None,
    availability: str | None = None,
) -> DerivativeKind | None:
    normalized_type = (media_type or "").casefold()
    normalized_kind = attachment_kind.casefold()
    normalized_extension = (extension or Path(filename or "").suffix).casefold()
    if normalized_extension and not normalized_extension.startswith("."):
        normalized_extension = f".{normalized_extension}"
    # Facebook frequently labels M4A voice messages as video/mp4. The source schema's explicit
    # attachment bucket is therefore more authoritative than the generic container MIME.
    if normalized_kind == "audio_files":
        return "audio_waveform"
    if normalized_kind == "videos":
        return "video_poster"
    if normalized_kind in {"photos", "thread_image", "gifs", "sticker"}:
        return "image_thumbnail"
    # Import normalizes resolved Meta attachment buckets to singular names. For those source
    # records the explicit bucket remains more authoritative than MIME/container metadata (voice
    # messages are commonly exported as video/mp4). Unlinked files, by contrast, receive generic
    # inferred kinds, so their concrete extension must retain precedence below.
    if availability != "unlinked":
        if normalized_kind == "audio":
            return "audio_waveform"
        if normalized_kind == "video":
            return "video_poster"
        if normalized_kind in {"photo", "gif"}:
            return "image_thumbnail"
    # Unlinked occurrences use a generic kind inferred from MIME. Recover common audio containers
    # from their filename/Blob extension before trusting that generic classification.
    if normalized_extension in {
        ".aac",
        ".amr",
        ".flac",
        ".m4a",
        ".mp3",
        ".oga",
        ".ogg",
        ".opus",
        ".wav",
    }:
        return "audio_waveform"
    if normalized_extension in {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}:
        return "video_poster"
    if normalized_extension in {
        ".avif",
        ".bmp",
        ".gif",
        ".heic",
        ".heif",
        ".jpeg",
        ".jpg",
        ".png",
        ".tif",
        ".tiff",
        ".webp",
    }:
        return "image_thumbnail"
    if normalized_kind == "audio":
        return "audio_waveform"
    if normalized_kind == "video":
        return "video_poster"
    if normalized_kind in {"photo", "gif"}:
        return "image_thumbnail"
    if normalized_type.startswith("image/"):
        return "image_thumbnail"
    if normalized_type.startswith("video/"):
        return "video_poster"
    if normalized_type.startswith("audio/"):
        return "audio_waveform"
    return None


def _candidates(engine: Engine) -> list[_Candidate]:
    by_key: dict[tuple[str, DerivativeKind], _Candidate] = {}
    with Session(engine) as session:
        rows = session.execute(
            select(
                Blob.sha256,
                Blob.size_bytes,
                Blob.relative_path,
                Blob.state,
                Blob.media_type,
                Blob.extension,
                AttachmentOccurrence.media_type,
                AttachmentOccurrence.kind,
                AttachmentOccurrence.filename,
                AttachmentOccurrence.availability,
            )
            .join(AttachmentOccurrence, AttachmentOccurrence.blob_sha256 == Blob.sha256)
            .where(
                Blob.state == "materialized",
                AttachmentOccurrence.deleted_at.is_(None),
                AttachmentOccurrence.availability.in_(("resolved", "unlinked")),
            )
            .order_by(Blob.sha256, AttachmentOccurrence.id)
        )
        for (
            sha256,
            size_bytes,
            relative_path,
            state,
            blob_type,
            blob_extension,
            occurrence_type,
            raw_kind,
            filename,
            availability,
        ) in rows:
            kind = _kind_from_metadata(
                blob_type or occurrence_type,
                raw_kind,
                extension=blob_extension,
                filename=filename,
                availability=availability,
            )
            if kind is None:
                continue
            key = (sha256, kind)
            by_key.setdefault(
                key,
                _Candidate(
                    blob_sha256=sha256,
                    size_bytes=size_bytes,
                    relative_path=relative_path,
                    state=state,
                    kind=kind,
                ),
            )
    return [by_key[key] for key in sorted(by_key)]


def _expected_relative_path(layout: ArchiveLayout, candidate: _Candidate) -> str:
    spec = SPECS[candidate.kind]
    return (
        layout.derivative_path(
            candidate.blob_sha256,
            candidate.kind,
            spec.extension,
            pipeline_version=PIPELINE_VERSION,
        )
        .relative_to(layout.root)
        .as_posix()
    )


def _validate_source_blob(layout: ArchiveLayout, candidate: _Candidate) -> Path:
    if not SHA256_PATTERN.fullmatch(candidate.blob_sha256):
        raise RuntimeError("Catalog contains an invalid blob identifier")
    if candidate.state != "materialized":
        raise RuntimeError("Derivative input is not a materialized catalog blob")
    source = layout.blob_path(candidate.blob_sha256)
    expected_relative = source.relative_to(layout.root).as_posix()
    if candidate.relative_path != expected_relative:
        raise RuntimeError("Catalogued blob path does not match its fixed content address")
    try:
        source_stat = regular_file_stat_beneath(source, layout.root)
    except (OSError, UnsafeArchivePathError) as exc:
        raise RuntimeError("Catalogued blob bytes are unavailable") from exc
    if source_stat.st_mode & 0o222:
        raise RuntimeError("Catalogued blob is writable and therefore not immutable")
    if source_stat.st_size != candidate.size_bytes:
        raise RuntimeError("Catalogued blob size does not match its record")
    if sha256_file(source) != candidate.blob_sha256:
        raise RuntimeError("Catalogued blob digest does not match its content address")
    return source


def _ready_output_is_valid(layout: ArchiveLayout, row: MediaDerivative) -> bool:
    if (
        row.pipeline_version != PIPELINE_VERSION
        or row.kind not in SPECS
        or not SHA256_PATTERN.fullmatch(row.blob_sha256)
    ):
        return False
    kind = row.kind
    spec = SPECS[kind]
    expected = layout.derivative_path(
        row.blob_sha256,
        kind,
        spec.extension,
        pipeline_version=row.pipeline_version,
    )
    if row.relative_path != expected.relative_to(layout.root).as_posix():
        return False
    try:
        output_stat = regular_file_stat_beneath(expected, layout.root)
    except (OSError, UnsafeArchivePathError):
        return False
    if output_stat.st_mode & 0o222:
        return False
    if row.size_bytes != output_stat.st_size or not row.output_sha256:
        return False
    return sha256_file(expected) == row.output_sha256


def ready_derivative_path(layout: ArchiveLayout, row: MediaDerivative) -> Path | None:
    """Resolve a ready derivative only when its fixed path and recorded bytes still match."""

    if row.status != "ready" or not _ready_output_is_valid(layout, row):
        return None
    kind = cast(DerivativeKind, row.kind)
    spec = SPECS[kind]
    return layout.derivative_path(
        row.blob_sha256,
        kind,
        spec.extension,
        pipeline_version=row.pipeline_version,
    )


def _safe_error(error: BaseException, *, layout: ArchiveLayout) -> str:
    message = str(error).replace(str(layout.root), "<archive>")
    return message[:MAX_ERROR_LENGTH]


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _generate_atomic(
    layout: ArchiveLayout,
    candidate: _Candidate,
    runner: MediaRunner,
) -> tuple[DerivativeResult, int, str]:
    source = _validate_source_blob(layout, candidate)
    spec = SPECS[candidate.kind]
    destination = layout.derivative_path(
        candidate.blob_sha256,
        candidate.kind,
        spec.extension,
        pipeline_version=PIPELINE_VERSION,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{candidate.kind}-",
        suffix=f".{spec.extension}.tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        result = runner.generate(candidate.kind, source, temporary)
        temporary_stat = temporary.lstat()
        if stat.S_ISLNK(temporary_stat.st_mode) or not stat.S_ISREG(temporary_stat.st_mode):
            raise RuntimeError("Media runner did not produce a regular derivative file")
        if temporary_stat.st_size <= 0 or temporary_stat.st_size > MAX_OUTPUT_BYTES:
            raise RuntimeError("Media runner produced an invalid derivative size")
        output_sha256 = sha256_file(temporary)
        os.chmod(temporary, 0o444)
        with temporary.open("rb", buffering=0) as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
        return result, temporary_stat.st_size, output_sha256
    finally:
        temporary.unlink(missing_ok=True)


def _prepare_job(engine: Engine, *, total: int, runner: MediaRunner) -> tuple[str, bool]:
    now = datetime.now(UTC)
    with Session(engine) as session, session.begin():
        previous = session.scalar(
            select(Job)
            .where(Job.kind == "media_derivatives", Job.status.in_(("running", "failed")))
            .order_by(Job.created_at.desc())
        )
        if previous is not None and previous.checkpoint_json.get("pipeline_version") == (
            PIPELINE_VERSION
        ):
            previous.status = "running"
            previous.stage = "Resuming local media derivatives"
            previous.progress_total = total
            previous.error_text = None
            previous.finished_at = None
            previous.started_at = now
            previous.counters_json = {"runner": runner.identity}
            return previous.id, True
        job_id = str(uuid.uuid4())
        session.add(
            Job(
                id=job_id,
                kind="media_derivatives",
                status="running",
                stage="Generating local media derivatives",
                progress_current=0,
                progress_total=total,
                checkpoint_json={"pipeline_version": PIPELINE_VERSION},
                counters_json={"runner": runner.identity},
                started_at=now,
            )
        )
        return job_id, False


def write_status_manifest(
    layout: ArchiveLayout,
    engine: Engine,
    *,
    catalog_session: Session | None = None,
) -> None:
    session_context = Session(engine) if catalog_session is None else nullcontext(catalog_session)
    with session_context as session:
        rows = list(
            session.scalars(
                select(MediaDerivative)
                .where(MediaDerivative.pipeline_version == PIPELINE_VERSION)
                .order_by(MediaDerivative.blob_sha256, MediaDerivative.kind)
            )
        )
        records = [
            {
                "id": row.id,
                "blob_sha256": row.blob_sha256,
                "kind": row.kind,
                "status": row.status,
                "relative_path": row.relative_path,
                "media_type": row.media_type,
                "size_bytes": row.size_bytes,
                "output_sha256": row.output_sha256,
                "duration_ms": row.duration_ms,
                "width": row.width,
                "height": row.height,
                "attempt_count": row.attempt_count,
                "error": row.error_text,
                "metadata": row.metadata_json,
            }
            for row in rows
        ]
    payload = {
        "schema_version": 1,
        "pipeline_version": PIPELINE_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "records": records,
    }
    layout.derivative_manifest.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".manifest-", suffix=".json.tmp", dir=layout.derivative_manifest.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_name, 0o444)
        os.replace(temporary_name, layout.derivative_manifest)
        _fsync_directory(layout.derivative_manifest.parent)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def build_media_derivatives(
    layout: ArchiveLayout,
    engine: Engine,
    *,
    runner: MediaRunner | None = None,
    retry_failed: bool = True,
    progress: ProgressCallback | None = None,
) -> DerivativeSummary:
    """Build unique-blob thumbnails, posters, and waveforms from immutable blob inputs."""

    lock_path = layout.thumbnails / ".build.lock"
    with ExitStack() as archive_locks:
        archive_locks.enter_context(deletion_lock(layout))
        layout.ensure_runtime_directories()
        lock_handle = archive_locks.enter_context(lock_path.open("a+b"))
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another media derivative build is already running") from exc
        assert_derived_publication_allowed(engine, publisher="Media derivative publication")
        active_runner = runner or FfmpegMediaRunner()
        candidates = _candidates(engine)
        job_id, resumed = _prepare_job(engine, total=len(candidates), runner=active_runner)
        generated = reused = failed = skipped_failed = retried = 0
        processed = 0
        try:
            for candidate in candidates:
                record_id = derivative_id(candidate.blob_sha256, candidate.kind)
                with Session(engine) as session:
                    existing = session.get(MediaDerivative, record_id)
                    was_retry = existing is not None and existing.status in {
                        "failed",
                        "processing",
                    }
                    if (
                        existing is not None
                        and existing.status == "ready"
                        and (_ready_output_is_valid(layout, existing))
                    ):
                        reused += 1
                        processed += 1
                        _update_job_progress(
                            engine,
                            job_id,
                            candidate=candidate,
                            processed=processed,
                            generated=generated,
                            reused=reused,
                            failed=failed,
                            skipped_failed=skipped_failed,
                            retried=retried,
                            resumed=resumed,
                        )
                        if progress:
                            progress("Reused media derivative", processed, len(candidates))
                        continue
                    if was_retry and not retry_failed:
                        skipped_failed += 1
                        processed += 1
                        _update_job_progress(
                            engine,
                            job_id,
                            candidate=candidate,
                            processed=processed,
                            generated=generated,
                            reused=reused,
                            failed=failed,
                            skipped_failed=skipped_failed,
                            retried=retried,
                            resumed=resumed,
                        )
                        continue
                if was_retry:
                    retried += 1
                spec = SPECS[candidate.kind]
                expected_relative = _expected_relative_path(layout, candidate)
                with Session(engine) as session, session.begin():
                    row = session.get(MediaDerivative, record_id)
                    if row is None:
                        row = MediaDerivative(
                            id=record_id,
                            blob_sha256=candidate.blob_sha256,
                            kind=candidate.kind,
                            pipeline_version=PIPELINE_VERSION,
                            status="processing",
                            relative_path=expected_relative,
                            media_type=spec.media_type,
                            attempt_count=1,
                            metadata_json={"runner": active_runner.identity},
                        )
                        session.add(row)
                    else:
                        row.blob_sha256 = candidate.blob_sha256
                        row.kind = candidate.kind
                        row.pipeline_version = PIPELINE_VERSION
                        row.status = "processing"
                        row.relative_path = expected_relative
                        row.media_type = spec.media_type
                        row.attempt_count += 1
                        row.error_text = None
                try:
                    result, output_size, output_sha256 = _generate_atomic(
                        layout, candidate, active_runner
                    )
                    with Session(engine) as session, session.begin():
                        row = session.get(MediaDerivative, record_id)
                        if row is None:
                            raise RuntimeError("Derivative manifest row disappeared during build")
                        row.status = "ready"
                        row.size_bytes = output_size
                        row.output_sha256 = output_sha256
                        row.duration_ms = result.duration_ms
                        row.width = result.width
                        row.height = result.height
                        row.error_text = None
                        row.generated_at = datetime.now(UTC)
                        row.metadata_json = {
                            "runner": active_runner.identity,
                            **(result.metadata or {}),
                        }
                    generated += 1
                except Exception as exc:  # one corrupt media file must not stop the archive run
                    with Session(engine) as session, session.begin():
                        row = session.get(MediaDerivative, record_id)
                        if row is None:
                            raise
                        row.status = "failed"
                        row.size_bytes = None
                        row.output_sha256 = None
                        row.duration_ms = None
                        row.width = None
                        row.height = None
                        row.error_text = _safe_error(exc, layout=layout)
                        row.generated_at = None
                    failed += 1
                processed += 1
                _update_job_progress(
                    engine,
                    job_id,
                    candidate=candidate,
                    processed=processed,
                    generated=generated,
                    reused=reused,
                    failed=failed,
                    skipped_failed=skipped_failed,
                    retried=retried,
                    resumed=resumed,
                )
                if progress:
                    progress("Generating local media derivatives", processed, len(candidates))
            finished = datetime.now(UTC)
            status = "completed_with_errors" if failed or skipped_failed else "completed"
            counters = {
                "pipeline_version": PIPELINE_VERSION,
                "runner": active_runner.identity,
                "total": len(candidates),
                "generated": generated,
                "reused": reused,
                "failed": failed,
                "skipped_failed": skipped_failed,
                "retried": retried,
                "resumed": resumed,
            }
            with Session(engine) as session, session.begin():
                job = session.get(Job, job_id)
                if job is None:
                    raise RuntimeError("Media derivative job disappeared during build")
                job.status = status
                job.stage = (
                    "Media derivatives ready"
                    if status == "completed"
                    else "Media derivatives completed with errors"
                )
                job.progress_current = processed
                job.progress_total = len(candidates)
                job.counters_json = counters
                job.checkpoint_json = {
                    "pipeline_version": PIPELINE_VERSION,
                    "processed": processed,
                    "last_blob_sha256": candidates[-1].blob_sha256 if candidates else None,
                    "last_kind": candidates[-1].kind if candidates else None,
                }
                job.finished_at = finished
            write_status_manifest(layout, engine)
            return DerivativeSummary(
                job_id=job_id,
                pipeline_version=PIPELINE_VERSION,
                total=len(candidates),
                generated=generated,
                reused=reused,
                failed=failed,
                skipped_failed=skipped_failed,
                retried=retried,
                manifest=str(layout.derivative_manifest),
            )
        except BaseException as exc:
            with Session(engine) as session, session.begin():
                job = session.get(Job, job_id)
                if job is not None:
                    job.status = "failed"
                    job.error_text = _safe_error(exc, layout=layout)
                    job.finished_at = datetime.now(UTC)
            write_status_manifest(layout, engine)
            raise


def _update_job_progress(
    engine: Engine,
    job_id: str,
    *,
    candidate: _Candidate,
    processed: int,
    generated: int,
    reused: int,
    failed: int,
    skipped_failed: int,
    retried: int,
    resumed: bool,
) -> None:
    with Session(engine) as session, session.begin():
        job = session.get(Job, job_id)
        if job is None:
            raise RuntimeError("Media derivative job disappeared during build")
        job.progress_current = processed
        job.stage = f"Processed {candidate.kind}"
        job.checkpoint_json = {
            "pipeline_version": PIPELINE_VERSION,
            "processed": processed,
            "last_blob_sha256": candidate.blob_sha256,
            "last_kind": candidate.kind,
        }
        job.counters_json = {
            "generated": generated,
            "reused": reused,
            "failed": failed,
            "skipped_failed": skipped_failed,
            "retried": retried,
            "resumed": resumed,
        }


def derivative_payload(row: MediaDerivative) -> dict[str, object]:
    """Serialize only safe, fixed-ID derivative metadata for API responses."""

    supported = row.pipeline_version == PIPELINE_VERSION and row.kind in SPECS
    spec = SPECS[cast(DerivativeKind, row.kind)] if supported else None
    return {
        "id": row.id,
        "kind": row.kind,
        "status": row.status,
        "media_type": spec.media_type if spec is not None else "application/octet-stream",
        "size_bytes": row.size_bytes,
        "duration_ms": row.duration_ms,
        "width": row.width,
        "height": row.height,
        "href": (f"/api/v1/derivatives/{row.id}" if row.status == "ready" and supported else None),
    }


def derivatives_by_blob(
    session: Session, blob_sha256s: Iterable[str | None]
) -> dict[str, list[dict[str, object]]]:
    """Return ready derivative metadata without exposing filesystem paths."""

    identifiers = sorted({value for value in blob_sha256s if value is not None})
    result: dict[str, list[dict[str, object]]] = {identifier: [] for identifier in identifiers}
    if not identifiers:
        return result
    rows = session.scalars(
        select(MediaDerivative)
        .where(
            MediaDerivative.blob_sha256.in_(identifiers),
            MediaDerivative.pipeline_version == PIPELINE_VERSION,
            MediaDerivative.status == "ready",
        )
        .order_by(MediaDerivative.blob_sha256, MediaDerivative.kind, MediaDerivative.id)
    )
    for row in rows:
        result[row.blob_sha256].append(derivative_payload(row))
    return result


def summary_payload(summary: DerivativeSummary) -> dict[str, object]:
    return asdict(summary)
