from __future__ import annotations

import fcntl
import hashlib
import json
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import uuid
import warnings
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from sqlalchemy import Engine, and_, select, text, update
from sqlalchemy.orm import Session

from meta_archive.models import (
    ActivityRecord,
    AttachmentOccurrence,
    AttachmentTextExtraction,
    Blob,
    Job,
)
from meta_archive.paths import (
    ArchiveLayout,
    UnsafeArchivePathError,
    open_regular_file_beneath,
    regular_file_stat_beneath,
)
from meta_archive.services.deletion_lock_lease import deletion_lock
from meta_archive.services.derived_publication_guard import assert_derived_publication_allowed
from meta_archive.services.hashing import sha256_file
from meta_archive.services.identifiers import stable_id
from meta_archive.services.jumps import activity_jump
from meta_archive.services.semantic_state import invalidate_ready_semantic_indexes
from meta_archive.services.text import normalize_text

PIPELINE_VERSION = "attachment-text-v1"
EXTRACTOR_REVISION = "bubblewrap-worker-v1"
MAX_CPU_SECONDS = 60
MAX_MEMORY_BYTES = 1_500 * 1024 * 1024
MAX_EXPANDED_BYTES = 500 * 1024 * 1024
MAX_ARCHIVE_ENTRIES = 1_000
MAX_ARCHIVE_DEPTH = 3
MAX_COMPRESSION_RATIO = 100
MAX_OUTPUT_BYTES = 20 * 1024 * 1024
MAX_INPUT_BYTES = 500 * 1024 * 1024
MAX_ERROR_LENGTH = 4_000
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
ProgressCallback = Callable[[str, int, int | None], None]


def attachment_text_lock_is_held(layout: ArchiveLayout) -> bool:
    """Probe the process lock without disturbing a running extraction."""

    if layout.catalog.is_symlink() or not layout.catalog.is_dir():
        return False
    lock_path = layout.catalog / "attachment-text.lock"
    with lock_path.open("a+b") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    return False


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    status: str
    text: str | None
    truncated: bool
    index_eligible: bool
    diagnostic: str | None
    warnings: tuple[str, ...] = ()
    archive_entries: int = 0
    expanded_bytes: int = 0


class AttachmentExtractor(Protocol):
    @property
    def identity(self) -> str: ...

    @property
    def revision(self) -> str: ...

    @property
    def sandbox_status(self) -> dict[str, object]: ...

    def extract(
        self, source: Path, *, filename: str, media_type: str | None
    ) -> ExtractionResult: ...


def sandbox_capability() -> dict[str, object]:
    bwrap = shutil.which("bwrap")
    prlimit = shutil.which("prlimit")
    available = sys.platform.startswith("linux") and bwrap is not None and prlimit is not None
    return {
        "available": available,
        "status": "available" if available else "unavailable_fail_closed",
        "bubblewrap": bwrap,
        "prlimit": prlimit,
        "network": "unshared",
        "input": "single read-only catalog blob",
        "temporary_storage": "isolated tmpfs",
        "limits": extraction_limits(),
    }


def extraction_limits() -> dict[str, int | float]:
    return {
        "cpu_seconds": MAX_CPU_SECONDS,
        "memory_bytes": MAX_MEMORY_BYTES,
        "archive_depth": MAX_ARCHIVE_DEPTH,
        "archive_entries": MAX_ARCHIVE_ENTRIES,
        "expanded_bytes": MAX_EXPANDED_BYTES,
        "compression_ratio": MAX_COMPRESSION_RATIO,
        "output_bytes": MAX_OUTPUT_BYTES,
    }


def _process_limits() -> Callable[[], None]:
    def apply() -> None:
        resource.setrlimit(resource.RLIMIT_CPU, (MAX_CPU_SECONDS + 2, MAX_CPU_SECONDS + 3))
        resource.setrlimit(resource.RLIMIT_AS, (MAX_MEMORY_BYTES, MAX_MEMORY_BYTES))
        resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_OUTPUT_BYTES, MAX_OUTPUT_BYTES))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))

    return apply


class BubblewrapAttachmentExtractor:
    """Fail-closed, no-network Bubblewrap worker with explicit prlimit bounds."""

    def __init__(self, *, timeout_seconds: int = MAX_CPU_SECONDS + 10) -> None:
        capability = sandbox_capability()
        if not capability["available"]:
            raise RuntimeError(
                "Broad attachment extraction is disabled because Bubblewrap/prlimit is unavailable"
            )
        self.bwrap = str(capability["bubblewrap"])
        self.prlimit = str(capability["prlimit"])
        self.timeout_seconds = timeout_seconds

    @property
    def identity(self) -> str:
        return "bubblewrap-prlimit-local-worker"

    @property
    def revision(self) -> str:
        return EXTRACTOR_REVISION

    @property
    def sandbox_status(self) -> dict[str, object]:
        return sandbox_capability()

    @staticmethod
    def _bind_if_present(argv: list[str], path: str) -> None:
        if Path(path).exists():
            argv.extend(("--ro-bind", path, path))

    def extract(self, source: Path, *, filename: str, media_type: str | None) -> ExtractionResult:
        worker = Path(__file__).with_name("attachment_worker.py").resolve()
        environment = Path(sys.prefix).resolve()
        # Production inputs use <archive>/blobs/sha256/<shard>/<digest>. Direct fixture calls use
        # their immediate parent as the no-follow boundary.
        source_root = (
            source.parents[3]
            if len(source.parents) > 3
            and source.parent.parent.name == "sha256"
            and source.parent.parent.parent.name == "blobs"
            else source.parent
        )
        source_handle = open_regular_file_beneath(source, source_root)
        argv = [
            self.bwrap,
            "--die-with-parent",
            "--new-session",
            "--unshare-all",
            "--clearenv",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--dir",
            "/app",
            "--dir",
            "/input",
            "--ro-bind",
            str(environment),
            "/venv",
            "--ro-bind",
            str(worker),
            "/app/attachment_worker.py",
            "--ro-bind-fd",
            str(source_handle.fileno()),
            "/input/source",
            "--setenv",
            "HOME",
            "/tmp",
            "--setenv",
            "LANG",
            "C.UTF-8",
            "--setenv",
            "LC_ALL",
            "C.UTF-8",
            "--setenv",
            "PYTHONNOUSERSITE",
            "1",
            "--setenv",
            "PYTHONDONTWRITEBYTECODE",
            "1",
            "--chdir",
            "/tmp",
        ]
        for path in ("/usr", "/usr/local", "/lib", "/lib64", "/etc/ld.so.cache"):
            self._bind_if_present(argv, path)
        argv.extend(
            (
                "--",
                self.prlimit,
                f"--cpu={MAX_CPU_SECONDS}",
                f"--as={MAX_MEMORY_BYTES}",
                f"--fsize={MAX_OUTPUT_BYTES}",
                "--nofile=64",
                "--",
                "/venv/bin/python",
                "-I",
                "/app/attachment_worker.py",
                "/input/source",
                "--filename",
                filename[:1_000],
            )
        )
        if media_type:
            argv.extend(("--media-type", media_type[:255]))
        try:
            # The unified CLI imports semantic-index support before dispatch. LanceDB then installs
            # an at-fork warning hook even though this isolated worker never uses LanceDB. Suppress
            # only that exact third-party warning so one line is not emitted for every attachment;
            # worker errors and every unrelated warning remain visible.
            with (
                source_handle,
                tempfile.TemporaryFile() as stdout,
                tempfile.TemporaryFile() as stderr,
            ):
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"lancedb fork support is experimental:.*",
                        category=RuntimeWarning,
                    )
                    completed = subprocess.run(
                        argv,
                        stdin=subprocess.DEVNULL,
                        stdout=stdout,
                        stderr=stderr,
                        check=False,
                        timeout=self.timeout_seconds,
                        env={},
                        close_fds=True,
                        pass_fds=(source_handle.fileno(),),
                        preexec_fn=_process_limits(),
                    )
                if isinstance(completed.stdout, bytes):
                    output = completed.stdout
                else:
                    stdout.seek(0, 2)
                    output_length = stdout.tell()
                    stdout.seek(0)
                    output = stdout.read(MAX_OUTPUT_BYTES + 1)
                    if output_length > MAX_OUTPUT_BYTES:
                        raise RuntimeError("Sandboxed attachment worker exceeded its output limit")
                if isinstance(completed.stderr, bytes):
                    error_digest = hashlib.sha256(completed.stderr).hexdigest()
                else:
                    stderr.seek(0)
                    diagnostic_hash = hashlib.sha256()
                    while chunk := stderr.read(1024 * 1024):
                        diagnostic_hash.update(chunk)
                    error_digest = diagnostic_hash.hexdigest()
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "Attachment extraction exceeded its 60-second CPU/time limit"
            ) from exc
        if completed.returncode != 0:
            raise RuntimeError(
                "Sandboxed attachment worker failed with exit code "
                f"{completed.returncode} (diagnostic sha256 {error_digest})"
            )
        if len(output) > MAX_OUTPUT_BYTES:
            raise RuntimeError("Sandboxed attachment worker exceeded its output limit")
        try:
            payload = json.loads(output)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Sandboxed attachment worker returned invalid output") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Sandboxed attachment worker returned an invalid result")
        return ExtractionResult(
            status=str(payload.get("status", "failed")),
            text=str(payload["text"]) if payload.get("text") is not None else None,
            truncated=bool(payload.get("truncated", False)),
            index_eligible=bool(payload.get("index_eligible", False)),
            diagnostic=str(payload["diagnostic"])
            if payload.get("diagnostic") is not None
            else None,
            warnings=tuple(str(value) for value in payload.get("warnings", [])),
            archive_entries=int(payload.get("entries", 0)),
            expanded_bytes=int(payload.get("expanded_bytes", 0)),
        )


@dataclass(frozen=True, slots=True)
class ExtractionSummary:
    job_id: str
    pipeline_version: str
    total: int
    extracted: int
    reused: int
    excluded: int
    unsupported: int
    limit_exceeded: int
    failed: int
    skipped_failed: int
    resumed: bool
    sandbox_status: str


@dataclass(frozen=True, slots=True)
class _Candidate:
    blob_sha256: str
    size_bytes: int
    relative_path: str | None
    state: str
    filename: str
    media_type: str | None


def extraction_id(blob_sha256: str) -> str:
    return stable_id("attachment-text", PIPELINE_VERSION, blob_sha256)


def _candidate_rows(engine: Engine) -> list[_Candidate]:
    by_blob: dict[str, _Candidate] = {}
    with Session(engine) as session:
        rows = session.execute(
            select(
                Blob.sha256,
                Blob.size_bytes,
                Blob.relative_path,
                Blob.state,
                Blob.media_type.label("blob_media_type"),
                Blob.extension,
                AttachmentOccurrence.filename,
                AttachmentOccurrence.media_type.label("occurrence_media_type"),
                AttachmentOccurrence.kind,
            )
            .join(AttachmentOccurrence, AttachmentOccurrence.blob_sha256 == Blob.sha256)
            .where(
                Blob.state == "materialized",
                AttachmentOccurrence.deleted_at.is_(None),
                AttachmentOccurrence.availability.in_(("resolved", "unlinked")),
            )
            .order_by(Blob.sha256, AttachmentOccurrence.id)
        ).mappings()
        for row in rows:
            kind = str(row["kind"]).casefold()
            media_type = str(row["blob_media_type"] or row["occurrence_media_type"] or "")
            extension = str(row["extension"] or Path(str(row["filename"] or "")).suffix)
            if kind in {"photos", "videos", "audio_files", "gifs", "sticker", "thread_image"}:
                continue
            if media_type.startswith(("image/", "audio/", "video/")):
                continue
            filename = str(row["filename"] or f"attachment{extension}")
            by_blob.setdefault(
                str(row["sha256"]),
                _Candidate(
                    blob_sha256=str(row["sha256"]),
                    size_bytes=int(row["size_bytes"]),
                    relative_path=str(row["relative_path"])
                    if row["relative_path"] is not None
                    else None,
                    state=str(row["state"]),
                    filename=filename,
                    media_type=media_type or None,
                ),
            )
    return [by_blob[key] for key in sorted(by_blob)]


def _validated_blob(layout: ArchiveLayout, candidate: _Candidate) -> Path:
    if not SHA256_PATTERN.fullmatch(candidate.blob_sha256):
        raise RuntimeError("Catalog contains an invalid blob identifier")
    source = layout.blob_path(candidate.blob_sha256)
    if candidate.relative_path != source.relative_to(layout.root).as_posix():
        raise RuntimeError("Catalogued blob path does not match its fixed content address")
    try:
        source_stat = regular_file_stat_beneath(source, layout.root)
    except (OSError, UnsafeArchivePathError) as exc:
        raise RuntimeError("Catalogued attachment bytes are unavailable") from exc
    if candidate.state != "materialized" or source_stat.st_mode & 0o222:
        raise RuntimeError("Attachment input is not an immutable materialized catalog blob")
    if source_stat.st_size > MAX_INPUT_BYTES:
        raise RuntimeError("Attachment exceeds the 500 MB extraction input limit")
    if source_stat.st_size != candidate.size_bytes or sha256_file(source) != candidate.blob_sha256:
        raise RuntimeError("Catalogued attachment does not match its content address")
    return source


def _prepare_job(
    engine: Engine,
    *,
    total: int,
    extractor: AttachmentExtractor,
    requested_job_id: str | None = None,
) -> tuple[str, bool]:
    now = datetime.now(UTC)
    with Session(engine) as session, session.begin():
        if requested_job_id is not None:
            requested = session.get(Job, requested_job_id)
            if requested is None or requested.kind != "attachment_text":
                raise RuntimeError("Queued attachment text job was not found")
            resumed = requested.status not in {"queued", "running"}
            requested.status = "running"
            requested.stage = "Extracting attachment text in Bubblewrap"
            requested.progress_current = 0
            requested.progress_total = total
            requested.checkpoint_json = {"pipeline_version": PIPELINE_VERSION}
            requested.counters_json = {"extractor": extractor.identity}
            requested.error_text = None
            requested.started_at = now
            requested.finished_at = None
            return requested.id, resumed
        previous = session.scalar(
            select(Job)
            .where(Job.kind == "attachment_text", Job.status.in_(("running", "failed")))
            .order_by(Job.created_at.desc())
        )
        if previous is not None and previous.checkpoint_json.get("pipeline_version") == (
            PIPELINE_VERSION
        ):
            previous.status = "running"
            previous.stage = "Resuming sandboxed attachment extraction"
            previous.progress_total = total
            previous.error_text = None
            previous.started_at = now
            previous.finished_at = None
            previous.counters_json = {"extractor": extractor.identity}
            return previous.id, True
        job_id = str(uuid.uuid4())
        session.add(
            Job(
                id=job_id,
                kind="attachment_text",
                status="running",
                stage="Extracting attachment text in Bubblewrap",
                progress_current=0,
                progress_total=total,
                checkpoint_json={"pipeline_version": PIPELINE_VERSION},
                counters_json={"extractor": extractor.identity},
                started_at=now,
            )
        )
        return job_id, False


def _safe_error(error: BaseException, layout: ArchiveLayout) -> str:
    return str(error).replace(str(layout.root), "<archive>")[:MAX_ERROR_LENGTH]


def build_attachment_text(
    layout: ArchiveLayout,
    engine: Engine,
    *,
    extractor: AttachmentExtractor | None = None,
    retry_failed: bool = True,
    progress: ProgressCallback | None = None,
    requested_job_id: str | None = None,
) -> ExtractionSummary:
    """Extract unique-blob text only through the fail-closed sandbox adapter."""

    lock_path = layout.catalog / "attachment-text.lock"
    with ExitStack() as archive_locks:
        archive_locks.enter_context(deletion_lock(layout))
        layout.ensure_runtime_directories()
        lock_handle = archive_locks.enter_context(lock_path.open("a+b"))
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another attachment text job is already running") from exc
        assert_derived_publication_allowed(engine, publisher="Attachment text publication")
        active_extractor = extractor or BubblewrapAttachmentExtractor()
        candidates = _candidate_rows(engine)
        job_id, resumed = _prepare_job(
            engine,
            total=len(candidates),
            extractor=active_extractor,
            requested_job_id=requested_job_id,
        )
        counts = {
            "extracted": 0,
            "reused": 0,
            "excluded": 0,
            "unsupported": 0,
            "limit_exceeded": 0,
            "failed": 0,
            "skipped_failed": 0,
        }
        try:
            for processed, candidate in enumerate(candidates, start=1):
                record_id = extraction_id(candidate.blob_sha256)
                with Session(engine) as session:
                    existing = session.get(AttachmentTextExtraction, record_id)
                    if existing is not None and existing.status in {
                        "completed",
                        "excluded",
                        "unsupported",
                        "limit_exceeded",
                    }:
                        counts["reused"] += 1
                        _update_job(engine, job_id, processed, len(candidates), candidate, counts)
                        continue
                    if existing is not None and existing.status == "failed" and not retry_failed:
                        counts["skipped_failed"] += 1
                        _update_job(engine, job_id, processed, len(candidates), candidate, counts)
                        continue
                    attempt_count = (existing.attempt_count if existing is not None else 0) + 1
                with Session(engine) as session, session.begin():
                    row = session.get(AttachmentTextExtraction, record_id)
                    if row is None:
                        row = AttachmentTextExtraction(
                            id=record_id,
                            blob_sha256=candidate.blob_sha256,
                            pipeline_version=PIPELINE_VERSION,
                            status="processing",
                            extractor_name=active_extractor.identity,
                            extractor_revision=active_extractor.revision,
                            filename=candidate.filename,
                            media_type=candidate.media_type,
                            character_count=0,
                            truncated=False,
                            index_eligible=False,
                            sandboxed=bool(active_extractor.sandbox_status.get("available")),
                            sandbox_status=str(active_extractor.sandbox_status.get("status")),
                            attempt_count=attempt_count,
                            limits_json=extraction_limits(),
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
                    result = active_extractor.extract(
                        source,
                        filename=candidate.filename,
                        media_type=candidate.media_type,
                    )
                    normalized = normalize_text(result.text or "")
                    with Session(engine) as session, session.begin():
                        row = session.get(AttachmentTextExtraction, record_id)
                        if row is None:
                            raise RuntimeError("Attachment extraction checkpoint disappeared")
                        session.execute(
                            text(
                                "DELETE FROM derived_text_fts "
                                "WHERE entity_kind = 'attachment_text' AND source_id = :source_id"
                            ),
                            {"source_id": record_id},
                        )
                        row.status = result.status
                        row.text_display = normalized.display if result.text else None
                        row.text_search = normalized.search if result.text else None
                        row.character_count = len(normalized.display) if result.text else 0
                        row.truncated = result.truncated
                        row.index_eligible = result.index_eligible and bool(result.text)
                        row.error_text = result.diagnostic
                        row.finished_at = datetime.now(UTC)
                        row.provenance_json = {
                            "blob_sha256": candidate.blob_sha256,
                            "pipeline_version": PIPELINE_VERSION,
                            "extractor_revision": active_extractor.revision,
                            "archive_entries": result.archive_entries,
                            "expanded_bytes": result.expanded_bytes,
                            "generated_locally": True,
                        }
                        warnings = list(result.warnings)
                        warnings.append(
                            "Extracted text is untrusted attachment data and is never rendered "
                            "as HTML."
                        )
                        row.warnings_json = {"items": warnings}
                        if row.index_eligible and row.text_search:
                            session.execute(
                                text(
                                    "INSERT INTO derived_text_fts"
                                    "(entity_kind, source_id, text_search) "
                                    "VALUES ('attachment_text', :source_id, :text_search)"
                                ),
                                {"source_id": record_id, "text_search": row.text_search},
                            )
                            invalidate_ready_semantic_indexes(
                                session,
                                reason="Searchable attachment text changed",
                                source_kind="attachment_text_extraction",
                                source_id=record_id,
                            )
                    key = "extracted" if result.status == "completed" else result.status
                    if key not in counts:
                        key = "unsupported"
                    counts[key] += 1
                except BaseException as exc:
                    counts["failed"] += 1
                    with Session(engine) as session, session.begin():
                        row = session.get(AttachmentTextExtraction, record_id)
                        if row is not None:
                            row.status = "failed"
                            row.index_eligible = False
                            row.error_text = _safe_error(exc, layout)
                            row.finished_at = datetime.now(UTC)
                _update_job(engine, job_id, processed, len(candidates), candidate, counts)
                if progress:
                    progress("Extracted attachment text", processed, len(candidates))
            final_status = (
                "completed_with_errors"
                if counts["failed"] or counts["skipped_failed"]
                else "completed"
            )
            with Session(engine) as session, session.begin():
                job = session.get(Job, job_id)
                if job is not None:
                    job.status = final_status
                    job.stage = "Attachment text extraction complete"
                    job.error_text = None if final_status == "completed" else "Some files failed"
                    job.finished_at = datetime.now(UTC)
            return ExtractionSummary(
                job_id=job_id,
                pipeline_version=PIPELINE_VERSION,
                total=len(candidates),
                resumed=resumed,
                sandbox_status=str(active_extractor.sandbox_status.get("status")),
                **counts,
            )
        except BaseException as exc:
            with Session(engine) as session, session.begin():
                session.execute(
                    update(Job)
                    .where(Job.id == job_id)
                    .values(
                        status="failed",
                        stage="Attachment text extraction interrupted",
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
    candidate: _Candidate,
    counts: dict[str, int],
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
                    "last_blob_sha256": candidate.blob_sha256,
                },
                counters_json=dict(counts),
            )
        )


def extraction_occurrence_jumps(engine: Engine, blob_sha256: str) -> list[dict[str, str | None]]:
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
            "href": (
                f"/conversations/{occurrence.conversation_id}?around={occurrence.message_id}"
                f"&media={occurrence.id}#message-{occurrence.message_id}"
                if occurrence.conversation_id and occurrence.message_id
                else activity_jump(active_activity_id)["href"]
                if active_activity_id
                else f"/media?occurrence={occurrence.id}"
            ),
        }
        for occurrence, active_activity_id in rows
    ]
