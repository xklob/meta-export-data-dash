from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from meta_archive.models import (
    ConflictCalibration,
    ConversationCorrection,
    DeletionOperation,
    ImportRun,
    Job,
)
from meta_archive.paths import (
    ArchiveLayout,
    UnsafeArchivePathError,
    create_directory_beneath,
    open_directory_beneath,
)

CACHE_VERSION = "analytics-response-v2"
_REVISION_JOB_KINDS = frozenset(
    {
        "activity_import",
        "conflict_aggregates",
        "messages",
        "nlp_annotations",
    }
)


class StaleAnalyticsPayloadError(RuntimeError):
    """The catalog changed after an analytics response began computing."""


_ANALYTICS_CACHE_LOCK_NAME = "analytics-cache.lock"


def _validate_analytics_cache_lock(
    descriptor: int,
    directory_descriptor: int,
    *,
    require_private_mode: bool,
) -> None:
    descriptor_stat = os.fstat(descriptor)
    path_stat = os.stat(
        _ANALYTICS_CACHE_LOCK_NAME,
        dir_fd=directory_descriptor,
        follow_symlinks=False,
    )
    if (
        not stat.S_ISREG(descriptor_stat.st_mode)
        or descriptor_stat.st_uid != os.geteuid()
        or descriptor_stat.st_nlink != 1
        or not stat.S_ISREG(path_stat.st_mode)
        or path_stat.st_uid != os.geteuid()
        or path_stat.st_nlink != 1
        or (descriptor_stat.st_dev, descriptor_stat.st_ino) != (path_stat.st_dev, path_stat.st_ino)
        or (
            require_private_mode
            and (
                stat.S_IMODE(descriptor_stat.st_mode) != 0o600
                or stat.S_IMODE(path_stat.st_mode) != 0o600
            )
        )
    ):
        raise OSError("Analytics cache lock is not one safe owner-only regular file")


def _open_analytics_cache_lock(layout: ArchiveLayout) -> tuple[int, int]:
    if not all(hasattr(os, flag) for flag in ("O_DIRECTORY", "O_NOFOLLOW", "O_CLOEXEC")):
        raise OSError("Secure no-follow analytics cache locks are unavailable")
    directory_descriptor: int | None = None
    descriptor: int | None = None
    try:
        create_directory_beneath(layout.catalog, layout.root, mode=0o700)
        directory_descriptor = open_directory_beneath(layout.catalog, layout.root)
        descriptor = os.open(
            _ANALYTICS_CACHE_LOCK_NAME,
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
            0o600,
            dir_fd=directory_descriptor,
        )
        _validate_analytics_cache_lock(
            descriptor,
            directory_descriptor,
            require_private_mode=False,
        )
        # Repair only the already validated single-link inode. A hardlinked outside file must be
        # rejected above before this function can change its permissions.
        os.fchmod(descriptor, 0o600)
        _validate_analytics_cache_lock(
            descriptor,
            directory_descriptor,
            require_private_mode=True,
        )
        return descriptor, directory_descriptor
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        if directory_descriptor is not None:
            os.close(directory_descriptor)
        raise


@contextmanager
def analytics_cache_lock(layout: ArchiveLayout, *, exclusive: bool) -> Iterator[None]:
    """Coordinate cache publication with irreversible purge filesystem cleanup."""

    descriptor: int | None = None
    directory_descriptor: int | None = None
    acquired = False
    try:
        try:
            descriptor, directory_descriptor = _open_analytics_cache_lock(layout)
        except (OSError, UnsafeArchivePathError) as exc:
            raise OSError("Could not securely open the analytics cache lock") from exc
        fcntl.flock(descriptor, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        acquired = True
        try:
            _validate_analytics_cache_lock(
                descriptor,
                directory_descriptor,
                require_private_mode=True,
            )
        except OSError as exc:
            raise OSError("The analytics cache lock path changed while acquiring it") from exc
        yield
    finally:
        try:
            if acquired and descriptor is not None:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if directory_descriptor is not None:
                os.close(directory_descriptor)


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def analytics_source_revision(layout: ArchiveLayout, engine: Engine) -> str:
    """Return a cheap revision for every source that can change cached aggregate responses."""

    marker = layout.library_completion_marker
    marker_payload: dict[str, Any] | None = None
    if marker.is_file() and not marker.is_symlink():
        stat = marker.stat(follow_symlinks=False)
        marker_payload = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": hashlib.sha256(marker.read_bytes()).hexdigest(),
        }
    with Session(engine) as session:
        import_runs = [
            (row.id, row.kind, row.status, row.finished_at, row.counters_json)
            for row in session.scalars(
                select(ImportRun)
                .where(ImportRun.kind.in_(("messages", "activity")))
                .order_by(ImportRun.started_at, ImportRun.id)
            )
        ]
        jobs = [
            (row.id, row.kind, row.status, row.updated_at, row.counters_json)
            for row in session.scalars(
                select(Job)
                .where(Job.kind.in_(_REVISION_JOB_KINDS))
                .order_by(Job.created_at, Job.id)
            )
        ]
        corrections = [
            (row.id, row.status, row.result_revision, row.revert_revision, row.reverted_at)
            for row in session.scalars(
                select(ConversationCorrection).order_by(
                    ConversationCorrection.created_at, ConversationCorrection.id
                )
            )
        ]
        deletions = [
            (
                row.id,
                row.status,
                row.updated_at,
                row.selection_sha256,
                row.purge_revision,
            )
            for row in session.scalars(
                select(DeletionOperation).order_by(
                    DeletionOperation.created_at, DeletionOperation.id
                )
            )
        ]
        calibrations = [
            (
                row.id,
                row.status,
                row.completed_at,
                row.aggregation_enabled,
                row.sample_count,
                row.direct_conflict_precision,
                row.macro_f1,
            )
            for row in session.scalars(
                select(ConflictCalibration).order_by(
                    ConflictCalibration.created_at, ConflictCalibration.id
                )
            )
        ]
    payload = {
        "cache_version": CACHE_VERSION,
        "library_marker": marker_payload,
        "import_runs": import_runs,
        "jobs": jobs,
        "conversation_corrections": corrections,
        "deletions": deletions,
        "conflict_calibrations": calibrations,
    }
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def _cache_path(layout: ArchiveLayout, namespace: str, parameters: Mapping[str, object]) -> Path:
    identity = hashlib.sha256(
        _json_bytes({"namespace": namespace, "parameters": dict(parameters)})
    ).hexdigest()
    return layout.analytics_cache / f"{identity}.json"


def read_analytics_cache(
    layout: ArchiveLayout,
    engine: Engine,
    *,
    namespace: str,
    parameters: Mapping[str, object],
) -> dict[str, Any] | None:
    with analytics_cache_lock(layout, exclusive=False):
        path = _cache_path(layout, namespace, parameters)
        if path.is_symlink() or not path.is_file():
            return None
        try:
            document: Any = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(document, dict) or document.get("version") != CACHE_VERSION:
            return None
        if document.get("source_revision") != analytics_source_revision(layout, engine):
            return None
        payload = document.get("payload")
        return payload if isinstance(payload, dict) else None


def write_analytics_cache(
    layout: ArchiveLayout,
    engine: Engine,
    *,
    namespace: str,
    parameters: Mapping[str, object],
    payload: dict[str, Any],
    expected_source_revision: str | None = None,
) -> dict[str, Any]:
    with analytics_cache_lock(layout, exclusive=False):
        current_revision = analytics_source_revision(layout, engine)
        if expected_source_revision is not None and current_revision != expected_source_revision:
            raise StaleAnalyticsPayloadError(
                "Archive data changed while analytics were computed; retry the request"
            )
        layout.analytics_cache.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = _cache_path(layout, namespace, parameters)
        document = {
            "version": CACHE_VERSION,
            "source_revision": current_revision,
            "namespace": namespace,
            "parameters": dict(parameters),
            "generated_at": datetime.now(UTC).isoformat(),
            "payload": payload,
        }
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(_json_bytes(document))
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    return payload


def clear_analytics_cache(layout: ArchiveLayout) -> int:
    with analytics_cache_lock(layout, exclusive=True):
        root = layout.analytics_cache
        if not root.exists():
            return 0
        if root.is_symlink() or not root.is_dir():
            raise RuntimeError(f"Analytics cache path is unsafe: {root}")
        removed = 0
        for path in root.iterdir():
            if path.is_symlink() or not path.is_file() or path.suffix != ".json":
                raise RuntimeError(f"Unexpected analytics cache entry: {path}")
            path.unlink()
            removed += 1
        return removed
