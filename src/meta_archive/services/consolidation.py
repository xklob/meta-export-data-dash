from __future__ import annotations

import fcntl
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from sqlalchemy import Engine, func, select, update
from sqlalchemy.orm import Session

from meta_archive.models import Blob, SourceChunk, SourceFile
from meta_archive.paths import (
    ArchiveLayout,
    UnsafeArchivePathError,
    archive_entry_stat_beneath,
    create_directory_beneath,
    open_directory_beneath,
)
from meta_archive.services.hashing import sha256_file
from meta_archive.services.identifiers import canonical_fingerprint, stable_id
from meta_archive.services.source_import_lock import (
    SourceImportPublicationLease,
    assert_source_import_publication_allowed,
    source_import_publication_lock,
)

ProgressCallback = Callable[[str, int, int | None], None]
READ_ONLY_BLOB_MODE = 0o444
PRIVATE_READ_ONLY_MODE = 0o400
CONFLICT_HOLDING_LOCK_NAME = "conflict-holding.lock"


class ConflictHoldingLockError(RuntimeError):
    """The conflict holding area cannot be mutated under its archive-wide lock."""


class ConflictHoldingLease:
    """Proof that this process still owns the conflict-holding advisory lock.

    The descriptor is intentionally retained by :func:`conflict_holding_lock`; callers receive
    only this lease.  A lease cannot be reused after its context exits or in a forked process.
    """

    __slots__ = ("_active", "_descriptor", "_device", "_inode", "_pid", "_root")

    def __init__(
        self,
        *,
        root: Path,
        descriptor: int,
        device: int,
        inode: int,
    ) -> None:
        self._root = Path(os.path.abspath(root))
        self._descriptor = descriptor
        self._device = device
        self._inode = inode
        self._pid = os.getpid()
        self._active = True

    def _deactivate(self) -> None:
        self._active = False

    def assert_held(self, layout: ArchiveLayout) -> None:
        """Fail closed unless the live descriptor and fixed lock pathname still agree."""

        if (
            not self._active
            or self._pid != os.getpid()
            or self._root != Path(os.path.abspath(layout.root))
        ):
            raise ConflictHoldingLockError(
                "A live conflict-holding lock lease for this archive is required"
            )
        try:
            descriptor_stat = os.fstat(self._descriptor)
            path_stat = archive_entry_stat_beneath(
                layout.catalog / CONFLICT_HOLDING_LOCK_NAME,
                layout.root,
            )
        except (OSError, UnsafeArchivePathError) as exc:
            raise ConflictHoldingLockError(
                "The conflict-holding lock path became unsafe while held"
            ) from exc
        if (
            not stat.S_ISREG(descriptor_stat.st_mode)
            or descriptor_stat.st_uid != os.geteuid()
            or descriptor_stat.st_nlink != 1
            or path_stat is None
            or not stat.S_ISREG(path_stat.st_mode)
            or (descriptor_stat.st_dev, descriptor_stat.st_ino)
            != (path_stat.st_dev, path_stat.st_ino)
            or (descriptor_stat.st_dev, descriptor_stat.st_ino) != (self._device, self._inode)
            or stat.S_IMODE(descriptor_stat.st_mode) != 0o600
        ):
            raise ConflictHoldingLockError(
                "The conflict-holding lock file is no longer a safe owner-only regular file"
            )


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("Could not write conflict-holding lock metadata")
        view = view[written:]


@contextmanager
def conflict_holding_lock(
    layout: ArchiveLayout,
    *,
    blocking: bool = False,
) -> Iterator[ConflictHoldingLease]:
    """Take the owner-only lock that serializes every conflict-tree mutation.

    The lock lives in ``catalog`` so a hard purge may stage or remove the entire ``conflicts``
    tree without changing the inode on which concurrent materializers synchronize.  Callers that
    also need purge locks must use the global prefix ``deletion -> source import/publication ->
    library -> conflict holding``; code holding this lock must never acquire the source or library
    lock.
    """

    lock_path = layout.catalog / CONFLICT_HOLDING_LOCK_NAME
    catalog_descriptor: int | None = None
    descriptor: int | None = None
    acquired = False
    lease: ConflictHoldingLease | None = None
    try:
        try:
            if not all(hasattr(os, name) for name in ("O_CLOEXEC", "O_NOFOLLOW")):
                raise ConflictHoldingLockError(
                    "Secure no-follow conflict-holding locks are unavailable"
                )
            # Prepare only the lock's own directory.  The broad runtime-permission repair scans
            # all ``*.lock`` entries and could chmod an external inode before this function has a
            # chance to reject an unsafe hardlinked lock file.
            create_directory_beneath(layout.catalog, layout.root, mode=0o700)
            catalog_descriptor = open_directory_beneath(layout.catalog, layout.root)
            descriptor = os.open(
                CONFLICT_HOLDING_LOCK_NAME,
                os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
                0o600,
                dir_fd=catalog_descriptor,
            )
            descriptor_stat = os.fstat(descriptor)
            path_stat = os.stat(
                CONFLICT_HOLDING_LOCK_NAME,
                dir_fd=catalog_descriptor,
                follow_symlinks=False,
            )
        except ConflictHoldingLockError:
            raise
        except (OSError, UnsafeArchivePathError) as exc:
            raise ConflictHoldingLockError(
                f"Unsafe conflict-holding lock state: {lock_path}"
            ) from exc

        if (
            not stat.S_ISREG(descriptor_stat.st_mode)
            or descriptor_stat.st_uid != os.geteuid()
            or descriptor_stat.st_nlink != 1
            or (descriptor_stat.st_dev, descriptor_stat.st_ino)
            != (path_stat.st_dev, path_stat.st_ino)
        ):
            raise ConflictHoldingLockError(
                "The conflict-holding lock must be one owner-owned regular file"
            )
        try:
            os.fchmod(descriptor, 0o600)
            lock_flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
            fcntl.flock(descriptor, lock_flags)
            acquired = True
            os.ftruncate(descriptor, 0)
            os.lseek(descriptor, 0, os.SEEK_SET)
            _write_all(
                descriptor,
                (json.dumps({"pid": os.getpid()}, sort_keys=True) + "\n").encode("utf-8"),
            )
            os.fsync(descriptor)
        except BlockingIOError as exc:
            raise ConflictHoldingLockError(
                "Another operation is currently mutating the conflict holding area"
            ) from exc
        except OSError as exc:
            raise ConflictHoldingLockError("Could not establish the conflict-holding lock") from exc

        lease = ConflictHoldingLease(
            root=layout.root,
            descriptor=descriptor,
            device=descriptor_stat.st_dev,
            inode=descriptor_stat.st_ino,
        )
        lease.assert_held(layout)
        try:
            yield lease
        except BaseException:
            raise
        else:
            # A successful destructive caller must not silently commit if the fixed lock path was
            # unlinked or replaced while it worked.  Error paths preserve their original failure.
            lease.assert_held(layout)
    finally:
        if lease is not None:
            lease._deactivate()
        try:
            if acquired and descriptor is not None:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            try:
                if descriptor is not None:
                    os.close(descriptor)
            finally:
                if catalog_descriptor is not None:
                    os.close(catalog_descriptor)


@dataclass(frozen=True, slots=True)
class ConflictHoldingSummary:
    group_count: int
    copy_count: int
    materialized_copy_count: int
    reused_copy_count: int
    manifest_count: int
    holding_root: str


def _regular_file_stat(path: Path) -> os.stat_result | None:
    """Return an lstat result only for a non-symlink regular file."""

    try:
        result = path.lstat()
    except FileNotFoundError:
        return None
    return result if stat.S_ISREG(result.st_mode) else None


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _fsync_directory(path: Path) -> None:
    """Persist a rename where the platform exposes directory fsync."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy_to_temp(source: Path, destination: Path) -> Path:
    """Copy to a same-directory temporary file without inheriting source metadata."""

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        # Open the already-created destination descriptor first so it is always owned by a
        # context manager even when opening the source fails.
        with (
            os.fdopen(descriptor, "wb", buffering=0) as destination_handle,
            source.open("rb", buffering=0) as source_handle,
        ):
            shutil.copyfileobj(source_handle, destination_handle, length=8 * 1024 * 1024)
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
    except BaseException:
        # os.fdopen owns the descriptor once entered. Closing an already-closed descriptor is
        # harmlessly avoided by letting that context manager handle it.
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _verify_materialized_bytes(path: Path, *, sha256: str, size_bytes: int) -> None:
    path_stat = _regular_file_stat(path)
    if path_stat is None:
        raise RuntimeError(f"Expected a regular materialized file: {path}")
    if path_stat.st_size != size_bytes or sha256_file(path) != sha256:
        raise RuntimeError(f"Materialized bytes do not match {sha256}: {path}")


def _publish_blob_copy(
    source: Path,
    destination: Path,
    *,
    sha256: str,
    size_bytes: int,
) -> None:
    """Publish an independently allocated, verified, read-only blob atomically."""

    temporary = _copy_to_temp(source, destination)
    try:
        # Verification intentionally happens after the copy and before publication. chmod is
        # likewise applied to the independent inode, never to an original source inode.
        _verify_materialized_bytes(temporary, sha256=sha256, size_bytes=size_bytes)
        os.chmod(temporary, READ_ONLY_BLOB_MODE)
        with temporary.open("rb", buffering=0) as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
        published_stat = _regular_file_stat(destination)
        if published_stat is None or stat.S_IMODE(published_stat.st_mode) & 0o222:
            raise RuntimeError(f"Published blob is not an immutable regular file: {destination}")
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _safe_manifest_path(value: object, *, field: str, allow_empty: bool = False) -> Path:
    if not isinstance(value, str):
        raise RuntimeError(f"Manifest field {field!r} must be a string")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or ".." in pure.parts
        or "\x00" in value
        or (not allow_empty and value in {"", "."})
    ):
        raise RuntimeError(f"Manifest field {field!r} is not a safe relative path: {value!r}")
    return Path(*(() if value in {"", "."} else pure.parts))


def _ensure_safe_directory(path: Path) -> None:
    chain: list[Path] = []
    current = path
    while True:
        chain.append(current)
        if current.parent == current:
            break
        current = current.parent
    for current in reversed(chain):
        try:
            current_stat = current.lstat()
        except FileNotFoundError:
            with suppress(FileExistsError):
                # Create one already-parent-validated component at a time. `parents=True` would
                # follow a symlinked holding-area parent before it could be rejected.
                current.mkdir()
            current_stat = current.lstat()
        if not stat.S_ISDIR(current_stat.st_mode):
            raise RuntimeError(f"Conflict holding path is not a safe directory: {current}")


def _validated_source_path(
    layout: ArchiveLayout,
    *,
    chunk_path: str,
    relative_path: str,
) -> Path:
    chunk = _safe_manifest_path(chunk_path, field="source chunk")
    relative = _safe_manifest_path(relative_path, field="source relative path")
    current = layout.root
    for part in (*chunk.parts, *relative.parts[:-1]):
        current /= part
        try:
            current_stat = current.lstat()
        except FileNotFoundError as exc:
            raise RuntimeError(f"Conflict source parent is missing: {current}") from exc
        if not stat.S_ISDIR(current_stat.st_mode):
            raise RuntimeError(f"Conflict source parent is not a safe directory: {current}")
    source = current / relative.name
    if _regular_file_stat(source) is None:
        raise RuntimeError(f"Conflict source is not a non-symlink regular file: {source}")
    return source


def _verify_read_only_independent_copy(
    source: Path,
    destination: Path,
    *,
    sha256: str,
    size_bytes: int,
) -> None:
    _verify_materialized_bytes(destination, sha256=sha256, size_bytes=size_bytes)
    destination_stat = destination.lstat()
    source_stat = source.lstat()
    if stat.S_IMODE(destination_stat.st_mode) & 0o222:
        raise RuntimeError(f"Conflict holding copy is writable: {destination}")
    if _same_inode(source_stat, destination_stat):
        raise RuntimeError(f"Conflict holding copy shares its source inode: {destination}")


def _publish_append_only_copy(
    source: Path,
    destination: Path,
    *,
    sha256: str,
    size_bytes: int,
) -> bool:
    """Create a verified conflict copy without ever replacing an existing path."""

    if destination.exists() or destination.is_symlink():
        _verify_read_only_independent_copy(
            source,
            destination,
            sha256=sha256,
            size_bytes=size_bytes,
        )
        return False

    temporary = _copy_to_temp(source, destination)
    try:
        _verify_materialized_bytes(temporary, sha256=sha256, size_bytes=size_bytes)
        os.chmod(temporary, READ_ONLY_BLOB_MODE)
        with temporary.open("rb", buffering=0) as handle:
            os.fsync(handle.fileno())
        try:
            # Linking the independently copied temporary inode is an atomic create-if-absent.
            # Unlike replace(), it can never overwrite a previously held conflict copy.
            os.link(temporary, destination, follow_symlinks=False)
            created = True
            _fsync_directory(destination.parent)
        except FileExistsError:
            created = False
    finally:
        temporary.unlink(missing_ok=True)
    _verify_read_only_independent_copy(
        source,
        destination,
        sha256=sha256,
        size_bytes=size_bytes,
    )
    return created


def _publish_append_only_manifest(path: Path, payload: dict[str, Any]) -> bool:
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    if path.exists() or path.is_symlink():
        if _regular_file_stat(path) is None or path.read_bytes() != encoded:
            raise RuntimeError(f"Conflict manifest already exists with different content: {path}")
        if stat.S_IMODE(path.lstat().st_mode) & 0o222:
            raise RuntimeError(f"Conflict manifest is writable: {path}")
        return False

    descriptor, temporary_name = tempfile.mkstemp(prefix=".manifest.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb", buffering=0) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, PRIVATE_READ_ONLY_MODE)
        try:
            os.link(temporary, path, follow_symlinks=False)
            created = True
            _fsync_directory(path.parent)
        except FileExistsError:
            created = False
    finally:
        temporary.unlink(missing_ok=True)
    if _regular_file_stat(path) is None or path.read_bytes() != encoded:
        raise RuntimeError(f"Conflict manifest publication failed validation: {path}")
    if stat.S_IMODE(path.lstat().st_mode) & 0o222:
        raise RuntimeError(f"Conflict manifest is writable: {path}")
    return created


def _materialize_conflict_holding_area(
    layout: ArchiveLayout,
    engine: Engine,
    *,
    progress: ProgressCallback | None = None,
) -> ConflictHoldingSummary:
    """Append verified physical copies for every differing-hash normalized-path group.

    Group and revision paths are derived only from stable hashes. A changed group receives a new
    revision directory; an existing held copy or manifest is validated and reused, never replaced.
    """

    conflicting_paths = (
        select(SourceFile.normalized_path)
        .where(SourceFile.scan_status == "discovered", SourceFile.sha256.is_not(None))
        .group_by(SourceFile.normalized_path)
        .having(func.count(func.distinct(SourceFile.sha256)) > 1)
    )
    with Session(engine) as session:
        rows = session.execute(
            select(
                SourceFile.id,
                SourceFile.normalized_path,
                SourceFile.relative_path,
                SourceFile.sha256,
                SourceFile.size_bytes,
                SourceChunk.chunk_key,
                SourceChunk.original_path,
            )
            .join(SourceChunk, SourceChunk.id == SourceFile.chunk_id)
            .where(SourceFile.normalized_path.in_(conflicting_paths))
            .where(SourceFile.scan_status == "discovered", SourceFile.sha256.is_not(None))
            .order_by(SourceFile.normalized_path, SourceChunk.ordinal, SourceFile.id)
        ).all()

    groups: dict[str, list[tuple[int, str, str, int, str, str]]] = {}
    for (
        source_id,
        normalized_path,
        relative_path,
        digest,
        size_bytes,
        chunk_key,
        chunk_path,
    ) in rows:
        if not isinstance(digest, str) or len(digest) != 64:
            raise RuntimeError(f"Conflict source {source_id} lacks a valid SHA-256")
        groups.setdefault(normalized_path, []).append(
            (source_id, relative_path, digest, size_bytes, chunk_key, chunk_path)
        )
    if not groups:
        return ConflictHoldingSummary(0, 0, 0, 0, 0, "conflicts/normalized-path")

    holding_root = layout.conflicts / "normalized-path"
    _ensure_safe_directory(holding_root)
    copy_count = 0
    materialized = 0
    reused = 0
    manifest_count = 0
    total = sum(len(copies) for copies in groups.values())

    for normalized_path, copies in groups.items():
        provenance = [
            {
                "source_file_id": source_id,
                "source_chunk": chunk_key,
                "source_chunk_path": chunk_path,
                "source_relative_path": relative_path,
                "sha256": digest,
                "size_bytes": size_bytes,
            }
            for source_id, relative_path, digest, size_bytes, chunk_key, chunk_path in copies
        ]
        group_id = stable_id("normalized-path-conflict", normalized_path)
        revision = canonical_fingerprint(
            "normalized-path-conflict-revision",
            {"normalized_path": normalized_path, "copies": provenance},
        )
        revision_root = holding_root / group_id / revision
        copies_root = revision_root / "copies"
        _ensure_safe_directory(copies_root)
        manifest_copies: list[dict[str, Any]] = []
        for ordinal, (
            (source_id, relative_path, digest, size_bytes, _chunk_key, chunk_path),
            record,
        ) in enumerate(zip(copies, provenance, strict=True), start=1):
            source = _validated_source_path(
                layout,
                chunk_path=chunk_path,
                relative_path=relative_path,
            )
            if source.lstat().st_size != size_bytes or sha256_file(source) != digest:
                raise RuntimeError(f"Conflict source bytes changed after inventory: {source}")
            filename = f"{ordinal:04d}--source-{source_id}--{digest}"
            destination = copies_root / filename
            created = _publish_append_only_copy(
                source,
                destination,
                sha256=digest,
                size_bytes=size_bytes,
            )
            materialized += int(created)
            reused += int(not created)
            copy_count += 1
            record["holding_path"] = destination.relative_to(layout.root).as_posix()
            manifest_copies.append(record)
            if progress and (copy_count % 250 == 0 or copy_count == total):
                progress("Holding conflicting source copies", copy_count, total)

        manifest = {
            "schema_version": 1,
            "kind": "normalized-path-byte-conflict",
            "group_id": group_id,
            "revision": revision,
            "normalized_path": normalized_path,
            "copy_count": len(manifest_copies),
            "distinct_sha256_count": len({record["sha256"] for record in manifest_copies}),
            "copies": manifest_copies,
        }
        _publish_append_only_manifest(revision_root / "manifest.json", manifest)
        manifest_count += 1

    return ConflictHoldingSummary(
        group_count=len(groups),
        copy_count=copy_count,
        materialized_copy_count=materialized,
        reused_copy_count=reused,
        manifest_count=manifest_count,
        holding_root=holding_root.relative_to(layout.root).as_posix(),
    )


def materialize_conflict_holding_area_locked(
    layout: ArchiveLayout,
    engine: Engine,
    *,
    source_lock: SourceImportPublicationLease,
    lock: ConflictHoldingLease,
    progress: ProgressCallback | None = None,
) -> ConflictHoldingSummary:
    """Materialize conflicts while reusing explicit source and conflict leases.

    This entry point lets a hard-purge transaction retain the global lock order instead of trying
    to reacquire the non-reentrant file lock.  The lease is checked both before and after a
    successful build so a replaced or unlinked lock pathname fails closed.
    """

    source_lock.assert_held(layout)
    lock.assert_held(layout)
    assert_source_import_publication_allowed(
        layout,
        engine,
        lease=source_lock,
        publisher="Conflict holding publication",
    )
    summary = _materialize_conflict_holding_area(layout, engine, progress=progress)
    lock.assert_held(layout)
    source_lock.assert_held(layout)
    return summary


def materialize_conflict_holding_area(
    layout: ArchiveLayout,
    engine: Engine,
    *,
    progress: ProgressCallback | None = None,
) -> ConflictHoldingSummary:
    """Materialize conflict copies under the global source-then-conflict lock order."""

    with (
        source_import_publication_lock(layout) as source_lock,
        conflict_holding_lock(layout) as lock,
    ):
        return materialize_conflict_holding_area_locked(
            layout,
            engine,
            source_lock=source_lock,
            lock=lock,
            progress=progress,
        )


def _write_jsonl_atomic(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def build_blob_store_locked(
    layout: ArchiveLayout,
    engine: Engine,
    *,
    lock: SourceImportPublicationLease,
    progress: ProgressCallback | None = None,
) -> tuple[int, int]:
    """Materialize the blob store while reusing a validated outer source lease.

    Older releases hardlinked blobs directly to source entries. If such a link is found, this
    function atomically replaces the blob path with a copied inode before applying its read-only
    mode. Re-running after either a normal or interrupted build is idempotent.

    The first return value counts newly materialized or legacy-repaired blobs. The second counts
    already-valid independent blobs.
    """

    assert_source_import_publication_allowed(
        layout,
        engine,
        lease=lock,
        publisher="Blob-store and relocation-manifest publication",
    )
    layout.ensure_runtime_directories()
    with Session(engine) as session:
        rows = session.execute(
            select(
                Blob.sha256,
                Blob.size_bytes,
                SourceChunk.original_path,
                SourceFile.relative_path,
                SourceFile.mode,
                SourceFile.mtime_ns,
            )
            .join(SourceFile, SourceFile.sha256 == Blob.sha256)
            .join(SourceChunk, SourceChunk.id == SourceFile.chunk_id)
            .where(SourceFile.scan_status == "discovered")
            .order_by(Blob.sha256, SourceFile.id)
        ).all()

    sources_by_hash: dict[str, list[tuple[Path, str, str, int, int]]] = {}
    size_by_hash: dict[str, int] = {}
    for sha256, size_bytes, chunk_path, relative_path, mode, mtime_ns in rows:
        size_by_hash.setdefault(sha256, size_bytes)
        if size_by_hash[sha256] != size_bytes:
            raise RuntimeError(f"Catalog contains inconsistent sizes for SHA-256 {sha256}")
        sources_by_hash.setdefault(sha256, []).append(
            (
                layout.root / chunk_path / relative_path,
                chunk_path,
                relative_path,
                mode,
                mtime_ns,
            )
        )

    materialized = 0
    reused = 0
    total = len(sources_by_hash)
    for index, (sha256, source_rows) in enumerate(sources_by_hash.items(), start=1):
        size_bytes = size_by_hash[sha256]
        destination = layout.blob_path(sha256)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination_stat = _regular_file_stat(destination)
        if (destination.exists() or destination.is_symlink()) and destination_stat is None:
            raise RuntimeError(f"Blob path is not a regular file: {destination}")

        if destination_stat is not None:
            _verify_materialized_bytes(destination, sha256=sha256, size_bytes=size_bytes)

        source_stats: list[tuple[Path, os.stat_result]] = []
        for source, _chunk, _relative, _mode, _mtime in source_rows:
            source_stat = _regular_file_stat(source)
            if source_stat is not None:
                source_stats.append((source, source_stat))

        shares_source_inode = destination_stat is not None and any(
            _same_inode(destination_stat, source_stat) for _source, source_stat in source_stats
        )
        destination_is_writable = destination_stat is not None and bool(
            stat.S_IMODE(destination_stat.st_mode) & 0o222
        )
        if destination_stat is None or shares_source_inode or destination_is_writable:
            # Copy from the existing verified blob when possible. This is important for recovery
            # after source retirement; for a legacy hardlink it also breaks the shared inode.
            copy_source = destination if destination_stat is not None else None
            if copy_source is None:
                copy_source = next((path for path, _source_stat in source_stats), None)
            if copy_source is None:
                raise RuntimeError(f"No readable source exists for missing blob {sha256}")
            _publish_blob_copy(
                copy_source,
                destination,
                sha256=sha256,
                size_bytes=size_bytes,
            )
            materialized += 1
        else:
            reused += 1

        final_stat = _regular_file_stat(destination)
        if final_stat is None or stat.S_IMODE(final_stat.st_mode) & 0o222:
            raise RuntimeError(f"Blob is not read-only after publication: {destination}")
        if any(_same_inode(final_stat, source_stat) for _source, source_stat in source_stats):
            raise RuntimeError(f"Blob still shares an inode with a source entry: {destination}")
        if progress and (index % 250 == 0 or index == total):
            progress("Building independent content-addressed blob store", index, total)

    relocation_records: list[dict[str, object]] = []
    for sha256, source_rows in sources_by_hash.items():
        for _source, chunk_path, relative_path, mode, mtime_ns in source_rows:
            relocation_records.append(
                {
                    "source": f"{chunk_path}/{relative_path}",
                    "chunk": chunk_path,
                    "relative_path": relative_path,
                    "blob": layout.blob_path(sha256).relative_to(layout.root).as_posix(),
                    "sha256": sha256,
                    "size_bytes": size_by_hash[sha256],
                    "mode": mode,
                    "mtime_ns": mtime_ns,
                    "operation": "independent-copy",
                }
            )
    relocation_records.sort(key=lambda record: str(record["source"]))
    _write_jsonl_atomic(layout.relocation_manifest, relocation_records)

    # The database is updated only after every blob and the full relocation manifest have been
    # durably published. A crash before this point is safely recoverable by re-running.
    with Session(engine) as session, session.begin():
        for sha256 in sources_by_hash:
            session.execute(
                update(Blob)
                .where(Blob.sha256 == sha256)
                .values(
                    relative_path=layout.blob_path(sha256).relative_to(layout.root).as_posix(),
                    state="materialized",
                )
            )
    return materialized, reused


def build_blob_store(
    layout: ArchiveLayout,
    engine: Engine,
    *,
    progress: ProgressCallback | None = None,
) -> tuple[int, int]:
    """Materialize the blob store and relocation manifest beneath one source lock."""

    with source_import_publication_lock(layout) as lock:
        return build_blob_store_locked(layout, engine, lock=lock, progress=progress)


def _load_original_file_metadata(layout: ArchiveLayout) -> dict[str, tuple[int, int]]:
    if not layout.original_files_manifest.is_file():
        raise RuntimeError("Original-files manifest does not exist; run a hashing scan first")
    metadata: dict[str, tuple[int, int]] = {}
    with layout.original_files_manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                original_record: Any = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid original-files manifest JSON on line {line_number}"
                ) from exc
            if not isinstance(original_record, dict):
                raise RuntimeError(f"Invalid original-files manifest record on line {line_number}")
            chunk = _safe_manifest_path(original_record.get("chunk"), field="chunk")
            relative = _safe_manifest_path(
                original_record.get("relative_path"), field="relative_path"
            )
            mode = original_record.get("mode")
            mtime_ns = original_record.get("mtime_ns")
            if not isinstance(mode, int) or not isinstance(mtime_ns, int):
                raise RuntimeError("Original-files manifest contains invalid stat metadata")
            source = (chunk / relative).as_posix()
            if source in metadata:
                raise RuntimeError(f"Duplicate source coordinate in original manifest: {source}")
            metadata[source] = (mode, mtime_ns)
    return metadata


def _copy_reconstructed_file(
    source: Path,
    destination: Path,
    *,
    sha256: str,
    size_bytes: int,
    mode: int,
    mtime_ns: int,
) -> None:
    temporary = _copy_to_temp(source, destination)
    try:
        _verify_materialized_bytes(temporary, sha256=sha256, size_bytes=size_bytes)
        os.chmod(temporary, mode & 0o7777)
        os.utime(temporary, ns=(mtime_ns, mtime_ns), follow_symlinks=False)
        with temporary.open("rb", buffering=0) as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def reconstruct_source_locked(
    layout: ArchiveLayout,
    target: Path,
    *,
    lock: SourceImportPublicationLease,
    overwrite: bool = False,
) -> int:
    """Reconstruct a source tree while reusing a validated source/publication lease."""

    lock.assert_held(layout)

    if not layout.relocation_manifest.is_file():
        raise RuntimeError("Relocation manifest does not exist; run consolidation first")
    if not layout.original_directories_manifest.is_file():
        raise RuntimeError("Original-directories manifest is required for complete reconstruction")
    original_file_metadata = _load_original_file_metadata(layout)
    archive_root = layout.root.expanduser().resolve()
    target = target.expanduser().resolve()
    if target == archive_root or archive_root in target.parents or target in archive_root.parents:
        raise RuntimeError(
            "Reconstruction target must be disjoint from the live archive root "
            "(not equal to, inside, or an ancestor of it)"
        )
    target.mkdir(parents=True, exist_ok=True)

    directory_records: list[tuple[Path, int, int]] = []
    with layout.original_directories_manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                directory_record: Any = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid original-directories manifest JSON on line {line_number}"
                ) from exc
            if not isinstance(directory_record, dict):
                raise RuntimeError(
                    f"Invalid original-directories manifest record on line {line_number}"
                )
            chunk = _safe_manifest_path(directory_record.get("chunk"), field="chunk")
            relative = _safe_manifest_path(
                directory_record.get("relative_path"),
                field="relative_path",
                allow_empty=True,
            )
            mode = directory_record.get("mode")
            mtime_ns = directory_record.get("mtime_ns")
            if not isinstance(mode, int) or not isinstance(mtime_ns, int):
                raise RuntimeError("Directory manifest contains invalid stat metadata")
            destination = target / chunk / relative
            if destination.is_symlink():
                raise RuntimeError(f"Refusing to reconstruct through a symlink: {destination}")
            destination.mkdir(parents=True, exist_ok=True)
            directory_records.append((destination, mode, mtime_ns))

    reconstructed = 0
    verified_blobs: set[str] = set()
    with layout.relocation_manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                relocation_record: Any = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid relocation manifest JSON on line {line_number}"
                ) from exc
            if not isinstance(relocation_record, dict):
                raise RuntimeError(f"Invalid relocation manifest record on line {line_number}")
            source_relative = _safe_manifest_path(relocation_record.get("source"), field="source")
            sha256 = relocation_record.get("sha256")
            size_bytes = relocation_record.get("size_bytes")
            if not isinstance(sha256, str) or len(sha256) != 64 or not isinstance(size_bytes, int):
                raise RuntimeError("Relocation manifest contains an invalid blob identity")
            expected_blob = layout.blob_path(sha256)
            blob_relative = _safe_manifest_path(relocation_record.get("blob"), field="blob")
            blob = layout.root / blob_relative
            if blob != expected_blob:
                raise RuntimeError(f"Relocation blob path does not match its digest: {blob}")
            if sha256 not in verified_blobs:
                _verify_materialized_bytes(blob, sha256=sha256, size_bytes=size_bytes)
                verified_blobs.add(sha256)

            source_key = source_relative.as_posix()
            fallback = original_file_metadata.get(source_key)
            mode = relocation_record.get("mode", fallback[0] if fallback else None)
            mtime_ns = relocation_record.get("mtime_ns", fallback[1] if fallback else None)
            if not isinstance(mode, int) or not isinstance(mtime_ns, int):
                raise RuntimeError(f"No valid stat metadata exists for {source_key}")
            destination = target / source_relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() or destination.is_symlink():
                if not overwrite:
                    raise FileExistsError(f"Refusing to overwrite {destination}")
                if destination.is_dir() and not destination.is_symlink():
                    raise IsADirectoryError(destination)
            _copy_reconstructed_file(
                blob,
                destination,
                sha256=sha256,
                size_bytes=size_bytes,
                mode=mode,
                mtime_ns=mtime_ns,
            )
            reconstructed += 1

    # Apply directory metadata last because creating children necessarily mutates parent mtimes.
    for destination, mode, mtime_ns in reversed(directory_records):
        os.chmod(destination, mode & 0o7777)
        os.utime(destination, ns=(mtime_ns, mtime_ns), follow_symlinks=False)
    lock.assert_held(layout)
    return reconstructed


def reconstruct_source(layout: ArchiveLayout, target: Path, *, overwrite: bool = False) -> int:
    """Recreate one consistent original source snapshot from manifests and blobs."""

    with source_import_publication_lock(layout) as lock:
        return reconstruct_source_locked(
            layout,
            target,
            lock=lock,
            overwrite=overwrite,
        )
