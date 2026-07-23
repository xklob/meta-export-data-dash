from __future__ import annotations

import fcntl
import json
import os
import stat
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from meta_archive.models import DeletionOperation
from meta_archive.paths import (
    ArchiveLayout,
    UnsafeArchivePathError,
    archive_entry_stat_beneath,
    create_directory_beneath,
    open_directory_beneath,
)

SOURCE_IMPORT_PUBLICATION_LOCK_NAME = "source-import-publication.lock"
SOURCE_PUBLICATION_BLOCKING_DELETION_STATUSES = frozenset(
    {"committed", "cleanup_pending", "normalized_purged"}
)
_ACTIVE_LEASES: dict[int, object] = {}
_ACTIVE_LEASES_GUARD = threading.Lock()


class SourceImportPublicationLockError(RuntimeError):
    """Source/catalog publication could not be serialized safely."""


class SourceImportPublicationBusyError(SourceImportPublicationLockError):
    """Another process currently owns the source/import publication lock."""


class SourceImportPublicationBlockedError(SourceImportPublicationLockError):
    """A retained deletion state makes source-derived publication unsafe."""

    def __init__(self, *, publisher: str, operation_id: str, status: str) -> None:
        super().__init__(
            f"{publisher} is unavailable while deletion {operation_id} is {status}; "
            "restore it or finish its workspace purge first"
        )
        self.publisher = publisher
        self.operation_id = operation_id
        self.status = status


class SourceImportPublicationLease:
    """Unforgeable process-local proof of a live source/publication lock.

    A caller may pass this lease to an explicit ``*_locked`` service entry point to compose
    several operations beneath one outer lock without reacquiring ``flock`` on another open-file
    description. The lease becomes invalid when its context exits and is never valid in a forked
    child.
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
        """Fail closed unless this process still owns the fixed safe lock inode."""

        with _ACTIVE_LEASES_GUARD:
            issued_and_active = _ACTIVE_LEASES.get(id(self)) is self
        if (
            not issued_and_active
            or not self._active
            or self._pid != os.getpid()
            or self._root != Path(os.path.abspath(layout.root))
        ):
            raise SourceImportPublicationLockError(
                "A live source/import publication lock lease for this archive is required"
            )
        try:
            descriptor_stat = os.fstat(self._descriptor)
            path_stat = archive_entry_stat_beneath(
                layout.catalog / SOURCE_IMPORT_PUBLICATION_LOCK_NAME,
                layout.root,
            )
        except (OSError, UnsafeArchivePathError) as exc:
            raise SourceImportPublicationLockError(
                "The source/import publication lock path became unsafe while held"
            ) from exc
        if (
            not stat.S_ISREG(descriptor_stat.st_mode)
            or descriptor_stat.st_uid != os.geteuid()
            or descriptor_stat.st_nlink != 1
            or stat.S_IMODE(descriptor_stat.st_mode) != 0o600
            or path_stat is None
            or not stat.S_ISREG(path_stat.st_mode)
            or path_stat.st_uid != os.geteuid()
            or path_stat.st_nlink != 1
            or stat.S_IMODE(path_stat.st_mode) != 0o600
            or (descriptor_stat.st_dev, descriptor_stat.st_ino)
            != (path_stat.st_dev, path_stat.st_ino)
            or (descriptor_stat.st_dev, descriptor_stat.st_ino) != (self._device, self._inode)
        ):
            raise SourceImportPublicationLockError(
                "The source/import publication lock is no longer one safe owner-only file"
            )


def _write_all(descriptor: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("Could not write source/import publication lock metadata")
        remaining = remaining[written:]


def assert_source_import_publication_allowed(
    layout: ArchiveLayout,
    engine: Engine,
    *,
    lease: SourceImportPublicationLease,
    publisher: str,
) -> None:
    """Reject a sequential re-import or manifest rewrite that could resurrect purged rows.

    The source/publication lease makes this status check atomic with deletion state transitions,
    provided deletion takes the same lock immediately after its own deletion lock. A committed
    quarantine can be restored exactly, ``cleanup_pending`` still owns durable finalization, and
    ``normalized_purged`` retains a delayed workspace-upgrade promise. All three states therefore
    block source-derived publication. A completed workspace purge is safe because its raw source
    and manifests already contain the durable purge representation.
    """

    lease.assert_held(layout)
    with Session(engine) as session:
        blocked = session.execute(
            select(DeletionOperation.id, DeletionOperation.status)
            .where(DeletionOperation.status.in_(SOURCE_PUBLICATION_BLOCKING_DELETION_STATUSES))
            .order_by(DeletionOperation.created_at, DeletionOperation.id)
            .limit(1)
        ).one_or_none()
    if blocked is not None:
        operation_id, status = blocked
        raise SourceImportPublicationBlockedError(
            publisher=publisher,
            operation_id=str(operation_id),
            status=str(status),
        )


@contextmanager
def source_import_publication_lock(
    layout: ArchiveLayout,
    *,
    blocking: bool = True,
) -> Iterator[SourceImportPublicationLease]:
    """Serialize every raw-source read and source/catalog/manifest publication.

    Composite destructive workflows must acquire locks in this global order::

        deletion -> source import/publication -> library -> conflict holding
        -> worker -> semantic -> analytics -> SQLite writer transaction

    Standalone import and consolidation commands normally block. Destructive callers should pass
    ``blocking=False`` while already holding their deletion lock, then pass the yielded lease to
    any explicit ``*_locked`` helper instead of trying to acquire this lock a second time.
    """

    lock_path = layout.catalog / SOURCE_IMPORT_PUBLICATION_LOCK_NAME
    catalog_descriptor: int | None = None
    descriptor: int | None = None
    acquired = False
    lease: SourceImportPublicationLease | None = None
    try:
        try:
            if not all(hasattr(os, name) for name in ("O_CLOEXEC", "O_NOFOLLOW")):
                raise SourceImportPublicationLockError(
                    "Secure no-follow source/import publication locks are unavailable"
                )
            create_directory_beneath(layout.catalog, layout.root, mode=0o700)
            catalog_descriptor = open_directory_beneath(layout.catalog, layout.root)
            descriptor = os.open(
                SOURCE_IMPORT_PUBLICATION_LOCK_NAME,
                os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
                0o600,
                dir_fd=catalog_descriptor,
            )
            descriptor_stat = os.fstat(descriptor)
            path_stat = os.stat(
                SOURCE_IMPORT_PUBLICATION_LOCK_NAME,
                dir_fd=catalog_descriptor,
                follow_symlinks=False,
            )
        except SourceImportPublicationLockError:
            raise
        except (OSError, UnsafeArchivePathError) as exc:
            raise SourceImportPublicationLockError(
                f"Unsafe source/import publication lock state: {lock_path}"
            ) from exc

        if (
            not stat.S_ISREG(descriptor_stat.st_mode)
            or descriptor_stat.st_uid != os.geteuid()
            or descriptor_stat.st_nlink != 1
            or stat.S_IMODE(descriptor_stat.st_mode) != 0o600
            or not stat.S_ISREG(path_stat.st_mode)
            or path_stat.st_uid != os.geteuid()
            or path_stat.st_nlink != 1
            or stat.S_IMODE(path_stat.st_mode) != 0o600
            or (descriptor_stat.st_dev, descriptor_stat.st_ino)
            != (path_stat.st_dev, path_stat.st_ino)
        ):
            raise SourceImportPublicationLockError(
                "The source/import publication lock must be one owner-only regular file"
            )

        try:
            operation = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
            fcntl.flock(descriptor, operation)
            acquired = True
            # Validate the fixed name again after acquiring. A process that opened the old inode
            # before a replacement must not proceed merely because that abandoned inode is free.
            lease = SourceImportPublicationLease(
                root=layout.root,
                descriptor=descriptor,
                device=descriptor_stat.st_dev,
                inode=descriptor_stat.st_ino,
            )
            with _ACTIVE_LEASES_GUARD:
                _ACTIVE_LEASES[id(lease)] = lease
            lease.assert_held(layout)
            os.ftruncate(descriptor, 0)
            os.lseek(descriptor, 0, os.SEEK_SET)
            _write_all(
                descriptor,
                (
                    json.dumps(
                        {
                            "operation": "source-import-publication",
                            "pid": os.getpid(),
                            "started_at": datetime.now(UTC).isoformat(),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                ).encode("utf-8"),
            )
            os.fsync(descriptor)
        except BlockingIOError as exc:
            raise SourceImportPublicationBusyError(
                "Another source import, publication, or destructive operation is active"
            ) from exc
        except SourceImportPublicationLockError:
            raise
        except OSError as exc:
            raise SourceImportPublicationLockError(
                "Could not establish the source/import publication lock"
            ) from exc

        try:
            yield lease
        except BaseException:
            raise
        else:
            lease.assert_held(layout)
    finally:
        if lease is not None:
            with _ACTIVE_LEASES_GUARD:
                _ACTIVE_LEASES.pop(id(lease), None)
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
