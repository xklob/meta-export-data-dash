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

from meta_archive.paths import (
    ArchiveLayout,
    UnsafeArchivePathError,
    create_directory_beneath,
    open_directory_beneath,
)

_ACTIVE_LEASES: dict[int, object] = {}
_ACTIVE_LEASES_GUARD = threading.Lock()
_ISSUER_TOKEN = object()


class DeletionError(RuntimeError):
    """The fixed archive deletion lock could not be acquired safely."""


class DeletionLockLeaseError(DeletionError):
    """A deletion-scoped publication lacks a live deletion lock capability."""


class DeletionLockLease:
    """Process-local proof that the fixed deletion lock descriptor owns ``flock``."""

    __slots__ = (
        "_active",
        "_descriptor",
        "_device",
        "_directory_descriptor",
        "_directory_device",
        "_directory_inode",
        "_inode",
        "_pid",
        "_root",
    )

    def __init__(
        self,
        *,
        root: Path,
        descriptor: int,
        directory_descriptor: int,
        directory_device: int,
        directory_inode: int,
        device: int,
        inode: int,
        issuer_token: object,
    ) -> None:
        if issuer_token is not _ISSUER_TOKEN:
            raise DeletionLockLeaseError("Deletion lock leases are issued only by deletion_lock")
        self._root = Path(os.path.abspath(root))
        self._descriptor = descriptor
        self._directory_descriptor = directory_descriptor
        self._directory_device = directory_device
        self._directory_inode = directory_inode
        self._device = device
        self._inode = inode
        self._pid = os.getpid()
        self._active = True

    def assert_held(self, layout: ArchiveLayout) -> None:
        with _ACTIVE_LEASES_GUARD:
            issued_and_active = _ACTIVE_LEASES.get(id(self)) is self
        if (
            not issued_and_active
            or not self._active
            or self._pid != os.getpid()
            or self._root != Path(os.path.abspath(layout.root))
        ):
            raise DeletionLockLeaseError("A live deletion lock lease for this archive is required")
        catalog_descriptor: int | None = None
        try:
            # Re-locking an already locked open-file description is a no-op. An unlocked forged
            # descriptor conflicts with the real owner and therefore fails this proof.
            fcntl.flock(self._descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            descriptor_stat = os.fstat(self._descriptor)
            directory_stat = os.fstat(self._directory_descriptor)
            catalog_descriptor = open_directory_beneath(layout.catalog, layout.root)
            current_catalog_stat = os.fstat(catalog_descriptor)
            path_stat = os.stat(
                "deletion.lock",
                dir_fd=self._directory_descriptor,
                follow_symlinks=False,
            )
        except (BlockingIOError, OSError, UnsafeArchivePathError) as exc:
            raise DeletionLockLeaseError(
                "The deletion lock lease is no longer held safely"
            ) from exc
        finally:
            if catalog_descriptor is not None:
                os.close(catalog_descriptor)
        if (
            not stat.S_ISDIR(directory_stat.st_mode)
            or (directory_stat.st_dev, directory_stat.st_ino)
            != (self._directory_device, self._directory_inode)
            or (current_catalog_stat.st_dev, current_catalog_stat.st_ino)
            != (self._directory_device, self._directory_inode)
            or not stat.S_ISREG(descriptor_stat.st_mode)
            or descriptor_stat.st_uid != os.geteuid()
            or descriptor_stat.st_nlink != 1
            or stat.S_IMODE(descriptor_stat.st_mode) != 0o600
            or not stat.S_ISREG(path_stat.st_mode)
            or path_stat.st_uid != os.geteuid()
            or path_stat.st_nlink != 1
            or stat.S_IMODE(path_stat.st_mode) != 0o600
            or (descriptor_stat.st_dev, descriptor_stat.st_ino)
            != (path_stat.st_dev, path_stat.st_ino)
            or (descriptor_stat.st_dev, descriptor_stat.st_ino) != (self._device, self._inode)
        ):
            raise DeletionLockLeaseError("The deletion lock lease no longer names one safe file")


def _issue_deletion_lock_lease(
    layout: ArchiveLayout,
    *,
    descriptor: int,
    directory_descriptor: int,
) -> DeletionLockLease:
    """Issue a registered lease after proving this descriptor owns the deletion flock."""

    catalog_descriptor: int | None = None
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        descriptor_stat = os.fstat(descriptor)
        directory_stat = os.fstat(directory_descriptor)
        catalog_descriptor = open_directory_beneath(layout.catalog, layout.root)
        current_catalog_stat = os.fstat(catalog_descriptor)
    except (BlockingIOError, OSError, UnsafeArchivePathError) as exc:
        raise DeletionLockLeaseError(
            "The deletion lock descriptor is not exclusively held"
        ) from exc
    finally:
        if catalog_descriptor is not None:
            os.close(catalog_descriptor)
    if not stat.S_ISDIR(directory_stat.st_mode) or (
        directory_stat.st_dev,
        directory_stat.st_ino,
    ) != (current_catalog_stat.st_dev, current_catalog_stat.st_ino):
        raise DeletionLockLeaseError(
            "The deletion lock descriptor is not rooted in the live archive catalog"
        )
    lease = DeletionLockLease(
        root=layout.root,
        descriptor=descriptor,
        directory_descriptor=directory_descriptor,
        directory_device=directory_stat.st_dev,
        directory_inode=directory_stat.st_ino,
        device=descriptor_stat.st_dev,
        inode=descriptor_stat.st_ino,
        issuer_token=_ISSUER_TOKEN,
    )
    with _ACTIVE_LEASES_GUARD:
        _ACTIVE_LEASES[id(lease)] = lease
    try:
        lease.assert_held(layout)
    except BaseException:
        _revoke_deletion_lock_lease(lease)
        raise
    return lease


def _revoke_deletion_lock_lease(lease: DeletionLockLease) -> None:
    """Invalidate an issued lease before its owning context unlocks the descriptor."""

    with _ACTIVE_LEASES_GUARD:
        _ACTIVE_LEASES.pop(id(lease), None)
    lease._active = False


def _open_private_lock_path(layout: ArchiveLayout, lock_path: Path) -> tuple[int, int]:
    """Open one archive lock without following or mutating an unsafe pathname."""

    if not all(hasattr(os, flag) for flag in ("O_DIRECTORY", "O_NOFOLLOW", "O_CLOEXEC")):
        raise OSError("Secure no-follow lock files are unavailable")
    absolute_path = Path(os.path.abspath(lock_path))
    archive_root = Path(os.path.abspath(layout.root))
    if absolute_path == archive_root or archive_root not in absolute_path.parents:
        raise OSError("Lock file is outside the archive root")
    name = absolute_path.name
    directory_descriptor: int | None = None
    descriptor: int | None = None
    try:
        create_directory_beneath(absolute_path.parent, layout.root, mode=0o700)
        directory_descriptor = open_directory_beneath(absolute_path.parent, layout.root)
        descriptor = os.open(
            name,
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
            0o600,
            dir_fd=directory_descriptor,
        )
        descriptor_stat = os.fstat(descriptor)
        path_stat = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        if (
            not stat.S_ISREG(descriptor_stat.st_mode)
            or descriptor_stat.st_uid != os.geteuid()
            or descriptor_stat.st_nlink != 1
            or (descriptor_stat.st_dev, descriptor_stat.st_ino)
            != (path_stat.st_dev, path_stat.st_ino)
        ):
            raise OSError("Lock file is not one owner-owned regular file")
        # Repair permissions only after proving this is a single-link owner-owned inode.
        os.fchmod(descriptor, 0o600)
        secured_stat = os.fstat(descriptor)
        if stat.S_IMODE(secured_stat.st_mode) != 0o600:
            raise OSError("Lock file permissions are not owner-only")
        return descriptor, directory_descriptor
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        if directory_descriptor is not None:
            os.close(directory_descriptor)
        raise


def _open_private_lock_file(layout: ArchiveLayout, name: str) -> tuple[int, int]:
    """Open one catalog lock without following or mutating an unsafe pathname."""

    return _open_private_lock_path(layout, layout.catalog / name)


def _validate_private_lock_file(
    descriptor: int,
    directory_descriptor: int,
    name: str,
) -> None:
    """Revalidate that a held descriptor still names the private lock pathname."""

    descriptor_stat = os.fstat(descriptor)
    path_stat = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    if (
        not stat.S_ISREG(descriptor_stat.st_mode)
        or descriptor_stat.st_uid != os.geteuid()
        or descriptor_stat.st_nlink != 1
        or stat.S_IMODE(descriptor_stat.st_mode) != 0o600
        or (descriptor_stat.st_dev, descriptor_stat.st_ino) != (path_stat.st_dev, path_stat.st_ino)
    ):
        raise OSError("Lock file path changed after it was opened")


@contextmanager
def deletion_lock(layout: ArchiveLayout) -> Iterator[DeletionLockLease]:
    """Acquire the cycle-free deletion-first lock and yield its live capability."""

    descriptor: int | None = None
    directory_descriptor: int | None = None
    lease: DeletionLockLease | None = None
    try:
        descriptor, directory_descriptor = _open_private_lock_file(layout, "deletion.lock")
    except (OSError, UnsafeArchivePathError) as exc:
        raise DeletionError("The deletion lock file is unsafe") from exc
    try:
        with os.fdopen(descriptor, "r+", encoding="utf-8") as handle:
            descriptor = None
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise DeletionError(
                    "Another deletion operation is currently mutating the catalog"
                ) from exc
            try:
                assert directory_descriptor is not None
                _validate_private_lock_file(handle.fileno(), directory_descriptor, "deletion.lock")
                handle.seek(0)
                handle.truncate()
                json.dump({"pid": os.getpid(), "started_at": datetime.now(UTC).isoformat()}, handle)
                handle.flush()
                os.fsync(handle.fileno())
            except OSError as exc:
                raise DeletionError("The deletion lock file became unsafe") from exc
            try:
                assert directory_descriptor is not None
                lease = _issue_deletion_lock_lease(
                    layout,
                    descriptor=handle.fileno(),
                    directory_descriptor=directory_descriptor,
                )
                try:
                    yield lease
                except BaseException:
                    raise
                else:
                    lease.assert_held(layout)
            except DeletionLockLeaseError as exc:
                raise DeletionError("The deletion lock lease became unsafe") from exc
            finally:
                if lease is not None:
                    _revoke_deletion_lock_lease(lease)
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if directory_descriptor is not None:
            os.close(directory_descriptor)
