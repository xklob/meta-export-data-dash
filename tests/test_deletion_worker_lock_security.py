from __future__ import annotations

import fcntl
import os
import stat
import threading
from pathlib import Path

import pytest

from meta_archive.paths import ArchiveLayout
from meta_archive.services.deletion import HardPurgeBlockedError, purge_worker_locks

WORKER_LOCK_CASES = (
    pytest.param(
        "thumbnails",
        ".build.lock",
        "media derivative build",
        id="thumbnail-builder",
    ),
    pytest.param(
        "catalog",
        "audio-transcription.lock",
        "audio transcription job",
        id="catalog-worker",
    ),
)


def _worker_lock_path(layout: ArchiveLayout, location: str, name: str) -> Path:
    parent = layout.thumbnails if location == "thumbnails" else layout.catalog
    return parent / name


@pytest.mark.parametrize(("location", "name", "label"), WORKER_LOCK_CASES)
def test_purge_worker_locks_reject_symlink_leaves_without_mutating_target(
    tmp_path: Path,
    location: str,
    name: str,
    label: str,
) -> None:
    layout = ArchiveLayout(tmp_path / f"meta-backup-worker-symlink-{location}")
    layout.root.mkdir()
    lock_path = _worker_lock_path(layout, location, name)
    lock_path.parent.mkdir(parents=True)
    external = tmp_path / f"external-worker-symlink-{location}"
    original_bytes = b"external symlink target must remain unchanged"
    external.write_bytes(original_bytes)
    external.chmod(0o644)
    original_mode = stat.S_IMODE(external.stat().st_mode)
    lock_path.symlink_to(external)

    with (
        pytest.raises(
            HardPurgeBlockedError,
            match=rf"The {label} lock file is unsafe",
        ),
        purge_worker_locks(layout),
    ):
        pytest.fail("purge acquired an unsafe symlink worker lock")

    assert lock_path.is_symlink()
    assert external.read_bytes() == original_bytes
    assert stat.S_IMODE(external.stat().st_mode) == original_mode


@pytest.mark.parametrize(("location", "name", "label"), WORKER_LOCK_CASES)
def test_purge_worker_locks_reject_hardlink_leaves_before_chmod_or_truncate(
    tmp_path: Path,
    location: str,
    name: str,
    label: str,
) -> None:
    layout = ArchiveLayout(tmp_path / f"meta-backup-worker-hardlink-{location}")
    layout.root.mkdir()
    lock_path = _worker_lock_path(layout, location, name)
    lock_path.parent.mkdir(parents=True)
    external = tmp_path / f"external-worker-hardlink-{location}"
    original_bytes = b"external hardlink target must remain unchanged"
    external.write_bytes(original_bytes)
    external.chmod(0o644)
    original_mode = stat.S_IMODE(external.stat().st_mode)
    os.link(external, lock_path)

    with (
        pytest.raises(
            HardPurgeBlockedError,
            match=rf"The {label} lock file is unsafe",
        ),
        purge_worker_locks(layout),
    ):
        pytest.fail("purge acquired a multiply linked worker lock")

    assert lock_path.stat().st_ino == external.stat().st_ino
    assert external.read_bytes() == original_bytes
    assert stat.S_IMODE(external.stat().st_mode) == original_mode


@pytest.mark.parametrize("symlinked_directory", ("catalog", "thumbnails"))
def test_purge_worker_locks_reject_intermediate_symlinked_directories(
    tmp_path: Path,
    symlinked_directory: str,
) -> None:
    layout = ArchiveLayout(tmp_path / f"meta-backup-worker-directory-{symlinked_directory}")
    layout.root.mkdir()
    external = tmp_path / f"external-worker-directory-{symlinked_directory}"
    external.mkdir()
    marker = external / "marker"
    marker.write_bytes(b"external directory must remain untouched")

    if symlinked_directory == "catalog":
        layout.catalog.symlink_to(external, target_is_directory=True)
    else:
        layout.catalog.mkdir()
        layout.thumbnails.symlink_to(external, target_is_directory=True)

    with (
        pytest.raises(HardPurgeBlockedError, match="lock file is unsafe"),
        purge_worker_locks(layout),
    ):
        pytest.fail("purge traversed an intermediate symlinked directory")

    assert marker.read_bytes() == b"external directory must remain untouched"
    assert sorted(path.name for path in external.iterdir()) == ["marker"]


@pytest.mark.parametrize(("location", "name", "_label"), WORKER_LOCK_CASES)
def test_purge_worker_locks_repair_valid_lock_mode_to_owner_only(
    tmp_path: Path,
    location: str,
    name: str,
    _label: str,
) -> None:
    layout = ArchiveLayout(tmp_path / f"meta-backup-worker-mode-{location}")
    layout.root.mkdir()
    lock_path = _worker_lock_path(layout, location, name)
    lock_path.parent.mkdir(parents=True)
    lock_path.write_bytes(b"existing worker metadata")
    lock_path.chmod(0o644)

    with purge_worker_locks(layout):
        lock_stat = lock_path.stat()
        assert stat.S_ISREG(lock_stat.st_mode)
        assert lock_stat.st_nlink == 1
        assert stat.S_IMODE(lock_stat.st_mode) == 0o600

    assert stat.S_IMODE(lock_path.stat().st_mode) == 0o600


@pytest.mark.parametrize(("location", "name", "label"), WORKER_LOCK_CASES)
def test_purge_worker_locks_preserve_nonblocking_contention(
    tmp_path: Path,
    location: str,
    name: str,
    label: str,
) -> None:
    layout = ArchiveLayout(tmp_path / f"meta-backup-worker-contention-{location}")
    layout.root.mkdir()
    lock_path = _worker_lock_path(layout, location, name)
    lock_path.parent.mkdir(parents=True)
    outcomes: list[BaseException | str] = []
    finished = threading.Event()

    def attempt_purge_lock() -> None:
        try:
            with purge_worker_locks(layout):
                outcomes.append("acquired")
        except BaseException as exc:
            outcomes.append(exc)
        finally:
            finished.set()

    with lock_path.open("a+b") as publisher_handle:
        fcntl.flock(publisher_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        worker = threading.Thread(target=attempt_purge_lock, daemon=True)
        worker.start()
        completed_while_held = finished.wait(timeout=2)
        fcntl.flock(publisher_handle.fileno(), fcntl.LOCK_UN)

    worker.join(timeout=2)
    assert completed_while_held, "purge blocked instead of failing fast on worker contention"
    assert not worker.is_alive()
    assert len(outcomes) == 1
    assert isinstance(outcomes[0], HardPurgeBlockedError)
    assert f"A {label} is currently running" in str(outcomes[0])
