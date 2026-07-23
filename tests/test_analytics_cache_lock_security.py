from __future__ import annotations

import fcntl
import os
import stat
import threading
from pathlib import Path

import pytest

from meta_archive.paths import ArchiveLayout
from meta_archive.services.analytics_cache import analytics_cache_lock


@pytest.mark.parametrize("link_kind", ("symlink", "hardlink"))
def test_analytics_lock_rejects_unsafe_leaf_without_mutating_external_file(
    tmp_path: Path,
    link_kind: str,
) -> None:
    layout = ArchiveLayout(tmp_path / f"meta-backup-analytics-{link_kind}")
    layout.catalog.mkdir(parents=True)
    external = tmp_path / f"analytics-{link_kind}-outside"
    external.write_bytes(b"external analytics lock bytes must remain unchanged")
    external.chmod(0o644)
    before_mode = stat.S_IMODE(external.stat().st_mode)
    lock_path = layout.catalog / "analytics-cache.lock"
    if link_kind == "symlink":
        lock_path.symlink_to(external)
    else:
        os.link(external, lock_path)

    with (
        pytest.raises(OSError, match="securely open"),
        analytics_cache_lock(layout, exclusive=True),
    ):
        pytest.fail("an unsafe analytics cache lock was acquired")

    assert external.read_bytes() == b"external analytics lock bytes must remain unchanged"
    assert stat.S_IMODE(external.stat().st_mode) == before_mode
    if link_kind == "hardlink":
        assert lock_path.stat().st_ino == external.stat().st_ino


@pytest.mark.parametrize("symlink_component", ("archive-root", "catalog"))
def test_analytics_lock_rejects_symlinked_parent_traversal(
    tmp_path: Path,
    symlink_component: str,
) -> None:
    outside = tmp_path / f"analytics-{symlink_component}-outside"
    outside.mkdir()
    marker = outside / "marker"
    marker.write_bytes(b"outside directory must remain untouched")
    layout = ArchiveLayout(tmp_path / f"meta-backup-analytics-{symlink_component}")
    if symlink_component == "archive-root":
        layout.root.symlink_to(outside, target_is_directory=True)
    else:
        layout.root.mkdir()
        layout.catalog.symlink_to(outside, target_is_directory=True)

    with (
        pytest.raises(OSError, match="securely open"),
        analytics_cache_lock(layout, exclusive=False),
    ):
        pytest.fail("an analytics lock beneath a symlinked directory was acquired")

    assert marker.read_bytes() == b"outside directory must remain untouched"
    assert sorted(path.name for path in outside.iterdir()) == ["marker"]


def test_analytics_lock_repairs_safe_mode_and_preserves_content(tmp_path: Path) -> None:
    layout = ArchiveLayout(tmp_path / "meta-backup-analytics-mode")
    layout.catalog.mkdir(parents=True)
    lock_path = layout.catalog / "analytics-cache.lock"
    lock_path.write_bytes(b"existing safe lock metadata")
    lock_path.chmod(0o644)

    with analytics_cache_lock(layout, exclusive=False):
        lock_stat = lock_path.stat()
        assert stat.S_ISREG(lock_stat.st_mode)
        assert lock_stat.st_uid == os.geteuid()
        assert lock_stat.st_nlink == 1
        assert stat.S_IMODE(lock_stat.st_mode) == 0o600

    assert lock_path.read_bytes() == b"existing safe lock metadata"


def test_analytics_lock_preserves_shared_and_blocking_exclusive_behavior(
    tmp_path: Path,
) -> None:
    layout = ArchiveLayout(tmp_path / "meta-backup-analytics-contention")
    layout.root.mkdir()
    shared_acquired = threading.Event()
    shared_release = threading.Event()
    exclusive_started = threading.Event()
    exclusive_acquired = threading.Event()
    exclusive_release = threading.Event()
    failures: list[BaseException] = []

    def take_shared_lock() -> None:
        try:
            with analytics_cache_lock(layout, exclusive=False):
                shared_acquired.set()
                assert shared_release.wait(timeout=2)
        except BaseException as exc:
            failures.append(exc)

    def take_exclusive_lock() -> None:
        try:
            exclusive_started.set()
            with analytics_cache_lock(layout, exclusive=True):
                exclusive_acquired.set()
                assert exclusive_release.wait(timeout=2)
        except BaseException as exc:
            failures.append(exc)

    shared_thread = threading.Thread(target=take_shared_lock)
    exclusive_thread = threading.Thread(target=take_exclusive_lock)
    try:
        with analytics_cache_lock(layout, exclusive=False):
            shared_thread.start()
            assert shared_acquired.wait(timeout=1)
            exclusive_thread.start()
            assert exclusive_started.wait(timeout=1)
            assert not exclusive_acquired.wait(timeout=0.1)
            shared_release.set()
            shared_thread.join(timeout=2)
            assert not shared_thread.is_alive()
            assert not exclusive_acquired.wait(timeout=0.1)

        assert exclusive_acquired.wait(timeout=1)
        exclusive_release.set()
        exclusive_thread.join(timeout=2)
        assert not exclusive_thread.is_alive()
        assert failures == []
    finally:
        shared_release.set()
        exclusive_release.set()
        for thread in (shared_thread, exclusive_thread):
            if thread.ident is not None:
                thread.join(timeout=2)


def test_analytics_lock_detects_path_replacement_after_flock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = ArchiveLayout(tmp_path / "meta-backup-analytics-replacement")
    layout.catalog.mkdir(parents=True)
    lock_path = layout.catalog / "analytics-cache.lock"
    lock_path.write_bytes(b"original lock inode")
    lock_path.chmod(0o600)
    real_flock = fcntl.flock
    replaced = False

    def replace_path_then_flock(descriptor: int, operation: int) -> None:
        nonlocal replaced
        if not replaced and operation != fcntl.LOCK_UN:
            lock_path.unlink()
            lock_path.write_bytes(b"replacement lock inode")
            lock_path.chmod(0o600)
            replaced = True
        real_flock(descriptor, operation)

    monkeypatch.setattr(fcntl, "flock", replace_path_then_flock)

    with (
        pytest.raises(OSError, match="path changed"),
        analytics_cache_lock(layout, exclusive=True),
    ):
        pytest.fail("a replaced analytics lock pathname was accepted")

    assert replaced is True
    assert lock_path.read_bytes() == b"replacement lock inode"
