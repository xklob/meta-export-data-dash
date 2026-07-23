from __future__ import annotations

import fcntl
import os
import stat
from contextlib import AbstractContextManager
from pathlib import Path

import pytest

from meta_archive.paths import ArchiveLayout
from meta_archive.services.deletion import (
    DeletionError,
    HardPurgeBlockedError,
    deletion_lock,
    vector_store_lock,
)
from meta_archive.services.deletion_lock_lease import (
    DeletionLockLeaseError,
    _issue_deletion_lock_lease,
)
from meta_archive.services.search_index import (
    SemanticIndexBusyError,
    semantic_index_lock,
    semantic_index_lock_is_held,
)


def _lock_context(case: str, layout: ArchiveLayout) -> AbstractContextManager[None]:
    if case == "deletion":
        return deletion_lock(layout)
    if case == "semantic-builder":
        return semantic_index_lock(layout)
    if case == "semantic-purge":
        return vector_store_lock(layout)
    raise AssertionError(f"unknown lock case: {case}")


@pytest.mark.parametrize(
    ("case", "lock_name", "error_type"),
    (
        pytest.param("deletion", "deletion.lock", DeletionError, id="deletion"),
        pytest.param(
            "semantic-builder",
            "semantic-index.lock",
            SemanticIndexBusyError,
            id="semantic-builder",
        ),
        pytest.param(
            "semantic-purge",
            "semantic-index.lock",
            HardPurgeBlockedError,
            id="semantic-purge",
        ),
    ),
)
def test_mutation_locks_reject_symlinks_without_touching_the_target(
    tmp_path: Path,
    case: str,
    lock_name: str,
    error_type: type[Exception],
) -> None:
    layout = ArchiveLayout(tmp_path / f"meta-backup-{case}-symlink")
    layout.catalog.mkdir(parents=True)
    external = tmp_path / f"{case}-external-symlink-target"
    external.write_bytes(b"outside bytes must remain unchanged")
    external.chmod(0o644)
    before_mode = stat.S_IMODE(external.stat().st_mode)
    (layout.catalog / lock_name).symlink_to(external)

    with (
        pytest.raises(error_type, match="unsafe"),
        _lock_context(case, layout),
    ):
        pytest.fail("an unsafe symlink lock was acquired")

    assert external.read_bytes() == b"outside bytes must remain unchanged"
    assert stat.S_IMODE(external.stat().st_mode) == before_mode


@pytest.mark.parametrize(
    ("case", "lock_name", "error_type"),
    (
        pytest.param("deletion", "deletion.lock", DeletionError, id="deletion"),
        pytest.param(
            "semantic-builder",
            "semantic-index.lock",
            SemanticIndexBusyError,
            id="semantic-builder",
        ),
        pytest.param(
            "semantic-purge",
            "semantic-index.lock",
            HardPurgeBlockedError,
            id="semantic-purge",
        ),
    ),
)
def test_mutation_locks_reject_hardlinks_before_chmod_or_truncate(
    tmp_path: Path,
    case: str,
    lock_name: str,
    error_type: type[Exception],
) -> None:
    layout = ArchiveLayout(tmp_path / f"meta-backup-{case}-hardlink")
    layout.catalog.mkdir(parents=True)
    external = tmp_path / f"{case}-external-hardlink-target"
    external.write_bytes(b"outside hardlink bytes must remain unchanged")
    external.chmod(0o644)
    before_mode = stat.S_IMODE(external.stat().st_mode)
    lock_path = layout.catalog / lock_name
    os.link(external, lock_path)

    with (
        pytest.raises(error_type, match="unsafe"),
        _lock_context(case, layout),
    ):
        pytest.fail("a multiply linked lock was acquired")

    assert external.read_bytes() == b"outside hardlink bytes must remain unchanged"
    assert stat.S_IMODE(external.stat().st_mode) == before_mode
    assert lock_path.stat().st_ino == external.stat().st_ino


def test_semantic_lock_status_fails_closed_on_unsafe_links(tmp_path: Path) -> None:
    for kind in ("symlink", "hardlink"):
        layout = ArchiveLayout(tmp_path / f"meta-backup-semantic-status-{kind}")
        layout.catalog.mkdir(parents=True)
        external = tmp_path / f"semantic-status-{kind}-target"
        external.write_bytes(b"status probes must not mutate this")
        external.chmod(0o644)
        lock_path = layout.catalog / "semantic-index.lock"
        if kind == "symlink":
            lock_path.symlink_to(external)
        else:
            os.link(external, lock_path)

        assert semantic_index_lock_is_held(layout) is True
        assert external.read_bytes() == b"status probes must not mutate this"
        assert stat.S_IMODE(external.stat().st_mode) == 0o644


@pytest.mark.parametrize(
    ("case", "error_type"),
    (
        pytest.param("deletion", DeletionError, id="deletion"),
        pytest.param("semantic-builder", SemanticIndexBusyError, id="semantic-builder"),
        pytest.param("semantic-purge", HardPurgeBlockedError, id="semantic-purge"),
    ),
)
@pytest.mark.parametrize("symlink_component", ("archive-root", "catalog"))
def test_mutation_locks_reject_symlinked_directory_traversal(
    tmp_path: Path,
    case: str,
    error_type: type[Exception],
    symlink_component: str,
) -> None:
    external = tmp_path / f"{case}-{symlink_component}-outside"
    external.mkdir()
    marker = external / "marker"
    marker.write_bytes(b"outside directory must remain untouched")
    layout = ArchiveLayout(tmp_path / f"meta-backup-{case}-{symlink_component}")
    if symlink_component == "archive-root":
        layout.root.symlink_to(external, target_is_directory=True)
    else:
        layout.root.mkdir()
        layout.catalog.symlink_to(external, target_is_directory=True)

    with (
        pytest.raises(error_type, match="unsafe"),
        _lock_context(case, layout),
    ):
        pytest.fail("a lock beneath a symlinked directory was acquired")

    assert marker.read_bytes() == b"outside directory must remain untouched"
    assert sorted(path.name for path in external.iterdir()) == ["marker"]


def test_lock_files_are_single_link_owner_only_and_remain_nonblocking(tmp_path: Path) -> None:
    layout = ArchiveLayout(tmp_path / "meta-backup-secure-lock-modes")
    layout.root.mkdir()

    with deletion_lock(layout):
        deletion_stat = (layout.catalog / "deletion.lock").stat()
        assert stat.S_ISREG(deletion_stat.st_mode)
        assert deletion_stat.st_nlink == 1
        assert stat.S_IMODE(deletion_stat.st_mode) == 0o600
        with (
            pytest.raises(DeletionError, match="currently mutating"),
            deletion_lock(layout),
        ):
            pytest.fail("a second deletion lock was acquired")

    assert semantic_index_lock_is_held(layout) is False
    with semantic_index_lock(layout):
        semantic_stat = (layout.catalog / "semantic-index.lock").stat()
        assert stat.S_ISREG(semantic_stat.st_mode)
        assert semantic_stat.st_nlink == 1
        assert stat.S_IMODE(semantic_stat.st_mode) == 0o600
        assert semantic_index_lock_is_held(layout) is True
        with (
            pytest.raises(HardPurgeBlockedError, match="currently being rebuilt"),
            vector_store_lock(layout),
        ):
            pytest.fail("the purge acquired an active semantic builder lock")


def test_deletion_lease_issuer_rejects_a_locked_file_outside_live_catalog(
    tmp_path: Path,
) -> None:
    layout = ArchiveLayout(tmp_path / "meta-backup-deletion-foreign-lease")
    layout.catalog.mkdir(parents=True)
    foreign_directory = tmp_path / "foreign-lock-directory"
    foreign_directory.mkdir()
    foreign_lock = foreign_directory / "deletion.lock"
    foreign_lock.touch(mode=0o600)
    directory_descriptor = os.open(
        foreign_directory,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    descriptor = os.open(
        "deletion.lock",
        os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=directory_descriptor,
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(DeletionLockLeaseError, match="live archive catalog"):
            _issue_deletion_lock_lease(
                layout,
                descriptor=descriptor,
                directory_descriptor=directory_descriptor,
            )
    finally:
        os.close(descriptor)
        os.close(directory_descriptor)
