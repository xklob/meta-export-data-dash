from __future__ import annotations

import os
import stat
import threading
from pathlib import Path

import pytest

from meta_archive.paths import (
    PRIVATE_DIRECTORY_MODE,
    PRIVATE_FILE_MODE,
    ArchiveLayout,
    ArchiveNotFoundError,
    UnsafeArchivePathError,
    open_regular_file_beneath,
    resolve_archive_root,
)


def test_discovers_exactly_one_meta_backup_directory(tmp_path: Path) -> None:
    expected = tmp_path / "meta-backup-example"
    expected.mkdir()

    assert resolve_archive_root(cwd=tmp_path) == expected


def test_refuses_ambiguous_auto_discovery(tmp_path: Path) -> None:
    (tmp_path / "meta-backup-one").mkdir()
    (tmp_path / "meta-backup-two").mkdir()

    with pytest.raises(ArchiveNotFoundError, match="found 2"):
        resolve_archive_root(cwd=tmp_path)


def test_explicit_archive_root_rejects_a_symlink(tmp_path: Path) -> None:
    target = tmp_path / "actual-meta-backup"
    target.mkdir()
    link = tmp_path / "meta-backup-link"
    link.symlink_to(target, target_is_directory=True)

    with pytest.raises(UnsafeArchivePathError, match="non-symlink directory"):
        resolve_archive_root(link)


def test_auto_discovery_rejects_a_symlink(tmp_path: Path) -> None:
    target = tmp_path / "actual-meta-backup"
    target.mkdir()
    link = tmp_path / "meta-backup-link"
    link.symlink_to(target, target_is_directory=True)

    with pytest.raises(UnsafeArchivePathError, match="non-symlink directory"):
        resolve_archive_root(cwd=tmp_path)


def test_chunk_discovery_excludes_generated_directories(tmp_path: Path) -> None:
    root = tmp_path / "meta-backup-example"
    root.mkdir()
    (root / "package-one").mkdir()
    layout = ArchiveLayout(root)
    layout.ensure_runtime_directories()

    assert [path.name for path in layout.chunk_directories()] == ["package-one"]


def test_runtime_boundary_repairs_private_directory_and_mutable_file_modes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "meta-backup-example"
    root.mkdir(mode=0o755)
    layout = ArchiveLayout(root)
    layout.catalog.mkdir(mode=0o755)
    layout.analytics_cache.mkdir(mode=0o755)
    layout.manifests.mkdir(mode=0o755)
    (layout.manifests / "integrity-reports").mkdir(mode=0o755)
    for path in (root, layout.catalog, layout.analytics_cache, layout.manifests):
        path.chmod(0o755)
    (layout.manifests / "integrity-reports").chmod(0o755)
    layout.database.write_bytes(b"")
    report = layout.manifests / "integrity-reports" / "latest.json"
    report.write_text("{}", encoding="utf-8")
    layout.database.chmod(0o644)
    report.chmod(0o644)

    layout.ensure_runtime_directories()

    assert all(
        stat.S_IMODE(path.lstat().st_mode) == PRIVATE_DIRECTORY_MODE
        for path in layout.private_runtime_directories()
    )
    assert stat.S_IMODE(layout.database.lstat().st_mode) == PRIVATE_FILE_MODE
    assert stat.S_IMODE(report.lstat().st_mode) == PRIVATE_FILE_MODE
    assert layout.private_permission_findings() == []


def test_runtime_boundary_rejects_symlinked_directory_without_touching_target(
    tmp_path: Path,
) -> None:
    root = tmp_path / "meta-backup-example"
    root.mkdir()
    target = tmp_path / "outside-catalog"
    target.mkdir(mode=0o755)
    target.chmod(0o755)
    layout = ArchiveLayout(root)
    layout.catalog.symlink_to(target, target_is_directory=True)

    with pytest.raises(UnsafeArchivePathError, match="runtime directory"):
        layout.ensure_runtime_directories()

    assert stat.S_IMODE(target.lstat().st_mode) == 0o755


def test_runtime_boundary_rejects_hardlinked_mutable_file_before_chmod(
    tmp_path: Path,
) -> None:
    root = tmp_path / "meta-backup-example"
    root.mkdir()
    layout = ArchiveLayout(root)
    layout.catalog.mkdir()
    outside = tmp_path / "outside-lock"
    outside.write_text("unchanged", encoding="utf-8")
    outside.chmod(0o644)
    os.link(outside, layout.catalog / "source-import-publication.lock")

    with pytest.raises(UnsafeArchivePathError, match="runtime file"):
        layout.ensure_runtime_directories()

    assert outside.read_text(encoding="utf-8") == "unchanged"
    assert stat.S_IMODE(outside.stat().st_mode) == 0o644


def test_secure_regular_file_open_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    root = tmp_path / "meta-backup-example"
    root.mkdir()
    fifo = root / "unexpected-fifo"
    os.mkfifo(fifo)
    errors: list[BaseException] = []

    def attempt_open() -> None:
        try:
            open_regular_file_beneath(fifo, root)
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=attempt_open, daemon=True)
    worker.start()
    worker.join(timeout=1)

    assert not worker.is_alive(), "secure archive open blocked on a FIFO"
    assert len(errors) == 1
    assert isinstance(errors[0], UnsafeArchivePathError)
