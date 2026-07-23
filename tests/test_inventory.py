from __future__ import annotations

import json
import os
import select as select_module
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

from meta_archive.models import Blob, IntegrityAnomaly, SourceFile
from meta_archive.paths import ArchiveLayout
from meta_archive.services.consolidation import (
    CONFLICT_HOLDING_LOCK_NAME,
    ConflictHoldingLockError,
    build_blob_store,
    conflict_holding_lock,
    materialize_conflict_holding_area,
    materialize_conflict_holding_area_locked,
    reconstruct_source,
)
from meta_archive.services.hashing import sha256_file
from meta_archive.services.inventory import scan_archive
from meta_archive.services.source_import_lock import source_import_publication_lock


def _write_fixture_archive(layout: ArchiveLayout, *, conflicting_copy: bool = False) -> None:
    first = layout.root / "package-a"
    second = layout.root / "package-b"
    message_path = Path("your_facebook_activity/messages/inbox/example/message_1.json")
    media_path = Path("your_facebook_activity/messages/inbox/example/photos/photo.jpg")
    for chunk in (first, second):
        (chunk / message_path).parent.mkdir(parents=True)
        (chunk / media_path).parent.mkdir(parents=True)
    (first / "your_facebook_activity/messages/inbox/empty-thread").mkdir(parents=True)
    payload = {
        "title": "Example",
        "participants": [{"name": "Example"}],
        "messages": [],
        "thread_path": "inbox/example",
        "is_still_participant": True,
        "magic_words": [],
    }
    encoded = json.dumps(payload, sort_keys=True)
    (first / message_path).write_text(encoded, encoding="utf-8")
    (second / message_path).write_text(encoded, encoding="utf-8")
    (first / media_path).write_bytes(b"jpeg-fixture")
    (second / media_path).write_bytes(b"different" if conflicting_copy else b"jpeg-fixture")


def test_hashing_inventory_is_idempotent_and_writes_manifest(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> None:
    _write_fixture_archive(archive_layout)

    first = scan_archive(archive_layout, catalog_engine, hash_files=True)
    second = scan_archive(archive_layout, catalog_engine, hash_files=True)

    assert first.chunk_count == 2
    assert first.directory_count > 2
    assert first.file_count == 4
    assert first.hashed_file_count == 4
    assert first.unique_blob_count == 2
    assert first.duplicate_file_count == 2
    assert first.normalized_collision_groups == 2
    assert first.conflicting_collision_groups == 0
    assert second.hashed_file_count == 0
    assert second.reused_hash_count == 4
    assert archive_layout.original_files_manifest.read_text(encoding="utf-8").count("\n") == 4
    assert "empty-thread" in archive_layout.original_directories_manifest.read_text(
        encoding="utf-8"
    )

    with Session(catalog_engine) as session:
        assert session.scalar(select(func.count(SourceFile.id))) == 4
        assert session.scalar(select(func.count(Blob.sha256))) == 2


def test_conflicting_normalized_paths_are_reported(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> None:
    _write_fixture_archive(archive_layout, conflicting_copy=True)

    summary = scan_archive(archive_layout, catalog_engine, hash_files=True)

    assert summary.conflicting_collision_groups == 1
    with Session(catalog_engine) as session:
        anomaly = session.scalar(
            select(IntegrityAnomaly).where(IntegrityAnomaly.code == "conflicting_normalized_path")
        )
    assert anomaly is not None
    assert anomaly.severity == "critical"


def test_conflicting_copies_are_held_with_provenance_and_never_overwritten(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> None:
    _write_fixture_archive(archive_layout, conflicting_copy=True)
    scan_archive(archive_layout, catalog_engine, hash_files=True)

    first = materialize_conflict_holding_area(archive_layout, catalog_engine)

    assert first.group_count == 1
    assert first.copy_count == 2
    assert first.materialized_copy_count == 2
    assert first.reused_copy_count == 0
    manifests = list((archive_layout.conflicts / "normalized-path").glob("*/*/manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["kind"] == "normalized-path-byte-conflict"
    assert manifest["copy_count"] == 2
    assert manifest["distinct_sha256_count"] == 2
    assert {record["source_chunk"] for record in manifest["copies"]} == {
        "package-a",
        "package-b",
    }
    assert all(
        record["source_relative_path"].endswith("photos/photo.jpg") for record in manifest["copies"]
    )

    held_paths = [archive_layout.root / record["holding_path"] for record in manifest["copies"]]
    for record, held in zip(manifest["copies"], held_paths, strict=True):
        assert held.is_file()
        assert sha256_file(held) == record["sha256"]
        assert held.stat().st_size == record["size_bytes"]
        assert stat.S_IMODE(held.stat().st_mode) == 0o444
        source = archive_layout.root / record["source_chunk_path"] / record["source_relative_path"]
        assert held.stat().st_ino != source.stat().st_ino
    assert stat.S_IMODE(manifests[0].stat().st_mode) == 0o400

    second = materialize_conflict_holding_area(archive_layout, catalog_engine)
    assert second.materialized_copy_count == 0
    assert second.reused_copy_count == 2
    assert len(list((archive_layout.conflicts / "normalized-path").glob("*/*/manifest.json"))) == 1

    protected = held_paths[0]
    os.chmod(protected, 0o600)
    protected.write_bytes(b"tampered-held-copy")
    with pytest.raises(RuntimeError, match="Materialized bytes do not match"):
        materialize_conflict_holding_area(archive_layout, catalog_engine)
    assert protected.read_bytes() == b"tampered-held-copy"


def test_zero_conflicts_create_no_holding_revisions(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> None:
    _write_fixture_archive(archive_layout)
    scan_archive(archive_layout, catalog_engine, hash_files=True)

    summary = materialize_conflict_holding_area(archive_layout, catalog_engine)

    assert summary.group_count == 0
    assert summary.copy_count == 0
    assert not (archive_layout.conflicts / "normalized-path").exists()


def test_conflict_holding_rejects_symlinked_parent_before_writing(
    archive_layout: ArchiveLayout, catalog_engine: Engine, tmp_path: Path
) -> None:
    _write_fixture_archive(archive_layout, conflicting_copy=True)
    scan_archive(archive_layout, catalog_engine, hash_files=True)
    outside = tmp_path / "outside-conflicts"
    outside.mkdir()
    archive_layout.conflicts.rmdir()
    archive_layout.conflicts.symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="not a safe directory"):
        materialize_conflict_holding_area(archive_layout, catalog_engine)

    assert list(outside.iterdir()) == []


def test_conflict_holding_lock_is_owner_only_non_reentrant_and_supports_locked_build(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> None:
    _write_fixture_archive(archive_layout)
    scan_archive(archive_layout, catalog_engine, hash_files=True)

    with (
        source_import_publication_lock(archive_layout) as source_lease,
        conflict_holding_lock(archive_layout) as lease,
    ):
        lock_path = archive_layout.catalog / CONFLICT_HOLDING_LOCK_NAME
        assert stat.S_IMODE(lock_path.stat().st_mode) == 0o600
        with (
            pytest.raises(ConflictHoldingLockError, match="currently mutating"),
            conflict_holding_lock(archive_layout),
        ):
            pytest.fail("conflict lock was re-entered")
        summary = materialize_conflict_holding_area_locked(
            archive_layout,
            catalog_engine,
            source_lock=source_lease,
            lock=lease,
        )
        assert summary.group_count == 0

    with (
        source_import_publication_lock(archive_layout) as source_lease,
        pytest.raises(ConflictHoldingLockError, match="live conflict-holding lock lease"),
    ):
        materialize_conflict_holding_area_locked(
            archive_layout,
            catalog_engine,
            source_lock=source_lease,
            lock=lease,
        )


def test_conflict_holding_lock_serializes_a_second_process(
    archive_layout: ArchiveLayout,
) -> None:
    archive_layout.ensure_runtime_directories()
    child_program = """
import sys
from pathlib import Path
from meta_archive.paths import ArchiveLayout
from meta_archive.services.consolidation import conflict_holding_lock

layout = ArchiveLayout(Path(sys.argv[1]))
print("ready", flush=True)
with conflict_holding_lock(layout, blocking=True):
    print("acquired", flush=True)
"""
    process: subprocess.Popen[str] | None = None
    try:
        with conflict_holding_lock(archive_layout):
            process = subprocess.Popen(
                [sys.executable, "-c", child_program, str(archive_layout.root)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            assert process.stdout is not None
            readable, _, _ = select_module.select([process.stdout], [], [], 10)
            assert readable, "child did not reach the conflict lock attempt"
            assert process.stdout.readline() == "ready\n"
            readable, _, _ = select_module.select([process.stdout], [], [], 0.5)
            assert not readable, "child acquired the lock while its parent still held it"

        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, stderr
        assert stdout == "acquired\n"
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_conflict_holding_lock_fails_fast_in_a_second_process(
    archive_layout: ArchiveLayout,
) -> None:
    child_program = """
import sys
from pathlib import Path
from meta_archive.paths import ArchiveLayout
from meta_archive.services.consolidation import ConflictHoldingLockError, conflict_holding_lock

try:
    with conflict_holding_lock(ArchiveLayout(Path(sys.argv[1]))):
        raise AssertionError("unexpected acquisition")
except ConflictHoldingLockError:
    print("blocked", flush=True)
"""
    with conflict_holding_lock(archive_layout):
        completed = subprocess.run(
            [sys.executable, "-c", child_program, str(archive_layout.root)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "blocked\n"


def test_conflict_holding_lock_releases_and_invalidates_lease_after_body_error(
    archive_layout: ArchiveLayout,
) -> None:
    lease = None
    with (
        pytest.raises(ValueError, match="body failed"),
        conflict_holding_lock(archive_layout) as acquired,
    ):
        lease = acquired
        raise ValueError("body failed")

    assert lease is not None
    with pytest.raises(ConflictHoldingLockError, match="live conflict-holding lock lease"):
        lease.assert_held(archive_layout)
    with conflict_holding_lock(archive_layout):
        pass


def test_conflict_holding_lock_rejects_symlink_and_multiply_linked_files(
    archive_layout: ArchiveLayout, tmp_path: Path
) -> None:
    archive_layout.ensure_runtime_directories()
    lock_path = archive_layout.catalog / CONFLICT_HOLDING_LOCK_NAME
    outside = tmp_path / "outside-lock"
    outside.write_text("unchanged", encoding="utf-8")
    lock_path.symlink_to(outside)

    with (
        pytest.raises(ConflictHoldingLockError, match="Unsafe conflict-holding lock state"),
        conflict_holding_lock(archive_layout),
    ):
        pytest.fail("unsafe symlink lock was acquired")
    assert outside.read_text(encoding="utf-8") == "unchanged"

    lock_path.unlink()
    outside.write_text("external lock bytes", encoding="utf-8")
    outside.chmod(0o644)
    os.link(outside, lock_path)
    with (
        pytest.raises(ConflictHoldingLockError, match="one owner-owned regular file"),
        conflict_holding_lock(archive_layout),
    ):
        pytest.fail("multiply linked lock was acquired")
    assert outside.read_text(encoding="utf-8") == "external lock bytes"
    assert stat.S_IMODE(outside.stat().st_mode) == 0o644


def test_locked_materializer_rejects_a_replaced_lock_path(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> None:
    _write_fixture_archive(archive_layout)
    scan_archive(archive_layout, catalog_engine, hash_files=True)
    lock_path = archive_layout.catalog / CONFLICT_HOLDING_LOCK_NAME

    with (
        pytest.raises(ConflictHoldingLockError, match="no longer a safe"),
        source_import_publication_lock(archive_layout) as source_lease,
        conflict_holding_lock(archive_layout) as lease,
    ):
        lock_path.unlink()
        lock_path.write_text("replacement", encoding="utf-8")
        with pytest.raises(ConflictHoldingLockError, match="no longer a safe"):
            materialize_conflict_holding_area_locked(
                archive_layout,
                catalog_engine,
                source_lock=source_lease,
                lock=lease,
            )


def test_blob_store_and_reconstruction_preserve_bytes_and_metadata_without_source_hardlinks(
    archive_layout: ArchiveLayout, catalog_engine: Engine, tmp_path: Path
) -> None:
    _write_fixture_archive(archive_layout)
    scan_archive(archive_layout, catalog_engine, hash_files=True)

    linked, reused = build_blob_store(archive_layout, catalog_engine)
    target = tmp_path / "reconstructed"
    count = reconstruct_source(archive_layout, target)

    assert linked == 2
    assert reused == 0
    assert count == 4
    original = (
        archive_layout.root
        / "package-a/your_facebook_activity/messages/inbox/example/photos/photo.jpg"
    )
    blob = archive_layout.blob_path(sha256_file(original))
    rebuilt = target / "package-a/your_facebook_activity/messages/inbox/example/photos/photo.jpg"
    assert rebuilt.read_bytes() == original.read_bytes()
    assert blob.read_bytes() == original.read_bytes()
    assert blob.stat().st_ino != original.stat().st_ino
    assert rebuilt.stat().st_ino != original.stat().st_ino
    assert rebuilt.stat().st_ino != blob.stat().st_ino
    assert stat.S_IMODE(blob.stat().st_mode) == 0o444
    assert stat.S_IMODE(rebuilt.stat().st_mode) == stat.S_IMODE(original.stat().st_mode)
    assert rebuilt.stat().st_mtime_ns == original.stat().st_mtime_ns
    assert (target / "package-a/your_facebook_activity/messages/inbox/empty-thread").is_dir()


def test_reconstruction_rejects_archive_tree_and_resolved_ancestor_but_allows_sibling(
    archive_layout: ArchiveLayout, catalog_engine: Engine, tmp_path: Path
) -> None:
    _write_fixture_archive(archive_layout)
    scan_archive(archive_layout, catalog_engine, hash_files=True)
    build_blob_store(archive_layout, catalog_engine)
    parent_alias = tmp_path / "archive-parent-alias"
    parent_alias.symlink_to(archive_layout.root.parent, target_is_directory=True)
    parent_entries = set(tmp_path.iterdir())

    unsafe_targets = (
        archive_layout.root,
        archive_layout.root / "reconstructed-inside",
        archive_layout.root.parent,
        parent_alias,
    )
    for unsafe_target in unsafe_targets:
        with pytest.raises(RuntimeError, match="must be disjoint"):
            reconstruct_source(archive_layout, unsafe_target)

    assert set(tmp_path.iterdir()) == parent_entries
    sibling = archive_layout.root.parent / "reconstructed-sibling"
    assert reconstruct_source(archive_layout, sibling) == 4
    assert (sibling / "package-a").is_dir()


def test_blob_store_breaks_legacy_source_hardlinks_and_is_idempotent(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> None:
    _write_fixture_archive(archive_layout)
    scan_archive(archive_layout, catalog_engine, hash_files=True)
    original = (
        archive_layout.root
        / "package-a/your_facebook_activity/messages/inbox/example/photos/photo.jpg"
    )
    digest = sha256_file(original)
    blob = archive_layout.blob_path(digest)
    blob.parent.mkdir(parents=True, exist_ok=True)
    os.link(original, blob)

    materialized, reused = build_blob_store(archive_layout, catalog_engine)

    assert materialized == 2
    assert reused == 0
    assert blob.stat().st_ino != original.stat().st_ino
    assert stat.S_IMODE(blob.stat().st_mode) == 0o444
    assert stat.S_IMODE(original.stat().st_mode) & stat.S_IWUSR

    second_materialized, second_reused = build_blob_store(archive_layout, catalog_engine)
    assert second_materialized == 0
    assert second_reused == 2
