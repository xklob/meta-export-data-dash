from __future__ import annotations

import os
import select as select_module
import stat
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from meta_archive.models import DeletionOperation
from meta_archive.paths import ArchiveLayout
from meta_archive.services.activity_importer import import_activities
from meta_archive.services.consolidation import (
    build_blob_store,
    materialize_conflict_holding_area,
)
from meta_archive.services.deletion import deletion_lock
from meta_archive.services.inventory import scan_archive
from meta_archive.services.library import (
    LibraryBuildError,
    build_filesystem_library,
    library_build_lock,
    reconcile_unlinked_media,
)
from meta_archive.services.message_importer import import_messages
from meta_archive.services.source_import_lock import (
    SOURCE_IMPORT_PUBLICATION_LOCK_NAME,
    SourceImportPublicationBlockedError,
    SourceImportPublicationBusyError,
    SourceImportPublicationLease,
    SourceImportPublicationLockError,
    assert_source_import_publication_allowed,
    source_import_publication_lock,
)
from meta_archive.services.source_retirement import retire_source_chunks


def _blocking_deletion(engine: Engine, *, status: str) -> str:
    operation_id = str(uuid.uuid4())
    with Session(engine) as session, session.begin():
        session.add(
            DeletionOperation(
                id=operation_id,
                status=status,
                selector_json={"kind": "synthetic"},
                impact_json={},
                catalog_revision="1" * 64,
                selection_sha256="2" * 64,
                purge_plan_json={},
                item_count=0,
            )
        )
    return operation_id


def test_source_import_publication_lock_is_private_non_reentrant_and_lease_scoped(
    archive_layout: ArchiveLayout,
) -> None:
    with source_import_publication_lock(archive_layout) as lease:
        lock_path = archive_layout.catalog / SOURCE_IMPORT_PUBLICATION_LOCK_NAME
        lock_stat = lock_path.stat(follow_symlinks=False)
        assert stat.S_ISREG(lock_stat.st_mode)
        assert stat.S_IMODE(lock_stat.st_mode) == 0o600
        assert lock_stat.st_uid == os.geteuid()
        assert lock_stat.st_nlink == 1
        lease.assert_held(archive_layout)
        with (
            pytest.raises(SourceImportPublicationBusyError, match=r"publication.*active"),
            source_import_publication_lock(archive_layout, blocking=False),
        ):
            pytest.fail("source/publication lock was re-entered")

    with pytest.raises(SourceImportPublicationLockError, match="live source/import"):
        lease.assert_held(archive_layout)


def test_source_import_publication_lock_serializes_processes(
    archive_layout: ArchiveLayout,
) -> None:
    child_program = """
import sys
from pathlib import Path
from meta_archive.paths import ArchiveLayout
from meta_archive.services.source_import_lock import source_import_publication_lock

layout = ArchiveLayout(Path(sys.argv[1]))
print("ready", flush=True)
with source_import_publication_lock(layout, blocking=True):
    print("acquired", flush=True)
"""
    process: subprocess.Popen[str] | None = None
    try:
        with source_import_publication_lock(archive_layout):
            process = subprocess.Popen(
                [sys.executable, "-c", child_program, str(archive_layout.root)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            assert process.stdout is not None
            readable, _, _ = select_module.select([process.stdout], [], [], 10)
            assert readable, "child did not reach its source/publication lock attempt"
            assert process.stdout.readline() == "ready\n"
            readable, _, _ = select_module.select([process.stdout], [], [], 0.5)
            assert not readable, "child acquired while its parent still held the lock"

        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, stderr
        assert stdout == "acquired\n"
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_source_import_publication_lock_is_released_by_process_crash(
    archive_layout: ArchiveLayout,
) -> None:
    child_program = """
import os
import sys
from pathlib import Path
from meta_archive.paths import ArchiveLayout
from meta_archive.services.source_import_lock import source_import_publication_lock

with source_import_publication_lock(ArchiveLayout(Path(sys.argv[1]))):
    print("acquired", flush=True)
    os._exit(23)
"""
    completed = subprocess.run(
        [sys.executable, "-c", child_program, str(archive_layout.root)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode == 23
    assert completed.stdout == "acquired\n"

    # The stale metadata file remains, but the kernel releases flock with the crashed process's
    # descriptor. Reacquisition validates and safely reuses that same private inode.
    with source_import_publication_lock(archive_layout, blocking=False):
        pass


def test_source_import_publication_lock_rejects_unsafe_artifacts_without_mutation(
    archive_layout: ArchiveLayout,
    tmp_path: Path,
) -> None:
    archive_layout.ensure_runtime_directories()
    lock_path = archive_layout.catalog / SOURCE_IMPORT_PUBLICATION_LOCK_NAME
    outside = tmp_path / "outside-lock"
    outside.write_text("unchanged", encoding="utf-8")
    outside.chmod(0o644)
    lock_path.symlink_to(outside)

    with (
        pytest.raises(SourceImportPublicationLockError, match=r"Unsafe.*lock state"),
        source_import_publication_lock(archive_layout),
    ):
        pytest.fail("unsafe symlink lock was acquired")
    assert outside.read_text(encoding="utf-8") == "unchanged"
    assert stat.S_IMODE(outside.stat().st_mode) == 0o644

    lock_path.unlink()
    os.link(outside, lock_path)
    with (
        pytest.raises(SourceImportPublicationLockError, match="one owner-only regular file"),
        source_import_publication_lock(archive_layout),
    ):
        pytest.fail("multiply linked lock was acquired")
    assert outside.read_text(encoding="utf-8") == "unchanged"
    assert stat.S_IMODE(outside.stat().st_mode) == 0o644

    lock_path.unlink()
    lock_path.write_text("permissive", encoding="utf-8")
    lock_path.chmod(0o644)
    with (
        pytest.raises(SourceImportPublicationLockError, match="one owner-only regular file"),
        source_import_publication_lock(archive_layout),
    ):
        pytest.fail("permissive lock was acquired")
    assert lock_path.read_text(encoding="utf-8") == "permissive"
    assert stat.S_IMODE(lock_path.stat().st_mode) == 0o644


def test_source_import_publication_lease_rejects_replaced_path(
    archive_layout: ArchiveLayout,
) -> None:
    lock_path = archive_layout.catalog / SOURCE_IMPORT_PUBLICATION_LOCK_NAME
    with (
        pytest.raises(SourceImportPublicationLockError, match="no longer one safe"),
        source_import_publication_lock(archive_layout) as lease,
    ):
        lock_path.unlink()
        lock_path.write_text("replacement", encoding="utf-8")
        lock_path.chmod(0o600)
        lease.assert_held(archive_layout)


def test_unlocked_same_inode_descriptor_cannot_forge_a_source_lease(
    archive_layout: ArchiveLayout,
) -> None:
    with source_import_publication_lock(archive_layout):
        lock_path = archive_layout.catalog / SOURCE_IMPORT_PUBLICATION_LOCK_NAME
        descriptor = os.open(lock_path, os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            descriptor_stat = os.fstat(descriptor)
            forged = SourceImportPublicationLease(
                root=archive_layout.root,
                descriptor=descriptor,
                device=descriptor_stat.st_dev,
                inode=descriptor_stat.st_ino,
            )
            with pytest.raises(SourceImportPublicationLockError, match="live source/import"):
                forged.assert_held(archive_layout)
        finally:
            os.close(descriptor)


@pytest.mark.parametrize("status", ["committed", "cleanup_pending", "normalized_purged"])
def test_source_publication_guard_blocks_states_that_could_resurrect_deleted_rows(
    archive_layout: ArchiveLayout,
    catalog_engine: Engine,
    status: str,
) -> None:
    operation_id = _blocking_deletion(catalog_engine, status=status)
    with (
        source_import_publication_lock(archive_layout) as lease,
        pytest.raises(SourceImportPublicationBlockedError) as raised,
    ):
        assert_source_import_publication_allowed(
            archive_layout,
            catalog_engine,
            lease=lease,
            publisher="Synthetic source publisher",
        )
    assert raised.value.operation_id == operation_id
    assert raised.value.status == status


@pytest.mark.parametrize("status", ["planned", "restored", "purged"])
def test_source_publication_guard_allows_non_resurrecting_deletion_states(
    archive_layout: ArchiveLayout,
    catalog_engine: Engine,
    status: str,
) -> None:
    _blocking_deletion(catalog_engine, status=status)
    with source_import_publication_lock(archive_layout) as lease:
        assert_source_import_publication_allowed(
            archive_layout,
            catalog_engine,
            lease=lease,
            publisher="Synthetic source publisher",
        )


@pytest.mark.parametrize(
    "publisher",
    [
        lambda layout, engine: scan_archive(layout, engine, hash_files=False),
        lambda layout, engine: import_messages(layout, engine, force=True),
        lambda layout, engine: import_activities(layout, engine, force=True),
        lambda layout, engine: build_blob_store(layout, engine),
        lambda layout, engine: materialize_conflict_holding_area(layout, engine),
        lambda layout, engine: build_filesystem_library(layout, engine),
        lambda layout, engine: reconcile_unlinked_media(layout, engine),
        lambda layout, engine: retire_source_chunks(layout, engine, ["synthetic"]),
    ],
    ids=(
        "inventory",
        "messages",
        "activities",
        "blob-store",
        "conflict-holding",
        "filesystem-library",
        "unlinked-reconciliation",
        "source-retirement",
    ),
)
def test_every_source_publisher_blocks_before_sequential_quarantine_resurrection(
    archive_layout: ArchiveLayout,
    catalog_engine: Engine,
    publisher: object,
) -> None:
    _blocking_deletion(catalog_engine, status="committed")
    with pytest.raises(SourceImportPublicationBlockedError):
        publisher(archive_layout, catalog_engine)  # type: ignore[operator]

    assert not archive_layout.original_files_manifest.exists()
    assert not archive_layout.original_directories_manifest.exists()
    assert not archive_layout.relocation_manifest.exists()


def test_ordinary_source_lease_cannot_authorize_deletion_scoped_library_build(
    archive_layout: ArchiveLayout,
    catalog_engine: Engine,
) -> None:
    operation_id = _blocking_deletion(catalog_engine, status="committed")
    with (
        source_import_publication_lock(archive_layout) as source_lease,
        library_build_lock(archive_layout),
        Session(catalog_engine) as session,
        session.begin(),
        pytest.raises(LibraryBuildError, match="authorization is incomplete"),
    ):
        build_filesystem_library(
            archive_layout,
            catalog_engine,
            catalog_session=session,
            lock_held=True,
            source_lock=source_lease,
            deletion_operation_id=operation_id,
        )


def test_deletion_scoped_library_authorization_is_bound_to_exact_active_operation(
    archive_layout: ArchiveLayout,
    catalog_engine: Engine,
) -> None:
    operation_id = _blocking_deletion(catalog_engine, status="restored")
    with (
        deletion_lock(archive_layout) as deletion_lease,
        source_import_publication_lock(archive_layout) as source_lease,
        library_build_lock(archive_layout),
        Session(catalog_engine) as session,
        session.begin(),
        pytest.raises(LibraryBuildError, match="does not match an active purge"),
    ):
        build_filesystem_library(
            archive_layout,
            catalog_engine,
            catalog_session=session,
            lock_held=True,
            source_lock=source_lease,
            deletion_lock_lease=deletion_lease,
            deletion_operation_id=operation_id,
        )
