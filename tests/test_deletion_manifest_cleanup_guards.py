from __future__ import annotations

import hashlib
import os
import shutil
import stat
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from meta_archive.database import create_catalog_engine, run_migrations
from meta_archive.models import DeletionOperation
from meta_archive.paths import ArchiveLayout
from meta_archive.services import deletion as deletion_service
from meta_archive.services.activity_importer import import_activities
from meta_archive.services.consolidation import build_blob_store
from meta_archive.services.deletion import (
    DeletionPreview,
    DeletionSelector,
    HardPurgeBlockedError,
    commit_deletion,
    execute_hard_purge,
    plan_hard_purge,
    preview_deletion,
    recover_pending_hard_purges,
    restore_deletion,
)
from meta_archive.services.inventory import scan_archive
from meta_archive.services.message_importer import import_messages

MESSAGE_FIXTURES = Path(__file__).parent / "fixtures" / "messages"
ACTIVITY_FIXTURES = Path(__file__).parent / "fixtures" / "activity"

ManifestVariant = Literal["regular", "symlink"]
OverlapStatus = Literal["previewed", "restored"]
TamperLocation = Literal["operation", "journal"]


@pytest.fixture
def deletion_archive(tmp_path: Path) -> Iterator[tuple[ArchiveLayout, Engine]]:
    layout = ArchiveLayout(tmp_path / "meta-backup-manifest-cleanup")
    shutil.copytree(MESSAGE_FIXTURES, layout.root)
    shutil.copytree(ACTIVITY_FIXTURES, layout.root / "activity-package")
    layout.ensure_runtime_directories()
    run_migrations(layout.database)
    engine = create_catalog_engine(layout.database)
    scan_archive(layout, engine, hash_files=True)
    import_messages(layout, engine)
    import_activities(layout, engine)
    build_blob_store(layout, engine)
    yield layout, engine
    engine.dispose()


def _message_preview(layout: ArchiveLayout, engine: Engine) -> DeletionPreview:
    return preview_deletion(
        layout,
        engine,
        DeletionSelector(entity_types=("message",), text="Older chunk", whole_field=True),
    )


def _commit(layout: ArchiveLayout, engine: Engine, preview: DeletionPreview) -> None:
    result = commit_deletion(
        layout,
        engine,
        preview.operation_id,
        commit_token=preview.commit_token,
        catalog_revision=preview.catalog_revision,
    )
    assert result.status == "committed"


def _leave_workspace_cleanup_pending(
    layout: ArchiveLayout,
    engine: Engine,
    preview: DeletionPreview,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    purge = plan_hard_purge(layout, engine, preview.operation_id, scope="workspace")
    assert purge.executable
    original_finalize = deletion_service._finalize_pending_hard_purge_locked

    def fail_before_cleanup(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("manifest cleanup guard fixture")

    monkeypatch.setattr(
        deletion_service,
        "_finalize_pending_hard_purge_locked",
        fail_before_cleanup,
    )
    with pytest.raises(RuntimeError, match="manifest cleanup guard fixture"):
        execute_hard_purge(
            layout,
            engine,
            preview.operation_id,
            scope="workspace",
            purge_token=purge.purge_token,
            purge_revision=purge.purge_revision,
            confirmation=preview.operation_id,
        )
    monkeypatch.setattr(
        deletion_service,
        "_finalize_pending_hard_purge_locked",
        original_finalize,
    )

    staging = layout.trash / "purge-staging" / preview.operation_id
    assert staging.is_dir()
    assert not Path(preview.manifest).exists()
    with Session(engine) as session:
        operation = session.get(DeletionOperation, preview.operation_id)
        assert operation is not None and operation.status == "cleanup_pending"
    return staging


def _forbid_recovery_unlinks(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_unlink(*_args: object, **_kwargs: object) -> None:
        pytest.fail("manifest validation must complete before any recovery unlink")

    monkeypatch.setattr(deletion_service, "_remove_tree_exact", fail_unlink)
    monkeypatch.setattr(deletion_service, "unlink_regular_file_beneath", fail_unlink)


def _tree_digest_snapshot(root: Path) -> dict[str, tuple[int, int, int, str]]:
    return {
        path.relative_to(root).as_posix(): (
            path.lstat().st_dev,
            path.lstat().st_ino,
            stat.S_IMODE(path.lstat().st_mode),
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _assert_cleanup_pending(engine: Engine, operation_id: str) -> None:
    with Session(engine) as session:
        operation = session.get(DeletionOperation, operation_id)
        assert operation is not None and operation.status == "cleanup_pending"


@pytest.mark.parametrize("variant", ["regular", "symlink"])
def test_cleanup_pending_workspace_rejects_recreated_current_manifest(
    deletion_archive: tuple[ArchiveLayout, Engine],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    variant: ManifestVariant,
) -> None:
    layout, engine = deletion_archive
    preview = _message_preview(layout, engine)
    _commit(layout, engine, preview)
    staging = _leave_workspace_cleanup_pending(layout, engine, preview, monkeypatch)
    staging_before = _tree_digest_snapshot(staging)

    manifest_path = Path(preview.manifest)
    payload = b"recreated current manifest must survive\n"
    target_path: Path | None = None
    if variant == "regular":
        manifest_path.write_bytes(payload)
        manifest_path.chmod(0o400)
    else:
        target_path = tmp_path / "recreated-current-manifest-target.json"
        target_path.write_bytes(payload)
        target_path.chmod(0o400)
        manifest_path.symlink_to(target_path)
    manifest_identity = manifest_path.lstat()
    target_identity = target_path.stat() if target_path is not None else None

    _forbid_recovery_unlinks(monkeypatch)
    with pytest.raises(
        HardPurgeBlockedError,
        match="restore manifest was unexpectedly recreated",
    ):
        recover_pending_hard_purges(layout, engine)

    assert manifest_path.lstat().st_ino == manifest_identity.st_ino
    assert manifest_path.lstat().st_dev == manifest_identity.st_dev
    assert _tree_digest_snapshot(staging) == staging_before
    if target_path is None:
        assert manifest_path.read_bytes() == payload
        assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o400
    else:
        assert manifest_path.is_symlink()
        assert os.readlink(manifest_path) == str(target_path)
        assert target_path.read_bytes() == payload
        assert target_identity is not None
        assert (target_path.stat().st_dev, target_path.stat().st_ino) == (
            target_identity.st_dev,
            target_identity.st_ino,
        )
        assert stat.S_IMODE(target_path.stat().st_mode) == 0o400
    _assert_cleanup_pending(engine, preview.operation_id)


@pytest.mark.parametrize("overlap_status", ["previewed", "restored"])
def test_cleanup_pending_workspace_rejects_recreated_overlap_manifest(
    deletion_archive: tuple[ArchiveLayout, Engine],
    monkeypatch: pytest.MonkeyPatch,
    overlap_status: OverlapStatus,
) -> None:
    layout, engine = deletion_archive
    overlap = _message_preview(layout, engine)
    if overlap_status == "restored":
        _commit(layout, engine, overlap)
        restored = restore_deletion(
            layout,
            engine,
            overlap.operation_id,
            confirmation=overlap.operation_id,
        )
        assert restored.status == "restored"
    current = _message_preview(layout, engine)
    _commit(layout, engine, current)
    staging = _leave_workspace_cleanup_pending(layout, engine, current, monkeypatch)
    staging_before = _tree_digest_snapshot(staging)

    overlap_manifest = Path(overlap.manifest)
    payload = f"recreated {overlap_status} overlap manifest\n".encode()
    overlap_manifest.write_bytes(payload)
    overlap_manifest.chmod(0o400)
    overlap_identity = overlap_manifest.stat()
    with Session(engine) as session:
        invalidated = session.get(DeletionOperation, overlap.operation_id)
        assert invalidated is not None
        assert invalidated.status == "invalidated"
        assert invalidated.manifest_relative_path is None

    _forbid_recovery_unlinks(monkeypatch)
    with pytest.raises(
        HardPurgeBlockedError,
        match="invalidated manifest was unexpectedly recreated",
    ):
        recover_pending_hard_purges(layout, engine)

    assert overlap_manifest.read_bytes() == payload
    assert (overlap_manifest.stat().st_dev, overlap_manifest.stat().st_ino) == (
        overlap_identity.st_dev,
        overlap_identity.st_ino,
    )
    assert stat.S_IMODE(overlap_manifest.stat().st_mode) == 0o400
    assert _tree_digest_snapshot(staging) == staging_before
    _assert_cleanup_pending(engine, current.operation_id)


@pytest.mark.parametrize("tamper_location", ["operation", "journal"])
def test_cleanup_pending_workspace_rejects_tampered_manifest_path_without_unlink(
    deletion_archive: tuple[ArchiveLayout, Engine],
    monkeypatch: pytest.MonkeyPatch,
    tamper_location: TamperLocation,
) -> None:
    layout, engine = deletion_archive
    preview = _message_preview(layout, engine)
    _commit(layout, engine, preview)
    staging = _leave_workspace_cleanup_pending(layout, engine, preview, monkeypatch)
    staging_before = _tree_digest_snapshot(staging)

    private_target = layout.catalog / "private-manifest-path-canary.json"
    private_payload = b"private 0400 file must never be unlinked\n"
    private_target.write_bytes(private_payload)
    private_target.chmod(0o400)
    private_identity = private_target.stat()
    malicious_relative = private_target.relative_to(layout.root).as_posix()
    with Session(engine) as session, session.begin():
        operation = session.get(DeletionOperation, preview.operation_id)
        assert operation is not None
        if tamper_location == "operation":
            operation.manifest_relative_path = malicious_relative
        else:
            operation.purge_plan_json = {
                **operation.purge_plan_json,
                "manifest_relative_path": malicious_relative,
            }

    _forbid_recovery_unlinks(monkeypatch)
    with pytest.raises(HardPurgeBlockedError, match="manifest path is invalid"):
        recover_pending_hard_purges(layout, engine)

    assert private_target.read_bytes() == private_payload
    assert (private_target.stat().st_dev, private_target.stat().st_ino) == (
        private_identity.st_dev,
        private_identity.st_ino,
    )
    assert stat.S_IMODE(private_target.stat().st_mode) == 0o400
    assert _tree_digest_snapshot(staging) == staging_before
    _assert_cleanup_pending(engine, preview.operation_id)
