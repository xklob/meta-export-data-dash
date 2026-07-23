from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

import lancedb  # type: ignore[import-untyped]
import pytest
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from meta_archive import paths as archive_paths
from meta_archive.database import create_catalog_engine, run_migrations
from meta_archive.models import (
    DeletionOperation,
    Message,
    SearchWindow,
    SearchWindowMessage,
    SemanticIndexConfig,
)
from meta_archive.paths import ArchiveLayout
from meta_archive.services import deletion as deletion_service
from meta_archive.services.consolidation import build_blob_store
from meta_archive.services.deletion import (
    DeletionSelector,
    HardPurgeBlockedError,
    commit_deletion,
    execute_hard_purge,
    plan_hard_purge,
    preview_deletion,
    restore_deletion,
)
from meta_archive.services.inventory import scan_archive
from meta_archive.services.library import build_filesystem_library
from meta_archive.services.message_importer import import_messages

MESSAGE_FIXTURES = Path(__file__).parent / "fixtures" / "messages"


@pytest.fixture
def precommit_archive(tmp_path: Path) -> Iterator[tuple[ArchiveLayout, Engine]]:
    layout = ArchiveLayout(tmp_path / "meta-backup-precommit-entrypoint")
    shutil.copytree(MESSAGE_FIXTURES, layout.root)
    layout.ensure_runtime_directories()
    run_migrations(layout.database)
    engine = create_catalog_engine(layout.database)
    scan_archive(layout, engine, hash_files=True)
    import_messages(layout, engine)
    build_blob_store(layout, engine)
    build_filesystem_library(layout, engine)
    yield layout, engine
    engine.dispose()


def _message(engine: Engine, content: str) -> Message:
    with Session(engine) as session:
        message = session.scalar(select(Message).where(Message.content_display == content))
        assert message is not None
        session.expunge(message)
        return message


def _preview_and_commit(
    layout: ArchiveLayout,
    engine: Engine,
    content: str,
) -> tuple[str, str]:
    preview = preview_deletion(
        layout,
        engine,
        DeletionSelector(entity_types=("message",), text=content, whole_field=True),
    )
    commit_deletion(
        layout,
        engine,
        preview.operation_id,
        commit_token=preview.commit_token,
        catalog_revision=preview.catalog_revision,
    )
    return preview.operation_id, preview.manifest


def _add_vector_fixture(
    layout: ArchiveLayout,
    engine: Engine,
    selected_message: Message,
) -> tuple[str, str, str]:
    config_id = "9" * 64
    selected_window_id = "8" * 64
    retained_window_id = "7" * 64
    table_name = "precommit_entrypoint_vectors"
    with Session(engine) as session, session.begin():
        retained_message = session.scalar(
            select(Message).where(Message.id != selected_message.id).order_by(Message.id)
        )
        assert retained_message is not None
        session.add(
            SemanticIndexConfig(
                id=config_id,
                active=True,
                status="ready",
                backend="fixture",
                backend_version="fixture-1",
                model_name="fixture-model",
                model_revision="fixture-revision",
                dimensions=3,
                distance_metric="cosine",
                document_prefix="",
                query_prefix="",
                vector_table=table_name,
                window_version="precommit-entrypoint-v1",
                window_count=2,
                audit_json={},
                completed_at=datetime.now(UTC),
            )
        )
        for window_id, linked_message in (
            (selected_window_id, selected_message),
            (retained_window_id, retained_message),
        ):
            session.add(
                SearchWindow(
                    id=window_id,
                    window_version="precommit-entrypoint-v1",
                    entity_type="message_window",
                    conversation_id=linked_message.conversation_id,
                    segment_id=linked_message.segment_id,
                    first_message_id=linked_message.id,
                    last_message_id=linked_message.id,
                    first_timestamp_ms=linked_message.timestamp_ms,
                    last_timestamp_ms=linked_message.timestamp_ms,
                    turn_count=1,
                    approx_token_count=4,
                    text_display="Precommit entrypoint recovery fixture.",
                    text_search="precommit entrypoint recovery fixture",
                    source_category="fixture",
                    participants_search=linked_message.sender_search,
                    source_ids_json={},
                    build_generation="precommit-entrypoint-generation",
                    embedding_config_hash=config_id,
                    embedding_updated_at=datetime.now(UTC),
                )
            )
            session.flush()
            session.add(
                SearchWindowMessage(
                    window_id=window_id,
                    message_id=linked_message.id,
                    ordinal=0,
                )
            )
    lancedb.connect(str(layout.vectors)).create_table(
        table_name,
        data=[
            {"id": selected_window_id, "vector": [1.0, 0.0, 0.0]},
            {"id": retained_window_id, "vector": [0.0, 1.0, 0.0]},
        ],
        mode="overwrite",
    )
    return table_name, selected_window_id, retained_window_id


def _tree_snapshot(root: Path) -> dict[str, tuple[int, int, str]]:
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): (
            stat.S_IMODE(path.stat().st_mode),
            path.stat().st_size,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def _workspace_snapshot(layout: ArchiveLayout) -> dict[str, tuple[int, int, str]]:
    database_paths = {
        layout.database,
        layout.database.with_name(f"{layout.database.name}-shm"),
        layout.database.with_name(f"{layout.database.name}-wal"),
    }
    snapshot: dict[str, tuple[int, int, str]] = {}
    for path in sorted(layout.root.rglob("*")):
        if not path.is_file() or path.is_symlink() or path in database_paths:
            continue
        relative = path.relative_to(layout.root).as_posix()
        if relative.startswith("trash/purge-staging/") or path.name.endswith(".lock"):
            continue
        path_stat = path.stat()
        snapshot[relative] = (
            stat.S_IMODE(path_stat.st_mode),
            path_stat.st_size,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    return snapshot


def _vector_ids(layout: ArchiveLayout, table_name: str) -> set[str]:
    table = lancedb.connect(str(layout.vectors)).open_table(table_name)
    return {str(row["id"]) for row in table.search().select(["id"]).limit(10).to_list()}


def _purge_catalog_snapshot(engine: Engine, operation_id: str, message_pk: int) -> str:
    """Return a stable rendering of the rows staging must not mutate."""

    with Session(engine) as session:
        operation = session.get(DeletionOperation, operation_id)
        message = session.get(Message, message_pk)
        assert operation is not None
        assert message is not None
        rows = {
            "operation": {
                column.name: getattr(operation, column.name)
                for column in DeletionOperation.__table__.columns
            },
            "message": {
                column.name: getattr(message, column.name) for column in Message.__table__.columns
            },
        }
    return json.dumps(rows, sort_keys=True, default=str)


def _fail_source_parent_fsync_after_replace(
    monkeypatch: pytest.MonkeyPatch,
    source: Path,
    *,
    after_replace: Callable[[], None] | None = None,
) -> tuple[Callable[[int], None], list[Path]]:
    """Inject one real post-rename failure at the emptied source parent."""

    real_fsync = os.fsync
    source_parent = source.parent.resolve()
    failed_at: list[Path] = []

    def fail_once(descriptor: int) -> None:
        try:
            descriptor_path = Path(os.readlink(f"/proc/self/fd/{descriptor}")).resolve()
        except (FileNotFoundError, OSError):
            descriptor_path = Path()
        if not failed_at and not source.exists() and descriptor_path == source_parent:
            if after_replace is not None:
                after_replace()
            failed_at.append(descriptor_path)
            raise OSError("simulated source-parent fsync failure after rename")
        real_fsync(descriptor)

    monkeypatch.setattr(archive_paths.os, "fsync", fail_once)
    return real_fsync, failed_at


def _leave_precommit_staging(
    monkeypatch: pytest.MonkeyPatch,
    layout: ArchiveLayout,
    engine: Engine,
    operation_id: str,
) -> tuple[object, object, object]:
    purge = plan_hard_purge(layout, engine, operation_id, scope="workspace")
    assert purge.executable, purge.blockers
    original_delete = deletion_service._delete_normalized_rows
    original_restore = deletion_service._restore_and_verify_staged_files
    original_remove_tree = deletion_service._remove_tree_exact

    def fail_before_sqlite_commit(*_args: object, **_kwargs: object) -> dict[str, int]:
        raise RuntimeError("simulated process death before SQLite commit")

    monkeypatch.setattr(deletion_service, "_delete_normalized_rows", fail_before_sqlite_commit)
    monkeypatch.setattr(
        deletion_service,
        "_restore_and_verify_staged_files",
        lambda _layout, _staging, _operation_id, _staged: None,
    )
    monkeypatch.setattr(
        deletion_service,
        "_remove_tree_exact",
        lambda _path, *, required_parent: None,
    )
    with pytest.raises(RuntimeError, match="simulated process death"):
        execute_hard_purge(
            layout,
            engine,
            operation_id,
            scope="workspace",
            purge_token=purge.purge_token,
            purge_revision=purge.purge_revision,
            confirmation=operation_id,
        )
    return original_delete, original_restore, original_remove_tree


def _restore_crash_patches(
    monkeypatch: pytest.MonkeyPatch,
    originals: tuple[object, object, object],
) -> None:
    original_delete, original_restore, original_remove_tree = originals
    monkeypatch.setattr(deletion_service, "_delete_normalized_rows", original_delete)
    monkeypatch.setattr(deletion_service, "_restore_and_verify_staged_files", original_restore)
    monkeypatch.setattr(deletion_service, "_remove_tree_exact", original_remove_tree)


def test_file_staging_post_rename_fsync_failure_restores_exactly_then_retries(
    precommit_archive: tuple[ArchiveLayout, Engine],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, engine = precommit_archive
    selected_message = _message(engine, "Older chunk")
    operation_id, manifest = _preview_and_commit(layout, engine, "Older chunk")
    manifest_path = Path(manifest)
    manifest_bytes = manifest_path.read_bytes()
    purge = plan_hard_purge(layout, engine, operation_id, scope="workspace")
    assert purge.executable, purge.blockers
    before_workspace = _workspace_snapshot(layout)
    before_catalog = _purge_catalog_snapshot(engine, operation_id, selected_message.pk)

    real_fsync, failed_at = _fail_source_parent_fsync_after_replace(
        monkeypatch,
        manifest_path,
    )
    with pytest.raises(HardPurgeBlockedError):
        execute_hard_purge(
            layout,
            engine,
            operation_id,
            scope="workspace",
            purge_token=purge.purge_token,
            purge_revision=purge.purge_revision,
            confirmation=operation_id,
        )
    monkeypatch.setattr(archive_paths.os, "fsync", real_fsync)

    staging = layout.trash / "purge-staging" / operation_id
    assert failed_at == [manifest_path.parent.resolve()]
    assert manifest_path.read_bytes() == manifest_bytes
    assert _workspace_snapshot(layout) == before_workspace
    assert _purge_catalog_snapshot(engine, operation_id, selected_message.pk) == before_catalog
    assert not staging.exists()

    result = execute_hard_purge(
        layout,
        engine,
        operation_id,
        scope="workspace",
        purge_token=purge.purge_token,
        purge_revision=purge.purge_revision,
        confirmation=operation_id,
    )
    assert result.status == "purged"


def test_tree_staging_post_rename_fsync_failure_restores_exactly_then_retries(
    precommit_archive: tuple[ArchiveLayout, Engine],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, engine = precommit_archive
    selected_message = _message(engine, "Older chunk")
    analytics_marker = layout.analytics_cache / "post-rename-tree.json"
    analytics_marker.write_bytes(b'{"private": "restore this exact tree"}\n')
    analytics_before = _tree_snapshot(layout.analytics_cache)
    operation_id, _manifest = _preview_and_commit(layout, engine, "Older chunk")
    purge = plan_hard_purge(layout, engine, operation_id, scope="normalized")
    assert purge.executable, purge.blockers
    assert purge.impact["analytics_cache_files"] > 0
    before_workspace = _workspace_snapshot(layout)
    before_catalog = _purge_catalog_snapshot(engine, operation_id, selected_message.pk)

    real_fsync, failed_at = _fail_source_parent_fsync_after_replace(
        monkeypatch,
        layout.analytics_cache,
    )
    with pytest.raises(HardPurgeBlockedError):
        execute_hard_purge(
            layout,
            engine,
            operation_id,
            scope="normalized",
            purge_token=purge.purge_token,
            purge_revision=purge.purge_revision,
            confirmation=operation_id,
        )
    monkeypatch.setattr(archive_paths.os, "fsync", real_fsync)

    staging = layout.trash / "purge-staging" / operation_id
    assert failed_at == [layout.analytics_cache.parent.resolve()]
    assert _tree_snapshot(layout.analytics_cache) == analytics_before
    assert _workspace_snapshot(layout) == before_workspace
    assert _purge_catalog_snapshot(engine, operation_id, selected_message.pk) == before_catalog
    assert not staging.exists()

    result = execute_hard_purge(
        layout,
        engine,
        operation_id,
        scope="normalized",
        purge_token=purge.purge_token,
        purge_revision=purge.purge_revision,
        confirmation=operation_id,
    )
    assert result.status == "normalized_purged"


def test_unprovable_post_rename_tree_rollback_retains_journal_until_recovery(
    precommit_archive: tuple[ArchiveLayout, Engine],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    layout, engine = precommit_archive
    selected_message = _message(engine, "Older chunk")
    analytics_marker = layout.analytics_cache / "post-rename-retained.json"
    analytics_marker.write_bytes(b'{"private": "retain staged original"}\n')
    analytics_before = _tree_snapshot(layout.analytics_cache)
    operation_id, _manifest = _preview_and_commit(layout, engine, "Older chunk")
    purge = plan_hard_purge(layout, engine, operation_id, scope="normalized")
    assert purge.executable, purge.blockers
    before_workspace = _workspace_snapshot(layout)
    before_catalog = _purge_catalog_snapshot(engine, operation_id, selected_message.pk)
    external = tmp_path / "external-post-rename-target"
    external.mkdir()
    external_marker = external / "must-survive.txt"
    external_marker.write_bytes(b"outside archive must remain untouched")

    def replace_live_tree_with_symlink() -> None:
        layout.analytics_cache.symlink_to(external, target_is_directory=True)

    real_fsync, failed_at = _fail_source_parent_fsync_after_replace(
        monkeypatch,
        layout.analytics_cache,
        after_replace=replace_live_tree_with_symlink,
    )
    with pytest.raises(HardPurgeBlockedError):
        execute_hard_purge(
            layout,
            engine,
            operation_id,
            scope="normalized",
            purge_token=purge.purge_token,
            purge_revision=purge.purge_revision,
            confirmation=operation_id,
        )
    monkeypatch.setattr(archive_paths.os, "fsync", real_fsync)

    staging = layout.trash / "purge-staging" / operation_id
    rollback_journal = staging / deletion_service.PURGE_ROLLBACK_JOURNAL_NAME
    staged_analytics = staging / layout.analytics_cache.relative_to(layout.root)
    assert failed_at == [layout.analytics_cache.parent.resolve()]
    assert layout.analytics_cache.is_symlink()
    assert external_marker.read_bytes() == b"outside archive must remain untouched"
    assert rollback_journal.is_file()
    journal = json.loads(rollback_journal.read_text(encoding="utf-8"))
    assert any(
        row["path"] == layout.analytics_cache.relative_to(layout.root).as_posix()
        for row in journal["targets"]
    )
    assert _tree_snapshot(staged_analytics) == analytics_before
    assert _purge_catalog_snapshot(engine, operation_id, selected_message.pk) == before_catalog

    layout.analytics_cache.unlink()
    assert deletion_service.recover_pending_hard_purges(layout, engine) == []
    assert not staging.exists()
    assert _tree_snapshot(layout.analytics_cache) == analytics_before
    assert _workspace_snapshot(layout) == before_workspace
    assert _purge_catalog_snapshot(engine, operation_id, selected_message.pk) != before_catalog
    with Session(engine) as session:
        operation = session.get(DeletionOperation, operation_id)
        assert operation is not None and operation.status == "committed"
        assert operation.purge_plan_json["precommit_recovery"]["completed"] is True
    assert external_marker.read_bytes() == b"outside archive must remain untouched"

    result = execute_hard_purge(
        layout,
        engine,
        operation_id,
        scope="normalized",
        purge_token=purge.purge_token,
        purge_revision=purge.purge_revision,
        confirmation=operation_id,
    )
    assert result.status == "normalized_purged"
    assert external_marker.read_bytes() == b"outside archive must remain untouched"


def test_new_trash_entry_root_is_fsynced_before_first_staging_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = ArchiveLayout(tmp_path / "meta-backup-new-trash-fsync")
    layout.root.mkdir()
    source = layout.catalog / "new-trash-private.bin"
    source.parent.mkdir()
    source_bytes = b"source must not move before the archive root is durable"
    source.write_bytes(source_bytes)
    operation_id = "new-trash-fsync-operation"
    assert not layout.trash.exists()
    staging = layout.trash / "purge-staging" / operation_id
    journal_path = staging / deletion_service.PURGE_ROLLBACK_JOURNAL_NAME
    events: list[str] = []
    real_fsync = os.fsync
    real_replace = os.replace

    def observe_fsync(descriptor: int) -> None:
        try:
            descriptor_path = Path(os.readlink(f"/proc/self/fd/{descriptor}")).resolve()
        except (FileNotFoundError, OSError):
            descriptor_path = Path()
        if journal_path.is_file() and descriptor_path == layout.root.resolve():
            events.append("root_fsync_after_journal")
        real_fsync(descriptor)

    def stop_at_first_staging_replace(
        source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        destination: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        if dst_dir_fd is not None:
            destination_parent = Path(os.readlink(f"/proc/self/fd/{dst_dir_fd}")).resolve()
            if destination_parent == staging or staging in destination_parent.parents:
                events.append("first_staging_replace")
                raise OSError("stop before first staging replace")
        real_replace(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(archive_paths.os, "fsync", observe_fsync)
    monkeypatch.setattr(archive_paths.os, "replace", stop_at_first_staging_replace)
    with pytest.raises(HardPurgeBlockedError):
        deletion_service._stage_workspace_files(
            layout,
            operation_id,
            {source},
            scope="workspace",
            tree_targets=set(),
            rollback_remove_paths=set(),
        )
    monkeypatch.setattr(archive_paths.os, "fsync", real_fsync)
    monkeypatch.setattr(archive_paths.os, "replace", real_replace)

    assert events == ["root_fsync_after_journal", "first_staging_replace"]
    assert not staging.exists()
    assert source.read_bytes() == source_bytes


def test_rollback_journal_publication_failure_removes_empty_staging_and_moves_nothing(
    precommit_archive: tuple[ArchiveLayout, Engine],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, engine = precommit_archive
    selected_message = _message(engine, "Older chunk")
    operation_id, _manifest = _preview_and_commit(layout, engine, "Older chunk")
    purge = plan_hard_purge(layout, engine, operation_id, scope="workspace")
    assert purge.executable, purge.blockers
    before_workspace = _workspace_snapshot(layout)
    before_catalog = _purge_catalog_snapshot(engine, operation_id, selected_message.pk)
    staging = layout.trash / "purge-staging" / operation_id
    failed_journal_fsync: list[Path] = []
    real_fsync = os.fsync
    real_move = deletion_service.replace_entry_beneath
    move_spy = Mock(wraps=real_move)

    def fail_journal_file_fsync(descriptor: int) -> None:
        try:
            descriptor_path = Path(os.readlink(f"/proc/self/fd/{descriptor}")).resolve()
        except (FileNotFoundError, OSError):
            descriptor_path = Path()
        if (
            not failed_journal_fsync
            and descriptor_path.parent == staging
            and descriptor_path.name.startswith(".rollback-")
        ):
            failed_journal_fsync.append(descriptor_path)
            raise OSError("simulated rollback-journal publication failure")
        real_fsync(descriptor)

    monkeypatch.setattr(archive_paths.os, "fsync", fail_journal_file_fsync)
    monkeypatch.setattr(deletion_service, "replace_entry_beneath", move_spy)
    with pytest.raises(OSError, match="rollback-journal publication failure"):
        execute_hard_purge(
            layout,
            engine,
            operation_id,
            scope="workspace",
            purge_token=purge.purge_token,
            purge_revision=purge.purge_revision,
            confirmation=operation_id,
        )
    monkeypatch.setattr(archive_paths.os, "fsync", real_fsync)
    monkeypatch.setattr(deletion_service, "replace_entry_beneath", real_move)

    assert len(failed_journal_fsync) == 1
    move_spy.assert_not_called()
    assert not staging.exists()
    assert _workspace_snapshot(layout) == before_workspace
    assert _purge_catalog_snapshot(engine, operation_id, selected_message.pk) == before_catalog

    result = execute_hard_purge(
        layout,
        engine,
        operation_id,
        scope="workspace",
        purge_token=purge.purge_token,
        purge_revision=purge.purge_revision,
        confirmation=operation_id,
    )
    assert result.status == "purged"


def test_direct_restore_recovers_precommit_files_before_restoring_rows(
    precommit_archive: tuple[ArchiveLayout, Engine],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, engine = precommit_archive
    selected_message = _message(engine, "Older chunk")
    table_name, selected_window_id, retained_window_id = _add_vector_fixture(
        layout, engine, selected_message
    )
    analytics_file = layout.analytics_cache / "precommit-entrypoint.json"
    analytics_file.write_text('{"private": "recover me"}\n')
    operation_id, manifest = _preview_and_commit(layout, engine, "Older chunk")
    before_workspace = _workspace_snapshot(layout)
    before_library = _tree_snapshot(layout.library)
    before_vectors = _tree_snapshot(layout.vectors)

    originals = _leave_precommit_staging(monkeypatch, layout, engine, operation_id)
    staging = layout.trash / "purge-staging" / operation_id
    assert (staging / deletion_service.PURGE_ROLLBACK_JOURNAL_NAME).is_file()
    assert not Path(manifest).exists()
    assert not layout.library.exists()
    assert not analytics_file.exists()
    assert not layout.vectors.exists()
    assert _vector_ids(ArchiveLayout(staging), table_name) == {
        selected_window_id,
        retained_window_id,
    }
    with Session(engine) as session:
        operation = session.get(DeletionOperation, operation_id)
        assert operation is not None and operation.status == "committed"

    _restore_crash_patches(monkeypatch, originals)
    original_recover = deletion_service._recover_precommit_purge_staging_locked
    recovery_observed: list[bool] = []

    def assert_exact_recovery_before_restore(
        recovery_layout: ArchiveLayout,
        recovery_engine: Engine,
    ) -> list[str]:
        with Session(recovery_engine) as session:
            operation = session.get(DeletionOperation, operation_id)
            assert operation is not None and operation.status == "committed"
        recovered = original_recover(recovery_layout, recovery_engine)
        assert recovered == [operation_id]
        assert _workspace_snapshot(recovery_layout) == before_workspace
        assert _tree_snapshot(recovery_layout.library) == before_library
        assert _tree_snapshot(recovery_layout.vectors) == before_vectors
        assert _vector_ids(recovery_layout, table_name) == {
            selected_window_id,
            retained_window_id,
        }
        with Session(recovery_engine) as session:
            operation = session.get(DeletionOperation, operation_id)
            assert operation is not None and operation.status == "committed"
        recovery_observed.append(True)
        return recovered

    monkeypatch.setattr(
        deletion_service,
        "_recover_precommit_purge_staging_locked",
        assert_exact_recovery_before_restore,
    )
    result = restore_deletion(layout, engine, operation_id, confirmation=operation_id)

    assert recovery_observed == [True]
    assert result.status == "restored"
    assert not staging.exists()
    assert Path(manifest).is_file()
    with Session(engine) as session:
        operation = session.get(DeletionOperation, operation_id)
        assert operation is not None and operation.status == "restored"
        restored_message = session.get(Message, selected_message.pk)
        assert restored_message is not None and restored_message.deleted_at is None


def test_direct_planner_recovers_unrelated_precommit_staging_before_planning(
    precommit_archive: tuple[ArchiveLayout, Engine],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, engine = precommit_archive
    crashed_operation_id, _manifest = _preview_and_commit(layout, engine, "Older chunk")
    planned_operation_id, _other_manifest = _preview_and_commit(layout, engine, "First group")
    before_workspace = _workspace_snapshot(layout)

    originals = _leave_precommit_staging(monkeypatch, layout, engine, crashed_operation_id)
    staging = layout.trash / "purge-staging" / crashed_operation_id
    assert (staging / deletion_service.PURGE_ROLLBACK_JOURNAL_NAME).is_file()
    _restore_crash_patches(monkeypatch, originals)

    planned = plan_hard_purge(layout, engine, planned_operation_id, scope="normalized")

    assert planned.executable, planned.blockers
    assert not staging.exists()
    assert _workspace_snapshot(layout) == before_workspace
    with Session(engine) as session:
        crashed = session.get(DeletionOperation, crashed_operation_id)
        planned_operation = session.get(DeletionOperation, planned_operation_id)
        assert crashed is not None and crashed.status == "committed"
        assert crashed.purge_plan_json["precommit_recovery"]["completed"] is True
        assert planned_operation is not None and planned_operation.status == "committed"
