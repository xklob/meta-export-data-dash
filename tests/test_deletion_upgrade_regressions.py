from __future__ import annotations

import hashlib
import json
import shutil
import stat
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lancedb  # type: ignore[import-untyped]
import pytest
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from meta_archive.database import create_catalog_engine, run_migrations
from meta_archive.models import (
    AttachmentOccurrence,
    DeletionOperation,
    Message,
    SearchWindow,
    SearchWindowMessage,
    SemanticIndexConfig,
    SemanticWindowEmbedding,
    SourceFile,
    SourceRecord,
)
from meta_archive.paths import ArchiveLayout
from meta_archive.services import deletion as deletion_service
from meta_archive.services.consolidation import build_blob_store
from meta_archive.services.deletion import (
    DeletionSelector,
    commit_deletion,
    execute_hard_purge,
    plan_hard_purge,
    preview_deletion,
    recover_pending_hard_purges,
)
from meta_archive.services.inventory import scan_archive
from meta_archive.services.message_importer import import_messages

MESSAGE_FIXTURES = Path(__file__).parent / "fixtures" / "messages"


@pytest.fixture
def upgrade_archive(tmp_path: Path) -> Iterator[tuple[ArchiveLayout, Engine]]:
    layout = ArchiveLayout(tmp_path / "meta-backup-deletion-upgrade")
    shutil.copytree(MESSAGE_FIXTURES, layout.root)
    layout.ensure_runtime_directories()
    run_migrations(layout.database)
    engine = create_catalog_engine(layout.database)
    scan_archive(layout, engine, hash_files=True)
    import_messages(layout, engine)
    build_blob_store(layout, engine)
    yield layout, engine
    engine.dispose()


def _row_values(row: Any) -> dict[str, Any]:
    return {column.name: getattr(row, column.name) for column in row.__table__.columns}


def _message(engine: Engine) -> Message:
    with Session(engine) as session:
        message = session.scalar(select(Message).where(Message.content_display == "Older chunk"))
        assert message is not None
        session.expunge(message)
        return message


def _add_selected_search_window(engine: Engine, message: Message) -> tuple[str, dict[str, Any]]:
    window = SearchWindow(
        id="d" * 64,
        window_version="upgrade-regression-v1",
        entity_type="message_window",
        conversation_id=message.conversation_id,
        segment_id=message.segment_id,
        activity_id=None,
        audio_segment_id=None,
        attachment_extraction_id=None,
        attachment_occurrence_id=None,
        first_message_id=message.id,
        last_message_id=message.id,
        first_timestamp_ms=message.timestamp_ms,
        last_timestamp_ms=message.timestamp_ms,
        turn_count=1,
        approx_token_count=4,
        text_display="Selected derived row fixture.",
        text_search="selected derived row fixture",
        source_category="fixture",
        participants_search=message.sender_search,
        source_ids_json={"messages": [message.id]},
        build_generation="upgrade-regression-generation",
        embedding_config_hash=None,
        embedding_updated_at=None,
    )
    with Session(engine) as session, session.begin():
        session.add(window)
        session.flush()
        values = _row_values(window)
    return str(values["id"]), values


def _add_registered_vector_pair(
    layout: ArchiveLayout, engine: Engine, selected_message: Message
) -> tuple[str, str, str]:
    config_id = "9" * 64
    selected_window_id = "8" * 64
    retained_window_id = "7" * 64
    table_name = "upgrade_regression_vectors"
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
                window_version="upgrade-regression-v1",
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
                    window_version="upgrade-regression-v1",
                    entity_type="message_window",
                    conversation_id=linked_message.conversation_id,
                    segment_id=linked_message.segment_id,
                    first_message_id=linked_message.id,
                    last_message_id=linked_message.id,
                    first_timestamp_ms=linked_message.timestamp_ms,
                    last_timestamp_ms=linked_message.timestamp_ms,
                    turn_count=1,
                    approx_token_count=4,
                    text_display="Registered vector fixture.",
                    text_search="registered vector fixture",
                    source_category="fixture",
                    participants_search=linked_message.sender_search,
                    source_ids_json={},
                    build_generation="upgrade-regression-generation",
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
    database = lancedb.connect(str(layout.vectors))
    database.create_table(
        table_name,
        data=[
            {"id": selected_window_id, "vector": [1.0, 0.0, 0.0]},
            {"id": retained_window_id, "vector": [0.0, 1.0, 0.0]},
        ],
        mode="overwrite",
    )
    return table_name, selected_window_id, retained_window_id


def _normalize_message(
    layout: ArchiveLayout,
    engine: Engine,
) -> tuple[Message, str, Path]:
    message = _message(engine)
    preview = preview_deletion(
        layout,
        engine,
        DeletionSelector(
            entity_types=("message",),
            text="Older chunk",
            whole_field=True,
        ),
    )
    commit_deletion(
        layout,
        engine,
        preview.operation_id,
        commit_token=preview.commit_token,
        catalog_revision=preview.catalog_revision,
    )
    purge = plan_hard_purge(layout, engine, preview.operation_id, scope="normalized")
    assert purge.executable, purge.blockers
    result = execute_hard_purge(
        layout,
        engine,
        preview.operation_id,
        scope="normalized",
        purge_token=purge.purge_token,
        purge_revision=purge.purge_revision,
        confirmation=preview.operation_id,
    )
    assert result.status == "normalized_purged"
    return message, preview.operation_id, Path(preview.manifest)


def _filesystem_snapshot(layout: ArchiveLayout) -> dict[str, tuple[int, int, str]]:
    database_paths = {
        layout.database,
        layout.database.with_name(f"{layout.database.name}-shm"),
        layout.database.with_name(f"{layout.database.name}-wal"),
    }
    snapshot: dict[str, tuple[int, int, str]] = {}
    for path in sorted(layout.root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        if path in database_paths or path.name.endswith(".lock"):
            continue
        relative = path.relative_to(layout.root).as_posix()
        if relative.startswith("trash/purge-staging/"):
            continue
        path_stat = path.stat()
        snapshot[relative] = (
            stat.S_IMODE(path_stat.st_mode),
            path_stat.st_size,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    return snapshot


def test_normalized_workspace_precommit_crash_recovers_exact_state_then_retries(
    upgrade_archive: tuple[ArchiveLayout, Engine],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, engine = upgrade_archive
    _message_row, operation_id, manifest_path = _normalize_message(layout, engine)
    workspace_plan = plan_hard_purge(layout, engine, operation_id, scope="workspace")
    assert workspace_plan.executable, workspace_plan.blockers
    before = _filesystem_snapshot(layout)

    original_apply = deletion_service.apply_raw_rewrite_catalog
    original_restore = deletion_service._restore_and_verify_staged_files
    original_remove_tree = deletion_service._remove_tree_exact

    def fail_before_sqlite_commit(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated delayed-upgrade process death")

    monkeypatch.setattr(deletion_service, "apply_raw_rewrite_catalog", fail_before_sqlite_commit)
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
    with pytest.raises(RuntimeError, match="delayed-upgrade process death"):
        execute_hard_purge(
            layout,
            engine,
            operation_id,
            scope="workspace",
            purge_token=workspace_plan.purge_token,
            purge_revision=workspace_plan.purge_revision,
            confirmation=operation_id,
        )

    staging = layout.trash / "purge-staging" / operation_id
    assert (staging / deletion_service.PURGE_ROLLBACK_JOURNAL_NAME).is_file()
    assert not manifest_path.exists()
    with Session(engine) as session:
        operation = session.get(DeletionOperation, operation_id)
        assert operation is not None and operation.status == "normalized_purged"

    monkeypatch.setattr(deletion_service, "apply_raw_rewrite_catalog", original_apply)
    monkeypatch.setattr(deletion_service, "_restore_and_verify_staged_files", original_restore)
    monkeypatch.setattr(deletion_service, "_remove_tree_exact", original_remove_tree)
    recover_pending_hard_purges(layout, engine)

    assert not staging.exists()
    assert _filesystem_snapshot(layout) == before
    with Session(engine) as session:
        operation = session.get(DeletionOperation, operation_id)
        assert operation is not None and operation.status == "normalized_purged"
        assert operation.purge_plan_json["precommit_recovery"]["completed"] is True

    result = execute_hard_purge(
        layout,
        engine,
        operation_id,
        scope="workspace",
        purge_token=workspace_plan.purge_token,
        purge_revision=workspace_plan.purge_revision,
        confirmation=operation_id,
    )
    assert result.status == "purged"
    assert not manifest_path.exists()


def test_upgrade_state_post_publish_failure_removes_orphan_then_retries(
    upgrade_archive: tuple[ArchiveLayout, Engine],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, engine = upgrade_archive
    message = _message(engine)
    preview = preview_deletion(
        layout,
        engine,
        DeletionSelector(entity_types=("message",), text="Older chunk", whole_field=True),
    )
    commit_deletion(
        layout,
        engine,
        preview.operation_id,
        commit_token=preview.commit_token,
        catalog_revision=preview.catalog_revision,
    )
    purge = plan_hard_purge(layout, engine, preview.operation_id, scope="normalized")
    assert purge.executable, purge.blockers
    state_path = Path(preview.manifest).parent / "workspace-upgrade.json"
    original_publish = deletion_service.publish_workspace_upgrade_state

    def publish_then_fail(*args: Any, **kwargs: Any) -> None:
        original_publish(*args, **kwargs)
        raise RuntimeError("simulated post-publication failure")

    monkeypatch.setattr(deletion_service, "publish_workspace_upgrade_state", publish_then_fail)
    with pytest.raises(RuntimeError, match="post-publication failure"):
        execute_hard_purge(
            layout,
            engine,
            preview.operation_id,
            scope="normalized",
            purge_token=purge.purge_token,
            purge_revision=purge.purge_revision,
            confirmation=preview.operation_id,
        )

    assert not state_path.exists()
    assert not (layout.trash / "purge-staging" / preview.operation_id).exists()
    with Session(engine) as session:
        operation = session.get(DeletionOperation, preview.operation_id)
        assert operation is not None and operation.status == "committed"
        assert session.get(Message, message.pk) is not None

    monkeypatch.setattr(deletion_service, "publish_workspace_upgrade_state", original_publish)
    result = execute_hard_purge(
        layout,
        engine,
        preview.operation_id,
        scope="normalized",
        purge_token=purge.purge_token,
        purge_revision=purge.purge_revision,
        confirmation=preview.operation_id,
    )
    assert result.status == "normalized_purged"
    assert state_path.is_file()


def test_source_manifest_drift_blocks_delayed_upgrade_without_clobbering(
    upgrade_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = upgrade_archive
    _message_row, operation_id, manifest_path = _normalize_message(layout, engine)
    relocation_before = layout.relocation_manifest.read_bytes()
    rows = [
        json.loads(line)
        for line in layout.original_files_manifest.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["regression_drift_marker"] = True
    changed_original = b"".join(json.dumps(row, sort_keys=True).encode() + b"\n" for row in rows)
    layout.original_files_manifest.write_bytes(changed_original)

    workspace_plan = plan_hard_purge(layout, engine, operation_id, scope="workspace")

    assert not workspace_plan.executable
    assert "source_manifests_changed_after_normalized_purge" in workspace_plan.blockers
    assert layout.original_files_manifest.read_bytes() == changed_original
    assert layout.relocation_manifest.read_bytes() == relocation_before
    assert manifest_path.is_file()
    with Session(engine) as session:
        operation = session.get(DeletionOperation, operation_id)
        assert operation is not None and operation.status == "normalized_purged"


def test_delayed_upgrade_discards_recreated_same_count_wrong_lance_store(
    upgrade_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = upgrade_archive
    message = _message(engine)
    table_name, selected_window_id, retained_window_id = _add_registered_vector_pair(
        layout, engine, message
    )
    _message_row, operation_id, manifest_path = _normalize_message(layout, engine)

    assert not layout.vectors.exists() or not any(layout.vectors.iterdir())
    with Session(engine) as session, session.begin():
        config = session.get(SemanticIndexConfig, "9" * 64)
        assert config is not None and config.status == "stale" and not config.active
        config.window_count = 1
        session.add(
            SemanticWindowEmbedding(
                config_id=config.id,
                window_id=retained_window_id,
            )
        )
    table = lancedb.connect(str(layout.vectors)).create_table(
        table_name,
        data=[{"id": selected_window_id, "vector": [0.0, 0.0, 1.0]}],
        mode="overwrite",
    )
    assert table.count_rows() == 1
    assert {str(row["id"]) for row in table.search().select(["id"]).limit(10).to_list()} == {
        selected_window_id
    }

    workspace_plan = plan_hard_purge(layout, engine, operation_id, scope="workspace")

    assert workspace_plan.executable
    assert workspace_plan.impact["vector_catalog_count_mismatches"] == []
    assert workspace_plan.impact["vector_catalog_identity_mismatches"]
    result = execute_hard_purge(
        layout,
        engine,
        operation_id,
        scope="workspace",
        purge_token=workspace_plan.purge_token,
        purge_revision=workspace_plan.purge_revision,
        confirmation=operation_id,
    )

    assert result.status == "purged"
    assert not manifest_path.exists()
    assert not layout.vectors.exists() or not any(layout.vectors.iterdir())
    with Session(engine) as session:
        operation = session.get(DeletionOperation, operation_id)
        config = session.get(SemanticIndexConfig, "9" * 64)
        assert operation is not None and operation.status == "purged"
        assert operation.purge_plan_json["vector_store_rewritten"] is True
        assert config is not None and config.status == "stale" and not config.active
        assert config.window_count == 0
        assert session.get(SearchWindow, selected_window_id) is None
        assert session.get(SearchWindow, retained_window_id) is not None


@pytest.mark.parametrize(
    ("recreated_kind", "expected_blocker"),
    [
        ("selected", "normalized_rows_recreated_after_normalized_purge"),
        ("source", "source_records_recreated_after_normalized_purge"),
        ("derived", "derived_rows_recreated_after_normalized_purge"),
    ],
)
def test_recreated_rows_block_delayed_upgrade(
    upgrade_archive: tuple[ArchiveLayout, Engine],
    recreated_kind: str,
    expected_blocker: str,
) -> None:
    layout, engine = upgrade_archive
    message = _message(engine)
    with Session(engine) as session:
        source_record = session.get(SourceRecord, message.source_record_id)
        assert source_record is not None
        source_record_values = _row_values(source_record)
    window_id, window_values = _add_selected_search_window(engine, message)
    message_values = _row_values(message)

    _message_row, operation_id, manifest_path = _normalize_message(layout, engine)
    with Session(engine) as session:
        assert session.get(Message, message.pk) is None
        assert session.get(SourceRecord, message.source_record_id) is None
        assert session.get(SearchWindow, window_id) is None

    with Session(engine) as session, session.begin():
        if recreated_kind in {"selected", "source"}:
            session.add(SourceRecord(**source_record_values))
            session.flush()
        if recreated_kind == "selected":
            session.add(Message(**message_values))
        elif recreated_kind == "derived":
            session.add(SearchWindow(**window_values))

    workspace_plan = plan_hard_purge(layout, engine, operation_id, scope="workspace")

    assert not workspace_plan.executable
    assert expected_blocker in workspace_plan.blockers
    assert manifest_path.is_file()
    with Session(engine) as session:
        operation = session.get(DeletionOperation, operation_id)
        assert operation is not None and operation.status == "normalized_purged"


@pytest.mark.parametrize(
    ("reference_kind", "expected_blocker"),
    [
        ("source_file", "new_source_file_references_block_workspace_upgrade"),
        ("attachment", "new_attachment_references_block_workspace_upgrade"),
    ],
)
def test_late_blob_reference_blocks_delayed_upgrade(
    upgrade_archive: tuple[ArchiveLayout, Engine],
    reference_kind: str,
    expected_blocker: str,
) -> None:
    layout, engine = upgrade_archive
    message = _message(engine)
    with Session(engine) as session:
        source_file = session.get(SourceFile, message.source_file_id)
        assert source_file is not None and source_file.sha256 is not None
        source_file_values = _row_values(source_file)
        old_sha256 = source_file.sha256

    _message_row, operation_id, manifest_path = _normalize_message(layout, engine)
    with Session(engine) as session, session.begin():
        if reference_kind == "source_file":
            source_file_values.pop("id")
            source_file_values.pop("created_at")
            source_file_values.pop("updated_at")
            source_file_values["relative_path"] = (
                f"{source_file_values['relative_path']}.late-reference"
            )
            source_file_values["normalized_path"] = (
                f"{source_file_values['normalized_path']}.late-reference"
            )
            session.add(SourceFile(**source_file_values))
        else:
            session.add(
                AttachmentOccurrence(
                    id="e" * 64,
                    message_id=None,
                    activity_record_id=None,
                    conversation_id=None,
                    source_file_id=None,
                    resolved_source_file_id=None,
                    blob_sha256=old_sha256,
                    kind="files",
                    ordinal=0,
                    source_uri=None,
                    filename="late-reference.bin",
                    media_type="application/octet-stream",
                    size_bytes=None,
                    timestamp_ms=None,
                    availability="unlinked",
                    metadata_json={"synthetic": True},
                )
            )

    workspace_plan = plan_hard_purge(layout, engine, operation_id, scope="workspace")

    assert not workspace_plan.executable
    assert expected_blocker in workspace_plan.blockers
    assert manifest_path.is_file()
    with Session(engine) as session:
        operation = session.get(DeletionOperation, operation_id)
        assert operation is not None and operation.status == "normalized_purged"
