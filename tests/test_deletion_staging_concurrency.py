from __future__ import annotations

import hashlib
import shutil
import threading
from datetime import UTC, datetime
from pathlib import Path

import lancedb  # type: ignore[import-untyped]
import pytest
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

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
    DeletionError,
    DeletionSelector,
    commit_deletion,
    execute_hard_purge,
    plan_hard_purge,
    preview_deletion,
    recover_pending_hard_purges,
)
from meta_archive.services.inventory import scan_archive
from meta_archive.services.media_derivatives import build_media_derivatives
from meta_archive.services.message_importer import import_messages

MESSAGE_FIXTURES = Path(__file__).parent / "fixtures" / "messages"


def _tree_digest_snapshot(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _add_fixture_vector_windows(
    layout: ArchiveLayout,
    engine: Engine,
    selected_message: Message,
) -> tuple[str, set[str]]:
    config_id = "staging-race-semantic-config"
    table_name = "staging_race_vectors"
    selected_window_id = "staging-race-selected-window"
    retained_window_id = "staging-race-retained-window"
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
                backend_version="fixture-v1",
                model_name="fixture-model",
                model_revision="fixture-revision",
                dimensions=3,
                distance_metric="cosine",
                vector_table=table_name,
                window_version="fixture-window-v1",
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
                    window_version="fixture-window-v1",
                    entity_type="message_window",
                    conversation_id=linked_message.conversation_id,
                    segment_id=linked_message.segment_id,
                    first_message_id=linked_message.id,
                    last_message_id=linked_message.id,
                    first_timestamp_ms=linked_message.timestamp_ms,
                    last_timestamp_ms=linked_message.timestamp_ms,
                    turn_count=1,
                    approx_token_count=4,
                    text_display="Fixture semantic staging window.",
                    text_search="fixture semantic staging window",
                    source_category="fixture",
                    participants_search=linked_message.sender_search,
                    source_ids_json={},
                    build_generation="fixture-generation",
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
    return table_name, {selected_window_id, retained_window_id}


def test_staged_workspace_rejects_publishers_and_preview_then_rolls_back_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = ArchiveLayout(tmp_path / "meta-backup-staging-race")
    shutil.copytree(MESSAGE_FIXTURES, layout.root)
    layout.ensure_runtime_directories()
    run_migrations(layout.database)
    engine = create_catalog_engine(layout.database)
    try:
        scan_archive(layout, engine, hash_files=True)
        import_messages(layout, engine)
        build_blob_store(layout, engine)
        with Session(engine) as session:
            message = session.scalar(
                select(Message).where(Message.content_display == "Older chunk")
            )
            assert message is not None
            session.expunge(message)
        table_name, expected_vector_ids = _add_fixture_vector_windows(layout, engine, message)

        selector = DeletionSelector(
            entity_types=("message",),
            text="Older chunk",
            whole_field=True,
        )
        preview = preview_deletion(layout, engine, selector)
        commit_deletion(
            layout,
            engine,
            preview.operation_id,
            commit_token=preview.commit_token,
            catalog_revision=preview.catalog_revision,
        )

        tree_markers = {
            layout.library / "staging-race.txt": b"library must return",
            layout.conflicts / "staging-race.json": b'{"conflict": "must return"}\n',
            layout.analytics_cache / "staging-race.json": b'{"cache": "must return"}\n',
        }
        for path, payload in tree_markers.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)

        purge = plan_hard_purge(layout, engine, preview.operation_id, scope="workspace")
        assert purge.executable, purge.blockers
        assert purge.impact["filesystem_library_present"] is True
        assert purge.impact["conflict_holding_files"] > 0
        assert purge.impact["analytics_cache_files"] > 0
        assert purge.impact["vector_store_present"] is True

        protected_trees = (
            layout.vectors,
            layout.library,
            layout.conflicts,
            layout.analytics_cache,
        )
        before = {path: _tree_digest_snapshot(path) for path in protected_trees}
        preview_count_before = 0
        with Session(engine) as session:
            preview_count_before = int(
                session.scalar(select(func.count(DeletionOperation.id))) or 0
            )

        staged = threading.Event()
        resume = threading.Event()
        failures: list[BaseException] = []
        original_verify = deletion_service._verify_precommit_targets

        def pause_after_staging(*args: object, **kwargs: object) -> None:
            original_verify(*args, **kwargs)
            if kwargs.get("restored") is False:
                staged.set()
                if not resume.wait(timeout=10):
                    raise TimeoutError("test did not release staged purge")

        def fail_catalog_delete(*_args: object, **_kwargs: object) -> dict[str, int]:
            raise RuntimeError("staging race rollback fixture")

        monkeypatch.setattr(
            deletion_service,
            "_verify_precommit_targets",
            pause_after_staging,
        )
        monkeypatch.setattr(deletion_service, "_delete_normalized_rows", fail_catalog_delete)

        def run_purge() -> None:
            try:
                execute_hard_purge(
                    layout,
                    engine,
                    preview.operation_id,
                    scope="workspace",
                    purge_token=purge.purge_token,
                    purge_revision=purge.purge_revision,
                    confirmation=preview.operation_id,
                )
            except BaseException as exc:
                failures.append(exc)

        worker = threading.Thread(target=run_purge)
        worker.start()
        try:
            assert staged.wait(timeout=10)
            assert all(not path.exists() for path in protected_trees)

            with pytest.raises(DeletionError, match="currently mutating"):
                build_media_derivatives(layout, engine)
            with pytest.raises(DeletionError, match="currently mutating"):
                preview_deletion(layout, engine, selector)

            assert all(not path.exists() for path in protected_trees)
        finally:
            resume.set()
            worker.join(timeout=15)

        assert not worker.is_alive()
        assert len(failures) == 1
        assert isinstance(failures[0], RuntimeError)
        assert "staging race rollback fixture" in str(failures[0])
        assert not (layout.trash / "purge-staging" / preview.operation_id).exists()
        assert recover_pending_hard_purges(layout, engine) == []
        assert {path: _tree_digest_snapshot(path) for path in protected_trees} == before

        restored_table = lancedb.connect(str(layout.vectors)).open_table(table_name)
        restored_ids = {
            str(row["id"]) for row in restored_table.search().select(["id"]).limit(10).to_list()
        }
        assert restored_ids == expected_vector_ids
        with Session(engine) as session:
            operation = session.get(DeletionOperation, preview.operation_id)
            selected_message = session.get(Message, message.pk)
            preview_count_after = int(session.scalar(select(func.count(DeletionOperation.id))) or 0)
            assert operation is not None and operation.status == "committed"
            assert selected_message is not None and selected_message.deleted_at is not None
            assert preview_count_after == preview_count_before
    finally:
        engine.dispose()
