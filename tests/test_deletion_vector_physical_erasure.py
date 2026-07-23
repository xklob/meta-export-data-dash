from __future__ import annotations

import hashlib
import shutil
import stat
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import lancedb  # type: ignore[import-untyped]
import pytest
from lancedb.index import FTS, IvfFlat  # type: ignore[import-untyped]
from sqlalchemy import Engine, func, insert, select
from sqlalchemy.orm import Session

from meta_archive.database import create_catalog_engine, run_migrations
from meta_archive.models import (
    DeletionOperation,
    Message,
    SearchWindow,
    SemanticIndexConfig,
    SemanticWindowEmbedding,
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
    recover_pending_hard_purges,
)
from meta_archive.services.inventory import scan_archive
from meta_archive.services.message_importer import import_messages

MESSAGE_FIXTURES = Path(__file__).parent / "fixtures" / "messages"
VECTOR_ROW_COUNT = 2_000
SENTINEL = "lance-private-selected-sentinel-8f3d17c2"


@dataclass(frozen=True, slots=True)
class VectorFixture:
    table_name: str
    config_id: str
    selected_window_id: str
    retained_window_ids: tuple[str, ...]


@pytest.fixture
def vector_archive(tmp_path: Path) -> Iterator[tuple[ArchiveLayout, Engine]]:
    layout = ArchiveLayout(tmp_path / "meta-backup-vector-physical-erasure")
    shutil.copytree(MESSAGE_FIXTURES, layout.root)
    layout.ensure_runtime_directories()
    run_migrations(layout.database)
    engine = create_catalog_engine(layout.database)
    scan_archive(layout, engine, hash_files=True)
    import_messages(layout, engine)
    build_blob_store(layout, engine)
    yield layout, engine
    engine.dispose()


def _message(engine: Engine, content: str) -> Message:
    with Session(engine) as session:
        message = session.scalar(select(Message).where(Message.content_display == content))
        assert message is not None
        session.expunge(message)
        return message


def _tree_snapshot(root: Path) -> dict[str, tuple[int, int, str]]:
    return {
        path.relative_to(root).as_posix(): (
            stat.S_IMODE(path.stat().st_mode),
            path.stat().st_size,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def _vector_files(layout: ArchiveLayout) -> list[Path]:
    if not layout.vectors.exists():
        return []
    return [
        path
        for path in sorted(layout.vectors.rglob("*"))
        if path.is_file() and not path.is_symlink()
    ]


def _publish_indexed_lance_table(
    layout: ArchiveLayout,
    *,
    table_name: str,
    rows: list[dict[str, object]],
) -> None:
    database = lancedb.connect(str(layout.vectors))
    table = database.create_table(table_name, data=rows, mode="overwrite")
    table.create_index(
        "vector",
        config=IvfFlat(distance_type="cosine", num_partitions=4),
        replace=True,
    )
    table.create_index("text", config=FTS(with_position=True), replace=True)
    assert table.count_rows() == len(rows)
    assert len(table.list_indices()) == 2
    assert any(row.get("text") == SENTINEL for row in rows)
    # Lance encodings may compress string columns. Keep an exact plaintext witness beside the
    # indexed table so the purge assertion also detects stale arbitrary files in the vector tree.
    (layout.vectors / "selected-row-plaintext-witness.txt").write_text(SENTINEL)
    assert any(SENTINEL.encode() in path.read_bytes() for path in _vector_files(layout))


def _add_large_registered_vector_table(
    layout: ArchiveLayout, engine: Engine, selected_message: Message
) -> VectorFixture:
    with Session(engine) as session:
        retained_message = session.scalar(
            select(Message).where(Message.id != selected_message.id).order_by(Message.id)
        )
        assert retained_message is not None
        session.expunge(retained_message)
    config_id = "9" * 64
    selected_window_id = "8" * 64
    retained_window_ids = tuple(f"{index:064x}" for index in range(1, VECTOR_ROW_COUNT))
    table_name = "physical_erasure_vectors"
    generated_at = datetime.now(UTC)
    selected_row = {
        "id": selected_window_id,
        "window_version": "physical-erasure-v1",
        "entity_type": "message_window",
        "conversation_id": selected_message.conversation_id,
        "segment_id": selected_message.segment_id,
        "first_message_id": selected_message.id,
        "last_message_id": selected_message.id,
        "first_timestamp_ms": selected_message.timestamp_ms,
        "last_timestamp_ms": selected_message.timestamp_ms,
        "turn_count": 1,
        "approx_token_count": 8,
        "text_display": "Selected semantic fixture.",
        "text_search": "selected semantic fixture",
        "source_category": "fixture",
        "participants_search": selected_message.sender_search,
        "source_ids_json": {},
        "build_generation": "physical-erasure-generation",
        "embedding_config_hash": config_id,
        "embedding_updated_at": generated_at,
    }
    retained_rows = [
        {
            "id": window_id,
            "window_version": "physical-erasure-v1",
            "entity_type": "message_window",
            "conversation_id": retained_message.conversation_id,
            "segment_id": retained_message.segment_id,
            "first_message_id": retained_message.id,
            "last_message_id": retained_message.id,
            "first_timestamp_ms": retained_message.timestamp_ms,
            "last_timestamp_ms": retained_message.timestamp_ms,
            "turn_count": 1,
            "approx_token_count": 8,
            "text_display": f"Retained semantic fixture {index}.",
            "text_search": f"retained semantic fixture {index}",
            "source_category": "fixture",
            "participants_search": retained_message.sender_search,
            "source_ids_json": {},
            "build_generation": "physical-erasure-generation",
            "embedding_config_hash": config_id,
            "embedding_updated_at": generated_at,
        }
        for index, window_id in enumerate(retained_window_ids, start=1)
    ]
    with Session(engine) as session, session.begin():
        session.add(
            SemanticIndexConfig(
                id=config_id,
                active=True,
                status="ready",
                backend="fixture",
                backend_version="fixture-1",
                model_name="fixture-model",
                model_revision="fixture-revision",
                dimensions=4,
                distance_metric="cosine",
                document_prefix="",
                query_prefix="",
                vector_table=table_name,
                window_version="physical-erasure-v1",
                window_count=VECTOR_ROW_COUNT,
                audit_json={},
                completed_at=generated_at,
            )
        )
        session.flush()
        session.execute(insert(SearchWindow), [selected_row, *retained_rows])
        session.execute(
            insert(SemanticWindowEmbedding),
            [
                {"config_id": config_id, "window_id": window_id}
                for window_id in (selected_window_id, *retained_window_ids)
            ],
        )

    lance_rows = [
        {
            "id": selected_window_id,
            "vector": [1.0, 0.0, 0.0, 0.0],
            "text": SENTINEL,
        },
        *[
            {
                "id": window_id,
                "vector": [
                    float(index % 3),
                    float(index % 5),
                    float(index % 7),
                    float(index % 11),
                ],
                "text": f"retained-vector-row-{index}",
            }
            for index, window_id in enumerate(retained_window_ids, start=1)
        ],
    ]
    _publish_indexed_lance_table(layout, table_name=table_name, rows=lance_rows)
    table = lancedb.connect(str(layout.vectors)).open_table(table_name)
    selected_payload = (
        table.search().where(f"id = '{selected_window_id}'").select(["text"]).limit(1).to_list()
    )
    assert selected_payload == [{"text": SENTINEL}]
    return VectorFixture(
        table_name=table_name,
        config_id=config_id,
        selected_window_id=selected_window_id,
        retained_window_ids=retained_window_ids,
    )


def _add_same_count_orphan_vector_table(
    layout: ArchiveLayout, engine: Engine, selected_message: Message
) -> VectorFixture:
    with Session(engine) as session:
        retained_message = session.scalar(
            select(Message).where(Message.id != selected_message.id).order_by(Message.id)
        )
        assert retained_message is not None
        session.expunge(retained_message)
    config_id = "a" * 64
    retained_window_id = "b" * 64
    orphan_window_id = "c" * 64
    table_name = "same_count_orphan_vectors"
    with Session(engine) as session, session.begin():
        session.add(
            SemanticIndexConfig(
                id=config_id,
                active=True,
                status="ready",
                backend="fixture",
                backend_version="fixture-1",
                model_name="fixture-model",
                model_revision="fixture-revision",
                dimensions=4,
                distance_metric="cosine",
                document_prefix="",
                query_prefix="",
                vector_table=table_name,
                window_version="orphan-erasure-v1",
                window_count=1,
                audit_json={},
                completed_at=datetime.now(UTC),
            )
        )
        session.flush()
        session.add(
            SearchWindow(
                id=retained_window_id,
                window_version="orphan-erasure-v1",
                entity_type="message_window",
                conversation_id=retained_message.conversation_id,
                segment_id=retained_message.segment_id,
                first_message_id=retained_message.id,
                last_message_id=retained_message.id,
                first_timestamp_ms=retained_message.timestamp_ms,
                last_timestamp_ms=retained_message.timestamp_ms,
                turn_count=1,
                approx_token_count=4,
                text_display="Retained orphan-control fixture.",
                text_search="retained orphan control fixture",
                source_category="fixture",
                participants_search=retained_message.sender_search,
                source_ids_json={},
                build_generation="orphan-erasure-generation",
                embedding_config_hash=config_id,
                embedding_updated_at=datetime.now(UTC),
            )
        )
        session.flush()
        session.add(
            SemanticWindowEmbedding(
                config_id=config_id,
                window_id=retained_window_id,
            )
        )
    table = lancedb.connect(str(layout.vectors)).create_table(
        table_name,
        data=[
            {
                "id": orphan_window_id,
                "vector": [1.0, 0.0, 0.0, 0.0],
                "text": SENTINEL,
            }
        ],
        mode="overwrite",
    )
    assert table.count_rows() == 1
    (layout.vectors / "selected-row-plaintext-witness.txt").write_text(SENTINEL)
    assert any(SENTINEL.encode() in path.read_bytes() for path in _vector_files(layout))
    return VectorFixture(
        table_name=table_name,
        config_id=config_id,
        selected_window_id=orphan_window_id,
        retained_window_ids=(retained_window_id,),
    )


def _recreate_identity_consistent_legacy_store(
    layout: ArchiveLayout,
    engine: Engine,
    fixture: VectorFixture,
) -> None:
    with Session(engine) as session, session.begin():
        config = session.get(SemanticIndexConfig, fixture.config_id)
        assert config is not None
        config.active = True
        config.status = "ready"
        config.window_count = len(fixture.retained_window_ids)
        session.execute(
            insert(SemanticWindowEmbedding),
            [
                {"config_id": fixture.config_id, "window_id": window_id}
                for window_id in fixture.retained_window_ids
            ],
        )
    rows: list[dict[str, object]] = [
        {
            "id": window_id,
            "vector": [
                float(index % 3),
                float(index % 5),
                float(index % 7),
                float(index % 11),
            ],
            # The first retained identity carries selected private plaintext. ID-only integrity
            # checks would accept this legacy/corrupt store, so delayed purge must erase it whole.
            "text": SENTINEL if index == 1 else f"legacy-retained-vector-row-{index}",
        }
        for index, window_id in enumerate(fixture.retained_window_ids, start=1)
    ]
    _publish_indexed_lance_table(layout, table_name=fixture.table_name, rows=rows)


def _committed_purge_plan(
    layout: ArchiveLayout,
    engine: Engine,
    *,
    scope: str,
) -> tuple[str, str, str, Path]:
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
    purge = plan_hard_purge(layout, engine, preview.operation_id, scope=scope)  # type: ignore[arg-type]
    assert purge.executable, purge.blockers
    return preview.operation_id, purge.purge_token, purge.purge_revision, Path(preview.manifest)


def _assert_vector_store_retired(
    layout: ArchiveLayout,
    engine: Engine,
    fixture: VectorFixture,
    operation_id: str,
    *,
    expected_status: str,
) -> None:
    vector_files = _vector_files(layout)
    assert all(SENTINEL.encode() not in path.read_bytes() for path in vector_files)
    assert not vector_files
    assert not layout.vectors.exists() or not any(layout.vectors.iterdir())
    assert not (layout.trash / "purge-staging" / operation_id).exists()
    with Session(engine) as session:
        operation = session.get(DeletionOperation, operation_id)
        config = session.get(SemanticIndexConfig, fixture.config_id)
        assert operation is not None and operation.status == expected_status
        assert operation.purge_plan_json["affected"]["semantic_store_discarded"] == 1
        assert config is not None
        assert config.status == "stale"
        assert config.active is False
        assert config.window_count == 0
        assert session.scalar(select(func.count(SemanticWindowEmbedding.window_id))) == 0
        assert session.get(SearchWindow, fixture.selected_window_id) is None
        assert session.scalar(
            select(func.count(SearchWindow.id)).where(
                SearchWindow.id.in_(fixture.retained_window_ids)
            )
        ) == len(fixture.retained_window_ids)
        assert (
            session.scalar(
                select(func.count(SearchWindow.id)).where(
                    SearchWindow.id.in_(fixture.retained_window_ids),
                    SearchWindow.embedding_config_hash.is_not(None),
                )
            )
            == 0
        )


def _prepare_vector_cleanup_pending(
    layout: ArchiveLayout,
    engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[str, Path]:
    selected_message = _message(engine, "Older chunk")
    _add_same_count_orphan_vector_table(layout, engine, selected_message)
    operation_id, purge_token, purge_revision, _manifest_path = _committed_purge_plan(
        layout, engine, scope="workspace"
    )
    original_finalize = deletion_service._finalize_pending_hard_purge_locked

    def fail_after_catalog_commit(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated stop before purge cleanup")

    monkeypatch.setattr(
        deletion_service,
        "_finalize_pending_hard_purge_locked",
        fail_after_catalog_commit,
    )
    with pytest.raises(RuntimeError, match="stop before purge cleanup"):
        execute_hard_purge(
            layout,
            engine,
            operation_id,
            scope="workspace",
            purge_token=purge_token,
            purge_revision=purge_revision,
            confirmation=operation_id,
        )
    monkeypatch.setattr(
        deletion_service,
        "_finalize_pending_hard_purge_locked",
        original_finalize,
    )

    staging = layout.trash / "purge-staging" / operation_id
    assert staging.is_dir()
    assert not layout.vectors.exists()
    with Session(engine) as session:
        operation = session.get(DeletionOperation, operation_id)
        assert operation is not None and operation.status == "cleanup_pending"
        assert operation.purged_at is None
        assert operation.purge_plan_json["vector_store_rewritten"] is True
    return operation_id, staging


@pytest.mark.parametrize(
    ("scope", "expected_status"),
    [("normalized", "normalized_purged"), ("workspace", "purged")],
)
def test_hard_purge_discards_entire_sentinel_bearing_lance_store(
    vector_archive: tuple[ArchiveLayout, Engine],
    scope: str,
    expected_status: str,
) -> None:
    layout, engine = vector_archive
    fixture = _add_large_registered_vector_table(layout, engine, _message(engine, "Older chunk"))
    operation_id, purge_token, purge_revision, _manifest_path = _committed_purge_plan(
        layout, engine, scope=scope
    )

    result = execute_hard_purge(
        layout,
        engine,
        operation_id,
        scope=scope,  # type: ignore[arg-type]
        purge_token=purge_token,
        purge_revision=purge_revision,
        confirmation=operation_id,
    )

    assert result.status == expected_status
    _assert_vector_store_retired(
        layout,
        engine,
        fixture,
        operation_id,
        expected_status=expected_status,
    )


def test_direct_purge_discards_same_count_orphan_store_without_selected_window_link(
    vector_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = vector_archive
    selected_message = _message(engine, "Older chunk")
    fixture = _add_same_count_orphan_vector_table(layout, engine, selected_message)
    with Session(engine) as session:
        assert (
            session.scalar(
                select(func.count(SearchWindow.id)).where(
                    SearchWindow.first_message_id == selected_message.id
                )
            )
            == 0
        )
        assert (
            session.scalar(
                select(func.count(SearchWindow.id)).where(
                    SearchWindow.last_message_id == selected_message.id
                )
            )
            == 0
        )
        assert session.scalar(select(func.count(SemanticWindowEmbedding.window_id))) == 1

    operation_id, purge_token, purge_revision, _manifest_path = _committed_purge_plan(
        layout, engine, scope="workspace"
    )
    with Session(engine) as session:
        operation = session.get(DeletionOperation, operation_id)
        assert operation is not None
        impact = operation.purge_plan_json["impact"]
        assert impact["search_windows"] == 0
        assert impact["vector_store_present"] is True
        assert impact["semantic_store_will_be_discarded"] is True
        assert impact["vector_catalog_count_mismatches"] == []

    result = execute_hard_purge(
        layout,
        engine,
        operation_id,
        scope="workspace",
        purge_token=purge_token,
        purge_revision=purge_revision,
        confirmation=operation_id,
    )

    assert result.status == "purged"
    _assert_vector_store_retired(
        layout,
        engine,
        fixture,
        operation_id,
        expected_status="purged",
    )
    with Session(engine) as session:
        operation = session.get(DeletionOperation, operation_id)
        assert operation is not None
        assert operation.purge_plan_json["vector_store_rewritten"] is True
        assert operation.purge_plan_json["deleted_vector_rows"] == 1


def test_cleanup_pending_recovery_rejects_recreated_live_vector_file(
    vector_archive: tuple[ArchiveLayout, Engine],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, engine = vector_archive
    operation_id, staging = _prepare_vector_cleanup_pending(layout, engine, monkeypatch)
    layout.vectors.mkdir(parents=True)
    sentinel_path = layout.vectors / "late-live-vector-sentinel.bin"
    sentinel_bytes = b"late cleanup_pending vector sentinel"
    sentinel_path.write_bytes(sentinel_bytes)

    with pytest.raises(HardPurgeBlockedError, match="unexpected residual files"):
        recover_pending_hard_purges(layout, engine)

    assert sentinel_path.read_bytes() == sentinel_bytes
    assert staging.is_dir()
    with Session(engine) as session:
        operation = session.get(DeletionOperation, operation_id)
        assert operation is not None and operation.status == "cleanup_pending"
        assert operation.purged_at is None


def test_cleanup_pending_recovery_rejects_live_vector_symlink_without_touching_target(
    vector_archive: tuple[ArchiveLayout, Engine],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    layout, engine = vector_archive
    operation_id, staging = _prepare_vector_cleanup_pending(layout, engine, monkeypatch)
    external_target = tmp_path / "external-vector-symlink-target.bin"
    external_bytes = b"external target must survive recovery"
    external_target.write_bytes(external_bytes)
    layout.vectors.mkdir(parents=True)
    symlink_path = layout.vectors / "late-live-vector-symlink"
    symlink_path.symlink_to(external_target)

    with pytest.raises(HardPurgeBlockedError, match="unexpected residual files"):
        recover_pending_hard_purges(layout, engine)

    assert symlink_path.is_symlink()
    assert symlink_path.resolve() == external_target.resolve()
    assert external_target.read_bytes() == external_bytes
    assert staging.is_dir()
    with Session(engine) as session:
        operation = session.get(DeletionOperation, operation_id)
        assert operation is not None and operation.status == "cleanup_pending"
        assert operation.purged_at is None


def test_delayed_workspace_upgrade_discards_identity_consistent_legacy_vector_tree(
    vector_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = vector_archive
    fixture = _add_large_registered_vector_table(layout, engine, _message(engine, "Older chunk"))
    operation_id, purge_token, purge_revision, _manifest_path = _committed_purge_plan(
        layout, engine, scope="normalized"
    )
    normalized = execute_hard_purge(
        layout,
        engine,
        operation_id,
        scope="normalized",
        purge_token=purge_token,
        purge_revision=purge_revision,
        confirmation=operation_id,
    )
    assert normalized.status == "normalized_purged"
    assert not _vector_files(layout)

    _recreate_identity_consistent_legacy_store(layout, engine, fixture)
    assert any(SENTINEL.encode() in path.read_bytes() for path in _vector_files(layout))
    workspace_plan = plan_hard_purge(layout, engine, operation_id, scope="workspace")
    assert workspace_plan.executable, workspace_plan.blockers
    assert workspace_plan.impact["vector_store_present"] is True
    assert workspace_plan.impact["semantic_store_will_be_discarded"] is True
    assert workspace_plan.impact["vector_catalog_count_mismatches"] == []
    assert workspace_plan.impact["vector_catalog_identity_mismatches"] == []

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
    _assert_vector_store_retired(
        layout,
        engine,
        fixture,
        operation_id,
        expected_status="purged",
    )
    with Session(engine) as session:
        operation = session.get(DeletionOperation, operation_id)
        assert operation is not None
        assert operation.purge_plan_json["vector_store_rewritten"] is True
        assert operation.purge_plan_json["affected"]["semantic_store_discarded"] == 1


def test_vector_tree_precommit_crash_recovers_exact_tree_then_retry_erases_it(
    vector_archive: tuple[ArchiveLayout, Engine],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, engine = vector_archive
    fixture = _add_large_registered_vector_table(layout, engine, _message(engine, "Older chunk"))
    operation_id, purge_token, purge_revision, manifest_path = _committed_purge_plan(
        layout, engine, scope="normalized"
    )
    vector_before = _tree_snapshot(layout.vectors)
    assert vector_before

    original_delete = deletion_service._delete_normalized_rows
    original_restore = deletion_service._restore_and_verify_staged_files
    original_remove_tree = deletion_service._remove_tree_exact

    def fail_before_sqlite_commit(*_args: object, **_kwargs: object) -> dict[str, int]:
        raise RuntimeError("simulated vector purge process death")

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
    with pytest.raises(RuntimeError, match="vector purge process death"):
        execute_hard_purge(
            layout,
            engine,
            operation_id,
            scope="normalized",
            purge_token=purge_token,
            purge_revision=purge_revision,
            confirmation=operation_id,
        )

    staging = layout.trash / "purge-staging" / operation_id
    staged_vectors = staging / layout.vectors.relative_to(layout.root)
    assert not layout.vectors.exists()
    assert _tree_snapshot(staged_vectors) == vector_before
    assert manifest_path.is_file()
    with Session(engine) as session:
        operation = session.get(DeletionOperation, operation_id)
        config = session.get(SemanticIndexConfig, fixture.config_id)
        assert operation is not None and operation.status == "committed"
        assert config is not None and config.active and config.status == "ready"
        assert config.window_count == VECTOR_ROW_COUNT
        assert (
            session.scalar(select(func.count(SemanticWindowEmbedding.window_id)))
            == VECTOR_ROW_COUNT
        )

    monkeypatch.setattr(deletion_service, "_delete_normalized_rows", original_delete)
    monkeypatch.setattr(deletion_service, "_restore_and_verify_staged_files", original_restore)
    monkeypatch.setattr(deletion_service, "_remove_tree_exact", original_remove_tree)
    recover_pending_hard_purges(layout, engine)

    assert not staging.exists()
    assert _tree_snapshot(layout.vectors) == vector_before
    assert any(SENTINEL.encode() in path.read_bytes() for path in _vector_files(layout))
    with Session(engine) as session:
        operation = session.get(DeletionOperation, operation_id)
        assert operation is not None and operation.status == "committed"
        assert operation.purge_plan_json["precommit_recovery"]["completed"] is True
        assert session.get(SearchWindow, fixture.selected_window_id) is not None

    result = execute_hard_purge(
        layout,
        engine,
        operation_id,
        scope="normalized",
        purge_token=purge_token,
        purge_revision=purge_revision,
        confirmation=operation_id,
    )
    assert result.status == "normalized_purged"
    _assert_vector_store_retired(
        layout,
        engine,
        fixture,
        operation_id,
        expected_status="normalized_purged",
    )
