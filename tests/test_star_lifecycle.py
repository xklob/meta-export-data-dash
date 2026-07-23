from __future__ import annotations

import json
import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

from meta_archive.database import create_catalog_engine, run_migrations
from meta_archive.models import (
    ActivityRecord,
    AttachmentOccurrence,
    Blob,
    Conversation,
    ConversationSegment,
    Message,
    Participant,
    SegmentParticipant,
    SourceChunk,
    SourceFile,
    Star,
)
from meta_archive.paths import ArchiveLayout
from meta_archive.services import activity_importer as activity_importer_service
from meta_archive.services import message_importer as message_importer_service
from meta_archive.services.activity_importer import import_activities
from meta_archive.services.consolidation import build_blob_store
from meta_archive.services.conversation_corrections import (
    conversation_identity_revision,
    merge_direct_histories,
    revert_conversation_correction,
    split_direct_history,
)
from meta_archive.services.deletion import (
    DeletionSelector,
    StaleDeletionPreviewError,
    commit_deletion,
    execute_hard_purge,
    plan_hard_purge,
    preview_deletion,
    restore_deletion,
)
from meta_archive.services.identifiers import stable_id
from meta_archive.services.inventory import scan_archive
from meta_archive.services.message_importer import import_messages
from meta_archive.services.stars import resolve_star_targets

MESSAGE_FIXTURES = Path(__file__).parent / "fixtures" / "messages"
ACTIVITY_FIXTURES = Path(__file__).parent / "fixtures" / "activity"


@pytest.fixture
def star_lifecycle_archive(tmp_path: Path) -> Iterator[tuple[ArchiveLayout, Engine]]:
    layout = ArchiveLayout(tmp_path / "meta-backup-stars")
    shutil.copytree(MESSAGE_FIXTURES, layout.root)
    layout.ensure_runtime_directories()
    run_migrations(layout.database)
    engine = create_catalog_engine(layout.database)
    scan_archive(layout, engine, hash_files=True)
    import_messages(layout, engine)
    build_blob_store(layout, engine)
    yield layout, engine
    engine.dispose()


@pytest.fixture
def star_activity_archive(tmp_path: Path) -> Iterator[tuple[ArchiveLayout, Engine, Path]]:
    layout = ArchiveLayout(tmp_path / "meta-backup-activity-stars")
    source_relative = Path("your_facebook_activity/facebook_marketplace/generic_records.json")
    source = ACTIVITY_FIXTURES / source_relative
    destination = layout.root / source_relative
    destination.parent.mkdir(parents=True)
    shutil.copy2(source, destination)
    layout.ensure_runtime_directories()
    run_migrations(layout.database)
    engine = create_catalog_engine(layout.database)
    scan_archive(layout, engine, hash_files=True)
    import_activities(layout, engine)
    yield layout, engine, destination
    engine.dispose()


def _add_star(engine: Engine, target_type: str, target_id: str) -> Star:
    with Session(engine) as session, session.begin():
        row = Star(
            id=str(uuid.uuid4()),
            target_type=target_type,
            target_id=target_id,
        )
        session.add(row)
        session.flush()
        session.refresh(row)
        session.expunge(row)
        return row


def _quarantine(
    layout: ArchiveLayout,
    engine: Engine,
    target_type: str,
    target_id: str,
) -> tuple[str, str]:
    preview = preview_deletion(
        layout,
        engine,
        DeletionSelector(
            entity_types=(target_type,),  # type: ignore[arg-type]
            entity_ids={target_type: (target_id,)},
        ),
    )
    commit_deletion(
        layout,
        engine,
        preview.operation_id,
        commit_token=preview.commit_token,
        catalog_revision=preview.catalog_revision,
    )
    return preview.operation_id, preview.commit_token


def _purge(
    layout: ArchiveLayout,
    engine: Engine,
    operation_id: str,
    *,
    scope: str,
) -> None:
    plan = plan_hard_purge(
        layout,
        engine,
        operation_id,
        scope=scope,  # type: ignore[arg-type]
    )
    assert plan.executable, plan.blockers
    execute_hard_purge(
        layout,
        engine,
        operation_id,
        scope=scope,  # type: ignore[arg-type]
        purge_token=plan.purge_token,
        purge_revision=plan.purge_revision,
        confirmation=operation_id,
    )


def test_reversible_deletion_hides_and_restores_the_same_star(
    star_lifecycle_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = star_lifecycle_archive
    with Session(engine) as session:
        message_id = session.scalar(select(Message.id).order_by(Message.id))
        assert message_id is not None
    star = _add_star(engine, "message", message_id)

    operation_id, _commit_token = _quarantine(layout, engine, "message", message_id)

    with Session(engine) as session:
        preserved = session.get(Star, star.id)
        assert preserved is not None
        assert preserved.starred_at == star.starred_at
        assert (
            resolve_star_targets(
                session,
                [("message", message_id)],
                include_payload=False,
            )[("message", message_id)].state
            == "latent"
        )

    restore_deletion(
        layout,
        engine,
        operation_id,
        confirmation=operation_id,
    )

    with Session(engine) as session:
        restored = session.get(Star, star.id)
        assert restored is not None
        assert restored.starred_at == star.starred_at
        assert (
            resolve_star_targets(
                session,
                [("message", message_id)],
                include_payload=False,
            )[("message", message_id)].state
            == "active"
        )


def test_hard_purge_revision_is_invalidated_by_a_new_affected_star(
    star_lifecycle_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = star_lifecycle_archive
    with Session(engine) as session:
        message_id = session.scalar(select(Message.id).order_by(Message.id))
        assert message_id is not None
    operation_id, _commit_token = _quarantine(layout, engine, "message", message_id)
    plan = plan_hard_purge(layout, engine, operation_id, scope="normalized")
    assert plan.executable

    star = _add_star(engine, "message", message_id)

    with pytest.raises(StaleDeletionPreviewError, match="changed after"):
        execute_hard_purge(
            layout,
            engine,
            operation_id,
            scope="normalized",
            purge_token=plan.purge_token,
            purge_revision=plan.purge_revision,
            confirmation=operation_id,
        )
    with Session(engine) as session:
        assert session.get(Star, star.id) is not None


def test_delayed_workspace_upgrade_preserves_then_removes_blob_star(
    star_lifecycle_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = star_lifecycle_archive
    with Session(engine) as session:
        unique_blob = (
            select(
                AttachmentOccurrence.blob_sha256.label("blob_sha256"),
                func.min(AttachmentOccurrence.id).label("attachment_id"),
            )
            .where(AttachmentOccurrence.blob_sha256.is_not(None))
            .group_by(AttachmentOccurrence.blob_sha256)
            .having(func.count(AttachmentOccurrence.id) == 1)
            .order_by(AttachmentOccurrence.blob_sha256)
            .limit(1)
            .subquery()
        )
        row = session.execute(select(unique_blob.c.blob_sha256, unique_blob.c.attachment_id)).one()
        blob_id, attachment_id = str(row.blob_sha256), str(row.attachment_id)
    attachment_star = _add_star(engine, "attachment", attachment_id)
    blob_star = _add_star(engine, "blob", blob_id)

    operation_id, _commit_token = _quarantine(
        layout,
        engine,
        "attachment",
        attachment_id,
    )
    _purge(layout, engine, operation_id, scope="normalized")

    with Session(engine) as session:
        assert session.get(Star, attachment_star.id) is None
        assert session.get(Star, blob_star.id) is not None
        assert session.get(Blob, blob_id) is not None

    _purge(layout, engine, operation_id, scope="workspace")

    with Session(engine) as session:
        assert session.get(Star, blob_star.id) is None
        assert session.get(Blob, blob_id) is None


def test_direct_workspace_purge_removes_occurrence_and_blob_stars_together(
    star_lifecycle_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = star_lifecycle_archive
    with Session(engine) as session:
        rows = session.execute(
            select(
                AttachmentOccurrence.blob_sha256,
                func.min(AttachmentOccurrence.id),
            )
            .where(AttachmentOccurrence.blob_sha256.is_not(None))
            .group_by(AttachmentOccurrence.blob_sha256)
            .having(func.count(AttachmentOccurrence.id) == 1)
            .order_by(AttachmentOccurrence.blob_sha256)
        ).first()
        assert rows is not None and rows[0] is not None
        blob_id, attachment_id = str(rows[0]), str(rows[1])
    attachment_star = _add_star(engine, "attachment", attachment_id)
    blob_star = _add_star(engine, "blob", blob_id)

    operation_id, _commit_token = _quarantine(
        layout,
        engine,
        "attachment",
        attachment_id,
    )
    _purge(layout, engine, operation_id, scope="workspace")

    with Session(engine) as session:
        assert session.get(Star, attachment_star.id) is None
        assert session.get(Star, blob_star.id) is None
        assert session.get(Blob, blob_id) is None


def test_final_conversation_removal_purges_its_messenger_contact_star(
    star_lifecycle_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = star_lifecycle_archive
    with Session(engine) as session:
        row = session.execute(
            select(
                Participant.name_search,
                func.min(Conversation.id),
            )
            .join(SegmentParticipant, SegmentParticipant.participant_id == Participant.id)
            .join(ConversationSegment, ConversationSegment.id == SegmentParticipant.segment_id)
            .join(Conversation, Conversation.id == ConversationSegment.conversation_id)
            .where(Participant.is_owner.is_(False))
            .group_by(Participant.name_search)
            .having(func.count(func.distinct(Conversation.id)) == 1)
            .order_by(Participant.name_search)
            .limit(1)
        ).one()
        name_search, conversation_id = str(row[0]), str(row[1])
    contact_star = _add_star(
        engine,
        "messenger_contact",
        stable_id("messenger-contact", name_search),
    )

    operation_id, _commit_token = _quarantine(
        layout,
        engine,
        "conversation",
        conversation_id,
    )
    _purge(layout, engine, operation_id, scope="normalized")

    with Session(engine) as session:
        assert session.get(Star, contact_star.id) is None


def test_reverting_split_removes_only_the_synthetic_conversation_star(
    star_lifecycle_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = star_lifecycle_archive
    with Session(engine) as session:
        conversation = session.scalar(
            select(Conversation)
            .join(ConversationSegment, ConversationSegment.conversation_id == Conversation.id)
            .where(Conversation.conversation_type == "direct")
            .group_by(Conversation.id)
            .having(func.count(ConversationSegment.id) >= 2)
            .order_by(Conversation.id)
        )
        assert conversation is not None
        segment_id = session.scalar(
            select(ConversationSegment.id)
            .where(ConversationSegment.conversation_id == conversation.id)
            .order_by(ConversationSegment.id)
        )
        assert segment_id is not None
        revision = conversation_identity_revision(session)
        source_id = conversation.id
    source_star = _add_star(engine, "conversation", source_id)
    split = split_direct_history(
        layout,
        engine,
        conversation_id=source_id,
        segment_ids=[segment_id],
        title="Synthetic starred split",
        expected_revision=revision,
    )
    synthetic_star = _add_star(
        engine,
        "conversation",
        split.redirect_conversation_id,
    )

    reverted = revert_conversation_correction(
        layout,
        engine,
        split.correction_id,
        expected_revision=split.identity_revision,
    )

    assert reverted.effect["stars_removed"] == 1
    with Session(engine) as session:
        assert session.get(Star, synthetic_star.id) is None
        assert session.get(Star, source_star.id) is not None


def test_merge_star_remains_latent_and_returns_on_revert(
    star_lifecycle_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = star_lifecycle_archive
    with Session(engine) as session:
        direct_ids = list(
            session.scalars(
                select(Conversation.id)
                .where(Conversation.conversation_type == "direct")
                .order_by(Conversation.id)
                .limit(2)
            )
        )
        assert len(direct_ids) == 2
        revision = conversation_identity_revision(session)
    source_id, target_id = direct_ids
    star = _add_star(engine, "conversation", source_id)

    merged = merge_direct_histories(
        layout,
        engine,
        conversation_ids=direct_ids,
        target_conversation_id=target_id,
        expected_revision=revision,
    )

    with Session(engine) as session:
        assert session.get(Star, star.id) is not None
        assert (
            resolve_star_targets(
                session,
                [("conversation", source_id)],
                include_payload=False,
            )[("conversation", source_id)].state
            == "latent"
        )

    revert_conversation_correction(
        layout,
        engine,
        merged.correction_id,
        expected_revision=merged.identity_revision,
    )

    with Session(engine) as session:
        assert session.get(Star, star.id) is not None
        assert (
            resolve_star_targets(
                session,
                [("conversation", source_id)],
                include_payload=False,
            )[("conversation", source_id)].state
            == "active"
        )


def test_successful_reimport_preserves_stable_star_and_prunes_removed_exact_target(
    star_lifecycle_archive: tuple[ArchiveLayout, Engine],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, engine = star_lifecycle_archive
    with Session(engine) as session:
        preserved = session.scalar(
            select(Message).order_by(Message.source_file_id, Message.source_array_index)
        )
        assert preserved is not None
        removed = session.scalar(
            select(Message)
            .where(Message.source_file_id != preserved.source_file_id)
            .order_by(Message.source_file_id, Message.source_array_index)
        )
        assert removed is not None
        source_path = session.execute(
            select(SourceChunk.original_path, SourceFile.relative_path)
            .join(SourceFile, SourceFile.chunk_id == SourceChunk.id)
            .where(SourceFile.id == removed.source_file_id)
        ).one()
        preserved_id = preserved.id
        removed_id = removed.id
    preserved_star = _add_star(engine, "message", preserved_id)
    removed_star = _add_star(engine, "message", removed_id)

    physical_source = layout.root / str(source_path[0]) / str(source_path[1])
    physical_source.write_text(
        '{"meta_archive_purged":{"version":1}}\n',
        encoding="utf-8",
    )
    original_reconcile = message_importer_service._reconcile_duplicate_messages

    def interrupt_after_file_transactions(_session: Session) -> int:
        raise RuntimeError("synthetic import interruption")

    monkeypatch.setattr(
        message_importer_service,
        "_reconcile_duplicate_messages",
        interrupt_after_file_transactions,
    )
    with pytest.raises(RuntimeError, match="synthetic import interruption"):
        import_messages(layout, engine, force=True)

    with Session(engine) as session:
        assert session.get(Star, removed_star.id) is not None
        assert session.scalar(select(Message.id).where(Message.id == removed_id)) is None

    monkeypatch.setattr(
        message_importer_service,
        "_reconcile_duplicate_messages",
        original_reconcile,
    )
    import_messages(layout, engine, force=True)

    with Session(engine) as session:
        assert session.get(Star, preserved_star.id) is not None
        assert session.scalar(select(Message.id).where(Message.id == preserved_id)) == preserved_id
        assert session.get(Star, removed_star.id) is None
        assert session.scalar(select(Message.id).where(Message.id == removed_id)) is None


def test_successful_activity_reimport_prunes_only_removed_owned_target(
    star_activity_archive: tuple[ArchiveLayout, Engine, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, engine, source = star_activity_archive
    with Session(engine) as session:
        activity_ids = list(
            session.scalars(select(ActivityRecord.id).order_by(ActivityRecord.json_pointer))
        )
        assert len(activity_ids) >= 2
    preserved_star = _add_star(engine, "activity", activity_ids[0])
    removed_star = _add_star(engine, "activity", activity_ids[1])

    payload = json.loads(source.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    source.write_text(json.dumps(payload[:1]), encoding="utf-8")
    original_refresh = activity_importer_service._refresh_coverage

    def interrupt_after_file_transactions(_session: Session) -> None:
        raise RuntimeError("synthetic activity import interruption")

    monkeypatch.setattr(
        activity_importer_service,
        "_refresh_coverage",
        interrupt_after_file_transactions,
    )
    with pytest.raises(RuntimeError, match="synthetic activity import interruption"):
        import_activities(layout, engine, force=True)
    with Session(engine) as session:
        assert session.get(Star, removed_star.id) is not None

    monkeypatch.setattr(
        activity_importer_service,
        "_refresh_coverage",
        original_refresh,
    )
    import_activities(layout, engine, force=True)

    with Session(engine) as session:
        assert session.get(Star, preserved_star.id) is not None
        assert session.get(Star, removed_star.id) is None
