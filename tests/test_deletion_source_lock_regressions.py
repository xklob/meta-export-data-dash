from __future__ import annotations

import hashlib
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

from meta_archive.database import create_catalog_engine, run_migrations
from meta_archive.models import (
    AttachmentOccurrence,
    AttachmentTextExtraction,
    Blob,
    DeletionAudit,
    DeletionOperation,
    Message,
    NlpAnnotation,
    SourceFile,
)
from meta_archive.paths import ArchiveLayout
from meta_archive.services.consolidation import build_blob_store
from meta_archive.services.deletion import (
    DeletionError,
    DeletionSelector,
    HardPurgeBlockedError,
    commit_deletion,
    execute_hard_purge,
    plan_hard_purge,
    preview_deletion,
    restore_deletion,
)
from meta_archive.services.inventory import scan_archive
from meta_archive.services.message_importer import import_messages
from meta_archive.services.source_import_lock import (
    SourceImportPublicationBlockedError,
    source_import_publication_lock,
)

MESSAGE_FIXTURES = Path(__file__).parent / "fixtures" / "messages"


@pytest.fixture
def deletion_archive(tmp_path: Path) -> Iterator[tuple[ArchiveLayout, Engine]]:
    layout = ArchiveLayout(tmp_path / "meta-backup-source-lock-regressions")
    shutil.copytree(MESSAGE_FIXTURES, layout.root)
    layout.ensure_runtime_directories()
    run_migrations(layout.database)
    engine = create_catalog_engine(layout.database)
    scan_archive(layout, engine, hash_files=True)
    import_messages(layout, engine)
    build_blob_store(layout, engine)
    yield layout, engine
    engine.dispose()


def _message(engine: Engine, content: str = "Older chunk") -> Message:
    with Session(engine) as session:
        message = session.scalar(select(Message).where(Message.content_display == content))
        assert message is not None
        session.expunge(message)
        return message


def _commit_message_deletion(layout: ArchiveLayout, engine: Engine, message: Message) -> str:
    preview = preview_deletion(
        layout,
        engine,
        DeletionSelector(
            entity_types=("message",),
            entity_ids={"message": (message.id,)},
        ),
    )
    commit_deletion(
        layout,
        engine,
        preview.operation_id,
        commit_token=preview.commit_token,
        catalog_revision=preview.catalog_revision,
    )
    return preview.operation_id


def _add_synthetic_attachment(
    layout: ArchiveLayout, engine: Engine, *, marker: str
) -> tuple[str, str, Path]:
    payload = f"source-lock-regression-{marker}".encode()
    blob_sha256 = hashlib.sha256(payload).hexdigest()
    blob_path = layout.blob_path(blob_sha256)
    blob_path.parent.mkdir(parents=True, exist_ok=True)
    blob_path.write_bytes(payload)
    blob_path.chmod(0o444)
    attachment_id = hashlib.sha256(f"attachment-{marker}".encode()).hexdigest()
    with Session(engine) as session, session.begin():
        session.add(
            Blob(
                sha256=blob_sha256,
                size_bytes=len(payload),
                media_type="application/octet-stream",
                extension=".bin",
                relative_path=blob_path.relative_to(layout.root).as_posix(),
                state="materialized",
            )
        )
        session.flush()
        session.add(
            AttachmentOccurrence(
                id=attachment_id,
                message_id=None,
                activity_record_id=None,
                conversation_id=None,
                source_file_id=None,
                resolved_source_file_id=None,
                blob_sha256=blob_sha256,
                kind="files",
                ordinal=0,
                source_uri=None,
                filename=f"{marker}.bin",
                media_type="application/octet-stream",
                size_bytes=len(payload),
                timestamp_ms=1_700_000_000_000,
                availability="unlinked",
                metadata_json={"synthetic": True},
            )
        )
    return attachment_id, blob_sha256, blob_path


def _add_source_file_for_blob(
    engine: Engine,
    *,
    blob_sha256: str,
    size_bytes: int,
    marker: str,
) -> int:
    with Session(engine) as session, session.begin():
        template = session.scalar(select(SourceFile).order_by(SourceFile.id).limit(1))
        assert template is not None
        source_file = SourceFile(
            chunk_id=template.chunk_id,
            relative_path=f"synthetic/{marker}.bin",
            normalized_path=f"synthetic/{marker}.bin",
            category="synthetic-regression",
            size_bytes=size_bytes,
            mtime_ns=0,
            mode=0o644,
            media_type="application/octet-stream",
            extension=".bin",
            sha256=blob_sha256,
            json_status="not_json",
            json_error=None,
            scan_status="hashed",
            imported_parser_version=None,
            imported_sha256=None,
        )
        session.add(source_file)
        session.flush()
        return source_file.id


def _add_blob_derived_annotation(engine: Engine, blob_sha256: str) -> str:
    extraction_id = "source-lock-regression-extraction"
    annotation_id = "source-lock-regression-annotation"
    with Session(engine) as session, session.begin():
        session.add(
            AttachmentTextExtraction(
                id=extraction_id,
                blob_sha256=blob_sha256,
                pipeline_version="source-lock-regression-v1",
                status="completed",
                extractor_name="fixture",
                extractor_revision="fixture-v1",
                filename="fixture.bin",
                media_type="application/octet-stream",
                text_display="Shared derived fixture",
                text_search="shared derived fixture",
                character_count=22,
                truncated=False,
                index_eligible=True,
                sandboxed=True,
                sandbox_status="completed",
                attempt_count=1,
                limits_json={},
                provenance_json={},
                warnings_json={},
            )
        )
        session.flush()
        session.add(
            NlpAnnotation(
                id=annotation_id,
                target_type="attachment_text",
                target_id=extraction_id,
                task="sentiment",
                pipeline_version="source-lock-regression-v1",
                model_name="fixture-model",
                model_revision="fixture-v1",
                input_sha256="6" * 64,
                status="completed",
                active=True,
                language="en",
                predicted_label="neutral",
                confidence=0.9,
                probabilities_json={"labels": {"neutral": 0.9}},
                attribution_json={},
                warnings_json={},
                provenance_json={},
            )
        )
    return annotation_id


def test_force_reimport_is_blocked_while_committed_quarantine_remains(
    deletion_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = deletion_archive
    message = _message(engine)
    operation_id = _commit_message_deletion(layout, engine, message)

    with Session(engine) as session:
        quarantined = session.get(Message, message.pk)
        assert quarantined is not None and quarantined.deleted_at is not None
        deleted_at = quarantined.deleted_at

    with pytest.raises(SourceImportPublicationBlockedError) as error:
        import_messages(layout, engine, force=True)

    assert error.value.operation_id == operation_id
    assert error.value.status == "committed"
    with Session(engine) as session:
        still_quarantined = session.get(Message, message.pk)
        assert still_quarantined is not None
        assert still_quarantined.id == message.id
        assert still_quarantined.deleted_at == deleted_at
        operation = session.get(DeletionOperation, operation_id)
        assert operation is not None and operation.status == "committed"


def test_source_publication_lock_contention_blocks_plan_without_mutation_then_retries(
    deletion_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = deletion_archive
    message = _message(engine)
    operation_id = _commit_message_deletion(layout, engine, message)
    with Session(engine) as session:
        operation = session.get(DeletionOperation, operation_id)
        assert operation is not None
        assert operation.purge_token_hash is None
        assert operation.purge_revision is None
        audit_count = int(
            session.scalar(
                select(func.count(DeletionAudit.id)).where(
                    DeletionAudit.operation_id == operation_id
                )
            )
            or 0
        )

    with (
        source_import_publication_lock(layout),
        pytest.raises(HardPurgeBlockedError, match="source import"),
    ):
        plan_hard_purge(layout, engine, operation_id, scope="workspace")

    with Session(engine) as session:
        operation = session.get(DeletionOperation, operation_id)
        assert operation is not None and operation.status == "committed"
        assert operation.purge_token_hash is None
        assert operation.purge_revision is None
        assert (
            int(
                session.scalar(
                    select(func.count(DeletionAudit.id)).where(
                        DeletionAudit.operation_id == operation_id
                    )
                )
                or 0
            )
            == audit_count
        )
        quarantined = session.get(Message, message.pk)
        assert quarantined is not None and quarantined.deleted_at is not None

    retry = plan_hard_purge(layout, engine, operation_id, scope="workspace")
    assert retry.executable, retry.blockers


def test_shared_resolved_source_file_is_preserved_by_selected_workspace_purge(
    deletion_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = deletion_archive
    selected_id, blob_sha256, blob_path = _add_synthetic_attachment(
        layout, engine, marker="shared-resolved-source"
    )
    source_file_id = _add_source_file_for_blob(
        engine,
        blob_sha256=blob_sha256,
        size_bytes=blob_path.stat().st_size,
        marker="shared-resolved-source",
    )
    retained_id = "a" * 64
    with Session(engine) as session, session.begin():
        selected = session.get(AttachmentOccurrence, selected_id)
        assert selected is not None
        selected.resolved_source_file_id = source_file_id
        session.add(
            AttachmentOccurrence(
                id=retained_id,
                message_id=None,
                activity_record_id=None,
                conversation_id=None,
                source_file_id=None,
                resolved_source_file_id=source_file_id,
                # The retained occurrence protects the resolved source even when its own blob
                # resolution has not been published yet.
                blob_sha256=None,
                kind="files",
                ordinal=1,
                source_uri="messages/media/shared-resolved-source.bin",
                filename="retained-shared-resolved-source.bin",
                media_type="application/octet-stream",
                size_bytes=None,
                timestamp_ms=1_700_000_000_001,
                availability="missing",
                metadata_json={"synthetic": True},
            )
        )
    preview = preview_deletion(
        layout,
        engine,
        DeletionSelector(
            entity_types=("attachment",),
            entity_ids={"attachment": (selected_id,)},
        ),
    )
    commit_deletion(
        layout,
        engine,
        preview.operation_id,
        commit_token=preview.commit_token,
        catalog_revision=preview.catalog_revision,
    )
    purge = plan_hard_purge(layout, engine, preview.operation_id, scope="workspace")

    assert purge.executable, purge.blockers
    assert purge.impact["shared_source_files_preserved"] == 1
    assert purge.impact["candidate_blobs"] == 0
    execute_hard_purge(
        layout,
        engine,
        preview.operation_id,
        scope="workspace",
        purge_token=purge.purge_token,
        purge_revision=purge.purge_revision,
        confirmation=preview.operation_id,
    )

    with Session(engine) as session:
        assert session.get(AttachmentOccurrence, selected_id) is None
        retained = session.get(AttachmentOccurrence, retained_id)
        assert retained is not None and retained.deleted_at is None
        source_file = session.get(SourceFile, source_file_id)
        assert source_file is not None and source_file.sha256 == blob_sha256
        assert session.get(Blob, blob_sha256) is not None
    assert blob_path.is_file()


def test_unlinked_source_file_reference_blocks_selected_blob_workspace_purge(
    deletion_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = deletion_archive
    attachment_id, blob_sha256, blob_path = _add_synthetic_attachment(
        layout, engine, marker="outside-source-reference"
    )
    source_file_id = _add_source_file_for_blob(
        engine,
        blob_sha256=blob_sha256,
        size_bytes=blob_path.stat().st_size,
        marker="outside-source-reference",
    )
    preview = preview_deletion(
        layout,
        engine,
        DeletionSelector(
            entity_types=("attachment",),
            entity_ids={"attachment": (attachment_id,)},
        ),
    )
    commit_deletion(
        layout,
        engine,
        preview.operation_id,
        commit_token=preview.commit_token,
        catalog_revision=preview.catalog_revision,
    )

    purge = plan_hard_purge(layout, engine, preview.operation_id, scope="workspace")

    assert not purge.executable
    assert "outside_source_file_references_require_review" in purge.blockers
    assert purge.impact["outside_source_file_references"] == 1
    with Session(engine) as session:
        selected = session.get(AttachmentOccurrence, attachment_id)
        assert selected is not None and selected.deleted_at is not None
        assert session.get(SourceFile, source_file_id) is not None
        assert session.get(Blob, blob_sha256) is not None
    assert blob_path.is_file()


def test_shared_restore_is_rejected_out_of_order_then_succeeds_after_later_purge(
    deletion_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = deletion_archive
    first_id, blob_sha256, blob_path = _add_synthetic_attachment(
        layout, engine, marker="restore-order"
    )
    second_id = "b" * 64
    with Session(engine) as session, session.begin():
        first = session.get(AttachmentOccurrence, first_id)
        assert first is not None
        session.add(
            AttachmentOccurrence(
                id=second_id,
                message_id=None,
                activity_record_id=None,
                conversation_id=None,
                source_file_id=None,
                resolved_source_file_id=None,
                blob_sha256=blob_sha256,
                kind=first.kind,
                ordinal=1,
                source_uri=None,
                filename="restore-order-copy.bin",
                media_type=first.media_type,
                size_bytes=first.size_bytes,
                timestamp_ms=first.timestamp_ms,
                availability="unlinked",
                metadata_json={"synthetic": True},
            )
        )

    annotation_id = _add_blob_derived_annotation(engine, blob_sha256)

    first_preview = preview_deletion(
        layout,
        engine,
        DeletionSelector(
            entity_types=("attachment",),
            entity_ids={"attachment": (first_id,)},
        ),
    )
    commit_deletion(
        layout,
        engine,
        first_preview.operation_id,
        commit_token=first_preview.commit_token,
        catalog_revision=first_preview.catalog_revision,
    )
    second_preview = preview_deletion(
        layout,
        engine,
        DeletionSelector(
            entity_types=("attachment",),
            entity_ids={"attachment": (second_id,)},
        ),
    )
    commit_deletion(
        layout,
        engine,
        second_preview.operation_id,
        commit_token=second_preview.commit_token,
        catalog_revision=second_preview.catalog_revision,
    )

    with pytest.raises(
        DeletionError,
        match="later shared-data quarantine must be restored or hard-purged first",
    ):
        restore_deletion(
            layout,
            engine,
            first_preview.operation_id,
            confirmation=first_preview.operation_id,
        )

    with Session(engine) as session:
        first_operation = session.get(DeletionOperation, first_preview.operation_id)
        second_operation = session.get(DeletionOperation, second_preview.operation_id)
        first = session.get(AttachmentOccurrence, first_id)
        second = session.get(AttachmentOccurrence, second_id)
        assert first_operation is not None and first_operation.status == "committed"
        assert second_operation is not None and second_operation.status == "committed"
        assert first is not None and first.deleted_at is not None
        assert second is not None and second.deleted_at is not None
        annotation = session.get(NlpAnnotation, annotation_id)
        assert annotation is not None and annotation.active is False

    second_purge = plan_hard_purge(
        layout,
        engine,
        second_preview.operation_id,
        scope="workspace",
    )
    assert second_purge.executable, second_purge.blockers
    execute_hard_purge(
        layout,
        engine,
        second_preview.operation_id,
        scope="workspace",
        purge_token=second_purge.purge_token,
        purge_revision=second_purge.purge_revision,
        confirmation=second_preview.operation_id,
    )
    assert blob_path.is_file()

    restore_deletion(
        layout,
        engine,
        first_preview.operation_id,
        confirmation=first_preview.operation_id,
    )
    with Session(engine) as session:
        first_operation = session.get(DeletionOperation, first_preview.operation_id)
        second_operation = session.get(DeletionOperation, second_preview.operation_id)
        first = session.get(AttachmentOccurrence, first_id)
        assert first_operation is not None and first_operation.status == "restored"
        assert second_operation is not None and second_operation.status == "purged"
        assert first is not None and first.deleted_at is None
        assert session.get(AttachmentOccurrence, second_id) is None
        assert session.get(Blob, blob_sha256) is not None
    assert blob_path.is_file()
