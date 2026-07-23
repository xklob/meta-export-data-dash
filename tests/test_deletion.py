from __future__ import annotations

import fcntl
import hashlib
import json
import multiprocessing
import os
import shutil
import stat
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import lancedb  # type: ignore[import-untyped]
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, func, select, text, update
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from meta_archive.api.main import create_app
from meta_archive.cli import app as cli_app
from meta_archive.database import create_catalog_engine, run_migrations
from meta_archive.models import (
    ActivityRecord,
    AttachmentOccurrence,
    AttachmentTextExtraction,
    AudioTranscription,
    AudioTranscriptSegment,
    Blob,
    Conversation,
    ConversationCorrection,
    ConversationSegment,
    ConversationSegmentOverride,
    DeletionAudit,
    DeletionOperation,
    Job,
    MediaDerivative,
    Message,
    NlpAnnotation,
    NlpHourlyRollup,
    SearchWindow,
    SearchWindowMessage,
    SemanticIndexConfig,
    SemanticWindowEmbedding,
    SourceFile,
    SourceRecord,
)
from meta_archive.paths import ArchiveLayout
from meta_archive.services import deletion as deletion_service
from meta_archive.services import topic_clusters as topic_cluster_service
from meta_archive.services.activity_importer import import_activities
from meta_archive.services.analytics_cache import read_analytics_cache, write_analytics_cache
from meta_archive.services.consolidation import build_blob_store, reconstruct_source
from meta_archive.services.deletion import (
    DeletionError,
    DeletionSelector,
    HardPurgeBlockedError,
    InvalidDeletionTokenError,
    StaleDeletionPreviewError,
    UnsafeDeletionError,
    _tree_revision_entries,
    commit_deletion,
    execute_hard_purge,
    plan_hard_purge,
    preview_deletion,
    recover_pending_hard_purges,
    restore_deletion,
)
from meta_archive.services.identifiers import canonical_fingerprint
from meta_archive.services.inventory import scan_archive
from meta_archive.services.library import build_filesystem_library
from meta_archive.services.media_derivatives import write_status_manifest
from meta_archive.services.message_importer import import_messages
from meta_archive.services.verification import source_manifest_artifacts, verify_catalog

MESSAGE_FIXTURES = Path(__file__).parent / "fixtures" / "messages"
ACTIVITY_FIXTURES = Path(__file__).parent / "fixtures" / "activity"


@pytest.fixture
def deletion_archive(tmp_path: Path) -> Iterator[tuple[ArchiveLayout, Engine]]:
    layout = ArchiveLayout(tmp_path / "meta-backup-deletion")
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


def _message(engine: Engine, content: str) -> Message:
    with Session(engine) as session:
        result = session.scalar(select(Message).where(Message.content_display == content))
        assert result is not None
        session.expunge(result)
        return result


def _mutation_and_exit_on_manifest_boundary(
    archive_root: str,
    action: str,
    publication_phase: str,
    operation_id: str,
    commit_token: str,
    catalog_revision: str,
    event_prefix: str,
    occurrence: int,
) -> None:
    """Run one deletion mutation in a child and terminate at a publication I/O boundary."""

    child_layout = ArchiveLayout(Path(archive_root))
    child_engine = create_catalog_engine(child_layout.database)
    original_publish = deletion_service.publish_private_file

    def crash_selected_publication(*args: object, **kwargs: object) -> object:
        assert len(args) >= 3 and isinstance(args[2], bytes)
        payload = json.loads(args[2].decode("utf-8"))
        publication = payload.get("publication") if isinstance(payload, dict) else None
        if not isinstance(publication, dict) or publication.get("phase") != publication_phase:
            return original_publish(*args, **kwargs)  # type: ignore[arg-type]
        seen = 0

        def terminate(event: str) -> None:
            nonlocal seen
            if event.startswith(event_prefix):
                seen += 1
                if seen == occurrence:
                    os._exit(91)

        kwargs["fault_injector"] = terminate
        return original_publish(*args, **kwargs)  # type: ignore[arg-type]

    deletion_service.publish_private_file = crash_selected_publication  # type: ignore[assignment]
    if action == "commit":
        commit_deletion(
            child_layout,
            child_engine,
            operation_id,
            commit_token=commit_token,
            catalog_revision=catalog_revision,
        )
    else:
        restore_deletion(
            child_layout,
            child_engine,
            operation_id,
            confirmation=operation_id,
        )
    child_engine.dispose()


def test_purge_worker_locks_fail_fast_for_every_derived_publisher(tmp_path: Path) -> None:
    layout = ArchiveLayout(tmp_path / "meta-backup-worker-locks")
    layout.ensure_runtime_directories()
    lock_paths = (
        layout.thumbnails / ".build.lock",
        layout.catalog / "audio-transcription.lock",
        layout.catalog / "attachment-text.lock",
        layout.catalog / "nlp-annotations.lock",
        layout.catalog / "conflict-calibration.lock",
        layout.catalog / "conflict-aggregates.lock",
    )
    for lock_path in lock_paths:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                with (
                    pytest.raises(HardPurgeBlockedError, match="currently running"),
                    deletion_service.purge_worker_locks(layout),
                ):
                    pytest.fail("hard purge acquired a worker lock held by another publisher")
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _fts_count(engine: Engine, message_pk: int, query: str) -> int:
    with engine.connect() as connection:
        return int(
            connection.execute(
                text(
                    "SELECT count(*) FROM message_fts "
                    "WHERE rowid = :message_pk AND message_fts MATCH :query"
                ),
                {"message_pk": message_pk, "query": query},
            ).scalar_one()
        )


def _tree_digest_snapshot(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_workspace_tree_revision_rejects_a_symlinked_parent(tmp_path: Path) -> None:
    layout = ArchiveLayout(tmp_path / "meta-backup-tree-safety")
    layout.root.mkdir()
    external = tmp_path / "external-tree"
    nested = external / "nested"
    nested.mkdir(parents=True)
    (nested / "fixture.json").write_text("{}\n")
    (layout.root / "alias").symlink_to(external, target_is_directory=True)

    with pytest.raises(HardPurgeBlockedError, match="unsafe parent directory"):
        _tree_revision_entries(layout, layout.root / "alias" / "nested")

    assert (nested / "fixture.json").read_text() == "{}\n"


def test_workspace_targets_reject_an_intermediate_symlink_without_touching_target(
    deletion_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = deletion_archive
    prefix = next(
        f"{value:02x}" for value in range(256) if not (layout.blobs / f"{value:02x}").exists()
    )
    sha256 = prefix + "a" * 62
    unrelated = layout.catalog / "unrelated-purge-target"
    unrelated.mkdir()
    redirected_target = unrelated / sha256
    redirected_target.write_text("must survive")
    (layout.blobs / prefix).symlink_to(unrelated, target_is_directory=True)

    with (
        Session(engine) as session,
        pytest.raises(HardPurgeBlockedError, match="not a regular file"),
    ):
        deletion_service._workspace_targets(layout, session, {sha256})

    assert redirected_target.read_text() == "must survive"
    assert (layout.blobs / prefix).is_symlink()


def test_remove_tree_rejects_a_symlink_root_without_touching_sibling(
    deletion_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, _engine = deletion_archive
    staging_root = layout.trash / "purge-staging"
    sibling = staging_root / "other-operation"
    sibling.mkdir(parents=True)
    rollback_journal = sibling / deletion_service.PURGE_ROLLBACK_JOURNAL_NAME
    rollback_journal.write_text("must survive")
    redirected = staging_root / "current-operation"
    redirected.symlink_to(sibling, target_is_directory=True)

    with pytest.raises(HardPurgeBlockedError, match="unsafe purge staging path"):
        deletion_service._remove_tree_exact(redirected, required_parent=staging_root)

    assert rollback_journal.read_text() == "must survive"
    assert sibling.is_dir()
    assert redirected.is_symlink()


def _add_fixture_vector_window(
    layout: ArchiveLayout, engine: Engine, message: Message
) -> tuple[str, str, str]:
    config_id = "9" * 64
    selected_window_id = "8" * 64
    retained_window_id = "7" * 64
    table_name = "deletion_fixture_vectors"
    with Session(engine) as session, session.begin():
        retained_message = session.scalar(
            select(Message).where(Message.id != message.id).order_by(Message.id)
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
                window_version="fixture-window-v1",
                window_count=2,
                audit_json={},
                completed_at=datetime.now(UTC),
            )
        )
        for window_id, linked_message in (
            (selected_window_id, message),
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
                    text_display="Fixture semantic deletion window.",
                    text_search="fixture semantic deletion window",
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


def _add_fixture_shared_vector_window(
    layout: ArchiveLayout,
    engine: Engine,
    first_message: Message,
    second_message: Message,
) -> tuple[str, str, str]:
    assert first_message.id != second_message.id
    assert first_message.conversation_id == second_message.conversation_id
    config_id = "4" * 64
    window_id = "5" * 64
    table_name = "deletion_shared_fixture_vectors"
    ordered_messages = sorted(
        (first_message, second_message),
        key=lambda message: (message.timestamp_ms or 0, message.id),
    )
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
                dimensions=3,
                distance_metric="cosine",
                document_prefix="",
                query_prefix="",
                vector_table=table_name,
                window_version="fixture-window-v1",
                window_count=1,
                audit_json={},
                completed_at=datetime.now(UTC),
            )
        )
        session.add(
            SearchWindow(
                id=window_id,
                window_version="fixture-window-v1",
                entity_type="message_window",
                conversation_id=first_message.conversation_id,
                segment_id=first_message.segment_id,
                first_message_id=ordered_messages[0].id,
                last_message_id=ordered_messages[-1].id,
                first_timestamp_ms=ordered_messages[0].timestamp_ms,
                last_timestamp_ms=ordered_messages[-1].timestamp_ms,
                turn_count=2,
                approx_token_count=8,
                text_display="Fixture shared semantic deletion window.",
                text_search="fixture shared semantic deletion window",
                source_category="fixture",
                participants_search=" ".join(
                    sorted({first_message.sender_search, second_message.sender_search})
                ),
                source_ids_json={},
                build_generation="fixture-shared-generation",
                embedding_config_hash=config_id,
                embedding_updated_at=datetime.now(UTC),
            )
        )
        session.flush()
        session.add_all(
            [
                SearchWindowMessage(window_id=window_id, message_id=message.id, ordinal=ordinal)
                for ordinal, message in enumerate(ordered_messages)
            ]
        )
    lancedb.connect(str(layout.vectors)).create_table(
        table_name,
        data=[{"id": window_id, "vector": [1.0, 0.0, 0.0]}],
        mode="overwrite",
    )
    return table_name, config_id, window_id


def _nlp_annotation(identifier: str, *, target_type: str, target_id: str) -> NlpAnnotation:
    return NlpAnnotation(
        id=identifier,
        target_type=target_type,
        target_id=target_id,
        task="sentiment",
        pipeline_version="deletion-fixture-v1",
        model_name="fixture-model",
        model_revision="fixture-revision",
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


def _add_fixture_derived_nlp(
    layout: ArchiveLayout, engine: Engine, blob_sha256: str
) -> tuple[str, str, set[str], Path]:
    transcription_id = "audio-transcription-deletion-fixture"
    segment_id = "audio-segment-deletion-fixture"
    extraction_id = "attachment-text-deletion-fixture"
    annotation_ids = {"annotation-audio-deletion", "annotation-text-deletion"}
    derivative_payload = b"fixture derivative bytes"
    derivative_path = layout.thumbnails / "media-v1" / "deletion-fixture.webp"
    derivative_path.parent.mkdir(parents=True, exist_ok=True)
    derivative_path.write_bytes(derivative_payload)
    with Session(engine) as session, session.begin():
        session.add(
            AudioTranscription(
                id=transcription_id,
                blob_sha256=blob_sha256,
                pipeline_version="deletion-fixture-v1",
                status="completed",
                model_name="fixture-whisper",
                model_revision="fixture-revision",
                device="cpu",
                compute_type="int8",
                language="en",
                segment_count=1,
                attempt_count=1,
                index_eligible=True,
                provenance_json={},
                warnings_json={},
            )
        )
        session.flush()
        session.add(
            AudioTranscriptSegment(
                id=segment_id,
                transcription_id=transcription_id,
                ordinal=0,
                start_ms=0,
                end_ms=1_000,
                text_display="Fixture transcript",
                text_search="fixture transcript",
                confidence=0.9,
                words_json={},
            )
        )
        session.add(
            AttachmentTextExtraction(
                id=extraction_id,
                blob_sha256=blob_sha256,
                pipeline_version="deletion-fixture-v1",
                status="completed",
                extractor_name="fixture",
                extractor_revision="fixture-revision",
                filename="fixture.bin",
                media_type="application/octet-stream",
                text_display="Fixture extracted text",
                text_search="fixture extracted text",
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
        session.add_all(
            [
                _nlp_annotation(
                    "annotation-audio-deletion",
                    target_type="audio_segment",
                    target_id=segment_id,
                ),
                _nlp_annotation(
                    "annotation-text-deletion",
                    target_type="attachment_text",
                    target_id=extraction_id,
                ),
            ]
        )
        session.add(
            MediaDerivative(
                id="media-derivative-deletion-fixture",
                blob_sha256=blob_sha256,
                kind="image_thumbnail",
                pipeline_version="media-v1",
                status="completed",
                relative_path=derivative_path.relative_to(layout.root).as_posix(),
                media_type="image/webp",
                size_bytes=len(derivative_payload),
                output_sha256=hashlib.sha256(derivative_payload).hexdigest(),
                attempt_count=1,
                metadata_json={},
            )
        )
    write_status_manifest(layout, engine)
    return transcription_id, extraction_id, annotation_ids, derivative_path


def _add_synthetic_attachment(
    layout: ArchiveLayout, engine: Engine, *, marker: str
) -> tuple[str, str, Path]:
    payload = f"stage5-workspace-purge-{marker}".encode()
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


def test_selector_uses_inclusive_display_dates_as_half_open_utc_range() -> None:
    selector = DeletionSelector(
        from_date="2024-03-10",
        to_date="2024-03-10",
        timezone="America/New_York",
    )
    start_ms, end_ms = selector.utc_range_ms()
    assert start_ms == int(datetime(2024, 3, 10, 5, tzinfo=UTC).timestamp() * 1000)
    assert end_ms == int(datetime(2024, 3, 11, 4, tzinfo=UTC).timestamp() * 1000)


def test_selector_refuses_unbounded_and_invalid_date_previews() -> None:
    with pytest.raises(UnsafeDeletionError, match="unbounded"):
        DeletionSelector().validate()
    with pytest.raises(UnsafeDeletionError, match="YYYY-MM-DD"):
        DeletionSelector(from_date="06/01/2024").validate()
    with pytest.raises(UnsafeDeletionError, match="not supported for attachment deletion"):
        DeletionSelector(entity_types=("attachment",), text="voice", text_mode="keyword").validate()


def test_date_selector_includes_both_local_day_boundaries_but_not_the_exclusive_end(
    deletion_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = deletion_archive
    selector = DeletionSelector(
        entity_types=("message",),
        from_date="2024-03-10",
        to_date="2024-03-10",
        timezone="America/New_York",
    )
    start_ms, end_ms = selector.utc_range_ms()
    assert start_ms is not None and end_ms is not None
    with Session(engine) as session, session.begin():
        messages = list(session.scalars(select(Message).order_by(Message.pk).limit(3)))
        assert len(messages) == 3
        messages[0].timestamp_ms = start_ms
        messages[1].timestamp_ms = end_ms - 1
        messages[2].timestamp_ms = end_ms

    preview = preview_deletion(layout, engine, selector)

    assert preview.impact["selected"]["message"] == 2


def test_case_insensitive_exact_selector_uses_unicode_casefold(
    deletion_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = deletion_archive
    preview = preview_deletion(
        layout,
        engine,
        DeletionSelector(
            entity_types=("message",),
            text="FRANÇAIS ABC",
            case_sensitive=False,
            whole_field=True,
        ),
    )
    assert preview.impact["selected"]["message"] == 1


def test_query_builder_supports_literal_keyword_raw_participant_and_media_filters(
    deletion_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = deletion_archive
    with Session(engine) as session, session.begin():
        actor_only = session.scalar(select(ActivityRecord).order_by(ActivityRecord.pk))
        assert actor_only is not None
        actor_only.actor_raw = "ActorOnlySentinel"
        actor_only.actor_display = "ActorOnlySentinel"
        actor_only.actor_search = "actoronlysentinel"
    case_sensitive = preview_deletion(
        layout,
        engine,
        DeletionSelector(entity_types=("message",), text="Older"),
    )
    wrong_case = preview_deletion(
        layout,
        engine,
        DeletionSelector(entity_types=("message",), text="older"),
    )
    keyword = preview_deletion(
        layout,
        engine,
        DeletionSelector(entity_types=("message",), text="Older chunk", text_mode="keyword"),
    )
    raw = preview_deletion(
        layout,
        engine,
        DeletionSelector(entity_types=("message",), text="FranÃ§ais", text_variant="raw"),
    )
    participant = preview_deletion(
        layout,
        engine,
        DeletionSelector(entity_types=("message",), participant="Alex Example"),
    )
    sender_is_not_message_text = preview_deletion(
        layout,
        engine,
        DeletionSelector(entity_types=("message",), text="Alex Example", whole_field=True),
    )
    actor_is_not_activity_text = preview_deletion(
        layout,
        engine,
        DeletionSelector(entity_types=("activity",), text="Archive User", whole_field=True),
    )
    keyword_sender_is_not_message_text = preview_deletion(
        layout,
        engine,
        DeletionSelector(entity_types=("message",), text='"Alex Example"', text_mode="keyword"),
    )
    keyword_actor_is_not_activity_text = preview_deletion(
        layout,
        engine,
        DeletionSelector(entity_types=("activity",), text="actoronlysentinel", text_mode="keyword"),
    )
    keyword_sender_column_override_is_blocked = preview_deletion(
        layout,
        engine,
        DeletionSelector(
            entity_types=("message",),
            text='sender_search:"Alex Example"',
            text_mode="keyword",
        ),
    )
    keyword_actor_column_override_is_blocked = preview_deletion(
        layout,
        engine,
        DeletionSelector(
            entity_types=("activity",),
            text="actor_search:actoronlysentinel",
            text_mode="keyword",
        ),
    )
    activity_participant = preview_deletion(
        layout,
        engine,
        DeletionSelector(entity_types=("activity",), participant="Archive User"),
    )
    audio = preview_deletion(
        layout,
        engine,
        DeletionSelector(entity_types=("attachment",), media_types=("audio",)),
    )
    missing = preview_deletion(
        layout,
        engine,
        DeletionSelector(entity_types=("attachment",), availability=("missing",)),
    )

    assert case_sensitive.impact["selected"]["message"] == 2
    assert wrong_case.impact["selected"]["message"] == 0
    assert keyword.impact["selected"]["message"] == 1
    assert raw.impact["selected"]["message"] == 1
    assert participant.impact["selected"]["message"] > 0
    assert sender_is_not_message_text.impact["selected"]["message"] == 0
    assert actor_is_not_activity_text.impact["selected"]["activity"] == 0
    assert keyword_sender_is_not_message_text.impact["selected"]["message"] == 0
    assert keyword_actor_is_not_activity_text.impact["selected"]["activity"] == 0
    assert keyword_sender_column_override_is_blocked.impact["selected"]["message"] == 0
    assert keyword_actor_column_override_is_blocked.impact["selected"]["activity"] == 0
    assert activity_participant.impact["selected"]["activity"] > 0
    assert audio.impact["selected"]["attachment"] > 0
    assert missing.impact["selected"]["attachment"] > 0


def test_keyword_conversation_selector_uses_message_fts_and_cascades(
    deletion_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = deletion_archive
    matching_message = _message(engine, "Older chunk")
    with Session(engine) as session:
        conversation = session.get(Conversation, matching_message.conversation_id)
        assert conversation is not None
        expected_message_count = conversation.message_count
        expected_attachment_count = conversation.attachment_count

    preview = preview_deletion(
        layout,
        engine,
        DeletionSelector(
            entity_types=("conversation",),
            text="Older chunk",
            text_mode="keyword",
        ),
    )

    assert preview.impact["direct"]["conversation"] == 1
    assert preview.impact["selected"]["conversation"] == 1
    assert preview.impact["selected"]["message"] == expected_message_count
    assert preview.impact["selected"]["attachment"] == expected_attachment_count


def test_exact_conversation_selector_matches_retained_message_text_and_cascades(
    deletion_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = deletion_archive
    matching_message = _message(engine, "Older chunk")
    with Session(engine) as session:
        conversation = session.get(Conversation, matching_message.conversation_id)
        assert conversation is not None
        expected_message_count = conversation.message_count
        expected_attachment_count = conversation.attachment_count

    preview = preview_deletion(
        layout,
        engine,
        DeletionSelector(
            entity_types=("conversation",),
            text="Older chunk",
            whole_field=True,
        ),
    )

    assert preview.impact["direct"]["conversation"] == 1
    assert preview.impact["selected"]["conversation"] == 1
    assert preview.impact["selected"]["message"] == expected_message_count
    assert preview.impact["selected"]["attachment"] == expected_attachment_count


def test_exact_preview_commit_and_restore_are_revision_bound_and_search_safe(
    deletion_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = deletion_archive
    message = _message(engine, "Older chunk")
    with Session(engine) as session:
        conversation = session.get(Conversation, message.conversation_id)
        assert conversation is not None
        original_count = conversation.message_count
    assert _fts_count(engine, message.pk, '"older"') == 1

    preview = preview_deletion(
        layout,
        engine,
        DeletionSelector(
            entity_types=("message",),
            text="Older chunk",
            whole_field=True,
        ),
    )
    assert preview.impact["selected"]["message"] == 1
    assert preview.impact["selected"]["activity"] == 0
    assert stat.S_IMODE(Path(preview.manifest).stat().st_mode) == 0o400
    with pytest.raises(InvalidDeletionTokenError):
        commit_deletion(
            layout,
            engine,
            preview.operation_id,
            commit_token="wrong-token",
            catalog_revision=preview.catalog_revision,
        )

    committed = commit_deletion(
        layout,
        engine,
        preview.operation_id,
        commit_token=preview.commit_token,
        catalog_revision=preview.catalog_revision,
    )
    assert committed.affected["message"] == 1
    assert _fts_count(engine, message.pk, '"older"') == 0
    with Session(engine) as session:
        quarantined = session.scalar(select(Message).where(Message.id == message.id))
        conversation = session.get(Conversation, message.conversation_id)
        assert quarantined is not None and quarantined.deleted_at is not None
        assert conversation is not None and conversation.message_count == original_count - 1

    restored = restore_deletion(
        layout, engine, preview.operation_id, confirmation=preview.operation_id
    )
    assert restored.affected["message"] == 1
    assert _fts_count(engine, message.pk, '"older"') == 1
    with Session(engine) as session:
        restored_message = session.scalar(select(Message).where(Message.id == message.id))
        conversation = session.get(Conversation, message.conversation_id)
        assert restored_message is not None and restored_message.deleted_at is None
        assert conversation is not None and conversation.message_count == original_count


def test_commit_manifest_publication_failure_retries_without_duplicate_mutation(
    deletion_archive: tuple[ArchiveLayout, Engine],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, engine = deletion_archive
    message = _message(engine, "Older chunk")
    preview = preview_deletion(
        layout,
        engine,
        DeletionSelector(entity_types=("message",), text="Older chunk", whole_field=True),
    )
    original_write = deletion_service._write_manifest

    def fail_published_manifest(*args: object, **kwargs: object) -> None:
        if kwargs.get("publication_phase") == "published":
            raise OSError("simulated published-manifest failure")
        original_write(*args, **kwargs)

    monkeypatch.setattr(deletion_service, "_write_manifest", fail_published_manifest)
    with pytest.raises(OSError, match="simulated published-manifest failure"):
        commit_deletion(
            layout,
            engine,
            preview.operation_id,
            commit_token=preview.commit_token,
            catalog_revision=preview.catalog_revision,
        )
    with Session(engine) as session:
        operation = session.get(DeletionOperation, preview.operation_id)
        quarantined = session.get(Message, message.pk)
        assert operation is not None and operation.status == "committed"
        assert quarantined is not None and quarantined.deleted_at is not None
        assert (
            session.scalar(
                select(func.count(DeletionAudit.id)).where(
                    DeletionAudit.operation_id == preview.operation_id,
                    DeletionAudit.action == "commit",
                )
            )
            == 1
        )

    monkeypatch.setattr(deletion_service, "_write_manifest", original_write)
    replay = commit_deletion(
        layout,
        engine,
        preview.operation_id,
        commit_token=preview.commit_token,
        catalog_revision=preview.catalog_revision,
    )
    assert replay.status == "committed"
    payload = json.loads(Path(preview.manifest).read_text(encoding="utf-8"))
    assert payload["publication"] == {"action": "commit", "phase": "published"}
    with Session(engine) as session:
        assert (
            session.scalar(
                select(func.count(DeletionAudit.id)).where(
                    DeletionAudit.operation_id == preview.operation_id,
                    DeletionAudit.action == "commit",
                )
            )
            == 1
        )


def test_restore_manifest_publication_failure_retries_without_duplicate_mutation(
    deletion_archive: tuple[ArchiveLayout, Engine],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, engine = deletion_archive
    message = _message(engine, "Older chunk")
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
    original_write = deletion_service._write_manifest

    def fail_published_manifest(*args: object, **kwargs: object) -> None:
        if kwargs.get("publication_phase") == "published":
            raise OSError("simulated published-restore-manifest failure")
        original_write(*args, **kwargs)

    monkeypatch.setattr(deletion_service, "_write_manifest", fail_published_manifest)
    with pytest.raises(OSError, match="simulated published-restore-manifest failure"):
        restore_deletion(
            layout,
            engine,
            preview.operation_id,
            confirmation=preview.operation_id,
        )
    with Session(engine) as session:
        operation = session.get(DeletionOperation, preview.operation_id)
        restored_message = session.get(Message, message.pk)
        assert operation is not None and operation.status == "restored"
        assert restored_message is not None and restored_message.deleted_at is None
        assert (
            session.scalar(
                select(func.count(DeletionAudit.id)).where(
                    DeletionAudit.operation_id == preview.operation_id,
                    DeletionAudit.action == "restore",
                )
            )
            == 1
        )

    monkeypatch.setattr(deletion_service, "_write_manifest", original_write)
    replay = restore_deletion(
        layout,
        engine,
        preview.operation_id,
        confirmation=preview.operation_id,
    )
    assert replay.status == "restored"
    payload = json.loads(Path(preview.manifest).read_text(encoding="utf-8"))
    assert payload["publication"] == {"action": "restore", "phase": "published"}
    with Session(engine) as session:
        assert (
            session.scalar(
                select(func.count(DeletionAudit.id)).where(
                    DeletionAudit.operation_id == preview.operation_id,
                    DeletionAudit.action == "restore",
                )
            )
            == 1
        )


@pytest.mark.parametrize(
    ("action", "publication_phase", "event_prefix", "occurrence"),
    [
        ("commit", "prepared", "after_rename_exchange:manifest.json", 1),
        ("commit", "published", "after_unlink:.publication-", 1),
        ("restore", "prepared", "after_rename_exchange:manifest.json", 1),
        ("restore", "published", "after_unlink:.publication-", 1),
    ],
)
def test_manifest_publication_process_death_replays_without_duplicate_mutation(
    deletion_archive: tuple[ArchiveLayout, Engine],
    action: str,
    publication_phase: str,
    event_prefix: str,
    occurrence: int,
) -> None:
    layout, engine = deletion_archive
    message = _message(engine, "Older chunk")
    preview = preview_deletion(
        layout,
        engine,
        DeletionSelector(entity_types=("message",), text="Older chunk", whole_field=True),
    )
    if action == "restore":
        commit_deletion(
            layout,
            engine,
            preview.operation_id,
            commit_token=preview.commit_token,
            catalog_revision=preview.catalog_revision,
        )

    context = multiprocessing.get_context("fork")
    process = context.Process(
        target=_mutation_and_exit_on_manifest_boundary,
        args=(
            str(layout.root),
            action,
            publication_phase,
            preview.operation_id,
            preview.commit_token,
            preview.catalog_revision,
            event_prefix,
            occurrence,
        ),
    )
    process.start()
    process.join(timeout=20)
    assert process.exitcode == 91

    manifest_path = Path(preview.manifest)
    interrupted_payload = manifest_path.read_bytes()
    with Session(engine) as session:
        operation = session.get(DeletionOperation, preview.operation_id)
        stored_message = session.get(Message, message.pk)
        assert operation is not None and stored_message is not None
        transaction_committed = publication_phase == "published"
        expected_status = (
            "committed"
            if action == "commit" and transaction_committed
            else "restored"
            if action == "restore" and transaction_committed
            else "previewed"
            if action == "commit"
            else "committed"
        )
        assert operation.status == expected_status
        assert (stored_message.deleted_at is not None) == (expected_status == "committed")
        expected_audits = 1 if transaction_committed else 0
        assert (
            session.scalar(
                select(func.count(DeletionAudit.id)).where(
                    DeletionAudit.operation_id == preview.operation_id,
                    DeletionAudit.action == action,
                )
            )
            == expected_audits
        )

    if action == "commit":
        result = commit_deletion(
            layout,
            engine,
            preview.operation_id,
            commit_token=preview.commit_token,
            catalog_revision=preview.catalog_revision,
        )
        assert result.status == "committed"
    else:
        result = restore_deletion(
            layout,
            engine,
            preview.operation_id,
            confirmation=preview.operation_id,
        )
        assert result.status == "restored"

    final_payload = manifest_path.read_bytes()
    if publication_phase == "prepared":
        assert final_payload != interrupted_payload
    else:
        assert final_payload == interrupted_payload
    decoded = json.loads(final_payload)
    assert decoded["publication"] == {"action": action, "phase": "published"}
    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o400
    assert manifest_path.stat().st_nlink == 1
    assert [path.name for path in manifest_path.parent.iterdir()] == ["manifest.json"]
    with Session(engine) as session:
        operation = session.get(DeletionOperation, preview.operation_id)
        stored_message = session.get(Message, message.pk)
        assert operation is not None and stored_message is not None
        assert operation.purge_plan_json["manifest_publication"]["phase"] == "published"
        assert (stored_message.deleted_at is not None) == (action == "commit")
        assert (
            session.scalar(
                select(func.count(DeletionAudit.id)).where(
                    DeletionAudit.operation_id == preview.operation_id,
                    DeletionAudit.action == action,
                )
            )
            == 1
        )


@pytest.mark.parametrize("publication_phase", ["prepared", "published"])
def test_commit_manifest_rejects_byte_identical_inode_substitution_and_retries(
    deletion_archive: tuple[ArchiveLayout, Engine],
    monkeypatch: pytest.MonkeyPatch,
    publication_phase: str,
) -> None:
    layout, engine = deletion_archive
    message = _message(engine, "Older chunk")
    preview = preview_deletion(
        layout,
        engine,
        DeletionSelector(entity_types=("message",), text="Older chunk", whole_field=True),
    )
    original_publish = deletion_service.publish_private_file
    substituted = False

    def substitute_before_publish(*args: object, **kwargs: object) -> object:
        nonlocal substituted
        assert len(args) >= 3 and isinstance(args[0], Path) and isinstance(args[1], str)
        assert isinstance(args[2], bytes)
        payload = json.loads(args[2].decode("utf-8"))
        publication = payload.get("publication") if isinstance(payload, dict) else None
        if (
            not substituted
            and isinstance(publication, dict)
            and publication.get("phase") == publication_phase
        ):
            substituted = True
            target = args[0] / args[1]
            replacement = target.with_name(".byte-identical-substitution")
            replacement.write_bytes(target.read_bytes())
            replacement.chmod(0o400)
            os.replace(replacement, target)
        return original_publish(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(deletion_service, "publish_private_file", substitute_before_publish)
    with pytest.raises(DeletionError, match="publication failed safely"):
        commit_deletion(
            layout,
            engine,
            preview.operation_id,
            commit_token=preview.commit_token,
            catalog_revision=preview.catalog_revision,
        )
    assert substituted
    with Session(engine) as session:
        operation = session.get(DeletionOperation, preview.operation_id)
        stored_message = session.get(Message, message.pk)
        assert operation is not None and stored_message is not None
        assert operation.status == ("previewed" if publication_phase == "prepared" else "committed")
        assert (stored_message.deleted_at is not None) == (publication_phase == "published")

    monkeypatch.setattr(deletion_service, "publish_private_file", original_publish)
    result = commit_deletion(
        layout,
        engine,
        preview.operation_id,
        commit_token=preview.commit_token,
        catalog_revision=preview.catalog_revision,
    )
    manifest_path = Path(preview.manifest)
    assert result.status == "committed"
    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o400
    assert manifest_path.stat().st_nlink == 1
    assert [path.name for path in manifest_path.parent.iterdir()] == ["manifest.json"]


@pytest.mark.parametrize(
    ("action", "publication_phase"),
    [
        ("commit", "prepared"),
        ("commit", "published"),
        ("restore", "prepared"),
        ("restore", "published"),
    ],
)
def test_conversation_snapshot_tamper_immediately_before_manifest_rewrite_is_rejected(
    deletion_archive: tuple[ArchiveLayout, Engine],
    monkeypatch: pytest.MonkeyPatch,
    action: str,
    publication_phase: str,
) -> None:
    layout, engine = deletion_archive
    preview = preview_deletion(
        layout,
        engine,
        DeletionSelector(entity_types=("conversation",), text="Older chunk", whole_field=True),
    )
    if action == "restore":
        commit_deletion(
            layout,
            engine,
            preview.operation_id,
            commit_token=preview.commit_token,
            catalog_revision=preview.catalog_revision,
        )
    original_write = deletion_service._write_manifest
    retained_bytes: bytes | None = None

    def tamper_before_rewrite(*args: object, **kwargs: object) -> None:
        nonlocal retained_bytes
        if retained_bytes is None and kwargs.get("publication_phase") == publication_phase:
            assert len(args) >= 2 and isinstance(args[0], ArchiveLayout)
            assert isinstance(args[1], str)
            target = args[0].root / args[1]
            retained_bytes = target.read_bytes()
            payload = json.loads(retained_bytes)
            conversation_item = next(
                item for item in payload["items"] if item["entity_type"] == "conversation"
            )
            conversation_item["snapshot"]["_tampered"] = True
            target.chmod(0o600)
            target.write_text(json.dumps(payload), encoding="utf-8")
            target.chmod(0o400)
        original_write(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(deletion_service, "_write_manifest", tamper_before_rewrite)
    with pytest.raises(DeletionError, match="conversation snapshot is invalid"):
        if action == "commit":
            commit_deletion(
                layout,
                engine,
                preview.operation_id,
                commit_token=preview.commit_token,
                catalog_revision=preview.catalog_revision,
            )
        else:
            restore_deletion(
                layout,
                engine,
                preview.operation_id,
                confirmation=preview.operation_id,
            )
    assert retained_bytes is not None

    manifest_path = Path(preview.manifest)
    manifest_path.chmod(0o600)
    manifest_path.write_bytes(retained_bytes)
    manifest_path.chmod(0o400)
    monkeypatch.setattr(deletion_service, "_write_manifest", original_write)
    if action == "commit":
        result = commit_deletion(
            layout,
            engine,
            preview.operation_id,
            commit_token=preview.commit_token,
            catalog_revision=preview.catalog_revision,
        )
    else:
        result = restore_deletion(
            layout,
            engine,
            preview.operation_id,
            confirmation=preview.operation_id,
        )
    assert result.status == ("committed" if action == "commit" else "restored")
    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o400
    assert manifest_path.stat().st_nlink == 1
    assert [path.name for path in manifest_path.parent.iterdir()] == ["manifest.json"]


def test_commit_rejects_catalog_changes_after_preview(
    deletion_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = deletion_archive
    message = _message(engine, "Older chunk")
    preview = preview_deletion(
        layout,
        engine,
        DeletionSelector(entity_types=("message",), text="Older chunk", whole_field=True),
    )
    with Session(engine) as session, session.begin():
        session.execute(
            update(Message).where(Message.id == message.id).values(content_display="Changed")
        )
    with pytest.raises(StaleDeletionPreviewError):
        commit_deletion(
            layout,
            engine,
            preview.operation_id,
            commit_token=preview.commit_token,
            catalog_revision=preview.catalog_revision,
        )


def test_commit_refuses_a_tampered_recovery_manifest_before_quarantine(
    deletion_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = deletion_archive
    message = _message(engine, "Older chunk")
    preview = preview_deletion(
        layout,
        engine,
        DeletionSelector(entity_types=("message",), text="Older chunk", whole_field=True),
    )
    manifest_path = Path(preview.manifest)
    manifest_path.chmod(0o600)
    payload = json.loads(manifest_path.read_text())
    payload["items"][0]["reason"] = "tampered"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    manifest_path.chmod(0o400)

    with pytest.raises(DeletionError, match="does not match catalog items"):
        commit_deletion(
            layout,
            engine,
            preview.operation_id,
            commit_token=preview.commit_token,
            catalog_revision=preview.catalog_revision,
        )
    with Session(engine) as session:
        retained = session.get(Message, message.pk)
        operation = session.get(DeletionOperation, preview.operation_id)
        assert retained is not None and retained.deleted_at is None
        assert operation is not None and operation.status == "previewed"


def test_restore_refuses_quarantined_rows_modified_after_commit(
    deletion_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = deletion_archive
    message = _message(engine, "Older chunk")
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
    with Session(engine) as session, session.begin():
        session.execute(
            update(Message)
            .where(Message.id == message.id)
            .values(content_display="modified while quarantined")
        )
    with pytest.raises(DeletionError, match="modified message row"):
        restore_deletion(layout, engine, preview.operation_id, confirmation=preview.operation_id)
    with Session(engine) as session:
        quarantined = session.scalar(select(Message).where(Message.id == message.id))
        assert quarantined is not None and quarantined.deleted_at is not None


def test_restore_refuses_a_manifest_item_list_modified_after_commit(
    deletion_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = deletion_archive
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
    manifest_path = Path(preview.manifest)
    manifest_path.chmod(0o600)
    payload = json.loads(manifest_path.read_text())
    payload["items"][0]["reason"] = "tampered"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    manifest_path.chmod(0o400)

    with pytest.raises(DeletionError, match="does not match catalog items"):
        restore_deletion(layout, engine, preview.operation_id, confirmation=preview.operation_id)
    with Session(engine) as session:
        operation = session.get(DeletionOperation, preview.operation_id)
        assert operation is not None and operation.status == "committed"


def test_hard_purge_refuses_a_tampered_restore_manifest(
    deletion_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = deletion_archive
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
    manifest_path = Path(preview.manifest)
    manifest_path.chmod(0o600)
    payload = json.loads(manifest_path.read_text())
    payload["items"][0]["reason"] = "tampered"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    manifest_path.chmod(0o400)

    purge = plan_hard_purge(layout, engine, preview.operation_id, scope="normalized")
    assert purge.executable is False
    assert "deletion_manifest_missing_or_unsafe" in purge.blockers
    with pytest.raises(HardPurgeBlockedError, match="deletion_manifest_missing_or_unsafe"):
        execute_hard_purge(
            layout,
            engine,
            preview.operation_id,
            scope="normalized",
            purge_token=purge.purge_token,
            purge_revision=purge.purge_revision,
            confirmation=preview.operation_id,
        )


def test_commit_and_restore_exactly_reactivate_nlp_annotations_and_hourly_rollups(
    deletion_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = deletion_archive
    message = _message(engine, "Older chunk")
    with Session(engine) as session, session.begin():
        session.add(
            _nlp_annotation(
                "annotation-message-deletion",
                target_type="message",
                target_id=message.id,
            )
        )
        session.add(
            NlpHourlyRollup(
                id="rollup-deletion-fixture",
                task="sentiment",
                hour_start_ms=1_700_000_000_000,
                direction="owner",
                pipeline_version="deletion-fixture-v1",
                model_name="fixture-model",
                model_revision="fixture-revision",
                configuration_sha256="5" * 64,
                configuration_json={"time_bucket": "UTC hour"},
                predicted_label="neutral",
                sample_count=1,
                confidence_sum=0.9,
                probability_sums_json={"labels": {"neutral": 0.9}},
                active=True,
            )
        )
    preview = preview_deletion(
        layout,
        engine,
        DeletionSelector(entity_types=("message",), text="Older chunk", whole_field=True),
    )
    committed = commit_deletion(
        layout,
        engine,
        preview.operation_id,
        commit_token=preview.commit_token,
        catalog_revision=preview.catalog_revision,
    )
    assert committed.affected["nlp_annotations_deactivated"] == 1
    assert committed.affected["nlp_hourly_rollups_invalidated"] == 1
    with Session(engine) as session:
        annotation = session.get(NlpAnnotation, "annotation-message-deletion")
        rollup = session.get(NlpHourlyRollup, "rollup-deletion-fixture")
        assert annotation is not None and annotation.active is False
        assert rollup is not None and rollup.active is False
    restored = restore_deletion(
        layout, engine, preview.operation_id, confirmation=preview.operation_id
    )
    assert restored.affected["nlp_annotations_reactivated"] == 1
    assert restored.affected["nlp_hourly_rollups_reactivated"] == 1
    with Session(engine) as session:
        annotation = session.get(NlpAnnotation, "annotation-message-deletion")
        rollup = session.get(NlpHourlyRollup, "rollup-deletion-fixture")
        assert annotation is not None and annotation.active is True
        assert rollup is not None and rollup.active is True


def test_restore_fails_closed_if_nlp_state_changes_during_quarantine(
    deletion_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = deletion_archive
    message = _message(engine, "Older chunk")
    with Session(engine) as session, session.begin():
        session.add(
            NlpHourlyRollup(
                id="rollup-drift-fixture",
                task="sentiment",
                hour_start_ms=1_700_000_000_000,
                direction="owner",
                pipeline_version="deletion-fixture-v1",
                model_name="fixture-model",
                model_revision="fixture-revision",
                configuration_sha256="6" * 64,
                configuration_json={"time_bucket": "UTC hour"},
                predicted_label="neutral",
                sample_count=1,
                confidence_sum=0.9,
                probability_sums_json={"labels": {"neutral": 0.9}},
                active=True,
            )
        )
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
    with Session(engine) as session, session.begin():
        rollup = session.get(NlpHourlyRollup, "rollup-drift-fixture")
        assert rollup is not None and rollup.active is False
        rollup.active = True

    with pytest.raises(DeletionError, match="changed while rows were quarantined"):
        restore_deletion(
            layout,
            engine,
            preview.operation_id,
            confirmation=preview.operation_id,
        )

    with Session(engine) as session:
        operation = session.get(DeletionOperation, preview.operation_id)
        quarantined_message = session.get(Message, message.pk)
        assert operation is not None and operation.status == "committed"
        assert quarantined_message is not None and quarantined_message.deleted_at is not None


def test_explicit_ids_do_not_broaden_other_entity_types_and_conversation_cascades(
    deletion_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = deletion_archive
    with Session(engine) as session:
        assert session.scalar(select(func.count(ActivityRecord.id)))
        conversation = session.scalar(
            select(Conversation).where(Conversation.title_display == "Alex Example")
        )
        assert conversation is not None
        original_messages = conversation.message_count
        original_attachments = conversation.attachment_count

    preview = preview_deletion(
        layout,
        engine,
        DeletionSelector(
            entity_types=("message", "activity", "conversation"),
            entity_ids={"conversation": (conversation.id,)},
        ),
    )
    assert preview.impact["direct"] == {
        "activity": 0,
        "attachment": 0,
        "conversation": 1,
        "message": 0,
    }
    assert preview.impact["selected"]["message"] == original_messages
    assert preview.impact["selected"]["attachment"] == original_attachments
    assert preview.impact["selected"]["activity"] == 0

    commit_deletion(
        layout,
        engine,
        preview.operation_id,
        commit_token=preview.commit_token,
        catalog_revision=preview.catalog_revision,
    )
    with Session(engine) as session:
        assert (
            session.scalar(
                select(func.count(Message.id)).where(
                    Message.conversation_id == conversation.id, Message.deleted_at.is_(None)
                )
            )
            == 0
        )
        assert session.scalar(
            select(func.count(ActivityRecord.id)).where(ActivityRecord.deleted_at.is_(None))
        )
    restore_deletion(layout, engine, preview.operation_id, confirmation=preview.operation_id)
    with Session(engine) as session:
        restored = session.get(Conversation, conversation.id)
        assert restored is not None
        assert restored.message_count == original_messages
        assert restored.attachment_count == original_attachments


def test_shared_blob_is_preserved_when_one_occurrence_is_purged_from_normalized_catalog(
    deletion_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = deletion_archive
    with Session(engine) as session, session.begin():
        original = session.scalar(
            select(AttachmentOccurrence)
            .where(AttachmentOccurrence.blob_sha256.is_not(None))
            .order_by(AttachmentOccurrence.id)
            .limit(1)
        )
        assert original is not None and original.blob_sha256 is not None
        blob_sha256 = original.blob_sha256
        original_id = original.id
        session.add(
            AttachmentOccurrence(
                id="e" * 64,
                message_id=original.message_id,
                activity_record_id=original.activity_record_id,
                conversation_id=original.conversation_id,
                source_file_id=original.source_file_id,
                resolved_source_file_id=original.resolved_source_file_id,
                blob_sha256=original.blob_sha256,
                kind=original.kind,
                ordinal=999,
                source_uri=original.source_uri,
                filename=original.filename,
                media_type=original.media_type,
                size_bytes=original.size_bytes,
                timestamp_ms=original.timestamp_ms,
                availability=original.availability,
                metadata_json={"shared_blob_fixture": True},
            )
        )

    preview = preview_deletion(
        layout,
        engine,
        DeletionSelector(entity_types=("attachment",), entity_ids={"attachment": (original_id,)}),
    )
    assert preview.impact["attachments"]["shared_blobs"] == 1
    commit_deletion(
        layout,
        engine,
        preview.operation_id,
        commit_token=preview.commit_token,
        catalog_revision=preview.catalog_revision,
    )
    purge = plan_hard_purge(layout, engine, preview.operation_id, scope="normalized")
    assert purge.executable
    execute_hard_purge(
        layout,
        engine,
        preview.operation_id,
        scope="normalized",
        purge_token=purge.purge_token,
        purge_revision=purge.purge_revision,
        confirmation=preview.operation_id,
    )
    with Session(engine) as session:
        assert session.get(AttachmentOccurrence, original_id) is None
        assert session.get(AttachmentOccurrence, "e" * 64) is not None
        assert session.get(Blob, blob_sha256) is not None
    assert layout.blob_path(blob_sha256).is_file()


def test_shared_derived_nlp_is_retained_then_exclusive_workspace_purge_removes_it(
    deletion_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = deletion_archive
    first_id, blob_sha256, blob_path = _add_synthetic_attachment(
        layout, engine, marker="derived-shared"
    )
    second_id = "4" * 64
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
                filename="derived-shared-copy.bin",
                media_type=first.media_type,
                size_bytes=first.size_bytes,
                timestamp_ms=first.timestamp_ms,
                availability="unlinked",
                metadata_json={"shared_derived_fixture": True},
            )
        )
    transcription_id, extraction_id, annotation_ids, derivative_path = _add_fixture_derived_nlp(
        layout, engine, blob_sha256
    )

    first_preview = preview_deletion(
        layout,
        engine,
        DeletionSelector(entity_types=("attachment",), entity_ids={"attachment": (first_id,)}),
    )
    assert first_preview.impact["attachments"]["shared_blobs"] == 1
    commit_deletion(
        layout,
        engine,
        first_preview.operation_id,
        commit_token=first_preview.commit_token,
        catalog_revision=first_preview.catalog_revision,
    )
    first_purge = plan_hard_purge(layout, engine, first_preview.operation_id, scope="normalized")
    execute_hard_purge(
        layout,
        engine,
        first_preview.operation_id,
        scope="normalized",
        purge_token=first_purge.purge_token,
        purge_revision=first_purge.purge_revision,
        confirmation=first_preview.operation_id,
    )
    with Session(engine) as session:
        assert all(session.get(NlpAnnotation, identifier).active for identifier in annotation_ids)
        assert session.get(AudioTranscription, transcription_id) is not None
        assert session.get(AttachmentTextExtraction, extraction_id) is not None

    second_preview = preview_deletion(
        layout,
        engine,
        DeletionSelector(entity_types=("attachment",), entity_ids={"attachment": (second_id,)}),
    )
    assert second_preview.impact["attachments"]["exclusive_blobs"] == 1
    committed = commit_deletion(
        layout,
        engine,
        second_preview.operation_id,
        commit_token=second_preview.commit_token,
        catalog_revision=second_preview.catalog_revision,
    )
    assert committed.affected["nlp_annotations_deactivated"] == 2
    second_purge = plan_hard_purge(layout, engine, second_preview.operation_id, scope="workspace")
    assert second_purge.executable
    execute_hard_purge(
        layout,
        engine,
        second_preview.operation_id,
        scope="workspace",
        purge_token=second_purge.purge_token,
        purge_revision=second_purge.purge_revision,
        confirmation=second_preview.operation_id,
    )
    assert not blob_path.exists()
    assert not derivative_path.exists()
    assert json.loads(layout.derivative_manifest.read_text())["records"] == []
    with Session(engine) as session:
        assert all(session.get(NlpAnnotation, identifier) is None for identifier in annotation_ids)
        assert session.get(AudioTranscription, transcription_id) is None
        assert session.get(AttachmentTextExtraction, extraction_id) is None


def test_sequential_shared_quarantines_delegate_restore_then_final_purge_removes_bytes(
    deletion_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = deletion_archive
    purged_b_canary = "PURGED_B_DERIVED_CANARY"
    first_id, blob_sha256, blob_path = _add_synthetic_attachment(
        layout, engine, marker="sequential-shared"
    )
    second_id = "7" * 64
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
                filename="sequential-shared-copy.bin",
                media_type=first.media_type,
                size_bytes=first.size_bytes,
                timestamp_ms=first.timestamp_ms,
                availability="unlinked",
                metadata_json={"sequential_shared_fixture": True},
            )
        )
    _transcription_id, _extraction_id, annotation_ids, derivative_path = _add_fixture_derived_nlp(
        layout, engine, blob_sha256
    )
    with Session(engine) as session, session.begin():
        session.add(
            NlpHourlyRollup(
                id="rollup-sequential-shared",
                task="sentiment",
                hour_start_ms=1_700_000_000_000,
                direction="owner",
                pipeline_version="deletion-fixture-v1",
                model_name="fixture-model",
                model_revision="fixture-revision",
                configuration_sha256="8" * 64,
                configuration_json={
                    "time_bucket": "UTC hour",
                    "private_canary": purged_b_canary,
                },
                predicted_label=purged_b_canary,
                sample_count=1,
                confidence_sum=0.9,
                probability_sums_json={"labels": {purged_b_canary: 0.9}},
                active=True,
            )
        )

    first_preview = preview_deletion(
        layout,
        engine,
        DeletionSelector(entity_types=("attachment",), entity_ids={"attachment": (first_id,)}),
    )
    first_commit = commit_deletion(
        layout,
        engine,
        first_preview.operation_id,
        commit_token=first_preview.commit_token,
        catalog_revision=first_preview.catalog_revision,
    )
    assert first_commit.affected["nlp_annotations_deactivated"] == 0

    second_preview = preview_deletion(
        layout,
        engine,
        DeletionSelector(entity_types=("attachment",), entity_ids={"attachment": (second_id,)}),
    )
    second_commit = commit_deletion(
        layout,
        engine,
        second_preview.operation_id,
        commit_token=second_preview.commit_token,
        catalog_revision=second_preview.catalog_revision,
    )
    assert second_commit.affected["nlp_annotations_deactivated"] == 2
    assert second_commit.affected["logical_restore_delegations"] == 2

    second_purge = plan_hard_purge(
        layout,
        engine,
        second_preview.operation_id,
        scope="workspace",
    )
    assert second_purge.executable
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
    assert derivative_path.is_file()
    with Session(engine) as session:
        assert session.get(NlpHourlyRollup, "rollup-sequential-shared") is None
        assert all(
            session.get(NlpAnnotation, identifier).active is False for identifier in annotation_ids
        )

    restored = restore_deletion(
        layout,
        engine,
        first_preview.operation_id,
        confirmation=first_preview.operation_id,
    )
    assert restored.affected["nlp_annotations_reactivated"] == 2
    assert restored.affected["global_derived_rebuild_required"] == 1
    with Session(engine) as session:
        first = session.get(AttachmentOccurrence, first_id)
        assert first is not None and first.deleted_at is None
        assert all(session.get(NlpAnnotation, identifier).active for identifier in annotation_ids)
        operation = session.get(DeletionOperation, first_preview.operation_id)
        assert operation is not None
        assert operation.purge_plan_json["restore_rebuild_required"] == {
            "required": True,
            "reason": "intervening_hard_purge",
            "product_kinds": [
                "search_windows",
                "semantic_index",
                "nlp_hourly_rollups",
                "conflict_aggregates",
                "topic_clusters",
            ],
            "invalidated_by_count": 1,
        }
        restore_audit = session.scalar(
            select(DeletionAudit)
            .where(
                DeletionAudit.operation_id == first_preview.operation_id,
                DeletionAudit.action == "restore",
            )
            .order_by(DeletionAudit.created_at.desc())
        )
        assert restore_audit is not None
        assert (
            restore_audit.detail_json["rebuild_required"]
            == operation.purge_plan_json["restore_rebuild_required"]
        )

    app = create_app(layout.root, security_disabled=True)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        served_payloads = [
            client.get("/api/v1/enrichment/nlp/hourly"),
            client.get("/api/v1/enrichment/conflict/status"),
            client.get("/api/v1/analytics/conflict-rate"),
            client.get("/api/v1/analytics/topics"),
        ]
        assert all(response.status_code == 200 for response in served_payloads)
        assert all(purged_b_canary not in response.text for response in served_payloads)
        assert served_payloads[0].json()["items"] == []
        assert served_payloads[3].json()["ready"] is False
    app.state.engine.dispose()

    final_preview = preview_deletion(
        layout,
        engine,
        DeletionSelector(entity_types=("attachment",), entity_ids={"attachment": (first_id,)}),
    )
    commit_deletion(
        layout,
        engine,
        final_preview.operation_id,
        commit_token=final_preview.commit_token,
        catalog_revision=final_preview.catalog_revision,
    )
    final_purge = plan_hard_purge(
        layout,
        engine,
        final_preview.operation_id,
        scope="workspace",
    )
    assert final_purge.executable
    execute_hard_purge(
        layout,
        engine,
        final_preview.operation_id,
        scope="workspace",
        purge_token=final_purge.purge_token,
        purge_revision=final_purge.purge_revision,
        confirmation=final_preview.operation_id,
    )
    assert not blob_path.exists()
    assert not derivative_path.exists()


def test_restore_rebuild_mode_tolerates_shared_search_window_removed_by_later_purge(
    deletion_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = deletion_archive
    first_message = _message(engine, "Older chunk")
    with Session(engine) as session:
        second_message = session.scalar(
            select(Message)
            .where(
                Message.id != first_message.id,
                Message.conversation_id == first_message.conversation_id,
                Message.deleted_at.is_(None),
            )
            .order_by(Message.timestamp_ms, Message.id)
        )
        assert second_message is not None
        session.expunge(second_message)
    _table_name, config_id, window_id = _add_fixture_shared_vector_window(
        layout,
        engine,
        first_message,
        second_message,
    )

    first_preview = preview_deletion(
        layout,
        engine,
        DeletionSelector(
            entity_types=("message",),
            entity_ids={"message": (first_message.id,)},
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
            entity_types=("message",),
            entity_ids={"message": (second_message.id,)},
        ),
    )
    commit_deletion(
        layout,
        engine,
        second_preview.operation_id,
        commit_token=second_preview.commit_token,
        catalog_revision=second_preview.catalog_revision,
    )
    second_purge = plan_hard_purge(
        layout,
        engine,
        second_preview.operation_id,
        scope="normalized",
    )
    assert second_purge.executable
    execute_hard_purge(
        layout,
        engine,
        second_preview.operation_id,
        scope="normalized",
        purge_token=second_purge.purge_token,
        purge_revision=second_purge.purge_revision,
        confirmation=second_preview.operation_id,
    )

    with Session(engine) as session:
        assert session.get(SearchWindow, window_id) is None
        assert session.get(Message, second_message.pk) is None
        first_row = session.get(Message, first_message.pk)
        assert first_row is not None and first_row.deleted_at is not None
        config = session.get(SemanticIndexConfig, config_id)
        assert config is not None
        assert config.active is False and config.status == "stale" and config.window_count == 0

    restored = restore_deletion(
        layout,
        engine,
        first_preview.operation_id,
        confirmation=first_preview.operation_id,
    )
    assert restored.affected["global_derived_rebuild_required"] == 1
    with Session(engine) as session:
        first_row = session.get(Message, first_message.pk)
        assert first_row is not None and first_row.deleted_at is None
        assert session.get(SearchWindow, window_id) is None
        config = session.get(SemanticIndexConfig, config_id)
        assert config is not None and config.active is False and config.status == "stale"
        operation = session.get(DeletionOperation, first_preview.operation_id)
        assert operation is not None
        assert operation.purge_plan_json["restore_rebuild_required"] == {
            "required": True,
            "reason": "intervening_hard_purge",
            "product_kinds": [
                "search_windows",
                "semantic_index",
                "nlp_hourly_rollups",
                "conflict_aggregates",
                "topic_clusters",
            ],
            "invalidated_by_count": 1,
        }
        restore_audit = session.scalar(
            select(DeletionAudit)
            .where(
                DeletionAudit.operation_id == first_preview.operation_id,
                DeletionAudit.action == "restore",
            )
            .order_by(DeletionAudit.created_at.desc())
        )
        assert restore_audit is not None
        assert (
            restore_audit.detail_json["rebuild_required"]
            == operation.purge_plan_json["restore_rebuild_required"]
        )


def test_workspace_purge_rewrites_raw_json_and_complete_source_manifests(
    deletion_archive: tuple[ArchiveLayout, Engine],
    tmp_path: Path,
) -> None:
    layout, engine = deletion_archive
    build_filesystem_library(layout, engine)
    message = _message(engine, "Older chunk")
    with Session(engine) as session:
        source = session.get(SourceFile, message.source_file_id)
        assert source is not None and source.sha256 is not None
        old_source_sha256 = source.sha256
        old_source_size = source.size_bytes
        source_record_id = message.source_record_id
    assert _fts_count(engine, message.pk, '"older"') == 1
    assert any(
        message.id in path.read_text(encoding="utf-8") for path in layout.library.rglob("*.ndjson")
    )
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
    purge = plan_hard_purge(layout, engine, preview.operation_id, scope="workspace")
    assert purge.executable
    assert purge.impact["raw_source_files_rewritten"] == 1
    assert purge.impact["filesystem_library_present"] is True
    result = execute_hard_purge(
        layout,
        engine,
        preview.operation_id,
        scope="workspace",
        purge_token=purge.purge_token,
        purge_revision=purge.purge_revision,
        confirmation=preview.operation_id,
    )
    assert result.affected["message"] == 1
    assert _fts_count(engine, message.pk, '"older"') == 0
    with Session(engine) as session:
        assert session.get(Message, message.pk) is None
        assert session.get(SourceRecord, source_record_id) is None
        rewritten = session.get(SourceFile, message.source_file_id)
        assert rewritten is not None and rewritten.sha256 is not None
        assert rewritten.sha256 != old_source_sha256
        assert rewritten.size_bytes != old_source_size
        assert layout.blob_path(rewritten.sha256).is_file()
        assert session.get(Blob, old_source_sha256) is None
        assert session.get(Blob, rewritten.sha256) is not None
        new_source_sha256 = rewritten.sha256
    assert not layout.blob_path(old_source_sha256).exists()
    assert layout.library_completion_marker.is_file()
    assert not any(
        message.id in path.read_text(encoding="utf-8") for path in layout.library.rglob("*.ndjson")
    )

    original_rows = [
        json.loads(line)
        for line in layout.original_files_manifest.read_text(encoding="utf-8").splitlines()
    ]
    relocation_rows = [
        json.loads(line)
        for line in layout.relocation_manifest.read_text(encoding="utf-8").splitlines()
    ]
    assert any(row["sha256"] == new_source_sha256 for row in original_rows)
    assert any(
        row["sha256"] == new_source_sha256 and row["operation"] == "hard-purge-tombstone-v1"
        for row in relocation_rows
    )
    rewritten_payload = json.loads(layout.blob_path(new_source_sha256).read_text())
    assert rewritten_payload["messages"][message.source_array_index] == {
        "meta_archive_purged": {"version": 1}
    }

    reconstruction_target = tmp_path / "reconstructed-after-purge"
    assert reconstruct_source(layout, reconstruction_target) == len(relocation_rows)
    rewritten_relocation = next(
        row for row in relocation_rows if row["sha256"] == new_source_sha256
    )
    reconstructed_payload = json.loads(
        (reconstruction_target / rewritten_relocation["source"]).read_text()
    )
    assert reconstructed_payload["messages"][message.source_array_index] == {
        "meta_archive_purged": {"version": 1}
    }

    # Force a parser pass instead of relying on imported-digest idempotency. Tombstones are valid
    # archive members and must never resurrect the purged stable ID.
    with Session(engine) as session, session.begin():
        session.execute(
            update(SourceFile)
            .where(SourceFile.id == message.source_file_id)
            .values(imported_parser_version=None, imported_sha256=None)
        )
    import_messages(layout, engine)
    with Session(engine) as session:
        assert session.scalar(select(Message).where(Message.id == message.id)) is None


def test_workspace_purge_executes_for_an_unreferenced_synthetic_blob(
    deletion_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = deletion_archive
    attachment_id, blob_sha256, blob_path = _add_synthetic_attachment(
        layout, engine, marker="success"
    )
    analytics_cache_file = layout.analytics_cache / "deletion-fixture.json"
    analytics_cache_file.write_text('{"cached": "selected archive aggregate"}\n')
    preview = preview_deletion(
        layout,
        engine,
        DeletionSelector(entity_types=("attachment",), entity_ids={"attachment": (attachment_id,)}),
    )
    commit_deletion(
        layout,
        engine,
        preview.operation_id,
        commit_token=preview.commit_token,
        catalog_revision=preview.catalog_revision,
    )
    purge = plan_hard_purge(layout, engine, preview.operation_id, scope="workspace")
    assert purge.executable
    assert purge.impact["workspace_files"] == 1
    assert purge.impact["analytics_cache_files"] == 1
    result = execute_hard_purge(
        layout,
        engine,
        preview.operation_id,
        scope="workspace",
        purge_token=purge.purge_token,
        purge_revision=purge.purge_revision,
        confirmation=preview.operation_id,
    )
    assert result.affected["attachment"] == 1
    assert result.affected["blobs"] == 1
    assert not blob_path.exists()
    assert not analytics_cache_file.exists()
    assert not layout.analytics_cache.exists()
    assert not Path(preview.manifest).exists()
    assert not (layout.trash / "purge-staging" / preview.operation_id).exists()
    with Session(engine) as session:
        assert session.get(AttachmentOccurrence, attachment_id) is None
        assert session.get(Blob, blob_sha256) is None


def test_workspace_purge_rolls_back_staged_files_when_catalog_mutation_fails(
    deletion_archive: tuple[ArchiveLayout, Engine], monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, engine = deletion_archive
    attachment_id, blob_sha256, blob_path = _add_synthetic_attachment(
        layout, engine, marker="rollback"
    )
    analytics_cache_file = layout.analytics_cache / "deletion-rollback-fixture.json"
    analytics_cache_payload = b'{"cached": "must be restored"}\n'
    analytics_cache_file.write_bytes(analytics_cache_payload)
    preview = preview_deletion(
        layout,
        engine,
        DeletionSelector(entity_types=("attachment",), entity_ids={"attachment": (attachment_id,)}),
    )
    commit_deletion(
        layout,
        engine,
        preview.operation_id,
        commit_token=preview.commit_token,
        catalog_revision=preview.catalog_revision,
    )
    purge = plan_hard_purge(layout, engine, preview.operation_id, scope="workspace")
    assert purge.executable

    def fail_catalog_mutation(*_args: object, **_kwargs: object) -> dict[str, int]:
        raise RuntimeError("fixture catalog failure")

    monkeypatch.setattr(
        "meta_archive.services.deletion._delete_normalized_rows", fail_catalog_mutation
    )
    with pytest.raises(RuntimeError, match="fixture catalog failure"):
        execute_hard_purge(
            layout,
            engine,
            preview.operation_id,
            scope="workspace",
            purge_token=purge.purge_token,
            purge_revision=purge.purge_revision,
            confirmation=preview.operation_id,
        )
    assert analytics_cache_file.read_bytes() == analytics_cache_payload
    assert blob_path.read_bytes() == b"stage5-workspace-purge-rollback"
    assert not (layout.trash / "purge-staging" / preview.operation_id).exists()
    assert Path(preview.manifest).is_file()
    with Session(engine) as session:
        attachment = session.get(AttachmentOccurrence, attachment_id)
        operation = session.get(DeletionOperation, preview.operation_id)
        assert attachment is not None and attachment.deleted_at is not None
        assert session.get(Blob, blob_sha256) is not None
        assert operation is not None and operation.status == "committed"
        assert operation.purge_token_hash is not None


def test_workspace_raw_rewrite_rolls_back_bytes_manifests_and_catalog(
    deletion_archive: tuple[ArchiveLayout, Engine], monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, engine = deletion_archive
    message = _message(engine, "Older chunk")
    with Session(engine) as session:
        source = session.get(SourceFile, message.source_file_id)
        assert source is not None and source.sha256 is not None
        source_sha256 = source.sha256
        source_size = source.size_bytes
    source_blob_bytes = layout.blob_path(source_sha256).read_bytes()
    original_manifest_bytes = layout.original_files_manifest.read_bytes()
    relocation_manifest_bytes = layout.relocation_manifest.read_bytes()
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
    purge = plan_hard_purge(layout, engine, preview.operation_id, scope="workspace")
    assert purge.executable

    def fail_catalog_mutation(*_args: object, **_kwargs: object) -> dict[str, int]:
        raise RuntimeError("raw rewrite rollback fixture")

    monkeypatch.setattr(
        "meta_archive.services.deletion._delete_normalized_rows", fail_catalog_mutation
    )
    with pytest.raises(RuntimeError, match="raw rewrite rollback fixture"):
        execute_hard_purge(
            layout,
            engine,
            preview.operation_id,
            scope="workspace",
            purge_token=purge.purge_token,
            purge_revision=purge.purge_revision,
            confirmation=preview.operation_id,
        )

    assert layout.blob_path(source_sha256).read_bytes() == source_blob_bytes
    assert layout.original_files_manifest.read_bytes() == original_manifest_bytes
    assert layout.relocation_manifest.read_bytes() == relocation_manifest_bytes
    with Session(engine) as session:
        source = session.get(SourceFile, message.source_file_id)
        operation = session.get(DeletionOperation, preview.operation_id)
        assert source is not None
        assert source.sha256 == source_sha256 and source.size_bytes == source_size
        assert session.get(Message, message.pk) is not None
        assert operation is not None and operation.status == "committed"


def test_attachment_only_workspace_purge_updates_retained_parent_fingerprint(
    deletion_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = deletion_archive
    with Session(engine) as session:
        exclusive_blob_ids = (
            select(AttachmentOccurrence.blob_sha256)
            .where(AttachmentOccurrence.blob_sha256.is_not(None))
            .group_by(AttachmentOccurrence.blob_sha256)
            .having(func.count(AttachmentOccurrence.id) == 1)
        )
        attachment = session.scalar(
            select(AttachmentOccurrence)
            .join(Message, Message.id == AttachmentOccurrence.message_id)
            .where(
                AttachmentOccurrence.message_id.is_not(None),
                AttachmentOccurrence.source_file_id.is_not(None),
                AttachmentOccurrence.resolved_source_file_id.is_not(None),
                AttachmentOccurrence.blob_sha256.in_(exclusive_blob_ids),
                Message.json_pointer.like("/messages/%"),
            )
            .order_by(AttachmentOccurrence.id)
        )
        assert attachment is not None and attachment.message_id is not None
        parent = session.scalar(select(Message).where(Message.id == attachment.message_id))
        assert parent is not None
        original_fingerprint = parent.logical_fingerprint
        parent_id = parent.id
        source_record_id = parent.source_record_id
        source_file_id = parent.source_file_id
        attachment_id = attachment.id
        media_source_file_id = attachment.resolved_source_file_id
        old_media_sha256 = attachment.blob_sha256
        assert media_source_file_id is not None and old_media_sha256 is not None

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
    assert purge.executable, purge.blockers
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
        retained = session.scalar(select(Message).where(Message.id == parent_id))
        source_record = session.get(SourceRecord, source_record_id)
        source = session.get(SourceFile, source_file_id)
        assert retained is not None and source_record is not None and source is not None
        assert retained.logical_fingerprint != original_fingerprint
        assert retained.logical_fingerprint == source_record.logical_fingerprint
        assert session.get(AttachmentOccurrence, attachment_id) is None
        rewritten_payload = json.loads(layout.blob_path(str(source.sha256)).read_text())
        raw_parent = rewritten_payload["messages"][retained.source_array_index]
        assert retained.logical_fingerprint == canonical_fingerprint("facebook-message", raw_parent)
        assert any(
            value == {"meta_archive_purged": {"version": 1}}
            for value in (
                *raw_parent.get("photos", []),
                *raw_parent.get("videos", []),
                *raw_parent.get("audio_files", []),
                *raw_parent.get("files", []),
                *raw_parent.get("gifs", []),
                raw_parent.get("sticker"),
            )
        )
        media_source = session.get(SourceFile, media_source_file_id)
        assert media_source is not None and media_source.sha256 is not None
        assert media_source.sha256 != old_media_sha256
        assert media_source.json_status == "purged"
        assert layout.blob_path(media_source.sha256).read_bytes() == (
            b'{"meta_archive_purged":{"kind":"binary","version":1}}\n'
        )
    assert not layout.blob_path(old_media_sha256).exists()


def test_activity_child_purge_updates_retained_parent_payload_and_fingerprint(
    deletion_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = deletion_archive
    with Session(engine) as session:
        child = session.scalar(
            select(ActivityRecord)
            .where(ActivityRecord.parent_id.is_not(None))
            .order_by(ActivityRecord.id)
        )
        assert child is not None and child.parent_id is not None
        parent = session.scalar(select(ActivityRecord).where(ActivityRecord.id == child.parent_id))
        assert parent is not None
        parent_source_record = session.get(SourceRecord, parent.source_record_id)
        assert parent_source_record is not None
        child_id = child.id
        child_source_record_id = child.source_record_id
        parent_id = parent.id
        parent_source_record_id = parent.source_record_id
        source_file_id = parent.source_file_id
        original_parent_fingerprint = parent_source_record.logical_fingerprint

    preview = preview_deletion(
        layout,
        engine,
        DeletionSelector(
            entity_types=("activity",),
            entity_ids={"activity": (child_id,)},
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
        retained_parent = session.scalar(
            select(ActivityRecord).where(ActivityRecord.id == parent_id)
        )
        retained_source_record = session.get(SourceRecord, parent_source_record_id)
        source = session.get(SourceFile, source_file_id)
        assert retained_parent is not None and retained_source_record is not None
        assert source is not None and source.sha256 is not None
        assert retained_source_record.logical_fingerprint != original_parent_fingerprint
        raw_payload = retained_parent.extra_json.get("raw_payload")
        assert raw_payload == json.loads(layout.blob_path(source.sha256).read_text())
        assert retained_source_record.logical_fingerprint == canonical_fingerprint(
            retained_source_record.record_type,
            {
                "category": retained_parent.category,
                "subtype": retained_parent.subtype,
                "payload": raw_payload,
            },
        )
        assert session.scalar(select(ActivityRecord).where(ActivityRecord.id == child_id)) is None
        assert session.get(SourceRecord, child_source_record_id) is None


def test_conversation_workspace_purge_tombstones_envelopes_and_prevents_reimport(
    deletion_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = deletion_archive
    with Session(engine) as session:
        conversation = session.scalar(
            select(Conversation).where(Conversation.title_display == "Alex Example")
        )
        assert conversation is not None
        conversation_id = conversation.id
    preview = preview_deletion(
        layout,
        engine,
        DeletionSelector(
            entity_types=("conversation",),
            entity_ids={"conversation": (conversation_id,)},
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
    assert purge.impact["raw_source_files_rewritten"] > 1
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
        assert session.get(Conversation, conversation_id) is None
    import_messages(layout, engine, force=True)
    with Session(engine) as session:
        assert session.get(Conversation, conversation_id) is None
        assert (
            session.scalar(select(Conversation).where(Conversation.title_display == "Alex Example"))
            is None
        )


def test_workspace_purge_rebuilds_library_against_rewritten_manifests(
    deletion_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = deletion_archive
    build_filesystem_library(layout, engine)
    message = _message(engine, "Older chunk")
    assert any(
        message.id in path.read_text(encoding="utf-8") for path in layout.library.rglob("*.ndjson")
    )
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
    purge = plan_hard_purge(layout, engine, preview.operation_id, scope="workspace")
    assert purge.executable and purge.impact["filesystem_library_present"] is True
    execute_hard_purge(
        layout,
        engine,
        preview.operation_id,
        scope="workspace",
        purge_token=purge.purge_token,
        purge_revision=purge.purge_revision,
        confirmation=preview.operation_id,
    )
    assert layout.library_completion_marker.is_file()
    marker = json.loads(layout.library_completion_marker.read_text())
    assert marker["source_manifests"] == source_manifest_artifacts(layout)
    assert not any(
        message.id in path.read_text(encoding="utf-8") for path in layout.library.rglob("*.ndjson")
    )
    checks = verify_catalog(layout, engine, deep=True)
    assert not [check for check in checks if check["status"] == "fail"]


def test_workspace_purge_restores_library_tree_when_rebuild_fails(
    deletion_archive: tuple[ArchiveLayout, Engine], monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, engine = deletion_archive
    build_filesystem_library(layout, engine)
    before = _tree_digest_snapshot(layout.library)
    message = _message(engine, "Older chunk")
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
    purge = plan_hard_purge(layout, engine, preview.operation_id, scope="workspace")
    assert purge.executable

    def fail_library_build(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("library rebuild fixture")

    monkeypatch.setattr(
        "meta_archive.services.deletion.build_filesystem_library", fail_library_build
    )
    with pytest.raises(RuntimeError, match="library rebuild fixture"):
        execute_hard_purge(
            layout,
            engine,
            preview.operation_id,
            scope="workspace",
            purge_token=purge.purge_token,
            purge_revision=purge.purge_revision,
            confirmation=preview.operation_id,
        )
    assert _tree_digest_snapshot(layout.library) == before
    with Session(engine) as session:
        retained = session.scalar(select(Message).where(Message.id == message.id))
        operation = session.get(DeletionOperation, preview.operation_id)
        assert retained is not None and retained.deleted_at is not None
        assert operation is not None and operation.status == "committed"


def test_normalized_purge_physically_discards_lance_store_and_stales_index(
    deletion_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = deletion_archive
    message = _message(engine, "Older chunk")
    table_name, selected_window_id, retained_window_id = _add_fixture_vector_window(
        layout, engine, message
    )
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
    assert purge.executable and purge.impact["vector_store_present"] is True
    result = execute_hard_purge(
        layout,
        engine,
        preview.operation_id,
        scope="normalized",
        purge_token=purge.purge_token,
        purge_revision=purge.purge_revision,
        confirmation=preview.operation_id,
    )
    assert not layout.vectors.exists() or not any(layout.vectors.iterdir())
    assert result.affected["semantic_store_discarded"] == 1
    with Session(engine) as session:
        assert session.get(SearchWindow, selected_window_id) is None
        retained = session.get(SearchWindow, retained_window_id)
        assert retained is not None
        assert retained.embedding_config_hash is None
        assert retained.embedding_updated_at is None
        config = session.scalar(
            select(SemanticIndexConfig).where(SemanticIndexConfig.vector_table == table_name)
        )
        assert config is not None
        assert config.active is False
        assert config.window_count == 0
        assert config.status == "stale"
        assert session.scalar(select(func.count(SemanticWindowEmbedding.window_id))) == 0


def test_vector_purge_restores_original_lance_tree_when_catalog_delete_fails(
    deletion_archive: tuple[ArchiveLayout, Engine], monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, engine = deletion_archive
    message = _message(engine, "Older chunk")
    table_name, selected_window_id, retained_window_id = _add_fixture_vector_window(
        layout, engine, message
    )
    before = _tree_digest_snapshot(layout.vectors)
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
    assert purge.executable

    def fail_catalog_mutation(*_args: object, **_kwargs: object) -> dict[str, int]:
        raise RuntimeError("vector rollback fixture")

    monkeypatch.setattr(
        "meta_archive.services.deletion._delete_normalized_rows", fail_catalog_mutation
    )
    with pytest.raises(RuntimeError, match="vector rollback fixture"):
        execute_hard_purge(
            layout,
            engine,
            preview.operation_id,
            scope="normalized",
            purge_token=purge.purge_token,
            purge_revision=purge.purge_revision,
            confirmation=preview.operation_id,
        )
    assert _tree_digest_snapshot(layout.vectors) == before
    table = lancedb.connect(str(layout.vectors)).open_table(table_name)
    ids = {str(row["id"]) for row in table.search().select(["id"]).limit(10).to_list()}
    assert ids == {selected_window_id, retained_window_id}
    with Session(engine) as session:
        retained = session.scalar(select(Message).where(Message.id == message.id))
        assert retained is not None and retained.deleted_at is not None


def test_deletion_api_exposes_preview_commit_restore_and_audit(
    deletion_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = deletion_archive
    engine.dispose()
    app = create_app(layout.root, security_disabled=True)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.post(
            "/api/v1/deletions/preview",
            json={
                "entity_types": ["message"],
                "text": "Older chunk",
                "whole_field": True,
            },
        )
        assert response.status_code == 201, response.text
        preview = response.json()
        assert set(preview) == {
            "operation_id",
            "commit_token",
            "catalog_revision",
            "selection_sha256",
            "impact",
        }
        assert str(layout.root) not in json.dumps(preview, sort_keys=True)
        wrong = client.post(
            f"/api/v1/deletions/{preview['operation_id']}/commit",
            json={
                "commit_token": "x" * 32,
                "catalog_revision": preview["catalog_revision"],
            },
        )
        assert wrong.status_code == 403
        committed = client.post(
            f"/api/v1/deletions/{preview['operation_id']}/commit",
            json={
                "commit_token": preview["commit_token"],
                "catalog_revision": preview["catalog_revision"],
            },
        )
        assert committed.status_code == 200, committed.text
        detail = client.get(f"/api/v1/deletions/{preview['operation_id']}")
        assert detail.status_code == 200
        assert detail.json()["status"] == "committed"
        assert [item["action"] for item in detail.json()["audits"]] == ["commit"]
        purge_preview = client.post(
            f"/api/v1/deletions/{preview['operation_id']}/purge/preview",
            json={},
        )
        assert purge_preview.status_code == 200, purge_preview.text
        assert purge_preview.json()["executable"] is True
        assert purge_preview.json()["scope"] == "workspace"
        public_impact = purge_preview.json()["impact"]
        assert public_impact["normalized_rows"] == 1
        assert public_impact["raw_source_files_rewritten"] == 1
        assert all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in public_impact.values()
        )
        assert (
            not {
                "raw_source_rewrite_error",
                "source_blob_references",
                "unregistered_vector_tables",
                "vector_catalog_count_mismatches",
                "vector_inventory_error",
                "deletion_manifest_error",
            }
            & public_impact.keys()
        )
        normalized_preview = client.post(
            f"/api/v1/deletions/{preview['operation_id']}/purge/preview",
            json={"scope": "normalized"},
        )
        assert normalized_preview.status_code == 200, normalized_preview.text
        assert normalized_preview.json()["scope"] == "normalized"
        restored = client.post(
            f"/api/v1/deletions/{preview['operation_id']}/restore",
            json={"confirmation": preview["operation_id"]},
        )
        assert restored.status_code == 200, restored.text
        second_preview_response = client.post(
            "/api/v1/deletions/preview",
            json={
                "entity_types": ["message"],
                "text": "Older chunk",
                "whole_field": True,
            },
        )
        assert second_preview_response.status_code == 201, second_preview_response.text
        second_preview = second_preview_response.json()
        history = client.get("/api/v1/deletions", params={"limit": 1})
        assert history.status_code == 200, history.text
        history_payload = history.json()
        assert history_payload["page"]["total"] == 2
        assert history_payload["page"]["next_cursor"]
        assert (
            client.get(
                "/api/v1/deletions",
                params={"limit": 2, "cursor": history_payload["page"]["next_cursor"]},
            ).status_code
            == 422
        )
        older_history = client.get(
            "/api/v1/deletions",
            params={"limit": 1, "cursor": history_payload["page"]["next_cursor"]},
        )
        assert older_history.status_code == 200, older_history.text
        older_history_payload = older_history.json()
        assert older_history_payload["page"]["next_cursor"] is None
        assert {
            history_payload["items"][0]["id"],
            older_history_payload["items"][0]["id"],
        } == {preview["operation_id"], second_preview["operation_id"]}
        assert (
            client.get("/api/v1/deletions", params={"cursor": "not-a-valid-cursor"}).status_code
            == 422
        )


def test_deletion_api_redacts_preview_manifest_and_service_error_paths(
    deletion_archive: tuple[ArchiveLayout, Engine], monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, engine = deletion_archive
    engine.dispose()
    app = create_app(layout.root, security_disabled=True)
    private_path = f"{layout.root}/PRIVATE_DELETION_ROUTE_CANARY/manifest.json"
    safe_preview = deletion_service.DeletionPreview(
        operation_id="11111111-1111-4111-8111-111111111111",
        commit_token="preview-token-with-sufficient-length",
        catalog_revision="a" * 64,
        selection_sha256="b" * 64,
        impact={"selected": {"message": 1}},
        manifest=private_path,
    )
    current_error: list[Exception | None] = [None]

    def preview_stub(*_args: object, **_kwargs: object) -> deletion_service.DeletionPreview:
        if current_error[0] is not None:
            raise current_error[0]
        return safe_preview

    monkeypatch.setattr("meta_archive.api.deletion_routes.preview_deletion", preview_stub)
    request_body = {"entity_types": ["message"], "text": "privacy canary"}
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.post("/api/v1/deletions/preview", json=request_body)
        assert response.status_code == 201, response.text
        assert response.json() == {
            "operation_id": safe_preview.operation_id,
            "commit_token": safe_preview.commit_token,
            "catalog_revision": safe_preview.catalog_revision,
            "selection_sha256": safe_preview.selection_sha256,
            "impact": safe_preview.impact,
        }
        assert private_path not in response.text
        assert "manifest" not in response.json()

        error_cases = [
            (
                InvalidDeletionTokenError(private_path),
                403,
                {
                    "code": "deletion_authorization_failed",
                    "message": (
                        "Deletion authorization failed. Refresh the operation and try again."
                    ),
                },
            ),
            (
                UnsafeDeletionError(private_path),
                422,
                {
                    "code": "deletion_request_unsafe",
                    "message": "Deletion request failed safety validation.",
                },
            ),
            (
                StaleDeletionPreviewError(private_path),
                409,
                {
                    "code": "deletion_preview_stale",
                    "message": ("Deletion preview is stale. Create a new preview and try again."),
                },
            ),
            (
                HardPurgeBlockedError(private_path),
                409,
                {
                    "code": "hard_purge_blocked",
                    "message": (
                        "Hard purge is blocked by a safety check. Preview again after resolving it."
                    ),
                },
            ),
            (
                DeletionError(private_path),
                409,
                {
                    "code": "deletion_operation_conflict",
                    "message": ("Deletion operation could not be completed in its current state."),
                },
            ),
        ]
        for service_error, status_code, public_detail in error_cases:
            current_error[0] = service_error
            failed = client.post("/api/v1/deletions/preview", json=request_body)
            assert failed.status_code == status_code
            assert failed.json() == {"detail": public_detail}
            assert private_path not in failed.text


def test_deletion_api_pages_revision_bound_privacy_safe_selection_items(
    deletion_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = deletion_archive
    with Session(engine) as session:
        activity = session.scalar(
            select(ActivityRecord)
            .where(ActivityRecord.deleted_at.is_(None))
            .order_by(ActivityRecord.id)
        )
        assert activity is not None
        activity_id = activity.id
    engine.dispose()
    app = create_app(layout.root, security_disabled=True)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.post(
            "/api/v1/deletions/preview",
            json={
                "entity_types": ["conversation"],
                "text": "Alex Example",
                "whole_field": True,
            },
        )
        assert response.status_code == 201, response.text
        preview = response.json()

        wrong_revision = client.get(
            f"/api/v1/deletions/{preview['operation_id']}/selection",
            params={"revision": "0" * 64},
        )
        assert wrong_revision.status_code == 409

        items: list[dict[str, object]] = []
        cursor: str | None = None
        first_cursor: str | None = None
        while True:
            params = {"revision": preview["selection_sha256"], "limit": 2}
            if cursor:
                params["cursor"] = cursor
            page_response = client.get(
                f"/api/v1/deletions/{preview['operation_id']}/selection",
                params=params,
            )
            assert page_response.status_code == 200, page_response.text
            page = page_response.json()
            assert page["operation_id"] == preview["operation_id"]
            assert page["revision"] == preview["selection_sha256"]
            assert page["status"] == "previewed"
            assert page["privacy"] == {
                "snippet_limit": 160,
                "label_limit": 96,
                "raw_fields_exposed": False,
            }
            assert page["page"]["total"] == preview["impact"]["record_count"]
            items.extend(page["items"])
            cursor = page["page"]["next_cursor"]
            first_cursor = first_cursor or cursor
            if cursor is None:
                break

        assert first_cursor is not None
        assert len(items) == preview["impact"]["record_count"]
        assert {item["entity_type"] for item in items} >= {
            "message",
            "conversation",
            "attachment",
        }
        allowed_fields = {
            "entity_type",
            "entity_id",
            "reason",
            "label",
            "label_truncated",
            "snippet",
            "snippet_truncated",
            "preview_limit",
            "timestamp_ms",
            "resource_kind",
            "availability",
            "media_type",
            "size_bytes",
            "quarantined",
            "jump",
        }
        for item in items:
            assert set(item) == allowed_fields
            assert len(str(item["label"])) <= 96
            assert item["preview_limit"] == 160
            if item["snippet"] is not None:
                assert len(str(item["snippet"])) <= 160
            assert item["quarantined"] is False
            assert item["jump"] is not None
        assert all(
            private_field not in json.dumps(items)
            for private_field in (
                "content_raw",
                "sender_raw",
                "sensitive_json",
                "source_uri",
                "json_pointer",
            )
        )
        conversation_item = next(item for item in items if item["entity_type"] == "conversation")
        message_item = next(item for item in items if item["entity_type"] == "message")
        attachment_item = next(item for item in items if item["entity_type"] == "attachment")
        assert str(conversation_item["jump"]["href"]).startswith("/conversations/")
        assert "?around=" in str(message_item["jump"]["href"])
        assert "media=" in str(attachment_item["jump"]["href"])

        activity_response = client.post(
            "/api/v1/deletions/preview",
            json={
                "entity_types": ["activity"],
                "entity_ids": {"activity": [activity_id]},
            },
        )
        assert activity_response.status_code == 201, activity_response.text
        activity_preview = activity_response.json()
        activity_selection = client.get(
            f"/api/v1/deletions/{activity_preview['operation_id']}/selection",
            params={"revision": activity_preview["selection_sha256"], "limit": 100},
        )
        assert activity_selection.status_code == 200, activity_selection.text
        activity_item = next(
            item for item in activity_selection.json()["items"] if item["entity_type"] == "activity"
        )
        assert activity_item["jump"]["href"].startswith("/timeline?around=activity:")
        assert activity_item["jump"]["href"].endswith(f"#activity-{activity_item['entity_id']}")

        cross_operation_cursor = client.get(
            f"/api/v1/deletions/{activity_preview['operation_id']}/selection",
            params={
                "revision": activity_preview["selection_sha256"],
                "limit": 2,
                "cursor": first_cursor,
            },
        )
        assert cross_operation_cursor.status_code == 422

        committed = client.post(
            f"/api/v1/deletions/{preview['operation_id']}/commit",
            json={
                "commit_token": preview["commit_token"],
                "catalog_revision": preview["catalog_revision"],
            },
        )
        assert committed.status_code == 200, committed.text
        committed_selection = client.get(
            f"/api/v1/deletions/{preview['operation_id']}/selection",
            params={"revision": preview["selection_sha256"], "limit": 100},
        )
        assert committed_selection.status_code == 200, committed_selection.text
        assert committed_selection.json()["status"] == "committed"
        assert all(item["quarantined"] for item in committed_selection.json()["items"])
        assert all(item["jump"] is None for item in committed_selection.json()["items"])

        restored = client.post(
            f"/api/v1/deletions/{preview['operation_id']}/restore",
            json={"confirmation": preview["operation_id"]},
        )
        assert restored.status_code == 200, restored.text
        unavailable = client.get(
            f"/api/v1/deletions/{preview['operation_id']}/selection",
            params={"revision": preview["selection_sha256"]},
        )
        assert unavailable.status_code == 409


def test_deletion_operation_detail_allowlists_purge_summary_without_private_state(
    deletion_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = deletion_archive
    engine.dispose()
    app = create_app(layout.root, security_disabled=True)
    private_canary = "PRIVATE_TOPIC_TERM_CANARY"
    private_path = "trash/purge-staging/private-user-path/message.json"
    public_upgrade_blockers = [
        "normalized_rows_missing",
        "outside_source_file_references_require_review",
        "normalized_rows_recreated_after_normalized_purge",
        "source_records_recreated_after_normalized_purge",
        "derived_rows_recreated_after_normalized_purge",
        "source_manifests_changed_after_normalized_purge",
        "physical_source_changed_after_normalized_purge",
        "new_source_file_references_block_workspace_upgrade",
        "new_attachment_references_block_workspace_upgrade",
        "raw_source_catalog_changed_after_normalized_purge",
        "vector_catalog_identity_mismatch",
    ]
    with TestClient(app, base_url="http://127.0.0.1") as client:
        preview_response = client.post(
            "/api/v1/deletions/preview",
            json={"entity_types": ["message"], "text": "Older chunk", "whole_field": True},
        )
        assert preview_response.status_code == 201, preview_response.text
        operation_id = preview_response.json()["operation_id"]
        with Session(app.state.engine) as session, session.begin():
            operation = session.get(DeletionOperation, operation_id)
            assert operation is not None
            operation.purge_revision = "b" * 64
            operation.purge_plan_json = {
                "scope": "workspace",
                "completed": False,
                "cleanup_pending": True,
                "cleanup_attempted": True,
                "planned_at": "2026-07-20T10:05:00+00:00",
                "completed_at": "2026-07-20T10:06:00+00:00",
                "logical_restore_state_sha256": "a" * 64,
                "logical_restore_state": {
                    "topic_jobs": [
                        {
                            "checkpoint_json": {"terms": [private_canary]},
                            "counters_json": {"private_term": private_canary},
                            "error_text": private_canary,
                        }
                    ],
                    "search_window_ids": [private_canary],
                },
                "manifest_publication": {
                    "action": "commit",
                    "phase": "prepared",
                    "affected": {"message": 1, private_canary: 99},
                    "audit_id": private_canary,
                },
                "impact": {
                    "normalized_rows": 1,
                    "workspace_files": 2,
                    "workspace_bytes": 4_096,
                    "raw_source_rewrite_error": private_canary,
                    "vector_catalog_count_mismatches": {private_canary: private_path},
                },
                "affected": {"message": 1, private_canary: 99},
                "staged_workspace_files": 2,
                "manifest_relative_path": private_path,
                "purge_token": private_canary,
                "precommit_recovery": {
                    "completed": True,
                    "completed_at": private_canary,
                    "journal_path": private_path,
                },
                "restore_rebuild_required": {
                    "required": True,
                    "reason": "intervening_hard_purge",
                    "product_kinds": [
                        "search_windows",
                        "semantic_index",
                        "nlp_hourly_rollups",
                        "conflict_aggregates",
                        "topic_clusters",
                        private_canary,
                    ],
                    "invalidated_by_count": 3,
                    "private_diagnostic": private_path,
                },
                "blockers": [*public_upgrade_blockers, private_canary],
            }
            session.add_all(
                [
                    DeletionAudit(
                        id="00000000-0000-0000-0000-000000000001",
                        operation_id=operation_id,
                        action="purge",
                        scope="workspace",
                        outcome="cleanup_pending",
                        detail_json=json.loads(json.dumps(operation.purge_plan_json)),
                    ),
                    DeletionAudit(
                        id="00000000-0000-0000-0000-000000000002",
                        operation_id=operation_id,
                        action="purge_preview",
                        scope="workspace",
                        outcome="blocked",
                        detail_json={
                            "impact": {
                                "workspace_files": 2,
                                "workspace_bytes": 4_096,
                                "raw_source_rewrite_error": private_canary,
                                "vector_catalog_count_mismatches": {private_canary: private_path},
                                "candidate_blobs": -1,
                                "search_windows": True,
                            },
                            "blockers": [
                                f"derivative_symlink:{private_path}",
                                private_canary,
                            ],
                            "purge_revision": "c" * 64,
                            "purge_token": private_canary,
                        },
                    ),
                    DeletionAudit(
                        id="00000000-0000-0000-0000-000000000003",
                        operation_id=operation_id,
                        action=private_canary,
                        scope=private_canary,
                        outcome=private_canary,
                        detail_json={private_canary: private_path},
                    ),
                    DeletionAudit(
                        id="00000000-0000-0000-0000-000000000004",
                        operation_id=operation_id,
                        action="restore",
                        scope="logical_quarantine",
                        outcome="completed",
                        detail_json={
                            "affected": {
                                "message": 1,
                                "topic_cluster_jobs_restored": 99,
                                private_canary: 99,
                            },
                            "rebuild_required": {
                                "required": True,
                                "reason": "later_quarantine",
                                "product_kinds": [
                                    "search_windows",
                                    "semantic_index",
                                    "topic_clusters",
                                    private_canary,
                                ],
                                "invalidated_by_count": 1,
                                "private_diagnostic": private_path,
                            },
                        },
                    ),
                ]
            )

        detail_response = client.get(f"/api/v1/deletions/{operation_id}")
        assert detail_response.status_code == 200, detail_response.text
        detail = detail_response.json()
        assert "purge_plan" not in detail
        assert "manifest_relative_path" not in detail
        assert detail["purge_summary"] == {
            "scope": "workspace",
            "status": {
                "operation": "previewed",
                "completed": False,
                "publication_action": "commit",
                "publication_phase": "prepared",
                "blockers": public_upgrade_blockers,
            },
            "counts": {
                "normalized_rows": 1,
                "workspace_files": 2,
                "workspace_bytes": 4_096,
                "affected_messages": 1,
                "staged_workspace_files": 2,
            },
            "digests": {
                "purge_revision": "b" * 64,
                "logical_restore_state_sha256": "a" * 64,
            },
            "timestamps": {
                "planned_at": "2026-07-20T10:05:00+00:00",
                "completed_at": "2026-07-20T10:06:00+00:00",
            },
        }
        assert detail["restore_rebuild_required"] == {
            "required": True,
            "reason": "intervening_hard_purge",
            "product_kinds": [
                "search_windows",
                "semantic_index",
                "nlp_hourly_rollups",
                "conflict_aggregates",
                "topic_clusters",
            ],
            "invalidated_by_count": 3,
        }
        public_audits = [
            {key: value for key, value in audit.items() if key != "created_at"}
            for audit in detail["audits"]
        ]
        assert public_audits == [
            {
                "id": "00000000-0000-0000-0000-000000000001",
                "action": "purge",
                "scope": "workspace",
                "outcome": "cleanup_pending",
                "detail": {
                    "counts": {
                        "normalized_rows": 1,
                        "workspace_files": 2,
                        "workspace_bytes": 4_096,
                        "affected_messages": 1,
                        "staged_workspace_files": 2,
                    },
                    "digests": {},
                    "status": {},
                },
            },
            {
                "id": "00000000-0000-0000-0000-000000000002",
                "action": "purge_preview",
                "scope": "workspace",
                "outcome": "blocked",
                "detail": {
                    "counts": {"workspace_files": 2, "workspace_bytes": 4_096},
                    "digests": {"purge_revision": "c" * 64},
                    "status": {"blockers": ["derivative_symlink"]},
                },
            },
            {
                "id": "00000000-0000-0000-0000-000000000003",
                "action": "unknown",
                "scope": None,
                "outcome": "unknown",
                "detail": {"counts": {}, "digests": {}, "status": {}},
            },
            {
                "id": "00000000-0000-0000-0000-000000000004",
                "action": "restore",
                "scope": "logical_quarantine",
                "outcome": "completed",
                "detail": {
                    "counts": {"affected_messages": 1},
                    "digests": {},
                    "status": {},
                    "restore_rebuild_required": {
                        "required": True,
                        "reason": "later_quarantine",
                        "product_kinds": [
                            "search_windows",
                            "semantic_index",
                            "topic_clusters",
                        ],
                        "invalidated_by_count": 1,
                    },
                },
            },
        ]
        serialized = json.dumps(detail, sort_keys=True)
        assert private_canary not in serialized
        assert private_path not in serialized
        assert 'logical_restore_state"' not in serialized
        assert "precommit_recovery" not in serialized
        assert "purge_token" not in serialized


def test_deletion_api_rejects_attachment_keyword_mode_instead_of_empty_preview(
    deletion_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = deletion_archive
    engine.dispose()
    app = create_app(layout.root, security_disabled=True)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.post(
            "/api/v1/deletions/preview",
            json={
                "entity_types": ["attachment"],
                "text": "voice",
                "text_mode": "keyword",
            },
        )

    assert response.status_code == 422
    assert response.json() == {
        "detail": {
            "code": "deletion_request_unsafe",
            "message": "Deletion request failed safety validation.",
        }
    }


def test_deletion_routes_require_local_session_and_csrf(
    deletion_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = deletion_archive
    engine.dispose()
    app = create_app(layout.root, access_token="z" * 32)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.get("/api/v1/deletions").status_code == 401
        assert (
            client.post(
                "/api/v1/deletions/preview",
                json={"text": "Older chunk"},
            ).status_code
            == 403
        )


def test_deletion_cli_requires_apply_and_commits_the_exact_preview(
    deletion_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = deletion_archive
    runner = CliRunner()
    preview_result = runner.invoke(
        cli_app,
        [
            "delete",
            "preview",
            "--archive-root",
            str(layout.root),
            "--entity",
            "message",
            "--text",
            "Older chunk",
            "--whole-field",
        ],
    )
    assert preview_result.exit_code == 0, preview_result.output
    preview = json.loads(preview_result.stdout)
    no_apply = runner.invoke(
        cli_app,
        [
            "delete",
            "commit",
            preview["operation_id"],
            "--commit-token",
            preview["commit_token"],
            "--catalog-revision",
            preview["catalog_revision"],
            "--archive-root",
            str(layout.root),
        ],
    )
    assert no_apply.exit_code == 2
    committed = runner.invoke(
        cli_app,
        [
            "delete",
            "commit",
            preview["operation_id"],
            "--commit-token",
            preview["commit_token"],
            "--catalog-revision",
            preview["catalog_revision"],
            "--archive-root",
            str(layout.root),
            "--apply",
        ],
    )
    assert committed.exit_code == 0, committed.output
    assert json.loads(committed.stdout)["status"] == "committed"
    with Session(engine) as session:
        message = session.scalar(select(Message).where(Message.content_display == "Older chunk"))
        assert message is not None and message.deleted_at is not None


def test_normalized_purge_preserves_workspace_upgrade_then_workspace_purge_finishes(
    deletion_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = deletion_archive
    message = _message(engine, "Older chunk")
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
    assert purge.executable
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
    assert Path(preview.manifest).is_file()
    assert _fts_count(engine, message.pk, '"older"') == 0
    with Session(engine) as session:
        assert session.scalar(select(Message).where(Message.id == message.id)) is None
        operation = session.get(DeletionOperation, preview.operation_id)
        assert operation is not None
        assert operation.status == "normalized_purged"
        assert operation.selector_json != {"purged": True}
        assert operation.purge_plan_json["workspace_upgrade_state"]["relative_path"]

    workspace_plan = plan_hard_purge(
        layout,
        engine,
        preview.operation_id,
        scope="workspace",
    )
    assert workspace_plan.executable
    workspace_result = execute_hard_purge(
        layout,
        engine,
        preview.operation_id,
        scope="workspace",
        purge_token=workspace_plan.purge_token,
        purge_revision=workspace_plan.purge_revision,
        confirmation=preview.operation_id,
    )
    assert workspace_result.status == "purged"
    assert not Path(preview.manifest).exists()
    with Session(engine) as session:
        operation = session.get(DeletionOperation, preview.operation_id)
        assert operation is not None
        assert operation.status == "purged"
        assert operation.selector_json == {"purged": True}
    with pytest.raises(InvalidDeletionTokenError):
        restore_deletion(layout, engine, preview.operation_id, confirmation="not-the-id")


def test_precommit_purge_crash_restores_journaled_files_and_lance_identities(
    deletion_archive: tuple[ArchiveLayout, Engine], monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, engine = deletion_archive
    message = _message(engine, "Older chunk")
    table_name, selected_window_id, retained_window_id = _add_fixture_vector_window(
        layout, engine, message
    )
    analytics_file = layout.analytics_cache / "precommit-private.json"
    analytics_file.write_text('{"private": "must return after rollback"}\n')
    vector_before = _tree_digest_snapshot(layout.vectors)
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
    assert purge.executable

    original_delete = deletion_service._delete_normalized_rows
    original_restore = deletion_service._restore_and_verify_staged_files
    original_remove_tree = deletion_service._remove_tree_exact

    def fail_before_sqlite_commit(*_args: object, **_kwargs: object) -> dict[str, int]:
        raise RuntimeError("simulated process death before SQLite commit")

    # Suppress the in-process exception cleanup to leave the exact state an abrupt process death
    # would leave: SQLite rolls back to committed and the complete original Lance tree remains in
    # durable staging while the live vector path is intentionally absent.
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
            preview.operation_id,
            scope="normalized",
            purge_token=purge.purge_token,
            purge_revision=purge.purge_revision,
            confirmation=preview.operation_id,
        )

    staging = layout.trash / "purge-staging" / preview.operation_id
    assert (staging / deletion_service.PURGE_ROLLBACK_JOURNAL_NAME).is_file()
    assert Path(preview.manifest).is_file()
    assert not analytics_file.exists()
    assert not layout.vectors.exists()
    staged_vectors = staging / layout.vectors.relative_to(layout.root)
    staged_table = lancedb.connect(str(staged_vectors)).open_table(table_name)
    staged_ids = {
        str(row["id"]) for row in staged_table.search().select(["id"]).limit(10).to_list()
    }
    assert staged_ids == {selected_window_id, retained_window_id}
    with Session(engine) as session:
        operation = session.get(DeletionOperation, preview.operation_id)
        assert operation is not None and operation.status == "committed"
        assert session.get(Message, message.pk) is not None

    monkeypatch.setattr(deletion_service, "_delete_normalized_rows", original_delete)
    monkeypatch.setattr(deletion_service, "_restore_and_verify_staged_files", original_restore)
    monkeypatch.setattr(deletion_service, "_remove_tree_exact", original_remove_tree)
    recovered_app = create_app(layout.root, security_disabled=True)
    recovered_app.state.engine.dispose()

    assert not staging.exists()
    assert Path(preview.manifest).is_file()
    assert analytics_file.read_text() == '{"private": "must return after rollback"}\n'
    assert _tree_digest_snapshot(layout.vectors) == vector_before
    restored_table = lancedb.connect(str(layout.vectors)).open_table(table_name)
    restored_ids = {
        str(row["id"]) for row in restored_table.search().select(["id"]).limit(10).to_list()
    }
    assert restored_ids == {selected_window_id, retained_window_id}
    with Session(engine) as session:
        operation = session.get(DeletionOperation, preview.operation_id)
        assert operation is not None and operation.status == "committed"
        assert operation.error_text is None
        assert operation.purge_plan_json["precommit_recovery"]["completed"] is True

    # The original authorization remains revision-bound and usable after exact rollback.
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
    assert Path(preview.manifest).is_file()


@pytest.mark.parametrize("failure_point", ["before_cleanup", "during_compaction"])
def test_hard_purge_cleanup_is_durable_and_recoverable(
    deletion_archive: tuple[ArchiveLayout, Engine],
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    layout, engine = deletion_archive
    message = _message(engine, "Older chunk")
    table_name, _selected_window_id, retained_window_id = _add_fixture_vector_window(
        layout, engine, message
    )
    layout.analytics_cache.mkdir(parents=True, exist_ok=True)
    (layout.analytics_cache / "private-derived.json").write_text("private fixture")
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
    assert purge.executable

    if failure_point == "before_cleanup":
        original = deletion_service._finalize_pending_hard_purge_locked

        def fail_before_cleanup(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("before cleanup fixture")

        monkeypatch.setattr(
            deletion_service, "_finalize_pending_hard_purge_locked", fail_before_cleanup
        )
    else:
        original = deletion_service._compact_catalog_after_purge

        def fail_compaction(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("compaction fixture")

        monkeypatch.setattr(deletion_service, "_compact_catalog_after_purge", fail_compaction)

    with pytest.raises(RuntimeError):
        execute_hard_purge(
            layout,
            engine,
            preview.operation_id,
            scope="normalized",
            purge_token=purge.purge_token,
            purge_revision=purge.purge_revision,
            confirmation=preview.operation_id,
        )

    staging = layout.trash / "purge-staging" / preview.operation_id
    with Session(engine) as session:
        operation = session.get(DeletionOperation, preview.operation_id)
        assert operation is not None and operation.status == "cleanup_pending"
        assert session.get(Message, message.pk) is None
    assert not layout.vectors.exists()
    assert staging.exists() is (failure_point == "before_cleanup")
    assert not layout.analytics_cache.exists()

    if failure_point == "before_cleanup":
        monkeypatch.setattr(deletion_service, "_finalize_pending_hard_purge_locked", original)
        recovered_app = create_app(layout.root, security_disabled=True)
        recovered_app.state.engine.dispose()
    else:
        monkeypatch.setattr(deletion_service, "_compact_catalog_after_purge", original)
        recovered = recover_pending_hard_purges(layout, engine)
        assert [result.status for result in recovered] == ["normalized_purged"]
    assert not staging.exists()
    assert Path(preview.manifest).is_file()
    with Session(engine) as session:
        operation = session.get(DeletionOperation, preview.operation_id)
        assert operation is not None and operation.status == "normalized_purged"
        config = session.scalar(
            select(SemanticIndexConfig).where(SemanticIndexConfig.vector_table == table_name)
        )
        assert config is not None
        assert config.active is False
        assert config.status == "stale"
        assert config.window_count == 0
        retained = session.get(SearchWindow, retained_window_id)
        assert retained is not None and retained.embedding_config_hash is None
        assert session.scalar(select(func.count(SemanticWindowEmbedding.window_id))) == 0
    assert not layout.vectors.exists() or not any(layout.vectors.iterdir())


def test_pending_hard_purge_rejects_a_recreated_lance_store(
    deletion_archive: tuple[ArchiveLayout, Engine], monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, engine = deletion_archive
    message = _message(engine, "Older chunk")
    table_name, _selected_window_id, _retained_window_id = _add_fixture_vector_window(
        layout, engine, message
    )
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
    assert purge.executable

    original_finalize = deletion_service._finalize_pending_hard_purge_locked

    def fail_before_cleanup(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("before cleanup identity fixture")

    monkeypatch.setattr(
        deletion_service, "_finalize_pending_hard_purge_locked", fail_before_cleanup
    )
    with pytest.raises(RuntimeError, match="identity fixture"):
        execute_hard_purge(
            layout,
            engine,
            preview.operation_id,
            scope="normalized",
            purge_token=purge.purge_token,
            purge_revision=purge.purge_revision,
            confirmation=preview.operation_id,
        )
    monkeypatch.setattr(deletion_service, "_finalize_pending_hard_purge_locked", original_finalize)

    assert not layout.vectors.exists()
    wrong_window_id = "6" * 64
    table = lancedb.connect(str(layout.vectors)).create_table(
        table_name,
        data=[{"id": wrong_window_id, "vector": [0.0, 0.0, 1.0]}],
        mode="overwrite",
    )
    assert table.count_rows() == 1

    with pytest.raises(HardPurgeBlockedError, match="unexpected residual files"):
        recover_pending_hard_purges(layout, engine)

    staging = layout.trash / "purge-staging" / preview.operation_id
    assert staging.is_dir()
    with Session(engine) as session:
        operation = session.get(DeletionOperation, preview.operation_id)
        assert operation is not None and operation.status == "cleanup_pending"
        assert operation.purge_plan_json["vector_store_rewritten"] is True
        assert operation.purge_plan_json["affected"]["semantic_store_discarded"] == 1
    actual_ids = {str(row["id"]) for row in table.search().select(["id"]).limit(10).to_list()}
    assert actual_ids == {wrong_window_id}


def test_logical_deletion_sanitizes_topic_cluster_results(
    deletion_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = deletion_archive
    with Session(engine) as session, session.begin():
        session.add(
            Job(
                id="topic-secret-fixture",
                kind="topic_clusters",
                status="completed",
                stage="ready",
                progress_current=1,
                progress_total=1,
                checkpoint_json={},
                counters_json={"result": {"clusters": [{"terms": ["private phrase"]}]}},
            )
        )
    write_analytics_cache(
        layout,
        engine,
        namespace="deletion-fixture",
        parameters={},
        payload={"private": "Older chunk"},
    )
    assert read_analytics_cache(layout, engine, namespace="deletion-fixture", parameters={}) == {
        "private": "Older chunk"
    }
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
    with Session(engine) as session:
        job = session.get(Job, "topic-secret-fixture")
        assert job is not None and job.status == "stale"
        assert job.counters_json["invalidated"] is True
        assert "private phrase" not in json.dumps(job.counters_json)
    assert read_analytics_cache(layout, engine, namespace="deletion-fixture", parameters={}) is None
    with pytest.raises(RuntimeError, match="logically deleted"):
        topic_cluster_service._assert_no_logical_quarantine(engine)


def test_corrected_conversation_blocks_hard_purge(
    deletion_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = deletion_archive
    with Session(engine) as session, session.begin():
        segment = session.scalar(select(ConversationSegment).order_by(ConversationSegment.id))
        assert segment is not None
        correction_id = "correction-purge-fixture"
        session.add(
            ConversationCorrection(
                id=correction_id,
                action="merge",
                status="applied",
                reason="fixture",
                request_json={},
                before_json={},
                after_json={},
                catalog_revision="1" * 64,
                result_revision="2" * 64,
                effect_json={},
            )
        )
        session.add(
            ConversationSegmentOverride(
                segment_id=segment.id,
                base_conversation_id=segment.conversation_id,
                conversation_id=segment.conversation_id,
                correction_id=correction_id,
            )
        )
        conversation_id = segment.conversation_id
    preview = preview_deletion(
        layout,
        engine,
        DeletionSelector(
            entity_types=("conversation",),
            entity_ids={"conversation": (conversation_id,)},
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
    assert purge.executable is False
    assert "dependent_rows_belong_to_another_operation" in purge.blockers
    assert purge.impact["dependent_rows_outside_operation"] >= 1


def test_unregistered_vector_table_is_diagnostic_before_full_store_discard(
    deletion_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = deletion_archive
    message = _message(engine, "Older chunk")
    registered_table, _selected, _retained = _add_fixture_vector_window(layout, engine, message)
    vector_database = lancedb.connect(str(layout.vectors))
    vector_database.open_table(registered_table).add(
        [{"id": "unregistered-row", "vector": [1.0, 1.0, 1.0]}]
    )
    vector_database.create_table(
        "orphan_private_vectors",
        data=[{"id": "orphan-row", "vector": [0.0, 0.0, 1.0]}],
        mode="overwrite",
    )
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
    assert purge.executable is True
    assert purge.blockers == ()
    assert purge.impact["semantic_store_will_be_discarded"] is True
    assert purge.impact["unregistered_vector_tables"] == ["orphan_private_vectors"]
    assert purge.impact["vector_catalog_count_mismatches"][0]["lance_rows"] == 3

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
    assert not layout.vectors.exists() or not any(layout.vectors.iterdir())
