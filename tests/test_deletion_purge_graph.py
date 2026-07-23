from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from meta_archive.database import create_catalog_engine, run_migrations
from meta_archive.models import (
    AttachmentOccurrence,
    AttachmentTextExtraction,
    AudioTranscription,
    AudioTranscriptSegment,
    Blob,
    ConflictCalibration,
    ConflictLabel,
    Conversation,
    ConversationCorrection,
    ConversationCorrectionAudit,
    ConversationCorrectionSegment,
    ConversationSegment,
    DeletionItem,
    DeletionOperation,
    Job,
    MediaDerivative,
    Participant,
    RagAudit,
    RelevanceFeedback,
    SearchWindow,
    SegmentParticipant,
)
from meta_archive.paths import ArchiveLayout
from meta_archive.services.deletion_purge_graph import (
    BlobOwnership,
    ConversationCorrectionPurgeInventory,
    build_derived_purge_graph,
    classify_blob_ownership,
    gc_conversation_candidate_participants,
    invalidate_overlapping_deletion_operations,
    inventory_conversation_corrections,
    inventory_deletion_overlaps,
    inventory_derivative_tree,
    purge_window_feedback_and_calibrations,
    scrub_blob_linked_job_checkpoints,
    scrub_reverted_conversation_corrections,
)


@pytest.fixture
def graph_catalog(tmp_path: Path) -> Iterator[tuple[ArchiveLayout, Engine]]:
    layout = ArchiveLayout(tmp_path / "meta-backup-purge-graph")
    layout.root.mkdir()
    layout.ensure_runtime_directories()
    run_migrations(layout.database)
    engine = create_catalog_engine(layout.database)
    yield layout, engine
    engine.dispose()


def _blob(sha256: str) -> Blob:
    return Blob(
        sha256=sha256,
        size_bytes=1,
        media_type="application/octet-stream",
        extension=".bin",
        relative_path=f"blobs/sha256/{sha256[:2]}/{sha256}",
        state="materialized",
    )


def _attachment(
    identifier: str,
    blob_sha256: str,
    *,
    deleted_at: datetime | None = None,
    ordinal: int = 0,
) -> AttachmentOccurrence:
    return AttachmentOccurrence(
        id=identifier,
        blob_sha256=blob_sha256,
        kind="file",
        ordinal=ordinal,
        filename=f"{identifier}.bin",
        size_bytes=1,
        availability="resolved",
        metadata_json={},
        deleted_at=deleted_at,
    )


def _conversation(identifier: str) -> Conversation:
    return Conversation(
        id=identifier,
        conversation_key=f"key:{identifier}",
        title_raw="Fixture",
        title_display="Fixture",
        title_search="fixture",
        conversation_type="direct",
        primary_category="inbox",
    )


def _segment(identifier: str, conversation_id: str) -> ConversationSegment:
    return ConversationSegment(
        id=identifier,
        conversation_id=conversation_id,
        thread_path=f"inbox/{identifier}",
        title_raw="Fixture",
        title_display="Fixture",
        title_search="fixture",
        source_category="inbox",
        metadata_json={},
    )


def _window(
    identifier: str,
    *,
    conversation_id: str | None = None,
    segment_id: str | None = None,
    attachment_occurrence_id: str | None = None,
    audio_segment_id: str | None = None,
    attachment_extraction_id: str | None = None,
    first_message_id: str | None = None,
) -> SearchWindow:
    return SearchWindow(
        id=identifier,
        window_version="fixture-v1",
        entity_type="message_window",
        conversation_id=conversation_id,
        segment_id=segment_id,
        attachment_occurrence_id=attachment_occurrence_id,
        audio_segment_id=audio_segment_id,
        attachment_extraction_id=attachment_extraction_id,
        first_message_id=first_message_id,
        last_message_id=first_message_id,
        turn_count=1,
        approx_token_count=1,
        text_display="Fixture text",
        text_search="fixture text",
        participants_search="fixture",
        source_ids_json={},
        build_generation="fixture-generation",
    )


def _operation(
    identifier: str,
    status: str,
    *,
    manifest: str | None = None,
) -> DeletionOperation:
    return DeletionOperation(
        id=identifier,
        status=status,
        selector_json={"text": "private fixture"},
        impact_json={"records": 1},
        catalog_revision="1" * 64,
        selection_sha256="2" * 64,
        manifest_relative_path=manifest,
        item_count=1,
        purge_plan_json={},
    )


def _item(operation_id: str, entity_id: str, *, entity_type: str = "message") -> DeletionItem:
    return DeletionItem(
        operation_id=operation_id,
        entity_type=entity_type,
        entity_id=entity_id,
        reason="direct",
        snapshot_sha256="3" * 64,
    )


def test_blob_ownership_preserves_live_and_recoverably_quarantined_references(
    graph_catalog: tuple[ArchiveLayout, Engine],
) -> None:
    _layout, engine = graph_catalog
    exclusive = "a" * 64
    live_shared = "b" * 64
    recoverable_shared = "c" * 64
    selected_ids = {"selected-exclusive", "selected-live", "selected-recoverable"}
    with Session(engine) as session, session.begin():
        session.add_all([_blob(exclusive), _blob(live_shared), _blob(recoverable_shared)])
        session.flush()
        session.add_all(
            [
                _attachment("selected-exclusive", exclusive),
                _attachment("selected-live", live_shared),
                _attachment("selected-recoverable", recoverable_shared),
                _attachment("retained-live", live_shared, ordinal=1),
                _attachment(
                    "retained-quarantine",
                    recoverable_shared,
                    deleted_at=datetime(2025, 1, 1, tzinfo=UTC),
                    ordinal=2,
                ),
            ]
        )
    with Session(engine) as session:
        ownership = classify_blob_ownership(session, selected_ids)

    assert ownership == BlobOwnership(
        selected_blob_ids=frozenset({exclusive, live_shared, recoverable_shared}),
        live_retained_blob_ids=frozenset({live_shared}),
        recoverable_retained_blob_ids=frozenset({recoverable_shared}),
    )
    assert ownership.shared_blob_ids == frozenset({live_shared, recoverable_shared})
    assert ownership.physically_exclusive_blob_ids == frozenset({exclusive})


def test_derived_graph_covers_fk_cascade_windows_and_physical_blob_products(
    graph_catalog: tuple[ArchiveLayout, Engine],
) -> None:
    _layout, engine = graph_catalog
    conversation_id = "conversation-selected"
    segment_id = "segment-selected"
    participant_id = "participant-selected"
    attachment_id = "attachment-selected"
    blob_sha256 = "d" * 64
    with Session(engine) as session, session.begin():
        session.add(_conversation(conversation_id))
        session.add(_segment(segment_id, conversation_id))
        session.add(
            Participant(
                id=participant_id,
                name_raw="Private Person",
                name_display="Private Person",
                name_search="private person",
                is_owner=False,
            )
        )
        session.flush()
        session.add(
            SegmentParticipant(
                segment_id=segment_id,
                participant_id=participant_id,
                ordinal=0,
            )
        )
        session.add(_blob(blob_sha256))
        session.flush()
        session.add(_attachment(attachment_id, blob_sha256))
        session.add(
            AudioTranscription(
                id="transcription-selected",
                blob_sha256=blob_sha256,
                pipeline_version="fixture-v1",
                status="completed",
                model_name="fixture",
                model_revision="fixture",
                device="cpu",
                compute_type="int8",
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
                id="audio-segment-selected",
                transcription_id="transcription-selected",
                ordinal=0,
                start_ms=0,
                end_ms=1,
                text_display="private transcript",
                text_search="private transcript",
                words_json={},
            )
        )
        session.add(
            AttachmentTextExtraction(
                id="extraction-selected",
                blob_sha256=blob_sha256,
                pipeline_version="fixture-v1",
                status="completed",
                extractor_name="fixture",
                extractor_revision="fixture",
                text_display="private extraction",
                text_search="private extraction",
                character_count=18,
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
        session.add(
            MediaDerivative(
                id="derivative-selected",
                blob_sha256=blob_sha256,
                kind="image_thumbnail",
                pipeline_version="fixture-v1",
                status="completed",
                relative_path="catalog/thumbnails/fixture.webp",
                media_type="image/webp",
                attempt_count=1,
                metadata_json={},
            )
        )
        session.flush()
        session.add_all(
            [
                _window(
                    "window-conversation",
                    conversation_id=conversation_id,
                    segment_id=segment_id,
                ),
                _window("window-endpoint", first_message_id="message-selected"),
                _window("window-attachment", attachment_occurrence_id=attachment_id),
                _window("window-audio", audio_segment_id="audio-segment-selected"),
                _window(
                    "window-extraction",
                    attachment_extraction_id="extraction-selected",
                ),
                _window("window-retained"),
            ]
        )

    with Session(engine) as session:
        graph = build_derived_purge_graph(
            session,
            message_ids={"message-selected"},
            conversation_ids={conversation_id},
            attachment_ids={attachment_id},
            physical_blob_ids={blob_sha256},
        )

    assert graph.conversation_segment_ids == frozenset({segment_id})
    assert graph.conversation_participant_candidate_ids == frozenset({participant_id})
    assert graph.logical_search_window_ids == frozenset(
        {"window-conversation", "window-endpoint", "window-attachment"}
    )
    assert graph.physical_search_window_ids == frozenset({"window-audio", "window-extraction"})
    assert graph.search_window_ids == frozenset(
        {
            "window-conversation",
            "window-endpoint",
            "window-attachment",
            "window-audio",
            "window-extraction",
        }
    )
    assert graph.audio_transcription_ids == frozenset({"transcription-selected"})
    assert graph.audio_segment_ids == frozenset({"audio-segment-selected"})
    assert graph.attachment_extraction_ids == frozenset({"extraction-selected"})
    assert graph.media_derivative_ids == frozenset({"derivative-selected"})


def test_derivative_inventory_finds_legacy_catalogued_and_canonical_orphan_files(
    graph_catalog: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = graph_catalog
    selected_blob = "a1" + "a" * 62
    retained_blob = "b2" + "b" * 62
    legacy = layout.thumbnails / "legacy" / "selected.webp"
    orphan = layout.thumbnails / "media-v1" / selected_blob[:2] / selected_blob / "orphan.webp"
    retained = layout.thumbnails / "media-v1" / retained_blob[:2] / retained_blob / "keep.webp"
    missing = layout.thumbnails / "legacy" / "missing.webp"
    for path in (legacy, orphan, retained):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    layout.derivative_manifest.write_text("{}\n")
    (layout.thumbnails / ".build.lock").write_text("")
    unknown = layout.thumbnails / "unattributed.bin"
    unknown.write_bytes(b"unknown")
    with Session(engine) as session, session.begin():
        session.add_all([_blob(selected_blob), _blob(retained_blob)])
        session.add_all(
            [
                MediaDerivative(
                    id="legacy-selected",
                    blob_sha256=selected_blob,
                    kind="image_thumbnail",
                    pipeline_version="fixture",
                    status="completed",
                    relative_path=legacy.relative_to(layout.root).as_posix(),
                    media_type="image/webp",
                    attempt_count=1,
                    metadata_json={},
                ),
                MediaDerivative(
                    id="missing-selected",
                    blob_sha256=selected_blob,
                    kind="video_poster",
                    pipeline_version="fixture",
                    status="completed",
                    relative_path=missing.relative_to(layout.root).as_posix(),
                    media_type="image/webp",
                    attempt_count=1,
                    metadata_json={},
                ),
            ]
        )

    with Session(engine) as session:
        blocked = inventory_derivative_tree(layout, session, {selected_blob})
    assert legacy in blocked.selected_catalogued_paths
    assert orphan in blocked.selected_orphan_paths
    assert retained not in blocked.selected_paths
    assert missing in blocked.missing_selected_catalogued_paths
    assert layout.derivative_manifest in blocked.global_metadata_paths
    assert layout.thumbnails / ".build.lock" in blocked.global_metadata_paths
    assert any(value.startswith("unattributed_derivative_file:") for value in blocked.blockers)

    unknown.unlink()
    with Session(engine) as session:
        clean = inventory_derivative_tree(layout, session, {selected_blob})
    assert clean.executable
    assert clean.selected_paths == frozenset({legacy, orphan})


def test_derivative_inventory_blocks_symlinks_without_following_them(
    graph_catalog: tuple[ArchiveLayout, Engine],
    tmp_path: Path,
) -> None:
    layout, engine = graph_catalog
    selected_blob = "c3" + "c" * 62
    external = tmp_path / "external-private.webp"
    external.write_bytes(b"must remain")
    linked = layout.thumbnails / "media-v1" / selected_blob[:2] / selected_blob / "linked.webp"
    linked.parent.mkdir(parents=True)
    linked.symlink_to(external)
    with Session(engine) as session:
        inventory = inventory_derivative_tree(layout, session, {selected_blob})

    assert not inventory.executable
    assert any(value.startswith("derivative_symlink:") for value in inventory.blockers)
    assert linked not in inventory.selected_paths
    assert external.read_bytes() == b"must remain"


def test_feedback_rag_audits_and_conflict_calibration_are_physically_scrubbed(
    graph_catalog: tuple[ArchiveLayout, Engine],
) -> None:
    _layout, engine = graph_catalog
    selected_window = "window-selected"
    retained_window = "window-retained"
    invalidated_at = datetime(2026, 1, 2, tzinfo=UTC)
    with Session(engine) as session, session.begin():
        session.add_all([_window(selected_window), _window(retained_window)])
        session.flush()
        session.add_all(
            [
                RelevanceFeedback(
                    id="feedback-selected",
                    query_sha256="1" * 64,
                    query_text="fixture",
                    search_mode="semantic",
                    filter_scope_sha256="2" * 64,
                    filter_scope_json={},
                    result_id=f"window:{selected_window}",
                    label="relevant",
                    evaluation_split="test",
                ),
                RelevanceFeedback(
                    id="feedback-retained",
                    query_sha256="1" * 64,
                    query_text="fixture",
                    search_mode="semantic",
                    filter_scope_sha256="2" * 64,
                    filter_scope_json={},
                    result_id=f"window:{retained_window}",
                    label="relevant",
                    evaluation_split="test",
                ),
                ConflictLabel(
                    id="label-selected",
                    window_id=selected_window,
                    label="direct_conflict",
                    evaluation_split="test",
                ),
                ConflictCalibration(
                    id="calibration-selected",
                    pipeline_version="fixture",
                    model_name="fixture",
                    model_revision="fixture",
                    status="calibrated",
                    sample_count=500,
                    direct_conflict_precision=0.9,
                    macro_f1=0.8,
                    aggregation_enabled=True,
                    metrics_json={
                        "held_out_predictions": [
                            {"window_id": selected_window, "expected": "direct_conflict"}
                        ]
                    },
                    provenance_json={"unique_window_count": 500},
                    warnings_json={},
                    completed_at=invalidated_at,
                ),
                RagAudit(
                    id="rag-selected",
                    status="completed",
                    model="fixture",
                    query_sha256="3" * 64,
                    query_text="",
                    search_mode="semantic",
                    retrieved_ids_json={"ids": [f"window:{selected_window}"]},
                    citation_ids_json={"ids": [f"window:{selected_window}"]},
                    excerpt_hashes_json={f"window:{selected_window}": "4" * 64},
                    usage_json={},
                ),
                RagAudit(
                    id="rag-retained",
                    status="completed",
                    model="fixture",
                    query_sha256="3" * 64,
                    query_text="",
                    search_mode="semantic",
                    retrieved_ids_json={"ids": [f"window:{retained_window}"]},
                    citation_ids_json={"ids": []},
                    excerpt_hashes_json={f"window:{retained_window}": "5" * 64},
                    usage_json={},
                ),
            ]
        )

    with Session(engine) as session, session.begin():
        result = purge_window_feedback_and_calibrations(
            session,
            {selected_window},
            invalidated_at=invalidated_at,
        )
    assert result.relevance_feedback_deleted == 1
    assert result.rag_audits_deleted == 1
    assert result.selected_conflict_labels == 1
    assert result.conflict_calibrations_scrubbed == 1
    with Session(engine) as session:
        assert session.get(RelevanceFeedback, "feedback-selected") is None
        assert session.get(RelevanceFeedback, "feedback-retained") is not None
        assert session.get(RagAudit, "rag-selected") is None
        assert session.get(RagAudit, "rag-retained") is not None
        calibration = session.get(ConflictCalibration, "calibration-selected")
        assert calibration is not None
        assert calibration.status == "stale"
        assert calibration.aggregation_enabled is False
        assert calibration.sample_count == 0
        assert calibration.direct_conflict_precision is None
        assert calibration.macro_f1 is None
        assert selected_window not in json.dumps(calibration.metrics_json)
        assert selected_window not in json.dumps(calibration.provenance_json)


def test_participant_gc_is_limited_to_candidates_and_rechecks_live_links(
    graph_catalog: tuple[ArchiveLayout, Engine],
) -> None:
    _layout, engine = graph_catalog
    with Session(engine) as session, session.begin():
        session.add(_conversation("conversation-retained"))
        session.add(_segment("segment-retained", "conversation-retained"))
        session.add_all(
            [
                Participant(
                    id=value,
                    name_raw=value,
                    name_display=value,
                    name_search=value,
                    is_owner=False,
                )
                for value in ("candidate-orphan", "candidate-linked", "unrelated-orphan")
            ]
        )
        session.flush()
        session.add(
            SegmentParticipant(
                segment_id="segment-retained",
                participant_id="candidate-linked",
                ordinal=0,
            )
        )

    with Session(engine) as session, session.begin():
        deleted = gc_conversation_candidate_participants(
            session,
            {"candidate-orphan", "candidate-linked"},
        )
    assert deleted == 1
    with Session(engine) as session:
        assert session.get(Participant, "candidate-orphan") is None
        assert session.get(Participant, "candidate-linked") is not None
        assert session.get(Participant, "unrelated-orphan") is not None


def test_committed_deletion_overlap_blocks_and_preview_restore_overlap_is_inventoried(
    graph_catalog: tuple[ArchiveLayout, Engine],
) -> None:
    _layout, engine = graph_catalog
    with Session(engine) as session, session.begin():
        session.add_all(
            [
                _operation("current", "committed"),
                _operation("other-committed", "committed", manifest="trash/committed.json"),
                _operation("other-preview", "previewed", manifest="trash/preview.json"),
                _operation("other-restored", "restored", manifest="trash/restored.json"),
            ]
        )
        session.flush()
        session.add_all(
            [
                _item("current", "message-selected"),
                _item("other-committed", "message-selected"),
                _item("other-preview", "message-selected"),
                _item("other-restored", "message-selected"),
            ]
        )

    with Session(engine) as session:
        inventory = inventory_deletion_overlaps(
            session,
            current_operation_id="current",
            selected_entity_ids={"message": {"message-selected"}},
        )
    assert inventory.blocking_operation_ids == frozenset({"other-committed"})
    assert inventory.invalidatable_operation_ids == frozenset({"other-preview", "other-restored"})
    assert inventory.invalidatable_manifest_paths == frozenset(
        {"trash/preview.json", "trash/restored.json"}
    )
    assert inventory.invalidatable_item_count == 2
    assert inventory.blockers == ("selected_rows_belong_to_another_committed_deletion",)
    with Session(engine) as session, session.begin(), pytest.raises(ValueError, match="Blocking"):
        invalidate_overlapping_deletion_operations(session, inventory)


def test_overlapping_preview_and_restored_operations_are_scrubbed_after_manifests_stage(
    graph_catalog: tuple[ArchiveLayout, Engine],
) -> None:
    _layout, engine = graph_catalog
    invalidated_at = datetime(2026, 2, 3, tzinfo=UTC)
    with Session(engine) as session, session.begin():
        session.add_all(
            [
                _operation("current", "committed"),
                _operation("preview", "previewed", manifest="trash/preview.json"),
                _operation("restored", "restored", manifest="trash/restored.json"),
                _operation("unrelated", "previewed", manifest="trash/unrelated.json"),
            ]
        )
        session.flush()
        session.add_all(
            [
                _item("current", "message-selected"),
                _item("preview", "message-selected"),
                _item("restored", "message-selected"),
                _item("unrelated", "message-retained"),
            ]
        )
    with Session(engine) as session, session.begin():
        inventory = inventory_deletion_overlaps(
            session,
            current_operation_id="current",
            selected_entity_ids={"message": {"message-selected"}},
        )
        count = invalidate_overlapping_deletion_operations(
            session,
            inventory,
            invalidated_at=invalidated_at,
        )
    assert count == 2
    with Session(engine) as session:
        for operation_id in ("preview", "restored"):
            operation = session.get(DeletionOperation, operation_id)
            assert operation is not None
            assert operation.status == "invalidated"
            assert operation.selector_json["invalidated"] is True
            assert operation.manifest_relative_path is None
            assert operation.item_count == 0
            assert (
                session.scalar(
                    select(DeletionItem.id).where(DeletionItem.operation_id == operation_id)
                )
                is None
            )
        unrelated = session.get(DeletionOperation, "unrelated")
        assert unrelated is not None and unrelated.status == "previewed"
        assert (
            session.scalar(select(DeletionItem.id).where(DeletionItem.operation_id == "unrelated"))
            is not None
        )


def test_reverted_correction_history_is_scrubbed_but_active_history_blocks(
    graph_catalog: tuple[ArchiveLayout, Engine],
) -> None:
    _layout, engine = graph_catalog
    conversation_id = "conversation-selected"
    segment_id = "segment-selected"
    invalidated_at = datetime(2026, 3, 4, tzinfo=UTC)
    with Session(engine) as session, session.begin():
        for correction_id, status in (("reverted", "reverted"), ("active", "active")):
            session.add(
                ConversationCorrection(
                    id=correction_id,
                    action="split",
                    status=status,
                    reason="private correction reason",
                    request_json={"conversation_id": conversation_id, "title": "Private title"},
                    before_json={"conversations": [{"id": conversation_id}]},
                    after_json={},
                    catalog_revision="1" * 64,
                    result_revision="2" * 64,
                    effect_json={},
                )
            )
        session.flush()
        session.add_all(
            [
                ConversationCorrectionSegment(
                    correction_id="reverted",
                    segment_id=segment_id,
                    thread_path="inbox/private_thread",
                    before_conversation_id=conversation_id,
                    after_conversation_id="other-conversation",
                ),
                ConversationCorrectionSegment(
                    correction_id="active",
                    segment_id=segment_id,
                    thread_path="inbox/private_thread_active",
                    before_conversation_id=conversation_id,
                    after_conversation_id="other-conversation",
                ),
                ConversationCorrectionAudit(
                    id="audit-reverted",
                    correction_id="reverted",
                    action="revert",
                    detail_json={"thread_path": "inbox/private_thread"},
                ),
            ]
        )

    with Session(engine) as session:
        blocked = inventory_conversation_corrections(
            session,
            conversation_ids={conversation_id},
            segment_ids={segment_id},
        )
    assert blocked.blocking_correction_ids == frozenset({"active"})
    assert blocked.scrub_correction_ids == frozenset({"reverted"})
    with Session(engine) as session, session.begin(), pytest.raises(ValueError, match="Active"):
        scrub_reverted_conversation_corrections(session, blocked)

    scrub_only = ConversationCorrectionPurgeInventory(
        blocking_correction_ids=frozenset(),
        scrub_correction_ids=frozenset({"reverted"}),
    )
    with Session(engine) as session, session.begin():
        count = scrub_reverted_conversation_corrections(
            session,
            scrub_only,
            invalidated_at=invalidated_at,
        )
    assert count == 1
    with Session(engine) as session:
        correction = session.get(ConversationCorrection, "reverted")
        assert correction is not None
        assert correction.status == "purged"
        assert correction.reason is None
        assert correction.request_json == {}
        assert correction.before_json == {}
        assert correction.after_json == {}
        assert (
            session.scalar(
                select(ConversationCorrectionSegment.id).where(
                    ConversationCorrectionSegment.correction_id == "reverted"
                )
            )
            is None
        )
        assert session.get(ConversationCorrectionAudit, "audit-reverted") is None
        active = session.get(ConversationCorrection, "active")
        assert active is not None and active.request_json["title"] == "Private title"


def test_blob_linked_job_checkpoint_is_scrubbed_without_touching_other_jobs(
    graph_catalog: tuple[ArchiveLayout, Engine],
) -> None:
    _layout, engine = graph_catalog
    selected_blob = "e" * 64
    retained_blob = "f" * 64
    invalidated_at = datetime(2026, 4, 5, tzinfo=UTC)
    with Session(engine) as session, session.begin():
        session.add_all(
            [
                Job(
                    id="selected-job",
                    kind="audio_transcription",
                    status="completed",
                    checkpoint_json={
                        "last_blob_sha256": selected_blob,
                        "nested": {"source": selected_blob},
                    },
                    counters_json={"transcribed": 1},
                ),
                Job(
                    id="retained-job",
                    kind="audio_transcription",
                    status="completed",
                    checkpoint_json={"last_blob_sha256": retained_blob},
                    counters_json={"transcribed": 1},
                ),
            ]
        )

    with Session(engine) as session, session.begin():
        result = scrub_blob_linked_job_checkpoints(
            session,
            {selected_blob},
            invalidated_at=invalidated_at,
        )
    assert result.scrubbed_job_ids == frozenset({"selected-job"})
    with Session(engine) as session:
        selected_job = session.get(Job, "selected-job")
        retained_job = session.get(Job, "retained-job")
        assert selected_job is not None and retained_job is not None
        assert selected_job.status == "stale"
        assert selected_blob not in json.dumps(selected_job.checkpoint_json)
        assert selected_blob not in json.dumps(selected_job.counters_json)
        assert retained_job.status == "completed"
        assert retained_job.checkpoint_json["last_blob_sha256"] == retained_blob
