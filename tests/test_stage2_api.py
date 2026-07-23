from __future__ import annotations

import json
import shutil
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event, func, select, update
from sqlalchemy.orm import Session

from meta_archive.api import media_routes
from meta_archive.api.aggregate_cache import (
    CATALOG_CHANGED_RETRY_DETAIL,
    RevisionBoundAggregateCache,
)
from meta_archive.api.main import create_app
from meta_archive.database import create_catalog_engine, run_migrations
from meta_archive.models import (
    AttachmentOccurrence,
    ConflictCalibration,
    Conversation,
    ConversationCorrection,
    ConversationCorrectionAudit,
    ConversationCorrectionSegment,
    ConversationSegment,
    ConversationSegmentOverride,
    MediaDerivative,
    Message,
    SearchWindow,
    SemanticIndexConfig,
    SourceChunk,
    SourceFile,
)
from meta_archive.paths import ArchiveLayout
from meta_archive.services import conversation_corrections as correction_service
from meta_archive.services.consolidation import build_blob_store
from meta_archive.services.inventory import scan_archive
from meta_archive.services.library import build_filesystem_library
from meta_archive.services.media_derivatives import PIPELINE_VERSION
from meta_archive.services.message_importer import import_messages

FIXTURES = Path(__file__).parent / "fixtures" / "messages"


@pytest.fixture
def stage2_client(tmp_path: Path) -> Iterator[TestClient]:
    layout = ArchiveLayout(tmp_path / "meta-backup-stage2")
    shutil.copytree(FIXTURES, layout.root)
    layout.ensure_runtime_directories()
    run_migrations(layout.database)
    engine = create_catalog_engine(layout.database)
    scan_archive(layout, engine, hash_files=True)
    import_messages(layout, engine)
    build_blob_store(layout, engine)
    # Add a second occurrence of one catalogued blob so both gallery modes exercise a real
    # occurrence-versus-unique distinction without requiring any private export data.
    with Session(engine) as session, session.begin():
        original = session.scalar(
            select(AttachmentOccurrence)
            .where(
                AttachmentOccurrence.blob_sha256.is_not(None),
                AttachmentOccurrence.message_id.is_not(None),
            )
            .order_by(AttachmentOccurrence.id)
            .limit(1)
        )
        assert original is not None
        session.add(
            AttachmentOccurrence(
                id="f" * 64,
                message_id=original.message_id,
                conversation_id=original.conversation_id,
                source_file_id=original.source_file_id,
                resolved_source_file_id=original.resolved_source_file_id,
                blob_sha256=original.blob_sha256,
                kind=original.kind,
                ordinal=99,
                source_uri=original.source_uri,
                filename=original.filename,
                media_type=original.media_type,
                size_bytes=original.size_bytes,
                timestamp_ms=original.timestamp_ms,
                availability=original.availability,
                metadata_json={"fixture_duplicate_occurrence": True},
            )
        )
        # Reproduce Facebook's real voice-message metadata: the source bucket says audio while
        # the MP4 container is reported as video. Source kind must remain authoritative.
        session.execute(
            update(AttachmentOccurrence)
            .where(AttachmentOccurrence.kind.in_(("audio_files", "audio")))
            .values(media_type="video/mp4")
        )
    engine.dispose()

    app = create_app(layout.root, security_disabled=True)
    with TestClient(app) as client:
        yield client


def _conversation(client: TestClient, title: str) -> dict[str, object]:
    response = client.get("/api/v1/conversations", params={"q": title, "limit": 100})
    assert response.status_code == 200
    matches = [item for item in response.json()["items"] if item["title"] == title]
    assert len(matches) == 1
    return matches[0]


def test_stage2_fixture_resolved_attachments_have_materialized_blobs(
    stage2_client: TestClient,
) -> None:
    response = stage2_client.get(
        "/api/v1/media",
        params={"availability": "resolved", "view": "occurrences", "limit": 100},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["page"]["next_cursor"] is None
    blob_hrefs = {item["blob_href"] for item in payload["items"]}
    assert blob_hrefs
    assert None not in blob_hrefs
    for href in blob_hrefs:
        blob = stage2_client.head(href)
        assert blob.status_code == 200
        assert int(blob.headers["content-length"]) >= 0


def test_conversation_dashboard_filters_sorts_and_uses_scoped_keyset_cursors(
    stage2_client: TestClient,
) -> None:
    first = stage2_client.get(
        "/api/v1/conversations",
        params={"sort": "messages", "direction": "desc", "limit": 2},
    )
    assert first.status_code == 200
    payload = first.json()
    assert payload["page"]["total"] == 7
    assert len(payload["items"]) == 2
    assert [item["message_counts"]["total"] for item in payload["items"]] == sorted(
        [item["message_counts"]["total"] for item in payload["items"]], reverse=True
    )
    assert payload["page"]["next_cursor"]

    second = stage2_client.get(
        "/api/v1/conversations",
        params={
            "sort": "messages",
            "direction": "desc",
            "limit": 2,
            "cursor": payload["page"]["next_cursor"],
        },
    )
    assert second.status_code == 200
    assert {item["id"] for item in second.json()["items"]}.isdisjoint(
        {item["id"] for item in payload["items"]}
    )

    mismatched = stage2_client.get(
        "/api/v1/conversations",
        params={
            "sort": "messages",
            "direction": "desc",
            "limit": 2,
            "conversation_type": "direct",
            "cursor": payload["page"]["next_cursor"],
        },
    )
    assert mismatched.status_code == 422

    direct = stage2_client.get(
        "/api/v1/conversations",
        params={"conversation_type": "direct", "source_category": "e2ee_cutover"},
    ).json()
    assert [item["title"] for item in direct["items"]] == ["Alex Example"]
    assert direct["items"][0]["source_categories"] == ["archived_threads", "e2ee_cutover"]
    assert direct["items"][0]["segment_count"] == 2

    groups = stage2_client.get(
        "/api/v1/conversations", params={"conversation_type": "group"}
    ).json()
    assert groups["page"]["total"] == 2
    assert {item["type"] for item in groups["items"]} == {"group"}

    for sort in (
        "title",
        "type",
        "source_category",
        "participants",
        "first_activity",
        "last_activity",
        "messages",
        "sent",
        "received",
        "attachments",
        "unique_attachments",
        "attachment_bytes",
        "active_days",
        "completeness",
        "photos",
        "videos",
        "audio",
    ):
        sorted_response = stage2_client.get(
            "/api/v1/conversations",
            params={"sort": sort, "direction": "asc", "limit": 100},
        )
        assert sorted_response.status_code == 200, (sort, sorted_response.text)
        assert sorted_response.json()["page"]["total"] == 7

    by_photos = stage2_client.get(
        "/api/v1/conversations",
        params={"sort": "photos", "direction": "desc", "limit": 100},
    ).json()
    assert [item["media_counts"]["photo"] for item in by_photos["items"]] == sorted(
        [item["media_counts"]["photo"] for item in by_photos["items"]], reverse=True
    )
    assert all(
        set(item["media_counts"])
        == {"photo", "video", "audio", "gif", "sticker", "document", "other"}
        for item in by_photos["items"]
    )
    with_photos = stage2_client.get(
        "/api/v1/conversations", params={"min_photos": 1, "max_photos": 50}
    )
    assert with_photos.status_code == 200
    assert with_photos.json()["items"]
    assert all(item["media_counts"]["photo"] >= 1 for item in with_photos.json()["items"])
    with_message_range = stage2_client.get(
        "/api/v1/conversations", params={"min_messages": 2, "max_messages": 7}
    )
    assert with_message_range.status_code == 200
    assert with_message_range.json()["items"]
    assert all(
        2 <= item["message_counts"]["total"] <= 7 for item in with_message_range.json()["items"]
    )
    alex = _conversation(stage2_client, "Alex Example")
    assert alex["media_counts"]["audio"] == 2
    assert alex["media_counts"]["video"] == 1
    assert (
        stage2_client.get(
            "/api/v1/conversations", params={"min_audio": 2, "max_audio": 1}
        ).status_code
        == 422
    )
    assert (
        stage2_client.get(
            "/api/v1/conversations", params={"min_messages": 8, "max_messages": 7}
        ).status_code
        == 422
    )


def test_conversation_totals_are_cached_by_scope_and_invalidated_by_catalog_mutation(
    stage2_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loads: Counter[str] = Counter()
    original = RevisionBoundAggregateCache.get_or_compute

    def tracked_get_or_compute(
        self: RevisionBoundAggregateCache,
        *,
        namespace: str,
        scope: str,
        revision: str,
        revision_is_current: Callable[[], bool],
        compute: Callable[[], object],
    ) -> object:
        def tracked_compute() -> object:
            loads[namespace] += 1
            return compute()

        return original(
            self,
            namespace=namespace,
            scope=scope,
            revision=revision,
            revision_is_current=revision_is_current,
            compute=tracked_compute,
        )

    monkeypatch.setattr(RevisionBoundAggregateCache, "get_or_compute", tracked_get_or_compute)

    first = stage2_client.get(
        "/api/v1/conversations",
        params={"sort": "messages", "direction": "desc", "limit": 2},
    )
    assert first.status_code == 200, first.text
    payload = first.json()
    second = stage2_client.get(
        "/api/v1/conversations",
        params={
            "sort": "messages",
            "direction": "desc",
            "limit": 2,
            "cursor": payload["page"]["next_cursor"],
        },
    )
    assert second.status_code == 200, second.text
    reordered = stage2_client.get(
        "/api/v1/conversations",
        params={"sort": "title", "direction": "asc", "limit": 2},
    )
    assert reordered.status_code == 200, reordered.text
    assert loads["conversation-list-total-v1"] == 1

    conversation = max(payload["items"], key=lambda item: item["message_counts"]["total"])
    conversation_id = conversation["id"]
    first_reader = stage2_client.get(
        f"/api/v1/conversations/{conversation_id}/messages",
        params={"start": "earliest", "limit": 2},
    )
    assert first_reader.status_code == 200, first_reader.text
    reader_payload = first_reader.json()
    assert reader_payload["page"]["after_cursor"]
    second_reader = stage2_client.get(
        f"/api/v1/conversations/{conversation_id}/messages",
        params={"limit": 2, "cursor": reader_payload["page"]["after_cursor"]},
    )
    assert second_reader.status_code == 200, second_reader.text
    assert loads["conversation-message-total-v1"] == 1

    first_search = stage2_client.get(
        f"/api/v1/conversations/{conversation_id}/message-search",
        params={"q": "message", "limit": 1},
    )
    repeated_search = stage2_client.get(
        f"/api/v1/conversations/{conversation_id}/message-search",
        params={"q": "message", "limit": 1},
    )
    assert first_search.status_code == repeated_search.status_code == 200
    assert first_search.json()["page"]["total"] == repeated_search.json()["page"]["total"]
    assert loads["conversation-message-search-total-v1"] == 1

    before_total = int(reader_payload["page"]["total"])
    with Session(stage2_client.app.state.engine) as session, session.begin():
        message = session.scalar(
            select(Message)
            .where(
                Message.conversation_id == conversation_id,
                Message.deleted_at.is_(None),
            )
            .order_by(Message.pk)
            .limit(1)
        )
        assert message is not None
        message.deleted_at = datetime.now(UTC)

    after_mutation = stage2_client.get(
        f"/api/v1/conversations/{conversation_id}/messages",
        params={"start": "earliest", "limit": 2},
    )
    assert after_mutation.status_code == 200, after_mutation.text
    assert after_mutation.json()["page"]["total"] == before_total - 1
    assert loads["conversation-message-total-v1"] == 2

    refreshed_list = stage2_client.get("/api/v1/conversations", params={"limit": 2})
    assert refreshed_list.status_code == 200, refreshed_list.text
    assert refreshed_list.json()["page"]["total"] == payload["page"]["total"]
    assert loads["conversation-list-total-v1"] == 2


def test_conversation_detail_preserves_segments_and_duplicate_participant_occurrences(
    stage2_client: TestClient,
) -> None:
    alex = _conversation(stage2_client, "Alex Example")
    response = stage2_client.get(f"/api/v1/conversations/{alex['id']}")
    assert response.status_code == 200
    payload = response.json()
    assert payload["conversation"]["message_counts"] == {"total": 7, "sent": 2, "received": 5}
    assert payload["conversation"]["media_counts"] == alex["media_counts"]
    assert {segment["source_category"] for segment in payload["segments"]} == {
        "archived_threads",
        "e2ee_cutover",
    }
    assert all(
        "name_raw" not in participant
        for segment in payload["segments"]
        for participant in segment["participants"]
    )

    group_response = stage2_client.get(
        "/api/v1/conversations", params={"conversation_type": "group", "limit": 10}
    ).json()
    group_ids = [item["id"] for item in group_response["items"]]
    group_details = [
        stage2_client.get(f"/api/v1/conversations/{conversation_id}").json()
        for conversation_id in group_ids
    ]
    duplicate_names = [
        [participant["name"] for participant in segment["participants"]]
        for detail in group_details
        for segment in detail["segments"]
        if sum(participant["name"] == "Repeated Label" for participant in segment["participants"])
        == 2
    ]
    assert duplicate_names == [["Archive Owner", "Repeated Label", "Repeated Label"]]


def test_direct_history_corrections_are_reversible_audited_and_reimport_safe(
    stage2_client: TestClient,
) -> None:
    alex = _conversation(stage2_client, "Alex Example")
    original_id = str(alex["id"])
    original_detail = stage2_client.get(f"/api/v1/conversations/{original_id}").json()
    assert original_detail["conversation"]["identity"] == {
        "status": "provisional",
        "basis": "exact_counterpart_display_label",
        "corrected_segment_count": 0,
        "can_correct": True,
    }
    assert len(original_detail["segments"]) == 2
    selected_segment = original_detail["segments"][0]
    selected_segment_id = selected_segment["id"]
    selected_thread_path = selected_segment["thread_path"]
    original_message_count = original_detail["conversation"]["message_counts"]["total"]

    history = stage2_client.get(
        "/api/v1/conversation-corrections", params={"conversation_id": original_id}
    )
    assert history.status_code == 200
    initial_revision = history.json()["identity_revision"]
    assert history.json()["items"] == []

    split = stage2_client.post(
        "/api/v1/conversation-corrections/split",
        json={
            "conversation_id": original_id,
            "segment_ids": [selected_segment_id],
            "title": "Alex archived history",
            "identity_revision": initial_revision,
            "reason": "The source segment belongs in a separate direct history.",
        },
    )
    assert split.status_code == 201, split.text
    split_payload = split.json()
    split_id = split_payload["correction_id"]
    split_conversation_id = split_payload["redirect_conversation_id"]
    assert split_payload["effect"]["segments"] == 1
    assert split_payload["effect"]["messages"] > 0

    old_after_split = stage2_client.get(f"/api/v1/conversations/{original_id}").json()
    new_after_split = stage2_client.get(f"/api/v1/conversations/{split_conversation_id}").json()
    assert len(old_after_split["segments"]) == len(new_after_split["segments"]) == 1
    assert new_after_split["segments"][0]["id"] == selected_segment_id
    assert new_after_split["segments"][0]["thread_path"] == selected_thread_path
    assert new_after_split["segments"][0]["assignment"] == {
        "status": "corrected",
        "correction_id": split_id,
        "base_conversation_id": original_id,
    }
    assert (
        old_after_split["conversation"]["message_counts"]["total"]
        + new_after_split["conversation"]["message_counts"]["total"]
        == original_message_count
    )

    layout = stage2_client.app.state.layout
    engine = stage2_client.app.state.engine
    import_messages(layout, engine, force=True)
    after_reimport = stage2_client.get(f"/api/v1/conversations/{split_conversation_id}").json()
    assert [row["id"] for row in after_reimport["segments"]] == [selected_segment_id]
    assert after_reimport["segments"][0]["thread_path"] == selected_thread_path

    stale_merge = stage2_client.post(
        "/api/v1/conversation-corrections/merge",
        json={
            "conversation_ids": [original_id, split_conversation_id],
            "target_conversation_id": original_id,
            "identity_revision": initial_revision,
        },
    )
    assert stale_merge.status_code == 409

    refreshed = stage2_client.get("/api/v1/conversation-corrections").json()
    merge = stage2_client.post(
        "/api/v1/conversation-corrections/merge",
        json={
            "conversation_ids": [original_id, split_conversation_id],
            "target_conversation_id": original_id,
            "identity_revision": refreshed["identity_revision"],
            "reason": "These two corrected histories are the same person after review.",
        },
    )
    assert merge.status_code == 201, merge.text
    merge_payload = merge.json()
    merge_id = merge_payload["correction_id"]
    assert stage2_client.get(f"/api/v1/conversations/{split_conversation_id}").status_code == 404
    merged_detail = stage2_client.get(f"/api/v1/conversations/{original_id}").json()
    assert len(merged_detail["segments"]) == 2
    assert merged_detail["conversation"]["message_counts"]["total"] == original_message_count

    out_of_order = stage2_client.post(
        f"/api/v1/conversation-corrections/{split_id}/revert",
        json={"identity_revision": merge_payload["identity_revision"]},
    )
    assert out_of_order.status_code == 409
    assert "later correction" in out_of_order.json()["detail"].lower()

    undo_merge = stage2_client.post(
        f"/api/v1/conversation-corrections/{merge_id}/revert",
        json={"identity_revision": merge_payload["identity_revision"]},
    )
    assert undo_merge.status_code == 200, undo_merge.text
    restored_split = stage2_client.get(f"/api/v1/conversations/{split_conversation_id}").json()
    assert [row["id"] for row in restored_split["segments"]] == [selected_segment_id]
    assert restored_split["segments"][0]["assignment"]["correction_id"] == split_id

    undo_split = stage2_client.post(
        f"/api/v1/conversation-corrections/{split_id}/revert",
        json={"identity_revision": undo_merge.json()["identity_revision"]},
    )
    assert undo_split.status_code == 200, undo_split.text
    final_detail = stage2_client.get(f"/api/v1/conversations/{original_id}").json()
    assert len(final_detail["segments"]) == 2
    assert final_detail["conversation"]["message_counts"]["total"] == original_message_count
    assert final_detail["conversation"]["identity"]["status"] == "provisional"
    assert stage2_client.get(f"/api/v1/conversations/{split_conversation_id}").status_code == 404

    with Session(engine) as session:
        assert session.scalar(select(func.count(ConversationSegmentOverride.segment_id))) == 0
        corrections = list(
            session.scalars(select(ConversationCorrection).order_by(ConversationCorrection.id))
        )
        assert {row.id: row.status for row in corrections} == {
            split_id: "reverted",
            merge_id: "reverted",
        }
        for correction in corrections:
            assert correction.effect_json["audit_id"]
            assert correction.effect_json["revert"]["audit_id"]
        assert session.scalar(select(func.count(ConversationCorrectionAudit.id))) == 4
        audit_segment = session.scalar(
            select(ConversationCorrectionSegment).where(
                ConversationCorrectionSegment.correction_id == split_id
            )
        )
        source_segment = session.get(ConversationSegment, selected_segment_id)
        assert audit_segment is not None and source_segment is not None
        assert audit_segment.thread_path == source_segment.thread_path == selected_thread_path


def test_corrections_reject_group_histories_and_splitting_every_segment(
    stage2_client: TestClient,
) -> None:
    revision = stage2_client.get("/api/v1/conversation-corrections").json()["identity_revision"]
    groups = stage2_client.get(
        "/api/v1/conversations", params={"conversation_type": "group", "limit": 10}
    ).json()["items"]
    assert len(groups) >= 2
    invalid_merge = stage2_client.post(
        "/api/v1/conversation-corrections/merge",
        json={
            "conversation_ids": [groups[0]["id"], groups[1]["id"]],
            "target_conversation_id": groups[0]["id"],
            "identity_revision": revision,
        },
    )
    assert invalid_merge.status_code == 422
    assert "direct histories" in invalid_merge.json()["detail"]

    alex = _conversation(stage2_client, "Alex Example")
    detail = stage2_client.get(f"/api/v1/conversations/{alex['id']}").json()
    invalid_split = stage2_client.post(
        "/api/v1/conversation-corrections/split",
        json={
            "conversation_id": alex["id"],
            "segment_ids": [segment["id"] for segment in detail["segments"]],
            "title": "Invalid complete move",
            "identity_revision": revision,
        },
    )
    assert invalid_split.status_code == 422
    assert "leave at least one" in invalid_split.json()["detail"]


def test_correction_rebuilds_library_and_invalidates_conversation_derived_state(
    stage2_client: TestClient,
) -> None:
    layout = stage2_client.app.state.layout
    engine = stage2_client.app.state.engine
    alex = _conversation(stage2_client, "Alex Example")
    conversation_id = str(alex["id"])
    detail = stage2_client.get(f"/api/v1/conversations/{conversation_id}").json()
    segment_id = detail["segments"][0]["id"]
    config_id = "9" * 64
    window_id = "8" * 64
    calibration_id = "7" * 64
    with Session(engine) as session, session.begin():
        message = session.scalar(
            select(Message).where(Message.segment_id == segment_id).order_by(Message.id)
        )
        assert message is not None
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
                vector_table="stage2_correction_vectors",
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
                conversation_id=conversation_id,
                segment_id=segment_id,
                first_message_id=message.id,
                last_message_id=message.id,
                first_timestamp_ms=message.timestamp_ms,
                last_timestamp_ms=message.timestamp_ms,
                turn_count=1,
                approx_token_count=4,
                text_display="Fixture correction window.",
                text_search="fixture correction window",
                source_category="fixture",
                participants_search=message.sender_search,
                source_ids_json={},
                build_generation="fixture-generation",
                embedding_config_hash=config_id,
                embedding_updated_at=datetime.now(UTC),
            )
        )
        session.add(
            ConflictCalibration(
                id=calibration_id,
                pipeline_version="fixture-v1",
                model_name="fixture-model",
                model_revision="fixture-revision",
                status="ready",
                sample_count=500,
                direct_conflict_precision=0.8,
                macro_f1=0.8,
                aggregation_enabled=True,
                metrics_json={},
                provenance_json={},
                warnings_json={},
                completed_at=datetime.now(UTC),
            )
        )
    build_filesystem_library(layout, engine)
    revision = stage2_client.get("/api/v1/conversation-corrections").json()["identity_revision"]
    split = stage2_client.post(
        "/api/v1/conversation-corrections/split",
        json={
            "conversation_id": conversation_id,
            "segment_ids": [segment_id],
            "title": "Derived state correction fixture",
            "identity_revision": revision,
        },
    )
    assert split.status_code == 201, split.text
    result = split.json()
    destination_id = result["redirect_conversation_id"]
    assert result["effect"]["library_rebuilt"] is True
    assert result["effect"]["semantic_configs_invalidated"] == 1
    assert result["effect"]["conflict_calibrations_invalidated"] == 1

    with Session(engine) as session:
        config = session.get(SemanticIndexConfig, config_id)
        window = session.get(SearchWindow, window_id)
        calibration = session.get(ConflictCalibration, calibration_id)
        assert config is not None and window is not None and calibration is not None
        assert (config.active, config.status) == (False, "stale")
        assert config.audit_json["stale_correction_id"] == result["correction_id"]
        assert window.conversation_id == destination_id
        assert (calibration.aggregation_enabled, calibration.status) == (False, "stale")

    manifests = [
        json.loads(line)
        for line in (layout.library / "conversations" / "manifest.ndjson")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    by_id = {row["conversation_id"]: row for row in manifests}
    assert conversation_id in by_id and destination_id in by_id
    assert (
        by_id[conversation_id]["message_count"] + by_id[destination_id]["message_count"]
        == alex["message_counts"]["total"]
    )


def test_correction_failure_restores_catalog_and_existing_library(
    stage2_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = stage2_client.app.state.layout
    engine = stage2_client.app.state.engine
    alex = _conversation(stage2_client, "Alex Example")
    conversation_id = str(alex["id"])
    detail = stage2_client.get(f"/api/v1/conversations/{conversation_id}").json()
    segment_id = detail["segments"][0]["id"]
    build_filesystem_library(layout, engine)
    manifest = layout.library / "conversations" / "manifest.ndjson"
    original_manifest = manifest.read_bytes()
    with Session(engine) as session:
        original_revision = correction_service.conversation_identity_revision(session)

    def fail_library(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("injected correction library failure")

    monkeypatch.setattr(correction_service, "build_filesystem_library", fail_library)
    with pytest.raises(RuntimeError, match="injected correction library failure"):
        correction_service.split_direct_history(
            layout,
            engine,
            conversation_id=conversation_id,
            segment_ids=[segment_id],
            title="This correction must roll back",
            expected_revision=original_revision,
        )

    assert manifest.read_bytes() == original_manifest
    with Session(engine) as session:
        assert correction_service.conversation_identity_revision(session) == original_revision
        assert session.scalar(select(func.count(ConversationCorrection.id))) == 0
        assert session.scalar(select(func.count(ConversationSegmentOverride.segment_id))) == 0
        source = session.get(ConversationSegment, segment_id)
        assert source is not None and source.conversation_id == conversation_id


def test_virtual_message_reader_pages_both_directions_and_keeps_stable_anchors(
    stage2_client: TestClient,
) -> None:
    alex = _conversation(stage2_client, "Alex Example")
    conversation_id = str(alex["id"])
    earliest = stage2_client.get(
        f"/api/v1/conversations/{conversation_id}/messages",
        params={"start": "earliest", "limit": 2},
    )
    assert earliest.status_code == 200
    first_page = earliest.json()
    assert first_page["page"]["total"] == 7
    assert first_page["page"]["before_cursor"] is None
    assert first_page["page"]["after_cursor"]
    assert [item["timestamp_ms"] for item in first_page["items"]] == sorted(
        item["timestamp_ms"] for item in first_page["items"]
    )
    assert all(
        "sensitive" not in item and "sensitive_json" not in item for item in first_page["items"]
    )

    next_page = stage2_client.get(
        f"/api/v1/conversations/{conversation_id}/messages",
        params={"limit": 2, "cursor": first_page["page"]["after_cursor"]},
    )
    assert next_page.status_code == 200
    next_payload = next_page.json()
    assert next_payload["page"]["before_cursor"]
    assert {item["id"] for item in first_page["items"]}.isdisjoint(
        {item["id"] for item in next_payload["items"]}
    )

    back = stage2_client.get(
        f"/api/v1/conversations/{conversation_id}/messages",
        params={"limit": 2, "cursor": next_payload["page"]["before_cursor"]},
    )
    assert [item["id"] for item in back.json()["items"]] == [
        item["id"] for item in first_page["items"]
    ]

    all_ids = [item["id"] for item in first_page["items"]]
    continuation = first_page["page"]["after_cursor"]
    while continuation:
        page = stage2_client.get(
            f"/api/v1/conversations/{conversation_id}/messages",
            params={"limit": 2, "cursor": continuation},
        ).json()
        all_ids.extend(item["id"] for item in page["items"])
        continuation = page["page"]["after_cursor"]
    assert len(all_ids) == len(set(all_ids)) == first_page["page"]["total"]

    complete_reader = stage2_client.get(
        f"/api/v1/conversations/{conversation_id}/messages",
        params={"start": "earliest", "limit": 100},
    ).json()
    kinds = {item["kind"] for item in complete_reader["items"]}
    assert {"payment", "unsent", "unavailable"} <= kinds
    assert any(item["reactions"] for item in complete_reader["items"])
    attachment_items = [
        attachment for item in complete_reader["items"] for attachment in item["attachments"]
    ]
    assert attachment_items
    assert any(attachment["availability"] == "missing" for attachment in attachment_items)
    assert all("source_uri" not in attachment for attachment in attachment_items)

    all_conversations = stage2_client.get("/api/v1/conversations", params={"limit": 100}).json()[
        "items"
    ]
    all_messages = [
        item
        for conversation in all_conversations
        for item in stage2_client.get(
            f"/api/v1/conversations/{conversation['id']}/messages",
            params={"start": "earliest", "limit": 100},
        ).json()["items"]
    ]
    call = next(item for item in all_messages if item["kind"] == "call")
    assert call["call"] == {"duration_seconds": 12, "missed": False}

    focus_id = next_payload["items"][0]["id"]
    around = stage2_client.get(
        f"/api/v1/conversations/{conversation_id}/messages",
        params={"around": focus_id, "limit": 5},
    )
    assert around.status_code == 200
    around_payload = around.json()
    assert around_payload["focus"]["message_id"] == focus_id
    assert focus_id in {item["id"] for item in around_payload["items"]}
    focused = next(item for item in around_payload["items"] if item["id"] == focus_id)
    assert focused["anchor"]["fragment"] == f"message-{focus_id}"

    invalid_position = stage2_client.get(
        f"/api/v1/conversations/{conversation_id}/messages",
        params={"around": focus_id, "after": focus_id},
    )
    assert invalid_position.status_code == 422

    explicit_before = stage2_client.get(
        f"/api/v1/conversations/{conversation_id}/messages",
        params={"before": focus_id, "limit": 2},
    ).json()
    explicit_after = stage2_client.get(
        f"/api/v1/conversations/{conversation_id}/messages",
        params={"after": focus_id, "limit": 2},
    ).json()
    assert all(item["timestamp_ms"] <= focused["timestamp_ms"] for item in explicit_before["items"])
    assert all(item["timestamp_ms"] >= focused["timestamp_ms"] for item in explicit_after["items"])

    latest = stage2_client.get(
        f"/api/v1/conversations/{conversation_id}/messages", params={"limit": 1}
    ).json()
    date_jump = stage2_client.get(
        f"/api/v1/conversations/{conversation_id}/messages/date-jump",
        params={"at_ms": latest["items"][0]["timestamp_ms"] + 1},
    )
    assert date_jump.status_code == 200
    assert date_jump.json()["jump"]["message_id"] == latest["items"][0]["id"]


def test_message_attachment_payload_includes_ready_media_derivatives(
    stage2_client: TestClient,
) -> None:
    engine = stage2_client.app.state.engine
    with Session(engine) as session, session.begin():
        attachment = session.scalar(
            select(AttachmentOccurrence)
            .where(
                AttachmentOccurrence.kind.in_(("audio_files", "audio")),
                AttachmentOccurrence.message_id.is_not(None),
                AttachmentOccurrence.conversation_id.is_not(None),
                AttachmentOccurrence.blob_sha256.is_not(None),
                AttachmentOccurrence.deleted_at.is_(None),
            )
            .order_by(AttachmentOccurrence.id)
            .limit(1)
        )
        assert attachment is not None
        assert attachment.message_id is not None
        assert attachment.conversation_id is not None
        assert attachment.blob_sha256 is not None
        message_id = attachment.message_id
        conversation_id = attachment.conversation_id
        attachment_id = attachment.id
        derivative_id = "1" * 64
        session.add(
            MediaDerivative(
                id=derivative_id,
                blob_sha256=attachment.blob_sha256,
                kind="audio_waveform",
                pipeline_version=PIPELINE_VERSION,
                status="ready",
                relative_path="artifacts/derivatives/fixture-waveform.png",
                media_type="image/png",
                size_bytes=456,
                output_sha256="2" * 64,
                duration_ms=12_345,
                width=800,
                height=180,
                attempt_count=1,
                metadata_json={"runner": "stage2-fixture"},
            )
        )

    response = stage2_client.get(
        f"/api/v1/conversations/{conversation_id}/messages",
        params={"around": message_id, "limit": 10},
    )
    assert response.status_code == 200
    payload = response.json()
    retained_attachment = next(
        candidate
        for message in payload["items"]
        for candidate in message["attachments"]
        if candidate["id"] == attachment_id
    )
    assert retained_attachment["derivatives"] == [
        {
            "id": derivative_id,
            "kind": "audio_waveform",
            "status": "ready",
            "media_type": "image/png",
            "size_bytes": 456,
            "duration_ms": 12_345,
            "width": 800,
            "height": 180,
            "href": f"/api/v1/derivatives/{derivative_id}",
        }
    ]


def test_local_thread_search_excludes_sensitive_fields_until_explicit_advanced_reveal(
    stage2_client: TestClient,
) -> None:
    alex = _conversation(stage2_client, "Alex Example")
    conversation_id = str(alex["id"])
    search = stage2_client.get(
        f"/api/v1/conversations/{conversation_id}/message-search",
        params={"q": "Français ABC"},
    )
    assert search.status_code == 200
    payload = search.json()
    assert payload["page"]["total"] == 1
    message = payload["items"][0]
    assert message["content"] == "Français ABC"
    assert message["matched_fields"] == ["content"]
    assert message["has_sensitive_ip"] is True
    assert message["has_payment_details"] is True
    assert "sensitive" not in message

    hidden_ip = stage2_client.get(
        f"/api/v1/conversations/{conversation_id}/message-search",
        params={"q": "192.0.2.10"},
    )
    assert hidden_ip.status_code == 200
    assert hidden_ip.json()["page"]["total"] == 0

    advanced = stage2_client.get(
        f"/api/v1/conversations/{conversation_id}/messages/{message['id']}/advanced"
    )
    assert advanced.status_code == 200
    assert advanced.json()["sensitive"] == {
        "ip": "192.0.2.10",
        "payment_info": {"amount": "fixture-only"},
    }


def test_media_dashboard_reconciles_occurrence_and_unique_views_and_filters(
    stage2_client: TestClient,
) -> None:
    occurrences = stage2_client.get(
        "/api/v1/media",
        params={"view": "occurrences", "sort": "size", "direction": "desc", "limit": 3},
    )
    assert occurrences.status_code == 200, occurrences.text
    payload = occurrences.json()
    assert payload["page"]["total"] == payload["summary"]["occurrences"] == 11
    assert payload["summary"]["unique_files"] < payload["summary"]["occurrences"]
    assert payload["summary"]["unique_bytes"] < payload["summary"]["occurrence_bytes"]
    sizes = [item["size_bytes"] for item in payload["items"] if item["size_bytes"] is not None]
    assert sizes == sorted(sizes, reverse=True)
    assert payload["page"]["next_cursor"]

    audio = stage2_client.get(
        "/api/v1/media", params={"view": "occurrences", "type": "audio", "limit": 100}
    )
    assert audio.status_code == 200
    assert audio.json()["page"]["total"] == 2
    assert all(item["kind"] in {"audio_files", "audio"} for item in audio.json()["items"])
    assert all(item["media_type"] == "video/mp4" for item in audio.json()["items"])
    focused_audio = audio.json()["items"][0]
    exact_occurrence = stage2_client.get(
        "/api/v1/media",
        params={"view": "occurrences", "occurrence": focused_audio["id"]},
    )
    assert exact_occurrence.status_code == 200, exact_occurrence.text
    assert exact_occurrence.json()["page"]["total"] == 1
    assert [item["id"] for item in exact_occurrence.json()["items"]] == [focused_audio["id"]]
    assert f"media={focused_audio['id']}" in exact_occurrence.json()["items"][0]["jump"]["href"]

    next_page = stage2_client.get(
        "/api/v1/media",
        params={
            "view": "occurrences",
            "sort": "size",
            "direction": "desc",
            "limit": 3,
            "cursor": payload["page"]["next_cursor"],
        },
    )
    assert next_page.status_code == 200
    assert {item["id"] for item in next_page.json()["items"]}.isdisjoint(
        {item["id"] for item in payload["items"]}
    )

    changed_filter = stage2_client.get(
        "/api/v1/media",
        params={
            "view": "occurrences",
            "type": "photo",
            "sort": "size",
            "direction": "desc",
            "limit": 3,
            "cursor": payload["page"]["next_cursor"],
        },
    )
    assert changed_filter.status_code == 422

    unique = stage2_client.get(
        "/api/v1/media", params={"view": "unique", "sort": "occurrences", "limit": 100}
    )
    assert unique.status_code == 200, unique.text
    unique_payload = unique.json()
    assert unique_payload["page"]["total"] == payload["summary"]["unique_files"]
    repeated = [item for item in unique_payload["items"] if item["occurrence_count"] > 1]
    assert repeated
    assert repeated[0]["occurrences_href"]

    unique_ids: list[str] = []
    unique_cursor = None
    while True:
        params = {
            "view": "unique",
            "sort": "date",
            "direction": "desc",
            "limit": 2,
        }
        if unique_cursor:
            params["cursor"] = unique_cursor
        unique_page = stage2_client.get("/api/v1/media", params=params)
        assert unique_page.status_code == 200, unique_page.text
        unique_ids.extend(item["view_key"] for item in unique_page.json()["items"])
        unique_cursor = unique_page.json()["page"]["next_cursor"]
        if not unique_cursor:
            break
    assert len(unique_ids) == len(set(unique_ids)) == unique_payload["page"]["total"]

    blob_occurrences = stage2_client.get(repeated[0]["occurrences_href"])
    assert blob_occurrences.status_code == 200
    assert blob_occurrences.json()["page"]["total"] == repeated[0]["occurrence_count"]
    assert all(item["jump"] for item in blob_occurrences.json()["items"])

    missing = stage2_client.get("/api/v1/media", params={"availability": "missing"}).json()
    assert missing["page"]["total"] == 2
    assert all(item["availability"] == "missing" for item in missing["items"])
    assert all(item["blob_href"] is None for item in missing["items"])
    assert all(item["jump"] for item in missing["items"])

    external = stage2_client.get("/api/v1/media", params={"availability": "external"}).json()
    assert external["page"]["total"] == 1
    assert external["items"][0]["is_external_reference"] is True
    assert external["items"][0]["blob_href"] is None
    assert "source_uri" not in external["items"][0]

    photos = stage2_client.get("/api/v1/media", params={"type": "photo"}).json()
    assert photos["items"]
    assert all(item["type"] == "photo" for item in photos["items"])

    alex = _conversation(stage2_client, "Alex Example")
    alex_media = stage2_client.get(
        "/api/v1/media",
        params={
            "conversation_q": "alex example",
            "source_category": "archived_threads",
            "min_size": 1,
        },
    ).json()
    assert alex_media["items"]
    assert all(item["conversation"]["id"] == alex["id"] for item in alex_media["items"])
    assert all(item["source_category"] == "archived_threads" for item in alex_media["items"])

    scoped_cursor_page = stage2_client.get(
        "/api/v1/media", params={"conversation_q": "Alex", "limit": 1}
    ).json()
    scoped_cursor = scoped_cursor_page["page"]["next_cursor"]
    assert scoped_cursor
    changed_title_search = stage2_client.get(
        "/api/v1/media",
        params={"conversation_q": "Support", "limit": 1, "cursor": scoped_cursor},
    )
    assert changed_title_search.status_code == 422

    for sort in ("date", "size", "type", "filename", "conversation", "availability"):
        sorted_media = stage2_client.get(
            "/api/v1/media",
            params={"view": "occurrences", "sort": sort, "direction": "asc", "limit": 100},
        )
        assert sorted_media.status_code == 200, (sort, sorted_media.text)
        assert sorted_media.json()["page"]["total"] == 11


def test_unique_media_projection_matches_independent_representative_and_cursor_order(
    stage2_client: TestClient,
) -> None:
    with Session(stage2_client.app.state.engine) as session:
        source_rows = session.execute(
            select(
                AttachmentOccurrence.id,
                AttachmentOccurrence.blob_sha256,
                AttachmentOccurrence.timestamp_ms,
                AttachmentOccurrence.size_bytes,
                AttachmentOccurrence.filename,
                AttachmentOccurrence.availability,
                media_routes._media_type_expression().label("media_class"),
                Conversation.title_display.label("conversation_title"),
            )
            .select_from(AttachmentOccurrence)
            .outerjoin(Conversation, Conversation.id == AttachmentOccurrence.conversation_id)
            .where(AttachmentOccurrence.deleted_at.is_(None))
        ).all()

    grouped: dict[str, list[object]] = {}
    for row in source_rows:
        group_key = f"blob:{row.blob_sha256}" if row.blob_sha256 else f"occurrence:{row.id}"
        grouped.setdefault(group_key, []).append(row)

    expected_rows: list[dict[str, object]] = []
    for group_key, occurrences in grouped.items():
        representative = min(
            occurrences,
            key=lambda row: (
                row.timestamp_ms is None,  # type: ignore[attr-defined]
                -(row.timestamp_ms or 0),  # type: ignore[attr-defined]
                row.id,  # type: ignore[attr-defined]
            ),
        )
        timestamps = [
            row.timestamp_ms  # type: ignore[attr-defined]
            for row in occurrences
            if row.timestamp_ms is not None  # type: ignore[attr-defined]
        ]
        expected_rows.append(
            {
                "id": representative.id,  # type: ignore[attr-defined]
                "view_key": group_key,
                "occurrence_count": len(occurrences),
                "first_timestamp_ms": min(timestamps) if timestamps else None,
                "last_timestamp_ms": max(timestamps) if timestamps else None,
                "date": max(timestamps) if timestamps else None,
                "size": representative.size_bytes,  # type: ignore[attr-defined]
                "type": representative.media_class,  # type: ignore[attr-defined]
                "filename": (
                    representative.filename.lower()  # type: ignore[attr-defined]
                    if representative.filename is not None  # type: ignore[attr-defined]
                    else None
                ),
                "conversation": (
                    representative.conversation_title.lower()  # type: ignore[attr-defined]
                    if representative.conversation_title is not None  # type: ignore[attr-defined]
                    else None
                ),
                "availability": representative.availability,  # type: ignore[attr-defined]
                "occurrences": len(occurrences),
            }
        )

    for sort in sorted(media_routes.MEDIA_SORT_NAMES):
        for direction in ("asc", "desc"):
            non_null = [row for row in expected_rows if row[sort] is not None]
            null = [row for row in expected_rows if row[sort] is None]
            reverse = direction == "desc"
            expected = [
                *sorted(non_null, key=lambda row: (row[sort], row["view_key"]), reverse=reverse),
                *sorted(null, key=lambda row: row["view_key"], reverse=reverse),
            ]
            actual: list[dict[str, object]] = []
            cursor = None
            while True:
                params = {
                    "view": "unique",
                    "sort": sort,
                    "direction": direction,
                    "limit": 2,
                }
                if cursor is not None:
                    params["cursor"] = cursor
                response = stage2_client.get("/api/v1/media", params=params)
                assert response.status_code == 200, (sort, direction, response.text)
                payload = response.json()
                actual.extend(payload["items"])
                cursor = payload["page"]["next_cursor"]
                if cursor is None:
                    break
            fields = (
                "id",
                "view_key",
                "occurrence_count",
                "first_timestamp_ms",
                "last_timestamp_ms",
            )
            assert [tuple(row[field] for field in fields) for row in actual] == [
                tuple(row[field] for field in fields) for row in expected
            ]


def test_occurrence_media_projection_matches_independent_all_sort_cursor_order(
    stage2_client: TestClient,
) -> None:
    with Session(stage2_client.app.state.engine) as session:
        source_rows = session.execute(
            select(
                AttachmentOccurrence.id,
                AttachmentOccurrence.timestamp_ms,
                AttachmentOccurrence.size_bytes,
                AttachmentOccurrence.filename,
                AttachmentOccurrence.availability,
                media_routes._media_type_expression().label("media_class"),
                Conversation.title_display.label("conversation_title"),
            )
            .select_from(AttachmentOccurrence)
            .outerjoin(Conversation, Conversation.id == AttachmentOccurrence.conversation_id)
            .where(AttachmentOccurrence.deleted_at.is_(None))
        ).all()
    expected_rows = [
        {
            "id": row.id,
            "date": row.timestamp_ms,
            "size": row.size_bytes,
            "type": row.media_class,
            "filename": row.filename.lower() if row.filename is not None else None,
            "conversation": (
                row.conversation_title.lower() if row.conversation_title is not None else None
            ),
            "availability": row.availability,
            "occurrences": 1,
        }
        for row in source_rows
    ]

    for sort in sorted(media_routes.MEDIA_SORT_NAMES):
        for direction in ("asc", "desc"):
            non_null = [row for row in expected_rows if row[sort] is not None]
            null = [row for row in expected_rows if row[sort] is None]
            reverse = direction == "desc"
            expected = [
                *sorted(non_null, key=lambda row: (row[sort], row["id"]), reverse=reverse),
                *sorted(null, key=lambda row: row["id"], reverse=reverse),
            ]
            actual: list[dict[str, object]] = []
            cursor = None
            while True:
                params = {
                    "view": "occurrences",
                    "sort": sort,
                    "direction": direction,
                    "limit": 2,
                }
                if cursor is not None:
                    params["cursor"] = cursor
                response = stage2_client.get("/api/v1/media", params=params)
                assert response.status_code == 200, (sort, direction, response.text)
                payload = response.json()
                actual.extend(payload["items"])
                cursor = payload["page"]["next_cursor"]
                if cursor is None:
                    break
            assert [row["id"] for row in actual] == [row["id"] for row in expected]
            assert all(row["view_key"] == f"occurrence:{row['id']}" for row in actual)


def test_occurrence_media_cursor_pages_reuse_one_bounded_projection_query(
    stage2_client: TestClient,
) -> None:
    statements: list[str] = []

    def capture_statement(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: object,
    ) -> None:
        statements.append(statement)

    engine = stage2_client.app.state.engine
    event.listen(engine, "before_cursor_execute", capture_statement)
    try:
        cursor = None
        for _page_number in range(3):
            params = {
                "view": "occurrences",
                "sort": "date",
                "direction": "desc",
                "limit": 2,
            }
            if cursor is not None:
                params["cursor"] = cursor
            response = stage2_client.get("/api/v1/media", params=params)
            assert response.status_code == 200, response.text
            cursor = response.json()["page"]["next_cursor"]
            assert cursor is not None
    finally:
        event.remove(engine, "before_cursor_execute", capture_statement)

    projection_sql = [
        statement for statement in statements if "media-occurrence-projection" in statement
    ]
    assert len(projection_sql) == 1
    normalized = projection_sql[0].lower()
    assert "row_number" not in normalized
    assert " over (" not in normalized


def test_unique_media_cursor_pages_reuse_one_window_free_grouped_projection(
    stage2_client: TestClient,
) -> None:
    statements: list[str] = []

    def capture_statement(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: object,
    ) -> None:
        statements.append(statement)

    engine = stage2_client.app.state.engine
    event.listen(engine, "before_cursor_execute", capture_statement)
    try:
        cursor = None
        for _page_number in range(3):
            params = {
                "view": "unique",
                "sort": "date",
                "direction": "desc",
                "limit": 2,
            }
            if cursor is not None:
                params["cursor"] = cursor
            response = stage2_client.get("/api/v1/media", params=params)
            assert response.status_code == 200, response.text
            cursor = response.json()["page"]["next_cursor"]
            assert cursor is not None
    finally:
        event.remove(engine, "before_cursor_execute", capture_statement)

    grouped_projection_sql = [
        statement for statement in statements if "filtered_media_occurrences" in statement
    ]
    assert len(grouped_projection_sql) == 1
    normalized = grouped_projection_sql[0].lower()
    assert "row_number" not in normalized
    assert " over (" not in normalized


def test_media_projection_cache_is_shared_opaque_immutable_and_strictly_bounded() -> None:
    cache = media_routes._MediaProjectionCache(
        max_entries=2,
        max_total_rows=3,
        max_total_bytes=20_000,
    )

    def projection(prefix: str, rows: int) -> media_routes._MediaProjection:
        ascending = tuple(
            media_routes._MediaProjectionRow(
                id=f"{prefix}-id-{index}",
                group_key=f"{prefix}-group-{index}",
                occurrence_count=1,
                first_timestamp_ms=index,
                last_timestamp_ms=index,
                sort_value=index,
                sort_null=0,
            )
            for index in range(rows)
        )
        return media_routes._MediaProjection(
            ascending=ascending,
            descending=tuple(reversed(ascending)),
            accounted_bytes=1_000 * rows,
        )

    secret_scope = "private filename and conversation title"
    first = cache.get_or_compute(
        namespace="unique",
        scope=secret_scope,
        revision="revision-one",
        sort="filename",
        revision_is_current=lambda: True,
        compute=lambda: projection("first", 2),
    )
    assert isinstance(first.ascending, tuple)
    assert all(isinstance(key, bytes) and len(key) == 32 for key in cache._entries)
    assert all(secret_scope.encode() not in key for key in cache._entries)

    cache.get_or_compute(
        namespace="occurrences",
        scope="second",
        revision="revision-one",
        sort="date",
        revision_is_current=lambda: True,
        compute=lambda: projection("second", 2),
    )
    assert len(cache) == 1
    assert cache.retained_rows == 2 <= cache.max_total_rows
    assert cache.retained_bytes == 2_000 <= cache.max_total_bytes

    oversized = cache.get_or_compute(
        namespace="occurrences",
        scope="oversized",
        revision="revision-one",
        sort="date",
        revision_is_current=lambda: True,
        compute=lambda: projection("oversized", 4),
    )
    assert oversized.row_count == 4
    assert len(cache) == 1
    assert cache.retained_rows <= cache.max_total_rows
    assert cache.retained_bytes <= cache.max_total_bytes


def test_media_summary_aggregate_is_revision_cached_and_avoids_default_joins(
    stage2_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[bool] = []
    original = media_routes._base_summary

    def tracked_summary(*args: object, **kwargs: object) -> object:
        calls.append(bool(kwargs["needs_joins"]))
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(media_routes, "_base_summary", tracked_summary)
    first = stage2_client.get("/api/v1/media", params={"limit": 3})
    second = stage2_client.get("/api/v1/media", params={"limit": 3})
    assert first.status_code == second.status_code == 200
    assert first.json()["summary"] == second.json()["summary"]
    assert first.json()["facets"] == second.json()["facets"]
    assert calls == [False]

    joined = stage2_client.get("/api/v1/media", params={"conversation_q": "Alex", "limit": 3})
    assert joined.status_code == 200
    assert calls == [False, True]

    aggregate_cache = stage2_client.app.state.catalog_aggregate_cache
    assert all(isinstance(key, bytes) and len(key) == 32 for key in aggregate_cache._entries)
    assert all(
        isinstance(value, media_routes._MediaSummaryAggregate)
        for value in aggregate_cache._entries.values()
    )

    with Session(stage2_client.app.state.engine) as session, session.begin():
        occurrence = session.scalar(
            select(AttachmentOccurrence)
            .where(AttachmentOccurrence.deleted_at.is_(None))
            .order_by(AttachmentOccurrence.id)
            .limit(1)
        )
        assert occurrence is not None
        occurrence.deleted_at = datetime.now(UTC)
    invalidated = stage2_client.get("/api/v1/media", params={"limit": 3})
    assert invalidated.status_code == 200
    assert (
        invalidated.json()["summary"]["occurrences"] == first.json()["summary"]["occurrences"] - 1
    )
    assert calls == [False, True, False]


def test_media_page_cache_is_deep_copied_and_invalidated_by_catalog_revision(
    stage2_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0
    projection_calls = 0
    original = media_routes._unique_page
    original_projection = media_routes._build_media_projection

    def tracked_page(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)  # type: ignore[arg-type]

    def tracked_projection(*args: object, **kwargs: object) -> object:
        nonlocal projection_calls
        projection_calls += 1
        return original_projection(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(media_routes, "_unique_page", tracked_page)
    monkeypatch.setattr(media_routes, "_build_media_projection", tracked_projection)
    params = {"view": "unique", "sort": "occurrences", "direction": "desc", "limit": 3}
    first = stage2_client.get("/api/v1/media", params=params)
    second = stage2_client.get("/api/v1/media", params=params)
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert calls == 1
    assert projection_calls == 1

    # Mutating one returned object cannot poison the cached deep copy.
    first_payload = first.json()
    first_payload["items"][0]["filename"] = "mutated-client-copy"
    third = stage2_client.get("/api/v1/media", params=params)
    assert third.json()["items"][0]["filename"] != "mutated-client-copy"
    assert calls == 1
    assert projection_calls == 1

    with Session(stage2_client.app.state.engine) as session, session.begin():
        occurrence = session.scalar(
            select(AttachmentOccurrence)
            .where(AttachmentOccurrence.deleted_at.is_(None))
            .order_by(AttachmentOccurrence.id)
            .limit(1)
        )
        assert occurrence is not None
        occurrence.deleted_at = datetime.now(UTC)

    after_revision = stage2_client.get("/api/v1/media", params=params)
    assert after_revision.status_code == 200
    assert calls == 2
    assert projection_calls == 2
    assert after_revision.json()["summary"]["occurrences"] == (
        first.json()["summary"]["occurrences"] - 1
    )


def test_occurrence_projection_reuses_middle_pages_and_invalidates_on_mutation(
    stage2_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projection_calls = 0
    original_projection = media_routes._build_media_projection

    def tracked_projection(*args: object, **kwargs: object) -> object:
        nonlocal projection_calls
        projection_calls += 1
        return original_projection(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(media_routes, "_build_media_projection", tracked_projection)
    params = {
        "view": "occurrences",
        "sort": "size",
        "direction": "asc",
        "limit": 2,
    }
    first = stage2_client.get("/api/v1/media", params=params)
    assert first.status_code == 200, first.text
    cursor = first.json()["page"]["next_cursor"]
    assert cursor is not None
    middle = stage2_client.get("/api/v1/media", params={**params, "cursor": cursor})
    assert middle.status_code == 200, middle.text
    assert projection_calls == 1

    with Session(stage2_client.app.state.engine) as session, session.begin():
        occurrence = session.scalar(
            select(AttachmentOccurrence)
            .where(AttachmentOccurrence.deleted_at.is_(None))
            .order_by(AttachmentOccurrence.id)
            .limit(1)
        )
        assert occurrence is not None
        occurrence.deleted_at = datetime.now(UTC)

    invalidated = stage2_client.get("/api/v1/media", params=params)
    assert invalidated.status_code == 200, invalidated.text
    assert projection_calls == 2
    assert invalidated.json()["page"]["total"] == first.json()["page"]["total"] - 1


def test_media_fails_closed_when_catalog_changes_after_a_warm_page_hit(
    stage2_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameters = {"view": "occurrences", "sort": "date", "direction": "desc", "limit": 3}
    warm = stage2_client.get("/api/v1/media", params=parameters)
    assert warm.status_code == 200, warm.text
    warm_total = int(warm.json()["page"]["total"])

    original = media_routes.derivatives_by_blob
    armed = True

    def commit_after_warm_page(
        session: Session,
        blob_sha256s: Iterable[str | None],
    ) -> dict[str, list[dict[str, object]]]:
        nonlocal armed
        if armed:
            armed = False
            with Session(stage2_client.app.state.engine) as mutation, mutation.begin():
                occurrence = mutation.scalar(
                    select(AttachmentOccurrence)
                    .where(AttachmentOccurrence.deleted_at.is_(None))
                    .order_by(AttachmentOccurrence.id)
                    .limit(1)
                )
                assert occurrence is not None
                occurrence.deleted_at = datetime.now(UTC)
        return original(session, blob_sha256s)

    monkeypatch.setattr(media_routes, "derivatives_by_blob", commit_after_warm_page)
    raced = stage2_client.get("/api/v1/media", params=parameters)
    assert raced.status_code == 409
    assert raced.json() == {"detail": CATALOG_CHANGED_RETRY_DETAIL}

    retried = stage2_client.get("/api/v1/media", params=parameters)
    assert retried.status_code == 200, retried.text
    assert retried.json()["page"]["total"] == warm_total - 1


def test_unlinked_media_exposes_only_safe_archive_relative_provenance(
    stage2_client: TestClient,
) -> None:
    with Session(stage2_client.app.state.engine) as session, session.begin():
        original = session.scalar(
            select(AttachmentOccurrence)
            .where(AttachmentOccurrence.resolved_source_file_id.is_not(None))
            .limit(1)
        )
        assert original is not None and original.resolved_source_file_id is not None
        source = session.get(SourceFile, original.resolved_source_file_id)
        assert source is not None and source.sha256 is not None
        chunk = session.get(SourceChunk, source.chunk_id)
        assert chunk is not None
        occurrence_id = "unlinked-provenance-fixture"
        session.add(
            AttachmentOccurrence(
                id=occurrence_id,
                message_id=None,
                conversation_id=None,
                source_file_id=source.id,
                resolved_source_file_id=source.id,
                blob_sha256=source.sha256,
                kind="file",
                ordinal=0,
                source_uri=source.relative_path,
                filename=Path(source.relative_path).name,
                media_type=source.media_type,
                size_bytes=source.size_bytes,
                timestamp_ms=None,
                availability="unlinked",
                metadata_json={"source_copy_count": 3},
            )
        )
        expected_source_path = f"{chunk.chunk_key}/{source.relative_path}"
        source_id = source.id
        source_chunk = chunk.chunk_key
        source_relative_path = source.relative_path
        source_extension = source.extension or ""

    response = stage2_client.get(
        "/api/v1/media",
        params={"availability": "unlinked", "limit": 100},
    )
    assert response.status_code == 200, response.text
    item = next(item for item in response.json()["items"] if item["id"] == occurrence_id)
    assert item["jump"] is None
    assert item["activity"] is None
    assert item["conversation"] is None
    assert item["provenance"] == {
        "source_file_id": source_id,
        "source_chunk": source_chunk,
        "source_relative_path": source_relative_path,
        "source_path": expected_source_path,
        "source_copy_count": 3,
        "library_path": (
            "library/media/unlinked/unknown-date/unlinked--"
            f"{Path(source_relative_path).stem}--{occurrence_id}{source_extension}"
        ),
        "explanation": (
            "This catalogued file was present in the export but was not referenced by an "
            "imported message or activity record."
        ),
    }
    assert not item["provenance"]["source_path"].startswith("/")
    assert ".." not in Path(item["provenance"]["source_path"]).parts
    assert str(stage2_client.app.state.layout.root) not in str(item["provenance"])

    unique = stage2_client.get(
        "/api/v1/media",
        params={"view": "unique", "availability": "unlinked", "limit": 100},
    )
    assert unique.status_code == 200, unique.text
    unique_item = next(item for item in unique.json()["items"] if item["id"] == occurrence_id)
    assert unique_item["provenance"] == item["provenance"]


def test_stage2_dashboard_totals_match_direct_catalog_queries(stage2_client: TestClient) -> None:
    with Session(stage2_client.app.state.engine) as session:
        expected_conversations = (
            session.scalar(
                select(func.count(Conversation.id)).where(Conversation.deleted_at.is_(None))
            )
            or 0
        )
        expected_messages = (
            session.scalar(select(func.count(Message.pk)).where(Message.deleted_at.is_(None))) or 0
        )
        expected_attachments = (
            session.scalar(
                select(func.count(AttachmentOccurrence.id)).where(
                    AttachmentOccurrence.deleted_at.is_(None)
                )
            )
            or 0
        )

    conversations = stage2_client.get("/api/v1/conversations", params={"limit": 1}).json()
    assert conversations["page"]["total"] == expected_conversations
    first_conversation = conversations["items"][0]
    reader = stage2_client.get(
        f"/api/v1/conversations/{first_conversation['id']}/messages", params={"limit": 1}
    ).json()
    assert reader["page"]["total"] == first_conversation["message_counts"]["total"]
    assert expected_messages == 14
    media = stage2_client.get("/api/v1/media", params={"limit": 1}).json()
    assert media["page"]["total"] == expected_attachments
