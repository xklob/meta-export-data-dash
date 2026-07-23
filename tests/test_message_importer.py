from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

from meta_archive.api.main import create_app
from meta_archive.database import create_catalog_engine, run_migrations
from meta_archive.models import (
    ArchiveSetting,
    AttachmentCandidate,
    AttachmentOccurrence,
    Conversation,
    ConversationSegment,
    CoverageArea,
    IntegrityAnomaly,
    Message,
    MessageReaction,
    Participant,
    SegmentParticipant,
    SourceFile,
    SourceRecord,
)
from meta_archive.paths import ArchiveLayout
from meta_archive.services.inventory import scan_archive
from meta_archive.services.message_importer import import_messages

FIXTURES = Path(__file__).parent / "fixtures" / "messages"


def _catalogued_fixture(
    tmp_path: Path,
    *,
    support_message_updates: dict[str, object] | None = None,
) -> tuple[ArchiveLayout, Engine]:
    layout = ArchiveLayout(tmp_path / "meta-backup-messages")
    shutil.copytree(FIXTURES, layout.root)
    if support_message_updates:
        support_path = (
            layout.root
            / "package-a"
            / "your_facebook_activity"
            / "messages"
            / "support_messages.json"
        )
        support_payload = json.loads(support_path.read_text(encoding="utf-8"))
        support_payload["support_messages"]["case/one~alpha"]["messages"][0].update(
            support_message_updates
        )
        support_path.write_text(
            json.dumps(support_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    layout.ensure_runtime_directories()
    run_migrations(layout.database)
    engine = create_catalog_engine(layout.database)
    scan_archive(layout, engine, hash_files=True)
    return layout, engine


def test_support_import_partitions_nested_sensitive_metadata_without_over_redacting(
    tmp_path: Path,
) -> None:
    ip_marker = "198.51.100.privacy-fixture"
    card_marker = "card-privacy-fixture"
    account_marker = "account-privacy-fixture"
    layout, engine = _catalogued_fixture(
        tmp_path,
        support_message_updates={
            "ipAddress": ip_marker,
            "paymentInfo": {"card_number": card_marker},
            "category": "account support",
            "status": "open",
            "metadata": {
                "topic": "shipping",
                "participant": "Support Agent",
                "network": {"source-ip": ip_marker},
                "history": [
                    {"status": "received"},
                    {
                        "billing_info": {"bank_account_number": account_marker},
                        "status": "reviewed",
                    },
                ],
            },
        },
    )
    try:
        import_messages(layout, engine)

        with Session(engine) as session:
            message = session.scalar(
                select(Message).where(Message.content_raw == "Newest support reply")
            )
            assert message is not None
            assert message.has_sensitive_ip is True
            assert message.has_payment is True
            assert message.extra_json == {
                "to": "Archive Owner",
                "subject": "Support Case One",
                "category": "account support",
                "status": "open",
                "metadata": {
                    "topic": "shipping",
                    "participant": "Support Agent",
                    "network": {},
                    "history": [
                        {"status": "received"},
                        {"status": "reviewed"},
                    ],
                },
            }
            public_json = json.dumps(message.extra_json, sort_keys=True)
            assert ip_marker not in public_json
            assert card_marker not in public_json
            assert account_marker not in public_json
            assert message.sensitive_json == {
                "ipAddress": ip_marker,
                "paymentInfo": {"card_number": card_marker},
                "metadata": {
                    "network": {"source-ip": ip_marker},
                    "history": [
                        None,
                        {"billing_info": {"bank_account_number": account_marker}},
                    ],
                },
            }
    finally:
        engine.dispose()


def _table_counts(engine: Engine) -> dict[str, int]:
    models = (
        Conversation,
        ConversationSegment,
        Participant,
        SegmentParticipant,
        SourceRecord,
        Message,
        MessageReaction,
        AttachmentOccurrence,
        AttachmentCandidate,
    )
    with Session(engine) as session:
        return {
            model.__tablename__: session.scalar(select(func.count()).select_from(model)) or 0
            for model in models
        }


def test_imports_every_observed_message_shape_with_provenance_and_aggregates(
    tmp_path: Path,
) -> None:
    layout, engine = _catalogued_fixture(tmp_path)
    try:
        summary = import_messages(layout, engine)

        assert summary.candidate_file_count == 8
        assert summary.imported_file_count == 8
        assert summary.skipped_file_count == 0
        assert summary.invalid_file_count == 0
        assert summary.unsupported_file_count == 0
        assert summary.normal_container_count == 7
        assert summary.support_container_count == 1
        assert summary.normal_segment_count == 6
        assert summary.support_thread_count == 2
        assert summary.conversation_count == 7
        assert summary.media_only_conversation_count == 0
        assert summary.normal_message_count == 11
        assert summary.support_message_count == 3
        assert summary.participant_occurrence_count == 18
        assert summary.reaction_count == 2
        assert summary.attachment_count == 9
        assert summary.thread_image_count == 1
        assert summary.resolved_attachment_count == 7
        assert summary.missing_attachment_count == 2
        assert summary.external_attachment_count == 1
        assert summary.cross_chunk_attachment_count == 7
        assert summary.duplicate_message_count == 1
        assert summary.missing_leading_segment_count == 1
        assert summary.internal_gap_segment_count == 1

        with Session(engine) as session:
            alex = session.scalar(
                select(Conversation).where(Conversation.title_display == "Alex Example")
            )
            assert alex is not None
            assert alex.conversation_type == "direct"
            assert alex.message_count == 7
            assert alex.sent_count == 2
            assert alex.received_count == 5
            assert alex.attachment_count == 9
            assert alex.unique_attachment_count == 6
            assert alex.attachment_bytes > 0
            assert alex.completeness == "incomplete"

            alex_segments = session.scalars(
                select(ConversationSegment).where(ConversationSegment.conversation_id == alex.id)
            ).all()
            assert {segment.source_category for segment in alex_segments} == {
                "archived_threads",
                "e2ee_cutover",
            }
            assert {segment.title_display for segment in alex_segments} == {
                "Alex Example",
                "A Different Historical Thread Title",
            }
            assert len(alex_segments) == 2

            groups = session.scalars(
                select(Conversation).where(Conversation.title_display == "Shared Group Title")
            ).all()
            assert len(groups) == 2
            assert {group.conversation_type for group in groups} == {"group"}
            assert len({group.id for group in groups}) == 2

            group_one = session.scalar(
                select(ConversationSegment).where(
                    ConversationSegment.thread_path == "filtered_threads/group_one"
                )
            )
            assert group_one is not None
            duplicate_labels = session.scalars(
                select(Participant.name_display)
                .join(SegmentParticipant, SegmentParticipant.participant_id == Participant.id)
                .where(SegmentParticipant.segment_id == group_one.id)
                .order_by(SegmentParticipant.ordinal)
            ).all()
            assert duplicate_labels == [
                "Archive Owner",
                "Repeated Label",
                "Repeated Label",
            ]

            special = session.scalar(
                select(ConversationSegment).where(
                    ConversationSegment.thread_path == "archived_threads/special_envelope"
                )
            )
            assert special is not None
            assert special.source_category == "archived_threads"

            normalized = session.scalar(
                select(Message).where(Message.content_raw == "FranÃ§ais \uff21\uff22\uff23")
            )
            assert normalized is not None
            assert normalized.content_display == "Français ABC"
            assert normalized.content_search == "français abc"
            assert normalized.sender_is_owner is True
            assert normalized.json_pointer == "/messages/0"
            assert normalized.chunk_number == 1
            assert normalized.source_array_index == 0
            assert normalized.has_sensitive_ip is True
            assert normalized.has_payment is True
            assert normalized.sensitive_json == {
                "ip": "192.0.2.10",
                "payment_info": {"amount": "fixture-only"},
            }
            assert "ip" not in normalized.extra_json
            assert "payment_info" not in normalized.extra_json
            assert normalized.extra_json["fixture_extension"] == {"kept": True}
            assert normalized.message_kind == "payment"

            unsent = session.scalar(select(Message).where(Message.message_kind == "unsent"))
            unavailable = session.scalar(
                select(Message).where(Message.message_kind == "unavailable")
            )
            transcribed = session.scalar(select(Message).where(Message.has_transcription.is_(True)))
            call = session.scalar(select(Message).where(Message.message_kind == "call"))
            assert unsent is not None and unsent.is_unsent is True
            assert unavailable is not None and unavailable.is_unavailable_body is True
            assert transcribed is not None
            assert transcribed.transcription_search == "transcribed fixture"
            assert call is not None
            assert call.call_duration_seconds == 12
            assert call.is_missed_call is False

            timestamped_reaction = session.scalar(
                select(MessageReaction).where(MessageReaction.timestamp_ms.is_not(None))
            )
            assert timestamped_reaction is not None
            assert timestamped_reaction.timestamp_ms == 1_700_000_000_000
            assert timestamped_reaction.reaction_display == "❤"

            duplicate = session.scalar(select(Message).where(Message.duplicate_of_id.is_not(None)))
            assert duplicate is not None
            source_alias = session.get(SourceRecord, duplicate.source_record_id)
            assert source_alias is not None
            assert source_alias.duplicate_of_id is not None

            coverage = {
                row.key: row
                for row in session.scalars(
                    select(CoverageArea).where(
                        CoverageArea.key.in_(
                            (
                                "messages",
                                "message_attachments",
                                "message_reactions",
                                "support_messages",
                            )
                        )
                    )
                )
            }
            assert coverage["messages"].raw_count == 14
            assert coverage["messages"].logical_count == 13
            assert coverage["messages"].source_paths_json["media_only_conversations"] == 0
            assert coverage["message_attachments"].raw_count == 10
            assert coverage["support_messages"].raw_count == 3
            assert coverage["support_messages"].logical_count == 2
    finally:
        engine.dispose()


def test_import_uses_existing_private_owner_setting(tmp_path: Path) -> None:
    layout, engine = _catalogued_fixture(tmp_path)
    try:
        with Session(engine) as session, session.begin():
            session.add(
                ArchiveSetting(
                    key="owner_identity",
                    value_json={"value": "Alex Example"},
                )
            )

        import_messages(layout, engine)

        with Session(engine) as session:
            alex_messages = session.scalars(
                select(Message).where(Message.sender_display == "Alex Example")
            ).all()
            neutral_messages = session.scalars(
                select(Message).where(Message.sender_display == "Archive Owner")
            ).all()
        assert alex_messages and all(message.sender_is_owner for message in alex_messages)
        assert neutral_messages and all(not message.sender_is_owner for message in neutral_messages)
    finally:
        engine.dispose()


def test_explicit_owner_identity_is_persisted_and_forces_reclassification(
    tmp_path: Path,
) -> None:
    layout, engine = _catalogued_fixture(tmp_path)
    try:
        first = import_messages(layout, engine)
        assert first.imported_file_count == first.candidate_file_count

        changed = import_messages(layout, engine, owner_name="Alex Example")

        assert changed.imported_file_count == changed.candidate_file_count
        assert changed.skipped_file_count == 0
        with Session(engine) as session:
            setting = session.get(ArchiveSetting, "owner_identity")
            alex_messages = session.scalars(
                select(Message).where(Message.sender_display == "Alex Example")
            ).all()
            neutral_messages = session.scalars(
                select(Message).where(Message.sender_display == "Archive Owner")
            ).all()
        assert setting is not None
        assert setting.value_json == {"value": "Alex Example"}
        assert alex_messages and all(message.sender_is_owner for message in alex_messages)
        assert neutral_messages and all(not message.sender_is_owner for message in neutral_messages)
    finally:
        engine.dispose()


def test_owner_reclassification_resumes_after_a_committed_partial_import(
    tmp_path: Path,
) -> None:
    layout, engine = _catalogued_fixture(tmp_path)
    try:
        baseline = import_messages(layout, engine)
        assert baseline.imported_file_count == baseline.candidate_file_count

        def crash_after_first_file(stage: str, current: int, total: int | None) -> None:
            if stage == "Importing message containers" and current == 1:
                assert total == baseline.candidate_file_count
                raise RuntimeError("fault-injected owner reclassification crash")

        with pytest.raises(RuntimeError, match="fault-injected"):
            import_messages(
                layout,
                engine,
                owner_name="Alex Example",
                progress=crash_after_first_file,
            )

        with Session(engine) as session:
            setting = session.get(ArchiveSetting, "owner_identity")
            assert setting is not None
            assert setting.value_json == {"value": "Alex Example"}
            message_mismatches = (
                session.scalar(
                    select(func.count(Message.pk)).where(
                        Message.sender_display == "Alex Example",
                        Message.sender_is_owner.is_(False),
                    )
                )
                or 0
            ) + (
                session.scalar(
                    select(func.count(Message.pk)).where(
                        Message.sender_display != "Alex Example",
                        Message.sender_is_owner.is_(True),
                    )
                )
                or 0
            )
            participant_mismatches = (
                session.scalar(
                    select(func.count(Participant.id)).where(
                        Participant.name_display == "Alex Example",
                        Participant.is_owner.is_(False),
                    )
                )
                or 0
            ) + (
                session.scalar(
                    select(func.count(Participant.id)).where(
                        Participant.name_display != "Alex Example",
                        Participant.is_owner.is_(True),
                    )
                )
                or 0
            )
        assert message_mismatches > 0
        assert participant_mismatches > 0

        resumed = import_messages(layout, engine, owner_name="Alex Example")

        assert resumed.imported_file_count == resumed.candidate_file_count
        assert resumed.skipped_file_count == 0
        with Session(engine) as session:
            assert (
                session.scalar(
                    select(func.count(Message.pk)).where(
                        Message.sender_display == "Alex Example",
                        Message.sender_is_owner.is_(False),
                    )
                )
                == 0
            )
            assert (
                session.scalar(
                    select(func.count(Message.pk)).where(
                        Message.sender_display != "Alex Example",
                        Message.sender_is_owner.is_(True),
                    )
                )
                == 0
            )
            assert (
                session.scalar(
                    select(func.count(Participant.id)).where(
                        Participant.name_display == "Alex Example",
                        Participant.is_owner.is_(False),
                    )
                )
                == 0
            )
            assert (
                session.scalar(
                    select(func.count(Participant.id)).where(
                        Participant.name_display != "Alex Example",
                        Participant.is_owner.is_(True),
                    )
                )
                == 0
            )
    finally:
        engine.dispose()


def test_explicit_owner_identity_rejects_blank_values(tmp_path: Path) -> None:
    layout, engine = _catalogued_fixture(tmp_path)
    try:
        with pytest.raises(ValueError, match="visible character"):
            import_messages(layout, engine, owner_name="  ")
        with Session(engine) as session:
            assert session.get(ArchiveSetting, "owner_identity") is None
    finally:
        engine.dispose()


def test_explicit_owner_identity_adopts_matching_legacy_classification_without_reimport(
    tmp_path: Path,
) -> None:
    layout, engine = _catalogued_fixture(tmp_path)
    try:
        imported = import_messages(layout, engine, owner_name="Alex Example")
        assert imported.imported_file_count == imported.candidate_file_count
        with Session(engine) as session, session.begin():
            setting = session.get(ArchiveSetting, "owner_identity")
            assert setting is not None
            session.delete(setting)

        adopted = import_messages(layout, engine, owner_name="Alex Example")

        assert adopted.imported_file_count == 0
        assert adopted.skipped_file_count == adopted.candidate_file_count
        with Session(engine) as session:
            setting = session.get(ArchiveSetting, "owner_identity")
            assert setting is not None
            assert setting.value_json == {"value": "Alex Example"}
    finally:
        engine.dispose()


def test_media_only_thread_is_classified_and_exposed_without_inventing_messages(
    tmp_path: Path,
) -> None:
    fixture_root = tmp_path / "meta-backup-media-only"
    shutil.copytree(FIXTURES, fixture_root)
    thread_root = (
        fixture_root
        / "package-a/your_facebook_activity/messages/archived_threads/media_only_fixture"
    )
    thread_root.mkdir(parents=True)
    image_path = thread_root / "thread-image.jpg"
    image_path.write_bytes(b"media-only-fixture")
    (thread_root / "message_1.json").write_text(
        json.dumps(
            {
                "title": "Media Only Fixture",
                "participants": [
                    {"name": "Archive Owner"},
                    {"name": "Media Only Contact"},
                ],
                "messages": [],
                "thread_path": "archived_threads/media_only_fixture",
                "is_still_participant": True,
                "magic_words": [],
                "image": {
                    "uri": (
                        "your_facebook_activity/messages/archived_threads/"
                        "media_only_fixture/thread-image.jpg"
                    )
                },
            }
        ),
        encoding="utf-8",
    )
    layout = ArchiveLayout(fixture_root)
    layout.ensure_runtime_directories()
    run_migrations(layout.database)
    engine = create_catalog_engine(layout.database)
    try:
        scan_archive(layout, engine, hash_files=True)
        summary = import_messages(layout, engine)
        assert summary.media_only_conversation_count == 1

        with Session(engine) as session:
            segment = session.scalar(
                select(ConversationSegment).where(
                    ConversationSegment.thread_path == "archived_threads/media_only_fixture"
                )
            )
            assert segment is not None
            conversation = session.get(Conversation, segment.conversation_id)
            assert conversation is not None
            assert conversation.message_count == 0
            assert conversation.attachment_count == 1
            assert (
                session.scalar(
                    select(func.count(Message.pk)).where(Message.conversation_id == conversation.id)
                )
                == 0
            )
            coverage = session.get(CoverageArea, "messages")
            assert coverage is not None
            assert coverage.source_paths_json["media_only_conversations"] == 1

        app = create_app(layout.root, security_disabled=True)
        with TestClient(app) as client:
            response = client.get("/api/v1/conversations", params={"q": "Media Only Contact"})
            assert response.status_code == 200
            item = response.json()["items"][0]
            assert item["message_counts"]["total"] == 0
            assert item["content_profile"] == {
                "kind": "media_only",
                "message_records": "not_present",
                "attachment_records": "available",
            }
            detail = client.get(f"/api/v1/conversations/{item['id']}")
            assert detail.status_code == 200
            assert detail.json()["conversation"]["content_profile"]["kind"] == "media_only"
    finally:
        engine.dispose()


def test_resolves_cross_chunk_media_and_keeps_missing_external_and_unsafe_rows(
    tmp_path: Path,
) -> None:
    layout, engine = _catalogued_fixture(tmp_path)
    try:
        import_messages(layout, engine)

        with Session(engine) as session:
            cross_chunk = session.scalar(
                select(AttachmentOccurrence).where(
                    AttachmentOccurrence.source_uri.like("%cross-chunk.jpg")
                )
            )
            assert cross_chunk is not None
            assert cross_chunk.availability == "resolved"
            assert cross_chunk.metadata_json["resolution_kind"] == "cross_chunk"
            assert cross_chunk.resolved_source_file_id is not None
            assert cross_chunk.blob_sha256 is not None
            candidates = session.scalars(
                select(AttachmentCandidate).where(
                    AttachmentCandidate.attachment_id == cross_chunk.id
                )
            ).all()
            assert len(candidates) == 1
            assert candidates[0].match_kind == "cross_chunk"
            assert candidates[0].selected is True

            missing = session.scalar(
                select(AttachmentOccurrence).where(
                    AttachmentOccurrence.source_uri.like("%missing.jpg")
                )
            )
            unsafe = session.scalar(
                select(AttachmentOccurrence).where(
                    AttachmentOccurrence.source_uri == "../outside.txt"
                )
            )
            external = session.scalar(
                select(AttachmentOccurrence).where(
                    AttachmentOccurrence.source_uri == "https://example.invalid/external.gif"
                )
            )
            thread_image = session.scalar(
                select(AttachmentOccurrence).where(AttachmentOccurrence.kind == "thread_image")
            )
            assert missing is not None and missing.availability == "missing"
            assert unsafe is not None and unsafe.availability == "missing"
            assert unsafe.metadata_json["resolution_kind"] == "unsafe"
            assert external is not None and external.availability == "external"
            assert external.resolved_source_file_id is None
            assert thread_image is not None
            assert thread_image.message_id is None
            assert thread_image.availability == "resolved"

            unresolved_codes = set(
                session.scalars(
                    select(IntegrityAnomaly.code).where(IntegrityAnomaly.resolved.is_(False))
                )
            )
            assert "unresolved_attachment" in unresolved_codes
            assert "unsafe_attachment_uri" in unresolved_codes
    finally:
        engine.dispose()


def test_support_adapter_uses_seconds_and_rfc6901_pointers(tmp_path: Path) -> None:
    layout, engine = _catalogued_fixture(tmp_path)
    try:
        import_messages(layout, engine)

        with Session(engine) as session:
            support_segments = session.scalars(
                select(ConversationSegment)
                .where(ConversationSegment.source_category == "support")
                .order_by(ConversationSegment.thread_path)
            ).all()
            assert len(support_segments) == 2
            assert {segment.thread_path for segment in support_segments} == {
                "support/case/one~alpha",
                "support/case-two",
            }
            assert all(segment.completeness == "complete" for segment in support_segments)

            escaped = session.scalar(
                select(Message).where(
                    Message.json_pointer == "/support_messages/case~1one~0alpha/messages/0"
                )
            )
            assert escaped is not None
            assert escaped.timestamp_ms == 8_000_000
            assert escaped.content_display == "Newest support reply"
            assert escaped.extra_json["to"] == "Archive Owner"
            assert escaped.extra_json["subject"] == "Support Case One"

            support_attachment = session.scalar(
                select(AttachmentOccurrence).where(AttachmentOccurrence.kind == "support_file")
            )
            assert support_attachment is not None
            assert support_attachment.timestamp_ms == 8_000_000
            assert support_attachment.metadata_json["caption"] == "preserved"
            assert support_attachment.metadata_json["resolution_kind"] == "cross_chunk"
    finally:
        engine.dispose()


def test_reimport_is_idempotent_and_changed_file_replaces_its_rows(tmp_path: Path) -> None:
    layout, engine = _catalogued_fixture(tmp_path)
    try:
        first = import_messages(layout, engine)
        initial_counts = _table_counts(engine)
        with Session(engine) as session:
            initial_ids = set(session.scalars(select(Message.id)))

        second = import_messages(layout, engine)
        assert second.imported_file_count == 0
        assert second.skipped_file_count == first.candidate_file_count
        assert _table_counts(engine) == initial_counts
        with Session(engine) as session:
            assert set(session.scalars(select(Message.id))) == initial_ids

        forced = import_messages(layout, engine, force=True)
        assert forced.imported_file_count == first.candidate_file_count
        assert forced.skipped_file_count == 0
        assert _table_counts(engine) == initial_counts

        target = (
            layout.root / "package-a/your_facebook_activity/messages/archived_threads/"
            "alex_example/message_3.json"
        )
        original = target.read_text(encoding="utf-8")
        target.write_text(
            original.replace("Older chunk", "A safely replaced source chunk"),
            encoding="utf-8",
        )
        scan_archive(layout, engine, hash_files=True)
        changed = import_messages(layout, engine)
        assert changed.imported_file_count == 1
        assert changed.skipped_file_count == first.candidate_file_count - 1
        assert _table_counts(engine) == initial_counts
        with Session(engine) as session:
            replacement = session.scalar(
                select(Message).where(Message.content_raw == "A safely replaced source chunk")
            )
            assert replacement is not None
            assert replacement.id in initial_ids
            shared_segment = session.scalar(
                select(ConversationSegment).where(
                    ConversationSegment.thread_path == "archived_threads/alex_example"
                )
            )
            assert shared_segment is not None
            assert shared_segment.source_chunk_count == 2
            assert (
                session.scalar(
                    select(func.count(Message.pk)).where(Message.segment_id == shared_segment.id)
                )
                == 6
            )
    finally:
        engine.dispose()


def test_retired_original_falls_back_to_digest_validated_catalogued_blob(
    tmp_path: Path,
) -> None:
    layout, engine = _catalogued_fixture(tmp_path)
    try:
        baseline = import_messages(layout, engine)
        with Session(engine) as session:
            source = session.scalar(
                select(SourceFile).where(
                    SourceFile.relative_path.like("%archived_threads/special_envelope_1.json")
                )
            )
            assert source is not None
            assert source.sha256 is not None
            source_digest = source.sha256
            source_relative_path = source.relative_path

        original = layout.root / "package-a" / source_relative_path
        blob = layout.blob_path(source_digest)
        blob.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, blob)
        original.unlink()

        fallback = import_messages(layout, engine, force=True)
        assert fallback.invalid_file_count == 0
        assert fallback.normal_message_count == baseline.normal_message_count
        with Session(engine) as session:
            assert (
                session.scalar(
                    select(func.count(Message.pk)).where(
                        Message.content_raw == "Structurally discovered outside a thread directory"
                    )
                )
                == 1
            )

        blob.write_bytes(b"tampered fixture blob")
        rejected = import_messages(layout, engine, force=True)
        assert rejected.invalid_file_count == 1
        with Session(engine) as session:
            refreshed = session.scalar(select(SourceFile).where(SourceFile.id == source.id))
            assert refreshed is not None
            assert refreshed.json_status == "invalid"
            assert refreshed.json_error is not None
            assert "digest mismatch" in refreshed.json_error
    finally:
        engine.dispose()


def test_invalid_reimport_rolls_back_that_file_without_disturbing_other_files(
    tmp_path: Path,
) -> None:
    layout, engine = _catalogued_fixture(tmp_path)
    try:
        baseline = import_messages(layout, engine)
        before = _table_counts(engine)
        target = (
            layout.root / "package-a/your_facebook_activity/messages/e2ee_cutover/"
            "alex_example_new/message_1.json"
        )
        target.write_text("{not-json", encoding="utf-8")
        scan_archive(layout, engine, hash_files=True)

        result = import_messages(layout, engine)
        assert result.invalid_file_count == 1
        assert result.skipped_file_count == baseline.candidate_file_count - 1
        after = _table_counts(engine)
        assert after[Message.__tablename__] == before[Message.__tablename__] - 1
        assert after[SourceRecord.__tablename__] == before[SourceRecord.__tablename__] - 1
        assert (
            after[ConversationSegment.__tablename__]
            == before[ConversationSegment.__tablename__] - 1
        )
        with Session(engine) as session:
            source = session.scalar(
                select(SourceFile).where(SourceFile.relative_path.like("%alex_example_new%json"))
            )
            assert source is not None
            assert source.json_status == "invalid"
            assert source.json_error is not None
            assert (
                session.scalar(
                    select(func.count(Message.pk)).where(
                        Message.content_raw == "A second source segment"
                    )
                )
                == 0
            )
    finally:
        engine.dispose()


def test_identical_payloads_in_different_segments_are_not_duplicate_aliases(
    tmp_path: Path,
) -> None:
    layout, engine = _catalogued_fixture(tmp_path)
    try:
        paths = [
            layout.root
            / "package-a/your_facebook_activity/messages/filtered_threads/group_one/message_1.json",
            layout.root
            / "package-a/your_facebook_activity/messages/filtered_threads/group_two/message_1.json",
        ]
        payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        shared = {
            "sender_name": "Archive Owner",
            "timestamp_ms": 1_700_111_222_333,
            "content": "Identical fixture payload in separate conversations",
        }
        for path, payload in zip(paths, payloads, strict=True):
            payload["messages"] = [shared]
            path.write_text(json.dumps(payload), encoding="utf-8")
        scan_archive(layout, engine, hash_files=True)

        import_messages(layout, engine)

        with Session(engine) as session:
            rows = session.scalars(
                select(Message).where(
                    Message.content_raw == "Identical fixture payload in separate conversations"
                )
            ).all()
        assert len(rows) == 2
        assert len({row.segment_id for row in rows}) == 2
        assert all(row.duplicate_of_id is None for row in rows)
    finally:
        engine.dispose()
