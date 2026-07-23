from __future__ import annotations

import json
import os
import shutil
import stat
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from meta_archive.models import Conversation, ConversationSegment
from meta_archive.paths import ArchiveLayout
from meta_archive.services.library import conversation_library_relative_root
from meta_archive.services.support_library_scrub import (
    SupportLibraryScrubError,
    scrub_support_library,
)

SUPPORT_ID = "a" * 64
ORDINARY_ID = "b" * 64
IP_MARKER = "198.51.100.support-library-private"
PAYMENT_MARKER = "support-library-payment-private"


def _conversation(identifier: str, *, title: str, category: str) -> Conversation:
    return Conversation(
        id=identifier,
        conversation_key=f"fixture:{identifier}",
        title_raw=title,
        title_display=title,
        title_search=title.casefold(),
        conversation_type="system" if category == "support" else "direct",
        primary_category=category,
        participant_count=1,
        message_count=1,
        sent_count=0,
        received_count=1,
        attachment_count=0,
        unique_attachment_count=0,
        attachment_bytes=0,
        active_days=1,
        completeness="complete",
    )


def _write_ndjson(path: Path, records: list[dict[str, object]], *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    path.chmod(mode)


def _prepared_library(
    layout: ArchiveLayout,
    engine: Engine,
) -> tuple[Path, Path, Path]:
    support = _conversation(SUPPORT_ID, title="Support Fixture", category="support")
    ordinary = _conversation(ORDINARY_ID, title="Ordinary Fixture", category="inbox")
    support_root = layout.library / conversation_library_relative_root(support) / "transcripts"
    ordinary_root = layout.library / conversation_library_relative_root(ordinary) / "transcripts"
    with Session(engine) as session, session.begin():
        session.add_all((support, ordinary))
        session.add_all(
            (
                ConversationSegment(
                    id="c" * 64,
                    conversation_id=support.id,
                    thread_path="support/fixture",
                    title_raw=support.title_raw,
                    title_display=support.title_display,
                    title_search=support.title_search,
                    source_category="support",
                    sequence=1,
                    source_chunk_count=1,
                    completeness="complete",
                    metadata_json={},
                ),
                ConversationSegment(
                    id="d" * 64,
                    conversation_id=ordinary.id,
                    thread_path="inbox/fixture",
                    title_raw=ordinary.title_raw,
                    title_display=ordinary.title_display,
                    title_search=ordinary.title_search,
                    source_category="inbox",
                    sequence=1,
                    source_chunk_count=1,
                    completeness="complete",
                    metadata_json={},
                ),
            )
        )

    support_transcript = support_root / "1970-01.ndjson"
    ordinary_transcript = ordinary_root / "1970-01.ndjson"
    support_markdown = support_root / "1970-01.md"
    records = [
        {"record_type": "segment_boundary", "segment_id": "c" * 64},
        {
            "record_type": "message",
            "message_id": "e" * 64,
            "segment_id": "c" * 64,
            "extra": {
                "to": "Archive Owner",
                "status": "open",
                "shipping": "preserved",
                "metadata": {
                    "ipAddress": IP_MARKER,
                    "payment_information": {"account_credentials": PAYMENT_MARKER},
                    "participant": "Support Agent",
                },
            },
            "sensitive_fields_omitted": [],
        },
    ]
    _write_ndjson(support_transcript, records, mode=0o644)
    _write_ndjson(
        ordinary_transcript,
        [
            {"record_type": "segment_boundary", "segment_id": "d" * 64},
            {
                "record_type": "message",
                "message_id": "f" * 64,
                "segment_id": "d" * 64,
                "extra": {"ipAddress": IP_MARKER},
                "sensitive_fields_omitted": [],
            },
        ],
    )
    support_markdown.write_text(
        f"# Support fixture\n\nMarkdown remains byte-identical: {IP_MARKER}\n",
        encoding="utf-8",
    )
    return support_transcript, ordinary_transcript, support_markdown


def _records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_scrub_support_library_is_targeted_private_and_idempotent(
    archive_layout: ArchiveLayout,
    catalog_engine: Engine,
    capsys: pytest.CaptureFixture[str],
) -> None:
    support, ordinary, markdown = _prepared_library(archive_layout, catalog_engine)
    ordinary_before = ordinary.read_bytes()
    ordinary_identity = (
        ordinary.stat().st_dev,
        ordinary.stat().st_ino,
        ordinary.stat().st_mtime_ns,
    )
    markdown_before = markdown.read_bytes()
    markdown_identity = (
        markdown.stat().st_dev,
        markdown.stat().st_ino,
        markdown.stat().st_mtime_ns,
    )

    first = scrub_support_library(archive_layout, catalog_engine)

    assert first.support_conversation_count == 1
    assert first.conversation_directories_found == 1
    assert first.transcript_files_scanned == 1
    assert first.transcript_records_scanned == 2
    assert first.message_records_scanned == 1
    assert first.message_records_changed == 1
    assert first.sensitive_fields_removed == 2
    assert first.transcript_files_rewritten == 1
    assert first.transcript_files_unchanged == 0
    assert stat.S_IMODE(support.stat().st_mode) == 0o600
    message = next(record for record in _records(support) if record["record_type"] == "message")
    assert message["extra"] == {
        "to": "Archive Owner",
        "status": "open",
        "shipping": "preserved",
        "metadata": {"participant": "Support Agent"},
    }
    assert message["sensitive_fields_omitted"] == ["ipAddress", "payment_information"]
    assert IP_MARKER not in support.read_text(encoding="utf-8")
    assert PAYMENT_MARKER not in support.read_text(encoding="utf-8")
    assert ordinary.read_bytes() == ordinary_before
    assert (ordinary.stat().st_dev, ordinary.stat().st_ino, ordinary.stat().st_mtime_ns) == (
        ordinary_identity
    )
    assert markdown.read_bytes() == markdown_before
    assert (markdown.stat().st_dev, markdown.stat().st_ino, markdown.stat().st_mtime_ns) == (
        markdown_identity
    )
    assert capsys.readouterr() == ("", "")

    repaired_before = support.read_bytes()
    repaired_identity = (support.stat().st_dev, support.stat().st_ino, support.stat().st_mtime_ns)
    second = scrub_support_library(archive_layout, catalog_engine)

    assert second.transcript_files_rewritten == 0
    assert second.transcript_files_unchanged == 1
    assert second.message_records_changed == 0
    assert second.sensitive_fields_removed == 0
    assert support.read_bytes() == repaired_before
    assert (support.stat().st_dev, support.stat().st_ino, support.stat().st_mtime_ns) == (
        repaired_identity
    )


def test_scrub_support_library_is_safe_when_library_or_support_rows_are_absent(
    archive_layout: ArchiveLayout,
    catalog_engine: Engine,
) -> None:
    empty_library = scrub_support_library(archive_layout, catalog_engine)
    assert empty_library.support_conversation_count == 0
    assert empty_library.transcript_files_scanned == 0

    shutil.rmtree(archive_layout.library)
    missing_library = scrub_support_library(archive_layout, catalog_engine)
    assert missing_library.support_conversation_count == 0
    assert missing_library.transcript_files_scanned == 0


def test_scrub_support_library_finds_pre_rename_root_by_stable_id(
    archive_layout: ArchiveLayout,
    catalog_engine: Engine,
) -> None:
    support, _ordinary, _markdown = _prepared_library(archive_layout, catalog_engine)
    with Session(catalog_engine) as session, session.begin():
        conversation = session.get(Conversation, SUPPORT_ID)
        assert conversation is not None
        conversation.title_raw = "Renamed Support Fixture"
        conversation.title_display = "Renamed Support Fixture"
        conversation.title_search = "renamed support fixture"

    summary = scrub_support_library(archive_layout, catalog_engine)

    assert summary.conversation_directories_found == 1
    assert summary.transcript_files_rewritten == 1
    assert IP_MARKER not in support.read_text(encoding="utf-8")


def test_scrub_support_library_allows_absent_deleted_conversation_root(
    archive_layout: ArchiveLayout,
    catalog_engine: Engine,
) -> None:
    support, _ordinary, _markdown = _prepared_library(archive_layout, catalog_engine)
    shutil.rmtree(support.parent.parent)
    with Session(catalog_engine) as session, session.begin():
        conversation = session.get(Conversation, SUPPORT_ID)
        assert conversation is not None
        conversation.deleted_at = datetime.now(UTC)

    summary = scrub_support_library(archive_layout, catalog_engine)

    assert summary.support_conversation_count == 1
    assert summary.conversation_directories_found == 0
    assert summary.transcript_files_scanned == 0


def test_scrub_support_library_only_repairs_messages_from_support_segments(
    archive_layout: ArchiveLayout,
    catalog_engine: Engine,
) -> None:
    support, _ordinary, _markdown = _prepared_library(archive_layout, catalog_engine)
    ordinary_segment_id = "1" * 64
    with Session(catalog_engine) as session, session.begin():
        session.add(
            ConversationSegment(
                id=ordinary_segment_id,
                conversation_id=SUPPORT_ID,
                thread_path="inbox/mixed-fixture",
                title_raw="Mixed Fixture",
                title_display="Mixed Fixture",
                title_search="mixed fixture",
                source_category="inbox",
                sequence=2,
                source_chunk_count=1,
                completeness="complete",
                metadata_json={},
            )
        )
    records = _records(support)
    records.extend(
        (
            {"record_type": "segment_boundary", "segment_id": ordinary_segment_id},
            {
                "record_type": "message",
                "message_id": "2" * 64,
                "segment_id": ordinary_segment_id,
                "extra": {"ipAddress": "198.51.100.ordinary-segment"},
                "sensitive_fields_omitted": [],
            },
        )
    )
    _write_ndjson(support, records, mode=0o644)

    summary = scrub_support_library(archive_layout, catalog_engine)

    messages = {
        record["message_id"]: record
        for record in _records(support)
        if record["record_type"] == "message"
    }
    assert summary.message_records_scanned == 1
    assert summary.message_records_changed == 1
    assert messages["e" * 64]["extra"] == {
        "to": "Archive Owner",
        "status": "open",
        "shipping": "preserved",
        "metadata": {"participant": "Support Agent"},
    }
    assert messages["2" * 64]["extra"] == {"ipAddress": "198.51.100.ordinary-segment"}


@pytest.mark.parametrize(
    "replacement",
    (
        b'{"record_type":\n',
        b'{"record_type":"unexpected"}\n',
        b'{"record_type":"message","extra":[],"sensitive_fields_omitted":[]}\n',
        (
            b'{"record_type":"message","segment_id":"'
            + b"9" * 64
            + b'","extra":{},"sensitive_fields_omitted":[]}\n'
        ),
    ),
)
def test_scrub_support_library_fails_closed_for_malformed_or_unexpected_records(
    archive_layout: ArchiveLayout,
    catalog_engine: Engine,
    replacement: bytes,
) -> None:
    support, _ordinary, _markdown = _prepared_library(archive_layout, catalog_engine)
    support.write_bytes(replacement)
    before = support.read_bytes()

    with pytest.raises(SupportLibraryScrubError):
        scrub_support_library(archive_layout, catalog_engine)

    assert support.read_bytes() == before
    assert not list(support.parent.glob(".support-library-scrub-*.tmp"))


def test_scrub_support_library_validates_all_files_before_publishing_any(
    archive_layout: ArchiveLayout,
    catalog_engine: Engine,
) -> None:
    support, _ordinary, _markdown = _prepared_library(archive_layout, catalog_engine)
    valid_before = support.read_bytes()
    malformed = support.with_name("1970-02.ndjson")
    malformed.write_bytes(b'{"record_type":\n')

    with pytest.raises(SupportLibraryScrubError):
        scrub_support_library(archive_layout, catalog_engine)

    assert support.read_bytes() == valid_before
    assert IP_MARKER in support.read_text(encoding="utf-8")
    assert stat.S_IMODE(support.stat().st_mode) == 0o644
    assert malformed.read_bytes() == b'{"record_type":\n'
    assert not list(support.parent.glob(".support-library-scrub-*.tmp"))


def test_scrub_support_library_refuses_hardlinked_transcripts(
    archive_layout: ArchiveLayout,
    catalog_engine: Engine,
) -> None:
    support, _ordinary, _markdown = _prepared_library(archive_layout, catalog_engine)
    second_link = archive_layout.root / "support-transcript-hardlink.ndjson"
    os.link(support, second_link)
    before = support.read_bytes()

    with pytest.raises(SupportLibraryScrubError, match="hard-linked"):
        scrub_support_library(archive_layout, catalog_engine)

    assert support.read_bytes() == before
    assert second_link.read_bytes() == before


def test_scrub_support_library_refuses_symlinked_transcripts_without_touching_target(
    archive_layout: ArchiveLayout,
    catalog_engine: Engine,
    tmp_path: Path,
) -> None:
    support, _ordinary, _markdown = _prepared_library(archive_layout, catalog_engine)
    outside = tmp_path / "outside-private-fixture.ndjson"
    outside.write_text(IP_MARKER, encoding="utf-8")
    support.unlink()
    support.symlink_to(outside)

    with pytest.raises(SupportLibraryScrubError, match="unsafe entry"):
        scrub_support_library(archive_layout, catalog_engine)

    assert support.is_symlink()
    assert outside.read_text(encoding="utf-8") == IP_MARKER
