from __future__ import annotations

import errno
import hashlib
import json
import multiprocessing
import os
import shutil
import stat
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine, func, select, update
from sqlalchemy.orm import Session

from meta_archive.database import create_catalog_engine, run_migrations
from meta_archive.models import AttachmentOccurrence, Message, SourceFile
from meta_archive.paths import ArchiveLayout
from meta_archive.services.activity_importer import import_activities
from meta_archive.services.consolidation import build_blob_store
from meta_archive.services.hashing import sha256_file
from meta_archive.services.inventory import scan_archive
from meta_archive.services.library import (
    UNLINKED_GENERATOR,
    LibraryBuildBusyError,
    LibraryBuildError,
    _BlobMaterializer,
    _BlobMetadata,
    _conversation_bucket,
    build_filesystem_library,
    library_build_lock,
    recover_library_build_state,
)
from meta_archive.services.message_importer import import_messages

FIXTURES = Path(__file__).parent / "fixtures"


def _prepared_archive(tmp_path: Path) -> tuple[ArchiveLayout, Engine]:
    layout = ArchiveLayout(tmp_path / "meta-backup-library")
    shutil.copytree(FIXTURES / "messages", layout.root)
    shutil.copytree(FIXTURES / "activity", layout.root / "package-c")
    loose = layout.root / "package-b" / "your_facebook_activity" / "posts" / "loose media.jpg"
    loose.parent.mkdir(parents=True, exist_ok=True)
    loose.write_bytes(b"unlinked library fixture bytes")
    duplicate_loose = (
        layout.root / "package-c" / "your_facebook_activity" / "posts" / "loose media.jpg"
    )
    duplicate_loose.parent.mkdir(parents=True, exist_ok=True)
    duplicate_loose.write_bytes(loose.read_bytes())
    non_media = (
        layout.root
        / "package-b"
        / "your_facebook_activity"
        / "messages"
        / "inbox"
        / "fixture"
        / "files"
        / "unlinked.java"
    )
    non_media.parent.mkdir(parents=True, exist_ok=True)
    non_media.write_text("class Fixture {}\n", encoding="utf-8")

    layout.ensure_runtime_directories()
    run_migrations(layout.database)
    engine = create_catalog_engine(layout.database)
    scan_archive(layout, engine, hash_files=True)
    build_blob_store(layout, engine)
    import_messages(layout, engine)
    import_activities(layout, engine)
    for chunk in layout.chunk_directories():
        shutil.rmtree(chunk)
    return layout, engine


def _ndjson(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _tree_snapshot(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _hold_library_build_lock(root: str, ready: Any, release: Any) -> None:
    layout = ArchiveLayout(Path(root))
    with library_build_lock(layout):
        (layout.root / ".library-build-active-fixture").mkdir()
        ready.set()
        if not release.wait(10):
            raise RuntimeError("library lock fixture timed out")


def _wait_for_library_build_lock(root: str, acquired: Any) -> None:
    layout = ArchiveLayout(Path(root))
    with library_build_lock(layout):
        acquired.set()


def test_library_build_lock_serializes_processes_and_preserves_active_staging(
    tmp_path: Path,
) -> None:
    layout = ArchiveLayout(tmp_path / "meta-backup-library-lock")
    context = multiprocessing.get_context("fork")
    ready = context.Event()
    release = context.Event()
    acquired = context.Event()
    holder = context.Process(
        target=_hold_library_build_lock,
        args=(str(layout.root), ready, release),
    )
    waiter = context.Process(
        target=_wait_for_library_build_lock,
        args=(str(layout.root), acquired),
    )
    holder.start()
    try:
        assert ready.wait(10)
        active_staging = layout.root / ".library-build-active-fixture"
        assert active_staging.is_dir()
        with (
            pytest.raises(LibraryBuildBusyError, match="publication is active"),
            library_build_lock(layout, blocking=False),
        ):
            pytest.fail("a second process acquired an active library lock")
        assert active_staging.is_dir()

        waiter.start()
        assert not acquired.wait(0.25)
        assert active_staging.is_dir()
        release.set()
        holder.join(10)
        assert holder.exitcode == 0
        assert acquired.wait(10)
        waiter.join(10)
        assert waiter.exitcode == 0

        lock_path = layout.catalog / "library-build.lock"
        assert stat.S_IMODE(lock_path.stat().st_mode) == 0o600
        recover_library_build_state(layout)
        assert not active_staging.exists()
    finally:
        release.set()
        for process in (holder, waiter):
            if process.is_alive():
                process.terminate()
            process.join(10)


def test_library_build_lock_refuses_hardlink_without_mutating_external_inode(
    tmp_path: Path,
) -> None:
    layout = ArchiveLayout(tmp_path / "meta-backup-library-lock-hardlink")
    layout.catalog.mkdir(parents=True)
    external = tmp_path / "external-lock-target"
    external.write_bytes(b"external bytes must remain unchanged")
    external.chmod(0o644)
    lock_path = layout.catalog / "library-build.lock"
    os.link(external, lock_path)
    before = external.read_bytes()
    before_mode = stat.S_IMODE(external.stat().st_mode)

    with (
        pytest.raises(LibraryBuildError, match="single-link"),
        library_build_lock(layout),
    ):
        pytest.fail("a hard-linked library lock was accepted")

    assert external.read_bytes() == before
    assert stat.S_IMODE(external.stat().st_mode) == before_mode
    assert lock_path.stat().st_ino == external.stat().st_ino


def test_library_build_recovery_restores_single_previous_tree_and_cleans_staging(
    tmp_path: Path,
) -> None:
    layout = ArchiveLayout(tmp_path / "meta-backup-library-recovery")
    layout.root.mkdir()
    previous = layout.root / ".library-previous-single"
    previous.mkdir()
    (previous / "preserved.txt").write_text("preserved", encoding="utf-8")
    staged = layout.root / ".library-build-abandoned"
    staged.mkdir()
    (staged / "partial.txt").write_text("partial", encoding="utf-8")

    recover_library_build_state(layout)

    assert (layout.library / "preserved.txt").read_text(encoding="utf-8") == "preserved"
    assert not previous.exists()
    assert not staged.exists()


def test_library_build_recovery_keeps_current_tree_and_removes_old_crash_roots(
    tmp_path: Path,
) -> None:
    layout = ArchiveLayout(tmp_path / "meta-backup-library-current")
    layout.root.mkdir()
    layout.library.mkdir()
    (layout.library / "current.txt").write_text("current", encoding="utf-8")
    previous = layout.root / ".library-previous-obsolete"
    previous.mkdir()
    (previous / "old.txt").write_text("old", encoding="utf-8")
    staged = layout.root / ".library-build-obsolete"
    staged.mkdir()

    recover_library_build_state(layout)

    assert (layout.library / "current.txt").read_text(encoding="utf-8") == "current"
    assert not previous.exists()
    assert not staged.exists()


def test_library_build_recovery_fails_closed_for_ambiguous_or_unsafe_roots(
    tmp_path: Path,
) -> None:
    ambiguous = ArchiveLayout(tmp_path / "meta-backup-library-ambiguous")
    ambiguous.root.mkdir()
    previous_a = ambiguous.root / ".library-previous-a"
    previous_b = ambiguous.root / ".library-previous-b"
    staged = ambiguous.root / ".library-build-preserved"
    for path in (previous_a, previous_b, staged):
        path.mkdir()

    with pytest.raises(LibraryBuildError, match="multiple previous trees"):
        recover_library_build_state(ambiguous)
    assert all(path.is_dir() for path in (previous_a, previous_b, staged))
    assert not ambiguous.library.exists()

    unsafe = ArchiveLayout(tmp_path / "meta-backup-library-unsafe")
    unsafe.root.mkdir()
    outside = tmp_path / "outside-library-fixture"
    outside.mkdir()
    unsafe_staging = unsafe.root / ".library-build-symlink"
    unsafe_staging.symlink_to(outside, target_is_directory=True)

    with pytest.raises(LibraryBuildError, match="not a real directory"):
        recover_library_build_state(unsafe)
    assert unsafe_staging.is_symlink()
    assert outside.is_dir()


def test_builds_deterministic_library_from_catalog_and_blobs_after_source_retirement(
    tmp_path: Path,
) -> None:
    layout, engine = _prepared_archive(tmp_path)
    try:
        legacy_ip_marker = "203.0.113.legacy-library-fixture"
        legacy_payment_marker = "legacy-payment-library-fixture"
        with Session(engine) as session, session.begin():
            legacy_support = session.scalar(
                select(Message).where(Message.content_raw == "Newest support reply")
            )
            assert legacy_support is not None
            legacy_support.extra_json = {
                **legacy_support.extra_json,
                "category": "account support",
                "status": "open",
                "legacy_metadata": {
                    "topic": "shipping",
                    "participant": "Support Agent",
                    "network": {"ip_address": legacy_ip_marker},
                    "history": [
                        {"status": "received"},
                        {
                            "payment_information": {"account_credentials": legacy_payment_marker},
                            "status": "reviewed",
                        },
                    ],
                },
            }

        first = build_filesystem_library(layout, engine)

        assert first.conversation_count > 0
        assert first.message_count > 0
        assert first.transcript_file_count > 0
        assert first.activity_record_count == 13
        assert first.hardlink_count > 0
        assert not layout.chunk_directories()

        root_manifest = json.loads((layout.library / "manifest.json").read_text())
        assert root_manifest["format_version"] == 1
        assert root_manifest["counts"]["message_count"] == first.message_count
        assert set(root_manifest["source_manifests"]) == {
            "original_files",
            "original_directories",
            "relocation",
        }
        assert all(
            len(record["sha256"]) == 64 for record in root_manifest["source_manifests"].values()
        )

        conversation_index = _ndjson(layout.library / "conversations" / "manifest.ndjson")
        alex = next(
            record for record in conversation_index if record["title_display"] == "Alex Example"
        )
        alex_root = layout.library / str(alex["library_path"])
        assert alex_root.parent.name == "direct"
        assert len(str(alex_root.name).rsplit("--", 1)[-1]) == 64
        transcript_files = sorted((alex_root / "transcripts").glob("*.ndjson"))
        assert transcript_files
        transcript = [record for path in transcript_files for record in _ndjson(path)]
        boundaries = [
            record for record in transcript if record["record_type"] == "segment_boundary"
        ]
        messages = [record for record in transcript if record["record_type"] == "message"]
        assert len({record["segment_id"] for record in boundaries}) == 2
        assert all(len(str(record["message_id"])) == 64 for record in messages)
        assert all(record["source"] is not None for record in messages)
        assert all("sensitive_json" not in record for record in messages)
        sensitive = next(record for record in messages if record["has_payment"] is True)
        assert sensitive["sensitive_fields_omitted"] == ["ip", "payment_info"]
        markdown = "\n".join(path.read_text() for path in (alex_root / "transcripts").glob("*.md"))
        assert f"message-{sensitive['message_id']}" in markdown

        all_transcript_messages = [
            record
            for path in (layout.library / "conversations").rglob("transcripts/*.ndjson")
            for record in _ndjson(path)
            if record.get("record_type") == "message"
        ]
        support = next(
            record
            for record in all_transcript_messages
            if record.get("content_raw") == "Newest support reply"
        )
        assert support["extra"] == {
            "to": "Archive Owner",
            "subject": "Support Case One",
            "category": "account support",
            "status": "open",
            "legacy_metadata": {
                "topic": "shipping",
                "participant": "Support Agent",
                "network": {},
                "history": [
                    {"status": "received"},
                    {"status": "reviewed"},
                ],
            },
        }
        assert support["sensitive_fields_omitted"] == ["ip_address", "payment_information"]
        all_ndjson = "\n".join(
            path.read_text(encoding="utf-8") for path in layout.library.rglob("*.ndjson")
        )
        all_markdown = "\n".join(
            path.read_text(encoding="utf-8") for path in layout.library.rglob("*.md")
        )
        for marker in (legacy_ip_marker, legacy_payment_marker):
            assert marker not in all_ndjson
            assert marker not in all_markdown

        assert any((layout.library / "conversations" / "groups").iterdir())
        assert any((layout.library / "conversations" / "system").iterdir())
        assert (layout.library / "conversations" / "unknown").is_dir()

        attachment_manifest = _ndjson(alex_root / "attachments" / "manifest.ndjson")
        linked_attachment = next(
            record
            for record in attachment_manifest
            if record["library_path"] and record["message_id"] is not None
        )
        linked_path = layout.library / str(linked_attachment["library_path"])
        blob_path = layout.blob_path(str(linked_attachment["blob_sha256"]))
        assert linked_path.stat().st_ino == blob_path.stat().st_ino
        assert linked_attachment["message_id"] is not None
        assert linked_attachment["source"] is not None
        assert linked_attachment["resolved_source"] is not None

        with Session(engine) as session:
            loose_sources = list(
                session.scalars(
                    select(SourceFile).where(SourceFile.relative_path.endswith("loose media.jpg"))
                )
            )
            assert len(loose_sources) == 2
            generated_rows = list(
                session.scalars(
                    select(AttachmentOccurrence).where(
                        AttachmentOccurrence.source_file_id.in_(
                            [source.id for source in loose_sources]
                        ),
                        AttachmentOccurrence.availability == "unlinked",
                    )
                )
            )
            assert len(generated_rows) == 1
            generated = generated_rows[0]
            assert generated.metadata_json["source_copy_count"] == 2
            assert {
                item["source_file_id"] for item in generated.metadata_json["source_copies"]
            } == {source.id for source in loose_sources}
            selected_source = session.get(SourceFile, generated.source_file_id)
            assert selected_source is not None
            assert selected_source.id in {source.id for source in loose_sources}
            assert generated.resolved_source_file_id == selected_source.id
            assert generated.blob_sha256 == selected_source.sha256
            assert generated.message_id is None
            assert generated.activity_record_id is None
            assert generated.conversation_id is None
            assert generated.metadata_json["generated_by"] == UNLINKED_GENERATOR
            generated_id = generated.id
            non_media_source = session.scalar(
                select(SourceFile).where(SourceFile.relative_path.endswith("unlinked.java"))
            )
            assert non_media_source is not None
            assert (
                session.scalar(
                    select(func.count(AttachmentOccurrence.id)).where(
                        AttachmentOccurrence.source_file_id == non_media_source.id
                    )
                )
                == 0
            )
            generated_count = session.scalar(
                select(func.count(AttachmentOccurrence.id)).where(
                    AttachmentOccurrence.availability == "unlinked"
                )
            )

        unlinked_records = _ndjson(layout.library / "media" / "unlinked" / "manifest.ndjson")
        loose_record = next(
            record for record in unlinked_records if record["attachment_id"] == generated_id
        )
        assert loose_record["timestamp_ms"] is None
        assert loose_record["date_status"] == "unknown"
        assert loose_record["availability"] == "unlinked"
        assert "unknown-date" in str(loose_record["library_path"])

        posts = _ndjson(layout.library / "posts" / "records.ndjson")
        comments = _ndjson(layout.library / "comments" / "records.ndjson")
        events_coverage = json.loads((layout.library / "events" / "coverage.json").read_text())
        people_coverage = json.loads((layout.library / "people" / "coverage.json").read_text())
        assert {record["subtype"] for record in posts} >= {"album", "album_photo"}
        assert {record["subtype"] for record in comments} == {"group_comment"}
        assert any(
            record["key"] == "events" and record["status"] == "unavailable"
            for record in events_coverage
        )
        assert any(
            record["key"] == "friends" and record["status"] == "unavailable"
            for record in people_coverage
        )

        before = _tree_snapshot(layout.library)
        second = build_filesystem_library(layout, engine)
        after = _tree_snapshot(layout.library)
        assert second == first
        assert after == before
        with Session(engine) as session:
            assert (
                session.scalar(
                    select(func.count(AttachmentOccurrence.id)).where(
                        AttachmentOccurrence.availability == "unlinked"
                    )
                )
                == generated_count
            )
            assert session.get(AttachmentOccurrence, generated_id) is not None
    finally:
        engine.dispose()


def test_unlinked_reconciliation_removes_only_stale_generated_rows(tmp_path: Path) -> None:
    layout, engine = _prepared_archive(tmp_path)
    try:
        build_filesystem_library(layout, engine)
        with Session(engine) as session, session.begin():
            sources = list(
                session.scalars(
                    select(SourceFile).where(SourceFile.relative_path.endswith("loose media.jpg"))
                )
            )
            assert len(sources) == 2
            generated_id = next(
                row.id
                for row in session.scalars(
                    select(AttachmentOccurrence).where(
                        AttachmentOccurrence.source_file_id.in_([source.id for source in sources])
                    )
                )
                if row.metadata_json.get("generated_by") == UNLINKED_GENERATOR
            )
            session.execute(
                update(SourceFile)
                .where(SourceFile.id.in_([source.id for source in sources]))
                .values(scan_status="not_seen")
            )

        build_filesystem_library(layout, engine)
        with Session(engine) as session:
            assert session.get(AttachmentOccurrence, generated_id) is None
            importer_occurrences = list(
                session.scalars(
                    select(AttachmentOccurrence).where(
                        AttachmentOccurrence.availability != "unlinked"
                    )
                )
            )
            assert importer_occurrences
            assert all(
                row.metadata_json.get("generated_by") != UNLINKED_GENERATOR
                for row in importer_occurrences
            )
    finally:
        engine.dispose()


def test_failed_blob_validation_keeps_previous_library_tree(tmp_path: Path) -> None:
    layout, engine = _prepared_archive(tmp_path)
    try:
        build_filesystem_library(layout, engine)
        before = _tree_snapshot(layout.library)
        with Session(engine) as session:
            digest = session.scalar(
                select(SourceFile.sha256).where(
                    SourceFile.relative_path.endswith("loose media.jpg")
                )
            )
        assert digest is not None
        corrupt_blob = layout.blob_path(digest)
        corrupt_blob.unlink()
        corrupt_blob.write_bytes(b"corrupt fixture blob")

        with pytest.raises(LibraryBuildError, match=r"(size|digest) mismatch"):
            build_filesystem_library(layout, engine)

        assert (layout.library / "manifest.json").is_file()
        # The corrupt hardlinked media itself changes, but no partially staged paths are promoted.
        assert set(_tree_snapshot(layout.library)) == set(before)
        assert not list(layout.root.glob(".library-build-*"))
    finally:
        engine.dispose()


def test_blob_materializer_falls_back_to_verified_copy_without_mutating_blob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = ArchiveLayout(tmp_path / "meta-backup-copy")
    layout.ensure_runtime_directories()
    content = b"immutable source blob"
    digest = hashlib.sha256(content).hexdigest()
    blob = layout.blob_path(digest)
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(content)
    original_mode = blob.stat().st_mode
    materializer = _BlobMaterializer(
        layout,
        {
            digest: _BlobMetadata(
                sha256=digest,
                size_bytes=len(content),
                media_type="application/octet-stream",
                extension=None,
            )
        },
    )

    def cross_device_link(_source: Path, _destination: Path, **_kwargs: object) -> None:
        raise OSError(errno.EXDEV, "synthetic cross-device link")

    monkeypatch.setattr("meta_archive.services.library.os.link", cross_device_link)
    destination = tmp_path / "generated" / "copy.bin"
    materializer.link(digest, destination)

    assert destination.read_bytes() == content
    assert destination.stat().st_ino != blob.stat().st_ino
    assert blob.read_bytes() == content
    assert blob.stat().st_mode == original_mode
    assert materializer.hardlinks == 0
    assert materializer.copies == 1


@pytest.mark.parametrize(
    ("conversation_type", "expected"),
    [
        ("direct", "direct"),
        ("group", "groups"),
        ("system", "system"),
        ("solo", "unknown"),
        ("future-type", "unknown"),
    ],
)
def test_conversation_taxonomy_is_total(conversation_type: str, expected: str) -> None:
    assert _conversation_bucket(conversation_type) == expected
