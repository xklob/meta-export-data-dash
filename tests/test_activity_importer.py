from __future__ import annotations

import shutil
from dataclasses import fields, replace
from pathlib import Path

import pytest
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

from meta_archive.baseline import AuditedBaseline
from meta_archive.models import (
    ActivityRecord,
    AttachmentCandidate,
    AttachmentOccurrence,
    CoverageArea,
    ImportRun,
    IntegrityAnomaly,
    Job,
    SourceFile,
    SourceRecord,
)
from meta_archive.paths import ArchiveLayout
from meta_archive.services import verification
from meta_archive.services.activity_importer import (
    ActivitySourceIntegrityError,
    _normalized_local_uri,
    import_activities,
    inspect_activity_payload,
)
from meta_archive.services.consolidation import build_blob_store
from meta_archive.services.inventory import scan_archive


def _prepare_fixture_archive(layout: ArchiveLayout, engine: Engine) -> None:
    fixture_root = Path(__file__).parent / "fixtures" / "activity"
    shutil.copytree(fixture_root, layout.root / "package-a")
    scan_archive(layout, engine, hash_files=True)


def test_activity_import_adapts_all_supported_shapes_and_media(
    archive_layout: ArchiveLayout,
    catalog_engine: Engine,
) -> None:
    _prepare_fixture_archive(archive_layout, catalog_engine)

    summary = import_activities(archive_layout, catalog_engine)

    assert summary.files_considered == 9
    assert summary.files_imported == 7
    assert summary.files_invalid == 1
    assert summary.files_unsupported == 1
    assert summary.schema_skipped == 0
    assert summary.records_imported == 13
    assert summary.duplicate_records == 1
    assert summary.attachments_imported == 5
    assert summary.attachments_resolved == 4
    assert summary.attachments_missing == 1
    assert summary.attachments_external == 0

    with Session(catalog_engine) as session:
        assert session.scalar(select(func.count(ActivityRecord.pk))) == 13
        assert session.scalar(select(func.count(SourceRecord.id))) == 13

        repaired = session.scalar(
            select(ActivityRecord).where(ActivityRecord.source_fbid == "fixture-record-1")
        )
        assert repaired is not None
        assert repaired.timestamp_ms == 1_700_000_000_000
        assert repaired.title_raw == "CafÃ© fixture listing"
        assert repaired.title_display == "Café fixture listing"
        assert repaired.title_search == "café fixture listing"
        assert repaired.extra_json["raw_payload"]["fbid"] == "fixture-record-1"
        assert repaired.extra_json["provenance"] == {
            "source_file_id": repaired.source_file_id,
            "relative_path": ("your_facebook_activity/facebook_marketplace/generic_records.json"),
            "json_pointer": "/0",
        }

        unknown_time = session.scalar(
            select(ActivityRecord).where(ActivityRecord.source_fbid == "fixture-record-2")
        )
        assert unknown_time is not None
        assert unknown_time.timestamp_ms is None

        comment = session.scalar(
            select(ActivityRecord).where(ActivityRecord.subtype == "group_comment")
        )
        assert comment is not None
        assert comment.timestamp_ms == 1_700_000_401_000
        assert comment.actor_raw == "RenÃ©e Example"
        assert comment.actor_display == "Renée Example"
        assert comment.content_display == "A café comment containing only fixture text"
        assert comment.target_uri == "https://example.invalid/fixture"
        unavailable_comment = session.scalar(
            select(ActivityRecord).where(
                ActivityRecord.title_raw == "A metadata-only unavailable group comment"
            )
        )
        assert unavailable_comment is not None
        assert unavailable_comment.availability == "unavailable_body"
        assert unavailable_comment.content_raw is None

        album = session.scalar(select(ActivityRecord).where(ActivityRecord.subtype == "album"))
        assert album is not None
        album_children = list(
            session.scalars(
                select(ActivityRecord)
                .where(ActivityRecord.parent_id == album.id)
                .order_by(ActivityRecord.json_pointer)
            )
        )
        assert [child.subtype for child in album_children] == [
            "album_cover_photo",
            "album_photo",
        ]

        occurrences = list(
            session.scalars(select(AttachmentOccurrence).order_by(AttachmentOccurrence.id))
        )
        assert len(occurrences) == 5
        assert sum(row.availability == "resolved" for row in occurrences) == 4
        assert sum(row.availability == "missing" for row in occurrences) == 1
        resolved = [row for row in occurrences if row.availability == "resolved"]
        assert all(row.resolved_source_file_id is not None for row in resolved)
        assert all(row.blob_sha256 is not None for row in resolved)
        assert all(row.kind == "photo" for row in occurrences)
        assert session.scalar(select(func.count(AttachmentCandidate.id))) == 4
        assert (
            session.scalar(
                select(func.count(AttachmentCandidate.id)).where(
                    AttachmentCandidate.selected.is_(True)
                )
            )
            == 4
        )


def test_verifier_missing_attachment_baseline_includes_activity_occurrences(
    archive_layout: ArchiveLayout,
    catalog_engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_fixture_archive(archive_layout, catalog_engine)
    import_activities(archive_layout, catalog_engine)
    empty_baseline = AuditedBaseline(**{field.name: 0 for field in fields(AuditedBaseline)})
    monkeypatch.setattr(
        verification,
        "load_audited_baseline",
        lambda _root: replace(empty_baseline, unresolved_local_uri_occurrences=1),
    )

    checks = verification.verify_catalog(archive_layout, catalog_engine)

    missing = next(
        check for check in checks if check["key"] == "baseline.unresolved_local_uri_occurrences"
    )
    assert missing["actual"] == 1
    assert missing["status"] == "pass"
    assert "message and activity" in str(missing["detail"])


def test_activity_import_is_idempotent_and_tracks_duplicates_and_lifecycle(
    archive_layout: ArchiveLayout,
    catalog_engine: Engine,
) -> None:
    _prepare_fixture_archive(archive_layout, catalog_engine)
    first = import_activities(archive_layout, catalog_engine)
    second = import_activities(archive_layout, catalog_engine)

    assert first.records_imported == 13
    assert second.records_imported == 0
    assert second.files_imported == 0
    assert second.files_skipped == 9

    with Session(catalog_engine) as session:
        assert session.scalar(select(func.count(ActivityRecord.pk))) == 13
        assert session.scalar(select(func.count(SourceRecord.id))) == 13
        assert session.scalar(select(func.count(AttachmentOccurrence.id))) == 5

        duplicate = session.scalar(
            select(SourceRecord).where(SourceRecord.duplicate_of_id.is_not(None))
        )
        assert duplicate is not None
        assert duplicate.record_type == "activity:reaction"
        assert duplicate.duplicate_of_id != duplicate.id

        runs = list(
            session.scalars(
                select(ImportRun).where(ImportRun.kind == "activity").order_by(ImportRun.started_at)
            )
        )
        assert len(runs) == 2
        assert all(run.status == "completed" for run in runs)
        jobs = list(session.scalars(select(Job).where(Job.kind == "activity_import")))
        assert len(jobs) == 2
        assert all(job.status == "completed" for job in jobs)

        statuses = {
            Path(path).name: status
            for path, status in session.execute(
                select(SourceFile.relative_path, SourceFile.json_status)
            )
        }
        assert statuses["invalid.json"] == "invalid"
        assert statuses["unsupported.json"] == "unsupported"
        assert statuses["message_1.json"] == "pending"
        assert statuses["support_messages.json"] == "pending"
        assert statuses["generic_records.json"] == "parsed"

        unrelated_file_id = session.scalar(
            select(SourceFile.id).where(SourceFile.relative_path.endswith("unsupported.json"))
        )
        assert unrelated_file_id is not None
        assert (
            session.scalar(
                select(func.count(SourceRecord.id)).where(
                    SourceRecord.source_file_id == unrelated_file_id
                )
            )
            == 0
        )

        anomaly_codes = {
            code: count
            for code, count in session.execute(
                select(IntegrityAnomaly.code, func.count(IntegrityAnomaly.id)).group_by(
                    IntegrityAnomaly.code
                )
            )
        }
        assert anomaly_codes["invalid_activity_json"] == 1
        assert anomaly_codes["unsupported_activity_schema"] == 1
        assert anomaly_codes["missing_activity_attachment"] == 1


def test_activity_coverage_never_treats_absent_history_as_zero(
    archive_layout: ArchiveLayout,
    catalog_engine: Engine,
) -> None:
    _prepare_fixture_archive(archive_layout, catalog_engine)
    import_activities(archive_layout, catalog_engine)

    with Session(catalog_engine) as session:
        coverage = {row.key: row for row in session.scalars(select(CoverageArea))}

    for key in ("friends", "events", "general_posts"):
        assert coverage[key].status == "unavailable"
        assert coverage[key].raw_count is None
        assert coverage[key].logical_count is None

    assert coverage["posts"].status == "partial"
    assert coverage["posts"].raw_count == 3
    assert coverage["albums"].status == "available"
    assert coverage["albums"].raw_count == 3
    assert coverage["reactions"].status == "available"
    assert coverage["reactions"].raw_count == 3
    assert coverage["reactions"].logical_count == 2
    assert coverage["group_comments"].raw_count == 2
    assert coverage["group_membership"].raw_count == 1


def test_forced_activity_reimport_replaces_each_file_without_growth(
    archive_layout: ArchiveLayout,
    catalog_engine: Engine,
) -> None:
    _prepare_fixture_archive(archive_layout, catalog_engine)
    import_activities(archive_layout, catalog_engine)

    forced = import_activities(archive_layout, catalog_engine, force=True)

    assert forced.files_imported == 7
    assert forced.files_invalid == 1
    assert forced.files_unsupported == 1
    assert forced.records_imported == 13
    with Session(catalog_engine) as session:
        assert session.scalar(select(func.count(ActivityRecord.pk))) == 13
        assert session.scalar(select(func.count(SourceRecord.id))) == 13
        assert session.scalar(select(func.count(AttachmentOccurrence.id))) == 5


def test_activity_import_reads_verified_blob_after_original_entry_is_retired(
    archive_layout: ArchiveLayout,
    catalog_engine: Engine,
) -> None:
    _prepare_fixture_archive(archive_layout, catalog_engine)
    build_blob_store(archive_layout, catalog_engine)
    source = archive_layout.root / "package-a" / "preferences" / "preferences" / "settings.json"
    source.unlink()

    summary = import_activities(archive_layout, catalog_engine)

    assert summary.records_imported == 13
    with Session(catalog_engine) as session:
        setting = session.scalar(
            select(ActivityRecord).where(ActivityRecord.source_fbid == "fixture-setting")
        )
        assert setting is not None
        assert setting.content_display == "Fixture preference: Enabled"


def test_activity_import_rejects_corrupt_catalogued_source_blob(
    archive_layout: ArchiveLayout,
    catalog_engine: Engine,
) -> None:
    _prepare_fixture_archive(archive_layout, catalog_engine)
    build_blob_store(archive_layout, catalog_engine)
    source = archive_layout.root / "package-a" / "preferences" / "preferences" / "settings.json"
    with Session(catalog_engine) as session:
        digest = session.scalar(
            select(SourceFile.sha256).where(
                SourceFile.relative_path == "preferences/preferences/settings.json"
            )
        )
    assert digest is not None
    source.unlink()
    blob_path = archive_layout.blob_path(digest)
    # Materialized blobs are deliberately read-only. A later controlled purge/repair path must
    # explicitly restore owner write permission before replacing one.
    blob_path.chmod(0o600)
    blob_path.write_bytes(b"corrupt synthetic fixture")

    with pytest.raises(ActivitySourceIntegrityError, match="digest mismatch"):
        import_activities(archive_layout, catalog_engine)

    with Session(catalog_engine) as session:
        failed_run = session.scalar(
            select(ImportRun)
            .where(ImportRun.kind == "activity")
            .order_by(ImportRun.started_at.desc())
        )
        assert failed_run is not None
        assert failed_run.status == "failed"


@pytest.mark.parametrize(
    "unsafe_uri",
    [
        "/etc/passwd",
        "%2Fetc/passwd",
        "../outside.jpg",
        "safe/../outside.jpg",
        "safe/./image.jpg",
        "C:\\archive\\image.jpg",
        "safe/image.jpg\x00.png",
    ],
)
def test_activity_media_uri_rejects_absolute_drive_nul_and_dot_segments(
    unsafe_uri: str,
) -> None:
    assert _normalized_local_uri(unsafe_uri) is None


def test_support_message_object_schema_is_identified_but_not_adapted() -> None:
    result = inspect_activity_payload(
        {"support_messages": {"fixture-case": {"messages": []}}},
        "your_facebook_activity/messages/support_messages.json",
    )

    assert result == {
        "schema": "support_messages",
        "record_count": 0,
        "attachment_count": 0,
    }


def test_activity_source_coordinate_must_remain_inside_its_chunk(
    archive_layout: ArchiveLayout,
    catalog_engine: Engine,
) -> None:
    _prepare_fixture_archive(archive_layout, catalog_engine)
    source = archive_layout.root / "package-a" / "preferences" / "preferences" / "settings.json"
    outside = archive_layout.root.parent / "outside-activity-fixture.json"
    outside.write_text('{"media": [], "label_values": [], "fbid": "outside"}', encoding="utf-8")
    source.unlink()
    source.symlink_to(outside)

    with pytest.raises(ActivitySourceIntegrityError, match="escapes its chunk"):
        import_activities(archive_layout, catalog_engine)
