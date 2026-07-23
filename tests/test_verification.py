from __future__ import annotations

import json
from dataclasses import fields

from sqlalchemy import Engine
from sqlalchemy.orm import Session

from meta_archive.baseline import (
    AUDITED_BASELINE_MANIFEST_RELATIVE_PATH,
    AuditedBaseline,
    archive_identity_sha256,
)
from meta_archive.models import (
    Conversation,
    ConversationCorrection,
    ConversationCorrectionSegment,
    ConversationSegment,
    ConversationSegmentOverride,
    Star,
)
from meta_archive.paths import ArchiveLayout
from meta_archive.services.inventory import scan_archive
from meta_archive.services.verification import (
    SOURCE_RETIREMENT_REQUIRED_CHECK_KEYS,
    mandatory_non_pass_checks,
    verify_catalog,
)


def _write_zero_baseline(layout: ArchiveLayout) -> None:
    baseline = AuditedBaseline(**{field.name: 0 for field in fields(AuditedBaseline)})
    path = layout.root / AUDITED_BASELINE_MANIFEST_RELATIVE_PATH
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "archive_identity_sha256": archive_identity_sha256(layout.root),
                "metrics": {
                    field.name: getattr(baseline, field.name) for field in fields(AuditedBaseline)
                },
            }
        ),
        encoding="utf-8",
    )
    path.chmod(0o600)


def _verification_conversation(identifier: str, title: str) -> Conversation:
    return Conversation(
        id=identifier,
        conversation_key=f"fixture:{identifier}",
        title_raw=title,
        title_display=title,
        title_search=title.casefold(),
        conversation_type="direct",
        primary_category="inbox",
    )


def _conversation_snapshot(identifier: str, title: str) -> dict[str, object]:
    return {
        "id": identifier,
        "conversation_key": f"fixture:{identifier}",
        "title_raw": title,
        "title_display": title,
        "title_search": title.casefold(),
        "conversation_type": "direct",
        "primary_category": "inbox",
        "participant_count": 0,
        "first_activity_ms": None,
        "last_activity_ms": None,
        "message_count": 0,
        "sent_count": 0,
        "received_count": 0,
        "attachment_count": 0,
        "unique_attachment_count": 0,
        "attachment_bytes": 0,
        "active_days": 0,
        "completeness": "unknown",
        "deleted_at": None,
    }


def test_synthetic_archives_do_not_inherit_real_export_baselines(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> None:
    source = archive_layout.root / "package-a" / "activity.json"
    source.parent.mkdir(parents=True)
    source.write_text("{}", encoding="utf-8")
    scan_archive(archive_layout, catalog_engine, hash_files=True)

    checks = verify_catalog(archive_layout, catalog_engine)

    assert not any(str(check["key"]).startswith("baseline.") for check in checks)
    assert not any(check["status"] == "fail" for check in checks)


def test_verification_can_require_missing_audited_baseline(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> None:
    checks = verify_catalog(
        archive_layout,
        catalog_engine,
        require_audited_baseline=True,
    )

    manifest = next(check for check in checks if check["key"] == "baseline.manifest")
    assert manifest["status"] == "fail"
    assert manifest["actual"] == "missing"
    assert manifest["mandatory"] is True


def test_verification_loads_valid_archive_local_audited_baseline(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> None:
    _write_zero_baseline(archive_layout)

    checks = verify_catalog(archive_layout, catalog_engine)

    manifest = next(check for check in checks if check["key"] == "baseline.manifest")
    assert manifest["status"] == "pass"
    json_files = next(check for check in checks if check["key"] == "baseline.json_files")
    assert json_files["expected"] == 0


def test_verification_reports_malformed_audited_baseline_instead_of_disabling_it(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> None:
    path = archive_layout.root / AUDITED_BASELINE_MANIFEST_RELATIVE_PATH
    path.write_text("{}", encoding="utf-8")
    path.chmod(0o600)

    checks = verify_catalog(archive_layout, catalog_engine)

    manifest = next(check for check in checks if check["key"] == "baseline.manifest")
    assert manifest["status"] == "fail"
    assert manifest["actual"] == "invalid"
    assert manifest["mandatory"] is True
    assert not any(check["key"] == "baseline.json_files" for check in checks)


def test_deep_verification_reports_missing_manifest_instead_of_raising(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> None:
    checks = verify_catalog(archive_layout, catalog_engine, deep=True)

    deep = next(check for check in checks if check["key"] == "deep.source_bytes")
    assert deep["status"] == "fail"
    assert "original-files.jsonl" in str(deep["detail"])


def test_verification_reports_private_permission_drift_without_blocking_integrity(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> None:
    archive_layout.analytics_cache.chmod(0o755)

    checks = verify_catalog(archive_layout, catalog_engine)

    permissions = next(check for check in checks if check["key"] == "runtime.private_permissions")
    assert permissions["status"] == "pending"
    assert permissions["mandatory"] is False
    assert "not owner-only" in str(permissions["detail"])
    assert permissions not in mandatory_non_pass_checks(checks)


def test_mandatory_non_pass_checks_fail_closed_for_pending_and_missing_keys() -> None:
    checks = [
        {"key": key, "status": "pass", "mandatory": True}
        for key in sorted(SOURCE_RETIREMENT_REQUIRED_CHECK_KEYS)
    ]
    assert mandatory_non_pass_checks(checks) == []

    checks[0]["status"] = "pending"
    non_pass = mandatory_non_pass_checks(checks[:-1])
    assert any(check["status"] == "pending" for check in non_pass)
    assert any(check["status"] == "missing" for check in non_pass)


def test_verification_fails_for_an_unresolvable_star_target(
    archive_layout: ArchiveLayout,
    catalog_engine: Engine,
) -> None:
    with Session(catalog_engine) as session, session.begin():
        session.add(
            Star(
                id="missing-star-target",
                target_type="message",
                target_id="f" * 64,
            )
        )

    checks = verify_catalog(archive_layout, catalog_engine)

    stars = next(check for check in checks if check["key"] == "stars.targets")
    assert stars["status"] == "fail"
    assert stars["expected"] == 0
    assert stars["actual"] == 1
    assert stars["detail"] == "active=0, latent=0, missing=1"


def test_globally_broken_merge_correction_does_not_mask_missing_star(
    archive_layout: ArchiveLayout,
    catalog_engine: Engine,
) -> None:
    before_a = "a" * 64
    before_b = "b" * 64
    destination_id = "c" * 64
    segment_a = "d" * 64
    segment_b = "e" * 64
    correction_id = "broken-multi-source-merge"
    with Session(catalog_engine) as session, session.begin():
        session.add(_verification_conversation(destination_id, "Merged"))
        session.flush()
        session.add_all(
            [
                ConversationSegment(
                    id=segment_a,
                    conversation_id=destination_id,
                    thread_path="inbox/a",
                    title_raw="A",
                    title_display="A",
                    title_search="a",
                    source_category="inbox",
                ),
                ConversationSegment(
                    id=segment_b,
                    conversation_id=destination_id,
                    thread_path="inbox/b",
                    title_raw="B",
                    title_display="B",
                    title_search="b",
                    source_category="inbox",
                ),
                ConversationCorrection(
                    id=correction_id,
                    action="merge",
                    status="active",
                    request_json={},
                    before_json={
                        "conversations": [
                            _conversation_snapshot(before_a, "A"),
                            _conversation_snapshot(before_b, "B"),
                        ]
                    },
                    after_json={},
                    catalog_revision="1" * 64,
                    result_revision="2" * 64,
                    effect_json={},
                ),
                Star(
                    id="star-before-a",
                    target_type="conversation",
                    target_id=before_a,
                ),
            ]
        )
        session.flush()
        session.add_all(
            [
                ConversationCorrectionSegment(
                    correction_id=correction_id,
                    segment_id=segment_a,
                    thread_path="inbox/a",
                    before_conversation_id=before_a,
                    after_conversation_id=destination_id,
                ),
                ConversationCorrectionSegment(
                    correction_id=correction_id,
                    segment_id=segment_b,
                    thread_path="inbox/b",
                    before_conversation_id=before_b,
                    after_conversation_id=destination_id,
                ),
                ConversationSegmentOverride(
                    segment_id=segment_a,
                    base_conversation_id=before_a,
                    conversation_id=destination_id,
                    correction_id=correction_id,
                ),
            ]
        )

    checks = verify_catalog(archive_layout, catalog_engine)

    stars = next(check for check in checks if check["key"] == "stars.targets")
    assert stars["status"] == "fail"
    assert stars["detail"] == "active=0, latent=0, missing=1"


def test_chained_merge_star_is_latent_when_actual_revert_preconditions_hold(
    archive_layout: ArchiveLayout,
    catalog_engine: Engine,
) -> None:
    base_id = "1" * 64
    before_id = "2" * 64
    destination_id = "3" * 64
    segment_id = "4" * 64
    correction_id = "chained-merge"
    with Session(catalog_engine) as session, session.begin():
        session.add(_verification_conversation(destination_id, "Destination"))
        session.flush()
        session.add_all(
            [
                ConversationSegment(
                    id=segment_id,
                    conversation_id=destination_id,
                    thread_path="inbox/chained",
                    title_raw="Chained",
                    title_display="Chained",
                    title_search="chained",
                    source_category="inbox",
                ),
                ConversationCorrection(
                    id=correction_id,
                    action="merge",
                    status="active",
                    request_json={},
                    before_json={
                        "conversations": [
                            _conversation_snapshot(before_id, "Intermediate"),
                            _conversation_snapshot(destination_id, "Destination"),
                        ]
                    },
                    after_json={},
                    catalog_revision="5" * 64,
                    result_revision="6" * 64,
                    effect_json={},
                ),
                Star(
                    id="star-intermediate",
                    target_type="conversation",
                    target_id=before_id,
                ),
            ]
        )
        session.flush()
        session.add_all(
            [
                ConversationCorrectionSegment(
                    correction_id=correction_id,
                    segment_id=segment_id,
                    thread_path="inbox/chained",
                    before_conversation_id=before_id,
                    after_conversation_id=destination_id,
                    previous_correction_id="earlier-split",
                ),
                ConversationSegmentOverride(
                    segment_id=segment_id,
                    # A prior split owns the base identity. Revert does not require it to
                    # equal the intermediate conversation restored by this merge.
                    base_conversation_id=base_id,
                    conversation_id=destination_id,
                    correction_id=correction_id,
                ),
            ]
        )

    checks = verify_catalog(archive_layout, catalog_engine)

    stars = next(check for check in checks if check["key"] == "stars.targets")
    assert stars["status"] == "pass"
    assert stars["detail"] == "active=0, latent=1, missing=0"


def test_stars_target_check_is_mandatory_for_source_retirement(
    archive_layout: ArchiveLayout,
    catalog_engine: Engine,
) -> None:
    checks = verify_catalog(archive_layout, catalog_engine)
    without_stars = [check for check in checks if check["key"] != "stars.targets"]

    non_pass = mandatory_non_pass_checks(without_stars)

    assert any(
        check["key"] == "stars.targets" and check["status"] == "missing" for check in non_pass
    )
