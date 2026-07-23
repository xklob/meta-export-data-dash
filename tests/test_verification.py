from __future__ import annotations

import json
from dataclasses import fields

from sqlalchemy import Engine

from meta_archive.baseline import (
    AUDITED_BASELINE_MANIFEST_RELATIVE_PATH,
    AuditedBaseline,
    archive_identity_sha256,
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
