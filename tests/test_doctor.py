from __future__ import annotations

import json
from pathlib import Path

from meta_archive.paths import ArchiveLayout
from meta_archive.services.doctor import _private_permissions_check, _source_chunks_check


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_source_chunk_check_accepts_present_export_directories(tmp_path: Path) -> None:
    layout = ArchiveLayout(tmp_path / "meta-backup-fixture")
    layout.root.mkdir()
    (layout.root / "facebook-export-part-one").mkdir()

    check = _source_chunks_check(layout)

    assert check.status == "pass"
    assert check.detail == "1 source directories present"


def test_source_chunk_check_accepts_manifest_backed_retired_state(tmp_path: Path) -> None:
    layout = ArchiveLayout(tmp_path / "meta-backup-fixture")
    layout.ensure_runtime_directories()
    layout.original_files_manifest.write_text('{"source":"one"}\n', encoding="utf-8")
    layout.relocation_manifest.write_text('{"source":"one"}\n', encoding="utf-8")
    layout.original_directories_manifest.write_text(
        '{"chunk":"facebook-export-part-one","relative_path":""}\n',
        encoding="utf-8",
    )
    _write_json(
        layout.manifests / "integrity-reports" / "latest.json",
        {
            "gate": {
                "all_mandatory_checks_pass": True,
                "deep_verified": True,
                "library_verified": True,
                "required_artifacts_present": True,
            },
            "checks": [{"key": "fixture", "mandatory": True, "status": "pass"}],
        },
    )
    _write_json(
        layout.manifests / "source-retirement" / "20260720T000000Z-fixture.json",
        {
            "schema_version": 1,
            "result": {
                "applied": True,
                "requested_chunks": ["facebook-export-part-one"],
                "manifest_file_count": 1,
            },
        },
    )

    check = _source_chunks_check(layout)

    assert check.status == "pass"
    assert "safely retired" in check.detail
    assert "remain reconstructable" in check.detail


def test_source_chunk_check_fails_closed_without_receipt_or_current_integrity(
    tmp_path: Path,
) -> None:
    layout = ArchiveLayout(tmp_path / "meta-backup-fixture")
    layout.ensure_runtime_directories()

    missing_receipt = _source_chunks_check(layout)
    assert missing_receipt.status == "fail"

    _write_json(
        layout.manifests / "source-retirement" / "20260720T000000Z-fixture.json",
        {
            "schema_version": 1,
            "result": {
                "applied": True,
                "requested_chunks": ["facebook-export-part-one"],
                "manifest_file_count": 1,
            },
        },
    )
    layout.original_files_manifest.write_text("{}\n", encoding="utf-8")
    layout.relocation_manifest.write_text("{}\n", encoding="utf-8")
    layout.original_directories_manifest.write_text("{}\n", encoding="utf-8")

    missing_integrity = _source_chunks_check(layout)
    assert missing_integrity.status == "fail"
    assert "deep-integrity gate" in missing_integrity.detail


def test_private_permissions_check_warns_then_passes_after_runtime_repair(
    tmp_path: Path,
) -> None:
    layout = ArchiveLayout(tmp_path / "meta-backup-fixture")
    layout.root.mkdir()
    layout.root.chmod(0o755)

    warning = _private_permissions_check(layout)
    assert warning.status == "warn"
    assert warning.required is False
    assert "not owner-only" in warning.detail

    layout.ensure_runtime_directories()
    passing = _private_permissions_check(layout)
    assert passing.status == "pass"
    assert passing.required is False
