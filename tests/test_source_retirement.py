from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from sqlalchemy import Engine

from meta_archive.paths import ArchiveLayout
from meta_archive.services.consolidation import build_blob_store, reconstruct_source
from meta_archive.services.hashing import sha256_file
from meta_archive.services.inventory import scan_archive
from meta_archive.services.source_retirement import (
    SourceRetirementError,
    retire_source_chunks,
)
from meta_archive.services.verification import (
    SOURCE_RETIREMENT_REQUIRED_CHECK_KEYS,
    source_manifest_artifacts,
    write_verification_report,
)


def _prepare_retirement_gate(layout: ArchiveLayout, engine: Engine) -> Path:
    for chunk, content in (("package-a", b"one"), ("package-b", b"two")):
        source = layout.root / chunk / "nested" / "record.bin"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(content)
    scan_archive(layout, engine, hash_files=True)
    build_blob_store(layout, engine)
    layout.library_completion_marker.parent.mkdir(parents=True, exist_ok=True)
    layout.library_completion_marker.write_text(
        json.dumps(
            {
                "format_version": 1,
                "counts": {
                    "conversation_count": 0,
                    "message_count": 0,
                    "attachment_occurrence_count": 0,
                    "activity_record_count": 0,
                    "unlinked_media_count": 0,
                },
                "source_manifests": source_manifest_artifacts(layout),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    checks = [
        {
            "key": key,
            "status": "pass",
            "expected": "pass",
            "actual": "pass",
            "detail": "synthetic all-pass retirement fixture",
            "mandatory": True,
        }
        for key in sorted(SOURCE_RETIREMENT_REQUIRED_CHECK_KEYS)
    ]
    return write_verification_report(layout, checks)


def test_retirement_is_dry_run_by_default_and_reconstruction_remains_complete(
    archive_layout: ArchiveLayout, catalog_engine: Engine, tmp_path: Path
) -> None:
    _prepare_retirement_gate(archive_layout, catalog_engine)
    original_manifest_digest = sha256_file(archive_layout.original_files_manifest)
    directory_manifest_digest = sha256_file(archive_layout.original_directories_manifest)

    preview = retire_source_chunks(archive_layout, catalog_engine, ["package-a"])

    assert not preview.applied
    assert preview.requested_chunks == ("package-a",)
    assert preview.present_file_count == 1
    assert (archive_layout.root / "package-a/nested/record.bin").is_file()
    assert (archive_layout.root / "package-b/nested/record.bin").is_file()

    applied = retire_source_chunks(archive_layout, catalog_engine, ["package-a"], apply=True)
    assert applied.applied
    assert applied.receipt_path is not None
    assert not (archive_layout.root / "package-a").exists()
    assert (archive_layout.root / "package-b/nested/record.bin").is_file()

    rescanned = scan_archive(archive_layout, catalog_engine, hash_files=True)
    assert rescanned.chunk_count == 2
    assert rescanned.file_count == 2
    assert rescanned.hashed_file_count == 0
    assert rescanned.reused_hash_count == 2
    assert len(archive_layout.original_files_manifest.read_text().splitlines()) == 2
    assert sha256_file(archive_layout.original_files_manifest) == original_manifest_digest
    assert sha256_file(archive_layout.original_directories_manifest) == directory_manifest_digest

    reconstructed = tmp_path / "reconstructed-after-retirement"
    assert reconstruct_source(archive_layout, reconstructed) == 2
    assert (reconstructed / "package-a/nested/record.bin").read_bytes() == b"one"
    assert (reconstructed / "package-b/nested/record.bin").read_bytes() == b"two"


def test_retirement_rejects_non_pass_gate_and_unmanifested_entries(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> None:
    report = _prepare_retirement_gate(archive_layout, catalog_engine)
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["checks"][0]["status"] = "pending"
    report.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SourceRetirementError, match="all-pass"):
        retire_source_chunks(archive_layout, catalog_engine, ["package-a"])

    _prepare_retirement_gate(archive_layout, catalog_engine)
    (archive_layout.root / "package-a" / "unexpected.txt").write_text(
        "not inventoried", encoding="utf-8"
    )
    with pytest.raises(SourceRetirementError, match="Unmanifested"):
        retire_source_chunks(archive_layout, catalog_engine, ["package-a"])


def test_retirement_requires_exact_known_chunk_names(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> None:
    _prepare_retirement_gate(archive_layout, catalog_engine)

    with pytest.raises(SourceRetirementError, match="exact source chunk"):
        retire_source_chunks(archive_layout, catalog_engine, ["package-*"])


def test_retirement_detects_same_size_same_mtime_source_tampering(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> None:
    _prepare_retirement_gate(archive_layout, catalog_engine)
    source = archive_layout.root / "package-a/nested/record.bin"
    original_stat = source.stat()
    source.write_bytes(b"bad")
    os.utime(
        source,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
        follow_symlinks=False,
    )

    with pytest.raises(SourceRetirementError, match="digest changed"):
        retire_source_chunks(archive_layout, catalog_engine, ["package-a"])

    assert source.read_bytes() == b"bad"
