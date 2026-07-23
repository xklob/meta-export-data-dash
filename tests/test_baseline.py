from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

import meta_archive.baseline as baseline_module
from meta_archive.baseline import (
    AUDITED_BASELINE_MANIFEST_RELATIVE_PATH,
    MAX_AUDITED_BASELINE_MANIFEST_BYTES,
    MAX_AUDITED_BASELINE_METRIC,
    AuditedBaselineManifestError,
    archive_identity_sha256,
    load_audited_baseline,
)


def _baseline_payload(archive_root: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "archive_identity_sha256": archive_identity_sha256(archive_root),
        "metrics": {
            "chunks": 1,
            "directories": 2,
            "files": 3,
            "bytes": 4,
            "json_files": 5,
            "transcript_containers": 6,
            "ordinary_segments": 7,
            "ordinary_messages": 8,
            "canonical_participants": 9,
            "message_reactions": 10,
            "message_attachments": 11,
            "support_segments": 12,
            "support_messages": 13,
            "activity_records": 14,
            "exact_duplicate_messages": 15,
            "normalized_path_collisions": 16,
            "unique_blobs": 17,
            "unique_blob_bytes": 18,
            "duplicate_file_occurrences": 19,
            "duplicate_bytes": 20,
            "unresolved_local_uri_occurrences": 21,
            "unlinked_media_paths": 22,
            "leading_message_gap_segments": 23,
            "internal_message_gap_segments": 24,
            "activity_reactions_raw": 25,
            "activity_reactions_logical": 26,
            "largest_conversation_messages": 27,
        },
    }


def _write_manifest(archive_root: Path, payload: object) -> Path:
    path = archive_root / AUDITED_BASELINE_MANIFEST_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(0o600)
    return path


def test_load_audited_baseline_returns_none_when_manifest_is_absent(tmp_path: Path) -> None:
    archive_root = tmp_path / "meta-backup-fixture"
    archive_root.mkdir()

    assert load_audited_baseline(archive_root) is None


def test_load_audited_baseline_accepts_exact_private_schema(tmp_path: Path) -> None:
    archive_root = tmp_path / "meta-backup-fixture"
    archive_root.mkdir()
    payload = _baseline_payload(archive_root)
    _write_manifest(archive_root, payload)

    baseline = load_audited_baseline(archive_root)

    assert baseline is not None
    assert asdict(baseline) == payload["metrics"]


@pytest.mark.parametrize(
    "mutation",
    (
        lambda payload: payload.update(schema_version=2),
        lambda payload: payload.update(unexpected=True),
        lambda payload: payload["metrics"].pop("files"),
        lambda payload: payload["metrics"].update(files=True),
        lambda payload: payload["metrics"].update(files=-1),
        lambda payload: payload["metrics"].update(files=MAX_AUDITED_BASELINE_METRIC + 1),
        lambda payload: payload.update(archive_identity_sha256="0" * 64),
    ),
)
def test_load_audited_baseline_rejects_schema_drift(tmp_path: Path, mutation: Any) -> None:
    archive_root = tmp_path / "meta-backup-fixture"
    archive_root.mkdir()
    payload = _baseline_payload(archive_root)
    mutation(payload)
    _write_manifest(archive_root, payload)

    with pytest.raises(AuditedBaselineManifestError):
        load_audited_baseline(archive_root)


def test_load_audited_baseline_rejects_duplicate_keys_and_non_private_mode(
    tmp_path: Path,
) -> None:
    archive_root = tmp_path / "meta-backup-fixture"
    archive_root.mkdir()
    path = archive_root / AUDITED_BASELINE_MANIFEST_RELATIVE_PATH
    path.parent.mkdir(parents=True)
    path.write_text(
        '{"schema_version":1,"schema_version":1,"archive_identity_sha256":"x","metrics":{}}',
        encoding="utf-8",
    )
    path.chmod(0o600)
    with pytest.raises(AuditedBaselineManifestError, match="duplicate"):
        load_audited_baseline(archive_root)

    path.write_text(json.dumps(_baseline_payload(archive_root)), encoding="utf-8")
    path.chmod(0o640)
    with pytest.raises(AuditedBaselineManifestError, match="owner-only"):
        load_audited_baseline(archive_root)


def test_load_audited_baseline_rejects_symlink(tmp_path: Path) -> None:
    archive_root = tmp_path / "meta-backup-fixture"
    archive_root.mkdir()
    path = archive_root / AUDITED_BASELINE_MANIFEST_RELATIVE_PATH
    path.parent.mkdir(parents=True)
    target = tmp_path / "outside.json"
    target.write_text(json.dumps(_baseline_payload(archive_root)), encoding="utf-8")
    target.chmod(0o600)
    path.symlink_to(target)

    with pytest.raises(AuditedBaselineManifestError, match="regular"):
        load_audited_baseline(archive_root)


def test_load_audited_baseline_revalidates_opened_descriptor_permissions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "meta-backup-fixture"
    archive_root.mkdir()
    path = _write_manifest(archive_root, _baseline_payload(archive_root))
    secure_open = baseline_module.open_regular_file_beneath

    def chmod_before_open(candidate: Path, root: Path):
        candidate.chmod(0o640)
        return secure_open(candidate, root)

    monkeypatch.setattr(baseline_module, "open_regular_file_beneath", chmod_before_open)

    with pytest.raises(AuditedBaselineManifestError, match="owner-only"):
        load_audited_baseline(archive_root)
    assert path.stat().st_mode & 0o077


def test_load_audited_baseline_revalidates_opened_descriptor_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "meta-backup-fixture"
    archive_root.mkdir()
    _write_manifest(archive_root, _baseline_payload(archive_root))
    secure_open = baseline_module.open_regular_file_beneath

    def enlarge_before_open(candidate: Path, root: Path):
        candidate.write_bytes(b"x" * (MAX_AUDITED_BASELINE_MANIFEST_BYTES + 1))
        return secure_open(candidate, root)

    monkeypatch.setattr(baseline_module, "open_regular_file_beneath", enlarge_before_open)

    with pytest.raises(AuditedBaselineManifestError, match="too large"):
        load_audited_baseline(archive_root)


def test_load_audited_baseline_rejects_inode_swap_between_stat_and_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "meta-backup-fixture"
    archive_root.mkdir()
    _write_manifest(archive_root, _baseline_payload(archive_root))
    replacement = tmp_path / "replacement.json"
    replacement.write_text(
        json.dumps(_baseline_payload(archive_root)),
        encoding="utf-8",
    )
    replacement.chmod(0o600)

    def open_replacement(_candidate: Path, _root: Path):
        return replacement.open("rb")

    monkeypatch.setattr(baseline_module, "open_regular_file_beneath", open_replacement)

    with pytest.raises(AuditedBaselineManifestError, match="changed during secure open"):
        load_audited_baseline(archive_root)
