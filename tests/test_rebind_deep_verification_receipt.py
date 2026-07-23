from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from meta_archive.database import run_migrations
from meta_archive.paths import ArchiveLayout
from meta_archive.services.deletion_lock_lease import DeletionError, deletion_lock
from meta_archive.services.source_import_lock import (
    SourceImportPublicationBusyError,
    source_import_publication_lock,
)
from meta_archive.services.verification import (
    SOURCE_RETIREMENT_REQUIRED_CHECK_KEYS,
    write_verification_report,
)


def _load_rebind_module() -> ModuleType:
    script = Path(__file__).resolve().parents[1] / "scripts" / "rebind_deep_verification_receipt.py"
    spec = importlib.util.spec_from_file_location("rebind_deep_verification_receipt", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rebind = _load_rebind_module()


def _check(key: str, *, status_value: str = "pass") -> dict[str, object]:
    return {
        "key": key,
        "status": status_value,
        "expected": 1,
        "actual": 1,
        "detail": "synthetic aggregate verification",
        "mandatory": True,
    }


def _passing_checks(*, deep_status: str = "pass") -> list[dict[str, object]]:
    return [
        _check(key, status_value=deep_status if key == "deep.source_bytes" else "pass")
        for key in sorted(SOURCE_RETIREMENT_REQUIRED_CHECK_KEYS)
    ]


def _current_checks() -> list[dict[str, object]]:
    return [check for check in _passing_checks() if check["key"] != "deep.source_bytes"]


def _write_private(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)
    path.chmod(0o600)


def _write_latest(layout: ArchiveLayout, checks: list[dict[str, object]]) -> Path:
    report = write_verification_report(layout, checks)
    report.chmod(0o600)
    return report


def _write_legacy_latest_without_stars(layout: ArchiveLayout) -> bytes:
    """Publish the exact pre-stars receipt shape used by the one-time compatibility bridge."""

    report = _write_latest(layout, _passing_checks())
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["checks"] = [check for check in payload["checks"] if check["key"] != "stars.targets"]
    payload["gate"] = {
        "all_mandatory_checks_pass": True,
        "deep_verified": True,
        "library_verified": True,
        "non_pass_keys": [],
        "required_artifacts_present": True,
        "source_retirement_eligible": True,
    }
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode()
    _write_private(report, encoded)
    return encoded


@pytest.fixture
def prepared_receipt(
    archive_layout: ArchiveLayout,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ArchiveLayout, list[dict[str, object]], bytes]:
    archive_layout.ensure_runtime_directories()
    run_migrations(archive_layout.database)
    for path in (
        archive_layout.original_files_manifest,
        archive_layout.original_directories_manifest,
        archive_layout.relocation_manifest,
    ):
        _write_private(path, b"")
    _write_private(archive_layout.library_completion_marker, b'\n{"format_version": 1}\n')
    report = _write_latest(archive_layout, _passing_checks())
    current = _current_checks()
    monkeypatch.setattr(rebind, "verify_catalog", lambda *_args, **_kwargs: current)
    return archive_layout, current, report.read_bytes()


def _latest_payload(layout: ArchiveLayout) -> dict[str, Any]:
    value = json.loads(
        (layout.manifests / "integrity-reports" / "latest.json").read_text(encoding="utf-8")
    )
    assert isinstance(value, dict)
    return value


def _carried_check(check: dict[str, object], digest: str) -> dict[str, object]:
    return {
        **check,
        "expected": "passing deep source-byte verification retained in the prior audit receipt",
        "actual": f"carried-forward; prior_report_sha256={digest}",
        "detail": (
            "Deep source-byte verification carried forward after exact artifact matching and a "
            f"passing current non-deep catalog verification; prior_report_sha256={digest}"
        ),
    }


def test_rebind_preserves_exact_prior_and_publishes_provenance_bound_receipt(
    prepared_receipt: tuple[ArchiveLayout, list[dict[str, object]], bytes],
) -> None:
    layout, current, prior_bytes = prepared_receipt
    prior_digest = hashlib.sha256(prior_bytes).hexdigest()

    assert rebind.rebind_deep_receipt(layout.root) is True

    reports = layout.manifests / "integrity-reports"
    audit = reports / f"prior-{prior_digest}.json"
    assert audit.read_bytes() == prior_bytes
    assert stat.S_IMODE(audit.stat().st_mode) == 0o600
    assert audit.stat().st_nlink == 1
    latest = reports / "latest.json"
    assert stat.S_IMODE(latest.stat().st_mode) == 0o600
    assert latest.stat().st_nlink == 1
    payload = _latest_payload(layout)
    assert payload["schema_version"] == 1
    assert payload["gate"] == {
        "all_mandatory_checks_pass": True,
        "deep_verified": True,
        "library_verified": True,
        "non_pass_keys": [],
        "required_artifacts_present": True,
        "source_retirement_eligible": True,
    }
    checks = payload["checks"]
    assert [check for check in checks if check["key"] != "deep.source_bytes"] == current
    deep = [check for check in checks if check["key"] == "deep.source_bytes"]
    direct_deep = next(c for c in _passing_checks() if c["key"] == "deep.source_bytes")
    assert deep == [_carried_check(direct_deep, prior_digest)]
    assert not list(reports.glob("rebind-*"))


def test_rebind_accepts_exact_legacy_receipt_missing_only_stars_targets(
    prepared_receipt: tuple[ArchiveLayout, list[dict[str, object]], bytes],
) -> None:
    layout, current, _prior = prepared_receipt
    prior_bytes = _write_legacy_latest_without_stars(layout)
    prior_digest = hashlib.sha256(prior_bytes).hexdigest()

    assert rebind.rebind_deep_receipt(layout.root) is True

    payload = _latest_payload(layout)
    assert [check for check in payload["checks"] if check["key"] != "deep.source_bytes"] == current
    assert (
        next(check for check in payload["checks"] if check["key"] == "stars.targets")["status"]
        == "pass"
    )
    assert (
        layout.manifests / "integrity-reports" / f"prior-{prior_digest}.json"
    ).read_bytes() == prior_bytes


def test_rebind_rejects_any_addition_beyond_stars_targets(
    prepared_receipt: tuple[ArchiveLayout, list[dict[str, object]], bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, current, _prior = prepared_receipt
    _write_legacy_latest_without_stars(layout)
    monkeypatch.setattr(
        rebind,
        "verify_catalog",
        lambda *_args, **_kwargs: [*current, _check("unexpected.new.check")],
    )

    with pytest.raises(rebind.ReceiptRebindError, match="check-identity-changed"):
        rebind.rebind_deep_receipt(layout.root)


def test_rebind_requires_fresh_stars_target_check_for_legacy_receipt(
    prepared_receipt: tuple[ArchiveLayout, list[dict[str, object]], bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, current, _prior = prepared_receipt
    _write_legacy_latest_without_stars(layout)
    without_stars = [check for check in current if check["key"] != "stars.targets"]
    monkeypatch.setattr(rebind, "verify_catalog", lambda *_args, **_kwargs: without_stars)

    with pytest.raises(rebind.ReceiptRebindError, match="check-identity-changed"):
        rebind.rebind_deep_receipt(layout.root)


def test_rebind_rejects_legacy_receipt_missing_another_required_check(
    prepared_receipt: tuple[ArchiveLayout, list[dict[str, object]], bytes],
) -> None:
    layout, _current, _prior = prepared_receipt
    prior = _write_legacy_latest_without_stars(layout)
    payload = json.loads(prior)
    payload["checks"] = [check for check in payload["checks"] if check["key"] != "sqlite.integrity"]
    latest = layout.manifests / "integrity-reports" / "latest.json"
    _write_private(latest, (json.dumps(payload, sort_keys=True) + "\n").encode())

    with pytest.raises(rebind.ReceiptRebindError, match="missing-required-check"):
        rebind.rebind_deep_receipt(layout.root)


@pytest.mark.parametrize("artifact_name", ["original-files.jsonl", "relocation.jsonl"])
def test_rebind_rejects_changed_recorded_artifact(
    prepared_receipt: tuple[ArchiveLayout, list[dict[str, object]], bytes],
    artifact_name: str,
) -> None:
    layout, _current, _prior = prepared_receipt
    _write_private(layout.manifests / artifact_name, b"changed\n")

    with pytest.raises(rebind.ReceiptRebindError, match="artifact-identity-changed"):
        rebind.rebind_deep_receipt(layout.root)


def test_rebind_rejects_changed_library_marker(
    prepared_receipt: tuple[ArchiveLayout, list[dict[str, object]], bytes],
) -> None:
    layout, _current, _prior = prepared_receipt
    _write_private(layout.library_completion_marker, b'{"format_version": 2}\n')

    with pytest.raises(rebind.ReceiptRebindError, match="artifact-identity-changed"):
        rebind.rebind_deep_receipt(layout.root)


@pytest.mark.parametrize("deep_state", ["missing", "nonpass", "stale"])
def test_rebind_rejects_missing_nonpassing_or_stale_deep_evidence(
    prepared_receipt: tuple[ArchiveLayout, list[dict[str, object]], bytes],
    deep_state: str,
) -> None:
    layout, current, prior = prepared_receipt
    if deep_state == "missing":
        _write_latest(layout, current)
    elif deep_state == "nonpass":
        _write_latest(layout, _passing_checks(deep_status="fail"))
    else:
        missing_digest = hashlib.sha256(prior + b"missing-audit").hexdigest()
        direct_deep = next(c for c in _passing_checks() if c["key"] == "deep.source_bytes")
        _write_latest(layout, [*current, _carried_check(direct_deep, missing_digest)])

    with pytest.raises(rebind.ReceiptRebindError):
        rebind.rebind_deep_receipt(layout.root)


def test_rebind_rejects_duplicate_json_keys(
    prepared_receipt: tuple[ArchiveLayout, list[dict[str, object]], bytes],
) -> None:
    layout, _current, _prior = prepared_receipt
    latest = layout.manifests / "integrity-reports" / "latest.json"
    payload = latest.read_bytes()
    duplicate = payload.replace(
        b'"schema_version": 1',
        b'"schema_version": 1, "schema_version": 1',
        1,
    )
    assert duplicate != payload
    _write_private(latest, duplicate)

    with pytest.raises(rebind.ReceiptRebindError, match="invalid-report-json"):
        rebind.rebind_deep_receipt(layout.root)


@pytest.mark.parametrize("unsafe_kind", ["latest-mode", "latest-hardlink", "audit-mode"])
def test_rebind_rejects_unsafe_report_or_audit_copy(
    prepared_receipt: tuple[ArchiveLayout, list[dict[str, object]], bytes],
    unsafe_kind: str,
) -> None:
    layout, current, prior = prepared_receipt
    reports = layout.manifests / "integrity-reports"
    latest = reports / "latest.json"
    if unsafe_kind == "latest-mode":
        latest.chmod(0o640)
    elif unsafe_kind == "latest-hardlink":
        os.link(latest, reports / "unexpected-link.json")
    else:
        digest = hashlib.sha256(prior).hexdigest()
        audit = reports / f"prior-{digest}.json"
        _write_private(audit, prior)
        audit.chmod(0o640)
        direct_deep = next(c for c in _passing_checks() if c["key"] == "deep.source_bytes")
        _write_latest(layout, [*current, _carried_check(direct_deep, digest)])

    with pytest.raises(rebind.ReceiptRebindError, match="unsafe-report-file"):
        rebind.rebind_deep_receipt(layout.root)


def test_rebind_rejects_current_non_deep_failure(
    prepared_receipt: tuple[ArchiveLayout, list[dict[str, object]], bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, current, _prior = prepared_receipt
    failed = [dict(check) for check in current]
    failed[0]["status"] = "fail"
    monkeypatch.setattr(rebind, "verify_catalog", lambda *_args, **_kwargs: failed)

    with pytest.raises(rebind.ReceiptRebindError, match="current-verification-nonpass"):
        rebind.rebind_deep_receipt(layout.root)


def test_rebind_never_overwrites_colliding_digest_named_audit_copy(
    prepared_receipt: tuple[ArchiveLayout, list[dict[str, object]], bytes],
) -> None:
    layout, _current, prior = prepared_receipt
    reports = layout.manifests / "integrity-reports"
    collision = reports / f"prior-{hashlib.sha256(prior).hexdigest()}.json"
    colliding_bytes = b'{"aggregate":"different"}\n'
    _write_private(collision, colliding_bytes)

    with pytest.raises(rebind.ReceiptRebindError, match="audit-copy-collision"):
        rebind.rebind_deep_receipt(layout.root)

    assert collision.read_bytes() == colliding_bytes


def test_audit_publication_fault_leaves_no_partial_digest_named_copy_and_retries(
    prepared_receipt: tuple[ArchiveLayout, list[dict[str, object]], bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, _current, prior = prepared_receipt
    reports = layout.manifests / "integrity-reports"
    audit = reports / f"prior-{hashlib.sha256(prior).hexdigest()}.json"
    original_rename = rebind._rename_noreplace

    def fail_publication(*_args: object, **_kwargs: object) -> bool:
        raise rebind.ReceiptRebindError("synthetic-publication-fault")

    monkeypatch.setattr(rebind, "_rename_noreplace", fail_publication)
    with pytest.raises(rebind.ReceiptRebindError, match="synthetic-publication-fault"):
        rebind.rebind_deep_receipt(layout.root)

    assert not audit.exists()
    assert (reports / "latest.json").read_bytes() == prior
    assert not list(reports.glob("rebind-*"))

    monkeypatch.setattr(rebind, "_rename_noreplace", original_rename)
    assert rebind.rebind_deep_receipt(layout.root) is True
    assert audit.read_bytes() == prior


def test_orphaned_partial_audit_staging_file_does_not_wedge_retry(
    prepared_receipt: tuple[ArchiveLayout, list[dict[str, object]], bytes],
) -> None:
    layout, _current, prior = prepared_receipt
    reports = layout.manifests / "integrity-reports"
    orphan = reports / "rebind-interrupted.audit.tmp"
    _write_private(orphan, b"partial")

    assert rebind.rebind_deep_receipt(layout.root) is True

    audit = reports / f"prior-{hashlib.sha256(prior).hexdigest()}.json"
    assert audit.read_bytes() == prior
    assert orphan.read_bytes() == b"partial"


def test_candidate_writer_fault_preserves_latest_and_cleans_owned_staging(
    prepared_receipt: tuple[ArchiveLayout, list[dict[str, object]], bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, _current, prior = prepared_receipt
    reports = layout.manifests / "integrity-reports"

    def fail_writer(*_args: object, **_kwargs: object) -> Path:
        raise OSError("synthetic private writer detail")

    monkeypatch.setattr(rebind, "write_verification_report", fail_writer)
    with pytest.raises(rebind.ReceiptRebindError, match="candidate-publication-failed"):
        rebind.rebind_deep_receipt(layout.root)

    assert (reports / "latest.json").read_bytes() == prior
    assert not list(reports.glob("rebind-*.candidate*"))


def test_latest_substitution_after_audit_copy_is_detected_before_publication(
    prepared_receipt: tuple[ArchiveLayout, list[dict[str, object]], bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, _current, prior = prepared_receipt
    reports = layout.manifests / "integrity-reports"
    latest = reports / "latest.json"
    original_preserve = rebind._preserve_prior

    def preserve_then_substitute(directory_descriptor: int, bound: object) -> None:
        original_preserve(directory_descriptor, bound)
        latest.unlink()
        _write_private(latest, prior)

    monkeypatch.setattr(rebind, "_preserve_prior", preserve_then_substitute)
    with pytest.raises(rebind.ReceiptRebindError, match="latest-report-substitution"):
        rebind.rebind_deep_receipt(layout.root)

    assert latest.read_bytes() == prior
    assert not list(reports.glob("rebind-*.candidate*"))


def test_rebind_is_byte_stable_on_idempotent_rerun(
    prepared_receipt: tuple[ArchiveLayout, list[dict[str, object]], bytes],
) -> None:
    layout, _current, _prior = prepared_receipt
    assert rebind.rebind_deep_receipt(layout.root) is True
    latest = layout.manifests / "integrity-reports" / "latest.json"
    first_bytes = latest.read_bytes()
    first_stat = latest.stat()
    first_audits = sorted(latest.parent.glob("prior-*.json"))

    assert rebind.rebind_deep_receipt(layout.root) is False

    second_stat = latest.stat()
    assert latest.read_bytes() == first_bytes
    assert (second_stat.st_dev, second_stat.st_ino, second_stat.st_mtime_ns) == (
        first_stat.st_dev,
        first_stat.st_ino,
        first_stat.st_mtime_ns,
    )
    assert sorted(latest.parent.glob("prior-*.json")) == first_audits


def test_rebind_holds_deletion_lock_through_current_verification(
    prepared_receipt: tuple[ArchiveLayout, list[dict[str, object]], bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, current, _prior = prepared_receipt
    verification_started = threading.Event()
    allow_verification_to_finish = threading.Event()
    result: list[bool] = []
    failure: list[BaseException] = []

    def paused_verification(*_args: object, **_kwargs: object) -> list[dict[str, object]]:
        verification_started.set()
        assert allow_verification_to_finish.wait(timeout=5)
        return current

    def run_rebind() -> None:
        try:
            result.append(rebind.rebind_deep_receipt(layout.root))
        except BaseException as exc:  # pragma: no cover - asserted through the failure list
            failure.append(exc)

    monkeypatch.setattr(rebind, "verify_catalog", paused_verification)
    worker = threading.Thread(target=run_rebind)
    worker.start()
    assert verification_started.wait(timeout=5)
    try:
        with (
            pytest.raises(DeletionError, match="currently mutating"),
            deletion_lock(layout),
        ):
            pytest.fail("a concurrent deletion acquired the receipt's outer lock")
    finally:
        allow_verification_to_finish.set()
        worker.join(timeout=10)

    assert not worker.is_alive()
    assert failure == []
    assert result == [True]
    with deletion_lock(layout):
        pass


def test_rebind_holds_deletion_and_source_locks_through_publication_fsync(
    prepared_receipt: tuple[ArchiveLayout, list[dict[str, object]], bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, _current, _prior = prepared_receipt
    original_publish = rebind._publish_report
    checkpoints: list[str] = []

    def assert_competing_publishers_are_excluded(label: str) -> None:
        with (
            pytest.raises(DeletionError, match="currently mutating"),
            deletion_lock(layout),
        ):
            pytest.fail("a concurrent deletion acquired the receipt's outer lock")
        with (
            pytest.raises(SourceImportPublicationBusyError, match="Another source import"),
            source_import_publication_lock(layout, blocking=False),
        ):
            pytest.fail("a concurrent source publisher acquired the receipt's inner lock")
        checkpoints.append(label)

    def observed_publish(*args: object, **kwargs: object) -> None:
        assert_competing_publishers_are_excluded("before")
        original_publish(*args, **kwargs)
        assert_competing_publishers_are_excluded("after-fsync")

    monkeypatch.setattr(rebind, "_publish_report", observed_publish)

    assert rebind.rebind_deep_receipt(layout.root) is True
    assert checkpoints == ["before", "after-fsync"]


def test_cli_output_never_echoes_private_report_content_or_path(
    prepared_receipt: tuple[ArchiveLayout, list[dict[str, object]], bytes],
    capsys: pytest.CaptureFixture[str],
) -> None:
    layout, _current, _prior = prepared_receipt
    secret = "private-message-sentinel"
    latest = layout.manifests / "integrity-reports" / "latest.json"
    _write_private(latest, f'{{"secret":"{secret}"}}'.encode())

    assert rebind.main(["--archive-root", str(layout.root)]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "deep receipt rebind: failed validation\n"
    assert secret not in captured.err
    assert str(layout.root) not in captured.err
