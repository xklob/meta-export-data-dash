#!/usr/bin/env python3
"""Rebind a passing deep-verification receipt after a catalog-only migration.

This command never re-hashes exported source bytes. It carries forward the one previously
passing ``deep.source_bytes`` check only when the source manifests and library marker still
exactly match their recorded artifact identities and a fresh non-deep catalog verification
passes. The exact prior report is retained as an owner-only, digest-named audit record.

Command output is intentionally aggregate-only. Receipt validation failures never include report
contents, archive paths, timestamps, or exception details.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import re
import secrets
import stat
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, cast

from sqlalchemy import Engine

from meta_archive.database import create_catalog_engine, require_current_migration
from meta_archive.paths import ArchiveLayout, open_directory_beneath
from meta_archive.services.deletion_lock_lease import deletion_lock
from meta_archive.services.source_import_lock import source_import_publication_lock
from meta_archive.services.verification import (
    SOURCE_RETIREMENT_REQUIRED_CHECK_KEYS,
    _artifact_record,
    mandatory_non_pass_checks,
    source_manifest_artifacts,
    verify_catalog,
    write_verification_report,
)

MAX_REPORT_BYTES = 8 * 1024 * 1024
MAX_PROVENANCE_DEPTH = 64
LATEST_REPORT_NAME = "latest.json"
AUDIT_NAME_PREFIX = "prior-"

_REPORT_KEYS = frozenset({"schema_version", "generated_at", "checks", "artifacts", "gate"})
_CHECK_KEYS = frozenset({"key", "status", "expected", "actual", "detail", "mandatory"})
_ARTIFACTS_KEYS = frozenset({"source_manifests", "library_marker"})
_SOURCE_MANIFEST_KEYS = frozenset({"original_files", "original_directories", "relocation"})
_ARTIFACT_RECORD_KEYS = frozenset({"path", "sha256", "line_count", "size_bytes"})
_GATE_KEYS = frozenset(
    {
        "all_mandatory_checks_pass",
        "deep_verified",
        "library_verified",
        "required_artifacts_present",
        "source_retirement_eligible",
        "non_pass_keys",
    }
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_CARRIED_ACTUAL_RE = re.compile(r"carried-forward; prior_report_sha256=([0-9a-f]{64})\Z")
_RENAME_NOREPLACE = 1
_CARRIED_EXPECTED = "passing deep source-byte verification retained in the prior audit receipt"


class ReceiptRebindError(RuntimeError):
    """The receipt could not be rebound without weakening its evidence."""


class _DuplicateKeyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class _BoundedFile:
    payload: bytes
    device: int
    inode: int
    size: int
    modified_ns: int

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()


def _fail(code: str) -> NoReturn:
    raise ReceiptRebindError(code)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError
        result[key] = value
    return result


def _strict_json(payload: bytes) -> dict[str, Any]:
    try:
        decoded = payload.decode("utf-8")
        value = json.loads(decoded, object_pairs_hook=_strict_object)
    except (UnicodeDecodeError, json.JSONDecodeError, _DuplicateKeyError):
        _fail("invalid-report-json")
    if not isinstance(value, dict):
        _fail("invalid-report-object")
    return cast(dict[str, Any], value)


def _validate_private_directory(descriptor: int) -> None:
    value = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(value.st_mode)
        or value.st_uid != os.geteuid()
        or stat.S_IMODE(value.st_mode) != 0o700
    ):
        _fail("unsafe-report-directory")


def _validate_private_regular(value: os.stat_result) -> None:
    if (
        not stat.S_ISREG(value.st_mode)
        or value.st_uid != os.geteuid()
        or value.st_nlink != 1
        or stat.S_IMODE(value.st_mode) != 0o600
    ):
        _fail("unsafe-report-file")


def _read_descriptor_bounded(descriptor: int, expected_size: int) -> bytes:
    if expected_size < 0 or expected_size > MAX_REPORT_BYTES:
        _fail("report-size-out-of-bounds")
    chunks: list[bytes] = []
    remaining = MAX_REPORT_BYTES + 1
    os.lseek(descriptor, 0, os.SEEK_SET)
    while remaining:
        chunk = os.read(descriptor, min(64 * 1024, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    payload = b"".join(chunks)
    if len(payload) != expected_size or len(payload) > MAX_REPORT_BYTES:
        _fail("report-changed-while-reading")
    return payload


def _safe_read_report(directory_descriptor: int, name: str) -> _BoundedFile:
    if not name or "/" in name or name in {".", ".."}:
        _fail("invalid-report-name")
    flags = (
        os.O_RDONLY
        | os.O_NOFOLLOW
        | int(getattr(os, "O_CLOEXEC", 0))
        | int(getattr(os, "O_NONBLOCK", 0))
    )
    try:
        before = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        descriptor = os.open(name, flags, dir_fd=directory_descriptor)
    except OSError:
        _fail("report-unavailable")
    try:
        opened = os.fstat(descriptor)
        _validate_private_regular(before)
        _validate_private_regular(opened)
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            _fail("report-substitution")
        payload = _read_descriptor_bounded(descriptor, opened.st_size)
        after_open = os.fstat(descriptor)
        after_path = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        _validate_private_regular(after_open)
        _validate_private_regular(after_path)
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
        if any(getattr(opened, field) != getattr(after_open, field) for field in stable_fields):
            _fail("report-substitution")
        if (after_open.st_dev, after_open.st_ino) != (after_path.st_dev, after_path.st_ino):
            _fail("report-substitution")
        return _BoundedFile(
            payload=payload,
            device=opened.st_dev,
            inode=opened.st_ino,
            size=opened.st_size,
            modified_ns=opened.st_mtime_ns,
        )
    except OSError:
        _fail("report-read-failed")
    finally:
        os.close(descriptor)


def _same_bound_file(left: _BoundedFile, right: _BoundedFile) -> bool:
    return (
        left.device,
        left.inode,
        left.size,
        left.modified_ns,
        left.payload,
    ) == (
        right.device,
        right.inode,
        right.size,
        right.modified_ns,
        right.payload,
    )


def _artifact_record_is_valid(value: object, *, jsonl: bool) -> bool:
    if not isinstance(value, dict) or frozenset(value) != _ARTIFACT_RECORD_KEYS:
        return False
    path = value.get("path")
    digest = value.get("sha256")
    line_count = value.get("line_count")
    size_bytes = value.get("size_bytes")
    return (
        isinstance(path, str)
        and bool(path)
        and isinstance(digest, str)
        and _SHA256_RE.fullmatch(digest) is not None
        and type(size_bytes) is int
        and size_bytes >= 0
        and (
            (jsonl and type(line_count) is int and line_count >= 0)
            or (not jsonl and line_count is None)
        )
    )


def _validate_artifacts(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or frozenset(value) != _ARTIFACTS_KEYS:
        _fail("invalid-artifact-identities")
    source = value.get("source_manifests")
    marker = value.get("library_marker")
    if not isinstance(source, dict) or frozenset(source) != _SOURCE_MANIFEST_KEYS:
        _fail("invalid-artifact-identities")
    if not all(_artifact_record_is_valid(source.get(key), jsonl=True) for key in source):
        _fail("invalid-artifact-identities")
    if not _artifact_record_is_valid(marker, jsonl=False):
        _fail("invalid-artifact-identities")
    return cast(dict[str, Any], value)


def _validate_checks(value: object) -> tuple[list[dict[str, object]], dict[str, object]]:
    if not isinstance(value, list) or not value:
        _fail("invalid-check-identities")
    checks: list[dict[str, object]] = []
    keys: set[str] = set()
    deep: dict[str, object] | None = None
    for candidate in value:
        if not isinstance(candidate, dict) or frozenset(candidate) != _CHECK_KEYS:
            _fail("invalid-check-identities")
        key = candidate.get("key")
        status_value = candidate.get("status")
        mandatory = candidate.get("mandatory")
        detail = candidate.get("detail")
        expected = candidate.get("expected")
        actual = candidate.get("actual")
        if (
            not isinstance(key, str)
            or not key
            or key in keys
            or status_value not in {"pass", "pending", "fail"}
            or type(mandatory) is not bool
            or not isinstance(detail, str)
            or not (expected is None or isinstance(expected, str) or type(expected) is int)
            or not (actual is None or isinstance(actual, str) or type(actual) is int)
        ):
            _fail("invalid-check-identities")
        keys.add(key)
        normalized = cast(dict[str, object], candidate)
        checks.append(normalized)
        if key in SOURCE_RETIREMENT_REQUIRED_CHECK_KEYS and mandatory is not True:
            _fail("invalid-required-check")
        if key == "deep.source_bytes":
            deep = normalized
    if not SOURCE_RETIREMENT_REQUIRED_CHECK_KEYS.issubset(keys):
        _fail("missing-required-check")
    if deep is None or deep.get("status") != "pass" or deep.get("mandatory") is not True:
        _fail("invalid-deep-check")
    deep_expected = deep.get("expected")
    if not (
        (type(deep_expected) is int and deep_expected >= 0) or deep_expected == _CARRIED_EXPECTED
    ):
        _fail("invalid-deep-check")
    return checks, deep


def _expected_gate(
    checks: list[dict[str, object]], artifacts: Mapping[str, object]
) -> dict[str, object]:
    non_pass = mandatory_non_pass_checks(checks)
    source = artifacts.get("source_manifests")
    marker = artifacts.get("library_marker")
    artifacts_complete = (
        isinstance(source, dict)
        and all(value is not None for value in source.values())
        and marker is not None
    )
    deep_verified = any(
        check.get("key") == "deep.source_bytes" and check.get("status") == "pass"
        for check in checks
    )
    library_verified = any(
        check.get("key") == "library.completion_marker" and check.get("status") == "pass"
        for check in checks
    )
    return {
        "all_mandatory_checks_pass": not non_pass,
        "deep_verified": deep_verified,
        "library_verified": library_verified,
        "required_artifacts_present": artifacts_complete,
        "source_retirement_eligible": (
            not non_pass and deep_verified and library_verified and artifacts_complete
        ),
        "non_pass_keys": [str(check.get("key", "unknown")) for check in non_pass],
    }


def _validate_receipt(
    payload: bytes, *, current_artifacts: Mapping[str, object]
) -> tuple[dict[str, Any], list[dict[str, object]], dict[str, object]]:
    report = _strict_json(payload)
    if frozenset(report) != _REPORT_KEYS or type(report.get("schema_version")) is not int:
        _fail("invalid-report-schema")
    if report["schema_version"] != 1 or not isinstance(report.get("generated_at"), str):
        _fail("invalid-report-schema")
    checks, deep = _validate_checks(report.get("checks"))
    artifacts = _validate_artifacts(report.get("artifacts"))
    if artifacts != current_artifacts:
        _fail("artifact-identity-changed")
    gate = report.get("gate")
    expected_gate = _expected_gate(checks, artifacts)
    if (
        not isinstance(gate, dict)
        or frozenset(gate) != _GATE_KEYS
        or gate != expected_gate
        or gate.get("source_retirement_eligible") is not True
    ):
        _fail("invalid-report-gate")
    return report, checks, deep


def _current_artifacts(layout: ArchiveLayout) -> dict[str, object]:
    try:
        source = source_manifest_artifacts(layout)
        marker = _artifact_record(layout, layout.library_completion_marker, jsonl=False)
    except (OSError, ValueError):
        _fail("artifact-read-failed")
    artifacts: dict[str, object] = {
        "source_manifests": source,
        "library_marker": marker,
    }
    _validate_artifacts(artifacts)
    return artifacts


def _carried_prior_digest(deep: Mapping[str, object]) -> str | None:
    actual = deep.get("actual")
    detail = deep.get("detail")
    if not isinstance(actual, str) or not isinstance(detail, str):
        return None
    match = _CARRIED_ACTUAL_RE.fullmatch(actual)
    has_marker_text = "carried-forward" in actual or "prior_report_sha256=" in detail
    if match is None:
        if has_marker_text:
            _fail("invalid-carried-provenance")
        return None
    digest = match.group(1)
    expected_detail = (
        "Deep source-byte verification carried forward after exact artifact matching and a "
        f"passing current non-deep catalog verification; prior_report_sha256={digest}"
    )
    if detail != expected_detail or deep.get("expected") != _CARRIED_EXPECTED:
        _fail("invalid-carried-provenance")
    return digest


def _audit_name(digest: str) -> str:
    if _SHA256_RE.fullmatch(digest) is None:
        _fail("invalid-prior-digest")
    return f"{AUDIT_NAME_PREFIX}{digest}.json"


def _validate_provenance_chain(
    directory_descriptor: int,
    deep: Mapping[str, object],
    *,
    current_artifacts: Mapping[str, object],
) -> None:
    seen: set[str] = set()
    current_deep = deep
    for _depth in range(MAX_PROVENANCE_DEPTH):
        digest = _carried_prior_digest(current_deep)
        if digest is None:
            expected = current_deep.get("expected")
            if type(expected) is not int or expected < 0:
                _fail("invalid-direct-deep-check")
            return
        if digest in seen:
            _fail("cyclic-carried-provenance")
        seen.add(digest)
        prior = _safe_read_report(directory_descriptor, _audit_name(digest))
        if prior.sha256 != digest:
            _fail("audit-digest-mismatch")
        _report, _checks, current_deep = _validate_receipt(
            prior.payload,
            current_artifacts=current_artifacts,
        )
    _fail("carried-provenance-too-deep")


def _make_carried_deep(deep: Mapping[str, object], *, prior_digest: str) -> dict[str, object]:
    if deep.get("key") != "deep.source_bytes" or deep.get("status") != "pass":
        _fail("invalid-deep-check")
    return {
        "key": "deep.source_bytes",
        "status": "pass",
        "expected": _CARRIED_EXPECTED,
        "actual": f"carried-forward; prior_report_sha256={prior_digest}",
        "detail": (
            "Deep source-byte verification carried forward after exact artifact matching and a "
            f"passing current non-deep catalog verification; prior_report_sha256={prior_digest}"
        ),
        "mandatory": True,
    }


def _validate_current_checks(
    current: object,
    *,
    prior_checks: Sequence[Mapping[str, object]],
    prior_deep: Mapping[str, object],
) -> list[dict[str, object]]:
    if not isinstance(current, list) or not current:
        _fail("current-verification-invalid")
    normalized: list[dict[str, object]] = []
    keys: set[str] = set()
    for candidate in current:
        if not isinstance(candidate, dict) or frozenset(candidate) != _CHECK_KEYS:
            _fail("current-verification-invalid")
        key = candidate.get("key")
        expected = candidate.get("expected")
        actual = candidate.get("actual")
        if (
            not isinstance(key, str)
            or not key
            or key in keys
            or key == "deep.source_bytes"
            or candidate.get("status") not in {"pass", "pending", "fail"}
            or type(candidate.get("mandatory")) is not bool
            or not isinstance(candidate.get("detail"), str)
            or not (expected is None or isinstance(expected, str) or type(expected) is int)
            or not (actual is None or isinstance(actual, str) or type(actual) is int)
        ):
            _fail("current-verification-invalid")
        keys.add(key)
        normalized_candidate = cast(dict[str, object], candidate)
        if key in SOURCE_RETIREMENT_REQUIRED_CHECK_KEYS and candidate.get("mandatory") is not True:
            _fail("current-verification-invalid")
        normalized.append(normalized_candidate)
    prior_non_deep_keys = {
        str(check.get("key")) for check in prior_checks if check.get("key") != "deep.source_bytes"
    }
    if keys != prior_non_deep_keys:
        _fail("check-identity-changed")
    combined = [*normalized, dict(prior_deep)]
    if mandatory_non_pass_checks(combined):
        _fail("current-verification-nonpass")
    return normalized


def _write_all(descriptor: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            _fail("audit-write-failed")
        remaining = remaining[written:]


def _rename_noreplace(
    directory_descriptor: int,
    source_name: str,
    destination_name: str,
) -> bool:
    """Atomically publish one staged name without ever replacing the destination."""

    libc: Any = ctypes.CDLL(None, use_errno=True)
    operation: Any = getattr(libc, "renameat2", None)
    if operation is None:
        _fail("atomic-no-replace-unavailable")
    operation.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    operation.restype = ctypes.c_int
    result = operation(
        directory_descriptor,
        os.fsencode(source_name),
        directory_descriptor,
        os.fsencode(destination_name),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return True
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        return False
    _fail("atomic-audit-publication-failed")


def _preserve_prior(directory_descriptor: int, prior: _BoundedFile) -> None:
    name = _audit_name(prior.sha256)
    try:
        os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        pass
    except OSError:
        _fail("audit-copy-read-failed")
    else:
        existing = _safe_read_report(directory_descriptor, name)
        if existing.payload != prior.payload or existing.sha256 != prior.sha256:
            _fail("audit-copy-collision")
        return

    staging_name = f"rebind-{secrets.token_hex(12)}.audit.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | int(getattr(os, "O_CLOEXEC", 0))
    try:
        descriptor = os.open(staging_name, flags, 0o600, dir_fd=directory_descriptor)
    except OSError:
        _fail("audit-copy-create-failed")
    published = False
    try:
        os.fchmod(descriptor, 0o600)
        _write_all(descriptor, prior.payload)
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
        linked = os.stat(staging_name, dir_fd=directory_descriptor, follow_symlinks=False)
        _validate_private_regular(opened)
        _validate_private_regular(linked)
        if (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino):
            _fail("audit-copy-substitution")
        staged = _safe_read_report(directory_descriptor, staging_name)
        if staged.payload != prior.payload or staged.sha256 != prior.sha256:
            _fail("audit-copy-verification-failed")
        published = _rename_noreplace(directory_descriptor, staging_name, name)
        if not published:
            existing = _safe_read_report(directory_descriptor, name)
            if existing.payload != prior.payload or existing.sha256 != prior.sha256:
                _fail("audit-copy-collision")
        os.fsync(directory_descriptor)
    except OSError:
        _fail("audit-copy-write-failed")
    finally:
        if not published:
            _unlink_own_file(directory_descriptor, staging_name, descriptor)
        os.close(descriptor)
    preserved = _safe_read_report(directory_descriptor, name)
    if preserved.payload != prior.payload or preserved.sha256 != prior.sha256:
        _fail("audit-copy-verification-failed")


def _identity_matches(value: os.stat_result, descriptor_value: os.stat_result) -> bool:
    return (value.st_dev, value.st_ino) == (descriptor_value.st_dev, descriptor_value.st_ino)


def _unlink_own_file(directory_descriptor: int, name: str, descriptor: int) -> None:
    try:
        path_value = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        descriptor_value = os.fstat(descriptor)
        if _identity_matches(path_value, descriptor_value) and stat.S_ISREG(path_value.st_mode):
            os.unlink(name, dir_fd=directory_descriptor)
            os.fsync(directory_descriptor)
    except FileNotFoundError:
        return
    except OSError:
        return


@contextmanager
def _restrictive_umask() -> Iterator[None]:
    previous = os.umask(0o077)
    try:
        yield
    finally:
        os.umask(previous)


def _publish_report(
    layout: ArchiveLayout,
    directory_descriptor: int,
    *,
    checks: list[dict[str, object]],
    current_artifacts: Mapping[str, object],
    prior: _BoundedFile,
) -> None:
    nonce = secrets.token_hex(12)
    candidate_name = f"rebind-{nonce}.candidate.json"
    candidate_tmp_name = f"rebind-{nonce}.candidate.tmp"
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | int(getattr(os, "O_CLOEXEC", 0))
    try:
        candidate_descriptor = os.open(
            candidate_tmp_name,
            flags,
            0o600,
            dir_fd=directory_descriptor,
        )
    except OSError:
        _fail("candidate-collision")
    promoted = False
    try:
        os.fchmod(candidate_descriptor, 0o600)
        with _restrictive_umask():
            written = write_verification_report(layout, checks, filename=candidate_name)
        if written.name != candidate_name:
            _fail("candidate-writer-mismatch")

        candidate_value = os.stat(
            candidate_name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        held_value = os.fstat(candidate_descriptor)
        _validate_private_regular(candidate_value)
        _validate_private_regular(held_value)
        if not _identity_matches(candidate_value, held_value):
            _fail("candidate-substitution")
        os.fsync(candidate_descriptor)

        candidate = _safe_read_report(directory_descriptor, candidate_name)
        report, report_checks, _deep = _validate_receipt(
            candidate.payload,
            current_artifacts=current_artifacts,
        )
        if report_checks != checks or report.get("gate") != _expected_gate(
            checks, current_artifacts
        ):
            _fail("candidate-content-mismatch")

        latest_again = _safe_read_report(directory_descriptor, LATEST_REPORT_NAME)
        if not _same_bound_file(latest_again, prior):
            _fail("latest-report-substitution")

        os.replace(
            candidate_name,
            LATEST_REPORT_NAME,
            src_dir_fd=directory_descriptor,
            dst_dir_fd=directory_descriptor,
        )
        os.fsync(directory_descriptor)
        promoted = True

        published = _safe_read_report(directory_descriptor, LATEST_REPORT_NAME)
        published_value = os.stat(
            LATEST_REPORT_NAME,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            not _identity_matches(published_value, os.fstat(candidate_descriptor))
            or published.payload != candidate.payload
        ):
            _fail("published-report-substitution")
        published_report, published_checks, _published_deep = _validate_receipt(
            published.payload,
            current_artifacts=current_artifacts,
        )
        if published_checks != checks or published_report.get("gate") != _expected_gate(
            checks, current_artifacts
        ):
            _fail("published-report-invalid")
    except OSError:
        _fail("candidate-publication-failed")
    finally:
        if not promoted:
            _unlink_own_file(directory_descriptor, candidate_tmp_name, candidate_descriptor)
            _unlink_own_file(directory_descriptor, candidate_name, candidate_descriptor)
        os.close(candidate_descriptor)


def rebind_deep_receipt(archive_root: Path) -> bool:
    """Rebind ``latest.json`` and return whether a new report was published."""

    layout = ArchiveLayout(Path(os.path.abspath(archive_root)))
    engine: Engine | None = None
    try:
        # Match the global destructive-publication prefix. Holding both locks across catalog
        # verification, report generation, fsync, and atomic promotion prevents a deletion from
        # changing the certified catalog between the fresh checks and receipt publication.
        with deletion_lock(layout), source_import_publication_lock(layout):
            reports_descriptor = open_directory_beneath(
                layout.manifests / "integrity-reports",
                layout.root,
            )
            try:
                _validate_private_directory(reports_descriptor)
                prior = _safe_read_report(reports_descriptor, LATEST_REPORT_NAME)
                artifacts_before = _current_artifacts(layout)
                _prior_report, prior_checks, prior_deep = _validate_receipt(
                    prior.payload,
                    current_artifacts=artifacts_before,
                )
                _validate_provenance_chain(
                    reports_descriptor,
                    prior_deep,
                    current_artifacts=artifacts_before,
                )

                try:
                    engine = create_catalog_engine(
                        layout.database,
                        read_only=True,
                        archive_root=layout.root,
                    )
                    require_current_migration(engine)
                    current_raw = verify_catalog(layout, engine, deep=False)
                except Exception:
                    _fail("current-catalog-verification-failed")
                current_checks = _validate_current_checks(
                    current_raw,
                    prior_checks=prior_checks,
                    prior_deep=prior_deep,
                )
                artifacts_after = _current_artifacts(layout)
                if artifacts_after != artifacts_before:
                    _fail("artifact-substitution")
                latest_again = _safe_read_report(reports_descriptor, LATEST_REPORT_NAME)
                if not _same_bound_file(latest_again, prior):
                    _fail("latest-report-substitution")

                prior_digest = _carried_prior_digest(prior_deep)
                prior_non_deep = [
                    check for check in prior_checks if check.get("key") != "deep.source_bytes"
                ]
                if prior_digest is not None and prior_non_deep == current_checks:
                    return False

                _preserve_prior(reports_descriptor, prior)
                carried_deep = _make_carried_deep(prior_deep, prior_digest=prior.sha256)
                new_checks = [*current_checks, carried_deep]
                if mandatory_non_pass_checks(new_checks):
                    _fail("rebound-gate-nonpass")
                _publish_report(
                    layout,
                    reports_descriptor,
                    checks=new_checks,
                    current_artifacts=artifacts_after,
                    prior=prior,
                )
                return True
            finally:
                os.close(reports_descriptor)
    except ReceiptRebindError:
        raise
    except Exception:
        _fail("receipt-rebind-failed")
    finally:
        if engine is not None:
            engine.dispose()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rebind a passing deep receipt after a catalog-only migration."
    )
    parser.add_argument("--archive-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        published = rebind_deep_receipt(arguments.archive_root)
    except Exception:
        print("deep receipt rebind: failed validation", file=sys.stderr)
        return 1
    status_value = "published=1" if published else "published=0 already_current=1"
    print(f"deep receipt rebind: passed {status_value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
