from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from meta_archive.paths import (
    UnsafeArchivePathError,
    archive_entry_stat_beneath,
    open_regular_file_beneath,
)

AUDITED_BASELINE_SCHEMA_VERSION = 1
AUDITED_BASELINE_MANIFEST_RELATIVE_PATH = Path("manifests/audited-baseline.json")
MAX_AUDITED_BASELINE_MANIFEST_BYTES = 64 * 1024
MAX_AUDITED_BASELINE_METRIC = (1 << 63) - 1


class AuditedBaselineManifestError(ValueError):
    """The private audited-baseline manifest is present but cannot be trusted."""


@dataclass(frozen=True, slots=True)
class AuditedBaseline:
    chunks: int
    directories: int
    files: int
    bytes: int
    json_files: int
    transcript_containers: int
    ordinary_segments: int
    ordinary_messages: int
    canonical_participants: int
    message_reactions: int
    message_attachments: int
    support_segments: int
    support_messages: int
    activity_records: int
    exact_duplicate_messages: int
    normalized_path_collisions: int
    unique_blobs: int
    unique_blob_bytes: int
    duplicate_file_occurrences: int
    duplicate_bytes: int
    unresolved_local_uri_occurrences: int
    unlinked_media_paths: int
    leading_message_gap_segments: int
    internal_message_gap_segments: int
    activity_reactions_raw: int
    activity_reactions_logical: int
    largest_conversation_messages: int


def audited_baseline_manifest_path(archive_root: Path) -> Path:
    """Return the fixed archive-local location for private audited counts."""

    return archive_root / AUDITED_BASELINE_MANIFEST_RELATIVE_PATH


def archive_identity_sha256(archive_root: Path) -> str:
    """Bind a private manifest to an archive without storing its directory name in source."""

    return hashlib.sha256(archive_root.name.encode("utf-8")).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AuditedBaselineManifestError("Audited baseline manifest has duplicate keys")
        result[key] = value
    return result


def _validate_payload(payload: Any, archive_root: Path) -> AuditedBaseline:
    if not isinstance(payload, dict):
        raise AuditedBaselineManifestError("Audited baseline manifest must be a JSON object")
    expected_top_level = {"schema_version", "archive_identity_sha256", "metrics"}
    if set(payload) != expected_top_level:
        raise AuditedBaselineManifestError(
            "Audited baseline manifest has missing or unknown top-level fields"
        )
    if type(payload["schema_version"]) is not int or (
        payload["schema_version"] != AUDITED_BASELINE_SCHEMA_VERSION
    ):
        raise AuditedBaselineManifestError("Unsupported audited baseline schema version")

    identity = payload["archive_identity_sha256"]
    if (
        not isinstance(identity, str)
        or len(identity) != 64
        or any(character not in "0123456789abcdef" for character in identity)
        or identity != archive_identity_sha256(archive_root)
    ):
        raise AuditedBaselineManifestError("Audited baseline archive identity does not match")

    metrics = payload["metrics"]
    if not isinstance(metrics, dict):
        raise AuditedBaselineManifestError("Audited baseline metrics must be a JSON object")
    expected_metrics = {field.name for field in fields(AuditedBaseline)}
    if set(metrics) != expected_metrics:
        raise AuditedBaselineManifestError(
            "Audited baseline metrics have missing or unknown fields"
        )
    for name, value in metrics.items():
        if type(value) is not int or not 0 <= value <= MAX_AUDITED_BASELINE_METRIC:
            raise AuditedBaselineManifestError(
                f"Audited baseline metric {name!r} must be a non-negative 64-bit integer"
            )
    return AuditedBaseline(**metrics)


def load_audited_baseline(archive_root: Path) -> AuditedBaseline | None:
    """Load a private archive-local baseline, returning ``None`` when it is absent.

    A present manifest is always treated as security-sensitive configuration: it must be an
    owner-only regular file beneath the archive root and match the exact versioned schema.
    Malformed or unsafe manifests fail closed rather than silently disabling reconciliation.
    """

    path = audited_baseline_manifest_path(archive_root)
    try:
        manifest_stat = archive_entry_stat_beneath(path, archive_root)
        if manifest_stat is None:
            return None
        if not stat.S_ISREG(manifest_stat.st_mode):
            raise AuditedBaselineManifestError(
                "Audited baseline manifest must be a regular, non-symlink file"
            )
        if stat.S_IMODE(manifest_stat.st_mode) & 0o077:
            raise AuditedBaselineManifestError("Audited baseline manifest must be owner-only")
        if manifest_stat.st_size > MAX_AUDITED_BASELINE_MANIFEST_BYTES:
            raise AuditedBaselineManifestError("Audited baseline manifest is too large")
        with open_regular_file_beneath(path, archive_root) as handle:
            opened_stat = os.fstat(handle.fileno())
            if not stat.S_ISREG(opened_stat.st_mode):
                raise AuditedBaselineManifestError(
                    "Audited baseline manifest must be a regular, non-symlink file"
                )
            if stat.S_IMODE(opened_stat.st_mode) & 0o077:
                raise AuditedBaselineManifestError("Audited baseline manifest must be owner-only")
            if opened_stat.st_size > MAX_AUDITED_BASELINE_MANIFEST_BYTES:
                raise AuditedBaselineManifestError("Audited baseline manifest is too large")
            if (opened_stat.st_dev, opened_stat.st_ino) != (
                manifest_stat.st_dev,
                manifest_stat.st_ino,
            ):
                raise AuditedBaselineManifestError(
                    "Audited baseline manifest changed during secure open"
                )
            raw = handle.read(MAX_AUDITED_BASELINE_MANIFEST_BYTES + 1)
    except AuditedBaselineManifestError:
        raise
    except (OSError, UnsafeArchivePathError) as exc:
        raise AuditedBaselineManifestError(
            "Audited baseline manifest could not be opened safely"
        ) from exc
    if len(raw) > MAX_AUDITED_BASELINE_MANIFEST_BYTES:
        raise AuditedBaselineManifestError("Audited baseline manifest is too large")
    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except AuditedBaselineManifestError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditedBaselineManifestError("Audited baseline manifest is not valid JSON") from exc
    return _validate_payload(payload, archive_root)
