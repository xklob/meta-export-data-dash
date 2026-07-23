from __future__ import annotations

import json
import os
import stat

from meta_archive.paths import (
    ArchiveLayout,
    UnsafeArchivePathError,
    open_regular_file_beneath,
    validate_directory_tree_beneath,
)

_MAX_LANCE_MANIFEST_HINT_BYTES = 256
_MAX_LANCE_VERSION = (1 << 64) - 1


def require_existing_lance_manifest(layout: ArchiveLayout) -> None:
    """Reject a Lance namespace that the installed reader would lazily initialize.

    Lance's local namespace format stores the logical version in a small JSON hint and names
    the corresponding manifest with the inverse unsigned-64-bit version. Validating both files
    before any ``lancedb.connect`` call prevents a read request from creating ``__manifest``
    metadata for an empty or partially initialized vector directory.
    """

    validate_directory_tree_beneath(layout.vectors, layout.root)
    hint_path = layout.vectors / "__manifest" / "_versions" / "latest_version_hint.json"
    with open_regular_file_beneath(hint_path, layout.root) as hint_handle:
        opened_hint = os.fstat(hint_handle.fileno())
        hint_bytes = hint_handle.read(_MAX_LANCE_MANIFEST_HINT_BYTES + 1)
        final_hint = os.fstat(hint_handle.fileno())
    if (
        not stat.S_ISREG(opened_hint.st_mode)
        or len(hint_bytes) > _MAX_LANCE_MANIFEST_HINT_BYTES
        or (
            opened_hint.st_dev,
            opened_hint.st_ino,
            opened_hint.st_size,
            opened_hint.st_mtime_ns,
            opened_hint.st_ctime_ns,
        )
        != (
            final_hint.st_dev,
            final_hint.st_ino,
            final_hint.st_size,
            final_hint.st_mtime_ns,
            final_hint.st_ctime_ns,
        )
    ):
        raise UnsafeArchivePathError("Lance manifest hint is unsafe or changed during validation")
    try:
        hint = json.loads(hint_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UnsafeArchivePathError("Lance manifest hint is malformed") from exc
    if not isinstance(hint, dict) or set(hint) != {"version"}:
        raise UnsafeArchivePathError("Lance manifest hint has an unsupported shape")
    version = hint["version"]
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or not 1 <= version <= _MAX_LANCE_VERSION
    ):
        raise UnsafeArchivePathError("Lance manifest hint has an invalid version")

    manifest_path = (
        layout.vectors / "__manifest" / "_versions" / f"{_MAX_LANCE_VERSION - version}.manifest"
    )
    with open_regular_file_beneath(manifest_path, layout.root) as manifest_handle:
        opened_manifest = os.fstat(manifest_handle.fileno())
        first_byte = manifest_handle.read(1)
        final_manifest = os.fstat(manifest_handle.fileno())
    if not first_byte or (
        opened_manifest.st_dev,
        opened_manifest.st_ino,
        opened_manifest.st_size,
        opened_manifest.st_mtime_ns,
        opened_manifest.st_ctime_ns,
    ) != (
        final_manifest.st_dev,
        final_manifest.st_ino,
        final_manifest.st_size,
        final_manifest.st_mtime_ns,
        final_manifest.st_ctime_ns,
    ):
        raise UnsafeArchivePathError("Lance current manifest is empty or changed during validation")
