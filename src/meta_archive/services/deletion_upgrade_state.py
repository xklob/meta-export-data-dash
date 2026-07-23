from __future__ import annotations

import base64
import binascii
import hashlib
import math
import os
import secrets
import stat
import uuid
from collections.abc import Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

import orjson

from meta_archive.paths import ArchiveLayout, open_directory_beneath, open_regular_file_beneath
from meta_archive.services.purge_rewrite import (
    BINARY_TOMBSTONE_BYTES,
    FingerprintUpdate,
    RawRewritePlan,
    RawSourceRewrite,
)

WORKSPACE_UPGRADE_STATE_VERSION = 1
WORKSPACE_UPGRADE_STATE_KIND = "meta-archive-workspace-upgrade-state"
WORKSPACE_UPGRADE_FILE_MODE = 0o600

_HEX_CHARACTERS = frozenset("0123456789abcdef")
_RESOURCE_FIELDS = frozenset(
    {
        "blob_ids",
        "shared_blob_ids",
        "recoverably_referenced_blob_ids",
        "exclusive_blob_ids",
        "physically_exclusive_blob_ids",
        "search_window_ids",
        "purge_search_window_ids",
        "audio_segment_ids",
        "attachment_extraction_ids",
        "physical_audio_segment_ids",
        "physical_attachment_extraction_ids",
        "source_record_ids",
        "conversation_participant_ids",
        "conversation_ids",
        "source_file_ids",
    }
)
_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "relative_path",
        "operation_id",
        "selection_sha256",
        "purge_revision",
        "state_sha256",
        "file_sha256",
        "size_bytes",
        "device",
        "inode",
        "mtime_ns",
    }
)


class DeletionUpgradeStateError(RuntimeError):
    """Saved normalized-purge state is missing, unsafe, or internally inconsistent."""


class DeletionUpgradeStateExistsError(DeletionUpgradeStateError):
    """An immutable upgrade-state destination already exists."""


def _require_exact_keys(value: Mapping[str, Any], expected: frozenset[str], *, label: str) -> None:
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise DeletionUpgradeStateError(
            f"Invalid {label} fields (missing={missing}, unexpected={unexpected})"
        )


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX_CHARACTERS for character in value)
    ):
        raise DeletionUpgradeStateError(f"Invalid {label} SHA-256")
    return value


def _require_identifier(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 1_024
        or "\x00" in value
        or any(ord(character) < 0x20 for character in value)
    ):
        raise DeletionUpgradeStateError(f"Invalid {label} identifier")
    return value


def _require_operation_id(value: object) -> str:
    if not isinstance(value, str):
        raise DeletionUpgradeStateError("Invalid workspace-upgrade operation ID")
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise DeletionUpgradeStateError("Invalid workspace-upgrade operation ID") from exc
    if str(parsed) != value:
        raise DeletionUpgradeStateError("Workspace-upgrade operation ID is not canonical")
    return value


def _require_nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DeletionUpgradeStateError(f"Invalid {label}")
    return value


def _require_positive_int(value: object, *, label: str) -> int:
    result = _require_nonnegative_int(value, label=label)
    if result == 0:
        raise DeletionUpgradeStateError(f"Invalid {label}")
    return result


def _validate_json_value(value: Any, *, label: str) -> None:
    if value is None or isinstance(value, bool | str):
        return
    if isinstance(value, int):
        if not -(2**63) <= value < 2**64:
            raise DeletionUpgradeStateError(f"{label} contains an out-of-range integer")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DeletionUpgradeStateError(f"{label} contains a non-finite number")
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_value(item, label=label)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or "\x00" in key:
                raise DeletionUpgradeStateError(f"{label} contains an invalid object key")
            _validate_json_value(item, label=label)
        return
    raise DeletionUpgradeStateError(f"{label} contains a non-JSON value")


def _canonical_json(value: Any) -> bytes:
    _validate_json_value(value, label="workspace-upgrade state")
    try:
        return orjson.dumps(value, option=orjson.OPT_SORT_KEYS)
    except (TypeError, orjson.JSONEncodeError) as exc:
        raise DeletionUpgradeStateError("Workspace-upgrade state is not serializable") from exc


def _safe_relative_text(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise DeletionUpgradeStateError(f"Invalid {label} path")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or value in {"", "."}
        or "\x00" in value
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise DeletionUpgradeStateError(f"Unsafe {label} path")
    canonical = pure.as_posix()
    if canonical != value:
        raise DeletionUpgradeStateError(f"Non-canonical {label} path")
    return canonical


def _archive_relative(path: Path, root: Path, *, label: str) -> str:
    if ".." in path.parts:
        raise DeletionUpgradeStateError(f"Non-canonical {label} path")
    absolute_root = Path(os.path.abspath(root))
    absolute_path = Path(os.path.abspath(path))
    try:
        relative = absolute_path.relative_to(absolute_root)
    except ValueError as exc:
        raise DeletionUpgradeStateError(f"{label} is outside the archive") from exc
    return _safe_relative_text(relative.as_posix(), label=label)


def _archive_path(root: Path, relative: object, *, label: str) -> Path:
    safe = _safe_relative_text(relative, label=label)
    return Path(os.path.abspath(root)) / Path(*PurePosixPath(safe).parts)


def _validate_json_pointer(value: object) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise DeletionUpgradeStateError("Invalid raw JSON pointer")
    if value == "":
        return value
    if not value.startswith("/"):
        raise DeletionUpgradeStateError("Invalid raw JSON pointer")
    index = 0
    while index < len(value):
        if value[index] != "~":
            index += 1
            continue
        if index + 1 >= len(value) or value[index + 1] not in {"0", "1"}:
            raise DeletionUpgradeStateError("Invalid raw JSON pointer escape")
        index += 2
    return value


def _canonical_identifiers(values: Iterable[object], *, label: str) -> tuple[str, ...]:
    result = tuple(sorted({_require_identifier(value, label=label) for value in values}))
    return result


def _canonical_sha256_values(values: Iterable[object], *, label: str) -> tuple[str, ...]:
    return tuple(sorted({_require_sha256(value, label=label) for value in values}))


def _canonical_source_file_ids(values: Iterable[object]) -> tuple[int, ...]:
    converted: set[int] = set()
    for value in values:
        if isinstance(value, str):
            if not value.isdigit():
                raise DeletionUpgradeStateError("Invalid source-file ID")
            value = int(value)
        converted.add(_require_positive_int(value, label="source-file ID"))
    return tuple(sorted(converted))


@dataclass(frozen=True, slots=True)
class SavedDeletionResources:
    """Content-free ownership and cleanup coordinates retained after normalized purge."""

    blob_ids: tuple[str, ...]
    shared_blob_ids: tuple[str, ...]
    recoverably_referenced_blob_ids: tuple[str, ...]
    exclusive_blob_ids: tuple[str, ...]
    physically_exclusive_blob_ids: tuple[str, ...]
    search_window_ids: tuple[str, ...]
    purge_search_window_ids: tuple[str, ...]
    audio_segment_ids: tuple[str, ...]
    attachment_extraction_ids: tuple[str, ...]
    physical_audio_segment_ids: tuple[str, ...]
    physical_attachment_extraction_ids: tuple[str, ...]
    source_record_ids: tuple[str, ...]
    conversation_participant_ids: tuple[str, ...]
    conversation_ids: tuple[str, ...]
    source_file_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        canonical_fields: tuple[tuple[object, ...], ...] = (
            _canonical_sha256_values(self.blob_ids, label="blob"),
            _canonical_sha256_values(self.shared_blob_ids, label="shared blob"),
            _canonical_sha256_values(
                self.recoverably_referenced_blob_ids, label="recoverably referenced blob"
            ),
            _canonical_sha256_values(self.exclusive_blob_ids, label="exclusive blob"),
            _canonical_sha256_values(
                self.physically_exclusive_blob_ids, label="physically exclusive blob"
            ),
            _canonical_identifiers(self.search_window_ids, label="search window"),
            _canonical_identifiers(self.purge_search_window_ids, label="physical purge window"),
            _canonical_identifiers(self.audio_segment_ids, label="audio segment"),
            _canonical_identifiers(self.attachment_extraction_ids, label="attachment extraction"),
            _canonical_identifiers(self.physical_audio_segment_ids, label="physical audio segment"),
            _canonical_identifiers(
                self.physical_attachment_extraction_ids,
                label="physical attachment extraction",
            ),
            _canonical_identifiers(self.source_record_ids, label="source record"),
            _canonical_identifiers(
                self.conversation_participant_ids, label="conversation participant"
            ),
            _canonical_identifiers(self.conversation_ids, label="conversation"),
            _canonical_source_file_ids(self.source_file_ids),
        )
        actual_fields: tuple[tuple[object, ...], ...] = (
            self.blob_ids,
            self.shared_blob_ids,
            self.recoverably_referenced_blob_ids,
            self.exclusive_blob_ids,
            self.physically_exclusive_blob_ids,
            self.search_window_ids,
            self.purge_search_window_ids,
            self.audio_segment_ids,
            self.attachment_extraction_ids,
            self.physical_audio_segment_ids,
            self.physical_attachment_extraction_ids,
            self.source_record_ids,
            self.conversation_participant_ids,
            self.conversation_ids,
            self.source_file_ids,
        )
        if canonical_fields != actual_fields:
            raise DeletionUpgradeStateError("Saved deletion resources are not canonical")
        blobs = set(self.blob_ids)
        shared = set(self.shared_blob_ids)
        recoverable = set(self.recoverably_referenced_blob_ids)
        exclusive = set(self.exclusive_blob_ids)
        physical = set(self.physically_exclusive_blob_ids)
        if not shared <= recoverable <= blobs:
            raise DeletionUpgradeStateError("Saved blob ownership sets are inconsistent")
        if exclusive != blobs - shared or physical != blobs - recoverable:
            raise DeletionUpgradeStateError("Saved blob exclusivity sets are inconsistent")
        if not set(self.purge_search_window_ids) <= set(self.search_window_ids):
            raise DeletionUpgradeStateError("Physical purge windows exceed logical windows")
        if not set(self.physical_audio_segment_ids) <= set(self.audio_segment_ids):
            raise DeletionUpgradeStateError("Physical audio targets exceed logical targets")
        if not set(self.physical_attachment_extraction_ids) <= set(self.attachment_extraction_ids):
            raise DeletionUpgradeStateError("Physical attachment targets exceed logical targets")

    @classmethod
    def create(
        cls,
        *,
        blob_ids: Iterable[object],
        shared_blob_ids: Iterable[object],
        recoverably_referenced_blob_ids: Iterable[object],
        exclusive_blob_ids: Iterable[object],
        physically_exclusive_blob_ids: Iterable[object],
        search_window_ids: Iterable[object],
        purge_search_window_ids: Iterable[object],
        audio_segment_ids: Iterable[object],
        attachment_extraction_ids: Iterable[object],
        physical_audio_segment_ids: Iterable[object],
        physical_attachment_extraction_ids: Iterable[object],
        source_record_ids: Iterable[object],
        conversation_participant_ids: Iterable[object],
        conversation_ids: Iterable[object],
        source_file_ids: Iterable[object],
    ) -> SavedDeletionResources:
        return cls(
            blob_ids=_canonical_sha256_values(blob_ids, label="blob"),
            shared_blob_ids=_canonical_sha256_values(shared_blob_ids, label="shared blob"),
            recoverably_referenced_blob_ids=_canonical_sha256_values(
                recoverably_referenced_blob_ids, label="recoverably referenced blob"
            ),
            exclusive_blob_ids=_canonical_sha256_values(exclusive_blob_ids, label="exclusive blob"),
            physically_exclusive_blob_ids=_canonical_sha256_values(
                physically_exclusive_blob_ids, label="physically exclusive blob"
            ),
            search_window_ids=_canonical_identifiers(search_window_ids, label="search window"),
            purge_search_window_ids=_canonical_identifiers(
                purge_search_window_ids, label="physical purge window"
            ),
            audio_segment_ids=_canonical_identifiers(audio_segment_ids, label="audio segment"),
            attachment_extraction_ids=_canonical_identifiers(
                attachment_extraction_ids, label="attachment extraction"
            ),
            physical_audio_segment_ids=_canonical_identifiers(
                physical_audio_segment_ids, label="physical audio segment"
            ),
            physical_attachment_extraction_ids=_canonical_identifiers(
                physical_attachment_extraction_ids, label="physical attachment extraction"
            ),
            source_record_ids=_canonical_identifiers(source_record_ids, label="source record"),
            conversation_participant_ids=_canonical_identifiers(
                conversation_participant_ids, label="conversation participant"
            ),
            conversation_ids=_canonical_identifiers(conversation_ids, label="conversation"),
            source_file_ids=_canonical_source_file_ids(source_file_ids),
        )

    @classmethod
    def from_selection_resources(
        cls,
        resources: Mapping[str, Iterable[object]],
        *,
        source_record_ids: Iterable[object],
    ) -> SavedDeletionResources:
        required = _RESOURCE_FIELDS - {"source_record_ids"}
        missing = sorted(field for field in required if field not in resources)
        if missing:
            raise DeletionUpgradeStateError(f"Selection resources are incomplete: {missing}")
        return cls.create(
            **{field: resources[field] for field in required},
            source_record_ids=source_record_ids,
        )

    def to_json(self) -> dict[str, list[str] | list[int]]:
        return {
            "blob_ids": list(self.blob_ids),
            "shared_blob_ids": list(self.shared_blob_ids),
            "recoverably_referenced_blob_ids": list(self.recoverably_referenced_blob_ids),
            "exclusive_blob_ids": list(self.exclusive_blob_ids),
            "physically_exclusive_blob_ids": list(self.physically_exclusive_blob_ids),
            "search_window_ids": list(self.search_window_ids),
            "purge_search_window_ids": list(self.purge_search_window_ids),
            "audio_segment_ids": list(self.audio_segment_ids),
            "attachment_extraction_ids": list(self.attachment_extraction_ids),
            "physical_audio_segment_ids": list(self.physical_audio_segment_ids),
            "physical_attachment_extraction_ids": list(self.physical_attachment_extraction_ids),
            "source_record_ids": list(self.source_record_ids),
            "conversation_participant_ids": list(self.conversation_participant_ids),
            "conversation_ids": list(self.conversation_ids),
            "source_file_ids": list(self.source_file_ids),
        }

    @classmethod
    def from_json(cls, value: object) -> SavedDeletionResources:
        if not isinstance(value, dict):
            raise DeletionUpgradeStateError("Saved deletion resources are not an object")
        _require_exact_keys(value, _RESOURCE_FIELDS, label="saved deletion resources")
        if any(not isinstance(value[field], list) for field in _RESOURCE_FIELDS):
            raise DeletionUpgradeStateError("Saved deletion resource fields must be arrays")
        return cls.create(**{field: value[field] for field in _RESOURCE_FIELDS})


@dataclass(frozen=True, slots=True)
class WorkspaceUpgradeStateReceipt:
    schema_version: int
    relative_path: str
    operation_id: str
    selection_sha256: str
    purge_revision: str
    state_sha256: str
    file_sha256: str
    size_bytes: int
    device: int
    inode: int
    mtime_ns: int

    def __post_init__(self) -> None:
        if self.schema_version != WORKSPACE_UPGRADE_STATE_VERSION:
            raise DeletionUpgradeStateError("Unsupported workspace-upgrade receipt version")
        _safe_relative_text(self.relative_path, label="upgrade-state")
        _require_operation_id(self.operation_id)
        _require_sha256(self.selection_sha256, label="selection")
        _require_sha256(self.purge_revision, label="purge revision")
        _require_sha256(self.state_sha256, label="state")
        _require_sha256(self.file_sha256, label="upgrade-state file")
        _require_positive_int(self.size_bytes, label="receipt file size")
        _require_nonnegative_int(self.device, label="receipt device")
        _require_positive_int(self.inode, label="receipt inode")
        _require_nonnegative_int(self.mtime_ns, label="receipt mtime")

    def to_json(self) -> dict[str, str | int]:
        return {
            "schema_version": self.schema_version,
            "relative_path": self.relative_path,
            "operation_id": self.operation_id,
            "selection_sha256": self.selection_sha256,
            "purge_revision": self.purge_revision,
            "state_sha256": self.state_sha256,
            "file_sha256": self.file_sha256,
            "size_bytes": self.size_bytes,
            "device": self.device,
            "inode": self.inode,
            "mtime_ns": self.mtime_ns,
        }

    @classmethod
    def from_json(cls, value: object) -> WorkspaceUpgradeStateReceipt:
        if not isinstance(value, dict):
            raise DeletionUpgradeStateError("Workspace-upgrade receipt is not an object")
        _require_exact_keys(value, _RECEIPT_FIELDS, label="workspace-upgrade receipt")
        version = _require_positive_int(value["schema_version"], label="receipt schema version")
        if version != WORKSPACE_UPGRADE_STATE_VERSION:
            raise DeletionUpgradeStateError("Unsupported workspace-upgrade receipt version")
        return cls(
            schema_version=version,
            relative_path=_safe_relative_text(value["relative_path"], label="upgrade-state"),
            operation_id=_require_operation_id(value["operation_id"]),
            selection_sha256=_require_sha256(value["selection_sha256"], label="selection"),
            purge_revision=_require_sha256(value["purge_revision"], label="purge revision"),
            state_sha256=_require_sha256(value["state_sha256"], label="state"),
            file_sha256=_require_sha256(value["file_sha256"], label="upgrade-state file"),
            size_bytes=_require_positive_int(value["size_bytes"], label="receipt file size"),
            device=_require_nonnegative_int(value["device"], label="receipt device"),
            inode=_require_positive_int(value["inode"], label="receipt inode"),
            mtime_ns=_require_nonnegative_int(value["mtime_ns"], label="receipt mtime"),
        )


@dataclass(frozen=True, slots=True)
class SavedWorkspaceUpgradeState:
    operation_id: str
    selection_sha256: str
    purge_revision: str
    state_sha256: str
    resources: SavedDeletionResources
    raw_plan: RawRewritePlan


def _fingerprint_update_to_json(value: FingerprintUpdate) -> dict[str, Any]:
    _require_identifier(value.source_record_id, label="fingerprint source record")
    if value.message_id is not None:
        _require_identifier(value.message_id, label="fingerprint message")
    if value.activity_id is not None:
        _require_identifier(value.activity_id, label="fingerprint activity")
    if value.message_id is not None and value.activity_id is not None:
        raise DeletionUpgradeStateError("Fingerprint update targets both message and activity")
    old_fingerprint = _require_sha256(value.old_fingerprint, label="old fingerprint")
    new_fingerprint = _require_sha256(value.new_fingerprint, label="new fingerprint")
    _validate_json_value(value.new_activity_payload, label="fingerprint activity payload")
    if value.activity_id is None and value.new_activity_payload is not None:
        raise DeletionUpgradeStateError("Non-activity fingerprint update has an activity payload")
    return {
        "source_record_id": value.source_record_id,
        "message_id": value.message_id,
        "activity_id": value.activity_id,
        "old_fingerprint": old_fingerprint,
        "new_fingerprint": new_fingerprint,
        "new_activity_payload": value.new_activity_payload,
    }


def _fingerprint_update_from_json(value: object) -> FingerprintUpdate:
    if not isinstance(value, dict):
        raise DeletionUpgradeStateError("Fingerprint update is not an object")
    fields = frozenset(
        {
            "source_record_id",
            "message_id",
            "activity_id",
            "old_fingerprint",
            "new_fingerprint",
            "new_activity_payload",
        }
    )
    _require_exact_keys(value, fields, label="fingerprint update")
    message_id = value["message_id"]
    activity_id = value["activity_id"]
    update = FingerprintUpdate(
        source_record_id=_require_identifier(
            value["source_record_id"], label="fingerprint source record"
        ),
        message_id=(
            None if message_id is None else _require_identifier(message_id, label="message")
        ),
        activity_id=(
            None if activity_id is None else _require_identifier(activity_id, label="activity")
        ),
        old_fingerprint=_require_sha256(value["old_fingerprint"], label="old fingerprint"),
        new_fingerprint=_require_sha256(value["new_fingerprint"], label="new fingerprint"),
        new_activity_payload=value["new_activity_payload"],
    )
    _fingerprint_update_to_json(update)
    return update


def _raw_rewrite_to_json(layout: ArchiveLayout, value: RawSourceRewrite) -> dict[str, Any]:
    source_file_id = _require_positive_int(value.source_file_id, label="rewrite source-file ID")
    source = _safe_relative_text(value.source, label="rewrite source")
    chunk = _safe_relative_text(value.chunk, label="rewrite chunk")
    if len(PurePosixPath(chunk).parts) != 1:
        raise DeletionUpgradeStateError("Rewrite chunk is not an archive root entry")
    relative_path = _safe_relative_text(value.relative_path, label="rewrite relative")
    if source != f"{chunk}/{relative_path}":
        raise DeletionUpgradeStateError("Rewrite source coordinate is inconsistent")
    physical_relative = (
        None
        if value.physical_path is None
        else _archive_relative(value.physical_path, layout.root, label="rewrite physical")
    )
    if physical_relative is not None and physical_relative != source:
        raise DeletionUpgradeStateError("Rewrite physical path differs from its source coordinate")
    mode = _require_nonnegative_int(value.mode, label="rewrite mode")
    if mode > 0o177777:
        raise DeletionUpgradeStateError("Rewrite mode is outside the POSIX file-mode range")
    mtime_ns = _require_nonnegative_int(value.mtime_ns, label="rewrite mtime")
    old_sha256 = _require_sha256(value.old_sha256, label="old source")
    new_sha256 = _require_sha256(value.new_sha256, label="new source")
    old_size = _require_nonnegative_int(value.old_size_bytes, label="old source size")
    new_size = _require_nonnegative_int(value.new_size_bytes, label="new source size")
    if (
        len(value.new_bytes) != new_size
        or hashlib.sha256(value.new_bytes).hexdigest() != new_sha256
    ):
        raise DeletionUpgradeStateError("Generated rewrite bytes do not match their identity")
    pointers = tuple(_validate_json_pointer(pointer) for pointer in value.pointers)
    if tuple(sorted(set(pointers), key=lambda item: (item.count("/"), item))) != pointers:
        raise DeletionUpgradeStateError("Rewrite pointers are not canonical")
    selected_records = _canonical_identifiers(
        value.selected_source_record_ids, label="selected source record"
    )
    if selected_records != value.selected_source_record_ids:
        raise DeletionUpgradeStateError("Selected source-record IDs are not canonical")
    fingerprint_updates = tuple(
        _fingerprint_update_to_json(update) for update in value.fingerprint_updates
    )
    update_ids = [str(update["source_record_id"]) for update in fingerprint_updates]
    if update_ids != sorted(set(update_ids)):
        raise DeletionUpgradeStateError("Fingerprint updates are not canonical")
    if set(update_ids) & set(selected_records):
        raise DeletionUpgradeStateError("Selected source records also have fingerprint updates")
    if value.binary:
        if (
            pointers
            or selected_records
            or fingerprint_updates
            or value.new_bytes != BINARY_TOMBSTONE_BYTES
        ):
            raise DeletionUpgradeStateError("Binary rewrite is not the canonical tombstone")
    else:
        try:
            orjson.loads(value.new_bytes)
        except orjson.JSONDecodeError as exc:
            raise DeletionUpgradeStateError("Generated JSON rewrite bytes are invalid") from exc
    return {
        "source_file_id": source_file_id,
        "source": source,
        "chunk": chunk,
        "relative_path": relative_path,
        "physical_path": physical_relative,
        "mode": mode,
        "mtime_ns": mtime_ns,
        "media_type": value.media_type,
        "old_sha256": old_sha256,
        "old_size_bytes": old_size,
        "new_sha256": new_sha256,
        "new_size_bytes": new_size,
        "new_bytes_base64": base64.b64encode(value.new_bytes).decode("ascii"),
        "binary": value.binary,
        "pointers": list(pointers),
        "selected_source_record_ids": list(selected_records),
        "fingerprint_updates": list(fingerprint_updates),
    }


_RAW_REWRITE_FIELDS = frozenset(
    {
        "source_file_id",
        "source",
        "chunk",
        "relative_path",
        "physical_path",
        "mode",
        "mtime_ns",
        "media_type",
        "old_sha256",
        "old_size_bytes",
        "new_sha256",
        "new_size_bytes",
        "new_bytes_base64",
        "binary",
        "pointers",
        "selected_source_record_ids",
        "fingerprint_updates",
    }
)


def _raw_rewrite_from_json(layout: ArchiveLayout, value: object) -> RawSourceRewrite:
    if not isinstance(value, dict):
        raise DeletionUpgradeStateError("Raw rewrite is not an object")
    _require_exact_keys(value, _RAW_REWRITE_FIELDS, label="raw rewrite")
    if not isinstance(value["new_bytes_base64"], str):
        raise DeletionUpgradeStateError("Generated rewrite bytes are not base64 text")
    try:
        new_bytes = base64.b64decode(value["new_bytes_base64"], validate=True)
    except (ValueError, binascii.Error) as exc:
        raise DeletionUpgradeStateError("Generated rewrite bytes are invalid base64") from exc
    if base64.b64encode(new_bytes).decode("ascii") != value["new_bytes_base64"]:
        raise DeletionUpgradeStateError("Generated rewrite bytes use non-canonical base64")
    physical_relative = value["physical_path"]
    physical_path = (
        None
        if physical_relative is None
        else _archive_path(layout.root, physical_relative, label="rewrite physical")
    )
    pointers = value["pointers"]
    selected_records = value["selected_source_record_ids"]
    fingerprint_updates = value["fingerprint_updates"]
    if not all(
        isinstance(field, list) for field in (pointers, selected_records, fingerprint_updates)
    ):
        raise DeletionUpgradeStateError("Raw rewrite arrays are invalid")
    if value["media_type"] is not None and not isinstance(value["media_type"], str):
        raise DeletionUpgradeStateError("Raw rewrite media type is invalid")
    if not isinstance(value["binary"], bool):
        raise DeletionUpgradeStateError("Raw rewrite binary marker is invalid")
    rewrite = RawSourceRewrite(
        source_file_id=_require_positive_int(value["source_file_id"], label="source-file ID"),
        source=_safe_relative_text(value["source"], label="rewrite source"),
        chunk=_safe_relative_text(value["chunk"], label="rewrite chunk"),
        relative_path=_safe_relative_text(value["relative_path"], label="rewrite relative"),
        physical_path=physical_path,
        mode=_require_nonnegative_int(value["mode"], label="rewrite mode"),
        mtime_ns=_require_nonnegative_int(value["mtime_ns"], label="rewrite mtime"),
        media_type=value["media_type"],
        old_sha256=_require_sha256(value["old_sha256"], label="old source"),
        old_size_bytes=_require_nonnegative_int(value["old_size_bytes"], label="old size"),
        new_sha256=_require_sha256(value["new_sha256"], label="new source"),
        new_size_bytes=_require_nonnegative_int(value["new_size_bytes"], label="new size"),
        new_bytes=new_bytes,
        binary=value["binary"],
        pointers=tuple(_validate_json_pointer(pointer) for pointer in pointers),
        selected_source_record_ids=tuple(
            _require_identifier(record, label="selected source record")
            for record in selected_records
        ),
        fingerprint_updates=tuple(
            _fingerprint_update_from_json(update) for update in fingerprint_updates
        ),
    )
    _raw_rewrite_to_json(layout, rewrite)
    return rewrite


def _raw_plan_to_json(layout: ArchiveLayout, plan: RawRewritePlan) -> dict[str, Any]:
    rewrites = tuple(_raw_rewrite_to_json(layout, rewrite) for rewrite in plan.rewrites)
    source_file_ids = [int(rewrite["source_file_id"]) for rewrite in rewrites]
    sources = [str(rewrite["source"]) for rewrite in rewrites]
    if source_file_ids != sorted(set(source_file_ids)) or len(sources) != len(set(sources)):
        raise DeletionUpgradeStateError("Raw rewrites are not canonical and unique")
    originals = tuple(plan.original_manifest_rows)
    relocations = tuple(plan.relocation_manifest_rows)
    for row in (*originals, *relocations):
        if not isinstance(row, dict):
            raise DeletionUpgradeStateError("Saved source manifest row is not an object")
        _validate_json_value(row, label="saved source manifest")
    obsolete = _canonical_sha256_values(plan.obsolete_blob_ids, label="obsolete source blob")
    if obsolete != plan.obsolete_blob_ids:
        raise DeletionUpgradeStateError("Obsolete source-blob IDs are not canonical")

    originals_by_source: dict[str, dict[str, Any]] = {}
    for row in originals:
        chunk = _safe_relative_text(row.get("chunk"), label="original manifest chunk")
        relative = _safe_relative_text(row.get("relative_path"), label="original manifest relative")
        source = f"{chunk}/{relative}"
        if source in originals_by_source:
            raise DeletionUpgradeStateError("Original source manifest has duplicate coordinates")
        originals_by_source[source] = row
    relocations_by_source: dict[str, dict[str, Any]] = {}
    for row in relocations:
        source = _safe_relative_text(row.get("source"), label="relocation manifest source")
        if source in relocations_by_source:
            raise DeletionUpgradeStateError("Relocation manifest has duplicate coordinates")
        relocations_by_source[source] = row
    if set(originals_by_source) != set(relocations_by_source):
        raise DeletionUpgradeStateError("Saved source manifests have different coordinates")
    for rewrite in rewrites:
        source = str(rewrite["source"])
        original = originals_by_source.get(source)
        relocation = relocations_by_source.get(source)
        if original is None or relocation is None:
            raise DeletionUpgradeStateError("Raw rewrite is absent from saved source manifests")
        for row in (original, relocation):
            if (
                row.get("sha256") != rewrite["new_sha256"]
                or row.get("size_bytes") != rewrite["new_size_bytes"]
            ):
                raise DeletionUpgradeStateError("Saved source manifest has stale rewrite identity")
        expected_blob = f"blobs/sha256/{str(rewrite['new_sha256'])[:2]}/{rewrite['new_sha256']}"
        if relocation.get("blob") != expected_blob:
            raise DeletionUpgradeStateError("Saved relocation manifest has an invalid blob path")
    return {
        "rewrites": list(rewrites),
        "original_manifest_rows": list(originals),
        "relocation_manifest_rows": list(relocations),
        "obsolete_blob_ids": list(obsolete),
    }


_RAW_PLAN_FIELDS = frozenset(
    {
        "rewrites",
        "original_manifest_rows",
        "relocation_manifest_rows",
        "obsolete_blob_ids",
    }
)


def _raw_plan_from_json(layout: ArchiveLayout, value: object) -> RawRewritePlan:
    if not isinstance(value, dict):
        raise DeletionUpgradeStateError("Saved raw rewrite plan is not an object")
    _require_exact_keys(value, _RAW_PLAN_FIELDS, label="raw rewrite plan")
    if any(not isinstance(value[field], list) for field in _RAW_PLAN_FIELDS):
        raise DeletionUpgradeStateError("Saved raw rewrite plan fields must be arrays")
    originals: list[dict[str, Any]] = []
    for row in value["original_manifest_rows"]:
        if not isinstance(row, dict):
            raise DeletionUpgradeStateError("Original source manifest row is invalid")
        originals.append(row)
    relocations: list[dict[str, Any]] = []
    for row in value["relocation_manifest_rows"]:
        if not isinstance(row, dict):
            raise DeletionUpgradeStateError("Relocation source manifest row is invalid")
        relocations.append(row)
    plan = RawRewritePlan(
        rewrites=tuple(_raw_rewrite_from_json(layout, rewrite) for rewrite in value["rewrites"]),
        original_manifest_rows=tuple(originals),
        relocation_manifest_rows=tuple(relocations),
        obsolete_blob_ids=tuple(
            _require_sha256(blob_id, label="obsolete source blob")
            for blob_id in value["obsolete_blob_ids"]
        ),
    )
    _raw_plan_to_json(layout, plan)
    return plan


def _state_document(
    layout: ArchiveLayout,
    *,
    operation_id: str,
    selection_sha256: str,
    purge_revision: str,
    raw_plan: RawRewritePlan,
    resources: SavedDeletionResources,
) -> tuple[dict[str, Any], str]:
    operation = _require_operation_id(operation_id)
    selection = _require_sha256(selection_sha256, label="selection")
    revision = _require_sha256(purge_revision, label="purge revision")
    plan_json = _raw_plan_to_json(layout, raw_plan)
    if set(raw_plan.selected_source_record_ids) != set(resources.source_record_ids):
        raise DeletionUpgradeStateError(
            "Saved source-record resources do not match the raw rewrite plan"
        )
    if not {rewrite.source_file_id for rewrite in raw_plan.rewrites} <= set(
        resources.source_file_ids
    ):
        raise DeletionUpgradeStateError("Raw rewrite references an unowned source file")
    body = {
        "schema_version": WORKSPACE_UPGRADE_STATE_VERSION,
        "kind": WORKSPACE_UPGRADE_STATE_KIND,
        "operation_id": operation,
        "selection_sha256": selection,
        "purge_revision": revision,
        "resources": resources.to_json(),
        "raw_rewrite_plan": plan_json,
    }
    state_sha256 = hashlib.sha256(_canonical_json(body)).hexdigest()
    return {**body, "state_sha256": state_sha256}, state_sha256


_STATE_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "operation_id",
        "selection_sha256",
        "purge_revision",
        "resources",
        "raw_rewrite_plan",
        "state_sha256",
    }
)


def _decode_state(layout: ArchiveLayout, encoded: bytes) -> SavedWorkspaceUpgradeState:
    try:
        value = orjson.loads(encoded)
    except orjson.JSONDecodeError as exc:
        raise DeletionUpgradeStateError("Workspace-upgrade state is invalid JSON") from exc
    if not isinstance(value, dict):
        raise DeletionUpgradeStateError("Workspace-upgrade state is not an object")
    _require_exact_keys(value, _STATE_FIELDS, label="workspace-upgrade state")
    if value["schema_version"] != WORKSPACE_UPGRADE_STATE_VERSION:
        raise DeletionUpgradeStateError("Unsupported workspace-upgrade state version")
    if value["kind"] != WORKSPACE_UPGRADE_STATE_KIND:
        raise DeletionUpgradeStateError("Workspace-upgrade state has the wrong kind")
    claimed_state_sha256 = _require_sha256(value["state_sha256"], label="state")
    body = {key: item for key, item in value.items() if key != "state_sha256"}
    actual_state_sha256 = hashlib.sha256(_canonical_json(body)).hexdigest()
    if claimed_state_sha256 != actual_state_sha256:
        raise DeletionUpgradeStateError("Workspace-upgrade state identity does not match")
    resources = SavedDeletionResources.from_json(value["resources"])
    raw_plan = _raw_plan_from_json(layout, value["raw_rewrite_plan"])
    canonical_document, rebuilt_state_sha256 = _state_document(
        layout,
        operation_id=_require_operation_id(value["operation_id"]),
        selection_sha256=_require_sha256(value["selection_sha256"], label="selection"),
        purge_revision=_require_sha256(value["purge_revision"], label="purge revision"),
        raw_plan=raw_plan,
        resources=resources,
    )
    if (
        rebuilt_state_sha256 != claimed_state_sha256
        or _canonical_json(canonical_document) != encoded
    ):
        raise DeletionUpgradeStateError("Workspace-upgrade state is not canonically encoded")
    return SavedWorkspaceUpgradeState(
        operation_id=value["operation_id"],
        selection_sha256=value["selection_sha256"],
        purge_revision=value["purge_revision"],
        state_sha256=claimed_state_sha256,
        resources=resources,
        raw_plan=raw_plan,
    )


def _state_relative_path(layout: ArchiveLayout, path: Path) -> str:
    return _archive_relative(path, layout.root, label="upgrade-state")


def _validate_private_file_stat(value: os.stat_result) -> None:
    if not stat.S_ISREG(value.st_mode) or stat.S_ISLNK(value.st_mode):
        raise DeletionUpgradeStateError("Workspace-upgrade state is not a regular file")
    if stat.S_IMODE(value.st_mode) != WORKSPACE_UPGRADE_FILE_MODE:
        raise DeletionUpgradeStateError("Workspace-upgrade state is not owner-only")
    if value.st_uid != os.geteuid():
        raise DeletionUpgradeStateError("Workspace-upgrade state has the wrong owner")


def _read_bound_file(path: Path, root: Path) -> tuple[bytes, os.stat_result, str]:
    try:
        handle: BinaryIO = open_regular_file_beneath(path, root)
    except Exception as exc:
        raise DeletionUpgradeStateError("Workspace-upgrade state cannot be opened safely") from exc
    with handle:
        before = os.fstat(handle.fileno())
        _validate_private_file_stat(before)
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        while chunk := handle.read(1_048_576):
            chunks.append(chunk)
            digest.update(chunk)
        after = os.fstat(handle.fileno())
    identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_mode", "st_uid")
    if any(getattr(before, field) != getattr(after, field) for field in identity_fields):
        raise DeletionUpgradeStateError("Workspace-upgrade state changed while it was read")
    encoded = b"".join(chunks)
    if len(encoded) != before.st_size:
        raise DeletionUpgradeStateError("Workspace-upgrade state size changed while it was read")
    return encoded, before, digest.hexdigest()


def _write_all(handle: BinaryIO, encoded: bytes) -> None:
    view = memoryview(encoded)
    offset = 0
    while offset < len(view):
        written = handle.write(view[offset:])
        if written is None or written <= 0:
            raise DeletionUpgradeStateError("Workspace-upgrade state write did not complete")
        offset += written


def _validate_receipt(
    layout: ArchiveLayout,
    path: Path,
    receipt: WorkspaceUpgradeStateReceipt,
    file_stat: os.stat_result,
    file_sha256: str,
) -> None:
    if receipt.schema_version != WORKSPACE_UPGRADE_STATE_VERSION:
        raise DeletionUpgradeStateError("Unsupported workspace-upgrade receipt version")
    if receipt.relative_path != _state_relative_path(layout, path):
        raise DeletionUpgradeStateError("Workspace-upgrade receipt path does not match")
    expected_identity = (
        receipt.device,
        receipt.inode,
        receipt.size_bytes,
        receipt.mtime_ns,
    )
    actual_identity = (
        int(file_stat.st_dev),
        int(file_stat.st_ino),
        int(file_stat.st_size),
        int(file_stat.st_mtime_ns),
    )
    if expected_identity != actual_identity:
        raise DeletionUpgradeStateError(
            "Workspace-upgrade file identity does not match its receipt"
        )
    if receipt.file_sha256 != file_sha256:
        raise DeletionUpgradeStateError("Workspace-upgrade file digest does not match its receipt")


def publish_workspace_upgrade_state(
    layout: ArchiveLayout,
    path: Path,
    *,
    operation_id: str,
    selection_sha256: str,
    purge_revision: str,
    raw_plan: RawRewritePlan,
    resources: SavedDeletionResources,
) -> WorkspaceUpgradeStateReceipt:
    """Atomically create one immutable, owner-only normalized-to-workspace handoff."""

    relative = _state_relative_path(layout, path)
    document, state_sha256 = _state_document(
        layout,
        operation_id=operation_id,
        selection_sha256=selection_sha256,
        purge_revision=purge_revision,
        raw_plan=raw_plan,
        resources=resources,
    )
    encoded = _canonical_json(document)
    file_sha256 = hashlib.sha256(encoded).hexdigest()
    try:
        parent_descriptor = open_directory_beneath(path.parent, layout.root)
    except Exception as exc:
        raise DeletionUpgradeStateError(
            "Workspace-upgrade destination directory is unsafe or missing"
        ) from exc
    temporary_name = f".workspace-upgrade-{secrets.token_hex(16)}.tmp"
    temporary_descriptor: int | None = None
    published = False
    try:
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        temporary_descriptor = os.open(
            temporary_name,
            flags,
            WORKSPACE_UPGRADE_FILE_MODE,
            dir_fd=parent_descriptor,
        )
        os.fchmod(temporary_descriptor, WORKSPACE_UPGRADE_FILE_MODE)
        with os.fdopen(temporary_descriptor, "wb", buffering=0) as handle:
            temporary_descriptor = None
            _write_all(handle, encoded)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_stat = os.fstat(handle.fileno())
        _validate_private_file_stat(temporary_stat)
        try:
            os.link(
                temporary_name,
                path.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            published = True
        except FileExistsError as exc:
            raise DeletionUpgradeStateExistsError(
                "Workspace-upgrade state already exists; refusing to overwrite it"
            ) from exc
        published_stat = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        if (published_stat.st_dev, published_stat.st_ino) != (
            temporary_stat.st_dev,
            temporary_stat.st_ino,
        ):
            raise DeletionUpgradeStateError("Workspace-upgrade publication changed identity")
        os.unlink(temporary_name, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
        _validate_private_file_stat(published_stat)
        return WorkspaceUpgradeStateReceipt(
            schema_version=WORKSPACE_UPGRADE_STATE_VERSION,
            relative_path=relative,
            operation_id=operation_id,
            selection_sha256=selection_sha256,
            purge_revision=purge_revision,
            state_sha256=state_sha256,
            file_sha256=file_sha256,
            size_bytes=int(published_stat.st_size),
            device=int(published_stat.st_dev),
            inode=int(published_stat.st_ino),
            mtime_ns=int(published_stat.st_mtime_ns),
        )
    except OSError as exc:
        phase = "after publication" if published else "before publication"
        raise DeletionUpgradeStateError(
            f"Workspace-upgrade state publication failed {phase}"
        ) from exc
    finally:
        if temporary_descriptor is not None:
            os.close(temporary_descriptor)
        with suppress(FileNotFoundError):
            os.unlink(temporary_name, dir_fd=parent_descriptor)
        os.close(parent_descriptor)


def load_workspace_upgrade_state(
    layout: ArchiveLayout,
    path: Path,
    *,
    receipt: WorkspaceUpgradeStateReceipt | Mapping[str, Any] | None = None,
    expected_operation_id: str | None = None,
    expected_selection_sha256: str | None = None,
    expected_purge_revision: str | None = None,
) -> SavedWorkspaceUpgradeState:
    """Load a saved plan only after file, receipt, and caller identities agree."""

    parsed_receipt = (
        WorkspaceUpgradeStateReceipt.from_json(receipt) if isinstance(receipt, Mapping) else receipt
    )
    encoded, file_stat, file_sha256 = _read_bound_file(path, layout.root)
    if parsed_receipt is not None:
        _validate_receipt(layout, path, parsed_receipt, file_stat, file_sha256)
    state = _decode_state(layout, encoded)
    expected_operation = (
        _require_operation_id(expected_operation_id) if expected_operation_id is not None else None
    )
    expected_selection = (
        _require_sha256(expected_selection_sha256, label="expected selection")
        if expected_selection_sha256 is not None
        else None
    )
    expected_revision = (
        _require_sha256(expected_purge_revision, label="expected purge revision")
        if expected_purge_revision is not None
        else None
    )
    if expected_operation is not None and state.operation_id != expected_operation:
        raise DeletionUpgradeStateError("Workspace-upgrade operation identity does not match")
    if expected_selection is not None and state.selection_sha256 != expected_selection:
        raise DeletionUpgradeStateError("Workspace-upgrade selection identity does not match")
    if expected_revision is not None and state.purge_revision != expected_revision:
        raise DeletionUpgradeStateError("Workspace-upgrade purge revision does not match")
    if parsed_receipt is not None and (
        parsed_receipt.operation_id != state.operation_id
        or parsed_receipt.selection_sha256 != state.selection_sha256
        or parsed_receipt.purge_revision != state.purge_revision
        or parsed_receipt.state_sha256 != state.state_sha256
    ):
        raise DeletionUpgradeStateError("Workspace-upgrade receipt identities do not match state")
    return state
