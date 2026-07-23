from __future__ import annotations

import json
import os
import re
import stat
import tempfile
import uuid
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from sqlalchemy import Engine

from meta_archive.paths import ArchiveLayout
from meta_archive.services.hashing import sha256_file
from meta_archive.services.source_import_lock import (
    SourceImportPublicationLease,
    assert_source_import_publication_allowed,
    source_import_publication_lock,
)
from meta_archive.services.verification import (
    mandatory_non_pass_checks,
    source_manifest_artifacts,
)

ProgressCallback = Callable[[str, int, int | None], None]
_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


class SourceRetirementError(RuntimeError):
    """The source promotion gate is not satisfied or the tree is unsafe to retire."""


@dataclass(frozen=True, slots=True)
class SourceRetirementResult:
    requested_chunks: tuple[str, ...]
    present_chunks: tuple[str, ...]
    already_retired_chunks: tuple[str, ...]
    manifest_file_count: int
    present_file_count: int
    total_bytes: int
    unique_blob_count: int
    applied: bool
    receipt_path: str | None = None


@dataclass(frozen=True, slots=True)
class _FileRecord:
    source: str
    chunk: str
    relative_path: str
    sha256: str
    size_bytes: int
    mode: int
    mtime_ns: int
    blob: str


def _safe_relative(value: object, *, field: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise SourceRetirementError(f"Manifest field {field!r} must be a string")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or ".." in pure.parts
        or "\x00" in value
        or (not allow_empty and value in {"", "."})
    ):
        raise SourceRetirementError(f"Unsafe {field} path in manifest: {value!r}")
    return "" if value in {"", "."} else pure.as_posix()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SourceRetirementError(f"Required {label} is missing or unsafe: {path}")
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceRetirementError(f"Required {label} is invalid: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SourceRetirementError(f"Required {label} must contain a JSON object: {path}")
    return payload


def _load_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise SourceRetirementError(f"Required {label} is missing or unsafe: {path}")
    records: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                value: Any = json.loads(line)
                if not isinstance(value, dict):
                    raise SourceRetirementError(f"{label} line {line_number} is not a JSON object")
                records.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceRetirementError(f"Required {label} is invalid: {path}: {exc}") from exc
    return records


def _file_artifact(layout: ArchiveLayout, path: Path) -> dict[str, object] | None:
    try:
        path_stat = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(path_stat.st_mode):
        return None
    return {
        "path": path.relative_to(layout.root).as_posix(),
        "sha256": sha256_file(path),
        "line_count": None,
        "size_bytes": path_stat.st_size,
    }


def _validate_gate(layout: ArchiveLayout, report_path: Path) -> dict[str, Any]:
    report = _load_json(report_path, label="verification report")
    checks_value = report.get("checks")
    if not isinstance(checks_value, list):
        raise SourceRetirementError("Verification report does not contain a complete check list")
    checks = [dict(check) for check in checks_value if isinstance(check, dict)]
    if not checks or len(checks) != len(checks_value):
        raise SourceRetirementError("Verification report does not contain a complete check list")
    non_pass = mandatory_non_pass_checks(checks)
    gate = report.get("gate")
    if (
        non_pass
        or not isinstance(gate, dict)
        or gate.get("all_mandatory_checks_pass") is not True
        or gate.get("deep_verified") is not True
        or gate.get("library_verified") is not True
        or gate.get("required_artifacts_present") is not True
        or gate.get("source_retirement_eligible") is not True
    ):
        keys = ", ".join(str(check.get("key", "unknown")) for check in non_pass)
        raise SourceRetirementError(
            "Source retirement requires an explicit all-pass deep verification gate"
            + (f"; mandatory non-pass checks: {keys}" if keys else "")
        )

    artifacts = report.get("artifacts")
    if not isinstance(artifacts, dict):
        raise SourceRetirementError("Verification report has no artifact bindings")
    current_manifests = source_manifest_artifacts(layout)
    if any(value is None for value in current_manifests.values()):
        raise SourceRetirementError("One or more complete source manifests are missing")
    if artifacts.get("source_manifests") != current_manifests:
        raise SourceRetirementError("Source manifests changed after the verification report")

    current_marker = _file_artifact(layout, layout.library_completion_marker)
    if current_marker is None or artifacts.get("library_marker") != current_marker:
        raise SourceRetirementError("Library completion marker changed after verification")
    marker = _load_json(layout.library_completion_marker, label="library completion marker")
    if marker.get("format_version") != 1 or marker.get("source_manifests") != current_manifests:
        raise SourceRetirementError(
            "Library completion marker is not bound to the current complete source manifests"
        )
    return report


def _parse_complete_manifests(
    layout: ArchiveLayout,
) -> tuple[dict[str, _FileRecord], dict[str, set[str]]]:
    original_rows = _load_jsonl(layout.original_files_manifest, label="original-files manifest")
    relocation_rows = _load_jsonl(layout.relocation_manifest, label="relocation manifest")
    directory_rows = _load_jsonl(
        layout.original_directories_manifest, label="original-directories manifest"
    )

    originals: dict[str, dict[str, Any]] = {}
    for row in original_rows:
        chunk = _safe_relative(row.get("chunk"), field="chunk")
        if "/" in chunk:
            raise SourceRetirementError(f"Chunk name is not an exact root entry: {chunk!r}")
        relative = _safe_relative(row.get("relative_path"), field="relative_path")
        source = f"{chunk}/{relative}"
        sha256 = row.get("sha256")
        size_bytes = row.get("size_bytes")
        mode = row.get("mode")
        mtime_ns = row.get("mtime_ns")
        if (
            not isinstance(sha256, str)
            or _DIGEST_PATTERN.fullmatch(sha256) is None
            or not isinstance(size_bytes, int)
            or size_bytes < 0
            or not isinstance(mode, int)
            or not isinstance(mtime_ns, int)
        ):
            raise SourceRetirementError(f"Invalid file metadata in original manifest: {source}")
        if source in originals:
            raise SourceRetirementError(f"Duplicate source coordinate: {source}")
        originals[source] = {
            "chunk": chunk,
            "relative_path": relative,
            "sha256": sha256,
            "size_bytes": size_bytes,
            "mode": mode,
            "mtime_ns": mtime_ns,
        }

    relocations: dict[str, dict[str, Any]] = {}
    for row in relocation_rows:
        source = _safe_relative(row.get("source"), field="source")
        if source in relocations:
            raise SourceRetirementError(f"Duplicate relocation coordinate: {source}")
        relocations[source] = row
    if originals.keys() != relocations.keys():
        missing = len(originals.keys() - relocations.keys())
        extra = len(relocations.keys() - originals.keys())
        raise SourceRetirementError(
            f"Relocation manifest is not complete: missing={missing}, unexpected={extra}"
        )

    records: dict[str, _FileRecord] = {}
    for source, original in originals.items():
        relocation = relocations[source]
        sha256 = original["sha256"]
        expected_blob = layout.blob_path(sha256).relative_to(layout.root).as_posix()
        for key in ("sha256", "size_bytes", "mode", "mtime_ns"):
            if relocation.get(key) != original[key]:
                raise SourceRetirementError(
                    f"Relocation metadata differs from original manifest for {source}: {key}"
                )
        if relocation.get("blob") != expected_blob:
            raise SourceRetirementError(f"Unsafe or incorrect blob mapping for {source}")
        records[source] = _FileRecord(
            source=source,
            chunk=str(original["chunk"]),
            relative_path=str(original["relative_path"]),
            sha256=str(sha256),
            size_bytes=int(original["size_bytes"]),
            mode=int(original["mode"]),
            mtime_ns=int(original["mtime_ns"]),
            blob=expected_blob,
        )

    directories: dict[str, set[str]] = {}
    for row in directory_rows:
        chunk = _safe_relative(row.get("chunk"), field="chunk")
        if "/" in chunk:
            raise SourceRetirementError(f"Chunk name is not an exact root entry: {chunk!r}")
        relative = _safe_relative(row.get("relative_path"), field="relative_path", allow_empty=True)
        if not isinstance(row.get("mode"), int) or not isinstance(row.get("mtime_ns"), int):
            raise SourceRetirementError(f"Invalid directory metadata: {chunk}/{relative}")
        chunk_directories = directories.setdefault(chunk, set())
        if relative in chunk_directories:
            raise SourceRetirementError(f"Duplicate directory coordinate: {chunk}/{relative}")
        chunk_directories.add(relative)
    if any("" not in values for values in directories.values()):
        raise SourceRetirementError("Every source chunk requires a root directory manifest row")
    if not {record.chunk for record in records.values()}.issubset(directories):
        raise SourceRetirementError(
            "File manifest references a chunk absent from directory metadata"
        )
    return records, directories


def _validate_requested_chunks(requested: Iterable[str], known: set[str]) -> tuple[str, ...]:
    chunks = tuple(requested)
    if not chunks:
        raise SourceRetirementError("At least one exact source chunk must be requested")
    if len(set(chunks)) != len(chunks):
        raise SourceRetirementError("Source chunk targets must not contain duplicates")
    for chunk in chunks:
        if (
            not isinstance(chunk, str)
            or chunk in {"", ".", ".."}
            or "/" in chunk
            or "\\" in chunk
            or chunk not in known
        ):
            raise SourceRetirementError(f"Unknown or unsafe exact source chunk target: {chunk!r}")
    return chunks


def _actual_chunk_entries(chunk_root: Path) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = {""}
    stack = [chunk_root]
    while stack:
        current = stack.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                path = Path(entry.path)
                relative = path.relative_to(chunk_root).as_posix()
                if entry.is_symlink():
                    raise SourceRetirementError(f"Unmanifested symlink blocks retirement: {path}")
                if entry.is_dir(follow_symlinks=False):
                    directories.add(relative)
                    stack.append(path)
                elif entry.is_file(follow_symlinks=False):
                    files.add(relative)
                else:
                    raise SourceRetirementError(
                        f"Special filesystem entry blocks retirement: {path}"
                    )
    return files, directories


def _write_receipt(layout: ArchiveLayout, payload: dict[str, object]) -> Path:
    receipts = layout.manifests / "source-retirement"
    receipts.mkdir(parents=True, exist_ok=True)
    target = receipts / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex}.json"
    descriptor, temporary_name = tempfile.mkstemp(prefix=".retirement.", dir=receipts)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def retire_source_chunks_locked(
    layout: ArchiveLayout,
    engine: Engine,
    chunk_names: Iterable[str],
    *,
    lock: SourceImportPublicationLease,
    apply: bool = False,
    verification_report: Path | None = None,
    progress: ProgressCallback | None = None,
) -> SourceRetirementResult:
    """Retire sources while reusing a validated outer source/publication lease.

    The operation never removes blobs, manifests, the catalog, or the library. It validates every
    target and every unique backing blob before mutating anything. Interrupted applications are
    resumable because already-absent manifested entries are safe and unmanifested entries always
    block further removal.
    """

    assert_source_import_publication_allowed(
        layout,
        engine,
        lease=lock,
        publisher="Source retirement publication",
    )
    report_path = (
        verification_report
        if verification_report is not None
        else layout.manifests / "integrity-reports" / "latest.json"
    )
    report_path = report_path.expanduser().resolve()
    reports_root = (layout.manifests / "integrity-reports").resolve()
    if report_path.parent != reports_root:
        raise SourceRetirementError(
            "Verification report must be an exact file in manifests/integrity-reports"
        )
    report = _validate_gate(layout, report_path)
    records, directory_records = _parse_complete_manifests(layout)
    requested = _validate_requested_chunks(chunk_names, set(directory_records))

    selected = [record for record in records.values() if record.chunk in requested]
    selected.sort(key=lambda record: record.source)
    files_by_chunk: dict[str, set[str]] = {chunk: set() for chunk in requested}
    for record in selected:
        files_by_chunk[record.chunk].add(record.relative_path)

    present_chunks: list[str] = []
    already_retired_chunks: list[str] = []
    present_files: list[Path] = []
    source_stats_by_hash: dict[str, list[os.stat_result]] = {}
    for chunk in requested:
        chunk_root = layout.root / chunk
        if chunk_root.is_symlink():
            raise SourceRetirementError(f"Source chunk root is a symlink: {chunk_root}")
        if not chunk_root.exists():
            already_retired_chunks.append(chunk)
            continue
        if not chunk_root.is_dir():
            raise SourceRetirementError(f"Source chunk root is not a safe directory: {chunk_root}")
        present_chunks.append(chunk)
        actual_files, actual_directories = _actual_chunk_entries(chunk_root)
        unexpected_files = actual_files - files_by_chunk[chunk]
        unexpected_directories = actual_directories - directory_records[chunk]
        if unexpected_files or unexpected_directories:
            raise SourceRetirementError(
                f"Unmanifested entries block retirement of {chunk}: "
                f"files={sorted(unexpected_files)[:5]}, "
                f"directories={sorted(unexpected_directories)[:5]}"
            )
        for record in selected:
            if record.chunk != chunk:
                continue
            source = chunk_root / record.relative_path
            try:
                source_stat = source.lstat()
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(source_stat.st_mode):
                raise SourceRetirementError(f"Manifest source is not a regular file: {source}")
            if (
                source_stat.st_size != record.size_bytes
                or source_stat.st_mtime_ns != record.mtime_ns
            ):
                raise SourceRetirementError(
                    f"Source changed after inventory and must be rescanned: {source}"
                )
            # Size and mtime are only a fast coordinate check. Re-hash immediately before the
            # destructive gate so a same-size, timestamp-restored edit cannot be retired under a
            # stale manifest identity.
            if sha256_file(source) != record.sha256:
                raise SourceRetirementError(
                    f"Source digest changed after inventory and must be rescanned: {source}"
                )
            present_files.append(source)
            source_stats_by_hash.setdefault(record.sha256, []).append(source_stat)

    verified_blobs: set[str] = set()
    unique_selected = {record.sha256: record for record in selected}
    total_blobs = len(unique_selected)
    for index, (sha256, record) in enumerate(sorted(unique_selected.items()), start=1):
        blob = layout.blob_path(sha256)
        try:
            blob_stat = blob.lstat()
        except FileNotFoundError as exc:
            raise SourceRetirementError(f"Backing blob is missing: {blob}") from exc
        if (
            not stat.S_ISREG(blob_stat.st_mode)
            or stat.S_IMODE(blob_stat.st_mode) & 0o222
            or blob_stat.st_size != record.size_bytes
            or sha256_file(blob) != sha256
        ):
            raise SourceRetirementError(f"Backing blob is unsafe or corrupt: {blob}")
        for candidate_source_stat in source_stats_by_hash.get(sha256, []):
            if (
                candidate_source_stat.st_dev == blob_stat.st_dev
                and candidate_source_stat.st_ino == blob_stat.st_ino
            ):
                raise SourceRetirementError(
                    f"Backing blob still shares an inode with its source: {blob}"
                )
        verified_blobs.add(sha256)
        if progress and (index % 250 == 0 or index == total_blobs):
            progress("Verifying retirement backing blobs", index, total_blobs)

    base_result = SourceRetirementResult(
        requested_chunks=requested,
        present_chunks=tuple(present_chunks),
        already_retired_chunks=tuple(already_retired_chunks),
        manifest_file_count=len(selected),
        present_file_count=len(present_files),
        total_bytes=sum(record.size_bytes for record in selected),
        unique_blob_count=len(verified_blobs),
        applied=False,
    )
    if not apply:
        return base_result

    # chmod is explicit and restricted to exact manifested directories. File unlink permission is
    # controlled by the parent directory; immutable blob modes are never changed here.
    for chunk in present_chunks:
        chunk_root = layout.root / chunk
        for relative in sorted(
            directory_records[chunk], key=lambda value: (value.count("/"), value)
        ):
            directory = chunk_root / relative if relative else chunk_root
            if directory.exists():
                directory_stat = directory.lstat()
                if not stat.S_ISDIR(directory_stat.st_mode):
                    raise SourceRetirementError(f"Expected a manifested directory: {directory}")
                os.chmod(
                    directory,
                    stat.S_IMODE(directory_stat.st_mode) | stat.S_IWUSR | stat.S_IXUSR,
                )
    for index, source in enumerate(present_files, start=1):
        source.unlink()
        if progress and (index % 250 == 0 or index == len(present_files)):
            progress("Retiring source files", index, len(present_files))
    for chunk in present_chunks:
        chunk_root = layout.root / chunk
        for relative in sorted(
            directory_records[chunk], key=lambda value: (value.count("/"), value), reverse=True
        ):
            directory = chunk_root / relative if relative else chunk_root
            if directory.exists():
                directory.rmdir()

    receipt_payload: dict[str, object] = {
        "schema_version": 1,
        "completed_at": datetime.now(UTC).isoformat(),
        "verification_report": report_path.relative_to(layout.root).as_posix(),
        "verification_report_sha256": sha256_file(report_path),
        "verification_generated_at": report.get("generated_at"),
        "result": asdict(base_result) | {"applied": True},
    }
    receipt = _write_receipt(layout, receipt_payload)
    return SourceRetirementResult(
        requested_chunks=base_result.requested_chunks,
        present_chunks=base_result.present_chunks,
        already_retired_chunks=base_result.already_retired_chunks,
        manifest_file_count=base_result.manifest_file_count,
        present_file_count=base_result.present_file_count,
        total_bytes=base_result.total_bytes,
        unique_blob_count=base_result.unique_blob_count,
        applied=True,
        receipt_path=receipt.relative_to(layout.root).as_posix(),
    )


def retire_source_chunks(
    layout: ArchiveLayout,
    engine: Engine,
    chunk_names: Iterable[str],
    *,
    apply: bool = False,
    verification_report: Path | None = None,
    progress: ProgressCallback | None = None,
) -> SourceRetirementResult:
    """Dry-run or retire exact source chunks beneath one source/publication lock."""

    with source_import_publication_lock(layout) as lock:
        return retire_source_chunks_locked(
            layout,
            engine,
            chunk_names,
            lock=lock,
            apply=apply,
            verification_report=verification_report,
            progress=progress,
        )


# Naming the promotion action explicitly makes orchestration intent clear while preserving the
# lower-level service name for callers that operate on individual source chunks.
promote_and_retire_sources = retire_source_chunks
