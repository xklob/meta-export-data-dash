from __future__ import annotations

import json
import mimetypes
import os
import tempfile
import uuid
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import magic
from sqlalchemy import Engine, func, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from meta_archive import __version__
from meta_archive.models import (
    Blob,
    ImportRun,
    IntegrityAnomaly,
    Job,
    SourceChunk,
    SourceFile,
)
from meta_archive.paths import ArchiveLayout
from meta_archive.services.hashing import sha256_file
from meta_archive.services.identifiers import stable_id
from meta_archive.services.purge_rewrite import BINARY_TOMBSTONE_BYTES
from meta_archive.services.source_import_lock import (
    SourceImportPublicationLease,
    assert_source_import_publication_allowed,
    source_import_publication_lock,
)

ProgressCallback = Callable[[str, int, int | None], None]


@dataclass(slots=True)
class InventorySummary:
    chunk_count: int = 0
    directory_count: int = 0
    file_count: int = 0
    total_bytes: int = 0
    json_file_count: int = 0
    hashed_file_count: int = 0
    reused_hash_count: int = 0
    unhashed_file_count: int = 0
    unique_blob_count: int = 0
    unique_blob_bytes: int = 0
    duplicate_file_count: int = 0
    duplicate_bytes: int = 0
    normalized_collision_groups: int = 0
    conflicting_collision_groups: int = 0


def source_category(relative_path: str) -> str:
    parts = Path(relative_path).parts
    if not parts:
        return "unknown"
    if parts[0] == "your_facebook_activity" and len(parts) > 1:
        return parts[1]
    return parts[0]


def _iter_regular_files(root: Path) -> Iterator[Path]:
    stack = [root]
    while stack:
        current = stack.pop()
        directories: list[Path] = []
        files: list[Path] = []
        with os.scandir(current) as entries:
            for entry in entries:
                path = Path(entry.path)
                if entry.is_dir(follow_symlinks=False):
                    directories.append(path)
                elif entry.is_file(follow_symlinks=False):
                    files.append(path)
        yield from sorted(files)
        stack.extend(reversed(sorted(directories)))


def _directory_manifest(chunk: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    stack = [chunk]
    while stack:
        current = stack.pop()
        stat = current.stat(follow_symlinks=False)
        relative = current.relative_to(chunk).as_posix()
        records.append(
            {
                "chunk": chunk.name,
                "relative_path": "" if relative == "." else relative,
                "mtime_ns": stat.st_mtime_ns,
                "mode": stat.st_mode,
            }
        )
        with os.scandir(current) as entries:
            children = [
                Path(entry.path) for entry in entries if entry.is_dir(follow_symlinks=False)
            ]
        stack.extend(reversed(sorted(children)))
    return records


def _media_type(path: Path) -> str:
    guessed, _encoding = mimetypes.guess_type(path.name)
    if guessed:
        return guessed
    try:
        detected = magic.from_file(str(path), mime=True)
        return str(detected) if detected else "application/octet-stream"
    except (OSError, magic.MagicException):
        return "application/octet-stream"


def _json_status(path: Path) -> str:
    if path.stat(follow_symlinks=False).st_size == len(BINARY_TOMBSTONE_BYTES):
        try:
            if path.read_bytes() == BINARY_TOMBSTONE_BYTES:
                return "purged"
        except OSError:
            pass
    return "pending" if path.suffix.casefold() == ".json" else "not_json"


def _write_manifest_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _manifest_line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("rb") as handle:
        return sum(1 for _line in handle)


def _read_manifest_records(path: Path, *, label: str) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Required {label} is missing or unsafe: {path}")
    records: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                value: Any = json.loads(line)
                if not isinstance(value, dict):
                    raise RuntimeError(f"{label} line {line_number} is not a JSON object")
                records.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Required {label} is invalid: {path}: {exc}") from exc
    return records


def _retired_chunk_names(layout: ArchiveLayout, *, present_names: set[str]) -> set[str]:
    """Return receipt-backed chunk roots whose source directories remain retired."""

    receipt_root = layout.manifests / "source-retirement"
    if not receipt_root.exists():
        return set()
    if receipt_root.is_symlink() or not receipt_root.is_dir():
        raise RuntimeError(f"Source-retirement receipt path is unsafe: {receipt_root}")
    retired: set[str] = set()
    for receipt in sorted(receipt_root.glob("*.json")):
        if receipt.is_symlink() or not receipt.is_file():
            raise RuntimeError(f"Source-retirement receipt is unsafe: {receipt}")
        try:
            payload: Any = json.loads(receipt.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Invalid source-retirement receipt: {receipt}: {exc}") from exc
        result = payload.get("result") if isinstance(payload, dict) else None
        chunks = result.get("requested_chunks") if isinstance(result, dict) else None
        if (
            not isinstance(result, dict)
            or result.get("applied") is not True
            or not isinstance(chunks, list)
        ):
            raise RuntimeError(f"Incomplete source-retirement receipt: {receipt}")
        for chunk in chunks:
            if (
                not isinstance(chunk, str)
                or chunk in {"", ".", ".."}
                or "/" in chunk
                or "\\" in chunk
            ):
                raise RuntimeError(f"Unsafe chunk name in source-retirement receipt: {chunk!r}")
            retired.add(chunk)
    # A reconstructed or newly supplied exact chunk root is scanned normally. Its old receipt
    # remains useful provenance but must not mask live bytes.
    return retired - present_names


def _preserved_retired_manifest_records(
    layout: ArchiveLayout, retired_names: set[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not retired_names:
        return [], []
    file_records = [
        record
        for record in _read_manifest_records(
            layout.original_files_manifest, label="original-files manifest"
        )
        if record.get("chunk") in retired_names
    ]
    directory_records = [
        record
        for record in _read_manifest_records(
            layout.original_directories_manifest,
            label="original-directories manifest",
        )
        if record.get("chunk") in retired_names
    ]
    represented_files = {record.get("chunk") for record in file_records}
    represented_directories = {record.get("chunk") for record in directory_records}
    if not retired_names.issubset(represented_directories) or not represented_files.issubset(
        retired_names
    ):
        raise RuntimeError("Retired chunks are not completely represented by source manifests")
    return file_records, directory_records


def _existing_file_state(
    session: Session,
) -> dict[tuple[int, str], tuple[int, int, str | None]]:
    rows = session.execute(
        select(
            SourceFile.chunk_id,
            SourceFile.relative_path,
            SourceFile.size_bytes,
            SourceFile.mtime_ns,
            SourceFile.sha256,
        )
    )
    return {
        (chunk_id, relative_path): (size_bytes, mtime_ns, sha256)
        for chunk_id, relative_path, size_bytes, mtime_ns, sha256 in rows
    }


def _record_scan_anomalies(session: Session) -> tuple[int, int]:
    collision_rows = session.execute(
        select(
            SourceFile.normalized_path,
            func.count(SourceFile.id),
            func.count(SourceFile.sha256),
            func.count(func.distinct(SourceFile.sha256)),
        )
        .where(SourceFile.scan_status == "discovered")
        .group_by(SourceFile.normalized_path)
        .having(func.count(SourceFile.id) > 1)
    ).all()
    conflicts = 0
    for normalized_path, physical_count, hashed_count, distinct_hashes in collision_rows:
        fully_hashed = hashed_count == physical_count
        is_conflict = fully_hashed and distinct_hashes > 1
        is_verified_duplicate = fully_hashed and distinct_hashes == 1
        if is_conflict:
            conflicts += 1
        anomaly_id = stable_id("normalized-path-collision", normalized_path)
        code = (
            "conflicting_normalized_path"
            if is_conflict
            else "duplicate_normalized_path"
            if is_verified_duplicate
            else "unverified_normalized_path"
        )
        severity = "critical" if is_conflict else "info" if is_verified_duplicate else "warning"
        summary = (
            "Copies at this normalized path differ and must remain quarantined."
            if is_conflict
            else "Copies at this normalized path are byte-identical."
            if is_verified_duplicate
            else "Copies at this normalized path require full hash verification."
        )
        session.execute(
            sqlite_insert(IntegrityAnomaly)
            .values(
                id=anomaly_id,
                code=code,
                severity=severity,
                summary=summary,
                detail_json={
                    "normalized_path": normalized_path,
                    "physical_count": physical_count,
                    "hashed_count": hashed_count,
                    "distinct_hashes": distinct_hashes,
                },
                resolved=is_verified_duplicate,
            )
            .on_conflict_do_update(
                index_elements=[IntegrityAnomaly.id],
                set_={
                    "code": code,
                    "severity": severity,
                    "summary": summary,
                    "detail_json": {
                        "normalized_path": normalized_path,
                        "physical_count": physical_count,
                        "hashed_count": hashed_count,
                        "distinct_hashes": distinct_hashes,
                    },
                    # Byte-identical copies are a verified deduplication fact, not an open issue.
                    "resolved": is_verified_duplicate,
                },
            )
        )
    return len(collision_rows), conflicts


def _summarize_inventory(session: Session, summary: InventorySummary) -> InventorySummary:
    active = SourceFile.scan_status == "discovered"
    summary.file_count, summary.total_bytes, summary.json_file_count = session.execute(
        select(
            func.count(SourceFile.id),
            func.coalesce(func.sum(SourceFile.size_bytes), 0),
            func.coalesce(
                func.sum(func.iif(SourceFile.json_status != "not_json", 1, 0)),
                0,
            ),
        ).where(active)
    ).one()
    active_blobs = (
        select(
            SourceFile.sha256.label("sha256"),
            func.min(SourceFile.size_bytes).label("size_bytes"),
        )
        .where(active)
        .where(SourceFile.sha256.is_not(None))
        .group_by(SourceFile.sha256)
        .subquery()
    )
    summary.unique_blob_count, summary.unique_blob_bytes = session.execute(
        select(
            func.count(active_blobs.c.sha256),
            func.coalesce(func.sum(active_blobs.c.size_bytes), 0),
        )
    ).one()
    summary.unhashed_file_count = (
        session.scalar(
            select(func.count(SourceFile.id)).where(active).where(SourceFile.sha256.is_(None))
        )
        or 0
    )
    duplicate_stats = session.execute(
        select(
            func.coalesce(func.sum(SourceFile.size_bytes), 0),
            func.count(SourceFile.id),
        )
        .where(active)
        .where(SourceFile.sha256.is_not(None))
        .where(
            SourceFile.id.not_in(
                select(func.min(SourceFile.id))
                .where(active)
                .where(SourceFile.sha256.is_not(None))
                .group_by(SourceFile.sha256)
            )
        )
    ).one()
    summary.duplicate_bytes = int(duplicate_stats[0] or 0)
    summary.duplicate_file_count = int(duplicate_stats[1] or 0)
    return summary


def scan_archive_locked(
    layout: ArchiveLayout,
    engine: Engine,
    *,
    lock: SourceImportPublicationLease,
    hash_files: bool,
    progress: ProgressCallback | None = None,
) -> InventorySummary:
    """Inventory sources while reusing a validated outer source/publication lease.

    The operation is idempotent. Existing hashes are reused only when both size and nanosecond
    mtime are unchanged; otherwise the stale hash is removed or recomputed.
    """

    assert_source_import_publication_allowed(
        layout,
        engine,
        lease=lock,
        publisher="Source inventory publication",
    )
    layout.ensure_runtime_directories()
    chunks = layout.chunk_directories()
    present_names = {chunk.name for chunk in chunks}
    retired_names = _retired_chunk_names(layout, present_names=present_names)
    preserved_files, preserved_directories = _preserved_retired_manifest_records(
        layout, retired_names
    )
    run_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())
    now = datetime.now(UTC)
    summary = InventorySummary(chunk_count=len(present_names | retired_names))
    if hash_files:
        preserved_hashes = [record.get("sha256") for record in preserved_files]
        if any(not isinstance(digest, str) or len(digest) != 64 for digest in preserved_hashes):
            raise RuntimeError("A retired source file lacks its required preserved SHA-256")
        summary.reused_hash_count = len(preserved_hashes)

    with Session(engine) as session, session.begin():
        session.add(
            ImportRun(
                id=run_id,
                kind="scan",
                status="running",
                parser_version=__version__,
                counters_json={},
                checkpoint_json={},
            )
        )
        session.add(
            Job(
                id=job_id,
                kind="scan",
                status="running",
                stage="Discovering source files",
                started_at=now,
                counters_json={},
                checkpoint_json={},
            )
        )
        retired_chunk_ids = list(
            session.scalars(select(SourceChunk.id).where(SourceChunk.chunk_key.in_(retired_names)))
        )
        if len(retired_chunk_ids) != len(retired_names):
            raise RuntimeError("A receipt-backed retired chunk is absent from the catalog")
        mark_not_seen = update(SourceFile)
        if retired_chunk_ids:
            mark_not_seen = mark_not_seen.where(SourceFile.chunk_id.not_in(retired_chunk_ids))
        session.execute(mark_not_seen.values(scan_status="not_seen"))
        if retired_chunk_ids:
            session.execute(
                update(SourceFile)
                .where(SourceFile.chunk_id.in_(retired_chunk_ids))
                .values(scan_status="discovered")
            )
        for ordinal, chunk in enumerate(chunks):
            session.execute(
                sqlite_insert(SourceChunk)
                .values(
                    chunk_key=chunk.name,
                    original_path=chunk.name,
                    ordinal=ordinal,
                )
                .on_conflict_do_update(
                    index_elements=[SourceChunk.chunk_key],
                    set_={"original_path": chunk.name, "ordinal": ordinal},
                )
            )

    manifest_records: list[dict[str, Any]] = list(preserved_files)
    directory_manifest_records: list[dict[str, Any]] = list(preserved_directories)
    scanned = 0
    with Session(engine) as session:
        chunk_ids: dict[str, int] = {
            chunk_key: chunk_id
            for chunk_key, chunk_id in session.execute(
                select(SourceChunk.chunk_key, SourceChunk.id)
            )
        }
        existing = _existing_file_state(session)

    try:
        for chunk in chunks:
            chunk_id = chunk_ids[chunk.name]
            directory_manifest_records.extend(_directory_manifest(chunk))
            with Session(engine) as session, session.begin():
                for source_path in _iter_regular_files(chunk):
                    relative_path = source_path.relative_to(chunk).as_posix()
                    stat = source_path.stat(follow_symlinks=False)
                    previous = existing.get((chunk_id, relative_path))
                    unchanged = previous is not None and previous[:2] == (
                        stat.st_size,
                        stat.st_mtime_ns,
                    )
                    previous_hash = previous[2] if previous else None
                    sha256: str | None
                    if hash_files and unchanged and previous_hash:
                        sha256 = previous_hash
                        summary.reused_hash_count += 1
                    elif hash_files:
                        sha256 = sha256_file(source_path)
                        summary.hashed_file_count += 1
                    elif unchanged:
                        sha256 = previous_hash
                    else:
                        sha256 = None

                    json_status = _json_status(source_path)
                    media_type = (
                        "application/x-meta-archive-purged"
                        if json_status == "purged"
                        else _media_type(source_path)
                    )
                    extension = source_path.suffix.casefold()[:32] or None
                    if sha256 is not None:
                        session.execute(
                            sqlite_insert(Blob)
                            .values(
                                sha256=sha256,
                                size_bytes=stat.st_size,
                                media_type=media_type,
                                extension=extension,
                                state="catalogued",
                            )
                            .on_conflict_do_nothing(index_elements=[Blob.sha256])
                        )

                    values: dict[str, Any] = {
                        "chunk_id": chunk_id,
                        "relative_path": relative_path,
                        "normalized_path": relative_path,
                        "category": source_category(relative_path),
                        "size_bytes": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                        "mode": stat.st_mode,
                        "media_type": media_type,
                        "extension": extension,
                        "sha256": sha256,
                        "json_status": json_status,
                        "json_error": None,
                        "scan_status": "discovered",
                    }
                    update_values = dict(values)
                    if unchanged:
                        update_values.pop("json_status")
                        update_values.pop("json_error")
                    else:
                        update_values["imported_parser_version"] = None
                        update_values["imported_sha256"] = None
                    session.execute(
                        sqlite_insert(SourceFile)
                        .values(**values)
                        .on_conflict_do_update(
                            index_elements=[SourceFile.chunk_id, SourceFile.relative_path],
                            set_=update_values,
                        )
                    )
                    manifest_records.append(
                        {
                            "chunk": chunk.name,
                            "relative_path": relative_path,
                            "size_bytes": stat.st_size,
                            "mtime_ns": stat.st_mtime_ns,
                            "mode": stat.st_mode,
                            "media_type": media_type,
                            "sha256": sha256,
                        }
                    )
                    scanned += 1
                    if progress and scanned % 250 == 0:
                        progress(
                            "Scanning and hashing" if hash_files else "Scanning",
                            scanned,
                            None,
                        )
                session.execute(
                    update(Job)
                    .where(Job.id == job_id)
                    .values(
                        progress_current=scanned,
                        stage=f"Completed {chunk.name}",
                        checkpoint_json={"last_chunk": chunk.name},
                    )
                )

        summary.directory_count = len(directory_manifest_records)
        with Session(engine) as session, session.begin():
            summary.normalized_collision_groups, summary.conflicting_collision_groups = (
                _record_scan_anomalies(session)
            )
            _summarize_inventory(session, summary)
            finished = datetime.now(UTC)
            session.execute(
                update(ImportRun)
                .where(ImportRun.id == run_id)
                .values(
                    status="completed",
                    finished_at=finished,
                    counters_json=asdict(summary),
                    checkpoint_json={"last_chunk": chunks[-1].name if chunks else None},
                )
            )
            session.execute(
                update(Job)
                .where(Job.id == job_id)
                .values(
                    status="completed",
                    finished_at=finished,
                    progress_current=summary.file_count,
                    progress_total=summary.file_count,
                    stage="Inventory complete",
                    counters_json=asdict(summary),
                )
            )
        manifest_records.sort(
            key=lambda record: (str(record.get("chunk")), str(record.get("relative_path")))
        )
        directory_manifest_records.sort(
            key=lambda record: (str(record.get("chunk")), str(record.get("relative_path")))
        )
        _write_manifest_atomic(layout.original_files_manifest, manifest_records)
        _write_manifest_atomic(
            layout.original_directories_manifest,
            directory_manifest_records,
        )
        if progress:
            progress("Inventory complete", summary.file_count, summary.file_count)
        return summary
    except BaseException as exc:
        with Session(engine) as session, session.begin():
            finished = datetime.now(UTC)
            session.execute(
                update(ImportRun)
                .where(ImportRun.id == run_id)
                .values(status="failed", finished_at=finished, error_text=str(exc))
            )
            session.execute(
                update(Job)
                .where(Job.id == job_id)
                .values(status="failed", finished_at=finished, error_text=str(exc))
            )
        raise


def scan_archive(
    layout: ArchiveLayout,
    engine: Engine,
    *,
    hash_files: bool,
    progress: ProgressCallback | None = None,
) -> InventorySummary:
    """Inventory every source file beneath one source/publication lock."""

    with source_import_publication_lock(layout) as lock:
        return scan_archive_locked(
            layout,
            engine,
            lock=lock,
            hash_files=hash_files,
            progress=progress,
        )


def inventory_summary(engine: Engine, layout: ArchiveLayout | None = None) -> InventorySummary:
    with Session(engine) as session:
        summary = InventorySummary(
            chunk_count=session.scalar(select(func.count(SourceChunk.id))) or 0,
            directory_count=(
                _manifest_line_count(layout.original_directories_manifest) if layout else 0
            ),
        )
        _summarize_inventory(session, summary)
        summary.normalized_collision_groups = (
            session.scalar(
                select(func.count()).select_from(
                    select(SourceFile.normalized_path)
                    .where(SourceFile.scan_status == "discovered")
                    .group_by(SourceFile.normalized_path)
                    .having(func.count(SourceFile.id) > 1)
                    .subquery()
                )
            )
            or 0
        )
        summary.conflicting_collision_groups = (
            session.scalar(
                select(func.count()).select_from(
                    select(SourceFile.normalized_path)
                    .where(SourceFile.scan_status == "discovered")
                    .group_by(SourceFile.normalized_path)
                    .having(func.count(func.distinct(SourceFile.sha256)) > 1)
                    .subquery()
                )
            )
            or 0
        )
        return summary
