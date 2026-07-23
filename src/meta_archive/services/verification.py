from __future__ import annotations

import json
import stat
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from sqlalchemy import Engine, func, select, text
from sqlalchemy.orm import Session

from meta_archive.baseline import (
    AUDITED_BASELINE_MANIFEST_RELATIVE_PATH,
    AuditedBaselineManifestError,
    load_audited_baseline,
)
from meta_archive.models import (
    ActivityRecord,
    AttachmentOccurrence,
    Blob,
    Conversation,
    ConversationSegment,
    ImportRun,
    IntegrityAnomaly,
    Message,
    MessageReaction,
    SegmentParticipant,
    SourceChunk,
    SourceFile,
    SourceRecord,
    Star,
)
from meta_archive.paths import ArchiveLayout
from meta_archive.services.hashing import sha256_file
from meta_archive.services.stars import StarTargetKey, resolve_star_targets

ProgressCallback = Callable[[str, int, int | None], None]
SOURCE_RETIREMENT_REQUIRED_CHECK_KEYS = frozenset(
    {
        "sqlite.integrity",
        "sqlite.foreign_keys",
        "inventory.manifest_rows",
        "inventory.relocation_rows",
        "inventory.hashes",
        "blobs.materialized_files",
        "blobs.immutable_independent",
        "json.parse_status",
        "messages.imported",
        "library.completion_marker",
        "stars.targets",
        "idempotency.messages",
        "idempotency.activity",
        "deep.source_bytes",
    }
)


@dataclass(slots=True)
class VerificationCheck:
    key: str
    status: str
    expected: int | str | None
    actual: int | str | None
    detail: str
    mandatory: bool = True


def mandatory_non_pass_checks(
    checks: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Return gate checks that are pending, failed, or structurally invalid.

    CLI presentation may distinguish ``pending`` from ``fail`` while an import is in progress,
    but neither state is sufficient for source promotion. Unknown/missing status values fail
    closed. Checks are mandatory unless a producer explicitly marks them otherwise.
    """

    result = [
        check
        for check in checks
        if check.get("mandatory", True) is not False or check.get("status") == "fail"
        if check.get("status") != "pass"
    ]
    present_keys = {check.get("key") for check in checks}
    result.extend(
        {
            "key": key,
            "status": "missing",
            "mandatory": True,
            "detail": "Required source-retirement verification check is absent",
        }
        for key in sorted(SOURCE_RETIREMENT_REQUIRED_CHECK_KEYS - present_keys)
    )
    return result


def _manifest_line_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        return sum(1 for _line in handle)


def _artifact_record(layout: ArchiveLayout, path: Path, *, jsonl: bool) -> dict[str, object] | None:
    try:
        path_stat = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(path_stat.st_mode):
        return None
    return {
        "path": path.relative_to(layout.root).as_posix(),
        "sha256": sha256_file(path),
        "line_count": _manifest_line_count(path) if jsonl else None,
        "size_bytes": path_stat.st_size,
    }


def source_manifest_artifacts(layout: ArchiveLayout) -> dict[str, dict[str, object] | None]:
    return {
        "original_files": _artifact_record(layout, layout.original_files_manifest, jsonl=True),
        "original_directories": _artifact_record(
            layout, layout.original_directories_manifest, jsonl=True
        ),
        "relocation": _artifact_record(layout, layout.relocation_manifest, jsonl=True),
    }


def _deep_manifest_check(
    layout: ArchiveLayout,
    *,
    progress: ProgressCallback | None,
) -> VerificationCheck:
    if not layout.original_files_manifest.is_file():
        return VerificationCheck(
            "deep.source_bytes",
            "fail",
            "readable original-files manifest",
            "missing",
            "Deep verification cannot run without manifests/original-files.jsonl",
        )
    checked = 0
    missing = 0
    mismatched = 0
    blob_hash_cache: dict[str, bool] = {}
    total = _manifest_line_count(layout.original_files_manifest) or 0
    try:
        handle = layout.original_files_manifest.open(encoding="utf-8")
    except OSError as exc:
        return VerificationCheck(
            "deep.source_bytes",
            "fail",
            "readable original-files manifest",
            "unreadable",
            f"Deep verification could not open the source manifest: {exc}",
        )
    with handle:
        for line in handle:
            try:
                record: Any = json.loads(line)
            except json.JSONDecodeError:
                mismatched += 1
                checked += 1
                continue
            if not isinstance(record, dict):
                mismatched += 1
                checked += 1
                continue
            chunk = record.get("chunk")
            relative_path = record.get("relative_path")
            expected_hash = record.get("sha256")
            expected_size = record.get("size_bytes")
            if (
                not isinstance(chunk, str)
                or not isinstance(relative_path, str)
                or not isinstance(expected_hash, str)
                or not isinstance(expected_size, int)
            ):
                mismatched += 1
                continue
            relative = Path(relative_path)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or "/" in chunk
                or chunk in {"", ".", ".."}
            ):
                mismatched += 1
                continue
            source = layout.root / chunk / relative
            try:
                source_stat = source.lstat()
            except FileNotFoundError:
                source_stat = None
            if source_stat is not None and stat.S_ISREG(source_stat.st_mode):
                if source_stat.st_size != expected_size or sha256_file(source) != expected_hash:
                    mismatched += 1
            else:
                blob = layout.blob_path(expected_hash)
                try:
                    blob_stat = blob.lstat()
                except FileNotFoundError:
                    blob_stat = None
                if blob_stat is None or not stat.S_ISREG(blob_stat.st_mode):
                    missing += 1
                elif expected_hash not in blob_hash_cache:
                    blob_hash_cache[expected_hash] = (
                        blob_stat.st_size == expected_size and sha256_file(blob) == expected_hash
                    )
                if expected_hash in blob_hash_cache and not blob_hash_cache[expected_hash]:
                    mismatched += 1
            checked += 1
            if progress and (checked % 250 == 0 or checked == total):
                progress("Deep source/blob verification", checked, total)
    return VerificationCheck(
        "deep.source_bytes",
        "pass" if checked == total and missing == 0 and mismatched == 0 else "fail",
        total,
        checked,
        f"missing={missing}, hash_or_manifest_mismatches={mismatched}",
    )


def _latest_idempotency_check(session: Session, *, kind: str) -> VerificationCheck:
    completed_statuses = ("completed", "completed_with_errors")
    runs = list(
        session.scalars(
            select(ImportRun)
            .where(ImportRun.kind == kind)
            .where(ImportRun.status.in_(completed_statuses))
            .order_by(ImportRun.started_at.desc(), ImportRun.id.desc())
            .limit(2)
        )
    )
    label = "message" if kind == "messages" else "activity"
    if len(runs) < 2:
        return VerificationCheck(
            f"idempotency.{kind}",
            "pending",
            "two completed imports; latest has zero growth",
            len(runs),
            f"Run the {label} importer again after a successful first import",
        )
    counters = runs[0].counters_json
    if kind == "messages":
        candidate_count = counters.get("candidate_file_count")
        skipped_count = counters.get("skipped_file_count")
        imported_count = counters.get("imported_file_count")
        zero_growth = (
            isinstance(candidate_count, int)
            and imported_count == 0
            and skipped_count == candidate_count
        )
        actual = f"imported={imported_count}, skipped={skipped_count}, candidates={candidate_count}"
    else:
        considered_count = counters.get("files_considered")
        skipped_count = counters.get("files_skipped")
        imported_count = counters.get("files_imported")
        record_count = counters.get("records_imported")
        zero_growth = (
            isinstance(considered_count, int)
            and imported_count == 0
            and record_count == 0
            and skipped_count == considered_count
        )
        actual = (
            f"files_imported={imported_count}, records_imported={record_count}, "
            f"skipped={skipped_count}, considered={considered_count}"
        )
    return VerificationCheck(
        f"idempotency.{kind}",
        "pass" if zero_growth else "fail",
        "zero growth on latest completed rerun",
        actual,
        f"Latest of {len(runs)} inspected completed {label} import runs",
    )


def _library_marker_check(
    layout: ArchiveLayout,
    *,
    conversation_count: int,
    message_count: int,
    attachment_count: int,
    activity_count: int,
    unlinked_count: int,
) -> VerificationCheck:
    marker = layout.library_completion_marker
    if not marker.is_file():
        return VerificationCheck(
            "library.completion_marker",
            "pending",
            "valid library/manifest.json matching catalog counts",
            "missing",
            "Build and atomically promote the complete filesystem library",
        )
    try:
        payload: Any = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return VerificationCheck(
            "library.completion_marker",
            "fail",
            "valid library/manifest.json matching catalog counts",
            "invalid",
            f"Library completion marker could not be read: {exc}",
        )
    counts = payload.get("counts") if isinstance(payload, dict) else None
    source_manifests = payload.get("source_manifests") if isinstance(payload, dict) else None
    expected_source_manifests = source_manifest_artifacts(layout)
    expected = {
        "conversation_count": conversation_count,
        "message_count": message_count,
        "attachment_occurrence_count": attachment_count,
        "activity_record_count": activity_count,
        "unlinked_media_count": unlinked_count,
    }
    actual = {key: counts.get(key) for key in expected} if isinstance(counts, dict) else {}
    valid = (
        isinstance(payload, dict)
        and payload.get("format_version") == 1
        and actual == expected
        and all(value is not None for value in expected_source_manifests.values())
        and source_manifests == expected_source_manifests
    )
    return VerificationCheck(
        "library.completion_marker",
        "pass" if valid else "fail",
        json.dumps(expected, sort_keys=True),
        json.dumps(actual, sort_keys=True),
        (
            "The marker is published only with a complete library tree, exact catalog counts, "
            "and the current source-manifest digests"
        ),
    )


def verify_catalog(
    layout: ArchiveLayout,
    engine: Engine,
    *,
    deep: bool = False,
    require_audited_baseline: bool = False,
    progress: ProgressCallback | None = None,
) -> list[dict[str, object]]:
    checks: list[VerificationCheck] = []
    with engine.connect() as connection:
        integrity = connection.execute(text("PRAGMA integrity_check")).scalar_one()
        checks.append(
            VerificationCheck(
                "sqlite.integrity",
                "pass" if integrity == "ok" else "fail",
                "ok",
                str(integrity),
                "SQLite structural integrity",
            )
        )
        foreign_key_errors = len(connection.execute(text("PRAGMA foreign_key_check")).all())
        checks.append(
            VerificationCheck(
                "sqlite.foreign_keys",
                "pass" if foreign_key_errors == 0 else "fail",
                0,
                foreign_key_errors,
                "Foreign-key violations",
            )
        )

    permission_findings = layout.private_permission_findings()
    finding_preview = "; ".join(permission_findings[:5])
    remaining_findings = len(permission_findings) - min(5, len(permission_findings))
    if remaining_findings:
        finding_preview = f"{finding_preview}; plus {remaining_findings} more"
    checks.append(
        VerificationCheck(
            "runtime.private_permissions",
            "pass" if not permission_findings else "pending",
            "owner-only runtime boundary",
            "owner-only" if not permission_findings else f"{len(permission_findings)} findings",
            finding_preview
            or "Archive runtime directories and mutable files are not accessible to other users",
            mandatory=False,
        )
    )

    baseline = None
    try:
        baseline = load_audited_baseline(layout.root)
    except AuditedBaselineManifestError as exc:
        checks.append(
            VerificationCheck(
                "baseline.manifest",
                "fail",
                "valid owner-only schema-v1 archive-local manifest",
                "invalid",
                str(exc),
            )
        )
    else:
        if baseline is not None:
            checks.append(
                VerificationCheck(
                    "baseline.manifest",
                    "pass",
                    "valid owner-only schema-v1 archive-local manifest",
                    "valid",
                    "Private audited counts are schema-validated and archive-identity-bound",
                )
            )
        elif require_audited_baseline:
            checks.append(
                VerificationCheck(
                    "baseline.manifest",
                    "fail",
                    AUDITED_BASELINE_MANIFEST_RELATIVE_PATH.as_posix(),
                    "missing",
                    "Real-export acceptance requires an archive-local audited baseline",
                )
            )

    with Session(engine) as session:
        source_files = (
            session.scalar(
                select(func.count(SourceFile.id)).where(SourceFile.scan_status == "discovered")
            )
            or 0
        )
        manifest_lines = _manifest_line_count(layout.original_files_manifest)
        checks.append(
            VerificationCheck(
                "inventory.manifest_rows",
                "pass" if manifest_lines == source_files else "fail",
                source_files,
                manifest_lines,
                "Every discovered physical file must have one original-manifest row",
            )
        )
        relocation_lines = _manifest_line_count(layout.relocation_manifest)
        checks.append(
            VerificationCheck(
                "inventory.relocation_rows",
                "pass" if relocation_lines == source_files else "pending",
                source_files,
                relocation_lines,
                "Every original file maps to one content-addressed blob",
            )
        )
        if baseline is not None:
            directory_manifest_lines = _manifest_line_count(layout.original_directories_manifest)
            checks.append(
                VerificationCheck(
                    "inventory.directory_manifest_rows",
                    "pass" if directory_manifest_lines == baseline.directories else "fail",
                    baseline.directories,
                    directory_manifest_lines,
                    "Empty and populated source directories remain reconstructable",
                )
            )
        inventory_row = session.execute(
            select(
                func.count(func.distinct(SourceChunk.id)),
                func.count(SourceFile.id),
                func.coalesce(func.sum(SourceFile.size_bytes), 0),
                func.coalesce(
                    func.sum(func.iif(SourceFile.json_status != "not_json", 1, 0)),
                    0,
                ),
            )
            .select_from(SourceFile)
            .join(SourceChunk, SourceChunk.id == SourceFile.chunk_id)
            .where(SourceFile.scan_status == "discovered")
        ).one()
        if baseline is not None:
            inventory_expectations = (
                ("baseline.chunks", baseline.chunks, inventory_row[0]),
                ("baseline.files", baseline.files, inventory_row[1]),
                ("baseline.bytes", baseline.bytes, inventory_row[2]),
                ("baseline.json_files", baseline.json_files, inventory_row[3]),
            )
            for key, expected, actual in inventory_expectations:
                checks.append(
                    VerificationCheck(
                        key,
                        "pass" if expected == actual else "fail",
                        expected,
                        actual,
                        "Audited real-export inventory baseline",
                    )
                )
        unhashed = (
            session.scalar(
                select(func.count(SourceFile.id))
                .where(SourceFile.scan_status == "discovered")
                .where(SourceFile.sha256.is_(None))
            )
            or 0
        )
        checks.append(
            VerificationCheck(
                "inventory.hashes",
                "pass" if unhashed == 0 else "pending",
                0,
                unhashed,
                "Run a hashing scan before physical consolidation",
            )
        )
        unique_blob_rows = (
            select(
                SourceFile.sha256.label("sha256"),
                func.min(SourceFile.size_bytes).label("size_bytes"),
            )
            .where(SourceFile.scan_status == "discovered")
            .where(SourceFile.sha256.is_not(None))
            .group_by(SourceFile.sha256)
            .subquery()
        )
        unique_blob_count, unique_blob_bytes = session.execute(
            select(
                func.count(unique_blob_rows.c.sha256),
                func.coalesce(func.sum(unique_blob_rows.c.size_bytes), 0),
            )
        ).one()
        collision_groups = (
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
        duplicate_file_occurrences, duplicate_bytes = session.execute(
            select(
                func.count(SourceFile.id),
                func.coalesce(func.sum(SourceFile.size_bytes), 0),
            )
            .where(SourceFile.scan_status == "discovered")
            .where(SourceFile.sha256.is_not(None))
            .where(
                SourceFile.id.not_in(
                    select(func.min(SourceFile.id))
                    .where(SourceFile.scan_status == "discovered")
                    .where(SourceFile.sha256.is_not(None))
                    .group_by(SourceFile.sha256)
                )
            )
        ).one()
        if baseline is not None:
            for key, expected, actual in (
                ("baseline.unique_blobs", baseline.unique_blobs, unique_blob_count),
                ("baseline.unique_blob_bytes", baseline.unique_blob_bytes, unique_blob_bytes),
                (
                    "baseline.normalized_path_collisions",
                    baseline.normalized_path_collisions,
                    collision_groups,
                ),
                (
                    "baseline.duplicate_file_occurrences",
                    baseline.duplicate_file_occurrences,
                    duplicate_file_occurrences,
                ),
                ("baseline.duplicate_bytes", baseline.duplicate_bytes, duplicate_bytes),
            ):
                checks.append(
                    VerificationCheck(
                        key,
                        "pass" if expected == actual else "pending" if unhashed else "fail",
                        expected,
                        actual,
                        "Every physical copy was hashed; each byte sequence has one blob identity",
                    )
                )
        missing_blob_files = 0
        materialized_blob_count = 0
        writable_blob_files = 0
        source_shared_blob_files: set[str] = set()
        source_paths_by_hash: dict[str, list[tuple[str, str]]] = {}
        for source_sha256, chunk_path, source_path in session.execute(
            select(SourceFile.sha256, SourceChunk.original_path, SourceFile.relative_path)
            .join(SourceChunk, SourceChunk.id == SourceFile.chunk_id)
            .where(SourceFile.scan_status == "discovered")
            .where(SourceFile.sha256.is_not(None))
        ):
            source_paths_by_hash.setdefault(source_sha256, []).append((chunk_path, source_path))
        materialized_rows = session.execute(
            select(Blob.sha256, Blob.relative_path).where(
                Blob.state.in_(("materialized", "linked"))
            )
        ).all()
        for sha256, relative_path in materialized_rows:
            materialized_blob_count += 1
            if relative_path is None:
                missing_blob_files += 1
                continue
            blob_path = layout.root / relative_path
            try:
                blob_stat = blob_path.lstat()
            except FileNotFoundError:
                missing_blob_files += 1
                continue
            if not stat.S_ISREG(blob_stat.st_mode):
                missing_blob_files += 1
                continue
            if stat.S_IMODE(blob_stat.st_mode) & 0o222:
                writable_blob_files += 1
            for chunk_path, source_path in source_paths_by_hash.get(sha256, []):
                try:
                    source_stat = (layout.root / chunk_path / source_path).lstat()
                except FileNotFoundError:
                    continue
                if (
                    stat.S_ISREG(source_stat.st_mode)
                    and source_stat.st_dev == blob_stat.st_dev
                    and source_stat.st_ino == blob_stat.st_ino
                ):
                    source_shared_blob_files.add(sha256)
                    break
        checks.append(
            VerificationCheck(
                "blobs.materialized_files",
                (
                    "pending"
                    if materialized_blob_count != unique_blob_count
                    else "pass"
                    if missing_blob_files == 0
                    else "fail"
                ),
                unique_blob_count,
                materialized_blob_count,
                f"missing or non-regular catalogued blob paths={missing_blob_files}",
            )
        )
        checks.append(
            VerificationCheck(
                "blobs.immutable_independent",
                (
                    "pending"
                    if materialized_blob_count != unique_blob_count or missing_blob_files
                    else "pass"
                    if writable_blob_files == 0 and not source_shared_blob_files
                    else "fail"
                ),
                0,
                writable_blob_files + len(source_shared_blob_files),
                f"writable={writable_blob_files}, source-shared={len(source_shared_blob_files)}",
            )
        )
        json_pending = (
            session.scalar(
                select(func.count(SourceFile.id)).where(SourceFile.json_status == "pending")
            )
            or 0
        )
        json_invalid = (
            session.scalar(
                select(func.count(SourceFile.id)).where(SourceFile.json_status == "invalid")
            )
            or 0
        )
        checks.append(
            VerificationCheck(
                "json.parse_status",
                "pass" if json_pending == 0 and json_invalid == 0 else "pending",
                0,
                json_pending + json_invalid,
                f"pending={json_pending}, invalid={json_invalid}",
            )
        )
        message_count = session.scalar(select(func.count(Message.pk))) or 0
        segment_count = session.scalar(select(func.count(ConversationSegment.id))) or 0
        checks.append(
            VerificationCheck(
                "messages.imported",
                "pass" if message_count > 0 and segment_count > 0 else "pending",
                "nonzero after ingest",
                message_count,
                f"{segment_count} conversation segments",
            )
        )
        ordinary_filter = ConversationSegment.source_category != "support"
        support_filter = ConversationSegment.source_category == "support"
        ordinary_segments = (
            session.scalar(select(func.count(ConversationSegment.id)).where(ordinary_filter)) or 0
        )
        support_segments = (
            session.scalar(select(func.count(ConversationSegment.id)).where(support_filter)) or 0
        )
        ordinary_messages = (
            session.scalar(
                select(func.count(Message.pk))
                .join(ConversationSegment, ConversationSegment.id == Message.segment_id)
                .where(ordinary_filter)
            )
            or 0
        )
        support_messages = (
            session.scalar(
                select(func.count(Message.pk))
                .join(ConversationSegment, ConversationSegment.id == Message.segment_id)
                .where(support_filter)
            )
            or 0
        )
        participant_rows = (
            session.scalar(
                select(func.count(SegmentParticipant.id))
                .join(ConversationSegment, ConversationSegment.id == SegmentParticipant.segment_id)
                .where(ordinary_filter)
            )
            or 0
        )
        reaction_rows = (
            session.scalar(
                select(func.count(MessageReaction.id))
                .join(Message, Message.id == MessageReaction.message_id)
                .join(ConversationSegment, ConversationSegment.id == Message.segment_id)
                .where(ordinary_filter)
            )
            or 0
        )
        attachment_rows = (
            session.scalar(
                select(func.count(AttachmentOccurrence.id))
                .join(Message, Message.id == AttachmentOccurrence.message_id)
                .join(ConversationSegment, ConversationSegment.id == Message.segment_id)
                .where(ordinary_filter)
            )
            or 0
        )
        activity_rows = session.scalar(select(func.count(ActivityRecord.pk))) or 0
        duplicate_rows = (
            session.scalar(
                select(func.count(Message.pk))
                .join(ConversationSegment, ConversationSegment.id == Message.segment_id)
                .where(ConversationSegment.source_category != "support")
                .where(Message.duplicate_of_id.is_not(None))
            )
            or 0
        )
        transcript_containers = int(
            session.scalar(
                text(
                    "SELECT count(DISTINCT containers.key) "
                    "FROM conversation_segments AS segments, "
                    "json_each(segments.metadata_json, '$.containers') AS containers "
                    "WHERE segments.source_category != 'support'"
                )
            )
            or 0
        )
        missing_local_attachment_occurrences = (
            session.scalar(
                select(func.count(AttachmentOccurrence.id))
                .where(AttachmentOccurrence.availability == "missing")
                .where(AttachmentOccurrence.deleted_at.is_(None))
            )
            or 0
        )
        unlinked_media = (
            session.scalar(
                select(func.count(AttachmentOccurrence.id)).where(
                    AttachmentOccurrence.availability == "unlinked",
                    AttachmentOccurrence.deleted_at.is_(None),
                )
            )
            or 0
        )
        leading_gap_segments = (
            session.scalar(
                select(func.count(IntegrityAnomaly.id))
                .where(IntegrityAnomaly.code == "missing_leading_message_chunk")
                .where(IntegrityAnomaly.resolved.is_(False))
            )
            or 0
        )
        internal_gap_segments = (
            session.scalar(
                select(func.count(IntegrityAnomaly.id))
                .where(IntegrityAnomaly.code == "missing_internal_message_chunk")
                .where(IntegrityAnomaly.resolved.is_(False))
            )
            or 0
        )
        activity_reactions_raw, activity_reactions_logical = session.execute(
            select(
                func.count(ActivityRecord.pk),
                func.count(func.distinct(SourceRecord.logical_fingerprint)),
            )
            .join(SourceRecord, SourceRecord.id == ActivityRecord.source_record_id)
            .where(ActivityRecord.category == "comments_and_reactions")
            .where(ActivityRecord.subtype == "reaction")
        ).one()
        if baseline is not None:
            content_expectations = (
                (
                    "baseline.transcript_containers",
                    baseline.transcript_containers,
                    transcript_containers,
                ),
                ("baseline.ordinary_segments", baseline.ordinary_segments, ordinary_segments),
                ("baseline.ordinary_messages", baseline.ordinary_messages, ordinary_messages),
                (
                    "baseline.canonical_participants",
                    baseline.canonical_participants,
                    participant_rows,
                ),
                ("baseline.message_reactions", baseline.message_reactions, reaction_rows),
                ("baseline.message_attachments", baseline.message_attachments, attachment_rows),
                ("baseline.support_segments", baseline.support_segments, support_segments),
                ("baseline.support_messages", baseline.support_messages, support_messages),
                ("baseline.activity_records", baseline.activity_records, activity_rows),
                (
                    "baseline.exact_duplicate_messages",
                    baseline.exact_duplicate_messages,
                    duplicate_rows,
                ),
                (
                    "baseline.unresolved_local_uri_occurrences",
                    baseline.unresolved_local_uri_occurrences,
                    missing_local_attachment_occurrences,
                ),
                (
                    "baseline.unlinked_media_paths",
                    baseline.unlinked_media_paths,
                    unlinked_media,
                ),
                (
                    "baseline.leading_message_gap_segments",
                    baseline.leading_message_gap_segments,
                    leading_gap_segments,
                ),
                (
                    "baseline.internal_message_gap_segments",
                    baseline.internal_message_gap_segments,
                    internal_gap_segments,
                ),
                (
                    "baseline.activity_reactions_raw",
                    baseline.activity_reactions_raw,
                    activity_reactions_raw,
                ),
                (
                    "baseline.activity_reactions_logical",
                    baseline.activity_reactions_logical,
                    activity_reactions_logical,
                ),
            )
            ingest_complete = json_pending == 0 and json_invalid == 0
            for key, expected, actual in content_expectations:
                detail = (
                    "All active missing attachment occurrences, including message and activity "
                    "sources"
                    if key == "baseline.unresolved_local_uri_occurrences"
                    else "Audited real-export ingestion baseline"
                )
                checks.append(
                    VerificationCheck(
                        key,
                        "pass" if expected == actual else "fail" if ingest_complete else "pending",
                        expected,
                        actual,
                        detail,
                    )
                )
        active_conversation_count = (
            session.scalar(
                select(func.count(Conversation.id)).where(Conversation.deleted_at.is_(None))
            )
            or 0
        )
        active_message_count = (
            session.scalar(select(func.count(Message.pk)).where(Message.deleted_at.is_(None))) or 0
        )
        active_attachment_count = (
            session.scalar(
                select(func.count(AttachmentOccurrence.id)).where(
                    AttachmentOccurrence.deleted_at.is_(None)
                )
            )
            or 0
        )
        active_activity_count = (
            session.scalar(
                select(func.count(ActivityRecord.pk)).where(ActivityRecord.deleted_at.is_(None))
            )
            or 0
        )
        star_targets = cast(
            list[StarTargetKey],
            list(
                session.execute(
                    select(Star.target_type, Star.target_id).order_by(
                        Star.target_type,
                        Star.target_id,
                    )
                )
            ),
        )
        star_resolutions = resolve_star_targets(
            session,
            star_targets,
            include_payload=False,
        )
        star_state_counts = {
            state: sum(resolution.state == state for resolution in star_resolutions.values())
            for state in ("active", "latent", "missing")
        }
        checks.append(
            VerificationCheck(
                "stars.targets",
                "pass" if star_state_counts["missing"] == 0 else "fail",
                0,
                star_state_counts["missing"],
                (
                    f"active={star_state_counts['active']}, "
                    f"latent={star_state_counts['latent']}, "
                    f"missing={star_state_counts['missing']}"
                ),
            )
        )
        checks.append(
            _library_marker_check(
                layout,
                conversation_count=active_conversation_count,
                message_count=active_message_count,
                attachment_count=active_attachment_count,
                activity_count=active_activity_count,
                unlinked_count=unlinked_media,
            )
        )
        checks.append(_latest_idempotency_check(session, kind="messages"))
        checks.append(_latest_idempotency_check(session, kind="activity"))
    if deep:
        checks.append(_deep_manifest_check(layout, progress=progress))
    return [asdict(check) for check in checks]


def write_verification_report(
    layout: ArchiveLayout, checks: list[dict[str, object]], *, filename: str = "latest.json"
) -> Path:
    target = layout.manifests / "integrity-reports" / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    source_manifests = source_manifest_artifacts(layout)
    marker_artifact = _artifact_record(layout, layout.library_completion_marker, jsonl=False)
    non_pass = mandatory_non_pass_checks(checks)
    deep_verified = any(
        check.get("key") == "deep.source_bytes" and check.get("status") == "pass"
        for check in checks
    )
    library_verified = any(
        check.get("key") == "library.completion_marker" and check.get("status") == "pass"
        for check in checks
    )
    artifacts_complete = (
        all(value is not None for value in source_manifests.values())
        and marker_artifact is not None
    )
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "checks": checks,
        "artifacts": {
            "source_manifests": source_manifests,
            "library_marker": marker_artifact,
        },
        "gate": {
            "all_mandatory_checks_pass": not non_pass,
            "deep_verified": deep_verified,
            "library_verified": library_verified,
            "required_artifacts_present": artifacts_complete,
            "source_retirement_eligible": (
                not non_pass and deep_verified and library_verified and artifacts_complete
            ),
            "non_pass_keys": [str(check.get("key", "unknown")) for check in non_pass],
        },
    }
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(target)
    return target
