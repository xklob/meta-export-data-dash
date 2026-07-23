from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import Text, case, cast, desc, func, or_, select
from sqlalchemy.orm import Session

from meta_archive import __version__
from meta_archive.api.schemas import SettingsUpdate
from meta_archive.api.security import require_session
from meta_archive.models import (
    ActivityRecord,
    ArchiveSetting,
    AttachmentOccurrence,
    Conversation,
    CoverageArea,
    IntegrityAnomaly,
    Job,
    Message,
    SourceChunk,
    SourceFile,
)
from meta_archive.paths import UnsafeArchivePathError, open_regular_file_beneath

router = APIRouter(prefix="/api/v1", dependencies=[Depends(require_session)])
public_router = APIRouter(prefix="/api/v1")


def _session(request: Request) -> Session:
    return Session(request.app.state.engine)


def _serialize_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    # SQLite does not retain timezone offsets. All persisted timestamps are UTC by contract, so
    # restore that fact explicitly instead of emitting an ambiguous browser-local timestamp.
    value = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    return value.isoformat()


def _setting_value(session: Session, key: str, default: Any) -> Any:
    row = session.get(ArchiveSetting, key)
    if row is None:
        return default
    return row.value_json.get("value", default)


def _set_setting(session: Session, key: str, value: Any) -> None:
    existing = session.get(ArchiveSetting, key)
    if existing is None:
        session.add(ArchiveSetting(key=key, value_json={"value": value}))
    else:
        existing.value_json = {"value": value}


def _verification_checks(request: Request) -> list[dict[str, Any]]:
    report_path = request.app.state.layout.manifests / "integrity-reports" / "latest.json"
    try:
        with open_regular_file_beneath(
            report_path,
            request.app.state.layout.root,
        ) as handle:
            payload = handle.read(16 * 1024 * 1024 + 1)
        if len(payload) > 16 * 1024 * 1024:
            return []
        report = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, UnsafeArchivePathError):
        return []
    checks = report.get("checks") if isinstance(report, dict) else None
    return (
        [check for check in checks if isinstance(check, dict)] if isinstance(checks, list) else []
    )


def _catalog_lifecycle(request: Request, session: Session) -> dict[str, Any]:
    source_count, pending_json, invalid_json = session.execute(
        select(
            func.count(SourceFile.id),
            func.coalesce(func.sum(case((SourceFile.json_status == "pending", 1), else_=0)), 0),
            func.coalesce(func.sum(case((SourceFile.json_status == "invalid", 1), else_=0)), 0),
        ).where(SourceFile.scan_status == "discovered")
    ).one()
    active_jobs = (
        session.scalar(
            select(func.count(Job.id)).where(Job.status.in_(("running", "pending", "queued")))
        )
        or 0
    )
    checks = _verification_checks(request)
    failed_checks = sum(check.get("status") == "fail" for check in checks)
    pending_checks = sum(check.get("status") != "pass" for check in checks) - failed_checks
    if active_jobs:
        state = "processing"
    elif not source_count:
        state = "uninitialized"
    elif invalid_json or failed_checks:
        state = "failed"
    elif pending_json:
        state = "partial"
    elif not checks or pending_checks:
        state = "verification_pending"
    else:
        state = "ready"
    return {
        "state": state,
        "ready": state == "ready",
        "active_job_count": active_jobs,
        "pending_json_count": int(pending_json or 0),
        "invalid_json_count": int(invalid_json or 0),
        "verification_check_count": len(checks),
        "verification_pending_count": pending_checks,
        "verification_failed_count": failed_checks,
    }


@public_router.get("/health")
def health(request: Request) -> dict[str, Any]:
    with _session(request) as session:
        lifecycle = _catalog_lifecycle(request, session)
    return {
        "status": "ok",
        "version": __version__,
        "catalog_ready": lifecycle["ready"],
        "catalog_state": lifecycle["state"],
    }


@router.get("/overview")
def overview(request: Request) -> dict[str, Any]:
    with _session(request) as session:
        source_stats = session.execute(
            select(
                func.count(SourceFile.id),
                func.coalesce(func.sum(SourceFile.size_bytes), 0),
                func.coalesce(
                    func.sum(case((SourceFile.json_status != "not_json", 1), else_=0)), 0
                ),
                func.coalesce(func.sum(case((SourceFile.json_status == "parsed", 1), else_=0)), 0),
                func.coalesce(func.sum(case((SourceFile.json_status == "invalid", 1), else_=0)), 0),
                func.coalesce(func.sum(case((SourceFile.json_status == "pending", 1), else_=0)), 0),
                func.coalesce(
                    func.sum(case((SourceFile.json_status == "unsupported", 1), else_=0)), 0
                ),
            ).where(SourceFile.scan_status == "discovered")
        ).one()
        content_stats = {
            "conversation_count": session.scalar(select(func.count(Conversation.id))) or 0,
            "message_count": session.scalar(
                select(func.count(Message.pk)).where(Message.deleted_at.is_(None))
            )
            or 0,
            "attachment_count": session.scalar(
                select(func.count(AttachmentOccurrence.id)).where(
                    AttachmentOccurrence.deleted_at.is_(None)
                )
            )
            or 0,
            "activity_count": session.scalar(
                select(func.count(ActivityRecord.pk)).where(ActivityRecord.deleted_at.is_(None))
            )
            or 0,
        }
        unresolved = IntegrityAnomaly.resolved.is_(False)
        integrity = {
            "unresolved_anomalies": session.scalar(
                select(func.count(IntegrityAnomaly.id)).where(unresolved)
            )
            or 0,
            "critical_anomalies": session.scalar(
                select(func.count(IntegrityAnomaly.id))
                .where(unresolved)
                .where(IntegrityAnomaly.severity == "critical")
            )
            or 0,
            "missing_attachments": session.scalar(
                select(func.count(AttachmentOccurrence.id)).where(
                    AttachmentOccurrence.availability == "missing"
                )
            )
            or 0,
            "unlinked_media": session.scalar(
                select(func.count(AttachmentOccurrence.id)).where(
                    AttachmentOccurrence.availability == "unlinked"
                )
            )
            or 0,
        }
        latest_job = session.scalar(select(Job).order_by(desc(Job.created_at)).limit(1))
        chunk_count = session.scalar(select(func.count(SourceChunk.id))) or 0
        lifecycle = _catalog_lifecycle(request, session)
    return {
        "archive": {
            "root_name": request.app.state.layout.root.name,
            "chunk_count": chunk_count,
            "source_file_count": source_stats[0],
            "total_bytes": source_stats[1],
            "json_file_count": source_stats[2],
            "json_valid_count": source_stats[3],
            "json_invalid_count": source_stats[4],
            "json_pending_count": source_stats[5],
            "json_unsupported_count": source_stats[6],
        },
        "content": content_stats,
        "integrity": integrity,
        "latest_job": _job_payload(latest_job) if latest_job else None,
        "lifecycle": lifecycle,
    }


@router.get("/coverage")
def coverage(request: Request) -> dict[str, Any]:
    with _session(request) as session:
        rows = session.scalars(select(CoverageArea).order_by(CoverageArea.label)).all()
    return {
        "items": [
            {
                "key": row.key,
                "label": row.label,
                "status": row.status,
                "raw_count": row.raw_count,
                "logical_count": row.logical_count,
                "description": row.description,
                "updated_at": _serialize_datetime(row.updated_at),
            }
            for row in rows
        ]
    }


@router.get("/integrity")
def integrity(
    request: Request,
    severity: str | None = None,
    unresolved_only: bool = True,
    q: str | None = Query(default=None, max_length=200),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    with _session(request) as session:
        statement = select(IntegrityAnomaly)
        if severity:
            statement = statement.where(IntegrityAnomaly.severity == severity)
        if unresolved_only:
            statement = statement.where(IntegrityAnomaly.resolved.is_(False))
        if q and (query := q.strip()):
            pattern = f"%{query}%"
            statement = statement.where(
                or_(
                    IntegrityAnomaly.code.ilike(pattern),
                    IntegrityAnomaly.summary.ilike(pattern),
                    cast(IntegrityAnomaly.detail_json, Text).ilike(pattern),
                )
            )
        filtered_total = (
            session.scalar(select(func.count()).select_from(statement.order_by(None).subquery()))
            or 0
        )
        rows = session.scalars(
            statement.order_by(desc(IntegrityAnomaly.created_at), IntegrityAnomaly.id)
            .offset(offset)
            .limit(limit)
        ).all()
        counts: dict[str, int] = {
            row_severity: count
            for row_severity, count in session.execute(
                select(IntegrityAnomaly.severity, func.count(IntegrityAnomaly.id))
                .where(IntegrityAnomaly.resolved.is_(False))
                .group_by(IntegrityAnomaly.severity)
            )
        }
        total = session.scalar(select(func.count(IntegrityAnomaly.id))) or 0
        unresolved = (
            session.scalar(
                select(func.count(IntegrityAnomaly.id)).where(IntegrityAnomaly.resolved.is_(False))
            )
            or 0
        )
    checks = _verification_checks(request)
    return {
        "summary": {
            "total": total,
            "unresolved": unresolved,
            "critical": counts.get("critical", 0),
            "warning": counts.get("warning", 0),
            "info": counts.get("info", 0),
        },
        "page": {"limit": limit, "offset": offset, "filtered_total": filtered_total},
        "items": [
            {
                "id": row.id,
                "code": row.code,
                "severity": row.severity,
                "summary": row.summary,
                "detail": row.detail_json,
                "resolved": row.resolved,
                "created_at": _serialize_datetime(row.created_at),
            }
            for row in rows
        ],
        "checks": checks,
    }


def _job_payload(row: Job) -> dict[str, Any]:
    return {
        "id": row.id,
        "kind": row.kind,
        "status": row.status,
        "progress_current": row.progress_current,
        "progress_total": row.progress_total,
        "stage": row.stage,
        "counters": row.counters_json,
        "started_at": _serialize_datetime(row.started_at),
        "finished_at": _serialize_datetime(row.finished_at),
        "error": row.error_text,
    }


@router.get("/jobs")
def jobs(
    request: Request,
    limit: int = Query(default=50, ge=1, le=250),
    offset: int = Query(default=0, ge=0),
    status_filter: str | None = Query(default=None, alias="status"),
    q: str | None = Query(default=None, max_length=200),
) -> dict[str, Any]:
    with _session(request) as session:
        statement = select(Job)
        if status_filter:
            states = {
                "running": ("running", "pending", "queued"),
                "complete": (
                    "complete",
                    "completed",
                    "completed_with_errors",
                    "success",
                    "succeeded",
                ),
                "failed": ("failed", "error", "cancelled"),
            }.get(status_filter, (status_filter,))
            statement = statement.where(Job.status.in_(states))
        if q and (query := q.strip()):
            pattern = f"%{query}%"
            statement = statement.where(
                or_(Job.kind.ilike(pattern), Job.stage.ilike(pattern), Job.id.ilike(pattern))
            )
        filtered_total = (
            session.scalar(select(func.count()).select_from(statement.order_by(None).subquery()))
            or 0
        )
        total = session.scalar(select(func.count(Job.id))) or 0
        status_counts = {
            value: count
            for value, count in session.execute(
                select(Job.status, func.count(Job.id)).group_by(Job.status)
            )
        }
        rows = session.scalars(
            statement.order_by(desc(Job.created_at)).offset(offset).limit(limit)
        ).all()
    return {
        "items": [_job_payload(row) for row in rows],
        "page": {
            "limit": limit,
            "offset": offset,
            "filtered_total": filtered_total,
            "total": total,
        },
        "summary": {
            "running": sum(
                status_counts.get(value, 0) for value in ("running", "pending", "queued")
            ),
            "completed": sum(
                status_counts.get(value, 0)
                for value in (
                    "complete",
                    "completed",
                    "completed_with_errors",
                    "success",
                    "succeeded",
                )
            ),
            "failed": sum(
                status_counts.get(value, 0) for value in ("failed", "error", "cancelled")
            ),
            "total": total,
        },
    }


def _settings_payload(request: Request, session: Session) -> dict[str, Any]:
    return {
        "timezone": _setting_value(session, "timezone", "America/New_York"),
        "owner_identity": _setting_value(session, "owner_identity", "Archive Owner"),
        "cloud_enabled": bool(_setting_value(session, "cloud_enabled", False)),
        "image_semantic_stage_enabled": False,
        "archive_root_display": request.app.state.layout.root.name,
        "access_mode": "localhost-only",
    }


@router.get("/settings")
def get_settings(request: Request) -> dict[str, Any]:
    with _session(request) as session:
        return _settings_payload(request, session)


@router.patch("/settings")
def update_settings(request: Request, payload: SettingsUpdate) -> dict[str, Any]:
    with _session(request) as session, session.begin():
        if payload.timezone is not None:
            try:
                ZoneInfo(payload.timezone)
            except ZoneInfoNotFoundError as exc:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Unknown IANA timezone",
                ) from exc
            _set_setting(session, "timezone", payload.timezone)
        if payload.cloud_enabled is not None:
            if payload.cloud_enabled and not os.environ.get("OPENAI_API_KEY"):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="OPENAI_API_KEY is not configured in the server environment",
                )
            _set_setting(session, "cloud_enabled", payload.cloud_enabled)
        session.flush()
        return _settings_payload(request, session)
