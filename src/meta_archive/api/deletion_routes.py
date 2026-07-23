from __future__ import annotations

import re
from dataclasses import asdict
from datetime import date, datetime
from typing import Any, Literal, NoReturn
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import and_, desc, func, or_, select
from sqlalchemy.orm import Session

from meta_archive.api.pagination import (
    InvalidCursor,
    decode_cursor,
    encode_cursor,
    require_cursor_int,
    require_cursor_str,
)
from meta_archive.api.security import require_session
from meta_archive.models import (
    ActivityRecord,
    AttachmentOccurrence,
    Conversation,
    DeletionAudit,
    DeletionItem,
    DeletionOperation,
    Message,
)
from meta_archive.services.deletion import (
    DeletionError,
    DeletionSelector,
    HardPurgeBlockedError,
    InvalidDeletionTokenError,
    StaleDeletionPreviewError,
    UnsafeDeletionError,
    commit_deletion,
    execute_hard_purge,
    plan_hard_purge,
    preview_deletion,
    restore_deletion,
)
from meta_archive.services.jumps import activity_jump

deletion_router = APIRouter(prefix="/api/v1/deletions", dependencies=[Depends(require_session)])

EntityType = Literal["message", "activity", "conversation", "attachment"]
SELECTION_PREVIEW_LIMIT = 160
SELECTION_LABEL_LIMIT = 96
SELECTION_PREVIEW_STATUSES = frozenset({"previewed", "committed"})
RESTORE_REBUILD_PRODUCT_KINDS = (
    "search_windows",
    "semantic_index",
    "nlp_hourly_rollups",
    "conflict_aggregates",
    "topic_clusters",
)
RESTORE_REBUILD_REASONS = frozenset({"intervening_hard_purge", "later_quarantine"})
PURGE_BLOCKER_CODES = frozenset(
    {
        "active_vector_table_missing",
        "catalogued_derivative_outside_root",
        "catalogued_derivative_owner_mismatch",
        "catalogued_derivative_path_unsafe",
        "active_conversation_correction_references_selected_rows",
        "deletion_manifest_missing_or_unsafe",
        "dependent_rows_belong_to_another_operation",
        "derivative_directory_unreadable",
        "derivative_entry_unreadable",
        "derivative_file_outside_root",
        "derivative_root_unreadable",
        "derivative_root_unsafe",
        "derivative_special_file",
        "derivative_symlink",
        "derived_rows_recreated_after_normalized_purge",
        "new_attachment_references_block_workspace_upgrade",
        "new_source_file_references_block_workspace_upgrade",
        "normalized_rows_missing",
        "normalized_rows_not_quarantined",
        "normalized_rows_recreated_after_normalized_purge",
        "outside_source_file_references_require_review",
        "physical_source_changed_after_normalized_purge",
        "physical_blob_gained_a_recoverable_reference",
        "raw_source_rewrite_validation_failed",
        "raw_source_catalog_changed_after_normalized_purge",
        "selected_blob_identifier_invalid",
        "selected_rows_belong_to_another_committed_deletion",
        "selected_rows_belong_to_an_unknown_deletion_state",
        "source_manifests_changed_after_normalized_purge",
        "source_records_recreated_after_normalized_purge",
        "unattributed_derivative_file",
        "unregistered_vector_tables_require_review",
        "vector_catalog_count_mismatch",
        "vector_catalog_identity_mismatch",
        "vector_store_inventory_failed",
    }
)
PURGE_SUMMARY_COUNT_SOURCES: dict[str, tuple[str, str]] = {
    "normalized_rows": ("impact", "normalized_rows"),
    "search_windows": ("impact", "search_windows"),
    "candidate_blobs": ("impact", "candidate_blobs"),
    "logically_hidden_blobs": ("impact", "logically_hidden_blobs"),
    "candidate_blob_bytes": ("impact", "candidate_blob_bytes"),
    "shared_blobs_preserved": ("impact", "shared_blobs_preserved"),
    "media_derivatives": ("impact", "media_derivatives"),
    "media_derivative_bytes": ("impact", "media_derivative_bytes"),
    "audio_transcriptions": ("impact", "audio_transcriptions"),
    "audio_transcript_segments": ("impact", "audio_transcript_segments"),
    "attachment_text_extractions": ("impact", "attachment_text_extractions"),
    "nlp_annotations": ("impact", "nlp_annotations"),
    "raw_source_files": ("impact", "raw_source_files"),
    "raw_source_files_rewritten": ("impact", "raw_source_files_rewritten"),
    "raw_source_obsolete_blobs": ("impact", "raw_source_obsolete_blobs"),
    "workspace_files": ("impact", "workspace_files"),
    "workspace_bytes": ("impact", "workspace_bytes"),
    "filesystem_library_files": ("impact", "filesystem_library_files"),
    "analytics_cache_files": ("impact", "analytics_cache_files"),
    "vector_store_files": ("impact", "vector_store_files"),
    "affected_messages": ("affected", "message"),
    "affected_activities": ("affected", "activity"),
    "affected_conversations": ("affected", "conversation"),
    "affected_attachments": ("affected", "attachment"),
    "affected_blobs": ("affected", "blobs"),
    "affected_source_records": ("affected", "source_records"),
    "affected_search_windows": ("affected", "search_windows"),
    "staged_workspace_files": ("root", "staged_workspace_files"),
    "deleted_vector_rows": ("root", "deleted_vector_rows"),
    "raw_source_rewrites": ("root", "raw_source_rewrites"),
}
AUDIT_ENTITY_COUNT_FIELDS: dict[str, str] = {
    "message": "affected_messages",
    "activity": "affected_activities",
    "conversation": "affected_conversations",
    "attachment": "affected_attachments",
}
AUDIT_ACTION_RULES: dict[str, tuple[frozenset[str | None], frozenset[str]]] = {
    "commit": (frozenset({"logical_quarantine"}), frozenset({"completed"})),
    "restore": (frozenset({"logical_quarantine"}), frozenset({"completed"})),
    "purge_preview": (
        frozenset({"normalized", "workspace"}),
        frozenset({"blocked", "ready"}),
    ),
    "purge_precommit_recovery": (
        frozenset({"normalized", "workspace"}),
        frozenset({"restored"}),
    ),
    "purge": (
        frozenset({"normalized", "workspace"}),
        frozenset({"cleanup_pending"}),
    ),
    "purge_finalize": (
        frozenset({"normalized", "workspace"}),
        frozenset({"completed"}),
    ),
}


def _default_entity_types() -> list[EntityType]:
    return ["message", "activity"]


class DeletionPreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_types: list[EntityType] = Field(
        default_factory=_default_entity_types, min_length=1, max_length=4
    )
    text: str | None = Field(default=None, min_length=1, max_length=10_000)
    text_mode: Literal["exact", "keyword"] = "exact"
    case_sensitive: bool = True
    whole_field: bool = False
    text_variant: Literal["raw", "display"] = "display"
    from_date: date | None = None
    to_date: date | None = None
    timezone: str = Field(default="America/New_York", min_length=1, max_length=100)
    conversation_ids: list[str] = Field(default_factory=list, max_length=10_000)
    participant: str | None = Field(default=None, min_length=1, max_length=500)
    media_types: list[Literal["photo", "video", "audio", "gif", "sticker", "document", "other"]] = (
        Field(default_factory=list, max_length=7)
    )
    availability: list[Literal["resolved", "missing", "unlinked", "external"]] = Field(
        default_factory=list, max_length=4
    )
    entity_ids: dict[EntityType, list[str]] = Field(default_factory=dict)
    allow_all: bool = False

    def selector(self) -> DeletionSelector:
        return DeletionSelector(
            entity_types=tuple(self.entity_types),
            text=self.text,
            text_mode=self.text_mode,
            case_sensitive=self.case_sensitive,
            whole_field=self.whole_field,
            text_variant=self.text_variant,
            from_date=self.from_date.isoformat() if self.from_date else None,
            to_date=self.to_date.isoformat() if self.to_date else None,
            timezone=self.timezone,
            conversation_ids=tuple(self.conversation_ids),
            participant=self.participant,
            media_types=tuple(self.media_types),
            availability=tuple(self.availability),
            entity_ids={key: tuple(values) for key, values in self.entity_ids.items()},
            allow_all=self.allow_all,
        )


class DeletionCommitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    commit_token: str = Field(min_length=20, max_length=500)
    catalog_revision: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")


class DeletionRestoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirmation: str = Field(min_length=36, max_length=36)


class DeletionPreviewPublicResponse(BaseModel):
    operation_id: str
    commit_token: str
    catalog_revision: str
    selection_sha256: str
    impact: dict[str, Any]


class HardPurgePreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope: Literal["normalized", "workspace"] = "workspace"


class HardPurgeCommitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope: Literal["normalized", "workspace"] = "workspace"
    purge_token: str = Field(min_length=20, max_length=500)
    purge_revision: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    confirmation: str = Field(min_length=36, max_length=36)


def _raise_service_error(exc: Exception) -> NoReturn:
    if isinstance(exc, InvalidDeletionTokenError):
        raise HTTPException(
            status_code=403,
            detail={
                "code": "deletion_authorization_failed",
                "message": "Deletion authorization failed. Refresh the operation and try again.",
            },
        ) from exc
    if isinstance(exc, UnsafeDeletionError):
        raise HTTPException(
            status_code=422,
            detail={
                "code": "deletion_request_unsafe",
                "message": "Deletion request failed safety validation.",
            },
        ) from exc
    if isinstance(exc, StaleDeletionPreviewError):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "deletion_preview_stale",
                "message": "Deletion preview is stale. Create a new preview and try again.",
            },
        ) from exc
    if isinstance(exc, HardPurgeBlockedError):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "hard_purge_blocked",
                "message": (
                    "Hard purge is blocked by a safety check. Preview again after resolving it."
                ),
            },
        ) from exc
    if isinstance(exc, DeletionError):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "deletion_operation_conflict",
                "message": "Deletion operation could not be completed in its current state.",
            },
        ) from exc
    raise exc


def _safe_sha256(value: object) -> str | None:
    if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value):
        return value
    return None


def _safe_timestamp(value: object) -> str | None:
    if not isinstance(value, str) or len(value) > 64:
        return None
    try:
        datetime.fromisoformat(value)
    except ValueError:
        return None
    return value


def _safe_count(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _public_purge_blockers(value: object) -> list[str]:
    """Return only stable blocker codes, stripping private path suffixes."""

    if not isinstance(value, (list, tuple)):
        return []
    blockers: list[str] = []
    for raw_blocker in value:
        if not isinstance(raw_blocker, str):
            continue
        blocker = raw_blocker.split(":", 1)[0]
        if blocker in PURGE_BLOCKER_CODES and blocker not in blockers:
            blockers.append(blocker)
    return blockers


def _public_purge_counts(raw: dict[str, Any]) -> dict[str, int]:
    raw_impact = raw.get("impact")
    raw_affected = raw.get("affected")
    impact: dict[str, Any] = raw_impact if isinstance(raw_impact, dict) else {}
    affected: dict[str, Any] = raw_affected if isinstance(raw_affected, dict) else {}
    sources: dict[str, dict[str, Any]] = {"root": raw, "impact": impact, "affected": affected}
    counts: dict[str, int] = {}
    for public_key, (source_name, private_key) in PURGE_SUMMARY_COUNT_SOURCES.items():
        value = _safe_count(sources[source_name].get(private_key))
        if value is not None:
            counts[public_key] = value
    return counts


def _public_purge_impact(raw_impact: object) -> dict[str, int]:
    """Project live purge impact without paths, diagnostics, or vector identities."""

    if not isinstance(raw_impact, dict):
        return {}
    return _public_purge_counts({"impact": raw_impact})


def _restore_rebuild_summary(value: object) -> dict[str, Any] | None:
    """Validate the small public receipt for derived products that must be rebuilt."""

    if not isinstance(value, dict):
        return None
    reason = value.get("reason")
    if value.get("required") is not True or reason not in RESTORE_REBUILD_REASONS:
        return None
    raw_product_kinds = value.get("product_kinds")
    invalidated_by_count = _safe_count(value.get("invalidated_by_count"))
    if not isinstance(raw_product_kinds, list) or invalidated_by_count is None:
        return None
    product_kinds = [
        product_kind
        for product_kind in RESTORE_REBUILD_PRODUCT_KINDS
        if product_kind in raw_product_kinds
    ]
    if not product_kinds:
        return None
    return {
        "required": True,
        "reason": reason,
        "product_kinds": product_kinds,
        "invalidated_by_count": invalidated_by_count,
    }


def _entity_affected_counts(raw: dict[str, Any]) -> dict[str, int]:
    affected = raw.get("affected")
    if not isinstance(affected, dict):
        return {}
    counts: dict[str, int] = {}
    for private_key, public_key in AUDIT_ENTITY_COUNT_FIELDS.items():
        value = _safe_count(affected.get(private_key))
        if value is not None:
            counts[public_key] = value
    return counts


def _public_audit_payload(audit: DeletionAudit) -> dict[str, Any]:
    """Project each audit action through a closed, action-specific public schema."""

    rules = AUDIT_ACTION_RULES.get(audit.action)
    if rules is None:
        return {
            "id": audit.id,
            "action": "unknown",
            "scope": None,
            "outcome": "unknown",
            "detail": {"counts": {}, "digests": {}, "status": {}},
            "created_at": audit.created_at,
        }

    allowed_scopes, allowed_outcomes = rules
    scope = audit.scope if audit.scope in allowed_scopes else None
    outcome = audit.outcome if audit.outcome in allowed_outcomes else "unknown"
    raw = audit.detail_json if isinstance(audit.detail_json, dict) else {}
    counts: dict[str, int] = {}
    digests: dict[str, str] = {}
    status: dict[str, Any] = {}

    if audit.action in {"commit", "restore"}:
        counts = _entity_affected_counts(raw)
        if audit.action == "commit":
            catalog_revision = _safe_sha256(raw.get("catalog_revision"))
            if catalog_revision:
                digests["catalog_revision"] = catalog_revision
    elif audit.action == "purge_preview":
        counts = _public_purge_impact(raw.get("impact"))
        purge_revision = _safe_sha256(raw.get("purge_revision"))
        if purge_revision:
            digests["purge_revision"] = purge_revision
        blockers = _public_purge_blockers(raw.get("blockers"))
        if blockers:
            status["blockers"] = blockers
    elif audit.action == "purge":
        counts = _public_purge_counts(raw)
    elif audit.action == "purge_finalize":
        counts = _entity_affected_counts(raw)
        recovered = raw.get("recovered")
        if isinstance(recovered, bool):
            status["recovered"] = recovered

    detail: dict[str, Any] = {"counts": counts, "digests": digests, "status": status}
    if audit.action == "restore":
        rebuild_required = _restore_rebuild_summary(raw.get("rebuild_required"))
        if rebuild_required:
            detail["restore_rebuild_required"] = rebuild_required
    return {
        "id": audit.id,
        "action": audit.action,
        "scope": scope,
        "outcome": outcome,
        "detail": detail,
        "created_at": audit.created_at,
    }


def _purge_summary(operation: DeletionOperation) -> dict[str, Any] | None:
    """Project private purge/recovery state through a small public allowlist."""

    raw = operation.purge_plan_json
    if not isinstance(raw, dict) or not raw:
        return None
    counts = _public_purge_counts(raw)

    status: dict[str, Any] = {"operation": operation.status}
    completed = raw.get("completed")
    if isinstance(completed, bool):
        status["completed"] = completed
    publication = raw.get("manifest_publication")
    if isinstance(publication, dict):
        action = publication.get("action")
        phase = publication.get("phase")
        if action in {"commit", "restore"}:
            status["publication_action"] = action
        if phase in {"preview", "prepared", "published"}:
            status["publication_phase"] = phase
    safe_blockers = _public_purge_blockers(raw.get("blockers"))
    if safe_blockers:
        status["blockers"] = safe_blockers

    digests: dict[str, str] = {}
    purge_revision = _safe_sha256(operation.purge_revision)
    restore_state_sha256 = _safe_sha256(raw.get("logical_restore_state_sha256"))
    if purge_revision:
        digests["purge_revision"] = purge_revision
    if restore_state_sha256:
        digests["logical_restore_state_sha256"] = restore_state_sha256

    timestamps: dict[str, str] = {}
    for key in ("planned_at", "completed_at"):
        value = _safe_timestamp(raw.get(key))
        if value:
            timestamps[key] = value

    scope = raw.get("scope") if raw.get("scope") in {"normalized", "workspace"} else None
    if scope is None and not counts and not digests and not timestamps and len(status) == 1:
        return None
    return {
        "scope": scope,
        "status": status,
        "counts": counts,
        "digests": digests,
        "timestamps": timestamps,
    }


def _operation_payload(operation: DeletionOperation, *, include_detail: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": operation.id,
        "status": operation.status,
        "item_count": operation.item_count,
        "impact": operation.impact_json,
        "created_at": operation.created_at,
        "updated_at": operation.updated_at,
        "committed_at": operation.committed_at,
        "restored_at": operation.restored_at,
        "purged_at": operation.purged_at,
    }
    if include_detail:
        payload.update(
            {
                "selector": operation.selector_json,
                "catalog_revision": operation.catalog_revision,
                "selection_sha256": operation.selection_sha256,
                "purge_summary": _purge_summary(operation),
                "restore_rebuild_required": _restore_rebuild_summary(
                    operation.purge_plan_json.get("restore_rebuild_required")
                    if isinstance(operation.purge_plan_json, dict)
                    else None
                ),
                "error_text": operation.error_text,
            }
        )
    return payload


def _bounded_display(value: str | None, *, limit: int) -> tuple[str | None, bool]:
    """Collapse whitespace and return only a deliberately short display projection."""

    if value is None:
        return None, False
    normalized = re.sub(r"\s+", " ", value).strip()
    if not normalized:
        return None, False
    return normalized[:limit], len(normalized) > limit


def _selection_jump(
    item: DeletionItem,
    row: Message | ActivityRecord | Conversation | AttachmentOccurrence,
) -> dict[str, str] | None:
    # Quarantined records are intentionally hidden from every public browser route. Returning
    # an anchor for them would look actionable but lead to a 404, so only retained rows get links.
    if row.deleted_at is not None:
        return None
    if item.entity_type == "message":
        message = row
        assert isinstance(message, Message)
        conversation_id = quote(message.conversation_id, safe="")
        message_id = quote(message.id, safe="")
        return {
            "href": (f"/conversations/{conversation_id}?around={message_id}#message-{message_id}"),
            "label": "Jump to message",
        }
    if item.entity_type == "activity":
        activity = row
        assert isinstance(activity, ActivityRecord)
        return {**activity_jump(activity.id), "label": "Jump to post or activity"}
    if item.entity_type == "conversation":
        conversation = row
        assert isinstance(conversation, Conversation)
        return {
            "href": f"/conversations/{quote(conversation.id, safe='')}",
            "label": "Open conversation",
        }

    attachment = row
    assert isinstance(attachment, AttachmentOccurrence)
    occurrence_id = quote(attachment.id, safe="")
    if attachment.conversation_id and attachment.message_id:
        conversation_id = quote(attachment.conversation_id, safe="")
        message_id = quote(attachment.message_id, safe="")
        return {
            "href": (
                f"/conversations/{conversation_id}?around={message_id}"
                f"&media={occurrence_id}#message-{message_id}"
            ),
            "label": "Jump to attachment in conversation",
        }
    if attachment.activity_record_id:
        return {
            **activity_jump(attachment.activity_record_id),
            "label": "Jump to attachment in post or activity",
        }
    if attachment.conversation_id:
        return {
            "href": f"/conversations/{quote(attachment.conversation_id, safe='')}",
            "label": "Open containing conversation",
        }
    return {
        "href": f"/media?occurrence={occurrence_id}",
        "label": "Open attachment occurrence",
    }


def _selection_item_payload(
    item: DeletionItem,
    row: Message | ActivityRecord | Conversation | AttachmentOccurrence | None,
) -> dict[str, Any]:
    label_source: str | None = None
    snippet_source: str | None = None
    timestamp_ms: int | None = None
    resource_kind: str | None = None
    availability: str | None = None
    media_type: str | None = None
    size_bytes: int | None = None

    if isinstance(row, Message):
        label_source = f"Message from {row.sender_display}"
        snippet_source = row.content_display or row.transcription_display
        timestamp_ms = row.timestamp_ms
        resource_kind = row.message_kind
    elif isinstance(row, ActivityRecord):
        label_source = row.title_display or f"{row.category}: {row.subtype}"
        snippet_source = row.content_display
        timestamp_ms = row.timestamp_ms
        resource_kind = row.subtype
        availability = row.availability
    elif isinstance(row, Conversation):
        label_source = row.title_display or "Conversation"
        timestamp_ms = row.last_activity_ms
        resource_kind = row.conversation_type
    elif isinstance(row, AttachmentOccurrence):
        # A filename may contain an exported source path. Show only its final component.
        filename = row.filename.replace("\\", "/").rsplit("/", 1)[-1] if row.filename else None
        label_source = filename or f"{row.kind} attachment"
        timestamp_ms = row.timestamp_ms
        resource_kind = row.kind
        availability = row.availability
        media_type = row.media_type
        size_bytes = row.size_bytes

    label, label_truncated = _bounded_display(
        label_source or f"Unavailable {item.entity_type} record", limit=SELECTION_LABEL_LIMIT
    )
    snippet, snippet_truncated = _bounded_display(snippet_source, limit=SELECTION_PREVIEW_LIMIT)
    return {
        "entity_type": item.entity_type,
        "entity_id": item.entity_id,
        "reason": item.reason,
        "label": label,
        "label_truncated": label_truncated,
        "snippet": snippet,
        "snippet_truncated": snippet_truncated,
        "preview_limit": SELECTION_PREVIEW_LIMIT,
        "timestamp_ms": timestamp_ms,
        "resource_kind": resource_kind,
        "availability": availability,
        "media_type": media_type,
        "size_bytes": size_bytes,
        "quarantined": bool(row is not None and row.deleted_at is not None),
        "jump": _selection_jump(item, row) if row is not None else None,
    }


def _selection_rows(
    session: Session, items: list[DeletionItem]
) -> dict[tuple[str, str], Message | ActivityRecord | Conversation | AttachmentOccurrence]:
    rows: dict[tuple[str, str], Message | ActivityRecord | Conversation | AttachmentOccurrence] = {}
    message_ids = [item.entity_id for item in items if item.entity_type == "message"]
    activity_ids = [item.entity_id for item in items if item.entity_type == "activity"]
    conversation_ids = [item.entity_id for item in items if item.entity_type == "conversation"]
    attachment_ids = [item.entity_id for item in items if item.entity_type == "attachment"]
    if message_ids:
        for message_row in session.scalars(select(Message).where(Message.id.in_(message_ids))):
            rows[("message", message_row.id)] = message_row
    if activity_ids:
        for activity_row in session.scalars(
            select(ActivityRecord).where(ActivityRecord.id.in_(activity_ids))
        ):
            rows[("activity", activity_row.id)] = activity_row
    if conversation_ids:
        for conversation_row in session.scalars(
            select(Conversation).where(Conversation.id.in_(conversation_ids))
        ):
            rows[("conversation", conversation_row.id)] = conversation_row
    if attachment_ids:
        for attachment_row in session.scalars(
            select(AttachmentOccurrence).where(AttachmentOccurrence.id.in_(attachment_ids))
        ):
            rows[("attachment", attachment_row.id)] = attachment_row
    return rows


@deletion_router.post("/preview", status_code=201, response_model=DeletionPreviewPublicResponse)
def create_deletion_preview(
    request: Request, payload: DeletionPreviewRequest
) -> DeletionPreviewPublicResponse:
    try:
        preview = preview_deletion(
            request.app.state.layout, request.app.state.engine, payload.selector()
        )
    except Exception as exc:
        _raise_service_error(exc)
    return DeletionPreviewPublicResponse(
        operation_id=preview.operation_id,
        commit_token=preview.commit_token,
        catalog_revision=preview.catalog_revision,
        selection_sha256=preview.selection_sha256,
        impact=preview.impact,
    )


@deletion_router.get("")
def list_deletion_operations(
    request: Request,
    operation_status: str | None = Query(default=None, alias="status", max_length=24),
    limit: int = Query(default=100, ge=1, le=500),
    cursor: str | None = Query(default=None, max_length=2_048),
) -> dict[str, Any]:
    scope = operation_status or "all"
    statement = select(DeletionOperation)
    if operation_status:
        statement = statement.where(DeletionOperation.status == operation_status)
    if cursor:
        try:
            decoded = decode_cursor(cursor, kind="deletion-operations")
            if require_cursor_str(decoded, "status") != scope:
                raise InvalidCursor("Pagination cursor does not match the deletion status filter")
            if require_cursor_int(decoded, "limit") != limit:
                raise InvalidCursor("Pagination cursor does not match the requested page size")
            cursor_created_at_text = require_cursor_str(decoded, "created_at")
            datetime.fromisoformat(cursor_created_at_text)
            cursor_id = require_cursor_str(decoded, "id")
        except (InvalidCursor, ValueError) as exc:
            raise HTTPException(
                status_code=422, detail=str(exc) or "Invalid deletion cursor"
            ) from exc
        operation_created_at = func.datetime(DeletionOperation.created_at)
        cursor_created_at = func.datetime(cursor_created_at_text)
        statement = statement.where(
            or_(
                operation_created_at < cursor_created_at,
                and_(
                    operation_created_at == cursor_created_at,
                    DeletionOperation.id < cursor_id,
                ),
            )
        )
    statement = statement.order_by(
        desc(func.datetime(DeletionOperation.created_at)), desc(DeletionOperation.id)
    ).limit(limit + 1)
    with Session(request.app.state.engine) as session:
        operations = list(session.scalars(statement))
        total_statement = select(func.count(DeletionOperation.id))
        if operation_status:
            total_statement = total_statement.where(DeletionOperation.status == operation_status)
        total = int(session.scalar(total_statement) or 0)
    page_rows = operations[:limit]
    next_cursor = None
    if len(operations) > limit and page_rows:
        last = page_rows[-1]
        next_cursor = encode_cursor(
            "deletion-operations",
            status=scope,
            limit=limit,
            created_at=last.created_at.isoformat(),
            id=last.id,
        )
    return {
        "items": [_operation_payload(item, include_detail=False) for item in page_rows],
        "page": {"total": total, "limit": limit, "next_cursor": next_cursor},
    }


@deletion_router.get("/{operation_id}/selection")
def get_deletion_selection(
    request: Request,
    operation_id: str,
    revision: str = Query(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"),
    limit: int = Query(default=25, ge=1, le=100),
    cursor: str | None = Query(default=None, max_length=2_048),
) -> dict[str, Any]:
    after_id: int | None = None
    if cursor:
        try:
            decoded = decode_cursor(cursor, kind="deletion-selection")
            if require_cursor_str(decoded, "operation_id") != operation_id:
                raise InvalidCursor("Pagination cursor does not match this deletion operation")
            if require_cursor_str(decoded, "revision") != revision:
                raise InvalidCursor("Pagination cursor does not match this selection revision")
            if require_cursor_int(decoded, "limit") != limit:
                raise InvalidCursor("Pagination cursor does not match the requested page size")
            after_id = require_cursor_int(decoded, "after_id")
            if after_id <= 0:
                raise InvalidCursor("Invalid pagination cursor")
        except InvalidCursor as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    with Session(request.app.state.engine) as session:
        operation = session.get(DeletionOperation, operation_id)
        if operation is None:
            raise HTTPException(status_code=404, detail="Deletion operation not found")
        if operation.selection_sha256 != revision:
            raise HTTPException(
                status_code=409,
                detail="Selection revision no longer matches this deletion operation",
            )
        if operation.status not in SELECTION_PREVIEW_STATUSES:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Exact selection browsing is available only while an operation is "
                    "previewed or committed"
                ),
            )
        if after_id is not None:
            anchor_exists = session.scalar(
                select(DeletionItem.id).where(
                    DeletionItem.id == after_id,
                    DeletionItem.operation_id == operation_id,
                )
            )
            if anchor_exists is None:
                raise HTTPException(status_code=422, detail="Invalid pagination cursor")

        total = int(
            session.scalar(
                select(func.count(DeletionItem.id)).where(DeletionItem.operation_id == operation_id)
            )
            or 0
        )
        if total != operation.item_count:
            raise HTTPException(
                status_code=409,
                detail="Stored deletion selection does not match its operation receipt",
            )
        statement = select(DeletionItem).where(DeletionItem.operation_id == operation_id)
        if after_id is not None:
            statement = statement.where(DeletionItem.id > after_id)
        items = list(session.scalars(statement.order_by(DeletionItem.id).limit(limit + 1)))
        page_items = items[:limit]
        rows = _selection_rows(session, page_items)
        payload_items = [
            _selection_item_payload(item, rows.get((item.entity_type, item.entity_id)))
            for item in page_items
        ]

    next_cursor = None
    if len(items) > limit and page_items:
        next_cursor = encode_cursor(
            "deletion-selection",
            operation_id=operation_id,
            revision=revision,
            limit=limit,
            after_id=page_items[-1].id,
        )
    return {
        "operation_id": operation_id,
        "revision": revision,
        "status": operation.status,
        "items": payload_items,
        "page": {"total": total, "limit": limit, "next_cursor": next_cursor},
        "privacy": {
            "snippet_limit": SELECTION_PREVIEW_LIMIT,
            "label_limit": SELECTION_LABEL_LIMIT,
            "raw_fields_exposed": False,
        },
    }


@deletion_router.get("/{operation_id}")
def get_deletion_operation(request: Request, operation_id: str) -> dict[str, Any]:
    with Session(request.app.state.engine) as session:
        operation = session.get(DeletionOperation, operation_id)
        if operation is None:
            raise HTTPException(status_code=404, detail="Deletion operation not found")
        payload = _operation_payload(operation, include_detail=True)
        audits = list(
            session.scalars(
                select(DeletionAudit)
                .where(DeletionAudit.operation_id == operation_id)
                .order_by(DeletionAudit.created_at, DeletionAudit.id)
            )
        )
        payload["audits"] = [_public_audit_payload(audit) for audit in audits]
    return payload


@deletion_router.post("/{operation_id}/commit")
def commit_deletion_operation(
    request: Request, operation_id: str, payload: DeletionCommitRequest
) -> dict[str, Any]:
    try:
        result = commit_deletion(
            request.app.state.layout,
            request.app.state.engine,
            operation_id,
            commit_token=payload.commit_token,
            catalog_revision=payload.catalog_revision,
        )
    except Exception as exc:
        _raise_service_error(exc)
    return asdict(result)


@deletion_router.post("/{operation_id}/restore")
def restore_deletion_operation(
    request: Request, operation_id: str, payload: DeletionRestoreRequest
) -> dict[str, Any]:
    try:
        result = restore_deletion(
            request.app.state.layout,
            request.app.state.engine,
            operation_id,
            confirmation=payload.confirmation,
        )
    except Exception as exc:
        _raise_service_error(exc)
    return asdict(result)


@deletion_router.post("/{operation_id}/purge/preview")
def create_hard_purge_preview(
    request: Request, operation_id: str, payload: HardPurgePreviewRequest
) -> dict[str, Any]:
    try:
        plan = plan_hard_purge(
            request.app.state.layout,
            request.app.state.engine,
            operation_id,
            scope=payload.scope,
        )
    except Exception as exc:
        _raise_service_error(exc)
    result = asdict(plan)
    result["impact"] = _public_purge_impact(plan.impact)
    result["blockers"] = _public_purge_blockers(plan.blockers)
    return result


@deletion_router.post("/{operation_id}/purge")
def purge_deletion_operation(
    request: Request, operation_id: str, payload: HardPurgeCommitRequest
) -> dict[str, Any]:
    try:
        result = execute_hard_purge(
            request.app.state.layout,
            request.app.state.engine,
            operation_id,
            scope=payload.scope,
            purge_token=payload.purge_token,
            purge_revision=payload.purge_revision,
            confirmation=payload.confirmation,
        )
    except Exception as exc:
        _raise_service_error(exc)
    return asdict(result)
