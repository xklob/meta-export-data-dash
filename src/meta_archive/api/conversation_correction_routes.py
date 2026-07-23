from __future__ import annotations

from dataclasses import asdict
from typing import Annotated, Any, NoReturn

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from meta_archive.api.security import require_session
from meta_archive.services.conversation_corrections import (
    ConversationCorrectionConflictError,
    ConversationCorrectionError,
    StaleConversationCorrectionError,
    conversation_identity_revision,
    list_conversation_corrections,
    merge_direct_histories,
    revert_conversation_correction,
    split_direct_history,
)
from meta_archive.services.deletion import DeletionError
from meta_archive.services.search_index import SemanticIndexBusyError

conversation_correction_router = APIRouter(
    prefix="/api/v1/conversation-corrections",
    dependencies=[Depends(require_session)],
)


class MergeCorrectionRequest(BaseModel):
    conversation_ids: Annotated[list[str], Field(min_length=2, max_length=100)]
    target_conversation_id: Annotated[str, Field(min_length=1, max_length=64)]
    identity_revision: Annotated[str, Field(min_length=64, max_length=64)]
    reason: Annotated[str | None, Field(max_length=1_000)] = None


class SplitCorrectionRequest(BaseModel):
    conversation_id: Annotated[str, Field(min_length=1, max_length=64)]
    segment_ids: Annotated[list[str], Field(min_length=1, max_length=100)]
    title: Annotated[str, Field(min_length=1, max_length=240)]
    identity_revision: Annotated[str, Field(min_length=64, max_length=64)]
    reason: Annotated[str | None, Field(max_length=1_000)] = None


class RevertCorrectionRequest(BaseModel):
    identity_revision: Annotated[str, Field(min_length=64, max_length=64)]


def _raise_correction_error(exc: Exception) -> NoReturn:
    if isinstance(
        exc,
        (
            StaleConversationCorrectionError,
            ConversationCorrectionConflictError,
            DeletionError,
            SemanticIndexBusyError,
        ),
    ):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    if isinstance(exc, ConversationCorrectionError):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc
    raise exc


@conversation_correction_router.get("")
def correction_history(
    request: Request,
    conversation_id: str | None = Query(default=None, max_length=64),
    limit: int = Query(default=100, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    with Session(request.app.state.engine) as session:
        items, total = list_conversation_corrections(
            session,
            conversation_id=conversation_id,
            limit=limit,
            offset=offset,
        )
        revision = conversation_identity_revision(session)
    return {
        "items": items,
        "identity_revision": revision,
        "page": {
            "limit": limit,
            "offset": offset,
            "total": total,
            "next_offset": offset + len(items) if offset + len(items) < total else None,
        },
        "notice": (
            "Direct histories are provisionally grouped by exact counterpart display label. "
            "Manual corrections change segment assignments only; source paths and records remain "
            "unchanged."
        ),
    }


@conversation_correction_router.post("/merge", status_code=status.HTTP_201_CREATED)
def merge_correction(request: Request, payload: MergeCorrectionRequest) -> dict[str, Any]:
    try:
        result = merge_direct_histories(
            request.app.state.layout,
            request.app.state.engine,
            conversation_ids=payload.conversation_ids,
            target_conversation_id=payload.target_conversation_id,
            expected_revision=payload.identity_revision,
            reason=payload.reason,
        )
    except Exception as exc:
        _raise_correction_error(exc)
    return asdict(result)


@conversation_correction_router.post("/split", status_code=status.HTTP_201_CREATED)
def split_correction(request: Request, payload: SplitCorrectionRequest) -> dict[str, Any]:
    try:
        result = split_direct_history(
            request.app.state.layout,
            request.app.state.engine,
            conversation_id=payload.conversation_id,
            segment_ids=payload.segment_ids,
            title=payload.title,
            expected_revision=payload.identity_revision,
            reason=payload.reason,
        )
    except Exception as exc:
        _raise_correction_error(exc)
    return asdict(result)


@conversation_correction_router.post("/{correction_id}/revert")
def revert_correction(
    request: Request,
    correction_id: str,
    payload: RevertCorrectionRequest,
) -> dict[str, Any]:
    try:
        result = revert_conversation_correction(
            request.app.state.layout,
            request.app.state.engine,
            correction_id,
            expected_revision=payload.identity_revision,
        )
    except Exception as exc:
        _raise_correction_error(exc)
    return asdict(result)
