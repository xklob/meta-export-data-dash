from __future__ import annotations

import hashlib
import json
from typing import Any

from fastapi import HTTPException, Request, status
from sqlalchemy.orm import Session

from meta_archive.models import CoverageArea


def session_for(request: Request) -> Session:
    return Session(request.app.state.engine)


def csv_values(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))


def escaped_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def scope_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def invalid_cursor(detail: str = "Invalid or mismatched pagination cursor") -> HTTPException:
    return HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=detail)


def coverage_payload(
    session: Session,
    key: str,
    *,
    label: str,
    unavailable_description: str,
    available_description: str | None = None,
    actual_count: int = 0,
    status_with_data: str = "partial",
) -> dict[str, Any]:
    """Return coverage without turning absent export data into a zero-history claim."""

    row = session.get(CoverageArea, key)
    if actual_count > 0:
        if row is not None and row.status != "unavailable":
            status_value = row.status
            raw_count: int | None = row.raw_count if row.raw_count is not None else actual_count
            logical_count: int | None = (
                row.logical_count if row.logical_count is not None else actual_count
            )
            description = row.description
            paths = row.source_paths_json
        else:
            status_value = status_with_data
            raw_count = actual_count
            logical_count = actual_count
            description = (
                available_description or f"Retained {label.casefold()} records are available."
            )
            paths = {}
    else:
        status_value = row.status if row is not None else "unavailable"
        raw_count = None if status_value == "unavailable" else row.raw_count if row else 0
        logical_count = None if status_value == "unavailable" else row.logical_count if row else 0
        description = row.description if row is not None else unavailable_description
        paths = row.source_paths_json if row is not None else {}
    return {
        "key": key,
        "label": row.label if row is not None else label,
        "status": status_value,
        "raw_count": raw_count,
        "logical_count": logical_count,
        "description": description,
        "source_paths": paths,
        "absence_is_not_zero_history": status_value == "unavailable",
    }


def unavailable_page(coverage: dict[str, Any], *, limit: int) -> dict[str, Any]:
    return {
        "coverage": coverage,
        "items": [],
        "page": {
            "total": None,
            "limit": limit,
            "next_cursor": None,
            "count_is_unknown": True,
        },
    }
