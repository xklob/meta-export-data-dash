from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any, Literal, cast
from urllib.parse import quote, urlencode

from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

from meta_archive.models import ConflictLabel, SearchWindow
from meta_archive.services.conflict_scoring import CONFLICT_LABELS

CONFLICT_EVALUATION_SPLITS = ("train", "validation", "test")
CONFLICT_SPLIT_ALGORITHM_VERSION = "conversation-sha256-v1"
CONFLICT_SPLIT_RATIOS = {"train": 70, "validation": 15, "test": 15}

ConflictEvaluationSplit = Literal["train", "validation", "test"]
ConflictReviewSplit = Literal["all", "train", "validation", "test"]


class ConflictCalibrationIneligibleError(RuntimeError):
    def __init__(self, eligibility: dict[str, Any]) -> None:
        super().__init__("Conflict calibration data gates have not been satisfied")
        self.eligibility = eligibility


def conflict_split_algorithm() -> dict[str, Any]:
    return {
        "version": CONFLICT_SPLIT_ALGORITHM_VERSION,
        "ratios": CONFLICT_SPLIT_RATIOS.copy(),
    }


def deterministic_conflict_split(conversation_id: str) -> ConflictEvaluationSplit:
    """Assign one whole conversation to a stable held-out evaluation split."""

    digest = hashlib.sha256(
        f"{CONFLICT_SPLIT_ALGORITHM_VERSION}\0{conversation_id}".encode()
    ).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    if bucket < CONFLICT_SPLIT_RATIOS["train"]:
        return "train"
    if bucket < CONFLICT_SPLIT_RATIOS["train"] + CONFLICT_SPLIT_RATIOS["validation"]:
        return "validation"
    return "test"


def _latest_labels(session: Session) -> dict[str, tuple[ConflictLabel, str, str | None]]:
    rows = session.execute(
        select(ConflictLabel, SearchWindow.entity_type, SearchWindow.conversation_id)
        .join(SearchWindow, SearchWindow.id == ConflictLabel.window_id)
        .order_by(ConflictLabel.created_at.desc(), ConflictLabel.id.desc())
    )
    latest: dict[str, tuple[ConflictLabel, str, str | None]] = {}
    for label, entity_type, conversation_id in rows:
        latest.setdefault(label.window_id, (label, entity_type, conversation_id))
    return latest


def _window_counts_by_conversation(session: Session) -> dict[str, int]:
    rows = session.execute(
        select(SearchWindow.conversation_id, func.count(SearchWindow.id))
        .where(
            SearchWindow.entity_type == "message_window",
            SearchWindow.conversation_id.is_not(None),
        )
        .group_by(SearchWindow.conversation_id)
    )
    return {cast(str, conversation_id): int(count) for conversation_id, count in rows}


def conflict_review_status(engine: Engine, *, minimum_label_count: int) -> dict[str, Any]:
    labels = tuple(str(label) for label in CONFLICT_LABELS)
    with Session(engine) as session:
        window_counts_by_conversation = _window_counts_by_conversation(session)
        latest = _latest_labels(session)

    by_label = {label: 0 for label in labels}
    by_split: dict[str, dict[str, Any]] = {
        split: {
            "eligible_window_count": 0,
            "manual_label_count": 0,
            "unlabelled_window_count": 0,
            "by_label": {label: 0 for label in labels},
        }
        for split in CONFLICT_EVALUATION_SPLITS
    }
    for conversation_id, count in window_counts_by_conversation.items():
        by_split[deterministic_conflict_split(conversation_id)]["eligible_window_count"] += count

    eligible_labelled_window_ids: set[str] = set()
    stored_splits_by_conversation: dict[str, set[str]] = {}
    missing_conversation_count = 0
    invalid_entity_count = 0
    unsupported_label_count = 0
    unsupported_split_count = 0
    split_assignment_mismatch_count = 0
    manual_label_count = 0

    for window_id, (row, entity_type, label_conversation_id) in latest.items():
        if entity_type != "message_window":
            invalid_entity_count += 1
            continue
        if label_conversation_id is None:
            missing_conversation_count += 1
            continue
        eligible_labelled_window_ids.add(window_id)
        expected_split = deterministic_conflict_split(label_conversation_id)
        if row.evaluation_split not in CONFLICT_EVALUATION_SPLITS:
            unsupported_split_count += 1
        else:
            stored_splits_by_conversation.setdefault(label_conversation_id, set()).add(
                row.evaluation_split
            )
            if row.evaluation_split != expected_split:
                split_assignment_mismatch_count += 1
        if row.label not in by_label:
            unsupported_label_count += 1
            continue
        manual_label_count += 1
        by_label[row.label] += 1
        by_split[expected_split]["manual_label_count"] += 1
        by_split[expected_split]["by_label"][row.label] += 1

    for split in CONFLICT_EVALUATION_SPLITS:
        split_status = by_split[split]
        split_status["unlabelled_window_count"] = max(
            0,
            int(split_status["eligible_window_count"])
            - sum(
                1
                for window_id in eligible_labelled_window_ids
                if (
                    latest[window_id][2] is not None
                    and deterministic_conflict_split(cast(str, latest[window_id][2])) == split
                )
            ),
        )

    leaked_conversation_count = sum(
        len(stored_splits) > 1 for stored_splits in stored_splits_by_conversation.values()
    )
    missing_strata = [
        {"evaluation_split": split, "label": label}
        for split in CONFLICT_EVALUATION_SPLITS
        for label in labels
        if by_split[split]["by_label"][label] == 0
    ]
    empty_split_count = sum(
        by_split[split]["manual_label_count"] == 0 for split in CONFLICT_EVALUATION_SPLITS
    )
    blockers: list[dict[str, Any]] = []
    if manual_label_count < minimum_label_count:
        blockers.append(
            {
                "code": "insufficient_manual_labels",
                "message": f"At least {minimum_label_count} genuine manual labels are required.",
                "count": minimum_label_count - manual_label_count,
            }
        )
    for code, message, count in (
        (
            "missing_conversation",
            "Every labelled window must belong to a conversation.",
            missing_conversation_count,
        ),
        (
            "unsupported_entity_type",
            "Only message-window labels may be calibrated.",
            invalid_entity_count,
        ),
        (
            "unsupported_label",
            "Every manual label must use the supported six-way label set.",
            unsupported_label_count,
        ),
        (
            "unsupported_evaluation_split",
            "Every manual label must use a supported evaluation split.",
            unsupported_split_count,
        ),
        (
            "split_assignment_mismatch",
            "Stored splits must match the deterministic conversation assignment.",
            split_assignment_mismatch_count,
        ),
        (
            "conversation_split_leakage",
            "A conversation may not appear in more than one evaluation split.",
            leaked_conversation_count,
        ),
        (
            "empty_evaluation_split",
            "Train, validation, and test must each contain manual labels.",
            empty_split_count,
        ),
        (
            "missing_strata",
            "Every evaluation split must contain all six manual labels.",
            len(missing_strata),
        ),
    ):
        if count:
            blockers.append({"code": code, "message": message, "count": count})

    eligible_window_count = sum(window_counts_by_conversation.values())
    validation = {
        "eligible_for_calibration": not blockers,
        "conversation_disjoint": leaked_conversation_count == 0,
        "stratified": not missing_strata,
        "exact_split_assignment": split_assignment_mismatch_count == 0
        and unsupported_split_count == 0,
        "missing_conversation_count": missing_conversation_count,
        "split_assignment_mismatch_count": split_assignment_mismatch_count,
        "leaked_conversation_count": leaked_conversation_count,
        "missing_strata": missing_strata,
        "blockers": blockers,
    }
    return {
        "labels": list(labels),
        "minimum_label_count": minimum_label_count,
        "eligible_window_count": eligible_window_count,
        "manual_label_count": manual_label_count,
        "unlabelled_window_count": max(
            0, eligible_window_count - len(eligible_labelled_window_ids)
        ),
        "evaluation_split_algorithm": conflict_split_algorithm(),
        "by_label": by_label,
        "by_split": by_split,
        "validation": validation,
        "relationship_ranking_enabled": False,
        "warning": "Manual conflict review must never be used to rank people or relationships.",
    }


def _latest_label_subquery() -> Any:
    ranked = select(
        ConflictLabel.window_id.label("window_id"),
        ConflictLabel.label.label("label"),
        ConflictLabel.notes.label("notes"),
        ConflictLabel.created_at.label("created_at"),
        func.row_number()
        .over(
            partition_by=ConflictLabel.window_id,
            order_by=(ConflictLabel.created_at.desc(), ConflictLabel.id.desc()),
        )
        .label("row_number"),
    ).subquery()
    return (
        select(
            ranked.c.window_id,
            ranked.c.label,
            ranked.c.notes,
            ranked.c.created_at,
        )
        .where(ranked.c.row_number == 1)
        .subquery()
    )


def _review_jump(window: SearchWindow) -> dict[str, str]:
    conversation_id = cast(str, window.conversation_id)
    encoded_conversation = quote(conversation_id, safe="")
    if window.first_message_id:
        encoded_message = quote(window.first_message_id, safe="")
        return {
            "href": (
                f"/conversations/{encoded_conversation}?around={encoded_message}"
                f"#message-{encoded_message}"
            ),
            "label": "Open conversation at this window",
        }
    return {
        "href": f"/conversations/{encoded_conversation}",
        "label": "Open conversation",
    }


def _iso_datetime(value: datetime | str | None) -> str | None:
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _next_unlabelled_href(split: ConflictReviewSplit, after: str | None) -> str:
    query: dict[str, str] = {"split": split}
    if after is not None:
        query["after"] = after
    return f"/api/v1/enrichment/conflict/review/next?{urlencode(query)}"


def conflict_review_queue(
    engine: Engine,
    *,
    label: str,
    split: ConflictReviewSplit,
    after: str | None,
    limit: int,
    minimum_label_count: int,
) -> dict[str, Any]:
    status = conflict_review_status(engine, minimum_label_count=minimum_label_count)
    if split == "all":
        total = (
            status["unlabelled_window_count"]
            if label == "unlabelled"
            else status["by_label"][label]
        )
    else:
        split_status = status["by_split"][split]
        total = (
            split_status["unlabelled_window_count"]
            if label == "unlabelled"
            else split_status["by_label"][label]
        )

    latest = _latest_label_subquery()
    with Session(engine) as session:
        statement = (
            select(
                SearchWindow,
                latest.c.label,
                latest.c.notes,
                latest.c.created_at,
            )
            .outerjoin(latest, latest.c.window_id == SearchWindow.id)
            .where(
                SearchWindow.entity_type == "message_window",
                SearchWindow.conversation_id.is_not(None),
            )
            .order_by(SearchWindow.id)
        )
        if after is not None:
            statement = statement.where(SearchWindow.id > after)
        if label == "unlabelled":
            statement = statement.where(latest.c.window_id.is_(None))
        else:
            statement = statement.where(latest.c.label == label)
        if split != "all":
            conversation_ids = [
                conversation_id
                for conversation_id in _window_counts_by_conversation(session)
                if deterministic_conflict_split(conversation_id) == split
            ]
            if not conversation_ids:
                rows: list[Any] = []
            else:
                rows = list(
                    session.execute(
                        statement.where(SearchWindow.conversation_id.in_(conversation_ids)).limit(
                            limit + 1
                        )
                    )
                )
        else:
            rows = list(session.execute(statement.limit(limit + 1)))

    has_more = len(rows) > limit
    page_rows = rows[:limit]
    items: list[dict[str, Any]] = []
    for window, current_label, notes, labelled_at in page_rows:
        conversation_id = cast(str, window.conversation_id)
        items.append(
            {
                "window_id": window.id,
                "conversation_id": conversation_id,
                "evaluation_split": deterministic_conflict_split(conversation_id),
                "first_message_id": window.first_message_id,
                "last_message_id": window.last_message_id,
                "first_timestamp_ms": window.first_timestamp_ms,
                "last_timestamp_ms": window.last_timestamp_ms,
                "turn_count": window.turn_count,
                "text_display": window.text_display,
                "current_manual_label": (
                    {
                        "label": current_label,
                        "notes": notes,
                        "created_at": _iso_datetime(labelled_at),
                    }
                    if current_label is not None
                    else None
                ),
                "jump": _review_jump(window),
            }
        )
    next_after = items[-1]["window_id"] if has_more and items else None
    navigation_after = items[-1]["window_id"] if items else after
    return {
        "items": items,
        "page": {
            "limit": limit,
            "returned": len(items),
            "total": int(total),
            "next_after": next_after,
        },
        "filters": {"label": label, "evaluation_split": split},
        "navigation": {"next_unlabelled_href": _next_unlabelled_href(split, navigation_after)},
    }


def next_unlabelled_conflict_window(
    engine: Engine,
    *,
    split: ConflictReviewSplit,
    after: str | None,
    minimum_label_count: int,
) -> dict[str, Any]:
    page = conflict_review_queue(
        engine,
        label="unlabelled",
        split=split,
        after=after,
        limit=1,
        minimum_label_count=minimum_label_count,
    )
    wrapped = False
    if not page["items"] and after is not None:
        page = conflict_review_queue(
            engine,
            label="unlabelled",
            split=split,
            after=None,
            limit=1,
            minimum_label_count=minimum_label_count,
        )
        wrapped = bool(page["items"])
    item = page["items"][0] if page["items"] else None
    return {
        "item": item,
        "wrapped": wrapped,
        "navigation": {
            "next_unlabelled_href": _next_unlabelled_href(
                split, item["window_id"] if item is not None else None
            )
        },
    }
