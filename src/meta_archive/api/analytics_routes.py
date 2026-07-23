from __future__ import annotations

import math
import re
import threading
import uuid
from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from copy import deepcopy
from datetime import UTC, date, datetime, timedelta
from itertools import pairwise
from statistics import fmean, median
from typing import Any, Literal, cast
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import Engine, Text, case, desc, func, or_, select, text, update
from sqlalchemy import cast as sql_cast
from sqlalchemy.orm import Session

from meta_archive.api.derived_publication import (
    guarded_derived_request,
    safe_derived_status_refresh,
)
from meta_archive.api.pagination import (
    InvalidCursor,
    decode_cursor,
    encode_cursor,
    require_cursor_int,
    require_cursor_str,
)
from meta_archive.api.security import require_session
from meta_archive.api.stage4_common import coverage_payload, invalid_cursor, scope_hash, session_for
from meta_archive.models import (
    ActivityRecord,
    AttachmentOccurrence,
    Blob,
    ConflictHourlyRollup,
    Conversation,
    ConversationSegment,
    CoverageArea,
    Job,
    Message,
    MessageReaction,
    NlpAnnotation,
    NlpHourlyRollup,
    Participant,
    SegmentParticipant,
    SemanticIndexConfig,
    SourceFile,
)
from meta_archive.services.analytics_cache import (
    StaleAnalyticsPayloadError,
    analytics_source_revision,
    read_analytics_cache,
    write_analytics_cache,
)
from meta_archive.services.conflict_aggregates import conflict_aggregate_status
from meta_archive.services.deletion_lock_lease import DeletionError
from meta_archive.services.derived_publication_guard import DerivedPublicationBlockedError
from meta_archive.services.identifiers import stable_id
from meta_archive.services.nlp_annotations import MODEL_SPECS, conflict_aggregation_status
from meta_archive.services.search import SearchFilters, _valid_window_statement, _window_items
from meta_archive.services.topic_clusters import (
    DEFAULT_CLUSTER_COUNT,
    DEFAULT_SAMPLE_LIMIT,
    build_topic_clusters,
    topic_cluster_lock_is_held,
)

analytics_router = APIRouter(prefix="/api/v1/analytics", dependencies=[Depends(require_session)])

TimeBucket = Literal["day", "week", "month", "year"]
NlpTimeBucket = Literal["hour", "day", "week", "month", "year"]
_URL_PATTERN = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
_WEEKDAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
_RESPONSE_BUCKETS = (
    (60_000, "under_1_minute"),
    (5 * 60_000, "1_to_5_minutes"),
    (30 * 60_000, "5_to_30_minutes"),
    (2 * 60 * 60_000, "30_minutes_to_2_hours"),
    (24 * 60 * 60_000, "2_to_24_hours"),
    (7 * 24 * 60 * 60_000, "1_to_7_days"),
)

_ANALYTICS_PERIOD_LIMIT = 120
_ANALYTICS_PERIOD_LIMIT_MAX = 500
_NLP_PERIOD_LIMIT = 24
_NLP_PERIOD_LIMIT_MAX = 500
_SERIES_SNAPSHOT_LOCK = threading.Lock()
_ACCEPTANCE_INVARIANT_STATE_LOCK = threading.Lock()
_ACCEPTANCE_INVARIANT_LIMIT = 64
_ACCEPTANCE_INVARIANT_LOCK_STRIPES = 16
_TOPIC_REQUEST_STATE_LOCK = threading.Lock()


class TopicClusterBuildRequest(BaseModel):
    cluster_count: int = Field(default=DEFAULT_CLUSTER_COUNT, ge=1, le=50)
    sample_limit: int = Field(default=DEFAULT_SAMPLE_LIMIT, ge=100, le=100_000)


def _topic_request_state(request: Request) -> tuple[threading.Lock, set[str]]:
    with _TOPIC_REQUEST_STATE_LOCK:
        request_lock = getattr(request.app.state, "topic_cluster_request_lock", None)
        if request_lock is None:
            request_lock = threading.Lock()
            request.app.state.topic_cluster_request_lock = request_lock
        dispatched_job_ids = getattr(
            request.app.state,
            "topic_cluster_dispatched_job_ids",
            None,
        )
        if dispatched_job_ids is None:
            dispatched_job_ids = set()
            request.app.state.topic_cluster_dispatched_job_ids = dispatched_job_ids
    return request_lock, dispatched_job_ids


def _period_page(
    items: list[dict[str, Any]],
    *,
    kind: str,
    scope: Mapping[str, object],
    limit: int,
    cursor: str | None,
    coordinate: str = "period",
) -> tuple[list[dict[str, Any]], dict[str, Any], set[str]]:
    """Return one stable, scope-bound page of chronological analytics buckets."""

    scope_token = scope_hash({"kind": kind, "limit": limit, **scope})
    ordered = sorted(items, key=lambda item: str(item[coordinate]))
    coordinates = [str(item[coordinate]) for item in ordered]
    start = 0
    if cursor is not None:
        try:
            decoded = decode_cursor(cursor, kind=f"analytics-{kind}")
            if require_cursor_str(decoded, "scope") != scope_token:
                raise InvalidCursor("Pagination cursor does not match these analytics filters")
            after = require_cursor_str(decoded, "after")
        except InvalidCursor as exc:
            raise invalid_cursor(str(exc)) from exc
        start = bisect_right(coordinates, after)
    page_items = ordered[start : start + limit]
    next_cursor = None
    if start + limit < len(ordered) and page_items:
        next_cursor = encode_cursor(
            f"analytics-{kind}",
            scope=scope_token,
            after=str(page_items[-1][coordinate]),
        )
    selected = {str(item[coordinate]) for item in page_items}
    return (
        page_items,
        {
            "limit": limit,
            "returned": len(page_items),
            "total": len(ordered),
            "next_cursor": next_cursor,
        },
        selected,
    )


def _series_snapshot_key(namespace: str, scope: Mapping[str, object]) -> str:
    return scope_hash({"namespace": namespace, **scope})


def _analytics_request_revision(request: Request) -> str:
    if bool(getattr(request.app.state, "acceptance_read_only", False)):
        revision = getattr(request.app.state, "analytics_acceptance_revision", None)
        if not isinstance(revision, str):
            revision = stable_id("acceptance-read-only-analytics", id(request.app.state.engine))
            request.app.state.analytics_acceptance_revision = revision
        return revision
    return analytics_source_revision(request.app.state.layout, request.app.state.engine)


def _series_snapshot(
    request: Request, namespace: str, scope: Mapping[str, object]
) -> dict[str, Any] | None:
    revision = _analytics_request_revision(request)
    key = _series_snapshot_key(namespace, scope)
    with _SERIES_SNAPSHOT_LOCK:
        snapshots = getattr(request.app.state, "analytics_series_snapshots", {})
        entry = snapshots.get(key)
        if isinstance(entry, dict) and entry.get("revision") == revision:
            return deepcopy(cast(dict[str, Any], entry["payload"]))
    return None


def _store_series_snapshot(
    request: Request,
    namespace: str,
    scope: Mapping[str, object],
    payload: dict[str, Any],
) -> None:
    key = _series_snapshot_key(namespace, scope)
    expected_revision = getattr(request.state, "analytics_cache_expected_revision", None)
    current_revision = _analytics_request_revision(request)
    if expected_revision is not None and current_revision != expected_revision:
        with _SERIES_SNAPSHOT_LOCK:
            snapshots = getattr(request.app.state, "analytics_series_snapshots", None)
            if isinstance(snapshots, dict):
                snapshots.pop(key, None)
        raise HTTPException(
            status_code=409,
            detail="Archive data changed while analytics were computed; retry the request",
        )
    revision = str(expected_revision or current_revision)
    with _SERIES_SNAPSHOT_LOCK:
        snapshots = getattr(request.app.state, "analytics_series_snapshots", None)
        if not isinstance(snapshots, dict):
            snapshots = {}
            request.app.state.analytics_series_snapshots = snapshots
        if len(snapshots) >= 32 and key not in snapshots:
            snapshots.clear()
        snapshots[key] = {"revision": revision, "payload": deepcopy(payload)}


def _page_series_snapshot(
    payload: dict[str, Any],
    *,
    kind: str,
    scope: Mapping[str, object],
    field: str,
    limit: int,
    cursor: str | None,
    coordinate: str = "period",
    clip_domain_periods: bool = False,
) -> dict[str, Any]:
    result = deepcopy(payload)
    items, page, selected = _period_page(
        cast(list[dict[str, Any]], result[field]),
        kind=kind,
        scope=scope,
        limit=limit,
        cursor=cursor,
        coordinate=coordinate,
    )
    result[field] = items
    result["page"] = page
    if clip_domain_periods:
        for domain in cast(list[dict[str, Any]], result.get("domains", [])):
            domain["periods"] = [
                row
                for row in cast(list[dict[str, Any]], domain.get("periods", []))
                if str(row["period"]) in selected
            ]
    return result


def _bucket_floor(value: datetime, bucket: NlpTimeBucket, zone: ZoneInfo) -> datetime:
    if bucket == "hour":
        return value.replace(minute=0, second=0, microsecond=0)
    local_date = value.date()
    if bucket == "week":
        local_date -= timedelta(days=local_date.weekday())
    elif bucket == "month":
        local_date = local_date.replace(day=1)
    elif bucket == "year":
        local_date = local_date.replace(month=1, day=1)
    return datetime(local_date.year, local_date.month, local_date.day, tzinfo=zone)


def _advance_bucket(value: datetime, bucket: NlpTimeBucket, count: int, zone: ZoneInfo) -> datetime:
    if bucket == "hour":
        return (value.astimezone(UTC) + timedelta(hours=count)).astimezone(zone)
    if bucket in {"day", "week"}:
        next_date = value.date() + timedelta(days=count * (7 if bucket == "week" else 1))
        return datetime(next_date.year, next_date.month, next_date.day, tzinfo=zone)
    if bucket == "month":
        ordinal = value.year * 12 + value.month - 1 + count
        year, month_index = divmod(ordinal, 12)
        return datetime(year, month_index + 1, 1, tzinfo=zone)
    return datetime(value.year + count, 1, 1, tzinfo=zone)


def _rollup_page_bounds(
    session: Session,
    *,
    hour_column: Any,
    filters: list[Any],
    kind: str,
    scope: Mapping[str, object],
    zone: ZoneInfo,
    bucket: NlpTimeBucket,
    limit: int,
    cursor: str | None,
) -> tuple[int | None, int | None, dict[str, Any]]:
    """Resolve one calendar-bucket range without materializing the full hourly history."""

    scope_token = scope_hash({"kind": kind, "limit": limit, **scope})
    start_ms: int | None
    if cursor is not None:
        try:
            decoded = decode_cursor(cursor, kind=f"analytics-{kind}")
            if require_cursor_str(decoded, "scope") != scope_token:
                raise InvalidCursor("Pagination cursor does not match these analytics filters")
            start_ms = require_cursor_int(decoded, "start_ms")
            if start_ms < 0:
                raise InvalidCursor("Invalid pagination cursor")
        except InvalidCursor as exc:
            raise invalid_cursor(str(exc)) from exc
    else:
        first_hour = session.scalar(select(func.min(hour_column)).where(*filters))
        if first_hour is None:
            return (
                None,
                None,
                {
                    "limit": limit,
                    "returned": 0,
                    "total": None,
                    "next_cursor": None,
                    "count_is_unknown": True,
                },
            )
        start = _bucket_floor(_local_datetime(int(first_hour), zone), bucket, zone)
        start_ms = int(start.astimezone(UTC).timestamp() * 1000)

    start = _local_datetime(start_ms, zone)
    end = _advance_bucket(start, bucket, limit, zone)
    end_ms = int(end.astimezone(UTC).timestamp() * 1000)
    next_hour = session.scalar(select(func.min(hour_column)).where(*filters, hour_column >= end_ms))
    next_cursor = None
    if next_hour is not None:
        next_start = _bucket_floor(_local_datetime(int(next_hour), zone), bucket, zone)
        next_cursor = encode_cursor(
            f"analytics-{kind}",
            scope=scope_token,
            start_ms=int(next_start.astimezone(UTC).timestamp() * 1000),
        )
    return (
        start_ms,
        end_ms,
        {
            "limit": limit,
            "returned": 0,
            "total": None,
            "next_cursor": next_cursor,
            "count_is_unknown": True,
        },
    )


def _acceptance_read_only(request: Request) -> bool:
    return bool(getattr(request.app.state, "acceptance_read_only", False))


def _acceptance_invariant_snapshot[AcceptanceInvariant](
    request: Request,
    namespace: str,
    scope: Mapping[str, object],
    loader: Callable[[], AcceptanceInvariant],
) -> AcceptanceInvariant:
    """Memoize immutable full-scope metadata only inside an acceptance process.

    Acceptance deliberately bypasses the persistent analytics response cache, but its catalog
    engine is bound to one read-only SQLite inode for the life of the process.  Cursor endpoints
    may therefore reuse invariant aggregate metadata in memory without skipping any page-local
    source rows or weakening the independent acceptance-runner reconciliation.
    """

    if not _acceptance_read_only(request):
        return loader()
    revision = _analytics_request_revision(request)
    key = _series_snapshot_key(f"acceptance-invariant:{namespace}", scope)
    with _ACCEPTANCE_INVARIANT_STATE_LOCK:
        snapshots = getattr(request.app.state, "analytics_acceptance_invariants", None)
        if not isinstance(snapshots, dict):
            snapshots = {}
            request.app.state.analytics_acceptance_invariants = snapshots
        raw_state_lock = getattr(
            request.app.state, "analytics_acceptance_invariant_state_lock", None
        )
        state_lock: Any = raw_state_lock
        if not hasattr(raw_state_lock, "acquire"):
            state_lock = threading.RLock()
            request.app.state.analytics_acceptance_invariant_state_lock = state_lock
        key_locks = getattr(request.app.state, "analytics_acceptance_invariant_key_locks", None)
        if not isinstance(key_locks, tuple) or len(key_locks) != _ACCEPTANCE_INVARIANT_LOCK_STRIPES:
            key_locks = tuple(
                threading.RLock() for _index in range(_ACCEPTANCE_INVARIANT_LOCK_STRIPES)
            )
            request.app.state.analytics_acceptance_invariant_key_locks = key_locks
    key_lock = key_locks[int(key[:8], 16) % len(key_locks)]
    with key_lock:
        with state_lock:
            entry = snapshots.get(key)
            if isinstance(entry, dict) and entry.get("revision") == revision:
                return deepcopy(cast(AcceptanceInvariant, entry["payload"]))
        payload = loader()
        if _analytics_request_revision(request) != revision:
            with state_lock:
                snapshots.pop(key, None)
            raise HTTPException(
                status_code=409,
                detail="Archive data changed while analytics were computed; retry the request",
            )
        with state_lock:
            if len(snapshots) >= _ACCEPTANCE_INVARIANT_LIMIT and key not in snapshots:
                snapshots.clear()
            snapshots[key] = {"revision": revision, "payload": deepcopy(payload)}
        return deepcopy(payload)


def _conflict_aggregate_status(request: Request) -> dict[str, Any]:
    engine = request.app.state.engine
    return _acceptance_invariant_snapshot(
        request,
        "conflict-aggregate-status",
        {},
        lambda: safe_derived_status_refresh(
            request.app.state.layout,
            engine,
            publisher="Conflict aggregate analytics status refresh",
            refresh=not _acceptance_read_only(request),
            loader=lambda refresh: conflict_aggregate_status(engine, refresh=refresh),
        ),
    )


def _cached_response(
    request: Request, namespace: str, parameters: Mapping[str, object]
) -> dict[str, Any] | None:
    request.state.analytics_cache_expected_revision = _analytics_request_revision(request)
    if _acceptance_read_only(request):
        return None
    return read_analytics_cache(
        request.app.state.layout,
        request.app.state.engine,
        namespace=namespace,
        parameters=parameters,
    )


def _store_response(
    request: Request,
    namespace: str,
    parameters: Mapping[str, object],
    payload: dict[str, Any],
) -> dict[str, Any]:
    if _acceptance_read_only(request):
        expected_revision = getattr(request.state, "analytics_cache_expected_revision", None)
        if (
            expected_revision is not None
            and _analytics_request_revision(request) != expected_revision
        ):
            raise HTTPException(
                status_code=409,
                detail="Archive data changed while analytics were computed; retry the request",
            )
        return payload
    try:
        return write_analytics_cache(
            request.app.state.layout,
            request.app.state.engine,
            namespace=namespace,
            parameters=parameters,
            payload=payload,
            expected_source_revision=getattr(
                request.state, "analytics_cache_expected_revision", None
            ),
        )
    except StaleAnalyticsPayloadError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def _zone(value: str) -> ZoneInfo:
    try:
        return ZoneInfo(value)
    except ZoneInfoNotFoundError as exc:
        raise HTTPException(status_code=422, detail=f"Unknown IANA timezone: {value}") from exc


def _local_datetime(timestamp_ms: int, zone: ZoneInfo) -> datetime:
    try:
        return datetime.fromtimestamp(timestamp_ms / 1000, zone)
    except (OSError, OverflowError, ValueError) as exc:
        raise HTTPException(
            status_code=422, detail="Timestamp is outside the supported range"
        ) from exc


def _period(value: datetime, bucket: TimeBucket) -> str:
    if bucket == "day":
        return value.strftime("%Y-%m-%d")
    if bucket == "week":
        iso_year, iso_week, _weekday = value.isocalendar()
        return f"{iso_year:04d}-W{iso_week:02d}"
    if bucket == "month":
        return value.strftime("%Y-%m")
    return value.strftime("%Y")


def _streaks(heatmap: dict[str, dict[str, int]]) -> dict[str, Any]:
    active = sorted(
        date.fromisoformat(key) for key, counts in heatmap.items() if counts["total"] > 0
    )
    if not active:
        return {
            "active_days": 0,
            "longest_days": 0,
            "latest_days": 0,
            "first_active_date": None,
            "last_active_date": None,
        }
    longest = 1
    current = 1
    latest = 1
    for previous, value in pairwise(active):
        if value == previous + timedelta(days=1):
            current += 1
        else:
            current = 1
        longest = max(longest, current)
    latest = current
    return {
        "active_days": len(active),
        "longest_days": longest,
        "latest_days": latest,
        "first_active_date": active[0].isoformat(),
        "last_active_date": active[-1].isoformat(),
    }


def _percentile(sorted_values: list[int], percentile: float) -> float | None:
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


def _response_summary(values: list[int]) -> dict[str, Any]:
    ordered = sorted(values)
    buckets: Counter[str] = Counter()
    for value in ordered:
        label = "over_7_days"
        lower = 0
        for ceiling, candidate in _RESPONSE_BUCKETS:
            if lower <= value < ceiling:
                label = candidate
                break
            lower = ceiling
        buckets[label] += 1
    return {
        "count": len(ordered),
        "mean_ms": fmean(ordered) if ordered else None,
        "median_ms": median(ordered) if ordered else None,
        "p90_ms": _percentile(ordered, 0.90),
        "buckets": {
            label: buckets[label]
            for label in [candidate for _ceiling, candidate in _RESPONSE_BUCKETS] + ["over_7_days"]
        },
    }


def _load_derived_nlp_status(request: Request) -> dict[str, dict[str, Any]]:
    with session_for(request) as session:
        rollup_stats = {
            str(task): {"buckets": int(buckets), "samples": int(samples or 0)}
            for task, buckets, samples in session.execute(
                select(
                    NlpHourlyRollup.task,
                    func.count(NlpHourlyRollup.id),
                    func.sum(NlpHourlyRollup.sample_count),
                )
                .where(NlpHourlyRollup.active.is_(True))
                .group_by(NlpHourlyRollup.task)
            )
        }
        # A complete active rollup is the authoritative analytics source for that task.  Avoid
        # rescanning millions of annotation payload rows merely to repeat its represented sample
        # count; only tasks without a rollup need the annotation table to distinguish processing
        # from unavailable.
        missing_rollups = tuple(
            task for task in ("sentiment", "emotion", "toxicity") if task not in rollup_stats
        )
        annotation_counts = (
            {
                str(task): int(count)
                for task, count in session.execute(
                    select(NlpAnnotation.task, func.count(NlpAnnotation.id))
                    .where(
                        NlpAnnotation.active.is_(True),
                        NlpAnnotation.status == "completed",
                        NlpAnnotation.task.in_(missing_rollups),
                    )
                    .group_by(NlpAnnotation.task)
                )
            }
            if missing_rollups
            else {}
        )
    result: dict[str, dict[str, Any]] = {}
    for task in ("sentiment", "emotion", "toxicity"):
        spec = MODEL_SPECS[cast(Any, task)]
        rollup = rollup_stats.get(task)
        annotations = rollup["samples"] if rollup else annotation_counts.get(task, 0)
        if rollup:
            status_value = "available"
            reason = (
                f"{rollup['samples']:,} active local annotations are available in "
                f"{rollup['buckets']:,} UTC-hour rollups."
            )
        elif annotations:
            status_value = "processing"
            reason = f"{annotations:,} annotations exist, but analytics rollups must be rebuilt."
        else:
            status_value = "unavailable"
            reason = "Derived NLP annotations have not been generated for this archive."
        result[task] = {
            "status": status_value,
            "reason": reason,
            "annotation_count": annotations,
            "rollup_bucket_count": rollup["buckets"] if rollup else 0,
            "sample_count": rollup["samples"] if rollup else 0,
            "model_name": spec.model_name,
            "model_revision": spec.model_revision,
            "attribution": spec.attribution,
            "warnings": list(spec.warnings),
        }
        if task == "toxicity":
            result[task]["relationship_ranking_enabled"] = False
    conflict = conflict_aggregation_status(request.app.state.engine)
    aggregate = _conflict_aggregate_status(request)
    result["conflict"] = {
        **conflict,
        "status": "available" if aggregate["available"] else aggregate["status"],
        "reason": aggregate["reason"] or conflict["warning"],
        "aggregate_available": aggregate["available"],
        "relationship_ranking_enabled": False,
    }
    return result


def _derived_nlp_status(request: Request) -> dict[str, dict[str, Any]]:
    return _acceptance_invariant_snapshot(
        request,
        "derived-nlp-status",
        {},
        lambda: _load_derived_nlp_status(request),
    )


def _nlp_period(value: datetime, bucket: NlpTimeBucket) -> str:
    if bucket == "hour":
        return value.strftime("%Y-%m-%dT%H:00%z")
    return _period(value, bucket)


def _message_range_filters(
    *, from_ms: int | None, to_ms: int | None, conversation_id: str | None
) -> list[Any]:
    if from_ms is not None and to_ms is not None and from_ms > to_ms:
        raise HTTPException(status_code=422, detail="Invalid analytics date range")
    filters: list[Any] = [Message.deleted_at.is_(None)]
    if from_ms is not None:
        filters.append(Message.timestamp_ms >= from_ms)
    if to_ms is not None:
        filters.append(Message.timestamp_ms <= to_ms)
    if conversation_id:
        filters.append(Message.conversation_id == conversation_id)
    return filters


def _attachment_type_expression() -> Any:
    media_type = func.lower(func.coalesce(AttachmentOccurrence.media_type, ""))
    return case(
        (AttachmentOccurrence.kind.in_(("gifs", "gif")), "gif"),
        (AttachmentOccurrence.kind == "sticker", "sticker"),
        (AttachmentOccurrence.kind.in_(("audio_files", "audio")), "audio"),
        (AttachmentOccurrence.kind.in_(("videos", "video")), "video"),
        (AttachmentOccurrence.kind.in_(("photos", "photo", "thread_image")), "photo"),
        (media_type.like("audio/%"), "audio"),
        (media_type.like("video/%"), "video"),
        (media_type.like("image/%"), "photo"),
        else_="file",
    )


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for child in value.values() for text in _strings(child)]
    if isinstance(value, list):
        return [text for child in value for text in _strings(child)]
    return []


def _message_link_hostnames(content: str | None, extra: Any) -> list[str]:
    hostnames: list[str] = []
    for candidate in ([content] if content else []) + _strings(extra):
        for match in _URL_PATTERN.findall(candidate):
            parsed = urlsplit(match.rstrip(".,);]}"))
            hostname = (parsed.hostname or "").casefold()
            if hostname:
                hostnames.append(hostname)
    return hostnames


def _message_has_link_candidate() -> Any:
    """Reject rows that cannot contain the URL grammar before JSON decoding in Python."""

    content = func.lower(func.coalesce(Message.content_display, ""))
    extra = func.lower(sql_cast(Message.extra_json, Text))
    return or_(
        func.instr(content, "http://") > 0,
        func.instr(content, "https://") > 0,
        func.instr(extra, "http://") > 0,
        func.instr(extra, "https://") > 0,
    )


def _link_domains(request: Request, *, limit: int = 100) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    with session_for(request) as session:
        rows = session.execute(
            select(Message.content_display, Message.extra_json).where(
                Message.deleted_at.is_(None),
                _message_has_link_candidate(),
            )
        ).yield_per(5_000)
        for content, extra in rows:
            counts.update(_message_link_hostnames(content, extra))
    return [{"domain": domain, "occurrences": count} for domain, count in counts.most_common(limit)]


@analytics_router.get("/interaction-series")
def interaction_series(
    request: Request,
    bucket: TimeBucket = "month",
    timezone: str = Query(default="America/New_York", max_length=100),
    from_ms: int | None = Query(default=None, ge=0),
    to_ms: int | None = Query(default=None, ge=0),
    domain_limit: int = Query(default=20, ge=1, le=100),
    limit: int = Query(default=_ANALYTICS_PERIOD_LIMIT, ge=1, le=_ANALYTICS_PERIOD_LIMIT_MAX),
    cursor: str | None = Query(default=None, max_length=2048),
) -> dict[str, Any]:
    """Return source-timestamped calls, reactions, audio, and links over time."""

    series_scope = {
        "bucket": bucket,
        "timezone": timezone,
        "from_ms": from_ms,
        "to_ms": to_ms,
        "domain_limit": domain_limit,
    }
    cache_parameters = {
        **series_scope,
        "limit": limit,
        "cursor": cursor,
    }
    if cached := _cached_response(request, "interaction-series", cache_parameters):
        return cached
    if snapshot := _series_snapshot(request, "interaction-series", series_scope):
        return _store_response(
            request,
            "interaction-series",
            cache_parameters,
            _page_series_snapshot(
                snapshot,
                kind="interaction-series",
                scope=series_scope,
                field="series",
                limit=limit,
                cursor=cursor,
                clip_domain_periods=True,
            ),
        )
    zone = _zone(timezone)
    message_filters = _message_range_filters(
        from_ms=from_ms,
        to_ms=to_ms,
        conversation_id=None,
    )
    series_counts: dict[str, Counter[str]] = defaultdict(Counter)
    domain_counts: Counter[str] = Counter()
    domain_period_counts: dict[str, Counter[str]] = defaultdict(Counter)
    totals: Counter[str] = Counter()
    type_expression = _attachment_type_expression()

    with session_for(request) as session:
        message_rows = session.execute(
            select(
                Message.timestamp_ms,
                Message.message_kind,
                Message.call_duration_seconds,
            ).where(*message_filters)
        ).yield_per(10_000)
        for timestamp_ms, message_kind, call_duration_seconds in message_rows:
            period = _period(_local_datetime(timestamp_ms, zone), bucket)
            period_counts = series_counts[period]
            if message_kind == "call" or call_duration_seconds is not None:
                duration = int(call_duration_seconds or 0)
                period_counts["calls"] += 1
                period_counts["call_duration_seconds"] += duration
                totals["calls"] += 1
                totals["call_duration_seconds"] += duration

        link_rows = session.execute(
            select(Message.timestamp_ms, Message.content_display, Message.extra_json).where(
                *message_filters,
                _message_has_link_candidate(),
            )
        ).yield_per(5_000)
        for timestamp_ms, content, extra in link_rows:
            link_period = _period(_local_datetime(timestamp_ms, zone), bucket)
            period_counts = series_counts[link_period]
            for hostname in _message_link_hostnames(content, extra):
                period_counts["link_occurrences"] += 1
                totals["link_occurrences"] += 1
                domain_counts[hostname] += 1
                domain_period_counts[hostname][link_period] += 1

        reaction_timestamp = func.coalesce(MessageReaction.timestamp_ms, Message.timestamp_ms)
        reaction_filters: list[Any] = [Message.deleted_at.is_(None)]
        if from_ms is not None:
            reaction_filters.append(reaction_timestamp >= from_ms)
        if to_ms is not None:
            reaction_filters.append(reaction_timestamp <= to_ms)
        reaction_rows = session.execute(
            select(MessageReaction.timestamp_ms, Message.timestamp_ms)
            .join(Message, Message.id == MessageReaction.message_id)
            .where(*reaction_filters)
            .order_by(reaction_timestamp, MessageReaction.id)
        ).yield_per(10_000)
        for explicit_timestamp_ms, message_timestamp_ms in reaction_rows:
            effective_timestamp_ms = (
                explicit_timestamp_ms if explicit_timestamp_ms is not None else message_timestamp_ms
            )
            period = _period(_local_datetime(effective_timestamp_ms, zone), bucket)
            series_counts[period]["reactions"] += 1
            totals["reactions"] += 1
            timestamp_basis = (
                "reaction_explicit_timestamp"
                if explicit_timestamp_ms is not None
                else "reaction_message_timestamp_fallback"
            )
            series_counts[period][timestamp_basis] += 1
            totals[timestamp_basis] += 1

        audio_base_filters: list[Any] = [
            AttachmentOccurrence.deleted_at.is_(None),
            type_expression == "audio",
        ]
        audio_unknown_date = (
            session.scalar(
                select(func.count(AttachmentOccurrence.id)).where(
                    *audio_base_filters,
                    AttachmentOccurrence.timestamp_ms.is_(None),
                )
            )
            or 0
        )
        audio_timed_filters = [
            *audio_base_filters,
            AttachmentOccurrence.timestamp_ms.is_not(None),
        ]
        if from_ms is not None:
            audio_timed_filters.append(AttachmentOccurrence.timestamp_ms >= from_ms)
        if to_ms is not None:
            audio_timed_filters.append(AttachmentOccurrence.timestamp_ms <= to_ms)
        for (timestamp_ms,) in session.execute(
            select(AttachmentOccurrence.timestamp_ms)
            .where(*audio_timed_filters)
            .order_by(AttachmentOccurrence.timestamp_ms, AttachmentOccurrence.id)
        ).yield_per(5_000):
            assert timestamp_ms is not None
            period = _period(_local_datetime(timestamp_ms, zone), bucket)
            series_counts[period]["audio_attachments"] += 1
            totals["audio_attachments"] += 1

    top_domains = domain_counts.most_common(domain_limit)
    series = [
        {
            "period": period,
            "calls": counts["calls"],
            "call_duration_seconds": counts["call_duration_seconds"],
            "reactions": counts["reactions"],
            "reaction_explicit_timestamp": counts["reaction_explicit_timestamp"],
            "reaction_message_timestamp_fallback": counts["reaction_message_timestamp_fallback"],
            "audio_attachments": counts["audio_attachments"],
            "link_occurrences": counts["link_occurrences"],
        }
        for period, counts in series_counts.items()
    ]
    full_result = {
        "bucket": bucket,
        "timezone": timezone,
        "series": series,
        "domains": [
            {
                "domain": domain,
                "occurrences": occurrences,
                "periods": [
                    {"period": period, "occurrences": count}
                    for period, count in sorted(domain_period_counts[domain].items())
                ],
            }
            for domain, occurrences in top_domains
        ],
        "summary": {
            "calls": totals["calls"],
            "call_duration_seconds": totals["call_duration_seconds"],
            "reactions": totals["reactions"],
            "reaction_explicit_timestamp": totals["reaction_explicit_timestamp"],
            "reaction_message_timestamp_fallback": totals["reaction_message_timestamp_fallback"],
            "audio_attachments": totals["audio_attachments"],
            "audio_unknown_date": int(audio_unknown_date),
            "link_occurrences": totals["link_occurrences"],
            "domain_count": len(domain_counts),
        },
        "provenance": {
            "calls": "Call records use their active source-message timestamps.",
            "reactions": (
                "Reaction source timestamps are used when present; otherwise the active source "
                "message timestamp is an explicitly counted fallback."
            ),
            "audio": (
                "Audio uses active attachment-occurrence timestamps. Unknown-date audio is "
                "reported separately as an archive-wide count and never assigned to a time "
                "bucket; it cannot be tested against a selected date range."
            ),
            "links": (
                "URL occurrences found in retained message display text and structured metadata "
                "use the active source-message timestamp."
            ),
        },
    }
    _store_series_snapshot(request, "interaction-series", series_scope, full_result)
    result = _page_series_snapshot(
        full_result,
        kind="interaction-series",
        scope=series_scope,
        field="series",
        limit=limit,
        cursor=cursor,
        clip_domain_periods=True,
    )
    return _store_response(request, "interaction-series", cache_parameters, result)


@analytics_router.get("/overview")
def analytics_overview(
    request: Request,
    timezone: str = Query(default="America/New_York", max_length=100),
) -> dict[str, Any]:
    cache_parameters: dict[str, object] = {"timezone": timezone}
    if cached := _cached_response(request, "overview", cache_parameters):
        return cached
    zone = _zone(timezone)
    with session_for(request) as session:
        message_count, sent_count, received_count, first_ms, last_ms, conversations = (
            session.execute(
                select(
                    func.count(Message.pk),
                    func.sum(case((Message.sender_is_owner.is_(True), 1), else_=0)),
                    func.sum(case((Message.sender_is_owner.is_(False), 1), else_=0)),
                    func.min(Message.timestamp_ms),
                    func.max(Message.timestamp_ms),
                    func.count(func.distinct(Message.conversation_id)),
                ).where(Message.deleted_at.is_(None))
            ).one()
        )
        reaction_count = (
            session.scalar(
                select(func.count(MessageReaction.id))
                .join(Message, Message.id == MessageReaction.message_id)
                .where(Message.deleted_at.is_(None))
            )
            or 0
        )
        reaction_types: dict[str, int] = {
            reaction: int(count)
            for reaction, count in session.execute(
                select(MessageReaction.reaction_display, func.count(MessageReaction.id))
                .join(Message, Message.id == MessageReaction.message_id)
                .where(Message.deleted_at.is_(None))
                .group_by(MessageReaction.reaction_display)
                .order_by(MessageReaction.reaction_display)
            )
        }
        call_count, missed_calls, call_seconds = session.execute(
            select(
                func.count(Message.pk),
                func.sum(case((Message.is_missed_call.is_(True), 1), else_=0)),
                func.coalesce(func.sum(Message.call_duration_seconds), 0),
            ).where(
                Message.deleted_at.is_(None),
                or_(Message.message_kind == "call", Message.call_duration_seconds.is_not(None)),
            )
        ).one()
        attachment_count, attachment_bytes = session.execute(
            select(
                func.count(AttachmentOccurrence.id),
                func.coalesce(func.sum(AttachmentOccurrence.size_bytes), 0),
            ).where(AttachmentOccurrence.deleted_at.is_(None))
        ).one()
        type_expression = _attachment_type_expression()
        audio_count = (
            session.scalar(
                select(func.count(AttachmentOccurrence.id)).where(
                    AttachmentOccurrence.deleted_at.is_(None), type_expression == "audio"
                )
            )
            or 0
        )
        activity_count = (
            session.scalar(
                select(func.count(ActivityRecord.pk)).where(ActivityRecord.deleted_at.is_(None))
            )
            or 0
        )
        coverage = coverage_payload(
            session,
            "messages",
            label="Messages",
            unavailable_description="Message history is unavailable in this export.",
            actual_count=int(message_count or 0),
            status_with_data="available",
        )
    calendar_days = (
        (_local_datetime(last_ms, zone).date() - _local_datetime(first_ms, zone).date()).days + 1
        if first_ms is not None and last_ms is not None
        else None
    )
    result = {
        "coverage": coverage,
        "timezone": timezone,
        "messages": {
            "total": int(message_count or 0),
            "sent": int(sent_count or 0),
            "received": int(received_count or 0),
            "first_timestamp_ms": first_ms,
            "last_timestamp_ms": last_ms,
            "calendar_days": calendar_days,
            "average_per_calendar_day": (
                float(message_count or 0) / calendar_days if calendar_days else None
            ),
            "active_conversations": int(conversations or 0),
        },
        "interactions": {
            "message_reactions": reaction_count,
            "reaction_types": reaction_types,
            "calls": int(call_count or 0),
            "missed_calls": int(missed_calls or 0),
            "call_duration_seconds": int(call_seconds or 0),
            "audio_attachments": audio_count,
        },
        "attachments": {
            "occurrences": int(attachment_count or 0),
            "occurrence_bytes": int(attachment_bytes or 0),
        },
        "activity_records": activity_count,
        "link_domains": _link_domains(request),
        "derived_nlp": _derived_nlp_status(request),
        "provenance": (
            "Counts are computed from active SQLite source-occurrence rows; calendar-day span "
            f"uses local dates in {timezone}."
        ),
    }
    return _store_response(request, "overview", cache_parameters, result)


@analytics_router.get("/messages")
def message_series(
    request: Request,
    bucket: TimeBucket = "day",
    timezone: str = Query(default="America/New_York", max_length=100),
    from_ms: int | None = Query(default=None, ge=0),
    to_ms: int | None = Query(default=None, ge=0),
    conversation_id: str | None = Query(default=None, max_length=64),
    limit: int = Query(default=_ANALYTICS_PERIOD_LIMIT, ge=1, le=_ANALYTICS_PERIOD_LIMIT_MAX),
    cursor: str | None = Query(default=None, max_length=2048),
) -> dict[str, Any]:
    series_scope = {
        "bucket": bucket,
        "timezone": timezone,
        "from_ms": from_ms,
        "to_ms": to_ms,
        "conversation_id": conversation_id,
    }
    cache_parameters = {
        **series_scope,
        "limit": limit,
        "cursor": cursor,
    }
    if cached := _cached_response(request, "messages", cache_parameters):
        return cached
    if snapshot := _series_snapshot(request, "messages", series_scope):
        return _store_response(
            request,
            "messages",
            cache_parameters,
            _page_series_snapshot(
                snapshot,
                kind="messages",
                scope=series_scope,
                field="series",
                limit=limit,
                cursor=cursor,
            ),
        )
    zone = _zone(timezone)
    filters = _message_range_filters(from_ms=from_ms, to_ms=to_ms, conversation_id=conversation_id)
    counts: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"messages": 0, "sent": 0, "received": 0, "conversations": set()}
    )
    with session_for(request) as session:
        rows = session.execute(
            select(Message.timestamp_ms, Message.sender_is_owner, Message.conversation_id).where(
                *filters
            )
        ).yield_per(10_000)
        for timestamp_ms, is_owner, row_conversation_id in rows:
            period = _period(_local_datetime(timestamp_ms, zone), bucket)
            item = counts[period]
            item["messages"] += 1
            item["sent" if is_owner else "received"] += 1
            item["conversations"].add(row_conversation_id)
        total = sum(int(item["messages"]) for item in counts.values())
        coverage = coverage_payload(
            session,
            "messages",
            label="Messages",
            unavailable_description="Message history is unavailable in this export.",
            actual_count=total,
            status_with_data="available",
        )
    all_series = [
        {
            "period": period,
            "messages": item["messages"],
            "sent": item["sent"],
            "received": item["received"],
            "active_conversations": len(item["conversations"]),
        }
        for period, item in sorted(counts.items())
    ]
    full_result = {
        "coverage": coverage,
        "bucket": bucket,
        "timezone": timezone,
        "series": all_series,
        "summary": {
            "messages": total,
            "observed_periods": len(all_series),
            "average_per_observed_period": total / len(all_series) if all_series else None,
        },
        "derived_nlp": _derived_nlp_status(request),
    }
    _store_series_snapshot(request, "messages", series_scope, full_result)
    result = _page_series_snapshot(
        full_result,
        kind="messages",
        scope=series_scope,
        field="series",
        limit=limit,
        cursor=cursor,
    )
    return _store_response(request, "messages", cache_parameters, result)


@analytics_router.get("/activity-patterns")
def activity_patterns(
    request: Request,
    timezone: str = Query(default="America/New_York", max_length=100),
    from_ms: int | None = Query(default=None, ge=0),
    to_ms: int | None = Query(default=None, ge=0),
    limit: int = Query(default=_ANALYTICS_PERIOD_LIMIT, ge=1, le=_ANALYTICS_PERIOD_LIMIT_MAX),
    cursor: str | None = Query(default=None, max_length=2048),
) -> dict[str, Any]:
    series_scope = {
        "timezone": timezone,
        "from_ms": from_ms,
        "to_ms": to_ms,
    }
    cache_parameters = {
        **series_scope,
        "limit": limit,
        "cursor": cursor,
    }
    if cached := _cached_response(request, "activity-patterns", cache_parameters):
        return cached
    if snapshot := _series_snapshot(request, "activity-patterns", series_scope):
        return _store_response(
            request,
            "activity-patterns",
            cache_parameters,
            _page_series_snapshot(
                snapshot,
                kind="activity-patterns",
                scope=series_scope,
                field="heatmap",
                limit=limit,
                cursor=cursor,
                coordinate="date",
            ),
        )
    zone = _zone(timezone)
    if from_ms is not None and to_ms is not None and from_ms > to_ms:
        raise HTTPException(status_code=422, detail="Invalid analytics date range")
    message_filters = _message_range_filters(from_ms=from_ms, to_ms=to_ms, conversation_id=None)
    activity_base_filters: list[Any] = [ActivityRecord.deleted_at.is_(None)]
    activity_filters: list[Any] = [
        *activity_base_filters,
        ActivityRecord.timestamp_ms.is_not(None),
    ]
    if from_ms is not None:
        activity_filters.append(ActivityRecord.timestamp_ms >= from_ms)
    if to_ms is not None:
        activity_filters.append(ActivityRecord.timestamp_ms <= to_ms)
    heatmap: dict[str, dict[str, int]] = defaultdict(
        lambda: {"messages": 0, "activities": 0, "total": 0}
    )
    message_hours: Counter[int] = Counter()
    sent_hours: Counter[int] = Counter()
    received_hours: Counter[int] = Counter()
    activity_hours: Counter[int] = Counter()
    message_weekdays: Counter[int] = Counter()
    activity_weekdays: Counter[int] = Counter()
    with session_for(request) as session:
        active_activity_records = int(
            session.scalar(select(func.count(ActivityRecord.pk)).where(*activity_base_filters)) or 0
        )
        unknown_date_activity_records = int(
            session.scalar(
                select(func.count(ActivityRecord.pk)).where(
                    *activity_base_filters,
                    ActivityRecord.timestamp_ms.is_(None),
                )
            )
            or 0
        )
        for timestamp_ms, is_owner in session.execute(
            select(Message.timestamp_ms, Message.sender_is_owner).where(*message_filters)
        ).yield_per(10_000):
            local = _local_datetime(timestamp_ms, zone)
            date_key = local.date().isoformat()
            heatmap[date_key]["messages"] += 1
            heatmap[date_key]["total"] += 1
            message_hours[local.hour] += 1
            (sent_hours if is_owner else received_hours)[local.hour] += 1
            message_weekdays[local.weekday()] += 1
        for (timestamp_ms,) in session.execute(
            select(ActivityRecord.timestamp_ms).where(*activity_filters)
        ).yield_per(5_000):
            assert timestamp_ms is not None
            local = _local_datetime(timestamp_ms, zone)
            date_key = local.date().isoformat()
            heatmap[date_key]["activities"] += 1
            heatmap[date_key]["total"] += 1
            activity_hours[local.hour] += 1
            activity_weekdays[local.weekday()] += 1
    messages_in_scope = sum(value["messages"] for value in heatmap.values())
    dated_activity_records_in_scope = sum(value["activities"] for value in heatmap.values())
    full_result = {
        "timezone": timezone,
        "heatmap": [{"date": date, **counts} for date, counts in heatmap.items()],
        "hours": [
            {
                "hour": hour,
                "messages": message_hours[hour],
                "sent": sent_hours[hour],
                "received": received_hours[hour],
                "activities": activity_hours[hour],
            }
            for hour in range(24)
        ],
        "weekdays": [
            {
                "weekday": weekday,
                "label": _WEEKDAY_NAMES[weekday],
                "messages": message_weekdays[weekday],
                "activities": activity_weekdays[weekday],
            }
            for weekday in range(7)
        ],
        "streaks": _streaks(heatmap),
        "peak_days": [
            {"date": day, **counts}
            for day, counts in sorted(
                heatmap.items(),
                key=lambda item: (-item[1]["total"], item[0]),
            )[:10]
        ],
        "summary": {
            "messages_in_scope": messages_in_scope,
            "dated_activity_records_in_scope": dated_activity_records_in_scope,
            "active_activity_records_archive_wide": active_activity_records,
            "unknown_date_activity_records": unknown_date_activity_records,
            "unknown_date_scope": "archive_wide",
            "unknown_date_included_in_time_buckets": False,
            "date_filter_active": from_ms is not None or to_ms is not None,
        },
        "range_provenance": (
            "Timestamp bounds are inclusive at millisecond precision. Unknown-date Facebook "
            "activity is reported archive-wide but cannot be tested against a selected range "
            "and is never assigned to a time bucket."
        ),
        "derived_nlp": _derived_nlp_status(request),
    }
    _store_series_snapshot(request, "activity-patterns", series_scope, full_result)
    result = _page_series_snapshot(
        full_result,
        kind="activity-patterns",
        scope=series_scope,
        field="heatmap",
        limit=limit,
        cursor=cursor,
        coordinate="date",
    )
    return _store_response(request, "activity-patterns", cache_parameters, result)


def _nlp_score(task: str, probabilities: dict[str, float]) -> tuple[float | None, str]:
    normalized = {key.casefold(): value for key, value in probabilities.items()}
    if task == "sentiment":
        positive = normalized.get("positive", normalized.get("label_2"))
        negative = normalized.get("negative", normalized.get("label_0"))
        if positive is None or negative is None:
            return None, "Positive probability minus negative probability; labels unavailable."
        return positive - negative, "Mean positive probability minus mean negative probability."
    if task == "toxicity":
        candidates = [
            value
            for label, value in normalized.items()
            if label not in {"non-toxic", "non_toxic", "neutral", "label_0"}
        ]
        return (
            max(candidates) if candidates else None,
            "Highest mean modeled toxicity-dimension score in the bucket.",
        )
    return None, "Emotion is represented by its dominant predicted label and full mean vector."


def _serialize_nlp_aggregates(
    aggregates: dict[tuple[str, str, str], dict[str, Any]], *, coordinate_name: str
) -> dict[str, list[dict[str, Any]]]:
    series_by_task: dict[str, list[dict[str, Any]]] = {
        "sentiment": [],
        "emotion": [],
        "toxicity": [],
    }
    for (task, coordinate, output_direction), aggregate in sorted(aggregates.items()):
        sample_count = int(aggregate["sample_count"])
        probability_sums = cast(dict[str, float], aggregate["probability_sums"])
        mean_probabilities = {
            label: value / sample_count for label, value in sorted(probability_sums.items())
        }
        score, score_definition = _nlp_score(task, mean_probabilities)
        predicted_counts = cast(Counter[str], aggregate["predicted_counts"])
        dominant_label = (
            sorted(predicted_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
            if predicted_counts
            else None
        )
        series_by_task.setdefault(task, []).append(
            {
                coordinate_name: coordinate,
                "direction": output_direction,
                "sample_count": sample_count,
                "mean_confidence": float(aggregate["confidence_sum"]) / sample_count,
                "score": score,
                "score_definition": score_definition,
                "dominant_label": dominant_label,
                "predicted_label_counts": dict(sorted(predicted_counts.items())),
                "mean_probabilities": mean_probabilities,
            }
        )
    return series_by_task


def _nlp_rollup_filters(
    *,
    direction: Literal["all", "owner", "other", "activity", "audio", "attachment"],
    from_ms: int | None,
    to_ms: int | None,
) -> list[Any]:
    filters: list[Any] = [NlpHourlyRollup.active.is_(True)]
    if direction != "all":
        filters.append(NlpHourlyRollup.direction == direction)
    if from_ms is not None:
        filters.append(NlpHourlyRollup.hour_start_ms + 3_600_000 > from_ms)
    if to_ms is not None:
        filters.append(NlpHourlyRollup.hour_start_ms <= to_ms)
    return filters


def _load_nlp_rollup_invariants(
    session: Session,
    *,
    direction: Literal["all", "owner", "other", "activity", "audio", "attachment"],
    from_ms: int | None,
    to_ms: int | None,
) -> dict[str, Any]:
    rollup_filters = _nlp_rollup_filters(
        direction=direction,
        from_ms=from_ms,
        to_ms=to_ms,
    )
    models: dict[str, dict[str, str]] = {}
    source_sample_count = int(
        session.scalar(
            select(func.coalesce(func.sum(NlpHourlyRollup.sample_count), 0)).where(*rollup_filters)
        )
        or 0
    )
    for task, model_name, model_revision, configuration_sha256 in session.execute(
        select(
            NlpHourlyRollup.task,
            NlpHourlyRollup.model_name,
            NlpHourlyRollup.model_revision,
            NlpHourlyRollup.configuration_sha256,
        )
        .where(*rollup_filters)
        .distinct()
        .order_by(
            NlpHourlyRollup.task,
            NlpHourlyRollup.model_name,
            NlpHourlyRollup.model_revision,
            NlpHourlyRollup.configuration_sha256,
        )
    ):
        models[task] = {
            "model_name": model_name,
            "model_revision": model_revision,
            "configuration_sha256": configuration_sha256,
        }
    return {"source_sample_count": source_sample_count, "models": models}


def _nlp_rollup_invariants(
    request: Request,
    session: Session,
    *,
    direction: Literal["all", "owner", "other", "activity", "audio", "attachment"],
    from_ms: int | None,
    to_ms: int | None,
) -> dict[str, Any]:
    return _acceptance_invariant_snapshot(
        request,
        "nlp-rollup",
        {"direction": direction, "from_ms": from_ms, "to_ms": to_ms},
        lambda: _load_nlp_rollup_invariants(
            session,
            direction=direction,
            from_ms=from_ms,
            to_ms=to_ms,
        ),
    )


def _conflict_rollup_filters(*, run_id: str, from_ms: int | None, to_ms: int | None) -> list[Any]:
    filters: list[Any] = [
        ConflictHourlyRollup.active.is_(True),
        ConflictHourlyRollup.run_id == run_id,
    ]
    if from_ms is not None:
        filters.append(ConflictHourlyRollup.hour_start_ms + 3_600_000 > from_ms)
    if to_ms is not None:
        filters.append(ConflictHourlyRollup.hour_start_ms <= to_ms)
    return filters


def _load_conflict_rollup_invariants(
    session: Session,
    *,
    run_id: str,
    from_ms: int | None,
    to_ms: int | None,
) -> dict[str, int]:
    sample_count, direct_count = session.execute(
        select(
            func.coalesce(func.sum(ConflictHourlyRollup.sample_count), 0),
            func.coalesce(func.sum(ConflictHourlyRollup.direct_conflict_count), 0),
        ).where(
            *_conflict_rollup_filters(
                run_id=run_id,
                from_ms=from_ms,
                to_ms=to_ms,
            )
        )
    ).one()
    return {"sample_count": int(sample_count), "direct_count": int(direct_count)}


def _conflict_rollup_invariants(
    request: Request,
    session: Session,
    *,
    run_id: str,
    from_ms: int | None,
    to_ms: int | None,
) -> dict[str, int]:
    return _acceptance_invariant_snapshot(
        request,
        "conflict-rollup",
        {"run_id": run_id, "from_ms": from_ms, "to_ms": to_ms},
        lambda: _load_conflict_rollup_invariants(
            session,
            run_id=run_id,
            from_ms=from_ms,
            to_ms=to_ms,
        ),
    )


@analytics_router.get("/nlp-trends")
def nlp_trends(
    request: Request,
    bucket: NlpTimeBucket = "month",
    timezone: str = Query(default="America/New_York", max_length=100),
    direction: Literal["all", "owner", "other", "activity", "audio", "attachment"] = "all",
    from_ms: int | None = Query(default=None, ge=0),
    to_ms: int | None = Query(default=None, ge=0),
    limit: int = Query(default=_NLP_PERIOD_LIMIT, ge=1, le=_NLP_PERIOD_LIMIT_MAX),
    cursor: str | None = Query(default=None, max_length=2048),
) -> dict[str, Any]:
    if from_ms is not None and to_ms is not None and from_ms > to_ms:
        raise HTTPException(status_code=422, detail="Invalid analytics date range")
    zone = _zone(timezone)
    rollup_filters = _nlp_rollup_filters(
        direction=direction,
        from_ms=from_ms,
        to_ms=to_ms,
    )
    aggregates: dict[tuple[str, str, str], dict[str, Any]] = {}
    hour_patterns: dict[tuple[str, str, str], dict[str, Any]] = {}
    models: dict[str, dict[str, str]] = {}
    source_sample_count = 0
    page_sample_count = 0
    non_whole_hour_offset = False
    with session_for(request) as session:
        invariants = _nlp_rollup_invariants(
            request,
            session,
            direction=direction,
            from_ms=from_ms,
            to_ms=to_ms,
        )
        models = cast(dict[str, dict[str, str]], invariants["models"])
        source_sample_count = int(invariants["source_sample_count"])
        start_ms, end_ms, page = _rollup_page_bounds(
            session,
            hour_column=NlpHourlyRollup.hour_start_ms,
            filters=rollup_filters,
            kind="nlp-trends",
            scope={
                "bucket": bucket,
                "timezone": timezone,
                "direction": direction,
                "from_ms": from_ms,
                "to_ms": to_ms,
            },
            zone=zone,
            bucket=bucket,
            limit=limit,
            cursor=cursor,
        )
        statement = select(NlpHourlyRollup).where(*rollup_filters)
        if start_ms is None or end_ms is None:
            statement = statement.where(NlpHourlyRollup.hour_start_ms < 0)
        else:
            statement = statement.where(
                NlpHourlyRollup.hour_start_ms >= start_ms,
                NlpHourlyRollup.hour_start_ms < end_ms,
            )
        statement = statement.order_by(
            NlpHourlyRollup.hour_start_ms,
            NlpHourlyRollup.task,
            NlpHourlyRollup.direction,
            NlpHourlyRollup.predicted_label,
        )
        rows = session.scalars(statement).yield_per(5_000)
        for row in rows:
            local = _local_datetime(row.hour_start_ms, zone)
            offset = local.utcoffset()
            non_whole_hour_offset = non_whole_hour_offset or bool(
                offset and int(offset.total_seconds()) % 3_600 != 0
            )
            period = _nlp_period(local, bucket)
            page_sample_count += row.sample_count
            output_directions = (row.direction, "all") if direction == "all" else (row.direction,)
            for output_direction in output_directions:
                for target, coordinate in (
                    (aggregates, period),
                    (hour_patterns, f"{local.hour:02d}"),
                ):
                    key = (row.task, coordinate, output_direction)
                    aggregate = target.setdefault(
                        key,
                        {
                            "sample_count": 0,
                            "confidence_sum": 0.0,
                            "probability_sums": defaultdict(float),
                            "predicted_counts": Counter(),
                        },
                    )
                    aggregate["sample_count"] += row.sample_count
                    aggregate["confidence_sum"] += row.confidence_sum
                    aggregate["predicted_counts"][row.predicted_label] += row.sample_count
                    labels = row.probability_sums_json.get("labels", {})
                    if isinstance(labels, dict):
                        for label, value in labels.items():
                            if isinstance(label, str) and isinstance(value, (int, float)):
                                aggregate["probability_sums"][label] += float(value)
    serialized_series = _serialize_nlp_aggregates(aggregates, coordinate_name="period")
    page["returned"] = len(
        {str(row["period"]) for task_rows in serialized_series.values() for row in task_rows}
    )
    warnings = [
        "All language signals are fallible local model estimates with sample sizes shown.",
        "Emotion labels do not establish a person's internal state.",
        "Toxicity models can be biased by dialect, identity terms, quotation, and banter.",
        "No person, conversation, or relationship is ranked by these values.",
    ]
    if from_ms is not None or to_ms is not None:
        warnings.append(
            "Date filters operate on UTC-hour rollups: every whole rollup hour overlapping the "
            "requested interval is included, so partial boundary hours are approximate."
        )
    if non_whole_hour_offset:
        warnings.append(
            "This timezone has a non-whole-hour UTC offset; UTC rollup hours can straddle local "
            "hour-of-day buckets."
        )
    return {
        "timezone": timezone,
        "bucket": bucket,
        "direction": direction,
        "source_sample_count": source_sample_count,
        "page_sample_count": page_sample_count,
        "models": models,
        "series": serialized_series,
        "hour_of_day": _serialize_nlp_aggregates(hour_patterns, coordinate_name="hour"),
        "hour_of_day_scope": "current_page",
        "page": page,
        "availability": _derived_nlp_status(request),
        "rollup_provenance": {
            "source_resolution": "UTC hour",
            "range_filter_semantics": "whole UTC hours overlapping the requested interval",
            "local_bucket_label": "local time at the UTC rollup hour start",
            "partial_boundary_hours_included": from_ms is not None or to_ms is not None,
            "non_whole_hour_timezone_offset_observed": non_whole_hour_offset,
        },
        "warnings": warnings,
    }


@analytics_router.get("/conflict-rate")
def conflict_rate(
    request: Request,
    bucket: NlpTimeBucket = "month",
    timezone: str = Query(default="America/New_York", max_length=100),
    from_ms: int | None = Query(default=None, ge=0),
    to_ms: int | None = Query(default=None, ge=0),
    limit: int = Query(default=_ANALYTICS_PERIOD_LIMIT, ge=1, le=_ANALYTICS_PERIOD_LIMIT_MAX),
    cursor: str | None = Query(default=None, max_length=2048),
) -> dict[str, Any]:
    """Return the gated model-estimated direct-conflict-window rate over time.

    This endpoint intentionally has no person, conversation, participant, or
    relationship dimensions. It cannot be used as a relationship leaderboard.
    """

    if from_ms is not None and to_ms is not None and from_ms > to_ms:
        raise HTTPException(status_code=422, detail="Invalid analytics date range")
    zone = _zone(timezone)
    availability = _conflict_aggregate_status(request)
    base: dict[str, Any] = {
        "available": availability["available"],
        "status": availability["status"],
        "reason": availability["reason"],
        "timezone": timezone,
        "bucket": bucket,
        "series": [],
        "page": {"limit": limit, "returned": 0, "total": 0, "next_cursor": None},
        "summary": {
            "sample_count": 0,
            "direct_conflict_window_count": 0,
            "direct_conflict_window_rate": None,
        },
        "warnings": [
            "This is a model-estimated share of overlapping text windows, not a count "
            "of arguments.",
            "Sample counts are shown because small buckets are especially uncertain.",
            "Conflict scores are ranking features, not probabilities.",
            "No person, conversation, or relationship is ranked by these values.",
        ],
        "relationship_ranking_enabled": False,
        "relationship_dimensions": [],
        "provenance": {
            "rollup_resolution": "UTC hour",
            "range_filter_semantics": "whole UTC hours overlapping the requested interval",
            "local_bucket_label": "local time at the UTC rollup hour start",
            "partial_boundary_hours_included": from_ms is not None or to_ms is not None,
        },
    }
    run = availability.get("run")
    if not availability["available"] or not isinstance(run, dict):
        return base
    run_id = str(run["id"])
    rollup_filters = _conflict_rollup_filters(
        run_id=run_id,
        from_ms=from_ms,
        to_ms=to_ms,
    )
    sample_count = 0
    direct_count = 0
    buckets: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    non_whole_hour_offset = False
    with session_for(request) as session:
        invariants = _conflict_rollup_invariants(
            request,
            session,
            run_id=run_id,
            from_ms=from_ms,
            to_ms=to_ms,
        )
        sample_count = invariants["sample_count"]
        direct_count = invariants["direct_count"]
        start_ms, end_ms, page = _rollup_page_bounds(
            session,
            hour_column=ConflictHourlyRollup.hour_start_ms,
            filters=rollup_filters,
            kind="conflict-rate",
            scope={
                "bucket": bucket,
                "timezone": timezone,
                "from_ms": from_ms,
                "to_ms": to_ms,
                "run_id": run_id,
            },
            zone=zone,
            bucket=bucket,
            limit=limit,
            cursor=cursor,
        )
        statement = select(ConflictHourlyRollup).where(*rollup_filters)
        if start_ms is None or end_ms is None:
            statement = statement.where(ConflictHourlyRollup.hour_start_ms < 0)
        else:
            statement = statement.where(
                ConflictHourlyRollup.hour_start_ms >= start_ms,
                ConflictHourlyRollup.hour_start_ms < end_ms,
            )
        statement = statement.order_by(ConflictHourlyRollup.hour_start_ms)
        for row in session.scalars(statement).yield_per(5_000):
            local = _local_datetime(row.hour_start_ms, zone)
            offset = local.utcoffset()
            non_whole_hour_offset = non_whole_hour_offset or bool(
                offset and int(offset.total_seconds()) % 3_600 != 0
            )
            period = _nlp_period(local, bucket)
            buckets[period][0] += int(row.sample_count)
            buckets[period][1] += int(row.direct_conflict_count)
    base["series"] = [
        {
            "period": period,
            "sample_count": values[0],
            "direct_conflict_window_count": values[1],
            "direct_conflict_window_rate": values[1] / values[0] if values[0] else None,
            "low_sample_warning": values[0] < 30,
        }
        for period, values in sorted(buckets.items())
    ]
    page["returned"] = len(base["series"])
    base["page"] = page
    base["summary"] = {
        "sample_count": int(sample_count),
        "direct_conflict_window_count": int(direct_count),
        "direct_conflict_window_rate": direct_count / sample_count if sample_count else None,
    }
    base["provenance"] = {
        "run_id": run["id"],
        "pipeline_version": run["pipeline_version"],
        "model_name": run["model_name"],
        "model_revision": run["model_revision"],
        "calibration_id": run["calibration_id"],
        "semantic_config_id": run["semantic_config_id"],
        "corpus_revision": run["corpus_revision"],
        "source_revision": run["source_revision"],
        "configuration_sha256": run["configuration_sha256"],
        "classification_rule": "top NLI entailment-logit label equals direct conflict",
        "rollup_resolution": "UTC hour",
        "range_filter_semantics": "whole UTC hours overlapping the requested interval",
        "local_bucket_label": "local time at the UTC rollup hour start",
        "partial_boundary_hours_included": from_ms is not None or to_ms is not None,
        "non_whole_hour_timezone_offset_observed": non_whole_hour_offset,
    }
    if from_ms is not None or to_ms is not None:
        base["warnings"].append(
            "Date filters include every whole UTC rollup hour overlapping the requested interval; "
            "partial boundary hours are approximate."
        )
    if non_whole_hour_offset:
        base["warnings"].append(
            "This timezone has a non-whole-hour UTC offset, so UTC rollup hours can straddle "
            "local time buckets."
        )
    return base


@analytics_router.get("/interaction-patterns")
def interaction_patterns(
    request: Request,
    bucket: TimeBucket = "month",
    timezone: str = Query(default="America/New_York", max_length=100),
    from_ms: int | None = Query(default=None, ge=0),
    to_ms: int | None = Query(default=None, ge=0),
    limit: int = Query(default=_ANALYTICS_PERIOD_LIMIT, ge=1, le=_ANALYTICS_PERIOD_LIMIT_MAX),
    cursor: str | None = Query(default=None, max_length=2048),
) -> dict[str, Any]:
    """Summarize direction changes without assigning relationship quality."""

    series_scope = {
        "bucket": bucket,
        "timezone": timezone,
        "from_ms": from_ms,
        "to_ms": to_ms,
    }
    cache_parameters = {
        **series_scope,
        "limit": limit,
        "cursor": cursor,
    }
    if cached := _cached_response(request, "interaction-patterns", cache_parameters):
        return cached
    if snapshot := _series_snapshot(request, "interaction-patterns", series_scope):
        return _store_response(
            request,
            "interaction-patterns",
            cache_parameters,
            _page_series_snapshot(
                snapshot,
                kind="interaction-patterns",
                scope=series_scope,
                field="active_participants",
                limit=limit,
                cursor=cursor,
            ),
        )
    zone = _zone(timezone)
    filters = _message_range_filters(from_ms=from_ms, to_ms=to_ms, conversation_id=None)
    owner_response_ms: list[int] = []
    other_response_ms: list[int] = []
    initiations: Counter[str] = Counter()
    participants_by_period: dict[str, set[str]] = defaultdict(set)
    conversations_by_period: dict[str, set[str]] = defaultdict(set)
    previous_conversation: str | None = None
    previous_timestamp: int | None = None
    previous_owner: bool | None = None
    # The active-row index is selective for ordinary list queries, but choosing it here forces
    # SQLite to sort the entire corpus before Python can consume the first row.  Unary `+` keeps
    # the predicate exact while making that index ineligible, so the existing conversation/time
    # index supplies the required streaming order.  The final PK tie-break remains deterministic.
    ordered_filters = [text("+messages.deleted_at IS NULL"), *filters[1:]]
    with session_for(request) as session:
        rows = session.execute(
            select(
                Message.conversation_id,
                Message.timestamp_ms,
                Message.sender_is_owner,
                Message.sender_search,
            )
            .where(*ordered_filters)
            .order_by(Message.conversation_id, Message.timestamp_ms, Message.pk)
        ).yield_per(10_000)
        for conversation_id, timestamp_ms, is_owner, sender_search in rows:
            local_period = _period(_local_datetime(timestamp_ms, zone), bucket)
            participants_by_period[local_period].add(sender_search)
            conversations_by_period[local_period].add(conversation_id)
            if conversation_id != previous_conversation:
                initiations["owner" if is_owner else "other"] += 1
                previous_conversation = conversation_id
                previous_timestamp = timestamp_ms
                previous_owner = bool(is_owner)
                continue
            if previous_timestamp is not None and previous_owner != bool(is_owner):
                elapsed = max(0, timestamp_ms - previous_timestamp)
                (owner_response_ms if is_owner else other_response_ms).append(elapsed)
            previous_timestamp = timestamp_ms
            previous_owner = bool(is_owner)
    active_participants = [
        {
            "period": period,
            "participants": len(participants_by_period[period]),
            "conversations": len(conversations_by_period[period]),
        }
        for period in participants_by_period
    ]
    full_result = {
        "timezone": timezone,
        "bucket": bucket,
        "responses": {
            "owner_replied": _response_summary(owner_response_ms),
            "other_replied": _response_summary(other_response_ms),
            "definition": (
                "Elapsed time between adjacent retained messages when sender direction changes. "
                "Repeated messages from one side do not create a response sample."
            ),
        },
        "initiations": {
            "owner": initiations["owner"],
            "other": initiations["other"],
            "definition": (
                "Direction of the first retained message in each conversation within the "
                "selected scope."
            ),
        },
        "active_participants": active_participants,
        "ranking_policy": (
            "Response intervals are aggregate timing measurements and are not used to rank people "
            "or infer relationship quality."
        ),
    }
    _store_series_snapshot(request, "interaction-patterns", series_scope, full_result)
    result = _page_series_snapshot(
        full_result,
        kind="interaction-patterns",
        scope=series_scope,
        field="active_participants",
        limit=limit,
        cursor=cursor,
    )
    return _store_response(request, "interaction-patterns", cache_parameters, result)


@analytics_router.get("/network")
def conversation_network(
    request: Request,
    limit: int = Query(default=40, ge=1, le=100),
) -> dict[str, Any]:
    """Return a bounded bipartite graph of source conversations and participant labels."""

    with session_for(request) as session:
        conversations = list(
            session.scalars(
                select(Conversation)
                .where(Conversation.deleted_at.is_(None))
                .order_by(Conversation.message_count.desc(), Conversation.id)
                .limit(limit)
            )
        )
        conversation_ids = [row.id for row in conversations]
        participant_rows = list(
            session.execute(
                select(
                    ConversationSegment.conversation_id,
                    Participant.name_display,
                    Participant.name_search,
                    Participant.is_owner,
                )
                .join(
                    SegmentParticipant,
                    SegmentParticipant.segment_id == ConversationSegment.id,
                )
                .join(Participant, Participant.id == SegmentParticipant.participant_id)
                .where(ConversationSegment.conversation_id.in_(conversation_ids))
                .order_by(
                    ConversationSegment.conversation_id,
                    Participant.name_search,
                    Participant.id,
                )
            )
        )
    conversation_nodes = [
        {
            "id": f"conversation:{row.id}",
            "kind": "conversation",
            "label": row.title_display or "Untitled conversation",
            "conversation_type": row.conversation_type,
            "value": row.message_count,
            "href": f"/conversations/{row.id}",
        }
        for row in conversations
    ]
    participant_by_key: dict[str, dict[str, Any]] = {}
    seen_edges: set[tuple[str, str]] = set()
    edges: list[dict[str, Any]] = []
    message_counts = {row.id: row.message_count for row in conversations}
    for conversation_id, display, search, is_owner in participant_rows:
        person_id = f"participant:{stable_id('network-participant', search)}"
        participant = participant_by_key.setdefault(
            search,
            {
                "id": person_id,
                "kind": "participant_label",
                "label": display,
                "is_owner": bool(is_owner),
                "value": 0,
            },
        )
        edge_key = (conversation_id, search)
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        participant["value"] += message_counts.get(conversation_id, 0)
        edges.append(
            {
                "id": stable_id("network-edge", conversation_id, search),
                "source": f"conversation:{conversation_id}",
                "target": person_id,
                "value": message_counts.get(conversation_id, 0),
            }
        )
    return {
        "nodes": conversation_nodes
        + sorted(
            participant_by_key.values(),
            key=lambda item: (-cast(int, item["value"]), str(item["label"])),
        ),
        "edges": edges,
        "conversation_count": len(conversations),
        "participant_label_count": len(participant_by_key),
        "scope": f"Top {len(conversations)} retained conversations by message occurrences.",
        "caveat": (
            "Participant labels are provisional export strings, not Facebook identity IDs. "
            "Node size reflects retained message occurrences, not relationship importance "
            "or quality."
        ),
    }


def _run_topic_cluster_job(
    *,
    layout: Any,
    engine: Engine,
    job_id: str,
    cluster_count: int,
    sample_limit: int,
    dispatch_done: Callable[[], None] | None = None,
) -> None:
    try:
        build_topic_clusters(
            layout,
            engine,
            requested_clusters=cluster_count,
            sample_limit=sample_limit,
            requested_job_id=job_id,
        )
    except (DerivedPublicationBlockedError, DeletionError):
        return
    except BaseException as exc:
        with Session(engine) as session, session.begin():
            session.execute(
                update(Job)
                .where(Job.id == job_id, Job.status != "completed")
                .values(
                    status="failed",
                    stage="Topic clustering failed",
                    error_text=str(exc)[:4_000],
                    finished_at=datetime.now(UTC),
                )
            )
    finally:
        if dispatch_done is not None:
            dispatch_done()


def _topic_job_payload(job: Job | None) -> dict[str, Any] | None:
    if job is None:
        return None
    return {
        "id": job.id,
        "status": job.status,
        "stage": job.stage,
        "progress_current": job.progress_current,
        "progress_total": job.progress_total,
        "error": job.error_text,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
    }


@analytics_router.get("/topics")
def topic_clusters(request: Request) -> dict[str, Any]:
    with session_for(request) as session:
        active_config = session.scalar(
            select(SemanticIndexConfig).where(
                SemanticIndexConfig.active.is_(True),
                SemanticIndexConfig.status == "ready",
            )
        )
        latest_job = session.scalar(
            select(Job).where(Job.kind == "topic_clusters").order_by(desc(Job.created_at)).limit(1)
        )
        completed_job = session.scalar(
            select(Job)
            .where(Job.kind == "topic_clusters", Job.status == "completed")
            .order_by(desc(Job.finished_at), desc(Job.created_at))
            .limit(1)
        )
        result_value = completed_job.counters_json.get("result") if completed_job else None
        result = result_value if isinstance(result_value, dict) else None
        result_config_id = str(result.get("semantic_config_id")) if result else None
        ready = bool(active_config and result and result_config_id == active_config.id)
        representative_ids = [
            str(window_id)
            for cluster in (result.get("clusters", []) if result else [])
            if isinstance(cluster, dict)
            for window_id in cluster.get("representative_window_ids", [])
            if isinstance(window_id, str)
        ]
        windows: dict[str, dict[str, Any]] = {}
        if ready and representative_ids:
            valid_rows = list(
                session.scalars(
                    _valid_window_statement(
                        representative_ids,
                        SearchFilters(),
                        config_hash=result_config_id,
                    )
                )
            )
            rows_by_id = {row.id: row for row in valid_rows}
            serialized = _window_items(
                session,
                rows_by_id,
                [(window_id, 1.0) for window_id in representative_ids],
                score_kind="topic_representative",
            )
            windows = {str(item["window_id"]): item for item in serialized if item.get("window_id")}
    if latest_job and latest_job.status in {"queued", "running"}:
        response_status = "building"
    elif latest_job and latest_job.status == "failed" and not ready:
        response_status = "failed"
    elif latest_job and latest_job.status == "stale":
        response_status = "stale"
    elif ready:
        response_status = "ready"
    elif result:
        response_status = "stale"
    else:
        response_status = "unavailable"
    clusters = []
    if ready and result:
        for value in result.get("clusters", []):
            if not isinstance(value, dict):
                continue
            representative_window_ids = value.get("representative_window_ids", [])
            representatives = [
                windows[window_id]
                for window_id in representative_window_ids
                if isinstance(window_id, str) and window_id in windows
            ]
            clusters.append(
                {
                    "id": value.get("id"),
                    "ordinal": value.get("ordinal"),
                    "sample_windows": value.get("sample_windows"),
                    "sample_share": value.get("sample_share"),
                    "terms": value.get("terms", []),
                    "representatives": representatives,
                }
            )
    return {
        "status": response_status,
        "ready": ready,
        "latest_job": _topic_job_payload(latest_job),
        "semantic_index": (
            {
                "id": active_config.id,
                "model_name": active_config.model_name,
                "model_revision": active_config.model_revision,
                "window_count": active_config.window_count,
            }
            if active_config
            else None
        ),
        "result": (
            {
                "pipeline_version": result.get("pipeline_version"),
                "generated_at": result.get("generated_at"),
                "corpus_window_count": result.get("corpus_window_count"),
                "sample_count": result.get("sample_count"),
                "sampling": result.get("sampling"),
                "warning": result.get("warning"),
                "clusters": clusters,
            }
            if ready and result
            else None
        ),
        "reason": (
            "Build the semantic text index before generating topic clusters."
            if active_config is None
            else "Generate topic clusters for the active semantic index."
            if not result
            else "Topic clusters were generated from an older semantic index; rebuild them."
            if not ready
            else None
        ),
    }


@analytics_router.post("/topics/build", status_code=status.HTTP_202_ACCEPTED)
def create_topic_clusters(
    request: Request,
    payload: TopicClusterBuildRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    request_lock, dispatched_job_ids = _topic_request_state(request)
    reused = False
    job_id: str | None = None
    try:
        with (
            guarded_derived_request(
                request.app.state.layout,
                request.app.state.engine,
                publisher="Topic cluster request",
            ),
            request_lock,
            session_for(request) as session,
            session.begin(),
        ):
            active_config = session.scalar(
                select(SemanticIndexConfig).where(
                    SemanticIndexConfig.active.is_(True),
                    SemanticIndexConfig.status == "ready",
                )
            )
            if active_config is None:
                raise HTTPException(
                    status_code=409,
                    detail="Build the semantic text index before topic clustering",
                )
            requested_checkpoint = {
                "semantic_config_id": active_config.id,
                "sample_limit": payload.sample_limit,
                "requested_clusters": payload.cluster_count,
            }
            active_jobs = list(
                session.scalars(
                    select(Job).where(
                        Job.kind == "topic_clusters",
                        Job.status.in_(("queued", "running")),
                    )
                )
            )
            lock_held = topic_cluster_lock_is_held(request.app.state.layout)
            reusable_id: str | None = None
            for job in active_jobs:
                pristine_queued = (
                    job.status == "queued"
                    and job.stage == "Topic clustering queued"
                    and job.progress_current == 0
                    and job.progress_total == payload.sample_limit
                    and job.checkpoint_json == requested_checkpoint
                    and job.counters_json == {}
                    and job.error_text is None
                    and job.started_at is None
                    and job.finished_at is None
                )
                if pristine_queued and job.id not in dispatched_job_ids:
                    if reusable_id is not None:
                        raise HTTPException(
                            status_code=409,
                            detail=f"Topic-cluster job {job.id} is already active",
                        )
                    reusable_id = job.id
                    continue
                if job.status == "running" and not lock_held:
                    job.status = "failed"
                    job.stage = "Interrupted topic clustering recovered"
                    job.error_text = "Recovered because no process owns the topic-cluster lock"
                    job.finished_at = datetime.now(UTC)
                    continue
                raise HTTPException(
                    status_code=409,
                    detail=f"Topic-cluster job {job.id} is already active",
                )
            if reusable_id is not None:
                job_id = reusable_id
                reused = True
            else:
                job_id = str(uuid.uuid4())
                session.add(
                    Job(
                        id=job_id,
                        kind="topic_clusters",
                        status="queued",
                        stage="Topic clustering queued",
                        progress_current=0,
                        progress_total=payload.sample_limit,
                        checkpoint_json=requested_checkpoint,
                        counters_json={},
                    )
                )
            dispatched_job_ids.add(job_id)
    except BaseException:
        if job_id is not None:
            with request_lock:
                dispatched_job_ids.discard(job_id)
        raise
    assert job_id is not None

    def dispatch_done() -> None:
        with request_lock:
            dispatched_job_ids.discard(job_id)

    try:
        background_tasks.add_task(
            _run_topic_cluster_job,
            layout=request.app.state.layout,
            engine=request.app.state.engine,
            job_id=job_id,
            cluster_count=payload.cluster_count,
            sample_limit=payload.sample_limit,
            dispatch_done=dispatch_done,
        )
    except BaseException:
        dispatch_done()
        raise
    return {
        "job_id": job_id,
        "status": "queued",
        "reused": reused,
        "status_href": "/api/v1/analytics/topics",
    }


@analytics_router.get("/attachments")
def attachment_analytics(
    request: Request,
    bucket: TimeBucket = "month",
    timezone: str = Query(default="America/New_York", max_length=100),
    from_ms: int | None = Query(default=None, ge=0),
    to_ms: int | None = Query(default=None, ge=0),
    limit: int = Query(default=_ANALYTICS_PERIOD_LIMIT, ge=1, le=_ANALYTICS_PERIOD_LIMIT_MAX),
    cursor: str | None = Query(default=None, max_length=2048),
) -> dict[str, Any]:
    series_scope = {
        "bucket": bucket,
        "timezone": timezone,
        "from_ms": from_ms,
        "to_ms": to_ms,
    }
    cache_parameters = {
        **series_scope,
        "limit": limit,
        "cursor": cursor,
    }
    if cached := _cached_response(request, "attachments", cache_parameters):
        return cached
    if snapshot := _series_snapshot(request, "attachments", series_scope):
        return _store_response(
            request,
            "attachments",
            cache_parameters,
            _page_series_snapshot(
                snapshot,
                kind="attachments",
                scope=series_scope,
                field="trends",
                limit=limit,
                cursor=cursor,
            ),
        )
    if from_ms is not None and to_ms is not None and from_ms > to_ms:
        raise HTTPException(status_code=422, detail="Invalid analytics date range")
    zone = _zone(timezone)
    type_expression = _attachment_type_expression()
    unique_key = case(
        (
            AttachmentOccurrence.blob_sha256.is_not(None),
            "blob:" + AttachmentOccurrence.blob_sha256,
        ),
        else_="occurrence:" + AttachmentOccurrence.id,
    )
    base_filters: list[Any] = [AttachmentOccurrence.deleted_at.is_(None)]
    filters = list(base_filters)
    if from_ms is not None:
        filters.append(AttachmentOccurrence.timestamp_ms >= from_ms)
    if to_ms is not None:
        filters.append(AttachmentOccurrence.timestamp_ms <= to_ms)
    trends: dict[str, dict[str, Any]] = {}
    with session_for(request) as session:
        type_rows = list(
            session.execute(
                select(
                    type_expression,
                    func.count(AttachmentOccurrence.id),
                    func.count(func.distinct(unique_key)),
                    func.coalesce(func.sum(AttachmentOccurrence.size_bytes), 0),
                )
                .where(*filters)
                .group_by(type_expression)
                .order_by(type_expression)
            )
        )
        availability_rows = list(
            session.execute(
                select(
                    AttachmentOccurrence.availability,
                    func.count(AttachmentOccurrence.id),
                    func.coalesce(func.sum(AttachmentOccurrence.size_bytes), 0),
                )
                .where(*filters)
                .group_by(AttachmentOccurrence.availability)
                .order_by(AttachmentOccurrence.availability)
            )
        )
        unknown_date_occurrences = (
            session.scalar(
                select(func.count(AttachmentOccurrence.id)).where(
                    *base_filters,
                    AttachmentOccurrence.timestamp_ms.is_(None),
                )
            )
            or 0
        )
        trend_rows = session.execute(
            select(
                AttachmentOccurrence.timestamp_ms,
                type_expression,
                AttachmentOccurrence.size_bytes,
                unique_key,
            ).where(*filters, AttachmentOccurrence.timestamp_ms.is_not(None))
        ).yield_per(5_000)
        for timestamp_ms, attachment_type, size_bytes, identity in trend_rows:
            assert timestamp_ms is not None
            period = _period(_local_datetime(timestamp_ms, zone), bucket)
            trend = trends.setdefault(
                period,
                {
                    "occurrences": 0,
                    "occurrence_bytes": 0,
                    "unique_files": set(),
                    "by_type": {},
                },
            )
            trend["occurrences"] += 1
            trend["occurrence_bytes"] += int(size_bytes or 0)
            trend["unique_files"].add(identity)
            by_type = trend["by_type"].setdefault(
                attachment_type,
                {"occurrences": 0, "occurrence_bytes": 0, "unique_files": set()},
            )
            by_type["occurrences"] += 1
            by_type["occurrence_bytes"] += int(size_bytes or 0)
            by_type["unique_files"].add(identity)
    total_occurrences = sum(int(row[1]) for row in type_rows)
    availability_counts = {str(row[0]): int(row[1]) for row in availability_rows}
    trend_items = [
        {
            "period": period,
            "occurrences": trend["occurrences"],
            "unique_files": len(trend["unique_files"]),
            "occurrence_bytes": trend["occurrence_bytes"],
            "by_type": {
                str(attachment_type): {
                    "occurrences": values["occurrences"],
                    "unique_files": len(values["unique_files"]),
                    "occurrence_bytes": values["occurrence_bytes"],
                }
                for attachment_type, values in sorted(trend["by_type"].items())
            },
        }
        for period, trend in trends.items()
    ]
    full_result = {
        "bucket": bucket,
        "timezone": timezone,
        "types": [
            {
                "type": attachment_type,
                "occurrences": occurrences,
                "unique_files": unique_files,
                "occurrence_bytes": size_bytes,
            }
            for attachment_type, occurrences, unique_files, size_bytes in type_rows
        ],
        "availability": [
            {
                "availability": availability,
                "occurrences": count,
                "bytes": size_bytes,
                "rate": count / total_occurrences if total_occurrences else None,
            }
            for availability, count, size_bytes in availability_rows
        ],
        "trends": trend_items,
        "summary": {
            "occurrences": total_occurrences,
            "dated_occurrences": (
                total_occurrences
                if from_ms is not None or to_ms is not None
                else total_occurrences - int(unknown_date_occurrences)
            ),
            "unknown_date_occurrences": int(unknown_date_occurrences),
            "unknown_date_scope": "archive_wide",
            "unknown_date_included_in_totals": from_ms is None and to_ms is None,
            "missing_occurrences": availability_counts.get("missing", 0),
            "missing_rate": (
                availability_counts.get("missing", 0) / total_occurrences
                if total_occurrences
                else None
            ),
            "unlinked_occurrences": availability_counts.get("unlinked", 0),
            "unlinked_rate": (
                availability_counts.get("unlinked", 0) / total_occurrences
                if total_occurrences
                else None
            ),
        },
        "count_semantics": (
            "Occurrence bytes count each send. Resolved blobs deduplicate by blob; occurrences "
            "without a blob identity count separately."
        ),
        "provenance": (
            "Trends use active attachment-occurrence timestamps in the selected IANA timezone; "
            + (
                "unknown-date attachments are reported archive-wide but cannot be tested against "
                "a date range and are excluded from these filtered totals and time buckets."
                if from_ms is not None or to_ms is not None
                else "unknown-date attachments remain in totals but not time buckets."
            )
        ),
    }
    _store_series_snapshot(request, "attachments", series_scope, full_result)
    result = _page_series_snapshot(
        full_result,
        kind="attachments",
        scope=series_scope,
        field="trends",
        limit=limit,
        cursor=cursor,
    )
    return _store_response(request, "attachments", cache_parameters, result)


@analytics_router.get("/conversations")
def conversation_analytics(request: Request) -> dict[str, Any]:
    with session_for(request) as session:
        rows = list(
            session.scalars(
                select(Conversation)
                .where(Conversation.deleted_at.is_(None))
                .order_by(Conversation.id)
            )
        )
    lifespans = [
        max(0.0, (row.last_activity_ms - row.first_activity_ms) / 86_400_000)
        for row in rows
        if row.first_activity_ms is not None and row.last_activity_ms is not None
    ]
    histogram: Counter[str] = Counter()
    for days in lifespans:
        if days < 1:
            histogram["same_day"] += 1
        elif days <= 7:
            histogram["1_to_7_days"] += 1
        elif days <= 30:
            histogram["8_to_30_days"] += 1
        elif days <= 365:
            histogram["31_to_365_days"] += 1
        else:
            histogram["over_one_year"] += 1
    histogram["unknown"] = len(rows) - len(lifespans)
    return {
        "conversation_count": len(rows),
        "by_type": dict(sorted(Counter(row.conversation_type for row in rows).items())),
        "longevity_days": {
            "mean": fmean(lifespans) if lifespans else None,
            "median": median(lifespans) if lifespans else None,
            "maximum": max(lifespans) if lifespans else None,
            "histogram": dict(histogram),
        },
        "active_days": {
            "mean": fmean(row.active_days for row in rows) if rows else None,
            "maximum": max((row.active_days for row in rows), default=None),
        },
        "ranking_policy": (
            "No people or relationships are ranked by toxicity, conflict, sentiment, or emotion."
        ),
        "derived_nlp": _derived_nlp_status(request),
    }


@analytics_router.get("/storage")
def storage_analytics(request: Request) -> dict[str, Any]:
    with session_for(request) as session:
        source_files, source_bytes = session.execute(
            select(
                func.count(SourceFile.id),
                func.coalesce(func.sum(SourceFile.size_bytes), 0),
            ).where(SourceFile.scan_status == "discovered")
        ).one()
        blob_count, blob_bytes = session.execute(
            select(func.count(Blob.sha256), func.coalesce(func.sum(Blob.size_bytes), 0))
        ).one()
        media_rows = list(
            session.execute(
                select(
                    case(
                        (func.lower(Blob.media_type).like("image/%"), "image"),
                        (func.lower(Blob.media_type).like("video/%"), "video"),
                        (func.lower(Blob.media_type).like("audio/%"), "audio"),
                        (func.lower(Blob.media_type).like("text/%"), "text"),
                        else_="other",
                    ).label("kind"),
                    func.count(Blob.sha256),
                    func.coalesce(func.sum(Blob.size_bytes), 0),
                )
                .group_by("kind")
                .order_by("kind")
            )
        )
    source_bytes_value = int(source_bytes or 0)
    blob_bytes_value = int(blob_bytes or 0)
    return {
        "source_occurrences": {"files": source_files, "bytes": source_bytes_value},
        "unique_blobs": {"files": blob_count, "bytes": blob_bytes_value},
        "deduplication": {
            "redundant_occurrences": max(0, int(source_files or 0) - int(blob_count or 0)),
            "duplicate_bytes": max(0, source_bytes_value - blob_bytes_value),
        },
        "by_media_class": [
            {"class": media_class, "files": count, "bytes": size_bytes}
            for media_class, count, size_bytes in media_rows
        ],
    }


@analytics_router.get("/coverage")
def analytics_coverage(request: Request) -> dict[str, Any]:
    with session_for(request) as session:
        rows = list(session.scalars(select(CoverageArea).order_by(CoverageArea.key)))
    counts = Counter(row.status for row in rows)
    items = [
        {
            "key": row.key,
            "label": row.label,
            "status": row.status,
            "raw_count": row.raw_count,
            "logical_count": row.logical_count,
            "description": row.description,
            "source_paths": row.source_paths_json,
            "absence_is_not_zero_history": row.status == "unavailable",
        }
        for row in rows
    ]
    observed_at = max((row.updated_at for row in rows), default=None)
    observed_at_text = observed_at.isoformat() if observed_at is not None else None
    snapshot_items = [dict(row) for row in items]
    return {
        "summary": dict(sorted(counts.items())),
        "items": items,
        "history": {
            "mode": "single_observed_export_snapshot" if rows else "unavailable",
            "historical_comparison_available": False,
            "description": (
                "This catalog contains one computed coverage observation for the loaded export. "
                "It does not contain prior export snapshots, so coverage change over time "
                "cannot be shown."
                if rows
                else "No export-coverage snapshot is available."
            ),
            "observed_at": observed_at_text,
            "observed_at_basis": (
                "Latest coverage-area catalog update time, not the export creation time."
                if rows
                else None
            ),
            "snapshots": (
                [
                    {
                        "id": stable_id(
                            "coverage-snapshot",
                            observed_at_text,
                            tuple(
                                (
                                    item["key"],
                                    item["status"],
                                    item["raw_count"],
                                    item["logical_count"],
                                )
                                for item in snapshot_items
                            ),
                        ),
                        "observed_at": observed_at_text,
                        "summary": dict(sorted(counts.items())),
                        "items": snapshot_items,
                    }
                ]
                if rows
                else []
            ),
        },
    }
