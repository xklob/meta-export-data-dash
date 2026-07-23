#!/usr/bin/env python3
"""Real-export reconciliation, jump, performance, and NLP acceptance audit.

The report deliberately excludes source text, display names, titles, filenames, paths inside the
export, and API response bodies. Run it through ``scripts/run-private-ui-review.sh`` so every
artifact is born with owner-only permissions.

The direct catalog connection is SQLite ``query_only``. Live checks require the dedicated
``--acceptance-read-only`` server, whose authenticated GET routes bypass derived-cache reads and
writes; the audit performs no destructive mutation.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import http.cookiejar
import json
import math
import os
import random
import re
import sqlite3
import stat
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from statistics import fmean, median
from typing import Any, Literal, cast
from zoneinfo import ZoneInfo

from PIL import Image, UnidentifiedImageError
from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.pool import NullPool

from meta_archive.baseline import (
    MAX_AUDITED_BASELINE_MANIFEST_BYTES,
    AuditedBaselineManifestError,
    audited_baseline_manifest_path,
    load_audited_baseline,
)
from meta_archive.paths import ArchiveLayout
from meta_archive.services.analytics_cache import CACHE_VERSION as ANALYTICS_CACHE_VERSION
from meta_archive.services.analytics_cache import analytics_source_revision
from meta_archive.services.attachment_text import (
    EXTRACTOR_REVISION as ATTACHMENT_EXTRACTOR_REVISION,
)
from meta_archive.services.attachment_text import (
    PIPELINE_VERSION as ATTACHMENT_TEXT_PIPELINE_VERSION,
)
from meta_archive.services.attachment_text import _candidate_rows as attachment_text_candidates
from meta_archive.services.conflict_scoring import (
    CONFLICT_LABELS,
    CONFLICT_WARNINGS,
    DEFAULT_CONFLICT_SPEC,
)
from meta_archive.services.direct_conflict_ranker import (
    DEFAULT_DIRECT_CONFLICT_PARAMETERS,
    DIRECT_CONFLICT_RANKER_VERSION,
    DIRECT_CONFLICT_RANKING_K,
    DIRECT_CONFLICT_SCORE_KIND,
    is_direct_conflict_trigger,
)
from meta_archive.services.direct_conflict_ranker import (
    DIRECT_CONFLICT_EVALUATION_INTENT as DIRECT_CONFLICT_DEVELOPMENT_INTENT,
)
from meta_archive.services.direct_conflict_ranker import (
    DIRECT_CONFLICT_MIN_NOT_RELEVANT as DIRECT_CONFLICT_DEVELOPMENT_MIN_NOT_RELEVANT_COUNT,
)
from meta_archive.services.direct_conflict_ranker import (
    DIRECT_CONFLICT_MIN_RELEVANT as DIRECT_CONFLICT_DEVELOPMENT_MIN_RELEVANT_COUNT,
)
from meta_archive.services.direct_conflict_ranker import (
    DIRECT_CONFLICT_VALIDATION_FOLDS as DIRECT_CONFLICT_DEVELOPMENT_FOLD_COUNT,
)
from meta_archive.services.direct_conflict_ranker import (
    development_feedback_revision_payload as development_feedback_revision,
)
from meta_archive.services.media_derivatives import (
    PIPELINE_VERSION as MEDIA_DERIVATIVE_PIPELINE_VERSION,
)
from meta_archive.services.media_derivatives import _candidates as media_derivative_candidates
from meta_archive.services.media_derivatives import derivative_id
from meta_archive.services.nlp_annotations import (
    CONFLICT_PIPELINE_VERSION,
    DIRECT_PROVENANCE_BINDING,
    LEGACY_PROVENANCE_BINDING,
    MAX_TEXT_CHARACTERS,
    MIN_CONFLICT_LABELS,
    MIN_DIRECT_CONFLICT_PRECISION,
    MIN_MACRO_F1,
    MODEL_SPECS,
    NLP_IMPLEMENTATION_SCHEMA,
    NLP_INFERENCE_CONFIGURATION_SCHEMA,
    NLP_TOKENIZER_MAX_LENGTH,
    PIPELINE_VERSION,
)
from meta_archive.services.reranking import DEFAULT_RERANKER_SPEC
from meta_archive.services.search_index import (
    HELD_OUT_MIN_QUERY_COUNT,
    HELD_OUT_THRESHOLDS,
    KNOWN_ITEM_THRESHOLDS,
    SEMANTIC_IR_EVALUATOR_VERSION,
    SEMANTIC_IR_K,
    embedding_implementation_identity,
    held_out_feedback_revision,
)
from meta_archive.services.search_index import (
    MODEL_SPECS as SEMANTIC_MODEL_SPECS,
)
from meta_archive.services.search_index import (
    WINDOW_VERSION as SEARCH_WINDOW_VERSION,
)
from meta_archive.services.semantic_state import (
    SemanticTableValidation,
    lance_table_names,
    validate_semantic_table,
)
from meta_archive.services.topic_clusters import PIPELINE_VERSION as TOPIC_CLUSTER_PIPELINE_VERSION
from meta_archive.services.transcription import MODEL_NAME as TRANSCRIPTION_MODEL_NAME
from meta_archive.services.transcription import MODEL_REVISION as TRANSCRIPTION_MODEL_REVISION
from meta_archive.services.transcription import PIPELINE_VERSION as TRANSCRIPTION_PIPELINE_VERSION
from meta_archive.services.transcription import _candidates as transcription_candidates
from meta_archive.services.verification import (
    SOURCE_RETIREMENT_REQUIRED_CHECK_KEYS,
    mandatory_non_pass_checks,
    verify_catalog,
)

REPORT_SCHEMA_VERSION = 2
BROWSER_REPORT_SCHEMA_VERSION = 5
BROWSER_INTERACTION_PROOF_METRIC_MAX = 100
DIRECT_CONFLICT_DEVELOPMENT_EVIDENCE_KIND = "conversation_grouped_development_calibration"
DIRECT_CONFLICT_DEVELOPMENT_BENCHMARK_KIND = "literal_direct_conflict_development"
DIRECT_CONFLICT_DEVELOPMENT_METHOD = "linear_svm"
DIRECT_CONFLICT_RANKING_PIPELINE_VERSION = "runtime-search-v2"
DIRECT_CONFLICT_CLASSIFIER = "LinearSVC"
DIRECT_CONFLICT_LITERAL_PIPELINE = "exact_cosine_then_literal_direct_conflict_linear_svm"
DIRECT_CONFLICT_LITERAL_TRIGGER = "exact_normalized_literal"
DIRECT_CONFLICT_PUBLIC_METRICS = (
    "average_precision",
    "roc_auc",
    "recall_at_10",
    "mrr_at_10",
    "ndcg_at_10",
)
DIRECT_CONFLICT_DEVELOPMENT_THRESHOLDS = {
    "recall_at_10": 0.60,
    "mrr_at_10": 0.50,
    "ndcg_at_10": 0.55,
}
DEFAULT_QUERY = "argument"
DEFAULT_REPEATS = 10
DEFAULT_SAMPLES = 12
OWNER_DIRECTORY_MODE = 0o700
OWNER_FILE_MODE = 0o600
MAX_BROWSER_REPORT_AGE = timedelta(hours=24)
MAX_BROWSER_REPORT_BYTES = 4 * 1024 * 1024
MAX_ANALYTICS_CACHE_BYTES = 128 * 1024 * 1024
MAX_API_RESPONSE_BYTES = 64 * 1024 * 1024
SORT_AUDIT_PAGE_SIZE = 250
RECONCILIATION_MODES = ("risk-bounded", "exhaustive", "production-spot-check")
RISK_RECONCILIATION_PROFILE_VERSION = "risk-bounded-v1"
PRODUCTION_SPOT_CHECK_PROFILE_VERSION = "production-spot-check-v2"
DEFAULT_RECONCILIATION_SEED = 20260722
RISK_SORT_PAGE_CAP = 2
PRODUCTION_SPOT_CHECK_SORT_STREAMS = 8
PRODUCTION_SPOT_CHECK_PAGES_PER_STREAM = 2
PRODUCTION_SPOT_CHECK_PAGE_SIZE = 25
PRODUCTION_SPOT_CHECK_MAX_TARGETED_LOOKUP_ROWS = (
    PRODUCTION_SPOT_CHECK_PAGES_PER_STREAM * PRODUCTION_SPOT_CHECK_PAGE_SIZE
)
PRODUCTION_SPOT_CHECK_SAMPLE_ROWS = 12
PRODUCTION_SPOT_CHECK_DERIVED_FAMILIES = 3
PRODUCTION_SPOT_CHECK_ANALYTICS_CALLS = 2
PRODUCTION_SPOT_CHECK_NLP_TASKS = ("sentiment", "emotion", "toxicity")
PRODUCTION_SPOT_CHECK_NLP_DIRECTIONS = (
    "owner",
    "other",
    "activity",
    "audio",
    "attachment",
    "all",
)
PRODUCTION_SPOT_CHECK_NLP_PERIODS = 12
PRODUCTION_SPOT_CHECK_NLP_MAX_ROWS_PER_TASK = (
    len(PRODUCTION_SPOT_CHECK_NLP_DIRECTIONS) * PRODUCTION_SPOT_CHECK_NLP_PERIODS
)
PRODUCTION_SPOT_CHECK_NLP_MAX_SERIES_ROWS = (
    len(PRODUCTION_SPOT_CHECK_NLP_TASKS) * PRODUCTION_SPOT_CHECK_NLP_MAX_ROWS_PER_TASK
)
PRODUCTION_SPOT_CHECK_SEARCH_CALLS = 3
PRODUCTION_SPOT_CHECK_MAX_API_CALLS = 48
PRODUCTION_SPOT_CHECK_MAX_API_RESPONSE_BYTES = 8 * 1024 * 1024
PRODUCTION_SPOT_CHECK_REPORT_MAX_BYTES = 8 * 1024 * 1024
PRODUCTION_SPOT_CHECK_LIBRARY_MARKER_MAX_BYTES = 1024 * 1024
EXPECTED_BROWSER_ROUTES = (
    "home",
    "coverage",
    "integrity",
    "jobs",
    "settings",
    "conversations",
    "media",
    "search",
    "enrichment",
    "people",
    "events",
    "activity",
    "timeline",
    "support",
    "analytics",
    "deletion",
    "largest-conversation",
)
EXPECTED_BROWSER_VIEWPORTS = ("desktop", "mobile")
EXPECTED_BROWSER_INTERACTIONS = (
    "coverage_status_filter",
    "conversation_sort_and_page",
    "conversation_reader_edges_bounded",
    "conversation_types_source_bound",
    "media_modes_and_sort",
    "media_message_exact_jump",
    "media_activity_exact_jump",
    "media_representative_kinds_and_jumps",
    "keyword_search_exact_jump",
    "exact_search_exact_jump",
    "semantic_search_exact_jump",
    "semantic_argument_search_exact_jump",
    "hybrid_search_exact_jump",
    "enrichment_exact_source_jump",
    "people_views_sort_and_page",
    "events_filter_sort_and_page",
    "activity_filter_sort_and_page",
    "activity_post_comment_source_jumps",
    "timeline_mode_filter_and_page",
    "analytics_bucket_queries",
    "deletion_preview_guard",
)
BROWSER_INTERACTION_PROOF_KEYS = (
    "assertion_count",
    "dom_api_reconciliation_count",
    "nonempty_dataset_count",
    "source_bound_empty_dataset_count",
    "pagination_bound_count",
    "pagination_exercised_count",
    "source_jump_count",
    "representative_kind_count",
    "conversation_type_count",
)
BROWSER_INTERACTION_PROOF_MINIMUMS: dict[str, dict[str, int]] = {
    "coverage_status_filter": {"assertion_count": 1, "nonempty_dataset_count": 1},
    "conversation_sort_and_page": {
        "assertion_count": 8,
        "dom_api_reconciliation_count": 10,
        "nonempty_dataset_count": 4,
        "pagination_bound_count": 1,
        "pagination_exercised_count": 1,
    },
    "conversation_reader_edges_bounded": {
        "assertion_count": 8,
        "dom_api_reconciliation_count": 8,
        "nonempty_dataset_count": 1,
        "pagination_bound_count": 2,
        "pagination_exercised_count": 2,
    },
    "conversation_types_source_bound": {
        "assertion_count": 4,
        "dom_api_reconciliation_count": 2,
        "nonempty_dataset_count": 2,
        "conversation_type_count": 2,
    },
    "media_modes_and_sort": {
        "assertion_count": 6,
        "dom_api_reconciliation_count": 4,
        "nonempty_dataset_count": 4,
    },
    "media_message_exact_jump": {
        "assertion_count": 2,
        "dom_api_reconciliation_count": 1,
        "nonempty_dataset_count": 1,
        "source_jump_count": 1,
    },
    "media_activity_exact_jump": {
        "assertion_count": 2,
        "dom_api_reconciliation_count": 1,
        "nonempty_dataset_count": 1,
        "source_jump_count": 1,
    },
    "media_representative_kinds_and_jumps": {
        "assertion_count": 8,
        "dom_api_reconciliation_count": 4,
        "nonempty_dataset_count": 4,
        "source_jump_count": 4,
        "representative_kind_count": 4,
    },
    "keyword_search_exact_jump": {
        "assertion_count": 2,
        "dom_api_reconciliation_count": 1,
        "nonempty_dataset_count": 1,
        "source_jump_count": 1,
    },
    "exact_search_exact_jump": {
        "assertion_count": 2,
        "dom_api_reconciliation_count": 1,
        "nonempty_dataset_count": 1,
        "source_jump_count": 1,
    },
    "semantic_search_exact_jump": {
        "assertion_count": 2,
        "dom_api_reconciliation_count": 1,
        "nonempty_dataset_count": 1,
        "source_jump_count": 1,
    },
    "semantic_argument_search_exact_jump": {
        "assertion_count": 2,
        "dom_api_reconciliation_count": 1,
        "nonempty_dataset_count": 1,
        "source_jump_count": 1,
    },
    "hybrid_search_exact_jump": {
        "assertion_count": 2,
        "dom_api_reconciliation_count": 1,
        "nonempty_dataset_count": 1,
        "source_jump_count": 1,
    },
    "enrichment_exact_source_jump": {"assertion_count": 1, "source_jump_count": 1},
    "people_views_sort_and_page": {
        "assertion_count": 8,
        "dom_api_reconciliation_count": 7,
        "nonempty_dataset_count": 1,
        "pagination_bound_count": 1,
        "pagination_exercised_count": 1,
    },
    "events_filter_sort_and_page": {
        "assertion_count": 2,
        "dom_api_reconciliation_count": 1,
        "pagination_bound_count": 1,
    },
    "activity_filter_sort_and_page": {
        "assertion_count": 3,
        "dom_api_reconciliation_count": 2,
        "nonempty_dataset_count": 1,
        "pagination_bound_count": 1,
    },
    "activity_post_comment_source_jumps": {
        "assertion_count": 4,
        "dom_api_reconciliation_count": 2,
        "nonempty_dataset_count": 2,
        "pagination_bound_count": 1,
        "pagination_exercised_count": 1,
        "source_jump_count": 2,
        "representative_kind_count": 2,
    },
    "timeline_mode_filter_and_page": {
        "assertion_count": 4,
        "dom_api_reconciliation_count": 2,
        "nonempty_dataset_count": 1,
        "pagination_bound_count": 2,
        "pagination_exercised_count": 1,
    },
    "analytics_bucket_queries": {
        "assertion_count": 4,
        "dom_api_reconciliation_count": 2,
    },
    "deletion_preview_guard": {"assertion_count": 3},
}
EXPECTED_BROWSER_PAGE_RUNS = len(EXPECTED_BROWSER_ROUTES) * len(EXPECTED_BROWSER_VIEWPORTS)
EXPECTED_BROWSER_INTERACTION_RUNS = len(EXPECTED_BROWSER_INTERACTIONS) * len(
    EXPECTED_BROWSER_VIEWPORTS
)
EXPECTED_BROWSER_ARTIFACTS = frozenset(
    {
        *(
            f"artifacts/{viewport}-{route}.png"
            for viewport in EXPECTED_BROWSER_VIEWPORTS
            for route in EXPECTED_BROWSER_ROUTES
        ),
        *(f"artifacts/{viewport}-trace.zip" for viewport in EXPECTED_BROWSER_VIEWPORTS),
    }
)
PERFORMANCE_THRESHOLDS_MS = {
    "list_api_p95": 500.0,
    "date_jump_api_p95": 500.0,
    "keyword_top_50_p95": 1_000.0,
    "semantic_top_50_p95": 2_000.0,
    "initial_page_content_p95": 2_500.0,
}
TASKS: tuple[Literal["sentiment"], Literal["emotion"], Literal["toxicity"]] = (
    "sentiment",
    "emotion",
    "toxicity",
)
ATTACHMENT_EXTRACTOR_NAME = "bubblewrap-prlimit-local-worker"
SESSION_COOKIE_NAME = "meta_archive_session"
_MEDIA_CLASS_SQL = """
CASE
  WHEN a.kind IN ('gifs', 'gif') THEN 'gif'
  WHEN a.kind = 'sticker' THEN 'sticker'
  WHEN a.kind IN ('audio_files', 'audio') THEN 'audio'
  WHEN a.kind IN ('videos', 'video') THEN 'video'
  WHEN a.kind IN ('photos', 'photo', 'thread_image') THEN 'photo'
  WHEN lower(coalesce(a.media_type, '')) LIKE 'audio/%' THEN 'audio'
  WHEN lower(coalesce(a.media_type, '')) LIKE 'video/%' THEN 'video'
  WHEN lower(coalesce(a.media_type, '')) LIKE 'image/%' THEN 'photo'
  WHEN a.kind IN ('files', 'file', 'support_file', 'document')
       OR lower(coalesce(a.media_type, '')) IN ('application/pdf', 'text/plain') THEN 'document'
  ELSE 'other'
END
""".strip()
_ANALYTICS_URL_PATTERN = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
_ANALYTICS_RESPONSE_BUCKETS = (
    (60_000, "under_1_minute"),
    (5 * 60_000, "1_to_5_minutes"),
    (30 * 60_000, "5_to_30_minutes"),
    (2 * 60 * 60_000, "30_minutes_to_2_hours"),
    (24 * 60 * 60_000, "2_to_24_hours"),
    (7 * 24 * 60 * 60_000, "1_to_7_days"),
)
ANALYTICS_DASHBOARD_ENDPOINTS = (
    "overview",
    "messages",
    "activity-patterns",
    "interaction-patterns",
    "interaction-series",
    "nlp-trends",
    "conflict-rate",
    "network",
    "topics",
    "attachments",
    "conversations",
    "storage",
    "coverage",
)
ANALYTICS_TIMEZONES = ("UTC", "America/New_York")
ANALYTICS_TIME_BUCKETS = ("day", "week", "month", "year")
ANALYTICS_NLP_TIME_BUCKETS = ("hour", *ANALYTICS_TIME_BUCKETS)


@dataclass(slots=True)
class AcceptanceCheck:
    key: str
    status: str
    expected: object
    actual: object
    detail: str
    required: bool = True


@dataclass(frozen=True, slots=True)
class ReconciliationPolicy:
    mode: str
    seed: int

    @property
    def profile_version(self) -> str:
        if self.mode == "risk-bounded":
            return RISK_RECONCILIATION_PROFILE_VERSION
        if self.mode == "production-spot-check":
            return PRODUCTION_SPOT_CHECK_PROFILE_VERSION
        return "exhaustive-v1"

    @property
    def sort_page_cap(self) -> int | None:
        return (
            RISK_SORT_PAGE_CAP if self.mode in {"risk-bounded", "production-spot-check"} else None
        )


def _check(
    key: str,
    passed: bool,
    *,
    expected: object,
    actual: object,
    detail: str,
    required: bool = True,
) -> AcceptanceCheck:
    return AcceptanceCheck(
        key=key,
        status="pass" if passed else "fail",
        expected=expected,
        actual=actual,
        detail=detail,
        required=required,
    )


def _pending(key: str, detail: str, *, required: bool = True) -> AcceptanceCheck:
    return AcceptanceCheck(
        key=key,
        status="pending",
        expected="completed acceptance evidence",
        actual="not supplied",
        detail=detail,
        required=required,
    )


def _opaque(value: object) -> str:
    return hashlib.sha256(str(value).encode()).hexdigest()[:16]


def _safe_json_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _safe_json_value(child) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_safe_json_value(child) for child in value]
    return str(value)


def _json_object(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _json_object_list(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _timing_summary(values: Sequence[float]) -> dict[str, object]:
    return {
        "samples": len(values),
        "min_ms": round(min(values), 3) if values else None,
        "median_ms": round(_percentile(values, 0.5) or 0.0, 3) if values else None,
        "p95_ms": round(_percentile(values, 0.95) or 0.0, 3) if values else None,
        "max_ms": round(max(values), 3) if values else None,
    }


def _validate_regular_file(path: Path, *, label: str) -> None:
    try:
        path_stat = path.lstat()
    except FileNotFoundError as exc:
        raise RuntimeError(f"{label} does not exist: {path}") from exc
    if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
        raise RuntimeError(f"{label} must be a regular, non-symlink file: {path}")


def _open_regular_nofollow(path: Path, *, label: str) -> tuple[int, os.stat_result]:
    """Open one bounded-trust input without a path-based check/read race."""

    flags = os.O_RDONLY
    for optional_flag in ("O_CLOEXEC", "O_NOFOLLOW", "O_NONBLOCK"):
        flags |= int(getattr(os, optional_flag, 0))
    try:
        descriptor = os.open(path, flags)
    except OSError:
        raise RuntimeError(f"{label} must be a readable regular, non-symlink file") from None
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise RuntimeError(f"{label} must be a regular, non-symlink file")
        return descriptor, file_stat
    except BaseException:
        os.close(descriptor)
        raise


def _read_regular_bytes(
    path: Path,
    *,
    label: str,
    maximum_bytes: int,
    owner_only: bool,
) -> bytes:
    descriptor, file_stat = _open_regular_nofollow(path, label=label)
    try:
        if owner_only and stat.S_IMODE(file_stat.st_mode) & 0o077:
            raise RuntimeError(f"{label} must not be group- or other-accessible")
        if file_stat.st_size > maximum_bytes:
            raise RuntimeError(f"{label} is unexpectedly large")
        chunks: list[bytes] = []
        remaining = maximum_bytes + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(1_048_576, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > maximum_bytes:
            raise RuntimeError(f"{label} is unexpectedly large")
        return payload
    finally:
        os.close(descriptor)


def _sha256_regular_file(path: Path, *, label: str) -> tuple[int, str, int]:
    descriptor, file_stat = _open_regular_nofollow(path, label=label)
    try:
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1_048_576):
            digest.update(chunk)
        return int(file_stat.st_size), digest.hexdigest(), stat.S_IMODE(file_stat.st_mode)
    finally:
        os.close(descriptor)


class _SameOriginRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Allow the expected auth redirect without ever leaving the loopback origin."""

    def __init__(self, origin: tuple[str, str, int | None]) -> None:
        super().__init__()
        self.origin = origin

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        parsed = urllib.parse.urlsplit(newurl)
        if (parsed.scheme, parsed.hostname or "", parsed.port) != self.origin:
            raise urllib.error.HTTPError(
                req.full_url,
                code,
                "Cross-origin redirect refused",
                headers,
                fp,
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _readonly_catalog_engine(database: Path) -> Engine:
    # ``mode=ro`` alone is not filesystem-read-only for a database whose persistent journal
    # mode is WAL: SQLite creates ``-wal`` and ``-shm`` sidecars even for a reader.  That would
    # both mutate the archive and make the already-running acceptance server fail its sidecar
    # gate.  Hold the exact no-follow database inode and reopen it through procfs as immutable so
    # every pooled connection remains sidecar-free and cannot be redirected by a path swap.
    database_descriptor, opened_stat = _open_regular_nofollow(
        database,
        label="Catalog database",
    )
    try:
        path_stat = database.lstat()
        if (
            stat.S_ISLNK(path_stat.st_mode)
            or not stat.S_ISREG(path_stat.st_mode)
            or (path_stat.st_dev, path_stat.st_ino) != (opened_stat.st_dev, opened_stat.st_ino)
        ):
            raise RuntimeError("Catalog database changed while it was opened")
    except BaseException:
        os.close(database_descriptor)
        raise
    database_uri = f"file:/proc/self/fd/{database_descriptor}?mode=ro&immutable=1"

    def connect() -> sqlite3.Connection:
        connection = sqlite3.connect(database_uri, uri=True, check_same_thread=False)
        connection.create_function(
            "unicode_casefold",
            1,
            lambda value: value.casefold() if isinstance(value, str) else "",
            deterministic=True,
        )
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA temp_store=MEMORY")
        if connection.execute("PRAGMA query_only").fetchone() != (1,):
            connection.close()
            raise RuntimeError("SQLite did not enable query_only mode")
        return connection

    try:
        engine = create_engine(
            "sqlite+pysqlite://",
            creator=connect,
            future=True,
            poolclass=NullPool,
            hide_parameters=True,
        )
    except BaseException:
        os.close(database_descriptor)
        raise

    descriptor_open = True

    @event.listens_for(engine, "engine_disposed")
    def close_catalog_descriptor(_engine: Engine) -> None:
        nonlocal descriptor_open
        if descriptor_open:
            os.close(database_descriptor)
            descriptor_open = False

    return engine


class LocalApiClient:
    def __init__(self, base_url: str, startup_token: str) -> None:
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise RuntimeError("Acceptance API URL must be an HTTP loopback address")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise RuntimeError(
                "Acceptance API URL must not contain credentials, query, or fragment"
            )
        if parsed.path not in {"", "/"}:
            raise RuntimeError("Acceptance API URL must be a loopback origin without a path")
        self.base_url = base_url.rstrip("/")
        cookie_jar = http.cookiejar.CookieJar()
        origin = (parsed.scheme, parsed.hostname or "", parsed.port)
        self._opener = urllib.request.build_opener(
            _SameOriginRedirectHandler(origin),
            urllib.request.HTTPCookieProcessor(cookie_jar),
        )
        auth_url = f"{self.base_url}/auth?{urllib.parse.urlencode({'token': startup_token})}"
        try:
            with self._opener.open(auth_url, timeout=30) as response:
                authenticated = urllib.parse.urlsplit(response.geturl())
                if (
                    authenticated.scheme,
                    authenticated.hostname or "",
                    authenticated.port,
                ) != origin or authenticated.path != "/":
                    raise RuntimeError("Dedicated acceptance-server authentication failed")
                response.read(1)
            if not any(cookie.name == SESSION_COOKIE_NAME for cookie in cookie_jar):
                raise RuntimeError("Dedicated acceptance-server authentication failed")
        except Exception:
            # urllib exception strings can reproduce their full URL, including the one-time token.
            raise RuntimeError("Dedicated acceptance-server authentication failed") from None
        finally:
            auth_url = ""

    def get_json(
        self,
        path: str,
        parameters: Mapping[str, object] | None = None,
        *,
        timeout: float = 120.0,
        maximum_bytes: int = MAX_API_RESPONSE_BYTES,
    ) -> tuple[dict[str, Any], float]:
        if not 1 <= maximum_bytes <= MAX_API_RESPONSE_BYTES:
            raise ValueError("API response byte cap is outside the supported range")
        encoded = urllib.parse.urlencode(
            {key: value for key, value in (parameters or {}).items() if value is not None}
        )
        url = f"{self.base_url}{path}{'?' + encoded if encoded else ''}"
        started = time.perf_counter()
        try:
            with self._opener.open(url, timeout=timeout) as response:
                content_length = response.headers.get("Content-Length")
                if (
                    isinstance(content_length, str)
                    and content_length.isdecimal()
                    and int(content_length) > maximum_bytes
                ):
                    raise RuntimeError("Acceptance API response exceeded its byte cap")
                response_bytes = response.read(maximum_bytes + 1)
                if len(response_bytes) > maximum_bytes:
                    raise RuntimeError("Acceptance API response exceeded its byte cap")
                payload = json.loads(response_bytes)
        except urllib.error.HTTPError as exc:
            # Never include the response body: API errors can contain private query excerpts.
            raise RuntimeError(f"Acceptance API GET returned HTTP {exc.code}") from None
        except (OSError, TimeoutError, json.JSONDecodeError):
            # Network exception strings and decode failures can contain the full request URL.
            raise RuntimeError("Acceptance API GET did not return a usable response") from None
        elapsed_ms = (time.perf_counter() - started) * 1_000
        if not isinstance(payload, dict):
            raise RuntimeError("Acceptance API GET did not return a JSON object")
        return payload, elapsed_ms


_ACCEPTANCE_IDENTITY_KEYS = frozenset(
    {
        "catalog_source_revision",
        "backend_version",
        "backend_source_sha256",
        "frontend_manifest_sha256",
        "server_nonce",
    }
)
_ACCEPTANCE_STABLE_IDENTITY_KEYS = (
    "catalog_source_revision",
    "backend_version",
    "backend_source_sha256",
    "frontend_manifest_sha256",
)
_LOWER_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _acceptance_server_identity(
    client: LocalApiClient | None,
) -> tuple[list[AcceptanceCheck], dict[str, object], dict[str, str]]:
    if client is None:
        return (
            [
                _pending(
                    "api.acceptance_read_only_identity",
                    "Run against a dedicated --acceptance-read-only loopback server",
                )
            ],
            {},
            {},
        )
    try:
        payload, _elapsed = client.get_json(
            "/api/v1/acceptance/identity",
            timeout=600.0,
        )
    except RuntimeError:
        payload = {}
    exact_schema = set(payload) == _ACCEPTANCE_IDENTITY_KEYS
    digests_valid = all(
        isinstance(payload.get(key), str)
        and _LOWER_SHA256_PATTERN.fullmatch(str(payload.get(key))) is not None
        for key in (
            "catalog_source_revision",
            "backend_source_sha256",
            "frontend_manifest_sha256",
            "server_nonce",
        )
    )
    backend_version_valid = (
        isinstance(payload.get("backend_version"), str)
        and 0 < len(str(payload.get("backend_version"))) <= 100
    )
    passed = exact_schema and digests_valid and backend_version_valid
    report: dict[str, object] = {
        "identity_response_observed": bool(payload),
        "exact_schema": exact_schema,
        "digest_fields_valid": digests_valid,
        "backend_version_valid": backend_version_valid,
        "identity_values_emitted": False,
    }
    return (
        [
            _check(
                "api.acceptance_read_only_identity",
                passed,
                expected="exact five-field identity from an acceptance-read-only server",
                actual=report,
                detail=(
                    "The identity-only route exists only in GET/HEAD-only query-only server mode; "
                    "identity values remain transient"
                ),
            )
        ],
        report,
        ({key: str(payload[key]) for key in _ACCEPTANCE_IDENTITY_KEYS} if passed else {}),
    )


def _scalar(connection: Any, statement: str, parameters: Mapping[str, object] | None = None) -> Any:
    return connection.execute(text(statement), parameters or {}).scalar_one()


def _rows(
    connection: Any, statement: str, parameters: Mapping[str, object] | None = None
) -> list[Any]:
    return list(connection.execute(text(statement), parameters or {}))


def _table_names(connection: Any) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            text("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")
        )
    }


def _catalog_totals(connection: Any) -> dict[str, int]:
    return {
        "conversations": int(
            _scalar(connection, "SELECT count(*) FROM conversations WHERE deleted_at IS NULL")
        ),
        "messages": int(
            _scalar(connection, "SELECT count(*) FROM messages WHERE deleted_at IS NULL")
        ),
        "attachments": int(
            _scalar(
                connection,
                "SELECT count(*) FROM attachment_occurrences WHERE deleted_at IS NULL",
            )
        ),
        "unique_media": int(
            _scalar(
                connection,
                """
                SELECT count(DISTINCT CASE
                    WHEN blob_sha256 IS NOT NULL THEN 'blob:' || blob_sha256
                    ELSE 'occurrence:' || id
                END)
                FROM attachment_occurrences
                WHERE deleted_at IS NULL
                """,
            )
        ),
        "activities": int(
            _scalar(connection, "SELECT count(*) FROM activity_records WHERE deleted_at IS NULL")
        ),
        "events": int(
            _scalar(
                connection,
                """
                SELECT count(*)
                FROM activity_records AS activity
                JOIN source_files AS source ON source.id = activity.source_file_id
                WHERE activity.deleted_at IS NULL
                  AND (
                    activity.category = 'events'
                    OR lower(source.normalized_path) LIKE '%/events/%'
                  )
                """,
            )
        ),
        "contacts": int(
            _scalar(
                connection,
                """
                SELECT count(DISTINCT participant.name_search)
                FROM participants AS participant
                JOIN segment_participants AS membership
                  ON membership.participant_id = participant.id
                JOIN conversation_segments AS segment ON segment.id = membership.segment_id
                JOIN conversations AS conversation ON conversation.id = segment.conversation_id
                WHERE participant.is_owner = 0
                  AND conversation.deleted_at IS NULL
                  AND conversation.conversation_type IN ('direct', 'group')
                """,
            )
        ),
        "timeline": int(
            _scalar(connection, "SELECT count(*) FROM messages WHERE deleted_at IS NULL")
            + _scalar(connection, "SELECT count(*) FROM activity_records WHERE deleted_at IS NULL")
        ),
    }


def _current_deep_integrity_gate(
    verification: Sequence[Mapping[str, object]],
) -> tuple[AcceptanceCheck, dict[str, object]]:
    """Gate the deep verifier results produced inside this acceptance run.

    A timestamped report can become stale immediately after it is written. Final acceptance
    therefore evaluates the current ``verify_catalog(..., deep=True)`` result directly and
    requires the complete source-retirement check inventory. Only check keys and aggregate counts
    are retained in the private acceptance report.
    """

    normalized = [dict(item) for item in verification if isinstance(item, Mapping)]
    malformed_count = len(verification) - len(normalized)
    keys = [str(item.get("key")) for item in normalized if isinstance(item.get("key"), str)]
    duplicate_keys = sorted({key for key in keys if keys.count(key) > 1})
    present_keys = set(keys)
    required_present = SOURCE_RETIREMENT_REQUIRED_CHECK_KEYS & present_keys
    non_pass = mandatory_non_pass_checks(normalized)
    non_pass_keys = sorted(
        {str(item.get("key", "unknown")) for item in non_pass if isinstance(item, Mapping)}
    )
    deep_results = [item for item in normalized if item.get("key") == "deep.source_bytes"]
    deep_verified = len(deep_results) == 1 and deep_results[0].get("status") == "pass"
    passed = (
        bool(normalized)
        and malformed_count == 0
        and len(keys) == len(normalized)
        and not duplicate_keys
        and required_present == SOURCE_RETIREMENT_REQUIRED_CHECK_KEYS
        and not non_pass
        and deep_verified
    )
    sanitized = {
        "executed_in_current_acceptance": True,
        "check_count": len(normalized),
        "malformed_check_count": malformed_count,
        "duplicate_key_count": len(duplicate_keys),
        "required_check_count": len(SOURCE_RETIREMENT_REQUIRED_CHECK_KEYS),
        "required_check_present_count": len(required_present),
        "deep_verified": deep_verified,
        "all_mandatory_checks_pass": not non_pass,
        "non_pass_keys": non_pass_keys,
    }
    return (
        _check(
            "catalog.current_deep_integrity_gate",
            passed,
            expected=(
                "current in-process deep verification with the complete required check inventory"
            ),
            actual=sanitized,
            detail=(
                "The production verifier re-hashed current source/blob bytes and reconciled the "
                "current catalog, manifests, library marker, and materialized blobs"
            ),
        ),
        sanitized,
    )


def _ordered_sql_identities(
    connection: Any,
    statement: str,
    parameters: Mapping[str, object] | None = None,
) -> Iterator[str]:
    result = connection.execute(text(statement), parameters or {})
    try:
        for row in result.yield_per(10_000):
            yield str(row[0])
    finally:
        result.close()


def _conversation_order(connection: Any, sort_name: str, direction: str) -> Iterator[str]:
    simple_sorts = {
        "title": "c.title_search",
        "type": "c.conversation_type",
        "source_category": "c.primary_category",
        "participants": "c.participant_count",
        "first_activity": "c.first_activity_ms",
        "last_activity": "c.last_activity_ms",
        "messages": "c.message_count",
        "sent": "c.sent_count",
        "received": "c.received_count",
        "attachments": "c.attachment_count",
        "unique_attachments": "c.unique_attachment_count",
        "attachment_bytes": "c.attachment_bytes",
        "active_days": "c.active_days",
        "completeness": "c.completeness",
    }
    if sort_name in {"photos", "videos", "audio"}:
        media_class = {"photos": "photo", "videos": "video", "audio": "audio"}[sort_name]
        expression = (
            "(SELECT count(*) FROM attachment_occurrences AS a "
            "WHERE a.conversation_id = c.id AND a.deleted_at IS NULL "
            f"AND ({_MEDIA_CLASS_SQL}) = :media_class)"
        )
        parameters: dict[str, object] = {"media_class": media_class}
    else:
        expression = simple_sorts[sort_name]
        parameters = {}
    sql_direction = "ASC" if direction == "asc" else "DESC"
    return _ordered_sql_identities(
        connection,
        f"""
            SELECT c.id
            FROM conversations AS c
            WHERE c.deleted_at IS NULL
            ORDER BY ({expression}) IS NULL ASC,
                     ({expression}) {sql_direction},
                     c.id {sql_direction}
            """,
        parameters,
    )


def _media_occurrence_order(connection: Any, sort_name: str, direction: str) -> Iterator[str]:
    sorts = {
        "date": "a.timestamp_ms",
        "size": "a.size_bytes",
        "type": f"({_MEDIA_CLASS_SQL})",
        "filename": "lower(a.filename)",
        "conversation": "lower(c.title_display)",
        "availability": "a.availability",
        "occurrences": "1",
    }
    expression = sorts[sort_name]
    sql_direction = "ASC" if direction == "asc" else "DESC"
    return _ordered_sql_identities(
        connection,
        f"""
            SELECT a.id
            FROM attachment_occurrences AS a
            LEFT JOIN conversations AS c ON c.id = a.conversation_id
            WHERE a.deleted_at IS NULL
            ORDER BY ({expression}) IS NULL ASC,
                     ({expression}) {sql_direction},
                     a.id {sql_direction}
            """,
    )


def _media_unique_order(connection: Any, sort_name: str, direction: str) -> Iterator[str]:
    sorts = {
        "date": "last_timestamp_ms",
        "size": "size_bytes",
        "type": "media_class",
        "filename": "lower(filename)",
        "conversation": "lower(conversation_title)",
        "availability": "availability",
        "occurrences": "occurrence_count",
    }
    expression = sorts[sort_name]
    sql_direction = "ASC" if direction == "asc" else "DESC"
    return _ordered_sql_identities(
        connection,
        f"""
            WITH ranked AS (
                SELECT
                    CASE WHEN a.blob_sha256 IS NOT NULL THEN 'blob:' || a.blob_sha256
                         ELSE 'occurrence:' || a.id END AS group_key,
                    a.filename,
                    a.size_bytes,
                    a.availability,
                    ({_MEDIA_CLASS_SQL}) AS media_class,
                    c.title_display AS conversation_title,
                    row_number() OVER (
                        PARTITION BY CASE
                            WHEN a.blob_sha256 IS NOT NULL THEN 'blob:' || a.blob_sha256
                            ELSE 'occurrence:' || a.id END
                        ORDER BY a.timestamp_ms IS NULL, a.timestamp_ms DESC, a.id
                    ) AS representative_rank,
                    count(*) OVER (
                        PARTITION BY CASE
                            WHEN a.blob_sha256 IS NOT NULL THEN 'blob:' || a.blob_sha256
                            ELSE 'occurrence:' || a.id END
                    ) AS occurrence_count,
                    max(a.timestamp_ms) OVER (
                        PARTITION BY CASE
                            WHEN a.blob_sha256 IS NOT NULL THEN 'blob:' || a.blob_sha256
                            ELSE 'occurrence:' || a.id END
                    ) AS last_timestamp_ms
                FROM attachment_occurrences AS a
                LEFT JOIN conversations AS c ON c.id = a.conversation_id
                WHERE a.deleted_at IS NULL
            )
            SELECT group_key
            FROM ranked
            WHERE representative_rank = 1
            ORDER BY ({expression}) IS NULL ASC,
                     ({expression}) {sql_direction},
                     group_key {sql_direction}
            """,
    )


def _stable_id(namespace: str, *parts: object) -> str:
    digest = hashlib.sha256(namespace.encode("utf-8"))
    for part in parts:
        digest.update(b"\x1f")
        digest.update(str(part).encode("utf-8", errors="surrogatepass"))
    return digest.hexdigest()


def _contact_order(connection: Any, sort_name: str, direction: str) -> tuple[Iterator[str], int]:
    occurrence_media_class_sql = _MEDIA_CLASS_SQL.replace("a.", "occurrence.")
    rows = _rows(
        connection,
        f"""
        WITH links AS (
            SELECT participant.name_search AS name_search,
                   min(participant.name_display) AS name_display,
                   conversation.id AS conversation_id,
                   conversation.conversation_type AS conversation_type,
                   conversation.message_count AS message_count,
                   conversation.last_activity_ms AS last_activity_ms
            FROM participants AS participant
            JOIN segment_participants AS membership
              ON membership.participant_id = participant.id
            JOIN conversation_segments AS segment ON segment.id = membership.segment_id
            JOIN conversations AS conversation ON conversation.id = segment.conversation_id
            WHERE participant.is_owner = 0
              AND conversation.deleted_at IS NULL
              AND conversation.conversation_type IN ('direct', 'group')
            GROUP BY participant.name_search, conversation.id
        ), attachment_counts AS (
            SELECT occurrence.conversation_id,
                   sum(CASE WHEN ({occurrence_media_class_sql}) = 'photo'
                            THEN 1 ELSE 0 END) AS photos,
                   sum(CASE WHEN ({occurrence_media_class_sql}) = 'video'
                            THEN 1 ELSE 0 END) AS videos,
                   sum(CASE WHEN ({occurrence_media_class_sql}) = 'audio'
                            THEN 1 ELSE 0 END) AS audio
            FROM attachment_occurrences AS occurrence
            WHERE occurrence.deleted_at IS NULL
              AND occurrence.message_id IS NOT NULL
            GROUP BY occurrence.conversation_id
        )
        SELECT links.name_search,
               sum(CASE WHEN links.conversation_type = 'direct'
                        THEN links.message_count ELSE 0 END) AS messages,
               sum(CASE WHEN links.conversation_type = 'direct'
                        THEN coalesce(attachment_counts.photos, 0) ELSE 0 END) AS photos,
               sum(CASE WHEN links.conversation_type = 'direct'
                        THEN coalesce(attachment_counts.videos, 0) ELSE 0 END) AS videos,
               sum(CASE WHEN links.conversation_type = 'direct'
                        THEN coalesce(attachment_counts.audio, 0) ELSE 0 END) AS audio,
               count(*) AS conversations,
               max(links.last_activity_ms) AS last_activity
        FROM links
        LEFT JOIN attachment_counts
          ON attachment_counts.conversation_id = links.conversation_id
        GROUP BY links.name_search
        """,
    )

    def key(row: Any) -> tuple[int, object, str]:
        value = row.name_search if sort_name == "name" else getattr(row, sort_name)
        identifier = _stable_id("messenger-contact", row.name_search)
        return (1 if value is None else 0, value if value is not None else 0, identifier)

    if direction == "desc":
        known = [row for row in rows if key(row)[0] == 0]
        unknown = [row for row in rows if key(row)[0] == 1]
        known.sort(key=key, reverse=True)
        unknown.sort(key=lambda row: _stable_id("messenger-contact", row.name_search), reverse=True)
        ordered = known + unknown
    else:
        ordered = sorted(rows, key=key)
    return (
        iter([_stable_id("messenger-contact", row.name_search) for row in ordered]),
        len(rows),
    )


def _activity_order(connection: Any, sort_name: str, direction: str) -> tuple[Iterator[str], int]:
    expressions = {
        "date": "activity.timestamp_ms",
        "title": "coalesce(activity.title_search, activity.content_search, '')",
        "actor": "coalesce(activity.actor_search, '')",
        "category": "activity.category",
        "subtype": "activity.subtype",
    }
    expression = expressions[sort_name]
    sql_direction = "ASC" if direction == "asc" else "DESC"
    total = int(
        _scalar(connection, "SELECT count(*) FROM activity_records WHERE deleted_at IS NULL")
    )
    identities = _ordered_sql_identities(
        connection,
        f"""
        SELECT activity.id
        FROM activity_records AS activity
        JOIN source_files AS source ON source.id = activity.source_file_id
        JOIN source_chunks AS chunk ON chunk.id = source.chunk_id
        WHERE activity.deleted_at IS NULL
        ORDER BY ({expression}) IS NULL ASC,
                 ({expression}) {sql_direction},
                 activity.id {sql_direction}
        """,
    )
    return identities, total


def _timeline_order(connection: Any, direction: str) -> tuple[Iterator[str], int]:
    sql_direction = "ASC" if direction == "asc" else "DESC"
    total = int(
        _scalar(connection, "SELECT count(*) FROM messages WHERE deleted_at IS NULL")
        + _scalar(connection, "SELECT count(*) FROM activity_records WHERE deleted_at IS NULL")
    )
    identities = _ordered_sql_identities(
        connection,
        f"""
        WITH entries AS (
            SELECT 'message' AS entity_type, id AS entity_id, timestamp_ms, source_file_id
            FROM messages WHERE deleted_at IS NULL
            UNION ALL
            SELECT 'activity' AS entity_type, id AS entity_id, timestamp_ms, source_file_id
            FROM activity_records WHERE deleted_at IS NULL
        )
        SELECT entries.entity_type || ':' || entries.entity_id AS identity
        FROM entries
        JOIN source_files AS source ON source.id = entries.source_file_id
        JOIN source_chunks AS chunk ON chunk.id = source.chunk_id
        ORDER BY entries.timestamp_ms IS NULL ASC,
                 entries.timestamp_ms {sql_direction},
                 identity {sql_direction}
        """,
    )
    return identities, total


_EVENT_STATUS_ALIASES = {
    "going": "went",
    "attended": "went",
    "events_you_attended": "went",
    "events_you_went_to": "went",
    "created_event": "created",
    "events_created_by_you": "created",
    "events_you_created": "created",
    "hosting": "hosted",
    "events_you_hosted": "hosted",
    "maybe": "interested",
    "events_you_were_interested_in": "interested",
    "events_you_were_invited_to": "invited",
    "events_you_declined": "declined",
    "cancelled_events": "cancelled",
    "not_going": "declined",
    "canceled": "cancelled",
}
_EVENT_STATUSES = {
    "hosted",
    "created",
    "went",
    "interested",
    "invited",
    "declined",
    "cancelled",
    "unknown",
}


def _event_status(extra_json: object, subtype: str) -> str:
    extra = extra_json
    if isinstance(extra, str):
        try:
            extra = json.loads(extra)
        except json.JSONDecodeError:
            extra = {}
    raw = extra.get("raw_payload") if isinstance(extra, dict) else None
    raw = raw if isinstance(raw, dict) else {}
    for candidate in (
        raw.get("event_response"),
        raw.get("rsvp_status"),
        raw.get("response"),
        raw.get("status"),
        subtype,
    ):
        if not isinstance(candidate, str):
            continue
        normalized = candidate.casefold().replace("-", "_").replace(" ", "_")
        normalized = _EVENT_STATUS_ALIASES.get(normalized, normalized)
        if normalized in _EVENT_STATUSES:
            return normalized
    return "unknown"


def _event_order(connection: Any, sort_name: str, direction: str) -> tuple[Iterator[str], int]:
    rows = _rows(
        connection,
        """
        SELECT activity.id, activity.timestamp_ms, activity.title_search,
               activity.subtype, activity.extra_json
        FROM activity_records AS activity
        JOIN source_files AS source ON source.id = activity.source_file_id
        WHERE activity.deleted_at IS NULL
          AND (activity.category = 'events'
               OR lower(source.normalized_path) LIKE '%/events/%')
        """,
    )
    reverse = direction == "desc"

    def value(row: Any) -> object | None:
        if sort_name == "date":
            timestamp_ms = row.timestamp_ms
            return (
                int(timestamp_ms)
                if isinstance(timestamp_ms, int) and not isinstance(timestamp_ms, bool)
                else None
            )
        if sort_name == "status":
            return _event_status(row.extra_json, str(row.subtype))
        return str(row.title_search or "")

    known = [row for row in rows if value(row) is not None]
    unknown = [row for row in rows if value(row) is None]
    known.sort(key=lambda row: (value(row), str(row.id)), reverse=reverse)
    unknown.sort(key=lambda row: str(row.id), reverse=reverse)
    ordered = known + unknown
    return iter([str(row.id) for row in ordered]), len(rows)


_RELATIONSHIP_PATH_PATTERNS = {
    "friends": ("%friends%", "%friendship%"),
    "requests": ("%friend%request%",),
    "followers": ("%followers%", "%following%"),
    "facebook-blocks": ("%facebook%block%", "%blocked_profiles%"),
    "messenger-blocks": ("%messenger%block%", "%blocked_messages%"),
}


def _relationship_order(
    connection: Any,
    view: str,
    sort_name: str,
    direction: str,
) -> tuple[Iterator[str], int]:
    expressions = {
        "name": "coalesce(activity.title_search, activity.actor_search, '')",
        "date": "activity.timestamp_ms",
        "status": "activity.subtype",
    }
    clauses = " OR ".join(
        f"lower(source.normalized_path) LIKE :pattern_{index}"
        for index, _pattern in enumerate(_RELATIONSHIP_PATH_PATTERNS[view])
    )
    parameters: dict[str, object] = {
        f"pattern_{index}": pattern
        for index, pattern in enumerate(_RELATIONSHIP_PATH_PATTERNS[view])
    }
    expression = expressions[sort_name]
    sql_direction = "ASC" if direction == "asc" else "DESC"
    total = int(
        _scalar(
            connection,
            f"""
            SELECT count(*)
            FROM activity_records AS activity
            JOIN source_files AS source ON source.id = activity.source_file_id
            WHERE activity.deleted_at IS NULL AND ({clauses})
            """,
            parameters,
        )
    )
    identities = _ordered_sql_identities(
        connection,
        f"""
        SELECT activity.id
        FROM activity_records AS activity
        JOIN source_files AS source ON source.id = activity.source_file_id
        JOIN source_chunks AS chunk ON chunk.id = source.chunk_id
        WHERE activity.deleted_at IS NULL AND ({clauses})
        ORDER BY ({expression}) IS NULL ASC,
                 ({expression}) {sql_direction},
                 activity.id {sql_direction}
        """,
        parameters,
    )
    return identities, total


def _first_mismatch(expected: Sequence[str], actual: Sequence[str]) -> dict[str, object] | None:
    for index, (expected_id, actual_id) in enumerate(zip(expected, actual, strict=False)):
        if expected_id != actual_id:
            return {
                "index": index,
                "expected_id_sha256": _opaque(expected_id),
                "actual_id_sha256": _opaque(actual_id),
            }
    if len(expected) != len(actual):
        return {"index": min(len(expected), len(actual)), "length_mismatch": True}
    return None


def _ordered_identity_digest(values: Sequence[str]) -> bytes:
    """Hash an ordered identity stream without ambiguous delimiter concatenation."""

    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8", errors="surrogatepass")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.digest()


def _update_ordered_identity_digest(digest: Any, value: str) -> None:
    encoded = value.encode("utf-8", errors="surrogatepass")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def _bounded_cursor_order_reconciliation(
    client: LocalApiClient,
    path: str,
    parameters: Mapping[str, object],
    *,
    identity_key: str,
    expected: Sequence[str] | Iterator[str],
    expected_total: int,
    page_cap: int | None = None,
) -> tuple[dict[str, object], list[float]]:
    """Reconcile a full or explicitly bounded API prefix against direct-SQL identity order."""

    if page_cap is not None and page_cap < 1:
        raise ValueError("page_cap must be positive when supplied")

    expected_iterator = iter(expected)
    issues: list[dict[str, object]] = []
    seen_cursors: set[str] = set()
    cursor: str | None = None
    timings: list[float] = []
    unavailable_coverage_observed = False
    continuation_observed = False
    expected_digest = hashlib.sha256()
    actual_digest = hashlib.sha256()
    expected_rows_seen = 0
    actual_rows_seen = 0
    first_mismatch: dict[str, object] | None = None
    previous_actual: str | None = None
    exhaustive = page_cap is None
    pages_required = max(1, math.ceil(expected_total / SORT_AUDIT_PAGE_SIZE))
    maximum_pages = pages_required + 1 if page_cap is None else min(pages_required, page_cap)

    for page_index in range(maximum_pages):
        request_parameters: dict[str, object] = {
            **parameters,
            "limit": SORT_AUDIT_PAGE_SIZE,
        }
        if cursor is not None:
            request_parameters["cursor"] = cursor
        try:
            payload, elapsed = client.get_json(path, request_parameters)
        except RuntimeError:
            issues.append({"kind": "api_request_failed", "page": page_index + 1})
            break
        timings.append(elapsed)
        page_number = page_index + 1

        raw_items = payload.get("items")
        items = _json_object_list(raw_items)
        if not isinstance(raw_items, list) or len(items) != len(raw_items):
            issues.append({"kind": "malformed_items", "page": page_number})
        if len(items) > SORT_AUDIT_PAGE_SIZE:
            issues.append(
                {
                    "kind": "page_limit_exceeded",
                    "page": page_number,
                    "row_count": len(items),
                }
            )

        page_ids: list[str] = []
        for item_index, item in enumerate(items):
            raw_identity = item.get(identity_key)
            if raw_identity is None or isinstance(raw_identity, (dict, list)):
                issues.append(
                    {
                        "kind": "missing_identity",
                        "page": page_number,
                        "index": item_index,
                    }
                )
                continue
            identity = str(raw_identity)
            if identity == previous_actual:
                issues.append(
                    {
                        "kind": "duplicate_identity",
                        "page": page_number,
                        "index": item_index,
                        "id_sha256": _opaque(identity),
                    }
                )
            previous_actual = identity
            page_ids.append(identity)
        for identity in page_ids:
            actual_rows_seen += 1
            _update_ordered_identity_digest(actual_digest, identity)
            try:
                expected_identity = next(expected_iterator)
            except StopIteration:
                expected_identity = None
            if expected_identity is not None:
                expected_rows_seen += 1
                _update_ordered_identity_digest(expected_digest, expected_identity)
            if first_mismatch is None and expected_identity != identity:
                first_mismatch = {
                    "index": actual_rows_seen - 1,
                    "expected_id_sha256": (
                        _opaque(expected_identity) if expected_identity is not None else None
                    ),
                    "actual_id_sha256": _opaque(identity),
                }

        page = _json_object(payload.get("page"))
        actual_total = page.get("total")
        count_is_unknown = page.get("count_is_unknown") is True
        coverage_status = _json_object(payload.get("coverage")).get("status")
        unavailable_coverage_observed = unavailable_coverage_observed or (
            expected_total == 0
            and ((actual_total is None and count_is_unknown) or coverage_status == "unavailable")
        )
        total_matches = (
            isinstance(actual_total, int)
            and not isinstance(actual_total, bool)
            and actual_total == expected_total
        ) or (expected_total == 0 and actual_total is None and count_is_unknown)
        if not total_matches:
            issues.append(
                {
                    "kind": "total_mismatch",
                    "page": page_number,
                    "expected_total": expected_total,
                    "actual_total": (
                        actual_total if isinstance(actual_total, int) else "unknown_or_malformed"
                    ),
                    "count_is_unknown": count_is_unknown,
                }
            )

        raw_next_cursor = page.get("next_cursor")
        next_cursor = (
            raw_next_cursor if isinstance(raw_next_cursor, str) and bool(raw_next_cursor) else None
        )
        if raw_next_cursor is not None and next_cursor is None:
            issues.append({"kind": "malformed_next_cursor", "page": page_number})
        if next_cursor is not None:
            continuation_observed = True
            if not items:
                issues.append({"kind": "empty_page_with_cursor", "page": page_number})
            if next_cursor in seen_cursors:
                issues.append({"kind": "cursor_cycle", "page": page_number})
                break
            seen_cursors.add(next_cursor)

        more_catalog_rows_expected = actual_rows_seen < expected_total
        if more_catalog_rows_expected and next_cursor is None:
            issues.append({"kind": "missing_continuation_cursor", "page": page_number})
            break
        if not more_catalog_rows_expected and next_cursor is not None:
            issues.append({"kind": "unexpected_terminal_cursor", "page": page_number})
            break
        cursor = next_cursor
        if cursor is None:
            break
    else:
        if exhaustive:
            issues.append({"kind": "page_bound_exceeded", "page_count": maximum_pages})

    full_stream = actual_rows_seen == expected_total and cursor is None
    try:
        if exhaustive or full_stream:
            for expected_identity in expected_iterator:
                expected_rows_seen += 1
                _update_ordered_identity_digest(expected_digest, expected_identity)
                if first_mismatch is None and expected_rows_seen > actual_rows_seen:
                    first_mismatch = {
                        "index": actual_rows_seen,
                        "expected_id_sha256": _opaque(expected_identity),
                        "actual_id_sha256": None,
                    }
    finally:
        close_expected = getattr(expected_iterator, "close", None)
        if callable(close_expected):
            close_expected()
    if full_stream and expected_rows_seen != expected_total:
        issues.append(
            {
                "kind": "direct_sql_stream_length",
                "expected_total": expected_total,
                "actual_rows": expected_rows_seen,
            }
        )
    if full_stream and actual_rows_seen != expected_total:
        issues.append(
            {
                "kind": "api_stream_length",
                "expected_total": expected_total,
                "actual_rows": actual_rows_seen,
            }
        )
    if not full_stream and expected_rows_seen != actual_rows_seen:
        issues.append(
            {
                "kind": "direct_sql_sample_length",
                "expected_rows": actual_rows_seen,
                "actual_rows": expected_rows_seen,
            }
        )
    if first_mismatch is not None:
        issues.append({"kind": "ordered_stream_mismatch", **first_mismatch})
    digest_matches = expected_rows_seen == actual_rows_seen and (
        expected_digest.digest() == actual_digest.digest()
    )
    if full_stream:
        digest_matches = digest_matches and actual_rows_seen == expected_total
    if expected_total == 0:
        coverage = "unavailable" if unavailable_coverage_observed else "empty"
    elif full_stream and expected_total <= SORT_AUDIT_PAGE_SIZE:
        coverage = "full_single_page"
    elif full_stream:
        coverage = "full_multi_page"
    else:
        coverage = "risk_bounded_edges"
    requested_rows = (
        expected_total if exhaustive else min(expected_total, maximum_pages * SORT_AUDIT_PAGE_SIZE)
    )
    summary: dict[str, object] = {
        "passed": not issues and digest_matches,
        "coverage": coverage,
        "expected_total": expected_total,
        "rows_checked": actual_rows_seen,
        "direct_sql_rows_checked": expected_rows_seen,
        "pages_checked": len(timings),
        "requested_pages": pages_required if exhaustive else maximum_pages,
        "requested_rows": requested_rows,
        "full_stream": full_stream,
        "risk_bounded": not exhaustive,
        "continuation_required": expected_total > SORT_AUDIT_PAGE_SIZE,
        "continuation_observed": continuation_observed,
        "cursor_boundaries_checked": max(0, len(timings) - 1),
        "outbound_cursor_observed": cursor is not None,
        "ordered_digest_matches": digest_matches,
        "issues": issues[:10],
        "issue_count": len(issues),
    }
    return summary, timings


def _sort_reconciliation_aggregate(
    results: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    def integer(result: Mapping[str, object], key: str) -> int:
        value = result.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) else 0

    return {
        "sort_direction_count": len(results),
        "requested_page_count": sum(integer(result, "requested_pages") for result in results),
        "api_page_count": sum(integer(result, "pages_checked") for result in results),
        "requested_row_identity_count": sum(
            integer(result, "requested_rows") for result in results
        ),
        "row_identity_count": sum(integer(result, "rows_checked") for result in results),
        "full_stream_count": sum(result.get("full_stream") is True for result in results),
        "risk_bounded_stream_count": sum(result.get("risk_bounded") is True for result in results),
        "all_streams_full": all(result.get("full_stream") is True for result in results),
        "large_stream_count": sum(
            integer(result, "expected_total") > RISK_SORT_PAGE_CAP * SORT_AUDIT_PAGE_SIZE
            for result in results
        ),
        "large_streams_fully_exhausted": all(
            result.get("full_stream") is True
            for result in results
            if integer(result, "expected_total") > RISK_SORT_PAGE_CAP * SORT_AUDIT_PAGE_SIZE
        ),
        "cursor_boundary_count": sum(
            integer(result, "cursor_boundaries_checked") for result in results
        ),
        "requested_cursor_boundary_count": sum(
            max(0, integer(result, "requested_pages") - 1) for result in results
        ),
        "continuation_required_count": sum(
            result.get("continuation_required") is True for result in results
        ),
        "continuation_observed_count": sum(
            result.get("continuation_observed") is True for result in results
        ),
        "continuation_proven_count": sum(
            result.get("continuation_required") is True
            and result.get("continuation_observed") is True
            and result.get("passed") is True
            for result in results
        ),
        "coverage_unavailable_count": sum(
            result.get("coverage") == "unavailable" for result in results
        ),
        "failed_count": sum(result.get("passed") is not True for result in results),
    }


def _api_reconciliation(
    connection: Any,
    client: LocalApiClient | None,
    totals: Mapping[str, int],
    *,
    policy: ReconciliationPolicy | None = None,
) -> tuple[list[AcceptanceCheck], dict[str, list[float]], dict[str, object]]:
    policy = policy or ReconciliationPolicy(mode="exhaustive", seed=DEFAULT_RECONCILIATION_SEED)
    checks: list[AcceptanceCheck] = []
    timings: dict[str, list[float]] = {"list_api": [], "date_jump_api": []}
    if client is None:
        checks.append(
            _pending(
                "api.reconciliation",
                "Pass --base-url and supply its token by prompt or owner-only token file",
            )
        )
        return (
            checks,
            timings,
            {
                "mode": policy.mode,
                "profile_version": policy.profile_version,
                "seed": policy.seed,
                "status": "unavailable",
            },
        )

    overview, elapsed = client.get_json("/api/v1/overview")
    timings["list_api"].append(elapsed)
    content = _json_object(overview.get("content"))
    overview_expected = {
        "conversation_count": totals["conversations"],
        "message_count": totals["messages"],
        "attachment_count": totals["attachments"],
        "activity_count": totals["activities"],
    }
    overview_actual = {key: content.get(key) for key in overview_expected}
    checks.append(
        _check(
            "api.overview_totals",
            overview_actual == overview_expected,
            expected=overview_expected,
            actual=overview_actual,
            detail="Overview API totals equal direct active-row SQLite counts",
        )
    )

    conversation_sorts = (
        "title",
        "type",
        "source_category",
        "participants",
        "first_activity",
        "last_activity",
        "messages",
        "sent",
        "received",
        "attachments",
        "unique_attachments",
        "attachment_bytes",
        "active_days",
        "completeness",
        "photos",
        "videos",
        "audio",
    )
    conversation_mismatches: list[dict[str, object]] = []
    conversation_results: list[dict[str, object]] = []
    for sort_name in conversation_sorts:
        for direction in ("asc", "desc"):
            expected = _conversation_order(
                connection,
                sort_name,
                direction,
            )
            result, elapsed_values = _bounded_cursor_order_reconciliation(
                client,
                "/api/v1/conversations",
                {"sort": sort_name, "direction": direction},
                identity_key="id",
                expected=expected,
                expected_total=totals["conversations"],
                page_cap=policy.sort_page_cap,
            )
            timings["list_api"].extend(elapsed_values)
            conversation_results.append(result)
            if result.get("passed") is not True:
                conversation_mismatches.append(
                    {"sort": sort_name, "direction": direction, **result}
                )
    checks.append(
        _check(
            "api.conversation_sorts",
            not conversation_mismatches,
            expected=(
                "17 sort fields in both directions match the direct-SQL identity order across "
                "the risk-bounded edge and cursor sample"
                if policy.mode == "risk-bounded"
                else "17 sort fields in both directions match the complete direct-SQL "
                "identity stream"
            ),
            actual={
                **_sort_reconciliation_aggregate(conversation_results),
                "mismatches": conversation_mismatches[:10],
            },
            detail=(
                "At most two edge pages per direction are sampled with exact totals, ordered "
                "prefix digests, cursor-boundary proof, and first-mismatch state; conversation "
                "IDs are never retained verbatim"
                if policy.mode == "risk-bounded"
                else "Every cursor page is exhausted with streaming digests and first-mismatch "
                "state; conversation IDs are never retained verbatim"
            ),
        )
    )

    media_mismatches: list[dict[str, object]] = []
    media_results: list[dict[str, object]] = []
    media_sorts = (
        "date",
        "size",
        "type",
        "filename",
        "conversation",
        "availability",
        "occurrences",
    )
    for view in ("occurrences", "unique"):
        expected_total = totals["attachments" if view == "occurrences" else "unique_media"]
        for sort_name in media_sorts:
            for direction in ("asc", "desc"):
                expected = (
                    _media_occurrence_order(
                        connection,
                        sort_name,
                        direction,
                    )
                    if view == "occurrences"
                    else _media_unique_order(
                        connection,
                        sort_name,
                        direction,
                    )
                )
                identity_key = "id" if view == "occurrences" else "view_key"
                result, elapsed_values = _bounded_cursor_order_reconciliation(
                    client,
                    "/api/v1/media",
                    {
                        "view": view,
                        "sort": sort_name,
                        "direction": direction,
                    },
                    identity_key=identity_key,
                    expected=expected,
                    expected_total=expected_total,
                    page_cap=policy.sort_page_cap,
                )
                timings["list_api"].extend(elapsed_values)
                media_results.append(result)
                if result.get("passed") is not True:
                    media_mismatches.append(
                        {
                            "view": view,
                            "sort": sort_name,
                            "direction": direction,
                            **result,
                        }
                    )
    checks.append(
        _check(
            "api.media_sorts",
            not media_mismatches,
            expected=(
                "7 sort fields in both directions and both views match direct-SQL identity order "
                "across the risk-bounded edge and cursor sample"
                if policy.mode == "risk-bounded"
                else "7 sort fields in both directions and both views match complete direct-SQL "
                "identity streams"
            ),
            actual={
                **_sort_reconciliation_aggregate(media_results),
                "mismatches": media_mismatches[:10],
            },
            detail=(
                "At most two edge pages per direction are sampled with exact totals, ordered "
                "prefix digests, cursor-boundary proof, and first-mismatch state; media identities "
                "and filenames are never retained verbatim"
                if policy.mode == "risk-bounded"
                else "Every cursor page is exhausted with streaming digests and first-mismatch "
                "state; media identities and filenames are never retained verbatim"
            ),
        )
    )

    stage4_mismatches: list[dict[str, object]] = []
    stage4_results: list[dict[str, object]] = []

    def reconcile_order(
        *,
        dashboard: str,
        sort_name: str,
        direction: str,
        expected: Sequence[str] | Iterator[str],
        expected_total: int,
        path: str,
        parameters: Mapping[str, object],
        identity_key: str = "id",
    ) -> None:
        result, elapsed_values = _bounded_cursor_order_reconciliation(
            client,
            path,
            parameters,
            identity_key=identity_key,
            expected=expected,
            expected_total=expected_total,
            page_cap=policy.sort_page_cap,
        )
        timings["list_api"].extend(elapsed_values)
        stage4_results.append(result)
        if result.get("passed") is not True:
            stage4_mismatches.append(
                {
                    "dashboard": dashboard,
                    "sort": sort_name,
                    "direction": direction,
                    **result,
                }
            )

    for sort_name in ("date", "title", "status"):
        for direction in ("asc", "desc"):
            expected, expected_total = _event_order(connection, sort_name, direction)
            reconcile_order(
                dashboard="events",
                sort_name=sort_name,
                direction=direction,
                expected=expected,
                expected_total=expected_total,
                path="/api/v1/events",
                parameters={"sort": sort_name, "direction": direction},
            )

    for sort_name in ("date", "title", "actor", "category", "subtype"):
        for direction in ("asc", "desc"):
            expected, expected_total = _activity_order(connection, sort_name, direction)
            reconcile_order(
                dashboard="activity",
                sort_name=sort_name,
                direction=direction,
                expected=expected,
                expected_total=expected_total,
                path="/api/v1/activity",
                parameters={"sort": sort_name, "direction": direction},
            )

    for direction in ("asc", "desc"):
        expected, expected_total = _timeline_order(connection, direction)
        reconcile_order(
            dashboard="timeline",
            sort_name="date",
            direction=direction,
            expected=expected,
            expected_total=expected_total,
            path="/api/v1/timeline",
            parameters={"direction": direction},
        )

    for sort_name in (
        "name",
        "messages",
        "photos",
        "videos",
        "audio",
        "conversations",
        "last_activity",
    ):
        for direction in ("asc", "desc"):
            expected, expected_total = _contact_order(connection, sort_name, direction)
            reconcile_order(
                dashboard="contacts",
                sort_name=sort_name,
                direction=direction,
                expected=expected,
                expected_total=expected_total,
                path="/api/v1/people",
                parameters={
                    "view": "contacts",
                    "sort": sort_name,
                    "direction": direction,
                },
            )

    for view in _RELATIONSHIP_PATH_PATTERNS:
        for sort_name in ("name", "date", "status"):
            for direction in ("asc", "desc"):
                expected, expected_total = _relationship_order(
                    connection,
                    view,
                    sort_name,
                    direction,
                )
                reconcile_order(
                    dashboard=f"people-{view}",
                    sort_name=sort_name,
                    direction=direction,
                    expected=expected,
                    expected_total=expected_total,
                    path="/api/v1/people",
                    parameters={
                        "view": view,
                        "sort": sort_name,
                        "direction": direction,
                    },
                )

    checks.append(
        _check(
            "api.events_people_activity_timeline_sorts",
            not stage4_mismatches,
            expected=(
                "Events, activity, timeline, Messenger contacts, and all exported relationship "
                "views match direct-SQL identity order across the risk-bounded edge and cursor "
                "sample in both directions"
                if policy.mode == "risk-bounded"
                else "Events, activity, timeline, Messenger contacts, and all exported "
                "relationship "
                "views match complete direct-SQL identity streams in both directions"
            ),
            actual={
                **_sort_reconciliation_aggregate(stage4_results),
                "mismatches": stage4_mismatches[:10],
            },
            detail=(
                "Unavailable export categories are explicit; at most two edge pages per direction "
                "are sampled and row identities remain transient with only opaque mismatch "
                "evidence retained"
                if policy.mode == "risk-bounded"
                else "Unavailable export categories are explicit; every cursor page is exhausted "
                "and row identities remain transient with only opaque mismatch evidence retained"
            ),
        )
    )

    simple_pages: tuple[tuple[str, str, dict[str, object], int], ...] = (
        ("events", "/api/v1/events", {}, totals["events"]),
        ("contacts", "/api/v1/people", {"view": "contacts"}, totals["contacts"]),
        ("activity", "/api/v1/activity", {}, totals["activities"]),
        ("timeline", "/api/v1/timeline", {}, totals["timeline"]),
    )
    page_results: dict[str, dict[str, object]] = {}
    for label, path, parameters, expected_total in simple_pages:
        payload, elapsed = client.get_json(path, {**parameters, "limit": 50})
        timings["list_api"].append(elapsed)
        page = _json_object(payload.get("page"))
        actual_total = page.get("total")
        count_is_unknown = page.get("count_is_unknown") is True
        passed = actual_total == expected_total or (
            expected_total == 0 and actual_total is None and count_is_unknown
        )
        page_results[label] = {
            "expected": expected_total,
            "actual": actual_total,
            "count_is_unknown": count_is_unknown,
            "passed": passed,
        }
    checks.append(
        _check(
            "api.events_people_activity_timeline_totals",
            all(bool(result["passed"]) for result in page_results.values()),
            expected="direct SQLite totals",
            actual=page_results,
            detail="Base Events, People, Activity, and Timeline dashboards reconcile",
        )
    )
    conversation_aggregate = _sort_reconciliation_aggregate(conversation_results)
    media_aggregate = _sort_reconciliation_aggregate(media_results)
    stage4_aggregate = _sort_reconciliation_aggregate(stage4_results)
    all_results = [*conversation_results, *media_results, *stage4_results]
    all_aggregate = _sort_reconciliation_aggregate(all_results)
    if policy.mode == "risk-bounded":
        strata_checks, strata_report, strata_timings = _risk_strata_acceptance(
            connection,
            client,
            policy=policy,
        )
        checks.extend(strata_checks)
        timings["list_api"].extend(strata_timings)
    else:
        strata_report = {
            "profile_version": policy.profile_version,
            "seed": policy.seed,
            "status": "not_required_in_exhaustive_mode",
            "complete": True,
        }
    expected_stream_count = 124
    profile_complete = bool(
        len(all_results) == expected_stream_count
        and all_aggregate["failed_count"] == 0
        and all_aggregate["requested_page_count"] == all_aggregate["api_page_count"]
        and all_aggregate["requested_row_identity_count"] == all_aggregate["row_identity_count"]
        and all_aggregate["requested_cursor_boundary_count"]
        == all_aggregate["cursor_boundary_count"]
        and (
            all_aggregate["risk_bounded_stream_count"] == expected_stream_count
            if policy.mode == "risk-bounded"
            else all_aggregate["all_streams_full"] is True
        )
        and strata_report.get("complete") is True
    )
    profile_specification = {
        "mode": policy.mode,
        "profile_version": policy.profile_version,
        "seed": policy.seed,
        "page_size": SORT_AUDIT_PAGE_SIZE,
        "page_cap": policy.sort_page_cap,
        "expected_stream_count": expected_stream_count,
        "direction_stream_counts": {"asc": 62, "desc": 62},
        "stratum_count": strata_report.get("stratum_count", 0),
    }
    profile_sha256 = hashlib.sha256(
        json.dumps(profile_specification, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    profile_report: dict[str, object] = {
        "mode": policy.mode,
        "profile_version": policy.profile_version,
        "seed": policy.seed,
        "profile_sha256": profile_sha256,
        "global_invariants_exact": True,
        "large_streams_fully_exhausted": all_aggregate["large_streams_fully_exhausted"],
        "page_size": SORT_AUDIT_PAGE_SIZE,
        "page_cap": policy.sort_page_cap,
        "expected_stream_count": expected_stream_count,
        "observed_stream_count": len(all_results),
        "direction_stream_counts": {"asc": 62, "desc": 62},
        "conversation": conversation_aggregate,
        "media": media_aggregate,
        "events_people_activity_timeline": stage4_aggregate,
        "all_streams": all_aggregate,
        "strata": strata_report,
        "reduced_assurance": policy.mode == "risk-bounded",
        "limitation": (
            "Large cursor streams are checked at both sorted edges and the first cursor boundary, "
            "not exhaustively page by page. Exact global totals and local integrity invariants "
            "remain mandatory."
            if policy.mode == "risk-bounded"
            else "All cursor streams are exhausted."
        ),
    }
    checks.append(
        _check(
            "api.reconciliation_profile_complete",
            profile_complete,
            expected={
                "mode": policy.mode,
                "profile_version": policy.profile_version,
                "stream_count": expected_stream_count,
                "requested_coverage_observed": True,
            },
            actual={
                "mode": policy.mode,
                "profile_version": policy.profile_version,
                "stream_count": len(all_results),
                "requested_page_count": all_aggregate["requested_page_count"],
                "observed_page_count": all_aggregate["api_page_count"],
                "requested_row_count": all_aggregate["requested_row_identity_count"],
                "observed_row_count": all_aggregate["row_identity_count"],
                "requested_cursor_boundary_count": all_aggregate["requested_cursor_boundary_count"],
                "observed_cursor_boundary_count": all_aggregate["cursor_boundary_count"],
                "failed_count": all_aggregate["failed_count"],
            },
            detail=(
                "The explicit reconciliation profile attempted every dashboard sort and both "
                "directions with exact requested-versus-observed aggregate evidence"
            ),
        )
    )
    return checks, timings, profile_report


def _deterministic_samples(
    connection: Any,
    *,
    table: str,
    pk_column: str,
    columns: str,
    where: str,
    count: int,
    seed: int,
) -> list[Any]:
    bounds = connection.execute(
        text(f"SELECT min({pk_column}), max({pk_column}), count(*) FROM {table} WHERE {where}")
    ).one()
    minimum, maximum, total = bounds
    if not total or minimum is None or maximum is None:
        return []
    desired = min(count, int(total))
    rng = random.Random(seed)
    candidates = sorted(
        {rng.randint(int(minimum), int(maximum)) for _ in range(max(count * 3, count))}
    )
    sampled: dict[object, Any] = {}
    for target in candidates:
        row = connection.execute(
            text(
                f"SELECT {columns} FROM {table} "
                f"WHERE {where} AND {pk_column} >= :target "
                f"ORDER BY {pk_column} LIMIT 1"
            ),
            {"target": target},
        ).first()
        if row is not None:
            sampled[row[0]] = row
        if len(sampled) >= desired:
            break
    # Sparse integer keys and repeated random targets must never silently reduce the requested
    # acceptance sample. Fill deterministically from the beginning with at most ``count`` small
    # keyset probes rather than scanning or randomly sorting the full real table.
    target = int(minimum)
    while len(sampled) < desired:
        row = connection.execute(
            text(
                f"SELECT {columns}, {pk_column} AS acceptance_sample_pk FROM {table} "
                f"WHERE {where} AND {pk_column} >= :target "
                f"ORDER BY {pk_column} LIMIT 1"
            ),
            {"target": target},
        ).first()
        if row is None:
            break
        sampled[row[0]] = row[:-1]
        target = int(row[-1]) + 1
    return list(sampled.values())[:desired]


def _risk_seed(master_seed: int, stratum: str) -> int:
    digest = hashlib.sha256(
        f"{RISK_RECONCILIATION_PROFILE_VERSION}\x1f{master_seed}\x1f{stratum}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _risk_stratum_result(
    *, eligible_count: int, observed_count: int, resolved_count: int
) -> dict[str, object]:
    requested_count = 1 if eligible_count else 0
    passed = observed_count == requested_count and resolved_count == requested_count
    return {
        "eligible_count": eligible_count,
        "requested_count": requested_count,
        "observed_count": observed_count,
        "resolved_count": resolved_count,
        "status": "pass" if passed else "fail",
    }


def _risk_message_jump(
    client: LocalApiClient,
    *,
    conversation_id: object,
    message_id: object,
    timestamp_ms: object,
    require_date_jump: bool,
    expected_date_message_id: object | None = None,
) -> tuple[bool, int]:
    requests = 0
    try:
        payload, _elapsed = client.get_json(
            f"/api/v1/conversations/{urllib.parse.quote(str(conversation_id))}/messages",
            {"around": message_id, "limit": 100},
        )
        requests += 1
        items = _json_object_list(payload.get("items"))
        focus = _json_object(payload.get("focus"))
        passed = any(item.get("id") == message_id for item in items) and (
            focus.get("message_id") == message_id
        )
        if require_date_jump:
            date_payload, _elapsed = client.get_json(
                f"/api/v1/conversations/{urllib.parse.quote(str(conversation_id))}/messages/date-jump",
                {"at_ms": timestamp_ms},
            )
            requests += 1
            jump = _json_object(date_payload.get("jump"))
            passed = (
                passed
                and jump.get("message_id") == expected_date_message_id
                and (date_payload.get("target_timestamp_ms") == timestamp_ms)
            )
        return passed, requests
    except RuntimeError:
        return False, requests


def _risk_activity_jump(client: LocalApiClient, *, activity_id: object) -> tuple[bool, int]:
    try:
        payload, _elapsed = client.get_json(
            "/api/v1/timeline",
            {"around": f"activity:{activity_id}", "limit": 50},
        )
    except RuntimeError:
        return False, 0
    around = _json_object(payload.get("around"))
    matched = next(
        (
            item
            for item in _json_object_list(payload.get("items"))
            if item.get("entity_type") == "activity" and item.get("entity_id") == activity_id
        ),
        None,
    )
    jump = _json_object(matched.get("jump")) if isinstance(matched, dict) else {}
    return bool(
        around.get("entity_id") == activity_id
        and matched is not None
        and jump.get("activity_id") == activity_id
        and jump.get("fragment") == f"activity-{activity_id}"
    ), 1


def _risk_media_jump(
    client: LocalApiClient,
    row: Any,
    *,
    expected_type: str | None,
) -> tuple[bool, int]:
    try:
        payload, _elapsed = client.get_json("/api/v1/media", {"occurrence": row.id, "limit": 1})
    except RuntimeError:
        return False, 0
    items = _json_object_list(payload.get("items"))
    item = items[0] if len(items) == 1 else {}
    jump = _json_object(item.get("jump")) or None
    expected_href = (
        f"/conversations/{row.conversation_id}?around={row.message_id}"
        f"&media={row.id}#message-{row.message_id}"
        if row.conversation_id is not None and row.message_id is not None
        else f"/conversations/{row.conversation_id}"
        if row.conversation_id is not None
        else f"/timeline?around=activity:{row.activity_id}#activity-{row.activity_id}"
        if row.activity_id is not None
        else None
    )
    expected_jump = expected_href is not None
    passed = bool(
        item.get("id") == row.id
        and item.get("availability") == row.availability
        and (expected_type is None or item.get("type") == expected_type)
        and bool(jump) is expected_jump
        and (
            jump is None
            or (
                jump.get("href") == expected_href
                and jump.get("conversation_id") == row.conversation_id
                and jump.get("message_id") == row.message_id
                and jump.get("activity_id") == row.activity_id
            )
        )
    )
    return passed, 1


def _risk_strata_acceptance(
    connection: Any,
    client: LocalApiClient,
    *,
    policy: ReconciliationPolicy,
) -> tuple[list[AcceptanceCheck], dict[str, object], list[float]]:
    """Exercise deterministic, represented source strata without retaining private values."""

    timings: list[float] = []
    thread_results: dict[str, dict[str, object]] = {}
    era_results: dict[str, dict[str, object]] = {}
    media_results: dict[str, dict[str, object]] = {}
    exact_jump_targets = {"message": 0, "activity": 0, "media": 0}

    thread_definitions: tuple[tuple[str, str, dict[str, object]], ...] = (
        (
            "direct",
            "c.conversation_type = 'direct'",
            {
                "conversation_type": "direct",
                "min_messages": 1,
                "sort": "messages",
                "direction": "desc",
            },
        ),
        (
            "group",
            "c.conversation_type = 'group'",
            {
                "conversation_type": "group",
                "min_messages": 1,
                "sort": "messages",
                "direction": "desc",
            },
        ),
        (
            "support",
            "EXISTS (SELECT 1 FROM conversation_segments AS segment "
            "WHERE segment.conversation_id = c.id "
            "AND segment.source_category = 'support')",
            {
                "source_category": "support",
                "min_messages": 1,
                "sort": "messages",
                "direction": "desc",
            },
        ),
        (
            "smallest_nonempty",
            "1 = 1",
            {"min_messages": 1, "sort": "messages", "direction": "asc"},
        ),
        (
            "largest",
            "1 = 1",
            {"min_messages": 1, "sort": "messages", "direction": "desc"},
        ),
    )
    for key, clause, api_parameters in thread_definitions:
        eligible_count = int(
            _scalar(
                connection,
                f"SELECT count(*) FROM conversations AS c WHERE c.deleted_at IS NULL "
                f"AND c.message_count >= 1 AND ({clause})",
            )
        )
        direction = str(api_parameters["direction"])
        sql_direction = "ASC" if direction == "asc" else "DESC"
        target = connection.execute(
            text(
                f"""
                SELECT c.id, c.message_count
                FROM conversations AS c
                WHERE c.deleted_at IS NULL AND c.message_count >= 1 AND ({clause})
                ORDER BY c.message_count {sql_direction}, c.id {sql_direction}
                LIMIT 1
                """
            )
        ).first()
        observed_count = 0
        resolved_count = 0
        try:
            payload, elapsed = client.get_json(
                "/api/v1/conversations", {**api_parameters, "limit": 1}
            )
            timings.append(elapsed)
            page = _json_object(payload.get("page"))
            items = _json_object_list(payload.get("items"))
            expected_id = target.id if target is not None else None
            list_resolved = page.get("total") == eligible_count and (
                (eligible_count == 0 and not items)
                or (len(items) == 1 and items[0].get("id") == expected_id)
            )
            observed_count = 1 if eligible_count and len(items) == 1 else 0
            if target is not None and list_resolved:
                message = connection.execute(
                    text(
                        """
                        SELECT id, conversation_id, timestamp_ms
                        FROM messages
                        WHERE conversation_id = :conversation_id AND deleted_at IS NULL
                        ORDER BY pk
                        LIMIT 1 OFFSET :offset
                        """
                    ),
                    {
                        "conversation_id": target.id,
                        "offset": _risk_seed(policy.seed, f"thread:{key}")
                        % int(target.message_count),
                    },
                ).first()
                jump_resolved, _requests = (
                    _risk_message_jump(
                        client,
                        conversation_id=message.conversation_id,
                        message_id=message.id,
                        timestamp_ms=message.timestamp_ms,
                        require_date_jump=False,
                    )
                    if message is not None
                    else (False, 0)
                )
                resolved_count = 1 if jump_resolved else 0
                exact_jump_targets["message"] += resolved_count
        except RuntimeError:
            pass
        thread_results[key] = _risk_stratum_result(
            eligible_count=eligible_count,
            observed_count=observed_count,
            resolved_count=resolved_count,
        )

    for entity_type, table_name in (("message", "messages"), ("activity", "activity_records")):
        total = int(
            _scalar(
                connection,
                f"SELECT count(*) FROM {table_name} WHERE deleted_at IS NULL "
                "AND timestamp_ms IS NOT NULL",
            )
        )
        era_offsets = {
            "early": 0,
            "middle": max(0, (total - 1) // 2),
            "late": max(0, total - 1),
        }
        for era, offset in era_offsets.items():
            target = connection.execute(
                text(
                    f"""
                    SELECT id, {"conversation_id," if entity_type == "message" else ""} timestamp_ms
                    FROM {table_name}
                    WHERE deleted_at IS NULL AND timestamp_ms IS NOT NULL
                    ORDER BY timestamp_ms, id
                    LIMIT 1 OFFSET :offset
                    """
                ),
                {"offset": offset},
            ).first()
            observed_count = 1 if target is not None else 0
            resolved = False
            if target is not None and entity_type == "message":
                expected_date_target = connection.execute(
                    text(
                        """
                        SELECT id
                        FROM messages
                        WHERE conversation_id = :conversation_id
                          AND deleted_at IS NULL
                          AND timestamp_ms >= :timestamp_ms
                        ORDER BY timestamp_ms, id
                        LIMIT 1
                        """
                    ),
                    {
                        "conversation_id": target.conversation_id,
                        "timestamp_ms": target.timestamp_ms,
                    },
                ).first()
                resolved, _requests = _risk_message_jump(
                    client,
                    conversation_id=target.conversation_id,
                    message_id=target.id,
                    timestamp_ms=target.timestamp_ms,
                    require_date_jump=True,
                    expected_date_message_id=(
                        expected_date_target.id if expected_date_target is not None else None
                    ),
                )
                exact_jump_targets["message"] += int(resolved)
            elif target is not None:
                resolved, _requests = _risk_activity_jump(client, activity_id=target.id)
                exact_jump_targets["activity"] += int(resolved)
            era_results[f"{entity_type}_{era}"] = _risk_stratum_result(
                eligible_count=total,
                observed_count=observed_count,
                resolved_count=int(resolved),
            )

    media_class_sql = _MEDIA_CLASS_SQL
    for dimension, values, expression in (
        (
            "type",
            ("photo", "video", "audio", "gif", "sticker", "document", "other"),
            media_class_sql,
        ),
        (
            "availability",
            ("resolved", "missing", "unlinked", "external"),
            "a.availability",
        ),
    ):
        for value in values:
            eligible_count = int(
                _scalar(
                    connection,
                    f"SELECT count(*) FROM attachment_occurrences AS a "
                    f"WHERE a.deleted_at IS NULL AND ({expression}) = :value",
                    {"value": value},
                )
            )
            offset = (
                _risk_seed(policy.seed, f"media:{dimension}:{value}") % eligible_count
                if eligible_count
                else 0
            )
            target = connection.execute(
                text(
                    f"""
                    SELECT a.id, a.conversation_id, a.message_id,
                           a.activity_record_id AS activity_id, a.availability
                    FROM attachment_occurrences AS a
                    WHERE a.deleted_at IS NULL AND ({expression}) = :value
                    ORDER BY a.id
                    LIMIT 1 OFFSET :offset
                    """
                ),
                {"value": value, "offset": offset},
            ).first()
            observed_count = 1 if target is not None else 0
            resolved = False
            if target is not None:
                resolved, _requests = _risk_media_jump(
                    client,
                    target,
                    expected_type=value if dimension == "type" else None,
                )
                exact_jump_targets["media"] += int(resolved)
            media_results[f"{dimension}_{value}"] = _risk_stratum_result(
                eligible_count=eligible_count,
                observed_count=observed_count,
                resolved_count=int(resolved),
            )

    all_results = [*thread_results.values(), *era_results.values(), *media_results.values()]
    complete = all(result.get("status") == "pass" for result in all_results)
    report: dict[str, object] = {
        "profile_version": policy.profile_version,
        "seed": policy.seed,
        "thread_strata": thread_results,
        "date_era_entity_strata": era_results,
        "media_strata": media_results,
        "exact_jump_targets": exact_jump_targets,
        "stratum_count": len(all_results),
        "represented_stratum_count": sum(
            isinstance(result.get("eligible_count"), int)
            and not isinstance(result.get("eligible_count"), bool)
            and cast(int, result.get("eligible_count")) > 0
            for result in all_results
        ),
        "resolved_stratum_count": sum(result.get("status") == "pass" for result in all_results),
        "complete": complete,
    }
    return (
        [
            _check(
                "api.risk_strata_complete",
                complete,
                expected=(
                    "every represented direct/group/support, smallest/largest, date-era entity, "
                    "media-type, and availability stratum resolves its exact source target"
                ),
                actual={
                    "stratum_count": report["stratum_count"],
                    "represented_stratum_count": report["represented_stratum_count"],
                    "resolved_stratum_count": report["resolved_stratum_count"],
                    "failed_count": sum(result.get("status") != "pass" for result in all_results),
                    "exact_jump_targets": exact_jump_targets,
                },
                detail=(
                    "Sampling is SHA-256-derived from the recorded seed; private IDs, timestamps, "
                    "names, titles, filenames, paths, cursors, and URLs are not retained"
                ),
            )
        ],
        report,
        timings,
    )


def _current_enrichment_item(
    client: LocalApiClient,
    *,
    path: str,
    target_id: str,
    completed_count: int,
) -> dict[str, Any] | None:
    """Find one current completed item without assuming it is first in API ordering."""

    page_size = 200
    for offset in range(0, completed_count, page_size):
        payload, _elapsed = client.get_json(
            path,
            {"status": "completed", "limit": page_size, "offset": offset},
        )
        items = _json_object_list(payload.get("items"))
        match = next((item for item in items if item.get("id") == target_id), None)
        if match is not None:
            return match
        if len(items) < page_size:
            break
    return None


def _jump_acceptance(
    connection: Any,
    client: LocalApiClient | None,
    *,
    sample_count: int,
    expected_largest_message_count: int | None,
) -> tuple[list[AcceptanceCheck], list[float], dict[str, object]]:
    if client is None:
        return (
            [
                _pending(
                    "api.jump_samples",
                    "Pass --base-url and supply its token by prompt or owner-only token file",
                )
            ],
            [],
            {},
        )
    checks: list[AcceptanceCheck] = []
    date_jump_timings: list[float] = []
    results: dict[str, object] = {}
    messages = _deterministic_samples(
        connection,
        table="messages",
        pk_column="pk",
        columns="id, conversation_id, timestamp_ms",
        where="deleted_at IS NULL",
        count=sample_count,
        seed=20260720,
    )
    message_failures: list[str] = []
    for message_id, conversation_id, _timestamp_ms in messages:
        payload, _elapsed = client.get_json(
            f"/api/v1/conversations/{urllib.parse.quote(str(conversation_id))}/messages",
            {"around": message_id, "limit": 100},
        )
        items = _json_object_list(payload.get("items"))
        focus = _json_object(payload.get("focus"))
        if (
            not any(item.get("id") == message_id for item in items)
            or focus.get("message_id") != message_id
        ):
            message_failures.append(_opaque(message_id))
    expected_message_samples = min(
        sample_count,
        int(_scalar(connection, "SELECT count(*) FROM messages WHERE deleted_at IS NULL")),
    )
    checks.append(
        _check(
            "api.message_jumps",
            not message_failures and len(messages) == expected_message_samples and bool(messages),
            expected={"sampled": expected_message_samples, "all_resolved": True},
            actual={"sampled": len(messages), "failed_id_sha256": message_failures[:10]},
            detail="Around jumps focus every deterministic active-message sample",
        )
    )

    dated_messages = _deterministic_samples(
        connection,
        table="messages",
        pk_column="pk",
        columns="id, conversation_id, timestamp_ms",
        where="deleted_at IS NULL AND timestamp_ms IS NOT NULL",
        count=sample_count,
        seed=20260721,
    )
    date_failures: list[str] = []
    for message_id, conversation_id, timestamp_ms in dated_messages:
        date_payload, elapsed = client.get_json(
            f"/api/v1/conversations/{urllib.parse.quote(str(conversation_id))}/messages/date-jump",
            {"at_ms": timestamp_ms},
        )
        date_jump_timings.append(elapsed)
        jump = _json_object(date_payload.get("jump"))
        expected_date_target = connection.execute(
            text(
                """
                SELECT id, timestamp_ms
                FROM messages
                WHERE conversation_id = :conversation_id
                  AND deleted_at IS NULL
                  AND timestamp_ms >= :timestamp_ms
                ORDER BY timestamp_ms, id
                LIMIT 1
                """
            ),
            {"conversation_id": conversation_id, "timestamp_ms": timestamp_ms},
        ).first()
        if (
            expected_date_target is None
            or jump.get("message_id") != expected_date_target[0]
            or date_payload.get("target_timestamp_ms") != expected_date_target[1]
        ):
            date_failures.append(_opaque(message_id))
    dated_total = int(
        _scalar(
            connection,
            "SELECT count(*) FROM messages WHERE deleted_at IS NULL AND timestamp_ms IS NOT NULL",
        )
    )
    expected_date_samples = min(sample_count, dated_total)
    checks.append(
        _check(
            "api.date_jumps",
            not date_failures
            and len(dated_messages) == expected_date_samples
            and bool(dated_messages),
            expected={"sampled": expected_date_samples, "all_resolved": True},
            actual={
                "sampled": len(dated_messages),
                "failed_id_sha256": date_failures[:10],
            },
            detail="Exact-timestamp jumps match the endpoint's deterministic SQLite ordering",
        )
    )

    attachment_results: dict[str, dict[str, object]] = {}
    for availability in ("resolved", "missing", "unlinked", "external"):
        row = connection.execute(
            text(
                """
                SELECT occurrence.id, occurrence.conversation_id, occurrence.message_id,
                       activity.id AS activity_id
                FROM attachment_occurrences AS occurrence
                LEFT JOIN activity_records AS activity
                  ON activity.id = occurrence.activity_record_id
                 AND activity.deleted_at IS NULL
                WHERE occurrence.deleted_at IS NULL
                  AND occurrence.availability = :availability
                ORDER BY occurrence.id
                LIMIT 1
                """
            ),
            {"availability": availability},
        ).first()
        if row is None:
            attachment_results[availability] = {"status": "not_present"}
            continue
        occurrence_id, conversation_id, message_id, activity_id = row
        payload, _elapsed = client.get_json(
            "/api/v1/media", {"occurrence": occurrence_id, "limit": 1}
        )
        items = _json_object_list(payload.get("items"))
        item = items[0] if items else {}
        attachment_jump = _json_object(item.get("jump")) or None
        expected_jump = conversation_id is not None or activity_id is not None
        expected_href = (
            f"/conversations/{conversation_id}?around={message_id}"
            f"&media={occurrence_id}#message-{message_id}"
            if conversation_id is not None and message_id is not None
            else f"/conversations/{conversation_id}"
            if conversation_id is not None
            else f"/timeline?around=activity:{activity_id}#activity-{activity_id}"
            if activity_id is not None
            else None
        )
        passed = (
            item.get("id") == occurrence_id
            and item.get("availability") == availability
            and bool(attachment_jump) is expected_jump
            and (
                attachment_jump is None
                or (
                    attachment_jump.get("conversation_id") == conversation_id
                    and attachment_jump.get("activity_id") == activity_id
                    and attachment_jump.get("href") == expected_href
                )
            )
            and (message_id is None or (attachment_jump or {}).get("message_id") == message_id)
            and (
                activity_id is None
                or (attachment_jump or {}).get("fragment") == f"activity-{activity_id}"
            )
        )
        attachment_results[availability] = {
            "status": "pass" if passed else "fail",
            "occurrence_id_sha256": _opaque(occurrence_id),
            "expected_jump": expected_jump,
        }
    checks.append(
        _check(
            "api.attachment_and_unavailable_media_jumps",
            all(
                result.get("status") in {"pass", "not_present"}
                for result in attachment_results.values()
            ),
            expected="Every represented availability class retains correct jump semantics",
            actual=attachment_results,
            detail="Missing and unlinked media remain visible without fabricating message anchors",
        )
    )

    activity_media_count = int(
        _scalar(
            connection,
            """
            SELECT count(*)
            FROM attachment_occurrences AS occurrence
            JOIN activity_records AS activity
              ON activity.id = occurrence.activity_record_id
            WHERE occurrence.deleted_at IS NULL
              AND activity.deleted_at IS NULL
              AND occurrence.conversation_id IS NULL
            """,
        )
    )
    activity_media_target = connection.execute(
        text(
            """
            SELECT occurrence.id AS occurrence_id,
                   occurrence.blob_sha256,
                   activity.id AS activity_id,
                   activity.title_display,
                   activity.category,
                   activity.subtype
            FROM attachment_occurrences AS occurrence
            JOIN activity_records AS activity
              ON activity.id = occurrence.activity_record_id
            WHERE occurrence.deleted_at IS NULL
              AND activity.deleted_at IS NULL
              AND occurrence.conversation_id IS NULL
            ORDER BY occurrence.blob_sha256 IS NULL, occurrence.id
            LIMIT 1
            """
        )
    ).first()
    activity_media_passed = activity_media_count == 0
    activity_media_evidence: dict[str, object] = {
        "activity_linked_occurrence_count": activity_media_count,
        "sampled": False,
    }
    if activity_media_target is not None:
        expected_activity_href = (
            f"/timeline?around=activity:{activity_media_target.activity_id}"
            f"#activity-{activity_media_target.activity_id}"
        )

        def activity_media_item_is_valid(item: Mapping[str, object]) -> bool:
            context = _json_object(item.get("activity"))
            jump = _json_object(item.get("jump"))
            return bool(
                item.get("id") == activity_media_target.occurrence_id
                and item.get("conversation") is None
                and item.get("availability") != "unlinked"
                and item.get("provenance") is None
                and item.get("source_category") == activity_media_target.category
                and context.get("id") == activity_media_target.activity_id
                and context.get("title") == activity_media_target.title_display
                and context.get("category") == activity_media_target.category
                and context.get("subtype") == activity_media_target.subtype
                and jump.get("activity_id") == activity_media_target.activity_id
                and jump.get("fragment") == f"activity-{activity_media_target.activity_id}"
                and jump.get("href") == expected_activity_href
                and jump.get("conversation_id") is None
                and jump.get("message_id") is None
            )

        occurrence_payload, _elapsed = client.get_json(
            "/api/v1/media",
            {
                "view": "occurrences",
                "occurrence": activity_media_target.occurrence_id,
                "limit": 1,
            },
        )
        unique_payload, _elapsed = client.get_json(
            "/api/v1/media",
            {
                "view": "unique",
                "occurrence": activity_media_target.occurrence_id,
                "limit": 1,
            },
        )
        occurrence_items = _json_object_list(occurrence_payload.get("items"))
        unique_items = _json_object_list(unique_payload.get("items"))
        occurrence_valid = len(occurrence_items) == 1 and activity_media_item_is_valid(
            occurrence_items[0]
        )
        unique_valid = len(unique_items) == 1 and activity_media_item_is_valid(unique_items[0])
        blob_required = isinstance(activity_media_target.blob_sha256, str)
        blob_valid = not blob_required
        if blob_required:
            blob_match: dict[str, Any] | None = None
            blob_cursor: str | None = None
            while True:
                blob_parameters: dict[str, object] = {"limit": 250}
                if blob_cursor is not None:
                    blob_parameters["cursor"] = blob_cursor
                try:
                    blob_payload, _elapsed = client.get_json(
                        f"/api/v1/media/blobs/{activity_media_target.blob_sha256}/occurrences",
                        blob_parameters,
                    )
                except RuntimeError:
                    # The request path contains a private blob digest. Record only a false gate;
                    # never allow that exception string to reach the sanitized acceptance report.
                    break
                blob_match = next(
                    (
                        item
                        for item in _json_object_list(blob_payload.get("items"))
                        if item.get("id") == activity_media_target.occurrence_id
                    ),
                    None,
                )
                if blob_match is not None:
                    break
                next_cursor = _json_object(blob_payload.get("page")).get("next_cursor")
                if not isinstance(next_cursor, str) or not next_cursor:
                    break
                blob_cursor = next_cursor
            blob_valid = isinstance(blob_match, dict) and activity_media_item_is_valid(blob_match)
        activity_media_passed = occurrence_valid and unique_valid and blob_valid
        activity_media_evidence = {
            "activity_linked_occurrence_count": activity_media_count,
            "sampled": True,
            "occurrence_id_sha256": _opaque(activity_media_target.occurrence_id),
            "activity_id_sha256": _opaque(activity_media_target.activity_id),
            "occurrence_view_resolved": occurrence_valid,
            "unique_view_resolved": unique_valid,
            "blob_occurrence_resolved": blob_valid,
            "blob_occurrence_required": blob_required,
        }
    checks.append(
        _check(
            "api.activity_linked_media_source_jumps",
            activity_media_passed,
            expected=(
                "a retained activity-linked occurrence exposes exact context and the canonical "
                "timeline jump in occurrence, unique, and represented blob views"
                if activity_media_count
                else "no retained activity-linked media; no source jump can be sampled"
            ),
            actual=activity_media_evidence,
            detail=(
                "Activity titles, categories, filenames, blob IDs, and jump URLs are inspected "
                "transiently but never emitted"
            ),
        )
    )

    activity_row = connection.execute(
        text(
            """
            SELECT id
            FROM activity_records
            WHERE deleted_at IS NULL AND lower(subtype) LIKE '%comment%'
            ORDER BY id
            LIMIT 1
            """
        )
    ).first()
    if activity_row is None:
        checks.append(
            _pending("api.comment_jump", "No active comment sample exists", required=False)
        )
    else:
        activity_id = str(activity_row[0])
        payload, _elapsed = client.get_json(
            "/api/v1/timeline", {"around": f"activity:{activity_id}", "limit": 50}
        )
        around = _json_object(payload.get("around"))
        items = _json_object_list(payload.get("items"))
        matched = next((item for item in items if item.get("entity_id") == activity_id), None)
        matched_jump = _json_object(matched.get("jump")) if isinstance(matched, dict) else {}
        passed = (
            around.get("entity_id") == activity_id
            and matched is not None
            and matched_jump.get("activity_id") == activity_id
            and matched_jump.get("fragment") == f"activity-{activity_id}"
        )
        checks.append(
            _check(
                "api.comment_jump",
                passed,
                expected="sampled comment activity is focused in the unified timeline",
                actual={"activity_id_sha256": _opaque(activity_id), "resolved": passed},
                detail="No comment title or content is retained in the report",
            )
        )

    transcription_count = int(
        _scalar(
            connection,
            """
            SELECT count(*)
            FROM audio_transcriptions
            WHERE pipeline_version = :pipeline_version
              AND model_name = :model_name
              AND model_revision = :model_revision
              AND status = 'completed'
            """,
            {
                "pipeline_version": TRANSCRIPTION_PIPELINE_VERSION,
                "model_name": TRANSCRIPTION_MODEL_NAME,
                "model_revision": TRANSCRIPTION_MODEL_REVISION,
            },
        )
    )
    transcription_jump_passed = transcription_count == 0
    transcription_jump_evidence: dict[str, object] = {
        "completed_transcription_count": transcription_count,
        "sampled": False,
    }
    transcription_target = connection.execute(
        text(
            """
            SELECT transcription.id AS transcription_id,
                   segment.id AS segment_id,
                   segment.start_ms,
                   occurrence.id AS occurrence_id,
                   occurrence.conversation_id,
                   occurrence.message_id,
                   activity.id AS activity_id
            FROM audio_transcriptions AS transcription
            JOIN audio_transcript_segments AS segment
              ON segment.transcription_id = transcription.id
            JOIN attachment_occurrences AS occurrence
              ON occurrence.blob_sha256 = transcription.blob_sha256
            LEFT JOIN activity_records AS activity
              ON activity.id = occurrence.activity_record_id
             AND activity.deleted_at IS NULL
            WHERE transcription.pipeline_version = :pipeline_version
              AND transcription.model_name = :model_name
              AND transcription.model_revision = :model_revision
              AND transcription.status = 'completed'
              AND occurrence.deleted_at IS NULL
            ORDER BY activity.id IS NULL, transcription.finished_at DESC, transcription.id,
                     segment.ordinal, occurrence.timestamp_ms IS NULL,
                     occurrence.timestamp_ms, occurrence.id
            LIMIT 1
            """
        ),
        {
            "pipeline_version": TRANSCRIPTION_PIPELINE_VERSION,
            "model_name": TRANSCRIPTION_MODEL_NAME,
            "model_revision": TRANSCRIPTION_MODEL_REVISION,
        },
    ).first()
    if transcription_target is not None:
        transcription = _current_enrichment_item(
            client,
            path="/api/v1/enrichment/transcriptions",
            target_id=str(transcription_target.transcription_id),
            completed_count=transcription_count,
        )
        segment = next(
            (
                item
                for item in _json_object_list((transcription or {}).get("segments"))
                if item.get("id") == transcription_target.segment_id
            ),
            None,
        )
        segment_jump = next(
            (
                jump
                for jump in _json_object_list((segment or {}).get("jumps"))
                if jump.get("occurrence_id") == transcription_target.occurrence_id
            )
            if isinstance(segment, dict)
            else iter(()),
            None,
        )
        start_ms = int(transcription_target.start_ms)
        expected_href = (
            f"/conversations/{transcription_target.conversation_id}"
            f"?around={transcription_target.message_id}"
            f"&media={transcription_target.occurrence_id}&t={start_ms}"
            f"#message-{transcription_target.message_id}"
            if transcription_target.conversation_id and transcription_target.message_id
            else f"/timeline?around=activity:{transcription_target.activity_id}"
            f"#activity-{transcription_target.activity_id}"
            if transcription_target.activity_id
            else f"/media?occurrence={transcription_target.occurrence_id}&t={start_ms}"
        )
        transcription_jump_passed = bool(
            isinstance(transcription, dict)
            and isinstance(segment, dict)
            and isinstance(segment_jump, dict)
            and segment.get("start_ms") == start_ms
            and segment_jump.get("start_ms") == start_ms
            and segment_jump.get("conversation_id") == transcription_target.conversation_id
            and segment_jump.get("message_id") == transcription_target.message_id
            and segment_jump.get("activity_id") == transcription_target.activity_id
            and segment_jump.get("href") == expected_href
        )
        transcription_jump_evidence = {
            "completed_transcription_count": transcription_count,
            "sampled": True,
            "transcription_id_sha256": _opaque(transcription_target.transcription_id),
            "segment_id_sha256": _opaque(transcription_target.segment_id),
            "occurrence_id_sha256": _opaque(transcription_target.occurrence_id),
            "timestamp_ms": start_ms,
            "resolved": transcription_jump_passed,
        }
    checks.append(
        _check(
            "api.audio_transcript_segment_jump",
            transcription_jump_passed,
            expected=(
                "a current completed segment links to an active occurrence with exact t= start"
                if transcription_count
                else "no completed current transcriptions; no segment jump required"
            ),
            actual=transcription_jump_evidence,
            detail="Transcript text and jump URL are inspected transiently but never emitted",
        )
    )

    extraction_count = int(
        _scalar(
            connection,
            """
            SELECT count(*)
            FROM attachment_text_extractions
            WHERE pipeline_version = :pipeline_version
              AND extractor_name = :extractor_name
              AND extractor_revision = :extractor_revision
              AND status = 'completed'
            """,
            {
                "pipeline_version": ATTACHMENT_TEXT_PIPELINE_VERSION,
                "extractor_name": ATTACHMENT_EXTRACTOR_NAME,
                "extractor_revision": ATTACHMENT_EXTRACTOR_REVISION,
            },
        )
    )
    extraction_jump_passed = extraction_count == 0
    extraction_jump_evidence: dict[str, object] = {
        "completed_extraction_count": extraction_count,
        "sampled": False,
    }
    extraction_target = connection.execute(
        text(
            """
            SELECT extraction.id AS extraction_id,
                   occurrence.id AS occurrence_id,
                   occurrence.conversation_id,
                   occurrence.message_id,
                   activity.id AS activity_id
            FROM attachment_text_extractions AS extraction
            JOIN attachment_occurrences AS occurrence
              ON occurrence.blob_sha256 = extraction.blob_sha256
            LEFT JOIN activity_records AS activity
              ON activity.id = occurrence.activity_record_id
             AND activity.deleted_at IS NULL
            WHERE extraction.pipeline_version = :pipeline_version
              AND extraction.extractor_name = :extractor_name
              AND extraction.extractor_revision = :extractor_revision
              AND extraction.status = 'completed'
              AND occurrence.deleted_at IS NULL
            ORDER BY activity.id IS NULL, extraction.finished_at DESC, extraction.id,
                     occurrence.timestamp_ms IS NULL, occurrence.timestamp_ms,
                     occurrence.id
            LIMIT 1
            """
        ),
        {
            "pipeline_version": ATTACHMENT_TEXT_PIPELINE_VERSION,
            "extractor_name": ATTACHMENT_EXTRACTOR_NAME,
            "extractor_revision": ATTACHMENT_EXTRACTOR_REVISION,
        },
    ).first()
    if extraction_target is not None:
        extraction = _current_enrichment_item(
            client,
            path="/api/v1/enrichment/attachment-text",
            target_id=str(extraction_target.extraction_id),
            completed_count=extraction_count,
        )
        extraction_jump = next(
            (
                jump
                for jump in _json_object_list((extraction or {}).get("jumps"))
                if jump.get("occurrence_id") == extraction_target.occurrence_id
            )
            if isinstance(extraction, dict)
            else iter(()),
            None,
        )
        expected_href = (
            f"/conversations/{extraction_target.conversation_id}"
            f"?around={extraction_target.message_id}"
            f"&media={extraction_target.occurrence_id}"
            f"#message-{extraction_target.message_id}"
            if extraction_target.conversation_id and extraction_target.message_id
            else f"/timeline?around=activity:{extraction_target.activity_id}"
            f"#activity-{extraction_target.activity_id}"
            if extraction_target.activity_id
            else f"/media?occurrence={extraction_target.occurrence_id}"
        )
        extraction_jump_passed = bool(
            isinstance(extraction, dict)
            and isinstance(extraction_jump, dict)
            and extraction_jump.get("conversation_id") == extraction_target.conversation_id
            and extraction_jump.get("message_id") == extraction_target.message_id
            and extraction_jump.get("activity_id") == extraction_target.activity_id
            and extraction_jump.get("href") == expected_href
        )
        extraction_jump_evidence = {
            "completed_extraction_count": extraction_count,
            "sampled": True,
            "extraction_id_sha256": _opaque(extraction_target.extraction_id),
            "occurrence_id_sha256": _opaque(extraction_target.occurrence_id),
            "resolved": extraction_jump_passed,
        }
    checks.append(
        _check(
            "api.attachment_text_occurrence_jump",
            extraction_jump_passed,
            expected=(
                "a current completed extraction links to an active source occurrence"
                if extraction_count
                else "no completed current extractions; no source jump required"
            ),
            actual=extraction_jump_evidence,
            detail="Extracted text, filenames, blob identifiers, and jump URLs are never emitted",
        )
    )

    largest = connection.execute(
        text(
            """
            SELECT id, message_count
            FROM conversations
            WHERE deleted_at IS NULL
            ORDER BY message_count DESC, id DESC
            LIMIT 1
            """
        )
    ).first()
    if largest is not None:
        conversation_id, message_count = largest
        payload, _elapsed = client.get_json(
            f"/api/v1/conversations/{urllib.parse.quote(str(conversation_id))}/messages",
            {"start": "latest", "limit": 250},
        )
        items = _json_object_list(payload.get("items"))
        page = _json_object(payload.get("page"))
        expected_largest = (
            int(message_count)
            if expected_largest_message_count is None
            else expected_largest_message_count
        )
        passed = (
            len(items) <= 250
            and page.get("total") == message_count
            and int(message_count) == expected_largest
        )
        checks.append(
            _check(
                "api.largest_conversation_bounded_page",
                passed,
                expected={"largest_message_count": expected_largest, "maximum_page_rows": 250},
                actual={"largest_message_count": int(message_count), "returned_rows": len(items)},
                detail=(
                    "API pagination is bounded; browser-report evidence separately checks DOM size"
                ),
            )
        )
        results["largest_conversation"] = {
            "id_sha256": _opaque(conversation_id),
            "message_count": int(message_count),
            "returned_rows": len(items),
        }
    return checks, date_jump_timings, results


def _expected_window_jump(connection: Any, window_id: str) -> dict[str, object] | None:
    row = connection.execute(
        text(
            """
            SELECT entity_type, conversation_id, first_message_id, activity_id,
                   attachment_occurrence_id, source_ids_json
            FROM search_windows
            WHERE id = :window_id
            """
        ),
        {"window_id": window_id},
    ).first()
    if row is None:
        return None
    source_ids = row.source_ids_json
    if isinstance(source_ids, str):
        try:
            source_ids = json.loads(source_ids)
        except json.JSONDecodeError:
            source_ids = {}
    source_ids = source_ids if isinstance(source_ids, dict) else {}
    if row.attachment_occurrence_id:
        occurrence = connection.execute(
            text(
                """
                SELECT conversation_id, message_id, activity_record_id
                FROM attachment_occurrences
                WHERE id = :occurrence_id AND deleted_at IS NULL
                """
            ),
            {"occurrence_id": row.attachment_occurrence_id},
        ).first()
        if occurrence is None:
            return None
        media_start = source_ids.get("media_start_ms")
        time_query = f"&t={media_start}" if isinstance(media_start, int) else ""
        if row.conversation_id and row.first_message_id:
            active_message = int(
                _scalar(
                    connection,
                    """
                    SELECT count(*) FROM messages
                    WHERE id = :message_id AND conversation_id = :conversation_id
                      AND deleted_at IS NULL
                    """,
                    {
                        "message_id": row.first_message_id,
                        "conversation_id": row.conversation_id,
                    },
                )
            )
            if not active_message:
                return None
            return {
                "kind": "conversation",
                "conversation_id": row.conversation_id,
                "message_id": row.first_message_id,
                "occurrence_id": row.attachment_occurrence_id,
                "fragment": f"message-{row.first_message_id}",
                "href": (
                    f"/conversations/{row.conversation_id}?around={row.first_message_id}"
                    f"&media={row.attachment_occurrence_id}{time_query}"
                    f"#message-{row.first_message_id}"
                ),
            }
        if row.activity_id:
            active_activity = int(
                _scalar(
                    connection,
                    """
                    SELECT count(*) FROM activity_records
                    WHERE id = :id AND deleted_at IS NULL
                    """,
                    {"id": row.activity_id},
                )
            )
            if not active_activity or occurrence.activity_record_id != row.activity_id:
                return None
            return {
                "kind": "activity",
                "activity_id": row.activity_id,
                "occurrence_id": row.attachment_occurrence_id,
                "fragment": f"activity-{row.activity_id}",
                "href": (f"/timeline?around=activity:{row.activity_id}#activity-{row.activity_id}"),
            }
        return {
            "kind": "media",
            "occurrence_id": row.attachment_occurrence_id,
            "fragment": f"media-{row.attachment_occurrence_id}",
            "href": f"/media?occurrence={row.attachment_occurrence_id}{time_query}",
        }
    if row.conversation_id and row.first_message_id:
        active_message = int(
            _scalar(
                connection,
                """
                SELECT count(*) FROM messages
                WHERE id = :message_id AND conversation_id = :conversation_id
                  AND deleted_at IS NULL
                """,
                {
                    "message_id": row.first_message_id,
                    "conversation_id": row.conversation_id,
                },
            )
        )
        if not active_message:
            return None
        return {
            "kind": "conversation",
            "conversation_id": row.conversation_id,
            "message_id": row.first_message_id,
            "fragment": f"message-{row.first_message_id}",
            "href": (
                f"/conversations/{row.conversation_id}?around={row.first_message_id}"
                f"#message-{row.first_message_id}"
            ),
        }
    if row.activity_id:
        active_activity = int(
            _scalar(
                connection,
                "SELECT count(*) FROM activity_records WHERE id = :id AND deleted_at IS NULL",
                {"id": row.activity_id},
            )
        )
        if not active_activity:
            return None
        return {
            "kind": "activity",
            "activity_id": row.activity_id,
            "fragment": f"activity-{row.activity_id}",
            "href": (f"/timeline?around=activity:{row.activity_id}#activity-{row.activity_id}"),
        }
    return None


def _search_result_jump_is_valid(connection: Any, item: Mapping[str, object]) -> bool:
    jump = _json_object(item.get("jump"))
    if not jump or not isinstance(jump.get("href"), str) or not jump["href"]:
        return False
    window_id = item.get("window_id")
    if isinstance(window_id, str) and window_id:
        expected = _expected_window_jump(connection, window_id)
        return expected is not None and all(
            jump.get(key) == value for key, value in expected.items()
        )

    entity_type = item.get("entity_type")
    source_id = item.get("source_id")
    if not isinstance(source_id, str) or not source_id:
        return False
    if entity_type == "message":
        row = connection.execute(
            text(
                """
                SELECT conversation_id FROM messages
                WHERE id = :id AND deleted_at IS NULL
                """
            ),
            {"id": source_id},
        ).first()
        expected = (
            {
                "kind": "conversation",
                "conversation_id": row.conversation_id,
                "message_id": source_id,
                "fragment": f"message-{source_id}",
                "href": (
                    f"/conversations/{row.conversation_id}?around={source_id}#message-{source_id}"
                ),
            }
            if row is not None
            else None
        )
    elif entity_type == "activity":
        exists = int(
            _scalar(
                connection,
                "SELECT count(*) FROM activity_records WHERE id = :id AND deleted_at IS NULL",
                {"id": source_id},
            )
        )
        expected = (
            {
                "kind": "activity",
                "activity_id": source_id,
                "fragment": f"activity-{source_id}",
                "href": f"/timeline?around=activity:{source_id}#activity-{source_id}",
            }
            if exists
            else None
        )
    elif entity_type in {"audio_transcript", "attachment_text"}:
        occurrence_id = jump.get("occurrence_id")
        if not isinstance(occurrence_id, str):
            return False
        occurrence = connection.execute(
            text(
                """
                SELECT occurrence.conversation_id, occurrence.message_id,
                       activity.id AS activity_id
                FROM attachment_occurrences AS occurrence
                LEFT JOIN activity_records AS activity
                  ON activity.id = occurrence.activity_record_id
                 AND activity.deleted_at IS NULL
                WHERE occurrence.id = :id AND occurrence.deleted_at IS NULL
                """
            ),
            {"id": occurrence_id},
        ).first()
        if occurrence is None:
            return False
        timing = _json_object(item.get("timing"))
        start_ms = timing.get("start_ms") if entity_type == "audio_transcript" else None
        time_query = f"&t={start_ms}" if isinstance(start_ms, int) else ""
        href = (
            f"/conversations/{occurrence.conversation_id}?around={occurrence.message_id}"
            f"&media={occurrence_id}{time_query}#message-{occurrence.message_id}"
            if occurrence.conversation_id and occurrence.message_id
            else f"/timeline?around=activity:{occurrence.activity_id}"
            f"#activity-{occurrence.activity_id}"
            if occurrence.activity_id
            else f"/media?occurrence={occurrence_id}{time_query}"
        )
        expected = (
            {
                "kind": "conversation",
                "conversation_id": occurrence.conversation_id,
                "message_id": occurrence.message_id,
                "occurrence_id": occurrence_id,
                "fragment": f"message-{occurrence.message_id}",
                "href": href,
            }
            if occurrence.conversation_id and occurrence.message_id
            else {
                "kind": "activity",
                "activity_id": occurrence.activity_id,
                "occurrence_id": occurrence_id,
                "fragment": f"activity-{occurrence.activity_id}",
                "href": href,
            }
            if occurrence.activity_id
            else {
                "kind": "media",
                "occurrence_id": occurrence_id,
                "fragment": f"media-{occurrence_id}",
                "href": href,
            }
        )
    else:
        return False
    return expected is not None and all(jump.get(key) == value for key, value in expected.items())


def _search_config_is_current(payload: object, active_config: Any | None) -> bool:
    config = _json_object(payload)
    reranker = _json_object(config.get("reranker"))
    matching_specs = [
        spec
        for spec in SEMANTIC_MODEL_SPECS.values()
        if active_config is not None
        and spec.model_name == active_config.model_name
        and spec.model_revision == active_config.model_revision
    ]
    return bool(
        active_config is not None
        and len(matching_specs) == 1
        and config.get("id") == active_config.id
        and config.get("model_name") == active_config.model_name
        and config.get("model_revision") == active_config.model_revision
        and config.get("window_version") == active_config.window_version
        and isinstance(config.get("artifact_fingerprint"), str)
        and bool(config["artifact_fingerprint"])
        and config.get("embedding_implementation")
        == embedding_implementation_identity(matching_specs[0])
        and reranker.get("model_name") == DEFAULT_RERANKER_SPEC.model_name
        and reranker.get("model_revision") == DEFAULT_RERANKER_SPEC.model_revision
        and reranker.get("device") == DEFAULT_RERANKER_SPEC.device
        and reranker.get("score_kind") == DEFAULT_RERANKER_SPEC.score_kind
        and reranker.get("max_candidates") == DEFAULT_RERANKER_SPEC.max_candidates
        and reranker.get("batch_size") == DEFAULT_RERANKER_SPEC.batch_size
        and reranker.get("max_sequence_tokens") == DEFAULT_RERANKER_SPEC.max_sequence_tokens
        and reranker.get("status") not in {None, "not_loaded", "unavailable"}
    )


def _active_membership_count(
    connection: Any,
    config_id: str | None,
    items: Sequence[Mapping[str, object]],
) -> tuple[int, int, int]:
    """Return result-window, unique-window, and active-membership counts."""

    result_window_ids = [
        str(item["window_id"])
        for item in items
        if isinstance(item.get("window_id"), str) and bool(item["window_id"])
    ]
    window_ids = set(result_window_ids)
    if config_id is None or not window_ids:
        return len(result_window_ids), len(window_ids), 0
    active_members = 0
    ordered_ids = sorted(window_ids)
    for start in range(0, len(ordered_ids), 400):
        chunk = ordered_ids[start : start + 400]
        placeholders = ", ".join(f":window_{index}" for index in range(len(chunk)))
        parameters: dict[str, object] = {
            f"window_{index}": window_id for index, window_id in enumerate(chunk)
        }
        parameters["config_id"] = config_id
        active_members += int(
            _scalar(
                connection,
                f"""
                SELECT count(*)
                FROM semantic_window_embeddings
                WHERE config_id = :config_id
                  AND window_id IN ({placeholders})
                """,
                parameters,
            )
        )
    return len(result_window_ids), len(window_ids), active_members


def _direct_conflict_public_ranker_metadata_valid(value: object) -> bool:
    """Validate the complete non-private LinearSVC ranker identity and audit summary."""

    ranker = _json_object(value)
    parameters = _json_object(ranker.get("parameters"))
    counts = _json_object(ranker.get("counts"))
    metrics = _json_object(ranker.get("metrics"))
    if (
        set(ranker)
        != {
            "version",
            "parameters",
            "counts",
            "metrics",
            "score_kind",
            "ranking_feature_only",
        }
        or ranker.get("version") != DIRECT_CONFLICT_RANKER_VERSION
        or ranker.get("score_kind") != DIRECT_CONFLICT_SCORE_KIND
        or ranker.get("ranking_feature_only") is not True
        or parameters != asdict(DEFAULT_DIRECT_CONFLICT_PARAMETERS)
        or set(counts)
        != {
            "reviewed_feedback_rows",
            "binary_training_rows",
            "relevant_rows",
            "not_relevant_rows",
            "uncertain_rows",
            "conversation_groups",
        }
        or set(metrics) != {"fold_count", *DIRECT_CONFLICT_PUBLIC_METRICS}
    ):
        return False
    if not all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in counts.values()
    ):
        return False
    reviewed = int(counts["reviewed_feedback_rows"])
    binary = int(counts["binary_training_rows"])
    relevant = int(counts["relevant_rows"])
    not_relevant = int(counts["not_relevant_rows"])
    uncertain = int(counts["uncertain_rows"])
    groups = int(counts["conversation_groups"])
    if not (
        reviewed == binary + uncertain
        and binary == relevant + not_relevant
        and relevant >= DIRECT_CONFLICT_DEVELOPMENT_MIN_RELEVANT_COUNT
        and not_relevant >= DIRECT_CONFLICT_DEVELOPMENT_MIN_NOT_RELEVANT_COUNT
        and groups >= DIRECT_CONFLICT_DEVELOPMENT_FOLD_COUNT
        and metrics.get("fold_count") == DIRECT_CONFLICT_DEVELOPMENT_FOLD_COUNT
    ):
        return False
    for name in DIRECT_CONFLICT_PUBLIC_METRICS:
        summary = _json_object(metrics.get(name))
        if set(summary) != {"mean", "minimum"}:
            return False
        mean = summary.get("mean")
        minimum = summary.get("minimum")
        if not (
            isinstance(mean, (int, float))
            and not isinstance(mean, bool)
            and math.isfinite(float(mean))
            and isinstance(minimum, (int, float))
            and not isinstance(minimum, bool)
            and math.isfinite(float(minimum))
            and 0.0 <= float(minimum) <= float(mean) <= 1.0
        ):
            return False
    return True


def _direct_conflict_ranker_matches_current_certification(
    connection: Any,
    active_config: Any,
    value: object,
) -> bool:
    """Bind public ranker provenance to the accepted calibration and current feedback."""

    if active_config is None or not _direct_conflict_public_ranker_metadata_valid(value):
        return False
    try:
        ranker = _json_object(value)
        audit = _database_json_object(active_config.audit_json) or {}
        accepted = _json_object(audit.get("accepted_semantic_ir_evaluation"))
        development = _json_object(accepted.get("direct_conflict_development_calibration"))
        model_metadata = _json_object(development.get("model_metadata"))
        grouped = _json_object(development.get("grouped_cross_validation"))
        minimum_metrics = _json_object(grouped.get("minimum_metrics"))
        mean_metrics = _json_object(grouped.get("mean_metrics"))
        current_revision = development_feedback_revision(connection.engine)
        expected_counts = {
            "reviewed_feedback_rows": development.get("reviewed_judgment_count"),
            "binary_training_rows": development.get("binary_judgment_count"),
            "relevant_rows": development.get("relevant_count"),
            "not_relevant_rows": development.get("not_relevant_count"),
            "uncertain_rows": development.get("uncertain_count"),
            "conversation_groups": development.get(
                "conversation_group_count", development.get("conversation_count")
            ),
        }
        expected_metrics: dict[str, object] = {"fold_count": development.get("fold_count")}
        expected_metrics.update(
            {
                name: {
                    "mean": mean_metrics.get(name),
                    "minimum": minimum_metrics.get(name),
                }
                for name in DIRECT_CONFLICT_PUBLIC_METRICS
            }
        )
        return bool(
            accepted.get("production_acceptance_passed") is True
            and development.get("required_for_production_acceptance") is True
            and development.get("status") == "passed"
            and development.get("passed") is True
            and development.get("ranking_pipeline") == DIRECT_CONFLICT_RANKING_PIPELINE_VERSION
            and development.get("feedback_revision_stable_during_evaluation") is True
            and development.get("feedback_revision") == current_revision
            and model_metadata.get("classifier") == DIRECT_CONFLICT_CLASSIFIER
            and model_metadata.get("ranker_version") == DIRECT_CONFLICT_RANKER_VERSION
            and model_metadata.get("score_kind") == DIRECT_CONFLICT_SCORE_KIND
            and _json_object(model_metadata.get("parameters"))
            == asdict(DEFAULT_DIRECT_CONFLICT_PARAMETERS)
            and ranker.get("counts") == expected_counts
            and ranker.get("metrics") == expected_metrics
        )
    except Exception:
        # Feedback rows and audit details are private. Any malformed or stale state fails closed.
        return False


def _search_user_story_acceptance(
    connection: Any,
    client: LocalApiClient | None,
    *,
    query: str,
) -> tuple[list[AcceptanceCheck], dict[str, object]]:
    if client is None:
        return (
            [
                _pending(
                    "api.semantic_hybrid_search_user_story",
                    "Pass --base-url to exercise semantic, hybrid, and conflict search",
                ),
                _pending(
                    "api.conflict_search_user_story",
                    "Pass --base-url to exercise the pinned six-label conflict path",
                ),
            ],
            {},
        )
    active_config = connection.execute(
        text(
            """
            SELECT id, model_name, model_revision, window_version, audit_json
            FROM semantic_index_configs
            WHERE active = 1 AND status = 'ready'
            ORDER BY id
            """
        )
    ).first()
    active_config_id = str(active_config.id) if active_config is not None else None
    mode_report: dict[str, object] = {}
    mode_passes: list[bool] = []
    for mode in ("semantic", "hybrid"):
        try:
            payload, _elapsed = client.get_json(
                "/api/v1/search", {"q": query, "mode": mode, "limit": 50}
            )
        except RuntimeError:
            mode_report[mode] = {"available": False, "returned": 0, "valid_jumps": 0}
            mode_passes.append(False)
            continue
        items = _json_object_list(payload.get("items"))
        page = _json_object(payload.get("page"))
        valid_jumps = sum(_search_result_jump_is_valid(connection, item) for item in items)
        result_window_count, unique_window_count, active_member_count = _active_membership_count(
            connection, active_config_id, items
        )
        literal_direct_conflict = mode == "semantic" and is_direct_conflict_trigger(query)
        expected_score_kind = (
            DIRECT_CONFLICT_SCORE_KIND
            if literal_direct_conflict
            else DEFAULT_RERANKER_SPEC.score_kind
            if mode == "semantic"
            else "rrf"
        )
        item_contracts_valid = all(
            isinstance(item.get("id"), str)
            and item.get("rank") == index
            and isinstance(item.get("score"), (int, float))
            and not isinstance(item.get("score"), bool)
            and math.isfinite(float(item["score"]))
            and item.get("score_kind") == expected_score_kind
            and item.get("deletion_state") == "retained"
            and _json_object(item.get("provenance")).get("source_id") == item.get("source_id")
            for index, item in enumerate(items, start=1)
        )
        semantic_provenance_valid = True
        if mode == "semantic":
            if literal_direct_conflict:
                index_config = _json_object(payload.get("index_config"))
                configured_ranker = index_config.get("direct_conflict_ranker")
                semantic_provenance_valid = bool(
                    index_config.get("ranking_pipeline") == DIRECT_CONFLICT_RANKING_PIPELINE_VERSION
                    and _direct_conflict_ranker_matches_current_certification(
                        connection, active_config, configured_ranker
                    )
                    and all(
                        _json_object(item.get("ranking_provenance")).get("pipeline")
                        == DIRECT_CONFLICT_LITERAL_PIPELINE
                        and _json_object(item.get("ranking_provenance")).get("ranking_pipeline")
                        == DIRECT_CONFLICT_RANKING_PIPELINE_VERSION
                        and _json_object(item.get("ranking_provenance")).get("trigger")
                        == DIRECT_CONFLICT_LITERAL_TRIGGER
                        and _json_object(item.get("ranking_provenance")).get("ranker")
                        == configured_ranker
                        and _direct_conflict_public_ranker_metadata_valid(
                            _json_object(item.get("ranking_provenance")).get("ranker")
                        )
                        and isinstance(item.get("window_id"), str)
                        and bool(item["window_id"])
                        for item in items
                    )
                )
            else:
                semantic_provenance_valid = all(
                    _json_object(item.get("ranking_provenance")).get("pipeline")
                    == "cosine_retrieval_then_local_cross_encoder"
                    and _json_object(
                        _json_object(item.get("ranking_provenance")).get("reranker")
                    ).get("model_revision")
                    == DEFAULT_RERANKER_SPEC.model_revision
                    and isinstance(item.get("window_id"), str)
                    and bool(item["window_id"])
                    for item in items
                )
        else:
            semantic_provenance_valid = all(
                not isinstance(item.get("window_id"), str)
                or (
                    _json_object(item.get("ranking_provenance")).get("pipeline")
                    == "reciprocal_rank_fusion"
                    and _json_object(
                        _json_object(item.get("ranking_provenance")).get("semantic_reranker")
                    ).get("model_revision")
                    == DEFAULT_RERANKER_SPEC.model_revision
                )
                for item in items
            )
        candidate_count = page.get("candidate_count")
        passed = bool(
            len(items) > 0
            and len(items) <= 50
            and len({str(item.get("id")) for item in items}) == len(items)
            and payload.get("query") == query
            and payload.get("mode") == mode
            and page.get("returned") == len(items)
            and page.get("limit") == 50
            and page.get("result_cap") == 1_000
            and isinstance(candidate_count, int)
            and not isinstance(candidate_count, bool)
            and candidate_count >= len(items)
            and _search_config_is_current(payload.get("index_config"), active_config)
            and item_contracts_valid
            and semantic_provenance_valid
            and result_window_count == unique_window_count
            and active_member_count == unique_window_count
            and (
                unique_window_count == len(items) if mode == "semantic" else unique_window_count > 0
            )
            and valid_jumps == len(items)
        )
        mode_passes.append(passed)
        mode_report[mode] = {
            "available": True,
            "returned": len(items),
            "candidate_count": candidate_count,
            "valid_jumps": valid_jumps,
            "result_windows": result_window_count,
            "unique_result_windows": unique_window_count,
            "active_member_windows": active_member_count,
            "current_config": _search_config_is_current(payload.get("index_config"), active_config),
            "contract_valid": item_contracts_valid and semantic_provenance_valid,
        }

    search_check = _check(
        "api.semantic_hybrid_search_user_story",
        len(mode_passes) == 2 and all(mode_passes),
        expected=(
            "semantic and hybrid top-50 return current, pinned, nonempty results with valid "
            "source jumps"
        ),
        actual=mode_report,
        detail="Query text, excerpts, people, titles, source IDs, and jump URLs are never emitted",
    )

    conflict_report: dict[str, object]
    conflict_passed = False
    try:
        conflict_payload, _elapsed = client.get_json(
            "/api/v1/enrichment/conflict/search",
            {"q": query, "label": "direct conflict", "limit": 25},
            timeout=600.0,
        )
    except RuntimeError:
        conflict_report = {"available": False, "returned": 0, "valid_jumps": 0}
    else:
        items = _json_object_list(conflict_payload.get("items"))
        page = _json_object(conflict_payload.get("page"))
        model = _json_object(conflict_payload.get("conflict_model"))
        aggregation = _json_object(conflict_payload.get("aggregation"))
        valid_jumps = sum(_search_result_jump_is_valid(connection, item) for item in items)
        result_window_count, unique_window_count, active_member_count = _active_membership_count(
            connection, active_config_id, items
        )
        item_contracts_valid = all(
            isinstance(item.get("window_id"), str)
            and bool(item["window_id"])
            and item.get("entity_type") == "message_window"
            and item.get("rank") == index
            and isinstance(item.get("score"), (int, float))
            and not isinstance(item.get("score"), bool)
            and math.isfinite(float(item["score"]))
            and item.get("score_kind") == DEFAULT_CONFLICT_SPEC.score_kind
            and _json_object(item.get("conflict_scoring")).get("score_kind")
            == DEFAULT_CONFLICT_SPEC.score_kind
            and set(_json_object(_json_object(item.get("conflict_scoring")).get("scores")))
            == set(CONFLICT_LABELS)
            and all(
                isinstance(score, (int, float))
                and not isinstance(score, bool)
                and math.isfinite(float(score))
                for score in _json_object(
                    _json_object(item.get("conflict_scoring")).get("scores")
                ).values()
            )
            and abs(
                float(item["score"])
                - float(
                    _json_object(_json_object(item.get("conflict_scoring")).get("scores"))[
                        "direct conflict"
                    ]
                )
            )
            <= 0.000001
            and _json_object(item.get("conflict_scoring")).get("target_label") == "direct conflict"
            and _json_object(item.get("conflict_scoring")).get("predicted_label") in CONFLICT_LABELS
            and _json_object(item.get("conflict_scoring")).get("model_name")
            == DEFAULT_CONFLICT_SPEC.model_name
            and _json_object(item.get("conflict_scoring")).get("model_revision")
            == DEFAULT_CONFLICT_SPEC.model_revision
            and _json_object(item.get("conflict_scoring")).get("quantization")
            == DEFAULT_CONFLICT_SPEC.quantization
            and _json_object(item.get("conflict_scoring")).get("ranking_feature_only") is True
            and _json_object(item.get("conflict_scoring")).get("aggregation_gate_bypassed") is False
            and set(_json_object(_json_object(item.get("conflict_scoring")).get("hypotheses")))
            == set(CONFLICT_LABELS)
            and _json_object(item.get("conflict_scoring")).get("warnings")
            == list(CONFLICT_WARNINGS)
            and _json_object(_json_object(item.get("conflict_scoring")).get("provenance")).get(
                "generated_locally"
            )
            is True
            and _json_object(_json_object(item.get("conflict_scoring")).get("provenance")).get(
                "pipeline_version"
            )
            == "conflict-nli-ranking-v1"
            and _json_object(_json_object(item.get("conflict_scoring")).get("provenance")).get(
                "model_revision"
            )
            == DEFAULT_CONFLICT_SPEC.model_revision
            and _json_object(_json_object(item.get("conflict_scoring")).get("provenance")).get(
                "quantization"
            )
            == DEFAULT_CONFLICT_SPEC.quantization
            for index, item in enumerate(items, start=1)
        )
        aggregation_enabled = aggregation.get("aggregation_enabled") is True
        latest_calibration = _json_object(aggregation.get("latest_calibration"))
        aggregation_honest = bool(
            aggregation.get("relationship_ranking_enabled") is False
            and aggregation.get("minimum_label_count") == MIN_CONFLICT_LABELS
            and isinstance(aggregation.get("sample_count"), int)
            and not isinstance(aggregation.get("sample_count"), bool)
            and aggregation.get("sample_count") == aggregation.get("label_count")
            and _json_object(aggregation.get("thresholds"))
            == {
                "direct_conflict_precision": MIN_DIRECT_CONFLICT_PRECISION,
                "macro_f1": MIN_MACRO_F1,
            }
            and (
                (
                    aggregation.get("status") == "enabled"
                    and latest_calibration.get("status") == "calibrated"
                    and latest_calibration.get("recorded_enabled") is True
                    and latest_calibration.get("evaluation_source") == "held_out_test_predictions"
                    and latest_calibration.get("conversation_disjoint") is True
                    and latest_calibration.get("stratified") is True
                    and isinstance(aggregation.get("label_count"), int)
                    and aggregation["label_count"] >= MIN_CONFLICT_LABELS
                    and isinstance(latest_calibration.get("sample_count"), int)
                    and latest_calibration["sample_count"] >= MIN_CONFLICT_LABELS
                    and isinstance(
                        latest_calibration.get("direct_conflict_precision"), (int, float)
                    )
                    and latest_calibration["direct_conflict_precision"]
                    >= MIN_DIRECT_CONFLICT_PRECISION
                    and isinstance(latest_calibration.get("macro_f1"), (int, float))
                    and latest_calibration["macro_f1"] >= MIN_MACRO_F1
                )
                if aggregation_enabled
                else (
                    aggregation.get("status") == "disabled_until_calibrated"
                    and isinstance(aggregation.get("reason"), str)
                    and bool(aggregation["reason"])
                )
            )
        )
        conflict_passed = bool(
            len(items) > 0
            and len(items) <= 25
            and conflict_payload.get("query") == query
            and conflict_payload.get("target_label") == "direct conflict"
            and page.get("returned") == len(items)
            and page.get("limit") == 25
            and isinstance(page.get("candidate_count"), int)
            and not isinstance(page.get("candidate_count"), bool)
            and page["candidate_count"] >= len(items)
            and _search_config_is_current(conflict_payload.get("retrieval_config"), active_config)
            and model.get("model_name") == DEFAULT_CONFLICT_SPEC.model_name
            and model.get("model_revision") == DEFAULT_CONFLICT_SPEC.model_revision
            and model.get("quantization") == DEFAULT_CONFLICT_SPEC.quantization
            and model.get("device") == DEFAULT_CONFLICT_SPEC.device
            and model.get("score_kind") == DEFAULT_CONFLICT_SPEC.score_kind
            and model.get("max_candidates") == DEFAULT_CONFLICT_SPEC.max_candidates
            and model.get("batch_size") == DEFAULT_CONFLICT_SPEC.batch_size
            and model.get("max_sequence_tokens") == DEFAULT_CONFLICT_SPEC.max_sequence_tokens
            and model.get("status")
            not in {None, "not_loaded", "unavailable", "not_needed_empty_retrieval"}
            and model.get("labels") == list(CONFLICT_LABELS)
            and model.get("warnings") == list(CONFLICT_WARNINGS)
            and model.get("ranking_feature_only") is True
            and model.get("aggregation_gate_bypassed") is False
            and item_contracts_valid
            and result_window_count == unique_window_count
            and unique_window_count == len(items)
            and active_member_count == unique_window_count
            and valid_jumps == len(items)
            and aggregation_honest
        )
        conflict_report = {
            "available": True,
            "returned": len(items),
            "candidate_count": page.get("candidate_count"),
            "valid_jumps": valid_jumps,
            "result_windows": result_window_count,
            "unique_result_windows": unique_window_count,
            "active_member_windows": active_member_count,
            "current_retrieval_config": _search_config_is_current(
                conflict_payload.get("retrieval_config"), active_config
            ),
            "six_score_contract_valid": item_contracts_valid,
            "model_revision": model.get("model_revision"),
            "aggregation_enabled": aggregation_enabled,
            "aggregation_gate_honest": aggregation_honest,
        }
    conflict_check = _check(
        "api.conflict_search_user_story",
        conflict_passed,
        expected=(
            "nonempty pinned six-label conflict results with exact source jumps and an honest "
            "aggregate calibration gate"
        ),
        actual=conflict_report,
        detail="Query text, result text, window IDs, scores, people, and jump URLs are omitted",
    )
    return [search_check, conflict_check], {"search": mode_report, "conflict": conflict_report}


def _canonical_json_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _database_json_object(value: object) -> dict[str, Any] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    return dict(value) if isinstance(value, Mapping) else None


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _is_nonnegative_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _nlp_inference_configuration_valid(
    payload: dict[str, Any],
    *,
    task: Literal["sentiment", "emotion", "toxicity"],
) -> bool:
    """Validate the complete production inference identity without retaining source content."""

    spec = MODEL_SPECS[task]
    if set(payload) != {
        "schema",
        "task",
        "model_name",
        "model_revision",
        "model_artifact",
        "implementation",
        "implementation_sha256",
    }:
        return False
    artifact = payload.get("model_artifact")
    implementation = payload.get("implementation")
    if not isinstance(artifact, dict) or not isinstance(implementation, dict):
        return False
    packages = implementation.get("packages")
    if not isinstance(packages, dict):
        return False
    expected_implementation_keys = {
        "schema",
        "pipeline_version",
        "provider",
        "packages",
        "tokenizer_class",
        "model_class",
        "device",
        "precision",
        "quantization",
        "evaluation_mode",
        "inference_mode",
        "activation",
        "input_character_guard",
        "tokenizer_padding",
        "tokenizer_truncation",
        "tokenizer_max_length",
        "label_mapping",
    }
    expected_package_keys = {"transformers", "torch", "tokenizers", "safetensors"}
    return bool(
        payload.get("schema") == NLP_INFERENCE_CONFIGURATION_SCHEMA
        and payload.get("task") == task
        and payload.get("model_name") == spec.model_name
        and payload.get("model_revision") == spec.model_revision
        and set(artifact) == {"schema", "sha256", "file_count", "total_bytes"}
        and artifact.get("schema") == "hf-logical-snapshot-v1"
        and _is_sha256(artifact.get("sha256"))
        and _is_nonnegative_integer(artifact.get("file_count"))
        and int(artifact["file_count"]) > 0
        and _is_nonnegative_integer(artifact.get("total_bytes"))
        and int(artifact["total_bytes"]) > 0
        and set(implementation) == expected_implementation_keys
        and implementation.get("schema") == NLP_IMPLEMENTATION_SCHEMA
        and implementation.get("pipeline_version") == PIPELINE_VERSION
        and implementation.get("provider") == "transformers-auto-sequence-classification"
        and set(packages) == expected_package_keys
        and all(isinstance(packages.get(name), str) and packages[name] for name in packages)
        and isinstance(implementation.get("tokenizer_class"), str)
        and bool(implementation["tokenizer_class"])
        and isinstance(implementation.get("model_class"), str)
        and bool(implementation["model_class"])
        and implementation.get("device") == spec.device
        and isinstance(implementation.get("precision"), str)
        and bool(implementation["precision"])
        and implementation.get("quantization") == spec.quantization
        and implementation.get("evaluation_mode") is True
        and implementation.get("inference_mode") == "torch.inference_mode"
        and implementation.get("activation") == spec.activation
        and implementation.get("input_character_guard") == MAX_TEXT_CHARACTERS
        and implementation.get("tokenizer_padding") is True
        and implementation.get("tokenizer_truncation") is True
        and implementation.get("tokenizer_max_length") == NLP_TOKENIZER_MAX_LENGTH
        and implementation.get("label_mapping")
        == "model.config.id2label casefold with label_<index> fallback"
        and _is_sha256(payload.get("implementation_sha256"))
        and payload.get("implementation_sha256") == _canonical_json_sha256(implementation)
    )


@dataclass(frozen=True, slots=True)
class _NlpProvenanceAudit:
    configuration_sha256: str | None
    configuration: dict[str, Any] | None
    binding_kinds: tuple[str, ...]
    report: dict[str, object]
    valid: bool


def _nlp_task_provenance_audit(
    connection: Any,
    *,
    task: Literal["sentiment", "emotion", "toxicity"],
) -> _NlpProvenanceAudit:
    allowed_bindings = (DIRECT_PROVENANCE_BINDING, LEGACY_PROVENANCE_BINDING)
    compact = connection.execute(
        text(
            """
            WITH active AS (
                SELECT
                    annotation.job_id,
                    json_extract(
                        annotation.provenance_json, '$.inference_configuration.sha256'
                    ) AS configuration_sha256,
                    json_extract(
                        annotation.provenance_json, '$.inference_configuration.binding_kind'
                    ) AS binding_kind,
                    annotation.provenance_json
                FROM nlp_annotations AS annotation
                WHERE annotation.active = 1 AND annotation.task = :task
            )
            SELECT
                count(*) AS active_rows,
                count(DISTINCT configuration_sha256) AS distinct_configurations,
                count(DISTINCT binding_kind) AS distinct_binding_kinds,
                sum(CASE
                    WHEN json_valid(provenance_json) = 0
                      OR json_type(provenance_json, '$.inference_configuration') != 'object'
                      OR json_type(
                          provenance_json, '$.inference_configuration.sha256'
                      ) != 'text'
                      OR length(configuration_sha256) != 64
                      OR configuration_sha256 GLOB '*[^0-9a-f]*'
                      OR json_type(
                          provenance_json, '$.inference_configuration.binding_kind'
                      ) != 'text'
                      OR (SELECT count(*) FROM json_each(
                          CASE WHEN json_valid(provenance_json)
                               THEN provenance_json ELSE '{}' END,
                          '$.inference_configuration'
                      )) != 2
                    THEN 1 ELSE 0 END
                ) AS invalid_compact_rows,
                sum(CASE WHEN binding_kind NOT IN (:direct_binding, :legacy_binding)
                         THEN 1 ELSE 0 END) AS invalid_binding_rows,
                sum(CASE WHEN job.id IS NULL
                              OR job.kind != 'nlp_annotations'
                              OR job.status != 'completed'
                         THEN 1 ELSE 0 END) AS invalid_job_rows,
                sum(CASE WHEN configuration_sha256 IS NOT json_extract(
                                  job.checkpoint_json,
                                  '$.task_inference_configuration_sha256.' || :task
                              )
                         THEN 1 ELSE 0 END) AS task_mapping_mismatch_rows,
                sum(CASE WHEN json_type(
                                  job.checkpoint_json,
                                  '$.inference_configurations."' || configuration_sha256 || '"'
                              ) IS NOT 'object'
                         THEN 1 ELSE 0 END) AS missing_registry_rows
            FROM active
            LEFT JOIN jobs AS job ON job.id = active.job_id
            """
        ),
        {
            "task": task,
            "direct_binding": DIRECT_PROVENANCE_BINDING,
            "legacy_binding": LEGACY_PROVENANCE_BINDING,
        },
    ).one()
    binding_rows = connection.execute(
        text(
            """
            SELECT DISTINCT json_extract(
                provenance_json, '$.inference_configuration.binding_kind'
            ) AS binding_kind
            FROM nlp_annotations
            WHERE active = 1 AND task = :task
            ORDER BY binding_kind
            """
        ),
        {"task": task},
    )
    binding_kinds = tuple(
        str(row.binding_kind)
        for row in binding_rows
        if isinstance(row.binding_kind, str) and row.binding_kind in allowed_bindings
    )

    job_count = 0
    invalid_job_checkpoints = 0
    registry_entries_checked = 0
    invalid_registry_entries = 0
    task_mappings_checked = 0
    invalid_task_mappings = 0
    selected_configuration_sha256: str | None = None
    selected_configuration: dict[str, Any] | None = None
    job_rows = connection.execute(
        text(
            """
            SELECT DISTINCT job.id, job.kind, job.status, job.checkpoint_json
            FROM nlp_annotations AS annotation
            JOIN jobs AS job ON job.id = annotation.job_id
            WHERE annotation.active = 1 AND annotation.task = :task
            ORDER BY job.id
            """
        ),
        {"task": task},
    )
    for job in job_rows:
        job_count += 1
        checkpoint = _database_json_object(job.checkpoint_json)
        checkpoint_valid = bool(
            job.kind == "nlp_annotations" and job.status == "completed" and checkpoint is not None
        )
        registry = checkpoint.get("inference_configurations") if checkpoint is not None else None
        task_mappings = (
            checkpoint.get("task_inference_configuration_sha256")
            if checkpoint is not None
            else None
        )
        checkpoint_valid = bool(
            checkpoint_valid and isinstance(registry, dict) and isinstance(task_mappings, dict)
        )
        if not isinstance(registry, dict) or not isinstance(task_mappings, dict):
            invalid_job_checkpoints += 1
            continue

        for registry_sha256, raw_payload in registry.items():
            registry_entries_checked += 1
            payload = dict(raw_payload) if isinstance(raw_payload, Mapping) else None
            payload_task = payload.get("task") if payload is not None else None
            entry_valid = bool(
                _is_sha256(registry_sha256)
                and payload is not None
                and _canonical_json_sha256(payload) == registry_sha256
                and payload_task in TASKS
                and _nlp_inference_configuration_valid(payload, task=payload_task)
            )
            if not entry_valid:
                invalid_registry_entries += 1

        for mapped_task, mapped_sha256 in task_mappings.items():
            task_mappings_checked += 1
            raw_payload = registry.get(mapped_sha256) if isinstance(mapped_sha256, str) else None
            payload = dict(raw_payload) if isinstance(raw_payload, Mapping) else None
            mapping_valid = bool(
                mapped_task in TASKS
                and _is_sha256(mapped_sha256)
                and payload is not None
                and payload.get("task") == mapped_task
                and _canonical_json_sha256(payload) == mapped_sha256
                and _nlp_inference_configuration_valid(payload, task=mapped_task)
            )
            if not mapping_valid:
                invalid_task_mappings += 1

        mapped_sha256 = task_mappings.get(task)
        raw_configuration = registry.get(mapped_sha256) if isinstance(mapped_sha256, str) else None
        configuration = dict(raw_configuration) if isinstance(raw_configuration, Mapping) else None
        current_valid = bool(
            _is_sha256(mapped_sha256)
            and configuration is not None
            and _canonical_json_sha256(configuration) == mapped_sha256
            and _nlp_inference_configuration_valid(configuration, task=task)
        )
        if not current_valid:
            checkpoint_valid = False
        elif selected_configuration_sha256 is None:
            selected_configuration_sha256 = str(mapped_sha256)
            selected_configuration = configuration
        elif (
            selected_configuration_sha256 != mapped_sha256
            or selected_configuration != configuration
        ):
            checkpoint_valid = False
        if not checkpoint_valid:
            invalid_job_checkpoints += 1

    active_rows = int(compact.active_rows or 0)
    invalid_compact_rows = int(compact.invalid_compact_rows or 0)
    invalid_binding_rows = int(compact.invalid_binding_rows or 0)
    invalid_job_rows = int(compact.invalid_job_rows or 0)
    task_mapping_mismatch_rows = int(compact.task_mapping_mismatch_rows or 0)
    missing_registry_rows = int(compact.missing_registry_rows or 0)
    valid = bool(
        active_rows > 0
        and int(compact.distinct_configurations or 0) == 1
        and invalid_compact_rows == 0
        and invalid_binding_rows == 0
        and invalid_job_rows == 0
        and task_mapping_mismatch_rows == 0
        and missing_registry_rows == 0
        and job_count > 0
        and invalid_job_checkpoints == 0
        and invalid_registry_entries == 0
        and invalid_task_mappings == 0
        and selected_configuration_sha256 is not None
        and selected_configuration is not None
        and bool(binding_kinds)
    )
    return _NlpProvenanceAudit(
        configuration_sha256=selected_configuration_sha256,
        configuration=selected_configuration,
        binding_kinds=binding_kinds,
        report={
            "active_rows_checked": active_rows,
            "distinct_configuration_count": int(compact.distinct_configurations or 0),
            "distinct_binding_kind_count": int(compact.distinct_binding_kinds or 0),
            "invalid_compact_provenance_rows": invalid_compact_rows,
            "invalid_binding_rows": invalid_binding_rows,
            "invalid_job_rows": invalid_job_rows,
            "task_mapping_mismatch_rows": task_mapping_mismatch_rows,
            "missing_registry_rows": missing_registry_rows,
            "job_checkpoints_checked": job_count,
            "invalid_job_checkpoints": invalid_job_checkpoints,
            "registry_entries_checked": registry_entries_checked,
            "invalid_registry_entries": invalid_registry_entries,
            "task_mappings_checked": task_mappings_checked,
            "invalid_task_mappings": invalid_task_mappings,
            "binding_kinds": list(binding_kinds),
            "configuration_sha256": selected_configuration_sha256,
            "reconciled": valid,
        },
        valid=valid,
    )


def _nlp_rollup_configuration_audit(
    connection: Any,
    *,
    task: Literal["sentiment", "emotion", "toxicity"],
    expected_configuration: dict[str, Any],
    expected_sha256: str,
) -> tuple[int, int]:
    groups_checked = 0
    invalid_rows = 0
    groups = connection.execute(
        text(
            """
            SELECT configuration_json, configuration_sha256, count(*) AS row_count
            FROM nlp_hourly_rollups
            WHERE active = 1 AND task = :task
            GROUP BY configuration_json, configuration_sha256
            """
        ),
        {"task": task},
    )
    for group in groups:
        groups_checked += 1
        configuration = _database_json_object(group.configuration_json)
        group_valid = bool(
            configuration is not None
            and configuration == expected_configuration
            and group.configuration_sha256 == expected_sha256
            and _canonical_json_sha256(configuration) == expected_sha256
        )
        if not group_valid:
            invalid_rows += int(group.row_count or 0)
    return groups_checked, invalid_rows


def _nlp_integrity(
    connection: Any, tables: set[str]
) -> tuple[list[AcceptanceCheck], dict[str, object]]:
    required_tables = {"jobs", "nlp_annotations", "nlp_hourly_rollups"}
    if not required_tables.issubset(tables):
        return (
            [
                _check(
                    "nlp.schema",
                    False,
                    expected=sorted(required_tables),
                    actual=sorted(required_tables & tables),
                    detail="NLP acceptance requires the current catalog schema",
                )
            ],
            {},
        )

    checks: list[AcceptanceCheck] = []
    report: dict[str, object] = {}
    candidate_count = int(
        _scalar(
            connection,
            """
            SELECT
              (SELECT count(*) FROM messages
               WHERE deleted_at IS NULL
                 AND (coalesce(content_display, '') != ''
                      OR coalesce(transcription_display, '') != ''))
              +
              (SELECT count(*) FROM activity_records
               WHERE deleted_at IS NULL
                 AND (coalesce(title_display, '') != ''
                      OR coalesce(content_display, '') != ''))
              +
              (SELECT count(*)
               FROM audio_transcript_segments AS segment
               JOIN audio_transcriptions AS transcription
                 ON transcription.id = segment.transcription_id
               WHERE transcription.status = 'completed'
                 AND transcription.index_eligible = 1
                 AND coalesce(segment.text_display, '') != '')
              +
              (SELECT count(*) FROM attachment_text_extractions
               WHERE status = 'completed'
                 AND index_eligible = 1
                 AND coalesce(text_display, '') != '')
            """,
        )
    )
    report["eligible_target_count"] = candidate_count

    duplicate_active_keys = int(
        _scalar(
            connection,
            """
            SELECT count(*)
            FROM (
                SELECT target_type, target_id, task
                FROM nlp_annotations
                WHERE active = 1
                GROUP BY target_type, target_id, task
                HAVING count(*) > 1
            )
            """,
        )
    )
    checks.append(
        _check(
            "nlp.active_target_task_uniqueness",
            duplicate_active_keys == 0,
            expected=0,
            actual=duplicate_active_keys,
            detail="At most one active annotation exists per target_type, target_id, and task",
        )
    )

    task_report: dict[str, object] = {}
    for task in TASKS:
        spec = MODEL_SPECS[task]
        expected_warnings = json.dumps(
            {"items": list(spec.warnings)},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        provenance_audit = _nlp_task_provenance_audit(connection, task=task)
        inference_configuration = provenance_audit.configuration or {}
        model_artifact = inference_configuration.get("model_artifact")
        implementation = inference_configuration.get("implementation")
        model_artifact = dict(model_artifact) if isinstance(model_artifact, Mapping) else {}
        implementation = dict(implementation) if isinstance(implementation, Mapping) else {}
        packages = implementation.get("packages")
        packages = dict(packages) if isinstance(packages, Mapping) else {}
        expected_rollup_configuration = {
            "activation": spec.activation,
            "language_scope": spec.language_scope,
            "device": spec.device,
            "quantization": spec.quantization,
            "max_text_characters": MAX_TEXT_CHARACTERS,
            "tokenizer_truncation_enabled": True,
            "tokenizer_max_length": NLP_TOKENIZER_MAX_LENGTH,
            "time_bucket": "UTC hour",
            "direction_values": ["owner", "other", "activity", "audio", "attachment"],
            "aggregation": "sample count plus confidence and per-label probability sums",
            "model_artifact": model_artifact,
            "inference_backend": {
                "provider": implementation.get("provider"),
                "packages": packages,
            },
            "inference_implementation_sha256": inference_configuration.get("implementation_sha256"),
            "inference_configuration_sha256": provenance_audit.configuration_sha256,
            "annotation_binding_kinds": list(provenance_audit.binding_kinds),
        }
        rollup_configuration_sha256 = _canonical_json_sha256(expected_rollup_configuration)
        target_coverage = connection.execute(
            text(
                """
                WITH eligible(target_type, target_id) AS (
                    SELECT 'message', id FROM messages
                    WHERE deleted_at IS NULL
                      AND (coalesce(content_display, '') != ''
                           OR coalesce(transcription_display, '') != '')
                    UNION ALL
                    SELECT 'activity', id FROM activity_records
                    WHERE deleted_at IS NULL
                      AND (coalesce(title_display, '') != ''
                           OR coalesce(content_display, '') != '')
                    UNION ALL
                    SELECT 'audio_segment', segment.id
                    FROM audio_transcript_segments AS segment
                    JOIN audio_transcriptions AS transcription
                      ON transcription.id = segment.transcription_id
                    WHERE transcription.status = 'completed'
                      AND transcription.index_eligible = 1
                      AND coalesce(segment.text_display, '') != ''
                    UNION ALL
                    SELECT 'attachment_text', id FROM attachment_text_extractions
                    WHERE status = 'completed'
                      AND index_eligible = 1
                      AND coalesce(text_display, '') != ''
                ), active_rows AS (
                    SELECT target_type, target_id
                    FROM nlp_annotations
                    WHERE active = 1 AND task = :task
                )
                SELECT
                    (SELECT count(*) FROM eligible
                     WHERE NOT EXISTS (
                         SELECT 1 FROM active_rows
                         WHERE active_rows.target_type = eligible.target_type
                           AND active_rows.target_id = eligible.target_id
                     )) AS missing_targets,
                    (SELECT count(*) FROM active_rows
                     WHERE NOT EXISTS (
                         SELECT 1 FROM eligible
                         WHERE eligible.target_type = active_rows.target_type
                           AND eligible.target_id = active_rows.target_id
                     )) AS extraneous_targets
                """
            ),
            {"task": task},
        ).one()
        counts = connection.execute(
            text(
                """
                SELECT
                    count(*) AS active_total,
                    sum(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed,
                    sum(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed,
                    sum(CASE WHEN status NOT IN ('completed', 'failed') THEN 1 ELSE 0 END)
                      AS incomplete,
                    count(DISTINCT model_name) AS model_names,
                    count(DISTINCT model_revision) AS model_revisions,
                    count(DISTINCT pipeline_version) AS pipeline_versions,
                    sum(CASE WHEN model_name != :model_name
                                   OR model_revision != :model_revision
                                   OR pipeline_version != :pipeline_version
                             THEN 1 ELSE 0 END) AS wrong_identity,
                    sum(CASE WHEN json_valid(probabilities_json) = 0
                                   OR json_type(probabilities_json) != 'object'
                                   OR json_type(probabilities_json, '$.labels') != 'object'
                             THEN 1 ELSE 0 END) AS invalid_probability_payload,
                    sum(CASE WHEN json_extract(probabilities_json, '$.activation') != :activation
                             THEN 1 ELSE 0 END) AS wrong_activation,
                    sum(CASE WHEN json_valid(attribution_json) = 0
                                   OR json_type(attribution_json) != 'object'
                                   OR json_valid(provenance_json) = 0
                                   OR json_type(provenance_json) != 'object'
                                   OR json_valid(warnings_json) = 0
                                   OR json_type(warnings_json) != 'object'
                                   OR json_type(warnings_json, '$.items') != 'array'
                                   OR json(
                                       CASE WHEN json_valid(warnings_json)
                                            THEN warnings_json ELSE '{}' END
                                   ) != json(:expected_warnings)
                                   OR json_extract(
                                       attribution_json, '$.model_attribution'
                                   ) IS NOT :model_attribution
                                   OR json_extract(
                                       provenance_json, '$.generated_locally'
                                   ) IS NOT 1
                                   OR json_extract(
                                       provenance_json, '$.language_scope'
                                   ) IS NOT :language_scope
                                   OR json_extract(
                                       provenance_json, '$.tokenizer_truncation_enabled'
                                   ) IS NOT 1
                                   OR json_extract(
                                       provenance_json, '$.tokenizer_max_length'
                                   ) IS NOT :tokenizer_max_length
                                   OR json_extract(
                                       provenance_json, '$.execution.device'
                                   ) IS NOT :device
                                   OR json_type(
                                       provenance_json, '$.execution.quantization'
                                   ) IS NULL
                                   OR json_extract(
                                       provenance_json, '$.execution.quantization'
                                   ) IS NOT :quantization
                             THEN 1 ELSE 0 END) AS invalid_metadata
                FROM nlp_annotations
                WHERE active = 1 AND task = :task
                """
            ),
            {
                "task": task,
                "model_name": spec.model_name,
                "model_revision": spec.model_revision,
                "pipeline_version": PIPELINE_VERSION,
                "activation": spec.activation,
                "model_attribution": spec.attribution,
                "language_scope": spec.language_scope,
                "tokenizer_max_length": NLP_TOKENIZER_MAX_LENGTH,
                "device": spec.device,
                "quantization": spec.quantization,
                "expected_warnings": expected_warnings,
            },
        ).one()
        active_total = int(counts.active_total or 0)
        completed = int(counts.completed or 0)
        failed = int(counts.failed or 0)
        incomplete = int(counts.incomplete or 0)
        vector = connection.execute(
            text(
                """
                SELECT
                    min((SELECT count(*) FROM json_each(
                        CASE WHEN json_valid(a.probabilities_json)
                             THEN a.probabilities_json ELSE '{"labels": {}}' END,
                        '$.labels'
                    )))
                      AS min_labels,
                    max((SELECT count(*) FROM json_each(
                        CASE WHEN json_valid(a.probabilities_json)
                             THEN a.probabilities_json ELSE '{"labels": {}}' END,
                        '$.labels'
                    )))
                      AS max_labels,
                    count(DISTINCT (
                        SELECT group_concat(key, char(31) ORDER BY key)
                        FROM json_each(
                            CASE WHEN json_valid(a.probabilities_json)
                                 THEN a.probabilities_json ELSE '{"labels": {}}' END,
                            '$.labels'
                        )
                    )) AS distinct_label_sets,
                    sum(CASE WHEN EXISTS (
                                  SELECT 1
                                  FROM json_each(
                                      CASE WHEN json_valid(a.probabilities_json)
                                           THEN a.probabilities_json ELSE '{"labels": {}}' END,
                                      '$.labels'
                                  )
                                  WHERE key IS NULL OR trim(CAST(key AS TEXT)) = ''
                             ) THEN 1 ELSE 0 END) AS invalid_label_keys,
                    sum(CASE WHEN :activation = 'softmax'
                                  AND (
                                      abs(coalesce((
                                          SELECT sum(CAST(value AS REAL))
                                          FROM json_each(
                                              CASE WHEN json_valid(a.probabilities_json)
                                                   THEN a.probabilities_json
                                                   ELSE '{"labels": {}}' END,
                                              '$.labels'
                                          )
                                      ), 0.0) - 1.0) > 0.00001
                                      OR EXISTS (
                                          SELECT 1
                                          FROM json_each(
                                              CASE WHEN json_valid(a.probabilities_json)
                                                   THEN a.probabilities_json
                                                   ELSE '{"labels": {}}' END,
                                              '$.labels'
                                          )
                                          WHERE type NOT IN ('integer', 'real')
                                             OR CAST(value AS REAL) < 0.0
                                             OR CAST(value AS REAL) > 1.0
                                      )
                                  )
                             THEN 1 ELSE 0 END) AS invalid_softmax,
                    sum(CASE WHEN :activation = 'sigmoid'
                                  AND EXISTS (
                                      SELECT 1
                                      FROM json_each(
                                          CASE WHEN json_valid(a.probabilities_json)
                                               THEN a.probabilities_json
                                               ELSE '{"labels": {}}' END,
                                          '$.labels'
                                      )
                                      WHERE type NOT IN ('integer', 'real')
                                         OR CAST(value AS REAL) < 0.0
                                         OR CAST(value AS REAL) > 1.0
                                  )
                             THEN 1 ELSE 0 END) AS invalid_sigmoid,
                    sum(CASE WHEN confidence IS NULL OR predicted_label IS NULL
                                  OR abs(confidence - coalesce((
                                      SELECT max(CAST(value AS REAL))
                                      FROM json_each(
                                          CASE WHEN json_valid(a.probabilities_json)
                                               THEN a.probabilities_json
                                               ELSE '{"labels": {}}' END,
                                          '$.labels'
                                      )
                                  ), -1.0)) > 0.000001
                             THEN 1 ELSE 0 END) AS confidence_mismatch,
                    sum(CASE WHEN predicted_label IS NULL OR NOT EXISTS (
                                  SELECT 1
                                  FROM json_each(
                                      CASE WHEN json_valid(a.probabilities_json)
                                           THEN a.probabilities_json ELSE '{"labels": {}}' END,
                                      '$.labels'
                                  ) AS label
                                  WHERE label.key = a.predicted_label
                                    AND abs(CAST(label.value AS REAL) - a.confidence) <= 0.000001
                             ) THEN 1 ELSE 0 END) AS label_mismatch
                FROM nlp_annotations AS a
                WHERE a.active = 1 AND a.task = :task AND a.status = 'completed'
                """
            ),
            {"task": task, "activation": spec.activation},
        ).one()
        rollup = connection.execute(
            text(
                """
                WITH eligible AS (
                    SELECT
                        CAST(json_extract(attribution_json, '$.timestamp_ms') AS INTEGER)
                          AS timestamp_ms,
                        CAST(json_extract(attribution_json, '$.direction') AS TEXT)
                          AS direction,
                        predicted_label,
                        confidence,
                        probabilities_json
                    FROM nlp_annotations
                    WHERE active = 1
                      AND status = 'completed'
                      AND task = :task
                      AND pipeline_version = :pipeline_version
                      AND model_name = :model_name
                      AND model_revision = :model_revision
                      AND json_type(attribution_json, '$.timestamp_ms') = 'integer'
                      AND json_extract(attribution_json, '$.direction') IN (
                          'owner', 'other', 'activity', 'audio', 'attachment'
                      )
                      AND predicted_label IS NOT NULL
                      AND predicted_label != ''
                      AND confidence IS NOT NULL
                ), expected AS (
                    SELECT
                        CASE
                            WHEN timestamp_ms >= 0
                            THEN timestamp_ms - (timestamp_ms % 3600000)
                            ELSE timestamp_ms - (
                                ((timestamp_ms % 3600000) + 3600000) % 3600000
                            )
                        END AS hour_start_ms,
                        direction,
                        predicted_label,
                        count(*) AS sample_count,
                        sum(confidence) AS confidence_sum
                    FROM eligible
                    GROUP BY hour_start_ms, direction, predicted_label
                ), expected_probabilities AS (
                    SELECT
                        CASE
                            WHEN timestamp_ms >= 0
                            THEN timestamp_ms - (timestamp_ms % 3600000)
                            ELSE timestamp_ms - (
                                ((timestamp_ms % 3600000) + 3600000) % 3600000
                            )
                        END AS hour_start_ms,
                        direction,
                        predicted_label,
                        CAST(probability.key AS TEXT) AS probability_label,
                        sum(CAST(probability.value AS REAL)) AS probability_sum
                    FROM eligible,
                         json_each(
                             CASE WHEN json_valid(eligible.probabilities_json)
                                  THEN eligible.probabilities_json ELSE '{"labels": {}}' END,
                             '$.labels'
                         ) AS probability
                    WHERE probability.type IN ('integer', 'real')
                    GROUP BY hour_start_ms, direction, predicted_label, probability_label
                ), published AS (
                    SELECT id, hour_start_ms, direction, predicted_label,
                           sample_count, confidence_sum, probability_sums_json,
                           configuration_json, configuration_sha256
                    FROM nlp_hourly_rollups
                    WHERE active = 1
                      AND task = :task
                      AND pipeline_version = :pipeline_version
                      AND model_name = :model_name
                      AND model_revision = :model_revision
                ), published_probabilities AS (
                    SELECT
                        published.hour_start_ms,
                        published.direction,
                        published.predicted_label,
                        CAST(probability.key AS TEXT) AS probability_label,
                        CAST(probability.value AS REAL) AS probability_sum
                    FROM published,
                         json_each(
                             CASE WHEN json_valid(published.probability_sums_json)
                                  THEN published.probability_sums_json
                                  ELSE '{"labels": {}}' END,
                             '$.labels'
                         ) AS probability
                    WHERE probability.type IN ('integer', 'real')
                )
                SELECT
                    (SELECT coalesce(sum(sample_count), 0) FROM expected)
                      AS expected_samples,
                    (SELECT coalesce(sum(confidence_sum), 0.0) FROM expected)
                      AS expected_confidence_sum,
                    (SELECT count(*) FROM published) AS buckets,
                    (SELECT coalesce(sum(sample_count), 0) FROM published)
                      AS actual_samples,
                    (SELECT coalesce(sum(confidence_sum), 0.0) FROM published)
                      AS actual_confidence_sum,
                    (SELECT count(*)
                     FROM expected
                     LEFT JOIN published USING (hour_start_ms, direction, predicted_label)
                     WHERE published.id IS NULL
                        OR expected.sample_count != published.sample_count
                        OR abs(expected.confidence_sum - published.confidence_sum) > 0.000001)
                    +
                    (SELECT count(*)
                    FROM published
                     LEFT JOIN expected USING (hour_start_ms, direction, predicted_label)
                     WHERE expected.hour_start_ms IS NULL)
                      AS bucket_mismatches,
                    (SELECT count(*)
                     FROM expected_probabilities
                     LEFT JOIN published_probabilities USING (
                         hour_start_ms, direction, predicted_label, probability_label
                     )
                     WHERE published_probabilities.hour_start_ms IS NULL
                        OR abs(
                            expected_probabilities.probability_sum
                            - published_probabilities.probability_sum
                        ) > 0.000001)
                    +
                    (SELECT count(*)
                     FROM published_probabilities
                     LEFT JOIN expected_probabilities USING (
                         hour_start_ms, direction, predicted_label, probability_label
                     )
                     WHERE expected_probabilities.hour_start_ms IS NULL)
                      AS probability_mismatches,
                    (SELECT count(*) FROM published
                     WHERE json_extract(configuration_json, '$.activation') IS NOT :activation
                        OR configuration_sha256 IS NOT :configuration_sha256
                        OR json_extract(configuration_json, '$.language_scope')
                           IS NOT :language_scope
                        OR json_extract(configuration_json, '$.device') IS NOT :device
                        OR json_type(configuration_json, '$.quantization') IS NULL
                        OR json_extract(configuration_json, '$.quantization')
                           IS NOT :quantization
                        OR json_extract(
                            configuration_json, '$.tokenizer_truncation_enabled'
                        ) IS NOT 1
                        OR json_extract(configuration_json, '$.tokenizer_max_length')
                           IS NOT :tokenizer_max_length
                        OR json_extract(configuration_json, '$.time_bucket')
                           IS NOT 'UTC hour'
                        OR json_extract(configuration_json, '$.max_text_characters')
                           IS NOT :max_text_characters
                        OR json(json_extract(configuration_json, '$.direction_values'))
                           != json(:direction_values)
                        OR json_extract(configuration_json, '$.aggregation')
                           IS NOT :aggregation)
                      AS invalid_configurations
                """
            ),
            {
                "task": task,
                "pipeline_version": PIPELINE_VERSION,
                "model_name": spec.model_name,
                "model_revision": spec.model_revision,
                "activation": spec.activation,
                "language_scope": spec.language_scope,
                "device": spec.device,
                "quantization": spec.quantization,
                "tokenizer_max_length": NLP_TOKENIZER_MAX_LENGTH,
                "max_text_characters": MAX_TEXT_CHARACTERS,
                "direction_values": json.dumps(
                    expected_rollup_configuration["direction_values"],
                    separators=(",", ":"),
                ),
                "aggregation": expected_rollup_configuration["aggregation"],
                "configuration_sha256": rollup_configuration_sha256,
            },
        ).one()
        rollup_configuration_groups, invalid_rollup_provenance_rows = (
            _nlp_rollup_configuration_audit(
                connection,
                task=task,
                expected_configuration=expected_rollup_configuration,
                expected_sha256=rollup_configuration_sha256,
            )
        )
        expected_samples = int(rollup.expected_samples or 0)
        actual_samples = int(rollup.actual_samples or 0)
        confidence_delta = abs(
            float(rollup.expected_confidence_sum or 0.0)
            - float(rollup.actual_confidence_sum or 0.0)
        )
        task_report[task] = {
            "eligible_targets": candidate_count,
            "missing_targets": int(target_coverage.missing_targets or 0),
            "extraneous_targets": int(target_coverage.extraneous_targets or 0),
            "active_total": active_total,
            "completed": completed,
            "failed": failed,
            "incomplete": incomplete,
            "label_count_min": int(vector.min_labels or 0),
            "label_count_max": int(vector.max_labels or 0),
            "distinct_label_sets": int(vector.distinct_label_sets or 0),
            "invalid_label_key_rows": int(vector.invalid_label_keys or 0),
            "invalid_softmax_rows": int(vector.invalid_softmax or 0),
            "invalid_sigmoid_rows": int(vector.invalid_sigmoid or 0),
            "confidence_mismatches": int(vector.confidence_mismatch or 0),
            "predicted_label_mismatches": int(vector.label_mismatch or 0),
            "wrong_identity_rows": int(counts.wrong_identity or 0),
            "invalid_metadata_rows": int(counts.invalid_metadata or 0),
            "rollup_buckets": int(rollup.buckets or 0),
            "rollup_expected_samples": expected_samples,
            "rollup_actual_samples": actual_samples,
            "rollup_confidence_delta": round(confidence_delta, 6),
            "rollup_bucket_mismatches": int(rollup.bucket_mismatches or 0),
            "rollup_probability_mismatches": int(rollup.probability_mismatches or 0),
            "rollup_invalid_configurations": int(rollup.invalid_configurations or 0),
            "rollup_configuration_groups_checked": rollup_configuration_groups,
            "rollup_invalid_inference_provenance_rows": invalid_rollup_provenance_rows,
            "inference_provenance": provenance_audit.report,
            "model_name": spec.model_name,
            "model_revision": spec.model_revision,
            "pipeline_version": PIPELINE_VERSION,
            "activation": spec.activation,
        }
        task_ok = (
            candidate_count > 0
            and int(target_coverage.missing_targets or 0) == 0
            and int(target_coverage.extraneous_targets or 0) == 0
            and active_total == candidate_count
            and completed == candidate_count
            and failed == 0
            and incomplete == 0
            and int(counts.model_names or 0) == 1
            and int(counts.model_revisions or 0) == 1
            and int(counts.pipeline_versions or 0) == 1
            and int(counts.wrong_identity or 0) == 0
            and int(counts.invalid_probability_payload or 0) == 0
            and int(counts.wrong_activation or 0) == 0
            and int(counts.invalid_metadata or 0) == 0
            and int(vector.min_labels or 0) > 0
            and vector.min_labels == vector.max_labels
            and int(vector.distinct_label_sets or 0) == 1
            and int(vector.invalid_label_keys or 0) == 0
            and int(vector.invalid_softmax or 0) == 0
            and int(vector.invalid_sigmoid or 0) == 0
            and int(vector.confidence_mismatch or 0) == 0
            and int(vector.label_mismatch or 0) == 0
            and expected_samples == actual_samples
            and int(rollup.buckets or 0) > 0
            and confidence_delta <= max(0.001, expected_samples * 1e-8)
            and int(rollup.bucket_mismatches or 0) == 0
            and int(rollup.probability_mismatches or 0) == 0
            and int(rollup.invalid_configurations or 0) == 0
            and rollup_configuration_groups == 1
            and invalid_rollup_provenance_rows == 0
            and provenance_audit.valid
        )
        checks.append(
            _check(
                f"nlp.{task}.complete_and_reconciled",
                task_ok,
                expected={
                    "active_completed": candidate_count,
                    "errors": 0,
                    "pinned_revision": spec.model_revision,
                    "activation": spec.activation,
                    "probability_rollup_and_inference_provenance": "valid",
                },
                actual=task_report[task],
                detail=(
                    "Softmax rows sum to one; sigmoid values remain in [0,1]; confidence and "
                    "predicted label equal the vector maximum; active rows, job registries, and "
                    "rollups share one byte-verified inference identity"
                ),
            )
        )
    report["tasks"] = task_report
    return checks, report


def _streamed_sql_identity_digest(connection: Any, statement: str) -> tuple[int, int, str]:
    digest = hashlib.sha256()
    count = 0
    duplicates = 0
    previous: str | None = None
    for row in connection.execute(text(statement)):
        identifier = str(row[0])
        if previous == identifier:
            duplicates += 1
        _update_ordered_identity_digest(digest, identifier)
        previous = identifier
        count += 1
    return count, duplicates, digest.hexdigest()


def _semantic_source_coverage(
    connection: Any,
) -> tuple[AcceptanceCheck, dict[str, object]]:
    """Reconcile every searchable source identity without retaining or reporting source text."""

    source_queries = {
        "messages": {
            "eligible": """
                SELECT id FROM messages
                WHERE deleted_at IS NULL
                  AND (coalesce(content_display, '') != ''
                       OR coalesce(transcription_display, '') != '')
            """,
            "pairs": """
                SELECT association.window_id, association.message_id AS id
                FROM search_window_messages AS association
                JOIN search_windows AS window ON window.id = association.window_id
                WHERE window.entity_type = 'message_window'
            """,
        },
        "activities": {
            "eligible": """
                SELECT id FROM activity_records
                WHERE deleted_at IS NULL
                  AND (coalesce(title_display, '') != ''
                       OR coalesce(content_display, '') != ''
                       OR coalesce(actor_display, '') != '')
            """,
            "pairs": """
                SELECT id AS window_id, activity_id AS id
                FROM search_windows
                WHERE entity_type = 'activity' AND activity_id IS NOT NULL
            """,
        },
        "audio_segments": {
            "eligible": """
                SELECT DISTINCT segment.id
                FROM audio_transcript_segments AS segment
                JOIN audio_transcriptions AS transcription
                  ON transcription.id = segment.transcription_id
                JOIN attachment_occurrences AS occurrence
                  ON occurrence.blob_sha256 = transcription.blob_sha256
                LEFT JOIN messages AS message ON message.id = occurrence.message_id
                WHERE transcription.status = 'completed'
                  AND transcription.index_eligible = 1
                  AND occurrence.deleted_at IS NULL
                  AND (occurrence.message_id IS NULL OR message.deleted_at IS NULL)
            """,
            "pairs": """
                SELECT id AS window_id, audio_segment_id AS id
                FROM search_windows
                WHERE entity_type = 'audio_transcript' AND audio_segment_id IS NOT NULL
            """,
        },
        "attachment_extractions": {
            "eligible": """
                SELECT DISTINCT extraction.id
                FROM attachment_text_extractions AS extraction
                JOIN attachment_occurrences AS occurrence
                  ON occurrence.blob_sha256 = extraction.blob_sha256
                LEFT JOIN messages AS message ON message.id = occurrence.message_id
                WHERE extraction.status = 'completed'
                  AND extraction.index_eligible = 1
                  AND extraction.text_display IS NOT NULL
                  AND occurrence.deleted_at IS NULL
                  AND (occurrence.message_id IS NULL OR message.deleted_at IS NULL)
            """,
            "pairs": """
                SELECT id AS window_id, attachment_extraction_id AS id
                FROM search_windows
                WHERE entity_type = 'attachment_text'
                  AND attachment_extraction_id IS NOT NULL
            """,
        },
    }
    report: dict[str, object] = {}
    all_valid = True
    for source_kind, query in source_queries.items():
        eligible_sql = query["eligible"]
        pairs_sql = query["pairs"]
        aggregate = connection.execute(
            text(
                f"""
                WITH eligible(id) AS ({eligible_sql}),
                represented_pairs(window_id, id) AS ({pairs_sql}),
                represented(id) AS (SELECT DISTINCT id FROM represented_pairs)
                SELECT
                    (SELECT count(*) FROM eligible) AS eligible_count,
                    (SELECT count(DISTINCT id) FROM eligible) AS eligible_unique_count,
                    (SELECT count(*) FROM represented) AS represented_count,
                    (SELECT count(*) FROM represented_pairs) AS representation_occurrences,
                    (SELECT count(*) FROM eligible
                     WHERE NOT EXISTS (
                         SELECT 1 FROM represented WHERE represented.id = eligible.id
                     )) AS missing_count,
                    (SELECT count(*) FROM represented
                     WHERE NOT EXISTS (
                         SELECT 1 FROM eligible WHERE eligible.id = represented.id
                     )) AS extra_count,
                    (SELECT count(*) FROM (
                         SELECT window_id, id FROM represented_pairs
                         GROUP BY window_id, id HAVING count(*) > 1
                     )) AS duplicate_pair_count
                """
            )
        ).one()
        eligible_count, eligible_stream_duplicates, eligible_digest = _streamed_sql_identity_digest(
            connection,
            f"WITH eligible(id) AS ({eligible_sql}) SELECT id FROM eligible ORDER BY id",
        )
        represented_count, represented_stream_duplicates, represented_digest = (
            _streamed_sql_identity_digest(
                connection,
                f"""
                WITH represented_pairs(window_id, id) AS ({pairs_sql})
                SELECT DISTINCT id FROM represented_pairs ORDER BY id
                """,
            )
        )
        source_duplicate_ids = int(aggregate.eligible_count or 0) - int(
            aggregate.eligible_unique_count or 0
        )
        missing_count = int(aggregate.missing_count or 0)
        extra_count = int(aggregate.extra_count or 0)
        duplicate_pair_count = int(aggregate.duplicate_pair_count or 0)
        source_valid = bool(
            eligible_count == int(aggregate.eligible_count or 0)
            and represented_count == int(aggregate.represented_count or 0)
            and eligible_stream_duplicates == source_duplicate_ids == 0
            and represented_stream_duplicates == 0
            and duplicate_pair_count == 0
            and missing_count == 0
            and extra_count == 0
            and eligible_count == represented_count
            and eligible_digest == represented_digest
        )
        all_valid = all_valid and source_valid
        representation_occurrences = int(aggregate.representation_occurrences or 0)
        report[source_kind] = {
            "eligible_source_count": eligible_count,
            "represented_source_count": represented_count,
            "representation_occurrence_count": representation_occurrences,
            "expected_fanout_occurrence_count": max(
                0, representation_occurrences - represented_count
            ),
            "missing_source_count": missing_count,
            "extra_source_count": extra_count,
            "duplicate_source_id_count": source_duplicate_ids,
            "duplicate_window_source_pair_count": duplicate_pair_count,
            "eligible_identity_sha256": eligible_digest,
            "represented_identity_sha256": represented_digest,
            "reconciled": source_valid,
        }
    return (
        _check(
            "semantic.search_window_source_coverage",
            all_valid,
            expected=(
                "exact duplicate-free source-ID coverage for every eligible message, activity, "
                "audio segment, and attachment extraction"
            ),
            actual=report,
            detail=(
                "Only aggregate counts and opaque ordered digests are retained; message overlap, "
                "text chunks, and occurrence expansion are reported as expected fan-out"
            ),
        ),
        report,
    )


def _semantic_integrity(
    layout: ArchiveLayout, engine: Engine, connection: Any, tables: set[str]
) -> tuple[list[AcceptanceCheck], dict[str, object]]:
    required_tables = {
        "relevance_feedback",
        "semantic_index_configs",
        "semantic_window_embeddings",
        "search_windows",
    }
    if not required_tables.issubset(tables):
        return (
            [
                _check(
                    "semantic.schema",
                    False,
                    expected=sorted(required_tables),
                    actual=sorted(required_tables & tables),
                    detail="Semantic acceptance requires the Stage 3 schema",
                )
            ],
            {},
        )
    source_coverage_check, source_coverage_report = _semantic_source_coverage(connection)
    checks: list[AcceptanceCheck] = [source_coverage_check]
    rows = _rows(
        connection,
        """
        SELECT id, status, backend, backend_version, model_name, model_revision,
               dimensions, distance_metric, document_prefix, query_prefix,
               vector_table, window_version, window_count, audit_json
        FROM semantic_index_configs
        WHERE active = 1
        ORDER BY id
        """,
    )
    if len(rows) != 1:
        return (
            [
                *checks,
                _check(
                    "semantic.active_ready_config",
                    False,
                    expected=1,
                    actual=len(rows),
                    detail="Exactly one active semantic configuration must be published",
                ),
            ],
            {
                "active_config_count": len(rows),
                "source_coverage": source_coverage_report,
            },
        )
    row = rows[0]
    config_id = str(row.id)
    legacy_marked = int(
        _scalar(
            connection,
            "SELECT count(*) FROM search_windows WHERE embedding_config_hash = :config_id",
            {"config_id": config_id},
        )
    )
    window_total = int(_scalar(connection, "SELECT count(*) FROM search_windows"))
    membership_count = int(
        _scalar(
            connection,
            """
            SELECT count(*) FROM semantic_window_embeddings
            WHERE config_id = :config_id
            """,
            {"config_id": config_id},
        )
    )
    catalog_duplicate_rows = window_total - int(
        _scalar(connection, "SELECT count(DISTINCT id) FROM search_windows")
    )
    membership_duplicate_rows = membership_count - int(
        _scalar(
            connection,
            """
            SELECT count(DISTINCT window_id) FROM semantic_window_embeddings
            WHERE config_id = :config_id
            """,
            {"config_id": config_id},
        )
    )
    type_counts = {
        str(item[0]): int(item[1])
        for item in _rows(
            connection,
            (
                "SELECT entity_type, count(*) FROM search_windows "
                "GROUP BY entity_type ORDER BY entity_type"
            ),
        )
    }
    audit = json.loads(row.audit_json) if isinstance(row.audit_json, str) else row.audit_json
    audit = audit if isinstance(audit, dict) else {}
    accepted = audit.get("accepted_semantic_ir_evaluation")
    accepted = accepted if isinstance(accepted, dict) else {}
    automatic = accepted.get("automatic_known_item")
    automatic = automatic if isinstance(automatic, dict) else {}
    automatic_sampling = automatic.get("sampling")
    automatic_sampling = automatic_sampling if isinstance(automatic_sampling, dict) else {}
    held_out = accepted.get("held_out_feedback")
    held_out = held_out if isinstance(held_out, dict) else {}
    held_out_metrics = held_out.get("metrics")
    held_out_metrics = held_out_metrics if isinstance(held_out_metrics, dict) else {}
    held_out_exclusions = held_out.get("exclusions")
    held_out_exclusions = held_out_exclusions if isinstance(held_out_exclusions, dict) else {}
    recorded_feedback_revision = held_out.get("feedback_revision")
    current_feedback_revision = held_out_feedback_revision(engine)
    feedback_revision_current = bool(
        isinstance(recorded_feedback_revision, dict)
        and recorded_feedback_revision == current_feedback_revision
        and held_out.get("feedback_revision_stable_during_evaluation") is True
    )

    def held_out_metric_passes(name: str) -> bool:
        value = held_out_metrics.get(name)
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) >= HELD_OUT_THRESHOLDS[name]
        )

    held_out_query_count = held_out_metrics.get("query_count")
    held_out_passed = (
        held_out.get("status") == "passed"
        and held_out.get("used_for_activation") is True
        and held_out.get("passed") is True
        and isinstance(held_out_query_count, int)
        and not isinstance(held_out_query_count, bool)
        and held_out_query_count >= HELD_OUT_MIN_QUERY_COUNT
        and all(held_out_metric_passes(name) for name in HELD_OUT_THRESHOLDS)
        and held_out_exclusions.get("without_relevant_window_judgment") == 0
        and held_out_exclusions.get("relevant_target_not_in_candidate") == 0
        and feedback_revision_current
    )
    development = accepted.get("direct_conflict_development_calibration")
    development = development if isinstance(development, dict) else {}
    recorded_development_revision = development.get("feedback_revision")
    current_development_revision = development_feedback_revision(engine)
    development_revision_current = bool(
        isinstance(recorded_development_revision, dict)
        and recorded_development_revision == current_development_revision
        and development.get("feedback_revision_stable_during_evaluation") is True
    )

    def _non_negative_int(value: object) -> int | None:
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
        return None

    def _metric_map(value: object) -> dict[str, float] | None:
        if not isinstance(value, dict):
            return None
        metrics: dict[str, float] = {}
        for name in DIRECT_CONFLICT_DEVELOPMENT_THRESHOLDS:
            metric = value.get(name)
            if (
                not isinstance(metric, (int, float))
                or isinstance(metric, bool)
                or not math.isfinite(float(metric))
            ):
                return None
            metrics[name] = float(metric)
        return metrics

    grouped_cross_validation = development.get("grouped_cross_validation")
    grouped_cross_validation = (
        grouped_cross_validation if isinstance(grouped_cross_validation, dict) else {}
    )
    cv_minimum_metrics = _metric_map(grouped_cross_validation.get("minimum_metrics"))
    cv_mean_metrics = _metric_map(grouped_cross_validation.get("mean_metrics"))
    literal_query_regression = development.get("literal_query_regression")
    literal_query_regression = (
        literal_query_regression if isinstance(literal_query_regression, dict) else {}
    )
    literal_metrics = _metric_map(literal_query_regression.get("metrics"))

    judgment_count = _non_negative_int(development.get("judgment_count"))
    reviewed_judgment_count = _non_negative_int(development.get("reviewed_judgment_count"))
    binary_judgment_count = _non_negative_int(development.get("binary_judgment_count"))
    relevant_count = _non_negative_int(development.get("relevant_count"))
    not_relevant_count = _non_negative_int(development.get("not_relevant_count"))
    uncertain_count = _non_negative_int(development.get("uncertain_count"))
    conversation_group_count = _non_negative_int(
        development.get("conversation_group_count", development.get("conversation_count"))
    )
    fold_count = _non_negative_int(development.get("fold_count"))
    revision_row_count = _non_negative_int(current_development_revision.get("row_count"))
    counts_valid = bool(
        judgment_count is not None
        and reviewed_judgment_count == judgment_count
        and binary_judgment_count is not None
        and relevant_count is not None
        and not_relevant_count is not None
        and uncertain_count is not None
        and binary_judgment_count == relevant_count + not_relevant_count
        and judgment_count == binary_judgment_count + uncertain_count
        and relevant_count >= DIRECT_CONFLICT_DEVELOPMENT_MIN_RELEVANT_COUNT
        and not_relevant_count >= DIRECT_CONFLICT_DEVELOPMENT_MIN_NOT_RELEVANT_COUNT
        and revision_row_count == judgment_count
        and conversation_group_count is not None
        and conversation_group_count >= DIRECT_CONFLICT_DEVELOPMENT_FOLD_COUNT
        and fold_count == DIRECT_CONFLICT_DEVELOPMENT_FOLD_COUNT
        and _non_negative_int(grouped_cross_validation.get("fold_count"))
        == DIRECT_CONFLICT_DEVELOPMENT_FOLD_COUNT
    )

    def _metrics_pass_thresholds(metrics: dict[str, float] | None) -> bool:
        return bool(
            metrics is not None
            and all(
                metrics[name] >= threshold
                for name, threshold in DIRECT_CONFLICT_DEVELOPMENT_THRESHOLDS.items()
            )
        )

    means_reconcile = bool(
        cv_minimum_metrics is not None
        and cv_mean_metrics is not None
        and all(
            cv_mean_metrics[name] >= cv_minimum_metrics[name]
            for name in DIRECT_CONFLICT_DEVELOPMENT_THRESHOLDS
        )
    )
    model_metadata = development.get("model_metadata")
    model_metadata = model_metadata if isinstance(model_metadata, dict) else {}
    provenance_valid = bool(
        development.get("method") == DIRECT_CONFLICT_DEVELOPMENT_METHOD
        and development.get("ranking_pipeline") == DIRECT_CONFLICT_RANKING_PIPELINE_VERSION
        and model_metadata.get("classifier") == DIRECT_CONFLICT_CLASSIFIER
        and isinstance(model_metadata.get("library_version"), str)
        and bool(model_metadata["library_version"])
    )
    development_passed = bool(
        accepted.get("production_acceptance_passed") is True
        and development.get("required_for_production_acceptance") is True
        and development.get("benchmark_kind") == DIRECT_CONFLICT_DEVELOPMENT_BENCHMARK_KIND
        and development.get("evaluation_intent") == DIRECT_CONFLICT_DEVELOPMENT_INTENT
        and development.get("evidence_kind") == DIRECT_CONFLICT_DEVELOPMENT_EVIDENCE_KIND
        and development.get("held_out") is False
        and development.get("query_details_persisted") is False
        and development.get("status") == "passed"
        and development.get("passed") is True
        and development_revision_current
        and counts_valid
        and _metrics_pass_thresholds(cv_minimum_metrics)
        and means_reconcile
        and _metrics_pass_thresholds(literal_metrics)
        and provenance_valid
    )
    accepted_candidate = accepted.get("candidate")
    accepted_candidate = accepted_candidate if isinstance(accepted_candidate, dict) else {}
    accepted_passed = (
        accepted.get("passed") is True
        and accepted.get("evaluator_version") == SEMANTIC_IR_EVALUATOR_VERSION
        and automatic.get("passed") is True
        and automatic.get("used_for_activation") is True
        and isinstance(automatic_sampling.get("sampled_query_count"), int)
        and automatic_sampling["sampled_query_count"] > 0
        and held_out_passed
        and development_passed
        and accepted_candidate.get("config_hash") == config_id
        and accepted_candidate.get("vector_table") == row.vector_table
        and accepted_candidate.get("window_count") == int(row.window_count)
        and accepted_candidate.get("distance_metric") == "cosine"
    )
    pinned_specs = [
        spec
        for spec in SEMANTIC_MODEL_SPECS.values()
        if spec.model_name == row.model_name
        and spec.model_revision == row.model_revision
        and spec.dimensions == row.dimensions
        and spec.document_prefix == row.document_prefix
        and spec.query_prefix == row.query_prefix
    ]
    model_is_pinned = len(pinned_specs) == 1
    lance_count: int | None = None
    lance_scanned_count: int | None = None
    lance_duplicate_rows: int | None = None
    exact_id_sets_reconcile = False
    lance_ready = False
    lance_validation: SemanticTableValidation | None = None
    vectors_stat = layout.vectors.lstat() if layout.vectors.exists() else None
    if (
        vectors_stat is not None
        and stat.S_ISDIR(vectors_stat.st_mode)
        and not stat.S_ISLNK(vectors_stat.st_mode)
    ):
        try:
            import lancedb  # type: ignore[import-untyped]

            vector_database = lancedb.connect(str(layout.vectors))
            table_names = lance_table_names(vector_database)
            if row.vector_table in table_names:
                vector_table = vector_database.open_table(row.vector_table)
                lance_validation = validate_semantic_table(
                    vector_table,
                    engine,
                    config_id=config_id,
                    dimensions=int(row.dimensions),
                    require_complete=True,
                    raise_on_error=False,
                )
                lance_count = lance_validation.row_count
                lance_scanned_count = lance_validation.row_count
                lance_duplicate_rows = lance_validation.duplicate_rows
                exact_id_sets_reconcile = bool(
                    lance_validation.row_count
                    == lance_validation.unique_count
                    == lance_validation.expected_window_count
                    == lance_validation.membership_count
                    and lance_validation.missing_lance_rows == 0
                    and lance_validation.extra_lance_rows == 0
                    and lance_validation.missing_membership_rows == 0
                    and lance_validation.membership_without_lance_rows == 0
                    and lance_validation.identity_sha256
                    == lance_validation.expected_identity_sha256
                    == lance_validation.membership_identity_sha256
                )
                lance_ready = lance_validation.valid
        except Exception:
            # Error details can contain local paths. A missing/unreadable vector corpus is enough
            # evidence for the fail-closed aggregate report.
            lance_ready = False
    ready = (
        row.status == "ready"
        and row.window_version == SEARCH_WINDOW_VERSION
        and row.distance_metric == "cosine"
        and catalog_duplicate_rows == 0
        and membership_duplicate_rows == 0
        and membership_count == int(row.window_count) == window_total
        and lance_ready
        and model_is_pinned
        and source_coverage_check.status == "pass"
    )
    payload = {
        "active_config_id_sha256": _opaque(config_id),
        "status": row.status,
        "backend": row.backend,
        "backend_version": row.backend_version,
        "model_name": row.model_name,
        "model_revision": row.model_revision,
        "dimensions": row.dimensions,
        "window_version": row.window_version,
        "configured_windows": int(row.window_count),
        "catalog_windows": window_total,
        "membership_windows": membership_count,
        "legacy_marker_windows": legacy_marked,
        "lance_windows": lance_count,
        "lance_scanned_windows": lance_scanned_count,
        "catalog_duplicate_rows": catalog_duplicate_rows,
        "membership_duplicate_rows": membership_duplicate_rows,
        "lance_duplicate_rows": lance_duplicate_rows,
        "lance_missing_catalog_rows": (
            lance_validation.missing_catalog_rows if lance_validation is not None else None
        ),
        "lance_missing_rows": (
            lance_validation.missing_lance_rows if lance_validation is not None else None
        ),
        "lance_extra_rows": (
            lance_validation.extra_lance_rows if lance_validation is not None else None
        ),
        "lance_missing_membership_rows": (
            lance_validation.missing_membership_rows if lance_validation is not None else None
        ),
        "lance_membership_without_row_count": (
            lance_validation.membership_without_lance_rows if lance_validation is not None else None
        ),
        "lance_invalid_vector_rows": (
            lance_validation.invalid_vector_rows if lance_validation is not None else None
        ),
        "lance_metadata_mismatch_rows": (
            lance_validation.metadata_mismatch_rows if lance_validation is not None else None
        ),
        "lance_identity_sha256": (
            lance_validation.identity_sha256 if lance_validation is not None else None
        ),
        "catalog_identity_sha256": (
            lance_validation.expected_identity_sha256 if lance_validation is not None else None
        ),
        "membership_identity_sha256": (
            lance_validation.membership_identity_sha256 if lance_validation is not None else None
        ),
        "lance_metadata_sha256": (
            lance_validation.metadata_sha256 if lance_validation is not None else None
        ),
        "catalog_metadata_sha256": (
            lance_validation.expected_metadata_sha256 if lance_validation is not None else None
        ),
        "exact_id_sets_reconcile": exact_id_sets_reconcile,
        "exact_metadata_reconciles": bool(
            lance_validation is not None
            and lance_validation.metadata_mismatch_rows == 0
            and lance_validation.metadata_sha256 == lance_validation.expected_metadata_sha256
        ),
        "vector_contract_reconciles": bool(
            lance_validation is not None and lance_validation.invalid_vector_rows == 0
        ),
        "entity_type_counts": type_counts,
        "source_coverage": source_coverage_report,
        "accepted_ir_evaluation": accepted_passed,
        "automatic_ir_sample_count": automatic_sampling.get("sampled_query_count"),
        "held_out_feedback_used": held_out.get("used_for_activation") is True,
        "held_out_feedback_passed": held_out_passed,
        "held_out_feedback_query_count": held_out_query_count,
        "held_out_feedback_revision_current": feedback_revision_current,
        "held_out_feedback_revision_row_count": current_feedback_revision["row_count"],
        "direct_conflict_development_calibration": {
            "benchmark_kind": development.get("benchmark_kind"),
            "evidence_kind": development.get("evidence_kind"),
            "held_out": development.get("held_out"),
            "evaluation_intent": development.get("evaluation_intent"),
            "required_for_production_acceptance": (
                development.get("required_for_production_acceptance")
            ),
            "status": development.get("status"),
            "passed": development_passed,
            "query_details_persisted": development.get("query_details_persisted"),
            "feedback_revision": {
                "row_count": revision_row_count,
                "current": development_revision_current,
                "stable_during_evaluation": development.get(
                    "feedback_revision_stable_during_evaluation"
                )
                is True,
            },
            "judgment_count": judgment_count,
            "reviewed_judgment_count": reviewed_judgment_count,
            "binary_judgment_count": binary_judgment_count,
            "relevant_count": relevant_count,
            "not_relevant_count": not_relevant_count,
            "uncertain_count": uncertain_count,
            "conversation_group_count": conversation_group_count,
            "fold_count": fold_count,
            "grouped_cross_validation": {
                "minimum_metrics": cv_minimum_metrics,
                "mean_metrics": cv_mean_metrics,
            },
            "literal_query_regression": {"metrics": literal_metrics},
            "method": development.get("method"),
            "ranking_pipeline": development.get("ranking_pipeline"),
            "model": {
                "classifier": model_metadata.get("classifier"),
                "library_version": model_metadata.get("library_version"),
            },
        },
        "pinned_model_identity": model_is_pinned,
    }
    checks.append(
        _check(
            "semantic.active_ready_config",
            ready and accepted_passed and bool(type_counts),
            expected=(
                "one pinned, ready configuration with exact SQLite/Lance counts and accepted "
                "automatic and held-out IR evidence plus current conversation-grouped "
                "direct-conflict development calibration"
            ),
            actual=payload,
            detail="Vector-table names, window IDs, paths, queries, and retrieved text are omitted",
        )
    )

    completed_topic_jobs = _rows(
        connection,
        """
        SELECT counters_json
        FROM jobs
        WHERE kind = 'topic_clusters' AND status = 'completed'
        ORDER BY finished_at DESC, created_at DESC, id DESC
        """,
    )
    topic_result: dict[str, Any] | None = None
    for topic_row in completed_topic_jobs:
        envelope = topic_row.counters_json
        if isinstance(envelope, str):
            try:
                envelope = json.loads(envelope)
            except json.JSONDecodeError:
                continue
        candidate = envelope.get("result") if isinstance(envelope, dict) else None
        if isinstance(candidate, dict) and candidate.get("semantic_config_id") == config_id:
            topic_result = candidate
            break
    raw_clusters = topic_result.get("clusters") if topic_result else None
    clusters = _json_object_list(raw_clusters)
    clusters_are_objects = isinstance(raw_clusters, list) and len(clusters) == len(raw_clusters)
    sample_count = topic_result.get("sample_count") if topic_result else None
    corpus_count = topic_result.get("corpus_window_count") if topic_result else None
    cluster_ids = [cluster.get("id") for cluster in clusters]
    representatives_valid = True
    representative_values: list[str] = []
    sample_shares: list[float] = []
    cluster_rows_valid = clusters_are_objects
    for cluster in clusters:
        sample_windows = cluster.get("sample_windows")
        sample_share = cluster.get("sample_share")
        representatives = cluster.get("representative_window_ids")
        cluster_rows_valid = cluster_rows_valid and (
            isinstance(cluster.get("id"), str)
            and bool(cluster["id"])
            and isinstance(sample_windows, int)
            and not isinstance(sample_windows, bool)
            and sample_windows > 0
            and isinstance(sample_share, (int, float))
            and not isinstance(sample_share, bool)
            and math.isfinite(float(sample_share))
            and 0 < float(sample_share) <= 1
        )
        if isinstance(sample_share, (int, float)) and not isinstance(sample_share, bool):
            sample_shares.append(float(sample_share))
        if not isinstance(representatives, list) or not 1 <= len(representatives) <= 4:
            representatives_valid = False
            continue
        if not all(isinstance(window_id, str) and window_id for window_id in representatives):
            representatives_valid = False
            continue
        representative_values.extend(representatives)
    representative_ids = set(representative_values)
    representatives_valid = representatives_valid and (
        len(representative_values) == len(representative_ids)
    )
    valid_representative_count = 0
    if representative_ids:
        for start in range(0, len(representative_ids), 400):
            chunk = sorted(representative_ids)[start : start + 400]
            placeholders = ", ".join(f":representative_{index}" for index in range(len(chunk)))
            parameters = {
                f"representative_{index}": window_id for index, window_id in enumerate(chunk)
            }
            parameters["config_id"] = config_id
            valid_representative_count += int(
                _scalar(
                    connection,
                    f"""
                    SELECT count(*) FROM semantic_window_embeddings
                    WHERE config_id = :config_id
                      AND window_id IN ({placeholders})
                    """,
                    parameters,
                )
            )
    declared_cluster_samples = sum(
        int(cluster.get("sample_windows", 0))
        for cluster in clusters
        if isinstance(cluster.get("sample_windows"), int)
        and not isinstance(cluster.get("sample_windows"), bool)
    )
    topic_ready = (
        topic_result is not None
        and topic_result.get("pipeline_version") == TOPIC_CLUSTER_PIPELINE_VERSION
        and topic_result.get("vector_table") == row.vector_table
        and topic_result.get("model_name") == row.model_name
        and topic_result.get("model_revision") == row.model_revision
        and isinstance(sample_count, int)
        and not isinstance(sample_count, bool)
        and sample_count > 0
        and isinstance(corpus_count, int)
        and not isinstance(corpus_count, bool)
        and corpus_count == window_total == membership_count == lance_count
        and sample_count <= corpus_count
        and bool(clusters)
        and cluster_rows_valid
        and len(cluster_ids) == len(set(cluster_ids))
        and all(
            cluster.get("ordinal") == ordinal for ordinal, cluster in enumerate(clusters, start=1)
        )
        and declared_cluster_samples == sample_count
        and len(sample_shares) == len(clusters)
        and abs(sum(sample_shares) - 1.0) <= 0.000001
        and representatives_valid
        and valid_representative_count == len(representative_ids)
    )
    topic_payload = {
        "ready_for_active_semantic_config": topic_ready,
        "corpus_window_count": corpus_count,
        "sample_count": sample_count,
        "cluster_count": len(clusters),
        "declared_cluster_samples": declared_cluster_samples,
        "sample_share_sum": round(sum(sample_shares), 6),
        "representative_count": len(representative_ids),
        "valid_representative_count": valid_representative_count,
    }
    checks.append(
        _check(
            "semantic.topic_clusters_current",
            topic_ready,
            expected="a completed topic product reconciled to the active semantic corpus",
            actual=topic_payload,
            detail="Cluster terms and representative window identifiers are never emitted",
        )
    )
    payload["topic_clusters"] = topic_payload
    return checks, payload


def _derived_artifact_integrity(
    layout: ArchiveLayout,
    engine: Engine,
    connection: Any,
    tables: set[str],
) -> tuple[list[AcceptanceCheck], dict[str, object]]:
    checks: list[AcceptanceCheck] = []
    report: dict[str, object] = {}

    if "media_derivatives" not in tables:
        checks.append(
            _check(
                "derived.media_derivatives",
                False,
                expected="media_derivatives table and manifest",
                actual="schema missing",
                detail="Derivative acceptance requires the current catalog schema",
            )
        )
    else:
        derivative_candidates = media_derivative_candidates(engine)
        rows = _rows(
            connection,
            """
            SELECT id, status
            FROM media_derivatives
            WHERE pipeline_version = :pipeline_version
            ORDER BY id
            """,
            {"pipeline_version": MEDIA_DERIVATIVE_PIPELINE_VERSION},
        )
        derivative_status_counts: dict[str, int] = {}
        for row in rows:
            derivative_status_counts[str(row.status)] = (
                derivative_status_counts.get(str(row.status), 0) + 1
            )
        manifest_valid = False
        manifest_record_count = 0
        manifest_status_counts: dict[str, int] = {}
        manifest_ids: set[str] = set()
        manifest_status_by_id: dict[str, str] = {}
        if layout.derivative_manifest.is_file() and not layout.derivative_manifest.is_symlink():
            try:
                manifest = json.loads(layout.derivative_manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                manifest = None
            records = manifest.get("records") if isinstance(manifest, dict) else None
            if (
                isinstance(manifest, dict)
                and manifest.get("schema_version") == 1
                and manifest.get("pipeline_version") == MEDIA_DERIVATIVE_PIPELINE_VERSION
                and isinstance(records, list)
            ):
                manifest_valid = True
                manifest_record_count = len(records)
                for item in records:
                    if not isinstance(item, dict):
                        manifest_valid = False
                        continue
                    record_id = item.get("id")
                    status_value = item.get("status")
                    if not isinstance(record_id, str) or not isinstance(status_value, str):
                        manifest_valid = False
                        continue
                    manifest_ids.add(record_id)
                    if record_id in manifest_status_by_id:
                        manifest_valid = False
                    manifest_status_by_id[record_id] = status_value
                    manifest_status_counts[status_value] = (
                        manifest_status_counts.get(status_value, 0) + 1
                    )
        database_ids = {str(row.id) for row in rows}
        database_status_by_id = {str(row.id): str(row.status) for row in rows}
        candidate_ids = {
            derivative_id(candidate.blob_sha256, candidate.kind)
            for candidate in derivative_candidates
        }
        passed = (
            manifest_valid
            and len(derivative_candidates) == len(rows) == manifest_record_count
            and candidate_ids == database_ids == manifest_ids
            and database_status_by_id == manifest_status_by_id
            and derivative_status_counts == manifest_status_counts
            and derivative_status_counts.get("processing", 0) == 0
            and set(derivative_status_counts) <= {"ready", "failed"}
        )
        derivative_report = {
            "pipeline_version": MEDIA_DERIVATIVE_PIPELINE_VERSION,
            "candidate_count": len(derivative_candidates),
            "database_record_count": len(rows),
            "manifest_record_count": manifest_record_count,
            "database_status_counts": derivative_status_counts,
            "manifest_status_counts": manifest_status_counts,
            "manifest_valid": manifest_valid,
        }
        report["media_derivatives"] = derivative_report
        checks.append(
            _check(
                "derived.media_derivatives",
                passed,
                expected="candidate, database, and atomic manifest IDs/statuses reconcile",
                actual=derivative_report,
                detail="No derivative path, filename, or blob identifier is emitted",
            )
        )

    if {"audio_transcriptions", "audio_transcript_segments"}.issubset(tables):
        audio_candidates = transcription_candidates(engine)
        rows = _rows(
            connection,
            """
            SELECT id, blob_sha256, status, segment_count
            FROM audio_transcriptions
            WHERE pipeline_version = :pipeline_version
              AND model_name = :model_name
              AND model_revision = :model_revision
            ORDER BY id
            """,
            {
                "pipeline_version": TRANSCRIPTION_PIPELINE_VERSION,
                "model_name": TRANSCRIPTION_MODEL_NAME,
                "model_revision": TRANSCRIPTION_MODEL_REVISION,
            },
        )
        transcription_status_counts: dict[str, int] = {}
        for row in rows:
            transcription_status_counts[str(row.status)] = (
                transcription_status_counts.get(str(row.status), 0) + 1
            )
        segment_rows = int(
            _scalar(
                connection,
                """
                SELECT count(*)
                FROM audio_transcript_segments AS segment
                JOIN audio_transcriptions AS transcription
                  ON transcription.id = segment.transcription_id
                WHERE transcription.pipeline_version = :pipeline_version
                  AND transcription.model_name = :model_name
                  AND transcription.model_revision = :model_revision
                """,
                {
                    "pipeline_version": TRANSCRIPTION_PIPELINE_VERSION,
                    "model_name": TRANSCRIPTION_MODEL_NAME,
                    "model_revision": TRANSCRIPTION_MODEL_REVISION,
                },
            )
        )
        candidate_blobs = {str(item.blob_sha256) for item in audio_candidates}
        database_blobs = {str(row.blob_sha256) for row in rows}
        declared_segments = sum(int(row.segment_count) for row in rows)
        passed = (
            candidate_blobs == database_blobs
            and len(rows) == len(audio_candidates)
            and transcription_status_counts
            == ({"completed": len(audio_candidates)} if audio_candidates else {})
            and declared_segments == segment_rows
        )
        transcription_report = {
            "pipeline_version": TRANSCRIPTION_PIPELINE_VERSION,
            "model_name": TRANSCRIPTION_MODEL_NAME,
            "model_revision": TRANSCRIPTION_MODEL_REVISION,
            "candidate_count": len(audio_candidates),
            "database_record_count": len(rows),
            "status_counts": transcription_status_counts,
            "declared_segment_count": declared_segments,
            "segment_row_count": segment_rows,
        }
        report["audio_transcriptions"] = transcription_report
        checks.append(
            _check(
                "derived.audio_transcriptions",
                passed,
                expected="every current audio candidate has a final row and exact segment count",
                actual=transcription_report,
                detail="No transcript text, filename, or blob identifier is emitted",
            )
        )
    else:
        checks.append(
            _check(
                "derived.audio_transcriptions",
                False,
                expected="current transcription schema",
                actual="schema missing",
                detail="Audio acceptance requires current tables",
            )
        )

    if "attachment_text_extractions" in tables:
        extraction_candidates = attachment_text_candidates(engine)
        rows = _rows(
            connection,
            """
            SELECT id, blob_sha256, status, index_eligible
            FROM attachment_text_extractions
            WHERE pipeline_version = :pipeline_version
              AND extractor_name = :extractor_name
              AND extractor_revision = :extractor_revision
            ORDER BY id
            """,
            {
                "pipeline_version": ATTACHMENT_TEXT_PIPELINE_VERSION,
                "extractor_name": ATTACHMENT_EXTRACTOR_NAME,
                "extractor_revision": ATTACHMENT_EXTRACTOR_REVISION,
            },
        )
        extraction_status_counts: dict[str, int] = {}
        for row in rows:
            extraction_status_counts[str(row.status)] = (
                extraction_status_counts.get(str(row.status), 0) + 1
            )
        candidate_blobs = {str(item.blob_sha256) for item in extraction_candidates}
        database_blobs = {str(row.blob_sha256) for row in rows}
        eligible_count = sum(bool(row.index_eligible) for row in rows)
        passed = (
            candidate_blobs == database_blobs
            and len(rows) == len(extraction_candidates)
            and extraction_status_counts.get("processing", 0) == 0
            and extraction_status_counts.get("failed", 0) == 0
            and set(extraction_status_counts)
            <= {"completed", "excluded", "unsupported", "limit_exceeded"}
        )
        extraction_report = {
            "pipeline_version": ATTACHMENT_TEXT_PIPELINE_VERSION,
            "extractor_name": ATTACHMENT_EXTRACTOR_NAME,
            "extractor_revision": ATTACHMENT_EXTRACTOR_REVISION,
            "candidate_count": len(extraction_candidates),
            "database_record_count": len(rows),
            "status_counts": extraction_status_counts,
            "index_eligible_count": eligible_count,
        }
        report["attachment_text"] = extraction_report
        checks.append(
            _check(
                "derived.attachment_text",
                passed,
                expected="every current document candidate has a final fail-closed result row",
                actual=extraction_report,
                detail=(
                    "Unsupported/excluded results are valid; source text and filenames are omitted"
                ),
            )
        )
    else:
        checks.append(
            _check(
                "derived.attachment_text",
                False,
                expected="current extraction schema",
                actual="schema missing",
                detail="Attachment-text acceptance requires the current table",
            )
        )
    return checks, report


def _analytics_json_mapping(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def _analytics_strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [text_value for child in value.values() for text_value in _analytics_strings(child)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [text_value for child in value for text_value in _analytics_strings(child)]
    return []


def _analytics_link_hostnames(content: object, extra: object) -> list[str]:
    hostnames: list[str] = []
    candidates = ([content] if isinstance(content, str) and content else []) + _analytics_strings(
        extra
    )
    for candidate in candidates:
        for match in _ANALYTICS_URL_PATTERN.findall(candidate):
            parsed = urllib.parse.urlsplit(match.rstrip(".,);]}"))
            hostname = (parsed.hostname or "").casefold()
            if hostname:
                hostnames.append(hostname)
    return hostnames


def _analytics_month(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, UTC).strftime("%Y-%m")


def _analytics_day(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, UTC).date().isoformat()


def _analytics_period(timestamp_ms: int, zone: ZoneInfo, bucket: str) -> str:
    local = datetime.fromtimestamp(timestamp_ms / 1000, zone)
    if bucket == "hour":
        return local.strftime("%Y-%m-%dT%H:00%z")
    if bucket == "day":
        return local.strftime("%Y-%m-%d")
    if bucket == "week":
        iso_year, iso_week, _weekday = local.isocalendar()
        return f"{iso_year:04d}-W{iso_week:02d}"
    if bucket == "month":
        return local.strftime("%Y-%m")
    if bucket == "year":
        return local.strftime("%Y")
    raise RuntimeError("Unsupported analytics time bucket")


def _analytics_filter_probe_bounds() -> tuple[int, int]:
    """Return the UI-equivalent inclusive bounds for the 25-hour 2023 DST fall-back day."""

    zone = ZoneInfo("America/New_York")
    start = datetime(2023, 11, 5, tzinfo=zone)
    return int(start.timestamp() * 1_000), int((start + timedelta(days=1)).timestamp() * 1_000) - 1


def _analytics_media_type(kind: object, media_type: object) -> str:
    kind_value = str(kind or "")
    media_value = str(media_type or "").casefold()
    if kind_value in {"gifs", "gif"}:
        return "gif"
    if kind_value == "sticker":
        return "sticker"
    if kind_value in {"audio_files", "audio"} or media_value.startswith("audio/"):
        return "audio"
    if kind_value in {"videos", "video"} or media_value.startswith("video/"):
        return "video"
    if kind_value in {"photos", "photo", "thread_image"} or media_value.startswith("image/"):
        return "photo"
    return "file"


def _analytics_blob_media_class(media_type: object) -> str:
    value = str(media_type or "").casefold()
    for prefix, label in (
        ("image/", "image"),
        ("video/", "video"),
        ("audio/", "audio"),
        ("text/", "text"),
    ):
        if value.startswith(prefix):
            return label
    return "other"


def _analytics_response_summary(values: Sequence[int]) -> dict[str, object]:
    ordered = sorted(values)
    buckets: Counter[str] = Counter()
    for value in ordered:
        label = "over_7_days"
        lower = 0
        for ceiling, candidate in _ANALYTICS_RESPONSE_BUCKETS:
            if lower <= value < ceiling:
                label = candidate
                break
            lower = ceiling
        buckets[label] += 1
    labels = [candidate for _ceiling, candidate in _ANALYTICS_RESPONSE_BUCKETS] + ["over_7_days"]
    p90 = _percentile(ordered, 0.90)
    return {
        "count": len(ordered),
        "mean_ms": fmean(ordered) if ordered else None,
        "median_ms": median(ordered) if ordered else None,
        "p90_ms": float(p90) if p90 is not None else None,
        "buckets": {label: buckets[label] for label in labels},
    }


def _analytics_streaks(heatmap: Mapping[str, Mapping[str, int]]) -> dict[str, object]:
    active = sorted(
        datetime.fromisoformat(day).date()
        for day, counts in heatmap.items()
        if int(counts.get("total", 0)) > 0
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
    for previous, value in pairwise(active):
        current = current + 1 if value == previous + timedelta(days=1) else 1
        longest = max(longest, current)
    return {
        "active_days": len(active),
        "longest_days": longest,
        "latest_days": current,
        "first_active_date": active[0].isoformat(),
        "last_active_date": active[-1].isoformat(),
    }


def _analytics_nlp_score(task: str, probabilities: Mapping[str, float]) -> float | None:
    normalized = {str(key).casefold(): float(value) for key, value in probabilities.items()}
    if task == "sentiment":
        positive = normalized.get("positive", normalized.get("label_2"))
        negative = normalized.get("negative", normalized.get("label_0"))
        return positive - negative if positive is not None and negative is not None else None
    if task == "toxicity":
        candidates = [
            value
            for label, value in normalized.items()
            if label not in {"non-toxic", "non_toxic", "neutral", "label_0"}
        ]
        return max(candidates) if candidates else None
    return None


class _AnalyticsDuplicateKeyError(ValueError):
    """A supposedly keyed analytics response contained more than one row for one key."""


def _analytics_unique_projection(
    rows: object,
    *,
    key: str,
    value_keys: Sequence[str],
) -> dict[str, object]:
    """Project list rows to a mapping while failing closed on repeated response keys.

    The exception intentionally omits the duplicate value because dimensions such as domains and
    coverage-area keys must never leak into the aggregate acceptance report.
    """

    result: dict[str, object] = {}
    for row in _json_object_list(rows):
        projected_key = str(row.get(key))
        if projected_key in result:
            raise _AnalyticsDuplicateKeyError("Duplicate keyed analytics response row")
        result[projected_key] = {value_key: row.get(value_key) for value_key in value_keys}
    return result


def _analytics_nlp_projection(rows: object, coordinate: str) -> dict[str, object]:
    result: dict[str, object] = {}
    task_rows = rows if isinstance(rows, Mapping) else {}
    for task, values in task_rows.items():
        if not isinstance(values, list):
            continue
        for value in values:
            if not isinstance(value, Mapping):
                continue
            key = f"{task}:{value.get(coordinate)}:{value.get('direction')}"
            if key in result:
                raise _AnalyticsDuplicateKeyError("Duplicate keyed analytics NLP response row")
            result[key] = {
                "sample_count": value.get("sample_count"),
                "mean_confidence": value.get("mean_confidence"),
                "score": value.get("score"),
                "predicted_label_counts": value.get("predicted_label_counts"),
                "mean_probabilities": value.get("mean_probabilities"),
            }
    return result


def _analytics_domain_projection(rows: object) -> dict[str, object]:
    result: dict[str, object] = {}
    for row in _json_object_list(rows):
        domain = str(row.get("domain"))
        if domain in result:
            raise _AnalyticsDuplicateKeyError("Duplicate keyed analytics domain response row")
        periods = _analytics_unique_projection(
            row.get("periods"), key="period", value_keys=("occurrences",)
        )
        result[domain] = {
            "occurrences": row.get("occurrences"),
            "periods": {
                period: _json_object(value).get("occurrences") for period, value in periods.items()
            },
        }
    return result


def _analytics_network_projection(payload: Mapping[str, Any]) -> dict[str, object]:
    nodes = _json_object_list(payload.get("nodes"))
    edges = _json_object_list(payload.get("edges"))
    node_records: list[tuple[object, ...]] = []
    edge_records: list[tuple[object, ...]] = []
    seen_node_ids: set[str] = set()
    seen_edge_ids: set[str] = set()
    for row in nodes:
        identifier = str(row.get("id"))
        if identifier in seen_node_ids:
            raise _AnalyticsDuplicateKeyError("Duplicate keyed analytics network node")
        seen_node_ids.add(identifier)
        node_records.append(
            (
                identifier,
                row.get("kind"),
                row.get("conversation_type"),
                row.get("is_owner"),
                row.get("value"),
                row.get("href"),
            )
        )
    for row in edges:
        identifier = str(row.get("id"))
        if identifier in seen_edge_ids:
            raise _AnalyticsDuplicateKeyError("Duplicate keyed analytics network edge")
        seen_edge_ids.add(identifier)
        edge_records.append((identifier, row.get("source"), row.get("target"), row.get("value")))
    return {
        "conversation_count": payload.get("conversation_count"),
        "participant_label_count": payload.get("participant_label_count"),
        "edge_count": len(edges),
        "node_identity_digest": _analytics_identity_digest(sorted(node_records)),
        "edge_topology_digest": _analytics_identity_digest(sorted(edge_records)),
    }


def _analytics_topic_projection(payload: Mapping[str, Any]) -> dict[str, object]:
    latest = _json_object(payload.get("latest_job"))
    semantic = _json_object(payload.get("semantic_index"))
    result = _json_object(payload.get("result"))
    clusters = _json_object_list(result.get("clusters"))
    cluster_records: list[tuple[object, ...]] = []
    term_records: list[tuple[object, ...]] = []
    representative_records: list[tuple[object, ...]] = []
    seen_cluster_ids: set[str] = set()
    for row in clusters:
        cluster_id = str(row.get("id"))
        if cluster_id in seen_cluster_ids:
            raise _AnalyticsDuplicateKeyError("Duplicate keyed analytics topic cluster")
        seen_cluster_ids.add(cluster_id)
        raw_terms = row.get("terms")
        terms: list[object] = raw_terms if isinstance(raw_terms, list) else []
        cluster_records.append(
            (
                cluster_id,
                row.get("ordinal"),
                row.get("sample_windows"),
                row.get("sample_share"),
            )
        )
        term_records.append((cluster_id, *terms))
        seen_representatives: set[str] = set()
        for representative in _json_object_list(row.get("representatives")):
            window_id = str(representative.get("window_id"))
            if window_id in seen_representatives:
                raise _AnalyticsDuplicateKeyError("Duplicate keyed analytics topic representative")
            seen_representatives.add(window_id)
            representative_records.append(
                (
                    cluster_id,
                    window_id,
                    _json_object(representative.get("jump")).get("href"),
                )
            )
    return {
        "status": payload.get("status"),
        "ready": payload.get("ready"),
        "latest_job": (
            {
                "status": latest.get("status"),
                "progress_current": latest.get("progress_current"),
                "progress_total": latest.get("progress_total"),
            }
            if latest
            else None
        ),
        "semantic_window_count": semantic.get("window_count") if semantic else None,
        "result": (
            {
                "corpus_window_count": result.get("corpus_window_count"),
                "sample_count": result.get("sample_count"),
                "cluster_count": len(clusters),
                "cluster_identity_digest": _analytics_identity_digest(cluster_records),
                "term_digest": _analytics_identity_digest(term_records),
                "representative_link_digest": _analytics_identity_digest(representative_records),
            }
            if result
            else None
        ),
    }


def _analytics_metric_projection(namespace: str, payload: Mapping[str, Any]) -> object:
    if namespace == "overview":
        return {
            "messages": payload.get("messages"),
            "interactions": payload.get("interactions"),
            "attachments": payload.get("attachments"),
            "activity_records": payload.get("activity_records"),
            "link_domains": {
                key: _json_object(value).get("occurrences")
                for key, value in _analytics_unique_projection(
                    payload.get("link_domains"), key="domain", value_keys=("occurrences",)
                ).items()
            },
        }
    if namespace == "messages":
        return {
            "series": _analytics_unique_projection(
                payload.get("series"),
                key="period",
                value_keys=("messages", "sent", "received", "active_conversations"),
            ),
            "summary": payload.get("summary"),
        }
    if namespace == "activity-patterns":
        return {
            "heatmap": _analytics_unique_projection(
                payload.get("heatmap"),
                key="date",
                value_keys=("messages", "activities", "total"),
            ),
            "hours": [
                {
                    key: row.get(key)
                    for key in ("hour", "messages", "sent", "received", "activities")
                }
                for row in _json_object_list(payload.get("hours"))
            ],
            "weekdays": [
                {key: row.get(key) for key in ("weekday", "messages", "activities")}
                for row in _json_object_list(payload.get("weekdays"))
            ],
            "streaks": payload.get("streaks"),
            "peak_days": _analytics_unique_projection(
                payload.get("peak_days"),
                key="date",
                value_keys=("messages", "activities", "total"),
            ),
            "summary": payload.get("summary"),
        }
    if namespace == "interaction-patterns":
        return {
            "responses": {
                key: payload.get("responses", {}).get(key)
                for key in ("owner_replied", "other_replied")
            },
            "initiations": {
                key: payload.get("initiations", {}).get(key) for key in ("owner", "other")
            },
            "active_participants": _analytics_unique_projection(
                payload.get("active_participants"),
                key="period",
                value_keys=("participants", "conversations"),
            ),
        }
    if namespace == "interaction-series":
        return {
            "series": _analytics_unique_projection(
                payload.get("series"),
                key="period",
                value_keys=(
                    "calls",
                    "call_duration_seconds",
                    "reactions",
                    "reaction_explicit_timestamp",
                    "reaction_message_timestamp_fallback",
                    "audio_attachments",
                    "link_occurrences",
                ),
            ),
            "domains": _analytics_domain_projection(payload.get("domains")),
            "summary": payload.get("summary"),
        }
    if namespace == "nlp-trends":
        return {
            "source_sample_count": payload.get("source_sample_count"),
            "series": _analytics_nlp_projection(payload.get("series"), "period"),
            "hour_of_day": _analytics_nlp_projection(payload.get("hour_of_day"), "hour"),
            "partial_boundary_hours_included": _json_object(payload.get("rollup_provenance")).get(
                "partial_boundary_hours_included"
            ),
            "non_whole_hour_timezone_offset_observed": _json_object(
                payload.get("rollup_provenance")
            ).get("non_whole_hour_timezone_offset_observed"),
        }
    if namespace == "conflict-rate":
        return {
            "available": payload.get("available"),
            "series": _analytics_unique_projection(
                payload.get("series"),
                key="period",
                value_keys=(
                    "sample_count",
                    "direct_conflict_window_count",
                    "direct_conflict_window_rate",
                    "low_sample_warning",
                ),
            ),
            "summary": payload.get("summary"),
            "relationship_ranking_enabled": payload.get("relationship_ranking_enabled"),
            "relationship_dimension_count": len(payload.get("relationship_dimensions", [])),
        }
    if namespace == "network":
        return _analytics_network_projection(payload)
    if namespace == "topics":
        return _analytics_topic_projection(payload)
    if namespace == "attachments":
        return {
            "types": _analytics_unique_projection(
                payload.get("types"),
                key="type",
                value_keys=("occurrences", "unique_files", "occurrence_bytes"),
            ),
            "availability": _analytics_unique_projection(
                payload.get("availability"),
                key="availability",
                value_keys=("occurrences", "bytes", "rate"),
            ),
            "trends": _analytics_unique_projection(
                payload.get("trends"),
                key="period",
                value_keys=("occurrences", "unique_files", "occurrence_bytes", "by_type"),
            ),
            "summary": payload.get("summary"),
        }
    if namespace == "conversations":
        return {
            key: payload.get(key)
            for key in ("conversation_count", "by_type", "longevity_days", "active_days")
        }
    if namespace == "storage":
        return {
            "source_occurrences": payload.get("source_occurrences"),
            "unique_blobs": payload.get("unique_blobs"),
            "deduplication": payload.get("deduplication"),
            "by_media_class": _analytics_unique_projection(
                payload.get("by_media_class"), key="class", value_keys=("files", "bytes")
            ),
        }
    if namespace == "coverage":
        history = _json_object(payload.get("history"))
        return {
            "summary": payload.get("summary"),
            "items": _analytics_unique_projection(
                payload.get("items"),
                key="key",
                value_keys=(
                    "status",
                    "raw_count",
                    "logical_count",
                    "absence_is_not_zero_history",
                ),
            ),
            "history": {
                "historical_comparison_available": history.get("historical_comparison_available"),
                "snapshot_count": len(history.get("snapshots", [])),
            },
        }
    raise RuntimeError("Unknown analytics dashboard endpoint")


_PAGED_ANALYTICS_FIELDS = {
    "messages": "series",
    "activity-patterns": "heatmap",
    "interaction-patterns": "active_participants",
    "interaction-series": "series",
    "attachments": "trends",
    "conflict-rate": "series",
}


def _merge_nlp_hour_rows(target: dict[str, Any], page: Mapping[str, Any]) -> None:
    target_tasks = _json_object(target.get("hour_of_day"))
    page_tasks = _json_object(page.get("hour_of_day"))
    for task in set(target_tasks) | set(page_tasks):
        keyed: dict[tuple[str, str], dict[str, Any]] = {}
        for row in [
            *_json_object_list(target_tasks.get(task)),
            *_json_object_list(page_tasks.get(task)),
        ]:
            key = (str(row.get("hour")), str(row.get("direction")))
            sample_count = int(row.get("sample_count") or 0)
            aggregate = keyed.setdefault(
                key,
                {
                    "hour": row.get("hour"),
                    "direction": row.get("direction"),
                    "sample_count": 0,
                    "confidence_sum": 0.0,
                    "probability_sums": Counter(),
                    "predicted_label_counts": Counter(),
                    "score_definition": row.get("score_definition"),
                },
            )
            aggregate["sample_count"] += sample_count
            aggregate["confidence_sum"] += float(row.get("mean_confidence") or 0.0) * sample_count
            for label, value in _json_object(row.get("mean_probabilities")).items():
                aggregate["probability_sums"][str(label)] += float(value) * sample_count
            for label, value in _json_object(row.get("predicted_label_counts")).items():
                aggregate["predicted_label_counts"][str(label)] += int(value)
        merged_rows: list[dict[str, Any]] = []
        for aggregate in keyed.values():
            sample_count = int(aggregate["sample_count"])
            probabilities = {
                label: value / sample_count
                for label, value in sorted(aggregate["probability_sums"].items())
            }
            predicted = dict(sorted(aggregate["predicted_label_counts"].items()))
            dominant = (
                sorted(predicted.items(), key=lambda item: (-item[1], item[0]))[0][0]
                if predicted
                else None
            )
            merged_rows.append(
                {
                    "hour": aggregate["hour"],
                    "direction": aggregate["direction"],
                    "sample_count": sample_count,
                    "mean_confidence": aggregate["confidence_sum"] / sample_count,
                    "score": _analytics_nlp_score(str(task), probabilities),
                    "score_definition": aggregate["score_definition"],
                    "dominant_label": dominant,
                    "predicted_label_counts": predicted,
                    "mean_probabilities": probabilities,
                }
            )
        target_tasks[task] = sorted(
            merged_rows, key=lambda row: (str(row["hour"]), str(row["direction"]))
        )
    target["hour_of_day"] = target_tasks


def _merge_analytics_page(namespace: str, target: dict[str, Any], page: Mapping[str, Any]) -> None:
    if namespace == "nlp-trends":
        target_series = _json_object(target.get("series"))
        page_series = _json_object(page.get("series"))
        for task in set(target_series) | set(page_series):
            target_series[task] = [
                *_json_object_list(target_series.get(task)),
                *_json_object_list(page_series.get(task)),
            ]
        target["series"] = target_series
        _merge_nlp_hour_rows(target, page)
        target["page_sample_count"] = int(target.get("page_sample_count") or 0) + int(
            page.get("page_sample_count") or 0
        )
        target_provenance = _json_object(target.get("rollup_provenance"))
        page_provenance = _json_object(page.get("rollup_provenance"))
        target_provenance["non_whole_hour_timezone_offset_observed"] = bool(
            target_provenance.get("non_whole_hour_timezone_offset_observed")
            or page_provenance.get("non_whole_hour_timezone_offset_observed")
        )
        target["rollup_provenance"] = target_provenance
        return
    field = _PAGED_ANALYTICS_FIELDS[namespace]
    target[field] = [
        *_json_object_list(target.get(field)),
        *_json_object_list(page.get(field)),
    ]
    if namespace == "interaction-series":
        domains = {str(row.get("domain")): row for row in _json_object_list(target.get("domains"))}
        for row in _json_object_list(page.get("domains")):
            domain = str(row.get("domain"))
            if domain not in domains:
                domains[domain] = row
            else:
                domains[domain]["periods"] = [
                    *_json_object_list(domains[domain].get("periods")),
                    *_json_object_list(row.get("periods")),
                ]
        target["domains"] = [domains[key] for key in sorted(domains)]


def _analytics_page_invariants(namespace: str, payload: Mapping[str, Any]) -> object:
    common = {key: payload.get(key) for key in ("bucket", "timezone", "summary") if key in payload}
    if namespace == "messages":
        return {**common, "coverage": payload.get("coverage")}
    if namespace == "activity-patterns":
        return {
            **common,
            "hours": payload.get("hours"),
            "weekdays": payload.get("weekdays"),
            "streaks": payload.get("streaks"),
            "peak_days": payload.get("peak_days"),
        }
    if namespace == "interaction-patterns":
        return {
            **common,
            "responses": payload.get("responses"),
            "initiations": payload.get("initiations"),
        }
    if namespace == "interaction-series":
        return {
            **common,
            "domain_totals": [
                (row.get("domain"), row.get("occurrences"))
                for row in _json_object_list(payload.get("domains"))
            ],
        }
    if namespace == "attachments":
        return {
            **common,
            "types": payload.get("types"),
            "availability": payload.get("availability"),
        }
    if namespace == "nlp-trends":
        provenance = _json_object(payload.get("rollup_provenance"))
        return {
            **common,
            "direction": payload.get("direction"),
            "source_sample_count": payload.get("source_sample_count"),
            "models": payload.get("models"),
            "availability": payload.get("availability"),
            "range_filter_semantics": provenance.get("range_filter_semantics"),
            "partial_boundary_hours_included": provenance.get("partial_boundary_hours_included"),
        }
    return common


def _analytics_all_pages(
    client: LocalApiClient,
    namespace: str,
    parameters: Mapping[str, object],
) -> tuple[dict[str, Any], float]:
    if namespace not in {*_PAGED_ANALYTICS_FIELDS, "nlp-trends"}:
        return client.get_json(f"/api/v1/analytics/{namespace}", parameters, timeout=300.0)
    limit = 500
    request_parameters = {**parameters, "limit": limit}
    payload, elapsed = client.get_json(
        f"/api/v1/analytics/{namespace}", request_parameters, timeout=300.0
    )
    merged = json.loads(json.dumps(payload))
    invariants = _analytics_page_invariants(namespace, payload)
    seen_cursors: set[str] = set()
    page_count = 1
    while True:
        metadata = _json_object(payload.get("page"))
        returned = metadata.get("returned")
        declared_limit = metadata.get("limit")
        if (
            not isinstance(returned, int)
            or isinstance(returned, bool)
            or not isinstance(declared_limit, int)
            or isinstance(declared_limit, bool)
            or returned < 0
            or returned > declared_limit
            or declared_limit != limit
        ):
            raise RuntimeError("Analytics endpoint emitted invalid bounded-page metadata")
        cursor = metadata.get("next_cursor")
        if cursor is None:
            break
        if not isinstance(cursor, str) or not cursor or cursor in seen_cursors:
            raise RuntimeError("Analytics endpoint emitted an invalid or cyclic cursor")
        seen_cursors.add(cursor)
        payload, page_elapsed = client.get_json(
            f"/api/v1/analytics/{namespace}",
            {**request_parameters, "cursor": cursor},
            timeout=300.0,
        )
        if _analytics_page_invariants(namespace, payload) != invariants:
            raise RuntimeError("Analytics full-scope aggregates changed between cursor pages")
        elapsed += page_elapsed
        _merge_analytics_page(namespace, merged, payload)
        page_count += 1
        if page_count > 10_000:
            raise RuntimeError("Analytics pagination exceeded its safety bound")
    merged["page"] = {
        **_json_object(merged.get("page")),
        "next_cursor": None,
        "page_count": page_count,
    }
    return merged, elapsed


def _analytics_metric_leaf_count(value: object) -> int:
    if isinstance(value, Mapping):
        return sum(_analytics_metric_leaf_count(child) for child in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return sum(_analytics_metric_leaf_count(child) for child in value)
    return 1


def _analytics_metric_mismatches(expected: object, actual: object) -> int:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return max(1, _analytics_metric_leaf_count(expected))
        missing = set(expected) ^ set(actual)
        return len(missing) + sum(
            _analytics_metric_mismatches(expected[key], actual[key])
            for key in set(expected) & set(actual)
        )
    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes, bytearray)):
        if not isinstance(actual, Sequence) or isinstance(actual, (str, bytes, bytearray)):
            return max(1, _analytics_metric_leaf_count(expected))
        return abs(len(expected) - len(actual)) + sum(
            _analytics_metric_mismatches(left, right)
            for left, right in zip(expected, actual, strict=False)
        )
    if type(expected) is not type(actual):
        return 1
    if isinstance(expected, float):
        assert isinstance(actual, float)
        return 0 if math.isclose(float(expected), float(actual), rel_tol=1e-10, abs_tol=1e-9) else 1
    return 0 if expected == actual else 1


def _analytics_identity_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _analytics_direct_sql_expectations(
    connection: Any,
) -> tuple[dict[str, object], dict[tuple[str, str, str, str], object]]:
    zones = {timezone: ZoneInfo(timezone) for timezone in ANALYTICS_TIMEZONES}
    time_scopes = tuple(
        (timezone, bucket) for timezone in ANALYTICS_TIMEZONES for bucket in ANALYTICS_TIME_BUCKETS
    )
    nlp_scopes = tuple(
        (timezone, bucket)
        for timezone in ANALYTICS_TIMEZONES
        for bucket in ANALYTICS_NLP_TIME_BUCKETS
    )
    message_total = 0
    sent_total = 0
    received_total = 0
    first_timestamp_ms: int | None = None
    last_timestamp_ms: int | None = None
    active_conversations: set[str] = set()
    calls = 0
    missed_calls = 0
    call_duration_seconds = 0
    message_periods: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(
        lambda: defaultdict(
            lambda: {"messages": 0, "sent": 0, "received": 0, "conversations": set()}
        )
    )
    heatmaps: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"messages": 0, "activities": 0, "total": 0})
    )
    message_hours: dict[str, Counter[int]] = defaultdict(Counter)
    sent_hours: dict[str, Counter[int]] = defaultdict(Counter)
    received_hours: dict[str, Counter[int]] = defaultdict(Counter)
    message_weekdays: dict[str, Counter[int]] = defaultdict(Counter)
    interaction_series: dict[tuple[str, str], dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    interaction_totals: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    domain_period_counts: dict[tuple[str, str], dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    participants_by_period: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    conversations_by_period: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    owner_response_ms: list[int] = []
    other_response_ms: list[int] = []
    initiations: Counter[str] = Counter()
    previous_by_conversation: dict[str, tuple[int, bool]] = {}

    message_statement = text(
        """
        SELECT pk, id, conversation_id, timestamp_ms, sender_is_owner, sender_search,
               content_display, extra_json, message_kind, call_duration_seconds,
               is_missed_call
        FROM messages
        WHERE deleted_at IS NULL
        ORDER BY timestamp_ms, pk
        """
    )
    for row in connection.execute(message_statement).mappings().yield_per(10_000):
        timestamp_ms = int(row["timestamp_ms"])
        owner = bool(row["sender_is_owner"])
        conversation_id = str(row["conversation_id"])
        message_total += 1
        sent_total += int(owner)
        received_total += int(not owner)
        first_timestamp_ms = (
            timestamp_ms if first_timestamp_ms is None else min(first_timestamp_ms, timestamp_ms)
        )
        last_timestamp_ms = (
            timestamp_ms if last_timestamp_ms is None else max(last_timestamp_ms, timestamp_ms)
        )
        active_conversations.add(conversation_id)
        sender_search = str(row["sender_search"])
        for timezone, zone in zones.items():
            local = datetime.fromtimestamp(timestamp_ms / 1000, zone)
            day = local.date().isoformat()
            heatmaps[timezone][day]["messages"] += 1
            heatmaps[timezone][day]["total"] += 1
            message_hours[timezone][local.hour] += 1
            (sent_hours if owner else received_hours)[timezone][local.hour] += 1
            message_weekdays[timezone][local.weekday()] += 1
            for bucket in ANALYTICS_TIME_BUCKETS:
                scope = (timezone, bucket)
                period = _analytics_period(timestamp_ms, zone, bucket)
                message_period = message_periods[scope][period]
                message_period["messages"] += 1
                message_period["sent" if owner else "received"] += 1
                message_period["conversations"].add(conversation_id)
                participants_by_period[scope][period].add(sender_search)
                conversations_by_period[scope][period].add(conversation_id)

        previous = previous_by_conversation.get(conversation_id)
        if previous is None:
            initiations["owner" if owner else "other"] += 1
        elif previous[1] != owner:
            elapsed = max(0, timestamp_ms - previous[0])
            (owner_response_ms if owner else other_response_ms).append(elapsed)
        previous_by_conversation[conversation_id] = (timestamp_ms, owner)

        duration = int(row["call_duration_seconds"] or 0)
        if row["message_kind"] == "call" or row["call_duration_seconds"] is not None:
            calls += 1
            missed_calls += int(bool(row["is_missed_call"]))
            call_duration_seconds += duration
            interaction_totals["calls"] += 1
            interaction_totals["call_duration_seconds"] += duration
            for timezone, bucket in time_scopes:
                period = _analytics_period(timestamp_ms, zones[timezone], bucket)
                interaction_series[(timezone, bucket)][period]["calls"] += 1
                interaction_series[(timezone, bucket)][period]["call_duration_seconds"] += duration
        extra = _analytics_json_mapping(row["extra_json"])
        for hostname in _analytics_link_hostnames(row["content_display"], extra):
            interaction_totals["link_occurrences"] += 1
            domain_counts[hostname] += 1
            for timezone, bucket in time_scopes:
                period = _analytics_period(timestamp_ms, zones[timezone], bucket)
                interaction_series[(timezone, bucket)][period]["link_occurrences"] += 1
                domain_period_counts[(timezone, bucket)][hostname][period] += 1

    reaction_types: Counter[str] = Counter()
    reaction_count = 0
    reaction_statement = text(
        """
        SELECT reaction.reaction_display, reaction.timestamp_ms AS explicit_timestamp_ms,
               message.timestamp_ms AS message_timestamp_ms
        FROM message_reactions AS reaction
        JOIN messages AS message ON message.id = reaction.message_id
        WHERE message.deleted_at IS NULL
        ORDER BY coalesce(reaction.timestamp_ms, message.timestamp_ms), reaction.id
        """
    )
    for row in connection.execute(reaction_statement).mappings().yield_per(10_000):
        reaction_count += 1
        reaction_types[str(row["reaction_display"])] += 1
        explicit = row["explicit_timestamp_ms"]
        timestamp_ms = int(explicit if explicit is not None else row["message_timestamp_ms"])
        basis = (
            "reaction_explicit_timestamp"
            if explicit is not None
            else "reaction_message_timestamp_fallback"
        )
        interaction_totals["reactions"] += 1
        interaction_totals[basis] += 1
        for timezone, bucket in time_scopes:
            period = _analytics_period(timestamp_ms, zones[timezone], bucket)
            interaction_series[(timezone, bucket)][period]["reactions"] += 1
            interaction_series[(timezone, bucket)][period][basis] += 1

    attachment_total = 0
    attachment_bytes = 0
    audio_total = 0
    audio_unknown_date = 0
    attachment_unknown_date = 0
    type_totals: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"occurrences": 0, "unique_files": set(), "occurrence_bytes": 0}
    )
    availability_totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {"occurrences": 0, "bytes": 0}
    )
    attachment_trends: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(
        lambda: defaultdict(
            lambda: {
                "occurrences": 0,
                "unique_files": set(),
                "occurrence_bytes": 0,
                "by_type": defaultdict(
                    lambda: {"occurrences": 0, "unique_files": set(), "occurrence_bytes": 0}
                ),
            }
        )
    )
    attachment_statement = text(
        """
        SELECT id, blob_sha256, kind, media_type, size_bytes, timestamp_ms, availability
        FROM attachment_occurrences
        WHERE deleted_at IS NULL
        ORDER BY timestamp_ms, id
        """
    )
    for row in connection.execute(attachment_statement).mappings().yield_per(5_000):
        attachment_type = _analytics_media_type(row["kind"], row["media_type"])
        size_bytes = int(row["size_bytes"] or 0)
        identity = (
            f"blob:{row['blob_sha256']}"
            if row["blob_sha256"] is not None
            else f"occurrence:{row['id']}"
        )
        attachment_total += 1
        attachment_bytes += size_bytes
        type_total = type_totals[attachment_type]
        type_total["occurrences"] += 1
        type_total["unique_files"].add(identity)
        type_total["occurrence_bytes"] += size_bytes
        availability_total = availability_totals[str(row["availability"])]
        availability_total["occurrences"] += 1
        availability_total["bytes"] += size_bytes

        timestamp_value = row["timestamp_ms"]
        if timestamp_value is None:
            attachment_unknown_date += 1
            if attachment_type == "audio":
                audio_unknown_date += 1
            continue
        timestamp_ms = int(timestamp_value)
        for timezone, bucket in time_scopes:
            period = _analytics_period(timestamp_ms, zones[timezone], bucket)
            trend = attachment_trends[(timezone, bucket)][period]
            trend["occurrences"] += 1
            trend["unique_files"].add(identity)
            trend["occurrence_bytes"] += size_bytes
            type_trend = trend["by_type"][attachment_type]
            type_trend["occurrences"] += 1
            type_trend["unique_files"].add(identity)
            type_trend["occurrence_bytes"] += size_bytes
        if attachment_type == "audio":
            audio_total += 1
            interaction_totals["audio_attachments"] += 1
            for timezone, bucket in time_scopes:
                period = _analytics_period(timestamp_ms, zones[timezone], bucket)
                interaction_series[(timezone, bucket)][period]["audio_attachments"] += 1
    audio_total += audio_unknown_date

    activity_total = 0
    activity_unknown_date = 0
    activity_hours: dict[str, Counter[int]] = defaultdict(Counter)
    activity_weekdays: dict[str, Counter[int]] = defaultdict(Counter)
    activity_statement = text(
        """
        SELECT timestamp_ms
        FROM activity_records
        WHERE deleted_at IS NULL
        """
    )
    for row in connection.execute(activity_statement).mappings().yield_per(5_000):
        activity_total += 1
        if row["timestamp_ms"] is None:
            activity_unknown_date += 1
            continue
        timestamp_ms = int(row["timestamp_ms"])
        for timezone, zone in zones.items():
            local = datetime.fromtimestamp(timestamp_ms / 1000, zone)
            day = local.date().isoformat()
            heatmaps[timezone][day]["activities"] += 1
            heatmaps[timezone][day]["total"] += 1
            activity_hours[timezone][local.hour] += 1
            activity_weekdays[timezone][local.weekday()] += 1

    baseline_scope = ("UTC", "month")
    baseline_heatmap = heatmaps["UTC"]
    calendar_days = (
        (
            datetime.fromtimestamp(last_timestamp_ms / 1000, UTC).date()
            - datetime.fromtimestamp(first_timestamp_ms / 1000, UTC).date()
        ).days
        + 1
        if first_timestamp_ms is not None and last_timestamp_ms is not None
        else None
    )
    message_series = {
        period: {
            "messages": values["messages"],
            "sent": values["sent"],
            "received": values["received"],
            "active_conversations": len(values["conversations"]),
        }
        for period, values in sorted(message_periods[baseline_scope].items())
    }
    heatmap_metrics = {day: dict(counts) for day, counts in sorted(baseline_heatmap.items())}
    activity_streaks = _analytics_streaks(baseline_heatmap)
    peak_days = {
        day: dict(counts)
        for day, counts in sorted(
            baseline_heatmap.items(), key=lambda item: (-item[1]["total"], item[0])
        )[:10]
    }
    top_domains = domain_counts.most_common(20)

    expectations: dict[str, object] = {
        "overview": {
            "messages": {
                "total": message_total,
                "sent": sent_total,
                "received": received_total,
                "first_timestamp_ms": first_timestamp_ms,
                "last_timestamp_ms": last_timestamp_ms,
                "calendar_days": calendar_days,
                "average_per_calendar_day": (
                    message_total / calendar_days if calendar_days else None
                ),
                "active_conversations": len(active_conversations),
            },
            "interactions": {
                "message_reactions": reaction_count,
                "reaction_types": dict(sorted(reaction_types.items())),
                "calls": calls,
                "missed_calls": missed_calls,
                "call_duration_seconds": call_duration_seconds,
                "audio_attachments": audio_total,
            },
            "attachments": {
                "occurrences": attachment_total,
                "occurrence_bytes": attachment_bytes,
            },
            "activity_records": activity_total,
            "link_domains": dict(domain_counts.most_common(100)),
        },
        "messages": {
            "series": message_series,
            "summary": {
                "messages": message_total,
                "observed_periods": len(message_series),
                "average_per_observed_period": (
                    message_total / len(message_series) if message_series else None
                ),
            },
        },
        "activity-patterns": {
            "heatmap": heatmap_metrics,
            "hours": [
                {
                    "hour": hour,
                    "messages": message_hours["UTC"][hour],
                    "sent": sent_hours["UTC"][hour],
                    "received": received_hours["UTC"][hour],
                    "activities": activity_hours["UTC"][hour],
                }
                for hour in range(24)
            ],
            "weekdays": [
                {
                    "weekday": weekday,
                    "messages": message_weekdays["UTC"][weekday],
                    "activities": activity_weekdays["UTC"][weekday],
                }
                for weekday in range(7)
            ],
            "streaks": activity_streaks,
            "peak_days": peak_days,
            "summary": {
                "messages_in_scope": message_total,
                "dated_activity_records_in_scope": activity_total - activity_unknown_date,
                "active_activity_records_archive_wide": activity_total,
                "unknown_date_activity_records": activity_unknown_date,
                "unknown_date_scope": "archive_wide",
                "unknown_date_included_in_time_buckets": False,
                "date_filter_active": False,
            },
        },
        "interaction-patterns": {
            "responses": {
                "owner_replied": _analytics_response_summary(owner_response_ms),
                "other_replied": _analytics_response_summary(other_response_ms),
            },
            "initiations": {
                "owner": initiations["owner"],
                "other": initiations["other"],
            },
            "active_participants": {
                period: {
                    "participants": len(participants_by_period[baseline_scope][period]),
                    "conversations": len(conversations_by_period[baseline_scope][period]),
                }
                for period in sorted(participants_by_period[baseline_scope])
            },
        },
        "interaction-series": {
            "series": {
                period: {
                    key: counts[key]
                    for key in (
                        "calls",
                        "call_duration_seconds",
                        "reactions",
                        "reaction_explicit_timestamp",
                        "reaction_message_timestamp_fallback",
                        "audio_attachments",
                        "link_occurrences",
                    )
                }
                for period, counts in sorted(interaction_series[baseline_scope].items())
            },
            "domains": {
                domain: {
                    "occurrences": count,
                    "periods": dict(sorted(domain_period_counts[baseline_scope][domain].items())),
                }
                for domain, count in top_domains
            },
            "summary": {
                "calls": interaction_totals["calls"],
                "call_duration_seconds": interaction_totals["call_duration_seconds"],
                "reactions": interaction_totals["reactions"],
                "reaction_explicit_timestamp": interaction_totals["reaction_explicit_timestamp"],
                "reaction_message_timestamp_fallback": interaction_totals[
                    "reaction_message_timestamp_fallback"
                ],
                "audio_attachments": interaction_totals["audio_attachments"],
                "audio_unknown_date": audio_unknown_date,
                "link_occurrences": interaction_totals["link_occurrences"],
                "domain_count": len(domain_counts),
            },
        },
        "attachments": {
            "types": {
                attachment_type: {
                    "occurrences": values["occurrences"],
                    "unique_files": len(values["unique_files"]),
                    "occurrence_bytes": values["occurrence_bytes"],
                }
                for attachment_type, values in sorted(type_totals.items())
            },
            "availability": {
                availability: {
                    "occurrences": values["occurrences"],
                    "bytes": values["bytes"],
                    "rate": values["occurrences"] / attachment_total if attachment_total else None,
                }
                for availability, values in sorted(availability_totals.items())
            },
            "trends": {
                period: {
                    "occurrences": trend["occurrences"],
                    "unique_files": len(trend["unique_files"]),
                    "occurrence_bytes": trend["occurrence_bytes"],
                    "by_type": {
                        attachment_type: {
                            "occurrences": values["occurrences"],
                            "unique_files": len(values["unique_files"]),
                            "occurrence_bytes": values["occurrence_bytes"],
                        }
                        for attachment_type, values in sorted(trend["by_type"].items())
                    },
                }
                for period, trend in sorted(attachment_trends[baseline_scope].items())
            },
            "summary": {
                "occurrences": attachment_total,
                "dated_occurrences": attachment_total - attachment_unknown_date,
                "unknown_date_occurrences": attachment_unknown_date,
                "unknown_date_scope": "archive_wide",
                "unknown_date_included_in_totals": True,
                "missing_occurrences": availability_totals["missing"]["occurrences"],
                "missing_rate": (
                    availability_totals["missing"]["occurrences"] / attachment_total
                    if attachment_total
                    else None
                ),
                "unlinked_occurrences": availability_totals["unlinked"]["occurrences"],
                "unlinked_rate": (
                    availability_totals["unlinked"]["occurrences"] / attachment_total
                    if attachment_total
                    else None
                ),
            },
        },
    }

    nlp_aggregates: dict[tuple[str, str], dict[tuple[str, str, str], dict[str, Any]]] = defaultdict(
        dict
    )
    nlp_hours: dict[str, dict[tuple[str, str, str], dict[str, Any]]] = defaultdict(dict)
    filter_zone_name = "America/New_York"
    filter_zone = zones[filter_zone_name]
    filter_from_ms, filter_to_ms = _analytics_filter_probe_bounds()
    nlp_filter_aggregates: dict[tuple[str, str, str], dict[str, Any]] = {}
    nlp_filter_hours: dict[tuple[str, str, str], dict[str, Any]] = {}
    nlp_filter_source_samples = 0
    nlp_source_samples = 0

    def add_nlp_aggregate(
        target: dict[tuple[str, str, str], dict[str, Any]],
        *,
        task: str,
        coordinate: str,
        output_direction: str,
        sample_count: int,
        confidence_sum: float,
        predicted_label: str,
        labels: object,
    ) -> None:
        key = (task, coordinate, output_direction)
        aggregate = target.setdefault(
            key,
            {
                "sample_count": 0,
                "confidence_sum": 0.0,
                "probability_sums": defaultdict(float),
                "predicted_counts": Counter(),
            },
        )
        aggregate["sample_count"] += sample_count
        aggregate["confidence_sum"] += confidence_sum
        aggregate["predicted_counts"][predicted_label] += sample_count
        if isinstance(labels, Mapping):
            for label, value in labels.items():
                if isinstance(label, str) and isinstance(value, (int, float)):
                    aggregate["probability_sums"][label] += float(value)

    nlp_statement = text(
        """
        SELECT task, hour_start_ms, direction, predicted_label, sample_count,
               confidence_sum, probability_sums_json
        FROM nlp_hourly_rollups
        WHERE active = 1
        ORDER BY hour_start_ms, task, direction, predicted_label
        """
    )
    for row in connection.execute(nlp_statement).mappings().yield_per(5_000):
        task = str(row["task"])
        direction = str(row["direction"])
        timestamp_ms = int(row["hour_start_ms"])
        sample_count = int(row["sample_count"])
        nlp_source_samples += sample_count
        labels = _analytics_json_mapping(row["probability_sums_json"]).get("labels", {})
        confidence_sum = float(row["confidence_sum"])
        predicted_label = str(row["predicted_label"])
        for timezone, zone in zones.items():
            local_hour = datetime.fromtimestamp(timestamp_ms / 1000, zone).strftime("%H")
            for output_direction in (direction, "all"):
                targets = [
                    (
                        nlp_aggregates[(timezone, bucket)],
                        _analytics_period(timestamp_ms, zone, bucket),
                    )
                    for bucket in ANALYTICS_NLP_TIME_BUCKETS
                ]
                targets.append((nlp_hours[timezone], local_hour))
                for target, coordinate in targets:
                    add_nlp_aggregate(
                        target,
                        task=task,
                        coordinate=coordinate,
                        output_direction=output_direction,
                        sample_count=sample_count,
                        confidence_sum=confidence_sum,
                        predicted_label=predicted_label,
                        labels=labels,
                    )
        if timestamp_ms + 3_600_000 > filter_from_ms and timestamp_ms <= filter_to_ms:
            nlp_filter_source_samples += sample_count
            filter_local = datetime.fromtimestamp(timestamp_ms / 1000, filter_zone)
            for output_direction in (direction, "all"):
                for target, coordinate in (
                    (
                        nlp_filter_aggregates,
                        _analytics_period(timestamp_ms, filter_zone, "hour"),
                    ),
                    (nlp_filter_hours, filter_local.strftime("%H")),
                ):
                    add_nlp_aggregate(
                        target,
                        task=task,
                        coordinate=coordinate,
                        output_direction=output_direction,
                        sample_count=sample_count,
                        confidence_sum=confidence_sum,
                        predicted_label=predicted_label,
                        labels=labels,
                    )

    def serialize_nlp_metrics(
        aggregates: Mapping[tuple[str, str, str], Mapping[str, Any]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for (task, coordinate, direction), aggregate in sorted(aggregates.items()):
            sample_count = int(aggregate["sample_count"])
            probability_sums = aggregate["probability_sums"]
            mean_probabilities = {
                str(label): float(value) / sample_count
                for label, value in sorted(probability_sums.items())
            }
            result[f"{task}:{coordinate}:{direction}"] = {
                "sample_count": sample_count,
                "mean_confidence": float(aggregate["confidence_sum"]) / sample_count,
                "score": _analytics_nlp_score(task, mean_probabilities),
                "predicted_label_counts": dict(sorted(aggregate["predicted_counts"].items())),
                "mean_probabilities": mean_probabilities,
            }
        return result

    expectations["nlp-trends"] = {
        "source_sample_count": nlp_source_samples,
        "series": serialize_nlp_metrics(nlp_aggregates[baseline_scope]),
        "hour_of_day": serialize_nlp_metrics(nlp_hours["UTC"]),
        "partial_boundary_hours_included": False,
        "non_whole_hour_timezone_offset_observed": False,
    }

    calibration = (
        connection.execute(
            text(
                """
            SELECT id, status, sample_count, direct_conflict_precision, macro_f1,
                   aggregation_enabled, metrics_json, provenance_json
            FROM conflict_calibrations
            WHERE pipeline_version = :pipeline_version
              AND model_name = :model_name
              AND model_revision = :model_revision
            ORDER BY completed_at DESC, created_at DESC
            LIMIT 1
            """
            ),
            {
                "pipeline_version": CONFLICT_PIPELINE_VERSION,
                "model_name": DEFAULT_CONFLICT_SPEC.model_name,
                "model_revision": DEFAULT_CONFLICT_SPEC.model_revision,
            },
        )
        .mappings()
        .one_or_none()
    )
    conflict_label_count = int(
        _scalar(connection, "SELECT count(DISTINCT window_id) FROM conflict_labels")
    )
    conflict_gate = False
    if calibration is not None:
        metrics = _analytics_json_mapping(calibration["metrics_json"])
        provenance = _analytics_json_mapping(calibration["provenance_json"])
        splits = provenance.get("evaluation_splits", {})
        conflict_gate = bool(
            calibration["status"] == "calibrated"
            and conflict_label_count >= MIN_CONFLICT_LABELS
            and int(calibration["sample_count"]) >= MIN_CONFLICT_LABELS
            and float(calibration["direct_conflict_precision"] or 0)
            >= MIN_DIRECT_CONFLICT_PRECISION
            and float(calibration["macro_f1"] or 0) >= MIN_MACRO_F1
            and metrics.get("evaluation_source") == "held_out_test_predictions"
            and metrics.get("conversation_disjoint") is True
            and metrics.get("stratified") is True
            and isinstance(splits, Mapping)
            and all(int(splits.get(split, 0)) > 0 for split in ("train", "validation", "test"))
            and bool(calibration["aggregation_enabled"])
        )
    conflict_run = (
        connection.execute(
            text(
                """
            SELECT id, active, status
            FROM conflict_aggregate_runs
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """
            )
        )
        .mappings()
        .one_or_none()
    )
    conflict_available = bool(
        conflict_gate
        and conflict_run is not None
        and bool(conflict_run["active"])
        and conflict_run["status"] == "ready"
    )
    conflict_buckets: dict[tuple[str, str], dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0])
    )
    conflict_filter_buckets: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    if conflict_available and conflict_run is not None:
        for row in (
            connection.execute(
                text(
                    """
                SELECT hour_start_ms, sample_count, direct_conflict_count
                FROM conflict_hourly_rollups
                WHERE active = 1 AND run_id = :run_id
                ORDER BY hour_start_ms
                """
                ),
                {"run_id": conflict_run["id"]},
            )
            .mappings()
            .yield_per(5_000)
        ):
            timestamp_ms = int(row["hour_start_ms"])
            for timezone, bucket in nlp_scopes:
                period = _analytics_period(timestamp_ms, zones[timezone], bucket)
                values = conflict_buckets[(timezone, bucket)][period]
                values[0] += int(row["sample_count"])
                values[1] += int(row["direct_conflict_count"])
            if timestamp_ms + 3_600_000 > filter_from_ms and timestamp_ms <= filter_to_ms:
                period = _analytics_period(timestamp_ms, filter_zone, "hour")
                values = conflict_filter_buckets[period]
                values[0] += int(row["sample_count"])
                values[1] += int(row["direct_conflict_count"])
    baseline_conflict_buckets = conflict_buckets[baseline_scope]
    conflict_sample_count = sum(values[0] for values in baseline_conflict_buckets.values())
    conflict_direct_count = sum(values[1] for values in baseline_conflict_buckets.values())
    expectations["conflict-rate"] = {
        "available": conflict_available,
        "series": {
            period: {
                "sample_count": values[0],
                "direct_conflict_window_count": values[1],
                "direct_conflict_window_rate": values[1] / values[0] if values[0] else None,
                "low_sample_warning": values[0] < 30,
            }
            for period, values in sorted(baseline_conflict_buckets.items())
        },
        "summary": {
            "sample_count": conflict_sample_count,
            "direct_conflict_window_count": conflict_direct_count,
            "direct_conflict_window_rate": (
                conflict_direct_count / conflict_sample_count if conflict_sample_count else None
            ),
        },
        "relationship_ranking_enabled": False,
        "relationship_dimension_count": 0,
    }

    conversation_rows = list(
        connection.execute(
            text(
                """
                SELECT id, conversation_type, first_activity_ms, last_activity_ms,
                       message_count, active_days
                FROM conversations
                WHERE deleted_at IS NULL
                ORDER BY id
                """
            )
        ).mappings()
    )
    lifespans = [
        max(
            0.0,
            (int(row["last_activity_ms"]) - int(row["first_activity_ms"])) / 86_400_000,
        )
        for row in conversation_rows
        if row["first_activity_ms"] is not None and row["last_activity_ms"] is not None
    ]
    lifespan_histogram: Counter[str] = Counter()
    for days in lifespans:
        if days < 1:
            lifespan_histogram["same_day"] += 1
        elif days <= 7:
            lifespan_histogram["1_to_7_days"] += 1
        elif days <= 30:
            lifespan_histogram["8_to_30_days"] += 1
        elif days <= 365:
            lifespan_histogram["31_to_365_days"] += 1
        else:
            lifespan_histogram["over_one_year"] += 1
    lifespan_histogram["unknown"] = len(conversation_rows) - len(lifespans)
    expectations["conversations"] = {
        "conversation_count": len(conversation_rows),
        "by_type": dict(
            sorted(Counter(str(row["conversation_type"]) for row in conversation_rows).items())
        ),
        "longevity_days": {
            "mean": fmean(lifespans) if lifespans else None,
            "median": median(lifespans) if lifespans else None,
            "maximum": max(lifespans) if lifespans else None,
            "histogram": dict(lifespan_histogram),
        },
        "active_days": {
            "mean": (
                fmean(int(row["active_days"]) for row in conversation_rows)
                if conversation_rows
                else None
            ),
            "maximum": max((int(row["active_days"]) for row in conversation_rows), default=None),
        },
    }

    network_conversations = list(
        connection.execute(
            text(
                """
                SELECT id, message_count, conversation_type
                FROM conversations
                WHERE deleted_at IS NULL
                ORDER BY message_count DESC, id
                LIMIT 40
                """
            )
        ).mappings()
    )
    network_ids = [str(row["id"]) for row in network_conversations]
    network_counts = {str(row["id"]): int(row["message_count"]) for row in network_conversations}
    participant_values: dict[str, dict[str, object]] = {}
    edge_records: list[tuple[object, ...]] = []
    if network_ids:
        parameters = {f"id_{index}": value for index, value in enumerate(network_ids)}
        placeholders = ", ".join(f":id_{index}" for index in range(len(network_ids)))
        participant_rows = connection.execute(
            text(
                f"""
                SELECT segment.conversation_id, participant.name_search, participant.is_owner
                FROM conversation_segments AS segment
                JOIN segment_participants AS membership ON membership.segment_id = segment.id
                JOIN participants AS participant ON participant.id = membership.participant_id
                WHERE segment.conversation_id IN ({placeholders})
                ORDER BY segment.conversation_id, participant.name_search, participant.id
                """
            ),
            parameters,
        ).mappings()
        seen_edges: set[tuple[str, str]] = set()
        for row in participant_rows:
            conversation_id = str(row["conversation_id"])
            participant = str(row["name_search"])
            edge = (conversation_id, participant)
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            value = network_counts.get(conversation_id, 0)
            participant_value = participant_values.setdefault(
                participant,
                {
                    "id": f"participant:{_stable_id('network-participant', participant)}",
                    "is_owner": bool(row["is_owner"]),
                    "value": 0,
                },
            )
            current_participant_value = participant_value["value"]
            assert isinstance(current_participant_value, int)
            participant_value["value"] = current_participant_value + value
            edge_records.append(
                (
                    _stable_id("network-edge", conversation_id, participant),
                    f"conversation:{conversation_id}",
                    participant_value["id"],
                    value,
                )
            )
    node_records = [
        (
            f"conversation:{row['id']}",
            "conversation",
            row["conversation_type"],
            None,
            int(row["message_count"]),
            f"/conversations/{row['id']}",
        )
        for row in network_conversations
    ] + [
        (
            value["id"],
            "participant_label",
            None,
            value["is_owner"],
            value["value"],
            None,
        )
        for value in participant_values.values()
    ]
    expectations["network"] = {
        "conversation_count": len(network_conversations),
        "participant_label_count": len(participant_values),
        "edge_count": len(edge_records),
        "node_identity_digest": _analytics_identity_digest(sorted(node_records)),
        "edge_topology_digest": _analytics_identity_digest(sorted(edge_records)),
    }

    source_files, source_bytes = connection.execute(
        text(
            """
            SELECT count(*), coalesce(sum(size_bytes), 0)
            FROM source_files
            WHERE scan_status = 'discovered'
            """
        )
    ).one()
    blob_count = 0
    blob_bytes = 0
    blob_classes: dict[str, dict[str, int]] = defaultdict(lambda: {"files": 0, "bytes": 0})
    for row in connection.execute(
        text("SELECT media_type, size_bytes FROM blobs ORDER BY sha256")
    ).mappings():
        blob_count += 1
        size_bytes = int(row["size_bytes"] or 0)
        blob_bytes += size_bytes
        media_class = _analytics_blob_media_class(row["media_type"])
        blob_classes[media_class]["files"] += 1
        blob_classes[media_class]["bytes"] += size_bytes
    source_file_count = int(source_files or 0)
    source_byte_count = int(source_bytes or 0)
    expectations["storage"] = {
        "source_occurrences": {"files": source_file_count, "bytes": source_byte_count},
        "unique_blobs": {"files": blob_count, "bytes": blob_bytes},
        "deduplication": {
            "redundant_occurrences": max(0, source_file_count - blob_count),
            "duplicate_bytes": max(0, source_byte_count - blob_bytes),
        },
        "by_media_class": {
            media_class: dict(values) for media_class, values in sorted(blob_classes.items())
        },
    }

    coverage_rows = list(
        connection.execute(
            text(
                """
                SELECT key, status, raw_count, logical_count
                FROM coverage_areas
                ORDER BY key
                """
            )
        ).mappings()
    )
    coverage_counts = Counter(str(row["status"]) for row in coverage_rows)
    expectations["coverage"] = {
        "summary": dict(sorted(coverage_counts.items())),
        "items": {
            str(row["key"]): {
                "status": row["status"],
                "raw_count": row["raw_count"],
                "logical_count": row["logical_count"],
                "absence_is_not_zero_history": row["status"] == "unavailable",
            }
            for row in coverage_rows
        },
        "history": {
            "historical_comparison_available": False,
            "snapshot_count": 1 if coverage_rows else 0,
        },
    }

    active_semantic = (
        connection.execute(
            text(
                """
            SELECT id, window_count
            FROM semantic_index_configs
            WHERE active = 1 AND status = 'ready'
            LIMIT 1
            """
            )
        )
        .mappings()
        .one_or_none()
    )
    latest_topic_job = (
        connection.execute(
            text(
                """
            SELECT status, progress_current, progress_total, counters_json
            FROM jobs
            WHERE kind = 'topic_clusters'
            ORDER BY created_at DESC
            LIMIT 1
            """
            )
        )
        .mappings()
        .one_or_none()
    )
    completed_topic_job = (
        connection.execute(
            text(
                """
            SELECT counters_json
            FROM jobs
            WHERE kind = 'topic_clusters' AND status = 'completed'
            ORDER BY finished_at DESC, created_at DESC
            LIMIT 1
            """
            )
        )
        .mappings()
        .one_or_none()
    )
    topic_counters = (
        _analytics_json_mapping(completed_topic_job["counters_json"])
        if completed_topic_job is not None
        else {}
    )
    topic_result_value = topic_counters.get("result")
    topic_result = topic_result_value if isinstance(topic_result_value, Mapping) else None
    topic_ready = bool(
        active_semantic is not None
        and topic_result is not None
        and str(topic_result.get("semantic_config_id")) == str(active_semantic["id"])
    )
    if latest_topic_job is not None and latest_topic_job["status"] in {"queued", "running"}:
        topic_status = "building"
    elif (
        latest_topic_job is not None and latest_topic_job["status"] == "failed" and not topic_ready
    ):
        topic_status = "failed"
    elif latest_topic_job is not None and latest_topic_job["status"] == "stale":
        topic_status = "stale"
    elif topic_ready:
        topic_status = "ready"
    elif topic_result is not None:
        topic_status = "stale"
    else:
        topic_status = "unavailable"
    topic_cluster_records: list[tuple[object, ...]] = []
    topic_term_records: list[tuple[object, ...]] = []
    topic_representative_records: list[tuple[object, ...]] = []
    if topic_ready and topic_result is not None and active_semantic is not None:
        raw_clusters = [row for row in topic_result.get("clusters", []) if isinstance(row, Mapping)]
        representative_ids = [
            str(window_id)
            for cluster in raw_clusters
            for window_id in cluster.get("representative_window_ids", [])
            if isinstance(window_id, str)
        ]
        window_rows: dict[str, Mapping[str, Any]] = {}
        if representative_ids:
            placeholders = ", ".join(
                f":topic_window_{index}" for index in range(len(representative_ids))
            )
            parameters = {
                f"topic_window_{index}": value for index, value in enumerate(representative_ids)
            }
            parameters["topic_config_id"] = str(active_semantic["id"])
            membership_count = int(
                _scalar(
                    connection,
                    """
                    SELECT count(*) FROM semantic_window_embeddings
                    WHERE config_id = :topic_config_id
                    """,
                    {"topic_config_id": str(active_semantic["id"])},
                )
            )
            membership_clause = (
                "EXISTS (SELECT 1 FROM semantic_window_embeddings AS membership "
                "WHERE membership.config_id = :topic_config_id "
                "AND membership.window_id = search_window.id)"
                if membership_count
                else "search_window.embedding_config_hash = :topic_config_id"
            )
            rows = connection.execute(
                text(
                    f"""
                    SELECT search_window.id, search_window.conversation_id,
                           search_window.activity_id, search_window.attachment_occurrence_id,
                           search_window.first_message_id, search_window.source_ids_json
                    FROM search_windows AS search_window
                    WHERE search_window.id IN ({placeholders})
                      AND {membership_clause}
                    """
                ),
                parameters,
            ).mappings()
            window_rows = {str(row["id"]): row for row in rows}

        def topic_representative_href(row: Mapping[str, Any]) -> str | None:
            conversation_id = row["conversation_id"]
            activity_id = row["activity_id"]
            occurrence_id = row["attachment_occurrence_id"]
            first_message_id = row["first_message_id"]
            source_ids = _analytics_json_mapping(row["source_ids_json"])
            media_start = source_ids.get("media_start_ms")
            timing = f"&t={media_start}" if media_start is not None else ""
            if occurrence_id is not None:
                if conversation_id is not None and first_message_id is not None:
                    return (
                        f"/conversations/{conversation_id}?around={first_message_id}"
                        f"&media={occurrence_id}{timing}#message-{first_message_id}"
                    )
                if activity_id is not None:
                    return f"/timeline?around=activity:{activity_id}#activity-{activity_id}"
                return f"/media?occurrence={occurrence_id}{timing}"
            if conversation_id is not None and first_message_id is not None:
                return (
                    f"/conversations/{conversation_id}?around={first_message_id}"
                    f"#message-{first_message_id}"
                )
            if activity_id is not None:
                return f"/timeline?around=activity:{activity_id}#activity-{activity_id}"
            return None

        for cluster in raw_clusters:
            cluster_id = str(cluster.get("id"))
            topic_cluster_records.append(
                (
                    cluster_id,
                    cluster.get("ordinal"),
                    cluster.get("sample_windows"),
                    cluster.get("sample_share"),
                )
            )
            raw_terms = cluster.get("terms")
            terms: list[object] = raw_terms if isinstance(raw_terms, list) else []
            topic_term_records.append((cluster_id, *terms))
            for window_id in cluster.get("representative_window_ids", []):
                if not isinstance(window_id, str) or window_id not in window_rows:
                    continue
                topic_representative_records.append(
                    (cluster_id, window_id, topic_representative_href(window_rows[window_id]))
                )
    expectations["topics"] = {
        "status": topic_status,
        "ready": topic_ready,
        "latest_job": (
            {
                "status": latest_topic_job["status"],
                "progress_current": latest_topic_job["progress_current"],
                "progress_total": latest_topic_job["progress_total"],
            }
            if latest_topic_job is not None
            else None
        ),
        "semantic_window_count": (
            active_semantic["window_count"] if active_semantic is not None else None
        ),
        "result": (
            {
                "corpus_window_count": topic_result.get("corpus_window_count"),
                "sample_count": topic_result.get("sample_count"),
                "cluster_count": len(topic_cluster_records),
                "cluster_identity_digest": _analytics_identity_digest(topic_cluster_records),
                "term_digest": _analytics_identity_digest(topic_term_records),
                "representative_link_digest": _analytics_identity_digest(
                    topic_representative_records
                ),
            }
            if topic_ready and topic_result is not None
            else None
        ),
    }

    scenario_expectations: dict[tuple[str, str, str, str], object] = {}
    static_namespaces = ("network", "topics", "conversations", "storage", "coverage")
    for namespace in static_namespaces:
        scenario_expectations[(namespace, "", "", "full")] = expectations[namespace]

    for timezone, zone in zones.items():
        timezone_calendar_days = (
            (
                datetime.fromtimestamp(last_timestamp_ms / 1000, zone).date()
                - datetime.fromtimestamp(first_timestamp_ms / 1000, zone).date()
            ).days
            + 1
            if first_timestamp_ms is not None and last_timestamp_ms is not None
            else None
        )
        overview_expectation = dict(_json_object(expectations["overview"]))
        overview_messages = dict(_json_object(overview_expectation["messages"]))
        overview_messages["calendar_days"] = timezone_calendar_days
        overview_messages["average_per_calendar_day"] = (
            message_total / timezone_calendar_days if timezone_calendar_days else None
        )
        overview_expectation["messages"] = overview_messages
        scenario_expectations[("overview", timezone, "", "full")] = overview_expectation

        scoped_heatmap = heatmaps[timezone]
        heatmap_metrics = {day: dict(counts) for day, counts in sorted(scoped_heatmap.items())}
        peak_days = {
            day: dict(counts)
            for day, counts in sorted(
                scoped_heatmap.items(), key=lambda item: (-item[1]["total"], item[0])
            )[:10]
        }
        scenario_expectations[("activity-patterns", timezone, "", "full")] = {
            "heatmap": heatmap_metrics,
            "hours": [
                {
                    "hour": hour,
                    "messages": message_hours[timezone][hour],
                    "sent": sent_hours[timezone][hour],
                    "received": received_hours[timezone][hour],
                    "activities": activity_hours[timezone][hour],
                }
                for hour in range(24)
            ],
            "weekdays": [
                {
                    "weekday": weekday,
                    "messages": message_weekdays[timezone][weekday],
                    "activities": activity_weekdays[timezone][weekday],
                }
                for weekday in range(7)
            ],
            "streaks": _analytics_streaks(scoped_heatmap),
            "peak_days": peak_days,
            "summary": {
                "messages_in_scope": message_total,
                "dated_activity_records_in_scope": activity_total - activity_unknown_date,
                "active_activity_records_archive_wide": activity_total,
                "unknown_date_activity_records": activity_unknown_date,
                "unknown_date_scope": "archive_wide",
                "unknown_date_included_in_time_buckets": False,
                "date_filter_active": False,
            },
        }

    for timezone, bucket in time_scopes:
        scope = (timezone, bucket)
        scoped_message_periods = message_periods[scope]
        message_series = {
            period: {
                "messages": values["messages"],
                "sent": values["sent"],
                "received": values["received"],
                "active_conversations": len(values["conversations"]),
            }
            for period, values in sorted(scoped_message_periods.items())
        }
        scenario_expectations[("messages", timezone, bucket, "full")] = {
            "series": message_series,
            "summary": {
                "messages": message_total,
                "observed_periods": len(message_series),
                "average_per_observed_period": (
                    message_total / len(message_series) if message_series else None
                ),
            },
        }
        scenario_expectations[("interaction-patterns", timezone, bucket, "full")] = {
            "responses": {
                "owner_replied": _analytics_response_summary(owner_response_ms),
                "other_replied": _analytics_response_summary(other_response_ms),
            },
            "initiations": {
                "owner": initiations["owner"],
                "other": initiations["other"],
            },
            "active_participants": {
                period: {
                    "participants": len(participants_by_period[scope][period]),
                    "conversations": len(conversations_by_period[scope][period]),
                }
                for period in sorted(participants_by_period[scope])
            },
        }
        top_domains = domain_counts.most_common(20)
        scenario_expectations[("interaction-series", timezone, bucket, "full")] = {
            "series": {
                period: {
                    key: counts[key]
                    for key in (
                        "calls",
                        "call_duration_seconds",
                        "reactions",
                        "reaction_explicit_timestamp",
                        "reaction_message_timestamp_fallback",
                        "audio_attachments",
                        "link_occurrences",
                    )
                }
                for period, counts in sorted(interaction_series[scope].items())
            },
            "domains": {
                domain: {
                    "occurrences": count,
                    "periods": dict(sorted(domain_period_counts[scope][domain].items())),
                }
                for domain, count in top_domains
            },
            "summary": dict(_json_object(expectations["interaction-series"])["summary"]),
        }
        scenario_expectations[("attachments", timezone, bucket, "full")] = {
            "types": dict(_json_object(expectations["attachments"])["types"]),
            "availability": dict(_json_object(expectations["attachments"])["availability"]),
            "trends": {
                period: {
                    "occurrences": trend["occurrences"],
                    "unique_files": len(trend["unique_files"]),
                    "occurrence_bytes": trend["occurrence_bytes"],
                    "by_type": {
                        attachment_type: {
                            "occurrences": values["occurrences"],
                            "unique_files": len(values["unique_files"]),
                            "occurrence_bytes": values["occurrence_bytes"],
                        }
                        for attachment_type, values in sorted(trend["by_type"].items())
                    },
                }
                for period, trend in sorted(attachment_trends[scope].items())
            },
            "summary": dict(_json_object(expectations["attachments"])["summary"]),
        }

    for timezone, bucket in nlp_scopes:
        scope = (timezone, bucket)
        scenario_expectations[("nlp-trends", timezone, bucket, "full")] = {
            "source_sample_count": nlp_source_samples,
            "series": serialize_nlp_metrics(nlp_aggregates[scope]),
            "hour_of_day": serialize_nlp_metrics(nlp_hours[timezone]),
            "partial_boundary_hours_included": False,
            "non_whole_hour_timezone_offset_observed": False,
        }
        scoped_conflict = conflict_buckets[scope]
        scoped_sample_count = sum(values[0] for values in scoped_conflict.values())
        scoped_direct_count = sum(values[1] for values in scoped_conflict.values())
        scenario_expectations[("conflict-rate", timezone, bucket, "full")] = {
            "available": conflict_available,
            "series": {
                period: {
                    "sample_count": values[0],
                    "direct_conflict_window_count": values[1],
                    "direct_conflict_window_rate": values[1] / values[0] if values[0] else None,
                    "low_sample_warning": values[0] < 30,
                }
                for period, values in sorted(scoped_conflict.items())
            },
            "summary": {
                "sample_count": scoped_sample_count,
                "direct_conflict_window_count": scoped_direct_count,
                "direct_conflict_window_rate": (
                    scoped_direct_count / scoped_sample_count if scoped_sample_count else None
                ),
            },
            "relationship_ranking_enabled": False,
            "relationship_dimension_count": 0,
        }

    scenario_expectations[("nlp-trends", filter_zone_name, "hour", "date-filter")] = {
        "source_sample_count": nlp_filter_source_samples,
        "series": serialize_nlp_metrics(nlp_filter_aggregates),
        "hour_of_day": serialize_nlp_metrics(nlp_filter_hours),
        "partial_boundary_hours_included": True,
        "non_whole_hour_timezone_offset_observed": False,
    }
    filter_conflict_samples = sum(values[0] for values in conflict_filter_buckets.values())
    filter_conflict_direct = sum(values[1] for values in conflict_filter_buckets.values())
    scenario_expectations[("conflict-rate", filter_zone_name, "hour", "date-filter")] = {
        "available": conflict_available,
        "series": {
            period: {
                "sample_count": values[0],
                "direct_conflict_window_count": values[1],
                "direct_conflict_window_rate": values[1] / values[0] if values[0] else None,
                "low_sample_warning": values[0] < 30,
            }
            for period, values in sorted(conflict_filter_buckets.items())
        },
        "summary": {
            "sample_count": filter_conflict_samples,
            "direct_conflict_window_count": filter_conflict_direct,
            "direct_conflict_window_rate": (
                filter_conflict_direct / filter_conflict_samples
                if filter_conflict_samples
                else None
            ),
        },
        "relationship_ranking_enabled": False,
        "relationship_dimension_count": 0,
    }

    return expectations, scenario_expectations


def _analytics_direct_sql_reconciliation(
    connection: Any, client: LocalApiClient | None
) -> tuple[list[AcceptanceCheck], dict[str, object]]:
    if client is None:
        return (
            [
                _pending(
                    "analytics.direct_sql_dashboard_metrics",
                    "Run against the authenticated loopback API to reconcile every "
                    "Analytics page metric",
                )
            ],
            {},
        )

    filter_from_ms, filter_to_ms = _analytics_filter_probe_bounds()
    scenarios: list[tuple[tuple[str, str, str, str], dict[str, object]]] = []
    for timezone in ANALYTICS_TIMEZONES:
        scenarios.extend(
            (
                (("overview", timezone, "", "full"), {"timezone": timezone}),
                (("activity-patterns", timezone, "", "full"), {"timezone": timezone}),
            )
        )
        for bucket in ANALYTICS_TIME_BUCKETS:
            scenarios.extend(
                (
                    (
                        ("messages", timezone, bucket, "full"),
                        {"bucket": bucket, "timezone": timezone},
                    ),
                    (
                        ("interaction-patterns", timezone, bucket, "full"),
                        {"bucket": bucket, "timezone": timezone},
                    ),
                    (
                        ("interaction-series", timezone, bucket, "full"),
                        {"bucket": bucket, "timezone": timezone, "domain_limit": 20},
                    ),
                    (
                        ("attachments", timezone, bucket, "full"),
                        {"bucket": bucket, "timezone": timezone},
                    ),
                )
            )
        for bucket in ANALYTICS_NLP_TIME_BUCKETS:
            scenarios.extend(
                (
                    (
                        ("nlp-trends", timezone, bucket, "full"),
                        {"bucket": bucket, "timezone": timezone, "direction": "all"},
                    ),
                    (
                        ("conflict-rate", timezone, bucket, "full"),
                        {"bucket": bucket, "timezone": timezone},
                    ),
                )
            )
    static_scenarios: tuple[tuple[str, dict[str, object]], ...] = (
        ("network", {"limit": 40}),
        ("topics", {}),
        ("conversations", {}),
        ("storage", {}),
        ("coverage", {}),
    )
    for namespace, static_parameters in static_scenarios:
        scenarios.append(((namespace, "", "", "full"), static_parameters))
    for namespace, additional in (
        ("nlp-trends", {"direction": "all"}),
        ("conflict-rate", {}),
    ):
        scenarios.append(
            (
                (namespace, "America/New_York", "hour", "date-filter"),
                {
                    "bucket": "hour",
                    "timezone": "America/New_York",
                    "from_ms": filter_from_ms,
                    "to_ms": filter_to_ms,
                    **additional,
                },
            )
        )

    baseline_expected, expected = _analytics_direct_sql_expectations(connection)
    endpoint_stats: dict[str, Counter[str]] = {
        namespace: Counter() for namespace in ANALYTICS_DASHBOARD_ENDPOINTS
    }
    total_metrics = 0
    total_mismatches = 0
    duplicate_key_failures = 0
    scenario_digest = hashlib.sha256()
    for scenario_key, parameters in scenarios:
        namespace = scenario_key[0]
        expectation = expected.get(scenario_key)
        metric_count = _analytics_metric_leaf_count(expectation)
        api_observed = False
        duplicate_key_failure = False
        try:
            payload, _elapsed = _analytics_all_pages(client, namespace, parameters)
            api_observed = True
            actual = _analytics_metric_projection(namespace, payload)
        except _AnalyticsDuplicateKeyError:
            duplicate_key_failure = True
            actual = None
        except RuntimeError:
            actual = None
        if not api_observed or duplicate_key_failure or expectation is None:
            mismatches = max(1, metric_count)
        else:
            mismatches = _analytics_metric_mismatches(expectation, actual)
        endpoint_passed = api_observed and mismatches == 0
        total_metrics += metric_count
        total_mismatches += mismatches
        duplicate_key_failures += int(duplicate_key_failure)
        stats = endpoint_stats[namespace]
        stats["request_count"] += 1
        stats["passed_request_count"] += int(endpoint_passed)
        stats["metric_count"] += metric_count
        stats["mismatch_count"] += mismatches
        stats["api_response_count"] += int(api_observed)
        stats["duplicate_key_failure_count"] += int(duplicate_key_failure)
        scenario_digest.update(
            f"{namespace}:{scenario_key[1]}:{scenario_key[2]}:{scenario_key[3]}:"
            f"{metric_count}:{mismatches}:{int(api_observed)}".encode()
        )

    endpoint_report: dict[str, dict[str, object]] = {}
    passed_count = 0
    for namespace, stats in endpoint_stats.items():
        endpoint_passed = (
            stats["request_count"] > 0
            and stats["passed_request_count"] == stats["request_count"]
            and stats["mismatch_count"] == 0
        )
        passed_count += int(endpoint_passed)
        endpoint_report[namespace] = {
            "request_count": stats["request_count"],
            "passed_request_count": stats["passed_request_count"],
            "metric_count": stats["metric_count"],
            "mismatch_count": stats["mismatch_count"],
            "api_response_count": stats["api_response_count"],
            "duplicate_key_failure_count": stats["duplicate_key_failure_count"],
            "passed": endpoint_passed,
        }

    report: dict[str, object] = {
        "timezone_count": len(ANALYTICS_TIMEZONES),
        "time_bucket_count": len(ANALYTICS_TIME_BUCKETS),
        "nlp_time_bucket_count": len(ANALYTICS_NLP_TIME_BUCKETS),
        "date_filter_probe_count": 2,
        "endpoint_count": len(ANALYTICS_DASHBOARD_ENDPOINTS),
        "passed_endpoint_count": passed_count,
        "request_count": len(scenarios),
        "metric_count": total_metrics,
        "mismatch_count": total_mismatches,
        "duplicate_key_failure_count": duplicate_key_failures,
        "scenario_result_digest": scenario_digest.hexdigest(),
        "endpoints": endpoint_report,
        "private_dimensions_emitted": False,
    }
    passed = (
        len(baseline_expected) == len(ANALYTICS_DASHBOARD_ENDPOINTS)
        and len(expected) == len(scenarios)
        and passed_count == len(ANALYTICS_DASHBOARD_ENDPOINTS)
        and total_metrics > 0
        and total_mismatches == 0
        and duplicate_key_failures == 0
    )
    return (
        [
            _check(
                "analytics.direct_sql_dashboard_metrics",
                passed,
                expected=(
                    "all Analytics page families, buckets, timezones, and date-boundary probes "
                    "equal independent SQLite query-only metrics"
                ),
                actual=report,
                detail=(
                    "The acceptance-read-only server bypasses analytics cache reads and writes. "
                    "Evidence contains only aggregate counts, mismatches, and a result digest; "
                    "labels, domains, identifiers, response bodies, and source text stay private"
                ),
            )
        ],
        report,
    )


def _analytics_cache_integrity(
    layout: ArchiveLayout, engine: Engine
) -> tuple[list[AcceptanceCheck], dict[str, object]]:
    timezone = "America/New_York"
    expected = (
        ("overview", {"timezone": timezone}),
        (
            "messages",
            {
                "bucket": "month",
                "timezone": timezone,
                "from_ms": None,
                "to_ms": None,
                "conversation_id": None,
            },
        ),
        (
            "activity-patterns",
            {"timezone": timezone, "from_ms": None, "to_ms": None},
        ),
        (
            "interaction-patterns",
            {"bucket": "month", "timezone": timezone, "from_ms": None, "to_ms": None},
        ),
        (
            "interaction-series",
            {
                "bucket": "month",
                "timezone": timezone,
                "from_ms": None,
                "to_ms": None,
                "domain_limit": 20,
            },
        ),
        (
            "attachments",
            {"bucket": "month", "timezone": timezone, "from_ms": None, "to_ms": None},
        ),
    )
    ready: list[str] = []
    source_revision_before = analytics_source_revision(layout, engine)
    for namespace, parameters in expected:
        cache_identity = hashlib.sha256(
            json.dumps(
                {"namespace": namespace, "parameters": dict(parameters)},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        cache_path = layout.analytics_cache / f"{cache_identity}.json"
        try:
            document = json.loads(
                _read_regular_bytes(
                    cache_path,
                    label="Analytics response cache",
                    maximum_bytes=MAX_ANALYTICS_CACHE_BYTES,
                    owner_only=True,
                ).decode("utf-8")
            )
        except (RuntimeError, UnicodeDecodeError, json.JSONDecodeError):
            document = None
        if (
            isinstance(document, dict)
            and document.get("version") == ANALYTICS_CACHE_VERSION
            and document.get("source_revision") == source_revision_before
            and document.get("namespace") == namespace
            and document.get("parameters") == parameters
            and isinstance(document.get("payload"), dict)
        ):
            ready.append(namespace)
    source_revision_stable = analytics_source_revision(layout, engine) == source_revision_before
    report: dict[str, object] = {
        "timezone": timezone,
        "expected_entry_count": len(expected),
        "current_entry_count": len(ready),
        "ready_namespaces": ready,
        "source_revision_stable": source_revision_stable,
        "cache_inspection_only": True,
        "acceptance_get_requests_bypassed_cache": True,
        "get_requests_may_have_populated_cache": False,
    }
    return (
        [
            _check(
                "analytics.default_dashboard_cache_warm",
                len(ready) == len(expected) and source_revision_stable,
                expected="all six revision-bound default analytics responses are current",
                actual=report,
                detail="Cached payloads are validated transiently but never copied into evidence",
            )
        ],
        report,
    )


def _benchmark_api(
    client: LocalApiClient | None,
    connection: Any,
    *,
    repeats: int,
    query: str,
    inherited_timings: Mapping[str, Sequence[float]],
) -> tuple[list[AcceptanceCheck], dict[str, object]]:
    if client is None:
        return (
            [
                _pending(
                    "performance.live_api",
                    "Pass --base-url and supply its token by prompt or owner-only token file",
                )
            ],
            {},
        )
    # Reconciliation calls intentionally exercise dozens of different endpoints and include
    # first-use costs. Keep their observations for diagnostics, but never mix them into the
    # explicitly warmed p95 samples below (especially not into date-jump latency).
    reconciliation_summaries = {
        key: _timing_summary(values) for key, values in inherited_timings.items()
    }
    timings: dict[str, list[float]] = {}

    def sample(label: str, path: str, parameters: Mapping[str, object]) -> None:
        client.get_json(path, parameters)
        for _ in range(repeats):
            _payload, elapsed = client.get_json(path, parameters)
            timings.setdefault(label, []).append(elapsed)

    sample(
        "list_api",
        "/api/v1/conversations",
        {"sort": "last_activity", "direction": "desc", "limit": 50},
    )
    sample(
        "list_api",
        "/api/v1/media",
        {"view": "occurrences", "sort": "date", "direction": "desc", "limit": 50},
    )
    sample("keyword_top_50", "/api/v1/search", {"q": query, "mode": "keyword", "limit": 50})
    semantic_available = True
    try:
        sample(
            "semantic_top_50",
            "/api/v1/search",
            {"q": query, "mode": "semantic", "limit": 50},
        )
    except RuntimeError:
        # Exception bodies can contain query text or other private response context.
        semantic_available = False

    anchor = connection.execute(
        text(
            """
            SELECT conversation_id, timestamp_ms
            FROM messages
            WHERE deleted_at IS NULL AND timestamp_ms IS NOT NULL
            ORDER BY pk
            LIMIT 1
            """
        )
    ).first()
    if anchor is not None:
        conversation_id, timestamp_ms = anchor
        sample(
            "date_jump_api",
            f"/api/v1/conversations/{urllib.parse.quote(str(conversation_id))}/messages/date-jump",
            {"at_ms": timestamp_ms},
        )

    summaries = {key: _timing_summary(values) for key, values in timings.items()}
    checks: list[AcceptanceCheck] = []
    for label, threshold_key in (
        ("list_api", "list_api_p95"),
        ("date_jump_api", "date_jump_api_p95"),
        ("keyword_top_50", "keyword_top_50_p95"),
    ):
        p95 = summaries.get(label, {}).get("p95_ms")
        threshold = PERFORMANCE_THRESHOLDS_MS[threshold_key]
        checks.append(
            _check(
                f"performance.{label}",
                isinstance(p95, (int, float)) and p95 <= threshold,
                expected=f"p95 <= {threshold:g} ms after warm-up",
                actual=summaries.get(label, {}),
                detail="Measured against the authenticated loopback API",
            )
        )
    semantic_p95 = summaries.get("semantic_top_50", {}).get("p95_ms")
    semantic_threshold = PERFORMANCE_THRESHOLDS_MS["semantic_top_50_p95"]
    checks.append(
        _check(
            "performance.semantic_top_50",
            semantic_available
            and isinstance(semantic_p95, (int, float))
            and semantic_p95 <= semantic_threshold,
            expected=f"p95 <= {semantic_threshold:g} ms after warm-up",
            actual={
                **summaries.get("semantic_top_50", {}),
                "available": semantic_available,
            },
            detail="OpenAI latency is not involved; this is the local semantic/reranker path",
        )
    )
    report_summaries: dict[str, object] = {
        **summaries,
        "reconciliation_observations": reconciliation_summaries,
    }
    return checks, report_summaries


def _private_review_input(path: Path, *, label: str) -> tuple[Path, bytes]:
    project_root = Path(__file__).resolve().parents[1]
    review_root = project_root / ".private-review"
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate = Path(os.path.abspath(candidate))
    try:
        relative = candidate.relative_to(review_root)
    except ValueError as exc:
        raise RuntimeError(f"{label} must stay under the ignored .private-review/") from exc
    if not relative.parts:
        raise RuntimeError(f"{label} must be a file below .private-review/")
    current = review_root
    for part in ("", *relative.parts[:-1]):
        if part:
            current = current / part
        try:
            current_stat = current.lstat()
        except FileNotFoundError:
            raise RuntimeError(f"{label} has a missing private parent directory") from None
        if stat.S_ISLNK(current_stat.st_mode) or not stat.S_ISDIR(current_stat.st_mode):
            raise RuntimeError(f"{label} has an unsafe private parent directory")
        if stat.S_IMODE(current_stat.st_mode) & 0o077:
            raise RuntimeError(f"{label} parent directories must be owner-only")
    return (
        candidate,
        _read_regular_bytes(
            candidate,
            label=label,
            maximum_bytes=MAX_BROWSER_REPORT_BYTES,
            owner_only=True,
        ),
    )


def _verify_png_artifact(path: Path, relative_path: str) -> bool:
    expected = (
        (1440, 1000)
        if relative_path.startswith("artifacts/desktop-")
        else (390, 844)
        if relative_path.startswith("artifacts/mobile-")
        else None
    )
    if expected is None:
        return False
    try:
        descriptor, _file_stat = _open_regular_nofollow(path, label="Private PNG artifact")
        with os.fdopen(descriptor, "rb") as handle, Image.open(handle) as image:
            if image.format != "PNG" or image.width != expected[0] or image.height < expected[1]:
                return False
            image.verify()
    except (RuntimeError, OSError, SyntaxError, UnidentifiedImageError, ValueError):
        return False
    return True


def _verify_trace_artifact(path: Path) -> bool:
    try:
        descriptor, file_stat = _open_regular_nofollow(path, label="Private trace artifact")
        with os.fdopen(descriptor, "rb") as handle, zipfile.ZipFile(handle) as archive:
            records = archive.infolist()
            names = [record.filename for record in records]
            if (
                file_stat.st_size <= 0
                or len(records) < 2
                or len(records) > 100_000
                or len(names) != len(set(names))
                or not {"trace.trace", "trace.network"}.issubset(names)
                or any(
                    not name
                    or name.startswith(("/", "\\"))
                    or ".." in Path(name).parts
                    or record.flag_bits & 0x1
                    for name, record in zip(names, records, strict=True)
                )
                or sum(record.file_size for record in records) > 2 * 1024 * 1024 * 1024
                or archive.testzip() is not None
            ):
                return False
    except (RuntimeError, OSError, ValueError, zipfile.BadZipFile, zipfile.LargeZipFile):
        return False
    return True


def _browser_nonbool_integer(
    value: object,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int | None:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        return None
    if maximum is not None and value > maximum:
        return None
    return value


def _browser_finite_number(
    value: object,
    *,
    minimum: float = 0,
    maximum: float | None = None,
) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        return None
    if maximum is not None and number > maximum:
        return None
    return number


def _minimum_browser_interaction_proof(name: str) -> dict[str, int]:
    proof = dict.fromkeys(BROWSER_INTERACTION_PROOF_KEYS, 0)
    proof.update(BROWSER_INTERACTION_PROOF_MINIMUMS.get(name, {}))
    if name == "events_filter_sort_and_page":
        proof["source_bound_empty_dataset_count"] = 1
    return proof


def _browser_interaction_proof_valid(name: object, value: object) -> bool:
    if not isinstance(name, str) or name not in BROWSER_INTERACTION_PROOF_MINIMUMS:
        return False
    if not isinstance(value, dict) or set(value) != set(BROWSER_INTERACTION_PROOF_KEYS):
        return False
    if not all(
        _browser_nonbool_integer(value.get(key), maximum=BROWSER_INTERACTION_PROOF_METRIC_MAX)
        is not None
        for key in BROWSER_INTERACTION_PROOF_KEYS
    ):
        return False
    if any(
        int(value[key]) < minimum
        for key, minimum in BROWSER_INTERACTION_PROOF_MINIMUMS[name].items()
    ):
        return False
    return name != "events_filter_sort_and_page" or (
        int(value["nonempty_dataset_count"]) + int(value["source_bound_empty_dataset_count"]) >= 1
    )


def _browser_summary_proof_totals_valid(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != set(BROWSER_INTERACTION_PROOF_KEYS):
        return False
    if not all(
        _browser_nonbool_integer(
            value.get(key),
            maximum=(EXPECTED_BROWSER_INTERACTION_RUNS * BROWSER_INTERACTION_PROOF_METRIC_MAX),
        )
        is not None
        for key in BROWSER_INTERACTION_PROOF_KEYS
    ):
        return False
    positive_keys = set(BROWSER_INTERACTION_PROOF_KEYS) - {"source_bound_empty_dataset_count"}
    return all(int(value[key]) > 0 for key in positive_keys)


def _browser_evidence(
    path: Path | None,
    *,
    expected_archive_identity: str,
    expected_totals: Mapping[str, int],
    expected_largest: Mapping[str, object] | None,
    expected_server_identity: Mapping[str, str],
) -> tuple[list[AcceptanceCheck], dict[str, object]]:
    if path is None:
        return (
            [
                _pending(
                    "browser.initial_content_dom_wcag",
                    "Pass --browser-report from the committed real UI review harness",
                )
            ],
            {},
        )
    report_path, report_bytes = _private_review_input(path, label="Browser acceptance report")
    try:
        payload = json.loads(report_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        payload = None
    if not isinstance(payload, dict):
        raise RuntimeError("Browser acceptance report must be a JSON object")

    summary = _json_object(payload.get("summary"))
    summary_proof_totals = _json_object(summary.get("interaction_proof_totals"))
    gate = _json_object(payload.get("gate"))
    safety = _json_object(payload.get("safety"))
    catalog = _json_object(payload.get("catalog"))
    browser_server_identity = _json_object(catalog.get("acceptance_identity"))
    catalog_totals = _json_object(catalog.get("totals"))
    largest = _json_object(catalog.get("largest_conversation"))
    pages = _json_object_list(payload.get("pages"))
    interactions = _json_object_list(payload.get("interactions"))
    artifact_records = _json_object_list(payload.get("artifacts"))
    payload_schema_valid = set(payload) == {
        "schema_version",
        "generated_at",
        "runner_sha256",
        "catalog",
        "summary",
        "pages",
        "interactions",
        "artifacts",
        "gate",
        "safety",
    }
    payload_schema_valid = payload_schema_valid and set(summary) == {
        "page_runs",
        "screenshot_count",
        "initial_content_p95_ms",
        "axe_violation_count",
        "axe_incomplete_count",
        "horizontal_overflow_count",
        "page_error_count",
        "external_request_count",
        "non_get_head_request_count",
        "websocket_request_count",
        "service_worker_registration_count",
        "reduced_motion_failure_count",
        "largest_rendered_message_count",
        "largest_loaded_message_count",
        "reader_counts_valid",
        "screenshots_complete",
        "traces_complete",
        "page_evidence_complete",
        "interaction_runs",
        "interaction_pass_count",
        "interaction_evidence_complete",
        "interaction_proof_complete",
        "interaction_proof_totals",
        "desktop_and_mobile",
    }
    payload_schema_valid = payload_schema_valid and _browser_summary_proof_totals_valid(
        summary_proof_totals
    )
    payload_schema_valid = payload_schema_valid and set(gate) == {"passed"}
    payload_schema_valid = payload_schema_valid and isinstance(gate.get("passed"), bool)
    payload_schema_valid = payload_schema_valid and set(catalog) == {
        "archive_identity_sha256",
        "acceptance_identity",
        "totals",
        "largest_conversation",
    }
    browser_identity_schema_valid = set(browser_server_identity) == _ACCEPTANCE_IDENTITY_KEYS
    browser_identity_schema_valid = browser_identity_schema_valid and all(
        isinstance(browser_server_identity.get(key), str)
        and _LOWER_SHA256_PATTERN.fullmatch(str(browser_server_identity.get(key))) is not None
        for key in (
            "catalog_source_revision",
            "backend_source_sha256",
            "frontend_manifest_sha256",
            "server_nonce",
        )
    )
    browser_identity_schema_valid = browser_identity_schema_valid and (
        isinstance(browser_server_identity.get("backend_version"), str)
        and 0 < len(str(browser_server_identity.get("backend_version"))) <= 100
    )
    payload_schema_valid = payload_schema_valid and browser_identity_schema_valid
    payload_schema_valid = payload_schema_valid and set(catalog_totals) == {
        "conversations",
        "messages",
        "attachments",
        "activities",
    }
    payload_schema_valid = payload_schema_valid and set(largest) == {
        "id_sha256",
        "message_count",
    }
    payload_schema_valid = payload_schema_valid and set(safety) == {
        "aggregate_report_only",
        "screenshots_and_traces_separate",
        "owner_only_artifacts",
        "source_text_emitted_in_report",
        "token_emitted",
        "external_network_blocked",
        "websockets_blocked",
        "service_workers_blocked",
        "acceptance_read_only_server",
        "only_get_head_requests",
        "analytics_cache_reads_writes_disabled",
        "mutating_user_actions_performed",
    }
    summary_integer_fields = {
        "page_runs",
        "screenshot_count",
        "axe_violation_count",
        "axe_incomplete_count",
        "horizontal_overflow_count",
        "page_error_count",
        "external_request_count",
        "non_get_head_request_count",
        "websocket_request_count",
        "service_worker_registration_count",
        "reduced_motion_failure_count",
        "largest_rendered_message_count",
        "largest_loaded_message_count",
        "interaction_runs",
        "interaction_pass_count",
    }
    summary_boolean_fields = {
        "reader_counts_valid",
        "screenshots_complete",
        "traces_complete",
        "page_evidence_complete",
        "interaction_evidence_complete",
        "interaction_proof_complete",
        "desktop_and_mobile",
    }
    payload_schema_valid = payload_schema_valid and all(
        _browser_nonbool_integer(summary.get(key)) is not None for key in summary_integer_fields
    )
    payload_schema_valid = payload_schema_valid and (
        _browser_finite_number(summary.get("initial_content_p95_ms")) is not None
    )
    payload_schema_valid = payload_schema_valid and all(
        isinstance(summary.get(key), bool) for key in summary_boolean_fields
    )
    payload_schema_valid = payload_schema_valid and all(
        _browser_nonbool_integer(catalog_totals.get(key)) is not None
        for key in ("conversations", "messages", "attachments", "activities")
    )
    payload_schema_valid = payload_schema_valid and (
        _browser_nonbool_integer(largest.get("message_count"), minimum=1) is not None
    )
    payload_schema_valid = payload_schema_valid and all(
        isinstance(safety.get(key), bool) for key in safety
    )

    generated_at = payload.get("generated_at")
    report_age_seconds: float | None = None
    report_is_fresh = False
    if isinstance(generated_at, str):
        try:
            generated = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
            if generated.tzinfo is None:
                generated = generated.replace(tzinfo=UTC)
            report_age_seconds = (datetime.now(UTC) - generated.astimezone(UTC)).total_seconds()
            report_is_fresh = 0 <= report_age_seconds <= MAX_BROWSER_REPORT_AGE.total_seconds()
        except ValueError:
            pass

    current_runner = Path(__file__).resolve().parents[1] / "web/scripts/real-ui-acceptance.mjs"
    _runner_size, current_runner_sha256, _runner_mode = _sha256_regular_file(
        current_runner, label="Current browser acceptance runner"
    )
    runner_matches = payload.get("runner_sha256") == current_runner_sha256

    expected_catalog_totals = {
        "conversations": expected_totals["conversations"],
        "messages": expected_totals["messages"],
        "attachments": expected_totals["attachments"],
        "activities": expected_totals["activities"],
    }
    stable_server_identity_matches = bool(expected_server_identity) and all(
        browser_server_identity.get(key) == expected_server_identity.get(key)
        for key in _ACCEPTANCE_STABLE_IDENTITY_KEYS
    )
    restarted_server_nonce_differs = bool(expected_server_identity) and (
        browser_server_identity.get("server_nonce") != expected_server_identity.get("server_nonce")
    )
    catalog_totals_match = all(
        _browser_nonbool_integer(catalog_totals.get(key))
        == _browser_nonbool_integer(expected_catalog_totals.get(key))
        and _browser_nonbool_integer(catalog_totals.get(key)) is not None
        for key in expected_catalog_totals
    )
    report_largest_count = _browser_nonbool_integer(largest.get("message_count"), minimum=1)
    expected_largest_count = (
        _browser_nonbool_integer(expected_largest.get("message_count"), minimum=1)
        if isinstance(expected_largest, Mapping)
        else None
    )
    catalog_matches = (
        catalog.get("archive_identity_sha256") == expected_archive_identity
        and catalog_totals_match
        and isinstance(expected_largest, Mapping)
        and largest.get("id_sha256") == expected_largest.get("id_sha256")
        and report_largest_count is not None
        and report_largest_count == expected_largest_count
        and browser_identity_schema_valid
        and stable_server_identity_matches
        and restarted_server_nonce_differs
    )

    expected_page_keys = {
        (viewport, route)
        for viewport in EXPECTED_BROWSER_VIEWPORTS
        for route in EXPECTED_BROWSER_ROUTES
    }
    actual_page_keys: set[tuple[str, str]] = set()
    page_rows_valid = len(pages) == EXPECTED_BROWSER_PAGE_RUNS
    initial_content_values: list[float] = []
    page_screenshots: set[str] = set()
    for page in pages:
        viewport = page.get("viewport")
        route = page.get("route")
        key = (str(viewport), str(route))
        if key in actual_page_keys:
            page_rows_valid = False
        actual_page_keys.add(key)
        timing = page.get("initial_content_ms")
        validated_timing = _browser_finite_number(timing)
        if validated_timing is not None:
            initial_content_values.append(validated_timing)
        else:
            page_rows_valid = False
        expected_screenshot = f"artifacts/{viewport}-{route}.png"
        screenshot = page.get("screenshot")
        if screenshot != expected_screenshot:
            page_rows_valid = False
        elif isinstance(screenshot, str):
            page_screenshots.add(screenshot)
        page_row_schema_valid = (
            set(page)
            == {
                "viewport",
                "route",
                "content_state",
                "initial_content_ms",
                "axe_violation_count",
                "axe_incomplete_count",
                "horizontal_overflow",
                "page_error_count",
                "external_request_count",
                "non_get_head_request_count",
                "reduced_motion_passed",
                "skip_link_passed",
                "screenshot",
            }
            and isinstance(viewport, str)
            and isinstance(route, str)
            and isinstance(page.get("content_state"), str)
            and validated_timing is not None
            and all(
                _browser_nonbool_integer(page.get(field)) is not None
                for field in (
                    "axe_violation_count",
                    "axe_incomplete_count",
                    "page_error_count",
                    "external_request_count",
                    "non_get_head_request_count",
                )
            )
            and isinstance(page.get("horizontal_overflow"), bool)
            and isinstance(page.get("reduced_motion_passed"), bool)
            and isinstance(page.get("skip_link_passed"), bool)
            and isinstance(screenshot, str)
        )
        payload_schema_valid = payload_schema_valid and page_row_schema_valid
        page_rows_valid = page_rows_valid and (
            page_row_schema_valid
            and page.get("content_state") in {"ready", "empty"}
            and _browser_nonbool_integer(page.get("axe_violation_count")) == 0
            and _browser_nonbool_integer(page.get("axe_incomplete_count")) == 0
            and page.get("horizontal_overflow") is False
            and _browser_nonbool_integer(page.get("page_error_count")) == 0
            and _browser_nonbool_integer(page.get("external_request_count")) == 0
            and _browser_nonbool_integer(page.get("non_get_head_request_count")) == 0
            and page.get("reduced_motion_passed") is True
            and page.get("skip_link_passed") is True
        )
    page_rows_valid = page_rows_valid and actual_page_keys == expected_page_keys

    expected_interaction_keys = {
        (viewport, interaction)
        for viewport in EXPECTED_BROWSER_VIEWPORTS
        for interaction in EXPECTED_BROWSER_INTERACTIONS
    }
    actual_interaction_keys: set[tuple[str, str]] = set()
    interaction_rows_valid = len(interactions) == EXPECTED_BROWSER_INTERACTION_RUNS
    interaction_proof_rows_valid = len(interactions) == EXPECTED_BROWSER_INTERACTION_RUNS
    calculated_proof_totals = dict.fromkeys(BROWSER_INTERACTION_PROOF_KEYS, 0)
    for interaction in interactions:
        viewport = interaction.get("viewport")
        name = interaction.get("interaction")
        key = (str(viewport), str(name))
        duration = interaction.get("duration_ms")
        proof = _json_object(interaction.get("proof"))
        proof_valid = _browser_interaction_proof_valid(name, proof)
        interaction_proof_rows_valid = interaction_proof_rows_valid and proof_valid
        for proof_key in BROWSER_INTERACTION_PROOF_KEYS:
            proof_value = proof.get(proof_key)
            validated_proof_value = _browser_nonbool_integer(proof_value)
            if validated_proof_value is not None:
                calculated_proof_totals[proof_key] += validated_proof_value
        if key in actual_interaction_keys:
            interaction_rows_valid = False
        actual_interaction_keys.add(key)
        proof_schema_valid = isinstance(proof, dict) and set(proof) == set(
            BROWSER_INTERACTION_PROOF_KEYS
        )
        proof_schema_valid = proof_schema_valid and all(
            _browser_nonbool_integer(
                proof.get(proof_key), maximum=BROWSER_INTERACTION_PROOF_METRIC_MAX
            )
            is not None
            for proof_key in BROWSER_INTERACTION_PROOF_KEYS
        )
        interaction_row_schema_valid = (
            set(interaction) == {"viewport", "interaction", "passed", "duration_ms", "proof"}
            and isinstance(viewport, str)
            and isinstance(name, str)
            and isinstance(interaction.get("passed"), bool)
            and _browser_finite_number(duration) is not None
            and proof_schema_valid
        )
        payload_schema_valid = payload_schema_valid and interaction_row_schema_valid
        interaction_rows_valid = interaction_rows_valid and (
            interaction_row_schema_valid and interaction.get("passed") is True and proof_valid
        )
    interaction_rows_valid = (
        interaction_rows_valid and actual_interaction_keys == expected_interaction_keys
    )
    interaction_proof_rows_valid = interaction_proof_rows_valid and (
        actual_interaction_keys == expected_interaction_keys
    )
    proof_totals_reconcile = (
        _browser_summary_proof_totals_valid(summary_proof_totals)
        and summary_proof_totals == calculated_proof_totals
    )

    expected_initial_p95 = _percentile(initial_content_values, 0.95)
    summary_p95 = summary.get("initial_content_p95_ms")
    validated_summary_p95 = _browser_finite_number(summary_p95)

    def evidence_integer(value: object) -> int:
        validated = _browser_nonbool_integer(value)
        return validated if validated is not None else -1

    summary_reconciles = (
        len(initial_content_values) == EXPECTED_BROWSER_PAGE_RUNS
        and validated_summary_p95 is not None
        and expected_initial_p95 is not None
        and abs(validated_summary_p95 - expected_initial_p95) <= 0.001
        and evidence_integer(summary.get("page_runs")) == EXPECTED_BROWSER_PAGE_RUNS
        and evidence_integer(summary.get("screenshot_count")) == EXPECTED_BROWSER_PAGE_RUNS
        and evidence_integer(summary.get("axe_violation_count"))
        == sum(evidence_integer(page.get("axe_violation_count")) for page in pages)
        and evidence_integer(summary.get("axe_incomplete_count"))
        == sum(evidence_integer(page.get("axe_incomplete_count")) for page in pages)
        and evidence_integer(summary.get("horizontal_overflow_count"))
        == sum(page.get("horizontal_overflow") is True for page in pages)
        and evidence_integer(summary.get("page_error_count"))
        == sum(evidence_integer(page.get("page_error_count")) for page in pages)
        and evidence_integer(summary.get("external_request_count"))
        == sum(evidence_integer(page.get("external_request_count")) for page in pages)
        and evidence_integer(summary.get("non_get_head_request_count"))
        == sum(evidence_integer(page.get("non_get_head_request_count")) for page in pages)
        and evidence_integer(summary.get("websocket_request_count")) == 0
        and evidence_integer(summary.get("service_worker_registration_count")) == 0
        and evidence_integer(summary.get("reduced_motion_failure_count"))
        == sum(page.get("reduced_motion_passed") is not True for page in pages)
        and evidence_integer(summary.get("interaction_runs")) == EXPECTED_BROWSER_INTERACTION_RUNS
        and evidence_integer(summary.get("interaction_pass_count"))
        == sum(interaction.get("passed") is True for interaction in interactions)
        and summary.get("interaction_evidence_complete") is interaction_rows_valid
        and summary.get("interaction_proof_complete") is interaction_proof_rows_valid
        and proof_totals_reconcile
    )

    artifact_manifest = {str(record.get("relative_path")): record for record in artifact_records}
    artifact_record_schemas_valid = all(
        set(record) == {"relative_path", "size_bytes", "sha256"}
        and isinstance(record.get("relative_path"), str)
        and _browser_nonbool_integer(record.get("size_bytes"), minimum=1) is not None
        and isinstance(record.get("sha256"), str)
        and _LOWER_SHA256_PATTERN.fullmatch(str(record.get("sha256"))) is not None
        for record in artifact_records
    )
    payload_schema_valid = payload_schema_valid and artifact_record_schemas_valid
    artifact_rows_valid = (
        len(artifact_records) == len(EXPECTED_BROWSER_ARTIFACTS)
        and set(artifact_manifest) == EXPECTED_BROWSER_ARTIFACTS
        and artifact_record_schemas_valid
    )
    verified_png_count = 0
    verified_trace_count = 0
    artifacts_directory = report_path.parent / "artifacts"
    try:
        artifacts_stat = artifacts_directory.lstat()
        artifact_rows_valid = artifact_rows_valid and (
            stat.S_ISDIR(artifacts_stat.st_mode)
            and not stat.S_ISLNK(artifacts_stat.st_mode)
            and stat.S_IMODE(artifacts_stat.st_mode) == OWNER_DIRECTORY_MODE
        )
        actual_artifact_paths = {
            f"artifacts/{child.name}" for child in artifacts_directory.iterdir()
        }
        artifact_rows_valid = artifact_rows_valid and (
            actual_artifact_paths == EXPECTED_BROWSER_ARTIFACTS
        )
    except OSError:
        artifact_rows_valid = False
    if artifact_rows_valid:
        for relative_path in sorted(EXPECTED_BROWSER_ARTIFACTS):
            record = artifact_manifest[relative_path]
            artifact_path = report_path.parent / relative_path
            try:
                size_bytes, sha256, mode = _sha256_regular_file(
                    artifact_path, label="Private browser artifact"
                )
            except RuntimeError:
                artifact_rows_valid = False
                break
            if (
                _browser_nonbool_integer(record.get("size_bytes"), minimum=1) != size_bytes
                or record.get("sha256") != sha256
                or mode != OWNER_FILE_MODE
            ):
                artifact_rows_valid = False
                break
            if relative_path.endswith(".png"):
                if not _verify_png_artifact(artifact_path, relative_path):
                    artifact_rows_valid = False
                    break
                verified_png_count += 1
            elif relative_path.endswith(".zip"):
                if not _verify_trace_artifact(artifact_path):
                    artifact_rows_valid = False
                    break
                verified_trace_count += 1
            else:
                artifact_rows_valid = False
                break
    artifact_rows_valid = artifact_rows_valid and page_screenshots == {
        value for value in EXPECTED_BROWSER_ARTIFACTS if value.endswith(".png")
    }
    artifact_rows_valid = artifact_rows_valid and (
        verified_png_count == EXPECTED_BROWSER_PAGE_RUNS
        and verified_trace_count == len(EXPECTED_BROWSER_VIEWPORTS)
    )

    safety_valid = all(
        safety.get(key) is expected
        for key, expected in {
            "aggregate_report_only": True,
            "screenshots_and_traces_separate": True,
            "owner_only_artifacts": True,
            "source_text_emitted_in_report": False,
            "token_emitted": False,
            "external_network_blocked": True,
            "websockets_blocked": True,
            "service_workers_blocked": True,
            "acceptance_read_only_server": True,
            "only_get_head_requests": True,
            "analytics_cache_reads_writes_disabled": True,
            "mutating_user_actions_performed": False,
        }.items()
    )

    # Import only aggregate evidence. Screenshots, text, titles, filenames, URLs, artifact
    # digests, and per-page private content never enter the catalog acceptance report.
    sanitized = {
        "page_runs": summary.get("page_runs"),
        "screenshot_count": summary.get("screenshot_count"),
        "initial_content_p95_ms": summary.get("initial_content_p95_ms"),
        "axe_violation_count": summary.get("axe_violation_count"),
        "axe_incomplete_count": summary.get("axe_incomplete_count"),
        "horizontal_overflow_count": summary.get("horizontal_overflow_count"),
        "page_error_count": summary.get("page_error_count"),
        "external_request_count": summary.get("external_request_count"),
        "non_get_head_request_count": summary.get("non_get_head_request_count"),
        "websocket_request_count": summary.get("websocket_request_count"),
        "service_worker_registration_count": summary.get("service_worker_registration_count"),
        "reduced_motion_failure_count": summary.get("reduced_motion_failure_count"),
        "largest_rendered_message_count": summary.get("largest_rendered_message_count"),
        "largest_loaded_message_count": summary.get("largest_loaded_message_count"),
        "reader_counts_valid": summary.get("reader_counts_valid"),
        "screenshots_complete": summary.get("screenshots_complete"),
        "traces_complete": summary.get("traces_complete"),
        "desktop_and_mobile": summary.get("desktop_and_mobile"),
        "page_evidence_complete": summary.get("page_evidence_complete"),
        "interaction_runs": summary.get("interaction_runs"),
        "interaction_pass_count": summary.get("interaction_pass_count"),
        "interaction_evidence_complete": summary.get("interaction_evidence_complete"),
        "interaction_proof_complete": interaction_proof_rows_valid,
        "interaction_proof_totals": dict(calculated_proof_totals),
        "report_fresh_within_24_hours": report_is_fresh,
        "report_age_seconds": (
            round(report_age_seconds, 1) if report_age_seconds is not None else None
        ),
        "runner_matches": runner_matches,
        "catalog_matches": catalog_matches,
        "acceptance_identity_schema_reconciles": browser_identity_schema_valid,
        "stable_server_identity_matches": stable_server_identity_matches,
        "restarted_server_nonce_differs": restarted_server_nonce_differs,
        "page_rows_reconcile": page_rows_valid and summary_reconciles,
        "interaction_rows_reconcile": interaction_rows_valid,
        "interaction_proof_totals_reconcile": proof_totals_reconcile,
        "artifact_inventory_reconciles": artifact_rows_valid,
        "report_object_schemas_reconcile": payload_schema_valid,
        "decoded_png_count": verified_png_count,
        "verified_trace_zip_count": verified_trace_count,
        "report_schema_version_matches": (
            payload.get("schema_version") == BROWSER_REPORT_SCHEMA_VERSION
        ),
    }
    largest_rendered_message_count = _browser_nonbool_integer(
        sanitized["largest_rendered_message_count"], minimum=1, maximum=100
    )
    largest_loaded_message_count = _browser_nonbool_integer(
        sanitized["largest_loaded_message_count"], minimum=1, maximum=250
    )
    passed = (
        payload.get("schema_version") == BROWSER_REPORT_SCHEMA_VERSION
        and payload_schema_valid
        and gate.get("passed") is True
        and report_is_fresh
        and runner_matches
        and catalog_matches
        and safety_valid
        and page_rows_valid
        and interaction_rows_valid
        and summary_reconciles
        and artifact_rows_valid
        and _browser_nonbool_integer(sanitized["page_runs"]) == EXPECTED_BROWSER_PAGE_RUNS
        and _browser_nonbool_integer(sanitized["screenshot_count"]) == EXPECTED_BROWSER_PAGE_RUNS
        and _browser_finite_number(
            sanitized["initial_content_p95_ms"],
            maximum=PERFORMANCE_THRESHOLDS_MS["initial_page_content_p95"],
        )
        is not None
        and _browser_nonbool_integer(sanitized["axe_violation_count"]) == 0
        and _browser_nonbool_integer(sanitized["axe_incomplete_count"]) == 0
        and _browser_nonbool_integer(sanitized["horizontal_overflow_count"]) == 0
        and _browser_nonbool_integer(sanitized["page_error_count"]) == 0
        and _browser_nonbool_integer(sanitized["external_request_count"]) == 0
        and _browser_nonbool_integer(sanitized["non_get_head_request_count"]) == 0
        and _browser_nonbool_integer(sanitized["websocket_request_count"]) == 0
        and _browser_nonbool_integer(sanitized["service_worker_registration_count"]) == 0
        and _browser_nonbool_integer(sanitized["reduced_motion_failure_count"]) == 0
        and largest_rendered_message_count is not None
        and largest_loaded_message_count is not None
        and largest_rendered_message_count <= largest_loaded_message_count
        and sanitized["reader_counts_valid"] is True
        and sanitized["screenshots_complete"] is True
        and sanitized["traces_complete"] is True
        and sanitized["desktop_and_mobile"] is True
        and sanitized["page_evidence_complete"] is True
        and _browser_nonbool_integer(sanitized["interaction_runs"])
        == EXPECTED_BROWSER_INTERACTION_RUNS
        and _browser_nonbool_integer(sanitized["interaction_pass_count"])
        == EXPECTED_BROWSER_INTERACTION_RUNS
        and sanitized["interaction_evidence_complete"] is True
        and sanitized["interaction_proof_complete"] is True
        and sanitized["interaction_proof_totals_reconcile"] is True
    )
    return (
        [
            _check(
                "browser.initial_content_dom_wcag",
                passed,
                expected={
                    "initial_content_p95_ms": "<=2500",
                    "axe_violations": 0,
                    "axe_incomplete": 0,
                    "overflow": 0,
                    "page_errors": 0,
                    "external_requests": 0,
                    "non_get_head_requests": 0,
                    "websocket_requests": 0,
                    "service_worker_registrations": 0,
                    "reduced_motion_failures": 0,
                    "bounded_reader_dom": True,
                    "private_screenshots_complete": True,
                    "private_traces_complete": True,
                    "desktop_and_mobile": True,
                    "read_only_interaction_matrix_complete": True,
                    "rendered_data_proof_complete": True,
                    "fresh_runner_catalog_artifact_binding": True,
                },
                actual=sanitized,
                detail=(
                    "Aggregate browser evidence only; screenshots and traces remain separate "
                    "and private"
                ),
            )
        ],
        sanitized,
    )


@dataclass(frozen=True, slots=True)
class _SpotDatabaseStat:
    device: int
    inode: int
    size_bytes: int
    mtime_ns: int


@dataclass(frozen=True, slots=True)
class _SpotFileBinding:
    device: int
    inode: int
    size_bytes: int
    mtime_ns: int
    ctime_ns: int
    sha256: str


class _SpotApiBudget:
    """Fail closed before the production spot check can exceed its request budget."""

    def __init__(self, client: LocalApiClient, maximum: int) -> None:
        self.client = client
        self.maximum = maximum
        self.calls = 0

    def get_json(
        self,
        path: str,
        parameters: Mapping[str, object] | None = None,
        *,
        timeout: float = 120.0,
    ) -> tuple[dict[str, Any], float]:
        if self.calls >= self.maximum:
            raise RuntimeError("Production spot-check API request budget exhausted")
        self.calls += 1
        if isinstance(self.client, LocalApiClient):
            return self.client.get_json(
                path,
                parameters,
                timeout=timeout,
                maximum_bytes=PRODUCTION_SPOT_CHECK_MAX_API_RESPONSE_BYTES,
            )
        return self.client.get_json(path, parameters, timeout=timeout)


def _spot_progress(stage: int, total: int, label: str) -> None:
    """Emit only fixed, non-private progress fields and flush them immediately."""

    print(
        json.dumps(
            {
                "profile": PRODUCTION_SPOT_CHECK_PROFILE_VERSION,
                "stage": stage,
                "stage_count": total,
                "status": label,
            },
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )


def _spot_database_stat(path: Path) -> _SpotDatabaseStat:
    descriptor, opened = _open_regular_nofollow(path, label="Catalog database")
    try:
        current = os.fstat(descriptor)
        if (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
        ) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
        ):
            raise RuntimeError("Catalog database changed while its identity was captured")
        return _SpotDatabaseStat(
            device=int(current.st_dev),
            inode=int(current.st_ino),
            size_bytes=int(current.st_size),
            mtime_ns=int(current.st_mtime_ns),
        )
    finally:
        os.close(descriptor)


def _spot_file_binding(
    path: Path,
    *,
    maximum_bytes: int,
) -> _SpotFileBinding:
    descriptor, opened = _open_regular_nofollow(path, label="Production spot-check binding")
    try:
        if stat.S_IMODE(opened.st_mode) & 0o077:
            raise RuntimeError("Production spot-check binding is not owner-only")
        if opened.st_size > maximum_bytes:
            raise RuntimeError("Production spot-check binding exceeds its byte cap")
        digest = hashlib.sha256()
        observed_bytes = 0
        while chunk := os.read(descriptor, min(1_048_576, maximum_bytes + 1 - observed_bytes)):
            observed_bytes += len(chunk)
            if observed_bytes > maximum_bytes:
                raise RuntimeError("Production spot-check binding exceeds its byte cap")
            digest.update(chunk)
        closed_over = os.fstat(descriptor)
        if (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        ) != (
            closed_over.st_dev,
            closed_over.st_ino,
            closed_over.st_size,
            closed_over.st_mtime_ns,
            closed_over.st_ctime_ns,
        ) or observed_bytes != opened.st_size:
            raise RuntimeError("Production spot-check binding changed while read")
        return _SpotFileBinding(
            device=int(opened.st_dev),
            inode=int(opened.st_ino),
            size_bytes=int(opened.st_size),
            mtime_ns=int(opened.st_mtime_ns),
            ctime_ns=int(opened.st_ctime_ns),
            sha256=digest.hexdigest(),
        )
    finally:
        os.close(descriptor)


def _spot_sqlite_sidecars(database: Path) -> dict[str, bool]:
    absent: dict[str, bool] = {}
    for suffix in ("-wal", "-shm", "-journal"):
        try:
            database.with_name(f"{database.name}{suffix}").lstat()
        except FileNotFoundError:
            absent[suffix] = True
        except OSError:
            absent[suffix] = False
        else:
            # A broken symlink is still a sidecar-path finding, so never use Path.exists().
            absent[suffix] = False
    return absent


def _spot_baseline_binding_check(
    before: _SpotFileBinding | None,
    after: _SpotFileBinding | None,
    *,
    schema_identity_valid: bool,
) -> tuple[AcceptanceCheck, dict[str, object]]:
    stable = before is not None and before == after
    report: dict[str, object] = {
        "observed": before is not None,
        "schema_identity_valid": schema_identity_valid,
        "stable_during_audit": stable,
        "file_identity_stable": bool(
            before is not None
            and after is not None
            and (before.device, before.inode) == (after.device, after.inode)
        ),
        "size_stable": bool(
            before is not None and after is not None and before.size_bytes == after.size_bytes
        ),
        "mtime_stable": bool(
            before is not None and after is not None and before.mtime_ns == after.mtime_ns
        ),
        "ctime_stable": bool(
            before is not None and after is not None and before.ctime_ns == after.ctime_ns
        ),
        "digest_stable": bool(
            before is not None and after is not None and before.sha256 == after.sha256
        ),
        "sha256": before.sha256 if before is not None else None,
        "private_metrics_emitted": False,
    }
    return (
        _check(
            "catalog.production_spot_check_audited_baseline_stable",
            bool(stable and schema_identity_valid),
            expected="current valid audited-baseline file remains byte-identical during audit",
            actual=report,
            detail="Only the file digest and aggregate validity are emitted; metrics stay private",
        ),
        report,
    )


def _spot_sidecar_check(
    before: Mapping[str, bool],
    after: Mapping[str, bool],
) -> tuple[AcceptanceCheck, dict[str, object]]:
    expected_suffixes = {"-wal", "-shm", "-journal"}
    exact_before = set(before) == expected_suffixes
    exact_after = set(after) == expected_suffixes
    absent_before = exact_before and all(before.values())
    absent_after = exact_after and all(after.values())
    report: dict[str, object] = {
        "exact_sidecar_set_at_start": exact_before,
        "exact_sidecar_set_at_end": exact_after,
        "all_absent_at_start": absent_before,
        "all_absent_at_end": absent_after,
        "late_sidecar_detected": absent_before and not absent_after,
    }
    return (
        _check(
            "catalog.production_spot_check_sqlite_sidecars_absent",
            bool(absent_before and absent_after),
            expected="SQLite WAL, SHM, and rollback-journal paths absent at start and end",
            actual=report,
            detail="Sidecar paths are fixed and are not emitted",
        ),
        report,
    )


def _spot_refetch_server_identity(
    client: _SpotApiBudget | None,
    initial_identity: Mapping[str, str],
) -> tuple[list[AcceptanceCheck], dict[str, object]]:
    if client is None or not initial_identity:
        return (
            [
                _pending(
                    "api.production_spot_check_identity_stable",
                    "A verified acceptance-read-only server is required",
                )
            ],
            {"available": False},
        )
    identity_checks, _identity_report, final_identity = _acceptance_server_identity(
        cast(LocalApiClient, client)
    )
    identity_response_valid = len(identity_checks) == 1 and identity_checks[0].status == "pass"
    stable_revisions = all(
        final_identity.get(key) == initial_identity.get(key)
        for key in _ACCEPTANCE_STABLE_IDENTITY_KEYS
    )
    nonce_stable = final_identity.get("server_nonce") == initial_identity.get("server_nonce")
    passed = bool(identity_response_valid and stable_revisions and nonce_stable)
    report: dict[str, object] = {
        "available": bool(final_identity),
        "identity_response_valid": identity_response_valid,
        "stable_revisions": stable_revisions,
        "server_nonce_stable": nonce_stable,
        "identity_values_emitted": False,
    }
    return (
        [
            _check(
                "api.production_spot_check_identity_stable",
                passed,
                expected="same acceptance server revisions and nonce at audit start and end",
                actual=report,
                detail="Raw identity values are kept transient",
            )
        ],
        report,
    )


def _spot_artifact_record(
    layout: ArchiveLayout,
    path: Path,
    *,
    jsonl: bool,
    maximum_bytes: int,
) -> dict[str, object]:
    descriptor, opened = _open_regular_nofollow(path, label="Recorded integrity artifact")
    try:
        if stat.S_IMODE(opened.st_mode) & 0o077:
            raise RuntimeError("Recorded integrity artifact is not owner-only")
        if opened.st_size > maximum_bytes:
            raise RuntimeError("Recorded integrity artifact exceeds the spot-check byte budget")
        digest = hashlib.sha256()
        byte_count = 0
        newline_count = 0
        final_byte = b""
        while chunk := os.read(descriptor, 1_048_576):
            byte_count += len(chunk)
            if byte_count > maximum_bytes:
                raise RuntimeError("Recorded integrity artifact exceeds the spot-check byte budget")
            digest.update(chunk)
            newline_count += chunk.count(b"\n")
            final_byte = chunk[-1:]
        closed_over = os.fstat(descriptor)
        stable = (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        ) == (
            closed_over.st_dev,
            closed_over.st_ino,
            closed_over.st_size,
            closed_over.st_mtime_ns,
            closed_over.st_ctime_ns,
        )
        if not stable or byte_count != opened.st_size:
            raise RuntimeError("Recorded integrity artifact changed while it was read")
        return {
            "path": path.relative_to(layout.root).as_posix(),
            "sha256": digest.hexdigest(),
            "line_count": (
                newline_count + int(byte_count > 0 and final_byte != b"\n") if jsonl else None
            ),
            "size_bytes": int(opened.st_size),
        }
    finally:
        os.close(descriptor)


_SPOT_BASELINE_KEYS = frozenset(
    {
        "baseline.manifest",
        "baseline.json_files",
        "baseline.transcript_containers",
        "baseline.ordinary_segments",
        "baseline.ordinary_messages",
        "baseline.support_segments",
        "baseline.support_messages",
        "baseline.exact_duplicate_messages",
        "baseline.unresolved_local_uri_occurrences",
        "baseline.unlinked_media_paths",
        "baseline.leading_message_gap_segments",
        "baseline.internal_message_gap_segments",
        "baseline.activity_reactions_raw",
        "baseline.activity_reactions_logical",
    }
)


def _spot_reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError("Integrity evidence contains duplicate JSON keys")
        result[key] = value
    return result


def _spot_prior_deep_report(
    layout: ArchiveLayout,
) -> tuple[list[AcceptanceCheck], dict[str, object], dict[str, int], int | None]:
    """Validate prior deep proof and re-hash only its small manifest/marker bindings."""

    summary: dict[str, object] = {
        "report_observed": False,
        "owner_only": False,
        "schema_valid": False,
        "all_mandatory_checks_pass": False,
        "deep_verified": False,
        "required_checks_present": False,
        "prior_report_baseline_checks_pass": False,
        "source_manifest_artifacts_current": False,
        "library_marker_artifact_current": False,
        "library_marker_bound_to_manifests": False,
    }
    marker_counts: dict[str, int] = {}
    generated_ns: int | None = None
    report_path = layout.manifests / "integrity-reports" / "latest.json"
    try:
        raw_report = _read_regular_bytes(
            report_path,
            label="Prior deep verification report",
            maximum_bytes=PRODUCTION_SPOT_CHECK_REPORT_MAX_BYTES,
            owner_only=True,
        )
        summary["report_observed"] = True
        summary["owner_only"] = True
        report = json.loads(
            raw_report.decode("utf-8"), object_pairs_hook=_spot_reject_duplicate_json_keys
        )
        if not isinstance(report, dict) or report.get("schema_version") != 1:
            raise RuntimeError("Prior deep verification report has an invalid schema")
        summary["schema_valid"] = True
        raw_checks = report.get("checks")
        if (
            not isinstance(raw_checks, list)
            or not raw_checks
            or not all(isinstance(item, dict) for item in raw_checks)
        ):
            raise RuntimeError("Prior deep verification report has incomplete checks")
        report_checks = [dict(item) for item in raw_checks]
        keys = [item.get("key") for item in report_checks]
        keys_valid = all(isinstance(key, str) and key for key in keys) and len(keys) == len(
            set(keys)
        )
        if not keys_valid:
            raise RuntimeError("Prior deep verification report has invalid check identities")
        key_set = {str(key) for key in keys}
        mandatory_pass = not mandatory_non_pass_checks(report_checks)
        summary["all_mandatory_checks_pass"] = mandatory_pass
        summary["required_checks_present"] = key_set >= SOURCE_RETIREMENT_REQUIRED_CHECK_KEYS
        deep_rows = [item for item in report_checks if item.get("key") == "deep.source_bytes"]
        summary["deep_verified"] = len(deep_rows) == 1 and deep_rows[0].get("status") == "pass"
        summary["prior_report_baseline_checks_pass"] = key_set >= _SPOT_BASELINE_KEYS and all(
            next(item for item in report_checks if item.get("key") == key).get("status") == "pass"
            for key in _SPOT_BASELINE_KEYS
        )
        gate = _json_object(report.get("gate"))
        gate_valid = all(
            gate.get(key) is True
            for key in (
                "all_mandatory_checks_pass",
                "deep_verified",
                "library_verified",
                "required_artifacts_present",
                "source_retirement_eligible",
            )
        )
        generated_at = report.get("generated_at")
        if isinstance(generated_at, str):
            generated = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
            if generated.tzinfo is None:
                generated = generated.replace(tzinfo=UTC)
            generated_ns = int(generated.astimezone(UTC).timestamp() * 1_000_000_000)
        if generated_ns is None or generated_ns > time.time_ns() + 60_000_000_000:
            raise RuntimeError("Prior deep verification report has an invalid timestamp")

        current_manifests = {
            "original_files": _spot_artifact_record(
                layout,
                layout.original_files_manifest,
                jsonl=True,
                maximum_bytes=512 * 1024 * 1024,
            ),
            "original_directories": _spot_artifact_record(
                layout,
                layout.original_directories_manifest,
                jsonl=True,
                maximum_bytes=512 * 1024 * 1024,
            ),
            "relocation": _spot_artifact_record(
                layout,
                layout.relocation_manifest,
                jsonl=True,
                maximum_bytes=512 * 1024 * 1024,
            ),
        }
        artifacts = _json_object(report.get("artifacts"))
        summary["source_manifest_artifacts_current"] = (
            artifacts.get("source_manifests") == current_manifests
        )
        current_marker = _spot_artifact_record(
            layout,
            layout.library_completion_marker,
            jsonl=False,
            maximum_bytes=PRODUCTION_SPOT_CHECK_LIBRARY_MARKER_MAX_BYTES,
        )
        summary["library_marker_artifact_current"] = (
            artifacts.get("library_marker") == current_marker
        )
        marker_bytes = _read_regular_bytes(
            layout.library_completion_marker,
            label="Library completion marker",
            maximum_bytes=PRODUCTION_SPOT_CHECK_LIBRARY_MARKER_MAX_BYTES,
            owner_only=True,
        )
        marker = json.loads(
            marker_bytes.decode("utf-8"),
            object_pairs_hook=_spot_reject_duplicate_json_keys,
        )
        marker_counts_raw = marker.get("counts") if isinstance(marker, dict) else None
        if isinstance(marker_counts_raw, dict) and all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in marker_counts_raw.values()
        ):
            marker_counts = {str(key): int(value) for key, value in marker_counts_raw.items()}
        summary["library_marker_bound_to_manifests"] = bool(
            isinstance(marker, dict)
            and marker.get("format_version") == 1
            and marker.get("source_manifests") == current_manifests
            and marker_counts
        )
        passed = bool(
            gate_valid
            and all(
                summary[key] is True
                for key in (
                    "owner_only",
                    "schema_valid",
                    "all_mandatory_checks_pass",
                    "deep_verified",
                    "required_checks_present",
                    "prior_report_baseline_checks_pass",
                    "source_manifest_artifacts_current",
                    "library_marker_artifact_current",
                    "library_marker_bound_to_manifests",
                )
            )
        )
    except (AuditedBaselineManifestError, OSError, RuntimeError, UnicodeDecodeError, ValueError):
        passed = False
    safe_summary = {
        **summary,
        "required_deep_check_count": len(SOURCE_RETIREMENT_REQUIRED_CHECK_KEYS),
        "required_baseline_check_count": len(_SPOT_BASELINE_KEYS),
        "private_values_emitted": False,
    }
    return (
        [
            _check(
                "catalog.prior_deep_integrity_evidence",
                passed,
                expected=(
                    "owner-only all-pass deep report bound to current source manifests, "
                    "library marker, and its recorded baseline-check statuses"
                ),
                actual=safe_summary,
                detail=(
                    "The spot check re-hashes only recorded manifest and marker artifacts; it "
                    "does not re-hash every source or blob byte"
                ),
            )
        ],
        safe_summary,
        marker_counts,
        generated_ns,
    )


def _spot_targeted_sort_values(
    connection: Any,
    *,
    entity: str,
    sort_name: str,
    identities: Sequence[str],
) -> dict[str, object]:
    """Load sort keys only for API-returned identities through primary-key predicates."""

    if len(identities) > PRODUCTION_SPOT_CHECK_MAX_TARGETED_LOOKUP_ROWS:
        raise RuntimeError("Production spot-check targeted lookup row cap exceeded")
    if len(identities) != len(set(identities)):
        raise RuntimeError("Production spot-check received duplicate sampled identities")
    if not identities:
        return {}
    statements = {
        ("conversation", "messages"): (
            "SELECT id, message_count AS sort_value FROM conversations "
            "WHERE deleted_at IS NULL AND id IN ({placeholders})"
        ),
        ("conversation", "last_activity"): (
            "SELECT id, last_activity_ms AS sort_value FROM conversations "
            "WHERE deleted_at IS NULL AND id IN ({placeholders})"
        ),
        ("media", "date"): (
            "SELECT id, timestamp_ms AS sort_value FROM attachment_occurrences "
            "WHERE deleted_at IS NULL AND id IN ({placeholders})"
        ),
        ("media", "size"): (
            "SELECT id, size_bytes AS sort_value FROM attachment_occurrences "
            "WHERE deleted_at IS NULL AND id IN ({placeholders})"
        ),
        ("activity", "date"): (
            "SELECT id, timestamp_ms AS sort_value FROM activity_records "
            "WHERE deleted_at IS NULL AND id IN ({placeholders})"
        ),
        ("activity", "category"): (
            "SELECT id, category AS sort_value FROM activity_records "
            "WHERE deleted_at IS NULL AND id IN ({placeholders})"
        ),
        ("event", "date"): (
            "SELECT activity.id, activity.timestamp_ms AS sort_value "
            "FROM activity_records AS activity "
            "JOIN source_files AS source ON source.id = activity.source_file_id "
            "WHERE activity.deleted_at IS NULL AND activity.id IN ({placeholders}) "
            "AND (activity.category = 'events' "
            "OR lower(source.normalized_path) LIKE '%/events/%')"
        ),
        ("event", "title"): (
            "SELECT activity.id, coalesce(activity.title_search, '') AS sort_value "
            "FROM activity_records AS activity "
            "JOIN source_files AS source ON source.id = activity.source_file_id "
            "WHERE activity.deleted_at IS NULL AND activity.id IN ({placeholders}) "
            "AND (activity.category = 'events' "
            "OR lower(source.normalized_path) LIKE '%/events/%')"
        ),
    }
    template = statements.get((entity, sort_name))
    if template is None:
        raise ValueError("Unknown production spot-check sort stream")
    placeholders = ", ".join(f":spot_id_{index}" for index in range(len(identities)))
    parameters = {f"spot_id_{index}": identity for index, identity in enumerate(identities)}
    rows = _rows(
        connection,
        template.format(placeholders=placeholders),
        parameters,
    )
    values = {str(row.id): row.sort_value for row in rows}
    if set(values) != set(identities) or len(rows) != len(identities):
        raise RuntimeError("Production spot-check targeted lookup did not resolve every sample")
    return values


def _spot_sorted_identities(
    identities: Sequence[str],
    values: Mapping[str, object],
    *,
    direction: str,
) -> list[str]:
    if direction not in {"asc", "desc"}:
        raise ValueError("Invalid production spot-check sort direction")
    reverse = direction == "desc"
    known = [identity for identity in identities if values.get(identity) is not None]
    unknown = [identity for identity in identities if values.get(identity) is None]
    known.sort(key=lambda identity: (values[identity], identity), reverse=reverse)
    unknown.sort(reverse=reverse)
    return [*known, *unknown]


def _spot_cursor_stream_sample(
    connection: Any,
    client: _SpotApiBudget,
    *,
    path: str,
    parameters: Mapping[str, object],
    entity: str,
    sort_name: str,
    direction: str,
    expected_total: int,
) -> tuple[dict[str, object], list[float]]:
    """Validate a fixed API cursor sample without independently sorting the corpus."""

    identities: list[str] = []
    timings: list[float] = []
    issues: list[str] = []
    seen_cursors: set[str] = set()
    cursor: str | None = None
    outbound_cursor = False
    for _page_index in range(PRODUCTION_SPOT_CHECK_PAGES_PER_STREAM):
        request_parameters: dict[str, object] = {
            **parameters,
            "limit": PRODUCTION_SPOT_CHECK_PAGE_SIZE,
        }
        if cursor is not None:
            request_parameters["cursor"] = cursor
        try:
            payload, elapsed = client.get_json(path, request_parameters)
        except RuntimeError:
            issues.append("api_request_failed")
            break
        timings.append(elapsed)
        raw_items = payload.get("items")
        items = _json_object_list(raw_items)
        if not isinstance(raw_items, list) or len(items) != len(raw_items):
            issues.append("malformed_items")
        if len(items) > PRODUCTION_SPOT_CHECK_PAGE_SIZE:
            issues.append("page_limit_exceeded")
        page_ids: list[str] = []
        for item in items:
            identity = item.get("id")
            if not isinstance(identity, str) or not identity:
                issues.append("missing_identity")
                continue
            page_ids.append(identity)
        if len(page_ids) != len(set(page_ids)) or set(page_ids) & set(identities):
            issues.append("duplicate_identity")
        identities.extend(page_ids)
        if not page_ids and expected_total > len(identities):
            issues.append("non_progress_page")
        if len(identities) > expected_total:
            issues.append("sample_exceeds_total")

        page = _json_object(payload.get("page"))
        if page.get("limit") != PRODUCTION_SPOT_CHECK_PAGE_SIZE:
            issues.append("page_limit_mismatch")
        actual_total = page.get("total")
        count_is_unknown = page.get("count_is_unknown") is True
        total_matches = (
            isinstance(actual_total, int)
            and not isinstance(actual_total, bool)
            and actual_total == expected_total
        ) or (expected_total == 0 and actual_total is None and count_is_unknown)
        if not total_matches:
            issues.append("total_mismatch")
        raw_next = page.get("next_cursor")
        next_cursor = raw_next if isinstance(raw_next, str) and raw_next else None
        if raw_next is not None and next_cursor is None:
            issues.append("malformed_cursor")
        more_expected = len(identities) < expected_total
        if more_expected and len(page_ids) < PRODUCTION_SPOT_CHECK_PAGE_SIZE:
            issues.append("short_nonterminal_page")
        if more_expected and next_cursor is None:
            issues.append("missing_continuation_cursor")
        if not more_expected and next_cursor is not None:
            issues.append("unexpected_terminal_cursor")
        if next_cursor is not None:
            if next_cursor in seen_cursors:
                issues.append("cursor_cycle")
                break
            seen_cursors.add(next_cursor)
        cursor = next_cursor
        if cursor is None:
            if len(identities) != expected_total:
                issues.append("terminal_count_mismatch")
            break
        outbound_cursor = True

    if len(identities) > PRODUCTION_SPOT_CHECK_MAX_TARGETED_LOOKUP_ROWS:
        issues.append("targeted_lookup_cap_exceeded")
    try:
        values = _spot_targeted_sort_values(
            connection,
            entity=entity,
            sort_name=sort_name,
            identities=identities,
        )
        expected_sample_order = _spot_sorted_identities(
            identities,
            values,
            direction=direction,
        )
        order_matches = identities == expected_sample_order
    except (RuntimeError, TypeError, ValueError):
        order_matches = False
        issues.append("targeted_lookup_failed")
    if not order_matches:
        issues.append("sample_order_mismatch")
    issues = list(dict.fromkeys(issues))
    full_stream = len(identities) == expected_total and cursor is None
    coverage = (
        "empty"
        if expected_total == 0
        else "complete_small_stream"
        if full_stream
        else "bounded_api_prefix_sample"
    )
    return (
        {
            "passed": not issues,
            "coverage": coverage,
            "expected_total": expected_total,
            "rows_checked": len(identities),
            "pages_checked": len(timings),
            "cursor_boundaries_checked": max(0, len(timings) - 1),
            "outbound_cursor_observed": outbound_cursor,
            "full_stream": full_stream,
            "sample_order_matches_targeted_values": order_matches,
            "targeted_lookup_rows": len(identities),
            "issue_count": len(issues),
            "issues": issues,
        },
        timings,
    )


def _spot_api_reconciliation(
    connection: Any,
    client: _SpotApiBudget | None,
    totals: Mapping[str, int],
) -> tuple[list[AcceptanceCheck], dict[str, object], dict[str, list[float]]]:
    if client is None:
        return (
            [_pending("api.production_spot_check", "Pass --base-url for bounded API evidence")],
            {"status": "unavailable"},
            {"list_api": [], "date_jump_api": []},
        )
    timings: dict[str, list[float]] = {"list_api": [], "date_jump_api": []}
    checks: list[AcceptanceCheck] = []
    overview, elapsed = client.get_json("/api/v1/overview")
    timings["list_api"].append(elapsed)
    content = _json_object(overview.get("content"))
    overview_expected = {
        "conversation_count": totals["conversations"],
        "message_count": totals["messages"],
        "attachment_count": totals["attachments"],
        "activity_count": totals["activities"],
    }
    overview_actual = {key: content.get(key) for key in overview_expected}
    checks.append(
        _check(
            "api.production_spot_check_overview",
            overview_actual == overview_expected,
            expected=overview_expected,
            actual=overview_actual,
            detail="The aggregate overview matches exact query-only catalog totals",
        )
    )

    stream_specs = (
        (
            "conversations_messages_desc",
            "/api/v1/conversations",
            {"sort": "messages", "direction": "desc"},
            "conversation",
            "messages",
            "desc",
            totals["conversations"],
        ),
        (
            "conversations_last_activity_asc",
            "/api/v1/conversations",
            {"sort": "last_activity", "direction": "asc"},
            "conversation",
            "last_activity",
            "asc",
            totals["conversations"],
        ),
        (
            "media_date_desc",
            "/api/v1/media",
            {"view": "occurrences", "sort": "date", "direction": "desc"},
            "media",
            "date",
            "desc",
            totals["attachments"],
        ),
        (
            "media_size_asc",
            "/api/v1/media",
            {"view": "occurrences", "sort": "size", "direction": "asc"},
            "media",
            "size",
            "asc",
            totals["attachments"],
        ),
        (
            "activity_date_desc",
            "/api/v1/activity",
            {"sort": "date", "direction": "desc"},
            "activity",
            "date",
            "desc",
            totals["activities"],
        ),
        (
            "activity_category_asc",
            "/api/v1/activity",
            {"sort": "category", "direction": "asc"},
            "activity",
            "category",
            "asc",
            totals["activities"],
        ),
        (
            "events_date_desc",
            "/api/v1/events",
            {"sort": "date", "direction": "desc"},
            "event",
            "date",
            "desc",
            totals["events"],
        ),
        (
            "events_title_asc",
            "/api/v1/events",
            {"sort": "title", "direction": "asc"},
            "event",
            "title",
            "asc",
            totals["events"],
        ),
    )
    if len(stream_specs) != PRODUCTION_SPOT_CHECK_SORT_STREAMS:
        raise RuntimeError("Production spot-check stream specification changed unexpectedly")
    stream_reports: dict[str, object] = {}
    raw_results: list[dict[str, object]] = []
    for label, path, parameters, entity, sort_name, direction, expected_total in stream_specs:
        result, elapsed_values = _spot_cursor_stream_sample(
            connection,
            client,
            path=path,
            parameters=parameters,
            entity=entity,
            sort_name=sort_name,
            direction=direction,
            expected_total=int(expected_total),
        )
        timings["list_api"].extend(elapsed_values)
        raw_results.append(result)
        stream_reports[label] = {
            key: result.get(key)
            for key in (
                "passed",
                "coverage",
                "expected_total",
                "rows_checked",
                "pages_checked",
                "cursor_boundaries_checked",
                "outbound_cursor_observed",
                "full_stream",
                "sample_order_matches_targeted_values",
                "targeted_lookup_rows",
                "issue_count",
            )
        }

    def result_count(result: Mapping[str, object], key: str) -> int:
        value = result.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0

    aggregate = {
        "api_page_count": sum(result_count(result, "pages_checked") for result in raw_results),
        "sampled_row_count": sum(result_count(result, "rows_checked") for result in raw_results),
        "targeted_lookup_row_count": sum(
            result_count(result, "targeted_lookup_rows") for result in raw_results
        ),
        "cursor_boundary_count": sum(
            result_count(result, "cursor_boundaries_checked") for result in raw_results
        ),
        "failed_count": sum(result.get("passed") is not True for result in raw_results),
        "full_stream_count": sum(result.get("full_stream") is True for result in raw_results),
    }
    stream_passed = bool(
        len(raw_results) == PRODUCTION_SPOT_CHECK_SORT_STREAMS
        and aggregate["failed_count"] == 0
        and aggregate["api_page_count"]
        <= PRODUCTION_SPOT_CHECK_SORT_STREAMS * PRODUCTION_SPOT_CHECK_PAGES_PER_STREAM
        and aggregate["targeted_lookup_row_count"]
        <= PRODUCTION_SPOT_CHECK_SORT_STREAMS * PRODUCTION_SPOT_CHECK_MAX_TARGETED_LOOKUP_ROWS
    )
    checks.append(
        _check(
            "api.production_spot_check_sort_cursor_strata",
            stream_passed,
            expected={
                "stream_count": PRODUCTION_SPOT_CHECK_SORT_STREAMS,
                "maximum_pages_per_stream": PRODUCTION_SPOT_CHECK_PAGES_PER_STREAM,
                "maximum_rows_per_page": PRODUCTION_SPOT_CHECK_PAGE_SIZE,
                "targeted_primary_key_lookup_only": True,
                "all_sampled_orders_match": True,
            },
            actual={**aggregate, "stream_count": len(raw_results)},
            detail=(
                "API cursor prefixes are sampled; only returned identities receive bounded "
                "primary-key sort-value lookups. Unsampled order positions are not certified."
            ),
        )
    )
    simple_results: dict[str, object] = {}
    for label, path, simple_parameters, expected_total in (
        ("contacts", "/api/v1/people", {"view": "contacts"}, totals["contacts"]),
        ("timeline", "/api/v1/timeline", {}, totals["timeline"]),
    ):
        payload, elapsed = client.get_json(path, {**simple_parameters, "limit": 1})
        timings["list_api"].append(elapsed)
        page = _json_object(payload.get("page"))
        simple_results[label] = {
            "expected": expected_total,
            "actual": page.get("total"),
            "passed": page.get("total") == expected_total,
        }
    checks.append(
        _check(
            "api.production_spot_check_remaining_totals",
            all(_json_object(result).get("passed") is True for result in simple_results.values()),
            expected="exact contact and timeline totals",
            actual=simple_results,
            detail="Dashboards omitted from the sort sample retain exact first-page totals",
        )
    )
    return (
        checks,
        {
            "mode": "production-spot-check",
            "profile_version": PRODUCTION_SPOT_CHECK_PROFILE_VERSION,
            "sample_semantics": "bounded_api_prefix_with_targeted_sort_value_validation",
            "unsampled_order_positions_certified": False,
            "independent_full_stream_sort_reconciliation_performed": False,
            "stream_count": len(raw_results),
            "maximum_stream_count": PRODUCTION_SPOT_CHECK_SORT_STREAMS,
            "maximum_pages_per_stream": PRODUCTION_SPOT_CHECK_PAGES_PER_STREAM,
            "maximum_rows_per_page": PRODUCTION_SPOT_CHECK_PAGE_SIZE,
            "maximum_targeted_lookup_rows_per_stream": (
                PRODUCTION_SPOT_CHECK_MAX_TARGETED_LOOKUP_ROWS
            ),
            "streams": stream_reports,
            "aggregate": aggregate,
            "simple_totals": simple_results,
        },
        timings,
    )


def _spot_jump_strata(
    connection: Any,
    client: _SpotApiBudget | None,
    *,
    audited_largest: int | None,
) -> tuple[list[AcceptanceCheck], dict[str, object], dict[str, object] | None]:
    largest = (
        connection.execute(
            text(
                "SELECT id, message_count FROM conversations "
                "WHERE deleted_at IS NULL AND message_count = :audited_largest "
                "ORDER BY id DESC LIMIT 1"
            ),
            {"audited_largest": audited_largest},
        ).first()
        if audited_largest is not None
        else None
    )
    largest_summary = (
        {"id_sha256": _opaque(largest.id), "message_count": int(largest.message_count)}
        if largest is not None
        else None
    )
    largest_matches = bool(
        largest is not None
        and audited_largest is not None
        and int(largest.message_count) == audited_largest
    )
    checks = [
        _check(
            "catalog.production_spot_check_largest_conversation",
            largest_matches,
            expected={"audited_largest_message_count": audited_largest},
            actual={
                "current_largest_message_count": largest_summary["message_count"]
                if largest_summary
                else None
            },
            detail=(
                "The current baseline-bound largest-thread scalar resolves by an equality lookup; "
                "its identity is only hashed"
            ),
        )
    ]
    if client is None:
        checks.append(_pending("api.production_spot_check_jump_strata", "Pass --base-url"))
        return checks, {"status": "unavailable"}, largest_summary

    represented = 0
    resolved = 0
    for direction in ("ASC", "DESC"):
        target = connection.execute(
            text(
                f"SELECT id, conversation_id, timestamp_ms FROM messages "
                f"WHERE deleted_at IS NULL AND timestamp_ms IS NOT NULL "
                f"ORDER BY timestamp_ms {direction}, id {direction} LIMIT 1"
            )
        ).first()
        if target is None:
            continue
        represented += 1
        passed, _requests = _risk_message_jump(
            cast(LocalApiClient, client),
            conversation_id=target.conversation_id,
            message_id=target.id,
            timestamp_ms=target.timestamp_ms,
            require_date_jump=True,
            expected_date_message_id=target.id,
        )
        resolved += int(passed)
    for direction in ("ASC", "DESC"):
        target = connection.execute(
            text(
                f"SELECT id FROM activity_records WHERE deleted_at IS NULL "
                f"AND timestamp_ms IS NOT NULL ORDER BY timestamp_ms {direction}, id {direction} "
                "LIMIT 1"
            )
        ).first()
        if target is None:
            continue
        represented += 1
        passed, _requests = _risk_activity_jump(cast(LocalApiClient, client), activity_id=target.id)
        resolved += int(passed)
    media_dimensions = (
        *(("type", value, _MEDIA_CLASS_SQL) for value in ("photo", "video", "audio", "document")),
        *(
            ("availability", value, "a.availability")
            for value in ("resolved", "missing", "unlinked", "external")
        ),
    )
    for dimension, value, expression in media_dimensions:
        target = connection.execute(
            text(
                f"SELECT a.id, a.conversation_id, a.message_id, "
                f"a.activity_record_id AS activity_id, a.availability "
                f"FROM attachment_occurrences AS a WHERE a.deleted_at IS NULL "
                f"AND ({expression}) = :value ORDER BY a.id LIMIT 1"
            ),
            {"value": value},
        ).first()
        if target is None:
            continue
        represented += 1
        passed, _requests = _risk_media_jump(
            cast(LocalApiClient, client),
            target,
            expected_type=value if dimension == "type" else None,
        )
        resolved += int(passed)
    passed = represented > 0 and represented <= 12 and represented == resolved
    report: dict[str, object] = {
        "maximum_samples": 12,
        "represented_samples": represented,
        "resolved_samples": resolved,
        "sample_private_values_emitted": False,
    }
    checks.append(
        _check(
            "api.production_spot_check_jump_strata",
            passed,
            expected={"maximum_samples": 12, "all_represented_resolve": True},
            actual=report,
            detail="Date-edge messages, activities, media classes, and availability states resolve",
        )
    )
    return checks, report, largest_summary


def _spot_keyset_rows(
    connection: Any,
    *,
    table: str,
    key_column: str,
    columns: str,
    where: str,
    parameters: Mapping[str, object],
    seed: int,
    label: str,
    count: int = PRODUCTION_SPOT_CHECK_SAMPLE_ROWS,
) -> list[Any]:
    allowed = {
        "nlp_annotations",
        "semantic_window_embeddings",
        "media_derivatives",
        "audio_transcriptions",
        "attachment_text_extractions",
    }
    if table not in allowed or count < 1 or count > PRODUCTION_SPOT_CHECK_SAMPLE_ROWS:
        raise ValueError("Invalid production spot-check keyset sample")
    pivot = hashlib.sha256(f"{seed}:{label}".encode()).hexdigest()
    base = dict(parameters)
    base.update({"spot_pivot": pivot, "spot_limit": count})
    rows = _rows(
        connection,
        f"SELECT {columns} FROM {table} WHERE ({where}) AND {key_column} >= :spot_pivot "
        f"ORDER BY {key_column} LIMIT :spot_limit",
        base,
    )
    if len(rows) < count:
        wrapped = dict(base)
        wrapped["spot_limit"] = count - len(rows)
        rows.extend(
            _rows(
                connection,
                f"SELECT {columns} FROM {table} WHERE ({where}) AND {key_column} < :spot_pivot "
                f"ORDER BY {key_column} LIMIT :spot_limit",
                wrapped,
            )
        )
    return rows


def _spot_nlp_sample(
    connection: Any, tables: set[str], *, seed: int
) -> tuple[list[AcceptanceCheck], dict[str, object]]:
    required = {"nlp_annotations", "nlp_hourly_rollups"}
    if not required <= tables:
        return (
            [
                _check(
                    "nlp.production_spot_check_sample",
                    False,
                    expected=sorted(required),
                    actual={"present_count": len(required & tables)},
                    detail="NLP sample tables are required",
                )
            ],
            {},
        )
    status_rows = _rows(
        connection,
        "SELECT task, status, count(*) AS row_count FROM nlp_annotations "
        "WHERE active = 1 GROUP BY task, status ORDER BY task, status",
    )
    status_counts = {f"{row.task}:{row.status}": int(row.row_count) for row in status_rows}
    samples = _spot_keyset_rows(
        connection,
        table="nlp_annotations",
        key_column="id",
        columns=(
            "id, target_type, target_id, task, pipeline_version, model_name, model_revision, "
            "input_sha256, status, active, predicted_label, confidence, error_text"
        ),
        where="active = 1",
        parameters={},
        seed=seed,
        label="nlp",
    )
    valid = bool(samples) and len(samples) <= PRODUCTION_SPOT_CHECK_SAMPLE_ROWS
    for row in samples:
        task = str(row.task)
        spec = MODEL_SPECS[task] if task in TASKS else None
        target_table = {
            "message": "messages",
            "activity": "activity_records",
            "audio_segment": "audio_transcript_segments",
            "attachment_text": "attachment_text_extractions",
        }.get(str(row.target_type))
        target_exists = bool(
            target_table
            and _scalar(
                connection,
                f"SELECT EXISTS(SELECT 1 FROM {target_table} WHERE id = :target_id)",
                {"target_id": row.target_id},
            )
        )
        confidence = row.confidence
        valid = valid and bool(
            spec is not None
            and row.pipeline_version == PIPELINE_VERSION
            and row.model_name == spec.model_name
            and row.model_revision == spec.model_revision
            and row.status == "completed"
            and bool(row.active)
            and isinstance(row.input_sha256, str)
            and _LOWER_SHA256_PATTERN.fullmatch(row.input_sha256)
            and isinstance(row.predicted_label, str)
            and bool(row.predicted_label)
            and isinstance(confidence, (int, float))
            and not isinstance(confidence, bool)
            and math.isfinite(float(confidence))
            and 0 <= float(confidence) <= 1
            and row.error_text is None
            and target_exists
        )
    global_status_valid = (
        bool(status_counts)
        and all(key.endswith(":completed") for key in status_counts)
        and {key.split(":", 1)[0] for key in status_counts} == set(TASKS)
    )
    report = {
        "active_status_counts": status_counts,
        "maximum_sample_rows": PRODUCTION_SPOT_CHECK_SAMPLE_ROWS,
        "sample_rows": len(samples),
        "sample_valid": valid,
        "full_annotation_reconstruction_performed": False,
    }
    return (
        [
            _check(
                "nlp.production_spot_check_sample",
                valid and global_status_valid,
                expected=(
                    "final active status totals plus a valid deterministic target-bound sample"
                ),
                actual=report,
                detail="No source text, target identity, labels, or provenance payload is retained",
            )
        ],
        report,
    )


def _spot_vector_is_valid(vector: object, dimensions: int) -> bool:
    if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes, bytearray)):
        return False
    try:
        values = [float(value) for value in vector]
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(
        len(values) == dimensions
        and all(math.isfinite(value) for value in values)
        and math.isclose(
            math.sqrt(math.fsum(value * value for value in values)), 1.0, abs_tol=1e-3, rel_tol=1e-3
        )
    )


def _spot_semantic_certification_metadata(
    engine: Engine,
    config: Any,
) -> tuple[bool, dict[str, object]]:
    """Revalidate current evaluator metadata and small feedback-revision tables."""

    report: dict[str, object] = {
        "metadata_observed": False,
        "evaluator_version_current": False,
        "candidate_binding_current": False,
        "automatic_evaluation_valid": False,
        "automatic_thresholds_current": False,
        "automatic_sampled_query_count": None,
        "held_out_evaluation_valid": False,
        "held_out_thresholds_current": False,
        "held_out_query_count": None,
        "held_out_feedback_revision_current": False,
        "development_evaluation_valid": False,
        "development_feedback_revision_current": False,
        "development_counts_valid": False,
        "development_metrics_valid": False,
        "development_thresholds_current": False,
        "query_details_emitted": False,
    }
    try:
        audit = _database_json_object(config.audit_json) or {}
        accepted = _json_object(audit.get("accepted_semantic_ir_evaluation"))
        report["metadata_observed"] = bool(accepted)
        automatic = _json_object(accepted.get("automatic_known_item"))
        automatic_sampling = _json_object(automatic.get("sampling"))
        automatic_metrics = _json_object(automatic.get("metrics"))
        automatic_thresholds_current = automatic.get("thresholds") == {
            **KNOWN_ITEM_THRESHOLDS,
            "k": SEMANTIC_IR_K,
        }
        held_out = _json_object(accepted.get("held_out_feedback"))
        held_metrics = _json_object(held_out.get("metrics"))
        held_exclusions = _json_object(held_out.get("exclusions"))
        held_out_thresholds_current = bool(
            held_out.get("thresholds") == {**HELD_OUT_THRESHOLDS, "k": SEMANTIC_IR_K}
            and held_out.get("minimum_query_count_for_activation") == HELD_OUT_MIN_QUERY_COUNT
        )
        current_feedback_revision = held_out_feedback_revision(engine)
        feedback_revision_current = bool(
            isinstance(held_out.get("feedback_revision"), dict)
            and held_out.get("feedback_revision") == current_feedback_revision
            and held_out.get("feedback_revision_stable_during_evaluation") is True
        )

        def metric_at_least(metrics: Mapping[str, object], name: str, threshold: float) -> bool:
            value = metrics.get(name)
            return bool(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and float(value) >= threshold
            )

        held_query_count = held_metrics.get("query_count")
        held_valid = bool(
            held_out.get("benchmark_kind") == "held_out_user_feedback"
            and held_out.get("status") == "passed"
            and held_out.get("passed") is True
            and held_out.get("used_for_activation") is True
            and held_out.get("query_details_persisted") is False
            and isinstance(held_query_count, int)
            and not isinstance(held_query_count, bool)
            and held_query_count >= HELD_OUT_MIN_QUERY_COUNT
            and all(
                metric_at_least(held_metrics, name, threshold)
                for name, threshold in HELD_OUT_THRESHOLDS.items()
            )
            and held_exclusions.get("without_relevant_window_judgment") == 0
            and held_exclusions.get("relevant_target_not_in_candidate") == 0
            and held_out_thresholds_current
            and feedback_revision_current
        )

        development = _json_object(accepted.get("direct_conflict_development_calibration"))
        current_development_revision = development_feedback_revision(engine)
        development_revision_current = bool(
            isinstance(development.get("feedback_revision"), dict)
            and development.get("feedback_revision") == current_development_revision
            and development.get("feedback_revision_stable_during_evaluation") is True
        )
        grouped = _json_object(development.get("grouped_cross_validation"))
        development_thresholds_current = bool(
            development.get("minimums")
            == {
                "relevant_count": DIRECT_CONFLICT_DEVELOPMENT_MIN_RELEVANT_COUNT,
                "not_relevant_count": DIRECT_CONFLICT_DEVELOPMENT_MIN_NOT_RELEVANT_COUNT,
                "fold_count": DIRECT_CONFLICT_DEVELOPMENT_FOLD_COUNT,
            }
            and development.get("thresholds")
            == {
                **DIRECT_CONFLICT_DEVELOPMENT_THRESHOLDS,
                "k": DIRECT_CONFLICT_RANKING_K,
            }
        )
        minimum_metrics = _json_object(grouped.get("minimum_metrics"))
        mean_metrics = _json_object(grouped.get("mean_metrics"))
        literal_metrics = _json_object(
            _json_object(development.get("literal_query_regression")).get("metrics")
        )
        development_metrics_valid = all(
            metric_at_least(minimum_metrics, name, threshold)
            and metric_at_least(literal_metrics, name, threshold)
            and isinstance(mean_metrics.get(name), (int, float))
            and not isinstance(mean_metrics.get(name), bool)
            and math.isfinite(float(mean_metrics[name]))
            and float(mean_metrics[name]) >= float(minimum_metrics[name])
            for name, threshold in DIRECT_CONFLICT_DEVELOPMENT_THRESHOLDS.items()
        )

        def non_negative_int(value: object) -> int | None:
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
            return None

        judgment_count = non_negative_int(development.get("judgment_count"))
        reviewed_count = non_negative_int(development.get("reviewed_judgment_count"))
        binary_count = non_negative_int(development.get("binary_judgment_count"))
        relevant_count = non_negative_int(development.get("relevant_count"))
        not_relevant_count = non_negative_int(development.get("not_relevant_count"))
        uncertain_count = non_negative_int(development.get("uncertain_count"))
        conversation_count = non_negative_int(
            development.get("conversation_group_count", development.get("conversation_count"))
        )
        fold_count = non_negative_int(development.get("fold_count"))
        revision_count = non_negative_int(current_development_revision.get("row_count"))
        development_counts_valid = bool(
            judgment_count is not None
            and reviewed_count == judgment_count
            and binary_count is not None
            and relevant_count is not None
            and not_relevant_count is not None
            and uncertain_count is not None
            and binary_count == relevant_count + not_relevant_count
            and judgment_count == binary_count + uncertain_count
            and relevant_count >= DIRECT_CONFLICT_DEVELOPMENT_MIN_RELEVANT_COUNT
            and not_relevant_count >= DIRECT_CONFLICT_DEVELOPMENT_MIN_NOT_RELEVANT_COUNT
            and revision_count == judgment_count
            and conversation_count is not None
            and conversation_count >= DIRECT_CONFLICT_DEVELOPMENT_FOLD_COUNT
            and fold_count == DIRECT_CONFLICT_DEVELOPMENT_FOLD_COUNT
            and non_negative_int(grouped.get("fold_count"))
            == DIRECT_CONFLICT_DEVELOPMENT_FOLD_COUNT
        )
        model_metadata = _json_object(development.get("model_metadata"))
        development_valid = bool(
            accepted.get("production_acceptance_passed") is True
            and development.get("required_for_production_acceptance") is True
            and development.get("benchmark_kind") == DIRECT_CONFLICT_DEVELOPMENT_BENCHMARK_KIND
            and development.get("evaluation_intent") == DIRECT_CONFLICT_DEVELOPMENT_INTENT
            and development.get("evidence_kind") == DIRECT_CONFLICT_DEVELOPMENT_EVIDENCE_KIND
            and development.get("held_out") is False
            and development.get("query_details_persisted") is False
            and development.get("status") == "passed"
            and development.get("passed") is True
            and development.get("method") == DIRECT_CONFLICT_DEVELOPMENT_METHOD
            and development.get("ranking_pipeline") == DIRECT_CONFLICT_RANKING_PIPELINE_VERSION
            and model_metadata.get("classifier") == DIRECT_CONFLICT_CLASSIFIER
            and isinstance(model_metadata.get("library_version"), str)
            and bool(model_metadata.get("library_version"))
            and development_revision_current
            and development_counts_valid
            and development_metrics_valid
            and development_thresholds_current
        )
        candidate = _json_object(accepted.get("candidate"))
        candidate_valid = bool(
            candidate.get("config_hash") == config.id
            and candidate.get("vector_table") == config.vector_table
            and candidate.get("window_count") == int(config.window_count)
            and candidate.get("distance_metric") == "cosine"
        )
        sampled_query_count = automatic_sampling.get("sampled_query_count")
        automatic_valid = bool(
            automatic.get("benchmark_kind") == "automatic_known_item"
            and automatic.get("status") == "passed"
            and automatic.get("passed") is True
            and automatic.get("used_for_activation") is True
            and isinstance(sampled_query_count, int)
            and not isinstance(sampled_query_count, bool)
            and sampled_query_count > 0
            and automatic_thresholds_current
            and all(
                metric_at_least(automatic_metrics, name, threshold)
                for name, threshold in KNOWN_ITEM_THRESHOLDS.items()
            )
        )
        evaluator_current = accepted.get("evaluator_version") == SEMANTIC_IR_EVALUATOR_VERSION
        passed = bool(
            accepted.get("passed") is True
            and evaluator_current
            and automatic_valid
            and held_valid
            and development_valid
            and candidate_valid
        )
        report.update(
            {
                "evaluator_version_current": evaluator_current,
                "candidate_binding_current": candidate_valid,
                "automatic_evaluation_valid": automatic_valid,
                "automatic_thresholds_current": automatic_thresholds_current,
                "automatic_sampled_query_count": sampled_query_count,
                "held_out_evaluation_valid": held_valid,
                "held_out_thresholds_current": held_out_thresholds_current,
                "held_out_query_count": held_query_count,
                "held_out_feedback_revision_current": feedback_revision_current,
                "development_evaluation_valid": development_valid,
                "development_feedback_revision_current": development_revision_current,
                "development_counts_valid": development_counts_valid,
                "development_metrics_valid": development_metrics_valid,
                "development_thresholds_current": development_thresholds_current,
            }
        )
        return passed, report
    except Exception:
        # SQL/JSON errors may include private details. The aggregate metadata gate simply fails.
        return False, report


def _spot_semantic_lance_sample(
    layout: ArchiveLayout,
    engine: Engine,
    connection: Any,
    tables: set[str],
    *,
    seed: int,
) -> tuple[list[AcceptanceCheck], dict[str, object]]:
    required = {"semantic_index_configs", "semantic_window_embeddings", "search_windows"}
    report: dict[str, object] = {
        "maximum_sqlite_sample_rows": PRODUCTION_SPOT_CHECK_SAMPLE_ROWS,
        "maximum_lance_sample_rows": PRODUCTION_SPOT_CHECK_SAMPLE_ROWS,
        "full_lance_scan_performed": False,
    }
    if not required <= tables:
        return (
            [
                _check(
                    "semantic.production_spot_check_sample",
                    False,
                    expected=sorted(required),
                    actual={"present_count": len(required & tables)},
                    detail="Semantic sample tables are required",
                )
            ],
            report,
        )
    configs = _rows(
        connection,
        "SELECT id, status, model_name, model_revision, dimensions, distance_metric, "
        "vector_table, window_version, window_count, audit_json FROM semantic_index_configs "
        "WHERE active = 1 ORDER BY id",
    )
    if len(configs) != 1:
        report["active_config_count"] = len(configs)
        return (
            [
                _check(
                    "semantic.production_spot_check_sample",
                    False,
                    expected=1,
                    actual=report,
                    detail="Exactly one active semantic configuration is required",
                )
            ],
            report,
        )
    config = configs[0]
    config_id = str(config.id)
    metadata_valid, metadata_report = _spot_semantic_certification_metadata(engine, config)
    membership_count = int(
        _scalar(
            connection,
            "SELECT count(*) FROM semantic_window_embeddings WHERE config_id = :config_id",
            {"config_id": config_id},
        )
    )
    window_count = int(_scalar(connection, "SELECT count(*) FROM search_windows"))
    samples = _spot_keyset_rows(
        connection,
        table="semantic_window_embeddings",
        key_column="window_id",
        columns="window_id AS id",
        where="config_id = :config_id",
        parameters={"config_id": config_id},
        seed=seed,
        label="semantic-membership",
    )
    sample_ids = [str(row.id) for row in samples]
    sqlite_sample_valid = False
    if sample_ids:
        placeholders = ", ".join(f":sample_{index}" for index in range(len(sample_ids)))
        parameters = {f"sample_{index}": identifier for index, identifier in enumerate(sample_ids)}
        present = int(
            _scalar(
                connection,
                f"SELECT count(*) FROM search_windows WHERE id IN ({placeholders}) "
                "AND window_version = :window_version AND coalesce(text_display, '') != ''",
                {**parameters, "window_version": config.window_version},
            )
        )
        sqlite_sample_valid = present == len(sample_ids) == len(set(sample_ids))
    lance_count: int | None = None
    lance_sample_rows = 0
    lance_sample_valid = False
    try:
        import lancedb

        vector_database = lancedb.connect(str(layout.vectors))
        vector_table = vector_database.open_table(str(config.vector_table))
        lance_count = int(vector_table.count_rows())
        if sample_ids and all(re.fullmatch(r"[0-9A-Za-z_-]{1,128}", value) for value in sample_ids):
            predicate = "id IN (" + ", ".join(f"'{value}'" for value in sample_ids) + ")"
            lance_rows = (
                vector_table.search()
                .where(predicate)
                .limit(PRODUCTION_SPOT_CHECK_SAMPLE_ROWS)
                .select(["id", "vector"])
                .to_list()
            )
            lance_sample_rows = len(lance_rows)
            lance_ids = [str(item.get("id")) for item in lance_rows]
            lance_sample_valid = bool(
                len(lance_rows) == len(sample_ids)
                and set(lance_ids) == set(sample_ids)
                and len(lance_ids) == len(set(lance_ids))
                and all(
                    _spot_vector_is_valid(item.get("vector"), int(config.dimensions))
                    for item in lance_rows
                )
            )
    except Exception:
        lance_sample_valid = False
    pinned = any(
        spec.model_name == config.model_name
        and spec.model_revision == config.model_revision
        and spec.dimensions == config.dimensions
        for spec in SEMANTIC_MODEL_SPECS.values()
    )
    exact_counts_valid = bool(
        config.status == "ready"
        and config.window_version == SEARCH_WINDOW_VERSION
        and config.distance_metric == "cosine"
        and membership_count == window_count == int(config.window_count) == lance_count
    )
    passed = bool(
        exact_counts_valid
        and pinned
        and metadata_valid
        and sqlite_sample_valid
        and lance_sample_valid
    )
    report.update(
        {
            "active_config_count": 1,
            "configured_windows": int(config.window_count),
            "catalog_windows": window_count,
            "membership_windows": membership_count,
            "lance_windows": lance_count,
            "sqlite_sample_rows": len(sample_ids),
            "lance_sample_rows": lance_sample_rows,
            "sqlite_sample_valid": sqlite_sample_valid,
            "lance_sample_valid": lance_sample_valid,
            "pinned_model_identity": pinned,
            "certification_metadata_current": metadata_valid,
            "certification_metadata": metadata_report,
        }
    )
    return (
        [
            _check(
                "semantic.production_spot_check_sample",
                passed,
                expected=(
                    "current evaluator, candidate, threshold and feedback-revision metadata; "
                    "exact active counts; and deterministic SQLite/Lance samples"
                ),
                actual=report,
                detail="No vector-table name, window identity, query, or text is emitted",
            )
        ],
        report,
    )


def _spot_derived_samples(
    layout: ArchiveLayout,
    connection: Any,
    tables: set[str],
    *,
    seed: int,
) -> tuple[list[AcceptanceCheck], dict[str, object]]:
    families = (
        (
            "media_derivatives",
            "pipeline_version = :version",
            {"version": MEDIA_DERIVATIVE_PIPELINE_VERSION},
            "id, status, relative_path, size_bytes, output_sha256",
        ),
        (
            "audio_transcriptions",
            "pipeline_version = :version AND model_name = :model AND model_revision = :revision",
            {
                "version": TRANSCRIPTION_PIPELINE_VERSION,
                "model": TRANSCRIPTION_MODEL_NAME,
                "revision": TRANSCRIPTION_MODEL_REVISION,
            },
            "id, status, segment_count, index_eligible",
        ),
        (
            "attachment_text_extractions",
            (
                "pipeline_version = :version AND extractor_name = :extractor "
                "AND extractor_revision = :revision"
            ),
            {
                "version": ATTACHMENT_TEXT_PIPELINE_VERSION,
                "extractor": ATTACHMENT_EXTRACTOR_NAME,
                "revision": ATTACHMENT_EXTRACTOR_REVISION,
            },
            "id, status, index_eligible, sandboxed, sandbox_status",
        ),
    )
    results: dict[str, object] = {}
    all_valid = len(families) == PRODUCTION_SPOT_CHECK_DERIVED_FAMILIES
    for table, where, parameters, columns in families:
        if table not in tables:
            results[table] = {"schema_present": False, "sample_valid": False}
            all_valid = False
            continue
        status_counts = {
            str(row.status): int(row.row_count)
            for row in _rows(
                connection,
                f"SELECT status, count(*) AS row_count FROM {table} WHERE {where} "
                "GROUP BY status ORDER BY status",
                parameters,
            )
        }
        rows = _spot_keyset_rows(
            connection,
            table=table,
            key_column="id",
            columns=columns,
            where=where,
            parameters=parameters,
            seed=seed,
            label=table,
        )
        valid = len(rows) <= PRODUCTION_SPOT_CHECK_SAMPLE_ROWS
        if table == "media_derivatives":
            valid = valid and all(row.status in {"ready", "failed"} for row in rows)
            for row in rows:
                if row.status != "ready":
                    continue
                relative = Path(str(row.relative_path))
                candidate = layout.root / relative
                try:
                    candidate_stat = candidate.lstat()
                except OSError:
                    valid = False
                    continue
                valid = valid and bool(
                    not relative.is_absolute()
                    and ".." not in relative.parts
                    and stat.S_ISREG(candidate_stat.st_mode)
                    and not stat.S_ISLNK(candidate_stat.st_mode)
                    and (row.size_bytes is None or int(row.size_bytes) == candidate_stat.st_size)
                    and isinstance(row.output_sha256, str)
                    and _LOWER_SHA256_PATTERN.fullmatch(row.output_sha256)
                )
        elif table == "audio_transcriptions":
            for row in rows:
                segment_count = int(
                    _scalar(
                        connection,
                        "SELECT count(*) FROM audio_transcript_segments "
                        "WHERE transcription_id = :id",
                        {"id": row.id},
                    )
                )
                valid = valid and bool(
                    row.status == "completed"
                    and int(row.segment_count) == segment_count
                    and bool(row.index_eligible) is (segment_count > 0)
                )
        else:
            allowed = {"completed", "excluded", "unsupported", "limit_exceeded"}
            valid = valid and all(
                row.status in allowed
                and bool(row.sandboxed)
                and isinstance(row.sandbox_status, str)
                and bool(row.sandbox_status)
                for row in rows
            )
        final_statuses = "processing" not in status_counts and "failed" not in status_counts
        if table == "media_derivatives":
            final_statuses = "processing" not in status_counts
        family_valid = bool(rows) and valid and final_statuses
        all_valid = all_valid and family_valid
        results[table] = {
            "schema_present": True,
            "status_counts": status_counts,
            "maximum_sample_rows": PRODUCTION_SPOT_CHECK_SAMPLE_ROWS,
            "sample_rows": len(rows),
            "sample_valid": family_valid,
        }
    report = {
        "family_count": len(results),
        "maximum_family_count": PRODUCTION_SPOT_CHECK_DERIVED_FAMILIES,
        "maximum_total_sample_rows": PRODUCTION_SPOT_CHECK_DERIVED_FAMILIES
        * PRODUCTION_SPOT_CHECK_SAMPLE_ROWS,
        "families": results,
        "full_candidate_reconstruction_performed": False,
    }
    return (
        [
            _check(
                "derived.production_spot_check_samples",
                all_valid,
                expected="three final-status artifact families with deterministic bounded samples",
                actual=report,
                detail="Only aggregate statuses and sample counts are retained",
            )
        ],
        report,
    )


def _spot_nlp_aggregate_row_valid(row: Mapping[str, object]) -> bool:
    if set(row) != {
        "period",
        "direction",
        "sample_count",
        "mean_confidence",
        "score",
        "score_definition",
        "dominant_label",
        "predicted_label_counts",
        "mean_probabilities",
    }:
        return False
    sample_count = row.get("sample_count")
    mean_confidence = row.get("mean_confidence")
    score = row.get("score")
    dominant_label = row.get("dominant_label")
    predicted_counts = row.get("predicted_label_counts")
    mean_probabilities = row.get("mean_probabilities")
    if not (
        isinstance(row.get("period"), str)
        and bool(row["period"])
        and len(str(row["period"])) <= 64
        and row.get("direction") in PRODUCTION_SPOT_CHECK_NLP_DIRECTIONS
        and isinstance(sample_count, int)
        and not isinstance(sample_count, bool)
        and sample_count >= 0
        and isinstance(mean_confidence, (int, float))
        and not isinstance(mean_confidence, bool)
        and math.isfinite(float(mean_confidence))
        and 0.0 <= float(mean_confidence) <= 1.0
        and (
            score is None
            or (
                isinstance(score, (int, float))
                and not isinstance(score, bool)
                and math.isfinite(float(score))
            )
        )
        and isinstance(row.get("score_definition"), str)
        and bool(row["score_definition"])
        and (dominant_label is None or isinstance(dominant_label, str))
        and isinstance(predicted_counts, Mapping)
        and isinstance(mean_probabilities, Mapping)
    ):
        return False
    if not all(
        isinstance(label, str)
        and bool(label)
        and isinstance(count, int)
        and not isinstance(count, bool)
        and count >= 0
        for label, count in predicted_counts.items()
    ):
        return False
    if sum(int(count) for count in predicted_counts.values()) != sample_count:
        return False
    return all(
        isinstance(label, str)
        and bool(label)
        and isinstance(probability, (int, float))
        and not isinstance(probability, bool)
        and math.isfinite(float(probability))
        and 0.0 <= float(probability) <= 1.0
        for label, probability in mean_probabilities.items()
    )


def _spot_analytics(
    client: _SpotApiBudget | None,
) -> tuple[list[AcceptanceCheck], dict[str, object]]:
    if client is None:
        return ([_pending("analytics.production_spot_check", "Pass --base-url")], {})
    valid = True
    coverage_count = 0
    nlp_rows = 0
    try:
        coverage, _elapsed = client.get_json("/api/v1/analytics/coverage")
        coverage_items = _json_object_list(coverage.get("items"))
        coverage_count = len(coverage_items)
        coverage_keys = [item.get("key") for item in coverage_items]
        valid = valid and bool(
            coverage_count <= 64
            and len(coverage_keys) == len(set(coverage_keys))
            and all(isinstance(key, str) and key for key in coverage_keys)
        )
        nlp, _elapsed = client.get_json(
            "/api/v1/analytics/nlp-trends",
            {"bucket": "year", "timezone": "UTC", "direction": "all", "limit": 12},
        )
        page = _json_object(nlp.get("page"))
        series = _json_object(nlp.get("series"))
        task_keys_valid = set(series) == set(PRODUCTION_SPOT_CHECK_NLP_TASKS)
        task_rows: dict[str, list[dict[str, Any]]] = {
            task: _json_object_list(series.get(task)) for task in PRODUCTION_SPOT_CHECK_NLP_TASKS
        }
        nlp_rows = sum(len(rows) for rows in task_rows.values())
        per_task_valid = all(
            isinstance(series.get(task), list)
            and len(rows) <= PRODUCTION_SPOT_CHECK_NLP_MAX_ROWS_PER_TASK
            and all(_spot_nlp_aggregate_row_valid(row) for row in rows)
            and len({(str(row.get("period")), str(row.get("direction"))) for row in rows})
            == len(rows)
            for task, rows in task_rows.items()
        )
        returned = page.get("returned")
        valid = valid and bool(
            task_keys_valid
            and per_task_valid
            and page.get("limit") == PRODUCTION_SPOT_CHECK_NLP_PERIODS
            and isinstance(returned, int)
            and not isinstance(returned, bool)
            and 0 <= returned <= PRODUCTION_SPOT_CHECK_NLP_PERIODS
            and nlp_rows <= PRODUCTION_SPOT_CHECK_NLP_MAX_SERIES_ROWS
        )
    except RuntimeError:
        valid = False
    report: dict[str, object] = {
        "request_count": PRODUCTION_SPOT_CHECK_ANALYTICS_CALLS,
        "maximum_request_count": PRODUCTION_SPOT_CHECK_ANALYTICS_CALLS,
        "coverage_item_count": coverage_count,
        "nlp_series_row_count": nlp_rows,
        "maximum_nlp_series_rows": PRODUCTION_SPOT_CHECK_NLP_MAX_SERIES_ROWS,
        "all_pages_collected": False,
    }
    return (
        [
            _check(
                "analytics.production_spot_check",
                valid,
                expected="two bounded, schema-valid analytics responses",
                actual=report,
                detail=(
                    "No all-pages analytics collection or direct-SQL dashboard reconstruction "
                    "is run"
                ),
            )
        ],
        report,
    )


def _build_production_spot_check_report(
    layout: ArchiveLayout,
    *,
    client: LocalApiClient | None,
    query: str,
    browser_report: Path | None,
    policy: ReconciliationPolicy,
) -> dict[str, object]:
    """Run the intentionally reduced, crash-safe production acceptance profile."""

    stage_count = 9
    _spot_progress(1, stage_count, "capture-database-identity")
    database_before = _spot_database_stat(layout.database)
    sidecars_before = _spot_sqlite_sidecars(layout.database)
    baseline_path = audited_baseline_manifest_path(layout.root)
    baseline_before: _SpotFileBinding | None = None
    audited_baseline: object | None = None
    baseline_schema_identity_valid = False
    try:
        baseline_opening = _spot_file_binding(
            baseline_path,
            maximum_bytes=MAX_AUDITED_BASELINE_MANIFEST_BYTES,
        )
        audited_baseline = load_audited_baseline(layout.root)
        baseline_before = _spot_file_binding(
            baseline_path,
            maximum_bytes=MAX_AUDITED_BASELINE_MANIFEST_BYTES,
        )
        baseline_schema_identity_valid = bool(
            audited_baseline is not None and baseline_opening == baseline_before
        )
    except (AuditedBaselineManifestError, OSError, RuntimeError):
        baseline_before = None
        audited_baseline = None
    engine = _readonly_catalog_engine(layout.database)
    checks: list[AcceptanceCheck] = []
    budget = _SpotApiBudget(client, PRODUCTION_SPOT_CHECK_MAX_API_CALLS) if client else None
    try:
        with engine.connect() as connection:
            query_only = int(_scalar(connection, "PRAGMA query_only"))
            revision_before = int(_scalar(connection, "PRAGMA data_version"))
            tables = _table_names(connection)
            _spot_progress(2, stage_count, "exact-catalog-totals")
            totals = _catalog_totals(connection)
            totals["unlinked_media"] = int(
                _scalar(
                    connection,
                    "SELECT count(*) FROM attachment_occurrences "
                    "WHERE deleted_at IS NULL AND availability = 'unlinked'",
                )
            )
            identity_checks, identity_report, server_identity = _acceptance_server_identity(
                cast(LocalApiClient | None, budget)
            )
            checks.extend(identity_checks)
            audit_client = budget if server_identity else None
            permission_findings = layout.private_permission_findings()
            checks.append(
                _check(
                    "catalog.private_runtime_permissions",
                    not permission_findings,
                    expected="owner-only symlink-free runtime boundaries",
                    actual={"finding_count": len(permission_findings)},
                    detail="Finding paths are omitted",
                )
            )

            _spot_progress(3, stage_count, "validate-prior-deep-evidence")
            deep_checks, deep_report, marker_counts, deep_generated_ns = _spot_prior_deep_report(
                layout
            )
            checks.extend(deep_checks)
            prior_deep_revalidated = bool(len(deep_checks) == 1 and deep_checks[0].status == "pass")
            marker_expected = {
                "conversation_count": totals["conversations"],
                "message_count": totals["messages"],
                "attachment_occurrence_count": totals["attachments"],
                "activity_record_count": totals["activities"],
                "unlinked_media_count": totals["unlinked_media"],
            }
            marker_actual = {key: marker_counts.get(key) for key in marker_expected}
            checks.append(
                _check(
                    "catalog.production_spot_check_library_totals",
                    marker_actual == marker_expected,
                    expected=marker_expected,
                    actual=marker_actual,
                    detail=(
                        "Current exact catalog totals remain bound to the current "
                        "filesystem-library marker"
                    ),
                )
            )

            _spot_progress(4, stage_count, "representative-api-strata")
            api_checks, reconciliation_report, timings = _spot_api_reconciliation(
                connection, audit_client, totals
            )
            checks.extend(api_checks)
            audited_largest = getattr(audited_baseline, "largest_conversation_messages", None)
            jump_checks, jump_report, largest_summary = _spot_jump_strata(
                connection,
                audit_client,
                audited_largest=audited_largest if isinstance(audited_largest, int) else None,
            )
            checks.extend(jump_checks)

            _spot_progress(5, stage_count, "search-user-stories")
            search_calls_before = budget.calls if budget else 0
            search_checks, search_report = _search_user_story_acceptance(
                connection, cast(LocalApiClient | None, audit_client), query=query
            )
            checks.extend(search_checks)
            search_calls = (budget.calls - search_calls_before) if budget else 0
            checks.append(
                _check(
                    "api.production_spot_check_search_budget",
                    search_calls <= PRODUCTION_SPOT_CHECK_SEARCH_CALLS,
                    expected={"maximum_calls": PRODUCTION_SPOT_CHECK_SEARCH_CALLS},
                    actual={"calls": search_calls},
                    detail="Semantic, hybrid, and conflict stories are strictly request-capped",
                )
            )

            _spot_progress(6, stage_count, "sample-derived-products")
            nlp_checks, nlp_report = _spot_nlp_sample(connection, tables, seed=policy.seed)
            semantic_checks, semantic_report = _spot_semantic_lance_sample(
                layout, engine, connection, tables, seed=policy.seed
            )
            derived_checks, derived_report = _spot_derived_samples(
                layout, connection, tables, seed=policy.seed
            )
            checks.extend([*nlp_checks, *semantic_checks, *derived_checks])

            _spot_progress(7, stage_count, "bounded-analytics")
            analytics_checks, analytics_report = _spot_analytics(audit_client)
            checks.extend(analytics_checks)

            _spot_progress(8, stage_count, "browser-evidence")
            try:
                browser_checks, browser = _browser_evidence(
                    browser_report,
                    expected_archive_identity=_opaque(layout.root.name),
                    expected_totals=totals,
                    expected_largest=largest_summary,
                    expected_server_identity=server_identity,
                )
            except RuntimeError:
                browser_checks = [
                    _check(
                        "browser.initial_content_dom_wcag",
                        False,
                        expected="valid current owner-only browser evidence",
                        actual={"valid": False},
                        detail="Browser evidence failed closed without retaining error paths",
                    )
                ]
                browser = {"valid": False}
            checks.extend(browser_checks)
            identity_stability_checks, identity_stability_report = _spot_refetch_server_identity(
                audit_client, server_identity
            )
            checks.extend(identity_stability_checks)
            revision_after = int(_scalar(connection, "PRAGMA data_version"))
    finally:
        engine.dispose()

    _spot_progress(9, stage_count, "final-stability-and-budget-gates")
    database_after = _spot_database_stat(layout.database)
    sidecars_after = _spot_sqlite_sidecars(layout.database)
    try:
        baseline_after = _spot_file_binding(
            baseline_path,
            maximum_bytes=MAX_AUDITED_BASELINE_MANIFEST_BYTES,
        )
    except (OSError, RuntimeError):
        baseline_after = None
    baseline_check, baseline_binding_report = _spot_baseline_binding_check(
        baseline_before,
        baseline_after,
        schema_identity_valid=baseline_schema_identity_valid,
    )
    sidecar_check, sidecar_report = _spot_sidecar_check(sidecars_before, sidecars_after)
    checks.extend((baseline_check, sidecar_check))
    database_stable = database_before == database_after and revision_before == revision_after
    deep_newer_than_database = bool(
        deep_generated_ns is not None and deep_generated_ns >= database_before.mtime_ns
    )
    checks.append(
        _check(
            "catalog.production_spot_check_database_stable",
            database_stable,
            expected="same database device/inode/size/mtime and data_version",
            actual={
                "identity_stable": (database_before.device, database_before.inode)
                == (database_after.device, database_after.inode),
                "size_stable": database_before.size_bytes == database_after.size_bytes,
                "mtime_stable": database_before.mtime_ns == database_after.mtime_ns,
                "data_version_stable": revision_before == revision_after,
                "size_bytes": database_before.size_bytes,
            },
            detail="Database paths and raw inode values are omitted",
        )
    )
    checks.append(
        _check(
            "catalog.production_spot_check_deep_report_db_recency",
            deep_newer_than_database,
            expected="deep report generated at or after current stable database mtime",
            actual={"report_not_older_than_database": deep_newer_than_database},
            detail="Raw timestamps are omitted",
        )
    )
    api_calls = budget.calls if budget else 0
    checks.append(
        _check(
            "api.production_spot_check_global_budget",
            api_calls <= PRODUCTION_SPOT_CHECK_MAX_API_CALLS,
            expected={"maximum_calls": PRODUCTION_SPOT_CHECK_MAX_API_CALLS},
            actual={"calls": api_calls},
            detail="The wrapper rejects a request before the fixed global cap can be exceeded",
        )
    )

    serialized_checks = [asdict(check) for check in checks]
    blocking = [
        check for check in serialized_checks if check["required"] and check["status"] != "pass"
    ]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "archive_identity_sha256": _opaque(layout.root.name),
        "profile": {
            "mode": "production-spot-check",
            "profile_version": PRODUCTION_SPOT_CHECK_PROFILE_VERSION,
            "seed": policy.seed,
            "reduced_assurance": True,
            "full_source_blob_rehash_performed": False,
            "full_nlp_reconstruction_performed": False,
            "full_lance_scan_performed": False,
            "full_derived_reconstruction_performed": False,
            "all_pages_analytics_performed": False,
            "all_124_sort_streams_performed": False,
            "offset_pagination_used": False,
            "prior_deep_evidence_revalidated": prior_deep_revalidated,
            "prior_deep_evidence_covers_derived_products": False,
            "exact_global_catalog_totals": True,
            "unsampled_derived_rows_certified": False,
            "unsampled_derived_artifacts_certified": False,
            "unsampled_cursor_positions_certified": False,
            "independent_full_stream_sort_reconciliation_performed": False,
            "api_response_prefix_bounded": True,
            "api_endpoint_internal_work_bounded_by_profile": False,
            "maximum_sort_streams": PRODUCTION_SPOT_CHECK_SORT_STREAMS,
            "maximum_pages_per_sort_stream": PRODUCTION_SPOT_CHECK_PAGES_PER_STREAM,
            "maximum_rows_per_sort_page": PRODUCTION_SPOT_CHECK_PAGE_SIZE,
            "maximum_targeted_lookup_rows_per_sort_stream": (
                PRODUCTION_SPOT_CHECK_MAX_TARGETED_LOOKUP_ROWS
            ),
            "maximum_sample_rows_per_family": PRODUCTION_SPOT_CHECK_SAMPLE_ROWS,
            "maximum_api_calls": PRODUCTION_SPOT_CHECK_MAX_API_CALLS,
            "maximum_api_response_bytes": PRODUCTION_SPOT_CHECK_MAX_API_RESPONSE_BYTES,
            "observed_api_calls": api_calls,
            "limitation": (
                "Deep source/blob bytes are inherited from current artifact-bound prior deep "
                "evidence. Derived status totals are exact and selected rows/artifacts are "
                "sampled; unsampled derived rows/artifacts and unsampled cursor positions are "
                "not certified by this run. API response prefixes are capped, but some existing "
                "dashboard endpoints may internally build a full filtered sort projection."
            ),
        },
        "safety": {
            "sqlite_query_only": query_only == 1,
            "direct_catalog_connection_query_only": query_only == 1,
            "acceptance_read_only_server_verified": bool(server_identity),
            "workspace_read_only": query_only == 1 and (client is None or bool(server_identity)),
            "destructive_mutations_performed": False,
            "source_text_emitted": False,
            "names_titles_filenames_emitted": False,
            "api_response_bodies_emitted": False,
            "cloud_requests_performed": False,
        },
        "catalog_totals": totals,
        "reconciliation": reconciliation_report,
        "verification": {
            "prior_deep_gate": deep_report,
            "library_marker_totals_reconciled": marker_actual == marker_expected,
            "audited_baseline_binding": baseline_binding_report,
            "sqlite_sidecars": sidecar_report,
        },
        "jump_summary": jump_report,
        "search_acceptance": search_report,
        "nlp_integrity": nlp_report,
        "semantic_integrity": semantic_report,
        "derived_artifacts": derived_report,
        "analytics_bounded": analytics_report,
        "acceptance_server": {
            **identity_report,
            "final_stability": identity_stability_report,
        },
        "performance": {key: _timing_summary(value) for key, value in timings.items()},
        "browser": browser,
        "checks": serialized_checks,
        "gate": {
            "passed": not blocking,
            "blocking_count": len(blocking),
            "blocking_keys": [str(check["key"]) for check in blocking],
        },
    }


def _private_output_path(requested: Path | None) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    review_root = project_root / ".private-review"
    if requested is not None:
        candidate = requested.expanduser()
        if not candidate.is_absolute():
            candidate = project_root / candidate
        candidate = Path(os.path.abspath(candidate))
        try:
            candidate.relative_to(review_root)
        except ValueError as exc:
            raise RuntimeError(
                "Acceptance output must stay under the ignored .private-review/"
            ) from exc
        return candidate
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return review_root / f"real-export-acceptance-{stamp}" / "report.json"


def _secure_output_directory(directory: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    review_root = project_root / ".private-review"
    try:
        relative_parts = directory.relative_to(review_root).parts
    except ValueError as exc:
        raise RuntimeError(
            "Acceptance output must stay under the ignored .private-review/"
        ) from exc
    current = review_root
    for part in ("", *relative_parts):
        if part:
            current = current / part
        try:
            current_stat = current.lstat()
        except FileNotFoundError:
            current.mkdir(mode=OWNER_DIRECTORY_MODE)
            current_stat = current.lstat()
        if stat.S_ISLNK(current_stat.st_mode) or not stat.S_ISDIR(current_stat.st_mode):
            raise RuntimeError(f"Unsafe acceptance output directory: {current}")
        os.chmod(current, OWNER_DIRECTORY_MODE, follow_symlinks=False)


def _write_private_json(path: Path, payload: Mapping[str, object]) -> Path:
    _secure_output_directory(path.parent)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, OWNER_FILE_MODE)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(_safe_json_value(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-link publication is atomic and fails if the final name already exists. Unlike
        # replace(), it cannot overwrite prior acceptance evidence after a check/use race.
        os.link(temporary, path, follow_symlinks=False)
        temporary.unlink()
        directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def _read_startup_token(path: Path | None) -> str:
    if path is None:
        if not sys.stdin.isatty():
            raise RuntimeError("Use --startup-token-file for non-interactive acceptance runs")
        token = getpass.getpass("Dedicated server startup token: ").strip()
    else:
        token_path = path.expanduser()
        if not token_path.is_absolute():
            token_path = Path.cwd() / token_path
        token_path = Path(os.path.abspath(token_path))
        try:
            token = (
                _read_regular_bytes(
                    token_path,
                    label="Startup-token file",
                    maximum_bytes=4_096,
                    owner_only=True,
                )
                .decode("utf-8")
                .strip()
            )
        except UnicodeDecodeError:
            raise RuntimeError("Startup token is missing or malformed") from None
    if len(token) < 20 or any(character.isspace() for character in token):
        raise RuntimeError("Startup token is missing or malformed")
    return token


def build_acceptance_report(
    layout: ArchiveLayout,
    *,
    client: LocalApiClient | None,
    repeats: int,
    sample_count: int,
    query: str,
    browser_report: Path | None,
    reconciliation_policy: ReconciliationPolicy | None = None,
) -> dict[str, object]:
    reconciliation_policy = reconciliation_policy or ReconciliationPolicy(
        mode="exhaustive",
        seed=DEFAULT_RECONCILIATION_SEED,
    )
    if reconciliation_policy.mode == "production-spot-check":
        return _build_production_spot_check_report(
            layout,
            client=client,
            query=query,
            browser_report=browser_report,
            policy=reconciliation_policy,
        )
    engine = _readonly_catalog_engine(layout.database)
    checks: list[AcceptanceCheck] = []
    try:
        with engine.connect() as connection:
            query_only = int(_scalar(connection, "PRAGMA query_only"))
            tables = _table_names(connection)
            revision_before = int(_scalar(connection, "PRAGMA data_version"))
            totals = _catalog_totals(connection)

            identity_checks, identity_report, server_identity = _acceptance_server_identity(client)
            checks.extend(identity_checks)
            # Fail closed before calling any data-bearing API route. A normal app server lacks
            # the identity-only endpoint and may read or refresh derived caches on GET.
            audit_client = client if server_identity else None

            permission_findings = layout.private_permission_findings()
            checks.append(
                _check(
                    "catalog.private_runtime_permissions",
                    not permission_findings,
                    expected="all archive runtime boundaries are owner-only and symlink-free",
                    actual={"finding_count": len(permission_findings)},
                    detail="Finding paths are deliberately omitted from acceptance evidence",
                )
            )

            # Final acceptance binds integrity evidence to the bytes and catalog observed during
            # this run. A previously written deep report is only a preflight artifact and is not
            # trusted as proof that the archive has remained unchanged.
            try:
                audited_baseline = load_audited_baseline(layout.root)
            except AuditedBaselineManifestError:
                audited_baseline = None
            verification = verify_catalog(
                layout,
                engine,
                deep=True,
                require_audited_baseline=True,
            )
            mandatory_verification_failures = [
                item
                for item in verification
                if item.get("mandatory", True) is not False and item.get("status") != "pass"
            ]
            checks.append(
                _check(
                    "catalog.verification",
                    not mandatory_verification_failures,
                    expected="all mandatory current deep verification checks pass",
                    actual={
                        "check_count": len(verification),
                        "non_pass_keys": [
                            str(item.get("key")) for item in mandatory_verification_failures
                        ],
                    },
                    detail=(
                        "Uses the production deep verifier in SQLite query-only mode and hashes "
                        "the current source/blob bytes"
                    ),
                )
            )
            deep_gate_check, deep_gate = _current_deep_integrity_gate(verification)
            checks.append(deep_gate_check)
            baseline_reconciliation: dict[str, object] = {}
            required_baselines = {
                "baseline.manifest",
                "baseline.json_files",
                "baseline.transcript_containers",
                "baseline.ordinary_segments",
                "baseline.ordinary_messages",
                "baseline.support_segments",
                "baseline.support_messages",
                "baseline.exact_duplicate_messages",
                "baseline.unresolved_local_uri_occurrences",
                "baseline.unlinked_media_paths",
                "baseline.leading_message_gap_segments",
                "baseline.internal_message_gap_segments",
                "baseline.activity_reactions_raw",
                "baseline.activity_reactions_logical",
            }
            baseline_by_key = {
                str(item.get("key")): item for item in verification if item.get("key")
            }
            baseline_reconciliation = {
                key: {
                    "expected": baseline_by_key.get(key, {}).get("expected"),
                    "actual": baseline_by_key.get(key, {}).get("actual"),
                    "status": baseline_by_key.get(key, {}).get("status"),
                }
                for key in sorted(required_baselines)
                if isinstance(baseline_by_key.get(key, {}).get("expected"), (bool, int, float))
                and isinstance(baseline_by_key.get(key, {}).get("actual"), (bool, int, float))
            }
            baseline_non_pass = [
                key
                for key in sorted(required_baselines)
                if baseline_by_key.get(key, {}).get("status") != "pass"
            ]
            checks.append(
                _check(
                    "catalog.audited_export_baselines",
                    not baseline_non_pass,
                    expected=sorted(required_baselines),
                    actual={
                        "present_count": sum(key in baseline_by_key for key in required_baselines),
                        "non_pass_keys": baseline_non_pass,
                    },
                    detail=(
                        "The private manifest, JSON classification, messages/support, duplicates, "
                        "unresolved and unlinked media, gaps, and raw/logical reactions reconcile "
                        "exactly"
                    ),
                )
            )

            api_checks, inherited_timings, reconciliation_report = _api_reconciliation(
                connection,
                audit_client,
                totals,
                policy=reconciliation_policy,
            )
            checks.extend(api_checks)
            jump_checks, jump_timings, jump_summary = _jump_acceptance(
                connection,
                audit_client,
                sample_count=sample_count,
                expected_largest_message_count=(
                    audited_baseline.largest_conversation_messages
                    if audited_baseline is not None
                    else None
                ),
            )
            checks.extend(jump_checks)
            inherited_timings.setdefault("date_jump_api", []).extend(jump_timings)
            search_checks, search_report = _search_user_story_acceptance(
                connection,
                audit_client,
                query=query,
            )
            checks.extend(search_checks)

            nlp_checks, nlp_report = _nlp_integrity(connection, tables)
            checks.extend(nlp_checks)
            semantic_checks, semantic_report = _semantic_integrity(
                layout, engine, connection, tables
            )
            checks.extend(semantic_checks)
            derived_checks, derived_report = _derived_artifact_integrity(
                layout, engine, connection, tables
            )
            checks.extend(derived_checks)
            analytics_direct_checks, analytics_direct_report = _analytics_direct_sql_reconciliation(
                connection, audit_client
            )
            checks.extend(analytics_direct_checks)
            analytics_checks, analytics_report = _analytics_cache_integrity(layout, engine)
            checks.extend(analytics_checks)
            performance_checks, performance = _benchmark_api(
                audit_client,
                connection,
                repeats=repeats,
                query=query,
                inherited_timings=inherited_timings,
            )
            checks.extend(performance_checks)
            largest_for_browser = jump_summary.get("largest_conversation")
            browser_checks, browser = _browser_evidence(
                browser_report,
                expected_archive_identity=_opaque(layout.root.name),
                expected_totals=totals,
                expected_largest=(
                    largest_for_browser if isinstance(largest_for_browser, Mapping) else None
                ),
                expected_server_identity=server_identity,
            )
            checks.extend(browser_checks)
            revision_after = int(_scalar(connection, "PRAGMA data_version"))
            checks.append(
                _check(
                    "catalog.stable_during_audit",
                    revision_before == revision_after,
                    expected=revision_before,
                    actual=revision_after,
                    detail=(
                        "Catalog data_version stayed stable while direct and API totals were "
                        "compared"
                    ),
                )
            )
    finally:
        engine.dispose()

    serialized_checks = [asdict(check) for check in checks]
    blocking = [
        check for check in serialized_checks if check["required"] and check["status"] != "pass"
    ]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "archive_identity_sha256": _opaque(layout.root.name),
        "safety": {
            "sqlite_query_only": query_only == 1,
            "direct_catalog_connection_query_only": query_only == 1,
            "acceptance_read_only_server_verified": bool(server_identity),
            "workspace_read_only": query_only == 1 and (client is None or bool(server_identity)),
            "authenticated_gets_may_write_revision_bound_cache": False,
            "analytics_cache_reads_writes_disabled_for_api": (
                client is None or bool(server_identity)
            ),
            "destructive_mutations_performed": False,
            "source_text_emitted": False,
            "names_titles_filenames_emitted": False,
            "api_response_bodies_emitted": False,
            "cloud_requests_performed": False,
        },
        "catalog_totals": totals,
        "reconciliation": reconciliation_report,
        "verification": {
            "current_deep_gate": deep_gate,
            "audited_baseline_reconciliation": baseline_reconciliation,
            "checks": [
                {
                    "key": item.get("key"),
                    "status": item.get("status"),
                    "mandatory": item.get("mandatory", True),
                }
                for item in verification
            ],
        },
        "jump_summary": jump_summary,
        "search_acceptance": search_report,
        "nlp_integrity": nlp_report,
        "semantic_integrity": semantic_report,
        "derived_artifacts": derived_report,
        "analytics_direct_sql": analytics_direct_report,
        "analytics_cache": analytics_report,
        "acceptance_server": identity_report,
        "performance": performance,
        "browser": browser,
        "checks": serialized_checks,
        "gate": {
            "passed": not blocking,
            "blocking_count": len(blocking),
            "blocking_keys": [str(check["key"]) for check in blocking],
        },
    }


def _arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", required=True, type=Path)
    parser.add_argument(
        "--base-url",
        help=(
            "Dedicated localhost app URL. The unconsumed startup token is prompted without "
            "echo unless --startup-token-file is supplied."
        ),
    )
    parser.add_argument(
        "--startup-token-file",
        type=Path,
        help="Owner-only file containing the dedicated server startup token.",
    )
    parser.add_argument("--browser-report", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument(
        "--reconciliation-mode",
        required=True,
        choices=RECONCILIATION_MODES,
        help=(
            "Explicitly choose production-spot-check-v2, risk-bounded-v1 edge/cursor "
            "sampling, or the exhaustive all-pages audit."
        ),
    )
    parser.add_argument(
        "--reconciliation-seed",
        type=int,
        default=DEFAULT_RECONCILIATION_SEED,
        help="Unsigned 64-bit deterministic seed recorded in the private acceptance report.",
    )
    arguments = parser.parse_args(argv)
    if arguments.repeats < 3 or arguments.repeats > 100:
        parser.error("--repeats must be between 3 and 100")
    if arguments.samples < 1 or arguments.samples > 100:
        parser.error("--samples must be between 1 and 100")
    if not str(arguments.query).strip():
        parser.error("--query must not be empty")
    if arguments.reconciliation_seed < 0 or arguments.reconciliation_seed > 2**64 - 1:
        parser.error("--reconciliation-seed must be an unsigned 64-bit integer")
    if arguments.startup_token_file is not None and not arguments.base_url:
        parser.error("--startup-token-file requires --base-url")
    return arguments


def _report_exit_code(report: Mapping[str, object]) -> int:
    gate = report.get("gate")
    return 0 if isinstance(gate, dict) and gate.get("passed") is True else 1


def main(argv: Sequence[str] | None = None) -> int:
    os.umask(0o077)
    arguments = _arguments(argv or sys.argv[1:])
    root = arguments.archive_root.expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f"Archive root does not exist: {root}")
    layout = ArchiveLayout(root)
    client: LocalApiClient | None = None
    if arguments.base_url:
        token = _read_startup_token(arguments.startup_token_file)
        client = LocalApiClient(str(arguments.base_url), token)
    report = build_acceptance_report(
        layout,
        client=client,
        repeats=int(arguments.repeats),
        sample_count=int(arguments.samples),
        query=str(arguments.query).strip(),
        browser_report=arguments.browser_report,
        reconciliation_policy=ReconciliationPolicy(
            mode=str(arguments.reconciliation_mode),
            seed=int(arguments.reconciliation_seed),
        ),
    )
    output = _write_private_json(_private_output_path(arguments.output), report)
    gate = report["gate"]
    assert isinstance(gate, dict)
    print(
        json.dumps(
            {
                "report": str(output),
                "passed": gate["passed"],
                "blocking_count": gate["blocking_count"],
                "blocking_keys": gate["blocking_keys"],
            },
            sort_keys=True,
        )
    )
    return _report_exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
