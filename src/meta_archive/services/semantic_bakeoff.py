from __future__ import annotations

import gc
import hashlib
import json
import os
import sqlite3
import stat
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from meta_archive.config import settings
from meta_archive.paths import ArchiveLayout
from meta_archive.services.direct_conflict_ranker import (
    DIRECT_CONFLICT_EVALUATION_INTENT,
    DirectConflictFeedbackRevision,
    DirectConflictRankerError,
    DirectConflictTrainingData,
    DirectConflictTrainingExample,
    fit_direct_conflict_ranker,
)
from meta_archive.services.search_index import (
    EmbeddingModelSpec,
    TextEmbedder,
    default_text_embedder,
    embedding_implementation_identity,
    embedding_model_spec,
    retrieval_metrics,
)
from meta_archive.services.text import normalize_text

BAKEOFF_SCHEMA_VERSION = "semantic-model-bakeoff-v1"
FEEDBACK_DIGEST_VERSION = "semantic-model-bakeoff-feedback-v1"
CORPUS_DIGEST_VERSION = "semantic-model-bakeoff-corpus-v1"
MANIFEST_DIGEST_VERSION = "semantic-model-bakeoff-manifest-v1"
DIRECT_CONFLICT_BAKEOFF_REVISION = "semantic-model-bakeoff-direct-conflict-v1"
CONTROL_MODEL_ALIAS = "multilingual-e5-small"
DEFAULT_CORPUS_CAP = 10_000
DEFAULT_HARD_DISTRACTOR_CAP = 2_500
DEFAULT_QUERY_CAP = 64
DEFAULT_JUDGMENT_CAP_PER_QUERY = 256
DEFAULT_EMBEDDING_BATCH_SIZE = 4
RETRIEVAL_KS = (10, 50, 200)
SUPPORTED_SEARCH_MODES = frozenset({"semantic", "hybrid"})
SUPPORTED_LABELS = frozenset({"relevant", "not_relevant", "uncertain"})
SUPPORTED_ENTITY_FILTERS = frozenset(
    {"all", "message", "activity", "audio_transcript", "attachment_text"}
)
EMPTY_FILTER_SCOPE_SHA256 = hashlib.sha256(b"{}").hexdigest()
_UNIT_NORM_TOLERANCE = 1e-3


class BakeoffError(RuntimeError):
    """Fail-closed error carrying only an aggregate-safe machine code and counts."""

    def __init__(self, code: str, *, counts: Mapping[str, int] | None = None) -> None:
        self.code = code
        self.counts = dict(counts or {})
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class BakeoffFilterScope:
    entity_type: str = "all"
    conversation_id: str | None = None
    participant: str | None = None
    source_category: str | None = None
    from_ms: int | None = None
    to_ms: int | None = None


@dataclass(frozen=True, slots=True)
class _Judgment:
    private_window_id: str
    label: str
    evaluation_split: str | None
    evaluation_intent: str | None


@dataclass(frozen=True, slots=True)
class _QueryGroup:
    private_query_sha256: str
    private_query_text: str
    source_search_mode: str
    private_filter_scope_sha256: str
    filter_scope: BakeoffFilterScope
    judgments: tuple[_Judgment, ...]

    @property
    def relevant_window_ids(self) -> frozenset[str]:
        return frozenset(
            judgment.private_window_id
            for judgment in self.judgments
            if judgment.label == "relevant"
        )

    @property
    def evaluation_intents(self) -> frozenset[str]:
        return frozenset(
            judgment.evaluation_intent
            for judgment in self.judgments
            if judgment.evaluation_intent is not None
        )


@dataclass(frozen=True, slots=True)
class _WindowMetadata:
    private_window_id: str
    entity_type: str
    private_conversation_id: str | None
    source_category: str | None
    first_timestamp_ms: int | None
    last_timestamp_ms: int | None
    approx_token_count: int
    private_participants_search: str


@dataclass(frozen=True, slots=True)
class _CorpusWindow:
    metadata: _WindowMetadata
    private_text: str
    sample_source: Literal["judged", "hard_distractor", "stable_distractor"]


@dataclass(frozen=True, slots=True)
class BakeoffManifest:
    """Private in-memory bakeoff material.

    Only :meth:`safe_summary` may be serialized. The query text, result identities, window
    identities, participant text, and corpus text intentionally remain outside that payload.
    """

    roster: tuple[EmbeddingModelSpec, ...]
    private_active_config_id: str
    queries: tuple[_QueryGroup, ...]
    corpus: tuple[_CorpusWindow, ...]
    feedback_digest_sha256: str
    corpus_digest_sha256: str
    manifest_digest_sha256: str
    active_window_count: int
    corpus_cap: int
    hard_distractor_cap: int
    query_cap: int
    judgment_cap_per_query: int

    def safe_summary(self) -> dict[str, Any]:
        label_counts = {
            label: sum(
                judgment.label == label for group in self.queries for judgment in group.judgments
            )
            for label in sorted(SUPPORTED_LABELS)
        }
        source_counts = {
            source: sum(window.sample_source == source for window in self.corpus)
            for source in ("judged", "hard_distractor", "stable_distractor")
        }
        entity_type_counts: dict[str, int] = {}
        for window in self.corpus:
            entity_type = window.metadata.entity_type
            entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1
        retrieval_query_count = sum(bool(group.relevant_window_ids) for group in self.queries)
        direct_conflict_judgment_count = sum(
            judgment.evaluation_intent == DIRECT_CONFLICT_EVALUATION_INTENT
            for group in self.queries
            for judgment in group.judgments
        )
        return {
            "schema_version": BAKEOFF_SCHEMA_VERSION,
            "development_only": True,
            "activation_permitted": False,
            "private_query_details_persisted": False,
            "private_result_details_persisted": False,
            "model_roster": [_safe_model_contract(spec) for spec in self.roster],
            "active_control": {
                "alias": CONTROL_MODEL_ALIAS,
                "contract_verified": True,
            },
            "selection_policy": {
                "feedback": (
                    "semantic or hybrid relevance feedback whose split is not test, including "
                    "explicit direct-conflict development calibration"
                ),
                "corpus": (
                    "all judged targets, then stable-hash stratified hard distractors and "
                    "stable-hash stratified general distractors"
                ),
                "query_cap": self.query_cap,
                "judgment_cap_per_query": self.judgment_cap_per_query,
                "corpus_cap": self.corpus_cap,
                "hard_distractor_cap": self.hard_distractor_cap,
                "cap_behavior": "fail_closed_without_truncation",
                "test_split_rows_used_for_selection": 0,
            },
            "manifest": {
                "sha256": self.manifest_digest_sha256,
                "feedback_sha256": self.feedback_digest_sha256,
                "corpus_sha256": self.corpus_digest_sha256,
                "query_group_count": len(self.queries),
                "retrieval_query_group_count": retrieval_query_count,
                "judgment_count": sum(label_counts.values()),
                "judgment_label_counts": label_counts,
                "direct_conflict_judgment_count": direct_conflict_judgment_count,
                "judged_target_count": source_counts["judged"],
                "active_window_count": self.active_window_count,
                "corpus_window_count": len(self.corpus),
                "sample_source_counts": source_counts,
                "entity_type_counts": dict(sorted(entity_type_counts.items())),
            },
        }


class BakeoffEmbedderFactory(Protocol):
    def __call__(self, spec: EmbeddingModelSpec) -> TextEmbedder: ...


BAKEOFF_MODEL_ALIASES = frozenset(
    {
        CONTROL_MODEL_ALIAS,
        "bge-base-en-v1.5",
        "e5-base-v2",
        "e5-small-v2",
        "mxbai-embed-large-v1",
        "nomic-embed-text-v1.5",
        "snowflake-arctic-embed-l",
    }
)


def available_model_aliases() -> tuple[str, ...]:
    return tuple(sorted(BAKEOFF_MODEL_ALIASES))


def explicit_model_roster(aliases: Sequence[str]) -> tuple[EmbeddingModelSpec, ...]:
    if not aliases:
        raise BakeoffError("model_roster_required")
    if len(set(aliases)) != len(aliases):
        raise BakeoffError("model_roster_duplicate")
    if CONTROL_MODEL_ALIAS not in aliases:
        raise BakeoffError("active_e5_control_required")
    if len(aliases) < 2:
        raise BakeoffError("candidate_model_required")
    if any(alias not in BAKEOFF_MODEL_ALIASES for alias in aliases):
        raise BakeoffError("model_roster_unsupported")
    try:
        return tuple(embedding_model_spec(alias) for alias in aliases)
    except ValueError:
        raise BakeoffError("model_roster_unsupported") from None


def _safe_model_contract(spec: EmbeddingModelSpec) -> dict[str, Any]:
    return {
        "alias": spec.alias,
        "model_name": spec.model_name,
        "model_revision": spec.model_revision,
        "dimensions": spec.dimensions,
        "language_scope": spec.language_scope,
        "provider": spec.provider,
        "pooling": spec.pooling,
        "max_sequence_tokens": spec.max_sequence_tokens,
        "precision": spec.precision,
        "document_prefix": spec.document_prefix,
        "query_prefix": spec.query_prefix,
        "normalization": "l2",
    }


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8", errors="surrogatepass")
    except (TypeError, ValueError):
        raise BakeoffError("private_material_invalid") from None


def _framed_update(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _digest_records(version: str, records: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    _framed_update(digest, version.encode("ascii"))
    for record in records:
        _framed_update(digest, _canonical_bytes(record))
    return digest.hexdigest()


def _open_read_only_catalog(layout: ArchiveLayout) -> sqlite3.Connection:
    try:
        database_stat = layout.database.lstat()
        if stat.S_ISLNK(database_stat.st_mode) or not stat.S_ISREG(database_stat.st_mode):
            raise BakeoffError("catalog_path_unsafe")
        uri = f"{layout.database.resolve().as_uri()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("BEGIN")
        return connection
    except BakeoffError:
        raise
    except (OSError, sqlite3.Error):
        raise BakeoffError("catalog_unavailable") from None


def _json_object(value: object, *, error_code: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
        except (TypeError, ValueError):
            raise BakeoffError(error_code) from None
        if isinstance(loaded, dict):
            return cast(dict[str, Any], loaded)
    raise BakeoffError(error_code)


def _active_control_config(connection: sqlite3.Connection) -> tuple[str, int]:
    rows = connection.execute(
        """
        SELECT id, model_name, model_revision, dimensions, distance_metric,
               document_prefix, query_prefix, window_count, audit_json
        FROM semantic_index_configs
        WHERE active = 1 AND status = 'ready'
        ORDER BY id
        LIMIT 2
        """
    ).fetchall()
    if len(rows) != 1:
        raise BakeoffError("active_semantic_control_ambiguous", counts={"active_count": len(rows)})
    row = rows[0]
    control = embedding_model_spec(CONTROL_MODEL_ALIAS)
    if (
        str(row["model_name"]) != control.model_name
        or str(row["model_revision"]) != control.model_revision
        or int(row["dimensions"]) != control.dimensions
        or str(row["distance_metric"]) != "cosine"
        or str(row["document_prefix"]) != control.document_prefix
        or str(row["query_prefix"]) != control.query_prefix
    ):
        raise BakeoffError("active_e5_control_contract_mismatch")
    audit = _json_object(row["audit_json"], error_code="active_e5_control_audit_invalid")
    implementation = audit.get("embedding_implementation")
    expected_implementation = embedding_implementation_identity(control)
    if not isinstance(implementation, dict) or any(
        implementation.get(key) != expected for key, expected in expected_implementation.items()
    ):
        raise BakeoffError("active_e5_control_implementation_mismatch")
    window_count = int(row["window_count"])
    if window_count < 1:
        raise BakeoffError("active_e5_control_empty")
    return str(row["id"]), window_count


def _filter_scope(value: object, expected_sha256: str) -> BakeoffFilterScope:
    raw = _json_object(value, error_code="feedback_filter_scope_invalid")
    canonical = _canonical_bytes(raw)
    if hashlib.sha256(canonical).hexdigest() != expected_sha256:
        raise BakeoffError("feedback_filter_scope_digest_mismatch")
    supported_keys = {
        "entity_type",
        "conversation_id",
        "participant",
        "source_category",
        "from_ms",
        "to_ms",
    }
    if set(raw) - supported_keys:
        raise BakeoffError("feedback_filter_scope_unsupported")
    entity_type = raw.get("entity_type", "all")
    if not isinstance(entity_type, str) or entity_type not in SUPPORTED_ENTITY_FILTERS:
        raise BakeoffError("feedback_filter_scope_invalid")
    string_values: dict[str, str | None] = {}
    for key in ("conversation_id", "participant", "source_category"):
        value_at_key = raw.get(key)
        if value_at_key is not None and not isinstance(value_at_key, str):
            raise BakeoffError("feedback_filter_scope_invalid")
        string_values[key] = value_at_key
    integer_values: dict[str, int | None] = {}
    for key in ("from_ms", "to_ms"):
        value_at_key = raw.get(key)
        if value_at_key is not None and (
            not isinstance(value_at_key, int) or isinstance(value_at_key, bool)
        ):
            raise BakeoffError("feedback_filter_scope_invalid")
        integer_values[key] = value_at_key
    from_ms = integer_values["from_ms"]
    to_ms = integer_values["to_ms"]
    if (
        (from_ms is not None and from_ms < 0)
        or (to_ms is not None and to_ms < 0)
        or (from_ms is not None and to_ms is not None and from_ms > to_ms)
    ):
        raise BakeoffError("feedback_filter_scope_invalid")
    return BakeoffFilterScope(
        entity_type=entity_type,
        conversation_id=string_values["conversation_id"],
        participant=string_values["participant"],
        source_category=string_values["source_category"],
        from_ms=from_ms,
        to_ms=to_ms,
    )


def _development_query_groups(
    connection: sqlite3.Connection,
    *,
    query_cap: int,
    judgment_cap_per_query: int,
) -> tuple[_QueryGroup, ...]:
    if query_cap < 1 or judgment_cap_per_query < 1:
        raise BakeoffError("feedback_cap_invalid")
    maximum_rows = query_cap * judgment_cap_per_query
    rows = connection.execute(
        """
        SELECT query_sha256, query_text, search_mode, filter_scope_sha256,
               filter_scope_json, result_id, label, evaluation_split, evaluation_intent
        FROM relevance_feedback
        WHERE search_mode IN ('semantic', 'hybrid')
          AND (evaluation_split IS NULL OR evaluation_split <> 'test')
        ORDER BY query_sha256, query_text, search_mode, filter_scope_sha256, result_id
        LIMIT ?
        """,
        (maximum_rows + 1,),
    ).fetchall()
    if len(rows) > maximum_rows:
        raise BakeoffError(
            "development_feedback_row_cap_exceeded",
            counts={"row_cap": maximum_rows},
        )
    grouped: dict[tuple[str, str, str, str], tuple[BakeoffFilterScope, list[_Judgment]]] = {}
    for row in rows:
        query_sha256 = str(row["query_sha256"])
        query_text = str(row["query_text"])
        if hashlib.sha256(query_text.encode("utf-8")).hexdigest() != query_sha256:
            raise BakeoffError("feedback_query_digest_mismatch")
        search_mode = str(row["search_mode"])
        if search_mode not in SUPPORTED_SEARCH_MODES:
            raise BakeoffError("feedback_search_mode_invalid")
        filter_scope_sha256 = str(row["filter_scope_sha256"])
        scope = _filter_scope(row["filter_scope_json"], filter_scope_sha256)
        result_id = str(row["result_id"])
        if not result_id.startswith("window:") or not result_id.removeprefix("window:"):
            raise BakeoffError("development_feedback_target_invalid")
        label = str(row["label"])
        if label not in SUPPORTED_LABELS:
            raise BakeoffError("development_feedback_label_invalid")
        evaluation_split = (
            str(row["evaluation_split"]) if row["evaluation_split"] is not None else None
        )
        if evaluation_split == "test":
            raise BakeoffError("test_feedback_selection_forbidden")
        evaluation_intent = (
            str(row["evaluation_intent"]) if row["evaluation_intent"] is not None else None
        )
        if evaluation_intent == DIRECT_CONFLICT_EVALUATION_INTENT and (
            evaluation_split != "validation"
            or search_mode != "semantic"
            or filter_scope_sha256 != EMPTY_FILTER_SCOPE_SHA256
            or scope != BakeoffFilterScope()
        ):
            raise BakeoffError("direct_conflict_development_scope_invalid")
        key = (query_sha256, query_text, search_mode, filter_scope_sha256)
        existing = grouped.get(key)
        if existing is None:
            existing = (scope, [])
            grouped[key] = existing
        elif existing[0] != scope:
            raise BakeoffError("feedback_filter_scope_inconsistent")
        judgments = existing[1]
        private_window_id = result_id.removeprefix("window:")
        if any(judgment.private_window_id == private_window_id for judgment in judgments):
            raise BakeoffError("development_feedback_target_duplicate")
        judgments.append(
            _Judgment(
                private_window_id=private_window_id,
                label=label,
                evaluation_split=evaluation_split,
                evaluation_intent=evaluation_intent,
            )
        )
        if len(judgments) > judgment_cap_per_query:
            raise BakeoffError(
                "development_judgment_cap_exceeded",
                counts={"judgment_cap_per_query": judgment_cap_per_query},
            )
    if len(grouped) > query_cap:
        raise BakeoffError(
            "development_query_cap_exceeded",
            counts={"query_cap": query_cap},
        )
    if not grouped:
        raise BakeoffError("development_feedback_unavailable")
    groups = tuple(
        _QueryGroup(
            private_query_sha256=key[0],
            private_query_text=key[1],
            source_search_mode=key[2],
            private_filter_scope_sha256=key[3],
            filter_scope=value[0],
            judgments=tuple(value[1]),
        )
        for key, value in sorted(grouped.items())
    )
    if not any(group.relevant_window_ids for group in groups):
        raise BakeoffError("development_relevant_feedback_unavailable")
    return groups


def _length_bucket(token_count: int) -> str:
    if token_count <= 64:
        return "000-064"
    if token_count <= 128:
        return "065-128"
    if token_count <= 256:
        return "129-256"
    if token_count <= 512:
        return "257-512"
    return "513-plus"


def _year_bucket(timestamp_ms: int | None) -> str:
    if timestamp_ms is None:
        return "unknown"
    try:
        return str(datetime.fromtimestamp(timestamp_ms / 1_000, tz=UTC).year)
    except (OSError, OverflowError, ValueError):
        return "out-of-range"


def _window_stratum(window: _WindowMetadata) -> tuple[str, str, str]:
    return (
        window.entity_type,
        _length_bucket(window.approx_token_count),
        _year_bucket(window.first_timestamp_ms),
    )


def _active_window_metadata(
    connection: sqlite3.Connection,
    *,
    active_config_id: str,
    expected_count: int,
) -> dict[str, _WindowMetadata]:
    rows = connection.execute(
        """
        SELECT window.id, window.entity_type, window.conversation_id, window.source_category,
               window.first_timestamp_ms, window.last_timestamp_ms, window.approx_token_count,
               window.participants_search
        FROM semantic_window_embeddings AS membership
        JOIN search_windows AS window ON window.id = membership.window_id
        WHERE membership.config_id = ?
        ORDER BY window.id
        """,
        (active_config_id,),
    )
    metadata: dict[str, _WindowMetadata] = {}
    for row in rows:
        private_window_id = str(row["id"])
        if private_window_id in metadata:
            raise BakeoffError("active_window_membership_duplicate")
        metadata[private_window_id] = _WindowMetadata(
            private_window_id=private_window_id,
            entity_type=str(row["entity_type"]),
            private_conversation_id=(
                str(row["conversation_id"]) if row["conversation_id"] is not None else None
            ),
            source_category=(
                str(row["source_category"]) if row["source_category"] is not None else None
            ),
            first_timestamp_ms=(
                int(row["first_timestamp_ms"]) if row["first_timestamp_ms"] is not None else None
            ),
            last_timestamp_ms=(
                int(row["last_timestamp_ms"]) if row["last_timestamp_ms"] is not None else None
            ),
            approx_token_count=int(row["approx_token_count"]),
            private_participants_search=str(row["participants_search"] or ""),
        )
    if len(metadata) != expected_count:
        raise BakeoffError(
            "active_window_membership_count_mismatch",
            counts={
                "declared_window_count": expected_count,
                "membership_window_count": len(metadata),
            },
        )
    return metadata


def _stable_hash(salt: str, private_value: str) -> str:
    return hashlib.sha256(
        f"{BAKEOFF_SCHEMA_VERSION}\0{salt}\0{private_value}".encode("utf-8", errors="surrogatepass")
    ).hexdigest()


def _stratified_stable_pick(
    candidates: Sequence[_WindowMetadata],
    *,
    limit: int,
    salt: str,
) -> list[_WindowMetadata]:
    if limit <= 0 or not candidates:
        return []
    by_stratum: dict[tuple[str, str, str], list[tuple[str, _WindowMetadata]]] = defaultdict(list)
    for candidate in candidates:
        by_stratum[_window_stratum(candidate)].append(
            (_stable_hash(salt, candidate.private_window_id), candidate)
        )
    for values in by_stratum.values():
        values.sort(key=lambda value: (value[0], value[1].private_window_id))
    stratum_order = sorted(
        by_stratum,
        key=lambda stratum: (_stable_hash(f"{salt}-stratum", repr(stratum)), stratum),
    )
    positions = {stratum: 0 for stratum in stratum_order}
    selected: list[_WindowMetadata] = []
    while len(selected) < limit:
        advanced = False
        for stratum in stratum_order:
            position = positions[stratum]
            values = by_stratum[stratum]
            if position >= len(values):
                continue
            selected.append(values[position][1])
            positions[stratum] = position + 1
            advanced = True
            if len(selected) >= limit:
                break
        if not advanced:
            break
    return selected


def _load_selected_texts(
    connection: sqlite3.Connection,
    selected_ids: Sequence[str],
) -> dict[str, str]:
    texts: dict[str, str] = {}
    for offset in range(0, len(selected_ids), 400):
        batch = list(selected_ids[offset : offset + 400])
        placeholders = ",".join("?" for _value in batch)
        rows = connection.execute(
            f"SELECT id, text_display FROM search_windows WHERE id IN ({placeholders})",
            batch,
        ).fetchall()
        for row in rows:
            private_window_id = str(row["id"])
            if private_window_id in texts:
                raise BakeoffError("selected_window_duplicate")
            text = str(row["text_display"] or "")
            if not text:
                raise BakeoffError("selected_window_text_unavailable")
            texts[private_window_id] = text
    if len(texts) != len(selected_ids):
        raise BakeoffError(
            "selected_window_text_incomplete",
            counts={
                "selected_window_count": len(selected_ids),
                "loaded_text_count": len(texts),
            },
        )
    return texts


def _matches_filter(window: _WindowMetadata, scope: BakeoffFilterScope) -> bool:
    entity_type = scope.entity_type
    if entity_type == "message" and window.entity_type != "message_window":
        return False
    if entity_type == "activity" and window.entity_type != "activity":
        return False
    if entity_type == "audio_transcript" and window.entity_type != "audio_transcript":
        return False
    if entity_type == "attachment_text" and window.entity_type != "attachment_text":
        return False
    if (
        scope.conversation_id is not None
        and window.private_conversation_id != scope.conversation_id
    ):
        return False
    if scope.source_category is not None and window.source_category != scope.source_category:
        return False
    if scope.from_ms is not None and (
        window.last_timestamp_ms is None or window.last_timestamp_ms < scope.from_ms
    ):
        return False
    if scope.to_ms is not None and (
        window.first_timestamp_ms is None or window.first_timestamp_ms > scope.to_ms
    ):
        return False
    if scope.participant is not None:
        participant = normalize_text(scope.participant).search
        if participant not in window.private_participants_search:
            return False
    return True


def _private_feedback_records(groups: Sequence[_QueryGroup]) -> Iterable[Any]:
    for group in groups:
        for judgment in group.judgments:
            yield [
                group.private_query_sha256,
                hashlib.sha256(
                    group.private_query_text.encode("utf-8", errors="surrogatepass")
                ).hexdigest(),
                group.source_search_mode,
                group.private_filter_scope_sha256,
                judgment.private_window_id,
                judgment.label,
                judgment.evaluation_split,
                judgment.evaluation_intent,
            ]


def build_bakeoff_manifest(
    layout: ArchiveLayout,
    *,
    model_aliases: Sequence[str],
    corpus_cap: int = DEFAULT_CORPUS_CAP,
    hard_distractor_cap: int = DEFAULT_HARD_DISTRACTOR_CAP,
    query_cap: int = DEFAULT_QUERY_CAP,
    judgment_cap_per_query: int = DEFAULT_JUDGMENT_CAP_PER_QUERY,
) -> BakeoffManifest:
    if corpus_cap < 1 or hard_distractor_cap < 0 or hard_distractor_cap > corpus_cap:
        raise BakeoffError("corpus_cap_invalid")
    roster = explicit_model_roster(model_aliases)
    connection = _open_read_only_catalog(layout)
    try:
        active_config_id, active_window_count = _active_control_config(connection)
        groups = _development_query_groups(
            connection,
            query_cap=query_cap,
            judgment_cap_per_query=judgment_cap_per_query,
        )
        metadata = _active_window_metadata(
            connection,
            active_config_id=active_config_id,
            expected_count=active_window_count,
        )
        judged_ids = {
            judgment.private_window_id for group in groups for judgment in group.judgments
        }
        missing_judged_count = sum(private_id not in metadata for private_id in judged_ids)
        if missing_judged_count:
            raise BakeoffError(
                "judged_target_missing_from_active_corpus",
                counts={"missing_judged_target_count": missing_judged_count},
            )
        if len(judged_ids) > corpus_cap:
            raise BakeoffError(
                "judged_targets_exceed_corpus_cap",
                counts={"judged_target_count": len(judged_ids), "corpus_cap": corpus_cap},
            )
        judged_conversations = {
            metadata[private_id].private_conversation_id
            for private_id in judged_ids
            if metadata[private_id].private_conversation_id is not None
        }
        judged_strata = {_window_stratum(metadata[private_id]) for private_id in judged_ids}
        unjudged = [
            window for private_id, window in metadata.items() if private_id not in judged_ids
        ]
        hard_candidates = [
            window
            for window in unjudged
            if (
                window.private_conversation_id is not None
                and window.private_conversation_id in judged_conversations
            )
            or _window_stratum(window) in judged_strata
        ]
        distractor_budget = corpus_cap - len(judged_ids)
        selected_hard = _stratified_stable_pick(
            hard_candidates,
            limit=min(distractor_budget, hard_distractor_cap),
            salt="hard-distractor",
        )
        hard_ids = {window.private_window_id for window in selected_hard}
        selected_stable = _stratified_stable_pick(
            [window for window in unjudged if window.private_window_id not in hard_ids],
            limit=distractor_budget - len(selected_hard),
            salt="stable-distractor",
        )
        sample_sources: dict[str, Literal["judged", "hard_distractor", "stable_distractor"]] = {
            private_id: "judged" for private_id in judged_ids
        }
        sample_sources.update(
            {window.private_window_id: "hard_distractor" for window in selected_hard}
        )
        sample_sources.update(
            {window.private_window_id: "stable_distractor" for window in selected_stable}
        )
        selected_ids = sorted(
            sample_sources,
            key=lambda private_id: (_stable_hash("corpus-order", private_id), private_id),
        )
        selected_texts = _load_selected_texts(connection, selected_ids)
        corpus = tuple(
            _CorpusWindow(
                metadata=metadata[private_id],
                private_text=selected_texts[private_id],
                sample_source=sample_sources[private_id],
            )
            for private_id in selected_ids
        )
        corpus_by_id = {window.metadata.private_window_id: window.metadata for window in corpus}
        relevant_filter_mismatches = sum(
            not _matches_filter(corpus_by_id[private_id], group.filter_scope)
            for group in groups
            for private_id in group.relevant_window_ids
        )
        if relevant_filter_mismatches:
            raise BakeoffError(
                "relevant_target_filter_mismatch",
                counts={"mismatch_count": relevant_filter_mismatches},
            )
        feedback_digest = _digest_records(
            FEEDBACK_DIGEST_VERSION,
            _private_feedback_records(groups),
        )
        corpus_digest = _digest_records(
            CORPUS_DIGEST_VERSION,
            (
                [
                    window.metadata.private_window_id,
                    hashlib.sha256(
                        window.private_text.encode("utf-8", errors="surrogatepass")
                    ).hexdigest(),
                    window.metadata.entity_type,
                    window.metadata.private_conversation_id,
                    window.metadata.source_category,
                    window.metadata.first_timestamp_ms,
                    window.metadata.last_timestamp_ms,
                    window.metadata.approx_token_count,
                    hashlib.sha256(
                        window.metadata.private_participants_search.encode(
                            "utf-8", errors="surrogatepass"
                        )
                    ).hexdigest(),
                    window.sample_source,
                ]
                for window in corpus
            ),
        )
        manifest_digest = _digest_records(
            MANIFEST_DIGEST_VERSION,
            (
                {
                    "feedback_sha256": feedback_digest,
                    "corpus_sha256": corpus_digest,
                    "roster": [_safe_model_contract(spec) for spec in roster],
                    "corpus_cap": corpus_cap,
                    "hard_distractor_cap": hard_distractor_cap,
                    "query_cap": query_cap,
                    "judgment_cap_per_query": judgment_cap_per_query,
                },
            ),
        )
        return BakeoffManifest(
            roster=roster,
            private_active_config_id=active_config_id,
            queries=groups,
            corpus=corpus,
            feedback_digest_sha256=feedback_digest,
            corpus_digest_sha256=corpus_digest,
            manifest_digest_sha256=manifest_digest,
            active_window_count=active_window_count,
            corpus_cap=corpus_cap,
            hard_distractor_cap=hard_distractor_cap,
            query_cap=query_cap,
            judgment_cap_per_query=judgment_cap_per_query,
        )
    except sqlite3.Error:
        raise BakeoffError("catalog_snapshot_failed") from None
    finally:
        connection.rollback()
        connection.close()


def _validated_matrix(
    vectors: Sequence[Sequence[float]],
    *,
    expected_count: int,
    dimensions: int,
) -> NDArray[np.float32]:
    try:
        matrix = np.asarray(vectors, dtype=np.float32)
    except (TypeError, ValueError, OverflowError):
        raise BakeoffError("embedding_vector_invalid") from None
    if matrix.shape != (expected_count, dimensions) or not bool(np.isfinite(matrix).all()):
        raise BakeoffError("embedding_vector_invalid")
    norms = np.linalg.norm(matrix, axis=1)
    if not bool(
        np.allclose(
            norms,
            np.ones(expected_count, dtype=np.float32),
            rtol=_UNIT_NORM_TOLERANCE,
            atol=_UNIT_NORM_TOLERANCE,
        )
    ):
        raise BakeoffError("embedding_normalization_contract_violated")
    return np.ascontiguousarray(matrix, dtype=np.float32)


def _average_precision_at_k(
    rankings: Sequence[Sequence[str]],
    relevant_sets: Sequence[frozenset[str]],
    *,
    k: int,
) -> float:
    if not rankings:
        return 0.0
    values: list[float] = []
    for ranking, relevant in zip(rankings, relevant_sets, strict=True):
        if not relevant:
            raise BakeoffError("retrieval_relevance_set_empty")
        hits = 0
        precision_sum = 0.0
        for rank, private_window_id in enumerate(ranking[:k], start=1):
            if private_window_id in relevant:
                hits += 1
                precision_sum += hits / rank
        values.append(precision_sum / min(len(relevant), k))
    return sum(values) / len(values)


def _metric_bundle(
    rankings: Sequence[Sequence[str]],
    relevant_sets: Sequence[frozenset[str]],
) -> dict[str, int | float]:
    at_ten = retrieval_metrics(rankings, relevant_sets, k=10)
    return {
        "query_count": int(at_ten["query_count"]),
        "recall_at_10": float(at_ten["recall_at_k"]),
        "mrr_at_10": float(at_ten["mrr"]),
        "ndcg_at_10": float(at_ten["ndcg_at_k"]),
        "recall_at_50": float(retrieval_metrics(rankings, relevant_sets, k=50)["recall_at_k"]),
        "recall_at_200": float(retrieval_metrics(rankings, relevant_sets, k=200)["recall_at_k"]),
        "average_precision_at_200": _average_precision_at_k(
            rankings,
            relevant_sets,
            k=200,
        ),
    }


def _direct_conflict_grouped_validation(
    manifest: BakeoffManifest,
    matrix: NDArray[np.float32],
    corpus_positions: Mapping[str, int],
    *,
    spec: EmbeddingModelSpec,
) -> dict[str, Any]:
    direct_judgments = [
        judgment
        for group in manifest.queries
        for judgment in group.judgments
        if judgment.evaluation_intent == DIRECT_CONFLICT_EVALUATION_INTENT
    ]
    if not direct_judgments:
        return {
            "status": "not_available",
            "judgment_count": 0,
            "query_details_persisted": False,
        }
    labels_by_id: dict[str, str] = {}
    for judgment in direct_judgments:
        previous = labels_by_id.get(judgment.private_window_id)
        if previous is not None:
            code = (
                "direct_conflict_feedback_label_conflict"
                if previous != judgment.label
                else "direct_conflict_feedback_target_duplicate"
            )
            return {
                "status": "failed",
                "failure_code": code,
                "judgment_count": len(direct_judgments),
                "query_details_persisted": False,
            }
        labels_by_id[judgment.private_window_id] = judgment.label
    corpus_by_id = {window.metadata.private_window_id: window for window in manifest.corpus}
    examples = tuple(
        DirectConflictTrainingExample(
            private_window_id=private_window_id,
            private_conversation_group=(
                corpus_by_id[private_window_id].metadata.private_conversation_id
                or f"window:{private_window_id}"
            ),
            label=cast(Any, label),
            vector=matrix[corpus_positions[private_window_id]],
        )
        for private_window_id, label in sorted(labels_by_id.items())
    )
    prepared = DirectConflictTrainingData(
        examples=examples,
        dimensions=spec.dimensions,
        private_config_material={
            "schema_version": BAKEOFF_SCHEMA_VERSION,
            "manifest_sha256": manifest.manifest_digest_sha256,
            "model": _safe_model_contract(spec),
        },
        feedback_revision=DirectConflictFeedbackRevision(
            version=DIRECT_CONFLICT_BAKEOFF_REVISION,
            sha256=manifest.feedback_digest_sha256,
            row_count=len(examples),
        ),
    )
    try:
        ranker = fit_direct_conflict_ranker(prepared)
    except DirectConflictRankerError as exc:
        return {
            "status": "failed",
            "failure_code": exc.code,
            "judgment_count": len(examples),
            "query_details_persisted": False,
        }
    audit = ranker.audit
    return {
        "status": "completed",
        "judgment_count": audit.counts.reviewed_feedback_rows,
        "binary_judgment_count": audit.counts.binary_training_rows,
        "relevant_count": audit.counts.relevant_rows,
        "not_relevant_count": audit.counts.not_relevant_rows,
        "uncertain_count": audit.counts.uncertain_rows,
        "conversation_count": audit.counts.conversation_groups,
        "fold_count": audit.metrics.fold_count,
        "minimum_metrics": {
            "average_precision": audit.metrics.average_precision.minimum,
            "roc_auc": audit.metrics.roc_auc.minimum,
            "recall_at_10": audit.metrics.recall_at_10.minimum,
            "mrr_at_10": audit.metrics.mrr_at_10.minimum,
            "ndcg_at_10": audit.metrics.ndcg_at_10.minimum,
        },
        "mean_metrics": {
            "average_precision": audit.metrics.average_precision.mean,
            "roc_auc": audit.metrics.roc_auc.mean,
            "recall_at_10": audit.metrics.recall_at_10.mean,
            "mrr_at_10": audit.metrics.mrr_at_10.mean,
            "ndcg_at_10": audit.metrics.ndcg_at_10.mean,
        },
        "query_details_persisted": False,
    }


def _evaluate_model(
    manifest: BakeoffManifest,
    spec: EmbeddingModelSpec,
    embedder_factory: BakeoffEmbedderFactory,
) -> dict[str, Any]:
    embedder = embedder_factory(spec)
    if embedder.spec != spec:
        raise BakeoffError("embedder_model_contract_mismatch")
    documents = [window.private_text for window in manifest.corpus]
    matrix = _validated_matrix(
        embedder.embed_documents(documents),
        expected_count=len(documents),
        dimensions=spec.dimensions,
    )
    corpus_positions = {
        window.metadata.private_window_id: index for index, window in enumerate(manifest.corpus)
    }
    rankings: list[list[str]] = []
    relevant_sets: list[frozenset[str]] = []
    modes: list[str] = []
    intents: list[frozenset[str]] = []
    for group in manifest.queries:
        relevant = group.relevant_window_ids
        if not relevant:
            continue
        query_vector = _validated_matrix(
            [embedder.embed_query(group.private_query_text)],
            expected_count=1,
            dimensions=spec.dimensions,
        )[0]
        eligible_positions = [
            index
            for index, window in enumerate(manifest.corpus)
            if _matches_filter(window.metadata, group.filter_scope)
        ]
        if not eligible_positions:
            raise BakeoffError("query_candidate_pool_empty")
        eligible_matrix = matrix[np.asarray(eligible_positions, dtype=np.int64)]
        scores = eligible_matrix @ query_vector
        if scores.shape != (len(eligible_positions),) or not bool(np.isfinite(scores).all()):
            raise BakeoffError("cosine_score_invalid")
        order = np.argsort(-scores, kind="stable")
        ranking = [
            manifest.corpus[eligible_positions[int(position)]].metadata.private_window_id
            for position in order[: max(RETRIEVAL_KS)]
        ]
        if not relevant.issubset(
            {
                manifest.corpus[position].metadata.private_window_id
                for position in eligible_positions
            }
        ):
            raise BakeoffError("relevant_target_missing_from_candidate_pool")
        rankings.append(ranking)
        relevant_sets.append(relevant)
        modes.append(group.source_search_mode)
        intents.append(group.evaluation_intents)
    if not rankings:
        raise BakeoffError("retrieval_queries_unavailable")
    metrics_by_mode = {
        mode: _metric_bundle(
            [
                ranking
                for ranking, source_mode in zip(rankings, modes, strict=True)
                if source_mode == mode
            ],
            [
                relevant
                for relevant, source_mode in zip(relevant_sets, modes, strict=True)
                if source_mode == mode
            ],
        )
        for mode in sorted(SUPPORTED_SEARCH_MODES)
    }
    direct_indices = [
        index
        for index, group_intents in enumerate(intents)
        if DIRECT_CONFLICT_EVALUATION_INTENT in group_intents
    ]
    direct_retrieval = _metric_bundle(
        [rankings[index] for index in direct_indices],
        [relevant_sets[index] for index in direct_indices],
    )
    grouped_validation = _direct_conflict_grouped_validation(
        manifest,
        matrix,
        corpus_positions,
        spec=spec,
    )
    del embedder
    del matrix
    gc.collect()
    return {
        "model": _safe_model_contract(spec),
        "retrieval": {
            "engine": "in-memory-exact-cosine",
            "reranker_used": False,
            "lexical_fusion_used": False,
            "metrics": _metric_bundle(rankings, relevant_sets),
            "metrics_by_feedback_source_mode": metrics_by_mode,
            "direct_conflict_query_metrics": direct_retrieval,
        },
        "direct_conflict_grouped_validation": grouped_validation,
        "private_query_details_persisted": False,
        "private_result_details_persisted": False,
    }


def _development_candidate_gate(
    candidate: Mapping[str, Any],
    control: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_metrics = cast(Mapping[str, int | float], candidate["retrieval"]["metrics"])
    control_metrics = cast(Mapping[str, int | float], control["retrieval"]["metrics"])
    protected_deltas = {
        name: float(candidate_metrics[name]) - float(control_metrics[name])
        for name in (
            "recall_at_10",
            "mrr_at_10",
            "ndcg_at_10",
            "recall_at_50",
            "recall_at_200",
            "average_precision_at_200",
        )
    }
    broad_recall_non_regression = all(
        protected_deltas[name] >= -0.01 for name in ("recall_at_50", "recall_at_200")
    )
    top_rank_non_regression = all(
        protected_deltas[name] >= -0.02
        for name in (
            "recall_at_10",
            "mrr_at_10",
            "ndcg_at_10",
            "average_precision_at_200",
        )
    )
    demonstrated_improvement = any(
        protected_deltas[name] >= 0.02
        for name in (
            "recall_at_10",
            "mrr_at_10",
            "ndcg_at_10",
            "average_precision_at_200",
        )
    )
    candidate_direct = candidate["retrieval"].get("direct_conflict_query_metrics")
    candidate_direct = candidate_direct if isinstance(candidate_direct, dict) else {}
    control_direct = control["retrieval"].get("direct_conflict_query_metrics")
    control_direct = control_direct if isinstance(control_direct, dict) else {}
    direct_query_deltas = {
        name: float(candidate_direct.get(name, 0.0)) - float(control_direct.get(name, 0.0))
        for name in ("recall_at_10", "mrr_at_10", "ndcg_at_10")
    }
    direct_query_non_regression = bool(
        isinstance(candidate_direct.get("query_count"), int)
        and not isinstance(candidate_direct.get("query_count"), bool)
        and candidate_direct["query_count"] > 0
        and candidate_direct["query_count"] == control_direct.get("query_count")
        and all(delta >= -0.02 for delta in direct_query_deltas.values())
    )
    grouped = candidate.get("direct_conflict_grouped_validation")
    grouped = grouped if isinstance(grouped, dict) else {}
    minimums = grouped.get("minimum_metrics")
    minimums = minimums if isinstance(minimums, dict) else {}
    direct_conflict_passed = bool(
        grouped.get("status") == "completed"
        and isinstance(minimums.get("recall_at_10"), (int, float))
        and not isinstance(minimums.get("recall_at_10"), bool)
        and float(minimums["recall_at_10"]) >= 0.60
        and isinstance(minimums.get("mrr_at_10"), (int, float))
        and not isinstance(minimums.get("mrr_at_10"), bool)
        and float(minimums["mrr_at_10"]) >= 0.50
        and isinstance(minimums.get("ndcg_at_10"), (int, float))
        and not isinstance(minimums.get("ndcg_at_10"), bool)
        and float(minimums["ndcg_at_10"]) >= 0.55
    )
    passed = bool(
        broad_recall_non_regression
        and top_rank_non_regression
        and demonstrated_improvement
        and direct_query_non_regression
        and direct_conflict_passed
    )
    return {
        "passed": passed,
        "metric_deltas": protected_deltas,
        "direct_conflict_query_metric_deltas": direct_query_deltas,
        "criteria": {
            "recall_at_50_and_200_minimum_delta": -0.01,
            "top_rank_and_average_precision_minimum_delta": -0.02,
            "minimum_demonstrated_improvement": 0.02,
            "direct_conflict_query_minimum_delta": -0.02,
            "direct_conflict_grouped_minimums": {
                "recall_at_10": 0.60,
                "mrr_at_10": 0.50,
                "ndcg_at_10": 0.55,
            },
        },
        "checks": {
            "broad_recall_non_regression": broad_recall_non_regression,
            "top_rank_non_regression": top_rank_non_regression,
            "demonstrated_improvement": demonstrated_improvement,
            "direct_conflict_query_non_regression": direct_query_non_regression,
            "direct_conflict_grouped_minimums": direct_conflict_passed,
        },
    }


def evaluate_bakeoff(
    manifest: BakeoffManifest,
    *,
    embedder_factory: BakeoffEmbedderFactory,
) -> dict[str, Any]:
    results = [_evaluate_model(manifest, spec, embedder_factory) for spec in manifest.roster]
    control = next(result for result in results if result["model"]["alias"] == CONTROL_MODEL_ALIAS)
    comparisons: list[dict[str, Any]] = []
    for result in results:
        alias = str(result["model"]["alias"])
        if alias == CONTROL_MODEL_ALIAS:
            continue
        gate = _development_candidate_gate(result, control)
        comparisons.append(
            {
                "candidate_alias": alias,
                "control_alias": CONTROL_MODEL_ALIAS,
                "development_gate": gate,
            }
        )
    passing_aliases = {
        str(comparison["candidate_alias"])
        for comparison in comparisons
        if comparison["development_gate"]["passed"] is True
    }
    passing_results = [
        result for result in results if str(result["model"]["alias"]) in passing_aliases
    ]
    recommendation = (
        max(
            passing_results,
            key=lambda result: (
                float(result["retrieval"]["metrics"]["recall_at_200"]),
                float(result["retrieval"]["metrics"]["ndcg_at_10"]),
                float(result["retrieval"]["metrics"]["recall_at_10"]),
                str(result["model"]["alias"]),
            ),
        )
        if passing_results
        else None
    )
    return {
        **manifest.safe_summary(),
        "status": "completed",
        "evaluation": {
            "kind": "bounded_development_model_selection",
            "distance": "cosine",
            "exact": True,
            "development_only": True,
            "activation_permitted": False,
            "results": results,
            "comparisons_to_control": comparisons,
            "development_recommendation": {
                "status": (
                    "candidate_passed" if recommendation is not None else "no_candidate_passed"
                ),
                "alias": recommendation["model"]["alias"] if recommendation is not None else None,
                "keep_active_control": recommendation is None,
                "basis": (
                    "Among candidates passing every explicit development gate, highest "
                    "Recall@200, then nDCG@10, then Recall@10 on the bounded exact-cosine sample"
                ),
                "requires_full_corpus_regression_acceptance": True,
                "blind_acceptance_available": False,
                "existing_test_cohort_role": "exposed_overlap_regression_only",
            },
        },
    }


def production_embedder_factory(
    layout: ArchiveLayout,
    *,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    local_files_only: bool = False,
) -> BakeoffEmbedderFactory:
    if batch_size < 1 or batch_size > 1_024:
        raise BakeoffError("embedding_batch_size_invalid")
    settings.embedding_batch_size = batch_size

    def factory(spec: EmbeddingModelSpec) -> TextEmbedder:
        if settings.embedding_batch_size != batch_size:
            raise BakeoffError("embedding_batch_size_changed")
        return default_text_embedder(
            layout,
            spec,
            local_files_only=local_files_only,
        )

    return factory


def write_private_report(path: Path, report: Mapping[str, Any]) -> None:
    """Atomically write one aggregate-only report with owner-only permissions."""

    rendered = json.dumps(
        report,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | int(getattr(os, "O_CLOEXEC", 0)),
            0o600,
        )
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600, follow_symlinks=False)
    except (OSError, TypeError, ValueError):
        raise BakeoffError("aggregate_report_write_failed") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        with suppress(FileNotFoundError):
            temporary.unlink()
