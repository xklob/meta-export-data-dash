from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from meta_archive.models import RelevanceFeedback, SearchWindow, SearchWindowMessage
from meta_archive.services.direct_conflict_ranker import is_direct_conflict_trigger

SEMANTIC_SPLIT_AUDIT_VERSION = "semantic-feedback-split-audit-v1"
_SUPPORTED_LABELS = frozenset(("relevant", "not_relevant", "uncertain"))
_WINDOW_TARGET_PREFIX = "window:"
_QUERY_DIGEST_BYTES = 32
_SQL_BATCH_SIZE = 500
_SOURCE_ID_KEYS = frozenset(
    (
        "message_ids",
        "message_id",
        "activity_id",
        "attachment_occurrence_id",
        "audio_segment_id",
        "attachment_extraction_id",
    )
)


class SemanticSplitAuditError(RuntimeError):
    """Fail-closed error whose message never contains private archive material."""

    def __init__(
        self,
        code: str,
        *,
        audit: SemanticSplitAudit | None = None,
    ) -> None:
        self.code = code
        self.audit = audit
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class FeedbackCohortSelector:
    """Declarative selector for one side of a semantic-feedback split audit."""

    evaluation_splits: frozenset[str | None]
    search_modes: frozenset[str] = frozenset(("semantic", "hybrid"))
    evaluation_intents: frozenset[str | None] | None = None

    def matches(self, row: RelevanceFeedback) -> bool:
        return bool(
            row.evaluation_split in self.evaluation_splits
            and row.search_mode in self.search_modes
            and (
                self.evaluation_intents is None
                or row.evaluation_intent in self.evaluation_intents
            )
        )


DEFAULT_DEVELOPMENT_SELECTOR = FeedbackCohortSelector(
    evaluation_splits=frozenset((None, "train", "validation")),
)
DEFAULT_TEST_SELECTOR = FeedbackCohortSelector(
    evaluation_splits=frozenset(("test",)),
)


@dataclass(frozen=True, slots=True)
class SemanticSplitAudit:
    """Aggregate-only evidence that no development/test identity leaked."""

    passed: bool
    cohort_counts: Mapping[str, int]
    overlap_counts: Mapping[str, int]
    overlap_digests_sha256: Mapping[str, str]
    invalid_counts: Mapping[str, int]
    allowed_literal_argument_query_digest_count: int
    allowed_literal_argument_query_digests_sha256: str
    failure_codes: tuple[str, ...]
    material_sha256: str
    version: str = SEMANTIC_SPLIT_AUDIT_VERSION

    def public_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "passed": self.passed,
            "cohort_counts": dict(self.cohort_counts),
            "overlap_counts": dict(self.overlap_counts),
            "overlap_digests_sha256": dict(self.overlap_digests_sha256),
            "invalid_counts": dict(self.invalid_counts),
            "allowed_literal_argument_query_digest_count": (
                self.allowed_literal_argument_query_digest_count
            ),
            "allowed_literal_argument_query_digests_sha256": (
                self.allowed_literal_argument_query_digests_sha256
            ),
            "failure_codes": list(self.failure_codes),
            "material_sha256": self.material_sha256,
            "private_query_text_persisted": False,
            "private_result_identity_persisted": False,
            "private_source_identity_persisted": False,
        }


@dataclass(frozen=True, slots=True)
class _FeedbackRow:
    private_row_id: str
    private_query_sha256: str
    private_query_text: str
    private_result_id: str
    private_window_id: str | None
    private_label: str
    private_search_mode: str
    private_filter_scope_sha256: str
    private_evaluation_split: str | None
    private_evaluation_intent: str | None

    def private_material(self) -> tuple[str | None, ...]:
        return (
            self.private_row_id,
            self.private_query_sha256,
            self.private_result_id,
            self.private_window_id,
            self.private_label,
            self.private_search_mode,
            self.private_filter_scope_sha256,
            self.private_evaluation_split,
            self.private_evaluation_intent,
        )


@dataclass(frozen=True, slots=True)
class _WindowIdentity:
    private_window_id: str
    private_source_message_ids: frozenset[str]
    private_conversation_ids: frozenset[str]
    private_non_conversation_source_roots: frozenset[str]


@dataclass(slots=True)
class _CohortMaterial:
    rows: list[_FeedbackRow]
    private_row_ids: set[str]
    private_query_digests: set[str]
    private_result_ids: set[str]
    private_window_ids: set[str]
    private_source_message_ids: set[str]
    private_conversation_ids: set[str]
    private_non_conversation_source_roots: set[str]
    private_by_query: dict[str, _CohortMaterial]

    @classmethod
    def empty(cls) -> _CohortMaterial:
        return cls([], set(), set(), set(), set(), set(), set(), set(), {})

    def add_row(
        self,
        row: _FeedbackRow,
        window: _WindowIdentity | None,
        *,
        index_by_query: bool = True,
    ) -> None:
        self.rows.append(row)
        self.private_row_ids.add(row.private_row_id)
        self.private_query_digests.add(row.private_query_sha256)
        self.private_result_ids.add(row.private_result_id)
        if window is not None:
            self.private_window_ids.add(window.private_window_id)
            self.private_source_message_ids.update(window.private_source_message_ids)
            self.private_conversation_ids.update(window.private_conversation_ids)
            self.private_non_conversation_source_roots.update(
                window.private_non_conversation_source_roots
            )
        if index_by_query:
            query_material = self.private_by_query.setdefault(
                row.private_query_sha256,
                _CohortMaterial.empty(),
            )
            query_material.add_row(row, window, index_by_query=False)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8", errors="surrogatepass")


def _update_framed(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _aggregate_digest(kind: str, values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    digest.update(SEMANTIC_SPLIT_AUDIT_VERSION.encode("ascii"))
    _update_framed(digest, kind.encode("ascii"))
    for value in sorted(set(values)):
        _update_framed(digest, value.encode("utf-8", errors="surrogatepass"))
    return digest.hexdigest()


def _batched(values: Sequence[str]) -> Iterable[Sequence[str]]:
    for start in range(0, len(values), _SQL_BATCH_SIZE):
        yield values[start : start + _SQL_BATCH_SIZE]


def _source_id_values(value: object, key: str) -> set[str]:
    if not isinstance(value, dict):
        return set()
    selected = value.get(key)
    if isinstance(selected, str) and selected:
        return {selected}
    if isinstance(selected, list):
        return {item for item in selected if isinstance(item, str) and item}
    return set()


def _source_ids_well_formed(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    for key in _SOURCE_ID_KEYS & value.keys():
        selected = value[key]
        if isinstance(selected, str) and selected:
            continue
        if (
            isinstance(selected, list)
            and selected
            and all(isinstance(item, str) and item for item in selected)
        ):
            continue
        return False
    return True


def _private_source_messages(window: SearchWindow, associated: Iterable[str]) -> frozenset[str]:
    identifiers = set(associated)
    identifiers.update(_source_id_values(window.source_ids_json, "message_ids"))
    identifiers.update(_source_id_values(window.source_ids_json, "message_id"))
    if window.first_message_id:
        identifiers.add(str(window.first_message_id))
    if window.last_message_id:
        identifiers.add(str(window.last_message_id))
    return frozenset(identifiers)


def _private_non_conversation_roots(window: SearchWindow) -> frozenset[str]:
    roots: set[str] = set()

    activity_ids = _source_id_values(window.source_ids_json, "activity_id")
    if window.activity_id:
        activity_ids.add(str(window.activity_id))
    roots.update(f"activity:{identifier}" for identifier in activity_ids)
    if roots:
        return frozenset(roots)

    occurrence_ids = _source_id_values(window.source_ids_json, "attachment_occurrence_id")
    if window.attachment_occurrence_id:
        occurrence_ids.add(str(window.attachment_occurrence_id))
    roots.update(f"attachment_occurrence:{identifier}" for identifier in occurrence_ids)
    if roots:
        return frozenset(roots)

    audio_segment_ids = _source_id_values(window.source_ids_json, "audio_segment_id")
    if window.audio_segment_id:
        audio_segment_ids.add(str(window.audio_segment_id))
    roots.update(f"audio_segment:{identifier}" for identifier in audio_segment_ids)
    if roots:
        return frozenset(roots)

    extraction_ids = _source_id_values(window.source_ids_json, "attachment_extraction_id")
    if window.attachment_extraction_id:
        extraction_ids.add(str(window.attachment_extraction_id))
    roots.update(f"attachment_extraction:{identifier}" for identifier in extraction_ids)
    return frozenset(roots)


def _load_window_identities(
    session: Session,
    private_window_ids: Sequence[str],
) -> tuple[dict[str, _WindowIdentity], dict[str, int]]:
    windows: dict[str, SearchWindow] = {}
    associated_messages: dict[str, set[str]] = {}
    for batch in _batched(private_window_ids):
        windows.update(
            {
                window.id: window
                for window in session.scalars(
                    select(SearchWindow).where(SearchWindow.id.in_(batch))
                )
            }
        )
        for window_id, message_id in session.execute(
            select(SearchWindowMessage.window_id, SearchWindowMessage.message_id).where(
                SearchWindowMessage.window_id.in_(batch)
            )
        ):
            associated_messages.setdefault(str(window_id), set()).add(str(message_id))

    invalid_counts = {
        "missing_window_target": 0,
        "malformed_source_ids": 0,
        "missing_source_identity": 0,
        "ambiguous_source_root": 0,
    }
    resolved: dict[str, _WindowIdentity] = {}
    for private_window_id in private_window_ids:
        window = windows.get(private_window_id)
        if window is None:
            invalid_counts["missing_window_target"] += 1
            continue
        if not _source_ids_well_formed(window.source_ids_json):
            invalid_counts["malformed_source_ids"] += 1
        source_messages = _private_source_messages(
            window,
            associated_messages.get(private_window_id, ()),
        )
        conversation_ids = (
            frozenset((str(window.conversation_id),)) if window.conversation_id else frozenset()
        )
        source_roots = (
            frozenset() if conversation_ids else _private_non_conversation_roots(window)
        )
        if len(source_roots) > 1:
            invalid_counts["ambiguous_source_root"] += 1
        if not conversation_ids and not source_roots:
            invalid_counts["missing_source_identity"] += 1
        if window.entity_type == "message_window" and not source_messages:
            invalid_counts["missing_source_identity"] += 1
        resolved[private_window_id] = _WindowIdentity(
            private_window_id=private_window_id,
            private_source_message_ids=source_messages,
            private_conversation_ids=conversation_ids,
            private_non_conversation_source_roots=source_roots,
        )
    return resolved, invalid_counts


def _selected_rows(
    session: Session,
    development: FeedbackCohortSelector,
    test: FeedbackCohortSelector,
) -> tuple[list[_FeedbackRow], list[_FeedbackRow], dict[str, int]]:
    selected_development: list[_FeedbackRow] = []
    selected_test: list[_FeedbackRow] = []
    invalid_counts = {
        "selector_overlap_rows": 0,
        "query_digest_mismatch": 0,
        "empty_query_text": 0,
        "malformed_filter_scope": 0,
        "filter_scope_digest_mismatch": 0,
        "malformed_query_digest": 0,
        "malformed_window_target": 0,
        "unsupported_label": 0,
    }
    rows = session.scalars(select(RelevanceFeedback).order_by(RelevanceFeedback.id))
    for row in rows:
        in_development = development.matches(row)
        in_test = test.matches(row)
        if not in_development and not in_test:
            continue
        if in_development and in_test:
            invalid_counts["selector_overlap_rows"] += 1

        query_text = str(row.query_text)
        query_sha256 = str(row.query_sha256)
        if not query_text:
            invalid_counts["empty_query_text"] += 1
        try:
            digest_bytes = bytes.fromhex(query_sha256)
        except ValueError:
            digest_bytes = b""
        if len(digest_bytes) != _QUERY_DIGEST_BYTES:
            invalid_counts["malformed_query_digest"] += 1
        computed_query_sha256 = hashlib.sha256(query_text.encode("utf-8")).hexdigest()
        if query_sha256 != computed_query_sha256:
            invalid_counts["query_digest_mismatch"] += 1

        filter_scope = row.filter_scope_json
        if not isinstance(filter_scope, dict):
            invalid_counts["malformed_filter_scope"] += 1
        try:
            computed_scope_sha256 = hashlib.sha256(_canonical_bytes(filter_scope)).hexdigest()
        except (TypeError, ValueError):
            computed_scope_sha256 = ""
        if str(row.filter_scope_sha256) != computed_scope_sha256:
            invalid_counts["filter_scope_digest_mismatch"] += 1

        result_id = str(row.result_id)
        private_window_id: str | None = None
        if result_id.startswith(_WINDOW_TARGET_PREFIX):
            candidate = result_id.removeprefix(_WINDOW_TARGET_PREFIX)
            if candidate:
                private_window_id = candidate
            else:
                invalid_counts["malformed_window_target"] += 1
        else:
            invalid_counts["malformed_window_target"] += 1
        label = str(row.label)
        if label not in _SUPPORTED_LABELS:
            invalid_counts["unsupported_label"] += 1
        selected = _FeedbackRow(
            private_row_id=str(row.id),
            private_query_sha256=query_sha256,
            private_query_text=query_text,
            private_result_id=result_id,
            private_window_id=private_window_id,
            private_label=label,
            private_search_mode=str(row.search_mode),
            private_filter_scope_sha256=str(row.filter_scope_sha256),
            private_evaluation_split=(
                str(row.evaluation_split) if row.evaluation_split is not None else None
            ),
            private_evaluation_intent=(
                str(row.evaluation_intent) if row.evaluation_intent is not None else None
            ),
        )
        if in_development:
            selected_development.append(selected)
        if in_test:
            selected_test.append(selected)
    return selected_development, selected_test, invalid_counts


def _material_for_rows(
    rows: Sequence[_FeedbackRow],
    windows: Mapping[str, _WindowIdentity],
) -> _CohortMaterial:
    material = _CohortMaterial.empty()
    for row in rows:
        window = windows.get(row.private_window_id) if row.private_window_id else None
        material.add_row(row, window)
    return material


def _intersections(
    development: _CohortMaterial,
    test: _CohortMaterial,
) -> dict[str, set[str]]:
    return {
        "feedback_row": development.private_row_ids & test.private_row_ids,
        "result": development.private_result_ids & test.private_result_ids,
        "window": development.private_window_ids & test.private_window_ids,
        "source_message": (
            development.private_source_message_ids & test.private_source_message_ids
        ),
        "conversation": development.private_conversation_ids & test.private_conversation_ids,
        "non_conversation_source_root": (
            development.private_non_conversation_source_roots
            & test.private_non_conversation_source_roots
        ),
    }


def _is_disjoint(
    development: _CohortMaterial,
    test: _CohortMaterial,
) -> bool:
    return not any(_intersections(development, test).values())


def _query_overlap(
    development: _CohortMaterial,
    test: _CohortMaterial,
) -> tuple[set[str], set[str]]:
    allowed_argument: set[str] = set()
    forbidden: set[str] = set()
    for query_sha256 in development.private_query_digests & test.private_query_digests:
        development_query = development.private_by_query[query_sha256]
        test_query = test.private_by_query[query_sha256]
        all_rows = development_query.rows + test_query.rows
        all_targets_resolved = all(
            row.private_window_id is not None
            and row.private_window_id in query_material.private_window_ids
            for query_material in (development_query, test_query)
            for row in query_material.rows
        )
        if (
            all_rows
            and all(is_direct_conflict_trigger(row.private_query_text) for row in all_rows)
            and all_targets_resolved
            and _is_disjoint(development_query, test_query)
        ):
            allowed_argument.add(query_sha256)
        else:
            forbidden.add(query_sha256)
    return allowed_argument, forbidden


def _cohort_material_digest(kind: str, material: _CohortMaterial) -> str:
    records = sorted(_canonical_bytes(row.private_material()) for row in material.rows)
    digest = hashlib.sha256()
    digest.update(SEMANTIC_SPLIT_AUDIT_VERSION.encode("ascii"))
    _update_framed(digest, kind.encode("ascii"))
    for record in records:
        _update_framed(digest, record)
    identity_sets = {
        "feedback_row": material.private_row_ids,
        "query_digest": material.private_query_digests,
        "result": material.private_result_ids,
        "window": material.private_window_ids,
        "source_message": material.private_source_message_ids,
        "conversation": material.private_conversation_ids,
        "non_conversation_source_root": material.private_non_conversation_source_roots,
    }
    for identity_kind, values in sorted(identity_sets.items()):
        _update_framed(
            digest,
            _aggregate_digest(f"{kind}:{identity_kind}", values).encode("ascii"),
        )
    return digest.hexdigest()


def _audit_material_digest(
    development: _CohortMaterial,
    test: _CohortMaterial,
    *,
    overlap_digests: Mapping[str, str],
    invalid_counts: Mapping[str, int],
) -> str:
    return _aggregate_digest(
        "audit_material",
        (
            _cohort_material_digest("development", development),
            _cohort_material_digest("test", test),
            _aggregate_digest(
                "overlap_digests",
                (
                    f"{key}:{value}"
                    for key, value in sorted(overlap_digests.items())
                ),
            ),
            _aggregate_digest(
                "invalid_counts",
                (f"{key}:{value}" for key, value in sorted(invalid_counts.items())),
            ),
        ),
    )


def audit_semantic_feedback_split(
    engine: Engine,
    *,
    development: FeedbackCohortSelector = DEFAULT_DEVELOPMENT_SELECTOR,
    test: FeedbackCohortSelector = DEFAULT_TEST_SELECTOR,
) -> SemanticSplitAudit:
    """Audit semantic feedback identity isolation without returning private material."""

    if not development.evaluation_splits or not test.evaluation_splits:
        raise SemanticSplitAuditError("semantic_split_selector_invalid")
    if not development.search_modes or not test.search_modes:
        raise SemanticSplitAuditError("semantic_split_selector_invalid")
    try:
        with Session(engine) as session:
            development_rows, test_rows, row_invalid = _selected_rows(
                session,
                development,
                test,
            )
            private_window_ids = sorted(
                {
                    row.private_window_id
                    for row in development_rows + test_rows
                    if row.private_window_id is not None
                }
            )
            windows, window_invalid = _load_window_identities(session, private_window_ids)
    except SemanticSplitAuditError:
        raise
    except Exception:
        raise SemanticSplitAuditError("semantic_split_audit_source_unavailable") from None

    invalid_counts = {
        **row_invalid,
        **window_invalid,
        "empty_development_cohort": int(not development_rows),
        "empty_test_cohort": int(not test_rows),
    }
    development_material = _material_for_rows(development_rows, windows)
    test_material = _material_for_rows(test_rows, windows)
    identity_intersections = _intersections(development_material, test_material)
    allowed_argument, forbidden_queries = _query_overlap(
        development_material,
        test_material,
    )
    intersections = {
        "query_digest": forbidden_queries,
        **identity_intersections,
    }
    overlap_counts = {
        key: len(value)
        for key, value in intersections.items()
    }
    overlap_digests = {
        key: _aggregate_digest(f"overlap:{key}", value)
        for key, value in intersections.items()
    }

    failure_codes: list[str] = []
    if any(invalid_counts.values()):
        failure_codes.append("invalid_or_incomplete_material")
    if forbidden_queries:
        failure_codes.append("query_digest_overlap")
    for key, values in identity_intersections.items():
        if values:
            failure_codes.append(f"{key}_overlap")
    failure_codes = sorted(set(failure_codes))
    passed = not failure_codes
    return SemanticSplitAudit(
        passed=passed,
        cohort_counts={
            "development_rows": len(development_rows),
            "development_query_digests": len(development_material.private_query_digests),
            "development_results": len(development_material.private_result_ids),
            "development_windows": len(development_material.private_window_ids),
            "development_source_messages": len(
                development_material.private_source_message_ids
            ),
            "development_conversations": len(development_material.private_conversation_ids),
            "development_non_conversation_source_roots": len(
                development_material.private_non_conversation_source_roots
            ),
            "test_rows": len(test_rows),
            "test_query_digests": len(test_material.private_query_digests),
            "test_results": len(test_material.private_result_ids),
            "test_windows": len(test_material.private_window_ids),
            "test_source_messages": len(test_material.private_source_message_ids),
            "test_conversations": len(test_material.private_conversation_ids),
            "test_non_conversation_source_roots": len(
                test_material.private_non_conversation_source_roots
            ),
        },
        overlap_counts=overlap_counts,
        overlap_digests_sha256=overlap_digests,
        invalid_counts=invalid_counts,
        allowed_literal_argument_query_digest_count=len(allowed_argument),
        allowed_literal_argument_query_digests_sha256=_aggregate_digest(
            "allowed_literal_argument_query_digest",
            allowed_argument,
        ),
        failure_codes=tuple(failure_codes),
        material_sha256=_audit_material_digest(
            development_material,
            test_material,
            overlap_digests=overlap_digests,
            invalid_counts=invalid_counts,
        ),
    )


def require_semantic_feedback_split_isolation(
    engine: Engine,
    *,
    development: FeedbackCohortSelector = DEFAULT_DEVELOPMENT_SELECTOR,
    test: FeedbackCohortSelector = DEFAULT_TEST_SELECTOR,
) -> SemanticSplitAudit:
    """Return aggregate evidence or reject use of a contaminated feedback split."""

    audit = audit_semantic_feedback_split(
        engine,
        development=development,
        test=test,
    )
    if not audit.passed:
        raise SemanticSplitAuditError(
            "semantic_feedback_split_not_isolated",
            audit=audit,
        )
    return audit
