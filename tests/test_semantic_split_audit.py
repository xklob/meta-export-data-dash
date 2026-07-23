from __future__ import annotations

import hashlib
import json

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from meta_archive.models import Conversation, RelevanceFeedback, SearchWindow
from meta_archive.services.semantic_split_audit import (
    DEFAULT_DEVELOPMENT_SELECTOR,
    DEFAULT_TEST_SELECTOR,
    FeedbackCohortSelector,
    SemanticSplitAuditError,
    audit_semantic_feedback_split,
    require_semantic_feedback_split_isolation,
)

_EMPTY_SCOPE_SHA256 = hashlib.sha256(b"{}").hexdigest()


def _add_conversation(session: Session, conversation_id: str) -> None:
    session.add(
        Conversation(
            id=conversation_id,
            conversation_key=f"key-{conversation_id}",
            title_raw=f"title-{conversation_id}",
            title_display=f"title-{conversation_id}",
            title_search=f"title-{conversation_id}",
            conversation_type="direct",
            primary_category="inbox",
        )
    )


def _add_window(
    session: Session,
    window_id: str,
    *,
    conversation_id: str | None = None,
    source_message_ids: tuple[str, ...] = (),
    activity_root: str | None = None,
) -> None:
    source_ids: dict[str, object] = {}
    if source_message_ids:
        source_ids["message_ids"] = list(source_message_ids)
    if activity_root is not None:
        source_ids["activity_id"] = activity_root
    session.add(
        SearchWindow(
            id=window_id,
            window_version="split-audit-fixture-v1",
            entity_type="message_window" if conversation_id else "activity",
            conversation_id=conversation_id,
            first_message_id=source_message_ids[0] if source_message_ids else None,
            last_message_id=source_message_ids[-1] if source_message_ids else None,
            turn_count=max(1, len(source_message_ids)),
            approx_token_count=4,
            text_display=f"private window text {window_id}",
            text_search=f"private window text {window_id}",
            participants_search="private participant",
            source_ids_json=source_ids,
            build_generation="split-audit-fixture-generation",
        )
    )


def _add_feedback(
    session: Session,
    row_id: str,
    *,
    query: str,
    split: str | None,
    window_id: str,
    intent: str | None = None,
    query_sha256: str | None = None,
) -> None:
    session.add(
        RelevanceFeedback(
            id=row_id,
            query_sha256=query_sha256 or hashlib.sha256(query.encode("utf-8")).hexdigest(),
            query_text=query,
            search_mode="semantic",
            filter_scope_sha256=_EMPTY_SCOPE_SHA256,
            filter_scope_json={},
            result_id=f"window:{window_id}",
            label="relevant",
            evaluation_split=split,
            evaluation_intent=intent,
        )
    )


def _seed_disjoint_feedback(engine: Engine) -> None:
    with Session(engine) as session, session.begin():
        _add_conversation(session, "development-conversation")
        _add_conversation(session, "test-conversation")
        _add_window(
            session,
            "development-window",
            conversation_id="development-conversation",
            source_message_ids=("development-message",),
        )
        _add_window(
            session,
            "test-window",
            conversation_id="test-conversation",
            source_message_ids=("test-message",),
        )
        _add_feedback(
            session,
            "development-feedback",
            query="development-only semantic query",
            split="validation",
            window_id="development-window",
        )
        _add_feedback(
            session,
            "test-feedback",
            query="test-only semantic query",
            split="test",
            window_id="test-window",
        )


def test_disjoint_semantic_feedback_split_passes_with_aggregate_only_evidence(
    catalog_engine: Engine,
) -> None:
    _seed_disjoint_feedback(catalog_engine)

    first = audit_semantic_feedback_split(catalog_engine)
    second = audit_semantic_feedback_split(catalog_engine)

    assert first.passed is True
    assert first.failure_codes == ()
    assert first.cohort_counts == {
        "development_rows": 1,
        "development_query_digests": 1,
        "development_results": 1,
        "development_windows": 1,
        "development_source_messages": 1,
        "development_conversations": 1,
        "development_non_conversation_source_roots": 0,
        "test_rows": 1,
        "test_query_digests": 1,
        "test_results": 1,
        "test_windows": 1,
        "test_source_messages": 1,
        "test_conversations": 1,
        "test_non_conversation_source_roots": 0,
    }
    assert all(count == 0 for count in first.overlap_counts.values())
    assert all(count == 0 for count in first.invalid_counts.values())
    assert first.public_payload() == second.public_payload()

    rendered = json.dumps(first.public_payload(), sort_keys=True)
    for private_value in (
        "development-only semantic query",
        "test-only semantic query",
        "development-window",
        "test-window",
        "development-message",
        "test-message",
        "development-conversation",
        "test-conversation",
    ):
        assert private_value not in rendered


def test_non_argument_query_digest_overlap_fails_closed(catalog_engine: Engine) -> None:
    _seed_disjoint_feedback(catalog_engine)
    with Session(catalog_engine) as session, session.begin():
        test_row = session.get(RelevanceFeedback, "test-feedback")
        assert test_row is not None
        test_row.query_text = "development-only semantic query"
        test_row.query_sha256 = hashlib.sha256(test_row.query_text.encode()).hexdigest()

    audit = audit_semantic_feedback_split(catalog_engine)

    assert audit.passed is False
    assert audit.overlap_counts["query_digest"] == 1
    assert audit.failure_codes == ("query_digest_overlap",)
    with pytest.raises(SemanticSplitAuditError) as failure:
        require_semantic_feedback_split_isolation(catalog_engine)
    assert failure.value.code == "semantic_feedback_split_not_isolated"
    assert failure.value.audit == audit
    assert str(failure.value) == "semantic_feedback_split_not_isolated"


def test_literal_argument_query_may_repeat_across_disjoint_cohorts(
    catalog_engine: Engine,
) -> None:
    _seed_disjoint_feedback(catalog_engine)
    argument_sha256 = hashlib.sha256(b"argument").hexdigest()
    with Session(catalog_engine) as session, session.begin():
        for row_id in ("development-feedback", "test-feedback"):
            row = session.get(RelevanceFeedback, row_id)
            assert row is not None
            row.query_text = "argument"
            row.query_sha256 = argument_sha256

    audit = require_semantic_feedback_split_isolation(catalog_engine)

    assert audit.passed is True
    assert audit.overlap_counts["query_digest"] == 0
    assert audit.allowed_literal_argument_query_digest_count == 1
    assert len(audit.allowed_literal_argument_query_digests_sha256) == 64


def test_literal_argument_query_repeat_requires_source_disjointness(
    catalog_engine: Engine,
) -> None:
    argument_sha256 = hashlib.sha256(b"argument").hexdigest()
    with Session(catalog_engine) as session, session.begin():
        _add_conversation(session, "shared-conversation")
        _add_window(
            session,
            "argument-development-window",
            conversation_id="shared-conversation",
            source_message_ids=("argument-development-message",),
        )
        _add_window(
            session,
            "argument-test-window",
            conversation_id="shared-conversation",
            source_message_ids=("argument-test-message",),
        )
        _add_feedback(
            session,
            "argument-development-feedback",
            query="argument",
            split="validation",
            window_id="argument-development-window",
            query_sha256=argument_sha256,
        )
        _add_feedback(
            session,
            "argument-test-feedback",
            query="argument",
            split="test",
            window_id="argument-test-window",
            query_sha256=argument_sha256,
        )

    audit = audit_semantic_feedback_split(catalog_engine)

    assert audit.passed is False
    assert audit.allowed_literal_argument_query_digest_count == 0
    assert audit.overlap_counts["query_digest"] == 1
    assert audit.overlap_counts["conversation"] == 1
    assert audit.failure_codes == ("conversation_overlap", "query_digest_overlap")


def test_target_window_and_result_overlap_are_each_reported(catalog_engine: Engine) -> None:
    with Session(catalog_engine) as session, session.begin():
        _add_conversation(session, "target-conversation")
        _add_window(
            session,
            "shared-target-window",
            conversation_id="target-conversation",
            source_message_ids=("shared-target-message",),
        )
        _add_feedback(
            session,
            "target-development-feedback",
            query="development target query",
            split=None,
            window_id="shared-target-window",
        )
        _add_feedback(
            session,
            "target-test-feedback",
            query="test target query",
            split="test",
            window_id="shared-target-window",
        )

    audit = audit_semantic_feedback_split(catalog_engine)

    assert audit.passed is False
    assert audit.overlap_counts["result"] == 1
    assert audit.overlap_counts["window"] == 1
    assert audit.overlap_counts["source_message"] == 1
    assert audit.overlap_counts["conversation"] == 1


def test_underlying_source_message_overlap_is_detected_across_distinct_windows(
    catalog_engine: Engine,
) -> None:
    with Session(catalog_engine) as session, session.begin():
        _add_conversation(session, "source-development-conversation")
        _add_conversation(session, "source-test-conversation")
        _add_window(
            session,
            "source-development-window",
            conversation_id="source-development-conversation",
            source_message_ids=("shared-source-message",),
        )
        _add_window(
            session,
            "source-test-window",
            conversation_id="source-test-conversation",
            source_message_ids=("shared-source-message",),
        )
        _add_feedback(
            session,
            "source-development-feedback",
            query="source development query",
            split="train",
            window_id="source-development-window",
        )
        _add_feedback(
            session,
            "source-test-feedback",
            query="source test query",
            split="test",
            window_id="source-test-window",
        )

    audit = audit_semantic_feedback_split(catalog_engine)

    assert audit.overlap_counts["source_message"] == 1
    assert audit.overlap_counts["conversation"] == 0
    assert audit.failure_codes == ("source_message_overlap",)


def test_non_conversation_source_root_overlap_is_detected(catalog_engine: Engine) -> None:
    with Session(catalog_engine) as session, session.begin():
        _add_window(
            session,
            "activity-development-window",
            activity_root="private-activity-root",
        )
        _add_window(
            session,
            "activity-test-window",
            activity_root="private-activity-root",
        )
        _add_feedback(
            session,
            "activity-development-feedback",
            query="activity development query",
            split="validation",
            window_id="activity-development-window",
        )
        _add_feedback(
            session,
            "activity-test-feedback",
            query="activity test query",
            split="test",
            window_id="activity-test-window",
        )

    audit = audit_semantic_feedback_split(catalog_engine)

    assert audit.overlap_counts["non_conversation_source_root"] == 1
    assert audit.overlap_counts["conversation"] == 0
    assert audit.failure_codes == ("non_conversation_source_root_overlap",)


def test_missing_target_and_empty_cohort_are_invalid_material(catalog_engine: Engine) -> None:
    with Session(catalog_engine) as session, session.begin():
        _add_feedback(
            session,
            "stale-development-feedback",
            query="stale private query",
            split="validation",
            window_id="missing-private-window",
        )

    audit = audit_semantic_feedback_split(catalog_engine)

    assert audit.passed is False
    assert audit.invalid_counts["missing_window_target"] == 1
    assert audit.invalid_counts["empty_test_cohort"] == 1
    assert audit.failure_codes == ("invalid_or_incomplete_material",)
    assert "stale private query" not in json.dumps(audit.public_payload())
    assert "missing-private-window" not in json.dumps(audit.public_payload())


def test_query_digest_tampering_and_custom_selectors_fail_closed(
    catalog_engine: Engine,
) -> None:
    _seed_disjoint_feedback(catalog_engine)
    with Session(catalog_engine) as session, session.begin():
        row = session.get(RelevanceFeedback, "development-feedback")
        assert row is not None
        row.query_sha256 = "0" * 64

    audit = audit_semantic_feedback_split(
        catalog_engine,
        development=FeedbackCohortSelector(
            evaluation_splits=frozenset(("validation",)),
            search_modes=frozenset(("semantic",)),
        ),
        test=DEFAULT_TEST_SELECTOR,
    )

    assert audit.passed is False
    assert audit.invalid_counts["query_digest_mismatch"] == 1
    assert DEFAULT_DEVELOPMENT_SELECTOR.evaluation_splits == frozenset(
        (None, "train", "validation")
    )
