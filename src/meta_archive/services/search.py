from __future__ import annotations

import hashlib
import math
import re
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import lancedb  # type: ignore[import-untyped]
import numpy as np
from sqlalchemy import Engine, and_, exists, func, or_, select, text
from sqlalchemy.orm import Session, aliased

from meta_archive.config import settings
from meta_archive.models import (
    ActivityRecord,
    AttachmentOccurrence,
    AttachmentTextExtraction,
    AudioTranscription,
    AudioTranscriptSegment,
    Conversation,
    ConversationSegment,
    CoverageArea,
    Message,
    Participant,
    SearchWindow,
    SearchWindowMessage,
    SegmentParticipant,
    SemanticIndexConfig,
)
from meta_archive.paths import ArchiveLayout, UnsafeArchivePathError
from meta_archive.services.direct_conflict_ranker import (
    DIRECT_CONFLICT_RANKER_VERSION,
    DIRECT_CONFLICT_SCORE_KIND,
    DirectConflictRanker,
    DirectConflictRankerError,
    LanceDirectConflictVectorSource,
    derive_direct_conflict_ranker,
    development_feedback_revision_payload,
    is_direct_conflict_trigger,
)
from meta_archive.services.jumps import activity_jump
from meta_archive.services.lance_read_only import require_existing_lance_manifest
from meta_archive.services.reranking import (
    DEFAULT_RERANKER_SPEC,
    MAX_RERANK_CANDIDATES,
    RerankerFactory,
    TextReranker,
    cached_cross_encoder_reranker,
    reranker_status_payload,
)
from meta_archive.services.search_index import (
    FASTEMBED_PROVIDER,
    INJECTED_TEXT_EMBEDDER_PROVIDER,
    INPUT_ORDER_BATCHING,
    MODEL_SPECS,
    PROVIDER_DECLARED_POOLING,
    SEMANTIC_RANKING_PIPELINE_VERSION,
    STABLE_TOKEN_LENGTH_BATCHING,
    TRANSFORMERS_MAX_SEQUENCE_TOKENS,
    TRANSFORMERS_POOLING,
    TRANSFORMERS_PRECISION,
    TRANSFORMERS_PROVIDER,
    EmbeddingModelSpec,
    TextEmbedder,
    default_text_embedder,
    embedding_artifact_fingerprint,
    embedding_implementation_identity,
    iter_window_message_ids,
)
from meta_archive.services.semantic_state import (
    LanceTableInventoryError,
    lance_table_names,
    semantic_window_membership_clause,
)
from meta_archive.services.text import normalize_text

SearchMode = Literal["keyword", "exact", "semantic", "hybrid"]
EntityFilter = Literal["all", "message", "activity", "audio_transcript", "attachment_text"]
EmbedderFactory = Callable[[EmbeddingModelSpec], TextEmbedder]

_FTS_TOKEN = re.compile(r"[\w]+", re.UNICODE)
_FTS_QUERY_PART = re.compile(r'"([^"\r\n]+)"|([\w]+)(\*)?', re.UNICODE)
_MAX_EXCERPT_CHARACTERS = 1_800
_RRF_K = 60
DERIVED_OCCURRENCE_PAGE_SIZE = 20
_DIRECT_CONFLICT_RANKER_CACHE_SIZE = 4
_DIRECT_CONFLICT_RANKER_CACHE: OrderedDict[tuple[str, str, str, str, str], DirectConflictRanker] = (
    OrderedDict()
)
_DIRECT_CONFLICT_RANKER_CACHE_LOCK = threading.Lock()


class SearchUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SearchFilters:
    entity_type: EntityFilter = "all"
    conversation_id: str | None = None
    participant: str | None = None
    source_category: str | None = None
    from_ms: int | None = None
    to_ms: int | None = None


@dataclass(frozen=True, slots=True)
class SearchPage:
    items: list[dict[str, Any]]
    total_candidates: int
    index_config: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class SemanticCandidateRanking:
    rows: list[tuple[str, float, float, float]]
    score_kind: str
    provenance: dict[str, Any]


_EMPTY_FILTERS = SearchFilters()


def query_sha256(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def _excerpt(value: str | None) -> str:
    if not value:
        return ""
    if len(value) <= _MAX_EXCERPT_CHARACTERS:
        return value
    return f"{value[:_MAX_EXCERPT_CHARACTERS].rstrip()} …"


def _match_excerpt(value: str | None, query: str, *, exact: bool) -> str:
    if not value or len(value) <= _MAX_EXCERPT_CHARACTERS:
        return value or ""
    searchable = value if exact else value.casefold()
    needles = [query] if exact else _FTS_TOKEN.findall(normalize_text(query).search)
    positions = [
        position
        for needle in needles
        if (position := searchable.find(needle if exact else needle.casefold())) >= 0
    ]
    if not positions:
        return _excerpt(value)
    start = max(0, min(positions) - _MAX_EXCERPT_CHARACTERS // 3)
    end = min(len(value), start + _MAX_EXCERPT_CHARACTERS)
    return ("… " if start else "") + value[start:end].strip() + (" …" if end < len(value) else "")


def _message_jump(conversation_id: str, message_id: str) -> dict[str, str]:
    return {
        "kind": "conversation",
        "conversation_id": conversation_id,
        "message_id": message_id,
        "fragment": f"message-{message_id}",
        "href": f"/conversations/{conversation_id}?around={message_id}#message-{message_id}",
    }


def _activity_jump(activity_id: str) -> dict[str, str]:
    return {
        "kind": "activity",
        **activity_jump(activity_id),
    }


def _fts_expression(query: str) -> str:
    normalized = normalize_text(query).search
    parts: list[str] = []
    for match in _FTS_QUERY_PART.finditer(normalized):
        phrase, token, prefix = match.groups()
        if phrase is not None:
            phrase_tokens = _FTS_TOKEN.findall(phrase)
            if phrase_tokens:
                parts.append(f'"{" ".join(phrase_tokens)}"')
        elif token is not None:
            parts.append(f'"{token}"{"*" if prefix else ""}')
    if not parts:
        raise ValueError("Search query must contain at least one letter or number")
    # Only our parsed phrase/prefix subset becomes MATCH syntax; operators and punctuation from
    # user text never pass through. Adjacent parts retain FTS5's implicit AND behavior.
    return " ".join(parts)


def _message_filter_sql(filters: SearchFilters, parameters: dict[str, Any]) -> str:
    clauses = ["m.deleted_at IS NULL"]
    if filters.conversation_id:
        clauses.append("m.conversation_id = :conversation_id")
        parameters["conversation_id"] = filters.conversation_id
    if filters.participant:
        clauses.append(
            "EXISTS (SELECT 1 FROM conversation_segments AS ps "
            "JOIN segment_participants AS psp ON psp.segment_id = ps.id "
            "JOIN participants AS pp ON pp.id = psp.participant_id "
            "WHERE ps.conversation_id = m.conversation_id "
            "AND pp.name_search LIKE :participant ESCAPE '\\')"
        )
        parameters["participant"] = _like_value(filters.participant)
    if filters.source_category:
        clauses.append("s.source_category = :source_category")
        parameters["source_category"] = filters.source_category
    if filters.from_ms is not None:
        clauses.append("m.timestamp_ms >= :from_ms")
        parameters["from_ms"] = filters.from_ms
    if filters.to_ms is not None:
        clauses.append("m.timestamp_ms <= :to_ms")
        parameters["to_ms"] = filters.to_ms
    return " AND ".join(clauses)


def _activity_filter_sql(filters: SearchFilters, parameters: dict[str, Any]) -> str:
    clauses = ["a.deleted_at IS NULL"]
    if filters.conversation_id:
        clauses.append("0")
    if filters.participant:
        clauses.append("a.actor_search LIKE :activity_participant ESCAPE '\\'")
        parameters["activity_participant"] = _like_value(filters.participant)
    if filters.source_category:
        clauses.append("a.category = :activity_category")
        parameters["activity_category"] = filters.source_category
    if filters.from_ms is not None:
        clauses.append("a.timestamp_ms >= :activity_from_ms")
        parameters["activity_from_ms"] = filters.from_ms
    if filters.to_ms is not None:
        clauses.append("a.timestamp_ms <= :activity_to_ms")
        parameters["activity_to_ms"] = filters.to_ms
    return " AND ".join(clauses)


def _like_value(value: str) -> str:
    normalized = normalize_text(value).search
    escaped = normalized.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _coverage_key(item: dict[str, Any], known_keys: set[str]) -> str:
    entity_type = str(item.get("entity_type") or "")
    if entity_type in {"message", "message_window"}:
        return "messages"
    if entity_type in {"audio_transcript", "attachment_text"}:
        return "message_attachments"
    category = str(item.get("source_category") or "").strip().casefold()
    category_key = f"activity_{category}" if category else ""
    if category_key in known_keys:
        return category_key
    if category == "posts" and "posts" in known_keys:
        return "posts"
    return category_key or "activity"


def _result_contract(engine: Engine, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach honest coverage, confidence, availability, and provenance to every result.

    Retrieval scores are ranking features and are deliberately never copied into the confidence
    field. The only numeric confidence currently exposed is the uncalibrated local decoder score
    attached to an audio transcript segment.
    """

    if not items:
        return items
    with Session(engine) as session:
        coverage_rows = {row.key: row for row in session.scalars(select(CoverageArea))}
        audio_ids = {
            str(item["source_id"])
            for item in items
            if item.get("entity_type") == "audio_transcript" and item.get("source_id")
        }
        audio_confidence = {
            segment_id: confidence
            for segment_id, confidence in session.execute(
                select(AudioTranscriptSegment.id, AudioTranscriptSegment.confidence).where(
                    AudioTranscriptSegment.id.in_(audio_ids)
                )
            )
        }
        activity_ids = {
            str(item["source_id"])
            for item in items
            if item.get("entity_type") == "activity" and item.get("source_id")
        }
        activity_availability = {
            activity_id: availability
            for activity_id, availability in session.execute(
                select(ActivityRecord.id, ActivityRecord.availability).where(
                    ActivityRecord.id.in_(activity_ids)
                )
            )
        }

    known_keys = set(coverage_rows)
    for item in items:
        key = _coverage_key(item, known_keys)
        coverage = coverage_rows.get(key)
        item["coverage"] = (
            {
                "key": coverage.key,
                "label": coverage.label,
                "status": coverage.status,
                "raw_count": coverage.raw_count,
                "logical_count": coverage.logical_count,
                "description": coverage.description,
                "source_paths": coverage.source_paths_json,
                "absence_is_not_zero_history": coverage.status == "unavailable",
            }
            if coverage is not None
            else {
                "key": key,
                "label": "Observed searchable records",
                "status": "partial",
                "raw_count": None,
                "logical_count": None,
                "description": (
                    "This result is present, but the export has no separate completeness manifest "
                    "for its source category."
                ),
                "source_paths": {},
                "absence_is_not_zero_history": False,
            }
        )
        source_id = str(item.get("source_id") or "")
        if item.get("entity_type") == "audio_transcript":
            item["confidence"] = audio_confidence.get(source_id)
            item["confidence_kind"] = "audio_decoder_score"
            item["confidence_calibrated"] = False
        else:
            item["confidence"] = None
            item["confidence_kind"] = "not_applicable"
            item["confidence_calibrated"] = False
        item["availability"] = (
            activity_availability.get(source_id, "available")
            if item.get("entity_type") == "activity"
            else "available"
        )
        item["deletion_state"] = "retained"
        item["provenance"] = {
            "entity_type": item.get("entity_type"),
            "source_id": item.get("source_id"),
            "window_id": item.get("window_id"),
            "source_category": item.get("source_category"),
            "representation": (
                "derived_text"
                if item.get("entity_type") in {"audio_transcript", "attachment_text"}
                else "semantic_window"
                if item.get("window_id")
                else "source_record"
            ),
        }
    return items


def _message_item(
    row: Any, *, rank: int, score: float, mode: str, query: str, exact: bool
) -> dict[str, Any]:
    content = "\n".join(
        value
        for value in (
            row.content_display,
            f"[Audio transcript] {row.transcription_display}"
            if row.transcription_display
            else None,
        )
        if value
    )
    return {
        "id": f"message:{row.id}",
        "entity_type": "message",
        "source_id": row.id,
        "rank": rank,
        "score": score,
        "score_kind": mode,
        "timestamp_ms": row.timestamp_ms,
        "excerpt": _match_excerpt(content, query, exact=exact),
        "sender": row.sender_display,
        "conversation": {"id": row.conversation_id, "title": row.conversation_title},
        "source_category": row.source_category,
        "window": None,
        "jump": _message_jump(row.conversation_id, row.id),
    }


def _activity_item(
    row: Any, *, rank: int, score: float, mode: str, query: str, exact: bool
) -> dict[str, Any]:
    content = "\n".join(
        value
        for value in (row.title_display, row.content_display)
        if isinstance(value, str) and value
    )
    return {
        "id": f"activity:{row.id}",
        "entity_type": "activity",
        "source_id": row.id,
        "rank": rank,
        "score": score,
        "score_kind": mode,
        "timestamp_ms": row.timestamp_ms,
        "excerpt": _match_excerpt(content, query, exact=exact),
        "sender": row.actor_display,
        "conversation": None,
        "source_category": row.category,
        "subtype": row.subtype,
        "window": None,
        "jump": _activity_jump(row.id),
    }


def _derived_occurrence_statement(*, blob_sha256: str, filters: SearchFilters) -> Any:
    source_category = func.coalesce(
        ConversationSegment.source_category,
        ActivityRecord.category,
    )
    statement = (
        select(
            AttachmentOccurrence,
            Conversation.title_display.label("conversation_title"),
            ActivityRecord.id.label("activity_id"),
            ActivityRecord.title_display.label("activity_title"),
            ActivityRecord.category.label("activity_category"),
            ActivityRecord.subtype.label("activity_subtype"),
            source_category.label("source_category"),
        )
        .outerjoin(Message, Message.id == AttachmentOccurrence.message_id)
        .outerjoin(Conversation, Conversation.id == AttachmentOccurrence.conversation_id)
        .outerjoin(ConversationSegment, ConversationSegment.id == Message.segment_id)
        .outerjoin(
            ActivityRecord,
            (ActivityRecord.id == AttachmentOccurrence.activity_record_id)
            & ActivityRecord.deleted_at.is_(None),
        )
        .where(
            AttachmentOccurrence.blob_sha256 == blob_sha256,
            AttachmentOccurrence.deleted_at.is_(None),
            or_(AttachmentOccurrence.message_id.is_(None), Message.deleted_at.is_(None)),
        )
        .order_by(AttachmentOccurrence.timestamp_ms, AttachmentOccurrence.id)
    )
    if filters.conversation_id:
        statement = statement.where(AttachmentOccurrence.conversation_id == filters.conversation_id)
    if filters.source_category:
        statement = statement.where(source_category == filters.source_category)
    if filters.from_ms is not None:
        statement = statement.where(AttachmentOccurrence.timestamp_ms >= filters.from_ms)
    if filters.to_ms is not None:
        statement = statement.where(AttachmentOccurrence.timestamp_ms <= filters.to_ms)
    if filters.participant:
        normalized = _like_value(filters.participant)
        participant_segment = aliased(ConversationSegment)
        participant_conversation = exists(
            select(SegmentParticipant.id)
            .join(Participant, Participant.id == SegmentParticipant.participant_id)
            .join(
                participant_segment,
                participant_segment.id == SegmentParticipant.segment_id,
            )
            .where(
                participant_segment.conversation_id == AttachmentOccurrence.conversation_id,
                Participant.name_search.like(normalized, escape="\\"),
            )
        )
        statement = statement.where(participant_conversation)
    return statement


def _derived_occurrences(
    session: Session,
    *,
    blob_sha256: str,
    filters: SearchFilters,
    start_ms: int | None,
    offset: int = 0,
    limit: int = DERIVED_OCCURRENCE_PAGE_SIZE,
) -> tuple[list[dict[str, Any]], int]:
    statement = _derived_occurrence_statement(blob_sha256=blob_sha256, filters=filters)
    total = int(
        session.scalar(select(func.count()).select_from(statement.order_by(None).subquery())) or 0
    )
    rows = session.execute(statement.offset(offset).limit(limit)).all()
    items: list[dict[str, Any]] = []
    for row in rows:
        occurrence = row[0]
        if occurrence.conversation_id and occurrence.message_id:
            href = (
                f"/conversations/{occurrence.conversation_id}?around={occurrence.message_id}"
                f"&media={occurrence.id}"
                f"{'&t=' + str(start_ms) if start_ms is not None else ''}"
                f"#message-{occurrence.message_id}"
            )
        elif row.activity_id:
            href = activity_jump(row.activity_id)["href"]
        else:
            href = (
                f"/media?occurrence={occurrence.id}"
                f"{'&t=' + str(start_ms) if start_ms is not None else ''}"
            )
        items.append(
            {
                "occurrence_id": occurrence.id,
                "conversation_id": occurrence.conversation_id,
                "conversation_title": row.conversation_title,
                "message_id": occurrence.message_id,
                "activity_id": row.activity_id,
                "activity": (
                    {
                        "id": row.activity_id,
                        "title": row.activity_title,
                        "category": row.activity_category,
                        "subtype": row.activity_subtype,
                    }
                    if row.activity_id
                    else None
                ),
                "timestamp_ms": occurrence.timestamp_ms,
                "source_category": row.source_category,
                "start_ms": start_ms,
                "href": href,
            }
        )
    return items, total


def derived_occurrence_page(
    engine: Engine,
    *,
    entity_type: Literal["audio_transcript", "attachment_text"],
    source_id: str,
    filters: SearchFilters = _EMPTY_FILTERS,
    offset: int = 0,
    limit: int = DERIVED_OCCURRENCE_PAGE_SIZE,
) -> dict[str, Any]:
    bounded_offset = max(0, offset)
    bounded_limit = max(1, min(limit, 100))
    with Session(engine) as session:
        if entity_type == "audio_transcript":
            result = session.execute(
                select(AudioTranscriptSegment, AudioTranscription)
                .join(
                    AudioTranscription,
                    AudioTranscription.id == AudioTranscriptSegment.transcription_id,
                )
                .where(
                    AudioTranscriptSegment.id == source_id,
                    AudioTranscription.status == "completed",
                    AudioTranscription.index_eligible.is_(True),
                )
            ).first()
            if result is None:
                raise SearchUnavailableError("Audio transcript result is no longer available")
            segment, transcription = result
            blob_sha256 = transcription.blob_sha256
            start_ms: int | None = segment.start_ms
        else:
            extraction = session.get(AttachmentTextExtraction, source_id)
            if (
                extraction is None
                or extraction.status != "completed"
                or not extraction.index_eligible
                or not extraction.text_display
            ):
                raise SearchUnavailableError("Attachment text result is no longer available")
            blob_sha256 = extraction.blob_sha256
            start_ms = None
        items, total = _derived_occurrences(
            session,
            blob_sha256=blob_sha256,
            filters=filters,
            start_ms=start_ms,
            offset=bounded_offset,
            limit=bounded_limit,
        )
    next_offset = bounded_offset + len(items) if bounded_offset + len(items) < total else None
    return {
        "items": items,
        "page": {
            "limit": bounded_limit,
            "offset": bounded_offset,
            "returned": len(items),
            "total": total,
            "next_offset": next_offset,
        },
    }


def _derived_occurrence_scope_sql(
    filters: SearchFilters,
    parameters: dict[str, Any],
    *,
    blob_expression: str,
) -> str:
    clauses = [
        f"occurrence.blob_sha256 = {blob_expression}",
        "occurrence.deleted_at IS NULL",
        "(occurrence.message_id IS NULL OR occurrence_message.deleted_at IS NULL)",
    ]
    if filters.conversation_id:
        clauses.append("occurrence.conversation_id = :derived_conversation_id")
        parameters["derived_conversation_id"] = filters.conversation_id
    if filters.source_category:
        clauses.append(
            "coalesce(occurrence_segment.source_category, occurrence_activity.category) "
            "= :derived_source_category"
        )
        parameters["derived_source_category"] = filters.source_category
    if filters.from_ms is not None:
        clauses.append("occurrence.timestamp_ms >= :derived_from_ms")
        parameters["derived_from_ms"] = filters.from_ms
    if filters.to_ms is not None:
        clauses.append("occurrence.timestamp_ms <= :derived_to_ms")
        parameters["derived_to_ms"] = filters.to_ms
    if filters.participant:
        clauses.append(
            "EXISTS (SELECT 1 FROM conversation_segments AS participant_segment "
            "JOIN segment_participants AS participant_link "
            "ON participant_link.segment_id = participant_segment.id "
            "JOIN participants AS participant "
            "ON participant.id = participant_link.participant_id "
            "WHERE participant_segment.conversation_id = occurrence.conversation_id "
            "AND participant.name_search LIKE :derived_participant ESCAPE '\\')"
        )
        parameters["derived_participant"] = _like_value(filters.participant)
    return (
        "EXISTS (SELECT 1 FROM attachment_occurrences AS occurrence "
        "LEFT JOIN messages AS occurrence_message "
        "ON occurrence_message.id = occurrence.message_id "
        "LEFT JOIN conversation_segments AS occurrence_segment "
        "ON occurrence_segment.id = occurrence_message.segment_id "
        "LEFT JOIN activity_records AS occurrence_activity "
        "ON occurrence_activity.id = occurrence.activity_record_id "
        "AND occurrence_activity.deleted_at IS NULL "
        f"WHERE {' AND '.join(clauses)})"
    )


def _derived_lexical_search(
    engine: Engine,
    query: str,
    *,
    filters: SearchFilters,
    exact: bool,
    limit: int,
) -> list[dict[str, Any]]:
    if filters.entity_type in {"message", "activity"}:
        return []
    allowed_kinds = (
        {"audio_segment"}
        if filters.entity_type == "audio_transcript"
        else {"attachment_text"}
        if filters.entity_type == "attachment_text"
        else {"audio_segment", "attachment_text"}
    )
    parameters: dict[str, Any] = {"limit": max(1, min(limit, 2_000))}
    branches: list[str] = []
    if "audio_segment" in allowed_kinds:
        audio_scope = _derived_occurrence_scope_sql(
            filters,
            parameters,
            blob_expression="transcription.blob_sha256",
        )
        branches.append(
            "SELECT 'audio_segment' AS entity_kind, segment.id AS source_id, "
            "segment.text_display AS search_text "
            "FROM audio_transcript_segments AS segment "
            "JOIN audio_transcriptions AS transcription "
            "ON transcription.id = segment.transcription_id "
            "WHERE transcription.status = 'completed' "
            "AND transcription.index_eligible = 1 "
            f"AND {audio_scope}"
        )
    if "attachment_text" in allowed_kinds:
        attachment_scope = _derived_occurrence_scope_sql(
            filters,
            parameters,
            blob_expression="extraction.blob_sha256",
        )
        branches.append(
            "SELECT 'attachment_text' AS entity_kind, extraction.id AS source_id, "
            "extraction.text_display AS search_text "
            "FROM attachment_text_extractions AS extraction "
            "WHERE extraction.status = 'completed' "
            "AND extraction.index_eligible = 1 "
            f"AND {attachment_scope}"
        )
    with Session(engine) as session:
        if exact:
            parameters["literal_query"] = query
            statement = text(
                "SELECT entity_kind, source_id, 0.0 AS lexical_score "
                f"FROM ({' UNION ALL '.join(branches)}) AS scoped_entities "
                "WHERE instr(search_text, :literal_query) > 0 "
                "ORDER BY entity_kind, source_id LIMIT :limit"
            )
        else:
            parameters["fts_query"] = _fts_expression(query)
            statement = text(
                f"WITH scoped_entities AS ({' UNION ALL '.join(branches)}) "
                "SELECT derived_text_fts.entity_kind, derived_text_fts.source_id, "
                "bm25(derived_text_fts) AS lexical_score "
                "FROM derived_text_fts JOIN scoped_entities "
                "ON scoped_entities.entity_kind = derived_text_fts.entity_kind "
                "AND scoped_entities.source_id = derived_text_fts.source_id "
                "WHERE derived_text_fts MATCH :fts_query "
                "ORDER BY lexical_score, derived_text_fts.source_id LIMIT :limit"
            )
        raw_rows = [
            (str(row.entity_kind), str(row.source_id), float(row.lexical_score))
            for row in session.execute(statement, parameters)
        ]
        items: list[dict[str, Any]] = []
        for kind, source_id, raw_score in raw_rows:
            if kind == "audio_segment":
                result = session.execute(
                    select(AudioTranscriptSegment, AudioTranscription)
                    .join(
                        AudioTranscription,
                        AudioTranscription.id == AudioTranscriptSegment.transcription_id,
                    )
                    .where(
                        AudioTranscriptSegment.id == source_id,
                        AudioTranscription.status == "completed",
                        AudioTranscription.index_eligible.is_(True),
                    )
                ).first()
                if result is None:
                    continue
                segment, transcription = result
                content = segment.text_display
                start_ms = segment.start_ms
                blob_sha256 = transcription.blob_sha256
                entity_type = "audio_transcript"
                timing: dict[str, int] | None = {
                    "start_ms": segment.start_ms,
                    "end_ms": segment.end_ms,
                }
                confidence = segment.confidence
            else:
                extraction = session.get(AttachmentTextExtraction, source_id)
                if (
                    extraction is None
                    or extraction.status != "completed"
                    or not extraction.index_eligible
                    or not extraction.text_display
                ):
                    continue
                content = extraction.text_display
                start_ms = None
                blob_sha256 = extraction.blob_sha256
                entity_type = "attachment_text"
                timing = None
                confidence = None
            occurrences, occurrence_total = _derived_occurrences(
                session,
                blob_sha256=blob_sha256,
                filters=filters,
                start_ms=start_ms,
            )
            if not occurrences:
                continue
            primary = occurrences[0]
            if primary["conversation_id"] and primary["message_id"]:
                primary_jump = _message_jump(
                    str(primary["conversation_id"]),
                    str(primary["message_id"]),
                )
                primary_jump.update(
                    occurrence_id=primary["occurrence_id"],
                    href=primary["href"],
                )
            elif primary["activity_id"]:
                primary_jump = _activity_jump(str(primary["activity_id"]))
                primary_jump["occurrence_id"] = primary["occurrence_id"]
            else:
                primary_jump = {
                    "kind": "media",
                    "href": primary["href"],
                    "occurrence_id": primary["occurrence_id"],
                    "fragment": f"media-{primary['occurrence_id']}",
                }
            items.append(
                {
                    "id": f"{entity_type}:{source_id}",
                    "entity_type": entity_type,
                    "source_id": source_id,
                    "rank": len(items) + 1,
                    "score": 1.0 if exact else -raw_score,
                    "score_kind": "exact" if exact else "fts5",
                    "timestamp_ms": primary["timestamp_ms"],
                    "excerpt": _match_excerpt(content, query, exact=exact),
                    "sender": None,
                    "conversation": {
                        "id": primary["conversation_id"],
                        "title": primary["conversation_title"],
                    }
                    if primary["conversation_id"]
                    else None,
                    "source_category": primary["source_category"],
                    "window": None,
                    "timing": timing,
                    "confidence": confidence,
                    "occurrences": occurrences,
                    "occurrence_page": {
                        "limit": DERIVED_OCCURRENCE_PAGE_SIZE,
                        "offset": 0,
                        "returned": len(occurrences),
                        "total": occurrence_total,
                        "next_offset": (
                            len(occurrences) if len(occurrences) < occurrence_total else None
                        ),
                    },
                    "jump": primary_jump,
                }
            )
            if len(items) >= limit:
                break
    return items


def lexical_search(
    engine: Engine,
    query: str,
    *,
    filters: SearchFilters = _EMPTY_FILTERS,
    exact: bool = False,
    limit: int = 100,
) -> list[dict[str, Any]]:
    if not query:
        raise ValueError("Search query must not be empty")
    parameters: dict[str, Any] = {"limit": max(1, min(limit, 2_000))}
    message_filter = _message_filter_sql(filters, parameters)
    activity_filter = _activity_filter_sql(filters, parameters)
    rows: list[tuple[float, str, Any]] = []
    with Session(engine) as session:
        if filters.entity_type in {"all", "message"}:
            if exact:
                parameters["literal_query"] = query
                statement = text(
                    f"""
                    SELECT m.*, c.title_display AS conversation_title,
                           s.source_category AS source_category, 0.0 AS lexical_score
                    FROM messages AS m
                    JOIN conversations AS c ON c.id = m.conversation_id
                    JOIN conversation_segments AS s ON s.id = m.segment_id
                    WHERE {message_filter}
                      AND (instr(coalesce(m.content_display, ''), :literal_query) > 0
                           OR instr(coalesce(m.transcription_display, ''), :literal_query) > 0
                           OR instr(coalesce(m.sender_display, ''), :literal_query) > 0)
                    ORDER BY m.timestamp_ms DESC, m.id
                    LIMIT :limit
                    """
                )
            else:
                parameters["fts_query"] = _fts_expression(query)
                statement = text(
                    f"""
                    SELECT m.*, c.title_display AS conversation_title,
                           s.source_category AS source_category,
                           bm25(message_fts, 0.35, 1.0, 0.8) AS lexical_score
                    FROM message_fts
                    JOIN messages AS m ON m.pk = message_fts.rowid
                    JOIN conversations AS c ON c.id = m.conversation_id
                    JOIN conversation_segments AS s ON s.id = m.segment_id
                    WHERE message_fts MATCH :fts_query AND {message_filter}
                    ORDER BY lexical_score, m.timestamp_ms DESC, m.id
                    LIMIT :limit
                    """
                )
            for row in session.execute(statement, parameters).mappings():
                rows.append((float(row["lexical_score"]), "message", row))

        if filters.entity_type in {"all", "activity"}:
            if exact:
                parameters["literal_query"] = query
                statement = text(
                    f"""
                    SELECT a.*, 0.0 AS lexical_score
                    FROM activity_records AS a
                    WHERE {activity_filter}
                      AND (instr(coalesce(a.title_display, ''), :literal_query) > 0
                           OR instr(coalesce(a.content_display, ''), :literal_query) > 0
                           OR instr(coalesce(a.actor_display, ''), :literal_query) > 0)
                    ORDER BY a.timestamp_ms DESC, a.id
                    LIMIT :limit
                    """
                )
            else:
                parameters["fts_query"] = _fts_expression(query)
                statement = text(
                    f"""
                    SELECT a.*, bm25(activity_fts, 0.8, 1.0, 0.35) AS lexical_score
                    FROM activity_fts
                    JOIN activity_records AS a ON a.pk = activity_fts.rowid
                    WHERE activity_fts MATCH :fts_query AND {activity_filter}
                    ORDER BY lexical_score, a.timestamp_ms DESC, a.id
                    LIMIT :limit
                    """
                )
            for row in session.execute(statement, parameters).mappings():
                rows.append((float(row["lexical_score"]), "activity", row))

    derived_items = _derived_lexical_search(
        engine,
        query,
        filters=filters,
        exact=exact,
        limit=limit,
    )

    if exact:
        rows.sort(key=lambda item: (-(item[2]["timestamp_ms"] or -1), item[2]["id"]))
    else:
        rows.sort(key=lambda item: (item[0], -(item[2]["timestamp_ms"] or -1), item[2]["id"]))
    items: list[dict[str, Any]] = []
    for rank, (raw_score, entity_type, row) in enumerate(rows[:limit], start=1):
        score = 1.0 if exact else -raw_score
        if entity_type == "message":
            items.append(
                _message_item(
                    row,
                    rank=rank,
                    score=score,
                    mode="exact" if exact else "fts5",
                    query=query,
                    exact=exact,
                )
            )
        else:
            items.append(
                _activity_item(
                    row,
                    rank=rank,
                    score=score,
                    mode="exact" if exact else "fts5",
                    query=query,
                    exact=exact,
                )
            )
    if derived_items:
        items.extend(derived_items)
        items.sort(
            key=lambda item: (
                -float(item["score"]),
                -(int(item["timestamp_ms"]) if item["timestamp_ms"] is not None else -1),
                str(item["id"]),
            )
        )
        for rank, item in enumerate(items[:limit], start=1):
            item["rank"] = rank
        items = items[:limit]
    return _result_contract(engine, items)


def _active_config(session: Session) -> SemanticIndexConfig:
    config = session.scalar(
        select(SemanticIndexConfig).where(
            SemanticIndexConfig.active.is_(True),
            SemanticIndexConfig.status == "ready",
        )
    )
    if config is None:
        raise SearchUnavailableError("Semantic search index is not ready")
    return config


def _spec_for_config(config: SemanticIndexConfig) -> EmbeddingModelSpec:
    for spec in MODEL_SPECS.values():
        if (
            spec.model_name == config.model_name
            and spec.model_revision == config.model_revision
            and spec.dimensions == config.dimensions
            and spec.document_prefix == config.document_prefix
            and spec.query_prefix == config.query_prefix
        ):
            return spec
    implementation = (
        config.audit_json.get("embedding_implementation")
        if isinstance(config.audit_json, dict)
        else None
    )
    implementation = implementation if isinstance(implementation, dict) else {}
    provider = str(implementation.get("provider") or "")
    if provider not in {
        FASTEMBED_PROVIDER,
        INJECTED_TEXT_EMBEDDER_PROVIDER,
        TRANSFORMERS_PROVIDER,
    }:
        provider = (
            TRANSFORMERS_PROVIDER
            if config.backend == "lancedb-transformers"
            else FASTEMBED_PROVIDER
        )
    pooling = str(implementation.get("pooling") or "")
    if not pooling:
        pooling = (
            TRANSFORMERS_POOLING if provider == TRANSFORMERS_PROVIDER else PROVIDER_DECLARED_POOLING
        )
    max_sequence_tokens_value = implementation.get("max_sequence_tokens")
    max_sequence_tokens = (
        int(max_sequence_tokens_value)
        if isinstance(max_sequence_tokens_value, int)
        and not isinstance(max_sequence_tokens_value, bool)
        and max_sequence_tokens_value > 0
        else TRANSFORMERS_MAX_SEQUENCE_TOKENS
        if provider == TRANSFORMERS_PROVIDER
        else None
    )
    precision = str(implementation.get("precision") or "")
    if not precision:
        precision = (
            TRANSFORMERS_PRECISION
            if provider == TRANSFORMERS_PROVIDER
            else "provider-declared"
            if provider == INJECTED_TEXT_EMBEDDER_PROVIDER
            else "provider-declared-onnx"
        )
    batching = implementation.get("document_batching")
    batch_order = INPUT_ORDER_BATCHING
    if isinstance(batching, dict):
        recorded_batch_order = str(batching.get("sort_order") or "")
        if recorded_batch_order == STABLE_TOKEN_LENGTH_BATCHING:
            batch_order = recorded_batch_order
    return EmbeddingModelSpec(
        alias=config.model_name,
        model_name=config.model_name,
        model_revision=config.model_revision,
        dimensions=config.dimensions,
        document_prefix=config.document_prefix,
        query_prefix=config.query_prefix,
        language_scope="Recorded by semantic index audit",
        approximate_download_gb=0,
        provider=provider,
        pooling=pooling,
        max_sequence_tokens=max_sequence_tokens,
        precision=precision,
        batch_order=batch_order,
    )


def _lance_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _lance_filter(filters: SearchFilters) -> str | None:
    clauses: list[str] = []
    if filters.entity_type == "message":
        clauses.append("entity_type = 'message_window'")
    elif filters.entity_type == "activity":
        clauses.append("entity_type = 'activity'")
    elif filters.entity_type == "audio_transcript":
        clauses.append("entity_type = 'audio_transcript'")
    elif filters.entity_type == "attachment_text":
        clauses.append("entity_type = 'attachment_text'")
    if filters.conversation_id:
        clauses.append(f"conversation_id = {_lance_literal(filters.conversation_id)}")
    if filters.source_category:
        clauses.append(f"source_category = {_lance_literal(filters.source_category)}")
    if filters.from_ms is not None:
        clauses.append(f"last_timestamp_ms >= {filters.from_ms}")
    if filters.to_ms is not None:
        clauses.append(f"first_timestamp_ms <= {filters.to_ms}")
    if filters.participant:
        participant = normalize_text(filters.participant).search
        escaped_participant = (
            participant.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        )
        clauses.append(
            f"participants_search LIKE {_lance_literal(f'%{escaped_participant}%')} ESCAPE '\\\\'"
        )
    return " AND ".join(clauses) or None


def _valid_window_statement(
    ids: Sequence[str], filters: SearchFilters, *, config_hash: str | None = None
) -> Any:
    statement = select(SearchWindow).where(SearchWindow.id.in_(ids))
    if config_hash is not None:
        statement = statement.where(semantic_window_membership_clause(config_hash))
    if filters.entity_type == "message":
        statement = statement.where(SearchWindow.entity_type == "message_window")
    elif filters.entity_type == "activity":
        statement = statement.where(SearchWindow.entity_type == "activity")
    elif filters.entity_type == "audio_transcript":
        statement = statement.where(SearchWindow.entity_type == "audio_transcript")
    elif filters.entity_type == "attachment_text":
        statement = statement.where(SearchWindow.entity_type == "attachment_text")
    if filters.conversation_id:
        statement = statement.where(SearchWindow.conversation_id == filters.conversation_id)
    if filters.participant:
        statement = statement.where(
            SearchWindow.participants_search.like(_like_value(filters.participant), escape="\\")
        )
    if filters.source_category:
        statement = statement.where(SearchWindow.source_category == filters.source_category)
    if filters.from_ms is not None:
        statement = statement.where(SearchWindow.last_timestamp_ms >= filters.from_ms)
    if filters.to_ms is not None:
        statement = statement.where(SearchWindow.first_timestamp_ms <= filters.to_ms)
    live_message_count = (
        select(func.count(SearchWindowMessage.id))
        .join(Message, Message.id == SearchWindowMessage.message_id)
        .where(
            SearchWindowMessage.window_id == SearchWindow.id,
            Message.deleted_at.is_(None),
        )
        .correlate(SearchWindow)
        .scalar_subquery()
    )
    live_activity = exists(
        select(ActivityRecord.pk).where(
            ActivityRecord.id == SearchWindow.activity_id,
            ActivityRecord.deleted_at.is_(None),
        )
    )
    derived_message = aliased(Message)
    derived_activity = aliased(ActivityRecord)
    live_audio = exists(
        select(AudioTranscriptSegment.id)
        .join(
            AudioTranscription,
            AudioTranscription.id == AudioTranscriptSegment.transcription_id,
        )
        .join(
            AttachmentOccurrence,
            AttachmentOccurrence.blob_sha256 == AudioTranscription.blob_sha256,
        )
        .outerjoin(derived_message, derived_message.id == AttachmentOccurrence.message_id)
        .outerjoin(
            derived_activity,
            derived_activity.id == AttachmentOccurrence.activity_record_id,
        )
        .where(
            AudioTranscriptSegment.id == SearchWindow.audio_segment_id,
            AttachmentOccurrence.id == SearchWindow.attachment_occurrence_id,
            AudioTranscription.status == "completed",
            AudioTranscription.index_eligible.is_(True),
            AttachmentOccurrence.deleted_at.is_(None),
            or_(AttachmentOccurrence.message_id.is_(None), derived_message.deleted_at.is_(None)),
            or_(
                AttachmentOccurrence.activity_record_id.is_(None),
                and_(
                    derived_activity.id.is_not(None),
                    derived_activity.deleted_at.is_(None),
                ),
            ),
        )
    )
    extraction_message = aliased(Message)
    extraction_activity = aliased(ActivityRecord)
    live_extraction = exists(
        select(AttachmentTextExtraction.id)
        .join(
            AttachmentOccurrence,
            AttachmentOccurrence.blob_sha256 == AttachmentTextExtraction.blob_sha256,
        )
        .outerjoin(extraction_message, extraction_message.id == AttachmentOccurrence.message_id)
        .outerjoin(
            extraction_activity,
            extraction_activity.id == AttachmentOccurrence.activity_record_id,
        )
        .where(
            AttachmentTextExtraction.id == SearchWindow.attachment_extraction_id,
            AttachmentOccurrence.id == SearchWindow.attachment_occurrence_id,
            AttachmentTextExtraction.status == "completed",
            AttachmentTextExtraction.index_eligible.is_(True),
            AttachmentOccurrence.deleted_at.is_(None),
            or_(
                AttachmentOccurrence.message_id.is_(None),
                extraction_message.deleted_at.is_(None),
            ),
            or_(
                AttachmentOccurrence.activity_record_id.is_(None),
                and_(
                    extraction_activity.id.is_not(None),
                    extraction_activity.deleted_at.is_(None),
                ),
            ),
        )
    )
    return statement.where(
        or_(
            (SearchWindow.entity_type == "message_window")
            & (live_message_count == SearchWindow.turn_count),
            (SearchWindow.entity_type == "activity") & live_activity,
            (SearchWindow.entity_type == "audio_transcript") & live_audio,
            (SearchWindow.entity_type == "attachment_text") & live_extraction,
        )
    )


def _window_items(
    session: Session,
    windows_by_id: dict[str, SearchWindow],
    ordered: Iterable[tuple[str, float]],
    *,
    score_kind: str,
    ranking_provenance: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    message_ids = iter_window_message_ids(session, windows_by_id)
    conversation_ids = {
        window.conversation_id for window in windows_by_id.values() if window.conversation_id
    }
    conversation_titles = {
        conversation_id: title
        for conversation_id, title in session.execute(
            select(Conversation.id, Conversation.title_display).where(
                Conversation.id.in_(conversation_ids)
            )
        )
    }
    items: list[dict[str, Any]] = []
    for window_id, score in ordered:
        window = windows_by_id.get(window_id)
        if window is None:
            continue
        ids = message_ids.get(window.id, [])
        first_message_id = window.first_message_id
        jump: dict[str, Any] | None
        if window.attachment_occurrence_id:
            media_start = window.source_ids_json.get("media_start_ms")
            if window.conversation_id and first_message_id:
                jump = _message_jump(window.conversation_id, first_message_id)
                jump["occurrence_id"] = window.attachment_occurrence_id
                jump["href"] = (
                    f"/conversations/{window.conversation_id}?around={first_message_id}"
                    f"&media={window.attachment_occurrence_id}"
                    f"{'&t=' + str(media_start) if media_start is not None else ''}"
                    f"#message-{first_message_id}"
                )
            elif window.activity_id:
                jump = _activity_jump(window.activity_id)
                jump["occurrence_id"] = window.attachment_occurrence_id
            else:
                jump = {
                    "kind": "media",
                    "occurrence_id": window.attachment_occurrence_id,
                    "fragment": f"media-{window.attachment_occurrence_id}",
                    "href": f"/media?occurrence={window.attachment_occurrence_id}"
                    f"{'&t=' + str(media_start) if media_start is not None else ''}",
                }
        else:
            jump = (
                _message_jump(window.conversation_id, first_message_id)
                if window.conversation_id and first_message_id
                else _activity_jump(window.activity_id)
                if window.activity_id
                else None
            )
        source_id = (
            window.audio_segment_id
            or window.attachment_extraction_id
            or window.activity_id
            or window.first_message_id
        )
        items.append(
            {
                "id": f"window:{window.id}",
                "window_id": window.id,
                "entity_type": window.entity_type,
                "source_id": source_id,
                "rank": len(items) + 1,
                "score": score,
                "score_kind": score_kind,
                "ranking_provenance": ranking_provenance,
                "timestamp_ms": window.first_timestamp_ms,
                "excerpt": _excerpt(window.text_display),
                "sender": None,
                "conversation": {
                    "id": window.conversation_id,
                    "title": conversation_titles.get(window.conversation_id),
                }
                if window.conversation_id
                else None,
                "participants": [
                    participant
                    for participant in window.participants_search.splitlines()
                    if participant
                ],
                "source_category": window.source_category,
                "window": {
                    "start_message_id": window.first_message_id,
                    "end_message_id": window.last_message_id,
                    "message_ids": ids,
                    "turn_count": window.turn_count,
                    "first_timestamp_ms": window.first_timestamp_ms,
                    "last_timestamp_ms": window.last_timestamp_ms,
                },
                "timing": {
                    "start_ms": window.source_ids_json.get("media_start_ms"),
                    "end_ms": window.source_ids_json.get("media_end_ms"),
                }
                if window.entity_type == "audio_transcript"
                else None,
                "jump": jump,
            }
        )
    return items


def _rank_semantic_candidates(
    query: str,
    candidates: Sequence[tuple[str, float, Sequence[float] | None]],
    documents_by_id: Mapping[str, str],
    reranker: TextReranker,
    *,
    direct_conflict_ranker: DirectConflictRanker | None = None,
) -> SemanticCandidateRanking:
    """Apply the one shared runtime/evaluator ranking path to an exact-cosine pool."""

    candidate_ids = [window_id for window_id, _cosine, _vector in candidates]
    if len(set(candidate_ids)) != len(candidate_ids) or any(
        window_id not in documents_by_id for window_id in candidate_ids
    ):
        raise SearchUnavailableError("Semantic candidate identities are inconsistent")
    if direct_conflict_ranker is None:
        try:
            cross_encoder_scores = reranker.rerank(
                query,
                [documents_by_id[window_id] for window_id in candidate_ids],
            )
        except SearchUnavailableError:
            raise
        except Exception as exc:
            raise SearchUnavailableError(
                "Semantic reranking is unavailable; the service will not silently return "
                "cosine-only results"
            ) from exc
        if len(cross_encoder_scores) != len(candidates) or any(
            not math.isfinite(score) for score in cross_encoder_scores
        ):
            raise SearchUnavailableError("The local reranker returned invalid ranking scores")
        rows = sorted(
            (
                (window_id, cross_encoder_scores[index], cross_encoder_scores[index], cosine)
                for index, (window_id, cosine, _vector) in enumerate(candidates)
            ),
            key=lambda item: (-item[1], -item[3], item[0]),
        )
        return SemanticCandidateRanking(
            rows=rows,
            score_kind=DEFAULT_RERANKER_SPEC.score_kind,
            provenance={"pipeline": "cosine_retrieval_then_local_cross_encoder"},
        )

    vectors: list[Sequence[float]] = []
    for _window_id, _cosine, vector in candidates:
        if vector is None:
            raise SearchUnavailableError(
                "The direct-conflict ranker cannot read the active semantic vectors"
            )
        vectors.append(vector)
    try:
        margins = direct_conflict_ranker.score(vectors)
    except Exception as exc:
        raise SearchUnavailableError(
            "The literal direct-conflict ranker is unavailable; no baseline fallback was used"
        ) from exc
    if len(margins) != len(candidates) or any(not math.isfinite(score) for score in margins):
        raise SearchUnavailableError("The literal direct-conflict ranker returned invalid scores")
    tied_indices_by_margin: dict[float, list[int]] = {}
    for index, margin in enumerate(margins):
        tied_indices_by_margin.setdefault(margin, []).append(index)
    tied_indices = [
        index
        for indices in tied_indices_by_margin.values()
        if len(indices) > 1
        for index in indices
    ]
    cross_encoder_scores = [0.0] * len(candidates)
    if tied_indices:
        try:
            tie_scores = reranker.rerank(
                query,
                [documents_by_id[candidate_ids[index]] for index in tied_indices],
            )
        except Exception as exc:
            raise SearchUnavailableError(
                "Semantic tie-breaking is unavailable; no baseline fallback was used"
            ) from exc
        if len(tie_scores) != len(tied_indices) or any(
            not math.isfinite(score) for score in tie_scores
        ):
            raise SearchUnavailableError("The local reranker returned invalid tie-break scores")
        for index, score in zip(tied_indices, tie_scores, strict=True):
            cross_encoder_scores[index] = score
    rows = sorted(
        (
            (
                window_id,
                margins[index],
                cross_encoder_scores[index],
                cosine,
            )
            for index, (window_id, cosine, _vector) in enumerate(candidates)
        ),
        key=lambda item: (-item[1], -item[2], -item[3], item[0]),
    )
    return SemanticCandidateRanking(
        rows=rows,
        score_kind=DIRECT_CONFLICT_SCORE_KIND,
        provenance={
            "pipeline": "exact_cosine_then_literal_direct_conflict_linear_svm",
            "ranking_pipeline": SEMANTIC_RANKING_PIPELINE_VERSION,
            "trigger": "exact_normalized_literal",
            "tie_breakers": [
                DEFAULT_RERANKER_SPEC.score_kind,
                "cosine_similarity",
                "stable_window_id",
            ],
            "cross_encoder_tie_candidate_count": len(tied_indices),
            "ranker": direct_conflict_ranker.audit.public_metadata(),
        },
    )


def _accepted_direct_conflict_ranker(
    layout: ArchiveLayout,
    engine: Engine,
    config: SemanticIndexConfig,
) -> DirectConflictRanker:
    calibration = _accepted_direct_conflict_calibration(config)
    if calibration is None:
        raise SearchUnavailableError(
            "The literal direct-conflict ranker lacks an accepted current calibration; "
            "no baseline fallback was used"
        )
    expected_fingerprint = calibration.get("private_material_fingerprint_sha256")
    try:
        current_revision = development_feedback_revision_payload(engine)
        revision_sha256 = current_revision.get("sha256")
        if not isinstance(revision_sha256, str) or not isinstance(expected_fingerprint, str):
            raise DirectConflictRankerError("direct_conflict_calibration_identity_invalid")
        cache_key = (
            str(layout.root.resolve()),
            str(id(engine)),
            str(config.id),
            revision_sha256,
            expected_fingerprint,
        )
        with _DIRECT_CONFLICT_RANKER_CACHE_LOCK:
            cached_ranker = _DIRECT_CONFLICT_RANKER_CACHE.get(cache_key)
            if cached_ranker is not None:
                _DIRECT_CONFLICT_RANKER_CACHE.move_to_end(cache_key)
        ranker = (
            cached_ranker
            if cached_ranker is not None
            else derive_direct_conflict_ranker(
                engine,
                LanceDirectConflictVectorSource(layout),
                config_id=config.id,
            )
        )
    except DirectConflictRankerError as exc:
        raise SearchUnavailableError(
            "The literal direct-conflict ranker is unavailable; no baseline fallback was used"
        ) from exc
    if (
        calibration.get("feedback_revision") != current_revision
        or ranker.audit.private_feedback_revision.private_payload() != current_revision
        or expected_fingerprint != ranker.audit.private_material_fingerprint_sha256
        or ranker.audit.version != DIRECT_CONFLICT_RANKER_VERSION
    ):
        raise SearchUnavailableError(
            "The literal direct-conflict calibration is stale; rebuild the semantic index"
        )
    if cached_ranker is None:
        with _DIRECT_CONFLICT_RANKER_CACHE_LOCK:
            _DIRECT_CONFLICT_RANKER_CACHE[cache_key] = ranker
            _DIRECT_CONFLICT_RANKER_CACHE.move_to_end(cache_key)
            while len(_DIRECT_CONFLICT_RANKER_CACHE) > _DIRECT_CONFLICT_RANKER_CACHE_SIZE:
                _DIRECT_CONFLICT_RANKER_CACHE.popitem(last=False)
    return ranker


def _accepted_direct_conflict_calibration(
    config: SemanticIndexConfig,
) -> dict[str, Any] | None:
    accepted = config.audit_json.get("accepted_semantic_ir_evaluation")
    calibration = (
        accepted.get("direct_conflict_development_calibration")
        if isinstance(accepted, dict)
        else None
    )
    if not isinstance(calibration, dict) or not (
        calibration.get("status") == "passed"
        and calibration.get("passed") is True
        and calibration.get("required_for_production_acceptance") is True
        and calibration.get("evidence_kind") == "conversation_grouped_development_calibration"
        and calibration.get("held_out") is False
        and calibration.get("ranking_pipeline") == SEMANTIC_RANKING_PIPELINE_VERSION
        and calibration.get("feedback_revision_stable_during_evaluation") is True
    ):
        return None
    return cast(dict[str, Any], calibration)


def _manual_direct_conflict_fallback(
    config: SemanticIndexConfig,
) -> dict[str, Any] | None:
    override = config.audit_json.get("manual_activation_override")
    if not isinstance(override, dict) or not (
        override.get("version") == "semantic-manual-activation-v1"
        and override.get("candidate_config_id") == config.id
        and override.get("acknowledged_failed_quality_gate") is True
        and override.get("source") == "cli_only"
    ):
        return None
    return {
        "status": "normal_cross_encoder_fallback",
        "specialized_ranker_used": False,
        "reason": "manual_override_without_accepted_direct_conflict_calibration",
        "trigger": "exact_normalized_literal",
        "pipeline": "cosine_retrieval_then_local_cross_encoder",
        "manual_activation_version": override["version"],
        "manual_activation_job_id": override.get("activation_job_id"),
    }


def direct_conflict_feedback_revision_required(config: SemanticIndexConfig) -> bool:
    """Return whether a literal trigger will use the accepted specialized ranker."""

    return bool(
        _accepted_direct_conflict_calibration(config) is not None
        or _manual_direct_conflict_fallback(config) is None
    )


def semantic_search(
    layout: ArchiveLayout,
    engine: Engine,
    query: str,
    *,
    filters: SearchFilters = _EMPTY_FILTERS,
    limit: int = 100,
    embedder_factory: EmbedderFactory | None = None,
    reranker_factory: RerankerFactory | None = None,
    acceptance_read_only: bool = False,
    literal_direct_conflict_ranking: bool = True,
) -> SearchPage:
    with Session(engine) as session:
        config = _active_config(session)
        spec = _spec_for_config(config)
        config_payload = {
            "id": config.id,
            "model_name": config.model_name,
            "model_revision": config.model_revision,
            "window_version": config.window_version,
            "artifact_fingerprint": config.audit_json.get("artifact_fingerprint"),
            "embedding_implementation": config.audit_json.get("embedding_implementation"),
            "manual_activation_override": config.audit_json.get("manual_activation_override"),
            "reranker": reranker_status_payload(),
        }
        table_name = config.vector_table

    embedder = (
        embedder_factory(spec)
        if embedder_factory is not None
        else default_text_embedder(layout, spec)
    )
    if config_payload["embedding_implementation"] != embedding_implementation_identity(spec):
        raise SearchUnavailableError(
            "Embedding implementation no longer matches the active index; rebuild it"
        )
    expected_fingerprint = config_payload["artifact_fingerprint"]
    actual_fingerprint = embedding_artifact_fingerprint(embedder)
    if expected_fingerprint and actual_fingerprint != expected_fingerprint:
        raise SearchUnavailableError(
            "Embedding model artifact no longer matches the active index; rebuild it"
        )
    vector = embedder.embed_query(query)
    vector_values = np.asarray(vector, dtype=np.float32)
    if (
        vector_values.shape != (spec.dimensions,)
        or not bool(np.isfinite(vector_values).all())
        or not math.isclose(float(np.linalg.norm(vector_values)), 1.0, rel_tol=1e-3, abs_tol=1e-3)
    ):
        raise SearchUnavailableError("Embedding backend violated the active index vector contract")
    if acceptance_read_only:
        try:
            require_existing_lance_manifest(layout)
        except UnsafeArchivePathError as exc:
            raise SearchUnavailableError(
                "Semantic vector storage changed or became unsafe during the acceptance run"
            ) from exc
    database = lancedb.connect(str(layout.vectors))
    try:
        table_names = lance_table_names(database)
    except LanceTableInventoryError as exc:
        raise SearchUnavailableError(
            "Semantic vector table inventory is incomplete; rebuild the index"
        ) from exc
    if table_name not in table_names:
        raise SearchUnavailableError("Semantic vector table is missing; rebuild the index")
    direct_conflict_triggered = literal_direct_conflict_ranking and is_direct_conflict_trigger(
        query
    )
    candidate_limit = (
        MAX_RERANK_CANDIDATES
        if direct_conflict_triggered
        else min(max(limit * settings.semantic_candidate_multiplier, 100), MAX_RERANK_CANDIDATES)
    )
    direct_conflict_fallback: dict[str, Any] | None = None
    direct_conflict_ranker: DirectConflictRanker | None = None
    if direct_conflict_triggered:
        if _accepted_direct_conflict_calibration(config) is not None:
            direct_conflict_ranker = _accepted_direct_conflict_ranker(
                layout,
                engine,
                config,
            )
        else:
            direct_conflict_fallback = _manual_direct_conflict_fallback(config)
            if direct_conflict_fallback is None:
                # Preserve the ordinary fail-closed contract. Only an explicit, audited CLI
                # override may use the normal cross-encoder for this special literal trigger.
                _accepted_direct_conflict_ranker(layout, engine, config)
    vector_query = (
        database.open_table(table_name)
        .search(vector, vector_column_name="vector")
        .distance_type("cosine")
        .bypass_vector_index()
    )
    if lance_filter := _lance_filter(filters):
        vector_query = vector_query.where(lance_filter)
    limited_query = vector_query.limit(candidate_limit)
    candidates = (
        limited_query.select(["id", "vector", "_distance"]).to_list()
        if direct_conflict_triggered
        else limited_query.to_list()
    )
    ordered_ids = [str(candidate["id"]) for candidate in candidates]
    if not ordered_ids:
        if direct_conflict_fallback is not None:
            config_payload["direct_conflict_ranker"] = direct_conflict_fallback
            config_payload["ranking_pipeline"] = SEMANTIC_RANKING_PIPELINE_VERSION
        return SearchPage(items=[], total_candidates=0, index_config=config_payload)
    distances = {
        str(candidate["id"]): float(candidate.get("_distance", 1.0)) for candidate in candidates
    }
    with Session(engine) as session:
        windows = list(
            session.scalars(
                _valid_window_statement(ordered_ids, filters, config_hash=str(config_payload["id"]))
            )
        )
        windows_by_id = {window.id: window for window in windows}
        vectors = {str(candidate["id"]): candidate.get("vector") for candidate in candidates}
        valid_ordered = [
            (
                window_id,
                max(0.0, 1.0 - distances[window_id]),
                cast(Sequence[float], vectors[window_id])
                if vectors[window_id] is not None
                else None,
            )
            for window_id in ordered_ids
            if window_id in windows_by_id
        ]
        try:
            active_reranker = (
                reranker_factory(DEFAULT_RERANKER_SPEC)
                if reranker_factory is not None
                else cached_cross_encoder_reranker(
                    str(layout.models / "reranker" / DEFAULT_RERANKER_SPEC.model_revision)
                )
            )
            if active_reranker.spec != DEFAULT_RERANKER_SPEC:
                raise SearchUnavailableError(
                    "The configured local reranker does not match the required pinned revision"
                )
            ranking = _rank_semantic_candidates(
                query,
                valid_ordered,
                {
                    window_id: windows_by_id[window_id].text_display
                    for window_id, _cosine, _vector in valid_ordered
                },
                active_reranker,
                direct_conflict_ranker=direct_conflict_ranker,
            )
        except SearchUnavailableError:
            raise
        except Exception as exc:
            raise SearchUnavailableError(
                "Semantic reranking is unavailable; the service will not silently return "
                "cosine-only results"
            ) from exc
        reranker_status = {
            **active_reranker.status,
            "candidate_count": len(ranking.rows),
            "evaluated_candidate_count": ranking.provenance.get(
                "cross_encoder_tie_candidate_count",
                len(ranking.rows),
            ),
            "retrieval_score_kind": "cosine_similarity",
        }
        config_payload["reranker"] = reranker_status
        config_payload["ranking_pipeline"] = SEMANTIC_RANKING_PIPELINE_VERSION
        if direct_conflict_ranker is not None:
            config_payload["direct_conflict_ranker"] = (
                direct_conflict_ranker.audit.public_metadata()
            )
        elif direct_conflict_fallback is not None:
            config_payload["direct_conflict_ranker"] = direct_conflict_fallback
        ranking_provenance = {
            **ranking.provenance,
            **(
                {"literal_direct_conflict": direct_conflict_fallback}
                if direct_conflict_fallback is not None
                else {}
            ),
        }
        items = _window_items(
            session,
            windows_by_id,
            [(window_id, score) for window_id, score, _cross, _cosine in ranking.rows[:limit]],
            score_kind=ranking.score_kind,
            ranking_provenance={**ranking_provenance, "reranker": reranker_status},
        )
    return SearchPage(
        items=_result_contract(engine, items),
        total_candidates=len(ranking.rows),
        index_config=config_payload,
    )


def _lexical_window_ranks(
    session: Session, lexical_items: Sequence[dict[str, Any]]
) -> tuple[list[str], dict[str, list[str]]]:
    message_ids = [
        str(item["source_id"]) for item in lexical_items if item["entity_type"] == "message"
    ]
    activity_ids = [
        str(item["source_id"]) for item in lexical_items if item["entity_type"] == "activity"
    ]
    audio_segment_ids = [
        str(item["source_id"])
        for item in lexical_items
        if item["entity_type"] == "audio_transcript"
    ]
    extraction_ids = [
        str(item["source_id"]) for item in lexical_items if item["entity_type"] == "attachment_text"
    ]
    rows: list[tuple[str, str]] = []
    if message_ids:
        rows = [
            (message_id, window_id)
            for message_id, window_id in session.execute(
                select(SearchWindowMessage.message_id, SearchWindowMessage.window_id)
                .join(SearchWindow, SearchWindow.id == SearchWindowMessage.window_id)
                .where(SearchWindowMessage.message_id.in_(message_ids))
                .order_by(SearchWindowMessage.message_id, SearchWindowMessage.ordinal)
            )
        ]
    activity_windows: dict[str, str] = {}
    if activity_ids:
        activity_windows = {
            activity_id: window_id
            for activity_id, window_id in session.execute(
                select(SearchWindow.activity_id, SearchWindow.id).where(
                    SearchWindow.activity_id.in_(activity_ids)
                )
            )
            if activity_id is not None
        }
    audio_windows: dict[str, list[str]] = {}
    if audio_segment_ids:
        for source_id, window_id in session.execute(
            select(SearchWindow.audio_segment_id, SearchWindow.id).where(
                SearchWindow.audio_segment_id.in_(audio_segment_ids)
            )
        ):
            if source_id is not None:
                audio_windows.setdefault(source_id, []).append(window_id)
    extraction_windows: dict[str, list[str]] = {}
    if extraction_ids:
        for source_id, window_id in session.execute(
            select(SearchWindow.attachment_extraction_id, SearchWindow.id).where(
                SearchWindow.attachment_extraction_id.in_(extraction_ids)
            )
        ):
            if source_id is not None:
                extraction_windows.setdefault(source_id, []).append(window_id)
    by_message: dict[str, list[str]] = {}
    for message_id, window_id in rows:
        by_message.setdefault(message_id, []).append(window_id)
    ranked: list[str] = []
    seen: set[str] = set()
    result_windows: dict[str, list[str]] = {}
    for item in lexical_items:
        if item["entity_type"] == "message":
            candidates = by_message.get(str(item["source_id"]), [])
        elif item["entity_type"] == "activity":
            window = activity_windows.get(str(item["source_id"]))
            candidates = [window] if window else []
        elif item["entity_type"] == "audio_transcript":
            candidates = audio_windows.get(str(item["source_id"]), [])
        else:
            candidates = extraction_windows.get(str(item["source_id"]), [])
        result_windows[str(item["id"])] = candidates
        for window_id in candidates:
            if window_id not in seen:
                ranked.append(window_id)
                seen.add(window_id)
    return ranked, result_windows


def hybrid_search(
    layout: ArchiveLayout,
    engine: Engine,
    query: str,
    *,
    filters: SearchFilters = _EMPTY_FILTERS,
    limit: int = 100,
    embedder_factory: EmbedderFactory | None = None,
    reranker_factory: RerankerFactory | None = None,
    acceptance_read_only: bool = False,
) -> SearchPage:
    candidate_limit = min(max(limit * settings.semantic_candidate_multiplier, 100), 2_000)
    lexical_items = lexical_search(engine, query, filters=filters, limit=candidate_limit)
    semantic = semantic_search(
        layout,
        engine,
        query,
        filters=filters,
        limit=candidate_limit,
        embedder_factory=embedder_factory,
        reranker_factory=reranker_factory,
        acceptance_read_only=acceptance_read_only,
    )
    semantic_ids = [str(item["window_id"]) for item in semantic.items]
    with Session(engine) as session:
        lexical_ids, lexical_result_windows = _lexical_window_ranks(session, lexical_items)
        window_scores: dict[str, float] = {}
        for rank, window_id in enumerate(lexical_ids, start=1):
            window_scores[window_id] = window_scores.get(window_id, 0.0) + 1.0 / (_RRF_K + rank)
        for rank, window_id in enumerate(semantic_ids, start=1):
            window_scores[window_id] = window_scores.get(window_id, 0.0) + 1.0 / (_RRF_K + rank)
        lexical_by_id = {str(item["id"]): item for item in lexical_items}
        config_hash = (
            str(semantic.index_config["id"]) if semantic.index_config is not None else None
        )
        windows = list(
            session.scalars(
                _valid_window_statement(
                    list(window_scores),
                    filters,
                    config_hash=config_hash,
                )
            )
        )
        windows_by_id = {window.id: window for window in windows}
        scores = {
            window_id: score
            for window_id, score in window_scores.items()
            if window_id in windows_by_id
        }
        for rank, item in enumerate(lexical_items, start=1):
            result_id = str(item["id"])
            if not any(
                window_id in windows_by_id
                for window_id in lexical_result_windows.get(result_id, [])
            ):
                scores[f"record:{result_id}"] = 1.0 / (_RRF_K + rank)
        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
        semantic_direct_conflict = (semantic.index_config or {}).get("direct_conflict_ranker")
        hybrid_provenance = {
            "pipeline": "reciprocal_rank_fusion",
            "semantic_reranker": (semantic.index_config or {}).get("reranker"),
            **(
                {"literal_direct_conflict": semantic_direct_conflict}
                if isinstance(semantic_direct_conflict, dict)
                else {}
            ),
        }
        window_items = _window_items(
            session,
            windows_by_id,
            [item for item in ordered if not item[0].startswith("record:")],
            score_kind="rrf",
            ranking_provenance=hybrid_provenance,
        )
        window_payloads = {str(item["window_id"]): item for item in window_items}
        items: list[dict[str, Any]] = []
        for result_key, score in ordered:
            if result_key.startswith("record:"):
                lexical = lexical_by_id.get(result_key.removeprefix("record:"))
                if lexical is None:
                    continue
                payload = dict(lexical)
                lexical_provenance = payload.get("ranking_provenance")
                payload.update(
                    score=score,
                    score_kind="rrf",
                    rank=len(items) + 1,
                    ranking_provenance={
                        **(lexical_provenance if isinstance(lexical_provenance, dict) else {}),
                        **hybrid_provenance,
                    },
                )
            else:
                window_payload = window_payloads.get(result_key)
                if window_payload is None:
                    continue
                payload = dict(window_payload)
                payload.update(rank=len(items) + 1)
            items.append(payload)
    return SearchPage(
        items=_result_contract(engine, items),
        total_candidates=len(scores),
        index_config=semantic.index_config,
    )


def run_search(
    layout: ArchiveLayout,
    engine: Engine,
    query: str,
    *,
    mode: SearchMode,
    filters: SearchFilters = _EMPTY_FILTERS,
    limit: int = 100,
    embedder_factory: EmbedderFactory | None = None,
    reranker_factory: RerankerFactory | None = None,
    acceptance_read_only: bool = False,
) -> SearchPage:
    if mode == "keyword":
        items = lexical_search(engine, query, filters=filters, limit=limit)
        return SearchPage(items=items, total_candidates=len(items))
    if mode == "exact":
        items = lexical_search(engine, query, filters=filters, exact=True, limit=limit)
        return SearchPage(items=items, total_candidates=len(items))
    if mode == "semantic":
        return semantic_search(
            layout,
            engine,
            query,
            filters=filters,
            limit=limit,
            embedder_factory=embedder_factory,
            reranker_factory=reranker_factory,
            acceptance_read_only=acceptance_read_only,
        )
    return hybrid_search(
        layout,
        engine,
        query,
        filters=filters,
        limit=limit,
        embedder_factory=embedder_factory,
        reranker_factory=reranker_factory,
        acceptance_read_only=acceptance_read_only,
    )
