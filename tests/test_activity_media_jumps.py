from __future__ import annotations

import hashlib
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy import Engine, select, text
from sqlalchemy.orm import Session

from meta_archive.api.main import create_app
from meta_archive.models import (
    ActivityRecord,
    AttachmentOccurrence,
    AttachmentTextExtraction,
    AudioTranscription,
    AudioTranscriptSegment,
    SearchWindow,
)
from meta_archive.paths import ArchiveLayout
from meta_archive.services.activity_importer import import_activities
from meta_archive.services.attachment_text import (
    EXTRACTOR_REVISION,
    extraction_occurrence_jumps,
)
from meta_archive.services.attachment_text import (
    PIPELINE_VERSION as ATTACHMENT_TEXT_PIPELINE_VERSION,
)
from meta_archive.services.inventory import scan_archive
from meta_archive.services.reranking import DEFAULT_RERANKER_SPEC
from meta_archive.services.search import (
    SearchFilters,
    derived_occurrence_page,
    hybrid_search,
    lexical_search,
    semantic_search,
)
from meta_archive.services.search_index import (
    EmbeddingModelSpec,
    build_semantic_index,
)
from meta_archive.services.text import normalize_text
from meta_archive.services.transcription import (
    COMPUTE_TYPE,
    DEVICE,
    MODEL_NAME,
    MODEL_REVISION,
    transcript_occurrence_jumps,
)
from meta_archive.services.transcription import (
    PIPELINE_VERSION as AUDIO_PIPELINE_VERSION,
)

ACTIVITY_FIXTURES = Path(__file__).parent / "fixtures" / "activity"
ATTACHMENT_TEXT = "Activity linked attachment exact needle"
AUDIO_TEXT = "Activity linked audio exact needle"

FIXTURE_EMBEDDING_SPEC = EmbeddingModelSpec(
    alias="activity-media-fixture",
    model_name="fixture/activity-media",
    model_revision="activity-media-v1",
    dimensions=4,
    document_prefix="",
    query_prefix="",
    language_scope="fixture",
    approximate_download_gb=0,
)


class _FakeEmbedder:
    spec = FIXTURE_EMBEDDING_SPEC

    @staticmethod
    def _vector(value: str) -> list[float]:
        digest = hashlib.sha256(value.encode()).digest()
        raw = [float(digest[index] + 1) for index in range(4)]
        norm = sum(item * item for item in raw) ** 0.5
        return [item / norm for item in raw]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(value) for value in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class _FakeReranker:
    spec = DEFAULT_RERANKER_SPEC

    @property
    def status(self) -> dict[str, object]:
        return {
            "status": "loaded_fixture",
            "model_name": self.spec.model_name,
            "model_revision": self.spec.model_revision,
            "device": "cpu",
            "score_kind": self.spec.score_kind,
        }

    def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        query_tokens = set(normalize_text(query).search.split())
        return [
            float(len(query_tokens & set(normalize_text(document).search.split())))
            for document in documents
        ]


def _prepare_activity_media(
    layout: ArchiveLayout,
    engine: Engine,
) -> tuple[AttachmentOccurrence, ActivityRecord, str, str]:
    shutil.copytree(ACTIVITY_FIXTURES, layout.root / "fixture-package")
    scan_archive(layout, engine, hash_files=True)
    import_activities(layout, engine)
    with Session(engine) as session:
        occurrence = session.scalar(
            select(AttachmentOccurrence)
            .where(
                AttachmentOccurrence.activity_record_id.is_not(None),
                AttachmentOccurrence.blob_sha256.is_not(None),
                AttachmentOccurrence.deleted_at.is_(None),
            )
            .order_by(AttachmentOccurrence.id)
            .limit(1)
        )
        assert occurrence is not None
        activity = session.scalar(
            select(ActivityRecord).where(
                ActivityRecord.id == occurrence.activity_record_id,
                ActivityRecord.deleted_at.is_(None),
            )
        )
        assert activity is not None and occurrence.blob_sha256 is not None
        occurrence_id = occurrence.id
        activity_id = activity.id
        blob_sha256 = occurrence.blob_sha256
        activity_category = activity.category

    extraction_id = "activity-attachment-extraction"
    segment_id = "activity-audio-segment"
    with Session(engine) as session, session.begin():
        session.add(
            AttachmentTextExtraction(
                id=extraction_id,
                blob_sha256=blob_sha256,
                pipeline_version=ATTACHMENT_TEXT_PIPELINE_VERSION,
                status="completed",
                extractor_name="bubblewrap-prlimit-local-worker",
                extractor_revision=EXTRACTOR_REVISION,
                filename="activity-note.txt",
                media_type="text/plain",
                text_display=ATTACHMENT_TEXT,
                text_search=normalize_text(ATTACHMENT_TEXT).search,
                character_count=len(ATTACHMENT_TEXT),
                truncated=False,
                index_eligible=True,
                sandboxed=True,
                sandbox_status="available",
                limits_json={},
                provenance_json={"generated_locally": True},
                warnings_json={"items": []},
            )
        )
        transcription_id = "activity-audio-transcription"
        session.add(
            AudioTranscription(
                id=transcription_id,
                blob_sha256=blob_sha256,
                pipeline_version=AUDIO_PIPELINE_VERSION,
                status="completed",
                model_name=MODEL_NAME,
                model_revision=MODEL_REVISION,
                device=DEVICE,
                compute_type=COMPUTE_TYPE,
                text_display=AUDIO_TEXT,
                text_search=normalize_text(AUDIO_TEXT).search,
                segment_count=1,
                index_eligible=True,
                provenance_json={"generated_locally": True},
                warnings_json={"items": []},
            )
        )
        session.flush()
        session.add(
            AudioTranscriptSegment(
                id=segment_id,
                transcription_id=transcription_id,
                ordinal=0,
                start_ms=1_250,
                end_ms=2_500,
                text_display=AUDIO_TEXT,
                text_search=normalize_text(AUDIO_TEXT).search,
                confidence=0.9,
                words_json={},
            )
        )
        session.execute(
            text(
                "INSERT INTO derived_text_fts(entity_kind, source_id, text_search) "
                "VALUES ('attachment_text', :source_id, :text_search)"
            ),
            {"source_id": extraction_id, "text_search": normalize_text(ATTACHMENT_TEXT).search},
        )
        session.execute(
            text(
                "INSERT INTO derived_text_fts(entity_kind, source_id, text_search) "
                "VALUES ('audio_segment', :source_id, :text_search)"
            ),
            {"source_id": segment_id, "text_search": normalize_text(AUDIO_TEXT).search},
        )

    # Detached ORM rows are intentionally not returned; only immutable identifiers are needed.
    with Session(engine) as session:
        occurrence = session.get(AttachmentOccurrence, occurrence_id)
        activity = session.scalar(select(ActivityRecord).where(ActivityRecord.id == activity_id))
        assert occurrence is not None and activity is not None
        assert activity.category == activity_category
        session.expunge(occurrence)
        session.expunge(activity)
    return occurrence, activity, extraction_id, segment_id


def _assert_activity_jump(jump: dict[str, Any], activity_id: str) -> None:
    assert jump["activity_id"] == activity_id
    assert jump["fragment"] == f"activity-{activity_id}"
    assert jump["href"] == (f"/timeline?around=activity:{activity_id}#activity-{activity_id}")


def test_activity_media_apis_expose_context_and_all_source_jumps(
    archive_layout: ArchiveLayout,
    catalog_engine: Engine,
) -> None:
    occurrence, activity, extraction_id, _segment_id = _prepare_activity_media(
        archive_layout,
        catalog_engine,
    )
    expected_context = {
        "id": activity.id,
        "title": activity.title_display,
        "category": activity.category,
        "subtype": activity.subtype,
    }
    app = create_app(archive_layout.root, security_disabled=True)
    with TestClient(app) as client:
        for view in ("occurrences", "unique"):
            response = client.get(
                "/api/v1/media",
                params={"view": view, "occurrence": occurrence.id, "limit": 10},
            )
            assert response.status_code == 200, response.text
            item = response.json()["items"][0]
            assert item["id"] == occurrence.id
            assert item["conversation"] is None
            assert item["activity"] == expected_context
            assert item["source_category"] == activity.category
            assert item["provenance"] is None
            _assert_activity_jump(item["jump"], activity.id)

        assert occurrence.blob_sha256 is not None
        blob_response = client.get(
            f"/api/v1/media/blobs/{occurrence.blob_sha256}/occurrences",
            params={"limit": 100},
        )
        assert blob_response.status_code == 200, blob_response.text
        blob_item = next(
            item for item in blob_response.json()["items"] if item["id"] == occurrence.id
        )
        assert blob_item["activity"] == expected_context
        _assert_activity_jump(blob_item["jump"], activity.id)

        extraction_response = client.get(
            "/api/v1/enrichment/attachment-text",
            params={"version": "history", "limit": 100},
        )
        extraction = next(
            item for item in extraction_response.json()["items"] if item["id"] == extraction_id
        )
        extraction_jump = next(
            jump for jump in extraction["jumps"] if jump["occurrence_id"] == occurrence.id
        )
        assert extraction_jump["activity_id"] == activity.id
        assert extraction_jump["href"] == (
            f"/timeline?around=activity:{activity.id}#activity-{activity.id}"
        )

        transcript_response = client.get(
            "/api/v1/enrichment/transcriptions",
            params={"version": "history", "limit": 100},
        )
        transcript = next(
            item
            for item in transcript_response.json()["items"]
            if item["id"] == "activity-audio-transcription"
        )
        transcript_jump = next(
            jump
            for jump in transcript["segments"][0]["jumps"]
            if jump["occurrence_id"] == occurrence.id
        )
        assert transcript_jump["activity_id"] == activity.id
        assert transcript_jump["start_ms"] == 1_250
        assert transcript_jump["href"] == (
            f"/timeline?around=activity:{activity.id}#activity-{activity.id}"
        )
        assert "t=" not in transcript_jump["href"]

    direct_transcript_jump = next(
        jump
        for jump in transcript_occurrence_jumps(
            catalog_engine,
            occurrence.blob_sha256,
            start_ms=1_250,
        )
        if jump["occurrence_id"] == occurrence.id
    )
    assert direct_transcript_jump["activity_id"] == activity.id
    assert direct_transcript_jump["href"] == (
        f"/timeline?around=activity:{activity.id}#activity-{activity.id}"
    )

    direct_extraction_jump = next(
        jump
        for jump in extraction_occurrence_jumps(catalog_engine, occurrence.blob_sha256)
        if jump["occurrence_id"] == occurrence.id
    )
    assert direct_extraction_jump["activity_id"] == activity.id
    assert direct_extraction_jump["href"] == (
        f"/timeline?around=activity:{activity.id}#activity-{activity.id}"
    )


def test_activity_media_keyword_semantic_and_hybrid_results_jump_to_activity(
    archive_layout: ArchiveLayout,
    catalog_engine: Engine,
) -> None:
    occurrence, activity, extraction_id, segment_id = _prepare_activity_media(
        archive_layout,
        catalog_engine,
    )
    with Session(catalog_engine) as session:
        activity_by_occurrence = {
            occurrence_id: activity_id
            for occurrence_id, activity_id in session.execute(
                select(AttachmentOccurrence.id, ActivityRecord.id)
                .join(
                    ActivityRecord,
                    ActivityRecord.id == AttachmentOccurrence.activity_record_id,
                )
                .where(
                    AttachmentOccurrence.blob_sha256 == occurrence.blob_sha256,
                    AttachmentOccurrence.deleted_at.is_(None),
                    ActivityRecord.deleted_at.is_(None),
                )
            )
        }
    build_semantic_index(
        archive_layout,
        catalog_engine,
        embedder=_FakeEmbedder(),
        batch_size=8,
    )

    with Session(catalog_engine) as session:
        derived_windows = list(
            session.scalars(
                select(SearchWindow).where(
                    SearchWindow.attachment_occurrence_id == occurrence.id,
                    SearchWindow.entity_type.in_(("audio_transcript", "attachment_text")),
                )
            )
        )
    assert {window.activity_id for window in derived_windows} == {activity.id}
    assert {window.source_category for window in derived_windows} == {activity.category}
    assert {window.audio_segment_id for window in derived_windows if window.audio_segment_id} == {
        segment_id
    }
    assert {
        window.attachment_extraction_id
        for window in derived_windows
        if window.attachment_extraction_id
    } == {extraction_id}

    lexical = lexical_search(
        catalog_engine,
        "attachment exact needle",
        filters=SearchFilters(entity_type="attachment_text"),
    )
    assert len(lexical) == 1
    primary_occurrence = lexical[0]["occurrences"][0]
    primary_activity_id = activity_by_occurrence[primary_occurrence["occurrence_id"]]
    _assert_activity_jump(lexical[0]["jump"], primary_activity_id)
    assert primary_occurrence["activity"]["id"] == primary_activity_id
    for source_occurrence in lexical[0]["occurrences"]:
        assert (
            source_occurrence["activity_id"]
            == activity_by_occurrence[source_occurrence["occurrence_id"]]
        )

    occurrence_page = derived_occurrence_page(
        catalog_engine,
        entity_type="attachment_text",
        source_id=extraction_id,
    )
    paged_occurrence = occurrence_page["items"][0]
    paged_activity_id = activity_by_occurrence[paged_occurrence["occurrence_id"]]
    assert paged_occurrence["activity_id"] == paged_activity_id
    assert occurrence_page["items"][0]["href"] == (
        f"/timeline?around=activity:{paged_activity_id}#activity-{paged_activity_id}"
    )

    for result in (
        semantic_search(
            archive_layout,
            catalog_engine,
            ATTACHMENT_TEXT,
            filters=SearchFilters(entity_type="attachment_text"),
            embedder_factory=lambda _spec: _FakeEmbedder(),
            reranker_factory=lambda _spec: _FakeReranker(),
        ),
        hybrid_search(
            archive_layout,
            catalog_engine,
            "attachment exact needle",
            filters=SearchFilters(entity_type="attachment_text"),
            embedder_factory=lambda _spec: _FakeEmbedder(),
            reranker_factory=lambda _spec: _FakeReranker(),
        ),
    ):
        item = next(item for item in result.items if item["source_id"] == extraction_id)
        result_occurrence_id = item["jump"]["occurrence_id"]
        _assert_activity_jump(item["jump"], activity_by_occurrence[result_occurrence_id])
