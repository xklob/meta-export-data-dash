from __future__ import annotations

import threading
from collections.abc import Callable
from typing import ClassVar

import pytest
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

from meta_archive.models import (
    AttachmentTextExtraction,
    AudioTranscription,
    AudioTranscriptSegment,
    ConflictAggregateRun,
    ConflictCalibration,
    DeletionOperation,
    Job,
    MediaDerivative,
    NlpAnnotation,
    NlpHourlyRollup,
    SearchWindow,
    SemanticIndexConfig,
)
from meta_archive.paths import ArchiveLayout
from meta_archive.services import attachment_text as attachment_text_service
from meta_archive.services import media_derivatives as media_derivative_service
from meta_archive.services import search_index as search_index_service
from meta_archive.services import transcription as transcription_service
from meta_archive.services.attachment_text import ExtractionResult, build_attachment_text
from meta_archive.services.conflict_aggregates import build_conflict_aggregates
from meta_archive.services.deletion import DeletionError, deletion_lock
from meta_archive.services.derived_publication_guard import DerivedPublicationBlockedError
from meta_archive.services.media_derivatives import DerivativeResult, build_media_derivatives
from meta_archive.services.nlp_annotations import (
    build_nlp_annotations,
    rebuild_nlp_hourly_rollups,
    record_conflict_calibration,
)
from meta_archive.services.search_index import build_semantic_index, rebuild_search_windows
from meta_archive.services.topic_clusters import build_topic_clusters
from meta_archive.services.transcription import TranscriptResult, build_audio_transcriptions

Publisher = Callable[[ArchiveLayout, Engine], object]


class _EmptyAudioAdapter:
    identity = "empty-audio-adapter"
    model_name = "fixture/audio"
    model_revision = "fixture-audio-v1"
    device = "cpu"
    compute_type = "int8"

    def transcribe(self, _source: object) -> TranscriptResult:
        raise AssertionError("empty candidate set must not invoke transcription")


class _EmptyAttachmentExtractor:
    identity = "empty-attachment-extractor"
    revision = "fixture-attachment-v1"
    sandbox_status: ClassVar[dict[str, object]] = {"available": True, "status": "fixture"}

    def extract(self, *_args: object, **_kwargs: object) -> ExtractionResult:
        raise AssertionError("empty candidate set must not invoke extraction")


class _EmptyMediaRunner:
    identity = "empty-media-runner"

    def generate(self, *_args: object, **_kwargs: object) -> DerivativeResult:
        raise AssertionError("empty candidate set must not invoke media generation")


def _semantic_index(layout: ArchiveLayout, engine: Engine) -> object:
    return build_semantic_index(layout, engine)


def _search_windows(layout: ArchiveLayout, engine: Engine) -> object:
    return rebuild_search_windows(layout, engine)


def _nlp_annotations(layout: ArchiveLayout, engine: Engine) -> object:
    return build_nlp_annotations(layout, engine)


def _nlp_rollups(layout: ArchiveLayout, engine: Engine) -> object:
    return rebuild_nlp_hourly_rollups(layout, engine)


def _conflict_aggregates(layout: ArchiveLayout, engine: Engine) -> object:
    return build_conflict_aggregates(layout, engine)


def _conflict_calibration(layout: ArchiveLayout, engine: Engine) -> object:
    return record_conflict_calibration(layout, engine)


def _topic_clusters(layout: ArchiveLayout, engine: Engine) -> object:
    return build_topic_clusters(layout, engine)


def _audio_transcriptions(layout: ArchiveLayout, engine: Engine) -> object:
    return build_audio_transcriptions(layout, engine)


def _attachment_text(layout: ArchiveLayout, engine: Engine) -> object:
    return build_attachment_text(layout, engine)


def _media_derivatives(layout: ArchiveLayout, engine: Engine) -> object:
    return build_media_derivatives(layout, engine)


PUBLISHERS: tuple[tuple[str, Publisher], ...] = (
    ("semantic-index", _semantic_index),
    ("search-window", _search_windows),
    ("nlp-annotation", _nlp_annotations),
    ("nlp-rollup", _nlp_rollups),
    ("conflict-aggregate", _conflict_aggregates),
    ("conflict-calibration", _conflict_calibration),
    ("topic-cluster", _topic_clusters),
    ("audio-transcription", _audio_transcriptions),
    ("attachment-text", _attachment_text),
    ("media-derivative", _media_derivatives),
)


def _add_deletion(engine: Engine, *, status: str) -> str:
    operation_id = f"guard-{status}"
    with Session(engine) as session, session.begin():
        session.add(
            DeletionOperation(
                id=operation_id,
                status=status,
                selector_json={"entity_types": ["message"]},
                impact_json={},
                catalog_revision="a" * 64,
                selection_sha256="b" * 64,
                purge_plan_json={},
                item_count=0,
            )
        )
    return operation_id


def _derived_row_counts(engine: Engine) -> tuple[int, ...]:
    models = (
        Job,
        SearchWindow,
        SemanticIndexConfig,
        AudioTranscription,
        AudioTranscriptSegment,
        AttachmentTextExtraction,
        MediaDerivative,
        NlpAnnotation,
        NlpHourlyRollup,
        ConflictCalibration,
        ConflictAggregateRun,
    )
    with Session(engine) as session:
        return tuple(int(session.scalar(select(func.count(model.id))) or 0) for model in models)


@pytest.mark.parametrize("status", ["committed", "cleanup_pending", "normalized_purged"])
@pytest.mark.parametrize(("publisher_name", "publisher"), PUBLISHERS)
def test_derived_publishers_reject_outstanding_deletion_without_mutation(
    archive_layout: ArchiveLayout,
    catalog_engine: Engine,
    status: str,
    publisher_name: str,
    publisher: Publisher,
) -> None:
    operation_id = _add_deletion(catalog_engine, status=status)
    before = _derived_row_counts(catalog_engine)

    with pytest.raises(DerivedPublicationBlockedError) as raised:
        publisher(archive_layout, catalog_engine)

    assert raised.value.operation_id == operation_id
    assert raised.value.status == status
    assert publisher_name in raised.value.publisher.casefold().replace(" ", "-")
    assert _derived_row_counts(catalog_engine) == before
    with Session(catalog_engine) as session:
        operation = session.get(DeletionOperation, operation_id)
        assert operation is not None
        assert operation.status == status


def test_blocked_semantic_job_remains_queued(
    archive_layout: ArchiveLayout,
    catalog_engine: Engine,
) -> None:
    _add_deletion(catalog_engine, status="committed")
    job_id = "queued-before-quarantine-build"
    with Session(catalog_engine) as session, session.begin():
        session.add(
            Job(
                id=job_id,
                kind="semantic_index",
                status="queued",
                stage="Semantic index queued",
                progress_current=0,
                checkpoint_json={},
                counters_json={},
            )
        )

    with pytest.raises(DerivedPublicationBlockedError):
        build_semantic_index(archive_layout, catalog_engine, job_id=job_id)

    with Session(catalog_engine) as session:
        job = session.get(Job, job_id)
        assert job is not None
        assert job.status == "queued"
        assert job.stage == "Semantic index queued"
        assert job.error_text is None


def test_semantic_publisher_holds_deletion_lock_for_complete_mutation(
    archive_layout: ArchiveLayout,
    catalog_engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mutation_started = threading.Event()
    allow_mutation_to_finish = threading.Event()
    failures: list[BaseException] = []

    def blocked_mutation(*_args: object, **_kwargs: object) -> object:
        mutation_started.set()
        if not allow_mutation_to_finish.wait(timeout=5):
            raise TimeoutError("test did not release synthetic semantic publication")
        return object()

    monkeypatch.setattr(search_index_service, "_build_semantic_index_locked", blocked_mutation)

    def publish() -> None:
        try:
            build_semantic_index(archive_layout, catalog_engine)
        except BaseException as exc:
            failures.append(exc)

    worker = threading.Thread(target=publish)
    worker.start()
    try:
        assert mutation_started.wait(timeout=5)
        with (
            pytest.raises(DeletionError, match="currently mutating"),
            deletion_lock(archive_layout),
        ):
            pytest.fail("deletion acquired its lock during semantic publication")
    finally:
        allow_mutation_to_finish.set()
        worker.join(timeout=5)

    assert not worker.is_alive()
    assert failures == []


def test_restored_deletion_does_not_block_new_publication(
    archive_layout: ArchiveLayout,
    catalog_engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _add_deletion(catalog_engine, status="restored")
    published = object()
    monkeypatch.setattr(
        search_index_service,
        "_build_semantic_index_locked",
        lambda *_args, **_kwargs: published,
    )

    assert build_semantic_index(archive_layout, catalog_engine) is published


@pytest.mark.parametrize("publisher", ["audio", "attachment"])
def test_file_derived_publishers_hold_deletion_lock_for_complete_work(
    archive_layout: ArchiveLayout,
    catalog_engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
    publisher: str,
) -> None:
    candidate_scan_started = threading.Event()
    allow_candidate_scan_to_finish = threading.Event()
    failures: list[BaseException] = []

    def blocked_candidates(_engine: Engine) -> list[object]:
        candidate_scan_started.set()
        if not allow_candidate_scan_to_finish.wait(timeout=5):
            raise TimeoutError("test did not release synthetic candidate scan")
        return []

    if publisher == "audio":
        monkeypatch.setattr(transcription_service, "_candidates", blocked_candidates)

        def publish() -> None:
            build_audio_transcriptions(
                archive_layout,
                catalog_engine,
                adapter=_EmptyAudioAdapter(),
            )

    else:
        monkeypatch.setattr(attachment_text_service, "_candidate_rows", blocked_candidates)

        def publish() -> None:
            build_attachment_text(
                archive_layout,
                catalog_engine,
                extractor=_EmptyAttachmentExtractor(),
            )

    def run_publisher() -> None:
        try:
            publish()
        except BaseException as exc:
            failures.append(exc)

    worker = threading.Thread(target=run_publisher)
    worker.start()
    try:
        assert candidate_scan_started.wait(timeout=5)
        with (
            pytest.raises(DeletionError, match="currently mutating"),
            deletion_lock(archive_layout),
        ):
            pytest.fail("deletion acquired its lock during derived publication")
    finally:
        allow_candidate_scan_to_finish.set()
        worker.join(timeout=5)

    assert not worker.is_alive()
    assert failures == []


def test_restored_deletion_allows_file_and_media_derived_publishers(
    archive_layout: ArchiveLayout,
    catalog_engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _add_deletion(catalog_engine, status="restored")
    monkeypatch.setattr(transcription_service, "_candidates", lambda _engine: [])
    monkeypatch.setattr(attachment_text_service, "_candidate_rows", lambda _engine: [])
    monkeypatch.setattr(media_derivative_service, "_candidates", lambda _engine: [])

    audio = build_audio_transcriptions(
        archive_layout,
        catalog_engine,
        adapter=_EmptyAudioAdapter(),
    )
    attachment = build_attachment_text(
        archive_layout,
        catalog_engine,
        extractor=_EmptyAttachmentExtractor(),
    )
    media = build_media_derivatives(
        archive_layout,
        catalog_engine,
        runner=_EmptyMediaRunner(),
    )

    assert audio.total == 0
    assert attachment.total == 0
    assert media.total == 0
