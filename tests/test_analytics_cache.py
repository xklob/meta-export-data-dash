from __future__ import annotations

import stat
import threading

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from meta_archive.models import Job
from meta_archive.paths import ArchiveLayout
from meta_archive.services.analytics_cache import (
    StaleAnalyticsPayloadError,
    analytics_cache_lock,
    analytics_source_revision,
    clear_analytics_cache,
    read_analytics_cache,
    write_analytics_cache,
)


def _job(job_id: str, kind: str) -> Job:
    return Job(
        id=job_id,
        kind=kind,
        status="completed",
        progress_current=1,
        progress_total=1,
        counters_json={"completed": 1},
        checkpoint_json={},
    )


def test_analytics_cache_is_atomic_revision_bound_and_ignores_derivative_progress(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> None:
    parameters = {"bucket": "month", "timezone": "America/New_York"}
    payload = {"series": [{"period": "2024-01", "messages": 3}]}

    assert (
        read_analytics_cache(
            archive_layout,
            catalog_engine,
            namespace="messages",
            parameters=parameters,
        )
        is None
    )
    assert (
        write_analytics_cache(
            archive_layout,
            catalog_engine,
            namespace="messages",
            parameters=parameters,
            payload=payload,
        )
        == payload
    )
    cache_file = next(archive_layout.analytics_cache.glob("*.json"))
    assert stat.S_IMODE(cache_file.stat().st_mode) == 0o600
    assert stat.S_IMODE((archive_layout.catalog / "analytics-cache.lock").stat().st_mode) == 0o600
    assert (
        read_analytics_cache(
            archive_layout,
            catalog_engine,
            namespace="messages",
            parameters=parameters,
        )
        == payload
    )

    with Session(catalog_engine) as session, session.begin():
        session.add(_job("media-progress", "media_derivatives"))
    assert (
        read_analytics_cache(
            archive_layout,
            catalog_engine,
            namespace="messages",
            parameters=parameters,
        )
        == payload
    )

    with Session(catalog_engine) as session, session.begin():
        session.add(_job("nlp-change", "nlp_annotations"))
    assert (
        read_analytics_cache(
            archive_layout,
            catalog_engine,
            namespace="messages",
            parameters=parameters,
        )
        is None
    )


def test_analytics_cache_clear_fails_closed_on_unexpected_entries(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> None:
    write_analytics_cache(
        archive_layout,
        catalog_engine,
        namespace="overview",
        parameters={},
        payload={"messages": {"total": 0}},
    )
    unexpected = archive_layout.analytics_cache / "unexpected.txt"
    unexpected.write_text("do not delete", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Unexpected analytics cache entry"):
        clear_analytics_cache(archive_layout)

    unexpected.unlink()
    assert clear_analytics_cache(archive_layout) == 1
    assert not list(archive_layout.analytics_cache.iterdir())


def test_analytics_cache_never_persists_a_payload_computed_before_revision_change(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> None:
    stale_revision = analytics_source_revision(archive_layout, catalog_engine)
    with Session(catalog_engine) as session, session.begin():
        session.add(_job("catalog-changed-during-computation", "nlp_annotations"))

    payload = {"private": "stale result must not be written"}
    with pytest.raises(StaleAnalyticsPayloadError, match="changed while analytics"):
        write_analytics_cache(
            archive_layout,
            catalog_engine,
            namespace="stale-computation",
            parameters={},
            payload=payload,
            expected_source_revision=stale_revision,
        )
    assert not list(archive_layout.analytics_cache.glob("*.json"))

    current_revision = analytics_source_revision(archive_layout, catalog_engine)
    write_analytics_cache(
        archive_layout,
        catalog_engine,
        namespace="current-computation",
        parameters={},
        payload={"safe": "current"},
        expected_source_revision=current_revision,
    )
    assert len(list(archive_layout.analytics_cache.glob("*.json"))) == 1


def test_analytics_cache_reader_waits_for_exclusive_purge_cleanup(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> None:
    parameters = {"bucket": "month"}
    write_analytics_cache(
        archive_layout,
        catalog_engine,
        namespace="purge-race",
        parameters=parameters,
        payload={"private": "must not cross purge cleanup"},
    )
    cache_file = next(archive_layout.analytics_cache.glob("*.json"))
    reader_started = threading.Event()
    reader_finished = threading.Event()
    result: list[dict[str, object] | None] = []

    def read_during_cleanup() -> None:
        reader_started.set()
        result.append(
            read_analytics_cache(
                archive_layout,
                catalog_engine,
                namespace="purge-race",
                parameters=parameters,
            )
        )
        reader_finished.set()

    with analytics_cache_lock(archive_layout, exclusive=True):
        reader = threading.Thread(target=read_during_cleanup)
        reader.start()
        assert reader_started.wait(timeout=1)
        assert not reader_finished.wait(timeout=0.1)
        cache_file.unlink()

    reader.join(timeout=2)
    assert not reader.is_alive()
    assert result == [None]
