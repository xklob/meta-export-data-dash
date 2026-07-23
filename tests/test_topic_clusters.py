from __future__ import annotations

import math
from datetime import UTC, datetime

import lancedb  # type: ignore[import-untyped]
import pytest
from sqlalchemy import Engine, select, update
from sqlalchemy.orm import Session

from meta_archive.models import Job, SearchWindow, SemanticIndexConfig, SemanticWindowEmbedding
from meta_archive.paths import ArchiveLayout
from meta_archive.services import topic_clusters as topic_service
from meta_archive.services.topic_clusters import (
    TopicSample,
    _even_offsets,
    build_topic_clusters,
    cluster_topic_sample,
)


def test_even_offsets_cover_the_full_immutable_table() -> None:
    assert _even_offsets(0, 10) == []
    assert _even_offsets(3, 10) == [0, 1, 2]
    offsets = _even_offsets(101, 5)
    assert offsets == [0, 25, 50, 75, 100]


def test_topic_sample_clustering_is_deterministic_and_bounded() -> None:
    sample = [
        TopicSample(
            id=f"window-food-{index:03d}",
            vector=(1.0, index / 10_000, 0.0),
            text=f"recipe dinner cooking pasta kitchen meal {index}",
        )
        for index in range(40)
    ] + [
        TopicSample(
            id=f"window-travel-{index:03d}",
            vector=(0.0, index / 10_000, 1.0),
            text=f"flight hotel travel beach vacation trip {index}",
        )
        for index in range(40)
    ]

    first = cluster_topic_sample(sample, semantic_config_id="config-a", requested_clusters=2)
    second = cluster_topic_sample(sample, semantic_config_id="config-a", requested_clusters=2)

    assert first == second
    assert len(first) == 2
    assert sum(cluster["sample_windows"] for cluster in first) == len(sample)
    assert math.isclose(sum(cluster["sample_share"] for cluster in first), 1.0)
    assert [cluster["ordinal"] for cluster in first] == [1, 2]
    assert all(1 <= len(cluster["representative_window_ids"]) <= 4 for cluster in first)
    assert {term for cluster in first for term in cluster["terms"]} & {
        "cooking",
        "travel",
        "recipe dinner",
        "flight hotel",
    }
    assert all("text" not in cluster for cluster in first)


def test_topic_sample_rejects_invalid_shape_and_cluster_count() -> None:
    with pytest.raises(ValueError, match="positive"):
        cluster_topic_sample([], semantic_config_id="config", requested_clusters=0)
    with pytest.raises(ValueError, match="embedding shape"):
        cluster_topic_sample(
            [
                TopicSample(id="a", vector=(1.0,), text="one"),
                TopicSample(id="b", vector=(1.0, 2.0), text="two"),
            ],
            semantic_config_id="config",
        )


def _seed_topic_index(layout: ArchiveLayout, engine: Engine) -> str:
    config_id = "a" * 64
    table_name = "fixture_topic_vectors"
    with Session(engine) as session, session.begin():
        session.add(
            SemanticIndexConfig(
                id=config_id,
                active=True,
                status="ready",
                backend="fixture",
                backend_version="fixture-v1",
                model_name="fixture/model",
                model_revision="fixture-revision",
                dimensions=3,
                distance_metric="cosine",
                document_prefix="",
                query_prefix="",
                vector_table=table_name,
                window_version="fixture-window-v1",
                window_count=100,
                audit_json={},
                completed_at=datetime.now(UTC),
            )
        )
        for ordinal in range(100):
            window_id = f"topic-window-{ordinal:04d}"
            session.add(
                SearchWindow(
                    id=window_id,
                    window_version="fixture-window-v1",
                    entity_type="activity",
                    first_timestamp_ms=ordinal,
                    last_timestamp_ms=ordinal,
                    turn_count=1,
                    approx_token_count=4,
                    text_display=f"fixture topic document {ordinal}",
                    text_search=f"fixture topic document {ordinal}",
                    participants_search="",
                    source_ids_json={},
                    build_generation="fixture-generation",
                    embedding_config_hash=config_id,
                )
            )
            session.add(SemanticWindowEmbedding(config_id=config_id, window_id=window_id))
    lancedb.connect(str(layout.vectors)).create_table(
        table_name,
        data=[
            {
                "id": f"topic-window-{ordinal:04d}",
                "vector": [1.0, (ordinal + 1) / 10_000, 0.0],
                "text": f"fixture topic document {ordinal}",
            }
            for ordinal in range(100)
        ],
        mode="overwrite",
    )
    return config_id


def test_topic_build_publishes_reconciled_result(
    archive_layout: ArchiveLayout,
    catalog_engine: Engine,
) -> None:
    config_id = _seed_topic_index(archive_layout, catalog_engine)
    progress_updates: list[tuple[str, int, int | None]] = []

    summary = build_topic_clusters(
        archive_layout,
        catalog_engine,
        requested_clusters=4,
        sample_limit=100,
        progress=lambda stage, current, total: progress_updates.append((stage, current, total)),
    )

    assert summary.semantic_config_id == config_id
    assert summary.pipeline_version == topic_service.PIPELINE_VERSION
    assert summary.corpus_window_count == 100
    assert summary.sample_count == 100
    assert summary.cluster_count == 4
    assert summary.resumed is False
    assert progress_updates == [("Sampled semantic windows", 100, 100)]

    with Session(catalog_engine) as session:
        config = session.get(SemanticIndexConfig, config_id)
        job = session.get(Job, summary.job_id)
        seeded_window_ids = set(
            session.scalars(
                select(SemanticWindowEmbedding.window_id).where(
                    SemanticWindowEmbedding.config_id == config_id
                )
            )
        )

        assert config is not None
        assert config.active is True
        assert config.status == "ready"
        assert config.vector_table == "fixture_topic_vectors"
        assert config.window_count == summary.corpus_window_count

        assert job is not None
        assert job.status == "completed"
        assert job.stage == "Topic clusters ready"
        assert job.progress_current == summary.sample_count
        assert job.progress_total == summary.sample_count
        assert job.finished_at is not None
        assert job.error_text is None
        assert job.checkpoint_json == {
            "pipeline_version": topic_service.PIPELINE_VERSION,
            "semantic_config_id": config_id,
            "sample_limit": 100,
            "requested_clusters": 4,
        }

        result = job.counters_json["result"]
        assert result["pipeline_version"] == summary.pipeline_version
        assert result["semantic_config_id"] == summary.semantic_config_id
        assert result["vector_table"] == config.vector_table
        assert result["model_name"] == config.model_name
        assert result["model_revision"] == config.model_revision
        assert result["corpus_window_count"] == summary.corpus_window_count
        assert result["sample_count"] == summary.sample_count
        assert len(result["clusters"]) == summary.cluster_count
        assert (
            sum(cluster["sample_windows"] for cluster in result["clusters"]) == summary.sample_count
        )
        assert sum(cluster["sample_share"] for cluster in result["clusters"]) == pytest.approx(1.0)
        assert [cluster["ordinal"] for cluster in result["clusters"]] == [1, 2, 3, 4]
        assert {
            window_id
            for cluster in result["clusters"]
            for window_id in cluster["representative_window_ids"]
        } <= seeded_window_ids

    vector_table = lancedb.connect(str(archive_layout.vectors)).open_table("fixture_topic_vectors")
    assert vector_table.count_rows() == summary.corpus_window_count


def test_topic_build_resumes_matching_failed_job(
    archive_layout: ArchiveLayout,
    catalog_engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_id = _seed_topic_index(archive_layout, catalog_engine)
    original_cluster = topic_service.cluster_topic_sample

    def fail_clustering(*args: object, **kwargs: object) -> list[dict[str, object]]:
        raise RuntimeError("synthetic clustering failure")

    monkeypatch.setattr(topic_service, "cluster_topic_sample", fail_clustering)
    with pytest.raises(RuntimeError, match="synthetic clustering failure"):
        build_topic_clusters(
            archive_layout,
            catalog_engine,
            requested_clusters=4,
            sample_limit=100,
        )

    with Session(catalog_engine) as session:
        failed_job = session.scalar(select(Job).where(Job.kind == "topic_clusters"))
        assert failed_job is not None
        failed_job_id = failed_job.id
        assert failed_job.status == "failed"
        assert failed_job.finished_at is not None

    monkeypatch.setattr(topic_service, "cluster_topic_sample", original_cluster)
    summary = build_topic_clusters(
        archive_layout,
        catalog_engine,
        requested_clusters=4,
        sample_limit=100,
    )

    assert summary.job_id == failed_job_id
    assert summary.semantic_config_id == config_id
    assert summary.resumed is True
    assert summary.corpus_window_count == 100
    assert summary.sample_count == 100
    assert summary.cluster_count == 4
    with Session(catalog_engine) as session:
        jobs = list(session.scalars(select(Job).where(Job.kind == "topic_clusters")))
        assert len(jobs) == 1
        assert jobs[0].status == "completed"
        assert jobs[0].stage == "Topic clusters ready"
        assert jobs[0].error_text is None
        assert len(jobs[0].counters_json["result"]["clusters"]) == summary.cluster_count


def _corrupt_topic_table_same_count(layout: ArchiveLayout, *, duplicate: bool) -> None:
    table = lancedb.connect(str(layout.vectors)).open_table("fixture_topic_vectors")
    rows = table.search().select(["id", "vector", "text"]).limit(3).to_list()
    assert len(rows) == 3
    removed_ids = [str(row["id"]) for row in rows[: (2 if duplicate else 1)]]
    table.delete(" OR ".join(f"id = '{window_id}'" for window_id in removed_ids))
    replacement = dict(rows[-1])
    replacement.pop("_distance", None)
    if duplicate:
        table.add([replacement, replacement])
    else:
        replacement["id"] = "topic-window-same-count-substitution"
        table.add([replacement])
    assert table.count_rows() == 100


@pytest.mark.parametrize("duplicate", [False, True], ids=["wrong-id", "duplicate-id"])
def test_topic_build_rejects_same_count_lance_identity_divergence_before_sampling(
    archive_layout: ArchiveLayout,
    catalog_engine: Engine,
    duplicate: bool,
) -> None:
    _seed_topic_index(archive_layout, catalog_engine)
    _corrupt_topic_table_same_count(archive_layout, duplicate=duplicate)

    with pytest.raises(RuntimeError, match="identities diverge"):
        build_topic_clusters(archive_layout, catalog_engine, sample_limit=100)

    with Session(catalog_engine) as session:
        job = session.scalar(select(Job).where(Job.kind == "topic_clusters"))
        assert job is not None and job.status == "failed"


@pytest.mark.parametrize("duplicate", [False, True], ids=["wrong-id", "duplicate-id"])
def test_topic_build_discards_same_count_lance_identity_change_before_publication(
    archive_layout: ArchiveLayout,
    catalog_engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
    duplicate: bool,
) -> None:
    _seed_topic_index(archive_layout, catalog_engine)
    original_cluster = topic_service.cluster_topic_sample

    def corrupt_during_cluster(*args: object, **kwargs: object) -> list[dict[str, object]]:
        clusters = original_cluster(*args, **kwargs)  # type: ignore[arg-type]
        _corrupt_topic_table_same_count(archive_layout, duplicate=duplicate)
        return clusters

    monkeypatch.setattr(topic_service, "cluster_topic_sample", corrupt_during_cluster)
    with pytest.raises(RuntimeError, match="changed while topic clusters were computed"):
        build_topic_clusters(archive_layout, catalog_engine, sample_limit=100)

    with Session(catalog_engine) as session:
        job = session.scalar(select(Job).where(Job.kind == "topic_clusters"))
        assert job is not None and job.status == "failed"
        assert "result" not in job.counters_json


def test_topic_build_rejects_config_sqlite_lance_count_divergence(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> None:
    config_id = _seed_topic_index(archive_layout, catalog_engine)
    with Session(catalog_engine) as session, session.begin():
        session.execute(
            update(SemanticIndexConfig)
            .where(SemanticIndexConfig.id == config_id)
            .values(window_count=99)
        )

    with pytest.raises(RuntimeError, match="counts diverge"):
        build_topic_clusters(archive_layout, catalog_engine, sample_limit=100)

    with Session(catalog_engine) as session:
        job = session.scalar(select(Job).where(Job.kind == "topic_clusters"))
        assert job is not None and job.status == "failed"


def test_topic_build_discards_result_if_semantic_config_turns_stale(
    archive_layout: ArchiveLayout,
    catalog_engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_id = _seed_topic_index(archive_layout, catalog_engine)
    original_cluster = topic_service.cluster_topic_sample

    def invalidate_during_cluster(*args: object, **kwargs: object) -> list[dict[str, object]]:
        clusters = original_cluster(*args, **kwargs)  # type: ignore[arg-type]
        with Session(catalog_engine) as session, session.begin():
            session.execute(
                update(SemanticIndexConfig)
                .where(SemanticIndexConfig.id == config_id)
                .values(active=False, status="stale")
            )
        return clusters

    monkeypatch.setattr(topic_service, "cluster_topic_sample", invalidate_during_cluster)
    with pytest.raises(RuntimeError, match="changed while topic clusters were computed"):
        build_topic_clusters(archive_layout, catalog_engine, sample_limit=100)

    with Session(catalog_engine) as session:
        job = session.scalar(select(Job).where(Job.kind == "topic_clusters"))
        assert job is not None and job.status == "failed"
        assert "result" not in job.counters_json
