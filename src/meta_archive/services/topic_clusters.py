from __future__ import annotations

import fcntl
import hashlib
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import lancedb  # type: ignore[import-untyped]
import numpy as np
from sklearn.cluster import MiniBatchKMeans  # type: ignore[import-untyped]
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-untyped]
from sqlalchemy import Engine, select, update
from sqlalchemy.orm import Session

from meta_archive.models import Job, SemanticIndexConfig
from meta_archive.paths import ArchiveLayout
from meta_archive.services.deletion import vector_store_lock
from meta_archive.services.deletion_lock_lease import deletion_lock
from meta_archive.services.derived_publication_guard import (
    DerivedPublicationBlockedError,
    assert_derived_publication_allowed,
)
from meta_archive.services.identifiers import stable_id
from meta_archive.services.semantic_state import (
    SemanticWindowIdentity,
    lance_table_names,
    lance_window_identity,
    semantic_config_window_count,
    semantic_config_window_ids,
    semantic_window_identity,
)

PIPELINE_VERSION = "topic-clusters-v1"
DEFAULT_CLUSTER_COUNT = 12
DEFAULT_SAMPLE_LIMIT = 20_000
MIN_SAMPLE_PER_CLUSTER = 20
MAX_TERMS = 8
MAX_REPRESENTATIVES = 4

ProgressCallback = Callable[[str, int, int | None], None]


@contextmanager
def _immediate_session(engine: Engine) -> Iterator[Session]:
    """Reserve SQLite's writer slot across final source validation and publication."""

    with engine.connect() as connection:
        connection.exec_driver_sql("BEGIN IMMEDIATE")
        session = Session(bind=connection)
        try:
            yield session
            session.flush()
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            session.close()


def _assert_no_logical_quarantine(engine: Engine) -> None:
    try:
        assert_derived_publication_allowed(engine, publisher="Topic cluster publication")
    except DerivedPublicationBlockedError as exc:
        raise RuntimeError(
            "Topic clustering is unavailable while logically deleted rows remain; "
            "restore or hard-purge the deletion first"
        ) from exc


@dataclass(frozen=True, slots=True)
class TopicSample:
    id: str
    vector: tuple[float, ...]
    text: str


@dataclass(frozen=True, slots=True)
class TopicClusterSummary:
    job_id: str
    pipeline_version: str
    semantic_config_id: str
    corpus_window_count: int
    sample_count: int
    cluster_count: int
    resumed: bool


def topic_cluster_lock_is_held(layout: ArchiveLayout) -> bool:
    lock_path = layout.catalog / "topic-clusters.lock"
    with lock_path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        finally:
            with suppress(OSError):
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return False


def _even_offsets(total: int, limit: int) -> list[int]:
    if total <= 0 or limit <= 0:
        return []
    if total <= limit:
        return list(range(total))
    # Sampling across physical row offsets avoids loading every vector. The active Lance table is
    # immutable after index activation, so the sampled offsets remain reproducible for that table.
    return sorted({int(value) for value in np.linspace(0, total - 1, num=limit)})


def _load_sample(
    layout: ArchiveLayout,
    engine: Engine,
    config: SemanticIndexConfig,
    *,
    sample_limit: int,
) -> tuple[list[TopicSample], int, SemanticWindowIdentity]:
    database = lancedb.connect(str(layout.vectors))
    table_names = lance_table_names(database)
    if config.vector_table not in table_names:
        raise RuntimeError("The active semantic vector table is missing; rebuild the index")
    table = database.open_table(config.vector_table)
    total = table.count_rows()
    with Session(engine) as session:
        sqlite_identity = semantic_window_identity(semantic_config_window_ids(session, config.id))
    lance_identity = lance_window_identity(table)
    if (
        total != int(config.window_count)
        or lance_identity.row_count != total
        or lance_identity.unique_count != total
        or sqlite_identity.row_count != total
        or sqlite_identity.unique_count != total
        or sqlite_identity != lance_identity
    ):
        raise RuntimeError(
            "The active semantic index counts diverge or identities diverge across its config, "
            "SQLite membership, and Lance table; rebuild the index"
        )
    offsets = _even_offsets(total, sample_limit)
    if not offsets:
        return [], total, lance_identity
    rows = table.take_offsets(offsets).select(["id", "vector", "text"]).to_list()
    sample: list[TopicSample] = []
    for row in rows:
        vector_value = row.get("vector")
        if not isinstance(vector_value, Sequence) or isinstance(vector_value, (str, bytes)):
            raise RuntimeError("Semantic vector table contains an invalid vector")
        vector = tuple(float(value) for value in vector_value)
        if len(vector) != config.dimensions or not all(np.isfinite(value) for value in vector):
            raise RuntimeError("Semantic vector table contains an invalid vector shape")
        sample.append(
            TopicSample(
                id=str(row["id"]),
                vector=vector,
                text=str(row.get("text") or ""),
            )
        )
    return sorted(sample, key=lambda row: row.id), total, lance_identity


def _topic_terms(texts: list[str], labels: np.ndarray[Any, Any], count: int) -> list[list[str]]:
    terms: list[list[str]] = [[] for _ in range(count)]
    try:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2 if len(texts) >= 40 else 1,
            max_df=0.92 if len(texts) >= 40 else 1.0,
            max_features=12_000,
        )
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return terms
    vocabulary = vectorizer.get_feature_names_out()
    for cluster_index in range(count):
        member_indices = np.flatnonzero(labels == cluster_index)
        if not len(member_indices):
            continue
        mean_scores = np.asarray(matrix[member_indices].mean(axis=0)).ravel()
        ranked = np.argsort(mean_scores)[::-1]
        terms[cluster_index] = [
            str(vocabulary[index]) for index in ranked if mean_scores[index] > 0
        ][:MAX_TERMS]
    return terms


def cluster_topic_sample(
    sample: Sequence[TopicSample],
    *,
    semantic_config_id: str,
    requested_clusters: int = DEFAULT_CLUSTER_COUNT,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    if requested_clusters < 1:
        raise ValueError("Topic cluster count must be positive")
    if not sample:
        return []
    dimensions = {len(row.vector) for row in sample}
    if len(dimensions) != 1 or 0 in dimensions:
        raise ValueError("Topic samples must use one non-empty embedding shape")
    sample_count = len(sample)
    supported_clusters = max(1, sample_count // MIN_SAMPLE_PER_CLUSTER)
    cluster_count = min(requested_clusters, supported_clusters, sample_count)
    matrix = np.asarray([row.vector for row in sample], dtype=np.float32)
    model = MiniBatchKMeans(
        n_clusters=cluster_count,
        random_state=random_state,
        n_init="auto",
        batch_size=min(2_048, max(64, sample_count)),
        reassignment_ratio=0.01,
    )
    labels = model.fit_predict(matrix)
    terms = _topic_terms([row.text for row in sample], labels, cluster_count)
    distances = model.transform(matrix)
    raw: list[dict[str, Any]] = []
    for cluster_index in range(cluster_count):
        member_indices = np.flatnonzero(labels == cluster_index)
        ranked_members = sorted(
            (int(index) for index in member_indices),
            key=lambda index: (float(distances[index, cluster_index]), sample[index].id),
        )
        centroid_digest = hashlib.sha256(
            np.asarray(model.cluster_centers_[cluster_index], dtype=np.float32).tobytes()
        ).hexdigest()
        raw.append(
            {
                "id": stable_id(
                    "topic-cluster", PIPELINE_VERSION, semantic_config_id, centroid_digest
                ),
                "sample_windows": len(member_indices),
                "sample_share": len(member_indices) / sample_count,
                "terms": terms[cluster_index],
                "representative_window_ids": [
                    sample[index].id for index in ranked_members[:MAX_REPRESENTATIVES]
                ],
            }
        )
    raw.sort(
        key=lambda row: (
            -int(row["sample_windows"]),
            tuple(str(term) for term in row["terms"]),
            str(row["id"]),
        )
    )
    for ordinal, row in enumerate(raw, start=1):
        row["ordinal"] = ordinal
    return raw


def _active_config(engine: Engine) -> SemanticIndexConfig:
    with Session(engine) as session:
        config = session.scalar(
            select(SemanticIndexConfig).where(
                SemanticIndexConfig.active.is_(True),
                SemanticIndexConfig.status == "ready",
            )
        )
    if config is None:
        raise RuntimeError("Build a semantic text index before topic clustering")
    return config


def _prepare_job(
    engine: Engine,
    *,
    config: SemanticIndexConfig,
    requested_job_id: str | None,
    sample_limit: int,
    requested_clusters: int,
) -> tuple[str, bool]:
    now = datetime.now(UTC)
    with Session(engine) as session, session.begin():
        if requested_job_id is not None:
            job = session.get(Job, requested_job_id)
            if job is None or job.kind != "topic_clusters":
                raise RuntimeError("Requested topic-cluster job does not exist")
            if job.status not in {"queued", "running"}:
                raise RuntimeError("Requested topic-cluster job is not active")
            expected_checkpoint = {
                "semantic_config_id": config.id,
                "sample_limit": sample_limit,
                "requested_clusters": requested_clusters,
            }
            if any(
                job.checkpoint_json.get(key) != value for key, value in expected_checkpoint.items()
            ):
                raise RuntimeError(
                    "Requested topic-cluster job no longer matches the active semantic index "
                    "and requested clustering configuration"
                )
            job.status = "running"
            job.stage = "Sampling semantic windows"
            job.started_at = now
            job.error_text = None
            return job.id, False
        previous = session.scalar(
            select(Job)
            .where(Job.kind == "topic_clusters", Job.status == "failed")
            .order_by(Job.created_at.desc())
        )
        resumed = bool(
            previous
            and previous.checkpoint_json.get("semantic_config_id") == config.id
            and previous.checkpoint_json.get("sample_limit") == sample_limit
            and previous.checkpoint_json.get("requested_clusters") == requested_clusters
        )
        if resumed and previous is not None:
            previous.status = "running"
            previous.stage = "Retrying topic clustering"
            previous.started_at = now
            previous.finished_at = None
            previous.error_text = None
            return previous.id, True
        job_id = str(uuid.uuid4())
        session.add(
            Job(
                id=job_id,
                kind="topic_clusters",
                status="running",
                stage="Sampling semantic windows",
                progress_current=0,
                progress_total=sample_limit,
                checkpoint_json={
                    "pipeline_version": PIPELINE_VERSION,
                    "semantic_config_id": config.id,
                    "sample_limit": sample_limit,
                    "requested_clusters": requested_clusters,
                },
                counters_json={},
                started_at=now,
            )
        )
        return job_id, False


def build_topic_clusters(
    layout: ArchiveLayout,
    engine: Engine,
    *,
    requested_clusters: int = DEFAULT_CLUSTER_COUNT,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
    requested_job_id: str | None = None,
    progress: ProgressCallback | None = None,
) -> TopicClusterSummary:
    if requested_clusters < 1 or requested_clusters > 50:
        raise ValueError("Topic cluster count must be between 1 and 50")
    if sample_limit < 100 or sample_limit > 100_000:
        raise ValueError("Topic sample limit must be between 100 and 100,000")
    lock_path = layout.catalog / "topic-clusters.lock"
    with ExitStack() as archive_locks, lock_path.open("a+b") as lock_handle:
        archive_locks.enter_context(deletion_lock(layout))
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another topic-cluster job is already running") from exc
        archive_locks.enter_context(vector_store_lock(layout))
        assert_derived_publication_allowed(engine, publisher="Topic cluster publication")
        config = _active_config(engine)
        job_id, resumed = _prepare_job(
            engine,
            config=config,
            requested_job_id=requested_job_id,
            sample_limit=sample_limit,
            requested_clusters=requested_clusters,
        )
        try:
            sample, corpus_count, source_identity = _load_sample(
                layout,
                engine,
                config,
                sample_limit=sample_limit,
            )
            if not sample:
                raise RuntimeError("The active semantic index has no searchable windows")
            with Session(engine) as session, session.begin():
                session.execute(
                    update(Job)
                    .where(Job.id == job_id)
                    .values(
                        stage="Clustering semantic windows",
                        progress_current=len(sample),
                        progress_total=len(sample),
                    )
                )
            if progress:
                progress("Sampled semantic windows", len(sample), len(sample))
            clusters = cluster_topic_sample(
                sample,
                semantic_config_id=config.id,
                requested_clusters=requested_clusters,
            )
            finished = datetime.now(UTC)
            result = {
                "pipeline_version": PIPELINE_VERSION,
                "semantic_config_id": config.id,
                "vector_table": config.vector_table,
                "model_name": config.model_name,
                "model_revision": config.model_revision,
                "corpus_window_count": corpus_count,
                "sample_count": len(sample),
                "sampling": "Even physical offsets from the immutable active Lance table.",
                "clusters": clusters,
                "warning": (
                    "Topics are unsupervised model estimates from a bounded sample. They are not "
                    "people, relationship, intent, or quality rankings."
                ),
                "generated_at": finished.isoformat(),
            }
            assert_derived_publication_allowed(engine, publisher="Topic cluster publication")
            with _immediate_session(engine) as session:
                current_config = session.scalar(
                    select(SemanticIndexConfig).where(
                        SemanticIndexConfig.active.is_(True),
                        SemanticIndexConfig.status == "ready",
                    )
                )
                current_membership_count = (
                    semantic_config_window_count(session, config.id)
                    if current_config is not None and current_config.id == config.id
                    else -1
                )
                current_identity: SemanticWindowIdentity | None = None
                if current_config is not None and current_config.id == config.id:
                    database = lancedb.connect(str(layout.vectors))
                    table_names = lance_table_names(database)
                    if current_config.vector_table in table_names:
                        table = database.open_table(current_config.vector_table)
                        current_sqlite_identity = semantic_window_identity(
                            semantic_config_window_ids(session, config.id)
                        )
                        current_lance_identity = lance_window_identity(table)
                        if (
                            current_sqlite_identity.row_count
                            == current_sqlite_identity.unique_count
                            == corpus_count
                            and current_lance_identity.row_count
                            == current_lance_identity.unique_count
                            == corpus_count
                            and current_sqlite_identity == current_lance_identity
                        ):
                            current_identity = current_lance_identity
                if (
                    current_config is None
                    or current_config.id != config.id
                    or int(current_config.window_count) != corpus_count
                    or current_membership_count != corpus_count
                    or current_identity != source_identity
                ):
                    raise RuntimeError(
                        "The active semantic index changed while topic clusters were computed; "
                        "discarding the stale result"
                    )
                session.execute(
                    update(Job)
                    .where(Job.id == job_id)
                    .values(
                        status="completed",
                        stage="Topic clusters ready",
                        progress_current=len(sample),
                        progress_total=len(sample),
                        counters_json={"result": result},
                        finished_at=finished,
                        error_text=None,
                    )
                )
            return TopicClusterSummary(
                job_id=job_id,
                pipeline_version=PIPELINE_VERSION,
                semantic_config_id=config.id,
                corpus_window_count=corpus_count,
                sample_count=len(sample),
                cluster_count=len(clusters),
                resumed=resumed,
            )
        except BaseException as exc:
            with Session(engine) as session, session.begin():
                session.execute(
                    update(Job)
                    .where(Job.id == job_id)
                    .values(
                        status="failed",
                        stage="Topic clustering failed",
                        error_text=str(exc)[:4_000],
                        finished_at=datetime.now(UTC),
                    )
                )
            raise
