from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lancedb
import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from sqlalchemy import text, update
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
from sqlalchemy.pool import QueuePool

import meta_archive.api.analytics_routes as analytics_routes
import meta_archive.api.main as main_module
from meta_archive import __version__
from meta_archive.api.main import create_app
from meta_archive.database import (
    CatalogMigrationError,
    create_catalog_engine,
    run_migrations,
)
from meta_archive.models import (
    ConflictAggregateRun,
    ConflictHourlyRollup,
    Conversation,
    NlpHourlyRollup,
    SemanticIndexConfig,
)
from meta_archive.paths import ArchiveLayout
from meta_archive.services.acceptance_identity import backend_source_digest
from meta_archive.services.inventory import scan_archive
from meta_archive.services.search_index import (
    EmbeddingModelSpec,
    embedding_implementation_identity,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_ACCEPTANCE_EMBEDDING_SPEC = EmbeddingModelSpec(
    alias="acceptance-fixture",
    model_name="fixture/acceptance-embedding",
    model_revision="fixture-v1",
    dimensions=4,
    document_prefix="passage: ",
    query_prefix="query: ",
    language_scope="Synthetic acceptance fixture",
    approximate_download_gb=0,
)


class _AcceptanceFixtureEmbedder:
    def __init__(self, spec: EmbeddingModelSpec = _ACCEPTANCE_EMBEDDING_SPEC) -> None:
        self.spec = spec

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _text in texts]

    def embed_query(self, _text: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]


def _fixture_archive(tmp_path: Path) -> tuple[ArchiveLayout, Path]:
    root = tmp_path / "meta-backup-acceptance"
    source_file = root / "package-a" / "your_facebook_activity" / "messages" / "fixture.json"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("{}", encoding="utf-8")
    layout = ArchiveLayout(root)
    layout.ensure_runtime_directories()
    # Acceptance starts against a pre-existing read-only Lance store. Initializing it here is
    # fixture setup, before any byte-exact tree snapshots or acceptance app construction.
    assert lancedb.connect(str(layout.vectors)).list_tables().tables == []
    run_migrations(layout.database)
    engine = create_catalog_engine(layout.database)
    scan_archive(layout, engine, hash_files=True)
    with Session(engine) as session, session.begin():
        session.add_all(
            [
                Conversation(
                    id="fixture-conversation",
                    conversation_key="fixture-conversation-key",
                    title_raw="Original fixture title",
                    title_display="Original fixture title",
                    title_search="original fixture title",
                    conversation_type="direct",
                    primary_category="inbox",
                ),
                ConflictAggregateRun(
                    id="stale-conflict-run",
                    active=True,
                    status="ready",
                    pipeline_version="conflict-aggregate-v1",
                    model_name="stale-model",
                    model_revision="stale-revision",
                    corpus_revision="a" * 64,
                    source_revision="b" * 64,
                    configuration_sha256="c" * 64,
                    configuration_json={},
                    provenance_json={},
                    warnings_json={},
                ),
            ]
        )
    engine.dispose()
    # A read-only server must tolerate and preserve imperfect permissions, and analytics must not
    # recreate its cache directory or lock file.
    layout.catalog.chmod(0o755)
    layout.analytics_cache.rmdir()
    return layout, source_file


def _tree_snapshot(root: Path) -> dict[str, tuple[str, int, int, str | None]]:
    snapshot: dict[str, tuple[str, int, int, str | None]] = {}
    for current_root, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(current_root)
        for name in sorted(directory_names):
            path = current / name
            value = path.lstat()
            kind = "symlink" if stat.S_ISLNK(value.st_mode) else "directory"
            snapshot[path.relative_to(root).as_posix()] = (
                kind,
                stat.S_IMODE(value.st_mode),
                value.st_size,
                None,
            )
        for name in sorted(file_names):
            path = current / name
            value = path.lstat()
            if stat.S_ISREG(value.st_mode):
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                kind = "file"
            else:
                digest = None
                kind = "symlink" if stat.S_ISLNK(value.st_mode) else "other"
            snapshot[path.relative_to(root).as_posix()] = (
                kind,
                stat.S_IMODE(value.st_mode),
                value.st_size,
                digest,
            )
    return snapshot


def _forbidden(*_args: object, **_kwargs: object) -> Any:
    raise AssertionError("acceptance read-only mode called a mutating/cache helper")


def _activate_fixture_semantic_config(layout: ArchiveLayout) -> None:
    engine = create_catalog_engine(layout.database)
    with Session(engine) as session, session.begin():
        session.add(
            SemanticIndexConfig(
                id="d" * 64,
                active=True,
                status="ready",
                backend="fixture",
                backend_version="fixture-v1",
                model_name=_ACCEPTANCE_EMBEDDING_SPEC.model_name,
                model_revision=_ACCEPTANCE_EMBEDDING_SPEC.model_revision,
                dimensions=_ACCEPTANCE_EMBEDDING_SPEC.dimensions,
                distance_metric="cosine",
                document_prefix=_ACCEPTANCE_EMBEDDING_SPEC.document_prefix,
                query_prefix=_ACCEPTANCE_EMBEDDING_SPEC.query_prefix,
                vector_table="fixture_semantic_vectors",
                window_version="fixture-window-v1",
                window_count=0,
                audit_json={
                    "artifact_fingerprint": "declared:fixture-v1",
                    "embedding_implementation": embedding_implementation_identity(
                        _ACCEPTANCE_EMBEDDING_SPEC
                    ),
                },
            )
        )
    engine.dispose()


def _add_acceptance_rollups(layout: ArchiveLayout) -> tuple[int, int, int]:
    timestamps = (
        1_672_531_200_000,
        1_672_617_600_000,
        1_672_704_000_000,
    )
    engine = create_catalog_engine(layout.database)
    with Session(engine) as session, session.begin():
        for index, timestamp_ms in enumerate(timestamps):
            sample_count = index + 1
            session.add(
                NlpHourlyRollup(
                    id=f"acceptance-nlp-rollup-{index}",
                    task="sentiment",
                    hour_start_ms=timestamp_ms,
                    direction="owner",
                    pipeline_version="fixture-nlp-v1",
                    model_name="fixture-sentiment",
                    model_revision="fixture-revision",
                    configuration_sha256="e" * 64,
                    configuration_json={"time_bucket": "UTC hour"},
                    predicted_label="positive",
                    sample_count=sample_count,
                    confidence_sum=0.8 * sample_count,
                    probability_sums_json={
                        "labels": {
                            "positive": 0.7 * sample_count,
                            "negative": 0.3 * sample_count,
                        }
                    },
                    active=True,
                    generated_at=datetime.now(UTC),
                )
            )
            session.add(
                ConflictHourlyRollup(
                    id=f"acceptance-conflict-rollup-{index}",
                    run_id="stale-conflict-run",
                    hour_start_ms=timestamp_ms,
                    sample_count=sample_count,
                    direct_conflict_count=index % 2,
                    active=True,
                    generated_at=datetime.now(UTC),
                )
            )
    with engine.connect() as connection:
        busy, _log_frames, _checkpointed_frames = connection.exec_driver_sql(
            "PRAGMA wal_checkpoint(TRUNCATE)"
        ).one()
    assert busy == 0
    engine.dispose()
    return timestamps


def test_acceptance_server_gets_leave_catalog_and_mutable_tree_byte_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, _source_file = _fixture_archive(tmp_path)
    sqlite_before = hashlib.sha256(layout.database.read_bytes()).hexdigest()
    tree_before = _tree_snapshot(layout.root)

    monkeypatch.setattr(ArchiveLayout, "ensure_runtime_directories", _forbidden)
    monkeypatch.setattr(main_module, "run_migrations", _forbidden)
    monkeypatch.setattr(main_module, "recover_pending_hard_purges", _forbidden)
    monkeypatch.setattr(analytics_routes, "analytics_source_revision", _forbidden)
    monkeypatch.setattr(analytics_routes, "read_analytics_cache", _forbidden)
    monkeypatch.setattr(analytics_routes, "write_analytics_cache", _forbidden)

    app = create_app(
        layout.root,
        access_token="a" * 32,
        security_disabled=True,
        acceptance_read_only=True,
    )
    assert app.state.acceptance_read_only is True
    with app.state.engine.connect() as connection:
        assert connection.exec_driver_sql("PRAGMA query_only").scalar_one() == 1
        assert connection.exec_driver_sql("PRAGMA cache_size").scalar_one() == -262_144
        with pytest.raises(OperationalError, match=r"readonly|read-only|write"):
            connection.execute(
                update(Conversation)
                .where(Conversation.id == "fixture-conversation")
                .values(title_display="forbidden")
            )

    with TestClient(app) as client:
        identity = client.get("/api/v1/acceptance/identity")
        overview = client.get("/api/v1/overview")
        analytics = client.get("/api/v1/analytics/overview")
        messages = client.get("/api/v1/analytics/messages")
        conflict_rate = client.get("/api/v1/analytics/conflict-rate")
        enrichment = client.get("/api/v1/enrichment/status")
        conflict = client.get("/api/v1/enrichment/conflict/status")
        activity = client.get("/api/v1/activity", params={"limit": 1})
        stars = client.get("/api/v1/stars", params={"limit": 1})
        browser = client.get("/")
        head = client.head("/api/v1/acceptance/identity")
        mutations = [
            client.post("/api/v1/search/index", json={}),
            client.patch("/api/v1/settings", json={"timezone": "UTC"}),
            client.delete("/api/v1/saved-searches/missing"),
            client.put("/api/v1/stars/message/fixture-message"),
            client.delete("/api/v1/stars/message/fixture-message"),
            client.options("/api/v1/health"),
        ]
        with Session(app.state.engine) as session:
            stale_run = session.get(ConflictAggregateRun, "stale-conflict-run")
            assert stale_run is not None
            assert (stale_run.status, stale_run.active, stale_run.error_text) == (
                "ready",
                True,
                None,
            )

    assert identity.status_code == 200
    assert overview.status_code == 200
    assert analytics.status_code == 200
    assert messages.status_code == 200
    assert conflict_rate.status_code == 200
    assert enrichment.status_code == 200
    assert conflict.status_code == 200
    assert activity.status_code == 200
    assert stars.status_code == 200
    assert stars.json()["page"]["total"] == 0
    assert browser.status_code == 200
    assert head.status_code == 200
    assert all(response.status_code == 405 for response in mutations)
    assert all(response.headers["allow"] == "GET, HEAD" for response in mutations)
    assert all(response.headers["cache-control"] == "no-store" for response in mutations)
    assert not layout.analytics_cache.exists()
    assert not (layout.catalog / "analytics-cache.lock").exists()
    assert hashlib.sha256(layout.database.read_bytes()).hexdigest() == sqlite_before
    assert _tree_snapshot(layout.root) == tree_before


def test_acceptance_rollup_invariants_reuse_without_skipping_cursor_rows_or_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, _source_file = _fixture_archive(tmp_path)
    first_ms, second_ms, third_ms = _add_acceptance_rollups(layout)
    tree_before = _tree_snapshot(layout.root)
    database_before = hashlib.sha256(layout.database.read_bytes()).hexdigest()
    calls = {
        "nlp_rollup": 0,
        "derived_nlp": 0,
        "conflict_status": 0,
        "conflict_rollup": 0,
    }
    original_nlp_rollup = analytics_routes._load_nlp_rollup_invariants
    original_derived_nlp = analytics_routes._load_derived_nlp_status
    original_conflict_rollup = analytics_routes._load_conflict_rollup_invariants

    def counted_nlp_rollup(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls["nlp_rollup"] += 1
        return original_nlp_rollup(*args, **kwargs)

    def counted_derived_nlp(*args: Any, **kwargs: Any) -> dict[str, dict[str, Any]]:
        calls["derived_nlp"] += 1
        return original_derived_nlp(*args, **kwargs)

    def counted_conflict_rollup(*args: Any, **kwargs: Any) -> dict[str, int]:
        calls["conflict_rollup"] += 1
        return original_conflict_rollup(*args, **kwargs)

    def fixture_conflict_status(_engine: Any, *, refresh: bool) -> dict[str, Any]:
        calls["conflict_status"] += 1
        assert refresh is False
        return {
            "available": True,
            "status": "ready",
            "reason": None,
            "gate": {"aggregation_enabled": True},
            "run": {
                "id": "stale-conflict-run",
                "status": "ready",
                "active": True,
                "pipeline_version": "conflict-aggregate-v1",
                "model_name": "fixture-conflict-model",
                "model_revision": "fixture-conflict-revision",
                "calibration_id": None,
                "semantic_config_id": None,
                "corpus_revision": "a" * 64,
                "source_revision": "b" * 64,
                "configuration_sha256": "c" * 64,
                "window_count": 6,
                "scored_count": 6,
                "direct_conflict_count": 1,
                "warnings": [],
                "completed_at": None,
            },
            "job": None,
            "warnings": [],
            "relationship_ranking_enabled": False,
            "relationship_dimensions": [],
        }

    monkeypatch.setattr(
        analytics_routes,
        "_load_nlp_rollup_invariants",
        counted_nlp_rollup,
    )
    monkeypatch.setattr(analytics_routes, "_load_derived_nlp_status", counted_derived_nlp)
    monkeypatch.setattr(
        analytics_routes,
        "_load_conflict_rollup_invariants",
        counted_conflict_rollup,
    )
    monkeypatch.setattr(analytics_routes, "conflict_aggregate_status", fixture_conflict_status)

    app = create_app(layout.root, security_disabled=True, acceptance_read_only=True)
    with TestClient(app) as client:
        nlp_parameters = {
            "bucket": "day",
            "timezone": "UTC",
            "direction": "all",
            "limit": 1,
        }
        first_nlp = client.get("/api/v1/analytics/nlp-trends", params=nlp_parameters)
        assert first_nlp.status_code == 200, first_nlp.text
        first_nlp_payload = first_nlp.json()
        nlp_cursor = first_nlp_payload["page"]["next_cursor"]
        assert nlp_cursor
        second_nlp = client.get(
            "/api/v1/analytics/nlp-trends",
            params={**nlp_parameters, "cursor": nlp_cursor},
        )
        assert second_nlp.status_code == 200, second_nlp.text
        third_nlp_cursor = second_nlp.json()["page"]["next_cursor"]
        assert third_nlp_cursor
        third_nlp = client.get(
            "/api/v1/analytics/nlp-trends",
            params={**nlp_parameters, "cursor": third_nlp_cursor},
        )
        assert third_nlp.status_code == 200, third_nlp.text
        assert third_nlp.json()["page"]["next_cursor"] is None
        assert first_nlp_payload["source_sample_count"] == 6
        assert second_nlp.json()["source_sample_count"] == 6
        assert third_nlp.json()["source_sample_count"] == 6
        assert (
            sum(
                payload["page_sample_count"]
                for payload in (
                    first_nlp_payload,
                    second_nlp.json(),
                    third_nlp.json(),
                )
            )
            == 6
        )
        assert {
            row["period"]
            for payload in (
                first_nlp_payload,
                second_nlp.json(),
                third_nlp.json(),
            )
            for row in payload["series"]["sentiment"]
        } == {"2023-01-01", "2023-01-02", "2023-01-03"}
        assert (
            first_nlp_payload["series"]["sentiment"][0]["period"]
            != second_nlp.json()["series"]["sentiment"][0]["period"]
        )

        cross_scope_nlp = client.get(
            "/api/v1/analytics/nlp-trends",
            params={
                "bucket": "month",
                "timezone": "America/New_York",
                "direction": "all",
                "limit": 1,
            },
        )
        assert cross_scope_nlp.status_code == 200, cross_scope_nlp.text
        assert cross_scope_nlp.json()["source_sample_count"] == 6
        filtered_nlp = client.get(
            "/api/v1/analytics/nlp-trends",
            params={
                **nlp_parameters,
                "from_ms": second_ms,
                "to_ms": third_ms,
            },
        )
        assert filtered_nlp.status_code == 200, filtered_nlp.text
        assert filtered_nlp.json()["source_sample_count"] == 5

        conflict_parameters = {"bucket": "day", "timezone": "UTC", "limit": 1}
        first_conflict = client.get(
            "/api/v1/analytics/conflict-rate",
            params=conflict_parameters,
        )
        assert first_conflict.status_code == 200, first_conflict.text
        first_conflict_payload = first_conflict.json()
        conflict_cursor = first_conflict_payload["page"]["next_cursor"]
        assert conflict_cursor
        second_conflict = client.get(
            "/api/v1/analytics/conflict-rate",
            params={**conflict_parameters, "cursor": conflict_cursor},
        )
        assert second_conflict.status_code == 200, second_conflict.text
        third_conflict_cursor = second_conflict.json()["page"]["next_cursor"]
        assert third_conflict_cursor
        third_conflict = client.get(
            "/api/v1/analytics/conflict-rate",
            params={**conflict_parameters, "cursor": third_conflict_cursor},
        )
        assert third_conflict.status_code == 200, third_conflict.text
        assert third_conflict.json()["page"]["next_cursor"] is None
        assert first_conflict_payload["summary"] == second_conflict.json()["summary"]
        assert first_conflict_payload["summary"] == third_conflict.json()["summary"]
        assert (
            sum(
                payload["series"][0]["sample_count"]
                for payload in (
                    first_conflict_payload,
                    second_conflict.json(),
                    third_conflict.json(),
                )
            )
            == 6
        )
        assert {
            payload["series"][0]["period"]
            for payload in (
                first_conflict_payload,
                second_conflict.json(),
                third_conflict.json(),
            )
        } == {"2023-01-01", "2023-01-02", "2023-01-03"}
        assert (
            first_conflict_payload["series"][0]["period"]
            != second_conflict.json()["series"][0]["period"]
        )
        cross_scope_conflict = client.get(
            "/api/v1/analytics/conflict-rate",
            params={"bucket": "month", "timezone": "America/New_York", "limit": 1},
        )
        assert cross_scope_conflict.status_code == 200, cross_scope_conflict.text
        assert cross_scope_conflict.json()["summary"] == first_conflict_payload["summary"]
        filtered_conflict = client.get(
            "/api/v1/analytics/conflict-rate",
            params={
                **conflict_parameters,
                "from_ms": second_ms,
                "to_ms": third_ms,
            },
        )
        assert filtered_conflict.status_code == 200, filtered_conflict.text
        assert filtered_conflict.json()["summary"]["sample_count"] == 5

        request = Request({"type": "http", "app": app})
        with Session(app.state.engine) as session:
            other_run = analytics_routes._conflict_rollup_invariants(
                request,
                session,
                run_id="different-run",
                from_ms=None,
                to_ms=None,
            )
        assert other_run == {"sample_count": 0, "direct_count": 0}

        loader_started = threading.Event()
        release_loader = threading.Event()
        loader_count = 0
        loader_count_lock = threading.Lock()

        def concurrent_loader() -> dict[str, list[str]]:
            nonlocal loader_count
            with loader_count_lock:
                loader_count += 1
            loader_started.set()
            assert release_loader.wait(timeout=5)
            return {"values": []}

        def concurrent_snapshot() -> dict[str, list[str]]:
            return analytics_routes._acceptance_invariant_snapshot(
                request,
                "concurrent-deep-copy-proof",
                {"scope": "same-key"},
                concurrent_loader,
            )

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(concurrent_snapshot) for _index in range(4)]
            assert loader_started.wait(timeout=5)
            release_loader.set()
            concurrent_results = [future.result(timeout=5) for future in futures]
        assert loader_count == 1
        concurrent_results[0]["values"].append("caller-local-mutation")
        assert all(result["values"] == [] for result in concurrent_results[1:])
        cached_result = analytics_routes._acceptance_invariant_snapshot(
            request,
            "concurrent-deep-copy-proof",
            {"scope": "same-key"},
            _forbidden,
        )
        assert cached_result == {"values": []}

    assert calls == {
        "nlp_rollup": 2,
        "derived_nlp": 1,
        "conflict_status": 1,
        "conflict_rollup": 3,
    }
    assert not layout.analytics_cache.exists()
    assert not (layout.catalog / "analytics-cache.lock").exists()
    assert hashlib.sha256(layout.database.read_bytes()).hexdigest() == database_before
    assert _tree_snapshot(layout.root) == tree_before

    second_app = create_app(layout.root, security_disabled=True, acceptance_read_only=True)
    with TestClient(second_app) as second_client:
        assert (
            second_client.get(
                "/api/v1/analytics/nlp-trends",
                params={
                    "bucket": "month",
                    "timezone": "UTC",
                    "direction": "all",
                },
            ).status_code
            == 200
        )
        assert second_client.get("/api/v1/analytics/conflict-rate").status_code == 200
    assert calls == {
        "nlp_rollup": 3,
        "derived_nlp": 2,
        "conflict_status": 2,
        "conflict_rollup": 4,
    }
    assert hashlib.sha256(layout.database.read_bytes()).hexdigest() == database_before
    assert _tree_snapshot(layout.root) == tree_before
    assert first_ms < second_ms < third_ms


def test_acceptance_identity_is_private_safe_and_changes_on_same_count_edits(
    tmp_path: Path,
) -> None:
    layout, source_file = _fixture_archive(tmp_path)

    first_app = create_app(layout.root, security_disabled=True, acceptance_read_only=True)
    with TestClient(first_app) as client:
        first = client.get("/api/v1/acceptance/identity").json()

    assert set(first) == {
        "catalog_source_revision",
        "backend_version",
        "backend_source_sha256",
        "frontend_manifest_sha256",
        "server_nonce",
    }
    assert first["backend_version"] == __version__
    for key in (
        "catalog_source_revision",
        "backend_source_sha256",
        "frontend_manifest_sha256",
        "server_nonce",
    ):
        assert _SHA256.fullmatch(first[key])
    rendered = str(first)
    assert str(layout.root) not in rendered
    assert "Original fixture title" not in rendered

    write_engine = create_catalog_engine(layout.database)
    with Session(write_engine) as session, session.begin():
        conversation = session.get(Conversation, "fixture-conversation")
        assert conversation is not None
        conversation.title_display = "Same count, different normalized value"
    write_engine.dispose()
    second_app = create_app(layout.root, security_disabled=True, acceptance_read_only=True)
    with TestClient(second_app) as client:
        second = client.get("/api/v1/acceptance/identity").json()
    assert second["catalog_source_revision"] != first["catalog_source_revision"]
    assert second["backend_source_sha256"] == first["backend_source_sha256"]
    assert second["frontend_manifest_sha256"] == first["frontend_manifest_sha256"]
    assert second["server_nonce"] != first["server_nonce"]

    source_file.write_text("[]", encoding="utf-8")
    third_app = create_app(layout.root, security_disabled=True, acceptance_read_only=True)
    with TestClient(third_app) as client:
        third = client.get("/api/v1/acceptance/identity").json()
    assert third["catalog_source_revision"] != second["catalog_source_revision"]


def test_acceptance_identity_binds_same_count_fts_and_vector_file_edits(
    tmp_path: Path,
) -> None:
    layout, _source_file = _fixture_archive(tmp_path)
    vector_file = layout.vectors / "fixture-index.lance" / "data" / "segment.bin"
    vector_file.parent.mkdir(parents=True)
    vector_file.write_bytes(b"vector-version-a")
    setup_engine = create_catalog_engine(layout.database)
    with setup_engine.begin() as connection:
        connection.exec_driver_sql(
            "INSERT INTO message_fts(rowid, sender_search, content_search, transcription_search) "
            "VALUES (999, 'sender', 'first indexed value', '')"
        )
    setup_engine.dispose()

    first_app = create_app(layout.root, security_disabled=True, acceptance_read_only=True)
    with TestClient(first_app) as client:
        first = client.get("/api/v1/acceptance/identity").json()

    engine = create_catalog_engine(layout.database)
    with engine.begin() as connection:
        fts_id, original_block = connection.exec_driver_sql(
            "SELECT id, block FROM message_fts_data WHERE length(block) > 0 ORDER BY id LIMIT 1"
        ).one()
        changed_block = bytearray(original_block)
        changed_block[-1] ^= 1
        connection.exec_driver_sql(
            "UPDATE message_fts_data SET block = ? WHERE id = ?",
            (bytes(changed_block), fts_id),
        )
        assert connection.exec_driver_sql("SELECT count(*) FROM message_fts_data").scalar_one() > 0
    engine.dispose()
    second_app = create_app(layout.root, security_disabled=True, acceptance_read_only=True)
    with TestClient(second_app) as client:
        second = client.get("/api/v1/acceptance/identity").json()
    assert second["catalog_source_revision"] != first["catalog_source_revision"]

    vector_file.write_bytes(b"vector-version-b")
    third_app = create_app(layout.root, security_disabled=True, acceptance_read_only=True)
    with TestClient(third_app) as client:
        third = client.get("/api/v1/acceptance/identity").json()
    assert third["catalog_source_revision"] != second["catalog_source_revision"]


def test_acceptance_identity_recomputes_vector_revision_in_same_app(
    tmp_path: Path,
) -> None:
    layout, _source_file = _fixture_archive(tmp_path)
    vector_file = layout.vectors / "fixture-index.lance" / "data" / "segment.bin"
    vector_file.parent.mkdir(parents=True)
    vector_file.write_bytes(b"vector-version-a")

    app = create_app(layout.root, security_disabled=True, acceptance_read_only=True)
    with TestClient(app) as client:
        first = client.get("/api/v1/acceptance/identity")
        assert first.status_code == 200
        first_payload = first.json()

        original_size = vector_file.stat().st_size
        vector_file.write_bytes(b"vector-version-b")
        assert vector_file.stat().st_size == original_size

        second = client.get("/api/v1/acceptance/identity")
        assert second.status_code == 200
        second_payload = second.json()

    assert second_payload["catalog_source_revision"] != first_payload["catalog_source_revision"]
    assert second_payload["server_nonce"] == first_payload["server_nonce"]
    assert second_payload["backend_source_sha256"] == first_payload["backend_source_sha256"]
    assert second_payload["frontend_manifest_sha256"] == first_payload["frontend_manifest_sha256"]


def test_acceptance_identity_fails_closed_when_sqlite_sidecar_appears_after_startup(
    tmp_path: Path,
) -> None:
    layout, _source_file = _fixture_archive(tmp_path)
    app = create_app(layout.root, security_disabled=True, acceptance_read_only=True)

    with TestClient(app, raise_server_exceptions=False) as client:
        baseline = client.get("/api/v1/acceptance/identity")
        assert baseline.status_code == 200

        Path(f"{layout.database}-wal").write_bytes(b"appeared-after-startup")
        rejected = client.get("/api/v1/acceptance/identity")

    assert rejected.status_code == 500
    assert str(layout.root) not in rejected.text
    assert "appeared-after-startup" not in rejected.text


def test_acceptance_engine_remains_bound_when_database_path_is_replaced(
    tmp_path: Path,
) -> None:
    layout, _source_file = _fixture_archive(tmp_path)
    replacement = tmp_path / "replacement.sqlite"
    replacement.write_bytes(layout.database.read_bytes())
    replacement_engine = create_catalog_engine(replacement)
    with Session(replacement_engine) as session, session.begin():
        conversation = session.get(Conversation, "fixture-conversation")
        assert conversation is not None
        conversation.title_display = "Replacement database title"
    replacement_engine.dispose()

    app = create_app(layout.root, security_disabled=True, acceptance_read_only=True)
    with TestClient(app, raise_server_exceptions=False) as client:
        first = client.get("/api/v1/acceptance/identity")
        assert first.status_code == 200

        # Atomically replace the pathname while the engine retains its open descriptor to the
        # unlinked original inode.
        os.replace(replacement, layout.database)

        with Session(app.state.engine) as session:
            bound_title = session.get(Conversation, "fixture-conversation")
            assert bound_title is not None
            assert bound_title.title_display == "Original fixture title"
        with sqlite3.connect(
            f"file:{layout.database}?mode=ro&immutable=1",
            uri=True,
        ) as path_connection:
            pathname_title = path_connection.execute(
                "SELECT title_display FROM conversations WHERE id = ?",
                ("fixture-conversation",),
            ).fetchone()
        assert pathname_title == ("Replacement database title",)

        rejected_identity = client.get("/api/v1/acceptance/identity")
        # A fresh checkout after dropping the connection bound before replacement may fail to
        # reopen an unlinked inode, but it must never fall through to the new pathname target.
        app.state.engine.pool.dispose()
        with (
            pytest.raises(OperationalError, match="unable to open database file"),
            Session(app.state.engine) as session,
        ):
            session.get(Conversation, "fixture-conversation")

    assert rejected_identity.status_code == 500
    assert str(layout.root) not in rejected_identity.text
    assert "Replacement database title" not in rejected_identity.text


def test_read_only_catalog_engine_requires_owning_archive_root(tmp_path: Path) -> None:
    layout, _source_file = _fixture_archive(tmp_path)

    with pytest.raises(CatalogMigrationError, match="requires the owning archive root"):
        create_catalog_engine(layout.database, read_only=True)


def test_acceptance_engine_uses_queue_pool_for_more_than_five_concurrent_reads(
    tmp_path: Path,
) -> None:
    layout, _source_file = _fixture_archive(tmp_path)
    app = create_app(layout.root, security_disabled=True, acceptance_read_only=True)
    assert isinstance(app.state.engine.pool, QueuePool)

    worker_count = 8
    database_barrier = threading.Barrier(worker_count)

    def concurrent_database_read() -> str:
        with app.state.engine.connect() as connection:
            database_barrier.wait(timeout=10)
            return str(
                connection.exec_driver_sql(
                    "SELECT title_display FROM conversations WHERE id = ?",
                    ("fixture-conversation",),
                ).scalar_one()
            )

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        database_results = list(executor.map(lambda _index: concurrent_database_read(), range(8)))
    assert database_results == ["Original fixture title"] * worker_count

    with TestClient(app) as client:
        api_barrier = threading.Barrier(worker_count)

        def concurrent_api_read() -> tuple[int, str]:
            api_barrier.wait(timeout=10)
            response = client.get("/api/v1/overview")
            return response.status_code, str(response.json()["archive"]["root_name"])

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            api_results = list(executor.map(lambda _index: concurrent_api_read(), range(8)))

    assert api_results == [(200, layout.root.name)] * worker_count


def test_acceptance_identity_requires_an_authenticated_session(tmp_path: Path) -> None:
    layout, _source_file = _fixture_archive(tmp_path)
    access_token = "acceptance-read-only-access-token"
    app = create_app(
        layout.root,
        access_token=access_token,
        acceptance_read_only=True,
    )

    with TestClient(app, base_url="http://127.0.0.1") as client:
        unauthenticated = client.get("/api/v1/acceptance/identity")
        authentication = client.get(
            "/auth",
            params={"token": access_token},
            follow_redirects=False,
        )
        authenticated = client.get("/api/v1/acceptance/identity")

    assert unauthenticated.status_code == 401
    assert authentication.status_code == 303
    assert authenticated.status_code == 200
    assert _SHA256.fullmatch(authenticated.json()["catalog_source_revision"])


@pytest.mark.parametrize("symlink_location", ["leaf", "intermediate"])
def test_integrity_get_helpers_do_not_follow_report_symlinks_outside_archive(
    tmp_path: Path,
    symlink_location: str,
) -> None:
    layout, _source_file = _fixture_archive(tmp_path)
    outside = tmp_path / "outside-reports"
    outside.mkdir()
    outside_report = outside / "latest.json"
    outside_report.write_text(
        '{"checks":[{"status":"fail","detail":"outside sentinel"}]}',
        encoding="utf-8",
    )
    app = create_app(layout.root, security_disabled=True, acceptance_read_only=True)
    reports = layout.manifests / "integrity-reports"
    if symlink_location == "leaf":
        (reports / "latest.json").symlink_to(outside_report)
    else:
        reports.rmdir()
        reports.symlink_to(outside, target_is_directory=True)

    with TestClient(app) as client:
        integrity_response = client.get("/api/v1/integrity")
        overview_response = client.get("/api/v1/overview")

    assert integrity_response.status_code == 200
    assert integrity_response.json()["checks"] == []
    assert overview_response.status_code == 200
    assert overview_response.json()["lifecycle"]["state"] == "partial"
    assert "outside sentinel" not in integrity_response.text
    assert "outside sentinel" not in overview_response.text


def test_acceptance_identity_refuses_vector_intermediate_symlink(
    tmp_path: Path,
) -> None:
    layout, _source_file = _fixture_archive(tmp_path)
    data_directory = layout.vectors / "fixture-index.lance" / "data"
    data_directory.mkdir(parents=True)
    (data_directory / "segment.bin").write_bytes(b"inside-vector-data")
    outside = tmp_path / "outside-vector-data"
    outside.mkdir()
    (outside / "sentinel.bin").write_bytes(b"outside-vector-sentinel")

    app = create_app(layout.root, security_disabled=True, acceptance_read_only=True)
    with TestClient(app, raise_server_exceptions=False) as client:
        baseline = client.get("/api/v1/acceptance/identity")
        assert baseline.status_code == 200

        retained = data_directory.with_name("data-retained")
        data_directory.rename(retained)
        data_directory.symlink_to(outside, target_is_directory=True)
        rejected = client.get("/api/v1/acceptance/identity")

    assert rejected.status_code == 500
    assert str(outside) not in rejected.text
    assert "outside-vector-sentinel" not in rejected.text


@pytest.mark.parametrize("symlink_location", ["leaf", "intermediate"])
def test_acceptance_startup_refuses_unsafe_vector_descendants(
    tmp_path: Path,
    symlink_location: str,
) -> None:
    layout, _source_file = _fixture_archive(tmp_path)
    outside = tmp_path / "outside-vector-startup"
    outside.mkdir()
    outside_file = outside / "sentinel.bin"
    outside_file.write_bytes(b"outside-startup-sentinel")

    if symlink_location == "leaf":
        (layout.vectors / "unsafe.bin").symlink_to(outside_file)
    else:
        index_directory = layout.vectors / "fixture-index.lance"
        index_directory.mkdir()
        (index_directory / "data").symlink_to(outside, target_is_directory=True)

    with pytest.raises(
        CatalogMigrationError,
        match="safe Lance manifest",
    ):
        create_app(layout.root, security_disabled=True, acceptance_read_only=True)


def test_acceptance_startup_refuses_empty_precreated_lance_manifest(tmp_path: Path) -> None:
    layout, _source_file = _fixture_archive(tmp_path)
    manifest = layout.vectors / "__manifest"
    manifest.rename(layout.vectors / "populated-manifest-retained")
    manifest.mkdir()
    before = _tree_snapshot(layout.root)

    with pytest.raises(
        CatalogMigrationError,
        match="safe Lance manifest",
    ):
        create_app(layout.root, security_disabled=True, acceptance_read_only=True)

    assert _tree_snapshot(layout.root) == before


def test_acceptance_startup_refuses_missing_referenced_lance_manifest(tmp_path: Path) -> None:
    layout, _source_file = _fixture_archive(tmp_path)
    versions = layout.vectors / "__manifest" / "_versions"
    hint = json.loads((versions / "latest_version_hint.json").read_text(encoding="utf-8"))
    referenced_manifest = versions / f"{(1 << 64) - 1 - int(hint['version'])}.manifest"
    assert referenced_manifest.is_file()
    referenced_manifest.rename(referenced_manifest.with_suffix(".manifest-retained"))
    before = _tree_snapshot(layout.root)

    with pytest.raises(
        CatalogMigrationError,
        match="safe Lance manifest",
    ):
        create_app(layout.root, security_disabled=True, acceptance_read_only=True)

    assert _tree_snapshot(layout.root) == before


@pytest.mark.parametrize("replacement_kind", ["empty", "empty_manifest"])
def test_acceptance_semantic_get_does_not_initialize_replaced_vector_tree(
    tmp_path: Path,
    replacement_kind: str,
) -> None:
    layout, _source_file = _fixture_archive(tmp_path)
    _activate_fixture_semantic_config(layout)
    app = create_app(layout.root, security_disabled=True, acceptance_read_only=True)
    app.state.search_embedder_factory = _AcceptanceFixtureEmbedder

    retained_vectors = layout.catalog / "vectors-retained"
    layout.vectors.rename(retained_vectors)
    layout.vectors.mkdir()
    if replacement_kind == "empty_manifest":
        (layout.vectors / "__manifest").mkdir()
    before = _tree_snapshot(layout.vectors)

    with TestClient(app) as client:
        response = client.get(
            "/api/v1/search",
            params={"q": "fixture", "mode": "semantic"},
        )

    assert response.status_code == 409
    assert _tree_snapshot(layout.vectors) == before
    assert not (layout.vectors / "__manifest" / "_versions").exists()


@pytest.mark.parametrize("symlink_location", ["leaf", "intermediate"])
def test_spa_index_symlinks_are_never_followed_or_served(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    symlink_location: str,
) -> None:
    layout, _source_file = _fixture_archive(tmp_path)
    fake_project = tmp_path / "fake-project"
    fake_module = fake_project / "src" / "meta_archive" / "api" / "main.py"
    static_root = fake_project / "web" / "dist"
    outside = tmp_path / "outside-static"
    outside.mkdir()
    outside_index = outside / "index.html"
    outside_index.write_text("outside SPA sentinel", encoding="utf-8")
    monkeypatch.setattr(main_module, "__file__", str(fake_module))
    static_root.mkdir(parents=True)
    index = static_root / "index.html"
    index.write_text("safe SPA fixture", encoding="utf-8")
    app = create_app(layout.root, security_disabled=True, acceptance_read_only=True)

    if symlink_location == "leaf":
        index.unlink()
        index.symlink_to(outside_index)
    else:
        static_root.rename(static_root.with_name("dist-retained"))
        static_root.symlink_to(outside, target_is_directory=True)

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code != 200
    assert "outside SPA sentinel" not in response.text


def test_acceptance_asset_root_replacement_symlink_is_not_followed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, _source_file = _fixture_archive(tmp_path)
    fake_project = tmp_path / "fake-project"
    fake_module = fake_project / "src" / "meta_archive" / "api" / "main.py"
    static_root = fake_project / "web" / "dist"
    assets = static_root / "assets"
    assets.mkdir(parents=True)
    (static_root / "index.html").write_text("safe SPA fixture", encoding="utf-8")
    (assets / "safe.css").write_text("body {}", encoding="utf-8")
    monkeypatch.setattr(main_module, "__file__", str(fake_module))
    app = create_app(layout.root, security_disabled=True, acceptance_read_only=True)

    outside = tmp_path / "outside-assets"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("outside asset sentinel", encoding="utf-8")
    assets.rename(static_root / "assets-retained")
    assets.symlink_to(outside, target_is_directory=True)

    with TestClient(app) as client:
        response = client.get("/assets/sentinel.txt")

    assert response.status_code != 200
    assert "outside asset sentinel" not in response.text


def test_backend_identity_refuses_a_symlinked_source_subtree(tmp_path: Path) -> None:
    project = tmp_path / "project"
    package = project / "src" / "meta_archive"
    package.mkdir(parents=True)
    outside = tmp_path / "outside-source"
    outside.mkdir()
    (outside / "sentinel.py").write_text("PRIVATE_OUTSIDE_SENTINEL = True\n", encoding="utf-8")
    (package / "omitted").symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="unsafe entry"):
        backend_source_digest(project)


def test_acceptance_server_refuses_stale_migration_without_repairing_it(tmp_path: Path) -> None:
    layout, _source_file = _fixture_archive(tmp_path)
    engine = create_catalog_engine(layout.database)
    with engine.begin() as connection:
        connection.execute(text("UPDATE alembic_version SET version_num = 'c4f78a1d2e60'"))
    engine.dispose()
    before = _tree_snapshot(layout.root)

    with pytest.raises(CatalogMigrationError, match="refuses to migrate"):
        create_app(layout.root, security_disabled=True, acceptance_read_only=True)

    assert _tree_snapshot(layout.root) == before
    readonly = create_catalog_engine(
        layout.database,
        read_only=True,
        archive_root=layout.root,
    )
    with readonly.connect() as connection:
        assert connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one() == (
            "c4f78a1d2e60"
        )
    readonly.dispose()


def test_acceptance_server_refuses_uncheckpointed_sqlite_sidecars(tmp_path: Path) -> None:
    layout, _source_file = _fixture_archive(tmp_path)
    sidecar = Path(f"{layout.database}-wal")
    sidecar.write_bytes(b"uncheckpointed-fixture")
    before = _tree_snapshot(layout.root)

    with pytest.raises(CatalogMigrationError, match="fully checkpointed"):
        create_app(layout.root, security_disabled=True, acceptance_read_only=True)

    assert _tree_snapshot(layout.root) == before


def test_acceptance_identity_route_does_not_exist_on_mutable_server(tmp_path: Path) -> None:
    layout, _source_file = _fixture_archive(tmp_path)
    app = create_app(layout.root, security_disabled=True)

    with TestClient(app) as client:
        response = client.get("/api/v1/acceptance/identity")

    assert response.status_code == 404
