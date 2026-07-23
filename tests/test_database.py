from __future__ import annotations

import hashlib
import json
import sqlite3
import stat
from pathlib import Path

import pytest
from alembic.config import Config
from sqlalchemy import Engine, text

from alembic import command
from meta_archive.database import create_catalog_engine, run_migrations
from meta_archive.paths import PRIVATE_FILE_MODE, UnsafeArchivePathError


def test_initial_migration_creates_fts_and_enforces_foreign_keys(
    catalog_engine: Engine,
) -> None:
    with catalog_engine.connect() as connection:
        tables = {
            name
            for (name,) in connection.execute(
                text("SELECT name FROM sqlite_master WHERE type = 'table'")
            )
        }
        foreign_keys = connection.execute(text("PRAGMA foreign_keys")).scalar_one()

    assert "message_fts" in tables
    assert "activity_fts" in tables
    assert "messages" in tables
    assert foreign_keys == 1


def test_catalog_database_and_sqlite_sidecars_are_owner_only(tmp_path: Path) -> None:
    database = tmp_path / "catalog" / "facebook.sqlite"
    run_migrations(database)
    engine = create_catalog_engine(database)
    try:
        with engine.begin() as connection:
            connection.execute(text("CREATE TABLE IF NOT EXISTS privacy_probe(id INTEGER)"))
            connection.execute(text("INSERT INTO privacy_probe VALUES (1)"))
            for path in (database, Path(f"{database}-wal"), Path(f"{database}-shm")):
                assert path.is_file()
                assert stat.S_IMODE(path.lstat().st_mode) == PRIVATE_FILE_MODE
    finally:
        engine.dispose()


def test_catalog_database_rejects_a_symlink_without_touching_target(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    target = tmp_path / "outside.sqlite"
    target.write_bytes(b"outside")
    target.chmod(0o644)
    database = catalog / "facebook.sqlite"
    database.symlink_to(target)

    with pytest.raises(UnsafeArchivePathError, match="SQLite runtime file"):
        create_catalog_engine(database)

    assert target.read_bytes() == b"outside"
    assert stat.S_IMODE(target.lstat().st_mode) == 0o644


def test_timeline_time_indexes_migrate_reversibly_and_drive_keyset_plans(
    tmp_path: Path,
) -> None:
    database = tmp_path / "pre-timeline-time-indexes.sqlite3"
    project_root = Path(__file__).resolve().parents[1]
    config = Config(str(project_root / "alembic.ini"))
    config.set_main_option("script_location", str(project_root / "alembic"))
    config.set_main_option("sqlalchemy.url", f"sqlite+pysqlite:///{database}")
    command.upgrade(config, "6e24b9f0c1d7")

    index_specs = {
        "messages": "ix_messages_deleted_time_id",
        "activity_records": "ix_activity_deleted_time_id",
    }
    with sqlite3.connect(database) as connection:
        for table_name, index_name in index_specs.items():
            indexes = {row[1] for row in connection.execute(f"PRAGMA index_list('{table_name}')")}
            assert index_name not in indexes

    command.upgrade(config, "head")
    with sqlite3.connect(database) as connection:
        for table_name, index_name in index_specs.items():
            columns = [row[2] for row in connection.execute(f"PRAGMA index_info('{index_name}')")]
            assert columns == ["deleted_at", "timestamp_ms", "id"]
            plan = [
                str(row[-1])
                for row in connection.execute(
                    f"""
                    EXPLAIN QUERY PLAN
                    SELECT id, timestamp_ms
                    FROM {table_name}
                    WHERE deleted_at IS NULL
                      AND (timestamp_ms, id) > (?, ?)
                    ORDER BY timestamp_ms ASC NULLS LAST, id ASC
                    LIMIT 251
                    """,
                    (0, "fixture-boundary"),
                )
            ]
            assert any(index_name in step for step in plan)
            assert not any("USE TEMP B-TREE" in step for step in plan)
        message_indexes = {row[1] for row in connection.execute("PRAGMA index_list('messages')")}
        assert "ix_messages_deleted" in message_indexes

    command.downgrade(config, "6e24b9f0c1d7")
    with sqlite3.connect(database) as connection:
        for table_name, index_name in index_specs.items():
            indexes = {row[1] for row in connection.execute(f"PRAGMA index_list('{table_name}')")}
            assert index_name not in indexes
        message_indexes = {row[1] for row in connection.execute("PRAGMA index_list('messages')")}
        assert "ix_messages_deleted" in message_indexes

    command.upgrade(config, "head")


def test_support_message_privacy_migration_scrubs_existing_catalog_rows(
    tmp_path: Path,
) -> None:
    database = tmp_path / "pre-support-message-privacy.sqlite3"
    project_root = Path(__file__).resolve().parents[1]
    config = Config(str(project_root / "alembic.ini"))
    config.set_main_option("script_location", str(project_root / "alembic"))
    config.set_main_option("sqlalchemy.url", f"sqlite+pysqlite:///{database}")
    command.upgrade(config, "9f2a4c7d8b31")

    ip_marker = "198.51.100.legacy-migration-fixture"
    payment_marker = "legacy-payment-migration-fixture"
    legacy_extra = {
        "ip": ip_marker,
        "payment_info": {"card_number": payment_marker},
        "to": "Archive Owner",
        "subject": "Support Case",
        "category": "account support",
        "status": "open",
        "metadata": {
            "shipping": "preserved",
            "topic": "preserved",
            "participant": "Support Agent",
            "network": {"source_ip": ip_marker},
            "history": [
                {"status": "received"},
                {
                    "billingInfo": {"account_credentials": payment_marker},
                    "status": "reviewed",
                },
            ],
        },
    }
    existing_sensitive = {
        "ip": ip_marker,
        "payment_info": {"card_number": payment_marker},
    }
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(
            """
            INSERT INTO import_runs(
                id, kind, status, parser_version, counters_json, checkpoint_json
            ) VALUES ('privacy-import', 'messages', 'completed', 'fixture', '{}', '{}')
            """
        )
        connection.execute(
            """
            INSERT INTO source_chunks(id, chunk_key, original_path, ordinal)
            VALUES (1, 'privacy-chunk', '/fixture/privacy-chunk', 1)
            """
        )
        connection.execute(
            """
            INSERT INTO source_files(
                id, chunk_id, relative_path, normalized_path, category, size_bytes,
                mtime_ns, mode, json_status, scan_status
            ) VALUES (
                1, 1, 'messages/support_messages.json', 'messages/support_messages.json',
                'messages', 1, 1, 384, 'valid', 'catalogued'
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO source_records(
                id, source_file_id, json_pointer, record_type, logical_fingerprint,
                import_run_id
            ) VALUES (?, 1, ?, 'support_message', ?, 'privacy-import')
            """,
            (
                ("support-record", "/support/0", "a" * 64),
                ("ordinary-record", "/ordinary/0", "b" * 64),
            ),
        )
        connection.execute(
            """
            INSERT INTO conversations(
                id, conversation_key, title_raw, title_display, title_search,
                conversation_type, primary_category, participant_count,
                message_count, sent_count, received_count, attachment_count,
                unique_attachment_count, attachment_bytes, active_days, completeness
            ) VALUES (
                'privacy-conversation', 'privacy-conversation', 'Fixture', 'Fixture',
                'fixture', 'system', 'support', 1, 2, 0, 2, 0, 0, 0, 1, 'complete'
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO conversation_segments(
                id, conversation_id, thread_path, title_raw, title_display,
                title_search, source_category, sequence, source_chunk_count,
                completeness, metadata_json
            ) VALUES (?, 'privacy-conversation', ?, 'Fixture', 'Fixture', 'fixture', ?, 1, 1,
                      'complete', '{}')
            """,
            (
                ("support-segment", "support/privacy", "support"),
                ("ordinary-segment", "inbox/privacy", "inbox"),
            ),
        )
        message_rows = (
            (
                "support-message",
                "support-record",
                "/support/0",
                "support-segment",
            ),
            (
                "ordinary-message",
                "ordinary-record",
                "/ordinary/0",
                "ordinary-segment",
            ),
        )
        connection.executemany(
            """
            INSERT INTO messages(
                id, source_record_id, source_file_id, json_pointer, chunk_number,
                source_array_index, conversation_id, segment_id, timestamp_ms,
                sender_raw, sender_display, sender_search, sender_is_owner,
                message_kind, has_payment, has_sensitive_ip, has_transcription,
                is_unsent, is_unavailable_body, logical_fingerprint, extra_json,
                sensitive_json
            ) VALUES (
                ?, ?, 1, ?, 1, 0, 'privacy-conversation', ?, 1,
                'Fixture', 'Fixture', 'fixture', 0, 'text', 0, 0, 0, 0, 0,
                ?, ?, ?
            )
            """,
            tuple(
                (
                    message_id,
                    source_record_id,
                    pointer,
                    segment_id,
                    ("c" if message_id == "support-message" else "d") * 64,
                    json.dumps(legacy_extra, sort_keys=True),
                    json.dumps(existing_sensitive, sort_keys=True),
                )
                for message_id, source_record_id, pointer, segment_id in message_rows
            ),
        )

    command.upgrade(config, "head")
    with sqlite3.connect(database) as connection:
        support_row = connection.execute(
            """
            SELECT extra_json, sensitive_json, has_sensitive_ip, has_payment
            FROM messages WHERE id = 'support-message'
            """
        ).fetchone()
        ordinary_row = connection.execute(
            """
            SELECT extra_json, sensitive_json, has_sensitive_ip, has_payment
            FROM messages WHERE id = 'ordinary-message'
            """
        ).fetchone()

    assert support_row is not None
    assert json.loads(support_row[0]) == {
        "to": "Archive Owner",
        "subject": "Support Case",
        "category": "account support",
        "status": "open",
        "metadata": {
            "shipping": "preserved",
            "topic": "preserved",
            "participant": "Support Agent",
            "network": {},
            "history": [
                {"status": "received"},
                {"status": "reviewed"},
            ],
        },
    }
    assert json.loads(support_row[1]) == {
        **existing_sensitive,
        "metadata": {
            "network": {"source_ip": ip_marker},
            "history": [
                None,
                {"billingInfo": {"account_credentials": payment_marker}},
            ],
        },
    }
    assert support_row[2:] == (1, 1)

    assert ordinary_row is not None
    assert json.loads(ordinary_row[0]) == legacy_extra
    assert json.loads(ordinary_row[1]) == existing_sensitive
    assert ordinary_row[2:] == (0, 0)

    command.downgrade(config, "9f2a4c7d8b31")
    with sqlite3.connect(database) as connection:
        downgraded = connection.execute(
            "SELECT extra_json, sensitive_json FROM messages WHERE id = 'support-message'"
        ).fetchone()
    assert downgraded == support_row[:2]


def test_nlp_status_covering_index_migration_preserves_exact_counts_and_plan(
    tmp_path: Path,
) -> None:
    database = tmp_path / "pre-nlp-status-index.sqlite3"
    project_root = Path(__file__).resolve().parents[1]
    config = Config(str(project_root / "alembic.ini"))
    config.set_main_option("script_location", str(project_root / "alembic"))
    config.set_main_option("sqlalchemy.url", f"sqlite+pysqlite:///{database}")
    command.upgrade(config, "3b77f95c2a10")

    query = """
        SELECT task, job_id, count(*)
        FROM nlp_annotations
        WHERE active IS 1 AND status = 'completed'
        GROUP BY task, job_id
        ORDER BY task, job_id
    """
    with sqlite3.connect(database) as connection:
        connection.executemany(
            """
            INSERT INTO jobs(
                id, kind, status, progress_current, checkpoint_json, counters_json
            ) VALUES (?, 'nlp_annotations', 'completed', 0, '{}', '{}')
            """,
            (("job-a",), ("job-b",)),
        )
        rows = [
            ("annotation-a", "sentiment", "job-a", "completed", 1),
            ("annotation-b", "sentiment", "job-a", "completed", 1),
            ("annotation-c", "toxicity", "job-b", "completed", 1),
            ("annotation-d", "emotion", None, "completed", 1),
            ("annotation-inactive", "sentiment", "job-a", "completed", 0),
            ("annotation-failed", "emotion", None, "failed", 1),
        ]
        connection.executemany(
            """
            INSERT INTO nlp_annotations(
                id, target_type, target_id, task, pipeline_version,
                model_name, model_revision, input_sha256, status, active,
                probabilities_json, attribution_json, warnings_json, provenance_json,
                job_id
            ) VALUES (?, 'message', ?, ?, 'fixture-v1', 'fixture/model',
                      'fixture-revision', ?, ?, ?, '{}', '{}', '{}', '{}', ?)
            """,
            (
                (identifier, identifier, task, f"{position:064x}", status, active, job_id)
                for position, (identifier, task, job_id, status, active) in enumerate(rows, 1)
            ),
        )
        expected = connection.execute(query).fetchall()
        before_plan = [row[-1] for row in connection.execute(f"EXPLAIN QUERY PLAN {query}")]

    assert expected == [
        ("emotion", None, 1),
        ("sentiment", "job-a", 2),
        ("toxicity", "job-b", 1),
    ]
    assert not any("ix_nlp_annotations_active_completed_task_job" in step for step in before_plan)

    command.upgrade(config, "head")
    with sqlite3.connect(database) as connection:
        migrated = connection.execute(query).fetchall()
        indexes = {row[1] for row in connection.execute("PRAGMA index_list('nlp_annotations')")}
        index_sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?",
            ("ix_nlp_annotations_active_completed_task_job",),
        ).fetchone()
        index_columns = [
            row[2]
            for row in connection.execute(
                "PRAGMA index_info('ix_nlp_annotations_active_completed_task_job')"
            )
        ]
        after_plan = [row[-1] for row in connection.execute(f"EXPLAIN QUERY PLAN {query}")]

    assert migrated == expected
    assert "ix_nlp_annotations_active_completed_task_job" in indexes
    assert index_sql is not None
    assert index_columns == ["active", "status", "task", "job_id"]
    assert any(
        "USING COVERING INDEX ix_nlp_annotations_active_completed_task_job" in step
        for step in after_plan
    )
    assert not any("USE TEMP B-TREE FOR GROUP BY" in step for step in after_plan)

    command.downgrade(config, "3b77f95c2a10")
    with sqlite3.connect(database) as connection:
        downgraded = connection.execute(query).fetchall()
        indexes = {row[1] for row in connection.execute("PRAGMA index_list('nlp_annotations')")}
    assert downgraded == expected
    assert "ix_nlp_annotations_active_completed_task_job" not in indexes

    command.upgrade(config, "head")


def test_feedback_scope_migration_round_trips_a_pre_scope_catalog(tmp_path: Path) -> None:
    database = tmp_path / "pre-scope.sqlite3"
    project_root = Path(__file__).resolve().parents[1]
    config = Config(str(project_root / "alembic.ini"))
    config.set_main_option("script_location", str(project_root / "alembic"))
    config.set_main_option("sqlalchemy.url", f"sqlite+pysqlite:///{database}")
    command.upgrade(config, "6f3c9e82a1b4")

    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            INSERT INTO relevance_feedback(
                id, query_sha256, query_text, search_mode, result_id, label
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("legacy-feedback", "a" * 64, "fixture", "semantic", "window:one", "relevant"),
        )

    command.upgrade(config, "head")
    with sqlite3.connect(database) as connection:
        migrated = connection.execute(
            "SELECT filter_scope_sha256, filter_scope_json FROM relevance_feedback"
        ).fetchone()
        indexes = {row[1] for row in connection.execute("PRAGMA index_list('relevance_feedback')")}
        assert migrated == (
            "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
            "{}",
        )
        assert "uq_relevance_feedback_query_mode_scope_result" in indexes
        connection.execute(
            """
            INSERT INTO relevance_feedback(
                id, query_sha256, query_text, search_mode,
                filter_scope_sha256, filter_scope_json, result_id, label
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "scoped-feedback",
                "a" * 64,
                "fixture",
                "semantic",
                "b" * 64,
                '{"entity_type":"message"}',
                "window:one",
                "not_relevant",
            ),
        )

    command.downgrade(config, "6f3c9e82a1b4")
    with sqlite3.connect(database) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info('relevance_feedback')")}
        indexes = {row[1] for row in connection.execute("PRAGMA index_list('relevance_feedback')")}
        row_count = connection.execute("SELECT count(*) FROM relevance_feedback").fetchone()[0]
    assert "filter_scope_json" not in columns
    assert "filter_scope_sha256" not in columns
    assert "uq_relevance_feedback_query_mode_result" in indexes
    assert row_count == 1

    command.upgrade(config, "head")


def test_argument_evaluation_intent_migration_backfills_only_the_public_flagship_digest(
    tmp_path: Path,
) -> None:
    database = tmp_path / "pre-argument-intent.sqlite3"
    project_root = Path(__file__).resolve().parents[1]
    config = Config(str(project_root / "alembic.ini"))
    config.set_main_option("script_location", str(project_root / "alembic"))
    config.set_main_option("sqlalchemy.url", f"sqlite+pysqlite:///{database}")
    command.upgrade(config, "7c2d4e8f1a90")

    flagship_digest = hashlib.sha256(b"argument").hexdigest()
    rows = [
        ("tagged", flagship_digest, "semantic", "window:tagged", "test"),
        ("train", flagship_digest, "semantic", "window:train", "train"),
        ("keyword", flagship_digest, "keyword", "window:keyword", "test"),
        ("other", "a" * 64, "hybrid", "window:other", "test"),
    ]
    with sqlite3.connect(database) as connection:
        connection.executemany(
            """
            INSERT INTO relevance_feedback(
                id, query_sha256, query_text, search_mode, result_id, label,
                evaluation_split
            ) VALUES (?, ?, 'private fixture query', ?, ?, 'relevant', ?)
            """,
            rows,
        )

    command.upgrade(config, "head")
    with sqlite3.connect(database) as connection:
        intents = dict(
            connection.execute("SELECT id, evaluation_intent FROM relevance_feedback ORDER BY id")
        )
    assert intents == {
        "keyword": None,
        "other": None,
        "tagged": "argument_direct_conflict",
        "train": None,
    }

    command.downgrade(config, "7c2d4e8f1a90")
    with sqlite3.connect(database) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info('relevance_feedback')")}
    assert "evaluation_intent" not in columns
    command.upgrade(config, "head")


def test_development_calibration_intent_migration_backfills_only_audited_validation_cohorts(
    tmp_path: Path,
) -> None:
    database = tmp_path / "pre-development-intent.sqlite3"
    project_root = Path(__file__).resolve().parents[1]
    config = Config(str(project_root / "alembic.ini"))
    config.set_main_option("script_location", str(project_root / "alembic"))
    config.set_main_option("sqlalchemy.url", f"sqlite+pysqlite:///{database}")
    command.upgrade(config, "2d6f8a19c4e7")

    flagship_digest = hashlib.sha256(b"argument").hexdigest()
    default_scope_digest = hashlib.sha256(b"{}").hexdigest()
    rows = [
        (
            "literal-validation",
            flagship_digest,
            "semantic",
            default_scope_digest,
            "{}",
            "window:literal-validation",
            "validation",
            None,
            None,
        ),
        (
            "review-marker-validation",
            "b" * 64,
            "semantic",
            default_scope_digest,
            "{}",
            "window:review-marker-validation",
            "validation",
            "blind-high-intensity-direct-conflict-v1: sealed development cohort",
            None,
        ),
        (
            "literal-hybrid",
            flagship_digest,
            "hybrid",
            default_scope_digest,
            "{}",
            "window:literal-hybrid",
            "validation",
            None,
            None,
        ),
        (
            "literal-test",
            flagship_digest,
            "semantic",
            default_scope_digest,
            "{}",
            "window:literal-test",
            "test",
            None,
            None,
        ),
        (
            "literal-scoped",
            flagship_digest,
            "semantic",
            "c" * 64,
            '{"entity_type":"message"}',
            "window:literal-scoped",
            "validation",
            None,
            None,
        ),
        (
            "literal-forged-scope",
            flagship_digest,
            "semantic",
            default_scope_digest,
            '{"entity_type":"message"}',
            "window:literal-forged-scope",
            "validation",
            None,
            None,
        ),
        (
            "marker-train",
            "d" * 64,
            "semantic",
            default_scope_digest,
            "{}",
            "window:marker-train",
            "train",
            "blind-high-intensity-direct-conflict-v1: train fixture",
            None,
        ),
        (
            "unrelated-validation",
            "e" * 64,
            "semantic",
            default_scope_digest,
            "{}",
            "window:unrelated-validation",
            "validation",
            None,
            None,
        ),
        (
            "existing-intent",
            flagship_digest,
            "semantic",
            default_scope_digest,
            "{}",
            "window:existing-intent",
            "validation",
            None,
            "argument_direct_conflict",
        ),
    ]
    with sqlite3.connect(database) as connection:
        connection.executemany(
            """
            INSERT INTO relevance_feedback(
                id, query_sha256, query_text, search_mode,
                filter_scope_sha256, filter_scope_json, result_id, label,
                evaluation_split, notes, evaluation_intent
            ) VALUES (?, ?, 'private fixture query', ?, ?, ?, ?, 'relevant', ?, ?, ?)
            """,
            rows,
        )

    command.upgrade(config, "head")
    with sqlite3.connect(database) as connection:
        intents = dict(
            connection.execute("SELECT id, evaluation_intent FROM relevance_feedback ORDER BY id")
        )
    assert intents == {
        "existing-intent": "argument_direct_conflict",
        "literal-hybrid": None,
        "literal-forged-scope": None,
        "literal-scoped": None,
        "literal-test": None,
        "literal-validation": "argument_direct_conflict_development",
        "marker-train": None,
        "review-marker-validation": "argument_direct_conflict_development",
        "unrelated-validation": None,
    }

    command.downgrade(config, "2d6f8a19c4e7")
    with sqlite3.connect(database) as connection:
        downgraded_intents = dict(
            connection.execute("SELECT id, evaluation_intent FROM relevance_feedback ORDER BY id")
        )
    assert downgraded_intents == {
        **intents,
        "literal-validation": None,
        "review-marker-validation": None,
    }
    command.upgrade(config, "head")


def test_rag_audit_privacy_migration_scrubs_legacy_private_text(tmp_path: Path) -> None:
    database = tmp_path / "pre-rag-privacy.sqlite3"
    project_root = Path(__file__).resolve().parents[1]
    config = Config(str(project_root / "alembic.ini"))
    config.set_main_option("script_location", str(project_root / "alembic"))
    config.set_main_option("sqlalchemy.url", f"sqlite+pysqlite:///{database}")
    command.upgrade(config, "9a71c5e2d4f8")

    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            INSERT INTO rag_audits(
                id, status, model, query_sha256, query_text, search_mode,
                retrieved_ids_json, citation_ids_json, excerpt_hashes_json,
                usage_json, error_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-rag-audit",
                "failed",
                "fixture-model",
                "a" * 64,
                "private fixture question that must be erased",
                "hybrid",
                '{"ids":[]}',
                '{"ids":[]}',
                "{}",
                "{}",
                "provider echoed a private fixture excerpt",
            ),
        )

    command.upgrade(config, "head")
    with sqlite3.connect(database) as connection:
        scrubbed = connection.execute(
            "SELECT query_text, error_text FROM rag_audits WHERE id = ?",
            ("legacy-rag-audit",),
        ).fetchone()

    assert scrubbed == ("", "LegacyRagFailure")


def test_semantic_membership_migration_round_trips_legacy_assignments(
    tmp_path: Path,
) -> None:
    database = tmp_path / "pre-semantic-membership.sqlite3"
    project_root = Path(__file__).resolve().parents[1]
    config = Config(str(project_root / "alembic.ini"))
    config.set_main_option("script_location", str(project_root / "alembic"))
    config.set_main_option("sqlalchemy.url", f"sqlite+pysqlite:///{database}")
    command.upgrade(config, "9a71c5e2d4f8")

    first_config = "a" * 64
    second_config = "b" * 64
    orphaned_config = "f" * 64
    with sqlite3.connect(database) as connection:
        connection.executemany(
            """
            INSERT INTO semantic_index_configs(
                id, active, status, backend, backend_version, model_name,
                model_revision, dimensions, distance_metric, document_prefix,
                query_prefix, vector_table, window_version, window_count, audit_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    first_config,
                    1,
                    "ready",
                    "fixture",
                    "1",
                    "fixture-model-a",
                    "fixture-revision",
                    3,
                    "cosine",
                    "passage: ",
                    "query: ",
                    "semantic_a",
                    "fixture-v1",
                    2,
                    "{}",
                ),
                (
                    second_config,
                    0,
                    "ready",
                    "fixture",
                    "1",
                    "fixture-model-b",
                    "fixture-revision",
                    3,
                    "cosine",
                    "passage: ",
                    "query: ",
                    "semantic_b",
                    "fixture-v1",
                    1,
                    "{}",
                ),
            ),
        )
        connection.executemany(
            """
            INSERT INTO search_windows(
                id, window_version, entity_type, turn_count, approx_token_count,
                text_display, text_search, participants_search, source_ids_json,
                build_generation, embedding_config_hash, embedding_updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    "window-a-newer",
                    "fixture-v1",
                    "message",
                    1,
                    3,
                    "first fixture",
                    "first fixture",
                    "",
                    "[]",
                    "generation-1",
                    first_config,
                    "2025-01-02 03:04:05",
                ),
                (
                    "window-a-no-timestamp",
                    "fixture-v1",
                    "message",
                    1,
                    3,
                    "second fixture",
                    "second fixture",
                    "",
                    "[]",
                    "generation-1",
                    first_config,
                    None,
                ),
                (
                    "window-b",
                    "fixture-v1",
                    "activity",
                    1,
                    3,
                    "third fixture",
                    "third fixture",
                    "",
                    "[]",
                    "generation-1",
                    second_config,
                    "2025-02-03 04:05:06",
                ),
                (
                    "window-orphaned-config",
                    "fixture-v1",
                    "message",
                    1,
                    3,
                    "orphan fixture",
                    "orphan fixture",
                    "",
                    "[]",
                    "generation-1",
                    orphaned_config,
                    "2025-03-04 05:06:07",
                ),
                (
                    "window-never-embedded",
                    "fixture-v1",
                    "message",
                    1,
                    3,
                    "plain fixture",
                    "plain fixture",
                    "",
                    "[]",
                    "generation-1",
                    None,
                    None,
                ),
            ),
        )

    expected_memberships = [
        (first_config, "window-a-newer"),
        (first_config, "window-a-no-timestamp"),
        (second_config, "window-b"),
    ]

    def assert_exact_memberships() -> None:
        with sqlite3.connect(database) as connection:
            memberships = connection.execute(
                """
                SELECT config_id, window_id
                FROM semantic_window_embeddings
                ORDER BY config_id, window_id
                """
            ).fetchall()
            duplicate_count = connection.execute(
                """
                SELECT count(*)
                FROM (
                    SELECT config_id, window_id
                    FROM semantic_window_embeddings
                    GROUP BY config_id, window_id
                    HAVING count(*) > 1
                )
                """
            ).fetchone()[0]
            preserved_timestamp = connection.execute(
                """
                SELECT embedded_at
                FROM semantic_window_embeddings
                WHERE config_id = ? AND window_id = ?
                """,
                (first_config, "window-a-newer"),
            ).fetchone()[0]
            synthesized_timestamp = connection.execute(
                """
                SELECT embedded_at
                FROM semantic_window_embeddings
                WHERE config_id = ? AND window_id = ?
                """,
                (first_config, "window-a-no-timestamp"),
            ).fetchone()[0]

        assert memberships == expected_memberships
        assert duplicate_count == 0
        assert preserved_timestamp == "2025-01-02 03:04:05"
        assert synthesized_timestamp is not None

    command.upgrade(config, "c4f78a1d2e60")
    assert_exact_memberships()

    command.upgrade(config, "head")
    assert_exact_memberships()

    command.downgrade(config, "9a71c5e2d4f8")
    with sqlite3.connect(database) as connection:
        membership_table = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name = 'semantic_window_embeddings'
            """
        ).fetchone()
        legacy_assignments = connection.execute(
            """
            SELECT id, embedding_config_hash
            FROM search_windows
            ORDER BY id
            """
        ).fetchall()

    assert membership_table is None
    assert legacy_assignments == [
        ("window-a-newer", first_config),
        ("window-a-no-timestamp", first_config),
        ("window-b", second_config),
        ("window-never-embedded", None),
        ("window-orphaned-config", orphaned_config),
    ]

    command.upgrade(config, "head")
    assert_exact_memberships()
