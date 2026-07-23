from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import exists, func, or_, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from meta_archive.models import Job, SearchWindow, SemanticIndexConfig, SemanticWindowEmbedding

LANCE_TABLE_PAGE_SIZE = 1_000
SEMANTIC_ROW_PAGE_SIZE = 256
SEMANTIC_VECTOR_NORM_TOLERANCE = 1e-3


class LanceTableInventoryError(RuntimeError):
    """LanceDB did not provide a complete, well-formed table inventory."""


class SemanticTableIntegrityError(RuntimeError):
    """A Lance table diverged from its authoritative SQLite window product."""


@dataclass(frozen=True, slots=True)
class SemanticTableValidation:
    row_count: int
    unique_count: int
    expected_window_count: int
    membership_count: int
    duplicate_rows: int
    missing_catalog_rows: int
    missing_lance_rows: int
    extra_lance_rows: int
    missing_membership_rows: int
    membership_without_lance_rows: int
    invalid_vector_rows: int
    metadata_mismatch_rows: int
    repaired_membership_rows: int
    identity_sha256: str
    expected_identity_sha256: str
    membership_identity_sha256: str
    metadata_sha256: str
    expected_metadata_sha256: str
    last_window_id: str | None
    valid: bool


def lance_table_names(database: Any, *, page_size: int = LANCE_TABLE_PAGE_SIZE) -> set[str]:
    """Enumerate every root-namespace LanceDB table and reject broken pagination.

    Older call sites supplied a fixed ``limit`` and silently ignored ``page_token``. That can omit
    a registered or orphan table and is especially unsafe during physical deletion. Keep the
    pagination contract in one place so search, indexing, analytics, and purge all see the same
    complete inventory.

    The installed local LanceDB backend returns the *next* table name as ``response.page_token``,
    but interprets a supplied token as an exclusive lower bound. Feeding the response token back
    therefore skips that boundary row. Resume after the last table actually observed instead; the
    response token remains the authoritative terminal/progress signal. These archive call sites
    intentionally use the local root namespace. Supplying ``namespace_path`` would route through
    LanceDB's optional namespace client, which is not available in the pinned local backend.
    """

    if page_size < 1:
        raise ValueError("Lance table inventory page size must be positive")

    names: set[str] = set()
    request_token: str | None = None
    consumed_request_tokens: set[str] = set()
    observed_response_tokens: set[str] = set()
    while True:
        if request_token is None:
            response = database.list_tables(limit=page_size)
        else:
            if request_token in consumed_request_tokens:
                raise LanceTableInventoryError("Lance table pagination repeated a page token")
            consumed_request_tokens.add(request_token)
            response = database.list_tables(page_token=request_token, limit=page_size)

        raw_tables = getattr(response, "tables", None)
        if not isinstance(raw_tables, (list, tuple)) or any(
            not isinstance(table_name, str) or not table_name for table_name in raw_tables
        ):
            raise LanceTableInventoryError("Lance table inventory returned invalid table names")
        previous_count = len(names)
        names.update(raw_tables)

        response_token = getattr(response, "page_token", None)
        if response_token is None:
            return names
        if not isinstance(response_token, str) or not response_token:
            raise LanceTableInventoryError("Lance table inventory returned an invalid page token")
        if response_token in observed_response_tokens:
            raise LanceTableInventoryError("Lance table pagination repeated a page token")
        observed_response_tokens.add(response_token)
        if len(names) == previous_count:
            raise LanceTableInventoryError("Lance table pagination made no progress")
        # A non-terminal page that added names must have a safe exclusive lower bound for the next
        # request. Using the final observed name avoids the installed backend's boundary skip.
        next_request_token = raw_tables[-1]
        if next_request_token == request_token or next_request_token in consumed_request_tokens:
            raise LanceTableInventoryError("Lance table pagination repeated a page token")
        request_token = next_request_token


def _window_metadata_statement() -> Any:
    return select(
        SearchWindow.id,
        SearchWindow.entity_type,
        SearchWindow.conversation_id,
        SearchWindow.segment_id,
        SearchWindow.activity_id,
        SearchWindow.attachment_occurrence_id,
        SearchWindow.first_message_id,
        SearchWindow.first_timestamp_ms,
        SearchWindow.last_timestamp_ms,
        SearchWindow.source_category,
        SearchWindow.participants_search,
        SearchWindow.text_display,
        SearchWindow.build_generation,
    )


def _sha256_text(value: object) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8", errors="surrogatepass")).hexdigest()


def _catalog_metadata_payload(row: Any) -> dict[str, object]:
    return {
        "entity_type": str(row.entity_type),
        "conversation_id": str(row.conversation_id or ""),
        "segment_id": str(row.segment_id or ""),
        "activity_id": str(row.activity_id or ""),
        "source_category": str(row.source_category or ""),
        "participants_search_sha256": _sha256_text(row.participants_search),
        "first_timestamp_ms": (
            None if row.first_timestamp_ms is None else int(row.first_timestamp_ms)
        ),
        "last_timestamp_ms": (
            None if row.last_timestamp_ms is None else int(row.last_timestamp_ms)
        ),
        "jump_id": str(
            row.first_message_id or row.activity_id or row.attachment_occurrence_id or ""
        ),
        "text_sha256": _sha256_text(row.text_display),
        "build_generation": str(row.build_generation),
    }


def _lance_metadata_payload(row: dict[str, Any]) -> dict[str, object]:
    return {
        "entity_type": str(row.get("entity_type") or ""),
        "conversation_id": str(row.get("conversation_id") or ""),
        "segment_id": str(row.get("segment_id") or ""),
        "activity_id": str(row.get("activity_id") or ""),
        "source_category": str(row.get("source_category") or ""),
        "participants_search_sha256": _sha256_text(row.get("participants_search")),
        "first_timestamp_ms": (
            None if row.get("first_timestamp_ms") is None else int(row["first_timestamp_ms"])
        ),
        "last_timestamp_ms": (
            None if row.get("last_timestamp_ms") is None else int(row["last_timestamp_ms"])
        ),
        "jump_id": str(row.get("jump_id") or ""),
        "text_sha256": _sha256_text(row.get("text")),
        "build_generation": str(row.get("build_generation") or ""),
    }


def _metadata_sha256(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _update_identity_digest(digest: Any, value: str) -> None:
    encoded = value.encode("utf-8", errors="surrogatepass")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def _update_metadata_digest(digest: Any, identifier: str, metadata_sha256: str) -> None:
    _update_identity_digest(digest, identifier)
    digest.update(bytes.fromhex(metadata_sha256))


def _vector_is_valid(vector: object, *, dimensions: int) -> bool:
    if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes, bytearray)):
        return False
    try:
        values = [float(value) for value in vector]
    except (TypeError, ValueError, OverflowError):
        return False
    if len(values) != dimensions or not all(math.isfinite(value) for value in values):
        return False
    norm = math.sqrt(math.fsum(value * value for value in values))
    return math.isclose(
        norm,
        1.0,
        rel_tol=SEMANTIC_VECTOR_NORM_TOLERANCE,
        abs_tol=SEMANTIC_VECTOR_NORM_TOLERANCE,
    )


def validate_semantic_table(
    table: Any,
    engine: Engine,
    *,
    config_id: str,
    dimensions: int,
    require_complete: bool,
    repair_missing_memberships: bool = False,
    page_size: int = SEMANTIC_ROW_PAGE_SIZE,
    raise_on_error: bool = True,
) -> SemanticTableValidation:
    """Boundedly reconcile every Lance row against SQLite metadata and membership.

    Lance does not expose a stable ordered scan in the installed embedded backend. Physical
    offsets are therefore consumed exactly once in bounded pages, while a private temporary
    SQLite database provides exact duplicate/set reconciliation without retaining every ID or
    text-bearing row in Python memory.
    """

    if dimensions < 1:
        raise ValueError("Semantic vector dimensions must be positive")
    if not 1 <= page_size <= 300:
        raise ValueError("Semantic validation page size must be between 1 and 300")

    selected_columns = [
        "id",
        "vector",
        "entity_type",
        "conversation_id",
        "segment_id",
        "activity_id",
        "source_category",
        "participants_search",
        "first_timestamp_ms",
        "last_timestamp_ms",
        "jump_id",
        "text",
        "build_generation",
    ]
    row_count = int(table.count_rows())
    duplicate_rows = 0
    invalid_vector_rows = 0
    metadata_mismatch_rows = 0
    last_window_id: str | None = None

    with tempfile.TemporaryDirectory(prefix="meta-archive-semantic-audit-") as temporary:
        inventory_path = Path(temporary) / "inventory.sqlite"
        inventory = sqlite3.connect(inventory_path)
        try:
            inventory.execute("PRAGMA journal_mode=OFF")
            inventory.execute("PRAGMA synchronous=OFF")
            inventory.execute(
                """
                CREATE TABLE seen (
                    id TEXT PRIMARY KEY,
                    metadata_sha256 TEXT NOT NULL,
                    catalog_present INTEGER NOT NULL,
                    membership_present INTEGER NOT NULL
                ) WITHOUT ROWID
                """
            )
            inventory.execute(
                """
                CREATE TABLE expected (
                    id TEXT PRIMARY KEY,
                    metadata_sha256 TEXT NOT NULL
                ) WITHOUT ROWID
                """
            )

            scanned_rows = 0
            for offset in range(0, row_count, page_size):
                stop = min(offset + page_size, row_count)
                lance_rows = (
                    table.take_offsets(list(range(offset, stop))).select(selected_columns).to_list()
                )
                if len(lance_rows) != stop - offset:
                    raise SemanticTableIntegrityError(
                        "Semantic table scan returned an incomplete physical page"
                    )
                scanned_rows += len(lance_rows)
                page_ids = [str(row.get("id") or "") for row in lance_rows]
                unique_page_ids = list(dict.fromkeys(page_ids))
                with Session(engine) as session:
                    catalog_rows = {
                        str(row.id): row
                        for row in session.execute(
                            _window_metadata_statement().where(SearchWindow.id.in_(unique_page_ids))
                        )
                    }
                    member_ids = set(
                        session.scalars(
                            select(SemanticWindowEmbedding.window_id).where(
                                SemanticWindowEmbedding.config_id == config_id,
                                SemanticWindowEmbedding.window_id.in_(unique_page_ids),
                            )
                        )
                    )

                page_inventory: list[tuple[str, str, int, int]] = []
                for identifier, lance_row in zip(page_ids, lance_rows, strict=True):
                    if last_window_id is None or identifier > last_window_id:
                        last_window_id = identifier
                    catalog_row = catalog_rows.get(identifier)
                    actual_metadata = _lance_metadata_payload(dict(lance_row))
                    actual_metadata_sha256 = _metadata_sha256(actual_metadata)
                    if not _vector_is_valid(lance_row.get("vector"), dimensions=dimensions):
                        invalid_vector_rows += 1
                    if catalog_row is not None:
                        expected_metadata = _catalog_metadata_payload(catalog_row)
                        if actual_metadata != expected_metadata:
                            metadata_mismatch_rows += 1
                    page_inventory.append(
                        (
                            identifier,
                            actual_metadata_sha256,
                            int(catalog_row is not None),
                            int(identifier in member_ids),
                        )
                    )
                changes_before = inventory.total_changes
                inventory.executemany(
                    """
                    INSERT OR IGNORE INTO seen (
                        id, metadata_sha256, catalog_present, membership_present
                    ) VALUES (?, ?, ?, ?)
                    """,
                    page_inventory,
                )
                duplicate_rows += len(page_inventory) - (inventory.total_changes - changes_before)
            if scanned_rows != row_count:
                raise SemanticTableIntegrityError(
                    "Semantic table scan did not consume the declared row count"
                )

            expected_statement = _window_metadata_statement().order_by(SearchWindow.id)
            with Session(engine) as expected_session:
                if not require_complete and last_window_id is None:
                    expected_rows: Iterable[Any] = ()
                elif not require_complete:
                    expected_rows = expected_session.execute(
                        expected_statement.where(SearchWindow.id <= last_window_id)
                    ).yield_per(page_size)
                else:
                    expected_rows = expected_session.execute(expected_statement).yield_per(
                        page_size
                    )
                expected_buffer: list[tuple[str, str]] = []
                for row in expected_rows:
                    expected_buffer.append(
                        (str(row.id), _metadata_sha256(_catalog_metadata_payload(row)))
                    )
                    if len(expected_buffer) >= page_size:
                        inventory.executemany(
                            "INSERT OR IGNORE INTO expected (id, metadata_sha256) VALUES (?, ?)",
                            expected_buffer,
                        )
                        expected_buffer.clear()
                if expected_buffer:
                    inventory.executemany(
                        "INSERT OR IGNORE INTO expected (id, metadata_sha256) VALUES (?, ?)",
                        expected_buffer,
                    )

            unique_count = int(inventory.execute("SELECT count(*) FROM seen").fetchone()[0])
            expected_window_count = int(
                inventory.execute("SELECT count(*) FROM expected").fetchone()[0]
            )
            missing_lance_rows = int(
                inventory.execute(
                    """
                    SELECT count(*) FROM expected
                    WHERE NOT EXISTS (SELECT 1 FROM seen WHERE seen.id = expected.id)
                    """
                ).fetchone()[0]
            )
            extra_lance_rows = int(
                inventory.execute(
                    """
                    SELECT count(*) FROM seen
                    WHERE NOT EXISTS (SELECT 1 FROM expected WHERE expected.id = seen.id)
                    """
                ).fetchone()[0]
            )
            missing_catalog_rows = int(
                inventory.execute("SELECT count(*) FROM seen WHERE catalog_present = 0").fetchone()[
                    0
                ]
            )
            missing_membership_rows = int(
                inventory.execute(
                    "SELECT count(*) FROM seen WHERE membership_present = 0"
                ).fetchone()[0]
            )
            with Session(engine) as session:
                membership_count = int(
                    session.scalar(
                        select(func.count(SemanticWindowEmbedding.window_id)).where(
                            SemanticWindowEmbedding.config_id == config_id
                        )
                    )
                    or 0
                )
            membership_seen = unique_count - missing_membership_rows
            membership_without_lance_rows = max(0, membership_count - membership_seen)

            structural_valid = (
                duplicate_rows == 0
                and missing_catalog_rows == 0
                and missing_lance_rows == 0
                and extra_lance_rows == 0
                and membership_without_lance_rows == 0
                and invalid_vector_rows == 0
                and metadata_mismatch_rows == 0
            )
            repaired_membership_rows = 0
            if repair_missing_memberships and structural_valid and missing_membership_rows:
                last_repair_id: str | None = None
                while True:
                    if last_repair_id is None:
                        repair_rows = inventory.execute(
                            """
                            SELECT id FROM seen
                            WHERE membership_present = 0
                            ORDER BY id LIMIT ?
                            """,
                            (page_size,),
                        ).fetchall()
                    else:
                        repair_rows = inventory.execute(
                            """
                            SELECT id FROM seen
                            WHERE membership_present = 0 AND id > ?
                            ORDER BY id LIMIT ?
                            """,
                            (last_repair_id, page_size),
                        ).fetchall()
                    if not repair_rows:
                        break
                    repair_ids = [str(row[0]) for row in repair_rows]
                    embedded_at = datetime.now(UTC)
                    with Session(engine) as session, session.begin():
                        statement = sqlite_insert(SemanticWindowEmbedding).values(
                            [
                                {
                                    "config_id": config_id,
                                    "window_id": identifier,
                                    "embedded_at": embedded_at,
                                }
                                for identifier in repair_ids
                            ]
                        )
                        session.execute(
                            statement.on_conflict_do_nothing(
                                index_elements=[
                                    SemanticWindowEmbedding.config_id,
                                    SemanticWindowEmbedding.window_id,
                                ]
                            )
                        )
                    inventory.executemany(
                        "UPDATE seen SET membership_present = 1 WHERE id = ?",
                        [(identifier,) for identifier in repair_ids],
                    )
                    repaired_membership_rows += len(repair_ids)
                    last_repair_id = repair_ids[-1]
                missing_membership_rows = 0
                membership_count += repaired_membership_rows
                membership_without_lance_rows = 0

            identity_digest = hashlib.sha256()
            metadata_digest = hashlib.sha256()
            for identifier, metadata_sha256 in inventory.execute(
                "SELECT id, metadata_sha256 FROM seen ORDER BY id"
            ):
                _update_identity_digest(identity_digest, str(identifier))
                _update_metadata_digest(metadata_digest, str(identifier), str(metadata_sha256))
            expected_identity_digest = hashlib.sha256()
            expected_metadata_digest = hashlib.sha256()
            for identifier, metadata_sha256 in inventory.execute(
                "SELECT id, metadata_sha256 FROM expected ORDER BY id"
            ):
                _update_identity_digest(expected_identity_digest, str(identifier))
                _update_metadata_digest(
                    expected_metadata_digest, str(identifier), str(metadata_sha256)
                )

            membership_identity_digest = hashlib.sha256()
            with Session(engine) as session:
                membership_rows = session.scalars(
                    select(SemanticWindowEmbedding.window_id)
                    .where(SemanticWindowEmbedding.config_id == config_id)
                    .order_by(SemanticWindowEmbedding.window_id)
                ).yield_per(page_size)
                for identifier in membership_rows:
                    _update_identity_digest(membership_identity_digest, str(identifier))

            identity_sha256 = identity_digest.hexdigest()
            expected_identity_sha256 = expected_identity_digest.hexdigest()
            membership_identity_sha256 = membership_identity_digest.hexdigest()
            metadata_sha256 = metadata_digest.hexdigest()
            expected_metadata_sha256 = expected_metadata_digest.hexdigest()
            valid = bool(
                structural_valid
                and missing_membership_rows == 0
                and membership_count == unique_count == expected_window_count
                and identity_sha256 == expected_identity_sha256 == membership_identity_sha256
                and metadata_sha256 == expected_metadata_sha256
            )
            validation = SemanticTableValidation(
                row_count=row_count,
                unique_count=unique_count,
                expected_window_count=expected_window_count,
                membership_count=membership_count,
                duplicate_rows=duplicate_rows,
                missing_catalog_rows=missing_catalog_rows,
                missing_lance_rows=missing_lance_rows,
                extra_lance_rows=extra_lance_rows,
                missing_membership_rows=missing_membership_rows,
                membership_without_lance_rows=membership_without_lance_rows,
                invalid_vector_rows=invalid_vector_rows,
                metadata_mismatch_rows=metadata_mismatch_rows,
                repaired_membership_rows=repaired_membership_rows,
                identity_sha256=identity_sha256,
                expected_identity_sha256=expected_identity_sha256,
                membership_identity_sha256=membership_identity_sha256,
                metadata_sha256=metadata_sha256,
                expected_metadata_sha256=expected_metadata_sha256,
                last_window_id=last_window_id,
                valid=valid,
            )
        finally:
            inventory.close()

    if raise_on_error and not validation.valid:
        raise SemanticTableIntegrityError(
            "Semantic table integrity failed "
            f"(rows={validation.row_count}, unique={validation.unique_count}, "
            f"expected={validation.expected_window_count}, "
            f"duplicates={validation.duplicate_rows}, "
            f"catalog_missing={validation.missing_catalog_rows}, "
            f"missing={validation.missing_lance_rows}, extra={validation.extra_lance_rows}, "
            f"membership_missing={validation.missing_membership_rows}, "
            f"membership_extra={validation.membership_without_lance_rows}, "
            f"invalid_vectors={validation.invalid_vector_rows}, "
            f"metadata_mismatches={validation.metadata_mismatch_rows})"
        )
    return validation


@dataclass(frozen=True, slots=True)
class SemanticWindowIdentity:
    """Deterministic, duplicate-aware identity for one semantic vector table."""

    row_count: int
    unique_count: int
    identity_sha256: str


def semantic_window_identity(values: Iterable[object]) -> SemanticWindowIdentity:
    identifiers = sorted(str(value) for value in values)
    digest = hashlib.sha256()
    for identifier in identifiers:
        encoded = identifier.encode("utf-8", errors="surrogatepass")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return SemanticWindowIdentity(
        row_count=len(identifiers),
        unique_count=len(set(identifiers)),
        identity_sha256=digest.hexdigest(),
    )


def lance_window_identity(table: Any) -> SemanticWindowIdentity:
    identifiers = table.search().select(["id"]).to_arrow().column("id").to_pylist()
    return semantic_window_identity(identifiers)


def semantic_window_membership_clause(config_id: str) -> ColumnElement[bool]:
    """Return the authoritative per-config window predicate with legacy-fixture fallback."""

    config_has_memberships = exists().where(SemanticWindowEmbedding.config_id == config_id)
    window_is_member = (
        exists()
        .where(
            SemanticWindowEmbedding.config_id == config_id,
            SemanticWindowEmbedding.window_id == SearchWindow.id,
        )
        .correlate(SearchWindow)
    )
    return or_(
        window_is_member,
        (~config_has_memberships) & (SearchWindow.embedding_config_hash == config_id),
    )


def semantic_config_window_count(session: Session, config_id: str) -> int:
    """Count a config's durable memberships, falling back for pre-migration fixture rows."""

    membership_count = int(
        session.scalar(
            select(func.count(SemanticWindowEmbedding.window_id)).where(
                SemanticWindowEmbedding.config_id == config_id
            )
        )
        or 0
    )
    if membership_count:
        return membership_count
    return int(
        session.scalar(
            select(func.count(SearchWindow.id)).where(
                SearchWindow.embedding_config_hash == config_id
            )
        )
        or 0
    )


def semantic_config_window_ids(session: Session, config_id: str) -> list[str]:
    """Return exact authoritative memberships, with the legacy-fixture fallback."""

    membership_ids = list(
        session.scalars(
            select(SemanticWindowEmbedding.window_id).where(
                SemanticWindowEmbedding.config_id == config_id
            )
        )
    )
    if membership_ids:
        return membership_ids
    return list(
        session.scalars(
            select(SearchWindow.id).where(SearchWindow.embedding_config_hash == config_id)
        )
    )


def invalidate_ready_semantic_indexes(
    session: Session,
    *,
    reason: str,
    source_kind: str,
    source_id: str,
) -> int:
    """Mark ready/candidate indexes stale in the searchable-corpus change transaction."""

    configs = list(
        session.scalars(
            select(SemanticIndexConfig).where(
                SemanticIndexConfig.status.in_(("ready", "building")),
            )
        )
    )
    stale_at = datetime.now(UTC).isoformat()
    for config in configs:
        config.active = False
        config.status = "stale"
        config.audit_json = {
            **config.audit_json,
            "stale": {
                "reason": reason,
                "source_kind": source_kind,
                "source_id": source_id,
                "stale_at": stale_at,
            },
        }
    active_jobs = list(
        session.scalars(
            select(Job).where(
                Job.kind == "semantic_index",
                Job.status.in_(("queued", "running")),
            )
        )
    )
    for job in active_jobs:
        job.status = "failed"
        job.stage = "Semantic index invalidated by searchable corpus change"
        job.error_text = reason
        job.finished_at = datetime.now(UTC)
    return len(configs)
