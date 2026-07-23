from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar, cast

from fastapi import HTTPException, Request, status
from sqlalchemy import Connection, Engine, event, text
from sqlalchemy.orm import Session

AggregateT = TypeVar("AggregateT")
_MISSING = object()
_CACHE_INITIALIZATION_LOCK = threading.Lock()
_EPOCH_INITIALIZATION_LOCK = threading.Lock()
CATALOG_CHANGED_RETRY_DETAIL = "Catalog changed while preparing this response; retry the request"


class CatalogCommitEpoch:
    """Engine-scoped monotonic invalidation for commits made by this process."""

    def __init__(self) -> None:
        self._value = 0
        self._lock = threading.Lock()

    def snapshot(self) -> int:
        with self._lock:
            return self._value

    def on_commit(self, _connection: Connection) -> None:
        # SQLAlchemy invokes this listener as part of commit. Incrementing before the DBAPI call
        # can cause a harmless extra invalidation if that commit subsequently fails, but can never
        # preserve a stale aggregate after a successful commit.
        with self._lock:
            self._value += 1


def _engine_commit_epoch(engine: Engine) -> CatalogCommitEpoch:
    existing = getattr(engine, "_meta_archive_catalog_commit_epoch", None)
    if isinstance(existing, CatalogCommitEpoch):
        return existing
    with _EPOCH_INITIALIZATION_LOCK:
        existing = getattr(engine, "_meta_archive_catalog_commit_epoch", None)
        if isinstance(existing, CatalogCommitEpoch):
            return existing
        created = CatalogCommitEpoch()
        event.listen(engine, "commit", created.on_commit)
        # The listener and counter live exactly as long as their owning Engine. Keeping the state
        # on the Engine avoids a global strong-reference registry and guarantees one listener even
        # when concurrent first requests initialize the aggregate cache.
        engine._meta_archive_catalog_commit_epoch = created  # type: ignore[attr-defined]
        return created


def _catalog_data_version(session: Session) -> int:
    return int(session.scalar(text("PRAGMA data_version")) or 0)


def _catalog_file_state(database: Path) -> tuple[tuple[int, int, int, int, int], ...]:
    paths = (database, Path(f"{database}-wal"))
    file_state: list[tuple[int, int, int, int, int]] = []
    for path in paths:
        try:
            stat = path.stat(follow_symlinks=False)
            file_state.append(
                (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            )
        except FileNotFoundError:
            file_state.append((0, 0, 0, 0, 0))
    return tuple(file_state)


def catalog_revision(request: Request, session: Session) -> str:
    """Return the cheap revision used to bind process-local catalog response caches.

    The owning Engine's monotonic commit epoch deterministically detects same-process commits.
    ``PRAGMA data_version`` plus database and WAL identity, size, and timestamps detect commits
    from other connections and processes, checkpoints, and sidecar replacement. Acceptance mode
    has no sidecars and serves an immutable database descriptor, so the same contract is safe in
    both server modes.
    """

    engine = request.app.state.engine
    commit_epoch = _engine_commit_epoch(engine).snapshot()
    data_version = _catalog_data_version(session)
    database = request.app.state.layout.database
    file_state = _catalog_file_state(database)
    return f"{commit_epoch}:{data_version}:{file_state!r}"


class RevisionBoundAggregateCache:
    """Bounded thread-safe LRU for immutable aggregate values.

    Namespace, filter scope, and catalog revision are hashed before storage, so private filter
    text and opaque entity identifiers are not retained in cache keys. Callers must cache only
    immutable aggregate values such as integers and tuples, never page rows or display content.
    """

    def __init__(self, *, max_entries: int = 512) -> None:
        if max_entries < 1:
            raise ValueError("Aggregate cache must retain at least one entry")
        self._max_entries = max_entries
        self._entries: OrderedDict[bytes, object] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def _key(*, namespace: str, scope: str, revision: str) -> bytes:
        framed = f"{namespace}\0{scope}\0{revision}".encode("utf-8", errors="surrogatepass")
        return hashlib.sha256(framed).digest()

    def get_or_compute(
        self,
        *,
        namespace: str,
        scope: str,
        revision: str,
        revision_is_current: Callable[[], bool],
        compute: Callable[[], AggregateT],
    ) -> AggregateT:
        key = self._key(namespace=namespace, scope=scope, revision=revision)
        with self._lock:
            cached = self._entries.get(key, _MISSING)
            if cached is not _MISSING:
                self._entries.move_to_end(key)
        if cached is not _MISSING and revision_is_current():
            return cast(AggregateT, cached)

        # Do not hold the shared LRU lock during an expensive SQLite aggregate. Concurrent cold
        # requests may duplicate work, but they cannot observe a partial cache entry.
        value = compute()
        # Never publish a value under a revision that changed while it was computed. The route-level
        # memo guard must still run after page rows are loaded: Python's sqlite3 legacy transaction
        # mode does not keep independent SELECT statements on one durable read snapshot.
        if not revision_is_current():
            return value
        with self._lock:
            existing = self._entries.get(key, _MISSING)
            if existing is not _MISSING:
                self._entries.move_to_end(key)
                return cast(AggregateT, existing)
            self._entries[key] = value
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
        return value

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


@dataclass(frozen=True, slots=True)
class CatalogAggregateMemo:
    request: Request
    session: Session
    cache: RevisionBoundAggregateCache
    revision: str

    def get_or_compute(
        self,
        *,
        namespace: str,
        scope: str,
        compute: Callable[[], AggregateT],
    ) -> AggregateT:
        return self.cache.get_or_compute(
            namespace=namespace,
            scope=scope,
            revision=self.revision,
            revision_is_current=lambda: (
                catalog_revision(self.request, self.session) == self.revision
            ),
            compute=compute,
        )

    def require_current(self) -> None:
        """Fail closed if aggregates and page rows no longer share one catalog revision."""

        require_catalog_revision_current(
            self.request,
            self.session,
            expected_revision=self.revision,
        )


def require_catalog_revision_current(
    request: Request,
    session: Session,
    *,
    expected_revision: str,
) -> None:
    """Reject a response assembled across a concurrent catalog commit."""

    if catalog_revision(request, session) != expected_revision:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=CATALOG_CHANGED_RETRY_DETAIL,
        )


def _aggregate_cache(request: Request) -> RevisionBoundAggregateCache:
    cache = getattr(request.app.state, "catalog_aggregate_cache", None)
    if isinstance(cache, RevisionBoundAggregateCache):
        return cache
    with _CACHE_INITIALIZATION_LOCK:
        cache = getattr(request.app.state, "catalog_aggregate_cache", None)
        if isinstance(cache, RevisionBoundAggregateCache):
            return cache
        created = RevisionBoundAggregateCache()
        request.app.state.catalog_aggregate_cache = created
        return created


def catalog_aggregate_memo(request: Request, session: Session) -> CatalogAggregateMemo:
    """Bind aggregate lookups to one pre-query catalog revision for this request."""

    return CatalogAggregateMemo(
        request=request,
        session=session,
        cache=_aggregate_cache(request),
        revision=catalog_revision(request, session),
    )
