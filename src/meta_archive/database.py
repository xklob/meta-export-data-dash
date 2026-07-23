from __future__ import annotations

import os
import sqlite3
import stat
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO

from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from sqlalchemy.pool import QueuePool

from alembic import command
from meta_archive.paths import (
    PRIVATE_FILE_MODE,
    UnsafeArchivePathError,
    archive_entry_stat_beneath,
    open_regular_file_beneath,
)
from meta_archive.services.text import normalize_classifier_token


class CatalogMigrationError(RuntimeError):
    """The catalog schema is unavailable or does not match the application head."""


def _secure_sqlite_file(path: Path, *, create: bool) -> None:
    """Create or repair a private regular SQLite file without following a symlink."""

    try:
        path_stat = path.lstat()
    except FileNotFoundError:
        if not create:
            return
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags, PRIVATE_FILE_MODE)
        except FileExistsError:
            path_stat = path.lstat()
        else:
            os.close(descriptor)
            path_stat = path.lstat()
    if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
        raise UnsafeArchivePathError(f"Unsafe SQLite runtime file: {path}")
    os.chmod(path, PRIVATE_FILE_MODE, follow_symlinks=False)


def secure_sqlite_runtime_files(database_path: Path, *, create_database: bool = False) -> None:
    """Keep the catalog and any current SQLite sidecars owner-readable only."""

    _secure_sqlite_file(database_path, create=create_database)
    for suffix in ("-wal", "-shm", "-journal"):
        _secure_sqlite_file(Path(f"{database_path}{suffix}"), create=False)


def create_catalog_engine(
    database_path: Path,
    *,
    echo: bool = False,
    read_only: bool = False,
    archive_root: Path | None = None,
) -> Engine:
    database_handle: BinaryIO | None = None
    creator: Callable[[], sqlite3.Connection] | None = None
    if read_only:
        if archive_root is None:
            raise CatalogMigrationError("Acceptance mode requires the owning archive root")
        try:
            database_handle = open_regular_file_beneath(database_path, archive_root)
            sidecars = [
                path
                for path in (
                    Path(f"{database_path}-wal"),
                    Path(f"{database_path}-shm"),
                    Path(f"{database_path}-journal"),
                )
                if archive_entry_stat_beneath(path, archive_root) is not None
            ]
            if sidecars:
                raise CatalogMigrationError(
                    "Acceptance mode requires a fully checkpointed catalog without SQLite sidecars"
                )
        except BaseException:
            if database_handle is not None:
                database_handle.close()
            raise
        held_descriptor = database_handle.fileno()

        def connect_read_only() -> sqlite3.Connection:
            return sqlite3.connect(
                f"file:/proc/self/fd/{held_descriptor}?mode=ro&immutable=1",
                uri=True,
                check_same_thread=False,
            )

        creator = connect_read_only
        database_url = "sqlite+pysqlite://"
    else:
        database_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        secure_sqlite_runtime_files(database_path, create_database=True)
        database_url = f"sqlite+pysqlite:///{database_path}"
    engine_options: dict[str, Any] = {
        "echo": echo,
        "future": True,
        "pool_pre_ping": True,
        # Database exceptions must not echo private message or attachment text through bound
        # parameter dumps. Developers can opt into SQL echo explicitly for synthetic fixtures.
        "hide_parameters": not echo,
    }
    if creator is not None:
        engine_options["creator"] = creator
        engine_options["poolclass"] = QueuePool
    try:
        engine = create_engine(database_url, **engine_options)
    except BaseException:
        if database_handle is not None:
            database_handle.close()
        raise
    if database_handle is not None:
        held_handle = database_handle
        engine._meta_archive_read_only_database_handle = held_handle  # type: ignore[attr-defined]

        @event.listens_for(engine, "engine_disposed")
        def close_read_only_database_descriptor(_engine: Engine) -> None:
            if not held_handle.closed:
                held_handle.close()

    @event.listens_for(engine, "connect")
    def configure_sqlite(dbapi_connection: object, _connection_record: object) -> None:
        dbapi_connection.create_function(  # type: ignore[attr-defined]
            "unicode_casefold",
            1,
            lambda value: value.casefold() if isinstance(value, str) else "",
            deterministic=True,
        )
        dbapi_connection.create_function(  # type: ignore[attr-defined]
            "normalize_classifier_token",
            1,
            lambda value: normalize_classifier_token(value) if isinstance(value, str) else "",
            deterministic=True,
        )
        cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
        if read_only:
            # These connection-local pragmas make accidental writes fail even if a future
            # refactor weakens the SQLite URI.  In particular, do not negotiate journal mode or
            # touch the catalog and its sidecars in acceptance mode.
            cursor.execute("PRAGMA query_only=ON")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=10000")
            cursor.execute("PRAGMA temp_store=MEMORY")
            cursor.execute("PRAGMA cache_size=-262144")
        else:
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA busy_timeout=10000")
            cursor.execute("PRAGMA temp_store=MEMORY")
            cursor.execute("PRAGMA cache_size=-262144")
        cursor.close()
        if not read_only:
            secure_sqlite_runtime_files(database_path)

    return engine


def read_only_database_descriptor(engine: Engine) -> int:
    """Return the securely held catalog descriptor for an acceptance read-only engine."""

    handle = getattr(engine, "_meta_archive_read_only_database_handle", None)
    if handle is None or bool(getattr(handle, "closed", True)):
        raise CatalogMigrationError("Acceptance engine is not bound to a live catalog descriptor")
    return int(handle.fileno())


def _migration_config() -> Config:
    project_root = Path(__file__).resolve().parents[2]
    config = Config(str(project_root / "alembic.ini"))
    config.set_main_option("script_location", str(project_root / "alembic"))
    return config


def require_current_migration(engine: Engine) -> None:
    """Fail closed unless the existing catalog is exactly at every Alembic head.

    This check is deliberately read-only.  Acceptance servers must never repair or upgrade the
    archive they are meant to observe.
    """

    expected = frozenset(ScriptDirectory.from_config(_migration_config()).get_heads())
    try:
        with engine.connect() as connection:
            current = frozenset(
                str(value)
                for value in connection.execute(text("SELECT version_num FROM alembic_version"))
                .scalars()
                .all()
            )
    except SQLAlchemyError as exc:
        raise CatalogMigrationError(
            "Acceptance mode requires a readable catalog at the current migration"
        ) from exc
    if current != expected:
        rendered_current = ", ".join(sorted(current)) or "<none>"
        rendered_expected = ", ".join(sorted(expected)) or "<none>"
        raise CatalogMigrationError(
            "Acceptance mode refuses to migrate the catalog: "
            f"current revision {rendered_current}; required revision {rendered_expected}"
        )


@contextmanager
def session_scope(engine: Engine) -> Iterator[Session]:
    with Session(engine) as session, session.begin():
        yield session


def run_migrations(database_path: Path) -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    secure_sqlite_runtime_files(database_path, create_database=True)
    config = _migration_config()
    config.set_main_option("sqlalchemy.url", f"sqlite+pysqlite:///{database_path}")
    command.upgrade(config, "head")
    secure_sqlite_runtime_files(database_path)
