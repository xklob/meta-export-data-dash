from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

from sqlalchemy import Engine

from meta_archive.database import read_only_database_descriptor
from meta_archive.paths import (
    ArchiveLayout,
    archive_entry_stat_beneath,
    open_directory_beneath,
    open_regular_file_beneath,
    validate_directory_tree_beneath,
)

_CATALOG_REVISION_DOMAIN = b"meta-archive-catalog-source-revision-v1\0"
_FRONTEND_REVISION_DOMAIN = b"meta-archive-frontend-manifest-v1\0"
_BACKEND_REVISION_DOMAIN = b"meta-archive-backend-source-manifest-v1\0"


class _HashWriter(Protocol):
    def update(self, value: bytes) -> object: ...


def _framed(parts: Iterable[bytes]) -> Iterable[bytes]:
    for part in parts:
        yield len(part).to_bytes(8, "big")
        yield part


def _sha256_regular_file(path: Path, *, root: Path) -> tuple[int, bytes]:
    """Hash the exact no-follow regular-file descriptor selected by ``path``."""

    with open_regular_file_beneath(path, root) as handle:
        opened_stat = os.fstat(handle.fileno())
        if not stat.S_ISREG(opened_stat.st_mode):
            raise RuntimeError("Acceptance identity refuses a non-regular file")
        content_digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            content_digest.update(chunk)
        final_stat = os.fstat(handle.fileno())
        if _stable_stat_key(final_stat) != _stable_stat_key(opened_stat):
            raise RuntimeError("Acceptance identity input changed while it was hashed")
        return opened_stat.st_size, content_digest.digest()


def _stable_stat_key(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _hash_open_file(descriptor: int) -> tuple[os.stat_result, bytes]:
    opened_stat = os.fstat(descriptor)
    if not stat.S_ISREG(opened_stat.st_mode):
        raise RuntimeError("Acceptance identity refuses a non-regular tree entry")
    digest = hashlib.sha256()
    while chunk := os.read(descriptor, 1024 * 1024):
        digest.update(chunk)
    final_stat = os.fstat(descriptor)
    if _stable_stat_key(final_stat) != _stable_stat_key(opened_stat):
        raise RuntimeError("Acceptance identity tree changed while it was hashed")
    return opened_stat, digest.digest()


def _hash_bound_file(descriptor: int) -> tuple[os.stat_result, bytes]:
    """Hash a held regular descriptor without changing its shared file offset."""

    opened_stat = os.fstat(descriptor)
    if not stat.S_ISREG(opened_stat.st_mode):
        raise RuntimeError("Acceptance identity refuses a non-regular catalog descriptor")
    digest = hashlib.sha256()
    position = 0
    while chunk := os.pread(descriptor, 1024 * 1024, position):
        digest.update(chunk)
        position += len(chunk)
    final_stat = os.fstat(descriptor)
    if _stable_stat_key(final_stat) != _stable_stat_key(opened_stat):
        raise RuntimeError("Acceptance catalog changed while it was hashed")
    return opened_stat, digest.digest()


def _tree_manifest_digest(
    root: Path,
    *,
    domain: bytes,
    archive_root: Path | None = None,
) -> bytes:
    """Hash one exact no-follow directory tree and refuse concurrent changes."""

    if not all(hasattr(os, name) for name in ("O_DIRECTORY", "O_NOFOLLOW")):
        raise RuntimeError("Acceptance identity requires no-follow directory traversal")
    directory_flags = (
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | int(getattr(os, "O_CLOEXEC", 0))
    )
    file_flags = (
        os.O_RDONLY
        | os.O_NOFOLLOW
        | int(getattr(os, "O_CLOEXEC", 0))
        | int(getattr(os, "O_NONBLOCK", 0))
    )
    root_descriptor = (
        open_directory_beneath(root, archive_root)
        if archive_root is not None
        else os.open(root, directory_flags)
    )
    digest = hashlib.sha256(domain)

    def visit(descriptor: int, relative: str) -> None:
        opened_stat = os.fstat(descriptor)
        if not stat.S_ISDIR(opened_stat.st_mode):
            raise RuntimeError("Acceptance identity refuses a non-directory tree entry")
        for part in _framed(
            (
                b"directory",
                relative.encode("utf-8", errors="surrogatepass"),
                str(stat.S_IMODE(opened_stat.st_mode)).encode("ascii"),
            )
        ):
            digest.update(part)
        with os.scandir(descriptor) as entries:
            names = sorted(entry.name for entry in entries)
        for name in names:
            value = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            child_relative = f"{relative}/{name}" if relative else name
            if stat.S_ISDIR(value.st_mode):
                child = os.open(name, directory_flags, dir_fd=descriptor)
                try:
                    child_stat = os.fstat(child)
                    if (child_stat.st_dev, child_stat.st_ino) != (
                        value.st_dev,
                        value.st_ino,
                    ):
                        raise RuntimeError("Acceptance identity tree changed during traversal")
                    visit(child, child_relative)
                finally:
                    os.close(child)
            elif stat.S_ISREG(value.st_mode):
                child = os.open(name, file_flags, dir_fd=descriptor)
                try:
                    file_stat, content_digest = _hash_open_file(child)
                    if (file_stat.st_dev, file_stat.st_ino) != (value.st_dev, value.st_ino):
                        raise RuntimeError("Acceptance identity tree changed during traversal")
                    for part in _framed(
                        (
                            b"file",
                            child_relative.encode("utf-8", errors="surrogatepass"),
                            str(stat.S_IMODE(file_stat.st_mode)).encode("ascii"),
                            str(file_stat.st_size).encode("ascii"),
                            str(file_stat.st_mtime_ns).encode("ascii"),
                            str(file_stat.st_ctime_ns).encode("ascii"),
                            content_digest,
                        )
                    ):
                        digest.update(part)
                finally:
                    os.close(child)
            else:
                raise RuntimeError("Acceptance identity refuses an unsafe tree entry")
        final_stat = os.fstat(descriptor)
        if _stable_stat_key(final_stat) != _stable_stat_key(opened_stat):
            raise RuntimeError("Acceptance identity tree changed during traversal")

    try:
        visit(root_descriptor, "")
    finally:
        os.close(root_descriptor)
    return digest.digest()


def _update_path_state(
    digest: _HashWriter,
    *,
    label: str,
    path: Path,
    root: Path,
) -> None:
    """Bind a catalogued source/blob path without reading private content into memory."""

    value = archive_entry_stat_beneath(path, root)
    if value is None:
        state = b"missing"
    else:
        if stat.S_ISLNK(value.st_mode) or not stat.S_ISREG(value.st_mode):
            state = b"unsafe"
        else:
            # The catalog already commits to the expected full-file SHA-256.  Size plus both
            # modification and inode-change timestamps detect a same-size byte rewrite after
            # that hash was verified, without rereading the roughly 40 GB archive at startup.
            state = f"regular:{value.st_size}:{value.st_mtime_ns}:{value.st_ctime_ns}".encode(
                "ascii"
            )
    for part in _framed((label.encode("utf-8", errors="surrogatepass"), state)):
        digest.update(part)


def _bound_database_stat(layout: ArchiveLayout, engine: Engine) -> os.stat_result:
    descriptor_stat = os.fstat(read_only_database_descriptor(engine))
    path_stat = archive_entry_stat_beneath(layout.database, layout.root)
    if (
        path_stat is None
        or not stat.S_ISREG(path_stat.st_mode)
        or (path_stat.st_dev, path_stat.st_ino) != (descriptor_stat.st_dev, descriptor_stat.st_ino)
    ):
        raise RuntimeError("Acceptance catalog pathname no longer names the served database")
    return descriptor_stat


def catalog_source_revision(layout: ArchiveLayout, engine: Engine) -> str:
    """Bind exact SQLite/FTS and Lance bytes plus catalogued source-file state.

    The SQLite database must be fully checkpointed: acceptance startup refuses WAL, SHM, and
    rollback-journal sidecars rather than altering or ambiguously omitting them.  Hashing the
    stable database descriptor includes normalized rows, schema, triggers, and every FTS shadow
    table.  The no-follow vector-tree manifest includes every Lance file's path, metadata, and
    bytes.  Catalogued expected source/blob SHA-256 values live in SQLite; current size, mtime,
    and ctime additionally bind those files against later same-size byte rewrites without
    rereading the roughly 40 GB source archive.
    """

    digest = hashlib.sha256(_CATALOG_REVISION_DOMAIN)
    _bound_database_stat(layout, engine)
    with engine.raw_connection() as connection:
        cursor = connection.cursor()
        try:
            cursor.execute(
                "SELECT source_chunks.original_path, source_files.relative_path "
                "FROM source_files JOIN source_chunks ON source_chunks.id = source_files.chunk_id "
                "ORDER BY source_chunks.original_path, source_files.relative_path"
            )
            while source_rows := cursor.fetchmany(4_096):
                for chunk_path, relative_path in source_rows:
                    label = f"source:{chunk_path}/{relative_path}"
                    _update_path_state(
                        digest,
                        label=label,
                        path=layout.root / str(chunk_path) / str(relative_path),
                        root=layout.root,
                    )
            cursor.execute(
                "SELECT relative_path FROM blobs WHERE relative_path IS NOT NULL "
                "ORDER BY relative_path"
            )
            while blob_rows := cursor.fetchmany(4_096):
                for (relative_path,) in blob_rows:
                    label = f"blob:{relative_path}"
                    _update_path_state(
                        digest,
                        label=label,
                        path=layout.root / str(relative_path),
                        root=layout.root,
                    )
        finally:
            cursor.close()
    sidecars = tuple(
        path
        for path in (
            Path(f"{layout.database}-wal"),
            Path(f"{layout.database}-shm"),
            Path(f"{layout.database}-journal"),
        )
        if archive_entry_stat_beneath(path, layout.root) is not None
    )
    if sidecars:
        raise RuntimeError("Acceptance identity requires a fully checkpointed SQLite catalog")
    database_stat, database_digest = _hash_bound_file(read_only_database_descriptor(engine))
    database_size = database_stat.st_size
    vector_digest = _tree_manifest_digest(
        layout.vectors,
        domain=b"meta-archive-lance-manifest-v1\0",
        archive_root=layout.root,
    )
    if any(
        archive_entry_stat_beneath(path, layout.root) is not None
        for path in (
            Path(f"{layout.database}-wal"),
            Path(f"{layout.database}-shm"),
            Path(f"{layout.database}-journal"),
        )
    ):
        raise RuntimeError("SQLite sidecars appeared while acceptance identity was computed")
    _bound_database_stat(layout, engine)
    for part in _framed(
        (
            b"sqlite",
            str(database_size).encode("ascii"),
            database_digest,
            b"lance",
            vector_digest,
        )
    ):
        digest.update(part)
    return digest.hexdigest()


def frontend_manifest_digest(static_root: Path, *, project_root: Path) -> str:
    """Hash a deterministic manifest of every built frontend file and its bytes."""

    try:
        with open_regular_file_beneath(static_root / "index.html", project_root):
            pass
        return _tree_manifest_digest(
            static_root,
            domain=_FRONTEND_REVISION_DOMAIN,
            archive_root=project_root,
        ).hex()
    except (OSError, RuntimeError) as exc:
        raise RuntimeError("Acceptance mode requires a completed safe frontend build") from exc


def backend_source_digest(project_root: Path) -> str:
    """Hash the explicit backend source/config allowlist used by the served application."""

    # ``Path.glob`` deliberately skips symlinked directories. Validate the relevant trees first
    # so a changed checkout cannot silently omit a package or migration subtree from evidence.
    validate_directory_tree_beneath(project_root / "src" / "meta_archive", project_root)
    validate_directory_tree_beneath(project_root / "alembic", project_root)
    allowlisted_paths = {
        project_root / "pyproject.toml",
        project_root / "uv.lock",
        project_root / "alembic.ini",
        project_root / "alembic" / "env.py",
        project_root / "alembic" / "script.py.mako",
        *(project_root.glob("src/meta_archive/**/*.py")),
        *(project_root.glob("alembic/versions/*.py")),
    }
    paths = sorted(allowlisted_paths)
    if not paths:
        raise RuntimeError("Acceptance mode could not identify backend source files")
    digest = hashlib.sha256(_BACKEND_REVISION_DOMAIN)
    for path in paths:
        path_stat = path.lstat()
        if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
            raise RuntimeError("Acceptance mode refuses unsafe backend source entries")
        relative = path.relative_to(project_root).as_posix().encode("utf-8")
        size, content_digest = _sha256_regular_file(path, root=project_root)
        for part in _framed((relative, str(size).encode("ascii"), content_digest)):
            digest.update(part)
    return digest.hexdigest()
