from __future__ import annotations

import os
import stat
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal

from meta_archive.config import settings

CANONICAL_DIRECTORIES = frozenset(
    {"manifests", "blobs", "catalog", "library", "trash", "conflicts"}
)
PRIVATE_DIRECTORY_MODE = 0o700
PRIVATE_FILE_MODE = 0o600
PRIVATE_PROCESS_UMASK = 0o077


class ArchiveNotFoundError(RuntimeError):
    pass


class UnsafeArchivePathError(ArchiveNotFoundError):
    """A private runtime path is a symlink or an unexpected filesystem object."""


def _lstat(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None


def regular_file_stat_beneath(path: Path, root: Path) -> os.stat_result:
    """Validate a regular file and every directory from ``root`` without following links.

    Catalog paths are fixed by content identifiers, but a symlink in an intermediate shard would
    otherwise be followed by ``Path.open``/``Path.stat``.  This static validation is used by local
    workers before handing a pathname to a sandboxed child.  HTTP streaming additionally uses
    :func:`open_regular_file_beneath` so the checked object itself remains bound to a descriptor.
    """

    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise UnsafeArchivePathError(f"Private file is outside the archive root: {path}") from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise UnsafeArchivePathError(f"Invalid private archive path: {path}")

    current = root
    for part in relative.parts[:-1]:
        current = current / part
        current_stat = _lstat(current)
        if (
            current_stat is None
            or stat.S_ISLNK(current_stat.st_mode)
            or not stat.S_ISDIR(current_stat.st_mode)
        ):
            raise UnsafeArchivePathError(f"Unsafe private archive directory: {current}")

    path_stat = _lstat(path)
    if path_stat is None or stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
        raise UnsafeArchivePathError(f"Unsafe private archive file: {path}")
    return path_stat


def open_regular_file_beneath(path: Path, root: Path) -> BinaryIO:
    """Open one regular archive file through no-follow directory descriptors.

    Traversing with ``dir_fd`` and ``O_NOFOLLOW`` binds streaming to the validated inode and closes
    the pathname check/open race.  This project already requires POSIX for its local extraction
    sandbox; fail closed if the platform cannot provide the necessary flags.
    """

    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise UnsafeArchivePathError(f"Private file is outside the archive root: {path}") from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise UnsafeArchivePathError(f"Invalid private archive path: {path}")
    if not all(hasattr(os, name) for name in ("O_DIRECTORY", "O_NOFOLLOW")):
        raise UnsafeArchivePathError("Secure no-follow archive file opens are unavailable")

    close_on_exec = getattr(os, "O_CLOEXEC", 0)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | close_on_exec
    file_flags = os.O_RDONLY | os.O_NOFOLLOW | close_on_exec | int(getattr(os, "O_NONBLOCK", 0))
    directory_fd: int | None = None
    file_fd: int | None = None
    try:
        directory_fd = os.open(root, directory_flags)
        for part in relative.parts[:-1]:
            next_fd = os.open(part, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        leaf_stat = os.stat(relative.parts[-1], dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISLNK(leaf_stat.st_mode) or not stat.S_ISREG(leaf_stat.st_mode):
            raise UnsafeArchivePathError(f"Unsafe private archive file: {path}")
        file_fd = os.open(relative.parts[-1], file_flags, dir_fd=directory_fd)
        file_stat = os.fstat(file_fd)
        if not stat.S_ISREG(file_stat.st_mode) or (file_stat.st_dev, file_stat.st_ino) != (
            leaf_stat.st_dev,
            leaf_stat.st_ino,
        ):
            raise UnsafeArchivePathError(f"Unsafe private archive file: {path}")
        handle = os.fdopen(file_fd, "rb", buffering=0)
        file_fd = None
        return handle
    except (OSError, ValueError) as exc:
        raise UnsafeArchivePathError(
            f"Could not securely open private archive file: {path}"
        ) from exc
    finally:
        if file_fd is not None:
            os.close(file_fd)
        if directory_fd is not None:
            os.close(directory_fd)


def _relative_archive_parts(path: Path, root: Path) -> tuple[str, ...]:
    """Return lexical path components below ``root`` without resolving symlinks."""

    absolute_root = Path(os.path.abspath(root))
    absolute_path = Path(os.path.abspath(path))
    try:
        relative = absolute_path.relative_to(absolute_root)
    except ValueError as exc:
        raise UnsafeArchivePathError(f"Private path is outside the archive root: {path}") from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise UnsafeArchivePathError(f"Invalid private archive path: {path}")
    return relative.parts


def _open_directory_parts(
    root: Path,
    parts: tuple[str, ...],
    *,
    create: bool = False,
    mode: int = PRIVATE_DIRECTORY_MODE,
) -> int:
    """Open one directory chain with ``openat`` semantics and no symlink traversal."""

    if not all(hasattr(os, name) for name in ("O_DIRECTORY", "O_NOFOLLOW")):
        raise UnsafeArchivePathError("Secure no-follow archive directory opens are unavailable")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | int(getattr(os, "O_CLOEXEC", 0))
    descriptor: int | None = None
    try:
        descriptor = os.open(root, flags)
        for part in parts:
            created = False
            if create:
                try:
                    os.mkdir(part, mode=mode, dir_fd=descriptor)
                    created = True
                except FileExistsError:
                    pass
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            try:
                if created:
                    # Persist each newly linked directory before callers move the only live copy
                    # of private data below it.  A later fsync of only the leaf parent cannot make
                    # a missing ancestor durable after power loss.
                    os.fsync(next_descriptor)
                    os.fsync(descriptor)
            except BaseException:
                os.close(next_descriptor)
                raise
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise UnsafeArchivePathError(
            "Could not securely traverse private archive directories"
        ) from exc


def open_directory_beneath(path: Path, root: Path) -> int:
    """Open one existing archive directory through a no-follow descriptor chain."""

    return _open_directory_parts(root, _relative_archive_parts(path, root))


def archive_entry_stat_beneath(path: Path, root: Path) -> os.stat_result | None:
    """Stat one entry through a no-follow parent chain, without following the leaf."""

    parts = _relative_archive_parts(path, root)
    try:
        parent_descriptor = _open_directory_parts(root, parts[:-1])
    except UnsafeArchivePathError as exc:
        if isinstance(exc.__cause__, FileNotFoundError):
            return None
        raise
    try:
        try:
            return os.stat(parts[-1], dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return None
    finally:
        os.close(parent_descriptor)


def create_directory_beneath(
    path: Path,
    root: Path,
    *,
    mode: int = PRIVATE_DIRECTORY_MODE,
    exist_ok: bool = True,
) -> None:
    """Create a private directory and its parents without accepting symlink components."""

    parts = _relative_archive_parts(path, root)
    parent_descriptor = _open_directory_parts(root, parts[:-1], create=True, mode=mode)
    try:
        try:
            os.mkdir(parts[-1], mode=mode, dir_fd=parent_descriptor)
        except FileExistsError as exc:
            if not exist_ok:
                raise UnsafeArchivePathError(
                    f"Private archive directory already exists: {path}"
                ) from exc
        directory_flags = (
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | int(getattr(os, "O_CLOEXEC", 0))
        )
        descriptor = os.open(parts[-1], directory_flags, dir_fd=parent_descriptor)
        try:
            os.fchmod(descriptor, mode)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.fsync(parent_descriptor)
    except OSError as exc:
        raise UnsafeArchivePathError(
            f"Could not securely create private directory: {path}"
        ) from exc
    finally:
        os.close(parent_descriptor)


def replace_entry_beneath(
    source: Path,
    destination: Path,
    root: Path,
    *,
    expected_kind: Literal["file", "directory"],
    replace_existing: bool = False,
) -> None:
    """Move one entry below ``root`` without following either parent chain or source leaf."""

    source_parts = _relative_archive_parts(source, root)
    destination_parts = _relative_archive_parts(destination, root)
    source_parent = _open_directory_parts(root, source_parts[:-1])
    destination_parent: int | None = None
    try:
        destination_parent = _open_directory_parts(root, destination_parts[:-1], create=True)
        source_stat = os.stat(source_parts[-1], dir_fd=source_parent, follow_symlinks=False)
        expected = (
            stat.S_ISREG(source_stat.st_mode)
            if expected_kind == "file"
            else stat.S_ISDIR(source_stat.st_mode)
        )
        if stat.S_ISLNK(source_stat.st_mode) or not expected:
            raise UnsafeArchivePathError(f"Unsafe archive {expected_kind} move source: {source}")
        try:
            destination_stat = os.stat(
                destination_parts[-1],
                dir_fd=destination_parent,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            destination_stat = None
        if destination_stat is not None:
            destination_kind_matches = (
                stat.S_ISREG(destination_stat.st_mode)
                if expected_kind == "file"
                else stat.S_ISDIR(destination_stat.st_mode)
            )
            if (
                not replace_existing
                or stat.S_ISLNK(destination_stat.st_mode)
                or not destination_kind_matches
            ):
                raise UnsafeArchivePathError(
                    f"Archive move destination already exists or is unsafe: {destination}"
                )
        os.replace(
            source_parts[-1],
            destination_parts[-1],
            src_dir_fd=source_parent,
            dst_dir_fd=destination_parent,
        )
        moved_stat = os.stat(
            destination_parts[-1],
            dir_fd=destination_parent,
            follow_symlinks=False,
        )
        if (moved_stat.st_dev, moved_stat.st_ino, moved_stat.st_mode) != (
            source_stat.st_dev,
            source_stat.st_ino,
            source_stat.st_mode,
        ):
            raise UnsafeArchivePathError("Archive move source changed during the operation")
        # Make the destination link durable before the source unlink.  Reversing this order can
        # lose the only durable name for the inode if power fails between the two fsync calls.
        os.fsync(destination_parent)
        os.fsync(source_parent)
    except OSError as exc:
        raise UnsafeArchivePathError("Could not securely move private archive entry") from exc
    finally:
        os.close(source_parent)
        if destination_parent is not None:
            os.close(destination_parent)


def unlink_regular_file_beneath(path: Path, root: Path, *, missing_ok: bool = False) -> bool:
    """Unlink one regular file through a no-follow parent descriptor."""

    parts = _relative_archive_parts(path, root)
    parent_descriptor = _open_directory_parts(root, parts[:-1])
    try:
        try:
            value = os.stat(parts[-1], dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            if missing_ok:
                return False
            raise
        if stat.S_ISLNK(value.st_mode) or not stat.S_ISREG(value.st_mode):
            raise UnsafeArchivePathError(f"Unsafe archive file removal target: {path}")
        os.unlink(parts[-1], dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
        return True
    except OSError as exc:
        raise UnsafeArchivePathError(
            f"Could not securely remove private archive file: {path}"
        ) from exc
    finally:
        os.close(parent_descriptor)


def _validate_tree_descriptor(descriptor: int) -> None:
    with os.scandir(descriptor) as entries:
        for entry in entries:
            value = entry.stat(follow_symlinks=False)
            if stat.S_ISDIR(value.st_mode):
                child = os.open(
                    entry.name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | int(getattr(os, "O_CLOEXEC", 0)),
                    dir_fd=descriptor,
                )
                try:
                    _validate_tree_descriptor(child)
                finally:
                    os.close(child)
            elif not stat.S_ISREG(value.st_mode):
                raise UnsafeArchivePathError("Archive tree contains an unsafe entry")


def validate_directory_tree_beneath(path: Path, root: Path) -> None:
    """Require an existing descendant tree containing only no-follow directories/files."""

    descriptor = open_directory_beneath(path, root)
    try:
        try:
            _validate_tree_descriptor(descriptor)
        except OSError as exc:
            raise UnsafeArchivePathError("Archive tree contains an unsafe entry") from exc
    finally:
        os.close(descriptor)


def _remove_tree_descriptor(descriptor: int) -> None:
    with os.scandir(descriptor) as entries:
        names = sorted(entry.name for entry in entries)
    for name in names:
        value = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if stat.S_ISDIR(value.st_mode):
            child = os.open(
                name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | int(getattr(os, "O_CLOEXEC", 0)),
                dir_fd=descriptor,
            )
            try:
                _remove_tree_descriptor(child)
            finally:
                os.close(child)
            os.rmdir(name, dir_fd=descriptor)
        elif stat.S_ISREG(value.st_mode):
            os.unlink(name, dir_fd=descriptor)
        else:
            raise UnsafeArchivePathError("Archive removal tree changed to an unsafe entry")
    os.fsync(descriptor)


def remove_directory_tree_beneath(path: Path, root: Path, *, missing_ok: bool = True) -> bool:
    """Remove an exact descendant tree without ever traversing symlinks."""

    parts = _relative_archive_parts(path, root)
    parent_descriptor = _open_directory_parts(root, parts[:-1])
    directory_descriptor: int | None = None
    try:
        try:
            value = os.stat(parts[-1], dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            if missing_ok:
                return False
            raise
        if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode):
            raise UnsafeArchivePathError(f"Unsafe archive directory removal target: {path}")
        directory_descriptor = os.open(
            parts[-1],
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | int(getattr(os, "O_CLOEXEC", 0)),
            dir_fd=parent_descriptor,
        )
        _validate_tree_descriptor(directory_descriptor)
        _remove_tree_descriptor(directory_descriptor)
        os.close(directory_descriptor)
        directory_descriptor = None
        os.rmdir(parts[-1], dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
        return True
    except OSError as exc:
        raise UnsafeArchivePathError(
            f"Could not securely remove private archive tree: {path}"
        ) from exc
    finally:
        if directory_descriptor is not None:
            os.close(directory_descriptor)
        os.close(parent_descriptor)


def _ensure_private_directory(path: Path) -> None:
    """Create or repair one owner-only directory without accepting a symlink."""

    path_stat = _lstat(path)
    if path_stat is None:
        with suppress(FileExistsError):
            path.mkdir(mode=PRIVATE_DIRECTORY_MODE)
        # Another local worker may have created the path between lstat and mkdir. Validate the
        # resulting object below rather than treating that race as an error.
        path_stat = _lstat(path)
    if path_stat is None or stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISDIR(path_stat.st_mode):
        raise UnsafeArchivePathError(f"Unsafe archive runtime directory: {path}")
    os.chmod(path, PRIVATE_DIRECTORY_MODE, follow_symlinks=False)


def _ensure_private_file(path: Path) -> None:
    """Repair one existing mutable owner-owned file without following symlinks."""

    path_stat = _lstat(path)
    if path_stat is None:
        return
    if (
        stat.S_ISLNK(path_stat.st_mode)
        or not stat.S_ISREG(path_stat.st_mode)
        or path_stat.st_uid != os.geteuid()
        or path_stat.st_nlink != 1
    ):
        raise UnsafeArchivePathError(f"Unsafe archive runtime file: {path}")
    os.chmod(path, PRIVATE_FILE_MODE, follow_symlinks=False)


def _private_tree_files(root: Path) -> Iterator[Path]:
    """Yield regular files below a small runtime metadata tree without following links."""

    root_stat = _lstat(root)
    if root_stat is None:
        return
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise UnsafeArchivePathError(f"Unsafe archive runtime directory: {root}")
    for current_root, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(current_root)
        safe_directories: list[str] = []
        for name in directory_names:
            candidate = current / name
            candidate_stat = candidate.lstat()
            if stat.S_ISLNK(candidate_stat.st_mode) or not stat.S_ISDIR(candidate_stat.st_mode):
                raise UnsafeArchivePathError(f"Unsafe archive runtime directory: {candidate}")
            safe_directories.append(name)
        directory_names[:] = safe_directories
        for name in file_names:
            candidate = current / name
            candidate_stat = candidate.lstat()
            if stat.S_ISLNK(candidate_stat.st_mode) or not stat.S_ISREG(candidate_stat.st_mode):
                raise UnsafeArchivePathError(f"Unsafe archive runtime file: {candidate}")
            yield candidate


def resolve_archive_root(explicit: Path | None = None, *, cwd: Path | None = None) -> Path:
    if explicit is not None:
        requested = explicit.expanduser()
        requested_stat = _lstat(requested)
        if requested_stat is None:
            raise ArchiveNotFoundError(f"Archive root does not exist: {requested}")
        if stat.S_ISLNK(requested_stat.st_mode) or not stat.S_ISDIR(requested_stat.st_mode):
            raise UnsafeArchivePathError(
                f"Archive root must be a non-symlink directory: {requested}"
            )
        return requested.resolve()

    if settings.archive_root is not None:
        requested = settings.archive_root.expanduser()
        requested_stat = _lstat(requested)
        if requested_stat is None:
            raise ArchiveNotFoundError(f"META_ARCHIVE_ROOT does not exist: {requested}")
        if stat.S_ISLNK(requested_stat.st_mode) or not stat.S_ISDIR(requested_stat.st_mode):
            raise UnsafeArchivePathError(
                f"META_ARCHIVE_ROOT must be a non-symlink directory: {requested}"
            )
        return requested.resolve()

    search_root = (cwd or Path.cwd()).resolve()
    candidates: list[Path] = []
    for path in search_root.iterdir():
        if not path.name.startswith("meta-backup"):
            continue
        path_stat = _lstat(path)
        if path_stat is not None and stat.S_ISLNK(path_stat.st_mode):
            raise UnsafeArchivePathError(
                f"Automatically discovered archive root must be a non-symlink directory: {path}"
            )
        if path_stat is not None and stat.S_ISDIR(path_stat.st_mode):
            candidates.append(path)
    candidates.sort()
    if len(candidates) != 1:
        raise ArchiveNotFoundError(
            "Expected exactly one directory starting with 'meta-backup' in "
            f"{search_root}; found {len(candidates)}. Pass --archive-root explicitly."
        )
    return candidates[0]


@dataclass(frozen=True, slots=True)
class ArchiveLayout:
    root: Path

    @classmethod
    def discover(cls, explicit: Path | None = None, *, cwd: Path | None = None) -> ArchiveLayout:
        return cls(resolve_archive_root(explicit, cwd=cwd))

    @property
    def catalog(self) -> Path:
        return self.root / "catalog"

    @property
    def database(self) -> Path:
        return self.catalog / "facebook.sqlite"

    @property
    def vectors(self) -> Path:
        return self.catalog / "vectors"

    @property
    def thumbnails(self) -> Path:
        return self.catalog / "thumbnails"

    @property
    def derivative_manifest(self) -> Path:
        return self.thumbnails / "manifest.json"

    @property
    def models(self) -> Path:
        return self.catalog / "models"

    @property
    def analytics_cache(self) -> Path:
        return self.catalog / "analytics-cache"

    @property
    def manifests(self) -> Path:
        return self.root / "manifests"

    @property
    def original_files_manifest(self) -> Path:
        return self.manifests / "original-files.jsonl"

    @property
    def original_directories_manifest(self) -> Path:
        return self.manifests / "original-directories.jsonl"

    @property
    def relocation_manifest(self) -> Path:
        return self.manifests / "relocation.jsonl"

    @property
    def blobs(self) -> Path:
        return self.root / "blobs" / "sha256"

    @property
    def library(self) -> Path:
        return self.root / "library"

    @property
    def library_completion_marker(self) -> Path:
        """Marker atomically promoted only after a complete filesystem-library build."""

        return self.library / "manifest.json"

    @property
    def trash(self) -> Path:
        return self.root / "trash"

    @property
    def conflicts(self) -> Path:
        return self.root / "conflicts"

    @property
    def access_token_file(self) -> Path:
        return self.catalog / "access-token"

    def blob_path(self, sha256: str) -> Path:
        return self.blobs / sha256[:2] / sha256

    def derivative_path(
        self,
        sha256: str,
        kind: str,
        extension: str,
        *,
        pipeline_version: str,
    ) -> Path:
        """Return the fixed-ID path for a generated media derivative.

        Callers must supply service-owned constants, never user-controlled path fragments.
        """

        return self.thumbnails / pipeline_version / sha256[:2] / sha256 / f"{kind}.{extension}"

    def chunk_directories(self) -> list[Path]:
        return sorted(
            path
            for path in self.root.iterdir()
            if path.is_dir()
            and path.name not in CANONICAL_DIRECTORIES
            and not path.name.startswith(".")
        )

    def private_runtime_directories(self) -> tuple[Path, ...]:
        """Return the ordered owner-only boundary for generated and consolidated data."""

        return (
            self.root,
            self.catalog,
            self.vectors,
            self.thumbnails,
            self.models,
            self.analytics_cache,
            self.manifests,
            self.manifests / "import-runs",
            self.manifests / "integrity-reports",
            self.manifests / "source-retirement",
            self.root / "blobs",
            self.blobs,
            self.library,
            self.trash,
            self.conflicts,
        )

    def private_mutable_files(self) -> tuple[Path, ...]:
        """Return existing mutable catalog/manifests that must be owner-only.

        Immutable blob and derivative payloads deliberately remain read-only. Their 0700 archive
        ancestry supplies the privacy boundary without making generated hardlinks writable.
        """

        candidates = [
            self.database,
            Path(f"{self.database}-wal"),
            Path(f"{self.database}-shm"),
            self.catalog / "facebook.pre-ingest.sqlite",
            self.access_token_file,
            self.original_files_manifest,
            self.original_directories_manifest,
            self.relocation_manifest,
        ]
        catalog_stat = _lstat(self.catalog)
        if catalog_stat is not None:
            if stat.S_ISLNK(catalog_stat.st_mode) or not stat.S_ISDIR(catalog_stat.st_mode):
                raise UnsafeArchivePathError(f"Unsafe archive runtime directory: {self.catalog}")
            candidates.extend(
                path
                for path in self.catalog.iterdir()
                if path.name.endswith(".lock") or path.name.endswith(".sqlite-journal")
            )
        for metadata_root in (
            self.analytics_cache,
            self.manifests / "import-runs",
            self.manifests / "integrity-reports",
            self.manifests / "source-retirement",
        ):
            candidates.extend(_private_tree_files(metadata_root))
        return tuple(dict.fromkeys(candidates))

    def private_permission_findings(self) -> list[str]:
        """Report unsafe types or group/other access without mutating the archive."""

        findings: list[str] = []
        for path in self.private_runtime_directories():
            path_stat = _lstat(path)
            relative = "." if path == self.root else path.relative_to(self.root).as_posix()
            if path_stat is None:
                continue
            if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISDIR(path_stat.st_mode):
                findings.append(f"unsafe directory {relative}")
            elif stat.S_IMODE(path_stat.st_mode) & 0o077:
                findings.append(f"directory {relative} is not owner-only")
        try:
            mutable_files = self.private_mutable_files()
        except UnsafeArchivePathError as exc:
            findings.append(str(exc).replace(str(self.root), "<archive>"))
            mutable_files = ()
        for path in mutable_files:
            path_stat = _lstat(path)
            if path_stat is None:
                continue
            try:
                relative = path.relative_to(self.root).as_posix()
            except ValueError:
                relative = "<outside-archive>"
            if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
                findings.append(f"unsafe mutable file {relative}")
            elif stat.S_IMODE(path_stat.st_mode) & 0o077:
                findings.append(f"mutable file {relative} is not owner-only")
        return sorted(set(findings))

    def ensure_runtime_directories(self) -> None:
        """Create/repair private runtime boundaries and install a restrictive process umask."""

        # Every worker in this process inherits the private default, including libraries that
        # create model caches, SQLite sidecars, Playwright-adjacent logs, or temporary manifests.
        os.umask(PRIVATE_PROCESS_UMASK)
        for path in self.private_runtime_directories():
            _ensure_private_directory(path)
        for path in self.private_mutable_files():
            _ensure_private_file(path)
