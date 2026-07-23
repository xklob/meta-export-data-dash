from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from meta_archive.paths import (
    UnsafeArchivePathError,
    open_directory_beneath,
    open_regular_file_beneath,
)

MODEL_ARTIFACT_FINGERPRINT_SCHEMA = "hf-logical-snapshot-v1"
_HASH_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True, slots=True)
class PinnedTransformersArtifact:
    """Content identity for one exact, already-cached Transformers snapshot."""

    snapshot: Path
    fingerprint_sha256: str
    file_count: int
    total_bytes: int
    schema: str = MODEL_ARTIFACT_FINGERPRINT_SCHEMA


def _require_directory(path: Path) -> None:
    try:
        value = path.lstat()
    except FileNotFoundError:
        raise RuntimeError("The pinned local model snapshot is unavailable") from None
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode):
        raise RuntimeError("The pinned local model snapshot is unsafe")


def _stable_entry_key(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _framed_hash_value(digest: hashlib._Hash, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, byteorder="big", signed=False))
    digest.update(value)


def _snapshot_file_digest(handle: object) -> tuple[bytes, int]:
    descriptor = handle.fileno()  # type: ignore[attr-defined]
    opened = os.fstat(descriptor)
    if not stat.S_ISREG(opened.st_mode):
        raise RuntimeError("The pinned local model snapshot is unsafe")
    digest = hashlib.sha256()
    total = 0
    while chunk := handle.read(_HASH_CHUNK_SIZE):  # type: ignore[attr-defined]
        digest.update(chunk)
        total += len(chunk)
    if _stable_entry_key(os.fstat(descriptor)) != _stable_entry_key(opened):
        raise RuntimeError("The pinned local model snapshot changed during fingerprinting")
    return digest.digest(), total


def _inspect_safe_snapshot_tree(
    snapshot: Path,
    *,
    cache_root: Path,
    model_cache: Path,
    fingerprint: bool,
) -> tuple[str | None, int, int]:
    """Validate a snapshot and optionally hash its logical paths and exact file bytes."""

    directory_flags = (
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | int(getattr(os, "O_CLOEXEC", 0))
    )
    file_flags = (
        os.O_RDONLY
        | os.O_NOFOLLOW
        | int(getattr(os, "O_CLOEXEC", 0))
        | int(getattr(os, "O_NONBLOCK", 0))
    )
    snapshot_relative = snapshot.relative_to(model_cache)
    manifest = hashlib.sha256()
    if fingerprint:
        _framed_hash_value(manifest, MODEL_ARTIFACT_FINGERPRINT_SCHEMA.encode("ascii"))
    file_count = 0
    total_bytes = 0

    def record(relative: Path, content_digest: bytes, size: int) -> None:
        nonlocal file_count, total_bytes
        file_count += 1
        total_bytes += size
        if not fingerprint:
            return
        _framed_hash_value(manifest, b"file")
        _framed_hash_value(manifest, relative.as_posix().encode("utf-8"))
        _framed_hash_value(manifest, size.to_bytes(8, byteorder="big", signed=False))
        _framed_hash_value(manifest, content_digest)

    def visit(descriptor: int, relative: Path) -> None:
        opened_directory = os.fstat(descriptor)
        with os.scandir(descriptor) as entries:
            names = sorted(entry.name for entry in entries)
        for name in names:
            value = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            child_relative = relative / name
            if stat.S_ISDIR(value.st_mode):
                if fingerprint:
                    _framed_hash_value(manifest, b"directory")
                    _framed_hash_value(manifest, child_relative.as_posix().encode("utf-8"))
                child = os.open(name, directory_flags, dir_fd=descriptor)
                try:
                    if _stable_entry_key(os.fstat(child)) != _stable_entry_key(value):
                        raise RuntimeError(
                            "The pinned local model snapshot changed during validation"
                        )
                    visit(child, child_relative)
                finally:
                    os.close(child)
            elif stat.S_ISREG(value.st_mode):
                child = os.open(name, file_flags, dir_fd=descriptor)
                try:
                    if _stable_entry_key(os.fstat(child)) != _stable_entry_key(value):
                        raise RuntimeError(
                            "The pinned local model snapshot changed during validation"
                        )
                    if fingerprint:
                        with os.fdopen(child, "rb", buffering=0) as handle:
                            child = -1
                            content_digest, size = _snapshot_file_digest(handle)
                        record(child_relative, content_digest, size)
                finally:
                    if child >= 0:
                        os.close(child)
            elif stat.S_ISLNK(value.st_mode):
                target = os.readlink(name, dir_fd=descriptor)
                if os.path.isabs(target):
                    raise RuntimeError("The pinned local model snapshot is unsafe")
                normalized = Path(
                    os.path.normpath(snapshot_relative / child_relative.parent / target)
                )
                if (
                    not normalized.parts
                    or normalized.parts[0] != "blobs"
                    or ".." in normalized.parts
                ):
                    raise RuntimeError("The pinned local model snapshot is unsafe")
                target_path = model_cache.joinpath(*normalized.parts)
                with open_regular_file_beneath(target_path, model_cache) as handle:
                    opened_target = os.fstat(handle.fileno())
                    if fingerprint:
                        content_digest, size = _snapshot_file_digest(handle)
                        record(child_relative, content_digest, size)
                with open_regular_file_beneath(target_path, model_cache) as confirmation:
                    if _stable_entry_key(os.fstat(confirmation.fileno())) != _stable_entry_key(
                        opened_target
                    ):
                        raise RuntimeError(
                            "The pinned local model snapshot changed during validation"
                        )
                final_value = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                if (
                    _stable_entry_key(final_value) != _stable_entry_key(value)
                    or os.readlink(name, dir_fd=descriptor) != target
                ):
                    raise RuntimeError("The pinned local model snapshot changed during validation")
            else:
                raise RuntimeError("The pinned local model snapshot is unsafe")
        if _stable_entry_key(os.fstat(descriptor)) != _stable_entry_key(opened_directory):
            raise RuntimeError("The pinned local model snapshot changed during validation")

    try:
        snapshot_descriptor = open_directory_beneath(snapshot, cache_root)
        try:
            visit(snapshot_descriptor, Path())
        finally:
            os.close(snapshot_descriptor)
    except (OSError, UnsafeArchivePathError) as exc:
        raise RuntimeError("The pinned local model snapshot is unsafe") from exc
    return manifest.hexdigest() if fingerprint else None, file_count, total_bytes


def _pinned_snapshot_paths(
    cache_dir: Path,
    *,
    model_name: str,
    model_revision: str,
) -> tuple[Path, Path, Path]:
    absolute_cache = Path(os.path.abspath(cache_dir))
    for ancestor in reversed((absolute_cache, *absolute_cache.parents)):
        _require_directory(ancestor)
    model_cache = absolute_cache / f"models--{model_name.replace('/', '--')}"
    snapshots = model_cache / "snapshots"
    snapshot = snapshots / model_revision
    for directory in (model_cache, snapshots, snapshot):
        _require_directory(directory)
    return absolute_cache, model_cache, snapshot


def pinned_transformers_snapshot(
    cache_dir: Path,
    *,
    model_name: str,
    model_revision: str,
) -> Path:
    """Return an already-cached pinned snapshot without creating or downloading anything."""

    absolute_cache, model_cache, snapshot = _pinned_snapshot_paths(
        cache_dir,
        model_name=model_name,
        model_revision=model_revision,
    )
    _inspect_safe_snapshot_tree(
        snapshot,
        cache_root=absolute_cache,
        model_cache=model_cache,
        fingerprint=False,
    )
    return snapshot


def pinned_transformers_artifact(
    cache_dir: Path,
    *,
    model_name: str,
    model_revision: str,
) -> PinnedTransformersArtifact:
    """Fingerprint exact bytes in an already-cached snapshot without downloads or mutation.

    File bytes are read from no-follow descriptors, but a later third-party loader still consumes
    the returned pathname. A caller using pre/post fingerprints therefore assumes the private
    model cache remains quiescent; an actor able to swap and restore entries between both
    observations is outside this pathname-based composition's trust boundary.
    """

    absolute_cache, model_cache, snapshot = _pinned_snapshot_paths(
        cache_dir,
        model_name=model_name,
        model_revision=model_revision,
    )
    fingerprint, file_count, total_bytes = _inspect_safe_snapshot_tree(
        snapshot,
        cache_root=absolute_cache,
        model_cache=model_cache,
        fingerprint=True,
    )
    if fingerprint is None:  # pragma: no cover - guarded by fingerprint=True
        raise RuntimeError("The pinned local model snapshot could not be fingerprinted")
    return PinnedTransformersArtifact(
        snapshot=snapshot,
        fingerprint_sha256=fingerprint,
        file_count=file_count,
        total_bytes=total_bytes,
    )
