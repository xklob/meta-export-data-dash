from __future__ import annotations

import ctypes
import errno
import hashlib
import itertools
import json
import os
import stat
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


class SecurePublicationError(RuntimeError):
    """A private publication path or inode could not be proven safe."""


FaultInjector = Callable[[str], None]

_DIRECTORY_MODE = 0o700
_MUTABLE_FILE_MODE = 0o600
_DEFAULT_FINAL_FILE_MODE = 0o400
_READ_CHUNK = 1024 * 1024
_MAX_PRIVATE_FILE_BYTES = 1024 * 1024 * 1024
_RENAME_EXCHANGE = 2
_AT_EMPTY_PATH = 0x1000
_AT_FDCWD = -100
_AT_SYMLINK_FOLLOW = 0x400


@dataclass(frozen=True, slots=True)
class DirectoryIdentity:
    device: int
    inode: int
    uid: int
    mode: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> DirectoryIdentity:
        return cls(
            device=int(value.st_dev),
            inode=int(value.st_ino),
            uid=int(value.st_uid),
            mode=stat.S_IMODE(value.st_mode),
        )

    def matches(self, value: os.stat_result) -> bool:
        return (
            stat.S_ISDIR(value.st_mode)
            and not stat.S_ISLNK(value.st_mode)
            and self.device == int(value.st_dev)
            and self.inode == int(value.st_ino)
            and self.uid == int(value.st_uid)
            and self.mode == stat.S_IMODE(value.st_mode)
        )


@dataclass(frozen=True, slots=True)
class PrivateFileIdentity:
    device: int
    inode: int
    uid: int
    mode: int
    link_count: int
    size_bytes: int
    modified_ns: int
    changed_ns: int
    sha256: str

    @classmethod
    def from_stat_and_payload(cls, value: os.stat_result, payload: bytes) -> PrivateFileIdentity:
        return cls(
            device=int(value.st_dev),
            inode=int(value.st_ino),
            uid=int(value.st_uid),
            mode=stat.S_IMODE(value.st_mode),
            link_count=int(value.st_nlink),
            size_bytes=int(value.st_size),
            modified_ns=int(value.st_mtime_ns),
            changed_ns=int(value.st_ctime_ns),
            sha256=hashlib.sha256(payload).hexdigest(),
        )

    def matches(self, value: os.stat_result, payload: bytes) -> bool:
        return (
            stat.S_ISREG(value.st_mode)
            and not stat.S_ISLNK(value.st_mode)
            and self.device == int(value.st_dev)
            and self.inode == int(value.st_ino)
            and self.uid == int(value.st_uid)
            and self.mode == stat.S_IMODE(value.st_mode)
            and self.link_count == int(value.st_nlink)
            and self.size_bytes == int(value.st_size)
            and self.modified_ns == int(value.st_mtime_ns)
            and self.changed_ns == int(value.st_ctime_ns)
            and self.sha256 == hashlib.sha256(payload).hexdigest()
        )


@dataclass(frozen=True, slots=True)
class PrivateFileSnapshot:
    relative_path: str
    payload: bytes
    identity: PrivateFileIdentity


@dataclass(frozen=True, slots=True)
class PublicationReceipt:
    relative_path: str
    identity: PrivateFileIdentity


@dataclass(frozen=True, slots=True)
class StagingPreparation:
    staging_relative_path: str
    directories: tuple[tuple[str, DirectoryIdentity], ...]


@dataclass(frozen=True, slots=True)
class StagingTreeEntry:
    relative_path: str
    kind: str
    device: int
    inode: int
    uid: int
    mode: int
    size_bytes: int
    modified_ns: int
    link_count: int

    @classmethod
    def from_stat(cls, relative_path: str, value: os.stat_result, *, kind: str) -> StagingTreeEntry:
        return cls(
            relative_path=relative_path,
            kind=kind,
            device=int(value.st_dev),
            inode=int(value.st_ino),
            uid=int(value.st_uid),
            mode=stat.S_IMODE(value.st_mode),
            size_bytes=int(value.st_size),
            modified_ns=int(value.st_mtime_ns),
            link_count=int(value.st_nlink),
        )

    def matches(self, value: os.stat_result, *, cleanup: bool = False) -> bool:
        expected_type = stat.S_ISDIR if self.kind == "directory" else stat.S_ISREG
        base_matches = (
            expected_type(value.st_mode)
            and not stat.S_ISLNK(value.st_mode)
            and self.device == int(value.st_dev)
            and self.inode == int(value.st_ino)
            and self.uid == int(value.st_uid)
            and self.mode == stat.S_IMODE(value.st_mode)
        )
        if not base_matches or (cleanup and self.kind == "directory"):
            return base_matches
        return (
            self.size_bytes == int(value.st_size)
            and self.modified_ns == int(value.st_mtime_ns)
            and (cleanup or self.link_count == int(value.st_nlink))
        )

    def to_json(self) -> dict[str, int | str]:
        return {
            "relative_path": self.relative_path,
            "kind": self.kind,
            "device": self.device,
            "inode": self.inode,
            "uid": self.uid,
            "mode": self.mode,
            "size_bytes": self.size_bytes,
            "modified_ns": self.modified_ns,
            "link_count": self.link_count,
        }

    @classmethod
    def from_json(cls, value: object) -> StagingTreeEntry:
        if not isinstance(value, dict):
            raise SecurePublicationError("Staging receipt entry is invalid")
        required = {
            "relative_path",
            "kind",
            "device",
            "inode",
            "uid",
            "mode",
            "size_bytes",
            "modified_ns",
            "link_count",
        }
        if set(value) != required or value.get("kind") not in {"directory", "file"}:
            raise SecurePublicationError("Staging receipt entry has invalid fields")
        relative_path = value.get("relative_path")
        if not isinstance(relative_path, str):
            raise SecurePublicationError("Staging receipt path is invalid")
        numeric: dict[str, int] = {}
        for key in required - {"relative_path", "kind"}:
            item = value.get(key)
            if not isinstance(item, int) or isinstance(item, bool) or item < 0:
                raise SecurePublicationError("Staging receipt identity is invalid")
            numeric[key] = item
        return cls(
            relative_path=relative_path,
            kind=str(value["kind"]),
            device=numeric["device"],
            inode=numeric["inode"],
            uid=numeric["uid"],
            mode=numeric["mode"],
            size_bytes=numeric["size_bytes"],
            modified_ns=numeric["modified_ns"],
            link_count=numeric["link_count"],
        )


@dataclass(frozen=True, slots=True)
class SealedStagingTree:
    staging_relative_path: str
    entries: tuple[StagingTreeEntry, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "version": 2,
            "staging_relative_path": self.staging_relative_path,
            "entries": [entry.to_json() for entry in self.entries],
        }

    @classmethod
    def from_json(cls, value: object) -> SealedStagingTree:
        if (
            not isinstance(value, dict)
            or set(value) != {"version", "staging_relative_path", "entries"}
            or value.get("version") != 2
            or not isinstance(value.get("staging_relative_path"), str)
            or not isinstance(value.get("entries"), list)
        ):
            raise SecurePublicationError("Staging receipt is invalid")
        entries = tuple(StagingTreeEntry.from_json(item) for item in value["entries"])
        if not entries or entries[0].relative_path != "." or entries[0].kind != "directory":
            raise SecurePublicationError("Staging receipt root is invalid")
        paths = [entry.relative_path for entry in entries]
        if len(paths) != len(set(paths)):
            raise SecurePublicationError("Staging receipt contains duplicate paths")
        return cls(str(value["staging_relative_path"]), entries)


@dataclass(slots=True)
class _BoundDirectory:
    relative_path: str
    descriptor: int
    identity: DirectoryIdentity
    created: bool
    name_in_parent: str | None


def _fault(injector: FaultInjector | None, event: str) -> None:
    if injector is not None:
        injector(event)


def _relative_parts(relative_path: str | PurePosixPath) -> tuple[str, ...]:
    raw = str(relative_path)
    pure = PurePosixPath(raw)
    if (
        not raw
        or pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} or "\x00" in part or "\\" in part for part in pure.parts)
    ):
        raise SecurePublicationError("Private publication path is invalid")
    return pure.parts


def _directory_flags() -> int:
    required = ("O_DIRECTORY", "O_NOFOLLOW")
    if not all(hasattr(os, name) for name in required):
        raise SecurePublicationError("Secure no-follow directory operations are unavailable")
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | int(getattr(os, "O_CLOEXEC", 0))


def _file_flags(*, writable: bool = False) -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        raise SecurePublicationError("Secure no-follow file operations are unavailable")
    access = os.O_RDWR if writable else os.O_RDONLY
    return (
        access
        | os.O_NOFOLLOW
        | int(getattr(os, "O_CLOEXEC", 0))
        | int(getattr(os, "O_NONBLOCK", 0))
    )


def _validate_directory(value: os.stat_result, *, private: bool = True) -> DirectoryIdentity:
    if (
        stat.S_ISLNK(value.st_mode)
        or not stat.S_ISDIR(value.st_mode)
        or int(value.st_uid) != os.geteuid()
        or (private and stat.S_IMODE(value.st_mode) & 0o077)
    ):
        raise SecurePublicationError("Private publication directory is unsafe")
    return DirectoryIdentity.from_stat(value)


def _validate_regular_file(
    value: os.stat_result,
    *,
    expected_mode: int | None = None,
    require_single_link: bool = True,
) -> None:
    if (
        stat.S_ISLNK(value.st_mode)
        or not stat.S_ISREG(value.st_mode)
        or int(value.st_uid) != os.geteuid()
        or (require_single_link and int(value.st_nlink) != 1)
        or (expected_mode is not None and stat.S_IMODE(value.st_mode) != expected_mode)
    ):
        raise SecurePublicationError("Private publication file is unsafe")


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return (int(left.st_dev), int(left.st_ino)) == (int(right.st_dev), int(right.st_ino))


def _verify_root_binding(root: Path, bound: _BoundDirectory) -> None:
    try:
        current = os.stat(root, follow_symlinks=False)
    except OSError as exc:
        raise SecurePublicationError("Archive root changed during private publication") from exc
    if not bound.identity.matches(current):
        raise SecurePublicationError("Archive root changed during private publication")


def _verify_chain(root: Path, chain: Sequence[_BoundDirectory]) -> None:
    if not chain:
        raise SecurePublicationError("Private publication directory chain is empty")
    _verify_root_binding(root, chain[0])
    for parent, child in itertools.pairwise(chain):
        if child.name_in_parent is None:
            raise SecurePublicationError("Private publication directory binding is invalid")
        try:
            value = os.stat(
                child.name_in_parent,
                dir_fd=parent.descriptor,
                follow_symlinks=False,
            )
            descriptor_value = os.fstat(child.descriptor)
        except OSError as exc:
            raise SecurePublicationError("Private publication directory changed") from exc
        if not child.identity.matches(value) or not _same_inode(value, descriptor_value):
            raise SecurePublicationError("Private publication directory changed")


def _fsync_descriptor(
    descriptor: int,
    *,
    label: str,
    fault_injector: FaultInjector | None,
) -> None:
    _fault(fault_injector, f"before_fsync:{label}")
    os.fsync(descriptor)
    _fault(fault_injector, f"after_fsync:{label}")


@contextmanager
def _open_directory_chain(
    root: Path,
    directory_parts: Sequence[str],
    *,
    create: bool,
    fault_injector: FaultInjector | None = None,
) -> Iterator[list[_BoundDirectory]]:
    chain: list[_BoundDirectory] = []
    try:
        root_descriptor = os.open(root, _directory_flags())
        try:
            root_identity = _validate_directory(os.fstat(root_descriptor))
        except BaseException:
            os.close(root_descriptor)
            raise
        chain.append(_BoundDirectory(".", root_descriptor, root_identity, False, None))
        for index, part in enumerate(directory_parts):
            parent = chain[-1]
            created = False
            try:
                descriptor = os.open(part, _directory_flags(), dir_fd=parent.descriptor)
            except FileNotFoundError as exc:
                if not create:
                    raise SecurePublicationError(
                        "Private publication directory is missing"
                    ) from exc
                _fault(fault_injector, f"before_mkdir:{'/'.join(directory_parts[: index + 1])}")
                try:
                    os.mkdir(part, mode=_DIRECTORY_MODE, dir_fd=parent.descriptor)
                    created = True
                except FileExistsError:
                    created = False
                _fault(fault_injector, f"after_mkdir:{'/'.join(directory_parts[: index + 1])}")
                descriptor = os.open(part, _directory_flags(), dir_fd=parent.descriptor)
            try:
                value = os.fstat(descriptor)
                identity = _validate_directory(value)
                entry_value = os.stat(part, dir_fd=parent.descriptor, follow_symlinks=False)
                if not identity.matches(entry_value) or not _same_inode(value, entry_value):
                    raise SecurePublicationError(
                        "Private publication directory changed while opening"
                    )
            except BaseException:
                os.close(descriptor)
                raise
            relative = "/".join(directory_parts[: index + 1])
            chain.append(_BoundDirectory(relative, descriptor, identity, created, part))
        if create:
            for item in reversed(chain):
                _verify_chain(root, chain)
                _fsync_descriptor(
                    item.descriptor,
                    label=f"directory:{item.relative_path}",
                    fault_injector=fault_injector,
                )
            _verify_chain(root, chain)
        yield chain
    except OSError as exc:
        raise SecurePublicationError("Secure private directory operation failed") from exc
    finally:
        for item in reversed(chain):
            with suppress(OSError):
                os.close(item.descriptor)


def _read_all(descriptor: int, *, maximum_bytes: int) -> bytes:
    result = bytearray()
    while True:
        chunk = os.read(descriptor, _READ_CHUNK)
        if not chunk:
            return bytes(result)
        result.extend(chunk)
        if len(result) > maximum_bytes:
            raise SecurePublicationError("Private publication file exceeds its safety limit")


def _open_and_read_leaf(
    parent: _BoundDirectory,
    leaf_name: str,
    *,
    expected_mode: int | None,
    require_single_link: bool = True,
    maximum_bytes: int = _MAX_PRIVATE_FILE_BYTES,
) -> tuple[int, os.stat_result, bytes]:
    try:
        entry_value = os.stat(leaf_name, dir_fd=parent.descriptor, follow_symlinks=False)
        _validate_regular_file(
            entry_value,
            expected_mode=expected_mode,
            require_single_link=require_single_link,
        )
        descriptor = os.open(leaf_name, _file_flags(), dir_fd=parent.descriptor)
    except OSError as exc:
        raise SecurePublicationError("Could not securely open private publication file") from exc
    try:
        descriptor_value = os.fstat(descriptor)
        _validate_regular_file(
            descriptor_value,
            expected_mode=expected_mode,
            require_single_link=require_single_link,
        )
        if not _same_inode(entry_value, descriptor_value):
            raise SecurePublicationError("Private publication file changed while opening")
        payload = _read_all(descriptor, maximum_bytes=maximum_bytes)
        final_value = os.stat(leaf_name, dir_fd=parent.descriptor, follow_symlinks=False)
        if not _same_inode(descriptor_value, final_value):
            raise SecurePublicationError("Private publication file changed while reading")
        return descriptor, descriptor_value, payload
    except BaseException:
        os.close(descriptor)
        raise


def read_private_file(
    root: Path,
    relative_path: str | PurePosixPath,
    *,
    expected_mode: int = _DEFAULT_FINAL_FILE_MODE,
    maximum_bytes: int = _MAX_PRIVATE_FILE_BYTES,
    missing_ok: bool = False,
) -> PrivateFileSnapshot | None:
    """Read a private file while retaining and validating every directory descriptor."""

    parts = _relative_parts(relative_path)
    try:
        with _open_directory_chain(root, parts[:-1], create=False) as chain:
            parent = chain[-1]
            try:
                descriptor, value, payload = _open_and_read_leaf(
                    parent,
                    parts[-1],
                    expected_mode=expected_mode,
                    maximum_bytes=maximum_bytes,
                )
            except SecurePublicationError as exc:
                cause = exc.__cause__
                if missing_ok and isinstance(cause, FileNotFoundError):
                    return None
                raise
            try:
                _verify_chain(root, chain)
                return PrivateFileSnapshot(
                    PurePosixPath(*parts).as_posix(),
                    payload,
                    PrivateFileIdentity.from_stat_and_payload(value, payload),
                )
            finally:
                os.close(descriptor)
    except SecurePublicationError:
        if missing_ok:
            # Missing parents are an ordinary first-publication state. Unsafe parents still fail.
            descriptors: list[int] = []
            try:
                try:
                    descriptor = os.open(root, _directory_flags())
                    descriptors.append(descriptor)
                    _validate_directory(os.fstat(descriptor))
                    for part in parts[:-1]:
                        try:
                            descriptor = os.open(
                                part,
                                _directory_flags(),
                                dir_fd=descriptors[-1],
                            )
                        except FileNotFoundError:
                            return None
                        descriptors.append(descriptor)
                        _validate_directory(os.fstat(descriptor))
                    try:
                        os.stat(parts[-1], dir_fd=descriptors[-1], follow_symlinks=False)
                    except FileNotFoundError:
                        return None
                except OSError as exc:
                    raise SecurePublicationError("Private publication path is unsafe") from exc
            finally:
                for descriptor in reversed(descriptors):
                    with suppress(OSError):
                        os.close(descriptor)
        raise


def _write_all(
    descriptor: int,
    payload: bytes,
    *,
    fault_injector: FaultInjector | None,
) -> None:
    offset = 0
    while offset < len(payload):
        _fault(fault_injector, f"before_write:{offset}")
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise SecurePublicationError("Private publication write made no progress")
        _fault(fault_injector, f"after_write:{offset}")
        offset += written


def _renameat2(
    source_parent: int,
    source_name: str,
    destination_parent: int,
    destination_name: str,
    *,
    flags: int,
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise SecurePublicationError("Atomic no-replace/exchange rename is unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        source_parent,
        os.fsencode(source_name),
        destination_parent,
        os.fsencode(destination_name),
        flags,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))


def _leaf_stat(parent: _BoundDirectory, leaf_name: str) -> os.stat_result | None:
    try:
        return os.stat(leaf_name, dir_fd=parent.descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _verify_bound_leaf(
    parent: _BoundDirectory,
    leaf_name: str,
    descriptor: int,
    *,
    require_single_link: bool,
) -> os.stat_result:
    entry_value = _leaf_stat(parent, leaf_name)
    if entry_value is None:
        raise SecurePublicationError("Private publication file disappeared")
    descriptor_value = os.fstat(descriptor)
    _validate_regular_file(
        descriptor_value,
        expected_mode=_DEFAULT_FINAL_FILE_MODE,
        require_single_link=require_single_link,
    )
    if not _same_inode(entry_value, descriptor_value):
        raise SecurePublicationError("Private publication file was substituted")
    return descriptor_value


def _unlink_bound_leaf(
    parent: _BoundDirectory,
    leaf_name: str,
    expected: os.stat_result,
    *,
    fault_injector: FaultInjector | None,
) -> None:
    _fault(fault_injector, f"before_unlink:{leaf_name}")
    current = _leaf_stat(parent, leaf_name)
    if current is None or not _same_inode(current, expected) or not stat.S_ISREG(current.st_mode):
        raise SecurePublicationError("Refusing to remove a substituted publication staging file")
    os.unlink(leaf_name, dir_fd=parent.descriptor)
    _fault(fault_injector, f"after_unlink:{leaf_name}")


@dataclass(frozen=True, slots=True)
class _SwapMarker:
    target_name: str
    stage_name: str
    marker_name: str
    desired_sha256: str
    desired_size: int
    old_device: int
    old_inode: int
    old_uid: int
    old_mode: int
    old_size: int
    old_sha256: str

    def payload(self) -> bytes:
        value = {
            "version": 1,
            "target_name": self.target_name,
            "stage_name": self.stage_name,
            "marker_name": self.marker_name,
            "desired_sha256": self.desired_sha256,
            "desired_size": self.desired_size,
            "old": {
                "device": self.old_device,
                "inode": self.old_inode,
                "uid": self.old_uid,
                "mode": self.old_mode,
                "size": self.old_size,
                "sha256": self.old_sha256,
            },
        }
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"

    @classmethod
    def from_payload(
        cls,
        payload: bytes,
        *,
        target_name: str,
        desired_payload: bytes,
    ) -> _SwapMarker:
        marker = cls.from_bound_payload(payload, target_name=target_name)
        desired_sha256 = hashlib.sha256(desired_payload).hexdigest()
        if marker.desired_sha256 != desired_sha256 or marker.desired_size != len(desired_payload):
            raise SecurePublicationError("Private publication marker desired payload is invalid")
        return marker

    @classmethod
    def from_bound_payload(
        cls,
        payload: bytes,
        *,
        target_name: str,
    ) -> _SwapMarker:
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SecurePublicationError("Private publication marker is invalid") from exc
        path_key = hashlib.sha256(target_name.encode()).hexdigest()[:12]
        desired_sha256 = value.get("desired_sha256") if isinstance(value, dict) else None
        desired_size = value.get("desired_size") if isinstance(value, dict) else None
        if (
            not isinstance(desired_sha256, str)
            or len(desired_sha256) != 64
            or any(character not in "0123456789abcdef" for character in desired_sha256)
            or not isinstance(desired_size, int)
            or isinstance(desired_size, bool)
            or desired_size < 0
        ):
            raise SecurePublicationError("Private publication marker desired payload is invalid")
        prefix = f".publication-{path_key}-{desired_sha256[:24]}"
        if (
            not isinstance(value, dict)
            or set(value)
            != {
                "version",
                "target_name",
                "stage_name",
                "marker_name",
                "desired_sha256",
                "desired_size",
                "old",
            }
            or value.get("version") != 1
            or value.get("target_name") != target_name
            or value.get("stage_name") != f"{prefix}.data"
            or value.get("marker_name") != f"{prefix}.marker"
            or not isinstance(value.get("old"), dict)
        ):
            raise SecurePublicationError("Private publication marker metadata is invalid")
        old = value["old"]
        old_keys = {"device", "inode", "uid", "mode", "size", "sha256"}
        if set(old) != old_keys or not isinstance(old.get("sha256"), str):
            raise SecurePublicationError("Private publication marker identity is invalid")
        numeric: dict[str, int] = {}
        for key in old_keys - {"sha256"}:
            item = old.get(key)
            if not isinstance(item, int) or isinstance(item, bool) or item < 0:
                raise SecurePublicationError("Private publication marker identity is invalid")
            numeric[key] = item
        return cls(
            target_name=target_name,
            stage_name=f"{prefix}.data",
            marker_name=f"{prefix}.marker",
            desired_sha256=desired_sha256,
            desired_size=desired_size,
            old_device=numeric["device"],
            old_inode=numeric["inode"],
            old_uid=numeric["uid"],
            old_mode=numeric["mode"],
            old_size=numeric["size"],
            old_sha256=str(old["sha256"]),
        )


def _marker_for_replacement(
    target_name: str,
    desired_payload: bytes,
    old: PrivateFileSnapshot,
) -> _SwapMarker:
    digest = hashlib.sha256(desired_payload).hexdigest()
    path_key = hashlib.sha256(target_name.encode()).hexdigest()[:12]
    prefix = f".publication-{path_key}-{digest[:24]}"
    return _SwapMarker(
        target_name=target_name,
        stage_name=f"{prefix}.data",
        marker_name=f"{prefix}.marker",
        desired_sha256=digest,
        desired_size=len(desired_payload),
        old_device=old.identity.device,
        old_inode=old.identity.inode,
        old_uid=old.identity.uid,
        old_mode=old.identity.mode,
        old_size=old.identity.size_bytes,
        old_sha256=old.identity.sha256,
    )


def _link_anonymous_file(descriptor: int, parent_descriptor: int, name: str) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    linkat = getattr(libc, "linkat", None)
    if linkat is None:
        raise SecurePublicationError("Anonymous private-file publication is unavailable")
    linkat.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    linkat.restype = ctypes.c_int
    result = linkat(descriptor, b"", parent_descriptor, os.fsencode(name), _AT_EMPTY_PATH)
    if result != 0 and ctypes.get_errno() in {errno.ENOENT, errno.EPERM}:
        # Unprivileged Linux may require the documented procfs route instead of AT_EMPTY_PATH.
        result = linkat(
            _AT_FDCWD,
            os.fsencode(f"/proc/self/fd/{descriptor}"),
            parent_descriptor,
            os.fsencode(name),
            _AT_SYMLINK_FOLLOW,
        )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))


def _prepared_anonymous_file(
    parent: _BoundDirectory,
    payload: bytes,
    *,
    label: str,
    fault_injector: FaultInjector | None,
) -> tuple[int, os.stat_result]:
    if not hasattr(os, "O_TMPFILE"):
        raise SecurePublicationError("Anonymous private-file publication is unavailable")
    try:
        descriptor = os.open(
            ".",
            os.O_RDWR | os.O_TMPFILE | int(getattr(os, "O_CLOEXEC", 0)),
            _MUTABLE_FILE_MODE,
            dir_fd=parent.descriptor,
        )
    except OSError as exc:
        raise SecurePublicationError("Could not create anonymous private publication file") from exc
    try:
        _write_all(descriptor, payload, fault_injector=fault_injector)
        _fault(fault_injector, f"before_chmod:{label}")
        os.fchmod(descriptor, _DEFAULT_FINAL_FILE_MODE)
        _fault(fault_injector, f"after_chmod:{label}")
        _fsync_descriptor(
            descriptor,
            label=f"anonymous-file:{label}",
            fault_injector=fault_injector,
        )
        value = os.fstat(descriptor)
        if (
            not stat.S_ISREG(value.st_mode)
            or value.st_uid != os.geteuid()
            or value.st_nlink != 0
            or stat.S_IMODE(value.st_mode) != _DEFAULT_FINAL_FILE_MODE
            or value.st_size != len(payload)
        ):
            raise SecurePublicationError("Anonymous private publication file is unsafe")
        return descriptor, value
    except BaseException:
        os.close(descriptor)
        raise


def _link_prepared_file(
    descriptor: int,
    expected: os.stat_result,
    parent: _BoundDirectory,
    name: str,
    *,
    event_label: str,
    fault_injector: FaultInjector | None,
) -> os.stat_result:
    _fault(fault_injector, f"before_link:{event_label}")
    if _leaf_stat(parent, name) is not None:
        raise SecurePublicationError("Private publication link target already exists")
    _link_anonymous_file(descriptor, parent.descriptor, name)
    _fault(fault_injector, f"after_link:{event_label}")
    value = _leaf_stat(parent, name)
    if value is None or not _same_inode(value, expected) or value.st_nlink != 1:
        raise SecurePublicationError("Private publication link was substituted")
    return value


def _read_optional_leaf(parent: _BoundDirectory, name: str) -> PrivateFileSnapshot | None:
    if _leaf_stat(parent, name) is None:
        return None
    descriptor, value, payload = _open_and_read_leaf(
        parent,
        name,
        expected_mode=_DEFAULT_FINAL_FILE_MODE,
    )
    try:
        return PrivateFileSnapshot(
            name,
            payload,
            PrivateFileIdentity.from_stat_and_payload(value, payload),
        )
    finally:
        os.close(descriptor)


def _snapshot_is_desired(snapshot: PrivateFileSnapshot | None, payload: bytes) -> bool:
    return (
        snapshot is not None
        and snapshot.payload == payload
        and snapshot.identity.link_count == 1
        and snapshot.identity.mode == _DEFAULT_FINAL_FILE_MODE
    )


def _snapshot_is_old(snapshot: PrivateFileSnapshot | None, marker: _SwapMarker) -> bool:
    return (
        snapshot is not None
        and snapshot.identity.device == marker.old_device
        and snapshot.identity.inode == marker.old_inode
        and snapshot.identity.uid == marker.old_uid
        and snapshot.identity.mode == marker.old_mode
        and snapshot.identity.link_count == 1
        and snapshot.identity.size_bytes == marker.old_size
        and snapshot.identity.sha256 == marker.old_sha256
    )


def _snapshot_matches_marker_desired(
    snapshot: PrivateFileSnapshot | None,
    marker: _SwapMarker,
) -> bool:
    return (
        snapshot is not None
        and snapshot.identity.uid == os.geteuid()
        and snapshot.identity.mode == _DEFAULT_FINAL_FILE_MODE
        and snapshot.identity.link_count == 1
        and snapshot.identity.size_bytes == marker.desired_size
        and snapshot.identity.sha256 == marker.desired_sha256
    )


def _load_swap_marker(
    parent: _BoundDirectory,
    target_name: str,
    desired_payload: bytes,
) -> tuple[_SwapMarker, PrivateFileSnapshot] | None:
    digest = hashlib.sha256(desired_payload).hexdigest()
    path_key = hashlib.sha256(target_name.encode()).hexdigest()[:12]
    marker_name = f".publication-{path_key}-{digest[:24]}.marker"
    snapshot = _read_optional_leaf(parent, marker_name)
    if snapshot is None:
        return None
    return (
        _SwapMarker.from_payload(
            snapshot.payload,
            target_name=target_name,
            desired_payload=desired_payload,
        ),
        snapshot,
    )


def _reject_unbound_publication_artifacts(
    parent: _BoundDirectory,
    target_name: str,
    desired_payload: bytes,
) -> None:
    path_key = hashlib.sha256(target_name.encode()).hexdigest()[:12]
    digest = hashlib.sha256(desired_payload).hexdigest()
    prefix = f".publication-{path_key}-"
    allowed = {
        f".publication-{path_key}-{digest[:24]}.marker",
        f".publication-{path_key}-{digest[:24]}.data",
    }
    with os.scandir(parent.descriptor) as entries:
        reserved = {entry.name for entry in entries if entry.name.startswith(prefix)}
    if reserved - allowed:
        raise SecurePublicationError("Unbound private publication recovery artifact exists")


def _recover_bound_swap(
    root: Path,
    chain: Sequence[_BoundDirectory],
    target_name: str,
    *,
    fault_injector: FaultInjector | None,
) -> bool:
    """Rollback an interrupted swap when possible, or finish its irreversible cleanup."""

    parent = chain[-1]
    path_key = hashlib.sha256(target_name.encode()).hexdigest()[:12]
    prefix = f".publication-{path_key}-"
    with os.scandir(parent.descriptor) as entries:
        reserved = sorted(entry.name for entry in entries if entry.name.startswith(prefix))
    marker_names = [name for name in reserved if name.endswith(".marker")]
    if not marker_names:
        if reserved:
            raise SecurePublicationError("Unbound private publication recovery artifact exists")
        return False
    if len(marker_names) != 1:
        raise SecurePublicationError("Ambiguous private publication recovery markers exist")
    marker_snapshot = _read_optional_leaf(parent, marker_names[0])
    if marker_snapshot is None:
        raise SecurePublicationError("Private publication recovery marker disappeared")
    marker = _SwapMarker.from_bound_payload(marker_snapshot.payload, target_name=target_name)
    allowed = {marker.marker_name, marker.stage_name}
    if set(reserved) - allowed:
        raise SecurePublicationError("Unbound private publication recovery artifact exists")
    target = _read_optional_leaf(parent, target_name)
    stage = _read_optional_leaf(parent, marker.stage_name)
    if _snapshot_is_old(target, marker):
        if stage is not None:
            if not _snapshot_matches_marker_desired(stage, marker):
                raise SecurePublicationError("Private publication recovery stage is invalid")
            _remove_bound_snapshot(parent, stage, fault_injector=fault_injector)
            _fsync_descriptor(
                parent.descriptor,
                label=f"directory:{parent.relative_path}:rollback-stage",
                fault_injector=fault_injector,
            )
    elif _snapshot_matches_marker_desired(target, marker):
        if stage is not None:
            if not _snapshot_is_old(stage, marker):
                raise SecurePublicationError("Private publication recovery stage is invalid")
            _fault(fault_injector, f"before_rename_exchange_rollback:{target_name}")
            _verify_chain(root, chain)
            _renameat2(
                parent.descriptor,
                marker.stage_name,
                parent.descriptor,
                target_name,
                flags=_RENAME_EXCHANGE,
            )
            _fault(fault_injector, f"after_rename_exchange_rollback:{target_name}")
            _fsync_descriptor(
                parent.descriptor,
                label=f"directory:{parent.relative_path}:rollback-exchange",
                fault_injector=fault_injector,
            )
            stage = _read_optional_leaf(parent, marker.stage_name)
            target = _read_optional_leaf(parent, target_name)
            if not _snapshot_is_old(target, marker) or not _snapshot_matches_marker_desired(
                stage, marker
            ):
                raise SecurePublicationError("Private publication rollback exchange is invalid")
            if stage is None:
                raise SecurePublicationError("Private publication rollback stage disappeared")
            _remove_bound_snapshot(parent, stage, fault_injector=fault_injector)
            _fsync_descriptor(
                parent.descriptor,
                label=f"directory:{parent.relative_path}:rollback-stage",
                fault_injector=fault_injector,
            )
        else:
            # The old inode was already durably discarded. Retain the exact desired target and
            # make its directory entry durable before removing the last recovery marker.
            _fsync_descriptor(
                parent.descriptor,
                label=f"directory:{parent.relative_path}:complete-target",
                fault_injector=fault_injector,
            )
    else:
        raise SecurePublicationError("Private publication recovery target is invalid")
    _remove_bound_snapshot(parent, marker_snapshot, fault_injector=fault_injector)
    _fsync_descriptor(
        parent.descriptor,
        label=f"directory:{parent.relative_path}:rollback-marker",
        fault_injector=fault_injector,
    )
    _verify_chain(root, chain)
    return True


def _remove_bound_snapshot(
    parent: _BoundDirectory,
    snapshot: PrivateFileSnapshot,
    *,
    fault_injector: FaultInjector | None,
) -> None:
    current = _read_optional_leaf(parent, snapshot.relative_path)
    if (
        current is None
        or current.identity != snapshot.identity
        or current.payload != snapshot.payload
    ):
        raise SecurePublicationError("Private publication recovery file was substituted")
    value = _leaf_stat(parent, snapshot.relative_path)
    if value is None:
        raise SecurePublicationError("Private publication recovery file disappeared")
    _unlink_bound_leaf(
        parent,
        snapshot.relative_path,
        value,
        fault_injector=fault_injector,
    )


def _finish_swap_recovery(
    root: Path,
    chain: Sequence[_BoundDirectory],
    target_name: str,
    desired_payload: bytes,
    marker: _SwapMarker,
    marker_snapshot: PrivateFileSnapshot,
    *,
    fault_injector: FaultInjector | None,
) -> bool:
    parent = chain[-1]
    target = _read_optional_leaf(parent, target_name)
    stage = _read_optional_leaf(parent, marker.stage_name)
    if _snapshot_is_desired(target, desired_payload):
        _fsync_descriptor(
            parent.descriptor,
            label=f"directory:{parent.relative_path}:replay-target",
            fault_injector=fault_injector,
        )
        if stage is not None:
            if not _snapshot_is_old(stage, marker):
                raise SecurePublicationError("Private publication recovery stage is invalid")
            _remove_bound_snapshot(parent, stage, fault_injector=fault_injector)
            _fsync_descriptor(
                parent.descriptor,
                label=f"directory:{parent.relative_path}:replay-stage-cleanup",
                fault_injector=fault_injector,
            )
        _remove_bound_snapshot(parent, marker_snapshot, fault_injector=fault_injector)
        _fsync_descriptor(
            parent.descriptor,
            label=f"directory:{parent.relative_path}:replay-marker-cleanup",
            fault_injector=fault_injector,
        )
        _verify_chain(root, chain)
        return True
    if not _snapshot_is_old(target, marker):
        raise SecurePublicationError("Private publication recovery target is invalid")
    if stage is None:
        return False
    if not _snapshot_is_desired(stage, desired_payload):
        raise SecurePublicationError("Private publication recovery stage is invalid")
    _fault(fault_injector, f"before_rename_exchange:{target_name}")
    _verify_chain(root, chain)
    _renameat2(
        parent.descriptor,
        marker.stage_name,
        parent.descriptor,
        target_name,
        flags=_RENAME_EXCHANGE,
    )
    _fault(fault_injector, f"after_rename_exchange:{target_name}")
    return _finish_swap_recovery(
        root,
        chain,
        target_name,
        desired_payload,
        marker,
        marker_snapshot,
        fault_injector=fault_injector,
    )


def _published_snapshot(
    root: Path,
    chain: Sequence[_BoundDirectory],
    leaf_name: str,
    relative_path: str,
    payload: bytes,
    *,
    fault_injector: FaultInjector | None,
) -> PublicationReceipt:
    parent = chain[-1]
    descriptor, _value, observed = _open_and_read_leaf(
        parent,
        leaf_name,
        expected_mode=_DEFAULT_FINAL_FILE_MODE,
    )
    try:
        if observed != payload:
            raise SecurePublicationError("Published private file bytes do not match")
        _fsync_descriptor(
            descriptor,
            label=f"file:{relative_path}",
            fault_injector=fault_injector,
        )
        final_value = _verify_bound_leaf(
            parent,
            leaf_name,
            descriptor,
            require_single_link=True,
        )
        _fsync_descriptor(
            parent.descriptor,
            label=f"directory:{parent.relative_path}:verified-target",
            fault_injector=fault_injector,
        )
        _verify_chain(root, chain)
        return PublicationReceipt(
            relative_path,
            PrivateFileIdentity.from_stat_and_payload(final_value, observed),
        )
    finally:
        os.close(descriptor)


def recover_private_publication(
    root: Path,
    relative_path: str | PurePosixPath,
    *,
    fault_injector: FaultInjector | None = None,
) -> bool:
    """Reconcile a named replacement marker before a caller binds the current target."""

    parts = _relative_parts(relative_path)
    try:
        with _open_directory_chain(root, parts[:-1], create=False) as chain:
            return _recover_bound_swap(
                root,
                chain,
                parts[-1],
                fault_injector=fault_injector,
            )
    except SecurePublicationError as exc:
        if isinstance(exc.__cause__, FileNotFoundError):
            return False
        raise


def publish_private_file(
    root: Path,
    relative_path: str | PurePosixPath,
    payload: bytes,
    *,
    expected_existing: PrivateFileSnapshot | None,
    fault_injector: FaultInjector | None = None,
) -> PublicationReceipt:
    """Atomically publish exact private bytes through retained no-follow descriptors.

    ``expected_existing`` is mandatory for replacement and binds the exchange to the inode and
    bytes that the caller validated. Passing ``None`` is a no-clobber first publication. A replay
    whose target already contains ``payload`` succeeds after exact validation.
    """

    parts = _relative_parts(relative_path)
    relative = PurePosixPath(*parts).as_posix()
    with _open_directory_chain(
        root,
        parts[:-1],
        create=True,
        fault_injector=fault_injector,
    ) as chain:
        parent = chain[-1]
        leaf_name = parts[-1]
        _reject_unbound_publication_artifacts(parent, leaf_name, payload)
        matching_marker = _load_swap_marker(parent, leaf_name, payload)
        if matching_marker is not None:
            marker, marker_snapshot = matching_marker
            if _finish_swap_recovery(
                root,
                chain,
                leaf_name,
                payload,
                marker,
                marker_snapshot,
                fault_injector=fault_injector,
            ):
                return _published_snapshot(
                    root,
                    chain,
                    leaf_name,
                    relative,
                    payload,
                    fault_injector=fault_injector,
                )

        # A matching marker can predate its stage link. In that reversible state, discard the
        # marker before starting a fresh attempt. A marker for another desired payload is rejected
        # above: callers must recover it before reading and binding their expected snapshot.
        _recover_bound_swap(
            root,
            chain,
            leaf_name,
            fault_injector=fault_injector,
        )
        _reject_unbound_publication_artifacts(parent, leaf_name, payload)

        current = _read_optional_leaf(parent, leaf_name)
        if current is not None and current.payload == payload:
            return _published_snapshot(
                root,
                chain,
                leaf_name,
                relative,
                payload,
                fault_injector=fault_injector,
            )
        if current is not None:
            if expected_existing is None or expected_existing.relative_path != relative:
                raise SecurePublicationError("Refusing to replace an unbound private file")
            if (
                current.payload != expected_existing.payload
                or current.identity != expected_existing.identity
            ):
                raise SecurePublicationError("Private publication target changed before replace")
        elif expected_existing is not None:
            raise SecurePublicationError("Expected private publication target is missing")

        anonymous_descriptor, anonymous_value = _prepared_anonymous_file(
            parent,
            payload,
            label=leaf_name,
            fault_injector=fault_injector,
        )
        try:
            _verify_chain(root, chain)
            if current is None:
                _link_prepared_file(
                    anonymous_descriptor,
                    anonymous_value,
                    parent,
                    leaf_name,
                    event_label=leaf_name,
                    fault_injector=fault_injector,
                )
                _fsync_descriptor(
                    parent.descriptor,
                    label=f"directory:{parent.relative_path}:link",
                    fault_injector=fault_injector,
                )
            else:
                if expected_existing is None:
                    raise SecurePublicationError("Private publication replacement is unbound")
                marker = _marker_for_replacement(leaf_name, payload, expected_existing)
                marker_descriptor, marker_value = _prepared_anonymous_file(
                    parent,
                    marker.payload(),
                    label=marker.marker_name,
                    fault_injector=fault_injector,
                )
                try:
                    _link_prepared_file(
                        marker_descriptor,
                        marker_value,
                        parent,
                        marker.marker_name,
                        event_label=marker.marker_name,
                        fault_injector=fault_injector,
                    )
                finally:
                    os.close(marker_descriptor)
                _fsync_descriptor(
                    parent.descriptor,
                    label=f"directory:{parent.relative_path}:marker",
                    fault_injector=fault_injector,
                )
                current_marker_snapshot = _read_optional_leaf(parent, marker.marker_name)
                if current_marker_snapshot is None:
                    raise SecurePublicationError("Private publication marker disappeared")
                _link_prepared_file(
                    anonymous_descriptor,
                    anonymous_value,
                    parent,
                    marker.stage_name,
                    event_label=marker.stage_name,
                    fault_injector=fault_injector,
                )
                _fsync_descriptor(
                    parent.descriptor,
                    label=f"directory:{parent.relative_path}:stage",
                    fault_injector=fault_injector,
                )
                if not _finish_swap_recovery(
                    root,
                    chain,
                    leaf_name,
                    payload,
                    marker,
                    current_marker_snapshot,
                    fault_injector=fault_injector,
                ):
                    raise SecurePublicationError("Private publication exchange did not complete")
            _verify_chain(root, chain)
            return _published_snapshot(
                root,
                chain,
                leaf_name,
                relative,
                payload,
                fault_injector=fault_injector,
            )
        except OSError as exc:
            raise SecurePublicationError("Atomic private publication failed") from exc
        finally:
            os.close(anonymous_descriptor)


def prepare_staging_directories(
    root: Path,
    staging_relative_path: str | PurePosixPath,
    destination_relative_paths: Sequence[str | PurePosixPath],
    *,
    fault_injector: FaultInjector | None = None,
) -> StagingPreparation:
    """Create and durably publish every nested staging parent before source renames."""

    staging_parts = _relative_parts(staging_relative_path)
    destinations = [_relative_parts(path) for path in destination_relative_paths]
    parent_parts: set[tuple[str, ...]] = {staging_parts}
    for destination in destinations:
        if destination[: len(staging_parts)] != staging_parts or len(destination) <= len(
            staging_parts
        ):
            raise SecurePublicationError("Staging destination escapes its operation directory")
        for end in range(len(staging_parts), len(destination)):
            parent_parts.add(destination[:end])
    for parts in sorted(parent_parts, key=lambda value: (len(value), value)):
        with _open_directory_chain(
            root,
            parts,
            create=True,
            fault_injector=fault_injector,
        ) as chain:
            _verify_chain(root, chain)
    identities: list[tuple[str, DirectoryIdentity]] = []
    for parts in sorted(parent_parts, key=lambda value: (len(value), value)):
        with _open_directory_chain(root, parts, create=False) as chain:
            identities.append((PurePosixPath(*parts).as_posix(), chain[-1].identity))
    return StagingPreparation(PurePosixPath(*staging_parts).as_posix(), tuple(identities))


def _staging_entry_relative(parent: str, name: str) -> str:
    return name if parent == "." else f"{parent}/{name}"


def _seal_directory(
    descriptor: int,
    relative_path: str,
    *,
    staging_relative_path: str,
    prepared_directories: dict[str, DirectoryIdentity],
    fault_injector: FaultInjector | None,
) -> list[StagingTreeEntry]:
    directory_value = os.fstat(descriptor)
    _validate_directory(directory_value)
    full_relative = (
        staging_relative_path
        if relative_path == "."
        else f"{staging_relative_path}/{relative_path}"
    )
    expected_directory = prepared_directories.get(full_relative)
    if expected_directory is not None and not expected_directory.matches(directory_value):
        raise SecurePublicationError("Prepared staging directory was substituted")
    result = [StagingTreeEntry.from_stat(relative_path, directory_value, kind="directory")]
    with os.scandir(descriptor) as entries:
        names = sorted(entry.name for entry in entries)
    for name in names:
        entry_value = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        child_relative = _staging_entry_relative(relative_path, name)
        if stat.S_ISDIR(entry_value.st_mode) and not stat.S_ISLNK(entry_value.st_mode):
            child = os.open(name, _directory_flags(), dir_fd=descriptor)
            try:
                child_value = os.fstat(child)
                _validate_directory(child_value)
                if not _same_inode(entry_value, child_value):
                    raise SecurePublicationError("Staging directory changed while sealing")
                result.extend(
                    _seal_directory(
                        child,
                        child_relative,
                        staging_relative_path=staging_relative_path,
                        prepared_directories=prepared_directories,
                        fault_injector=fault_injector,
                    )
                )
                final_value = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                if not _same_inode(child_value, final_value):
                    raise SecurePublicationError("Staging directory was substituted")
            finally:
                os.close(child)
            continue
        _validate_regular_file(entry_value, require_single_link=False)
        child = os.open(name, _file_flags(), dir_fd=descriptor)
        try:
            child_value = os.fstat(child)
            _validate_regular_file(child_value, require_single_link=False)
            if not _same_inode(entry_value, child_value):
                raise SecurePublicationError("Staging file changed while sealing")
            _fsync_descriptor(
                child,
                label=f"staging-file:{child_relative}",
                fault_injector=fault_injector,
            )
            final_value = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if not _same_inode(child_value, final_value):
                raise SecurePublicationError("Staging file was substituted")
            result.append(StagingTreeEntry.from_stat(child_relative, child_value, kind="file"))
        finally:
            os.close(child)
    _fsync_descriptor(
        descriptor,
        label=f"staging-directory:{relative_path}",
        fault_injector=fault_injector,
    )
    return result


def _validate_staging_link_topology(entries: Sequence[StagingTreeEntry]) -> None:
    file_entries = [entry for entry in entries if entry.kind == "file"]
    internal_links = Counter((entry.device, entry.inode) for entry in file_entries)
    for entry in file_entries:
        if entry.link_count != internal_links[(entry.device, entry.inode)]:
            raise SecurePublicationError("Staging file has a link outside the sealed tree")


def seal_staging_tree(
    root: Path,
    preparation: StagingPreparation,
    *,
    fault_injector: FaultInjector | None = None,
) -> SealedStagingTree:
    """Fsync a staged tree bottom-up and bind its complete inode inventory for cleanup."""

    staging_parts = _relative_parts(preparation.staging_relative_path)
    with _open_directory_chain(root, staging_parts, create=False) as chain:
        prepared = dict(preparation.directories)
        for item in chain:
            if item.relative_path == ".":
                continue
            expected = prepared.get(item.relative_path)
            if expected is not None and expected != item.identity:
                raise SecurePublicationError("Prepared staging directory was substituted")
        entries = _seal_directory(
            chain[-1].descriptor,
            ".",
            staging_relative_path=preparation.staging_relative_path,
            prepared_directories=prepared,
            fault_injector=fault_injector,
        )
        observed_directories = {
            (
                preparation.staging_relative_path
                if entry.relative_path == "."
                else f"{preparation.staging_relative_path}/{entry.relative_path}"
            )
            for entry in entries
            if entry.kind == "directory"
        }
        if not set(prepared).issubset(observed_directories):
            raise SecurePublicationError("Prepared staging directory is missing")
        _validate_staging_link_topology(entries)
        for item in reversed(chain[:-1]):
            _verify_chain(root, chain)
            _fsync_descriptor(
                item.descriptor,
                label=f"staging-ancestor:{item.relative_path}",
                fault_injector=fault_injector,
            )
        _verify_chain(root, chain)
    return SealedStagingTree(preparation.staging_relative_path, tuple(entries))


def _direct_children(
    entries: dict[str, StagingTreeEntry], relative_path: str
) -> dict[str, StagingTreeEntry]:
    prefix = "" if relative_path == "." else f"{relative_path}/"
    result: dict[str, StagingTreeEntry] = {}
    for path, entry in entries.items():
        if path == relative_path or not path.startswith(prefix):
            continue
        remainder = path[len(prefix) :]
        if "/" not in remainder:
            result[remainder] = entry
    return result


def _remove_sealed_directory(
    descriptor: int,
    relative_path: str,
    entries: dict[str, StagingTreeEntry],
    *,
    remaining_links: Counter[tuple[int, int]],
    fault_injector: FaultInjector | None,
) -> None:
    expected_children = _direct_children(entries, relative_path)
    with os.scandir(descriptor) as current_entries:
        current_names = {entry.name for entry in current_entries}
    unexpected = current_names - set(expected_children)
    if unexpected:
        raise SecurePublicationError("Sealed staging tree contains an unexpected entry")
    for name in sorted(current_names):
        expected = expected_children[name]
        current = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if not expected.matches(current, cleanup=True):
            raise SecurePublicationError("Sealed staging entry was substituted")
        if expected.kind == "file":
            _fault(fault_injector, f"before_unlink_staging:{expected.relative_path}")
            current = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            identity = (expected.device, expected.inode)
            if (
                not expected.matches(current, cleanup=True)
                or int(current.st_nlink) != remaining_links[identity]
            ):
                raise SecurePublicationError("Refusing to unlink substituted staged data")
            os.unlink(name, dir_fd=descriptor)
            remaining_links[identity] -= 1
            _fault(fault_injector, f"after_unlink_staging:{expected.relative_path}")
            continue
        child = os.open(name, _directory_flags(), dir_fd=descriptor)
        try:
            if not expected.matches(os.fstat(child), cleanup=True):
                raise SecurePublicationError("Sealed staging directory was substituted")
            _remove_sealed_directory(
                child,
                expected.relative_path,
                entries,
                remaining_links=remaining_links,
                fault_injector=fault_injector,
            )
            _fsync_descriptor(
                child,
                label=f"cleanup-directory:{expected.relative_path}",
                fault_injector=fault_injector,
            )
        finally:
            os.close(child)
        _fault(fault_injector, f"before_rmdir_staging:{expected.relative_path}")
        current = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if not expected.matches(current, cleanup=True):
            raise SecurePublicationError("Refusing to remove substituted staging directory")
        os.rmdir(name, dir_fd=descriptor)
        _fault(fault_injector, f"after_rmdir_staging:{expected.relative_path}")
    _fsync_descriptor(
        descriptor,
        label=f"cleanup-directory:{relative_path}",
        fault_injector=fault_injector,
    )


def _fsync_absent_path_ancestry(
    root: Path,
    parts: Sequence[str],
    *,
    fault_injector: FaultInjector | None,
) -> bool:
    """Prove an absent entry durable by fsyncing its nearest surviving ancestry."""

    chain: list[_BoundDirectory] = []
    try:
        root_descriptor = os.open(root, _directory_flags())
        try:
            root_identity = _validate_directory(os.fstat(root_descriptor))
        except BaseException:
            os.close(root_descriptor)
            raise
        chain.append(
            _BoundDirectory(
                ".",
                root_descriptor,
                root_identity,
                False,
                None,
            )
        )
        for index, part in enumerate(parts):
            parent = chain[-1]
            try:
                entry_value = os.stat(part, dir_fd=parent.descriptor, follow_symlinks=False)
            except FileNotFoundError:
                for item in reversed(chain):
                    _fsync_descriptor(
                        item.descriptor,
                        label=f"cleanup-absence:{item.relative_path}",
                        fault_injector=fault_injector,
                    )
                _verify_chain(root, chain)
                return True
            if index == len(parts) - 1:
                return False
            identity = _validate_directory(entry_value)
            descriptor = os.open(part, _directory_flags(), dir_fd=parent.descriptor)
            try:
                descriptor_value = os.fstat(descriptor)
                if not identity.matches(descriptor_value) or not _same_inode(
                    entry_value, descriptor_value
                ):
                    raise SecurePublicationError("Cleanup ancestry changed while opening")
            except BaseException:
                os.close(descriptor)
                raise
            chain.append(
                _BoundDirectory(
                    "/".join(parts[: index + 1]),
                    descriptor,
                    identity,
                    False,
                    part,
                )
            )
        return False
    except OSError as exc:
        raise SecurePublicationError("Could not prove absent cleanup path durable") from exc
    finally:
        for item in reversed(chain):
            with suppress(OSError):
                os.close(item.descriptor)


def _cleanup_entry_matches(
    expected: StagingTreeEntry,
    current: StagingTreeEntry,
) -> bool:
    return (
        expected.relative_path == current.relative_path
        and expected.kind == current.kind
        and expected.device == current.device
        and expected.inode == current.inode
        and expected.uid == current.uid
        and expected.mode == current.mode
        and (
            expected.kind == "directory"
            or (
                expected.size_bytes == current.size_bytes
                and expected.modified_ns == current.modified_ns
            )
        )
    )


def remove_sealed_staging_tree(
    root: Path,
    receipt: SealedStagingTree,
    *,
    fault_injector: FaultInjector | None = None,
) -> bool:
    """Idempotently remove only inodes recorded by a sealed staging receipt.

    Missing recorded entries are treated as completed cleanup. Unknown or substituted entries
    fail closed before that entry is removed.
    """

    parts = _relative_parts(receipt.staging_relative_path)
    expected_entries = {entry.relative_path: entry for entry in receipt.entries}
    root_entry = expected_entries.get(".")
    if root_entry is None or root_entry.kind != "directory":
        raise SecurePublicationError("Sealed staging receipt lacks its root")
    try:
        with _open_directory_chain(root, parts[:-1], create=False) as parent_chain:
            parent = parent_chain[-1]
            current = _leaf_stat(parent, parts[-1])
            if current is None:
                for item in reversed(parent_chain):
                    _fsync_descriptor(
                        item.descriptor,
                        label=f"cleanup-absence:{item.relative_path}",
                        fault_injector=fault_injector,
                    )
                _verify_chain(root, parent_chain)
                return False
            if not root_entry.matches(current, cleanup=True):
                raise SecurePublicationError("Sealed staging root was substituted")
            staging_descriptor = os.open(parts[-1], _directory_flags(), dir_fd=parent.descriptor)
            try:
                if not root_entry.matches(os.fstat(staging_descriptor), cleanup=True):
                    raise SecurePublicationError("Sealed staging root changed while opening")
                current_entries = _seal_directory(
                    staging_descriptor,
                    ".",
                    staging_relative_path=receipt.staging_relative_path,
                    prepared_directories={},
                    fault_injector=None,
                )
                _validate_staging_link_topology(current_entries)
                for current_entry in current_entries:
                    expected_entry = expected_entries.get(current_entry.relative_path)
                    if expected_entry is None or not _cleanup_entry_matches(
                        expected_entry, current_entry
                    ):
                        raise SecurePublicationError("Sealed staging entry was substituted")
                remaining_links = Counter(
                    (entry.device, entry.inode) for entry in current_entries if entry.kind == "file"
                )
                _remove_sealed_directory(
                    staging_descriptor,
                    ".",
                    expected_entries,
                    remaining_links=remaining_links,
                    fault_injector=fault_injector,
                )
            finally:
                os.close(staging_descriptor)
            _fault(fault_injector, "before_rmdir_staging:.")
            current = os.stat(parts[-1], dir_fd=parent.descriptor, follow_symlinks=False)
            if not root_entry.matches(current, cleanup=True):
                raise SecurePublicationError("Refusing to remove substituted staging root")
            os.rmdir(parts[-1], dir_fd=parent.descriptor)
            _fault(fault_injector, "after_rmdir_staging:.")
            for item in reversed(parent_chain):
                _fsync_descriptor(
                    item.descriptor,
                    label=f"cleanup-ancestor:{item.relative_path}",
                    fault_injector=fault_injector,
                )
            _verify_chain(root, parent_chain)
            return True
    except SecurePublicationError as exc:
        if isinstance(exc.__cause__, FileNotFoundError) and _fsync_absent_path_ancestry(
            root,
            parts,
            fault_injector=fault_injector,
        ):
            return False
        raise
