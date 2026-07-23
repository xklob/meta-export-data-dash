#!/usr/bin/env python3
"""Run the production archive spot check as one privately supervised process tree.

This runner is intended to be the single workload beneath ``supervise_acceptance.py``'s monitored
leader shim. It starts the read-only API server and the acceptance client without creating another
session or process group, so the outer supervisor observes and can terminate the complete tree.
Child output is discarded and the one-time access token is exchanged only through an owner-only
file.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import signal
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Final, Never, Protocol, cast

OWNER_DIRECTORY_MODE: Final = 0o700
OWNER_FILE_MODE: Final = 0o600
DEFAULT_PORT: Final = 8765
DEFAULT_READY_TIMEOUT_SECONDS: Final = 300.0
DEFAULT_POLL_SECONDS: Final = 0.25
DEFAULT_SHUTDOWN_GRACE_SECONDS: Final = 2.0
GROUP_CLEANUP_TIMEOUT_SECONDS: Final = 1.0
GROUP_CLEANUP_POLL_SECONDS: Final = 0.05
MAX_CONTROL_LOOP_SLEEP_SECONDS: Final = 0.25
MAX_HEALTH_PROBE_TIMEOUT_SECONDS: Final = 1.0
DEFAULT_RECONCILIATION_SEED: Final = 20260722
MAX_READY_TIMEOUT_SECONDS: Final = 3600.0
MAX_POLL_SECONDS: Final = 10.0
MAX_SHUTDOWN_GRACE_SECONDS: Final = 2.0
MAX_HEALTH_RESPONSE_BYTES: Final = 4096
MAX_TOKEN_BYTES: Final = 512
MAX_ACCEPTANCE_REPORT_BYTES: Final = 64 * 1024 * 1024
EXPECTED_ACCEPTANCE_REPORT_SCHEMA_VERSION: Final = 2
EXPECTED_PROFILE_IDENTIFIER: Final = "production-spot-check-v2"
LEADER_SHIM_PID_ENVIRONMENT_VARIABLE: Final = "META_ARCHIVE_ACCEPTANCE_LEADER_SHIM_PID"
MAX_LEADER_SHIM_ANCESTOR_DEPTH: Final = 16

EXIT_DESTINATION_REFUSED: Final = 70
EXIT_SERVER_START_FAILED: Final = 71
EXIT_READINESS_TIMEOUT: Final = 72
EXIT_SERVER_EXITED: Final = 73
EXIT_CLIENT_START_FAILED: Final = 74
EXIT_EVIDENCE_INVALID: Final = 75
EXIT_CLEANUP_FAILED: Final = 76
EXIT_INTERNAL_ERROR: Final = 77

_requested_signal: int | None = None


class ChildProcess(Protocol):
    """The subprocess surface used by the orchestration loop."""

    pid: int

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...


@dataclass(frozen=True)
class StackConfig:
    archive_root: Path
    browser_report: Path
    work_directory: Path
    port: int = DEFAULT_PORT
    readiness_timeout_seconds: float = DEFAULT_READY_TIMEOUT_SECONDS
    poll_seconds: float = DEFAULT_POLL_SECONDS
    shutdown_grace_seconds: float = DEFAULT_SHUTDOWN_GRACE_SECONDS
    reconciliation_seed: int = DEFAULT_RECONCILIATION_SEED


@dataclass(frozen=True)
class StackResult:
    status: str
    exit_code: int


@dataclass(frozen=True)
class PrivateFileIdentity:
    device: int
    inode: int
    size_bytes: int
    ctime_ns: int


@dataclass(frozen=True)
class AcceptanceEvidence:
    sha256: str | None
    well_formed: bool
    profile_matches: bool
    gate_passed: bool


SpawnProcess = Callable[[Sequence[str], Path, Mapping[str, str]], ChildProcess]
HealthProbe = Callable[[str, float], bool]
TokenProbe = Callable[[Path], PrivateFileIdentity | None]
EvidenceProbe = Callable[[Path], AcceptanceEvidence]
TerminateProcess = Callable[[ChildProcess, float], bool]
GroupCleanupProbe = Callable[[], bool]


@dataclass(frozen=True)
class RunnerDependencies:
    spawn: SpawnProcess
    health_probe: HealthProbe
    token_probe: TokenProbe
    evidence_probe: EvidenceProbe
    terminate: TerminateProcess
    group_cleanup_probe: GroupCleanupProbe
    monotonic: Callable[[], float]
    sleep: Callable[[float], None]


class DestinationRefusedError(RuntimeError):
    """A destination or required private input failed a fail-closed validation."""


def _signal_handler(signum: int, _frame: FrameType | None) -> None:
    global _requested_signal
    _requested_signal = signum


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _open_directory_without_symlinks(path: Path) -> int:
    """Open an absolute directory while refusing every symlink in its ancestry."""

    absolute = _absolute_path(path)
    if not absolute.is_absolute():
        raise DestinationRefusedError
    flags = os.O_RDONLY | os.O_DIRECTORY | int(getattr(os, "O_CLOEXEC", 0))
    no_follow = int(getattr(os, "O_NOFOLLOW", 0))
    descriptor = os.open("/", flags)
    try:
        for component in absolute.parts[1:]:
            next_descriptor = os.open(component, flags | no_follow, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _validate_owner_only_directory(descriptor: int) -> None:
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        raise DestinationRefusedError
    if hasattr(os, "geteuid") and metadata.st_uid != os.geteuid():
        raise DestinationRefusedError
    if stat.S_IMODE(metadata.st_mode) != OWNER_DIRECTORY_MODE:
        raise DestinationRefusedError


def _create_fresh_work_directory(requested: Path) -> Path:
    """Create one new 0700 leaf beneath an existing, owner-only real directory."""

    destination = _absolute_path(requested)
    if destination == Path("/") or destination.name in {"", ".", ".."}:
        raise DestinationRefusedError
    parent_descriptor: int | None = None
    created_descriptor: int | None = None
    try:
        parent_descriptor = _open_directory_without_symlinks(destination.parent)
        _validate_owner_only_directory(parent_descriptor)
        os.mkdir(destination.name, OWNER_DIRECTORY_MODE, dir_fd=parent_descriptor)
        flags = os.O_RDONLY | os.O_DIRECTORY | int(getattr(os, "O_CLOEXEC", 0))
        flags |= int(getattr(os, "O_NOFOLLOW", 0))
        created_descriptor = os.open(destination.name, flags, dir_fd=parent_descriptor)
        _validate_owner_only_directory(created_descriptor)
        os.fsync(parent_descriptor)
    except (FileExistsError, NotADirectoryError, OSError, DestinationRefusedError) as exc:
        raise DestinationRefusedError from exc
    finally:
        if created_descriptor is not None:
            os.close(created_descriptor)
        if parent_descriptor is not None:
            os.close(parent_descriptor)
    return destination


def _token_identity(path: Path) -> PrivateFileIdentity | None:
    try:
        metadata = path.lstat()
    except (FileNotFoundError, OSError):
        return None
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != OWNER_FILE_MODE
        or metadata.st_nlink != 1
        or not 0 < metadata.st_size <= MAX_TOKEN_BYTES
        or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
    ):
        return None
    return PrivateFileIdentity(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        size_bytes=metadata.st_size,
        ctime_ns=metadata.st_ctime_ns,
    )


def _inspect_acceptance_evidence(path: Path) -> AcceptanceEvidence:
    flags = os.O_RDONLY | int(getattr(os, "O_CLOEXEC", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return AcceptanceEvidence(None, False, False, False)
    try:
        metadata_before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata_before.st_mode)
            or stat.S_IMODE(metadata_before.st_mode) != OWNER_FILE_MODE
            or metadata_before.st_nlink != 1
            or (hasattr(os, "geteuid") and metadata_before.st_uid != os.geteuid())
            or not 0 < metadata_before.st_size <= MAX_ACCEPTANCE_REPORT_BYTES
        ):
            return AcceptanceEvidence(None, False, False, False)
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            report_bytes = handle.read(MAX_ACCEPTANCE_REPORT_BYTES + 1)
        metadata_after = os.fstat(descriptor)
        try:
            path_metadata = path.lstat()
        except OSError:
            return AcceptanceEvidence(None, False, False, False)
        stable_identity = (
            metadata_before.st_dev,
            metadata_before.st_ino,
            metadata_before.st_size,
            metadata_before.st_mtime_ns,
            metadata_before.st_ctime_ns,
        )
        if stable_identity != (
            metadata_after.st_dev,
            metadata_after.st_ino,
            metadata_after.st_size,
            metadata_after.st_mtime_ns,
            metadata_after.st_ctime_ns,
        ) or (path_metadata.st_dev, path_metadata.st_ino) != (
            metadata_before.st_dev,
            metadata_before.st_ino,
        ):
            return AcceptanceEvidence(None, False, False, False)
    except OSError:
        return AcceptanceEvidence(None, False, False, False)
    finally:
        os.close(descriptor)
    if len(report_bytes) != metadata_before.st_size:
        return AcceptanceEvidence(None, False, False, False)
    report_sha256 = hashlib.sha256(report_bytes).hexdigest()
    try:
        payload = json.loads(report_bytes)
    except (UnicodeError, json.JSONDecodeError):
        return AcceptanceEvidence(report_sha256, False, False, False)
    if not isinstance(payload, dict):
        return AcceptanceEvidence(report_sha256, False, False, False)
    schema_version = payload.get("schema_version")
    schema_matches = (
        type(schema_version) is int and schema_version == EXPECTED_ACCEPTANCE_REPORT_SCHEMA_VERSION
    )
    profile = payload.get("profile")
    gate = payload.get("gate")
    profile_matches = (
        isinstance(profile, dict) and profile.get("profile_version") == EXPECTED_PROFILE_IDENTIFIER
    )
    gate_passed = isinstance(gate, dict) and gate.get("passed") is True
    return AcceptanceEvidence(report_sha256, schema_matches, profile_matches, gate_passed)


def _remove_exact_token(
    path: Path,
    identity: PrivateFileIdentity | None,
    *,
    confirmed_server_shutdown: bool,
) -> bool:
    """Remove only the validated token inode and confirm its pathname is absent."""

    if identity is None:
        return False
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        # The server owns the token lifecycle and may unlink it during an orderly shutdown. The
        # identity captured before readiness still binds this absence to the exact token file.
        return confirmed_server_shutdown
    except OSError:
        return False
    current_identity = (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_ctime_ns,
    )
    expected_identity = (
        identity.device,
        identity.inode,
        identity.size_bytes,
        identity.ctime_ns,
    )
    if current_identity != expected_identity:
        return False
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        return False
    try:
        path.unlink()
    except OSError:
        return False
    return not os.path.lexists(path)


def _write_private_receipt(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically publish one owner-only final receipt without overwrite semantics."""

    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | int(getattr(os, "O_CLOEXEC", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    descriptor = os.open(temporary, flags, OWNER_FILE_MODE)
    try:
        os.fchmod(descriptor, OWNER_FILE_MODE)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path, follow_symlinks=False)
        temporary.unlink()
        directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        with contextlib.suppress(FileNotFoundError, OSError):
            temporary.unlink()
        # Never unlink the final pathname after publication. Another process could have replaced
        # it between a failed post-link operation and cleanup; the caller fails closed instead.
        raise


def _health_is_ready(url: str, timeout_seconds: float) -> bool:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "Connection": "close"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            if response.status != 200:
                return False
            declared_length = response.headers.get("Content-Length")
            if declared_length is not None and int(declared_length) > MAX_HEALTH_RESPONSE_BYTES:
                return False
            body = response.read(MAX_HEALTH_RESPONSE_BYTES + 1)
    except (OSError, TimeoutError, ValueError, urllib.error.URLError):
        return False
    if len(body) > MAX_HEALTH_RESPONSE_BYTES:
        return False
    try:
        payload = json.loads(body)
    except (UnicodeError, json.JSONDecodeError):
        return False
    return (
        isinstance(payload, dict)
        and payload.get("status") == "ok"
        and payload.get("catalog_ready") is True
    )


def _spawn_process(
    command: Sequence[str], working_directory: Path, environment: Mapping[str, str]
) -> ChildProcess:
    """Spawn in the inherited process group with no shell and no child output."""

    return subprocess.Popen(
        list(command),
        cwd=working_directory,
        env=dict(environment),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        shell=False,
        start_new_session=False,
    )


def _proc_group_identity(pid: int) -> tuple[int, int, int, str] | None:
    try:
        content = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except (FileNotFoundError, OSError, PermissionError, ProcessLookupError):
        return None
    closing_parenthesis = content.rfind(")")
    if closing_parenthesis < 0:
        return None
    fields = content[closing_parenthesis + 2 :].split()
    if len(fields) <= 19:
        return None
    try:
        return int(fields[1]), int(fields[2]), int(fields[19]), fields[0]
    except ValueError:
        return None


def _proc_identity(pid: int) -> tuple[int, str] | None:
    identity = _proc_group_identity(pid)
    if identity is None:
        return None
    _parent_pid, _process_group, start_ticks, state = identity
    return start_ticks, state


def _direct_children(pid: int) -> list[int]:
    try:
        content = Path(f"/proc/{pid}/task/{pid}/children").read_text(encoding="ascii")
    except (FileNotFoundError, OSError, PermissionError, ProcessLookupError):
        return []
    children: list[int] = []
    for value in content.split():
        with contextlib.suppress(ValueError):
            children.append(int(value))
    return children


def _process_group_is_clean(expected_group: int, ignored_pids: Collection[int]) -> bool:
    """Return true when a Linux process group has no live member except verified ignores."""

    try:
        entries = os.scandir("/proc")
    except OSError:
        return False
    with entries:
        for entry in entries:
            if not entry.name.isdecimal() or int(entry.name) in ignored_pids:
                continue
            try:
                content = Path(f"/proc/{entry.name}/stat").read_text(encoding="ascii")
            except (FileNotFoundError, ProcessLookupError):
                continue
            except OSError:
                return False
            closing_parenthesis = content.rfind(")")
            if closing_parenthesis < 0:
                return False
            fields = content[closing_parenthesis + 2 :].split()
            if len(fields) <= 2:
                return False
            try:
                process_group = int(fields[2])
            except ValueError:
                return False
            if process_group == expected_group and fields[0] not in {"X", "Z"}:
                return False
    return True


def _verified_leader_shim_ancestor_pids() -> frozenset[int]:
    raw_pid = os.environ.get(LEADER_SHIM_PID_ENVIRONMENT_VARIABLE)
    if raw_pid is None:
        raise ValueError
    if not raw_pid.isascii() or not raw_pid.isdecimal() or len(raw_pid) > 10:
        raise ValueError
    shim_pid = int(raw_pid)
    runner_pid = os.getpid()
    runner_parent_pid = os.getppid()
    expected_group = os.getpgrp()
    if (
        shim_pid <= 1
        or str(shim_pid) != raw_pid
        or shim_pid == runner_pid
        or runner_parent_pid <= 1
        or runner_parent_pid == runner_pid
    ):
        raise ValueError

    identities: dict[int, tuple[int, int, int, str]] = {}
    current_pid = runner_parent_pid
    for _depth in range(MAX_LEADER_SHIM_ANCESTOR_DEPTH):
        if current_pid in identities:
            raise ValueError
        identity = _proc_group_identity(current_pid)
        if identity is None:
            raise ValueError
        parent_pid, process_group, start_ticks, state = identity
        if process_group != expected_group or start_ticks <= 0 or state in {"X", "Z"}:
            raise ValueError
        identities[current_pid] = identity
        if current_pid == shim_pid:
            break
        if parent_pid <= 1 or parent_pid == current_pid:
            raise ValueError
        current_pid = parent_pid
    else:
        raise ValueError

    if current_pid != shim_pid or os.getppid() != runner_parent_pid:
        raise ValueError
    if any(_proc_group_identity(pid) != identity for pid, identity in identities.items()):
        raise ValueError
    return frozenset(identities)


def _inherited_process_group_is_clean() -> bool:
    """Confirm the inherited group contains only the runner and verified supervisor chain."""

    try:
        supervisor_ancestor_pids = _verified_leader_shim_ancestor_pids()
    except (OSError, ValueError):
        return False
    ignored_pids = {os.getpid(), *supervisor_ancestor_pids}
    return _process_group_is_clean(os.getpgrp(), ignored_pids)


def _process_tree_identities(root_pid: int) -> dict[int, int]:
    identities: dict[int, int] = {}
    pending = [root_pid]
    while pending:
        pid = pending.pop()
        if pid in identities:
            continue
        identity = _proc_identity(pid)
        if identity is None:
            continue
        start_ticks, _state = identity
        identities[pid] = start_ticks
        pending.extend(_direct_children(pid))
    return identities


def _live_identities(identities: Mapping[int, int]) -> dict[int, int]:
    live: dict[int, int] = {}
    for pid, expected_start_ticks in identities.items():
        current = _proc_identity(pid)
        if current is None:
            continue
        current_start_ticks, state = current
        if current_start_ticks == expected_start_ticks and state not in {"X", "Z"}:
            live[pid] = expected_start_ticks
    return live


def _signal_processes(identities: Mapping[int, int], signum: int) -> None:
    # Descendants are listed after their parent. Reversing reduces the chance of an intermediary
    # exiting and reparenting a still-live server before it receives the signal.
    for pid in reversed(identities):
        current = _proc_identity(pid)
        if current is None or current[0] != identities[pid] or current[1] in {"X", "Z"}:
            continue
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(pid, signum)


def _terminate_process_tree(process: ChildProcess, grace_seconds: float) -> bool:
    """Terminate a child and every captured descendant without signaling the shared group."""

    started = time.monotonic()
    deadline = started + grace_seconds
    term_deadline = started + (grace_seconds * 0.6)
    identities = _process_tree_identities(process.pid)
    _signal_processes(identities, signal.SIGTERM)
    while time.monotonic() < term_deadline:
        # Capture children created during a narrow shutdown race while the root still exists.
        identities.update(_process_tree_identities(process.pid))
        if process.poll() is not None and not _live_identities(identities):
            break
        time.sleep(min(0.05, max(0.0, term_deadline - time.monotonic())))

    survivors = _live_identities(identities)
    if survivors:
        _signal_processes(survivors, signal.SIGKILL)
    try:
        process.wait(timeout=max(0.0, deadline - time.monotonic()))
    except (OSError, subprocess.TimeoutExpired):
        return False

    while _live_identities(identities) and time.monotonic() < deadline:
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    return process.poll() is not None and not _live_identities(identities)


def _default_dependencies() -> RunnerDependencies:
    return RunnerDependencies(
        spawn=_spawn_process,
        health_probe=_health_is_ready,
        token_probe=_token_identity,
        evidence_probe=_inspect_acceptance_evidence,
        terminate=_terminate_process_tree,
        group_cleanup_probe=_inherited_process_group_is_clean,
        monotonic=time.monotonic,
        sleep=time.sleep,
    )


def _child_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop(LEADER_SHIM_PID_ENVIRONMENT_VARIABLE, None)
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    return environment


def _server_command(config: StackConfig, token_file: Path) -> list[str]:
    return [
        "uv",
        "run",
        "meta-archive",
        "serve",
        "--acceptance-read-only",
        "--archive-root",
        os.fspath(config.archive_root),
        "--startup-token-file",
        os.fspath(token_file),
        "--port",
        str(config.port),
    ]


def _client_command(config: StackConfig, token_file: Path, acceptance_report: Path) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        "scripts/real_export_acceptance.py",
        "--archive-root",
        os.fspath(config.archive_root),
        "--base-url",
        f"http://127.0.0.1:{config.port}",
        "--startup-token-file",
        os.fspath(token_file),
        "--browser-report",
        os.fspath(config.browser_report),
        "--reconciliation-mode",
        "production-spot-check",
        "--reconciliation-seed",
        str(config.reconciliation_seed),
        "--output",
        os.fspath(acceptance_report),
    ]


def _normalized_return_code(return_code: int) -> int:
    return return_code if return_code >= 0 else 128 + abs(return_code)


def _reap_finished(process: ChildProcess) -> bool:
    try:
        process.wait(timeout=0.0)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return True


def _terminate_safely(
    dependencies: RunnerDependencies, process: ChildProcess, grace_seconds: float
) -> bool:
    try:
        return dependencies.terminate(process, grace_seconds)
    except BaseException:
        return False


def _wait_for_group_cleanup(dependencies: RunnerDependencies) -> bool:
    """Allow brief process-reaping latency while keeping cleanup strictly bounded."""

    try:
        deadline = dependencies.monotonic() + GROUP_CLEANUP_TIMEOUT_SECONDS
        while True:
            if dependencies.group_cleanup_probe():
                return True
            remaining = deadline - dependencies.monotonic()
            if remaining <= 0:
                return False
            dependencies.sleep(min(GROUP_CLEANUP_POLL_SECONDS, remaining))
    except BaseException:
        # A probe, clock, or sleep failure cannot establish that the inherited group is clean.
        return False


def _wait_for_readiness(
    server: ChildProcess,
    token_file: Path,
    health_url: str,
    config: StackConfig,
    dependencies: RunnerDependencies,
) -> tuple[StackResult | None, PrivateFileIdentity | None]:
    deadline = dependencies.monotonic() + config.readiness_timeout_seconds
    token_identity: PrivateFileIdentity | None = None
    while True:
        if _requested_signal is not None:
            return StackResult("interrupted", 128 + _requested_signal), token_identity
        if server.poll() is not None:
            _reap_finished(server)
            return StackResult("server_exited_before_ready", EXIT_SERVER_EXITED), token_identity
        remaining = deadline - dependencies.monotonic()
        if remaining <= 0:
            return StackResult("readiness_timeout", EXIT_READINESS_TIMEOUT), token_identity
        token_identity = dependencies.token_probe(token_file) or token_identity
        if token_identity is not None and dependencies.health_probe(
            health_url, min(MAX_HEALTH_PROBE_TIMEOUT_SECONDS, remaining)
        ):
            if server.poll() is None:
                return None, token_identity
            _reap_finished(server)
            return StackResult("server_exited_before_ready", EXIT_SERVER_EXITED), token_identity
        dependencies.sleep(
            min(config.poll_seconds, MAX_CONTROL_LOOP_SLEEP_SECONDS, max(0.0, remaining))
        )


def _validate_config(config: StackConfig) -> None:
    archive_root = config.archive_root.lstat()
    browser_report = config.browser_report.lstat()
    if not stat.S_ISDIR(archive_root.st_mode) or stat.S_ISLNK(archive_root.st_mode):
        raise DestinationRefusedError
    if not stat.S_ISREG(browser_report.st_mode) or stat.S_ISLNK(browser_report.st_mode):
        raise DestinationRefusedError
    if not 1 <= config.port <= 65535:
        raise DestinationRefusedError
    bounded_durations = (
        (config.readiness_timeout_seconds, MAX_READY_TIMEOUT_SECONDS),
        (config.poll_seconds, MAX_POLL_SECONDS),
        (config.shutdown_grace_seconds, MAX_SHUTDOWN_GRACE_SECONDS),
    )
    if any(
        not math.isfinite(value) or not 0 < value <= maximum for value, maximum in bounded_durations
    ):
        raise DestinationRefusedError
    if not 0 <= config.reconciliation_seed <= 2**64 - 1:
        raise DestinationRefusedError


def run_stack(config: StackConfig, dependencies: RunnerDependencies | None = None) -> StackResult:
    """Run server and client, returning only a privacy-safe status and exit code."""

    global _requested_signal
    dependencies = dependencies or _default_dependencies()
    _requested_signal = None
    _validate_config(config)
    project_root = Path(__file__).resolve().parents[1]
    work_directory = _create_fresh_work_directory(config.work_directory)
    token_file = work_directory / "startup-token"
    acceptance_report = work_directory / "acceptance-report.json"
    receipt_file = work_directory / "stack-receipt.json"
    health_url = f"http://127.0.0.1:{config.port}/api/v1/health"
    environment = _child_environment()
    server: ChildProcess | None = None
    client: ChildProcess | None = None
    result = StackResult("internal_error", EXIT_INTERNAL_ERROR)
    evidence = AcceptanceEvidence(None, False, False, False)
    token_identity: PrivateFileIdentity | None = None
    server_started = False
    server_health_ready = False
    server_cleanup_confirmed = True
    client_reaped = True
    client_exit_code: int | None = None
    token_exact_inode_removed = False
    inherited_process_group_clean = False

    previous_handlers = {
        signal.SIGINT: signal.getsignal(signal.SIGINT),
        signal.SIGTERM: signal.getsignal(signal.SIGTERM),
    }
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    try:
        try:
            server = dependencies.spawn(
                _server_command(config, token_file), project_root, environment
            )
        except OSError:
            result = StackResult("server_start_failed", EXIT_SERVER_START_FAILED)
        else:
            server_started = True
            readiness_result, token_identity = _wait_for_readiness(
                server, token_file, health_url, config, dependencies
            )
            if readiness_result is not None:
                result = readiness_result
            else:
                server_health_ready = True
                try:
                    client = dependencies.spawn(
                        _client_command(config, token_file, acceptance_report),
                        project_root,
                        environment,
                    )
                except OSError:
                    result = StackResult("client_start_failed", EXIT_CLIENT_START_FAILED)
                else:
                    while True:
                        if _requested_signal is not None:
                            result = StackResult("interrupted", 128 + _requested_signal)
                            break
                        if server.poll() is not None:
                            _reap_finished(server)
                            result = StackResult("server_exited_during_client", EXIT_SERVER_EXITED)
                            break
                        client_return_code = client.poll()
                        if client_return_code is not None:
                            client_reaped = _reap_finished(client)
                            client_exit_code = _normalized_return_code(client_return_code)
                            evidence = dependencies.evidence_probe(acceptance_report)
                            if client_exit_code != 0:
                                result = StackResult("client_failed", client_exit_code)
                            elif (
                                evidence.well_formed
                                and evidence.profile_matches
                                and evidence.gate_passed
                            ):
                                result = StackResult("passed", 0)
                            else:
                                result = StackResult(
                                    "acceptance_evidence_invalid", EXIT_EVIDENCE_INVALID
                                )
                            break
                        dependencies.sleep(min(config.poll_seconds, MAX_CONTROL_LOOP_SLEEP_SECONDS))
    except BaseException:
        result = StackResult("internal_error", EXIT_INTERNAL_ERROR)
    finally:
        cleanup_grace_seconds = config.shutdown_grace_seconds
        if client is not None:
            client_reaped = _terminate_safely(dependencies, client, cleanup_grace_seconds)
        if client is not None and client.poll() is not None:
            polled_client_exit = client.poll()
            if polled_client_exit is not None:
                client_exit_code = _normalized_return_code(polled_client_exit)
        if server is not None:
            server_cleanup_confirmed = _terminate_safely(
                dependencies, server, cleanup_grace_seconds
            )
        token_identity = token_identity or dependencies.token_probe(token_file)
        token_exact_inode_removed = _remove_exact_token(
            token_file,
            token_identity,
            confirmed_server_shutdown=server_cleanup_confirmed,
        )
        inherited_process_group_clean = _wait_for_group_cleanup(dependencies)
        for watched_signal, previous_handler in previous_handlers.items():
            signal.signal(watched_signal, previous_handler)

    exact_cleanup_confirmed = (
        client_reaped
        and server_cleanup_confirmed
        and token_exact_inode_removed
        and not os.path.lexists(token_file)
        and inherited_process_group_clean
    )
    if result.exit_code == 0 and (
        not server_started
        or not server_health_ready
        or not evidence.well_formed
        or not evidence.profile_matches
        or not evidence.gate_passed
        or client_exit_code != 0
        or not exact_cleanup_confirmed
    ):
        result = StackResult("cleanup_or_evidence_unconfirmed", EXIT_CLEANUP_FAILED)
    elif (
        (server_started and not server_cleanup_confirmed)
        or (token_identity is not None and not token_exact_inode_removed)
        or not inherited_process_group_clean
    ):
        result = StackResult("cleanup_failed", EXIT_CLEANUP_FAILED)

    receipt: dict[str, object] = {
        "schema_version": 1,
        "acceptance_report_sha256": evidence.sha256,
        "acceptance_report_well_formed": evidence.well_formed,
        "expected_public_profile_identifier": EXPECTED_PROFILE_IDENTIFIER,
        "profile_matches": evidence.profile_matches,
        "gate_passed": evidence.gate_passed,
        "client_exit_code": client_exit_code,
        "server_started": server_started,
        "server_health_ready": server_health_ready,
        "cleanup": {
            "client_terminated_or_reaped": client_reaped,
            "server_tree_terminated_and_waited": server_cleanup_confirmed,
            "inherited_process_group_clean": inherited_process_group_clean,
            "token_exact_inode_removed": token_exact_inode_removed,
            "exact_cleanup_confirmed": exact_cleanup_confirmed,
        },
        "stack_status": result.status,
        "stack_exit_code": result.exit_code,
    }
    try:
        _write_private_receipt(receipt_file, receipt)
    except BaseException:
        return StackResult("receipt_write_failed", EXIT_INTERNAL_ERROR)
    return result


class _PrivateArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        del message
        raise DestinationRefusedError


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError
    return parsed


def _unsigned_seed(value: str) -> int:
    parsed = int(value)
    if not 0 <= parsed <= 2**64 - 1:
        raise argparse.ArgumentTypeError
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = _PrivateArgumentParser(
        description="Run one supervised production archive spot-check process tree."
    )
    parser.add_argument("--archive-root", required=True, type=Path)
    parser.add_argument("--browser-report", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--ready-timeout-seconds", type=_positive_float, default=DEFAULT_READY_TIMEOUT_SECONDS
    )
    parser.add_argument("--poll-seconds", type=_positive_float, default=DEFAULT_POLL_SECONDS)
    parser.add_argument(
        "--shutdown-grace-seconds",
        type=_positive_float,
        default=DEFAULT_SHUTDOWN_GRACE_SECONDS,
    )
    parser.add_argument(
        "--reconciliation-seed", type=_unsigned_seed, default=DEFAULT_RECONCILIATION_SEED
    )
    return parser


def _arguments(argv: Sequence[str]) -> StackConfig:
    arguments = _parser().parse_args(argv)
    config = StackConfig(
        archive_root=_absolute_path(cast(Path, arguments.archive_root)),
        browser_report=_absolute_path(cast(Path, arguments.browser_report)),
        work_directory=_absolute_path(cast(Path, arguments.work_dir)),
        port=cast(int, arguments.port),
        readiness_timeout_seconds=cast(float, arguments.ready_timeout_seconds),
        poll_seconds=cast(float, arguments.poll_seconds),
        shutdown_grace_seconds=cast(float, arguments.shutdown_grace_seconds),
        reconciliation_seed=cast(int, arguments.reconciliation_seed),
    )
    if (
        config.readiness_timeout_seconds > MAX_READY_TIMEOUT_SECONDS
        or config.poll_seconds > MAX_POLL_SECONDS
        or config.shutdown_grace_seconds > MAX_SHUTDOWN_GRACE_SECONDS
    ):
        raise DestinationRefusedError
    return config


def main(argv: Sequence[str] | None = None) -> int:
    os.umask(0o077)
    try:
        config = _arguments(argv if argv is not None else sys.argv[1:])
        result = run_stack(config)
    except (DestinationRefusedError, FileNotFoundError, NotADirectoryError, PermissionError):
        result = StackResult("destination_refused", EXIT_DESTINATION_REFUSED)
    except KeyboardInterrupt:
        result = StackResult("interrupted", 130)
    except Exception:
        # Exception strings can contain archive paths, response URLs, or private subprocess data.
        result = StackResult("internal_error", EXIT_INTERNAL_ERROR)
    print(f"production-acceptance-stack: {result.status} (exit {result.exit_code})")
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
