#!/usr/bin/env python3
"""Run a private acceptance command inside a resource-bounded process group.

The supervisor deliberately records no command line, environment value, working directory,
process identifier, or child output. Its JSONL journal is fsynced after every event and its
aggregate report is replaced atomically after every sample, so an interrupted run retains the
last owner-only checkpoint.

Linux ``/proc/<pid>/io`` supplies both ``read_bytes`` (physical storage I/O) and ``rchar``
(logical bytes returned by read-like syscalls). Those counters include waited-for children, so
the proc-tree mode uses the monotonic peak of each instantaneous process-group sum instead of
adding per-process deltas. A delegated cgroup v2 ``io.stat`` counter is also available as a
fail-closed physical-read source. Logical reads always use the proc-tree-inclusive measurement.
The requested command runs beneath a leader shim that waits for it and stays alive until one
post-fold sample is acknowledged, closing the otherwise unobservable final-exit interval.
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import json
import math
import os
import secrets
import signal
import stat
import struct
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType
from typing import Final, Literal, TextIO, cast

OWNER_DIRECTORY_MODE: Final = 0o700
OWNER_FILE_MODE: Final = 0o600

# Conservative limits for an 11.4 GiB WSL VM. The RSS ceiling leaves most memory available to
# the API server, the desktop, and other shells. Every limit can be lowered or raised explicitly.
DEFAULT_WALL_SECONDS: Final = 45 * 60.0
DEFAULT_MAX_RSS_BYTES: Final = 4 * 1024**3
DEFAULT_MAX_READ_BYTES: Final = 32 * 1024**3
DEFAULT_MAX_RCHAR_BYTES: Final = 256 * 1024**3
DEFAULT_SAMPLE_SECONDS: Final = 0.5
DEFAULT_TERM_GRACE_SECONDS: Final = 8.0
DEFAULT_MAX_PROCESS_IDENTITIES: Final = 16_384

MIN_WALL_SECONDS: Final = 0.05
MAX_WALL_SECONDS: Final = 24 * 60 * 60.0
MIN_SAMPLE_SECONDS: Final = 0.01
MAX_SAMPLE_SECONDS: Final = 5.0
MIN_TERM_GRACE_SECONDS: Final = 0.05
MAX_TERM_GRACE_SECONDS: Final = 60.0
MAX_LIMIT_BYTES: Final = 2**63 - 1
MAX_PROCESS_IDENTITIES: Final = 1_000_000
MAX_PROC_SELF_CGROUP_BYTES: Final = 64 * 1024
MAX_CGROUP_IO_STAT_BYTES: Final = 1024 * 1024
MAX_CGROUP_COUNTER: Final = 2**64 - 1
MAX_CGROUP_DEVICE_NUMBER: Final = 2**32 - 1

PROC_TREE_INCLUSIVE_SOURCE: Final = "proc-tree-inclusive"
CGROUP_V2_PHYSICAL_READ_SOURCE: Final = "cgroup-v2"
PhysicalReadSource = Literal["proc-tree-inclusive", "cgroup-v2"]

PROC_SELF_CGROUP_PATH: Final = Path("/proc/self/cgroup")
CGROUP_V2_ROOT: Final = Path("/sys/fs/cgroup")

EXIT_TIMEOUT: Final = 124
EXIT_RSS_LIMIT: Final = 125
EXIT_READ_LIMIT: Final = 126
EXIT_RCHAR_LIMIT: Final = 123
EXIT_PROCESS_IDENTITY_LIMIT: Final = 122
EXIT_SUPERVISOR_ERROR: Final = 127

LEADER_SHIM_STATUS_STARTED: Final = 0
LEADER_SHIM_STATUS_START_FAILED: Final = 1
LEADER_SHIM_STATUS_FORMAT: Final = "!Bi"
LEADER_SHIM_STATUS_BYTES: Final = struct.calcsize(LEADER_SHIM_STATUS_FORMAT)
LEADER_SHIM_ACK: Final = b"\x01"
LEADER_SHIM_PID_ENVIRONMENT_VARIABLE: Final = "META_ARCHIVE_ACCEPTANCE_LEADER_SHIM_PID"

_requested_signal: int | None = None


@dataclass(frozen=True)
class Limits:
    wall_seconds: float
    max_rss_bytes: int
    max_read_bytes: int
    max_rchar_bytes: int
    sample_seconds: float
    term_grace_seconds: float
    max_process_identities: int


@dataclass(frozen=True)
class ProcReading:
    identity: tuple[int, int]
    state: str
    rss_bytes: int
    read_bytes: int
    rchar_bytes: int


@dataclass(frozen=True)
class ResourceSample:
    process_count: int
    tree_rss_bytes: int
    tree_read_bytes: int
    tree_rchar_bytes: int
    leader_observed: bool


@dataclass(frozen=True)
class LeaderShimStatus:
    workload_started: bool
    raw_exit_code: int


class ProcMeasurementUnavailable(RuntimeError):
    """The supervisor could not safely measure its child process group."""


class CgroupMeasurementUnavailable(RuntimeError):
    """The supervisor could not safely measure its own unified cgroup."""


class ProcessIdentityLimitExceeded(RuntimeError):
    """The bounded process-identity tracker reached its configured ceiling."""


@dataclass
class ResourceTracker:
    """Track inclusive proc counters without double-counting waited-for children.

    Linux folds a waited-for child's I/O counters into its parent. Summing deltas by process
    identity therefore counts the same bytes once for the child and again for the parent. The
    instantaneous sum across current group members already represents the inclusive tree total.
    Retaining its monotonic peak bridges ordinary zombie/reap gaps while preventing a fold from
    increasing the observed total.
    """

    last_read_bytes: dict[tuple[int, int], int] = field(default_factory=dict)
    last_rchar_bytes: dict[tuple[int, int], int] = field(default_factory=dict)
    start_ticks_by_pid: dict[int, int] = field(default_factory=dict)
    observed_identities: set[tuple[int, int]] = field(default_factory=set)
    total_read_bytes: int = 0
    total_rchar_bytes: int = 0
    peak_rss_bytes: int = 0
    peak_process_count: int = 0
    sample_count: int = 0

    def sample(
        self,
        process_group: int,
        *,
        leader_pid: int,
        max_process_identities: int,
    ) -> ResourceSample:
        readings = _read_process_group(process_group)
        identities = {reading.identity for reading in readings}
        prospective_identities = self.observed_identities.union(identities)
        if len(prospective_identities) > max_process_identities:
            raise ProcessIdentityLimitExceeded
        self.observed_identities = prospective_identities
        rss_bytes = 0
        process_count = 0
        instantaneous_read_bytes = 0
        instantaneous_rchar_bytes = 0
        for reading in readings:
            pid, start_ticks = reading.identity
            previous_start_ticks = self.start_ticks_by_pid.get(pid)
            if previous_start_ticks is not None and previous_start_ticks != start_ticks:
                raise ProcMeasurementUnavailable
            self.start_ticks_by_pid[pid] = start_ticks
            if reading.state not in {"X", "Z"}:
                previous = self.last_read_bytes.get(reading.identity)
                previous_rchar = self.last_rchar_bytes.get(reading.identity)
                read_bytes = reading.read_bytes
                rchar_bytes = reading.rchar_bytes
                if (previous is not None and read_bytes < previous) or (
                    previous_rchar is not None and rchar_bytes < previous_rchar
                ):
                    # waitpid accounting can race a /proc read while the leader folds its child.
                    # Confirm the same live identity once more and reject any stable decrease.
                    confirmed_stat = _read_proc_stat(pid)
                    if confirmed_stat is None:
                        raise ProcMeasurementUnavailable
                    confirmed_group, confirmed_start_ticks, confirmed_state = confirmed_stat
                    if (
                        confirmed_group != process_group
                        or confirmed_start_ticks != start_ticks
                        or confirmed_state in {"X", "Z"}
                    ):
                        raise ProcMeasurementUnavailable
                    confirmed_io = _read_io_bytes(pid)
                    if confirmed_io is None:
                        raise ProcMeasurementUnavailable
                    read_bytes, rchar_bytes = confirmed_io
                    if (previous is not None and read_bytes < previous) or (
                        previous_rchar is not None and rchar_bytes < previous_rchar
                    ):
                        raise ProcMeasurementUnavailable
                self.last_read_bytes[reading.identity] = read_bytes
                self.last_rchar_bytes[reading.identity] = rchar_bytes
                instantaneous_read_bytes += read_bytes
                instantaneous_rchar_bytes += rchar_bytes
            if reading.state not in {"X", "Z"}:
                rss_bytes += reading.rss_bytes
                process_count += 1

        self.sample_count += 1
        self.total_read_bytes = max(self.total_read_bytes, instantaneous_read_bytes)
        self.total_rchar_bytes = max(self.total_rchar_bytes, instantaneous_rchar_bytes)
        self.peak_rss_bytes = max(self.peak_rss_bytes, rss_bytes)
        self.peak_process_count = max(self.peak_process_count, process_count)
        return ResourceSample(
            process_count=process_count,
            tree_rss_bytes=rss_bytes,
            tree_read_bytes=self.total_read_bytes,
            tree_rchar_bytes=self.total_rchar_bytes,
            leader_observed=any(reading.identity[0] == leader_pid for reading in readings),
        )


def _read_bounded_descriptor(descriptor: int, maximum_bytes: int) -> bytes:
    content = bytearray()
    while len(content) <= maximum_bytes:
        chunk = os.read(descriptor, min(64 * 1024, maximum_bytes + 1 - len(content)))
        if not chunk:
            return bytes(content)
        content.extend(chunk)
    raise CgroupMeasurementUnavailable


def _read_bounded_nofollow_file(path: Path, maximum_bytes: int) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise CgroupMeasurementUnavailable
        return _read_bounded_descriptor(descriptor, maximum_bytes)
    except (OSError, UnicodeError) as exc:
        raise CgroupMeasurementUnavailable from exc
    finally:
        if descriptor is not None:
            with contextlib.suppress(OSError):
                os.close(descriptor)


def _parse_unified_cgroup_components(content: bytes) -> tuple[str, ...]:
    try:
        decoded = content.decode("ascii")
    except UnicodeDecodeError as exc:
        raise CgroupMeasurementUnavailable from exc
    if "\x00" in decoded or "\r" in decoded:
        raise CgroupMeasurementUnavailable
    lines = decoded.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if len(lines) != 1 or not lines[0]:
        raise CgroupMeasurementUnavailable
    fields = lines[0].split(":", 2)
    if len(fields) != 3:
        raise CgroupMeasurementUnavailable
    hierarchy, controllers, cgroup_path = fields
    if hierarchy != "0" or controllers != "" or not cgroup_path.startswith("/"):
        raise CgroupMeasurementUnavailable
    if cgroup_path == "/":
        return ()
    raw_components = cgroup_path[1:].split("/")
    if any(
        not component
        or component in {".", ".."}
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in component)
        for component in raw_components
    ):
        raise CgroupMeasurementUnavailable
    return tuple(raw_components)


def _self_unified_cgroup_components() -> tuple[str, ...]:
    return _parse_unified_cgroup_components(
        _read_bounded_nofollow_file(PROC_SELF_CGROUP_PATH, MAX_PROC_SELF_CGROUP_BYTES)
    )


def _process_unified_cgroup_components(pid: int) -> tuple[str, ...]:
    if pid <= 0:
        raise CgroupMeasurementUnavailable
    return _parse_unified_cgroup_components(
        _read_bounded_nofollow_file(
            Path(f"/proc/{pid}/cgroup"),
            MAX_PROC_SELF_CGROUP_BYTES,
        )
    )


def _open_cgroup_directory(components: Sequence[str]) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(CGROUP_V2_ROOT, flags)
        for component in components:
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        result = descriptor
        descriptor = None
        return result
    except OSError as exc:
        raise CgroupMeasurementUnavailable from exc
    finally:
        if descriptor is not None:
            with contextlib.suppress(OSError):
                os.close(descriptor)


def _read_cgroup_io_stat(directory_fd: int) -> dict[tuple[int, int], int]:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open("io.stat", flags, dir_fd=directory_fd)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise CgroupMeasurementUnavailable
        content = _read_bounded_descriptor(descriptor, MAX_CGROUP_IO_STAT_BYTES)
    except OSError as exc:
        raise CgroupMeasurementUnavailable from exc
    finally:
        if descriptor is not None:
            with contextlib.suppress(OSError):
                os.close(descriptor)

    try:
        decoded = content.decode("ascii")
    except UnicodeDecodeError as exc:
        raise CgroupMeasurementUnavailable from exc
    if "\x00" in decoded or "\r" in decoded:
        raise CgroupMeasurementUnavailable

    reads_by_device: dict[tuple[int, int], int] = {}
    for line in decoded.splitlines():
        tokens = line.split()
        if not tokens:
            continue
        device_components = tokens[0].split(":")
        if len(device_components) != 2 or not all(
            component.isascii() and component.isdecimal() and len(component) <= 10
            for component in device_components
        ):
            raise CgroupMeasurementUnavailable
        device = (int(device_components[0]), int(device_components[1]))
        if any(component > MAX_CGROUP_DEVICE_NUMBER for component in device):
            raise CgroupMeasurementUnavailable
        if device in reads_by_device:
            raise CgroupMeasurementUnavailable
        fields: dict[str, int] = {}
        for token in tokens[1:]:
            key_value = token.split("=", 1)
            if len(key_value) != 2:
                raise CgroupMeasurementUnavailable
            key, raw_value = key_value
            if (
                not key
                or not key.isascii()
                or not key[0].isalpha()
                or any(not (character.isalnum() or character == "_") for character in key)
                or key in fields
                or not raw_value.isascii()
                or not raw_value.isdecimal()
                or len(raw_value) > 20
            ):
                raise CgroupMeasurementUnavailable
            value = int(raw_value)
            if value > MAX_CGROUP_COUNTER:
                raise CgroupMeasurementUnavailable
            fields[key] = value
        if "rbytes" not in fields:
            raise CgroupMeasurementUnavailable
        reads_by_device[device] = fields["rbytes"]
    if sum(reads_by_device.values()) > MAX_CGROUP_COUNTER:
        raise CgroupMeasurementUnavailable
    return reads_by_device


@dataclass
class CgroupV2ReadTracker:
    """Measure one unified cgroup's physical read delta from an anchored ``io.stat``."""

    directory_fd: int
    cgroup_components: tuple[str, ...]
    baseline_reads: dict[tuple[int, int], int]
    previous_reads: dict[tuple[int, int], int]
    total_read_bytes: int = 0

    @classmethod
    def start(cls) -> CgroupV2ReadTracker:
        components = _self_unified_cgroup_components()
        directory_fd = _open_cgroup_directory(components)
        try:
            baseline = _read_cgroup_io_stat(directory_fd)
        except BaseException:
            os.close(directory_fd)
            raise
        return cls(directory_fd, components, baseline, dict(baseline))

    def sample(self, *, required_member_pid: int | None = None) -> int:
        if _self_unified_cgroup_components() != self.cgroup_components:
            raise CgroupMeasurementUnavailable
        if (
            required_member_pid is not None
            and _process_unified_cgroup_components(required_member_pid) != self.cgroup_components
        ):
            raise CgroupMeasurementUnavailable
        current_reads = _read_cgroup_io_stat(self.directory_fd)
        if not set(self.previous_reads).issubset(current_reads):
            raise CgroupMeasurementUnavailable
        for device, previous in self.previous_reads.items():
            if current_reads[device] < previous:
                raise CgroupMeasurementUnavailable
        observed = sum(
            current - self.baseline_reads.get(device, 0)
            for device, current in current_reads.items()
        )
        if observed < self.total_read_bytes or observed > MAX_CGROUP_COUNTER:
            raise CgroupMeasurementUnavailable
        self.previous_reads = current_reads
        self.total_read_bytes = observed
        return observed

    def close(self) -> None:
        os.close(self.directory_fd)


class JsonlJournal:
    def __init__(self, directory_fd: int) -> None:
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(
            "progress.jsonl",
            flags,
            OWNER_FILE_MODE,
            dir_fd=directory_fd,
        )
        os.fchmod(descriptor, OWNER_FILE_MODE)
        self._handle: TextIO = os.fdopen(descriptor, "w", encoding="utf-8")

    def write(self, payload: Mapping[str, object]) -> None:
        self._handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def close(self) -> None:
        self._handle.close()


def _signal_handler(signum: int, _frame: FrameType | None) -> None:
    global _requested_signal
    _requested_signal = signum


def _read_proc_stat(pid: int) -> tuple[int, int, str] | None:
    """Return (process group, start ticks, state), tolerating exit races."""

    try:
        content = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except (FileNotFoundError, ProcessLookupError):
        return None
    except OSError as exc:
        raise ProcMeasurementUnavailable from exc
    closing_parenthesis = content.rfind(")")
    if closing_parenthesis < 0:
        raise ProcMeasurementUnavailable
    fields = content[closing_parenthesis + 2 :].split()
    if len(fields) <= 19:
        raise ProcMeasurementUnavailable
    try:
        return int(fields[2]), int(fields[19]), fields[0]
    except ValueError as exc:
        raise ProcMeasurementUnavailable from exc


def _read_rss_bytes(pid: int) -> int | None:
    try:
        fields = Path(f"/proc/{pid}/statm").read_text(encoding="ascii").split()
        return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")
    except (FileNotFoundError, ProcessLookupError):
        return None
    except (IndexError, OSError, ValueError) as exc:
        raise ProcMeasurementUnavailable from exc


def _read_io_bytes(pid: int) -> tuple[int, int] | None:
    try:
        lines = Path(f"/proc/{pid}/io").read_text(encoding="ascii").splitlines()
    except (FileNotFoundError, ProcessLookupError):
        return None
    except OSError as exc:
        raise ProcMeasurementUnavailable from exc
    read_bytes: int | None = None
    rchar_bytes: int | None = None
    for line in lines:
        if line.startswith("read_bytes:"):
            try:
                read_bytes = int(line.split(":", 1)[1])
            except ValueError as exc:
                raise ProcMeasurementUnavailable from exc
        elif line.startswith("rchar:"):
            try:
                rchar_bytes = int(line.split(":", 1)[1])
            except ValueError as exc:
                raise ProcMeasurementUnavailable from exc
    if read_bytes is None or rchar_bytes is None:
        raise ProcMeasurementUnavailable
    return read_bytes, rchar_bytes


def _read_process_group(process_group: int) -> list[ProcReading]:
    readings: list[ProcReading] = []
    try:
        entries = os.scandir("/proc")
    except OSError as exc:
        raise ProcMeasurementUnavailable from exc
    with entries:
        for entry in entries:
            if not entry.name.isdecimal():
                continue
            pid = int(entry.name)
            stat_fields = _read_proc_stat(pid)
            if stat_fields is None:
                continue
            group, start_ticks, state = stat_fields
            if group != process_group:
                continue
            if state in {"X", "Z"}:
                # Linux can deny /proc/<pid>/io once a process is a zombie. Its final unsampled
                # tail is unavailable, but the process is already stopped and must not turn an
                # otherwise confirmed clean exit into a false measurement outage.
                readings.append(
                    ProcReading(
                        identity=(pid, start_ticks),
                        state=state,
                        rss_bytes=0,
                        read_bytes=0,
                        rchar_bytes=0,
                    )
                )
                continue
            io_bytes: tuple[int, int] | None = None
            rss_bytes: int | None = None
            try:
                io_bytes = _read_io_bytes(pid)
                rss_bytes = _read_rss_bytes(pid)
            except ProcMeasurementUnavailable:
                pass
            if io_bytes is None or rss_bytes is None:
                # A process can become a zombie after the first stat read, at which point Linux
                # may deny its io file before stat exposes the terminal state. Retry this exact
                # identity across a tiny bounded race window. A persistently live unreadable
                # member still fails closed; a confirmed terminal member is covered by the
                # retained peak and the leader shim's mandatory post-fold sample.
                recovered = False
                terminal = False
                for _attempt in range(3):
                    terminal_stat = _read_proc_stat(pid)
                    if terminal_stat is None:
                        terminal = True
                        break
                    terminal_group, terminal_start_ticks, terminal_state = terminal_stat
                    if terminal_group != process_group or terminal_start_ticks != start_ticks:
                        raise ProcMeasurementUnavailable
                    if terminal_state in {"X", "Z"}:
                        readings.append(
                            ProcReading(
                                identity=(pid, start_ticks),
                                state=terminal_state,
                                rss_bytes=0,
                                read_bytes=0,
                                rchar_bytes=0,
                            )
                        )
                        terminal = True
                        break
                    try:
                        io_bytes = _read_io_bytes(pid)
                        rss_bytes = _read_rss_bytes(pid)
                    except ProcMeasurementUnavailable:
                        io_bytes = None
                        rss_bytes = None
                    if io_bytes is None or rss_bytes is None:
                        time.sleep(0.001)
                        continue
                    recovered = True
                    break
                if terminal:
                    continue
                if not recovered:
                    raise ProcMeasurementUnavailable
            assert io_bytes is not None and rss_bytes is not None
            read_bytes, rchar_bytes = io_bytes
            readings.append(
                ProcReading(
                    identity=(pid, start_ticks),
                    state=state,
                    rss_bytes=rss_bytes,
                    read_bytes=read_bytes,
                    rchar_bytes=rchar_bytes,
                )
            )
    return readings


def _normalize_return_code(return_code: int) -> int:
    return return_code if return_code >= 0 else 128 + abs(return_code)


def _write_pipe_payload(descriptor: int, payload: bytes) -> None:
    written = 0
    while written < len(payload):
        try:
            count = os.write(descriptor, payload[written:])
        except InterruptedError:
            continue
        if count <= 0:
            raise OSError("pipe write failed")
        written += count


def _leader_shim_main(arguments: Sequence[str]) -> int:
    """Wait for the workload, expose folded counters, then await supervisor acknowledgement."""

    if len(arguments) < 4 or arguments[2] != "--":
        return EXIT_SUPERVISOR_ERROR
    try:
        status_descriptor = int(arguments[0])
        acknowledgement_descriptor = int(arguments[1])
    except ValueError:
        return EXIT_SUPERVISOR_ERROR
    if (
        status_descriptor < 3
        or acknowledgement_descriptor < 3
        or status_descriptor == acknowledgement_descriptor
    ):
        return EXIT_SUPERVISOR_ERROR
    try:
        if not stat.S_ISFIFO(os.fstat(status_descriptor).st_mode) or not stat.S_ISFIFO(
            os.fstat(acknowledgement_descriptor).st_mode
        ):
            return EXIT_SUPERVISOR_ERROR
    except OSError:
        return EXIT_SUPERVISOR_ERROR

    command = list(arguments[3:])
    if not command:
        return EXIT_SUPERVISOR_ERROR
    workload_environment = dict(os.environ)
    workload_environment[LEADER_SHIM_PID_ENVIRONMENT_VARIABLE] = str(os.getpid())
    workload_started = True
    try:
        workload = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            env=workload_environment,
        )
    except OSError:
        workload_started = False
        raw_exit_code = EXIT_SUPERVISOR_ERROR
    else:
        raw_exit_code = workload.wait()

    kind = LEADER_SHIM_STATUS_STARTED if workload_started else LEADER_SHIM_STATUS_START_FAILED
    try:
        _write_pipe_payload(
            status_descriptor,
            struct.pack(LEADER_SHIM_STATUS_FORMAT, kind, raw_exit_code),
        )
        while True:
            try:
                acknowledgement = os.read(acknowledgement_descriptor, 1)
                break
            except InterruptedError:
                continue
        if acknowledgement != LEADER_SHIM_ACK:
            return EXIT_SUPERVISOR_ERROR
    except OSError:
        return EXIT_SUPERVISOR_ERROR
    finally:
        with contextlib.suppress(OSError):
            os.close(status_descriptor)
        with contextlib.suppress(OSError):
            os.close(acknowledgement_descriptor)
    return _normalize_return_code(raw_exit_code) if workload_started else EXIT_SUPERVISOR_ERROR


def _spawn_leader_shim(
    command: Sequence[str],
) -> tuple[subprocess.Popen[bytes], int, int]:
    status_read_descriptor: int | None = None
    status_write_descriptor: int | None = None
    acknowledgement_read_descriptor: int | None = None
    acknowledgement_write_descriptor: int | None = None
    try:
        status_read_descriptor, status_write_descriptor = os.pipe2(os.O_CLOEXEC)
        acknowledgement_read_descriptor, acknowledgement_write_descriptor = os.pipe2(os.O_CLOEXEC)
        os.set_blocking(status_read_descriptor, False)
        child = subprocess.Popen(
            [
                sys.executable,
                os.path.abspath(__file__),
                "--_leader-shim",
                str(status_write_descriptor),
                str(acknowledgement_read_descriptor),
                "--",
                *command,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            pass_fds=(status_write_descriptor, acknowledgement_read_descriptor),
            start_new_session=True,
        )
        os.close(status_write_descriptor)
        status_write_descriptor = None
        os.close(acknowledgement_read_descriptor)
        acknowledgement_read_descriptor = None
        result = (child, status_read_descriptor, acknowledgement_write_descriptor)
        status_read_descriptor = None
        acknowledgement_write_descriptor = None
        return result
    except OSError:
        raise
    finally:
        for descriptor in (
            status_read_descriptor,
            status_write_descriptor,
            acknowledgement_read_descriptor,
            acknowledgement_write_descriptor,
        ):
            if descriptor is not None:
                with contextlib.suppress(OSError):
                    os.close(descriptor)


def _read_leader_shim_status(
    descriptor: int,
    buffer: bytearray,
) -> LeaderShimStatus | None:
    while len(buffer) < LEADER_SHIM_STATUS_BYTES:
        try:
            chunk = os.read(descriptor, LEADER_SHIM_STATUS_BYTES - len(buffer))
        except BlockingIOError:
            return None
        except InterruptedError:
            continue
        except OSError as exc:
            raise ProcMeasurementUnavailable from exc
        if not chunk:
            raise ProcMeasurementUnavailable
        buffer.extend(chunk)
    try:
        kind, raw_exit_code = struct.unpack(LEADER_SHIM_STATUS_FORMAT, buffer)
    except struct.error as exc:
        raise ProcMeasurementUnavailable from exc
    if kind not in {LEADER_SHIM_STATUS_STARTED, LEADER_SHIM_STATUS_START_FAILED}:
        raise ProcMeasurementUnavailable
    if not -255 <= raw_exit_code <= 255:
        raise ProcMeasurementUnavailable
    return LeaderShimStatus(kind == LEADER_SHIM_STATUS_STARTED, raw_exit_code)


def _acknowledge_leader_shim(descriptor: int) -> None:
    try:
        _write_pipe_payload(descriptor, LEADER_SHIM_ACK)
    except OSError as exc:
        raise ProcMeasurementUnavailable from exc


def _atomic_write_json(directory_fd: int, payload: Mapping[str, object]) -> None:
    temporary_name: str | None = None
    descriptor: int | None = None
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    for _attempt in range(16):
        candidate = f".report-{secrets.token_hex(12)}"
        try:
            descriptor = os.open(candidate, flags, OWNER_FILE_MODE, dir_fd=directory_fd)
        except FileExistsError:
            continue
        temporary_name = candidate
        break
    if descriptor is None or temporary_name is None:
        raise OSError("could not reserve a private report checkpoint")
    try:
        os.fchmod(descriptor, OWNER_FILE_MODE)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        descriptor = None
        os.replace(
            temporary_name,
            "report.json",
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        temporary_name = None
        os.fsync(directory_fd)
    finally:
        if descriptor is not None:
            with contextlib.suppress(OSError):
                os.close(descriptor)
        if temporary_name is not None:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary_name, dir_fd=directory_fd)


def _open_directory_at(parent_fd: int, component: str) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        return os.open(component, flags, dir_fd=parent_fd)
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ValueError("output path contains a symlink or non-directory component") from None
        raise


def _prepare_output_directory(path: Path) -> int:
    """Atomically create a brand-new private output directory and return its anchored fd."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    components = absolute.parts[1:]
    if not components:
        raise ValueError("output destination must not be the filesystem root")

    directory_fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for component in components[:-1]:
            try:
                next_fd = _open_directory_at(directory_fd, component)
            except FileNotFoundError:
                with contextlib.suppress(FileExistsError):
                    os.mkdir(component, OWNER_DIRECTORY_MODE, dir_fd=directory_fd)
                next_fd = _open_directory_at(directory_fd, component)
            os.close(directory_fd)
            directory_fd = next_fd

        final_component = components[-1]
        os.mkdir(final_component, OWNER_DIRECTORY_MODE, dir_fd=directory_fd)
        output_fd: int | None = None
        try:
            output_fd = _open_directory_at(directory_fd, final_component)
            os.fchmod(output_fd, OWNER_DIRECTORY_MODE)
            os.fsync(directory_fd)
            result = output_fd
            output_fd = None
            return result
        finally:
            if output_fd is not None:
                os.close(output_fd)
    finally:
        os.close(directory_fd)


def _report_payload(
    *,
    status: str,
    reason: str | None,
    started_at: str,
    finished_at: str | None,
    elapsed_seconds: float,
    exit_code: int | None,
    child_exit_code: int | None,
    termination: str,
    cleanup_complete: bool,
    limits: Limits,
    tracker: ResourceTracker,
    physical_read_source: PhysicalReadSource,
    physical_read_bytes: int,
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "status": status,
        "reason": reason,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "exit_code": exit_code,
        "child_exit_code": child_exit_code,
        "termination": termination,
        "cleanup_complete": cleanup_complete,
        "measurement": {
            "physical_read_source": physical_read_source,
            "logical_read_source": PROC_TREE_INCLUSIVE_SOURCE,
            "process_tree_protocol": "leader-shim-v1",
        },
        "limits": {
            "wall_seconds": limits.wall_seconds,
            "max_tree_rss_bytes": limits.max_rss_bytes,
            "max_physical_read_bytes": limits.max_read_bytes,
            "max_tree_proc_rchar_bytes": limits.max_rchar_bytes,
            "sample_seconds": limits.sample_seconds,
            "term_grace_seconds": limits.term_grace_seconds,
            "max_process_identities": limits.max_process_identities,
        },
        "observed": {
            "peak_tree_rss_bytes": tracker.peak_rss_bytes,
            "physical_read_bytes": physical_read_bytes,
            "tree_proc_rchar_bytes": tracker.total_rchar_bytes,
            "peak_process_count": tracker.peak_process_count,
            "sample_count": tracker.sample_count,
        },
        "privacy": {
            "command_recorded": False,
            "environment_recorded": False,
            "paths_recorded": False,
            "child_output_recorded": False,
        },
    }


def _group_has_live_processes(process_group: int) -> bool:
    return any(reading.state not in {"X", "Z"} for reading in _read_process_group(process_group))


def _process_tree_is_stopped(
    process_group: int,
    child: subprocess.Popen[bytes],
) -> bool:
    return child.poll() is not None and not _group_has_live_processes(process_group)


def _wait_for_process_tree_stop(
    process_group: int,
    child: subprocess.Popen[bytes],
    timeout_seconds: float,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            if _process_tree_is_stopped(process_group, child):
                return True
        except ProcMeasurementUnavailable:
            # Without a readable process-group snapshot, cleanup cannot be claimed complete.
            pass
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.05, remaining))


def _signal_process_tree(
    process_group: int,
    child: subprocess.Popen[bytes],
    watched_signal: signal.Signals,
) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(process_group, watched_signal)
    if child.poll() is None:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            child.send_signal(watched_signal)


def _terminate_process_group(
    process_group: int,
    child: subprocess.Popen[bytes],
    grace_seconds: float,
) -> tuple[str, bool]:
    _signal_process_tree(process_group, child, signal.SIGTERM)
    if _wait_for_process_tree_stop(process_group, child, grace_seconds):
        return "sigterm", True

    _signal_process_tree(process_group, child, signal.SIGKILL)
    cleanup_complete = _wait_for_process_tree_stop(process_group, child, grace_seconds)
    return "sigkill", cleanup_complete


def supervise(
    command: Sequence[str],
    output_directory: Path,
    limits: Limits,
    *,
    physical_read_source: PhysicalReadSource = PROC_TREE_INCLUSIVE_SOURCE,
) -> int:
    """Supervise one command without recording any part of it."""

    if physical_read_source not in {
        PROC_TREE_INCLUSIVE_SOURCE,
        CGROUP_V2_PHYSICAL_READ_SOURCE,
    }:
        raise ValueError("unsupported physical-read source")

    global _requested_signal
    _requested_signal = None
    tracker = ResourceTracker()
    cgroup_tracker: CgroupV2ReadTracker | None = None
    physical_read_bytes = 0
    started_at = datetime.now(UTC).isoformat()
    started_monotonic = time.monotonic()
    status = "supervisor_error"
    reason: str | None = "child_start_failed"
    termination = "none"
    exit_code = EXIT_SUPERVISOR_ERROR
    child_exit_code: int | None = None
    child: subprocess.Popen[bytes] | None = None
    leader_shim_status_descriptor: int | None = None
    leader_shim_acknowledgement_descriptor: int | None = None
    leader_shim_status_buffer = bytearray()
    leader_shim_status: LeaderShimStatus | None = None
    process_group: int | None = None
    cleanup_complete = True
    output_directory_fd: int | None = None
    journal: JsonlJournal | None = None

    watched_signals = [signal.SIGINT, signal.SIGTERM]
    for optional_name in ("SIGHUP", "SIGQUIT"):
        optional_signal = getattr(signal, optional_name, None)
        if optional_signal is not None and optional_signal not in watched_signals:
            watched_signals.append(optional_signal)
    previous_handlers = {
        watched_signal: signal.getsignal(watched_signal) for watched_signal in watched_signals
    }
    for watched_signal in watched_signals:
        signal.signal(watched_signal, _signal_handler)

    try:
        output_directory_fd = _prepare_output_directory(output_directory)
        journal = JsonlJournal(output_directory_fd)
    except Exception:
        if journal is not None:
            with contextlib.suppress(Exception):
                journal.close()
        if output_directory_fd is not None:
            with contextlib.suppress(OSError):
                os.close(output_directory_fd)
        for watched_signal, previous_handler in previous_handlers.items():
            signal.signal(watched_signal, previous_handler)
        raise

    def checkpoint(current_status: str, current_reason: str | None = None) -> None:
        elapsed = time.monotonic() - started_monotonic
        assert output_directory_fd is not None
        _atomic_write_json(
            output_directory_fd,
            _report_payload(
                status=current_status,
                reason=current_reason,
                started_at=started_at,
                finished_at=None,
                elapsed_seconds=elapsed,
                exit_code=None,
                child_exit_code=None,
                termination="none",
                cleanup_complete=child is None,
                limits=limits,
                tracker=tracker,
                physical_read_source=physical_read_source,
                physical_read_bytes=physical_read_bytes,
            ),
        )

    def request_termination(
        requested_status: str,
        requested_reason: str,
        requested_exit_code: int,
    ) -> None:
        nonlocal status, reason, exit_code, termination, cleanup_complete, child_exit_code
        status = requested_status
        reason = requested_reason
        exit_code = requested_exit_code
        if child is None or process_group is None:
            cleanup_complete = True
            return
        assert journal is not None
        elapsed = time.monotonic() - started_monotonic
        termination, cleanup_complete = _terminate_process_group(
            process_group,
            child,
            limits.term_grace_seconds,
        )
        try:
            journal.write(
                {
                    "event": "termination_requested",
                    "elapsed_seconds": round(elapsed, 3),
                    "reason": reason,
                }
            )
        except Exception:
            status = "supervisor_error"
            reason = "evidence_write_failed"
            exit_code = EXIT_SUPERVISOR_ERROR
        raw_child_exit_code = child.poll()
        if (
            raw_child_exit_code is not None
            and child_exit_code is None
            and (leader_shim_status is None or leader_shim_status.workload_started)
        ):
            child_exit_code = _normalize_return_code(raw_child_exit_code)
        if not cleanup_complete:
            status = "supervisor_error"
            reason = "cleanup_incomplete"
            exit_code = EXIT_SUPERVISOR_ERROR

    def interruptible_sample_wait() -> None:
        deadline = time.monotonic() + limits.sample_seconds
        while _requested_signal is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(0.05, remaining))

    try:
        assert journal is not None
        journal.write({"event": "started", "elapsed_seconds": 0.0})
        checkpoint("starting")
        if _requested_signal is not None:
            status = "interrupted"
            reason = "external_signal"
            exit_code = 128 + _requested_signal
        else:
            measurement_ready = True
            if physical_read_source == CGROUP_V2_PHYSICAL_READ_SOURCE:
                try:
                    cgroup_tracker = CgroupV2ReadTracker.start()
                except CgroupMeasurementUnavailable:
                    journal.write(
                        {
                            "event": "resource_measurement_unavailable",
                            "elapsed_seconds": round(time.monotonic() - started_monotonic, 3),
                        }
                    )
                    status = "supervisor_error"
                    reason = "resource_measurement_unavailable"
                    exit_code = EXIT_SUPERVISOR_ERROR
                    measurement_ready = False
            if not measurement_ready:
                pass
            else:
                try:
                    (
                        child,
                        leader_shim_status_descriptor,
                        leader_shim_acknowledgement_descriptor,
                    ) = _spawn_leader_shim(
                        command,
                    )
                except OSError:
                    journal.write({"event": "child_start_failed", "elapsed_seconds": 0.0})
                else:
                    process_group = child.pid
                    cleanup_complete = False
                    journal.write({"event": "child_started", "elapsed_seconds": 0.0})
                    while True:
                        elapsed = time.monotonic() - started_monotonic
                        try:
                            sample = tracker.sample(
                                process_group,
                                leader_pid=child.pid,
                                max_process_identities=limits.max_process_identities,
                            )
                            if cgroup_tracker is None:
                                physical_read_bytes = sample.tree_read_bytes
                            else:
                                physical_read_bytes = cgroup_tracker.sample(
                                    required_member_pid=child.pid
                                )
                        except (ProcMeasurementUnavailable, CgroupMeasurementUnavailable):
                            request_termination(
                                "supervisor_error",
                                "resource_measurement_unavailable",
                                EXIT_SUPERVISOR_ERROR,
                            )
                            break
                        except ProcessIdentityLimitExceeded:
                            request_termination(
                                "limit_exceeded",
                                "process_identity_count",
                                EXIT_PROCESS_IDENTITY_LIMIT,
                            )
                            break

                        try:
                            assert leader_shim_status_descriptor is not None
                            leader_shim_status = _read_leader_shim_status(
                                leader_shim_status_descriptor,
                                leader_shim_status_buffer,
                            )
                        except ProcMeasurementUnavailable:
                            request_termination(
                                "supervisor_error",
                                "resource_measurement_unavailable",
                                EXIT_SUPERVISOR_ERROR,
                            )
                            break
                        polled_child_exit_code = child.poll()
                        if polled_child_exit_code is not None or not sample.leader_observed:
                            request_termination(
                                "supervisor_error",
                                "resource_measurement_unavailable",
                                EXIT_SUPERVISOR_ERROR,
                            )
                            break
                        if leader_shim_status is not None:
                            if leader_shim_status.workload_started:
                                child_exit_code = _normalize_return_code(
                                    leader_shim_status.raw_exit_code
                                )
                            try:
                                # The shim writes its status only after waitpid has folded the
                                # workload counters, and remains live until this sample is accepted.
                                sample = tracker.sample(
                                    process_group,
                                    leader_pid=child.pid,
                                    max_process_identities=limits.max_process_identities,
                                )
                                if cgroup_tracker is None:
                                    physical_read_bytes = sample.tree_read_bytes
                                else:
                                    physical_read_bytes = cgroup_tracker.sample(
                                        required_member_pid=child.pid
                                    )
                            except (ProcMeasurementUnavailable, CgroupMeasurementUnavailable):
                                request_termination(
                                    "supervisor_error",
                                    "resource_measurement_unavailable",
                                    EXIT_SUPERVISOR_ERROR,
                                )
                                break
                            except ProcessIdentityLimitExceeded:
                                request_termination(
                                    "limit_exceeded",
                                    "process_identity_count",
                                    EXIT_PROCESS_IDENTITY_LIMIT,
                                )
                                break

                        if _requested_signal is not None:
                            request_termination(
                                "interrupted",
                                "external_signal",
                                128 + _requested_signal,
                            )
                            break
                        if elapsed > limits.wall_seconds:
                            request_termination("limit_exceeded", "wall_time", EXIT_TIMEOUT)
                            break
                        if sample.tree_rss_bytes > limits.max_rss_bytes:
                            request_termination("limit_exceeded", "tree_rss", EXIT_RSS_LIMIT)
                            break
                        if physical_read_bytes > limits.max_read_bytes:
                            request_termination(
                                "limit_exceeded",
                                "physical_read_bytes",
                                EXIT_READ_LIMIT,
                            )
                            break
                        if sample.tree_rchar_bytes > limits.max_rchar_bytes:
                            request_termination(
                                "limit_exceeded",
                                "proc_rchar_bytes",
                                EXIT_RCHAR_LIMIT,
                            )
                            break

                        journal.write(
                            {
                                "event": "sample",
                                "elapsed_seconds": round(elapsed, 3),
                                "process_count": sample.process_count,
                                "tree_rss_bytes": sample.tree_rss_bytes,
                                "physical_read_source": physical_read_source,
                                "physical_read_bytes": physical_read_bytes,
                                "tree_proc_rchar_bytes": sample.tree_rchar_bytes,
                            }
                        )
                        checkpoint("running")

                        if leader_shim_status is not None:
                            try:
                                assert leader_shim_acknowledgement_descriptor is not None
                                _acknowledge_leader_shim(leader_shim_acknowledgement_descriptor)
                                os.close(leader_shim_acknowledgement_descriptor)
                                leader_shim_acknowledgement_descriptor = None
                                shim_exit_code = _normalize_return_code(
                                    child.wait(timeout=limits.term_grace_seconds)
                                )
                                if shim_exit_code != _normalize_return_code(
                                    leader_shim_status.raw_exit_code
                                ):
                                    raise ProcMeasurementUnavailable
                            except (
                                OSError,
                                ProcMeasurementUnavailable,
                                subprocess.TimeoutExpired,
                            ):
                                request_termination(
                                    "supervisor_error",
                                    "resource_measurement_unavailable",
                                    EXIT_SUPERVISOR_ERROR,
                                )
                                break
                            try:
                                lingering_processes = _group_has_live_processes(process_group)
                            except ProcMeasurementUnavailable:
                                request_termination(
                                    "supervisor_error",
                                    "resource_measurement_unavailable",
                                    EXIT_SUPERVISOR_ERROR,
                                )
                                break
                            if lingering_processes:
                                request_termination(
                                    "supervisor_error",
                                    "lingering_child_group",
                                    EXIT_SUPERVISOR_ERROR,
                                )
                                break
                            cleanup_complete = True
                            if leader_shim_status.workload_started:
                                assert child_exit_code is not None
                                exit_code = child_exit_code
                                status = "passed" if exit_code == 0 else "child_failed"
                                reason = None if exit_code == 0 else "child_exit_nonzero"
                            else:
                                journal.write(
                                    {"event": "child_start_failed", "elapsed_seconds": 0.0}
                                )
                                exit_code = EXIT_SUPERVISOR_ERROR
                                status = "supervisor_error"
                                reason = "child_start_failed"
                            break
                        interruptible_sample_wait()
    except KeyboardInterrupt:
        status = "interrupted"
        reason = "keyboard_interrupt"
        exit_code = 130
    except Exception:
        # Exception text is intentionally omitted because it could contain command arguments or
        # private paths. The enum and exit code are enough for the caller to fail closed.
        status = "supervisor_error"
        reason = "internal_error"
        exit_code = EXIT_SUPERVISOR_ERROR
    finally:
        try:
            if child is not None and process_group is not None and not cleanup_complete:
                try:
                    termination, cleanup_complete = _terminate_process_group(
                        process_group,
                        child,
                        limits.term_grace_seconds,
                    )
                except Exception:
                    cleanup_complete = False
            if child is not None:
                raw_child_exit_code = child.poll()
                if (
                    raw_child_exit_code is not None
                    and child_exit_code is None
                    and (leader_shim_status is None or leader_shim_status.workload_started)
                ):
                    child_exit_code = _normalize_return_code(raw_child_exit_code)
            if not cleanup_complete:
                status = "supervisor_error"
                reason = "cleanup_incomplete"
                exit_code = EXIT_SUPERVISOR_ERROR

            elapsed = time.monotonic() - started_monotonic
            finished_at = datetime.now(UTC).isoformat()
            if journal is not None:
                try:
                    journal.write(
                        {
                            "event": "finished",
                            "elapsed_seconds": round(elapsed, 3),
                            "exit_code": exit_code,
                            "reason": reason,
                            "status": status,
                        }
                    )
                except Exception:
                    status = "supervisor_error"
                    reason = "evidence_write_failed"
                    exit_code = EXIT_SUPERVISOR_ERROR

            final_payload = _report_payload(
                status=status,
                reason=reason,
                started_at=started_at,
                finished_at=finished_at,
                elapsed_seconds=elapsed,
                exit_code=exit_code,
                child_exit_code=child_exit_code,
                termination=termination,
                cleanup_complete=cleanup_complete,
                limits=limits,
                tracker=tracker,
                physical_read_source=physical_read_source,
                physical_read_bytes=physical_read_bytes,
            )
            report_written = False
            if output_directory_fd is not None:
                try:
                    _atomic_write_json(output_directory_fd, final_payload)
                    report_written = True
                except Exception:
                    status = "supervisor_error"
                    reason = "evidence_write_failed"
                    exit_code = EXIT_SUPERVISOR_ERROR
                    if journal is not None:
                        with contextlib.suppress(Exception):
                            journal.write(
                                {
                                    "event": "evidence_write_failed",
                                    "elapsed_seconds": round(elapsed, 3),
                                    "exit_code": exit_code,
                                    "reason": reason,
                                    "status": status,
                                }
                            )
            if not report_written and output_directory_fd is not None:
                # A transient checkpoint failure gets one sanitized retry with the failed status.
                final_payload = _report_payload(
                    status=status,
                    reason=reason,
                    started_at=started_at,
                    finished_at=finished_at,
                    elapsed_seconds=elapsed,
                    exit_code=exit_code,
                    child_exit_code=child_exit_code,
                    termination=termination,
                    cleanup_complete=cleanup_complete,
                    limits=limits,
                    tracker=tracker,
                    physical_read_source=physical_read_source,
                    physical_read_bytes=physical_read_bytes,
                )
                with contextlib.suppress(Exception):
                    _atomic_write_json(output_directory_fd, final_payload)
        except Exception:
            # Cleanup and evidence errors must never expose command arguments or private paths.
            status = "supervisor_error"
            reason = "internal_error"
            exit_code = EXIT_SUPERVISOR_ERROR
        finally:
            if journal is not None:
                with contextlib.suppress(Exception):
                    journal.close()
            if output_directory_fd is not None:
                with contextlib.suppress(OSError):
                    os.close(output_directory_fd)
            if cgroup_tracker is not None:
                with contextlib.suppress(OSError):
                    cgroup_tracker.close()
            for descriptor in (
                leader_shim_status_descriptor,
                leader_shim_acknowledgement_descriptor,
            ):
                if descriptor is not None:
                    with contextlib.suppress(OSError):
                        os.close(descriptor)
            for watched_signal, previous_handler in previous_handlers.items():
                with contextlib.suppress(Exception):
                    signal.signal(watched_signal, previous_handler)

    print(f"acceptance-supervisor: {status} (exit {exit_code})")
    return exit_code


def _bounded_float(minimum: float, maximum: float) -> Callable[[str], float]:
    def parse(value: str) -> float:
        try:
            parsed = float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("must be a finite number") from exc
        if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
            raise argparse.ArgumentTypeError(f"must be between {minimum:g} and {maximum:g}")
        return parsed

    return parse


def _bounded_int(minimum: int, maximum: int) -> Callable[[str], int]:
    def parse(value: str) -> int:
        try:
            parsed = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("must be an integer") from exc
        if not minimum <= parsed <= maximum:
            raise argparse.ArgumentTypeError(f"must be between {minimum} and {maximum}")
        return parsed

    return parse


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an acceptance command with private resource ceilings."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--physical-read-source",
        choices=(PROC_TREE_INCLUSIVE_SOURCE, CGROUP_V2_PHYSICAL_READ_SOURCE),
        default=PROC_TREE_INCLUSIVE_SOURCE,
        help=(
            "physical-read measurement: inclusive /proc group peak, or the supervisor's "
            "delegated cgroup v2 io.stat delta"
        ),
    )
    parser.add_argument(
        "--wall-seconds",
        type=_bounded_float(MIN_WALL_SECONDS, MAX_WALL_SECONDS),
        default=DEFAULT_WALL_SECONDS,
    )
    parser.add_argument(
        "--max-rss-bytes",
        type=_bounded_int(1, MAX_LIMIT_BYTES),
        default=DEFAULT_MAX_RSS_BYTES,
    )
    parser.add_argument(
        "--max-read-bytes",
        type=_bounded_int(1, MAX_LIMIT_BYTES),
        default=DEFAULT_MAX_READ_BYTES,
    )
    parser.add_argument(
        "--max-rchar-bytes",
        type=_bounded_int(1, MAX_LIMIT_BYTES),
        default=DEFAULT_MAX_RCHAR_BYTES,
    )
    parser.add_argument(
        "--sample-seconds",
        type=_bounded_float(MIN_SAMPLE_SECONDS, MAX_SAMPLE_SECONDS),
        default=DEFAULT_SAMPLE_SECONDS,
    )
    parser.add_argument(
        "--term-grace-seconds",
        type=_bounded_float(MIN_TERM_GRACE_SECONDS, MAX_TERM_GRACE_SECONDS),
        default=DEFAULT_TERM_GRACE_SECONDS,
    )
    parser.add_argument(
        "--max-process-identities",
        type=_bounded_int(1, MAX_PROCESS_IDENTITIES),
        default=DEFAULT_MAX_PROCESS_IDENTITIES,
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    if raw_arguments and raw_arguments[0] == "--_leader-shim":
        return _leader_shim_main(raw_arguments[1:])
    arguments = _parser().parse_args(raw_arguments)
    command = cast(list[str], arguments.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        _parser().error("a command is required after --")
    if cast(float, arguments.sample_seconds) > cast(float, arguments.wall_seconds):
        print(f"acceptance-supervisor: invalid resource limits (exit {EXIT_SUPERVISOR_ERROR})")
        return EXIT_SUPERVISOR_ERROR
    limits = Limits(
        wall_seconds=cast(float, arguments.wall_seconds),
        max_rss_bytes=cast(int, arguments.max_rss_bytes),
        max_read_bytes=cast(int, arguments.max_read_bytes),
        max_rchar_bytes=cast(int, arguments.max_rchar_bytes),
        sample_seconds=cast(float, arguments.sample_seconds),
        term_grace_seconds=cast(float, arguments.term_grace_seconds),
        max_process_identities=cast(int, arguments.max_process_identities),
    )
    try:
        return supervise(
            command,
            cast(Path, arguments.output_dir),
            limits,
            physical_read_source=cast(PhysicalReadSource, arguments.physical_read_source),
        )
    except (FileExistsError, OSError, ValueError):
        # Filesystem paths are deliberately omitted from this refusal.
        print(f"acceptance-supervisor: evidence destination refused (exit {EXIT_SUPERVISOR_ERROR})")
        return EXIT_SUPERVISOR_ERROR
    except Exception:
        # Never emit a traceback containing a private output path or child argument.
        print(f"acceptance-supervisor: internal setup failure (exit {EXIT_SUPERVISOR_ERROR})")
        return EXIT_SUPERVISOR_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
