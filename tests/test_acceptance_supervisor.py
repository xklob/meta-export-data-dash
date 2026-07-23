from __future__ import annotations

import importlib.util
import json
import os
import signal
import stat
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

SUPERVISOR = Path(__file__).resolve().parents[1] / "scripts" / "supervise_acceptance.py"


def _load_supervisor_module() -> Any:
    spec = importlib.util.spec_from_file_location("acceptance_supervisor_under_test", SUPERVISOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run_supervisor(
    output_directory: Path,
    child_command: Sequence[str],
    *,
    wall_seconds: float = 5.0,
    max_rss_bytes: int = 1024**3,
    max_read_bytes: int = 1024**3,
    max_rchar_bytes: int = 1024**3,
    max_process_identities: int = 16_384,
    physical_read_source: str = "proc-tree-inclusive",
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SUPERVISOR),
            "--output-dir",
            str(output_directory),
            "--physical-read-source",
            physical_read_source,
            "--wall-seconds",
            str(wall_seconds),
            "--max-rss-bytes",
            str(max_rss_bytes),
            "--max-read-bytes",
            str(max_read_bytes),
            "--max-rchar-bytes",
            str(max_rchar_bytes),
            "--max-process-identities",
            str(max_process_identities),
            "--sample-seconds",
            "0.02",
            "--term-grace-seconds",
            "0.2",
            "--",
            *child_command,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )


def _report(output_directory: Path) -> dict[str, Any]:
    payload = json.loads((output_directory / "report.json").read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _journal(output_directory: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (output_directory / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def _process_is_live(pid: int) -> bool:
    try:
        content = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except FileNotFoundError:
        return False
    closing_parenthesis = content.rfind(")")
    return closing_parenthesis >= 0 and content[closing_parenthesis + 2 :].split()[0] != "Z"


def _proc_reading(
    supervisor: Any,
    pid: int,
    start_ticks: int,
    *,
    read_bytes: int,
    rchar_bytes: int,
    state: str = "S",
) -> Any:
    return supervisor.ProcReading(
        identity=(pid, start_ticks),
        state=state,
        rss_bytes=4096 if state not in {"X", "Z"} else 0,
        read_bytes=read_bytes,
        rchar_bytes=rchar_bytes,
    )


def test_proc_tree_peak_does_not_double_count_waited_child_fold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _load_supervisor_module()
    snapshots = iter(
        [
            [
                _proc_reading(supervisor, 10, 100, read_bytes=10, rchar_bytes=20),
                _proc_reading(supervisor, 11, 101, read_bytes=100, rchar_bytes=200),
            ],
            [
                _proc_reading(supervisor, 10, 100, read_bytes=10, rchar_bytes=20),
                _proc_reading(
                    supervisor,
                    11,
                    101,
                    read_bytes=0,
                    rchar_bytes=0,
                    state="Z",
                ),
            ],
            [_proc_reading(supervisor, 10, 100, read_bytes=110, rchar_bytes=220)],
        ]
    )
    monkeypatch.setattr(supervisor, "_read_process_group", lambda _group: next(snapshots))
    tracker = supervisor.ResourceTracker()

    before_wait = tracker.sample(10, leader_pid=10, max_process_identities=8)
    reap_gap = tracker.sample(10, leader_pid=10, max_process_identities=8)
    after_wait = tracker.sample(10, leader_pid=10, max_process_identities=8)

    assert (before_wait.process_count, reap_gap.process_count, after_wait.process_count) == (
        2,
        1,
        1,
    )
    assert [
        before_wait.tree_read_bytes,
        reap_gap.tree_read_bytes,
        after_wait.tree_read_bytes,
    ] == [110, 110, 110]
    assert [
        before_wait.tree_rchar_bytes,
        reap_gap.tree_rchar_bytes,
        after_wait.tree_rchar_bytes,
    ] == [220, 220, 220]


def test_proc_tree_peak_accumulates_sequential_waited_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _load_supervisor_module()
    snapshots = iter(
        [
            [
                _proc_reading(supervisor, 20, 200, read_bytes=0, rchar_bytes=0),
                _proc_reading(supervisor, 21, 201, read_bytes=100, rchar_bytes=200),
            ],
            [_proc_reading(supervisor, 20, 200, read_bytes=100, rchar_bytes=200)],
            [
                _proc_reading(supervisor, 20, 200, read_bytes=100, rchar_bytes=200),
                _proc_reading(supervisor, 22, 202, read_bytes=80, rchar_bytes=160),
            ],
            [_proc_reading(supervisor, 20, 200, read_bytes=180, rchar_bytes=360)],
        ]
    )
    monkeypatch.setattr(supervisor, "_read_process_group", lambda _group: next(snapshots))
    tracker = supervisor.ResourceTracker()

    samples = [tracker.sample(20, leader_pid=20, max_process_identities=8) for _ in range(4)]

    assert [sample.tree_read_bytes for sample in samples] == [100, 100, 180, 180]
    assert [sample.tree_rchar_bytes for sample in samples] == [200, 200, 360, 360]


@pytest.mark.parametrize("failure", ["counter-decrease", "pid-reuse"])
def test_proc_tree_counter_decrease_or_pid_reuse_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    supervisor = _load_supervisor_module()
    second = (
        _proc_reading(supervisor, 30, 300, read_bytes=99, rchar_bytes=200)
        if failure == "counter-decrease"
        else _proc_reading(supervisor, 30, 301, read_bytes=100, rchar_bytes=200)
    )
    snapshots = iter(
        [
            [_proc_reading(supervisor, 30, 300, read_bytes=100, rchar_bytes=200)],
            [second],
        ]
    )
    monkeypatch.setattr(supervisor, "_read_process_group", lambda _group: next(snapshots))
    tracker = supervisor.ResourceTracker()
    tracker.sample(30, leader_pid=30, max_process_identities=8)

    with pytest.raises(supervisor.ProcMeasurementUnavailable):
        tracker.sample(30, leader_pid=30, max_process_identities=8)


def test_proc_group_io_denial_is_tolerated_only_after_confirmed_zombie_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _load_supervisor_module()

    class FakeEntry:
        name = "40"

    class FakeProcScan:
        def __iter__(self) -> Any:
            return iter([FakeEntry()])

        def __enter__(self) -> list[FakeEntry]:
            return [FakeEntry()]

        def __exit__(self, *_args: object) -> None:
            return None

    stats = iter([(40, 400, "R"), (40, 400, "Z")])
    monkeypatch.setattr(supervisor.os, "scandir", lambda _path: FakeProcScan())
    monkeypatch.setattr(supervisor, "_read_proc_stat", lambda _pid: next(stats))

    def denied(_pid: int) -> None:
        raise supervisor.ProcMeasurementUnavailable

    monkeypatch.setattr(supervisor, "_read_io_bytes", denied)

    assert supervisor._read_process_group(40) == [
        supervisor.ProcReading(
            identity=(40, 400),
            state="Z",
            rss_bytes=0,
            read_bytes=0,
            rchar_bytes=0,
        )
    ]


def test_proc_group_io_denial_for_still_live_identity_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _load_supervisor_module()

    class FakeEntry:
        name = "41"

    class FakeProcScan:
        def __iter__(self) -> Any:
            return iter([FakeEntry()])

        def __enter__(self) -> list[FakeEntry]:
            return [FakeEntry()]

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(supervisor.os, "scandir", lambda _path: FakeProcScan())
    monkeypatch.setattr(supervisor, "_read_proc_stat", lambda _pid: (41, 410, "R"))

    def denied(_pid: int) -> None:
        raise supervisor.ProcMeasurementUnavailable

    monkeypatch.setattr(supervisor, "_read_io_bytes", denied)

    with pytest.raises(supervisor.ProcMeasurementUnavailable):
        supervisor._read_process_group(41)


def test_proc_group_transient_io_denial_recovers_same_live_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _load_supervisor_module()

    class FakeEntry:
        name = "42"

    class FakeProcScan:
        def __iter__(self) -> Any:
            return iter([FakeEntry()])

        def __enter__(self) -> list[FakeEntry]:
            return [FakeEntry()]

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(supervisor.os, "scandir", lambda _path: FakeProcScan())
    monkeypatch.setattr(supervisor, "_read_proc_stat", lambda _pid: (42, 420, "R"))
    calls = 0

    def transient(_pid: int) -> tuple[int, int]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise supervisor.ProcMeasurementUnavailable
        return (12, 34)

    monkeypatch.setattr(supervisor, "_read_io_bytes", transient)
    monkeypatch.setattr(supervisor, "_read_rss_bytes", lambda _pid: 4096)

    assert supervisor._read_process_group(42) == [
        supervisor.ProcReading(
            identity=(42, 420),
            state="R",
            rss_bytes=4096,
            read_bytes=12,
            rchar_bytes=34,
        )
    ]


@pytest.mark.parametrize("missing_counter", ["io", "rss"])
def test_proc_group_missing_counter_for_still_live_identity_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    missing_counter: str,
) -> None:
    supervisor = _load_supervisor_module()

    class FakeEntry:
        name = "43"

    class FakeProcScan:
        def __iter__(self) -> Any:
            return iter([FakeEntry()])

        def __enter__(self) -> list[FakeEntry]:
            return [FakeEntry()]

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(supervisor.os, "scandir", lambda _path: FakeProcScan())
    monkeypatch.setattr(supervisor, "_read_proc_stat", lambda _pid: (43, 430, "R"))
    monkeypatch.setattr(
        supervisor,
        "_read_io_bytes",
        (lambda _pid: None) if missing_counter == "io" else (lambda _pid: (12, 34)),
    )
    monkeypatch.setattr(
        supervisor,
        "_read_rss_bytes",
        (lambda _pid: None) if missing_counter == "rss" else (lambda _pid: 4096),
    )

    with pytest.raises(supervisor.ProcMeasurementUnavailable):
        supervisor._read_process_group(43)


def test_proc_group_missing_counter_accepts_confirmed_gone_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _load_supervisor_module()

    class FakeEntry:
        name = "44"

    class FakeProcScan:
        def __iter__(self) -> Any:
            return iter([FakeEntry()])

        def __enter__(self) -> list[FakeEntry]:
            return [FakeEntry()]

        def __exit__(self, *_args: object) -> None:
            return None

    stats = iter([(44, 440, "R"), None])
    monkeypatch.setattr(supervisor.os, "scandir", lambda _path: FakeProcScan())
    monkeypatch.setattr(supervisor, "_read_proc_stat", lambda _pid: next(stats))
    monkeypatch.setattr(supervisor, "_read_io_bytes", lambda _pid: None)
    monkeypatch.setattr(supervisor, "_read_rss_bytes", lambda _pid: 4096)

    assert supervisor._read_process_group(44) == []


def _fake_cgroup_files(
    supervisor: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    io_stat: str,
) -> Path:
    proc_self_cgroup = tmp_path / "self.cgroup"
    proc_self_cgroup.write_text("0::/unit.slice/test.scope\n", encoding="ascii")
    cgroup_root = tmp_path / "cgroup"
    cgroup_directory = cgroup_root / "unit.slice" / "test.scope"
    cgroup_directory.mkdir(parents=True)
    io_stat_path = cgroup_directory / "io.stat"
    io_stat_path.write_text(io_stat, encoding="ascii")
    monkeypatch.setattr(supervisor, "PROC_SELF_CGROUP_PATH", proc_self_cgroup)
    monkeypatch.setattr(supervisor, "CGROUP_V2_ROOT", cgroup_root)
    return io_stat_path


def test_cgroup_v2_io_stat_baseline_delta_and_multiple_devices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _load_supervisor_module()
    io_stat = _fake_cgroup_files(
        supervisor,
        tmp_path,
        monkeypatch,
        "8:0 rbytes=100 wbytes=9\n8:16 rbytes=200 rios=2\n7:0 rbytes=0\n",
    )
    tracker = supervisor.CgroupV2ReadTracker.start()
    try:
        assert tracker.sample() == 0
        io_stat.write_text(
            "8:0 rbytes=150 wbytes=9\n8:16 rbytes=210 rios=2\n7:0 rbytes=0\n8:32 rbytes=20\n",
            encoding="ascii",
        )
        assert tracker.sample() == 80
    finally:
        tracker.close()


def test_cgroup_v2_required_workload_member_must_share_supervisor_cgroup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _load_supervisor_module()
    _fake_cgroup_files(supervisor, tmp_path, monkeypatch, "8:0 rbytes=0\n")
    tracker = supervisor.CgroupV2ReadTracker.start()
    monkeypatch.setattr(
        supervisor,
        "_process_unified_cgroup_components",
        lambda _pid: ("escaped.slice",),
    )
    try:
        with pytest.raises(supervisor.CgroupMeasurementUnavailable):
            tracker.sample(required_member_pid=1234)
    finally:
        tracker.close()


@pytest.mark.parametrize(
    "replacement",
    [
        "8:0 rbytes=99\n",
        "8:0 rbytes=100\n8:0 rbytes=100\n",
        "8:0 rbytes=-1\n",
        f"8:0 rbytes={'9' * 21}\n",
        "4294967296:0 rbytes=1\n",
        "not-a-device rbytes=100\n",
        "8:0 rbytes=100 rbytes=101\n",
        "8:0 wbytes=100\n",
        "",
    ],
)
def test_cgroup_v2_counter_decrease_malformed_or_missing_device_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement: str,
) -> None:
    supervisor = _load_supervisor_module()
    io_stat = _fake_cgroup_files(
        supervisor,
        tmp_path,
        monkeypatch,
        "8:0 rbytes=100\n",
    )
    tracker = supervisor.CgroupV2ReadTracker.start()
    try:
        io_stat.write_text(replacement, encoding="ascii")
        with pytest.raises(supervisor.CgroupMeasurementUnavailable):
            tracker.sample()
    finally:
        tracker.close()


@pytest.mark.parametrize(
    "membership",
    [
        b"2:cpu:/legacy\n",
        b"0:cpu:/mixed\n",
        b"0::/safe/../escape\n",
        b"0::/safe//escape\n",
        b"0::relative\n",
        b"0::/one\n0::/two\n",
        b"malformed\n",
    ],
)
def test_cgroup_v2_membership_parser_rejects_nonunified_or_unsafe_paths(
    membership: bytes,
) -> None:
    supervisor = _load_supervisor_module()

    with pytest.raises(supervisor.CgroupMeasurementUnavailable):
        supervisor._parse_unified_cgroup_components(membership)


def test_cgroup_v2_missing_rbytes_at_baseline_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _load_supervisor_module()
    _fake_cgroup_files(supervisor, tmp_path, monkeypatch, "8:0 wbytes=100\n")

    with pytest.raises(supervisor.CgroupMeasurementUnavailable):
        supervisor.CgroupV2ReadTracker.start()


def test_cgroup_v2_missing_io_stat_fails_before_child_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _load_supervisor_module()
    io_stat = _fake_cgroup_files(supervisor, tmp_path, monkeypatch, "8:0 rbytes=0\n")
    io_stat.unlink()
    marker = tmp_path / "child-ran"
    limits = supervisor.Limits(
        wall_seconds=5.0,
        max_rss_bytes=1024**3,
        max_read_bytes=1024**3,
        max_rchar_bytes=1024**3,
        sample_seconds=0.02,
        term_grace_seconds=0.2,
        max_process_identities=16_384,
    )

    exit_code = supervisor.supervise(
        [
            sys.executable,
            "-c",
            "import pathlib, sys; pathlib.Path(sys.argv[1]).write_text('ran')",
            str(marker),
        ],
        tmp_path / "missing-io-stat",
        limits,
        physical_read_source="cgroup-v2",
    )

    assert exit_code == 127
    assert not marker.exists()
    report = _report(tmp_path / "missing-io-stat")
    assert report["reason"] == "resource_measurement_unavailable"
    assert report["measurement"]["physical_read_source"] == "cgroup-v2"


def test_cgroup_v2_oversized_or_symlinked_control_files_are_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _load_supervisor_module()
    io_stat = _fake_cgroup_files(supervisor, tmp_path, monkeypatch, "8:0 rbytes=0\n")
    io_stat.write_bytes(b"x" * (supervisor.MAX_CGROUP_IO_STAT_BYTES + 1))
    with pytest.raises(supervisor.CgroupMeasurementUnavailable):
        supervisor.CgroupV2ReadTracker.start()

    target = tmp_path / "outside-io.stat"
    target.write_text("8:0 rbytes=0\n", encoding="ascii")
    io_stat.unlink()
    io_stat.symlink_to(target)
    with pytest.raises(supervisor.CgroupMeasurementUnavailable):
        supervisor.CgroupV2ReadTracker.start()


def test_cgroup_v2_symlinked_proc_membership_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _load_supervisor_module()
    target = tmp_path / "membership-target"
    target.write_text("0::/\n", encoding="ascii")
    membership = tmp_path / "membership-link"
    membership.symlink_to(target)
    monkeypatch.setattr(supervisor, "PROC_SELF_CGROUP_PATH", membership)

    with pytest.raises(supervisor.CgroupMeasurementUnavailable):
        supervisor._self_unified_cgroup_components()


def test_cgroup_v2_physical_read_limit_is_enforced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _load_supervisor_module()

    class FakeCgroupTracker:
        @classmethod
        def start(cls) -> FakeCgroupTracker:
            return cls()

        def sample(self, *, required_member_pid: int | None = None) -> int:
            assert required_member_pid is not None
            return 2048

        def close(self) -> None:
            return None

    monkeypatch.setattr(supervisor, "CgroupV2ReadTracker", FakeCgroupTracker)
    limits = supervisor.Limits(
        wall_seconds=5.0,
        max_rss_bytes=1024**3,
        max_read_bytes=1024,
        max_rchar_bytes=1024**3,
        sample_seconds=0.02,
        term_grace_seconds=0.2,
        max_process_identities=16_384,
    )

    exit_code = supervisor.supervise(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        tmp_path / "cgroup-limit",
        limits,
        physical_read_source="cgroup-v2",
    )

    assert exit_code == 126
    report = _report(tmp_path / "cgroup-limit")
    assert report["reason"] == "physical_read_bytes"
    assert report["observed"]["physical_read_bytes"] == 2048
    assert report["measurement"]["physical_read_source"] == "cgroup-v2"


def test_real_waited_child_io_fold_does_not_double_count_rchar(tmp_path: Path) -> None:
    payload_bytes = 32 * 1024**2
    fixture = tmp_path / "logical-read-fixture.bin"
    with fixture.open("wb") as handle:
        handle.truncate(payload_bytes)
    child_source = """
import sys
import time

with open(sys.argv[1], 'rb', buffering=0) as source:
    while source.read(65_536):
        pass
time.sleep(0.3)
"""
    parent_source = """
import subprocess
import sys
import time

child = subprocess.Popen([sys.executable, '-c', sys.argv[2], sys.argv[1]])
child.wait()
time.sleep(0.3)
"""
    output = tmp_path / "real-child-fold"

    result = _run_supervisor(
        output,
        [sys.executable, "-c", parent_source, str(fixture), child_source],
        max_rchar_bytes=4 * payload_bytes,
    )

    assert result.returncode == 0
    report = _report(output)
    observed = report["observed"]["tree_proc_rchar_bytes"]
    assert payload_bytes <= observed < int(payload_bytes * 1.75)
    sample_counts = [
        event["process_count"] for event in _journal(output) if event["event"] == "sample"
    ]
    first_two_process_sample = sample_counts.index(2)
    assert 1 in sample_counts[first_two_process_sample + 1 :]


@pytest.mark.parametrize("attempt", range(8))
def test_immediate_exit_leader_final_rchar_is_enforced(tmp_path: Path, attempt: int) -> None:
    output = tmp_path / f"immediate-rchar-{attempt}"
    child_source = """
with open('/dev/zero', 'rb', buffering=0) as source:
    remaining = 32 * 1024 * 1024
    while remaining:
        remaining -= len(source.read(min(65_536, remaining)))
"""

    result = _run_supervisor(
        output,
        [sys.executable, "-c", child_source],
        max_rchar_bytes=1024**2,
    )

    assert result.returncode == 123
    report = _report(output)
    assert report["reason"] == "proc_rchar_bytes"
    assert report["observed"]["tree_proc_rchar_bytes"] > 1024**2


def test_immediate_exit_leader_fold_is_present_in_final_report(tmp_path: Path) -> None:
    output = tmp_path / "immediate-rchar-final-report"
    child_source = """
with open('/dev/zero', 'rb', buffering=0) as source:
    remaining = 32 * 1024 * 1024
    while remaining:
        remaining -= len(source.read(min(65_536, remaining)))
"""

    result = _run_supervisor(
        output,
        [sys.executable, "-c", child_source],
        max_rchar_bytes=1024**3,
    )

    assert result.returncode == 0
    report = _report(output)
    assert report["observed"]["tree_proc_rchar_bytes"] >= 32 * 1024**2


@pytest.mark.skipif(not hasattr(os, "posix_fadvise"), reason="requires POSIX cache advice")
def test_immediate_exit_leader_final_physical_reads_are_enforced(tmp_path: Path) -> None:
    fixture = tmp_path / "immediate-physical-fixture.bin"
    fixture.write_bytes(os.urandom(8 * 1024**2))
    with fixture.open("rb") as handle:
        os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    child_source = """
import os
import sys

descriptor = os.open(sys.argv[1], os.O_RDONLY)
try:
    while os.read(descriptor, 65_536):
        pass
finally:
    os.close(descriptor)
"""
    output = tmp_path / "immediate-physical"

    result = _run_supervisor(
        output,
        [sys.executable, "-c", child_source, str(fixture)],
        max_read_bytes=1024,
        max_rchar_bytes=1024**3,
    )

    report = _report(output)
    if report["observed"]["physical_read_bytes"] == 0:
        pytest.skip("this filesystem did not expose physical reads through /proc")
    assert result.returncode == 126
    assert report["reason"] == "physical_read_bytes"
    assert report["child_exit_code"] == 0


def test_pass_exit_propagation_privacy_and_permissions(tmp_path: Path) -> None:
    secret = "private-token-that-must-not-be-recorded"
    private_argument = str(tmp_path / "private archive" / "message.json")

    passed_output = tmp_path / "passed"
    passed = _run_supervisor(
        passed_output,
        [sys.executable, "-c", "raise SystemExit(0)", secret, private_argument],
    )
    assert passed.returncode == 0
    assert passed.stderr == ""
    passed_report = _report(passed_output)
    assert passed_report["status"] == "passed"
    assert passed_report["exit_code"] == 0
    assert passed_report["cleanup_complete"] is True
    assert passed_report["schema_version"] == 2
    assert passed_report["measurement"] == {
        "logical_read_source": "proc-tree-inclusive",
        "physical_read_source": "proc-tree-inclusive",
        "process_tree_protocol": "leader-shim-v1",
    }
    assert passed_report["privacy"] == {
        "child_output_recorded": False,
        "command_recorded": False,
        "environment_recorded": False,
        "paths_recorded": False,
    }

    serialized_artifacts = (passed_output / "report.json").read_text(encoding="utf-8") + (
        passed_output / "progress.jsonl"
    ).read_text(encoding="utf-8")
    assert secret not in serialized_artifacts
    assert private_argument not in serialized_artifacts
    assert secret not in passed.stdout
    assert private_argument not in passed.stdout
    assert stat.S_IMODE(passed_output.stat().st_mode) == 0o700
    assert stat.S_IMODE((passed_output / "report.json").stat().st_mode) == 0o600
    assert stat.S_IMODE((passed_output / "progress.jsonl").stat().st_mode) == 0o600
    assert _journal(passed_output)[-1]["event"] == "finished"

    failed_output = tmp_path / "failed"
    failed = _run_supervisor(
        failed_output,
        [sys.executable, "-c", "raise SystemExit(7)"],
    )
    assert failed.returncode == 7
    failed_report = _report(failed_output)
    assert failed_report["status"] == "child_failed"
    assert failed_report["child_exit_code"] == 7
    assert failed_report["exit_code"] == 7
    assert failed_report["cleanup_complete"] is True

    missing_command = str(tmp_path / "private missing executable")
    start_failure_output = tmp_path / "start-failure"
    start_failure = _run_supervisor(start_failure_output, [missing_command, secret])
    assert start_failure.returncode == 127
    start_failure_report = _report(start_failure_output)
    assert start_failure_report["status"] == "supervisor_error"
    assert start_failure_report["reason"] == "child_start_failed"
    assert start_failure_report["child_exit_code"] is None
    assert start_failure_report["cleanup_complete"] is True
    start_failure_artifacts = (start_failure_output / "report.json").read_text(encoding="utf-8") + (
        start_failure_output / "progress.jsonl"
    ).read_text(encoding="utf-8")
    assert missing_command not in start_failure_artifacts
    assert secret not in start_failure_artifacts


@pytest.mark.parametrize("existing_name", ["report.json", "progress.jsonl"])
def test_existing_evidence_is_never_overwritten(tmp_path: Path, existing_name: str) -> None:
    output = tmp_path / "existing-evidence"
    output.mkdir()
    existing = output / existing_name
    sentinel = b"prior evidence must survive byte-for-byte\n"
    existing.write_bytes(sentinel)
    child_marker = tmp_path / "child-ran"

    child_source = "import pathlib, sys; pathlib.Path(sys.argv[1]).write_text('ran')"
    result = _run_supervisor(
        output,
        [sys.executable, "-c", child_source, str(child_marker)],
    )

    assert result.returncode == 127
    assert result.stderr == ""
    assert "evidence destination refused" in result.stdout
    assert existing.read_bytes() == sentinel
    assert not child_marker.exists()
    other_name = "progress.jsonl" if existing_name == "report.json" else "report.json"
    assert not (output / other_name).exists()


def test_existing_empty_directory_is_refused_atomically(tmp_path: Path) -> None:
    output = tmp_path / "already-exists"
    output.mkdir(mode=0o700)
    child_marker = tmp_path / "child-ran"
    child_source = "import pathlib, sys; pathlib.Path(sys.argv[1]).write_text('ran')"

    result = _run_supervisor(
        output,
        [sys.executable, "-c", child_source, str(child_marker)],
    )

    assert result.returncode == 127
    assert result.stderr == ""
    assert "evidence destination refused" in result.stdout
    assert not child_marker.exists()
    assert list(output.iterdir()) == []


def test_output_parent_symlink_is_refused_without_touching_target(tmp_path: Path) -> None:
    private_component = "private-parent-that-must-not-be-printed"
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / private_component
    alias.symlink_to(target, target_is_directory=True)
    output = alias / "evidence"
    child_marker = tmp_path / "child-ran"
    child_source = "import pathlib, sys; pathlib.Path(sys.argv[1]).write_text('ran')"

    result = _run_supervisor(
        output,
        [sys.executable, "-c", child_source, str(child_marker)],
    )

    assert result.returncode == 127
    assert result.stderr == ""
    assert private_component not in result.stdout
    assert not child_marker.exists()
    assert not (target / "evidence").exists()


def test_output_permission_error_is_sanitized_without_a_traceback(tmp_path: Path) -> None:
    private_component = "private-unreadable-parent"
    blocked = tmp_path / private_component
    blocked.mkdir(mode=0o700)
    blocked.chmod(0)
    try:
        result = _run_supervisor(
            blocked / "evidence",
            [sys.executable, "-c", "raise SystemExit(0)"],
        )
    finally:
        blocked.chmod(0o700)

    assert result.returncode == 127
    assert result.stderr == ""
    assert private_component not in result.stdout
    assert "evidence destination refused" in result.stdout


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--wall-seconds", "nan"),
        ("--wall-seconds", "inf"),
        ("--sample-seconds", "0.001"),
        ("--term-grace-seconds", "61"),
        ("--max-rss-bytes", str(2**63)),
        ("--max-process-identities", "1000001"),
    ],
)
def test_nonfinite_and_out_of_range_limits_are_rejected_without_running_child(
    tmp_path: Path,
    option: str,
    value: str,
) -> None:
    output = tmp_path / f"invalid-{option.removeprefix('--')}"
    marker = tmp_path / "child-ran"
    secret = "private-child-argument"
    child_source = "import pathlib, sys; pathlib.Path(sys.argv[1]).write_text('ran')"
    result = subprocess.run(
        [
            sys.executable,
            str(SUPERVISOR),
            "--output-dir",
            str(output),
            option,
            value,
            "--",
            sys.executable,
            "-c",
            child_source,
            str(marker),
            secret,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 2
    assert not marker.exists()
    assert not output.exists()
    assert secret not in result.stdout + result.stderr


def test_sample_interval_may_not_exceed_wall_limit(tmp_path: Path) -> None:
    output = tmp_path / "invalid-relation"
    marker = tmp_path / "child-ran"
    result = subprocess.run(
        [
            sys.executable,
            str(SUPERVISOR),
            "--output-dir",
            str(output),
            "--wall-seconds",
            "0.05",
            "--sample-seconds",
            "0.1",
            "--",
            sys.executable,
            "-c",
            "import pathlib, sys; pathlib.Path(sys.argv[1]).write_text('ran')",
            str(marker),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 127
    assert result.stderr == ""
    assert "invalid resource limits" in result.stdout
    assert not marker.exists()
    assert not output.exists()


def test_timeout_kills_only_the_spawned_process_group(tmp_path: Path) -> None:
    child_pid_file = tmp_path / "grandchild.pid"
    child_source = """
import pathlib
import subprocess
import sys
import time

child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
pathlib.Path(sys.argv[1]).write_text(str(child.pid), encoding="ascii")
time.sleep(30)
"""
    output = tmp_path / "timeout"
    result = _run_supervisor(
        output,
        [sys.executable, "-c", child_source, str(child_pid_file)],
        wall_seconds=0.15,
    )
    assert result.returncode == 124
    report = _report(output)
    assert report["status"] == "limit_exceeded"
    assert report["reason"] == "wall_time"
    assert report["termination"] in {"sigterm", "sigkill"}
    assert report["cleanup_complete"] is True

    grandchild_pid = int(child_pid_file.read_text(encoding="ascii"))
    deadline = time.monotonic() + 2.0
    while _process_is_live(grandchild_pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not _process_is_live(grandchild_pid)


def test_sigkill_cleanup_is_confirmed_when_child_ignores_sigterm(tmp_path: Path) -> None:
    output = tmp_path / "sigkill"
    child_source = """
import signal
import time

signal.signal(signal.SIGTERM, signal.SIG_IGN)
time.sleep(30)
"""
    result = _run_supervisor(
        output,
        [sys.executable, "-c", child_source],
        wall_seconds=0.15,
    )

    assert result.returncode == 124
    report = _report(output)
    assert report["reason"] == "wall_time"
    assert report["termination"] == "sigkill"
    assert report["cleanup_complete"] is True


@pytest.mark.parametrize("watched_signal", [signal.SIGHUP, signal.SIGQUIT])
def test_terminal_signals_stop_child_and_leave_final_evidence(
    tmp_path: Path,
    watched_signal: signal.Signals,
) -> None:
    output = tmp_path / f"signal-{watched_signal.value}"
    child_pid_file = tmp_path / f"child-{watched_signal.value}.pid"
    child_source = """
import os
import pathlib
import sys
import time

pathlib.Path(sys.argv[1]).write_text(str(os.getpid()), encoding='ascii')
time.sleep(30)
"""
    supervisor = subprocess.Popen(
        [
            sys.executable,
            str(SUPERVISOR),
            "--output-dir",
            str(output),
            "--wall-seconds",
            "5",
            "--sample-seconds",
            "0.02",
            "--term-grace-seconds",
            "0.2",
            "--",
            sys.executable,
            "-c",
            child_source,
            str(child_pid_file),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 3.0
    while not child_pid_file.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert child_pid_file.exists()
    child_pid = int(child_pid_file.read_text(encoding="ascii"))

    os.kill(supervisor.pid, watched_signal)
    stdout, stderr = supervisor.communicate(timeout=5)

    assert supervisor.returncode == 128 + watched_signal.value
    assert stderr == ""
    assert "interrupted" in stdout
    report = _report(output)
    assert report["status"] == "interrupted"
    assert report["reason"] == "external_signal"
    assert report["cleanup_complete"] is True
    assert not _process_is_live(child_pid)


def test_distinct_process_identity_ceiling_fails_closed(tmp_path: Path) -> None:
    output = tmp_path / "process-identities"
    child_source = """
import subprocess
import sys
import time

children = [
    subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    for _ in range(2)
]
time.sleep(30)
"""
    result = _run_supervisor(
        output,
        [sys.executable, "-c", child_source],
        max_process_identities=1,
    )

    assert result.returncode == 122
    report = _report(output)
    assert report["status"] == "limit_exceeded"
    assert report["reason"] == "process_identity_count"
    assert report["cleanup_complete"] is True


def test_unavailable_proc_measurement_fails_closed_and_cleans_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _load_supervisor_module()

    def unavailable(*_args: object, **_kwargs: object) -> None:
        raise supervisor.ProcMeasurementUnavailable

    monkeypatch.setattr(supervisor.ResourceTracker, "sample", unavailable)
    output = tmp_path / "measurement-unavailable"
    limits = supervisor.Limits(
        wall_seconds=5.0,
        max_rss_bytes=1024**3,
        max_read_bytes=1024**3,
        max_rchar_bytes=1024**3,
        sample_seconds=0.02,
        term_grace_seconds=0.2,
        max_process_identities=16_384,
    )

    exit_code = supervisor.supervise(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        output,
        limits,
    )

    assert exit_code == 127
    report = _report(output)
    assert report["status"] == "supervisor_error"
    assert report["reason"] == "resource_measurement_unavailable"
    assert report["cleanup_complete"] is True


def test_missing_live_leader_measurement_fails_closed_and_cleans_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _load_supervisor_module()

    def missing_leader(*_args: object, **_kwargs: object) -> Any:
        return supervisor.ResourceSample(
            process_count=0,
            tree_rss_bytes=0,
            tree_read_bytes=0,
            tree_rchar_bytes=0,
            leader_observed=False,
        )

    monkeypatch.setattr(supervisor.ResourceTracker, "sample", missing_leader)
    limits = supervisor.Limits(
        wall_seconds=5.0,
        max_rss_bytes=1024**3,
        max_read_bytes=1024**3,
        max_rchar_bytes=1024**3,
        sample_seconds=0.02,
        term_grace_seconds=0.2,
        max_process_identities=16_384,
    )

    exit_code = supervisor.supervise(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        tmp_path / "missing-live-leader",
        limits,
    )

    assert exit_code == 127
    report = _report(tmp_path / "missing-live-leader")
    assert report["reason"] == "resource_measurement_unavailable"
    assert report["cleanup_complete"] is True


def test_unconfirmed_cleanup_overrides_original_limit_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _load_supervisor_module()
    monkeypatch.setattr(supervisor, "_wait_for_process_tree_stop", lambda *_args: False)
    output = tmp_path / "cleanup-unconfirmed"
    limits = supervisor.Limits(
        wall_seconds=0.05,
        max_rss_bytes=1024**3,
        max_read_bytes=1024**3,
        max_rchar_bytes=1024**3,
        sample_seconds=0.01,
        term_grace_seconds=0.05,
        max_process_identities=16_384,
    )

    exit_code = supervisor.supervise(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        output,
        limits,
    )

    assert exit_code == 127
    report = _report(output)
    assert report["status"] == "supervisor_error"
    assert report["reason"] == "cleanup_incomplete"
    assert report["cleanup_complete"] is False


def test_final_report_failure_is_retried_with_sanitized_failed_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _load_supervisor_module()
    original_write = supervisor._atomic_write_json
    failed_once = False

    def fail_first_final_write(directory_fd: int, payload: dict[str, object]) -> None:
        nonlocal failed_once
        if payload.get("finished_at_utc") is not None and not failed_once:
            failed_once = True
            raise OSError("private path must not escape")
        original_write(directory_fd, payload)

    monkeypatch.setattr(supervisor, "_atomic_write_json", fail_first_final_write)
    output = tmp_path / "final-write-retry"
    limits = supervisor.Limits(
        wall_seconds=5.0,
        max_rss_bytes=1024**3,
        max_read_bytes=1024**3,
        max_rchar_bytes=1024**3,
        sample_seconds=0.02,
        term_grace_seconds=0.2,
        max_process_identities=16_384,
    )

    exit_code = supervisor.supervise(
        [sys.executable, "-c", "raise SystemExit(0)"],
        output,
        limits,
    )

    assert failed_once is True
    assert exit_code == 127
    report = _report(output)
    assert report["status"] == "supervisor_error"
    assert report["reason"] == "evidence_write_failed"
    assert report["cleanup_complete"] is True
    assert _journal(output)[-1]["event"] == "evidence_write_failed"


def test_rss_ceiling_leaves_a_final_report(tmp_path: Path) -> None:
    output = tmp_path / "rss"
    result = _run_supervisor(
        output,
        [sys.executable, "-c", "import time; payload = bytearray(8_000_000); time.sleep(5)"],
        max_rss_bytes=1024,
    )
    assert result.returncode == 125
    report = _report(output)
    assert report["status"] == "limit_exceeded"
    assert report["reason"] == "tree_rss"
    assert report["observed"]["peak_tree_rss_bytes"] > 1024
    assert report["cleanup_complete"] is True


def test_logical_read_ceiling_uses_proc_rchar(tmp_path: Path) -> None:
    output = tmp_path / "rchar"
    child_source = """
import os
import time

with open('/dev/zero', 'rb', buffering=0) as source:
    for _ in range(32):
        source.read(65_536)
time.sleep(5)
"""
    result = _run_supervisor(
        output,
        [sys.executable, "-c", child_source],
        max_rchar_bytes=1024,
    )
    assert result.returncode == 123
    report = _report(output)
    assert report["reason"] == "proc_rchar_bytes"
    assert report["observed"]["tree_proc_rchar_bytes"] > 1024


@pytest.mark.skipif(not hasattr(os, "posix_fadvise"), reason="requires POSIX cache advice")
def test_physical_read_ceiling_uses_proc_tree_inclusive_bytes(tmp_path: Path) -> None:
    fixture = tmp_path / "read-fixture.bin"
    fixture.write_bytes(os.urandom(4 * 1024**2))
    with fixture.open("rb") as handle:
        os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)

    output = tmp_path / "physical-read"
    child_source = """
import os
import sys
import time

descriptor = os.open(sys.argv[1], os.O_RDONLY)
try:
    while True:
        os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
        os.lseek(descriptor, 0, os.SEEK_SET)
        while os.read(descriptor, 65_536):
            pass
finally:
    os.close(descriptor)
time.sleep(5)
"""
    result = _run_supervisor(
        output,
        [sys.executable, "-c", child_source, str(fixture)],
        wall_seconds=1.0,
        max_read_bytes=1024,
        max_rchar_bytes=1024**3,
    )
    report = _report(output)
    if report["observed"]["physical_read_bytes"] == 0:
        pytest.skip("this filesystem did not expose physical reads through /proc")
    assert result.returncode == 126
    assert report["reason"] == "physical_read_bytes"
    assert report["measurement"]["physical_read_source"] == "proc-tree-inclusive"
    assert report["observed"]["physical_read_bytes"] > 1024


@pytest.mark.skipif(not hasattr(os, "posix_fadvise"), reason="requires POSIX cache advice")
def test_live_runaway_child_trips_proc_tree_read_limit_before_parent_exit(tmp_path: Path) -> None:
    fixture = tmp_path / "nested-read-fixture.bin"
    fixture.write_bytes(os.urandom(4 * 1024**2))
    with fixture.open("rb") as handle:
        os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    child_source = """
import os
import sys

descriptor = os.open(sys.argv[1], os.O_RDONLY)
while True:
    os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
    os.lseek(descriptor, 0, os.SEEK_SET)
    while os.read(descriptor, 65_536):
        pass
"""
    parent_source = """
import subprocess
import sys
import time

subprocess.Popen([sys.executable, '-c', sys.argv[2], sys.argv[1]])
time.sleep(30)
"""
    output = tmp_path / "nested-physical-read"

    result = _run_supervisor(
        output,
        [sys.executable, "-c", parent_source, str(fixture), child_source],
        wall_seconds=1.5,
        max_read_bytes=1024,
        max_rchar_bytes=1024**3,
    )

    report = _report(output)
    if report["observed"]["physical_read_bytes"] == 0:
        pytest.skip("this filesystem did not expose physical reads through /proc")
    assert result.returncode == 126
    assert report["reason"] == "physical_read_bytes"
    assert report["termination"] in {"sigterm", "sigkill"}
    assert report["cleanup_complete"] is True
    assert report["observed"]["peak_process_count"] >= 2
