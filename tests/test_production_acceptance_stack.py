from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import time
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


def _stack_module() -> ModuleType:
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_production_acceptance.py"
    spec = importlib.util.spec_from_file_location("run_production_acceptance", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeProcess:
    _next_pid = 900_000

    def __init__(self, return_code: int | None = None) -> None:
        self.pid = self._next_pid
        type(self)._next_pid += 1
        self.return_code = return_code
        self.terminated = False
        self.reaped = False

    def poll(self) -> int | None:
        return self.return_code

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        if self.return_code is None:
            raise subprocess.TimeoutExpired("private-command", 0)
        self.reaped = True
        return self.return_code


class _Clock:
    def __init__(self, *, on_sleep: Callable[[], None] | None = None) -> None:
        self.now = 0.0
        self.on_sleep = on_sleep

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        if self.on_sleep is not None:
            callback, self.on_sleep = self.on_sleep, None
            callback()


def _fixture_config(module: ModuleType, tmp_path: Path) -> Any:
    archive = tmp_path / "private-archive"
    archive.mkdir(mode=0o700)
    browser = tmp_path / "private-browser-report.json"
    browser.write_text('{"private":"browser evidence"}\n', encoding="utf-8")
    browser.chmod(0o600)
    parent = tmp_path / "private-runs"
    parent.mkdir(mode=0o700)
    return module.StackConfig(
        archive_root=archive,
        browser_report=browser,
        work_directory=parent / "fresh-run",
        readiness_timeout_seconds=2.0,
        poll_seconds=0.1,
        shutdown_grace_seconds=0.2,
    )


def _acceptance_payload(
    module: ModuleType,
    *,
    gate_passed: bool = True,
    profile: str | None = None,
    schema_version: object = 2,
) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "profile": {
            "profile_version": profile or module.EXPECTED_PROFILE_IDENTIFIER,
            "private_probe": "must not enter the stack receipt",
        },
        "gate": {"passed": gate_passed},
    }


def _dependencies(
    module: ModuleType,
    config: Any,
    *,
    client_exit: int | None = 0,
    gate_passed: bool = True,
    profile: str | None = None,
    schema_version: object = 2,
    health_ready: bool = True,
    clock: _Clock | None = None,
) -> tuple[Any, list[list[str]], _FakeProcess, list[_FakeProcess], list[str]]:
    commands: list[list[str]] = []
    health_urls: list[str] = []
    server = _FakeProcess()
    clients: list[_FakeProcess] = []
    token_secret = "private-one-time-token-that-must-not-leak"

    def spawn(
        command: Sequence[str], working_directory: Path, environment: Mapping[str, str]
    ) -> _FakeProcess:
        module_file = module.__file__
        assert module_file is not None
        assert working_directory == Path(module_file).resolve().parents[1]
        assert environment["OMP_NUM_THREADS"] == "1"
        commands.append(list(command))
        if "serve" in command:
            token_path = Path(command[command.index("--startup-token-file") + 1])
            token_path.write_text(token_secret, encoding="ascii")
            token_path.chmod(0o600)
            return server
        output = Path(command[command.index("--output") + 1])
        output.write_text(
            json.dumps(
                _acceptance_payload(
                    module,
                    gate_passed=gate_passed,
                    profile=profile,
                    schema_version=schema_version,
                ),
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        output.chmod(0o600)
        client = _FakeProcess(client_exit)
        clients.append(client)
        return client

    def health_probe(url: str, timeout: float) -> bool:
        assert 0 < timeout <= module.MAX_HEALTH_PROBE_TIMEOUT_SECONDS
        health_urls.append(url)
        return health_ready

    def terminate(process: _FakeProcess, grace_seconds: float) -> bool:
        assert grace_seconds == config.shutdown_grace_seconds
        process.terminated = True
        if process.return_code is None:
            process.return_code = -signal.SIGTERM
        process.wait(timeout=grace_seconds)
        return True

    active_clock = clock or _Clock()
    dependencies = module.RunnerDependencies(
        spawn=spawn,
        health_probe=health_probe,
        token_probe=module._token_identity,
        evidence_probe=module._inspect_acceptance_evidence,
        terminate=terminate,
        group_cleanup_probe=lambda: True,
        monotonic=active_clock.monotonic,
        sleep=active_clock.sleep,
    )
    return dependencies, commands, server, clients, health_urls


def _receipt(config: Any) -> dict[str, Any]:
    payload = json.loads((config.work_directory / "stack-receipt.json").read_text())
    assert isinstance(payload, dict)
    return payload


def test_happy_path_binds_report_profile_and_exact_cleanup(tmp_path: Path) -> None:
    module = _stack_module()
    config = _fixture_config(module, tmp_path)
    dependencies, commands, server, clients, health_urls = _dependencies(module, config)

    result = module.run_stack(config, dependencies)

    assert result == module.StackResult("passed", 0)
    assert commands[0][:5] == ["uv", "run", "meta-archive", "serve", "--acceptance-read-only"]
    assert commands[1][:4] == ["uv", "run", "python", "scripts/real_export_acceptance.py"]
    assert "production-spot-check" in commands[1]
    assert health_urls == ["http://127.0.0.1:8765/api/v1/health"]
    assert server.terminated and server.reaped
    assert len(clients) == 1 and clients[0].terminated and clients[0].reaped
    assert not (config.work_directory / "startup-token").exists()

    report = config.work_directory / "acceptance-report.json"
    receipt_path = config.work_directory / "stack-receipt.json"
    receipt = _receipt(config)
    assert receipt == {
        "schema_version": 1,
        "acceptance_report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        "acceptance_report_well_formed": True,
        "expected_public_profile_identifier": "production-spot-check-v2",
        "profile_matches": True,
        "gate_passed": True,
        "client_exit_code": 0,
        "server_started": True,
        "server_health_ready": True,
        "cleanup": {
            "client_terminated_or_reaped": True,
            "server_tree_terminated_and_waited": True,
            "inherited_process_group_clean": True,
            "token_exact_inode_removed": True,
            "exact_cleanup_confirmed": True,
        },
        "stack_status": "passed",
        "stack_exit_code": 0,
    }
    serialized_receipt = receipt_path.read_text(encoding="utf-8")
    assert "private-one-time-token" not in serialized_receipt
    assert "private_probe" not in serialized_receipt
    assert str(config.archive_root) not in serialized_receipt
    assert stat.S_IMODE(config.work_directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(report.stat().st_mode) == 0o600
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o600


def test_readiness_timeout_never_starts_client_and_still_cleans_server_token(
    tmp_path: Path,
) -> None:
    module = _stack_module()
    config = _fixture_config(module, tmp_path)
    clock = _Clock()
    dependencies, commands, server, clients, health_urls = _dependencies(
        module, config, health_ready=False, clock=clock
    )

    result = module.run_stack(config, dependencies)

    assert result == module.StackResult("readiness_timeout", module.EXIT_READINESS_TIMEOUT)
    assert len(commands) == 1
    assert clients == []
    assert health_urls
    assert server.terminated and server.reaped
    assert not (config.work_directory / "startup-token").exists()
    receipt = _receipt(config)
    assert receipt["server_started"] is True
    assert receipt["server_health_ready"] is False
    assert receipt["cleanup"]["token_exact_inode_removed"] is True
    assert receipt["stack_exit_code"] == module.EXIT_READINESS_TIMEOUT


def test_client_failure_and_false_gate_are_recorded_and_fail_closed(tmp_path: Path) -> None:
    module = _stack_module()
    config = _fixture_config(module, tmp_path)
    dependencies, _commands, server, clients, _health_urls = _dependencies(
        module, config, client_exit=9, gate_passed=False
    )

    result = module.run_stack(config, dependencies)

    assert result == module.StackResult("client_failed", 9)
    assert server.terminated and server.reaped
    assert len(clients) == 1 and clients[0].terminated and clients[0].reaped
    receipt = _receipt(config)
    assert receipt["client_exit_code"] == 9
    assert receipt["gate_passed"] is False
    assert receipt["profile_matches"] is True
    assert receipt["stack_exit_code"] == 9


def test_profile_mismatch_with_zero_client_exit_fails_closed(tmp_path: Path) -> None:
    module = _stack_module()
    config = _fixture_config(module, tmp_path)
    dependencies, *_rest = _dependencies(module, config, profile="unexpected-public-profile")

    result = module.run_stack(config, dependencies)

    assert result == module.StackResult("acceptance_evidence_invalid", module.EXIT_EVIDENCE_INVALID)
    receipt = _receipt(config)
    assert receipt["profile_matches"] is False
    assert "unexpected-public-profile" not in json.dumps(receipt)


@pytest.mark.parametrize("schema_version", [None, 1, 3, 2.0, "2", True])
def test_report_schema_mismatch_with_zero_client_exit_fails_closed(
    tmp_path: Path, schema_version: object
) -> None:
    module = _stack_module()
    config = _fixture_config(module, tmp_path)
    dependencies, *_rest = _dependencies(
        module,
        config,
        schema_version=schema_version,
    )

    result = module.run_stack(config, dependencies)

    assert result == module.StackResult("acceptance_evidence_invalid", module.EXIT_EVIDENCE_INVALID)
    receipt = _receipt(config)
    assert receipt["acceptance_report_well_formed"] is False
    assert receipt["profile_matches"] is True
    assert receipt["gate_passed"] is True
    assert receipt["stack_status"] == "acceptance_evidence_invalid"
    assert receipt["stack_exit_code"] == module.EXIT_EVIDENCE_INVALID


def test_report_hash_and_gate_share_one_stable_file_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _stack_module()
    report = tmp_path / "acceptance-report.json"
    report.write_text(
        json.dumps(_acceptance_payload(module), sort_keys=True) + "\n", encoding="utf-8"
    )
    report.chmod(0o600)
    real_fstat: Callable[[int], os.stat_result] = os.fstat
    fstat_calls = 0

    def replace_path_after_descriptor_read(descriptor: int) -> os.stat_result:
        nonlocal fstat_calls
        metadata = real_fstat(descriptor)
        fstat_calls += 1
        if fstat_calls == 2:
            report.unlink()
            report.write_text(
                json.dumps(_acceptance_payload(module, gate_passed=False), sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report.chmod(0o600)
        return metadata

    monkeypatch.setattr(module.os, "fstat", replace_path_after_descriptor_read)
    evidence = module._inspect_acceptance_evidence(report)

    assert evidence == module.AcceptanceEvidence(None, False, False, False)


def test_sigterm_path_terminates_and_reaps_both_children(tmp_path: Path) -> None:
    module = _stack_module()
    config = _fixture_config(module, tmp_path)
    clock = _Clock(on_sleep=lambda: module._signal_handler(signal.SIGTERM, None))
    dependencies, _commands, server, clients, _health_urls = _dependencies(
        module, config, client_exit=None, clock=clock
    )

    result = module.run_stack(config, dependencies)

    assert result == module.StackResult("interrupted", 128 + signal.SIGTERM)
    assert server.terminated and server.reaped
    assert len(clients) == 1 and clients[0].terminated and clients[0].reaped
    receipt = _receipt(config)
    assert receipt["client_exit_code"] == 128 + signal.SIGTERM
    assert receipt["cleanup"]["client_terminated_or_reaped"] is True
    assert receipt["cleanup"]["server_tree_terminated_and_waited"] is True
    assert receipt["cleanup"]["token_exact_inode_removed"] is True
    assert receipt["stack_exit_code"] == 128 + signal.SIGTERM


def test_external_signal_cleanup_budget_stays_below_outer_eight_second_grace(
    tmp_path: Path,
) -> None:
    module = _stack_module()
    base_config = _fixture_config(module, tmp_path)
    config = replace(base_config, shutdown_grace_seconds=2.0, poll_seconds=10.0)
    clock = _Clock(on_sleep=lambda: module._signal_handler(signal.SIGTERM, None))
    dependencies, _commands, _server, _clients, _health_urls = _dependencies(
        module, config, client_exit=None, clock=clock
    )
    observed_graces: list[float] = []
    base_terminate: Callable[[_FakeProcess, float], bool] = dependencies.terminate

    def record_grace(process: _FakeProcess, grace_seconds: float) -> bool:
        observed_graces.append(grace_seconds)
        return base_terminate(process, grace_seconds)

    dependencies = replace(dependencies, terminate=record_grace)
    result = module.run_stack(config, dependencies)

    assert result == module.StackResult("interrupted", 128 + signal.SIGTERM)
    assert observed_graces == [2.0, 2.0]
    assert clock.now == pytest.approx(module.MAX_CONTROL_LOOP_SLEEP_SECONDS)
    assert clock.now + sum(observed_graces) + module.GROUP_CLEANUP_TIMEOUT_SECONDS < 8.0


def test_orderly_server_token_unlink_counts_as_exact_inode_removal(tmp_path: Path) -> None:
    module = _stack_module()
    config = _fixture_config(module, tmp_path)
    dependencies, _commands, server, _client, _health_urls = _dependencies(module, config)
    base_terminate: Callable[[_FakeProcess, float], bool] = dependencies.terminate

    def terminate_after_server_unlinks_token(process: _FakeProcess, grace_seconds: float) -> bool:
        if process is server:
            (config.work_directory / "startup-token").unlink()
        return base_terminate(process, grace_seconds)

    dependencies = replace(dependencies, terminate=terminate_after_server_unlinks_token)
    result = module.run_stack(config, dependencies)

    assert result == module.StackResult("passed", 0)
    receipt = _receipt(config)
    assert receipt["cleanup"]["token_exact_inode_removed"] is True
    assert receipt["cleanup"]["exact_cleanup_confirmed"] is True


def test_replaced_token_inode_is_retained_and_cleanup_fails_closed(tmp_path: Path) -> None:
    module = _stack_module()
    config = _fixture_config(module, tmp_path)
    dependencies, _commands, server, _client, _health_urls = _dependencies(module, config)
    base_terminate: Callable[[_FakeProcess, float], bool] = dependencies.terminate
    replacement = b"different inode must be retained\n"

    def replace_token_during_server_shutdown(process: _FakeProcess, grace_seconds: float) -> bool:
        if process is server:
            token = config.work_directory / "startup-token"
            token.unlink()
            token.write_bytes(replacement)
            token.chmod(0o600)
        return base_terminate(process, grace_seconds)

    dependencies = replace(dependencies, terminate=replace_token_during_server_shutdown)
    result = module.run_stack(config, dependencies)

    assert result == module.StackResult(
        "cleanup_or_evidence_unconfirmed", module.EXIT_CLEANUP_FAILED
    )
    assert (config.work_directory / "startup-token").read_bytes() == replacement
    receipt = _receipt(config)
    assert receipt["cleanup"]["token_exact_inode_removed"] is False
    assert receipt["stack_exit_code"] == module.EXIT_CLEANUP_FAILED


def test_inherited_group_member_disappearing_during_retry_allows_success(
    tmp_path: Path,
) -> None:
    module = _stack_module()
    config = _fixture_config(module, tmp_path)
    clock = _Clock()
    dependencies, _commands, _server, clients, _health_urls = _dependencies(
        module, config, clock=clock
    )
    probe_results = iter((False, False, True))
    probe_calls = 0

    def group_becomes_clean() -> bool:
        nonlocal probe_calls
        probe_calls += 1
        return next(probe_results)

    dependencies = replace(dependencies, group_cleanup_probe=group_becomes_clean)

    result = module.run_stack(config, dependencies)

    assert result == module.StackResult("passed", 0)
    assert len(clients) == 1 and clients[0].terminated and clients[0].reaped
    assert probe_calls == 3
    assert clock.now == pytest.approx(2 * module.GROUP_CLEANUP_POLL_SECONDS)
    receipt = _receipt(config)
    assert receipt["cleanup"]["inherited_process_group_clean"] is True
    assert receipt["cleanup"]["exact_cleanup_confirmed"] is True


def test_lingering_inherited_group_member_through_timeout_prevents_success_receipt(
    tmp_path: Path,
) -> None:
    module = _stack_module()
    config = _fixture_config(module, tmp_path)
    clock = _Clock()
    dependencies, _commands, _server, clients, _health_urls = _dependencies(
        module, config, clock=clock
    )
    probe_calls = 0

    def group_stays_unclean() -> bool:
        nonlocal probe_calls
        probe_calls += 1
        return False

    dependencies = replace(dependencies, group_cleanup_probe=group_stays_unclean)

    result = module.run_stack(config, dependencies)

    assert len(clients) == 1 and clients[0].terminated and clients[0].reaped
    assert result == module.StackResult(
        "cleanup_or_evidence_unconfirmed", module.EXIT_CLEANUP_FAILED
    )
    receipt = _receipt(config)
    assert receipt["cleanup"]["inherited_process_group_clean"] is False
    assert receipt["cleanup"]["exact_cleanup_confirmed"] is False
    assert probe_calls > 1
    assert clock.now == pytest.approx(module.GROUP_CLEANUP_TIMEOUT_SECONDS)


@pytest.mark.parametrize("unsafe_kind", ["existing", "symlink", "open-parent"])
def test_existing_or_unsafe_destination_is_refused_without_overwrite(
    tmp_path: Path, unsafe_kind: str
) -> None:
    module = _stack_module()
    config = _fixture_config(module, tmp_path)
    parent = config.work_directory.parent
    sentinel = b"prior private evidence must survive byte-for-byte\n"
    if unsafe_kind == "existing":
        config.work_directory.mkdir(mode=0o700)
        (config.work_directory / "sentinel").write_bytes(sentinel)
    elif unsafe_kind == "symlink":
        target = tmp_path / "outside"
        target.mkdir(mode=0o700)
        config.work_directory.symlink_to(target, target_is_directory=True)
    else:
        parent.chmod(0o755)

    with pytest.raises(module.DestinationRefusedError):
        module.run_stack(config)

    if unsafe_kind == "existing":
        assert (config.work_directory / "sentinel").read_bytes() == sentinel
    if unsafe_kind == "symlink":
        assert config.work_directory.is_symlink()
    assert not (config.work_directory / "stack-receipt.json").exists()


def test_cli_refusal_emits_only_one_sanitized_status_line(tmp_path: Path) -> None:
    module = _stack_module()
    config = _fixture_config(module, tmp_path)
    config.work_directory.mkdir(mode=0o700)
    secret = "private-sentinel-must-not-leak"
    sentinel = config.work_directory / secret
    sentinel.write_text(secret, encoding="utf-8")
    module_file = module.__file__
    assert module_file is not None
    script = Path(module_file)

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--archive-root",
            str(config.archive_root),
            "--browser-report",
            str(config.browser_report),
            "--work-dir",
            str(config.work_directory),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert result.returncode == module.EXIT_DESTINATION_REFUSED
    assert result.stderr == ""
    assert result.stdout == "production-acceptance-stack: destination_refused (exit 70)\n"
    assert secret not in result.stdout
    assert str(config.work_directory) not in result.stdout
    assert sentinel.read_text(encoding="utf-8") == secret


def test_spawn_uses_no_shell_and_inherits_process_group(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _stack_module()
    observed: dict[str, object] = {}
    process = _FakeProcess(0)

    def fake_popen(command: Sequence[str], **kwargs: object) -> _FakeProcess:
        observed["command"] = list(command)
        observed.update(kwargs)
        return process

    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)
    spawned = module._spawn_process(["uv", "run", "private-command"], Path("/"), {})

    assert spawned is process
    assert observed["shell"] is False
    assert observed["start_new_session"] is False
    assert "preexec_fn" not in observed
    assert observed["stdin"] is subprocess.DEVNULL
    assert observed["stdout"] is subprocess.DEVNULL
    assert observed["stderr"] is subprocess.DEVNULL


def test_real_process_tree_termination_reaps_parent_and_stops_grandchild(tmp_path: Path) -> None:
    module = _stack_module()
    grandchild_pid_file = tmp_path / "grandchild.pid"
    source = """
import pathlib
import subprocess
import sys
import time

child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
pathlib.Path(sys.argv[1]).write_text(str(child.pid), encoding="ascii")
time.sleep(30)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", source, str(grandchild_pid_file)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        start_new_session=False,
    )
    deadline = time.monotonic() + 2.0
    while not grandchild_pid_file.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert grandchild_pid_file.exists()
    grandchild_pid = int(grandchild_pid_file.read_text(encoding="ascii"))

    assert module._terminate_process_tree(process, 0.5) is True
    assert process.poll() is not None
    identity = module._proc_identity(grandchild_pid)
    assert identity is None or identity[1] in {"X", "Z"}


def test_group_probe_detects_grandchild_after_direct_parent_exit(tmp_path: Path) -> None:
    module = _stack_module()
    grandchild_pid_file = tmp_path / "orphan.pid"
    source = """
import pathlib
import subprocess
import sys

child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
pathlib.Path(sys.argv[1]).write_text(str(child.pid), encoding="ascii")
"""
    parent = subprocess.Popen(
        [sys.executable, "-c", source, str(grandchild_pid_file)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        start_new_session=True,
    )
    parent.wait(timeout=2)
    grandchild_pid = int(grandchild_pid_file.read_text(encoding="ascii"))
    try:
        assert module._process_group_is_clean(parent.pid, {parent.pid}) is False
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(grandchild_pid, signal.SIGKILL)
    deadline = time.monotonic() + 2.0
    while (
        not module._process_group_is_clean(parent.pid, {parent.pid}) and time.monotonic() < deadline
    ):
        time.sleep(0.01)
    assert module._process_group_is_clean(parent.pid, {parent.pid}) is True


def test_group_probe_fails_closed_on_proc_stat_io_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _stack_module()

    class _ProcEntry:
        name = "123"

    class _ProcEntries:
        def __enter__(self) -> _ProcEntries:
            return self

        def __exit__(self, *args: object) -> None:
            del args

        def __iter__(self) -> Iterator[_ProcEntry]:
            return iter((_ProcEntry(),))

    def fail_read_text(path: Path, encoding: str | None = None, errors: str | None = None) -> str:
        del path, encoding, errors
        raise PermissionError("simulated proc stat denial")

    monkeypatch.setattr(module.os, "scandir", lambda _path: _ProcEntries())
    monkeypatch.setattr(module.Path, "read_text", fail_read_text)

    assert module._process_group_is_clean(expected_group=456, ignored_pids={789}) is False


def test_verified_leader_shim_ancestor_chain_is_the_only_additional_group_ignore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _stack_module()
    monkeypatch.setenv(module.LEADER_SHIM_PID_ENVIRONMENT_VARIABLE, "123")
    monkeypatch.setattr(module.os, "getpid", lambda: 789)
    monkeypatch.setattr(module.os, "getppid", lambda: 200)
    monkeypatch.setattr(module.os, "getpgrp", lambda: 456)
    identities = {
        200: (150, 456, 2000, "S"),
        150: (123, 456, 1500, "S"),
        123: (1, 456, 1230, "S"),
    }
    monkeypatch.setattr(module, "_proc_group_identity", identities.get)
    observed: dict[str, object] = {}

    def group_is_clean(expected_group: int, ignored_pids: Collection[int]) -> bool:
        observed["expected_group"] = expected_group
        observed["ignored_pids"] = set(ignored_pids)
        return True

    monkeypatch.setattr(module, "_process_group_is_clean", group_is_clean)

    assert module._inherited_process_group_is_clean() is True
    assert observed == {"expected_group": 456, "ignored_pids": {123, 150, 200, 789}}


@pytest.mark.parametrize(
    ("raw_pid", "identities"),
    [
        (None, {200: (123, 456, 99, "S"), 123: (1, 456, 99, "S")}),
        (" 123", {200: (123, 456, 99, "S"), 123: (1, 456, 99, "S")}),
        ("0123", {200: (123, 456, 99, "S"), 123: (1, 456, 99, "S")}),
        ("123", {}),
        ("123", {200: (123, 999, 99, "S"), 123: (1, 456, 99, "S")}),
        ("123", {200: (123, 456, 99, "Z"), 123: (1, 456, 99, "S")}),
        ("123", {200: (123, 456, 0, "S"), 123: (1, 456, 99, "S")}),
        ("123", {200: (200, 456, 99, "S")}),
        ("123", {200: (124, 456, 99, "S"), 124: (1, 456, 99, "S")}),
    ],
)
def test_invalid_leader_shim_ancestor_chain_fails_closed_without_group_scan(
    monkeypatch: pytest.MonkeyPatch,
    raw_pid: str | None,
    identities: dict[int, tuple[int, int, int, str]],
) -> None:
    module = _stack_module()
    if raw_pid is None:
        monkeypatch.delenv(module.LEADER_SHIM_PID_ENVIRONMENT_VARIABLE, raising=False)
    else:
        monkeypatch.setenv(module.LEADER_SHIM_PID_ENVIRONMENT_VARIABLE, raw_pid)
    monkeypatch.setattr(module.os, "getpid", lambda: 789)
    monkeypatch.setattr(module.os, "getppid", lambda: 200)
    monkeypatch.setattr(module.os, "getpgrp", lambda: 456)
    monkeypatch.setattr(module, "_proc_group_identity", identities.get)
    scanned = False

    def group_is_clean(_expected_group: int, _ignored_pids: Collection[int]) -> bool:
        nonlocal scanned
        scanned = True
        return True

    monkeypatch.setattr(module, "_process_group_is_clean", group_is_clean)

    assert module._inherited_process_group_is_clean() is False
    assert scanned is False


def test_leader_shim_ancestor_chain_fails_closed_when_identity_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _stack_module()
    monkeypatch.setenv(module.LEADER_SHIM_PID_ENVIRONMENT_VARIABLE, "123")
    monkeypatch.setattr(module.os, "getpid", lambda: 789)
    monkeypatch.setattr(module.os, "getppid", lambda: 200)
    monkeypatch.setattr(module.os, "getpgrp", lambda: 456)
    reads = iter(
        (
            (123, 456, 2000, "S"),
            (1, 456, 1230, "S"),
            (123, 456, 2001, "S"),
        )
    )
    monkeypatch.setattr(module, "_proc_group_identity", lambda _pid: next(reads))
    scanned = False

    def group_is_clean(_expected_group: int, _ignored_pids: Collection[int]) -> bool:
        nonlocal scanned
        scanned = True
        return True

    monkeypatch.setattr(module, "_process_group_is_clean", group_is_clean)

    assert module._inherited_process_group_is_clean() is False
    assert scanned is False


def test_child_environment_does_not_propagate_reserved_shim_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _stack_module()
    monkeypatch.setenv(module.LEADER_SHIM_PID_ENVIRONMENT_VARIABLE, "123")

    environment = module._child_environment()

    assert module.LEADER_SHIM_PID_ENVIRONMENT_VARIABLE not in environment
    assert environment["OMP_NUM_THREADS"] == "1"


def test_real_supervisor_wrapper_accepts_only_its_live_leader_shim(tmp_path: Path) -> None:
    module_file = Path(__file__).resolve().parents[1] / "scripts" / "run_production_acceptance.py"
    supervisor = Path(__file__).resolve().parents[1] / "scripts" / "supervise_acceptance.py"
    uv = shutil.which("uv")
    assert uv is not None
    output = tmp_path / "wrapped-group-cleanup"
    source = """
import importlib.util
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('wrapped_stack_runner', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
chain = module._verified_leader_shim_ancestor_pids()
raise SystemExit(0 if len(chain) >= 2 and module._inherited_process_group_is_clean() else 76)
"""

    result = subprocess.run(
        [
            sys.executable,
            str(supervisor),
            "--output-dir",
            str(output),
            "--sample-seconds",
            "0.02",
            "--term-grace-seconds",
            "0.2",
            "--",
            uv,
            "run",
            "python",
            "-c",
            source,
            str(module_file),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["cleanup_complete"] is True


def test_real_supervisor_wrapper_still_detects_unrelated_live_group_member(
    tmp_path: Path,
) -> None:
    module_file = Path(__file__).resolve().parents[1] / "scripts" / "run_production_acceptance.py"
    supervisor = Path(__file__).resolve().parents[1] / "scripts" / "supervise_acceptance.py"
    uv = shutil.which("uv")
    assert uv is not None
    output = tmp_path / "wrapped-group-lingerer"
    source = """
import importlib.util
import pathlib
import subprocess
import sys

path = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('wrapped_stack_runner_lingerer', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
try:
    detected = not module._inherited_process_group_is_clean()
finally:
    child.terminate()
    child.wait(timeout=2)
clean = module._inherited_process_group_is_clean()
raise SystemExit(0 if detected and clean else 76)
"""

    result = subprocess.run(
        [
            sys.executable,
            str(supervisor),
            "--output-dir",
            str(output),
            "--sample-seconds",
            "0.02",
            "--term-grace-seconds",
            "0.2",
            "--",
            uv,
            "run",
            "python",
            "-c",
            source,
            str(module_file),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "passed"


@pytest.mark.parametrize(
    ("option", "value"),
    (
        ("--ready-timeout-seconds", "nan"),
        ("--ready-timeout-seconds", "inf"),
        ("--ready-timeout-seconds", "3601"),
        ("--poll-seconds", "11"),
        ("--shutdown-grace-seconds", "2.1"),
    ),
)
def test_nonfinite_or_unbounded_durations_are_refused(option: str, value: str) -> None:
    module = _stack_module()
    with pytest.raises(module.DestinationRefusedError):
        module._arguments(
            [
                "--archive-root",
                "/private/archive",
                "--browser-report",
                "/private/browser.json",
                "--work-dir",
                "/private/run",
                option,
                value,
            ]
        )


def test_receipt_failure_never_unlinks_a_raced_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _stack_module()
    private_directory = tmp_path / "private"
    private_directory.mkdir(mode=0o700)
    receipt = private_directory / "stack-receipt.json"
    sentinel = b"raced replacement must survive\n"
    real_fsync = module.os.fsync
    calls = 0

    def fail_after_publication(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            receipt.unlink()
            receipt.write_bytes(sentinel)
            receipt.chmod(0o600)
            raise OSError("simulated directory fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(module.os, "fsync", fail_after_publication)
    with pytest.raises(OSError, match="simulated directory fsync failure"):
        module._write_private_receipt(receipt, {"stack_exit_code": 0})

    assert receipt.read_bytes() == sentinel


def test_health_probe_requires_ready_catalog_and_caps_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _stack_module()

    class _Headers:
        def __init__(self, length: int) -> None:
            self.length = length

        def get(self, name: str) -> str | None:
            assert name == "Content-Length"
            return str(self.length)

    class _Response:
        status = 200

        def __init__(self, payload: bytes, declared_length: int | None = None) -> None:
            self.payload = payload
            length = declared_length if declared_length is not None else len(payload)
            self.headers = _Headers(length)

        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *args: object) -> None:
            del args

        def read(self, maximum: int) -> bytes:
            return self.payload[:maximum]

    responses = iter(
        [
            _Response(b'{"status":"ok","catalog_ready":true}'),
            _Response(b'{"status":"ok","catalog_ready":false}'),
            _Response(b"{}", module.MAX_HEALTH_RESPONSE_BYTES + 1),
        ]
    )

    def fake_urlopen(request: object, timeout: float) -> _Response:
        del request, timeout
        return next(responses)

    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)
    assert module._health_is_ready("http://127.0.0.1:8765/api/v1/health", 1.0) is True
    assert module._health_is_ready("http://127.0.0.1:8765/api/v1/health", 1.0) is False
    assert module._health_is_ready("http://127.0.0.1:8765/api/v1/health", 1.0) is False
