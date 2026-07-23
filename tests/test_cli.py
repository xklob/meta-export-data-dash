from __future__ import annotations

import json
import os
import stat
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from typer.testing import CliRunner

import meta_archive.cli as cli_module
from meta_archive.cli import _mandatory_non_pass, app
from meta_archive.paths import ArchiveLayout
from meta_archive.services.consolidation import ConflictHoldingSummary
from meta_archive.services.inventory import InventorySummary
from meta_archive.services.support_library_scrub import (
    SupportLibraryScrubError,
    SupportLibraryScrubSummary,
)


def test_mandatory_non_pass_rejects_pending_but_ignores_optional_warning() -> None:
    checks = [
        {"key": "ready", "status": "pass", "mandatory": True},
        {"key": "pending", "status": "pending", "mandatory": True},
        {"key": "optional", "status": "pending", "mandatory": False},
    ]

    assert [check["key"] for check in _mandatory_non_pass(checks)] == ["pending"]


def test_retire_source_cli_requires_exact_explicit_chunk_options() -> None:
    runner = CliRunner()

    missing = runner.invoke(app, ["retire-source"])
    help_result = runner.invoke(app, ["retire-source", "--help"])

    assert missing.exit_code == 2
    assert "Missing option '--chunk'" in missing.output
    assert help_result.exit_code == 0
    assert "Exact source-chunk root name" in help_result.output


def test_consolidate_cli_exposes_explicit_dry_run_and_apply_modes() -> None:
    result = CliRunner().invoke(app, ["consolidate", "--help"])

    assert result.exit_code == 0, result.output
    assert "--dry-run" in result.output
    assert "--apply" in result.output


def test_ingest_cli_exposes_private_owner_identity_option() -> None:
    result = CliRunner().invoke(app, ["ingest", "--help"])

    assert result.exit_code == 0
    assert "--owner-name" in result.output
    assert "Persisted in the private" in result.output
    assert "catalog" in result.output


def test_scrub_support_library_cli_emits_only_aggregate_results_and_redacts_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = ArchiveLayout(tmp_path)
    engine = object()
    private_marker = "private-support-library-error-marker"
    monkeypatch.setattr(cli_module, "_layout", lambda _archive_root: layout)
    monkeypatch.setattr(cli_module, "_catalog", lambda _layout: engine)
    monkeypatch.setattr(
        cli_module,
        "scrub_support_library",
        lambda _layout, _engine: SupportLibraryScrubSummary(
            support_conversation_count=2,
            transcript_files_scanned=3,
            transcript_files_rewritten=1,
        ),
    )

    success = CliRunner().invoke(
        app,
        ["scrub-support-library", "--archive-root", str(tmp_path)],
    )

    assert success.exit_code == 0, success.output
    assert json.loads(success.output) == {
        "support_conversation_count": 2,
        "conversation_directories_found": 0,
        "transcript_files_scanned": 3,
        "transcript_records_scanned": 0,
        "message_records_scanned": 0,
        "message_records_changed": 0,
        "sensitive_fields_removed": 0,
        "transcript_files_rewritten": 1,
        "transcript_files_unchanged": 0,
        "stale_temp_files_removed": 0,
    }
    assert private_marker not in success.output

    def fail_safely(_layout: object, _engine: object) -> None:
        raise SupportLibraryScrubError(private_marker)

    monkeypatch.setattr(cli_module, "scrub_support_library", fail_safely)
    failure = CliRunner().invoke(
        app,
        ["scrub-support-library", "--archive-root", str(tmp_path)],
    )

    assert failure.exit_code == 1
    assert "Support transcript repair failed safely" in failure.output
    assert private_marker not in failure.output


def test_consolidate_holds_conflicts_and_remains_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = ArchiveLayout(tmp_path)
    engine = object()
    inventory = InventorySummary(conflicting_collision_groups=1)
    holding = ConflictHoldingSummary(
        group_count=1,
        copy_count=2,
        materialized_copy_count=2,
        reused_copy_count=0,
        manifest_count=1,
        holding_root="conflicts/normalized-path",
    )
    held = False

    monkeypatch.setattr(cli_module, "_layout", lambda _archive_root: layout)
    monkeypatch.setattr(cli_module, "_catalog", lambda _layout: engine)
    monkeypatch.setattr(
        cli_module,
        "source_import_publication_lock",
        lambda *_args, **_kwargs: nullcontext(object()),
    )
    monkeypatch.setattr(
        cli_module, "conflict_holding_lock", lambda *_args, **_kwargs: nullcontext(object())
    )
    monkeypatch.setattr(cli_module, "scan_archive_locked", lambda *_args, **_kwargs: inventory)

    def hold_conflicts(*_args: object, **_kwargs: object) -> ConflictHoldingSummary:
        nonlocal held
        held = True
        return holding

    def unexpected_blob_build(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("blob-store apply must remain blocked by conflicts")

    monkeypatch.setattr(cli_module, "materialize_conflict_holding_area_locked", hold_conflicts)
    monkeypatch.setattr(cli_module, "build_blob_store_locked", unexpected_blob_build)

    result = CliRunner().invoke(
        app,
        ["consolidate", "--archive-root", str(tmp_path), "--apply"],
    )

    assert result.exit_code == 1
    assert held is True
    assert "append-only conflict holding" in result.output
    assert "Blob-store promotion is disabled" in result.output
    assert '"copy_count": 2' in result.output


def test_consolidate_dry_run_reports_conflicts_without_materializing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = ArchiveLayout(tmp_path)
    engine = object()
    inventory = InventorySummary(conflicting_collision_groups=1)

    monkeypatch.setattr(cli_module, "_layout", lambda _archive_root: layout)
    monkeypatch.setattr(cli_module, "_catalog", lambda _layout: engine)
    monkeypatch.setattr(
        cli_module,
        "source_import_publication_lock",
        lambda *_args, **_kwargs: nullcontext(object()),
    )
    monkeypatch.setattr(cli_module, "scan_archive_locked", lambda *_args, **_kwargs: inventory)

    def unexpected_holding(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("dry-run must not create conflict holding files")

    monkeypatch.setattr(cli_module, "materialize_conflict_holding_area_locked", unexpected_holding)

    result = CliRunner().invoke(
        app,
        ["consolidate", "--archive-root", str(tmp_path), "--dry-run"],
    )

    assert result.exit_code == 1
    assert "Dry run found conflicting normalized paths" in result.output
    assert "No holding-area files were created" in result.output


def test_topic_cluster_cli_exposes_bounded_local_controls() -> None:
    result = CliRunner().invoke(app, ["build-topic-clusters", "--help"])

    assert result.exit_code == 0
    assert "--clusters" in result.output
    assert "--sample-limit" in result.output
    assert "active semantic-text index" in result.output


def test_warm_analytics_passes_the_requested_timezone_to_every_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = ArchiveLayout(tmp_path)
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(cli_module, "_layout", lambda _archive_root: layout)
    monkeypatch.setattr(cli_module, "_catalog", lambda _layout: object())
    monkeypatch.setattr(cli_module, "clear_analytics_cache", lambda _layout: 0)

    def overview(_request: object, *, timezone: str) -> dict[str, Any]:
        calls.append(("overview", timezone))
        return {}

    def operation(name: str) -> Any:
        def invoke(_request: object, **kwargs: Any) -> dict[str, Any]:
            calls.append((name, kwargs["timezone"]))
            return {}

        return invoke

    monkeypatch.setattr(cli_module, "analytics_overview", overview)
    monkeypatch.setattr(cli_module, "message_series", operation("messages-month"))
    monkeypatch.setattr(cli_module, "activity_patterns", operation("activity-patterns"))
    monkeypatch.setattr(
        cli_module,
        "interaction_patterns",
        operation("interaction-patterns-month"),
    )
    monkeypatch.setattr(
        cli_module,
        "interaction_series",
        operation("interaction-series-month"),
    )
    monkeypatch.setattr(cli_module, "attachment_analytics", operation("attachments-month"))

    result = CliRunner().invoke(
        app,
        [
            "warm-analytics",
            "--archive-root",
            str(tmp_path),
            "--timezone",
            "Europe/Paris",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        ("overview", "Europe/Paris"),
        ("messages-month", "Europe/Paris"),
        ("activity-patterns", "Europe/Paris"),
        ("interaction-patterns-month", "Europe/Paris"),
        ("interaction-series-month", "Europe/Paris"),
        ("attachments-month", "Europe/Paris"),
    ]


def test_serve_cli_wires_explicit_acceptance_read_only_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = ArchiveLayout(tmp_path)
    captured: dict[str, Any] = {}
    application = object()

    monkeypatch.setattr(cli_module, "_layout", lambda _archive_root: layout)
    monkeypatch.setattr(cli_module, "generate_access_token", lambda: "a" * 32)

    def fake_create_app(
        root: Path,
        *,
        access_token: str,
        acceptance_read_only: bool,
    ) -> object:
        captured.update(
            root=root,
            access_token=access_token,
            acceptance_read_only=acceptance_read_only,
        )
        return application

    def fake_run(app_value: object, **kwargs: Any) -> None:
        captured.update(application=app_value, uvicorn=kwargs)

    monkeypatch.setattr(cli_module, "create_app", fake_create_app)
    monkeypatch.setattr(uvicorn, "run", fake_run)

    result = CliRunner().invoke(
        app,
        [
            "serve",
            "--archive-root",
            str(tmp_path),
            "--acceptance-read-only",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "root": tmp_path,
        "access_token": "a" * 32,
        "acceptance_read_only": True,
        "application": application,
        "uvicorn": {"host": "127.0.0.1", "port": 8765, "access_log": False},
    }
    assert "Meta Archive Explorer is starting." in result.output
    assert "Meta Archive Explorer is ready." not in result.output
    assert "Acceptance read-only mode is active" in result.output
    assert f"http://127.0.0.1:8765/auth?token={'a' * 32}" in result.output


def test_serve_startup_token_file_is_atomic_private_and_suppresses_secret_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = ArchiveLayout(tmp_path)
    private_parent = tmp_path / "private"
    private_parent.mkdir(mode=0o700)
    private_parent.chmod(0o700)
    token_file = private_parent / "startup-token"
    token = "private-ephemeral-token-value-123456"
    captured: dict[str, Any] = {}
    observed_file: dict[str, Any] = {}
    application = object()

    monkeypatch.setattr(cli_module, "_layout", lambda _archive_root: layout)
    monkeypatch.setattr(cli_module, "generate_access_token", lambda: token)

    def fake_create_app(
        root: Path,
        *,
        access_token: str,
        acceptance_read_only: bool,
    ) -> object:
        captured.update(
            root=root,
            access_token=access_token,
            acceptance_read_only=acceptance_read_only,
        )
        return application

    def fake_run(app_value: object, **kwargs: Any) -> None:
        token_stat = token_file.lstat()
        observed_file.update(
            content=token_file.read_bytes(),
            mode=stat.S_IMODE(token_stat.st_mode),
            regular=stat.S_ISREG(token_stat.st_mode),
            symlink=stat.S_ISLNK(token_stat.st_mode),
        )
        captured.update(application=app_value, uvicorn=kwargs)

    monkeypatch.setattr(cli_module, "create_app", fake_create_app)
    monkeypatch.setattr(uvicorn, "run", fake_run)

    result = CliRunner().invoke(
        app,
        [
            "serve",
            "--archive-root",
            str(tmp_path),
            "--acceptance-read-only",
            "--startup-token-file",
            str(token_file),
        ],
    )

    assert result.exit_code == 0, result.output
    assert observed_file == {
        "content": f"{token}\n".encode(),
        "mode": 0o600,
        "regular": True,
        "symlink": False,
    }
    assert not token_file.exists()
    assert token not in result.output
    assert "/auth?token=" not in result.output
    assert "requested owner-only file" in result.output
    assert "Acceptance read-only mode is active" in result.output
    assert captured == {
        "root": tmp_path,
        "access_token": token,
        "acceptance_read_only": True,
        "application": application,
        "uvicorn": {"host": "127.0.0.1", "port": 8765, "access_log": False},
    }
    assert list(private_parent.glob(".meta-archive-token-*.tmp")) == []


def test_serve_startup_token_file_cleans_up_after_post_link_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = ArchiveLayout(tmp_path)
    private_parent = tmp_path / "private"
    private_parent.mkdir(mode=0o700)
    private_parent.chmod(0o700)
    token_file = private_parent / "startup-token"
    token = "token-published-before-injected-link-failure"

    monkeypatch.setattr(cli_module, "_layout", lambda _archive_root: layout)
    monkeypatch.setattr(cli_module, "generate_access_token", lambda: token)
    monkeypatch.setattr(
        cli_module,
        "create_app",
        lambda _root, *, access_token, acceptance_read_only: object(),
    )
    monkeypatch.setattr(
        uvicorn,
        "run",
        lambda *_args, **_kwargs: pytest.fail("server must not start after publication failure"),
    )
    real_link = os.link

    def link_then_fail(
        source: str,
        destination: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
        follow_symlinks: bool,
    ) -> None:
        real_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )
        raise OSError("injected failure after successful link")

    monkeypatch.setattr(cli_module.os, "link", link_then_fail)

    result = CliRunner().invoke(
        app,
        ["serve", "--archive-root", str(tmp_path), "--startup-token-file", str(token_file)],
    )

    assert result.exit_code == 2
    assert token not in result.output
    assert not token_file.exists()
    assert list(private_parent.glob(".meta-archive-token-*.tmp")) == []


def test_serve_startup_token_file_cleans_up_when_uvicorn_startup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = ArchiveLayout(tmp_path)
    private_parent = tmp_path / "private"
    private_parent.mkdir(mode=0o700)
    private_parent.chmod(0o700)
    token_file = private_parent / "startup-token"
    token = "token-visible-only-during-failed-server-startup"
    observed = False

    monkeypatch.setattr(cli_module, "_layout", lambda _archive_root: layout)
    monkeypatch.setattr(cli_module, "generate_access_token", lambda: token)
    monkeypatch.setattr(
        cli_module,
        "create_app",
        lambda _root, *, access_token, acceptance_read_only: object(),
    )

    def failed_startup(*_args: object, **_kwargs: object) -> None:
        nonlocal observed
        observed = token_file.read_bytes() == f"{token}\n".encode()
        raise OSError("injected Uvicorn bind failure")

    monkeypatch.setattr(uvicorn, "run", failed_startup)

    result = CliRunner().invoke(
        app,
        ["serve", "--archive-root", str(tmp_path), "--startup-token-file", str(token_file)],
    )

    assert result.exit_code == 1
    assert observed is True
    assert isinstance(result.exception, OSError)
    assert token not in result.output
    assert not token_file.exists()
    assert list(private_parent.glob(".meta-archive-token-*.tmp")) == []


def test_serve_startup_token_cleanup_preserves_a_raced_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = ArchiveLayout(tmp_path)
    private_parent = tmp_path / "private"
    private_parent.mkdir(mode=0o700)
    private_parent.chmod(0o700)
    token_file = private_parent / "startup-token"
    token = "token-whose-inode-will-be-replaced"
    replacement = b"replacement-owned-by-another-writer\n"
    inode_changed = False

    monkeypatch.setattr(cli_module, "_layout", lambda _archive_root: layout)
    monkeypatch.setattr(cli_module, "generate_access_token", lambda: token)
    monkeypatch.setattr(
        cli_module,
        "create_app",
        lambda _root, *, access_token, acceptance_read_only: object(),
    )

    def replace_then_fail(*_args: object, **_kwargs: object) -> None:
        nonlocal inode_changed
        published_descriptor = os.open(token_file, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            published_inode = os.fstat(published_descriptor).st_ino
            token_file.unlink()
            token_file.write_bytes(replacement)
            inode_changed = token_file.lstat().st_ino != published_inode
        finally:
            os.close(published_descriptor)
        raise OSError("injected failure after replacement")

    monkeypatch.setattr(uvicorn, "run", replace_then_fail)

    result = CliRunner().invoke(
        app,
        ["serve", "--archive-root", str(tmp_path), "--startup-token-file", str(token_file)],
    )

    assert result.exit_code == 1
    assert inode_changed is True
    assert token not in result.output
    assert token_file.read_bytes() == replacement
    assert list(private_parent.glob(".meta-archive-token-*.tmp")) == []


@pytest.mark.parametrize("existing_kind", ["regular", "symlink", "fifo"])
def test_serve_startup_token_file_refuses_every_existing_destination_kind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_kind: str,
) -> None:
    layout = ArchiveLayout(tmp_path)
    private_parent = tmp_path / "private"
    private_parent.mkdir(mode=0o700)
    private_parent.chmod(0o700)
    token_file = private_parent / "startup-token"
    target = private_parent / "symlink-target"
    if existing_kind == "regular":
        token_file.write_bytes(b"existing-regular\n")
    elif existing_kind == "symlink":
        target.write_bytes(b"existing-target\n")
        token_file.symlink_to(target)
    else:
        os.mkfifo(token_file, mode=0o600)
    token = "new-private-token-that-must-not-escape"
    server_started = False

    monkeypatch.setattr(cli_module, "_layout", lambda _archive_root: layout)
    monkeypatch.setattr(cli_module, "generate_access_token", lambda: token)
    monkeypatch.setattr(
        cli_module,
        "create_app",
        lambda _root, *, access_token, acceptance_read_only: object(),
    )

    def fake_run(*_args: object, **_kwargs: object) -> None:
        nonlocal server_started
        server_started = True

    monkeypatch.setattr(uvicorn, "run", fake_run)

    result = CliRunner().invoke(
        app,
        ["serve", "--archive-root", str(tmp_path), "--startup-token-file", str(token_file)],
    )

    assert result.exit_code == 2
    assert "already exists; refusing to overwrite" in " ".join(result.output.split())
    assert token not in result.output
    assert server_started is False
    if existing_kind == "regular":
        assert token_file.read_bytes() == b"existing-regular\n"
    elif existing_kind == "symlink":
        assert token_file.is_symlink()
        assert target.read_bytes() == b"existing-target\n"
    else:
        assert stat.S_ISFIFO(token_file.lstat().st_mode)
    assert list(private_parent.glob(".meta-archive-token-*.tmp")) == []


def test_serve_startup_token_file_loses_create_race_without_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = ArchiveLayout(tmp_path)
    private_parent = tmp_path / "private"
    private_parent.mkdir(mode=0o700)
    private_parent.chmod(0o700)
    token_file = private_parent / "startup-token"
    token = "token-that-must-lose-the-publication-race"

    monkeypatch.setattr(cli_module, "_layout", lambda _archive_root: layout)
    monkeypatch.setattr(cli_module, "generate_access_token", lambda: token)
    monkeypatch.setattr(
        cli_module,
        "create_app",
        lambda _root, *, access_token, acceptance_read_only: object(),
    )
    monkeypatch.setattr(
        uvicorn,
        "run",
        lambda *_args, **_kwargs: pytest.fail("server must not start after a token-file race"),
    )

    def competing_link(
        _source: str,
        destination: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
        follow_symlinks: bool,
    ) -> None:
        assert src_dir_fd == dst_dir_fd
        assert follow_symlinks is False
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=dst_dir_fd,
        )
        with os.fdopen(descriptor, "wb", buffering=0) as handle:
            handle.write(b"competing-writer\n")
        raise FileExistsError

    monkeypatch.setattr(cli_module.os, "link", competing_link)

    result = CliRunner().invoke(
        app,
        ["serve", "--archive-root", str(tmp_path), "--startup-token-file", str(token_file)],
    )

    assert result.exit_code == 2
    assert token_file.read_bytes() == b"competing-writer\n"
    assert token not in result.output
    assert list(private_parent.glob(".meta-archive-token-*.tmp")) == []


def test_serve_startup_token_file_requires_an_owner_only_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = ArchiveLayout(tmp_path)
    shared_parent = tmp_path / "shared"
    shared_parent.mkdir(mode=0o755)
    shared_parent.chmod(0o755)
    token_file = shared_parent / "startup-token"
    token = "token-that-must-not-enter-a-shared-directory"

    monkeypatch.setattr(cli_module, "_layout", lambda _archive_root: layout)
    monkeypatch.setattr(cli_module, "generate_access_token", lambda: token)
    monkeypatch.setattr(
        cli_module,
        "create_app",
        lambda _root, *, access_token, acceptance_read_only: object(),
    )
    monkeypatch.setattr(
        uvicorn,
        "run",
        lambda *_args, **_kwargs: pytest.fail("server must not start with an unsafe token parent"),
    )

    result = CliRunner().invoke(
        app,
        ["serve", "--archive-root", str(tmp_path), "--startup-token-file", str(token_file)],
    )

    assert result.exit_code == 2
    assert "owner-owned, owner-writable, and owner-only" in " ".join(result.output.split())
    assert token not in result.output
    assert not token_file.exists()


def test_serve_help_documents_acceptance_read_only_contract() -> None:
    result = CliRunner().invoke(app, ["serve", "--help"])

    assert result.exit_code == 0
    assert "--acceptance-read-only" in result.output
    assert "--startup-token-file" in result.output
    assert "catalog" in result.output
    assert "purge-recovery writes" in result.output
