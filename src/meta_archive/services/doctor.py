from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from meta_archive.paths import CANONICAL_DIRECTORIES, ArchiveLayout


@dataclass(slots=True)
class Check:
    name: str
    status: str
    detail: str
    required: bool = True


def _version(command: list[str]) -> str | None:
    executable = shutil.which(command[0])
    if executable is None:
        return None
    completed = subprocess.run(
        [executable, *command[1:]],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    output = (completed.stdout or completed.stderr).strip().splitlines()
    return output[0] if output else f"exit {completed.returncode}"


def _hardlink_check(root: Path) -> Check:
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="meta-archive-doctor-", dir=root) as directory:
        temporary = Path(directory)
        source = temporary / "source"
        destination = temporary / "link"
        source.write_bytes(b"hardlink-check")
        try:
            os.link(source, destination)
        except OSError as exc:
            return Check("filesystem hardlinks", "fail", str(exc))
        same_inode = source.stat().st_ino == destination.stat().st_ino
        return Check(
            "filesystem hardlinks",
            "pass" if same_inode else "fail",
            "same inode" if same_inode else "link did not share an inode",
        )


def _safe_json_object(path: Path) -> dict[str, object] | None:
    if path.is_symlink() or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _manifest_line_count(path: Path) -> int | None:
    if path.is_symlink() or not path.is_file():
        return None
    try:
        with path.open("rb") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return None


def _source_chunks_check(layout: ArchiveLayout) -> Check:
    chunks = layout.chunk_directories()
    if chunks:
        return Check("source chunks", "pass", f"{len(chunks)} source directories present")

    receipts_root = layout.manifests / "source-retirement"
    if receipts_root.is_symlink() or not receipts_root.is_dir():
        return Check(
            "source chunks",
            "fail",
            "0 source directories and no safe source-retirement receipt",
        )
    receipts = sorted(
        path
        for path in receipts_root.iterdir()
        if path.suffix == ".json" and path.is_file() and not path.is_symlink()
    )
    receipt_path = receipts[-1] if receipts else None
    receipt = _safe_json_object(receipt_path) if receipt_path is not None else None
    result = receipt.get("result") if receipt is not None else None
    if receipt is None or receipt.get("schema_version") != 1 or not isinstance(result, dict):
        return Check("source chunks", "fail", "source-retirement receipt is missing or invalid")

    requested = result.get("requested_chunks")
    manifest_file_count = result.get("manifest_file_count")
    requested_chunks = (
        [name for name in requested if isinstance(name, str)] if isinstance(requested, list) else []
    )
    requested_is_safe = (
        isinstance(requested, list)
        and bool(requested_chunks)
        and len(requested_chunks) == len(requested)
        and all(
            name not in {"", ".", ".."}
            and name not in CANONICAL_DIRECTORIES
            and "/" not in name
            and "\\" not in name
            for name in requested_chunks
        )
        and len(set(requested_chunks)) == len(requested_chunks)
    )
    if (
        result.get("applied") is not True
        or not requested_is_safe
        or not isinstance(manifest_file_count, int)
        or manifest_file_count <= 0
        or any((layout.root / name).exists() for name in requested_chunks)
    ):
        return Check("source chunks", "fail", "source-retirement receipt is incomplete or stale")

    original_count = _manifest_line_count(layout.original_files_manifest)
    relocation_count = _manifest_line_count(layout.relocation_manifest)
    directory_count = _manifest_line_count(layout.original_directories_manifest)
    if (
        original_count != manifest_file_count
        or relocation_count != manifest_file_count
        or directory_count is None
        or directory_count < len(requested_chunks)
    ):
        return Check(
            "source chunks",
            "fail",
            "source-retirement receipt no longer matches the complete source manifests",
        )

    report = _safe_json_object(layout.manifests / "integrity-reports" / "latest.json")
    gate = report.get("gate") if report is not None else None
    checks = report.get("checks") if report is not None else None
    current_gate_passes = (
        isinstance(gate, dict)
        and gate.get("all_mandatory_checks_pass") is True
        and gate.get("deep_verified") is True
        and gate.get("library_verified") is True
        and gate.get("required_artifacts_present") is True
        and isinstance(checks, list)
        and bool(checks)
        and all(
            isinstance(check, dict)
            and (check.get("mandatory") is False or check.get("status") == "pass")
            for check in checks
        )
    )
    if not current_gate_passes:
        return Check(
            "source chunks",
            "fail",
            "source directories are retired but the current deep-integrity gate is not all-pass",
        )

    assert receipt_path is not None
    return Check(
        "source chunks",
        "pass",
        (
            f"{len(requested_chunks)} source directories safely retired; "
            f"{manifest_file_count} files remain reconstructable; "
            f"receipt {receipt_path.relative_to(layout.root)}"
        ),
    )


def _private_permissions_check(layout: ArchiveLayout) -> Check:
    findings = layout.private_permission_findings()
    if not findings:
        return Check(
            "private runtime permissions",
            "pass",
            "archive runtime directories are owner-only; mutable runtime files are private",
            False,
        )
    preview = "; ".join(findings[:5])
    remaining = len(findings) - min(5, len(findings))
    suffix = f"; plus {remaining} more" if remaining else ""
    return Check(
        "private runtime permissions",
        "warn",
        f"{preview}{suffix}; run an archive command to repair owner-only modes",
        False,
    )


def run_doctor(layout: ArchiveLayout) -> list[dict[str, object]]:
    checks: list[Check] = []
    checks.append(
        Check(
            "archive root",
            "pass" if layout.root.is_dir() else "fail",
            str(layout.root),
        )
    )
    checks.append(_private_permissions_check(layout))
    checks.append(_source_chunks_check(layout))
    try:
        connection = sqlite3.connect(":memory:")
        connection.execute("CREATE VIRTUAL TABLE test_fts USING fts5(content)")
        connection.close()
        checks.append(Check("SQLite FTS5", "pass", sqlite3.sqlite_version))
    except sqlite3.Error as exc:
        checks.append(Check("SQLite FTS5", "fail", str(exc)))

    user_node = Path.home() / ".local" / "share" / "pnpm" / "bin" / "node"
    node_command = str(user_node) if user_node.is_file() else "node"
    command_checks = [
        ("Python", ["python3", "--version"], True),
        ("uv", ["uv", "--version"], True),
        ("Node 24 runtime", [node_command, "--version"], True),
        ("pnpm", ["pnpm", "--version"], True),
        ("FFmpeg", ["ffmpeg", "-version"], False),
        ("FFprobe", ["ffprobe", "-version"], False),
        ("Bubblewrap", ["bwrap", "--version"], False),
        ("file/libmagic", ["file", "--version"], True),
    ]
    for name, command, required in command_checks:
        version = _version(command)
        version_is_acceptable = version is not None
        if name == "Node 24 runtime" and version is not None:
            version_is_acceptable = version.lstrip("v").split(".", maxsplit=1)[0] == "24"
        checks.append(
            Check(
                name,
                "pass" if version_is_acceptable else ("fail" if required else "warn"),
                version or "not found on PATH",
                required,
            )
        )
    checks.append(_hardlink_check(layout.root))
    usage = shutil.disk_usage(layout.root)
    checks.append(
        Check(
            "free disk space",
            "pass" if usage.free >= 100 * 1024**3 else "warn",
            f"{usage.free / 1024**3:.1f} GiB free",
            False,
        )
    )
    return [asdict(check) for check in checks]
