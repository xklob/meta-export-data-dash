from __future__ import annotations

import json
import os
import secrets
import stat
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Annotated, Any, cast

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy import text
from sqlalchemy.engine import Engine

from meta_archive.api.analytics_routes import (
    activity_patterns,
    analytics_overview,
    attachment_analytics,
    interaction_patterns,
    interaction_series,
    message_series,
)
from meta_archive.api.main import create_app, generate_access_token
from meta_archive.config import settings
from meta_archive.database import CatalogMigrationError, create_catalog_engine, run_migrations
from meta_archive.deletion_cli import deletion_app
from meta_archive.paths import ArchiveLayout, ArchiveNotFoundError
from meta_archive.services.activity_importer import import_activities_locked
from meta_archive.services.analytics_cache import clear_analytics_cache
from meta_archive.services.attachment_text import (
    BubblewrapAttachmentExtractor,
    build_attachment_text,
)
from meta_archive.services.conflict_aggregates import build_conflict_aggregates
from meta_archive.services.conflict_scoring import CONFLICT_LABELS, conflict_search
from meta_archive.services.consolidation import (
    build_blob_store_locked,
    conflict_holding_lock,
    materialize_conflict_holding_area_locked,
    reconstruct_source,
)
from meta_archive.services.doctor import run_doctor
from meta_archive.services.inventory import inventory_summary, scan_archive, scan_archive_locked
from meta_archive.services.library import (
    LibraryBuildError,
    build_filesystem_library,
    library_build_lock,
)
from meta_archive.services.media_derivatives import FfmpegMediaRunner, build_media_derivatives
from meta_archive.services.message_importer import import_messages_locked
from meta_archive.services.nlp_annotations import (
    NlpTask,
    build_nlp_annotations,
    conflict_aggregation_status,
)
from meta_archive.services.search import SearchFilters, SearchUnavailableError
from meta_archive.services.search_index import (
    SemanticIndexManualActivationError,
    activate_rejected_semantic_candidate,
    build_semantic_index,
    queue_semantic_index_job,
)
from meta_archive.services.source_import_lock import source_import_publication_lock
from meta_archive.services.source_retirement import retire_source_chunks
from meta_archive.services.support_library_scrub import (
    SupportLibraryScrubError,
    scrub_support_library,
)
from meta_archive.services.topic_clusters import build_topic_clusters
from meta_archive.services.transcription import build_audio_transcriptions
from meta_archive.services.verification import verify_catalog, write_verification_report

app = typer.Typer(
    name="meta-archive",
    help="Consolidate, verify, index, and browse a local Facebook archive export.",
    no_args_is_help=True,
)
app.add_typer(deletion_app)
console = Console()

ArchiveRootOption = Annotated[
    Path | None,
    typer.Option(
        "--archive-root",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        help="Export root. Defaults to the single meta-backup* directory in the working directory.",
    ),
]


class StartupTokenFileError(RuntimeError):
    """A startup-token file could not be published without weakening its privacy."""


@contextmanager
def _startup_token_file_publication(path: Path, token: str) -> Iterator[None]:
    """Publish ``token`` for exactly the lifetime of the yielded server scope.

    The destination's parent must already be an owner-only directory owned by this process's
    effective user.  A held no-follow directory descriptor binds every create/link/unlink to that
    validated directory, and hard-link publication provides create-if-absent semantics without a
    check/use overwrite race.  Cleanup compares the destination's device/inode with the private
    temporary inode, so it cannot unlink a path another writer replaced while the server ran.
    """

    if not all(hasattr(os, name) for name in ("O_DIRECTORY", "O_NOFOLLOW")) or not hasattr(
        os, "geteuid"
    ):
        raise StartupTokenFileError("Secure startup-token file creation is unavailable")

    destination = Path(os.path.abspath(path.expanduser()))
    leaf = destination.name
    if not leaf or leaf in {".", ".."}:
        raise StartupTokenFileError("Startup-token file must name a file in an existing directory")

    directory_flags = (
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | int(getattr(os, "O_CLOEXEC", 0))
    )
    file_flags = (
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | int(getattr(os, "O_CLOEXEC", 0))
    )
    parent_descriptor: int | None = None
    temporary_descriptor: int | None = None
    temporary_name: str | None = None
    temporary_stat: os.stat_result | None = None
    try:
        try:
            parent_before = os.stat(destination.parent, follow_symlinks=False)
            if not stat.S_ISDIR(parent_before.st_mode) or stat.S_ISLNK(parent_before.st_mode):
                raise StartupTokenFileError(
                    "Startup-token file parent must be an existing non-symlink directory"
                )
            parent_descriptor = os.open(destination.parent, directory_flags)
            parent_opened = os.fstat(parent_descriptor)
            if (parent_opened.st_dev, parent_opened.st_ino) != (
                parent_before.st_dev,
                parent_before.st_ino,
            ):
                raise StartupTokenFileError("Startup-token file parent changed during validation")
            parent_mode = stat.S_IMODE(parent_opened.st_mode)
            if (
                parent_opened.st_uid != os.geteuid()
                or parent_mode & 0o077
                or parent_mode & 0o300 != 0o300
            ):
                raise StartupTokenFileError(
                    "Startup-token file parent must be owner-owned, owner-writable, and owner-only"
                )

            for _attempt in range(16):
                temporary_name = f".meta-archive-token-{secrets.token_hex(16)}.tmp"
                try:
                    temporary_descriptor = os.open(
                        temporary_name,
                        file_flags,
                        0o600,
                        dir_fd=parent_descriptor,
                    )
                    break
                except FileExistsError:
                    temporary_name = None
            else:
                raise StartupTokenFileError("Could not allocate a private temporary token file")

            os.fchmod(temporary_descriptor, 0o600)
            temporary_stat = os.fstat(temporary_descriptor)
            if (
                not stat.S_ISREG(temporary_stat.st_mode)
                or stat.S_IMODE(temporary_stat.st_mode) != 0o600
            ):
                raise StartupTokenFileError(
                    "Temporary startup-token file is not a private regular file"
                )
            with os.fdopen(temporary_descriptor, "wb", buffering=0) as handle:
                temporary_descriptor = None
                handle.write(f"{token}\n".encode())
                os.fsync(handle.fileno())

            os.link(
                temporary_name,
                leaf,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            published_stat = os.stat(leaf, dir_fd=parent_descriptor, follow_symlinks=False)
            if (
                not stat.S_ISREG(published_stat.st_mode)
                or stat.S_IMODE(published_stat.st_mode) != 0o600
                or (published_stat.st_dev, published_stat.st_ino)
                != (temporary_stat.st_dev, temporary_stat.st_ino)
            ):
                raise StartupTokenFileError("Startup-token file publication failed validation")

            os.unlink(temporary_name, dir_fd=parent_descriptor)
            temporary_name = None
            parent_after = os.stat(destination.parent, follow_symlinks=False)
            if (parent_after.st_dev, parent_after.st_ino) != (
                parent_opened.st_dev,
                parent_opened.st_ino,
            ):
                raise StartupTokenFileError("Startup-token file parent changed during publication")
            os.fsync(parent_descriptor)
        except FileExistsError as exc:
            raise StartupTokenFileError(
                "Startup-token file already exists; refusing to overwrite it"
            ) from exc
        except StartupTokenFileError:
            raise
        except (OSError, UnicodeError, ValueError) as exc:
            raise StartupTokenFileError("Could not securely create the startup-token file") from exc

        yield
    finally:
        if temporary_descriptor is not None:
            os.close(temporary_descriptor)
        if parent_descriptor is not None:
            try:
                directory_changed = False
                if temporary_stat is not None:
                    try:
                        destination_stat = os.stat(
                            leaf,
                            dir_fd=parent_descriptor,
                            follow_symlinks=False,
                        )
                        if (destination_stat.st_dev, destination_stat.st_ino) == (
                            temporary_stat.st_dev,
                            temporary_stat.st_ino,
                        ):
                            os.unlink(leaf, dir_fd=parent_descriptor)
                            directory_changed = True
                    except FileNotFoundError:
                        pass
                if temporary_name is not None:
                    try:
                        os.unlink(temporary_name, dir_fd=parent_descriptor)
                        directory_changed = True
                    except FileNotFoundError:
                        pass
                if directory_changed:
                    os.fsync(parent_descriptor)
            finally:
                os.close(parent_descriptor)


def _layout(archive_root: Path | None) -> ArchiveLayout:
    try:
        return ArchiveLayout.discover(archive_root)
    except ArchiveNotFoundError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(2) from exc


def _catalog(layout: ArchiveLayout) -> Engine:
    # Migration and SQLite runtime-file preparation are catalog mutations too. Serialize them
    # before returning the engine; command-specific source work takes a fresh lease afterward and
    # revalidates any deletion state that may have changed in the handoff gap.
    with source_import_publication_lock(layout):
        layout.ensure_runtime_directories()
        run_migrations(layout.database)
        return create_catalog_engine(layout.database)


def _progress(stage: str, current: int, total: int | None) -> None:
    if total:
        console.print(f"[cyan]{stage}[/cyan] {current:,}/{total:,}")
    else:
        console.print(f"[cyan]{stage}[/cyan] {current:,}")


def _mandatory_non_pass(checks: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return mandatory checks that are not yet safe to accept."""

    return [
        check
        for check in checks
        if check.get("mandatory", True) is not False and check.get("status") != "pass"
    ]


@app.command("warm-analytics")
def warm_analytics_command(
    archive_root: ArchiveRootOption = None,
    timezone: Annotated[
        str,
        typer.Option(
            "--timezone",
            help="IANA timezone used by the default dashboard cache.",
        ),
    ] = "America/New_York",
) -> None:
    """Precompute revision-bound responses for the initial analytics dashboard."""

    layout = _layout(archive_root)
    engine = _catalog(layout)
    removed = clear_analytics_cache(layout)
    request = cast(
        object,
        SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(layout=layout, engine=engine)),
            state=SimpleNamespace(),
        ),
    )
    operations: tuple[tuple[str, Callable[[], dict[str, Any]]], ...] = (
        (
            "overview",
            lambda: analytics_overview(cast(Any, request), timezone=timezone),
        ),
        (
            "messages-month",
            lambda: message_series(
                cast(Any, request),
                bucket="month",
                timezone=timezone,
                from_ms=None,
                to_ms=None,
                conversation_id=None,
                limit=120,
                cursor=None,
            ),
        ),
        (
            "activity-patterns",
            lambda: activity_patterns(
                cast(Any, request),
                timezone=timezone,
                from_ms=None,
                to_ms=None,
                limit=120,
                cursor=None,
            ),
        ),
        (
            "interaction-patterns-month",
            lambda: interaction_patterns(
                cast(Any, request),
                bucket="month",
                timezone=timezone,
                from_ms=None,
                to_ms=None,
                limit=120,
                cursor=None,
            ),
        ),
        (
            "interaction-series-month",
            lambda: interaction_series(
                cast(Any, request),
                bucket="month",
                timezone=timezone,
                from_ms=None,
                to_ms=None,
                domain_limit=20,
                limit=120,
                cursor=None,
            ),
        ),
        (
            "attachments-month",
            lambda: attachment_analytics(
                cast(Any, request),
                bucket="month",
                timezone=timezone,
                from_ms=None,
                to_ms=None,
                limit=120,
                cursor=None,
            ),
        ),
    )
    timings: dict[str, float] = {}
    for label, operation in operations:
        started = perf_counter()
        operation()
        timings[label] = round(perf_counter() - started, 3)
    console.print_json(
        json.dumps(
            {
                "cleared_entries": removed,
                "timezone": timezone,
                "cached_responses": len(operations),
                "generation_seconds": timings,
            }
        )
    )


@app.command()
def doctor(archive_root: ArchiveRootOption = None, output_json: bool = False) -> None:
    """Check the local toolchain and archive without changing source files."""

    layout = _layout(archive_root)
    checks = run_doctor(layout)
    if output_json:
        console.print_json(json.dumps({"checks": checks}))
    else:
        table = Table("Check", "Status", "Detail")
        for check in checks:
            color = {"pass": "green", "warn": "yellow", "fail": "red"}[str(check["status"])]
            table.add_row(
                str(check["name"]),
                f"[{color}]{check['status']}[/{color}]",
                str(check["detail"]),
            )
        console.print(table)
    if any(check["status"] == "fail" and check["required"] for check in checks):
        raise typer.Exit(1)


@app.command()
def scan(
    archive_root: ArchiveRootOption = None,
    hash_files: Annotated[
        bool,
        typer.Option("--hash/--no-hash", help="SHA-256 every changed physical source file."),
    ] = False,
    output_json: bool = False,
) -> None:
    """Inventory source files. Re-running is idempotent and resumes from stored hashes."""

    layout = _layout(archive_root)
    engine = _catalog(layout)
    summary = scan_archive(
        layout,
        engine,
        hash_files=hash_files,
        progress=None if output_json else _progress,
    )
    payload = asdict(summary)
    if output_json:
        console.print_json(json.dumps(payload))
    else:
        console.print_json(json.dumps(payload))


@app.command()
def consolidate(
    archive_root: ArchiveRootOption = None,
    apply: Annotated[
        bool,
        typer.Option(
            "--apply/--dry-run",
            help=(
                "Apply the blob-store build, or explicitly run the non-mutating "
                "hash/parity preview."
            ),
        ),
    ] = False,
) -> None:
    """Hash every source copy, report deduplication, and optionally materialize blobs."""

    layout = _layout(archive_root)
    engine = _catalog(layout)
    with source_import_publication_lock(layout) as source_lock:
        summary = scan_archive_locked(
            layout,
            engine,
            lock=source_lock,
            hash_files=True,
            progress=_progress,
        )
        console.print_json(json.dumps(asdict(summary)))
        if summary.conflicting_collision_groups:
            if apply:
                with conflict_holding_lock(layout) as conflict_lock:
                    holding = materialize_conflict_holding_area_locked(
                        layout,
                        engine,
                        source_lock=source_lock,
                        lock=conflict_lock,
                        progress=_progress,
                    )
                console.print_json(json.dumps({"conflict_holding": asdict(holding)}))
                console.print(
                    "[red]Conflicting normalized paths were copied into the append-only conflict "
                    "holding area. Blob-store promotion is disabled until they are reviewed.[/red]"
                )
            else:
                console.print(
                    "[red]Dry run found conflicting normalized paths. No holding-area files were "
                    "created; rerun with --apply to preserve every variant for review.[/red]"
                )
            raise typer.Exit(1)
        if not apply:
            console.print(
                "[yellow]Dry run complete. Re-run with --apply to build independent blobs.[/yellow]"
            )
            return
        materialized, reused = build_blob_store_locked(
            layout,
            engine,
            lock=source_lock,
            progress=_progress,
        )
    console.print(
        f"[green]Blob store ready:[/green] {materialized:,} materialized or repaired, "
        f"{reused:,} reused"
    )


@app.command()
def ingest(
    archive_root: ArchiveRootOption = None,
    owner_name: Annotated[
        str | None,
        typer.Option(
            "--owner-name",
            help=(
                "Archive owner's exact Facebook display name. Persisted in the private catalog; "
                "an existing setting is reused when omitted."
            ),
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Reparse source files even when parser and digest match."),
    ] = False,
    messages: Annotated[
        bool,
        typer.Option("--messages/--no-messages", help="Import conversation and support messages."),
    ] = True,
    activities: Annotated[
        bool,
        typer.Option("--activities/--no-activities", help="Import non-message activity JSON."),
    ] = True,
    output_json: bool = False,
) -> None:
    """Import every supported JSON schema and explicitly classify unsupported JSON."""

    if not messages and not activities:
        console.print("[red]At least one importer must be enabled.[/red]")
        raise typer.Exit(2)
    layout = _layout(archive_root)
    engine = _catalog(layout)
    with source_import_publication_lock(layout) as source_lock:
        # Keep scan, both importers, FTS optimization, and the resulting verification receipt on
        # one source snapshot. The explicit locked entry points avoid reacquiring this flock.
        scan_summary = scan_archive_locked(
            layout,
            engine,
            lock=source_lock,
            hash_files=False,
            progress=None,
        )
        payload: dict[str, object] = {"scan": asdict(scan_summary)}
        if messages:
            payload["messages"] = asdict(
                import_messages_locked(
                    layout,
                    engine,
                    lock=source_lock,
                    owner_name=owner_name,
                    force=force,
                    progress=None if output_json else _progress,
                )
            )
        if activities:
            payload["activities"] = asdict(
                import_activities_locked(
                    layout,
                    engine,
                    lock=source_lock,
                    force=force,
                    progress=None if output_json else _progress,
                )
            )
        with engine.begin() as connection:
            connection.execute(text("INSERT INTO message_fts(message_fts) VALUES ('optimize')"))
            connection.execute(text("INSERT INTO activity_fts(activity_fts) VALUES ('optimize')"))
        checks = verify_catalog(layout, engine)
        report = write_verification_report(layout, checks)
    payload["verification_report"] = str(report)
    payload["failed_checks"] = [str(check["key"]) for check in _mandatory_non_pass(checks)]
    console.print_json(json.dumps(payload))
    if payload["failed_checks"]:
        raise typer.Exit(1)


@app.command("build-library")
def build_library_command(
    archive_root: ArchiveRootOption = None,
    output_json: bool = False,
) -> None:
    """Atomically rebuild the human-readable filesystem library from the catalog."""

    layout = _layout(archive_root)
    engine = _catalog(layout)
    with (
        source_import_publication_lock(layout) as source_lock,
        library_build_lock(layout),
    ):
        summary = build_filesystem_library(
            layout,
            engine,
            progress=None if output_json else _progress,
            lock_held=True,
            source_lock=source_lock,
        )
        checks = verify_catalog(layout, engine)
        report = write_verification_report(layout, checks)
    payload = {
        "library": asdict(summary),
        "verification_report": str(report),
        "non_pass_checks": [str(check["key"]) for check in _mandatory_non_pass(checks)],
    }
    console.print_json(json.dumps(payload))
    if payload["non_pass_checks"]:
        raise typer.Exit(1)


@app.command("scrub-support-library")
def scrub_support_library_command(
    archive_root: ArchiveRootOption = None,
) -> None:
    """Repair legacy sensitive metadata in existing support transcript NDJSON files."""

    layout = _layout(archive_root)
    engine = _catalog(layout)
    try:
        summary = scrub_support_library(layout, engine)
    except (LibraryBuildError, SupportLibraryScrubError):
        console.print("[red]Error:[/red] Support transcript repair failed safely.")
        raise typer.Exit(1) from None
    console.print_json(json.dumps(asdict(summary)))


@app.command("build-derivatives")
def build_derivatives_command(
    archive_root: ArchiveRootOption = None,
    ffmpeg: Annotated[
        Path | None,
        typer.Option(
            "--ffmpeg",
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help="Explicit user-local ffmpeg executable. Defaults to ffmpeg on PATH.",
        ),
    ] = None,
    retry_failed: Annotated[
        bool,
        typer.Option(
            "--retry-failed/--skip-failed",
            help="Retry previously failed or interrupted unique-blob derivatives.",
        ),
    ] = True,
    output_json: bool = False,
) -> None:
    """Build local thumbnails, video posters, durations, and audio waveforms."""

    layout = _layout(archive_root)
    engine = _catalog(layout)
    runner = FfmpegMediaRunner(ffmpeg)
    summary = build_media_derivatives(
        layout,
        engine,
        runner=runner,
        retry_failed=retry_failed,
        progress=None if output_json else _progress,
    )
    console.print_json(json.dumps(asdict(summary)))
    if summary.failed or summary.skipped_failed:
        raise typer.Exit(1)


@app.command("transcribe-audio")
def transcribe_audio_command(
    archive_root: ArchiveRootOption = None,
    retry_failed: Annotated[
        bool,
        typer.Option(
            "--retry-failed/--skip-failed",
            help="Retry failed/interrupted unique audio blobs.",
        ),
    ] = True,
) -> None:
    """Transcribe unique audio blobs locally with multilingual Whisper small CPU INT8."""

    layout = _layout(archive_root)
    engine = _catalog(layout)
    summary = build_audio_transcriptions(
        layout,
        engine,
        retry_failed=retry_failed,
        progress=_progress,
    )
    console.print_json(json.dumps(asdict(summary)))
    if summary.failed or summary.skipped_failed:
        raise typer.Exit(1)


@app.command("extract-attachment-text")
def extract_attachment_text_command(
    archive_root: ArchiveRootOption = None,
    retry_failed: Annotated[
        bool,
        typer.Option(
            "--retry-failed/--skip-failed",
            help="Retry failed/interrupted unique attachment blobs.",
        ),
    ] = True,
) -> None:
    """Extract safe document/archive text in a no-network Bubblewrap sandbox."""

    layout = _layout(archive_root)
    engine = _catalog(layout)
    try:
        extractor = BubblewrapAttachmentExtractor()
    except RuntimeError as exc:
        console.print(f"[red]Attachment extraction is disabled:[/red] {exc}")
        raise typer.Exit(2) from exc
    summary = build_attachment_text(
        layout,
        engine,
        extractor=extractor,
        retry_failed=retry_failed,
        progress=_progress,
    )
    console.print_json(json.dumps(asdict(summary)))
    if summary.failed or summary.skipped_failed:
        raise typer.Exit(1)


@app.command("annotate-text")
def annotate_text_command(
    archive_root: ArchiveRootOption = None,
    task: Annotated[
        list[str] | None,
        typer.Option(
            "--task",
            help="Repeat for sentiment, emotion, or toxicity; defaults to all three.",
        ),
    ] = None,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            min=1,
            max=32,
            help="CPU classifier batch size; 16 is conservative and 32 may be faster with RAM.",
        ),
    ] = 16,
) -> None:
    """Create revision-pinned local sentiment, emotion, and toxicity probability vectors."""

    selected = tuple(task or ("sentiment", "emotion", "toxicity"))
    supported = {"sentiment", "emotion", "toxicity"}
    if any(value not in supported for value in selected):
        console.print("[red]--task must be sentiment, emotion, or toxicity.[/red]")
        raise typer.Exit(2)
    layout = _layout(archive_root)
    engine = _catalog(layout)
    summary = build_nlp_annotations(
        layout,
        engine,
        tasks=tuple(cast(NlpTask, value) for value in selected),
        batch_size=batch_size,
        progress=_progress,
    )
    console.print_json(json.dumps(asdict(summary)))
    if summary.failed:
        raise typer.Exit(1)


@app.command("build-topic-clusters")
def build_topic_clusters_command(
    archive_root: ArchiveRootOption = None,
    cluster_count: Annotated[
        int,
        typer.Option("--clusters", min=1, max=50, help="Requested topic-cluster count."),
    ] = 12,
    sample_limit: Annotated[
        int,
        typer.Option(
            "--sample-limit",
            min=100,
            max=100_000,
            help="Maximum evenly sampled semantic windows used for clustering.",
        ),
    ] = 20_000,
) -> None:
    """Build local, bounded topic clusters from the active semantic-text index."""

    layout = _layout(archive_root)
    engine = _catalog(layout)
    try:
        summary = build_topic_clusters(
            layout,
            engine,
            requested_clusters=cluster_count,
            sample_limit=sample_limit,
            progress=_progress,
        )
    except (RuntimeError, ValueError) as exc:
        console.print(f"[red]Topic clustering failed:[/red] {exc}")
        raise typer.Exit(2) from exc
    console.print_json(json.dumps(asdict(summary)))


@app.command("search-conflict")
def search_conflict_command(
    query: Annotated[str, typer.Argument(help="Semantic query used to retrieve message windows.")],
    archive_root: ArchiveRootOption = None,
    label: Annotated[
        str,
        typer.Option(
            "--label",
            help="NLI ranking target: direct conflict, calm disagreement, banter, "
            "third-party conflict, no conflict, or uncertain.",
        ),
    ] = "direct conflict",
    conversation_id: Annotated[
        str | None,
        typer.Option("--conversation-id", help="Restrict retrieval to one conversation."),
    ] = None,
    participant: Annotated[
        str | None,
        typer.Option("--participant", help="Restrict retrieval by participant name."),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", min=1, max=100)] = 25,
) -> None:
    """Rank semantic message-window candidates with local ModernBERT NLI hypotheses."""

    if label not in CONFLICT_LABELS:
        console.print(f"[red]--label must be one of: {', '.join(CONFLICT_LABELS)}.[/red]")
        raise typer.Exit(2)
    layout = _layout(archive_root)
    engine = _catalog(layout)
    try:
        page = conflict_search(
            layout,
            engine,
            query,
            target_label=label,
            filters=SearchFilters(
                entity_type="message",
                conversation_id=conversation_id,
                participant=participant,
            ),
            limit=limit,
        )
    except SearchUnavailableError as exc:
        console.print(f"[red]Conflict search is unavailable:[/red] {exc}")
        raise typer.Exit(2) from exc
    payload = {
        **asdict(page),
        "query": query,
        "target_label": label,
        "aggregation": conflict_aggregation_status(engine),
    }
    console.print_json(json.dumps(payload, default=str))


@app.command("build-conflict-aggregates")
def build_conflict_aggregates_command(
    archive_root: ArchiveRootOption = None,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            min=1,
            max=100,
            help="CPU scorer batch size; full-corpus candidates remain bounded and resumable.",
        ),
    ] = 32,
) -> None:
    """Build time-only conflict aggregates after the held-out calibration gate passes."""

    layout = _layout(archive_root)
    engine = _catalog(layout)
    try:
        summary = build_conflict_aggregates(
            layout,
            engine,
            batch_size=batch_size,
            progress=_progress,
        )
    except (RuntimeError, ValueError) as exc:
        console.print(f"[red]Conflict aggregation is unavailable:[/red] {exc}")
        raise typer.Exit(2) from exc
    console.print_json(json.dumps(asdict(summary)))


@app.command("verify")
def verify_command(
    archive_root: ArchiveRootOption = None,
    output_json: bool = False,
    deep: Annotated[
        bool,
        typer.Option("--deep", help="Re-hash every original file or retained blob."),
    ] = False,
) -> None:
    """Run catalog, manifest, provenance, and reconciliation checks."""

    layout = _layout(archive_root)
    engine = _catalog(layout)
    with source_import_publication_lock(layout):
        checks = verify_catalog(
            layout,
            engine,
            deep=deep,
            progress=None if output_json else _progress,
        )
        report = write_verification_report(layout, checks)
    if output_json:
        console.print_json(json.dumps({"checks": checks, "report": str(report)}))
    else:
        table = Table("Invariant", "Status", "Expected", "Actual", "Detail")
        for check in checks:
            color = {"pass": "green", "pending": "yellow", "fail": "red"}[str(check["status"])]
            table.add_row(
                str(check["key"]),
                f"[{color}]{check['status']}[/{color}]",
                str(check["expected"]),
                str(check["actual"]),
                str(check["detail"]),
            )
        console.print(table)
        console.print(f"Report: {report}")
    if _mandatory_non_pass(checks):
        raise typer.Exit(1)


@app.command("reconstruct-source")
def reconstruct_source_command(
    target: Annotated[Path, typer.Argument(file_okay=False, resolve_path=True)],
    archive_root: ArchiveRootOption = None,
) -> None:
    """Recreate the original chunk tree from the relocation manifest and blob store."""

    layout = _layout(archive_root)
    count = reconstruct_source(layout, target)
    console.print(f"[green]Reconstructed {count:,} files under {target}[/green]")


@app.command("retire-source")
def retire_source_command(
    chunk: Annotated[
        list[str],
        typer.Option(
            "--chunk",
            help="Exact source-chunk root name. Repeat for each intended target.",
        ),
    ],
    archive_root: ArchiveRootOption = None,
    apply: Annotated[
        bool,
        typer.Option(
            "--apply",
            help="Unlink the exact requested sources after every retirement gate passes.",
        ),
    ] = False,
) -> None:
    """Preview or perform fail-closed retirement of explicitly named source chunks."""

    layout = _layout(archive_root)
    engine = _catalog(layout)
    result = retire_source_chunks(
        layout,
        engine,
        chunk,
        apply=apply,
        progress=_progress,
    )
    console.print_json(json.dumps(asdict(result)))


@app.command()
def index(archive_root: ArchiveRootOption = None) -> None:
    """Rebuild and optimize the SQLite keyword indexes."""

    layout = _layout(archive_root)
    engine = _catalog(layout)
    with engine.begin() as connection:
        connection.execute(text("INSERT INTO message_fts(message_fts) VALUES ('rebuild')"))
        connection.execute(text("INSERT INTO activity_fts(activity_fts) VALUES ('rebuild')"))
        connection.execute(text("INSERT INTO message_fts(message_fts) VALUES ('optimize')"))
        connection.execute(text("INSERT INTO activity_fts(activity_fts) VALUES ('optimize')"))
    console.print("[green]Keyword indexes rebuilt.[/green]")


@app.command("index-semantic")
def index_semantic(
    archive_root: ArchiveRootOption = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            help="Embedding model alias. Default/auto: bge-base-en-v1.5.",
        ),
    ] = None,
    rebuild_windows: Annotated[
        bool,
        typer.Option(
            "--rebuild-windows/--reuse-windows",
            help="Rebuild deterministic message/activity windows before embedding.",
        ),
    ] = True,
) -> None:
    """Build or resume the local LanceDB semantic-text index.

    Rejected complete candidates remain inactive. An explicit audited override uses
    `activate-semantic-candidate CONFIG_ID --acknowledge-failed-quality-gate`.
    """

    layout = _layout(archive_root)
    engine = _catalog(layout)
    job_id = queue_semantic_index_job(
        engine,
        model_alias=model,
        rebuild_windows=rebuild_windows,
    )
    try:
        summary = build_semantic_index(
            layout,
            engine,
            model_alias=model,
            rebuild_windows=rebuild_windows,
            job_id=job_id,
        )
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(2) from exc
    except Exception as exc:
        console.print(f"[red]Semantic index job {job_id} failed:[/red] {type(exc).__name__}: {exc}")
        raise typer.Exit(1) from None
    console.print_json(json.dumps(asdict(summary)))


@app.command("activate-semantic-candidate")
def activate_semantic_candidate(
    config_id: Annotated[
        str,
        typer.Argument(
            help="Exact 64-character config ID of one complete rejected semantic candidate."
        ),
    ],
    archive_root: ArchiveRootOption = None,
    acknowledge_failed_quality_gate: Annotated[
        bool,
        typer.Option(
            "--acknowledge-failed-quality-gate",
            help=(
                "Acknowledge that this candidate failed the automatic retrieval quality gate "
                "and activate it anyway."
            ),
        ),
    ] = False,
) -> None:
    """Manually activate one rejected semantic candidate through an audited CLI-only path."""

    layout = _layout(archive_root)
    engine = _catalog(layout)
    try:
        summary = activate_rejected_semantic_candidate(
            layout,
            engine,
            config_hash=config_id,
            acknowledge_failed_quality_gate=acknowledge_failed_quality_gate,
        )
    except (SemanticIndexManualActivationError, ValueError) as exc:
        console.print(f"[red]Manual semantic activation refused:[/red] {exc}")
        raise typer.Exit(2) from None
    except Exception as exc:
        console.print(f"[red]Manual semantic activation failed:[/red] {type(exc).__name__}: {exc}")
        raise typer.Exit(1) from None
    console.print_json(json.dumps(asdict(summary)))


@app.command()
def status(archive_root: ArchiveRootOption = None) -> None:
    """Print the current catalog summary."""

    layout = _layout(archive_root)
    engine = _catalog(layout)
    summary = inventory_summary(engine, layout)
    console.print_json(json.dumps(asdict(summary)))


@app.command()
def serve(
    archive_root: ArchiveRootOption = None,
    host: str = settings.host,
    port: Annotated[int, typer.Option(min=1, max=65535)] = settings.port,
    acceptance_read_only: Annotated[
        bool,
        typer.Option(
            "--acceptance-read-only",
            help=(
                "Serve an already-migrated archive without any catalog, cache, permission, "
                "or purge-recovery writes."
            ),
        ),
    ] = False,
    startup_token_file: Annotated[
        Path | None,
        typer.Option(
            "--startup-token-file",
            dir_okay=False,
            resolve_path=False,
            help=(
                "Atomically create a new owner-only file containing the ephemeral access token; "
                "suppress the token-bearing URL from stdout."
            ),
        ),
    ] = None,
) -> None:
    """Serve the authenticated dashboard on localhost."""

    if host != "127.0.0.1":
        console.print("[red]The dashboard may only bind to 127.0.0.1.[/red]")
        raise typer.Exit(2)
    import uvicorn

    layout = _layout(archive_root)
    token = generate_access_token()
    try:
        application = create_app(
            layout.root,
            access_token=token,
            acceptance_read_only=acceptance_read_only,
        )
    except CatalogMigrationError as exc:
        console.print(f"[red]Acceptance server refused to start:[/red] {exc}")
        raise typer.Exit(2) from exc

    def run_server() -> None:
        console.print("[green]Meta Archive Explorer is starting.[/green]")
        if acceptance_read_only:
            console.print(
                "[cyan]Acceptance read-only mode is active; mutations are disabled.[/cyan]"
            )
        if startup_token_file is None:
            console.print(f"Open: http://{host}:{port}/auth?token={token}")
        else:
            console.print("The access token was written to the requested owner-only file.")
        console.print("The access token expires when this process stops.")
        uvicorn.run(application, host=host, port=port, access_log=False)

    if startup_token_file is None:
        run_server()
        return
    try:
        with _startup_token_file_publication(startup_token_file, token):
            run_server()
    except StartupTokenFileError as exc:
        console.print(f"[red]Startup token file refused:[/red] {exc}")
        raise typer.Exit(2) from exc


if __name__ == "__main__":
    app()
