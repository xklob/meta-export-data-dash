from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Literal, cast

import typer
from rich.console import Console
from sqlalchemy import Engine

from meta_archive.database import create_catalog_engine, run_migrations
from meta_archive.paths import ArchiveLayout, ArchiveNotFoundError
from meta_archive.services.deletion import (
    AVAILABILITY,
    ENTITY_TYPES,
    MEDIA_TYPES,
    DeletionError,
    DeletionSelector,
    EntityType,
    commit_deletion,
    execute_hard_purge,
    plan_hard_purge,
    preview_deletion,
    restore_deletion,
)
from meta_archive.services.source_import_lock import source_import_publication_lock

deletion_app = typer.Typer(
    name="delete",
    help="Preview, quarantine, restore, and explicitly purge selected archive records.",
    no_args_is_help=True,
)
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


def _catalog(archive_root: Path | None) -> tuple[ArchiveLayout, Engine]:
    try:
        layout = ArchiveLayout.discover(archive_root)
    except ArchiveNotFoundError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(2) from exc
    with source_import_publication_lock(layout):
        layout.ensure_runtime_directories()
        run_migrations(layout.database)
        return layout, create_catalog_engine(layout.database)


def _print(value: object) -> None:
    console.print_json(json.dumps(value, default=str))


def _entity_id_map(values: list[str] | None) -> dict[str, tuple[str, ...]]:
    parsed: dict[str, list[str]] = {}
    for value in values or []:
        entity_type, separator, entity_id = value.partition(":")
        if not separator or entity_type not in ENTITY_TYPES or not entity_id:
            raise typer.BadParameter(
                "--entity-id must use one of message:ID, activity:ID, conversation:ID, "
                "or attachment:ID"
            )
        parsed.setdefault(entity_type, []).append(entity_id)
    return {key: tuple(identifiers) for key, identifiers in parsed.items()}


def _validated_values(
    values: list[str] | None, allowed: frozenset[str], label: str
) -> tuple[str, ...]:
    normalized = tuple(values or ())
    invalid = set(normalized) - allowed
    if invalid:
        raise typer.BadParameter(f"Unknown {label}: {', '.join(sorted(invalid))}")
    return normalized


def _service_error(exc: DeletionError) -> None:
    console.print(f"[red]Error:[/red] {exc}")
    raise typer.Exit(2) from exc


@deletion_app.command("preview")
def preview(
    archive_root: ArchiveRootOption = None,
    entity: Annotated[list[str] | None, typer.Option("--entity")] = None,
    text_value: Annotated[str | None, typer.Option("--text")] = None,
    text_mode: Annotated[Literal["exact", "keyword"], typer.Option()] = "exact",
    case_sensitive: Annotated[bool, typer.Option("--case-sensitive/--case-insensitive")] = True,
    whole_field: Annotated[bool, typer.Option("--whole-field")] = False,
    text_variant: Annotated[Literal["raw", "display"], typer.Option()] = "display",
    from_date: Annotated[str | None, typer.Option()] = None,
    to_date: Annotated[str | None, typer.Option()] = None,
    timezone: Annotated[str, typer.Option()] = "America/New_York",
    conversation: Annotated[list[str] | None, typer.Option("--conversation")] = None,
    participant: Annotated[str | None, typer.Option()] = None,
    media_type: Annotated[list[str] | None, typer.Option("--media-type")] = None,
    availability: Annotated[list[str] | None, typer.Option()] = None,
    entity_id: Annotated[
        list[str] | None,
        typer.Option("--entity-id", help="Repeat TYPE:ID to select stable entity IDs."),
    ] = None,
    allow_all: Annotated[
        bool,
        typer.Option(help="Explicitly allow a preview with no narrowing filter."),
    ] = False,
) -> None:
    """Create a non-mutating preview and one-time commit token."""

    entities = _validated_values(entity, ENTITY_TYPES, "entity type")
    selector = DeletionSelector(
        entity_types=cast(tuple[EntityType, ...], entities or ("message", "activity")),
        text=text_value,
        text_mode=text_mode,
        case_sensitive=case_sensitive,
        whole_field=whole_field,
        text_variant=text_variant,
        from_date=from_date,
        to_date=to_date,
        timezone=timezone,
        conversation_ids=tuple(conversation or ()),
        participant=participant,
        media_types=_validated_values(media_type, MEDIA_TYPES, "media type"),
        availability=_validated_values(availability, AVAILABILITY, "availability"),
        entity_ids=_entity_id_map(entity_id),
        allow_all=allow_all,
    )
    layout, engine = _catalog(archive_root)
    try:
        _print(asdict(preview_deletion(layout, engine, selector)))
    except DeletionError as exc:
        _service_error(exc)
    finally:
        engine.dispose()


@deletion_app.command("commit")
def commit(
    operation_id: str,
    commit_token: Annotated[str, typer.Option(prompt=False)],
    catalog_revision: Annotated[str, typer.Option()],
    archive_root: ArchiveRootOption = None,
    apply: Annotated[bool, typer.Option(help="Required destructive-action confirmation.")] = False,
) -> None:
    """Commit an unchanged preview into logical quarantine."""

    if not apply:
        raise typer.BadParameter("Commit requires --apply")
    layout, engine = _catalog(archive_root)
    try:
        _print(
            asdict(
                commit_deletion(
                    layout,
                    engine,
                    operation_id,
                    commit_token=commit_token,
                    catalog_revision=catalog_revision,
                )
            )
        )
    except DeletionError as exc:
        _service_error(exc)
    finally:
        engine.dispose()


@deletion_app.command("restore")
def restore(
    operation_id: str,
    confirmation: Annotated[str, typer.Option(help="Must exactly equal OPERATION_ID.")],
    archive_root: ArchiveRootOption = None,
    apply: Annotated[bool, typer.Option(help="Required mutation confirmation.")] = False,
) -> None:
    """Restore every row in a committed logical quarantine."""

    if not apply:
        raise typer.BadParameter("Restore requires --apply")
    layout, engine = _catalog(archive_root)
    try:
        _print(
            asdict(
                restore_deletion(
                    layout,
                    engine,
                    operation_id,
                    confirmation=confirmation,
                )
            )
        )
    except DeletionError as exc:
        _service_error(exc)
    finally:
        engine.dispose()


@deletion_app.command("purge-preview")
def purge_preview(
    operation_id: str,
    scope: Annotated[Literal["normalized", "workspace"], typer.Option()] = "workspace",
    archive_root: ArchiveRootOption = None,
) -> None:
    """Plan an explicit-scope irreversible purge and print its blockers/token."""

    layout, engine = _catalog(archive_root)
    try:
        _print(
            asdict(
                plan_hard_purge(
                    layout,
                    engine,
                    operation_id,
                    scope=scope,
                )
            )
        )
    except DeletionError as exc:
        _service_error(exc)
    finally:
        engine.dispose()


@deletion_app.command("purge")
def purge(
    operation_id: str,
    purge_token: Annotated[str, typer.Option()],
    purge_revision: Annotated[str, typer.Option()],
    confirmation: Annotated[str, typer.Option(help="Must exactly equal OPERATION_ID.")],
    scope: Annotated[Literal["normalized", "workspace"], typer.Option()] = "workspace",
    archive_root: ArchiveRootOption = None,
    apply: Annotated[bool, typer.Option(help="Required irreversible-action confirmation.")] = False,
) -> None:
    """Execute an unchanged, blocker-free hard-purge plan."""

    if not apply:
        raise typer.BadParameter("Purge requires --apply")
    layout, engine = _catalog(archive_root)
    try:
        _print(
            asdict(
                execute_hard_purge(
                    layout,
                    engine,
                    operation_id,
                    scope=scope,
                    purge_token=purge_token,
                    purge_revision=purge_revision,
                    confirmation=confirmation,
                )
            )
        )
    except DeletionError as exc:
        _service_error(exc)
    finally:
        engine.dispose()
