from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from meta_archive.database import create_catalog_engine, run_migrations
from meta_archive.models import Message, SourceChunk, SourceFile
from meta_archive.paths import ArchiveLayout
from meta_archive.services.consolidation import build_blob_store
from meta_archive.services.inventory import scan_archive
from meta_archive.services.message_importer import import_messages
from meta_archive.services.purge_rewrite import (
    RawRewritePlan,
    RawSourceRewriteError,
    build_raw_rewrite_plan,
    publish_raw_rewrite_files,
    validate_raw_rewrite_physical_sources,
)

MESSAGE_FIXTURES = Path(__file__).parent / "fixtures" / "messages"


@pytest.fixture
def rewrite_archive(tmp_path: Path) -> Iterator[tuple[ArchiveLayout, Engine]]:
    layout = ArchiveLayout(tmp_path / "meta-backup-purge-rewrite")
    shutil.copytree(MESSAGE_FIXTURES, layout.root)
    layout.ensure_runtime_directories()
    run_migrations(layout.database)
    engine = create_catalog_engine(layout.database)
    scan_archive(layout, engine, hash_files=True)
    import_messages(layout, engine)
    build_blob_store(layout, engine)
    yield layout, engine
    engine.dispose()


def _selected_message_plan(
    layout: ArchiveLayout,
    engine: Engine,
) -> tuple[RawRewritePlan, Path]:
    with Session(engine) as session:
        message = session.scalar(select(Message).where(Message.content_display == "Older chunk"))
        assert message is not None
        source_file = session.get(SourceFile, message.source_file_id)
        assert source_file is not None
        chunk = session.get(SourceChunk, source_file.chunk_id)
        assert chunk is not None
        selected = {
            "message": {message.id},
            "activity": set(),
            "conversation": set(),
            "attachment": set(),
        }
        plan = build_raw_rewrite_plan(layout, session, selected)
        physical = layout.root / chunk.original_path / source_file.relative_path
    assert len(plan.rewrites) == 1
    assert plan.rewrites[0].physical_path == physical
    return plan, physical


def _different_bytes_with_same_size(payload: bytes) -> bytes:
    assert payload
    return bytes([payload[0] ^ 1]) + payload[1:]


def test_build_plan_rejects_physical_source_content_drift(
    rewrite_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = rewrite_archive
    _plan, physical = _selected_message_plan(layout, engine)
    original = physical.read_bytes()
    physical.write_bytes(_different_bytes_with_same_size(original))

    with (
        Session(engine) as session,
        pytest.raises(
            RawSourceRewriteError,
            match=r"Physical raw source .* SHA-256 changed after inventory",
        ),
    ):
        message = session.scalar(select(Message).where(Message.content_display == "Older chunk"))
        assert message is not None
        selected = {
            "message": {message.id},
            "activity": set(),
            "conversation": set(),
            "attachment": set(),
        }
        build_raw_rewrite_plan(layout, session, selected)


@pytest.mark.parametrize("drift", ["edited", "symlink", "missing"])
def test_publication_rejects_physical_source_drift_without_publishing(
    rewrite_archive: tuple[ArchiveLayout, Engine],
    drift: str,
) -> None:
    layout, engine = rewrite_archive
    plan, physical = _selected_message_plan(layout, engine)
    rewrite = plan.rewrites[0]
    original_manifest = layout.original_files_manifest.read_bytes()
    relocation_manifest = layout.relocation_manifest.read_bytes()
    original = physical.read_bytes()

    if drift == "edited":
        physical.write_bytes(_different_bytes_with_same_size(original))
    elif drift == "symlink":
        physical.unlink()
        physical.symlink_to(layout.blob_path(rewrite.old_sha256))
    else:
        physical.unlink()

    with pytest.raises(RawSourceRewriteError, match="Physical raw source"):
        publish_raw_rewrite_files(layout, plan)

    assert layout.original_files_manifest.read_bytes() == original_manifest
    assert layout.relocation_manifest.read_bytes() == relocation_manifest
    assert not layout.blob_path(rewrite.new_sha256).exists()
    if drift == "edited":
        assert physical.read_bytes() != original
    elif drift == "symlink":
        assert physical.is_symlink()
    else:
        assert not physical.exists()


def test_publication_revalidates_the_source_in_rollback_staging(
    rewrite_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = rewrite_archive
    plan, physical = _selected_message_plan(layout, engine)
    rewrite = plan.rewrites[0]
    original = physical.read_bytes()
    staging_source = (
        layout.trash
        / "purge-staging"
        / "11111111-2222-3333-4444-555555555555"
        / physical.relative_to(layout.root)
    )
    staging_source.parent.mkdir(parents=True)
    os.replace(physical, staging_source)

    created = publish_raw_rewrite_files(layout, plan)

    assert physical.read_bytes() == rewrite.new_bytes
    assert staging_source.read_bytes() == original
    assert layout.blob_path(rewrite.new_sha256) in created


def test_exported_plan_validator_supports_delayed_upgrade_preflight(
    rewrite_archive: tuple[ArchiveLayout, Engine],
) -> None:
    layout, engine = rewrite_archive
    plan, physical = _selected_message_plan(layout, engine)
    validate_raw_rewrite_physical_sources(layout, plan)

    original = physical.read_bytes()
    physical.write_bytes(_different_bytes_with_same_size(original))
    with pytest.raises(RawSourceRewriteError, match="SHA-256 changed after inventory"):
        validate_raw_rewrite_physical_sources(layout, plan)
