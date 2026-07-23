from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine

from meta_archive.database import create_catalog_engine, run_migrations
from meta_archive.paths import ArchiveLayout


@pytest.fixture
def archive_layout(tmp_path: Path) -> ArchiveLayout:
    root = tmp_path / "meta-backup-fixture"
    root.mkdir()
    return ArchiveLayout(root)


@pytest.fixture
def catalog_engine(archive_layout: ArchiveLayout) -> Iterator[Engine]:
    archive_layout.ensure_runtime_directories()
    run_migrations(archive_layout.database)
    engine = create_catalog_engine(archive_layout.database)
    yield engine
    engine.dispose()
