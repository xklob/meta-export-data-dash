from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lancedb  # type: ignore[import-untyped]
import pytest

from meta_archive.services.semantic_state import LanceTableInventoryError, lance_table_names


@dataclass(frozen=True)
class _TablePage:
    tables: list[str]
    page_token: str | None


class _PagedDatabase:
    def __init__(self, pages: dict[str | None, _TablePage]) -> None:
        self.pages = pages
        self.calls: list[tuple[str | None, int]] = []

    def list_tables(self, *, limit: int, page_token: str | None = None) -> _TablePage:
        self.calls.append((page_token, limit))
        return self.pages[page_token]


def test_lance_table_inventory_includes_known_and_extra_tables_after_first_page() -> None:
    first_page = [f"a_table_{index:04d}" for index in range(1_000)]
    database = _PagedDatabase(
        {
            None: _TablePage(first_page, "z_registered_after_limit"),
            first_page[-1]: _TablePage(["z_registered_after_limit", "zz_orphan_after_limit"], None),
        }
    )

    names = lance_table_names(database)

    assert len(names) == 1_002
    assert "z_registered_after_limit" in names
    assert "zz_orphan_after_limit" in names
    assert database.calls == [(None, 1_000), (first_page[-1], 1_000)]


def test_lance_table_inventory_matches_a_real_local_database_across_page_boundaries(
    tmp_path: Path,
) -> None:
    database = lancedb.connect(str(tmp_path / "vectors"))
    expected = {f"table_{index:04d}" for index in range(1_002)}
    for table_name in sorted(expected):
        database.create_table(table_name, data=[{"id": table_name}], mode="create")

    assert lance_table_names(database, page_size=1) == expected
    assert lance_table_names(database, page_size=1_000) == expected


def test_lance_table_inventory_rejects_a_repeated_page_token() -> None:
    database = _PagedDatabase(
        {
            None: _TablePage(["first"], "boundary"),
            "first": _TablePage(["second"], "boundary"),
        }
    )

    with pytest.raises(LanceTableInventoryError, match="repeated a page token"):
        lance_table_names(database)


def test_lance_table_inventory_rejects_a_page_that_makes_no_progress() -> None:
    database = _PagedDatabase(
        {
            None: _TablePage(["same"], "page-2"),
            "same": _TablePage(["same"], "page-3"),
        }
    )

    with pytest.raises(LanceTableInventoryError, match="made no progress"):
        lance_table_names(database)
