from __future__ import annotations

from meta_archive.services.text import normalize_text, safe_filesystem_label


def test_normalization_preserves_raw_repairs_display_and_casefolds_search() -> None:
    normalized = normalize_text("cafÃ©")

    assert normalized.raw == "cafÃ©"
    assert normalized.display == "café"
    assert normalized.search == "café"


def test_filesystem_label_is_bounded_and_safe() -> None:
    label = safe_filesystem_label("  A / complicated: conversation?  ", limit=20)

    assert label == "a-complicated-conver"
    assert "/" not in label
