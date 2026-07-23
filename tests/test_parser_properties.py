from __future__ import annotations

import math
import re
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from meta_archive.services.activity_importer import (
    _adapt_payload,
    _json_pointer_join,
    _normalized_local_uri,
    _timestamp_ms,
)
from meta_archive.services.message_importer import (
    _json_pointer_token,
    _normal_envelope_error,
    _seconds_to_ms,
)
from meta_archive.services.text import normalize_text, safe_filesystem_label


@given(st.text())
def test_text_normalization_is_stable_and_preserves_the_raw_value(raw: str) -> None:
    normalized = normalize_text(raw)

    assert normalized.raw == raw
    assert normalize_text(normalized.display).display == normalized.display
    assert normalize_text(normalized.display).search == normalized.search


@given(st.text(), st.integers(min_value=1, max_value=200))
def test_filesystem_labels_are_bounded_single_safe_path_components(raw: str, limit: int) -> None:
    label = safe_filesystem_label(raw, fallback="untitled", limit=limit)

    assert 1 <= len(label) <= limit
    assert re.fullmatch(r"[a-z0-9._-]+", label)
    assert "/" not in label
    assert "\\" not in label
    assert label not in {".", ".."}


@given(st.integers(min_value=1, max_value=99_999_999_999))
def test_facebook_epoch_seconds_are_normalized_to_milliseconds(seconds: int) -> None:
    assert _timestamp_ms(seconds) == seconds * 1000
    assert _seconds_to_ms(seconds) == seconds * 1000


@given(st.integers(min_value=100_000_000_000, max_value=9_999_999_999_999))
def test_already_normalized_epoch_milliseconds_are_not_scaled(milliseconds: int) -> None:
    assert _timestamp_ms(milliseconds) == milliseconds
    assert _seconds_to_ms(milliseconds) == milliseconds


@pytest.mark.parametrize("value", [None, True, False, 0, -1, "1700000000", math.nan, math.inf])
def test_activity_timestamps_fail_closed_for_unknown_or_invalid_values(value: object) -> None:
    assert _timestamp_ms(value) is None


@given(st.text())
def test_json_pointer_tokens_round_trip_all_unicode(token: str) -> None:
    escaped = _json_pointer_token(token)
    joined = _json_pointer_join("", token)

    assert joined == f"/{escaped}"
    assert escaped.replace("~1", "/").replace("~0", "~") == token
    assert "/" not in escaped
    assert re.search(r"~(?![01])", escaped) is None


@given(
    title=st.text(min_size=1),
    thread_path=st.text(),
    participant_names=st.lists(st.text(), max_size=12),
    messages=st.lists(st.dictionaries(st.text(max_size=20), st.integers()), max_size=8),
)
def test_normal_message_envelopes_accept_all_supported_field_values(
    title: str,
    thread_path: str,
    participant_names: list[str],
    messages: list[dict[str, int]],
) -> None:
    payload = {
        "title": title,
        "participants": [{"name": name} for name in participant_names],
        "messages": messages,
        "thread_path": thread_path,
        "is_still_participant": True,
        "magic_words": [],
    }

    assert _normal_envelope_error(payload) is None


@given(
    field_and_value=st.sampled_from(
        [
            ("title", 1),
            ("participants", {}),
            ("messages", {}),
            ("thread_path", 1),
            ("is_still_participant", "yes"),
            ("magic_words", {}),
        ]
    )
)
def test_normal_message_envelopes_reject_wrong_required_field_types(
    field_and_value: tuple[str, object],
) -> None:
    payload: dict[str, Any] = {
        "title": "Fixture",
        "participants": [{"name": "Archive Owner"}],
        "messages": [],
        "thread_path": "inbox/fixture",
        "is_still_participant": True,
        "magic_words": [],
    }
    field, value = field_and_value
    payload[field] = value

    assert _normal_envelope_error(payload) is not None


@given(
    title=st.text(min_size=1),
    timestamp=st.integers(min_value=1, max_value=9_999_999_999_999),
)
def test_generic_activity_schema_preserves_title_and_timestamp_units(
    title: str, timestamp: int
) -> None:
    payload = {
        "title": title,
        "timestamp": timestamp,
        "label_values": [],
        "media": [],
    }

    schema, records = _adapt_payload(payload, "your_facebook_activity/example/history.json")

    assert schema == "generic"
    assert len(records) == 1
    assert records[0].title_raw == title
    assert records[0].timestamp_ms == _timestamp_ms(timestamp)


@given(
    name=st.text(),
    timestamp=st.integers(min_value=1, max_value=99_999_999_999),
    photo_count=st.integers(min_value=0, max_value=20),
)
def test_album_schema_retains_parent_and_child_coordinates(
    name: str, timestamp: int, photo_count: int
) -> None:
    payload = {
        "name": name,
        "last_modified_timestamp": timestamp,
        "photos": [{"uri": f"photos/{index}.jpg"} for index in range(photo_count)],
        "cover_photo": {"uri": "photos/cover.jpg"},
    }

    schema, records = _adapt_payload(payload, "your_facebook_activity/posts/album.json")

    assert schema == "album"
    assert len(records) == photo_count + 2
    assert records[0].json_pointer == ""
    assert records[0].timestamp_ms == timestamp * 1000
    assert all(record.parent_pointer == "" for record in records[1:])
    assert len({record.json_pointer for record in records}) == len(records)


@given(
    segments=st.lists(
        st.text(
            alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._- ",
            min_size=1,
            max_size=30,
        ).filter(lambda value: value.strip() not in {"", ".", ".."}),
        min_size=1,
        max_size=8,
    )
)
def test_safe_relative_attachment_paths_normalize_without_escaping(segments: list[str]) -> None:
    raw = "/".join(segments)

    normalized = _normalized_local_uri(raw)

    assert normalized is not None
    assert not normalized.startswith("/")
    assert ".." not in normalized.split("/")


@given(
    prefix=st.text(
        alphabet=st.characters(
            blacklist_characters="/\\\x00",
            blacklist_categories=("Cs",),
        ),
        min_size=1,
        max_size=30,
    ).filter(lambda value: value not in {".", ".."}),
    suffix=st.sampled_from(("../secret", "./secret", "..%2Fsecret", ".%2Fsecret")),
)
def test_attachment_path_traversal_is_rejected(prefix: str, suffix: str) -> None:
    assert _normalized_local_uri(f"{prefix}/{suffix}") is None
