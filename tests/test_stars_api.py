from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from meta_archive.api import star_routes as star_route_module
from meta_archive.api.aggregate_cache import CATALOG_CHANGED_RETRY_DETAIL
from meta_archive.api.main import create_app
from meta_archive.api.security import CSRF_COOKIE
from meta_archive.models import (
    AttachmentOccurrence,
    Blob,
    Conversation,
    ConversationSegment,
    MediaDerivative,
    Participant,
    SegmentParticipant,
    Star,
)
from meta_archive.services.identifiers import stable_id
from meta_archive.services.media_derivatives import PIPELINE_VERSION
from meta_archive.services.stars import (
    messenger_contact_ids_eliminated_by_conversation_removal,
)


def _conversation(identifier: str, title: str, *, last_activity_ms: int) -> Conversation:
    return Conversation(
        id=identifier,
        conversation_key=f"fixture:{identifier}",
        title_raw=title,
        title_display=title,
        title_search=title.casefold(),
        conversation_type="direct",
        primary_category="inbox",
        participant_count=1,
        first_activity_ms=last_activity_ms - 1_000,
        last_activity_ms=last_activity_ms,
        message_count=3,
        attachment_count=1,
    )


def _star_app(
    tmp_path: Path,
    *,
    security_disabled: bool,
) -> tuple[Any, dict[str, str]]:
    directory_name = "meta-backup-stars-open" if security_disabled else "meta-backup-stars-secure"
    root = tmp_path / directory_name
    root.mkdir()
    app = create_app(
        root,
        access_token="a" * 32,
        security_disabled=security_disabled,
    )
    first_id = "1" * 64
    second_id = "2" * 64
    segment_id = "3" * 64
    participant_id = "4" * 64
    attachment_id = "5" * 64
    blob_id = "6" * 64
    contact_id = stable_id("messenger-contact", "alex example")
    with Session(app.state.engine) as session, session.begin():
        session.add_all(
            [
                _conversation(first_id, "Alex Example", last_activity_ms=2_000),
                _conversation(second_id, "Second Conversation", last_activity_ms=3_000),
                Blob(
                    sha256=blob_id,
                    size_bytes=42,
                    media_type="image/jpeg",
                    extension=".jpg",
                    relative_path="blobs/fixture.jpg",
                ),
            ]
        )
        session.flush()
        session.add(
            ConversationSegment(
                id=segment_id,
                conversation_id=first_id,
                thread_path="inbox/alex_example",
                title_raw="Alex Example",
                title_display="Alex Example",
                title_search="alex example",
                source_category="inbox",
                sequence=0,
                source_chunk_count=1,
                first_activity_ms=1_000,
                last_activity_ms=2_000,
            )
        )
        session.add(
            Participant(
                id=participant_id,
                name_raw="Alex Example",
                name_display="Alex Example",
                name_search="alex example",
                is_owner=False,
            )
        )
        session.flush()
        session.add(
            SegmentParticipant(
                segment_id=segment_id,
                participant_id=participant_id,
                ordinal=0,
            )
        )
        session.add(
            AttachmentOccurrence(
                id=attachment_id,
                conversation_id=first_id,
                blob_sha256=blob_id,
                kind="photos",
                ordinal=0,
                filename="fixture.jpg",
                media_type="image/jpeg",
                size_bytes=42,
                timestamp_ms=1_500,
                availability="resolved",
            )
        )
    return app, {
        "first_conversation": first_id,
        "second_conversation": second_id,
        "attachment": attachment_id,
        "blob": blob_id,
        "contact": contact_id,
    }


@pytest.fixture
def star_client(tmp_path: Path) -> Iterator[tuple[TestClient, Any, dict[str, str]]]:
    app, identifiers = _star_app(tmp_path, security_disabled=True)
    with TestClient(app) as client:
        yield client, app, identifiers


def test_star_schema_has_polymorphic_integrity_without_target_foreign_keys(
    catalog_engine,
) -> None:  # type: ignore[no-untyped-def]
    inspector = inspect(catalog_engine)
    assert inspector.get_foreign_keys("stars") == []
    assert {constraint["name"] for constraint in inspector.get_unique_constraints("stars")} == {
        "uq_stars_target"
    }
    assert {constraint["name"] for constraint in inspector.get_check_constraints("stars")} == {
        "ck_stars_target_type"
    }
    assert {index["name"] for index in inspector.get_indexes("stars")} == {
        "ix_stars_starred_at_id",
        "ix_stars_target_type_starred_at_id",
    }

    with pytest.raises(IntegrityError), Session(catalog_engine) as session, session.begin():
        session.add(
            Star(
                id="not-a-supported-target",
                target_type="derived_window",
                target_id="fixture",
            )
        )


def test_star_mutations_are_idempotent_and_list_current_mixed_content(
    star_client: tuple[TestClient, Any, dict[str, str]],
) -> None:
    client, app, identifiers = star_client
    with Session(app.state.engine) as session, session.begin():
        session.add(
            MediaDerivative(
                id="d" * 64,
                blob_sha256=identifiers["blob"],
                kind="image_thumbnail",
                pipeline_version=PIPELINE_VERSION,
                status="ready",
                relative_path="thumbnails/fixture.jpg",
                media_type="image/jpeg",
                size_bytes=21,
                attempt_count=1,
                metadata_json={},
            )
        )
    targets = (
        ("conversation", identifiers["first_conversation"]),
        ("attachment", identifiers["attachment"]),
        ("blob", identifiers["blob"]),
        ("messenger_contact", identifiers["contact"]),
    )
    created: dict[tuple[str, str], dict[str, Any]] = {}
    for target_type, target_id in targets:
        response = client.put(f"/api/v1/stars/{target_type}/{target_id}")
        assert response.status_code == 200
        assert response.json()["created"] is True
        assert response.json()["star_target"] == {
            "target_type": target_type,
            "target_id": target_id,
        }
        created[(target_type, target_id)] = response.json()

    repeated = client.put(f"/api/v1/stars/conversation/{identifiers['first_conversation']}")
    assert repeated.status_code == 200
    assert repeated.json()["created"] is False
    assert (
        repeated.json()["star"]
        == created[("conversation", identifiers["first_conversation"])]["star"]
    )

    listing = client.get("/api/v1/stars")
    assert listing.status_code == 200
    payload = listing.json()
    assert payload["page"]["total"] == 4
    assert payload["facets"] == {
        "all": 4,
        "conversations": 1,
        "messages": 0,
        "media": 2,
        "activity": 0,
        "people": 1,
    }
    assert {item["kind"] for item in payload["items"]} == {
        "conversation",
        "media",
        "person",
    }
    assert all(item["star_target"] and item["star"] for item in payload["items"])
    media = client.get("/api/v1/stars", params={"category": "media"}).json()
    assert media["page"]["total"] == 2
    assert {item["media_scope"] for item in media["items"]} == {"occurrence", "unique"}
    assert all(item["jump"] is not None for item in media["items"])
    assert all(
        [derivative["kind"] for derivative in item["media"]["derivatives"]] == ["image_thumbnail"]
        for item in media["items"]
    )

    target_url = f"/api/v1/stars/attachment/{identifiers['attachment']}"
    assert client.delete(target_url).status_code == 204
    assert client.delete(target_url).status_code == 204
    assert client.get("/api/v1/stars").json()["page"]["total"] == 3


def test_star_list_hides_quarantined_target_and_restore_preserves_identity(
    star_client: tuple[TestClient, Any, dict[str, str]],
) -> None:
    client, app, identifiers = star_client
    target_id = identifiers["first_conversation"]
    created = client.put(f"/api/v1/stars/conversation/{target_id}").json()["star"]

    with Session(app.state.engine) as session, session.begin():
        conversation = session.get(Conversation, target_id)
        assert conversation is not None
        conversation.deleted_at = datetime.now(UTC)

    hidden = client.get("/api/v1/stars").json()
    assert hidden["items"] == []
    assert hidden["page"]["total"] == 0
    with Session(app.state.engine) as session:
        retained = session.scalar(
            select(Star).where(
                Star.target_type == "conversation",
                Star.target_id == target_id,
            )
        )
        assert retained is not None
        assert retained.id == created["id"]

    with Session(app.state.engine) as session, session.begin():
        conversation = session.get(Conversation, target_id)
        assert conversation is not None
        conversation.deleted_at = None

    restored = client.get("/api/v1/stars").json()["items"]
    assert len(restored) == 1
    assert restored[0]["star"] == created


def test_star_cursor_is_filter_bound_and_missing_targets_are_rejected(
    star_client: tuple[TestClient, Any, dict[str, str]],
) -> None:
    client, _app, identifiers = star_client
    for target_id in (
        identifiers["first_conversation"],
        identifiers["second_conversation"],
    ):
        assert client.put(f"/api/v1/stars/conversation/{target_id}").status_code == 200

    first = client.get(
        "/api/v1/stars",
        params={"category": "conversations", "direction": "desc", "limit": 1},
    ).json()
    assert first["page"]["total"] == 2
    assert first["page"]["next_cursor"]
    second = client.get(
        "/api/v1/stars",
        params={
            "category": "conversations",
            "direction": "desc",
            "limit": 1,
            "cursor": first["page"]["next_cursor"],
        },
    )
    assert second.status_code == 200
    assert second.json()["items"][0]["star"]["id"] != first["items"][0]["star"]["id"]
    assert (
        client.get(
            "/api/v1/stars",
            params={"limit": 2, "cursor": first["page"]["next_cursor"]},
        ).status_code
        == 422
    )
    assert client.put(f"/api/v1/stars/message/{'9' * 64}").status_code == 404
    assert client.put(f"/api/v1/stars/unsupported/{'9' * 64}").status_code == 422
    assert client.put("/api/v1/stars/conversation/not valid").status_code == 422


def test_starred_items_order_rapid_clicks_by_precise_server_time(
    star_client: tuple[TestClient, Any, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _app, identifiers = star_client
    timestamps = iter(
        (
            datetime(2030, 1, 1, 12, 0, 0, 1, tzinfo=UTC),
            datetime(2030, 1, 1, 12, 0, 0, 2, tzinfo=UTC),
        )
    )
    star_ids = iter(
        (
            "ffffffff-ffff-ffff-ffff-ffffffffffff",
            "00000000-0000-0000-0000-000000000000",
        )
    )

    class PreciseClock:
        @classmethod
        def now(cls, timezone: Any) -> datetime:
            assert timezone is UTC
            return next(timestamps)

    monkeypatch.setattr(star_route_module, "datetime", PreciseClock)
    monkeypatch.setattr(star_route_module.uuid, "uuid4", lambda: next(star_ids))

    first = client.put(f"/api/v1/stars/conversation/{identifiers['first_conversation']}").json()
    second = client.put(f"/api/v1/stars/conversation/{identifiers['second_conversation']}").json()

    assert first["star"]["starred_at"] == "2030-01-01T12:00:00.000001+00:00"
    assert second["star"]["starred_at"] == "2030-01-01T12:00:00.000002+00:00"
    listing = client.get(
        "/api/v1/stars",
        params={"category": "conversations", "direction": "desc"},
    ).json()
    assert [item["star_target"]["target_id"] for item in listing["items"]] == [
        identifiers["second_conversation"],
        identifiers["first_conversation"],
    ]


def test_messenger_contact_matches_people_and_purge_semantics(
    star_client: tuple[TestClient, Any, dict[str, str]],
) -> None:
    client, app, identifiers = star_client
    support_id = "7" * 64
    support_segment_id = "8" * 64
    with Session(app.state.engine) as session, session.begin():
        support = _conversation(support_id, "Support Case", last_activity_ms=4_000)
        support.conversation_type = "support"
        session.add(support)
        session.flush()
        session.add(
            ConversationSegment(
                id=support_segment_id,
                conversation_id=support_id,
                thread_path="support/support_case",
                title_raw="Support Case",
                title_display="Support Case",
                title_search="support case",
                source_category="support",
                sequence=0,
                source_chunk_count=1,
                first_activity_ms=3_000,
                last_activity_ms=4_000,
            )
        )
        session.flush()
        session.add(
            SegmentParticipant(
                segment_id=support_segment_id,
                participant_id="4" * 64,
                ordinal=0,
            )
        )

    assert (
        client.put(f"/api/v1/stars/messenger_contact/{identifiers['contact']}").status_code == 200
    )
    item = client.get("/api/v1/stars", params={"category": "people"}).json()["items"][0]
    assert item["contact"]["conversation_count"] == 1
    assert [conversation["type"] for conversation in item["conversations"]] == ["direct"]
    # The fixture photo has no message identity, so People does not count it as an exchange.
    assert item["counts"]["photos"] == 0

    with Session(app.state.engine) as session:
        eliminated = messenger_contact_ids_eliminated_by_conversation_removal(
            session,
            [identifiers["first_conversation"]],
        )
    assert eliminated == {identifiers["contact"]}


def test_starred_items_fail_closed_if_catalog_changes_during_hydration(
    star_client: tuple[TestClient, Any, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, app, identifiers = star_client
    assert (
        client.put(f"/api/v1/stars/conversation/{identifiers['first_conversation']}").status_code
        == 200
    )
    original = star_route_module.resolve_star_targets
    armed = True

    def resolve_with_concurrent_commit(
        session: Session,
        targets: Any,
        *,
        include_payload: bool = True,
    ) -> Any:
        nonlocal armed
        value = original(session, targets, include_payload=include_payload)
        if armed and include_payload:
            armed = False
            with Session(app.state.engine) as mutation, mutation.begin():
                conversation = mutation.get(
                    Conversation,
                    identifiers["first_conversation"],
                )
                assert conversation is not None
                conversation.title_display = "Changed concurrently"
        return value

    monkeypatch.setattr(
        star_route_module,
        "resolve_star_targets",
        resolve_with_concurrent_commit,
    )
    raced = client.get("/api/v1/stars")

    assert raced.status_code == 409
    assert raced.json() == {"detail": CATALOG_CHANGED_RETRY_DETAIL}
    assert client.get("/api/v1/stars").status_code == 200


def test_star_mutation_uses_existing_session_csrf_and_origin_security(tmp_path: Path) -> None:
    app, identifiers = _star_app(tmp_path, security_disabled=False)
    target = f"/api/v1/stars/conversation/{identifiers['first_conversation']}"
    with TestClient(
        app,
        base_url="http://127.0.0.1:8765",
        follow_redirects=False,
    ) as client:
        assert client.get("/api/v1/stars").status_code == 401
        assert client.get("/auth", params={"token": "a" * 32}).status_code == 303
        assert client.put(target).status_code == 403
        accepted = client.put(
            target,
            headers={
                "origin": "http://127.0.0.1:8765",
                "x-csrf-token": client.cookies[CSRF_COOKIE],
            },
        )
    assert accepted.status_code == 200
