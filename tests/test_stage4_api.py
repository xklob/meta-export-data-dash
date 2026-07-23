from __future__ import annotations

import shutil
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from sqlalchemy import Engine, case, event, func, literal, select, text
from sqlalchemy.orm import Session

from meta_archive.api import aggregate_cache as aggregate_cache_module
from meta_archive.api import analytics_routes, people_routes
from meta_archive.api.activity_routes import _timeline_page_statements, _timeline_union
from meta_archive.api.aggregate_cache import (
    CATALOG_CHANGED_RETRY_DETAIL,
    RevisionBoundAggregateCache,
)
from meta_archive.api.event_routes import _event_status
from meta_archive.api.security import LocalSecurity
from meta_archive.api.stage4_routes import stage4_router
from meta_archive.database import create_catalog_engine, run_migrations
from meta_archive.models import (
    ActivityRecord,
    AttachmentOccurrence,
    Blob,
    Conversation,
    ConversationSegment,
    CoverageArea,
    Job,
    Message,
    MessageReaction,
    NlpHourlyRollup,
    SearchWindow,
    SearchWindowMessage,
    SemanticIndexConfig,
    SourceFile,
)
from meta_archive.paths import ArchiveLayout
from meta_archive.services.activity_importer import import_activities
from meta_archive.services.inventory import scan_archive
from meta_archive.services.message_importer import import_messages

FIXTURES = Path(__file__).parent / "fixtures"


@dataclass(frozen=True, slots=True)
class Stage4Catalog:
    client: TestClient
    engine: Engine
    layout: ArchiveLayout


@pytest.fixture
def stage4_catalog(tmp_path: Path) -> Iterator[Stage4Catalog]:
    layout = ArchiveLayout(tmp_path / "meta-backup-stage4")
    shutil.copytree(FIXTURES / "messages", layout.root)
    shutil.copytree(FIXTURES / "activity", layout.root / "package-c")
    layout.ensure_runtime_directories()
    run_migrations(layout.database)
    engine = create_catalog_engine(layout.database)
    scan_archive(layout, engine, hash_files=True)
    import_messages(layout, engine)
    import_activities(layout, engine)

    app = FastAPI()
    app.state.layout = layout
    app.state.engine = engine
    app.state.security = LocalSecurity(disabled=True)
    app.include_router(stage4_router)
    with TestClient(app) as client:
        yield Stage4Catalog(client=client, engine=engine, layout=layout)
    engine.dispose()


@pytest.fixture
def stage4_positive_catalog(stage4_catalog: Stage4Catalog) -> Stage4Catalog:
    shutil.copytree(
        FIXTURES / "stage4_positive",
        stage4_catalog.layout.root / "package-d",
    )
    scan_archive(stage4_catalog.layout, stage4_catalog.engine, hash_files=True)
    import_activities(stage4_catalog.layout, stage4_catalog.engine)
    return stage4_catalog


def test_activity_views_filters_cursors_and_source_jumps(stage4_catalog: Stage4Catalog) -> None:
    client = stage4_catalog.client
    first = client.get(
        "/api/v1/activity",
        params={"sort": "date", "direction": "desc", "limit": 2},
    )
    assert first.status_code == 200
    first_payload = first.json()
    assert first_payload["page"]["total"] == 13
    assert first_payload["page"]["next_cursor"]
    assert len(first_payload["items"]) == 2
    assert all(item["source"]["relative_path"] for item in first_payload["items"])
    assert all(item["source"]["json_pointer"] is not None for item in first_payload["items"])
    assert all(
        item["jump"]["fragment"] == f"activity-{item['id']}" for item in first_payload["items"]
    )

    second = client.get(
        "/api/v1/activity",
        params={
            "sort": "date",
            "direction": "desc",
            "limit": 2,
            "cursor": first_payload["page"]["next_cursor"],
        },
    )
    assert second.status_code == 200
    assert {item["id"] for item in second.json()["items"]}.isdisjoint(
        {item["id"] for item in first_payload["items"]}
    )

    mismatched = client.get(
        "/api/v1/activity",
        params={"q": "fixture", "cursor": first_payload["page"]["next_cursor"]},
    )
    assert mismatched.status_code == 422

    expected_counts = {
        "posts": 3,
        "comments": 2,
        "groups": 3,
        "marketplace": 2,
    }
    for view, expected in expected_counts.items():
        response = client.get(f"/api/v1/activity/views/{view}")
        assert response.status_code == 200, (view, response.text)
        payload = response.json()
        assert payload["page"]["total"] == expected
        assert len(payload["items"]) == expected
        assert payload["coverage"]["status"] in {"available", "partial"}

    posts = client.get("/api/v1/activity/views/posts").json()
    general_posts = {row["key"]: row for row in posts["coverage_components"]}["general_posts"]
    assert general_posts["status"] == "unavailable"
    assert general_posts["raw_count"] is None
    assert general_posts["logical_count"] is None
    assert general_posts["absence_is_not_zero_history"] is True

    for view in ("pages", "check-ins", "support"):
        payload = client.get(f"/api/v1/activity/views/{view}").json()
        assert payload["coverage"]["status"] == "unavailable"
        assert payload["page"]["total"] is None
        assert payload["page"]["count_is_unknown"] is True

    searched = client.get("/api/v1/activity", params={"q": "quiet, synthetic"}).json()
    assert searched["page"]["total"] == 1
    assert searched["items"][0]["title"] == "Café fixture listing"
    attached = client.get("/api/v1/activity", params={"has_attachments": True}).json()
    assert attached["items"]
    assert all(item["attachment_count"] > 0 for item in attached["items"])
    unavailable_body = client.get(
        "/api/v1/activity",
        params={"availability": "unavailable_body"},
    ).json()
    assert unavailable_body["page"]["total"] == 1

    marketplace = client.get("/api/v1/activity/views/marketplace").json()
    record = next(item for item in marketplace["items"] if item["attachment_count"])
    detail = client.get(f"/api/v1/activity/records/{record['id']}")
    assert detail.status_code == 200
    detail_payload = detail.json()
    assert len(detail_payload["attachments"]) == record["attachment_count"]
    assert all("source_uri" not in item for item in detail_payload["attachments"])
    assert all(
        item["blob_href"].startswith("/api/v1/blobs/") for item in detail_payload["attachments"]
    )

    for sort in ("date", "title", "actor", "category", "subtype"):
        response = client.get(
            "/api/v1/activity",
            params={"sort": sort, "direction": "asc", "limit": 100},
        )
        assert response.status_code == 200, (sort, response.text)
        assert response.json()["page"]["total"] == 13


def test_observed_activity_overrides_a_stale_unavailable_coverage_row(
    stage4_catalog: Stage4Catalog,
) -> None:
    with Session(stage4_catalog.engine) as session:
        coverage = session.get(CoverageArea, "activity_all")
        if coverage is None:
            coverage = CoverageArea(
                key="activity_all",
                label="Facebook activity",
                status="unavailable",
                raw_count=None,
                logical_count=None,
                description="No supported non-message activity is available.",
                source_paths_json={},
            )
            session.add(coverage)
        else:
            coverage.status = "unavailable"
            coverage.raw_count = None
            coverage.logical_count = None
            coverage.description = "No supported non-message activity is available."
        session.commit()

    response = stage4_catalog.client.get("/api/v1/activity", params={"limit": 1})
    assert response.status_code == 200
    payload = response.json()
    assert payload["page"]["total"] == 13
    assert payload["coverage"] == {
        "key": "activity_all",
        "label": "Facebook activity",
        "status": "partial",
        "raw_count": 13,
        "logical_count": 13,
        "description": (
            "Retained non-message Facebook activity records are available; coverage remains "
            "partial where export categories are absent."
        ),
        "source_paths": {},
        "absence_is_not_zero_history": False,
    }


def test_timeline_combines_sources_with_stable_jumps_and_known_empty_filters(
    stage4_catalog: Stage4Catalog,
) -> None:
    client = stage4_catalog.client
    first = client.get(
        "/api/v1/timeline",
        params={"entity_type": "message", "direction": "asc", "limit": 3},
    )
    assert first.status_code == 200
    payload = first.json()
    assert payload["coverage"]["status"] == "source_backed"
    assert payload["page"]["total"] == 14
    assert payload["page"]["next_cursor"]
    assert all(item["entity_type"] == "message" for item in payload["items"])
    assert all(item["source"]["relative_path"] for item in payload["items"])
    assert all(item["jump"]["conversation_id"] for item in payload["items"])
    assert all(
        item["jump"]["fragment"] == f"message-{item['entity_id']}" for item in payload["items"]
    )

    second = client.get(
        "/api/v1/timeline",
        params={
            "entity_type": "message",
            "direction": "asc",
            "limit": 3,
            "cursor": payload["page"]["next_cursor"],
        },
    )
    assert second.status_code == 200
    assert {item["id"] for item in second.json()["items"]}.isdisjoint(
        {item["id"] for item in payload["items"]}
    )
    mismatch = client.get(
        "/api/v1/timeline",
        params={"entity_type": "activity", "cursor": payload["page"]["next_cursor"]},
    )
    assert mismatch.status_code == 422

    empty = client.get(
        "/api/v1/timeline",
        params={"entity_type": "message", "q": "definitely-not-in-this-fixture"},
    ).json()
    assert empty["coverage"]["status"] == "source_backed"
    assert empty["page"]["total"] == 0
    assert empty["page"]["count_is_unknown"] is False

    message_match = client.get(
        "/api/v1/timeline",
        params={"entity_type": "message", "q": "absent"},
    )
    assert message_match.status_code == 200, message_match.text
    assert message_match.json()["page"]["total"] == 1
    assert message_match.json()["items"][0]["content"] == "message_1 is absent"

    activity_match = client.get(
        "/api/v1/timeline",
        params={"entity_type": "activity", "q": "quiet, synthetic"},
    )
    assert activity_match.status_code == 200, activity_match.text
    assert activity_match.json()["page"]["total"] == 1
    assert activity_match.json()["items"][0]["title"] == "Café fixture listing"
    assert client.get("/api/v1/timeline", params={"q": "!!!"}).status_code == 422

    timeline_sql = str(
        _timeline_union(
            q="quiet synthetic",
            entity_types=("message", "activity"),
            from_ms=None,
            to_ms=None,
        )
    ).upper()
    assert " MATCH " in timeline_sql
    assert " LIKE " not in timeline_sql

    on_this_day = client.get(
        "/api/v1/timeline/on-this-day",
        params={"month": 1, "day": 1, "timezone": "UTC", "entity_type": "message"},
    )
    assert on_this_day.status_code == 200
    assert len(on_this_day.json()["items"]) == 14
    assert on_this_day.json()["query_strategy"] == "indexed_local_day_utc_ranges"
    assert all(item["local_date"].endswith("-01-01") for item in on_this_day.json()["items"])

    activity_items = client.get(
        "/api/v1/activity",
        params={"sort": "date", "direction": "asc", "limit": 100},
    ).json()["items"]
    around_target = activity_items[-1]
    sought = client.get(
        "/api/v1/timeline",
        params={
            "entity_type": "activity",
            "direction": "asc",
            "limit": 2,
            "around": f"activity:{around_target['id']}",
        },
    )
    assert sought.status_code == 200, sought.text
    sought_payload = sought.json()
    assert sought_payload["items"][0]["entity_id"] == around_target["id"]
    assert sought_payload["around"] == {
        "requested": f"activity:{around_target['id']}",
        "entity_type": "activity",
        "entity_id": around_target["id"],
        "resolved": True,
    }
    assert (
        client.get("/api/v1/timeline", params={"around": "message:not-supported"}).status_code
        == 422
    )
    assert (
        client.get(
            "/api/v1/timeline",
            params={"around": f"activity:{'a' * 64}"},
        ).status_code
        == 404
    )
    assert (
        client.get(
            "/api/v1/timeline",
            params={
                "entity_type": "message",
                "around": f"activity:{around_target['id']}",
            },
        ).status_code
        == 422
    )

    cursor = None
    on_this_day_ids: list[str] = []
    while True:
        page = client.get(
            "/api/v1/timeline/on-this-day",
            params={
                "month": 1,
                "day": 1,
                "timezone": "UTC",
                "entity_type": "message",
                "limit": 3,
                "cursor": cursor,
            },
        )
        assert page.status_code == 200, page.text
        page_payload = page.json()
        assert page_payload["page"]["total"] == 14
        on_this_day_ids.extend(item["id"] for item in page_payload["items"])
        cursor = page_payload["page"]["next_cursor"]
        if cursor is None:
            break
    assert len(on_this_day_ids) == 14
    assert len(set(on_this_day_ids)) == 14

    first_on_this_day = client.get(
        "/api/v1/timeline/on-this-day",
        params={
            "month": 1,
            "day": 1,
            "timezone": "UTC",
            "entity_type": "message",
            "limit": 3,
        },
    ).json()
    assert (
        client.get(
            "/api/v1/timeline/on-this-day",
            params={
                "month": 1,
                "day": 2,
                "timezone": "UTC",
                "entity_type": "message",
                "limit": 3,
                "cursor": first_on_this_day["page"]["next_cursor"],
            },
        ).status_code
        == 422
    )


def test_timeline_branch_merge_matches_exhaustive_union_order_across_pages_and_filters(
    stage4_catalog: Stage4Catalog,
) -> None:
    with Session(stage4_catalog.engine) as session, session.begin():
        message = session.scalar(select(Message).order_by(Message.id).limit(1))
        activities = list(
            session.scalars(select(ActivityRecord).order_by(ActivityRecord.id).limit(2))
        )
        assert message is not None
        assert len(activities) == 2
        tied_activity, unknown_activity = activities
        tied_timestamp = 1_700_000_000_000
        message.timestamp_ms = tied_timestamp
        tied_activity.timestamp_ms = tied_timestamp
        unknown_activity.timestamp_ms = None
        message_id = message.id
        tied_activity_id = tied_activity.id
        unknown_activity_id = unknown_activity.id

    def expected_ids(
        *,
        direction: str,
        q: str | None = None,
        entity_types: tuple[str, ...] = (),
        from_ms: int | None = None,
        to_ms: int | None = None,
    ) -> list[str]:
        entries = _timeline_union(
            q=q,
            entity_types=entity_types,
            from_ms=from_ms,
            to_ms=to_ms,
        )
        identity = entries.c.entity_type + literal(":") + entries.c.entity_id
        null_rank = case((entries.c.timestamp_ms.is_(None), 1), else_=0)
        timestamp_order = (
            entries.c.timestamp_ms.asc() if direction == "asc" else entries.c.timestamp_ms.desc()
        )
        identity_order = identity.asc() if direction == "asc" else identity.desc()
        with Session(stage4_catalog.engine) as session:
            rows = list(
                session.execute(
                    select(
                        entries.c.entity_type,
                        entries.c.entity_id,
                        entries.c.timestamp_ms,
                    ).order_by(null_rank.asc(), timestamp_order, identity_order)
                )
            )
        return [f"{row.entity_type}:{row.entity_id}" for row in rows]

    scopes = [
        ({}, (), None, None, None),
        ({"entity_type": "message"}, ("message",), None, None, None),
        ({"entity_type": "activity"}, ("activity",), None, None, None),
        (
            {"from_ms": tied_timestamp, "to_ms": tied_timestamp},
            (),
            None,
            tied_timestamp,
            tied_timestamp,
        ),
        ({"entity_type": "message", "q": "absent"}, ("message",), "absent", None, None),
        (
            {"entity_type": "activity", "q": "quiet synthetic"},
            ("activity",),
            "quiet synthetic",
            None,
            None,
        ),
    ]
    for direction in ("asc", "desc"):
        for api_scope, entity_types, q, from_ms, to_ms in scopes:
            expected = expected_ids(
                direction=direction,
                q=q,
                entity_types=entity_types,
                from_ms=from_ms,
                to_ms=to_ms,
            )
            cursor: str | None = None
            observed: list[str] = []
            while True:
                response = stage4_catalog.client.get(
                    "/api/v1/timeline",
                    params={
                        **api_scope,
                        "direction": direction,
                        "limit": 2,
                        **({"cursor": cursor} if cursor is not None else {}),
                    },
                )
                assert response.status_code == 200, response.text
                payload = response.json()
                assert payload["page"]["total"] == len(expected)
                observed.extend(item["id"] for item in payload["items"])
                cursor = payload["page"]["next_cursor"]
                if cursor is None:
                    break
                assert len(observed) <= len(expected)
            assert observed == expected

    ascending = expected_ids(direction="asc")
    descending = expected_ids(direction="desc")
    message_identity = f"message:{message_id}"
    activity_identity = f"activity:{tied_activity_id}"
    assert ascending.index(activity_identity) < ascending.index(message_identity)
    assert descending.index(message_identity) < descending.index(activity_identity)
    with Session(stage4_catalog.engine) as session:
        unknown_identities = {
            f"activity:{identifier}"
            for identifier in session.scalars(
                select(ActivityRecord.id).where(ActivityRecord.timestamp_ms.is_(None))
            )
        }
    assert unknown_identities
    for ordered in (ascending, descending):
        first_unknown = min(ordered.index(identity) for identity in unknown_identities)
        assert set(ordered[first_unknown:]) == unknown_identities

    for direction in ("asc", "desc"):
        around = stage4_catalog.client.get(
            "/api/v1/timeline",
            params={
                "entity_type": "activity",
                "direction": direction,
                "around": f"activity:{unknown_activity_id}",
                "limit": 2,
            },
        )
        assert around.status_code == 200, around.text
        assert around.json()["items"][0]["id"] == f"activity:{unknown_activity_id}"


@pytest.mark.parametrize("direction", ["asc", "desc"])
def test_timeline_page_branches_use_composite_time_indexes_without_union_sort(
    stage4_catalog: Stage4Catalog,
    direction: str,
) -> None:
    statements = _timeline_page_statements(
        q=None,
        entity_types=(),
        from_ms=None,
        to_ms=None,
        direction=direction,
        candidate_limit=251,
        seeks=((1_700_000_000_000, "message:fixture-boundary", False),),
    )
    assert len(statements) == 2
    expected_indexes = (
        "ix_messages_deleted_time_id",
        "ix_activity_deleted_time_id",
    )
    with stage4_catalog.engine.connect() as connection:
        for statement, expected_index in zip(statements, expected_indexes, strict=True):
            sql = str(
                statement.compile(
                    stage4_catalog.engine,
                    compile_kwargs={"literal_binds": True},
                )
            )
            plan = [str(row[-1]) for row in connection.execute(text(f"EXPLAIN QUERY PLAN {sql}"))]
            assert "UNION" not in sql.upper()
            assert any(expected_index in step for step in plan)
            assert not any("USE TEMP B-TREE" in step for step in plan)


def test_activity_and_timeline_totals_are_revision_cached_across_cursor_pages(
    stage4_catalog: Stage4Catalog,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Stabilize both external-change signals so this test proves that an ORM commit on the app's
    # own Engine invalidates through the deterministic monotonic commit epoch alone.
    monkeypatch.setattr(aggregate_cache_module, "_catalog_data_version", lambda _session: 7)
    monkeypatch.setattr(
        aggregate_cache_module,
        "_catalog_file_state",
        lambda _database: ((1, 2, 3, 4, 5), (0, 0, 0, 0, 0)),
    )
    loads: Counter[str] = Counter()
    original = RevisionBoundAggregateCache.get_or_compute

    def tracked_get_or_compute(
        self: RevisionBoundAggregateCache,
        *,
        namespace: str,
        scope: str,
        revision: str,
        revision_is_current: Callable[[], bool],
        compute: Callable[[], object],
    ) -> object:
        def tracked_compute() -> object:
            loads[namespace] += 1
            return compute()

        return original(
            self,
            namespace=namespace,
            scope=scope,
            revision=revision,
            revision_is_current=revision_is_current,
            compute=tracked_compute,
        )

    monkeypatch.setattr(RevisionBoundAggregateCache, "get_or_compute", tracked_get_or_compute)
    client = stage4_catalog.client

    first_activity = client.get(
        "/api/v1/activity", params={"sort": "date", "direction": "desc", "limit": 2}
    )
    assert first_activity.status_code == 200, first_activity.text
    activity_payload = first_activity.json()
    assert activity_payload["page"]["next_cursor"]
    second_activity = client.get(
        "/api/v1/activity",
        params={
            "sort": "date",
            "direction": "desc",
            "limit": 2,
            "cursor": activity_payload["page"]["next_cursor"],
        },
    )
    assert second_activity.status_code == 200, second_activity.text
    # Acceptance and normal serving deliberately use the exact same revision-bound path.
    client.app.state.acceptance_read_only = True
    reordered_activity = client.get(
        "/api/v1/activity", params={"sort": "title", "direction": "asc", "limit": 2}
    )
    assert reordered_activity.status_code == 200, reordered_activity.text
    assert loads["activity-source-summary-v1"] == 1
    assert loads["activity-filter-total-v1"] == 1

    first_timeline = client.get("/api/v1/timeline", params={"direction": "desc", "limit": 2})
    assert first_timeline.status_code == 200, first_timeline.text
    timeline_payload = first_timeline.json()
    assert timeline_payload["page"]["next_cursor"]
    second_timeline = client.get(
        "/api/v1/timeline",
        params={
            "direction": "desc",
            "limit": 2,
            "cursor": timeline_payload["page"]["next_cursor"],
        },
    )
    assert second_timeline.status_code == 200, second_timeline.text
    reverse_timeline = client.get("/api/v1/timeline", params={"direction": "asc", "limit": 2})
    assert reverse_timeline.status_code == 200, reverse_timeline.text
    assert loads["timeline-source-summary-v1"] == 1
    assert loads["timeline-filter-total-v1"] == 1

    activity_only = client.get("/api/v1/timeline", params={"entity_type": "activity", "limit": 2})
    assert activity_only.status_code == 200, activity_only.text
    assert loads["timeline-source-summary-v1"] == 1
    assert loads["timeline-filter-total-v1"] == 2

    client.app.state.acceptance_read_only = False
    before_mutation = int(timeline_payload["page"]["total"])
    synthetic_request = Request({"type": "http", "app": client.app})
    with Session(stage4_catalog.engine) as session:
        revision_before = aggregate_cache_module.catalog_revision(synthetic_request, session)
    epoch = aggregate_cache_module._engine_commit_epoch(stage4_catalog.engine)
    assert aggregate_cache_module._engine_commit_epoch(stage4_catalog.engine) is epoch
    epoch_before = epoch.snapshot()
    with Session(stage4_catalog.engine) as session, session.begin():
        message = session.scalar(
            select(Message).where(Message.deleted_at.is_(None)).order_by(Message.pk).limit(1)
        )
        assert message is not None
        message.deleted_at = datetime.now(UTC)
    assert epoch.snapshot() == epoch_before + 1
    with Session(stage4_catalog.engine) as session:
        revision_after = aggregate_cache_module.catalog_revision(synthetic_request, session)
    assert revision_after != revision_before

    after_mutation = client.get("/api/v1/timeline", params={"limit": 2})
    assert after_mutation.status_code == 200, after_mutation.text
    assert after_mutation.json()["page"]["total"] == before_mutation - 1
    assert loads["timeline-source-summary-v1"] == 2
    assert loads["timeline-filter-total-v1"] == 3


def test_timeline_fails_closed_when_catalog_changes_after_a_warm_total_hit(
    stage4_catalog: Stage4Catalog,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = stage4_catalog.client
    warm = client.get("/api/v1/timeline", params={"limit": 2})
    assert warm.status_code == 200, warm.text
    warm_total = int(warm.json()["page"]["total"])

    original = RevisionBoundAggregateCache.get_or_compute
    armed = True

    def commit_after_warm_total(
        self: RevisionBoundAggregateCache,
        *,
        namespace: str,
        scope: str,
        revision: str,
        revision_is_current: Callable[[], bool],
        compute: Callable[[], object],
    ) -> object:
        nonlocal armed
        value = original(
            self,
            namespace=namespace,
            scope=scope,
            revision=revision,
            revision_is_current=revision_is_current,
            compute=compute,
        )
        if armed and namespace == "timeline-filter-total-v1":
            armed = False
            with Session(stage4_catalog.engine) as session, session.begin():
                message = session.scalar(
                    select(Message)
                    .where(Message.deleted_at.is_(None))
                    .order_by(Message.pk)
                    .limit(1)
                )
                assert message is not None
                message.deleted_at = datetime.now(UTC)
        return value

    monkeypatch.setattr(
        RevisionBoundAggregateCache,
        "get_or_compute",
        commit_after_warm_total,
    )
    raced = client.get("/api/v1/timeline", params={"limit": 250})
    assert raced.status_code == 409
    assert raced.json() == {"detail": CATALOG_CHANGED_RETRY_DETAIL}

    retried = client.get("/api/v1/timeline", params={"limit": 250})
    assert retried.status_code == 200, retried.text
    assert retried.json()["page"]["total"] == warm_total - 1


def test_event_and_people_coverage_never_infers_zero_or_friendship(
    stage4_catalog: Stage4Catalog,
) -> None:
    client = stage4_catalog.client
    events = client.get("/api/v1/events")
    assert events.status_code == 200
    event_payload = events.json()
    assert event_payload["coverage"]["status"] == "unavailable"
    assert event_payload["coverage"]["raw_count"] is None
    assert event_payload["coverage"]["logical_count"] is None
    assert event_payload["coverage"]["absence_is_not_zero_history"] is True
    assert event_payload["page"]["total"] is None
    assert event_payload["summary"] == {
        "by_status": None,
        "hosted": None,
        "attended": None,
    }
    assert client.get("/api/v1/events", params={"event_status": "invented"}).status_code == 422
    assert client.get("/api/v1/events", params={"from_ms": 20, "to_ms": 10}).status_code == 422

    contacts = client.get("/api/v1/people", params={"sort": "messages", "direction": "desc"})
    assert contacts.status_code == 200
    contact_payload = contacts.json()
    assert contact_payload["authority"] == "messenger_contact_only"
    assert contact_payload["page"]["total"] == 6
    assert all(item["is_friend"] is None for item in contact_payload["items"])
    assert all(item["friendship_status"] == "unknown" for item in contact_payload["items"])

    alex = next(item for item in contact_payload["items"] if item["name"] == "Alex Example")
    assert alex["counts"] == {"messages": 7, "photos": 2, "videos": 1, "audio": 1}
    assert alex["direct_counts"] == alex["counts"]
    assert alex["group_conversation_count"] == 0
    assert alex["conversations"][0]["href"] == f"/conversations/{alex['conversations'][0]['id']}"

    repeated = next(item for item in contact_payload["items"] if item["name"] == "Repeated Label")
    assert repeated["conversation_count"] == 1
    assert repeated["direct_conversation_count"] == 0
    assert repeated["group_conversation_count"] == 1
    assert repeated["counts"] == {"messages": 0, "photos": 0, "videos": 0, "audio": 0}
    assert repeated["direct_counts"] == repeated["counts"]
    assert repeated["shared_group_counts"]["messages"] == 1
    assert repeated["combined_thread_counts"]["messages"] == 1
    assert "not attributed to this person" in repeated["count_semantics"]

    detail = client.get(f"/api/v1/people/contacts/{alex['id']}")
    assert detail.status_code == 200
    assert detail.json()["friendship_history"]["status"] == "unavailable"
    assert detail.json()["friendship_history"]["is_friend"] is None

    empty_filter = client.get("/api/v1/people", params={"q": "not-a-contact"}).json()
    assert empty_filter["coverage"]["status"] == "derived"
    assert empty_filter["page"]["total"] == 0
    assert empty_filter["page"]["count_is_unknown"] is False

    first = client.get("/api/v1/people", params={"limit": 2}).json()
    assert first["page"]["next_cursor"]
    second = client.get(
        "/api/v1/people",
        params={"limit": 2, "cursor": first["page"]["next_cursor"]},
    )
    assert second.status_code == 200
    assert {item["id"] for item in second.json()["items"]}.isdisjoint(
        {item["id"] for item in first["items"]}
    )
    assert (
        client.get(
            "/api/v1/people",
            params={"q": "Alex", "cursor": first["page"]["next_cursor"]},
        ).status_code
        == 422
    )

    for view in (
        "friends",
        "requests",
        "followers",
        "facebook-blocks",
        "messenger-blocks",
    ):
        response = client.get(f"/api/v1/people/views/{view}")
        assert response.status_code == 200, (view, response.text)
        payload = response.json()
        assert payload["contacts_are_not_included"] is True
        assert payload["coverage"]["status"] == "unavailable"
        assert payload["page"]["total"] is None
        assert payload["page"]["count_is_unknown"] is True

    missing_context = client.get(
        "/api/v1/activity/views/posts",
        params={"post_context": "own_wall"},
    )
    assert missing_context.status_code == 200, missing_context.text
    missing_context_payload = missing_context.json()
    assert missing_context_payload["page"]["total"] == 0
    assert missing_context_payload["post_context_coverage"]["status"] == "unknown"
    assert missing_context_payload["post_context_coverage"]["absence_is_not_zero_history"] is True
    assert missing_context_payload["post_context_coverage"]["counts"]["unknown"] == 3

    group_comments = client.get(
        "/api/v1/activity/views/comments",
        params={"post_context": "group"},
    )
    assert group_comments.status_code == 200, group_comments.text
    assert group_comments.json()["page"]["total"] == 2
    assert {item["post_context"] for item in group_comments.json()["items"]} == {"group"}
    assert {item["post_context_basis"] for item in group_comments.json()["items"]} == {
        "source_subtype"
    }
    assert (
        group_comments.json()["post_context_coverage"]["classification"]
        == "source_fields_categories_and_subtypes"
    )


def test_event_attendance_requires_an_explicit_supported_source_status() -> None:
    assert (
        _event_status(
            SimpleNamespace(
                extra_json={"raw_payload": {"status": "events_you_went_to"}},
                subtype="events",
            )
        )
        == "went"
    )
    assert (
        _event_status(
            SimpleNamespace(
                extra_json={"raw_payload": {"status": "not_went"}},
                subtype="events",
            )
        )
        == "unknown"
    )


def test_positive_event_relationship_and_explicit_post_context_sources(
    stage4_positive_catalog: Stage4Catalog,
) -> None:
    client = stage4_positive_catalog.client

    coverage_items = client.get("/api/v1/analytics/coverage").json()["items"]
    coverage_by_key = {item["key"]: item for item in coverage_items}
    assert coverage_by_key["friends"]["status"] == "partial"
    assert coverage_by_key["friends"]["raw_count"] == 4
    assert coverage_by_key["friend_requests"]["raw_count"] == 1
    assert coverage_by_key["events"]["status"] == "partial"
    assert coverage_by_key["events"]["raw_count"] == 4
    assert coverage_by_key["general_posts"]["status"] == "partial"
    assert coverage_by_key["general_posts"]["raw_count"] == 7

    events = client.get(
        "/api/v1/events",
        params={"sort": "date", "direction": "asc", "limit": 2},
    )
    assert events.status_code == 200, events.text
    event_payload = events.json()
    assert event_payload["coverage"]["status"] == "partial"
    assert event_payload["page"]["total"] == 4
    assert event_payload["page"]["next_cursor"]
    assert [item["status"] for item in event_payload["items"]] == ["created", "went"]
    assert event_payload["summary"]["attended"] == 1
    assert event_payload["summary"]["hosted"] == 1

    second_events = client.get(
        "/api/v1/events",
        params={
            "sort": "date",
            "direction": "asc",
            "limit": 2,
            "cursor": event_payload["page"]["next_cursor"],
        },
    )
    assert second_events.status_code == 200, second_events.text
    assert [item["status"] for item in second_events.json()["items"]] == [
        "invited",
        "cancelled",
    ]
    assert (
        client.get("/api/v1/events", params={"event_status": "went"}).json()["page"]["total"] == 1
    )
    assert client.get("/api/v1/events", params={"hosted": False}).json()["page"]["total"] == 1

    cancelled = second_events.json()["items"][-1]
    sought_event = client.get(
        "/api/v1/events",
        params={
            "sort": "date",
            "direction": "asc",
            "limit": 1,
            "around": cancelled["id"],
        },
    )
    assert sought_event.status_code == 200, sought_event.text
    assert sought_event.json()["items"][0]["id"] == cancelled["id"]
    assert sought_event.json()["around"]["resolved"] is True
    assert (
        client.get(
            "/api/v1/events",
            params={"event_status": "went", "around": cancelled["id"]},
        ).status_code
        == 422
    )

    assert client.get("/api/v1/events", params={"around": "bad target!"}).status_code == 422
    assert client.get("/api/v1/events", params={"around": "a" * 64}).status_code == 404

    friends = client.get(
        "/api/v1/people",
        params={
            "view": "friends",
            "sort": "name",
            "direction": "asc",
            "limit": 2,
        },
    )
    assert friends.status_code == 200, friends.text
    friends_payload = friends.json()
    assert friends_payload["coverage"]["status"] == "partial"
    assert friends_payload["page"]["total"] == 4
    assert [item["name"] for item in friends_payload["items"]] == [
        "Alice Fixture",
        "Bob Fixture",
    ]
    assert friends_payload["page"]["next_cursor"]
    more_friends = client.get(
        "/api/v1/people",
        params={
            "view": "friends",
            "sort": "name",
            "direction": "asc",
            "limit": 2,
            "cursor": friends_payload["page"]["next_cursor"],
        },
    )
    assert more_friends.status_code == 200, more_friends.text
    assert [item["name"] for item in more_friends.json()["items"]] == [
        "Charlie Fixture",
        "Removed Fixture",
    ]

    bob = friends_payload["items"][1]
    sought_friend = client.get(
        "/api/v1/people",
        params={
            "view": "friends",
            "sort": "name",
            "direction": "asc",
            "limit": 1,
            "around": bob["id"],
        },
    )
    assert sought_friend.status_code == 200, sought_friend.text
    sought_friend_payload = sought_friend.json()
    assert sought_friend_payload["items"][0]["id"] == bob["id"]
    assert f"around={bob['id']}" in sought_friend_payload["items"][0]["jump"]["href"]
    assert sought_friend_payload["around"]["resolved"] is True
    assert (
        client.get(
            "/api/v1/people",
            params={"view": "friends", "q": "Alice", "around": bob["id"]},
        ).status_code
        == 422
    )
    assert (
        client.get(
            "/api/v1/people",
            params={"view": "friends", "around": "bad target!"},
        ).status_code
        == 422
    )
    assert (
        client.get(
            "/api/v1/people",
            params={"view": "friends", "around": "a" * 64},
        ).status_code
        == 404
    )
    assert (
        client.get(
            "/api/v1/people",
            params={"view": "friends", "sort": "messages"},
        ).status_code
        == 422
    )
    assert client.get("/api/v1/people", params={"sort": "date"}).status_code == 422
    requests = client.get("/api/v1/people", params={"view": "requests"})
    assert requests.status_code == 200, requests.text
    assert requests.json()["page"]["total"] == 1
    assert requests.json()["items"][0]["name"] == "Request Fixture"

    expected_contexts = {
        "own_wall",
        "others_wall",
        "group",
        "reshared",
        "tagged",
        "others_on_your_wall",
    }
    for post_context in expected_contexts:
        response = client.get(
            "/api/v1/activity/views/posts",
            params={"post_context": post_context},
        )
        assert response.status_code == 200, (post_context, response.text)
        payload = response.json()
        assert payload["page"]["total"] == 1
        assert payload["items"][0]["post_context"] == post_context
        assert payload["items"][0]["post_context_basis"] == "explicit_source_field"
        assert payload["post_context_coverage"]["status"] == "source_backed"
    unknown_posts = client.get(
        "/api/v1/activity/views/posts",
        params={"post_context": "unknown"},
    ).json()
    assert unknown_posts["page"]["total"] == 4
    assert all(item["post_context"] == "unknown" for item in unknown_posts["items"])
    other_wall_comments = client.get(
        "/api/v1/activity/views/comments",
        params={"post_context": "others_wall"},
    )
    assert other_wall_comments.status_code == 200, other_wall_comments.text
    assert other_wall_comments.json()["page"]["total"] == 1
    assert (
        client.get(
            "/api/v1/activity/views/posts",
            params={"post_context": "invented"},
        ).status_code
        == 422
    )
    assert (
        client.get(
            "/api/v1/activity/views/groups",
            params={"post_context": "group"},
        ).status_code
        == 422
    )


def test_relationship_records_link_conversation_stats_only_by_provisional_exact_label(
    stage4_positive_catalog: Stage4Catalog,
) -> None:
    with Session(stage4_positive_catalog.engine) as session, session.begin():
        relationship = session.scalar(
            select(ActivityRecord).where(ActivityRecord.title_display == "Alice Fixture")
        )
        assert relationship is not None
        relationship.title_raw = "Alex Example"
        relationship.title_display = "Alex Example"
        relationship.title_search = "alex example"

    response = stage4_positive_catalog.client.get(
        "/api/v1/people",
        params={"view": "friends", "q": "Alex Example", "sort": "name"},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["messenger_contact_matches_are_provisional"] is True
    assert payload["page"]["total"] == 1
    item = payload["items"][0]
    assert item["status"] == "your_friends"
    assert item["counts"] == {"messages": 7, "photos": 2, "videos": 1, "audio": 1}
    assert item["direct_counts"] == item["counts"]
    assert item["messenger_contact_match"]["match_basis"] == ("exact_normalized_display_label")
    assert item["messenger_contact_match"]["authoritative_identity"] is False
    assert item["conversations"][0]["href"].startswith("/conversations/")
    assert "not a Facebook identity match" in item["count_semantics"]


def test_people_projection_and_relationship_totals_reuse_across_cursor_pages(
    stage4_positive_catalog: Stage4Catalog,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projection_builds = 0
    relationship_counts = 0
    original_projection = people_routes._build_contact_projection
    original_count = people_routes._relationship_record_count

    def counted_projection(session: Session) -> people_routes._ContactProjectionSet:
        nonlocal projection_builds
        projection_builds += 1
        return original_projection(session)

    def counted_relationships(session: Session, filters: list[object]) -> int:
        nonlocal relationship_counts
        relationship_counts += 1
        return original_count(session, filters)

    monkeypatch.setattr(people_routes, "_build_contact_projection", counted_projection)
    monkeypatch.setattr(people_routes, "_relationship_record_count", counted_relationships)

    first = stage4_positive_catalog.client.get(
        "/api/v1/people",
        params={"view": "friends", "sort": "name", "direction": "asc", "limit": 2},
    )
    assert first.status_code == 200, first.text
    first_payload = first.json()
    repeated = stage4_positive_catalog.client.get(
        "/api/v1/people",
        params={"view": "friends", "sort": "name", "direction": "asc", "limit": 2},
    )
    assert repeated.status_code == 200, repeated.text
    assert repeated.json() == first_payload

    second = stage4_positive_catalog.client.get(
        "/api/v1/people",
        params={
            "view": "friends",
            "sort": "name",
            "direction": "asc",
            "limit": 2,
            "cursor": first_payload["page"]["next_cursor"],
        },
    )
    assert second.status_code == 200, second.text
    assert {item["id"] for item in second.json()["items"]}.isdisjoint(
        {item["id"] for item in first_payload["items"]}
    )

    filtered = stage4_positive_catalog.client.get(
        "/api/v1/people",
        params={
            "view": "friends",
            "q": "Fixture",
            "sort": "name",
            "direction": "asc",
            "limit": 2,
        },
    )
    assert filtered.status_code == 200, filtered.text
    filtered_more = stage4_positive_catalog.client.get(
        "/api/v1/people",
        params={
            "view": "friends",
            "q": "Fixture",
            "sort": "name",
            "direction": "asc",
            "limit": 2,
            "cursor": filtered.json()["page"]["next_cursor"],
        },
    )
    assert filtered_more.status_code == 200, filtered_more.text

    contacts = stage4_positive_catalog.client.get(
        "/api/v1/people",
        params={"view": "contacts", "sort": "name", "limit": 2},
    )
    assert contacts.status_code == 200, contacts.text
    contact_payload = contacts.json()
    first_contact = contact_payload["items"][0]
    first_contact["conversations"][0]["title"] = "locally mutated response"
    detail = stage4_positive_catalog.client.get(f"/api/v1/people/contacts/{first_contact['id']}")
    assert detail.status_code == 200, detail.text
    assert detail.json()["contact"]["conversations"][0]["title"] != "locally mutated response"

    assert projection_builds == 1
    assert relationship_counts == 2


def test_people_projection_and_relationship_totals_invalidate_after_mutation(
    stage4_positive_catalog: Stage4Catalog,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projection_builds = 0
    original_projection = people_routes._build_contact_projection

    def counted_projection(session: Session) -> people_routes._ContactProjectionSet:
        nonlocal projection_builds
        projection_builds += 1
        return original_projection(session)

    monkeypatch.setattr(people_routes, "_build_contact_projection", counted_projection)
    before = stage4_positive_catalog.client.get(
        "/api/v1/people",
        params={"view": "contacts", "q": "Alex Example"},
    )
    assert before.status_code == 200, before.text
    before_contact = before.json()["items"][0]
    conversation_id = before_contact["conversations"][0]["id"]

    with Session(stage4_positive_catalog.engine) as session, session.begin():
        conversation = session.get(Conversation, conversation_id)
        assert conversation is not None
        conversation.message_count += 5

    after = stage4_positive_catalog.client.get(
        "/api/v1/people",
        params={"view": "contacts", "q": "Alex Example"},
    )
    assert after.status_code == 200, after.text
    assert after.json()["items"][0]["counts"]["messages"] == (
        before_contact["counts"]["messages"] + 5
    )
    assert projection_builds == 2

    friends_before = stage4_positive_catalog.client.get(
        "/api/v1/people",
        params={"view": "friends", "limit": 250},
    )
    assert friends_before.status_code == 200, friends_before.text
    assert friends_before.json()["page"]["total"] == 4
    relationship_id = friends_before.json()["items"][0]["id"]
    with Session(stage4_positive_catalog.engine) as session, session.begin():
        relationship = session.scalar(
            select(ActivityRecord).where(ActivityRecord.id == relationship_id)
        )
        assert relationship is not None
        relationship.deleted_at = datetime.now(UTC)

    friends_after = stage4_positive_catalog.client.get(
        "/api/v1/people",
        params={"view": "friends", "limit": 250},
    )
    assert friends_after.status_code == 200, friends_after.text
    assert friends_after.json()["page"]["total"] == 3


def test_people_projection_invalidates_after_different_engine_sqlite_commit(
    stage4_catalog: Stage4Catalog,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projection_builds = 0
    original_projection = people_routes._build_contact_projection

    def counted_projection(session: Session) -> people_routes._ContactProjectionSet:
        nonlocal projection_builds
        projection_builds += 1
        return original_projection(session)

    monkeypatch.setattr(people_routes, "_build_contact_projection", counted_projection)
    warm = stage4_catalog.client.get(
        "/api/v1/people",
        params={"view": "contacts", "q": "Alex Example"},
    )
    assert warm.status_code == 200, warm.text
    warm_contact = warm.json()["items"][0]
    conversation_id = warm_contact["conversations"][0]["id"]

    synthetic_request = Request({"type": "http", "app": stage4_catalog.client.app})
    app_epoch = aggregate_cache_module._engine_commit_epoch(stage4_catalog.engine)
    epoch_before = app_epoch.snapshot()
    with Session(stage4_catalog.engine) as session:
        revision_before = aggregate_cache_module.catalog_revision(synthetic_request, session)

    external_engine = create_catalog_engine(stage4_catalog.layout.database)
    try:
        assert external_engine is not stage4_catalog.engine
        with Session(external_engine) as session, session.begin():
            conversation = session.get(Conversation, conversation_id)
            assert conversation is not None
            conversation.message_count += 3
    finally:
        external_engine.dispose()

    # The app Engine did not observe a local commit; file/data-version signals must invalidate it.
    assert app_epoch.snapshot() == epoch_before
    with Session(stage4_catalog.engine) as session:
        revision_after = aggregate_cache_module.catalog_revision(synthetic_request, session)
    assert revision_after != revision_before

    refreshed = stage4_catalog.client.get(
        "/api/v1/people",
        params={"view": "contacts", "q": "Alex Example"},
    )
    assert refreshed.status_code == 200, refreshed.text
    assert refreshed.json()["items"][0]["counts"]["messages"] == (
        warm_contact["counts"]["messages"] + 3
    )
    assert projection_builds == 2


def test_people_response_fails_closed_when_catalog_changes_after_projection_hit(
    stage4_catalog: Stage4Catalog,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warm = stage4_catalog.client.get(
        "/api/v1/people",
        params={"view": "contacts", "q": "Alex Example"},
    )
    assert warm.status_code == 200, warm.text
    warm_contact = warm.json()["items"][0]
    conversation_id = warm_contact["conversations"][0]["id"]
    original = people_routes._RevisionBoundPeopleProjectionCache.get_or_compute
    armed = True

    def commit_after_warm_projection(
        self: people_routes._RevisionBoundPeopleProjectionCache,
        **kwargs: object,
    ) -> object:
        nonlocal armed
        value = original(self, **kwargs)  # type: ignore[arg-type]
        if armed:
            armed = False
            with Session(stage4_catalog.engine) as session, session.begin():
                conversation = session.get(Conversation, conversation_id)
                assert conversation is not None
                conversation.message_count += 1
        return value

    monkeypatch.setattr(
        people_routes._RevisionBoundPeopleProjectionCache,
        "get_or_compute",
        commit_after_warm_projection,
    )
    raced = stage4_catalog.client.get(
        "/api/v1/people",
        params={"view": "contacts", "q": "Alex Example"},
    )
    assert raced.status_code == 409
    assert raced.json() == {"detail": CATALOG_CHANGED_RETRY_DETAIL}

    retried = stage4_catalog.client.get(
        "/api/v1/people",
        params={"view": "contacts", "q": "Alex Example"},
    )
    assert retried.status_code == 200, retried.text
    assert retried.json()["items"][0]["counts"]["messages"] == (
        warm_contact["counts"]["messages"] + 1
    )


def test_post_context_filter_matches_serialized_first_nonblank_classifier(
    stage4_positive_catalog: Stage4Catalog,
) -> None:
    client = stage4_positive_catalog.client
    original = client.get(
        "/api/v1/activity/views/posts", params={"post_context": "own_wall"}
    ).json()["items"][0]
    with Session(stage4_positive_catalog.engine) as session, session.begin():
        row = session.scalar(select(ActivityRecord).where(ActivityRecord.id == original["id"]))
        assert row is not None
        raw_payload = dict(row.extra_json.get("raw_payload", {}))
        raw_payload["activity_type"] = "   "
        raw_payload["post_type"] = "Your / Timeline"
        row.extra_json = {**row.extra_json, "raw_payload": raw_payload}

    response = client.get("/api/v1/activity/views/posts", params={"post_context": "own_wall"})
    assert response.status_code == 200, response.text
    item = next(value for value in response.json()["items"] if value["id"] == original["id"])
    assert item["post_context"] == "own_wall"
    assert item["post_context_basis"] == "explicit_source_field"


def test_support_threads_are_browsable_separately(stage4_catalog: Stage4Catalog) -> None:
    response = stage4_catalog.client.get("/api/v1/support", params={"limit": 1})
    assert response.status_code == 200
    payload = response.json()
    assert payload["coverage"]["status"] == "available"
    assert payload["page"]["total"] == 2
    assert len(payload["items"]) == 1
    assert payload["page"]["next_cursor"]
    with Session(stage4_catalog.engine) as session, session.begin():
        session.add(
            Conversation(
                id="support-inserted-after-page-one",
                conversation_key="support-inserted-after-page-one",
                title_raw="Newer Inserted Support Case",
                title_display="Newer Inserted Support Case",
                title_search="newer inserted support case",
                conversation_type="direct",
                primary_category="support",
                participant_count=1,
                first_activity_ms=9_000_000_000_000,
                last_activity_ms=9_000_000_000_000,
                message_count=0,
                sent_count=0,
                received_count=0,
                attachment_count=0,
                unique_attachment_count=0,
                attachment_bytes=0,
                active_days=1,
                completeness="complete",
            )
        )
        session.add(
            ConversationSegment(
                id="support-inserted-segment",
                conversation_id="support-inserted-after-page-one",
                thread_path="support/inserted-after-page-one",
                title_raw="Newer Inserted Support Case",
                title_display="Newer Inserted Support Case",
                title_search="newer inserted support case",
                source_category="support",
                sequence=0,
                source_chunk_count=1,
                first_activity_ms=9_000_000_000_000,
                last_activity_ms=9_000_000_000_000,
                completeness="complete",
                metadata_json={},
            )
        )
    second_response = stage4_catalog.client.get(
        "/api/v1/support",
        params={"limit": 1, "cursor": payload["page"]["next_cursor"]},
    )
    assert second_response.status_code == 200, second_response.text
    second_payload = second_response.json()
    assert second_payload["page"]["total"] == 3
    assert second_payload["page"]["next_cursor"] is None
    items = [*payload["items"], *second_payload["items"]]
    assert {item["title"] for item in items} == {
        "Support Case One",
        "Support Case Two",
    }
    assert sum(item["message_count"] for item in items) == 3
    assert all(item["href"] == f"/conversations/{item['id']}" for item in items)
    assert "Newer Inserted Support Case" not in {item["title"] for item in items}
    refreshed = stage4_catalog.client.get("/api/v1/support", params={"limit": 1})
    assert refreshed.status_code == 200, refreshed.text
    assert refreshed.json()["items"][0]["title"] == "Newer Inserted Support Case"
    assert (
        stage4_catalog.client.get(
            "/api/v1/support",
            params={"limit": 2, "cursor": payload["page"]["next_cursor"]},
        ).status_code
        == 422
    )
    assert (
        stage4_catalog.client.get(
            "/api/v1/support", params={"cursor": "not-a-valid-cursor"}
        ).status_code
        == 422
    )


def test_topic_clusters_have_explicit_build_state_and_live_source_jumps(
    stage4_catalog: Stage4Catalog,
) -> None:
    client = stage4_catalog.client
    engine = stage4_catalog.engine
    unavailable = client.get("/api/v1/analytics/topics")
    assert unavailable.status_code == 200
    assert unavailable.json()["status"] == "unavailable"
    assert unavailable.json()["ready"] is False
    assert client.post("/api/v1/analytics/topics/build", json={}).status_code == 409

    with Session(engine) as session, session.begin():
        message = session.scalar(select(Message).where(Message.deleted_at.is_(None)))
        assert message is not None
        message_id = message.id
        conversation_id = message.conversation_id
        config_id = "a" * 64
        window_id = "b" * 64
        session.add(
            SemanticIndexConfig(
                id=config_id,
                active=True,
                status="ready",
                backend="fixture",
                backend_version="fixture-1",
                model_name="fixture-embedding-model",
                model_revision="fixture-revision",
                dimensions=3,
                distance_metric="cosine",
                document_prefix="",
                query_prefix="",
                vector_table="fixture_vectors",
                window_version="fixture-windows",
                window_count=1,
                audit_json={},
                completed_at=datetime.now(UTC),
            )
        )
        session.add(
            SearchWindow(
                id=window_id,
                window_version="fixture-windows",
                entity_type="message_window",
                conversation_id=conversation_id,
                segment_id=message.segment_id,
                first_message_id=message_id,
                last_message_id=message_id,
                first_timestamp_ms=message.timestamp_ms,
                last_timestamp_ms=message.timestamp_ms,
                turn_count=1,
                approx_token_count=5,
                text_display="Representative retained fixture conversation window.",
                text_search="representative retained fixture conversation window",
                source_category="fixture",
                participants_search=message.sender_search,
                source_ids_json={},
                build_generation="fixture-generation",
                embedding_config_hash=config_id,
                embedding_updated_at=datetime.now(UTC),
            )
        )
        session.flush()
        session.add(SearchWindowMessage(window_id=window_id, message_id=message_id, ordinal=0))
        session.add(
            Job(
                id="fixture-topic-job",
                kind="topic_clusters",
                status="completed",
                stage="Topic clusters ready",
                progress_current=1,
                progress_total=1,
                checkpoint_json={"semantic_config_id": config_id},
                counters_json={
                    "result": {
                        "pipeline_version": "topic-clusters-v1",
                        "semantic_config_id": config_id,
                        "generated_at": datetime.now(UTC).isoformat(),
                        "corpus_window_count": 1,
                        "sample_count": 1,
                        "sampling": "Fixture sample.",
                        "warning": "Unsupervised estimate.",
                        "clusters": [
                            {
                                "id": "fixture-cluster",
                                "ordinal": 1,
                                "sample_windows": 1,
                                "sample_share": 1.0,
                                "terms": ["fixture conversation"],
                                "representative_window_ids": [window_id],
                            }
                        ],
                    }
                },
                finished_at=datetime.now(UTC),
            )
        )

    response = client.get("/api/v1/analytics/topics")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["ready"] is True
    assert payload["result"]["sample_count"] == 1
    cluster = payload["result"]["clusters"][0]
    assert cluster["terms"] == ["fixture conversation"]
    representative = cluster["representatives"][0]
    assert representative["window_id"] == window_id
    assert representative["jump"]["href"].startswith(
        f"/conversations/{conversation_id}?around={message_id}"
    )

    with Session(engine) as session, session.begin():
        retained_message = session.scalar(select(Message).where(Message.id == message_id))
        assert retained_message is not None
        retained_message.deleted_at = datetime.now(UTC)
    filtered = client.get("/api/v1/analytics/topics").json()
    assert filtered["result"]["clusters"][0]["representatives"] == []


def test_nlp_trends_remap_versioned_utc_rollups_with_samples_and_direction(
    stage4_catalog: Stage4Catalog,
) -> None:
    engine = stage4_catalog.engine
    hour_start_ms = 1_699_999_200_000
    rows = [
        (
            "sentiment",
            "owner",
            "positive",
            2,
            1.6,
            {"positive": 1.4, "neutral": 0.4, "negative": 0.2},
        ),
        (
            "sentiment",
            "other",
            "negative",
            1,
            0.8,
            {"positive": 0.1, "neutral": 0.1, "negative": 0.8},
        ),
        ("emotion", "owner", "joy", 2, 1.5, {"joy": 1.5, "neutral": 0.5}),
        ("toxicity", "other", "toxic", 1, 0.7, {"toxic": 0.7, "insult": 0.2}),
    ]
    with Session(engine) as session, session.begin():
        for index, (task, direction, label, count, confidence_sum, probabilities) in enumerate(
            rows
        ):
            session.add(
                NlpHourlyRollup(
                    id=f"rollup-{index}",
                    task=task,
                    hour_start_ms=hour_start_ms,
                    direction=direction,
                    pipeline_version="fixture-nlp-v1",
                    model_name=f"fixture-{task}",
                    model_revision="fixture-revision",
                    configuration_sha256="c" * 64,
                    configuration_json={"time_bucket": "UTC hour"},
                    predicted_label=label,
                    sample_count=count,
                    confidence_sum=confidence_sum,
                    probability_sums_json={"labels": probabilities},
                    active=True,
                    generated_at=datetime.now(UTC),
                )
            )

    response = stage4_catalog.client.get(
        "/api/v1/analytics/nlp-trends",
        params={"bucket": "hour", "timezone": "UTC"},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["source_sample_count"] == 6
    sentiment = payload["series"]["sentiment"]
    combined = next(row for row in sentiment if row["direction"] == "all")
    assert combined["sample_count"] == 3
    assert combined["dominant_label"] == "positive"
    assert combined["mean_confidence"] == pytest.approx(0.8)
    assert combined["score"] == pytest.approx((1.5 - 1.0) / 3)
    assert {row["direction"] for row in sentiment} == {"owner", "other", "all"}
    assert payload["series"]["emotion"][0]["dominant_label"] == "joy"
    sentiment_hour = next(
        row for row in payload["hour_of_day"]["sentiment"] if row["direction"] == "all"
    )
    assert sentiment_hour["hour"] == "22"
    assert sentiment_hour["sample_count"] == 3
    toxicity = next(row for row in payload["series"]["toxicity"] if row["direction"] == "all")
    assert toxicity["score"] == pytest.approx(0.7)
    assert payload["availability"]["sentiment"]["status"] == "available"
    assert payload["availability"]["toxicity"]["relationship_ranking_enabled"] is False
    assert any("No person" in warning for warning in payload["warnings"])

    partial_hour = stage4_catalog.client.get(
        "/api/v1/analytics/nlp-trends",
        params={
            "bucket": "hour",
            "timezone": "Asia/Kathmandu",
            "from_ms": hour_start_ms + 1,
            "to_ms": hour_start_ms + 2,
        },
    ).json()
    assert partial_hour["source_sample_count"] == 6
    assert partial_hour["rollup_provenance"]["partial_boundary_hours_included"] is True
    assert partial_hour["rollup_provenance"]["non_whole_hour_timezone_offset_observed"] is True
    assert any("partial boundary hours" in warning for warning in partial_hour["warnings"])
    assert any("non-whole-hour" in warning for warning in partial_hour["warnings"])

    owner = stage4_catalog.client.get(
        "/api/v1/analytics/nlp-trends",
        params={"bucket": "month", "timezone": "America/New_York", "direction": "owner"},
    ).json()
    assert {row["direction"] for row in owner["series"]["sentiment"]} == {"owner"}
    for bucket in ("day", "week", "year"):
        bucketed = stage4_catalog.client.get(
            "/api/v1/analytics/nlp-trends",
            params={"bucket": bucket, "timezone": "UTC"},
        )
        assert bucketed.status_code == 200, (bucket, bucketed.text)
        assert bucketed.json()["source_sample_count"] == 6
        assert bucketed.json()["series"]["sentiment"]
    assert (
        stage4_catalog.client.get(
            "/api/v1/analytics/nlp-trends", params={"timezone": "Not/A_Timezone"}
        ).status_code
        == 422
    )
    assert (
        stage4_catalog.client.get(
            "/api/v1/analytics/nlp-trends", params={"from_ms": 20, "to_ms": 10}
        ).status_code
        == 422
    )


def test_analytics_period_cursors_are_continuous_and_scope_bound(
    stage4_catalog: Stage4Catalog,
) -> None:
    with Session(stage4_catalog.engine) as session, session.begin():
        messages = list(session.scalars(select(Message).order_by(Message.pk)))
        assert len(messages) >= 3
        for index, message in enumerate(messages):
            message.timestamp_ms = 1_672_531_200_000 + index * 86_400_000

    parameters: dict[str, object] = {"bucket": "day", "timezone": "UTC", "limit": 1}
    seen_periods: list[str] = []
    cursor: str | None = None
    full_summary: dict[str, object] | None = None
    first_cursor: str | None = None
    while True:
        response = stage4_catalog.client.get(
            "/api/v1/analytics/messages",
            params={**parameters, **({"cursor": cursor} if cursor else {})},
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["page"]["limit"] == 1
        assert payload["page"]["returned"] == len(payload["series"]) <= 1
        assert full_summary is None or payload["summary"] == full_summary
        full_summary = payload["summary"]
        seen_periods.extend(row["period"] for row in payload["series"])
        cursor = payload["page"]["next_cursor"]
        first_cursor = first_cursor or cursor
        if cursor is None:
            break
    assert len(seen_periods) == len(set(seen_periods)) == full_summary["observed_periods"]
    assert first_cursor is not None
    assert (
        stage4_catalog.client.get(
            "/api/v1/analytics/messages",
            params={**parameters, "timezone": "America/New_York", "cursor": first_cursor},
        ).status_code
        == 422
    )
    assert (
        stage4_catalog.client.get(
            "/api/v1/analytics/messages",
            params={**parameters, "limit": 2, "cursor": first_cursor},
        ).status_code
        == 422
    )


def test_nlp_trend_cursor_bounds_scanned_periods_and_rejects_scope_mismatch(
    stage4_catalog: Stage4Catalog,
) -> None:
    with Session(stage4_catalog.engine) as session, session.begin():
        for index in range(3):
            session.add(
                NlpHourlyRollup(
                    id=f"paged-rollup-{index}",
                    task="sentiment",
                    hour_start_ms=1_672_531_200_000 + index * 86_400_000,
                    direction="owner",
                    pipeline_version="fixture-nlp-v1",
                    model_name="fixture-sentiment",
                    model_revision="fixture-revision",
                    configuration_sha256="d" * 64,
                    configuration_json={"time_bucket": "UTC hour"},
                    predicted_label="positive",
                    sample_count=index + 1,
                    confidence_sum=0.8 * (index + 1),
                    probability_sums_json={
                        "labels": {"positive": 0.7 * (index + 1), "negative": 0.3 * (index + 1)}
                    },
                    active=True,
                    generated_at=datetime.now(UTC),
                )
            )

    parameters: dict[str, object] = {
        "bucket": "day",
        "timezone": "UTC",
        "direction": "owner",
        "limit": 1,
    }
    first = stage4_catalog.client.get("/api/v1/analytics/nlp-trends", params=parameters)
    assert first.status_code == 200, first.text
    first_payload = first.json()
    assert first_payload["source_sample_count"] == 6
    assert first_payload["page_sample_count"] == 1
    assert first_payload["page"]["total"] is None
    assert first_payload["page"]["count_is_unknown"] is True
    assert first_payload["page"]["returned"] == 1
    cursor = first_payload["page"]["next_cursor"]
    assert cursor
    second = stage4_catalog.client.get(
        "/api/v1/analytics/nlp-trends", params={**parameters, "cursor": cursor}
    )
    assert second.status_code == 200, second.text
    second_payload = second.json()
    assert second_payload["source_sample_count"] == 6
    assert second_payload["page_sample_count"] == 2
    assert {
        first_payload["series"]["sentiment"][0]["period"],
        second_payload["series"]["sentiment"][0]["period"],
    } == {"2023-01-01", "2023-01-02"}
    assert (
        stage4_catalog.client.get(
            "/api/v1/analytics/nlp-trends",
            params={**parameters, "direction": "all", "cursor": cursor},
        ).status_code
        == 422
    )
    assert (
        stage4_catalog.client.get(
            "/api/v1/analytics/nlp-trends",
            params={**parameters, "limit": 2, "cursor": cursor},
        ).status_code
        == 422
    )


def test_analytics_prefilters_link_candidates_and_streams_conversation_order(
    stage4_catalog: Stage4Catalog,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with Session(stage4_catalog.engine) as session:
        active_messages = int(
            session.scalar(select(func.count(Message.pk)).where(Message.deleted_at.is_(None))) or 0
        )
        link_candidates = int(
            session.scalar(
                select(func.count(Message.pk)).where(
                    Message.deleted_at.is_(None),
                    analytics_routes._message_has_link_candidate(),
                )
            )
            or 0
        )
        query_plan = [
            str(row[-1])
            for row in session.execute(
                text(
                    "EXPLAIN QUERY PLAN "
                    "SELECT conversation_id, timestamp_ms, sender_is_owner, sender_search "
                    "FROM messages WHERE +deleted_at IS NULL "
                    "ORDER BY conversation_id, timestamp_ms, pk"
                )
            )
        ]

    assert 0 < link_candidates < active_messages
    assert any("ix_messages_conversation_time" in step for step in query_plan)
    assert not any("ix_messages_deleted" in step for step in query_plan)

    parsed_rows = 0
    parse = analytics_routes._message_link_hostnames

    def counted_parse(content: str | None, extra: object) -> list[str]:
        nonlocal parsed_rows
        parsed_rows += 1
        return parse(content, extra)

    monkeypatch.setattr(analytics_routes, "_message_link_hostnames", counted_parse)
    response = stage4_catalog.client.get("/api/v1/analytics/overview")
    assert response.status_code == 200, response.text
    assert parsed_rows == link_candidates
    assert any(row["domain"] == "example.invalid" for row in response.json()["link_domains"])


def test_analytics_nlp_status_uses_complete_active_rollups_without_annotation_rescan(
    stage4_catalog: Stage4Catalog,
) -> None:
    with Session(stage4_catalog.engine) as session, session.begin():
        for index, task in enumerate(("sentiment", "emotion", "toxicity")):
            session.add(
                NlpHourlyRollup(
                    id=f"status-rollup-{task}",
                    task=task,
                    hour_start_ms=1_700_000_000_000,
                    direction="owner",
                    pipeline_version="fixture-nlp-v1",
                    model_name=f"fixture-{task}",
                    model_revision="fixture-revision",
                    configuration_sha256=f"{index + 1}" * 64,
                    configuration_json={"time_bucket": "UTC hour"},
                    predicted_label="fixture-label",
                    sample_count=index + 2,
                    confidence_sum=1.0,
                    probability_sums_json={"labels": {"fixture-label": 1.0}},
                    active=True,
                    generated_at=datetime.now(UTC),
                )
            )

    statements: list[str] = []

    def record_statement(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        statements.append(statement)

    event.listen(stage4_catalog.engine, "before_cursor_execute", record_statement)
    try:
        response = stage4_catalog.client.get("/api/v1/analytics/overview")
    finally:
        event.remove(stage4_catalog.engine, "before_cursor_execute", record_statement)

    assert response.status_code == 200, response.text
    derived = response.json()["derived_nlp"]
    assert [derived[task]["annotation_count"] for task in ("sentiment", "emotion", "toxicity")] == [
        2,
        3,
        4,
    ]
    assert not any("FROM nlp_annotations" in statement for statement in statements)


def test_analytics_match_backing_rows_and_keep_nlp_unavailable(
    stage4_catalog: Stage4Catalog,
) -> None:
    client = stage4_catalog.client
    engine = stage4_catalog.engine
    with Session(engine) as session:
        expected = {
            "messages": session.scalar(
                select(func.count(Message.pk)).where(Message.deleted_at.is_(None))
            )
            or 0,
            "activities": session.scalar(
                select(func.count(ActivityRecord.pk)).where(ActivityRecord.deleted_at.is_(None))
            )
            or 0,
            "dated_activities": session.scalar(
                select(func.count(ActivityRecord.pk)).where(
                    ActivityRecord.deleted_at.is_(None),
                    ActivityRecord.timestamp_ms.is_not(None),
                )
            )
            or 0,
            "attachments": session.scalar(
                select(func.count(AttachmentOccurrence.id)).where(
                    AttachmentOccurrence.deleted_at.is_(None)
                )
            )
            or 0,
            "attachment_bytes": session.scalar(
                select(func.coalesce(func.sum(AttachmentOccurrence.size_bytes), 0)).where(
                    AttachmentOccurrence.deleted_at.is_(None)
                )
            )
            or 0,
            "dated_attachments": session.scalar(
                select(func.count(AttachmentOccurrence.id)).where(
                    AttachmentOccurrence.deleted_at.is_(None),
                    AttachmentOccurrence.timestamp_ms.is_not(None),
                )
            )
            or 0,
            "reactions": session.scalar(
                select(func.count(MessageReaction.id))
                .join(Message, Message.id == MessageReaction.message_id)
                .where(Message.deleted_at.is_(None))
            )
            or 0,
            "conversations": session.scalar(
                select(func.count(Conversation.id)).where(Conversation.deleted_at.is_(None))
            )
            or 0,
            "messaged_conversations": session.scalar(
                select(func.count(func.distinct(Message.conversation_id))).where(
                    Message.deleted_at.is_(None)
                )
            )
            or 0,
            "source_files": session.scalar(
                select(func.count(SourceFile.id)).where(SourceFile.scan_status == "discovered")
            )
            or 0,
            "blobs": session.scalar(select(func.count(Blob.sha256))) or 0,
        }

    overview = client.get("/api/v1/analytics/overview")
    assert overview.status_code == 200
    overview_payload = overview.json()
    assert overview_payload["messages"]["total"] == expected["messages"]
    assert (
        overview_payload["messages"]["sent"] + overview_payload["messages"]["received"]
        == expected["messages"]
    )
    assert overview_payload["activity_records"] == expected["activities"]
    assert overview_payload["attachments"]["occurrences"] == expected["attachments"]
    assert overview_payload["attachments"]["occurrence_bytes"] == expected["attachment_bytes"]
    assert (
        sum(overview_payload["interactions"]["reaction_types"].values())
        == (overview_payload["interactions"]["message_reactions"])
    )
    assert overview_payload["interactions"]["calls"] == 1
    assert overview_payload["interactions"]["call_duration_seconds"] == 12
    assert overview_payload["interactions"]["audio_attachments"] == 1
    assert {"domain": "example.invalid", "occurrences": 1} in overview_payload["link_domains"]
    assert {key: value["status"] for key, value in overview_payload["derived_nlp"].items()} == {
        "sentiment": "unavailable",
        "emotion": "unavailable",
        "toxicity": "unavailable",
        "conflict": "unavailable",
    }
    assert overview_payload["derived_nlp"]["toxicity"]["relationship_ranking_enabled"] is False
    assert overview_payload["derived_nlp"]["conflict"]["relationship_ranking_enabled"] is False
    los_angeles_overview = client.get(
        "/api/v1/analytics/overview", params={"timezone": "America/Los_Angeles"}
    ).json()
    first_timestamp = los_angeles_overview["messages"]["first_timestamp_ms"]
    last_timestamp = los_angeles_overview["messages"]["last_timestamp_ms"]
    assert first_timestamp is not None and last_timestamp is not None
    zone = ZoneInfo("America/Los_Angeles")
    expected_calendar_days = (
        datetime.fromtimestamp(last_timestamp / 1000, zone).date()
        - datetime.fromtimestamp(first_timestamp / 1000, zone).date()
    ).days + 1
    assert los_angeles_overview["timezone"] == "America/Los_Angeles"
    assert los_angeles_overview["messages"]["calendar_days"] == expected_calendar_days

    for bucket in ("day", "week", "month", "year"):
        response = client.get(
            "/api/v1/analytics/messages",
            params={"bucket": bucket, "timezone": "UTC"},
        )
        assert response.status_code == 200, (bucket, response.text)
        payload = response.json()
        assert sum(item["messages"] for item in payload["series"]) == expected["messages"]
        assert (
            sum(item["sent"] + item["received"] for item in payload["series"])
            == expected["messages"]
        )
    assert (
        client.get(
            "/api/v1/analytics/messages",
            params={"timezone": "Not/A_Timezone"},
        ).status_code
        == 422
    )
    assert (
        client.get(
            "/api/v1/analytics/messages",
            params={"from_ms": 20, "to_ms": 10},
        ).status_code
        == 422
    )

    patterns = client.get(
        "/api/v1/analytics/activity-patterns",
        params={"timezone": "UTC"},
    ).json()
    assert sum(item["messages"] for item in patterns["heatmap"]) == expected["messages"]
    assert sum(item["activities"] for item in patterns["heatmap"]) == expected["dated_activities"]
    assert sum(item["messages"] for item in patterns["hours"]) == expected["messages"]
    assert (
        sum(item["sent"] + item["received"] for item in patterns["hours"]) == expected["messages"]
    )
    assert sum(item["messages"] for item in patterns["weekdays"]) == expected["messages"]
    assert patterns["streaks"]["active_days"] == len(patterns["heatmap"])
    assert patterns["streaks"]["longest_days"] >= 1
    assert len(patterns["peak_days"]) <= 10
    assert [item["total"] for item in patterns["peak_days"]] == sorted(
        (item["total"] for item in patterns["peak_days"]), reverse=True
    )
    assert patterns["summary"] == {
        "messages_in_scope": expected["messages"],
        "dated_activity_records_in_scope": expected["dated_activities"],
        "active_activity_records_archive_wide": expected["activities"],
        "unknown_date_activity_records": expected["activities"] - expected["dated_activities"],
        "unknown_date_scope": "archive_wide",
        "unknown_date_included_in_time_buckets": False,
        "date_filter_active": False,
    }
    assert "inclusive at millisecond precision" in patterns["range_provenance"]
    assert "cannot be tested" in patterns["range_provenance"]
    with Session(engine) as session:
        boundary_ms = session.scalar(
            select(func.min(ActivityRecord.timestamp_ms)).where(
                ActivityRecord.deleted_at.is_(None),
                ActivityRecord.timestamp_ms.is_not(None),
            )
        )
        assert boundary_ms is not None
        boundary_count = int(
            session.scalar(
                select(func.count(ActivityRecord.pk)).where(
                    ActivityRecord.deleted_at.is_(None),
                    ActivityRecord.timestamp_ms == boundary_ms,
                )
            )
            or 0
        )
    exact_boundary_patterns = client.get(
        "/api/v1/analytics/activity-patterns",
        params={"timezone": "UTC", "from_ms": boundary_ms, "to_ms": boundary_ms},
    ).json()
    assert exact_boundary_patterns["summary"]["dated_activity_records_in_scope"] == (boundary_count)
    assert exact_boundary_patterns["summary"]["unknown_date_activity_records"] == (
        expected["activities"] - expected["dated_activities"]
    )
    assert exact_boundary_patterns["summary"]["date_filter_active"] is True

    interactions_response = client.get(
        "/api/v1/analytics/interaction-patterns",
        params={"bucket": "month", "timezone": "UTC"},
    )
    assert interactions_response.status_code == 200, interactions_response.text
    interactions = interactions_response.json()
    assert (
        interactions["initiations"]["owner"] + interactions["initiations"]["other"]
        == (expected["messaged_conversations"])
    )
    for direction in ("owner_replied", "other_replied"):
        response = interactions["responses"][direction]
        assert sum(response["buckets"].values()) == response["count"]
        assert response["median_ms"] is None or response["median_ms"] >= 0
        assert response["p90_ms"] is None or response["p90_ms"] >= response["median_ms"]
    assert sum(item["participants"] for item in interactions["active_participants"]) > 0
    assert "not used to rank people" in interactions["ranking_policy"]
    assert (
        client.get(
            "/api/v1/analytics/interaction-patterns",
            params={"from_ms": 20, "to_ms": 10},
        ).status_code
        == 422
    )

    for interaction_bucket in ("day", "week", "month", "year"):
        interaction_series_response = client.get(
            "/api/v1/analytics/interaction-series",
            params={"bucket": interaction_bucket, "timezone": "UTC"},
        )
        assert interaction_series_response.status_code == 200, (
            interaction_bucket,
            interaction_series_response.text,
        )
        interaction_series = interaction_series_response.json()
        assert sum(row["calls"] for row in interaction_series["series"]) == 1
        assert sum(row["call_duration_seconds"] for row in interaction_series["series"]) == 12
        assert (
            sum(row["reactions"] for row in interaction_series["series"]) == expected["reactions"]
        )
        assert all(
            row["reaction_explicit_timestamp"] + row["reaction_message_timestamp_fallback"]
            == row["reactions"]
            for row in interaction_series["series"]
        )
        assert (
            interaction_series["summary"]["reaction_explicit_timestamp"]
            + interaction_series["summary"]["reaction_message_timestamp_fallback"]
            == expected["reactions"]
        )
        assert (
            sum(row["audio_attachments"] for row in interaction_series["series"])
            + interaction_series["summary"]["audio_unknown_date"]
            == overview_payload["interactions"]["audio_attachments"]
        )
        assert (
            sum(row["link_occurrences"] for row in interaction_series["series"])
            == interaction_series["summary"]["link_occurrences"]
        )
        assert any(row["domain"] == "example.invalid" for row in interaction_series["domains"])
        assert all(row["periods"] for row in interaction_series["domains"])
        assert "fallback" in interaction_series["provenance"]["reactions"]
        assert "archive-wide count" in interaction_series["provenance"]["audio"]
        assert "never assigned" in interaction_series["provenance"]["audio"]
    assert (
        client.get(
            "/api/v1/analytics/interaction-series",
            params={"timezone": "Not/A_Timezone"},
        ).status_code
        == 422
    )
    assert (
        client.get(
            "/api/v1/analytics/interaction-series",
            params={"from_ms": 20, "to_ms": 10},
        ).status_code
        == 422
    )

    network_response = client.get("/api/v1/analytics/network", params={"limit": 5})
    assert network_response.status_code == 200, network_response.text
    network = network_response.json()
    node_ids = {node["id"] for node in network["nodes"]}
    assert network["conversation_count"] == min(5, expected["conversations"])
    assert network["participant_label_count"] > 0
    assert network["edges"]
    assert all(
        edge["source"] in node_ids and edge["target"] in node_ids for edge in network["edges"]
    )
    assert "not Facebook identity IDs" in network["caveat"]

    attachments = client.get("/api/v1/analytics/attachments").json()
    assert attachments["bucket"] == "month"
    assert attachments["timezone"] == "America/New_York"
    assert sum(item["occurrences"] for item in attachments["types"]) == expected["attachments"]
    assert (
        sum(item["occurrence_bytes"] for item in attachments["types"])
        == expected["attachment_bytes"]
    )
    assert {item["type"] for item in attachments["types"]} >= {
        "photo",
        "video",
        "audio",
        "gif",
        "sticker",
        "file",
    }
    assert attachments["summary"]["occurrences"] == expected["attachments"]
    assert attachments["summary"]["dated_occurrences"] == expected["dated_attachments"]
    assert (
        attachments["summary"]["unknown_date_occurrences"]
        == expected["attachments"] - expected["dated_attachments"]
    )
    assert (
        sum(item["occurrences"] for item in attachments["trends"]) == expected["dated_attachments"]
    )
    assert all(
        sum(value["occurrences"] for value in item["by_type"].values()) == item["occurrences"]
        for item in attachments["trends"]
    )
    assert "unknown-date attachments" in attachments["provenance"]
    assert attachments["summary"]["unknown_date_scope"] == "archive_wide"
    assert attachments["summary"]["unknown_date_included_in_totals"] is True
    assert "occurrences without a blob identity count separately" in attachments["count_semantics"]
    ranged_attachments = client.get(
        "/api/v1/analytics/attachments",
        params={"from_ms": 0, "to_ms": 9_999_999_999_999},
    ).json()
    assert ranged_attachments["summary"]["occurrences"] == expected["dated_attachments"]
    assert ranged_attachments["summary"]["dated_occurrences"] == expected["dated_attachments"]
    assert ranged_attachments["summary"]["unknown_date_occurrences"] == (
        expected["attachments"] - expected["dated_attachments"]
    )
    assert ranged_attachments["summary"]["unknown_date_included_in_totals"] is False
    assert "archive-wide" in ranged_attachments["provenance"]
    assert (
        client.get(
            "/api/v1/analytics/attachments",
            params={"from_ms": 20, "to_ms": 10},
        ).status_code
        == 422
    )

    conversations = client.get("/api/v1/analytics/conversations").json()
    assert conversations["conversation_count"] == expected["conversations"]
    assert sum(conversations["by_type"].values()) == expected["conversations"]
    assert "No people or relationships are ranked" in conversations["ranking_policy"]
    assert conversations["derived_nlp"]["toxicity"]["status"] == "unavailable"

    storage = client.get("/api/v1/analytics/storage").json()
    assert storage["source_occurrences"]["files"] == expected["source_files"]
    assert storage["unique_blobs"]["files"] == expected["blobs"]
    assert sum(item["files"] for item in storage["by_media_class"]) == expected["blobs"]

    coverage = client.get("/api/v1/analytics/coverage").json()
    coverage_by_key = {item["key"]: item for item in coverage["items"]}
    for key in ("friends", "events", "general_posts"):
        assert coverage_by_key[key]["status"] == "unavailable"
        assert coverage_by_key[key]["raw_count"] is None
        assert coverage_by_key[key]["logical_count"] is None
        assert coverage_by_key[key]["absence_is_not_zero_history"] is True
    assert coverage["history"]["mode"] == "single_observed_export_snapshot"
    assert coverage["history"]["historical_comparison_available"] is False
    assert "does not contain prior export snapshots" in coverage["history"]["description"]
    assert coverage["history"]["observed_at"]
    assert "not the export creation time" in coverage["history"]["observed_at_basis"]
    assert len(coverage["history"]["snapshots"]) == 1
    snapshot = coverage["history"]["snapshots"][0]
    assert snapshot["observed_at"] == coverage["history"]["observed_at"]
    assert snapshot["summary"] == coverage["summary"]
    assert len(snapshot["items"]) == len(coverage["items"])

    with Session(engine) as session:
        assert (
            session.scalar(select(func.count(Message.pk)).where(Message.deleted_at.is_(None)))
            == expected["messages"]
        )
        assert (
            session.scalar(
                select(func.count(ActivityRecord.pk)).where(ActivityRecord.deleted_at.is_(None))
            )
            == expected["activities"]
        )


def test_interaction_series_response_is_revision_cached(
    stage4_catalog: Stage4Catalog,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params = {
        "bucket": "year",
        "timezone": "UTC",
        "domain_limit": 7,
    }
    first = stage4_catalog.client.get(
        "/api/v1/analytics/interaction-series",
        params=params,
    )
    assert first.status_code == 200, first.text

    def unexpected_link_rescan(_content: str | None, _extra: object) -> list[str]:
        raise AssertionError("A revision-matched response should be served from the warm cache")

    monkeypatch.setattr(
        analytics_routes,
        "_message_link_hostnames",
        unexpected_link_rescan,
    )
    second = stage4_catalog.client.get(
        "/api/v1/analytics/interaction-series",
        params=params,
    )
    assert second.status_code == 200, second.text
    assert second.json() == first.json()


def test_interaction_series_cursor_pages_reuse_one_revision_snapshot(
    stage4_catalog: Stage4Catalog,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with Session(stage4_catalog.engine) as session, session.begin():
        messages = list(session.scalars(select(Message).order_by(Message.pk).limit(2)))
        assert len(messages) == 2
        for index, message in enumerate(messages):
            message.timestamp_ms = 1_672_531_200_000 + index * 86_400_000
            message.content_display = f"https://snapshot-{index}.invalid"
            message.extra_json = {}
            message.message_kind = "call"
            message.call_duration_seconds = index + 1

    params = {"bucket": "day", "timezone": "UTC", "limit": 1}
    first = stage4_catalog.client.get("/api/v1/analytics/interaction-series", params=params)
    assert first.status_code == 200, first.text
    cursor = first.json()["page"]["next_cursor"]
    assert cursor

    def unexpected_link_rescan(_content: str | None, _extra: object) -> list[str]:
        raise AssertionError("Cursor pages should reuse the in-memory revision snapshot")

    monkeypatch.setattr(analytics_routes, "_message_link_hostnames", unexpected_link_rescan)
    second = stage4_catalog.client.get(
        "/api/v1/analytics/interaction-series", params={**params, "cursor": cursor}
    )
    assert second.status_code == 200, second.text
    assert second.json()["series"][0]["period"] != first.json()["series"][0]["period"]


def test_series_snapshot_refuses_revision_change_during_scan(
    stage4_catalog: Stage4Catalog,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revisions = iter(("revision-before", "revision-before", "revision-after"))
    monkeypatch.setattr(
        analytics_routes,
        "analytics_source_revision",
        lambda _layout, _engine: next(revisions),
    )
    response = stage4_catalog.client.get(
        "/api/v1/analytics/messages",
        params={"bucket": "day", "timezone": "UTC", "limit": 1},
    )
    assert response.status_code == 409
    assert "changed while analytics were computed" in response.json()["detail"]
    snapshots = getattr(stage4_catalog.client.app.state, "analytics_series_snapshots", {})
    assert snapshots == {}
