from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from meta_archive.api.main import create_app
from meta_archive.api.security import CSRF_COOKIE
from meta_archive.database import create_catalog_engine, run_migrations
from meta_archive.models import IntegrityAnomaly, Job
from meta_archive.paths import ArchiveLayout
from meta_archive.services.inventory import scan_archive


def _catalogued_app(tmp_path: Path, *, security_disabled: bool = False):  # type: ignore[no-untyped-def]
    root = tmp_path / "meta-backup-api"
    source = root / "package-a" / "your_facebook_activity" / "messages"
    source.mkdir(parents=True)
    (source / "settings.json").write_text("{}", encoding="utf-8")
    layout = ArchiveLayout(root)
    layout.ensure_runtime_directories()
    run_migrations(layout.database)
    engine = create_catalog_engine(layout.database)
    scan_archive(layout, engine, hash_files=False)
    engine.dispose()
    return create_app(root, access_token="a" * 32, security_disabled=security_disabled)


def test_health_is_public_but_catalog_routes_require_session(tmp_path: Path) -> None:
    app = _catalogued_app(tmp_path)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        health = client.get("/api/v1/health")
        protected = client.get("/api/v1/overview")

    assert health.status_code == 200
    assert health.json()["catalog_ready"] is False
    assert health.json()["catalog_state"] == "partial"
    assert health.headers["x-content-type-options"] == "nosniff"
    assert health.headers["cross-origin-resource-policy"] == "same-origin"
    assert protected.status_code == 401


def test_access_token_becomes_http_only_session_and_csrf_protects_mutations(
    tmp_path: Path,
) -> None:
    app = _catalogued_app(tmp_path)
    with TestClient(app, base_url="http://127.0.0.1", follow_redirects=False) as client:
        invalid = client.get("/auth", params={"token": "b" * 32})
        authenticated = client.get("/auth", params={"token": "a" * 32})
        overview = client.get("/api/v1/overview")
        rejected_patch = client.patch("/api/v1/settings", json={"timezone": "UTC"})
        accepted_patch = client.patch(
            "/api/v1/settings",
            json={"timezone": "UTC"},
            headers={
                "origin": "http://127.0.0.1",
                "x-csrf-token": client.cookies[CSRF_COOKIE],
            },
        )

    assert invalid.status_code == 401
    assert authenticated.status_code == 303
    assert "httponly" in authenticated.headers["set-cookie"].casefold()
    assert overview.status_code == 200
    assert overview.json()["archive"]["source_file_count"] == 1
    assert rejected_patch.status_code == 403
    assert accepted_patch.status_code == 200
    assert accepted_patch.json()["timezone"] == "UTC"


def test_access_token_url_is_single_use_and_does_not_replace_an_existing_session(
    tmp_path: Path,
) -> None:
    app = _catalogued_app(tmp_path)
    with (
        TestClient(app, base_url="http://127.0.0.1", follow_redirects=False) as first_client,
        TestClient(app, base_url="http://127.0.0.1", follow_redirects=False) as replay_client,
    ):
        exchanged = first_client.get("/auth", params={"token": "a" * 32})
        replayed = replay_client.get("/auth", params={"token": "a" * 32})
        original_session = first_client.get("/api/v1/overview")
        replay_session = replay_client.get("/api/v1/overview")

    assert exchanged.status_code == 303
    assert replayed.status_code == 401
    assert original_session.status_code == 200
    assert replay_session.status_code == 401


def test_strict_host_origin_and_cookie_boundaries(tmp_path: Path) -> None:
    app = _catalogued_app(tmp_path)
    with TestClient(app, base_url="http://127.0.0.1:8765", follow_redirects=False) as client:
        rejected_host = client.get("/api/v1/health", headers={"host": "archive.example"})
        rejected_test_host = client.get("/api/v1/health", headers={"host": "testserver"})
        authenticated = client.get("/auth", params={"token": "a" * 32})
        csrf = client.cookies[CSRF_COOKIE]
        rejected_missing_origin = client.patch(
            "/api/v1/settings",
            json={"timezone": "UTC"},
            headers={
                "referer": "http://127.0.0.1:8765/",
                "sec-fetch-site": "same-origin",
                "x-csrf-token": csrf,
            },
        )
        rejected_null_origin = client.patch(
            "/api/v1/settings",
            json={"timezone": "UTC"},
            headers={"origin": "null", "x-csrf-token": csrf},
        )
        rejected_invalid_origin = client.patch(
            "/api/v1/settings",
            json={"timezone": "UTC"},
            headers={"origin": "http://[invalid", "x-csrf-token": csrf},
        )
        rejected_origin = client.patch(
            "/api/v1/settings",
            json={"timezone": "UTC"},
            headers={"origin": "http://archive.example", "x-csrf-token": csrf},
        )
        rejected_scheme = client.patch(
            "/api/v1/settings",
            json={"timezone": "UTC"},
            headers={"origin": "https://127.0.0.1:8765", "x-csrf-token": csrf},
        )
        accepted_origin = client.patch(
            "/api/v1/settings",
            json={"timezone": "UTC"},
            headers={"origin": "http://127.0.0.1:8765", "x-csrf-token": csrf},
        )

    cookies = authenticated.headers.get_list("set-cookie")
    session_cookie = next(value for value in cookies if value.startswith("meta_archive_session="))
    csrf_cookie = next(value for value in cookies if value.startswith("meta_archive_csrf="))
    assert rejected_host.status_code == 400
    assert rejected_test_host.status_code == 400
    assert rejected_missing_origin.status_code == 403
    assert rejected_null_origin.status_code == 403
    assert rejected_invalid_origin.status_code == 403
    assert rejected_origin.status_code == 403
    assert rejected_scheme.status_code == 403
    assert accepted_origin.status_code == 200
    assert "httponly" in session_cookie.casefold()
    assert "samesite=strict" in session_cookie.casefold()
    assert "path=/" in session_cookie.casefold()
    assert "httponly" not in csrf_cookie.casefold()
    assert "samesite=strict" in csrf_cookie.casefold()
    assert "path=/" in csrf_cookie.casefold()


def test_settings_keep_image_stage_locked_and_require_key_for_cloud(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = _catalogued_app(tmp_path, security_disabled=True)
    with TestClient(app) as client:
        settings = client.get("/api/v1/settings")
        cloud = client.patch("/api/v1/settings", json={"cloud_enabled": True})
        image_gate = client.patch("/api/v1/settings", json={"image_semantic_stage_enabled": True})

    assert settings.json()["image_semantic_stage_enabled"] is False
    assert cloud.status_code == 409
    assert image_gate.status_code == 422


def test_integrity_can_include_resolved_findings(tmp_path: Path) -> None:
    app = _catalogued_app(tmp_path, security_disabled=True)
    with Session(app.state.engine) as session, session.begin():
        session.add_all(
            [
                IntegrityAnomaly(
                    id="open-finding",
                    code="open_fixture",
                    severity="warning",
                    summary="Open fixture finding",
                    detail_json={},
                    resolved=False,
                ),
                IntegrityAnomaly(
                    id="resolved-finding",
                    code="resolved_fixture",
                    severity="info",
                    summary="Resolved fixture finding",
                    detail_json={},
                    resolved=True,
                ),
            ]
        )

    with TestClient(app) as client:
        unresolved = client.get("/api/v1/integrity")
        all_findings = client.get("/api/v1/integrity", params={"unresolved_only": "false"})

    assert [item["id"] for item in unresolved.json()["items"]] == ["open-finding"]
    assert {item["id"] for item in all_findings.json()["items"]} == {
        "open-finding",
        "resolved-finding",
    }


def test_api_datetimes_are_explicit_utc_instants(tmp_path: Path) -> None:
    app = _catalogued_app(tmp_path, security_disabled=True)

    with TestClient(app) as client:
        payload = client.get("/api/v1/jobs").json()

    assert payload["items"]
    assert payload["items"][0]["started_at"].endswith("+00:00")


def test_integrity_and_jobs_filters_report_server_side_totals(tmp_path: Path) -> None:
    app = _catalogued_app(tmp_path, security_disabled=True)
    with Session(app.state.engine) as session, session.begin():
        session.add_all(
            [
                IntegrityAnomaly(
                    id="searchable-integrity",
                    code="fixture_warning",
                    severity="warning",
                    summary="Needle finding",
                    detail_json={"path": "needle-source"},
                    resolved=False,
                ),
                IntegrityAnomaly(
                    id="other-integrity",
                    code="fixture_info",
                    severity="info",
                    summary="Other finding",
                    detail_json={},
                    resolved=False,
                ),
                Job(
                    id="completed-with-errors-job",
                    kind="activity_import",
                    status="completed_with_errors",
                    stage="Needle stage",
                    counters_json={},
                    checkpoint_json={},
                ),
            ]
        )

    with TestClient(app) as client:
        integrity = client.get(
            "/api/v1/integrity", params={"severity": "warning", "q": "needle", "limit": 1}
        ).json()
        jobs = client.get(
            "/api/v1/jobs", params={"status": "complete", "q": "needle", "limit": 1}
        ).json()

    assert [item["id"] for item in integrity["items"]] == ["searchable-integrity"]
    assert integrity["page"]["filtered_total"] == 1
    assert [item["id"] for item in jobs["items"]] == ["completed-with-errors-job"]
    assert jobs["page"]["filtered_total"] == 1
    assert jobs["summary"]["completed"] >= 2
