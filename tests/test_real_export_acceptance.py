from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sqlite3
import stat
import subprocess
import sys
import time
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import Engine, select, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from meta_archive.api.main import create_app
from meta_archive.database import create_catalog_engine, run_migrations
from meta_archive.models import (
    AttachmentOccurrence,
    AudioTranscription,
    AudioTranscriptSegment,
    Blob,
    SearchWindow,
)
from meta_archive.paths import ArchiveLayout
from meta_archive.services.activity_importer import import_activities
from meta_archive.services.inventory import scan_archive
from meta_archive.services.message_importer import import_messages
from meta_archive.services.nlp_annotations import (
    DIRECT_PROVENANCE_BINDING,
    MAX_TEXT_CHARACTERS,
    NLP_IMPLEMENTATION_SCHEMA,
    NLP_INFERENCE_CONFIGURATION_SCHEMA,
    NLP_TOKENIZER_MAX_LENGTH,
    PIPELINE_VERSION,
    ClassificationResult,
    NlpInferenceConfiguration,
    NlpModelSpec,
    build_nlp_annotations,
)
from meta_archive.services.search_index import _vector_row, rebuild_search_windows

MESSAGE_FIXTURES = Path(__file__).parent / "fixtures" / "messages"
ACTIVITY_FIXTURES = Path(__file__).parent / "fixtures" / "activity"


def _checkpoint_catalog(engine: Engine) -> None:
    """Materialize committed WAL rows before exercising the immutable acceptance reader."""

    with engine.connect() as connection:
        busy, _log_frames, _checkpointed_frames = connection.exec_driver_sql(
            "PRAGMA wal_checkpoint(TRUNCATE)"
        ).one()
    assert busy == 0


def _fixture_acceptance_identity(*, nonce_character: str) -> dict[str, str]:
    assert len(nonce_character) == 1 and nonce_character in "0123456789abcdef"
    return {
        "catalog_source_revision": "1" * 64,
        "backend_version": "0.1.0",
        "backend_source_sha256": "2" * 64,
        "frontend_manifest_sha256": "3" * 64,
        "server_nonce": nonce_character * 64,
    }


def _acceptance_module() -> ModuleType:
    script = Path(__file__).resolve().parents[1] / "scripts" / "real_export_acceptance.py"
    spec = importlib.util.spec_from_file_location("real_export_acceptance", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def acceptance_catalog(tmp_path: Path) -> Iterator[tuple[ArchiveLayout, Engine]]:
    layout = ArchiveLayout(tmp_path / "meta-backup-acceptance")
    shutil.copytree(MESSAGE_FIXTURES, layout.root)
    shutil.copytree(ACTIVITY_FIXTURES, layout.root / "package-c")
    layout.ensure_runtime_directories()
    run_migrations(layout.database)
    engine = create_catalog_engine(layout.database)
    scan_archive(layout, engine, hash_files=True)
    import_messages(layout, engine)
    import_activities(layout, engine)
    _checkpoint_catalog(engine)
    yield layout, engine
    engine.dispose()


class _TestApiClient:
    def __init__(self, client: TestClient) -> None:
        self.client = client

    def get_json(
        self,
        path: str,
        parameters: Mapping[str, object] | None = None,
        *,
        timeout: float = 120.0,
    ) -> tuple[dict[str, Any], float]:
        del timeout
        started = time.perf_counter()
        response = self.client.get(path, params=parameters)
        elapsed = (time.perf_counter() - started) * 1_000
        assert response.status_code == 200, response.text
        payload = response.json()
        assert isinstance(payload, dict)
        return payload, elapsed


class _CursorAuditClient:
    def __init__(
        self,
        identities: Sequence[str],
        *,
        defect: str | None = None,
        count_is_unknown: bool = False,
        coverage_status: str | None = None,
    ) -> None:
        self.identities = list(identities)
        self.defect = defect
        self.count_is_unknown = count_is_unknown
        self.coverage_status = coverage_status
        self.calls: list[dict[str, object]] = []

    def get_json(
        self,
        path: str,
        parameters: Mapping[str, object] | None = None,
        *,
        timeout: float = 120.0,
    ) -> tuple[dict[str, Any], float]:
        del path, timeout
        query = dict(parameters or {})
        self.calls.append(query)
        page_size = int(query.get("limit", 50))
        raw_cursor = query.get("cursor")
        start = 0 if raw_cursor is None else int(str(raw_cursor).rsplit("-", 1)[1])
        end = min(start + page_size, len(self.identities))
        page_identities = self.identities[start:end]
        page_number = (start // page_size) + 1
        if page_number == 2 and self.defect == "gap":
            page_identities = self.identities[start + 1 : min(end + 1, len(self.identities))]
        elif page_number == 2 and self.defect == "duplicate" and page_identities:
            page_identities = [self.identities[start - 1], *page_identities[1:]]
        elif page_number == 2 and self.defect == "tie_swap" and len(page_identities) >= 2:
            page_identities = [page_identities[1], page_identities[0], *page_identities[2:]]
        elif page_number == 2 and self.defect == "middle_corruption" and page_identities:
            middle = len(page_identities) // 2
            page_identities[middle] = "private-corrupted-middle-row"

        next_cursor = f"private-cursor-{end}" if end < len(self.identities) else None
        if page_number == 2 and self.defect == "cursor_cycle":
            next_cursor = str(raw_cursor)
        elif page_number == 2 and self.defect == "missing_cursor":
            next_cursor = None
        total = len(self.identities) + (
            1 if page_number == 2 and self.defect == "total_mismatch" else 0
        )
        payload: dict[str, Any] = {
            "items": [{"id": identity} for identity in page_identities],
            "page": {
                "total": None if self.count_is_unknown else total,
                "limit": page_size,
                "next_cursor": next_cursor,
                "count_is_unknown": self.count_is_unknown,
            },
        }
        if self.coverage_status is not None:
            payload["coverage"] = {"status": self.coverage_status}
        return payload, 1.0


class _FixtureClassifier:
    def inference_configuration(self, spec: NlpModelSpec) -> NlpInferenceConfiguration:
        implementation = {
            "schema": NLP_IMPLEMENTATION_SCHEMA,
            "pipeline_version": PIPELINE_VERSION,
            "provider": "transformers-auto-sequence-classification",
            "packages": {
                "transformers": "fixture",
                "torch": "fixture",
                "tokenizers": "fixture",
                "safetensors": "fixture",
            },
            "tokenizer_class": "fixture.AcceptanceTokenizer",
            "model_class": "fixture.AcceptanceModel",
            "device": spec.device,
            "precision": "float32",
            "quantization": spec.quantization,
            "evaluation_mode": True,
            "inference_mode": "torch.inference_mode",
            "activation": spec.activation,
            "input_character_guard": MAX_TEXT_CHARACTERS,
            "tokenizer_padding": True,
            "tokenizer_truncation": True,
            "tokenizer_max_length": NLP_TOKENIZER_MAX_LENGTH,
            "label_mapping": "model.config.id2label casefold with label_<index> fallback",
        }
        implementation_sha256 = hashlib.sha256(
            json.dumps(
                implementation,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        payload = {
            "schema": NLP_INFERENCE_CONFIGURATION_SCHEMA,
            "task": spec.task,
            "model_name": spec.model_name,
            "model_revision": spec.model_revision,
            "model_artifact": {
                "schema": "hf-logical-snapshot-v1",
                "sha256": hashlib.sha256(f"fixture-artifact:{spec.task}".encode()).hexdigest(),
                "file_count": 1,
                "total_bytes": 1,
            },
            "implementation": implementation,
            "implementation_sha256": implementation_sha256,
        }
        configuration_sha256 = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return NlpInferenceConfiguration(
            sha256=configuration_sha256,
            payload=payload,
            direct_binding_kind=DIRECT_PROVENANCE_BINDING,
        )

    def classify(self, spec: NlpModelSpec, texts: Sequence[str]) -> list[ClassificationResult]:
        probabilities = (
            {"negative": 0.1, "neutral": 0.2, "positive": 0.7}
            if spec.task == "sentiment"
            else {"joy": 0.6, "sadness": 0.4}
            if spec.task == "emotion"
            else {"toxic": 0.25, "insult": 0.1}
        )
        return [ClassificationResult(probabilities=probabilities) for _text in texts]


class _IdentityApiClient:
    def __init__(self, payload: Mapping[str, object] | None) -> None:
        self.payload = payload

    def get_json(
        self,
        path: str,
        parameters: Mapping[str, object] | None = None,
        *,
        timeout: float = 120.0,
    ) -> tuple[dict[str, Any], float]:
        del parameters, timeout
        assert path == "/api/v1/acceptance/identity"
        if self.payload is None:
            raise RuntimeError("private route and query must stay hidden")
        return dict(self.payload), 1.0


def test_readonly_catalog_engine_rejects_writes(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> None:
    acceptance = _acceptance_module()
    with catalog_engine.begin() as connection:
        connection.execute(text("INSERT INTO archive_settings(key, value_json) VALUES ('x', '{}')"))

    readonly = acceptance._readonly_catalog_engine(archive_layout.database)
    try:
        with readonly.connect() as connection:
            assert connection.exec_driver_sql("PRAGMA query_only").scalar_one() == 1
            with pytest.raises(OperationalError):
                connection.exec_driver_sql("DELETE FROM archive_settings")
    finally:
        readonly.dispose()

    with catalog_engine.connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM archive_settings")).scalar_one() == 1


def test_readonly_catalog_engine_never_creates_wal_sidecars(tmp_path: Path) -> None:
    acceptance = _acceptance_module()
    database = tmp_path / "wal-catalog.sqlite"
    connection = sqlite3.connect(database)
    try:
        assert connection.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)
        connection.execute("CREATE TABLE fixture(value INTEGER NOT NULL)")
        connection.execute("INSERT INTO fixture(value) VALUES (1)")
        connection.commit()
    finally:
        connection.close()
    sidecars = tuple(Path(f"{database}{suffix}") for suffix in ("-wal", "-shm", "-journal"))
    assert not any(path.exists() for path in sidecars)

    readonly = acceptance._readonly_catalog_engine(database)
    try:
        with readonly.connect() as connection:
            assert connection.exec_driver_sql("SELECT value FROM fixture").scalar_one() == 1
            assert connection.exec_driver_sql("PRAGMA query_only").scalar_one() == 1
            assert not any(path.exists() for path in sidecars)
    finally:
        readonly.dispose()

    assert not any(path.exists() for path in sidecars)


def test_acceptance_server_identity_requires_exact_private_safe_schema() -> None:
    acceptance = _acceptance_module()
    identity = _fixture_acceptance_identity(nonce_character="a")

    checks, report, transient = acceptance._acceptance_server_identity(_IdentityApiClient(identity))

    assert checks[0].status == "pass"
    assert transient == identity
    assert report == {
        "identity_response_observed": True,
        "exact_schema": True,
        "digest_fields_valid": True,
        "backend_version_valid": True,
        "identity_values_emitted": False,
    }
    assert not any(value in json.dumps(report) for value in identity.values())

    malformed = {**identity, "unexpected": "private-value"}
    malformed_checks, malformed_report, malformed_transient = (
        acceptance._acceptance_server_identity(_IdentityApiClient(malformed))
    )
    assert malformed_checks[0].status == "fail"
    assert malformed_report["exact_schema"] is False
    assert malformed_transient == {}
    failed_checks, failed_report, failed_transient = acceptance._acceptance_server_identity(
        _IdentityApiClient(None)
    )
    assert failed_checks[0].status == "fail"
    assert failed_report["identity_response_observed"] is False
    assert failed_transient == {}


def test_private_report_writer_refuses_overwrite_and_uses_owner_only_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    acceptance = _acceptance_module()
    fake_script = tmp_path / "scripts" / "real_export_acceptance.py"
    fake_script.parent.mkdir()
    fake_script.write_text("", encoding="utf-8")
    monkeypatch.setattr(acceptance, "__file__", str(fake_script))
    output = tmp_path / ".private-review" / "fixture" / "report.json"

    result = acceptance._write_private_json(output, {"safe": True})

    assert json.loads(result.read_text(encoding="utf-8")) == {"safe": True}
    assert stat.S_IMODE(result.lstat().st_mode) == 0o600
    assert stat.S_IMODE(result.parent.lstat().st_mode) == 0o700
    assert stat.S_IMODE(result.parent.parent.lstat().st_mode) == 0o700
    with pytest.raises(FileExistsError):
        acceptance._write_private_json(output, {"safe": False})

    raced_output = output.with_name("raced.json")
    real_link = acceptance.os.link

    def create_destination_before_publish(
        source: Path, destination: Path, *, follow_symlinks: bool
    ) -> None:
        destination.write_text("prior evidence", encoding="utf-8")
        destination.chmod(0o600)
        real_link(source, destination, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(acceptance.os, "link", create_destination_before_publish)
    with pytest.raises(FileExistsError):
        acceptance._write_private_json(raced_output, {"safe": False})
    assert raced_output.read_text(encoding="utf-8") == "prior evidence"


def test_startup_token_file_must_be_owner_only(tmp_path: Path) -> None:
    acceptance = _acceptance_module()
    token = tmp_path / "token"
    token.write_text("a" * 32, encoding="utf-8")
    token.chmod(0o644)

    with pytest.raises(RuntimeError, match="group- or other-accessible"):
        acceptance._read_startup_token(token)

    token.chmod(0o600)
    assert acceptance._read_startup_token(token) == "a" * 32

    link = tmp_path / "token-link"
    link.symlink_to(token)
    with pytest.raises(RuntimeError, match="regular, non-symlink"):
        acceptance._read_startup_token(link)


def test_fixture_acceptance_builds_sanitized_blocking_report(
    acceptance_catalog: tuple[ArchiveLayout, Engine], monkeypatch: pytest.MonkeyPatch
) -> None:
    acceptance = _acceptance_module()
    archive_layout, catalog_engine = acceptance_catalog
    with catalog_engine.connect() as connection:
        private_row = connection.execute(
            text(
                """
                SELECT message.id, message.source_record_id, attachment.id,
                       source.normalized_path, attachment.filename
                FROM messages AS message
                JOIN source_files AS source ON source.id = message.source_file_id
                JOIN attachment_occurrences AS attachment ON attachment.message_id = message.id
                WHERE attachment.filename = 'cross-chunk.jpg'
                ORDER BY attachment.id
                LIMIT 1
                """
            )
        ).one()
    private_values = {
        "person_name": "Alex Example",
        "conversation_title": "Shared Group Title",
        "filename": str(private_row.filename),
        "source_path": str(private_row.normalized_path),
        "message_text": "Exact duplicate fixture",
        "comment_text": "Private fixture comment payload",
        "message_jump_id": str(private_row[0]),
        "source_record_id": str(private_row.source_record_id),
        "attachment_jump_id": str(private_row[2]),
    }
    real_verify = acceptance.verify_catalog
    observed_deep_arguments: list[object] = []

    def verification_with_private_probe(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        observed_deep_arguments.append(kwargs.get("deep"))
        checks = real_verify(*args, **kwargs)
        checks.append(
            {
                "key": "privacy.redaction_probe",
                "status": "fail",
                "expected": private_values,
                "actual": private_values,
                "mandatory": True,
            }
        )
        return checks

    monkeypatch.setattr(acceptance, "verify_catalog", verification_with_private_probe)
    stale_report = archive_layout.manifests / "integrity-reports" / "latest.json"
    stale_report.parent.mkdir(parents=True, exist_ok=True)
    stale_report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at": "2000-01-01T00:00:00+00:00",
                "checks": [],
                "gate": {
                    "all_mandatory_checks_pass": True,
                    "deep_verified": True,
                    "library_verified": True,
                    "required_artifacts_present": True,
                    "source_retirement_eligible": True,
                },
            }
        ),
        encoding="utf-8",
    )
    stale_report.chmod(0o600)
    catalog_engine.dispose()

    report = acceptance.build_acceptance_report(
        archive_layout,
        client=None,
        repeats=3,
        sample_count=1,
        query="argument",
        browser_report=None,
    )

    assert report["safety"] == {
        "sqlite_query_only": True,
        "direct_catalog_connection_query_only": True,
        "acceptance_read_only_server_verified": False,
        "workspace_read_only": True,
        "authenticated_gets_may_write_revision_bound_cache": False,
        "analytics_cache_reads_writes_disabled_for_api": True,
        "destructive_mutations_performed": False,
        "source_text_emitted": False,
        "names_titles_filenames_emitted": False,
        "api_response_bodies_emitted": False,
        "cloud_requests_performed": False,
    }
    assert report["gate"]["passed"] is False
    assert report["schema_version"] == 2
    assert report["reconciliation"]["mode"] == "exhaustive"
    assert observed_deep_arguments == [True]
    assert report["verification"]["current_deep_gate"]["executed_in_current_acceptance"] is True
    current_gate = next(
        check for check in report["checks"] if check["key"] == "catalog.current_deep_integrity_gate"
    )
    assert current_gate["status"] == "fail"
    baseline_manifest = next(
        check for check in report["verification"]["checks"] if check["key"] == "baseline.manifest"
    )
    assert baseline_manifest["status"] == "fail"
    baseline_gate = next(
        check for check in report["checks"] if check["key"] == "catalog.audited_export_baselines"
    )
    assert baseline_gate["status"] == "fail"
    assert "baseline.manifest" in baseline_gate["actual"]["non_pass_keys"]
    assert "catalog.audited_export_baselines" in report["gate"]["blocking_keys"]
    serialized = json.dumps(report)
    assert str(archive_layout.root) not in serialized
    assert "argument" not in serialized
    for private_value in private_values.values():
        assert private_value not in serialized


def test_fixture_api_orders_totals_and_jump_samples_match_direct_sql(
    acceptance_catalog: tuple[ArchiveLayout, Engine],
) -> None:
    acceptance = _acceptance_module()
    layout, _engine = acceptance_catalog
    app = create_app(layout.root, security_disabled=True)
    readonly = acceptance._readonly_catalog_engine(layout.database)
    try:
        with TestClient(app) as web_client, readonly.connect() as connection:
            client = _TestApiClient(web_client)
            totals = acceptance._catalog_totals(connection)
            reconciliation, timings, reconciliation_report = acceptance._api_reconciliation(
                connection, client, totals
            )
            bounded_checks, bounded_timings, bounded_report = acceptance._api_reconciliation(
                connection,
                client,
                totals,
                policy=acceptance.ReconciliationPolicy(
                    mode="risk-bounded",
                    seed=acceptance.DEFAULT_RECONCILIATION_SEED,
                ),
            )
            jumps, jump_timings, jump_summary = acceptance._jump_acceptance(
                connection,
                client,
                sample_count=3,
                expected_largest_message_count=None,
            )
            wrong_largest, _wrong_timings, _wrong_summary = acceptance._jump_acceptance(
                connection,
                client,
                sample_count=1,
                expected_largest_message_count=0,
            )
    finally:
        readonly.dispose()

    assert all(check.status == "pass" for check in reconciliation)
    assert timings["list_api"]
    assert reconciliation_report["mode"] == "exhaustive"
    assert all(check.status == "pass" for check in bounded_checks)
    assert bounded_timings["list_api"]
    assert bounded_report["mode"] == "risk-bounded"
    assert bounded_report["observed_stream_count"] == 124
    assert bounded_report["strata"]["complete"] is True
    assert all(not check.required or check.status == "pass" for check in jumps)
    assert jump_timings
    assert jump_summary["largest_conversation"]["returned_rows"] <= 250
    largest = next(
        check for check in wrong_largest if check.key == "api.largest_conversation_bounded_page"
    )
    assert largest.status == "fail"
    assert largest.expected == {"largest_message_count": 0, "maximum_page_rows": 250}


def test_cursor_sort_audit_exhausts_small_multi_page_orders_without_leaking_ids() -> None:
    acceptance = _acceptance_module()
    identities = [f"private-row-{index:03d}" for index in range(300)]

    summary, timings = acceptance._bounded_cursor_order_reconciliation(
        _CursorAuditClient(identities),
        "/api/v1/fixture",
        {"sort": "date", "direction": "asc"},
        identity_key="id",
        expected=identities,
        expected_total=len(identities),
    )

    assert summary["passed"] is True
    assert summary["coverage"] == "full_multi_page"
    assert summary["pages_checked"] == 2
    assert summary["rows_checked"] == 300
    assert summary["continuation_observed"] is True
    assert len(timings) == 2
    serialized = json.dumps(summary)
    assert "private-row" not in serialized
    assert "private-cursor" not in serialized


def test_cursor_sort_audit_exhausts_complete_large_orders() -> None:
    acceptance = _acceptance_module()
    identities = [f"private-large-row-{index:03d}" for index in range(600)]

    summary, timings = acceptance._bounded_cursor_order_reconciliation(
        _CursorAuditClient(identities),
        "/api/v1/fixture",
        {"sort": "date", "direction": "desc"},
        identity_key="id",
        expected=iter(identities),
        expected_total=len(identities),
    )

    assert summary["passed"] is True
    assert summary["coverage"] == "full_multi_page"
    assert summary["full_stream"] is True
    assert summary["pages_checked"] == 3
    assert summary["rows_checked"] == 600
    assert summary["direct_sql_rows_checked"] == 600
    assert len(timings) == 3


def test_cursor_sort_risk_profile_stops_after_two_pages_and_closes_sql_stream() -> None:
    acceptance = _acceptance_module()
    identities = [f"private-risk-row-{index:03d}" for index in range(600)]
    client = _CursorAuditClient(identities)
    closed = False

    def expected_stream() -> Iterator[str]:
        nonlocal closed
        try:
            yield from identities
        finally:
            closed = True

    summary, timings = acceptance._bounded_cursor_order_reconciliation(
        client,
        "/api/v1/fixture",
        {"sort": "date", "direction": "asc"},
        identity_key="id",
        expected=expected_stream(),
        expected_total=len(identities),
        page_cap=2,
    )

    assert summary["passed"] is True
    assert summary["coverage"] == "risk_bounded_edges"
    assert summary["risk_bounded"] is True
    assert summary["full_stream"] is False
    assert summary["requested_pages"] == summary["pages_checked"] == 2
    assert summary["requested_rows"] == summary["rows_checked"] == 500
    assert summary["cursor_boundaries_checked"] == 1
    assert summary["outbound_cursor_observed"] is True
    assert len(timings) == len(client.calls) == 2
    assert closed is True
    serialized = json.dumps(summary)
    assert "private-risk-row" not in serialized
    assert "private-cursor" not in serialized


def test_risk_profile_seed_and_cli_selection_are_explicit_and_reproducible(tmp_path: Path) -> None:
    acceptance = _acceptance_module()
    first = acceptance._risk_seed(20260722, "message:early")
    assert first == acceptance._risk_seed(20260722, "message:early")
    assert first != acceptance._risk_seed(20260723, "message:early")
    assert first != acceptance._risk_seed(20260722, "message:late")

    arguments = acceptance._arguments(
        [
            "--archive-root",
            str(tmp_path),
            "--reconciliation-mode",
            "risk-bounded",
            "--reconciliation-seed",
            "42",
        ]
    )
    assert arguments.reconciliation_mode == "risk-bounded"
    assert arguments.reconciliation_seed == 42
    with pytest.raises(SystemExit):
        acceptance._arguments(["--archive-root", str(tmp_path)])
    with pytest.raises(SystemExit):
        acceptance._arguments(
            [
                "--archive-root",
                str(tmp_path),
                "--reconciliation-mode",
                "risk-bounded",
                "--reconciliation-seed",
                "-1",
            ]
        )


def test_production_spot_check_cli_profile_is_explicit(tmp_path: Path) -> None:
    acceptance = _acceptance_module()

    arguments = acceptance._arguments(
        [
            "--archive-root",
            str(tmp_path),
            "--reconciliation-mode",
            "production-spot-check",
            "--reconciliation-seed",
            "987654321",
        ]
    )

    assert arguments.reconciliation_mode == "production-spot-check"
    assert arguments.reconciliation_seed == 987654321
    policy = acceptance.ReconciliationPolicy(
        mode=arguments.reconciliation_mode,
        seed=arguments.reconciliation_seed,
    )
    assert policy.profile_version == "production-spot-check-v2"
    assert policy.sort_page_cap == 2
    assert "production-spot-check" in acceptance.RECONCILIATION_MODES


def test_production_spot_check_api_budget_fails_before_overrun() -> None:
    acceptance = _acceptance_module()

    class Client:
        def __init__(self) -> None:
            self.calls = 0

        def get_json(
            self,
            path: str,
            parameters: Mapping[str, object] | None = None,
            *,
            timeout: float = 120.0,
        ) -> tuple[dict[str, Any], float]:
            del path, parameters, timeout
            self.calls += 1
            return {}, 0.0

    client = Client()
    budget = acceptance._SpotApiBudget(client, 2)

    budget.get_json("/first")
    budget.get_json("/second")
    with pytest.raises(RuntimeError, match="budget exhausted"):
        budget.get_json("/third")

    assert client.calls == budget.calls == 2


def test_local_api_client_rejects_oversized_unknown_length_body_before_json_decode() -> None:
    acceptance = _acceptance_module()

    class Response:
        def __init__(self) -> None:
            self.headers: dict[str, str] = {}
            self.requested_bytes: int | None = None

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, amount: int) -> bytes:
            self.requested_bytes = amount
            return b"{" + b"x" * (amount - 1)

    class Opener:
        def __init__(self, response: Response) -> None:
            self.response = response

        def open(self, _url: str, *, timeout: float) -> Response:
            assert timeout == 120.0
            return self.response

    response = Response()
    client = object.__new__(acceptance.LocalApiClient)
    client.base_url = "http://127.0.0.1:8000"
    client._opener = Opener(response)

    with pytest.raises(RuntimeError, match="byte cap"):
        client.get_json("/api/v1/fixture", maximum_bytes=16)

    assert response.requested_bytes == 17


def test_production_spot_check_media_streams_use_established_view_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acceptance = _acceptance_module()
    observed: list[tuple[str, dict[str, object]]] = []

    class Client:
        def get_json(
            self,
            path: str,
            parameters: Mapping[str, object] | None = None,
            *,
            timeout: float = 120.0,
        ) -> tuple[dict[str, Any], float]:
            del timeout
            observed.append((path, dict(parameters or {})))
            if path == "/api/v1/overview":
                return {
                    "content": {
                        "conversation_count": 0,
                        "message_count": 0,
                        "attachment_count": 0,
                        "activity_count": 0,
                    }
                }, 0.0
            limit = int((parameters or {}).get("limit", 1))
            return {
                "items": [],
                "page": {
                    "total": 0,
                    "limit": limit,
                    "next_cursor": None,
                    "count_is_unknown": False,
                },
            }, 0.0

    def forbidden(*_args: object, **_kwargs: object) -> Any:
        raise AssertionError("unbounded full-stream reconciliation was called")

    for name in (
        "_bounded_cursor_order_reconciliation",
        "_conversation_order",
        "_media_occurrence_order",
        "_activity_order",
    ):
        monkeypatch.setattr(acceptance, name, forbidden)
    totals = {
        "conversations": 0,
        "messages": 0,
        "attachments": 0,
        "activities": 0,
        "events": 0,
        "contacts": 0,
        "timeline": 0,
    }

    checks, report, _timings = acceptance._spot_api_reconciliation(
        object(), acceptance._SpotApiBudget(Client(), 48), totals
    )

    assert all(check.status == "pass" for check in checks)
    assert report["stream_count"] == 8
    media_parameters = [parameters for path, parameters in observed if path == "/api/v1/media"]
    assert len(media_parameters) == 2
    assert all(parameters["view"] == "occurrences" for parameters in media_parameters)
    assert all("mode" not in parameters for parameters in media_parameters)
    assert report["independent_full_stream_sort_reconciliation_performed"] is False


def test_production_spot_check_targeted_prefix_rejects_corrupt_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acceptance = _acceptance_module()
    targeted: list[str] = []

    class Client:
        def get_json(
            self,
            _path: str,
            parameters: Mapping[str, object] | None = None,
            *,
            timeout: float = 120.0,
        ) -> tuple[dict[str, Any], float]:
            del timeout
            return {
                "items": [{"id": "later"}, {"id": "earlier"}],
                "page": {
                    "total": 2,
                    "limit": (parameters or {})["limit"],
                    "next_cursor": None,
                },
            }, 0.0

    def targeted_values(
        _connection: object,
        *,
        entity: str,
        sort_name: str,
        identities: Sequence[str],
    ) -> dict[str, object]:
        assert (entity, sort_name) == ("conversation", "messages")
        targeted.extend(identities)
        return {"earlier": 1, "later": 2}

    monkeypatch.setattr(acceptance, "_spot_targeted_sort_values", targeted_values)
    result, _timings = acceptance._spot_cursor_stream_sample(
        object(),
        acceptance._SpotApiBudget(Client(), 1),
        path="/api/v1/conversations",
        parameters={"sort": "messages", "direction": "asc"},
        entity="conversation",
        sort_name="messages",
        direction="asc",
        expected_total=2,
    )

    assert result["passed"] is False
    assert "sample_order_mismatch" in result["issues"]
    assert targeted == ["later", "earlier"]
    assert result["targeted_lookup_rows"] == 2


def test_production_spot_check_targeted_lookup_is_id_bounded_without_order_by(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acceptance = _acceptance_module()
    statements: list[str] = []

    def rows(
        _connection: object,
        statement: str,
        parameters: Mapping[str, object] | None = None,
    ) -> list[SimpleNamespace]:
        statements.append(statement)
        assert " ID IN (" in statement.upper()
        assert "ORDER BY" not in statement.upper()
        return [
            SimpleNamespace(id=value, sort_value=index)
            for index, value in enumerate((parameters or {}).values())
        ]

    monkeypatch.setattr(acceptance, "_rows", rows)
    values = acceptance._spot_targeted_sort_values(
        object(),
        entity="conversation",
        sort_name="messages",
        identities=["sample-a", "sample-b"],
    )

    assert values == {"sample-a": 0, "sample-b": 1}
    assert len(statements) == 1
    with pytest.raises(RuntimeError, match="row cap exceeded"):
        acceptance._spot_targeted_sort_values(
            object(),
            entity="conversation",
            sort_name="messages",
            identities=[
                f"sample-{index}"
                for index in range(acceptance.PRODUCTION_SPOT_CHECK_MAX_TARGETED_LOOKUP_ROWS + 1)
            ],
        )
    assert len(statements) == 1


def test_production_spot_check_prefix_rejects_non_progress_cursor() -> None:
    acceptance = _acceptance_module()

    class Client:
        def __init__(self) -> None:
            self.calls = 0

        def get_json(
            self,
            _path: str,
            parameters: Mapping[str, object] | None = None,
            *,
            timeout: float = 120.0,
        ) -> tuple[dict[str, Any], float]:
            del timeout
            self.calls += 1
            return {
                "items": [],
                "page": {
                    "total": 1,
                    "limit": (parameters or {})["limit"],
                    "next_cursor": f"cursor-{self.calls}",
                },
            }, 0.0

    client = Client()
    result, _timings = acceptance._spot_cursor_stream_sample(
        object(),
        acceptance._SpotApiBudget(client, 2),
        path="/api/v1/conversations",
        parameters={"sort": "messages", "direction": "asc"},
        entity="conversation",
        sort_name="messages",
        direction="asc",
        expected_total=1,
    )

    assert result["passed"] is False
    assert "non_progress_page" in result["issues"]
    assert "short_nonterminal_page" in result["issues"]
    assert client.calls == 2


def _spot_semantic_evaluation_fixture(
    acceptance: ModuleType,
    *,
    config_id: str,
    vector_table: str,
    window_count: int,
    corpus_revision: str,
    held_revision: Mapping[str, object],
    development_revision: Mapping[str, object],
    held_metric_value: float = 1.0,
    incumbent_metrics: Mapping[str, float] | None = None,
) -> dict[str, object]:
    relevant = acceptance.DIRECT_CONFLICT_DEVELOPMENT_MIN_RELEVANT_COUNT
    not_relevant = acceptance.DIRECT_CONFLICT_DEVELOPMENT_MIN_NOT_RELEVANT_COUNT
    revision_count = development_revision.get("row_count")
    judgment_count = (
        revision_count
        if isinstance(revision_count, int)
        and not isinstance(revision_count, bool)
        and revision_count >= relevant + not_relevant
        else relevant + not_relevant
    )
    uncertain = judgment_count - relevant - not_relevant
    query_count = acceptance.HELD_OUT_MIN_QUERY_COUNT

    def retrieval_metrics(count: int, value: float) -> dict[str, object]:
        return {
            "query_count": count,
            "k": acceptance.SEMANTIC_IR_K,
            "recall_at_k": value,
            "mrr": value,
            "ndcg_at_k": value,
        }

    def candidate_recall(count: int, value: float) -> dict[str, object]:
        return {
            "query_count": count,
            **{f"recall_at_{k}": value for k in acceptance.SEMANTIC_CANDIDATE_RECALL_KS},
        }

    semantic_metrics = retrieval_metrics(query_count, held_metric_value)
    hybrid_metrics = retrieval_metrics(0, 0.0)
    semantic_recall = candidate_recall(query_count, held_metric_value)
    hybrid_recall = candidate_recall(0, 0.0)
    development_public_metrics = {name: 0.9 for name in acceptance.DIRECT_CONFLICT_PUBLIC_METRICS}
    if incumbent_metrics is None:
        incumbent_comparison: dict[str, object] = {
            "status": "not_required_no_different_active_incumbent",
            "required_for_activation": False,
            "passed": None,
            "reason_codes": [],
            "non_regression_tolerance": (acceptance.SEMANTIC_UPGRADE_NON_REGRESSION_TOLERANCE),
            "demonstrated_improvement_threshold": (
                acceptance.SEMANTIC_UPGRADE_DEMONSTRATED_IMPROVEMENT
            ),
            "demonstrated_improvement_required": False,
            "demonstrated_improvement": None,
        }
    else:
        comparison_metrics: dict[str, object] = {}
        for name in acceptance.SEMANTIC_UPGRADE_METRICS:
            candidate_value = float(semantic_metrics[name])
            incumbent_value = float(incumbent_metrics[name])
            delta = candidate_value - incumbent_value
            comparison_metrics[name] = {
                "candidate": candidate_value,
                "incumbent": incumbent_value,
                "delta": delta,
                "regression_exceeds_tolerance": (
                    incumbent_value - candidate_value
                    > acceptance.SEMANTIC_UPGRADE_NON_REGRESSION_TOLERANCE
                ),
                "demonstrated_improvement": (
                    delta >= acceptance.SEMANTIC_UPGRADE_DEMONSTRATED_IMPROVEMENT
                ),
            }
        incumbent_comparison = {
            "status": "passed",
            "required_for_activation": True,
            "passed": True,
            "non_regression_tolerance": (acceptance.SEMANTIC_UPGRADE_NON_REGRESSION_TOLERANCE),
            "demonstrated_improvement_threshold": (
                acceptance.SEMANTIC_UPGRADE_DEMONSTRATED_IMPROVEMENT
            ),
            "demonstrated_improvement_required": True,
            "demonstrated_improvement": any(
                bool(row["demonstrated_improvement"]) for row in comparison_metrics.values()
            ),
            "metrics": comparison_metrics,
        }
    return {
        "passed": True,
        "production_acceptance_passed": True,
        "decision": "accepted",
        "evaluator_version": acceptance.SEMANTIC_IR_EVALUATOR_VERSION,
        "candidate": {
            "config_hash": config_id,
            "vector_table": vector_table,
            "corpus_revision": corpus_revision,
            "window_count": window_count,
            "distance_metric": "cosine",
            "query_engine": "lancedb-exact-vector-search",
            "approximate_index_requested": window_count >= 256,
        },
        "automatic_known_item": {
            "benchmark_kind": "automatic_known_item",
            "status": "passed",
            "passed": True,
            "used_for_activation": True,
            "sampling": {
                "method": "stable-id-prefix-stratified-by-entity-type",
                "query_cap": acceptance.KNOWN_ITEM_QUERY_CAP,
                "sampled_query_count": 2,
                "query_text_persisted": False,
                "query_details_persisted": False,
            },
            "thresholds": {
                **acceptance.KNOWN_ITEM_THRESHOLDS,
                "k": acceptance.SEMANTIC_IR_K,
            },
            "metrics": retrieval_metrics(2, 1.0),
        },
        "held_out_feedback": {
            "benchmark_kind": "held_out_user_feedback",
            "status": "passed",
            "passed": True,
            "used_for_activation": True,
            "ranking_pipeline": acceptance.SEMANTIC_RANKING_PIPELINE_VERSION,
            "reranker": asdict(acceptance.DEFAULT_RERANKER_SPEC),
            "query_details_persisted": False,
            "feedback_revision": dict(held_revision),
            "feedback_revision_stable_during_evaluation": True,
            "minimum_query_count_for_activation": acceptance.HELD_OUT_MIN_QUERY_COUNT,
            "query_cap": acceptance.FEEDBACK_QUERY_CAP,
            "judgment_cap_per_query": acceptance.FEEDBACK_JUDGMENT_CAP_PER_QUERY,
            "thresholds": {
                **acceptance.HELD_OUT_THRESHOLDS,
                "k": acceptance.SEMANTIC_IR_K,
            },
            "metrics": semantic_metrics,
            "query_group_counts_by_mode": {
                "semantic": query_count,
                "hybrid": 0,
            },
            "metrics_by_mode": {
                "semantic": semantic_metrics,
                "hybrid": hybrid_metrics,
            },
            "cosine_candidate_recall": semantic_recall,
            "cosine_candidate_recall_by_mode": {
                "semantic": semantic_recall,
                "hybrid": hybrid_recall,
            },
            "evidence_truncated": False,
            "query_selection_truncated": False,
            "judgments_truncated": False,
            "exclusions": {
                "without_relevant_window_judgment": 0,
                "relevant_target_not_in_candidate": 0,
            },
        },
        "active_incumbent_comparison": incumbent_comparison,
        "direct_conflict_development_calibration": {
            "benchmark_kind": acceptance.DIRECT_CONFLICT_DEVELOPMENT_BENCHMARK_KIND,
            "evaluation_intent": acceptance.DIRECT_CONFLICT_DEVELOPMENT_INTENT,
            "evidence_kind": acceptance.DIRECT_CONFLICT_DEVELOPMENT_EVIDENCE_KIND,
            "held_out": False,
            "required_for_production_acceptance": True,
            "status": "passed",
            "passed": True,
            "method": acceptance.DIRECT_CONFLICT_DEVELOPMENT_METHOD,
            "ranking_pipeline": acceptance.DIRECT_CONFLICT_RANKING_PIPELINE_VERSION,
            "query_details_persisted": False,
            "model_metadata": {
                "classifier": acceptance.DIRECT_CONFLICT_CLASSIFIER,
                "library": "scikit-learn",
                "library_version": "fixture",
                "ranker_version": acceptance.DIRECT_CONFLICT_RANKER_VERSION,
                "score_kind": acceptance.DIRECT_CONFLICT_SCORE_KIND,
                "parameters": asdict(acceptance.DEFAULT_DIRECT_CONFLICT_PARAMETERS),
            },
            "minimums": {
                "relevant_count": relevant,
                "not_relevant_count": not_relevant,
                "fold_count": acceptance.DIRECT_CONFLICT_DEVELOPMENT_FOLD_COUNT,
            },
            "thresholds": {
                **acceptance.DIRECT_CONFLICT_DEVELOPMENT_THRESHOLDS,
                "k": acceptance.DIRECT_CONFLICT_RANKING_K,
            },
            "judgment_count": judgment_count,
            "reviewed_judgment_count": judgment_count,
            "binary_judgment_count": relevant + not_relevant,
            "conversation_group_count": max(
                relevant,
                acceptance.DIRECT_CONFLICT_DEVELOPMENT_FOLD_COUNT,
            ),
            "fold_count": acceptance.DIRECT_CONFLICT_DEVELOPMENT_FOLD_COUNT,
            "relevant_count": relevant,
            "not_relevant_count": not_relevant,
            "uncertain_count": uncertain,
            "grouped_cross_validation": {
                "fold_count": acceptance.DIRECT_CONFLICT_DEVELOPMENT_FOLD_COUNT,
                "minimum_metrics": development_public_metrics,
                "mean_metrics": development_public_metrics,
                "passed": True,
            },
            "literal_query_regression": {
                "candidate_count": (acceptance.DIRECT_CONFLICT_LITERAL_REGRESSION_REVIEWED_COUNT),
                "reviewed_judgment_count": (
                    acceptance.DIRECT_CONFLICT_LITERAL_REGRESSION_REVIEWED_COUNT
                ),
                "relevant_judgment_count": (
                    acceptance.DIRECT_CONFLICT_LITERAL_REGRESSION_RELEVANT_COUNT
                ),
                "not_relevant_judgment_count": (
                    acceptance.DIRECT_CONFLICT_LITERAL_REGRESSION_NOT_RELEVANT_COUNT
                ),
                "uncertain_judgment_count": 0,
                "cohort_complete": True,
                "metrics": {
                    "query_count": 1,
                    "k": acceptance.DIRECT_CONFLICT_RANKING_K,
                    **{name: 0.9 for name in acceptance.DIRECT_CONFLICT_DEVELOPMENT_THRESHOLDS},
                },
                "passed": True,
                "failure_code": None,
            },
            "feedback_revision": dict(development_revision),
            "feedback_revision_stable_during_evaluation": True,
            "private_material_fingerprint_sha256": "f" * 64,
        },
    }


def _semantic_config_audit_fixture(
    acceptance: ModuleType,
    *,
    spec: object,
    accepted_evaluation: Mapping[str, object],
    artifact_fingerprint: str,
    corpus_revision: str,
) -> dict[str, object]:
    return {
        "selected_model": asdict(spec),
        "language_audit": {
            "recommended_model": acceptance.AUTO_EMBEDDING_MODEL_ALIAS,
            "selection_policy": acceptance.SEMANTIC_MODEL_SELECTION_POLICY,
            "threshold_used_for_selection": False,
        },
        "recommended_model": acceptance.AUTO_EMBEDDING_MODEL_ALIAS,
        "model_selection": "automatic_default",
        "artifact_fingerprint": artifact_fingerprint,
        "embedding_implementation": acceptance.embedding_implementation_identity(spec),
        "corpus_revision": corpus_revision,
        "accepted_semantic_ir_evaluation": dict(accepted_evaluation),
    }


def _legacy_e5_evaluation_fixture(
    acceptance: ModuleType,
    evaluation: Mapping[str, object],
) -> dict[str, object]:
    legacy = json.loads(json.dumps(evaluation))
    legacy["evaluator_version"] = acceptance.LEGACY_SEMANTIC_IR_EVALUATOR_VERSION
    # The build-time comparator intentionally used held-out acceptance, not the later
    # direct-conflict production flag.
    legacy["production_acceptance_passed"] = False
    legacy.pop("active_incumbent_comparison", None)
    held_out = legacy["held_out_feedback"]
    for field in (
        "metrics_by_mode",
        "cosine_candidate_recall",
        "cosine_candidate_recall_by_mode",
        "evidence_truncated",
    ):
        held_out.pop(field, None)
    return legacy


def test_production_spot_check_parses_string_semantic_audit_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acceptance = _acceptance_module()
    import lancedb

    spec = acceptance.SEMANTIC_MODEL_SPECS[acceptance.AUTO_EMBEDDING_MODEL_ALIAS]
    window_id = "a" * 64
    artifact_fingerprint = f"sha256:{'a' * 64}"
    corpus_revision = "b" * 64
    config_id = acceptance.embedding_config_hash(
        spec,
        corpus_revision=corpus_revision,
        artifact_fingerprint=artifact_fingerprint,
    )
    vector_table = f"text_windows_{config_id[:16]}"
    backend, backend_version = acceptance.embedding_backend(spec)
    incumbent_spec = acceptance.SEMANTIC_MODEL_SPECS["multilingual-e5-small"]
    incumbent_artifact_fingerprint = f"sha256:{'e' * 64}"
    incumbent_backend, incumbent_backend_version = acceptance.embedding_backend(incumbent_spec)
    incumbent_config_id = acceptance._legacy_e5_config_hash(
        incumbent_spec,
        backend=incumbent_backend,
        backend_version=incumbent_backend_version,
        corpus_revision=corpus_revision,
        artifact_fingerprint=incumbent_artifact_fingerprint,
    )
    incumbent_vector_table = f"text_windows_{incumbent_config_id[:16]}"
    held_revision = {"version": "held", "sha256": "c" * 64, "row_count": 5}
    development_row_count = (
        acceptance.DIRECT_CONFLICT_DEVELOPMENT_MIN_RELEVANT_COUNT
        + acceptance.DIRECT_CONFLICT_DEVELOPMENT_MIN_NOT_RELEVANT_COUNT
    )
    development_revision = {
        "version": "development",
        "sha256": "d" * 64,
        "row_count": development_row_count,
    }
    incumbent_metrics = {name: 0.9 for name in acceptance.SEMANTIC_UPGRADE_METRICS}
    incumbent_evaluation = _legacy_e5_evaluation_fixture(
        acceptance,
        _spot_semantic_evaluation_fixture(
            acceptance,
            config_id=incumbent_config_id,
            vector_table=incumbent_vector_table,
            window_count=1,
            corpus_revision=corpus_revision,
            held_revision=held_revision,
            development_revision=development_revision,
            held_metric_value=0.9,
        ),
    )
    incumbent_audit = _semantic_config_audit_fixture(
        acceptance,
        spec=incumbent_spec,
        accepted_evaluation=incumbent_evaluation,
        artifact_fingerprint=incumbent_artifact_fingerprint,
        corpus_revision=corpus_revision,
    )
    incumbent_audit["selected_model"] = acceptance._legacy_e5_model_identity(incumbent_spec)
    incumbent_audit["embedding_implementation"] = acceptance._legacy_e5_embedding_implementation(
        incumbent_spec
    )
    incumbent_audit["model_selection"] = "explicit_config"
    accepted_evaluation = _spot_semantic_evaluation_fixture(
        acceptance,
        config_id=config_id,
        vector_table=vector_table,
        window_count=1,
        corpus_revision=corpus_revision,
        held_revision=held_revision,
        development_revision=development_revision,
        incumbent_metrics=incumbent_metrics,
    )
    audit = _semantic_config_audit_fixture(
        acceptance,
        spec=spec,
        accepted_evaluation=accepted_evaluation,
        artifact_fingerprint=artifact_fingerprint,
        corpus_revision=corpus_revision,
    )
    config = SimpleNamespace(
        id=config_id,
        status="ready",
        backend=backend,
        backend_version=backend_version,
        model_name=spec.model_name,
        model_revision=spec.model_revision,
        dimensions=spec.dimensions,
        distance_metric="cosine",
        document_prefix=spec.document_prefix,
        query_prefix=spec.query_prefix,
        vector_table=vector_table,
        window_version=acceptance.SEARCH_WINDOW_VERSION,
        window_count=1,
        audit_json=json.dumps(audit),
    )
    configs = [config]
    incumbent_configs = [
        SimpleNamespace(
            id=incumbent_config_id,
            backend=incumbent_backend,
            backend_version=incumbent_backend_version,
            distance_metric="cosine",
            vector_table=incumbent_vector_table,
            window_version=acceptance.SEARCH_WINDOW_VERSION,
            window_count=1,
            audit_json=json.dumps(incumbent_audit),
        )
    ]

    def rows(
        _connection: object, statement: str, *_args: object, **_kwargs: object
    ) -> list[object]:
        return incumbent_configs if "active = 0" in statement else configs

    monkeypatch.setattr(acceptance, "_rows", rows)
    monkeypatch.setattr(acceptance, "held_out_feedback_revision", lambda _engine: held_revision)
    monkeypatch.setattr(
        acceptance,
        "development_feedback_revision",
        lambda _engine: development_revision,
    )

    def scalar(_connection: object, statement: str, _parameters: object = None) -> int:
        assert "OFFSET" not in statement.upper()
        return 1

    monkeypatch.setattr(acceptance, "_scalar", scalar)
    monkeypatch.setattr(
        acceptance,
        "_spot_keyset_rows",
        lambda *_args, **_kwargs: [SimpleNamespace(id=window_id)],
    )

    class Table:
        def count_rows(self) -> int:
            return 1

        def search(self) -> Table:
            return self

        def where(self, _predicate: str) -> Table:
            return self

        def limit(self, _limit: int) -> Table:
            return self

        def select(self, _columns: list[str]) -> Table:
            return self

        def to_list(self) -> list[dict[str, object]]:
            return [
                {
                    "id": window_id,
                    "vector": [1.0, *([0.0] * (spec.dimensions - 1))],
                }
            ]

    class Database:
        def open_table(self, table_name: str) -> Table:
            assert table_name == vector_table
            return Table()

    monkeypatch.setattr(lancedb, "connect", lambda _path: Database())
    monkeypatch.setattr(
        acceptance,
        "pinned_transformers_artifact",
        lambda *_args, **_kwargs: pytest.fail(
            "bounded spot acceptance must not hash the model artifact"
        ),
    )

    checks, report = acceptance._spot_semantic_lance_sample(
        ArchiveLayout(Path("/fixture/archive")),
        object(),
        object(),
        {"semantic_index_configs", "semantic_window_embeddings", "search_windows"},
        seed=20260722,
    )

    assert checks[0].status == "pass"
    assert report["certification_metadata_current"] is True
    assert report["lance_sample_valid"] is True
    assert report["production_model_provenance"]["artifact_verification_mode"] == (
        "bounded-metadata-only-reduced-assurance"
    )

    tampered_incumbent_audit = json.loads(json.dumps(incumbent_audit))
    tampered_incumbent_audit["selected_model"]["model_revision"] = "stale"
    incumbent_configs[0] = SimpleNamespace(
        **{
            **vars(incumbent_configs[0]),
            "audit_json": json.dumps(tampered_incumbent_audit),
        }
    )
    incumbent_valid, incumbent_report = acceptance._semantic_upgrade_incumbent_evidence(
        object(), accepted_evaluation
    )
    assert incumbent_valid is False
    assert incumbent_report["provenance_current_incumbent_count"] == 0
    incumbent_configs[0] = SimpleNamespace(
        **{
            **vars(incumbent_configs[0]),
            "audit_json": json.dumps(incumbent_audit),
        }
    )

    stale_evaluation = json.loads(json.dumps(accepted_evaluation))
    stale_evaluation["automatic_known_item"]["metrics"]["mrr"] = 0.0
    configs[0] = SimpleNamespace(
        **{
            **vars(config),
            "audit_json": json.dumps(
                {
                    **audit,
                    "accepted_semantic_ir_evaluation": stale_evaluation,
                }
            ),
        }
    )
    stale, stale_report = acceptance._spot_semantic_certification_metadata(object(), configs[0])
    assert stale is False
    assert stale_report["automatic_evaluation_valid"] is False

    configs[0] = config
    monkeypatch.setattr(
        acceptance,
        "held_out_feedback_revision",
        lambda _engine: {**held_revision, "sha256": "e" * 64},
    )
    stale_revision, stale_revision_report = acceptance._spot_semantic_certification_metadata(
        object(), config
    )
    assert stale_revision is False
    assert stale_revision_report["held_out_feedback_revision_current"] is False


def test_current_semantic_evidence_rejects_incomplete_or_forged_payloads() -> None:
    acceptance = _acceptance_module()
    evaluation = _spot_semantic_evaluation_fixture(
        acceptance,
        config_id="a" * 64,
        vector_table=f"text_windows_{'a' * 16}",
        window_count=255,
        corpus_revision="b" * 64,
        held_revision={"version": "held", "sha256": "c" * 64, "row_count": 5},
        development_revision={
            "version": "development",
            "sha256": "d" * 64,
            "row_count": 20,
        },
    )

    valid, report = acceptance._semantic_v7_evidence(evaluation)
    assert valid is True
    assert report["search_mode_schema_valid"] is True
    assert report["aggregate_metrics_reconcile_by_mode"] is True

    improved_evaluation = _spot_semantic_evaluation_fixture(
        acceptance,
        config_id="a" * 64,
        vector_table=f"text_windows_{'a' * 16}",
        window_count=255,
        corpus_revision="b" * 64,
        held_revision={"version": "held", "sha256": "c" * 64, "row_count": 5},
        development_revision={
            "version": "development",
            "sha256": "d" * 64,
            "row_count": 20,
        },
        incumbent_metrics={name: 0.9 for name in acceptance.SEMANTIC_UPGRADE_METRICS},
    )
    improved_valid, improved_report = acceptance._semantic_v7_evidence(improved_evaluation)
    assert improved_valid is True
    assert improved_report["active_incumbent_comparison_valid"] is True

    flat_evaluation = _spot_semantic_evaluation_fixture(
        acceptance,
        config_id="a" * 64,
        vector_table=f"text_windows_{'a' * 16}",
        window_count=255,
        corpus_revision="b" * 64,
        held_revision={"version": "held", "sha256": "c" * 64, "row_count": 5},
        development_revision={
            "version": "development",
            "sha256": "d" * 64,
            "row_count": 20,
        },
        incumbent_metrics={name: 1.0 for name in acceptance.SEMANTIC_UPGRADE_METRICS},
    )
    flat_valid, flat_report = acceptance._semantic_v7_evidence(flat_evaluation)
    assert flat_valid is False
    assert flat_report["active_incumbent_comparison_valid"] is False

    wrong_approximation_policy = json.loads(json.dumps(evaluation))
    wrong_approximation_policy["candidate"]["approximate_index_requested"] = True
    wrong_approximation_valid, wrong_approximation_report = acceptance._semantic_v7_evidence(
        wrong_approximation_policy
    )
    assert wrong_approximation_valid is False
    assert wrong_approximation_report["candidate_metadata_current"] is False

    for field in (
        "evidence_truncated",
        "query_selection_truncated",
        "judgments_truncated",
    ):
        truncated = json.loads(json.dumps(evaluation))
        truncated["held_out_feedback"][field] = True
        truncated_valid, truncated_report = acceptance._semantic_v7_evidence(truncated)
        assert truncated_valid is False
        assert truncated_report["evidence_not_truncated"] is False

    relabeled_v6 = json.loads(json.dumps(evaluation))
    relabeled_v6["held_out_feedback"].pop("cosine_candidate_recall")
    relabeled_valid, relabeled_report = acceptance._semantic_v7_evidence(relabeled_v6)
    assert relabeled_valid is False
    assert relabeled_report["cosine_candidate_recall_valid"] is False

    unreconciled = json.loads(json.dumps(evaluation))
    unreconciled["held_out_feedback"]["metrics"]["mrr"] = 0.95
    unreconciled_valid, unreconciled_report = acceptance._semantic_v7_evidence(unreconciled)
    assert unreconciled_valid is False
    assert unreconciled_report["aggregate_metrics_reconcile_by_mode"] is False

    invalid_automatic_count = json.loads(json.dumps(evaluation))
    invalid_automatic_count["automatic_known_item"]["sampling"]["sampled_query_count"] = True
    invalid_automatic_count["automatic_known_item"]["metrics"]["query_count"] = -1
    invalid_automatic_valid, invalid_automatic_report = acceptance._semantic_v7_evidence(
        invalid_automatic_count
    )
    assert invalid_automatic_valid is False
    assert invalid_automatic_report["automatic_evidence_valid"] is False

    boolean_automatic_metric_count = json.loads(json.dumps(evaluation))
    boolean_automatic_metric_count["automatic_known_item"]["sampling"]["sampled_query_count"] = 1
    boolean_automatic_metric_count["automatic_known_item"]["metrics"]["query_count"] = True
    boolean_metric_valid, boolean_metric_report = acceptance._semantic_v7_evidence(
        boolean_automatic_metric_count
    )
    assert boolean_metric_valid is False
    assert boolean_metric_report["automatic_evidence_valid"] is False

    boolean_zero_mode_counts = json.loads(json.dumps(evaluation))
    boolean_zero_mode_counts["held_out_feedback"]["metrics_by_mode"]["hybrid"]["query_count"] = (
        False
    )
    boolean_zero_mode_counts["held_out_feedback"]["cosine_candidate_recall_by_mode"]["hybrid"][
        "query_count"
    ] = False
    boolean_zero_valid, boolean_zero_report = acceptance._semantic_v7_evidence(
        boolean_zero_mode_counts
    )
    assert boolean_zero_valid is False
    assert boolean_zero_report["metrics_by_mode_valid"] is False
    assert boolean_zero_report["cosine_candidate_recall_by_mode_valid"] is False

    automatic_over_cap = json.loads(json.dumps(evaluation))
    automatic_over_cap["automatic_known_item"]["sampling"]["sampled_query_count"] = (
        acceptance.KNOWN_ITEM_QUERY_CAP + 1
    )
    automatic_over_cap["automatic_known_item"]["metrics"]["query_count"] = (
        acceptance.KNOWN_ITEM_QUERY_CAP + 1
    )
    automatic_over_cap_valid, automatic_over_cap_report = acceptance._semantic_v7_evidence(
        automatic_over_cap
    )
    assert automatic_over_cap_valid is False
    assert automatic_over_cap_report["automatic_evidence_valid"] is False

    held_over_cap = json.loads(json.dumps(evaluation))
    over_cap_count = acceptance.FEEDBACK_QUERY_CAP + 1
    held_over_cap["held_out_feedback"]["metrics"]["query_count"] = over_cap_count
    held_over_cap["held_out_feedback"]["query_group_counts_by_mode"]["semantic"] = over_cap_count
    held_over_cap["held_out_feedback"]["metrics_by_mode"]["semantic"]["query_count"] = (
        over_cap_count
    )
    held_over_cap["held_out_feedback"]["cosine_candidate_recall"]["query_count"] = over_cap_count
    held_over_cap["held_out_feedback"]["cosine_candidate_recall_by_mode"]["semantic"][
        "query_count"
    ] = over_cap_count
    held_over_cap_valid, held_over_cap_report = acceptance._semantic_v7_evidence(held_over_cap)
    assert held_over_cap_valid is False
    assert held_over_cap_report["aggregate_metrics_valid"] is False

    forged_regression = json.loads(json.dumps(evaluation))
    forged_regression["active_incumbent_comparison"] = {
        "status": "passed",
        "required_for_activation": True,
        "passed": True,
        "non_regression_tolerance": acceptance.SEMANTIC_UPGRADE_NON_REGRESSION_TOLERANCE,
        "demonstrated_improvement_threshold": (
            acceptance.SEMANTIC_UPGRADE_DEMONSTRATED_IMPROVEMENT
        ),
        "demonstrated_improvement_required": True,
        "demonstrated_improvement": False,
        "metrics": {
            name: {
                "candidate": 0.5,
                "incumbent": 0.8,
                "delta": -0.3,
                "regression_exceeds_tolerance": False,
                "demonstrated_improvement": False,
            }
            for name in acceptance.SEMANTIC_UPGRADE_METRICS
        },
    }
    forged_valid, forged_report = acceptance._semantic_v7_evidence(forged_regression)
    assert forged_valid is False
    assert forged_report["active_incumbent_comparison_valid"] is False

    complete_direct_conflict, direct_conflict_report = (
        acceptance._complete_direct_conflict_evidence(
            evaluation["direct_conflict_development_calibration"],
            current_revision={
                "version": "development",
                "sha256": "d" * 64,
                "row_count": 20,
            },
        )
    )
    assert complete_direct_conflict is True
    assert direct_conflict_report["literal_regression_cohort_complete"] is True
    incomplete_direct_conflict = json.loads(json.dumps(evaluation))
    incomplete_direct_conflict["direct_conflict_development_calibration"][
        "literal_query_regression"
    ]["cohort_complete"] = False
    incomplete_valid, incomplete_report = acceptance._complete_direct_conflict_evidence(
        incomplete_direct_conflict["direct_conflict_development_calibration"],
        current_revision={
            "version": "development",
            "sha256": "d" * 64,
            "row_count": 20,
        },
    )
    assert incomplete_valid is False
    assert incomplete_report["literal_regression_cohort_complete"] is False

    incumbent_evaluation = _spot_semantic_evaluation_fixture(
        acceptance,
        config_id="e" * 64,
        vector_table=f"text_windows_{'e' * 16}",
        window_count=255,
        corpus_revision="b" * 64,
        held_revision={"version": "held", "sha256": "c" * 64, "row_count": 5},
        development_revision={
            "version": "development",
            "sha256": "d" * 64,
            "row_count": 20,
        },
        held_metric_value=0.9,
    )
    incumbent_evaluation["evaluator_version"] = (
        acceptance.PREVIOUS_SEMANTIC_IR_EVALUATOR_VERSION
    )
    candidate_evaluation = _spot_semantic_evaluation_fixture(
        acceptance,
        config_id="a" * 64,
        vector_table=f"text_windows_{'a' * 16}",
        window_count=255,
        corpus_revision="b" * 64,
        held_revision={"version": "held", "sha256": "c" * 64, "row_count": 5},
        development_revision={
            "version": "development",
            "sha256": "d" * 64,
            "row_count": 20,
        },
        incumbent_metrics={name: 0.9 for name in acceptance.SEMANTIC_UPGRADE_METRICS},
    )
    comparison_valid, comparison_report = acceptance._semantic_upgrade_comparison_binding(
        candidate_evaluation,
        incumbent_evaluation,
    )
    assert comparison_valid is True
    assert comparison_report["stored_values_bound_to_candidate_and_incumbent"] is True
    assert comparison_report["incumbent_metrics"]["previous_evaluator_evidence"] is True
    unbound_candidate = json.loads(json.dumps(candidate_evaluation))
    unbound_row = unbound_candidate["active_incumbent_comparison"]["metrics"]["mrr"]
    unbound_row.update(
        {
            "candidate": 0.99,
            "delta": 0.09,
            "regression_exceeds_tolerance": False,
            "demonstrated_improvement": True,
        }
    )
    unbound_valid, unbound_report = acceptance._semantic_upgrade_comparison_binding(
        unbound_candidate,
        incumbent_evaluation,
    )
    assert unbound_valid is False
    assert unbound_report["stored_comparison_required_and_valid"] is True
    assert unbound_report["stored_values_bound_to_candidate_and_incumbent"] is False


def test_production_semantic_config_provenance_requires_exact_automatic_model_hashes() -> None:
    acceptance = _acceptance_module()
    spec = acceptance.SEMANTIC_MODEL_SPECS[acceptance.AUTO_EMBEDDING_MODEL_ALIAS]
    artifact_fingerprint = f"sha256:{'a' * 64}"
    corpus_revision = "b" * 64
    config_id = acceptance.embedding_config_hash(
        spec,
        corpus_revision=corpus_revision,
        artifact_fingerprint=artifact_fingerprint,
    )
    vector_table = f"text_windows_{config_id[:16]}"
    evaluation = _spot_semantic_evaluation_fixture(
        acceptance,
        config_id=config_id,
        vector_table=vector_table,
        window_count=1,
        corpus_revision=corpus_revision,
        held_revision={"version": "held", "sha256": "c" * 64, "row_count": 5},
        development_revision={
            "version": "development",
            "sha256": "d" * 64,
            "row_count": 20,
        },
    )
    audit = _semantic_config_audit_fixture(
        acceptance,
        spec=spec,
        accepted_evaluation=evaluation,
        artifact_fingerprint=artifact_fingerprint,
        corpus_revision=corpus_revision,
    )
    backend, backend_version = acceptance.embedding_backend(spec)
    config = SimpleNamespace(
        id=config_id,
        backend=backend,
        backend_version=backend_version,
        model_name=spec.model_name,
        model_revision=spec.model_revision,
        dimensions=spec.dimensions,
        distance_metric="cosine",
        document_prefix=spec.document_prefix,
        query_prefix=spec.query_prefix,
        vector_table=vector_table,
        window_version=acceptance.SEARCH_WINDOW_VERSION,
        window_count=1,
        audit_json=audit,
    )

    valid, report = acceptance._production_semantic_config_provenance(config)
    assert valid is True
    assert report["candidate_corpus_revision_current"] is True
    assert report["selection_metadata_mode"] == "current"
    assert report["artifact_verification_mode"] == ("bounded-metadata-only-reduced-assurance")

    fully_verified, full_report = acceptance._production_semantic_config_provenance(
        config,
        independently_recomputed_artifact_fingerprint=artifact_fingerprint,
        require_independent_artifact_fingerprint=True,
    )
    assert fully_verified is True
    assert full_report["artifact_fingerprint_independently_verified"] is True
    mismatched_artifact, mismatched_artifact_report = (
        acceptance._production_semantic_config_provenance(
            config,
            independently_recomputed_artifact_fingerprint=f"sha256:{'0' * 64}",
            require_independent_artifact_fingerprint=True,
        )
    )
    assert mismatched_artifact is False
    assert mismatched_artifact_report["artifact_fingerprint_independently_verified"] is False

    malformed = SimpleNamespace(
        **{
            **vars(config),
            "audit_json": {
                **audit,
                "artifact_fingerprint": "sha256:fixture",
            },
        }
    )
    malformed_valid, malformed_report = acceptance._production_semantic_config_provenance(malformed)
    assert malformed_valid is False
    assert malformed_report["artifact_fingerprint_recorded"] is False

    mismatched_candidate = json.loads(json.dumps(audit))
    mismatched_candidate["accepted_semantic_ir_evaluation"]["candidate"]["corpus_revision"] = (
        "e" * 64
    )
    mismatch_valid, mismatch_report = acceptance._production_semantic_config_provenance(
        SimpleNamespace(**{**vars(config), "audit_json": mismatched_candidate})
    )
    assert mismatch_valid is False
    assert mismatch_report["candidate_corpus_revision_current"] is False

    for field, stale_value in (
        ("backend", "stale-backend"),
        ("backend_version", "stale-version"),
        ("dimensions", spec.dimensions + 1),
        ("distance_metric", "dot"),
        ("document_prefix", "stale document: "),
        ("query_prefix", "stale query: "),
        ("vector_table", "stale_vector_table"),
        ("window_version", "stale-window-version"),
        ("window_count", 2),
        ("id", "0" * 64),
    ):
        stale_config = SimpleNamespace(**{**vars(config), field: stale_value})
        stale_valid, _stale_report = acceptance._production_semantic_config_provenance(stale_config)
        assert stale_valid is False, field

    for audit_mutation in (
        lambda value: value.update({"corpus_revision": "malformed"}),
        lambda value: value["selected_model"].update({"model_revision": "stale"}),
        lambda value: value["embedding_implementation"].update({"pooling": "stale"}),
        lambda value: value.update({"model_selection": "injected"}),
        lambda value: value["language_audit"].update({"selection_policy": "injected"}),
    ):
        stale_audit = json.loads(json.dumps(audit))
        audit_mutation(stale_audit)
        stale_valid, _stale_report = acceptance._production_semantic_config_provenance(
            SimpleNamespace(**{**vars(config), "audit_json": stale_audit})
        )
        assert stale_valid is False


def test_independent_embedding_artifact_fingerprint_uses_provider_pins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acceptance = _acceptance_module()
    layout = ArchiveLayout(Path("/fixture/archive"))
    transformer_spec = acceptance.SEMANTIC_MODEL_SPECS["gte-modernbert-base"]
    fastembed_spec = acceptance.SEMANTIC_MODEL_SPECS["bge-base-en-v1.5"]
    fastembed_artifact = acceptance.PINNED_FASTEMBED_ARTIFACTS[fastembed_spec.alias]
    calls: list[tuple[Path, str, str]] = []

    def pinned_artifact(
        cache_dir: Path,
        *,
        model_name: str,
        model_revision: str,
    ) -> SimpleNamespace:
        calls.append((cache_dir, model_name, model_revision))
        return SimpleNamespace(fingerprint_sha256="a" * 64)

    monkeypatch.setattr(acceptance, "pinned_transformers_artifact", pinned_artifact)

    assert (
        acceptance._independent_embedding_artifact_fingerprint(
            layout,
            transformer_spec,
        )
        == f"sha256:{'a' * 64}"
    )
    assert (
        acceptance._independent_embedding_artifact_fingerprint(
            layout,
            fastembed_spec,
        )
        == f"sha256:{'a' * 64}"
    )
    assert calls == [
        (
            layout.models / "transformers",
            transformer_spec.model_name,
            transformer_spec.model_revision,
        ),
        (
            layout.models / "fastembed",
            fastembed_artifact.repository,
            fastembed_artifact.revision,
        ),
    ]

    unpinned_fastembed = SimpleNamespace(
        **{
            **asdict(fastembed_spec),
            "alias": "not-in-the-pinned-registry",
        }
    )
    mismatched_revision = SimpleNamespace(
        **{
            **asdict(fastembed_spec),
            "model_revision": "0" * 40,
        }
    )
    assert (
        acceptance._independent_embedding_artifact_fingerprint(
            layout,
            unpinned_fastembed,
        )
        is None
    )
    assert (
        acceptance._independent_embedding_artifact_fingerprint(
            layout,
            mismatched_revision,
        )
        is None
    )
    assert len(calls) == 2


def test_production_spot_check_keyset_sample_is_deterministic_and_has_no_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acceptance = _acceptance_module()
    calls: list[tuple[str, dict[str, object]]] = []

    def capture_rows(
        _connection: object,
        statement: str,
        parameters: Mapping[str, object] | None = None,
    ) -> list[object]:
        calls.append((statement, dict(parameters or {})))
        return []

    monkeypatch.setattr(acceptance, "_rows", capture_rows)
    for _attempt in range(2):
        assert (
            acceptance._spot_keyset_rows(
                object(),
                table="nlp_annotations",
                key_column="id",
                columns="id",
                where="active = 1",
                parameters={},
                seed=20260722,
                label="nlp",
            )
            == []
        )

    assert len(calls) == 4
    assert calls[0][1]["spot_pivot"] == calls[2][1]["spot_pivot"]
    assert calls[1][1]["spot_pivot"] == calls[3][1]["spot_pivot"]
    assert all("OFFSET" not in statement.upper() for statement, _parameters in calls)
    for _statement, parameters in calls:
        limit = parameters["spot_limit"]
        assert isinstance(limit, int)
        assert limit <= acceptance.PRODUCTION_SPOT_CHECK_SAMPLE_ROWS


def test_production_spot_check_deep_report_failure_is_sanitized(tmp_path: Path) -> None:
    acceptance = _acceptance_module()
    private_name = "meta-backup-private-name"
    layout = ArchiveLayout(tmp_path / private_name)

    checks, report, marker_counts, generated_ns = acceptance._spot_prior_deep_report(layout)

    assert checks[0].status == "fail"
    assert report["report_observed"] is False
    assert marker_counts == {}
    assert generated_ns is None
    serialized = json.dumps({"report": report})
    assert private_name not in serialized
    assert str(layout.root) not in serialized


def test_production_spot_check_rejects_baseline_mutation() -> None:
    acceptance = _acceptance_module()
    before = acceptance._SpotFileBinding(
        device=1,
        inode=2,
        size_bytes=100,
        mtime_ns=3,
        ctime_ns=4,
        sha256="a" * 64,
    )
    after = acceptance._SpotFileBinding(
        device=1,
        inode=2,
        size_bytes=100,
        mtime_ns=3,
        ctime_ns=5,
        sha256="b" * 64,
    )

    check, report = acceptance._spot_baseline_binding_check(
        before,
        after,
        schema_identity_valid=True,
    )

    assert check.status == "fail"
    assert report["stable_during_audit"] is False
    assert report["digest_stable"] is False
    assert report["ctime_stable"] is False


def test_production_spot_check_rejects_late_or_symlink_sqlite_sidecar(
    tmp_path: Path,
) -> None:
    acceptance = _acceptance_module()
    database = tmp_path / "catalog.sqlite"
    database.touch()
    before = acceptance._spot_sqlite_sidecars(database)
    database.with_name(f"{database.name}-wal").symlink_to(tmp_path / "missing-target")
    after = acceptance._spot_sqlite_sidecars(database)

    check, report = acceptance._spot_sidecar_check(before, after)

    assert check.status == "fail"
    assert report["all_absent_at_start"] is True
    assert report["all_absent_at_end"] is False
    assert report["late_sidecar_detected"] is True


def test_production_spot_check_rejects_final_server_identity_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acceptance = _acceptance_module()
    initial = _fixture_acceptance_identity(nonce_character="a")
    final = {
        **initial,
        "backend_source_sha256": "b" * 64,
        "server_nonce": "c" * 64,
    }
    identity_pass = acceptance.AcceptanceCheck(
        key="api.acceptance_read_only_identity",
        status="pass",
        expected=True,
        actual=True,
        detail="fixture",
    )
    monkeypatch.setattr(
        acceptance,
        "_acceptance_server_identity",
        lambda _client: ([identity_pass], {}, final),
    )

    class Client:
        def get_json(
            self,
            _path: str,
            _parameters: Mapping[str, object] | None = None,
            *,
            timeout: float = 120.0,
        ) -> tuple[dict[str, Any], float]:
            del timeout
            return {}, 0.0

    checks, report = acceptance._spot_refetch_server_identity(
        acceptance._SpotApiBudget(Client(), 1),
        initial,
    )

    assert checks[0].status == "fail"
    assert report["identity_response_valid"] is True
    assert report["stable_revisions"] is False
    assert report["server_nonce_stable"] is False


def test_production_spot_check_orchestration_skips_expensive_full_audits(
    acceptance_catalog: tuple[ArchiveLayout, Engine],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    acceptance = _acceptance_module()
    layout, catalog_engine = acceptance_catalog
    with catalog_engine.connect() as connection:
        totals = acceptance._catalog_totals(connection)
        totals["unlinked_media"] = int(
            connection.execute(
                text(
                    "SELECT count(*) FROM attachment_occurrences "
                    "WHERE deleted_at IS NULL AND availability = 'unlinked'"
                )
            ).scalar_one()
        )
        largest = connection.execute(
            text(
                "SELECT id, message_count FROM conversations WHERE deleted_at IS NULL "
                "ORDER BY message_count DESC, id DESC LIMIT 1"
            )
        ).one()

    def passed(key: str) -> Any:
        return acceptance.AcceptanceCheck(
            key=key,
            status="pass",
            expected=True,
            actual=True,
            detail="fixture",
        )

    marker_counts = {
        "conversation_count": totals["conversations"],
        "message_count": totals["messages"],
        "attachment_occurrence_count": totals["attachments"],
        "activity_record_count": totals["activities"],
        "unlinked_media_count": totals["unlinked_media"],
    }
    identity = _fixture_acceptance_identity(nonce_character="a")
    baseline = SimpleNamespace(largest_conversation_messages=int(largest.message_count))
    baseline_binding = acceptance._SpotFileBinding(
        device=1,
        inode=2,
        size_bytes=100,
        mtime_ns=3,
        ctime_ns=4,
        sha256="f" * 64,
    )
    monkeypatch.setattr(
        acceptance, "_spot_file_binding", lambda *_args, **_kwargs: baseline_binding
    )
    monkeypatch.setattr(acceptance, "load_audited_baseline", lambda _root: baseline)
    monkeypatch.setattr(
        acceptance,
        "_spot_sqlite_sidecars",
        lambda _database: {"-wal": True, "-shm": True, "-journal": True},
    )
    monkeypatch.setattr(
        acceptance,
        "_acceptance_server_identity",
        lambda _client: ([passed("api.acceptance_read_only_identity")], {}, identity),
    )
    monkeypatch.setattr(
        acceptance,
        "_spot_prior_deep_report",
        lambda _layout: (
            [passed("catalog.prior_deep_integrity_evidence")],
            {"private_values_emitted": False},
            marker_counts,
            time.time_ns(),
        ),
    )
    monkeypatch.setattr(
        acceptance,
        "_spot_api_reconciliation",
        lambda _connection, _client, _totals: (
            [passed("api.production_spot_check")],
            {"stream_count": acceptance.PRODUCTION_SPOT_CHECK_SORT_STREAMS},
            {"list_api": [], "date_jump_api": []},
        ),
    )
    monkeypatch.setattr(
        acceptance,
        "_spot_jump_strata",
        lambda _connection, _client, audited_largest: (
            [passed("api.production_spot_check_jump_strata")],
            {"represented_samples": 1},
            {
                "id_sha256": acceptance._opaque(largest.id),
                "message_count": audited_largest,
            },
        ),
    )
    monkeypatch.setattr(
        acceptance,
        "_search_user_story_acceptance",
        lambda _connection, _client, query: (
            [
                passed("api.semantic_hybrid_search_user_story"),
                passed("api.conflict_search_user_story"),
            ],
            {"query_emitted": False},
        ),
    )
    monkeypatch.setattr(
        acceptance,
        "_spot_nlp_sample",
        lambda _connection, _tables, seed: ([passed("nlp.sample")], {"sample_rows": 1}),
    )
    monkeypatch.setattr(
        acceptance,
        "_spot_semantic_lance_sample",
        lambda _layout, _engine, _connection, _tables, seed: (
            [passed("semantic.sample")],
            {"sample_rows": 1},
        ),
    )
    monkeypatch.setattr(
        acceptance,
        "_spot_derived_samples",
        lambda _layout, _connection, _tables, seed: (
            [passed("derived.sample")],
            {"sample_rows": 1},
        ),
    )
    monkeypatch.setattr(
        acceptance,
        "_spot_analytics",
        lambda _client: ([passed("analytics.sample")], {"request_count": 2}),
    )
    monkeypatch.setattr(
        acceptance,
        "_browser_evidence",
        lambda *_args, **_kwargs: ([passed("browser.initial_content_dom_wcag")], {"valid": True}),
    )
    monkeypatch.setattr(
        acceptance,
        "_spot_refetch_server_identity",
        lambda _client, _identity: (
            [passed("api.production_spot_check_identity_stable")],
            {
                "available": True,
                "identity_response_valid": True,
                "stable_revisions": True,
                "server_nonce_stable": True,
            },
        ),
    )
    monkeypatch.setattr(
        acceptance.ArchiveLayout,
        "private_permission_findings",
        lambda _layout: [],
    )

    def forbidden(*_args: object, **_kwargs: object) -> Any:
        raise AssertionError("full-corpus function called by production spot check")

    for name in (
        "verify_catalog",
        "_api_reconciliation",
        "_jump_acceptance",
        "_nlp_integrity",
        "_semantic_integrity",
        "_derived_artifact_integrity",
        "_analytics_direct_sql_reconciliation",
        "_analytics_cache_integrity",
        "_analytics_all_pages",
        "_benchmark_api",
        "validate_semantic_table",
        "lance_table_names",
    ):
        monkeypatch.setattr(acceptance, name, forbidden)

    report = acceptance.build_acceptance_report(
        layout,
        client=None,
        repeats=3,
        sample_count=1,
        query="private argument query",
        browser_report=None,
        reconciliation_policy=acceptance.ReconciliationPolicy(
            mode="production-spot-check",
            seed=20260722,
        ),
    )

    assert report["gate"]["passed"] is True
    assert report["profile"]["profile_version"] == "production-spot-check-v2"
    assert report["profile"]["reduced_assurance"] is True
    assert report["profile"]["all_124_sort_streams_performed"] is False
    assert report["profile"]["offset_pagination_used"] is False
    assert report["profile"]["maximum_sort_streams"] == 8
    assert report["profile"]["maximum_api_calls"] == 48
    assert report["profile"]["prior_deep_evidence_covers_derived_products"] is False
    assert report["profile"]["unsampled_derived_rows_certified"] is False
    assert report["profile"]["unsampled_cursor_positions_certified"] is False
    assert report["profile"]["api_endpoint_internal_work_bounded_by_profile"] is False
    assert "not certified" in report["profile"]["limitation"]
    assert report["verification"]["audited_baseline_binding"]["digest_stable"] is True
    serialized = json.dumps(report, sort_keys=True)
    assert "private argument query" not in serialized
    assert str(layout.root) not in serialized
    progress = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    assert [item["stage"] for item in progress] == list(range(1, 10))
    assert all(item["profile"] == "production-spot-check-v2" for item in progress)


@pytest.mark.parametrize(
    ("defect", "expected_issue"),
    (
        ("gap", "ordered_stream_mismatch"),
        ("duplicate", "duplicate_identity"),
        ("tie_swap", "ordered_stream_mismatch"),
        ("cursor_cycle", "cursor_cycle"),
        ("missing_cursor", "missing_continuation_cursor"),
        ("total_mismatch", "total_mismatch"),
    ),
)
def test_cursor_sort_risk_profile_fails_closed_on_sampled_boundary_defects(
    defect: str, expected_issue: str
) -> None:
    acceptance = _acceptance_module()
    identities = [f"private-risk-defect-{index:03d}" for index in range(600)]
    client = _CursorAuditClient(identities, defect=defect)

    summary, _timings = acceptance._bounded_cursor_order_reconciliation(
        client,
        "/api/v1/fixture",
        {"sort": "date", "direction": "desc"},
        identity_key="id",
        expected=iter(identities),
        expected_total=len(identities),
        page_cap=2,
    )

    assert summary["passed"] is False
    assert expected_issue in {issue["kind"] for issue in summary["issues"]}
    assert len(client.calls) == 2
    assert "private-risk-defect" not in json.dumps(summary)


def test_cursor_sort_audit_detects_middle_corruption_in_600_row_stream() -> None:
    acceptance = _acceptance_module()
    identities = [f"private-middle-row-{index:03d}" for index in range(600)]

    summary, timings = acceptance._bounded_cursor_order_reconciliation(
        _CursorAuditClient(identities, defect="middle_corruption"),
        "/api/v1/fixture/private-dynamic-id",
        {"sort": "date", "direction": "asc"},
        identity_key="id",
        expected=iter(identities),
        expected_total=len(identities),
    )

    assert summary["passed"] is False
    assert summary["full_stream"] is True
    assert summary["rows_checked"] == 600
    assert len(timings) == 3
    assert "ordered_stream_mismatch" in {issue["kind"] for issue in summary["issues"]}
    serialized = json.dumps(summary)
    assert "private-middle-row" not in serialized
    assert "private-corrupted-middle-row" not in serialized


@pytest.mark.parametrize(
    ("defect", "expected_issue"),
    (
        ("gap", "ordered_stream_mismatch"),
        ("duplicate", "duplicate_identity"),
        ("tie_swap", "ordered_stream_mismatch"),
        ("cursor_cycle", "cursor_cycle"),
        ("missing_cursor", "missing_continuation_cursor"),
        ("total_mismatch", "total_mismatch"),
    ),
)
def test_cursor_sort_audit_fails_closed_on_second_page_defects(
    defect: str, expected_issue: str
) -> None:
    acceptance = _acceptance_module()
    identities = [f"private-defect-row-{index:03d}" for index in range(600)]

    summary, _timings = acceptance._bounded_cursor_order_reconciliation(
        _CursorAuditClient(identities, defect=defect),
        "/api/v1/fixture",
        {"sort": "date", "direction": "asc"},
        identity_key="id",
        expected=identities,
        expected_total=len(identities),
    )

    assert summary["passed"] is False
    issues = summary["issues"]
    assert isinstance(issues, list)
    assert expected_issue in {issue["kind"] for issue in issues}
    assert "private-defect-row" not in json.dumps(summary)
    assert "private-cursor" not in json.dumps(summary)


def test_cursor_sort_audit_accounts_for_unavailable_empty_coverage() -> None:
    acceptance = _acceptance_module()
    clients = (
        _CursorAuditClient((), count_is_unknown=True),
        _CursorAuditClient((), coverage_status="unavailable"),
    )

    for client in clients:
        summary, timings = acceptance._bounded_cursor_order_reconciliation(
            client,
            "/api/v1/fixture",
            {"sort": "date", "direction": "asc"},
            identity_key="id",
            expected=(),
            expected_total=0,
        )

        assert summary["passed"] is True
        assert summary["coverage"] == "unavailable"
        assert summary["pages_checked"] == 1
        assert len(timings) == 1


def test_fixture_nlp_probability_and_rollup_invariants_reconcile(
    acceptance_catalog: tuple[ArchiveLayout, Engine],
) -> None:
    acceptance = _acceptance_module()
    layout, engine = acceptance_catalog
    summary = build_nlp_annotations(
        layout,
        engine,
        classifier=_FixtureClassifier(),
        tasks=("sentiment", "emotion", "toxicity"),
        batch_size=4,
    )
    assert summary.failed == 0
    engine.dispose()

    readonly = acceptance._readonly_catalog_engine(layout.database)
    try:
        with readonly.connect() as connection:
            checks, report = acceptance._nlp_integrity(
                connection, acceptance._table_names(connection)
            )
    finally:
        readonly.dispose()

    assert all(check.status == "pass" for check in checks)
    assert report["eligible_target_count"] > 0
    for task in ("sentiment", "emotion", "toxicity"):
        task_report = report["tasks"][task]
        assert task_report["completed"] == report["eligible_target_count"]
        assert task_report["rollup_actual_samples"] == task_report["rollup_expected_samples"]
        assert task_report["confidence_mismatches"] == 0
        assert task_report["predicted_label_mismatches"] == 0
        assert task_report["rollup_probability_mismatches"] == 0

    app = create_app(layout.root, security_disabled=True)
    readonly = acceptance._readonly_catalog_engine(layout.database)
    try:
        with TestClient(app) as web_client, readonly.connect() as connection:
            analytics_checks, analytics_report = acceptance._analytics_direct_sql_reconciliation(
                connection, _TestApiClient(web_client)
            )
    finally:
        readonly.dispose()
    assert analytics_checks[0].status == "pass"
    assert analytics_report["endpoints"]["nlp-trends"]["metric_count"] > 5

    writable = create_catalog_engine(layout.database)
    try:
        with writable.begin() as connection:
            connection.execute(
                text(
                    """
                    UPDATE nlp_annotations
                    SET probabilities_json = :probabilities,
                        predicted_label = 'positive',
                        confidence = 0.9,
                        provenance_json = json_set(
                            provenance_json,
                            '$.inference_configuration.binding_kind',
                            'forged_unverified_binding'
                        )
                    WHERE id = (
                        SELECT id FROM nlp_annotations
                        WHERE active = 1 AND task = 'sentiment'
                        ORDER BY id LIMIT 1
                    )
                    """
                ),
                {
                    "probabilities": json.dumps(
                        {
                            "activation": "softmax",
                            "labels": {"negative": -0.1, "neutral": 0.2, "positive": 0.9},
                        }
                    )
                },
            )
            connection.execute(
                text(
                    """
                    UPDATE nlp_annotations
                    SET active = 0
                    WHERE id = (
                        SELECT id FROM nlp_annotations
                        WHERE active = 1 AND task = 'emotion'
                        ORDER BY id LIMIT 1
                    )
                    """
                )
            )
            connection.execute(
                text(
                    """
                    UPDATE jobs
                    SET checkpoint_json = json_set(
                        checkpoint_json,
                        '$.task_inference_configuration_sha256.toxicity',
                        :invalid_configuration_sha256
                    )
                    WHERE id = (
                        SELECT job_id FROM nlp_annotations
                        WHERE active = 1 AND task = 'toxicity'
                        ORDER BY id LIMIT 1
                    )
                    """
                ),
                {"invalid_configuration_sha256": "0" * 64},
            )
            connection.execute(
                text(
                    """
                    UPDATE nlp_hourly_rollups
                    SET configuration_json = json_set(
                        configuration_json,
                        '$.inference_backend.provider',
                        'forged-unverified-backend'
                    )
                    WHERE id = (
                        SELECT id FROM nlp_hourly_rollups
                        WHERE active = 1 AND task = 'sentiment'
                        ORDER BY id LIMIT 1
                    )
                    """
                )
            )
    finally:
        writable.dispose()

    readonly = acceptance._readonly_catalog_engine(layout.database)
    try:
        with readonly.connect() as connection:
            invalid_checks, invalid_report = acceptance._nlp_integrity(
                connection, acceptance._table_names(connection)
            )
    finally:
        readonly.dispose()

    assert (
        next(
            check
            for check in invalid_checks
            if check.key == "nlp.sentiment.complete_and_reconciled"
        ).status
        == "fail"
    )
    assert invalid_report["tasks"]["sentiment"]["invalid_softmax_rows"] == 1
    sentiment_provenance = invalid_report["tasks"]["sentiment"]["inference_provenance"]
    assert sentiment_provenance["invalid_binding_rows"] == 1
    assert sentiment_provenance["invalid_task_mappings"] == 1
    assert invalid_report["tasks"]["sentiment"]["rollup_invalid_inference_provenance_rows"] == 1
    assert (
        next(
            check for check in invalid_checks if check.key == "nlp.emotion.complete_and_reconciled"
        ).status
        == "fail"
    )
    assert invalid_report["tasks"]["emotion"]["missing_targets"] == 1
    toxicity_provenance = invalid_report["tasks"]["toxicity"]["inference_provenance"]
    assert toxicity_provenance["task_mapping_mismatch_rows"] > 0
    assert toxicity_provenance["invalid_task_mappings"] == 1


def test_semantic_source_coverage_rejects_a_stale_audio_window_corpus(
    acceptance_catalog: tuple[ArchiveLayout, Engine],
) -> None:
    acceptance = _acceptance_module()
    layout, engine = acceptance_catalog
    rebuild_search_windows(layout, engine, generation="semantic-source-coverage-before-audio")

    with engine.connect() as connection:
        baseline_check, baseline = acceptance._semantic_source_coverage(connection)
    assert baseline_check.status == "pass", json.dumps(baseline, sort_keys=True)

    blob_sha256 = hashlib.sha256(b"stale semantic audio coverage fixture").hexdigest()
    with Session(engine) as session, session.begin():
        session.add(
            Blob(
                sha256=blob_sha256,
                size_bytes=0,
                media_type="audio/mp4",
                extension=".m4a",
                state="materialized",
            )
        )
        session.flush()
        session.add(
            AttachmentOccurrence(
                id="stale-semantic-audio-occurrence",
                blob_sha256=blob_sha256,
                kind="audio_file",
                ordinal=0,
                media_type="audio/mp4",
                size_bytes=0,
                availability="resolved",
                metadata_json={},
            )
        )
        session.add(
            AudioTranscription(
                id="stale-semantic-audio-transcription",
                blob_sha256=blob_sha256,
                pipeline_version="fixture-audio-v1",
                status="completed",
                model_name="fixture-audio-model",
                model_revision="fixture-audio-revision",
                device="cpu",
                compute_type="int8",
                text_display="Stale semantic audio coverage fixture.",
                text_search="stale semantic audio coverage fixture",
                segment_count=1,
                attempt_count=1,
                index_eligible=True,
                provenance_json={},
                warnings_json={},
            )
        )
        session.flush()
        session.add(
            AudioTranscriptSegment(
                id="stale-semantic-audio-segment",
                transcription_id="stale-semantic-audio-transcription",
                ordinal=0,
                start_ms=0,
                end_ms=1_000,
                text_display="Stale semantic audio coverage fixture.",
                text_search="stale semantic audio coverage fixture",
                words_json={},
            )
        )

    _checkpoint_catalog(engine)
    readonly = acceptance._readonly_catalog_engine(layout.database)
    try:
        with readonly.connect() as connection:
            coverage_check, coverage = acceptance._semantic_source_coverage(connection)
            semantic_checks, semantic_report = acceptance._semantic_integrity(
                layout,
                readonly,
                connection,
                acceptance._table_names(connection),
            )
    finally:
        readonly.dispose()

    assert coverage_check.status == "fail"
    audio = coverage["audio_segments"]
    assert audio["eligible_source_count"] == 1
    assert audio["represented_source_count"] == 0
    assert audio["missing_source_count"] == 1
    assert audio["extra_source_count"] == 0
    assert audio["duplicate_source_id_count"] == 0
    assert audio["duplicate_window_source_pair_count"] == 0
    assert (
        next(
            check
            for check in semantic_checks
            if check.key == "semantic.search_window_source_coverage"
        ).status
        == "fail"
    )
    serialized = json.dumps(semantic_report, sort_keys=True)
    assert "stale-semantic-audio-segment" not in serialized
    assert "Stale semantic audio coverage fixture" not in serialized


def test_semantic_integrity_requires_exact_active_membership_and_lance_id_sets(
    acceptance_catalog: tuple[ArchiveLayout, Engine],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lancedb  # type: ignore[import-untyped]

    acceptance = _acceptance_module()
    layout, engine = acceptance_catalog
    rebuild_search_windows(layout, engine, generation="real-acceptance-semantic-fixture")
    with Session(engine) as session:
        windows = list(session.scalars(select(SearchWindow).order_by(SearchWindow.id)))
    window_ids = [window.id for window in windows]
    assert len(window_ids) > 1

    spec = acceptance.SEMANTIC_MODEL_SPECS[acceptance.AUTO_EMBEDDING_MODEL_ALIAS]
    artifact_fingerprint = f"sha256:{'a' * 64}"
    corpus_revision = "b" * 64
    config_id = acceptance.embedding_config_hash(
        spec,
        corpus_revision=corpus_revision,
        artifact_fingerprint=artifact_fingerprint,
    )
    vector_table_name = f"text_windows_{config_id[:16]}"
    backend, backend_version = acceptance.embedding_backend(spec)
    development_query_texts = [f"private development fixture {index}" for index in range(21)]
    development_feedback_rows = [
        {
            "id": f"development-feedback-{index}",
            "query_sha256": hashlib.sha256(query_text.encode()).hexdigest(),
            "query_text": query_text,
            "result_id": f"window:{window_ids[index % len(window_ids)]}",
            "label": ("relevant" if index < 10 else "not_relevant" if index < 20 else "uncertain"),
            "evaluation_intent": acceptance.DIRECT_CONFLICT_DEVELOPMENT_INTENT,
        }
        for index, query_text in enumerate(development_query_texts)
    ]
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO relevance_feedback(
                    id, query_sha256, query_text, search_mode, result_id, label,
                    evaluation_split, evaluation_intent
                ) VALUES (
                    :id, :query_sha256, :query_text, 'semantic', :result_id, :label,
                    'validation', :evaluation_intent
                )
                """
            ),
            development_feedback_rows,
        )
    development_revision = acceptance.development_feedback_revision(engine)
    held_revision = acceptance.held_out_feedback_revision(engine)
    incumbent_spec = acceptance.SEMANTIC_MODEL_SPECS["multilingual-e5-small"]
    incumbent_artifact_fingerprint = f"sha256:{'e' * 64}"
    incumbent_backend, incumbent_backend_version = acceptance.embedding_backend(incumbent_spec)
    incumbent_config_id = acceptance._legacy_e5_config_hash(
        incumbent_spec,
        backend=incumbent_backend,
        backend_version=incumbent_backend_version,
        corpus_revision=corpus_revision,
        artifact_fingerprint=incumbent_artifact_fingerprint,
    )
    incumbent_vector_table = f"text_windows_{incumbent_config_id[:16]}"
    incumbent_metrics = {name: 0.9 for name in acceptance.SEMANTIC_UPGRADE_METRICS}
    incumbent_evaluation = _legacy_e5_evaluation_fixture(
        acceptance,
        _spot_semantic_evaluation_fixture(
            acceptance,
            config_id=incumbent_config_id,
            vector_table=incumbent_vector_table,
            window_count=len(window_ids),
            corpus_revision=corpus_revision,
            held_revision=held_revision,
            development_revision=development_revision,
            held_metric_value=0.9,
        ),
    )
    incumbent_audit = _semantic_config_audit_fixture(
        acceptance,
        spec=incumbent_spec,
        accepted_evaluation=incumbent_evaluation,
        artifact_fingerprint=incumbent_artifact_fingerprint,
        corpus_revision=corpus_revision,
    )
    incumbent_audit["selected_model"] = acceptance._legacy_e5_model_identity(incumbent_spec)
    incumbent_audit["embedding_implementation"] = acceptance._legacy_e5_embedding_implementation(
        incumbent_spec
    )
    incumbent_audit["model_selection"] = "explicit_config"
    accepted_evaluation = _spot_semantic_evaluation_fixture(
        acceptance,
        config_id=config_id,
        vector_table=vector_table_name,
        window_count=len(window_ids),
        corpus_revision=corpus_revision,
        held_revision=held_revision,
        development_revision=development_revision,
        incumbent_metrics=incumbent_metrics,
    )
    semantic_audit = _semantic_config_audit_fixture(
        acceptance,
        spec=spec,
        accepted_evaluation=accepted_evaluation,
        artifact_fingerprint=artifact_fingerprint,
        corpus_revision=corpus_revision,
    )

    def audit_json(evaluation: Mapping[str, object]) -> str:
        return json.dumps(
            {
                **semantic_audit,
                "accepted_semantic_ir_evaluation": dict(evaluation),
            }
        )

    with engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO semantic_index_configs (
                    id, active, status, backend, backend_version, model_name,
                    model_revision, dimensions, distance_metric, document_prefix,
                    query_prefix, vector_table, window_version, window_count, audit_json
                ) VALUES (
                    :id, 0, 'ready', :backend, :backend_version, :model_name,
                    :model_revision, :dimensions, 'cosine', :document_prefix,
                    :query_prefix, :vector_table, :window_version, :window_count, :audit_json
                )
                """
            ),
            {
                "id": incumbent_config_id,
                "backend": incumbent_backend,
                "backend_version": incumbent_backend_version,
                "model_name": incumbent_spec.model_name,
                "model_revision": incumbent_spec.model_revision,
                "dimensions": incumbent_spec.dimensions,
                "document_prefix": incumbent_spec.document_prefix,
                "query_prefix": incumbent_spec.query_prefix,
                "vector_table": incumbent_vector_table,
                "window_version": acceptance.SEARCH_WINDOW_VERSION,
                "window_count": len(window_ids),
                "audit_json": json.dumps(incumbent_audit),
            },
        )
        connection.execute(
            text(
                """
                INSERT INTO semantic_index_configs (
                    id, active, status, backend, backend_version, model_name,
                    model_revision, dimensions, distance_metric, document_prefix,
                    query_prefix, vector_table, window_version, window_count, audit_json
                ) VALUES (
                    :id, 1, 'ready', :backend, :backend_version, :model_name,
                    :model_revision, :dimensions, 'cosine', :document_prefix,
                    :query_prefix, :vector_table, :window_version, :window_count, :audit_json
                )
                """
            ),
            {
                "id": config_id,
                "backend": backend,
                "backend_version": backend_version,
                "model_name": spec.model_name,
                "model_revision": spec.model_revision,
                "dimensions": spec.dimensions,
                "document_prefix": spec.document_prefix,
                "query_prefix": spec.query_prefix,
                "vector_table": vector_table_name,
                "window_version": acceptance.SEARCH_WINDOW_VERSION,
                "window_count": len(window_ids),
                "audit_json": audit_json(accepted_evaluation),
            },
        )
        connection.execute(
            text(
                """
                INSERT INTO semantic_window_embeddings (config_id, window_id)
                VALUES (:config_id, :window_id)
                """
            ),
            [{"config_id": config_id, "window_id": window_id} for window_id in window_ids],
        )
        # A complete legacy marker must not hide a damaged authoritative membership relation.
        connection.execute(
            text("UPDATE search_windows SET embedding_config_hash = :config_id"),
            {"config_id": config_id},
        )

    vector_database = lancedb.connect(str(layout.vectors))
    unit_vector = [1.0, *([0.0] * (spec.dimensions - 1))]
    vector_table = vector_database.create_table(
        vector_table_name,
        data=[_vector_row(window, unit_vector) for window in windows],
    )
    vector_database.create_table(
        "acceptance_decoy_table",
        data=[{"id": "decoy", "vector": [1.0]}],
    )
    real_lance_table_names = acceptance.lance_table_names
    lance_inventory_calls = 0

    def paginated_lance_table_names(database: object) -> set[str]:
        nonlocal lance_inventory_calls
        lance_inventory_calls += 1
        return real_lance_table_names(database, page_size=1)

    monkeypatch.setattr(acceptance, "lance_table_names", paginated_lance_table_names)
    artifact_fingerprint_calls = 0
    if spec.provider == acceptance.FASTEMBED_PROVIDER:
        pinned_fastembed = acceptance.PINNED_FASTEMBED_ARTIFACTS[spec.alias]
        expected_artifact = (
            layout.models / "fastembed",
            pinned_fastembed.repository,
            pinned_fastembed.revision,
        )
    else:
        assert spec.provider == acceptance.TRANSFORMERS_PROVIDER
        expected_artifact = (
            layout.models / "transformers",
            spec.model_name,
            spec.model_revision,
        )

    def pinned_artifact(
        cache_dir: Path,
        *,
        model_name: str,
        model_revision: str,
    ) -> SimpleNamespace:
        nonlocal artifact_fingerprint_calls
        artifact_fingerprint_calls += 1
        assert (cache_dir, model_name, model_revision) == expected_artifact
        return SimpleNamespace(fingerprint_sha256="a" * 64)

    monkeypatch.setattr(acceptance, "pinned_transformers_artifact", pinned_artifact)

    def active_check() -> tuple[Any, dict[str, object]]:
        _checkpoint_catalog(engine)
        readonly = acceptance._readonly_catalog_engine(layout.database)
        try:
            with readonly.connect() as connection:
                checks, report = acceptance._semantic_integrity(
                    layout,
                    readonly,
                    connection,
                    acceptance._table_names(connection),
                )
        finally:
            readonly.dispose()
        return next(
            check for check in checks if check.key == "semantic.active_ready_config"
        ), report

    valid_check, valid_report = active_check()
    assert valid_check.status == "pass", json.dumps(valid_report, sort_keys=True)
    assert lance_inventory_calls == 1
    assert artifact_fingerprint_calls == 1
    assert valid_report["exact_id_sets_reconcile"] is True
    assert valid_report["membership_windows"] == len(window_ids)
    assert valid_report["held_out_feedback_used"] is True
    assert valid_report["held_out_feedback_passed"] is True
    assert valid_report["held_out_feedback_revision_current"] is True
    assert valid_report["held_out_feedback_revision_row_count"] == 0
    development_report = valid_report["direct_conflict_development_calibration"]
    assert development_report["evidence_kind"] == (
        acceptance.DIRECT_CONFLICT_DEVELOPMENT_EVIDENCE_KIND
    )
    assert development_report["held_out"] is False
    assert development_report["passed"] is True
    assert development_report["feedback_revision"] == {
        "row_count": 21,
        "current": True,
        "stable_during_evaluation": True,
    }
    assert development_report["relevant_count"] == 10
    assert development_report["not_relevant_count"] == 10
    assert development_report["uncertain_count"] == 1
    assert development_report["fold_count"] == 4
    valid_serialized = json.dumps(valid_report, sort_keys=True)
    assert development_query_texts[0] not in valid_serialized
    assert development_feedback_rows[0]["query_sha256"] not in valid_serialized
    assert development_feedback_rows[0]["result_id"] not in valid_serialized
    assert str(development_revision["sha256"]) not in valid_serialized

    with engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO relevance_feedback(
                    id, query_sha256, query_text, search_mode, result_id, label,
                    evaluation_split
                ) VALUES (
                    'revision-fixture', :query_sha256, :query_text, 'semantic',
                    :result_id, 'not_relevant', 'test'
                )
                """
            ),
            {
                "query_sha256": hashlib.sha256(b"private revision fixture").hexdigest(),
                "query_text": "private revision fixture",
                "result_id": f"window:{window_ids[0]}",
            },
        )
    stale_feedback_check, stale_feedback_report = active_check()
    assert stale_feedback_check.status == "fail"
    assert stale_feedback_report["held_out_feedback_revision_current"] is False
    stale_serialized = json.dumps(stale_feedback_report, sort_keys=True)
    assert "private revision fixture" not in stale_serialized
    assert window_ids[0] not in stale_serialized
    with engine.begin() as connection:
        connection.execute(text("DELETE FROM relevance_feedback WHERE id = 'revision-fixture'"))

    with engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO relevance_feedback(
                    id, query_sha256, query_text, search_mode, result_id, label,
                    evaluation_split, evaluation_intent
                ) VALUES (
                    'development-revision-fixture', :query_sha256, :query_text, 'semantic',
                    :result_id, 'not_relevant', 'validation', :evaluation_intent
                )
                """
            ),
            {
                "query_sha256": hashlib.sha256(b"private development revision").hexdigest(),
                "query_text": "private development revision",
                "result_id": f"window:{window_ids[0]}",
                "evaluation_intent": acceptance.DIRECT_CONFLICT_DEVELOPMENT_INTENT,
            },
        )
    stale_development_check, stale_development_report = active_check()
    assert stale_development_check.status == "fail"
    stale_development = stale_development_report["direct_conflict_development_calibration"]
    assert stale_development["passed"] is False
    assert stale_development["feedback_revision"]["current"] is False
    stale_development_serialized = json.dumps(stale_development_report, sort_keys=True)
    assert "private development revision" not in stale_development_serialized
    assert hashlib.sha256(b"private development revision").hexdigest() not in (
        stale_development_serialized
    )
    with engine.begin() as connection:
        connection.execute(
            text("DELETE FROM relevance_feedback WHERE id = 'development-revision-fixture'")
        )

    evaluation_without_development_gate = json.loads(json.dumps(accepted_evaluation))
    evaluation_without_development_gate["production_acceptance_passed"] = False
    evaluation_without_development_gate.pop("direct_conflict_development_calibration")
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                UPDATE semantic_index_configs
                SET audit_json = :audit_json
                WHERE id = :config_id
                """
            ),
            {
                "audit_json": audit_json(evaluation_without_development_gate),
                "config_id": config_id,
            },
        )
    no_development_check, no_development_report = active_check()
    assert no_development_check.status == "fail"
    assert no_development_report["direct_conflict_development_calibration"]["passed"] is False
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                UPDATE semantic_index_configs
                SET audit_json = :audit_json
                WHERE id = :config_id
                """
            ),
            {
                "audit_json": audit_json(accepted_evaluation),
                "config_id": config_id,
            },
        )

    evaluation_with_failed_development_gate = json.loads(json.dumps(accepted_evaluation))
    evaluation_with_failed_development_gate["direct_conflict_development_calibration"][
        "literal_query_regression"
    ]["metrics"]["recall_at_10"] = 0.59
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                UPDATE semantic_index_configs
                SET audit_json = :audit_json
                WHERE id = :config_id
                """
            ),
            {
                "audit_json": audit_json(evaluation_with_failed_development_gate),
                "config_id": config_id,
            },
        )
    failed_development_check, failed_development_report = active_check()
    assert failed_development_check.status == "fail"
    assert failed_development_report["direct_conflict_development_calibration"]["passed"] is False
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                UPDATE semantic_index_configs
                SET audit_json = :audit_json
                WHERE id = :config_id
                """
            ),
            {
                "audit_json": audit_json(accepted_evaluation),
                "config_id": config_id,
            },
        )

    evaluation_without_held_out = json.loads(json.dumps(accepted_evaluation))
    evaluation_without_held_out["held_out_feedback"] = {
        "status": "not_available",
        "passed": None,
        "used_for_activation": False,
        "metrics": {"query_count": 0},
        "exclusions": {
            "without_relevant_window_judgment": 0,
            "relevant_target_not_in_candidate": 0,
        },
    }
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                UPDATE semantic_index_configs
                SET audit_json = :audit_json
                WHERE id = :config_id
                """
            ),
            {
                "audit_json": audit_json(evaluation_without_held_out),
                "config_id": config_id,
            },
        )
    no_held_out_check, no_held_out_report = active_check()
    assert no_held_out_check.status == "fail"
    assert no_held_out_report["held_out_feedback_used"] is False
    assert no_held_out_report["held_out_feedback_passed"] is False
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                UPDATE semantic_index_configs
                SET audit_json = :audit_json
                WHERE id = :config_id
                """
            ),
            {
                "audit_json": audit_json(accepted_evaluation),
                "config_id": config_id,
            },
        )

    with engine.begin() as connection:
        connection.execute(
            text(
                """
                DELETE FROM semantic_window_embeddings
                WHERE config_id = :config_id AND window_id = :window_id
                """
            ),
            {"config_id": config_id, "window_id": window_ids[-1]},
        )
    missing_check, missing_report = active_check()
    assert missing_check.status == "fail"
    assert missing_report["legacy_marker_windows"] == len(window_ids)
    assert missing_report["membership_windows"] == len(window_ids) - 1

    with engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO semantic_window_embeddings (config_id, window_id)
                VALUES (:config_id, :window_id)
                """
            ),
            {"config_id": config_id, "window_id": window_ids[-1]},
        )
    vector_table.delete(f"id = '{window_ids[-1]}'")
    duplicate = vector_table.search().where(f"id = '{window_ids[0]}'").limit(1).to_list()[0]
    duplicate.pop("_distance", None)
    vector_table.add([duplicate])
    mismatched_check, mismatched_report = active_check()
    assert mismatched_check.status == "fail"
    assert mismatched_report["lance_windows"] == len(window_ids)
    assert mismatched_report["lance_duplicate_rows"] == 1
    assert mismatched_report["exact_id_sets_reconcile"] is False


def test_semantic_hybrid_and_conflict_user_stories_require_live_results_and_membership(
    acceptance_catalog: tuple[ArchiveLayout, Engine],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acceptance = _acceptance_module()
    layout, engine = acceptance_catalog
    rebuild_search_windows(layout, engine, generation="real-acceptance-search-story-fixture")
    with engine.connect() as connection:
        window_id = str(
            connection.execute(
                text(
                    """
                    SELECT id FROM search_windows
                    WHERE entity_type = 'message_window' AND first_message_id IS NOT NULL
                    ORDER BY id LIMIT 1
                    """
                )
            ).scalar_one()
        )
    spec = acceptance.SEMANTIC_MODEL_SPECS[acceptance.AUTO_EMBEDDING_MODEL_ALIAS]
    artifact_fingerprint = f"sha256:{'f' * 64}"
    corpus_revision = "c" * 64
    backend, backend_version = acceptance.embedding_backend(spec)
    config_id = acceptance.embedding_config_hash(
        spec,
        corpus_revision=corpus_revision,
        artifact_fingerprint=artifact_fingerprint,
    )
    vector_table = f"text_windows_{config_id[:16]}"
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO semantic_index_configs (
                    id, active, status, backend, backend_version, model_name,
                    model_revision, dimensions, distance_metric, document_prefix,
                    query_prefix, vector_table, window_version, window_count, audit_json
                ) VALUES (
                    :id, 1, 'ready', :backend, :backend_version, :model_name,
                    :model_revision, :dimensions, 'cosine', :document_prefix,
                    :query_prefix, :vector_table, :window_version, 1, '{}'
                )
                """
            ),
            {
                "id": config_id,
                "backend": backend,
                "backend_version": backend_version,
                "model_name": spec.model_name,
                "model_revision": spec.model_revision,
                "dimensions": spec.dimensions,
                "document_prefix": spec.document_prefix,
                "query_prefix": spec.query_prefix,
                "vector_table": vector_table,
                "window_version": acceptance.SEARCH_WINDOW_VERSION,
            },
        )
        connection.execute(
            text(
                """
                INSERT INTO semantic_window_embeddings (config_id, window_id)
                VALUES (:config_id, :window_id)
                """
            ),
            {"config_id": config_id, "window_id": window_id},
        )

    _checkpoint_catalog(engine)
    readonly = acceptance._readonly_catalog_engine(layout.database)
    try:
        with readonly.connect() as connection:
            jump = acceptance._expected_window_jump(connection, window_id)
    finally:
        readonly.dispose()
    assert jump is not None

    reranker = {
        "model_name": acceptance.DEFAULT_RERANKER_SPEC.model_name,
        "model_revision": acceptance.DEFAULT_RERANKER_SPEC.model_revision,
        "device": acceptance.DEFAULT_RERANKER_SPEC.device,
        "score_kind": acceptance.DEFAULT_RERANKER_SPEC.score_kind,
        "max_candidates": acceptance.DEFAULT_RERANKER_SPEC.max_candidates,
        "batch_size": acceptance.DEFAULT_RERANKER_SPEC.batch_size,
        "max_sequence_tokens": acceptance.DEFAULT_RERANKER_SPEC.max_sequence_tokens,
        "status": "ready",
    }
    direct_conflict_ranker = {
        "version": acceptance.DIRECT_CONFLICT_RANKER_VERSION,
        "parameters": asdict(acceptance.DEFAULT_DIRECT_CONFLICT_PARAMETERS),
        "counts": {
            "reviewed_feedback_rows": 20,
            "binary_training_rows": 20,
            "relevant_rows": 10,
            "not_relevant_rows": 10,
            "uncertain_rows": 0,
            "conversation_groups": 10,
        },
        "metrics": {
            "fold_count": acceptance.DIRECT_CONFLICT_DEVELOPMENT_FOLD_COUNT,
            **{
                name: {"mean": 0.9, "minimum": 0.8}
                for name in acceptance.DIRECT_CONFLICT_PUBLIC_METRICS
            },
        },
        "score_kind": acceptance.DIRECT_CONFLICT_SCORE_KIND,
        "ranking_feature_only": True,
    }
    development_revision = {
        "version": "fixture-development-feedback-v1",
        "sha256": "d" * 64,
        "row_count": 20,
    }
    monkeypatch.setattr(
        acceptance,
        "development_feedback_revision",
        lambda _engine: dict(development_revision),
    )
    development_calibration = {
        "required_for_production_acceptance": True,
        "status": "passed",
        "passed": True,
        "ranking_pipeline": acceptance.DIRECT_CONFLICT_RANKING_PIPELINE_VERSION,
        "feedback_revision": development_revision,
        "feedback_revision_stable_during_evaluation": True,
        "model_metadata": {
            "classifier": acceptance.DIRECT_CONFLICT_CLASSIFIER,
            "ranker_version": acceptance.DIRECT_CONFLICT_RANKER_VERSION,
            "score_kind": acceptance.DIRECT_CONFLICT_SCORE_KIND,
            "parameters": asdict(acceptance.DEFAULT_DIRECT_CONFLICT_PARAMETERS),
        },
        "reviewed_judgment_count": 20,
        "binary_judgment_count": 20,
        "relevant_count": 10,
        "not_relevant_count": 10,
        "uncertain_count": 0,
        "conversation_group_count": 10,
        "fold_count": acceptance.DIRECT_CONFLICT_DEVELOPMENT_FOLD_COUNT,
        "grouped_cross_validation": {
            "minimum_metrics": {
                name: direct_conflict_ranker["metrics"][name]["minimum"]
                for name in acceptance.DIRECT_CONFLICT_PUBLIC_METRICS
            },
            "mean_metrics": {
                name: direct_conflict_ranker["metrics"][name]["mean"]
                for name in acceptance.DIRECT_CONFLICT_PUBLIC_METRICS
            },
        },
    }
    accepted_evaluation = {
        "candidate": {
            "config_hash": config_id,
            "vector_table": vector_table,
            "window_count": 1,
            "distance_metric": "cosine",
            "corpus_revision": corpus_revision,
        },
        "production_acceptance_passed": True,
        "direct_conflict_development_calibration": development_calibration,
    }
    semantic_audit = _semantic_config_audit_fixture(
        acceptance,
        spec=spec,
        accepted_evaluation=accepted_evaluation,
        artifact_fingerprint=artifact_fingerprint,
        corpus_revision=corpus_revision,
    )
    with engine.begin() as connection:
        connection.execute(
            text("UPDATE semantic_index_configs SET audit_json = :audit_json WHERE id = :id"),
            {
                "id": config_id,
                "audit_json": json.dumps(semantic_audit),
            },
        )
    _checkpoint_catalog(engine)
    retrieval_config = {
        "id": config_id,
        "model_name": spec.model_name,
        "model_revision": spec.model_revision,
        "window_version": acceptance.SEARCH_WINDOW_VERSION,
        "artifact_fingerprint": artifact_fingerprint,
        "embedding_implementation": acceptance.embedding_implementation_identity(spec),
        "reranker": reranker,
        "ranking_pipeline": acceptance.DIRECT_CONFLICT_RANKING_PIPELINE_VERSION,
        "direct_conflict_ranker": direct_conflict_ranker,
    }
    base_item = {
        "id": f"window:{window_id}",
        "window_id": window_id,
        "entity_type": "message_window",
        "source_id": "fixture-source",
        "rank": 1,
        "score": 0.8,
        "deletion_state": "retained",
        "provenance": {"source_id": "fixture-source"},
        "jump": jump,
    }
    scores = {label: 0.1 for label in acceptance.CONFLICT_LABELS}
    scores["direct conflict"] = 0.8
    conflict_scoring = {
        "scores": scores,
        "score_kind": acceptance.DEFAULT_CONFLICT_SPEC.score_kind,
        "target_label": "direct conflict",
        "predicted_label": "direct conflict",
        "model_name": acceptance.DEFAULT_CONFLICT_SPEC.model_name,
        "model_revision": acceptance.DEFAULT_CONFLICT_SPEC.model_revision,
        "quantization": acceptance.DEFAULT_CONFLICT_SPEC.quantization,
        "ranking_feature_only": True,
        "aggregation_gate_bypassed": False,
        "hypotheses": {
            label: f"fixture hypothesis {index}"
            for index, label in enumerate(acceptance.CONFLICT_LABELS)
        },
        "warnings": list(acceptance.CONFLICT_WARNINGS),
        "provenance": {
            "generated_locally": True,
            "pipeline_version": "conflict-nli-ranking-v1",
            "model_revision": acceptance.DEFAULT_CONFLICT_SPEC.model_revision,
            "quantization": acceptance.DEFAULT_CONFLICT_SPEC.quantization,
        },
    }

    class SearchStoryClient:
        def __init__(
            self,
            *,
            empty: bool = False,
            wrong_literal_ranker_provenance: bool = False,
        ) -> None:
            self.empty = empty
            self.wrong_literal_ranker_provenance = wrong_literal_ranker_provenance
            self.conflict_timeouts: list[float] = []

        def get_json(
            self,
            path: str,
            parameters: Mapping[str, object] | None = None,
            *,
            timeout: float = 120.0,
        ) -> tuple[dict[str, Any], float]:
            assert parameters is not None
            requested_query = str(parameters["q"])
            if path == "/api/v1/search":
                mode = str(parameters["mode"])
                items = []
                if not self.empty:
                    literal_direct_conflict = (
                        mode == "semantic"
                        and acceptance.is_direct_conflict_trigger(requested_query)
                    )
                    if literal_direct_conflict:
                        item_ranker = json.loads(json.dumps(direct_conflict_ranker))
                        if self.wrong_literal_ranker_provenance:
                            item_ranker["version"] = "stale-private-ranker"
                        ranking_provenance = {
                            "pipeline": acceptance.DIRECT_CONFLICT_LITERAL_PIPELINE,
                            "ranking_pipeline": (
                                acceptance.DIRECT_CONFLICT_RANKING_PIPELINE_VERSION
                            ),
                            "trigger": acceptance.DIRECT_CONFLICT_LITERAL_TRIGGER,
                            "ranker": item_ranker,
                        }
                    elif mode == "semantic":
                        ranking_provenance = {
                            "pipeline": "cosine_retrieval_then_local_cross_encoder",
                            "reranker": reranker,
                        }
                    else:
                        ranking_provenance = {
                            "pipeline": "reciprocal_rank_fusion",
                            "semantic_reranker": reranker,
                        }
                    items = [
                        {
                            **base_item,
                            "score_kind": (
                                acceptance.DIRECT_CONFLICT_SCORE_KIND
                                if literal_direct_conflict
                                else acceptance.DEFAULT_RERANKER_SPEC.score_kind
                                if mode == "semantic"
                                else "rrf"
                            ),
                            "ranking_provenance": ranking_provenance,
                        }
                    ]
                return (
                    {
                        "query": requested_query,
                        "mode": mode,
                        "items": items,
                        "page": {
                            "returned": len(items),
                            "limit": 50,
                            "result_cap": 1_000,
                            "candidate_count": len(items),
                        },
                        "index_config": retrieval_config,
                    },
                    1.0,
                )
            assert path == "/api/v1/enrichment/conflict/search"
            self.conflict_timeouts.append(timeout)
            items = []
            if not self.empty:
                items = [
                    {
                        **base_item,
                        "score_kind": acceptance.DEFAULT_CONFLICT_SPEC.score_kind,
                        "conflict_scoring": conflict_scoring,
                    }
                ]
            return (
                {
                    "query": requested_query,
                    "target_label": "direct conflict",
                    "items": items,
                    "page": {
                        "returned": len(items),
                        "limit": 25,
                        "candidate_count": len(items),
                    },
                    "retrieval_config": retrieval_config,
                    "conflict_model": {
                        "model_name": acceptance.DEFAULT_CONFLICT_SPEC.model_name,
                        "model_revision": acceptance.DEFAULT_CONFLICT_SPEC.model_revision,
                        "device": acceptance.DEFAULT_CONFLICT_SPEC.device,
                        "quantization": acceptance.DEFAULT_CONFLICT_SPEC.quantization,
                        "score_kind": acceptance.DEFAULT_CONFLICT_SPEC.score_kind,
                        "max_candidates": acceptance.DEFAULT_CONFLICT_SPEC.max_candidates,
                        "batch_size": acceptance.DEFAULT_CONFLICT_SPEC.batch_size,
                        "max_sequence_tokens": (
                            acceptance.DEFAULT_CONFLICT_SPEC.max_sequence_tokens
                        ),
                        "status": "loaded",
                        "labels": list(acceptance.CONFLICT_LABELS),
                        "warnings": list(acceptance.CONFLICT_WARNINGS),
                        "ranking_feature_only": True,
                        "aggregation_gate_bypassed": False,
                    },
                    "aggregation": {
                        "aggregation_enabled": False,
                        "relationship_ranking_enabled": False,
                        "sample_count": 0,
                        "label_count": 0,
                        "minimum_label_count": acceptance.MIN_CONFLICT_LABELS,
                        "thresholds": {
                            "direct_conflict_precision": (acceptance.MIN_DIRECT_CONFLICT_PRECISION),
                            "macro_f1": acceptance.MIN_MACRO_F1,
                        },
                        "status": "disabled_until_calibrated",
                        "reason": "fixture has not met the held-out calibration gate",
                    },
                },
                1.0,
            )

    client = SearchStoryClient()
    readonly = acceptance._readonly_catalog_engine(layout.database)
    try:
        with readonly.connect() as connection:
            checks, report = acceptance._search_user_story_acceptance(
                connection, client, query="argument"
            )
    finally:
        readonly.dispose()
    assert all(check.status == "pass" for check in checks)
    assert client.conflict_timeouts == [600.0]
    serialized = json.dumps(report)
    assert "argument" not in serialized
    assert window_id not in serialized
    assert str(jump["href"]) not in serialized

    general_client = SearchStoryClient()
    readonly = acceptance._readonly_catalog_engine(layout.database)
    try:
        with readonly.connect() as connection:
            general_checks, _general_report = acceptance._search_user_story_acceptance(
                connection, general_client, query="argument!"
            )
    finally:
        readonly.dispose()
    assert all(check.status == "pass" for check in general_checks)

    wrong_ranker_client = SearchStoryClient(wrong_literal_ranker_provenance=True)
    readonly = acceptance._readonly_catalog_engine(layout.database)
    try:
        with readonly.connect() as connection:
            wrong_ranker_checks, wrong_ranker_report = acceptance._search_user_story_acceptance(
                connection, wrong_ranker_client, query="argument"
            )
    finally:
        readonly.dispose()
    assert wrong_ranker_checks[0].status == "fail"
    assert wrong_ranker_checks[1].status == "pass"
    assert wrong_ranker_report["search"]["semantic"]["contract_valid"] is False

    monkeypatch.setattr(
        acceptance,
        "development_feedback_revision",
        lambda _engine: {**development_revision, "sha256": "e" * 64},
    )
    readonly = acceptance._readonly_catalog_engine(layout.database)
    try:
        with readonly.connect() as connection:
            stale_feedback_checks, stale_feedback_report = acceptance._search_user_story_acceptance(
                connection, SearchStoryClient(), query="argument"
            )
    finally:
        readonly.dispose()
    assert stale_feedback_checks[0].status == "fail"
    assert stale_feedback_checks[1].status == "pass"
    assert stale_feedback_report["search"]["semantic"]["contract_valid"] is False
    monkeypatch.setattr(
        acceptance,
        "development_feedback_revision",
        lambda _engine: dict(development_revision),
    )

    empty_client = SearchStoryClient(empty=True)
    readonly = acceptance._readonly_catalog_engine(layout.database)
    try:
        with readonly.connect() as connection:
            empty_checks, _empty_report = acceptance._search_user_story_acceptance(
                connection, empty_client, query="argument"
            )
    finally:
        readonly.dispose()
    assert all(check.status == "fail" for check in empty_checks)

    with engine.begin() as connection:
        connection.execute(
            text(
                """
                DELETE FROM semantic_window_embeddings
                WHERE config_id = :config_id AND window_id = :window_id
                """
            ),
            {"config_id": config_id, "window_id": window_id},
        )
    _checkpoint_catalog(engine)
    readonly = acceptance._readonly_catalog_engine(layout.database)
    try:
        with readonly.connect() as connection:
            stale_checks, stale_report = acceptance._search_user_story_acceptance(
                connection, SearchStoryClient(), query="argument"
            )
    finally:
        readonly.dispose()
    assert all(check.status == "fail" for check in stale_checks)
    assert stale_report["search"]["semantic"]["active_member_windows"] == 0
    assert stale_report["conflict"]["active_member_windows"] == 0


def _spot_nlp_series_fixture(
    acceptance: ModuleType,
    *,
    rows_per_task: int,
) -> dict[str, list[dict[str, object]]]:
    pairs = [
        (f"{2000 + period:04d}", direction)
        for period in range(13)
        for direction in acceptance.PRODUCTION_SPOT_CHECK_NLP_DIRECTIONS
    ]
    assert rows_per_task <= len(pairs)
    return {
        task: [
            {
                "period": period,
                "direction": direction,
                "sample_count": 1,
                "mean_confidence": 0.8,
                "score": None if task == "emotion" else 0.25,
                "score_definition": "fixture aggregate score",
                "dominant_label": "fixture-label",
                "predicted_label_counts": {"fixture-label": 1},
                "mean_probabilities": {"fixture-label": 1.0},
            }
            for period, direction in pairs[:rows_per_task]
        ]
        for task in acceptance.PRODUCTION_SPOT_CHECK_NLP_TASKS
    }


class _SpotAnalyticsFixtureClient:
    def __init__(self, nlp_payload: Mapping[str, object]) -> None:
        self.nlp_payload = nlp_payload
        self.calls: list[str] = []

    def get_json(
        self,
        path: str,
        parameters: Mapping[str, object] | None = None,
        *,
        timeout: float = 120.0,
    ) -> tuple[dict[str, Any], float]:
        del timeout
        self.calls.append(path)
        if path == "/api/v1/analytics/coverage":
            assert parameters is None
            return {"items": [{"key": "fixture", "status": "available"}]}, 1.0
        assert path == "/api/v1/analytics/nlp-trends"
        assert parameters == {
            "bucket": "year",
            "timezone": "UTC",
            "direction": "all",
            "limit": 12,
        }
        return dict(self.nlp_payload), 1.0


def test_production_spot_analytics_accepts_bounded_observed_like_nlp_series() -> None:
    acceptance = _acceptance_module()
    series = _spot_nlp_series_fixture(acceptance, rows_per_task=57)
    client = _SpotAnalyticsFixtureClient({"series": series, "page": {"limit": 12, "returned": 10}})

    checks, report = acceptance._spot_analytics(client)

    assert checks[0].status == "pass"
    assert report["nlp_series_row_count"] == 171
    assert report["maximum_nlp_series_rows"] == 216
    assert report["maximum_request_count"] == 2
    assert client.calls == [
        "/api/v1/analytics/coverage",
        "/api/v1/analytics/nlp-trends",
    ]


@pytest.mark.parametrize("corruption", ("too_many", "malformed", "duplicate"))
def test_production_spot_analytics_rejects_invalid_nlp_series(corruption: str) -> None:
    acceptance = _acceptance_module()
    rows_per_task = 73 if corruption == "too_many" else 57
    series = _spot_nlp_series_fixture(acceptance, rows_per_task=rows_per_task)
    if corruption == "malformed":
        series["sentiment"][0]["sample_count"] = -1
    elif corruption == "duplicate":
        series["sentiment"][1] = dict(series["sentiment"][0])
    client = _SpotAnalyticsFixtureClient({"series": series, "page": {"limit": 12, "returned": 12}})

    checks, report = acceptance._spot_analytics(client)

    assert checks[0].status == "fail"
    if corruption == "too_many":
        assert report["nlp_series_row_count"] == 219
    assert report["maximum_nlp_series_rows"] == 216
    assert client.calls == [
        "/api/v1/analytics/coverage",
        "/api/v1/analytics/nlp-trends",
    ]


class _FixedTimingClient:
    def get_json(
        self,
        path: str,
        parameters: Mapping[str, object] | None = None,
        *,
        timeout: float = 120.0,
    ) -> tuple[dict[str, Any], float]:
        del path, parameters, timeout
        return {}, 1.0


def test_warmed_performance_samples_exclude_reconciliation_outliers(
    acceptance_catalog: tuple[ArchiveLayout, Engine],
) -> None:
    acceptance = _acceptance_module()
    layout, _engine = acceptance_catalog
    readonly = acceptance._readonly_catalog_engine(layout.database)
    try:
        with readonly.connect() as connection:
            checks, report = acceptance._benchmark_api(
                _FixedTimingClient(),
                connection,
                repeats=3,
                query="private-query-not-emitted",
                inherited_timings={"list_api": [9_999.0], "date_jump_api": [8_888.0]},
            )
    finally:
        readonly.dispose()

    assert all(check.status == "pass" for check in checks)
    assert report["list_api"]["p95_ms"] == 1.0
    assert report["date_jump_api"]["p95_ms"] == 1.0
    assert report["reconciliation_observations"]["list_api"]["p95_ms"] == 9_999.0


def test_analytics_acceptance_probe_does_not_create_a_lock_file(
    acceptance_catalog: tuple[ArchiveLayout, Engine],
) -> None:
    acceptance = _acceptance_module()
    layout, _engine = acceptance_catalog
    lock_path = layout.catalog / "analytics-cache.lock"
    existed_before = lock_path.exists()
    readonly = acceptance._readonly_catalog_engine(layout.database)
    try:
        checks, report = acceptance._analytics_cache_integrity(layout, readonly)
    finally:
        readonly.dispose()

    assert lock_path.exists() is existed_before
    assert checks[0].status == "fail"
    assert report["current_entry_count"] == 0


class _CorruptAnalyticsClient:
    def __init__(self, wrapped: _TestApiClient) -> None:
        self.wrapped = wrapped

    def get_json(
        self,
        path: str,
        parameters: Mapping[str, object] | None = None,
        *,
        timeout: float = 120.0,
    ) -> tuple[dict[str, Any], float]:
        payload, elapsed = self.wrapped.get_json(path, parameters, timeout=timeout)
        if path == "/api/v1/analytics/overview":
            payload = json.loads(json.dumps(payload))
            payload["messages"]["total"] += 1
        return payload, elapsed


class _DuplicateKeyAnalyticsClient:
    def __init__(self, wrapped: _TestApiClient) -> None:
        self.wrapped = wrapped

    def get_json(
        self,
        path: str,
        parameters: Mapping[str, object] | None = None,
        *,
        timeout: float = 120.0,
    ) -> tuple[dict[str, Any], float]:
        payload, elapsed = self.wrapped.get_json(path, parameters, timeout=timeout)
        payload = json.loads(json.dumps(payload))
        if path == "/api/v1/analytics/overview":
            row = {"domain": "private-duplicate-domain", "occurrences": 1}
            payload["link_domains"] = [row, dict(row)]
        elif path == "/api/v1/analytics/messages":
            row = {
                "period": "private-duplicate-period",
                "messages": 1,
                "sent": 1,
                "received": 0,
                "active_conversations": 1,
            }
            payload["series"] = [row, dict(row)]
        elif path == "/api/v1/analytics/attachments":
            row = {
                "type": "private-duplicate-type",
                "occurrences": 1,
                "unique_files": 1,
                "occurrence_bytes": 1,
            }
            payload["types"] = [row, dict(row)]
        elif path == "/api/v1/analytics/coverage":
            row = {
                "key": "private-duplicate-coverage",
                "status": "available",
                "raw_count": 1,
                "logical_count": 1,
                "absence_is_not_zero_history": False,
            }
            payload["items"] = [row, dict(row)]
        elif path == "/api/v1/analytics/nlp-trends":
            row = {
                "period": "private-duplicate-nlp",
                "direction": "all",
                "sample_count": 1,
                "mean_confidence": 1.0,
                "score": 0.0,
                "predicted_label_counts": {},
                "mean_probabilities": {},
            }
            payload["series"] = {"sentiment": [row, dict(row)]}
        return payload, elapsed


def test_analytics_page_collector_exhausts_cursors_and_merges_page_local_aggregates() -> None:
    acceptance = _acceptance_module()

    class FakePagedClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []

        def get_json(
            self,
            path: str,
            parameters: Mapping[str, object] | None = None,
            *,
            timeout: float = 120.0,
        ) -> tuple[dict[str, Any], float]:
            del timeout
            query = dict(parameters or {})
            self.calls.append((path, query))
            second = query.get("cursor") == "page-two"
            if path.endswith("/nlp-trends"):
                count = 2 if second else 1
                period = "2023-02" if second else "2023-01"
                probabilities = (
                    {"positive": 0.0, "negative": 1.0}
                    if second
                    else {
                        "positive": 1.0,
                        "negative": 0.0,
                    }
                )
                label = "negative" if second else "positive"
                point = {
                    "direction": "owner",
                    "sample_count": count,
                    "mean_confidence": 0.6 if second else 0.8,
                    "score": -1.0 if second else 1.0,
                    "score_definition": "positive minus negative",
                    "dominant_label": label,
                    "predicted_label_counts": {label: count},
                    "mean_probabilities": probabilities,
                }
                return (
                    {
                        "bucket": "month",
                        "timezone": "UTC",
                        "direction": "owner",
                        "source_sample_count": 3,
                        "page_sample_count": count,
                        "models": {},
                        "series": {"sentiment": [{"period": period, **point}]},
                        "hour_of_day": {"sentiment": [{"hour": "09", **point}]},
                        "availability": {},
                        "rollup_provenance": {
                            "range_filter_semantics": "whole hours",
                            "partial_boundary_hours_included": False,
                            "non_whole_hour_timezone_offset_observed": False,
                        },
                        "page": {
                            "limit": 500,
                            "returned": 1,
                            "total": None,
                            "next_cursor": None if second else "page-two",
                            "count_is_unknown": True,
                        },
                    },
                    1.0,
                )
            period = "2023-02" if second else "2023-01"
            return (
                {
                    "bucket": "month",
                    "timezone": "UTC",
                    "series": [{"period": period, "calls": 1}],
                    "domains": [
                        {
                            "domain": "example.invalid",
                            "occurrences": 2,
                            "periods": [{"period": period, "occurrences": 1}],
                        }
                    ],
                    "summary": {"calls": 2},
                    "page": {
                        "limit": 500,
                        "returned": 1,
                        "total": 2,
                        "next_cursor": None if second else "page-two",
                    },
                },
                1.0,
            )

    client = FakePagedClient()
    nlp, elapsed = acceptance._analytics_all_pages(
        client, "nlp-trends", {"bucket": "month", "timezone": "UTC", "direction": "owner"}
    )
    assert elapsed == 2.0
    assert [row["period"] for row in nlp["series"]["sentiment"]] == ["2023-01", "2023-02"]
    hour = nlp["hour_of_day"]["sentiment"][0]
    assert hour["sample_count"] == 3
    assert hour["mean_confidence"] == pytest.approx(2 / 3)
    assert hour["mean_probabilities"] == pytest.approx({"positive": 1 / 3, "negative": 2 / 3})
    assert hour["score"] == pytest.approx(-1 / 3)
    assert nlp["page_sample_count"] == 3

    interaction, _elapsed = acceptance._analytics_all_pages(
        client, "interaction-series", {"bucket": "month", "timezone": "UTC"}
    )
    assert [row["period"] for row in interaction["series"]] == ["2023-01", "2023-02"]
    assert [row["period"] for row in interaction["domains"][0]["periods"]] == ["2023-01", "2023-02"]
    assert all(call[1]["limit"] == 500 for call in client.calls)
    assert sum("cursor" in call[1] for call in client.calls) == 2


def test_all_analytics_page_metrics_match_direct_sql_and_detect_corruption(
    acceptance_catalog: tuple[ArchiveLayout, Engine],
) -> None:
    import lancedb

    acceptance = _acceptance_module()
    layout, _engine = acceptance_catalog
    _engine.dispose()
    assert lancedb.connect(str(layout.vectors)).list_tables().tables == []
    app = create_app(layout.root, security_disabled=True, acceptance_read_only=True)
    readonly = acceptance._readonly_catalog_engine(layout.database)
    try:
        with TestClient(app) as web_client, readonly.connect() as connection:
            client = _TestApiClient(web_client)
            checks, report = acceptance._analytics_direct_sql_reconciliation(connection, client)
            corrupt_checks, corrupt_report = acceptance._analytics_direct_sql_reconciliation(
                connection, _CorruptAnalyticsClient(client)
            )
    finally:
        readonly.dispose()

    assert checks[0].status == "pass"
    assert report["endpoint_count"] == 13
    assert report["passed_endpoint_count"] == 13
    assert report["metric_count"] > 0
    assert report["mismatch_count"] == 0
    assert set(report["endpoints"]) == set(acceptance.ANALYTICS_DASHBOARD_ENDPOINTS)
    assert corrupt_checks[0].status == "fail"
    assert corrupt_report["endpoints"]["overview"]["mismatch_count"] == 2
    serialized = json.dumps(report)
    assert "Alex Example" not in serialized
    assert "example.com" not in serialized


def test_analytics_reconciliation_fails_closed_on_all_duplicate_key_families(
    acceptance_catalog: tuple[ArchiveLayout, Engine],
) -> None:
    acceptance = _acceptance_module()
    layout, _engine = acceptance_catalog
    app = create_app(layout.root, security_disabled=True)
    readonly = acceptance._readonly_catalog_engine(layout.database)
    try:
        with TestClient(app) as web_client, readonly.connect() as connection:
            checks, report = acceptance._analytics_direct_sql_reconciliation(
                connection,
                _DuplicateKeyAnalyticsClient(_TestApiClient(web_client)),
            )
    finally:
        readonly.dispose()

    assert checks[0].status == "fail"
    assert report["duplicate_key_failure_count"] > 0
    for namespace in ("overview", "messages", "attachments", "coverage", "nlp-trends"):
        assert report["endpoints"][namespace]["duplicate_key_failure_count"] > 0
    serialized = json.dumps(report)
    assert "private-duplicate" not in serialized


@pytest.mark.parametrize(
    ("expected", "actual"),
    ((True, 1), (1, True), (1, 1.0), (1.0, 1), ("1", 1)),
)
def test_analytics_metric_mismatch_requires_exact_scalar_types(
    expected: object, actual: object
) -> None:
    acceptance = _acceptance_module()
    assert acceptance._analytics_metric_mismatches(expected, actual) == 1


def test_analytics_network_and_topic_digests_detect_identity_topology_and_content_changes() -> None:
    acceptance = _acceptance_module()
    network = {
        "conversation_count": 1,
        "participant_label_count": 1,
        "nodes": [
            {
                "id": "conversation:private-one",
                "kind": "conversation",
                "conversation_type": "direct",
                "value": 7,
                "href": "/conversations/private-one",
            },
            {
                "id": "participant:private-two",
                "kind": "participant_label",
                "is_owner": False,
                "value": 7,
            },
        ],
        "edges": [
            {
                "id": "private-edge",
                "source": "conversation:private-one",
                "target": "participant:private-two",
                "value": 7,
            }
        ],
    }
    expected_network = acceptance._analytics_metric_projection("network", network)
    corrupt_network = json.loads(json.dumps(network))
    corrupt_network["edges"][0]["target"] = "participant:private-other"
    actual_network = acceptance._analytics_metric_projection("network", corrupt_network)
    assert acceptance._analytics_metric_mismatches(expected_network, actual_network) == 1

    topic = {
        "status": "ready",
        "ready": True,
        "latest_job": {"status": "completed", "progress_current": 2, "progress_total": 2},
        "semantic_index": {"window_count": 2},
        "result": {
            "corpus_window_count": 2,
            "sample_count": 2,
            "clusters": [
                {
                    "id": "private-cluster",
                    "ordinal": 1,
                    "sample_windows": 2,
                    "sample_share": 1.0,
                    "terms": ["private-term"],
                    "representatives": [
                        {
                            "window_id": "private-window",
                            "jump": {"href": "/conversations/private-conversation"},
                        }
                    ],
                }
            ],
        },
    }
    expected_topic = acceptance._analytics_metric_projection("topics", topic)
    corrupt_term = json.loads(json.dumps(topic))
    corrupt_term["result"]["clusters"][0]["terms"] = ["changed-private-term"]
    corrupt_link = json.loads(json.dumps(topic))
    corrupt_link["result"]["clusters"][0]["representatives"][0]["jump"]["href"] = (
        "/conversations/changed-private-conversation"
    )
    assert (
        acceptance._analytics_metric_mismatches(
            expected_topic, acceptance._analytics_metric_projection("topics", corrupt_term)
        )
        == 1
    )
    assert (
        acceptance._analytics_metric_mismatches(
            expected_topic, acceptance._analytics_metric_projection("topics", corrupt_link)
        )
        == 1
    )
    serialized = json.dumps({"network": expected_network, "topic": expected_topic})
    assert "private" not in serialized


def _browser_fixture_payload(
    acceptance: ModuleType,
    project_root: Path,
    *,
    archive_identity: str,
    totals: Mapping[str, int],
    largest: Mapping[str, object],
) -> tuple[Path, Path]:
    runner = project_root / "web" / "scripts" / "real-ui-acceptance.mjs"
    runner.parent.mkdir(parents=True)
    runner.write_text("// fixture acceptance runner\n", encoding="utf-8")
    runner_sha256 = hashlib.sha256(runner.read_bytes()).hexdigest()

    report_path = project_root / ".private-review" / "fixture-browser" / "report.json"
    acceptance._secure_output_directory(report_path.parent)
    artifacts = report_path.parent / "artifacts"
    artifacts.mkdir(mode=0o700)
    artifact_records: list[dict[str, object]] = []
    for relative_path in sorted(acceptance.EXPECTED_BROWSER_ARTIFACTS):
        artifact_path = report_path.parent / relative_path
        if relative_path.endswith(".png"):
            dimensions = (1440, 1000) if "desktop-" in relative_path else (390, 844)
            Image.new("RGB", dimensions, color=(32, 64, 96)).save(artifact_path, format="PNG")
        else:
            with zipfile.ZipFile(artifact_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("trace.trace", "fixture trace")
                archive.writestr("trace.network", "fixture network")
        artifact_path.chmod(0o600)
        artifact_records.append(
            {
                "relative_path": relative_path,
                "size_bytes": artifact_path.stat().st_size,
                "sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
            }
        )

    pages: list[dict[str, object]] = []
    interactions: list[dict[str, object]] = []
    timings: list[float] = []
    for viewport in acceptance.EXPECTED_BROWSER_VIEWPORTS:
        for index, route in enumerate(acceptance.EXPECTED_BROWSER_ROUTES, start=1):
            timing = float(100 + index)
            timings.append(timing)
            pages.append(
                {
                    "viewport": viewport,
                    "route": route,
                    "content_state": "ready",
                    "initial_content_ms": timing,
                    "axe_violation_count": 0,
                    "axe_incomplete_count": 0,
                    "horizontal_overflow": False,
                    "page_error_count": 0,
                    "external_request_count": 0,
                    "non_get_head_request_count": 0,
                    "reduced_motion_passed": True,
                    "skip_link_passed": True,
                    "screenshot": f"artifacts/{viewport}-{route}.png",
                }
            )
        interactions.extend(
            {
                "viewport": viewport,
                "interaction": interaction,
                "passed": True,
                "duration_ms": 25.0,
                "proof": acceptance._minimum_browser_interaction_proof(interaction),
            }
            for interaction in acceptance.EXPECTED_BROWSER_INTERACTIONS
        )
    proof_totals = {
        key: sum(int(row["proof"][key]) for row in interactions)
        for key in acceptance.BROWSER_INTERACTION_PROOF_KEYS
    }
    summary = {
        "page_runs": acceptance.EXPECTED_BROWSER_PAGE_RUNS,
        "screenshot_count": acceptance.EXPECTED_BROWSER_PAGE_RUNS,
        "initial_content_p95_ms": acceptance._percentile(timings, 0.95),
        "axe_violation_count": 0,
        "axe_incomplete_count": 0,
        "horizontal_overflow_count": 0,
        "page_error_count": 0,
        "external_request_count": 0,
        "non_get_head_request_count": 0,
        "websocket_request_count": 0,
        "service_worker_registration_count": 0,
        "reduced_motion_failure_count": 0,
        "largest_rendered_message_count": 48,
        "largest_loaded_message_count": 250,
        "reader_counts_valid": True,
        "screenshots_complete": True,
        "traces_complete": True,
        "desktop_and_mobile": True,
        "page_evidence_complete": True,
        "interaction_runs": acceptance.EXPECTED_BROWSER_INTERACTION_RUNS,
        "interaction_pass_count": acceptance.EXPECTED_BROWSER_INTERACTION_RUNS,
        "interaction_evidence_complete": True,
        "interaction_proof_complete": True,
        "interaction_proof_totals": proof_totals,
    }
    payload = {
        "schema_version": acceptance.BROWSER_REPORT_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "runner_sha256": runner_sha256,
        "catalog": {
            "archive_identity_sha256": archive_identity,
            "acceptance_identity": _fixture_acceptance_identity(nonce_character="4"),
            "totals": dict(totals),
            "largest_conversation": dict(largest),
        },
        "summary": summary,
        "pages": pages,
        "interactions": interactions,
        "artifacts": artifact_records,
        "gate": {"passed": True},
        "safety": {
            "aggregate_report_only": True,
            "screenshots_and_traces_separate": True,
            "owner_only_artifacts": True,
            "source_text_emitted_in_report": False,
            "token_emitted": False,
            "external_network_blocked": True,
            "websockets_blocked": True,
            "service_workers_blocked": True,
            "acceptance_read_only_server": True,
            "only_get_head_requests": True,
            "analytics_cache_reads_writes_disabled": True,
            "mutating_user_actions_performed": False,
        },
    }
    acceptance._write_private_json(report_path, payload)
    return report_path, report_path.parent / sorted(acceptance.EXPECTED_BROWSER_ARTIFACTS)[0]


def test_browser_evidence_is_fresh_catalog_bound_and_artifact_bound(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    acceptance = _acceptance_module()
    fake_script = tmp_path / "scripts" / "real_export_acceptance.py"
    fake_script.parent.mkdir()
    fake_script.write_text("", encoding="utf-8")
    monkeypatch.setattr(acceptance, "__file__", str(fake_script))
    archive_identity = "0123456789abcdef"
    totals = {"conversations": 4, "messages": 20, "attachments": 7, "activities": 3}
    largest = {"id_sha256": "fedcba9876543210", "message_count": 12}
    report_path, artifact_path = _browser_fixture_payload(
        acceptance,
        tmp_path,
        archive_identity=archive_identity,
        totals=totals,
        largest=largest,
    )

    checks, sanitized = acceptance._browser_evidence(
        report_path,
        expected_archive_identity=archive_identity,
        expected_totals={**totals, "events": 0, "contacts": 0, "timeline": 23},
        expected_largest=largest,
        expected_server_identity=_fixture_acceptance_identity(nonce_character="5"),
    )

    assert checks[0].status == "pass"
    assert sanitized["artifact_inventory_reconciles"] is True
    assert sanitized["interaction_rows_reconcile"] is True
    assert sanitized["interaction_proof_complete"] is True
    assert sanitized["interaction_proof_totals_reconcile"] is True
    assert set(sanitized["interaction_proof_totals"]) == set(
        acceptance.BROWSER_INTERACTION_PROOF_KEYS
    )
    assert "sha256" not in json.dumps(sanitized)

    same_nonce_checks, same_nonce_evidence = acceptance._browser_evidence(
        report_path,
        expected_archive_identity=archive_identity,
        expected_totals={**totals, "events": 0, "contacts": 0, "timeline": 23},
        expected_largest=largest,
        expected_server_identity=_fixture_acceptance_identity(nonce_character="4"),
    )
    assert same_nonce_checks[0].status == "fail"
    assert same_nonce_evidence["restarted_server_nonce_differs"] is False

    mismatched_identity = _fixture_acceptance_identity(nonce_character="5")
    mismatched_identity["backend_source_sha256"] = "9" * 64
    identity_checks, identity_evidence = acceptance._browser_evidence(
        report_path,
        expected_archive_identity=archive_identity,
        expected_totals={**totals, "events": 0, "contacts": 0, "timeline": 23},
        expected_largest=largest,
        expected_server_identity=mismatched_identity,
    )
    assert identity_checks[0].status == "fail"
    assert identity_evidence["stable_server_identity_matches"] is False

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["schema_version"] = 4
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    report_path.chmod(0o600)
    legacy_checks, legacy_evidence = acceptance._browser_evidence(
        report_path,
        expected_archive_identity=archive_identity,
        expected_totals={**totals, "events": 0, "contacts": 0, "timeline": 23},
        expected_largest=largest,
        expected_server_identity=_fixture_acceptance_identity(nonce_character="5"),
    )
    assert legacy_checks[0].status == "fail"
    assert legacy_evidence["report_object_schemas_reconcile"] is True
    assert legacy_evidence["report_schema_version_matches"] is False
    payload["schema_version"] = acceptance.BROWSER_REPORT_SCHEMA_VERSION
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    report_path.chmod(0o600)

    artifact_path.write_bytes(b"tampered fixture evidence")
    artifact_path.chmod(0o600)
    tampered_checks, tampered = acceptance._browser_evidence(
        report_path,
        expected_archive_identity=archive_identity,
        expected_totals={**totals, "events": 0, "contacts": 0, "timeline": 23},
        expected_largest=largest,
        expected_server_identity=_fixture_acceptance_identity(nonce_character="5"),
    )
    assert tampered_checks[0].status == "fail"
    assert tampered["artifact_inventory_reconciles"] is False

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["interactions"][0]["passed"] = False
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    report_path.chmod(0o600)
    interaction_checks, interaction_evidence = acceptance._browser_evidence(
        report_path,
        expected_archive_identity=archive_identity,
        expected_totals={**totals, "events": 0, "contacts": 0, "timeline": 23},
        expected_largest=largest,
        expected_server_identity=_fixture_acceptance_identity(nonce_character="5"),
    )
    assert interaction_checks[0].status == "fail"
    assert interaction_evidence["interaction_rows_reconcile"] is False


@pytest.mark.parametrize("artifact_kind", ("png", "trace"))
def test_browser_evidence_decodes_artifacts_beyond_manifest_hashes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, artifact_kind: str
) -> None:
    acceptance = _acceptance_module()
    fake_script = tmp_path / "scripts" / "real_export_acceptance.py"
    fake_script.parent.mkdir()
    fake_script.write_text("", encoding="utf-8")
    monkeypatch.setattr(acceptance, "__file__", str(fake_script))
    totals = {"conversations": 4, "messages": 20, "attachments": 7, "activities": 3}
    largest = {"id_sha256": "fedcba9876543210", "message_count": 12}
    report_path, _artifact_path = _browser_fixture_payload(
        acceptance,
        tmp_path,
        archive_identity="0123456789abcdef",
        totals=totals,
        largest=largest,
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    suffix = ".png" if artifact_kind == "png" else ".zip"
    record = next(row for row in payload["artifacts"] if str(row["relative_path"]).endswith(suffix))
    artifact = report_path.parent / str(record["relative_path"])
    if artifact_kind == "png":
        artifact.write_bytes(b"not a decoded PNG")
    else:
        with zipfile.ZipFile(artifact, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("trace.trace", "missing required network stream")
    artifact.chmod(0o600)
    record["size_bytes"] = artifact.stat().st_size
    record["sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    report_path.chmod(0o600)

    checks, sanitized = acceptance._browser_evidence(
        report_path,
        expected_archive_identity="0123456789abcdef",
        expected_totals={**totals, "events": 0, "contacts": 0, "timeline": 23},
        expected_largest=largest,
        expected_server_identity=_fixture_acceptance_identity(nonce_character="5"),
    )

    assert checks[0].status == "fail"
    assert sanitized["artifact_inventory_reconciles"] is False


def test_browser_evidence_rejects_unexpected_report_object_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    acceptance = _acceptance_module()
    fake_script = tmp_path / "scripts" / "real_export_acceptance.py"
    fake_script.parent.mkdir()
    fake_script.write_text("", encoding="utf-8")
    monkeypatch.setattr(acceptance, "__file__", str(fake_script))
    totals = {"conversations": 4, "messages": 20, "attachments": 7, "activities": 3}
    largest = {"id_sha256": "fedcba9876543210", "message_count": 12}
    report_path, _artifact_path = _browser_fixture_payload(
        acceptance,
        tmp_path,
        archive_identity="0123456789abcdef",
        totals=totals,
        largest=largest,
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["summary"]["unexpected_private_extension"] = True
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    report_path.chmod(0o600)

    checks, sanitized = acceptance._browser_evidence(
        report_path,
        expected_archive_identity="0123456789abcdef",
        expected_totals={**totals, "events": 0, "contacts": 0, "timeline": 23},
        expected_largest=largest,
        expected_server_identity=_fixture_acceptance_identity(nonce_character="5"),
    )

    assert checks[0].status == "fail"
    assert sanitized["report_object_schemas_reconcile"] is False


@pytest.mark.parametrize(
    "case",
    (
        "page_timing_bool",
        "page_count_bool",
        "interaction_duration_bool",
        "summary_zero_count_bool",
        "summary_reader_bools",
        "summary_nonfinite_timing",
        "catalog_zero_count_bool",
        "catalog_largest_count_bool",
        "artifact_size_bool",
    ),
)
def test_browser_evidence_rejects_numeric_type_smuggling(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case: str,
) -> None:
    acceptance = _acceptance_module()
    fake_script = tmp_path / "scripts" / "real_export_acceptance.py"
    fake_script.parent.mkdir()
    fake_script.write_text("", encoding="utf-8")
    monkeypatch.setattr(acceptance, "__file__", str(fake_script))
    totals = {
        "conversations": 4,
        "messages": 20,
        "attachments": 7,
        "activities": 0 if case == "catalog_zero_count_bool" else 3,
    }
    largest = {
        "id_sha256": "fedcba9876543210",
        "message_count": 1 if case == "catalog_largest_count_bool" else 12,
    }
    report_path, _artifact_path = _browser_fixture_payload(
        acceptance,
        tmp_path,
        archive_identity="0123456789abcdef",
        totals=totals,
        largest=largest,
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if case == "page_timing_bool":
        for page in payload["pages"]:
            page["initial_content_ms"] = True
        payload["summary"]["initial_content_p95_ms"] = 1
    elif case == "page_count_bool":
        payload["pages"][0]["axe_violation_count"] = False
    elif case == "interaction_duration_bool":
        payload["interactions"][0]["duration_ms"] = True
    elif case == "summary_zero_count_bool":
        payload["summary"]["external_request_count"] = False
    elif case == "summary_reader_bools":
        payload["summary"]["largest_rendered_message_count"] = True
        payload["summary"]["largest_loaded_message_count"] = True
    elif case == "summary_nonfinite_timing":
        payload["summary"]["initial_content_p95_ms"] = float("nan")
    elif case == "catalog_zero_count_bool":
        payload["catalog"]["totals"]["activities"] = False
    elif case == "catalog_largest_count_bool":
        payload["catalog"]["largest_conversation"]["message_count"] = True
    elif case == "artifact_size_bool":
        payload["artifacts"][0]["size_bytes"] = True
    else:  # pragma: no cover - the parametrization is exhaustive
        raise AssertionError(f"Unhandled numeric-smuggling case: {case}")
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    report_path.chmod(0o600)

    checks, sanitized = acceptance._browser_evidence(
        report_path,
        expected_archive_identity="0123456789abcdef",
        expected_totals={**totals, "events": 0, "contacts": 0, "timeline": 20},
        expected_largest=largest,
        expected_server_identity=_fixture_acceptance_identity(nonce_character="5"),
    )

    assert checks[0].status == "fail"
    assert (
        sanitized["report_object_schemas_reconcile"] is False
        or sanitized["page_rows_reconcile"] is False
        or sanitized["interaction_rows_reconcile"] is False
        or sanitized["artifact_inventory_reconciles"] is False
    )


def test_browser_evidence_rejects_vacuous_or_unreconciled_interaction_proof(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    acceptance = _acceptance_module()
    assert set(acceptance.BROWSER_INTERACTION_PROOF_MINIMUMS) == set(
        acceptance.EXPECTED_BROWSER_INTERACTIONS
    )
    fake_script = tmp_path / "scripts" / "real_export_acceptance.py"
    fake_script.parent.mkdir()
    fake_script.write_text("", encoding="utf-8")
    monkeypatch.setattr(acceptance, "__file__", str(fake_script))
    totals = {"conversations": 4, "messages": 20, "attachments": 7, "activities": 3}
    largest = {"id_sha256": "fedcba9876543210", "message_count": 12}
    report_path, _artifact_path = _browser_fixture_payload(
        acceptance,
        tmp_path,
        archive_identity="0123456789abcdef",
        totals=totals,
        largest=largest,
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    conversation = next(
        row for row in payload["interactions"] if row["interaction"] == "conversation_sort_and_page"
    )
    conversation["proof"]["pagination_exercised_count"] = 0
    payload["summary"]["interaction_proof_totals"]["pagination_exercised_count"] -= 1
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    report_path.chmod(0o600)

    checks, sanitized = acceptance._browser_evidence(
        report_path,
        expected_archive_identity="0123456789abcdef",
        expected_totals={**totals, "events": 0, "contacts": 0, "timeline": 23},
        expected_largest=largest,
        expected_server_identity=_fixture_acceptance_identity(nonce_character="5"),
    )
    assert checks[0].status == "fail"
    assert sanitized["interaction_rows_reconcile"] is False
    assert sanitized["interaction_proof_complete"] is False

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    conversation["proof"] = acceptance._minimum_browser_interaction_proof(
        "conversation_sort_and_page"
    )
    payload["interactions"] = [
        conversation if row["interaction"] == "conversation_sort_and_page" else row
        for row in payload["interactions"]
    ]
    payload["summary"]["interaction_proof_totals"]["assertion_count"] += 1
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    report_path.chmod(0o600)
    totals_checks, totals_sanitized = acceptance._browser_evidence(
        report_path,
        expected_archive_identity="0123456789abcdef",
        expected_totals={**totals, "events": 0, "contacts": 0, "timeline": 23},
        expected_largest=largest,
        expected_server_identity=_fixture_acceptance_identity(nonce_character="5"),
    )
    assert totals_checks[0].status == "fail"
    assert totals_sanitized["interaction_proof_totals_reconcile"] is False


def test_current_deep_integrity_gate_requires_complete_unique_passing_inventory() -> None:
    acceptance = _acceptance_module()
    passing = [
        {
            "key": key,
            "status": "pass",
            "expected": 1,
            "actual": 1,
            "detail": "fixture",
            "mandatory": True,
        }
        for key in sorted(acceptance.SOURCE_RETIREMENT_REQUIRED_CHECK_KEYS)
    ]

    passed, evidence = acceptance._current_deep_integrity_gate(passing)
    assert passed.status == "pass"
    assert evidence["executed_in_current_acceptance"] is True
    assert evidence["deep_verified"] is True
    assert evidence["required_check_present_count"] == evidence["required_check_count"]

    empty, _empty_evidence = acceptance._current_deep_integrity_gate([])
    assert empty.status == "fail"

    missing, missing_evidence = acceptance._current_deep_integrity_gate(passing[1:])
    assert missing.status == "fail"
    assert (
        missing_evidence["required_check_present_count"] < missing_evidence["required_check_count"]
    )

    failed_deep_inventory = [dict(item) for item in passing]
    next(item for item in failed_deep_inventory if item["key"] == "deep.source_bytes")["status"] = (
        "fail"
    )
    failed, failed_evidence = acceptance._current_deep_integrity_gate(failed_deep_inventory)
    assert failed.status == "fail"
    assert failed_evidence["deep_verified"] is False
    assert "deep.source_bytes" in failed_evidence["non_pass_keys"]

    duplicate, duplicate_evidence = acceptance._current_deep_integrity_gate(
        [*passing, dict(passing[0])]
    )
    assert duplicate.status == "fail"
    assert duplicate_evidence["duplicate_key_count"] == 1


def test_private_review_wrapper_rejects_nested_symlinks_and_runs_on_regular_tree(
    tmp_path: Path,
) -> None:
    source = Path(__file__).resolve().parents[1] / "scripts" / "run-private-ui-review.sh"
    script = tmp_path / "scripts" / source.name
    script.parent.mkdir()
    shutil.copyfile(source, script)
    private_root = tmp_path / ".private-review"
    private_root.mkdir()
    target = tmp_path / "outside"
    target.mkdir()
    nested_link = private_root / "nested-link"
    nested_link.symlink_to(target, target_is_directory=True)

    rejected = subprocess.run(
        ["sh", str(script), "true"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode == 2
    assert "Refusing nested symlink" in rejected.stderr

    nested_link.unlink()
    accepted = subprocess.run(
        ["sh", str(script), "true"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert accepted.returncode == 0, accepted.stderr
    assert stat.S_IMODE(private_root.stat().st_mode) == 0o700


def test_report_exit_code_fails_closed_for_pending_or_malformed_gate() -> None:
    acceptance = _acceptance_module()

    assert acceptance._report_exit_code({"gate": {"passed": True}}) == 0
    assert acceptance._report_exit_code({"gate": {"passed": False}}) == 1
    assert acceptance._report_exit_code({"gate": {"status": "pending"}}) == 1
    assert acceptance._report_exit_code({}) == 1


def test_opaque_mismatch_evidence_never_contains_raw_identifiers() -> None:
    acceptance = _acceptance_module()
    private_expected = "person-or-conversation-private-id"
    private_actual = "another-private-id"

    mismatch = acceptance._first_mismatch([private_expected], [private_actual])

    assert mismatch is not None
    serialized = json.dumps(mismatch)
    assert private_expected not in serialized
    assert private_actual not in serialized
    assert mismatch["expected_id_sha256"] == acceptance._opaque(private_expected)
