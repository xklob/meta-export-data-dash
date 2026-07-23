from __future__ import annotations

import hashlib
import json
import os
import subprocess
import warnings
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session
from typer.testing import CliRunner

import meta_archive.cli as cli_module
from meta_archive.api.main import create_app
from meta_archive.models import AttachmentOccurrence, Blob, Job, MediaDerivative
from meta_archive.paths import ArchiveLayout
from meta_archive.services.media_derivatives import (
    DerivativeKind,
    DerivativeResult,
    DerivativeSummary,
    FfmpegMediaRunner,
    MediaCommandError,
    build_media_derivatives,
)


class FakeMediaRunner:
    identity = "fake-media-runner"

    def __init__(
        self,
        *,
        failures: dict[DerivativeKind, int] | None = None,
        interrupt: bool = False,
    ) -> None:
        self.failures = dict(failures or {})
        self.interrupt = interrupt
        self.calls: list[tuple[DerivativeKind, Path, Path]] = []

    def generate(
        self,
        kind: DerivativeKind,
        source: Path,
        destination: Path,
    ) -> DerivativeResult:
        self.calls.append((kind, source, destination))
        destination.write_bytes(f"fixture-{kind}-{len(self.calls)}".encode())
        if self.interrupt:
            raise KeyboardInterrupt
        remaining = self.failures.get(kind, 0)
        if remaining:
            self.failures[kind] = remaining - 1
            raise MediaCommandError(f"fixture failure at {source}")
        duration = 12_345 if kind in {"video_poster", "audio_waveform"} else None
        return DerivativeResult(duration_ms=duration, width=480, height=180)


def _materialized_media(
    layout: ArchiveLayout,
    engine: Engine,
    *,
    content: bytes,
    media_type: str,
    kind: str,
    occurrences: int = 1,
    extension: str = ".fixture",
    availability: str = "unlinked",
) -> str:
    sha256 = hashlib.sha256(content).hexdigest()
    blob_path = layout.blob_path(sha256)
    blob_path.parent.mkdir(parents=True, exist_ok=True)
    blob_path.write_bytes(content)
    blob_path.chmod(0o444)
    with Session(engine) as session, session.begin():
        session.add(
            Blob(
                sha256=sha256,
                size_bytes=len(content),
                media_type=media_type,
                extension=extension,
                relative_path=blob_path.relative_to(layout.root).as_posix(),
                state="materialized",
            )
        )
        session.flush()
        for ordinal in range(occurrences):
            session.add(
                AttachmentOccurrence(
                    id=hashlib.sha256(f"{sha256}:{ordinal}".encode()).hexdigest(),
                    kind=kind,
                    ordinal=ordinal,
                    filename=f"fixture-{ordinal}",
                    media_type=media_type,
                    size_bytes=len(content),
                    availability=availability,
                    blob_sha256=sha256,
                    metadata_json={},
                )
            )
    return sha256


@pytest.fixture
def derivative_catalog(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> Iterator[tuple[ArchiveLayout, Engine, dict[str, str]]]:
    identifiers = {
        "image": _materialized_media(
            archive_layout,
            catalog_engine,
            content=b"fixture-image-source",
            media_type="image/jpeg",
            kind="photos",
            occurrences=2,
        ),
        "video": _materialized_media(
            archive_layout,
            catalog_engine,
            content=b"fixture-video-source",
            media_type="video/mp4",
            kind="videos",
        ),
        "audio": _materialized_media(
            archive_layout,
            catalog_engine,
            content=b"fixture-audio-source",
            # Facebook commonly exports M4A voice messages with this generic container MIME.
            media_type="video/mp4",
            kind="audio_files",
        ),
    }
    yield archive_layout, catalog_engine, identifiers


def test_unique_blob_derivatives_are_atomic_idempotent_and_manifested(
    derivative_catalog: tuple[ArchiveLayout, Engine, dict[str, str]],
) -> None:
    layout, engine, identifiers = derivative_catalog
    runner = FakeMediaRunner()

    first = build_media_derivatives(layout, engine, runner=runner)
    second = build_media_derivatives(layout, engine, runner=runner)

    assert first.total == first.generated == 3
    assert first.failed == first.reused == 0
    assert second.total == second.reused == 3
    assert second.generated == second.failed == 0
    assert len(runner.calls) == 3
    assert {source for _kind, source, _target in runner.calls} == {
        layout.blob_path(identifier) for identifier in identifiers.values()
    }
    assert all(layout.thumbnails in target.parents for _kind, _source, target in runner.calls)

    with Session(engine) as session:
        rows = list(session.scalars(select(MediaDerivative).order_by(MediaDerivative.kind)))
        jobs = list(
            session.scalars(
                select(Job).where(Job.kind == "media_derivatives").order_by(Job.created_at)
            )
        )
    assert len(rows) == 3
    assert {row.kind for row in rows} == {
        "image_thumbnail",
        "video_poster",
        "audio_waveform",
    }
    assert all(row.status == "ready" and row.attempt_count == 1 for row in rows)
    assert all(row.duration_ms == 12_345 for row in rows if row.kind != "image_thumbnail")
    assert next(row for row in rows if row.kind == "image_thumbnail").duration_ms is None
    for row in rows:
        output = layout.root / row.relative_path
        assert output.is_file()
        assert output.stat().st_mode & 0o222 == 0
        assert hashlib.sha256(output.read_bytes()).hexdigest() == row.output_sha256
    assert [job.status for job in jobs] == ["completed", "completed"]
    assert jobs[-1].progress_current == jobs[-1].progress_total == 3
    manifest = json.loads(layout.derivative_manifest.read_text())
    assert manifest["pipeline_version"] == "media-v1"
    assert len(manifest["records"]) == 3
    assert {record["status"] for record in manifest["records"]} == {"ready"}
    assert layout.derivative_manifest.stat().st_mode & 0o222 == 0


def test_failure_status_can_be_skipped_then_retried_without_partial_output(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> None:
    _materialized_media(
        archive_layout,
        catalog_engine,
        content=b"retry-audio",
        media_type="audio/m4a",
        kind="audio_files",
    )
    runner = FakeMediaRunner(failures={"audio_waveform": 1})

    failed = build_media_derivatives(archive_layout, catalog_engine, runner=runner)
    skipped = build_media_derivatives(
        archive_layout, catalog_engine, runner=runner, retry_failed=False
    )
    recovered = build_media_derivatives(archive_layout, catalog_engine, runner=runner)

    assert failed.failed == 1
    assert skipped.skipped_failed == 1
    assert recovered.generated == recovered.retried == 1
    assert len(runner.calls) == 2
    with Session(catalog_engine) as session:
        row = session.scalar(select(MediaDerivative))
        jobs = list(
            session.scalars(
                select(Job).where(Job.kind == "media_derivatives").order_by(Job.created_at)
            )
        )
    assert row is not None
    assert row.status == "ready"
    assert row.attempt_count == 2
    assert row.error_text is None
    assert [job.status for job in jobs] == [
        "completed_with_errors",
        "completed_with_errors",
        "completed",
    ]


def test_interrupted_job_resumes_same_job_and_processing_record(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> None:
    _materialized_media(
        archive_layout,
        catalog_engine,
        content=b"interrupt-video",
        media_type="video/mp4",
        kind="videos",
    )
    interrupter = FakeMediaRunner(interrupt=True)

    with pytest.raises(KeyboardInterrupt):
        build_media_derivatives(archive_layout, catalog_engine, runner=interrupter)
    with Session(catalog_engine) as session:
        failed_job = session.scalar(select(Job).where(Job.kind == "media_derivatives"))
        interrupted_row = session.scalar(select(MediaDerivative))
    assert failed_job is not None and failed_job.status == "failed"
    assert interrupted_row is not None and interrupted_row.status == "processing"

    recovered = build_media_derivatives(archive_layout, catalog_engine, runner=FakeMediaRunner())
    assert recovered.job_id == failed_job.id
    assert recovered.retried == recovered.generated == 1
    with Session(catalog_engine) as session:
        assert session.get(Job, failed_job.id).status == "completed"  # type: ignore[union-attr]
        assert session.scalar(select(MediaDerivative)).status == "ready"  # type: ignore[union-attr]


def test_worker_never_falls_back_to_source_paths_or_accepts_mutable_blobs(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> None:
    content = b"tempting-original-media"
    sha256 = hashlib.sha256(content).hexdigest()
    original = archive_layout.root / "package-a" / "media.jpg"
    original.parent.mkdir(parents=True)
    original.write_bytes(content)
    expected_blob = archive_layout.blob_path(sha256)
    with Session(catalog_engine) as session, session.begin():
        session.add(
            Blob(
                sha256=sha256,
                size_bytes=len(content),
                media_type="image/jpeg",
                extension=".jpg",
                relative_path=expected_blob.relative_to(archive_layout.root).as_posix(),
                state="materialized",
            )
        )
        session.flush()
        session.add(
            AttachmentOccurrence(
                id="a" * 64,
                kind="photos",
                ordinal=0,
                source_uri=original.relative_to(archive_layout.root).as_posix(),
                media_type="image/jpeg",
                size_bytes=len(content),
                availability="resolved",
                blob_sha256=sha256,
                metadata_json={},
            )
        )
    runner = FakeMediaRunner()

    missing_blob = build_media_derivatives(archive_layout, catalog_engine, runner=runner)
    assert missing_blob.failed == 1
    assert runner.calls == []
    with Session(catalog_engine) as session:
        row = session.scalar(select(MediaDerivative))
    assert row is not None
    assert str(archive_layout.root) not in (row.error_text or "")

    expected_blob.parent.mkdir(parents=True, exist_ok=True)
    os.link(original, expected_blob)
    hardlinked = build_media_derivatives(archive_layout, catalog_engine, runner=runner)
    assert hardlinked.failed == 1
    assert runner.calls == []
    assert expected_blob.stat().st_nlink == 2


def test_read_only_library_hardlinks_do_not_make_a_blob_ineligible(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> None:
    sha256 = _materialized_media(
        archive_layout,
        catalog_engine,
        content=b"library-linked-image",
        media_type="image/jpeg",
        kind="photos",
    )
    blob = archive_layout.blob_path(sha256)
    library_link = archive_layout.library / "fixture.jpg"
    os.link(blob, library_link)
    assert blob.stat().st_nlink == 2

    result = build_media_derivatives(archive_layout, catalog_engine, runner=FakeMediaRunner())

    assert result.generated == 1
    assert result.failed == 0


def test_unlinked_m4a_extension_overrides_generic_video_mime(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> None:
    _materialized_media(
        archive_layout,
        catalog_engine,
        content=b"unlinked-m4a",
        media_type="video/mp4",
        kind="video",
        extension=".m4a",
    )
    runner = FakeMediaRunner()

    result = build_media_derivatives(archive_layout, catalog_engine, runner=runner)

    assert result.generated == 1
    assert [kind for kind, _source, _destination in runner.calls] == ["audio_waveform"]


def test_resolved_normalized_audio_kind_overrides_generic_video_mime(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> None:
    _materialized_media(
        archive_layout,
        catalog_engine,
        content=b"resolved-meta-voice-message",
        media_type="video/mp4",
        kind="audio",
        extension=".mp4",
        availability="resolved",
    )
    runner = FakeMediaRunner()

    result = build_media_derivatives(archive_layout, catalog_engine, runner=runner)

    assert result.generated == 1
    assert [kind for kind, _source, _destination in runner.calls] == ["audio_waveform"]


def test_protected_derivative_route_uses_fixed_ids_and_conservative_mime(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> None:
    _materialized_media(
        archive_layout,
        catalog_engine,
        content=b"route-image",
        media_type="image/svg+xml",
        kind="photos",
    )
    summary = build_media_derivatives(archive_layout, catalog_engine, runner=FakeMediaRunner())
    with Session(catalog_engine) as session:
        row = session.scalar(select(MediaDerivative))
        assert row is not None
        derivative_id = row.id
        derivative_bytes = (archive_layout.root / row.relative_path).read_bytes()

    protected_app = create_app(archive_layout.root, access_token="a" * 32)
    with TestClient(protected_app, base_url="http://127.0.0.1") as protected_client:
        assert protected_client.get(f"/api/v1/derivatives/{derivative_id}").status_code == 401

    app = create_app(archive_layout.root, security_disabled=True)
    with TestClient(app) as client:
        response = client.get(f"/api/v1/derivatives/{derivative_id}")
        head = client.head(f"/api/v1/derivatives/{derivative_id}")
        invalid = client.get("/api/v1/derivatives/../../etc/passwd")
        media = client.get("/api/v1/media", params={"limit": 10})
    assert response.status_code == 200
    assert response.content == derivative_bytes
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["content-disposition"].startswith("inline;")
    assert head.status_code == 200 and head.content == b""
    assert head.headers["content-length"] == str(len(derivative_bytes))
    assert invalid.status_code == 404
    media_item = next(item for item in media.json()["items"] if item["blob_sha256"] is not None)
    assert media_item["derivatives"] == [
        {
            "id": derivative_id,
            "kind": "image_thumbnail",
            "status": "ready",
            "media_type": "image/jpeg",
            "size_bytes": len(derivative_bytes),
            "duration_ms": None,
            "width": 480,
            "height": 180,
            "href": f"/api/v1/derivatives/{derivative_id}",
        }
    ]
    assert summary.generated == 1


def test_tampered_ready_derivative_is_not_served_and_is_regenerated(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> None:
    _materialized_media(
        archive_layout,
        catalog_engine,
        content=b"tamper-video",
        media_type="video/mp4",
        kind="videos",
    )
    first_runner = FakeMediaRunner()
    build_media_derivatives(archive_layout, catalog_engine, runner=first_runner)
    with Session(catalog_engine) as session:
        row = session.scalar(select(MediaDerivative))
        assert row is not None
        output = archive_layout.root / row.relative_path
        derivative_id = row.id
    output.chmod(0o644)
    output.write_bytes(b"tampered")
    output.chmod(0o444)

    app = create_app(archive_layout.root, security_disabled=True)
    with TestClient(app) as client:
        assert client.get(f"/api/v1/derivatives/{derivative_id}").status_code == 409

    repair_runner = FakeMediaRunner()
    repaired = build_media_derivatives(archive_layout, catalog_engine, runner=repair_runner)
    assert repaired.generated == 1
    assert repaired.reused == 0
    assert len(repair_runner.calls) == 1


def test_derivative_route_rejects_a_symlinked_output_directory(
    archive_layout: ArchiveLayout, catalog_engine: Engine
) -> None:
    _materialized_media(
        archive_layout,
        catalog_engine,
        content=b"symlink-route-image",
        media_type="image/jpeg",
        kind="photos",
    )
    build_media_derivatives(archive_layout, catalog_engine, runner=FakeMediaRunner())
    with Session(catalog_engine) as session:
        row = session.scalar(select(MediaDerivative))
        assert row is not None
        derivative_id = row.id
        output = archive_layout.root / row.relative_path
    relocated = archive_layout.root.parent / "relocated-private-derivative-directory"
    output.parent.rename(relocated)
    output.parent.symlink_to(relocated, target_is_directory=True)

    app = create_app(archive_layout.root, security_disabled=True)
    with TestClient(app) as client:
        response = client.get(f"/api/v1/derivatives/{derivative_id}")

    assert response.status_code == 409
    assert response.json()["detail"] == (
        "Media derivative bytes are unavailable or do not match the catalog"
    )


def test_ffmpeg_runner_uses_bounded_argv_only_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "ffmpeg"
    executable.write_bytes(b"fixture executable")
    executable.chmod(0o755)
    source = tmp_path / "source;not-a-command.m4a"
    source.write_bytes(b"fixture")
    destination = tmp_path / "waveform.png"
    observed: dict[str, object] = {}

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        warnings.warn(
            "lancedb fork support is experimental: fixture at-fork warning",
            RuntimeWarning,
            stacklevel=2,
        )
        warnings.warn("unrelated runner warning", UserWarning, stacklevel=2)
        observed["argv"] = argv
        observed.update(kwargs)
        stderr = kwargs["stderr"]
        assert hasattr(stderr, "write")
        stderr.write(b"Input #0\n  Duration: 00:00:01.50, start: 0.000000\n")
        destination.write_bytes(b"fixture-waveform")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = FfmpegMediaRunner(executable, timeout_seconds=17).generate(
            "audio_waveform", source, destination
        )

    argv = observed["argv"]
    assert isinstance(argv, list)
    assert argv[0] == str(executable.resolve())
    assert str(source) in argv
    assert argv.count(str(source)) == 1
    assert observed.get("shell", False) is False
    assert observed["timeout"] == 17
    assert callable(observed["preexec_fn"])
    assert observed["env"] == {"LC_ALL": "C", "LANG": "C"}
    assert result.duration_ms == 1_500
    assert [str(item.message) for item in caught] == ["unrelated runner warning"]


def test_build_derivatives_cli_wires_retry_and_reports_failures(
    archive_layout: ArchiveLayout, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_layout.ensure_runtime_directories()
    captured: dict[str, object] = {}

    class StubRunner:
        def __init__(self, executable: Path | None = None) -> None:
            captured["executable"] = executable

    def stub_build(
        layout: ArchiveLayout,
        _engine: Engine,
        *,
        runner: object,
        retry_failed: bool,
        progress: object,
    ) -> DerivativeSummary:
        captured.update(
            layout=layout,
            runner=runner,
            retry_failed=retry_failed,
            progress=progress,
        )
        return DerivativeSummary(
            job_id="fixture-job",
            pipeline_version="media-v1",
            total=1,
            generated=0,
            reused=0,
            failed=1,
            skipped_failed=0,
            retried=0,
            manifest=str(layout.derivative_manifest),
        )

    monkeypatch.setattr(cli_module, "FfmpegMediaRunner", StubRunner)
    monkeypatch.setattr(cli_module, "build_media_derivatives", stub_build)
    result = CliRunner().invoke(
        cli_module.app,
        [
            "build-derivatives",
            "--archive-root",
            str(archive_layout.root),
            "--skip-failed",
            "--output-json",
        ],
    )

    assert result.exit_code == 1
    assert captured["retry_failed"] is False
    assert captured["progress"] is None
    assert '"failed": 1' in result.output
