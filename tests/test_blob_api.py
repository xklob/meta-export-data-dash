from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from meta_archive.api.main import create_app
from meta_archive.database import create_catalog_engine, run_migrations
from meta_archive.models import SourceFile
from meta_archive.paths import ArchiveLayout
from meta_archive.services.consolidation import build_blob_store
from meta_archive.services.inventory import scan_archive


@pytest.fixture
def blob_catalog(tmp_path: Path) -> Iterator[tuple[TestClient, ArchiveLayout, dict[str, str]]]:
    layout = ArchiveLayout(tmp_path / "meta-backup-blob-api")
    source = layout.root / "package-a" / "your_facebook_activity" / "messages" / "files"
    source.mkdir(parents=True)
    payloads = {
        "photo.png": b"fixture-png-bytes",
        "unsafe.html": b"<script>fixture only</script>",
        "private.pem": b"-----BEGIN PRIVATE KEY-----\nfixture\n",
    }
    for name, content in payloads.items():
        (source / name).write_bytes(content)
    layout.ensure_runtime_directories()
    run_migrations(layout.database)
    engine = create_catalog_engine(layout.database)
    scan_archive(layout, engine, hash_files=True)
    build_blob_store(layout, engine)
    with Session(engine) as session:
        hashes = {
            row.relative_path.rsplit("/", 1)[-1]: row.sha256
            for row in session.scalars(select(SourceFile))
            if row.sha256 is not None
        }
    engine.dispose()
    app = create_app(layout.root, security_disabled=True)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        yield client, layout, hashes


def test_blob_stream_supports_get_head_and_single_byte_ranges(
    blob_catalog: tuple[TestClient, ArchiveLayout, dict[str, str]],
) -> None:
    client, _layout, hashes = blob_catalog
    sha256 = hashes["photo.png"]
    full = client.get(f"/api/v1/blobs/{sha256}", params={"filename": "photo.png"})
    assert full.status_code == 200
    assert full.content == b"fixture-png-bytes"
    assert full.headers["accept-ranges"] == "bytes"
    assert full.headers["etag"] == f'"{sha256}"'
    assert full.headers["content-disposition"].startswith("inline;")
    assert full.headers["content-type"] == "image/png"
    assert full.headers["cross-origin-resource-policy"] == "same-origin"

    head = client.head(f"/api/v1/blobs/{sha256}")
    assert head.status_code == 200
    assert head.content == b""
    assert head.headers["content-length"] == str(len(b"fixture-png-bytes"))

    bounded = client.get(f"/api/v1/blobs/{sha256}", headers={"range": "bytes=1-4"})
    assert bounded.status_code == 206
    assert bounded.content == b"ixtu"
    assert bounded.headers["content-range"] == "bytes 1-4/17"
    assert bounded.headers["content-length"] == "4"

    open_ended = client.get(f"/api/v1/blobs/{sha256}", headers={"range": "bytes=14-"})
    assert open_ended.status_code == 206
    assert open_ended.content == b"tes"

    suffix = client.get(f"/api/v1/blobs/{sha256}", headers={"range": "bytes=-5"})
    assert suffix.status_code == 206
    assert suffix.content == b"bytes"


@pytest.mark.parametrize("range_header", ["items=0-2", "bytes=99-", "bytes=4-2", "bytes=0-1,4-5"])
def test_blob_stream_rejects_invalid_or_multiple_ranges(
    blob_catalog: tuple[TestClient, ArchiveLayout, dict[str, str]], range_header: str
) -> None:
    client, _layout, hashes = blob_catalog
    response = client.get(f"/api/v1/blobs/{hashes['photo.png']}", headers={"range": range_header})
    assert response.status_code == 416
    assert response.headers["content-range"] == "bytes */17"
    assert response.headers["accept-ranges"] == "bytes"


def test_dangerous_extensions_are_forced_to_download_and_names_are_sanitized(
    blob_catalog: tuple[TestClient, ArchiveLayout, dict[str, str]],
) -> None:
    client, _layout, hashes = blob_catalog
    html = client.get(f"/api/v1/blobs/{hashes['unsafe.html']}")
    key = client.get(f"/api/v1/blobs/{hashes['private.pem']}")
    renamed = client.get(
        f"/api/v1/blobs/{hashes['photo.png']}", params={"filename": "../../renamed.html"}
    )

    for response in (html, key, renamed):
        assert response.status_code == 200
        assert response.headers["content-disposition"].startswith("attachment;")
        assert response.headers["content-type"] == "application/octet-stream"
        assert "../" not in response.headers["content-disposition"]
    assert "renamed.html" in renamed.headers["content-disposition"]


def test_blob_route_never_falls_back_to_original_source_paths(
    blob_catalog: tuple[TestClient, ArchiveLayout, dict[str, str]],
) -> None:
    client, layout, hashes = blob_catalog
    sha256 = hashes["photo.png"]
    layout.blob_path(sha256).unlink()

    response = client.get(f"/api/v1/blobs/{sha256}")

    assert response.status_code == 404
    assert "unavailable" in response.json()["detail"]


def test_blob_route_rejects_a_symlinked_parent_shard(
    blob_catalog: tuple[TestClient, ArchiveLayout, dict[str, str]],
) -> None:
    client, layout, hashes = blob_catalog
    sha256 = hashes["photo.png"]
    blob_path = layout.blob_path(sha256)
    relocated_shard = layout.root.parent / "relocated-blob-shard"
    blob_path.parent.rename(relocated_shard)
    blob_path.parent.symlink_to(relocated_shard, target_is_directory=True)

    response = client.get(f"/api/v1/blobs/{sha256}")

    assert response.status_code == 409
    assert response.json()["detail"] == "Catalogued blob path contains an unsafe directory"


def test_blob_route_is_session_protected_and_rejects_non_digest_paths(tmp_path: Path) -> None:
    root = tmp_path / "meta-backup-protected-blob"
    (root / "package-a").mkdir(parents=True)
    app = create_app(root, access_token="a" * 32)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        protected = client.get("/api/v1/blobs/" + "0" * 64)
    assert protected.status_code == 401

    app_disabled = create_app(root, security_disabled=True)
    with TestClient(app_disabled) as client:
        invalid = client.get("/api/v1/blobs/not-a-digest")
    assert invalid.status_code == 404
