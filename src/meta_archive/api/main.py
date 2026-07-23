from __future__ import annotations

import mimetypes
import os
import secrets
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from meta_archive import __version__
from meta_archive.api.blob_routes import blob_router
from meta_archive.api.conversation_correction_routes import conversation_correction_router
from meta_archive.api.conversation_routes import conversation_router
from meta_archive.api.deletion_routes import deletion_router
from meta_archive.api.derivative_routes import derivative_router
from meta_archive.api.enrichment_routes import enrichment_router
from meta_archive.api.media_routes import media_router
from meta_archive.api.routes import public_router, router
from meta_archive.api.search_routes import search_router
from meta_archive.api.security import (
    CSRF_COOKIE,
    SESSION_COOKIE,
    LocalSecurity,
    SecurityHeadersMiddleware,
    require_session,
)
from meta_archive.api.stage4_routes import stage4_router
from meta_archive.database import (
    CatalogMigrationError,
    create_catalog_engine,
    require_current_migration,
    run_migrations,
)
from meta_archive.paths import (
    ArchiveLayout,
    UnsafeArchivePathError,
    open_regular_file_beneath,
    regular_file_stat_beneath,
)
from meta_archive.services.acceptance_identity import (
    backend_source_digest,
    catalog_source_revision,
    frontend_manifest_digest,
)
from meta_archive.services.deletion import recover_pending_hard_purges
from meta_archive.services.lance_read_only import require_existing_lance_manifest
from meta_archive.services.source_import_lock import source_import_publication_lock

_MAX_FRONTEND_INDEX_BYTES = 8 * 1024 * 1024
_MAX_FRONTEND_ASSET_BYTES = 64 * 1024 * 1024


def _secure_frontend_payload(
    project_root: Path,
    static_root: Path,
    relative_path: str,
    *,
    max_bytes: int,
) -> bytes:
    """Read one bounded frontend file without following any project-tree symlink."""

    relative = PurePosixPath(relative_path)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} or "\\" in part for part in relative.parts)
    ):
        raise UnsafeArchivePathError("Invalid frontend asset path")
    with open_regular_file_beneath(static_root.joinpath(*relative.parts), project_root) as handle:
        opened = os.fstat(handle.fileno())
        if opened.st_size > max_bytes:
            raise UnsafeArchivePathError("Frontend asset exceeds its local serving limit")
        payload = handle.read(max_bytes + 1)
        final = os.fstat(handle.fileno())
    if (
        len(payload) > max_bytes
        or len(payload) != opened.st_size
        or (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        != (
            final.st_dev,
            final.st_ino,
            final.st_size,
            final.st_mtime_ns,
            final.st_ctime_ns,
        )
    ):
        raise UnsafeArchivePathError("Frontend asset changed while it was read")
    return payload


def create_app(
    archive_root: Path,
    *,
    access_token: str | None = None,
    security_disabled: bool = False,
    acceptance_read_only: bool = False,
) -> FastAPI:
    layout = ArchiveLayout.discover(archive_root)
    static_root = Path(__file__).resolve().parents[3] / "web" / "dist"
    project_root = Path(__file__).resolve().parents[3]
    acceptance_nonce: str | None = None
    if acceptance_read_only:
        regular_file_stat_beneath(layout.database, layout.root)
        engine = create_catalog_engine(
            layout.database,
            read_only=True,
            archive_root=layout.root,
        )
        try:
            require_current_migration(engine)
            try:
                require_existing_lance_manifest(layout)
                frontend_manifest_digest(static_root, project_root=project_root)
            except RuntimeError as exc:
                raise CatalogMigrationError(
                    "Acceptance mode requires a populated safe Lance manifest "
                    "and safe frontend manifest"
                ) from exc
            acceptance_nonce = secrets.token_hex(32)
        except BaseException:
            engine.dispose()
            raise
    else:
        with source_import_publication_lock(layout):
            layout.ensure_runtime_directories()
            run_migrations(layout.database)
            engine = create_catalog_engine(layout.database)
        recover_pending_hard_purges(layout, engine)
    security = LocalSecurity(access_token, disabled=security_disabled)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        engine.dispose()

    app = FastAPI(
        title="Meta Archive Explorer",
        version=__version__,
        docs_url="/api/docs" if security_disabled else None,
        redoc_url=None,
        openapi_url="/api/openapi.json" if security_disabled else None,
        lifespan=lifespan,
    )
    app.state.layout = layout
    app.state.engine = engine
    app.state.security = security
    app.state.acceptance_read_only = acceptance_read_only
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=[
            "127.0.0.1",
            "localhost",
            *(["testserver"] if security_disabled else []),
        ],
    )
    app.add_middleware(
        SecurityHeadersMiddleware,
        security=security,
        acceptance_read_only=acceptance_read_only,
    )

    @app.get("/auth", include_in_schema=False)
    def authenticate(
        token: str = Query(min_length=20),
    ) -> RedirectResponse:
        if not security.consume_access_token(token):
            raise HTTPException(status_code=401, detail="Invalid local access token")
        response = RedirectResponse(url="/", status_code=303)
        response.set_cookie(
            SESSION_COOKIE,
            security.session_token,
            httponly=True,
            samesite="strict",
            secure=False,
            path="/",
        )
        response.set_cookie(
            CSRF_COOKIE,
            security.csrf_token,
            httponly=False,
            samesite="strict",
            secure=False,
            path="/",
        )
        return response

    app.include_router(public_router)
    app.include_router(router)
    app.include_router(blob_router)
    app.include_router(derivative_router)
    app.include_router(conversation_correction_router)
    app.include_router(conversation_router)
    app.include_router(deletion_router)
    app.include_router(media_router)
    app.include_router(search_router)
    app.include_router(enrichment_router)
    app.include_router(stage4_router)

    if acceptance_nonce is not None:
        identity_lock = threading.Lock()

        @app.api_route(
            "/api/v1/acceptance/identity",
            methods=["GET", "HEAD"],
            dependencies=[Depends(require_session)],
            include_in_schema=False,
        )
        def acceptance_server_identity() -> dict[str, str]:
            """Rebind evidence to the current catalog, build, backend, and this process."""

            with identity_lock:
                return {
                    "catalog_source_revision": catalog_source_revision(layout, engine),
                    "backend_version": __version__,
                    "backend_source_sha256": backend_source_digest(project_root),
                    "frontend_manifest_sha256": frontend_manifest_digest(
                        static_root,
                        project_root=project_root,
                    ),
                    "server_nonce": acceptance_nonce,
                }

    assets = static_root / "assets"
    if acceptance_read_only:

        @app.get("/assets/{asset_path:path}", include_in_schema=False)
        def acceptance_asset(asset_path: str) -> Response:
            try:
                payload = _secure_frontend_payload(
                    project_root,
                    static_root,
                    f"assets/{asset_path}",
                    max_bytes=_MAX_FRONTEND_ASSET_BYTES,
                )
            except UnsafeArchivePathError as exc:
                raise HTTPException(status_code=404, detail="Frontend asset not found") from exc
            media_type, content_encoding = mimetypes.guess_type(asset_path)
            headers = {"Content-Encoding": content_encoding} if content_encoding else None
            return Response(
                payload,
                media_type=media_type or "application/octet-stream",
                headers=headers,
            )

    elif assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> Response:
        if path.startswith("api/"):
            raise HTTPException(status_code=404)
        try:
            payload = _secure_frontend_payload(
                project_root,
                static_root,
                "index.html",
                max_bytes=_MAX_FRONTEND_INDEX_BYTES,
            )
        except UnsafeArchivePathError as exc:
            raise HTTPException(status_code=503, detail="Frontend has not been built") from exc
        return Response(payload, media_type="text/html")

    return app


def generate_access_token() -> str:
    return secrets.token_urlsafe(32)
