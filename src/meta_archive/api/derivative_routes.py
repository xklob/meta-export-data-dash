from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Iterator
from typing import BinaryIO

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from meta_archive.api.security import require_session
from meta_archive.models import MediaDerivative
from meta_archive.paths import UnsafeArchivePathError, open_regular_file_beneath
from meta_archive.services.media_derivatives import SPECS, ready_derivative_path

derivative_router = APIRouter(prefix="/api/v1/derivatives", dependencies=[Depends(require_session)])
_FIXED_ID = re.compile(r"[0-9a-f]{64}")


def _chunks(handle: BinaryIO) -> Iterator[bytes]:
    with handle:
        while chunk := handle.read(1024 * 1024):
            yield chunk


def _sha256(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    while chunk := handle.read(1024 * 1024):
        digest.update(chunk)
    handle.seek(0)
    return digest.hexdigest()


@derivative_router.api_route("/{derivative_id}", methods=["GET", "HEAD"])
def stream_derivative(request: Request, derivative_id: str) -> Response:
    if not _FIXED_ID.fullmatch(derivative_id):
        raise HTTPException(status_code=404, detail="Media derivative not found")
    with Session(request.app.state.engine) as session:
        row = session.get(MediaDerivative, derivative_id)
        if row is None or row.kind not in SPECS:
            raise HTTPException(status_code=404, detail="Media derivative not found")
        path = ready_derivative_path(request.app.state.layout, row)
        if path is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Media derivative bytes are unavailable or do not match the catalog",
            )
        media_type = SPECS[row.kind].media_type
        size_bytes = row.size_bytes
        output_sha256 = row.output_sha256
        kind = row.kind
    if size_bytes is None or output_sha256 is None:
        raise HTTPException(status_code=409, detail="Media derivative record is incomplete")
    try:
        handle = open_regular_file_beneath(path, request.app.state.layout.root)
    except UnsafeArchivePathError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Media derivative bytes are unavailable or unsafe",
        ) from exc
    if os.fstat(handle.fileno()).st_size != size_bytes or _sha256(handle) != output_sha256:
        handle.close()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Media derivative bytes do not match the catalog",
        )
    headers = {
        "Cache-Control": "private, max-age=31536000, immutable",
        "Content-Disposition": f'inline; filename="{derivative_id}-{kind}.{path.suffix[1:]}"',
        "Content-Length": str(size_bytes),
        "ETag": f'"{output_sha256}"',
    }
    if request.method == "HEAD":
        handle.close()
        return Response(media_type=media_type, headers=headers)
    return StreamingResponse(_chunks(handle), media_type=media_type, headers=headers)
