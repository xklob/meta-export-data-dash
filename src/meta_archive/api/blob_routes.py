from __future__ import annotations

import os
import re
from collections.abc import Iterator
from pathlib import Path, PurePath
from typing import BinaryIO
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import Response, StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from meta_archive.api.security import require_session
from meta_archive.models import AttachmentOccurrence, Blob, SourceFile
from meta_archive.paths import UnsafeArchivePathError, open_regular_file_beneath

blob_router = APIRouter(prefix="/api/v1/blobs", dependencies=[Depends(require_session)])

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DANGEROUS_EXTENSIONS = frozenset(
    {
        ".7z",
        ".apk",
        ".app",
        ".bat",
        ".bash",
        ".bz2",
        ".cer",
        ".cjs",
        ".cmd",
        ".com",
        ".crt",
        ".dll",
        ".docm",
        ".dylib",
        ".exe",
        ".fish",
        ".gz",
        ".htm",
        ".html",
        ".jar",
        ".js",
        ".key",
        ".mjs",
        ".msi",
        ".p12",
        ".pem",
        ".pfx",
        ".php",
        ".pptm",
        ".ps1",
        ".rar",
        ".sh",
        ".so",
        ".svg",
        ".tar",
        ".tgz",
        ".xhtml",
        ".xlsm",
        ".xz",
        ".zip",
        ".zsh",
    }
)
_DANGEROUS_MEDIA_TYPES = frozenset(
    {
        "application/ecmascript",
        "application/java-archive",
        "application/javascript",
        "application/vnd.microsoft.portable-executable",
        "application/x-7z-compressed",
        "application/x-executable",
        "application/x-httpd-php",
        "application/x-rar-compressed",
        "application/x-sh",
        "application/x-tar",
        "application/xhtml+xml",
        "application/zip",
        "image/svg+xml",
        "text/html",
        "text/javascript",
    }
)
_INLINE_MEDIA_TYPES = frozenset(
    {
        "audio/aac",
        "audio/flac",
        "audio/m4a",
        "audio/mp4",
        "audio/mpeg",
        "audio/ogg",
        "audio/wav",
        "audio/webm",
        "image/avif",
        "image/bmp",
        "image/gif",
        "image/jpeg",
        "image/png",
        "image/webp",
        "video/mp4",
        "video/mpeg",
        "video/ogg",
        "video/quicktime",
        "video/webm",
    }
)


def _safe_filename(value: str) -> str:
    basename = PurePath(value.replace("\\", "/")).name
    cleaned = "".join(
        character for character in basename if character >= " " and character != "\x7f"
    )
    return cleaned.strip().strip(".")[:240] or "attachment"


def _content_disposition(mode: str, filename: str) -> str:
    safe = _safe_filename(filename)
    ascii_name = safe.encode("ascii", "ignore").decode("ascii") or "attachment"
    ascii_name = ascii_name.replace('"', "'").replace("\\", "_")
    return f"{mode}; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(safe)}"


def _normalized_media_type(value: str | None) -> str:
    return (value or "application/octet-stream").split(";", 1)[0].strip().casefold()


def _is_dangerous(*, media_types: set[str], extensions: set[str]) -> bool:
    if extensions & _DANGEROUS_EXTENSIONS:
        return True
    return bool(media_types & _DANGEROUS_MEDIA_TYPES) or any(
        media_type.startswith("application/x-")
        and any(token in media_type for token in ("executable", "archive", "compressed"))
        for media_type in media_types
    )


def _parse_range(value: str, size: int) -> tuple[int, int]:
    try:
        unit, specification = value.split("=", 1)
        if unit.strip().casefold() != "bytes" or "," in specification:
            raise ValueError
        first, last = (part.strip() for part in specification.split("-", 1))
        if not first:
            suffix = int(last)
            if suffix <= 0 or size <= 0:
                raise ValueError
            start = max(size - suffix, 0)
            return start, size - 1
        start = int(first)
        if start < 0 or start >= size:
            raise ValueError
        end = size - 1 if not last else min(int(last), size - 1)
        if end < start:
            raise ValueError
        return start, end
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_416_RANGE_NOT_SATISFIABLE,
            detail="Requested byte range is not satisfiable",
            headers={"Content-Range": f"bytes */{size}", "Accept-Ranges": "bytes"},
        ) from exc


def _file_chunks(handle: BinaryIO, *, start: int, length: int) -> Iterator[bytes]:
    remaining = length
    with handle:
        handle.seek(start)
        while remaining:
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


@blob_router.api_route("/{sha256}", methods=["GET", "HEAD"])
def stream_blob(
    request: Request,
    sha256: str,
    filename: str | None = Query(default=None, max_length=300),
) -> Response:
    if not _SHA256.fullmatch(sha256):
        raise HTTPException(status_code=404, detail="Catalogued blob not found")
    with Session(request.app.state.engine) as session:
        blob = session.get(Blob, sha256)
        if blob is None or blob.state not in {"catalogued", "linked", "materialized"}:
            raise HTTPException(status_code=404, detail="Catalogued blob not found")
        occurrence_names = list(
            session.scalars(
                select(AttachmentOccurrence.filename)
                .where(
                    AttachmentOccurrence.blob_sha256 == sha256,
                    AttachmentOccurrence.filename.is_not(None),
                    AttachmentOccurrence.deleted_at.is_(None),
                )
                .order_by(AttachmentOccurrence.id)
            )
        )
        source_metadata = session.execute(
            select(SourceFile.extension, SourceFile.media_type).where(SourceFile.sha256 == sha256)
        ).all()

    path = request.app.state.layout.blob_path(sha256)
    expected_relative = path.relative_to(request.app.state.layout.root).as_posix()
    if blob.relative_path is not None and blob.relative_path != expected_relative:
        raise HTTPException(
            status_code=409, detail="Catalogued blob path does not match its record"
        )
    try:
        handle = open_regular_file_beneath(path, request.app.state.layout.root)
    except UnsafeArchivePathError as exc:
        if isinstance(exc.__cause__, FileNotFoundError):
            raise HTTPException(
                status_code=404, detail="Catalogued blob bytes are unavailable"
            ) from exc
        raise HTTPException(
            status_code=409, detail="Catalogued blob path contains an unsafe directory"
        ) from exc
    path_stat = os.fstat(handle.fileno())
    if path_stat.st_size != blob.size_bytes:
        handle.close()
        raise HTTPException(
            status_code=409, detail="Catalogued blob size does not match its record"
        )

    selected_name = _safe_filename(
        filename
        or next((name for name in occurrence_names if isinstance(name, str) and name), "")
        or f"{sha256}{blob.extension or ''}"
    )
    extensions = {
        extension.casefold() if extension.startswith(".") else f".{extension.casefold()}"
        for extension, _media_type in source_metadata
        if extension
    }
    selected_extension = Path(selected_name).suffix.casefold()
    if selected_extension:
        extensions.add(selected_extension)
    if blob.extension:
        extensions.add(
            blob.extension.casefold()
            if blob.extension.startswith(".")
            else f".{blob.extension.casefold()}"
        )
    media_types = {
        _normalized_media_type(media_type)
        for _extension, media_type in source_metadata
        if media_type
    }
    effective_media_type = _normalized_media_type(blob.media_type)
    media_types.add(effective_media_type)
    dangerous = _is_dangerous(media_types=media_types, extensions=extensions)
    inline = not dangerous and effective_media_type in _INLINE_MEDIA_TYPES
    disposition = _content_disposition("inline" if inline else "attachment", selected_name)
    response_media_type = effective_media_type if inline else "application/octet-stream"
    size = blob.size_bytes
    range_header = request.headers.get("range")
    start = 0
    end = size - 1
    response_status = status.HTTP_200_OK
    try:
        if range_header is not None:
            start, end = _parse_range(range_header, size)
            response_status = status.HTTP_206_PARTIAL_CONTENT
    except BaseException:
        handle.close()
        raise
    length = max(end - start + 1, 0)
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Disposition": disposition,
        "Content-Length": str(length),
        "ETag": f'"{sha256}"',
    }
    if response_status == status.HTTP_206_PARTIAL_CONTENT:
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    if request.method == "HEAD":
        handle.close()
        return Response(
            status_code=response_status,
            media_type=response_media_type,
            headers=headers,
        )
    return StreamingResponse(
        _file_chunks(handle, start=start, length=length),
        status_code=response_status,
        media_type=response_media_type,
        headers=headers,
    )
