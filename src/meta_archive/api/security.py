from __future__ import annotations

import hmac
import secrets
from collections.abc import Awaitable, Callable
from threading import Lock
from typing import Any
from urllib.parse import urlsplit

from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

SESSION_COOKIE = "meta_archive_session"
CSRF_COOKIE = "meta_archive_csrf"


class LocalSecurity:
    def __init__(self, access_token: str | None = None, *, disabled: bool = False) -> None:
        self.access_token = access_token or secrets.token_urlsafe(32)
        self.session_token = secrets.token_urlsafe(48)
        self.csrf_token = secrets.token_urlsafe(32)
        self.disabled = disabled
        self._access_token_available = True
        self._access_token_lock = Lock()

    def consume_access_token(self, candidate: str) -> bool:
        """Atomically exchange the startup token exactly once.

        The URL carrying this token can remain in browser history.  Consuming it prevents a
        copied or recovered URL from minting another session later in the same server process.
        """

        with self._access_token_lock:
            if not self._access_token_available or not hmac.compare_digest(
                candidate, self.access_token
            ):
                return False
            self._access_token_available = False
            return True

    def verify_session(self, candidate: str | None) -> bool:
        return self.disabled or (
            candidate is not None and hmac.compare_digest(candidate, self.session_token)
        )

    def verify_csrf(self, candidate: str | None) -> bool:
        return self.disabled or (
            candidate is not None and hmac.compare_digest(candidate, self.csrf_token)
        )


def require_session(request: Request) -> None:
    security: LocalSecurity = request.app.state.security
    if not security.verify_session(request.cookies.get(SESSION_COOKIE)):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Local session required"
        )


def _has_strict_same_origin(request: Request) -> bool:
    """Accept exactly one browser ``Origin`` matching this request's origin.

    Mutations intentionally require ``Origin`` even from loopback clients.  We do not fall
    back to ``Referer`` or ``Sec-Fetch-Site`` because either fallback would make a missing
    standard browser signal ambiguous.  Non-browser local clients can send the same explicit
    ``Origin`` header after obtaining the session and CSRF tokens.
    """

    origins = request.headers.getlist("origin")
    hosts = request.headers.getlist("host")
    if len(origins) != 1 or len(hosts) != 1:
        return False

    origin = origins[0]
    request_host = hosts[0]
    if not origin or origin != origin.strip():
        return False

    try:
        parsed = urlsplit(origin)
        hostname = parsed.hostname
    except ValueError:
        return False

    return (
        parsed.scheme in {"http", "https"}
        and parsed.scheme == request.url.scheme
        and parsed.netloc == request_host
        and hostname is not None
        and parsed.username is None
        and parsed.password is None
        and parsed.path == ""
        and parsed.query == ""
        and parsed.fragment == ""
    )


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app: Any,
        *,
        security: LocalSecurity,
        acceptance_read_only: bool = False,
    ) -> None:
        super().__init__(app)
        self.security = security
        self.acceptance_read_only = acceptance_read_only

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if self.acceptance_read_only and request.method not in {"GET", "HEAD"}:
            response = Response(
                "Acceptance server permits only GET and HEAD requests",
                status_code=405,
                headers={"Allow": "GET, HEAD"},
            )
        elif not self.security.disabled and request.method not in {"GET", "HEAD", "OPTIONS"}:
            session = request.cookies.get(SESSION_COOKIE)
            csrf_header = request.headers.get("x-csrf-token")
            if not self.security.verify_session(session) or not self.security.verify_csrf(
                csrf_header
            ):
                response = Response(
                    "Mutation requires a valid local session and CSRF token", status_code=403
                )
            elif not _has_strict_same_origin(request):
                response = Response("Origin rejected", status_code=403)
            else:
                response = await call_next(request)
        else:
            response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; "
            "style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; "
            "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
        )
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=(), payment=(), usb=()"
        )
        response.headers["Cache-Control"] = "no-store"
        return response
