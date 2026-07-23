from __future__ import annotations

import base64
import json
from typing import Any


class InvalidCursor(ValueError):
    """Raised when an opaque API cursor cannot be trusted or reused."""


def encode_cursor(kind: str, **values: str | int | None) -> str:
    payload: dict[str, Any] = {"v": 1, "kind": kind, **values}
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(encoded).rstrip(b"=").decode("ascii")


def decode_cursor(token: str, *, kind: str) -> dict[str, Any]:
    if not token or len(token) > 2048:
        raise InvalidCursor("Invalid pagination cursor")
    try:
        padding = "=" * (-len(token) % 4)
        decoded = base64.b64decode(token + padding, altchars=b"-_", validate=True)
        payload = json.loads(decoded)
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidCursor("Invalid pagination cursor") from exc
    if not isinstance(payload, dict) or payload.get("v") != 1 or payload.get("kind") != kind:
        raise InvalidCursor("Pagination cursor does not match this resource")
    return payload


def require_cursor_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise InvalidCursor("Invalid pagination cursor")
    return value


def require_cursor_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise InvalidCursor("Invalid pagination cursor")
    return value
