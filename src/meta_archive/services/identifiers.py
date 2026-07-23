from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_id(namespace: str, *parts: object) -> str:
    digest = hashlib.sha256()
    digest.update(namespace.encode("utf-8"))
    for part in parts:
        digest.update(b"\x1f")
        digest.update(str(part).encode("utf-8", errors="surrogatepass"))
    return digest.hexdigest()


def canonical_fingerprint(namespace: str, payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8", errors="surrogatepass")
    digest = hashlib.sha256()
    digest.update(namespace.encode("utf-8"))
    digest.update(b"\x1f")
    digest.update(encoded)
    return digest.hexdigest()
