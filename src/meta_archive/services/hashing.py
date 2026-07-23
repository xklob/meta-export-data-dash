from __future__ import annotations

import hashlib
from pathlib import Path

HASH_BLOCK_SIZE = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        buffer = bytearray(HASH_BLOCK_SIZE)
        view = memoryview(buffer)
        while read := handle.readinto(view):
            digest.update(view[:read])
    return digest.hexdigest()
