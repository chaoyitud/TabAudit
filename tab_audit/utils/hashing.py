from __future__ import annotations

import hashlib
from pathlib import Path


def hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def hash_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    p = Path(path)
    digest = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_slug(text: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in text)
    return "_".join(filter(None, safe.split("_")))
