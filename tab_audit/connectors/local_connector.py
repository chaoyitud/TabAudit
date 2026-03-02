from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from tab_audit.utils.hashing import hash_file


def _metadata_for_path(path: Path, source_id: str) -> dict:
    return {
        "source": "local",
        "retrieved_at": datetime.now(UTC).isoformat(),
        "file_hash": hash_file(path),
        "license": None,
        "source_id": source_id,
    }


def fetch_local_dataset(path: str, cache_dir: str, **_: object) -> dict:
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Local dataset not found: {p}")
    return {
        "data_path": str(p.resolve()),
        "metadata": _metadata_for_path(p, str(p)),
    }


def list_local_datasets(path: str, pattern: str = "*.csv") -> list[dict]:
    root = Path(path)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Local directory not found: {root}")

    files = sorted([p for p in root.rglob(pattern) if p.is_file()])
    payloads = []
    for f in files:
        payloads.append(
            {
                "path": str(f.resolve()),
                "name": f.stem,
                "metadata": _metadata_for_path(f, str(f)),
            }
        )
    return payloads
