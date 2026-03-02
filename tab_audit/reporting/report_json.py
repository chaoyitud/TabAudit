from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from tab_audit.io.cache import ensure_dir, write_json


def write_dataset_report_json(out_dir: str | Path, dataset_slug: str, payload: dict[str, Any]) -> Path:
    folder = ensure_dir(Path(out_dir) / dataset_slug)
    path = folder / "report.json"
    write_json(path, payload)
    return path


def write_summary_csv(out_dir: str | Path, rows: list[dict[str, Any]]) -> Path:
    out = ensure_dir(out_dir)
    path = out / "summary.csv"
    if not rows:
        return path

    fields = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: "" if v is None else str(v) for k, v in row.items()} for row in rows])
    return path
