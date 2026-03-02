from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

SUPPORTED_SUFFIXES = {".csv", ".parquet", ".json", ".jsonl", ".tsv"}
ENCODING_CANDIDATES = ["utf-8", "utf-8-sig", "latin-1"]


def infer_textish(series: pd.Series) -> bool:
    if series.dtype != "object":
        return False
    sampled = series.dropna().astype(str).head(200)
    if sampled.empty:
        return False
    avg_len = sampled.map(len).mean()
    return avg_len >= 30


def _sniff_sep(path: Path, encoding: str) -> str:
    with path.open("r", encoding=encoding, errors="replace") as f:
        sample = f.read(8192)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        return ","


def _read_csv_with_fallback(
    path: Path,
    max_rows: int | None = None,
    chunksize: int = 100_000,
    sep: str | None = None,
) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ENCODING_CANDIDATES:
        try:
            actual_sep = sep or _sniff_sep(path, encoding)
            if max_rows is not None:
                return pd.read_csv(path, nrows=max_rows, encoding=encoding, sep=actual_sep, on_bad_lines="skip")
            chunks: list[pd.DataFrame] = []
            for chunk in pd.read_csv(
                path,
                chunksize=chunksize,
                encoding=encoding,
                sep=actual_sep,
                on_bad_lines="skip",
            ):
                chunks.append(chunk)
            return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        except Exception as exc:  # try next encoding
            last_error = exc
    raise ValueError(f"Failed to read CSV with supported encodings: {path}") from last_error


def read_tabular(
    path: str | Path,
    max_rows: int | None = None,
    chunksize: int = 100_000,
    sep: str | None = None,
) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")
    suffix = p.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported format {suffix}. Supported: {sorted(SUPPORTED_SUFFIXES)}")

    if suffix in {".csv", ".tsv"}:
        forced_sep = "\t" if suffix == ".tsv" and sep is None else sep
        return _read_csv_with_fallback(p, max_rows=max_rows, chunksize=chunksize, sep=forced_sep)
    if suffix == ".parquet":
        df = pd.read_parquet(p)
        return df.head(max_rows) if max_rows is not None else df
    if suffix == ".jsonl":
        df = pd.read_json(p, lines=True)
        return df.head(max_rows) if max_rows is not None else df
    df = pd.read_json(p)
    return df.head(max_rows) if max_rows is not None else df
