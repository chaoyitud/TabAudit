from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml


def _dedupe_keep_order(items: Iterable[Any]) -> list[Any]:
    seen: set[Any] = set()
    out: list[Any] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _strip_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def _tokenize_text_values(text: str) -> list[str]:
    tokens: list[str] = []
    for raw_line in text.splitlines():
        line = _strip_comment(raw_line)
        if not line:
            continue
        tokens.extend(tok for tok in re.split(r"[\s,;]+", line) if tok)
    return tokens


def _load_structured_file(path: str | Path) -> Any:
    p = Path(path)
    suffix = p.suffix.lower()
    text = p.read_text(encoding="utf-8")
    if suffix == ".json":
        return json.loads(text)
    if suffix in {".yaml", ".yml"}:
        return yaml.safe_load(text)
    return text


def _extract_openml_ids_from_obj(obj: Any) -> list[int]:
    ids: list[int] = []
    if obj is None:
        return ids
    if isinstance(obj, int):
        return [obj]
    if isinstance(obj, str):
        for tok in _tokenize_text_values(obj):
            ids.append(int(tok))
        return ids
    if isinstance(obj, list):
        for item in obj:
            ids.extend(_extract_openml_ids_from_obj(item))
        return ids
    if isinstance(obj, dict):
        if "dataset_id" in obj:
            ids.extend(_extract_openml_ids_from_obj(obj["dataset_id"]))
        if "dataset_ids" in obj:
            ids.extend(_extract_openml_ids_from_obj(obj["dataset_ids"]))
        if "openml" in obj:
            ids.extend(_extract_openml_ids_from_obj(obj["openml"]))
        if "datasets" in obj and isinstance(obj["datasets"], list):
            for entry in obj["datasets"]:
                if isinstance(entry, dict) and entry.get("source") == "openml":
                    ids.extend(_extract_openml_ids_from_obj(entry.get("dataset_id")))
        return ids
    raise ValueError(f"Unsupported OpenML dataset-id input type: {type(obj).__name__}")


def _extract_kaggle_specs_from_obj(obj: Any) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    if obj is None:
        return specs
    if isinstance(obj, str):
        for line in obj.splitlines():
            entry = _strip_comment(line)
            if not entry:
                continue
            if ":" not in entry:
                raise ValueError(f"Invalid Kaggle dataset entry '{entry}'. Expected slug:file")
            slug, file = entry.split(":", 1)
            specs.append((slug.strip(), file.strip()))
        return specs
    if isinstance(obj, list):
        for item in obj:
            specs.extend(_extract_kaggle_specs_from_obj(item))
        return specs
    if isinstance(obj, dict):
        if "slug" in obj and "file" in obj:
            specs.append((str(obj["slug"]).strip(), str(obj["file"]).strip()))
        if "datasets" in obj:
            specs.extend(_extract_kaggle_specs_from_obj(obj["datasets"]))
        if "kaggle" in obj:
            specs.extend(_extract_kaggle_specs_from_obj(obj["kaggle"]))
        if "datasets" in obj and isinstance(obj["datasets"], list):
            for entry in obj["datasets"]:
                if isinstance(entry, dict) and entry.get("source") == "kaggle":
                    slug = entry.get("slug")
                    file = entry.get("file")
                    if slug and file:
                        specs.append((str(slug).strip(), str(file).strip()))
        return specs
    raise ValueError(f"Unsupported Kaggle dataset input type: {type(obj).__name__}")


def parse_openml_dataset_ids(
    dataset_ids: list[int] | None = None,
    dataset_id_lists: list[str] | None = None,
    dataset_ids_file: str | None = None,
) -> list[int]:
    out: list[int] = []
    out.extend(dataset_ids or [])
    for token_blob in dataset_id_lists or []:
        out.extend(_extract_openml_ids_from_obj(token_blob))
    if dataset_ids_file:
        out.extend(_extract_openml_ids_from_obj(_load_structured_file(dataset_ids_file)))
    parsed = [int(x) for x in out]
    return _dedupe_keep_order(parsed)


def parse_kaggle_dataset_specs(
    datasets: list[str] | None = None,
    datasets_file: str | None = None,
) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    out.extend(_extract_kaggle_specs_from_obj(datasets or []))
    if datasets_file:
        out.extend(_extract_kaggle_specs_from_obj(_load_structured_file(datasets_file)))
    normalized = [(slug.strip(), file.strip()) for slug, file in out if slug and file]
    return _dedupe_keep_order(normalized)


def _extract_local_paths_from_obj(obj: Any) -> list[str]:
    paths: list[str] = []
    if obj is None:
        return paths
    if isinstance(obj, str):
        tokens = _tokenize_text_values(obj)
        if not tokens and obj.strip():
            return [obj.strip()]
        return [tok.strip() for tok in tokens if tok.strip()]
    if isinstance(obj, list):
        for item in obj:
            paths.extend(_extract_local_paths_from_obj(item))
        return paths
    if isinstance(obj, dict):
        if "path" in obj:
            paths.extend(_extract_local_paths_from_obj(obj["path"]))
        if "paths" in obj:
            paths.extend(_extract_local_paths_from_obj(obj["paths"]))
        if "local_files" in obj:
            paths.extend(_extract_local_paths_from_obj(obj["local_files"]))
        if "local_dirs" in obj:
            paths.extend(_extract_local_paths_from_obj(obj["local_dirs"]))
        if "datasets" in obj and isinstance(obj["datasets"], list):
            for entry in obj["datasets"]:
                if isinstance(entry, dict) and entry.get("source") in {"local", "local_dir"} and entry.get("path"):
                    paths.append(str(entry["path"]).strip())
        return paths
    raise ValueError(f"Unsupported local path input type: {type(obj).__name__}")


def parse_local_file_paths(
    local_files: list[str] | None = None,
    local_file_lists: list[str] | None = None,
    local_files_file: str | None = None,
) -> list[str]:
    out: list[str] = []
    out.extend(_extract_local_paths_from_obj(local_files or []))
    for token_blob in local_file_lists or []:
        out.extend(_extract_local_paths_from_obj(token_blob))
    if local_files_file:
        out.extend(_extract_local_paths_from_obj(_load_structured_file(local_files_file)))
    return _dedupe_keep_order([p for p in out if p])


def parse_local_dir_paths(
    local_dirs: list[str] | None = None,
    local_dirs_file: str | None = None,
) -> list[str]:
    out: list[str] = []
    out.extend(_extract_local_paths_from_obj(local_dirs or []))
    if local_dirs_file:
        out.extend(_extract_local_paths_from_obj(_load_structured_file(local_dirs_file)))
    return _dedupe_keep_order([p for p in out if p])
