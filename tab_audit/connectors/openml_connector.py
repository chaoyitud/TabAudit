from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from tab_audit.io.cache import ensure_dir, read_json, write_json


def fetch_openml_dataset(dataset_id: int, cache_dir: str, **_: object) -> dict:
    try:
        import openml
    except ImportError as exc:
        raise RuntimeError("openml extra not installed. Use pip install -e '.[openml]'") from exc

    # Force openml-python internal cache into project-writable space.
    openml_client_cache = Path(cache_dir) / "openml_client"
    openml_client_cache.mkdir(parents=True, exist_ok=True)
    openml.config.set_root_cache_directory(str(openml_client_cache))

    cache_path = ensure_dir(Path(cache_dir) / "openml" / str(dataset_id))
    data_path = cache_path / "dataset.parquet"
    meta_path = cache_path / "metadata.json"

    if data_path.exists() and meta_path.exists():
        cached = read_json(meta_path)
        return {
            "data_path": str(data_path),
            "default_target": cached.get("default_target"),
            "metadata": cached.get("metadata", {}),
        }

    ds = openml.datasets.get_dataset(dataset_id)
    X, y, _, _ = ds.get_data(target=ds.default_target_attribute)
    df = X.copy()
    if y is not None and ds.default_target_attribute and ds.default_target_attribute not in df.columns:
        df[ds.default_target_attribute] = y

    df.to_parquet(data_path, index=False)

    payload = {
        "data_path": str(data_path),
        "default_target": ds.default_target_attribute,
        "metadata": {
            "source": "openml",
            "retrieved_at": datetime.now(UTC).isoformat(),
            "file_hash": None,
            "license": getattr(ds, "licence", None) or getattr(ds, "license", None),
            "source_id": f"openml:{dataset_id}",
            "name": ds.name,
        },
    }
    write_json(
        meta_path,
        {
            "default_target": payload["default_target"],
            "metadata": payload["metadata"],
        },
    )
    return payload
