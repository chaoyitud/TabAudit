from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

from tab_audit.io.cache import ensure_dir
from tab_audit.utils.hashing import hash_file


def fetch_kaggle_dataset(slug: str, file: str, cache_dir: str, **_: object) -> dict:
    # Force Kaggle API config directory into project-writable space.
    kaggle_cfg_dir = Path(cache_dir) / "kaggle_config"
    kaggle_cfg_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("KAGGLE_CONFIG_DIR", str(kaggle_cfg_dir))

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise RuntimeError("kaggle extra not installed. Use pip install -e '.[kaggle]'") from exc
    except BaseException as exc:
        raise RuntimeError(
            "Kaggle API initialization failed. "
            f"Set credentials in {kaggle_cfg_dir}/kaggle.json or configure KAGGLE_CONFIG_DIR."
        ) from exc

    dataset_dir = ensure_dir(Path(cache_dir) / "kaggle" / slug.replace("/", "__"))
    file_path = dataset_dir / file
    if file_path.exists():
        return {
            "data_path": str(file_path),
            "default_target": None,
            "metadata": {
                "source": "kaggle",
                "retrieved_at": datetime.now(UTC).isoformat(),
                "file_hash": hash_file(file_path),
                "license": None,
                "source_id": f"kaggle:{slug}/{file}",
                "name": slug,
            },
        }

    try:
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(slug, path=str(dataset_dir), unzip=True, quiet=True)
    except BaseException as exc:
        raise RuntimeError(
            "Kaggle API authentication/download failed. "
            f"Set credentials in {kaggle_cfg_dir}/kaggle.json or configure KAGGLE_CONFIG_DIR."
        ) from exc

    if not file_path.exists():
        raise FileNotFoundError(f"File {file} not found in downloaded Kaggle dataset {slug}")

    return {
        "data_path": str(file_path),
        "default_target": None,
        "metadata": {
            "source": "kaggle",
            "retrieved_at": datetime.now(UTC).isoformat(),
            "file_hash": hash_file(file_path),
            "license": None,
            "source_id": f"kaggle:{slug}/{file}",
            "name": slug,
        },
    }
