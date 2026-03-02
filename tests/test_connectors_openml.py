from __future__ import annotations

import sys
import types

import pandas as pd

from tab_audit.connectors.openml_connector import fetch_openml_dataset


def test_fetch_openml_dataset_and_cache(monkeypatch, tmp_path):
    calls: dict[str, object] = {}

    class FakeConfig:
        @staticmethod
        def set_root_cache_directory(path: str) -> None:
            calls["cache_root"] = path

    class FakeDataset:
        default_target_attribute = "target"
        name = "fake_ds"
        license = "MIT"

        @staticmethod
        def get_data(target: str):  # noqa: ANN001
            assert target == "target"
            x = pd.DataFrame({"f1": [1, 2, 3]})
            y = pd.Series([0, 1, 0], name="target")
            return x, y, None, None

    class FakeDatasets:
        @staticmethod
        def get_dataset(dataset_id: int) -> FakeDataset:
            calls["dataset_id"] = dataset_id
            return FakeDataset()

    fake_openml = types.SimpleNamespace(config=FakeConfig(), datasets=FakeDatasets())
    monkeypatch.setitem(sys.modules, "openml", fake_openml)

    out = fetch_openml_dataset(dataset_id=61, cache_dir=str(tmp_path))
    assert out["default_target"] == "target"
    assert out["metadata"]["source"] == "openml"
    assert out["metadata"]["name"] == "fake_ds"
    assert "openml_client" in str(calls["cache_root"])

    # Cached call should not invoke OpenML dataset fetch again.
    class FailDatasets:
        @staticmethod
        def get_dataset(dataset_id: int):  # noqa: ANN001
            raise AssertionError("should not fetch from OpenML when cache exists")

    fake_openml_cached = types.SimpleNamespace(config=FakeConfig(), datasets=FailDatasets())
    monkeypatch.setitem(sys.modules, "openml", fake_openml_cached)

    out2 = fetch_openml_dataset(dataset_id=61, cache_dir=str(tmp_path))
    assert out2["default_target"] == "target"
    assert out2["data_path"].endswith("dataset.parquet")
