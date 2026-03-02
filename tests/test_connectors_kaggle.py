from __future__ import annotations

import sys
import types

import pytest

from tab_audit.connectors.kaggle_connector import fetch_kaggle_dataset


def _install_fake_kaggle(monkeypatch, api_cls):  # noqa: ANN001
    mod_root = types.ModuleType("kaggle")
    mod_api = types.ModuleType("kaggle.api")
    mod_ext = types.ModuleType("kaggle.api.kaggle_api_extended")
    mod_ext.KaggleApi = api_cls

    monkeypatch.setitem(sys.modules, "kaggle", mod_root)
    monkeypatch.setitem(sys.modules, "kaggle.api", mod_api)
    monkeypatch.setitem(sys.modules, "kaggle.api.kaggle_api_extended", mod_ext)


def test_fetch_kaggle_dataset_download_success(monkeypatch, tmp_path):
    class FakeKaggleApi:
        def authenticate(self) -> None:
            return None

        def dataset_download_files(self, slug: str, path: str, unzip: bool, quiet: bool) -> None:
            assert slug == "owner/ds"
            assert unzip is True
            assert quiet is True
            (tmp_path / "kaggle" / "owner__ds").mkdir(parents=True, exist_ok=True)
            (tmp_path / "kaggle" / "owner__ds" / "train.csv").write_text("a,b\n1,2\n")

    _install_fake_kaggle(monkeypatch, FakeKaggleApi)

    out = fetch_kaggle_dataset(slug="owner/ds", file="train.csv", cache_dir=str(tmp_path))
    assert out["metadata"]["source"] == "kaggle"
    assert out["data_path"].endswith("train.csv")


def test_fetch_kaggle_dataset_download_failure(monkeypatch, tmp_path):
    class FailingKaggleApi:
        def authenticate(self) -> None:
            raise RuntimeError("auth failed")

        def dataset_download_files(self, slug: str, path: str, unzip: bool, quiet: bool) -> None:  # noqa: ARG002
            return None

    _install_fake_kaggle(monkeypatch, FailingKaggleApi)

    with pytest.raises(RuntimeError, match="Kaggle API authentication/download failed"):
        fetch_kaggle_dataset(slug="owner/ds", file="train.csv", cache_dir=str(tmp_path))
