from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


class ScoringWeights(BaseModel):
    cleanliness: float = 0.35
    structure: float = 0.25
    learnability: float = 0.25
    label_quality: float = 0.15

    @model_validator(mode="after")
    def validate_sum(self) -> ScoringWeights:
        total = self.cleanliness + self.structure + self.learnability + self.label_quality
        if abs(total - 1.0) > 1e-8:
            raise ValueError(f"scoring weights must sum to 1.0, got {total}")
        return self


class GlobalConfig(BaseModel):
    max_rows_profile: int = 200_000
    max_rows_model: int = 200_000
    max_columns: int = 500
    min_rows_warn: int = 1000
    random_seed: int = 42
    cache_dir: str = ".cache/tab_audit"
    reports_dir: str = "reports"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    model_backend_preference: list[Literal["xgboost", "lightgbm", "catboost", "sklearn"]] = Field(
        default_factory=lambda: ["xgboost", "lightgbm", "catboost", "sklearn"]
    )
    batch_concurrency: int = 1
    unsup_learnability: Literal["auto", "on", "off"] = "auto"
    rcpl_target_cols: int = 10
    rcpl_repeats: int = 3
    rcpl_max_rows: int = 200_000
    # RCPL is run many times (repeats * target_cols); 3-fold CV is a good speed/variance tradeoff.
    rcpl_cv_folds: int = 3
    rcpl_seed: int = 42
    rcpl_max_cat_card: int = 200
    rcpl_skip_id_threshold: float = 0.98
    scoring_weights: ScoringWeights = Field(default_factory=ScoringWeights)


class DatasetSpec(BaseModel):
    name: str
    source: Literal["openml", "kaggle", "local", "local_dir"]
    dataset_id: int | None = None
    slug: str | None = None
    file: str | None = None
    path: str | None = None
    pattern: str | None = None
    target: str | None = "auto"

    @model_validator(mode="after")
    def source_requirements(self) -> DatasetSpec:
        if self.source == "openml" and self.dataset_id is None:
            raise ValueError("openml dataset requires dataset_id")
        if self.source == "kaggle" and (self.slug is None or self.file is None):
            raise ValueError("kaggle dataset requires slug and file")
        if self.source == "local" and self.path is None:
            raise ValueError("local dataset requires path")
        if self.source == "local_dir" and self.path is None:
            raise ValueError("local_dir dataset requires path")
        return self

    @field_validator("target")
    @classmethod
    def normalize_target(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if isinstance(v, str) and v.strip().lower() in {"", "none", "null"}:
            return None
        return v


class BatchConfig(BaseModel):
    datasets: list[DatasetSpec]
    global_: GlobalConfig = Field(default_factory=GlobalConfig, alias="global")

    model_config = {"populate_by_name": True}


def load_config(path: str | Path) -> BatchConfig:
    p = Path(path)
    raw = yaml.safe_load(p.read_text())
    try:
        return BatchConfig.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"Invalid config {p}: {exc}") from exc
