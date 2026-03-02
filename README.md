<p align="center">
  <img src="assets/icon.png" alt="TabAudit icon" width="160" />
</p>

<h1 align="center">TabAudit</h1>

<p align="center">
  <strong>Audit tabular datasets before you waste training time.</strong><br/>
  OpenML + Kaggle + local CSVs, unified quality scoring, and batch leaderboards.
</p>

<p align="center">
  <a href="https://github.com/chaoyitud/TabAudit/actions/workflows/ci.yml">
    <img src="https://github.com/chaoyitud/TabAudit/actions/workflows/ci.yml/badge.svg" alt="CI" />
  </a>
  <a href="https://github.com/chaoyitud/TabAudit/actions/workflows/cd.yml">
    <img src="https://github.com/chaoyitud/TabAudit/actions/workflows/cd.yml/badge.svg" alt="CD" />
  </a>
</p>

<p align="center">
  <a href="#quickstart-uv-recommended">Quickstart</a> •
  <a href="#what-you-get">Artifacts</a> •
  <a href="#scoring-model">Scoring</a> •
  <a href="#multi-dataset-batch">Batch</a>
</p>

TabAudit is a production-oriented, config-driven **tabular dataset quality auditor** for ML teams.
It helps you quickly decide which datasets are worth modeling by generating:
- standardized quality checks,
- composite and sub-scores,
- per-dataset reports,
- and a global leaderboard.

## Why TabAudit
- **Saves modeling time**: rank candidate datasets before training expensive models.
- **Works in real-world mess**: mixed encodings, bad CSV lines, missing targets, large files.
- **Supervised + label-free learnability**: fallback RCPL score when no target exists.
- **Built for batch operations**: OpenML lists, Kaggle lists, directories, mixed YAML specs.
- **Reproducible by default**: deterministic sampling, caching, fixed seeds, machine-readable outputs.

## Key Features
- Single dataset evaluation (`local`, `openml`, `kaggle`)
- Multi-dataset batch execution:
  - OpenML ID list
  - Kaggle `slug:file` list
  - Local directory glob (`*.csv` etc.)
  - Mixed YAML descriptors (`openml`, `kaggle`, `local`, `local_dir`)
- Robust CSV ingestion:
  - encoding fallback (`utf-8`, `latin-1`)
  - separator sniffing (comma/tab/semicolon)
  - bad-line tolerant parsing
- Baseline learnability (CPU + optional GPU backends)
- Label-free learnability via RCPL (Random Column Prediction Learnability)
- Composite quality score + detailed warnings/errors
- Global leaderboard (`leaderboard.csv`, `leaderboard.html`, `summary.parquet`)
- Rich CLI progress indicators with elapsed/remaining time

## Quickstart (`uv` recommended)

### 1) Create and activate environment
```bash
uv venv
source .venv/bin/activate
```

### 2) Install
Base:
```bash
uv pip install -e .
```

Common extras:
```bash
uv pip install -e ".[openml,kaggle,profiling,ml,test,lint]"
```

Optional GPU extras:
```bash
uv pip install -e ".[gpu]"
# or pick one backend:
uv pip install -e ".[gpu-xgboost]"
uv pip install -e ".[gpu-lightgbm]"
uv pip install -e ".[gpu-catboost]"
```

### 3) Try a local file
```bash
tabaudit evaluate local --path data/my.csv --target target_col --out reports/
```

## CLI Examples

### Single dataset
```bash
# local supervised
tabaudit evaluate local --path data/my.csv --target y --out reports/

# local label-free learnability
tabaudit evaluate local --path data/my.csv --target none --unsup-learnability on --out reports/

# OpenML
tabaudit evaluate openml --dataset-id 61 --target auto --out reports/

# Kaggle
tabaudit evaluate kaggle --dataset zynicide/wine-reviews --file winemag-data-130k-v2.csv --target auto --out reports/
```

### Multi-dataset batch

OpenML list (repeat `--dataset-ids`):
```bash
tabaudit batch openml \
  --dataset-ids 61 \
  --dataset-ids 15 \
  --dataset-ids 37 \
  --target auto \
  --out reports/
```

OpenML from file (`.txt`, `.json`, `.yaml`):
```bash
tabaudit batch openml --dataset-ids-file configs/openml_ids.txt --target auto --out reports/
```
Supported file shapes:
- `.txt`: `61 15 37` or one id per line
- `.json`: `[61, 15, 37]` or `{\"dataset_ids\": [61, 15, 37]}`
- `.yaml`: `dataset_ids: [61, 15, 37]`

Kaggle list (`slug:file`, repeat `--datasets`):
```bash
tabaudit batch kaggle \
  --datasets "zynicide/wine-reviews:winemag-data-130k-v2.csv" \
  --datasets "owner/other-dataset:data.csv" \
  --target auto \
  --out reports/
```

Kaggle from file (`.txt`, `.json`, `.yaml`):
```bash
tabaudit batch kaggle --datasets-file configs/kaggle_datasets.yaml --target auto --out reports/
```

Local directory:
```bash
tabaudit batch dir --path data_dir/ --pattern "*.csv" --target none --out reports/
```

Mixed sources in one run (single leaderboard):
```bash
tabaudit batch mixed \
  --openml-dataset-ids 61 \
  --kaggle-datasets "zynicide/wine-reviews:winemag-data-130k-v2.csv" \
  --local-files data/train.csv \
  --local-dirs data/benchmark_csvs \
  --pattern "*.csv" \
  --target auto \
  --out reports/
```

Mixed sources from files:
```bash
tabaudit batch mixed \
  --openml-dataset-ids-file configs/openml_ids.txt \
  --kaggle-datasets-file configs/kaggle_datasets.yaml \
  --local-files-file configs/local_files.txt \
  --local-dirs-file configs/local_dirs.yaml \
  --out reports/
```

YAML:
```bash
tabaudit batch --config configs/example_multi.yaml --out reports/
# or
tabaudit batch yaml --config configs/example_multi.yaml --out reports/
```

Rebuild leaderboard from existing reports:
```bash
tabaudit leaderboard --reports-dir reports/
```

## What You Get
For each dataset:
- `reports/<dataset_slug>/report.json`
- `reports/<dataset_slug>/report.html`
- `reports/<dataset_slug>/processed_sample.parquet`
- optional `reports/<dataset_slug>/profiling.html`

Global artifacts:
- `reports/summary.csv`
- `reports/leaderboard.csv`
- `reports/leaderboard.html`
- `reports/summary.parquet` (best-effort)

## Scoring Model
All sub-scores are in `[0,100]`:
- `cleanliness_score`
- `structure_score`
- `learnability_score`
- `label_quality_score`

Composite score:
```text
quality_score = w_cleanliness*cleanliness_score
              + w_structure*structure_score
              + w_learnability*learnability_score
              + w_label_quality*label_quality_score
```

Default weights:
- cleanliness: `0.35`
- structure: `0.25`
- learnability: `0.25`
- label quality: `0.15`

## Learnability Modes

### Supervised learnability
If target exists, TabAudit trains a fast baseline pipeline (CV-based) and reports performance, stability, backend, device, and training time.

### Label-free learnability (RCPL)
If target is missing (or explicitly `none`), TabAudit can run RCPL:
1. sample target columns,
2. predict each target from remaining columns,
3. aggregate normalized performance into an unsupervised learnability score.

Useful flags:
- `--unsup-learnability on|off`
- `--rcpl-target-cols 10`
- `--rcpl-repeats 3`
- `--rcpl-max-rows 200000`
- `--rcpl-cv-folds 5`
- `--rcpl-seed 42`
- `--rcpl-max-cat-card 200`
- `--rcpl-skip-id-threshold 0.98`

## Device and Backend Selection
Use `--device auto|cpu|cuda` (default `auto`).

Behavior:
- `auto`: uses GPU backend if CUDA + backend library are available.
- `cuda`: tries GPU first, then falls back to sklearn CPU with warning.
- `cpu`: CPU only.

Default backend preference:
```yaml
["xgboost", "lightgbm", "catboost", "sklearn"]
```

## Leaderboard Semantics
Leaderboard includes both successful and failed datasets.

Columns include:
- dataset metadata (`dataset_name`, `source`, `target`)
- shape (`n_rows`, `n_cols`)
- overall + sub-scores
- warnings count
- baseline metrics (when available)
- `device`, `backend`, `runtime_sec`
- `status` (`OK`/`FAILED`) and error snippet

Sorting:
1. `quality_score` descending
2. `learnability_score` descending (tie-break)

## Config Example
See: [`configs/example_multi.yaml`](configs/example_multi.yaml)

## Testing
```bash
pytest -q
```

Coverage:
```bash
pytest -q --cov=tab_audit --cov-report=term-missing --cov-report=xml
```

Lint:
```bash
ruff check tab_audit tests
pylint tab_audit --disable=C0114,C0115,C0116,R0903,R0913,R0914,W1203 --fail-under=8.0
```

## CI/CD
- `CI` workflow (`.github/workflows/ci.yml`)
  - Runs on pull requests and pushes to `main`
  - Linting: `ruff` (required), `pylint` (informational)
  - Tests on Python `3.11` and `3.12`
  - Coverage generated and uploaded as artifacts
- `CD` workflow (`.github/workflows/cd.yml`)
  - Runs on pushes to `main` and manual dispatch
  - Builds wheel/sdist
  - Validates package metadata via `twine check`
  - Uploads `dist/*` as build artifacts

## Practical Assumptions
- OpenML `--target auto` uses OpenML default target attribute when present.
- Kaggle/local often lack explicit targets; RCPL is designed for this case.
- Leakage detection is heuristic and practical, not formal proof.
- Batch mode continues after per-dataset failures and records them in leaderboard outputs.

## Contributing
PRs are welcome. High-impact areas:
- stronger leakage heuristics,
- additional dataset connectors,
- improved HTML report UX,
- calibration of score mappings on large benchmark suites.
