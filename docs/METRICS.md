# TabAudit Metrics Specification

This document explains how TabAudit computes dataset metrics and scores.
It is intended as an implementation-level reference for users and contributors.

## 1. Output Structure
Each dataset report (`report.json`) contains:
- `basic_stats`: structure and missingness statistics
- `checks`: check outputs (cleanliness, schema, leakage, baseline model, RCPL, label quality)
- `scores`: sub-scores and final composite score
- `warnings` / `errors`: non-fatal and fatal signals

Primary implementation files:
- `tab_audit/checks/basic_stats.py`
- `tab_audit/checks/cleanliness.py`
- `tab_audit/checks/schema_checks.py`
- `tab_audit/checks/leakage.py`
- `tab_audit/checks/baseline_model.py`
- `tab_audit/checks/unsup_learnability.py`
- `tab_audit/checks/label_quality.py`
- `tab_audit/scoring/score.py`

## 2. Basic Stats (always computed)
Implemented in `compute_basic_stats(df)`.

### 2.1 Core size/type fields
- `n_rows`, `n_cols`: dataframe shape
- `type_summary`: per-column bucket counts among:
  - `numeric`
  - `categorical`
  - `bool`
  - `datetime`
  - `textish` (object dtype with average sample string length >= 30)

### 2.2 Missingness and duplicates
- `missing_fraction_overall = mean(df.isna())` over all cells
- `missing_top_columns`: top-10 columns by missing ratio
- `duplicate_rows_fraction = mean(df.duplicated())`

### 2.3 Constant/all-missing/id-like/high-cardinality
- `constant_columns_count`: columns with `nunique(dropna=False) <= 1`
- `all_missing_columns_count`: columns where all values are missing
- `unique_id_like_columns_count`: columns with
  - `nunique(dropna=True) / n_rows >= 0.95`
  - and `nunique(dropna=True) > 20`
- `high_cardinality_categorical_count`: columns in `{categorical,textish}` with
  - `nunique > min(200, max(20, 0.5*n_rows))`

### 2.4 Memory
- `memory_usage_bytes = df.memory_usage(deep=True).sum()`

## 3. Cleanliness Checks (always computed)
Implemented in `run_cleanliness_checks(df)`.

### 3.1 Invalid numeric count
For numeric columns:
- cast to numeric with `errors="coerce"`
- count `+/-inf` via `np.isinf(...)`
- aggregate into `invalid_numeric_count`

### 3.2 Whitespace-only strings
For non-numeric columns:
- cast non-null values to string
- count values matching regex `^\s*$`
- aggregate into `whitespace_only_count`

### 3.3 Mixed-type column heuristic
For non-numeric columns:
- `numeric_like = fraction of values parsable by pd.to_numeric(errors="coerce")`
- if `0.1 < numeric_like < 0.9`, column is flagged mixed-type
- output:
  - `mixed_type_columns_count`
  - `mixed_type_columns`

### 3.4 Outlier heuristic
For numeric columns:
- IQR-based outlier fraction with bounds `[Q1 - 3*IQR, Q3 + 3*IQR]`
- only evaluated when at least 20 numeric values exist
- columns with outlier fraction > 0.05 reported in `extreme_outlier_columns`

### 3.5 Datetime parse issues heuristic
For non-numeric columns:
- parse values with `pd.to_datetime(errors="coerce", utc=True)`
- compute `parse_fail_rate`
- if
  - `0.2 < parse_fail_rate < 0.95`, and
  - more than 30% values contain one of `- / :`
  then flag in `datetime_parse_issue_columns`

## 4. Schema/Constraint Checks
Implemented in `run_schema_checks(...)`.

Main validations:
- target column exists when specified
- no duplicate column names
- max columns threshold
- minimum rows warning threshold
- optional downsampling pathways for large inputs handled in CLI/model/profile flow

## 5. Leakage Heuristics (supervised only)
Implemented in `run_leakage_checks(df, target, task, random_seed)`.

If no valid supervised setup: returns warning and empty signals.

### 5.1 Identical-to-target check
For each non-target column:
- normalize both column and target as lowercased, stripped strings
- if equal element-wise -> `identical_to_target` (high severity)

### 5.2 Near-perfect correlation (regression)
For each non-target column:
- cast target and feature to numeric
- if sufficient non-null values (`>=20`)
- if `abs(corr(feature,target)) > 0.999` -> `near_perfect_corr` (high)

### 5.3 Class-separating constants (classification)
For each non-target column:
- group by target and compute `nunique` in each group
- if every class has <=1 unique value and overall column cardinality >= class count
  -> `class_constant_separator` (medium)

### 5.4 ID-leakage proxy model check
For up to 20 near-unique non-target columns (`unique_ratio > 0.95`):
- train/test split (70/30)
- classification:
  - 1-NN and depth-2 decision tree on encoded single-column feature
  - take best accuracy
- regression:
  - 1-NN and depth-2 decision tree
  - take best R2
- if best score > 0.95 -> `id_leakage_suspected` (high)

If any signal exists, a general warning is added.

## 6. Supervised Baseline Learnability
Implemented in `run_baseline_model(...)` + modeling backends in `tab_audit/modeling/`.

### 6.1 Availability gates
Baseline is unavailable if:
- no target / target not found
- rows < 25
- target has <=1 unique non-null value
- classification with insufficient smallest class size for CV folds

### 6.2 Task detection
`classification` when:
- target dtype object/category, or
- target unique count <= 20
Else `regression`.

### 6.3 CV metrics
Classification:
- primary metric: ROC-AUC (binary) or macro-F1 (multiclass)
- secondary: accuracy

Regression:
- primary metric: R2
- secondary: RMSE (`neg_root_mean_squared_error` converted to positive)

### 6.4 Stability and overfit proxies
- `primary_metric_cv_std = std(test_primary_across_folds)`
- `overfit_gap = mean(train_primary) - mean(test_primary)`

### 6.5 Baseline normalization to [0,1]
- classification: `baseline_score_norm = clamp(primary_metric_cv, 0, 1)`
- regression: `baseline_score_norm = clamp((primary_metric_cv + 1)/2, 0, 1)`

### 6.6 Device/backend fields
Each result records:
- `backend` (`sklearn`, `xgboost`, `lightgbm`, `catboost`)
- `device` (`cpu`/`cuda` resolved)
- `training_time_sec`

If preferred backend fails, model backend layer falls back to sklearn with warning.

## 7. Label-Free Learnability (RCPL)
Implemented in `run_rcpl_learnability(...)`.

Used when unsupervised learnability is enabled (typically when no supervised target).

### 7.1 Eligible RCPL target columns
A candidate column is skipped if:
- constant/single-valued
- ID-like (`unique_ratio >= rcpl_skip_id_threshold`)
- non-numeric with too many classes (`nunique > rcpl_max_cat_card`)

### 7.2 Proxy tasks
For each repeat:
- sample up to `rcpl_target_cols` eligible columns
- for each sampled target-column:
  - run supervised baseline predicting that column from all other columns
  - collect `baseline_score_norm`

### 7.3 RCPL aggregation
- each repeat score = mean valid per-target normalized scores
- `base_norm = mean(repeat_scores)`
- `stability_std = std(repeat_scores)`
- stability penalty: `min(0.3, stability_std)`
- final `rcpl_score_norm = clamp(base_norm * (1 - stability_penalty), 0, 1)`
- `rcpl_score = 100 * rcpl_score_norm`

If no valid proxy tasks succeed:
- RCPL unavailable with explanatory reason/warnings

## 8. Label Quality (classification + cleanlab)
Implemented in `run_label_quality(...)`.

Availability requires:
- supervised classification target
- cleanlab installed
- >=2 classes
- minimum class size >= 5

Method:
- preprocess via imputation + one-hot encoding
- LogisticRegression model
- stratified 5-fold out-of-fold predicted probabilities
- cleanlab `find_label_issues(labels, pred_probs)`

Outputs:
- `issues_count`
- `issue_fraction = issues_count / n_rows`
- optional `suspicious_row_indices` when enabled

## 9. Sub-score Formulas
Implemented in `tab_audit/scoring/score.py`.

All scores are in `[0,100]`.

### 9.1 Cleanliness score
Inputs:
- missing fraction
- duplicate fraction
- invalid numeric rate
- constant-column fraction
- mixed-type-column fraction
- all-missing-column fraction

Penalty:
- `0.35*missing`
- `0.20*duplicates`
- `0.15*invalids`
- `0.15*constant_frac`
- `0.10*mixed_frac`
- `0.05*all_missing_frac`

Then:
- `cleanliness_score = 100 * (1 - clamp01(penalty))`

### 9.2 Structure score
Compute:
- `diversity = (#non-empty type buckets)/5`

Penalties:
- feature-count penalty:
  - +0.2 if `n_cols < 3`
  - +min(0.4, (n_cols - 500)/1000) if `n_cols > 500`
- high-cardinality penalty: `min(0.3, high_card_count/n_cols)`
- id-like penalty: `min(0.3, id_like_count/n_cols)`

Then:
- `structure_score = 100 * (0.7*diversity + 0.3*(1 - clamp01(sum_penalties)))`

### 9.3 Learnability score
If supervised baseline available:
- `norm = baseline_score_norm`
- `penalty = min(0.7, 0.4*cv_std + 0.3*overfit_gap + 0.3*imbalance_penalty)`
- `imbalance_penalty = max(0, 0.3 - class_imbalance)` when classification
- `learnability_score = 100 * clamp01(norm) * (1 - penalty)`

Else if RCPL available:
- `learnability_score = clamp(rcpl_score, 0, 100)`

Else:
- fallback `learnability_score = 50` with warning

### 9.4 Label quality score
If label quality unavailable:
- fallback `label_quality_score = 50` with warnings

Else:
- `label_quality_score = 100 * (1 - clamp01(issue_fraction))`

## 10. Composite Quality Score
Final score uses configured weights:

`quality_score =`
- `w_cleanliness * cleanliness_score`
- `+ w_structure * structure_score`
- `+ w_learnability * learnability_score`
- `+ w_label_quality * label_quality_score`

Default weights (must sum to 1.0):
- cleanliness: 0.35
- structure: 0.25
- learnability: 0.25
- label_quality: 0.15

## 11. Fallbacks and Warnings
Key fallback behavior:
- no target -> supervised baseline unavailable
- no target + RCPL enabled -> RCPL learns from proxy targets
- no target + RCPL unavailable/disabled -> learnability fallback 50
- label-quality unavailable (e.g., cleanlab missing) -> label-quality fallback 50

Warnings are intentionally emitted to explain reduced confidence when fallbacks occur.

## 12. Notes for Interpretation
- Scores are heuristic quality signals, not guarantees of model performance.
- Leakage checks are practical detectors, not formal proof of leakage.
- RCPL is a label-free proxy of predictability/structure, not task-specific utility.
- Compare scores across datasets with similar schema/task assumptions for best use.
