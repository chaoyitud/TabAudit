from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from tab_audit.checks.baseline_model import run_baseline_model
from tab_audit.checks.basic_stats import compute_basic_stats
from tab_audit.checks.cleanliness import run_cleanliness_checks
from tab_audit.checks.label_quality import run_label_quality
from tab_audit.checks.leakage import run_leakage_checks
from tab_audit.checks.schema_checks import run_schema_checks
from tab_audit.checks.unsup_learnability import run_rcpl_learnability
from tab_audit.config import BatchConfig, DatasetSpec, GlobalConfig, load_config
from tab_audit.connectors.local_connector import list_local_datasets
from tab_audit.io.cache import ensure_dir
from tab_audit.io.dataset_inputs import (
    parse_kaggle_dataset_specs,
    parse_local_dir_paths,
    parse_local_file_paths,
    parse_openml_dataset_ids,
)
from tab_audit.io.read_tabular import read_tabular
from tab_audit.leaderboard.build import build_leaderboard
from tab_audit.registry import CONNECTOR_REGISTRY
from tab_audit.reporting.report_html import write_dataset_report_html
from tab_audit.reporting.report_json import write_dataset_report_json, write_summary_csv
from tab_audit.scoring.score import compute_scores
from tab_audit.utils.hashing import stable_slug
from tab_audit.utils.logging import configure_logging
from tab_audit.utils.sampling import maybe_sample_rows

app = typer.Typer(help="Tabular dataset quality auditor")
evaluate_app = typer.Typer(help="Evaluate one dataset")
batch_app = typer.Typer(help="Batch evaluation")
app.add_typer(evaluate_app, name="evaluate")
app.add_typer(batch_app, name="batch")

console = Console()
logger = logging.getLogger(__name__)


def _normalize_target_option(target: str | None) -> str | None:
    if target is None:
        return None
    if target.lower() in {"none", "null"}:
        return None
    return target


def _maybe_generate_profile(df: pd.DataFrame, out_path: Path) -> str | None:
    try:
        from ydata_profiling import ProfileReport
    except ImportError:
        return None
    profile = ProfileReport(df, minimal=True, title="Tab Audit Profiling")
    profile.to_file(out_path)
    return str(out_path)


def _resolve_target(spec_target: str | None, cli_target: str | None, connector_default: str | None) -> str | None:
    target = cli_target if cli_target is not None else spec_target
    target = _normalize_target_option(target)
    if target is None:
        return None
    if isinstance(target, str) and target.lower() == "auto":
        return connector_default
    return target


def _summary_row_from_report(report: dict[str, Any], source: str) -> dict[str, Any]:
    basic = report.get("basic_stats", {})
    scores = report.get("scores", {})
    baseline = report.get("checks", {}).get("baseline_model", {})
    return {
        "dataset_slug": report.get("dataset_slug"),
        "dataset_name": report.get("dataset_name"),
        "source": source,
        "target": report.get("target"),
        "status": report.get("status", "OK"),
        "n_rows": basic.get("n_rows"),
        "n_cols": basic.get("n_cols"),
        "quality_score": scores.get("quality_score"),
        "cleanliness_score": scores.get("cleanliness_score"),
        "structure_score": scores.get("structure_score"),
        "learnability_score": scores.get("learnability_score"),
        "label_quality_score": scores.get("label_quality_score"),
        "warnings_count": len(report.get("warnings", [])),
        "errors_count": len(report.get("errors", [])),
        "backend": baseline.get("backend"),
        "device": baseline.get("device"),
        "runtime_sec": baseline.get("training_time_sec"),
        "error_message": report.get("error_message"),
    }


def _write_failure_report(spec: DatasetSpec, out_dir: str, exc: Exception) -> dict[str, Any]:
    dataset_slug = stable_slug(spec.name)
    report = {
        "dataset_name": spec.name,
        "dataset_slug": dataset_slug,
        "target": _normalize_target_option(spec.target),
        "metadata": {"source": spec.source},
        "basic_stats": {"n_rows": None, "n_cols": None},
        "checks": {"baseline_model": {"backend": None, "device": None, "training_time_sec": None}},
        "scores": {
            "quality_score": None,
            "cleanliness_score": None,
            "structure_score": None,
            "learnability_score": None,
            "label_quality_score": None,
        },
        "warnings": [],
        "errors": [str(exc)],
        "status": "FAILED",
        "error_message": f"{exc.__class__.__name__}: {exc}",
        "artifacts": {},
    }
    write_dataset_report_json(out_dir, dataset_slug, report)
    write_dataset_report_html(out_dir, dataset_slug, report)
    return _summary_row_from_report(report, spec.source)


def _evaluate_one(
    spec: DatasetSpec,
    global_cfg: GlobalConfig,
    out_dir: str,
    target_override: str | None = None,
    save_row_ids: bool = False,
    enable_profiling: bool = False,
    sep: str | None = None,
    unsup_learnability: str | None = None,
    rcpl_target_cols: int | None = None,
    rcpl_repeats: int | None = None,
    rcpl_max_rows: int | None = None,
    rcpl_cv_folds: int | None = None,
    rcpl_seed: int | None = None,
    rcpl_max_cat_card: int | None = None,
    rcpl_skip_id_threshold: float | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if spec.source == "local_dir":
        raise ValueError("local_dir cannot be evaluated as a single dataset; expand it in batch mode")

    warnings: list[str] = []
    errors: list[str] = []

    connector = CONNECTOR_REGISTRY[spec.source]
    conn_kwargs: dict[str, Any] = {}
    if spec.source == "local":
        conn_kwargs = {"path": spec.path}
    elif spec.source == "openml":
        conn_kwargs = {"dataset_id": spec.dataset_id}
    elif spec.source == "kaggle":
        conn_kwargs = {"slug": spec.slug, "file": spec.file}

    if progress_callback:
        progress_callback("Fetching dataset")
    retrieved = connector(cache_dir=global_cfg.cache_dir, **conn_kwargs)
    data_path = retrieved["data_path"]

    effective_sep = sep if spec.source == "local" else None
    if progress_callback:
        progress_callback("Reading tabular file")
    df = read_tabular(data_path, sep=effective_sep)

    target = _resolve_target(spec.target, target_override, retrieved.get("default_target"))
    if target is None:
        warnings.append("no target resolved; using unsupervised-only scoring for learnability/label quality")

    if progress_callback:
        progress_callback("Running schema checks")
    schema = run_schema_checks(
        df,
        target=target,
        max_columns=global_cfg.max_columns,
        min_rows_warn=global_cfg.min_rows_warn,
    )
    warnings.extend(schema["warnings"])
    errors.extend(schema["errors"])

    if progress_callback:
        progress_callback("Computing stats and cleanliness")
    basic = compute_basic_stats(df)
    clean = run_cleanliness_checks(df)

    supervised_available = bool(target and target in df.columns)
    unsup_mode = unsup_learnability or global_cfg.unsup_learnability
    use_unsup = (unsup_mode == "on") or (unsup_mode == "auto" and not supervised_available)

    unsup = {
        "available": False,
        "reason": "unsup_learnability_disabled",
        "rcpl_score": None,
        "rcpl_score_norm": None,
        "warnings": [],
    }

    if progress_callback:
        progress_callback("Running learnability and quality checks")
    if supervised_available:
        baseline = run_baseline_model(
            df,
            target=target,
            max_rows_model=global_cfg.max_rows_model,
            random_seed=global_cfg.random_seed,
            device=global_cfg.device,
            model_backend_preference=global_cfg.model_backend_preference,
        )
        task = baseline.get("task") if baseline.get("available") else None
        warnings.extend(baseline.get("warnings", []))
        leakage = run_leakage_checks(df, target, task, global_cfg.random_seed)
        warnings.extend(leakage.get("warnings", []))
        label_q = run_label_quality(df, target, task, global_cfg.random_seed, save_row_ids=save_row_ids)
    else:
        baseline = {"available": False, "reason": "no_target", "backend": "none", "device": global_cfg.device}
        leakage = {"warnings": ["leakage checks skipped (no target)"], "signals": []}
        label_q = {
            "available": False,
            "issue_fraction": None,
            "issues_count": None,
            "warnings": ["label quality skipped (no target)"],
        }

    if use_unsup:
        unsup = run_rcpl_learnability(
            df,
            rcpl_target_cols=rcpl_target_cols if rcpl_target_cols is not None else global_cfg.rcpl_target_cols,
            rcpl_repeats=rcpl_repeats if rcpl_repeats is not None else global_cfg.rcpl_repeats,
            rcpl_max_rows=rcpl_max_rows if rcpl_max_rows is not None else global_cfg.rcpl_max_rows,
            rcpl_cv_folds=rcpl_cv_folds if rcpl_cv_folds is not None else global_cfg.rcpl_cv_folds,
            rcpl_seed=rcpl_seed if rcpl_seed is not None else global_cfg.rcpl_seed,
            rcpl_max_cat_card=rcpl_max_cat_card if rcpl_max_cat_card is not None else global_cfg.rcpl_max_cat_card,
            rcpl_skip_id_threshold=(
                rcpl_skip_id_threshold if rcpl_skip_id_threshold is not None else global_cfg.rcpl_skip_id_threshold
            ),
            device=global_cfg.device,
            model_backend_preference=global_cfg.model_backend_preference,
        )
        warnings.extend(unsup.get("warnings", []))

    weights = {
        "cleanliness": global_cfg.scoring_weights.cleanliness,
        "structure": global_cfg.scoring_weights.structure,
        "learnability": global_cfg.scoring_weights.learnability,
        "label_quality": global_cfg.scoring_weights.label_quality,
    }
    if progress_callback:
        progress_callback("Computing composite score")
    scores = compute_scores(basic, clean, baseline, unsup, label_q, weights)
    warnings.extend(scores.pop("warnings", []))

    dataset_slug = stable_slug(spec.name)
    ds_out_dir = ensure_dir(Path(out_dir) / dataset_slug)

    sampled_profile_df = maybe_sample_rows(df, global_cfg.max_rows_profile, global_cfg.random_seed)
    sampled_profile_df.to_parquet(ds_out_dir / "processed_sample.parquet", index=False)

    profiling_path = None
    if enable_profiling:
        profiling_path = _maybe_generate_profile(sampled_profile_df, ds_out_dir / "profiling.html")
        if profiling_path is None:
            warnings.append("ydata-profiling not installed; profiling skipped")

    if progress_callback:
        progress_callback("Writing reports")
    report = {
        "dataset_name": spec.name,
        "dataset_slug": dataset_slug,
        "target": target,
        "metadata": retrieved.get("metadata", {}),
        "basic_stats": basic,
        "checks": {
            "cleanliness": clean,
            "schema": schema,
            "leakage": leakage,
            "baseline_model": baseline,
            "unsupervised_learnability": unsup,
            "label_quality": label_q,
        },
        "scores": scores,
        "warnings": warnings,
        "errors": errors,
        "status": "OK" if not errors else "FAILED",
        "error_message": None,
        "artifacts": {
            "processed_sample_parquet": str(ds_out_dir / "processed_sample.parquet"),
            "profiling_html": profiling_path,
        },
    }

    write_dataset_report_json(out_dir, dataset_slug, report)
    write_dataset_report_html(out_dir, dataset_slug, report)
    if progress_callback:
        progress_callback("Done")
    return report, _summary_row_from_report(report, spec.source)


def _evaluate_with_progress(
    spec: DatasetSpec,
    cfg: GlobalConfig,
    out: str,
    target_override: str | None = None,
    save_row_ids: bool = False,
    enable_profiling: bool = False,
    sep: str | None = None,
    unsup_learnability: str | None = None,
    rcpl_target_cols: int | None = None,
    rcpl_repeats: int | None = None,
    rcpl_max_rows: int | None = None,
    rcpl_cv_folds: int | None = None,
    rcpl_seed: int | None = None,
    rcpl_max_cat_card: int | None = None,
    rcpl_skip_id_threshold: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"[cyan]Evaluating {spec.name}", total=8)

        def _step(description: str) -> None:
            progress.update(task, description=f"[cyan]{description}", advance=1)

        return _evaluate_one(
            spec,
            cfg,
            out,
            target_override=target_override,
            save_row_ids=save_row_ids,
            enable_profiling=enable_profiling,
            sep=sep,
            unsup_learnability=unsup_learnability,
            rcpl_target_cols=rcpl_target_cols,
            rcpl_repeats=rcpl_repeats,
            rcpl_max_rows=rcpl_max_rows,
            rcpl_cv_folds=rcpl_cv_folds,
            rcpl_seed=rcpl_seed,
            rcpl_max_cat_card=rcpl_max_cat_card,
            rcpl_skip_id_threshold=rcpl_skip_id_threshold,
            progress_callback=_step,
        )


def _print_result(report: dict[str, Any]) -> None:
    t = Table(title=f"Dataset audit: {report['dataset_slug']}")
    t.add_column("Metric")
    t.add_column("Value")
    t.add_row("Status", str(report.get("status", "OK")))
    t.add_row("Quality", str(report.get("scores", {}).get("quality_score")))
    t.add_row("Rows", str(report.get("basic_stats", {}).get("n_rows")))
    t.add_row("Cols", str(report.get("basic_stats", {}).get("n_cols")))
    t.add_row("Warnings", str(len(report.get("warnings", []))))
    t.add_row("Errors", str(len(report.get("errors", []))))
    console.print(t)


def _expand_yaml_specs(cfg: BatchConfig) -> list[DatasetSpec]:
    expanded: list[DatasetSpec] = []
    for spec in cfg.datasets:
        if spec.source != "local_dir":
            expanded.append(spec)
            continue
        for entry in list_local_datasets(spec.path or "", spec.pattern or "*.csv"):
            expanded.append(
                DatasetSpec(
                    name=entry["name"],
                    source="local",
                    path=entry["path"],
                    target=spec.target,
                )
            )
    return expanded


def _run_batch(
    specs: list[DatasetSpec],
    cfg: GlobalConfig,
    out: str,
    save_row_ids: bool,
    enable_profiling: bool,
    sep: str | None = None,
    unsup_learnability: str | None = None,
    rcpl_target_cols: int | None = None,
    rcpl_repeats: int | None = None,
    rcpl_max_rows: int | None = None,
    rcpl_cv_folds: int | None = None,
    rcpl_seed: int | None = None,
    rcpl_max_cat_card: int | None = None,
    rcpl_skip_id_threshold: float | None = None,
) -> None:
    ensure_dir(out)
    summary_rows: list[dict[str, Any]] = []

    def worker(spec: DatasetSpec) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            return _evaluate_one(
                spec,
                cfg,
                out,
                save_row_ids=save_row_ids,
                enable_profiling=enable_profiling,
                sep=sep,
                unsup_learnability=unsup_learnability,
                rcpl_target_cols=rcpl_target_cols,
                rcpl_repeats=rcpl_repeats,
                rcpl_max_rows=rcpl_max_rows,
                rcpl_cv_folds=rcpl_cv_folds,
                rcpl_seed=rcpl_seed,
                rcpl_max_cat_card=rcpl_max_cat_card,
                rcpl_skip_id_threshold=rcpl_skip_id_threshold,
            )
        except Exception as exc:
            logger.exception("Dataset %s failed: %s", spec.name, exc)
            row = _write_failure_report(spec, out, exc)
            report = {
                "dataset_slug": row["dataset_slug"],
                "scores": {"quality_score": None},
                "basic_stats": {"n_rows": None, "n_cols": None},
                "warnings": [],
                "errors": [row.get("error_message")],
                "status": "FAILED",
            }
            return report, row

    max_workers = max(1, int(cfg.batch_concurrency))
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Starting batch run", total=max(1, len(specs)))
        if max_workers == 1:
            for spec in specs:
                progress.update(task, description=f"[cyan]Evaluating {spec.name}")
                report, row = worker(spec)
                summary_rows.append(row)
                _print_result(report)
                progress.advance(task)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(worker, spec): spec for spec in specs}
                for future in as_completed(futures):
                    spec = futures[future]
                    report, row = future.result()
                    summary_rows.append(row)
                    _print_result(report)
                    progress.update(task, description=f"[cyan]Completed {spec.name}")
                    progress.advance(task)

    write_summary_csv(out, summary_rows)
    leaderboard = build_leaderboard(out)
    console.print(f"Batch complete. datasets={len(specs)} leaderboard_rows={len(leaderboard)} out={out}")


@evaluate_app.command("local")
def evaluate_local(
    path: str = typer.Option(..., help="Path to local file"),
    target: str | None = typer.Option("auto", help="Target column name, auto, or none"),
    out: str = typer.Option("reports", help="Output reports directory"),
    max_rows_profile: int = typer.Option(200_000),
    max_rows_model: int = typer.Option(200_000),
    max_columns: int = typer.Option(500),
    random_seed: int = typer.Option(42),
    device: str = typer.Option("auto", help="auto|cpu|cuda"),
    sep: str | None = typer.Option(None, help="Optional CSV separator for local files"),
    unsup_learnability: str | None = typer.Option(None, help="on|off; default auto (on when no supervised target)"),
    rcpl_target_cols: int = typer.Option(10),
    rcpl_repeats: int = typer.Option(3),
    rcpl_max_rows: int = typer.Option(200_000),
    rcpl_cv_folds: int = typer.Option(3),
    rcpl_seed: int = typer.Option(42),
    rcpl_max_cat_card: int = typer.Option(200),
    rcpl_skip_id_threshold: float = typer.Option(0.98),
    save_row_ids: bool = typer.Option(False),
    enable_profiling: bool = typer.Option(False),
) -> None:
    configure_logging()
    cfg = GlobalConfig(
        max_rows_profile=max_rows_profile,
        max_rows_model=max_rows_model,
        max_columns=max_columns,
        random_seed=random_seed,
        device=device,
        unsup_learnability=unsup_learnability or "auto",
        rcpl_target_cols=rcpl_target_cols,
        rcpl_repeats=rcpl_repeats,
        rcpl_max_rows=rcpl_max_rows,
        rcpl_cv_folds=rcpl_cv_folds,
        rcpl_seed=rcpl_seed,
        rcpl_max_cat_card=rcpl_max_cat_card,
        rcpl_skip_id_threshold=rcpl_skip_id_threshold,
    )
    spec = DatasetSpec(name=Path(path).stem, source="local", path=path, target=target)
    report, row = _evaluate_with_progress(
        spec,
        cfg,
        out,
        save_row_ids=save_row_ids,
        enable_profiling=enable_profiling,
        sep=sep,
        unsup_learnability=unsup_learnability,
        rcpl_target_cols=rcpl_target_cols,
        rcpl_repeats=rcpl_repeats,
        rcpl_max_rows=rcpl_max_rows,
        rcpl_cv_folds=rcpl_cv_folds,
        rcpl_seed=rcpl_seed,
        rcpl_max_cat_card=rcpl_max_cat_card,
        rcpl_skip_id_threshold=rcpl_skip_id_threshold,
    )
    write_summary_csv(out, [row])
    _print_result(report)


@evaluate_app.command("openml")
def evaluate_openml(
    dataset_id: int = typer.Option(..., help="OpenML dataset id"),
    target: str | None = typer.Option("auto", help="Target column name, auto, or none"),
    out: str = typer.Option("reports", help="Output reports directory"),
    max_rows_profile: int = typer.Option(200_000),
    max_rows_model: int = typer.Option(200_000),
    max_columns: int = typer.Option(500),
    random_seed: int = typer.Option(42),
    device: str = typer.Option("auto", help="auto|cpu|cuda"),
    unsup_learnability: str | None = typer.Option(None, help="on|off; default auto (on when no supervised target)"),
    rcpl_target_cols: int = typer.Option(10),
    rcpl_repeats: int = typer.Option(3),
    rcpl_max_rows: int = typer.Option(200_000),
    rcpl_cv_folds: int = typer.Option(3),
    rcpl_seed: int = typer.Option(42),
    rcpl_max_cat_card: int = typer.Option(200),
    rcpl_skip_id_threshold: float = typer.Option(0.98),
    save_row_ids: bool = typer.Option(False),
    enable_profiling: bool = typer.Option(False),
) -> None:
    configure_logging()
    cfg = GlobalConfig(
        max_rows_profile=max_rows_profile,
        max_rows_model=max_rows_model,
        max_columns=max_columns,
        random_seed=random_seed,
        device=device,
        unsup_learnability=unsup_learnability or "auto",
        rcpl_target_cols=rcpl_target_cols,
        rcpl_repeats=rcpl_repeats,
        rcpl_max_rows=rcpl_max_rows,
        rcpl_cv_folds=rcpl_cv_folds,
        rcpl_seed=rcpl_seed,
        rcpl_max_cat_card=rcpl_max_cat_card,
        rcpl_skip_id_threshold=rcpl_skip_id_threshold,
    )
    spec = DatasetSpec(name=f"openml_{dataset_id}", source="openml", dataset_id=dataset_id, target=target)
    report, row = _evaluate_with_progress(
        spec,
        cfg,
        out,
        save_row_ids=save_row_ids,
        enable_profiling=enable_profiling,
        unsup_learnability=unsup_learnability,
        rcpl_target_cols=rcpl_target_cols,
        rcpl_repeats=rcpl_repeats,
        rcpl_max_rows=rcpl_max_rows,
        rcpl_cv_folds=rcpl_cv_folds,
        rcpl_seed=rcpl_seed,
        rcpl_max_cat_card=rcpl_max_cat_card,
        rcpl_skip_id_threshold=rcpl_skip_id_threshold,
    )
    write_summary_csv(out, [row])
    _print_result(report)


@evaluate_app.command("kaggle")
def evaluate_kaggle(
    dataset: str = typer.Option(..., "--dataset", help="Kaggle dataset slug owner/name"),
    file: str = typer.Option(..., help="File inside dataset archive"),
    target: str | None = typer.Option("auto", help="Target column name, auto, or none"),
    out: str = typer.Option("reports", help="Output reports directory"),
    max_rows_profile: int = typer.Option(200_000),
    max_rows_model: int = typer.Option(200_000),
    max_columns: int = typer.Option(500),
    random_seed: int = typer.Option(42),
    device: str = typer.Option("auto", help="auto|cpu|cuda"),
    unsup_learnability: str | None = typer.Option(None, help="on|off; default auto (on when no supervised target)"),
    rcpl_target_cols: int = typer.Option(10),
    rcpl_repeats: int = typer.Option(3),
    rcpl_max_rows: int = typer.Option(200_000),
    rcpl_cv_folds: int = typer.Option(3),
    rcpl_seed: int = typer.Option(42),
    rcpl_max_cat_card: int = typer.Option(200),
    rcpl_skip_id_threshold: float = typer.Option(0.98),
    save_row_ids: bool = typer.Option(False),
    enable_profiling: bool = typer.Option(False),
) -> None:
    configure_logging()
    cfg = GlobalConfig(
        max_rows_profile=max_rows_profile,
        max_rows_model=max_rows_model,
        max_columns=max_columns,
        random_seed=random_seed,
        device=device,
        unsup_learnability=unsup_learnability or "auto",
        rcpl_target_cols=rcpl_target_cols,
        rcpl_repeats=rcpl_repeats,
        rcpl_max_rows=rcpl_max_rows,
        rcpl_cv_folds=rcpl_cv_folds,
        rcpl_seed=rcpl_seed,
        rcpl_max_cat_card=rcpl_max_cat_card,
        rcpl_skip_id_threshold=rcpl_skip_id_threshold,
    )
    spec = DatasetSpec(
        name=f"kaggle_{dataset.replace('/', '_')}",
        source="kaggle",
        slug=dataset,
        file=file,
        target=target,
    )
    report, row = _evaluate_with_progress(
        spec,
        cfg,
        out,
        save_row_ids=save_row_ids,
        enable_profiling=enable_profiling,
        unsup_learnability=unsup_learnability,
        rcpl_target_cols=rcpl_target_cols,
        rcpl_repeats=rcpl_repeats,
        rcpl_max_rows=rcpl_max_rows,
        rcpl_cv_folds=rcpl_cv_folds,
        rcpl_seed=rcpl_seed,
        rcpl_max_cat_card=rcpl_max_cat_card,
        rcpl_skip_id_threshold=rcpl_skip_id_threshold,
    )
    write_summary_csv(out, [row])
    _print_result(report)


@batch_app.callback(invoke_without_command=True)
def batch_default(
    ctx: typer.Context,
    config: str | None = typer.Option(None, help="YAML config path (legacy mode)"),
    out: str = typer.Option("reports", help="Output reports directory"),
    max_datasets: int | None = typer.Option(None, help="Max datasets to run"),
    unsup_learnability: str | None = typer.Option(None, help="on|off; default auto (on when no supervised target)"),
    rcpl_target_cols: int = typer.Option(10),
    rcpl_repeats: int = typer.Option(3),
    rcpl_max_rows: int = typer.Option(200_000),
    rcpl_cv_folds: int = typer.Option(3),
    rcpl_seed: int = typer.Option(42),
    rcpl_max_cat_card: int = typer.Option(200),
    rcpl_skip_id_threshold: float = typer.Option(0.98),
    save_row_ids: bool = typer.Option(False),
    enable_profiling: bool = typer.Option(False),
) -> None:
    if ctx.invoked_subcommand is not None:
        return
    if config is None:
        raise typer.BadParameter("Provide --config or use a batch subcommand")

    configure_logging()
    cfg = load_config(config)
    specs = _expand_yaml_specs(cfg)
    if max_datasets is not None:
        specs = specs[:max_datasets]
    _run_batch(
        specs,
        cfg.global_,
        out,
        save_row_ids=save_row_ids,
        enable_profiling=enable_profiling,
        unsup_learnability=unsup_learnability,
        rcpl_target_cols=rcpl_target_cols,
        rcpl_repeats=rcpl_repeats,
        rcpl_max_rows=rcpl_max_rows,
        rcpl_cv_folds=rcpl_cv_folds,
        rcpl_seed=rcpl_seed,
        rcpl_max_cat_card=rcpl_max_cat_card,
        rcpl_skip_id_threshold=rcpl_skip_id_threshold,
    )


@batch_app.command("yaml")
def batch_yaml(
    config: str = typer.Option(..., help="YAML config path"),
    out: str = typer.Option("reports", help="Output reports directory"),
    max_datasets: int | None = typer.Option(None),
    unsup_learnability: str | None = typer.Option(None, help="on|off; default auto (on when no supervised target)"),
    rcpl_target_cols: int = typer.Option(10),
    rcpl_repeats: int = typer.Option(3),
    rcpl_max_rows: int = typer.Option(200_000),
    rcpl_cv_folds: int = typer.Option(3),
    rcpl_seed: int = typer.Option(42),
    rcpl_max_cat_card: int = typer.Option(200),
    rcpl_skip_id_threshold: float = typer.Option(0.98),
    save_row_ids: bool = typer.Option(False),
    enable_profiling: bool = typer.Option(False),
) -> None:
    configure_logging()
    cfg = load_config(config)
    specs = _expand_yaml_specs(cfg)
    if max_datasets is not None:
        specs = specs[:max_datasets]
    _run_batch(
        specs,
        cfg.global_,
        out,
        save_row_ids=save_row_ids,
        enable_profiling=enable_profiling,
        unsup_learnability=unsup_learnability,
        rcpl_target_cols=rcpl_target_cols,
        rcpl_repeats=rcpl_repeats,
        rcpl_max_rows=rcpl_max_rows,
        rcpl_cv_folds=rcpl_cv_folds,
        rcpl_seed=rcpl_seed,
        rcpl_max_cat_card=rcpl_max_cat_card,
        rcpl_skip_id_threshold=rcpl_skip_id_threshold,
    )


@batch_app.command("openml")
def batch_openml(
    dataset_ids: list[int] = typer.Option(
        [],
        "--dataset-ids",
        help="OpenML dataset ID (repeatable): --dataset-ids 61 --dataset-ids 15",
    ),
    dataset_id_list: list[str] = typer.Option(
        [],
        "--dataset-id-list",
        help="Comma/space-separated OpenML IDs (repeatable), e.g. '61,15,37'",
    ),
    dataset_ids_file: str | None = typer.Option(
        None,
        "--dataset-ids-file",
        help="Path to .txt/.json/.yaml containing OpenML dataset IDs",
    ),
    target: str | None = typer.Option("auto", help="Target column name, auto, or none"),
    out: str = typer.Option("reports"),
    device: str = typer.Option("auto", help="auto|cpu|cuda"),
    batch_concurrency: int = typer.Option(1),
    unsup_learnability: str | None = typer.Option(None, help="on|off; default auto (on when no supervised target)"),
    rcpl_target_cols: int = typer.Option(10),
    rcpl_repeats: int = typer.Option(3),
    rcpl_max_rows: int = typer.Option(200_000),
    rcpl_cv_folds: int = typer.Option(3),
    rcpl_seed: int = typer.Option(42),
    rcpl_max_cat_card: int = typer.Option(200),
    rcpl_skip_id_threshold: float = typer.Option(0.98),
) -> None:
    configure_logging()
    cfg = GlobalConfig(
        device=device,
        batch_concurrency=batch_concurrency,
        unsup_learnability=unsup_learnability or "auto",
        rcpl_target_cols=rcpl_target_cols,
        rcpl_repeats=rcpl_repeats,
        rcpl_max_rows=rcpl_max_rows,
        rcpl_cv_folds=rcpl_cv_folds,
        rcpl_seed=rcpl_seed,
        rcpl_max_cat_card=rcpl_max_cat_card,
        rcpl_skip_id_threshold=rcpl_skip_id_threshold,
    )
    resolved_ids = parse_openml_dataset_ids(
        dataset_ids=dataset_ids,
        dataset_id_lists=dataset_id_list,
        dataset_ids_file=dataset_ids_file,
    )
    if not resolved_ids:
        raise typer.BadParameter(
            "No OpenML dataset IDs resolved. Provide --dataset-ids, --dataset-id-list, or --dataset-ids-file."
        )

    specs = [
        DatasetSpec(name=f"openml_{dsid}", source="openml", dataset_id=dsid, target=target)
        for dsid in resolved_ids
    ]
    _run_batch(
        specs,
        cfg,
        out,
        save_row_ids=False,
        enable_profiling=False,
        unsup_learnability=unsup_learnability,
        rcpl_target_cols=rcpl_target_cols,
        rcpl_repeats=rcpl_repeats,
        rcpl_max_rows=rcpl_max_rows,
        rcpl_cv_folds=rcpl_cv_folds,
        rcpl_seed=rcpl_seed,
        rcpl_max_cat_card=rcpl_max_cat_card,
        rcpl_skip_id_threshold=rcpl_skip_id_threshold,
    )


@batch_app.command("kaggle")
def batch_kaggle(
    datasets: list[str] = typer.Option(
        [],
        "--datasets",
        help="Kaggle dataset entry (repeatable), format slug:file",
    ),
    datasets_file: str | None = typer.Option(
        None,
        "--datasets-file",
        help="Path to .txt/.json/.yaml containing Kaggle entries",
    ),
    target: str | None = typer.Option("auto", help="Target column name, auto, or none"),
    out: str = typer.Option("reports"),
    device: str = typer.Option("auto", help="auto|cpu|cuda"),
    batch_concurrency: int = typer.Option(1),
    unsup_learnability: str | None = typer.Option(None, help="on|off; default auto (on when no supervised target)"),
    rcpl_target_cols: int = typer.Option(10),
    rcpl_repeats: int = typer.Option(3),
    rcpl_max_rows: int = typer.Option(200_000),
    rcpl_cv_folds: int = typer.Option(3),
    rcpl_seed: int = typer.Option(42),
    rcpl_max_cat_card: int = typer.Option(200),
    rcpl_skip_id_threshold: float = typer.Option(0.98),
) -> None:
    configure_logging()
    cfg = GlobalConfig(
        device=device,
        batch_concurrency=batch_concurrency,
        unsup_learnability=unsup_learnability or "auto",
        rcpl_target_cols=rcpl_target_cols,
        rcpl_repeats=rcpl_repeats,
        rcpl_max_rows=rcpl_max_rows,
        rcpl_cv_folds=rcpl_cv_folds,
        rcpl_seed=rcpl_seed,
        rcpl_max_cat_card=rcpl_max_cat_card,
        rcpl_skip_id_threshold=rcpl_skip_id_threshold,
    )
    resolved_specs = parse_kaggle_dataset_specs(datasets=datasets, datasets_file=datasets_file)
    if not resolved_specs:
        raise typer.BadParameter("No Kaggle datasets resolved. Provide --datasets or --datasets-file.")

    specs: list[DatasetSpec] = []
    for slug, file in resolved_specs:
        specs.append(
            DatasetSpec(
                name=f"kaggle_{slug.replace('/', '_')}_{Path(file).stem}",
                source="kaggle",
                slug=slug,
                file=file,
                target=target,
            )
        )
    _run_batch(
        specs,
        cfg,
        out,
        save_row_ids=False,
        enable_profiling=False,
        unsup_learnability=unsup_learnability,
        rcpl_target_cols=rcpl_target_cols,
        rcpl_repeats=rcpl_repeats,
        rcpl_max_rows=rcpl_max_rows,
        rcpl_cv_folds=rcpl_cv_folds,
        rcpl_seed=rcpl_seed,
        rcpl_max_cat_card=rcpl_max_cat_card,
        rcpl_skip_id_threshold=rcpl_skip_id_threshold,
    )


@batch_app.command("dir")
def batch_dir(
    path: str = typer.Option(..., help="Directory with CSV files"),
    pattern: str = typer.Option("*.csv", help="Glob pattern"),
    target: str | None = typer.Option("none", help="Target column name, auto, or none"),
    out: str = typer.Option("reports"),
    device: str = typer.Option("auto", help="auto|cpu|cuda"),
    batch_concurrency: int = typer.Option(1),
    sep: str | None = typer.Option(None, help="Optional fixed separator for all files"),
    unsup_learnability: str | None = typer.Option(None, help="on|off; default auto (on when no supervised target)"),
    rcpl_target_cols: int = typer.Option(10),
    rcpl_repeats: int = typer.Option(3),
    rcpl_max_rows: int = typer.Option(200_000),
    rcpl_cv_folds: int = typer.Option(3),
    rcpl_seed: int = typer.Option(42),
    rcpl_max_cat_card: int = typer.Option(200),
    rcpl_skip_id_threshold: float = typer.Option(0.98),
) -> None:
    configure_logging()
    cfg = GlobalConfig(
        device=device,
        batch_concurrency=batch_concurrency,
        unsup_learnability=unsup_learnability or "auto",
        rcpl_target_cols=rcpl_target_cols,
        rcpl_repeats=rcpl_repeats,
        rcpl_max_rows=rcpl_max_rows,
        rcpl_cv_folds=rcpl_cv_folds,
        rcpl_seed=rcpl_seed,
        rcpl_max_cat_card=rcpl_max_cat_card,
        rcpl_skip_id_threshold=rcpl_skip_id_threshold,
    )
    entries = list_local_datasets(path=path, pattern=pattern)
    specs = [DatasetSpec(name=e["name"], source="local", path=e["path"], target=target) for e in entries]
    _run_batch(
        specs,
        cfg,
        out,
        save_row_ids=False,
        enable_profiling=False,
        sep=sep,
        unsup_learnability=unsup_learnability,
        rcpl_target_cols=rcpl_target_cols,
        rcpl_repeats=rcpl_repeats,
        rcpl_max_rows=rcpl_max_rows,
        rcpl_cv_folds=rcpl_cv_folds,
        rcpl_seed=rcpl_seed,
        rcpl_max_cat_card=rcpl_max_cat_card,
        rcpl_skip_id_threshold=rcpl_skip_id_threshold,
    )


@batch_app.command("mixed")
def batch_mixed(
    openml_dataset_ids: list[int] = typer.Option(
        [],
        "--openml-dataset-ids",
        help="OpenML dataset ID (repeatable)",
    ),
    openml_dataset_id_list: list[str] = typer.Option(
        [],
        "--openml-dataset-id-list",
        help="Comma/space-separated OpenML IDs (repeatable), e.g. '61,15,37'",
    ),
    openml_dataset_ids_file: str | None = typer.Option(
        None,
        "--openml-dataset-ids-file",
        help="Path to .txt/.json/.yaml containing OpenML dataset IDs",
    ),
    kaggle_datasets: list[str] = typer.Option(
        [],
        "--kaggle-datasets",
        help="Kaggle dataset entry (repeatable), format slug:file",
    ),
    kaggle_datasets_file: str | None = typer.Option(
        None,
        "--kaggle-datasets-file",
        help="Path to .txt/.json/.yaml containing Kaggle dataset entries",
    ),
    local_files: list[str] = typer.Option(
        [],
        "--local-files",
        help="Local file path (repeatable)",
    ),
    local_file_list: list[str] = typer.Option(
        [],
        "--local-file-list",
        help="Comma/space-separated local file paths (repeatable)",
    ),
    local_files_file: str | None = typer.Option(
        None,
        "--local-files-file",
        help="Path to .txt/.json/.yaml containing local file paths",
    ),
    local_dirs: list[str] = typer.Option(
        [],
        "--local-dirs",
        help="Local directory path (repeatable)",
    ),
    local_dirs_file: str | None = typer.Option(
        None,
        "--local-dirs-file",
        help="Path to .txt/.json/.yaml containing local directory paths",
    ),
    pattern: str = typer.Option("*.csv", help="Glob pattern for local directories"),
    target: str | None = typer.Option("auto", help="Global target applied to all datasets (auto|none|<col>)"),
    out: str = typer.Option("reports"),
    device: str = typer.Option("auto", help="auto|cpu|cuda"),
    batch_concurrency: int = typer.Option(1),
    sep: str | None = typer.Option(None, help="Optional fixed separator for local files"),
    unsup_learnability: str | None = typer.Option(None, help="on|off; default auto (on when no supervised target)"),
    rcpl_target_cols: int = typer.Option(10),
    rcpl_repeats: int = typer.Option(3),
    rcpl_max_rows: int = typer.Option(200_000),
    rcpl_cv_folds: int = typer.Option(5),
    rcpl_seed: int = typer.Option(42),
    rcpl_max_cat_card: int = typer.Option(200),
    rcpl_skip_id_threshold: float = typer.Option(0.98),
) -> None:
    configure_logging()
    cfg = GlobalConfig(
        device=device,
        batch_concurrency=batch_concurrency,
        unsup_learnability=unsup_learnability or "auto",
        rcpl_target_cols=rcpl_target_cols,
        rcpl_repeats=rcpl_repeats,
        rcpl_max_rows=rcpl_max_rows,
        rcpl_cv_folds=rcpl_cv_folds,
        rcpl_seed=rcpl_seed,
        rcpl_max_cat_card=rcpl_max_cat_card,
        rcpl_skip_id_threshold=rcpl_skip_id_threshold,
    )

    specs: list[DatasetSpec] = []

    openml_ids = parse_openml_dataset_ids(
        dataset_ids=openml_dataset_ids,
        dataset_id_lists=openml_dataset_id_list,
        dataset_ids_file=openml_dataset_ids_file,
    )
    for dsid in openml_ids:
        specs.append(DatasetSpec(name=f"openml_{dsid}", source="openml", dataset_id=dsid, target=target))

    kaggle_specs = parse_kaggle_dataset_specs(
        datasets=kaggle_datasets,
        datasets_file=kaggle_datasets_file,
    )
    for slug, file in kaggle_specs:
        specs.append(
            DatasetSpec(
                name=f"kaggle_{slug.replace('/', '_')}_{Path(file).stem}",
                source="kaggle",
                slug=slug,
                file=file,
                target=target,
            )
        )

    for path in parse_local_file_paths(
        local_files=local_files,
        local_file_lists=local_file_list,
        local_files_file=local_files_file,
    ):
        specs.append(DatasetSpec(name=Path(path).stem, source="local", path=path, target=target))

    for directory in parse_local_dir_paths(local_dirs=local_dirs, local_dirs_file=local_dirs_file):
        for entry in list_local_datasets(path=directory, pattern=pattern):
            specs.append(DatasetSpec(name=entry["name"], source="local", path=entry["path"], target=target))

    if not specs:
        raise typer.BadParameter(
            "No datasets resolved. Provide at least one of: "
            "--openml-dataset-ids/--openml-dataset-id-list/--openml-dataset-ids-file, "
            "--kaggle-datasets/--kaggle-datasets-file, "
            "--local-files/--local-file-list/--local-files-file, "
            "--local-dirs/--local-dirs-file."
        )

    _run_batch(
        specs,
        cfg,
        out,
        save_row_ids=False,
        enable_profiling=False,
        sep=sep,
        unsup_learnability=unsup_learnability,
        rcpl_target_cols=rcpl_target_cols,
        rcpl_repeats=rcpl_repeats,
        rcpl_max_rows=rcpl_max_rows,
        rcpl_cv_folds=rcpl_cv_folds,
        rcpl_seed=rcpl_seed,
        rcpl_max_cat_card=rcpl_max_cat_card,
        rcpl_skip_id_threshold=rcpl_skip_id_threshold,
    )


@app.command("leaderboard")
def leaderboard(
    reports_dir: str = typer.Option("reports", help="Reports directory containing dataset folders"),
) -> None:
    configure_logging()
    df = build_leaderboard(reports_dir)
    console.print(f"Leaderboard written: {reports_dir}/leaderboard.csv ({len(df)} rows)")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
