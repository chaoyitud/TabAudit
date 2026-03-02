from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
<meta charset=\"utf-8\" />
<title>TabAudit Leaderboard</title>
<style>
body {
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif;
  margin: 24px;
  background: #f4f7fb;
  color: #162034;
}
h1 { margin-bottom: 8px; }
table { border-collapse: collapse; width: 100%; background: white; border-radius: 8px; overflow: hidden; }
th, td { border: 1px solid #d8e0ef; padding: 8px; text-align: left; font-size: 0.9rem; }
th { background: #e9f0ff; cursor: pointer; }
tr:nth-child(even) { background: #fafcff; }
.ok { color: #106d39; font-weight: 600; }
.failed { color: #b3261e; font-weight: 600; }
</style>
</head>
<body>
<h1>TabAudit Leaderboard</h1>
<p>Click column headers to sort.</p>
<table id=\"leaderboard\"></table>
<script>
const data = {{ data_json }};
const columns = {{ columns_json }};
let sortCol = 'quality_score';
let sortDesc = true;

function render() {
  const table = document.getElementById('leaderboard');
  const sorted = [...data].sort((a,b) => {
    const av = a[sortCol];
    const bv = b[sortCol];
    const an = parseFloat(av);
    const bn = parseFloat(bv);
    const bothNum = !Number.isNaN(an) && !Number.isNaN(bn);
    if (bothNum) return sortDesc ? bn - an : an - bn;
    return sortDesc ? String(bv).localeCompare(String(av)) : String(av).localeCompare(String(bv));
  });
  let html = '<thead><tr>';
  columns.forEach(c => { html += `<th onclick=\"setSort('${c}')\">${c}</th>`; });
  html += '</tr></thead><tbody>';
  sorted.forEach(row => {
    html += '<tr>';
    columns.forEach(c => {
      const cls = c === 'status' && row[c] === 'FAILED' ? 'failed' : (c === 'status' ? 'ok' : '');
      html += `<td class=\"${cls}\">${row[c] ?? ''}</td>`;
    });
    html += '</tr>';
  });
  html += '</tbody>';
  table.innerHTML = html;
}

function setSort(col) {
  if (sortCol === col) sortDesc = !sortDesc;
  else { sortCol = col; sortDesc = col === 'quality_score' || col === 'learnability_score'; }
  render();
}

render();
</script>
</body>
</html>
"""


def _safe_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def report_to_row(report: dict[str, Any]) -> dict[str, Any]:
    scores = report.get("scores", {})
    checks = report.get("checks", {})
    baseline = checks.get("baseline_model", {})
    unsup = checks.get("unsupervised_learnability", {})

    backend = baseline.get("backend")
    device = baseline.get("device")
    runtime_sec = baseline.get("training_time_sec")
    primary_metric_cv = baseline.get("primary_metric_cv")
    if (not baseline.get("available")) and unsup:
        backend = backend or ("rcpl" if unsup.get("available") else "rcpl_unavailable")
        device = device or report.get("metadata", {}).get("device")
        runtime_sec = runtime_sec if runtime_sec is not None else unsup.get("training_time_sec")
        primary_metric_cv = primary_metric_cv if primary_metric_cv is not None else unsup.get("rcpl_score_norm")

    status = report.get("status") or ("FAILED" if report.get("errors") else "OK")
    error_msg = ""
    if status == "FAILED":
        if report.get("error_message"):
            error_msg = str(report.get("error_message"))[:220]
        elif report.get("errors"):
            error_msg = str(report.get("errors")[0])[:220]

    return {
        "dataset_name": report.get("dataset_name"),
        "dataset_slug": report.get("dataset_slug"),
        "source": report.get("metadata", {}).get("source"),
        "n_rows": report.get("basic_stats", {}).get("n_rows"),
        "n_cols": report.get("basic_stats", {}).get("n_cols"),
        "target": report.get("target") if report.get("target") is not None else "none",
        "quality_score": _safe_float(scores.get("quality_score")),
        "cleanliness_score": _safe_float(scores.get("cleanliness_score")),
        "structure_score": _safe_float(scores.get("structure_score")),
        "learnability_score": _safe_float(scores.get("learnability_score")),
        "label_quality_score": _safe_float(scores.get("label_quality_score")),
        "warnings_count": len(report.get("warnings", [])),
        "primary_metric_cv": primary_metric_cv,
        "accuracy_cv": baseline.get("accuracy_cv"),
        "rmse_cv": baseline.get("rmse_cv"),
        "backend": backend,
        "device": device,
        "runtime_sec": runtime_sec,
        "status": status,
        "error": error_msg,
    }


def build_leaderboard(reports_dir: str | Path) -> pd.DataFrame:
    base = Path(reports_dir)
    report_files = sorted(base.glob("*/report.json"))
    rows: list[dict[str, Any]] = []

    for report_file in report_files:
        try:
            report = json.loads(report_file.read_text())
            rows.append(report_to_row(report))
        except Exception as exc:
            rows.append(
                {
                    "dataset_name": report_file.parent.name,
                    "dataset_slug": report_file.parent.name,
                    "source": "unknown",
                    "n_rows": None,
                    "n_cols": None,
                    "target": "none",
                    "quality_score": None,
                    "cleanliness_score": None,
                    "structure_score": None,
                    "learnability_score": None,
                    "label_quality_score": None,
                    "warnings_count": 0,
                    "primary_metric_cv": None,
                    "accuracy_cv": None,
                    "rmse_cv": None,
                    "backend": None,
                    "device": None,
                    "runtime_sec": None,
                    "status": "FAILED",
                    "error": f"invalid report.json: {exc.__class__.__name__}",
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "dataset_name",
                "dataset_slug",
                "source",
                "n_rows",
                "n_cols",
                "target",
                "quality_score",
                "cleanliness_score",
                "structure_score",
                "learnability_score",
                "label_quality_score",
                "warnings_count",
                "primary_metric_cv",
                "accuracy_cv",
                "rmse_cv",
                "backend",
                "device",
                "runtime_sec",
                "status",
                "error",
            ]
        )

    df = df.sort_values(by=["quality_score", "learnability_score"], ascending=[False, False], na_position="last")

    csv_path = base / "leaderboard.csv"
    html_path = base / "leaderboard.html"
    parquet_path = base / "summary.parquet"

    df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception:
        pass

    html = HTML_TEMPLATE.replace("{{ data_json }}", df.fillna("").to_json(orient="records"))
    html = html.replace("{{ columns_json }}", json.dumps(df.columns.tolist()))
    html_path.write_text(html)

    return df
