from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from jinja2 import Template

from tab_audit.io.cache import ensure_dir

TEMPLATE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Tab Audit Report - {{ dataset_slug }}</title>
<style>
:root {
  --bg: #f6f8fb;
  --panel: #ffffff;
  --ink: #182033;
  --muted: #516079;
  --good: #137333;
  --warn: #9a6700;
  --bad: #b3261e;
}
body {
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif;
  background: linear-gradient(120deg, #f6f8fb, #eef3ff);
  color: var(--ink);
  margin: 0;
}
.container { max-width: 1080px; margin: 24px auto; padding: 0 16px; }
.panel {
  background: var(--panel);
  border-radius: 10px;
  box-shadow: 0 2px 10px rgba(12, 20, 39, .08);
  padding: 16px;
  margin-bottom: 16px;
}
.grid { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }
.kpi { background: #f9fbff; border: 1px solid #dce5f5; border-radius: 8px; padding: 10px; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.small { color: var(--muted); font-size: 0.9rem; }
ul { margin-top: 8px; }
</style>
</head>
<body>
  <div class="container">
    <div class="panel">
      <h1>Tabular Dataset Quality Report</h1>
      <p class="small">Dataset: <span class="mono">{{ dataset_slug }}</span></p>
      <p class="small">Source: {{ report.metadata.source }} | Retrieved: {{ report.metadata.retrieved_at }}</p>
    </div>

    <div class="panel">
      <h2>Scores</h2>
      <div class="grid">
        <div class="kpi"><b>Overall</b><br>{{ report.scores.quality_score }}</div>
        <div class="kpi"><b>Cleanliness</b><br>{{ report.scores.cleanliness_score }}</div>
        <div class="kpi"><b>Structure</b><br>{{ report.scores.structure_score }}</div>
        <div class="kpi"><b>Learnability</b><br>{{ report.scores.learnability_score }}</div>
        <div class="kpi"><b>Label Quality</b><br>{{ report.scores.label_quality_score }}</div>
      </div>
    </div>

    <div class="panel">
      <h2>Warnings</h2>
      {% if report.warnings %}
      <ul>{% for w in report.warnings %}<li>{{ w }}</li>{% endfor %}</ul>
      {% else %}<p>None</p>{% endif %}
      <h2>Errors</h2>
      {% if report.errors %}
      <ul>{% for e in report.errors %}<li>{{ e }}</li>{% endfor %}</ul>
      {% else %}<p>None</p>{% endif %}
    </div>

    <div class="panel">
      <h2>Basic Stats</h2>
      <pre>{{ basic_stats_json }}</pre>
    </div>

    <div class="panel">
      <h2>Check Details</h2>
      <pre>{{ checks_json }}</pre>
    </div>
  </div>
</body>
</html>
"""


def write_dataset_report_html(out_dir: str | Path, dataset_slug: str, payload: dict[str, Any]) -> Path:
    folder = ensure_dir(Path(out_dir) / dataset_slug)
    path = folder / "report.html"
    template = Template(TEMPLATE)
    content = template.render(
        dataset_slug=html.escape(dataset_slug),
        report=payload,
        basic_stats_json=json.dumps(payload.get("basic_stats", {}), indent=2, default=str),
        checks_json=json.dumps(payload.get("checks", {}), indent=2, default=str),
    )
    path.write_text(content)
    return path
