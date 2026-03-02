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
  --bg: #f3f7fc;
  --panel: #ffffff;
  --ink: #13213b;
  --muted: #5a6b87;
  --line: #d7e2f3;
  --accent: #1083ff;
  --good: #0f8a4b;
  --warn: #b87000;
  --bad: #c0352b;
  --shadow: 0 10px 28px rgba(15, 34, 66, 0.08);
}
body {
  font-family: "Avenir Next", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  background:
    radial-gradient(circle at 5% 0%, #dff1ff 0%, transparent 28%),
    radial-gradient(circle at 95% 10%, #e8ebff 0%, transparent 24%),
    var(--bg);
  color: var(--ink);
  margin: 0;
}
.container { max-width: 1120px; margin: 24px auto 40px; padding: 0 16px; }
.panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 14px;
  box-shadow: var(--shadow);
  padding: 18px;
  margin-bottom: 16px;
}
.hero { display: flex; gap: 14px; align-items: center; justify-content: space-between; flex-wrap: wrap; }
.chips { display: flex; gap: 8px; flex-wrap: wrap; }
.chip {
  border-radius: 999px;
  border: 1px solid var(--line);
  background: #f8fbff;
  color: var(--muted);
  padding: 4px 10px;
  font-size: 12px;
  font-weight: 700;
}
.chip.good { color: var(--good); border-color: #b7e4cd; background: #effbf5; }
.chip.warn { color: var(--warn); border-color: #f0d2a3; background: #fff7ea; }
.chip.bad { color: var(--bad); border-color: #f3c2bf; background: #fff0ef; }
.grid { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(205px, 1fr)); }
.kpi { background: #fbfdff; border: 1px solid var(--line); border-radius: 12px; padding: 12px; }
.k-label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }
.k-value { font-size: 28px; font-weight: 800; margin-top: 4px; }
.bar { height: 8px; border-radius: 999px; background: #e8eef9; overflow: hidden; margin-top: 8px; }
.bar > span { display: block; height: 100%; background: linear-gradient(90deg, #15d1cf, #1083ff); }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.small { color: var(--muted); font-size: 0.92rem; margin: 0; }
h1 { margin: 2px 0 8px; font-size: 1.7rem; }
h2 { margin: 2px 0 12px; font-size: 1.08rem; letter-spacing: .02em; }
ul { margin: 8px 0 0; padding-left: 18px; }
li { margin: 6px 0; }
pre {
  margin: 0;
  overflow: auto;
  max-height: 420px;
  background: #0f1d35;
  color: #d5e7ff;
  border-radius: 12px;
  padding: 12px;
  font-size: 12px;
  line-height: 1.45;
}
details > summary {
  cursor: pointer;
  color: var(--muted);
  font-weight: 700;
  margin-bottom: 10px;
}
@media (max-width: 640px) {
  .container { margin-top: 14px; }
  h1 { font-size: 1.4rem; }
  .k-value { font-size: 24px; }
}
</style>
</head>
<body>
  <div class="container">
    <div class="panel">
      <div class="hero">
        <div>
          <h1>Tabular Dataset Quality Report</h1>
          <p class="small">Dataset: <span class="mono">{{ dataset_slug }}</span></p>
          <p class="small">Source: {{ report.metadata.source }} | Retrieved: {{ report.metadata.retrieved_at }}</p>
        </div>
        <div class="chips">
          <span class="chip {% if report.status == 'FAILED' %}bad{% else %}good{% endif %}">
            {{ report.status or 'OK' }}
          </span>
          <span class="chip {% if report.warnings|length > 0 %}warn{% endif %}">
            Warnings {{ report.warnings|length }}
          </span>
          <span class="chip {% if report.errors|length > 0 %}bad{% endif %}">
            Errors {{ report.errors|length }}
          </span>
        </div>
      </div>
    </div>

    <div class="panel">
      <h2>Quality Scores</h2>
      <div class="grid">
        <div class="kpi">
          <div class="k-label">Overall</div>
          <div class="k-value">{{ report.scores.quality_score }}</div>
          <div class="bar"><span style="width: {{ report.scores.quality_score or 0 }}%;"></span></div>
        </div>
        <div class="kpi">
          <div class="k-label">Cleanliness</div>
          <div class="k-value">{{ report.scores.cleanliness_score }}</div>
          <div class="bar"><span style="width: {{ report.scores.cleanliness_score or 0 }}%;"></span></div>
        </div>
        <div class="kpi">
          <div class="k-label">Structure</div>
          <div class="k-value">{{ report.scores.structure_score }}</div>
          <div class="bar"><span style="width: {{ report.scores.structure_score or 0 }}%;"></span></div>
        </div>
        <div class="kpi">
          <div class="k-label">Learnability</div>
          <div class="k-value">{{ report.scores.learnability_score }}</div>
          <div class="bar"><span style="width: {{ report.scores.learnability_score or 0 }}%;"></span></div>
        </div>
        <div class="kpi">
          <div class="k-label">Label Quality</div>
          <div class="k-value">{{ report.scores.label_quality_score }}</div>
          <div class="bar"><span style="width: {{ report.scores.label_quality_score or 0 }}%;"></span></div>
        </div>
      </div>
    </div>

    <div class="panel grid" style="grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));">
      <div>
      <h2>Warnings</h2>
      {% if report.warnings %}
      <ul>{% for w in report.warnings %}<li>{{ w }}</li>{% endfor %}</ul>
      {% else %}<p>None</p>{% endif %}
      </div>
      <div>
      <h2>Errors</h2>
      {% if report.errors %}
      <ul>{% for e in report.errors %}<li>{{ e }}</li>{% endfor %}</ul>
      {% else %}<p>None</p>{% endif %}
      </div>
    </div>

    <div class="panel">
      <h2>Basic Stats</h2>
      <details open>
        <summary>Expand/Collapse JSON</summary>
      <pre>{{ basic_stats_json }}</pre>
      </details>
    </div>

    <div class="panel">
      <h2>Check Details</h2>
      <details open>
        <summary>Expand/Collapse JSON</summary>
      <pre>{{ checks_json }}</pre>
      </details>
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
