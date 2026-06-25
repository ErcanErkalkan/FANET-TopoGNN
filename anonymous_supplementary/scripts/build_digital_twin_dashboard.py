from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/digital_twin_dashboard"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(row: pd.Series, key: str, digits: int = 3) -> str:
    return f"{float(row[key]):.{digits}f}"


def _copy_asset(source: Path, assets_dir: Path) -> str:
    if not source.is_file():
        raise FileNotFoundError(source)
    assets_dir.mkdir(parents=True, exist_ok=True)
    destination = assets_dir / source.name
    shutil.copy2(source, destination)
    return f"assets/{destination.name}"


def _load_payload() -> dict:
    paper_summary = _read_json(ROOT / "outputs/paper_like_submission/summary.json")
    compact_summary = _read_json(ROOT / "outputs/publication_compact/summary.json")
    neural_summary = _read_json(ROOT / "outputs/publication_neural_5seed_extension/summary.json")
    metrics = pd.read_csv(ROOT / "outputs/paper_like_submission/metrics_overall.csv")
    ktg = metrics.loc[metrics["Model"] == "Kinetic-TopoGuard"].iloc[0]
    union = metrics.loc[metrics["Model"] == "Union-Find detection oracle"].iloc[0]

    operating = pd.read_csv(ROOT / "outputs/operating_point/operating_point_summary.csv")
    deployable = operating.loc[operating["policy"].str.startswith("Deployable")].iloc[0]
    strict = operating.loc[operating["policy"].str.startswith("Strict")].iloc[0]

    a2a_metrics = pd.read_csv(ROOT / "outputs/uav_to_uav_mmwave_validation/uav_to_uav_link_model_metrics.csv")
    a2a_logistic = a2a_metrics.loc[a2a_metrics["model"] == "logistic A2A RF model"].iloc[0]
    a2a_baseline = a2a_metrics.loc[a2a_metrics["model"] == "training-prior baseline"].iloc[0]
    a2a_summary = pd.read_csv(ROOT / "outputs/uav_to_uav_mmwave_validation/uav_to_uav_forestry_beta0_summary.csv")

    aerpaw_availability = pd.read_csv(ROOT / "outputs/aerpaw_cellular_validation/aerpaw_lte_availability_metrics.csv")
    aerpaw_throughput = pd.read_csv(ROOT / "outputs/aerpaw_cellular_validation/aerpaw_throughput_metrics.csv")
    d22 = aerpaw_availability[
        (aerpaw_availability["dataset"] == "dataset22_lte_semicircle")
        & (aerpaw_availability["model"] == "Logistic RF/KPI model")
    ].iloc[0]
    d23 = aerpaw_availability[
        (aerpaw_availability["dataset"] == "dataset23_lte_two_sweeps")
        & (aerpaw_availability["model"] == "Logistic RF/KPI model")
    ].iloc[0]
    throughput = aerpaw_throughput.loc[aerpaw_throughput["model"] == "RF/KPI random forest"].iloc[0]

    external = pd.read_csv(ROOT / "outputs/external_validation/external_metrics_summary.csv")
    external_ktg = external.loc[
        (external["Model"] == "Kinetic-TopoGuard") & (external["Radius_quantile"] == 0.5)
    ].iloc[0]

    audit_path = ROOT / "submission_readiness_audit.json"
    audit = _read_json(audit_path) if audit_path.is_file() else {"status": "not_run", "checks": []}

    return {
        "generated_from": {
            "paper_like_submission": "outputs/paper_like_submission",
            "publication_compact": "outputs/publication_compact",
            "publication_neural_5seed_extension": "outputs/publication_neural_5seed_extension",
            "operating_point": "outputs/operating_point",
            "uav_to_uav_mmwave": "outputs/uav_to_uav_mmwave_validation",
            "aerpaw_cellular": "outputs/aerpaw_cellular_validation",
            "external_forestry": "outputs/external_validation",
        },
        "experiment": {
            "confirmatory_seeds": int(paper_summary["n_seeds"]),
            "confirmatory_snapshots": int(paper_summary["total_snapshots"]),
            "neural_extension_seeds": int(neural_summary["n_seeds"]),
            "forecast_horizon_steps": int(paper_summary["forecast_horizon_steps"]),
            "torch_backend_verified": bool(compact_summary.get("torch_available") and not compact_summary.get("surrogate_used", True)),
        },
        "topology_forecaster": {
            "model": "Kinetic-TopoGuard",
            "mae": float(ktg["MAE_mean"]),
            "r2": float(ktg["R2_mean"]),
            "risk_pr_auc": float(ktg["Risk_PR_AUC_mean"]),
            "risk_roc_auc": float(ktg["Risk_ROC_AUC_mean"]),
            "inference_ms": float(ktg["Inference_ms_mean"]),
            "union_oracle_f1": float(union["Risk_F1_mean"]),
        },
        "operating_points": {
            "deployable": {
                "threshold": float(deployable["threshold"]),
                "f1": float(deployable["f1_mean"]),
                "precision": float(deployable["precision_mean"]),
                "recall": float(deployable["recall_mean"]),
                "false_alarms_per_minute": float(deployable["false_alarms_per_minute_mean"]),
                "relative_alarm_reduction_pct": float(deployable["relative_alarm_reduction_pct"]),
            },
            "strict": {
                "threshold": float(strict["threshold"]),
                "f1": float(strict["f1_mean"]),
                "precision": float(strict["precision_mean"]),
                "recall": float(strict["recall_mean"]),
                "false_alarms_per_minute": float(strict["false_alarms_per_minute_mean"]),
                "relative_alarm_reduction_pct": float(strict["relative_alarm_reduction_pct"]),
            },
        },
        "measured_validation": {
            "uav_to_uav_60ghz": {
                "heldout_test_rows": int(a2a_logistic["test_n"]),
                "baseline_f1": float(a2a_baseline["f1"]),
                "model_f1": float(a2a_logistic["f1"]),
                "model_pr_auc": float(a2a_logistic["pr_auc"]),
                "expected_beta0_snr7": float(
                    a2a_summary.loc[a2a_summary["snr_threshold_db"] == 7, "expected_beta0_mean"].iloc[0]
                ),
            },
            "aerpaw_cellular": {
                "d22_lte_f1": float(d22["f1"]),
                "d23_lte_f1": float(d23["f1"]),
                "iperf_mae_mbps": float(throughput["mae_mbps"]),
                "iperf_r2": float(throughput["r2"]),
            },
            "forestry_transfer": {
                "radius_median_m": float(external_ktg["Radius_m"]),
                "risk_f1": float(external_ktg["Risk_F1_mean"]),
                "mae": float(external_ktg["MAE_mean"]),
            },
        },
        "audit": {
            "status": audit.get("status", "not_run"),
            "pass_count": sum(1 for item in audit.get("checks", []) if item.get("status") == "pass"),
            "fail_count": sum(1 for item in audit.get("checks", []) if item.get("status") == "fail"),
        },
    }


def _card(title: str, value: str, label: str) -> str:
    return (
        "<section class=\"metric-card\">"
        f"<span>{html.escape(title)}</span>"
        f"<strong>{html.escape(value)}</strong>"
        f"<p>{html.escape(label)}</p>"
        "</section>"
    )


def _render_html(payload: dict, assets: dict[str, str]) -> str:
    operating_json = json.dumps(payload["operating_points"], ensure_ascii=False)
    p = payload
    deploy = p["operating_points"]["deployable"]
    strict = p["operating_points"]["strict"]
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FANET Topology Digital Twin Dashboard</title>
  <style>
    :root {{
      --ink: #172026;
      --muted: #5e6a72;
      --line: #d6dde2;
      --panel: #f5f7f8;
      --accent: #0f766e;
      --accent-2: #7c2d12;
      --bg: #ffffff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: var(--bg);
      font-family: Arial, Helvetica, sans-serif;
      line-height: 1.45;
    }}
    header {{
      padding: 28px 36px 18px;
      border-bottom: 1px solid var(--line);
      background: #eef3f4;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: clamp(24px, 3vw, 34px);
      letter-spacing: 0;
    }}
    h2 {{
      margin: 0 0 12px;
      font-size: 20px;
      letter-spacing: 0;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 24px;
    }}
    .subhead {{
      margin: 0;
      color: var(--muted);
      max-width: 920px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 24px;
    }}
    .metric-card, .panel {{
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
    }}
    .metric-card {{
      padding: 14px 16px;
      min-height: 126px;
    }}
    .metric-card span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .06em;
    }}
    .metric-card strong {{
      display: block;
      margin: 10px 0 6px;
      font-size: 26px;
      letter-spacing: 0;
    }}
    .metric-card p {{
      margin: 0;
      color: var(--muted);
      font-size: 13px;
    }}
    .panel {{
      padding: 18px;
      margin-bottom: 18px;
    }}
    .two-col {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
    }}
    .asset {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      display: block;
    }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 12px 0;
    }}
    button {{
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 4px;
      padding: 8px 10px;
      cursor: pointer;
      color: var(--ink);
    }}
    button.active {{
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
    }}
    .bar-track {{
      width: 100%;
      height: 12px;
      border-radius: 999px;
      background: #e7ecef;
      overflow: hidden;
      margin: 8px 0 16px;
    }}
    .bar {{
      height: 100%;
      width: 0;
      background: var(--accent-2);
      transition: width .2s ease;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      padding: 8px 6px;
      border-bottom: 1px solid var(--line);
      text-align: left;
    }}
    th {{ color: var(--muted); font-weight: 600; }}
    code {{
      background: #eef2f3;
      border-radius: 4px;
      padding: 2px 4px;
    }}
    footer {{
      color: var(--muted);
      font-size: 12px;
      padding: 8px 0 24px;
    }}
    @media (max-width: 880px) {{
      .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .two-col {{ grid-template-columns: 1fr; }}
      header {{ padding: 22px 20px 16px; }}
      main {{ padding: 18px; }}
    }}
    @media (max-width: 560px) {{
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>FANET Topology Digital Twin Dashboard</h1>
    <p class="subhead">Static build artifact generated from the executed simulation, measured AERPAW cellular traces, measured UAV-to-UAV 60 GHz channel data, external forestry trajectory transfer, and operating-point selection outputs.</p>
  </header>
  <main>
    <div class="grid">
      {_card("Confirmatory run", f"{p['experiment']['confirmatory_seeds']} seeds", f"{p['experiment']['confirmatory_snapshots']:,} snapshots")}
      {_card("Topology model", _metric(pd.Series(p['topology_forecaster']), "mae"), "Kinetic-TopoGuard mean absolute beta0 error")}
      {_card("Latency", f"{p['topology_forecaster']['inference_ms']:.2f} ms", "Mean per-snapshot inference")}
      {_card("Neural extension", f"{p['experiment']['neural_extension_seeds']} seeds", "Nine PyTorch neural models, no surrogate backend")}
    </div>

    <section class="panel">
      <h2>Operating Point</h2>
      <div class="controls">
        <button id="btn-deployable" class="active" data-policy="deployable">Deployable 25/min</button>
        <button id="btn-strict" data-policy="strict">Strict 10/min</button>
      </div>
      <p><strong>Risk threshold:</strong> <span id="threshold">{deploy['threshold']:.1f}</span></p>
      <p><strong>F1:</strong> <span id="f1">{deploy['f1']:.3f}</span> &nbsp; <strong>Precision:</strong> <span id="precision">{deploy['precision']:.3f}</span> &nbsp; <strong>Recall:</strong> <span id="recall">{deploy['recall']:.3f}</span></p>
      <p><strong>False alarms/min:</strong> <span id="false-alarms">{deploy['false_alarms_per_minute']:.1f}</span> &nbsp; <strong>Alarm reduction:</strong> <span id="reduction">{deploy['relative_alarm_reduction_pct']:.1f}%</span></p>
      <div class="bar-track" aria-label="Relative false alarm reduction"><div id="reduction-bar" class="bar"></div></div>
      <img class="asset" src="{assets['operating']}" alt="Operating-point selection plot">
    </section>

    <section class="two-col">
      <div class="panel">
        <h2>Measured RF Validation</h2>
        <table>
          <tr><th>Evidence</th><th>Metric</th><th>Value</th></tr>
          <tr><td>UAV-to-UAV 60 GHz hold-out</td><td>F1</td><td>{p['measured_validation']['uav_to_uav_60ghz']['model_f1']:.3f}</td></tr>
          <tr><td>UAV-to-UAV 60 GHz hold-out</td><td>PR-AUC</td><td>{p['measured_validation']['uav_to_uav_60ghz']['model_pr_auc']:.3f}</td></tr>
          <tr><td>AERPAW D22 LTE</td><td>F1</td><td>{p['measured_validation']['aerpaw_cellular']['d22_lte_f1']:.3f}</td></tr>
          <tr><td>AERPAW D23 LTE</td><td>F1</td><td>{p['measured_validation']['aerpaw_cellular']['d23_lte_f1']:.3f}</td></tr>
          <tr><td>AERPAW iPerf throughput</td><td>MAE</td><td>{p['measured_validation']['aerpaw_cellular']['iperf_mae_mbps']:.1f} Mbps</td></tr>
        </table>
      </div>
      <div class="panel">
        <h2>External Transfer</h2>
        <table>
          <tr><th>Check</th><th>Value</th></tr>
          <tr><td>Forestry median radius</td><td>{p['measured_validation']['forestry_transfer']['radius_median_m']:.2f} m</td></tr>
          <tr><td>Forestry beta0 MAE</td><td>{p['measured_validation']['forestry_transfer']['mae']:.3f}</td></tr>
          <tr><td>Forestry fragmentation F1</td><td>{p['measured_validation']['forestry_transfer']['risk_f1']:.3f}</td></tr>
          <tr><td>A2A-calibrated expected beta0 at 7 dB</td><td>{p['measured_validation']['uav_to_uav_60ghz']['expected_beta0_snr7']:.3f}</td></tr>
          <tr><td>Readiness audit</td><td>{html.escape(str(p['audit']['status']).upper())} ({p['audit']['pass_count']} pass, {p['audit']['fail_count']} fail)</td></tr>
        </table>
      </div>
    </section>

    <section class="two-col">
      <div class="panel">
        <h2>UAV-to-UAV Channel Integration</h2>
        <img class="asset" src="{assets['a2a']}" alt="UAV-to-UAV channel validation plot">
      </div>
      <div class="panel">
        <h2>AERPAW Cellular Validation</h2>
        <img class="asset" src="{assets['aerpaw']}" alt="AERPAW cellular validation plot">
      </div>
    </section>

    <footer>
      Data payload: <code>dashboard_payload.json</code>. Source directories: {html.escape(', '.join(p['generated_from'].values()))}.
    </footer>
  </main>
  <script>
    const operatingPoints = {operating_json};
    const fields = {{
      threshold: document.getElementById("threshold"),
      f1: document.getElementById("f1"),
      precision: document.getElementById("precision"),
      recall: document.getElementById("recall"),
      falseAlarms: document.getElementById("false-alarms"),
      reduction: document.getElementById("reduction"),
      bar: document.getElementById("reduction-bar")
    }};
    function renderPolicy(name) {{
      const point = operatingPoints[name];
      fields.threshold.textContent = point.threshold.toFixed(1);
      fields.f1.textContent = point.f1.toFixed(3);
      fields.precision.textContent = point.precision.toFixed(3);
      fields.recall.textContent = point.recall.toFixed(3);
      fields.falseAlarms.textContent = point.false_alarms_per_minute.toFixed(1);
      fields.reduction.textContent = point.relative_alarm_reduction_pct.toFixed(1) + "%";
      fields.bar.style.width = Math.max(0, Math.min(100, point.relative_alarm_reduction_pct)).toFixed(1) + "%";
      for (const button of document.querySelectorAll("button[data-policy]")) {{
        button.classList.toggle("active", button.dataset.policy === name);
      }}
    }}
    for (const button of document.querySelectorAll("button[data-policy]")) {{
      button.addEventListener("click", () => renderPolicy(button.dataset.policy));
    }}
    renderPolicy("deployable");
  </script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a static digital-twin dashboard from executed evidence outputs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = args.output_dir / "assets"
    payload = _load_payload()
    assets = {
        "operating": _copy_asset(ROOT / "paper/figures/generated/operating_point_selection.png", assets_dir),
        "a2a": _copy_asset(ROOT / "paper/figures/generated/uav_to_uav_mmwave_validation.png", assets_dir),
        "aerpaw": _copy_asset(ROOT / "paper/figures/generated/aerpaw_cellular_validation.png", assets_dir),
    }
    (args.output_dir / "dashboard_payload.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (args.output_dir / "index.html").write_text(_render_html(payload, assets), encoding="utf-8")
    print(f"Wrote {args.output_dir.relative_to(ROOT) / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
