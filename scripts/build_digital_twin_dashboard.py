from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "digital_twin_dashboard"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_asset(source: Path, assets_dir: Path) -> str | None:
    if not source.is_file():
        return None
    assets_dir.mkdir(parents=True, exist_ok=True)
    destination = assets_dir / source.name
    shutil.copy2(source, destination)
    return f"assets/{destination.name}"


def _policy_payload(row: pd.Series) -> dict:
    return {
        "validation_budget": float(row["Validation_False_Alert_Budget_per_minute_mean"]),
        "threshold": float(row["Selected_Threshold_mean"]),
        "f1": float(row["Test_Risk_F1_mean"]),
        "precision": float(row["Test_Risk_Precision_mean"]),
        "recall": float(row["Test_Risk_Recall_mean"]),
        "false_alert_events_per_minute": float(row["Test_False_Alert_Events_per_minute_mean"]),
    }


def _load_replay(sample_stride: int = 5) -> dict:
    trace = pd.read_csv(ROOT / "data" / "external_validation" / "derived" / "forestry_multidrone_trace.csv")
    series = pd.read_csv(
        ROOT / "outputs" / "uav_to_uav_mmwave_validation" / "uav_to_uav_forestry_beta0_timeseries.csv"
    )
    vehicles = sorted(trace["vehicle_id"].astype(str).unique())
    times = sorted(trace["timestamp_s"].astype(float).unique())[:: max(sample_stride, 1)]
    trace = trace[trace["timestamp_s"].isin(times)].copy()
    bounds = {
        "x_min": float(trace["x_m"].min()),
        "x_max": float(trace["x_m"].max()),
        "y_min": float(trace["y_m"].min()),
        "y_max": float(trace["y_m"].max()),
    }
    position_map = {}
    for timestamp, group in trace.groupby("timestamp_s"):
        position_map[float(timestamp)] = {
            str(row.vehicle_id): [float(row.x_m), float(row.y_m), float(row.z_m)]
            for row in group.itertuples()
        }
    threshold_payload = {}
    for threshold, group in series[series["timestamp_s"].isin(times)].groupby("snr_threshold_db"):
        frames = []
        for row in group.sort_values("timestamp_s").itertuples():
            links = []
            for idx in range(1, 4):
                left, right = str(getattr(row, f"pair_{idx}")).split("-")
                links.append(
                    {
                        "source": left,
                        "target": right,
                        "distance_m": float(getattr(row, f"pair_{idx}_distance_m")),
                        "probability": float(getattr(row, f"pair_{idx}_link_probability")),
                    }
                )
            frames.append(
                {
                    "time_s": float(row.timestamp_s),
                    "positions": position_map[float(row.timestamp_s)],
                    "links": links,
                    "expected_beta0": float(row.expected_beta0),
                    "fragmentation_probability": float(row.fragmentation_probability),
                }
            )
        threshold_payload[str(int(threshold))] = frames
    return {
        "scope": "Offline replay of a public measured-motion trace with transported 60 GHz peer-link sensitivity.",
        "sample_stride": sample_stride,
        "sample_period_s": 0.1 * sample_stride,
        "vehicles": vehicles,
        "bounds": bounds,
        "thresholds_db": sorted(int(float(value)) for value in threshold_payload),
        "frames": threshold_payload,
    }


def _load_payload() -> dict:
    paper_summary = _read_json(ROOT / "outputs" / "paper_like_submission" / "summary.json")
    compact_summary = _read_json(ROOT / "outputs" / "publication_compact" / "summary.json")
    neural_summary = _read_json(ROOT / "outputs" / "publication_neural_5seed_extension" / "summary.json")
    metrics = pd.read_csv(ROOT / "outputs" / "paper_like_submission" / "metrics_overall.csv")
    ktg = metrics.loc[metrics["Model"] == "Kinetic-TopoGuard"].iloc[0]
    persistence = metrics.loc[metrics["Model"] == "Current-state persistence baseline"].iloc[0]

    operating = pd.read_csv(ROOT / "outputs" / "operating_point" / "operating_point_summary.csv")
    deployable = operating.loc[operating["Policy"] == "deployable"].iloc[0]
    strict = operating.loc[operating["Policy"] == "strict"].iloc[0]

    a2a_metrics = pd.read_csv(
        ROOT / "outputs" / "uav_to_uav_mmwave_validation" / "uav_to_uav_link_model_metrics.csv"
    )
    a2a_logistic = a2a_metrics.loc[a2a_metrics["model"] == "logistic A2A RF model"].iloc[0]
    a2a_baseline = a2a_metrics.loc[a2a_metrics["model"] == "training-prior baseline"].iloc[0]

    availability = pd.read_csv(
        ROOT / "outputs" / "aerpaw_cellular_validation" / "aerpaw_lte_availability_metrics.csv"
    )
    throughput = pd.read_csv(
        ROOT / "outputs" / "aerpaw_cellular_validation" / "aerpaw_throughput_metrics.csv"
    )
    d22 = availability[
        (availability["dataset"] == "dataset22_lte_semicircle")
        & (availability["model"] == "Logistic RF/KPI model")
    ].iloc[0]
    d23 = availability[
        (availability["dataset"] == "dataset23_lte_two_sweeps")
        & (availability["model"] == "Logistic RF/KPI model")
    ].iloc[0]
    throughput_primary = throughput[
        (throughput["split"] == "final-30%-by-time")
        & (throughput["model"] == "RF/KPI random forest")
    ].iloc[0]
    throughput_baseline = throughput[
        (throughput["split"] == "final-30%-by-time")
        & (throughput["model"] == "Training-mean baseline")
    ].iloc[0]

    external = pd.read_csv(ROOT / "outputs" / "external_validation" / "external_metrics_summary.csv")
    external_ktg = external[
        (external["Model"] == "Kinetic-TopoGuard") & (external["Radius_quantile"] == 0.5)
    ].iloc[0]
    latency = pd.read_csv(ROOT / "outputs" / "end_to_end_latency" / "end_to_end_latency_summary.csv")
    latency_n30 = latency.loc[latency["n_nodes"] == 30].iloc[0]

    packet_path = ROOT / "outputs" / "packet_level_controller" / "packet_metrics_summary.csv"
    packet = pd.read_csv(packet_path) if packet_path.is_file() else pd.DataFrame()
    packet_ktg = packet.loc[packet["Model"] == "Kinetic-TopoGuard"].iloc[0] if not packet.empty else None

    audit_path = ROOT / "submission_readiness_audit.json"
    audit = _read_json(audit_path) if audit_path.is_file() else {"status": "not_run", "checks": []}
    return {
        "generated_from": {
            "confirmatory": "outputs/paper_like_submission",
            "compact": "outputs/publication_compact",
            "neural_extension": "outputs/publication_neural_5seed_extension",
            "operating_point": "outputs/operating_point",
            "uav_to_uav_rf": "outputs/uav_to_uav_mmwave_validation",
            "aerpaw": "outputs/aerpaw_cellular_validation",
            "forestry": "outputs/external_validation",
            "latency": "outputs/end_to_end_latency",
            "packet_level": "outputs/packet_level_controller",
        },
        "experiment": {
            "confirmatory_seeds": int(paper_summary["n_seeds"]),
            "confirmatory_snapshots": int(paper_summary["total_snapshots"]),
            "neural_extension_seeds": int(neural_summary["n_seeds"]),
            "forecast_horizon_steps": int(paper_summary["forecast_horizon_steps"]),
            "torch_backend_verified": bool(
                compact_summary.get("torch_available") and not compact_summary.get("surrogate_used", True)
            ),
        },
        "topology_forecaster": {
            "mae": float(ktg["MAE_mean"]),
            "risk_f1": float(ktg["Risk_F1_mean"]),
            "risk_pr_auc": float(ktg["Risk_PR_AUC_mean"]),
            "persistence_f1": float(persistence["Risk_F1_mean"]),
            "end_to_end_p95_ms_n30": float(latency_n30["p95_ms"]),
        },
        "operating_points": {
            "deployable": _policy_payload(deployable),
            "strict": _policy_payload(strict),
        },
        "measured_validation": {
            "uav_to_uav_60ghz": {
                "heldout_test_rows": int(a2a_logistic["test_n"]),
                "baseline_f1": float(a2a_baseline["f1"]),
                "model_f1": float(a2a_logistic["f1"]),
                "model_pr_auc": float(a2a_logistic["pr_auc"]),
            },
            "aerpaw_cellular": {
                "d22_lte_f1": float(d22["f1"]),
                "d23_lte_f1": float(d23["f1"]),
                "throughput_model_mae_mbps": float(throughput_primary["mae_mbps"]),
                "throughput_baseline_mae_mbps": float(throughput_baseline["mae_mbps"]),
                "throughput_r2": float(throughput_primary["r2"]),
                "throughput_split": str(throughput_primary["split"]),
            },
            "forestry_transfer": {
                "radius_median_m": float(external_ktg["Radius_m"]),
                "risk_f1": float(external_ktg["Risk_F1_mean"]),
                "mae": float(external_ktg["MAE_mean"]),
            },
        },
        "packet_level": (
            {
                "pdr": float(packet_ktg["PDR_mean"]),
                "mean_delay_ms": float(packet_ktg["Mean_Delay_ms_mean"]),
            }
            if packet_ktg is not None
            else None
        ),
        "replay": _load_replay(),
        "audit": {
            "status": audit.get("status", "not_run"),
            "pass_count": sum(item.get("status") == "pass" for item in audit.get("checks", [])),
            "fail_count": sum(item.get("status") == "fail" for item in audit.get("checks", [])),
        },
    }


def _metric_card(title: str, value: str, label: str) -> str:
    return (
        '<section class="metric-card">'
        f"<span>{html.escape(title)}</span><strong>{html.escape(value)}</strong>"
        f"<p>{html.escape(label)}</p></section>"
    )


def _asset_panel(title: str, source: str | None, alt: str) -> str:
    if source is None:
        return ""
    return f'<section class="panel"><h2>{html.escape(title)}</h2><img class="asset" src="{source}" alt="{html.escape(alt)}"></section>'


def _render_html(payload: dict, assets: dict[str, str | None]) -> str:
    p = payload
    replay_json = json.dumps(p["replay"], separators=(",", ":"))
    operating_json = json.dumps(p["operating_points"], separators=(",", ":"))
    packet_text = (
        f"{100.0 * p['packet_level']['pdr']:.2f}% PDR, {p['packet_level']['mean_delay_ms']:.2f} ms mean delay"
        if p["packet_level"]
        else "Not generated"
    )
    evidence_panels = "".join(
        [
            _asset_panel("Forecast-Horizon Sensitivity", assets["horizon"], "Forecast horizon sensitivity"),
            _asset_panel("Equal-Learner Feature Ablation", assets["ablation"], "Factorial feature ablation"),
            _asset_panel("Simplified Packet-Level Validation", assets["packet"], "Packet-level controller metrics"),
            _asset_panel("Measured RF Checks", assets["a2a"], "Measured UAV-to-UAV RF validation"),
        ]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Offline FANET Predictive-Twin Replay</title>
  <style>
    :root {{ --ink:#172026; --muted:#5e6a72; --line:#d5dde1; --panel:#f4f7f8; --teal:#0f766e; --red:#a23b2a; --blue:#2f6db3; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; color:var(--ink); background:#fff; font-family:Arial,Helvetica,sans-serif; line-height:1.4; }}
    header {{ padding:24px 30px 18px; border-bottom:1px solid var(--line); background:#eef3f4; }}
    h1 {{ margin:0 0 6px; font-size:30px; letter-spacing:0; }}
    h2 {{ margin:0 0 12px; font-size:19px; letter-spacing:0; }}
    p {{ margin:6px 0; }}
    .subhead {{ color:var(--muted); max-width:980px; }}
    main {{ max-width:1180px; margin:0 auto; padding:20px; }}
    .grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin-bottom:16px; }}
    .metric-card,.panel {{ border:1px solid var(--line); border-radius:6px; background:#fff; }}
    .metric-card {{ padding:13px 15px; min-height:118px; }}
    .metric-card span {{ display:block; color:var(--muted); font-size:12px; text-transform:uppercase; }}
    .metric-card strong {{ display:block; margin:9px 0 5px; font-size:24px; }}
    .metric-card p {{ color:var(--muted); font-size:13px; }}
    .panel {{ padding:16px; margin-bottom:14px; }}
    .replay-layout {{ display:grid; grid-template-columns:minmax(0,2fr) minmax(250px,1fr); gap:14px; }}
    .canvas-wrap {{ width:100%; aspect-ratio:16/9; min-height:320px; background:var(--panel); border:1px solid var(--line); }}
    canvas {{ width:100%; height:100%; display:block; }}
    .controls {{ display:flex; align-items:center; flex-wrap:wrap; gap:8px; margin:10px 0; }}
    button {{ width:38px; height:34px; border:1px solid var(--line); border-radius:4px; background:#fff; cursor:pointer; font-size:16px; }}
    button.active {{ color:#fff; background:var(--teal); border-color:var(--teal); }}
    input[type=range] {{ flex:1; min-width:180px; }}
    .readout {{ width:100%; border-collapse:collapse; font-size:14px; }}
    th,td {{ padding:7px 5px; text-align:left; border-bottom:1px solid var(--line); }}
    th {{ color:var(--muted); font-weight:600; }}
    .bar-track {{ height:10px; background:#e5eaed; margin-top:5px; }}
    .bar {{ height:100%; width:0; background:var(--red); }}
    .two-col {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
    .asset {{ display:block; width:100%; border:1px solid var(--line); background:var(--panel); }}
    .note {{ color:var(--muted); font-size:12px; }}
    footer {{ padding:4px 0 18px; color:var(--muted); font-size:12px; }}
    @media(max-width:860px) {{ .grid{{grid-template-columns:repeat(2,1fr)}} .replay-layout,.two-col{{grid-template-columns:1fr}} }}
    @media(max-width:520px) {{ .grid{{grid-template-columns:1fr}} header{{padding:20px}} main{{padding:14px}} h1{{font-size:25px}} }}
  </style>
</head>
<body>
  <header>
    <h1>Offline FANET Predictive-Twin Replay</h1>
    <p class="subhead">Evidence replay generated from executed experiments. It is not a live, bidirectional, or hardware-connected digital twin; no control command is sent to a physical UAV.</p>
  </header>
  <main>
    <div class="grid">
      {_metric_card("Confirmatory study", f"{p['experiment']['confirmatory_seeds']} seeds", f"{p['experiment']['confirmatory_snapshots']:,} simulated snapshots")}
      {_metric_card("Kinetic-TopoGuard", f"{p['topology_forecaster']['mae']:.3f} MAE", f"Risk F1 {p['topology_forecaster']['risk_f1']:.3f}")}
      {_metric_card("Full host path", f"{p['topology_forecaster']['end_to_end_p95_ms_n30']:.2f} ms", "N=30 P95: graph, PH and inference")}
      {_metric_card("Packet study", packet_text, "Simplified SimPy single-collision-domain model")}
    </div>

    <section class="panel">
      <h2>Measured-Motion Topology Replay</h2>
      <div class="controls">
        <button id="play" title="Play or pause" aria-label="Play or pause">&#9654;</button>
        <button class="snr" data-snr="5" title="5 dB SNR threshold">5</button>
        <button class="snr active" data-snr="7" title="7 dB SNR threshold">7</button>
        <button class="snr" data-snr="10" title="10 dB SNR threshold">10</button>
        <input id="time" type="range" min="0" max="1" value="0" step="1" aria-label="Replay time">
      </div>
      <div class="replay-layout">
        <div class="canvas-wrap"><canvas id="scene"></canvas></div>
        <div>
          <table class="readout">
            <tr><th>Time</th><td id="r-time">0.0 s</td></tr>
            <tr><th>SNR threshold</th><td id="r-snr">7 dB</td></tr>
            <tr><th>Expected beta0</th><td id="r-beta">0.000</td></tr>
            <tr><th>Fragmentation probability</th><td id="r-frag">0.000</td></tr>
          </table>
          <p class="note">Line opacity is the transported peer-link success probability. Positions come from the public forestry trajectory; RF sensitivity comes from a separate measured 60 GHz UAV-to-UAV dataset.</p>
          <div class="bar-track"><div id="frag-bar" class="bar"></div></div>
        </div>
      </div>
    </section>

    <section class="two-col">
      <div class="panel">
        <h2>Validation-Selected Operating Point</h2>
        <div class="controls">
          <button class="policy active" data-policy="deployable" title="Deployable policy">D</button>
          <button class="policy" data-policy="strict" title="Strict policy">S</button>
        </div>
        <table class="readout">
          <tr><th>Validation budget</th><td id="o-budget"></td></tr>
          <tr><th>Selected threshold</th><td id="o-threshold"></td></tr>
          <tr><th>Independent-test F1</th><td id="o-f1"></td></tr>
          <tr><th>Independent-test precision / recall</th><td id="o-pr"></td></tr>
          <tr><th>Independent-test false alert events/min</th><td id="o-false"></td></tr>
        </table>
      </div>
      <div class="panel">
        <h2>Measured and External Evidence</h2>
        <table class="readout">
          <tr><th>UAV-to-UAV 60 GHz F1</th><td>{p['measured_validation']['uav_to_uav_60ghz']['baseline_f1']:.3f} to {p['measured_validation']['uav_to_uav_60ghz']['model_f1']:.3f}</td></tr>
          <tr><th>AERPAW LTE F1</th><td>{p['measured_validation']['aerpaw_cellular']['d22_lte_f1']:.3f} / {p['measured_validation']['aerpaw_cellular']['d23_lte_f1']:.3f}</td></tr>
          <tr><th>AERPAW throughput MAE</th><td>{p['measured_validation']['aerpaw_cellular']['throughput_model_mae_mbps']:.1f} vs {p['measured_validation']['aerpaw_cellular']['throughput_baseline_mae_mbps']:.1f} Mbps baseline</td></tr>
          <tr><th>AERPAW throughput R2</th><td>{p['measured_validation']['aerpaw_cellular']['throughput_r2']:.3f} ({html.escape(p['measured_validation']['aerpaw_cellular']['throughput_split'])})</td></tr>
          <tr><th>Forestry transfer MAE / F1</th><td>{p['measured_validation']['forestry_transfer']['mae']:.3f} / {p['measured_validation']['forestry_transfer']['risk_f1']:.3f}</td></tr>
          <tr><th>Readiness audit</th><td>{html.escape(str(p['audit']['status']).upper())}: {p['audit']['pass_count']} pass, {p['audit']['fail_count']} fail</td></tr>
        </table>
      </div>
    </section>

    <div class="two-col">{evidence_panels}</div>
    <footer>Payload: dashboard_payload.json. This artifact separates measured inputs, simulated labels, transported RF sensitivity, and protocol-only boundaries.</footer>
  </main>
  <script>
    const replay = {replay_json};
    const operating = {operating_json};
    const canvas = document.getElementById("scene");
    const ctx = canvas.getContext("2d");
    const slider = document.getElementById("time");
    let snr = "7";
    let frameIndex = 0;
    let timer = null;
    function resize() {{
      const ratio = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width * ratio));
      canvas.height = Math.max(1, Math.floor(rect.height * ratio));
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      renderFrame();
    }}
    function point(position) {{
      const b = replay.bounds, pad = 34;
      const width = canvas.clientWidth - 2 * pad, height = canvas.clientHeight - 2 * pad;
      const x = pad + (position[0] - b.x_min) / Math.max(b.x_max - b.x_min, 1e-9) * width;
      const y = canvas.clientHeight - pad - (position[1] - b.y_min) / Math.max(b.y_max - b.y_min, 1e-9) * height;
      return [x, y];
    }}
    function renderFrame() {{
      const frames = replay.frames[snr];
      if (!frames || !frames.length) return;
      frameIndex = Math.max(0, Math.min(frameIndex, frames.length - 1));
      slider.max = String(frames.length - 1);
      slider.value = String(frameIndex);
      const frame = frames[frameIndex];
      ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
      ctx.fillStyle = "#f4f7f8";
      ctx.fillRect(0, 0, canvas.clientWidth, canvas.clientHeight);
      for (const link of frame.links) {{
        const a = point(frame.positions[link.source]), b = point(frame.positions[link.target]);
        ctx.strokeStyle = `rgba(15,118,110,${{Math.max(.08, link.probability)}})`;
        ctx.lineWidth = 1 + 5 * link.probability;
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
        const mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2;
        ctx.fillStyle = "#43515a"; ctx.font = "12px Arial";
        ctx.fillText(link.probability.toFixed(2), mx + 4, my - 4);
      }}
      replay.vehicles.forEach((vehicle, index) => {{
        const p = point(frame.positions[vehicle]);
        ctx.fillStyle = ["#2f6db3", "#a23b2a", "#3b7a57"][index % 3];
        ctx.beginPath(); ctx.arc(p[0], p[1], 9, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = "#172026"; ctx.font = "bold 13px Arial"; ctx.fillText(vehicle, p[0] + 12, p[1] + 4);
      }});
      document.getElementById("r-time").textContent = frame.time_s.toFixed(1) + " s";
      document.getElementById("r-snr").textContent = snr + " dB";
      document.getElementById("r-beta").textContent = frame.expected_beta0.toFixed(3);
      document.getElementById("r-frag").textContent = frame.fragmentation_probability.toFixed(3);
      document.getElementById("frag-bar").style.width = (100 * frame.fragmentation_probability).toFixed(1) + "%";
    }}
    function togglePlay() {{
      if (timer) {{
        clearInterval(timer); timer = null; document.getElementById("play").innerHTML = "&#9654;";
      }} else {{
        timer = setInterval(() => {{
          frameIndex = (frameIndex + 1) % replay.frames[snr].length; renderFrame();
        }}, 180);
        document.getElementById("play").innerHTML = "&#10074;&#10074;";
      }}
    }}
    function renderPolicy(name) {{
      const point = operating[name];
      document.getElementById("o-budget").textContent = point.validation_budget.toFixed(1) + " events/min";
      document.getElementById("o-threshold").textContent = point.threshold.toFixed(2);
      document.getElementById("o-f1").textContent = point.f1.toFixed(3);
      document.getElementById("o-pr").textContent = point.precision.toFixed(3) + " / " + point.recall.toFixed(3);
      document.getElementById("o-false").textContent = point.false_alert_events_per_minute.toFixed(2);
      document.querySelectorAll(".policy").forEach(button => button.classList.toggle("active", button.dataset.policy === name));
    }}
    document.getElementById("play").addEventListener("click", togglePlay);
    slider.addEventListener("input", () => {{ frameIndex = Number(slider.value); renderFrame(); }});
    document.querySelectorAll(".snr").forEach(button => button.addEventListener("click", () => {{
      snr = button.dataset.snr; frameIndex = 0;
      document.querySelectorAll(".snr").forEach(item => item.classList.toggle("active", item === button));
      renderFrame();
    }}));
    document.querySelectorAll(".policy").forEach(button => button.addEventListener("click", () => renderPolicy(button.dataset.policy)));
    window.addEventListener("resize", resize);
    renderPolicy("deployable");
    resize();
  </script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an offline predictive-twin evidence replay.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = args.output_dir / "assets"
    payload = _load_payload()
    assets = {
        "horizon": _copy_asset(ROOT / "outputs" / "horizon_sweep" / "horizon_sweep.png", assets_dir),
        "ablation": _copy_asset(
            ROOT / "outputs" / "factorial_feature_ablation" / "factorial_feature_ablation.png",
            assets_dir,
        ),
        "packet": _copy_asset(
            ROOT / "outputs" / "packet_level_controller" / "packet_level_controller.png",
            assets_dir,
        ),
        "a2a": _copy_asset(
            ROOT / "paper" / "figures" / "generated" / "uav_to_uav_mmwave_validation.png",
            assets_dir,
        ),
    }
    (args.output_dir / "dashboard_payload.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "index.html").write_text(_render_html(payload, assets), encoding="utf-8")
    print(f"Wrote {args.output_dir.relative_to(ROOT) / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
