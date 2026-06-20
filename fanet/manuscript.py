from __future__ import annotations

from pathlib import Path
import json
import pandas as pd


def _fmt(mean: float, low: float, high: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def _latex_cell(value: object) -> str:
    text = str(value)
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
    )


def _simple_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    align = "l" + "r" * max(len(df.columns) - 1, 0)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{adjustbox}{max width=\\textwidth}",
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
    ]
    lines.append(" & ".join(_latex_cell(col) for col in df.columns) + " \\\\")
    lines.append("\\midrule")
    for _, row in df.iterrows():
        lines.append(" & ".join(_latex_cell(value) for value in row.tolist()) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}", "\\end{table}", ""])
    return "\n".join(lines)


def export_manuscript_tables(
    out_path: Path,
    metrics_df: pd.DataFrame,
    leads_df: pd.DataFrame,
    network_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    summary: dict,
) -> None:
    perf = pd.DataFrame(
        {
            "Model": metrics_df["Model"],
            "MAE": [
                _fmt(m, lo, hi)
                for m, lo, hi in zip(metrics_df["MAE_mean"], metrics_df["MAE_ci95_low"], metrics_df["MAE_ci95_high"])
            ],
            "MSE": [
                _fmt(m, lo, hi)
                for m, lo, hi in zip(metrics_df["MSE_mean"], metrics_df["MSE_ci95_low"], metrics_df["MSE_ci95_high"])
            ],
            "R2": [
                _fmt(m, lo, hi)
                for m, lo, hi in zip(metrics_df["R2_mean"], metrics_df["R2_ci95_low"], metrics_df["R2_ci95_high"])
            ],
            "Latency (ms)": metrics_df["Inference_ms_mean"].map(lambda x: f"{x:.3f}"),
        }
    )
    lead = pd.DataFrame(
        {
            "Model": leads_df["Model"],
            "Topology-change lead median (ms)": leads_df["Lead_median_ms_mean"].map(lambda x: f"{x:.2f}"),
            "Topology-change lead mean (ms)": leads_df["Lead_mean_ms_mean"].map(lambda x: f"{x:.2f}"),
            "Topology-change lead 95th (ms)": leads_df["Lead_95th_ms_mean"].map(lambda x: f"{x:.2f}"),
        }
    )
    network = pd.DataFrame(
        {
            "Model": network_df["Model"],
            "Connectivity ratio": network_df["Connectivity ratio_mean"].map(lambda x: f"{x:.3f}"),
            "PDR (%)": network_df["PDR (%)_mean"].map(lambda x: f"{x:.2f}"),
            "Delay (ms)": network_df["Avg. end-to-end delay (ms)_mean"].map(lambda x: f"{x:.2f}"),
        }
    )
    if "Proactive reroute (%)_mean" in network_df.columns:
        network["Reroute (%)"] = network_df["Proactive reroute (%)_mean"].map(lambda x: f"{x:.2f}")
    if "DTN buffered (%)_mean" in network_df.columns:
        network["DTN (%)"] = network_df["DTN buffered (%)_mean"].map(lambda x: f"{x:.2f}")
    risk = pd.DataFrame(
        {
            "Model": risk_df["Model"],
            "Risk F1": risk_df["Risk_F1_mean"].map(lambda x: f"{x:.3f}"),
            "Risk Recall": risk_df["Risk_Recall_mean"].map(lambda x: f"{x:.3f}"),
            "Risk Precision": risk_df["Risk_Precision_mean"].map(lambda x: f"{x:.3f}"),
        }
    )
    blocks = [
        "% Auto-generated manuscript tables",
        f"% Experiment: {summary['experiment_name']}",
        "",
        _simple_latex_table(perf, "Overall prediction performance.", "tab:perf_overall_auto"),
        _simple_latex_table(lead, "Topology-change lead-time summary.", "tab:lead_time_auto"),
        _simple_latex_table(network, "Network-level performance under the connectivity-aware controller.", "tab:network_metrics_auto"),
        _simple_latex_table(risk, "Fragmentation-risk forecasting metrics.", "tab:risk_metrics_auto"),
    ]
    out_path.write_text("\n".join(blocks), encoding="utf-8")


def export_manuscript_summary(out_path: Path, summary: dict, metrics_df: pd.DataFrame, leads_df: pd.DataFrame) -> None:
    best_r2_row = metrics_df.sort_values("R2_mean", ascending=False).iloc[0] if not metrics_df.empty else None
    best_risk_row = metrics_df.sort_values("Risk_F1_mean", ascending=False).iloc[0] if not metrics_df.empty else None
    payload = {
        "experiment_name": summary["experiment_name"],
        "best_model_by_mae": summary["best_model_by_mae"],
        "best_model_by_lead": summary["best_model_by_lead"],
        "best_models_by_lead": summary.get("best_models_by_lead", []),
        "top_mae": metrics_df.iloc[0]["MAE_mean"] if not metrics_df.empty else None,
        "best_model_by_r2": best_r2_row["Model"] if best_r2_row is not None else None,
        "top_r2": best_r2_row["R2_mean"] if best_r2_row is not None else None,
        "best_model_by_risk_f1": best_risk_row["Model"] if best_risk_row is not None else None,
        "top_risk_f1": best_risk_row["Risk_F1_mean"] if best_risk_row is not None else None,
        "top_lead_median_ms": leads_df.iloc[0]["Lead_median_ms_mean"] if not leads_df.empty else None,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
