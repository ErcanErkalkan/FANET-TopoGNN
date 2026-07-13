from __future__ import annotations

from pathlib import Path
import shutil

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _interval(row: pd.Series, metric: str) -> str:
    return f"{row[f'{metric}_mean']:.3f} [{row[f'{metric}_ci95_low']:.3f}, {row[f'{metric}_ci95_high']:.3f}]"


def generate_confirmatory_table() -> Path:
    source_dir = ROOT / "outputs/paper_like_submission"
    frame = pd.read_csv(source_dir / "metrics_overall.csv")
    labels = {
        "Current-state persistence baseline": "Current-state persistence",
        "Kinetic-TopoGuard": "Kinetic-TopoGuard",
        "Shallow ML": "Shallow ML",
    }
    order = ["Kinetic-TopoGuard", "Shallow ML", "Current-state persistence baseline"]
    frame = frame.set_index("Model").loc[order].reset_index()
    lines = [
        "\\begin{tabular}{lrrrrrrrr}",
        "\\toprule",
        "Model & MAE [95\\% CI] & $R^2$ [95\\% CI] & Sample $F_1$ & Event precision & Event recall & Event $F_1$ & False events/min & PR-AUC \\\\",
        "\\midrule",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"{labels[row['Model']]} & {_interval(row, 'MAE')} & {_interval(row, 'R2')} & "
            f"{row['Risk_F1_mean']:.3f} & {row['Alert_Event_Precision_mean']:.3f} & "
            f"{row['Alert_Event_Recall_mean']:.3f} & {row['Alert_Event_F1_mean']:.3f} & "
            f"{row['False_Alert_Events_per_minute_mean']:.2f} & {row['Risk_PR_AUC_mean']:.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    output = source_dir / "confirmatory_metrics_table.tex"
    output.write_text("\n".join(lines), encoding="utf-8")
    manuscript_output = ROOT / "paper/tables/generated/confirmatory_metrics_table.tex"
    manuscript_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output, manuscript_output)
    return output


def _write_generated(name: str, lines: list[str]) -> Path:
    output = ROOT / "paper" / "tables" / "generated" / name
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def generate_horizon_table() -> Path:
    frame = pd.read_csv(ROOT / "outputs/horizon_sweep/horizon_sweep_summary.csv")
    frame = frame[frame["Model"] == "Kinetic-TopoGuard"].sort_values("horizon_steps")
    lines = [
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"Horizon (s) & MAE & Sample $F_1$ & Event recall & Event $F_1$ \\",
        r"\midrule",
    ]
    for row in frame.itertuples():
        lines.append(
            f"{row.horizon_s:.1f} & {row.MAE_mean:.3f} & {row.Risk_F1_mean:.3f} & "
            f"{row.Alert_Event_Recall_mean:.3f} & {row.Alert_Event_F1_mean:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return _write_generated("horizon_sweep_table.tex", lines)


def generate_factorial_table() -> Path:
    frame = pd.read_csv(
        ROOT / "outputs/factorial_feature_ablation/factorial_ablation_summary.csv"
    ).sort_values("feature_sources")
    labels = {
        "current-only": "Current state only",
        "graph": "Graph",
        "topology": "Topology",
        "kinematic": "Kinematic",
        "graph+topology": "Graph + topology",
        "graph+kinematic": "Graph + kinematic",
        "topology+kinematic": "Topology + kinematic",
        "graph+topology+kinematic": "All sources",
    }
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Feature sources & MAE & Sample $F_1$ & Event $F_1$ \\",
        r"\midrule",
    ]
    for row in frame.itertuples():
        lines.append(
            f"{labels[row.feature_sources]} & {row.MAE_mean:.3f} & "
            f"{row.Risk_F1_mean:.3f} & {row.Alert_Event_F1_mean:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return _write_generated("factorial_ablation_table.tex", lines)


def generate_packet_table() -> Path:
    frame = pd.read_csv(
        ROOT / "outputs/packet_level_controller/packet_metrics_summary.csv"
    ).sort_values(["Packets_per_tick", "Model"])
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Model & Load & PDR & Mean delay & P95 delay & Deadline drop \\",
        r"\midrule",
    ]
    for row in frame.itertuples():
        model = str(row.Model).replace("Current-state persistence baseline", "Current persistence")
        lines.append(
            f"{model} & {row.Packets_per_tick:d} & {row.PDR_mean:.3f} & "
            f"{row.Mean_Delay_ms_mean:.1f} & {row.P95_Delay_ms_mean:.1f} & "
            f"{row.Contention_Deadline_Drop_Rate_mean:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return _write_generated("packet_level_table.tex", lines)


def generate_latency_table() -> Path:
    frame = pd.read_csv(ROOT / "outputs/end_to_end_latency/latency_summary.csv")
    lines = [
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"UAVs & Samples & Mean & P50 & P95 & P99 (ms) \\",
        r"\midrule",
    ]
    for row in frame.itertuples():
        lines.append(
            f"{row.n_nodes:d} & {row.samples:d} & {row.total_mean_ms:.2f} & "
            f"{row.total_p50_ms:.2f} & {row.total_p95_ms:.2f} & {row.total_p99_ms:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return _write_generated("latency_table.tex", lines)


def copy_miluv_table() -> Path:
    source = ROOT / "outputs/miluv_validation/miluv_validation_table.tex"
    output = ROOT / "paper/tables/generated/miluv_validation_table.tex"
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    return output


def copy_external_table() -> Path:
    source = ROOT / "outputs/external_validation/external_metrics_table.tex"
    output = ROOT / "paper/tables/generated/external_metrics_table.tex"
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    return output


def copy_protocol_figures() -> Path:
    sources = {
        "horizon_sweep.pdf": ROOT / "outputs/horizon_sweep/horizon_sweep.pdf",
        "factorial_feature_ablation.pdf": ROOT
        / "outputs/factorial_feature_ablation/factorial_feature_ablation.pdf",
        "packet_level_controller.pdf": ROOT
        / "outputs/packet_level_controller/packet_level_controller.pdf",
        "end_to_end_latency.pdf": ROOT / "outputs/end_to_end_latency/end_to_end_latency.pdf",
        "miluv_validation.pdf": ROOT / "outputs/miluv_validation/miluv_validation.pdf",
    }
    output_dir = ROOT / "paper/figures/generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        shutil.copy2(source, output_dir / name)
    return output_dir


def main() -> None:
    generators = [
        generate_confirmatory_table,
        generate_horizon_table,
        generate_factorial_table,
        generate_packet_table,
        generate_latency_table,
        copy_miluv_table,
        copy_external_table,
        copy_protocol_figures,
    ]
    for generator in generators:
        print(generator())


if __name__ == "__main__":
    main()
