from __future__ import annotations

import inspect
from pathlib import Path
import textwrap
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    }
)


MODEL_COLORS = {
    "Density + min-distance heuristics": "#7f7f7f",
    "Current-state persistence baseline": "#4d4d4d",
    "GCN": "#2f6db3",
    "GAT": "#8c564b",
    "GraphSAGE": "#3b7a57",
    "PI+MLP": "#9467bd",
    "Kinetic-TopoGuard": "#006d77",
    "FANET-TopoGNN": "#d62728",
    "FANET-TopoGNN (concat)": "#ff7f0e",
}

MODEL_SHORT_LABELS = {
    "Density + min-distance heuristics": "Heuristic",
    "Current-state persistence baseline": "Persistence",
    "Kinetic-TopoGuard": "Kinetic",
    "Shallow ML": "Shallow",
    "GraphSAGE": "GraphSAGE",
    "PI+MLP": "PI+MLP",
    "FANET-TopoGNN": "TopoGNN",
    "FANET-TopoGNN (concat)": "TopoGNN-concat",
    "STGCN (w=5)": "STGCN",
    "T-GCN (w=5)": "T-GCN",
    "TGN (w=5)": "TGN",
}


def _save_figure(fig, out_path: Path, dpi: int = 300) -> None:
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    if out_path.suffix.lower() != ".pdf":
        fig.savefig(out_path.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight", pad_inches=0.04)


def _wrap_label(label: object, width: int = 24) -> str:
    text = MODEL_SHORT_LABELS.get(str(label), str(label))
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False))


def _wrapped_labels(labels: list[object], width: int = 24) -> list[str]:
    return [_wrap_label(label, width=width) for label in labels]


def _barh_with_ci(ax, df: pd.DataFrame, metric: str, title: str, lower_is_better: bool = False, xlabel: str | None = None) -> None:
    mean_col = f"{metric}_mean"
    low_col = f"{metric}_ci95_low"
    high_col = f"{metric}_ci95_high"
    if mean_col not in df.columns:
        ax.set_axis_off()
        return
    ordered = df.sort_values(mean_col, ascending=lower_is_better)
    labels = ordered["Model"].tolist()
    means = ordered[mean_col].to_numpy(dtype=float)
    lows = ordered[low_col].to_numpy(dtype=float) if low_col in ordered else means
    highs = ordered[high_col].to_numpy(dtype=float) if high_col in ordered else means
    xerr = np.vstack([np.maximum(means - lows, 0.0), np.maximum(highs - means, 0.0)])
    y_pos = np.arange(len(labels))
    colors = [MODEL_COLORS.get(model, "#2f6db3") for model in labels]
    ax.barh(y_pos, means, xerr=xerr, capsize=3, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(y_pos, labels=_wrapped_labels(labels, width=23))
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(xlabel or metric)
    ax.grid(axis="x", alpha=0.25)


def _latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _to_simple_latex(df: pd.DataFrame, floatfmt: str = "%.3f") -> str:
    align = "l" + "r" * max(len(df.columns) - 1, 0)
    lines = [f"\\begin{{tabular}}{{{align}}}", "\\toprule"]
    header = " & ".join(_latex_escape(col) for col in df.columns) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")
    for _, row in df.iterrows():
        cells = []
        for value in row.tolist():
            if isinstance(value, (float, np.floating)):
                cells.append(floatfmt % value)
            else:
                cells.append(_latex_escape(value))
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def save_table(df: pd.DataFrame, out_dir: Path, stem: str, floatfmt: str = "%.3f") -> None:
    df.to_csv(out_dir / f"{stem}.csv", index=False)
    latex = _to_simple_latex(df, floatfmt=floatfmt)
    (out_dir / f"{stem}.tex").write_text(latex, encoding="utf-8")


def plot_lead_cdf(lead_map: dict[str, list[float]], out_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    if not lead_map:
        plt.xlabel("Topology-change lead time (ms)")
        plt.ylabel("Empirical CDF")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        _save_figure(plt.gcf(), out_path, dpi=200)
        plt.close()
        return
    for name, values in lead_map.items():
        arr = np.sort(np.asarray(values if values else [0.0], dtype=float))
        y = np.linspace(0, 1, len(arr))
        plt.plot(arr, y, linewidth=2, label=name)
    plt.xlabel("Topology-change lead time (ms)")
    plt.ylabel("Empirical CDF")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    _save_figure(plt.gcf(), out_path, dpi=200)
    plt.close()


def plot_metric_bars(df: pd.DataFrame, metric: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    ordered = df.sort_values(metric, ascending=(metric == "MAE"))
    labels = ordered["Model"].tolist()
    values = ordered[metric].to_numpy(dtype=float)
    colors = [MODEL_COLORS.get(model, "#2f6db3") for model in ordered["Model"]]
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(y_pos, labels=_wrapped_labels(labels, width=25))
    ax.invert_yaxis()
    ax.set_xlabel(metric)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, out_path, dpi=200)
    plt.close(fig)


def plot_latency(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    ordered = df.sort_values("Inference_ms", ascending=False)
    labels = ordered["Model"].tolist()
    values = ordered["Inference_ms"].to_numpy(dtype=float)
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color="#a33b20", edgecolor="black", linewidth=0.4)
    ax.axvline(20.0, linestyle="--", color="black", linewidth=1, label="20 ms budget")
    ax.set_yticks(y_pos, labels=_wrapped_labels(labels, width=25))
    ax.invert_yaxis()
    ax.set_xlabel("Inference latency (ms)")
    ax.set_xlim(0.0, max(21.0, float(np.nanmax(values)) * 1.15 if len(values) else 21.0))
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save_figure(fig, out_path, dpi=200)
    plt.close(fig)


def plot_network_metrics(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 8.6))
    metrics = ["Connectivity ratio", "PDR (%)", "Avg. end-to-end delay (ms)"]
    colors = ["#3b7a57", "#2f6db3", "#a33b20"]
    for ax, metric, color in zip(axes, metrics, colors):
        ordered = df.sort_values(metric, ascending=(metric == "Avg. end-to-end delay (ms)"))
        labels = ordered["Model"].tolist()
        values = ordered[metric].to_numpy(dtype=float)
        y_pos = np.arange(len(labels))
        ax.barh(y_pos, values, color=color, edgecolor="black", linewidth=0.4)
        ax.set_yticks(y_pos, labels=_wrapped_labels(labels, width=23))
        ax.invert_yaxis()
        ax.set_title(metric)
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, out_path, dpi=200)
    plt.close(fig)


def _bar_with_ci(ax, df: pd.DataFrame, metric: str, title: str, lower_is_better: bool = False) -> None:
    _barh_with_ci(ax, df, metric, title, lower_is_better=lower_is_better)


def plot_publication_performance(metrics_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 8.8))
    _bar_with_ci(axes[0], metrics_df, "MAE", "MAE with 95% CI", lower_is_better=True)
    _bar_with_ci(axes[1], metrics_df, "R2", "R2 with 95% CI", lower_is_better=False)
    _bar_with_ci(axes[2], metrics_df, "Risk_F1", "Risk F1 with 95% CI", lower_is_better=False)
    axes[0].set_xlabel("MAE (lower is better)")
    axes[1].set_xlabel("R2 (higher is better)")
    axes[2].set_xlabel("Risk F1 (higher is better)")
    fig.tight_layout()
    _save_figure(fig, out_path, dpi=300)
    plt.close(fig)


def plot_lead_time_summary(leads_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    _barh_with_ci(ax, leads_df, "Lead_median_ms", "Median topology-change lead time", lower_is_better=False, xlabel="Lead time (ms)")
    fig.tight_layout()
    _save_figure(fig, out_path, dpi=300)
    plt.close(fig)


def plot_dataset_overview(dataset_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    if dataset_df.empty:
        for ax in axes.ravel():
            ax.set_axis_off()
        fig.tight_layout()
        _save_figure(fig, out_path, dpi=300)
        plt.close(fig)
        return

    by_mobility = dataset_df.groupby("mobility", as_index=False)["snapshots"].sum()
    axes[0, 0].bar(by_mobility["mobility"], by_mobility["snapshots"], color="#2f6db3")
    axes[0, 0].set_title("Dataset composition by mobility")
    axes[0, 0].set_ylabel("Snapshots per seed")

    if {"graph_policy", "connected_ratio"}.issubset(dataset_df.columns):
        by_policy = dataset_df.groupby("graph_policy", as_index=False)["connected_ratio"].mean()
        axes[0, 1].bar(by_policy["graph_policy"], by_policy["connected_ratio"], color="#3b7a57")
        axes[0, 1].set_title("Connectivity ratio by graph policy")
        axes[0, 1].set_ylim(0, 1)
    else:
        axes[0, 1].set_axis_off()

    if {"n_nodes", "graph_policy", "edge_count_mean"}.issubset(dataset_df.columns):
        for policy, group in dataset_df.groupby("graph_policy"):
            series = group.groupby("n_nodes", as_index=False)["edge_count_mean"].mean()
            axes[1, 0].plot(series["n_nodes"], series["edge_count_mean"], marker="o", linewidth=2, label=str(policy))
        axes[1, 0].set_title("Mean edge count by swarm size")
        axes[1, 0].set_xlabel("Number of UAVs")
        axes[1, 0].set_ylabel("Mean edges")
        axes[1, 0].legend()
    else:
        axes[1, 0].set_axis_off()

    if {"radio_scenario", "graph_policy", "beta_current_mean"}.issubset(dataset_df.columns):
        for policy, group in dataset_df.groupby("graph_policy"):
            series = group.groupby("radio_scenario", as_index=False)["beta_current_mean"].mean()
            axes[1, 1].plot(series["radio_scenario"], series["beta_current_mean"], marker="o", linewidth=2, label=str(policy))
        axes[1, 1].set_title("Mean beta0 by radio scenario")
        axes[1, 1].set_ylabel("Mean beta0")
        axes[1, 1].tick_params(axis="x", rotation=30)
        axes[1, 1].legend()
    else:
        axes[1, 1].set_axis_off()

    for ax in axes.ravel():
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, out_path, dpi=300)
    plt.close(fig)


def plot_radio_policy_sensitivity(dataset_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    needed = {"radio_scenario", "graph_policy", "connected_ratio"}
    if dataset_df.empty or not needed.issubset(dataset_df.columns):
        ax.set_axis_off()
        fig.tight_layout()
        _save_figure(fig, out_path, dpi=300)
        plt.close(fig)
        return
    pivot = dataset_df.pivot_table(index="radio_scenario", columns="graph_policy", values="connected_ratio", aggfunc="mean")
    values = pivot.to_numpy(dtype=float)
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=0.0, vmax=1.0)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = float(values[i, j])
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1.0, 1.0, facecolor=cmap(norm(value)), edgecolor="white", linewidth=1.0))
    ax.set_xticks(np.arange(len(pivot.columns)), labels=pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
    ax.set_xlim(-0.5, len(pivot.columns) - 0.5)
    ax.set_ylim(len(pivot.index) - 0.5, -0.5)
    ax.set_title("Connectivity sensitivity across radio and graph policies")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.iloc[i, j]:.2f}", ha="center", va="center", color="white" if pivot.iloc[i, j] < 0.6 else "black")
    fig.tight_layout()
    _save_figure(fig, out_path, dpi=300)
    plt.close(fig)


def plot_network_extended_metrics(network_df: pd.DataFrame, out_path: Path) -> None:
    metrics = [
        ("Connectivity ratio_mean", "Connectivity ratio"),
        ("PDR (%)_mean", "PDR (%)"),
        ("Avg. end-to-end delay (ms)_mean", "Delay (ms)"),
        ("Proactive reroute (%)_mean", "Reroute (%)"),
        ("DTN buffered (%)_mean", "DTN buffered (%)"),
        ("Relay actions_mean", "Relay actions"),
    ]
    metrics = [(col, title) for col, title in metrics if col in network_df.columns]
    n_cols = 2
    n_rows = int(np.ceil(len(metrics) / n_cols)) if metrics else 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11.4, 4.6 * n_rows))
    axes_arr = np.asarray(axes).reshape(-1)
    for ax, (col, title) in zip(axes_arr, metrics):
        ordered = network_df.sort_values(col, ascending=(col == "Avg. end-to-end delay (ms)_mean"))
        labels = ordered["Model"].tolist()
        values = ordered[col].to_numpy(dtype=float)
        y_pos = np.arange(len(labels))
        colors = [MODEL_COLORS.get(model, "#2f6db3") for model in ordered["Model"]]
        ax.barh(y_pos, values, color=colors, edgecolor="black", linewidth=0.4)
        ax.set_yticks(y_pos, labels=_wrapped_labels(labels, width=22))
        ax.invert_yaxis()
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
    for ax in axes_arr[len(metrics) :]:
        ax.set_axis_off()
    fig.tight_layout()
    _save_figure(fig, out_path, dpi=300)
    plt.close(fig)


def plot_ablation_summary(metrics_df: pd.DataFrame, out_path: Path) -> None:
    models = ["PI+MLP", "GraphSAGE", "FANET-TopoGNN", "FANET-TopoGNN (concat)", "Kinetic-TopoGuard"]
    subset = metrics_df[metrics_df["Model"].isin(models)]
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 7.6))
    for ax, metric, title, lower in [
        (axes[0], "MAE", "Fusion ablation: MAE", True),
        (axes[1], "R2", "Fusion ablation: R2", False),
        (axes[2], "Risk_F1", "Fusion ablation: Risk F1", False),
    ]:
        if subset.empty:
            ax.set_axis_off()
            continue
        _bar_with_ci(ax, subset, metric, title, lower_is_better=lower)
    fig.tight_layout()
    _save_figure(fig, out_path, dpi=300)
    plt.close(fig)


def plot_statistical_summary(stats_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    if stats_df.empty:
        ax.set_axis_off()
        fig.tight_layout()
        _save_figure(fig, out_path, dpi=300)
        plt.close(fig)
        return
    rows = []
    for _, row in stats_df.iterrows():
        candidate = row["candidate"]
        for col, label in [
            ("mae_fdr_pvalue", "MAE"),
            ("lead_fdr_pvalue", "Lead"),
            ("risk_f1_fdr_pvalue", "Risk F1"),
        ]:
            if col in row and pd.notna(row[col]):
                rows.append({"candidate": candidate, "metric": label, "value": -np.log10(max(float(row[col]), 1e-300))})
    frame = pd.DataFrame(rows)
    if frame.empty:
        ax.set_axis_off()
    else:
        pivot = frame.pivot_table(index="candidate", columns="metric", values="value", aggfunc="mean").fillna(0.0)
        pivot.plot(kind="barh", ax=ax, color=["#2f6db3", "#3b7a57", "#d62728"])
        ax.axvline(-np.log10(0.05), color="black", linestyle="--", linewidth=1, label="FDR 0.05")
        ax.set_yticklabels(_wrapped_labels([tick.get_text() for tick in ax.get_yticklabels()], width=24))
        ax.invert_yaxis()
        ax.set_xlabel("-log10(FDR-adjusted p-value)")
        reference = str(stats_df["reference"].iloc[0]) if "reference" in stats_df.columns and not stats_df.empty else "reference"
        ax.set_title(f"Paired statistical evidence vs {reference}")
        ax.legend()
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, out_path, dpi=300)
    plt.close(fig)


def plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, s=10, alpha=0.5, color="#2f6db3")
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], linestyle="--", color="black")
    plt.xlabel("True beta0")
    plt.ylabel("Predicted beta0")
    plt.tight_layout()
    _save_figure(plt.gcf(), out_path, dpi=200)
    plt.close()


def plot_residual_box(residual_map: dict[str, np.ndarray], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    labels = list(residual_map)
    data = [residual_map[label] for label in labels]
    boxplot_params = inspect.signature(ax.boxplot).parameters
    boxplot_kwargs: dict[str, object] = {"showfliers": False}
    label_key = "tick_labels" if "tick_labels" in boxplot_params else "labels"
    boxplot_kwargs[label_key] = _wrapped_labels(labels, width=25)
    if "orientation" in boxplot_params:
        boxplot_kwargs["orientation"] = "horizontal"
    else:
        boxplot_kwargs["vert"] = False
    ax.boxplot(data, **boxplot_kwargs)
    ax.axvline(0.0, linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("Residual")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, out_path, dpi=200)
    plt.close(fig)


def write_markdown_report(
    out_path: Path,
    summary: dict,
    metrics_df: pd.DataFrame,
    leads_df: pd.DataFrame,
    network_df: pd.DataFrame,
    risk_df: pd.DataFrame,
) -> None:
    best_mae = metrics_df.iloc[0] if not metrics_df.empty else None
    best_lead = leads_df.iloc[0] if not leads_df.empty else None
    lines = [
        f"# {summary['experiment_name']} report",
        "",
        f"- Seeds: {summary['n_seeds']}",
        f"- Forecast horizon steps: {summary['forecast_horizon_steps']}",
        f"- Best model by MAE: {summary['best_model_by_mae']}",
        f"- Best model by lead: {summary['best_model_by_lead'] if summary['best_model_by_lead'] is not None else 'No positive lead observed in this run'}",
    ]
    if "runtime_seconds" in summary:
        lines.append(f"- Wall-clock runtime: {float(summary['runtime_seconds']):.1f} s")
    if "cuda_available" in summary and summary["cuda_available"] is not None:
        lines.append(f"- CUDA available during run: {str(bool(summary['cuda_available'])).lower()}")
    lines.extend(["", "## Accuracy"])
    if best_mae is not None:
        lines.extend(
            [
                f"- Best MAE model: {best_mae['Model']}",
                f"- MAE mean: {best_mae['MAE_mean']:.4f}",
                f"- R2 mean: {best_mae['R2_mean']:.4f}",
                "",
            ]
        )
    lines.append("## Early warning")
    if best_lead is not None and summary["best_model_by_lead"] is not None:
        lines.extend(
            [
                f"- Best median lead model: {summary['best_model_by_lead']}",
                f"- Median lead mean: {best_lead['Lead_median_ms_mean']:.2f} ms",
                "",
            ]
        )
    else:
        lines.extend(["- No positive lead time was observed in this run.", ""])
    lines.extend(
        [
            "## Risk detection",
            risk_df[["Model", "Risk_F1_mean", "Risk_Recall_mean", "Risk_Precision_mean"]].to_string(index=False),
            "",
            "## Network impact",
        ]
    )
    if network_df.empty:
        lines.extend(["- No controller models were selected for this run.", ""])
    else:
        lines.extend(
            [
                network_df[
                    [
                        col
                        for col in [
                            "Model",
                            "Connectivity ratio_mean",
                            "PDR (%)_mean",
                            "Avg. end-to-end delay (ms)_mean",
                            "Proactive reroute (%)_mean",
                            "DTN buffered (%)_mean",
                            "Relay actions_mean",
                        ]
                        if col in network_df.columns
                    ]
                ].to_string(index=False),
                "",
            ]
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _row_by_model(df: pd.DataFrame, model: str) -> pd.Series | None:
    subset = df[df["Model"] == model]
    if subset.empty:
        return None
    return subset.iloc[0]


def _best_models(df: pd.DataFrame, metric: str, higher_is_better: bool = True, tol: float = 1e-9) -> tuple[list[str], float] | None:
    if df.empty or metric not in df.columns:
        return None
    values = df[metric].to_numpy(dtype=float)
    best = float(np.max(values) if higher_is_better else np.min(values))
    mask = np.abs(values - best) <= tol
    return df.loc[mask, "Model"].astype(str).tolist(), best


def write_claims_summary(
    out_path: Path,
    summary: dict,
    metrics_df: pd.DataFrame,
    leads_df: pd.DataFrame,
    network_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    dataset_df: pd.DataFrame,
) -> None:
    best_mae = _best_models(metrics_df, "MAE_mean", higher_is_better=False)
    best_r2 = _best_models(metrics_df, "R2_mean", higher_is_better=True)
    best_risk = _best_models(risk_df, "Risk_F1_mean", higher_is_better=True)
    best_lead = _best_models(leads_df, "Lead_median_ms_mean", higher_is_better=True, tol=1e-6)
    best_network = _best_models(network_df, "Connectivity ratio_mean", higher_is_better=True, tol=1e-6)
    kinetic = _row_by_model(metrics_df, "Kinetic-TopoGuard")
    fanet = _row_by_model(metrics_df, "FANET-TopoGNN")
    concat = _row_by_model(metrics_df, "FANET-TopoGNN (concat)")
    total_snapshots = int(summary.get("total_snapshots", 0))
    policies = ", ".join(summary.get("graph_policies", [])) or "not recorded"
    radio_scenarios = ", ".join(summary.get("radio_scenarios", [])) or "not recorded"
    lines = [
        f"# Claims summary for {summary['experiment_name']}",
        "",
        "## Dataset basis",
        f"- Seeds: {summary['n_seeds']}",
        f"- Total labelled snapshots represented in tables: {total_snapshots:,}",
        f"- Graph policies: {policies}",
        f"- Radio scenarios: {radio_scenarios}",
        f"- Forecast horizon steps: {summary['forecast_horizon_steps']}",
    ]
    if "runtime_seconds" in summary:
        lines.append(f"- Recorded wall-clock runtime: {float(summary['runtime_seconds']):.1f} s.")
    if "cuda_available" in summary and summary["cuda_available"] is not None:
        lines.append(f"- CUDA available during recorded run: {str(bool(summary['cuda_available'])).lower()}.")
    lines.extend(["", "## Data-supported claims"])
    if best_mae is not None:
        models, value = best_mae
        lines.append(f"- Lowest MAE: {', '.join(models)} with MAE={value:.4f}.")
    if best_r2 is not None:
        models, value = best_r2
        lines.append(f"- Highest R2: {', '.join(models)} with R2={value:.4f}.")
    if best_risk is not None:
        models, value = best_risk
        lines.append(f"- Best fragmentation-risk F1: {', '.join(models)} with F1={value:.4f}.")
    if best_lead is not None and best_lead[1] > 0.0:
        models, value = best_lead
        lines.append(f"- Longest median topology-change lead: {', '.join(models)} with {value:.2f} ms.")
    if best_network is not None:
        models, value = best_network
        lines.append(
            f"- Best/tied controller connectivity ratio: {', '.join(models)} with {value:.4f}."
        )
    if kinetic is not None:
        kinetic_risk = _row_by_model(risk_df, "Kinetic-TopoGuard")
        kinetic_lead = _row_by_model(leads_df, "Kinetic-TopoGuard")
        lines.extend(
            [
                "",
                "## Kinetic-TopoGuard status",
                f"- MAE={float(kinetic['MAE_mean']):.4f}, R2={float(kinetic['R2_mean']):.4f}.",
            ]
        )
        if kinetic_risk is not None:
            lines.append(f"- Risk-F1={float(kinetic_risk['Risk_F1_mean']):.4f}.")
        if kinetic_lead is not None:
            lines.append(f"- Median topology-change lead={float(kinetic_lead['Lead_median_ms_mean']):.2f} ms.")
        if fanet is not None:
            lines.extend(
                [
                    f"- MAE delta vs FANET-TopoGNN (positive favours Kinetic-TopoGuard): {float(fanet['MAE_mean'] - kinetic['MAE_mean']):.4f} absolute.",
                    f"- R2 delta vs FANET-TopoGNN (positive favours Kinetic-TopoGuard): {float(kinetic['R2_mean'] - fanet['R2_mean']):.4f} absolute.",
                ]
            )
    if fanet is not None and concat is not None:
        mae_delta = float(concat["MAE_mean"] - fanet["MAE_mean"])
        r2_delta = float(fanet["R2_mean"] - concat["R2_mean"])
        f1_delta = float(fanet["Risk_F1_mean"] - concat["Risk_F1_mean"])
        lines.extend(
            [
                "",
                "## FANET-TopoGNN vs naive concat",
                f"- MAE delta vs concat (positive favours FANET-TopoGNN): {mae_delta:.4f} absolute.",
                f"- R2 delta vs concat (positive favours FANET-TopoGNN): {r2_delta:.4f} absolute.",
                f"- Risk-F1 delta vs concat (positive favours FANET-TopoGNN): {f1_delta:.4f} absolute.",
            ]
        )
        if mae_delta < 0.0 or r2_delta < 0.0 or f1_delta < 0.0:
            lines.append("- Interpretation: the concat ablation outperforms the adaptive gated variant on at least one reported metric in this run.")
    if not stats_df.empty:
        lines.extend(
            [
                "",
                "## Statistical testing",
                "- Paired tests and FDR-adjusted p-values are available in `stats_tests.csv` and `stats_tests.tex`.",
            ]
        )
    if not dataset_df.empty and {"graph_policy", "radio_scenario"}.issubset(dataset_df.columns):
        lines.extend(
            [
                "",
                "## Sensitivity evidence",
                "- Fixed/adaptive graph-policy and radio-scenario summaries are available in `dataset_summary.csv`.",
                "- Publication figures `dataset_overview.png` and `radio_policy_sensitivity.png` visualise this evidence.",
            ]
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_artifact_manifest(out_path: Path, root_dir: Path) -> None:
    lines = []
    for path in sorted(root_dir.rglob("*")):
        if path.is_file():
            lines.append(str(path.relative_to(root_dir)))
    out_path.write_text("\n".join(lines), encoding="utf-8")
