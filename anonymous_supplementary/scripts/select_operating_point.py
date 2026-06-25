from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs/paper_like_submission/risk_threshold_sensitivity.csv"
DEFAULT_OUTPUT = ROOT / "outputs/operating_point"
PAPER_TABLE = ROOT / "paper/tables/generated/operating_point_table.tex"
PAPER_FIGURE_PDF = ROOT / "paper/figures/generated/operating_point_selection.pdf"
PAPER_FIGURE_PNG = ROOT / "paper/figures/generated/operating_point_selection.png"


def _fmt_ci(row: pd.Series, column: str, digits: int = 3) -> str:
    value = float(row[f"{column}_mean"])
    low = float(row[f"{column}_ci95_low"])
    high = float(row[f"{column}_ci95_high"])
    return f"{value:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def _select_threshold(rows: pd.DataFrame, alarm_budget: float) -> pd.Series:
    feasible = rows[rows["False_Alarms_per_minute_mean"] <= alarm_budget].copy()
    if feasible.empty:
        raise RuntimeError(f"No Kinetic-TopoGuard threshold satisfies <= {alarm_budget:g} false alarms/min")
    feasible = feasible.sort_values(
        ["Risk_F1_mean", "Risk_Recall_mean", "Risk_Precision_mean"],
        ascending=[False, False, False],
    )
    return feasible.iloc[0]


def _write_table(summary: pd.DataFrame, output_path: Path) -> None:
    display = summary[
        [
            "policy",
            "threshold",
            "precision_ci",
            "recall_ci",
            "f1_ci",
            "false_alarms_per_min_ci",
            "relative_alarm_reduction_pct",
        ]
    ].copy()
    display["threshold"] = display["threshold"].map(lambda value: f"{value:.1f}")
    display["relative_alarm_reduction_pct"] = display["relative_alarm_reduction_pct"].map(lambda value: f"{value:.1f}\\%")
    lines = [
        "\\begin{tabular}{llllllr}",
        "\\toprule",
        "Policy & $\\tau_{\\mathrm{frag}}$ & Precision & Recall & F1 & False alarms/min & Reduction \\\\",
        "\\midrule",
    ]
    for _, row in display.iterrows():
        lines.append(
            f"{row['policy']} & {row['threshold']} & {row['precision_ci']} & "
            f"{row['recall_ci']} & {row['f1_ci']} & {row['false_alarms_per_min_ci']} & "
            f"{row['relative_alarm_reduction_pct']} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(kinetic: pd.DataFrame, selected: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))

    axes[0].plot(kinetic["Threshold"], kinetic["Risk_F1_mean"], marker="o", label="F1")
    axes[0].plot(kinetic["Threshold"], kinetic["Risk_Precision_mean"], marker="s", label="Precision")
    axes[0].plot(kinetic["Threshold"], kinetic["Risk_Recall_mean"], marker="^", label="Recall")
    for _, row in selected.iterrows():
        axes[0].axvline(float(row["threshold"]), color="#444444", linestyle=":", linewidth=1.0)
    axes[0].set_xlabel("Risk threshold")
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0, 1.02)
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(
        kinetic["Threshold"],
        kinetic["False_Alarms_per_minute_mean"],
        marker="o",
        color="#a23b2a",
        label="False alarms/min",
    )
    axes[1].axhline(25, color="#555555", linestyle="--", linewidth=1.0, label="25/min budget")
    axes[1].axhline(10, color="#777777", linestyle="-.", linewidth=1.0, label="10/min budget")
    for _, row in selected.iterrows():
        axes[1].axvline(float(row["threshold"]), color="#444444", linestyle=":", linewidth=1.0)
    axes[1].set_xlabel("Risk threshold")
    axes[1].set_ylabel("False alarms per minute")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "operating_point_selection.png", dpi=220)
    fig.savefig(output_dir / "operating_point_selection.pdf")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Select deployable risk thresholds from the 20-seed sensitivity run.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--primary-budget", type=float, default=25.0)
    parser.add_argument("--strict-budget", type=float, default=10.0)
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    kinetic = df[df["Model"] == "Kinetic-TopoGuard"].sort_values("Threshold").copy()
    if kinetic.empty:
        raise RuntimeError("Kinetic-TopoGuard rows are missing from the threshold sensitivity file")

    default = kinetic.loc[(kinetic["Threshold"] - 0.5).abs().idxmin()]
    primary = _select_threshold(kinetic, args.primary_budget)
    strict = _select_threshold(kinetic, args.strict_budget)

    rows: list[dict] = []
    for policy, budget, row in [
        ("F1-max reference", float("inf"), default),
        (f"Deployable <= {args.primary_budget:g}/min", args.primary_budget, primary),
        (f"Strict <= {args.strict_budget:g}/min", args.strict_budget, strict),
    ]:
        rows.append(
            {
                "policy": policy,
                "false_alarm_budget_per_min": budget,
                "threshold": float(row["Threshold"]),
                "precision_mean": float(row["Risk_Precision_mean"]),
                "recall_mean": float(row["Risk_Recall_mean"]),
                "f1_mean": float(row["Risk_F1_mean"]),
                "false_alarms_per_minute_mean": float(row["False_Alarms_per_minute_mean"]),
                "precision_ci": _fmt_ci(row, "Risk_Precision"),
                "recall_ci": _fmt_ci(row, "Risk_Recall"),
                "f1_ci": _fmt_ci(row, "Risk_F1"),
                "false_alarms_per_min_ci": _fmt_ci(row, "False_Alarms_per_minute", digits=1),
                "relative_alarm_reduction_pct": 100.0
                * (float(default["False_Alarms_per_minute_mean"]) - float(row["False_Alarms_per_minute_mean"]))
                / float(default["False_Alarms_per_minute_mean"]),
            }
        )
    summary = pd.DataFrame(rows)

    kinetic.to_csv(args.output_dir / "kinetic_threshold_sensitivity.csv", index=False)
    summary.to_csv(args.output_dir / "operating_point_summary.csv", index=False)
    _write_table(summary, args.output_dir / "operating_point_table.tex")
    _plot(kinetic, summary[summary["policy"].str.startswith(("Deployable", "Strict"))], args.output_dir)

    payload = {
        "source": str(args.input.relative_to(ROOT)),
        "selection_rule": "highest mean F1 among Kinetic-TopoGuard thresholds satisfying the false-alarm budget",
        "primary_budget_false_alarms_per_minute": args.primary_budget,
        "strict_budget_false_alarms_per_minute": args.strict_budget,
        "selected": [
            {
                key: (None if isinstance(value, float) and not math.isfinite(value) else value)
                for key, value in row.items()
            }
            for row in summary.to_dict(orient="records")
        ],
    }
    (args.output_dir / "operating_point_protocol.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )

    PAPER_TABLE.parent.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURE_PDF.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.output_dir / "operating_point_table.tex", PAPER_TABLE)
    shutil.copy2(args.output_dir / "operating_point_selection.pdf", PAPER_FIGURE_PDF)
    shutil.copy2(args.output_dir / "operating_point_selection.png", PAPER_FIGURE_PNG)

    print(f"Wrote {args.output_dir.relative_to(ROOT)}")
    print(summary[["policy", "threshold", "f1_mean", "false_alarms_per_minute_mean", "relative_alarm_reduction_pct"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
