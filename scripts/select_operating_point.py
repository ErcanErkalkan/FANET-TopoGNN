from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs" / "paper_like_submission" / "operating_point_metrics.csv"
DEFAULT_PER_SEED = ROOT / "outputs" / "paper_like_submission" / "per_seed_operating_point_metrics.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "operating_point"
PAPER_TABLE = ROOT / "paper" / "tables" / "generated" / "operating_point_table.tex"
PAPER_FIGURE_PDF = ROOT / "paper" / "figures" / "generated" / "operating_point_selection.pdf"
PAPER_FIGURE_PNG = ROOT / "paper" / "figures" / "generated" / "operating_point_selection.png"


def _write_table(summary: pd.DataFrame, output_path: Path) -> None:
    lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Policy & Validation budget & Feasible seeds & $\tau$ & Event precision & Event recall & Event F1 & False events/min \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"{str(row['Policy']).title()} & "
            f"{row['Validation_False_Alert_Budget_per_minute_mean']:.1f} & "
            f"{100.0 * row['Validation_Constraint_Met_mean']:.0f}\\% & "
            f"{row['Selected_Threshold_mean']:.2f} & "
            f"{row['Test_Alert_Event_Precision_mean']:.3f} & "
            f"{row['Test_Alert_Event_Recall_mean']:.3f} & "
            f"{row['Test_Alert_Event_F1_mean']:.3f} & "
            f"{row['Test_False_Alert_Events_per_minute_mean']:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(summary: pd.DataFrame, output_dir: Path) -> None:
    labels = summary["Policy"].str.title()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5))
    axes[0].bar(labels, summary["Test_Alert_Event_F1_mean"], color=["#2f6db3", "#3b7a57"])
    axes[0].set_ylabel("Independent-test event F1")
    axes[0].set_ylim(0, 1)
    axes[1].bar(
        labels,
        summary["Test_False_Alert_Events_per_minute_mean"],
        color=["#a23b2a", "#6f4e7c"],
    )
    axes[1].plot(
        labels,
        summary["Validation_False_Alert_Budget_per_minute_mean"],
        color="black",
        marker="o",
        linestyle="--",
        label="Validation budget",
    )
    axes[1].set_ylabel("False alert events per minute")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "operating_point_selection.png", dpi=220)
    fig.savefig(output_dir / "operating_point_selection.pdf")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarise validation-selected operating points on held-out test runs.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--per-seed", type=Path, default=DEFAULT_PER_SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if not args.per_seed.is_file():
        raise FileNotFoundError(args.per_seed)
    summary = pd.read_csv(args.input)
    per_seed = pd.read_csv(args.per_seed)
    summary = summary[
        (summary["Model"] == "Kinetic-TopoGuard")
        & summary["Policy"].isin(["deployable", "strict"])
    ].copy()
    if len(summary) != 2:
        raise RuntimeError("Expected deployable and strict Kinetic-TopoGuard operating-point rows")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "operating_point_summary.csv", index=False)
    per_seed.to_csv(args.output_dir / "operating_point_per_seed.csv", index=False)
    _write_table(summary, args.output_dir / "operating_point_table.tex")
    _plot(summary, args.output_dir)

    payload = {
        "source": str(args.input.relative_to(ROOT)),
        "per_seed_source": str(args.per_seed.relative_to(ROOT)),
        "selection_split": "validation",
        "evaluation_split": "test",
        "selection_rule": (
            "For each seed and policy, maximize validation event F1 among thresholds satisfying the "
            "validation false-alert-event budget; evaluate that fixed threshold once on disjoint test runs."
        ),
        "policies": summary.to_dict(orient="records"),
    }
    (args.output_dir / "operating_point_protocol.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    PAPER_TABLE.parent.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURE_PDF.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.output_dir / "operating_point_table.tex", PAPER_TABLE)
    shutil.copy2(args.output_dir / "operating_point_selection.pdf", PAPER_FIGURE_PDF)
    shutil.copy2(args.output_dir / "operating_point_selection.png", PAPER_FIGURE_PNG)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
