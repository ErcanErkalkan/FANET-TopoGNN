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
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Model & MAE [95\\% CI] & $R^2$ [95\\% CI] & Risk $F_1$ [95\\% CI] & PR-AUC & Brier \\\\",
        "\\midrule",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"{labels[row['Model']]} & {_interval(row, 'MAE')} & {_interval(row, 'R2')} & "
            f"{_interval(row, 'Risk_F1')} & {row['Risk_PR_AUC_mean']:.3f} & {row['Risk_Brier_mean']:.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    output = source_dir / "confirmatory_metrics_table.tex"
    output.write_text("\n".join(lines), encoding="utf-8")
    manuscript_output = ROOT / "paper/tables/generated/confirmatory_metrics_table.tex"
    manuscript_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output, manuscript_output)
    return output


def main() -> None:
    print(generate_confirmatory_table())


if __name__ == "__main__":
    main()
