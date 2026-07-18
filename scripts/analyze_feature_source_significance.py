from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fanet.provenance import relative_repo_path, sha256_file


DEFAULT_INPUT = (
    ROOT
    / "outputs"
    / "factorial_feature_ablation_20seed"
    / "factorial_ablation_per_seed.csv"
)
REFERENCE = "current-only"
PRIMARY_CANDIDATE = "graph+topology+kinematic"
EXPECTED_COMBINATIONS = {
    "current-only",
    "graph",
    "topology",
    "kinematic",
    "graph+topology",
    "graph+kinematic",
    "topology+kinematic",
    "graph+topology+kinematic",
}
METRICS = {
    "Alert_Event_F1": {"label": "Event F1", "higher_is_better": True, "role": "primary"},
    "Alert_Event_Precision": {"label": "Event precision", "higher_is_better": True, "role": "secondary"},
    "Alert_Event_Recall": {"label": "Event recall", "higher_is_better": True, "role": "secondary"},
    "False_Alert_Events_per_minute": {"label": "False alert events/min", "higher_is_better": False, "role": "secondary"},
    "MAE": {"label": "MAE", "higher_is_better": False, "role": "secondary"},
    "R2": {"label": "R2", "higher_is_better": True, "role": "secondary"},
    "Risk_Brier": {"label": "Brier score", "higher_is_better": False, "role": "secondary"},
    "Risk_ECE": {"label": "Expected calibration error", "higher_is_better": False, "role": "secondary"},
}


def _stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def holm_adjust(pvalues: list[float]) -> list[float]:
    clean = np.asarray([float(value) if np.isfinite(value) else 1.0 for value in pvalues])
    count = len(clean)
    if count == 0:
        return []
    order = np.argsort(clean, kind="mergesort")
    adjusted = np.empty(count, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        corrected = min(1.0, float(clean[index]) * (count - rank))
        running = max(running, corrected)
        adjusted[index] = running
    return adjusted.tolist()


def paired_permutation_pvalue(
    differences: np.ndarray,
    rounds: int = 100_000,
    seed: int = 20260714,
) -> float:
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0 or np.allclose(values, 0.0):
        return 1.0
    observed = abs(float(values.mean()))
    tolerance = 1e-15
    if values.size <= 16:
        assignments = np.arange(2**values.size, dtype=np.uint32)[:, None]
        bits = (assignments >> np.arange(values.size, dtype=np.uint32)) & 1
        signs = np.where(bits == 0, -1.0, 1.0)
        statistics = np.abs((signs * values).mean(axis=1))
        return float(np.mean(statistics >= observed - tolerance))
    rng = np.random.default_rng(seed)
    exceedances = 0
    remaining = max(int(rounds), 1)
    chunk_size = 10_000
    while remaining:
        current = min(chunk_size, remaining)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(current, values.size))
        statistics = np.abs((signs * values).mean(axis=1))
        exceedances += int(np.sum(statistics >= observed - tolerance))
        remaining -= current
    return float((exceedances + 1) / (max(int(rounds), 1) + 1))


def paired_bootstrap_ci(
    differences: np.ndarray,
    rounds: int = 20_000,
    seed: int = 20260714,
) -> tuple[float, float]:
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if np.allclose(values, values[0]):
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(max(int(rounds), 1), values.size))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def analyze_paired_arrays(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    higher_is_better: bool,
    bootstrap_rounds: int = 20_000,
    permutation_rounds: int = 100_000,
    random_seed: int = 20260714,
) -> dict[str, float | int | str | bool]:
    candidate_values = np.asarray(candidate, dtype=float)
    reference_values = np.asarray(reference, dtype=float)
    if candidate_values.shape != reference_values.shape:
        raise ValueError("candidate and reference must contain the same paired seeds")
    finite = np.isfinite(candidate_values) & np.isfinite(reference_values)
    if not bool(np.all(finite)):
        return {
            "analysis_status": "invalid_missing_metric_values",
            "n_pairs": int(candidate_values.size),
            "n_valid_pairs": int(finite.sum()),
        }
    if candidate_values.size < 3:
        return {
            "analysis_status": "insufficient_seed_pairs",
            "n_pairs": int(candidate_values.size),
            "n_valid_pairs": int(candidate_values.size),
        }

    raw_difference = candidate_values - reference_values
    direction = 1.0 if higher_is_better else -1.0
    benefit_difference = direction * raw_difference
    raw_ci_low, raw_ci_high = paired_bootstrap_ci(
        raw_difference, bootstrap_rounds, random_seed
    )
    benefit_ci_low, benefit_ci_high = paired_bootstrap_ci(
        benefit_difference, bootstrap_rounds, random_seed
    )
    if np.allclose(raw_difference, 0.0):
        wilcoxon_p = 1.0
        permutation_p = 1.0
        cohens_dz = 0.0
    else:
        try:
            wilcoxon_p = float(
                wilcoxon(raw_difference, zero_method="wilcox", alternative="two-sided").pvalue
            )
        except ValueError:
            wilcoxon_p = 1.0
        permutation_p = paired_permutation_pvalue(
            raw_difference, permutation_rounds, random_seed
        )
        standard_deviation = float(np.std(raw_difference, ddof=1))
        cohens_dz = (
            0.0
            if not np.isfinite(standard_deviation) or standard_deviation <= 1e-15
            else float(np.mean(raw_difference) / standard_deviation)
        )
    return {
        "analysis_status": "ok",
        "n_pairs": int(candidate_values.size),
        "n_valid_pairs": int(candidate_values.size),
        "paired_mean_difference": float(np.mean(raw_difference)),
        "paired_median_difference": float(np.median(raw_difference)),
        "bootstrap_ci95_low": raw_ci_low,
        "bootstrap_ci95_high": raw_ci_high,
        "benefit_mean_difference": float(np.mean(benefit_difference)),
        "benefit_bootstrap_ci95_low": benefit_ci_low,
        "benefit_bootstrap_ci95_high": benefit_ci_high,
        "wilcoxon_pvalue": wilcoxon_p,
        "paired_permutation_pvalue": permutation_p,
        "cohens_dz": cohens_dz,
        "benefit_cohens_dz": direction * cohens_dz,
        "all_differences_zero": bool(np.allclose(raw_difference, 0.0)),
    }


def classify_result(result: dict, adjusted_permutation_pvalue: float) -> dict[str, object]:
    supported_superiority = False
    supported_degradation = False
    if result.get("analysis_status") == "ok":
        mean = float(result["benefit_mean_difference"])
        low = float(result["benefit_bootstrap_ci95_low"])
        high = float(result["benefit_bootstrap_ci95_high"])
        significant = np.isfinite(adjusted_permutation_pvalue) and adjusted_permutation_pvalue < 0.05
        supported_superiority = bool(mean > 0.0 and low > 0.0 and significant)
        supported_degradation = bool(mean < 0.0 and high < 0.0 and significant)
    if supported_superiority:
        decision = "statistically_supported_superiority"
    elif supported_degradation:
        decision = "statistically_supported_degradation"
    else:
        decision = "no_supported_superiority"
    return {
        "decision": decision,
        "statistically_supported_superiority": supported_superiority,
        "no_supported_superiority": not supported_superiority and not supported_degradation,
        "statistically_supported_degradation": supported_degradation,
    }


def validate_pairing(frame: pd.DataFrame) -> list[int]:
    required = {"seed", "feature_sources", "Model", *METRICS}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"input is missing required columns: {missing_columns}")
    duplicates = frame.duplicated(["feature_sources", "seed"], keep=False)
    if duplicates.any():
        rows = frame.loc[duplicates, ["feature_sources", "seed"]].to_dict("records")
        raise ValueError(f"duplicate seed rows detected: {rows}")
    combinations = set(frame["feature_sources"].astype(str))
    if combinations != EXPECTED_COMBINATIONS:
        raise ValueError(
            f"expected all eight feature combinations; missing={sorted(EXPECTED_COMBINATIONS - combinations)}, "
            f"unexpected={sorted(combinations - EXPECTED_COMBINATIONS)}"
        )
    reference_seeds = set(
        frame.loc[frame["feature_sources"] == REFERENCE, "seed"].astype(int)
    )
    if len(reference_seeds) < 3:
        raise ValueError(f"at least three reference seed pairs are required, found {len(reference_seeds)}")
    for combination, group in frame.groupby("feature_sources"):
        seeds = set(group["seed"].astype(int))
        if seeds != reference_seeds:
            raise ValueError(
                f"seed pairing mismatch for {combination}: missing={sorted(reference_seeds - seeds)}, "
                f"unexpected={sorted(seeds - reference_seeds)}"
            )
    current_models = set(
        frame.loc[frame["feature_sources"] == REFERENCE, "Model"].astype(str)
    )
    if current_models != {"Current-state ExtraTrees"}:
        raise ValueError(f"unexpected current-only model name: {sorted(current_models)}")
    return sorted(reference_seeds)


def analyze_frame(
    frame: pd.DataFrame,
    *,
    bootstrap_rounds: int = 20_000,
    permutation_rounds: int = 100_000,
) -> pd.DataFrame:
    seeds = validate_pairing(frame)
    indexed = frame.set_index(["feature_sources", "seed"]).sort_index()
    rows: list[dict] = []
    for candidate in sorted(EXPECTED_COMBINATIONS):
        candidate_model = str(
            frame.loc[frame["feature_sources"] == candidate, "Model"].iloc[0]
        )
        for metric, metadata in METRICS.items():
            seed = _stable_seed(candidate, metric)
            result = analyze_paired_arrays(
                indexed.loc[(candidate, seeds), metric].to_numpy(dtype=float),
                indexed.loc[(REFERENCE, seeds), metric].to_numpy(dtype=float),
                higher_is_better=bool(metadata["higher_is_better"]),
                bootstrap_rounds=bootstrap_rounds,
                permutation_rounds=permutation_rounds,
                random_seed=seed,
            )
            rows.append(
                {
                    "candidate_feature_sources": candidate,
                    "candidate_model": candidate_model,
                    "reference_feature_sources": REFERENCE,
                    "reference_model": "Current-state ExtraTrees",
                    "metric": metric,
                    "metric_label": metadata["label"],
                    "metric_role": metadata["role"],
                    "higher_is_better": bool(metadata["higher_is_better"]),
                    "comparison_scope": (
                        "primary_candidate" if candidate == PRIMARY_CANDIDATE else "secondary_candidate"
                    ),
                    **result,
                }
            )
    results = pd.DataFrame(rows)
    results["wilcoxon_holm_adjusted_pvalue"] = holm_adjust(
        results["wilcoxon_pvalue"].fillna(1.0).tolist()
    )
    results["paired_permutation_holm_adjusted_pvalue"] = holm_adjust(
        results["paired_permutation_pvalue"].fillna(1.0).tolist()
    )
    decisions = [
        classify_result(row, float(row["paired_permutation_holm_adjusted_pvalue"]))
        for row in results.to_dict("records")
    ]
    for key in (
        "decision",
        "statistically_supported_superiority",
        "no_supported_superiority",
        "statistically_supported_degradation",
    ):
        results[key] = [item[key] for item in decisions]
    return results


def _latex_escape(value: object) -> str:
    return str(value).replace("_", r"\_").replace("%", r"\%")


def write_latex_table(results: pd.DataFrame, path: Path) -> None:
    main = results.loc[results["candidate_feature_sources"] == PRIMARY_CANDIDATE]
    lines = [
        r"\begin{tabular}{lrrrrl}",
        r"\toprule",
        r"Metric & Mean diff. & 95\% CI & Holm $p_{perm}$ & Cohen's $d_z$ & Decision \\",
        r"\midrule",
    ]
    for _, row in main.iterrows():
        interval = f"[{row['bootstrap_ci95_low']:.4f}, {row['bootstrap_ci95_high']:.4f}]"
        lines.append(
            f"{_latex_escape(row['metric_label'])} & {row['paired_mean_difference']:.4f} & "
            f"{interval} & {row['paired_permutation_holm_adjusted_pvalue']:.4g} & "
            f"{row['cohens_dz']:.3f} & {_latex_escape(row['decision'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_effect_plot(results: pd.DataFrame, path: Path) -> None:
    main = results.loc[results["candidate_feature_sources"] == PRIMARY_CANDIDATE].copy()
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.8))
    for axis, (_, row) in zip(axes.flat, main.iterrows()):
        mean = float(row["paired_mean_difference"])
        low = float(row["bootstrap_ci95_low"])
        high = float(row["bootstrap_ci95_high"])
        axis.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
        axis.errorbar(
            [mean],
            [0.0],
            xerr=[[mean - low], [high - mean]],
            fmt="o",
            color="#2f6db3",
            capsize=4,
        )
        axis.set_yticks([])
        axis.set_title(str(row["metric_label"]))
        axis.set_xlabel("Full minus current-only")
        axis.grid(axis="x", alpha=0.25)
    fig.suptitle("Paired seed-level effects with clustered-bootstrap 95% intervals")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _json_value(value: object) -> object:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(float(value)) else float(value)
    return value


def _records(frame: pd.DataFrame) -> list[dict]:
    return [
        {key: _json_value(value) for key, value in row.items()}
        for row in frame.to_dict("records")
    ]


def _git_state() -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"commit": None, "dirty": None, "error": str(exc)}


def write_interpretation(results: pd.DataFrame, path: Path, input_path: Path) -> None:
    main = results.loc[results["candidate_feature_sources"] == PRIMARY_CANDIDATE]
    lines = [
        "# Feature-source paired interpretation",
        "",
        f"Source: `{relative_repo_path(input_path, ROOT)}`.",
        "",
        "All tests pair the full feature model and Current-state ExtraTrees by simulation seed. "
        "The 20 seeds, rather than snapshots or sample rows, are the independent units.",
        "",
        "Raw differences are full model minus current-only. The primary inferential test is a "
        "two-sided paired sign-permutation test. Holm adjustment is applied across all 64 reported "
        "metric-by-candidate rows. Superiority requires a favorable direction, a seed-bootstrap "
        "95% interval excluding zero, and Holm-adjusted permutation p < 0.05.",
        "",
        "No equivalence claim is made because no equivalence margin was predeclared.",
        "",
        "## Full model versus current-only",
        "",
        "| Metric | Mean difference | 95% CI | Holm-adjusted permutation p | Decision |",
        "|---|---:|---:|---:|---|",
    ]
    for _, row in main.iterrows():
        lines.append(
            f"| {row['metric_label']} | {row['paired_mean_difference']:.6f} | "
            f"[{row['bootstrap_ci95_low']:.6f}, {row['bootstrap_ci95_high']:.6f}] | "
            f"{row['paired_permutation_holm_adjusted_pvalue']:.6g} | "
            f"`{row['decision']}` |"
        )
    lines.extend(
        [
            "",
            "`no_supported_superiority` is reported in prose as no supported difference: neither "
            "superiority nor degradation was supported under the predeclared rule. It does not "
            "mean equivalence.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Paired seed-level feature-source significance analysis.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--bootstrap-rounds", type=int, default=20_000)
    parser.add_argument("--permutation-rounds", type=int, default=100_000)
    parser.add_argument(
        "--docs-output",
        type=Path,
        default=ROOT / "docs" / "eaai_revision" / "FEATURE_SOURCE_INTERPRETATION.md",
    )
    args = parser.parse_args()
    if args.bootstrap_rounds < 1 or args.permutation_rounds < 1:
        raise ValueError("bootstrap and permutation rounds must be positive")
    input_path = args.input.resolve()
    output_dir = input_path.parent
    started = time.perf_counter()
    frame = pd.read_csv(input_path)
    seeds = validate_pairing(frame)
    results = analyze_frame(
        frame,
        bootstrap_rounds=args.bootstrap_rounds,
        permutation_rounds=args.permutation_rounds,
    )
    csv_path = output_dir / "paired_tests.csv"
    tex_path = output_dir / "paired_tests.tex"
    pdf_path = output_dir / "paired_effects.pdf"
    decision_path = output_dir / "feature_source_decision.json"
    results.to_csv(csv_path, index=False)
    write_latex_table(results, tex_path)
    save_effect_plot(results, pdf_path)
    write_interpretation(results, args.docs_output.resolve(), input_path)

    main_rows = results.loc[results["candidate_feature_sources"] == PRIMARY_CANDIDATE]
    payload = {
        "schema_version": 1,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": relative_repo_path(input_path, ROOT),
        "input_sha256": sha256_file(input_path),
        "seed_list": seeds,
        "seed_count": len(seeds),
        "pairing_unit": "simulation seed",
        "reference": REFERENCE,
        "primary_candidate": PRIMARY_CANDIDATE,
        "primary_metric": "Alert_Event_F1",
        "secondary_metrics": [metric for metric in METRICS if metric != "Alert_Event_F1"],
        "difference_definition": "candidate minus current-only",
        "direction_rule": "higher-is-better metrics retain the raw sign; lower-is-better metrics reverse it for decisions",
        "decision_rule": {
            "primary_test": "two-sided paired sign-permutation test",
            "multiple_testing": "Holm adjustment across all 64 metric-by-candidate rows",
            "alpha": 0.05,
            "superiority": "benefit mean > 0, benefit bootstrap CI excludes zero, Holm-adjusted permutation p < 0.05",
            "degradation": "benefit mean < 0, benefit bootstrap CI excludes zero, Holm-adjusted permutation p < 0.05",
            "equivalence": "not assessed; no equivalence margin was predeclared",
        },
        "bootstrap": {"rounds": args.bootstrap_rounds, "cluster": "simulation seed"},
        "permutation": {"rounds": args.permutation_rounds, "unit": "paired seed difference"},
        "main_comparison": _records(main_rows),
        "all_comparisons": _records(results),
        "runtime_seconds": float(time.perf_counter() - started),
        "package_versions": {
            "python": platform.python_version(),
            "numpy": importlib.metadata.version("numpy"),
            "pandas": importlib.metadata.version("pandas"),
            "scipy": importlib.metadata.version("scipy"),
            "matplotlib": importlib.metadata.version("matplotlib"),
        },
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
        },
        "git": _git_state(),
        "source_files": [
            {
                "relative_path": relative_repo_path(Path(__file__).resolve(), ROOT),
                "bytes": Path(__file__).stat().st_size,
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            {
                "relative_path": relative_repo_path(input_path, ROOT),
                "bytes": input_path.stat().st_size,
                "sha256": sha256_file(input_path),
            },
        ],
        "artifacts": {
            relative_repo_path(path, ROOT): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in (csv_path, tex_path, pdf_path, args.docs_output.resolve())
        },
    }
    decision_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(main_rows[["metric", "paired_mean_difference", "bootstrap_ci95_low", "bootstrap_ci95_high", "paired_permutation_holm_adjusted_pvalue", "decision"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
