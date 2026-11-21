"""Fail-closed R8 local outcome audit. Never promotes a run to publication readiness.

Input: R7/R8 new isolated output, NOT historical R4 summary archive.
Can be run on a partial output (--allow-partial) for progress.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

# This script is installed in R8 tools/ within the workspace.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty table: {path}")
    return rows


def holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in p_values):
        raise ValueError("p-values must be finite probabilities")
    ordered = sorted(range(len(p_values)), key=lambda i: p_values[i])
    result = [0.0] * len(p_values)
    prior = 0.0
    for rank, index in enumerate(ordered):
        prior = min(1.0, max(prior, p_values[index] * (len(p_values) - rank)))
        result[index] = prior
    return result


def check_score_rows(path: Path, *, expected_horizon: int | None = None) -> tuple[set[tuple[str, int]], dict[tuple[str, int], tuple[str, ...]]]:
    rows = load_csv(path)
    required = {"run_id", "time_index", "future_time_index", "split_group_id", "beta_target", "frag_at_horizon", "risk_score", "selected_threshold"}
    if not required.issubset(rows[0]):
        raise ValueError(f"Missing score columns {required-set(rows[0])}: {path}")
    signatures = {}
    for row in rows:
        key = (row["run_id"], int(row["time_index"]))
        if key in signatures:
            raise ValueError(f"Duplicate time key {key} in {path}")
        target_index = int(row["future_time_index"])
        if target_index <= key[1] or (expected_horizon is not None and target_index != key[1] + expected_horizon):
            raise ValueError(f"Invalid forward target {key} in {path}")
        if row["frag_at_horizon"] not in {"0", "1"}:
            raise ValueError(f"Nonbinary risk target in {path}")
        if not math.isfinite(float(row["selected_threshold"])):
            raise ValueError(f"Nonfinite alarm threshold in {path}")
        if not (math.isfinite(float(row["risk_score"])) and math.isfinite(float(row["beta_target"]))):
            raise ValueError(f"Nonfinite model score/label in {path}")
        # A model may have scores outside [0,1] if it is an explicitly uncalibrated baseline.
        signatures[key] = (row["future_time_index"], row["split_group_id"], row["beta_target"], row["frag_at_horizon"])
    return set(signatures), signatures


def audit(root: Path, config: Path, *, allow_partial: bool = False) -> dict[str, Any]:
    root = root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from fanet.config import load_config
    from fanet.pipeline import _cache_stem, _config_signature, _seed_complete
    from fanet.r4_evidence import verify_evidence
    cfg = load_config(config)
    output = (root / cfg.raw["output_dir"]).resolve()
    if not output.is_relative_to(root):
        raise ValueError("Output path must remain inside the isolated R8 workspace")
    spec = json.loads((root / "configs/revision_neural_20seed_full.json").read_text(encoding="utf-8"))
    tasks = list(spec["selected_models"])
    expected_seeds = [int(x) for x in cfg.training["seed_list"]]
    if set(cfg.training["selected_models"]) != set(tasks) or len(tasks) != 14 or len(expected_seeds) != len(set(expected_seeds)):
        raise ValueError("Frozen model or seed inventory mismatch")
    neural = set(spec["neural_models_requiring_pytorch"])
    if len(neural) != 9:
        raise ValueError("Neural inventory must have nine models")
    if not all(cfg.training.get(x) for x in ("r4_export_raw_evidence", "r4_strict_validation_roles", "r7_full_truth_event_diagnostics", "r7_export_all_model_scores")):
        raise ValueError("Required R4/R7 evidence flags not enabled")
    signature = _config_signature(cfg)
    rows_by_seed = []
    completed_seeds = []
    errors = []
    for seed in expected_seeds:
        seed_dir = output / "per_seed" / f"seed_{seed}"
        if not seed_dir.exists():
            continue
        try:
            if not _seed_complete(seed_dir, cfg):
                raise ValueError("seed incomplete: metric/cache/artifact preconditions")
            metrics = load_csv(seed_dir / "metrics_overall.csv")
            full_truth = load_csv(seed_dir / "r7_full_truth_event_metrics.csv")
            if len(metrics) != 14 or len(full_truth) != 14:
                raise ValueError("expected fourteen per-model metric and full-truth rows")
            if len({r["Model"] for r in metrics}) != 14 or len({r["Model"] for r in full_truth}) != 14:
                raise ValueError("duplicate model names")
            full_events = {int(round(float(r["Ground_Truth_Fragmentation_Events"]))) for r in full_truth}
            if len(full_events) != 1:
                raise ValueError("different full truth event counts among models")
            score_inventory = None
            score_labels = None
            for task in tasks:
                path = seed_dir / "r7_scored_predictions" / f"{_cache_stem(task)}.csv"
                keys, labels = check_score_rows(path, expected_horizon=int(cfg.sim["forecast_horizon_steps"]))
                if score_inventory is None:
                    score_inventory, score_labels = keys, labels
                elif keys != score_inventory or labels != score_labels:
                    raise ValueError(f"different model time/label support for {task}")
                if task in neural and not verify_evidence(seed_dir, task, signature, require_role_isolation=True):
                    raise ValueError(f"invalid neural evidence: {task}")
            for r in full_truth:
                if int(r["Forecast_Scored_Rows"]) != len(score_inventory):
                    raise ValueError("forecast scored count differs from shared support")
            by_model = {r["Model"]: r for r in full_truth}
            if "Current-state ExtraTrees" not in by_model:
                raise ValueError("ExtraTrees reference missing")
            for name in by_model:
                e = float(by_model[name]["Alert_Event_F1"])
                if not 0.0 <= e <= 1.0:
                    raise ValueError(f"Event-F1 outside [0,1]: {name}")
                rows_by_seed.append({"seed": seed, "model": name, "event_f1": e,
                                     "truth_events": next(iter(full_events)), "n_test": len(score_inventory)})
            completed_seeds.append(seed)
        except Exception as exc:
            errors.append(f"seed {seed}: {type(exc).__name__}: {exc}")
    missing = [s for s in expected_seeds if s not in completed_seeds]
    is_complete = not missing and not errors and len(rows_by_seed) == len(expected_seeds) * 14
    summary: dict[str, Any] = {
        "schema": "fanet_r8_local_scientific_execution_audit_v1",
        "scope": "Isolated R8 outputs; NOT historical R4 validation",
        "status": "ENGINEERING_EXPECTED_SEEDS_PASS_PUBLICATION_HOLD" if is_complete else "INCOMPLETE_OR_INVALID",
        "scientific_submission": "HOLD",
        "output_dir": str(output.relative_to(root)),
        "expected_seeds": len(expected_seeds), "verified_seed_count": len(completed_seeds),
        "expected_models_per_seed": 14, "expected_neural_checkpoints": len(expected_seeds)*9,
        "verified_neural_checkpoints": len(completed_seeds)*9,
        "missing_or_invalid_seeds": missing,
        "errors": errors,
        "known_unresolved_scientific_gates": [
            "R2 h6 schema version lineage",
            "classical-vs-neural validation/calibration protocol parity",
            "packet and closed-loop controller physical replay",
            "model comparison paper evidence and independent replication",
            "bibliography, ethics, and journal formatting checks",
        ],
    }
    if is_complete:
        from scipy.stats import wilcoxon
        matrix: dict[str, dict[int, float]] = {}
        for row in rows_by_seed:
            matrix.setdefault(row["model"], {})[row["seed"]] = row["event_f1"]
        base = matrix["Current-state ExtraTrees"]
        comparisons = []
        for model, scored in sorted(matrix.items()):
            if model == "Current-state ExtraTrees":
                continue
            paired = [scored[seed]-base[seed] for seed in expected_seeds]
            if max(map(abs, paired)) < 1e-14:
                p = 1.0
            else:
                p = float(wilcoxon(paired, alternative="two-sided", zero_method="wilcox").pvalue)
            comparisons.append({"comparison": f"{model} - Current-state ExtraTrees", "n_seed": len(paired),
                                "mean_event_f1_delta": sum(paired)/len(paired), "p_raw": p})
        holms = holm_adjust([r["p_raw"] for r in comparisons])
        for row, h in zip(comparisons, holms):
            row["p_holm_exploratory"] = h
        summary["exploratory_seed_level_comparisons"] = comparisons
        summary["warning"] = "Not a confirmatory claim: prior selections and unequal classical/neural validation protocols remain."
    summary["verified_seed_inventory"] = completed_seeds
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/r7_full_20seed_isolated.json")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    result = audit(args.root, args.config, allow_partial=args.allow_partial)
    if args.report:
        if args.report.exists():
            raise FileExistsError(f"Will not overwrite existing audit: {args.report}")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("exploratory_seed_level_comparisons", "errors")}, ensure_ascii=False, indent=2))
    for error in result["errors"][:15]:
        print(error, file=sys.stderr)
    return 0 if result["status"].startswith("ENGINEERING") or args.allow_partial else 2


if __name__ == "__main__":
    raise SystemExit(main())
