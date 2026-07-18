from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fanet.provenance import build_file_manifest, relative_repo_path


def paired_protocol_deltas(metrics: pd.DataFrame) -> list[dict]:
    required = {"Seed", "Sequence", "Model", "Protocol", "primary_fpp_threshold", "Alert_Event_F1", "False_Alert_Events_per_minute", "MAE", "Risk_Brier", "Risk_ECE"}
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"MILUV metrics missing columns: {sorted(missing)}")
    primary = metrics[metrics["primary_fpp_threshold"].astype(bool)].copy()
    keys = ["Seed", "Sequence", "Model"]
    zero = primary[primary.Protocol == "zero_shot_frozen"]
    if zero.duplicated(keys).any():
        raise ValueError("zero-shot metrics contain duplicate paired keys")
    rows = []
    for protocol in ("chronological_calibration", "few_shot_prediction_head"):
        adapted = primary[primary.Protocol == protocol]
        if adapted.duplicated(keys).any():
            raise ValueError(f"{protocol} metrics contain duplicate paired keys")
        joined = zero.merge(adapted, on=keys, suffixes=("_zero", "_adapted"), validate="one_to_one")
        if len(joined) != len(zero):
            raise ValueError(f"{protocol} does not cover every zero-shot paired key")
        for model, group in joined.groupby("Model", sort=True):
            deltas = {
                "event_f1": group.Alert_Event_F1_adapted - group.Alert_Event_F1_zero,
                "false_events_per_minute": group.False_Alert_Events_per_minute_zero - group.False_Alert_Events_per_minute_adapted,
                "count_mae": group.MAE_zero - group.MAE_adapted,
                "brier": group.Risk_Brier_zero - group.Risk_Brier_adapted,
                "ece": group.Risk_ECE_zero - group.Risk_ECE_adapted,
            }
            row = {"protocol": protocol, "model": model, "pair_count": int(len(group))}
            for name, values in deltas.items():
                array = np.asarray(values, dtype=float)
                row[f"{name}_improvement_mean"] = float(np.mean(array))
                row[f"{name}_improved_pairs"] = int(np.sum(array > 0.0))
                row[f"{name}_worsened_pairs"] = int(np.sum(array < 0.0))
                row[f"{name}_tied_pairs"] = int(np.sum(array == 0.0))
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Derive the MILUV adaptation decision only from generated metrics.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "miluv_adaptation")
    args = parser.parse_args()
    metrics_path = args.output_dir / "per_seed_or_scenario_metrics.csv"
    metrics = pd.read_csv(metrics_path)
    deltas = paired_protocol_deltas(metrics)
    event_supported = any(
        row["event_f1_improvement_mean"] > 0.0 and row["event_f1_improved_pairs"] == row["pair_count"]
        for row in deltas
    )
    decision = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision_source": build_file_manifest([metrics_path, Path(__file__).resolve(), ROOT / "fanet" / "miluv_adaptation.py"], ROOT),
        "existing_zero_shot_primary_evidence_preserved": True,
        "existing_zero_shot_primary_artifact": "outputs/miluv_validation/miluv_metrics_per_seed.csv",
        "paired_unit": ["seed", "sequence", "model"],
        "paired_descriptive_deltas": deltas,
        "adaptation_supported_for_event_detection": bool(event_supported),
        "event_detection_conclusion": (
            "event-F1 benefit observed consistently in paired rows"
            if event_supported
            else "no event-F1 benefit was observed; this is not an equivalence conclusion"
        ),
        "adaptation_study_role": "secondary domain-shift analysis; not deployment readiness evidence",
        "leave_one_scenario_out": {"status": "unavailable", "reason": "only one locally extracted and provenance-verified compatible three-robot sequence"},
        "test_used_for_model_calibration_threshold_or_staleness_selection": False,
        "claims_excluded": ["deployment-ready", "IP packet delivery", "outdoor FANET transfer", "equivalent"],
    }
    destination = args.output_dir / "adaptation_decision.json"
    destination.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(f"wrote {relative_repo_path(destination, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
