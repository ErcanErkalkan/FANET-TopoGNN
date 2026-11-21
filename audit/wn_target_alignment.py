"""Describe endpoint-label and onset differences on the existing R9.10 records."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

from verify_r9_10_strict import COOLDOWN, HORIZON, ROOT, THRESHOLD, rows, verify

KEYS = (
    "events", "warnable_events", "warnable_events_without_positive_endpoint",
    "matched_events", "matched_events_with_connected_endpoint", "false_alarms",
)
OUTPUT = ROOT / "data_derived/wn_r9_10_target_alignment.csv"


def assess_run(truth, scores):
    """truth maps frames to component counts, scores maps scored frames to the fixed score."""
    onsets = [t for t in sorted(truth) if truth.get(t - 1) == 1 and truth[t] > 1]
    eligible = {t for t in scores if truth[t] == 1}
    counts = Counter(dict.fromkeys(KEYS, 0))
    counts["events"] = len(onsets)
    for onset in onsets:
        prior = eligible.intersection(range(onset - HORIZON, onset))
        if prior:
            counts["warnable_events"] += 1
            if all(truth[t + HORIZON] == 1 for t in prior):
                counts["warnable_events_without_positive_endpoint"] += 1

    above = {t for t in eligible if scores[t] >= THRESHOLD}
    emitted = []
    for t in sorted(above):
        if t - 1 not in above and (not emitted or t - emitted[-1] >= COOLDOWN):
            emitted.append(t)
    unmatched = set(onsets)
    for t in emitted:
        match = next((u for u in range(t + 1, t + HORIZON + 1) if u in unmatched), None)
        if match is None:
            counts["false_alarms"] += 1
        else:
            unmatched.remove(match)
            counts["matched_events"] += 1
            counts["matched_events_with_connected_endpoint"] += int(truth[t + HORIZON] == 1)
    return counts


def audit_report(directory):
    verify(directory)
    truth, scores = defaultdict(dict), defaultdict(dict)
    for row in rows(directory / "evaluation_full_truth.csv"):
        truth[row["run_id"]][int(row["time_index"])] = int(float(row["beta_current"]))
    for row in rows(directory / "evaluation_scores.csv"):
        scores[row["run_id"]][int(row["time_index"])] = float(row["p_fragment"])

    pooled = Counter(dict.fromkeys(KEYS, 0))
    for row in rows(directory / "run_counts.csv"):
        run = row["run_id"]
        c = assess_run(truth[run], scores[run])
        for a, b in (("events", "events"), ("warnable_events", "warnable_events"),
                     ("matched_events", "true_alerts"), ("false_alarms", "false_alerts")):
            if c[a] != int(row[b]):
                raise ValueError(f"Target audit did not reproduce the verified {b} in {run}")
        pooled.update(c)
    return dict(pooled)


def reports():
    return [next((ROOT / "audit" / f"reference_{host}").iterdir()) for host in ("linux", "windows")
            ] + [ROOT / "evidence/pilot_linux"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-csv", action="store_true")
    args = parser.parse_args()
    result = [audit_report(path) for path in reports()]
    if any(x != result[0] for x in result[1:]):
        raise ValueError("Target alignment counts differ across verified records")
    if args.write_csv:
        with OUTPUT.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=KEYS, lineterminator="\n")
            writer.writeheader()
            writer.writerow(result[0])
    print(json.dumps({"verified_records": len(result), **result[0]}, indent=2))


if __name__ == "__main__":
    main()
