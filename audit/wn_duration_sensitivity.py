"""Re-evaluate frozen alarms under minimum observed fragmentation durations."""
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from statistics import median

from verify_r9_10_strict import COOLDOWN, HORIZON, ROOT, THRESHOLD, rows, verify
from wn_target_alignment import reports

MIN_FRAMES = (1, 2, 3)
DATA = ROOT / "data_derived"
DURATION_CSV = DATA / "wn_r9_10_duration_sensitivity.csv"
HISTOGRAM_CSV = DATA / "wn_r9_10_episode_lengths.csv"
# Publication output is kept outside the research repository.
DURATION_MARKER = "VERIFIED R9.10 BY MINIMUM EPISODE LENGTH"
HISTOGRAM_MARKER = "VERIFIED R9.10 EPISODE LENGTH DISTRIBUTION"


def episode_lengths(truth):
    """Each record is (onset frame, observed disconnected frames, right-censored).

    A run already disconnected at its initial frame has no observed onset for
    that initial episode. A trailing disconnected episode has a lower-bound
    length even if it already meets a given minimum.
    """
    times = sorted(truth)
    start = None
    episodes = []
    for t in times:
        if truth[t] > 1 and truth.get(t - 1) == 1:
            start = t
        elif truth[t] == 1 and start is not None:
            episodes.append((start, t - start, False))
            start = None
    if start is not None:
        episodes.append((start, times[-1] - start + 1, True))
    return episodes


def frozen_alarms(truth, scores):
    above = {t for t, score in scores.items() if truth[t] == 1 and score >= THRESHOLD}
    alarms = []
    for t in sorted(above):
        if t - 1 not in above and (not alarms or t - alarms[-1] >= COOLDOWN):
            alarms.append(t)
    return alarms


def assess(episodes, alarms, minimum):
    """Chronologically match confirmed onsets, then flag unresolvable alarms."""
    confirmed = [t for t, count, _ in episodes if count >= minimum]
    indeterminate = [t for t, count, censored in episodes if censored and count < minimum]
    available = confirmed.copy()
    leads = []
    unresolved = false_alarms = 0
    for alarm in alarms:
        idx = bisect_right(available, alarm)
        if idx < len(available) and available[idx] <= alarm + HORIZON:
            leads.append(available.pop(idx) - alarm)
        else:
            idx = bisect_left(indeterminate, alarm + 1)
            if idx < len(indeterminate) and indeterminate[idx] <= alarm + HORIZON:
                unresolved += 1
            else:
                false_alarms += 1
    return Counter(confirmed_events=len(confirmed), right_censored_unconfirmed_events=len(indeterminate),
                   confirmed_shorter_events=len(episodes) - len(confirmed) - len(indeterminate),
                   emitted_alarms=len(alarms), matched_events=len(leads),
                   unmatched_resolved_alarms=false_alarms, unresolved_alarms=unresolved), leads


def evaluate(directory):
    verified = verify(directory)
    truth, scores = defaultdict(dict), defaultdict(dict)
    for row in rows(directory / "evaluation_full_truth.csv"):
        truth[row["run_id"]][int(row["time_index"])] = int(float(row["beta_current"]))
    for row in rows(directory / "evaluation_scores.csv"):
        scores[row["run_id"]][int(row["time_index"])] = float(row["p_fragment"])

    histogram = Counter()
    totals = {k: Counter() for k in MIN_FRAMES}
    all_leads = defaultdict(list)
    for row in rows(directory / "run_counts.csv"):
        run = row["run_id"]
        episodes = episode_lengths(truth[run])
        alarms = frozen_alarms(truth[run], scores[run])
        histogram.update((n, censored) for _, n, censored in episodes)
        for minimum in MIN_FRAMES:
            result, lead = assess(episodes, alarms, minimum)
            totals[minimum].update(result)
            all_leads[minimum].extend(lead)
            if minimum == 1:
                for a, b in (("confirmed_events", "events"), ("matched_events", "true_alerts"),
                             ("unmatched_resolved_alarms", "false_alerts")):
                    if result[a] != int(row[b]):
                        raise ValueError(f"Baseline replay did not reproduce {b}: {run}")

    pooled = next(r for r in verified["strata"] if r["scope"] == "pooled")
    minutes = pooled["minutes"]
    if sum(histogram.values()) != pooled["events"]:
        raise ValueError("Episode distribution does not reconcile with the verified event total")
    distribution = [
        {"observed_disconnected_frames": n, "ended_before_record_end": histogram[n, False],
         "right_censored": histogram[n, True]}
        for n in sorted({n for n, _ in histogram})
    ]
    summary = []
    for minimum, x in totals.items():
        leads = all_leads[minimum]
        n, matched, fp, unresolved = (x[k] for k in ("confirmed_events", "matched_events",
                                                       "unmatched_resolved_alarms", "unresolved_alarms"))
        if (matched + fp + unresolved != x["emitted_alarms"] or
                n + x["right_censored_unconfirmed_events"] + x["confirmed_shorter_events"] != pooled["events"]):
            raise ValueError("Duration counts do not reconcile")
        summary.append({
            "min_observed_frames": minimum, **x, "missed_confirmed_events": n - matched,
            "simulated_minutes": minutes, "recall": matched / n if n else None,
            "precision_resolved_alarms": matched / (matched + fp) if matched + fp else None,
            "unmatched_resolved_alarms_per_minute": fp / minutes,
            "median_lead_seconds": round(median(leads) / 10, 2) if leads else None,
            "matched_at_most_two_frames_ahead": sum(v <= 2 for v in leads),
        })
    if summary[0]["median_lead_seconds"] != verified["median_lead_seconds"]:
        raise ValueError("Baseline lead time does not match the verified record")
    return summary, distribution


def consistent_records():
    results = [evaluate(path) for path in reports()]
    if any(r != results[0] for r in results[1:]):
        raise ValueError("Verified records disagree on the duration analysis")
    return results[0]


def format_duration(row):
    fields = (row["min_observed_frames"], row["confirmed_events"],
              row["right_censored_unconfirmed_events"], row["matched_events"],
              f'{100*row["recall"]:.1f}', row["unmatched_resolved_alarms"],
              row["unresolved_alarms"], f'{row["unmatched_resolved_alarms_per_minute"]:.2f}',
              f'{row["median_lead_seconds"]:.2f}')
    return " & ".join(map(str,fields)) + r" \\" + "\n"


def grouped_lengths(histogram):
    bins = [(1, 1, "1"), (2, 2, "2"), (3, 5, "3--5"), (6, 10, "6--10"), (11, float("inf"), "11 or more")]
    for lo, hi, label in bins:
        subset = [r for r in histogram if lo <= r["observed_disconnected_frames"] <= hi]
        ended = sum(x["ended_before_record_end"] for x in subset)
        censored = sum(x["right_censored"] for x in subset)
        yield label, ended, censored, ended + censored


def format_histogram(row):
    return " & ".join(map(str, row)) + r" \\" + "\n"


def export_csv(path, data):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(data)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--write-csv", action="store_true")
    args = p.parse_args()
    summary, histogram = consistent_records()
    if args.write_csv:
        export_csv(DURATION_CSV, summary)
        export_csv(HISTOGRAM_CSV, histogram)
    print(json.dumps({"verified_records": 3, "summary": summary,
                      "episode_length_bins": list(grouped_lengths(histogram))}, indent=2))


if __name__ == "__main__":
    main()
