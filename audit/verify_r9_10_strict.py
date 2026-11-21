"""Independently verify the frozen R9.10 development protocol and exported outcomes."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "frozen/plan/R9_5_TRAIN_VALIDATION_MATRIX.csv"
PROTOCOL = ROOT / "audit/R9_10_PROTOCOL.json"
REPORT = "R9_10_FIXED_THRESHOLD_AUDIT.json"
IDENTITY = ("replicate", "split_group_id", "mobility", "n_nodes", "graph_policy", "radio_scenario")
FILES = {"evaluation_scores.csv", "evaluation_full_truth.csv", "run_counts.csv", "budget_summary.csv"}
ROLES = {"training", "model_selection_or_early_stopping", "calibration_fit", "threshold_selection"}
MOBILITIES = ("rwp", "gm", "mission")
BLOCKS = (2, 3, 4, 5, 6)
STEPS, HORIZON, COOLDOWN, THRESHOLD, BUDGET = 200, 6, 6, 0.15, 2.0
POSSIBLE_DECISIONS = set(range(4, STEPS - HORIZON))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path):
    return json.loads(path.read_bytes())


def rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        require(len(header) == len(set(header)) and bool(header), f"Invalid CSV header: {path.name}")
        result = list(reader)
        require(all(None not in row and None not in row.values() for row in result),
                f"Malformed CSV row: {path.name}")
        return result


def unique(rows_, key, context):
    result = {key(row): row for row in rows_}
    require(len(result) == len(rows_), f"Duplicate key in {context}")
    return result


def plan_for_audit():
    plan = unique(rows(PLAN), lambda r: r["run_id"], "frozen development plan")
    require(len(plan) == 2880 and {row["role"] for row in plan.values()} == ROLES,
            "Unexpected roles or rows in frozen development plan")
    for field in ("run_seed", "split_group_id"):
        roles_by_key = defaultdict(set)
        for row in plan.values():
            roles_by_key[row[field]].add(row["role"])
        require(all(len(v) == 1 for v in roles_by_key.values()), f"Cross-role leakage: {field}")
    groups_by_key = defaultdict(set)
    for row in plan.values():
        groups_by_key[row["run_seed"]].add(row["split_group_id"])
    require(all(len(g) == 1 for g in groups_by_key.values()), "A seed identifies several trajectory groups")
    evaluation = {k:v for k,v in plan.items() if v["role"] == "threshold_selection"
                  and int(v["replicate"]) in BLOCKS}
    groups = defaultdict(list)
    for record in evaluation.values():
        groups[record["split_group_id"]].append(record)
        require(int(record["simulator_steps"]) == STEPS and int(record["horizon_steps"]) == HORIZON
                and float(record["dt"]) == .1, "Inconsistent timing in development plan")
    # Use the configured node counts directly, and audit the same complete cell set in every block.
    nodes = {r["n_nodes"] for r in evaluation.values()}
    require(len(nodes) == 3, "Expected three planned node counts")
    expected_cases = {(m,n,radio,graph) for m in MOBILITIES for n in nodes
                      for radio in ("nominal","degraded") for graph in ("fixed","adaptive")}
    for block in BLOCKS:
        part = [r for r in evaluation.values() if int(r["replicate"]) == block]
        cases = {(r["mobility"],r["n_nodes"],r["radio_scenario"],r["graph_policy"]) for r in part}
        require(len(part) == 36 and cases == expected_cases, f"Invalid scenario grid in block {block}")
    require(len(evaluation) == 180 and len(groups) == 45, "Incorrect evaluation grouping")
    for group in groups.values():
        require(len(group) == 4 and len({(r["replicate"],r["mobility"],r["n_nodes"],r["run_seed"])
                                         for r in group}) == 1 and
                len({(r["radio_scenario"], r["graph_policy"]) for r in group}) == 4,
                "A trajectory must contain four paired variants")
    return evaluation


def event_matches(truth, probability):
    """Chronological one-to-one matching. Missing eligible frames reset the trigger edge."""
    times = sorted(truth)
    event_times = {t for t in times if t-1 in truth and truth[t-1] <= 1 < truth[t]}
    connected = {t for t in probability if truth[t] <= 1}
    warnable = sum(any(t in connected for t in range(event-HORIZON, event))
                   for event in event_times)
    above = {t for t in connected if probability[t] >= THRESHOLD}
    emitted = []
    for t in sorted(above):
        if t-1 not in above and (not emitted or t-emitted[-1] >= COOLDOWN):
            emitted.append(t)
    unmatched = event_times.copy()
    leads = []
    for al in emitted:
        candidates = unmatched.intersection(range(al+1, al+HORIZON+1))
        if candidates:
            event = min(candidates)
            unmatched.remove(event)
            leads.append(event-al)
    return {
        "events": len(event_times), "warnable_events": warnable, "true_alerts": len(leads),
        "false_alerts": len(emitted)-len(leads), "missed_events": len(unmatched),
        "observed_frames": len(truth), "decision_rows": len(probability),
        "connected_decision_frames": len(connected), "lead_steps": leads,
    }


def metrics(scope, block, mobility, selection):
    sums = {key: sum(run[key] for run in selection) for key in
            ("events", "warnable_events", "true_alerts", "false_alerts", "observed_frames")}
    events, tp, fp = (sums[k] for k in ("events", "true_alerts", "false_alerts"))
    minutes = sums.pop("observed_frames") / 600
    return {
        "scope": scope, "replicate": block, "mobility": mobility, "runs": len(selection),
        "trajectory_groups": len({r["split_group_id"] for r in selection}),
        "minutes": minutes, **sums, "false_alerts_per_minute": fp / minutes,
        "within_budget": fp <= BUDGET * minutes, "event_recall": tp/events if events else None,
        "event_precision": tp/(tp+fp) if tp+fp else None,
        "event_f1": 2*tp/(tp+fp+events) if tp+fp+events else None,
    }


def compare_summary(observed, expected, label):
    require(set(observed) == set(expected), f"Unexpected summary columns: {label}")
    for key, value in expected.items():
        actual = observed[key]
        if isinstance(value, float):
            ok = isinstance(actual,(float,int)) and not isinstance(actual,bool) and math.isclose(
                actual,value,rel_tol=1e-12,abs_tol=1e-14)
        elif value is None or isinstance(value,bool):
            ok = actual is value
        else:
            ok = type(actual) is type(value) and actual == value
        require(ok, f"Independent summary mismatch: {label}/{key}")


def check_report(report):
    require(report["status"] == "R9_10_DEVELOPMENT_TRANSFER_COMPLETE_PUBLICATION_HOLD"
            and report["scientific_submission"] == "HOLD", "Invalid development publication status")
    for key in ("blind_holdout_read", "external_freeze_complete", "threshold_search", "calibration_refit"):
        require(report[key] is False, f"Invalid protocol flag: {key}")
    require(report["model_reconstructed_from_replicate_1"] is True, "Missing fixed operating point")
    require(report["evaluation_replicates"] == list(BLOCKS) and report["replayed_evaluation_runs"] == 180
            and report["threshold"] == THRESHOLD and report["false_alert_budget_per_minute"] == BUDGET,
            "Fixed R9.10 evaluation specification changed")
    host = report["platform"].split("-",1)[0]
    require(host in {"Linux","Windows"}, "Unsupported evidence platform")
    anchor_path = ROOT / "frozen/anchor" / host / "R9_9_CALIBRATION_ALERT_PILOT.json"
    anchor = load_json(anchor_path)
    require(report["calibrator"] == anchor["models"]["fanet_topognn"]["calibrator"]
            and report["versions"] == anchor["versions"], "Same-platform frozen calibration or versions changed")
    require(math.isfinite(report["anchor_regression_max_abs_delta"])
            and 0 <= report["anchor_regression_max_abs_delta"] <= 1e-6, "Anchor tolerance failed")
    require(all(anchor["models"]["fanet_topognn"]["selection_diagnostic_only"].get(k) == v
                for k,v in report["anchor_event_metrics"].items())
            and len(report["anchor_event_metrics"]) == 9, "R9.9 anchor metrics changed")
    expected_paths = {
        "development_csv_sha256": PLAN, "protocol_sha256": PROTOCOL, "r99_report_sha256":anchor_path,
        "runner_sha256": ROOT / "audit/r9_10_fixed_threshold_audit.py",
        "manifest_sha256": PLAN.with_name("R9_5_PLAN_MANIFEST.json"),
        "r96_code_sha256": ROOT / "frozen/r9_6_cost_pilot.py",
        "r96_windows_report_sha256": ROOT / "frozen/R9_6_COST_PILOT.json",
        "r99_runner_sha256": ROOT / "frozen/r9_9_calibration_alert.py",
    }
    for key,path in expected_paths.items():
        require(report[key] == sha(path), f"Input provenance mismatch: {key}")
    for key in ("simulator_sha256","model_source_sha256"):
        require(report[key] == anchor[key], f"Anchored source inventory changed: {key}")
        for name,digest in report[key].items():
            require(digest == sha(ROOT/"core"/name), f"Frozen model or simulator hash mismatch: {name}")
    return host


def verify(directory):
    directory = Path(directory)
    report = load_json(directory / REPORT)
    platform = check_report(report)
    require(set(report["output_sha256"]) == FILES, "Unexpected report output inventory")
    for name,digest in report["output_sha256"].items():
        require(sha(directory/name) == digest, f"Output SHA mismatch: {name}")

    planned = plan_for_audit()
    counts = unique(rows(directory/"run_counts.csv"), lambda r:r["run_id"], "run counts")
    require(counts.keys() == planned.keys(), "Run IDs do not match the locked evaluation plan")
    for key, entry in counts.items():
        for name in IDENTITY:
            require(entry[name] == planned[key][name], f"Plan identity mismatch: {key}/{name}")

    truth, probability = defaultdict(dict), defaultdict(dict)
    for line in rows(directory/"evaluation_full_truth.csv"):
        key,t = line["run_id"],int(line["time_index"])
        require(key in planned and line["replicate"] == planned[key]["replicate"]
                and 0 <= t < STEPS and t not in truth[key], f"Truth row does not match plan: {key}/{t}")
        beta = float(line["beta_current"])
        require(beta.is_integer() and beta >= 1, f"Invalid component count: {key}/{t}")
        truth[key][t] = beta
    require(truth.keys()==planned.keys() and all(set(t)==set(range(STEPS)) for t in truth.values()),
            "Incomplete full trajectories")
    slope,intercept = (report["calibrator"][key] for key in ("slope","intercept"))
    for line in rows(directory/"evaluation_scores.csv"):
        key,t = line["run_id"],int(line["time_index"])
        require(key in planned and t in POSSIBLE_DECISIONS and t not in probability[key],
                f"Invalid or duplicate decision row: {key}/{t}")
        p,pred = float(line["p_fragment"]),float(line["predicted_beta"])
        z = slope*pred + intercept
        require(math.isfinite(z) and math.isfinite(p) and 0<=p<=1, "Invalid probability or prediction")
        encoded = 1/(1+math.exp(-z)) if z>=0 else math.exp(z)/(1+math.exp(z))
        require(math.isclose(p,encoded,abs_tol=1e-14,rel_tol=0)
                and line["replicate"]==planned[key]["replicate"]
                and line["split_group_id"]==planned[key]["split_group_id"]
                and int(line["future_time_index"])==t+HORIZON
                and float(line["beta_current"])==truth[key][t]
                and float(line["beta_target"])==truth[key][t+HORIZON]
                and int(line["frag_at_horizon"])==int(truth[key][t+HORIZON]>1),
                f"Decision target or calibrated probability mismatch: {key}/{t}")
        probability[key][t] = p
    require(probability.keys()==planned.keys() and all(set(t)==POSSIBLE_DECISIONS for t in probability.values()),
            "Incomplete decision coverage")

    runs = {}
    for key,p in probability.items():
        run = event_matches(truth[key],p)
        for name in ("events","warnable_events","true_alerts","false_alerts","observed_frames","decision_rows"):
            require(int(counts[key][name])==run[name], f"Event replay mismatch: {key}/{name}")
        runs[key] = run | {"split_group_id":planned[key]["split_group_id"]}
    keys = [("pooled",None,None)]
    keys += [("pooled_mobility",None,m) for m in MOBILITIES]
    for b in BLOCKS:
        keys += [("replicate",b,None)] + [("replicate_mobility",b,m) for m in MOBILITIES]
    derived = {}
    for scope,b,m in keys:
        selected = [run for k,run in runs.items() if
                    (b is None or b==int(planned[k]["replicate"])) and
                    (m is None or m==planned[k]["mobility"])]
        derived[(scope,b,m)] = metrics(scope,b,m,selected)

    key_fn = lambda x:(x["scope"],x["replicate"],x["mobility"])
    official = unique(report["evaluation_strata"],key_fn,"report strata")
    csv_rows = unique(rows(directory/"budget_summary.csv"),
                      lambda x:(x["scope"],int(x["replicate"]) if x["replicate"] else None,x["mobility"] or None),
                      "summary CSV")
    require(official.keys()==csv_rows.keys()==derived.keys(), "Stratum inventory mismatch")
    for key,expected in derived.items():
        compare_summary(official[key],expected,str(key))
        require(csv_rows[key]=={k:"" if v is None else str(v) for k,v in official[key].items()},
                f"CSV/JSON mismatch: {key}")
    require(report["all_audited_budgets_met"] is all(r["within_budget"] for r in derived.values()),
            "Budget conclusion mismatch")
    leads = [step for run in runs.values() for step in run["lead_steps"]]
    return {
        "platform":platform, "verified_runs":len(runs), "verified_strata":len(derived),
        "output_files_sha256_verified":len(FILES),
        "decision_rows":sum(r["decision_rows"] for r in runs.values()),
        "connected_decision_minutes":sum(r["connected_decision_frames"] for r in runs.values())/600,
        "median_lead_seconds":median(leads)/10 if leads else None,
        "block_mobility_over_budget":sum(not r["within_budget"] for r in derived.values()
                                         if r["scope"]=="replicate_mobility"),
        "scientific_submission":report["scientific_submission"],
        "strata":list(derived.values()),
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_directory",type=Path)
    args=parser.parse_args()
    try:
        print(json.dumps(verify(args.report_directory),indent=2,allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError, ArithmeticError) as exc:
        parser.exit(1,f"Strict R9.10 audit rejected: {exc}\n")


if __name__=="__main__":
    main()
