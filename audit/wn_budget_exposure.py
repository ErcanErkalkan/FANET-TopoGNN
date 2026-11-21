"""Step-4 R9.10 exposure/selection audit. Independent of manuscript tables.

Run from the author-review root with:
    python audit/wn_budget_exposure.py --sync-tex
The primary metric remains false alerts per FULL simulated minute. The connected
eligible-decision denominator is a separate descriptive sensitivity calculation.
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import math
import re

from verify_r9_10_strict import ROOT, BUDGET, verify

MOBILITIES = ("rwp", "gm", "mission")
MOBILITY_NAMES = {"rwp": "Random waypoint (RWP)", "gm": "Gauss--Markov (GM)", "mission": "Mission-based"}
ROWS_PER_RUN = 190
FULL_FRAMES_PER_RUN = 200
DT = .1


def load_csv(path):
    with path.open(newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def selection_receipt():
    path = ROOT / "frozen/anchor/Linux/R9_9_CALIBRATION_ALERT_PILOT.json"
    other = ROOT / "frozen/anchor/Windows/R9_9_CALIBRATION_ALERT_PILOT.json"
    a, b = (json.loads(p.read_text()) for p in (path, other))
    # Do not infer an out-of-sample benefit from the diagnostic selection role.
    for report in (a, b):
        assert report["replicate"] == 1
        assert report["threshold_grid"] == [v / 100 for v in range(5, 100, 5)]
        assert report["threshold_role_exposure_minutes"] == 12.0
        assert report["false_alert_budget_per_full_trajectory_minute"] == BUDGET
        model = report["models"]["fanet_topognn"]
        x = model["selection_diagnostic_only"]
        assert math.isclose(x["threshold"], .15)
        assert x["False_Alert_Events"] == 24 and x["True_Alert_Events"] == 22
        assert x["Ground_Truth_Fragmentation_Events"] == 159
        assert x["False_Alert_Events_per_minute"] == 2.0
        assert model["feasible_grid_points"] == 17
        assert model["feasible_grid_points_with_true_alert"] == 10
        assert model["operating_point_status"] == "DESCRIPTIVE_ONLY_IN_SAMPLE_OPERATING_POINT"
        assert model["deployment_threshold"] is None
    # Confirm selection-rule ordering against the PINNED code, not only its report.
    code = (ROOT / "frozen/r9_9_calibration_alert.py").read_text()
    assert "GRID = tuple(n / 100 for n in range(5, 100, 5))" in code
    assert "-x['Alert_Event_F1']" in code
    assert "x['False_Alert_Events_per_minute']" in code
    assert "-x['Alert_Event_Recall']" in code
    return {"threshold": .15, "selection_role_runs": 36, "selection_full_minutes": 12,
            "selection_true_alerts": 22, "selection_false_alerts": 24,
            "selection_event_f1": a["models"]["fanet_topognn"]["selection_diagnostic_only"]["Alert_Event_F1"],
            "candidate_thresholds": 19, "feasible_under_full_budget": 17,
            "feasible_with_true_alert": 10, "in_sample_diagnostic_only": True}


def build_for_source(directory):
    verified = verify(directory)
    scores = load_csv(directory / "evaluation_scores.csv")
    assert len(scores) == 180 * ROWS_PER_RUN, "Unexpected decision rows"
    connected = defaultdict(int)
    decision_rows = defaultdict(int)
    run_ids = defaultdict(set)
    per_run = defaultdict(int)
    for r in scores:
        b = int(r["replicate"])
        mob = next((x for x in MOBILITIES if r["run_id"].startswith(x + "_")), None)
        if mob is None:
            raise ValueError("Unexpected run mobility")
        key = r["run_id"]
        per_run[key] += 1
        frames = [("pooled", None, None), ("pooled_mobility", None, mob),
                  ("replicate", b, None), ("replicate_mobility", b, mob)]
        for group in frames:
            decision_rows[group] += 1
            run_ids[group].add(key)
            if float(r["beta_current"]) == 1.0:
                connected[group] += 1
    assert len(per_run) == 180 and all(n == ROWS_PER_RUN for n in per_run.values())
    rows = []
    for result in verified["strata"]:
        group = (result["scope"], result["replicate"], result["mobility"])
        assert result["runs"] == len(run_ids[group])
        assert decision_rows[group] == result["runs"] * ROWS_PER_RUN
        full_minutes = result["minutes"]
        assert math.isclose(full_minutes, result["runs"] * FULL_FRAMES_PER_RUN * DT / 60)
        c = connected[group]
        connected_minutes = c * DT / 60
        false = result["false_alerts"]
        alt_rate = false / connected_minutes if c else None
        rows.append({
            "scope": result["scope"], "replicate": result["replicate"],
            "mobility": result["mobility"], "runs": result["runs"],
            "decision_rows": decision_rows[group], "connected_decision_frames": c,
            "full_minutes": full_minutes, "connected_decision_minutes": connected_minutes,
            "false_alerts": false, "full_false_alarms_per_minute": false/full_minutes,
            "connected_decision_false_alarms_per_minute": alt_rate,
            "full_budget_met": (false/full_minutes <= BUDGET),
            "connected_numerical_reference_met": alt_rate is not None and alt_rate <= BUDGET,
        })
    assert len(rows) == 24
    return {"platform": verified["platform"], "rows": rows}


def row_key(r):
    return (r["scope"], r["replicate"], r["mobility"])


def reconcile(sources):
    reference = {row_key(r):r for r in sources[0]["rows"]}
    for source in sources[1:]:
        other = {row_key(r):r for r in source["rows"]}
        if reference.keys() != other.keys():
            raise ValueError("Exposure groups disagree between evidence sources")
        for key, r in reference.items():
            if r != other[key]:
                raise ValueError(f"Exposure mismatch across independent records: {key}")
    return reference


def get(keyed, scope, block=None, mobility=None):
    return keyed[(scope, block, mobility)]


def latex_rate(r):
    value = r["connected_decision_false_alarms_per_minute"]
    if value is None:
        return "--"
    s=f"{value:.2f}"
    return r"\textbf{"+s+"}" if value > BUDGET else s


def produce_tex(keyed):
    main = [get(keyed,"pooled")]+[get(keyed,"pooled_mobility", mobility=m) for m in MOBILITIES]
    pooled_table = "".join(" & ".join(("All runs" if x["mobility"] is None else MOBILITY_NAMES[x["mobility"]],
                         str(x["false_alerts"]),f"{x['full_minutes']:.0f}",f"{x['full_false_alarms_per_minute']:.2f}",
                         str(x["connected_decision_frames"]),f"{x['connected_decision_minutes']:.2f}",
                         latex_rate(x)))+r" \\"+"\n" for x in main)
    detail = "".join(" & ".join((str(b),MOBILITY_NAMES[m],str(get(keyed,"replicate_mobility",b,m)["false_alerts"]),
                    f"{get(keyed,'replicate_mobility',b,m)['full_false_alarms_per_minute']:.2f}",
                    str(get(keyed,"replicate_mobility",b,m)["connected_decision_frames"]),
                    f"{get(keyed,'replicate_mobility',b,m)['connected_decision_minutes']:.2f}",
                    latex_rate(get(keyed,"replicate_mobility",b,m))))+r" \\"+"\n"
                    for b in (2,3,4,5,6) for m in MOBILITIES)
    return pooled_table, detail


def replace_block(file, label, replacement):
    before = file.read_text(encoding="utf-8")
    start, end = f"% BEGIN {label}", f"% END {label}"
    expression=rf"(?m:^{re.escape(start)}$\n).*?(?m:^{re.escape(end)}$)"
    after,count = re.subn(expression, lambda match:start+"\n"+replacement+end, before, flags=re.S)
    if count != 1:
        raise ValueError(f"Missing unique block '{label}' in {file.name}")
    file.write_text(after,encoding="utf-8",newline="\n")


def run(sync_tex=False):
    selection=selection_receipt()
    paths=[next((ROOT/f"audit/reference_{host}").iterdir()) for host in ("linux","windows")]
    paths.append(ROOT/"evidence/pilot_linux")
    sources=[build_for_source(x) for x in paths]
    keyed=reconcile(sources)
    pooled=get(keyed,"pooled")
    assert pooled["false_alerts"]==88
    assert pooled["decision_rows"] == 34200
    assert pooled["connected_decision_frames"] == 28301
    full_over=sum(not r["full_budget_met"] for r in keyed.values() if r["scope"]=="replicate_mobility")
    connected_over=sum(not r["connected_numerical_reference_met"] for r in keyed.values() if r["scope"]=="replicate_mobility")
    assert (full_over,connected_over)==(6,8)
    results=list(keyed.values())
    target=ROOT/"data_derived/wn_r9_10_exposure_sensitivity.csv"
    with target.open("w",newline="",encoding="utf-8") as f:
        wr=csv.DictWriter(f, fieldnames=list(results[0]), lineterminator="\n")
        wr.writeheader();wr.writerows(results)
    metadata={"schema":"wn_r9_10_budget_exposure_sensitivity_v1",
        "reference_sources_verified": [s["platform"] for s in sources],
        "selection":selection, "threshold_unchanged":True,
        "full_minutes": pooled["full_minutes"], "eligible_connected_decision_frames": pooled["connected_decision_frames"],
        "eligible_connected_decision_minutes":pooled["connected_decision_minutes"],
        "false_alarms":pooled["false_alerts"], "full_exposure_rate":pooled["full_false_alarms_per_minute"],
        "connected_exposure_rate":pooled["connected_decision_false_alarms_per_minute"],
        "block_mobility_full_budget_exceedances":full_over,
        "block_mobility_connected_numerical_reference_exceedances":connected_over,
        "connected_exposure_is_secondary_sensitivity_not_operational_budget":True,
        "no_new_training_or_threshold_tuning":True}
    output=ROOT/"evidence/STEP4_EXPOSURE_THRESHOLD_RECEIPT.json"
    output.write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
    if sync_tex:
        pooled_tex, detail_tex=produce_tex(keyed)
        supplementary=ROOT/"paper/manuscript/ESM_1.tex"
        replace_block(supplementary,"VERIFIED EXPOSURE SENSITIVITY POOLED",pooled_tex)
        replace_block(supplementary,"VERIFIED EXPOSURE SENSITIVITY BLOCK MOBILITY",detail_tex)
    return metadata


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sync-tex",action="store_true")
    a=parser.parse_args()
    print(json.dumps(run(sync_tex=a.sync_tex),indent=2))
