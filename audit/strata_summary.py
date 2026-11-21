"""Compute and export research summary tables from verified records."""
import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
import re

from verify_r9_10_strict import BUDGET, ROOT, verify

MOBILITY = {None:"All runs","rwp":"Random waypoint (RWP)","gm":"Gauss--Markov (GM)",
            "mission":"Mission-based"}
NODE_COUNTS = (10, 20, 30)
MOBILITY_ORDER = ("rwp", "gm", "mission")
DEFAULT_TABLE = "VERIFIED R9.10 TABLE"
NODE_TABLE = "VERIFIED R9.10 BY NODE COUNT"
MOBILITY_NODE_TABLE = "VERIFIED R9.10 BY MOBILITY AND NODE COUNT"

def refresh(path, text, table=DEFAULT_TABLE):
    start, end = f"% BEGIN {table}", f"% END {table}"
    old=path.read_text(encoding="utf-8")
    pattern=rf"(?m:^{re.escape(start)}$\n).*?(?m:^{re.escape(end)}$)"
    updated,n=re.subn(pattern,lambda _:start+"\n"+text+end,old,flags=re.S)
    if n!=1:
        raise ValueError(f"Expected one generated block in {path}")
    path.write_text(updated,encoding="utf-8",newline="\n")

def size_strata(run_counts_path):
    """Use per-run counts already checked against trajectories and decisions by verify()."""
    groups = defaultdict(list)
    with run_counts_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            n = int(row["n_nodes"])
            groups[None, n].append(row)
            groups[row["mobility"], n].append(row)
    keys = [(None, n) for n in NODE_COUNTS]
    keys += [(mobility, n) for mobility in MOBILITY_ORDER for n in NODE_COUNTS]
    if set(groups) != set(keys):
        raise ValueError("Unexpected mobility and node-count groups")
    result = []
    for mobility, n in keys:
        rows = groups[mobility, n]
        runs = len(rows)
        trajectory_groups = len({row["split_group_id"] for row in rows})
        if (runs, trajectory_groups) != ((60, 15) if mobility is None else (20, 5)):
            raise ValueError(f"Incomplete node-count group: {mobility}/{n}")
        totals = {k: sum(int(r[k]) for r in rows) for k in
                  ("observed_frames", "events", "warnable_events", "true_alerts", "false_alerts")}
        minutes = totals.pop("observed_frames") / 600
        events, tp, fp = (totals[k] for k in ("events", "true_alerts", "false_alerts"))
        result.append({
            "scope": "node_count" if mobility is None else "mobility_node_count",
            "mobility": mobility, "n_nodes": n, "runs": runs,
            "trajectory_groups": trajectory_groups, "minutes": minutes, **totals,
            "alerts": tp + fp, "missed_events": events - tp,
            "event_recall": tp / events if events else None,
            "event_precision": tp / (tp + fp) if tp + fp else None,
            "false_alerts_per_minute": fp / minutes, "within_budget": fp / minutes <= BUDGET,
        })
    return result


def export(strata, by_size):
    d=ROOT/"data_derived"
    d.mkdir(exist_ok=True)
    for filename,selected in (("wn_r9_10_aggregate.csv",[x for x in strata if x["replicate"] is None]),
                              ("wn_r9_10_all_strata.csv",strata),
                              ("wn_r9_10_by_node_count.csv",by_size[:3]),
                              ("wn_r9_10_by_mobility_node_count.csv",by_size[3:])):
        with (d/filename).open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,list(selected[0]),lineterminator="\n")
            w.writeheader()
            w.writerows(selected)

def short_row(x):
    v=x["event_recall"]
    rate=x["false_alerts_per_minute"]
    rate_text=(r"\textbf{"+f"{rate:.2f}"+"}") if not x["within_budget"] else f"{rate:.2f}"
    return " & ".join((MOBILITY[x["mobility"]],str(x["events"]),str(x["true_alerts"]),
                      f"{100*v:.1f}" if v is not None else "--",str(x["false_alerts"]),
                      f'{x["minutes"]:.0f}',rate_text))+r" \\"+"\n"

def detailed_row(x):
    text=(str(x["replicate"]),MOBILITY[x["mobility"]],str(x["trajectory_groups"]),
          str(x["runs"]),str(x["events"]),str(x["true_alerts"]),str(x["false_alerts"]),
          f'{x["minutes"]:.0f}',f'{x["false_alerts_per_minute"]:.2f}',
          "Yes" if x["within_budget"] else "No")
    return " & ".join(text)+r" \\"+"\n"

def size_row(x, detailed=False):
    def pct(value):
        return f"{100*value:.1f}" if value is not None else "--"
    rate = f'{x["false_alerts_per_minute"]:.2f}'
    if not x["within_budget"]:
        rate = r"\textbf{" + rate + "}"
    prefix = (MOBILITY[x["mobility"]],) if detailed else ()
    counts = (str(x["n_nodes"]), str(x["runs"]), str(x["events"]), str(x["true_alerts"]),
              str(x["false_alerts"]))
    suffix = (pct(x["event_recall"]), pct(x["event_precision"]), rate)
    return " & ".join(prefix + counts + suffix) + r" \\" + "\n"


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot",type=Path)
    args=p.parse_args()
    paths=[next((ROOT/"audit"/f"reference_{host}").iterdir()) for host in ("linux","windows")]
    if args.pilot:
        paths.append(args.pilot)
    refs=[verify(path) for path in paths]
    if not all(x["strata"]==refs[0]["strata"] and
               x["median_lead_seconds"]==refs[0]["median_lead_seconds"] for x in refs):
        raise ValueError("Verified result sources do not agree")
    strata=refs[0]["strata"]
    by_size=[size_strata(path/"run_counts.csv") for path in paths]
    if not all(x == by_size[0] for x in by_size):
        raise ValueError("Verified sources disagree on node-count strata")
    for key in ("runs", "events", "warnable_events", "true_alerts", "false_alerts", "minutes"):
        for part in (by_size[0][:3], by_size[0][3:]):
            if not abs(sum(x[key] for x in part) - strata[0][key]) < 1e-9:
                raise ValueError(f"Node-count strata do not reconcile to pooled {key}")
    export(strata, by_size[0])
    print(json.dumps({"verified_sources":len(refs),"pooled_rows":4,"node_count_rows":3,
                      "block_mobility_rows":15,"mobility_node_rows":9,"original_strata":len(strata)}))

if __name__=="__main__":
    main()
