"""Check node-count results, data boundaries, and generated manuscript rows."""
import csv
from pathlib import Path
import sys
import unittest

AUDIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AUDIT))
from generate_wn_tables import MOBILITY_NODE_TABLE, NODE_TABLE, size_row, size_strata

ROOT = AUDIT.parent
PILOT = ROOT / "evidence/pilot_linux/run_counts.csv"
REFERENCES = [next((AUDIT / f"reference_{host}").iterdir()) / "run_counts.csv"
              for host in ("linux", "windows")]


class NodeStrataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = size_strata(PILOT)

    def test_three_sources_agree(self):
        for reference in REFERENCES:
            self.assertEqual(size_strata(reference), self.result)

    def test_node_totals_and_silence(self):
        a, b, c = self.result[:3]
        self.assertEqual([(x["n_nodes"], x["events"], x["true_alerts"], x["false_alerts"])
                          for x in (a, b, c)],
                         [(10, 599, 158, 77), (20, 433, 25, 11), (30, 82, 0, 0)])
        self.assertEqual(c["event_recall"], 0.0)
        self.assertIsNone(c["event_precision"])
        self.assertEqual(c["alerts"], 0)
        self.assertTrue(c["within_budget"])
        self.assertAlmostEqual(a["false_alerts_per_minute"], 3.85)

    def test_mobility_node_margins_and_precision(self):
        groups = self.result[3:]
        self.assertEqual(len(groups), 9)
        self.assertTrue(all(x["runs"] == 20 and x["trajectory_groups"] == 5 for x in groups))
        for a in self.result[:3]:
            b = [r for r in groups if r["n_nodes"] == a["n_nodes"]]
            for col in ("events", "false_alerts", "true_alerts", "runs"):
                self.assertEqual(sum(r[col] for r in b), a[col])
        x = {(r["mobility"], r["n_nodes"]): r for r in groups}
        self.assertIsNone(x["mission", 20]["event_precision"])
        self.assertEqual(x["rwp", 20]["event_precision"], 0.0)
        self.assertEqual(x["gm", 10]["false_alerts"], 39)

    def test_released_tables_match_code(self):
        targets = [
            ("main.tex", NODE_TABLE, self.result[:3], False, "by_node_count"),
            ("ESM_1.tex", MOBILITY_NODE_TABLE, self.result[3:], True, "by_mobility_node_count"),
        ]
        for name, marker, rows, detailed, key in targets:
            s = (ROOT / "paper/manuscript" / name).read_text().split(f"% BEGIN {marker}\n")[1]
            self.assertEqual(s.split(f"% END {marker}")[0],
                             "".join(size_row(x, detailed) for x in rows))
            with (ROOT / "data_derived" / f"wn_r9_10_{key}.csv").open(newline="") as f:
                actual = list(csv.DictReader(f))
            self.assertEqual(actual,
                             [{k: "" if v is None else str(v) for k, v in row.items()} for row in rows])
