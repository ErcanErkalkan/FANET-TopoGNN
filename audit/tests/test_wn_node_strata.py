"""Check node-count results and data boundaries."""
import csv
from pathlib import Path
import sys
import unittest

AUDIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AUDIT))
from strata_summary import size_strata

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

    def test_derived_csv_matches_verified_data(self):
        for filename,expected in (("wn_r9_10_by_node_count.csv",self.result[:3]),
                                  ("wn_r9_10_by_mobility_node_count.csv",self.result[3:])):
            with (ROOT/"data_derived"/filename).open(newline="") as f:
                actual=list(csv.DictReader(f))
            self.assertEqual(actual,[{k:"" if v is None else str(v) for k,v in row.items()}
                                     for row in expected])
