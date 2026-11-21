"""Regression tests for the endpoint-state versus onset diagnostic."""
import csv
from pathlib import Path
import sys
import unittest

AUDIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AUDIT))
from wn_target_alignment import KEYS, OUTPUT, assess_run, audit_report, reports


class TargetAlignmentTests(unittest.TestCase):
    def test_transient_onset_differs_from_endpoint(self):
        c = assess_run({t: 2 if t == 5 else 1 for t in range(13)}, {4: .15})
        self.assertEqual([c[k] for k in KEYS], [1, 1, 1, 1, 1, 0])

    def test_exact_right_boundary_is_included(self):
        c = assess_run({t: 1 if t < 10 else 2 for t in range(13)}, {4: .15})
        self.assertEqual([c[k] for k in KEYS], [1, 1, 0, 1, 0, 0])

    def test_warnability_requires_eligible_connected_decision(self):
        truth = {t: 2 if t < 4 or t >= 8 else 1 for t in range(12)}
        c = assess_run(truth, {3: .95})
        self.assertEqual([c[k] for k in KEYS], [1, 0, 0, 0, 0, 0])

    def test_event_and_matched_alert_diagnostics_use_different_denominators(self):
        truth = {t: 2 if t in {7, 8} else 1 for t in range(13)}
        c = assess_run(truth, {1: .10, 3: .15})
        self.assertEqual([c[k] for k in KEYS], [1, 1, 0, 1, 1, 0])

    def test_three_verified_records_and_csv_agree(self):
        expected = dict(zip(KEYS, (1114, 1088, 102, 183, 89, 88)))
        for path in reports():
            self.assertEqual(audit_report(path), expected)
        with OUTPUT.open(newline="", encoding="utf-8") as f:
            self.assertEqual(list(csv.DictReader(f)), [{k: str(v) for k, v in expected.items()}])
