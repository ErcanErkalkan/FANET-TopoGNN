"""Regression tests of the strict independent R9.10 verifier."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

AUDIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AUDIT))
from verify_r9_10_strict import verify, event_matches, metrics, rows, REPORT

REFERENCE = next((AUDIT / "reference_linux").iterdir())
WINDOWS = next((AUDIT / "reference_windows").iterdir())


def dump_csv(path, data):
    with path.open("w", newline="",encoding="utf-8") as f:
        out = csv.DictWriter(f, fieldnames=list(data[0]), lineterminator="\n")
        out.writeheader()
        out.writerows(data)


class EventMatchingTests(unittest.TestCase):
    def test_inclusive_horizon_and_last_frame(self):
        truth = {t: 2 if t==199 else 1 for t in range(200)}
        hit = event_matches(truth,{193:.15})
        self.assertEqual(hit["lead_steps"],[6])
        self.assertEqual(hit["missed_events"],0)
        miss = event_matches(truth,{192:.15})
        self.assertEqual(miss["false_alerts"],1)
        self.assertEqual(miss["missed_events"],1)
        self.assertEqual(miss["warnable_events"],0)

    def test_no_double_count_of_alarm_episode(self):
        truth = {t:1 if t<11 else 2 for t in range(20)}
        a = event_matches(truth,{4:.15,5:.35,6:.65})
        self.assertEqual(a["false_alerts"],1)
        self.assertEqual(a["true_alerts"],0)

    def test_cooldown_uses_last_emitted_alarm(self):
        truth = {t:1 for t in range(20)}
        a = event_matches(truth,{4:.9,8:.8,10:.4})
        self.assertEqual(a["false_alerts"],2)

    def test_no_alert_recall_precision_and_f1(self):
        truth = {0:1,1:2,2:2}
        a = event_matches(truth,{0:.1})
        self.assertEqual(a["missed_events"],1)
        m=metrics("pooled",None,None,[a | {"split_group_id":"a"}])
        self.assertEqual(m["event_recall"],0)
        self.assertIsNone(m["event_precision"])
        self.assertEqual(m["event_f1"],0)
        z=event_matches({0:1,1:1},{0:.1})
        zero=metrics("pooled",None,None,[z | {"split_group_id":"a"}])
        self.assertIsNone(zero["event_f1"])

    def test_connected_decisions_alone_are_eligible(self):
        a=event_matches({0:1,1:2,2:2,3:1,4:2},{1:.99,3:.15})
        self.assertEqual(a["events"],2)
        self.assertEqual(a["true_alerts"],1)
        self.assertEqual(a["false_alerts"],0)
        self.assertEqual(a["connected_decision_frames"],1)

    def test_one_alert_matches_at_most_one_event(self):
        a=event_matches({0:1,1:2,2:1,3:2,4:1},{0:.9})
        self.assertEqual(a["true_alerts"],1)
        self.assertEqual(a["missed_events"],1)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.copy=Path(self.tmp.name)/"report"
        shutil.copytree(REFERENCE,self.copy)
        self.jsonp=self.copy/REPORT

    def patch_report(self,fn):
        report=json.loads(self.jsonp.read_bytes())
        fn(report)
        self.jsonp.write_text(json.dumps(report))

    def patch_csv(self, filename,fn, report_action=lambda r:None):
        p=self.copy/filename
        data=rows(p)
        fn(data)
        dump_csv(p,data)
        def adjust(r):
            r["output_sha256"][filename]=hashlib.sha256(p.read_bytes()).hexdigest()
            report_action(r)
        self.patch_report(adjust)

    def test_frozen_linux_and_windows_agree(self):
        a,b = verify(REFERENCE),verify(WINDOWS)
        self.assertEqual(a["verified_runs"],180)
        self.assertEqual(a["strata"],b["strata"])
        self.assertEqual(a["median_lead_seconds"],.2)

    def test_self_consistent_but_false_metrics_are_rejected(self):
        for key in ("event_precision","event_recall","event_f1"):
            with self.subTest(metric=key):
                shutil.copyfile(REFERENCE/"budget_summary.csv",self.copy/"budget_summary.csv")
                shutil.copyfile(REFERENCE/REPORT,self.jsonp)
                self.patch_csv("budget_summary.csv",lambda x:x[0].update({key:"0.99"}),
                               lambda r:r["evaluation_strata"][0].update({key:.99}))
                with self.assertRaisesRegex(ValueError,"Independent summary mismatch"):
                    verify(self.copy)

    def test_plan_identity_mismatch_rejected(self):
        self.patch_csv("run_counts.csv",lambda x:x[0].update({"graph_policy":"WRONG"}))
        with self.assertRaisesRegex(ValueError,"Plan identity mismatch"):
            verify(self.copy)

    def test_historical_status_cannot_be_promoted(self):
        self.patch_report(lambda r:r.update(scientific_submission="PASS",
                                             external_freeze_complete=True))
        with self.assertRaisesRegex(ValueError,"publication status"):
            verify(self.copy)

    def test_external_freeze_flag_must_be_false(self):
        self.patch_report(lambda r:r.update(external_freeze_complete=True))
        with self.assertRaisesRegex(ValueError,"external_freeze_complete"):
            verify(self.copy)

    def test_duplicate_decision_rejected(self):
        self.patch_csv("evaluation_scores.csv",lambda x:x.append(x[0].copy()))
        with self.assertRaisesRegex(ValueError,"duplicate decision row"):
            verify(self.copy)

    def test_missing_decision_rejected(self):
        self.patch_csv("evaluation_scores.csv",lambda x:x.pop())
        with self.assertRaisesRegex(ValueError,"Incomplete decision coverage"):
            verify(self.copy)

    def test_calibration_must_match_same_platform_anchor(self):
        self.patch_report(lambda r:r["calibrator"].update(slope=0.0))
        with self.assertRaisesRegex(ValueError,"frozen calibration"):
            verify(self.copy)


if __name__=="__main__":
    unittest.main()
