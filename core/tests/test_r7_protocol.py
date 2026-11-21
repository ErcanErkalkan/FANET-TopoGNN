"""R7 regression checks for full-truth event protocol and comparable model support."""
from __future__ import annotations

from dataclasses import replace
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from fanet.dataset import build_dataset_views, train_val_test_split, relabel_forecast_horizon
from fanet.r7_protocol import select_full_test_trajectory, score_and_audit, validate_support_inventory
import test_strict_horizon as r0_tests


class FullTruthProtocolTests(unittest.TestCase):
    def setUp(self):
        self.cfg = r0_tests.StrictHorizonTests.config()
        self.views = build_dataset_views(self.cfg, seed=40)
        self.train, self.val, self.test = train_val_test_split(self.views.forecast_eligible, split_seed=2026)
        self.full = select_full_test_trajectory(self.views.full_trajectory, self.test)

    def _score(self, destination=None, model='SampleModel', snaps=None, risk=None):
        snaps = self.test if snaps is None else snaps
        risk = np.zeros(len(snaps)) if risk is None else risk
        return score_and_audit(scored_snapshots=snaps, predictions=np.ones(len(snaps)), risks=risk,
                               full_test_trajectory=self.full, dt=.1, horizon_steps=2,
                               threshold=.5, model=model,
                               score_kind='uncalibrated_or_binary_risk_score', destination=destination)

    def test_full_test_has_more_frames_than_labeled_subset(self):
        self.assertEqual({s.run_id for s in self.full}, {s.run_id for s in self.test})
        self.assertEqual(len(self.full)-len(self.test), 2*len({s.run_id for s in self.full}))

    def test_new_metric_covers_full_observed_timeline(self):
        r=self._score()
        self.assertEqual(r['forecast_scored_rows'], len(self.test))
        self.assertEqual(r['full_truth_rows'], len(self.full))
        self.assertFalse(r['legacy_event_metrics_replaced'])

    def test_terminal_truth_event_survives(self):
        seq = sorted([s for s in self.full if s.run_id == self.full[0].run_id],key=lambda s:s.time_index)
        changed=[replace(s,beta_current=(2. if s.time_index == seq[-1].time_index else 1.)) for s in seq]
        relabeled=relabel_forecast_horizon(changed,2,include_unlabeled_tail=True)
        eligible=[s for s in relabeled if s.target_available]
        risks=np.zeros(len(eligible));risks[-1]=1.
        r=score_and_audit(scored_snapshots=eligible,predictions=np.ones(len(eligible)),risks=risks,
                          full_test_trajectory=relabeled,dt=.1,horizon_steps=2,threshold=.5,
                          model='edge',score_kind='uncalibrated_or_binary_risk_score')
        self.assertEqual(r['full_timeline_event_metrics']['Ground_Truth_Fragmentation_Events'],1.)
        self.assertEqual(r['full_timeline_event_metrics']['True_Alert_Events'],1.)

    def test_reject_misaligned_risk_and_score(self):
        with self.assertRaises(ValueError):self._score(risk=np.zeros(len(self.test)-1))
        with self.assertRaises(ValueError):self._score(risk=np.full(len(self.test),np.nan))
        with self.assertRaises(ValueError):self._score(risk=np.full(len(self.test),1.1))

    def test_reject_test_run_inventory_drift(self):
        with self.assertRaises(ValueError):
            self._score(snaps=self.test[:-6])

    def test_immutable_csv_and_same_support(self):
        with tempfile.TemporaryDirectory() as td:
            a=Path(td)/'a.csv';b=Path(td)/'b.csv'
            self._score(destination=a,model='a');self._score(destination=b,model='b')
            self.assertEqual(validate_support_inventory([a,b],['a','b'])['status'], 'COMMON_TEST_SUPPORT_PASS')
            with self.assertRaises(FileExistsError):self._score(destination=a,model='a')
            with b.open('a') as f:f.write('\n')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                validate_support_inventory([a,b],['a','b'])

    def test_reject_unequal_model_support_even_when_hashes_updated(self):
        with tempfile.TemporaryDirectory() as td:
            a=Path(td)/'a.csv';b=Path(td)/'b.csv'
            self._score(destination=a,model='a')
            self._score(destination=b,model='b',snaps=self.test[1:],risk=np.zeros(len(self.test)-1))
            with self.assertRaisesRegex(ValueError,'test time support'):
                validate_support_inventory([a,b],['a','b'])

if __name__=='__main__':unittest.main()

class CrossModelTamperTests(FullTruthProtocolTests):
    def test_reject_truth_change_after_attacker_rehashes(self):
        from fanet.r7_protocol import sha256
        with tempfile.TemporaryDirectory() as td:
            a=Path(td)/'a.csv';b=Path(td)/'b.csv'
            self._score(destination=a,model='a')
            self._score(destination=b,model='b')
            with b.open(newline='',encoding='utf-8') as fh:
                reader=csv.DictReader(fh);cols=reader.fieldnames;rows=list(reader)
            rows[0]['beta_current']=str(float(rows[0]['beta_current'])+0.125)
            with b.open('w',newline='',encoding='utf-8') as fh:
                writer=csv.DictWriter(fh,fieldnames=cols);writer.writeheader();writer.writerows(rows)
            meta=json.loads(b.with_suffix('.json').read_text())
            meta['score_csv_sha256']=sha256(b)
            b.with_suffix('.json').write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError,'Nonidentical target truth'):
                validate_support_inventory([a,b],['a','b'])
