"""R4 tests: real torch CPU export plus isolation and predeclared threshold."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

from fanet.r4_evidence import (
    split_early_stopping_validation, _validation_partition,
    select_probability_threshold, export_torch_evidence, verify_evidence,
    evidence_location,
)


def item(group: str, t: int, positive: bool):
    return SimpleNamespace(
        run_id=f'r_{group}', split_group_id=group,
        time_index=t, future_time_index=t+6,
        beta_current=1.0, beta_target=2.0 if positive else 1.0,
        frag_at_horizon=int(positive), target_available=True,
    )


class StrictRoleTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.validation=[item(f'group_{k}', t, t%3==0) for k in range(9) for t in range(12)]
        self.test=[item('test_only', t, t%3==0) for t in range(15)]
        self.model=nn.Linear(1,1)

    def _export(self):
        early, remaining=split_early_stopping_validation(self.validation, 2026)
        val_pred=np.array([1.75 if s.frag_at_horizon else 1.07 for s in remaining])
        test_pred=np.array([1.75 if s.frag_at_horizon else 1.07 for s in self.test])
        return export_torch_evidence(
            seed_dir=self.root, seed=2026, task_name='GCN', model=self.model,
            train_cfg={'epochs':1}, config_sha256='a'*64,
            horizon_steps=6, validation_snapshots=remaining, validation_beta_hat=val_pred,
            test_snapshots=self.test, test_beta_hat=test_pred,
            early_stopping_group_ids=sorted({s.split_group_id for s in early}),
        )

    def test_three_roles_are_disjoint_deterministic(self):
        a,b=split_early_stopping_validation(self.validation,2026)
        c,d=split_early_stopping_validation(self.validation,2026)
        self.assertEqual([s.run_id for s in a],[s.run_id for s in c])
        self.assertEqual([s.run_id for s in b],[s.run_id for s in d])
        mask_cal,mask_sel=_validation_partition(b,2026)
        roles=[{s.split_group_id for s in a},
               {s.split_group_id for s,flag in zip(b,mask_cal) if flag},
               {s.split_group_id for s,flag in zip(b,mask_sel) if flag},
               {s.split_group_id for s in self.test}]
        self.assertTrue(all(roles))
        self.assertTrue(all(roles[i].isdisjoint(roles[j]) for i in range(4) for j in range(i+1,4)))

    def test_requires_three_groups(self):
        with self.assertRaisesRegex(ValueError,'>=3'):
            split_early_stopping_validation(self.validation[:12*2],2026)

    def test_export_contains_frozen_threshold_and_probability(self):
        manifest=self._export()
        self.assertEqual(manifest['schema'],'fanet_r4_neural_evidence_v2')
        self.assertTrue(manifest['early_stopping_groups'])
        self.assertTrue(verify_evidence(self.root,'GCN','a'*64))
        edir=evidence_location(self.root,'GCN')
        self.assertEqual(json.loads((edir/'threshold_selection.json').read_text()),
                         manifest['threshold_selection'])
        self.assertTrue(manifest['threshold_selection']['selected_threshold']>=0.05)
        with (edir/'test.csv').open(newline='') as f:
            rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),len(self.test))
        self.assertTrue(all(0<=float(x['risk_probability'])<=1 for x in rows))
        weights=torch.load(edir/'checkpoint_state_dict.pt',weights_only=True,map_location='cpu')
        for name,value in weights.items():
            torch.testing.assert_close(value,self.model.state_dict()[name])

    def test_unvalidated_early_stop_overlap_is_rejected(self):
        early,remain=split_early_stopping_validation(self.validation,2026)
        val_pred=np.array([1.7 if s.frag_at_horizon else 1.1 for s in remain])
        with self.assertRaisesRegex(ValueError,'Early stopping group leakage'):
            export_torch_evidence(seed_dir=self.root,seed=2026,task_name='GCN',
                model=self.model,train_cfg={},config_sha256='a'*64,horizon_steps=6,
                validation_snapshots=remain,validation_beta_hat=val_pred,
                test_snapshots=self.test,test_beta_hat=np.ones(len(self.test)),
                early_stopping_group_ids=[remain[0].split_group_id])

    def test_corrupt_threshold_file_is_rejected(self):
        self._export()
        p=evidence_location(self.root,'GCN')/'threshold_selection.json'
        p.write_text(p.read_text().replace('0.05','0.051'))
        self.assertFalse(verify_evidence(self.root,'GCN','a'*64))

    def test_threshold_selection_is_frozen_and_explicit(self):
        out=select_probability_threshold([0.1,0.2,0.8,0.9],[0,0,1,1])
        self.assertEqual(out['protocol'],'validation_threshold_only_sample_risk_f1_v1')
        self.assertEqual(out['selected_threshold'],0.8)

    def test_threshold_rejects_nan_or_mismatch(self):
        with self.assertRaises(ValueError): select_probability_threshold([float('nan')],[0])
        with self.assertRaises(ValueError): select_probability_threshold([0.3,0.5],[1])


if __name__=='__main__': unittest.main()
