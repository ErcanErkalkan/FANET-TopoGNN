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

from fanet.r4_evidence import (export_torch_evidence, verify_evidence, evidence_location,
                               _validation_partition, _check_split, NEURAL_NAMES)


class TinyRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer=nn.Linear(1,1)

    def forward(self, x):
        return self.layer(x)


def make_snap(group, idx, positive, *, horizon=6, prefix='run'):
    return SimpleNamespace(run_id=f'{prefix}_{group}', split_group_id=group, time_index=idx,
                           future_time_index=idx+horizon, frag_at_horizon=int(positive),
                           beta_current=1., beta_target=(2. if positive else 1.),
                           target_available=True)


def samples():
    val=[]
    for g in ['v1','v2','v3','v4','v5','v6']:
        for t in range(10):
            val.append(make_snap(g,t,t%2==0))
    test=[make_snap('test_a',t,t%2==0) for t in range(12)]
    vp=np.asarray([1.65 if s.frag_at_horizon else 1.05 for s in val])
    tp=np.asarray([1.65 if s.frag_at_horizon else 1.05 for s in test])
    return val, vp, test, tp


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.model=TinyRegressor()
        self.val,self.vpred,self.test,self.tpred=samples()
        self.args=dict(seed_dir=self.root, seed=123, task_name='GCN', model=self.model,
                       train_cfg={'lr':0.002},config_sha256='a'*64,horizon_steps=6,
                       validation_snapshots=self.val,validation_beta_hat=self.vpred,
                       test_snapshots=self.test,test_beta_hat=self.tpred)

    def test_real_state_dict_and_calibrated_scores_persist(self):
        d=export_torch_evidence(**self.args)
        self.assertTrue(verify_evidence(self.root,'GCN','a'*64))
        self.assertFalse(verify_evidence(self.root,'GCN','b'*64))
        path=evidence_location(self.root,'GCN')
        weights=torch.load(path/'checkpoint_state_dict.pt',weights_only=True,map_location='cpu')
        self.assertEqual(weights.keys(),self.model.state_dict().keys())
        for k,v in weights.items():
            torch.testing.assert_close(v,self.model.state_dict()[k])
        with (path/'validation_calibration_fit.csv').open(newline='') as f: fit=list(csv.DictReader(f))
        with (path/'validation_threshold_selection.csv').open(newline='') as f: select=list(csv.DictReader(f))
        with (path/'test.csv').open(newline='') as f: test=list(csv.DictReader(f))
        self.assertFalse({x['split_group_id'] for x in fit}&{x['split_group_id'] for x in select})
        self.assertFalse({x['split_group_id'] for x in select}&{x['split_group_id'] for x in test})
        self.assertTrue(all(x['risk_probability']=='' for x in fit))
        calibration=json.loads((path/'calibration.json').read_text())
        for row in select+test:
            z=float(row['predicted_beta'])*calibration['coefficient']+calibration['intercept']
            p=1/(1+np.exp(-z))
            self.assertAlmostEqual(p,float(row['risk_probability']),places=11)
        self.assertEqual(d['samples']['test'],12)
        self.assertTrue(len(select)>0)

    def test_missing_tampered_files_fail_closed(self):
        export_torch_evidence(**self.args)
        path=evidence_location(self.root,'GCN')
        with (path/'test.csv').open('a') as f:f.write('tampered\n')
        self.assertFalse(verify_evidence(self.root,'GCN','a'*64))

    def test_overwrite_refused(self):
        export_torch_evidence(**self.args)
        with self.assertRaises(FileExistsError):export_torch_evidence(**self.args)

    def test_invalid_horizon_rejected(self):
        self.test[0].future_time_index+=1
        with self.assertRaisesRegex(ValueError,'horizon'):
            export_torch_evidence(**self.args)

    def test_misaligned_scores_rejected(self):
        self.args['test_beta_hat']=self.tpred[:-1]
        with self.assertRaisesRegex(ValueError,'alignment'):
            export_torch_evidence(**self.args)

    def test_leakage_rejected(self):
        self.test[0].split_group_id=self.val[0].split_group_id
        with self.assertRaisesRegex(ValueError,'leakage'):
            export_torch_evidence(**self.args)

    def test_single_class_calibration_rejected(self):
        for s in self.val:s.frag_at_horizon=0;s.beta_target=1.
        with self.assertRaisesRegex(ValueError,'one class'):
            export_torch_evidence(**self.args)

    def test_surrogate_rejected(self):
        self.args['model']=object()
        with self.assertRaisesRegex(TypeError,'torch.nn.Module'):
            export_torch_evidence(**self.args)

    def test_duplicate_keys_rejected(self):
        self.test[1].run_id=self.test[0].run_id
        self.test[1].time_index=self.test[0].time_index
        with self.assertRaisesRegex(ValueError,'duplicate'):
            export_torch_evidence(**self.args)

    def test_no_unlabeled_tail(self):
        self.test[1].target_available=False
        with self.assertRaisesRegex(ValueError,'unlabeled'):
            export_torch_evidence(**self.args)

    def test_frozen_neural_inventory(self):
        self.assertEqual(len(NEURAL_NAMES),9)
        self.assertIn('tgn:5',NEURAL_NAMES)
        self.assertNotIn('Shallow ML',NEURAL_NAMES)


if __name__=='__main__': unittest.main()
