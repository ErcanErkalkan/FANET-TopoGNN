from __future__ import annotations
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from scripts.audit_r4_new_evidence import audit_r4_evidence
from fanet.r4_evidence import export_torch_evidence


def snap(group,t,positive):
 return SimpleNamespace(run_id=f'run_{group}',split_group_id=group,
    time_index=t,future_time_index=t+6,frag_at_horizon=int(positive),
    beta_current=1.,beta_target=2. if positive else 1.,target_available=True)

class AuditTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory()
  self.addCleanup(self.tmp.cleanup)
  self.output=Path(self.tmp.name)
  self.seed_dir=self.output/'per_seed'/'seed_7'
  self.val=[snap(f'v{g}',t,t%2==0) for g in range(6) for t in range(8)]
  self.test=[snap('t',t,t%2==0) for t in range(8)]
  self.v_pred=[1.8 if s.frag_at_horizon else 1.05 for s in self.val]
  self.t_pred=[1.8 if s.frag_at_horizon else 1.05 for s in self.test]
 def export(self,name,test=None):
  return export_torch_evidence(seed_dir=self.seed_dir,seed=7,task_name=name,
      model=torch.nn.Linear(1,1),train_cfg={'epochs':1},config_sha256='a'*64,
      horizon_steps=6,validation_snapshots=self.val,validation_beta_hat=self.v_pred,
      test_snapshots=test or self.test,test_beta_hat=self.t_pred,
      early_stopping_group_ids=['stop_only'])
 def test_missing_one_evidence_returns_hold(self):
  self.export('GCN')
  audit=audit_r4_evidence(self.output,[7],expected_models={'GCN','GAT'},expected_signature='a'*64)
  self.assertEqual(audit['status'],'HOLD')
  self.assertEqual(audit['model_seed_evidence_verified'],1)
 def test_two_identical_models_pass_inventory_not_scientific_release(self):
  self.export('GCN');self.export('GAT')
  audit=audit_r4_evidence(self.output,[7],expected_models={'GCN','GAT'},expected_signature='a'*64)
  self.assertEqual(audit['status'],'PASS')
  self.assertEqual(audit['scientific_submission'],'HOLD')
 def test_cross_model_label_difference_is_rejected(self):
  self.export('GCN')
  other=[snap('t',t,t%2==0) for t in range(8)]
  other[0].frag_at_horizon=0;other[0].beta_target=1.
  self.export('GAT',other)
  audit=audit_r4_evidence(self.output,[7],expected_models={'GCN','GAT'},expected_signature='a'*64)
  self.assertEqual(audit['status'],'HOLD')
  self.assertTrue(any('different held-out' in s for s in audit['errors']))

if __name__=='__main__':unittest.main()
