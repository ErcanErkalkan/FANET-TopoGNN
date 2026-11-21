"""One tiny true PyTorch training+export smoke; not 20-seed FANET experiment."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import torch

from fanet.training import train_torch_model
from fanet.evaluation import predict_generic
from fanet.r4_evidence import split_early_stopping_validation, export_torch_evidence, verify_evidence, evidence_location


def snap(group, t, positive):
    x = np.zeros((6, 4), dtype=np.float32)
    x[:, 0] = 1 if positive else 0
    x[:, 1] = t / 10
    adj=np.zeros((6,6), dtype=np.float32)
    for k in range(5): adj[k,k+1]=1; adj[k+1,k]=1
    return SimpleNamespace(run_id=f'run_{group}',split_group_id=group,
        time_index=t,future_time_index=t+2,target_available=True,
        n_nodes=6,node_features=x,adjacency=adj,pi=np.ones(16,dtype=np.float32)*0.2,
        beta_current=1.,beta_target=(2. if positive else 1.),frag_at_horizon=int(positive))


class ActualTorchSmoke(unittest.TestCase):
    def test_fanet_topognn_cpu_train_save_reopen_calibrate(self):
        torch.set_num_threads(1)
        train=[snap('train',t,t%2==0) for t in range(10)]
        val=[snap(f'v_{g}',t,(t+g)%2==0) for g in range(9) for t in range(6)]
        test=[snap('test',t,t%2==0) for t in range(8)]
        early,hold=split_early_stopping_validation(val,17)
        result=train_torch_model('FANET-TopoGNN',train,early,{
            'hidden_dim':8,'dropout':0.0,'learning_rate':0.001,'weight_decay':0.,
            'batch_size':4,'epochs':1,'patience':1},16,seed=17)
        self.assertIsInstance(result.model,torch.nn.Module)
        vp,_,_,aligned_val,_=predict_generic(result,hold)
        tp,_,_,aligned_test,_=predict_generic(result,test)
        with tempfile.TemporaryDirectory() as tmp:
            manifest=export_torch_evidence(seed_dir=Path(tmp),seed=17,task_name='FANET-TopoGNN',
                model=result.model, train_cfg={'epochs':1},config_sha256='f'*64,horizon_steps=2,
                validation_snapshots=aligned_val,validation_beta_hat=vp,
                test_snapshots=aligned_test,test_beta_hat=tp,
                early_stopping_group_ids=sorted({s.split_group_id for s in early}))
            self.assertTrue(verify_evidence(Path(tmp),'FANET-TopoGNN','f'*64))
            target=evidence_location(Path(tmp),'FANET-TopoGNN')
            weights=torch.load(target/'checkpoint_state_dict.pt',weights_only=True,map_location='cpu')
            self.assertTrue(weights)
            self.assertTrue(all(torch.isfinite(w).all() for w in weights.values()))
            with (target/'test.csv').open(newline='') as f: rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),len(test))
            self.assertIn('threshold_selection',manifest)

if __name__=='__main__': unittest.main()
