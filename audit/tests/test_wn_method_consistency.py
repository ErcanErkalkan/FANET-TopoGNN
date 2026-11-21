"""Verify frozen model and radio assumptions against recorded configurations."""
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CFG = json.loads((ROOT/'core/configs/r7_full_20seed_isolated.json').read_text())
SIM = CFG['sim']
MODEL = (ROOT/'core/fanet/models.py').read_text()
DATASET = (ROOT/'core/fanet/dataset.py').read_text()
RADIO = (ROOT/'core/fanet/radio.py').read_text()
TRAINING = (ROOT/'core/fanet/training.py').read_text()

class MethodConsistency(unittest.TestCase):
    def test_frozen_scenario_parameters(self):
        self.assertEqual((SIM['area_size'],SIM['time_steps'],SIM['dt']), (1000.0,200,0.1))
        self.assertEqual(SIM['swarm_sizes'], [10,20,30])
        self.assertEqual(SIM['mobility_models'], ['rwp','gm','mission'])
        self.assertEqual(SIM['link_model'], 'physical')
        self.assertEqual(SIM['graph_policies'], ['fixed','adaptive'])
        self.assertEqual(SIM['base_radius'], 420.0)
        self.assertEqual(SIM['adaptive_scale'], 1.35)
        self.assertEqual(SIM['density_threshold'], 2e-05)
        self.assertEqual(SIM['speed_range'], [10.0,30.0])
        self.assertEqual(SIM['gauss_markov_alpha'],0.8)
        self.assertEqual(SIM['mission_regions'],3)
        self.assertEqual((SIM['pi_resolution'], SIM['forecast_horizon_steps']), (16,6))
    def test_radio_scenarios(self):
        scenarios = {r['name']:r for r in SIM['radio_scenarios']}
        self.assertEqual(set(scenarios), {'nominal','degraded'})
        self.assertEqual(tuple(scenarios['nominal'][k] for k in ('path_loss_exponent','shadowing_sigma_db','receiver_sensitivity_dbm')), (2.35,3.,-96.))
        self.assertEqual(tuple(scenarios['degraded'][k] for k in ('path_loss_exponent','shadowing_sigma_db','receiver_sensitivity_dbm')), (2.65,4.5,-92.))
        self.assertIn('directed & directed.T',RADIO)
        self.assertIn('density(points) > density_threshold', DATASET)
        self.assertIn('build_fixed_adaptive_adjacencies', DATASET)
    def test_model_is_static_graph_and_fused_geometry(self):
        beginning = MODEL.index('class FANETTopoGNN(nn.Module)')
        end = MODEL.index('class FANETTopoGNNConcat',beginning)
        x = MODEL[beginning:end]
        for t in ('GCNEncoder', 'pi_net', 'Sigmoid', 'gate * g + (1.0 - gate) * p'):
            self.assertIn(t,x)
        self.assertIn("elif model_name == \"FANET-TopoGNN\"", TRAINING)
        self.assertIn('SnapshotDataset(train_data)',TRAINING)

if __name__ == '__main__':
    unittest.main()
