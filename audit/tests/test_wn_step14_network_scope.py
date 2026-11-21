"""Step 14: tests source-to-claim boundaries for graph vs network service."""
import csv
import json
import re
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RADIO=(ROOT/'core/fanet/radio.py').read_text(encoding='utf-8')
DATA=(ROOT/'core/fanet/dataset.py').read_text(encoding='utf-8')
CFG=json.loads((ROOT/'core/configs/r7_full_20seed_isolated.json').read_text(encoding='utf-8'))['sim']
FROZEN=(ROOT/'audit/r9_10_fixed_threshold_audit.py').read_text(encoding='utf-8')

class NetworkServiceScope(unittest.TestCase):
    def test_actual_r9_link_mode_not_optional_sinr(self):
        self.assertEqual(CFG['link_model'],'physical')
        self.assertIn('if link_model in {"physical", "path_loss", "power"}',RADIO)
        self.assertIn('return physical_power_adjacency(',RADIO)
        self.assertIn('if link_model in {"sinr", "interference"}',RADIO)
    def test_bidirectional_threshold_and_cap(self):
        self.assertIn('rx_power >= sensitivity',RADIO)
        self.assertIn('distances <= radius',RADIO)
        self.assertIn('directed & directed.T',RADIO)
        self.assertIn('build_fixed_adaptive_adjacencies(',DATA)
    def test_source_uses_temporal_propagation_proxy(self):
        for token in ('path_loss_exponent','shadowing_temporal_rho','fading_temporal_rho','rayleigh','receiver_sensitivity_dbm'):
            self.assertIn(token,RADIO)
    def test_no_packet_simulation_called_by_r9_replay(self):
        self.assertNotIn('simulate_packet_tick(',FROZEN)
        self.assertNotIn('packet_sim',FROZEN)
        self.assertTrue((ROOT/'core/fanet/packet_sim.py').exists())
    def test_reported_csv_fields_are_graph_and_scores(self):
        truth=next((ROOT/'audit/reference_linux').glob('*/evaluation_full_truth.csv'))
        scores=next((ROOT/'audit/reference_linux').glob('*/evaluation_scores.csv'))
        with truth.open(newline='') as f:t=set(next(csv.reader(f)))
        with scores.open(newline='') as f:s=set(next(csv.reader(f)))
        self.assertTrue({'beta_current','time_index'}<=t)
        self.assertFalse({'pdr','throughput','end_to_end_delay','packet_delivered'}&(t|s))

if __name__ == '__main__': unittest.main()
