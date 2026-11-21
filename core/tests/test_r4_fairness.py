"""R4 fairness preflight tests (synthetic trace only, not original R4 data)."""
from __future__ import annotations

from dataclasses import replace
import unittest
import numpy as np

from fanet.dataset import simulate_run, relabel_forecast_horizon
from fanet.r4_fairness import ModelSplitScores, compare_r4_models, validate_common_support
import test_strict_horizon as r0_tests


def make_run(run_seed: int, fragment_at: int) -> tuple[list, list]:
    cfg = {**r0_tests.StrictHorizonTests.config(), 'graph_policy': 'fixed', 'radio_scenarios': []}
    full = simulate_run(cfg, 'rwp', 6, run_seed, include_unlabeled_tail=True)
    full = [replace(s, beta_current=2.0 if s.time_index == fragment_at else 1.0) for s in full]
    full = relabel_forecast_horizon(full, 2, include_unlabeled_tail=True)
    return full, [s for s in full if s.target_available]


class FairR4Tests(unittest.TestCase):
    def setUp(self):
        self.vfull,self.v = make_run(21,7)
        self.tfull,self.t = make_run(22,6)
        self.vscore = np.asarray([0.1,0.1,0.1,0.1,0.1,0.9])
        self.tscore = np.asarray([0.1,0.1,0.1,0.1,0.9,0.1])

    def record(self,snaps,values,kind='native_probability'):
        return ModelSplitScores(snaps,np.asarray(values,dtype=float),kind)

    def test_reject_hard_binary_proxy_from_regression(self):
        with self.assertRaisesRegex(ValueError,'genuine risk probabilities'):
            validate_common_support({'N':self.record(self.v, self.vscore,'hard_binary_from_beta_count'),
                                     'E':self.record(self.v, self.vscore)},2)

    def test_reject_different_temporal_support(self):
        with self.assertRaisesRegex(ValueError,'support differs'):
            validate_common_support({'N':self.record(self.v[:-1],self.vscore[:-1]),
                                     'E':self.record(self.v,self.vscore)},2)

    def test_reject_duplicate_prediction_keys(self):
        with self.assertRaisesRegex(ValueError,'duplicate prediction key'):
            validate_common_support({'N':self.record(self.v+[self.v[0]],np.r_[self.vscore,[.9]]),
                                     'E':self.record(self.v,self.vscore)},2)

    def test_reject_validation_test_overlap(self):
        other_val = self.record(self.v,self.vscore)
        with self.assertRaisesRegex(ValueError,'split_group_id leakage'):
            compare_r4_models({'N':other_val,'E':other_val},
                              {'N':other_val,'E':other_val},self.vfull,self.vfull,dt=.1,horizon_steps=2)

    def test_threshold_selected_from_validation_never_test(self):
        v={'A':self.record(self.v,self.vscore), 'B':self.record(self.v,self.vscore)}
        t={'A':self.record(self.t,self.tscore), 'B':self.record(self.t,np.zeros_like(self.tscore))}
        result=compare_r4_models(v,t,self.vfull,self.tfull,dt=.1,horizon_steps=2)
        self.assertEqual(len(result),2)
        self.assertEqual(result[0]['Selected_Threshold'],result[1]['Selected_Threshold'])
        self.assertGreater(result[0]['Test_Alert_Event_F1'],result[1]['Test_Alert_Event_F1'])
        self.assertEqual(result[0]['Validation_Decision_Count'],6)

    def test_label_conflict_fails(self):
        wrong=[replace(s,frag_at_horizon=1-s.frag_at_horizon) for s in self.v]
        with self.assertRaisesRegex(ValueError,'inconsistent target'):
            validate_common_support({'A':self.record(self.v,self.vscore),
                                     'B':self.record(wrong,self.vscore)},2)


if __name__ == '__main__': unittest.main()
