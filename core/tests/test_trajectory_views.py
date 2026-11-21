"""R1: complete topology truth vs strictly forecast-eligible observations."""
from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from fanet.dataset import (
    build_dataset, build_dataset_views, relabel_forecast_horizon,
    simulate_run, to_frame, train_val_test_split, require_forecast_eligible,
)
from fanet.evaluation import alert_event_metrics_full_trajectory, run_network_controller
import test_strict_horizon as r0_tests


class CompleteTrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.cfg = r0_tests.StrictHorizonTests.config()

    def test_full_timeline_keeps_unlabeled_tail_without_fictitious_targets(self):
        views = build_dataset_views(self.cfg, seed=40)
        self.assertEqual(len(views.full_trajectory), 3 * 2 * 2 * 8)
        self.assertEqual(len(views.forecast_eligible), 3 * 2 * 2 * 6)
        self.assertEqual(views.horizon_steps, 2)
        frame = to_frame(views.full_trajectory)
        self.assertEqual(int(frame['target_available'].sum()), 3 * 2 * 2 * 6)
        invalid = [s for s in views.full_trajectory if not s.target_available]
        self.assertEqual(len(invalid), 24)
        for s in invalid:
            self.assertTrue(np.isnan(s.beta_target))
            self.assertEqual(s.frag_at_horizon, -1)
            self.assertEqual(s.future_time_index, -1)
            self.assertIn(s.time_index, {6, 7})
        self.assertEqual(len(build_dataset(self.cfg, seed=40)), len(views.forecast_eligible))

    def test_eligible_view_matches_original_default_bytewise_fields(self):
        eligible = build_dataset(self.cfg, seed=40)
        views = build_dataset_views(self.cfg, seed=40)
        self.assertEqual([(s.run_id,s.time_index,s.beta_target) for s in eligible],
                         [(s.run_id,s.time_index,s.beta_target) for s in views.forecast_eligible])
        self.assertTrue(all(np.array_equal(a.pi, b.pi) and np.array_equal(a.positions,b.positions)
                            for a,b in zip(eligible, views.forecast_eligible)))

    def test_multihorizon_relabel_recovers_terminal_observation(self):
        cfg = {**self.cfg, 'forecast_horizon_steps': 3}
        full = simulate_run({**cfg, 'graph_policy': 'fixed'}, 'rwp', 6, 35, include_unlabeled_tail=True)
        relabeled = relabel_forecast_horizon(full, 1)
        self.assertEqual([s.time_index for s in relabeled], list(range(7)))
        self.assertEqual(relabeled[-1].future_time_index, 7)
        self.assertEqual(relabeled[-1].beta_target, full[-1].beta_current)
        relabeled_full = relabel_forecast_horizon(full, 5, include_unlabeled_tail=True)
        self.assertEqual(len(relabeled_full), 8)
        self.assertEqual(sum(s.target_available for s in relabeled_full), 3)
        self.assertEqual([s.future_time_index for s in relabeled_full][-5:], [-1]*5)

    def test_no_full_trajectory_allowed_in_supervised_train_split(self):
        full = build_dataset_views(self.cfg, 40).full_trajectory
        with self.assertRaisesRegex(ValueError, 'unlabeled forecast target'):
            train_val_test_split(full, split_seed=17)
        with self.assertRaisesRegex(ValueError, 'unlabeled forecast target'):
            require_forecast_eligible(full)

    def test_event_at_terminal_time_is_counted_as_truth(self):
        cfg = {**self.cfg, 'graph_policy': 'fixed', 'radio_scenarios': []}
        full = simulate_run(cfg, 'rwp', 6, 36, include_unlabeled_tail=True)
        # Artificial trace: one fragmentation only at the final observed t=7.
        full = [replace(s, beta_current=(2.0 if s.time_index == 7 else 1.0)) for s in full]
        # Reconstruct target labels consistently from that full topology.
        full = relabel_forecast_horizon(full, 2, include_unlabeled_tail=True)
        eligible = [s for s in full if s.target_available]
        scores = np.zeros(len(eligible))
        scores[-1] = 0.9  # alert at t=5, event at t=7
        result = alert_event_metrics_full_trajectory(eligible, scores, full, 0.5, 0.1, 2)
        self.assertEqual(result['Ground_Truth_Fragmentation_Events'], 1.0)
        self.assertEqual(result['True_Alert_Events'], 1.0)
        self.assertEqual(result['Alert_Event_F1'], 1.0)

    def test_event_scores_not_silently_zipped_or_shifted(self):
        cfg = {**self.cfg, 'graph_policy': 'fixed', 'radio_scenarios': []}
        full = simulate_run(cfg, 'rwp', 6, 36, include_unlabeled_tail=True)
        eligible = [s for s in full if s.target_available]
        with self.assertRaisesRegex(ValueError, 'misaligned risk scores'):
            alert_event_metrics_full_trajectory(eligible, np.zeros(len(eligible)-1), full, 0.5, 0.1, 2)
        with self.assertRaisesRegex(ValueError, 'specified horizon'):
            alert_event_metrics_full_trajectory(eligible, np.zeros(len(eligible)), full, 0.5, 0.1, 1)
        with self.assertRaisesRegex(ValueError, 'misaligned risk scores'):
            alert_event_metrics_full_trajectory(eligible, np.full(len(eligible), np.nan), full, 0.5, 0.1, 2)

    def test_controller_rejects_implicit_tail_shift(self):
        cfg = {**self.cfg, 'graph_policy': 'fixed', 'radio_scenarios': []}
        full = simulate_run(cfg, 'rwp', 6, 36, include_unlabeled_tail=True)
        eligible = [s for s in full if s.target_available]
        with self.assertRaisesRegex(ValueError, 'align 1:1'):
            run_network_controller(full, np.ones(len(eligible)), np.zeros(len(eligible)), 1.0, 0.5)
        with self.assertRaisesRegex(ValueError, 'unlabeled full-trajectory'):
            run_network_controller(full, np.ones(len(full)), np.zeros(len(full)), 1.0, 0.5)

    def test_grouped_full_trajectory_is_split_safe_after_filter(self):
        full = build_dataset_views(self.cfg, 43)
        train,val,test,frame = train_val_test_split(full.forecast_eligible, split_seed=2026,return_mapping=True)
        self.assertTrue(train and val and test)
        self.assertEqual(frame.groupby('split_group_id')['split'].nunique().max(),1)


if __name__ == '__main__': unittest.main()
