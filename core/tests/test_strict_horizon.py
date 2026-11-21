"""Focused regression tests for the 2026-11-03 forecast-label patch.

Run: python -m unittest discover -s tests -p test_strict_horizon.py -v
"""
from __future__ import annotations

from dataclasses import replace
import unittest

from fanet.dataset import (
    _raw_run,
    build_dataset,
    relabel_forecast_horizon,
    simulate_run,
    train_val_test_split,
)


class StrictHorizonTests(unittest.TestCase):
    @staticmethod
    def config() -> dict:
        return {
            "area_size": 300.0,
            "time_steps": 8,
            "dt": 0.1,
            "swarm_sizes": [6],
            "runs_per_setting": 3,
            "mobility_models": ["rwp"],
            "base_radius": 90.0,
            "graph_policies": ["fixed", "adaptive"],
            "link_model": "radius",
            "adaptive_scale": 1.4,
            "density_threshold": 0.00015,
            "speed_range": [5.0, 12.0],
            "gauss_markov_alpha": 0.8,
            "gauss_markov_mu": 8.0,
            "gauss_markov_sigma": 2.0,
            "mission_regions": 2,
            "pi_resolution": 8,
            "pi_sigma": 10.0,
            "pi_max_radius": 180.0,
            "forecast_horizon_steps": 2,
            "radio_scenarios": [{"name": "normal"}, {"name": "weaker", "shadowing_sigma_db": 2.0}],
        }

    def test_simulated_horizon_excludes_terminal_frames_without_future(self):
        cfg = self.config()
        raw = _raw_run({**cfg, "graph_policy": "fixed", "radio_scenario": "normal"}, "rwp", 6, 31)
        snaps = simulate_run({**cfg, "graph_policy": "fixed", "radio_scenario": "normal"}, "rwp", 6, 31)
        self.assertEqual(len(snaps), 6)
        self.assertEqual([s.time_index for s in snaps], list(range(6)))
        self.assertEqual([s.future_time_index for s in snaps], list(range(2, 8)))
        self.assertTrue(all(s.future_time_index - s.time_index == 2 for s in snaps))
        self.assertTrue(all(s.beta_target == raw[s.time_index + 2]["beta_current"] for s in snaps))

    def test_overlong_horizon_cannot_create_fictitious_labels(self):
        cfg = self.config()
        cfg["forecast_horizon_steps"] = cfg["time_steps"]
        self.assertEqual(simulate_run({**cfg, "graph_policy": "fixed"}, "rwp", 6, 33), [])

    def test_negative_horizons_rejected(self):
        cfg = self.config()
        cfg["forecast_horizon_steps"] = -1
        with self.assertRaises(ValueError):
            simulate_run({**cfg, "graph_policy": "fixed"}, "rwp", 6, 31)
        with self.assertRaises(ValueError):
            relabel_forecast_horizon([], -2)

    def test_relabel_uses_exact_future_time_and_discards_ineligible_tail(self):
        cfg = self.config()
        snaps = simulate_run({**cfg, "graph_policy": "fixed"}, "rwp", 6, 36)
        relabeled = relabel_forecast_horizon(snaps, 2)
        self.assertEqual([s.time_index for s in relabeled], [0, 1, 2, 3])
        self.assertEqual([s.future_time_index for s in relabeled], [2, 3, 4, 5])
        self.assertTrue(all(s.future_time_index - s.time_index == 2 for s in relabeled))

    def test_relabel_rejects_gap_as_wrong_horizon(self):
        cfg = self.config()
        snaps = simulate_run({**cfg, "graph_policy": "fixed"}, "rwp", 6, 36)
        with_gap = [snaps[0], snaps[2], snaps[3]]
        relabeled = relabel_forecast_horizon(with_gap, 1)
        self.assertEqual([s.time_index for s in relabeled], [2])
        self.assertEqual(relabeled[0].future_time_index, 3)

    def test_relabel_rejects_duplicate_time_index(self):
        cfg = self.config()
        snaps = simulate_run({**cfg, "graph_policy": "fixed"}, "rwp", 6, 36)
        with self.assertRaises(ValueError):
            relabel_forecast_horizon([snaps[0], replace(snaps[1], time_index=0)], 1)

    def test_split_groups_never_cross_train_val_test_for_variants(self):
        cfg = self.config()
        snaps = build_dataset(cfg, seed=40)
        train, val, test, frame = train_val_test_split(
            snaps, split_seed=2026,
            stratify_by=("mobility", "graph_policy", "radio_scenario"),
            return_mapping=True,
        )
        self.assertTrue(train and val and test)
        self.assertEqual(frame.groupby("split_group_id")["split"].nunique().max(), 1)
        self.assertEqual(set(frame["split"]), {"train", "val", "test"})
        self.assertTrue(all(s.future_time_index - s.time_index == 2 for s in snaps))
        self.assertEqual(len(snaps), 3 * 2 * 2 * 6)  # runs × graph variants × radio variants × eligible frames


if __name__ == "__main__":
    unittest.main()
