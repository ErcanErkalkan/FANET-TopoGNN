from __future__ import annotations

import unittest

import numpy as np

from fanet.dataset import Snapshot, build_dataset, to_frame, train_val_test_split
from fanet.evaluation import evaluate_predictions, event_warning_leads, run_network_controller
from fanet.pyg_utils import snapshot_to_pyg_data, torch_geometric_available
from fanet.training import fit_kinetic_topoguard, kinetic_topoguard_feature_vector


class CorePipelineTests(unittest.TestCase):
    def _small_config(self) -> dict:
        return {
            "area_size": 300.0,
            "time_steps": 8,
            "dt": 0.1,
            "swarm_sizes": [6],
            "runs_per_setting": 3,
            "mobility_models": ["rwp", "gm", "mission"],
            "base_radius": 90.0,
            "graph_policies": ["fixed", "adaptive"],
            "link_model": "physical",
            "adaptive_scale": 1.4,
            "density_threshold": 0.00015,
            "split_seed": 2026,
            "speed_range": [5.0, 12.0],
            "gauss_markov_alpha": 0.8,
            "gauss_markov_mu": 8.0,
            "gauss_markov_sigma": 2.0,
            "mission_regions": 2,
            "mission_tracking_noise_m": 1.0,
            "mission_heading_noise_rad": 0.02,
            "mission_speed_jitter": 0.04,
            "physical_layer": {
                "path_loss_exponent": 2.2,
                "shadowing_sigma_db": 1.0,
                "nakagami_m": 1.0,
                "receiver_sensitivity_dbm": -92.0,
                "apply_radius_cap": True,
            },
            "radio_scenarios": [
                {"name": "low_shadow", "shadowing_sigma_db": 0.5, "receiver_sensitivity_dbm": -94.0},
                {"name": "high_shadow", "shadowing_sigma_db": 2.0, "receiver_sensitivity_dbm": -90.0},
            ],
            "pi_resolution": 8,
            "pi_sigma": 10.0,
            "pi_max_radius": 180.0,
            "forecast_horizon_steps": 2,
        }

    def test_dataset_contains_paper_level_graph_fields(self) -> None:
        snapshots = build_dataset(self._small_config(), seed=7)
        self.assertTrue(snapshots)
        first = snapshots[0]
        self.assertEqual(first.link_model, "physical")
        self.assertIn(first.graph_policy, {"fixed", "adaptive"})
        self.assertIn(first.radio_scenario, {"low_shadow", "high_shadow"})
        self.assertEqual(first.adjacency.shape, first.adjacency_fixed.shape)
        self.assertEqual(first.adjacency.shape, first.adjacency_adaptive.shape)
        self.assertGreaterEqual(first.beta_fixed, 1.0)
        self.assertGreaterEqual(first.beta_adaptive, 1.0)
        frame = to_frame(snapshots)
        self.assertEqual(set(frame["graph_policy"]), {"fixed", "adaptive"})
        self.assertEqual(set(frame["radio_scenario"]), {"low_shadow", "high_shadow"})
        self.assertIn("frag_at_horizon", frame.columns)

    def test_split_is_run_wise_and_non_empty(self) -> None:
        snapshots = build_dataset(self._small_config(), seed=11)
        train, val, test, split_frame = train_val_test_split(
            snapshots,
            split_seed=2026,
            stratify_by=("mobility", "graph_policy", "radio_scenario"),
            return_mapping=True,
        )
        self.assertTrue(train)
        self.assertTrue(val)
        self.assertTrue(test)
        self.assertEqual(set(split_frame["split"]), {"train", "val", "test"})
        self.assertFalse(split_frame.duplicated("run_id").any())
        self.assertTrue((split_frame.groupby("split_group_id")["split"].nunique() == 1).all())
        self.assertIn("radio_scenario", split_frame.columns)

    def test_link_flip_lead_times_are_computed_from_edges(self) -> None:
        snapshots = build_dataset(self._small_config(), seed=13)
        run = [snap for snap in snapshots if snap.run_id == snapshots[0].run_id]
        scores = np.ones(len(run), dtype=float)
        leads, normalised = event_warning_leads(run, scores, dt=0.1, horizon_steps=2, risk_threshold=0.5)
        self.assertEqual(len(leads), len(normalised))

    def test_lead_time_alignment_uses_matching_risk_scores(self) -> None:
        connected = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        disconnected = np.zeros((2, 2), dtype=np.float32)

        def make_snapshot(run_id: str, time_index: int, adjacency: np.ndarray) -> Snapshot:
            beta = 1.0 if adjacency.sum() > 0 else 2.0
            return Snapshot(
                run_id=run_id,
                split_group_id=run_id,
                time_index=time_index,
                mobility="unit",
                n_nodes=2,
                positions=np.zeros((2, 2), dtype=np.float32),
                velocities=np.zeros((2, 2), dtype=np.float32),
                node_features=np.zeros((2, 4), dtype=np.float32),
                adjacency=adjacency,
                adjacency_fixed=adjacency,
                adjacency_adaptive=adjacency,
                pi=np.zeros(4, dtype=np.float32),
                stats=np.zeros(12, dtype=np.float32),
                beta_current=beta,
                beta_target=beta,
                beta_fixed=beta,
                beta_adaptive=beta,
                radius=1.0,
                radius_fixed=1.0,
                radius_adaptive=1.0,
                edge_count_fixed=int(adjacency.sum() // 2),
                edge_count_adaptive=int(adjacency.sum() // 2),
                link_model="unit",
                graph_policy="fixed",
                radio_scenario="unit",
                is_connected=int(beta == 1.0),
                future_time_index=time_index,
                frag_at_horizon=int(beta > 1.0),
            )

        run_b = [make_snapshot("run_b", idx, disconnected) for idx in range(4)]
        run_a = [
            make_snapshot("run_a", 0, connected),
            make_snapshot("run_a", 1, connected),
            make_snapshot("run_a", 2, disconnected),
            make_snapshot("run_a", 3, disconnected),
        ]
        snapshots = run_b + run_a
        preds = np.asarray([snap.beta_current for snap in snapshots], dtype=float)
        risk_scores = (preds > 1.0).astype(float)
        _, leads, _ = evaluate_predictions(
            "alignment-check",
            snapshots,
            preds,
            risk_scores,
            np.zeros(len(snapshots), dtype=float),
            dt=0.1,
            bootstrap_rounds=1,
            horizon_steps=2,
            risk_threshold=0.5,
        )
        self.assertTrue(leads)
        self.assertEqual(max(leads), 0.0)

    def test_network_controller_reports_routing_and_dtn_metrics(self) -> None:
        snapshots = build_dataset(self._small_config(), seed=17)
        run = [snap for snap in snapshots if snap.run_id == snapshots[0].run_id]
        preds = np.ones(len(run), dtype=float) * 2.0
        scores = np.ones(len(run), dtype=float)
        metrics = run_network_controller(run, preds, scores, boost=1.2, risk_threshold=0.5)
        self.assertIn("Proactive reroute (%)", metrics)
        self.assertIn("DTN buffered (%)", metrics)
        self.assertIn("Relay actions", metrics)
        self.assertGreaterEqual(metrics["PDR (%)"], 0.0)

    def test_network_controller_uses_snapshot_adjacency_for_delivery_metrics(self) -> None:
        adjacency = np.zeros((2, 2), dtype=np.float32)
        snap = Snapshot(
            run_id="unit_seed3",
            split_group_id="unit_seed3",
            time_index=0,
            mobility="unit",
            n_nodes=2,
            positions=np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
            velocities=np.zeros((2, 2), dtype=np.float32),
            node_features=np.zeros((2, 4), dtype=np.float32),
            adjacency=adjacency,
            adjacency_fixed=adjacency,
            adjacency_adaptive=adjacency,
            pi=np.zeros(4, dtype=np.float32),
            stats=np.zeros(17, dtype=np.float32),
            beta_current=2.0,
            beta_target=2.0,
            beta_fixed=2.0,
            beta_adaptive=2.0,
            radius=10.0,
            radius_fixed=10.0,
            radius_adaptive=10.0,
            edge_count_fixed=0,
            edge_count_adaptive=0,
            link_model="physical",
            graph_policy="fixed",
            radio_scenario="unit",
            is_connected=0,
            future_time_index=0,
            frag_at_horizon=1,
        )
        metrics = run_network_controller(
            [snap],
            np.asarray([1.0], dtype=float),
            np.asarray([0.0], dtype=float),
            boost=2.0,
            risk_threshold=0.5,
        )
        self.assertEqual(metrics["Connectivity ratio"], 0.0)
        self.assertEqual(metrics["PDR (%)"], 0.0)

    def test_kinetic_topoguard_fits_and_scores_snapshots(self) -> None:
        config = self._small_config()
        snapshots = build_dataset(config, seed=23)
        train, val, test = train_val_test_split(
            snapshots,
            split_seed=2026,
            stratify_by=("mobility", "graph_policy", "radio_scenario"),
        )
        feature, score = kinetic_topoguard_feature_vector(train[0], None, config["forecast_horizon_steps"], config["dt"])
        self.assertTrue(np.isfinite(feature).all())
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        result = fit_kinetic_topoguard(train, val, config["forecast_horizon_steps"], config["dt"], seed=23)
        preds, risk, aligned = result.model.predict_snapshots(test)
        self.assertEqual(result.model_name, "Kinetic-TopoGuard")
        self.assertEqual(len(preds), len(aligned))
        self.assertEqual(len(risk), len(aligned))
        self.assertTrue(np.isfinite(preds).all())
        self.assertTrue(((risk >= 0.0) & (risk <= 1.0)).all())

    def test_pyg_adapter_is_available_or_fails_clearly(self) -> None:
        snapshot = build_dataset(self._small_config(), seed=19)[0]
        if torch_geometric_available():
            data = snapshot_to_pyg_data(snapshot)
            self.assertEqual(data.x.shape[0], snapshot.n_nodes)
            self.assertEqual(data.edge_index.shape[0], 2)
            self.assertEqual(data.radio_scenario, snapshot.radio_scenario)
        else:
            with self.assertRaises(ModuleNotFoundError):
                snapshot_to_pyg_data(snapshot)


if __name__ == "__main__":
    unittest.main()
