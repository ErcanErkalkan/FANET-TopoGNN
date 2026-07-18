from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DECISION_PATH = ROOT / "docs/eaai_revision/scientific_positioning_decision.json"


def _decision() -> dict:
    return json.loads(DECISION_PATH.read_text(encoding="utf-8"))


def test_route_b_is_tied_to_locked_confirmatory_primary_result() -> None:
    decision = _decision()
    paired = pd.read_csv(ROOT / "outputs/eaai_locked_confirmatory/paired_tests.csv")
    row = paired.loc[
        paired["candidate"].eq("Source-Gated Kinetic-TopoGuard")
        & paired["reference"].eq("Current-state ExtraTrees")
        & paired["metric"].eq("Alert_Event_F1")
    ].iloc[0]
    gate = decision["route_a_gate"]["locked_event_f1_superiority"]

    assert decision["selected_route"] == "ROUTE_B_BENCHMARK_STRESS_TEST"
    assert gate["met"] is False
    assert np.isclose(gate["paired_mean_difference"], row["paired_mean_difference"])
    assert np.allclose(
        gate["bootstrap_ci95"],
        [row["bootstrap_ci95_low"], row["bootstrap_ci95_high"]],
    )
    assert np.isclose(
        gate["holm_adjusted_permutation_pvalue"],
        row["paired_permutation_holm_pvalue"],
    )


def test_closed_loop_gate_cannot_claim_engineering_benefit() -> None:
    decision = _decision()
    paired = pd.read_csv(ROOT / "outputs/closed_loop_controller/paired_tests.csv")
    supported = paired["Engineering_Benefit_Supported"].astype(bool)
    source_gated_pdr = paired.loc[
        paired["Candidate_Policy"].eq("Source-Gated Kinetic-TopoGuard intervention")
        & paired["Metric"].eq("Packet_Delivery_Ratio")
    ]
    gate = decision["route_a_gate"]["closed_loop_engineering_benefit"]

    assert int(supported.sum()) == gate["supported_rows"] == 0
    assert len(paired) == gate["total_rows"] == 360
    assert len(source_gated_pdr) == gate["source_gated_pdr_total_cells"] == 9
    assert gate["met"] is False


def test_count_forecasting_is_not_promoted_when_residual_branch_is_unsupported() -> None:
    decision = _decision()
    residual = json.loads(
        (ROOT / "outputs/residual_branch_audit/residual_decision.json").read_text(
            encoding="utf-8"
        )
    )

    assert residual["count_branch_supported"] is False
    assert decision["route_b_triggers"]["residual_count_branch_unsupported"] is True
    assert decision["count_forecasting_primary_contribution"] is False


def test_method_centered_title_is_rejected() -> None:
    decision = _decision()

    assert decision["current_title_can_be_retained"] is False
    assert decision["remove_topological_artificial_intelligence_from_title"] is True
    assert "Benchmark" in decision["title_recommendation"]
