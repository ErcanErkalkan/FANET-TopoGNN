from __future__ import annotations

from pathlib import Path

import pytest

from fanet.external_claims import lint_external_claim_path, lint_external_claims


@pytest.mark.parametrize(
    ("text", "rule_id"),
    [
        ("The forestry dataset provides measured RF links between the UAVs.", "forestry_measured_rf"),
        ("AERPAW validates inter-UAV communication performance.", "aerpaw_inter_uav"),
        ("MILUV provides IP packet delivery ratio measurements.", "miluv_ip_pdr"),
        ("The transported WiNES 60 GHz analysis is a same-site calibration.", "transported_wines_same_site"),
        ("The counterfactual radius graph represents measured RF connectivity.", "counterfactual_radius_measured_rf"),
    ],
)
def test_forbidden_external_claims_are_rejected(text: str, rule_id: str) -> None:
    violations = lint_external_claims(text)
    assert [item["rule_id"] for item in violations] == [rule_id]


def test_explicit_claim_boundaries_are_allowed() -> None:
    text = (
        "Forestry provides measured motion without measured peer RF. "
        "AERPAW supports UAV-to-infrastructure LTE state prediction, not inter-UAV links. "
        "MILUV measures UWB topology, not IP packet delivery ratio. "
        "The transported WiNES result is not same-site calibration."
    )
    assert lint_external_claims(text) == []


def test_current_manuscript_passes_external_claim_lint() -> None:
    root = Path(__file__).resolve().parents[1]
    assert lint_external_claim_path(root / "paper/main.tex") == []
