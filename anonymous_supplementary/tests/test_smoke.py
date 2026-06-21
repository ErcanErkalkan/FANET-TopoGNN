from __future__ import annotations

import json
from pathlib import Path

from fanet import load_config, run_experiment


ROOT = Path(__file__).resolve().parents[1]


def test_smoke_profile_runs_and_writes_outputs(tmp_path: Path) -> None:
    raw = json.loads((ROOT / "configs" / "smoke_30s.json").read_text(encoding="utf-8"))
    raw["output_dir"] = str(tmp_path / "smoke-output")
    config_path = tmp_path / "smoke.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    summary = run_experiment(load_config(config_path))
    output_dir = Path(raw["output_dir"])

    assert summary["n_seeds"] == 1
    assert output_dir.is_dir()
    assert (output_dir / "summary.json").is_file()
    assert (output_dir / "metrics_overall.csv").is_file()
    assert (output_dir / "risk_threshold_sensitivity.csv").is_file()
    for field in ["model_backend", "torch_available", "torch_version", "device", "surrogate_used"]:
        assert field in summary
