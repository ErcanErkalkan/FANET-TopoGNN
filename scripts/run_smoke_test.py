from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "smoke_30s.json"


def main() -> int:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "main.py"), "--config", str(CONFIG)],
        cwd=ROOT,
        check=False,
    )
    if completed.returncode:
        return completed.returncode

    output_dir = ROOT / "outputs" / "smoke_30s"
    summary_path = output_dir / "summary.json"
    metrics_path = output_dir / "metrics_overall.csv"
    if not summary_path.exists() or not metrics_path.exists():
        raise FileNotFoundError("Smoke run did not create summary.json and metrics_overall.csv")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    required = {"model_backend", "torch_available", "torch_version", "device", "surrogate_used"}
    missing = required.difference(summary)
    if missing:
        raise KeyError(f"Smoke summary is missing backend metadata: {sorted(missing)}")
    print(f"Smoke test passed: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
