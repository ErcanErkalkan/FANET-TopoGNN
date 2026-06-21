from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "paper_like_submission.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the strict 20-seed submission profile")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    seeds = config.get("training", {}).get("seed_list", [])
    if len(seeds) != 20 or len(set(seeds)) != 20:
        raise ValueError(f"Submission profile must contain 20 unique seeds; found {len(set(seeds))}")
    selected = config.get("training", {}).get("selected_models", [])
    required = {"union_find", "shallow", "kinetic_topoguard"}
    if set(selected) != required:
        raise ValueError(
            "The confirmatory profile must contain exactly union_find, shallow, and kinetic_topoguard; "
            f"found {selected}"
        )
    command = [sys.executable, str(ROOT / "main.py"), "--config", str(config_path)]
    if args.resume:
        command.append("--resume")
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
