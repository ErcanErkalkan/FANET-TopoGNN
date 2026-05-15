from __future__ import annotations

import argparse
from pathlib import Path

from fanet_q1 import load_config, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FANET-TopoGNN Q1 reproducibility pipeline")
    parser.add_argument("--config", required=True, help="Path to JSON config")
    parser.add_argument("--resume", action="store_true", help="Resume from completed seeds and cached per-model artifacts when available")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    summary = run_experiment(config, resume=args.resume)
    print(summary)


if __name__ == "__main__":
    main()
