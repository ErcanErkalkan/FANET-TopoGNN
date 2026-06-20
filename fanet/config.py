from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExperimentConfig:
    raw: dict

    @property
    def experiment_name(self) -> str:
        return self.raw["experiment_name"]

    @property
    def output_dir(self) -> Path:
        return Path(self.raw["output_dir"])

    @property
    def sim(self) -> dict:
        return self.raw["sim"]

    @property
    def training(self) -> dict:
        return self.raw["training"]

    @property
    def evaluation(self) -> dict:
        return self.raw["evaluation"]


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return ExperimentConfig(data)
