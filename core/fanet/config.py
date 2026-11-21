from __future__ import annotations

import json
from copy import deepcopy
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


def _deep_merge(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    parent = data.get("extends")
    if parent:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = config_path.parent / parent_path
        parent_data = load_config(parent_path).raw
        data = _deep_merge(parent_data, data)
    return ExperimentConfig(data)
