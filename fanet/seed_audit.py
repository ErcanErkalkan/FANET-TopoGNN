from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any


SEED_KEY = re.compile(r"seed", re.IGNORECASE)
SEED_DIRECTORY = re.compile(r"seed_(-?\d+)$")
CONFIRMATORY_DOMAIN = "FANET-TopoGNN:EAAI:locked-confirmatory:v1"


def _json_seed_values(value: Any, key: str = "") -> set[int]:
    found: set[int] = set()
    if isinstance(value, dict):
        for child_key, child in value.items():
            found.update(_json_seed_values(child, str(child_key)))
    elif isinstance(value, list):
        for child in value:
            found.update(_json_seed_values(child, key))
    elif isinstance(value, int) and not isinstance(value, bool) and SEED_KEY.search(key):
        found.add(int(value))
    return found


def discover_prior_seeds(
    root: str | Path,
    excluded_paths: set[str] | None = None,
) -> tuple[set[int], list[dict[str, Any]]]:
    """Discover explicit seed fields/columns and ``seed_<n>`` directories."""
    root = Path(root).resolve()
    excluded = {path.replace("\\", "/") for path in (excluded_paths or set())}
    seeds: set[int] = set()
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if any(part in {".git", ".pytest_cache", "__pycache__"} for part in path.parts):
            continue
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        directory_match = SEED_DIRECTORY.fullmatch(path.name) if path.is_dir() else None
        if directory_match:
            value = int(directory_match.group(1))
            seeds.add(value)
            records.append({"path": relative, "kind": "seed_directory", "seeds": [value]})
            continue
        if not path.is_file():
            continue
        found: set[int] = set()
        if path.suffix.lower() == ".json":
            try:
                found = _json_seed_values(json.loads(path.read_text(encoding="utf-8-sig")))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
        elif path.suffix.lower() == ".csv":
            try:
                with path.open("r", encoding="utf-8-sig", newline="") as handle:
                    reader = csv.DictReader(handle)
                    seed_columns = [name for name in (reader.fieldnames or []) if SEED_KEY.search(name)]
                    if not seed_columns:
                        continue
                    for row in reader:
                        for column in seed_columns:
                            raw = row.get(column, "")
                            try:
                                number = float(raw)
                            except (TypeError, ValueError):
                                continue
                            if number.is_integer():
                                found.add(int(number))
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
        if found:
            seeds.update(found)
            records.append({"path": relative, "kind": path.suffix.lower().lstrip("."), "seeds": sorted(found)})
    return seeds, records


def deterministic_confirmatory_seeds(excluded: set[int], count: int = 20) -> list[int]:
    """Generate stable positive 31-bit seeds from a domain-separated SHA-256 stream."""
    selected: list[int] = []
    counter = 0
    while len(selected) < count:
        digest = hashlib.sha256(f"{CONFIRMATORY_DOMAIN}:{counter}".encode("utf-8")).digest()
        candidate = 100_000 + int.from_bytes(digest[:8], "big") % 2_000_000_000
        if candidate not in excluded and candidate not in selected:
            selected.append(candidate)
        counter += 1
    return selected
