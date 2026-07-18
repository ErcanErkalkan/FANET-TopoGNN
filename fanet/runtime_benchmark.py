from __future__ import annotations

import json
import os
import platform
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


STAGES = (
    "pairwise_distance",
    "radio_link_graph",
    "graph_statistics",
    "persistence_image",
    "feature_assembly",
    "count_inference",
    "risk_inference",
    "calibration",
    "controller_decision",
)
TOTAL_STAGE = "total_host_loop"


def summarize_latency_samples(samples: pd.DataFrame) -> pd.DataFrame:
    """Summarise timed (non-warm-up) samples without treating warm-up as evidence."""
    required = {"model", "thread_mode", "n_nodes", "stage", "latency_ms", "phase"}
    missing = required - set(samples.columns)
    if missing:
        raise ValueError(f"latency samples missing columns: {sorted(missing)}")
    timed = samples.loc[samples["phase"].eq("timed")].copy()
    keys = ["model", "thread_mode", "n_nodes", "stage"]
    rows = []
    for values, group in timed.groupby(keys, sort=True):
        latency = group["latency_ms"].to_numpy(dtype=float)
        rows.append(
            {
                **dict(zip(keys, values)),
                "mean_ms": float(np.mean(latency)),
                "std_ms": float(np.std(latency, ddof=1)) if len(latency) > 1 else 0.0,
                "p50_ms": float(np.quantile(latency, 0.50)),
                "p90_ms": float(np.quantile(latency, 0.90)),
                "p95_ms": float(np.quantile(latency, 0.95)),
                "p99_ms": float(np.quantile(latency, 0.99)),
                "sample_count": int(len(latency)),
            }
        )
    return pd.DataFrame(rows)


def validate_percentile_order(summary: pd.DataFrame) -> None:
    columns = ["p50_ms", "p90_ms", "p95_ms", "p99_ms"]
    if not set(columns).issubset(summary.columns):
        raise ValueError("summary lacks percentile columns")
    values = summary[columns].to_numpy(dtype=float)
    if np.any(np.diff(values, axis=1) < -1e-12):
        raise ValueError("latency percentiles are not monotonically ordered")


def validate_stage_totals(samples: pd.DataFrame, *, atol_ms: float = 0.05) -> None:
    """Check that total_host_loop is the measured sum of its exclusive stages."""
    keys = ["model", "thread_mode", "n_nodes", "phase", "iteration"]
    pivot = samples.pivot_table(index=keys, columns="stage", values="latency_ms", aggfunc="first")
    missing = set(STAGES + (TOTAL_STAGE,)) - set(pivot.columns)
    if missing:
        raise ValueError(f"stage samples missing: {sorted(missing)}")
    residual = np.abs(pivot[TOTAL_STAGE] - pivot[list(STAGES)].sum(axis=1))
    if bool((residual > atol_ms).any()):
        raise ValueError(f"host-loop stage sum mismatch; max error={float(residual.max()):.6f} ms")


def thread_limit_context(mode: str):
    if mode == "default":
        return nullcontext()
    if mode != "single":
        raise ValueError(f"unknown thread mode: {mode}")
    try:
        from threadpoolctl import threadpool_limits

        return threadpool_limits(limits=1)
    except ImportError:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        return nullcontext()


def package_versions(names: Iterable[str]) -> dict[str, str | None]:
    from importlib import metadata

    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def environment_manifest() -> dict:
    memory_bytes = None
    cpu_physical = None
    cpu_logical = os.cpu_count()
    peak_rss_source = "unavailable"
    try:
        import psutil

        memory_bytes = int(psutil.virtual_memory().total)
        cpu_physical = psutil.cpu_count(logical=False)
        cpu_logical = psutil.cpu_count(logical=True)
        peak_rss_source = "psutil.Process.memory_info().rss sampled around each benchmark block"
    except ImportError:
        pass
    gpu = {"available": False, "devices": []}
    try:
        import torch

        gpu["available"] = bool(torch.cuda.is_available())
        if gpu["available"]:
            gpu["devices"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    except ImportError:
        gpu["reason"] = "PyTorch is not installed"
    return {
        "platform": platform.platform(),
        "os": {"system": platform.system(), "release": platform.release(), "version": platform.version()},
        "cpu": {
            "model": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER") or "unavailable",
            "physical_cores": cpu_physical,
            "logical_cores": cpu_logical,
        },
        "ram_bytes": memory_bytes,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable_name": Path(sys.executable).name,
        },
        "packages": package_versions(
            ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "torch", "torch-geometric", "joblib", "psutil", "threadpoolctl", "onnx", "onnxruntime"]
        ),
        "gpu": gpu,
        "peak_rss_measurement": peak_rss_source,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
