from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[1]
STAGING = ROOT / ".anonymous_supplementary_staging"
FINAL = ROOT / "anonymous_supplementary"
BACKUP = ROOT / ".anonymous_supplementary_backup"
ZIP_PATH = ROOT / "anonymous_supplementary.zip"
ZIP_TEMP = ROOT / ".anonymous_supplementary.zip.tmp"

ROOT_FILES = (
    "main.py",
    "pyproject.toml",
    "pytest.ini",
    "requirements.txt",
    "requirements-deep.txt",
    "requirements-pyg.txt",
    "requirements-external.txt",
    "environment.yml",
    "ANONYMOUS_SUPPLEMENTARY_README.md",
)

CONFIG_FILES = (
    "paper_like.json",
    "paper_like_compact.json",
    "paper_like_submission.json",
    "publication_compact.json",
    "publication_compact_provenance.json",
    "publication_neural_extension.json",
    "quickstart.json",
    "smoke_30s.json",
)

SCRIPT_FILES = (
    "run_paper_like.ps1",
    "run_quickstart.ps1",
    "setup_conda_env.ps1",
    "setup_deep_venv.ps1",
    "setup_pyg_venv.ps1",
    "setup_scientific_venv.ps1",
    "run_smoke_test.py",
    "run_submission_20seed.py",
    "download_external_validation.ps1",
    "download_aerpaw_cellular_validation.ps1",
    "inspect_rosbag_topics.py",
    "extract_forestry_trace.py",
    "run_external_validation.py",
    "run_aerpaw_cellular_validation.py",
    "run_uav_to_uav_mmwave_validation.py",
    "download_miluv_validation.py",
    "run_miluv_validation.py",
    "run_horizon_sweep.py",
    "run_factorial_feature_ablation.py",
    "run_packet_level_controller_validation.py",
    "benchmark_end_to_end_latency.py",
    "select_operating_point.py",
    "build_digital_twin_dashboard.py",
    "build_neural_seed_extension.py",
    "generate_evidence_tables.py",
    "audit_submission_readiness.py",
    "build_anonymous_supplementary.py",
)

TEXT_SUFFIXES = {
    ".cfg",
    ".csv",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".tex",
    ".toml",
    ".txt",
    ".yml",
    ".yaml",
}

IDENTITY_PATTERNS = (
    re.compile(r"ercan", re.IGNORECASE),
    re.compile(r"erkalkan", re.IGNORECASE),
    re.compile(r"marmara", re.IGNORECASE),
    re.compile(r"20226053", re.IGNORECASE),
    re.compile(r"0000-0001-9259-7112", re.IGNORECASE),
    re.compile(r"mehmet\s+gen(?:c|ç)", re.IGNORECASE),
    re.compile(r"kartal", re.IGNORECASE),
)

ANONYMOUS_LICENSE = """Academic License Agreement

Copyright (c) 2025-2026 Anonymous author(s)

Permission is hereby granted, free of charge, to any individual or institution
obtaining a copy of this software and associated documentation files (the
\"Software\"), to use the Software strictly for academic research and
educational purposes, including reproducing results, conducting scientific
comparisons, citing the Software, and teaching or demonstration.

Commercial use, distribution, sublicensing, or incorporation into proprietary
products is prohibited without prior written permission from the copyright
holder.

THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
COPYRIGHT HOLDER BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.

Author-identifying contact information has been removed from this
double-anonymous review copy.
"""


def _assert_workspace_path(path: Path) -> None:
    resolved_root = ROOT.resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(resolved_root) or resolved == resolved_root:
        raise RuntimeError(f"Refusing filesystem operation outside workspace: {resolved}")


def _remove_tree(path: Path) -> None:
    _assert_workspace_path(path)
    if path.exists():
        shutil.rmtree(path)


def _copy_file(relative: str) -> None:
    source = ROOT / relative
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = STAGING / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_tree(relative: str, *, excluded_names: set[str] | None = None) -> None:
    excluded_names = excluded_names or set()
    source_root = ROOT / relative
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    for source in sorted(source_root.rglob("*")):
        if not source.is_file():
            continue
        rel = source.relative_to(ROOT)
        parts = set(rel.parts)
        if parts & {"__pycache__", ".pytest_cache", ".resume"}:
            continue
        if source.name in excluded_names:
            continue
        if source.suffix.lower() in {".pyc", ".pyo", ".bag", ".part", ".log"}:
            continue
        destination = STAGING / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _write_generated_files() -> None:
    readme = STAGING / "ANONYMOUS_SUPPLEMENTARY_README.md"
    shutil.copy2(readme, STAGING / "README.md")
    (STAGING / "LICENSE").write_text(ANONYMOUS_LICENSE, encoding="utf-8")


def _validate_evidence() -> None:
    seeds = sorted((STAGING / "outputs" / "paper_like_submission" / "per_seed").glob("seed_*"))
    if len(seeds) != 20:
        raise RuntimeError(f"Expected 20 confirmatory seed directories, found {len(seeds)}")

    summary = (STAGING / "outputs" / "publication_compact" / "summary.json").read_text(
        encoding="utf-8"
    )
    if '"torch_available": true' not in summary:
        raise RuntimeError("Compact summary does not verify torch_available=true")
    if '"surrogate_used": false' not in summary:
        raise RuntimeError("Compact summary does not verify surrogate_used=false")
    if '"cache_version": "kinetic_topoguard_v6_fragmentation_events_correlated_radio"' not in summary:
        raise RuntimeError("Compact summary is stale or uses the pre-event-metric cache")
    summary_payload = json.loads(summary)
    neural_tasks = {
        "GCN",
        "GAT",
        "GraphSAGE",
        "PI+MLP",
        "FANET-TopoGNN",
        "FANET-TopoGNN (concat)",
        "tgcn:5",
        "stgcn:5",
        "tgn:5",
    }
    wrong_backends = {
        task: summary_payload.get("model_backend", {}).get(task)
        for task in neural_tasks
        if summary_payload.get("model_backend", {}).get(task) != "pytorch"
    }
    if wrong_backends:
        raise RuntimeError(f"Compact neural backend mismatch: {wrong_backends}")

    predictions = STAGING / "outputs" / "external_validation" / "external_predictions.csv"
    trace = STAGING / "data" / "external_validation" / "derived" / "forestry_multidrone_trace.csv"
    if not predictions.is_file() or not trace.is_file():
        raise RuntimeError("External transfer predictions or derived trace are missing")

    aerpaw = STAGING / "outputs" / "aerpaw_cellular_validation"
    required_aerpaw = [
        aerpaw / "aerpaw_cellular_protocol.json",
        aerpaw / "aerpaw_lte_availability_metrics.csv",
        aerpaw / "aerpaw_throughput_metrics.csv",
        aerpaw / "aerpaw_cellular_table.tex",
        aerpaw / "aerpaw_cellular_validation.pdf",
    ]
    missing_aerpaw = [path for path in required_aerpaw if not path.is_file()]
    if missing_aerpaw:
        raise RuntimeError("AERPAW cellular evidence is missing: " + ", ".join(str(path) for path in missing_aerpaw))

    a2a = STAGING / "outputs" / "uav_to_uav_mmwave_validation"
    required_a2a = [
        a2a / "uav_to_uav_mmwave_protocol.json",
        a2a / "uav_to_uav_link_model_metrics.csv",
        a2a / "uav_to_uav_forestry_beta0_summary.csv",
        a2a / "uav_to_uav_mmwave_table.tex",
        a2a / "uav_to_uav_mmwave_validation.pdf",
    ]
    missing_a2a = [path for path in required_a2a if not path.is_file()]
    if missing_a2a:
        raise RuntimeError("UAV-to-UAV mmWave evidence is missing: " + ", ".join(str(path) for path in missing_a2a))

    miluv = STAGING / "outputs" / "miluv_validation"
    required_miluv = [
        miluv / "miluv_protocol.json",
        miluv / "miluv_metrics_per_seed.csv",
        miluv / "miluv_metrics_summary.csv",
        miluv / "miluv_measured_topology_trace.csv",
        STAGING / "data" / "external_validation" / "raw" / "miluv" / "cirObstacles_3_random_0" / "manifest.json",
    ]
    missing_miluv = [path for path in required_miluv if not path.is_file()]
    if missing_miluv:
        raise RuntimeError("MILUV evidence is missing: " + ", ".join(str(path) for path in missing_miluv))

    experiment_outputs = {
        "horizon sweep": [
            "outputs/horizon_sweep/horizon_sweep_protocol.json",
            "outputs/horizon_sweep/horizon_sweep_summary.csv",
        ],
        "factorial ablation": [
            "outputs/factorial_feature_ablation/factorial_ablation_protocol.json",
            "outputs/factorial_feature_ablation/factorial_ablation_summary.csv",
        ],
        "packet validation": [
            "outputs/packet_level_controller/packet_level_protocol.json",
            "outputs/packet_level_controller/packet_metrics_summary.csv",
        ],
        "latency benchmark": [
            "outputs/end_to_end_latency/latency_protocol.json",
            "outputs/end_to_end_latency/latency_summary.csv",
        ],
    }
    for label, relative_paths in experiment_outputs.items():
        missing = [path for path in relative_paths if not (STAGING / path).is_file()]
        if missing:
            raise RuntimeError(f"{label} evidence is missing: {missing}")

    operating = STAGING / "outputs" / "operating_point"
    required_operating = [
        operating / "operating_point_protocol.json",
        operating / "operating_point_summary.csv",
        operating / "operating_point_table.tex",
        operating / "operating_point_selection.pdf",
    ]
    missing_operating = [path for path in required_operating if not path.is_file()]
    if missing_operating:
        raise RuntimeError("Operating-point evidence is missing: " + ", ".join(str(path) for path in missing_operating))

    dashboard = STAGING / "outputs" / "digital_twin_dashboard"
    required_dashboard = [
        dashboard / "index.html",
        dashboard / "dashboard_payload.json",
        dashboard / "assets" / "operating_point_selection.png",
        dashboard / "assets" / "uav_to_uav_mmwave_validation.png",
        dashboard / "assets" / "aerpaw_cellular_validation.png",
        dashboard / "assets" / "horizon_sweep.png",
        dashboard / "assets" / "factorial_feature_ablation.png",
        dashboard / "assets" / "packet_level_controller.png",
        dashboard / "assets" / "miluv_validation.png",
    ]
    missing_dashboard = [path for path in required_dashboard if not path.is_file()]
    if missing_dashboard:
        raise RuntimeError("Digital-twin dashboard evidence is missing: " + ", ".join(str(path) for path in missing_dashboard))

    neural_raw = STAGING / "outputs" / "publication_neural_extension"
    neural_aggregate = STAGING / "outputs" / "publication_neural_5seed_extension"
    required_neural = [
        neural_raw / "summary.json",
        neural_raw / "per_seed" / "seed_37" / "metrics_overall.csv",
        neural_raw / "per_seed" / "seed_47" / "metrics_overall.csv",
        neural_aggregate / "summary.json",
        neural_aggregate / "per_seed_metrics.csv",
        neural_aggregate / "metrics_overall.csv",
        neural_aggregate / "neural_seed_extension_table.tex",
        neural_aggregate / "neural_seed_extension_full_table.tex",
    ]
    missing_neural = [path for path in required_neural if not path.is_file()]
    if missing_neural:
        raise RuntimeError("Neural seed-extension evidence is missing: " + ", ".join(str(path) for path in missing_neural))


def _scan_text(path: Path, text: str) -> list[str]:
    findings: list[str] = []
    for pattern in IDENTITY_PATTERNS:
        if pattern.search(text):
            findings.append(f"{path.relative_to(STAGING).as_posix()}: {pattern.pattern}")
    return findings


def _validate_anonymity() -> None:
    findings: list[str] = []
    forbidden_suffixes = {".bag", ".part", ".pyc", ".pyo", ".aux", ".log"}
    for path in sorted(STAGING.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() in forbidden_suffixes:
            findings.append(f"forbidden file: {path.relative_to(STAGING).as_posix()}")
        if path.suffix.lower() in TEXT_SUFFIXES:
            findings.extend(_scan_text(path, path.read_text(encoding="utf-8", errors="replace")))

    pdf = STAGING / "paper" / "main_anonymized.pdf"
    pdftotext = shutil.which("pdftotext")
    if pdftotext is None:
        raise RuntimeError("pdftotext is required for the anonymous PDF identity scan")
    result = subprocess.run(
        [pdftotext, str(pdf), "-"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    findings.extend(_scan_text(pdf, result.stdout))

    pdfinfo = shutil.which("pdfinfo")
    if pdfinfo is None:
        raise RuntimeError("pdfinfo is required for the anonymous PDF page-count check")
    info = subprocess.run(
        [pdfinfo, str(pdf)], check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout
    match = re.search(r"^Pages:\s+(\d+)$", info, flags=re.MULTILINE)
    if match is None or int(match.group(1)) > 55:
        raise RuntimeError(f"Anonymous single-column manuscript must remain at or below 55 pages; pdfinfo was:\n{info}")

    if findings:
        raise RuntimeError("Anonymous package validation failed:\n" + "\n".join(findings))


def _write_manifest() -> None:
    rows: list[str] = []
    for path in sorted(STAGING.rglob("*")):
        if not path.is_file() or path.name == "PACKAGE_MANIFEST.txt":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{digest}  {path.relative_to(STAGING).as_posix()}")
    (STAGING / "PACKAGE_MANIFEST.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_zip() -> None:
    _assert_workspace_path(ZIP_TEMP)
    ZIP_TEMP.unlink(missing_ok=True)
    with zipfile.ZipFile(ZIP_TEMP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(STAGING.rglob("*")):
            if path.is_file():
                arcname = Path("anonymous_supplementary") / path.relative_to(STAGING)
                archive.write(path, arcname.as_posix())
    with zipfile.ZipFile(ZIP_TEMP, "r") as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP CRC validation failed at {bad}")


def _publish() -> None:
    _assert_workspace_path(FINAL)
    _assert_workspace_path(BACKUP)
    _assert_workspace_path(ZIP_PATH)
    _remove_tree(BACKUP)
    if FINAL.exists():
        FINAL.rename(BACKUP)
    try:
        STAGING.rename(FINAL)
    except Exception:
        if BACKUP.exists() and not FINAL.exists():
            BACKUP.rename(FINAL)
        raise
    os.replace(ZIP_TEMP, ZIP_PATH)
    _remove_tree(BACKUP)


def main() -> None:
    _remove_tree(STAGING)
    STAGING.mkdir(parents=True)

    for relative in ROOT_FILES:
        _copy_file(relative)
    for name in CONFIG_FILES:
        _copy_file(f"configs/{name}")
    for name in SCRIPT_FILES:
        _copy_file(f"scripts/{name}")
    _copy_tree("fanet")
    _copy_tree("tests")
    _copy_tree("outputs/publication_compact", excluded_names={"README.md"})
    _copy_tree("outputs/paper_like_submission")
    _copy_tree("outputs/external_validation")
    _copy_tree("outputs/aerpaw_cellular_validation")
    _copy_tree("outputs/uav_to_uav_mmwave_validation")
    _copy_tree("outputs/miluv_validation")
    _copy_tree("outputs/horizon_sweep")
    _copy_tree("outputs/factorial_feature_ablation")
    _copy_tree("outputs/packet_level_controller")
    _copy_tree("outputs/end_to_end_latency")
    _copy_tree("outputs/operating_point")
    _copy_tree("outputs/digital_twin_dashboard")
    _copy_tree("outputs/publication_neural_extension")
    _copy_tree("outputs/publication_neural_5seed_extension")
    _copy_tree("data/external_validation/derived")
    _copy_tree("data/external_validation/raw/miluv/cirObstacles_3_random_0")
    _copy_file("data/external_validation/README.md")
    _copy_file("paper/main_anonymized.pdf")
    _copy_file("paper/ANONYMOUS_SUPPLEMENTARY_NOTES.md")

    _write_generated_files()
    _validate_evidence()
    _validate_anonymity()
    _write_manifest()
    _write_zip()
    _publish()

    files = sum(1 for path in FINAL.rglob("*") if path.is_file())
    size_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
    print(f"Built {ZIP_PATH.name}: {files} files, {size_mb:.2f} MiB")
    print("Validated: 20-seed primary run, 5-seed PyTorch neural extension, external/AERPAW/UAV-to-UAV evidence, operating point, dashboard, anonymity, single-column PDF <=55 pages, ZIP CRC")


if __name__ == "__main__":
    main()
