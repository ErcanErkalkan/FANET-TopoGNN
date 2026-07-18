from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fanet.external_claims import claim_rules_manifest, lint_external_claim_path
from fanet.provenance import build_file_manifest, relative_repo_path, verify_manifest


DEFAULT_OUTPUT = ROOT / "outputs" / "external_evidence_matrix"
PROTOCOLS = {
    "aerpaw": ROOT / "outputs/aerpaw_cellular_validation/aerpaw_cellular_protocol.json",
    "wines": ROOT / "outputs/uav_to_uav_mmwave_validation/uav_to_uav_mmwave_protocol.json",
    "forestry": ROOT / "outputs/external_validation/external_validation_protocol.json",
}


def _load_verified(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"required evidence protocol is missing: {relative_repo_path(path, ROOT)}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    verification = verify_manifest(payload.get("source_files", []), ROOT)
    if not verification["valid"]:
        raise RuntimeError(f"source verification failed for {path.name}: " + "; ".join(verification["errors"]))
    return payload


def _rows(aerpaw: dict, wines: dict, forestry: dict) -> list[dict]:
    common_forestry = {
        "dataset": "Forestry multi-drone field experiment",
        "source_identifier": forestry["source_identifier"],
        "license": forestry["license"],
        "sampling": f"{forestry['sampling']['sample_rate_hz']} Hz aligned GNSS trace",
    }
    return [
        {
            "evidence_id": "forestry_measured_motion",
            **common_forestry,
            "evidence_kind": "measured motion",
            "directly_measured": "GNSS latitude; longitude; altitude; timestamps",
            "derived_or_transported": "aligned local ENU trajectories and pair distances",
            "split_or_support": "shared valid GNSS interval; bounded interpolation gaps",
            "allowed_claim": "measured three-UAV field motion",
            "prohibited_claim": "measured peer RF; packet delivery; measured connectivity",
            "exclusions": "; ".join(forestry["exclusions"]),
        },
        {
            "evidence_id": "forestry_counterfactual_radius",
            **common_forestry,
            "evidence_kind": "derived counterfactual graph sensitivity",
            "directly_measured": "motion only",
            "derived_or_transported": "radius graphs; component labels; frozen simulation-model predictions",
            "split_or_support": "25th/50th/75th pair-distance radii fixed without outcome selection",
            "allowed_claim": "motion-domain sensitivity under explicit radius assumptions",
            "prohibited_claim": "measured RF; measured PDR; field-validated connectivity",
            "exclusions": "; ".join(forestry["exclusions"]),
        },
        {
            "evidence_id": "aerpaw_lte_connection_state",
            "dataset": "AERPAW Datasets 22/23 LTE",
            "source_identifier": "AERPAW Dataset-22 and Dataset-23 landing pages",
            "license": next(iter(aerpaw["sources"].values()))["license"],
            "sampling": aerpaw["sampling"]["dataset22_lte"],
            "evidence_kind": "measured UAV-to-infrastructure cellular KPI/state",
            "directly_measured": "LTE connection state; RSRP; RSRQ; RSSI; position; altitude; timestamps",
            "derived_or_transported": "local geometry; RSRP threshold; logistic connection probability",
            "split_or_support": "chronological final 30% test",
            "allowed_claim": "chronological UAV-to-infrastructure LTE connection-state prediction",
            "prohibited_claim": "inter-UAV validation; FANET PDR",
            "exclusions": "; ".join(aerpaw["exclusions"]),
        },
        {
            "evidence_id": "aerpaw_iperf_throughput",
            "dataset": "AERPAW Dataset 23 iPerf",
            "source_identifier": aerpaw["sources"]["dataset23_iperf_two_sweeps"]["source_identifier"],
            "license": aerpaw["sources"]["dataset23_iperf_two_sweeps"]["license"],
            "sampling": aerpaw["sampling"]["dataset23"],
            "evidence_kind": "measured UAV-to-infrastructure throughput",
            "directly_measured": "iPerf throughput; LTE KPIs; position; timestamps",
            "derived_or_transported": "timestamp-aligned RF/KPI throughput predictions",
            "split_or_support": "primary chronological final 30%; first-half-to-second-half robustness",
            "allowed_claim": "chronological throughput prediction including negative R2",
            "prohibited_claim": "robust throughput transfer; inter-UAV PDR",
            "exclusions": "; ".join(aerpaw["exclusions"]),
        },
        {
            "evidence_id": "wines_measured_peer_rf",
            "dataset": "WiNES UAV-to-UAV 60 GHz",
            "source_identifier": wines["source_identifier"],
            "license": wines["license"],
            "sampling": wines["sampling"],
            "evidence_kind": "measured peer RF",
            "directly_measured": "; ".join(wines["directly_measured_variables"]),
            "derived_or_transported": "link viability and held-out-distance model scores",
            "split_or_support": "train 6-32 m; held-out test 33-40 m",
            "allowed_claim": "held-out-distance UAV-to-UAV 60 GHz link viability",
            "prohibited_claim": "packet delivery; forestry same-site calibration",
            "exclusions": "; ".join(wines["exclusions"]),
        },
        {
            "evidence_id": "wines_forestry_transport_in_support",
            "dataset": "WiNES RF transported to forestry motion",
            "source_identifier": wines["source_identifier"] + " + " + forestry["source_identifier"],
            "license": wines["license"] + "; forestry " + forestry["license"],
            "sampling": "10 Hz forestry motion with discrete-distance WiNES interpolation",
            "evidence_kind": "transported in-support sensitivity",
            "directly_measured": "WiNES RF at source site; forestry motion at separate site",
            "derived_or_transported": "interpolated pair-link probabilities and expected component count",
            "split_or_support": "primary rows require every forestry pair within 6-40 m measured WiNES support",
            "allowed_claim": "in-support transported RF sensitivity",
            "prohibited_claim": "same-site calibration; measured forestry RF; packet delivery",
            "exclusions": "rows with any pair outside WiNES distance support",
        },
        {
            "evidence_id": "wines_forestry_transport_clamped",
            "dataset": "WiNES RF transported to forestry motion",
            "source_identifier": wines["source_identifier"] + " + " + forestry["source_identifier"],
            "license": wines["license"] + "; forestry " + forestry["license"],
            "sampling": "10 Hz forestry motion with boundary-clamped WiNES probabilities",
            "evidence_kind": "secondary out-of-support sensitivity",
            "directly_measured": "WiNES RF at source site; forestry motion at separate site",
            "derived_or_transported": "nearest-support clamping and expected component count",
            "split_or_support": "out-of-support rows explicitly labelled full_trace_clamped",
            "allowed_claim": "secondary clamped sensitivity only",
            "prohibited_claim": "extrapolative validation; same-site calibration; measured forestry RF",
            "exclusions": "not eligible for primary transported summary",
        },
    ]


def _write_tex(frame: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{tabular}{p{0.16\linewidth}p{0.17\linewidth}p{0.22\linewidth}p{0.20\linewidth}p{0.17\linewidth}}",
        r"\toprule",
        r"Evidence & Direct measurement & Derived/transported quantity & Allowed claim & Boundary \\",
        r"\midrule",
    ]
    for row in frame.itertuples():
        values = [row.dataset, row.directly_measured, row.derived_or_transported, row.allowed_claim, row.prohibited_claim]
        escaped = [str(value).replace("_", r"\_").replace("&", r"\&") for value in values]
        lines.append(" & ".join(escaped) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _claim_markdown(frame: pd.DataFrame) -> str:
    lines = ["# External evidence claim boundaries", "", "Each row separates direct measurements from derived or transported quantities.", ""]
    for row in frame.itertuples():
        lines.extend([
            f"## {row.evidence_id}", "",
            f"- Direct: {row.directly_measured}",
            f"- Derived/transported: {row.derived_or_transported}",
            f"- Allowed: {row.allowed_claim}",
            f"- Prohibited: {row.prohibited_claim}", "",
        ])
    return "\n".join(lines)


def main() -> int:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description="Build a unified external-evidence matrix and enforce claim boundaries.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manuscript", type=Path, default=ROOT / "paper/main.tex")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing evidence output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payloads = {name: _load_verified(path) for name, path in PROTOCOLS.items()}
    violations = lint_external_claim_path(args.manuscript)
    if violations:
        raise RuntimeError("external manuscript claim lint failed: " + json.dumps(violations, indent=2))
    frame = pd.DataFrame(_rows(payloads["aerpaw"], payloads["wines"], payloads["forestry"]))
    frame.to_csv(args.output_dir / "evidence_matrix.csv", index=False)
    _write_tex(frame, args.output_dir / "evidence_matrix.tex")
    _write_tex(frame, ROOT / "paper/tables/generated/evidence_matrix.tex")
    (args.output_dir / "claim_boundaries.md").write_text(_claim_markdown(frame), encoding="utf-8")

    forestry_manifest = json.loads((ROOT / "data/external_validation/derived/forestry_multidrone_trace_manifest.json").read_text(encoding="utf-8"))
    source_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "aerpaw": {"source_identifier": [value["source_identifier"] for value in payloads["aerpaw"]["sources"].values()], "license": next(iter(payloads["aerpaw"]["sources"].values()))["license"], "files": payloads["aerpaw"]["source_files"]},
            "wines_60ghz": {"source_identifier": payloads["wines"]["source_identifier"], "license": payloads["wines"]["license"], "files": [payloads["wines"]["source_files"][0]]},
            "forestry": {"source_identifier": payloads["forestry"]["source_identifier"], "license": payloads["forestry"]["license"], "derived_files": payloads["forestry"]["source_files"], "upstream_raw_sources": [{**row, "local_file_available": (ROOT / row["relative_path"]).is_file()} for row in forestry_manifest.get("sources", [])]},
        },
        "raw_to_derived_transforms": {name: payload.get("raw_to_derived_transform") for name, payload in payloads.items()},
        "sampling": {name: payload.get("sampling") for name, payload in payloads.items()},
        "exclusions": {name: payload.get("exclusions") for name, payload in payloads.items()},
    }
    (args.output_dir / "source_manifest.json").write_text(json.dumps(source_manifest, indent=2), encoding="utf-8")
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "evidence_rows": int(len(frame)),
        "claim_rules": claim_rules_manifest(),
        "manuscript_claim_lint": {"source": relative_repo_path(args.manuscript, ROOT), "status": "pass", "violations": []},
        "input_protocols": build_file_manifest(PROTOCOLS.values(), ROOT),
        "source_files": build_file_manifest([Path(__file__).resolve(), ROOT / "fanet/external_claims.py", args.manuscript], ROOT),
        "git_commit": git or "unavailable",
        "hardware": {"platform": platform.platform()},
        "package_versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scikit-learn")
        },
        "runtime_seconds": float(time.perf_counter() - started),
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"Wrote {relative_repo_path(args.output_dir, ROOT)} with {len(frame)} evidence rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
