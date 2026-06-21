from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NEURAL_MODELS = {
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


def _check(name: str, condition: bool, detail: str) -> dict:
    return {"check": name, "status": "pass" if condition else "fail", "detail": detail}


def _extract_environment(text: str, environment: str) -> str:
    match = re.search(rf"\\begin\{{{re.escape(environment)}\}}(.*?)\\end\{{{re.escape(environment)}\}}", text, re.S)
    return match.group(1).strip() if match else ""


def _compact_checks(root: Path) -> list[dict]:
    summary_path = root / "outputs/publication_compact/summary.json"
    if not summary_path.exists():
        return [_check("compact_summary", False, f"missing {summary_path.relative_to(root)}")]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    backend = summary.get("model_backend", {})
    wrong = {name: backend.get(name) for name in NEURAL_MODELS if backend.get(name) != "pytorch"}
    return [
        _check("compact_summary", True, "summary.json exists"),
        _check("compact_torch_available", summary.get("torch_available") is True, str(summary.get("torch_available"))),
        _check("compact_no_surrogate", summary.get("surrogate_used") is False, str(summary.get("surrogate_used"))),
        _check("compact_neural_backends", not wrong, f"non-PyTorch entries: {wrong}"),
    ]


def _confirmatory_checks(root: Path) -> list[dict]:
    config = json.loads((root / "configs/paper_like_submission.json").read_text(encoding="utf-8"))
    expected = [int(seed) for seed in config["training"]["seed_list"]]
    summary_path = root / "outputs/paper_like_submission/summary.json"
    if not summary_path.exists():
        return [_check("confirmatory_summary", False, f"missing {summary_path.relative_to(root)}")]
    seed_dirs = sorted((root / "outputs/paper_like_submission/per_seed").glob("seed_*"))
    found = sorted(int(path.name.split("_")[-1]) for path in seed_dirs if (path / "metrics_overall.csv").exists())
    metrics = pd.read_csv(root / "outputs/paper_like_submission/metrics_overall.csv")
    models = set(metrics["Model"].astype(str))
    required_models = {"Union-Find detection oracle", "Kinetic-TopoGuard"}
    return [
        _check("confirmatory_summary", True, "summary.json exists"),
        _check("confirmatory_seed_count", len(expected) == 20 and len(set(expected)) == 20, f"configured={len(set(expected))}"),
        _check("confirmatory_seed_outputs", found == sorted(expected), f"completed={found}"),
        _check("confirmatory_models", required_models.issubset(models) and any(name.startswith("Shallow ML") for name in models), str(sorted(models))),
    ]


def _external_checks(root: Path) -> list[dict]:
    manifest_path = root / "data/external_validation/derived/forestry_multidrone_trace_manifest.json"
    metrics_path = root / "outputs/external_validation/external_metrics_summary.csv"
    predictions_path = root / "outputs/external_validation/external_predictions.csv"
    if not manifest_path.exists() or not metrics_path.exists() or not predictions_path.exists():
        return [_check("external_validation", False, f"manifest={manifest_path.exists()}, metrics={metrics_path.exists()}")]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = pd.read_csv(metrics_path)
    predictions = pd.read_csv(predictions_path)
    expected_md5 = {
        "d9d88fbb378e51bdf6ac57a4ca2be08f",
        "ef86998f372689d95a4be696b3d92b88",
        "adb55c9c6282b7bbd609cda5361b00f6",
    }
    actual_md5 = {item.get("md5") for item in manifest.get("sources", [])}
    return [
        _check("external_validation", True, "derived trace manifest and metrics exist"),
        _check("external_source_doi", manifest.get("doi") == "10.5281/zenodo.14701641", str(manifest.get("doi"))),
        _check("external_vehicle_count", len(manifest.get("sources", [])) == 3, f"sources={len(manifest.get('sources', []))}"),
        _check("external_source_checksums", actual_md5 == expected_md5, str(sorted(actual_md5))),
        _check("external_radius_sensitivity", metrics["Radius_quantile"].nunique() == 3, f"radii={metrics['Radius_quantile'].nunique()}"),
        _check("external_clipping_audit", {"Raw_MAE_mean", "Prediction_Clipped_Fraction_mean"}.issubset(metrics.columns), "raw and clipped metrics present"),
        _check("external_prediction_rows", len(predictions) == 107730, f"rows={len(predictions)}"),
    ]


def _metadata_checks(root: Path) -> list[dict]:
    manuscript = (root / "paper/main.tex").read_text(encoding="utf-8")
    title_page = (root / "paper/title_page.tex").read_text(encoding="utf-8")
    bibliography = (root / "paper/cas-refs.bib").read_text(encoding="utf-8")
    highlights_body = _extract_environment(manuscript, "highlights")
    highlights = [item.strip() for item in re.findall(r"\\item\s+([^\n]+)", highlights_body)]
    perslay_match = re.search(r"@inproceedings\{carriere2020perslay,(.*?)\n\}", bibliography, re.S)
    perslay = perslay_match.group(1) if perslay_match else ""
    required_perslay = ["booktitle", "pages", "year", "editor", "volume", "series", "publisher", "url"]
    mojibake = re.compile(r"(?:Ã.|Â.|�)")
    metadata_files = [root / ".zenodo.json", root / "CITATION.cff", root / "paper/title_page.tex"]
    portal_metadata = root / "EAAI_PORTAL_METADATA.md"
    cover_letter = root / "EAAI_COVER_LETTER.md"
    portal_files = [
        root / "paper/main_anonymized.pdf",
        root / "paper/title_page.pdf",
        root / "anonymous_supplementary.zip",
        root / "paper/figures/plantuml/graphical_abstract.pdf",
    ]
    return [
        _check("elsevier_highlight_count", 3 <= len(highlights) <= 5, f"count={len(highlights)}"),
        _check("elsevier_highlight_length", bool(highlights) and all(len(item) <= 85 for item in highlights), str([len(item) for item in highlights])),
        _check("perslay_bibliography", bool(perslay) and all(re.search(rf"\b{field}\s*=", perslay) for field in required_perslay), str(required_perslay)),
        _check("external_dataset_bibliography", "10.5281/zenodo.14701641" in bibliography, "Zenodo DOI present"),
        _check("title_page_declarations", all(label in title_page for label in ["CRediT", "Funding", "competing interest", "Data and code availability", "generative AI"]), "required declarations present"),
        _check("metadata_encoding", not any(mojibake.search(path.read_text(encoding="utf-8")) for path in metadata_files), "no common mojibake markers"),
        _check("orcid_consistency", all("0000-0001-9259-7112" in path.read_text(encoding="utf-8") for path in metadata_files), "ORCID present in title page, CFF, and Zenodo metadata"),
        _check("portal_metadata_package", portal_metadata.exists() and cover_letter.exists(), "metadata and cover-letter files exist"),
        _check("portal_upload_files", all(path.exists() and path.stat().st_size > 0 for path in portal_files), str([path.name for path in portal_files])),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit evidence and journal-package readiness.")
    parser.add_argument("--output", type=Path, default=ROOT / "submission_readiness_audit.json")
    args = parser.parse_args()
    checks = [*_compact_checks(ROOT), *_confirmatory_checks(ROOT), *_external_checks(ROOT), *_metadata_checks(ROOT)]
    payload = {
        "status": "pass" if all(item["status"] == "pass" for item in checks) else "fail",
        "checks": checks,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for item in checks:
        print(f"[{item['status'].upper()}] {item['check']}: {item['detail']}")
    print(f"Overall: {payload['status'].upper()}")
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
