from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fanet.provenance import relative_repo_path, sha256_file, verify_manifest
from fanet.external_claims import lint_external_claim_path
from scripts.scan_anonymity import (
    AnonymityScanError,
    extract_anonymous_tex,
    format_failures,
    load_pattern_config,
    scan_paths,
    scan_text,
)
from scripts.audit_bibliography import audit_bibliography, render_markdown
from scripts.validate_submission_text import run_validation as validate_submission_text


CACHE_VERSION = "kinetic_topoguard_v6_fragmentation_events_correlated_radio"
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


def _bibliography_checks(root: Path) -> list[dict]:
    bib_path = root / "paper/cas-refs.bib"
    tex_paths = [root / "paper/main.tex"]
    payload = audit_bibliography(bib_path, tex_paths)
    output_dir = root / "docs/eaai_revision"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "bibliography_audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "BIBLIOGRAPHY_AUDIT.md").write_text(
        render_markdown(payload), encoding="utf-8"
    )
    coverage = payload["required_group_coverage"]
    return [
        _check("bibliography_integrity", payload["status"] == "pass", "; ".join(payload["hard_failures"]) or "no duplicate DOI/title/key or missing citation"),
        _check("bibliography_required_coverage", all(item["present"] and item["cited"] for item in coverage.values()), str({key: item for key, item in coverage.items() if not (item["present"] and item["cited"])})),
        _check("bibliography_dataset_entries", len(payload["dataset_entries"]) >= 5, str(payload["dataset_entries"])),
        _check("bibliography_software_metadata", all(item["version"] and item["archive"] and item["url"] for item in payload["software_entries"]), str(payload["software_entries"])),
    ]


def _submission_text_checks(root: Path) -> list[dict]:
    payload = validate_submission_text(root)
    report_path = root / "docs/eaai_revision/submission_text_audit.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    failures = [item["name"] for item in payload["checks"] if not item["passed"]]
    return [
        _check(
            "submission_text_validation",
            payload["status"] == "pass",
            f"abstract_words={payload['abstract_word_count']}; failures={failures}",
        )
    ]


def _provenance_check(name: str, manifest: object, root: Path) -> dict:
    verification = verify_manifest(manifest, root)
    if verification["valid"]:
        detail = "verified=" + str(
            [item["relative_path"] for item in verification["files"]]
        )
    else:
        detail = "; ".join(verification["errors"])
    return _check(name, verification["valid"], detail)


def _posix_manifest_paths(manifest: object) -> tuple[bool, str]:
    entries = manifest.get("files", []) if isinstance(manifest, dict) else manifest
    if not isinstance(entries, list) or not entries:
        return False, "manifest contains no file paths"
    paths = [item.get("relative_path") for item in entries if isinstance(item, dict)]
    valid = len(paths) == len(entries) and all(
        isinstance(path, str) and "\\" not in path for path in paths
    )
    return valid, str(paths)


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
        _check("compact_cache_version", summary.get("cache_version") == CACHE_VERSION, str(summary.get("cache_version"))),
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
    risks = pd.read_csv(root / "outputs/paper_like_submission/risk_metrics.csv")
    operating = pd.read_csv(root / "outputs/paper_like_submission/operating_point_metrics.csv")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    models = set(metrics["Model"].astype(str))
    required_models = {"Current-state persistence baseline", "Kinetic-TopoGuard"}
    event_columns = {
        "Alert_Event_Precision_mean",
        "Alert_Event_Recall_mean",
        "Alert_Event_F1_mean",
        "False_Alert_Events_per_minute_mean",
    }
    return [
        _check("confirmatory_summary", True, "summary.json exists"),
        _check("confirmatory_cache_version", summary.get("cache_version") == CACHE_VERSION, str(summary.get("cache_version"))),
        _check("confirmatory_seed_count", len(expected) == 20 and len(set(expected)) == 20, f"configured={len(set(expected))}"),
        _check("confirmatory_seed_outputs", found == sorted(expected), f"completed={found}"),
        _check("confirmatory_models", required_models.issubset(models) and any(name.startswith("Shallow ML") for name in models), str(sorted(models))),
        _check("confirmatory_event_metrics", event_columns.issubset(risks.columns), str(sorted(set(risks.columns) & event_columns))),
        _check(
            "confirmatory_validation_only_operating_points",
            {
                "Validation_Constraint_Met_mean",
                "Selected_Threshold_mean",
                "Test_Alert_Event_F1_mean",
            }.issubset(operating.columns),
            str(sorted(operating.columns)),
        ),
    ]


def _neural_extension_checks(root: Path) -> list[dict]:
    raw_summary_path = root / "outputs/publication_neural_extension/summary.json"
    protocol_path = root / "outputs/publication_neural_5seed_extension/full_coverage_protocol.json"
    metrics_path = root / "outputs/publication_neural_5seed_extension/full_metrics.csv"
    per_seed_path = root / "outputs/publication_neural_5seed_extension/per_seed_full_metrics.csv"
    model_protocol_path = root / "outputs/publication_neural_5seed_extension/model_protocol.csv"
    failures_path = root / "outputs/publication_neural_5seed_extension/training_failures.csv"
    table_path = root / "paper/tables/generated/neural_seed_extension_table.tex"
    full_table_path = root / "paper/tables/generated/neural_seed_extension_full_table.tex"
    required_paths = [raw_summary_path, protocol_path, metrics_path, per_seed_path, model_protocol_path, failures_path, table_path, full_table_path]
    missing = [str(path.relative_to(root)) for path in required_paths if not path.exists()]
    if missing:
        return [_check("neural_seed_extension", False, f"missing={missing}")]

    raw_summary = json.loads(raw_summary_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    metrics = pd.read_csv(metrics_path)
    per_seed = pd.read_csv(per_seed_path)
    model_protocol = pd.read_csv(model_protocol_path)
    failures = pd.read_csv(failures_path)
    expected_models = {
        "Current-state persistence baseline", "Current-state ExtraTrees", "Shallow ML",
        "Kinetic-TopoGuard", "Source-Gated Kinetic-TopoGuard", "GCN", "GAT", "GraphSAGE",
        "PI+MLP", "FANET-TopoGNN", "FANET-TopoGNN (concat)", "T-GCN (w=5)",
        "STGCN (w=5)", "TGN (w=5)",
    }
    neural = metrics[metrics["Model"].isin({
        "GCN",
        "GAT",
        "GraphSAGE",
        "PI+MLP",
        "FANET-TopoGNN",
        "FANET-TopoGNN (concat)",
        "T-GCN (w=5)",
        "STGCN (w=5)",
        "TGN (w=5)",
    })]
    counts = per_seed.groupby("Model")["seed"].nunique().to_dict()
    budget_columns = {
        "input_feature_set", "temporal_window", "parameter_count", "trainable_parameter_count",
        "optimizer", "learning_rate", "batch_size", "max_epochs", "early_stopping",
        "validation_objective", "threshold_selection_rule", "training_time_seconds",
        "training_time_status", "inference_time_ms_mean", "backend", "backend_version", "seed_count",
    }
    return [
        _check("neural_seed_extension", True, "raw runs and full five-seed aggregate exist"),
        _check("neural_extension_raw_backend", raw_summary.get("n_seeds") == 2 and raw_summary.get("torch_available") is True and raw_summary.get("surrogate_used") is False, str(raw_summary.get("model_backend", {}))),
        _check("neural_extension_seed_count", protocol.get("n_seeds") == 5 and protocol.get("seeds") == [7, 17, 27, 37, 47], str(protocol.get("seeds"))),
        _check("baseline_full_coverage", set(metrics["Model"].astype(str)) == expected_models and all(counts.get(model) == 5 for model in expected_models), f"models={len(metrics)}, counts={counts}"),
        _check("neural_extension_model_coverage", len(neural) == 9 and all(int(value) == 5 for value in neural["seed_count"]), f"models={len(neural)}, seeds={sorted(set(neural['seed_count'].astype(int)))}"),
        _check("neural_extension_pytorch", set(neural["backends"].astype(str)) == {"pytorch"}, str(sorted(set(neural["backends"].astype(str))))),
        _check("neural_extension_no_silent_failures", failures.empty, failures.to_dict("records")),
        _check("neural_extension_model_budgets", set(model_protocol["Model"].astype(str)) == expected_models and budget_columns.issubset(model_protocol.columns), str(sorted(model_protocol.columns))),
        _check("neural_extension_parameter_counts", model_protocol[model_protocol["Model"].isin(neural["Model"])]["parameter_count"].fillna(0).gt(0).all(), model_protocol[["Model", "parameter_count"]].to_dict("records")),
        _check(
            "neural_extension_event_metrics",
            {"Alert_Event_Precision_mean", "Alert_Event_Recall_mean", "Alert_Event_F1_mean", "False_Alert_Events_per_minute_mean"}.issubset(metrics.columns),
            str(sorted(metrics.columns)),
        ),
    ]


def _explainability_checks(root: Path) -> list[dict]:
    output = root / "outputs/explainability"
    paths = {
        "per_seed": output / "grouped_importance_per_seed.csv",
        "summary": output / "grouped_importance_summary.csv",
        "gates": output / "gate_coefficients.csv",
        "local": output / "local_explanations.csv",
        "feature_plot": output / "feature_importance.pdf",
        "local_plot": output / "local_explanation_examples.pdf",
        "protocol": output / "protocol.json",
        "paper_table": root / "paper/tables/generated/explainability_table.tex",
        "paper_figure": root / "paper/figures/generated/explainability.pdf",
    }
    missing = [relative_repo_path(path, root) for path in paths.values() if not path.is_file()]
    if missing:
        return [_check("explainability", False, f"missing={missing}")]
    per_seed = pd.read_csv(paths["per_seed"])
    gates = pd.read_csv(paths["gates"])
    local = pd.read_csv(paths["local"])
    protocol = json.loads(paths["protocol"].read_text(encoding="utf-8"))
    groups = {
        "current state", "graph summaries", "persistence image bins",
        "persistence-image summaries", "current pair distances",
        "projected pair distances", "pair-distance deltas", "speed summaries",
        "projected fragmentation statistics",
    }
    models = {"Kinetic-TopoGuard", "Source-Gated Kinetic-TopoGuard"}
    permutation = per_seed[per_seed["method"] == "grouped_permutation"]
    selected_local = local[local["status"] == "selected"]
    shap_states = {str(value.get("status")) for value in protocol.get("shap", {}).values()}
    manifest = verify_manifest(protocol.get("source_files", []), root)
    return [
        _check("explainability", True, "five-seed grouped permutation, optional TreeSHAP, local examples, and paper assets exist"),
        _check("explainability_seed_coverage", sorted(per_seed["seed"].unique().tolist()) == [7, 17, 27, 37, 47], str(sorted(per_seed["seed"].unique().tolist()))),
        _check("explainability_grouped_permutation_coverage", len(permutation) == 180 and set(permutation["Model"]) == models and set(permutation["feature_group"]) == groups, f"rows={len(permutation)}, models={sorted(set(permutation['Model']))}"),
        _check("explainability_run_block_unit", protocol.get("grouped_permutation", {}).get("unit") == "complete run block" and protocol.get("grouped_permutation", {}).get("equal_length_donor_required") is True, str(protocol.get("grouped_permutation"))),
        _check("explainability_no_selection_leakage", protocol.get("model_selection_uses_explanation") is False and "post-freeze" in protocol.get("test_partition_role", ""), str(protocol.get("test_partition_role"))),
        _check("explainability_prediction_immutability", protocol.get("prediction_immutability_verified") is True, str(protocol.get("prediction_immutability_verified"))),
        _check("explainability_shap_status", bool(shap_states) and shap_states.issubset({"completed", "not_installed_or_import_failed", "failed_without_aborting_grouped_permutation"}), str(sorted(shap_states))),
        _check("explainability_gate_coverage", len(gates) == 40 and set(gates["feature_source"]) == {"current", "graph", "topology", "kinematic"}, f"rows={len(gates)}"),
        _check("explainability_local_examples", set(selected_local["example_type"]) == {"true_positive_event", "false_positive_event", "false_negative_event", "true_negative_interval"} and selected_local.groupby(["seed", "Model", "example_type"])["feature_group"].nunique().eq(9).all(), str(selected_local.groupby(["Model", "example_type"]).size().to_dict())),
        _check("explainability_source_hashes", manifest["valid"], str(manifest)),
    ]


def _external_checks(root: Path) -> list[dict]:
    manifest_path = root / "data/external_validation/derived/forestry_multidrone_trace_manifest.json"
    protocol_path = root / "outputs/external_validation/external_validation_protocol.json"
    metrics_path = root / "outputs/external_validation/external_metrics_summary.csv"
    predictions_path = root / "outputs/external_validation/external_predictions.csv"
    required = [manifest_path, protocol_path, metrics_path, predictions_path]
    missing = [relative_repo_path(path, root) for path in required if not path.is_file()]
    if missing:
        return [_check("external_validation", False, f"missing={missing}")]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    metrics = pd.read_csv(metrics_path)
    predictions = pd.read_csv(predictions_path)
    expected_md5 = {
        "d9d88fbb378e51bdf6ac57a4ca2be08f",
        "ef86998f372689d95a4be696b3d92b88",
        "adb55c9c6282b7bbd609cda5361b00f6",
    }
    actual_md5 = {item.get("md5") for item in manifest.get("sources", [])}
    derived_files = manifest.get("files") or [
        {
            "relative_path": manifest.get("derived_file", ""),
            "sha256": manifest.get("derived_sha256"),
        }
    ]
    source_files = [
        {
            "relative_path": item.get("relative_path")
            or f"data/external_validation/raw/{item.get('file', '')}",
            "bytes": item.get("bytes"),
            "sha256": item.get("sha256"),
        }
        for item in manifest.get("sources", [])
    ]
    derived_posix, derived_paths = _posix_manifest_paths(derived_files)
    protocol_posix, protocol_paths = _posix_manifest_paths(protocol.get("source_files", []))
    return [
        _check("external_validation", True, "derived trace manifest and metrics exist"),
        _check("external_source_doi", manifest.get("doi") == "10.5281/zenodo.14701641", str(manifest.get("doi"))),
        _check("external_vehicle_count", len(manifest.get("sources", [])) == 3, f"sources={len(manifest.get('sources', []))}"),
        _check("external_source_checksum_records", actual_md5 == expected_md5, str(sorted(actual_md5))),
        _provenance_check("forestry_derived_hashes", derived_files, root),
        _provenance_check("forestry_source_hashes", source_files, root),
        _provenance_check("external_protocol_source_hashes", protocol.get("source_files", []), root),
        _check("forestry_manifest_posix_paths", derived_posix, derived_paths),
        _check("external_protocol_posix_paths", protocol_posix, protocol_paths),
        _check("external_radius_sensitivity", metrics["Radius_quantile"].nunique() == 3, f"radii={metrics['Radius_quantile'].nunique()}"),
        _check("external_clipping_audit", {"Raw_MAE_mean", "Prediction_Clipped_Fraction_mean"}.issubset(metrics.columns), "raw and clipped metrics present"),
        _check(
            "external_event_metrics",
            {"Alert_Event_Recall_mean", "Alert_Event_F1_mean"}.issubset(metrics.columns),
            str(sorted(metrics.columns)),
        ),
        _check("external_prediction_rows", len(predictions) == 107730, f"rows={len(predictions)}"),
    ]


def _aerpaw_cellular_checks(root: Path) -> list[dict]:
    base = root / "outputs/aerpaw_cellular_validation"
    protocol_path = base / "aerpaw_cellular_protocol.json"
    availability_path = base / "aerpaw_lte_availability_metrics.csv"
    throughput_path = base / "aerpaw_throughput_metrics.csv"
    table_path = root / "paper/tables/generated/aerpaw_cellular_table.tex"
    figure_path = root / "paper/figures/generated/aerpaw_cellular_validation.pdf"
    required_paths = [protocol_path, availability_path, throughput_path, table_path, figure_path]
    missing = [str(path.relative_to(root)) for path in required_paths if not path.exists()]
    if missing:
        return [_check("aerpaw_cellular_validation", False, f"missing={missing}")]

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    availability = pd.read_csv(availability_path)
    throughput = pd.read_csv(throughput_path)
    sources = set(protocol.get("sources", {}))
    source_files = protocol.get("source_files") or list(protocol.get("sources", {}).values())
    posix_paths, path_detail = _posix_manifest_paths(source_files)

    logistic = availability[availability["model"] == "Logistic RF/KPI model"]
    threshold = availability[availability["model"].astype(str).str.startswith("RSRP threshold")]
    d22_model = float(logistic.loc[logistic["dataset"] == "dataset22_lte_semicircle", "f1"].iloc[0])
    d22_baseline = float(threshold.loc[threshold["dataset"] == "dataset22_lte_semicircle", "f1"].iloc[0])
    d23_model = float(logistic.loc[logistic["dataset"] == "dataset23_lte_two_sweeps", "f1"].iloc[0])
    d23_baseline = float(threshold.loc[threshold["dataset"] == "dataset23_lte_two_sweeps", "f1"].iloc[0])
    expected_sources = {
        "dataset22_lte_semicircle",
        "dataset23_lte_two_sweeps",
        "dataset23_nr_two_sweeps",
        "dataset23_iperf_two_sweeps",
    }
    return [
        _check("aerpaw_cellular_validation", True, "protocol, metrics, table, and figure exist"),
        _check("aerpaw_scope_boundary", "not an inter-UAV" in protocol.get("scope", ""), protocol.get("scope", "")),
        _check("aerpaw_source_manifest", sources == expected_sources, str(sorted(sources))),
        _provenance_check("aerpaw_source_hashes", source_files, root),
        _check("aerpaw_source_posix_paths", posix_paths, path_detail),
        _check("aerpaw_availability_rows", len(availability) == 4, f"rows={len(availability)}"),
        _check("aerpaw_d22_lte_gain", d22_model > d22_baseline and d22_model >= 0.9, f"{d22_baseline:.3f}->{d22_model:.3f}"),
        _check("aerpaw_d23_lte_gain", d23_model > d23_baseline and d23_model >= 0.9, f"{d23_baseline:.3f}->{d23_model:.3f}"),
        _check(
            "aerpaw_temporal_robustness_splits",
            set(throughput["split"].astype(str)) == {"final-30%-by-time", "first-half-to-second-half"}
            and throughput.groupby("split")["model"].nunique().eq(2).all()
            and protocol.get("throughput_protocol", {}).get("interleaved_blocks_used") is False,
            f"splits={sorted(throughput['split'].astype(str).unique())}",
        ),
        _check(
            "aerpaw_throughput_autocorrelation_reported",
            float(protocol.get("throughput_protocol", {}).get("throughput_lag1_autocorrelation", 0.0)) > 0.9,
            str(protocol.get("throughput_protocol", {})),
        ),
    ]


def _uav_to_uav_mmwave_checks(root: Path) -> list[dict]:
    base = root / "outputs/uav_to_uav_mmwave_validation"
    protocol_path = base / "uav_to_uav_mmwave_protocol.json"
    metrics_path = base / "uav_to_uav_link_model_metrics.csv"
    beta_path = base / "uav_to_uav_forestry_beta0_summary.csv"
    table_path = root / "paper/tables/generated/uav_to_uav_mmwave_table.tex"
    figure_path = root / "paper/figures/generated/uav_to_uav_mmwave_validation.pdf"
    required_paths = [protocol_path, metrics_path, beta_path, table_path, figure_path]
    missing = [str(path.relative_to(root)) for path in required_paths if not path.exists()]
    if missing:
        return [_check("uav_to_uav_mmwave_validation", False, f"missing={missing}")]

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    metrics = pd.read_csv(metrics_path)
    beta = pd.read_csv(beta_path)
    source_files = protocol.get("source_files") or [
        {
            "relative_path": protocol.get("relative_path"),
            "bytes": protocol.get("bytes"),
            "sha256": protocol.get("sha256"),
        }
    ]
    posix_paths, path_detail = _posix_manifest_paths(source_files)
    logistic = metrics.loc[metrics["model"] == "logistic A2A RF model"].iloc[0]
    baseline = metrics.loc[metrics["model"] == "training-prior baseline"].iloc[0]
    primary = beta[beta["support_scope"] == "all_pairs_within_measured_distance_support"]
    snr_thresholds = set(primary["snr_threshold_db"].astype(float))
    return [
        _check("uav_to_uav_mmwave_validation", True, "protocol, metrics, table, and figure exist"),
        _check("uav_to_uav_source_hash", protocol.get("sha256") == "9a8c0e4c2e473f7d5909040e71860ac0a6175c1821a9935b2c18f23c92202b33", str(protocol.get("sha256"))),
        _provenance_check("uav_to_uav_source_file", source_files, root),
        _check("uav_to_uav_source_posix_paths", posix_paths, path_detail),
        _check(
            "uav_to_uav_scope",
            "UAV-to-UAV" in protocol.get("scope", "")
            and "not same-site calibration" in protocol.get("scope", "")
            and "not" in protocol.get("scope", "")
            and "packet-delivery" in protocol.get("scope", ""),
            protocol.get("scope", ""),
        ),
        _check("uav_to_uav_holdout", int(logistic["test_n"]) >= 1700 and ">=33" in str(logistic["split"]), f"test_n={int(logistic['test_n'])}, split={logistic['split']}"),
        _check("uav_to_uav_link_gain", float(logistic["f1"]) > float(baseline["f1"]) and float(logistic["f1"]) >= 0.9, f"F1 {float(baseline['f1']):.3f}->{float(logistic['f1']):.3f}"),
        _check("uav_to_uav_pr_auc", float(logistic["pr_auc"]) >= 0.95, f"PR-AUC={float(logistic['pr_auc']):.3f}"),
        _check("uav_to_uav_beta_thresholds", snr_thresholds == {5.0, 7.0, 10.0}, str(sorted(snr_thresholds))),
        _check("uav_to_uav_primary_support", len(primary) == 3 and int(primary["samples"].min()) > 0, f"rows={len(primary)}"),
        _check(
            "uav_to_uav_consistent_link_rule",
            set(protocol.get("link_viability_rule", {})) == {"post_snr_db_min", "pRx_dbm_min"},
            str(protocol.get("link_viability_rule", {})),
        ),
    ]


def _miluv_checks(root: Path) -> list[dict]:
    base = root / "outputs/miluv_validation"
    protocol_path = base / "miluv_protocol.json"
    metrics_path = base / "miluv_metrics_per_seed.csv"
    summary_path = base / "miluv_metrics_summary.csv"
    trace_path = base / "miluv_measured_topology_trace.csv"
    manifest_path = root / "data/external_validation/raw/miluv/cirObstacles_3_random_0/manifest.json"
    required = [protocol_path, metrics_path, summary_path, trace_path, manifest_path]
    missing = [str(path.relative_to(root)) for path in required if not path.exists()]
    if missing:
        return [_check("miluv_validation", False, f"missing={missing}")]
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = pd.read_csv(metrics_path)
    summary = pd.read_csv(summary_path)
    source_files = manifest.get("files", [])
    protocol_sources = protocol.get("source_files", [])
    posix_paths, path_detail = _posix_manifest_paths(source_files)
    protocol_posix, protocol_path_detail = _posix_manifest_paths(protocol_sources)
    return [
        _check("miluv_validation", True, "measured UWB topology outputs exist"),
        _check("miluv_dataset_doi", protocol.get("dataset_doi") == "10.25452/figshare.plus.28386041.v1", str(protocol.get("dataset_doi"))),
        _check("miluv_three_uav", len(manifest.get("files", [])) == 6 and "three-UAV" in protocol.get("scope", ""), f"files={len(manifest.get('files', []))}"),
        _provenance_check("miluv_source_hashes", source_files, root),
        _provenance_check("miluv_protocol_manifest_hash", protocol_sources, root),
        _check("miluv_source_posix_paths", posix_paths, path_detail),
        _check("miluv_protocol_posix_paths", protocol_posix, protocol_path_detail),
        _check("miluv_frozen_transfer", "No MILUV samples" in protocol.get("training_domain", ""), protocol.get("training_domain", "")),
        _check(
            "miluv_threshold_predeclaration",
            "no threshold is optimized" in protocol.get("threshold_selection", ""),
            protocol.get("threshold_selection", ""),
        ),
        _check("miluv_seed_coverage", sorted(metrics["Seed"].astype(int).unique()) == [7, 17, 27, 37, 47], str(sorted(metrics["Seed"].astype(int).unique()))),
        _check("miluv_threshold_coverage", set(metrics["FPP_Threshold_dBm"].astype(float)) == {-90.0, -92.0, -95.0}, str(sorted(metrics["FPP_Threshold_dBm"].astype(float).unique()))),
        _check("miluv_fragmentation_events", int(summary["Fragmentation_Events"].max()) > 0, f"max={int(summary['Fragmentation_Events'].max())}"),
        _check("miluv_event_metrics", {"Alert_Event_Recall_mean", "Alert_Event_F1_mean"}.issubset(summary.columns), str(sorted(summary.columns))),
    ]


def _external_evidence_matrix_checks(root: Path) -> list[dict]:
    base = root / "outputs/external_evidence_matrix"
    required = {
        "matrix_csv": base / "evidence_matrix.csv",
        "matrix_tex": base / "evidence_matrix.tex",
        "source_manifest": base / "source_manifest.json",
        "claim_boundaries": base / "claim_boundaries.md",
        "protocol": base / "protocol.json",
        "paper_table": root / "paper/tables/generated/evidence_matrix.tex",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        return [_check("external_evidence_matrix", False, f"missing={missing}")]
    matrix = pd.read_csv(required["matrix_csv"])
    protocol = json.loads(required["protocol"].read_text(encoding="utf-8"))
    sources = json.loads(required["source_manifest"].read_text(encoding="utf-8"))
    aerpaw_protocol = json.loads((root / "outputs/aerpaw_cellular_validation/aerpaw_cellular_protocol.json").read_text(encoding="utf-8"))
    wines_protocol = json.loads((root / "outputs/uav_to_uav_mmwave_validation/uav_to_uav_mmwave_protocol.json").read_text(encoding="utf-8"))
    forestry_protocol = json.loads((root / "outputs/external_validation/external_validation_protocol.json").read_text(encoding="utf-8"))
    throughput = pd.read_csv(root / "outputs/aerpaw_cellular_validation/aerpaw_throughput_metrics.csv")
    violations = lint_external_claim_path(root / "paper/main.tex")
    expected = {
        "forestry_measured_motion", "forestry_counterfactual_radius", "aerpaw_lte_connection_state",
        "aerpaw_iperf_throughput", "wines_measured_peer_rf", "wines_forestry_transport_in_support",
        "wines_forestry_transport_clamped",
    }
    boundaries_complete = {
        "directly_measured", "derived_or_transported", "allowed_claim", "prohibited_claim",
        "source_identifier", "license", "sampling", "exclusions",
    }.issubset(matrix.columns) and not matrix[list({
        "directly_measured", "derived_or_transported", "allowed_claim", "prohibited_claim",
        "source_identifier", "license", "sampling", "exclusions",
    })].isna().any().any()
    forestry_raw = sources["sources"]["forestry"]["upstream_raw_sources"]
    return [
        _check("external_evidence_matrix", set(matrix.evidence_id) == expected, str(sorted(matrix.evidence_id.tolist()))),
        _check("external_evidence_boundaries_complete", boundaries_complete, str(sorted(matrix.columns.tolist()))),
        _provenance_check("external_evidence_input_protocol_hashes", protocol.get("input_protocols", []), root),
        _provenance_check("external_evidence_builder_hashes", protocol.get("source_files", []), root),
        _check("external_claim_boundary_check", not violations, json.dumps(violations)),
        _check("external_evidence_aerpaw_uav_to_infrastructure", "inter-UAV" in matrix.loc[matrix.evidence_id == "aerpaw_lte_connection_state", "prohibited_claim"].iloc[0], matrix.loc[matrix.evidence_id == "aerpaw_lte_connection_state", "allowed_claim"].iloc[0]),
        _check("external_evidence_aerpaw_chronological_negative_r2", aerpaw_protocol["throughput_protocol"]["primary_split"] == "final-30%-by-time" and not aerpaw_protocol["throughput_protocol"]["interleaved_blocks_used"] and bool((throughput.r2 < 0.0).any()) and aerpaw_protocol["throughput_outcomes"]["negative_r2_preserved"], str(throughput[["split", "model", "r2"]].to_dict("records"))),
        _check("external_evidence_wines_support_split", {"wines_forestry_transport_in_support", "wines_forestry_transport_clamped"}.issubset(set(matrix.evidence_id)), str(matrix[matrix.evidence_id.str.startswith("wines_")].split_or_support.tolist())),
        _check("external_evidence_wines_distance_support", wines_protocol["distance_support"] == {"training_m": [6.0, 32.0], "held_out_test_m": [33.0, 40.0], "support_overlap": False, "split_verified_from_rows": True} and set(wines_protocol["support_scopes_reported_separately"]) == {"full_trace_clamped", "all_pairs_within_measured_distance_support"}, str(wines_protocol["distance_support"])),
        _check("external_evidence_forestry_boundary", all(not bool(row.get("local_file_available")) for row in forestry_raw) and all(row.get("md5") for row in forestry_raw), "raw bags absent but authoritative MD5 records retained; derived trace verified separately"),
        _check("external_evidence_forestry_motion_only", forestry_protocol["measurement_boundary"] == {"motion_measured": True, "peer_rf_labels_present": False, "packet_labels_present": False, "radius_graph_counterfactual": True}, str(forestry_protocol["measurement_boundary"])),
    ]


def _miluv_adaptation_checks(root: Path) -> list[dict]:
    base = root / "outputs/miluv_adaptation"
    required = {
        "sequence_manifest": base / "sequence_manifest.json",
        "split_manifest": base / "split_manifest.csv",
        "metrics": base / "per_seed_or_scenario_metrics.csv",
        "summary": base / "summary.csv",
        "calibration_curves": base / "calibration_curves.pdf",
        "protocol": base / "protocol.json",
        "decision": base / "adaptation_decision.json",
        "table": root / "paper/tables/generated/miluv_adaptation_table.tex",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        return [_check("miluv_adaptation_outputs", False, f"missing={missing}")]
    metrics = pd.read_csv(required["metrics"])
    splits = pd.read_csv(required["split_manifest"])
    protocol = json.loads(required["protocol"].read_text(encoding="utf-8"))
    decision = json.loads(required["decision"].read_text(encoding="utf-8"))
    expected_models = {
        "Current-state persistence baseline", "Current-state ExtraTrees", "Kinetic-TopoGuard",
        "Source-Gated Kinetic-TopoGuard", "Shallow ML",
    }
    expected_protocols = {"zero_shot_frozen", "chronological_calibration", "few_shot_prediction_head"}
    return [
        _check("miluv_adaptation_outputs", True, "all requested adaptation outputs exist"),
        _provenance_check("miluv_adaptation_run_sources", protocol.get("source_files", []), root),
        _provenance_check("miluv_adaptation_decision_sources", decision.get("decision_source", []), root),
        _check("miluv_adaptation_seed_coverage", sorted(metrics.Seed.astype(int).unique()) == [7, 17, 27, 37, 47], str(sorted(metrics.Seed.astype(int).unique()))),
        _check("miluv_adaptation_model_coverage", set(metrics.Model) == expected_models, str(sorted(metrics.Model.unique()))),
        _check("miluv_adaptation_protocol_coverage", set(metrics.Protocol) == expected_protocols, str(sorted(metrics.Protocol.unique()))),
        _check("miluv_adaptation_no_test_selection", not splits.test_used_for_selection.astype(bool).any() and not bool(decision.get("test_used_for_model_calibration_threshold_or_staleness_selection", True)), "selection is restricted to chronological calibration prefixes"),
        _check("miluv_adaptation_guard", float(splits.guard_seconds.min()) >= float(protocol["split"]["guard_seconds"]), f"minimum_guard={float(splits.guard_seconds.min())}"),
        _check("miluv_adaptation_test_size", int(splits.test_rows.min()) >= int(protocol["split"]["minimum_test_rows"]), f"minimum_test_rows={int(splits.test_rows.min())}"),
        _check("miluv_loso_honest_unavailable", protocol["protocols"]["C"].get("status") == "unavailable" and decision["leave_one_scenario_out"].get("status") == "unavailable", str(protocol["protocols"]["C"])),
    ]


def _horizon_checks(root: Path) -> list[dict]:
    base = root / "outputs/horizon_sweep"
    protocol_path = base / "horizon_sweep_protocol.json"
    per_seed_path = base / "horizon_sweep_per_seed.csv"
    summary_path = base / "horizon_sweep_summary.csv"
    if not all(path.exists() for path in [protocol_path, per_seed_path, summary_path]):
        return [_check("horizon_sweep", False, "protocol/per-seed/summary output missing")]
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    per_seed = pd.read_csv(per_seed_path)
    return [
        _check("horizon_sweep", True, "six-horizon five-seed sweep exists"),
        _check("horizon_grid", protocol.get("horizon_steps") == [2, 4, 6, 10, 15, 20], str(protocol.get("horizon_steps"))),
        _check("horizon_seed_coverage", per_seed["seed"].nunique() == 5 and len(per_seed) == 90, f"seeds={per_seed['seed'].nunique()}, rows={len(per_seed)}"),
        _check("horizon_event_metrics", {"Alert_Event_Recall", "Alert_Event_F1"}.issubset(per_seed.columns), str(sorted(per_seed.columns))),
        _check("horizon_trajectory_reuse", "simulated once" in protocol.get("trajectory_reuse", ""), protocol.get("trajectory_reuse", "")),
    ]


def _factorial_checks(root: Path) -> list[dict]:
    base = root / "outputs/factorial_feature_ablation_20seed"
    protocol_path = base / "factorial_ablation_protocol.json"
    per_seed_path = base / "factorial_ablation_per_seed.csv"
    summary_path = base / "factorial_ablation_summary.csv"
    if not protocol_path.exists() or not per_seed_path.exists() or not summary_path.exists():
        return [_check("factorial_ablation", False, "20-seed protocol/per-seed/summary output missing")]
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    per_seed = pd.read_csv(per_seed_path)
    summary = pd.read_csv(summary_path)
    expected_seeds = [7, 17, 27, 37, 47, 57, 67, 77, 87, 97, 107, 117, 127, 137, 147, 157, 167, 177, 187, 197]
    metric_columns = {
        "MAE", "R2", "Risk_Precision", "Risk_Recall", "Risk_F1",
        "Alert_Event_Precision", "Alert_Event_Recall", "Alert_Event_F1",
        "False_Alert_Events_per_minute", "Risk_Brier", "Risk_ECE",
        "selected_threshold", "residual_scale", "feature_count",
        "Training_Time_s", "Inference_Time_ms",
    }
    completion_markers = list((base / "per_seed").glob("seed_*/_SUCCESS.json"))
    split_protocols = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (base / "per_seed").glob("seed_*/seed_protocol.json")
    ]
    split_clean = len(split_protocols) == 20 and all(
        not any(item.get("split", {}).get("run_id_intersections", {}).values())
        for item in split_protocols
    )
    aggregate_consistent = all(
        abs(
            float(per_seed.loc[per_seed["feature_sources"] == row["feature_sources"], "MAE"].mean())
            - float(row["MAE_mean"])
        ) < 1e-12
        for _, row in summary.iterrows()
    )
    return [
        _check("factorial_ablation", protocol.get("status") == "complete", str(protocol.get("status"))),
        _check("factorial_combinations", per_seed["feature_sources"].nunique() == 8 and len(per_seed) == 160, f"combinations={per_seed['feature_sources'].nunique()}, rows={len(per_seed)}"),
        _check("factorial_seed_coverage", sorted(per_seed["seed"].astype(int).unique()) == expected_seeds and protocol.get("seed_list") == expected_seeds, f"seeds={sorted(per_seed['seed'].astype(int).unique())}"),
        _check("factorial_current_state_model", set(per_seed.loc[per_seed["feature_sources"] == "current-only", "Model"]) == {"Current-state ExtraTrees"}, str(set(per_seed.loc[per_seed["feature_sources"] == "current-only", "Model"]))),
        _check("factorial_equal_learner", set(protocol.get("learner_hyperparameters", {})) == {"regressor", "classifier"}, str(protocol.get("learner_hyperparameters", {}))),
        _check("factorial_source_isolation", set(protocol.get("feature_groups", {})) == {"current", "graph", "topology", "kinematic"}, str(protocol.get("feature_groups", {}))),
        _check("factorial_metrics", metric_columns.issubset(per_seed.columns), str(sorted(metric_columns - set(per_seed.columns)))),
        _check("factorial_split_leakage", split_clean, f"seed_protocols={len(split_protocols)}"),
        _check("factorial_completion_markers", len(completion_markers) == 20, f"markers={len(completion_markers)}"),
        _check("factorial_seed_clustered_bootstrap", "Alert_Event_F1_seed_bootstrap_ci95_low" in summary.columns and set(summary["seeds"]) == {20}, str(sorted(summary.columns))),
        _check("factorial_aggregate_consistency", aggregate_consistent, f"summary_rows={len(summary)}"),
    ]


def _feature_significance_checks(root: Path) -> list[dict]:
    base = root / "outputs/factorial_feature_ablation_20seed"
    input_path = base / "factorial_ablation_per_seed.csv"
    tests_path = base / "paired_tests.csv"
    decision_path = base / "feature_source_decision.json"
    required_artifacts = [
        tests_path,
        base / "paired_tests.tex",
        base / "paired_effects.pdf",
        decision_path,
        root / "docs/eaai_revision/FEATURE_SOURCE_INTERPRETATION.md",
    ]
    missing = [relative_repo_path(path, root) for path in required_artifacts if not path.is_file()]
    if missing:
        return [_check("feature_source_significance", False, f"missing={missing}")]
    tests = pd.read_csv(tests_path)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    main = tests[
        (tests["candidate_feature_sources"] == "graph+topology+kinematic")
        & (tests["metric"] == "Alert_Event_F1")
    ]
    main_decision = str(main.iloc[0]["decision"]) if len(main) == 1 else "missing"
    return [
        _check("feature_source_significance", decision.get("status") == "complete", str(decision.get("status"))),
        _check("feature_source_seed_pairing", len(tests) == 64 and set(tests["n_pairs"]) == {20}, f"rows={len(tests)}, pairs={sorted(set(tests['n_pairs']))}"),
        _check("feature_source_comparison_coverage", tests["candidate_feature_sources"].nunique() == 8 and tests["metric"].nunique() == 8, f"candidates={tests['candidate_feature_sources'].nunique()}, metrics={tests['metric'].nunique()}"),
        _check("feature_source_input_hash", decision.get("input_sha256") == sha256_file(input_path), str(decision.get("input_sha256"))),
        _check("feature_source_primary_decision", main_decision == "statistically_supported_degradation", main_decision),
        _check("feature_source_no_equivalence_claim", "not assessed" in decision.get("decision_rule", {}).get("equivalence", ""), decision.get("decision_rule", {}).get("equivalence", "")),
    ]


def _packet_checks(root: Path) -> list[dict]:
    base = root / "outputs/packet_level_controller_v2"
    protocol_path = base / "packet_level_protocol.json"
    per_seed_path = base / "packet_metrics_per_seed.csv"
    paired_path = base / "packet_paired_tests.csv"
    if not protocol_path.exists() or not per_seed_path.exists() or not paired_path.exists():
        return [_check("packet_level_validation", False, "protocol/per-seed output missing")]
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    per_seed = pd.read_csv(per_seed_path)
    paired = pd.read_csv(paired_path)
    expected_metrics = {"PDR", "Deadline_Success", "Latency_P50_ms", "Latency_P95_ms", "Latency_P99_ms", "Mean_Queue_Occupancy", "No_Route_Drop_Rate", "Queue_Drop_Rate", "Contention_Deadline_Drop_Rate", "Link_Failure_Drop_Rate", "Intervention_Transition_Drop_Rate"}
    accounting_error = (
        per_seed["PDR"]
        + per_seed["No_Route_Drop_Rate"]
        + per_seed["Queue_Drop_Rate"]
        + per_seed["Contention_Deadline_Drop_Rate"]
        + per_seed["Link_Failure_Drop_Rate"]
        + per_seed["Intervention_Transition_Drop_Rate"]
        - 1.0
    ).abs().max()
    return [
        _check("packet_level_validation", True, "SimPy load sweep exists"),
        _check("packet_engine", protocol.get("engine") == "SimPy 4 discrete-event simulation", str(protocol.get("engine"))),
        _check("packet_load_grid", protocol.get("packets_per_tick") == [8, 16, 32, 64], str(protocol.get("packets_per_tick"))),
        _check("packet_seed_and_model_coverage", per_seed["seed"].nunique() == 5 and per_seed["Model"].nunique() == 3 and len(per_seed) == 60, f"rows={len(per_seed)}"),
        _check("packet_metrics", expected_metrics.issubset(per_seed.columns), str(sorted(per_seed.columns))),
        _check("packet_accounting", float(accounting_error) < 1e-9, f"max_error={float(accounting_error):.3e}"),
        _check("packet_paired_demand", per_seed.groupby(["seed", "Packets_per_tick"])["arrival_trace_signature"].nunique().eq(1).all(), "one arrival trace per seed/load across models"),
        _check("packet_seed_level_tests", paired["Paired_Seed_Count"].astype(int).eq(5).all(), f"rows={len(paired)}"),
        _check("packet_protocol_parameters", all(key in protocol for key in ["traffic_arrival_model", "queue_model", "packet_size_bytes", "link_capacity_mbps", "deadline_seconds", "routing", "retransmission", "drop_causes"]), str(sorted(protocol.keys()))),
        _check("packet_not_proxy", "PDR is produced by packet events" in protocol.get("scope", ""), protocol.get("scope", "")),
    ]


def _closed_loop_checks(root: Path) -> list[dict]:
    base = root / "outputs/closed_loop_controller_packet_v2"
    required = [
        base / "metrics_per_seed.csv",
        base / "metrics_summary.csv",
        base / "paired_tests.csv",
        base / "action_log.csv",
        base / "protocol.json",
        base / "closed_loop_tradeoff.pdf",
        base / "closed_loop_packet_tradeoff.pdf",
        root / "paper/tables/generated/closed_loop_packet_table.tex",
        root / "paper/figures/generated/closed_loop_packet_tradeoff.pdf",
    ]
    missing = [relative_repo_path(path, root) for path in required if not path.is_file()]
    if missing:
        return [_check("closed_loop_controller", False, f"missing={missing}")]
    metrics = pd.read_csv(required[0])
    paired = pd.read_csv(required[2])
    actions = pd.read_csv(required[3])
    protocol = json.loads(required[4].read_text(encoding="utf-8"))
    pairing = actions.groupby(["seed", "run_id", "time_index", "traffic_load"]).agg(
        hashes=("initial_condition_hash", "nunique"),
        policies=("policy", "nunique"),
    )
    no_action = actions[actions["policy"] == "No intervention"]
    manifest = verify_manifest(protocol.get("source_files", []), root)
    action_config = protocol.get("action_config", {})
    return [
        _check("closed_loop_controller", protocol.get("status") == "complete", str(protocol.get("status"))),
        _check("closed_loop_policy_coverage", metrics["Policy"].nunique() == 6 and len(metrics) == 270, f"policies={metrics['Policy'].nunique()}, rows={len(metrics)}"),
        _check("closed_loop_seed_load_density_coverage", metrics["seed"].nunique() == 5 and sorted(metrics["Traffic_Load"].unique()) == [8, 16, 32] and sorted(metrics["Topology_Density_Nodes"].unique()) == [10, 20, 30], "5 seeds, 3 loads, 3 densities"),
        _check("closed_loop_paired_initial_conditions", bool((pairing["hashes"] == 1).all() and (pairing["policies"] == 6).all()), f"paired_ticks={len(pairing)}"),
        _check("closed_loop_no_future_information", not actions["future_label_read_by_policy"].astype(bool).any() and (actions["information_time_index"] == actions["time_index"]).all() and protocol.get("per_seed_protocol") and all(not item.get("test_used_for_policy_tuning", True) for item in protocol["per_seed_protocol"]), "causal score/action interface and no test tuning"),
        _check("closed_loop_motion_limits", actions["commanded_max_speed_mps"].max() <= float(action_config.get("max_speed_mps", -1)) + 1e-8 and actions["commanded_max_acceleration_mps2"].max() <= float(action_config.get("max_acceleration_mps2", -1)) + 1e-8, f"speed={actions['commanded_max_speed_mps'].max():.6g}, acceleration={actions['commanded_max_acceleration_mps2'].max():.6g}"),
        _check("closed_loop_no_intervention_action", not no_action["action_started"].astype(bool).any() and not no_action["action_active"].astype(bool).any() and float(no_action["relay_travel_m"].sum()) == 0.0, "no actions or relay travel"),
        _check("closed_loop_packet_metrics", {"Packet_Delivery_Ratio", "Packet_Deadline_Success", "Latency_P50_ms", "Latency_P95_ms", "Latency_P99_ms", "Mean_Queue_Occupancy", "Queue_Delay_ms", "End_to_End_Delay_ms", "No_Route_Drop_Rate", "Queue_Overflow_Drop_Rate", "Deadline_Miss_Drop_Rate", "Link_Failure_Drop_Rate", "Intervention_Transition_Drop_Rate"}.issubset(metrics.columns), "packet outcomes are separate from connectivity proxies"),
        _check("closed_loop_drop_accounting", float(metrics["Drop_Accounting_Error"].max()) < 1e-9, f"max_error={float(metrics['Drop_Accounting_Error'].max()):.3e}"),
        _check("closed_loop_packet_protocol", all(key in protocol.get("packet_model", {}) for key in ["arrival_model", "packet_deadline_s", "packet_bytes", "bitrate_mbps", "queue_limit", "routing", "max_retransmissions", "drop_accounting"]), str(protocol.get("packet_model", {}))),
        _check("closed_loop_paired_tests", paired["Paired_Seed_Count"].astype(int).eq(5).all() and "Engineering_Benefit_Supported" in paired.columns, f"rows={len(paired)}"),
        _check("closed_loop_source_hashes", bool(manifest.get("valid")), str(manifest)),
    ]


def _latency_checks(root: Path) -> list[dict]:
    base = root / "outputs/end_to_end_latency"
    protocol_path = base / "latency_protocol.json"
    samples_path = base / "latency_samples.csv"
    summary_path = base / "latency_summary.csv"
    if not all(path.exists() for path in [protocol_path, samples_path, summary_path]):
        return [_check("end_to_end_latency", False, "protocol/sample/summary output missing")]
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    samples = pd.read_csv(samples_path)
    summary = pd.read_csv(summary_path)
    return [
        _check("end_to_end_latency", True, "full host-side path timed"),
        _check("latency_seed_coverage", sorted(samples["seed"].astype(int).unique()) == [7, 17, 27, 37, 47], str(sorted(samples["seed"].astype(int).unique()))),
        _check("latency_stage_coverage", len(protocol.get("timed_stages", [])) == 3 and "pairwise distance" in protocol["timed_stages"][0], str(protocol.get("timed_stages"))),
        _check("latency_tail_metrics", {"total_p95_ms", "total_p99_ms"}.issubset(summary.columns), str(sorted(summary.columns))),
        _check("latency_platform", all(protocol.get("platform", {}).get(key) for key in ["system", "python"]), str(protocol.get("platform", {}))),
    ]


def _edge_runtime_checks(root: Path) -> list[dict]:
    base = root / "outputs/edge_runtime_benchmark"
    required = [
        base / "latency_samples.csv", base / "latency_summary.csv",
        base / "memory_summary.csv", base / "model_sizes.csv",
        base / "environment.json", base / "protocol.json",
        base / "runtime_scaling.pdf", root / "paper/tables/generated/runtime_table.tex",
    ]
    missing = [str(path.relative_to(root)) for path in required if not path.exists()]
    if missing:
        return [_check("edge_runtime_benchmark", False, f"missing={missing}")]
    samples = pd.read_csv(base / "latency_samples.csv")
    summary = pd.read_csv(base / "latency_summary.csv")
    sizes = pd.read_csv(base / "model_sizes.csv")
    protocol = json.loads((base / "protocol.json").read_text(encoding="utf-8"))
    expected_models = {
        "Current-state ExtraTrees", "Kinetic-TopoGuard",
        "Source-Gated Kinetic-TopoGuard", "FANET-TopoGNN",
    }
    expected_stages = {
        "pairwise_distance", "radio_link_graph", "graph_statistics",
        "persistence_image", "feature_assembly", "count_inference",
        "risk_inference", "calibration", "controller_decision", "total_host_loop",
    }
    timed = samples.loc[samples["phase"].eq("timed")]
    return [
        _check("edge_runtime_benchmark", True, "stage-resolved host-only timing exists"),
        _check("edge_runtime_model_coverage", set(samples["model"]) == expected_models, str(sorted(samples["model"].unique()))),
        _check("edge_runtime_node_coverage", set(samples["n_nodes"].astype(int)) >= {10, 20, 30}, str(sorted(samples["n_nodes"].unique()))),
        _check("edge_runtime_stage_coverage", set(samples["stage"]) == expected_stages, str(sorted(samples["stage"].unique()))),
        _check("edge_runtime_sample_count", timed.groupby(["model", "thread_mode", "n_nodes", "stage"]).size().eq(30).all(), "30 timed samples per cell"),
        _check("edge_runtime_tail_metrics", {"p50_ms", "p90_ms", "p95_ms", "p99_ms", "mean_ms", "std_ms", "sample_count"}.issubset(summary.columns), str(sorted(summary.columns))),
        _check("edge_runtime_serialization", sizes["prediction_consistent"].astype(bool).all() and sizes["file_size_bytes"].gt(0).all(), f"models={len(sizes)}"),
        _check("edge_runtime_scope_boundary", protocol.get("not_a_sensor_to_actuator_measurement") is True and protocol.get("jetson", {}).get("measured") is False, "host-only; no Jetson claim"),
    ]


def _operating_point_checks(root: Path) -> list[dict]:
    base = root / "outputs/operating_point"
    protocol_path = base / "operating_point_protocol.json"
    summary_path = base / "operating_point_summary.csv"
    table_path = root / "paper/tables/generated/operating_point_table.tex"
    figure_path = root / "paper/figures/generated/operating_point_selection.pdf"
    required_paths = [protocol_path, summary_path, table_path, figure_path]
    missing = [str(path.relative_to(root)) for path in required_paths if not path.exists()]
    if missing:
        return [_check("operating_point_selection", False, f"missing={missing}")]

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    summary = pd.read_csv(summary_path)
    policies = set(summary["Policy"].astype(str))
    return [
        _check("operating_point_selection", True, "protocol, summary, table, and figure exist"),
        _check("operating_point_rule", "validation event F1" in protocol.get("selection_rule", ""), protocol.get("selection_rule", "")),
        _check("operating_point_policies", policies == {"deployable", "strict"}, str(sorted(policies))),
        _check("operating_point_validation_test_split", protocol.get("selection_split") == "validation" and protocol.get("evaluation_split") == "test", str(protocol)),
        _check(
            "operating_point_event_columns",
            {
                "Validation_False_Alert_Budget_per_minute_mean",
                "Test_Alert_Event_Precision_mean",
                "Test_Alert_Event_Recall_mean",
                "Test_Alert_Event_F1_mean",
                "Test_False_Alert_Events_per_minute_mean",
            }.issubset(summary.columns),
            str(sorted(summary.columns)),
        ),
    ]


def _dashboard_checks(root: Path) -> list[dict]:
    base = root / "outputs/digital_twin_dashboard"
    index_path = base / "index.html"
    payload_path = base / "dashboard_payload.json"
    assets = [
        base / "assets/operating_point_selection.png",
        base / "assets/uav_to_uav_mmwave_validation.png",
        base / "assets/aerpaw_cellular_validation.png",
        base / "assets/horizon_sweep.png",
        base / "assets/factorial_feature_ablation.png",
        base / "assets/packet_level_controller.png",
        base / "assets/miluv_validation.png",
    ]
    required_paths = [index_path, payload_path, *assets]
    missing = [str(path.relative_to(root)) for path in required_paths if not path.exists()]
    if missing:
        return [_check("digital_twin_dashboard", False, f"missing={missing}")]

    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    html_text = index_path.read_text(encoding="utf-8")
    return [
        _check("digital_twin_dashboard", True, "index, payload, and assets exist"),
        _check("dashboard_confirmatory_binding", payload.get("experiment", {}).get("confirmatory_seeds") == 20, str(payload.get("experiment", {}))),
        _check("dashboard_neural_extension_binding", payload.get("experiment", {}).get("neural_extension_seeds") == 5, str(payload.get("experiment", {}))),
        _check("dashboard_a2a_binding", payload.get("measured_validation", {}).get("uav_to_uav_60ghz", {}).get("model_f1", 0) >= 0.9, str(payload.get("measured_validation", {}).get("uav_to_uav_60ghz", {}))),
        _check(
            "dashboard_aerpaw_robustness_binding",
            payload.get("measured_validation", {}).get("aerpaw_cellular", {}).get("robustness_r2", 1.0) < 0.0,
            str(payload.get("measured_validation", {}).get("aerpaw_cellular", {})),
        ),
        _check(
            "dashboard_miluv_binding",
            payload.get("measured_validation", {}).get("miluv_measured_topology", {}).get("fragmentation_events", 0) > 0,
            str(payload.get("measured_validation", {}).get("miluv_measured_topology", {})),
        ),
        _check("dashboard_interactive_control", "renderPolicy" in html_text and "button.dataset.policy" in html_text, "operating-point JavaScript control present"),
        _check("dashboard_scope_label", "Offline FANET Predictive-Twin Replay" in html_text and "not a live" in html_text, "offline replay boundary present"),
    ]


def _format_checks(root: Path) -> list[dict]:
    pdfinfo = shutil.which("pdfinfo")
    pdftotext = shutil.which("pdftotext")
    paths = {
        "single_column": root / "paper/main.pdf",
        "anonymous_single_column": root / "paper/main_anonymized.pdf",
        "elsevier_5p": root / "paper/main_5p.pdf",
        "accepted_reference": root / "1-s2.0-S0952197626017422-main.pdf",
    }
    missing = [str(path.relative_to(root)) for path in paths.values() if not path.exists()]
    if missing or pdfinfo is None or pdftotext is None:
        return [_check("format_aware_page_count", False, f"missing={missing}, pdfinfo={pdfinfo}, pdftotext={pdftotext}")]

    pages: dict[str, int] = {}
    for name, path in paths.items():
        info = subprocess.run(
            [pdfinfo, str(path)],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        ).stdout
        match = re.search(r"^Pages:\s+(\d+)$", info, flags=re.MULTILINE)
        if match is None:
            return [_check("format_aware_page_count", False, f"unable to parse {path.relative_to(root)}")]
        pages[name] = int(match.group(1))
    words = {}
    for name in ["elsevier_5p", "accepted_reference"]:
        text = subprocess.run(
            [pdftotext, str(paths[name]), "-"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        ).stdout
        words[name] = len(re.findall(r"\b[\w'-]+\b", text))
    return [
        _check("single_column_page_count", pages["single_column"] <= 35 and pages["anonymous_single_column"] <= 35, str(pages)),
        _check(
            "elsevier_5p_page_count",
            pages["elsevier_5p"] <= pages["accepted_reference"] + 3
            and pages["elsevier_5p"] < pages["single_column"],
            str(pages),
        ),
        _check(
            "format_normalized_word_count",
            0.5 * words["accepted_reference"] <= words["elsevier_5p"] <= 1.2 * words["accepted_reference"],
            str(words),
        ),
    ]


def _manuscript_claim_checks(root: Path) -> list[dict]:
    manuscript_path = root / "paper/main.tex"
    if not manuscript_path.exists():
        return [_check("manuscript_claims", False, "paper/main.tex missing")]
    text = manuscript_path.read_text(encoding="utf-8")
    lowered = text.lower()
    stale = [
        token
        for token in [
            "75.2",
            "0.713",
            "union-find detection oracle",
            "rf-calibrated external",
            "real-time digital twin dashboard",
        ]
        if token in lowered
    ]
    required_concepts = {
        "fragmentation-event metric": "event $f_1$",
        "temporally correlated radio": "temporally correlated",
        "physical relay limits": "maximum acceleration",
        "MILUV measured topology": "miluv",
        "AERPAW negative throughput finding": "negative",
        "transported mmWave sensitivity": "transported",
        "packet-level validation": "simpy",
        "factorial ablation": "factorial",
        "offline replay boundary": "offline replay",
    }
    missing = [name for name, phrase in required_concepts.items() if phrase not in lowered]
    return [
        _check("manuscript_stale_claims", not stale, str(stale)),
        _check("manuscript_required_evidence", not missing, str(missing)),
        _check("manuscript_no_field_deployment_claim", "field-deployed predictive twin" not in lowered, "bounded deployment language"),
    ]


def _anonymity_checks(root: Path) -> list[dict]:
    config_path = root / "scripts" / "anonymity_patterns.json"
    main_tex = root / "paper" / "main.tex"
    anonymous_pdf = root / "paper" / "main_anonymized.pdf"
    package_dir = root / "anonymous_supplementary"
    try:
        config = load_pattern_config(config_path)
        if not main_tex.is_file():
            raise AnonymityScanError(f"anonymous TeX source is missing: {main_tex}")
        effective_tex = extract_anonymous_tex(main_tex.read_text(encoding="utf-8"))
        tex_findings = scan_text(effective_tex, "paper/main_anonymized.tex", config)
        tex_detail = "no identity patterns in the effective anonymous TeX branch"
        if tex_findings:
            tex_detail = format_failures({"errors": [], "findings": tex_findings})

        pdf_report = scan_paths([anonymous_pdf], root=root, config_path=config_path)
        package_report = scan_paths([package_dir], root=root, config_path=config_path)
        return [
            _check("anonymous_tex_identity_scan", not tex_findings, tex_detail),
            _check(
                "anonymous_pdf_identity_scan",
                pdf_report["status"] == "pass",
                "no identity patterns in extracted PDF text"
                if pdf_report["status"] == "pass"
                else format_failures(pdf_report),
            ),
            _check(
                "anonymous_supplement_identity_scan",
                package_report["status"] == "pass",
                "no identity patterns in anonymous supplementary files"
                if package_report["status"] == "pass"
                else format_failures(package_report),
            ),
        ]
    except (OSError, AnonymityScanError) as exc:
        return [
            _check("anonymous_tex_identity_scan", False, str(exc)),
            _check("anonymous_pdf_identity_scan", False, str(exc)),
            _check("anonymous_supplement_identity_scan", False, str(exc)),
        ]


def _package_checks(root: Path) -> list[dict]:
    zip_path = root / "anonymous_supplementary.zip"
    package_dir = root / "anonymous_supplementary"
    if not zip_path.exists() or not package_dir.exists():
        return [_check("anonymous_package", False, "ZIP or unpacked package missing")]
    import zipfile

    with zipfile.ZipFile(zip_path) as archive:
        crc_error = archive.testzip()
        names = set(archive.namelist())
    required_suffixes = {
        "outputs/horizon_sweep/horizon_sweep_summary.csv",
        "outputs/factorial_feature_ablation/factorial_ablation_summary.csv",
        "outputs/packet_level_controller/packet_metrics_summary.csv",
        "outputs/end_to_end_latency/latency_summary.csv",
        "outputs/miluv_validation/miluv_metrics_summary.csv",
        "scripts/run_miluv_validation.py",
    }
    packaged = {
        suffix
        for suffix in required_suffixes
        if any(name.endswith(suffix) for name in names)
    }
    newest_input = max(
        path.stat().st_mtime
        for path in [
            root / "paper/main_anonymized.pdf",
            root / "outputs/paper_like_submission/summary.json",
            root / "outputs/publication_compact/summary.json",
            root / "outputs/publication_neural_5seed_extension/summary.json",
            root / "outputs/external_validation/external_validation_protocol.json",
            root / "outputs/miluv_validation/miluv_protocol.json",
            root / "outputs/digital_twin_dashboard/dashboard_payload.json",
        ]
        if path.exists()
    )
    return [
        _check("anonymous_package_crc", crc_error is None, str(crc_error)),
        _check("anonymous_package_evidence", packaged == required_suffixes, str(sorted(required_suffixes - packaged))),
        _check("anonymous_package_freshness", zip_path.stat().st_mtime >= newest_input, f"zip={zip_path.stat().st_mtime}, evidence={newest_input}"),
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
    metadata_files = [root / ".zenodo.json", root / "CITATION.cff", root / "paper/title_page.tex"]
    encoding_files = [*metadata_files, root / "paper/cas-refs.bib"]
    mojibake = re.compile("[\\ufffd\\u00c3\\u00c2\\u00c4\\u00c5\\u00e2]|\\u00ef\\u00bf\\u00bd")
    portal_metadata = root / "EAAI_PORTAL_METADATA.md"
    cover_letter = root / "EAAI_COVER_LETTER.md"
    portal_files = [
        root / "paper/main_anonymized.pdf",
        root / "paper/title_page.pdf",
        root / "anonymous_supplementary.zip",
        root / "paper/figures/plantuml/graphical_abstract.pdf",
    ]
    author_orcid = "0000-0001-9259-" + "7112"
    return [
        _check("elsevier_highlight_count", 3 <= len(highlights) <= 5, f"count={len(highlights)}"),
        _check("elsevier_highlight_length", bool(highlights) and all(len(item) <= 85 for item in highlights), str([len(item) for item in highlights])),
        _check("perslay_bibliography", bool(perslay) and all(re.search(rf"\b{field}\s*=", perslay) for field in required_perslay), str(required_perslay)),
        _check(
            "external_dataset_bibliography",
            all(item in bibliography for item in ["10.5281/zenodo.14701641", "aerpawDataset22", "aerpawDataset23", "winesUavToUav60GHz", "10.25452/figshare.plus.28386041.v1"]),
            "Zenodo, AERPAW, UAV-to-UAV mmWave, and MILUV entries present",
        ),
        _check("title_page_declarations", all(label in title_page for label in ["CRediT", "Funding", "competing interest", "Data and code availability", "generative AI"]), "required declarations present"),
        _check("metadata_encoding", not any(mojibake.search(path.read_text(encoding="utf-8")) for path in encoding_files), "no common mojibake markers"),
        _check("orcid_consistency", all(author_orcid in path.read_text(encoding="utf-8") for path in metadata_files), "ORCID present in title page, CFF, and Zenodo metadata"),
        _check("portal_metadata_package", portal_metadata.exists() and cover_letter.exists(), "metadata and cover-letter files exist"),
        _check("portal_upload_files", all(path.exists() and path.stat().st_size > 0 for path in portal_files), str([path.name for path in portal_files])),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit evidence and journal-package readiness.")
    parser.add_argument("--output", type=Path, default=ROOT / "submission_readiness_audit.json")
    args = parser.parse_args()
    checks = [
        *_compact_checks(ROOT),
        *_confirmatory_checks(ROOT),
        *_neural_extension_checks(ROOT),
        *_explainability_checks(ROOT),
        *_external_checks(ROOT),
        *_external_evidence_matrix_checks(ROOT),
        *_aerpaw_cellular_checks(ROOT),
        *_uav_to_uav_mmwave_checks(ROOT),
        *_miluv_checks(ROOT),
        *_miluv_adaptation_checks(ROOT),
        *_horizon_checks(ROOT),
        *_factorial_checks(ROOT),
        *_feature_significance_checks(ROOT),
        *_packet_checks(ROOT),
        *_closed_loop_checks(ROOT),
        *_latency_checks(ROOT),
        *_edge_runtime_checks(ROOT),
        *_operating_point_checks(ROOT),
        *_dashboard_checks(ROOT),
        *_format_checks(ROOT),
        *_manuscript_claim_checks(ROOT),
        *_anonymity_checks(ROOT),
        *_bibliography_checks(ROOT),
        *_metadata_checks(ROOT),
        *_submission_text_checks(ROOT),
        *_package_checks(ROOT),
    ]
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
