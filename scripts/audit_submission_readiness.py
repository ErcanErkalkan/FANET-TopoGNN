from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    aggregate_summary_path = root / "outputs/publication_neural_5seed_extension/summary.json"
    metrics_path = root / "outputs/publication_neural_5seed_extension/metrics_overall.csv"
    per_seed_path = root / "outputs/publication_neural_5seed_extension/per_seed_metrics.csv"
    table_path = root / "paper/tables/generated/neural_seed_extension_table.tex"
    required_paths = [raw_summary_path, aggregate_summary_path, metrics_path, per_seed_path, table_path]
    missing = [str(path.relative_to(root)) for path in required_paths if not path.exists()]
    if missing:
        return [_check("neural_seed_extension", False, f"missing={missing}")]

    raw_summary = json.loads(raw_summary_path.read_text(encoding="utf-8"))
    aggregate_summary = json.loads(aggregate_summary_path.read_text(encoding="utf-8"))
    metrics = pd.read_csv(metrics_path)
    per_seed = pd.read_csv(per_seed_path)
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
    return [
        _check("neural_seed_extension", True, "raw two-seed run and five-seed aggregate exist"),
        _check("neural_extension_raw_backend", raw_summary.get("n_seeds") == 2 and raw_summary.get("torch_available") is True and raw_summary.get("surrogate_used") is False, str(raw_summary.get("model_backend", {}))),
        _check("neural_extension_seed_count", aggregate_summary.get("n_seeds") == 5 and aggregate_summary.get("seeds") == [7, 17, 27, 37, 47], str(aggregate_summary.get("seeds"))),
        _check(
            "neural_extension_cache_versions",
            set(aggregate_summary.get("source_cache_versions", {}).values()) == {CACHE_VERSION},
            str(aggregate_summary.get("source_cache_versions", {})),
        ),
        _check("neural_extension_model_coverage", len(neural) == 9 and all(int(value) == 5 for value in neural["seed_count"]), f"models={len(neural)}, seeds={sorted(set(neural['seed_count'].astype(int)))}"),
        _check("neural_extension_pytorch", set(neural["backends"].astype(str)) == {"pytorch"}, str(sorted(set(neural["backends"].astype(str))))),
        _check("neural_extension_per_seed_coverage", all(counts.get(model) == 5 for model in neural["Model"].astype(str)), str(counts)),
        _check(
            "neural_extension_event_metrics",
            {"Alert_Event_Recall_mean", "Alert_Event_F1_mean"}.issubset(metrics.columns),
            str(sorted(metrics.columns)),
        ),
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
    source_hashes_ok = all(
        (root / item["relative_path"]).is_file()
        and _sha256(root / item["relative_path"]) == item["sha256"]
        for item in protocol.get("sources", {}).values()
    )

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
        _check("aerpaw_source_hashes", source_hashes_ok, str(source_hashes_ok)),
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
    source_path = root / protocol.get("relative_path", "")
    logistic = metrics.loc[metrics["model"] == "logistic A2A RF model"].iloc[0]
    baseline = metrics.loc[metrics["model"] == "training-prior baseline"].iloc[0]
    primary = beta[beta["support_scope"] == "all_pairs_within_measured_distance_support"]
    snr_thresholds = set(primary["snr_threshold_db"].astype(float))
    return [
        _check("uav_to_uav_mmwave_validation", True, "protocol, metrics, table, and figure exist"),
        _check("uav_to_uav_source_hash", protocol.get("sha256") == "9a8c0e4c2e473f7d5909040e71860ac0a6175c1821a9935b2c18f23c92202b33", str(protocol.get("sha256"))),
        _check(
            "uav_to_uav_source_file",
            source_path.is_file() and _sha256(source_path) == protocol.get("sha256"),
            str(source_path.relative_to(root) if source_path.is_relative_to(root) else source_path),
        ),
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
    source_checks = []
    for item in manifest.get("files", []):
        source = root / item["relative_path"]
        source_checks.append(
            source.is_file()
            and source.stat().st_size == int(item["bytes"])
            and _sha256(source) == item["sha256"]
        )
    return [
        _check("miluv_validation", True, "measured UWB topology outputs exist"),
        _check("miluv_dataset_doi", protocol.get("dataset_doi") == "10.25452/figshare.plus.28386041.v1", str(protocol.get("dataset_doi"))),
        _check("miluv_three_uav", len(manifest.get("files", [])) == 6 and "three-UAV" in protocol.get("scope", ""), f"files={len(manifest.get('files', []))}"),
        _check("miluv_source_hashes", len(source_checks) == 6 and all(source_checks), str(source_checks)),
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
    base = root / "outputs/factorial_feature_ablation"
    protocol_path = base / "factorial_ablation_protocol.json"
    per_seed_path = base / "factorial_ablation_per_seed.csv"
    if not protocol_path.exists() or not per_seed_path.exists():
        return [_check("factorial_ablation", False, "protocol/per-seed output missing")]
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    per_seed = pd.read_csv(per_seed_path)
    source_isolation = protocol.get("source_isolation", {})
    return [
        _check("factorial_ablation", True, "equal-learner full factorial exists"),
        _check("factorial_combinations", per_seed["feature_sources"].nunique() == 8 and len(per_seed) == 40, f"combinations={per_seed['feature_sources'].nunique()}, rows={len(per_seed)}"),
        _check("factorial_seed_coverage", per_seed["seed"].nunique() == 5, f"seeds={per_seed['seed'].nunique()}"),
        _check("factorial_equal_learner", "for every row" in protocol.get("learner", ""), protocol.get("learner", "")),
        _check("factorial_source_isolation", set(source_isolation) == {"state_context", "graph", "topology", "kinematic"}, str(source_isolation)),
        _check("factorial_event_metric", "Alert_Event_F1" in per_seed.columns, str(sorted(per_seed.columns))),
    ]


def _packet_checks(root: Path) -> list[dict]:
    base = root / "outputs/packet_level_controller"
    protocol_path = base / "packet_level_protocol.json"
    per_seed_path = base / "packet_metrics_per_seed.csv"
    if not protocol_path.exists() or not per_seed_path.exists():
        return [_check("packet_level_validation", False, "protocol/per-seed output missing")]
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    per_seed = pd.read_csv(per_seed_path)
    expected_metrics = {"PDR", "Mean_Delay_ms", "P95_Delay_ms", "Queue_Drop_Rate"}
    accounting_error = (
        per_seed["PDR"]
        + per_seed["No_Route_Drop_Rate"]
        + per_seed["Queue_Drop_Rate"]
        + per_seed["Contention_Deadline_Drop_Rate"]
        - 1.0
    ).abs().max()
    return [
        _check("packet_level_validation", True, "SimPy load sweep exists"),
        _check("packet_engine", protocol.get("engine") == "SimPy 4 discrete-event simulation", str(protocol.get("engine"))),
        _check("packet_load_grid", protocol.get("packets_per_tick") == [8, 16, 32, 64], str(protocol.get("packets_per_tick"))),
        _check("packet_seed_and_model_coverage", per_seed["seed"].nunique() == 5 and per_seed["Model"].nunique() == 3 and len(per_seed) == 60, f"rows={len(per_seed)}"),
        _check("packet_metrics", expected_metrics.issubset(per_seed.columns), str(sorted(per_seed.columns))),
        _check("packet_accounting", float(accounting_error) < 1e-9, f"max_error={float(accounting_error):.3e}"),
        _check("packet_paired_demand", "Identical packet" in protocol.get("scope", ""), protocol.get("scope", "")),
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
        _check("orcid_consistency", all("0000-0001-9259-7112" in path.read_text(encoding="utf-8") for path in metadata_files), "ORCID present in title page, CFF, and Zenodo metadata"),
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
        *_external_checks(ROOT),
        *_aerpaw_cellular_checks(ROOT),
        *_uav_to_uav_mmwave_checks(ROOT),
        *_miluv_checks(ROOT),
        *_horizon_checks(ROOT),
        *_factorial_checks(ROOT),
        *_packet_checks(ROOT),
        *_latency_checks(ROOT),
        *_operating_point_checks(ROOT),
        *_dashboard_checks(ROOT),
        *_format_checks(ROOT),
        *_manuscript_claim_checks(ROOT),
        *_metadata_checks(ROOT),
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
