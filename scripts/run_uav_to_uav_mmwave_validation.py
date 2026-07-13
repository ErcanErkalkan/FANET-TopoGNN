from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fanet.external_validation import load_flight_trace_csv

RAW_PATH = ROOT / "data" / "external_validation" / "raw" / "uav_to_uav_mmwave" / "dataset.csv"
DEFAULT_TRACE = ROOT / "data" / "external_validation" / "derived" / "forestry_multidrone_trace.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "uav_to_uav_mmwave_validation"
DEFAULT_MANUSCRIPT = ROOT / "paper"
DATASET_URL = "https://github.com/wineslab/uav-to-uav-60-ghz-channel-model"
RAW_URL = "https://raw.githubusercontent.com/wineslab/uav-to-uav-60-ghz-channel-model/master/dataset.csv"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_input(path: Path) -> None:
    if path.exists():
        return
    raise FileNotFoundError(
        f"Missing {path.relative_to(ROOT)}\n"
        "Download with:\n"
        "New-Item -ItemType Directory -Force data\\external_validation\\raw\\uav_to_uav_mmwave | Out-Null\n"
        f"curl.exe -L --fail -o data\\external_validation\\raw\\uav_to_uav_mmwave\\dataset.csv {RAW_URL}"
    )


def _load_a2a(path: Path, snr_db: float, prx_dbm: float) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"distance", "altitude", "tx_beam", "rx_beam", "post_snr", "pRx", "path_loss"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"UAV-to-UAV mmWave dataset is missing columns: {sorted(missing)}")
    work = frame.dropna(subset=list(required)).copy()
    work["link_viable"] = ((work["post_snr"] >= snr_db) & (work["pRx"] >= prx_dbm)).astype(int)
    return work


def _score(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    return {
        "test_n": float(len(y_true)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, np.clip(y_score, 0.0, 1.0))),
    }


def _link_model_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    feature_cols = ["distance", "altitude", "tx_beam", "rx_beam", "tx_gain_idx", "rx_rf_gain_idx", "rx_if_gain_idx"]
    train = frame["distance"] < 33
    test = ~train
    y_train = frame.loc[train, "link_viable"].to_numpy(dtype=int)
    y_test = frame.loc[test, "link_viable"].to_numpy(dtype=int)
    rows: list[dict[str, float | str]] = []

    prior = np.full(len(y_test), float(y_train.mean()))
    rows.append(
        {
            "model": "training-prior baseline",
            "split": "held-out long distances >=33 m",
            "train_n": float(train.sum()),
            **_score(y_test, prior),
        }
    )

    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    )
    logistic.fit(frame.loc[train, feature_cols], y_train)
    logistic_scores = logistic.predict_proba(frame.loc[test, feature_cols])[:, 1]
    rows.append(
        {
            "model": "logistic A2A RF model",
            "split": "held-out long distances >=33 m",
            "train_n": float(train.sum()),
            **_score(y_test, logistic_scores),
        }
    )

    forest = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, class_weight="balanced", random_state=42)
    forest.fit(frame.loc[train, feature_cols], y_train)
    forest_scores = forest.predict_proba(frame.loc[test, feature_cols])[:, 1]
    rows.append(
        {
            "model": "random-forest A2A RF model",
            "split": "held-out long distances >=33 m",
            "train_n": float(train.sum()),
            **_score(y_test, forest_scores),
        }
    )
    return pd.DataFrame(rows)


def _distance_success_table(
    frame: pd.DataFrame,
    snr_thresholds: tuple[float, ...],
    prx_dbm: float,
) -> pd.DataFrame:
    rows = []
    for threshold in snr_thresholds:
        work = frame.copy()
        work["link_viable"] = (
            (work["post_snr"] >= threshold) & (work["pRx"] >= prx_dbm)
        ).astype(int)
        grouped = work.groupby("distance", as_index=False).agg(
            success_probability=("link_viable", "mean"),
            mean_post_snr=("post_snr", "mean"),
            mean_pRx=("pRx", "mean"),
            samples=("link_viable", "size"),
        )
        grouped.insert(0, "snr_threshold_db", float(threshold))
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def _three_node_beta_expectation(probabilities: list[float]) -> tuple[float, float, float, float]:
    p12, p13, p23 = probabilities
    p_connected = (
        p12 * p13 * (1 - p23)
        + p12 * p23 * (1 - p13)
        + p13 * p23 * (1 - p12)
        + p12 * p13 * p23
    )
    p_two_components = (
        p12 * (1 - p13) * (1 - p23)
        + p13 * (1 - p12) * (1 - p23)
        + p23 * (1 - p12) * (1 - p13)
    )
    p_three_components = (1 - p12) * (1 - p13) * (1 - p23)
    expected_beta = p_connected + 2 * p_two_components + 3 * p_three_components
    return float(expected_beta), float(1 - p_connected), float(p_two_components), float(p_three_components)


def _forestry_beta_trace(trace_path: Path, distance_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    trace = load_flight_trace_csv(trace_path)
    rows = []
    for threshold, table in distance_table.groupby("snr_threshold_db"):
        table = table.sort_values("distance")
        distances_known = table["distance"].to_numpy(dtype=float)
        probs_known = table["success_probability"].to_numpy(dtype=float)
        for time_idx, timestamp in enumerate(trace.timestamps_s):
            pair_rows = []
            probs = []
            for left in range(len(trace.vehicle_ids)):
                for right in range(left + 1, len(trace.vehicle_ids)):
                    distance = float(np.linalg.norm(trace.positions_m[time_idx, left] - trace.positions_m[time_idx, right]))
                    probability = float(
                        np.interp(distance, distances_known, probs_known, left=probs_known[0], right=probs_known[-1])
                    )
                    in_support = bool(
                        distances_known[0] <= distance <= distances_known[-1]
                    )
                    probs.append(probability)
                    pair_rows.append(
                        (
                            trace.vehicle_ids[left],
                            trace.vehicle_ids[right],
                            distance,
                            probability,
                            in_support,
                        )
                    )
            expected_beta, frag_probability, p_two, p_three = _three_node_beta_expectation(probs)
            row = {
                "snr_threshold_db": float(threshold),
                "timestamp_s": float(timestamp),
                "expected_beta0": expected_beta,
                "fragmentation_probability": frag_probability,
                "p_two_components": p_two,
                "p_three_components": p_three,
                "all_pairs_in_measured_support": bool(
                    all(item[4] for item in pair_rows)
                ),
            }
            for idx, (left_id, right_id, distance, probability, in_support) in enumerate(pair_rows, start=1):
                row[f"pair_{idx}"] = f"{left_id}-{right_id}"
                row[f"pair_{idx}_distance_m"] = distance
                row[f"pair_{idx}_link_probability"] = probability
                row[f"pair_{idx}_in_measured_support"] = in_support
            rows.append(row)
    series = pd.DataFrame(rows)
    summaries = []
    for scope, subset in [
        ("full_trace_clamped", series),
        (
            "all_pairs_within_measured_distance_support",
            series[series["all_pairs_in_measured_support"]],
        ),
    ]:
        grouped = subset.groupby("snr_threshold_db", as_index=False).agg(
            samples=("timestamp_s", "size"),
            expected_beta0_mean=("expected_beta0", "mean"),
            expected_beta0_min=("expected_beta0", "min"),
            expected_beta0_max=("expected_beta0", "max"),
            fragmentation_probability_mean=("fragmentation_probability", "mean"),
            fragmentation_probability_min=("fragmentation_probability", "min"),
            fragmentation_probability_max=("fragmentation_probability", "max"),
        )
        grouped.insert(1, "support_scope", scope)
        summaries.append(grouped)
    summary = pd.concat(summaries, ignore_index=True)
    return series, summary


def _write_table(path: Path, metrics: pd.DataFrame, beta_summary: pd.DataFrame) -> None:
    logistic = metrics[metrics["model"] == "logistic A2A RF model"].iloc[0]
    baseline = metrics[metrics["model"] == "training-prior baseline"].iloc[0]
    rows = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Evidence & Target & $n$ & Condition & Estimate A & Estimate B \\",
        r"\midrule",
        (
            "UAV--UAV 60 GHz & Link viability & "
            f"{int(logistic['test_n'])} & $F_1$ & {baseline['f1']:.3f} & {logistic['f1']:.3f} \\\\"
        ),
        (
            "UAV--UAV 60 GHz & Link viability & "
            f"{int(logistic['test_n'])} & PR-AUC & {baseline['pr_auc']:.3f} & {logistic['pr_auc']:.3f} \\\\"
        ),
    ]
    primary_summary = beta_summary[
        beta_summary["support_scope"]
        == "all_pairs_within_measured_distance_support"
    ]
    for _, item in primary_summary.iterrows():
        rows.append(
            "Transported RF sensitivity & Expected $\\beta_0$ & "
            f"{int(item['samples'])} & SNR {item['snr_threshold_db']:.0f} dB & "
            f"{item['expected_beta0_mean']:.3f} & {item['fragmentation_probability_mean']:.3f} \\\\"
        )
    rows.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _plot(out_dir: Path, distance_table: pd.DataFrame, beta_series: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    for threshold, group in distance_table.groupby("snr_threshold_db"):
        axes[0].plot(group["distance"], group["success_probability"], marker="o", linewidth=1.4, label=f"SNR >= {threshold:.0f} dB")
    axes[0].set_xlabel("Measured UAV-to-UAV distance (m)")
    axes[0].set_ylabel("Empirical link probability")
    axes[0].set_title("60 GHz A2A beam-scan viability")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    for threshold, group in beta_series.groupby("snr_threshold_db"):
        axes[1].plot(group["timestamp_s"], group["expected_beta0"], linewidth=1.0, label=f"SNR >= {threshold:.0f} dB")
    axes[1].set_xlabel("Elapsed time (s)")
    axes[1].set_ylabel("Expected $\\beta_0$")
    axes[1].set_title("Transported 60 GHz forestry sensitivity")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"uav_to_uav_mmwave_validation.{suffix}", dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and integrate public measured UAV-to-UAV mmWave RF data.")
    parser.add_argument("--dataset", type=Path, default=RAW_PATH)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manuscript-dir", type=Path, default=DEFAULT_MANUSCRIPT)
    parser.add_argument("--link-snr-db", type=float, default=5.0)
    parser.add_argument("--link-prx-dbm", type=float, default=-88.0)
    args = parser.parse_args()

    _require_input(args.dataset)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = _load_a2a(args.dataset, args.link_snr_db, args.link_prx_dbm)
    metrics = _link_model_metrics(frame)
    distance_table = _distance_success_table(
        frame,
        (5.0, 7.0, 10.0),
        args.link_prx_dbm,
    )
    beta_series, beta_summary = _forestry_beta_trace(args.trace, distance_table)

    metrics.to_csv(args.output_dir / "uav_to_uav_link_model_metrics.csv", index=False)
    distance_table.to_csv(args.output_dir / "uav_to_uav_distance_success_table.csv", index=False)
    beta_series.to_csv(args.output_dir / "uav_to_uav_forestry_beta0_timeseries.csv", index=False)
    beta_summary.to_csv(args.output_dir / "uav_to_uav_forestry_beta0_summary.csv", index=False)

    table_path = args.output_dir / "uav_to_uav_mmwave_table.tex"
    _write_table(table_path, metrics, beta_summary)
    _plot(args.output_dir, distance_table, beta_series)

    protocol = {
        "scope": "Measured UAV-to-UAV 60 GHz RF validation plus a transported forestry sensitivity analysis; this is not same-site calibration or packet-delivery ground truth.",
        "source": "WiNES/Northeastern UAV-to-UAV 60 GHz experimental channel model",
        "dataset_url": DATASET_URL,
        "raw_url": RAW_URL,
        "relative_path": str(args.dataset.relative_to(ROOT)),
        "sha256": _sha256(args.dataset),
        "bytes": args.dataset.stat().st_size,
        "link_viability_rule": {
            "post_snr_db_min": args.link_snr_db,
            "pRx_dbm_min": args.link_prx_dbm,
        },
        "a2a_rows": int(len(frame)),
        "held_out_distance_split": "train distance < 33 m, test distance >= 33 m",
        "forestry_trace": str(args.trace.relative_to(ROOT)),
        "forestry_beta0": "Pairwise SNR-and-pRx link probabilities are interpolated from measured UAV-to-UAV beam-scan success rates and converted to three-node expected beta0 under an explicit independent-edge sensitivity assumption.",
        "measured_distance_support_m": [
            float(distance_table["distance"].min()),
            float(distance_table["distance"].max()),
        ],
        "primary_forestry_scope": "all_pairs_within_measured_distance_support",
        "out_of_support_policy": "Full-trace rows are clamped to nearest measured distance and reported only as secondary sensitivity; primary summaries require all three pairs in support.",
    }
    (args.output_dir / "uav_to_uav_mmwave_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    if args.manuscript_dir:
        table_dest = args.manuscript_dir / "tables" / "generated"
        figure_dest = args.manuscript_dir / "figures" / "generated"
        table_dest.mkdir(parents=True, exist_ok=True)
        figure_dest.mkdir(parents=True, exist_ok=True)
        (table_dest / "uav_to_uav_mmwave_table.tex").write_text(table_path.read_text(encoding="utf-8"), encoding="utf-8")
        for suffix in ("png", "pdf"):
            (figure_dest / f"uav_to_uav_mmwave_validation.{suffix}").write_bytes(
                (args.output_dir / f"uav_to_uav_mmwave_validation.{suffix}").read_bytes()
            )

    print(f"Wrote {args.output_dir.relative_to(ROOT)}")
    print(metrics.to_string(index=False))
    print(beta_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
