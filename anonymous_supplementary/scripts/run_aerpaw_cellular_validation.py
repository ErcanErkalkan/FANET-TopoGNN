from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data" / "external_validation" / "raw"
DEFAULT_OUTPUT = ROOT / "outputs" / "aerpaw_cellular_validation"
DEFAULT_MANUSCRIPT = ROOT / "paper"

DATASETS = {
    "dataset22_lte_semicircle": {
        "path": RAW_ROOT / "aerpaw_dataset22" / "Logs" / "lte.csv",
        "source": "AERPAW Dataset-22",
        "url": "https://aerpaw.org/dataset/dataset-22-android-based-4g-lte-measurements-for-semi-circular-uav-trajectory-around-a-private-aerpaw-base-station/",
        "drive": "https://drive.google.com/drive/folders/1S7jqTYkPfZD9gf_5tFq1ufyuqEnXgQPn",
    },
    "dataset23_lte_two_sweeps": {
        "path": RAW_ROOT / "aerpaw_dataset23" / "Logs" / "4G_lte.csv",
        "source": "AERPAW Dataset-23 LTE",
        "url": "https://aerpaw.org/dataset/dataset-23-android-based-4g-lte-5g-nr-and-throughput-measurements-for-two-sweeps-of-a-uav-near-a-private-aerpaw-base-station/",
        "drive": "https://drive.google.com/drive/folders/1yo3B6xCxHx46ceTwO_NL6zhvplD0l-RV",
    },
    "dataset23_nr_two_sweeps": {
        "path": RAW_ROOT / "aerpaw_dataset23" / "Logs" / "5G_nr.csv",
        "source": "AERPAW Dataset-23 NR",
        "url": "https://aerpaw.org/dataset/dataset-23-android-based-4g-lte-5g-nr-and-throughput-measurements-for-two-sweeps-of-a-uav-near-a-private-aerpaw-base-station/",
        "drive": "https://drive.google.com/drive/folders/1yo3B6xCxHx46ceTwO_NL6zhvplD0l-RV",
    },
    "dataset23_iperf_two_sweeps": {
        "path": RAW_ROOT / "aerpaw_dataset23" / "Logs" / "iperf_throughput.csv",
        "source": "AERPAW Dataset-23 iPerf",
        "url": "https://aerpaw.org/dataset/dataset-23-android-based-4g-lte-5g-nr-and-throughput-measurements-for-two-sweeps-of-a-uav-near-a-private-aerpaw-base-station/",
        "drive": "https://drive.google.com/drive/folders/1yo3B6xCxHx46ceTwO_NL6zhvplD0l-RV",
    },
}

SENTINELS = {2147483647, -2147483648}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_inputs() -> None:
    missing = [str(meta["path"].relative_to(ROOT)) for meta in DATASETS.values() if not meta["path"].exists()]
    if missing:
        commands = [
            "python -m pip install -r requirements-external.txt",
            'python -m gdown --folder "https://drive.google.com/drive/folders/1S7jqTYkPfZD9gf_5tFq1ufyuqEnXgQPn" -O "data/external_validation/raw/aerpaw_dataset22"',
            'python -m gdown --folder "https://drive.google.com/drive/folders/1yo3B6xCxHx46ceTwO_NL6zhvplD0l-RV" -O "data/external_validation/raw/aerpaw_dataset23"',
        ]
        raise FileNotFoundError(
            "Missing AERPAW CSV files:\n"
            + "\n".join(f"- {item}" for item in missing)
            + "\nDownload with:\n"
            + "\n".join(commands)
        )


def _numeric(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    return out.mask(out.isin(SENTINELS))


def _add_local_geometry(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    lat = _numeric(out["latitude"])
    lon = _numeric(out["longitude"])
    lat0 = float(lat.dropna().median())
    lon0 = float(lon.dropna().median())
    radius = 6_371_000.0
    out["x_m"] = np.deg2rad(lon - lon0) * radius * np.cos(np.deg2rad(lat0))
    out["y_m"] = np.deg2rad(lat - lat0) * radius
    out["distance_from_median_m"] = np.hypot(out["x_m"], out["y_m"])
    return out


def _clean_lte(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for column in [
        "dbm",
        "rsrp",
        "rsrq",
        "rssi",
        "asu",
        "pci",
        "level",
        "altitude",
        "longitude",
        "latitude",
        "rel_time",
        "companion_abs_time",
        "is_connected",
    ]:
        if column in df.columns:
            df[column] = _numeric(df[column])
    df = _add_local_geometry(df)
    return df.sort_values("rel_time").reset_index(drop=True)


def _clean_nr(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for column in [
        "ss_rsrp",
        "ss_rsrq",
        "ss_sinr",
        "asu",
        "level",
        "altitude",
        "longitude",
        "latitude",
        "rel_time",
        "companion_abs_time",
    ]:
        if column in df.columns:
            df[column] = _numeric(df[column])
    df = _add_local_geometry(df)
    return df.sort_values("rel_time").reset_index(drop=True)


def _availability_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    out = {
        "test_n": float(len(y_true)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, np.clip(y_score, 0.0, 1.0))),
    }
    if len(np.unique(y_true)) > 1:
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    else:
        out["pr_auc"] = float("nan")
        out["roc_auc"] = float("nan")
    return out


def _best_rsrp_threshold(train: pd.DataFrame) -> float:
    values = np.sort(train["rsrp"].dropna().unique())
    if len(values) == 0:
        return -100.0
    candidates = np.unique(np.quantile(values, np.linspace(0.02, 0.98, 97)))
    best = (0.0, float(candidates[0]))
    y_true = train["is_connected"].to_numpy(dtype=int)
    for threshold in candidates:
        pred = (train["rsrp"].to_numpy(dtype=float) >= threshold).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best[0]:
            best = (float(score), float(threshold))
    return best[1]


def _run_availability(dataset_key: str, path: Path) -> tuple[list[dict[str, float | str]], pd.DataFrame]:
    df = _clean_lte(path)
    feature_cols = ["rsrp", "rsrq", "rssi", "altitude", "distance_from_median_m"]
    work = df.dropna(subset=[*feature_cols, "is_connected", "rel_time"]).copy()
    work["is_connected"] = work["is_connected"].astype(int)
    split = int(np.floor(len(work) * 0.7))
    train = work.iloc[:split].copy()
    test = work.iloc[split:].copy()

    threshold = _best_rsrp_threshold(train)
    threshold_scores = (test["rsrp"].to_numpy(dtype=float) >= threshold).astype(float)
    threshold_row = {
        "dataset": dataset_key,
        "target": "LTE serving/connection state",
        "model": f"RSRP threshold ({threshold:.1f} dBm)",
        "split": "final-30%-by-time",
        "train_n": float(len(train)),
        "threshold": threshold,
        **_availability_metrics(test["is_connected"].to_numpy(dtype=int), threshold_scores),
    }

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    )
    model.fit(train[feature_cols], train["is_connected"])
    logistic_scores = model.predict_proba(test[feature_cols])[:, 1]
    logistic_row = {
        "dataset": dataset_key,
        "target": "LTE serving/connection state",
        "model": "Logistic RF/KPI model",
        "split": "final-30%-by-time",
        "train_n": float(len(train)),
        "threshold": 0.5,
        **_availability_metrics(test["is_connected"].to_numpy(dtype=int), logistic_scores),
    }
    return [threshold_row, logistic_row], work


def _run_throughput(lte_path: Path, throughput_path: Path) -> tuple[dict[str, float | str], dict[str, float | str], pd.DataFrame]:
    lte = _clean_lte(lte_path)
    iperf = pd.read_csv(throughput_path)
    iperf["throughput"] = _numeric(iperf["throughput"])
    iperf["abs_time"] = _numeric(iperf["abs_time"])
    for column in ["longitude", "latitude", "altitude"]:
        iperf[column] = _numeric(iperf[column])
    iperf = _add_local_geometry(iperf)

    lte = lte[lte["is_connected"] == 1].dropna(subset=["companion_abs_time", "rsrp", "rsrq", "rssi"]).copy()
    lte = lte.sort_values("companion_abs_time")
    iperf = iperf.dropna(subset=["abs_time", "throughput"]).sort_values("abs_time")

    merged = pd.merge_asof(
        iperf,
        lte,
        left_on="abs_time",
        right_on="companion_abs_time",
        direction="nearest",
        tolerance=1500.0,
        suffixes=("_iperf", "_lte"),
    )
    feature_cols = ["rsrp", "rsrq", "rssi", "altitude_lte", "distance_from_median_m_lte"]
    merged = merged.dropna(subset=[*feature_cols, "throughput"]).copy()
    merged = merged.sort_values("abs_time").reset_index(drop=True)
    block_index = (merged["abs_time"] // 5000).astype(int)
    test_mask = (block_index % 5) == 0
    train = merged.loc[~test_mask].copy()
    test = merged.loc[test_mask].copy()

    baseline = np.full(len(test), float(train["throughput"].mean()))
    baseline_row = {
        "dataset": "dataset23_iperf_two_sweeps",
        "target": "iPerf throughput (Mbps)",
        "model": "Training-mean baseline",
        "split": "interleaved-5s-blocks",
        "train_n": float(len(train)),
        "test_n": float(len(test)),
        "mae_mbps": float(mean_absolute_error(test["throughput"], baseline)),
        "rmse_mbps": float(mean_squared_error(test["throughput"], baseline) ** 0.5),
        "r2": float(r2_score(test["throughput"], baseline)),
    }

    regressor = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=42)
    regressor.fit(train[feature_cols], train["throughput"])
    pred = regressor.predict(test[feature_cols])
    rf_row = {
        "dataset": "dataset23_iperf_two_sweeps",
        "target": "iPerf throughput (Mbps)",
        "model": "RF/KPI random forest",
        "split": "interleaved-5s-blocks",
        "train_n": float(len(train)),
        "test_n": float(len(test)),
        "mae_mbps": float(mean_absolute_error(test["throughput"], pred)),
        "rmse_mbps": float(mean_squared_error(test["throughput"], pred) ** 0.5),
        "r2": float(r2_score(test["throughput"], pred)),
    }
    merged.loc[test.index, "throughput_pred_rf"] = pred
    merged.loc[test.index, "throughput_pred_baseline"] = baseline
    return baseline_row, rf_row, merged


def _write_table(out_path: Path, availability: pd.DataFrame, throughput: pd.DataFrame) -> None:
    best_availability = (
        availability.sort_values(["dataset", "f1"], ascending=[True, False])
        .groupby("dataset", as_index=False)
        .first()
    )
    best_throughput = throughput.sort_values("mae_mbps", ascending=True).iloc[0]
    baseline_throughput = throughput[throughput["model"] == "Training-mean baseline"].iloc[0]
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"External evidence & Target & Test $n$ & Metric & Baseline & Best model \\",
        r"\midrule",
    ]
    for _, row in best_availability.iterrows():
        label = "AERPAW D22 LTE" if row["dataset"] == "dataset22_lte_semicircle" else "AERPAW D23 LTE"
        baseline = availability[
            (availability["dataset"] == row["dataset"])
            & availability["model"].astype(str).str.startswith("RSRP threshold")
        ].iloc[0]
        lines.append(
            f"{label} & Link state & {int(row['test_n'])} & $F_1$ & "
            f"{baseline['f1']:.3f} & {row['f1']:.3f} \\\\"
        )
    lines.append(
        "AERPAW D23 iPerf & Throughput & "
        f"{int(best_throughput['test_n'])} & MAE & "
        f"{baseline_throughput['mae_mbps']:.1f} & {best_throughput['mae_mbps']:.1f} \\\\"
    )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_outputs(out_dir: Path, availability_records: dict[str, pd.DataFrame], throughput: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for label, df in availability_records.items():
        axes[0].scatter(
            df["rel_time"],
            df["rsrp"],
            s=8,
            alpha=0.45,
            label=label.replace("dataset", "D").replace("_", " "),
            c=df["is_connected"],
            cmap="coolwarm",
        )
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("LTE RSRP (dBm)")
    axes[0].set_title("Measured aerial LTE link state")
    axes[0].grid(alpha=0.2)

    test = throughput.dropna(subset=["throughput_pred_rf"])
    axes[1].scatter(test["throughput"], test["throughput_pred_rf"], s=12, alpha=0.6)
    low = float(min(test["throughput"].min(), test["throughput_pred_rf"].min()))
    high = float(max(test["throughput"].max(), test["throughput_pred_rf"].max()))
    axes[1].plot([low, high], [low, high], color="black", linewidth=1)
    axes[1].set_xlabel("Measured throughput (Mbps)")
    axes[1].set_ylabel("Predicted throughput (Mbps)")
    axes[1].set_title("AERPAW D23 iPerf")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(out_dir / f"aerpaw_cellular_validation.{suffix}", dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run real AERPAW cellular RF/KPI validation.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manuscript-dir", type=Path, default=DEFAULT_MANUSCRIPT)
    args = parser.parse_args()

    _require_inputs()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    availability_rows = []
    availability_records: dict[str, pd.DataFrame] = {}
    for key in ["dataset22_lte_semicircle", "dataset23_lte_two_sweeps"]:
        rows, frame = _run_availability(key, DATASETS[key]["path"])
        availability_rows.extend(rows)
        availability_records[key] = frame
    availability_df = pd.DataFrame(availability_rows)

    baseline_row, rf_row, throughput_frame = _run_throughput(
        DATASETS["dataset23_lte_two_sweeps"]["path"],
        DATASETS["dataset23_iperf_two_sweeps"]["path"],
    )
    throughput_df = pd.DataFrame([baseline_row, rf_row])

    nr = _clean_nr(DATASETS["dataset23_nr_two_sweeps"]["path"])
    nr_summary = {
        "rows": int(len(nr)),
        "ss_rsrp_mean": float(nr["ss_rsrp"].mean()),
        "ss_sinr_mean": float(nr["ss_sinr"].mean()),
        "altitude_mean_m": float(nr["altitude"].mean()),
    }

    availability_df.to_csv(args.output_dir / "aerpaw_lte_availability_metrics.csv", index=False)
    throughput_df.to_csv(args.output_dir / "aerpaw_throughput_metrics.csv", index=False)
    throughput_frame.to_csv(args.output_dir / "aerpaw_throughput_aligned_predictions.csv", index=False)

    table_path = args.output_dir / "aerpaw_cellular_table.tex"
    _write_table(table_path, availability_df, throughput_df)
    _plot_outputs(args.output_dir, availability_records, throughput_frame)

    manifest = {
        "scope": (
            "Measured aerial cellular RF/KPI validation. This is real UAV-to-cellular infrastructure evidence; "
            "it is not an inter-UAV FANET packet-link dataset."
        ),
        "sources": {
            key: {
                "source": meta["source"],
                "url": meta["url"],
                "drive": meta["drive"],
                "relative_path": str(meta["path"].relative_to(ROOT)),
                "sha256": _sha256(meta["path"]),
                "bytes": meta["path"].stat().st_size,
            }
            for key, meta in DATASETS.items()
        },
        "availability_rows": int(len(availability_df)),
        "throughput_rows": int(len(throughput_df)),
        "nr_summary": nr_summary,
    }
    (args.output_dir / "aerpaw_cellular_protocol.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.manuscript_dir:
        table_dest = args.manuscript_dir / "tables" / "generated"
        figure_dest = args.manuscript_dir / "figures" / "generated"
        table_dest.mkdir(parents=True, exist_ok=True)
        figure_dest.mkdir(parents=True, exist_ok=True)
        (table_dest / "aerpaw_cellular_table.tex").write_text(table_path.read_text(encoding="utf-8"), encoding="utf-8")
        for suffix in ["png", "pdf"]:
            target = figure_dest / f"aerpaw_cellular_validation.{suffix}"
            target.write_bytes((args.output_dir / f"aerpaw_cellular_validation.{suffix}").read_bytes())

    print(f"Wrote {args.output_dir.relative_to(ROOT)}")
    print(availability_df.to_string(index=False))
    print(throughput_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
