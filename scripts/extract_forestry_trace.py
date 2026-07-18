from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from rosbags.highlevel import AnyReader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fanet.provenance import build_file_manifest, relative_repo_path, sha256_file


RECORD_DOI = "10.5281/zenodo.14701641"
EARTH_RADIUS_M = 6_378_137.0
EXPECTED_MD5 = {
    "uav6_swarm_formation_2024-12-04-15-29-12_0_ok1.bag": "d9d88fbb378e51bdf6ac57a4ca2be08f",
    "uav8_swarm_formation_2024-12-04-15-29-08_0_ok1.bag": "ef86998f372689d95a4be696b3d92b88",
    "uav9_swarm_formation_2024-12-04-15-29-04_0_ok1.bag": "adb55c9c6282b7bbd609cda5361b00f6",
}


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _choose_navsat_connection(reader: AnyReader):
    candidates = [
        connection
        for connection in reader.connections
        if connection.msgtype.endswith("sensor_msgs/msg/NavSatFix")
        or connection.msgtype.endswith("sensor_msgs/NavSatFix")
    ]
    if not candidates:
        available = sorted({f"{item.topic} [{item.msgtype}]" for item in reader.connections})
        raise RuntimeError("No sensor_msgs/NavSatFix topic found. Available topics:\n" + "\n".join(available))
    candidates.sort(
        key=lambda item: (
            "gnss_verifier" not in item.topic.lower(),
            "rtk" not in item.topic.lower(),
            "gps" not in item.topic.lower() and "gnss" not in item.topic.lower(),
            -int(getattr(item, "msgcount", 0)),
            item.topic,
        )
    )
    return candidates[0]


def _read_navsat(path: Path) -> tuple[pd.DataFrame, str]:
    rows = []
    with AnyReader([path]) as reader:
        connection = _choose_navsat_connection(reader)
        for _, timestamp_ns, rawdata in reader.messages(connections=[connection]):
            message = reader.deserialize(rawdata, connection.msgtype)
            latitude = float(message.latitude)
            longitude = float(message.longitude)
            altitude = float(message.altitude)
            status = int(getattr(getattr(message, "status", None), "status", 0))
            if status < 0:
                continue
            if not (math.isfinite(latitude) and math.isfinite(longitude) and math.isfinite(altitude)):
                continue
            if abs(latitude) < 1e-8 and abs(longitude) < 1e-8:
                continue
            rows.append((timestamp_ns * 1e-9, latitude, longitude, altitude))
    if len(rows) < 2:
        raise RuntimeError(f"Too few valid NavSatFix samples in {path.name}")
    frame = pd.DataFrame(rows, columns=["timestamp_s", "latitude", "longitude", "altitude_m"])
    frame = frame.drop_duplicates("timestamp_s").sort_values("timestamp_s").reset_index(drop=True)
    return frame, connection.topic


def _interpolation_validity(source_times: np.ndarray, target_times: np.ndarray, max_gap_s: float) -> np.ndarray:
    right = np.searchsorted(source_times, target_times, side="left")
    right = np.clip(right, 1, len(source_times) - 1)
    left = right - 1
    return (
        (target_times >= source_times[0])
        & (target_times <= source_times[-1])
        & ((source_times[right] - source_times[left]) <= max_gap_s)
    )


def _longest_true_slice(mask: np.ndarray) -> slice:
    best_start = best_stop = start = 0
    for idx, value in enumerate(np.append(mask, False)):
        if value:
            continue
        if idx - start > best_stop - best_start:
            best_start, best_stop = start, idx
        start = idx + 1
    return slice(best_start, best_stop)


def _to_local_xyz(frames: dict[str, pd.DataFrame], sample_rate_hz: float, max_gap_s: float) -> tuple[pd.DataFrame, dict]:
    start = max(float(frame.timestamp_s.min()) for frame in frames.values())
    stop = min(float(frame.timestamp_s.max()) for frame in frames.values())
    if stop <= start:
        raise RuntimeError("The bag files have no overlapping GNSS interval")
    step = 1.0 / sample_rate_hz
    target = np.arange(math.ceil(start / step) * step, stop, step)
    valid = np.ones(len(target), dtype=bool)
    for frame in frames.values():
        valid &= _interpolation_validity(frame.timestamp_s.to_numpy(dtype=float), target, max_gap_s)
    segment = _longest_true_slice(valid)
    target = target[segment]
    if len(target) < max(10, int(30 * sample_rate_hz)):
        raise RuntimeError("No shared GNSS segment of at least 30 seconds passed the gap filter")

    all_lat = np.concatenate([frame.latitude.to_numpy(dtype=float) for frame in frames.values()])
    all_lon = np.concatenate([frame.longitude.to_numpy(dtype=float) for frame in frames.values()])
    lat0 = float(np.median(all_lat))
    lon0 = float(np.median(all_lon))
    alt0 = float(np.median(np.concatenate([frame.altitude_m.to_numpy(dtype=float) for frame in frames.values()])))
    cos_lat0 = math.cos(math.radians(lat0))
    rows = []
    for vehicle_id, frame in sorted(frames.items()):
        source_t = frame.timestamp_s.to_numpy(dtype=float)
        lat = np.interp(target, source_t, frame.latitude.to_numpy(dtype=float))
        lon = np.interp(target, source_t, frame.longitude.to_numpy(dtype=float))
        alt = np.interp(target, source_t, frame.altitude_m.to_numpy(dtype=float))
        x = EARTH_RADIUS_M * np.radians(lon - lon0) * cos_lat0
        y = EARTH_RADIUS_M * np.radians(lat - lat0)
        z = alt - alt0
        rows.extend(
            {
                "timestamp_s": float(timestamp - target[0]),
                "vehicle_id": vehicle_id,
                "x_m": float(px),
                "y_m": float(py),
                "z_m": float(pz),
            }
            for timestamp, px, py, pz in zip(target, x, y, z)
        )
    metadata = {
        "sample_rate_hz": sample_rate_hz,
        "max_interpolation_gap_s": max_gap_s,
        "shared_segment_start_unix_s": float(target[0]),
        "shared_segment_stop_unix_s": float(target[-1]),
        "shared_segment_duration_s": float(target[-1] - target[0]),
        "samples_per_vehicle": len(target),
        "coordinate_transform": "local equirectangular ENU approximation around the joint median GNSS origin",
        "origin_latitude_deg": lat0,
        "origin_longitude_deg": lon0,
        "origin_altitude_m": alt0,
    }
    return pd.DataFrame(rows), metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract an aligned real multi-UAV GNSS trace from public forestry ROS bags.")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "external_validation" / "raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "external_validation" / "derived")
    parser.add_argument("--sample-rate-hz", type=float, default=10.0)
    parser.add_argument("--max-gap-s", type=float, default=0.5)
    args = parser.parse_args()
    bags = sorted(args.raw_dir.glob("*.bag"))
    if len(bags) != 3:
        raise SystemExit(f"Expected exactly three bags in {args.raw_dir}; found {len(bags)}")

    frames: dict[str, pd.DataFrame] = {}
    sources = []
    for bag in bags:
        vehicle_id = bag.name.split("_")[0]
        frame, topic = _read_navsat(bag)
        actual_md5 = _md5(bag)
        expected_md5 = EXPECTED_MD5.get(bag.name)
        if expected_md5 is None:
            raise RuntimeError(f"No authoritative checksum is registered for {bag.name}")
        if actual_md5 != expected_md5:
            raise RuntimeError(f"MD5 mismatch for {bag.name}: expected {expected_md5}, got {actual_md5}")
        frames[vehicle_id] = frame
        sources.append(
            {
                "vehicle_id": vehicle_id,
                "file": bag.name,
                "relative_path": relative_repo_path(bag, ROOT),
                "bytes": bag.stat().st_size,
                "md5": actual_md5,
                "sha256": sha256_file(bag),
                "topic": topic,
                "valid_navsat_samples": len(frame),
            }
        )
        print(f"[{vehicle_id}] {topic}: {len(frame):,} valid GNSS samples")

    trace, metadata = _to_local_xyz(frames, args.sample_rate_hz, args.max_gap_s)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = args.output_dir / "forestry_multidrone_trace.csv"
    manifest_path = args.output_dir / "forestry_multidrone_trace_manifest.json"
    trace.to_csv(trace_path, index=False, float_format="%.6f")
    manifest = {
        "dataset_title": "Dataset of A Multi-Drone System Proof of Concept for Forestry Applications",
        "record_url": "https://zenodo.org/records/14701641",
        "doi": RECORD_DOI,
        "license": "CC-BY-4.0",
        "source_kind": "real multi-drone forestry field experiment",
        "directly_measured_variables": ["GNSS latitude", "GNSS longitude", "GNSS altitude", "ROS message timestamp"],
        "derived_variables": ["shared-time local ENU position trace"],
        "raw_to_derived_transform": "valid NavSatFix filtering, shared-interval selection, bounded-gap interpolation, and local equirectangular ENU projection",
        "allowed_claim": ["measured multi-UAV field motion"],
        "prohibited_claim": ["measured peer RF", "measured packet delivery", "measured radius-graph connectivity"],
        "measurement_boundary": {"motion_measured": True, "peer_rf_labels_present": False, "packet_labels_present": False},
        "sources": sources,
        "files": build_file_manifest([trace_path], ROOT),
        "derived_file": relative_repo_path(trace_path, ROOT),
        "derived_sha256": sha256_file(trace_path),
        **metadata,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {trace_path} ({metadata['samples_per_vehicle']:,} samples per vehicle)")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
