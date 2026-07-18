from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import requests
from remotezip import RemoteZip


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fanet.provenance import build_file_manifest, relative_repo_path


ARTICLE_ID = 28386041
FILE_ID = 52317539
ARCHIVE_NAME = "cirObstacles_3_random_0.zip"
ARCHIVE_MD5 = "b2087e3d2b3f2e23b835a5cb1b46528a"
EXPERIMENT = "cirObstacles_3_random_0"
DEFAULT_OUTPUT = ROOT / "data" / "external_validation" / "raw" / "miluv" / EXPERIMENT
CATALOG_PATH = ROOT / "data" / "external_validation" / "raw" / "miluv" / "sequence_catalog.json"
MEMBERS = [
    f"{EXPERIMENT}/{robot}/{filename}"
    for robot in ("ifo001", "ifo002", "ifo003")
    for filename in ("mocap.csv", "uwb_range.csv")
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Range-download the minimal MILUV three-UAV validation files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--catalog-only", action="store_true", help="Discover official three-robot archives without downloading them.")
    args = parser.parse_args()

    metadata_url = f"https://api.figshare.com/v2/articles/{ARTICLE_ID}"
    metadata = requests.get(metadata_url, timeout=60).json()
    three_robot = [
        {
            "archive_file_id": int(item["id"]),
            "archive_name": str(item["name"]),
            "archive_bytes": int(item["size"]),
            "archive_md5_from_figshare": item.get("computed_md5") or item.get("supplied_md5"),
            "experiment": Path(str(item["name"])).stem,
            "locally_extracted": (DEFAULT_OUTPUT.parent / Path(str(item["name"])).stem / "manifest.json").is_file(),
        }
        for item in metadata.get("files", [])
        if str(item.get("name", "")).endswith(".zip") and "_3_" in str(item.get("name", ""))
    ]
    catalog = {
        "dataset": "MILUV: A Multi-UAV Indoor Localization dataset with UWB and Vision",
        "dataset_doi": "10.25452/figshare.plus.28386041.v1",
        "article_id": ARTICLE_ID,
        "selection_rule": "official ZIP filenames containing '_3_'; compatibility requires verified ifo001-ifo003 mocap.csv and uwb_range.csv members",
        "candidate_count": len(three_robot),
        "sequences": sorted(three_robot, key=lambda row: row["archive_name"]),
    }
    CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CATALOG_PATH.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    if args.catalog_only:
        print(f"Wrote {relative_repo_path(CATALOG_PATH, ROOT)} with {len(three_robot)} candidates")
        return 0
    file_metadata = next(item for item in metadata["files"] if int(item["id"]) == FILE_ID)
    if file_metadata["name"] != ARCHIVE_NAME:
        raise RuntimeError(f"Unexpected Figshare archive name: {file_metadata['name']}")
    if file_metadata.get("computed_md5") != ARCHIVE_MD5:
        raise RuntimeError("Figshare archive MD5 metadata changed")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    extracted = []
    with RemoteZip(file_metadata["download_url"]) as archive:
        available = {item.filename: item for item in archive.infolist()}
        missing = sorted(set(MEMBERS).difference(available))
        if missing:
            raise RuntimeError(f"MILUV archive members missing: {missing}")
        for member in MEMBERS:
            robot = Path(member).parent.name
            destination = args.output_dir / robot / Path(member).name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(member))
            info = available[member]
            provenance = build_file_manifest([destination], ROOT)[0]
            extracted.append(
                {
                    "archive_member": member,
                    "zip_crc32": f"{info.CRC:08x}",
                    **provenance,
                }
            )

    manifest = {
        "dataset": "MILUV: A Multi-UAV Indoor Localization dataset with UWB and Vision",
        "dataset_doi": "10.25452/figshare.plus.28386041.v1",
        "article_id": ARTICLE_ID,
        "archive_file_id": FILE_ID,
        "archive_name": ARCHIVE_NAME,
        "archive_bytes": int(file_metadata["size"]),
        "archive_md5_from_figshare": ARCHIVE_MD5,
        "download_method": "HTTP range extraction through the ZIP central directory",
        "experiment": EXPERIMENT,
        "files": extracted,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
