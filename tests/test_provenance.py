from __future__ import annotations

from pathlib import Path

import pytest

from fanet.provenance import (
    build_file_manifest,
    relative_repo_path,
    verify_manifest,
)


def test_windows_style_relative_path_is_normalized(tmp_path: Path) -> None:
    assert relative_repo_path(r"data\external_validation\source.csv", tmp_path) == (
        "data/external_validation/source.csv"
    )


def test_posix_relative_path_is_preserved(tmp_path: Path) -> None:
    assert relative_repo_path("data/external_validation/source.csv", tmp_path) == (
        "data/external_validation/source.csv"
    )


def test_correct_hash_manifest_is_accepted(tmp_path: Path) -> None:
    source = tmp_path / "data" / "source.csv"
    source.parent.mkdir(parents=True)
    source.write_text("value\n1\n", encoding="utf-8")
    manifest = build_file_manifest([source], tmp_path)

    verification = verify_manifest(manifest, tmp_path)

    assert verification["valid"] is True
    assert verification["errors"] == []
    assert verification["files"][0]["status"] == "pass"


def test_wrong_hash_is_rejected_without_rewriting_manifest(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("original", encoding="utf-8")
    manifest = build_file_manifest([source], tmp_path)
    expected = manifest[0]["sha256"]
    source.write_text("changed", encoding="utf-8")

    verification = verify_manifest(manifest, tmp_path)

    assert verification["valid"] is False
    assert "SHA-256 mismatch" in verification["errors"][0]
    assert manifest[0]["sha256"] == expected


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    manifest = [
        {
            "relative_path": "data/missing.csv",
            "bytes": 1,
            "sha256": "0" * 64,
        }
    ]

    verification = verify_manifest(manifest, tmp_path)

    assert verification["valid"] is False
    assert verification["files"][0]["status"] == "fail"
    assert "source file is missing" in verification["errors"][0]


def test_build_manifest_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing source file"):
        build_file_manifest([tmp_path / "missing.csv"], tmp_path)


def test_malformed_manifest_entry_is_not_silently_skipped(tmp_path: Path) -> None:
    verification = verify_manifest(["not-a-file-record"], tmp_path)

    assert verification["valid"] is False
    assert verification["errors"] == ["entry 0: file record must be a mapping"]
