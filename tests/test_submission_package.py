from pathlib import Path
import zipfile

from scripts.build_submission_package import (
    ROOT,
    absolute_path_findings,
    archive_member_records,
    cited_keys,
    latex_dependencies,
    materialize_anonymous_tex,
    safe_archive_members,
    sanitize_build_log,
    sanitized_bibliography,
    sha256_bytes,
)


def test_materialized_source_is_anonymous_single_column_preprint() -> None:
    source = (ROOT / "paper/main.tex").read_text(encoding="utf-8")
    anonymous = materialize_anonymous_tex(source)
    assert r"\documentclass[preprint]{elsarticle}" in anonymous
    assert "twocolumn" not in anonymous
    assert "Ercan" not in anonymous
    assert "Marmara" not in anonymous
    assert "Anonymous author(s)" in anonymous


def test_sanitized_bibliography_keeps_only_cited_entries() -> None:
    bib = """@article{used, title={Used}, year={2020}}
@misc{identity, title={Author archive}, year={2021}}
"""
    selected = sanitized_bibliography(bib, ["used"])
    assert "@article{used" in selected
    assert "identity" not in selected
    assert cited_keys(r"Text \cite{used, other}.") == ["other", "used"]


def test_archive_structure_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.txt", "bad")
    report = safe_archive_members(archive)
    assert report["valid"] is False
    assert report["unsafe_names"] == ["../escape.txt"]


def test_absolute_path_scan_ignores_urls_and_rejects_host_paths(tmp_path: Path) -> None:
    (tmp_path / "clean.txt").write_text("https://doi.org/10.1/example\n", encoding="utf-8")
    assert absolute_path_findings(tmp_path) == []
    (tmp_path / "bad.txt").write_text("C:\\Users\\person\\data.csv\n", encoding="utf-8")
    findings = absolute_path_findings(tmp_path)
    assert len(findings) == 1
    assert findings[0]["file"] == "bad.txt"


def test_archive_member_manifest_hashes_actual_bytes(tmp_path: Path) -> None:
    archive = tmp_path / "EAAI_anonymous_manuscript_sources.zip"
    payload = b"anonymous source"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("main_anonymized.tex", payload)
    records = archive_member_records(archive, archive.name, anonymous=True)
    assert records == [
        {
            "filename": f"{archive.name}::main_anonymized.tex",
            "role": "anonymous_manuscript_tex",
            "size": len(payload),
            "sha256": sha256_bytes(payload),
            "anonymous": True,
            "source_generated_from": [archive.name],
        }
    ]


def test_declared_manuscript_dependencies_exist() -> None:
    paper = ROOT / "paper"
    tex = materialize_anonymous_tex((paper / "main.tex").read_text(encoding="utf-8"))
    dependencies = latex_dependencies(tex, paper)
    assert dependencies["tables"]
    assert dependencies["figures"]
    assert all((paper / path).is_file() for paths in dependencies.values() for path in paths)


def test_build_log_masks_user_home() -> None:
    home = str(Path.home())
    sanitized = sanitize_build_log(f"input={home}\\AppData\\MiKTeX\nwrapped=C:/Use\nrs/Ercan/AppData")
    assert str(Path.home()) not in sanitized
    assert "<USER_HOME>" in sanitized
    assert "Ercan" not in sanitized
