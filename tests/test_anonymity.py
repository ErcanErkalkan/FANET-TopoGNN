from __future__ import annotations

from pathlib import Path

from matplotlib import pyplot as plt
from pypdf import PdfReader, PdfWriter

from scripts.scan_anonymity import (
    extract_anonymous_tex,
    load_pattern_config,
    scan_paths,
    scan_text,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "scripts" / "anonymity_patterns.json"


def _private_given_name() -> str:
    return "Er" + "can"


def _credit_heading() -> str:
    return "CRe" + "diT authorship contribution statement"


def test_forbidden_name_in_temporary_file_is_rejected(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.txt"
    candidate.write_text(f"Author: {_private_given_name()}\n", encoding="utf-8")

    report = scan_paths([candidate], root=tmp_path, config_path=CONFIG)

    assert report["status"] == "fail"
    assert any(item["pattern_id"] == "author_given_name" for item in report["findings"]), report


def test_clean_file_passes(tmp_path: Path) -> None:
    candidate = tmp_path / "clean.md"
    candidate.write_text("Anonymous reproducibility package.\n", encoding="utf-8")

    report = scan_paths([candidate], root=tmp_path, config_path=CONFIG)

    assert report["status"] == "pass"
    assert report["findings"] == []
    assert report["errors"] == []


def test_anonymous_credit_section_is_rejected() -> None:
    config = load_pattern_config(CONFIG)
    text = f"\\section*{{{_credit_heading()}}}\nAnonymous contributor: Writing.\n"

    findings = scan_text(text, "paper/main_anonymized.tex", config)

    assert any(item["pattern_id"] == "credit_section" for item in findings)


def test_pdf_text_identity_leak_is_rejected(tmp_path: Path) -> None:
    candidate = tmp_path / "leaking.pdf"
    figure = plt.figure(figsize=(3, 2))
    figure.text(0.1, 0.5, f"Author: {_private_given_name()}")
    figure.savefig(candidate, format="pdf")
    plt.close(figure)

    report = scan_paths([candidate], root=tmp_path, config_path=CONFIG)

    assert report["status"] == "fail"
    assert any(item["pattern_id"] == "author_given_name" for item in report["findings"]), report


def test_pdf_metadata_identity_leak_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    candidate = tmp_path / "metadata-leak.pdf"
    figure = plt.figure(figsize=(3, 2))
    figure.text(0.1, 0.5, "Anonymous document")
    figure.savefig(source, format="pdf")
    plt.close(figure)
    reader = PdfReader(source)
    writer = PdfWriter()
    writer.append_pages_from_reader(reader)
    writer.add_metadata({"/Author": _private_given_name()})
    with candidate.open("wb") as handle:
        writer.write(handle)

    report = scan_paths([candidate], root=tmp_path, config_path=CONFIG)

    assert report["status"] == "fail"
    assert any(item["pattern_id"] == "author_given_name" for item in report["findings"]), report


def test_effective_anonymous_tex_hides_credit_but_title_page_retains_it() -> None:
    main_text = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    title_page = (ROOT / "paper" / "title_page.tex").read_text(encoding="utf-8")
    anonymous_text = extract_anonymous_tex(main_text)

    assert _private_given_name() not in anonymous_text
    assert _credit_heading() not in anonymous_text
    assert _private_given_name() in title_page
    assert _credit_heading() in title_page
