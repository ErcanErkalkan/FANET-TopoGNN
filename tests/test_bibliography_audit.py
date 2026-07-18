from pathlib import Path

from scripts.audit_bibliography import audit_bibliography, citation_keys, parse_bibtex


ROOT = Path(__file__).resolve().parents[1]


def test_parser_handles_nested_braces_and_citations() -> None:
    entries, errors = parse_bibtex(
        "@article{x, title={{A} Nested Title}, author={A and B}, year={2024}}"
    )
    assert not errors
    assert entries[0].fields["title"] == "{A} Nested Title"
    assert citation_keys(r"Text \cite{x,y} and \citep{z}.") == {"x", "y", "z"}


def test_duplicate_doi_and_title_are_hard_failures(tmp_path: Path) -> None:
    bib = tmp_path / "refs.bib"
    tex = tmp_path / "main.tex"
    bib.write_text(
        """@article{a, title={Same title}, doi={10.1/example}, year={2020}}
@article{b, title={{Same} title}, doi={https://doi.org/10.1/EXAMPLE}, year={2021}}
""",
        encoding="utf-8",
    )
    tex.write_text(r"\cite{a,b}", encoding="utf-8")
    payload = audit_bibliography(bib, [tex])
    assert payload["status"] == "fail"
    assert payload["duplicate_dois"]
    assert payload["duplicate_titles"]


def test_missing_citation_key_is_rejected(tmp_path: Path) -> None:
    bib = tmp_path / "refs.bib"
    tex = tmp_path / "main.tex"
    bib.write_text("@article{present, title={Present}, year={2024}}", encoding="utf-8")
    tex.write_text(r"\cite{missing}", encoding="utf-8")
    payload = audit_bibliography(bib, [tex])
    assert payload["missing_citation_keys"] == ["missing"]
    assert payload["status"] == "fail"


def test_repository_bibliography_has_complete_required_coverage() -> None:
    payload = audit_bibliography(ROOT / "paper/cas-refs.bib", [ROOT / "paper/main.tex"])
    assert payload["status"] == "pass", payload["hard_failures"]
    assert all(
        item["present"] and item["cited"]
        for item in payload["required_group_coverage"].values()
    )
