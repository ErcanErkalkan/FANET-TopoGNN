from pathlib import Path

from scripts.audit_manuscript_route import (
    abstract_word_count,
    missing_inputs,
    run_audit,
    unresolved_abstract_acronyms,
)


ROOT = Path(__file__).resolve().parents[1]


def test_abstract_is_below_safety_limit_and_defines_acronyms() -> None:
    text = (ROOT / "paper/main.tex").read_text(encoding="utf-8")
    assert abstract_word_count(text) <= 230
    assert unresolved_abstract_acronyms(text) == []


def test_every_manuscript_input_exists() -> None:
    text = (ROOT / "paper/main.tex").read_text(encoding="utf-8")
    assert missing_inputs(text) == []


def test_route_and_evidence_audit_passes() -> None:
    report = run_audit()
    assert report["passed"], report["checks"]
