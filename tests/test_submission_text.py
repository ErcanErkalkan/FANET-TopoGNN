from pathlib import Path

from scripts.validate_submission_text import (
    acronym_tokens,
    parse_highlights,
    run_validation,
    undefined_acronyms,
    word_count,
)


ROOT = Path(__file__).resolve().parents[1]


def test_acronym_definition_detection() -> None:
    text = "An unmanned aerial vehicle (UAV) network is evaluated."
    assert acronym_tokens(text) == ["UAV"]
    assert undefined_acronyms(text) == []
    assert undefined_acronyms("A UAV network is evaluated.") == ["UAV"]
    assert "F1" not in acronym_tokens("The event F1 score is reported.")


def test_highlights_are_line_based_and_count_spaces() -> None:
    parsed = parse_highlights("- First supported result.\n- Second supported result.\n")
    assert parsed == ["First supported result.", "Second supported result."]
    assert [len(item) for item in parsed] == [23, 24]


def test_word_count_handles_hyphenated_terms() -> None:
    assert word_count("A cross-domain stress test uses 20-seed evaluation.") == 7


def test_repository_submission_text_is_synchronized() -> None:
    report = run_validation(ROOT)
    assert report["status"] == "pass", report["checks"]
    assert report["abstract_word_count"] <= 250
    assert report["undefined_abstract_acronyms"] == []
    assert 1 <= len(report["keywords"]) <= 6
    assert all(length <= 85 for length in report["highlight_lengths_including_spaces"])
