#!/usr/bin/env python3
"""Validate EAAI portal text, editable highlights, and identified/anonymous boundaries."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.scan_anonymity import extract_anonymous_tex


DEFAULT_REPORT = ROOT / "docs" / "eaai_revision" / "submission_text_audit.json"


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def _command_argument(text: str, command: str) -> str:
    marker = f"\\{command}{{"
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"missing \\{command} command")
    pos = start + len(marker)
    depth = 1
    for index in range(pos, len(text)):
        if text[index] == "{" and text[index - 1] != "\\":
            depth += 1
        elif text[index] == "}" and text[index - 1] != "\\":
            depth -= 1
            if depth == 0:
                return text[pos:index]
    raise ValueError(f"unterminated \\{command} command")


def _environment(text: str, name: str) -> str:
    match = re.search(
        rf"\\begin\{{{re.escape(name)}\}}(.*?)\\end\{{{re.escape(name)}\}}",
        text,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError(f"missing {name} environment")
    return match.group(1).strip()


def latex_plain(text: str) -> str:
    percent_token = "__LITERAL_PERCENT__"
    text = text.replace(r"\%", percent_token)
    text = re.sub(r"(?<!\\)%.*", " ", text)
    text = text.replace(r"\\", " ").replace("~", " ")
    text = re.sub(r"\\(?:text|emph|mathrm|mathbf|operatorname|texttt|url)\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^\]]*\])?", " ", text)
    text = text.replace("$", "").replace("{", "").replace("}", "")
    text = text.replace(percent_token, "%")
    return re.sub(r"\s+", " ", text).strip()


def markdown_section(text: str, heading: str) -> str:
    match = re.search(
        rf"^## {re.escape(heading)}\s*\n(.*?)(?=^## |\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if match is None:
        raise ValueError(f"missing Markdown section: {heading}")
    return match.group(1).strip()


def metadata_field(text: str, label: str) -> str:
    match = re.search(rf"^- {re.escape(label)}:\s*(.+)$", text, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"missing metadata field: {label}")
    return match.group(1).strip()


def acronym_tokens(text: str) -> list[str]:
    return sorted(set(re.findall(r"\b[A-Z]{2,}[A-Z0-9]*\b", text)))


def undefined_acronyms(text: str) -> list[str]:
    unresolved: list[str] = []
    for token in acronym_tokens(text):
        definition = re.search(
            rf"\b(?:[A-Za-z][A-Za-z-]*\s+){{1,8}}\({re.escape(token)}\)", text
        )
        if definition is None:
            unresolved.append(token)
    return unresolved


def word_count(text: str) -> int:
    return len(re.findall(r"\b[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*\b", text))


def parse_highlights(text: str) -> list[str]:
    highlights: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        highlights.append(re.sub(r"^(?:[-*]|\d+[.)])\s+", "", stripped))
    return highlights


def parse_metadata_keywords(text: str) -> list[str]:
    body = markdown_section(text, "Keywords")
    return [
        match.group(1).strip()
        for line in body.splitlines()
        if (match := re.match(r"^\s*\d+[.)]\s+(.+?)\s*$", line))
    ]


def parse_manuscript_keywords(text: str) -> list[str]:
    return [latex_plain(item) for item in re.split(r"\\sep", _environment(text, "keyword")) if latex_plain(item)]


def _run_text_command(command: list[str]) -> tuple[bool, str]:
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    output = completed.stdout if completed.returncode == 0 else completed.stderr
    return completed.returncode == 0, output


def graphical_abstract_checks(path: Path) -> tuple[list[Check], dict]:
    info_ok, info = _run_text_command(["pdfinfo", str(path)]) if path.is_file() else (False, "missing file")
    text_ok, extracted = _run_text_command(["pdftotext", "-layout", str(path), "-"]) if path.is_file() else (False, "missing file")
    pages = re.search(r"^Pages:\s+(\d+)", info, flags=re.MULTILINE)
    size = re.search(r"^Page size:\s+([0-9.]+) x ([0-9.]+) pts", info, flags=re.MULTILINE)
    width = float(size.group(1)) if size else 0.0
    height = float(size.group(2)) if size else 0.0
    required_text = ["benchmark", "locked", "domain", "packet", "host-side", "do not consistently"]
    lowered = extracted.lower()
    checks = [
        Check("graphical_abstract_pdf", path.suffix.lower() == ".pdf" and path.is_file() and path.stat().st_size >= 10_000, f"bytes={path.stat().st_size if path.is_file() else 0}"),
        Check("graphical_abstract_single_page", info_ok and pages is not None and pages.group(1) == "1", info.strip()),
        Check("graphical_abstract_dimensions", width >= 800 and height >= 300, f"{width} x {height} pt"),
        Check("graphical_abstract_readable_text", text_ok and all(term in lowered for term in required_text), extracted.strip()),
        Check("graphical_abstract_no_superiority_claim", "superior" not in lowered and "outperform" not in lowered, extracted.strip()),
    ]
    return checks, {"pdfinfo": info.strip(), "extracted_text": extracted.strip()}


def run_validation(root: Path = ROOT) -> dict:
    manuscript = (root / "paper/main.tex").read_text(encoding="utf-8")
    title_page = (root / "paper/title_page.tex").read_text(encoding="utf-8")
    metadata = (root / "EAAI_PORTAL_METADATA.md").read_text(encoding="utf-8")
    cover = (root / "EAAI_COVER_LETTER.md").read_text(encoding="utf-8")
    highlight_text = (root / "EAAI_HIGHLIGHTS.txt").read_text(encoding="utf-8")

    manuscript_title = latex_plain(_command_argument(manuscript, "title"))
    title_page_title = latex_plain(_command_argument(title_page, "title"))
    metadata_title = metadata_field(metadata, "Manuscript title")
    cover_match = re.search(r'manuscript\s+"([^"]+)"', cover, flags=re.IGNORECASE)
    cover_title = cover_match.group(1).strip() if cover_match else ""

    manuscript_abstract = latex_plain(_environment(manuscript, "abstract"))
    metadata_abstract = re.sub(r"\s+", " ", markdown_section(metadata, "Abstract")).strip()
    abstract_acronyms = acronym_tokens(manuscript_abstract)
    undefined_abstract = undefined_acronyms(manuscript_abstract)

    manuscript_keywords = parse_manuscript_keywords(manuscript)
    metadata_keywords = parse_metadata_keywords(metadata)
    keyword_acronyms = sorted(set(token for keyword in metadata_keywords for token in acronym_tokens(keyword)))
    keywords_english = all(re.fullmatch(r"[A-Za-z0-9 -]+", item) for item in metadata_keywords)

    manuscript_highlights = [latex_plain(item) for item in re.findall(r"\\item\s+([^\n]+)", _environment(manuscript, "highlights"))]
    editable_highlights = parse_highlights(highlight_text)
    highlight_acronyms = sorted(set(token for item in editable_highlights for token in acronym_tokens(item)))
    highlight_lengths = [len(item) for item in editable_highlights]

    title_page_required = [
        "Ercan Erkalkan", "Marmara University", "Corresponding author", "ercan.erkalkan@marmara.edu.tr",
        "CRediT authorship contribution statement", "Funding", "Declaration of competing interest",
        "Data and code availability", "Declaration of generative AI and AI-assisted technologies",
    ]
    anonymous_declaration_names = [
        "CRediT authorship contribution statement", "Declaration of competing interest",
        "Declaration of generative AI and AI-assisted technologies",
    ]
    anonymous_branch = extract_anonymous_tex(manuscript)
    identified_data = [
        "https://github.com/ErcanErkalkan/FANET-TopoGNN", "10.5281/zenodo.20226053",
        "10.5281/zenodo.14701641", "10.25452/figshare.plus.28386041.v1",
    ]
    disclosure_clauses = [
        "OpenAI GPT-based tooling", "language refinement", "software-development assistance",
        "local reproducibility cross-checks", "takes responsibility for the submitted work",
    ]
    normalized_anonymous = re.sub(r"\s+", " ", anonymous_branch)
    normalized_disclosures = [re.sub(r"\s+", " ", text) for text in [manuscript, title_page, metadata]]

    graphical_checks, graphical_details = graphical_abstract_checks(
        root / "paper/figures/plantuml/graphical_abstract.pdf"
    )
    checks = [
        Check("title_acronym", not acronym_tokens(manuscript_title), str(acronym_tokens(manuscript_title))),
        Check("title_equality", len({manuscript_title, title_page_title, metadata_title, cover_title}) == 1, str([manuscript_title, title_page_title, metadata_title, cover_title])),
        Check("abstract_word_count", word_count(manuscript_abstract) <= 250, str(word_count(manuscript_abstract))),
        Check("abstract_acronym_definitions", not undefined_abstract, str(undefined_abstract)),
        Check("metadata_manuscript_abstract_equality", manuscript_abstract == metadata_abstract, "character-for-character plain-text equality"),
        Check("keyword_count", 1 <= len(metadata_keywords) <= 6, str(len(metadata_keywords))),
        Check("keyword_english", keywords_english, str(metadata_keywords)),
        Check("keyword_acronyms", not keyword_acronyms, str(keyword_acronyms)),
        Check("keyword_equality", manuscript_keywords == metadata_keywords, str([manuscript_keywords, metadata_keywords])),
        Check("highlight_count", 3 <= len(editable_highlights) <= 5, str(len(editable_highlights))),
        Check("highlight_length", bool(editable_highlights) and all(length <= 85 for length in highlight_lengths), str(highlight_lengths)),
        Check("highlight_acronyms", not highlight_acronyms, str(highlight_acronyms)),
        Check("highlight_equality", manuscript_highlights == editable_highlights, str([manuscript_highlights, editable_highlights])),
        Check("title_page_required_fields", all(item in title_page for item in title_page_required), str([item for item in title_page_required if item not in title_page])),
        Check("anonymous_identified_declarations_absent", not any(item in anonymous_branch for item in anonymous_declaration_names), "identified declarations are title-page/identified-only"),
        Check("identified_data_availability", all(item in title_page or item in manuscript for item in identified_data), str(identified_data)),
        Check("anonymous_data_availability_masked", "Author-identifying archive metadata is withheld" in normalized_anonymous, "generic masked statement present"),
        Check("generative_ai_disclosure", all(all(clause in text for clause in disclosure_clauses) for text in normalized_disclosures), str(disclosure_clauses)),
        *graphical_checks,
    ]
    return {
        "schema_version": 1,
        "status": "pass" if all(check.passed for check in checks) else "fail",
        "abstract_word_count": word_count(manuscript_abstract),
        "abstract_acronyms": abstract_acronyms,
        "undefined_abstract_acronyms": undefined_abstract,
        "keywords": metadata_keywords,
        "highlight_lengths_including_spaces": highlight_lengths,
        "graphical_abstract": graphical_details,
        "checks": [asdict(check) for check in checks],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = run_validation(ROOT)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Submission text validation: {report['status'].upper()}")
    for check in report["checks"]:
        print(f"[{check['passed'] and 'PASS' or 'FAIL'}] {check['name']}: {check['detail']}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
