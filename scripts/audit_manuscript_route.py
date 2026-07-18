#!/usr/bin/env python3
"""Audit route consistency and evidence-bound manuscript claims for the EAAI revision."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "paper" / "main.tex"
DECISION = ROOT / "docs" / "eaai_revision" / "scientific_positioning_decision.json"
REPORT = ROOT / "docs" / "eaai_revision" / "MANUSCRIPT_ROUTE_AUDIT.json"


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def _environment(text: str, name: str) -> str:
    match = re.search(
        rf"\\begin\{{{re.escape(name)}\}}(.*?)\\end\{{{re.escape(name)}\}}",
        text,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError(f"missing {name} environment")
    return match.group(1)


def _latex_plain(text: str) -> str:
    text = re.sub(r"(?<!\\)%.*", " ", text)
    text = re.sub(r"\\(?:SI|text|emph|mathrm|mathbf|operatorname)\{([^{}]*)\}(?:\{([^{}]*)\})?", r" \1 \2 ", text)
    text = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^\]]*\])?", " ", text)
    text = text.replace("$", " ").replace("{", " ").replace("}", " ")
    text = re.sub(r"[~\\]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def abstract_word_count(text: str) -> int:
    plain = _latex_plain(_environment(text, "abstract"))
    return len(re.findall(r"\b[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*\b", plain))


def unresolved_abstract_acronyms(text: str) -> list[str]:
    plain = _latex_plain(_environment(text, "abstract"))
    tokens = sorted(set(re.findall(r"\b[A-Z]{2,}[A-Z0-9]*\b", plain)))
    unresolved: list[str] = []
    for token in tokens:
        definition = re.search(rf"\b(?:[A-Za-z-]+\s+){{1,8}}\({re.escape(token)}\)", plain)
        if definition is None:
            unresolved.append(token)
    return unresolved


def missing_inputs(text: str) -> list[str]:
    missing: list[str] = []
    for raw in re.findall(r"\\input\{([^}]+)\}", text):
        path = ROOT / "paper" / raw
        if path.suffix == "":
            path = path.with_suffix(".tex")
        if not path.is_file():
            missing.append(path.relative_to(ROOT).as_posix())
    return missing


def _contains_number(text: str, value: float, digits: int) -> bool:
    rendered = f"{value:.{digits}f}"
    return rendered in text


def evidence_number_checks(text: str) -> list[Check]:
    checks: list[Check] = []

    locked = pd.read_csv(ROOT / "outputs/eaai_locked_confirmatory/paired_tests.csv")
    locked = locked[
        (locked["candidate"] == "Source-Gated Kinetic-TopoGuard")
        & (locked["reference"] == "Current-state ExtraTrees")
    ]
    event = locked[locked["metric"] == "Alert_Event_F1"].iloc[0]
    false_rate = locked[locked["metric"] == "False_Alert_Events_per_minute"].iloc[0]
    locked_values = [
        (event.paired_mean_difference, 4),
        (event.bootstrap_ci95_low, 4),
        (event.bootstrap_ci95_high, 4),
        (event.paired_permutation_holm_pvalue, 4),
        (false_rate.paired_mean_difference, 2),
    ]
    checks.append(Check("locked_confirmatory_numbers", all(_contains_number(text, v, d) for v, d in locked_values), str(locked_values)))

    factorial = pd.read_csv(ROOT / "outputs/factorial_feature_ablation_20seed/paired_tests.csv")
    factorial = factorial[
        (factorial["candidate_feature_sources"] == "graph+topology+kinematic")
        & (factorial["reference_feature_sources"] == "current-only")
    ]
    f_event = factorial[factorial["metric"] == "Alert_Event_F1"].iloc[0]
    factorial_values = [
        (f_event.paired_mean_difference, 4),
        (f_event.bootstrap_ci95_low, 4),
        (f_event.bootstrap_ci95_high, 4),
        (f_event.paired_permutation_holm_adjusted_pvalue, 4),
    ]
    checks.append(Check("factorial_numbers", all(_contains_number(text, v, d) for v, d in factorial_values), str(factorial_values)))

    residual = json.loads((ROOT / "outputs/residual_branch_audit/residual_decision.json").read_text(encoding="utf-8"))
    residual_ok = (
        _contains_number(text, 100.0 * residual["alpha_zero_seed_ratio"], 0)
        and _contains_number(text, residual["paired_mae_difference_mean"], 6)
        and str(residual["improved_seed_count"]) in text
        and str(residual["worsened_seed_count"]) in text
    )
    checks.append(Check("residual_numbers", residual_ok, json.dumps(residual, sort_keys=True)))

    closed = pd.read_csv(ROOT / "outputs/closed_loop_controller/paired_tests.csv")
    supported = int(closed["Engineering_Benefit_Supported"].astype(bool).sum())
    checks.append(Check("closed_loop_numbers", supported == 0 and "none of 360" in text.lower(), f"supported={supported}, rows={len(closed)}"))

    runtime = pd.read_csv(ROOT / "outputs/edge_runtime_benchmark/latency_summary.csv")
    runtime = runtime[(runtime["thread_mode"] == "default") & (runtime["n_nodes"] == 30) & (runtime["stage"] == "total_host_loop")]
    expected = {row.model: row.p95_ms for row in runtime.itertuples()}
    runtime_ok = all(_contains_number(text, expected[name], 2) for name in ["FANET-TopoGNN", "Kinetic-TopoGuard", "Current-state ExtraTrees", "Source-Gated Kinetic-TopoGuard"])
    checks.append(Check("runtime_numbers", runtime_ok, json.dumps(expected, sort_keys=True)))
    return checks


def latex_build_summary() -> dict:
    summary: dict[str, dict] = {}
    for stem in ("main", "main_anonymized"):
        log_path = ROOT / "paper" / f"{stem}.log"
        pdf_path = ROOT / "paper" / f"{stem}.pdf"
        log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
        summary[stem] = {
            "pdf_exists": pdf_path.is_file(),
            "pdf_bytes": pdf_path.stat().st_size if pdf_path.is_file() else None,
            "latex_warning_count": len(re.findall(r"LaTeX Warning:", log)),
            "package_warning_count": len(re.findall(r"Package .* Warning:", log)),
            "undefined_reference_count": len(re.findall(r"(?:undefined references|Reference .* undefined)", log, flags=re.IGNORECASE)),
            "undefined_citation_count": len(re.findall(r"Citation.*undefined", log, flags=re.IGNORECASE)),
            "overfull_box_count": len(re.findall(r"Overfull \\[hv]box", log)),
            "underfull_box_count": len(re.findall(r"Underfull \\[hv]box", log)),
        }
    return summary


def run_audit() -> dict:
    text = MANUSCRIPT.read_text(encoding="utf-8")
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    title = re.search(r"\\title\{(.*?)\}", text, flags=re.DOTALL)
    title_plain = _latex_plain(title.group(1)) if title else ""
    word_count = abstract_word_count(text)
    acronyms = unresolved_abstract_acronyms(text)
    missing = missing_inputs(text)
    route_b = decision["selected_route"].startswith("ROUTE_B")
    builds = latex_build_summary()
    clean_builds = all(
        item["pdf_exists"]
        and item["undefined_reference_count"] == 0
        and item["undefined_citation_count"] == 0
        and item["overfull_box_count"] == 0
        for item in builds.values()
    )

    checks = [
        Check("decision_route_b", route_b, decision["selected_route"]),
        Check("title_matches_decision", title_plain == decision["title_recommendation"], title_plain),
        Check("abstract_word_count", word_count <= 230, str(word_count)),
        Check("abstract_acronyms_defined", not acronyms, str(acronyms)),
        Check("all_inputs_exist", not missing, str(missing)),
        Check("required_engineering_section", r"\section{Engineering implications and deployment boundaries}" in text, "section present"),
        Check("results_and_discussion_separate", r"\section{Results}" in text and r"\section{Discussion}" in text, "separate sections present"),
        Check("unsupported_title_terms_absent", "Topological Artificial Intelligence" not in title_plain, title_plain),
        Check("no_deployment_ready_claim", "is deployment-ready" not in text.lower(), "bounded wording"),
        Check("route_consistent_conclusion", "reproducible benchmark and cross-domain stress test" in _environment(text, "document").lower(), "benchmark route stated"),
        Check("latex_builds_clean", clean_builds, json.dumps(builds, sort_keys=True)),
        *evidence_number_checks(text),
    ]
    return {
        "manuscript": MANUSCRIPT.relative_to(ROOT).as_posix(),
        "decision": DECISION.relative_to(ROOT).as_posix(),
        "abstract_word_count": word_count,
        "unresolved_abstract_acronyms": acronyms,
        "latex_builds": builds,
        "passed": all(check.passed for check in checks),
        "checks": [asdict(check) for check in checks],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    report = run_audit()
    output = args.report if args.report.is_absolute() else ROOT / args.report
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
