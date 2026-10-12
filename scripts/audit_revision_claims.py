from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HARD_RULES = {
    "cross_domain_validation": re.compile(r"\bcross[- ]domain validation\b", re.I),
    "real_world_fanet_validation": re.compile(r"\breal[- ]world FANET validation\b", re.I),
    "deployment_ready": re.compile(r"\bdeployment[- ]ready\b", re.I),
    "demonstrated_transferability": re.compile(r"\bdemonstrat(?:e|es|ed|ing) transferability\b", re.I),
    "engineering_application_heading": re.compile(r"\\textbf\{Engineering application:\}", re.I),
    "source_gated_section": re.compile(r"\\(?:sub)*section\{Source-Gated Kinetic-TopoGuard\}", re.I),
    "learned_source_selection": re.compile(r"\blearned source selection\b", re.I),
}

SOFT_RULES = {
    "real_world": re.compile(r"\breal[- ]world\b", re.I),
    "field_validation": re.compile(r"\bfield[- ]validat(?:e|ed|ion)\b", re.I),
    "engineering_application": re.compile(r"\bengineering application\b", re.I),
    "transferability": re.compile(r"\btransferability\b", re.I),
    "onboard_latency": re.compile(r"\bonboard latency\b", re.I),
    "gating_term": re.compile(r"\bgating\b", re.I),
}

NEGATION_WINDOW = re.compile(r"\b(?:no|not|without|does not|cannot|is not|are not|rather than)\b", re.I)


def line_matches(path: Path, rules: dict[str, re.Pattern[str]], allow_negated: bool) -> list[dict]:
    findings = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        for rule_id, pattern in rules.items():
            for match in pattern.finditer(line):
                if allow_negated:
                    left = line[max(0, match.start() - 80):match.start()]
                    if NEGATION_WINDOW.search(left):
                        continue
                findings.append({
                    "rule_id": rule_id,
                    "line": lineno,
                    "text": line.strip(),
                })
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit post-review manuscript claim language.")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    payload = {"files": [], "hard_violations": 0, "soft_warnings": 0}
    for path in args.paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        hard = line_matches(path, HARD_RULES, allow_negated=False)
        soft = line_matches(path, SOFT_RULES, allow_negated=True)
        payload["files"].append({"path": str(path), "hard": hard, "soft": soft})
        payload["hard_violations"] += len(hard)
        payload["soft_warnings"] += len(soft)

    payload["status"] = "pass" if payload["hard_violations"] == 0 else "fail"
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    print(rendered)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    return 0 if payload["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
