from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable


@dataclass(frozen=True)
class ClaimRule:
    rule_id: str
    description: str
    pattern: str
    negation_pattern: str | None = None


CLAIM_RULES = (
    ClaimRule(
        "forestry_measured_rf",
        "Forestry supplies measured motion, not measured peer RF.",
        r"\bforestry\b.{0,100}\b(?:provides?|contains?|supplies?|validates?|measures?)\b.{0,80}\bmeasured\s+(?:peer[- ]?)?(?:rf|radio|channel|link)",
        r"\b(?:no|not|without|does not)\b.{0,80}\bmeasured\s+(?:peer[- ]?)?(?:rf|radio|channel|link)",
    ),
    ClaimRule(
        "aerpaw_inter_uav",
        "AERPAW Datasets 22/23 are UAV-to-infrastructure, not inter-UAV evidence.",
        r"\baerpaw\b.{0,100}\b(?:provides?|supports?|validates?|measures?)\b.{0,80}\b(?:inter[- ]uav|uav[- ]to[- ]uav|peer[- ]link)",
        r"\b(?:not|no|does not|cannot)\b.{0,60}\b(?:inter[- ]uav|uav[- ]to[- ]uav|peer[- ]link)",
    ),
    ClaimRule(
        "miluv_ip_pdr",
        "MILUV supports measured UWB topology, not IP packet delivery/PDR.",
        r"\bmiluv\b.{0,100}\b(?:provides?|supports?|validates?|measures?|establishes?)\b.{0,80}\b(?:ip[- ]?pdr|ip\s+packet delivery|packet delivery ratio)",
        r"\b(?:not|no|does not|cannot)\b.{0,80}\b(?:ip[- ]?pdr|ip\s+packet delivery|packet delivery ratio)",
    ),
    ClaimRule(
        "transported_wines_same_site",
        "WiNES-on-forestry results are transported sensitivity, not same-site calibration.",
        r"\b(?:transported.{0,40}(?:wines|60\s*ghz)|(?:wines|60\s*ghz).{0,40}transported)\b.{0,100}\bsame[- ]site calibration\b",
        r"\b(?:not|no|is not|cannot be)\b.{0,50}\bsame[- ]site calibration\b",
    ),
    ClaimRule(
        "counterfactual_radius_measured_rf",
        "Counterfactual radius graphs are derived scenarios, not measured RF.",
        r"\b(?:counterfactual radius|radius[- ]graph)\b.{0,100}\b(?:is|are|provides?|represents?)\b.{0,50}\bmeasured\s+(?:rf|radio|link)",
        r"\b(?:not|no|does not)\b.{0,60}\bmeasured\s+(?:rf|radio|link)",
    ),
)


def _plain_tex(text: str) -> str:
    text = re.sub(r"(?<!\\)%.*", "", text)
    text = re.sub(r"\\(?:cite|ref|label|input|includegraphics)(?:\[[^]]*\])?\{[^}]*\}", " ", text)
    text = re.sub(r"\\[A-Za-z@]+", " ", text)
    return re.sub(r"[{}$~]", " ", text)


def _claim_units(text: str) -> Iterable[tuple[int, str]]:
    plain = _plain_tex(text)
    lines = plain.splitlines()
    paragraph: list[tuple[int, str]] = []
    for number, line in enumerate(lines, start=1):
        normalized = " ".join(line.split())
        if normalized:
            paragraph.append((number, normalized))
        elif paragraph:
            joined = " ".join(value for _, value in paragraph)
            for sentence in re.split(r"(?<=[.!?])\s+", joined):
                if sentence.strip():
                    yield paragraph[0][0], sentence.strip()
            paragraph = []
    if paragraph:
        joined = " ".join(value for _, value in paragraph)
        for sentence in re.split(r"(?<=[.!?])\s+", joined):
            if sentence.strip():
                yield paragraph[0][0], sentence.strip()


def lint_external_claims(text: str, source: str = "<text>") -> list[dict[str, object]]:
    violations: list[dict[str, object]] = []
    for line, unit in _claim_units(text):
        lowered = unit.lower()
        for rule in CLAIM_RULES:
            if not re.search(rule.pattern, lowered, flags=re.I | re.S):
                continue
            if rule.negation_pattern and re.search(rule.negation_pattern, lowered, flags=re.I | re.S):
                continue
            violations.append(
                {
                    "rule_id": rule.rule_id,
                    "source": source,
                    "line": int(line),
                    "text": unit,
                    "boundary": rule.description,
                }
            )
    return violations


def lint_external_claim_path(path: str | Path) -> list[dict[str, object]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"claim-lint source is missing: {source}")
    return lint_external_claims(source.read_text(encoding="utf-8"), source.as_posix())


def claim_rules_manifest() -> list[dict[str, str | None]]:
    return [
        {
            "rule_id": rule.rule_id,
            "description": rule.description,
            "pattern": rule.pattern,
            "negation_pattern": rule.negation_pattern,
        }
        for rule in CLAIM_RULES
    ]
