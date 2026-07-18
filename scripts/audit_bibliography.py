from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BibEntry:
    entry_type: str
    key: str
    fields: dict[str, str]


REQUIRED_GROUPS: dict[str, tuple[str, ...]] = {
    "extra_trees": ("geurts2006extremely",),
    "scikit_learn": ("pedregosa2011scikit",),
    "pytorch": ("paszke2019pytorch",),
    "pytorch_geometric_optional": ("fey2019fast",),
    "simpy": ("simpy412",),
    "gcn": ("kipf2017semi",),
    "gat": ("velickovic2018graph",),
    "graphsage": ("hamilton2017inductive",),
    "t_gcn": ("zhao2020tgcn",),
    "stgcn": ("yu2018stgcn",),
    "tgn": ("rossi2020tgn",),
    "persistence_images": ("adams2017persistence",),
    "persistence_stability": ("cohensteiner2005stability",),
    "treeshap": ("lundberg2020local",),
    "permutation_importance": ("fisher2019all",),
    "brier_score": ("brier1950verification",),
    "probability_calibration_ece": ("guo2017calibration",),
    "event_evaluation": ("tatbul2018precision",),
    "fanet_connectivity_and_prediction": ("BEKMEZCI20131254", "pasandideh2023review", "pu2026must"),
    "uav_relay": ("yanmaz2022positioning",),
    "radio_models": ("gudmundson1991correlation", "nakagami1960mdistribution", "goldsmith2005wireless"),
    "forestry_dataset": ("araujo2025multidrone",),
    "aerpaw_datasets": ("aerpawDataset22", "aerpawDataset23"),
    "wines_dataset_and_method": ("winesUavToUav60GHz", "polese2020experimental"),
    "miluv_dataset_and_article": ("miluvDataset", "shalaby2026miluv"),
    "archived_software_release": ("fanetTopoGNNSoftware",),
}


# Metadata checked against the linked publisher, proceedings, or archive record on
# 2026-07-16.  This is deliberately explicit: the routine never treats a syntactically
# plausible DOI as verified merely because it resolves.
VERIFIED_DOIS: dict[str, str] = {
    "10.1007/s10994-006-6226-1": "https://link.springer.com/article/10.1007/s10994-006-6226-1",
    "10.1016/j.adhoc.2012.12.004": "https://www.sciencedirect.com/science/article/abs/pii/S1570870512002193",
    "10.1145/321879.321884": "https://sigmod.org/publications/dblp/db/journals/jacm/Tarjan75.html",
    "10.3390/drones7070448": "https://doi.org/10.3390/drones7070448",
    "10.3390/drones6110334": "https://www.mdpi.com/2504-446X/6/11/334",
    "10.1109/lnet.2025.3542762": "https://ieeexplore.ieee.org/document/10891581/",
    "10.1109/tmc.2022.3146881": "https://ieeexplore.ieee.org/document/9697395/",
    "10.1016/j.adhoc.2025.103801": "https://www.sciencedirect.com/science/article/pii/S1570870525000496",
    "10.1016/j.engappai.2026.115458": "https://www.sciencedirect.com/science/article/pii/S0952197626017422",
    "10.1002/rob.22157": "https://onlinelibrary.wiley.com/doi/10.1002/rob.22157",
    "10.1109/tkde.2026.3677398": "https://doi.org/10.1109/TKDE.2026.3677398",
    "10.1109/tits.2019.2935152": "https://doi.org/10.1109/TITS.2019.2935152",
    "10.24963/ijcai.2018/505": "https://www.ijcai.org/proceedings/2018/505",
    "10.1145/1064092.1064133": "https://doi.org/10.1145/1064092.1064133",
    "10.1038/s42256-019-0138-9": "https://www.nature.com/articles/s42256-019-0138-9",
    "10.1175/1520-0493(1950)078<0001:vofeit>2.0.co;2": "https://doi.org/10.1175/1520-0493(1950)078%3C0001:VOFEIT%3E2.0.CO;2",
    "10.1049/el:19911328": "https://doi.org/10.1049/el:19911328",
    "10.1016/b978-0-08-009306-2.50005-4": "https://doi.org/10.1016/B978-0-08-009306-2.50005-4",
    "10.1017/cbo9780511841224": "https://doi.org/10.1017/CBO9780511841224",
    "10.1016/j.adhoc.2022.102800": "https://doi.org/10.1016/j.adhoc.2022.102800",
    "10.5281/zenodo.14701641": "https://doi.org/10.5281/zenodo.14701641",
    "10.1177/02783649251405898": "https://doi.org/10.1177/02783649251405898",
    "10.25452/figshare.plus.28386041.v1": "https://doi.org/10.25452/figshare.plus.28386041.v1",
    "10.5281/zenodo.20226053": "https://doi.org/10.5281/zenodo.20226053",
    "10.1145/3412060.3418431": "https://doi.org/10.1145/3412060.3418431",
}

VERIFIED_NO_DOI: dict[str, str] = {
    "pedregosa2011scikit": "https://jmlr.org/papers/v12/pedregosa11a.html",
    "paszke2019pytorch": "https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html",
    "fey2019fast": "https://arxiv.org/abs/1903.02428",
    "simpy412": "https://pypi.org/project/simpy/4.1.2/",
    "kipf2017semi": "https://openreview.net/forum?id=SJU4ayYgl",
    "velickovic2018graph": "https://openreview.net/forum?id=rJXMpikCZ",
    "hamilton2017inductive": "https://proceedings.neurips.cc/paper/2017/hash/5dd9db5e033da9c6fb5ba83c7a7ebea9-Abstract.html",
    "rossi2020tgn": "https://arxiv.org/abs/2006.10637",
    "adams2017persistence": "https://jmlr.org/papers/v18/16-337.html",
    "fisher2019all": "https://jmlr.org/papers/v20/18-760.html",
    "guo2017calibration": "https://proceedings.mlr.press/v70/guo17a.html",
    "tatbul2018precision": "https://proceedings.neurips.cc/paper/2018/hash/8f468c873a32bb0619eaeb2050ba45d1-Abstract.html",
    "hofer2017deep": "https://proceedings.neurips.cc/paper_files/paper/2017/hash/883e881bb4d22a7add958f2d6b052c9f-Abstract.html",
    "carriere2020perslay": "https://proceedings.mlr.press/v108/carriere20a.html",
    "cormen2009introduction": "https://mitpress.mit.edu/9780262033848/introduction-to-algorithms/",
    "aerpawDataset22": "https://aerpaw.org/dataset/",
    "aerpawDataset23": "https://aerpaw.org/dataset/",
    "winesUavToUav60GHz": "https://ece.northeastern.edu/wineslab/datasets.php",
}


def _strip_outer(value: str) -> str:
    value = value.strip().rstrip(",").strip()
    if len(value) >= 2 and ((value[0] == "{" and value[-1] == "}") or (value[0] == '"' and value[-1] == '"')):
        return value[1:-1].strip()
    return value


def parse_bibtex(text: str) -> tuple[list[BibEntry], list[str]]:
    entries: list[BibEntry] = []
    errors: list[str] = []
    pos = 0
    while True:
        match = re.search(r"@([A-Za-z]+)\s*\{\s*([^,\s]+)\s*,", text[pos:])
        if not match:
            break
        start = pos + match.start()
        body_start = pos + match.end()
        depth = 1
        quote = False
        escaped = False
        end = body_start
        while end < len(text) and depth:
            char = text[end]
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quote = not quote
            elif not quote and char == "{":
                depth += 1
            elif not quote and char == "}":
                depth -= 1
            end += 1
        if depth:
            errors.append(f"unterminated entry at offset {start}")
            break
        raw = text[body_start : end - 1]
        fields: dict[str, str] = {}
        field_pos = 0
        while field_pos < len(raw):
            field_match = re.search(r"([A-Za-z][A-Za-z0-9_-]*)\s*=\s*", raw[field_pos:])
            if not field_match:
                break
            name = field_match.group(1).lower()
            value_start = field_pos + field_match.end()
            if value_start >= len(raw):
                errors.append(f"{match.group(2)}: missing value for {name}")
                break
            opener = raw[value_start]
            if opener in "{\"":
                closer = "}" if opener == "{" else '"'
                level = 1
                idx = value_start + 1
                escaped = False
                while idx < len(raw) and level:
                    char = raw[idx]
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif opener == "{" and char == "{":
                        level += 1
                    elif char == closer:
                        level -= 1
                    idx += 1
                value = raw[value_start:idx]
                field_pos = idx
            else:
                comma = raw.find(",", value_start)
                idx = len(raw) if comma < 0 else comma
                value = raw[value_start:idx]
                field_pos = idx + 1
            fields[name] = _strip_outer(value)
        entries.append(BibEntry(match.group(1).lower(), match.group(2), fields))
        pos = end
    if not entries:
        errors.append("no BibTeX entries parsed")
    return entries, errors


def citation_keys(tex: str) -> set[str]:
    keys: set[str] = set()
    clean = re.sub(r"(?m)(?<!\\)%.*$", "", tex)
    for match in re.finditer(r"\\cite[a-zA-Z*]*\s*(?:\[[^]]*\]\s*){0,2}\{([^}]*)\}", clean):
        keys.update(item.strip() for item in match.group(1).split(",") if item.strip())
    return keys


def _normalise_doi(value: str) -> str:
    return re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value.strip(), flags=re.I).lower()


def _normalise_title(value: str) -> str:
    value = re.sub(r"\\[A-Za-z]+\s*", "", value)
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _duplicates(entries: Iterable[BibEntry], field: str, normalise) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for entry in entries:
        value = entry.fields.get(field, "")
        if value:
            found.setdefault(normalise(value), []).append(entry.key)
    return {value: keys for value, keys in found.items() if len(keys) > 1}


def audit_bibliography(bib_path: Path, tex_paths: Iterable[Path]) -> dict:
    entries, parse_errors = parse_bibtex(bib_path.read_text(encoding="utf-8"))
    keys = [entry.key for entry in entries]
    key_duplicates = sorted({key for key in keys if keys.count(key) > 1})
    cited: set[str] = set()
    for path in tex_paths:
        cited.update(citation_keys(path.read_text(encoding="utf-8")))
    entry_map = {entry.key: entry for entry in entries}
    duplicate_dois = _duplicates(entries, "doi", _normalise_doi)
    duplicate_titles = _duplicates(entries, "title", _normalise_title)
    missing = sorted(cited - set(entry_map))
    unused = sorted(set(entry_map) - cited)
    coverage = {
        group: {
            "required_keys": list(required),
            "present": all(key in entry_map for key in required),
            "cited": all(key in cited for key in required),
        }
        for group, required in REQUIRED_GROUPS.items()
    }
    doi_checks = []
    doi_url_inconsistencies = []
    for entry in entries:
        doi = entry.fields.get("doi")
        if doi:
            normalized = _normalise_doi(doi)
            url = entry.fields.get("url", "")
            if not re.match(r"^https://doi\.org/", url, flags=re.I):
                doi_url_inconsistencies.append({"key": entry.key, "doi": doi, "url": url or None})
            doi_checks.append({
                "key": entry.key,
                "entry_type": entry.entry_type,
                "doi": doi,
                "title": entry.fields.get("title"),
                "authors": entry.fields.get("author"),
                "year": entry.fields.get("year"),
                "container": entry.fields.get("journal") or entry.fields.get("booktitle") or entry.fields.get("publisher"),
                "volume": entry.fields.get("volume"),
                "number": entry.fields.get("number"),
                "pages_or_article_number": entry.fields.get("pages") or entry.fields.get("article-number"),
                "verification_status": "verified" if normalized in VERIFIED_DOIS else "unverified",
                "verification_source": VERIFIED_DOIS.get(normalized),
            })
    unverified = [item for item in doi_checks if item["verification_status"] == "unverified"]
    non_doi_verification = [
        {
            "key": entry.key,
            "title": entry.fields.get("title"),
            "verification_status": "verified" if entry.key in VERIFIED_NO_DOI else "unverified",
            "verification_source": VERIFIED_NO_DOI.get(entry.key),
            "reason_no_doi": "No publisher DOI was identified for this cited proceedings, preprint, dataset, or software record.",
        }
        for entry in entries
        if "doi" not in entry.fields and entry.key in cited
    ]
    dataset_entries = sorted(entry.key for entry in entries if entry.fields.get("entrysubtype", "").lower() == "dataset")
    software_entries = []
    for entry in entries:
        if entry.fields.get("entrysubtype", "").lower() == "software":
            software_entries.append({
                "key": entry.key,
                "version": entry.fields.get("version"),
                "archive": entry.fields.get("publisher") or entry.fields.get("howpublished"),
                "doi": entry.fields.get("doi"),
                "url": entry.fields.get("url"),
            })
    repository_version = None
    pyproject_path = bib_path.resolve().parents[1] / "pyproject.toml"
    if pyproject_path.is_file():
        version_match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', pyproject_path.read_text(encoding="utf-8"))
        repository_version = version_match.group(1) if version_match else None
    archived = next((item for item in software_entries if item["key"] == "fanetTopoGNNSoftware"), None)
    software_version_boundary = {
        "working_tree_version": repository_version,
        "cited_archive_version": archived["version"] if archived else None,
        "same_version": bool(repository_version and archived and repository_version == archived["version"]),
        "interpretation": "The archive DOI is cited only as its recorded release; it is not presented as an archive of the current working tree.",
    }
    hard_failures = []
    if parse_errors:
        hard_failures.extend(parse_errors)
    if key_duplicates:
        hard_failures.append(f"duplicate keys: {key_duplicates}")
    if duplicate_dois:
        hard_failures.append(f"duplicate DOI values: {duplicate_dois}")
    if duplicate_titles:
        hard_failures.append(f"duplicate normalized titles: {duplicate_titles}")
    if missing:
        hard_failures.append(f"citation keys missing from bibliography: {missing}")
    if doi_url_inconsistencies:
        hard_failures.append(f"DOI records without canonical resolver URL: {doi_url_inconsistencies}")
    incomplete_groups = [group for group, item in coverage.items() if not (item["present"] and item["cited"])]
    if incomplete_groups:
        hard_failures.append(f"required groups absent or uncited: {incomplete_groups}")
    return {
        "schema_version": 1,
        "status": "pass" if not hard_failures else "fail",
        "bib_path": bib_path.resolve().relative_to(ROOT).as_posix() if bib_path.resolve().is_relative_to(ROOT) else bib_path.as_posix(),
        "tex_paths": [path.resolve().relative_to(ROOT).as_posix() if path.resolve().is_relative_to(ROOT) else path.as_posix() for path in tex_paths],
        "entry_count": len(entries),
        "citation_key_count": len(cited),
        "parse_errors": parse_errors,
        "duplicate_keys": key_duplicates,
        "duplicate_dois": duplicate_dois,
        "duplicate_titles": duplicate_titles,
        "missing_citation_keys": missing,
        "doi_url_inconsistencies": doi_url_inconsistencies,
        "unused_entries": unused,
        "unused_policy": "Retained for review; exact duplicates may be merged, but unrelated records are not silently deleted.",
        "required_group_coverage": coverage,
        "dataset_entries": dataset_entries,
        "software_entries": software_entries,
        "software_version_boundary": software_version_boundary,
        "doi_verification": doi_checks,
        "non_doi_verification": non_doi_verification,
        "unverified_doi_records": unverified,
        "hard_failures": hard_failures,
    }


def render_markdown(payload: dict) -> str:
    lines = [
        "# Bibliography audit",
        "",
        f"Status: **{payload['status'].upper()}**",
        "",
        f"Parsed entries: {payload['entry_count']}; cited keys: {payload['citation_key_count']}.",
        "",
        "## Integrity checks",
        "",
        f"- Parse errors: {payload['parse_errors'] or 'none'}",
        f"- Duplicate keys: {payload['duplicate_keys'] or 'none'}",
        f"- Duplicate DOI values: {payload['duplicate_dois'] or 'none'}",
        f"- Duplicate normalized titles: {payload['duplicate_titles'] or 'none'}",
        f"- Missing citation keys: {payload['missing_citation_keys'] or 'none'}",
        f"- DOI/URL inconsistencies: {payload['doi_url_inconsistencies'] or 'none'}",
        "",
        "## Required evidence groups",
        "",
        "| Group | Keys | Present | Cited |",
        "|---|---|---:|---:|",
    ]
    for group, item in payload["required_group_coverage"].items():
        lines.append(f"| {group} | {', '.join(item['required_keys'])} | {item['present']} | {item['cited']} |")
    lines.extend(["", "## Dataset and software records", ""])
    lines.append(f"Dataset entries: {', '.join(payload['dataset_entries']) or 'none' }.")
    lines.append("")
    for item in payload["software_entries"]:
        lines.append(f"- `{item['key']}`: version={item['version']}; archive={item['archive']}; DOI={item['doi'] or 'not assigned'}; URL={item['url']}")
    boundary = payload["software_version_boundary"]
    lines.append("")
    lines.append(f"Working tree version `{boundary['working_tree_version']}` and cited archive version `{boundary['cited_archive_version']}` are not conflated. {boundary['interpretation']}")
    lines.extend(["", "## DOI verification boundary", ""])
    lines.append("`verified` means that the record metadata was checked against the linked publisher/proceedings/archive source on 2026-07-16. Other pre-existing DOI records remain `unverified`; they are not promoted to verified merely because their syntax is plausible.")
    lines.append("")
    for item in payload["doi_verification"]:
        if item["verification_status"] == "verified":
            lines.append(f"- `{item['key']}` — `{item['doi']}` — verified: {item['verification_source']}")
    lines.append("")
    lines.append("Unverified DOI records:")
    lines.append("")
    for item in payload["unverified_doi_records"]:
        lines.append(f"- `{item['key']}` — `{item['doi']}` — unverified in this audit snapshot")
    lines.extend(["", "Cited records without a DOI:", ""])
    for item in payload["non_doi_verification"]:
        lines.append(f"- `{item['key']}` — {item['verification_status']} — {item['verification_source'] or item['reason_no_doi']}")
    lines.extend(["", "## Unused records", ""])
    lines.append(payload["unused_policy"])
    lines.append("")
    for key in payload["unused_entries"]:
        lines.append(f"- `{key}` — retained pending relevance/metadata review")
    if payload["hard_failures"]:
        lines.extend(["", "## Hard failures", ""])
        lines.extend(f"- {item}" for item in payload["hard_failures"])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit manuscript BibTeX coverage and integrity.")
    parser.add_argument("--bib", type=Path, default=ROOT / "paper/cas-refs.bib")
    parser.add_argument("--tex", type=Path, action="append", default=None)
    parser.add_argument("--json-output", type=Path, default=ROOT / "docs/eaai_revision/bibliography_audit.json")
    parser.add_argument("--markdown-output", type=Path, default=ROOT / "docs/eaai_revision/BIBLIOGRAPHY_AUDIT.md")
    args = parser.parse_args()
    tex_paths = args.tex or [ROOT / "paper/main.tex"]
    payload = audit_bibliography(args.bib, tex_paths)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.markdown_output.write_text(render_markdown(payload), encoding="utf-8")
    print(f"Bibliography audit: {payload['status'].upper()} ({payload['entry_count']} entries, {payload['citation_key_count']} cited keys)")
    for failure in payload["hard_failures"]:
        print(f"[FAIL] {failure}")
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
