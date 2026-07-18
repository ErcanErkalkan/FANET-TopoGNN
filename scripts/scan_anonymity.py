from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name("anonymity_patterns.json")
TEXT_SUFFIXES = {
    ".bib",
    ".cff",
    ".cfg",
    ".csv",
    ".html",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


class AnonymityScanError(RuntimeError):
    """Raised when a target cannot be scanned reliably."""


def load_pattern_config(path: Path = DEFAULT_CONFIG) -> dict:
    config_path = Path(path)
    if not config_path.is_file():
        raise AnonymityScanError(f"pattern configuration is missing: {config_path}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnonymityScanError(f"cannot read pattern configuration {config_path}: {exc}") from exc

    patterns = config.get("patterns")
    allowlist = config.get("allowlist", [])
    if not isinstance(patterns, list) or not patterns:
        raise AnonymityScanError("pattern configuration must contain a non-empty patterns list")
    if not isinstance(allowlist, list):
        raise AnonymityScanError("allowlist must be a list")

    seen_ids: set[str] = set()
    for entry in patterns:
        if not isinstance(entry, dict) or not all(
            isinstance(entry.get(key), str) and entry[key].strip()
            for key in ("id", "description", "regex")
        ):
            raise AnonymityScanError("each pattern requires non-empty id, description, and regex fields")
        if entry["id"] in seen_ids:
            raise AnonymityScanError(f"duplicate pattern id: {entry['id']}")
        seen_ids.add(entry["id"])
        try:
            re.compile(entry["regex"], flags=re.IGNORECASE)
        except re.error as exc:
            raise AnonymityScanError(f"invalid regex for {entry['id']}: {exc}") from exc

    for entry in allowlist:
        if not isinstance(entry, dict):
            raise AnonymityScanError("each allowlist entry must be an object")
        pattern_id = entry.get("pattern_id")
        relative_path = entry.get("path")
        line = entry.get("line")
        reason = entry.get("reason")
        if pattern_id not in seen_ids:
            raise AnonymityScanError(f"allowlist references unknown pattern id: {pattern_id}")
        if not isinstance(relative_path, str) or not relative_path.strip() or "\\" in relative_path:
            raise AnonymityScanError("allowlist path must be a non-empty repository-relative POSIX path")
        if Path(relative_path).is_absolute() or ".." in Path(relative_path).parts:
            raise AnonymityScanError(f"allowlist path must remain relative: {relative_path}")
        if not isinstance(line, int) or isinstance(line, bool) or line < 1:
            raise AnonymityScanError(f"allowlist line must be a positive integer: {relative_path}")
        if not isinstance(reason, str) or not reason.strip():
            raise AnonymityScanError(
                f"allowlist entry requires a non-empty explanation: {relative_path}:{line}"
            )
    return config


def extract_pdf_text(path: Path) -> str:
    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise AnonymityScanError(f"PDF is missing: {pdf_path}")
    pdftotext = shutil.which("pdftotext")
    if pdftotext:
        result = subprocess.run(
            [pdftotext, str(pdf_path), "-"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode == 0:
            extracted_text = result.stdout
            pdfinfo = shutil.which("pdfinfo")
            metadata_text = ""
            if pdfinfo:
                metadata_result = subprocess.run(
                    [pdfinfo, str(pdf_path)],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                if metadata_result.returncode == 0:
                    metadata_text = metadata_result.stdout
            if not metadata_text:
                try:
                    from pypdf import PdfReader

                    metadata = PdfReader(str(pdf_path)).metadata or {}
                    metadata_text = "\n".join(
                        f"{key}: {value}" for key, value in metadata.items() if value is not None
                    )
                except Exception:
                    metadata_text = ""
            if extracted_text.strip() or metadata_text.strip():
                return extracted_text + ("\nPDF metadata\n" + metadata_text if metadata_text else "")
        pdftotext_error = result.stderr.strip() or f"exit code {result.returncode}"
    else:
        pdftotext_error = "not installed"

    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        page_text = "\n\f\n".join(page.extract_text() or "" for page in reader.pages)
        metadata = reader.metadata or {}
        metadata_text = "\n".join(
            f"{key}: {value}" for key, value in metadata.items() if value is not None
        )
        return page_text + ("\nPDF metadata\n" + metadata_text if metadata_text else "")
    except Exception as exc:  # pypdf exposes backend-specific parse exceptions
        raise AnonymityScanError(
            f"cannot extract text from {pdf_path}; pdftotext={pdftotext_error}; pypdf={exc}"
        ) from exc


def extract_anonymous_tex(text: str) -> str:
    r"""Keep each explicit ``\ifanonymous`` true branch and leave other TeX unchanged."""
    conditional_re = re.compile(r"\\if(?:defined|[A-Za-z@]+)|\\else\b|\\fi\b")
    marker_re = re.compile(r"(?m)^[ \t]*\\ifanonymous\b")
    output: list[str] = []
    cursor = 0
    while True:
        marker_match = marker_re.search(text, cursor)
        if marker_match is None:
            output.append(text[cursor:])
            break
        start = marker_match.start()
        marker_end = marker_match.end()
        output.append(text[cursor:start])
        depth = 1
        else_start: int | None = None
        end_match: re.Match[str] | None = None
        for match in conditional_re.finditer(text, marker_end):
            token = match.group(0)
            if token.startswith("\\if"):
                depth += 1
            elif token == "\\else" and depth == 1:
                if else_start is not None:
                    raise AnonymityScanError("malformed anonymous TeX conditional: repeated \\else")
                else_start = match.start()
            elif token == "\\fi":
                depth -= 1
                if depth == 0:
                    end_match = match
                    break
        if end_match is None:
            raise AnonymityScanError("malformed anonymous TeX conditional: unterminated \\ifanonymous")
        true_end = else_start if else_start is not None else end_match.start()
        output.append(text[marker_end:true_end])
        cursor = end_match.end()
    filtered = "".join(output)
    return extract_anonymous_tex(filtered) if marker_re.search(filtered) else filtered


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def scan_text(text: str, display_path: str, config: dict) -> list[dict]:
    patterns = [
        (entry, re.compile(entry["regex"], flags=re.IGNORECASE))
        for entry in config["patterns"]
    ]
    allowed = {
        (entry["pattern_id"], entry["path"], entry["line"])
        for entry in config.get("allowlist", [])
    }
    findings: list[dict] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for entry, pattern in patterns:
            for match in pattern.finditer(line):
                if (entry["id"], display_path, line_number) in allowed:
                    continue
                findings.append(
                    {
                        "path": display_path,
                        "line": line_number,
                        "pattern_id": entry["id"],
                        "description": entry["description"],
                        "match": match.group(0),
                    }
                )
    return findings


def scan_paths(
    paths: Iterable[Path],
    *,
    root: Path = ROOT,
    config_path: Path = DEFAULT_CONFIG,
) -> dict:
    scan_root = Path(root)
    config_file = Path(config_path).resolve()
    config = load_pattern_config(config_file)
    candidates: set[Path] = set()
    errors: list[str] = []
    for raw_path in paths:
        target = Path(raw_path)
        if not target.exists():
            errors.append(f"target is missing: {target}")
        elif target.is_dir():
            candidates.update(path for path in target.rglob("*") if path.is_file())
        else:
            candidates.add(target)

    findings: list[dict] = []
    scanned_files: list[str] = []
    for path in sorted(candidates, key=lambda item: item.as_posix().lower()):
        if path.resolve() == config_file:
            continue
        suffix = path.suffix.lower()
        if suffix not in TEXT_SUFFIXES and suffix != ".pdf":
            continue
        display_path = _display_path(path, scan_root)
        try:
            text = extract_pdf_text(path) if suffix == ".pdf" else path.read_text(
                encoding="utf-8", errors="replace"
            )
        except (OSError, AnonymityScanError) as exc:
            errors.append(f"{display_path}: {exc}")
            continue
        scanned_files.append(display_path)
        findings.extend(scan_text(text, display_path, config))

    return {
        "schema_version": 1,
        "status": "pass" if not findings and not errors else "fail",
        "scanned_files": scanned_files,
        "findings": findings,
        "errors": errors,
    }


def format_failures(report: dict) -> str:
    lines = list(report.get("errors", []))
    lines.extend(
        f"{item['path']}:{item['line']}: {item['pattern_id']} matched {item['match']!r}"
        for item in report.get("findings", [])
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail on author identity in anonymous artifacts.")
    parser.add_argument("paths", nargs="+", type=Path, help="Files or directories to scan")
    parser.add_argument("--patterns", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    try:
        report = scan_paths(args.paths, root=args.root, config_path=args.patterns)
    except AnonymityScanError as exc:
        report = {
            "schema_version": 1,
            "status": "fail",
            "scanned_files": [],
            "findings": [],
            "errors": [str(exc)],
        }
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
