#!/usr/bin/env python3
"""Build and validate the clean EAAI submission deliverables."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fanet.provenance import relative_repo_path, sha256_file
from scripts.scan_anonymity import extract_anonymous_tex, load_pattern_config, scan_paths, scan_text


OUTPUT = ROOT / "submission"
ARCHIVE_EXTENSIONS = {".zip", ".tar", ".tgz", ".gz", ".bz2", ".xz", ".7z", ".rar"}
NATIVE_EXECUTABLE_EXTENSIONS = {".exe", ".dll", ".so", ".dylib", ".scr", ".com", ".msi"}
TEXT_EXTENSIONS = {".tex", ".bib", ".bst", ".cls", ".md", ".txt", ".json", ".csv", ".html", ".py", ".ps1"}
ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?:\b[A-Za-z]:[\\/](?![\\/])|(?<!:)\/(?:Users|home|mnt)\/|\\\\[A-Za-z0-9._-]+\\[A-Za-z0-9$._-]+(?:\\|$))",
    flags=re.IGNORECASE,
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_archive_members(archive: Path) -> dict:
    names: list[str] = []
    duplicate_names: list[str] = []
    unsafe_names: list[str] = []
    symlinks: list[str] = []
    nested_archives: list[str] = []
    native_executables: list[str] = []
    seen: set[str] = set()
    with zipfile.ZipFile(archive) as handle:
        bad_member = handle.testzip()
        for info in handle.infolist():
            name = info.filename
            names.append(name)
            pure = PurePosixPath(name)
            if name in seen:
                duplicate_names.append(name)
            seen.add(name)
            if pure.is_absolute() or ".." in pure.parts or "\\" in name or not name:
                unsafe_names.append(name)
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                symlinks.append(name)
            suffix = pure.suffix.lower()
            if suffix in ARCHIVE_EXTENSIONS:
                nested_archives.append(name)
            if suffix in NATIVE_EXECUTABLE_EXTENSIONS:
                native_executables.append(name)
    return {
        "valid": not any([bad_member, duplicate_names, unsafe_names, symlinks, nested_archives, native_executables]),
        "member_count": len(names),
        "crc_error": bad_member,
        "duplicate_names": duplicate_names,
        "unsafe_names": unsafe_names,
        "symlinks": symlinks,
        "nested_archives": nested_archives,
        "native_executables": native_executables,
    }


def _bibtex_blocks(text: str) -> dict[str, str]:
    blocks: dict[str, str] = {}
    cursor = 0
    header = re.compile(r"@([A-Za-z]+)\s*\{\s*([^,\s]+)\s*,")
    while match := header.search(text, cursor):
        start = match.start()
        pos = match.end()
        depth = 1
        while pos < len(text) and depth:
            char = text[pos]
            if char == "{" and (pos == 0 or text[pos - 1] != "\\"):
                depth += 1
            elif char == "}" and (pos == 0 or text[pos - 1] != "\\"):
                depth -= 1
            pos += 1
        if depth:
            raise ValueError(f"unterminated BibTeX entry {match.group(2)}")
        blocks[match.group(2)] = text[start:pos].strip() + "\n"
        cursor = pos
    return blocks


def cited_keys(tex: str) -> list[str]:
    keys: set[str] = set()
    for group in re.findall(r"\\cite\w*\s*(?:\[[^\]]*\]\s*)*\{([^}]+)\}", tex):
        keys.update(item.strip() for item in group.split(",") if item.strip())
    return sorted(keys)


def sanitized_bibliography(bib_text: str, keys: list[str]) -> str:
    blocks = _bibtex_blocks(bib_text)
    missing = sorted(set(keys) - set(blocks))
    if missing:
        raise ValueError(f"citation keys missing from bibliography: {missing}")
    return "\n\n".join(blocks[key].strip() for key in keys) + "\n"


def materialize_anonymous_tex(text: str) -> str:
    anonymous = extract_anonymous_tex(text)
    anonymous = re.sub(
        r"\\ifdefined\\journalfinal\s+\\documentclass\[final,5p,times,twocolumn\]\{elsarticle\}\s+\\else\s+\\documentclass\[preprint\]\{elsarticle\}\s+\\fi",
        r"\\documentclass[preprint]{elsarticle}",
        anonymous,
        count=1,
        flags=re.DOTALL,
    )
    anonymous = re.sub(
        r"\\newif\\ifanonymous\s+\\ifdefined\\anonymousversion\s+\\anonymoustrue\s+\\else\s+\\anonymousfalse\s+\\fi",
        "",
        anonymous,
        count=1,
        flags=re.DOTALL,
    )
    if "twocolumn" in anonymous:
        raise ValueError("materialized anonymous source still contains twocolumn mode")
    return re.sub(r"\n{3,}", "\n\n", anonymous).strip() + "\n"


def latex_dependencies(tex: str, paper_root: Path) -> dict[str, list[Path]]:
    tables: list[Path] = []
    figures: list[Path] = []
    for value in re.findall(r"\\input\{([^}]+)\}", tex):
        path = Path(value)
        if path.suffix == "":
            path = path.with_suffix(".tex")
        tables.append(path)
    for value in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex):
        path = Path(value)
        if path.suffix:
            figures.append(path)
            continue
        candidates = [path.with_suffix(ext) for ext in (".pdf", ".png", ".jpg", ".jpeg")]
        found = next((item for item in candidates if (paper_root / item).is_file()), None)
        if found is None:
            raise FileNotFoundError(f"unresolved figure: {value}")
        figures.append(found)
    return {"tables": tables, "figures": figures}


def copy_source_tree(tex: str, bib: str, destination: Path, *, original_wrapper: bool) -> dict:
    paper = ROOT / "paper"
    destination.mkdir(parents=True, exist_ok=True)
    if original_wrapper:
        shutil.copy2(paper / "main.tex", destination / "main.tex")
        shutil.copy2(paper / "main_anonymized.tex", destination / "main_anonymized.tex")
        shutil.copy2(paper / "cas-refs.bib", destination / "cas-refs.bib")
        dependency_tex = (paper / "main.tex").read_text(encoding="utf-8")
    else:
        (destination / "main_anonymized.tex").write_text(tex, encoding="utf-8")
        (destination / "cas-refs.bib").write_text(bib, encoding="utf-8")
        dependency_tex = tex

    dependencies = latex_dependencies(dependency_tex, paper)
    for group in ("tables", "figures"):
        for relative in dependencies[group]:
            source = paper / relative
            if not source.is_file():
                raise FileNotFoundError(source)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    for name in ("elsarticle.cls", "elsarticle-num.bst"):
        located = subprocess.run(
            ["kpsewhich", name], check=True, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=30
        ).stdout.strip()
        if not located:
            raise FileNotFoundError(f"kpsewhich did not locate {name}")
        shutil.copy2(Path(located), destination / name)

    instructions = (
        "Build the anonymous preprint from this directory:\n"
        "  pdflatex -interaction=nonstopmode -halt-on-error main_anonymized.tex\n"
        "  bibtex main_anonymized\n"
        "  pdflatex -interaction=nonstopmode -halt-on-error main_anonymized.tex\n"
        "  pdflatex -interaction=nonstopmode -halt-on-error main_anonymized.tex\n"
    )
    if not original_wrapper:
        (destination / "BUILD_INSTRUCTIONS.txt").write_text(instructions, encoding="utf-8")
    return dependencies


def sanitize_build_log(text: str) -> str:
    home = str(Path.home())
    variants = {
        home,
        home.replace("\\", "/"),
        home.lower(),
        home.replace("\\", "/").lower(),
    }
    sanitized = text
    for value in sorted(variants, key=len, reverse=True):
        sanitized = re.sub(re.escape(value), "<USER_HOME>", sanitized, flags=re.IGNORECASE)
    for pattern in load_pattern_config()["patterns"]:
        sanitized = re.sub(pattern["regex"], "<IDENTITY_REDACTED>", sanitized, flags=re.IGNORECASE)
    return sanitized


def run_build(directory: Path, log_path: Path) -> Path:
    commands = [
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-recorder", "main_anonymized.tex"],
        ["bibtex", "main_anonymized"],
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-recorder", "main_anonymized.tex"],
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-recorder", "main_anonymized.tex"],
    ]
    transcript: list[str] = []
    final_output = ""
    for command in commands:
        completed = subprocess.run(
            command, cwd=directory, check=False, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=600
        )
        final_output = (completed.stdout or "") + (completed.stderr or "")
        transcript.extend([f"$ {' '.join(command)}", completed.stdout or "", completed.stderr or ""])
        if completed.returncode:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(sanitize_build_log("\n".join(transcript)), encoding="utf-8")
            raise RuntimeError(f"clean LaTeX build failed: {' '.join(command)}")
    final_failures = re.findall(
        r"(?:Citation .* undefined|There were undefined citations|Reference .* undefined|"
        r"There were undefined references|Overfull \\[hv]box|! LaTeX Error|Emergency stop)",
        final_output,
        flags=re.IGNORECASE,
    )
    if final_failures:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(sanitize_build_log("\n".join(transcript)), encoding="utf-8")
        raise RuntimeError(f"final LaTeX pass is not clean: {final_failures}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(sanitize_build_log("\n".join(transcript)), encoding="utf-8")
    pdf = directory / "main_anonymized.pdf"
    if not pdf.is_file():
        raise FileNotFoundError(pdf)
    return pdf


def zip_tree(source: Path, archive: Path) -> None:
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(source).as_posix())


def extract_safe(archive: Path, destination: Path) -> dict:
    structure = safe_archive_members(archive)
    if not structure["valid"]:
        raise RuntimeError(f"unsafe archive structure: {structure}")
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(destination)
    return structure


def absolute_path_findings(root: Path) -> list[dict]:
    findings: list[dict] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), start=1):
            if ABSOLUTE_PATH_PATTERN.search(line):
                findings.append({"file": path.relative_to(root).as_posix(), "line": number, "text": line.strip()})
    return findings


def dependency_findings(root: Path) -> list[str]:
    tex_path = root / "main_anonymized.tex"
    tex = tex_path.read_text(encoding="utf-8")
    missing: list[str] = []
    dependencies = latex_dependencies(tex, root)
    for paths in dependencies.values():
        missing.extend(path.as_posix() for path in paths if not (root / path).is_file())
    for name in ("cas-refs.bib", "elsarticle.cls", "elsarticle-num.bst"):
        if not (root / name).is_file():
            missing.append(name)
    return sorted(set(missing))


def _run(command: list[str], *, cwd: Path | None = None, timeout: int = 120) -> tuple[int, str]:
    completed = subprocess.run(
        command, cwd=cwd, check=False, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout
    )
    return completed.returncode, ((completed.stdout or "") + (completed.stderr or "")).strip()


def pdf_validation(path: Path) -> dict:
    qpdf = shutil.which("qpdf")
    qpdf_result = {"status": "not_available", "detail": "qpdf not installed"}
    if qpdf:
        code, output = _run([qpdf, "--check", str(path)])
        qpdf_result = {"status": "pass" if code == 0 else "fail", "detail": output}
    info_code, info = _run(["pdfinfo", str(path)])
    fonts_code, fonts = _run(["pdffonts", str(path)])
    text_code, _ = _run(["pdftotext", str(path), "-"])
    font_rows = [line for line in fonts.splitlines()[2:] if line.strip()]
    embedded = bool(font_rows) and all(re.search(r"\s+yes\s+(?:yes|no)\s+(?:yes|no)\s+\d+\s+\d+\s*$", line) for line in font_rows)
    pages = re.search(r"^Pages:\s+(\d+)", info, flags=re.MULTILINE)
    size = re.search(r"^Page size:\s+(.+)$", info, flags=re.MULTILINE)
    return {
        "valid": info_code == 0 and fonts_code == 0 and text_code == 0 and embedded and qpdf_result["status"] != "fail",
        "qpdf": qpdf_result,
        "pages": int(pages.group(1)) if pages else None,
        "page_size": size.group(1).strip() if size else None,
        "bytes": path.stat().st_size,
        "font_count": len(font_rows),
        "all_fonts_embedded": embedded,
        "pdfinfo": info,
        "pdffonts": fonts,
    }


def normalized_pdf_text(path: Path) -> str:
    code, text = _run(["pdftotext", "-layout", str(path), "-"])
    if code:
        raise RuntimeError(f"pdftotext failed for {path}: {text}")
    return re.sub(r"\s+", " ", text).strip()


def git_commit() -> str | None:
    code, output = _run(["git", "rev-parse", "HEAD"], cwd=ROOT)
    return output if code == 0 else None


def file_record(path: Path, role: str, anonymous: bool, generated_from: list[str], base: Path) -> dict:
    return {
        "filename": path.relative_to(base).as_posix(),
        "role": role,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
        "anonymous": anonymous,
        "source_generated_from": generated_from,
    }


def archive_member_records(archive: Path, archive_name: str, *, anonymous: bool) -> list[dict]:
    records: list[dict] = []
    with zipfile.ZipFile(archive) as handle:
        for info in sorted(handle.infolist(), key=lambda item: item.filename):
            if info.is_dir():
                continue
            data = handle.read(info.filename)
            suffix = PurePosixPath(info.filename).suffix.lower()
            if archive_name.endswith("manuscript_sources.zip"):
                if info.filename == "main_anonymized.tex":
                    role = "anonymous_manuscript_tex"
                elif suffix == ".bib":
                    role = "sanitized_bibliography"
                elif suffix == ".cls":
                    role = "latex_class"
                elif suffix == ".bst":
                    role = "bibliography_style"
                elif info.filename.startswith("figures/"):
                    role = "manuscript_figure"
                elif info.filename.startswith("tables/"):
                    role = "generated_table"
                else:
                    role = "build_instruction"
            else:
                head = PurePosixPath(info.filename).parts[0] if PurePosixPath(info.filename).parts else ""
                role = {
                    "configs": "reproduction_config",
                    "scripts": "reproduction_script",
                    "outputs": "derived_scientific_artifact",
                    "data": "derived_data_or_source_manifest",
                    "fanet": "reproduction_library",
                    "tests": "reproduction_test",
                }.get(head, "supplementary_support_file")
            records.append(
                {
                    "filename": f"{archive_name}::{info.filename}",
                    "role": role,
                    "size": info.file_size,
                    "sha256": sha256_bytes(data),
                    "anonymous": anonymous,
                    "source_generated_from": [archive_name],
                }
            )
    return records


def build(output: Path, *, overwrite: bool = False) -> dict:
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty submission directory: {output}")
    resolved_root = ROOT.resolve()
    resolved_output = output.resolve()
    if not resolved_output.is_relative_to(resolved_root) or resolved_output == resolved_root:
        raise RuntimeError(f"submission output must remain inside repository: {resolved_output}")

    manuscript_text = (ROOT / "paper/main.tex").read_text(encoding="utf-8")
    anonymous_tex = materialize_anonymous_tex(manuscript_text)
    keys = cited_keys(anonymous_tex)
    sanitized_bib = sanitized_bibliography(
        (ROOT / "paper/cas-refs.bib").read_text(encoding="utf-8"), keys
    )

    with tempfile.TemporaryDirectory(prefix="fanet_eaai_submission_") as temporary:
        temp = Path(temporary)
        staged_output = temp / "submission"
        logs = staged_output / "build_logs"
        staged_output.mkdir(parents=True)
        logs.mkdir()

        wrapper_tree = temp / "wrapper_clean_build"
        copy_source_tree(anonymous_tex, sanitized_bib, wrapper_tree, original_wrapper=True)
        run_build(wrapper_tree, logs / "local_main_anonymized_clean_build.log")

        source_tree = temp / "anonymous_sources"
        dependencies = copy_source_tree(anonymous_tex, sanitized_bib, source_tree, original_wrapper=False)
        source_zip = staged_output / "EAAI_anonymous_manuscript_sources.zip"
        zip_tree(source_tree, source_zip)

        source_extract = temp / "source_extract_validation"
        source_structure = extract_safe(source_zip, source_extract)
        source_absolute_paths = absolute_path_findings(source_extract)
        source_missing = dependency_findings(source_extract)
        source_identity = scan_paths(
            sorted(path for path in source_extract.rglob("*") if path.is_file()),
            root=source_extract,
        )
        if source_absolute_paths or source_missing or source_identity["findings"] or source_identity["errors"]:
            raise RuntimeError(
                "anonymous source validation failed: "
                + json.dumps(
                    {
                        "absolute_paths": source_absolute_paths,
                        "missing": source_missing,
                        "identity": source_identity,
                    },
                    ensure_ascii=False,
                )
            )
        rebuilt_pdf = run_build(source_extract, logs / "source_archive_clean_build.log")

        shutil.copy2(ROOT / "paper/main_anonymized.pdf", staged_output / "EAAI_anonymous_manuscript.pdf")
        shutil.copy2(ROOT / "paper/title_page.pdf", staged_output / "EAAI_title_page.pdf")
        shutil.copy2(ROOT / "paper/title_page.tex", staged_output / "EAAI_title_page.tex")
        shutil.copy2(ROOT / "EAAI_HIGHLIGHTS.txt", staged_output / "EAAI_highlights.txt")
        shutil.copy2(
            ROOT / "paper/figures/plantuml/graphical_abstract.pdf",
            staged_output / "EAAI_graphical_abstract.pdf",
        )
        canonical_supplement = ROOT / "anonymous_supplementary.zip"
        shutil.copy2(canonical_supplement, staged_output / "EAAI_anonymous_supplementary.zip")

        supplement_zip = staged_output / "EAAI_anonymous_supplementary.zip"
        supplement_structure = safe_archive_members(supplement_zip)
        supplement_extract = temp / "supplement_extract_validation"
        extract_safe(supplement_zip, supplement_extract)
        with zipfile.ZipFile(supplement_zip) as supplement_handle:
            supplement_paths = [PurePosixPath(name) for name in supplement_handle.namelist()]
        forbidden_names = {".git", "CITATION.cff", ".zenodo.json", "title_page.tex", "title_page.pdf"}
        forbidden_members = sorted(
            path.as_posix()
            for path in supplement_paths
            if any(part in forbidden_names for part in path.parts)
        )
        supplement_identity = scan_paths(
            sorted(path for path in supplement_extract.rglob("*") if path.is_file()),
            root=supplement_extract,
        )
        supplement_absolute_paths = absolute_path_findings(supplement_extract)
        raw_bags = [path.relative_to(supplement_extract).as_posix() for path in supplement_extract.rglob("*.bag")]
        if (
            not supplement_structure["valid"]
            or forbidden_members
            or supplement_identity["findings"]
            or supplement_identity["errors"]
            or supplement_absolute_paths
            or raw_bags
        ):
            raise RuntimeError(
                "anonymous supplementary validation failed: "
                + json.dumps(
                    {
                        "structure": supplement_structure,
                        "forbidden_members": forbidden_members,
                        "identity": supplement_identity,
                        "absolute_paths": supplement_absolute_paths,
                        "raw_bags": raw_bags,
                    },
                    ensure_ascii=False,
                )
            )

        submitted_pdf = staged_output / "EAAI_anonymous_manuscript.pdf"
        submitted_pdf_validation = pdf_validation(submitted_pdf)
        rebuilt_pdf_validation = pdf_validation(rebuilt_pdf)
        title_pdf_validation = pdf_validation(staged_output / "EAAI_title_page.pdf")
        graphical_pdf_validation = pdf_validation(staged_output / "EAAI_graphical_abstract.pdf")
        text_match = normalized_pdf_text(rebuilt_pdf) == normalized_pdf_text(submitted_pdf)
        figure_hashes_match = all(
            sha256_file(source_extract / path) == sha256_file(ROOT / "paper" / path)
            for path in dependencies["figures"]
        )
        preprint_single_column = (
            r"\documentclass[preprint]{elsarticle}" in anonymous_tex and "twocolumn" not in anonymous_tex
        )
        if not all(
            [
                submitted_pdf_validation["valid"],
                rebuilt_pdf_validation["valid"],
                title_pdf_validation["valid"],
                graphical_pdf_validation["valid"],
                text_match,
                figure_hashes_match,
                preprint_single_column,
            ]
        ):
            raise RuntimeError("PDF or clean-build equivalence validation failed")

        verification = {
            "schema_version": 1,
            "status": "pass",
            "local_main_anonymized_clean_build": "pass",
            "source_archive_clean_build": "pass",
            "source_archive_structure": source_structure,
            "source_identity_scan": source_identity,
            "source_absolute_path_findings": source_absolute_paths,
            "source_missing_dependencies": source_missing,
            "supplement_archive_structure": supplement_structure,
            "supplement_identity_scan": supplement_identity,
            "supplement_absolute_path_findings": supplement_absolute_paths,
            "supplement_forbidden_members": forbidden_members,
            "supplement_raw_bag_files": raw_bags,
            "canonical_supplement_sha256": sha256_file(canonical_supplement),
            "submission_supplement_sha256": sha256_file(supplement_zip),
            "pdf_text_content_match": text_match,
            "figure_source_hashes_match": figure_hashes_match,
            "preprint_single_column_source": preprint_single_column,
            "submitted_anonymous_pdf": submitted_pdf_validation,
            "rebuilt_anonymous_pdf": rebuilt_pdf_validation,
            "title_page_pdf": title_pdf_validation,
            "graphical_abstract_pdf": graphical_pdf_validation,
            "qpdf_availability": "available" if shutil.which("qpdf") else "not_available",
        }
        (logs / "package_verification.json").write_text(
            json.dumps(verification, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (logs / "source_identity_scan.json").write_text(
            json.dumps(source_identity, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (logs / "supplement_identity_scan.json").write_text(
            json.dumps(supplement_identity, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        pattern_config = load_pattern_config()
        log_identity_findings = [
            finding
            for log in sorted(logs.iterdir())
            for finding in scan_text(
                log.read_text(encoding="utf-8", errors="replace"),
                f"build_logs/{log.name}",
                pattern_config,
            )
        ]
        if log_identity_findings:
            raise RuntimeError(f"build-log identity scan failed: {log_identity_findings}")
        verification["build_log_identity_findings"] = log_identity_findings
        (logs / "package_verification.json").write_text(
            json.dumps(verification, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        artifacts = [
            file_record(source_zip, "anonymous_manuscript_source_archive", True, ["paper/main.tex", "paper/cas-refs.bib", "used paper figures and tables"], staged_output),
            file_record(submitted_pdf, "anonymous_manuscript_pdf", True, ["paper/main_anonymized.tex"], staged_output),
            file_record(staged_output / "EAAI_title_page.pdf", "identified_title_page_pdf", False, ["paper/title_page.tex"], staged_output),
            file_record(staged_output / "EAAI_title_page.tex", "identified_title_page_editable_source", False, ["paper/title_page.tex"], staged_output),
            file_record(supplement_zip, "anonymous_reproducibility_supplement", True, ["anonymous_supplementary.zip", "scripts/build_anonymous_supplementary.py"], staged_output),
            file_record(staged_output / "EAAI_highlights.txt", "editable_highlights", True, ["EAAI_HIGHLIGHTS.txt"], staged_output),
            file_record(staged_output / "EAAI_graphical_abstract.pdf", "graphical_abstract", True, ["paper/scripts/create_graphical_abstract.py"], staged_output),
        ]
        for log in sorted(logs.iterdir()):
            artifacts.append(file_record(log, "build_or_validation_log", True, ["scripts/build_submission_package.py"], staged_output))

        archive_members = archive_member_records(source_zip, source_zip.name, anonymous=True)
        archive_members.extend(archive_member_records(supplement_zip, supplement_zip.name, anonymous=True))
        manifest = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(),
            "canonical_anonymous_supplementary": "anonymous_supplementary.zip",
            "self_hash_excluded": True,
            "artifacts": artifacts,
            "archive_members": archive_members,
            "validation": {
                "status": "pass",
                "source_archive_member_count": source_structure["member_count"],
                "supplement_archive_member_count": supplement_structure["member_count"],
                "source_clean_build": True,
                "local_wrapper_clean_build": True,
                "identity_scans": True,
                "absolute_path_scans": True,
                "missing_dependency_scan": True,
                "archive_structure_scan": True,
                "pdf_text_content_match": text_match,
                "figure_source_hashes_match": figure_hashes_match,
                "preprint_single_column": preprint_single_column,
                "qpdf": "checked" if shutil.which("qpdf") else "not_available",
                "font_embedding": True,
            },
        }
        (staged_output / "SUBMISSION_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        if output.exists():
            shutil.rmtree(output)
        shutil.copytree(staged_output, output)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    manifest = build(output, overwrite=args.overwrite)
    print(
        f"Built {output.relative_to(ROOT).as_posix()}: "
        f"{len(manifest['artifacts'])} artifacts, {len(manifest['archive_members'])} archive members"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
