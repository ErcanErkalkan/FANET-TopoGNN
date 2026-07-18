from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Return the SHA-256 digest of a required regular file."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Cannot hash missing source file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_repo_path(
    path: str | os.PathLike[str],
    root: str | os.PathLike[str],
) -> str:
    """Return a repository-relative POSIX path from Windows or POSIX input."""

    root_path = Path(root).resolve()
    raw = os.fspath(path)
    if not raw:
        raise ValueError("Repository-relative path cannot be empty")
    normalized = raw.replace("\\", "/")
    candidate = Path(normalized)
    if re.match(r"^[A-Za-z]:/", normalized) and not candidate.is_absolute():
        raise ValueError(f"Windows absolute path cannot be repository-relative: {raw}")
    if candidate.is_absolute():
        try:
            relative = candidate.resolve().relative_to(root_path)
        except ValueError as exc:
            raise ValueError(f"Path is outside repository root: {candidate}") from exc
        return relative.as_posix()

    pure = PurePosixPath(normalized)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"Unsafe repository-relative path: {raw}")
    parts = tuple(part for part in pure.parts if part not in {"", "."})
    if not parts:
        raise ValueError("Repository-relative path cannot resolve to the root")
    return PurePosixPath(*parts).as_posix()


def _resolve_repo_file(relative_path: str, root: Path) -> Path:
    normalized = relative_repo_path(relative_path, root)
    candidate = root.joinpath(*PurePosixPath(normalized).parts)
    try:
        candidate.resolve(strict=False).relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Manifest path escapes repository root: {relative_path}") from exc
    return candidate


def build_file_manifest(
    paths: Iterable[str | os.PathLike[str]],
    root: str | os.PathLike[str],
) -> list[dict[str, Any]]:
    """Build deterministic SHA-256 records for required repository files."""

    root_path = Path(root).resolve()
    records: list[dict[str, Any]] = []
    for value in paths:
        relative = relative_repo_path(value, root_path)
        source = _resolve_repo_file(relative, root_path)
        if not source.is_file():
            raise FileNotFoundError(f"Cannot add missing source file to manifest: {relative}")
        records.append(
            {
                "relative_path": relative,
                "bytes": source.stat().st_size,
                "sha256": sha256_file(source),
            }
        )
    return records


def _manifest_entries(manifest: Any) -> list[Any]:
    if isinstance(manifest, Mapping):
        if "files" in manifest:
            entries = manifest["files"]
        elif "relative_path" in manifest:
            entries = [manifest]
        else:
            return []
    else:
        entries = manifest
    if isinstance(entries, (str, bytes)) or not isinstance(entries, Iterable):
        return []
    return list(entries)


def verify_manifest(
    manifest: Any,
    root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Verify every manifest file without updating expected provenance values."""

    root_path = Path(root).resolve()
    entries = _manifest_entries(manifest)
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    if not entries:
        errors.append("manifest contains no verifiable file entries")

    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            message = f"entry {index}: file record must be a mapping"
            errors.append(message)
            results.append(
                {
                    "index": index,
                    "relative_path": None,
                    "expected_sha256": None,
                    "actual_sha256": None,
                    "status": "fail",
                    "error": message,
                }
            )
            continue
        raw_path = entry.get("relative_path")
        expected = str(entry.get("sha256", "")).lower()
        result: dict[str, Any] = {
            "index": index,
            "relative_path": raw_path,
            "expected_sha256": expected or None,
            "actual_sha256": None,
            "status": "fail",
        }
        if not isinstance(raw_path, str) or not raw_path:
            message = f"entry {index}: missing relative_path"
            result["error"] = message
            errors.append(message)
            results.append(result)
            continue
        try:
            normalized = relative_repo_path(raw_path, root_path)
            result["relative_path"] = normalized
            source = _resolve_repo_file(normalized, root_path)
        except ValueError as exc:
            message = f"{raw_path}: {exc}"
            result["error"] = message
            errors.append(message)
            results.append(result)
            continue
        if not source.is_file():
            message = f"{normalized}: source file is missing"
            result["error"] = message
            errors.append(message)
            results.append(result)
            continue
        if not _SHA256_RE.fullmatch(expected):
            message = f"{normalized}: expected SHA-256 is missing or malformed"
            result["error"] = message
            errors.append(message)
            results.append(result)
            continue

        actual = sha256_file(source)
        result["actual_sha256"] = actual
        if actual != expected:
            message = f"{normalized}: SHA-256 mismatch; expected {expected}, got {actual}"
            result["error"] = message
            errors.append(message)
            results.append(result)
            continue
        expected_bytes = entry.get("bytes")
        if expected_bytes is not None and int(expected_bytes) != source.stat().st_size:
            message = (
                f"{normalized}: byte-size mismatch; expected {expected_bytes}, "
                f"got {source.stat().st_size}"
            )
            result["error"] = message
            errors.append(message)
            results.append(result)
            continue
        result["status"] = "pass"
        results.append(result)

    return {
        "valid": bool(entries) and not errors and len(results) == len(entries),
        "files": results,
        "errors": errors,
    }
