#!/usr/bin/env python3
"""Reconstitute the one integrity-checked active research ZIP from transport chunks."""
from pathlib import Path
from hashlib import sha256
from zipfile import ZipFile
ROOT = Path(__file__).resolve().parent
EXPECTED = "4b224751ebda13025dcac322af4ab009c6da67eb4e1e5196c739bad1ec3ff3ea"
PARTS = sorted((ROOT / "snapshot").glob("current.zip.part[0-9][0-9]"))
if len(PARTS) != 8:
    raise SystemExit(f"Expected 8 parts; got {len(PARTS)}")
out = ROOT / "FANET-TopoGNN_Current.zip"
with out.open("wb") as dest:
    for part in PARTS:
        dest.write(part.read_bytes())
actual = sha256(out.read_bytes()).hexdigest()
if actual != EXPECTED:
    out.unlink(missing_ok=True)
    raise SystemExit(f"Integrity check failed: {actual}")
with ZipFile(out) as archive:
    damaged = archive.testzip()
    if damaged:
        raise SystemExit(f"Damaged ZIP member: {damaged}")
print("Current ZIP verified:", out, actual)
