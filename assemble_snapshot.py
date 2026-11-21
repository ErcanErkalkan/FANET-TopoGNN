#!/usr/bin/env python3
"""Reassemble the only current FANET-TopoGNN snapshot from the checked-in binary chunks."""
from pathlib import Path
from hashlib import sha256
from zipfile import ZipFile
ROOT = Path(__file__).resolve().parent
EXPECTED = "8b06e9235a46ae9fc85887f901f0619e37b86f6216ca879afc8f743a24d7267d"
PARTS = sorted((ROOT / "snapshot").glob("current.zip.part[0-9][0-9]"))
if len(PARTS) != 7:
    raise SystemExit(f"Expected seven snapshot chunks; found {len(PARTS)}")
dst = ROOT / "FANET-TopoGNN_Current.zip"
with dst.open("wb") as out:
    for part in PARTS:
        out.write(part.read_bytes())
actual = sha256(dst.read_bytes()).hexdigest()
if actual != EXPECTED:
    dst.unlink(missing_ok=True)
    raise SystemExit(f"SHA-256 mismatch: {actual}")
with ZipFile(dst) as archive:
    bad = archive.testzip()
    if bad: raise SystemExit(f"Corrupt ZIP entry: {bad}")
print("Verified complete current snapshot:", dst, actual)
print("Extract this archive, then run python run.py verify inside the extracted directory.")
