#!/usr/bin/env python3
"""Reassemble the only current research archive from transport chunks."""
from pathlib import Path
from hashlib import sha256
from zipfile import ZipFile
ROOT=Path(__file__).resolve().parent
EXPECTED="c83ce3c2d1a4b0d27ec20d94aba34563cd42b58a83ff4acb244e493650ec594b"
PARTS=sorted((ROOT/"snapshot").glob("current.zip.part[0-9][0-9]"))
if len(PARTS)!=9:raise SystemExit("Expected 9 chunks; got "+str(len(PARTS)))
out=ROOT/"FANET-TopoGNN_Current.zip"
with out.open("wb") as writer:
 for part in PARTS:writer.write(part.read_bytes())
actual=sha256(out.read_bytes()).hexdigest()
if actual!=EXPECTED:
 out.unlink(missing_ok=True)
 raise SystemExit("SHA-256 mismatch: "+actual)
with ZipFile(out) as z:
 bad=z.testzip()
 if bad:raise SystemExit("Corrupt ZIP item: "+bad)
print("Verified single current snapshot:",actual)
