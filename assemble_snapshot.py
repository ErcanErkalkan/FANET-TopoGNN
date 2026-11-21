"""Verify and unpack the current research-only archive without overwriting local changes."""
from pathlib import Path
from zipfile import ZipFile
from hashlib import sha256
import json
ROOT=Path(__file__).resolve().parent
ARCHIVE=ROOT/'snapshot/research_only.zip'
PARTS=sorted((ROOT/'snapshot').glob('research.part[0-9][0-9]'))
EXPECTED_PARTS=14
EXPECTED='52ffe43ffa58c4d248465055dde8ef82ce2cb1993de59938058cd19d9c4693f4'

def main():
    if len(PARTS)!=EXPECTED_PARTS:
        raise SystemExit(f'Expected {EXPECTED_PARTS} research transport parts, found {len(PARTS)}')
    if not ARCHIVE.exists():
        ARCHIVE.write_bytes(b''.join(part.read_bytes() for part in PARTS))
    actual=sha256(ARCHIVE.read_bytes()).hexdigest()
    if actual!=EXPECTED:
        raise SystemExit('Research archive SHA-256 mismatch: '+actual)
    manifest=json.loads((ROOT/'MANIFEST.json').read_text(encoding='utf-8'))['sha256']
    count=0
    with ZipFile(ARCHIVE) as archive:
        if archive.testzip() is not None:
            raise SystemExit('Corrupt research ZIP entry')
        for entry in archive.infolist():
            name=entry.filename
            if name not in manifest:
                raise SystemExit('Unrecognized research ZIP entry: '+name)
            dest=ROOT/name
            if not dest.resolve().is_relative_to(ROOT.resolve()):
                raise SystemExit('Unsafe ZIP path: '+name)
            data=archive.read(entry)
            if sha256(data).hexdigest()!=manifest[name]:
                raise SystemExit('File checksum mismatch: '+name)
            if dest.exists():
                if sha256(dest.read_bytes()).hexdigest()!=manifest[name]:
                    raise SystemExit('Existing local file differs; refusing overwrite: '+name)
            else:
                dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_bytes(data)
                count+=1
    print(f'Research-only snapshot verified; {count} missing files restored.')

if __name__=='__main__':main()
