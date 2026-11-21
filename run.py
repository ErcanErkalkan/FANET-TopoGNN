"""Research verification entry point for FANET-TopoGNN."""

import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
AUDIT = ROOT / 'audit'
FROZEN = ROOT / 'frozen'
REPORTS = {
    'Linux': AUDIT / 'reference_linux/fixed_threshold_20261115T192417_406441Z',
    'Windows': AUDIT / 'reference_windows/fixed_threshold_20261116T161416_063392Z',
}
REPORT = 'R9_10_FIXED_THRESHOLD_AUDIT.json'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_package():
    manifest = json.loads((ROOT / 'MANIFEST.json').read_text(encoding='utf-8'))
    for rel, expected in manifest['sha256'].items():
        if sha(ROOT / rel) != expected:
            raise ValueError(f'Package SHA-256 mismatch: {rel}')
    return len(manifest['sha256'])


def checked_runner(action):
    host = platform.system()
    if host not in REPORTS:
        raise ValueError(f'No pinned R9.9/R9.10 environment for {host}')
    args = {
        'workspace': ROOT / 'core',
        'plan': FROZEN / 'plan',
        'r96-code': FROZEN / 'r9_6_cost_pilot.py',
        'r96-windows-report': FROZEN / 'R9_6_COST_PILOT.json',
        'r97-code': FROZEN / 'r9_7_training_pilot.py',
        'r98-code': FROZEN / 'r9_8_shared_pilot.py',
        'r99-code': FROZEN / 'r9_9_calibration_alert.py',
        'reference': FROZEN / 'anchor' / host,
    }
    cli = [sys.executable, '-I', '-B', str(AUDIT / 'r9_10_fixed_threshold_audit.py')]
    for name, path in args.items():
        cli.extend(('--' + name, str(path)))
    cli.extend(['--check'] if action == 'check' else ['--output', str(ROOT / 'outputs')])
    subprocess.run(cli, check=True)


def verify(report):
    cmd = [sys.executable, '-I', '-B', str(AUDIT / 'verify_r9_10_strict.py'), str(report)]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'verify', 'pilot', 'test'))
    parser.add_argument('--report', type=Path, help='Verify a specific report directory.')
    args = parser.parse_args()
    if args.report and args.action != 'verify':
        parser.error('--report is available only with verify')
    print(f'Verified {verify_package()} frozen package files.', flush=True)
    if args.action == 'verify':
        for name, path in ({'Custom': args.report.resolve()} if args.report else REPORTS).items():
            print(name, flush=True)
            verify(path)
    elif args.action == 'test':
        subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s',
                        str(AUDIT / 'tests'), '-v'], cwd=ROOT, check=True)
    else:
        checked_runner(args.action)


if __name__ == '__main__':
    main()
