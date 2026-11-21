"""Independent, fail-closed inventory gate for NEW R4 model artifacts only.

Never promotes prior R4 summary JSONs. This is not a physical FANET replay or
validation of the legacy 280/280 numerical audit.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fanet.r4_evidence import NEURAL_NAMES, evidence_location, verify_evidence


def _read_rows(filename: Path) -> dict[tuple[str, int], tuple[int, float]]:
    with filename.open('r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = {}
        for row in reader:
            key = (row['run_id'], int(row['time_index']))
            if key in rows:
                raise ValueError(f'Duplicate snapshot key in {filename}: {key}')
            rows[key] = (int(row['frag_at_horizon']), float(row['beta_target']))
    if not rows:
        raise ValueError(f'Empty evidence: {filename}')
    return rows


def audit_r4_evidence(output_dir: Path, seeds: list[int], *, expected_models: set[str] | None = None,
                      expected_signature: str | None = None) -> dict:
    models = sorted(expected_models if expected_models is not None else NEURAL_NAMES)
    if not models or not seeds or len(seeds) != len(set(seeds)):
        raise ValueError('Expected models/seeds must be unique and nonempty')
    failures=[]
    verified=0
    for seed in seeds:
        seed_dir=Path(output_dir)/'per_seed'/f'seed_{seed}'
        reference=None
        reference_cal=None
        reference_sel=None
        for model in models:
            if not verify_evidence(seed_dir,model,expected_signature,require_role_isolation=True):
                failures.append(f'{seed}/{model}: missing or unverifiable evidence')
                continue
            directory=evidence_location(seed_dir,model)
            try:
                rows=_read_rows(directory/'test.csv')
                calibration=_read_rows(directory/'validation_calibration_fit.csv')
                selection=_read_rows(directory/'validation_threshold_selection.csv')
                if reference is not None and rows != reference:
                    raise ValueError('different held-out test snapshot keys or labels')
                if reference_cal is not None and calibration != reference_cal:
                    raise ValueError('different calibration keys or labels')
                if reference_sel is not None and selection != reference_sel:
                    raise ValueError('different threshold-selection keys or labels')
                reference=rows
                reference_cal=calibration
                reference_sel=selection
                verified+=1
            except (OSError, ValueError, KeyError, TypeError) as exc:
                failures.append(f'{seed}/{model}: {exc}')
    return dict(status='PASS' if not failures and verified==len(seeds)*len(models) else 'HOLD',
                scientific_submission='HOLD',
                model_seed_evidence_verified=verified,
                model_seed_evidence_expected=len(seeds)*len(models),
                errors=failures,
                scope='New R4 artifact identity/hash/role coverage; NOT independent model-training or packet-replay reproduction')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--signature',type=str,default=None)
    parser.add_argument('--report',type=Path,default=None)
    args=parser.parse_args()
    spec=json.loads(args.spec.read_text(encoding='utf-8'))
    result=audit_r4_evidence(args.output_dir, [int(v) for v in spec['seed_list']],
                             expected_signature=args.signature)
    blob=json.dumps(result,indent=2,ensure_ascii=False)+'\n'
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(blob,encoding='utf-8')
    print(blob)
    if result['status']!='PASS':raise SystemExit(2)

if __name__=='__main__': main()
