"""Independent cross-archive table evidence check, without invented raw scores."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
from zipfile import ZipFile


def _csv(z:ZipFile,name:str):
    return list(csv.DictReader(io.StringIO(z.read(name).decode('utf-8-sig'))))


def _hash(path:Path):return hashlib.sha256(path.read_bytes()).hexdigest()


def gate(ref:Path, output:Path|None=None):
    ref=Path(ref)
    r2=ref/'FANET_R2_STRICT_V3_600_NUMERICAL_SCIENTIFIC_AUDIT_2026-10-16.zip'
    r4=ref/'FANET_R4_280_BAGIMSIZ_BILIMSEL_DENETIM_2026-10-19.zip'
    with ZipFile(r2) as z2, ZipFile(r4) as z4:
        if z2.testzip() or z4.testzip():raise ValueError('Corrupt historical audit ZIP CRC')
        t2=_csv(z2,'R2_V3_ALL_600_SEED_MODEL_METRICS.csv')
        p2=_csv(z2,'R2_V3_120_CELL_PROVENANCE.csv')
        t4=_csv(z4,'R4_280_INDEPENDENT_EXACT_RECORDS.csv')
        a2=json.loads(z2.read('R2_V3_INDEPENDENT_AUDIT.json'))
        a4=json.loads(z4.read('R4_INDEPENDENT_INTEGRITY_AUDIT.json'))
    if len(t2)!=600 or len(p2)!=120 or len(t4)!=280:raise ValueError('R2/R4 source table coverage incomplete')
    if a4['r2_reference_schema_counts']!={'r2_strict_v2_1':20}:raise ValueError('Unexpected R2 version label')
    if set(x['runner_version'] for x in p2)!={'r2_strict_v3_1'}:raise ValueError('Unexpected R2 v3 provenance')
    r2h6={(int(row['seed']),row['model']):row for row in t2 if int(row['horizon_steps'])==6}
    r4refs={(int(row['seed']),row['model']):row for row in t4 if row['source']=='R2 verified, reused'}
    if len(r2h6)!=100 or len(r4refs)!=100 or set(r2h6)!=set(r4refs):
        raise ValueError('100/100 R4 classical baseline identity mismatch')
    equals=0;greatest_diff=0.
    fields=[('n_test_scored','n_test_scored'),('Alert_Event_F1','event_f1'),('Risk_F1','risk_f1'),
            ('Ground_Truth_Fragmentation_Events','true_events'),('True_Alert_Events','tp'),
            ('False_Alert_Events','fp'),('Missed_Fragmentation_Events','fn'),
            ('selected_threshold','threshold')]
    for key,v in r2h6.items():
        reference=r4refs[key]
        for r2col,r4col in fields:
            diff=abs(float(v[r2col])-float(reference[r4col]))
            greatest_diff=max(greatest_diff,diff)
            if not math.isclose(float(v[r2col]),float(reference[r4col]),rel_tol=1e-9,abs_tol=1e-8):
                raise ValueError(f'R2/R4 source mismatch {key}/{r2col}: {diff}')
        equals+=1
    r5audit=json.loads((ref/'R5_STRESS_INDEPENDENT_INTEGRITY_AUDIT.json').read_text())
    stress=list(csv.DictReader((ref/'R5_STRESS_8_SCENARIOS_6_POLICY_SUMMARY.csv').open(newline='')))
    paired=list(csv.DictReader((ref/'R5_STRESS_40_PAIRED_POLICY_VS_NO_INTERVENTION_HOLM.csv').open(newline='')))
    if len(stress)!=48 or len(paired)!=40:raise ValueError('R5 statistical table coverage missing')
    if any(float(x['p_holm'])<0.05 or x['significant_005']=='True' for x in paired):
        raise ValueError('R5 Holm decision differs from source table')
    baseline=[x for x in stress if x['scenario']=='baseline']
    if len(baseline)!=6:raise ValueError('R5 baseline policy inventory mismatch')
    active=[r for r in baseline if r['policy']!='No intervention']
    if len(active)!=5:raise ValueError('R5 active policy inventory mismatch')
    budget={r['policy']:{'action_duty':float(r['mean_action_duty']),
                          'relay_m_per_minute':float(r['mean_relay_travel']),
                          'duty_limit_satisfied':float(r['mean_action_duty'])<=0.25,
                          'relay_limit_satisfied':float(r['mean_relay_travel'])<=300.} for r in active}
    out={
        'schema':'fanet_r7_cross_archive_evidence_gate_v1',
        'status':'NUMERICAL_ARCHIVE_CROSS_CHECK_PASS__SCIENTIFIC_HOLD',
        'r2_models':len(t2), 'r2_provenance_cells':len(p2),
        'r4_models':len(t4),'r2_h6_r4_identical_classical_models':equals,
        'max_cross_archive_abs_numeric_diff':greatest_diff,
        'r2_v3_provenance_version':'r2_strict_v3_1',
        'r4_embedded_r2_schema_tag':'r2_strict_v2_1',
        'schema_tag_lineage_resolved':False,
        'schema_tag_lineage_limitation':'Original v3 runner source and raw h6 JSON are needed; no retroactive version-label edit permitted',
        'r5_stress_metrics':len(stress), 'r5_paired_tests':len(paired),
        'r5_holm_significant_005':0,
        'r5_paired_min_holm_p':min(float(x['p_holm']) for x in paired),
        'r5_baseline_policy_budgets':budget,
        'r5_baseline_all_active_relay_budget_failed':all(not x['relay_limit_satisfied'] for x in budget.values()),
        'r5_audit_recorded_checks':r5audit['independent_checks'],
        'r2_audit_zip_sha256':_hash(r2), 'r4_audit_zip_sha256':_hash(r4),
        'neural_retraining_completed_20_seed':False,
        'publisher_submission':'HOLD',
    }
    if output:
        output=Path(output)
        if output.exists():raise FileExistsError(output)
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    return out

if __name__=='__main__':
    ref=Path(__file__).resolve().parents[1]/'references/historical'
    out=Path(__file__).resolve().parents[1]/'reports/R7_R2_R4_R5_HISTORICAL_GATE.json'
    value=gate(ref,out)
    print(json.dumps({key:value[key] for key in ('status','r2_models','r4_models','r2_h6_r4_identical_classical_models','r5_holm_significant_005','schema_tag_lineage_resolved')},ensure_ascii=False))
