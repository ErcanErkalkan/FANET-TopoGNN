from __future__ import annotations
import csv
import sys
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.r8_audit_results import holm_adjust, check_score_rows

FIELDS=['run_id','time_index','future_time_index','split_group_id','beta_target','frag_at_horizon','risk_score','selected_threshold']
VALID=dict(zip(FIELDS,['a','1','7','split','2.0','1','0.4','0.3']))

class R8GateTests(unittest.TestCase):
    def write(self,path,rows,fields=FIELDS):
        with path.open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    def test_holm_simple(self):
        self.assertEqual(holm_adjust([.01,.02,.4]),[.03,.04,.4])
    def test_holm_order(self):
        self.assertEqual(holm_adjust([.4,.02,.01]),[.4,.04,.03])
    def test_holm_invalid(self):
        with self.assertRaises(ValueError):holm_adjust([float('nan')])
    def test_holm_empty(self):
        self.assertEqual(holm_adjust([]),[])
    def test_score_valid(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'scores.csv';self.write(path,[VALID]);keys,signatures=check_score_rows(path)
            self.assertEqual(keys,{('a',1)});self.assertEqual(signatures[('a',1)][-1],'1')
    def test_score_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'scores.csv';self.write(path,[VALID,VALID])
            with self.assertRaisesRegex(ValueError,'Duplicate'):check_score_rows(path)
    def test_nonfuture_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'scores.csv';self.write(path,[{**VALID,'future_time_index':'1'}])
            with self.assertRaisesRegex(ValueError,'forward'):check_score_rows(path)
    def test_wrong_horizon_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'scores.csv';self.write(path,[{**VALID,'future_time_index':'9'}])
            with self.assertRaisesRegex(ValueError,'forward'):check_score_rows(path,expected_horizon=6)
    def test_nonbinary_target_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'scores.csv';self.write(path,[{**VALID,'frag_at_horizon':'2'}])
            with self.assertRaisesRegex(ValueError,'Nonbinary'):check_score_rows(path)
    def test_nonfinite_score_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'scores.csv';self.write(path,[{**VALID,'risk_score':'nan'}])
            with self.assertRaisesRegex(ValueError,'Nonfinite'):check_score_rows(path)
    def test_missing_field_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'scores.csv';fields=[x for x in FIELDS if x!='frag_at_horizon'];self.write(path,[{k:VALID[k] for k in fields}],fields)
            with self.assertRaisesRegex(ValueError,'Missing score'):check_score_rows(path)

if __name__=='__main__':unittest.main()
