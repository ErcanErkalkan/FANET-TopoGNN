"""R6 independent GCN replay gates, including adversarial archive edits."""
from __future__ import annotations
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

HERE=Path(__file__).resolve()
REPLAY=HERE.parents[1]/'scripts'/'r6_independent_checkpoint_replay.py'
spec=importlib.util.spec_from_file_location('r6_independent_checkpoint_replay',REPLAY)
r6=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=r6
spec.loader.exec_module(r6)

# Test paths supplied by env when running out-of-tree; default project-local tests
# can locate R5 artifacts via an explicit operator-specified directory.
import os
ROOT=Path(os.environ.get('FANET_R6_PROJECT_ROOT',HERE.parents[2])).resolve()
EVIDENCE=Path(os.environ.get('FANET_R6_PILOT_EVIDENCE','')).resolve() if os.environ.get('FANET_R6_PILOT_EVIDENCE') else None
R5_REPORT=Path(os.environ.get('FANET_R6_PILOT_REPORT','')).resolve() if os.environ.get('FANET_R6_PILOT_REPORT') else None


class TestR6Replay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if EVIDENCE is None or not EVIDENCE.exists():
            raise unittest.SkipTest('Set FANET_R6_PILOT_EVIDENCE and project path for archived-pilot verification')
        if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
        cls.original_views=r6._pilot_views(ROOT,101)

    def _copy(self):
        temp=tempfile.TemporaryDirectory(prefix='r6_adversarial_')
        path=Path(temp.name)/'GCN'
        shutil.copytree(EVIDENCE,path)
        self.addCleanup(temp.cleanup)
        return path

    def _edit_csv(self,path,name,change):
        p=path/name
        with p.open(encoding='utf-8',newline='') as f:
            rdr=csv.DictReader(f)
            fields=rdr.fieldnames
            rows=list(rdr)
        change(rows)
        with p.open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        data=json.loads((path/'MANIFEST.json').read_text())
        data['artifact_sha256'][name]=hashlib.sha256(p.read_bytes()).hexdigest()
        (path/'MANIFEST.json').write_text(json.dumps(data,indent=2))

    def _audit(self,path):
        # Source reconstruction is performed once per test class to make tests
        # fast; this mocks only the regenerated view, NEVER the checkpoint or CSVs.
        with patch.object(r6,'_pilot_views',return_value=self.original_views):
            return r6.audit(root=ROOT,evidence=path,source_report=R5_REPORT)

    def test_01_untouched_evidence_replays(self):
        result=self._audit(EVIDENCE)
        self.assertEqual(result['status'],'ENGINEERING_PILOT_REPLAY_PASS')
        self.assertEqual(result['replayed_prediction_rows'],960)
        self.assertEqual(result['test_rows'],576)
        self.assertEqual(result['engineering_test_confusion']['tn'],0)

    def test_02_unupdated_sha_is_rejected(self):
        path=self._copy()
        with (path/'test.csv').open('ab') as f:f.write(b'\n')
        with self.assertRaisesRegex(r6.ReplayFailure,'SHA256 mismatch'):
            self._audit(path)

    def test_03_renamed_time_rejected_even_if_manifest_rehashed(self):
        path=self._copy()
        self._edit_csv(path,'test.csv',lambda rows:rows[0].__setitem__('time_index','999'))
        with self.assertRaisesRegex(r6.ReplayFailure,'Row identity mismatch'):
            self._audit(path)

    def test_04_tampered_label_rejected_even_if_manifest_rehashed(self):
        path=self._copy()
        self._edit_csv(path,'test.csv',lambda rows:rows[0].__setitem__('frag_at_horizon', '0' if rows[0]['frag_at_horizon']=='1' else '1'))
        with self.assertRaisesRegex(r6.ReplayFailure,'frag_at_horizon mismatch'):
            self._audit(path)

    def test_05_tampered_prediction_rejected_even_if_manifest_rehashed(self):
        path=self._copy()
        self._edit_csv(path,'test.csv',lambda rows:rows[0].__setitem__('predicted_beta',str(float(rows[0]['predicted_beta'])+0.2)))
        with self.assertRaisesRegex(r6.ReplayFailure,'Checkpoint replay mismatch'):
            self._audit(path)

    def test_06_tampered_calibrated_probability_rejected_even_if_manifest_rehashed(self):
        path=self._copy()
        self._edit_csv(path,'test.csv',lambda rows:rows[0].__setitem__('risk_probability',str(float(rows[0]['risk_probability'])+0.1)))
        with self.assertRaisesRegex(r6.ReplayFailure,'Calibrated probability replay mismatch'):
            self._audit(path)

    def test_07_independent_threshold_tiebreak(self):
        probs=np.array([0.3,0.6,0.9])
        labels=np.array([0,1,1])
        thr,info=r6._select_threshold_independent(probs,labels)
        self.assertAlmostEqual(thr,0.6)
        self.assertEqual((info['tp'],info['fp'],info['fn']),(2,0,0))

    def test_08_source_hash_mismatch_rejected(self):
        path=self._copy()
        data=json.loads((path/'MANIFEST.json').read_text())
        data['source_sha256']['training.py']='0'*64
        (path/'MANIFEST.json').write_text(json.dumps(data))
        with self.assertRaisesRegex(r6.ReplayFailure,'Source SHA256 drift'):
            self._audit(path)


if __name__=='__main__':unittest.main()
