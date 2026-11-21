from __future__ import annotations
import json
from pathlib import Path
import tempfile
import unittest

from scripts.r5_preflight import run_preflight, canonical_seeds
from scripts.r5_geometry_torch_pilot import run_pilot

ROOT=Path(__file__).resolve().parents[1]


class R5PreflightTests(unittest.TestCase):
    def setUp(self):
        self.spec=ROOT/'configs/revision_neural_20seed_full.json'
    def test_original_20_seeds_are_frozen(self):
        spec=json.loads(self.spec.read_text())
        self.assertEqual(spec['seed_list'],canonical_seeds())
    def test_missing_simpy_blocks_full_execution(self):
        out=run_preflight(root=ROOT,spec_path=self.spec,
            available=lambda x:x!='simpy')
        self.assertEqual(out['status'],'NO_GO')
        self.assertIn('simpy',out['dependencies'])
        self.assertFalse(out['dependencies']['simpy'])
        self.assertEqual(out['estimated_complete_frames_per_seed'],21600)
        self.assertEqual(out['estimated_eligible_frames_per_seed'],20952)
        self.assertEqual(out['scientific_submission'],'HOLD')
    def test_complete_dependencies_are_not_a_scientific_pass(self):
        out=run_preflight(root=ROOT,spec_path=self.spec,available=lambda x:True)
        self.assertEqual(out['status'],'READY_TO_ATTEMPT_FULL_PIPELINE')
        self.assertEqual(out['scientific_submission'],'HOLD')
    def test_tampered_seeds_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'spec.json'
            spec=json.loads(self.spec.read_text())
            spec['seed_list'][0]=42
            p.write_text(json.dumps(spec))
            out=run_preflight(root=ROOT,spec_path=p,available=lambda x:True)
            self.assertEqual(out['status'],'NO_GO')
    def test_duplicate_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'spec.json'
            spec=json.loads(self.spec.read_text())
            spec['selected_models'][0]=spec['selected_models'][1]
            p.write_text(json.dumps(spec))
            out=run_preflight(root=ROOT,spec_path=p,available=lambda x:True)
            self.assertEqual(out['status'],'NO_GO')


class R5ActualPilotTests(unittest.TestCase):
    def test_actual_radio_geometry_pytorch_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'new_pilot'
            result=run_pilot(root=ROOT,output_dir=path,seed=101,epochs=1)
            self.assertEqual(result['status'],'ENGINEERING_PILOT_PASS')
            self.assertEqual(result['full_snapshots'],3240)
            self.assertEqual(result['valid_forecast_snapshots'],2592)
            self.assertEqual(result['test_rows_exported'],result['role_rows']['test'])
            self.assertEqual(result['scientific_submission'],'HOLD')
            for g in result['role_groups'].values():
                self.assertGreaterEqual(g,1)
            with self.assertRaises(FileExistsError):
                run_pilot(root=ROOT,output_dir=path,seed=101,epochs=1)


if __name__=='__main__':
    unittest.main()

class R5FullSystemConfigTests(unittest.TestCase):
    def test_pilot_config_does_not_use_original_seed_or_output(self):
        from scripts.r5_full_system_smoke import build_smoke_config
        c=build_smoke_config(ROOT)
        self.assertEqual(c['training']['seed_list'],[101])
        self.assertNotIn(101,canonical_seeds())
        self.assertEqual(c['training']['selected_models'],['current_state_extratrees','GCN','tgcn:5'])
        self.assertEqual(c['training']['epochs'],1)
        self.assertTrue(c['training']['r4_strict_validation_roles'])
        self.assertEqual(c['output_dir'],'outputs/r5_full_system_smoke_seed101')
    def test_prepare_only_and_no_overwrite(self):
        from scripts.r5_full_system_smoke import prepare
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'smoke.json'
            report=Path(td)/'result.json'
            one=prepare(root=ROOT,config_path=p,report_path=report)
            self.assertEqual(one['status'],'PREPARED_ONLY_NOT_EXECUTED')
            self.assertEqual(one['scientific_submission'],'HOLD')
            self.assertTrue(p.exists())
            # Repeat with identical config is allowed and does not rewrite.
            old=p.read_bytes()
            prepare(root=ROOT,config_path=p,report_path=report)
            self.assertEqual(p.read_bytes(),old)
            p.write_text('{"tampered":true}')
            with self.assertRaises(FileExistsError):prepare(root=ROOT,config_path=p,report_path=report)
