"""Static wiring guards; not substitutes for full SimPy experiment integration."""
from __future__ import annotations
import ast
from pathlib import Path
import unittest


class WiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parents[1]
        cls.source=(root/'fanet'/'pipeline.py').read_text()
        cls.runner=(root/'scripts'/'run_revision_neural_20seed_full_R3.py').read_text()

    def test_pipeline_optional_gate_and_cache_validation(self):
        ast.parse(self.source)
        self.assertIn("config.training.get('r4_export_raw_evidence', False)",self.source)
        self.assertIn('verify_evidence(seed_dir, task_name, signature, require_role_isolation=',self.source)
        self.assertIn('export_torch_evidence(',self.source)

    def test_r3_runner_isolation(self):
        ast.parse(self.runner)
        self.assertIn('_r3_evidence',self.runner)
        self.assertIn('r4_export_raw_evidence',self.runner)
        self.assertIn('require_evidence=True',self.runner)

if __name__=='__main__':unittest.main()
