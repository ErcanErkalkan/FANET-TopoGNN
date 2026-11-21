"""Step-4 regression checks for selection and exposure arithmetic."""
import csv
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'audit'))
import wn_budget_exposure as w


class BudgetExposureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_paths=[next((ROOT/f'audit/reference_{host}').iterdir()) for host in ('linux','windows')]
        cls.source_paths.append(ROOT/'evidence/pilot_linux')
        cls.sources=[w.build_for_source(d) for d in cls.source_paths]
        cls.rows=w.reconcile(cls.sources)

    def test_selection_is_first_block_in_sample_no_deployment(self):
        x=w.selection_receipt()
        self.assertEqual((x['candidate_thresholds'],x['feasible_under_full_budget'],
                          x['feasible_with_true_alert'],x['selection_false_alerts']), (19,17,10,24))
        self.assertEqual(x['selection_false_alerts']/x['selection_full_minutes'],2.0)
        self.assertTrue(x['in_sample_diagnostic_only'])

    def test_180_runs_frames_and_two_denominators(self):
        p=w.get(self.rows,'pooled')
        self.assertEqual(p['runs'],180)
        self.assertEqual(p['decision_rows'],34200)
        self.assertEqual(p['connected_decision_frames'],28301)
        self.assertAlmostEqual(p['connected_decision_minutes'],28301/600,places=10)
        self.assertEqual(p['full_minutes'],60)
        self.assertEqual(p['false_alerts'],88)
        self.assertAlmostEqual(p['full_false_alarms_per_minute'],88/60)
        self.assertAlmostEqual(p['connected_decision_false_alarms_per_minute'],88*600/28301)

    def test_mobility_strata_reconcile_to_total(self):
        m=[w.get(self.rows,'pooled_mobility',mobility=x) for x in w.MOBILITIES]
        self.assertEqual(sum(x['connected_decision_frames'] for x in m),28301)
        self.assertEqual([x['connected_decision_frames'] for x in m],[10130,8472,9699])
        self.assertEqual([x['false_alerts'] for x in m],[18,49,21])
        self.assertAlmostEqual(m[1]['connected_decision_false_alarms_per_minute'],49*600/8472)

    def test_over_limit_changes_from_six_to_eight_only_by_denominator(self):
        cells=[x for x in self.rows.values() if x['scope']=='replicate_mobility']
        self.assertEqual(len(cells),15)
        self.assertEqual(sum(not x['full_budget_met'] for x in cells),6)
        self.assertEqual(sum(not x['connected_numerical_reference_met'] for x in cells),8)
        new={(x['replicate'],x['mobility']) for x in cells
             if x['full_budget_met'] and not x['connected_numerical_reference_met']}
        self.assertEqual(new,{(2,'mission'),(3,'gm')})
        self.assertEqual(sum(x['false_alerts'] for x in cells),88)

    def test_three_evidence_sources_and_csv_match_exactly(self):
        self.assertEqual(len(self.sources),3)
        self.assertEqual({len(s['rows']) for s in self.sources},{24})
        with (ROOT/'data_derived/wn_r9_10_exposure_sensitivity.csv').open(newline='',encoding='utf-8') as f:
            rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),24)
        self.assertEqual({(r['scope'],r['replicate'],r['mobility']) for r in rows},
                         {(k[0],'' if k[1] is None else str(k[1]),k[2] or '') for k in self.rows})
        receipt=json.loads((ROOT/'evidence/STEP4_EXPOSURE_THRESHOLD_RECEIPT.json').read_text())
        self.assertEqual(receipt['block_mobility_full_budget_exceedances'],6)
        self.assertEqual(receipt['block_mobility_connected_numerical_reference_exceedances'],8)
        self.assertTrue(receipt['no_new_training_or_threshold_tuning'])

    def test_supplement_generated_blocks_are_current(self):
        text=(ROOT/'paper/manuscript/ESM_1.tex').read_text()
        pooled, detailed=w.produce_tex(self.rows)
        for marker,payload in [('VERIFIED EXPOSURE SENSITIVITY POOLED',pooled),
                               ('VERIFIED EXPOSURE SENSITIVITY BLOCK MOBILITY',detailed)]:
            body=text.split(f'% BEGIN {marker}\n')[1].split(f'% END {marker}')[0]
            self.assertEqual(body,payload)

if __name__=='__main__':
    unittest.main()
