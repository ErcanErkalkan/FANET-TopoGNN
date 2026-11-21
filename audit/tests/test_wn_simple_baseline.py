"""Independent structural/semantic tests for the post-hoc history-only reference."""
from __future__ import annotations
import csv
import importlib.util
import json
import math
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data_derived/wn_simple_baseline'
SPEC=importlib.util.spec_from_file_location('wn_simple_baseline',ROOT/'audit/wn_simple_baseline_comparison.py')
BASE=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)


def read(path):
    with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

class TestHistoryFour(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.receipt=json.loads((OUT/'COMPARISON_RECEIPT.json').read_text())
        cls.sel=read(OUT/'threshold_selection_grid.csv')
        cls.paired=read(OUT/'paired_evaluation_by_run.csv')
        cls.summary=read(OUT/'paired_summary.csv')

    def test_history_uses_past_only(self):
        truth={t:1 for t in range(200)}
        truth[2]=2;truth[3]=3
        a=BASE.heuristic(truth)
        self.assertEqual(a[4],.5)
        self.assertEqual(a[5],.5)
        self.assertEqual(a[6],.5)
        # future and current fragmentation cannot influence today's score.
        truth[4]=99;truth[10]=7
        b=BASE.heuristic(truth)
        self.assertEqual(a[4],b[4])
        self.assertEqual(set(a),set(range(4,194)))

    def test_same_event_matcher_zero_alert(self):
        truth={t:1 for t in range(200)};truth[19]=3;truth[20]=1
        r=BASE.evaluate(truth,{t:0 for t in range(4,194)},.30)
        self.assertEqual((r['events'],r['true_alerts'],r['false_alerts']), (1,0,0))

    def test_selection_grid_and_budget(self):
        self.assertEqual(len(self.sel),19)
        self.assertEqual([float(x['threshold']) for x in self.sel],list(BASE.GRID))
        chosen=next(x for x in self.sel if float(x['threshold'])==.30)
        self.assertEqual((int(chosen['true_alerts']),int(chosen['false_alerts'])),(43,21))
        feasible=[x for x in self.sel if x['within_budget']=='True']
        winner=min(feasible,key=lambda x:(-float(x['event_f1']),float(x['false_alerts_per_minute']),
                                         -float(x['recall']),float(x['threshold'])))
        self.assertEqual(float(winner['threshold']),.30)

    def test_pairing_and_independence(self):
        self.assertEqual(len(self.paired),360)
        runs={k:{x['run_id'] for x in self.paired if x['model']==k} for k in ['TopoGNN','History-4']}
        self.assertEqual(runs['TopoGNN'],runs['History-4']);self.assertEqual(len(runs['TopoGNN']),180)
        self.assertEqual(len({x['split_group_id'] for x in self.paired}),45)
        self.assertTrue(all(sum(x['split_group_id']==gid and x['model']=='TopoGNN' for x in self.paired)==4
                            for gid in {x['split_group_id'] for x in self.paired}))
        self.assertEqual({int(x['block']) for x in self.paired},{2,3,4,5,6})
        self.assertEqual({int(x['observed_frames']) for x in self.paired},{200})

    def test_original_replay_matches_saved_reference(self):
        original=next((ROOT/'audit/reference_linux').iterdir())
        saved={r['run_id']:r for r in read(original/'run_counts.csv')}
        for r in self.paired:
            if r['model']=='TopoGNN':
                self.assertEqual(tuple(int(r[x]) for x in ('events','true_alerts','false_alerts')),
                                 tuple(int(saved[r['run_id']][x]) for x in ('events','true_alerts','false_alerts')))

    def test_group_denominator_and_total(self):
        for model,tp,fp in [('TopoGNN',183,88),('History-4',381,135)]:
            subset=[r for r in self.paired if r['model']==model]
            self.assertEqual(sum(int(r['events']) for r in subset),1114)
            self.assertEqual(sum(int(r['true_alerts']) for r in subset),tp)
            self.assertEqual(sum(int(r['false_alerts']) for r in subset),fp)
            self.assertEqual(sum(int(r['observed_frames']) for r in subset),36000)
            self.assertEqual(BASE.summarize([{k:int(r[k]) for k in ['events','true_alerts','false_alerts','observed_frames']} for r in subset])['false_alerts_per_minute'],fp/60)

    def test_transfer_budget_and_no_tuning(self):
        self.assertAlmostEqual(self.receipt['evaluation']['TopoGNN']['false_alerts_per_minute'],88/60)
        self.assertAlmostEqual(self.receipt['evaluation']['History-4']['false_alerts_per_minute'],135/60)
        self.assertTrue(self.receipt['evaluation']['TopoGNN']['within_budget'])
        self.assertFalse(self.receipt['evaluation']['History-4']['within_budget'])
        self.assertEqual(self.receipt['baseline_selected_threshold'],.30)
        self.assertEqual(self.receipt['status'],'POST_HOC_DEVELOPMENT_ONLY_COMPARISON_PUBLICATION_HOLD')
        self.assertFalse(self.receipt['prospectively_preregistered'])
        self.assertFalse(self.receipt['blind_holdout_read'])
        self.assertFalse(self.receipt['superiority_test'])

    def test_mobility_and_strata_sum(self):
        for m in ('TopoGNN','History-4'):
            pooled=next(x for x in self.summary if x['model']==m and x['scope']=='pooled')
            mob=[x for x in self.summary if x['model']==m and x['scope']=='mobility']
            node=[x for x in self.summary if x['model']==m and x['scope']=='node_count']
            for field in ('events','true_alerts','false_alerts','alerts'):
                self.assertEqual(sum(int(r[field]) for r in mob),int(pooled[field]))
                self.assertEqual(sum(int(r[field]) for r in node),int(pooled[field]))

    def test_main_and_supplement_claims(self):
        s=(ROOT/'paper/manuscript/main.tex').read_text()
        t=(ROOT/'paper/manuscript/ESM_1.tex').read_text()
        for needle in ('381','0.467','2.25','post-hoc','not prospective blind tests'):
            self.assertIn(needle,s)
        for needle in ('381','135','1.75','0.30','NOT blind'):
            self.assertIn(needle,t)
        self.assertIn('neither deployment readiness nor model superiority',s)

    def test_paired_resampling_not_significance_claim(self):
        b=self.receipt['bootstrap']
        self.assertEqual(b['resamples'],10000)
        self.assertEqual(b['status'],'DESCRIPTIVE_POST_HOC_NOT_CONFIRMATORY')
        self.assertEqual(len(b['event_f1_topognn_minus_baseline_95pct']),2)
        self.assertTrue(all(math.isfinite(x) for x in b['event_f1_topognn_minus_baseline_95pct']))

if __name__=='__main__':unittest.main()
