"""Group-level dependence and deterministic descriptive sensitivity tests."""
import sys
import unittest
from copy import deepcopy
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wn_group_variability import load_groups, totals, bootstrap, analyze
from verify_r9_10_strict import ROOT

class ClusterStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.groups=load_groups(ROOT/'evidence/pilot_linux')

    def test_45_groups_are_not_180_independent_runs(self):
        self.assertEqual(len(self.groups),45)
        self.assertEqual(totals(self.groups)['runs'],180)
        self.assertEqual(len({r['split_group_id'] for r in self.groups}),45)

    def test_strata_have_five_independent_group_units(self):
        from collections import Counter
        sizes=Counter((r['mobility'],r['n_nodes']) for r in self.groups)
        self.assertEqual(len(sizes),9)
        self.assertEqual(set(sizes.values()),{5})
        self.assertEqual(set(r['block'] for r in self.groups),set(range(2,7)))

    def test_pooled_counts_and_full_exposure(self):
        x=totals(self.groups)
        self.assertEqual((x['events'],x['true_alerts'],x['false_alerts'],x['minutes']),(1114,183,88,60))

    def test_block_counts_sum_and_block_6_exceeds_budget(self):
        blocks,_,_=analyze(self.groups)
        self.assertEqual([b['false_alerts'] for b in blocks],[19,8,10,19,32])
        self.assertEqual([b['within_budget'] for b in blocks],[True,True,True,True,False])
        self.assertEqual(sum(x['events'] for x in blocks),1114)

    def test_leave_one_out_gauss_markov_overbudget(self):
        _,_,gm=analyze(self.groups)
        self.assertTrue(all(b['false_alerts_per_minute']>2 for b in gm))
        self.assertEqual({b['trajectory_groups'] for b in gm},{12})

    def test_resampling_is_reproducible(self):
        a=bootstrap(self.groups,seed=20261121,resamples=180)
        b=bootstrap(self.groups,seed=20261121,resamples=180)
        self.assertEqual(a,b)
        self.assertEqual([r['trajectory_groups'] for r in a],[45,15,15,15])

    def test_rate_and_recall_inside_percentile_bands_for_reference(self):
        a=bootstrap(self.groups,seed=20261121,resamples=1200)
        for r in a:
            self.assertLessEqual(r['rate_percentile_2_5'],r['rate_percentile_97_5'])
            self.assertLessEqual(r['recall_percentile_2_5'],r['recall_percentile_97_5'])
            self.assertTrue(0 <= r['recall_percentile_97_5'] <= 1)
        self.assertGreater(a[0]['rate_percentile_97_5'],2.0)


    def test_rejects_corrupt_missing_fourth_variant(self):
        from tempfile import TemporaryDirectory
        import shutil
        with TemporaryDirectory() as td:
            p=Path(td)
            data=(ROOT/'evidence/pilot_linux/run_counts.csv').read_text()
            lines=data.splitlines()
            (p/'run_counts.csv').write_text('\n'.join(lines[:-1])+'\n')
            with self.assertRaises(ValueError):load_groups(p)

if __name__=='__main__': unittest.main()
