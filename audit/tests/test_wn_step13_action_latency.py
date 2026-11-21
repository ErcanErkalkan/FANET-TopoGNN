"""Step 13: unmodified matching, hypothetical latency and rendering source checks."""
import csv
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'audit'))
import wn_lead_latency_sensitivity as study

class ActionLatencyAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data,cls.hist,cls.mob=study.consistent()

    def sample(self,method,L):
        return next(r for r in self.data if r['method']==method and
                    r['assumed_effective_action_latency_seconds']==L)

    def test_preserved_topognn_counts(self):
        a=self.sample('TopoGNN','0.1')
        self.assertEqual((a['events_total'],a['matched_alerts'],a['false_alerts_unchanged']), (1114,183,88))
    def test_preserved_history_counts(self):
        a=self.sample('History-4','0.1')
        self.assertEqual((a['events_total'],a['matched_alerts'],a['false_alerts_unchanged']), (1114,381,135))
    def test_lead_histograms_reconcile(self):
        self.assertEqual([int(r['matched_alerts']) for r in self.hist if r['method']=='TopoGNN'], [85,40,25,13,13,7])
        self.assertEqual([int(r['matched_alerts']) for r in self.hist if r['method']=='History-4'],[170,79,40,40,24,28])
    def test_latency_topognn(self):
        expected={'0.1':183,'0.2':98,'0.3':58,'0.4':33,'0.5':20,'0.6':7}
        for latency,count in expected.items():
            self.assertEqual(self.sample('TopoGNN',latency)['potentially_timely_matched_alerts'],count)
    def test_latency_history(self):
        expected={'0.1':381,'0.2':211,'0.3':132,'0.4':92,'0.5':52,'0.6':28}
        for latency,count in expected.items():
            self.assertEqual(self.sample('History-4',latency)['potentially_timely_matched_alerts'],count)
    def test_monotonic_timely_fraction_and_fixed_false_alerts(self):
        for method in ('TopoGNN','History-4'):
            selected=[r for r in self.data if r['method']==method]
            a=[r['potentially_timely_matched_alerts'] for r in selected]
            self.assertEqual(a,sorted(a,reverse=True))
            self.assertEqual(len({r['false_alerts_unchanged'] for r in selected}),1)
            self.assertEqual({r['events_total'] for r in selected},{1114})
    def test_mobility_sums_reconcile(self):
        for method in ('TopoGNN','History-4'):
            for L in ('0.1','0.2','0.3'):
                row=self.sample(method,L)
                subset=[m for m in self.mob if m['method']==method and m['latency_seconds']==L]
                self.assertEqual(len(subset),3)
                self.assertEqual(sum(m['potentially_timely'] for m in subset),row['potentially_timely_matched_alerts'])
    def test_csv_matches_recomputation(self):
        files=[('wn_r9_10_action_latency_sensitivity.csv',self.data),('wn_r9_10_lead_distribution.csv',self.hist),('wn_r9_10_action_latency_by_mobility.csv',self.mob)]
        for name,computed in files:
            with self.subTest(name=name):
                with (ROOT/'data_derived'/name).open(newline='',encoding='utf-8') as f:
                    stored=list(csv.DictReader(f))
                self.assertEqual(len(stored),len(computed))
                self.assertEqual(stored,[{key:str(value) for key,value in row.items()} for row in computed])

if __name__=='__main__':unittest.main()
