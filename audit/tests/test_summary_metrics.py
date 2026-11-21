"""Independent checks for frozen derived result summaries."""
from pathlib import Path
import csv
import unittest
ROOT=Path(__file__).resolve().parents[2]
D=ROOT/'data_derived'

def rows(name):
    with (D/name).open(newline='',encoding='utf-8') as f:
        return list(csv.DictReader(f))

class SummaryMetricTests(unittest.TestCase):
    def test_node_by_mobility_totals(self):
        r=rows('wn_r9_10_by_mobility_node_count.csv')
        self.assertEqual(len(r),9)
        self.assertEqual(sum(int(x['events']) for x in r),1114)
        self.assertEqual(sum(int(x['true_alerts']) for x in r),183)
        self.assertEqual(sum(int(x['false_alerts']) for x in r),88)
        self.assertEqual(sum(float(x['false_alerts_per_minute'])>2 for x in r),3)
        thirty=[x for x in r if int(x['n_nodes'])==30]
        self.assertEqual(sum(int(x['events']) for x in thirty),82)
        self.assertTrue(all(int(x['alerts'])==0 for x in thirty))
    def test_block_by_mobility_budgets(self):
        r=[x for x in rows('wn_r9_10_all_strata.csv') if x['scope']=='replicate_mobility']
        self.assertEqual(len(r),15)
        self.assertEqual(sum(float(x['false_alerts_per_minute'])>2 for x in r),6)
        self.assertEqual(sum(int(x['false_alerts']) for x in r),88)
    def test_lead_time(self):
        r=[x for x in rows('wn_r9_10_lead_distribution.csv') if x['method']=='TopoGNN']
        self.assertEqual(len(r),6)
        self.assertEqual([int(x['matched_alerts']) for x in r],[85,40,25,13,13,7])
    def test_hypothetical_action_latency(self):
        r=[x for x in rows('wn_r9_10_action_latency_sensitivity.csv') if x['method']=='TopoGNN']
        r=[x for x in r if float(x['assumed_effective_action_latency_seconds'])==.3]
        self.assertEqual(len(r),1)
        self.assertEqual(int(r[0]['potentially_timely_matched_alerts']),58)
