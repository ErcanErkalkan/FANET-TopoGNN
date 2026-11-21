"""Verify the current paper figures against retained frozen derived results."""
from pathlib import Path
import csv
import unittest

ROOT=Path(__file__).resolve().parents[2]
F=ROOT/'paper/manuscript/figures'
D=ROOT/'data_derived'
T=(ROOT/'paper/manuscript/main.tex').read_text(encoding='utf-8')

def read_csv(name):
    with (D/name).open(newline='',encoding='utf-8') as f:
        return list(csv.DictReader(f))

class CurrentFigureContract(unittest.TestCase):
    def test_all_five_figures_exist_and_have_tex_captions(self):
        names=['fig1_warning_pipeline','fig2_target_mismatch_schematic',
               'fig3_mobility_node_heatmaps','fig4_block_mobility_budget',
               'fig5_lead_action_latency']
        self.assertEqual(len(list(F.glob('fig*.pdf'))),5)
        for i,name in enumerate(names,1):
            self.assertTrue((F/(name+'.pdf')).stat().st_size>2000)
            self.assertIn(name+'.pdf',T)
            self.assertIn(r'\label{fig:',T)
    def test_node_mobility_denominators(self):
        r=read_csv('wn_r9_10_by_mobility_node_count.csv')
        self.assertEqual(len(r),9)
        self.assertEqual(sum(int(x['events']) for x in r),1114)
        self.assertEqual(sum(int(x['true_alerts']) for x in r),183)
        self.assertEqual(sum(int(x['false_alerts']) for x in r),88)
        self.assertEqual(sum(float(x['false_alerts_per_minute'])>2 for x in r),3)
        thirty=[x for x in r if int(x['n_nodes'])==30]
        self.assertEqual(sum(int(x['events']) for x in thirty),82)
        self.assertTrue(all(int(x['alerts'])==0 for x in thirty))
    def test_block_mobility_budget_exceedances(self):
        r=[x for x in read_csv('wn_r9_10_all_strata.csv') if x['scope']=='replicate_mobility']
        self.assertEqual(len(r),15)
        self.assertEqual(sum(float(x['false_alerts_per_minute'])>2 for x in r),6)
        self.assertEqual(sum(int(x['false_alerts']) for x in r),88)
    def test_lead_time_distribution(self):
        r=[x for x in read_csv('wn_r9_10_lead_distribution.csv') if x['method']=='TopoGNN']
        self.assertEqual(len(r),6)
        self.assertEqual(sum(int(x['matched_alerts']) for x in r),183)
        self.assertEqual([int(x['matched_alerts']) for x in r],[85,40,25,13,13,7])
    def test_latency_interpretation(self):
        r=[x for x in read_csv('wn_r9_10_action_latency_sensitivity.csv') if x['method']=='TopoGNN']
        pt=[x for x in r if float(x['assumed_effective_action_latency_seconds'])==.3]
        self.assertEqual(len(pt),1)
        self.assertEqual(int(pt[0]['potentially_timely_matched_alerts']),58)
        self.assertIn('hypothetical',T.lower())
    def test_figure_provenance_labels(self):
        src=(ROOT/'paper/make_figures.py').read_text(encoding='utf-8')
        self.assertIn('Schematic only: not a recorded simulation run',src)
        self.assertIn('No controller or mitigation was simulated',src)
        self.assertIn('Evaluation target: onset within six frames',src)

if __name__=='__main__':
    unittest.main()
