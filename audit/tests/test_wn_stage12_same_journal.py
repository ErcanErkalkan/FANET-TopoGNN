"""Step 12 literature integrity: claims, citation paths, and no false cross-study ranking."""
from pathlib import Path
import re
import unittest

ROOT=Path(__file__).resolve().parents[2]
MAIN=(ROOT/'paper/manuscript/main.tex').read_text(encoding='utf-8')
BIB=(ROOT/'paper/manuscript/references.bib').read_text(encoding='utf-8')
SECTION=MAIN.split(r'\section{Related work and evaluation gap}',1)[1].split(r'\section{Experimental design',1)[0]

class SameJournalComparison(unittest.TestCase):
    def test_latex_table_present_and_uniquely_labeled(self):
        self.assertEqual(MAIN.count(r'\label{tab:journal_comparison}'),1)
        self.assertIn(r'Table~\ref{tab:journal_comparison}',SECTION)
        self.assertIn(r'\begin{table}',SECTION)
    def test_three_journal_studies_cited(self):
        for key in ('cocoma2024','karegar2025uav','chin2014oulpt'):
            with self.subTest(key=key):
                self.assertIn(r'\cite{'+key+'}',SECTION)
                self.assertIn('@article{'+key+',',BIB)
        self.assertEqual(SECTION.count('This study. Whole-FANET onset warning audit'),1)
    def test_pubisher_doi_provenance(self):
        for doi in ('10.1007/s11276-022-03031-8','10.1007/s11276-024-03744-y','10.1007/s11276-013-0610-0'):
            with self.subTest(doi=doi):self.assertIn(doi,BIB)
    def test_metric_incomparability_and_no_routing_claim(self):
        self.assertIn('a common-protocol performance ranking',SECTION)
        self.assertIn('not whole-graph false alarms/min',SECTION)
        self.assertIn('reproduced here',SECTION)
        self.assertIn('No routing intervention',SECTION)
    def test_citation_cross_reference(self):
        cited=set()
        for p in re.findall(r'\\cite\{([^}]*)\}',MAIN):
            cited.update(k.strip() for k in p.split(','))
        defined=set(re.findall(r'@[A-Za-z]+\{([^,]+),',BIB))
        self.assertFalse(cited-defined)
    def test_supplement_references_updated_table_number(self):
        esm=(ROOT/"paper/manuscript/ESM_1.tex").read_text(encoding="utf-8")
        self.assertIn("Table 5 of the manuscript",esm)
        self.assertNotIn("Table 4 of the manuscript",esm)

if __name__=='__main__':unittest.main()
