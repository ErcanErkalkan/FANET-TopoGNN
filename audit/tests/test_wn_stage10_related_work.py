"""Bibliography and disclosure regression for the first journal review fix."""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "paper/manuscript/main.tex"
BIB = ROOT / "paper/manuscript/references.bib"

class LiteratureCrosscheck(unittest.TestCase):
    def test_ou_lpt_cited_in_related_work(self):
        doc = TEX.read_text(encoding="utf-8")
        related = doc.split(r"\section{Related work and evaluation gap}", 1)[1].split(r"\section{Experimental design", 1)[0]
        self.assertIn(r"\cite{chin2014oulpt}", related)
        self.assertIn("wireless-mesh links", related)

    def test_bibliography_entry_metadata(self):
        bib = BIB.read_text(encoding="utf-8")
        self.assertEqual(bib.count("@article{chin2014oulpt,"), 1)
        for value in ["Chin, Eric", "Chieng, David", "Teh, Victor", "Natkaniec, Marek", "Loziak, Krzysztof", "Gozdecki, Janusz", "year={2014}", "volume={20}", "pages={379--396}", "doi={10.1007/s11276-013-0610-0}"]:
            with self.subTest(value=value):
                self.assertIn(value, bib)

    def test_no_invalid_head_to_head_claim(self):
        related = TEX.read_text(encoding="utf-8").split(r"\section{Related work and evaluation gap}", 1)[1].split(r"\section{Experimental design", 1)[0]
        self.assertIn("cannot be compared", related)
        self.assertIn("We do not implement or benchmark", related)
        self.assertIn("claim no comparative improvement", related)

    def test_all_citation_keys_resolve(self):
        doc = TEX.read_text(encoding="utf-8")
        bib = BIB.read_text(encoding="utf-8")
        keys=set(re.findall(r"@[A-Za-z]+\{([^,]+),", bib))
        requested = set()
        for cite in re.findall(r"\\cite(?:\[[^\]]*\])*\{([^}]*)\}", doc):
            requested.update(k.strip() for k in cite.split(","))
        self.assertFalse(requested-keys, f"Missing BibTeX keys: {requested-keys}")

if __name__ == "__main__": unittest.main()
