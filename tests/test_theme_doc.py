from pathlib import Path
import unittest


THEME_DOC = Path(__file__).resolve().parents[1] / "docs" / "theme.md"


class TestThemeDoc(unittest.TestCase):
    def test_theme_doc_requires_full_lexicon_vector_ranking(self) -> None:
        text = THEME_DOC.read_text(encoding="utf-8")

        self.assertIn("# Semantic Theme Generation Plan", text)
        self.assertIn("seeded runtime generation should still target under 1 second wall clock", text)
        self.assertIn("This plan is based on the experiments performed so far.", text)
        self.assertIn("A direct seeded search experiment for `doggy`", text)
        self.assertIn("src/byewords/data/word_vectors.json", text)
        self.assertIn("store one semantic vector for every bundled word", text)
        self.assertIn("Scoring every word against one seed is cheap:", text)
        self.assertIn("1,485,000", text)
        self.assertIn("semantics decides ordering", text)
        self.assertIn("the search index decides legality", text)
        self.assertIn("fill quality is a hard requirement", text)
        self.assertIn("entry uniqueness is a hard requirement", text)
        self.assertIn("## Findings from literature", text)
        self.assertIn("weakest-link coherence", text)
        self.assertIn("theme-bearing answers", text)
        self.assertIn("bridge fill", text)
        self.assertIn("Maximal Marginal Relevance", text)
        self.assertIn("intrusion-style evaluation", text)
        self.assertIn("first use `clue_bank.json` while generating and storing candidate puzzles for all bundled seed words", text)
        self.assertIn("after the full seed corpus has been generated, choose the best 100 puzzles based only on their answers", text)
        self.assertIn("regenerate clues for those 100 puzzles", text)
        self.assertIn("hard requirements: fill quality and uniqueness", text)
        self.assertIn("answer/theme coherence", text)
        self.assertIn("clue quality and clue/theme coherence", text)
        self.assertIn("`<= 750 ms` for semantic ranking plus search work", text)
        self.assertIn("## Expected performance and optimizations", text)
        self.assertIn("Expected steady-state runtime for one seeded request:", text)
        self.assertIn("cache the loaded vector table", text)
        self.assertIn("precompute seed-to-word scores once per request", text)
        self.assertIn("keep semantic ordering as a sort key on already-legal viable rows", text)
        self.assertIn("## Progressive implementation plan", text)
        self.assertIn("### Stage 1. Baseline measurement", text)
        self.assertIn("### Stage 7. Top-100 clue stage", text)
        self.assertIn("write lexicon-wide cached puzzle records to `src/byewords/data/puzzles.json`", text)
        self.assertIn("with no seeds: build or refresh the offline `puzzles.json` cache", text)
        self.assertIn("Decision gate:", text)
        self.assertIn("## Appendix: Research papers", text)
        self.assertIn("https://aclanthology.org/N10-1012/", text)
        self.assertIn("https://aclanthology.org/P18-2088/", text)
        self.assertIn("https://arxiv.org/abs/2308.04688", text)


if __name__ == "__main__":
    unittest.main()
