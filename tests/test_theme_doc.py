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
        self.assertIn("Stage 1 is now complete.", text)
        self.assertIn("Stage 2 is now fully closed.", text)
        self.assertIn("Stage 3 is now fully closed.", text)
        self.assertIn("Stage 4 is now fully closed.", text)
        self.assertIn("Stage 5 has deterministic runtime budget enforcement and is now fully closed.", text)
        self.assertIn("Stage 6 is now fully closed.", text)
        self.assertIn("Stage 7 is now fully closed.", text)
        self.assertIn("the representative easy, medium, and hard seed corpus now lives in code via `THEME_BENCHMARK_SEEDS`", text)
        self.assertIn(
            "the lightweight manual-review corpus for answer/theme plausibility now lives in code via `THEME_MANUAL_REVIEW_CASES`",
            text,
        )
        self.assertIn(
            "the representative retrieval-quality review corpus now lives in code via `THEME_RETRIEVAL_REVIEW_CASES`",
            text,
        )
        self.assertIn(
            "the intrusion-style review corpus now lives in code via `THEME_INTRUSION_REVIEW_CASES`",
            text,
        )
        self.assertIn(
            "cosine and rank-overlap retrieval reports can now be compared deterministically against the review corpus without changing runtime ranking",
            text,
        )
        self.assertIn(
            "intrusion-review helpers can now deterministically verify that the theme scorer rejects unrelated answers from a fixed-size themed subset",
            text,
        )
        self.assertIn(
            "viable-row ordering can now apply lightweight MMR-style novelty penalties against provisional theme-bearing rows without changing legality pruning",
            text,
        )
        self.assertIn(
            "benchmark search attempts can now capture heuristic-baseline counter snapshots for deterministic before/after search comparisons",
            text,
        )
        self.assertIn(
            "seeded semantic search now enforces a deterministic runtime budget and falls back to heuristic row ordering when it expires",
            text,
        )
        self.assertIn(
            "benchmark reports now surface budget exhaustion, heuristic fallback, and selected theme-subset telemetry",
            text,
        )
        self.assertIn(
            "seeded generation now emits a per-request runtime report through progress updates, and the CLI now surfaces that report after puzzle generation",
            text,
        )
        self.assertIn("the offline cache path now performs real per-seed puzzle generation", text)
        self.assertIn("the offline cache now retains a bounded top-N answer-only candidate set per seed", text)
        self.assertIn("offline answer-only curation now ranks records without clue-score influence", text)
        self.assertIn(
            "the offline cache now uses process-based worker execution for default lexicon-wide generation while preserving an in-process fallback for tests and patched generators",
            text,
        )
        self.assertIn(
            "the offline cache now refreshes clues only for the deterministic top-100 answer-only slice, validates those refreshed clue sets, and stores clue-stage ranking metadata beside each selected record",
            text,
        )
        self.assertIn(
            "the top-100 clue stage now uses `clue_bank.json` first and only attempts Groq regeneration for selected answers that still lack non-generic clues",
            text,
        )
        self.assertIn(
            "final offline ranking can now include clue-quality signal only after top-100 clue refresh, while preserving answer-only selection for the initial slice",
            text,
        )
        self.assertIn(
            "clue-aware reranking can now be reviewed deterministically against expected answer-only and clue-stage winners for a small editorial corpus",
            text,
        )
        self.assertIn(
            "`theme_index_builder.py` now supports vector building, lexicon-wide cache generation, and deterministic retrieval and intrusion review reports for offline evaluation",
            text,
        )
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
        self.assertIn("the offline cache now refreshes clues only for the deterministic global top-100 answer-only slice", text)
        self.assertIn(
            "refreshed clue sets are now validated for non-generic multi-word clue text before clue-stage ranking metadata is persisted",
            text,
        )
        self.assertIn(
            "clue-stage reranking now happens only within the selected top-100 slice, so clue quality cannot pull weaker answer-only records into the curated set",
            text,
        )
        self.assertIn(
            "Groq regeneration is now attempted only for selected answers that still lack non-generic clue-bank coverage, preserving offline-first batch builds when API credentials are absent",
            text,
        )
        self.assertIn(
            "clue-stage review helpers can now compare answer-only winners against clue-stage winners for a small reviewed corpus without changing offline ranking inputs",
            text,
        )
        self.assertIn("Stage 1 is now fully closed:", text)
        self.assertIn("Stage 2 is now fully closed:", text)
        self.assertIn("Stage 4 is now fully closed:", text)
        self.assertIn(
            "cosine and rank-overlap can now be compared deterministically through offline retrieval-review helpers",
            text,
        )
        self.assertIn(
            "the retrieval-quality review corpus is checked against bundled lexicon coverage so ranking quality is measured rather than assumed",
            text,
        )
        self.assertIn(
            "seeded and generic runtime search now order already-legal viable rows by semantic score first and branching score second",
            text,
        )
        self.assertIn(
            "completed-grid scoring can now add vector-backed `theme_score` during runtime ranking when the bundled table matches the active lexicon",
            text,
        )
        self.assertIn(
            "completed-grid ranking now hard-rejects weak fills and weak semantic subsets while surfacing theme-bearing subset metadata in `CandidateGrid`",
            text,
        )
        self.assertIn("the offline cache now stores real per-seed winners together with answer-only metadata", text)
        self.assertIn(
            "the offline cache can now retain multiple answer-only candidates per seed before later top-100 clue selection",
            text,
        )
        self.assertIn("answer-only ranking now ignores clue score during offline curation", text)
        self.assertIn(
            "process-based execution and deterministic intrusion evaluation now exist for the default offline batch path",
            text,
        )
        self.assertIn(
            "store clue-stage validation and rerank metadata for the selected top-100 slice without changing answer-only curation inputs",
            text,
        )
        self.assertNotIn("It still should:", text)
        self.assertIn("write lexicon-wide cached puzzle records to `src/byewords/data/puzzles.json`", text)
        self.assertIn("with no seeds: build or refresh the offline `puzzles.json` cache", text)
        self.assertIn(
            "clue-aware reranking is now measurable enough to decide whether the reviewed clue corpus should expand",
            text,
        )
        self.assertIn("Decision gate:", text)
        self.assertIn(
            "explicit fill-quality gates and weakest-link thresholds now reject weak completed grids during runtime ranking",
            text,
        )
        self.assertIn(
            "theme-bearing subset metadata is now surfaced on ranked runtime candidates and reused by the offline puzzle store",
            text,
        )
        self.assertIn("semantic row ordering is now wired into the runtime search hot path without changing legality pruning", text)
        self.assertIn(
            "semantic vector loading, viable-row ordering, and completed-grid reranking are now wired into seeded selection when the bundled table matches the active lexicon",
            text,
        )
        self.assertIn(
            "lightweight MMR-style novelty penalties now down-rank semantically redundant provisional theme rows during viable-row ordering",
            text,
        )
        self.assertIn(
            "deterministic benchmark attempts now include heuristic-baseline counter snapshots for before/after search comparisons",
            text,
        )
        self.assertIn(
            "benchmark reports now surface budget exhaustion, fallback usage, and selected theme-subset coherence telemetry",
            text,
        )
        self.assertIn(
            "seeded generation now emits a per-request runtime report through progress updates, and the CLI surfaces a stable post-build runtime summary for interactive and non-interactive runs",
            text,
        )
        self.assertIn("## Appendix: Research papers", text)
        self.assertIn("https://aclanthology.org/N10-1012/", text)
        self.assertIn("https://aclanthology.org/P18-2088/", text)
        self.assertIn("https://arxiv.org/abs/2308.04688", text)


if __name__ == "__main__":
    unittest.main()
