import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from unittest.mock import patch

from byewords.generate import (
    DEFAULT_DEMO_ENTRIES,
    benchmark_generation,
    generate_puzzle,
    generate_puzzle_candidates,
    generate_puzzle_cached,
    load_default_inputs,
)
from byewords.grid import distinct_entries, make_grid
from byewords.theme import lexicon_hash, load_word_vectors
from byewords.types import GenerateConfig, ProgressUpdate
from tests.fixtures import TEST_GRID_ROWS, TEST_LEXICON


def _write_vector_table(
    path: Path,
    lexicon: tuple[str, ...],
    vectors: dict[str, list[int]],
) -> None:
    dimensions = len(next(iter(vectors.values())))
    payload = {
        "version": 1,
        "source": "unit-test-vectors",
        "dimensions": dimensions,
        "lexicon_hash": lexicon_hash(lexicon),
        "quantization": {
            "scheme": "int8",
            "scale": 0.25,
        },
        "vectors": vectors,
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


class TestGenerate(unittest.TestCase):
    def test_load_default_inputs_backfills_missing_words_with_fallback_clues(self) -> None:
        with (
            patch("byewords.generate.load_word_list", return_value=("abase", "snail")),
            patch("byewords.generate.load_clue_bank", return_value={"snail": ("Handwritten clue",)}),
        ):
            lexicon_words, clue_bank = load_default_inputs()
            _, curated_only = load_default_inputs(include_fallback_clues=False)

        self.assertEqual(lexicon_words, ("abase", "snail"))
        self.assertEqual(tuple(clue_bank), ("abase", "snail"))
        self.assertEqual(clue_bank["snail"], ("Handwritten clue",))
        self.assertEqual(curated_only, {"snail": ("Handwritten clue",)})
        self.assertEqual(
            clue_bank["abase"][0],
            "Entry that starts with A and ends with E and has a repeated letter",
        )

    def test_benchmark_generation_reports_seeded_search_work(self) -> None:
        benchmark = benchmark_generation(
            seeds=("snail",),
            lexicon_words=TEST_LEXICON,
            clue_bank={"snail": ("Mollusk hauling its studio apartment",)},
        )

        self.assertEqual(benchmark.requested_seeds, ("snail",))
        self.assertEqual(benchmark.normalized_seeds, ("snail",))
        self.assertEqual(benchmark.available_seeds, ("snail",))
        self.assertEqual(benchmark.candidate_count, len(TEST_LEXICON))
        self.assertEqual(benchmark.candidate_window_sizes, (len(TEST_LEXICON),))
        self.assertEqual(len(benchmark.attempts), 1)
        self.assertEqual(benchmark.attempts[0].strategy, "seeded")
        self.assertEqual(benchmark.attempts[0].candidate_count, len(TEST_LEXICON))
        self.assertGreater(benchmark.attempts[0].stats.states_visited, 0)
        selected_grid = benchmark.selected_grid
        self.assertIsNotNone(selected_grid)
        if selected_grid is None:
            raise AssertionError("expected a selected grid")
        self.assertEqual(selected_grid.rows, TEST_LEXICON[:5])
        self.assertEqual(benchmark.selected_theme_words, ("snail",))
        self.assertEqual(benchmark.selected_theme_subset, ())
        self.assertEqual(benchmark.selected_theme_weakest_link, 0.0)
        self.assertFalse(benchmark.budget_exhausted)
        self.assertFalse(benchmark.used_budget_fallback)
        self.assertFalse(benchmark.used_demo_grid)

    def test_benchmark_generation_reports_generic_search_for_unknown_seed(self) -> None:
        benchmark = benchmark_generation(
            seeds=("beach",),
            lexicon_words=TEST_LEXICON,
            clue_bank={},
        )

        self.assertEqual(benchmark.requested_seeds, ("beach",))
        self.assertEqual(benchmark.normalized_seeds, ("beach",))
        self.assertEqual(benchmark.available_seeds, ())
        self.assertEqual(len(benchmark.attempts), 1)
        self.assertEqual(benchmark.attempts[0].strategy, "generic")
        self.assertFalse(benchmark.attempts[0].used_semantic_ordering)
        selected_grid = benchmark.selected_grid
        self.assertIsNotNone(selected_grid)
        if selected_grid is None:
            raise AssertionError("expected a selected grid")
        self.assertEqual(set(distinct_entries(selected_grid)), set(TEST_LEXICON))
        self.assertEqual(benchmark.selected_theme_words, ())

    def test_benchmark_generation_reports_demo_shortcut(self) -> None:
        benchmark = benchmark_generation(
            seeds=("ozone",),
            lexicon_words=DEFAULT_DEMO_ENTRIES,
            clue_bank={},
        )

        self.assertTrue(benchmark.used_demo_grid)
        self.assertEqual(benchmark.attempts, ())
        selected_grid = benchmark.selected_grid
        self.assertIsNotNone(selected_grid)
        if selected_grid is None:
            raise AssertionError("expected a selected grid")
        self.assertEqual(selected_grid.rows, DEFAULT_DEMO_ENTRIES[:5])
        self.assertEqual(benchmark.selected_theme_words, ("ozone",))
        self.assertEqual(benchmark.selected_theme_subset, ())
        self.assertFalse(benchmark.budget_exhausted)

    def test_generate_puzzle_builds_complete_puzzle(self) -> None:
        puzzle = generate_puzzle(
            seeds=("snail",),
            lexicon_words=TEST_LEXICON,
            clue_bank={"snail": ("Mollusk hauling its studio apartment",)},
        )

        self.assertEqual(puzzle.title, "SNAIL Mini")
        self.assertEqual(len(puzzle.across), 5)
        self.assertEqual(len(puzzle.down), 5)
        self.assertEqual(puzzle.grid.rows[3], "snail")
        self.assertEqual(puzzle.across[3].text, "Mollusk hauling its studio apartment")
        self.assertEqual(len(set(distinct_entries(puzzle.grid))), 10)

    def test_generate_puzzle_raises_on_impossible_input(self) -> None:
        with self.assertRaises(ValueError):
            generate_puzzle(
                seeds=("snail",),
                lexicon_words=("snail", "abase"),
                clue_bank={"snail": ("Mollusk hauling its studio apartment",)},
            )

    def test_generate_puzzle_accepts_empty_seed_list(self) -> None:
        puzzle = generate_puzzle(
            seeds=(),
            lexicon_words=TEST_LEXICON,
            clue_bank={},
        )

        self.assertEqual(puzzle.title, "BYEWORDS Mini")
        self.assertEqual(len(set(distinct_entries(puzzle.grid))), 10)

    def test_generate_puzzle_ignores_unknown_seed_words_when_generic_fill_exists(self) -> None:
        puzzle = generate_puzzle(
            seeds=("beach",),
            lexicon_words=TEST_LEXICON,
            clue_bank={},
        )

        self.assertEqual(puzzle.title, "BYEWORDS Mini")
        self.assertEqual(len(set(distinct_entries(puzzle.grid))), 10)

    def test_generate_puzzle_falls_back_to_generic_grid_when_requested_seed_cannot_fit(self) -> None:
        puzzle = generate_puzzle(
            seeds=("cable",),
            lexicon_words=TEST_LEXICON + ("cable", "agues", "buses", "leese", "esses"),
            clue_bank={},
        )

        self.assertEqual(puzzle.title, "BYEWORDS Mini")
        self.assertNotIn("cable", distinct_entries(puzzle.grid))

    def test_generate_puzzle_uses_demo_grid_when_seed_matches_demo_entry(self) -> None:
        puzzle = generate_puzzle(
            seeds=("ozone",),
            lexicon_words=DEFAULT_DEMO_ENTRIES,
            clue_bank={},
        )

        self.assertEqual(puzzle.title, "OZONE Mini")
        self.assertEqual(puzzle.grid.rows, DEFAULT_DEMO_ENTRIES[:5])
        self.assertIn("ozone", distinct_entries(puzzle.grid))

    def test_generate_puzzle_reports_failure_when_no_grid_can_be_built(self) -> None:
        with self.assertRaisesRegex(ValueError, "unable to generate a valid 5x5 puzzle"):
            generate_puzzle(
                seeds=("beach",),
                lexicon_words=("beach", "ozone", "liven", "inert", "verve", "ester"),
                clue_bank={},
            )

    def test_generate_puzzle_rejects_any_reused_word_in_final_puzzle(self) -> None:
        with self.assertRaises(ValueError):
            generate_puzzle(
                seeds=(),
                lexicon_words=("cable", "agues", "buses", "leese", "esses"),
                clue_bank={},
            )

    def test_generate_puzzle_prefers_seeded_fill_when_one_is_available(self) -> None:
        puzzle = generate_puzzle(
            seeds=("snail",),
            lexicon_words=TEST_LEXICON,
            clue_bank={"snail": ("Mollusk hauling its studio apartment",)},
        )

        self.assertEqual(puzzle.title, "SNAIL Mini")
        self.assertIn("snail", distinct_entries(puzzle.grid))

    def test_generate_puzzle_candidates_uses_semantic_grid_ranking_when_vectors_match(self) -> None:
        neutral_grid = make_grid(("abcde", "fghij", "klmno", "pqrst", "uvwxy"))
        themed_grid = make_grid(("zebra", "cumin", "vodka", "glyph", "fjord"))

        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "vectors.json"
            neutral_entries = distinct_entries(neutral_grid)
            themed_entries = distinct_entries(themed_grid)
            lexicon = ("beach",) + neutral_entries + themed_entries
            _write_vector_table(
                path,
                lexicon,
                {
                    "beach": [4, 0, 0, 0],
                    **{word: [0, 0, 4, 0] for word in neutral_entries},
                    **{
                        word: ([4, 1, 0, 0], [3, 3, 0, 0], [2, 4, 0, 0])[index % 3]
                        for index, word in enumerate(themed_entries)
                    },
                },
            )
            vectors = load_word_vectors(str(path))

        with patch(
            "byewords.generate._find_candidate_grids",
            return_value=((neutral_grid, themed_grid), ("beach",)),
        ), patch("byewords.generate._load_semantic_vectors", return_value=vectors), patch(
            "byewords.score.score_fill_quality",
            return_value=1.0,
        ), patch(
            "byewords.score.score_entry_diversity",
            return_value=1.0,
        ):
            puzzles = generate_puzzle_candidates(
                seeds=("beach",),
                lexicon_words=lexicon,
                clue_bank={},
            )

        self.assertEqual(puzzles[0].grid, themed_grid)

    def test_find_candidate_grids_passes_vector_backed_row_scores_into_seeded_search(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "vectors.json"
            _write_vector_table(
                path,
                TEST_LEXICON,
                {
                    "adieu": [1, 0, 0, 0],
                    "booed": [1, 1, 0, 0],
                    "antra": [1, 2, 0, 0],
                    "snail": [4, 0, 0, 0],
                    "eases": [2, 1, 0, 0],
                    "abase": [1, 0, 1, 0],
                    "donna": [1, 0, 2, 0],
                    "iotas": [0, 1, 1, 0],
                    "eerie": [0, 1, 2, 0],
                    "udals": [0, 0, 1, 1],
                },
            )
            vectors = load_word_vectors(str(path))

        observed_scores: list[dict[str, float] | None] = []

        def fake_seeded_search(**kwargs: object) -> tuple[object, ...]:
            row_scores = kwargs.get("row_scores")
            if isinstance(row_scores, dict) and all(
                isinstance(word, str) and isinstance(score, int | float)
                for word, score in row_scores.items()
            ):
                observed_scores.append(cast(dict[str, float], row_scores))
            else:
                observed_scores.append(None)
            return ()

        with patch("byewords.generate._load_semantic_vectors", return_value=vectors), patch(
            "byewords.generate._search_seeded_grids",
            side_effect=fake_seeded_search,
        ), patch(
            "byewords.generate.search_grids",
            return_value=(make_grid(TEST_GRID_ROWS),),
        ):
            puzzles = generate_puzzle_candidates(
                seeds=("snail",),
                lexicon_words=TEST_LEXICON,
                clue_bank={},
            )

        self.assertTrue(observed_scores)
        first_scores = observed_scores[0]
        self.assertIsNotNone(first_scores)
        if first_scores is None:
            raise AssertionError("expected semantic row scores")
        self.assertAlmostEqual(first_scores["snail"], 1.0, places=5)
        self.assertGreater(first_scores["antra"], first_scores["udals"])
        self.assertEqual(puzzles[0].grid.rows, TEST_GRID_ROWS)

    def test_generate_puzzle_reports_progress_updates(self) -> None:
        updates: list[ProgressUpdate] = []

        puzzle = generate_puzzle(
            seeds=("snail",),
            lexicon_words=TEST_LEXICON,
            clue_bank={"snail": ("Mollusk hauling its studio apartment",)},
            progress_callback=updates.append,
        )

        self.assertTrue(any(update.stage == "window" for update in updates))
        self.assertTrue(any(update.stage == "search" and update.partial_rows for update in updates))
        self.assertTrue(any(update.stage == "candidate_solution" for update in updates))
        runtime_reports = [update for update in updates if update.stage == "runtime_report"]
        self.assertEqual(len(runtime_reports), 1)
        runtime_report = runtime_reports[0].runtime_report
        self.assertIsNotNone(runtime_report)
        if runtime_report is None:
            raise AssertionError("expected a runtime report")
        self.assertEqual(runtime_report.requested_seeds, ("snail",))
        self.assertEqual(runtime_report.available_seeds, ("snail",))
        self.assertFalse(runtime_report.used_budget_fallback)
        self.assertIn("Runtime report:", runtime_reports[0].message)
        solution_rows = {
            update.partial_rows
            for update in updates
            if update.stage == "solution"
        }
        self.assertTrue(solution_rows)
        self.assertIn(puzzle.grid.rows, solution_rows)

    def test_generate_puzzle_falls_back_to_heuristic_ordering_after_budget_exhaustion(self) -> None:
        observed_row_scores: list[dict[str, float] | None] = []

        def fake_seeded_search(**kwargs: object) -> tuple[object, ...]:
            stats = kwargs.get("stats")
            row_scores = cast(dict[str, float] | None, kwargs.get("row_scores"))
            observed_row_scores.append(row_scores)
            if row_scores is not None:
                if hasattr(stats, "budget_exhausted"):
                    setattr(stats, "budget_exhausted", True)
                return ()
            return (make_grid(TEST_GRID_ROWS),)

        with patch("byewords.generate._load_semantic_vectors", return_value=None), patch(
            "byewords.generate._semantic_row_scores",
            return_value={word: float(index) for index, word in enumerate(TEST_LEXICON)},
        ), patch(
            "byewords.generate._search_seeded_grids",
            side_effect=fake_seeded_search,
        ):
            puzzle = generate_puzzle(
                seeds=("snail",),
                lexicon_words=TEST_LEXICON,
                clue_bank={"snail": ("Mollusk hauling its studio apartment",)},
                config=GenerateConfig(runtime_budget_ms=1),
            )

        self.assertGreaterEqual(len(observed_row_scores), 2)
        self.assertIsNotNone(observed_row_scores[0])
        self.assertIsNone(observed_row_scores[-1])
        self.assertEqual(puzzle.grid.rows, TEST_GRID_ROWS)

    def test_benchmark_generation_records_budget_fallback_attempts(self) -> None:
        def fake_seeded_search(**kwargs: object) -> tuple[object, ...]:
            stats = kwargs.get("stats")
            row_scores = cast(dict[str, float] | None, kwargs.get("row_scores"))
            if row_scores is not None:
                if hasattr(stats, "budget_exhausted"):
                    setattr(stats, "budget_exhausted", True)
                return ()
            return (make_grid(TEST_GRID_ROWS),)

        with patch("byewords.generate._load_semantic_vectors", return_value=None), patch(
            "byewords.generate._semantic_row_scores",
            return_value={word: float(index) for index, word in enumerate(TEST_LEXICON)},
        ), patch(
            "byewords.generate._search_seeded_grids",
            side_effect=fake_seeded_search,
        ):
            benchmark = benchmark_generation(
                seeds=("snail",),
                lexicon_words=TEST_LEXICON,
                clue_bank={},
                config=GenerateConfig(runtime_budget_ms=1),
            )

        self.assertTrue(benchmark.budget_exhausted)
        self.assertTrue(benchmark.used_budget_fallback)
        self.assertEqual(len(benchmark.attempts), 2)
        self.assertTrue(benchmark.attempts[0].used_semantic_ordering)
        self.assertFalse(benchmark.attempts[1].used_semantic_ordering)
        self.assertIsNotNone(benchmark.selected_grid)
        if benchmark.selected_grid is None:
            raise AssertionError("expected a selected grid")
        self.assertEqual(benchmark.selected_grid.rows, TEST_GRID_ROWS)

    def test_benchmark_generation_records_heuristic_baseline_for_semantic_attempts(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "vectors.json"
            _write_vector_table(
                path,
                TEST_LEXICON,
                {
                    "adieu": [5, 1, 0, 0],
                    "booed": [5, 2, 0, 0],
                    "antra": [4, 3, 0, 0],
                    "snail": [6, 0, 0, 0],
                    "eases": [4, 4, 0, 0],
                    "abase": [0, 0, 4, 0],
                    "donna": [4, 2, 0, 0],
                    "iotas": [0, 0, 3, 1],
                    "eerie": [0, 0, 2, 2],
                    "udals": [0, 0, 1, 3],
                },
            )
            vectors = load_word_vectors(str(path))

        with patch("byewords.generate._load_semantic_vectors", return_value=vectors):
            benchmark = benchmark_generation(
                seeds=("snail",),
                lexicon_words=TEST_LEXICON,
                clue_bank={},
            )

        self.assertEqual(len(benchmark.attempts), 1)
        attempt = benchmark.attempts[0]
        self.assertTrue(attempt.used_semantic_ordering)
        self.assertIsNotNone(attempt.heuristic_baseline)
        if attempt.heuristic_baseline is None:
            raise AssertionError("expected heuristic baseline")
        self.assertGreater(attempt.stats.semantic_reranks, 0)
        self.assertGreater(attempt.stats.novelty_penalties_applied, 0)
        self.assertGreater(attempt.heuristic_baseline.solutions_found, 0)
        self.assertEqual(attempt.heuristic_baseline.stats.semantic_reranks, 0)
        self.assertEqual(attempt.heuristic_baseline.stats.novelty_penalties_applied, 0)

    def test_generate_puzzle_cached_reuses_saved_puzzle(self) -> None:
        with TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            seeds = ("snail",)
            clue_bank: dict[str, tuple[str, ...]] = {
                "snail": ("Mollusk hauling its studio apartment",),
            }

            first = generate_puzzle_cached(
                seeds=seeds,
                lexicon_words=TEST_LEXICON,
                clue_bank=clue_bank,
                cache_dir=cache_dir,
            )

            with patch("byewords.generate.generate_puzzle", side_effect=AssertionError("cache miss")):
                second = generate_puzzle_cached(
                    seeds=seeds,
                    lexicon_words=TEST_LEXICON,
                    clue_bank=clue_bank,
                    cache_dir=cache_dir,
                )

            self.assertEqual(first, second)
            self.assertEqual(len(tuple(cache_dir.glob("*.json"))), 1)

    def test_generate_puzzle_cached_reports_cache_hits(self) -> None:
        with TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            updates: list[ProgressUpdate] = []
            seeds = ("snail",)
            clue_bank: dict[str, tuple[str, ...]] = {
                "snail": ("Mollusk hauling its studio apartment",),
            }

            generate_puzzle_cached(
                seeds=seeds,
                lexicon_words=TEST_LEXICON,
                clue_bank=clue_bank,
                cache_dir=cache_dir,
            )

            with patch("byewords.generate.generate_puzzle", side_effect=AssertionError("cache miss")):
                cached = generate_puzzle_cached(
                    seeds=seeds,
                    lexicon_words=TEST_LEXICON,
                    clue_bank=clue_bank,
                    cache_dir=cache_dir,
                    progress_callback=updates.append,
                )

            self.assertEqual(len(updates), 1)
            self.assertEqual(updates[0].stage, "cache_hit")
            self.assertEqual(updates[0].partial_rows, cached.grid.rows)

    def test_generate_puzzle_cached_invalidates_cache_when_data_changes(self) -> None:
        with TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            seeds = ("snail",)
            first_clue_bank: dict[str, tuple[str, ...]] = {
                "snail": ("First clue",),
            }
            second_clue_bank: dict[str, tuple[str, ...]] = {
                "snail": ("Second clue",),
            }

            generate_puzzle_cached(
                seeds=seeds,
                lexicon_words=TEST_LEXICON,
                clue_bank=first_clue_bank,
                cache_dir=cache_dir,
            )

            second = generate_puzzle_cached(
                seeds=seeds,
                lexicon_words=TEST_LEXICON,
                clue_bank=second_clue_bank,
                cache_dir=cache_dir,
            )

            self.assertEqual(second.across[3].text, "Second clue")
            self.assertEqual(len(tuple(cache_dir.glob("*.json"))), 2)


if __name__ == "__main__":
    unittest.main()
