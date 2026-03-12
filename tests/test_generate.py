import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from byewords.generate import (
    DEFAULT_DEMO_ENTRIES,
    benchmark_generation,
    generate_puzzle,
    generate_puzzle_candidates,
    generate_puzzle_cached,
)
from byewords.grid import distinct_entries, make_grid
from byewords.theme import lexicon_hash, load_word_vectors
from byewords.types import ProgressUpdate
from tests.fixtures import TEST_LEXICON


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
        solution_rows = {
            update.partial_rows
            for update in updates
            if update.stage == "solution"
        }
        self.assertTrue(solution_rows)
        self.assertIn(puzzle.grid.rows, solution_rows)

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
