import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from byewords.generate import generate_puzzle, generate_puzzle_cached
from byewords.grid import distinct_entries
from tests.fixtures import TEST_LEXICON


class TestGenerate(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
