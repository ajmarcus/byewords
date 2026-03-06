import unittest

from byewords.generate import generate_puzzle
from byewords.types import GenerateConfig
from tests.fixtures import TEST_LEXICON


class TestGenerate(unittest.TestCase):
    def test_generate_puzzle_builds_complete_puzzle(self) -> None:
        puzzle = generate_puzzle(
            seeds=("snail",),
            lexicon_words=TEST_LEXICON,
            related_map={"snail": ("snail", "adieu", "booed", "antra", "eases")},
            clue_bank={"snail": ("Garden crawler with a spiral shell",)},
        )

        self.assertEqual(puzzle.title, "SNAIL Mini")
        self.assertEqual(len(puzzle.across), 5)
        self.assertEqual(len(puzzle.down), 5)
        self.assertEqual(puzzle.grid.rows[3], "snail")
        self.assertEqual(puzzle.across[3].text, "Garden crawler with a spiral shell")

    def test_generate_puzzle_raises_on_impossible_input(self) -> None:
        with self.assertRaises(ValueError):
            generate_puzzle(
                seeds=("snail",),
                lexicon_words=("snail", "abase"),
                related_map={"snail": ("snail",)},
                clue_bank={"snail": ("Garden crawler with a spiral shell",)},
            )

    def test_generate_puzzle_rejects_seeds_without_enough_theme_words(self) -> None:
        with self.assertRaises(ValueError):
            generate_puzzle(
                seeds=("xxxxx",),
                lexicon_words=TEST_LEXICON,
                related_map={},
                clue_bank={},
            )

    def test_generate_puzzle_enforces_minimum_theme_entries(self) -> None:
        with self.assertRaises(ValueError):
            generate_puzzle(
                seeds=("snail",),
                lexicon_words=TEST_LEXICON,
                related_map={"snail": ("snail", "adieu", "booed", "antra", "eases")},
                clue_bank={"snail": ("Garden crawler with a spiral shell",)},
                config=GenerateConfig(min_theme_words=6),
            )


if __name__ == "__main__":
    unittest.main()
