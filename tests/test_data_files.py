import unittest

from byewords.generate import generate_puzzle, load_default_inputs


class TestBundledData(unittest.TestCase):
    def test_default_data_is_large_and_consistent(self) -> None:
        lexicon_words, related_map, clue_bank = load_default_inputs()
        lexicon_set = set(lexicon_words)

        self.assertGreaterEqual(len(lexicon_words), 150)
        self.assertGreaterEqual(len(clue_bank), 150)
        self.assertGreaterEqual(len(related_map), 20)
        self.assertEqual(set(clue_bank), lexicon_set)

        for related_words in related_map.values():
            self.assertTrue(set(related_words).issubset(lexicon_set))

    def test_default_snail_puzzle_uses_bundled_clues(self) -> None:
        lexicon_words, related_map, clue_bank = load_default_inputs()
        puzzle = generate_puzzle(("snail",), lexicon_words, related_map, clue_bank)

        self.assertEqual(puzzle.title, "SNAIL Mini")
        self.assertEqual(len(puzzle.across), 5)
        self.assertEqual(len(puzzle.down), 5)

        for clue in puzzle.across + puzzle.down:
            self.assertIn(clue.answer.lower(), clue_bank)
            self.assertEqual(clue.text, clue_bank[clue.answer.lower()][0])


if __name__ == "__main__":
    unittest.main()
