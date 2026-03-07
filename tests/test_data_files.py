import unittest

from byewords.generate import generate_puzzle, load_default_inputs


class TestBundledData(unittest.TestCase):
    def test_default_data_is_large_and_consistent(self) -> None:
        lexicon_words, related_map, clue_bank = load_default_inputs()
        lexicon_set = set(lexicon_words)

        self.assertGreaterEqual(len(lexicon_words), 5000)
        self.assertGreaterEqual(len(clue_bank), 150)
        self.assertGreaterEqual(len(related_map), 20)
        self.assertTrue(set(clue_bank).issubset(lexicon_set))

        for related_words in related_map.values():
            self.assertTrue(set(related_words).issubset(lexicon_set))

    def test_default_data_includes_san_francisco_theme_cluster(self) -> None:
        lexicon_words, related_map, clue_bank = load_default_inputs()
        lexicon_set = set(lexicon_words)

        expected_words = {"buses", "cable", "ikeas", "lurie", "mayor", "parks", "piers"}
        self.assertTrue(expected_words.issubset(lexicon_set))
        self.assertTrue(expected_words.issubset(set(clue_bank)))
        self.assertEqual(clue_bank["lurie"][0], "Daniel who got the San Francisco mayor keys in 2022")
        self.assertEqual(clue_bank["buses"][0], "Public transit vehicles")
        self.assertEqual(clue_bank["ikeas"][0], "Bay Area furniture mazes with cinnamon buns")
        self.assertTrue({"lurie", "mayor", "buses", "cable", "ocean", "parks", "piers"}.issubset(set(related_map["lurie"])))
        self.assertTrue({"ikeas", "civic", "metro", "parks", "plaza", "route", "urban"}.issubset(set(related_map["ikeas"])))

    def test_default_data_includes_current_event_flavored_clues(self) -> None:
        _, _, clue_bank = load_default_inputs()

        self.assertIn("oscar", clue_bank)
        self.assertIn("trade", clue_bank)
        self.assertIn("March", clue_bank["oscar"][0])
        self.assertIn("tariff", clue_bank["trade"][0].lower())

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
