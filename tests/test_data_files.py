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

    def test_default_snail_puzzle_uses_bundled_clues(self) -> None:
        lexicon_words, related_map, clue_bank = load_default_inputs()
        puzzle = generate_puzzle(("snail",), lexicon_words, related_map, clue_bank)

        self.assertEqual(puzzle.title, "SNAIL Mini")
        self.assertEqual(len(puzzle.across), 5)
        self.assertEqual(len(puzzle.down), 5)

        for clue in puzzle.across + puzzle.down:
            self.assertIn(clue.answer.lower(), clue_bank)
            self.assertEqual(clue.text, clue_bank[clue.answer.lower()][0])

    def test_default_data_rejects_cable_cluster_seeds_that_only_fit_with_duplicates(self) -> None:
        lexicon_words, related_map, clue_bank = load_default_inputs()

        for seed in ("cable", "lurie"):
            with self.subTest(seed=seed):
                with self.assertRaises(ValueError):
                    generate_puzzle((seed,), lexicon_words, related_map, clue_bank)


if __name__ == "__main__":
    unittest.main()
