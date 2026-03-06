import unittest

from byewords.generate import generate_puzzle, load_default_inputs
from byewords.grid import distinct_entries


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
        self.assertEqual(clue_bank["lurie"][0], "Daniel, San Francisco's 46th mayor")
        self.assertEqual(clue_bank["buses"][0], "Many Muni vehicles")
        self.assertEqual(clue_bank["ikeas"][0], "San Francisco and Emeryville furniture stores")
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

    def test_default_data_supports_cable_cluster_seeds(self) -> None:
        lexicon_words, related_map, clue_bank = load_default_inputs()

        for seed, title in (("cable", "CABLE Mini"), ("lurie", "LURIE Mini")):
            with self.subTest(seed=seed):
                puzzle = generate_puzzle((seed,), lexicon_words, related_map, clue_bank)

                self.assertEqual(puzzle.title, title)
                self.assertEqual(puzzle.grid.rows, ("cable", "agues", "buses", "leese", "esses"))
                self.assertGreaterEqual(
                    sum(entry in set(puzzle.theme_words) for entry in distinct_entries(puzzle.grid)),
                    4,
                )
                self.assertEqual(puzzle.across[1].text, "Feverish chills")
                self.assertEqual(puzzle.across[3].text, "Archaic verb meaning lose")
                self.assertEqual(puzzle.across[4].text, "Plural of the letter S")


if __name__ == "__main__":
    unittest.main()
