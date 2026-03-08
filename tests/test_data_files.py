import unittest

from byewords.generate import build_demo_puzzle, generate_puzzle, load_default_inputs
from byewords.grid import distinct_entries


class TestBundledData(unittest.TestCase):
    def test_default_data_is_large_and_consistent(self) -> None:
        lexicon_words, related_map, clue_bank = load_default_inputs()
        lexicon_set = set(lexicon_words)

        self.assertGreaterEqual(len(lexicon_words), 5000)
        self.assertGreaterEqual(len(related_map), 20)
        self.assertTrue(set(clue_bank).issubset(lexicon_set))
        self.assertTrue(all(clues and all(clue.strip() for clue in clues) for clues in clue_bank.values()))

        for related_words in related_map.values():
            self.assertTrue(set(related_words).issubset(lexicon_set))

    def test_default_data_keeps_common_theme_clusters_and_excludes_junk(self) -> None:
        lexicon_words, related_map, clue_bank = load_default_inputs()
        lexicon_set = set(lexicon_words)

        expected_words = {"beach", "music", "ocean", "piano", "tempo", "waves", "wharf"}
        self.assertTrue(expected_words.issubset(lexicon_set))
        self.assertTrue({"beach", "ocean", "shore", "waves", "wharf"}.issubset(set(related_map["beach"])))
        self.assertTrue({"music", "album", "opera", "piano", "tempo"}.issubset(set(related_map["music"])))

        for removed_word in ("aahed", "antra", "ikeas", "lurie", "udals"):
            self.assertNotIn(removed_word, lexicon_set)
            self.assertNotIn(removed_word, clue_bank)

    def test_default_demo_puzzle_uses_fallback_clues_when_cache_is_empty(self) -> None:
        _, related_map, clue_bank = load_default_inputs()
        puzzle = build_demo_puzzle(related_map, clue_bank)

        self.assertEqual(puzzle.title, "WATER Mini")
        self.assertEqual(len(puzzle.across), 5)
        self.assertEqual(len(puzzle.down), 5)

        for clue in puzzle.across:
            self.assertTrue(clue.text.strip())

        for clue in puzzle.down:
            self.assertTrue(clue.text.strip())

    def test_readme_seed_example_generates_a_puzzle(self) -> None:
        lexicon_words, related_map, clue_bank = load_default_inputs()
        seeds = ("ozone", "liven", "inert", "verve", "ester")

        puzzle = generate_puzzle(seeds, lexicon_words, related_map, clue_bank)

        self.assertEqual(puzzle.title, "OZONE Mini")
        self.assertEqual(len(puzzle.across), 5)
        self.assertEqual(len(puzzle.down), 5)
        self.assertTrue(set(seeds).issubset(set(distinct_entries(puzzle.grid))))
        self.assertEqual(len(set(distinct_entries(puzzle.grid))), 10)

if __name__ == "__main__":
    unittest.main()
