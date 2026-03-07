import unittest

from byewords.clue_bank import preferred_clue_words
from byewords.generate import build_demo_puzzle, load_default_inputs


class TestBundledData(unittest.TestCase):
    def test_default_data_is_large_and_consistent(self) -> None:
        lexicon_words, related_map, clue_bank = load_default_inputs()
        lexicon_set = set(lexicon_words)

        self.assertGreaterEqual(len(lexicon_words), 5000)
        self.assertEqual(len(clue_bank), len(lexicon_words))
        self.assertGreaterEqual(len(related_map), 20)
        self.assertTrue(set(clue_bank).issubset(lexicon_set))
        self.assertTrue(all(len(clues) == 1 and clues[0].strip() for clues in clue_bank.values()))

        for related_words in related_map.values():
            self.assertTrue(set(related_words).issubset(lexicon_set))

    def test_default_data_keeps_common_theme_clusters_and_excludes_junk(self) -> None:
        lexicon_words, related_map, clue_bank = load_default_inputs()
        lexicon_set = set(lexicon_words)

        expected_words = {"beach", "music", "ocean", "piano", "tempo", "waves", "wharf"}
        self.assertTrue(expected_words.issubset(lexicon_set))
        self.assertTrue(expected_words.issubset(set(clue_bank)))
        self.assertEqual(clue_bank["beach"][0], "Place to catch some rays")
        self.assertEqual(clue_bank["music"][0], "Art form with scales and chords")
        self.assertTrue({"beach", "ocean", "shore", "waves", "wharf"}.issubset(set(related_map["beach"])))
        self.assertTrue({"music", "album", "opera", "piano", "tempo"}.issubset(set(related_map["music"])))

        for removed_word in ("aahed", "antra", "ikeas", "lurie", "udals"):
            self.assertNotIn(removed_word, lexicon_set)
            self.assertNotIn(removed_word, clue_bank)

    def test_default_data_prefers_handwritten_clues(self) -> None:
        _, _, clue_bank = load_default_inputs()
        preferred_words = set(preferred_clue_words(clue_bank))

        self.assertIn("snail", preferred_words)
        self.assertIn("trade", preferred_words)
        self.assertNotIn("asked", preferred_words)
        self.assertEqual(clue_bank["asked"][0], 'Past tense of "ask"')
        self.assertTrue(clue_bank["aback"][0].startswith('Bundled-lexicon entry between "'))

    def test_default_demo_puzzle_uses_bundled_clues(self) -> None:
        _, related_map, clue_bank = load_default_inputs()
        puzzle = build_demo_puzzle(related_map, clue_bank)

        self.assertEqual(puzzle.title, "WATER Mini")
        self.assertEqual(len(puzzle.across), 5)
        self.assertEqual(len(puzzle.down), 5)

        for clue in puzzle.across:
            self.assertIn(clue.answer.lower(), clue_bank)
            self.assertEqual(clue.text, clue_bank[clue.answer.lower()][0])

        for clue in puzzle.down:
            self.assertIn(clue.answer.lower(), clue_bank)
            self.assertTrue(clue.text.strip())

if __name__ == "__main__":
    unittest.main()
