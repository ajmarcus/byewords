import json
import tempfile
import unittest
from pathlib import Path

from byewords.lexicon import (
    filter_legal_words,
    load_clue_bank,
    normalize_word,
)


class TestLexicon(unittest.TestCase):
    def test_normalize_word_accepts_only_five_letter_alpha_words(self) -> None:
        self.assertEqual(normalize_word("Snail"), "snail")
        self.assertIsNone(normalize_word("four"))
        self.assertIsNone(normalize_word("toolong"))
        self.assertIsNone(normalize_word("abc12"))

    def test_filter_legal_words_normalizes_and_deduplicates(self) -> None:
        self.assertEqual(
            filter_legal_words(("Snail", "snail", "eases", "toolong", "bad!")),
            ("snail", "eases"),
        )

    def test_load_clue_bank_preserves_string_clues(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            clue_path = Path(directory, "clues.json")
            clue_path.write_text(json.dumps({"Snail": ["Mollusk hauling its studio apartment"]}), encoding="utf-8")

            self.assertEqual(load_clue_bank(str(clue_path)), {"snail": ("Mollusk hauling its studio apartment",)})


if __name__ == "__main__":
    unittest.main()
