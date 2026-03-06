import json
import tempfile
import unittest
from pathlib import Path

from byewords.lexicon import (
    filter_legal_words,
    load_clue_bank,
    load_related_words,
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
            ("eases", "snail"),
        )

    def test_json_loaders_normalize_word_lists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            related_path = Path(directory, "related.json")
            clue_path = Path(directory, "clues.json")
            related_path.write_text(json.dumps({"Snail": ["Slime", "trail", "bad!"]}), encoding="utf-8")
            clue_path.write_text(json.dumps({"Snail": ["Garden crawler with a spiral shell"]}), encoding="utf-8")

            self.assertEqual(load_related_words(str(related_path)), {"snail": ("slime", "trail")})
            self.assertEqual(load_clue_bank(str(clue_path)), {"snail": ("Garden crawler with a spiral shell",)})


if __name__ == "__main__":
    unittest.main()
