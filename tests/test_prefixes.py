import unittest

from byewords.prefixes import build_prefix_index, has_prefix, words_with_prefix


class TestPrefixes(unittest.TestCase):
    def test_build_prefix_index_groups_words_by_prefix(self) -> None:
        prefix_index = build_prefix_index(("sator", "salsa", "opera"))

        self.assertEqual(words_with_prefix(prefix_index, "sa"), ("salsa", "sator"))
        self.assertEqual(words_with_prefix(prefix_index, "opera"), ("opera",))

    def test_has_prefix_distinguishes_missing_prefixes(self) -> None:
        prefix_index = build_prefix_index(("sator", "opera"))

        self.assertTrue(has_prefix(prefix_index, "op"))
        self.assertFalse(has_prefix(prefix_index, "zz"))

    def test_empty_prefix_returns_full_word_list(self) -> None:
        prefix_index = build_prefix_index(("opera", "sator", "opera"))

        self.assertEqual(words_with_prefix(prefix_index, ""), ("opera", "sator"))


if __name__ == "__main__":
    unittest.main()
