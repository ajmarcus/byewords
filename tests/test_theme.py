import unittest

from byewords.theme import (
    build_candidate_pool,
    expand_theme_words,
    normalize_seeds,
    rank_theme_candidates,
)


class TestTheme(unittest.TestCase):
    def test_normalize_seeds_filters_invalid_entries(self) -> None:
        self.assertEqual(normalize_seeds(("Snail", "bad!", "eases", "snail")), ("snail", "eases"))

    def test_expand_theme_words_intersects_related_words_with_lexicon(self) -> None:
        lexicon = ("snail", "slime", "shell", "eases")
        related_map: dict[str, tuple[str, ...]] = {"snail": ("slime", "trail", "shell")}

        self.assertEqual(
            expand_theme_words(("snail",), related_map, lexicon),
            ("snail", "slime", "shell"),
        )

    def test_rank_theme_candidates_is_deterministic(self) -> None:
        ranked = rank_theme_candidates(("snail",), ("shell", "snail", "slime"))

        self.assertEqual(ranked, ("snail", "slime", "shell"))

    def test_build_candidate_pool_puts_theme_words_first(self) -> None:
        pool = build_candidate_pool(
            ("snail",),
            ("snail", "slime"),
            ("snail", "slime", "eases", "abase"),
            allow_neutral_fill=True,
        )

        self.assertEqual(pool, ("snail", "slime", "eases", "abase"))

    def test_build_candidate_pool_prioritizes_preferred_fill_before_other_neutral_words(self) -> None:
        pool = build_candidate_pool(
            ("snail",),
            ("snail",),
            ("snail", "abase", "eases", "slime"),
            allow_neutral_fill=True,
            preferred_words=("eases",),
        )

        self.assertEqual(pool, ("snail", "eases", "abase", "slime"))


if __name__ == "__main__":
    unittest.main()
