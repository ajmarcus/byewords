import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from byewords.theme import (
    build_candidate_pool,
    diversify_theme_words,
    lexicon_hash,
    load_word_vectors,
    normalize_seeds,
    rank_lexicon_for_seed,
    rank_theme_candidates,
    score_theme_subset,
    score_word_for_seed,
    validate_seed_words,
)


def _write_vector_table(
    path: Path,
    lexicon: tuple[str, ...],
    vectors: dict[str, list[int]],
) -> None:
    dimensions = len(next(iter(vectors.values())))
    payload = {
        "version": 1,
        "source": "unit-test-vectors",
        "dimensions": dimensions,
        "lexicon_hash": lexicon_hash(lexicon),
        "quantization": {
            "scheme": "int8",
            "scale": 0.25,
        },
        "vectors": vectors,
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


class TestTheme(unittest.TestCase):
    def test_normalize_seeds_filters_invalid_entries(self) -> None:
        self.assertEqual(normalize_seeds(("Snail", "bad!", "eases", "snail")), ("snail", "eases"))

    def test_validate_seed_words_rejects_words_missing_from_lexicon(self) -> None:
        with self.assertRaisesRegex(ValueError, "BEACH"):
            validate_seed_words(("snail", "beach"), ("snail", "slime"))

    def test_load_word_vectors_is_cached_for_unchanged_files(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "vectors.json"
            lexicon = ("beach", "ocean")
            _write_vector_table(
                path,
                lexicon,
                {
                    "beach": [4, 0, 0, 0],
                    "ocean": [3, 1, 0, 0],
                },
            )

            first = load_word_vectors(str(path))
            second = load_word_vectors(str(path))

        self.assertIs(first, second)
        self.assertEqual(first.lexicon_hash, lexicon_hash(lexicon))

    def test_score_word_for_seed_uses_cosine_similarity(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "vectors.json"
            lexicon = ("beach", "ocean", "music")
            _write_vector_table(
                path,
                lexicon,
                {
                    "beach": [4, 0, 0, 0],
                    "ocean": [3, 1, 0, 0],
                    "music": [0, 0, 4, 0],
                },
            )

            vectors = load_word_vectors(str(path))

        self.assertAlmostEqual(score_word_for_seed("ocean", ("beach",), vectors), 0.948683, places=5)
        self.assertAlmostEqual(score_word_for_seed("music", ("beach",), vectors), 0.0, places=5)

    def test_rank_lexicon_for_seed_orders_by_similarity_then_preferred_words(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "vectors.json"
            lexicon = ("beach", "ocean", "coast", "music")
            _write_vector_table(
                path,
                lexicon,
                {
                    "beach": [4, 0, 0, 0],
                    "ocean": [3, 1, 0, 0],
                    "coast": [2, 2, 0, 0],
                    "music": [0, 0, 4, 0],
                },
            )

            vectors = load_word_vectors(str(path))
            ranked = rank_lexicon_for_seed(("beach",), lexicon, vectors, preferred_words=("coast",))

        self.assertEqual(ranked, ("beach", "coast", "ocean", "music"))

    def test_rank_lexicon_for_seed_rejects_mismatched_lexicon_tables(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "vectors.json"
            _write_vector_table(
                path,
                ("beach", "ocean"),
                {
                    "beach": [4, 0, 0, 0],
                    "ocean": [3, 1, 0, 0],
                },
            )

            vectors = load_word_vectors(str(path))

        with self.assertRaisesRegex(ValueError, "missing lexicon entries: MUSIC"):
            rank_lexicon_for_seed(("beach",), ("beach", "music"), vectors)

    def test_diversify_theme_words_skips_near_duplicate_answers(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "vectors.json"
            lexicon = ("beach", "coast", "music", "ocean", "waves")
            _write_vector_table(
                path,
                lexicon,
                {
                    "beach": [4, 0, 0, 0],
                    "coast": [6, 4, 0, 0],
                    "music": [0, 0, 4, 0],
                    "ocean": [4, 1, 0, 0],
                    "waves": [2, 3, 0, 0],
                },
            )

            vectors = load_word_vectors(str(path))
            diversified = diversify_theme_words(
                ("ocean", "coast", "waves", "music"),
                ("beach",),
                vectors,
                limit=4,
            )

        self.assertEqual(diversified, ("ocean", "waves"))

    def test_score_theme_subset_selects_theme_bearing_answers(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "vectors.json"
            lexicon = ("beach", "coast", "music", "ocean", "waves")
            _write_vector_table(
                path,
                lexicon,
                {
                    "beach": [4, 0, 0, 0],
                    "coast": [6, 4, 0, 0],
                    "music": [0, 0, 4, 0],
                    "ocean": [4, 1, 0, 0],
                    "waves": [2, 3, 0, 0],
                },
            )

            vectors = load_word_vectors(str(path))
            breakdown = score_theme_subset(lexicon, ("beach",), vectors)

        self.assertEqual(breakdown.selected_words, ("ocean", "waves"))
        self.assertGreater(breakdown.mean_relevance, 0.75)
        self.assertGreater(breakdown.weakest_link, 0.7)
        self.assertGreaterEqual(breakdown.diversity, 0.0)
        self.assertGreater(breakdown.total, 1.5)

    def test_score_theme_subset_returns_zero_without_related_answers(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "vectors.json"
            lexicon = ("beach", "music", "piano")
            _write_vector_table(
                path,
                lexicon,
                {
                    "beach": [4, 0, 0, 0],
                    "music": [0, 4, 0, 0],
                    "piano": [0, 3, 1, 0],
                },
            )

            vectors = load_word_vectors(str(path))
            breakdown = score_theme_subset(("beach", "music", "piano"), ("beach",), vectors)

        self.assertEqual(breakdown.selected_words, ())
        self.assertEqual(breakdown.total, 0.0)

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
