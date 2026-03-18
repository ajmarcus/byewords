import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from byewords.grid import distinct_entries, make_grid
from byewords.score import rank_grids, score_entry_diversity, score_grid
from byewords.theme import lexicon_hash, load_word_vectors
from tests.fixtures import TEST_GRID_ROWS


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


class TestScore(unittest.TestCase):
    def test_entry_diversity_rewards_distinct_answers(self) -> None:
        distinct_grid = make_grid(TEST_GRID_ROWS)
        duplicate_grid = make_grid(("sator", "arepo", "tenet", "opera", "rotas"))

        self.assertGreater(score_entry_diversity(distinct_grid), score_entry_diversity(duplicate_grid))

    def test_rank_grids_orders_higher_scoring_grid_first(self) -> None:
        themed_grid = make_grid(TEST_GRID_ROWS)
        neutral_grid = make_grid(("abcde", "fghij", "klmno", "pqrst", "uvwxy"))

        ranked = rank_grids((neutral_grid, themed_grid))

        self.assertEqual(ranked[0].grid, neutral_grid)
        self.assertGreater(ranked[0].total_score, ranked[1].total_score)

    def test_score_grid_returns_decomposed_scores(self) -> None:
        grid = make_grid(TEST_GRID_ROWS)

        scored = score_grid(grid)

        self.assertGreater(scored.total_score, 0)
        self.assertEqual(scored.theme_score, 0)
        self.assertEqual(scored.grid, grid)
        self.assertEqual(scored.theme_subset, ())
        self.assertTrue(scored.passes_quality_gates)

    def test_score_grid_adds_semantic_theme_score_when_vectors_match(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "vectors.json"
            grid = make_grid(TEST_GRID_ROWS)
            lexicon = ("snail",) + distinct_entries(grid)
            _write_vector_table(
                path,
                lexicon,
                {
                    "snail": [4, 0, 0, 0],
                    "adieu": [0, 0, 4, 0],
                    "booed": [0, 4, 0, 0],
                    "antra": [3, 3, 0, 0],
                    "eases": [4, 1, 0, 0],
                    "abase": [0, 4, 0, 0],
                    "donna": [1, 3, 0, 0],
                    "iotas": [0, 0, 4, 0],
                    "eerie": [0, 4, 0, 0],
                    "udals": [0, 4, 0, 0],
                },
            )
            vectors = load_word_vectors(str(path))

        scored = score_grid(grid, seeds=("snail",), vectors=vectors)

        self.assertGreater(scored.theme_score, 0.0)
        self.assertGreater(scored.total_score, scored.fill_score + scored.clue_score)
        self.assertEqual(scored.theme_subset, ("eases", "antra", "donna"))
        self.assertGreater(scored.theme_weakest_link, 0.5)
        self.assertTrue(scored.passes_quality_gates)

    def test_rank_grids_uses_theme_score_when_fill_scores_tie(self) -> None:
        neutral_grid = make_grid(("abcde", "fghij", "klmno", "pqrst", "uvwxy"))
        themed_grid = make_grid(("zebra", "cumin", "vodka", "glyph", "fjord"))
        neutral_entries = distinct_entries(neutral_grid)
        themed_entries = distinct_entries(themed_grid)

        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "vectors.json"
            lexicon = ("beach",) + neutral_entries + themed_entries
            neutral_vectors = {word: [0, 0, 4, 0] for word in neutral_entries}
            themed_vector_cycle = ([4, 1, 0, 0], [3, 3, 0, 0], [2, 4, 0, 0])
            themed_vectors = {
                word: themed_vector_cycle[index % len(themed_vector_cycle)]
                for index, word in enumerate(themed_entries)
            }
            _write_vector_table(
                path,
                lexicon,
                {
                    "beach": [4, 0, 0, 0],
                    **neutral_vectors,
                    **themed_vectors,
                },
            )
            vectors = load_word_vectors(str(path))

        with patch("byewords.score.score_fill_quality", return_value=1.0), patch(
            "byewords.score.score_entry_diversity",
            return_value=1.0,
        ):
            ranked = rank_grids((neutral_grid, themed_grid), seeds=("beach",), vectors=vectors)

        self.assertEqual(ranked[0].grid, themed_grid)
        self.assertEqual(len(ranked), 2)
        self.assertGreater(ranked[0].theme_score, ranked[1].theme_score)

    def test_score_grid_rejects_duplicate_heavy_fill(self) -> None:
        scored = score_grid(make_grid(("aaaaa", "bbbbb", "ccccc", "ddddd", "eeeee")))

        self.assertLess(scored.fill_score, 0.3)
        self.assertFalse(scored.passes_quality_gates)

    def test_rank_grids_filters_semantically_weak_candidates(self) -> None:
        weak_grid = make_grid(("abcde", "fghij", "klmno", "pqrst", "uvwxy"))
        themed_grid = make_grid(("zebra", "cumin", "vodka", "glyph", "fjord"))
        weak_entries = distinct_entries(weak_grid)
        themed_entries = distinct_entries(themed_grid)

        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "vectors.json"
            lexicon = ("beach",) + weak_entries + themed_entries
            weak_vector_cycle = ([4, 6, 0, 0], [4, -6, 0, 0])
            weak_vectors = {
                word: weak_vector_cycle[index % len(weak_vector_cycle)]
                for index, word in enumerate(weak_entries)
            }
            themed_vector_cycle = ([4, 1, 0, 0], [3, 3, 0, 0], [2, 4, 0, 0])
            themed_vectors = {
                word: themed_vector_cycle[index % len(themed_vector_cycle)]
                for index, word in enumerate(themed_entries)
            }
            _write_vector_table(
                path,
                lexicon,
                {
                    "beach": [4, 0, 0, 0],
                    **weak_vectors,
                    **themed_vectors,
                },
            )
            vectors = load_word_vectors(str(path))

        with patch("byewords.score.score_fill_quality", return_value=1.0), patch(
            "byewords.score.score_entry_diversity",
            return_value=1.0,
        ):
            ranked = rank_grids((weak_grid, themed_grid), seeds=("beach",), vectors=vectors)

        self.assertEqual(tuple(candidate.grid for candidate in ranked), (themed_grid,))


if __name__ == "__main__":
    unittest.main()
