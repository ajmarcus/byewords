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
        bland_grid = make_grid(("sator", "arepo", "tenet", "opera", "rotas"))

        ranked = rank_grids((bland_grid, themed_grid))

        self.assertEqual(ranked[0].grid.rows, TEST_GRID_ROWS)
        self.assertGreater(ranked[0].total_score, ranked[1].total_score)

    def test_score_grid_returns_decomposed_scores(self) -> None:
        grid = make_grid(TEST_GRID_ROWS)

        scored = score_grid(grid)

        self.assertGreater(scored.total_score, 0)
        self.assertEqual(scored.theme_score, 0)
        self.assertEqual(scored.grid, grid)

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

    def test_rank_grids_uses_theme_score_when_fill_scores_tie(self) -> None:
        neutral_grid = make_grid(("abcde", "fghij", "klmno", "pqrst", "uvwxy"))
        themed_grid = make_grid(("zebra", "cumin", "vodka", "glyph", "fjord"))
        neutral_entries = distinct_entries(neutral_grid)
        themed_entries = distinct_entries(themed_grid)

        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "vectors.json"
            lexicon = ("beach",) + neutral_entries + themed_entries
            neutral_vectors = {word: [0, 0, 4, 0] for word in neutral_entries}
            themed_vectors = {
                word: [4, 1, 0, 0]
                for word in themed_entries
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
        self.assertGreater(ranked[0].theme_score, ranked[1].theme_score)


if __name__ == "__main__":
    unittest.main()
