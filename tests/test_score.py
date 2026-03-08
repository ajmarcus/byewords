import unittest

from byewords.grid import make_grid
from byewords.score import rank_grids, score_entry_diversity, score_grid
from tests.fixtures import TEST_GRID_ROWS


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


if __name__ == "__main__":
    unittest.main()
