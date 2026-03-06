import unittest

from byewords.grid import (
    distinct_entries,
    grid_columns,
    is_full_grid_valid,
    make_grid,
    partial_column_prefixes,
    slot_numbers,
)


class TestGrid(unittest.TestCase):
    def test_grid_columns_follow_row_order(self) -> None:
        grid = make_grid(("sator", "arepo", "tenet", "opera", "rotas"))

        self.assertEqual(
            grid_columns(grid),
            ("sator", "arepo", "tenet", "opera", "rotas"),
        )

    def test_partial_column_prefixes_grow_by_depth(self) -> None:
        self.assertEqual(
            partial_column_prefixes(("sator", "arepo", "tenet")),
            ("sat", "are", "ten", "ope", "rot"),
        )

    def test_is_full_grid_valid_checks_rows_and_columns(self) -> None:
        grid = make_grid(("sator", "arepo", "tenet", "opera", "rotas"))
        lexicon = {"sator", "arepo", "tenet", "opera", "rotas"}

        self.assertTrue(is_full_grid_valid(grid, lexicon))
        self.assertFalse(is_full_grid_valid(grid, lexicon - {"rotas"}))

    def test_distinct_entries_returns_rows_then_columns(self) -> None:
        grid = make_grid(("sator", "arepo", "tenet", "opera", "rotas"))

        self.assertEqual(
            distinct_entries(grid),
            (
                "sator",
                "arepo",
                "tenet",
                "opera",
                "rotas",
                "sator",
                "arepo",
                "tenet",
                "opera",
                "rotas",
            ),
        )

    def test_slot_numbers_are_fixed_for_full_grid(self) -> None:
        self.assertEqual(slot_numbers(), (1, 2, 3, 4, 5))

    def test_make_grid_rejects_invalid_rows(self) -> None:
        with self.assertRaises(ValueError):
            make_grid(("short", "ok", "words", "here!", "later"))


if __name__ == "__main__":
    unittest.main()
