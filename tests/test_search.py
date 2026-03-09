import unittest

from byewords.grid import grid_columns
from byewords.prefixes import build_prefix_index
from byewords.search import search_grids, valid_next_rows
from tests.fixtures import TEST_GRID_COLUMNS, TEST_GRID_ROWS, TEST_LEXICON


class TestSearch(unittest.TestCase):
    def test_valid_next_rows_respects_prefix_constraints(self) -> None:
        prefix_index = build_prefix_index(TEST_LEXICON)

        self.assertEqual(
            valid_next_rows(TEST_GRID_ROWS[:1], TEST_GRID_ROWS, prefix_index),
            ("booed",),
        )

    def test_valid_next_rows_honors_fixed_row_constraints(self) -> None:
        prefix_index = build_prefix_index(TEST_LEXICON)

        self.assertEqual(
            valid_next_rows(
                (),
                TEST_GRID_ROWS,
                prefix_index,
                fixed_rows={0: "adieu"},
            ),
            ("adieu",),
        )

    def test_valid_next_rows_honors_fixed_column_constraints(self) -> None:
        prefix_index = build_prefix_index(TEST_LEXICON)

        self.assertEqual(
            valid_next_rows(
                (),
                TEST_GRID_ROWS,
                prefix_index,
                fixed_columns={0: "abase"},
            ),
            ("adieu",),
        )

    def test_search_grids_finds_expected_grid_from_known_corpus(self) -> None:
        prefix_index = build_prefix_index(TEST_LEXICON)

        grids = search_grids(
            candidate_words=TEST_GRID_ROWS,
            prefix_index=prefix_index,
            beam_width=10,
            max_candidates=5,
        )

        self.assertEqual(len(grids), 1)
        self.assertEqual(grids[0].rows, TEST_GRID_ROWS)
        self.assertEqual(grid_columns(grids[0]), TEST_GRID_COLUMNS)

    def test_search_grids_returns_empty_tuple_when_no_grid_exists(self) -> None:
        prefix_index = build_prefix_index(("adieu", "booed", "abase"))

        self.assertEqual(
            search_grids(
                candidate_words=("adieu", "booed"),
                prefix_index=prefix_index,
                beam_width=10,
                max_candidates=5,
            ),
            (),
        )

    def test_search_grids_rejects_duplicate_entries_across_and_down(self) -> None:
        candidate_words = ("cable", "agues", "buses", "leese", "esses")
        prefix_index = build_prefix_index(candidate_words)

        self.assertEqual(
            search_grids(
                candidate_words=candidate_words,
                prefix_index=prefix_index,
                beam_width=10,
                max_candidates=5,
            ),
            (),
        )

    def test_search_grids_can_anchor_a_required_row(self) -> None:
        prefix_index = build_prefix_index(TEST_LEXICON)

        grids = search_grids(
            candidate_words=TEST_GRID_ROWS,
            prefix_index=prefix_index,
            beam_width=10,
            max_candidates=5,
            fixed_rows={3: "snail"},
        )

        self.assertEqual(len(grids), 1)
        self.assertEqual(grids[0].rows, TEST_GRID_ROWS)

    def test_search_grids_can_anchor_a_required_row_not_present_in_candidates(self) -> None:
        prefix_index = build_prefix_index(TEST_LEXICON)

        grids = search_grids(
            candidate_words=tuple(word for word in TEST_GRID_ROWS if word != "snail"),
            prefix_index=prefix_index,
            beam_width=10,
            max_candidates=5,
            fixed_rows={3: "snail"},
        )

        self.assertEqual(len(grids), 1)
        self.assertEqual(grids[0].rows, TEST_GRID_ROWS)

    def test_search_grids_can_anchor_a_required_column(self) -> None:
        prefix_index = build_prefix_index(TEST_LEXICON)

        grids = search_grids(
            candidate_words=TEST_GRID_ROWS,
            prefix_index=prefix_index,
            beam_width=10,
            max_candidates=5,
            fixed_columns={0: "abase"},
        )

        self.assertEqual(len(grids), 1)
        self.assertEqual(grids[0].rows, TEST_GRID_ROWS)

if __name__ == "__main__":
    unittest.main()
