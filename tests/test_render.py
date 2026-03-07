import runpy
import unittest
from typing import cast
from unittest.mock import patch

from byewords.generate import load_default_inputs
from byewords.grid import make_grid
from byewords.render import puzzle_to_dict, render_clues, render_grid_ascii, render_puzzle_text
from byewords.types import Clue, Puzzle
from tests.fixtures import TEST_GRID_COLUMNS, TEST_GRID_ROWS


def build_test_puzzle() -> Puzzle:
    grid = make_grid(TEST_GRID_ROWS)
    across = tuple(
        Clue(number=index + 1, direction="across", answer=answer, text=f"Across clue {index + 1}")
        for index, answer in enumerate(TEST_GRID_ROWS)
    )
    down = tuple(
        Clue(number=index + 1, direction="down", answer=answer, text=f"Down clue {index + 1}")
        for index, answer in enumerate(TEST_GRID_COLUMNS)
    )
    return Puzzle(
        grid=grid,
        across=across,
        down=down,
        theme_words=("snail", "eases"),
        title="SNAIL Mini",
    )


class TestRender(unittest.TestCase):
    def test_render_grid_ascii_uppercases_rows(self) -> None:
        self.assertEqual(render_grid_ascii(make_grid(TEST_GRID_ROWS)).splitlines()[0], "A D I E U")

    def test_render_clues_lists_across_then_down(self) -> None:
        rendered = render_clues(build_test_puzzle())

        self.assertIn("Across", rendered)
        self.assertIn("1. Across clue 1", rendered)
        self.assertIn("Down", rendered)
        self.assertIn("1. Down clue 1", rendered)

    def test_render_puzzle_text_includes_title_theme_and_grid(self) -> None:
        rendered = render_puzzle_text(build_test_puzzle())

        self.assertIn("SNAIL Mini", rendered)
        self.assertIn("Theme words: SNAIL, EASES", rendered)
        self.assertIn("A D I E U", rendered)

    def test_puzzle_to_dict_serializes_expected_shape(self) -> None:
        payload = puzzle_to_dict(build_test_puzzle())
        across = cast(list[dict[str, object]], payload["across"])
        down = cast(list[dict[str, object]], payload["down"])

        self.assertEqual(payload["title"], "SNAIL Mini")
        self.assertEqual(payload["grid"], list(TEST_GRID_ROWS))
        self.assertEqual(payload["theme_words"], ["snail", "eases"])
        self.assertEqual(across[0]["answer"], "adieu")
        self.assertEqual(down[0]["answer"], "abase")

    def test_load_default_inputs_reads_packaged_data(self) -> None:
        lexicon_words, related_words, clue_bank = load_default_inputs()

        self.assertIn("snail", lexicon_words)
        self.assertIn("eases", related_words["snail"])
        self.assertEqual(clue_bank["snail"][0], "Slow walker carrying its whole rent situation")

    def test_module_entrypoint_exits_with_main_status(self) -> None:
        with patch("byewords.cli.main", return_value=7):
            with self.assertRaises(SystemExit) as context:
                runpy.run_module("byewords.__main__", run_name="__main__")

        self.assertEqual(context.exception.code, 7)


if __name__ == "__main__":
    unittest.main()
