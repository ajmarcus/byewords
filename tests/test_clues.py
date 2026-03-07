import unittest

from byewords.clues import clue_for_slot, make_across_clues, make_down_clues
from byewords.grid import make_grid
from byewords.types import Slot
from tests.fixtures import TEST_GRID_ROWS


class TestClues(unittest.TestCase):
    def test_clue_for_slot_uses_answer_specific_clue_when_available(self) -> None:
        clue_bank: dict[str, tuple[str, ...]] = {"snail": ("Mollusk hauling its studio apartment",)}
        clue = clue_for_slot(
            Slot(direction="across", index=3, answer="snail"),
            clue_bank=clue_bank,
        )

        self.assertEqual(clue.text, "Mollusk hauling its studio apartment")

    def test_make_clues_builds_all_slots(self) -> None:
        grid = make_grid(TEST_GRID_ROWS)
        clue_bank: dict[str, tuple[str, ...]] = {"snail": ("Mollusk hauling its studio apartment",)}

        self.assertEqual(len(make_across_clues(grid, clue_bank)), 5)
        self.assertEqual(len(make_down_clues(grid, clue_bank)), 5)

    def test_clue_for_slot_falls_back_when_answer_is_missing(self) -> None:
        clue = clue_for_slot(
            Slot(direction="across", index=0, answer="abase"),
            clue_bank={},
        )

        self.assertEqual(clue.text, "Word with a repeated letter")


if __name__ == "__main__":
    unittest.main()
