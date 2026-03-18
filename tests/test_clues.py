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

    def test_clue_for_slot_prefers_latest_stored_clue(self) -> None:
        clue = clue_for_slot(
            Slot(direction="across", index=3, answer="snail"),
            clue_bank={
                "snail": (
                    "Mollusk hauling its studio apartment",
                    "Creature living the ultimate one-bag lifestyle",
                )
            },
        )

        self.assertEqual(clue.text, "Creature living the ultimate one-bag lifestyle")

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

        self.assertEqual(clue.text, "Entry that starts with A and ends with E and has a repeated letter")

    def test_make_clues_picks_an_unused_variant_when_answer_repeats(self) -> None:
        grid = make_grid(("cable", "agues", "buses", "leese", "esses"))
        used_texts: set[str] = set()

        across = make_across_clues(grid, {"cable": ("San Francisco streetcar",)}, used_texts)
        down = make_down_clues(grid, {"cable": ("San Francisco streetcar",)}, used_texts)

        self.assertEqual(across[0].text, "San Francisco streetcar")
        self.assertEqual(down[0].text, "Entry that starts with C and ends with E")

    def test_clue_for_slot_falls_back_to_older_stored_clue_when_latest_is_used(self) -> None:
        used_texts = {"Creature living the ultimate one-bag lifestyle"}
        clue = clue_for_slot(
            Slot(direction="across", index=3, answer="snail"),
            clue_bank={
                "snail": (
                    "Mollusk hauling its studio apartment",
                    "Creature living the ultimate one-bag lifestyle",
                )
            },
            used_texts=used_texts,
        )

        self.assertEqual(clue.text, "Mollusk hauling its studio apartment")


if __name__ == "__main__":
    unittest.main()
