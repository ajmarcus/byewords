import unittest

from byewords.grid import grid_columns, make_grid
from byewords.puz import puzzle_has_consistent_answers, puzzle_to_puz_bytes
from byewords.types import Clue, Puzzle
from tests.fixtures import TEST_GRID_COLUMNS, TEST_GRID_ROWS


def _checksum(data: bytes, seed: int = 0) -> int:
    checksum = seed
    for byte in data:
        checksum = (checksum >> 1) | ((checksum & 1) << 15)
        checksum = (checksum + byte) & 0xFFFF
    return checksum


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
        theme_words=("snail",),
        title="SNAIL Mini",
    )


class TestPuz(unittest.TestCase):
    def test_puzzle_has_consistent_answers_for_generated_clues(self) -> None:
        self.assertTrue(puzzle_has_consistent_answers(build_test_puzzle()))

    def test_puzzle_to_puz_bytes_writes_expected_header_and_payload(self) -> None:
        payload = puzzle_to_puz_bytes(build_test_puzzle())

        self.assertEqual(payload[2:14], b"ACROSS&DOWN\x00")
        self.assertEqual(payload[24:28], b"1.3\x00")
        self.assertEqual(payload[44], 5)
        self.assertEqual(payload[45], 5)
        self.assertEqual(int.from_bytes(payload[46:48], "little"), 10)
        self.assertEqual(int.from_bytes(payload[48:50], "little"), 1)
        self.assertEqual(int.from_bytes(payload[50:52], "little"), 0)
        self.assertEqual(payload[52:77], b"ADIEUBOOEDANTRASNAILEASES")
        self.assertEqual(payload[77:102], b"-------------------------")
        self.assertIn(b"SNAIL Mini\x00\x00\x00Across clue 1\x00", payload)
        self.assertTrue(payload.endswith(b"Down clue 5\x00\x00"))

    def test_puzzle_to_puz_bytes_uses_valid_overall_checksum(self) -> None:
        payload = puzzle_to_puz_bytes(build_test_puzzle())
        cib = payload[44:52]
        solution = payload[52:77]
        fill = payload[77:102]
        strings = payload[102:]

        expected = _checksum(strings, _checksum(fill, _checksum(solution, _checksum(cib))))
        self.assertEqual(int.from_bytes(payload[0:2], "little"), expected)

    def test_puzzle_to_puz_bytes_derives_down_entries_from_grid(self) -> None:
        puzzle = build_test_puzzle()

        self.assertEqual(tuple(clue.answer for clue in puzzle.down), grid_columns(puzzle.grid))


if __name__ == "__main__":
    unittest.main()
