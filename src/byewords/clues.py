from byewords.grid import grid_columns, slot_numbers
from byewords.types import Clue, Grid, Slot


def _fallback_clue(answer: str) -> str:
    if answer.endswith("ed"):
        return "Past-tense verb"
    if answer.endswith("s"):
        return "Plural noun"
    if len(set(answer)) < len(answer):
        return "Word with a repeated letter"
    return "Five-letter entry"


def _best_clue(answer: str, clue_bank: dict[str, tuple[str, ...]]) -> str:
    clues = clue_bank.get(answer.lower(), ())
    if clues:
        return clues[0]
    return _fallback_clue(answer)


def clue_for_slot(
    slot: Slot,
    clue_bank: dict[str, tuple[str, ...]],
) -> Clue:
    return Clue(
        number=slot.index + 1,
        direction=slot.direction,
        answer=slot.answer,
        text=_best_clue(slot.answer, clue_bank),
    )


def make_across_clues(
    grid: Grid,
    clue_bank: dict[str, tuple[str, ...]],
) -> tuple[Clue, ...]:
    clues = []
    for index, _ in enumerate(slot_numbers()):
        slot = Slot(direction="across", index=index, answer=grid.rows[index])
        clues.append(clue_for_slot(slot, clue_bank))
    return tuple(clues)


def make_down_clues(
    grid: Grid,
    clue_bank: dict[str, tuple[str, ...]],
) -> tuple[Clue, ...]:
    clues = []
    for index, _ in enumerate(slot_numbers()):
        slot = Slot(direction="down", index=index, answer=grid_columns(grid)[index])
        clues.append(clue_for_slot(slot, clue_bank))
    return tuple(clues)
