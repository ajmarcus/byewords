from byewords.grid import grid_columns, slot_numbers
from byewords.types import Clue, Grid, Slot


def _fallback_clue_variants(answer: str) -> tuple[str, ...]:
    upper_answer = answer.upper()
    facts = [f"starts with {upper_answer[0]}"]
    if upper_answer[-1] != upper_answer[0]:
        facts.append(f"ends with {upper_answer[-1]}")
    if answer.endswith("ed"):
        facts.insert(0, "is past-tense")
    elif answer.endswith("s"):
        facts.insert(0, "is plural")
    if len(set(answer)) < len(answer):
        facts.append("has a repeated letter")

    variants = ["Entry that " + " and ".join(facts)]
    if answer.endswith("ed"):
        variants.append("Past-tense entry")
    elif answer.endswith("s"):
        variants.append("Plural entry")
    if len(set(answer)) < len(answer):
        variants.append("Word with a repeated letter")
    variants.extend(
        (
            f"Entry starting with {upper_answer[0]}",
            f"Entry ending with {upper_answer[-1]}",
            "Five-letter entry",
        )
    )
    return tuple(dict.fromkeys(variants))


def _fallback_clue(answer: str) -> str:
    return _fallback_clue_variants(answer)[0]


def _clue_candidates(answer: str, clue_bank: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    bank_clues = clue_bank.get(answer.lower(), ())
    latest_first = bank_clues[-1:] + bank_clues[:-1]
    return tuple(dict.fromkeys(latest_first + _fallback_clue_variants(answer)))


def _best_clue(
    answer: str,
    clue_bank: dict[str, tuple[str, ...]],
    used_texts: set[str] | None = None,
) -> str:
    for clue in _clue_candidates(answer, clue_bank):
        if used_texts is None or clue not in used_texts:
            if used_texts is not None:
                used_texts.add(clue)
            return clue

    clue = _fallback_clue(answer)
    if used_texts is not None:
        used_texts.add(clue)
    return clue


def clue_for_slot(
    slot: Slot,
    clue_bank: dict[str, tuple[str, ...]],
    used_texts: set[str] | None = None,
) -> Clue:
    return Clue(
        number=slot.index + 1,
        direction=slot.direction,
        answer=slot.answer,
        text=_best_clue(slot.answer, clue_bank, used_texts),
    )


def make_across_clues(
    grid: Grid,
    clue_bank: dict[str, tuple[str, ...]],
    used_texts: set[str] | None = None,
) -> tuple[Clue, ...]:
    clues = []
    for index, _ in enumerate(slot_numbers()):
        slot = Slot(direction="across", index=index, answer=grid.rows[index])
        clues.append(clue_for_slot(slot, clue_bank, used_texts))
    return tuple(clues)


def make_down_clues(
    grid: Grid,
    clue_bank: dict[str, tuple[str, ...]],
    used_texts: set[str] | None = None,
) -> tuple[Clue, ...]:
    clues = []
    for index, _ in enumerate(slot_numbers()):
        slot = Slot(direction="down", index=index, answer=grid_columns(grid)[index])
        clues.append(clue_for_slot(slot, clue_bank, used_texts))
    return tuple(clues)
