GENERIC_CLUE_PREFIXES = (
    "Plural of ",
    "Past tense of ",
    "One who ",
    "Bundled-lexicon entry ",
    "First entry in the bundled lexicon",
    "Last entry in the bundled lexicon",
)


def is_generic_clue(clue: str) -> bool:
    return clue.startswith(GENERIC_CLUE_PREFIXES)


def preferred_clue_words(clue_bank: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    preferred = []
    for answer, clues in clue_bank.items():
        if clues and not is_generic_clue(clues[0]):
            preferred.append(answer)
    return tuple(preferred)
