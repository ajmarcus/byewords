import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from byewords.lexicon import filter_legal_words, load_clue_bank, normalize_word

ClueBank: TypeAlias = dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class DataMaintenanceResult:
    word_count: int
    clue_entry_count: int
    removed_clue_answers: tuple[str, ...]


def sort_words(words: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(filter_legal_words(words)))


def curate_clue_bank(
    clue_bank: ClueBank,
    valid_words: set[str],
) -> tuple[ClueBank, tuple[str, ...]]:
    curated: ClueBank = {}
    removed_answers: list[str] = []

    for answer in sorted(clue_bank):
        normalized_answer = normalize_word(answer)
        if normalized_answer is None or normalized_answer not in valid_words:
            removed_answers.append(answer.lower())
            continue

        cleaned_clues = tuple(
            dict.fromkeys(clue.strip() for clue in clue_bank[normalized_answer] if clue.strip())
        )
        if cleaned_clues:
            curated[normalized_answer] = cleaned_clues

    return curated, tuple(sorted(removed_answers))


def persist_word_list(path: Path, words: tuple[str, ...]) -> None:
    path.write_text("\n".join(words) + "\n", encoding="utf-8")


def persist_clue_bank(path: Path, clue_bank: ClueBank) -> None:
    serializable = {
        answer: list(clues)
        for answer, clues in clue_bank.items()
    }
    path.write_text(json.dumps(serializable, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sort_bundled_data_files(words_path: Path, clue_bank_path: Path) -> DataMaintenanceResult:
    sorted_words = sort_words(tuple(Path(words_path).read_text(encoding="utf-8").splitlines()))
    word_set = set(sorted_words)
    clue_bank = load_clue_bank(str(clue_bank_path))
    curated_clue_bank, removed_clue_answers = curate_clue_bank(clue_bank, word_set)

    persist_word_list(words_path, sorted_words)
    persist_clue_bank(clue_bank_path, curated_clue_bank)

    return DataMaintenanceResult(
        word_count=len(sorted_words),
        clue_entry_count=len(curated_clue_bank),
        removed_clue_answers=removed_clue_answers,
    )
