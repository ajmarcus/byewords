import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, cast

from byewords.clue_bank import is_generic_clue
from byewords.grid import distinct_entries, make_grid
from byewords.lexicon import filter_legal_words, load_clue_bank, normalize_word
from byewords.puzzle_store import (
    StoredPuzzleRecord,
    _refresh_stored_record_clues,
    load_puzzle_store,
    persist_puzzle_store,
    puzzle_store_version,
)
from byewords.theme import lexicon_hash

ClueBank: TypeAlias = dict[str, tuple[str, ...]]
JsonObject: TypeAlias = dict[str, object]


@dataclass(frozen=True)
class DataMaintenanceResult:
    word_count: int
    clue_entry_count: int
    vector_entry_count: int
    puzzle_record_count: int
    removed_clue_answers: tuple[str, ...]
    removed_vector_words: tuple[str, ...]
    missing_vector_words: tuple[str, ...]
    removed_puzzle_records: tuple[str, ...]


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
            dict.fromkeys(
                clue.strip()
                for clue in clue_bank[normalized_answer]
                if clue.strip() and not is_generic_clue(clue.strip())
            )
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


def curate_word_vectors(
    payload: JsonObject,
    words: tuple[str, ...],
) -> tuple[JsonObject, tuple[str, ...], tuple[str, ...]]:
    raw_vectors = payload.get("vectors")
    if not isinstance(raw_vectors, dict):
        raise ValueError("word_vectors.json must contain an object-valued 'vectors' field")

    valid_words = set(words)
    curated_vectors: dict[str, object] = {}
    removed_words: list[str] = []
    for raw_word, vector in sorted(raw_vectors.items()):
        if not isinstance(raw_word, str):
            removed_words.append(str(raw_word))
            continue
        normalized_word = normalize_word(raw_word)
        if normalized_word is None or normalized_word not in valid_words:
            removed_words.append(raw_word.lower())
            continue
        curated_vectors.setdefault(normalized_word, vector)

    curated_payload = dict(payload)
    curated_payload["lexicon_hash"] = lexicon_hash(words)
    curated_payload["vectors"] = {
        word: curated_vectors[word]
        for word in words
        if word in curated_vectors
    }
    missing_words = tuple(word for word in words if word not in curated_vectors)
    return curated_payload, tuple(sorted(set(removed_words))), missing_words


def persist_word_vectors(path: Path, payload: JsonObject) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def curate_puzzle_store(
    store: dict[str, StoredPuzzleRecord],
    words: tuple[str, ...],
    clue_bank: ClueBank,
) -> tuple[dict[str, StoredPuzzleRecord], tuple[str, ...]]:
    valid_words = set(words)
    preferred_version = puzzle_store_version(words, clue_bank)
    curated_store: dict[str, StoredPuzzleRecord] = {}
    removed_public_ids: list[str] = []

    for public_id, raw_record in sorted(store.items()):
        record = _curate_puzzle_record(
            raw_record,
            valid_words=valid_words,
            preferred_version=preferred_version,
            clue_bank=clue_bank,
        )
        if record is None:
            removed_public_ids.append(public_id)
            continue
        curated_store[public_id] = record

    return curated_store, tuple(removed_public_ids)


def _curate_puzzle_record(
    raw_record: object,
    *,
    valid_words: set[str],
    preferred_version: str,
    clue_bank: ClueBank,
) -> StoredPuzzleRecord | None:
    if not isinstance(raw_record, dict):
        return None

    record = cast(JsonObject, raw_record)
    seed = record.get("seed")
    title = record.get("title")
    uuid_text = record.get("uuid")
    raw_grid = record.get("grid")
    if not isinstance(seed, str) or not isinstance(title, str) or not isinstance(uuid_text, str):
        return None
    normalized_seed = normalize_word(seed)
    if normalized_seed is None or normalized_seed not in valid_words:
        return None
    if not isinstance(raw_grid, list) or len(raw_grid) != 5 or not all(isinstance(row, str) for row in raw_grid):
        return None

    try:
        grid = make_grid(cast(tuple[str, str, str, str, str], tuple(raw_grid)))
    except ValueError:
        return None

    answers = distinct_entries(grid)
    if any(answer not in valid_words for answer in answers):
        return None

    updated_record = dict(record)
    updated_record["seed"] = normalized_seed
    updated_record["version"] = preferred_version
    updated_record["grid"] = list(grid.rows)
    updated_record["answers"] = list(answers)
    updated_record["theme_words"] = _normalize_word_sequence(record.get("theme_words"), valid_words)
    updated_record["theme_subset"] = _normalize_word_sequence(record.get("theme_subset"), valid_words)
    refreshed_record = _refresh_stored_record_clues(
        cast(StoredPuzzleRecord, updated_record),
        clue_bank=clue_bank,
    )
    return refreshed_record


def _normalize_word_sequence(value: object, valid_words: set[str]) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized_words: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        normalized_word = normalize_word(item)
        if normalized_word is None or normalized_word not in valid_words:
            continue
        normalized_words.append(normalized_word)
    return list(dict.fromkeys(normalized_words))


def sort_bundled_data_files(
    words_path: Path,
    clue_bank_path: Path,
    *,
    vectors_path: Path | None = None,
    puzzles_path: Path | None = None,
) -> DataMaintenanceResult:
    sorted_words = sort_words(tuple(Path(words_path).read_text(encoding="utf-8").splitlines()))
    word_set = set(sorted_words)
    clue_bank = load_clue_bank(str(clue_bank_path))
    curated_clue_bank, removed_clue_answers = curate_clue_bank(clue_bank, word_set)

    persist_word_list(words_path, sorted_words)
    persist_clue_bank(clue_bank_path, curated_clue_bank)

    vector_entry_count = 0
    removed_vector_words: tuple[str, ...] = ()
    missing_vector_words: tuple[str, ...] = ()
    if vectors_path is not None:
        raw_vector_payload = json.loads(Path(vectors_path).read_text(encoding="utf-8"))
        if not isinstance(raw_vector_payload, dict):
            raise ValueError("word_vectors.json must contain a JSON object")
        curated_vectors, removed_vector_words, missing_vector_words = curate_word_vectors(
            cast(JsonObject, raw_vector_payload),
            sorted_words,
        )
        persist_word_vectors(vectors_path, curated_vectors)
        vector_values = curated_vectors.get("vectors")
        vector_entry_count = len(vector_values) if isinstance(vector_values, dict) else 0

    puzzle_record_count = 0
    removed_puzzle_records: tuple[str, ...] = ()
    if puzzles_path is not None:
        curated_store, removed_puzzle_records = curate_puzzle_store(
            load_puzzle_store(puzzles_path),
            sorted_words,
            curated_clue_bank,
        )
        persist_puzzle_store(curated_store, puzzles_path)
        puzzle_record_count = len(curated_store)

    return DataMaintenanceResult(
        word_count=len(sorted_words),
        clue_entry_count=len(curated_clue_bank),
        vector_entry_count=vector_entry_count,
        puzzle_record_count=puzzle_record_count,
        removed_clue_answers=removed_clue_answers,
        removed_vector_words=removed_vector_words,
        missing_vector_words=missing_vector_words,
        removed_puzzle_records=removed_puzzle_records,
    )
