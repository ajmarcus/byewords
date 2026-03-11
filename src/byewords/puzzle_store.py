from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from concurrent.futures import ThreadPoolExecutor
from importlib import resources
from pathlib import Path
from typing import TypedDict, cast
from uuid import UUID

from byewords.cache import CluePayload, PuzzlePayload
from byewords.generate import generate_puzzle
from byewords.grid import distinct_entries
from byewords.render import puzzle_to_dict
from byewords.score import score_grid
from byewords.types import Puzzle

_BASE62_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


class StoredAnswerScores(TypedDict):
    fill_score: float
    theme_score: float
    clue_score: float
    total_score: float
    seed_entry_count: int
    seed_row_count: int


class StoredPuzzleRecord(TypedDict):
    uuid: str
    seed: str
    version: str
    title: str
    theme_words: list[str]
    grid: list[str]
    answers: list[str]
    answer_scores: StoredAnswerScores
    across: list[CluePayload]
    down: list[CluePayload]


def default_puzzle_store_path() -> Path:
    return Path(str(resources.files("byewords").joinpath("data", "puzzles.json")))


def puzzle_store_version(
    lexicon_words: tuple[str, ...],
    clue_bank: dict[str, tuple[str, ...]],
) -> str:
    payload = {
        "lexicon_words": lexicon_words,
        "clue_bank": clue_bank,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def load_puzzle_store(path: Path | None = None) -> dict[str, StoredPuzzleRecord]:
    store_path = path or default_puzzle_store_path()
    if not store_path.exists():
        return {}
    raw = json.loads(store_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("puzzles.json must contain an object keyed by public puzzle id")
    return cast(dict[str, StoredPuzzleRecord], raw)


def persist_puzzle_store(store: dict[str, StoredPuzzleRecord], path: Path | None = None) -> Path:
    store_path = path or default_puzzle_store_path()
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store_path.write_text(json.dumps(store, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return store_path


def build_batch_puzzle_cache(
    lexicon_words: tuple[str, ...],
    clue_bank: dict[str, tuple[str, ...]],
    path: Path | None = None,
) -> tuple[Path, int, int]:
    store_path = path or default_puzzle_store_path()
    store = load_puzzle_store(store_path)
    version = puzzle_store_version(lexicon_words, clue_bank)
    cached_seeds = {
        record["seed"]
        for record in store.values()
        if record.get("version") == version
    }
    pending_seeds = tuple(word for word in lexicon_words if word not in cached_seeds)
    pending_seed_set = set(pending_seeds)
    if pending_seed_set:
        store = {
            public_id: record
            for public_id, record in store.items()
            if record["seed"] not in pending_seed_set or record.get("version") == version
        }
    if not pending_seeds:
        persist_puzzle_store(store, store_path)
        return store_path, len(store), 0

    worker_count = max(1, os.cpu_count() or 1)
    batch_size = worker_count
    generated_records = 0
    for offset in range(0, len(pending_seeds), batch_size):
        batch = pending_seeds[offset:offset + batch_size]
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for result in executor.map(
                _record_for_seed_task,
                (
                    (seed, lexicon_words, clue_bank, version)
                    for seed in batch
                ),
            ):
                if result is None:
                    continue
                public_id, record = result
                store[public_id] = record
                generated_records += 1
    persist_puzzle_store(store, store_path)
    return store_path, len(store), generated_records


def puzzle_answers_for_id(
    puzzle_id: str,
    path: Path | None = None,
) -> tuple[str, ...]:
    store = load_puzzle_store(path)
    if puzzle_id in store:
        return _record_answers(store[puzzle_id])
    for public_id, record in store.items():
        del public_id
        if record["uuid"] == puzzle_id:
            return _record_answers(record)
    raise ValueError(f"puzzle id not found in puzzles.json: {puzzle_id}")


def _record_for_seed_task(
    task: tuple[str, tuple[str, ...], dict[str, tuple[str, ...]], str],
) -> tuple[str, StoredPuzzleRecord] | None:
    seed, lexicon_words, clue_bank, version = task
    try:
        puzzle = generate_puzzle((seed,), lexicon_words, clue_bank)
    except ValueError:
        return None
    if seed not in puzzle.theme_words:
        return None
    uuid_text = _uuid7_string()
    return _make_public_id(UUID(uuid_text)), _record_from_puzzle(seed, puzzle, version, uuid_text)

def _record_from_puzzle(seed: str, puzzle: Puzzle, version: str, uuid_text: str) -> StoredPuzzleRecord:
    payload = cast(PuzzlePayload, puzzle_to_dict(puzzle))
    answers = distinct_entries(puzzle.grid)
    scores = score_grid(puzzle.grid)
    return StoredPuzzleRecord(
        uuid=uuid_text,
        seed=seed,
        version=version,
        title=str(payload["title"]),
        theme_words=list(payload["theme_words"]),
        grid=list(payload["grid"]),
        answers=list(answers),
        answer_scores=StoredAnswerScores(
            fill_score=scores.fill_score,
            theme_score=scores.theme_score,
            clue_score=scores.clue_score,
            total_score=scores.total_score,
            seed_entry_count=sum(answer == seed for answer in answers),
            seed_row_count=sum(row == seed for row in puzzle.grid.rows),
        ),
        across=list(payload["across"]),
        down=list(payload["down"]),
    )


def _uuid7_string() -> str:
    unix_ms = int(time.time() * 1000) & ((1 << 48) - 1)
    rand_a = secrets.randbits(12)
    rand_b = secrets.randbits(62)
    value = unix_ms << 80
    value |= 0x7 << 76
    value |= rand_a << 64
    value |= 0b10 << 62
    value |= rand_b
    return str(UUID(int=value))


def _make_public_id(uuid_value: UUID) -> str:
    value = uuid_value.int
    if value == 0:
        return _BASE62_ALPHABET[0]
    digits: list[str] = []
    while value:
        value, remainder = divmod(value, 62)
        digits.append(_BASE62_ALPHABET[remainder])
    return "".join(reversed(digits))


def _record_answers(record: StoredPuzzleRecord) -> tuple[str, ...]:
    stored_answers = record.get("answers")
    if isinstance(stored_answers, list) and all(isinstance(answer, str) for answer in stored_answers):
        return tuple(dict.fromkeys(stored_answers))
    clues = record["across"] + record["down"]
    return tuple(dict.fromkeys(clue["answer"] for clue in clues))
