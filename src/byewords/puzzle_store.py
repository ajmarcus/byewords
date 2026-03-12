from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import NotRequired, TypedDict, cast
from uuid import UUID

from byewords.cache import CluePayload, PuzzlePayload
from byewords.generate import generate_puzzle_candidates
from byewords.grid import distinct_entries, make_grid
from byewords.render import puzzle_to_dict
from byewords.score import score_grid
from byewords.theme import WordVectorTable, lexicon_hash, load_word_vectors
from byewords.types import GenerateConfig, Puzzle

_BASE62_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
DEFAULT_CANDIDATES_PER_SEED = 2
_PROCESS_POOL_DISABLE_ENV = "BYEWWORDS_DISABLE_PROCESS_POOL"


class StoredAnswerScores(TypedDict):
    fill_score: float
    theme_score: float
    clue_score: float
    total_score: float
    answer_only_score: NotRequired[float]
    seed_entry_count: int
    seed_row_count: int


class StoredPuzzleRecord(TypedDict):
    uuid: str
    seed: str
    version: str
    title: str
    theme_words: list[str]
    theme_subset: NotRequired[list[str]]
    grid: list[str]
    answers: list[str]
    answer_scores: StoredAnswerScores
    across: list[CluePayload]
    down: list[CluePayload]


@dataclass(frozen=True)
class OfflineBatchContext:
    lexicon_words: tuple[str, ...]
    clue_bank: dict[str, tuple[str, ...]]
    version: str
    semantic_vectors: WordVectorTable | None
    candidates_per_seed: int


_ORIGINAL_GENERATE_PUZZLE_CANDIDATES = generate_puzzle_candidates
_OFFLINE_BATCH_CONTEXT: OfflineBatchContext | None = None


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


def top_answer_only_records(
    store: dict[str, StoredPuzzleRecord],
    preferred_version: str,
    limit: int = 100,
) -> tuple[tuple[str, StoredPuzzleRecord], ...]:
    if limit <= 0:
        return ()
    curated_store = _curate_seed_records(store, preferred_version, per_seed_limit=1)
    ranked_records = sorted(
        curated_store.items(),
        key=lambda item: _record_rank_key(item[0], item[1], preferred_version),
        reverse=True,
    )
    return tuple(ranked_records[:limit])


def build_batch_puzzle_cache(
    lexicon_words: tuple[str, ...],
    clue_bank: dict[str, tuple[str, ...]],
    path: Path | None = None,
    vectors: WordVectorTable | None = None,
    candidates_per_seed: int = DEFAULT_CANDIDATES_PER_SEED,
) -> tuple[Path, int, int]:
    if candidates_per_seed <= 0:
        raise ValueError("candidates_per_seed must be positive")
    store_path = path or default_puzzle_store_path()
    store = _upgrade_store_records(load_puzzle_store(store_path), vectors)
    version = puzzle_store_version(lexicon_words, clue_bank)
    semantic_vectors = _resolve_semantic_vectors(lexicon_words, vectors)
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
            if record["seed"] not in pending_seed_set
        }
    if not pending_seeds:
        store = _curate_seed_records(store, version, per_seed_limit=candidates_per_seed)
        persist_puzzle_store(store, store_path)
        return store_path, len(store), 0

    worker_count = max(1, os.cpu_count() or 1)
    batch_context = OfflineBatchContext(
        lexicon_words=lexicon_words,
        clue_bank=clue_bank,
        version=version,
        semantic_vectors=semantic_vectors,
        candidates_per_seed=candidates_per_seed,
    )
    generated_records = 0
    for result in _seed_record_results(pending_seeds, batch_context, worker_count):
        if not result:
            continue
        for public_id, record in result:
            store[public_id] = record
            generated_records += 1
    store = _curate_seed_records(store, version, per_seed_limit=candidates_per_seed)
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


def _seed_record_results(
    pending_seeds: tuple[str, ...],
    batch_context: OfflineBatchContext,
    worker_count: int,
) -> tuple[tuple[tuple[str, StoredPuzzleRecord], ...], ...]:
    if _use_process_pool(worker_count):
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_initialize_offline_batch_context,
            initargs=(batch_context,),
        ) as executor:
            return tuple(executor.map(_record_for_seed_task, pending_seeds, chunksize=1))
    _initialize_offline_batch_context(batch_context)
    return tuple(_record_for_seed_task(seed) for seed in pending_seeds)


def _use_process_pool(worker_count: int) -> bool:
    if worker_count <= 1:
        return False
    if os.environ.get(_PROCESS_POOL_DISABLE_ENV) == "1":
        return False
    return generate_puzzle_candidates is _ORIGINAL_GENERATE_PUZZLE_CANDIDATES


def _initialize_offline_batch_context(batch_context: OfflineBatchContext) -> None:
    global _OFFLINE_BATCH_CONTEXT
    _OFFLINE_BATCH_CONTEXT = batch_context


def _record_for_seed_task(seed: str) -> tuple[tuple[str, StoredPuzzleRecord], ...]:
    batch_context = _OFFLINE_BATCH_CONTEXT
    if batch_context is None:
        raise RuntimeError("offline batch context is not initialized")
    return _record_for_seed(seed, batch_context)


def _record_for_seed(
    seed: str,
    batch_context: OfflineBatchContext,
) -> tuple[tuple[str, StoredPuzzleRecord], ...]:
    try:
        candidate_puzzles = generate_puzzle_candidates(
            (seed,),
            batch_context.lexicon_words,
            batch_context.clue_bank,
            config=GenerateConfig(
                max_candidates=max(
                    batch_context.candidates_per_seed * 2,
                    batch_context.candidates_per_seed,
                )
            ),
        )
    except ValueError:
        return ()

    records: list[tuple[str, StoredPuzzleRecord]] = []
    for puzzle in candidate_puzzles:
        if seed not in puzzle.theme_words:
            continue
        uuid_text = _uuid7_string()
        public_id = _make_public_id(UUID(uuid_text))
        records.append(
            (
                public_id,
                _record_from_puzzle(
                    seed,
                    puzzle,
                    batch_context.version,
                    uuid_text,
                    semantic_vectors=batch_context.semantic_vectors,
                ),
            )
        )

    ranked_records = sorted(
        records,
        key=lambda item: _record_rank_key(item[0], item[1], batch_context.version),
        reverse=True,
    )
    return tuple(ranked_records[:batch_context.candidates_per_seed])


def _record_from_puzzle(
    seed: str,
    puzzle: Puzzle,
    version: str,
    uuid_text: str,
    *,
    semantic_vectors: WordVectorTable | None = None,
) -> StoredPuzzleRecord:
    payload = cast(PuzzlePayload, puzzle_to_dict(puzzle))
    answers = distinct_entries(puzzle.grid)
    scores = score_grid(puzzle.grid, seeds=(seed,), vectors=semantic_vectors)
    return StoredPuzzleRecord(
        uuid=uuid_text,
        seed=seed,
        version=version,
        title=str(payload["title"]),
        theme_words=list(payload["theme_words"]),
        theme_subset=list(scores.theme_subset),
        grid=list(payload["grid"]),
        answers=list(answers),
        answer_scores=StoredAnswerScores(
            fill_score=scores.fill_score,
            theme_score=scores.theme_score,
            clue_score=scores.clue_score,
            answer_only_score=scores.fill_score + scores.theme_score,
            total_score=scores.fill_score + scores.theme_score + scores.clue_score,
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


def _resolve_semantic_vectors(
    lexicon_words: tuple[str, ...],
    vectors: WordVectorTable | None,
) -> WordVectorTable | None:
    if vectors is not None:
        _validate_vector_table(lexicon_words, vectors)
        return vectors
    try:
        default_vectors = load_word_vectors(str(resources.files("byewords").joinpath("data", "word_vectors.json")))
    except (FileNotFoundError, ValueError):
        return None
    return default_vectors if _vector_table_matches_lexicon(lexicon_words, default_vectors) else None


def _validate_vector_table(
    lexicon_words: tuple[str, ...],
    vectors: WordVectorTable,
) -> None:
    if not _vector_table_matches_lexicon(lexicon_words, vectors):
        raise ValueError("word vectors do not match the requested lexicon")


def _vector_table_matches_lexicon(
    lexicon_words: tuple[str, ...],
    vectors: WordVectorTable,
) -> bool:
    unique_lexicon = tuple(dict.fromkeys(lexicon_words))
    if vectors.lexicon_hash != lexicon_hash(unique_lexicon):
        return False
    return all(word in vectors.vectors for word in unique_lexicon)


def _curate_seed_records(
    store: dict[str, StoredPuzzleRecord],
    preferred_version: str,
    per_seed_limit: int = 1,
) -> dict[str, StoredPuzzleRecord]:
    best_records: dict[str, list[tuple[str, StoredPuzzleRecord]]] = {}
    for public_id, record in store.items():
        if not _record_has_rank_metadata(record):
            continue
        best_records.setdefault(record["seed"], []).append((public_id, record))

    curated_records: list[tuple[str, StoredPuzzleRecord]] = []
    for seed, records in best_records.items():
        del seed
        ranked = sorted(
            records,
            key=lambda item: _record_rank_key(item[0], item[1], preferred_version),
            reverse=True,
        )
        curated_records.extend(ranked[:per_seed_limit])

    return {
        public_id: record
        for public_id, record in sorted(curated_records, key=lambda item: item[0])
    }


def _record_rank_key(
    public_id: str,
    record: StoredPuzzleRecord,
    preferred_version: str,
) -> tuple[int, float, float, float, int, str]:
    scores = record["answer_scores"]
    answer_only_score = _answer_only_score(scores)
    return (
        1 if record.get("version") == preferred_version else 0,
        answer_only_score,
        scores["theme_score"],
        scores["fill_score"],
        scores["seed_row_count"],
        public_id,
    )


def _answer_only_score(scores: StoredAnswerScores) -> float:
    stored_score = scores.get("answer_only_score")
    if isinstance(stored_score, int | float):
        return float(stored_score)
    return float(scores["fill_score"] + scores["theme_score"])


def _upgrade_store_records(
    store: dict[str, StoredPuzzleRecord],
    vectors: WordVectorTable | None,
) -> dict[str, StoredPuzzleRecord]:
    return {
        public_id: _upgrade_store_record(record, vectors)
        for public_id, record in store.items()
    }


def _upgrade_store_record(
    record: StoredPuzzleRecord,
    vectors: WordVectorTable | None,
) -> StoredPuzzleRecord:
    if _record_has_rank_metadata(record) and isinstance(record.get("answers"), list):
        return record

    raw_grid = record.get("grid")
    if not isinstance(raw_grid, list) or len(raw_grid) != 5 or not all(
        isinstance(row, str) for row in raw_grid
    ):
        return record
    try:
        grid = make_grid(cast(tuple[str, str, str, str, str], tuple(raw_grid)))
    except ValueError:
        return record

    seed = record.get("seed")
    if not isinstance(seed, str):
        return record
    answers = _record_answers(record)
    scores = score_grid(grid, seeds=(seed,), vectors=vectors)
    upgraded = dict(record)
    upgraded["answers"] = list(answers)
    upgraded["theme_subset"] = list(scores.theme_subset)
    upgraded["answer_scores"] = StoredAnswerScores(
        fill_score=scores.fill_score,
        theme_score=scores.theme_score,
        clue_score=scores.clue_score,
        answer_only_score=scores.fill_score + scores.theme_score,
        total_score=scores.fill_score + scores.theme_score + scores.clue_score,
        seed_entry_count=sum(answer == seed for answer in answers),
        seed_row_count=sum(row == seed for row in grid.rows),
    )
    return cast(StoredPuzzleRecord, upgraded)


def _record_has_rank_metadata(record: StoredPuzzleRecord) -> bool:
    answer_scores = record.get("answer_scores")
    if not isinstance(answer_scores, dict):
        return False
    return all(
        key in answer_scores
        for key in ("fill_score", "theme_score", "clue_score", "seed_row_count")
    )
