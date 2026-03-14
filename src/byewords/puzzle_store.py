from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import NotRequired, Protocol, TypedDict, TypeAlias, cast
from uuid import UUID

from byewords.clue_bank import is_generic_clue
from byewords.clues import make_across_clues, make_down_clues
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
    clue_stage: NotRequired["StoredClueStage"]
    across: list[CluePayload]
    down: list[CluePayload]


class StoredClueStage(TypedDict):
    answer_only_rank: int
    selected_rank: int
    clue_score: float
    total_score: float
    validation_passed: bool
    validation_errors: list[str]
    cached_answer_count: int
    regenerated_answer_count: int


@dataclass(frozen=True)
class OfflineBatchContext:
    lexicon_words: tuple[str, ...]
    clue_bank: dict[str, tuple[str, ...]]
    version: str
    semantic_vectors: WordVectorTable | None
    candidates_per_seed: int


@dataclass(frozen=True)
class ClueStageReviewCase:
    seed: str
    expected_answer_only_grid: tuple[str, str, str, str, str]
    expected_clue_stage_grid: tuple[str, str, str, str, str]
    note: str


@dataclass(frozen=True)
class ClueStageReviewReport:
    seed: str
    expected_answer_only_grid: tuple[str, str, str, str, str]
    expected_clue_stage_grid: tuple[str, str, str, str, str]
    answer_only_grid: tuple[str, str, str, str, str] | None
    clue_stage_grid: tuple[str, str, str, str, str] | None
    answer_only_matches: bool
    clue_stage_matches: bool
    rerank_changed: bool
    clue_stage_validation_passed: bool


_ORIGINAL_GENERATE_PUZZLE_CANDIDATES = generate_puzzle_candidates
_OFFLINE_BATCH_CONTEXT: OfflineBatchContext | None = None


class CluePackageLike(Protocol):
    answer: str
    cached: bool
    clues: tuple[str, ...]


ClueRegenerator: TypeAlias = Callable[
    [tuple[str, ...], dict[str, tuple[str, ...]], str],
    tuple[CluePackageLike, ...],
]


def default_puzzle_store_path() -> Path:
    return Path(str(resources.files("byewords").joinpath("data", "puzzles.json")))


def default_clue_bank_path() -> str:
    return str(resources.files("byewords").joinpath("data", "clue_bank.json"))


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


def top_clue_stage_records(
    store: dict[str, StoredPuzzleRecord],
    preferred_version: str,
    limit: int = 100,
) -> tuple[tuple[str, StoredPuzzleRecord], ...]:
    if limit <= 0:
        return ()
    ranked_records = sorted(
        (
            (public_id, record)
            for public_id, record in store.items()
            if isinstance(record.get("clue_stage"), dict)
        ),
        key=lambda item: _record_clue_stage_rank_key(item[0], item[1], preferred_version),
        reverse=True,
    )
    return tuple(ranked_records[:limit])


def review_clue_stage_reranking(
    cases: tuple[ClueStageReviewCase, ...],
    store: dict[str, StoredPuzzleRecord],
    preferred_version: str,
) -> tuple[ClueStageReviewReport, ...]:
    reports: list[ClueStageReviewReport] = []
    for case in cases:
        seed_store = {
            public_id: record
            for public_id, record in store.items()
            if record.get("seed") == case.seed
        }
        answer_only_ranked = top_answer_only_records(seed_store, preferred_version=preferred_version, limit=1)
        clue_stage_ranked = top_clue_stage_records(seed_store, preferred_version=preferred_version, limit=1)
        answer_only_record = answer_only_ranked[0][1] if answer_only_ranked else None
        clue_stage_record = clue_stage_ranked[0][1] if clue_stage_ranked else None
        answer_only_grid = _record_grid_rows(answer_only_record)
        clue_stage_grid = _record_grid_rows(clue_stage_record)
        clue_stage_metadata = clue_stage_record.get("clue_stage") if clue_stage_record is not None else None
        clue_stage_validation_passed = bool(
            isinstance(clue_stage_metadata, dict) and clue_stage_metadata.get("validation_passed")
        )
        reports.append(
            ClueStageReviewReport(
                seed=case.seed,
                expected_answer_only_grid=case.expected_answer_only_grid,
                expected_clue_stage_grid=case.expected_clue_stage_grid,
                answer_only_grid=answer_only_grid,
                clue_stage_grid=clue_stage_grid,
                answer_only_matches=answer_only_grid == case.expected_answer_only_grid,
                clue_stage_matches=clue_stage_grid == case.expected_clue_stage_grid,
                rerank_changed=answer_only_grid != clue_stage_grid,
                clue_stage_validation_passed=clue_stage_validation_passed,
            )
        )
    return tuple(reports)


def build_batch_puzzle_cache(
    lexicon_words: tuple[str, ...],
    clue_bank: dict[str, tuple[str, ...]],
    path: Path | None = None,
    vectors: WordVectorTable | None = None,
    candidates_per_seed: int = DEFAULT_CANDIDATES_PER_SEED,
    clue_bank_path: str | None = None,
    top_clue_limit: int = 100,
    clue_regenerator: ClueRegenerator | None = None,
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
    available_seeds = {
        record["seed"]
        for record in store.values()
    }
    pending_seeds = tuple(word for word in lexicon_words if word not in cached_seeds)
    if (
        path is None
        and store
        and pending_seeds
        and all(word in available_seeds for word in lexicon_words)
    ):
        store = _curate_seed_records(store, version, per_seed_limit=candidates_per_seed)
        store = _apply_top_clue_stage(
            store,
            preferred_version=version,
            clue_bank=clue_bank,
            clue_bank_path=clue_bank_path or default_clue_bank_path(),
            limit=top_clue_limit,
            clue_regenerator=clue_regenerator,
        )
        persist_puzzle_store(store, store_path)
        return store_path, len(store), 0
    pending_seed_set = set(pending_seeds)
    if pending_seed_set:
        store = {
            public_id: record
            for public_id, record in store.items()
            if record["seed"] not in pending_seed_set
        }
    if not pending_seeds:
        store = _curate_seed_records(store, version, per_seed_limit=candidates_per_seed)
        store = _apply_top_clue_stage(
            store,
            preferred_version=version,
            clue_bank=clue_bank,
            clue_bank_path=clue_bank_path or default_clue_bank_path(),
            limit=top_clue_limit,
            clue_regenerator=clue_regenerator,
        )
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
    store = _apply_top_clue_stage(
        store,
        preferred_version=version,
        clue_bank=clue_bank,
        clue_bank_path=clue_bank_path or default_clue_bank_path(),
        limit=top_clue_limit,
        clue_regenerator=clue_regenerator,
    )
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


def _record_grid_rows(
    record: StoredPuzzleRecord | None,
) -> tuple[str, str, str, str, str] | None:
    if record is None:
        return None
    raw_grid = record.get("grid")
    if not isinstance(raw_grid, list) or len(raw_grid) != 5 or not all(
        isinstance(row, str) for row in raw_grid
    ):
        return None
    return cast(tuple[str, str, str, str, str], tuple(raw_grid))


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


def _record_clue_stage_rank_key(
    public_id: str,
    record: StoredPuzzleRecord,
    preferred_version: str,
) -> tuple[int, int, float, float, float, float, int, str]:
    clue_stage = record.get("clue_stage")
    if not isinstance(clue_stage, dict):
        return (0, 0, float("-inf"), float("-inf"), float("-inf"), float("-inf"), 0, public_id)
    scores = record["answer_scores"]
    return (
        1 if record.get("version") == preferred_version else 0,
        1 if clue_stage["validation_passed"] else 0,
        clue_stage["total_score"],
        clue_stage["clue_score"],
        _answer_only_score(scores),
        scores["theme_score"],
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
    upgraded.pop("clue_stage", None)
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


def _apply_top_clue_stage(
    store: dict[str, StoredPuzzleRecord],
    *,
    preferred_version: str,
    clue_bank: dict[str, tuple[str, ...]],
    clue_bank_path: str,
    limit: int,
    clue_regenerator: ClueRegenerator | None,
) -> dict[str, StoredPuzzleRecord]:
    cleared_store = {
        public_id: _without_clue_stage(record)
        for public_id, record in store.items()
    }
    selected_records = top_answer_only_records(cleared_store, preferred_version=preferred_version, limit=limit)
    if not selected_records:
        return cleared_store

    selected_answers = tuple(
        dict.fromkeys(
            answer
            for _, record in selected_records
            for answer in _record_answers(record)
        )
    )
    packages = _regenerate_selected_clues(
        selected_answers,
        clue_bank,
        clue_bank_path,
        clue_regenerator,
    )
    package_by_answer = {package.answer: package for package in packages}

    staged_records: dict[str, StoredPuzzleRecord] = dict(cleared_store)
    answer_only_ranks: dict[str, int] = {}
    for answer_only_rank, (public_id, record) in enumerate(selected_records, start=1):
        staged_records[public_id] = _refresh_clue_stage_record(
            record,
            answer_only_rank=answer_only_rank,
            clue_bank=clue_bank,
            package_by_answer=package_by_answer,
        )
        answer_only_ranks[public_id] = answer_only_rank

    reranked_records = top_clue_stage_records(staged_records, preferred_version=preferred_version, limit=limit)
    for selected_rank, (public_id, record) in enumerate(reranked_records, start=1):
        clue_stage = record["clue_stage"]
        updated_clue_stage = dict(clue_stage)
        updated_clue_stage["answer_only_rank"] = answer_only_ranks[public_id]
        updated_clue_stage["selected_rank"] = selected_rank
        updated_record = dict(record)
        updated_record["clue_stage"] = cast(StoredClueStage, updated_clue_stage)
        staged_records[public_id] = cast(StoredPuzzleRecord, updated_record)

    return {
        public_id: record
        for public_id, record in sorted(staged_records.items())
    }


def _without_clue_stage(record: StoredPuzzleRecord) -> StoredPuzzleRecord:
    if "clue_stage" not in record:
        return record
    updated = dict(record)
    updated.pop("clue_stage", None)
    return cast(StoredPuzzleRecord, updated)


def _regenerate_selected_clues(
    answers: tuple[str, ...],
    clue_bank: dict[str, tuple[str, ...]],
    clue_bank_path: str,
    clue_regenerator: ClueRegenerator | None,
) -> tuple[CluePackageLike, ...]:
    if not answers:
        return ()
    regenerator = clue_regenerator or _default_clue_regenerator
    return regenerator(answers, clue_bank, clue_bank_path)


def _default_clue_regenerator(
    answers: tuple[str, ...],
    clue_bank: dict[str, tuple[str, ...]],
    clue_bank_path: str,
) -> tuple[CluePackageLike, ...]:
    from byewords.groq_clues import CluePackage, cached_clues_for_answer, regenerate_clues

    cached_packages = []
    missing_answers = []
    for answer in answers:
        cached_clues = cached_clues_for_answer(answer, clue_bank)
        if cached_clues is None:
            missing_answers.append(answer)
            continue
        cached_packages.append(CluePackage(answer=answer, cached=True, clues=cached_clues))

    if not missing_answers:
        return tuple(cached_packages)

    try:
        generated_packages = regenerate_clues(
            missing_answers,
            clue_bank,
            clue_bank_path,
        )
    except (OSError, RuntimeError, ValueError):
        return tuple(cached_packages)

    return tuple(cached_packages) + tuple(generated_packages)


def _refresh_clue_stage_record(
    record: StoredPuzzleRecord,
    *,
    answer_only_rank: int,
    clue_bank: dict[str, tuple[str, ...]],
    package_by_answer: dict[str, CluePackageLike],
) -> StoredPuzzleRecord:
    raw_grid = cast(tuple[str, str, str, str, str], tuple(record["grid"]))
    grid = make_grid(raw_grid)
    used_clues: set[str] = set()
    across = make_across_clues(grid, clue_bank, used_clues)
    down = make_down_clues(grid, clue_bank, used_clues)
    validation_errors = _validate_clues(across + down)
    clue_score = _score_clues(across + down, validation_errors)
    answers = _record_answers(record)
    cached_answer_count = 0
    regenerated_answer_count = 0
    for answer in answers:
        package = package_by_answer.get(answer)
        if package is None:
            continue
        if package.cached:
            cached_answer_count += 1
        else:
            regenerated_answer_count += 1

    updated_record = dict(record)
    updated_record["across"] = [cast(CluePayload, clue.__dict__) for clue in across]
    updated_record["down"] = [cast(CluePayload, clue.__dict__) for clue in down]
    updated_record["clue_stage"] = StoredClueStage(
        answer_only_rank=answer_only_rank,
        selected_rank=0,
        clue_score=clue_score,
        total_score=_answer_only_score(record["answer_scores"]) + clue_score,
        validation_passed=not validation_errors,
        validation_errors=list(validation_errors),
        cached_answer_count=cached_answer_count,
        regenerated_answer_count=regenerated_answer_count,
    )
    return cast(StoredPuzzleRecord, updated_record)


def _validate_clues(clues: tuple[object, ...]) -> tuple[str, ...]:
    errors: list[str] = []
    seen_texts: set[str] = set()
    for clue in clues:
        text = getattr(clue, "text").strip()
        answer = getattr(clue, "answer")
        if not text:
            errors.append(f"{answer}: empty clue")
            continue
        normalized_text = text.lower()
        if normalized_text in seen_texts:
            errors.append(f"{answer}: duplicate clue text")
        seen_texts.add(normalized_text)
        if is_generic_clue(text):
            errors.append(f"{answer}: generic clue")
        if len(text.split()) < 2:
            errors.append(f"{answer}: clue must contain at least two words")
    return tuple(errors)


def _score_clues(clues: tuple[object, ...], validation_errors: tuple[str, ...]) -> float:
    if not clues:
        return 0.0
    total = sum(_score_clue_text(getattr(clue, "text")) for clue in clues)
    average = total / len(clues)
    if validation_errors:
        average *= 0.5
    return round(average, 6)


def _score_clue_text(text: str) -> float:
    stripped = text.strip()
    if not stripped or is_generic_clue(stripped):
        return 0.0
    words = stripped.split()
    score = 0.4
    if 2 <= len(words) <= 8:
        score += 0.3
    elif len(words) > 8:
        score += 0.15
    if len(stripped) >= 16:
        score += 0.2
    if len({word.lower() for word in words}) == len(words):
        score += 0.1
    return min(score, 1.0)
