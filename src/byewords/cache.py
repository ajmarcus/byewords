import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Literal, TypedDict, cast

from byewords.grid import make_grid
from byewords.render import puzzle_to_dict
from byewords.theme import normalize_seeds
from byewords.types import Clue, GenerateConfig, Grid, Puzzle

DEFAULT_CACHE_DIRNAME = ".byewords-cache"


class CluePayload(TypedDict):
    number: int
    direction: Literal["across", "down"]
    answer: str
    text: str


class PuzzlePayload(TypedDict):
    title: str
    theme_words: list[str]
    grid: list[str]
    across: list[CluePayload]
    down: list[CluePayload]


def default_cache_dir() -> Path:
    configured = os.environ.get("BYEWORDS_CACHE_DIR")
    if configured:
        return Path(configured)
    return Path.cwd() / DEFAULT_CACHE_DIRNAME


def cache_key(seeds: tuple[str, ...], config: GenerateConfig, version: str = "") -> str:
    payload = {
        "seeds": normalize_seeds(seeds),
        "config": asdict(config),
        "version": version,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def cache_path(
    seeds: tuple[str, ...],
    config: GenerateConfig,
    cache_dir: Path | None = None,
    version: str = "",
) -> Path:
    normalized_seeds = normalize_seeds(seeds)
    label = "-".join(normalized_seeds) if normalized_seeds else "empty"
    return (cache_dir or default_cache_dir()) / f"{label}-{cache_key(seeds, config, version)}.json"


def load_cached_puzzle(
    seeds: tuple[str, ...],
    config: GenerateConfig,
    cache_dir: Path | None = None,
    version: str = "",
) -> Puzzle | None:
    path = cache_path(seeds, config, cache_dir, version)
    if not path.exists():
        return None
    payload = cast(PuzzlePayload, json.loads(path.read_text(encoding="utf-8")))
    return puzzle_from_dict(payload)


def save_cached_puzzle(
    seeds: tuple[str, ...],
    config: GenerateConfig,
    puzzle: Puzzle,
    cache_dir: Path | None = None,
    version: str = "",
) -> Path:
    path = cache_path(seeds, config, cache_dir, version)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(puzzle_to_dict(puzzle), indent=2), encoding="utf-8")
    return path


def puzzle_from_dict(payload: PuzzlePayload) -> Puzzle:
    grid = make_grid(cast(tuple[str, str, str, str, str], tuple(payload["grid"])))
    across = tuple(Clue(**clue) for clue in payload["across"])
    down = tuple(Clue(**clue) for clue in payload["down"])
    return Puzzle(
        grid=Grid(rows=grid.rows),
        across=across,
        down=down,
        theme_words=tuple(payload["theme_words"]),
        title=str(payload["title"]),
    )
