from dataclasses import dataclass
from typing import Literal

Direction = Literal["across", "down"]


@dataclass(frozen=True)
class Grid:
    rows: tuple[str, str, str, str, str]


@dataclass(frozen=True)
class Slot:
    direction: Direction
    index: int
    answer: str


@dataclass(frozen=True)
class Clue:
    number: int
    direction: Direction
    answer: str
    text: str


@dataclass(frozen=True)
class Puzzle:
    grid: Grid
    across: tuple[Clue, ...]
    down: tuple[Clue, ...]
    theme_words: tuple[str, ...]
    title: str


@dataclass(frozen=True)
class CandidateGrid:
    grid: Grid
    theme_score: float
    fill_score: float
    clue_score: float
    total_score: float


@dataclass(frozen=True)
class GenerateConfig:
    max_candidates: int = 500
    beam_width: int = 100
    min_theme_words: int = 4
    allow_neutral_fill: bool = True
    allow_theme_fallback: bool = True
    random_seed: int = 0
