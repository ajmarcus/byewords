from dataclasses import dataclass
from typing import Callable, Literal, TypeAlias

Direction = Literal["across", "down"]
ProgressStage = Literal["cache_hit", "search", "seed_search", "solution", "window"]


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
class ThemeScoreBreakdown:
    selected_words: tuple[str, ...]
    mean_relevance: float
    weakest_link: float
    diversity: float
    total: float


@dataclass(frozen=True)
class GenerateConfig:
    max_candidates: int = 500
    beam_width: int = 25
    allow_neutral_fill: bool = True
    random_seed: int = 0


@dataclass(frozen=True, slots=True)
class ProgressUpdate:
    stage: ProgressStage
    message: str
    partial_rows: tuple[str, ...] = ()


ProgressCallback: TypeAlias = Callable[[ProgressUpdate], None]
