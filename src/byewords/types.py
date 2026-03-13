from dataclasses import dataclass
from typing import Callable, Literal, TypeAlias

Direction = Literal["across", "down"]
ProgressStage = Literal["cache_hit", "runtime_report", "search", "seed_search", "solution", "window"]


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
    theme_subset: tuple[str, ...] = ()
    theme_weakest_link: float = 0.0
    passes_quality_gates: bool = True


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
    runtime_budget_ms: int | None = None


@dataclass(frozen=True)
class RuntimeReport:
    requested_seeds: tuple[str, ...]
    normalized_seeds: tuple[str, ...]
    available_seeds: tuple[str, ...]
    candidate_count: int
    candidate_window_sizes: tuple[int, ...]
    semantic_ordering: bool
    used_demo_grid: bool
    budget_exhausted: bool
    used_budget_fallback: bool
    selected_theme_words: tuple[str, ...] = ()
    selected_theme_subset: tuple[str, ...] = ()
    selected_theme_weakest_link: float = 0.0


@dataclass(frozen=True, slots=True)
class ProgressUpdate:
    stage: ProgressStage
    message: str
    partial_rows: tuple[str, ...] = ()
    runtime_report: RuntimeReport | None = None


ProgressCallback: TypeAlias = Callable[[ProgressUpdate], None]
