from byewords.grid import distinct_entries, grid_columns
from byewords.types import CandidateGrid, Grid


def score_fill_quality(grid: Grid) -> float:
    entries = distinct_entries(grid)
    unique_letters = len(set("".join(entries)))
    repeated_letters = sum(len(entry) - len(set(entry)) for entry in entries)
    return unique_letters / 26 - repeated_letters / 100


def score_theme_density(grid: Grid, theme_words: set[str]) -> float:
    across_hits = sum(row in theme_words for row in grid.rows) / len(grid.rows)
    down_words = grid_columns(grid)
    down_hits = sum(word in theme_words for word in down_words) / len(down_words)
    return across_hits * 0.7 + down_hits * 0.3


def score_entry_diversity(grid: Grid) -> float:
    entries = distinct_entries(grid)
    return len(set(entries)) / len(entries)


def score_grid(grid: Grid, theme_words: set[str]) -> CandidateGrid:
    theme_score = score_theme_density(grid, theme_words)
    fill_score = score_fill_quality(grid)
    diversity_score = score_entry_diversity(grid)
    clue_score = diversity_score
    total_score = theme_score * 2.0 + fill_score + diversity_score
    return CandidateGrid(
        grid=grid,
        theme_score=theme_score,
        fill_score=fill_score,
        clue_score=clue_score,
        total_score=total_score,
    )


def rank_grids(grids: tuple[Grid, ...], theme_words: set[str]) -> tuple[CandidateGrid, ...]:
    scored = tuple(score_grid(grid, theme_words) for grid in grids)
    return tuple(sorted(scored, key=lambda candidate: (-candidate.total_score, candidate.grid.rows)))
