from byewords.grid import distinct_entries
from byewords.types import CandidateGrid, Grid


def score_fill_quality(grid: Grid) -> float:
    entries = distinct_entries(grid)
    unique_letters = len(set("".join(entries)))
    repeated_letters = sum(len(entry) - len(set(entry)) for entry in entries)
    return unique_letters / 26 - repeated_letters / 100

def score_entry_diversity(grid: Grid) -> float:
    entries = distinct_entries(grid)
    return len(set(entries)) / len(entries)


def score_grid(grid: Grid) -> CandidateGrid:
    fill_score = score_fill_quality(grid)
    diversity_score = score_entry_diversity(grid)
    clue_score = diversity_score
    total_score = fill_score + diversity_score
    return CandidateGrid(
        grid=grid,
        theme_score=0.0,
        fill_score=fill_score,
        clue_score=clue_score,
        total_score=total_score,
    )


def rank_grids(grids: tuple[Grid, ...]) -> tuple[CandidateGrid, ...]:
    scored = tuple(score_grid(grid) for grid in grids)
    return tuple(sorted(scored, key=lambda candidate: (-candidate.total_score, candidate.grid.rows)))
