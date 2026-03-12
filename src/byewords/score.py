from byewords.grid import distinct_entries
from byewords.theme import WordVectorTable, normalize_seeds, score_theme_subset
from byewords.types import CandidateGrid, Grid


def score_fill_quality(grid: Grid) -> float:
    entries = distinct_entries(grid)
    unique_letters = len(set("".join(entries)))
    repeated_letters = sum(len(entry) - len(set(entry)) for entry in entries)
    return unique_letters / 26 - repeated_letters / 100


def score_entry_diversity(grid: Grid) -> float:
    entries = distinct_entries(grid)
    return len(set(entries)) / len(entries)


def _score_theme_quality(
    grid: Grid,
    seeds: tuple[str, ...],
    vectors: WordVectorTable | None,
) -> float:
    if vectors is None:
        return 0.0
    normalized_seeds = tuple(
        seed for seed in normalize_seeds(seeds) if seed in vectors.vectors
    )
    if not normalized_seeds:
        return 0.0
    return score_theme_subset(distinct_entries(grid), normalized_seeds, vectors).total


def score_grid(
    grid: Grid,
    seeds: tuple[str, ...] = (),
    vectors: WordVectorTable | None = None,
) -> CandidateGrid:
    fill_score = score_fill_quality(grid)
    diversity_score = score_entry_diversity(grid)
    theme_score = _score_theme_quality(grid, seeds, vectors)
    clue_score = diversity_score
    total_score = fill_score + diversity_score + theme_score
    return CandidateGrid(
        grid=grid,
        theme_score=theme_score,
        fill_score=fill_score,
        clue_score=clue_score,
        total_score=total_score,
    )


def rank_grids(
    grids: tuple[Grid, ...],
    seeds: tuple[str, ...] = (),
    vectors: WordVectorTable | None = None,
) -> tuple[CandidateGrid, ...]:
    scored = tuple(score_grid(grid, seeds=seeds, vectors=vectors) for grid in grids)
    return tuple(
        sorted(
            scored,
            key=lambda candidate: (
                -candidate.total_score,
                -candidate.theme_score,
                candidate.grid.rows,
            ),
        )
    )
