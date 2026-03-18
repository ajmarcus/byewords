from byewords.grid import distinct_entries
from byewords.theme import WordVectorTable, normalize_seeds, score_theme_subset
from byewords.types import CandidateGrid, Grid, ThemeScoreBreakdown

MIN_FILL_SCORE = 0.3
MIN_THEME_WORDS = 2
MIN_THEME_WEAKEST_LINK = -0.2


def score_fill_quality(grid: Grid) -> float:
    entries = distinct_entries(grid)
    unique_letters = len(set("".join(entries)))
    repeated_letters = sum(len(entry) - len(set(entry)) for entry in entries)
    return unique_letters / 26 - repeated_letters / 100


def score_entry_diversity(grid: Grid) -> float:
    entries = distinct_entries(grid)
    return len(set(entries)) / len(entries)


def _theme_breakdown(
    grid: Grid,
    seeds: tuple[str, ...],
    vectors: WordVectorTable | None,
) -> ThemeScoreBreakdown:
    if vectors is None:
        return ThemeScoreBreakdown((), 0.0, 0.0, 0.0, 0.0)
    normalized_seeds = tuple(
        seed for seed in normalize_seeds(seeds) if seed in vectors.vectors
    )
    if not normalized_seeds:
        return ThemeScoreBreakdown((), 0.0, 0.0, 0.0, 0.0)
    return score_theme_subset(distinct_entries(grid), normalized_seeds, vectors)


def _passes_theme_quality_gate(
    breakdown: ThemeScoreBreakdown,
    seeds: tuple[str, ...],
    vectors: WordVectorTable | None,
) -> bool:
    if vectors is None:
        return True
    normalized_seeds = tuple(
        seed for seed in normalize_seeds(seeds) if seed in vectors.vectors
    )
    if not normalized_seeds:
        return True
    if len(breakdown.selected_words) < MIN_THEME_WORDS:
        return True
    return (
        breakdown.weakest_link >= MIN_THEME_WEAKEST_LINK
    )


def score_grid(
    grid: Grid,
    seeds: tuple[str, ...] = (),
    vectors: WordVectorTable | None = None,
) -> CandidateGrid:
    fill_score = score_fill_quality(grid)
    diversity_score = score_entry_diversity(grid)
    theme_breakdown = _theme_breakdown(grid, seeds, vectors)
    theme_score = theme_breakdown.total
    clue_score = diversity_score
    total_score = fill_score + diversity_score + theme_score
    passes_quality_gates = (
        fill_score >= MIN_FILL_SCORE
        and _passes_theme_quality_gate(theme_breakdown, seeds, vectors)
    )
    return CandidateGrid(
        grid=grid,
        theme_score=theme_score,
        fill_score=fill_score,
        clue_score=clue_score,
        total_score=total_score,
        theme_subset=theme_breakdown.selected_words,
        theme_weakest_link=theme_breakdown.weakest_link,
        passes_quality_gates=passes_quality_gates,
    )


def rank_grids(
    grids: tuple[Grid, ...],
    seeds: tuple[str, ...] = (),
    vectors: WordVectorTable | None = None,
) -> tuple[CandidateGrid, ...]:
    scored = tuple(score_grid(grid, seeds=seeds, vectors=vectors) for grid in grids)
    eligible = tuple(candidate for candidate in scored if candidate.passes_quality_gates)
    return tuple(
        sorted(
            eligible,
            key=lambda candidate: (
                -candidate.total_score,
                -candidate.theme_weakest_link,
                -candidate.theme_score,
                candidate.grid.rows,
            ),
        )
    )
