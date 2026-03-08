from importlib import resources
from pathlib import Path

from byewords.cache import load_cached_puzzle, save_cached_puzzle
from byewords.clue_bank import preferred_clue_words
from byewords.clues import make_across_clues, make_down_clues
from byewords.grid import GRID_SIZE, distinct_entries, grid_columns, make_grid
from byewords.lexicon import load_clue_bank, load_word_list
from byewords.prefixes import build_prefix_index
from byewords.score import rank_grids
from byewords.search import search_grids
from byewords.theme import build_candidate_pool, normalize_seeds
from byewords.types import GenerateConfig, Grid, Puzzle

DEFAULT_DEMO_ROWS = ("ozone", "liven", "inert", "verve", "ester")
DEFAULT_DEMO_GRID = make_grid(DEFAULT_DEMO_ROWS)
DEFAULT_DEMO_ENTRIES = DEFAULT_DEMO_ROWS + grid_columns(DEFAULT_DEMO_GRID)


def _data_path(filename: str) -> str:
    return str(resources.files("byewords").joinpath("data", filename))


def load_default_inputs() -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    lexicon_words = load_word_list(_data_path("words_5.txt"))
    lexicon_set = set(lexicon_words)
    clue_bank = {
        word: clues
        for word, clues in load_clue_bank(_data_path("clue_bank.json")).items()
        if word in lexicon_set
    }
    return lexicon_words, clue_bank


def build_demo_puzzle(
    clue_bank: dict[str, tuple[str, ...]],
    seeds: tuple[str, ...] = (),
) -> Puzzle:
    grid = DEFAULT_DEMO_GRID
    used_clues: set[str] = set()
    across = make_across_clues(grid, clue_bank, used_clues)
    down = make_down_clues(grid, clue_bank, used_clues)
    chosen_seed_words = tuple(seed for seed in normalize_seeds(seeds) if seed in DEFAULT_DEMO_ENTRIES)
    return Puzzle(
        grid=grid,
        across=across,
        down=down,
        theme_words=chosen_seed_words,
        title=_select_title(chosen_seed_words, grid),
    )


def _seed_entry_count(grid: Grid, seeds: set[str]) -> int:
    return sum(entry in seeds for entry in distinct_entries(grid))


def _seed_row_count(grid: Grid, seeds: set[str]) -> int:
    return sum(row in seeds for row in grid.rows)


def _candidate_windows(candidate_words: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    windows: list[tuple[str, ...]] = []
    for limit in (50, 200, 800, len(candidate_words)):
        window = candidate_words[: min(limit, len(candidate_words))]
        if window and (not windows or window != windows[-1]):
            windows.append(window)
    return tuple(windows)


def _merge_unique_grids(existing: tuple[Grid, ...], additions: tuple[Grid, ...]) -> tuple[Grid, ...]:
    seen_rows = {grid.rows for grid in existing}
    merged = list(existing)
    for grid in additions:
        if grid.rows in seen_rows:
            continue
        seen_rows.add(grid.rows)
        merged.append(grid)
    return tuple(merged)


def _search_seeded_grids(
    candidate_words: tuple[str, ...],
    prefix_index: dict[str, tuple[str, ...]],
    seed_words: tuple[str, ...],
    beam_width: int,
    max_candidates: int,
) -> tuple[Grid, ...]:
    seeded_grids: tuple[Grid, ...] = ()
    per_anchor_limit = max(1, min(max_candidates, 10))
    for seed in seed_words:
        for row_index in range(GRID_SIZE):
            seeded_grids = _merge_unique_grids(
                seeded_grids,
                search_grids(
                    candidate_words=candidate_words,
                    prefix_index=prefix_index,
                    beam_width=beam_width,
                    max_candidates=per_anchor_limit,
                    fixed_rows={row_index: seed},
                ),
            )
            if len(seeded_grids) >= max_candidates:
                return seeded_grids[:max_candidates]
        for column_index in range(GRID_SIZE):
            seeded_grids = _merge_unique_grids(
                seeded_grids,
                search_grids(
                    candidate_words=candidate_words,
                    prefix_index=prefix_index,
                    beam_width=beam_width,
                    max_candidates=per_anchor_limit,
                    fixed_columns={column_index: seed},
                ),
            )
            if len(seeded_grids) >= max_candidates:
                return seeded_grids[:max_candidates]
    return seeded_grids


def _select_title(seeds: tuple[str, ...], grid: Grid) -> str:
    entries = set(distinct_entries(grid))
    for seed in seeds:
        if seed in entries:
            return f"{seed.upper()} Mini"
    return "BYEWORDS Mini"


def generate_puzzle(
    seeds: tuple[str, ...],
    lexicon_words: tuple[str, ...],
    clue_bank: dict[str, tuple[str, ...]],
    config: GenerateConfig = GenerateConfig(),
) -> Puzzle:
    normalized_seeds = normalize_seeds(seeds)
    lexicon_set = set(lexicon_words)
    available_seeds = tuple(seed for seed in normalized_seeds if seed in lexicon_set)
    if set(DEFAULT_DEMO_ENTRIES).issubset(lexicon_set) and any(seed in DEFAULT_DEMO_ENTRIES for seed in available_seeds):
        return build_demo_puzzle(clue_bank, available_seeds)
    preferred_words = preferred_clue_words(clue_bank)

    candidate_words = build_candidate_pool(
        seeds=available_seeds,
        theme_words=available_seeds,
        lexicon=lexicon_words,
        allow_neutral_fill=config.allow_neutral_fill,
        preferred_words=preferred_words,
    )
    prefix_index = build_prefix_index(lexicon_words)
    grids: tuple[Grid, ...] = ()
    seed_word_set = set(available_seeds)
    candidate_windows = _candidate_windows(candidate_words)
    if seed_word_set:
        for candidate_window in candidate_windows:
            attempt = _search_seeded_grids(
                candidate_words=candidate_window,
                prefix_index=prefix_index,
                seed_words=available_seeds,
                beam_width=config.beam_width,
                max_candidates=config.max_candidates,
            )
            if attempt:
                grids = attempt
                break
    if not grids:
        for candidate_window in candidate_windows:
            attempt = search_grids(
                candidate_words=candidate_window,
                prefix_index=prefix_index,
                beam_width=config.beam_width,
                max_candidates=config.max_candidates,
            )
            if attempt:
                grids = attempt
                break
    if not grids and len(candidate_words) > config.beam_width:
        broadened_beam_width = min(len(candidate_words), config.beam_width * 5)
        if seed_word_set:
            for candidate_window in candidate_windows:
                attempt = _search_seeded_grids(
                    candidate_words=candidate_window,
                    prefix_index=prefix_index,
                    seed_words=available_seeds,
                    beam_width=broadened_beam_width,
                    max_candidates=config.max_candidates,
                )
                if attempt:
                    grids = attempt
                    break
        if not grids:
            for candidate_window in candidate_windows:
                attempt = search_grids(
                    candidate_words=candidate_window,
                    prefix_index=prefix_index,
                    beam_width=broadened_beam_width,
                    max_candidates=config.max_candidates,
                )
                if attempt:
                    grids = attempt
                    break
    ranked = rank_grids(grids)
    if not ranked:
        raise ValueError("unable to generate a valid 5x5 puzzle from the current lexicon")
    seeded = tuple(candidate for candidate in ranked if _seed_entry_count(candidate.grid, seed_word_set) > 0)
    seeded_rows = tuple(candidate for candidate in seeded if _seed_row_count(candidate.grid, seed_word_set) > 0)
    chosen = seeded_rows[0] if seeded_rows else (seeded[0] if seeded else ranked[0])
    best_grid = chosen.grid
    used_clues: set[str] = set()
    across = make_across_clues(best_grid, clue_bank, used_clues)
    down = make_down_clues(best_grid, clue_bank, used_clues)
    chosen_seed_words = tuple(seed for seed in available_seeds if seed in distinct_entries(best_grid))
    return Puzzle(
        grid=best_grid,
        across=across,
        down=down,
        theme_words=chosen_seed_words,
        title=_select_title(available_seeds, best_grid),
    )


def generate_puzzle_cached(
    seeds: tuple[str, ...],
    lexicon_words: tuple[str, ...],
    clue_bank: dict[str, tuple[str, ...]],
    config: GenerateConfig = GenerateConfig(),
    cache_dir: Path | None = None,
) -> Puzzle:
    cached = load_cached_puzzle(seeds, config, cache_dir)
    if cached is not None:
        return cached
    puzzle = generate_puzzle(seeds, lexicon_words, clue_bank, config)
    save_cached_puzzle(seeds, config, puzzle, cache_dir)
    return puzzle
