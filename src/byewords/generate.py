from importlib import resources

from byewords.clue_bank import preferred_clue_words
from byewords.clues import make_across_clues, make_down_clues
from byewords.grid import distinct_entries, make_grid
from byewords.lexicon import load_clue_bank, load_related_words, load_word_list
from byewords.prefixes import build_prefix_index
from byewords.score import rank_grids
from byewords.search import search_grids
from byewords.theme import build_candidate_pool, expand_theme_words, normalize_seeds
from byewords.types import GenerateConfig, Grid, Puzzle

DEFAULT_DEMO_ROWS = ("water", "alive", "tides", "event", "rests")


def _data_path(filename: str) -> str:
    return str(resources.files("byewords").joinpath("data", filename))


def load_default_inputs() -> tuple[tuple[str, ...], dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    lexicon_words = load_word_list(_data_path("words_5.txt"))
    related_words = load_related_words(_data_path("related_words.json"))
    clue_bank = load_clue_bank(_data_path("clue_bank.json"))
    return lexicon_words, related_words, clue_bank


def build_demo_puzzle(
    related_map: dict[str, tuple[str, ...]],
    clue_bank: dict[str, tuple[str, ...]],
) -> Puzzle:
    grid = make_grid(DEFAULT_DEMO_ROWS)
    used_clues: set[str] = set()
    across = make_across_clues(grid, clue_bank, used_clues)
    down = make_down_clues(grid, clue_bank, used_clues)
    return Puzzle(
        grid=grid,
        across=across,
        down=down,
        theme_words=related_map.get("water", DEFAULT_DEMO_ROWS),
        title="WATER Mini",
    )


def _theme_entry_count(grid: Grid, theme_words: set[str]) -> int:
    return sum(entry in theme_words for entry in distinct_entries(grid))


def _seed_entry_count(grid: Grid, seeds: set[str]) -> int:
    return sum(entry in seeds for entry in distinct_entries(grid))


def _candidate_windows(candidate_words: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    windows: list[tuple[str, ...]] = []
    for limit in (50, 200, 800):
        window = candidate_words[: min(limit, len(candidate_words))]
        if window and (not windows or window != windows[-1]):
            windows.append(window)
    return tuple(windows)


def _has_seeded_grid(grids: tuple[Grid, ...], seeds: set[str]) -> bool:
    return any(_seed_entry_count(grid, seeds) > 0 for grid in grids)


def _select_title_seed(seeds: tuple[str, ...], grid: Grid) -> str:
    entries = set(distinct_entries(grid))
    for seed in seeds:
        if seed in entries:
            return seed
    raise ValueError("unable to derive a puzzle title from the generated grid")


def generate_puzzle(
    seeds: tuple[str, ...],
    lexicon_words: tuple[str, ...],
    related_map: dict[str, tuple[str, ...]],
    clue_bank: dict[str, tuple[str, ...]],
    config: GenerateConfig = GenerateConfig(),
) -> Puzzle:
    normalized_seeds = normalize_seeds(seeds)
    if not normalized_seeds:
        raise ValueError("provide at least one five-letter seed word")
    lexicon_set = set(lexicon_words)
    available_seeds = tuple(seed for seed in normalized_seeds if seed in lexicon_set)
    if not available_seeds:
        raise ValueError("none of the provided seed words are available in the bundled 5-letter lexicon")
    theme_words = expand_theme_words(normalized_seeds, related_map, lexicon_words)
    preferred_words = preferred_clue_words(clue_bank)

    candidate_words = build_candidate_pool(
        seeds=available_seeds,
        theme_words=theme_words,
        lexicon=lexicon_words,
        allow_neutral_fill=config.allow_neutral_fill,
        preferred_words=preferred_words,
    )
    prefix_index = build_prefix_index(lexicon_words)
    grids: tuple[Grid, ...] = ()
    seed_word_set = set(available_seeds)
    for candidate_window in _candidate_windows(candidate_words):
        grids = search_grids(
            candidate_words=candidate_window,
            prefix_index=prefix_index,
            beam_width=config.beam_width,
            max_candidates=config.max_candidates,
        )
        if _has_seeded_grid(grids, seed_word_set):
            break
    if not grids and len(candidate_words) > config.beam_width:
        broadened_beam_width = min(len(candidate_words), config.beam_width * 5)
        for candidate_window in _candidate_windows(candidate_words):
            grids = search_grids(
                candidate_words=candidate_window,
                prefix_index=prefix_index,
                beam_width=broadened_beam_width,
                max_candidates=config.max_candidates,
            )
            if _has_seeded_grid(grids, seed_word_set):
                break
    theme_word_set = set(theme_words)
    ranked = rank_grids(grids, theme_word_set)
    seeded = tuple(
        candidate
        for candidate in ranked
        if _seed_entry_count(candidate.grid, seed_word_set) > 0
    )
    qualified = tuple(
        candidate
        for candidate in seeded
        if _theme_entry_count(candidate.grid, theme_word_set) >= config.min_theme_words
    )
    chosen = qualified[0] if qualified else (seeded[0] if seeded and config.allow_theme_fallback else None)
    if chosen is None:
        raise ValueError("unable to generate a valid 5x5 puzzle containing the provided seed words")
    best_grid = chosen.grid
    used_clues: set[str] = set()
    across = make_across_clues(best_grid, clue_bank, used_clues)
    down = make_down_clues(best_grid, clue_bank, used_clues)
    title_seed = _select_title_seed(available_seeds, best_grid).upper()
    return Puzzle(
        grid=best_grid,
        across=across,
        down=down,
        theme_words=theme_words,
        title=f"{title_seed} Mini",
    )
