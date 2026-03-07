from importlib import resources

from byewords.clues import make_across_clues, make_down_clues
from byewords.grid import distinct_entries
from byewords.lexicon import load_clue_bank, load_related_words, load_word_list
from byewords.prefixes import build_prefix_index
from byewords.score import rank_grids
from byewords.search import search_grids
from byewords.theme import build_candidate_pool, expand_theme_words, normalize_seeds
from byewords.types import GenerateConfig, Grid, Puzzle


def _data_path(filename: str) -> str:
    return str(resources.files("byewords").joinpath("data", filename))


def load_default_inputs() -> tuple[tuple[str, ...], dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    lexicon_words = load_word_list(_data_path("words_5.txt"))
    related_words = load_related_words(_data_path("related_words.json"))
    clue_bank = load_clue_bank(_data_path("clue_bank.json"))
    return lexicon_words, related_words, clue_bank


def _theme_entry_count(grid: Grid, theme_words: set[str]) -> int:
    return sum(entry in theme_words for entry in distinct_entries(grid))


def _select_title_seed(seeds: tuple[str, ...], theme_words: tuple[str, ...]) -> str:
    for seed in seeds:
        if seed in theme_words:
            return seed
    if theme_words:
        return theme_words[0]
    raise ValueError("unable to derive a puzzle title from the provided seeds")


def generate_puzzle(
    seeds: tuple[str, ...],
    lexicon_words: tuple[str, ...],
    related_map: dict[str, tuple[str, ...]],
    clue_bank: dict[str, tuple[str, ...]],
    config: GenerateConfig = GenerateConfig(),
) -> Puzzle:
    normalized_seeds = normalize_seeds(seeds)
    theme_words = expand_theme_words(normalized_seeds, related_map, lexicon_words)
    if len(theme_words) < config.min_theme_words:
        raise ValueError("unable to derive enough theme words from the provided seeds")

    candidate_words = build_candidate_pool(
        seeds=normalized_seeds,
        theme_words=theme_words,
        lexicon=lexicon_words,
        allow_neutral_fill=config.allow_neutral_fill,
    )
    prefix_index = build_prefix_index(lexicon_words)
    grids = search_grids(
        candidate_words=candidate_words,
        prefix_index=prefix_index,
        beam_width=config.beam_width,
        max_candidates=config.max_candidates,
    )
    theme_word_set = set(theme_words)
    ranked = rank_grids(grids, theme_word_set)
    qualified = tuple(
        candidate
        for candidate in ranked
        if _theme_entry_count(candidate.grid, theme_word_set) >= config.min_theme_words
    )
    if not qualified:
        raise ValueError("unable to generate a valid 5x5 puzzle from the provided seeds")
    best_grid = qualified[0].grid
    used_clues: set[str] = set()
    across = make_across_clues(best_grid, clue_bank, used_clues)
    down = make_down_clues(best_grid, clue_bank, used_clues)
    title_seed = _select_title_seed(normalized_seeds, theme_words).upper()
    return Puzzle(
        grid=best_grid,
        across=across,
        down=down,
        theme_words=theme_words,
        title=f"{title_seed} Mini",
    )
