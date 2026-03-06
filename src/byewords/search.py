from typing import cast

from byewords.grid import GRID_SIZE, grid_columns, make_grid, partial_column_prefixes
from byewords.prefixes import has_prefix, words_with_prefix
from byewords.types import Grid


def _next_prefixes(partial_rows: tuple[str, ...], next_row: str) -> tuple[str, str, str, str, str]:
    return partial_column_prefixes(partial_rows + (next_row,))


def _is_prefix_compatible(
    prefixes: tuple[str, str, str, str, str],
    prefix_index: dict[str, tuple[str, ...]],
) -> bool:
    return all(has_prefix(prefix_index, prefix) for prefix in prefixes)


def valid_next_rows(
    partial_rows: tuple[str, ...],
    candidate_words: tuple[str, ...],
    prefix_index: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    used_rows = set(partial_rows)
    valid_rows = []
    for candidate in candidate_words:
        normalized = candidate.lower()
        if normalized in used_rows:
            continue
        prefixes = _next_prefixes(partial_rows, normalized)
        if _is_prefix_compatible(prefixes, prefix_index):
            valid_rows.append(normalized)
    return tuple(valid_rows)


def search_grids(
    candidate_words: tuple[str, ...],
    prefix_index: dict[str, tuple[str, ...]],
    beam_width: int,
    max_candidates: int,
) -> tuple[Grid, ...]:
    ordered_candidates = tuple(dict.fromkeys(word.lower() for word in candidate_words))
    found_grids: list[Grid] = []

    def search(partial_rows: tuple[str, ...]) -> None:
        if len(found_grids) >= max_candidates:
            return
        if len(partial_rows) == GRID_SIZE:
            grid = make_grid(cast(tuple[str, str, str, str, str], partial_rows))
            if all(column in words_with_prefix(prefix_index, column) for column in grid_columns(grid)):
                found_grids.append(grid)
            return

        next_rows = valid_next_rows(partial_rows, ordered_candidates, prefix_index)
        for next_row in next_rows[:beam_width]:
            search(partial_rows + (next_row,))
            if len(found_grids) >= max_candidates:
                return

    search(())
    return tuple(found_grids)
