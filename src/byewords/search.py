from typing import cast

from byewords.grid import GRID_SIZE, grid_columns, has_unique_entries, make_grid, partial_column_prefixes
from byewords.prefixes import has_prefix, words_with_prefix
from byewords.types import Grid


def _next_prefixes(partial_rows: tuple[str, ...], next_row: str) -> tuple[str, str, str, str, str]:
    return partial_column_prefixes(partial_rows + (next_row,))


def _is_prefix_compatible(
    prefixes: tuple[str, str, str, str, str],
    prefix_index: dict[str, tuple[str, ...]],
) -> bool:
    return all(has_prefix(prefix_index, prefix) for prefix in prefixes)


def _prefix_branching_score(
    prefixes: tuple[str, str, str, str, str],
    prefix_index: dict[str, tuple[str, ...]],
) -> tuple[int, int, tuple[int, ...]]:
    counts = tuple(len(prefix_index[prefix]) for prefix in prefixes)
    return (max(counts), sum(counts), counts)


def valid_next_rows(
    partial_rows: tuple[str, ...],
    candidate_words: tuple[str, ...],
    prefix_index: dict[str, tuple[str, ...]],
    fixed_rows: dict[int, str] | None = None,
    fixed_columns: dict[int, str] | None = None,
) -> tuple[str, ...]:
    used_rows = set(partial_rows)
    valid_rows: list[tuple[tuple[int, int, tuple[int, ...]], str]] = []
    next_index = len(partial_rows)
    row_candidates = candidate_words if fixed_rows is None or next_index not in fixed_rows else (fixed_rows[next_index],)
    for candidate in row_candidates:
        normalized = candidate.lower()
        if normalized in used_rows:
            continue
        if fixed_columns is not None:
            if any(normalized[column_index] != fixed_word[next_index] for column_index, fixed_word in fixed_columns.items()):
                continue
        prefixes = _next_prefixes(partial_rows, normalized)
        if _is_prefix_compatible(prefixes, prefix_index):
            valid_rows.append((_prefix_branching_score(prefixes, prefix_index), normalized))
    valid_rows.sort(key=lambda item: (item[0], item[1]))
    return tuple(row for _, row in valid_rows)


def search_grids(
    candidate_words: tuple[str, ...],
    prefix_index: dict[str, tuple[str, ...]],
    beam_width: int,
    max_candidates: int,
    fixed_rows: dict[int, str] | None = None,
    fixed_columns: dict[int, str] | None = None,
) -> tuple[Grid, ...]:
    ordered_candidates = tuple(dict.fromkeys(word.lower() for word in candidate_words))
    found_grids: list[Grid] = []

    def search(partial_rows: tuple[str, ...]) -> None:
        if len(found_grids) >= max_candidates:
            return
        if len(partial_rows) == GRID_SIZE:
            grid = make_grid(cast(tuple[str, str, str, str, str], partial_rows))
            if has_unique_entries(grid) and all(column in words_with_prefix(prefix_index, column) for column in grid_columns(grid)):
                found_grids.append(grid)
            return

        next_rows = valid_next_rows(
            partial_rows,
            ordered_candidates,
            prefix_index,
            fixed_rows=fixed_rows,
            fixed_columns=fixed_columns,
        )
        for next_row in next_rows[:beam_width]:
            search(partial_rows + (next_row,))
            if len(found_grids) >= max_candidates:
                return

    search(())
    return tuple(found_grids)
