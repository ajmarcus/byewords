from collections import defaultdict
from typing import cast

from byewords.grid import GRID_SIZE, grid_columns, has_unique_entries, make_grid, partial_column_prefixes
from byewords.prefixes import has_prefix
from byewords.types import Grid

PositionLetterIndex = tuple[dict[str, frozenset[str]], ...]
PrefixExtensionIndex = dict[str, frozenset[str]]


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


def _build_position_letter_index(words: tuple[str, ...]) -> PositionLetterIndex:
    buckets: list[dict[str, set[str]]] = [defaultdict(set) for _ in range(GRID_SIZE)]
    for word in words:
        for index, letter in enumerate(word):
            buckets[index][letter].add(word)
    return tuple(
        {letter: frozenset(matches) for letter, matches in bucket.items()}
        for bucket in buckets
    )


def _build_prefix_extension_index(
    prefix_index: dict[str, tuple[str, ...]],
) -> PrefixExtensionIndex:
    extensions: PrefixExtensionIndex = {}
    for prefix, words in prefix_index.items():
        if len(prefix) >= GRID_SIZE:
            continue
        extensions[prefix] = frozenset(word[len(prefix)] for word in words)
    return extensions


def _rows_matching_letters(
    candidate_words: tuple[str, ...],
    allowed_letters: tuple[frozenset[str], ...],
    position_letter_index: PositionLetterIndex,
) -> frozenset[str]:
    matching_rows = frozenset(candidate_words)
    if not matching_rows:
        return matching_rows
    constrained_positions: list[tuple[int, frozenset[str]]] = []
    for index, letters in enumerate(allowed_letters):
        rows_for_position = frozenset().union(
            *(position_letter_index[index].get(letter, frozenset()) for letter in letters)
        )
        constrained_positions.append((len(rows_for_position), rows_for_position))
    for _, rows_for_position in sorted(constrained_positions, key=lambda item: item[0]):
        matching_rows &= rows_for_position
        if not matching_rows:
            return frozenset()
    return matching_rows


def _fixed_row_candidates(
    partial_rows: tuple[str, ...],
    candidate: str,
    prefix_index: dict[str, tuple[str, ...]],
    fixed_columns: dict[int, str] | None,
) -> tuple[str, ...]:
    normalized = candidate.lower()
    if normalized in partial_rows:
        return ()
    next_index = len(partial_rows)
    if fixed_columns is not None:
        if any(
            normalized[column_index] != fixed_word[next_index]
            for column_index, fixed_word in fixed_columns.items()
        ):
            return ()
    prefixes = _next_prefixes(partial_rows, normalized)
    if not _is_prefix_compatible(prefixes, prefix_index):
        return ()
    return (normalized,)


def valid_next_rows(
    partial_rows: tuple[str, ...],
    candidate_words: tuple[str, ...],
    prefix_index: dict[str, tuple[str, ...]],
    fixed_rows: dict[int, str] | None = None,
    fixed_columns: dict[int, str] | None = None,
    position_letter_index: PositionLetterIndex | None = None,
    prefix_extension_index: PrefixExtensionIndex | None = None,
) -> tuple[str, ...]:
    next_index = len(partial_rows)
    if fixed_rows is not None and next_index in fixed_rows:
        return _fixed_row_candidates(partial_rows, fixed_rows[next_index], prefix_index, fixed_columns)

    prefixes = partial_column_prefixes(partial_rows)
    if position_letter_index is None:
        position_letter_index = _build_position_letter_index(candidate_words)
    if prefix_extension_index is None:
        prefix_extension_index = _build_prefix_extension_index(prefix_index)

    allowed_letters: list[frozenset[str]] = []
    for column_index, prefix in enumerate(prefixes):
        letters = prefix_extension_index.get(prefix, frozenset())
        if fixed_columns is not None and column_index in fixed_columns:
            letters = letters & frozenset({fixed_columns[column_index][next_index]})
        if not letters:
            return ()
        allowed_letters.append(letters)
    allowed_letters_tuple: tuple[frozenset[str], ...] = tuple(allowed_letters)

    used_rows = set(partial_rows)
    matching_rows = _rows_matching_letters(
        candidate_words,
        allowed_letters_tuple,
        position_letter_index,
    )
    valid_rows: list[tuple[tuple[int, int, tuple[int, ...]], str]] = []
    for candidate in matching_rows:
        if candidate in used_rows:
            continue
        next_prefixes = _next_prefixes(partial_rows, candidate)
        valid_rows.append((_prefix_branching_score(next_prefixes, prefix_index), candidate))
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
    position_letter_index = _build_position_letter_index(ordered_candidates)
    prefix_extension_index = _build_prefix_extension_index(prefix_index)
    found_grids: list[Grid] = []

    def search(partial_rows: tuple[str, ...]) -> None:
        if len(found_grids) >= max_candidates:
            return
        if len(partial_rows) == GRID_SIZE:
            grid = make_grid(cast(tuple[str, str, str, str, str], partial_rows))
            if has_unique_entries(grid) and all(has_prefix(prefix_index, column) for column in grid_columns(grid)):
                found_grids.append(grid)
            return

        next_rows = valid_next_rows(
            partial_rows,
            ordered_candidates,
            prefix_index,
            fixed_rows=fixed_rows,
            fixed_columns=fixed_columns,
            position_letter_index=position_letter_index,
            prefix_extension_index=prefix_extension_index,
        )
        for next_row in next_rows[:beam_width]:
            search(partial_rows + (next_row,))
            if len(found_grids) >= max_candidates:
                return

    search(())
    return tuple(found_grids)
