from collections import defaultdict
from dataclasses import dataclass
import time
from typing import cast

from byewords.grid import GRID_SIZE, grid_columns, has_unique_entries, make_grid, partial_column_prefixes
from byewords.prefixes import has_prefix
from byewords.theme import SemanticRowOrdering, SemanticRowOrderingContext
from byewords.types import Grid, ProgressCallback, ProgressUpdate

PositionLetterIndex = tuple[dict[str, int], ...]
PrefixExtensionIndex = dict[str, frozenset[str]]
PrefixRowMaskIndex = tuple[dict[str, int], ...]
RowScoreMap = dict[str, float]


@dataclass(frozen=True)
class SearchIndex:
    candidate_words: tuple[str, ...]
    row_bits: dict[str, int]
    all_rows_mask: int
    position_letter_index: PositionLetterIndex
    prefix_extension_index: PrefixExtensionIndex
    prefix_row_mask_index: PrefixRowMaskIndex


@dataclass(slots=True)
class SearchStats:
    states_visited: int = 0
    dead_ends: int = 0
    mask_intersections: int = 0
    candidate_rows_ranked: int = 0
    fixed_row_shortcuts: int = 0
    semantic_reranks: int = 0
    novelty_penalties_applied: int = 0
    budget_exhausted: bool = False

    def snapshot(self) -> "SearchStatsSnapshot":
        return SearchStatsSnapshot(
            states_visited=self.states_visited,
            dead_ends=self.dead_ends,
            mask_intersections=self.mask_intersections,
            candidate_rows_ranked=self.candidate_rows_ranked,
            fixed_row_shortcuts=self.fixed_row_shortcuts,
            semantic_reranks=self.semantic_reranks,
            novelty_penalties_applied=self.novelty_penalties_applied,
            budget_exhausted=self.budget_exhausted,
        )


@dataclass(frozen=True)
class SearchStatsSnapshot:
    states_visited: int = 0
    dead_ends: int = 0
    mask_intersections: int = 0
    candidate_rows_ranked: int = 0
    fixed_row_shortcuts: int = 0
    semantic_reranks: int = 0
    novelty_penalties_applied: int = 0
    budget_exhausted: bool = False


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
    buckets: list[dict[str, int]] = [defaultdict(int) for _ in range(GRID_SIZE)]
    for row_index, word in enumerate(words):
        row_bit = 1 << row_index
        for index, letter in enumerate(word):
            buckets[index][letter] |= row_bit
    return tuple(dict(bucket) for bucket in buckets)


def _build_prefix_extension_index(
    prefix_index: dict[str, tuple[str, ...]],
) -> PrefixExtensionIndex:
    extensions: PrefixExtensionIndex = {}
    for prefix, words in prefix_index.items():
        if len(prefix) >= GRID_SIZE:
            continue
        extensions[prefix] = frozenset(word[len(prefix)] for word in words)
    return extensions


def _build_prefix_row_mask_index(
    prefix_extension_index: PrefixExtensionIndex,
    position_letter_index: PositionLetterIndex,
) -> PrefixRowMaskIndex:
    prefix_row_masks: list[dict[str, int]] = []
    for position_index in range(GRID_SIZE):
        masks_for_position: dict[str, int] = {}
        for prefix, letters in prefix_extension_index.items():
            row_mask = 0
            for letter in letters:
                row_mask |= position_letter_index[position_index].get(letter, 0)
            if row_mask:
                masks_for_position[prefix] = row_mask
        prefix_row_masks.append(masks_for_position)
    return tuple(prefix_row_masks)


def build_search_index(
    candidate_words: tuple[str, ...],
    prefix_index: dict[str, tuple[str, ...]],
) -> SearchIndex:
    ordered_candidates = tuple(dict.fromkeys(word.lower() for word in candidate_words))
    row_bits = {word: 1 << row_index for row_index, word in enumerate(ordered_candidates)}
    position_letter_index = _build_position_letter_index(ordered_candidates)
    prefix_extension_index = _build_prefix_extension_index(prefix_index)
    prefix_row_mask_index = _build_prefix_row_mask_index(
        prefix_extension_index,
        position_letter_index,
    )
    return SearchIndex(
        candidate_words=ordered_candidates,
        row_bits=row_bits,
        all_rows_mask=(1 << len(ordered_candidates)) - 1,
        position_letter_index=position_letter_index,
        prefix_extension_index=prefix_extension_index,
        prefix_row_mask_index=prefix_row_mask_index,
    )


def _matching_row_mask(
    prefixes: tuple[str, str, str, str, str],
    next_index: int,
    search_index: SearchIndex,
    fixed_columns: dict[int, str] | None,
    remaining_rows_mask: int,
    stats: SearchStats | None,
) -> int:
    matching_rows_mask = remaining_rows_mask
    for column_index, prefix in enumerate(prefixes):
        rows_for_prefix = search_index.prefix_row_mask_index[column_index].get(prefix, 0)
        if fixed_columns is not None and column_index in fixed_columns:
            fixed_word = fixed_columns[column_index].lower()
            rows_for_prefix &= search_index.position_letter_index[column_index].get(
                fixed_word[next_index],
                0,
            )
        matching_rows_mask &= rows_for_prefix
        if stats is not None:
            stats.mask_intersections += 1
        if not matching_rows_mask:
            return 0
    return matching_rows_mask


def _iter_masked_rows(rows_mask: int, candidate_words: tuple[str, ...]) -> tuple[str, ...]:
    rows: list[str] = []
    remaining_mask = rows_mask
    while remaining_mask:
        row_bit = remaining_mask & -remaining_mask
        row_index = row_bit.bit_length() - 1
        rows.append(candidate_words[row_index])
        remaining_mask ^= row_bit
    return tuple(rows)


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
            normalized[column_index] != fixed_word.lower()[next_index]
            for column_index, fixed_word in fixed_columns.items()
        ):
            return ()
    prefixes = _next_prefixes(partial_rows, normalized)
    if not _is_prefix_compatible(prefixes, prefix_index):
        return ()
    return (normalized,)


def _candidate_row_score(
    candidate: str,
    row_scores: RowScoreMap | None,
    semantic_context: SemanticRowOrderingContext | None,
    stats: SearchStats | None,
) -> float:
    if semantic_context is None:
        return row_scores.get(candidate, 0.0) if row_scores is not None else 0.0
    semantic_score = semantic_context.score(candidate)
    if stats is not None:
        stats.semantic_reranks += 1
        if semantic_score.redundancy > 0.0:
            stats.novelty_penalties_applied += 1
    return semantic_score.score


def valid_next_rows(
    partial_rows: tuple[str, ...],
    candidate_words: tuple[str, ...],
    prefix_index: dict[str, tuple[str, ...]],
    fixed_rows: dict[int, str] | None = None,
    fixed_columns: dict[int, str] | None = None,
    search_index: SearchIndex | None = None,
    row_scores: RowScoreMap | None = None,
    semantic_ordering: SemanticRowOrdering | None = None,
    stats: SearchStats | None = None,
) -> tuple[str, ...]:
    next_index = len(partial_rows)
    if fixed_rows is not None and next_index in fixed_rows:
        if stats is not None:
            stats.fixed_row_shortcuts += 1
        return _fixed_row_candidates(partial_rows, fixed_rows[next_index], prefix_index, fixed_columns)

    if search_index is None:
        search_index = build_search_index(candidate_words, prefix_index)

    prefixes = partial_column_prefixes(partial_rows)
    semantic_context = semantic_ordering.context(partial_rows) if semantic_ordering is not None else None
    matching_rows_mask = _matching_row_mask(
        prefixes=prefixes,
        next_index=next_index,
        search_index=search_index,
        fixed_columns=fixed_columns,
        remaining_rows_mask=search_index.all_rows_mask & ~sum(
            search_index.row_bits.get(row, 0) for row in partial_rows
        ),
        stats=stats,
    )
    if not matching_rows_mask:
        return ()

    valid_rows: list[tuple[float, tuple[int, int, tuple[int, ...]], str]] = []
    for candidate in _iter_masked_rows(matching_rows_mask, search_index.candidate_words):
        next_prefixes = _next_prefixes(partial_rows, candidate)
        valid_rows.append(
            (
                _candidate_row_score(candidate, row_scores, semantic_context, stats),
                _prefix_branching_score(next_prefixes, prefix_index),
                candidate,
            )
        )
        if stats is not None:
            stats.candidate_rows_ranked += 1
    valid_rows.sort(key=lambda item: (-item[0], item[1], item[2]))
    return tuple(row for _, _, row in valid_rows)


def search_grids(
    candidate_words: tuple[str, ...],
    prefix_index: dict[str, tuple[str, ...]],
    beam_width: int,
    max_candidates: int,
    fixed_rows: dict[int, str] | None = None,
    fixed_columns: dict[int, str] | None = None,
    search_index: SearchIndex | None = None,
    row_scores: RowScoreMap | None = None,
    semantic_ordering: SemanticRowOrdering | None = None,
    stats: SearchStats | None = None,
    progress_callback: ProgressCallback | None = None,
    deadline_monotonic: float | None = None,
) -> tuple[Grid, ...]:
    if search_index is None:
        search_index = build_search_index(candidate_words, prefix_index)
    found_grids: list[Grid] = []
    budget_exhausted = False

    def _budget_reached() -> bool:
        return deadline_monotonic is not None and time.monotonic() >= deadline_monotonic

    def search(partial_rows: tuple[str, ...], remaining_rows_mask: int) -> None:
        nonlocal budget_exhausted
        if budget_exhausted:
            return
        if _budget_reached():
            budget_exhausted = True
            if stats is not None:
                stats.budget_exhausted = True
            return
        if stats is not None:
            stats.states_visited += 1
        if progress_callback is not None and partial_rows:
            progress_callback(
                ProgressUpdate(
                    stage="search",
                    message=f"Locked {len(partial_rows)}/{GRID_SIZE} rows",
                    partial_rows=partial_rows,
                )
            )
        if len(found_grids) >= max_candidates:
            return
        if len(partial_rows) == GRID_SIZE:
            grid = make_grid(cast(tuple[str, str, str, str, str], partial_rows))
            if has_unique_entries(grid) and all(has_prefix(prefix_index, column) for column in grid_columns(grid)):
                found_grids.append(grid)
                if progress_callback is not None:
                    progress_callback(
                        ProgressUpdate(
                            stage="candidate_solution",
                            message="Found a complete candidate grid; continuing search",
                            partial_rows=grid.rows,
                        )
                    )
            else:
                if stats is not None:
                    stats.dead_ends += 1
            return

        next_index = len(partial_rows)
        if fixed_rows is not None and next_index in fixed_rows:
            next_rows = _fixed_row_candidates(
                partial_rows,
                fixed_rows[next_index],
                prefix_index,
                fixed_columns,
            )
            if stats is not None:
                stats.fixed_row_shortcuts += 1
        else:
            prefixes = partial_column_prefixes(partial_rows)
            semantic_context = (
                semantic_ordering.context(partial_rows) if semantic_ordering is not None else None
            )
            matching_rows_mask = _matching_row_mask(
                prefixes=prefixes,
                next_index=next_index,
                search_index=search_index,
                fixed_columns=fixed_columns,
                remaining_rows_mask=remaining_rows_mask,
                stats=stats,
            )
            next_rows = ()
            if matching_rows_mask:
                ranked_rows: list[tuple[float, tuple[int, int, tuple[int, ...]], str, int]] = []
                for candidate in _iter_masked_rows(matching_rows_mask, search_index.candidate_words):
                    row_bit = search_index.row_bits[candidate]
                    next_prefixes = _next_prefixes(partial_rows, candidate)
                    ranked_rows.append(
                        (
                            _candidate_row_score(
                                candidate,
                                row_scores,
                                semantic_context,
                                stats,
                            ),
                            _prefix_branching_score(next_prefixes, prefix_index),
                            candidate,
                            row_bit,
                        )
                    )
                    if stats is not None:
                        stats.candidate_rows_ranked += 1
                ranked_rows.sort(key=lambda item: (-item[0], item[1], item[2]))
                for _, _, candidate, row_bit in ranked_rows[:beam_width]:
                    if _budget_reached():
                        budget_exhausted = True
                        if stats is not None:
                            stats.budget_exhausted = True
                        return
                    search(partial_rows + (candidate,), remaining_rows_mask & ~row_bit)
                    if budget_exhausted or len(found_grids) >= max_candidates:
                        return
                return
        if not next_rows:
            if stats is not None:
                stats.dead_ends += 1
            return
        for next_row in next_rows[:beam_width]:
            if _budget_reached():
                budget_exhausted = True
                if stats is not None:
                    stats.budget_exhausted = True
                return
            next_row_bit = search_index.row_bits.get(next_row, 0)
            search(partial_rows + (next_row,), remaining_rows_mask & ~next_row_bit)
            if budget_exhausted or len(found_grids) >= max_candidates:
                return

    search((), search_index.all_rows_mask)
    return tuple(found_grids)
