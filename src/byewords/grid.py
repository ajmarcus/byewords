from typing import cast

from byewords.types import Grid

GRID_SIZE = 5


def make_grid(rows: tuple[str, str, str, str, str]) -> Grid:
    if len(rows) != GRID_SIZE:
        raise ValueError("grid must have exactly 5 rows")
    normalized_rows = tuple(row.lower() for row in rows)
    if any(len(row) != GRID_SIZE or not row.isalpha() for row in normalized_rows):
        raise ValueError("grid rows must be 5-letter alphabetic words")
    return Grid(rows=cast(tuple[str, str, str, str, str], normalized_rows))


def grid_columns(grid: Grid) -> tuple[str, str, str, str, str]:
    return (
        "".join(row[0] for row in grid.rows),
        "".join(row[1] for row in grid.rows),
        "".join(row[2] for row in grid.rows),
        "".join(row[3] for row in grid.rows),
        "".join(row[4] for row in grid.rows),
    )


def partial_column_prefixes(rows: tuple[str, ...]) -> tuple[str, str, str, str, str]:
    if len(rows) > GRID_SIZE:
        raise ValueError("partial grid cannot exceed 5 rows")
    normalized_rows = tuple(row.lower() for row in rows)
    if any(len(row) != GRID_SIZE or not row.isalpha() for row in normalized_rows):
        raise ValueError("partial rows must be 5-letter alphabetic words")
    return (
        "".join(row[0] for row in normalized_rows),
        "".join(row[1] for row in normalized_rows),
        "".join(row[2] for row in normalized_rows),
        "".join(row[3] for row in normalized_rows),
        "".join(row[4] for row in normalized_rows),
    )


def is_full_grid_valid(grid: Grid, lexicon_set: set[str]) -> bool:
    entries = distinct_entries(grid)
    return len(set(entries)) == len(entries) and all(entry in lexicon_set for entry in entries)


def distinct_entries(grid: Grid) -> tuple[str, ...]:
    return grid.rows + grid_columns(grid)


def has_unique_entries(grid: Grid) -> bool:
    entries = distinct_entries(grid)
    return len(set(entries)) == len(entries)


def slot_numbers() -> tuple[int, int, int, int, int]:
    return (1, 2, 3, 4, 5)
