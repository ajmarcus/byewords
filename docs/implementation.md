# Crossword Generator Layout

## Current status

The codebase now implements a complete bundled-data 5x5 mini generator with:

- deterministic seed normalization
- bundled lexicon and clue-bank loading
- candidate-pool ranking from seeds and clue quality
- reusable bitmask-based search indexes
- seeded row and column anchoring
- simple grid scoring
- clue-bank-backed clue writing with fallbacks
- cache-backed CLI generation

The implementation is working and well covered by unit tests.

## Current limitations

The main gaps are product quality, not missing plumbing:

- grid scoring is still shallow
- theme expansion is based on letter overlap rather than a richer semantic source
- bundled fill quality still determines puzzle quality
- search diagnostics are limited to test counters rather than user-facing explanations
- only full 5x5 grids are supported

## Directory layout

```text
src/byewords/
├── __init__.py
├── __main__.py
├── cache.py
├── cli.py
├── clue_bank.py
├── clues.py
├── data_maintenance.py
├── generate.py
├── grid.py
├── groq_clues.py
├── lexicon.py
├── prefixes.py
├── render.py
├── score.py
├── search.py
├── theme.py
├── types.py
└── data/
    ├── clue_bank.json
    └── words_5.txt

tests/
├── fixtures.py
├── test_cli.py
├── test_clues.py
├── test_data_files.py
├── test_data_maintenance.py
├── test_generate.py
├── test_grid.py
├── test_groq_clues.py
├── test_lexicon.py
├── test_prefixes.py
├── test_public_index.py
├── test_render.py
├── test_score.py
├── test_search.py
├── test_theme.py
└── test_wrangler_config.py
```

## Data model

`types.py` defines the core immutable structures:

```python
from dataclasses import dataclass
from typing import Literal

Direction = Literal["across", "down"]

@dataclass(frozen=True)
class Grid:
    rows: tuple[str, str, str, str, str]

@dataclass(frozen=True)
class Slot:
    direction: Direction
    index: int
    answer: str

@dataclass(frozen=True)
class Clue:
    number: int
    direction: Direction
    answer: str
    text: str

@dataclass(frozen=True)
class Puzzle:
    grid: Grid
    across: tuple[Clue, ...]
    down: tuple[Clue, ...]
    theme_words: tuple[str, ...]
    title: str

@dataclass(frozen=True)
class CandidateGrid:
    grid: Grid
    theme_score: float
    fill_score: float
    clue_score: float
    total_score: float

@dataclass(frozen=True)
class GenerateConfig:
    max_candidates: int = 500
    beam_width: int = 25
    allow_neutral_fill: bool = True
    random_seed: int = 0
```

## Module responsibilities

### `lexicon.py`

Responsibilities:

- load the bundled word list
- normalize words to lowercase ASCII alphabetic 5-letter entries
- filter invalid entries
- load the clue bank JSON

Current API:

```python
def load_word_list(path: str) -> tuple[str, ...]: ...
def normalize_word(word: str) -> str | None: ...
def filter_legal_words(words: tuple[str, ...]) -> tuple[str, ...]: ...
def load_clue_bank(path: str) -> dict[str, tuple[str, ...]]: ...
```

### `theme.py`

Responsibilities:

- normalize seed words
- rank theme candidates
- build the search candidate pool

The current theme system is heuristic, not semantic. It prioritizes exact seeds first, then words with shared letters or matching positions, then the rest of the lexicon.

Current API:

```python
def normalize_seeds(seeds: tuple[str, ...]) -> tuple[str, ...]: ...
def rank_theme_candidates(
    seeds: tuple[str, ...],
    candidates: tuple[str, ...],
) -> tuple[str, ...]: ...
def build_candidate_pool(
    seeds: tuple[str, ...],
    theme_words: tuple[str, ...],
    lexicon: tuple[str, ...],
    allow_neutral_fill: bool,
    preferred_words: tuple[str, ...] = (),
) -> tuple[str, ...]: ...
```

### `prefixes.py`

Responsibilities:

- build prefix buckets for the lexicon
- answer prefix-existence checks
- list words matching a prefix

Current API:

```python
def build_prefix_index(words: tuple[str, ...]) -> dict[str, tuple[str, ...]]: ...
def has_prefix(prefix_index: dict[str, tuple[str, ...]], prefix: str) -> bool: ...
def words_with_prefix(
    prefix_index: dict[str, tuple[str, ...]],
    prefix: str,
) -> tuple[str, ...]: ...
```

### `grid.py`

Responsibilities:

- construct and validate 5x5 grids
- derive column words
- derive partial column prefixes
- check entry uniqueness

Important helpers:

```python
GRID_SIZE = 5

def make_grid(rows: tuple[str, str, str, str, str]) -> Grid: ...
def grid_columns(grid: Grid) -> tuple[str, str, str, str, str]: ...
def partial_column_prefixes(rows: tuple[str, ...]) -> tuple[str, str, str, str, str]: ...
def distinct_entries(grid: Grid) -> tuple[str, ...]: ...
def has_unique_entries(grid: Grid) -> bool: ...
```

### `search.py`

Responsibilities:

- precompute reusable candidate search state
- find valid next rows under prefix and fixed-row/fixed-column constraints
- search complete 5x5 grids
- expose deterministic search counters for tests

Important types:

```python
@dataclass(frozen=True)
class SearchIndex:
    candidate_words: tuple[str, ...]
    row_bits: dict[str, int]
    all_rows_mask: int
    position_letter_index: tuple[dict[str, int], ...]
    prefix_extension_index: dict[str, frozenset[str]]
    prefix_row_mask_index: tuple[dict[str, int], ...]

@dataclass(slots=True)
class SearchStats:
    states_visited: int = 0
    dead_ends: int = 0
    mask_intersections: int = 0
    candidate_rows_ranked: int = 0
    fixed_row_shortcuts: int = 0
```

Current API:

```python
def build_search_index(
    candidate_words: tuple[str, ...],
    prefix_index: dict[str, tuple[str, ...]],
) -> SearchIndex: ...

def valid_next_rows(
    partial_rows: tuple[str, ...],
    candidate_words: tuple[str, ...],
    prefix_index: dict[str, tuple[str, ...]],
    fixed_rows: dict[int, str] | None = None,
    fixed_columns: dict[int, str] | None = None,
    search_index: SearchIndex | None = None,
    stats: SearchStats | None = None,
) -> tuple[str, ...]: ...

def search_grids(
    candidate_words: tuple[str, ...],
    prefix_index: dict[str, tuple[str, ...]],
    beam_width: int,
    max_candidates: int,
    fixed_rows: dict[int, str] | None = None,
    fixed_columns: dict[int, str] | None = None,
    search_index: SearchIndex | None = None,
    stats: SearchStats | None = None,
) -> tuple[Grid, ...]: ...
```

### `score.py`

Responsibilities:

- score fill quality by letter diversity and repetition
- rank grids deterministically

The scoring is intentionally simple and does not yet model theme coherence directly.

### `clue_bank.py`

Responsibilities:

- identify overly generic clue patterns
- expose the words whose leading clue looks specific enough to prefer during fill

### `clues.py`

Responsibilities:

- choose a clue for each slot
- avoid reusing clue text within a puzzle
- provide deterministic fallback clue variants

Across and down clues are derived directly from the final grid rows and columns.

### `generate.py`

Responsibilities:

- load bundled default inputs
- build candidate windows
- try seeded searches first
- reuse `SearchIndex` instances across search attempts
- rank candidate grids
- build final `Puzzle` values
- provide cached generation

Notable behavior:

- candidate windows expand through increasing limits
- a built-in demo grid is used for a known good fallback path
- when a seed can be anchored into a result, seeded grids are preferred

### `cache.py`

Responsibilities:

- serialize generated puzzles
- read and write cached puzzle JSON keyed by seeds and config

### `data_maintenance.py`

Responsibilities:

- sort the bundled lexicon
- normalize and prune the clue bank
- drop clue entries whose answers are no longer in the lexicon

This is the source of truth for keeping `words_5.txt` and `clue_bank.json` in sync.

### `cli.py` and `render.py`

Responsibilities:

- parse seed arguments
- call cached generation
- render a printable text form of the puzzle

## Fill pipeline

The actual fill pipeline in the current code is:

1. load the bundled lexicon and clue bank
2. normalize input seeds
3. derive preferred clue-backed words
4. build the candidate pool
5. build a lexicon prefix index
6. build reusable search indexes for several candidate windows
7. search seed-anchored grids when seeds are available
8. fall back to generic search if needed
9. rank resulting grids
10. choose the best seeded result, or the best result overall
11. write clues and return a `Puzzle`

## Data maintenance workflow

When bundled data changes:

1. edit `src/byewords/data/words_5.txt`
2. run `uv run python tools/sort_bundled_data.py`
3. let the script sort the word list and prune orphan clue entries
4. run the test suite

That keeps the word list and clue bank synchronized.

## Testing strategy

The tests currently cover:

- parsing and rendering
- word and clue data integrity
- data maintenance behavior
- theme ranking
- prefix index behavior
- grid validation
- search correctness and search-work bounds
- scoring
- clue generation
- end-to-end puzzle generation

The most important search-regression protection is in `tests/test_search.py`, which checks:

- expected fills on a known corpus
- anchored row and column behavior
- reusable `SearchIndex` support
- bounded search work using `SearchStats`

## Next work

The next implementation steps should follow the actual code shape:

- improve the bundled lexicon and clue bank
- make theme expansion more meaningful than letter-overlap heuristics
- strengthen scoring before final grid selection
- add better diagnostics for failed seed requests
- keep performance tests based on search counters rather than timing
