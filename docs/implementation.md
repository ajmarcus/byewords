# Crossword Generator Layout

## Current status

Completed:
- immutable core types, grid helpers, prefix index, and constrained 5x5 search
- seed normalization and theme expansion from bundled related-word data
- grid scoring and deterministic ranking
- CLI wiring and text rendering
- curated clue-bank clue generation with deterministic fallbacks
- unit coverage for each implemented module plus end-to-end generation tests

Current limitations:
- the bundled lexicon is still tiny and includes weak fill such as `antra` and `udals`
- clue quality is now meaningfully better, but overall puzzle quality is capped by the fill set
- theme expansion is still mostly hand-authored and does not yet support a broad seed vocabulary

## Comparison with `Phil`

[`Phil`](https://github.com/keiranking/Phil) is a general constructor and autofill tool, not a direct peer to this generator. Its public code paths use:

- arbitrary blocked grids rather than a fixed full 5×5
- symmetry-aware pattern generation
- a WebAssembly build of the Glucose SAT solver for autofill
- "quick" solving that can return forced letters before a full fill lands
- a large external word list as the primary quality lever

By contrast, this project currently does:

- fixed-size full-grid search only
- deterministic row-by-row prefix pruning
- scoring after generating candidate fills
- theme expansion before search rather than slot-by-slot fill assistance

That means the correct takeaway is not "replace the solver." The correct takeaway is:

- keep the current specialized search for the default 5×5 path
- borrow Phil's emphasis on constructor-grade word data and debugging feedback
- preserve an abstraction boundary so a blocked-grid solver can be added later without infecting the small core

## What is next

Next priority:
- replace the toy lexicon with a stronger, cleaner 5-letter word list so generated grids contain more clueable entries
- add search diagnostics that explain impossible seeds in terms of dead prefixes and over-constrained positions
- expose optional "forced letter" style hints during debugging, inspired by Phil's quick autofill mode

After that:
- expand the clue bank alongside the better lexicon
- improve theme coverage so more user-supplied seeds lead to non-fallback themed grids
- revisit scoring so familiarity and clueability matter more directly during grid selection
- define a separate solver interface before considering blocked-grid layouts or a SAT/CP backend

## Target shape

A small package with a strict separation between:

- pure domain/data types
- pure generation logic
- lexicon/theme inputs
- rendering/output
- a thin orchestration layer

That keeps almost everything testable without mocks.

## Directory layout

```text
src/byewords/
├── __init__.py
├── cli.py
├── types.py
├── lexicon.py
├── theme.py
├── prefixes.py
├── grid.py
├── search.py
├── score.py
├── clues.py
├── generate.py
├── render.py
└── data/
    ├── words_5.txt
    ├── related_words.json
    ├── clue_bank.json
    └── stopwords.txt

tests/
├── test_cli.py
├── test_lexicon.py
├── test_theme.py
├── test_prefixes.py
├── test_grid.py
├── test_search.py
├── test_score.py
├── test_clues.py
├── test_generate.py
└── fixtures.py
```

## Module responsibilities

### `types.py`

Owns all immutable data structures.

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
    beam_width: int = 100
    min_theme_words: int = 4
    allow_neutral_fill: bool = True
    random_seed: int = 0
```

### `lexicon.py`

Loads and filters word sources.

Responsibilities:
- load base 5-letter lexicon
- normalize casing
- reject junk, proper nouns, rare abbreviations
- optionally attach metadata like familiarity or tags

Core functions:

```python
def load_word_list(path: str) -> tuple[str, ...]: ...
def normalize_word(word: str) -> str | None: ...
def filter_legal_words(words: tuple[str, ...]) -> tuple[str, ...]: ...
def load_related_words(path: str) -> dict[str, tuple[str, ...]]: ...
def load_clue_bank(path: str) -> dict[str, tuple[str, ...]]: ...
```

### `theme.py`

Expands seed words into a ranked candidate pool.

Responsibilities:
- normalize seeds
- derive related words
- rank by closeness to seeds
- split words into themed vs neutral support fill

Core functions:

```python
def normalize_seeds(seeds: tuple[str, ...]) -> tuple[str, ...]: ...
def expand_theme_words(
    seeds: tuple[str, ...],
    related_map: dict[str, tuple[str, ...]],
    lexicon: tuple[str, ...],
) -> tuple[str, ...]: ...

def rank_theme_candidates(
    seeds: tuple[str, ...],
    candidates: tuple[str, ...],
) -> tuple[str, ...]: ...

def build_candidate_pool(
    seeds: tuple[str, ...],
    theme_words: tuple[str, ...],
    lexicon: tuple[str, ...],
    allow_neutral_fill: bool,
) -> tuple[str, ...]: ...
```

### `prefixes.py`

Provides trie-like prefix lookup.

Responsibilities:
- fast prefix pruning during search
- minimal API

Core functions:

```python
def build_prefix_index(words: tuple[str, ...]) -> dict[str, tuple[str, ...]]: ...
def has_prefix(prefix_index: dict[str, tuple[str, ...]], prefix: str) -> bool: ...
def words_with_prefix(
    prefix_index: dict[str, tuple[str, ...]],
    prefix: str,
) -> tuple[str, ...]: ...
```

### `grid.py`

Pure grid operations.

Responsibilities:
- construct/read 5×5 grids
- extract columns
- validate rows/columns
- compute incremental prefixes during row-by-row build

Core functions:

```python
def make_grid(rows: tuple[str, str, str, str, str]) -> Grid: ...
def grid_columns(grid: Grid) -> tuple[str, str, str, str, str]: ...
def partial_column_prefixes(rows: tuple[str, ...]) -> tuple[str, str, str, str, str]: ...
def is_full_grid_valid(grid: Grid, lexicon_set: set[str]) -> bool: ...
def distinct_entries(grid: Grid) -> tuple[str, ...]: ...
def slot_numbers() -> tuple[int, int, int, int, int]: ...
```

For a full 5×5 with no blocks, numbering is fixed:
- across 1–5 = rows top to bottom
- down 1–5 = columns left to right

### `search.py`

Core fill generation.

Responsibilities:
- row-by-row constrained search
- prefix pruning
- ranked search order
- top-k candidate collection
- search diagnostics for failed branches
- optional forced-letter summaries for partial fills

Core functions:

```python
def valid_next_rows(
    partial_rows: tuple[str, ...],
    candidate_words: tuple[str, ...],
    prefix_index: dict[str, tuple[str, ...]],
) -> tuple[str, ...]: ...

def search_grids(
    candidate_words: tuple[str, ...],
    prefix_index: dict[str, tuple[str, ...]],
    beam_width: int,
    max_candidates: int,
) -> tuple[Grid, ...]: ...

def analyze_search_failure(
    candidate_words: tuple[str, ...],
    prefix_index: dict[str, tuple[str, ...]],
) -> SearchDiagnostics: ...
```

Internal helpers:

```python
def _next_prefixes(partial_rows: tuple[str, ...], next_row: str) -> tuple[str, ...]: ...
def _is_prefix_compatible(prefixes: tuple[str, ...], prefix_index: dict[str, tuple[str, ...]]) -> bool: ...
def _forced_letters(partial_rows: tuple[str, ...], prefix_index: dict[str, tuple[str, ...]]) -> tuple[frozenset[str], ...]: ...
def _search_dfs(...): ...
def _search_beam(...): ...
```

Expose one public strategy initially, likely beam search with deterministic ordering.

Phil suggests one implementation detail worth copying here: do not let failure remain opaque. Even if we keep the search simple, we should return enough structured information to show which prefixes died, which columns have only one legal next letter, and whether the candidate pool itself was too narrow.

Suggested diagnostics model:

```python
@dataclass(frozen=True)
class SearchDiagnostics:
    deepest_row: int
    dead_prefixes: tuple[str, ...]
    forced_letters_by_column: tuple[frozenset[str], ...]
    surviving_rows: tuple[str, ...]
```

### `score.py`

Ranks valid grids.

Responsibilities:
- score fill quality
- score theme density
- score diversity / non-duplication
- use lexicon metadata when available
- return decomposed scoring for debugging and tests

Core functions:

```python
def score_fill_quality(grid: Grid) -> float: ...
def score_theme_density(grid: Grid, theme_words: set[str]) -> float: ...
def score_entry_diversity(grid: Grid) -> float: ...
def score_grid(grid: Grid, theme_words: set[str]) -> CandidateGrid: ...
def rank_grids(grids: tuple[Grid, ...], theme_words: set[str]) -> tuple[CandidateGrid, ...]: ...
```

Phil's strongest lesson for scoring is indirect: solver sophistication matters less than word-list quality. `score_fill_quality` should eventually consume entry metadata such as familiarity, clueability, part-of-speech variety, and constructor blacklists so the search does not keep surfacing technically valid but editorially weak fills.

### `clues.py`

Builds clue text from answers plus a curated clue bank.

Responsibilities:
- direct answer-level clues from the bundled bank
- themed standalone clues when appropriate
- fallback clue heuristics
- answer-aware surface realization
- deterministic clue generation

Core functions:

```python
def make_across_clues(
    grid: Grid,
    clue_bank: dict[str, tuple[str, ...]],
) -> tuple[Clue, ...]: ...

def make_down_clues(
    grid: Grid,
    clue_bank: dict[str, tuple[str, ...]],
) -> tuple[Clue, ...]: ...

def clue_for_slot(
    slot: Slot,
    clue_bank: dict[str, tuple[str, ...]],
) -> Clue: ...
```

Internals:

```python
def _best_clue(answer: str, clue_bank: dict[str, tuple[str, ...]]) -> str: ...
def _fallback_clue(answer: str) -> str: ...
```

### `generate.py`

Thin orchestration and public API.

Responsibilities:
- wire the pipeline together
- keep side effects at the boundary
- return a complete `Puzzle`

Core function:

```python
def generate_puzzle(
    seeds: tuple[str, ...],
    lexicon_words: tuple[str, ...],
    related_map: dict[str, tuple[str, ...]],
    clue_bank: dict[str, tuple[str, ...]],
    config: GenerateConfig = GenerateConfig(),
) -> Puzzle: ...
```

Pipeline:

1. normalize seeds
2. expand theme words
3. build candidate pool
4. build prefix index
5. search valid grids
6. score and rank grids
7. generate clues
8. build the final puzzle

If search fails, return structured diagnostics internally and surface a concise explanation at the CLI boundary. That keeps the core pure while giving users the kind of actionable feedback Phil provides during construction.

### `render.py`

Formatting only.

Responsibilities:
- ASCII rendering for tests/debugging
- clue list rendering
- optional JSON export

Core functions:

```python
def render_grid_ascii(grid: Grid) -> str: ...
def render_clues(puzzle: Puzzle) -> str: ...
def render_puzzle_text(puzzle: Puzzle) -> str: ...
def puzzle_to_dict(puzzle: Puzzle) -> dict: ...
```

## Dependency direction

This should stay one-way:

```text
types
  ↑
lexicon   prefixes   grid
   ↑         ↑        ↑
 theme ---- search ---|
   ↑         ↑
   score   clues
      \     /
       generate
          ↑
        render
```

`generate.py` is the composition root.  
`render.py` should never be imported by search or scoring logic.

## Functional style rules

To keep it testable:

- all domain objects immutable
- no hidden globals
- no random calls except through a passed-in seeded RNG or deterministic sort key
- IO only in loader/render entry points
- search, scoring, and clue generation are pure functions

Bad:

```python
WORDS = load_word_list(...)
random.shuffle(candidates)
```

Good:

```python
def rank_candidates(candidates: tuple[str, ...], seed: int) -> tuple[str, ...]: ...
```

Better still: avoid randomness entirely until it is needed.

## Public API surface

Keep it tiny:

```python
from crossword.generate import generate_puzzle
from crossword.render import render_puzzle_text
```

Everything else can remain internal-ish while still being directly testable.

## Test layout

### `test_lexicon.py`
Checks normalization and filtering.

Examples:
- rejects non-alpha
- keeps only 5-letter words
- lowercases consistently

### `test_theme.py`
Checks theme expansion behavior.

Examples:
- seed normalization
- related words intersect with lexicon only
- deterministic ranking

### `test_prefixes.py`
Checks prefix index correctness.

Examples:
- known prefixes resolve
- impossible prefixes fail
- empty prefix supported if chosen

### `test_grid.py`
Checks grid helpers.

Examples:
- row to column conversion
- partial prefixes at each depth
- full validity check

### `test_search.py`
Checks search engine.

Examples:
- finds a valid grid from a known small corpus
- returns empty tuple on impossible corpus
- deterministic candidate ordering

### `test_score.py`
Checks ranking.

Examples:
- theme-dense grid outranks bland grid
- duplicate penalty works
- score decomposition is stable

### `test_clues.py`
Checks clue text generation.

Examples:
- answer-specific clues are preferred when available
- clue strings are deterministic
- no answer leakage in clue text unless explicitly intended

### `test_generate.py`
End-to-end.

Examples:
- given fixed seeds + tiny lexicon, produces exact expected puzzle
- output has 5 across and 5 down
- every clue maps to an actual answer

## Minimal first milestone

Before full entertainment logic, build in this order:

1. `types.py`
2. `grid.py`
3. `prefixes.py`
4. `search.py`
5. `score.py`
6. `theme.py`
7. `clues.py`
8. `generate.py`
9. `render.py`

That is the fastest path to a valid, testable core.

## Suggested v1/v2 split

### v1
- full 5×5 no-block grid
- all answers length 5
- seed-based candidate pool
- deterministic prefix-pruned search
- curated clue bank plus deterministic fallbacks

### v2
- optional blocked 5×5 layouts
- richer theme-specific clue generation
- multiple clue styles per answer
- difficulty tuning
- export to web UI / JSON schema

## Concrete file skeleton

```python
# crossword/generate.py

from crossword.types import GenerateConfig, Puzzle
from crossword.theme import normalize_seeds, expand_theme_words, build_candidate_pool
from crossword.prefixes import build_prefix_index
from crossword.search import search_grids
from crossword.score import rank_grids
from crossword.clues import make_across_clues, make_down_clues

def generate_puzzle(
    seeds: tuple[str, ...],
    lexicon_words: tuple[str, ...],
    related_map: dict[str, tuple[str, ...]],
    clue_bank: dict[str, tuple[str, ...]],
    config: GenerateConfig = GenerateConfig(),
) -> Puzzle:
    normalized = normalize_seeds(seeds)
    theme_words = expand_theme_words(normalized, related_map, lexicon_words)
    candidate_pool = build_candidate_pool(
        normalized,
        theme_words,
        lexicon_words,
        allow_neutral_fill=config.allow_neutral_fill,
    )
    prefix_index = build_prefix_index(candidate_pool)
    grids = search_grids(
        candidate_pool,
        prefix_index,
        beam_width=config.beam_width,
        max_candidates=config.max_candidates,
    )
    ranked = rank_grids(grids, set(theme_words))
    best = ranked[0].grid
    across = make_across_clues(best, clue_bank)
    down = make_down_clues(best, clue_bank)

    return Puzzle(
        grid=best,
        across=across,
        down=down,
        theme_words=tuple(theme_words),
        title=" / ".join(normalized),
    )
```

## Recommendation

For the first implementation, keep the entire core under roughly 600 lines excluding tests and data files. This structure is enough to stay clean without becoming framework-heavy.
