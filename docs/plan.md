# Crossword Generator Plan

## Goal

Build a deterministic **5x5 mini crossword generator** in Python that:

- accepts optional seed words
- fills a full 5x5 word-square-style grid
- uses bundled words and clues only
- produces clueable, distinct across and down entries
- stays small, testable, and dependency-light

## Current product shape

The generator today is a fixed-format mini builder:

- grid shape: full 5x5 with no blocks
- entries: 5 across and 5 down
- fill source: bundled `words_5.txt`
- clue source: bundled `clue_bank.json` plus deterministic fallbacks
- interface: CLI entry point `byewords`

This fixed shape is intentional. It keeps the fill problem small enough to solve with specialized search instead of a general blocked-grid engine.

## What the code must do

### 1) Normalize inputs

Seed words are optional. Invalid or non-5-letter inputs are dropped. Valid seeds are normalized to lowercase and deduplicated.

### 2) Build a candidate pool

The candidate pool should prioritize:

- exact seeds
- clue-bank-backed words
- words sharing letters or positions with the seeds
- the rest of the bundled lexicon

This is already how the current code behaves.

### 3) Fill a legal 5x5 grid

The fill engine must produce five row words such that:

- every row is in the candidate set
- every derived column is a valid word in the lexicon prefix index
- all 10 final entries are unique

### 4) Rank candidate grids

The generator should prefer grids with:

- more distinct letters
- fewer repeated letters within entries
- fully distinct across/down answers

The current scoring remains intentionally simple.

### 5) Write clues

Clues come from the bundled clue bank when available. If a clue is missing, the generator falls back to deterministic pattern-based clues such as plural, past-tense, repeated-letter, and first/last-letter descriptions.

## Chosen fill strategy

Use a specialized row-by-row search for a 5x5 full grid.

Why this remains the right fit:

- the search space is small enough to prune aggressively with prefixes
- the code stays much simpler than a general crossword solver
- correctness is easy to test end-to-end
- the grid format guarantees all letters are checked

## Current search design

The shipped search is no longer just naive prefix pruning. It now carries precomputed search state.

### Search index

For each candidate window, the code builds a reusable `SearchIndex` containing:

- normalized candidate words
- a bitmask per row word
- a position-letter index using integer bitmasks
- a prefix-extension index
- a prefix-to-row-mask index for each position

This avoids rebuilding row filters for every recursive branch.

### Search loop

At each depth, the generator:

1. computes the current column prefixes from the placed rows
2. intersects row masks for those prefixes
3. applies any fixed row or fixed column constraint
4. ranks surviving rows by downstream prefix branching score
5. recurses until 5 rows are placed

At the terminal state it verifies:

- all rows and columns are legal words
- all across and down answers are unique

## Current generation flow

`generate_puzzle()` currently works like this:

1. normalize seeds
2. load clue-bank-backed preferred words
3. build the candidate pool
4. build the lexicon prefix index
5. search progressively larger candidate windows
6. try seed-anchored searches before generic fill when seeds are available
7. rank resulting grids
8. choose the best seeded result when possible
9. build across/down clues and return a `Puzzle`

The generator also has:

- a cached wrapper `generate_puzzle_cached()`
- a built-in demo grid for a known-good fallback path

## Constraints to keep

The project should continue to enforce:

- 5-letter alphabetic entries only
- lowercase normalized internal words
- no duplicate across/down answers
- deterministic generation for the same bundled data and config
- stdlib-only runtime behavior

## What is already done

Implemented in the current code:

- immutable core data types
- bundled lexicon and clue-bank loading
- seed normalization
- candidate-pool ranking
- prefix index construction
- bitmask-based reusable search indexes
- seeded row/column anchoring
- simple grid scoring
- clue-bank-backed clue selection with fallbacks
- cache support
- CLI rendering
- data maintenance tooling for sorting and pruning bundled data
- baseline theme benchmarking via generation-path search reports and immutable `SearchStats` snapshots

## What still needs work

Highest-value next steps:

- bundle and validate the semantic vector table described in `docs/theme.md`
- compare whole-lexicon seed ranking quality before changing search ordering
- improve bundled fill quality so more generated grids feel natural
- strengthen scoring to reflect clueability and familiarity, not just letter diversity
- improve theme expansion beyond letter-overlap heuristics
- add better search diagnostics for impossible seeds
- add more regression tests around candidate-pool quality and bundled clue coverage

Theme-specific planning note:

- Stage 1 baseline measurement has started: the generator can now emit deterministic per-window search reports without changing runtime behavior
- the current search counters appear expressive enough for the theme rollout, so Stage 2 should focus on vector data loading and ranking quality rather than adding new legality telemetry

Possible later work:

- richer clue selection heuristics
- optional blocked-grid support behind a separate solver interface
- stronger branch-and-bound pruning if the lexicon grows substantially

## Testing plan

The project should keep protecting both correctness and performance characteristics.

### Correctness tests

Continue unit coverage for:

- lexicon normalization and filtering
- prefix lookup
- grid validation
- search results and anchored fills
- clue generation
- full puzzle generation
- data maintenance

### Ongoing performance tests

Prefer deterministic work-budget assertions over wall-clock timing.

The current search tests already do this with `SearchStats` by checking:

- bounded visited states on a benchmark corpus
- bounded ranked candidate rows under constrained prefixes
- correct behavior with reused `SearchIndex` instances

That should remain the main performance safety net. If the search changes, extend these counters rather than adding flaky timing tests.
