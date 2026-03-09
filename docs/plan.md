# Crossword Generator Plan

## Goal

Build a **5×5 crossword puzzle creator in Python** that takes a few seed words and programmatically generates a highly entertaining puzzle with related words and sharp, answer-specific clues.

The implementation should be:

- simple
- minimal-dependency
- functional in style
- easy to test thoroughly

## What the generator needs to solve

A good 5×5 mini is not just “find 10 words that cross.” It must solve four distinct problems:

1. **Grid fill**  
   The across and down entries must satisfy all crossing constraints simultaneously. In the general case, crossword filling is combinatorial and is commonly modeled as a constraint-satisfaction problem. General crossword generation is treated as intractable in the worst case, so heuristics matter.

2. **Theme coherence from seed words**  
   Given a few seeds, the system must expand them into a tight semantic neighborhood rather than a random bag of fill.

3. **Entertaining clue writing**  
   Clues should feel lively rather than dictionary-like. Clue generation is its own problem, separate from grid fill.

4. **Clueability and surface quality**  
   The fill must support concise, lively clues rather than generic dictionary-like prompts.

## Constraints worth adopting

For a satisfying American-style mini:

- every letter should be checked by both an across and a down answer
- entries should generally be at least 3 letters
- the grid should be fully connected
- rotational symmetry is common, but for a 5×5 mini it can be relaxed if it hurts quality

For a **5×5**, the cleanest format is often a **full grid with no blocks**, which gives exactly **5 across + 5 down**, with every cell checked automatically.

## Alternatives and tradeoffs

### Option A: General blocked-grid crossword generator

Model arbitrary black-square patterns, generate slots, then fill with backtracking/CSP.

**Pros**
- more flexible
- can avoid hard full-grid fills

**Cons**
- more moving parts
- harder to keep code minimal
- harder to keep clue quality systematic
- overkill for a 5×5

### Option B: Exact-cover / DLX fill engine

Encode fill as exact cover and solve with Algorithm X / DLX.

**Pros**
- elegant for pure fill problems
- fast when the formulation is right

**Cons**
- awkward once soft objectives matter: theme quality, clue quality, fun, word quality
- more complex than needed here

### Option C: Stochastic / hill-climbing fill

Try many candidate fills, mutate, keep high-scoring boards.

**Pros**
- flexible for optimizing “fun”
- supports soft objectives naturally

**Cons**
- less deterministic
- harder to test cleanly
- can be flaky on small corpora

### Option D: Word-square-style full-grid generator with constrained search

Treat the puzzle as a 5×5 letter matrix where each row is an across answer and each column is a down answer. Build it incrementally using prefix indexes and a scoring function.

**Pros**
- best fit for 5×5
- small code footprint
- naturally functional
- easy to test
- easy to attach across/down clue relationships
- strong pruning via prefix constraints

**Cons**
- needs a good 5-letter lexicon
- full-grid fills can fail if the themed candidate pool is too narrow

### Comparison: `Phil`

[`Phil`](https://github.com/keiranking/Phil) is solving a different problem.
It is a general crossword-construction tool for arbitrary blocked grids, partial fills, and interactive autofill. Its README and source show a browser UI that sends the current grid plus a word list into a WebAssembly build of the Glucose SAT solver (`xw_worker.js`), and a quick mode that can surface forced letters before a full fill is found.

That differs materially from this project:

- **Phil**: blocked-grid constructor assistant, arbitrary slot lengths, partial words, symmetry tooling, SAT-backed fill, large custom word list
- **Byewords**: theme-first 5×5 mini generator, no blocks, all entries length 5, deterministic prefix-pruned search, puzzle scoring after fill generation

This distinction matters because SAT is attractive once the grid shape is arbitrary, the fill is partial, and the constructor wants interactive feedback. For a fixed 5×5 full grid, however, row-by-row prefix pruning is the simpler and better fit.

## Best approach

Use **Option D**:

> Generate a 5×5 full grid using prefix-pruned backtracking over 5-letter candidate words, then score candidate fills for theme coherence, word quality, and clueability.

This is the simplest approach that still gives high puzzle quality.

Phil does not change that recommendation. A SAT-backed solver would add substantial machinery, a custom build pipeline, and more opaque debugging without solving the main problem here, which is selecting a strong themed 5-letter pool and ranking entertaining fills.

## Proposed architecture

### 1) Data model

Keep the model tiny and immutable.

```python
Word = str

@dataclass(frozen=True)
class Grid:
    rows: tuple[str, str, str, str, str]

@dataclass(frozen=True)
class Clue:
    answer: str
    text: str
    direction: str
    number: int
    partner: int | None

@dataclass(frozen=True)
class Puzzle:
    grid: Grid
    across: tuple[Clue, ...]
    down: tuple[Clue, ...]
    theme: str
```

All core functions should be pure:

- `expand_theme(seeds, lexicon, associations) -> list[Word]`
- `build_prefix_index(words) -> dict[prefix, tuple[Word, ...]]`
- `generate_candidate_grids(...) -> Iterable[Grid]`
- `score_grid(grid, theme_words, frequencies) -> float`
- `write_clues(grid, theme) -> Puzzle`

### 2) Lexicon strategy

Runtime should have **no heavy dependencies**.

Use:

- a bundled word list of common 5-letter entries
- an optional small metadata file with frequency / familiarity / tags
- an optional seed-to-related-word map derived offline

At runtime, stick to stdlib only:

- `json`
- `dataclasses`
- `functools`
- `itertools`
- `collections`

Crossword quality depends heavily on fill quality, but dependency-heavy NLP at runtime would hurt simplicity and determinism.

### 3) Theme expansion from seed words

Input: a few seeds such as `("beach", "sun")`.

Output: a ranked pool of 5-letter candidate answers such as:

- direct semantic neighbors
- category members
- pop-culture-adjacent words
- playful near-neighbors when appropriate

Best practical design:

- an **offline curated association graph** or small JSON thesaurus-style map
- plus deterministic transforms:
  - singular/plural normalization
  - simple related categories
  - hand-authored “fun” expansions

Do not make online API calls or large embedding dependencies part of the core generator.

### 4) Fill algorithm

This is the core.

#### Representation

Build the grid row by row.

If rows are `r0..r4`, then after placing the first `k` rows, each column has a prefix of length `k`.  
A candidate next row is valid only if, for each column, the new prefix exists in the prefix index of valid 5-letter words.

#### Search

Use depth-first search with pruning:

1. start from a ranked list of likely theme rows
2. at each depth, compute the 5 column prefixes that would result from adding a candidate row
3. reject immediately if any prefix is impossible
4. continue until 5 rows are placed
5. at depth 5, verify that all 5 columns are valid full words and sufficiently distinct

A trie or prefix map is the right lightweight structure here.

#### Heuristics

Use strong ordering:

- prefer rows with rare letters in constrained positions early
- prefer words semantically close to seeds
- penalize repeated stems and dull fill
- reject duplicate across/down answers
- penalize proper nouns unless explicitly allowed

#### Why this beats generic CSP here

A general CSP engine is a valid abstraction, but for a **5×5 full grid**, prefix-pruned row construction is just a specialized CSP with much less machinery and clearer tests.

Phil is useful as a boundary marker:

- once we allow arbitrary black-square patterns, the problem starts looking much more like Phil's SAT-style fill problem
- while we stay with full 5×5 word-square-style generation, the current specialized search remains the right default

### 5) Scoring function

Do not use “first valid fill wins.”  
Use “best entertaining fill wins.”

A weighted score should combine:

#### Fill quality
- commonness / familiarity
- no junk entries
- minimal obscurity
- no duplicates

#### Theme coherence
- semantic closeness to the seed set
- density of themed entries
- at least 6–8 of the 10 entries clearly on-theme

#### Variety
- mix of parts of speech or clue styles
- avoid five bland nouns

#### Cross quality
- reward high-information intersections
- penalize overly repetitive letter patterns

#### Clueability
Reward entries that support lively, fair clues:

- common enough to clue cleanly
- specific enough to avoid dull definitions
- varied in tone and part of speech
- compatible with the theme when possible

### 6) Clue-generation strategy

Use a rule-based system first, with optional richer creativity later.

Templates:

#### A. Straight clue
- “Morning wake-up sound”
- “Cheesy delivery order”

#### B. Themed clue
- “What this beach puzzle has plenty of”

#### C. Specific clue-bank clue
Use a curated answer-level clue whenever possible:

- `snail` -> “Garden crawler with a spiral shell”
- `eases` -> “Takes the edge off”
- `iotas` -> “Tiny traces”

#### D. Playful themed clue
A few clues can acknowledge the mini’s theme without referencing another clue directly:
- “Beachgoer’s slow-moving stowaway, maybe”
- “What a stress-ball session does”

### 7) Failure handling

Full 5×5 themed grids can fail often if the candidate pool is too tight.

So the generator should try, in order:

1. strict theme-only fill
2. theme-heavy fill with a few neutral support words
3. broader candidate pool with the same score function
4. fallback to returning the highest-scoring near-miss diagnostics in tests

The search API should return structured results:

```python
Success(Puzzle)
Failure(reason="insufficient compatible 5-letter candidates", diagnostics=...)
```

That keeps it debuggable and testable.

## Recommended implementation plan

### Phase 1: Core fill engine

Build:

- 5-letter lexicon loader
- prefix index
- pure row-by-row generator
- validation helpers
- deterministic scoring

Deliverable:
- generate valid 5×5 full grids from a supplied candidate set

### Phase 2: Theme expansion

Build:

- seed normalization
- related-word expansion from bundled JSON
- ranking by closeness and word quality

Deliverable:
- candidate pool builder from seed words

### Phase 3: Entertaining clue system

Build:

- curated clue bank
- themed standalone clue writer
- fallback clue heuristics

Deliverable:
- concise, lively clues that stand on their own

### Phase 4: Search + ranking

Build:

- beam search or bounded DFS
- top-k candidate collection
- best-puzzle selection
- search diagnostics that explain why a candidate pool failed
- optional "forced letter" reporting for partial searches and debugging

Deliverable:
- robust generation rather than brittle single-path fill

### Phase 5: Test suite

Test at four levels.

#### Unit tests
- prefix index correctness
- grid validation
- column extraction
- clue-bank lookup and fallback rendering

#### Property-style tests
- every output row is length 5
- every column is length 5
- all rows/columns are valid lexicon entries
- deterministic output under fixed seed

#### Search tests
- can generate from known candidate pools
- fails gracefully on impossible seeds

#### Snapshot tests
- clue text and puzzle rendering for fixed seeds

## Specific design choices

### Keep runtime dependencies at zero
Use stdlib only for the generator.  
If richer theme expansion is needed later, make it an optional offline preprocessing step.

Phil reinforces this tradeoff. Its SAT path depends on a compiled Glucose WebAssembly build, which makes sense for an interactive constructor but is a poor fit for a small pure-Python mini generator unless we expand the scope significantly.

### Prefer full-grid 5×5 over blocked-grid 5×5
For this exact use case, it is simpler and better.

### Use bounded DFS or beam search, not generic DLX
DLX is elegant, but once “entertaining” becomes part of the objective, scored search is the better abstraction.

### Generate clue relations deliberately
Do not hope they emerge by chance from a good fill.

### Bundle a curated 5-letter lexicon
Crossword quality rises or falls with the word list.

This is the clearest lesson from Phil. Its solving workflow assumes a serious constructor word list. Our current algorithm is good enough for 5×5 minis, but it will only be as good as the lexicon and metadata it searches.

### Learn from Phil without copying Phil wholesale

Adopt:

- stronger lexicon curation and per-entry quality metadata
- structured search diagnostics instead of silent failure
- optional "forced" letter or allowed-letter reporting for debugging impossible seeds
- a future escape hatch for blocked-grid support behind a separate solver interface

Avoid for now:

- a SAT solver dependency for the default 5×5 generator
- arbitrary blocked-grid construction in v1
- browser-oriented interactive constructor features that do not improve puzzle quality directly

## Main risks

### 1) Candidate pool too small
Seed-driven themed pools may not contain enough mutually compatible 5-letter words.

Mitigation:
- add neutral support words
- widen semantic radius
- maintain a larger curated common-word fallback pool
- report which column prefixes or positions became impossible so the failure is actionable

### 2) Valid but boring fill
A pure satisfiability solver will happily produce dull boards.

Mitigation:
- require score thresholds
- rank by theme density and clueability

### 3) Clues become repetitive
Template systems can sound canned.

Mitigation:
- use a mix of straight, paired, and playful clue templates
- keep a library of answer-level clues and themed clue variants

### 4) Overly obscure words
This is a classic crossword problem.

Mitigation:
- familiarity score cutoff
- blacklist crosswordese / obscure abbreviations

## Bottom-line recommendation

Use:

> **A full 5×5 word-square-style generator using prefix-pruned backtracking plus a scoring layer for theme coherence and strong standalone clues.**

Steal from Phil at the edges, not at the core:

- better constructor-grade word data
- better diagnostics
- a solver abstraction that could support blocked grids later

Do not replace the core search with SAT unless the product goal changes from "generate themed 5×5 minis" to "assist interactive construction of arbitrary crossword layouts."

It provides the best balance of:

- small code
- no heavy dependencies
- functional design
- strong testability
- good puzzle quality
- direct support for related words and interacting clues
