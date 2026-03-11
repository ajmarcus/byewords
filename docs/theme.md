# Semantic Theme Generation Plan

## Goal

Build a semantic theme system that improves puzzle quality without breaking the generator's practical constraints.

Required outcomes:

- fill quality is a hard requirement
- entry uniqueness is a hard requirement
- answer/theme coherence is optimized only after those hard constraints are met
- clue generation is a downstream evaluation step, not part of the search hot path
- seeded runtime generation should still target under 1 second wall clock

## Decision summary

The current best plan is:

- store one semantic vector for every bundled word
- keep the vector backend pluggable, but start with a compact offline-built word embedding table
- rank the full lexicon for the seed on demand
- keep the full lexicon available to the existing search index
- treat only a bounded subset of answers as theme-bearing answers and allow the rest to remain bridge fill
- use semantic scores to order viable next rows, not to pre-prune the lexicon aggressively
- reject weak fills before any semantic or clue-based promotion
- use weakest-link coherence and diversity-aware reranking for completed grids
- use an offline batch pipeline to generate and curate puzzles for the full lexicon
- generate clues only for the top 100 answer-only puzzles

This plan is based on the experiments performed so far.

## Current implementation progress

Stage 1 has started.

Implemented now:

- `benchmark_generation()` mirrors the current generation path and records per-window search attempts
- `SearchStatsSnapshot` captures deterministic counters without mutating the runtime search behavior
- regression tests cover seeded, unseeded, and demo-grid benchmark cases
- Stage 2 vector loading now exists with a bundled `word_vectors.json` table and deterministic whole-lexicon ranking APIs
- the CLI now supports an offline-first cache build that writes lexicon-wide puzzle records to `src/byewords/data/puzzles.json`
- clue regeneration can now be forced from both the Groq clue tool and the main puzzle CLI, but clue quality is still intentionally out of the ranking path until the top-100 offline stage

What remains before Stage 1 is fully closed:

- define the representative easy, medium, and hard seed corpus explicitly in code or fixtures
- add a lightweight manual-review corpus for checking whether the current heuristic fill produces plausible theme-bearing answers

Working conclusion:

- the existing search counters appear sufficient for the semantic rollout
- the next implementation chunk should focus on vector data loading and full-lexicon ranking rather than inventing more legality telemetry

## Findings from experiments

Two observations changed the design.

### 1. Search, not vector math, is the expensive part

The bundled lexicon has `4,950` words.

Scoring every word against one seed is cheap:

- with `300` dimensions, one ranking pass is `4,950 * 300 = 1,485,000` multiply-accumulate terms

That is small compared with recursive grid search.

### 2. Hard semantic windows are too brittle

The current generator already has slow seeds:

- `generate_puzzle((), ...)` took about `3.64 s`
- `generate_puzzle(("snail",), ...)` took about `0.61 s`
- `generate_puzzle(("doggy",), ...)` took about `7.83 s`

A direct seeded search experiment for `doggy` with truncated candidate pools found no puzzle at:

- `64`
- `128`
- `256`
- `512`

That does not prove semantic ranking is wrong. It does show that hard top-K candidate truncation is the wrong control point for this problem. Crossword fill needs legally useful bridge words, not just semantically close words.

## Findings from literature

The academic literature changes the plan in four important ways.

### 1. Theme quality is a set property, not just a nearest-neighbor property

Research on automatic word-puzzle generation and topic evaluation shows that related-word quality depends on the whole set, not only on each member's closeness to the seed.

- Pinter et al. build puzzle sets from topic dictionaries and explicitly filter for set consistency rather than simple pairwise retrieval.
- Newman et al. show that coherence metrics correlate with human judgments better when the topic is evaluated as a group.
- Bhatia et al. show that coherence alone is not enough; intrusion-style evaluation catches weak or overly generic sets that still look superficially coherent.

Implications for this project:

- score completed grids on a bounded theme-bearing subset rather than on all 10 answers equally
- gate themed grids on weakest-link coherence, not only mean similarity
- add an offline intrusion-style evaluation harness that inserts one unrelated answer into a candidate theme set and checks whether the scorer rejects it

### 2. Relevance needs diversity and outlier control

Embedding retrieval literature is clear that raw cosine ranking is useful but incomplete.

- Carbonell and Goldstein's Maximal Marginal Relevance (MMR) shows that reranking should balance query relevance against novelty.
- Santus et al. show that a rank-based similarity metric matches cosine on similarity estimation and does better on outlier detection and clustering.
- Reimers and Gurevych show that precomputed embeddings are the right shape for fast similarity search.

Implications for this project:

- use precomputed vectors in the runtime path
- add an MMR-style reranker for selecting theme-bearing answers so the system does not collapse into synonym piles or tiny morphological variants
- benchmark cosine against a lightweight rank-overlap secondary score before freezing the scoring function

### 3. Constraint-first compilation still wins

Crossword-generation literature, old and new, points in the same architectural direction.

- Smith and Steen used bit lists and heuristic tree search for practical crossword compilation.
- XENO supported thematic puzzles by accepting keywords while still using tree search for legality.
- Wilson found integer programming informative but still concluded that simpler compilation approaches were preferable in practice.
- Majima and Ishihara model topic inclusion as an optimization objective and report that themed generation stays feasible even when only some answers are topic-derived.

Implications for this project:

- semantics remains a soft optimization layer over legality search
- the search index should still see the full lexicon
- the plan should target a bounded number of theme-bearing answers, not force all answers to be on-theme

### 4. Clue generation should remain a separate quality stage

Recent crossword-clue literature focuses on separate clue datasets and separate clue evaluation.

- Clue-Instruct builds a dedicated educational clue dataset and evaluates clue quality independently.
- ArabIcros similarly treats clue generation and clue quality control as their own subsystem.

Implications for this project:

- clue generation should stay downstream from answer-only ranking
- clue quality should never rescue a weak fill or a weak theme
- the clue stage should gain its own regression set and human spot-check process

## Chosen design

### Semantic data

Store one vector for every bundled word.

Suggested file:

- `src/byewords/data/word_vectors.json`

Suggested logical structure:

```json
{
  "version": 1,
  "source": "offline-embedding-build",
  "dimensions": 128,
  "lexicon_hash": "abcd1234ef567890",
  "vectors": {
    "doggy": [12, -8, 4, 19, 3],
    "hound": [14, -7, 5, 22, 1]
  },
  "quantization": {
    "scheme": "int8",
    "scale": 0.03125
  }
}
```

Rules:

- every key must be a legal bundled 5-letter word
- vector dimensionality is fixed for the whole file
- vectors are computed offline and bundled with the repo
- runtime ranking uses only bundled vectors
- the loader should expose the vector source name so different offline encoders can be compared without changing runtime APIs

### Theme-bearing answers

The generator should distinguish between:

- theme-bearing answers: the small subset that carries the seed concept
- bridge fill: legal entries needed to make the grid compile cleanly

For a 5x5 full grid, the answer-only scorer should usually treat only the best `3` to `4` non-seed answers as theme-bearing. The rest of the grid should be judged almost entirely on fill quality and legality.

This is the main change suggested by the literature. A good themed mini does not require all 10 entries to be semantically close to the seed.

### Similarity and diversification

Runtime ordering should start simple and stay cheap.

Suggested components:

- `seed_relevance(word)`: maximum or mean similarity to the normalized seeds
- `theme_novelty(word, selected)`: penalty for being too close to already selected theme-bearing answers
- `theme_set_coherence(theme_words)`: minimum pairwise similarity among the selected theme-bearing answers

Suggested MMR-style reranker:

```text
mmr(word) =
    lambda * seed_relevance(word)
    - (1 - lambda) * max_similarity(word, selected_theme_words)
```

Rules:

- use cosine as the initial fast path
- keep the scorer pluggable so a rank-overlap metric can be tested offline
- hard-reject theme sets whose weakest pair falls below a fixed threshold

### Data size

One vector per word is practical at this lexicon size.

Raw float32 storage:

- `128` dimensions: `2,534,400` bytes, about `2.42 MiB`
- `300` dimensions: `5,940,000` bytes, about `5.66 MiB`

Raw int8 storage:

- `128` dimensions: about `0.60 MiB`
- `300` dimensions: about `1.42 MiB`

The logical design should assume one vector per word. The physical format can start as JSON and move to a compact binary format later if parsing time matters.

## Runtime model

The runtime path should be:

1. normalize the supplied seeds
2. fail fast if any normalized seed is missing from the bundled lexicon
3. load the bundled vector table
4. compute a seed score for every bundled word
5. build or reuse the normal search index over the full lexicon
6. ask the search index for viable next rows
7. order those viable rows by semantic score first and branching quality second
8. maintain a small candidate set of likely theme-bearing answers while searching
9. stop once a high-quality valid grid is found within budget
10. build clues from bundled data and deterministic fallbacks only

Key principle:

- semantics decides ordering
- the search index decides legality

The runtime path must never require all answers to score highly. It should only prefer rows that improve the eventual theme-bearing subset without starving the legal search.

## Ranking model

### Hard gates

A puzzle is ineligible unless all of the following are true:

- all 10 entries are valid bundled words
- all across and down entries are unique
- fill quality clears a fixed threshold
- duplicate-heavy or awkward fill patterns are rejected

This is a hard filter, not a tie-breaker.

### Theme-bearing subset selection

For every completed legal grid:

1. identify the top `3` to `4` non-seed answers by seed relevance
2. drop candidates that are near-duplicates of each other
3. compute weakest-link coherence on that subset
4. treat the remaining answers as bridge fill

This keeps the semantic objective aligned with crossword practice and with the literature on coherent word sets.

### Answer-only ranking

Among puzzles that pass the hard gates, rank by:

1. theme-bearing subset passes the weakest-link coherence threshold
2. answer/theme coherence
3. theme diversity after MMR-style reranking
4. secondary fill-quality tie-breakers

Practical interpretation:

- a semantically strong but ugly grid is rejected
- a legal but repetitive grid is rejected
- among strong fills, prefer the grid whose theme-bearing answers are jointly coherent and not redundant

### Clue-stage ranking

After answer-only ranking:

1. first use `clue_bank.json` while generating and storing candidate puzzles for all bundled seed words
2. after the full seed corpus has been generated, choose the best 100 puzzles based only on their answers
3. regenerate clues for those 100 puzzles
4. only then rerank those 100 using clue quality and clue/theme coherence

Clue quality may affect the final ordering of already-strong puzzles. It must not rescue a weak grid.

Final ranking order:

1. hard requirements: fill quality and uniqueness
2. answer/theme coherence
3. clue quality and clue/theme coherence

## Offline batch pipeline

If runtime generation is fast enough, the preferred long-term system is an offline lexicon-wide build.

The batch process should:

1. iterate over every bundled word as a seed
2. generate candidate puzzles for that seed in parallel
3. store the generated puzzles in `puzzles.json` together with their seed metadata
4. reject any candidate that fails fill quality or uniqueness requirements
5. select the theme-bearing subset for each surviving grid
6. reject any surviving grid whose weakest-link coherence is too low
7. rank surviving puzzles by answer/theme coherence and diversity-aware theme quality
8. run an intrusion-style evaluation corpus against the scorer
9. retain the best answer-only puzzles per seed
10. after the full-seed batch is complete, regenerate clues only for the top 100 puzzles
11. only after regeneration, evaluate clue quality and store the final curated winners and their scoring metadata

This work is CPU-bound and should use process-based parallelism rather than threads.

## Runtime budget

The generator should still enforce a strict budget.

Recommended target:

- `<= 750 ms` for semantic ranking plus search work
- `<= 900 ms` for the full runtime generation path

If the budget is exhausted:

- return the best valid grid found so far, if any
- otherwise fall back to the existing heuristic ordering

## Expected performance and optimizations

The semantic pipeline should be fast enough for seeded interactive generation if it keeps vector work linear and search work tightly pruned.

Expected steady-state runtime for one seeded request:

- seed normalization and lexicon validation: `<= 1 ms`
- vector table load from a warm process cache: `<= 5 ms`
- full-lexicon seed scoring: `<= 10 ms`
- viable-row ordering overhead during search: `<= 25 ms`
- legality search and completed-grid scoring: `<= 700 ms`
- total seeded runtime path: `<= 900 ms`

Expected cold-start runtime:

- JSON vector load and decode may add `10` to `40 ms`
- the first search-index build may dominate the request if the cache is empty

The implementation should optimize for the warm path because that is the normal CLI and batch-generation regime.

Required optimizations:

- cache the loaded vector table and any normalized vector norms at process scope
- precompute seed-to-word scores once per request, not inside recursive search
- reuse the existing `SearchIndex` and prefix structures instead of creating semantic-specific candidate pools
- keep semantic ordering as a sort key on already-legal viable rows rather than as a separate filtering pass
- cap the provisional theme-bearing subset to a very small fixed size
- compute weakest-link coherence only for completed grids or very late-stage candidates
- use integer or quantized vector storage if JSON parse or memory pressure becomes noticeable
- preserve deterministic counters for states visited, candidate rows ranked, and semantic reranks

Optional optimizations to evaluate only if needed:

- switch `word_vectors.json` to a compact binary format
- precompute a top-neighbor table per lexicon word for offline curation
- add per-depth expansion caps on the semantic reranker for hard seeds
- keep a small beam of high-quality completed grids instead of exhaustively rescoring every terminal grid
- memoize theme-subset scoring for repeated answer sets in offline batch generation

The key constraint is that optimizations must reduce search work or semantic overhead without changing legality, determinism, or the hard fill-quality gates.

## Unknowns that matter

These unknowns should be resolved early because they affect the rest of the plan:

- whether JSON load time for `word_vectors.json` is negligible or whether a binary format is needed immediately
- which embedding family is best for short single-word entries in this lexicon
- whether cosine is enough or whether a rank-overlap secondary metric improves coherence and outlier rejection
- what weakest-link threshold removes bad themes without destroying yield
- what target count of theme-bearing answers is best for a 5x5 grid
- whether semantic ordering of viable rows materially improves answer/theme coherence without increasing visited states too much
- what fill-quality threshold actually rejects bad grids without collapsing yield
- whether the runtime generator can reliably stay under budget for difficult seeds

## Progressive implementation plan

Implementation should proceed in stages, with explicit decision points after each stage.

### Stage 1. Baseline measurement

Implement:

- a benchmark harness for seeded generation
- deterministic counters for search work under a fixed corpus
- a small corpus of representative easy, medium, and hard seeds
- an offline seed-to-theme evaluation set for manual review

Status update:

- done: benchmark harness for seeded generation
- done: deterministic search counters are now captured as immutable benchmark snapshots
- remaining: seed corpus selection
- remaining: manual review fixtures for answer/theme plausibility

Unknown resolved:

- current runtime and search-state distribution
- which seeds produce plausible theme-bearing answers under the current heuristics

Decision gate:

- if the tail is already too large, prioritize search control before semantic scoring

### Stage 2. Vector table and lexicon ranking

Implement:

- offline vector build tool
- bundled `word_vectors` loader
- whole-lexicon seed ranking
- cosine and rank-overlap offline comparison
- tests for deterministic loading and ranking

Unknown resolved:

- vector format, size, load cost, and baseline retrieval quality

Decision gate:

- if JSON load cost is noticeable, switch the storage format before touching search

### Stage 3. Seed-aware search ordering

Implement:

- semantic row scores in the search path
- ordering of viable rows by semantic score then branching score
- lightweight MMR-style novelty penalties for provisional theme-bearing answers
- counters that compare visited states before and after the change

Unknown resolved:

- whether semantic ordering improves answer coherence without exploding search work

Decision gate:

- if visited states rise too much, add tighter per-depth expansion caps before moving on

### Stage 4. Quality gates and answer-only scoring

Implement:

- explicit fill-quality thresholding
- uniqueness gating
- theme-bearing subset selection
- weakest-link coherence scoring
- diversity-aware answer/theme coherence scoring for completed grids

Unknown resolved:

- which thresholds eliminate bad fills without eliminating too many good puzzles
- whether theme-bearing subset selection is stable across seeds

Decision gate:

- if yield becomes too low, revise the fill thresholds and theme-bearing count before adding clue work

### Stage 5. Runtime path

Implement:

- budgeted seeded runtime generation
- fallback behavior for hard seeds
- regression tests for bounded work and deterministic behavior
- telemetry for how often the runtime path finds enough coherent theme-bearing answers

Unknown resolved:

- whether the seeded runtime path can reliably meet the sub-second target

Decision gate:

- if runtime still misses the target on the hard corpus, narrow the product promise and rely more on offline precomputation

### Stage 6. Offline lexicon-wide generation

Implement:

- process-based batch generation for all seeds
- storage of answer-only winners and their metadata
- intrusion-style evaluation of the theme scorer
- reporting on yield, failure modes, and throughput

Unknown resolved:

- whether the full-lexicon batch is operationally practical
- whether the coherence metrics agree with manual review often enough to trust for curation

Decision gate:

- if throughput is poor or the scorer disagrees with manual review too often, optimize search and scoring before adding clue generation

### Stage 7. Top-100 clue stage

Implement:

- clue regeneration for the top 100 answer-only puzzles, using `clue_bank.json` first and Groq only after answer-only selection
- clue validation
- final reranking with clue quality included only after regeneration
- clue-quality regression checks on a small reviewed corpus

Unknown resolved:

- whether clue-based reranking adds signal or only noise

Decision gate:

- if clue reranking is unstable, keep clues as presentation polish rather than a ranking feature

## Module changes

### `src/byewords/theme.py`

Replace letter-overlap ranking with vector-based lexicon ranking plus set-level theme scoring.

Likely API shape:

```python
def normalize_seeds(seeds: tuple[str, ...]) -> tuple[str, ...]: ...

def validate_seed_words(
    seeds: tuple[str, ...],
    lexicon: tuple[str, ...],
) -> tuple[str, ...]: ...

def load_word_vectors(path: str) -> WordVectorTable: ...

def rank_lexicon_for_seed(
    seeds: tuple[str, ...],
    lexicon: tuple[str, ...],
    vectors: WordVectorTable,
    preferred_words: tuple[str, ...] = (),
) -> tuple[str, ...]: ...

def score_word_for_seed(
    word: str,
    seeds: tuple[str, ...],
    vectors: WordVectorTable,
) -> float: ...

def diversify_theme_words(
    ranked_words: tuple[str, ...],
    seeds: tuple[str, ...],
    vectors: WordVectorTable,
    limit: int,
) -> tuple[str, ...]: ...

def score_theme_subset(
    words: tuple[str, ...],
    seeds: tuple[str, ...],
    vectors: WordVectorTable,
) -> ThemeScoreBreakdown: ...
```

### `src/byewords/search.py`

Keep the current legality checks and prefix pruning, but allow viable rows to be ordered by a seed-aware runtime score that can reward probable theme-bearing answers without constraining the legal search space.

### `src/byewords/generate.py`

Change the seeded path so that it:

- validates seeds against the lexicon
- loads the bundled vector table
- ranks the full lexicon on demand
- passes semantic row scores into search ordering
- tracks a provisional theme-bearing subset during search
- enforces a time budget
- falls back cleanly if the budget is exceeded

### `src/byewords/puzzle_store.py`

Add a deterministic puzzle-record store for offline batch output.

It should:

- write lexicon-wide cached puzzle records to `src/byewords/data/puzzles.json`
- preserve a stable puzzle id and canonical UUID per stored record
- support lookups by stored puzzle id for later clue-regeneration work

### `src/byewords/cli.py`

The default CLI should now support two modes:

- with explicit seeds: generate one puzzle, optionally forcing clue regeneration afterward
- with no seeds: build or refresh the offline `puzzles.json` cache for the full bundled lexicon in CPU-sized batches

### `src/byewords/groq_clues.py`

The clue-regeneration CLI should now support:

- `--force` to regenerate clues even when non-generic clue-bank entries already exist
- an optional `<puzzle_uuid>` argument that can resolve answers from `puzzles.json`
- puzzle-level regeneration flows that reuse the same regeneration path as the main CLI

### Offline builder

Add an offline tool such as `src/byewords/theme_index_builder.py`.

It should support vector building first, then lexicon-wide parallel puzzle generation, then the top-100 clue stage, plus coherence and intrusion evaluation reports. It should never run in normal puzzle generation.

## Testing plan

Tests should protect both correctness and decision discipline.

Required coverage:

- deterministic loading of `word_vectors.json`
- deterministic full-lexicon ranking for a seed
- stable identification of theme-bearing answers for a completed grid
- viable-row ordering that respects semantic rank
- hard rejection of puzzles that fail fill quality or uniqueness requirements
- hard rejection of theme subsets that fail weakest-link coherence
- diversity-aware reranking that avoids near-duplicate theme answers
- answer-only ranking ahead of clue-based ranking
- top-100 clue-stage selection happening only after answer-only ranking
- bounded-work regression tests for a fixed benchmark corpus
- intrusion-style evaluation fixtures for the theme scorer

Prefer deterministic counters over wall-clock assertions wherever possible.

## Appendix: Research papers

- [Pinter et al. 2012, "Automated Word Puzzle Generation via Topic Dictionaries"](https://doi.org/10.48550/arXiv.1206.0377): motivates evaluating coherent word sets rather than isolated nearest neighbors.
- [Newman et al. 2010, "Automatic Evaluation of Topic Coherence"](https://aclanthology.org/N10-1012/): supports group-level coherence metrics as a proxy for human judgment.
- [Bhatia et al. 2018, "Topic Intrusion for Automatic Topic Model Evaluation"](https://aclanthology.org/D18-1098/): motivates intrusion-style evaluation so generic or weak themes are penalized.
- [Carbonell and Goldstein 1997, "The Use of MMR and Diversity-Based Reranking in Document Reranking and Summarization"](https://kilthub.cmu.edu/articles/journal_contribution/The_Use_of_MMR_and_Diversity-Based_Reranking_in_Document_Reranking_and_Summarization/6610814): motivates balancing theme relevance against redundancy.
- [Santus et al. 2018, "A Rank-Based Similarity Metric for Word Embeddings"](https://aclanthology.org/P18-2088/): motivates benchmarking cosine against rank-overlap for outlier rejection and clustering quality.
- [Reimers and Gurevych 2019, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"](https://aclanthology.org/D19-1410/): supports precomputed embedding retrieval as a practical fast path, while keeping the encoder backend swappable.
- [Smith and Steen 1981, "A prototype crossword compiler"](https://academic.oup.com/comjnl/article/24/2/107/338073): supports bitset-heavy heuristic search as a practical legality engine.
- [Smith 1983, "XENO: Computer-Assisted Compilation of Crossword Puzzles"](https://doi.org/10.1093/comjnl/26.4.296): directly supports keyword-driven thematic puzzles layered on top of tree search.
- [Wilson 1989, "Crossword Compilation Using Integer Programming"](https://academic.oup.com/comjnl/article-abstract/32/3/273/331526): supports using optimization ideas as guidance without replacing simpler search-first compilation.
- [Majima and Ishihara 2023, "Generating News-Centric Crossword Puzzles As A Constraint Satisfaction and Optimization Problem"](https://arxiv.org/abs/2308.04688): supports treating topic inclusion as a soft objective and not assuming that every answer must be topic-derived.
- [Zugarini et al. 2024, "Clue-Instruct: Text-Based Clue Generation for Educational Crossword Puzzles"](https://aclanthology.org/2024.lrec-main.297/): supports keeping clue generation and clue evaluation as a distinct stage.
- [Zeinalipour et al. 2023, "ArabIcros: AI-Powered Arabic Crossword Puzzle Generation for Educational Applications"](https://aclanthology.org/2023.arabicnlp-1.23/): supports explicit clue-quality control and separate clue-generation infrastructure.
