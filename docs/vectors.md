# Vector Model Migration

## Goal

Keep semantic theme ranking useful without letting vector loading and cosine scoring dominate crossword generation.

## Model Decision

Current bundled model: `BAAI/bge-small-en-v1.5`

Why this is the bundled default now:

- it stays in the same BGE family as the previous large model
- it cuts vector width from `1024` to `384`
- it reduces both cold-load cost and search-time cosine cost without changing the runtime architecture

## Plan

1. Replace the old large bundled table with a smaller BGE-family table.
2. Rebuild the bundled `word_vectors.json`.
3. Re-measure seeded generation on the real runtime path.
4. Keep the repo checks green:
   - `uv run ruff check .`
   - `uv run ty check`
   - `uv run python -m unittest discover -s tests`
   - `uv run byewords`

## Completed

- Switched [`src/byewords/theme_index_builder.py`](/root/byewords/src/byewords/theme_index_builder.py) defaults from `BAAI/bge-large-en-v1.5` to `BAAI/bge-small-en-v1.5`.
- Regenerated [`src/byewords/data/word_vectors.json`](/root/byewords/src/byewords/data/word_vectors.json) from `BAAI/bge-small-en-v1.5`.
- Updated bundled-data, builder, README, and vector-doc tests for the smaller model.
- Re-ran targeted generation benchmarks after the new bundle landed.

Current bundled vector metadata:

- source: `baai-bge-small-en-v1.5`
- model name: `BAAI/bge-small-en-v1.5`
- dimensions: `384`
- vectors: `4950`
- file size: about `5.6 MB`
- license: `MIT`

## Performance Bottleneck

The switch helped substantially, but the semantic hot path is still single-threaded Python work.

Measured locally on `2026-03-13` with the bundled data for `benchmark_generation(("beach",), ..., runtime_budget_ms=3000)`:

- previous `bge-large` bundle: about `16.4s` cold and about `6.5s` warm
- current `bge-small` bundle: about `4.1s` cold and about `3.6s` warm
- heuristic baseline with semantics disabled: about `0.6s`

What improved:

- no budget fallback in the normal semantic benchmark
- bundled vector size dropped from about `15.1 MB` to about `5.6 MB`
- cold-start loading is much cheaper because there is less JSON and fewer numeric components to parse
- each cosine score is cheaper because the vectors are `384` dims instead of `1024`

What still dominates:

1. Search-time semantic reranking in [`src/byewords/search.py`](/root/byewords/src/byewords/search.py)
   - [_candidate_row_score()](/root/byewords/src/byewords/search.py#L199) calls semantic scoring repeatedly while ranking candidate rows
2. Cosine work in [`src/byewords/theme.py`](/root/byewords/src/byewords/theme.py)
   - [_cosine_similarity()](/root/byewords/src/byewords/theme.py#L365) still dominates the semantic branch
   - [SemanticRowOrderingContext.score()](/root/byewords/src/byewords/theme.py#L176), [seed_relevance_scores()](/root/byewords/src/byewords/theme.py#L394), and [diversify_theme_words()](/root/byewords/src/byewords/theme.py#L596) all repeatedly walk vector components in Python

Warm-profile snapshot after the switch:

- `_cosine_similarity()` was still one of the largest cumulative-cost functions
- the recursive search and semantic reranking remain single-threaded
- the model swap fixed the worst regression, but not the structure of the hot path

## Root Cause Summary

The semantic path is still expensive because generation does all of this in-process and in Python:

1. score the lexicon against the seed
2. rerank candidate rows during recursive search
3. recompute pairwise similarity while choosing the final theme subset

Smaller vectors reduced the constant factor. They did not change the algorithmic shape.

## Research Notes

Primary sources reviewed on `2026-03-13`:

- BAAI model cards for the English `v1.5` family:
  - `BAAI/bge-large-en-v1.5`: <https://huggingface.co/BAAI/bge-large-en-v1.5>
  - `BAAI/bge-base-en-v1.5`: <https://huggingface.co/BAAI/bge-base-en-v1.5>
  - `BAAI/bge-small-en-v1.5`: <https://huggingface.co/BAAI/bge-small-en-v1.5>
- Sentence Transformers quantization docs:
  - <https://www.sbert.net/docs/package_reference/sentence_transformer/quantization.html>
- Sentence Transformers model API docs:
  - <https://www.sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html>
  - <https://www.sbert.net/docs/package_reference/util.html>

Relevant takeaways:

- `small` uses `384` dims, `base` uses `768`, and `large` uses `1024`
- lower-precision and smaller vectors help memory footprint and retrieval speed
- truncation exists, but it is a quality tradeoff and not the first thing to reach for here

## Best Candidates

### 1. Keep `bge-small-en-v1.5` as the bundled default

Why it is the right current choice:

- it removed the most serious regression without invasive runtime changes
- it keeps the same builder flow and review corpus
- it is good enough to avoid semantic budget fallback in the main benchmark path

### 2. Add pairwise cosine caching in the runtime semantic layer

Why it is next:

- the hot path still spends a lot of time recomputing the same word-pair similarities
- that is especially visible in search-time reranking
- it directly addresses the single-threaded pure-Python bottleneck the current profile still shows

### 3. Replace JSON vectors with a packed runtime format

Why it still matters:

- cold load is better now, but JSON decoding and tuple construction are still unnecessary overhead
- a binary payload with precomputed norms would reduce startup cost for any future model

### 4. Only consider `bge-base-en-v1.5` if retrieval quality evidence pushes upward

Why it is lower priority now:

- the runtime pain came from vectors being too large, not too weak
- moving back up to `768` dims would spend more CPU before there is evidence that `small` is insufficient

## Recommendation

Best next sequence from here:

1. Keep the bundled model on `BAAI/bge-small-en-v1.5`.
2. Add a runtime cache for semantic pair scores so repeated `_cosine_similarity()` calls stop recomputing identical pairs.
3. If startup still matters, move `word_vectors.json` to a packed numeric format with precomputed norms.
4. Re-run retrieval and intrusion review before considering a larger model again.

## Not Completed Yet

- The semantic reranking path is still single-threaded.
- `_cosine_similarity()` is still a major runtime cost inside semantic search ordering.
- The bundled vectors still use JSON rather than a packed runtime format.
- Pairwise similarity caching has not been implemented yet.
