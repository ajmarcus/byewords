# Vector Model Migration

## Goal

Replace the old bundled semantic vectors with a newer open embedding model that fits the actual `byewords` workflow:

- offline vector generation
- bundled int8 vectors at runtime
- stdlib-only runtime path
- deterministic tests and cache builds

## Model Decision

Chosen model: `BAAI/bge-large-en-v1.5`

Why this model:

- much newer than ConceptNet Numberbatch
- strong open English embedding model for short text retrieval
- practical to run today from Hugging Face with `sentence-transformers`
- lower integration risk than the newer Qwen embedding stack for this repo

Why not Qwen right now:

- likely stronger on modern embedding benchmarks
- heavier inference stack and more moving pieces for an offline builder
- higher execution risk for this repo's current "generate once, bundle JSON, keep runtime simple" architecture

## Plan

1. Replace the Numberbatch-specific vector builder with an offline transformer-backed builder.
2. Generate bundled vectors from `BAAI/bge-large-en-v1.5`.
3. Update tests and README metadata to match the new source and dimensionality.
4. Run the required checks:
   - `uv run ruff check .`
   - `uv run ty check`
   - `uv run python -m unittest discover -s tests`
   - `uv run byewords`
5. Review any regressions in runtime behavior or retrieval quality.
6. Commit and push once the full required suite passes.

## Completed

- Researched newer open embedding options and rejected ConceptNet Numberbatch for this migration.
- Chose `BAAI/bge-large-en-v1.5` as the implementation target.
- Reworked [`src/byewords/theme_index_builder.py`](/root/byewords/src/byewords/theme_index_builder.py) to build vectors from a transformer model instead of a downloaded Numberbatch file.
- Kept the heavy dependency path offline-only by importing `sentence-transformers` only inside the vector builder.
- Updated [`tests/test_theme_index_builder.py`](/root/byewords/tests/test_theme_index_builder.py) for the new builder path.
- Updated [`tests/test_data_files.py`](/root/byewords/tests/test_data_files.py) for the new bundled metadata.
- Updated [`README.md`](/root/byewords/README.md) to document the new bundled vector source.
- Generated a new [`src/byewords/data/word_vectors.json`](/root/byewords/src/byewords/data/word_vectors.json) from `BAAI/bge-large-en-v1.5`.

Current bundled vector metadata:

- source: `baai-bge-large-en-v1.5`
- model name: `BAAI/bge-large-en-v1.5`
- dimensions: `1024`
- vectors: `4950`
- license: `MIT`

## Not Completed Yet

- Full required check suite has not finished cleanly yet.
- Full `unittest discover` still needs a clean run after clearing stale `uv run` processes.
- `ruff`, `ty`, and `uv run byewords` still need to be run on the final state.
- Retrieval-quality review against the bundled review corpus still needs a clean validation pass with the new vectors.
- Commit and push are still pending.

## Current Status

The migration is partly complete but not shippable yet.

What is working:

- the offline builder now generates bundled vectors from `BAAI/bge-large-en-v1.5`
- the new `word_vectors.json` has already been generated successfully
- targeted builder and documentation tests are passing

Current blocker:

- seeded generation is far too slow with the new `1024`-dimensional vectors in the current runtime path

Measured runtime so far:

- `beach`: about `39.7s`
- `ocean`: about `35.8s`

This is the main reason the work is not ready to commit and push yet.

## Immediate Next Step

Decide between these two paths:

1. optimize the semantic hot path enough to keep `bge-large-en-v1.5`
2. switch to a smaller newer model that still improves on Numberbatch but fits the runtime budget better

## Risks To Watch

- The new bundled JSON is larger than the previous file because the vectors are `1024`-dimensional.
- Search ordering may change because semantic scores now come from a different model family.
- Some seeded generation tests may run slower if the new ranking changes search behavior materially.
