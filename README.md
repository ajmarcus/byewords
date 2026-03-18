byewords
========

A 5x5 crossword generator in python. See the docs folder for `plan.md` and `implementation.md`.

Run the CLI with no arguments to build or refresh the offline `puzzles.json` cache for the full bundled lexicon:

```bash
uv run byewords
```

Run the CLI with explicit seed words to generate one puzzle at a time:

```bash
uv run byewords --seed snail
```

Five reliable single-word seeds with end-to-end regression coverage:

```text
beach
ocean
music
piano
tempo
```

Those five are verified against the bundled lexicon in `tests/test_data_files.py`: each seed must generate a real themed puzzle, keep all ten answers distinct, and clear a minimum fill-quality threshold.

If you want to force fresh Groq clues for a generated puzzle, add `--regenerate-clues` to a seeded run.

If you run `uv run byewords-generate-clues`, existing clue-bank entries stay unchanged unless you pass `--force`, which appends freshly generated clues without deleting the existing stored clues for the requested answers.

Created puzzles are cached on disk in `.byewords-cache/` by normalized seed set and generation config, so rerunning the same request reuses the saved puzzle instead of searching again. Set `BYEWORDS_CACHE_DIR` to place the cache somewhere else.

The bundled semantic vectors in `src/byewords/data/word_vectors.json` are quantized embeddings generated offline from `BAAI/bge-small-en-v1.5`. The upstream model is distributed under the `MIT` license.

Bundled data maintenance
------------------------

The bundled lexicon lives in `src/byewords/data/words_5.txt` and acts as the canonical word list for the other bundled data files.

After editing either file, run the maintenance script to keep them consistent:

```bash
uv run python tools/sort_bundled_data.py
```

That script:

- sorts `words_5.txt` in lexicographical order
- sorts and prunes `clue_bank.json` by answer
- sorts and prunes `word_vectors.json` by answer
- refreshes `puzzles.json` so every cached record only references words that still exist in `words_5.txt`
- normalizes and rewrites all bundled data files deterministically

You can also point it at custom files:

```bash
uv run python tools/sort_bundled_data.py \
  --words path/to/words_5.txt \
  --clue-bank path/to/clue_bank.json \
  --vectors path/to/word_vectors.json \
  --puzzles path/to/puzzles.json
```
