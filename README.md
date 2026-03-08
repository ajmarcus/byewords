byewords
========

A 5x5 crossword generator in python. See the docs folder for `plan.md` and `implementation.md`.

Run the CLI with no arguments to generate any available 5x5:

```bash
uv run byewords
```

Verified single-word seeds that return the bundled starter puzzle immediately:

```text
ozone
liven
inert
verve
ester
olive
zines
overt
nerve
enter
```

You can still pass seed words if you want to nudge the fill toward specific entries:

```bash
uv run byewords --seed snail
```

Created puzzles are cached on disk in `.byewords-cache/` by normalized seed set and generation config, so rerunning the same request reuses the saved puzzle instead of searching again. Set `BYEWORDS_CACHE_DIR` to place the cache somewhere else.

Bundled data maintenance
------------------------

The bundled lexicon lives in `src/byewords/data/words_5.txt` and the bundled clue bank lives in `src/byewords/data/clue_bank.json`.

After editing either file, run the maintenance script to keep them consistent:

```bash
uv run python tools/sort_bundled_data.py
```

That script:

- sorts `words_5.txt` in lexicographical order
- sorts `clue_bank.json` by answer
- removes clue-bank entries whose answers no longer exist in `words_5.txt`
- normalizes and rewrites both files deterministically

You can also point it at custom files:

```bash
uv run python tools/sort_bundled_data.py \
  --words path/to/words_5.txt \
  --clue-bank path/to/clue_bank.json
```
