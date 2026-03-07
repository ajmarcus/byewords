byewords
========

A 5x5 crossword generator in python. See the docs folder for `plan.md` and `implementation.md`.

Known-good seed words that generate a puzzle with the bundled data:

```text
ozone liven inert verve ester
```

Run the CLI with positional seed words:

```bash
uv run byewords ozone liven inert verve ester
```

Or with repeated `--seed` flags:

```bash
uv run byewords --seed ozone --seed liven --seed inert --seed verve --seed ester
```
