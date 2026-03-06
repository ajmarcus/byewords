# Issues Summary

## Issues found

- `generate_puzzle()` could accept unsupported seeds and still return an unrelated puzzle with a misleading title.
- `GenerateConfig.min_theme_words` existed but was not enforced, so the generator could produce weakly themed results.
- The CLI allowed positional seeds and `--seed` flags together, which changed seed precedence in a surprising way.
- Rendering, packaged-data loading, and the `python -m byewords` entrypoint had little or no direct test coverage.

## Fixes made

- Added theme validation so puzzle generation now fails when there are not enough usable theme words.
- Enforced `min_theme_words` against the generated grid before selecting a winner.
- Derived the puzzle title from a validated themed seed instead of raw input.
- Made CLI seed usage explicit by rejecting mixed positional and `--seed` forms.
- Added tests for invalid theme input, CLI parsing, rendering helpers, serialization, packaged data loading, and the module entrypoint.

## Future issues to fix

- `GenerateConfig.random_seed` is still unused and should either drive deterministic search behavior or be removed.
- The bundled lexicon is still very small and contains weak fill, which limits puzzle quality.
- Theme expansion and clue coverage are still narrow, so many user seeds will not yet produce strong themed puzzles.
