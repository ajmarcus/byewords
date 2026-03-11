# Puzzle UID Plan

## Goal

Give every stored puzzle a public id that is:

- opaque to clients
- short and URL-safe
- stable once assigned
- independent of the current puzzle shape
- easy to keep when storage moves from a file to a database

## Recommended v1

Use a random stored id, not a content-derived id.

For each new puzzle:

1. Generate a `UUIDv7` once.
2. Encode the 128-bit UUID value as a URL-safe base62 string.
3. Use that base62 string as the public puzzle id.
4. Store the id with the puzzle metadata in `puzzles.json`.

Example:

```text
UUIDv7:   019577fd-8d7e-7a3d-9a4b-c2f6b6a1d233
Public id: 2zIa7ZJKCNdqA8J5qu7MZ
```

## Why this is the right tradeoff

- `UUIDv7` gives us a simple unique identifier with useful time ordering.
- Base62 makes the public id shorter than the canonical UUID string.
- The id stays valid if the puzzle format changes later.
- We do not need to freeze a canonical payload format yet.
- We do not need encryption, hashing policy, or key management.

## Storage plan

For v1, keep puzzle records in:

- `src/byewords/data/puzzles.json`

Suggested shape:

```json
{
  "2zIa7ZJKCNdqA8J5qu7MZ": {
    "seed": "doggy",
    "entries": ["fetch", "hound", "pooch", "scent", "treat"]
  }
}
```

Important rule:

- the id represents the stored puzzle record, not a canonical puzzle content hash

That means the same puzzle contents can receive different ids if they are saved as separate records. That is acceptable for this project.

## Operational rules

- Generate the `UUIDv7` exactly once when creating a puzzle record.
- Never regenerate the id from puzzle contents.
- Reuse the stored id on reads and updates.
- Preserve the same public id format if storage moves to a database later.

## What we are explicitly not doing

We are not building ids from fields such as `primary_seed_word` or `grid_letters`.

That approach would force us to define and preserve a canonical puzzle payload too early. It would also add complexity that the current product does not need.

## Summary

The plan is simple:

- assign each new puzzle a `UUIDv7`
- encode it as a URL-safe base62 string
- store that string with the puzzle metadata
- treat that string as the permanent public puzzle id
