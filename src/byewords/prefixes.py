def build_prefix_index(words: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    buckets: dict[str, list[str]] = {"": []}
    for word in words:
        normalized = word.lower()
        buckets[""].append(normalized)
        for prefix_length in range(1, len(normalized) + 1):
            prefix = normalized[:prefix_length]
            buckets.setdefault(prefix, []).append(normalized)
    return {
        prefix: tuple(sorted(dict.fromkeys(matches)))
        for prefix, matches in buckets.items()
    }


def has_prefix(prefix_index: dict[str, tuple[str, ...]], prefix: str) -> bool:
    return prefix.lower() in prefix_index


def words_with_prefix(
    prefix_index: dict[str, tuple[str, ...]],
    prefix: str,
) -> tuple[str, ...]:
    return prefix_index.get(prefix.lower(), ())
