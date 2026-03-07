from byewords.lexicon import normalize_word


def normalize_seeds(seeds: tuple[str, ...]) -> tuple[str, ...]:
    normalized = []
    for seed in seeds:
        value = normalize_word(seed)
        if value is not None:
            normalized.append(value)
    return tuple(dict.fromkeys(normalized))


def expand_theme_words(
    seeds: tuple[str, ...],
    related_map: dict[str, tuple[str, ...]],
    lexicon: tuple[str, ...],
) -> tuple[str, ...]:
    lexicon_set = set(lexicon)
    expanded: list[str] = []
    for seed in normalize_seeds(seeds):
        if seed in lexicon_set:
            expanded.append(seed)
        expanded.extend(word for word in related_map.get(seed, ()) if word in lexicon_set)
    return rank_theme_candidates(seeds, tuple(dict.fromkeys(expanded)))


def rank_theme_candidates(
    seeds: tuple[str, ...],
    candidates: tuple[str, ...],
) -> tuple[str, ...]:
    normalized_seeds = normalize_seeds(seeds)

    def score(word: str) -> tuple[int, int, int, str]:
        seed_bonus = 1 if word in normalized_seeds else 0
        shared_letters = max(
            (len(set(word) & set(seed)) for seed in normalized_seeds),
            default=0,
        )
        shared_positions = max(
            (sum(letter == seed[index] for index, letter in enumerate(word)) for seed in normalized_seeds),
            default=0,
        )
        return (-seed_bonus, -shared_letters, -shared_positions, word)

    return tuple(sorted(dict.fromkeys(candidates), key=score))


def _seed_neighbor_words(seeds: tuple[str, ...], candidates: tuple[str, ...]) -> tuple[str, ...]:
    normalized_seeds = normalize_seeds(seeds)
    neighbors: list[str] = []
    for candidate in candidates:
        if any(
            len(set(candidate) & set(seed)) >= 2 or
            sum(letter == seed[index] for index, letter in enumerate(candidate)) >= 1
            for seed in normalized_seeds
        ):
            neighbors.append(candidate)
    return tuple(dict.fromkeys(neighbors))


def build_candidate_pool(
    seeds: tuple[str, ...],
    theme_words: tuple[str, ...],
    lexicon: tuple[str, ...],
    allow_neutral_fill: bool,
    preferred_words: tuple[str, ...] = (),
) -> tuple[str, ...]:
    ranked_theme_words = rank_theme_candidates(seeds, theme_words)
    if not allow_neutral_fill:
        return ranked_theme_words
    theme_set = set(ranked_theme_words)
    preferred_set = set(preferred_words)
    preferred_fill = tuple(word for word in lexicon if word not in theme_set and word in preferred_set)
    neutral_words = tuple(word for word in lexicon if word not in theme_set and word not in preferred_set)
    seed_neighbors = _seed_neighbor_words(seeds, neutral_words)
    seed_neighbor_set = set(seed_neighbors)
    remaining_neutral_words = tuple(word for word in neutral_words if word not in seed_neighbor_set)
    return ranked_theme_words + preferred_fill + seed_neighbors + remaining_neutral_words
