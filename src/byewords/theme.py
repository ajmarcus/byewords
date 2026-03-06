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


def build_candidate_pool(
    seeds: tuple[str, ...],
    theme_words: tuple[str, ...],
    lexicon: tuple[str, ...],
    allow_neutral_fill: bool,
) -> tuple[str, ...]:
    ranked_theme_words = rank_theme_candidates(seeds, theme_words)
    if not allow_neutral_fill:
        return ranked_theme_words
    theme_set = set(ranked_theme_words)
    neutral_words = tuple(word for word in lexicon if word not in theme_set)
    return ranked_theme_words + neutral_words
