from __future__ import annotations

import hashlib
import json
import math
from itertools import combinations
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from byewords.lexicon import normalize_word
from byewords.types import ThemeScoreBreakdown

DEFAULT_THEME_WORD_LIMIT = 4
DEFAULT_MMR_LAMBDA = 0.8
DEFAULT_REDUNDANCY_THRESHOLD = 0.9
THEME_BENCHMARK_SEEDS = {
    "easy": ("beach", "music", "ocean"),
    "medium": ("snail", "tempo", "water"),
    "hard": ("doggy", "llama", "wharf"),
}


@dataclass(frozen=True)
class ThemeReviewCase:
    seed: str
    expected_related_words: tuple[str, ...]
    note: str


THEME_MANUAL_REVIEW_CASES = (
    ThemeReviewCase(
        seed="beach",
        expected_related_words=("ocean", "waves", "wharf"),
        note="Coastal vocabulary should surface without collapsing into generic filler.",
    ),
    ThemeReviewCase(
        seed="music",
        expected_related_words=("choir", "piano", "tempo"),
        note="Performance and instrument words should outrank unrelated bridge fill.",
    ),
    ThemeReviewCase(
        seed="snail",
        expected_related_words=("shell", "slime", "trail"),
        note="The review set should catch when retrieval drifts away from a concrete organism theme.",
    ),
)


@dataclass(frozen=True)
class WordVectorTable:
    version: int
    source: str
    dimensions: int
    lexicon_hash: str
    quantization_scheme: str
    quantization_scale: float
    vectors: dict[str, tuple[int, ...]]
    norms: dict[str, float]


def normalize_seeds(seeds: tuple[str, ...]) -> tuple[str, ...]:
    normalized = []
    for seed in seeds:
        value = normalize_word(seed)
        if value is not None:
            normalized.append(value)
    return tuple(dict.fromkeys(normalized))


def validate_seed_words(
    seeds: tuple[str, ...],
    lexicon: tuple[str, ...],
) -> tuple[str, ...]:
    normalized = normalize_seeds(seeds)
    lexicon_set = set(lexicon)
    missing = tuple(seed for seed in normalized if seed not in lexicon_set)
    if missing:
        missing_text = ", ".join(word.upper() for word in missing)
        raise ValueError(f"seed words missing from lexicon: {missing_text}")
    return normalized


def lexicon_hash(words: tuple[str, ...]) -> str:
    encoded = "\n".join(words).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _vector_cache_key(path: str) -> tuple[str, int, int]:
    resolved_path = Path(path).resolve()
    stats = resolved_path.stat()
    return (str(resolved_path), stats.st_mtime_ns, stats.st_size)


def load_word_vectors(path: str) -> WordVectorTable:
    return _load_word_vectors_cached(*_vector_cache_key(path))


@lru_cache(maxsize=8)
def _load_word_vectors_cached(
    resolved_path: str,
    modified_time_ns: int,
    file_size: int,
) -> WordVectorTable:
    del modified_time_ns
    del file_size
    raw_payload = json.loads(Path(resolved_path).read_text(encoding="utf-8"))
    if not isinstance(raw_payload, dict):
        raise ValueError("word vector payload must be a JSON object")

    version = raw_payload.get("version")
    source = raw_payload.get("source")
    dimensions = raw_payload.get("dimensions")
    lexicon_signature = raw_payload.get("lexicon_hash")
    quantization = raw_payload.get("quantization")
    vectors = raw_payload.get("vectors")

    if not isinstance(version, int) or version <= 0:
        raise ValueError("word vector version must be a positive integer")
    if not isinstance(source, str) or not source:
        raise ValueError("word vector source must be a non-empty string")
    if not isinstance(dimensions, int) or dimensions <= 0:
        raise ValueError("word vector dimensions must be a positive integer")
    if not isinstance(lexicon_signature, str) or not lexicon_signature:
        raise ValueError("word vector lexicon_hash must be a non-empty string")
    if not isinstance(quantization, dict):
        raise ValueError("word vector quantization must be an object")
    if not isinstance(vectors, dict) or not vectors:
        raise ValueError("word vector table must contain vectors")

    scheme = quantization.get("scheme")
    scale = quantization.get("scale")
    if not isinstance(scheme, str) or not scheme:
        raise ValueError("word vector quantization scheme must be a non-empty string")
    if not isinstance(scale, int | float) or scale <= 0:
        raise ValueError("word vector quantization scale must be positive")

    parsed_vectors: dict[str, tuple[int, ...]] = {}
    norms: dict[str, float] = {}
    for raw_word, raw_vector in vectors.items():
        if not isinstance(raw_word, str):
            raise ValueError("word vector keys must be strings")
        normalized_word = normalize_word(raw_word)
        if normalized_word != raw_word:
            raise ValueError(f"invalid vector word: {raw_word!r}")
        if not isinstance(raw_vector, list) or len(raw_vector) != dimensions:
            raise ValueError(f"vector for {raw_word!r} must contain {dimensions} components")
        vector_components: list[int] = []
        for component in raw_vector:
            if not isinstance(component, int):
                raise ValueError(f"vector for {raw_word!r} must use integer components")
            if component < -127 or component > 127:
                raise ValueError(f"vector for {raw_word!r} contains out-of-range int8 values")
            vector_components.append(component)
        norm = math.sqrt(sum(component * component for component in vector_components))
        if norm == 0:
            raise ValueError(f"vector for {raw_word!r} must not be all zeros")
        parsed_vectors[raw_word] = tuple(vector_components)
        norms[raw_word] = norm

    return WordVectorTable(
        version=version,
        source=source,
        dimensions=dimensions,
        lexicon_hash=lexicon_signature,
        quantization_scheme=scheme,
        quantization_scale=float(scale),
        vectors=parsed_vectors,
        norms=norms,
    )


def _require_lexicon_vectors(
    lexicon: tuple[str, ...],
    vectors: WordVectorTable,
) -> tuple[str, ...]:
    missing_words = tuple(word for word in lexicon if word not in vectors.vectors)
    if missing_words:
        missing_text = ", ".join(word.upper() for word in missing_words[:5])
        raise ValueError(f"word vectors missing lexicon entries: {missing_text}")
    signature = lexicon_hash(lexicon)
    if signature != vectors.lexicon_hash:
        raise ValueError(
            "word vector table does not match the requested lexicon "
            f"(expected {signature}, found {vectors.lexicon_hash})"
        )
    return lexicon


def _cosine_similarity(
    left_word: str,
    right_word: str,
    vectors: WordVectorTable,
) -> float:
    left_vector = vectors.vectors.get(left_word)
    right_vector = vectors.vectors.get(right_word)
    if left_vector is None:
        raise ValueError(f"missing vector for word: {left_word.upper()}")
    if right_vector is None:
        raise ValueError(f"missing vector for word: {right_word.upper()}")
    dot_product = sum(left * right for left, right in zip(left_vector, right_vector, strict=True))
    return dot_product / (vectors.norms[left_word] * vectors.norms[right_word])


def score_word_for_seed(
    word: str,
    seeds: tuple[str, ...],
    vectors: WordVectorTable,
) -> float:
    normalized_word = normalize_word(word)
    if normalized_word is None:
        raise ValueError(f"invalid theme word: {word!r}")
    normalized_seeds = normalize_seeds(seeds)
    if not normalized_seeds:
        return 0.0
    return max(_cosine_similarity(normalized_word, seed, vectors) for seed in normalized_seeds)


def seed_relevance_scores(
    words: tuple[str, ...],
    seeds: tuple[str, ...],
    vectors: WordVectorTable,
) -> dict[str, float]:
    normalized_seeds = normalize_seeds(seeds)
    if not normalized_seeds:
        return {}
    missing_seeds = tuple(seed for seed in normalized_seeds if seed not in vectors.vectors)
    if missing_seeds:
        missing_text = ", ".join(word.upper() for word in missing_seeds)
        raise ValueError(f"missing vector for word: {missing_text}")
    return {
        word: score_word_for_seed(word, normalized_seeds, vectors)
        for word in dict.fromkeys(words)
        if word in vectors.vectors
    }


def diversify_theme_words(
    ranked_words: tuple[str, ...],
    seeds: tuple[str, ...],
    vectors: WordVectorTable,
    limit: int,
    mmr_lambda: float = DEFAULT_MMR_LAMBDA,
    redundancy_threshold: float = DEFAULT_REDUNDANCY_THRESHOLD,
) -> tuple[str, ...]:
    if limit <= 0:
        return ()
    normalized_seeds = normalize_seeds(seeds)
    if not normalized_seeds:
        return ()

    candidates = tuple(
        word
        for word in dict.fromkeys(ranked_words)
        if word in vectors.vectors and word not in normalized_seeds
    )
    if not candidates:
        return ()

    relevance_scores = {
        word: score_word_for_seed(word, normalized_seeds, vectors)
        for word in candidates
    }
    selected: list[str] = []
    remaining = list(candidates)

    while remaining and len(selected) < limit:
        scored_words: list[tuple[float, float, str]] = []
        for word in remaining:
            redundancy = max(
                (_cosine_similarity(word, selected_word, vectors) for selected_word in selected),
                default=0.0,
            )
            mmr_score = mmr_lambda * relevance_scores[word] - (1.0 - mmr_lambda) * redundancy
            scored_words.append((mmr_score, relevance_scores[word], word))

        _, relevance_score, chosen_word = min(
            scored_words,
            key=lambda item: (-item[0], -item[1], item[2]),
        )
        if relevance_score <= 0.0:
            break
        if selected:
            max_similarity = max(
                _cosine_similarity(chosen_word, selected_word, vectors)
                for selected_word in selected
            )
            if max_similarity >= redundancy_threshold:
                remaining.remove(chosen_word)
                continue
        selected.append(chosen_word)
        remaining.remove(chosen_word)

    return tuple(selected)


def score_theme_subset(
    words: tuple[str, ...],
    seeds: tuple[str, ...],
    vectors: WordVectorTable,
    limit: int = DEFAULT_THEME_WORD_LIMIT,
) -> ThemeScoreBreakdown:
    normalized_seeds = normalize_seeds(seeds)
    if not normalized_seeds or limit <= 0:
        return ThemeScoreBreakdown((), 0.0, 0.0, 0.0, 0.0)

    candidate_words = tuple(
        word
        for word in dict.fromkeys(words)
        if word in vectors.vectors and word not in normalized_seeds
    )
    if not candidate_words:
        return ThemeScoreBreakdown((), 0.0, 0.0, 0.0, 0.0)

    relevance_scores = {
        word: score_word_for_seed(word, normalized_seeds, vectors)
        for word in candidate_words
    }
    ranked_words = tuple(
        sorted(candidate_words, key=lambda word: (-relevance_scores[word], word))
    )
    selected_words = diversify_theme_words(ranked_words, normalized_seeds, vectors, limit)
    if not selected_words:
        return ThemeScoreBreakdown((), 0.0, 0.0, 0.0, 0.0)

    mean_relevance = sum(relevance_scores[word] for word in selected_words) / len(selected_words)
    if len(selected_words) == 1:
        weakest_link = mean_relevance
        diversity = 0.0
    else:
        pairwise_similarities = tuple(
            _cosine_similarity(left_word, right_word, vectors)
            for left_word, right_word in combinations(selected_words, 2)
        )
        weakest_link = min(pairwise_similarities)
        diversity = sum((1.0 - similarity) / 2.0 for similarity in pairwise_similarities) / len(
            pairwise_similarities
        )

    total = mean_relevance + weakest_link + diversity
    return ThemeScoreBreakdown(
        selected_words=selected_words,
        mean_relevance=mean_relevance,
        weakest_link=weakest_link,
        diversity=diversity,
        total=total,
    )


def rank_lexicon_for_seed(
    seeds: tuple[str, ...],
    lexicon: tuple[str, ...],
    vectors: WordVectorTable,
    preferred_words: tuple[str, ...] = (),
) -> tuple[str, ...]:
    unique_lexicon = tuple(dict.fromkeys(lexicon))
    _require_lexicon_vectors(unique_lexicon, vectors)
    validated_seeds = validate_seed_words(seeds, unique_lexicon)
    preferred_set = set(preferred_words)
    word_scores = seed_relevance_scores(unique_lexicon, validated_seeds, vectors)

    def sort_key(word: str) -> tuple[int, int, float, str]:
        seed_penalty = 0 if word in validated_seeds else 1
        preferred_penalty = 0 if word in preferred_set else 1
        return (seed_penalty, preferred_penalty, -word_scores[word], word)

    return tuple(sorted(unique_lexicon, key=sort_key))


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
            (
                sum(letter == seed[index] for index, letter in enumerate(word))
                for seed in normalized_seeds
            ),
            default=0,
        )
        return (-seed_bonus, -shared_letters, -shared_positions, word)

    return tuple(sorted(dict.fromkeys(candidates), key=score))


def _seed_neighbor_words(seeds: tuple[str, ...], candidates: tuple[str, ...]) -> tuple[str, ...]:
    normalized_seeds = normalize_seeds(seeds)
    neighbors: list[str] = []
    for candidate in candidates:
        if any(
            len(set(candidate) & set(seed)) >= 2
            or sum(letter == seed[index] for index, letter in enumerate(candidate)) >= 1
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
