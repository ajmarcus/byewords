from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import asdict
from importlib import resources
from pathlib import Path
from typing import Sequence

from byewords.lexicon import load_clue_bank, load_word_list
from byewords.puzzle_store import DEFAULT_CANDIDATES_PER_SEED, build_batch_puzzle_cache, default_puzzle_store_path
from byewords.theme import (
    DEFAULT_THEME_WORD_LIMIT,
    THEME_INTRUSION_REVIEW_CASES,
    THEME_RETRIEVAL_REVIEW_CASES,
    ThemeIntrusionComparison,
    ThemeRetrievalComparison,
    lexicon_hash,
    load_word_vectors,
    review_theme_intrusions,
    review_theme_retrieval,
)

DEFAULT_DIMENSIONS = 128
TOKEN_PATTERN = re.compile(r"[a-z]+")
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "be",
        "for",
        "from",
        "in",
        "into",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
)
_COMMANDS = frozenset(("vectors", "cache", "retrieval-review", "intrusion-review"))


def _data_path(filename: str) -> Path:
    return Path(str(resources.files("byewords").joinpath("data", filename)))


def _hash_feature(feature: str, dimensions: int) -> tuple[int, int]:
    digest = hashlib.sha256(feature.encode("utf-8")).digest()
    index = int.from_bytes(digest[:8], "big") % dimensions
    sign = 1 if digest[8] % 2 == 0 else -1
    return index, sign


def _clue_tokens(clues: tuple[str, ...], answer: str) -> list[str]:
    tokens: list[str] = []
    for clue in clues:
        for token in TOKEN_PATTERN.findall(clue.lower()):
            if len(token) < 3 or token in STOPWORDS or token == answer:
                continue
            tokens.append(token)
    return tokens


def _word_features(
    word: str,
    clues: tuple[str, ...],
) -> list[tuple[str, float]]:
    features: list[tuple[str, float]] = [(f"answer:{word}", 4.0)]
    for ngram_size in (2, 3):
        for index in range(len(word) - ngram_size + 1):
            ngram = word[index:index + ngram_size]
            features.append((f"ngram:{ngram}", 1.0))
    tokens = _clue_tokens(clues, word)
    token_counts: dict[str, int] = {}
    for token in tokens:
        token_counts[token] = token_counts.get(token, 0) + 1
    for token, count in sorted(token_counts.items()):
        features.append((f"token:{token}", 1.5 + (count - 1) * 0.25))
    for left_token, right_token in zip(tokens, tokens[1:], strict=False):
        features.append((f"pair:{left_token}_{right_token}", 0.75))
    return features


def _raw_vector(
    word: str,
    clues: tuple[str, ...],
    dimensions: int,
) -> list[float]:
    vector = [0.0] * dimensions
    for feature, weight in _word_features(word, clues):
        index, sign = _hash_feature(feature, dimensions)
        vector[index] += weight * sign
    norm = math.sqrt(sum(component * component for component in vector))
    if norm == 0:
        vector[0] = 1.0
        return vector
    return [component / norm for component in vector]


def build_word_vector_payload(
    lexicon_words: tuple[str, ...],
    clue_bank: dict[str, tuple[str, ...]],
    dimensions: int = DEFAULT_DIMENSIONS,
) -> dict[str, object]:
    float_vectors = {
        word: _raw_vector(word, clue_bank.get(word, ()), dimensions)
        for word in lexicon_words
    }
    max_abs = max(abs(component) for vector in float_vectors.values() for component in vector)
    scale = max_abs / 127 if max_abs else 1 / 127
    quantized_vectors = {
        word: [max(-127, min(127, int(round(component / scale)))) for component in vector]
        for word, vector in float_vectors.items()
    }
    return {
        "version": 1,
        "source": "hashed-clue-features-v1",
        "dimensions": dimensions,
        "lexicon_hash": lexicon_hash(lexicon_words),
        "quantization": {
            "scheme": "int8",
            "scale": round(scale, 10),
        },
        "vectors": quantized_vectors,
    }


def write_word_vectors(
    output_path: Path,
    dimensions: int = DEFAULT_DIMENSIONS,
) -> None:
    lexicon_words = load_word_list(str(_data_path("words_5.txt")))
    clue_bank = load_clue_bank(str(_data_path("clue_bank.json")))
    payload = build_word_vector_payload(lexicon_words, clue_bank, dimensions)
    output_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    if not raw_args or raw_args[0] not in _COMMANDS:
        raw_args = ["vectors", *raw_args]

    parser = argparse.ArgumentParser(
        description="Offline tooling for semantic theme vectors, cache builds, and review reports.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    vectors_parser = subparsers.add_parser("vectors", help="Build the bundled semantic word vector table.")
    vectors_parser.add_argument(
        "--output",
        type=Path,
        default=_data_path("word_vectors.json"),
        help="Path to the generated vector table JSON.",
    )
    vectors_parser.add_argument(
        "--dimensions",
        type=int,
        default=DEFAULT_DIMENSIONS,
        help="Embedding dimensionality for the generated table.",
    )

    cache_parser = subparsers.add_parser(
        "cache",
        help="Build or refresh the offline puzzle cache, including the top-100 clue stage.",
    )
    cache_parser.add_argument(
        "--output",
        type=Path,
        default=default_puzzle_store_path(),
        help="Path to the generated puzzles.json cache.",
    )
    cache_parser.add_argument(
        "--candidates-per-seed",
        type=int,
        default=DEFAULT_CANDIDATES_PER_SEED,
        help="How many answer-only candidates to retain per seed before top-100 clue curation.",
    )
    cache_parser.add_argument(
        "--top-clue-limit",
        type=int,
        default=100,
        help="How many answer-only winners to carry into the clue stage.",
    )

    retrieval_parser = subparsers.add_parser(
        "retrieval-review",
        help="Run deterministic retrieval-review reports against the bundled review corpus.",
    )
    retrieval_parser.add_argument(
        "--vectors",
        type=Path,
        default=_data_path("word_vectors.json"),
        help="Path to the bundled vector table JSON.",
    )
    retrieval_parser.add_argument(
        "--top-n",
        type=int,
        default=8,
        help="How many top-ranked words to evaluate for each review seed.",
    )
    retrieval_parser.add_argument(
        "--neighbor-count",
        type=int,
        default=32,
        help="Neighbor depth for the rank-overlap comparison metric.",
    )
    retrieval_parser.add_argument(
        "--json",
        action="store_true",
        help="Print structured JSON instead of a text report.",
    )

    intrusion_parser = subparsers.add_parser(
        "intrusion-review",
        help="Run deterministic intrusion-review reports against the bundled review corpus.",
    )
    intrusion_parser.add_argument(
        "--vectors",
        type=Path,
        default=_data_path("word_vectors.json"),
        help="Path to the bundled vector table JSON.",
    )
    intrusion_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_THEME_WORD_LIMIT,
        help="Maximum size of the selected theme-bearing subset during intrusion tests.",
    )
    intrusion_parser.add_argument(
        "--json",
        action="store_true",
        help="Print structured JSON instead of a text report.",
    )

    return parser.parse_args(raw_args)


def _load_bundled_inputs() -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    return (
        load_word_list(str(_data_path("words_5.txt"))),
        load_clue_bank(str(_data_path("clue_bank.json"))),
    )


def _load_validated_vectors(path: Path, lexicon_words: tuple[str, ...]):
    vectors = load_word_vectors(str(path))
    expected_hash = lexicon_hash(tuple(dict.fromkeys(lexicon_words)))
    if vectors.lexicon_hash != expected_hash:
        raise ValueError(
            "word vector table does not match the bundled lexicon "
            f"(expected {expected_hash}, found {vectors.lexicon_hash})"
        )
    missing_words = [word for word in lexicon_words if word not in vectors.vectors]
    if missing_words:
        missing_text = ", ".join(word.upper() for word in missing_words[:5])
        raise ValueError(f"word vectors missing lexicon entries: {missing_text}")
    return vectors


def _print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _print_retrieval_report(reports: tuple[ThemeRetrievalComparison, ...]) -> None:
    for report in reports:
        cosine_hits = ", ".join(word.upper() for word in report.cosine.expected_hits) or "none"
        cosine_intrusions = ", ".join(word.upper() for word in report.cosine.unexpected_hits) or "none"
        overlap_hits = ", ".join(word.upper() for word in report.rank_overlap.expected_hits) or "none"
        overlap_intrusions = ", ".join(word.upper() for word in report.rank_overlap.unexpected_hits) or "none"
        print(
            f"{report.seed.upper()}: "
            f"cosine hits={cosine_hits} intrusions={cosine_intrusions}; "
            f"rank_overlap hits={overlap_hits} intrusions={overlap_intrusions}"
        )


def _print_intrusion_report(reports: tuple[ThemeIntrusionComparison, ...]) -> None:
    for report in reports:
        selected_intruders = tuple(
            trial.intruder.upper()
            for trial in report.trials
            if trial.intruder_selected
        )
        intruder_text = ", ".join(selected_intruders) or "none"
        baseline_text = ", ".join(word.upper() for word in report.baseline_selected_words) or "none"
        print(
            f"{report.seed.upper()}: "
            f"pass_rate={report.pass_rate:.2f}; "
            f"baseline={baseline_text}; "
            f"intruders_selected={intruder_text}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.command == "vectors":
        write_word_vectors(args.output, args.dimensions)
        print(f"Wrote semantic vectors to {args.output}")
        return 0

    lexicon_words, clue_bank = _load_bundled_inputs()

    if args.command == "cache":
        vectors = _load_validated_vectors(_data_path("word_vectors.json"), lexicon_words)
        store_path, total_records, generated_records = build_batch_puzzle_cache(
            lexicon_words,
            clue_bank,
            path=args.output,
            vectors=vectors,
            candidates_per_seed=args.candidates_per_seed,
            top_clue_limit=args.top_clue_limit,
        )
        print(
            f"Cached {total_records} puzzles in {store_path} "
            f"({generated_records} generated in this run)."
        )
        return 0

    if args.command == "retrieval-review":
        vectors = _load_validated_vectors(args.vectors, lexicon_words)
        reports = review_theme_retrieval(
            THEME_RETRIEVAL_REVIEW_CASES,
            lexicon_words,
            vectors,
            top_n=args.top_n,
            neighbor_count=args.neighbor_count,
        )
        if args.json:
            _print_json([asdict(report) for report in reports])
        else:
            _print_retrieval_report(reports)
        return 0

    vectors = _load_validated_vectors(args.vectors, lexicon_words)
    reports = review_theme_intrusions(
        THEME_INTRUSION_REVIEW_CASES,
        lexicon_words,
        vectors,
        limit=args.limit,
    )
    if args.json:
        _print_json([asdict(report) for report in reports])
    else:
        _print_intrusion_report(reports)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
