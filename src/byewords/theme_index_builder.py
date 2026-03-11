from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from importlib import resources
from pathlib import Path

from byewords.lexicon import load_clue_bank, load_word_list
from byewords.theme import lexicon_hash

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
    max_abs = max(
        abs(component)
        for vector in float_vectors.values()
        for component in vector
    )
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the bundled semantic word vector table.")
    parser.add_argument(
        "--output",
        type=Path,
        default=_data_path("word_vectors.json"),
        help="Path to the generated vector table JSON.",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=DEFAULT_DIMENSIONS,
        help="Embedding dimensionality for the generated table.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    write_word_vectors(args.output, args.dimensions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
