from __future__ import annotations

import argparse
import importlib
import json
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

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_EMBEDDING_SOURCE = "baai-bge-small-en-v1.5"
DEFAULT_EMBEDDING_URL = "https://huggingface.co/BAAI/bge-small-en-v1.5"
DEFAULT_EMBEDDING_LICENSE = "MIT"
DEFAULT_EMBEDDING_ATTRIBUTION = (
    "This data contains semantic vectors derived from BAAI/bge-small-en-v1.5, "
    "released under the MIT license."
)
_COMMANDS = frozenset(("vectors", "cache", "retrieval-review", "intrusion-review"))


def _data_path(filename: str) -> Path:
    return Path(str(resources.files("byewords").joinpath("data", filename)))


def _quantize_vectors(float_vectors: dict[str, list[float]]) -> tuple[dict[str, list[int]], float]:
    max_abs = max(abs(component) for vector in float_vectors.values() for component in vector)
    scale = max_abs / 127 if max_abs else 1 / 127
    quantized_vectors = {
        word: [max(-127, min(127, int(round(component / scale)))) for component in vector]
        for word, vector in float_vectors.items()
    }
    return quantized_vectors, scale


def _load_sentence_transformer(model_name: str, device: str | None):
    try:
        sentence_transformers = importlib.import_module("sentence_transformers")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Building semantic vectors requires sentence-transformers and torch; run "
            "`uv run --with sentence-transformers --with torch python -m byewords.theme_index_builder vectors`."
        ) from exc
    SentenceTransformer = sentence_transformers.SentenceTransformer
    return SentenceTransformer(model_name, device=device)


def _load_embedding_vectors(
    lexicon_words: tuple[str, ...],
    model_name: str,
    batch_size: int,
    device: str | None,
) -> tuple[dict[str, list[float]], int]:
    model = _load_sentence_transformer(model_name, device)
    raw_vectors = model.encode(
        list(lexicon_words),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    if len(raw_vectors) != len(lexicon_words):
        raise ValueError("embedding model returned the wrong number of vectors")

    vectors: dict[str, list[float]] = {}
    dimensions = 0
    for word, raw_vector in zip(lexicon_words, raw_vectors, strict=True):
        components = raw_vector.tolist() if hasattr(raw_vector, "tolist") else list(raw_vector)
        vector = [float(component) for component in components]
        if not vector:
            raise ValueError(f"embedding model returned an empty vector for {word.upper()}")
        if dimensions == 0:
            dimensions = len(vector)
        elif len(vector) != dimensions:
            raise ValueError(
                "embedding model returned inconsistent vector dimensions "
                f"(expected {dimensions}, found {len(vector)} for {word.upper()})"
            )
        vectors[word] = vector
    return vectors, dimensions


def build_word_vector_payload(
    lexicon_words: tuple[str, ...],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 64,
    device: str | None = None,
) -> dict[str, object]:
    float_vectors, dimensions = _load_embedding_vectors(lexicon_words, model_name, batch_size, device)
    quantized_vectors, scale = _quantize_vectors(float_vectors)
    return {
        "version": 1,
        "source": DEFAULT_EMBEDDING_SOURCE,
        "dimensions": dimensions,
        "lexicon_hash": lexicon_hash(lexicon_words),
        "model_name": model_name,
        "source_url": DEFAULT_EMBEDDING_URL,
        "license": DEFAULT_EMBEDDING_LICENSE,
        "attribution": DEFAULT_EMBEDDING_ATTRIBUTION,
        "quantization": {
            "scheme": "int8",
            "scale": round(scale, 10),
        },
        "vectors": quantized_vectors,
    }


def write_word_vectors(
    output_path: Path,
    batch_size: int = 64,
    device: str | None = None,
) -> None:
    lexicon_words = load_word_list(str(_data_path("words_5.txt")))
    payload = build_word_vector_payload(
        lexicon_words,
        batch_size=batch_size,
        device=device,
    )
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
        "--batch-size",
        type=int,
        default=64,
        help="Batch size to use while encoding the bundled lexicon with the embedding model.",
    )
    vectors_parser.add_argument(
        "--device",
        default=None,
        help=(
            "Optional device override passed to sentence-transformers, such as "
            "`cpu`, `cuda`, or `mps`."
        ),
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
        write_word_vectors(args.output, batch_size=args.batch_size, device=args.device)
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
