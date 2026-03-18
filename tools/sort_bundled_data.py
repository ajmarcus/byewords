import argparse
from pathlib import Path

from byewords.data_maintenance import sort_bundled_data_files


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description=(
            "Sort the bundled lexicon, then prune and rewrite the bundled data files "
            "to match it."
        )
    )
    parser.add_argument(
        "--words",
        type=Path,
        default=project_root / "src/byewords/data/words_5.txt",
        help="Path to the bundled five-letter word list.",
    )
    parser.add_argument(
        "--clue-bank",
        type=Path,
        default=project_root / "src/byewords/data/clue_bank.json",
        help="Path to the bundled clue bank JSON file.",
    )
    parser.add_argument(
        "--vectors",
        type=Path,
        default=project_root / "src/byewords/data/word_vectors.json",
        help="Path to the bundled word vector JSON file.",
    )
    parser.add_argument(
        "--puzzles",
        type=Path,
        default=project_root / "src/byewords/data/puzzles.json",
        help="Path to the bundled puzzle cache JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = sort_bundled_data_files(
        args.words,
        args.clue_bank,
        vectors_path=args.vectors,
        puzzles_path=args.puzzles,
    )
    print(
        "Sorted bundled data:",
        f"{result.word_count} words,",
        f"{result.clue_entry_count} clue entries,",
        f"{result.vector_entry_count} vectors,",
        f"{result.puzzle_record_count} puzzle records,",
        f"removed {len(result.removed_clue_answers)} orphan clue answers,",
        f"{len(result.removed_vector_words)} orphan vector entries,",
        f"and {len(result.removed_puzzle_records)} stale puzzle records.",
    )
    if result.missing_vector_words:
        print(
            "Missing vector entries for canonical words:",
            ", ".join(result.missing_vector_words),
        )


if __name__ == "__main__":
    main()
