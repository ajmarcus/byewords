import argparse
from pathlib import Path

from byewords.data_maintenance import sort_bundled_data_files


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Sort the bundled lexicon, sort the clue bank, and drop orphan clue entries."
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = sort_bundled_data_files(args.words, args.clue_bank)
    print(
        "Sorted bundled data:",
        f"{result.word_count} words,",
        f"{result.clue_entry_count} clue entries,",
        f"removed {len(result.removed_clue_answers)} orphan clue answers.",
    )


if __name__ == "__main__":
    main()
