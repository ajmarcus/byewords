import argparse

from byewords.generate import generate_puzzle_cached, load_default_inputs
from byewords.render import render_puzzle_text


def parse_args(argv: list[str] | None = None) -> tuple[str, ...]:
    parser = argparse.ArgumentParser(
        prog="byewords",
        description="Generate a 5x5 mini crossword.",
    )
    parser.add_argument(
        "seeds",
        nargs="*",
        help="Optional seed words to nudge fill selection.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        action="append",
        dest="seed_flags",
        default=[],
        help="Add a seed word. May be passed multiple times.",
    )
    args = parser.parse_args(argv)
    if args.seed_flags and args.seeds:
        parser.error("use either positional seeds or repeated --seed flags, not both")
    seeds = tuple(args.seed_flags) + tuple(args.seeds)
    return seeds


def main() -> int:
    lexicon_words, clue_bank = load_default_inputs()
    seeds = parse_args()
    try:
        puzzle = generate_puzzle_cached(seeds, lexicon_words, clue_bank)
    except ValueError as exc:
        print(f"error: {exc}")
        return 1
    print(render_puzzle_text(puzzle))
    return 0
