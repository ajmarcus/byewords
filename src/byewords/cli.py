import argparse

from byewords.generate import generate_puzzle, load_default_inputs
from byewords.render import render_puzzle_text


def parse_args(argv: list[str] | None = None) -> tuple[str, ...]:
    parser = argparse.ArgumentParser(
        prog="byewords",
        description="Generate a 5x5 mini crossword from seed words.",
    )
    parser.add_argument(
        "seeds",
        nargs="*",
        help="Seed words to steer the puzzle theme.",
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
    return seeds or ("snail",)


def main() -> int:
    seeds = parse_args()
    lexicon_words, related_map, clue_bank = load_default_inputs()
    puzzle = generate_puzzle(seeds, lexicon_words, related_map, clue_bank)
    print(render_puzzle_text(puzzle))
    return 0
