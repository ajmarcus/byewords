import argparse
import sys
import time
from typing import TextIO

from byewords.generate import generate_puzzle_cached, load_default_inputs
from byewords.render import render_puzzle_text
from byewords.types import ProgressUpdate


class BuildAnimator:
    _frames = ("-", "\\", "|", "/")

    def __init__(self, stream: TextIO, frame_interval: float = 0.03) -> None:
        self._stream = stream
        self._enabled = stream.isatty()
        self._frame_interval = frame_interval
        self._frame_index = 0
        self._last_draw = 0.0
        self._line_count = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def update(self, progress: ProgressUpdate) -> None:
        if not self._enabled:
            return
        now = time.monotonic()
        should_force = progress.stage in {"cache_hit", "solution"}
        if not should_force and now - self._last_draw < self._frame_interval:
            return
        lines = self._render_lines(progress)
        if self._line_count:
            self._stream.write(f"\x1b[{self._line_count}F\x1b[J")
        else:
            self._stream.write("\x1b[?25l")
        self._stream.write("\n".join(lines) + "\n")
        self._stream.flush()
        self._line_count = len(lines)
        self._last_draw = now
        self._frame_index += 1

    def finish(self) -> None:
        if not self._enabled or not self._line_count:
            return
        self._stream.write(f"\x1b[{self._line_count}F\x1b[J\x1b[?25h")
        self._stream.flush()
        self._line_count = 0

    def _render_lines(self, progress: ProgressUpdate) -> list[str]:
        spinner = self._frames[self._frame_index % len(self._frames)]
        rows: list[str] = []
        active_row_index = min(len(progress.partial_rows), 4)
        active_column_index = self._frame_index % 5
        for row_index in range(5):
            if row_index < len(progress.partial_rows):
                rows.append(" ".join(progress.partial_rows[row_index].upper()))
                continue
            cells = ["."] * 5
            if progress.stage not in {"cache_hit", "solution"} and row_index == active_row_index:
                cells[active_column_index] = spinner
            rows.append(" ".join(cells))
        return [f"{spinner} {progress.message}"] + rows


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
    animator = BuildAnimator(sys.stderr)
    try:
        if animator.enabled:
            puzzle = generate_puzzle_cached(
                seeds,
                lexicon_words,
                clue_bank,
                progress_callback=animator.update,
            )
        else:
            puzzle = generate_puzzle_cached(seeds, lexicon_words, clue_bank)
    except ValueError as exc:
        animator.finish()
        print(f"error: {exc}")
        return 1
    animator.finish()
    print(render_puzzle_text(puzzle))
    return 0
