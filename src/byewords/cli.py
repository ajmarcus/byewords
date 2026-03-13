import argparse
import sys
import time
from pathlib import Path
from typing import TextIO

from byewords.clues import make_across_clues, make_down_clues
from byewords.generate import generate_puzzle_cached, load_default_inputs
from byewords.groq_clues import default_clue_bank_path, regenerate_clues as run_clue_regeneration
from byewords.puz import puzzle_to_puz_bytes
from byewords.puzzle_store import build_batch_puzzle_cache
from byewords.render import render_puzzle_text
from byewords.types import ProgressUpdate, Puzzle, RuntimeReport


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    parser.add_argument(
        "--format",
        choices=("text", "puz"),
        default="text",
        help="Choose the output format.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Write the puzzle to a file instead of stdout.",
    )
    parser.add_argument(
        "--regenerate-clues",
        action="store_true",
        help="Force Groq clue regeneration for the generated puzzle before rendering output.",
    )
    args = parser.parse_args(argv)
    if args.seed_flags and args.seeds:
        parser.error("use either positional seeds or repeated --seed flags, not both")
    args.seeds = tuple(args.seed_flags) + tuple(args.seeds)
    return args


def _write_text_output(text: str, output_path: str | None, stdout: TextIO) -> None:
    if output_path is None:
        print(text)
        return
    Path(output_path).write_text(text + "\n", encoding="utf-8")


def _write_puz_output(payload: bytes, output_path: str | None) -> None:
    if output_path is not None:
        Path(output_path).write_bytes(payload)
        return
    if sys.stdout.isatty():
        raise ValueError("refusing to write binary .puz data to an interactive terminal; use --output")
    buffer = getattr(sys.stdout, "buffer", None)
    if buffer is None:
        raise ValueError("binary .puz output requires a binary stdout buffer or --output")
    buffer.write(payload)
    buffer.flush()


def main() -> int:
    lexicon_words, clue_bank = load_default_inputs()
    args = parse_args()
    if not args.seeds:
        if args.format != "text":
            print("error: batch mode only supports text output")
            return 1
        if args.output is not None:
            print("error: batch mode does not support --output")
            return 1
        if args.regenerate_clues:
            print("error: batch mode does not support --regenerate-clues")
            return 1
        store_path, total_records, generated_records = build_batch_puzzle_cache(lexicon_words, clue_bank)
        print(
            f"Cached {total_records} puzzles in {store_path} "
            f"({generated_records} generated in this run)."
        )
        return 0
    animator = BuildAnimator(sys.stderr)
    runtime_report: RuntimeReport | None = None

    def handle_progress(progress: ProgressUpdate) -> None:
        nonlocal runtime_report
        if progress.runtime_report is not None:
            runtime_report = progress.runtime_report
            return
        animator.update(progress)

    try:
        if animator.enabled:
            puzzle = generate_puzzle_cached(
                args.seeds,
                lexicon_words,
                clue_bank,
                progress_callback=handle_progress,
            )
        else:
            puzzle = generate_puzzle_cached(
                args.seeds,
                lexicon_words,
                clue_bank,
                progress_callback=handle_progress,
            )
    except ValueError as exc:
        animator.finish()
        print(f"error: {exc}")
        return 1
    animator.finish()
    if runtime_report is not None:
        theme_subset = ", ".join(word.upper() for word in runtime_report.selected_theme_subset) or "none"
        print(
            (
                "runtime: "
                f"semantic={'on' if runtime_report.semantic_ordering else 'off'} "
                f"fallback={'yes' if runtime_report.used_budget_fallback else 'no'} "
                f"theme_subset={theme_subset} "
                f"weakest_link={runtime_report.selected_theme_weakest_link:.3f}"
            ),
            file=sys.stderr,
        )
    if args.regenerate_clues:
        try:
            run_clue_regeneration(
                answers=tuple(dict.fromkeys(clue.answer for clue in puzzle.across + puzzle.down)),
                clue_bank=clue_bank,
                clue_bank_path=default_clue_bank_path(),
                env=None,
                errors=sys.stderr,
                force=True,
            )
        except (RuntimeError, ValueError) as exc:
            print(f"error: {exc}")
            return 1
        puzzle = _refresh_puzzle_clues(puzzle, clue_bank)
    if args.format == "puz":
        try:
            _write_puz_output(puzzle_to_puz_bytes(puzzle), args.output)
        except ValueError as exc:
            print(f"error: {exc}")
            return 1
        return 0
    _write_text_output(render_puzzle_text(puzzle), args.output, sys.stdout)
    return 0


def _refresh_puzzle_clues(puzzle: Puzzle, clue_bank: dict[str, tuple[str, ...]]) -> Puzzle:
    used_clues: set[str] = set()
    return Puzzle(
        grid=puzzle.grid,
        across=make_across_clues(puzzle.grid, clue_bank, used_clues),
        down=make_down_clues(puzzle.grid, clue_bank, used_clues),
        theme_words=puzzle.theme_words,
        title=puzzle.title,
    )
