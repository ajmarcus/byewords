import argparse
import sys
import time
from pathlib import Path
from typing import TextIO

from byewords.clues import make_across_clues, make_down_clues
from byewords.generate import generate_puzzle_cached, load_default_inputs
from byewords.groq_clues import (
    CLI_DESCRIPTION as CLUES_DESCRIPTION,
    configure_parser as configure_clues_parser,
    default_clue_bank_path,
    regenerate_clues as run_clue_regeneration,
    run as run_clues_command,
)
from byewords.puz import puzzle_to_puz_bytes
from byewords.render import render_puzzle_text
from byewords.theme_index_builder import (
    _COMMANDS as THEME_TOOL_COMMANDS,
    add_subcommands as add_theme_tool_subcommands,
    run as run_theme_tool_command,
)
from byewords.types import ProgressUpdate, Puzzle, RuntimeReport

CLI_PROG = "bzw"
GENERATE_COMMAND = "generate"
CLI_DESCRIPTION = "Generate Byewords minis and run bundled maintenance tools."
CLI_EPILOG = (
    "Docs: README.md, docs/plan.md, docs/implementation.md\n\n"
    "Examples:\n"
    "  uv run bzw --seed snail\n"
    "  uv run bzw cache\n"
    "  uv run bzw clues snail\n"
    "  uv run bzw vectors"
)
TOP_LEVEL_COMMANDS = frozenset({GENERATE_COMMAND, "clues", *THEME_TOOL_COMMANDS})


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
        should_force = progress.stage in {"cache_hit", "candidate_solution", "solution"}
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
            if progress.stage not in {"cache_hit", "candidate_solution", "solution"} and row_index == active_row_index:
                cells[active_column_index] = spinner
            rows.append(" ".join(cells))
        return [f"{spinner} {progress.message}"] + rows


def configure_generate_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
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
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=CLI_PROG,
        description=CLI_DESCRIPTION,
        epilog=CLI_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    generate_parser = subparsers.add_parser(
        GENERATE_COMMAND,
        help="Generate a 5x5 mini crossword.",
        description="Generate a 5x5 mini crossword.",
    )
    configure_generate_parser(generate_parser)

    clues_parser = subparsers.add_parser(
        "clues",
        help="Generate or refresh crossword clues with Groq.",
        description=CLUES_DESCRIPTION,
    )
    configure_clues_parser(clues_parser)

    add_theme_tool_subcommands(subparsers)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    if not raw_args:
        return argparse.Namespace(command="help")
    if raw_args[0] not in TOP_LEVEL_COMMANDS and raw_args[0] not in {"-h", "--help"}:
        raw_args = [GENERATE_COMMAND, *raw_args]
    parser = build_parser()
    args = parser.parse_args(raw_args)
    if args.command == GENERATE_COMMAND and args.seed_flags and args.seeds:
        parser.error("use either positional seeds or repeated --seed flags, not both")
    if getattr(args, "command", None) == GENERATE_COMMAND:
        args.seeds = tuple(args.seed_flags) + tuple(args.seeds)
    return args


def _write_text_output(text: str, output_path: str | None, stdout: TextIO) -> None:
    if output_path is None:
        print(text, file=stdout)
        return
    Path(output_path).write_text(text + "\n", encoding="utf-8")


def _write_puz_output(payload: bytes, output_path: str | None, stdout: TextIO) -> None:
    if output_path is not None:
        Path(output_path).write_bytes(payload)
        return
    if stdout.isatty():
        raise ValueError("refusing to write binary .puz data to an interactive terminal; use --output")
    buffer = getattr(stdout, "buffer", None)
    if buffer is None:
        raise ValueError("binary .puz output requires a binary stdout buffer or --output")
    buffer.write(payload)
    buffer.flush()


def run_generate_command(args: argparse.Namespace, stdout: TextIO, stderr: TextIO) -> int:
    lexicon_words, clue_bank = load_default_inputs(include_fallback_clues=False)
    if not args.seeds:
        print(
            "error: generate requires at least one seed; use `bzw cache` to build the offline puzzle store",
            file=stdout,
        )
        return 1
    animator = BuildAnimator(stderr)
    runtime_report: RuntimeReport | None = None
    candidate_solution_reported = False
    persistent_candidate_updates = (
        args.output is None and args.format == "text" and stdout.isatty()
    )

    def handle_progress(progress: ProgressUpdate) -> None:
        nonlocal candidate_solution_reported, runtime_report
        if progress.runtime_report is not None:
            runtime_report = progress.runtime_report
            return
        if (
            persistent_candidate_updates
            and progress.stage == "candidate_solution"
            and not candidate_solution_reported
        ):
            print(progress.message, file=stdout, flush=True)
            candidate_solution_reported = True
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
        print(f"error: {exc}", file=stdout)
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
            file=stderr,
        )
    if args.regenerate_clues:
        try:
            run_clue_regeneration(
                answers=tuple(dict.fromkeys(clue.answer for clue in puzzle.across + puzzle.down)),
                clue_bank=clue_bank,
                clue_bank_path=default_clue_bank_path(),
                env=None,
                errors=stderr,
                force=True,
            )
        except (RuntimeError, ValueError) as exc:
            print(f"error: {exc}", file=stdout)
            return 1
        puzzle = _refresh_puzzle_clues(puzzle, clue_bank)
    if args.format == "puz":
        try:
            _write_puz_output(puzzle_to_puz_bytes(puzzle), args.output, stdout)
        except ValueError as exc:
            print(f"error: {exc}", file=stdout)
            return 1
        return 0
    _write_text_output(render_puzzle_text(puzzle), args.output, stdout)
    return 0


def main(
    argv: list[str] | None = None,
    *,
    env: dict[str, str] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    output = stdout if stdout is not None else sys.stdout
    errors = stderr if stderr is not None else sys.stderr
    parser = build_parser()
    args = parse_args(argv)
    if args.command == "help":
        parser.print_help(file=output)
        return 0
    if args.command == GENERATE_COMMAND:
        return run_generate_command(args, output, errors)
    if args.command == "clues":
        return run_clues_command(args, env=env, stdout=output, stderr=errors)
    return run_theme_tool_command(args, stdout=output)


def _refresh_puzzle_clues(puzzle: Puzzle, clue_bank: dict[str, tuple[str, ...]]) -> Puzzle:
    used_clues: set[str] = set()
    return Puzzle(
        grid=puzzle.grid,
        across=make_across_clues(puzzle.grid, clue_bank, used_clues),
        down=make_down_clues(puzzle.grid, clue_bank, used_clues),
        theme_words=puzzle.theme_words,
        title=puzzle.title,
    )
