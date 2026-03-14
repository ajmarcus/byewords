import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import ANY, patch

from byewords.cli import main, parse_args
from byewords.types import ProgressUpdate, RuntimeReport
from tests.test_puz import build_test_puzzle


class FakeTty(StringIO):
    def isatty(self) -> bool:
        return True


class TestCli(unittest.TestCase):
    def test_cli_without_arguments_builds_batch_cache(self) -> None:
        buf = StringIO()

        with (
            redirect_stdout(buf),
            patch("sys.argv", ["byewords"]),
            patch("byewords.cli.load_default_inputs", return_value=((), {})),
            patch("byewords.cli.build_batch_puzzle_cache", return_value=(Path("/tmp/puzzles.json"), 2, 2)) as build_batch,
        ):
            main()

        output = buf.getvalue()
        build_batch.assert_called_once_with((), {})
        self.assertIn("Cached 2 puzzles in /tmp/puzzles.json (2 generated in this run).", output)

    def test_cli_writes_text_output_to_file(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir, "puzzle.txt")
            with (
                patch("sys.argv", ["byewords", "--output", str(output_path)]),
                patch("byewords.cli.load_default_inputs", return_value=((), {})),
                patch("byewords.cli.build_batch_puzzle_cache"),
            ):
                exit_code = main()

        self.assertEqual(exit_code, 1)
        self.assertFalse(output_path.exists())

    def test_cli_writes_puz_output_to_file(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir, "puzzle.puz")
            with (
                patch("sys.argv", ["byewords", "--format", "puz", "--output", str(output_path)]),
                patch("byewords.cli.load_default_inputs", return_value=((), {})),
                patch("byewords.cli.build_batch_puzzle_cache"),
            ):
                exit_code = main()

        self.assertEqual(exit_code, 1)
        self.assertFalse(output_path.exists())

    def test_cli_rejects_binary_output_to_terminal(self) -> None:
        stdout = FakeTty()

        with (
            patch("sys.stdout", stdout),
            patch("sys.argv", ["byewords", "--seed", "snail", "--format", "puz"]),
            patch("byewords.cli.load_default_inputs", return_value=((), {})),
            patch("byewords.cli.generate_puzzle_cached") as generate_cached,
            patch("byewords.cli.puzzle_to_puz_bytes", return_value=b"puz-bytes"),
        ):
            generate_cached.return_value = object()
            exit_code = main()

        self.assertEqual(exit_code, 1)
        self.assertIn("refusing to write binary .puz data", stdout.getvalue())

    def test_cli_returns_error_for_seed_that_is_not_in_the_lexicon(self) -> None:
        stdout = StringIO()

        with redirect_stdout(stdout), patch("sys.argv", ["byewords", "--seed", "zzzzz"]):
            exit_code = main()

        self.assertEqual(exit_code, 0)
        self.assertIn("BYEWORDS Mini", stdout.getvalue())

    def test_cli_reports_when_no_puzzle_is_possible(self) -> None:
        stdout = StringIO()

        with (
            redirect_stdout(stdout),
            patch("sys.argv", ["byewords", "--seed", "snail"]),
            patch("byewords.cli.load_default_inputs", return_value=((), {})),
            patch("byewords.cli.generate_puzzle_cached", side_effect=ValueError("unable to generate a valid 5x5 puzzle from the current lexicon")),
        ):
            exit_code = main()

        self.assertEqual(exit_code, 1)
        self.assertIn("error: unable to generate a valid 5x5 puzzle", stdout.getvalue())

    def test_cli_uses_cached_generation_for_seeded_runs(self) -> None:
        stdout = StringIO()

        with (
            redirect_stdout(stdout),
            patch("sys.argv", ["byewords", "--seed", "snail"]),
            patch("byewords.cli.load_default_inputs", return_value=((), {})),
            patch("byewords.cli.generate_puzzle_cached") as generate_cached,
            patch("byewords.cli.render_puzzle_text", return_value="cached puzzle"),
        ):
            generate_cached.return_value = object()
            exit_code = main()

        self.assertEqual(exit_code, 0)
        generate_cached.assert_called_once_with(("snail",), (), {}, progress_callback=ANY)
        self.assertEqual(stdout.getvalue(), "cached puzzle\n")

    def test_cli_regenerates_clues_for_seeded_runs(self) -> None:
        stdout = StringIO()
        puzzle = build_test_puzzle()

        with (
            redirect_stdout(stdout),
            patch("sys.argv", ["byewords", "--seed", "snail", "--regenerate-clues"]),
            patch("byewords.cli.load_default_inputs", return_value=((), {})),
            patch("byewords.cli.generate_puzzle_cached") as generate_cached,
            patch("byewords.cli.run_clue_regeneration") as regenerate,
            patch("byewords.cli.render_puzzle_text", return_value="regenerated puzzle"),
            patch("byewords.cli._refresh_puzzle_clues", side_effect=lambda current, clue_bank: current) as refresh,
        ):
            generate_cached.return_value = puzzle
            exit_code = main()

        self.assertEqual(exit_code, 0)
        regenerate.assert_called_once()
        refresh.assert_called_once()
        self.assertEqual(stdout.getvalue(), "regenerated puzzle\n")

    def test_cli_renders_animation_to_stderr_when_interactive(self) -> None:
        stdout = StringIO()
        stderr = FakeTty()

        def fake_generate(
            seeds: tuple[str, ...],
            lexicon_words: tuple[str, ...],
            clue_bank: dict[str, tuple[str, ...]],
            *,
            progress_callback,
        ) -> object:
            self.assertEqual(seeds, ("snail",))
            progress_callback(
                ProgressUpdate(
                    stage="search",
                    message="Locked 1/5 rows",
                    partial_rows=("snail",),
                )
            )
            progress_callback(
                ProgressUpdate(
                    stage="runtime_report",
                    message="Runtime report: semantic on",
                    runtime_report=RuntimeReport(
                        requested_seeds=("snail",),
                        normalized_seeds=("snail",),
                        available_seeds=("snail",),
                        candidate_count=10,
                        candidate_window_sizes=(10,),
                        semantic_ordering=True,
                        used_demo_grid=False,
                        budget_exhausted=False,
                        used_budget_fallback=False,
                        selected_theme_words=("snail",),
                        selected_theme_subset=("eases", "antra", "donna"),
                        selected_theme_weakest_link=0.25,
                    ),
                )
            )
            progress_callback(
                ProgressUpdate(
                    stage="candidate_solution",
                    message="Found a complete candidate grid; continuing search",
                    partial_rows=("snail", "oases", "atone", "ileum", "lends"),
                )
            )
            progress_callback(
                ProgressUpdate(
                    stage="solution",
                    message="Built a puzzle",
                    partial_rows=("snail", "oases", "atone", "ileum", "lends"),
                )
            )
            return object()

        with (
            redirect_stdout(stdout),
            patch("sys.stderr", stderr),
            patch("sys.argv", ["byewords", "--seed", "snail"]),
            patch("byewords.cli.load_default_inputs", return_value=((), {})),
            patch("byewords.cli.generate_puzzle_cached", side_effect=fake_generate),
            patch("byewords.cli.render_puzzle_text", return_value="animated puzzle"),
        ):
            exit_code = main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue(), "animated puzzle\n")
        self.assertIn("Found a complete candidate grid; continuing search", stderr.getvalue())
        self.assertIn("S N A I L", stderr.getvalue())
        self.assertIn("runtime: semantic=on fallback=no theme_subset=EASES, ANTRA, DONNA weakest_link=0.250", stderr.getvalue())
        self.assertIn("\x1b[?25l", stderr.getvalue())

    def test_cli_persists_candidate_solution_notice_to_stdout_during_interactive_runs(self) -> None:
        stdout = FakeTty()
        stderr = FakeTty()

        def fake_generate(
            seeds: tuple[str, ...],
            lexicon_words: tuple[str, ...],
            clue_bank: dict[str, tuple[str, ...]],
            *,
            progress_callback,
        ) -> object:
            progress_callback(
                ProgressUpdate(
                    stage="candidate_solution",
                    message="Found a complete candidate grid; continuing search",
                    partial_rows=("piano", "input", "skirt", "tense", "edger"),
                )
            )
            progress_callback(
                ProgressUpdate(
                    stage="solution",
                    message="Built a puzzle",
                    partial_rows=("piano", "input", "skirt", "tense", "edger"),
                )
            )
            return object()

        with (
            patch("sys.stdout", stdout),
            patch("sys.stderr", stderr),
            patch("sys.argv", ["byewords", "--seed", "piano"]),
            patch("byewords.cli.load_default_inputs", return_value=((), {})),
            patch("byewords.cli.generate_puzzle_cached", side_effect=fake_generate),
            patch("byewords.cli.render_puzzle_text", return_value="interactive puzzle"),
        ):
            exit_code = main()

        self.assertEqual(exit_code, 0)
        self.assertIn("Found a complete candidate grid; continuing search", stdout.getvalue())
        self.assertIn("interactive puzzle", stdout.getvalue())

    def test_parse_args_supports_positional_seeds(self) -> None:
        args = parse_args(["snail", "eases"])

        self.assertEqual(args.seeds, ("snail", "eases"))
        self.assertEqual(args.format, "text")
        self.assertIsNone(args.output)
        self.assertFalse(args.regenerate_clues)

    def test_parse_args_supports_repeated_seed_flags(self) -> None:
        args = parse_args(["--seed", "snail", "--seed", "eases"])

        self.assertEqual(args.seeds, ("snail", "eases"))

    def test_parse_args_defaults_to_empty_seed_list(self) -> None:
        args = parse_args([])

        self.assertEqual(args.seeds, ())

    def test_parse_args_accepts_puz_format_and_output(self) -> None:
        args = parse_args(["--format", "puz", "--output", "mini.puz"])

        self.assertEqual(args.format, "puz")
        self.assertEqual(args.output, "mini.puz")

    def test_parse_args_accepts_regenerate_clues(self) -> None:
        args = parse_args(["--seed", "snail", "--regenerate-clues"])

        self.assertTrue(args.regenerate_clues)

    def test_parse_args_rejects_mixed_seed_styles(self) -> None:
        with self.assertRaises(SystemExit), patch("sys.stderr", new_callable=StringIO):
            parse_args(["snail", "--seed", "eases"])


if __name__ == "__main__":
    unittest.main()
