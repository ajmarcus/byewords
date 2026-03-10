import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from byewords.cli import main, parse_args
from byewords.types import ProgressUpdate


class FakeTty(StringIO):
    def isatty(self) -> bool:
        return True


class TestCli(unittest.TestCase):
    def test_cli_prints_expected_message(self) -> None:
        buf = StringIO()

        with (
            redirect_stdout(buf),
            patch("sys.argv", ["byewords"]),
            patch("byewords.cli.load_default_inputs", return_value=((), {})),
            patch("byewords.cli.generate_puzzle_cached") as generate_cached,
            patch("byewords.cli.render_puzzle_text", return_value="generated puzzle"),
        ):
            generate_cached.return_value = object()
            main()

        output = buf.getvalue()
        generate_cached.assert_called_once_with((), (), {})
        self.assertEqual(output, "generated puzzle\n")

    def test_cli_writes_text_output_to_file(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir, "puzzle.txt")
            with (
                patch("sys.argv", ["byewords", "--output", str(output_path)]),
                patch("byewords.cli.load_default_inputs", return_value=((), {})),
                patch("byewords.cli.generate_puzzle_cached") as generate_cached,
                patch("byewords.cli.render_puzzle_text", return_value="generated puzzle"),
            ):
                generate_cached.return_value = object()
                exit_code = main()

            written_text = output_path.read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertEqual(written_text, "generated puzzle\n")

    def test_cli_writes_puz_output_to_file(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir, "puzzle.puz")
            with (
                patch("sys.argv", ["byewords", "--format", "puz", "--output", str(output_path)]),
                patch("byewords.cli.load_default_inputs", return_value=((), {})),
                patch("byewords.cli.generate_puzzle_cached") as generate_cached,
                patch("byewords.cli.puzzle_to_puz_bytes", return_value=b"puz-bytes"),
            ):
                generate_cached.return_value = object()
                exit_code = main()

            written_bytes = output_path.read_bytes()

        self.assertEqual(exit_code, 0)
        self.assertEqual(written_bytes, b"puz-bytes")

    def test_cli_rejects_binary_output_to_terminal(self) -> None:
        stdout = FakeTty()

        with (
            patch("sys.stdout", stdout),
            patch("sys.argv", ["byewords", "--format", "puz"]),
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
            patch("sys.argv", ["byewords"]),
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
        generate_cached.assert_called_once_with(("snail",), (), {})
        self.assertEqual(stdout.getvalue(), "cached puzzle\n")

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
                    stage="solution",
                    message="Locked the final grid",
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
        self.assertIn("Locked the final grid", stderr.getvalue())
        self.assertIn("S N A I L", stderr.getvalue())
        self.assertIn("\x1b[?25l", stderr.getvalue())

    def test_parse_args_supports_positional_seeds(self) -> None:
        args = parse_args(["snail", "eases"])

        self.assertEqual(args.seeds, ("snail", "eases"))
        self.assertEqual(args.format, "text")
        self.assertIsNone(args.output)

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

    def test_parse_args_rejects_mixed_seed_styles(self) -> None:
        with self.assertRaises(SystemExit), patch("sys.stderr", new_callable=StringIO):
            parse_args(["snail", "--seed", "eases"])


if __name__ == "__main__":
    unittest.main()
