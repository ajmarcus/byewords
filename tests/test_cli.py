import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from byewords.cli import main, parse_args


class TestCli(unittest.TestCase):
    def test_cli_prints_expected_message(self) -> None:
        buf = StringIO()

        with redirect_stdout(buf), patch("sys.argv", ["byewords"]):
            main()

        output = buf.getvalue()
        self.assertIn("BYEWORDS Mini", output)
        self.assertIn("Across", output)
        self.assertIn("Down", output)

    def test_cli_returns_error_for_seed_that_is_not_in_the_lexicon(self) -> None:
        stdout = StringIO()

        with redirect_stdout(stdout), patch("sys.argv", ["byewords", "--seed", "zzzzz"]):
            exit_code = main()

        self.assertEqual(exit_code, 0)
        self.assertIn("BYEWORDS Mini", stdout.getvalue())

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

    def test_parse_args_supports_positional_seeds(self) -> None:
        self.assertEqual(parse_args(["snail", "eases"]), ("snail", "eases"))

    def test_parse_args_supports_repeated_seed_flags(self) -> None:
        self.assertEqual(parse_args(["--seed", "snail", "--seed", "eases"]), ("snail", "eases"))

    def test_parse_args_defaults_to_empty_seed_list(self) -> None:
        self.assertEqual(parse_args([]), ())

    def test_parse_args_rejects_mixed_seed_styles(self) -> None:
        with self.assertRaises(SystemExit), patch("sys.stderr", new_callable=StringIO):
            parse_args(["snail", "--seed", "eases"])


if __name__ == "__main__":
    unittest.main()
