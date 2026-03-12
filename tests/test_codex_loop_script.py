import os
import stat
import subprocess
import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "tools" / "run_codex_theme_loop.sh"
EXPECTED_PROMPT = textwrap.dedent(
    """\
    read docs/theme.md and continue implementation in
    small testable chunks that represent a
    meaningful milestone you can measure and
    confirm. update docs/theme.md when you are
    done with the chunk. finally commit and push
    your changes. once the full docs/theme.md
    plan is done, just return <DONE>.
    """
).rstrip()


class TestCodexLoopScript(unittest.TestCase):
    def test_script_loops_until_done(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fake_bin = temp_path / "bin"
            fake_bin.mkdir()
            state_path = temp_path / "state.txt"
            prompt_path = temp_path / "prompt.txt"
            argv_path = temp_path / "argv.txt"
            self._write_fake_codex(
                fake_bin / "codex",
                state_path=state_path,
                prompt_path=prompt_path,
                argv_path=argv_path,
                done_after=2,
            )

            result = subprocess.run(
                [str(SCRIPT_PATH)],
                cwd=REPO_ROOT,
                env=self._env_with_fake_codex(fake_bin),
                check=False,
                capture_output=True,
                text=True,
            )
            state_text = state_path.read_text(encoding="utf-8")
            prompt_text = prompt_path.read_text(encoding="utf-8")
            argv_text = argv_path.read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["working", "<DONE>"])
        self.assertIn("iteration 1/100", result.stderr)
        self.assertIn("iteration 2/100", result.stderr)
        self.assertEqual(state_text, "2")
        self.assertEqual(prompt_text, EXPECTED_PROMPT)
        self.assertIn("--sandbox\nworkspace-write\n", argv_text)
        self.assertIn(f"-C\n{REPO_ROOT}\n", argv_text)

    def test_script_fails_after_max_iterations(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fake_bin = temp_path / "bin"
            fake_bin.mkdir()
            state_path = temp_path / "state.txt"
            prompt_path = temp_path / "prompt.txt"
            self._write_fake_codex(
                fake_bin / "codex",
                state_path=state_path,
                prompt_path=prompt_path,
                argv_path=temp_path / "argv.txt",
                done_after=99,
            )
            env = self._env_with_fake_codex(fake_bin)
            env["CODEX_THEME_LOOP_MAX_ITERS"] = "2"

            result = subprocess.run(
                [str(SCRIPT_PATH)],
                cwd=REPO_ROOT,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            state_text = state_path.read_text(encoding="utf-8")
            prompt_text = prompt_path.read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout.splitlines(), ["working", "working"])
        self.assertIn("without receiving <DONE>", result.stderr)
        self.assertEqual(state_text, "2")
        self.assertEqual(prompt_text, EXPECTED_PROMPT)

    def test_script_treats_done_with_surrounding_whitespace_as_complete(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fake_bin = temp_path / "bin"
            fake_bin.mkdir()
            state_path = temp_path / "state.txt"
            prompt_path = temp_path / "prompt.txt"
            self._write_fake_codex(
                fake_bin / "codex",
                state_path=state_path,
                prompt_path=prompt_path,
                argv_path=temp_path / "argv.txt",
                done_after=2,
                done_message="  <DONE>  \n",
            )

            result = subprocess.run(
                [str(SCRIPT_PATH)],
                cwd=REPO_ROOT,
                env=self._env_with_fake_codex(fake_bin),
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["working", "  <DONE>  "])

    def test_script_suppresses_codex_stdout_noise(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fake_bin = temp_path / "bin"
            fake_bin.mkdir()
            state_path = temp_path / "state.txt"
            prompt_path = temp_path / "prompt.txt"
            self._write_fake_codex(
                fake_bin / "codex",
                state_path=state_path,
                prompt_path=prompt_path,
                argv_path=temp_path / "argv.txt",
                done_after=2,
                stdout_message="diff --git a/file b/file\n+new line\n",
            )

            result = subprocess.run(
                [str(SCRIPT_PATH)],
                cwd=REPO_ROOT,
                env=self._env_with_fake_codex(fake_bin),
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["working", "<DONE>"])
        self.assertNotIn("diff --git", result.stdout)

    def test_script_fails_when_codex_does_not_write_output(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fake_bin = temp_path / "bin"
            fake_bin.mkdir()
            state_path = temp_path / "state.txt"
            prompt_path = temp_path / "prompt.txt"
            self._write_fake_codex(
                fake_bin / "codex",
                state_path=state_path,
                prompt_path=prompt_path,
                argv_path=temp_path / "argv.txt",
                done_after=99,
                missing_output_calls=(2,),
            )
            env = self._env_with_fake_codex(fake_bin)
            env["CODEX_THEME_LOOP_MAX_ITERS"] = "2"

            result = subprocess.run(
                [str(SCRIPT_PATH)],
                cwd=REPO_ROOT,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            state_text = state_path.read_text(encoding="utf-8")
            prompt_text = prompt_path.read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout.splitlines(), ["working"])
        self.assertIn("did not produce a non-empty output file", result.stderr)
        self.assertEqual(state_text, "2")
        self.assertEqual(prompt_text, EXPECTED_PROMPT)

    def test_script_rejects_invalid_sleep_seconds(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fake_bin = temp_path / "bin"
            fake_bin.mkdir()
            env = self._env_with_fake_codex(fake_bin)
            env["CODEX_THEME_LOOP_SLEEP_SECONDS"] = "soon"

            result = subprocess.run(
                [str(SCRIPT_PATH)],
                cwd=REPO_ROOT,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertIn(
            "CODEX_THEME_LOOP_SLEEP_SECONDS must be a non-negative number",
            result.stderr,
        )

    def test_script_rejects_invalid_sandbox_mode(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fake_bin = temp_path / "bin"
            fake_bin.mkdir()
            env = self._env_with_fake_codex(fake_bin)
            env["CODEX_THEME_LOOP_SANDBOX"] = "editable"

            result = subprocess.run(
                [str(SCRIPT_PATH)],
                cwd=REPO_ROOT,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertIn(
            "CODEX_THEME_LOOP_SANDBOX must be one of",
            result.stderr,
        )

    def test_script_honors_overridden_sandbox_mode(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fake_bin = temp_path / "bin"
            fake_bin.mkdir()
            state_path = temp_path / "state.txt"
            prompt_path = temp_path / "prompt.txt"
            argv_path = temp_path / "argv.txt"
            self._write_fake_codex(
                fake_bin / "codex",
                state_path=state_path,
                prompt_path=prompt_path,
                argv_path=argv_path,
                done_after=1,
            )
            env = self._env_with_fake_codex(fake_bin)
            env["CODEX_THEME_LOOP_SANDBOX"] = "danger-full-access"

            result = subprocess.run(
                [str(SCRIPT_PATH)],
                cwd=REPO_ROOT,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            argv_text = argv_path.read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--sandbox\ndanger-full-access\n", argv_text)

    def _env_with_fake_codex(self, fake_bin: Path) -> dict[str, str]:
        env = os.environ.copy()
        env["PATH"] = f"{fake_bin}:{env['PATH']}"
        env["CODEX_THEME_LOOP_SLEEP_SECONDS"] = "0"
        return env

    def _write_fake_codex(
        self,
        path: Path,
        *,
        state_path: Path,
        prompt_path: Path,
        argv_path: Path,
        done_after: int,
        done_message: str = "<DONE>",
        missing_output_calls: tuple[int, ...] = (),
        stdout_message: str = "",
    ) -> None:
        script = textwrap.dedent(
            f"""\
            #!/usr/bin/env python3
            import pathlib
            import sys

            state_path = pathlib.Path({str(state_path)!r})
            prompt_path = pathlib.Path({str(prompt_path)!r})
            argv_path = pathlib.Path({str(argv_path)!r})
            done_after = {done_after}
            done_message = {done_message!r}
            missing_output_calls = {missing_output_calls!r}
            stdout_message = {stdout_message!r}

            args = sys.argv[1:]
            output_path = None
            prompt = None
            index = 0
            while index < len(args):
                arg = args[index]
                if arg in {{"-o", "--output-last-message"}}:
                    output_path = pathlib.Path(args[index + 1])
                    index += 2
                    continue
                if not arg.startswith("-"):
                    prompt = arg
                index += 1

            if output_path is None or prompt is None:
                raise SystemExit("missing expected codex arguments")

            argv_path.write_text("\\n".join(args) + "\\n", encoding="utf-8")
            current = int(state_path.read_text(encoding="utf-8")) if state_path.exists() else 0
            current += 1
            state_path.write_text(str(current), encoding="utf-8")
            prompt_path.write_text(prompt, encoding="utf-8")
            if stdout_message:
                print(stdout_message, end="")
            if current not in missing_output_calls:
                output_path.write_text(
                    done_message if current >= done_after else "working",
                    encoding="utf-8",
                )
            """
        )
        path.write_text(script, encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)


if __name__ == "__main__":
    unittest.main()
