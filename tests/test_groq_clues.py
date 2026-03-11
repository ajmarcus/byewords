import json
import threading
import time
import unittest
from email.message import Message
from io import StringIO
from tempfile import TemporaryDirectory
from pathlib import Path
from typing import cast
from unittest.mock import Mock, patch
from urllib.error import HTTPError

from byewords.groq_clues import (
    DEFAULT_CLUE_COUNT,
    DEFAULT_PARALLELISM,
    DEFAULT_USER_AGENT,
    GracefulExit,
    MODEL_NAME,
    GroqClient,
    StatusReporter,
    answer_needs_new_clue,
    build_clue_payload,
    cached_clues_for_answer,
    format_clue_package,
    generate_clue_package,
    generate_clue_packages_parallel,
    main,
    parse_clue_package,
    parse_args,
    regenerate_clues,
    require_api_key,
    select_answers_to_clue,
)


class FakeGroqClient:
    def __init__(self) -> None:
        self.payloads: list[dict[str, object]] = []
        self._lock = threading.Lock()

    def create_chat_completion(self, payload: dict[str, object]) -> dict[str, object]:
        with self._lock:
            self.payloads.append(payload)

        answer = _answer_from_payload(payload)
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"clues": [f"{answer.title()} option A", f"{answer.title()} option B"]}
                        )
                    }
                }
            ]
        }


class SlowFakeGroqClient(FakeGroqClient):
    def create_chat_completion(self, payload: dict[str, object]) -> dict[str, object]:
        answer = _answer_from_payload(payload)
        if answer == "abase":
            time.sleep(0.02)
        return super().create_chat_completion(payload)


class StopAfterOneReporter(StatusReporter):
    def __init__(self, stream: StringIO, total: int, stop_event: threading.Event) -> None:
        super().__init__(stream, total)
        self.stop_event = stop_event

    def completed_word(self) -> None:
        super().completed_word()
        if self.completed == 1:
            self.stop_event.set()


class TestGroqClues(unittest.TestCase):
    def test_parse_args_uses_five_way_parallelism_by_default(self) -> None:
        args = parse_args([])

        self.assertEqual(args.parallelism, DEFAULT_PARALLELISM)
        self.assertEqual(args.parallelism, 5)

    def test_parse_args_allows_empty_answer_list_for_bulk_mode(self) -> None:
        args = parse_args(["--parallelism", "3", "--limit", "2", "--json"])

        self.assertEqual(args.answers, ())
        self.assertIsNone(args.puzzle_uuid)
        self.assertEqual(args.count, DEFAULT_CLUE_COUNT)
        self.assertEqual(args.parallelism, 3)
        self.assertEqual(args.limit, 2)
        self.assertTrue(args.json)

    def test_parse_args_extracts_optional_puzzle_uuid(self) -> None:
        args = parse_args(["019577fd-8d7e-7a3d-9a4b-c2f6b6a1d233", "snail"])

        self.assertEqual(args.puzzle_uuid, "019577fd-8d7e-7a3d-9a4b-c2f6b6a1d233")
        self.assertEqual(args.answers, ("snail",))

    def test_parse_args_accepts_force_regeneration(self) -> None:
        args = parse_args(["--force", "snail"])

        self.assertTrue(args.force)
        self.assertEqual(args.answers, ("snail",))

    def test_parse_args_rejects_counts_other_than_two(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["--count", "3"])

    def test_require_api_key_explains_how_to_fix_missing_env_var(self) -> None:
        with self.assertRaisesRegex(ValueError, "GROQ_API_KEY is not set"):
            require_api_key({})

    def test_groq_client_sends_a_user_agent_header(self) -> None:
        response = Mock()
        response.read.return_value = b'{"choices": [{"message": {"content": "[]"}}]}'
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=None)

        with patch("byewords.groq_clues.urlopen", return_value=response) as mock_urlopen:
            GroqClient("test-key").create_chat_completion({"messages": []})

        request = mock_urlopen.call_args.args[0]
        self.assertEqual(request.get_header("User-agent"), DEFAULT_USER_AGENT)

    def test_groq_client_retries_after_429_using_retry_after_header(self) -> None:
        headers = Message()
        headers["Retry-After"] = "7"
        rate_limited = HTTPError(
            url="https://example.com",
            code=429,
            msg="Too Many Requests",
            hdrs=headers,
            fp=None,
        )
        response = Mock()
        response.read.return_value = b'{"choices": [{"message": {"content": "{\\"clues\\": [\\"A\\", \\"B\\"]}"}}]}'
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=None)
        mock_sleep = Mock()

        with (
            patch("byewords.groq_clues.urlopen", side_effect=[rate_limited, response]) as mock_urlopen,
        ):
            payload = GroqClient("test-key", sleep_fn=mock_sleep).create_chat_completion({"messages": []})

        self.assertEqual(mock_urlopen.call_count, 2)
        mock_sleep.assert_called_once_with(7.0)
        self.assertEqual(payload["choices"][0]["message"]["content"], '{"clues": ["A", "B"]}')

    def test_status_reporter_logs_every_fifty_completed_words(self) -> None:
        progress = StringIO()
        reporter = StatusReporter(progress, total=51)
        with TemporaryDirectory() as directory:
            clue_bank_path = str(Path(directory, "clue_bank.json"))
            Path(clue_bank_path).write_text("{}\n", encoding="utf-8")
            answers = tuple(f"word{index:02d}" for index in range(51))

            packages = generate_clue_packages_parallel(
                client=FakeGroqClient(),
                answers=answers,
                clue_bank={},
                clue_bank_path=clue_bank_path,
                count=DEFAULT_CLUE_COUNT,
                parallelism=5,
                reporter=reporter,
            )

        self.assertEqual(len(packages), 51)
        self.assertIn("50 out of 51 words completed", progress.getvalue())
        self.assertIn("51 out of 51 words completed", progress.getvalue())

    def test_generate_clue_packages_parallel_supports_graceful_exit(self) -> None:
        progress = StringIO()
        stop_event = threading.Event()
        reporter = StopAfterOneReporter(progress, total=3, stop_event=stop_event)

        with TemporaryDirectory() as directory:
            clue_bank_path = str(Path(directory, "clue_bank.json"))
            Path(clue_bank_path).write_text("{}\n", encoding="utf-8")

            with self.assertRaises(GracefulExit) as context:
                generate_clue_packages_parallel(
                    client=FakeGroqClient(),
                    answers=("alpha", "bravo", "charl"),
                    clue_bank={},
                    clue_bank_path=clue_bank_path,
                    count=DEFAULT_CLUE_COUNT,
                    parallelism=1,
                    reporter=reporter,
                    stop_event=stop_event,
                )

        self.assertEqual(context.exception.completed, 1)
        self.assertEqual(context.exception.total, 3)

    def test_answer_needs_new_clue_skips_non_generic_entries(self) -> None:
        self.assertFalse(
            answer_needs_new_clue("snail", {"snail": ("Slow walker carrying its whole rent situation",)})
        )
        self.assertTrue(answer_needs_new_clue("asked", {"asked": ('Past tense of "ask"',)}))
        self.assertTrue(answer_needs_new_clue("abase", {}))

    def test_cached_clues_for_answer_ignores_generic_entries(self) -> None:
        self.assertIsNone(cached_clues_for_answer("asked", {"asked": ('Past tense of "ask"',)}))
        self.assertEqual(
            cached_clues_for_answer("snail", {"snail": ("Slow walker carrying its whole rent situation",)}),
            ("Slow walker carrying its whole rent situation",),
        )

    def test_select_answers_to_clue_uses_full_lexicon_and_skips_already_clued_words(self) -> None:
        selection = select_answers_to_clue(
            requested_answers=(),
            lexicon_words=("snail", "asked", "abase"),
            clue_bank={
                "snail": ("Slow walker carrying its whole rent situation",),
                "asked": ('Past tense of "ask"',),
            },
            limit=0,
        )

        self.assertEqual(selection.queued_answers, ("asked", "abase"))
        self.assertEqual(selection.skipped_answers, ("snail",))

    def test_select_answers_to_clue_force_mode_queues_full_lexicon(self) -> None:
        selection = select_answers_to_clue(
            requested_answers=(),
            lexicon_words=("snail", "asked", "abase"),
            clue_bank={"snail": ("Slow walker carrying its whole rent situation",)},
            limit=0,
            force=True,
        )

        self.assertEqual(selection.queued_answers, ("snail", "asked", "abase"))
        self.assertEqual(selection.skipped_answers, ())

    def test_select_answers_to_clue_keeps_requested_answers_even_when_cached(self) -> None:
        selection = select_answers_to_clue(
            requested_answers=("snail", "abase"),
            lexicon_words=("snail", "abase"),
            clue_bank={"snail": ("Slow walker carrying its whole rent situation",)},
            limit=0,
        )

        self.assertEqual(selection.queued_answers, ("snail", "abase"))
        self.assertEqual(selection.skipped_answers, ())

    def test_build_clue_payload_includes_quality_examples_and_two_clue_contract(self) -> None:
        payload = build_clue_payload(
            answer="snail",
            count=DEFAULT_CLUE_COUNT,
        )

        response_format = payload["response_format"]
        system_prompt = payload["messages"][0]["content"]
        user_prompt = payload["messages"][1]["content"]
        self.assertEqual(payload["model"], MODEL_NAME)
        self.assertEqual(response_format["type"], "json_schema")
        self.assertTrue(response_format["json_schema"]["strict"])
        self.assertIn("Example 1", system_prompt)
        self.assertIn("Answer: TRADE", system_prompt)
        self.assertIn("Answer: CABLE", system_prompt)
        self.assertIn("best crossword editors of all time", system_prompt)
        self.assertIn("Write two funny, memorable and precise clues", system_prompt)
        self.assertIn("clear reference to the provided answer", system_prompt)
        self.assertIn("not be an obsure reference", system_prompt)
        self.assertIn("Return only a JSON object with one key, 'clues'", system_prompt)
        self.assertIn("Return exactly 2 standalone clues", user_prompt)
        self.assertIn("Return only a JSON object with one key, 'clues'", user_prompt)
        schema = response_format["json_schema"]["schema"]
        self.assertEqual(schema["type"], "object")
        self.assertEqual(schema["required"], ["clues"])
        self.assertEqual(schema["additionalProperties"], False)
        self.assertEqual(schema["properties"]["clues"]["type"], "array")
        self.assertEqual(schema["properties"]["clues"]["minItems"], 2)
        self.assertEqual(schema["properties"]["clues"]["maxItems"], 2)
        self.assertEqual(schema["properties"]["clues"]["items"]["type"], "string")

    def test_generate_clue_package_makes_one_structured_request(self) -> None:
        client = FakeGroqClient()

        package = generate_clue_package(
            client=client,
            answer="snail",
            clue_bank={},
            clue_bank_path="/tmp/test_clue_bank.json",
            lock=threading.Lock(),
            count=DEFAULT_CLUE_COUNT,
        )

        self.assertEqual(len(client.payloads), 1)
        self.assertEqual(package.answer, "snail")
        self.assertFalse(package.cached)
        self.assertEqual(package.clues, ("Snail option A", "Snail option B"))

    def test_generate_clue_packages_parallel_preserves_input_order(self) -> None:
        client = SlowFakeGroqClient()
        with TemporaryDirectory() as directory:
            clue_bank_path = str(Path(directory, "clue_bank.json"))
            Path(clue_bank_path).write_text("{}\n", encoding="utf-8")

            packages = generate_clue_packages_parallel(
                client=client,
                answers=("abase", "asked"),
                clue_bank={},
                clue_bank_path=clue_bank_path,
                count=DEFAULT_CLUE_COUNT,
                parallelism=2,
            )

        self.assertEqual(tuple(package.answer for package in packages), ("abase", "asked"))

    def test_parse_clue_package_returns_clue_list(self) -> None:
        clues = parse_clue_package(
            json.dumps(
                {
                    "clues": [
                        "Slowpoke with a shell",
                        "Garden guest wearing its apartment",
                    ]
                }
            )
        )

        self.assertEqual(
            clues,
            (
                "Slowpoke with a shell",
                "Garden guest wearing its apartment",
            ),
        )

    def test_format_clue_package_lists_only_clues(self) -> None:
        with TemporaryDirectory() as directory:
            clue_bank_path = str(Path(directory, "clue_bank.json"))
            Path(clue_bank_path).write_text("{}\n", encoding="utf-8")
            rendered = format_clue_package(
                generate_clue_package(
                    client=FakeGroqClient(),
                    answer="snail",
                    clue_bank={},
                    clue_bank_path=clue_bank_path,
                    lock=threading.Lock(),
                    count=DEFAULT_CLUE_COUNT,
                )
            )

        self.assertIn("SNAIL", rendered)
        self.assertIn("Cached: no", rendered)
        self.assertIn("1. Snail option A", rendered)
        self.assertIn("2. Snail option B", rendered)

    def test_generate_clue_package_uses_cached_clue_without_calling_groq(self) -> None:
        client = FakeGroqClient()

        package = generate_clue_package(
            client=client,
            answer="snail",
            clue_bank={
                "snail": (
                    "Slow walker carrying its whole rent situation",
                    "Creature living the ultimate one-bag lifestyle",
                )
            },
            clue_bank_path="/tmp/test_clue_bank.json",
            lock=threading.Lock(),
            count=DEFAULT_CLUE_COUNT,
        )

        self.assertEqual(len(client.payloads), 0)
        self.assertTrue(package.cached)
        self.assertEqual(
            package.clues,
            (
                "Slow walker carrying its whole rent situation",
                "Creature living the ultimate one-bag lifestyle",
            ),
        )

    def test_generate_clue_package_force_regenerates_even_with_cached_clues(self) -> None:
        client = FakeGroqClient()

        package = generate_clue_package(
            client=client,
            answer="snail",
            clue_bank={"snail": ("Slow walker carrying its whole rent situation",)},
            clue_bank_path="/tmp/test_clue_bank.json",
            lock=threading.Lock(),
            count=DEFAULT_CLUE_COUNT,
            force=True,
        )

        self.assertEqual(len(client.payloads), 1)
        self.assertFalse(package.cached)
        self.assertEqual(package.clues, ("Snail option A", "Snail option B"))

    def test_generate_clue_package_writes_generated_clues_to_clue_bank(self) -> None:
        client = FakeGroqClient()

        with TemporaryDirectory() as directory:
            clue_bank_path = str(Path(directory, "clue_bank.json"))
            Path(clue_bank_path).write_text("{}\n", encoding="utf-8")
            clue_bank: dict[str, tuple[str, ...]] = {}

            package = generate_clue_package(
                client=client,
                answer="snail",
                clue_bank=clue_bank,
                clue_bank_path=clue_bank_path,
                lock=threading.Lock(),
                count=DEFAULT_CLUE_COUNT,
            )

            persisted = json.loads(Path(clue_bank_path).read_text(encoding="utf-8"))

        self.assertFalse(package.cached)
        self.assertEqual(clue_bank["snail"], ("Snail option A", "Snail option B"))
        self.assertEqual(persisted["snail"], ["Snail option A", "Snail option B"])

    def test_main_reports_missing_api_key_to_stderr(self) -> None:
        stdout = StringIO()
        stderr = StringIO()

        with patch(
            "byewords.groq_clues.load_default_answer_inputs",
            return_value=(("asked",), {"asked": ('Past tense of "ask"',)}),
        ), patch("byewords.groq_clues.default_clue_bank_path", return_value="/tmp/test_clue_bank.json"):
            exit_code = main([], env={}, stdout=stdout, stderr=stderr)

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("GROQ_API_KEY is not set", stderr.getvalue())

    def test_main_uses_bulk_mode_and_skips_existing_handwritten_clues(self) -> None:
        stdout = StringIO()
        stderr = StringIO()

        with (
            patch(
                "byewords.groq_clues.load_default_answer_inputs",
                return_value=(
                    ("snail", "asked", "abase"),
                    {
                        "snail": ("Slow walker carrying its whole rent situation",),
                        "asked": ('Past tense of "ask"',),
                    },
                ),
            ),
            patch("byewords.groq_clues.default_clue_bank_path", return_value="/tmp/test_clue_bank.json"),
            patch("byewords.groq_clues.GroqClient", return_value=FakeGroqClient()),
            patch("byewords.groq_clues.persist_clue_bank"),
        ):
            exit_code = main(
                ["--json", "--parallelism", "2"],
                env={"GROQ_API_KEY": "test-key"},
                stdout=stdout,
                stderr=stderr,
            )

        self.assertEqual(exit_code, 0)
        parsed = json.loads(stdout.getvalue())
        self.assertEqual([item["answer"] for item in parsed], ["asked", "abase"])
        self.assertIn("Skipping 1 answers", stderr.getvalue())

    def test_main_returns_cached_requested_answer(self) -> None:
        stdout = StringIO()
        stderr = StringIO()

        with (
            patch(
                "byewords.groq_clues.load_default_answer_inputs",
                return_value=(
                    ("snail",),
                    {"snail": ("Slow walker carrying its whole rent situation",)},
                ),
            ),
            patch("byewords.groq_clues.default_clue_bank_path", return_value="/tmp/test_clue_bank.json"),
            patch("byewords.groq_clues.GroqClient", return_value=FakeGroqClient()),
        ):
            exit_code = main(
                ["--json", "snail"],
                env={"GROQ_API_KEY": "test-key"},
                stdout=stdout,
                stderr=stderr,
            )

        self.assertEqual(exit_code, 0)
        parsed = json.loads(stdout.getvalue())
        self.assertEqual(
            parsed,
            [
                {
                    "answer": "snail",
                    "cached": True,
                    "clues": ["Slow walker carrying its whole rent situation"],
                }
            ],
        )
        self.assertEqual(stderr.getvalue(), "")

    def test_main_returns_cached_requested_answer_without_api_key(self) -> None:
        stdout = StringIO()
        stderr = StringIO()

        with (
            patch(
                "byewords.groq_clues.load_default_answer_inputs",
                return_value=(
                    ("snail",),
                    {"snail": ("Slow walker carrying its whole rent situation",)},
                ),
            ),
            patch("byewords.groq_clues.default_clue_bank_path", return_value="/tmp/test_clue_bank.json"),
        ):
            exit_code = main(
                ["--json", "snail"],
                env={},
                stdout=stdout,
                stderr=stderr,
            )

        self.assertEqual(exit_code, 0)
        parsed = json.loads(stdout.getvalue())
        self.assertEqual(parsed[0]["cached"], True)
        self.assertEqual(stderr.getvalue(), "")

    def test_regenerate_clues_returns_cached_packages_without_api_key_when_force_is_false(self) -> None:
        packages = regenerate_clues(
            answers=("snail",),
            clue_bank={"snail": ("Slow walker carrying its whole rent situation",)},
            clue_bank_path="/tmp/test_clue_bank.json",
            env={},
            errors=StringIO(),
            force=False,
        )

        self.assertEqual(len(packages), 1)
        self.assertTrue(packages[0].cached)

    def test_main_force_regenerates_cached_requested_answer(self) -> None:
        stdout = StringIO()
        stderr = StringIO()

        with (
            patch(
                "byewords.groq_clues.load_default_answer_inputs",
                return_value=(
                    ("snail",),
                    {"snail": ("Slow walker carrying its whole rent situation",)},
                ),
            ),
            patch("byewords.groq_clues.default_clue_bank_path", return_value="/tmp/test_clue_bank.json"),
            patch("byewords.groq_clues.GroqClient", return_value=FakeGroqClient()),
            patch("byewords.groq_clues.persist_clue_bank"),
        ):
            exit_code = main(
                ["--json", "--force", "snail"],
                env={"GROQ_API_KEY": "test-key"},
                stdout=stdout,
                stderr=stderr,
            )

        self.assertEqual(exit_code, 0)
        parsed = json.loads(stdout.getvalue())
        self.assertEqual(parsed[0]["cached"], False)
        self.assertEqual(parsed[0]["clues"], ["Snail option A", "Snail option B"])

    def test_main_json_output_wraps_packages_when_puzzle_uuid_is_supplied(self) -> None:
        stdout = StringIO()
        stderr = StringIO()

        with (
            patch(
                "byewords.groq_clues.load_default_answer_inputs",
                return_value=(
                    ("snail",),
                    {"snail": ("Slow walker carrying its whole rent situation",)},
                ),
            ),
            patch("byewords.groq_clues.default_clue_bank_path", return_value="/tmp/test_clue_bank.json"),
        ):
            exit_code = main(
                ["--json", "019577fd-8d7e-7a3d-9a4b-c2f6b6a1d233", "snail"],
                env={},
                stdout=stdout,
                stderr=stderr,
            )

        self.assertEqual(exit_code, 0)
        parsed = json.loads(stdout.getvalue())
        self.assertEqual(parsed["puzzle_uuid"], "019577fd-8d7e-7a3d-9a4b-c2f6b6a1d233")
        self.assertEqual(parsed["packages"][0]["answer"], "snail")

    def test_main_uses_puzzle_uuid_to_lookup_answers_when_none_are_supplied(self) -> None:
        stdout = StringIO()
        stderr = StringIO()

        with (
            patch(
                "byewords.groq_clues.load_default_answer_inputs",
                return_value=(("snail",), {"snail": ("Slow walker carrying its whole rent situation",)}),
            ),
            patch("byewords.groq_clues.puzzle_answers_for_id", return_value=("snail",)) as puzzle_answers,
            patch("byewords.groq_clues.default_clue_bank_path", return_value="/tmp/test_clue_bank.json"),
        ):
            exit_code = main(
                ["--json", "019577fd-8d7e-7a3d-9a4b-c2f6b6a1d233"],
                env={},
                stdout=stdout,
                stderr=stderr,
            )

        self.assertEqual(exit_code, 0)
        puzzle_answers.assert_called_once_with("019577fd-8d7e-7a3d-9a4b-c2f6b6a1d233")


def _answer_from_payload(payload: dict[str, object]) -> str:
    messages = payload["messages"]
    if not isinstance(messages, list):
        raise AssertionError("payload messages should be a list")
    message = messages[1]
    if not isinstance(message, dict):
        raise AssertionError("payload message should be an object")
    typed_message = cast(dict[str, object], message)
    content = typed_message["content"]
    if not isinstance(content, str):
        raise AssertionError("payload content should be a string")
    first_line = content.splitlines()[0]
    return first_line.removeprefix("Answer: ").strip()


if __name__ == "__main__":
    unittest.main()
