import json
import threading
import time
import unittest
from datetime import date
from io import StringIO
from typing import cast
from unittest.mock import patch

from byewords.groq_clues import (
    MODEL_NAME,
    answer_needs_new_clue,
    build_clue_payload,
    build_research_payload,
    format_clue_package,
    generate_clue_package,
    generate_clue_packages_parallel,
    main,
    parse_clue_package,
    parse_args,
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
        if "response_format" not in payload:
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "VERDICT: HYBRID\n"
                                "HOOKS:\n"
                                f"- {answer.title()} can support a fair timely metaphor.\n"
                                "RISKS:\n"
                                "- Do not claim the answer itself was a headline."
                            )
                        }
                    }
                ]
            }

        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "answer": answer,
                                "research_verdict": "HYBRID",
                                "recent_hook_summary": f"{answer.title()} has a clean hybrid angle.",
                                "editor_note": "Lead with the crispest image.",
                                "best_index": 1,
                                "clues": [
                                    {
                                        "clue": f"{answer.title()} option A",
                                        "angle": "safer angle",
                                        "freshness": "hybrid",
                                        "why_it_works": "Clean and fair.",
                                    },
                                    {
                                        "clue": f"{answer.title()} option B",
                                        "angle": "bolder angle",
                                        "freshness": "hybrid",
                                        "why_it_works": "Sharper and more memorable.",
                                    },
                                ],
                            }
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


class TestGroqClues(unittest.TestCase):
    def test_parse_args_allows_empty_answer_list_for_bulk_mode(self) -> None:
        args = parse_args(["--parallelism", "3", "--limit", "2", "--json"])

        self.assertEqual(args.answers, [])
        self.assertEqual(args.parallelism, 3)
        self.assertEqual(args.limit, 2)
        self.assertTrue(args.json)

    def test_require_api_key_explains_how_to_fix_missing_env_var(self) -> None:
        with self.assertRaisesRegex(ValueError, "GROQ_API_KEY is not set"):
            require_api_key({})

    def test_answer_needs_new_clue_skips_non_generic_entries(self) -> None:
        self.assertFalse(
            answer_needs_new_clue("snail", {"snail": ("Slow walker carrying its whole rent situation",)})
        )
        self.assertTrue(answer_needs_new_clue("asked", {"asked": ('Past tense of "ask"',)}))
        self.assertTrue(answer_needs_new_clue("abase", {}))

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

    def test_build_research_payload_uses_browser_search_with_requested_model(self) -> None:
        payload = build_research_payload("snail", "", (), date(2026, 3, 7))

        self.assertEqual(payload["model"], MODEL_NAME)
        self.assertEqual(payload["tool_choice"], "required")
        self.assertEqual(payload["tools"], [{"type": "browser_search"}])
        self.assertIn("Answer: snail", payload["messages"][1]["content"])

    def test_build_clue_payload_includes_quality_examples(self) -> None:
        payload = build_clue_payload(
            answer="snail",
            theme="",
            recent_context=(),
            research_notes="VERDICT: HYBRID",
            count=4,
            current_date=date(2026, 3, 7),
        )

        response_format = payload["response_format"]
        system_prompt = payload["messages"][0]["content"]
        self.assertEqual(payload["model"], MODEL_NAME)
        self.assertEqual(response_format["type"], "json_schema")
        self.assertTrue(response_format["json_schema"]["strict"])
        self.assertIn("Example 1", system_prompt)
        self.assertIn("Answer: TRADE", system_prompt)
        self.assertIn("Answer: CABLE", system_prompt)

    def test_generate_clue_package_runs_research_then_structured_generation(self) -> None:
        client = FakeGroqClient()

        package = generate_clue_package(
            client=client,
            answer="snail",
            count=2,
            current_date=date(2026, 3, 7),
        )

        self.assertEqual(len(client.payloads), 2)
        self.assertEqual(package.answer, "snail")
        self.assertEqual(package.best_index, 1)
        self.assertEqual(package.clues[package.best_index].clue, "Snail option B")

    def test_generate_clue_packages_parallel_preserves_input_order(self) -> None:
        client = SlowFakeGroqClient()

        packages = generate_clue_packages_parallel(
            client=client,
            answers=("abase", "asked"),
            count=2,
            parallelism=2,
            current_date=date(2026, 3, 7),
        )

        self.assertEqual(tuple(package.answer for package in packages), ("abase", "asked"))

    def test_parse_clue_package_falls_back_to_first_option_for_bad_best_index(self) -> None:
        package = parse_clue_package(
            json.dumps(
                {
                    "answer": "snail",
                    "research_verdict": "TIMELESS",
                    "recent_hook_summary": "No clean recent hook.",
                    "editor_note": "Use the strongest evergreen angle.",
                    "best_index": 99,
                    "clues": [
                        {
                            "clue": "Slowpoke with a shell",
                            "angle": "straight image",
                            "freshness": "timeless",
                            "why_it_works": "Clean and fair.",
                        }
                    ],
                }
            )
        )

        self.assertEqual(package.best_index, 0)

    def test_format_clue_package_includes_best_marker(self) -> None:
        package = parse_clue_package(
            json.dumps(
                {
                    "answer": "snail",
                    "research_verdict": "HYBRID",
                    "recent_hook_summary": "Awards-season pacing jokes give a fair hybrid angle.",
                    "editor_note": "Lead with the cleaner joke.",
                    "best_index": 0,
                    "clues": [
                        {
                            "clue": "Mascot for a speech that needs the wrap-it-up music",
                            "angle": "memorable image",
                            "freshness": "hybrid",
                            "why_it_works": "It is vivid, answer-specific, and fair.",
                        }
                    ],
                }
            )
        )

        rendered = format_clue_package(package)
        self.assertIn("Best: Mascot for a speech that needs the wrap-it-up music", rendered)
        self.assertIn("[best]", rendered)

    def test_main_reports_missing_api_key_to_stderr(self) -> None:
        stdout = StringIO()
        stderr = StringIO()

        with patch(
            "byewords.groq_clues.load_default_answer_inputs",
            return_value=(("asked",), {"asked": ('Past tense of "ask"',)}),
        ):
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
            patch("byewords.groq_clues.GroqClient", return_value=FakeGroqClient()),
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
