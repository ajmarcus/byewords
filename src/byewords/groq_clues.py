from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from importlib import resources
from typing import Any, Protocol, TextIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from byewords.clue_bank import is_generic_clue
from byewords.lexicon import load_clue_bank, load_word_list, normalize_word

MODEL_NAME = "openai/gpt-oss-120b"
API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_CLUE_COUNT = 5
DEFAULT_PARALLELISM = 8
DEFAULT_TIMEOUT_SECONDS = 60.0
QUALITY_EXAMPLES = (
    "Example 1\n"
    "Answer: TRADE\n"
    "Clue: What a tariff threat can rattle overnight\n"
    "Why it works: topical, concise, and precise without requiring obscure trivia.\n"
    "\n"
    "Example 2\n"
    "Answer: EERIE\n"
    "Clue: Like a hallway after the lights blink out\n"
    "Why it works: vivid sensory image, memorable surface, exact definition.\n"
    "\n"
    "Example 3\n"
    "Answer: CABLE\n"
    "Clue: San Francisco car pulled by an underground loop\n"
    "Why it works: concrete, specific, and rich with place without sounding dusty."
)


@dataclass(frozen=True)
class ClueOption:
    clue: str
    angle: str
    freshness: str
    why_it_works: str


@dataclass(frozen=True)
class CluePackage:
    answer: str
    research_verdict: str
    recent_hook_summary: str
    editor_note: str
    best_index: int
    clues: tuple[ClueOption, ...]


@dataclass(frozen=True)
class AnswerSelection:
    queued_answers: tuple[str, ...]
    skipped_answers: tuple[str, ...]


class CompletionClient(Protocol):
    def create_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]: ...


class GroqClient:
    def __init__(
        self,
        api_key: str,
        api_url: str = API_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.api_key = api_key
        self.api_url = api_url
        self.timeout_seconds = timeout_seconds

    def create_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_body = json.dumps(payload).encode("utf-8")
        request = Request(
            self.api_url,
            data=request_body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                response_body = response.read().decode("utf-8")
        except HTTPError as exc:
            raise RuntimeError(_format_http_error(exc)) from exc
        except URLError as exc:
            raise RuntimeError(f"Could not reach the Groq API: {exc.reason}") from exc

        parsed = json.loads(response_body)
        if not isinstance(parsed, dict):
            raise RuntimeError("Groq API returned an unexpected response shape.")
        return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="byewords-groq-clues",
        description="Generate sharp crossword clues with Groq's GPT OSS 120B model.",
    )
    parser.add_argument(
        "answers",
        nargs="*",
        help="Optional bundled five-letter answers to clue. If omitted, the script scans the full bundled word list.",
    )
    parser.add_argument(
        "--theme",
        default="",
        help="Optional puzzle theme or editorial framing.",
    )
    parser.add_argument(
        "--recent-context",
        action="append",
        dest="recent_context",
        default=[],
        help="Extra recent-events context to bias the clue search. May be passed multiple times.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_CLUE_COUNT,
        help=f"How many clue options to request per answer. Default: {DEFAULT_CLUE_COUNT}.",
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=DEFAULT_PARALLELISM,
        help=f"How many answers to process concurrently. Default: {DEFAULT_PARALLELISM}.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on how many unclued answers to process. Default: 0 (no cap).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print structured JSON instead of formatted text.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.count < 1:
        parser.error("--count must be at least 1")
    if args.parallelism < 1:
        parser.error("--parallelism must be at least 1")
    if args.limit < 0:
        parser.error("--limit must be 0 or greater")
    return args


def require_api_key(env: Mapping[str, str] | None = None) -> str:
    source = env if env is not None else os.environ
    api_key = source.get("GROQ_API_KEY", "").strip()
    if api_key:
        return api_key
    raise ValueError(
        "GROQ_API_KEY is not set. Export your Groq API key and rerun, for example: "
        'export GROQ_API_KEY="gsk_..."'
    )


def today_utc() -> date:
    return datetime.now(UTC).date()


def build_research_payload(
    answer: str,
    theme: str,
    recent_context: Sequence[str],
    current_date: date,
) -> dict[str, Any]:
    theme_text = theme or "None"
    context_lines = "\n".join(f"- {item}" for item in recent_context) or "- None supplied"
    date_text = _format_date(current_date)
    return {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are researching clue angles for a first-rate New York Times crossword editor. "
                    "Use browser search to find timely hooks tied to the answer. Focus on the last 45 days when possible. "
                    "Do not write clues. Return plain text with exactly these sections:\n"
                    "VERDICT: RECENT or HYBRID or TIMELESS\n"
                    "HOOKS:\n"
                    "- bullet points with concrete current hooks or a direct statement that no fair timely hook exists\n"
                    "RISKS:\n"
                    "- bullet points describing fairness traps, stale angles, or ambiguity risks\n"
                    "Keep the whole response under 180 words."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Answer: {answer.strip()}\n"
                    f"Theme: {theme_text}\n"
                    f"Today's date: {date_text}\n"
                    "Extra recent context to consider:\n"
                    f"{context_lines}\n"
                    "Find timely but fair clue angles. If the answer has no clean recent-news hook, say so plainly."
                ),
            },
        ],
        "tool_choice": "required",
        "tools": [{"type": "browser_search"}],
        "max_completion_tokens": 700,
    }


def build_clue_payload(
    answer: str,
    theme: str,
    recent_context: Sequence[str],
    research_notes: str,
    count: int,
    current_date: date,
) -> dict[str, Any]:
    theme_text = theme or "None"
    context_lines = "\n".join(f"- {item}" for item in recent_context) or "- None supplied"
    date_text = _format_date(current_date)
    return {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a first-rate New York Times crossword editor. "
                    "Write clues that are spicy, precise, memorable, and fair. "
                    "Every clue must point cleanly to the exact answer. "
                    "Use a current-events angle only when the research notes support it clearly. "
                    "Never invent or exaggerate a recent event. "
                    "Avoid dull dictionary definitions, fill-in-the-blank prompts, partials, giveaway spelling cues, and niche fan-wiki trivia. "
                    "Return distinct options with at least one safer clue and one bolder clue.\n\n"
                    "Study these contrasting examples of high-quality clues and match their standard of specificity and snap:\n"
                    f"{QUALITY_EXAMPLES}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Answer: {answer.strip()}\n"
                    f"Theme: {theme_text}\n"
                    f"Today's date: {date_text}\n"
                    "Extra recent context:\n"
                    f"{context_lines}\n"
                    "Research notes:\n"
                    f"{research_notes}\n"
                    f"Return exactly {count} clue options. "
                    "If no fair timely clue exists, say that in the summaries and produce the strongest timeless or hybrid clue instead."
                ),
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "crossword_clue_package",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"},
                        "research_verdict": {
                            "type": "string",
                            "enum": ["RECENT", "HYBRID", "TIMELESS"],
                        },
                        "recent_hook_summary": {"type": "string"},
                        "editor_note": {"type": "string"},
                        "best_index": {"type": "integer"},
                        "clues": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "clue": {"type": "string"},
                                    "angle": {"type": "string"},
                                    "freshness": {
                                        "type": "string",
                                        "enum": ["recent_event", "hybrid", "timeless"],
                                    },
                                    "why_it_works": {"type": "string"},
                                },
                                "required": ["clue", "angle", "freshness", "why_it_works"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": [
                        "answer",
                        "research_verdict",
                        "recent_hook_summary",
                        "editor_note",
                        "best_index",
                        "clues",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "max_completion_tokens": 1200,
    }


def extract_message_content(response: Mapping[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("Groq API returned no choices.")
    first_choice = choices[0]
    if not isinstance(first_choice, Mapping):
        raise RuntimeError("Groq API returned an invalid choice payload.")
    message = first_choice.get("message")
    if not isinstance(message, Mapping):
        raise RuntimeError("Groq API response is missing the completion message.")
    refusal = message.get("refusal")
    if isinstance(refusal, str) and refusal.strip():
        raise RuntimeError(f"Groq refused the request: {refusal.strip()}")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Groq API returned an empty completion.")
    return content


def parse_clue_package(payload: str) -> CluePackage:
    raw = json.loads(payload)
    if not isinstance(raw, dict):
        raise RuntimeError("Structured clue response was not a JSON object.")
    raw_clues = raw.get("clues")
    if not isinstance(raw_clues, list) or not raw_clues:
        raise RuntimeError("Structured clue response did not include any clue options.")
    clues = []
    for raw_clue in raw_clues:
        if not isinstance(raw_clue, dict):
            raise RuntimeError("Structured clue response included an invalid clue option.")
        clues.append(
            ClueOption(
                clue=str(raw_clue["clue"]),
                angle=str(raw_clue["angle"]),
                freshness=str(raw_clue["freshness"]),
                why_it_works=str(raw_clue["why_it_works"]),
            )
        )

    best_index = raw.get("best_index", 0)
    if not isinstance(best_index, int) or not 0 <= best_index < len(clues):
        best_index = 0

    return CluePackage(
        answer=str(raw.get("answer", "")),
        research_verdict=str(raw.get("research_verdict", "TIMELESS")),
        recent_hook_summary=str(raw.get("recent_hook_summary", "")),
        editor_note=str(raw.get("editor_note", "")),
        best_index=best_index,
        clues=tuple(clues),
    )


def generate_clue_package(
    client: CompletionClient,
    answer: str,
    theme: str = "",
    recent_context: Sequence[str] = (),
    count: int = DEFAULT_CLUE_COUNT,
    current_date: date | None = None,
) -> CluePackage:
    effective_date = current_date or today_utc()
    research_payload = build_research_payload(answer, theme, recent_context, effective_date)
    research_response = client.create_chat_completion(research_payload)
    research_notes = extract_message_content(research_response)

    clue_payload = build_clue_payload(
        answer=answer,
        theme=theme,
        recent_context=recent_context,
        research_notes=research_notes,
        count=count,
        current_date=effective_date,
    )
    clue_response = client.create_chat_completion(clue_payload)
    clue_content = extract_message_content(clue_response)
    package = parse_clue_package(clue_content)

    if not package.answer.strip():
        return CluePackage(
            answer=answer.strip(),
            research_verdict=package.research_verdict,
            recent_hook_summary=package.recent_hook_summary,
            editor_note=package.editor_note,
            best_index=package.best_index,
            clues=package.clues,
    )
    return package


def load_default_answer_inputs() -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    words_path = str(resources.files("byewords").joinpath("data", "words_5.txt"))
    clue_bank_path = str(resources.files("byewords").joinpath("data", "clue_bank.json"))
    return load_word_list(words_path), load_clue_bank(clue_bank_path)


def answer_needs_new_clue(answer: str, clue_bank: Mapping[str, tuple[str, ...]]) -> bool:
    clues = clue_bank.get(answer, ())
    if not clues:
        return True
    return not any(clue.strip() and not is_generic_clue(clue) for clue in clues)


def select_answers_to_clue(
    requested_answers: Sequence[str],
    lexicon_words: tuple[str, ...],
    clue_bank: Mapping[str, tuple[str, ...]],
    limit: int = 0,
) -> AnswerSelection:
    if requested_answers:
        source_answers = _normalize_requested_answers(requested_answers)
    else:
        source_answers = lexicon_words

    queued = []
    skipped = []
    for answer in source_answers:
        if answer_needs_new_clue(answer, clue_bank):
            queued.append(answer)
        else:
            skipped.append(answer)

    if limit > 0:
        queued = queued[:limit]
    return AnswerSelection(queued_answers=tuple(queued), skipped_answers=tuple(skipped))


def generate_clue_packages_parallel(
    client: CompletionClient,
    answers: Sequence[str],
    theme: str = "",
    recent_context: Sequence[str] = (),
    count: int = DEFAULT_CLUE_COUNT,
    parallelism: int = DEFAULT_PARALLELISM,
    current_date: date | None = None,
) -> tuple[CluePackage, ...]:
    if not answers:
        return ()

    effective_date = current_date or today_utc()
    results: list[CluePackage | None] = [None for _ in answers]
    failures: list[str] = []
    worker_count = min(parallelism, len(answers))

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_job = {
            executor.submit(
                generate_clue_package,
                client,
                answer,
                theme,
                recent_context,
                count,
                effective_date,
            ): (index, answer)
            for index, answer in enumerate(answers)
        }
        for future in as_completed(future_to_job):
            index, answer = future_to_job[future]
            try:
                results[index] = future.result()
            except RuntimeError as exc:
                failures.append(f"{answer}: {exc}")

    if failures:
        preview = "\n".join(failures[:10])
        remainder = len(failures) - min(len(failures), 10)
        suffix = "" if remainder <= 0 else f"\n... and {remainder} more failures"
        raise RuntimeError(f"Failed to generate clues for {len(failures)} answers:\n{preview}{suffix}")

    return tuple(result for result in results if result is not None)


def clue_package_to_dict(package: CluePackage) -> dict[str, Any]:
    data = asdict(package)
    data["best_clue"] = package.clues[package.best_index].clue
    return data


def format_clue_package(package: CluePackage) -> str:
    lines = [
        package.answer.upper(),
        f"Best: {package.clues[package.best_index].clue}",
        f"Freshness: {package.research_verdict}",
        f"Hook: {package.recent_hook_summary}",
    ]
    if package.editor_note:
        lines.append(f"Note: {package.editor_note}")
    lines.append("Options:")
    for index, clue in enumerate(package.clues, start=1):
        best_marker = " [best]" if index - 1 == package.best_index else ""
        lines.append(
            f"{index}. {clue.clue}{best_marker} ({clue.freshness}; {clue.angle})"
        )
    return "\n".join(lines)


def main(
    argv: Sequence[str] | None = None,
    env: Mapping[str, str] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    output = stdout if stdout is not None else sys.stdout
    errors = stderr if stderr is not None else sys.stderr
    args = parse_args(argv)

    try:
        lexicon_words, clue_bank = load_default_answer_inputs()
        selection = select_answers_to_clue(args.answers, lexicon_words, clue_bank, limit=args.limit)
    except ValueError as exc:
        print(f"error: {exc}", file=errors)
        return 1

    if not selection.queued_answers:
        if selection.skipped_answers:
            print(
                f"Skipping {len(selection.skipped_answers)} answers that already have non-generic clues.",
                file=errors,
            )
        if args.json:
            output.write("[]\n")
        else:
            output.write("No answers need new non-generic clues.\n")
        return 0

    try:
        api_key = require_api_key(env)
    except ValueError as exc:
        print(f"error: {exc}", file=errors)
        return 1

    if selection.skipped_answers:
        print(
            f"Skipping {len(selection.skipped_answers)} answers that already have non-generic clues.",
            file=errors,
        )

    client = GroqClient(api_key)
    try:
        packages = generate_clue_packages_parallel(
            client=client,
            answers=selection.queued_answers,
            theme=args.theme,
            recent_context=tuple(args.recent_context),
            count=args.count,
            parallelism=args.parallelism,
        )
    except RuntimeError as exc:
        print(f"error: {exc}", file=errors)
        return 1

    if args.json:
        json.dump([clue_package_to_dict(package) for package in packages], output, indent=2)
        output.write("\n")
        return 0

    output.write("\n\n".join(format_clue_package(package) for package in packages))
    output.write("\n")
    return 0


def _format_http_error(error: HTTPError) -> str:
    body = error.read().decode("utf-8", errors="replace")
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        raw_error = parsed.get("error")
        if isinstance(raw_error, dict):
            message = raw_error.get("message")
            if isinstance(message, str) and message.strip():
                return message.strip()
        message = parsed.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()

    if body.strip():
        return f"Groq API request failed with HTTP {error.code}: {body.strip()}"
    return f"Groq API request failed with HTTP {error.code}."


def _format_date(value: date) -> str:
    return f"{value:%B} {value.day}, {value:%Y}"


def _normalize_requested_answers(requested_answers: Sequence[str]) -> tuple[str, ...]:
    normalized = []
    invalid = []
    for answer in requested_answers:
        normalized_answer = normalize_word(answer)
        if normalized_answer is None:
            invalid.append(answer)
        else:
            normalized.append(normalized_answer)
    if invalid:
        bad_answers = ", ".join(invalid)
        raise ValueError(
            "All requested answers must be bundled five-letter entries. "
            f"Invalid input: {bad_answers}"
        )
    return tuple(dict.fromkeys(normalized))


if __name__ == "__main__":
    raise SystemExit(main())
