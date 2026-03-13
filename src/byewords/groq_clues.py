from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Protocol, TextIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import UUID

from byewords.clue_bank import is_generic_clue
from byewords.lexicon import load_clue_bank, load_word_list, normalize_word
from byewords.puzzle_store import puzzle_answers_for_id

MODEL_NAME = "openai/gpt-oss-120b"
API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_CLUE_COUNT = 2
DEFAULT_PARALLELISM = 5
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
QUALITY_EXAMPLES = (
    "Answer: TRADE\n"
    "Clue 1: Market-moving exchange at the center of tariff talk\n"
    "Clue 2: Business between nations that can spark a customs fight\n"
    "\n"
    "Answer: EERIE\n"
    "Clue 1: Like a nursery rhyme heard through static\n"
    "Clue 2: Unsettling in the way an empty playground feels at dusk\n"
    "\n"
    "Answer: CABLE\n"
    "Clue 1: Coiled connector hiding in a junk drawer\n"
    "Clue 2: Wire that turns low battery panic into relief"
)
CLUE_RULES = (
    "Follow these rules for every clue:\n"
    "- Your clue MUST be the same part of speech as the answer.\n"
    "- You MUST NOT use words from the answer or inflected forms of those words in the clue.\n"
    "- Avoid reusing prefixes and suffixes from the answer in the clue.\n"
    "- Avoid reusing words from the puzzle in the clue.\n"
    "- You MUST NOT define the answer by example.\n"
    "- You MUST NOT editorialize.\n"
    "- You MUST NOT create single-word clues.\n"
    "- You MUST use a mix of diverse clue styles.\n"
    "- You MUST capitalize the first word of every clue."
)


@dataclass(frozen=True)
class CluePackage:
    answer: str
    cached: bool
    clues: tuple[str, ...]


@dataclass(frozen=True)
class AnswerSelection:
    queued_answers: tuple[str, ...]
    skipped_answers: tuple[str, ...]


class GracefulExit(Exception):
    def __init__(self, completed: int, total: int) -> None:
        super().__init__(f"Stopped after {completed} out of {total} words completed.")
        self.completed = completed
        self.total = total


class StatusReporter:
    def __init__(self, stream: TextIO, total: int) -> None:
        self.stream = stream
        self.total = total
        self.completed = 0
        self._lock = threading.Lock()
        self._stop_noted = False

    def completed_word(self) -> None:
        with self._lock:
            self.completed += 1
            should_log = self.completed % 50 == 0 or (
                self.total >= 50 and self.completed == self.total
            )
            message = f"{self.completed} out of {self.total} words completed" if should_log else None
        if message is not None:
            print(message, file=self.stream)

    def rate_limited(self, retry_after_seconds: float) -> None:
        print(
            f"Rate limited by Groq. Waiting {retry_after_seconds:g} seconds before retrying.",
            file=self.stream,
        )

    def stop_requested(self) -> None:
        with self._lock:
            if self._stop_noted:
                return
            self._stop_noted = True
        print("Graceful stop requested. Finishing in-flight requests before exit.", file=self.stream)


class CompletionClient(Protocol):
    def create_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]: ...


class UnavailableClient:
    def create_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("Groq client was not configured for uncached clue generation.")


class GroqClient:
    def __init__(
        self,
        api_key: str,
        api_url: str = API_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        rate_limit_callback: Callable[[float], None] | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.api_key = api_key
        self.api_url = api_url
        self.timeout_seconds = timeout_seconds
        self.rate_limit_callback = rate_limit_callback
        self.sleep_fn = sleep_fn

    def create_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_body = json.dumps(payload).encode("utf-8")
        request = Request(
            self.api_url,
            data=request_body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": DEFAULT_USER_AGENT,
            },
            method="POST",
        )

        while True:
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    response_body = response.read().decode("utf-8")
            except HTTPError as exc:
                retry_after_seconds = _retry_after_seconds(exc)
                if exc.code == 429 and retry_after_seconds is not None:
                    if self.rate_limit_callback is not None:
                        self.rate_limit_callback(retry_after_seconds)
                    self.sleep_fn(retry_after_seconds)
                    continue
                raise RuntimeError(_format_http_error(exc)) from exc
            except URLError as exc:
                raise RuntimeError(f"Could not reach the Groq API: {exc.reason}") from exc
            break

        parsed = json.loads(response_body)
        if not isinstance(parsed, dict):
            raise RuntimeError("Groq API returned an unexpected response shape.")
        return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="byewords-generate-clues",
        description="Generate sharp crossword clues with Groq's GPT OSS 120B model.",
    )
    parser.add_argument(
        "targets",
        nargs="*",
        help="Optional puzzle UUID followed by bundled five-letter answers. If answers are omitted, the script scans the full bundled word list.",
    )
    parser.add_argument(
        "--theme",
        default="",
        help="Ignored. Clues are now standalone rather than theme-driven.",
    )
    parser.add_argument(
        "--recent-context",
        action="append",
        dest="recent_context",
        default=[],
        help="Ignored. Clues are now timeless rather than tied to recent context.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_CLUE_COUNT,
        help=f"How many clue options to request per answer. Must be {DEFAULT_CLUE_COUNT}.",
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
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate clues through the API even when non-generic clue-bank entries already exist.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.count != DEFAULT_CLUE_COUNT:
        parser.error(f"--count must be exactly {DEFAULT_CLUE_COUNT}")
    if args.parallelism < 1:
        parser.error("--parallelism must be at least 1")
    if args.limit < 0:
        parser.error("--limit must be 0 or greater")
    args.puzzle_uuid, args.answers = _split_targets(args.targets)
    del args.targets
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


def build_clue_payload(
    answer: str,
    count: int,
) -> dict[str, Any]:
    return {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an one of the best crossword editors of all time on par with Will Shortz, "
                    "Anna Shechtman, Brendan Emmett Quigley and Patrick Berry. "
                    "Your work has been published in The New York Times, The Wall Street Journal, and The New Yorker. "
                    "Write two funny, memorable and precise clues for the provided answer. "
                    "Review your work by evaluating whether each clue is a clear reference to the provided answer. "
                    "The clue should present some challenge to the player but not be an obsure reference. "
                    "Return only a JSON object with one key, 'clues', whose value is an array of two distinct clue strings.\n\n"
                    f"{CLUE_RULES}\n\n"
                    "Study these contrasting examples of high-quality clues and match their standard of specificity and snap:\n"
                    f"{QUALITY_EXAMPLES}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Answer: {answer.strip()}\n"
                    f"Return exactly {count} standalone clues. "
                    "Return only a JSON object with one key, 'clues'."
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
                        "clues": {
                            "type": "array",
                            "minItems": DEFAULT_CLUE_COUNT,
                            "maxItems": DEFAULT_CLUE_COUNT,
                            "items": {"type": "string"},
                        }
                    },
                    "required": ["clues"],
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


def parse_clue_package(payload: str) -> tuple[str, ...]:
    raw = json.loads(payload)
    if isinstance(raw, Mapping):
        raw = raw.get("clues")
    if not isinstance(raw, list) or len(raw) != DEFAULT_CLUE_COUNT:
        raise RuntimeError(
            f"Structured clue response did not include exactly {DEFAULT_CLUE_COUNT} clue options."
        )
    clues = []
    for raw_clue in raw:
        if not isinstance(raw_clue, str):
            raise RuntimeError("Structured clue response included an invalid clue option.")
        clue = raw_clue.strip()
        if not clue:
            raise RuntimeError("Structured clue response included an empty clue option.")
        clues.append(clue)
    return tuple(clues)


def generate_clue_package(
    client: CompletionClient,
    answer: str,
    clue_bank: dict[str, tuple[str, ...]],
    clue_bank_path: str,
    lock: threading.Lock,
    count: int = DEFAULT_CLUE_COUNT,
    force: bool = False,
) -> CluePackage:
    cached_clues = cached_clues_for_answer(answer, clue_bank)
    if cached_clues and not force:
        return CluePackage(answer=answer.strip(), cached=True, clues=cached_clues)

    clue_payload = build_clue_payload(
        answer=answer,
        count=count,
    )
    clue_response = client.create_chat_completion(clue_payload)
    clue_content = extract_message_content(clue_response)
    clues = parse_clue_package(clue_content)
    with lock:
        clue_bank[answer.strip()] = clues
        persist_clue_bank(clue_bank_path, clue_bank)
    return CluePackage(answer=answer.strip(), cached=False, clues=clues)


def load_default_answer_inputs() -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    words_path = str(resources.files("byewords").joinpath("data", "words_5.txt"))
    clue_bank_path = str(resources.files("byewords").joinpath("data", "clue_bank.json"))
    return load_word_list(words_path), load_clue_bank(clue_bank_path)


def default_clue_bank_path() -> str:
    return str(resources.files("byewords").joinpath("data", "clue_bank.json"))


def answer_needs_new_clue(answer: str, clue_bank: Mapping[str, tuple[str, ...]]) -> bool:
    return cached_clues_for_answer(answer, clue_bank) is None


def cached_clues_for_answer(
    answer: str,
    clue_bank: Mapping[str, tuple[str, ...]],
) -> tuple[str, ...] | None:
    clues = tuple(
        clue.strip()
        for clue in clue_bank.get(answer, ())
        if clue.strip() and not is_generic_clue(clue)
    )
    if not clues:
        return None
    return clues


def persist_clue_bank(path: str, clue_bank: Mapping[str, tuple[str, ...]]) -> None:
    serializable = {
        answer: list(clues)
        for answer, clues in sorted(clue_bank.items())
    }
    Path(path).write_text(json.dumps(serializable, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def select_answers_to_clue(
    requested_answers: Sequence[str],
    lexicon_words: tuple[str, ...],
    clue_bank: Mapping[str, tuple[str, ...]],
    limit: int = 0,
    force: bool = False,
) -> AnswerSelection:
    if requested_answers:
        source_answers = _normalize_requested_answers(requested_answers)
        if limit > 0:
            source_answers = source_answers[:limit]
        return AnswerSelection(queued_answers=source_answers, skipped_answers=())
    if force:
        source_answers = lexicon_words[:limit] if limit > 0 else lexicon_words
        return AnswerSelection(queued_answers=source_answers, skipped_answers=())

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
    clue_bank: dict[str, tuple[str, ...]],
    clue_bank_path: str,
    count: int = DEFAULT_CLUE_COUNT,
    parallelism: int = DEFAULT_PARALLELISM,
    reporter: StatusReporter | None = None,
    stop_event: threading.Event | None = None,
    force: bool = False,
) -> tuple[CluePackage, ...]:
    if not answers:
        return ()

    results: list[CluePackage | None] = [None for _ in answers]
    failures: list[str] = []
    worker_count = min(parallelism, len(answers))
    lock = threading.Lock()
    pending_jobs = list(enumerate(answers))
    next_job_index = 0
    completed = 0

    def submit_next_jobs(
        executor: ThreadPoolExecutor,
        future_to_job: dict[Future[CluePackage], tuple[int, str]],
    ) -> None:
        nonlocal next_job_index
        while next_job_index < len(pending_jobs) and len(future_to_job) < worker_count:
            if stop_event is not None and stop_event.is_set():
                return
            index, answer = pending_jobs[next_job_index]
            next_job_index += 1
            future = executor.submit(
                generate_clue_package,
                client,
                answer,
                clue_bank,
                clue_bank_path,
                lock,
                count,
                force,
            )
            future_to_job[future] = (index, answer)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_job: dict[Future[CluePackage], tuple[int, str]] = {}
        submit_next_jobs(executor, future_to_job)
        while future_to_job:
            future = next(as_completed(future_to_job))
            index, answer = future_to_job.pop(future)
            try:
                results[index] = future.result()
            except RuntimeError as exc:
                failures.append(f"{answer}: {exc}")
            completed += 1
            if reporter is not None:
                reporter.completed_word()
            submit_next_jobs(executor, future_to_job)

    if failures:
        preview = "\n".join(failures[:10])
        remainder = len(failures) - min(len(failures), 10)
        suffix = "" if remainder <= 0 else f"\n... and {remainder} more failures"
        raise RuntimeError(f"Failed to generate clues for {len(failures)} answers:\n{preview}{suffix}")

    if stop_event is not None and stop_event.is_set() and next_job_index < len(pending_jobs):
        raise GracefulExit(completed=completed, total=len(answers))

    return tuple(result for result in results if result is not None)


def clue_package_to_dict(package: CluePackage) -> dict[str, Any]:
    return asdict(package)


def format_clue_package(package: CluePackage) -> str:
    lines = [package.answer.upper()]
    lines.append(f"Cached: {'yes' if package.cached else 'no'}")
    for index, clue in enumerate(package.clues, start=1):
        lines.append(f"{index}. {clue}")
    return "\n".join(lines)


def regenerate_clues(
    answers: Sequence[str],
    clue_bank: dict[str, tuple[str, ...]],
    clue_bank_path: str,
    *,
    env: Mapping[str, str] | None = None,
    errors: TextIO | None = None,
    force: bool = False,
    count: int = DEFAULT_CLUE_COUNT,
    parallelism: int = DEFAULT_PARALLELISM,
) -> tuple[CluePackage, ...]:
    if not answers:
        return ()

    error_stream = errors if errors is not None else sys.stderr
    reporter = StatusReporter(error_stream, total=len(answers))
    stop_event = threading.Event()
    needs_generation = force or any(answer_needs_new_clue(answer, clue_bank) for answer in answers)

    previous_sigint = None
    previous_sigterm = None
    if needs_generation:
        api_key = require_api_key(env)

        def _request_stop(signum: int, frame: object) -> None:
            del signum, frame
            reporter.stop_requested()
            stop_event.set()

        previous_sigint = signal.getsignal(signal.SIGINT)
        previous_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, _request_stop)
        signal.signal(signal.SIGTERM, _request_stop)
        client: CompletionClient = GroqClient(api_key, rate_limit_callback=reporter.rate_limited)
    else:
        client = UnavailableClient()

    try:
        return generate_clue_packages_parallel(
            client=client,
            answers=answers,
            clue_bank=clue_bank,
            clue_bank_path=clue_bank_path,
            count=count,
            parallelism=parallelism,
            reporter=reporter,
            stop_event=stop_event,
            force=force,
        )
    finally:
        if needs_generation:
            signal.signal(signal.SIGINT, previous_sigint)
            signal.signal(signal.SIGTERM, previous_sigterm)


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
        requested_answers = args.answers
        if args.puzzle_uuid is not None and not requested_answers:
            requested_answers = puzzle_answers_for_id(args.puzzle_uuid)
        selection = select_answers_to_clue(
            requested_answers,
            lexicon_words,
            clue_bank,
            limit=args.limit,
            force=args.force,
        )
        clue_bank_path = default_clue_bank_path()
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

    if selection.skipped_answers:
        print(
            f"Skipping {len(selection.skipped_answers)} answers that already have non-generic clues.",
            file=errors,
        )

    try:
        packages = regenerate_clues(
            answers=selection.queued_answers,
            clue_bank=clue_bank,
            clue_bank_path=clue_bank_path,
            env=env,
            errors=errors,
            force=args.force,
            count=args.count,
            parallelism=args.parallelism,
        )
    except GracefulExit as exc:
        print(
            f"Graceful stop complete: {exc.completed} out of {exc.total} words completed.",
            file=errors,
        )
        return 130
    except RuntimeError as exc:
        print(f"error: {exc}", file=errors)
        return 1
    except ValueError as exc:
        print(f"error: {exc}", file=errors)
        return 1

    if args.json:
        payload: Any
        package_dicts = [clue_package_to_dict(package) for package in packages]
        if args.puzzle_uuid is None:
            payload = package_dicts
        else:
            payload = {
                "puzzle_uuid": args.puzzle_uuid,
                "packages": package_dicts,
            }
        json.dump(payload, output, indent=2)
        output.write("\n")
        return 0

    if args.puzzle_uuid is not None:
        output.write(f"Puzzle UUID: {args.puzzle_uuid}\n\n")
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


def _retry_after_seconds(error: HTTPError) -> float | None:
    retry_after = error.headers.get("Retry-After")
    if retry_after is None:
        return None
    try:
        seconds = float(retry_after.strip())
    except ValueError:
        return None
    if seconds < 0:
        return None
    return seconds


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


def _split_targets(targets: Sequence[str]) -> tuple[str | None, tuple[str, ...]]:
    if not targets:
        return None, ()
    first_target = targets[0]
    try:
        parsed_uuid = UUID(first_target)
    except ValueError:
        return None, tuple(targets)
    return str(parsed_uuid), tuple(targets[1:])


if __name__ == "__main__":
    raise SystemExit(main())
