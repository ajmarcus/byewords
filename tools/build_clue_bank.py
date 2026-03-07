from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORDS_PATH = ROOT / "src/byewords/data/words_5.txt"
OVERRIDES_PATH = ROOT / "src/byewords/data/clue_overrides.json"
OUTPUT_PATH = ROOT / "src/byewords/data/clue_bank.json"
HUNSPELL_PATH = Path("/usr/share/hunspell/en_US.dic")


def load_words() -> list[str]:
    return [word.strip().lower() for word in WORDS_PATH.read_text(encoding="utf-8").splitlines() if word.strip()]


def load_overrides() -> dict[str, str]:
    raw = json.loads(OVERRIDES_PATH.read_text(encoding="utf-8"))
    return {
        answer.lower(): clue.strip()
        for answer, clue in raw.items()
        if isinstance(answer, str) and isinstance(clue, str) and clue.strip()
    }


def load_hunspell_flags() -> dict[str, frozenset[str]]:
    entries: dict[str, frozenset[str]] = {}
    for line in HUNSPELL_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()[1:]:
        raw_word, _, raw_flags = line.partition("/")
        word = raw_word.lower()
        if word.isalpha():
            entries[word] = frozenset(raw_flags)
    return entries


def _past_tense_clue(word: str, entries: dict[str, frozenset[str]]) -> str | None:
    if word.endswith("ied"):
        base = word[:-3] + "y"
        if "D" in entries.get(base, ()):
            return f'Past tense of "{base}"'
    if word.endswith("ed"):
        for base in (word[:-1], word[:-2]):
            if "D" in entries.get(base, ()):
                return f'Past tense of "{base}"'
    return None


def _plural_clue(word: str, entries: dict[str, frozenset[str]]) -> str | None:
    singular = word[:-1]
    if word.endswith("s") and "Z" in entries.get(singular, ()):
        return f'Plural of "{singular}"'
    if word.endswith("ies"):
        base = word[:-3] + "y"
        if "S" in entries.get(base, ()):
            return f'Plural of "{base}"'
    if word.endswith("es"):
        for base in (word[:-1], word[:-2]):
            if "S" in entries.get(base, ()):
                return f'Plural of "{base}"'
    if word.endswith("s") and "S" in entries.get(singular, ()):
        return f'Plural of "{singular}"'
    return None


def _agent_noun_clue(word: str, entries: dict[str, frozenset[str]]) -> str | None:
    if not word.endswith("er"):
        return None
    for base in (word[:-1], word[:-2]):
        if "Z" in entries.get(base, ()):
            return f'One who "{base}s"'
    return None


def _generated_clue(
    word: str,
    entries: dict[str, frozenset[str]],
    words: list[str],
    index_by_word: dict[str, int],
) -> str:
    for builder in (_past_tense_clue, _plural_clue, _agent_noun_clue):
        clue = builder(word, entries)
        if clue is not None:
            return clue

    position = index_by_word[word]
    if position == 0:
        return "First entry in the bundled lexicon"
    if position == len(words) - 1:
        return "Last entry in the bundled lexicon"
    previous_word = words[position - 1]
    next_word = words[position + 1]
    return f'Bundled-lexicon entry between "{previous_word}" and "{next_word}"'


def build_clue_bank() -> dict[str, list[str]]:
    words = load_words()
    overrides = load_overrides()
    entries = load_hunspell_flags()
    index_by_word = {word: index for index, word in enumerate(words)}
    return {
        word: [overrides.get(word) or _generated_clue(word, entries, words, index_by_word)]
        for word in words
    }


def main() -> int:
    clue_bank = build_clue_bank()
    OUTPUT_PATH.write_text(json.dumps(clue_bank, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
