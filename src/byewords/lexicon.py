import json
from pathlib import Path


def load_word_list(path: str) -> tuple[str, ...]:
    contents = Path(path).read_text(encoding="utf-8").splitlines()
    return filter_legal_words(tuple(contents))


def normalize_word(word: str) -> str | None:
    normalized = word.strip().lower()
    if len(normalized) != 5 or not normalized.isascii() or not normalized.isalpha():
        return None
    return normalized


def filter_legal_words(words: tuple[str, ...]) -> tuple[str, ...]:
    legal_words = []
    for word in words:
        normalized = normalize_word(word)
        if normalized is not None:
            legal_words.append(normalized)
    return tuple(dict.fromkeys(legal_words))


def load_related_words(path: str) -> dict[str, tuple[str, ...]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        key.lower(): filter_legal_words(tuple(value))
        for key, value in raw.items()
    }


def load_clue_bank(path: str) -> dict[str, tuple[str, ...]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        key.lower(): tuple(pattern for pattern in value if isinstance(pattern, str))
        for key, value in raw.items()
    }
