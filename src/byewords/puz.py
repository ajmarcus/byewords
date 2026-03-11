from __future__ import annotations

from dataclasses import dataclass

from byewords.grid import grid_columns
from byewords.types import Puzzle

_HEADER_MAGIC = b"ACROSS&DOWN\x00"
_MASKED_CHECKSUM_MAGIC = b"ICHEATED"
_VERSION = b"1.3\x00"
_PUZZLE_TYPE_NORMAL = 0x0001
_SOLUTION_STATE_UNLOCKED = 0x0000
_EMPTY_CELL = b"-"


@dataclass(frozen=True)
class _PuzPayload:
    solution: bytes
    fill: bytes
    title: bytes
    author: bytes
    copyright: bytes
    clues: tuple[bytes, ...]
    notes: bytes


def _normalize_puz_text(text: str) -> str:
    replacements = {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
        "\u202f": " ",
        "\u00a0": " ",
    }
    normalized = "".join(replacements.get(character, character) for character in text)
    return normalized.encode("latin-1", errors="replace").decode("latin-1")


def _encode_puz_text(text: str) -> bytes:
    return _normalize_puz_text(text).encode("latin-1")


def _build_payload(puzzle: Puzzle) -> _PuzPayload:
    solution = "".join(row.upper() for row in puzzle.grid.rows).encode("ascii")
    fill = _EMPTY_CELL * len(solution)
    ordered_clues = puzzle.across + puzzle.down
    return _PuzPayload(
        solution=solution,
        fill=fill,
        title=_encode_puz_text(puzzle.title),
        author=b"",
        copyright=b"",
        clues=tuple(_encode_puz_text(clue.text) for clue in ordered_clues),
        notes=b"",
    )


def _checksum(data: bytes, seed: int = 0) -> int:
    checksum = seed
    for byte in data:
        checksum = (checksum >> 1) | ((checksum & 1) << 15)
        checksum = (checksum + byte) & 0xFFFF
    return checksum


def _with_terminator(field: bytes) -> bytes:
    return field + b"\x00"


def _header_checksum_bytes(
    cib_checksum: int,
    solution_checksum: int,
    fill_checksum: int,
    strings_checksum: int,
) -> tuple[bytes, bytes]:
    low = bytes(
        value ^ mask
        for value, mask in zip(
            (
                cib_checksum & 0xFF,
                solution_checksum & 0xFF,
                fill_checksum & 0xFF,
                strings_checksum & 0xFF,
            ),
            _MASKED_CHECKSUM_MAGIC[:4],
            strict=True,
        )
    )
    high = bytes(
        value ^ mask
        for value, mask in zip(
            (
                (cib_checksum >> 8) & 0xFF,
                (solution_checksum >> 8) & 0xFF,
                (fill_checksum >> 8) & 0xFF,
                (strings_checksum >> 8) & 0xFF,
            ),
            _MASKED_CHECKSUM_MAGIC[4:],
            strict=True,
        )
    )
    return low, high


def _string_fields(payload: _PuzPayload) -> tuple[bytes, ...]:
    return (
        _with_terminator(payload.title),
        _with_terminator(payload.author),
        _with_terminator(payload.copyright),
        *(_with_terminator(clue) for clue in payload.clues),
        _with_terminator(payload.notes),
    )


def _clue_count(puzzle: Puzzle) -> int:
    return len(puzzle.across) + len(puzzle.down)


def puzzle_to_puz_bytes(puzzle: Puzzle) -> bytes:
    width = len(puzzle.grid.rows[0])
    height = len(puzzle.grid.rows)
    payload = _build_payload(puzzle)
    strings = b"".join(_string_fields(payload))
    cib = bytes(
        (
            width,
            height,
            _clue_count(puzzle) & 0xFF,
            (_clue_count(puzzle) >> 8) & 0xFF,
            _PUZZLE_TYPE_NORMAL & 0xFF,
            (_PUZZLE_TYPE_NORMAL >> 8) & 0xFF,
            _SOLUTION_STATE_UNLOCKED & 0xFF,
            (_SOLUTION_STATE_UNLOCKED >> 8) & 0xFF,
        )
    )
    cib_checksum = _checksum(cib)
    solution_checksum = _checksum(payload.solution)
    fill_checksum = _checksum(payload.fill)
    strings_checksum = _checksum(strings)
    header_low, header_high = _header_checksum_bytes(
        cib_checksum,
        solution_checksum,
        fill_checksum,
        strings_checksum,
    )
    overall_checksum = _checksum(strings, _checksum(payload.fill, _checksum(payload.solution, cib_checksum)))

    header = bytearray()
    header.extend(overall_checksum.to_bytes(2, "little"))
    header.extend(_HEADER_MAGIC)
    header.extend(cib_checksum.to_bytes(2, "little"))
    header.extend(header_low)
    header.extend(header_high)
    header.extend(_VERSION)
    header.extend((0).to_bytes(2, "little"))
    header.extend((0).to_bytes(2, "little"))
    header.extend(b"\x00" * 12)
    header.extend(cib)

    return bytes(header) + payload.solution + payload.fill + strings


def puzzle_has_consistent_answers(puzzle: Puzzle) -> bool:
    across_answers = tuple(clue.answer for clue in puzzle.across)
    down_answers = tuple(clue.answer for clue in puzzle.down)
    return across_answers == puzzle.grid.rows and down_answers == grid_columns(puzzle.grid)
