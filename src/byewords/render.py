from byewords.types import Grid, Puzzle


def render_grid_ascii(grid: Grid) -> str:
    return "\n".join(" ".join(letter.upper() for letter in row) for row in grid.rows)


def render_clues(puzzle: Puzzle) -> str:
    across_lines = ["Across"]
    across_lines.extend(f"{clue.number}. {clue.text}" for clue in puzzle.across)
    down_lines = ["Down"]
    down_lines.extend(f"{clue.number}. {clue.text}" for clue in puzzle.down)
    return "\n".join(across_lines + [""] + down_lines)


def render_puzzle_text(puzzle: Puzzle) -> str:
    sections = [puzzle.title, ""]
    if puzzle.theme_words:
        sections.extend(
            [
                f"Seed words: {', '.join(word.upper() for word in puzzle.theme_words)}",
                "",
            ]
        )
    sections.extend([render_grid_ascii(puzzle.grid), "", render_clues(puzzle)])
    return "\n".join(sections)


def puzzle_to_dict(puzzle: Puzzle) -> dict[str, object]:
    return {
        "title": puzzle.title,
        "theme_words": list(puzzle.theme_words),
        "grid": list(puzzle.grid.rows),
        "across": [clue.__dict__ for clue in puzzle.across],
        "down": [clue.__dict__ for clue in puzzle.down],
    }
