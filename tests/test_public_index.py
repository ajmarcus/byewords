from pathlib import Path
import re
import unittest


INDEX_HTML = Path(__file__).resolve().parents[1] / "public" / "index.html"
FAVICON_ICO = Path(__file__).resolve().parents[1] / "public" / "favicon.ico"


class TestPublicIndex(unittest.TestCase):
    def test_document_links_to_favicon(self) -> None:
        html = INDEX_HTML.read_text(encoding="utf-8")

        self.assertIn('<link rel="icon" href="favicon.ico" sizes="16x16" />', html)

    def test_favicon_file_exists_and_uses_ico_header(self) -> None:
        icon = FAVICON_ICO.read_bytes()

        self.assertGreater(len(icon), 100)
        self.assertEqual(icon[:4], b"\x00\x00\x01\x00")

    def test_embedded_demo_puzzle_matches_snail_sample(self) -> None:
        html = INDEX_HTML.read_text(encoding="utf-8")

        self.assertIn('rows: ["ADIEU", "BOOED", "ANTRA", "SNAIL", "EASES"]', html)
        self.assertIn('"French farewell"', html)
        self.assertIn('"Slow garden crawler"', html)
        self.assertIn('"Old Norse landholdings"', html)
        self.assertIn('rows: ["ABASE", "DONNA", "IOTAS", "EERIE", "UDALS"]', html)
        self.assertIn("words.map(function (word) {", html)
        self.assertIn('}).join(" / ");', html)

    def test_embedded_puzzle_bank_has_two_full_grids(self) -> None:
        html = INDEX_HTML.read_text(encoding="utf-8")

        self.assertEqual(len(re.findall(r'rows: \["[A-Z]{5}", "[A-Z]{5}", "[A-Z]{5}", "[A-Z]{5}", "[A-Z]{5}"\]', html)), 2)
        self.assertIn("const puzzles = [", html)
        self.assertIn('const PUZZLE_CURSOR_KEY = "byewords-puzzle-cursor-v1";', html)

    def test_embedded_puzzle_bank_never_repeats_clue_text(self) -> None:
        html = INDEX_HTML.read_text(encoding="utf-8")

        clue_blocks = re.findall(r'(?:acrossClues|downClues): \[(.*?)\]', html, re.S)
        clues = []
        for block in clue_blocks:
            clues.extend(re.findall(r'"([^"]+)"', block))

        self.assertEqual(len(clues), 20)
        self.assertEqual(len(clues), len(set(clues)))

    def test_embedded_puzzle_bank_uses_unique_entries_per_board(self) -> None:
        html = INDEX_HTML.read_text(encoding="utf-8")

        boards = re.findall(r'rows: \["([A-Z]{5})", "([A-Z]{5})", "([A-Z]{5})", "([A-Z]{5})", "([A-Z]{5})"\]', html)
        self.assertEqual(len(boards), 2)

        for rows in boards:
            columns = tuple("".join(row[index] for row in rows) for index in range(5))
            entries = rows + columns
            self.assertEqual(len(entries), 10)
            self.assertEqual(len(set(entries)), 10)

    def test_rotation_logic_advances_puzzles_on_load_and_finish(self) -> None:
        html = INDEX_HTML.read_text(encoding="utf-8")

        self.assertIn("function normalizePuzzleIndex(index) {", html)
        self.assertIn("function takeNextPuzzleIndex() {", html)
        self.assertIn("window.localStorage.getItem(PUZZLE_CURSOR_KEY)", html)
        self.assertIn("writePuzzleCursor((nextIndex + 1) % puzzles.length);", html)
        self.assertIn("function queueNextPuzzle() {", html)
        self.assertIn("const storedIndex = readPuzzleCursor();", html)
        self.assertIn("writePuzzleCursor(nextIndex);", html)
        self.assertIn("queueNextPuzzle();", html)
        self.assertIn("writePuzzleCursor((activePuzzleIndex + 1) % puzzles.length);", html)
        self.assertIn("activatePuzzle(takeNextPuzzleIndex());", html)
        self.assertIn("if (pendingPuzzleIndex !== null) {", html)
        self.assertIn("activatePuzzle(pendingPuzzleIndex);", html)
        self.assertIn("queued puzzle stays aligned with stored cursor until reset", html)

    def test_mobile_layout_fills_dynamic_viewport_height(self) -> None:
        html = INDEX_HTML.read_text(encoding="utf-8")

        self.assertIn(".app { height: 100vh; height: 100dvh; min-height: 100dvh; padding: 0; align-items: stretch; }", html)
        self.assertIn(".shell { height: 100vh; height: 100dvh; min-height: 100dvh; max-height: 100dvh; }", html)

    def test_keyboard_actions_only_keep_delete_button(self) -> None:
        html = INDEX_HTML.read_text(encoding="utf-8")

        self.assertIn('<button class="key" id="delete-key" type="button">Delete</button>', html)
        self.assertNotIn('id="reset-key"', html)
        self.assertNotIn(">Reset</button>", html)

    def test_keyboard_and_modal_buttons_have_extra_vertical_space(self) -> None:
        html = INDEX_HTML.read_text(encoding="utf-8")

        self.assertRegex(html, re.compile(r"--key-gap:\s*8px;"))
        self.assertRegex(html, re.compile(r"--key-pad-y:\s*12px;"))
        self.assertIn(".key-row:last-child { margin-bottom: 0; }", html)
        self.assertIn(".win-actions .key {", html)
        self.assertIn("padding-top: 14px;", html)
        self.assertIn("padding-right: 18px;", html)
        self.assertIn("padding-bottom: 14px;", html)
        self.assertIn("padding-left: 18px;", html)

    def test_mobile_board_size_is_limited_by_available_viewport_height(self) -> None:
        html = INDEX_HTML.read_text(encoding="utf-8")

        self.assertIn("--mobile-pad-bottom: calc(12px + env(safe-area-inset-bottom, 0px));", html)
        self.assertIn("--cell-size: min(", html)
        self.assertIn("100dvh - var(--mobile-topbar-height) - var(--mobile-pad-top) - var(--mobile-pad-bottom)", html)
        self.assertIn("--mobile-topbar-height: 48px;", html)
        self.assertIn("--mobile-keyboard-height: calc(((var(--key-font) + (var(--key-pad-y) * 2) + 4px) * 4) + (var(--key-gap) * 3));", html)
        self.assertIn(".content {", html)
        self.assertIn("grid-template-rows: auto auto auto;", html)
        self.assertIn("align-content: start;", html)
        self.assertIn(".keyboard { margin-top: 0; padding-top: 0; }", html)

    def test_clue_card_uses_fixed_height_and_fit_text_logic(self) -> None:
        html = INDEX_HTML.read_text(encoding="utf-8")

        self.assertIn("--mobile-clue-height: 124px;", html)
        self.assertIn("--clue-height: 156px;", html)
        self.assertIn("grid-auto-rows: var(--clue-height);", html)
        self.assertIn(".clue {", html)
        self.assertIn("height: 100%;", html)
        self.assertIn(".nav-btn {", html)
        self.assertIn("font-size: var(--clue-text-max);", html)
        self.assertIn("function fitClueText() {", html)
        self.assertIn('getPropertyValue("--clue-text-max")', html)
        self.assertIn('getPropertyValue("--clue-text-min")', html)
        self.assertIn("window.requestAnimationFrame(fitClueText);", html)
        self.assertIn('window.addEventListener("resize", fitClueText);', html)
        self.assertIn("overflow: hidden;", html)

    def test_keyboard_handler_preserves_browser_shortcuts(self) -> None:
        html = INDEX_HTML.read_text(encoding="utf-8")

        self.assertIn("if (event.metaKey || event.ctrlKey || event.altKey) {", html)
        self.assertIn("return;", html)
        self.assertIn('if (/^[A-Z]$/.test(key)) {', html)

    def test_board_uses_gap_without_cell_borders_for_uniform_lines(self) -> None:
        html = INDEX_HTML.read_text(encoding="utf-8")

        self.assertIn("--board-gap: 2px;", html)
        self.assertIn("gap: var(--board-gap);", html)
        self.assertIn("padding: var(--board-gap);", html)
        self.assertIn("--cell-edge: max(var(--cell-min), var(--cell-size));", html)
        self.assertIn("grid-template-columns: repeat(5, var(--cell-edge));", html)
        self.assertIn("grid-template-rows: repeat(5, var(--cell-edge));", html)
        self.assertIn("width: fit-content;", html)
        self.assertIn("border: 0;", html)
        self.assertIn("appearance: none;", html)


if __name__ == "__main__":
    unittest.main()
