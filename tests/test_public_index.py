from pathlib import Path
import re
import unittest


INDEX_HTML = Path(__file__).resolve().parents[1] / "public" / "index.html"


class TestPublicIndex(unittest.TestCase):
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
        self.assertIn(".win-actions .key {", html)
        self.assertIn("padding-top: 22px;", html)
        self.assertIn("padding-bottom: 22px;", html)

    def test_mobile_board_size_is_limited_by_available_viewport_height(self) -> None:
        html = INDEX_HTML.read_text(encoding="utf-8")

        self.assertIn("--mobile-pad-bottom: calc(12px + env(safe-area-inset-bottom, 0px));", html)
        self.assertIn("--cell-size: min(", html)
        self.assertIn("100dvh - var(--mobile-topbar-height) - var(--mobile-pad-top) - var(--mobile-pad-bottom)", html)
        self.assertIn(".content {", html)
        self.assertIn("grid-template-rows: auto auto auto;", html)
        self.assertIn(".keyboard { margin-top: 0; padding-top: 0; }", html)


if __name__ == "__main__":
    unittest.main()
