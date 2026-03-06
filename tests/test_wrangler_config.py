import json
from pathlib import Path
import unittest


WRANGLER_JSONC = Path(__file__).resolve().parents[1] / "public" / "wrangler.jsonc"


class TestWranglerConfig(unittest.TestCase):
    def test_public_wrangler_is_static_assets_only(self) -> None:
        config = json.loads(WRANGLER_JSONC.read_text(encoding="utf-8"))

        self.assertEqual(config["name"], "byewords")
        self.assertEqual(config["assets"], {"directory": "."})
        self.assertNotIn("main", config)
        self.assertNotIn("compatibility_flags", config)
        self.assertNotIn("observability", config)


if __name__ == "__main__":
    unittest.main()
