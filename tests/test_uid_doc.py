from pathlib import Path
from uuid import UUID
import unittest


UID_DOC = Path(__file__).resolve().parents[1] / "docs" / "uid.md"
BASE62_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def decode_base62(value: str) -> int:
    decoded = 0
    for character in value:
        decoded = decoded * 62 + BASE62_ALPHABET.index(character)
    return decoded


class TestUidDoc(unittest.TestCase):
    def test_uid_doc_exists_and_covers_required_design_points(self) -> None:
        text = UID_DOC.read_text(encoding="utf-8")

        self.assertIn("# Puzzle UID Plan", text)
        self.assertIn("UUIDv7", text)
        self.assertIn("URL-safe base62 string", text)
        self.assertIn("puzzles.json", text)
        self.assertIn("stored puzzle record", text)
        self.assertIn("not a canonical puzzle content hash", text)
        self.assertIn("Never regenerate the id from puzzle contents.", text)
        self.assertIn("grid_letters", text)
        self.assertIn("primary_seed_word", text)
        self.assertIn("moves to a database later", text)
        self.assertIn("## Summary", text)

    def test_example_base62_id_round_trips_to_v7_uuid(self) -> None:
        uuid_value = "019577fd-8d7e-7a3d-9a4b-c2f6b6a1d233"
        public_id = "2zIa7ZJKCNdqA8J5qu7MZ"
        parsed = UUID(uuid_value)

        self.assertTrue(all(character in BASE62_ALPHABET for character in public_id))
        self.assertEqual(decode_base62(public_id), parsed.int)
        self.assertEqual(parsed.version, 7)
        self.assertEqual(parsed.variant, UUID("00000000-0000-4000-8000-000000000000").variant)


if __name__ == "__main__":
    unittest.main()
