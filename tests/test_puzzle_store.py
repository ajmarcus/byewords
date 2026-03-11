from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from byewords.puzzle_store import build_batch_puzzle_cache, load_puzzle_store, puzzle_answers_for_id
from tests.fixtures import TEST_LEXICON


class TestPuzzleStore(unittest.TestCase):
    def test_build_batch_puzzle_cache_populates_one_record_per_seed(self) -> None:
        with TemporaryDirectory() as temp_dir:
            store_path = Path(temp_dir) / "puzzles.json"

            written_path, total_records, generated_records = build_batch_puzzle_cache(
                TEST_LEXICON,
                {},
                path=store_path,
            )

            store = load_puzzle_store(store_path)

        self.assertEqual(written_path, store_path)
        self.assertEqual(total_records, len(TEST_LEXICON))
        self.assertEqual(generated_records, len(TEST_LEXICON))
        self.assertEqual({record["seed"] for record in store.values()}, set(TEST_LEXICON))

    def test_build_batch_puzzle_cache_reuses_existing_versioned_records(self) -> None:
        with TemporaryDirectory() as temp_dir:
            store_path = Path(temp_dir) / "puzzles.json"
            build_batch_puzzle_cache(TEST_LEXICON, {}, path=store_path)

            _, total_records, generated_records = build_batch_puzzle_cache(
                TEST_LEXICON,
                {},
                path=store_path,
            )

        self.assertEqual(total_records, len(TEST_LEXICON))
        self.assertEqual(generated_records, 0)

    def test_puzzle_answers_for_id_supports_public_id_and_uuid_lookup(self) -> None:
        with TemporaryDirectory() as temp_dir:
            store_path = Path(temp_dir) / "puzzles.json"
            build_batch_puzzle_cache(TEST_LEXICON, {}, path=store_path)
            store = load_puzzle_store(store_path)
            public_id, record = next(iter(store.items()))

            public_answers = puzzle_answers_for_id(public_id, store_path)
            uuid_answers = puzzle_answers_for_id(record["uuid"], store_path)

        self.assertEqual(public_answers, uuid_answers)
        self.assertTrue(public_answers)


if __name__ == "__main__":
    unittest.main()
