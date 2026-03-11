from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from byewords.puzzle_store import build_batch_puzzle_cache, load_puzzle_store, puzzle_answers_for_id
from byewords.types import Puzzle
from tests.fixtures import TEST_LEXICON
from tests.test_puz import build_test_puzzle


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
        self.assertTrue(all(record["theme_words"] == [record["seed"]] for record in store.values()))
        self.assertTrue(all(len(record["answers"]) == 10 for record in store.values()))
        self.assertTrue(all(record["answer_scores"]["seed_entry_count"] == 1 for record in store.values()))

    def test_build_batch_puzzle_cache_generates_each_seed_individually(self) -> None:
        with TemporaryDirectory() as temp_dir:
            store_path = Path(temp_dir) / "puzzles.json"
            calls: list[tuple[str, ...]] = []

            def fake_generate(
                seeds: tuple[str, ...],
                lexicon_words: tuple[str, ...],
                clue_bank: dict[str, tuple[str, ...]],
            ) -> Puzzle:
                self.assertEqual(lexicon_words, ("adieu", "snail"))
                self.assertEqual(clue_bank, {})
                calls.append(seeds)
                base = build_test_puzzle()
                seed = seeds[0]
                return Puzzle(
                    grid=base.grid,
                    across=base.across,
                    down=base.down,
                    theme_words=(seed,),
                    title=f"{seed.upper()} Mini",
                )

            with patch("byewords.puzzle_store.generate_puzzle", side_effect=fake_generate):
                _, total_records, generated_records = build_batch_puzzle_cache(
                    ("adieu", "snail"),
                    {},
                    path=store_path,
                )

        self.assertEqual(sorted(calls), [("adieu",), ("snail",)])
        self.assertEqual(total_records, 2)
        self.assertEqual(generated_records, 2)

    def test_build_batch_puzzle_cache_skips_generic_fallback_results(self) -> None:
        with TemporaryDirectory() as temp_dir:
            store_path = Path(temp_dir) / "puzzles.json"

            def fake_generate(
                seeds: tuple[str, ...],
                lexicon_words: tuple[str, ...],
                clue_bank: dict[str, tuple[str, ...]],
            ) -> Puzzle:
                self.assertEqual(lexicon_words, ("beach", "snail"))
                self.assertEqual(clue_bank, {})
                base = build_test_puzzle()
                if seeds == ("snail",):
                    return base
                return Puzzle(
                    grid=base.grid,
                    across=base.across,
                    down=base.down,
                    theme_words=(),
                    title="BYEWORDS Mini",
                )

            with patch("byewords.puzzle_store.generate_puzzle", side_effect=fake_generate):
                _, total_records, generated_records = build_batch_puzzle_cache(
                    ("beach", "snail"),
                    {},
                    path=store_path,
                )
                store = load_puzzle_store(store_path)

        self.assertEqual(total_records, 1)
        self.assertEqual(generated_records, 1)
        self.assertEqual({record["seed"] for record in store.values()}, {"snail"})

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
