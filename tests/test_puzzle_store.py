import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from byewords.puzzle_store import (
    build_batch_puzzle_cache,
    load_puzzle_store,
    persist_puzzle_store,
    puzzle_answers_for_id,
    puzzle_store_version,
)
from byewords.theme import lexicon_hash, load_word_vectors
from byewords.types import Puzzle
from tests.fixtures import TEST_LEXICON
from tests.test_puz import build_test_puzzle


def _write_vector_table(
    path: Path,
    lexicon: tuple[str, ...],
    vectors: dict[str, list[int]],
) -> None:
    dimensions = len(next(iter(vectors.values())))
    payload = {
        "version": 1,
        "source": "unit-test-vectors",
        "dimensions": dimensions,
        "lexicon_hash": lexicon_hash(lexicon),
        "quantization": {
            "scheme": "int8",
            "scale": 0.25,
        },
        "vectors": vectors,
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


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

    def test_build_batch_puzzle_cache_adds_semantic_theme_subset_when_vectors_match(self) -> None:
        with TemporaryDirectory() as temp_dir:
            store_path = Path(temp_dir) / "puzzles.json"
            vector_path = Path(temp_dir) / "vectors.json"
            _write_vector_table(
                vector_path,
                TEST_LEXICON,
                {
                    "abase": [0, 4, 0, 0],
                    "adieu": [0, 0, 4, 0],
                    "antra": [3, 3, 0, 0],
                    "booed": [0, 4, 0, 0],
                    "donna": [1, 3, 0, 0],
                    "eases": [4, 1, 0, 0],
                    "eerie": [0, 4, 0, 0],
                    "iotas": [0, 0, 4, 0],
                    "snail": [4, 0, 0, 0],
                    "udals": [0, 4, 0, 0],
                },
            )
            vectors = load_word_vectors(str(vector_path))

            _, _, _ = build_batch_puzzle_cache(
                TEST_LEXICON,
                {},
                path=store_path,
                vectors=vectors,
            )
            store = load_puzzle_store(store_path)

        snail_record = next(record for record in store.values() if record["seed"] == "snail")
        self.assertEqual(snail_record["theme_subset"], ["eases", "antra", "donna"])
        self.assertGreater(snail_record["answer_scores"]["theme_score"], 0.0)
        self.assertGreater(
            snail_record["answer_scores"]["total_score"],
            snail_record["answer_scores"]["fill_score"] + snail_record["answer_scores"]["clue_score"],
        )

    def test_build_batch_puzzle_cache_curates_best_current_version_per_seed(self) -> None:
        with TemporaryDirectory() as temp_dir:
            store_path = Path(temp_dir) / "puzzles.json"
            version = puzzle_store_version(("adieu", "snail"), {})
            persist_puzzle_store(
                {
                    "older": {
                        "uuid": "00000000-0000-7000-8000-000000000001",
                        "seed": "snail",
                        "version": version,
                        "title": "SNAIL Mini",
                        "theme_words": ["snail"],
                        "theme_subset": ["eases"],
                        "grid": ["adieu"] * 5,
                        "answers": ["adieu", "snail"],
                        "answer_scores": {
                            "fill_score": 0.5,
                            "theme_score": 0.2,
                            "clue_score": 0.3,
                            "total_score": 1.0,
                            "seed_entry_count": 1,
                            "seed_row_count": 1,
                        },
                        "across": [],
                        "down": [],
                    },
                    "newer": {
                        "uuid": "00000000-0000-7000-8000-000000000002",
                        "seed": "snail",
                        "version": version,
                        "title": "SNAIL Mini",
                        "theme_words": ["snail"],
                        "theme_subset": ["eases", "antra"],
                        "grid": ["snail"] * 5,
                        "answers": ["snail", "adieu"],
                        "answer_scores": {
                            "fill_score": 0.7,
                            "theme_score": 0.6,
                            "clue_score": 0.4,
                            "total_score": 1.7,
                            "seed_entry_count": 1,
                            "seed_row_count": 1,
                        },
                        "across": [],
                        "down": [],
                    },
                    "adieu": {
                        "uuid": "00000000-0000-7000-8000-000000000003",
                        "seed": "adieu",
                        "version": version,
                        "title": "ADIEU Mini",
                        "theme_words": ["adieu"],
                        "theme_subset": ["snail"],
                        "grid": ["adieu"] * 5,
                        "answers": ["adieu", "snail"],
                        "answer_scores": {
                            "fill_score": 0.4,
                            "theme_score": 0.2,
                            "clue_score": 0.2,
                            "total_score": 0.8,
                            "seed_entry_count": 1,
                            "seed_row_count": 1,
                        },
                        "across": [],
                        "down": [],
                    },
                },
                store_path,
            )

            _, total_records, generated_records = build_batch_puzzle_cache(
                ("adieu", "snail"),
                {},
                path=store_path,
            )
            store = load_puzzle_store(store_path)

        self.assertEqual(total_records, 2)
        self.assertEqual(generated_records, 0)
        self.assertEqual(set(store), {"adieu", "newer"})
        self.assertEqual(store["newer"]["answer_scores"]["total_score"], 1.7)

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
