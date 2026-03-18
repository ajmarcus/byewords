import json
import tempfile
import unittest
from pathlib import Path

from byewords.data_maintenance import sort_bundled_data_files
from byewords.puzzle_store import load_puzzle_store, puzzle_store_version
from byewords.theme import lexicon_hash


class TestDataMaintenance(unittest.TestCase):
    def test_sort_bundled_data_files_sorts_and_prunes_all_bundled_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            words_path = Path(directory, "words_5.txt")
            clue_bank_path = Path(directory, "clue_bank.json")
            vectors_path = Path(directory, "word_vectors.json")
            puzzles_path = Path(directory, "puzzles.json")

            words_path.write_text(
                "snail\naback\nrotas\nsator\nSnail\nbad!\narepo\ntenet\nopera\n",
                encoding="utf-8",
            )
            clue_bank_path.write_text(
                json.dumps(
                    {
                        "snail": [" Slow walker ", "", "One who leaves a trail"],
                        "aback": ["Taken by surprise"],
                        "sator": ["Ancient square word"],
                        "arepo": ["Ancient square word"],
                        "tenet": ["Guiding principle"],
                        "opera": ["Stage work"],
                        "rotas": ["Ancient square word"],
                        "bogon": ["Network junk packet"],
                        "toolong": ["Not a legal answer"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            vectors_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "source": "unit-test-vectors",
                        "dimensions": 2,
                        "lexicon_hash": "stalehash",
                        "quantization": {"scheme": "int8", "scale": 0.5},
                        "vectors": {
                            "Snail": [1, 1],
                            "aback": [1, 2],
                            "arepo": [2, 3],
                            "opera": [3, 4],
                            "rotas": [4, 5],
                            "sator": [5, 6],
                            "bogon": [7, 8],
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            puzzles_path.write_text(
                json.dumps(
                    {
                        "good-id": {
                            "uuid": "good-id",
                            "seed": "SATOR",
                            "version": "stale",
                            "title": "SATOR Mini",
                            "theme_words": ["Sator", "bogon"],
                            "theme_subset": ["Rotas", "bogon"],
                            "grid": ["SATOR", "AREPO", "TENET", "OPERA", "ROTAS"],
                            "answers": ["bogon"],
                            "answer_scores": {
                                "fill_score": 0.6,
                                "theme_score": 0.2,
                                "clue_score": 0.5,
                                "total_score": 1.3,
                                "answer_only_score": 0.8,
                                "seed_entry_count": 1,
                                "seed_row_count": 1,
                            },
                            "across": [],
                            "down": [],
                        },
                        "bad-id": {
                            "uuid": "bad-id",
                            "seed": "bogon",
                            "version": "stale",
                            "title": "BOGON Mini",
                            "theme_words": ["bogon"],
                            "theme_subset": ["bogon"],
                            "grid": ["BOGON", "BOGON", "BOGON", "BOGON", "BOGON"],
                            "answers": ["bogon"],
                            "answer_scores": {
                                "fill_score": 0.1,
                                "theme_score": 0.0,
                                "clue_score": 0.1,
                                "total_score": 0.2,
                                "answer_only_score": 0.1,
                                "seed_entry_count": 1,
                                "seed_row_count": 5,
                            },
                            "across": [],
                            "down": [],
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = sort_bundled_data_files(
                words_path,
                clue_bank_path,
                vectors_path=vectors_path,
                puzzles_path=puzzles_path,
            )

            expected_words = ("aback", "arepo", "opera", "rotas", "sator", "snail", "tenet")
            self.assertEqual(
                words_path.read_text(encoding="utf-8"),
                "aback\narepo\nopera\nrotas\nsator\nsnail\ntenet\n",
            )
            self.assertEqual(
                json.loads(clue_bank_path.read_text(encoding="utf-8")),
                {
                    "aback": ["Taken by surprise"],
                    "arepo": ["Ancient square word"],
                    "opera": ["Stage work"],
                    "rotas": ["Ancient square word"],
                    "sator": ["Ancient square word"],
                    "snail": ["Slow walker"],
                    "tenet": ["Guiding principle"],
                },
            )
            self.assertEqual(
                json.loads(vectors_path.read_text(encoding="utf-8")),
                {
                    "dimensions": 2,
                    "lexicon_hash": lexicon_hash(expected_words),
                    "quantization": {"scale": 0.5, "scheme": "int8"},
                    "source": "unit-test-vectors",
                    "vectors": {
                        "aback": [1, 2],
                        "arepo": [2, 3],
                        "opera": [3, 4],
                        "rotas": [4, 5],
                        "sator": [5, 6],
                        "snail": [1, 1],
                    },
                    "version": 1,
                },
            )
            store = load_puzzle_store(puzzles_path)
            self.assertEqual(tuple(store), ("good-id",))
            self.assertEqual(store["good-id"]["seed"], "sator")
            self.assertEqual(store["good-id"]["theme_words"], ["sator"])
            self.assertEqual(store["good-id"]["theme_subset"], ["rotas"])
            self.assertEqual(store["good-id"]["answers"], list(("sator", "arepo", "tenet", "opera", "rotas", "sator", "arepo", "tenet", "opera", "rotas")))
            self.assertEqual(
                store["good-id"]["version"],
                puzzle_store_version(
                    expected_words,
                    {
                        "aback": ("Taken by surprise",),
                        "arepo": ("Ancient square word",),
                        "opera": ("Stage work",),
                        "rotas": ("Ancient square word",),
                        "sator": ("Ancient square word",),
                        "snail": ("Slow walker",),
                        "tenet": ("Guiding principle",),
                    },
                ),
            )
            self.assertTrue(store["good-id"]["across"])
            self.assertTrue(store["good-id"]["down"])
            self.assertEqual(result.word_count, 7)
            self.assertEqual(result.clue_entry_count, 7)
            self.assertEqual(result.vector_entry_count, 6)
            self.assertEqual(result.puzzle_record_count, 1)
            self.assertEqual(result.removed_clue_answers, ("bogon", "toolong"))
            self.assertEqual(result.removed_vector_words, ("bogon",))
            self.assertEqual(result.missing_vector_words, ("tenet",))
            self.assertEqual(result.removed_puzzle_records, ("bad-id",))


if __name__ == "__main__":
    unittest.main()
