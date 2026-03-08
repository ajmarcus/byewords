import json
import tempfile
import unittest
from pathlib import Path

from byewords.data_maintenance import sort_bundled_data_files


class TestDataMaintenance(unittest.TestCase):
    def test_sort_bundled_data_files_sorts_and_prunes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            words_path = Path(directory, "words_5.txt")
            clue_bank_path = Path(directory, "clue_bank.json")

            words_path.write_text("snail\naback\nSnail\nbad!\n", encoding="utf-8")
            clue_bank_path.write_text(
                json.dumps(
                    {
                        "snail": [" Slow walker ", ""],
                        "aback": ["Taken by surprise"],
                        "bogon": ["Network junk packet"],
                        "toolong": ["Not a legal answer"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = sort_bundled_data_files(words_path, clue_bank_path)

            self.assertEqual(words_path.read_text(encoding="utf-8"), "aback\nsnail\n")
            self.assertEqual(
                json.loads(clue_bank_path.read_text(encoding="utf-8")),
                {
                    "aback": ["Taken by surprise"],
                    "snail": ["Slow walker"],
                },
            )
            self.assertEqual(result.word_count, 2)
            self.assertEqual(result.clue_entry_count, 2)
            self.assertEqual(result.removed_clue_answers, ("bogon", "toolong"))


if __name__ == "__main__":
    unittest.main()
