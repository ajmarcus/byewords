import unittest

from byewords.generate import DEFAULT_DEMO_ENTRIES, build_demo_puzzle, generate_puzzle, load_default_inputs
from byewords.grid import distinct_entries


class TestBundledData(unittest.TestCase):
    def test_default_data_is_large_and_consistent(self) -> None:
        lexicon_words, clue_bank = load_default_inputs()

        self.assertGreaterEqual(len(lexicon_words), 1000)
        self.assertGreaterEqual(len(clue_bank), 1000)
        self.assertEqual(lexicon_words, tuple(sorted(lexicon_words)))
        self.assertEqual(tuple(clue_bank), tuple(sorted(clue_bank)))
        self.assertTrue({"snail", "water", "ozone"}.issubset(set(clue_bank)))
        self.assertTrue(set(clue_bank).issubset(set(lexicon_words)))
        self.assertTrue(all(clues and all(clue.strip() for clue in clues) for clues in clue_bank.values()))

    def test_default_data_keeps_common_theme_clusters_and_excludes_junk(self) -> None:
        lexicon_words, clue_bank = load_default_inputs()
        lexicon_set = set(lexicon_words)

        expected_words = {"beach", "music", "ocean", "piano", "tempo", "waves", "wharf"}
        self.assertTrue(expected_words.issubset(lexicon_set))

        for removed_word in ("aahed", "antra", "ikeas", "lurie", "udals"):
            self.assertNotIn(removed_word, lexicon_set)
            self.assertNotIn(removed_word, clue_bank)

    def test_default_data_excludes_offensive_and_obscure_fill(self) -> None:
        lexicon_words, _ = load_default_inputs()
        lexicon_set = set(lexicon_words)

        removed_words = {
            "alkyd",
            "anent",
            "ankhs",
            "apses",
            "arsed",
            "asses",
            "assoc",
            "attar",
            "auxin",
            "ayahs",
            "baaed",
            "baccy",
            "bauds",
            "bairn",
            "baize",
            "bawdy",
            "bawds",
            "begum",
            "besom",
            "bimbo",
            "bitch",
            "bogon",
            "boink",
            "boner",
            "boobs",
            "broad",
            "bumph",
            "busby",
            "caber",
            "caffs",
            "chink",
            "clits",
            "clvii",
            "clxii",
            "clxiv",
            "clxix",
            "clxvi",
            "cocks",
            "cohos",
            "coons",
            "contd",
            "cunts",
            "coypu",
            "dagos",
            "dding",
            "deice",
            "dhows",
            "dicks",
            "dildo",
            "dipso",
            "dykes",
            "effed",
            "eruct",
            "fagot",
            "fatso",
            "fichu",
            "fanny",
            "farts",
            "fucks",
            "gimps",
            "gonks",
            "gooks",
            "gorps",
            "gypsy",
            "gyved",
            "hdqrs",
            "hicks",
            "homos",
            "honky",
            "horny",
            "hying",
            "ictus",
            "iambi",
            "instr",
            "japed",
            "jatos",
            "kepis",
            "kraut",
            "labia",
            "lepta",
            "limns",
            "limey",
            "luffs",
            "lxvii",
            "mammy",
            "micks",
            "milfs",
            "moues",
            "mulct",
            "nooky",
            "pekoe",
            "penis",
            "pewit",
            "piing",
            "pinko",
            "pimps",
            "pommy",
            "ponce",
            "poofs",
            "porno",
            "prick",
            "pubes",
            "pubic",
            "pubis",
            "pussy",
            "pyxes",
            "pzazz",
            "quirt",
            "scrog",
            "sexed",
            "shits",
            "shoat",
            "spics",
            "sperm",
            "spunk",
            "sputa",
            "squaw",
            "stdio",
            "thews",
            "topee",
            "trugs",
            "titty",
            "twats",
            "turds",
            "ulnae",
            "ukase",
            "umiak",
            "vised",
            "vulva",
            "wanks",
            "wazoo",
            "weest",
            "welsh",
            "wench",
            "whore",
            "whups",
            "willy",
            "wived",
            "xcvii",
            "yeggs",
            "zebus",
            "zorch",
        }

        self.assertTrue(removed_words.isdisjoint(lexicon_set))

    def test_default_demo_puzzle_uses_fallback_clues_when_cache_is_empty(self) -> None:
        _, clue_bank = load_default_inputs()
        puzzle = build_demo_puzzle(clue_bank)

        self.assertEqual(puzzle.title, "BYEWORDS Mini")
        self.assertEqual(len(puzzle.across), 5)
        self.assertEqual(len(puzzle.down), 5)

        for clue in puzzle.across:
            self.assertTrue(clue.text.strip())

        for clue in puzzle.down:
            self.assertTrue(clue.text.strip())

    def test_readme_example_generates_a_puzzle_without_seed_words(self) -> None:
        _, clue_bank = load_default_inputs()

        puzzle = build_demo_puzzle(clue_bank)

        self.assertEqual(puzzle.title, "BYEWORDS Mini")
        self.assertEqual(len(puzzle.across), 5)
        self.assertEqual(len(puzzle.down), 5)
        self.assertEqual(len(set(distinct_entries(puzzle.grid))), 10)

    def test_demo_grid_entries_are_verified_single_seed_words(self) -> None:
        lexicon_words, clue_bank = load_default_inputs()

        for seed in DEFAULT_DEMO_ENTRIES:
            with self.subTest(seed=seed):
                puzzle = generate_puzzle((seed,), lexicon_words, clue_bank)

                self.assertEqual(puzzle.title, f"{seed.upper()} Mini")
                self.assertIn(seed, distinct_entries(puzzle.grid))
                self.assertEqual(puzzle.grid.rows, ("ozone", "liven", "inert", "verve", "ester"))

if __name__ == "__main__":
    unittest.main()
