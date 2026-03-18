import unittest
from importlib import resources
from pathlib import Path

from byewords.clue_bank import is_generic_clue
from byewords.generate import DEFAULT_DEMO_ENTRIES, build_demo_puzzle, generate_puzzle, load_default_inputs
from byewords.grid import distinct_entries
from byewords.lexicon import load_clue_bank
from byewords.puzzle_store import load_puzzle_store, puzzle_store_version
from byewords.score import score_grid
from byewords.theme import (
    THEME_BENCHMARK_SEEDS,
    THEME_INTRUSION_REVIEW_CASES,
    THEME_MANUAL_REVIEW_CASES,
    THEME_RETRIEVAL_REVIEW_CASES,
    lexicon_hash,
    load_word_vectors,
)


README_PATH = Path(__file__).resolve().parents[1] / "README.md"
README_RECOMMENDED_SEEDS = ("beach", "ocean", "music", "piano", "tempo")
NON_GENERIC_CLUE_REGRESSION_ANSWERS = (
    "bride",
    "buyer",
    "curer",
    "fayer",
    "firer",
    "hider",
    "idler",
    "owner",
    "taker",
    "trier",
)


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

    def test_bundled_clue_bank_contains_no_generic_clues(self) -> None:
        clue_bank_path = str(resources.files("byewords").joinpath("data", "clue_bank.json"))
        clue_bank = load_clue_bank(clue_bank_path)

        for answer, clues in clue_bank.items():
            with self.subTest(answer=answer):
                self.assertTrue(all(not is_generic_clue(clue) for clue in clues))

    def test_known_regression_answers_keep_only_non_generic_clues(self) -> None:
        clue_bank_path = str(resources.files("byewords").joinpath("data", "clue_bank.json"))
        clue_bank = load_clue_bank(clue_bank_path)

        for answer in NON_GENERIC_CLUE_REGRESSION_ANSWERS:
            with self.subTest(answer=answer):
                self.assertIn(answer, clue_bank)
                self.assertTrue(clue_bank[answer])
                self.assertTrue(all(not is_generic_clue(clue) for clue in clue_bank[answer]))

    def test_default_word_vectors_cover_the_full_bundled_lexicon(self) -> None:
        lexicon_words, _ = load_default_inputs()
        vector_path = str(resources.files("byewords").joinpath("data", "word_vectors.json"))

        vectors = load_word_vectors(vector_path)

        self.assertEqual(vectors.version, 1)
        self.assertEqual(vectors.quantization_scheme, "int8")
        self.assertIn(
            (vectors.source, vectors.dimensions),
            {
                ("hashed-clue-features-v1", 128),
                ("baai-bge-small-en-v1.5", 384),
            },
        )
        self.assertEqual(vectors.lexicon_hash, lexicon_hash(lexicon_words))
        self.assertEqual(tuple(sorted(vectors.vectors)), lexicon_words)

    def test_bundled_puzzle_cache_uses_only_current_lexicon_words(self) -> None:
        lexicon_words, clue_bank = load_default_inputs(include_fallback_clues=False)
        lexicon_set = set(lexicon_words)
        store = load_puzzle_store()

        self.assertTrue(store)
        expected_version = puzzle_store_version(lexicon_words, clue_bank)
        for public_id, record in store.items():
            with self.subTest(public_id=public_id):
                self.assertEqual(record["version"], expected_version)
                self.assertIn(record["seed"], lexicon_set)
                self.assertTrue(set(record["answers"]).issubset(lexicon_set))
                self.assertTrue(set(record.get("theme_words", [])).issubset(lexicon_set))
                self.assertTrue(set(record.get("theme_subset", [])).issubset(lexicon_set))

    def test_readme_lists_five_verified_seed_recommendations(self) -> None:
        text = README_PATH.read_text(encoding="utf-8")

        self.assertIn("Five reliable single-word seeds with end-to-end regression coverage:", text)
        for seed in README_RECOMMENDED_SEEDS:
            self.assertIn(seed, text)
        self.assertIn("BAAI/bge-small-en-v1.5", text)
        self.assertIn("MIT", text)

    def test_readme_recommended_seeds_generate_seeded_puzzles(self) -> None:
        lexicon_words, clue_bank = load_default_inputs()

        for seed in README_RECOMMENDED_SEEDS:
            with self.subTest(seed=seed):
                puzzle = generate_puzzle((seed,), lexicon_words, clue_bank)
                entries = distinct_entries(puzzle.grid)
                scored = score_grid(puzzle.grid)

                self.assertEqual(puzzle.title, f"{seed.upper()} Mini")
                self.assertEqual(puzzle.theme_words, (seed,))
                self.assertIn(seed, entries)
                self.assertEqual(len(set(entries)), 10)
                self.assertGreaterEqual(scored.fill_score, 0.35)

    def test_default_data_keeps_common_theme_clusters_and_excludes_junk(self) -> None:
        lexicon_words, clue_bank = load_default_inputs()
        lexicon_set = set(lexicon_words)

        expected_words = {"beach", "music", "ocean", "piano", "tempo", "waves", "wharf"}
        self.assertTrue(expected_words.issubset(lexicon_set))

        for removed_word in ("aahed", "antra", "ikeas", "lurie", "udals"):
            self.assertNotIn(removed_word, lexicon_set)
            self.assertNotIn(removed_word, clue_bank)

    def test_theme_seed_corpora_are_backed_by_bundled_words(self) -> None:
        lexicon_words, clue_bank = load_default_inputs()
        lexicon_set = set(lexicon_words)

        for difficulty, seeds in THEME_BENCHMARK_SEEDS.items():
            with self.subTest(difficulty=difficulty):
                self.assertTrue(seeds)
                self.assertTrue(set(seeds).issubset(lexicon_set))

        for case in THEME_MANUAL_REVIEW_CASES:
            with self.subTest(seed=case.seed):
                expected_words = {case.seed, *case.expected_related_words}
                self.assertTrue(expected_words.issubset(lexicon_set))
                self.assertTrue(expected_words.issubset(set(clue_bank)))

        for case in THEME_RETRIEVAL_REVIEW_CASES:
            with self.subTest(seed=f"{case.seed}-retrieval"):
                expected_words = {case.seed, *case.expected_top_words, *case.unexpected_top_words}
                self.assertTrue(expected_words.issubset(lexicon_set))
                self.assertTrue(set(case.expected_top_words).issubset(set(clue_bank)))

        for case in THEME_INTRUSION_REVIEW_CASES:
            with self.subTest(seed=f"{case.seed}-intrusion"):
                expected_words = {case.seed, *case.expected_theme_words, *case.intruder_words}
                self.assertTrue(expected_words.issubset(lexicon_set))
                self.assertTrue(set(case.expected_theme_words).issubset(set(clue_bank)))

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
            "abaft",
            "abbes",
            "abeam",
            "acmes",
            "admix",
            "adzes",
            "arums",
            "bawdy",
            "bawds",
            "bedim",
            "begum",
            "besom",
            "besot",
            "bimbo",
            "bitch",
            "blags",
            "bogon",
            "boink",
            "boner",
            "boobs",
            "booby",
            "broad",
            "bumph",
            "butch",
            "busby",
            "caber",
            "caffs",
            "chink",
            "chivy",
            "clits",
            "clvii",
            "clxii",
            "clxiv",
            "clxix",
            "clxvi",
            "cocks",
            "cloys",
            "cohos",
            "coons",
            "contd",
            "cunts",
            "coypu",
            "dagos",
            "daces",
            "dding",
            "deice",
            "dhows",
            "dicks",
            "dildo",
            "dipso",
            "dykes",
            "effed",
            "epees",
            "eruct",
            "faffs",
            "fagot",
            "fatso",
            "fichu",
            "fanny",
            "farts",
            "fucks",
            "gelds",
            "gimps",
            "gonks",
            "gonad",
            "gooks",
            "gorps",
            "gypsy",
            "gyves",
            "gyved",
            "hasps",
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
            "kayos",
            "kepis",
            "kinky",
            "kraut",
            "labia",
            "lases",
            "lepta",
            "limns",
            "limey",
            "lilos",
            "luffs",
            "lxvii",
            "mammy",
            "micks",
            "milfs",
            "milts",
            "moils",
            "moues",
            "mulct",
            "neaps",
            "nimbi",
            "nooky",
            "nuder",
            "nudes",
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
            "pules",
            "pussy",
            "pyxes",
            "pzazz",
            "queer",
            "quirt",
            "redye",
            "resew",
            "resow",
            "roues",
            "sades",
            "scrog",
            "semen",
            "sexed",
            "shits",
            "shirr",
            "shoat",
            "slave",
            "spics",
            "sperm",
            "spumy",
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
            "umped",
            "veeps",
            "velds",
            "viand",
            "vised",
            "vulva",
            "wanks",
            "wazoo",
            "weens",
            "weest",
            "welsh",
            "wench",
            "whore",
            "whups",
            "willy",
            "wived",
            "wryer",
            "xcvii",
            "yeggs",
            "zebus",
            "zorch",
        }

        self.assertTrue(removed_words.isdisjoint(lexicon_set))

    def test_default_clues_keep_non_generic_leading_variants_for_sensitive_entries(self) -> None:
        _, clue_bank = load_default_inputs()
        for answer in ("abuse", "bares", "naked", "slurs", "trans", "tubes"):
            with self.subTest(answer=answer):
                leading_clues = clue_bank[answer][:2]
                self.assertEqual(len(leading_clues), 2)
                self.assertTrue(all(clue.strip() for clue in leading_clues))
                self.assertTrue(all(not is_generic_clue(clue) for clue in leading_clues))

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
