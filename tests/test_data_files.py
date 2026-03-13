import unittest
from importlib import resources
from pathlib import Path

from byewords.generate import DEFAULT_DEMO_ENTRIES, build_demo_puzzle, generate_puzzle, load_default_inputs
from byewords.grid import distinct_entries
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

    def test_default_clues_avoid_sensitive_or_needlessly_harsh_phrasing(self) -> None:
        _, clue_bank = load_default_inputs()

        self.assertEqual(clue_bank["abuse"], ("Mistreat cruelly", "Improper or harmful use"))
        self.assertEqual(clue_bank["bares"], ("Reveals openly", "Exposes, as a secret"))
        self.assertEqual(clue_bank["naked"], ("Without any covering", "Plainly exposed to view"))
        self.assertEqual(clue_bank["slurs"], ("Connect smoothly, as notes in a phrase", "Blends together, as syllables"))
        self.assertEqual(clue_bank["trans"], ("Prefix meaning 'across' or 'beyond'", "Short for transmissions, informally"))
        self.assertEqual(clue_bank["tubes"], ("Beach floaties and underground rails, collectively", "Test ___ and old televisions, slangily"))

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
