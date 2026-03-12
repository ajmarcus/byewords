import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from byewords.theme import (
    THEME_BENCHMARK_SEEDS,
    THEME_INTRUSION_REVIEW_CASES,
    THEME_MANUAL_REVIEW_CASES,
    THEME_RETRIEVAL_REVIEW_CASES,
    ThemeIntrusionReviewCase,
    build_candidate_pool,
    compare_theme_intrusions,
    compare_retrieval_metrics,
    diversify_theme_words,
    lexicon_hash,
    load_word_vectors,
    normalize_seeds,
    rank_lexicon_for_seed,
    rank_overlap_relevance_scores,
    rank_theme_candidates,
    review_theme_intrusions,
    review_theme_retrieval,
    score_theme_subset,
    score_word_for_seed,
    validate_seed_words,
)


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


class TestTheme(unittest.TestCase):
    def test_theme_benchmark_seed_corpus_is_grouped_by_difficulty(self) -> None:
        self.assertEqual(tuple(THEME_BENCHMARK_SEEDS), ("easy", "medium", "hard"))
        self.assertEqual(THEME_BENCHMARK_SEEDS["easy"], ("beach", "music", "ocean"))
        self.assertEqual(THEME_BENCHMARK_SEEDS["medium"], ("snail", "tempo", "water"))
        self.assertEqual(THEME_BENCHMARK_SEEDS["hard"], ("doggy", "llama", "wharf"))

    def test_theme_manual_review_cases_are_unique_and_non_empty(self) -> None:
        self.assertEqual(len(THEME_MANUAL_REVIEW_CASES), 3)
        self.assertEqual(
            tuple(case.seed for case in THEME_MANUAL_REVIEW_CASES),
            ("beach", "music", "snail"),
        )
        for case in THEME_MANUAL_REVIEW_CASES:
            with self.subTest(seed=case.seed):
                self.assertTrue(case.note)
                self.assertEqual(len(case.expected_related_words), 3)
                self.assertEqual(len(set(case.expected_related_words)), 3)

    def test_theme_retrieval_review_cases_are_unique_and_non_empty(self) -> None:
        self.assertEqual(len(THEME_RETRIEVAL_REVIEW_CASES), 3)
        self.assertEqual(
            tuple(case.seed for case in THEME_RETRIEVAL_REVIEW_CASES),
            ("beach", "music", "snail"),
        )
        for case in THEME_RETRIEVAL_REVIEW_CASES:
            with self.subTest(seed=case.seed):
                self.assertTrue(case.note)
                self.assertEqual(len(case.expected_top_words), 3)
                self.assertEqual(len(case.unexpected_top_words), 3)
                self.assertTrue(set(case.expected_top_words).isdisjoint(case.unexpected_top_words))

    def test_theme_intrusion_review_cases_are_unique_and_non_empty(self) -> None:
        self.assertEqual(len(THEME_INTRUSION_REVIEW_CASES), 3)
        self.assertEqual(
            tuple(case.seed for case in THEME_INTRUSION_REVIEW_CASES),
            ("beach", "music", "snail"),
        )
        for case in THEME_INTRUSION_REVIEW_CASES:
            with self.subTest(seed=case.seed):
                self.assertTrue(case.note)
                self.assertEqual(len(case.expected_theme_words), 3)
                self.assertEqual(len(case.intruder_words), 3)
                self.assertTrue(set(case.expected_theme_words).isdisjoint(case.intruder_words))

    def test_normalize_seeds_filters_invalid_entries(self) -> None:
        self.assertEqual(normalize_seeds(("Snail", "bad!", "eases", "snail")), ("snail", "eases"))

    def test_validate_seed_words_rejects_words_missing_from_lexicon(self) -> None:
        with self.assertRaisesRegex(ValueError, "BEACH"):
            validate_seed_words(("snail", "beach"), ("snail", "slime"))

    def test_load_word_vectors_is_cached_for_unchanged_files(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "vectors.json"
            lexicon = ("beach", "ocean")
            _write_vector_table(
                path,
                lexicon,
                {
                    "beach": [4, 0, 0, 0],
                    "ocean": [3, 1, 0, 0],
                },
            )

            first = load_word_vectors(str(path))
            second = load_word_vectors(str(path))

        self.assertIs(first, second)
        self.assertEqual(first.lexicon_hash, lexicon_hash(lexicon))

    def test_score_word_for_seed_uses_cosine_similarity(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "vectors.json"
            lexicon = ("beach", "ocean", "music")
            _write_vector_table(
                path,
                lexicon,
                {
                    "beach": [4, 0, 0, 0],
                    "ocean": [3, 1, 0, 0],
                    "music": [0, 0, 4, 0],
                },
            )

            vectors = load_word_vectors(str(path))

        self.assertAlmostEqual(score_word_for_seed("ocean", ("beach",), vectors), 0.948683, places=5)
        self.assertAlmostEqual(score_word_for_seed("music", ("beach",), vectors), 0.0, places=5)

    def test_rank_lexicon_for_seed_orders_by_similarity_then_preferred_words(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "vectors.json"
            lexicon = ("beach", "ocean", "coast", "music")
            _write_vector_table(
                path,
                lexicon,
                {
                    "beach": [4, 0, 0, 0],
                    "ocean": [3, 1, 0, 0],
                    "coast": [2, 2, 0, 0],
                    "music": [0, 0, 4, 0],
                },
            )

            vectors = load_word_vectors(str(path))
            ranked = rank_lexicon_for_seed(("beach",), lexicon, vectors, preferred_words=("coast",))

        self.assertEqual(ranked, ("beach", "coast", "ocean", "music"))

    def test_rank_lexicon_for_seed_rejects_mismatched_lexicon_tables(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "vectors.json"
            _write_vector_table(
                path,
                ("beach", "ocean"),
                {
                    "beach": [4, 0, 0, 0],
                    "ocean": [3, 1, 0, 0],
                },
            )

            vectors = load_word_vectors(str(path))

        with self.assertRaisesRegex(ValueError, "missing lexicon entries: MUSIC"):
            rank_lexicon_for_seed(("beach",), ("beach", "music"), vectors)

    def test_rank_lexicon_for_seed_rejects_unknown_similarity_metric(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "vectors.json"
            lexicon = ("beach", "ocean")
            _write_vector_table(
                path,
                lexicon,
                {
                    "beach": [4, 0, 0, 0],
                    "ocean": [3, 1, 0, 0],
                },
            )

            vectors = load_word_vectors(str(path))

        with self.assertRaisesRegex(ValueError, "unsupported similarity metric"):
            rank_lexicon_for_seed(
                ("beach",),
                lexicon,
                vectors,
                similarity_metric="overlap",  # type: ignore[arg-type]
            )

    def test_rank_overlap_scores_reward_shared_neighbor_structure(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "vectors.json"
            lexicon = (
                "beach",
                "ocean",
                "waves",
                "wharf",
                "music",
                "piano",
                "choir",
                "tempo",
                "snail",
                "shell",
                "slime",
                "trail",
            )
            _write_vector_table(
                path,
                lexicon,
                {
                    "beach": [10, 0, 0, 0],
                    "ocean": [10, 1, 0, 0],
                    "waves": [9, 2, 0, 0],
                    "wharf": [9, 1, 0, 0],
                    "music": [8, 0, 8, 0],
                    "piano": [8, 0, 9, 0],
                    "choir": [9, 0, 8, 0],
                    "tempo": [7, 0, 8, 0],
                    "snail": [0, 10, 0, 0],
                    "shell": [0, 9, 1, 0],
                    "slime": [0, 8, 2, 0],
                    "trail": [0, 7, 3, 0],
                },
            )

            vectors = load_word_vectors(str(path))
            overlap_scores = rank_overlap_relevance_scores(
                lexicon,
                ("beach",),
                lexicon,
                vectors,
                neighbor_count=4,
            )
            ranked = rank_lexicon_for_seed(
                ("beach",),
                lexicon,
                vectors,
                similarity_metric="rank_overlap",
                neighbor_count=4,
            )

        self.assertGreater(overlap_scores["ocean"], overlap_scores["choir"])
        self.assertGreater(overlap_scores["waves"], overlap_scores["music"])
        self.assertEqual(ranked[0], "beach")
        self.assertEqual(set(ranked[1:4]), {"ocean", "waves", "wharf"})

    def test_compare_retrieval_metrics_reports_hits_and_intrusions(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "vectors.json"
            lexicon = (
                "beach",
                "ocean",
                "waves",
                "wharf",
                "music",
                "piano",
                "choir",
                "tempo",
                "snail",
                "shell",
                "slime",
                "trail",
            )
            _write_vector_table(
                path,
                lexicon,
                {
                    "beach": [10, 0, 0, 0],
                    "ocean": [10, 1, 0, 0],
                    "waves": [9, 2, 0, 0],
                    "wharf": [9, 1, 0, 0],
                    "music": [8, 0, 8, 0],
                    "piano": [8, 0, 9, 0],
                    "choir": [9, 0, 8, 0],
                    "tempo": [7, 0, 8, 0],
                    "snail": [0, 10, 0, 0],
                    "shell": [0, 9, 1, 0],
                    "slime": [0, 8, 2, 0],
                    "trail": [0, 7, 3, 0],
                },
            )

            vectors = load_word_vectors(str(path))
            comparison = compare_retrieval_metrics(
                THEME_RETRIEVAL_REVIEW_CASES[0],
                lexicon,
                vectors,
                top_n=3,
                neighbor_count=4,
            )

        self.assertEqual(comparison.seed, "beach")
        self.assertEqual(comparison.cosine.expected_hits, ("ocean", "waves", "wharf"))
        self.assertEqual(comparison.rank_overlap.expected_hits, ("ocean", "waves", "wharf"))
        self.assertEqual(comparison.cosine.unexpected_hits, ())
        self.assertEqual(comparison.rank_overlap.unexpected_hits, ())
        self.assertEqual(set(comparison.rank_overlap.top_words), {"ocean", "waves", "wharf"})

    def test_review_theme_retrieval_returns_one_report_per_case(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "vectors.json"
            lexicon = (
                "beach",
                "ocean",
                "waves",
                "wharf",
                "music",
                "piano",
                "choir",
                "tempo",
                "snail",
                "shell",
                "slime",
                "trail",
            )
            _write_vector_table(
                path,
                lexicon,
                {
                    "beach": [10, 0, 0, 0],
                    "ocean": [10, 1, 0, 0],
                    "waves": [9, 2, 0, 0],
                    "wharf": [9, 1, 0, 0],
                    "music": [8, 0, 8, 0],
                    "piano": [8, 0, 9, 0],
                    "choir": [9, 0, 8, 0],
                    "tempo": [7, 0, 8, 0],
                    "snail": [0, 10, 0, 0],
                    "shell": [0, 9, 1, 0],
                    "slime": [0, 8, 2, 0],
                    "trail": [0, 7, 3, 0],
                },
            )

            vectors = load_word_vectors(str(path))
            reports = review_theme_retrieval(
                THEME_RETRIEVAL_REVIEW_CASES,
                lexicon,
                vectors,
                top_n=3,
                neighbor_count=4,
            )

        self.assertEqual(tuple(report.seed for report in reports), ("beach", "music", "snail"))
        self.assertTrue(all(report.cosine.top_words for report in reports))
        self.assertTrue(all(report.rank_overlap.top_words for report in reports))

    def test_diversify_theme_words_skips_near_duplicate_answers(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "vectors.json"
            lexicon = ("beach", "coast", "music", "ocean", "waves")
            _write_vector_table(
                path,
                lexicon,
                {
                    "beach": [4, 0, 0, 0],
                    "coast": [6, 4, 0, 0],
                    "music": [0, 0, 4, 0],
                    "ocean": [4, 1, 0, 0],
                    "waves": [2, 3, 0, 0],
                },
            )

            vectors = load_word_vectors(str(path))
            diversified = diversify_theme_words(
                ("ocean", "coast", "waves", "music"),
                ("beach",),
                vectors,
                limit=4,
            )

        self.assertEqual(diversified, ("ocean", "waves"))

    def test_score_theme_subset_selects_theme_bearing_answers(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "vectors.json"
            lexicon = ("beach", "coast", "music", "ocean", "waves")
            _write_vector_table(
                path,
                lexicon,
                {
                    "beach": [4, 0, 0, 0],
                    "coast": [6, 4, 0, 0],
                    "music": [0, 0, 4, 0],
                    "ocean": [4, 1, 0, 0],
                    "waves": [2, 3, 0, 0],
                },
            )

            vectors = load_word_vectors(str(path))
            breakdown = score_theme_subset(lexicon, ("beach",), vectors)

        self.assertEqual(breakdown.selected_words, ("ocean", "waves"))
        self.assertGreater(breakdown.mean_relevance, 0.75)
        self.assertGreater(breakdown.weakest_link, 0.7)
        self.assertGreaterEqual(breakdown.diversity, 0.0)
        self.assertGreater(breakdown.total, 1.5)

    def test_score_theme_subset_returns_zero_without_related_answers(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "vectors.json"
            lexicon = ("beach", "music", "piano")
            _write_vector_table(
                path,
                lexicon,
                {
                    "beach": [4, 0, 0, 0],
                    "music": [0, 4, 0, 0],
                    "piano": [0, 3, 1, 0],
                },
            )

            vectors = load_word_vectors(str(path))
            breakdown = score_theme_subset(("beach", "music", "piano"), ("beach",), vectors)

        self.assertEqual(breakdown.selected_words, ())
        self.assertEqual(breakdown.total, 0.0)

    def test_compare_theme_intrusions_rejects_irrelevant_answers(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "vectors.json"
            lexicon = (
                "beach",
                "ocean",
                "waves",
                "wharf",
                "piano",
                "choir",
                "snail",
            )
            _write_vector_table(
                path,
                lexicon,
                {
                    "beach": [10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    "ocean": [7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    "waves": [7, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    "wharf": [7, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0],
                    "piano": [0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0],
                    "choir": [0, 0, 0, 0, 7, 7, 0, 0, 0, 0, 0, 0],
                    "snail": [0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0],
                },
            )

            vectors = load_word_vectors(str(path))
            comparison = compare_theme_intrusions(
                THEME_INTRUSION_REVIEW_CASES[0],
                lexicon,
                vectors,
            )

        self.assertEqual(comparison.seed, "beach")
        self.assertEqual(set(comparison.baseline_selected_words), {"ocean", "waves", "wharf"})
        self.assertEqual(len(comparison.baseline_selected_words), 3)
        self.assertEqual(len(comparison.trials), 3)
        self.assertEqual(comparison.pass_rate, 1.0)
        self.assertTrue(all(trial.passed for trial in comparison.trials))
        self.assertTrue(all(not trial.intruder_selected for trial in comparison.trials))

    def test_compare_theme_intrusions_flags_when_intruder_survives(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "vectors.json"
            lexicon = ("beach", "ocean", "waves", "wharf", "piano")
            _write_vector_table(
                path,
                lexicon,
                {
                    "beach": [10, 0, 0, 0, 0],
                    "ocean": [7, 7, 0, 0, 0],
                    "waves": [7, 0, 7, 0, 0],
                    "wharf": [1, 0, 0, 9, 0],
                    "piano": [7, 0, 0, 7, 0],
                },
            )

            vectors = load_word_vectors(str(path))
            case = ThemeIntrusionReviewCase(
                seed="beach",
                expected_theme_words=("ocean", "waves", "wharf"),
                intruder_words=("piano",),
                note="unit test case",
            )
            comparison = compare_theme_intrusions(
                case,
                lexicon,
                vectors,
            )

        failed_trial = next(trial for trial in comparison.trials if trial.intruder == "piano")
        self.assertFalse(failed_trial.passed)
        self.assertTrue(failed_trial.intruder_selected)
        self.assertIn("piano", failed_trial.selected_words)
        self.assertLess(comparison.pass_rate, 1.0)

    def test_review_theme_intrusions_returns_one_report_per_case(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "vectors.json"
            lexicon = (
                "beach",
                "ocean",
                "waves",
                "wharf",
                "music",
                "piano",
                "choir",
                "tempo",
                "snail",
                "shell",
                "slime",
                "trail",
            )
            _write_vector_table(
                path,
                lexicon,
                {
                    "beach": [10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    "ocean": [7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    "waves": [7, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    "wharf": [7, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0],
                    "music": [0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0],
                    "piano": [0, 0, 0, 0, 7, 7, 0, 0, 0, 0, 0, 0],
                    "choir": [0, 0, 0, 0, 7, 0, 7, 0, 0, 0, 0, 0],
                    "tempo": [0, 0, 0, 0, 7, 0, 0, 7, 0, 0, 0, 0],
                    "snail": [0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0],
                    "shell": [0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 0, 0],
                    "slime": [0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 7, 0],
                    "trail": [0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 7],
                },
            )

            vectors = load_word_vectors(str(path))
            reports = review_theme_intrusions(
                THEME_INTRUSION_REVIEW_CASES,
                lexicon,
                vectors,
            )

        self.assertEqual(tuple(report.seed for report in reports), ("beach", "music", "snail"))
        self.assertTrue(all(report.trials for report in reports))
        self.assertTrue(all(report.pass_rate == 1.0 for report in reports))

    def test_rank_theme_candidates_is_deterministic(self) -> None:
        ranked = rank_theme_candidates(("snail",), ("shell", "snail", "slime"))

        self.assertEqual(ranked, ("snail", "slime", "shell"))

    def test_build_candidate_pool_puts_theme_words_first(self) -> None:
        pool = build_candidate_pool(
            ("snail",),
            ("snail", "slime"),
            ("snail", "slime", "eases", "abase"),
            allow_neutral_fill=True,
        )

        self.assertEqual(pool, ("snail", "slime", "eases", "abase"))

    def test_build_candidate_pool_prioritizes_preferred_fill_before_other_neutral_words(self) -> None:
        pool = build_candidate_pool(
            ("snail",),
            ("snail",),
            ("snail", "abase", "eases", "slime"),
            allow_neutral_fill=True,
            preferred_words=("eases",),
        )

        self.assertEqual(pool, ("snail", "eases", "abase", "slime"))


if __name__ == "__main__":
    unittest.main()
