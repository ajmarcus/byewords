from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
import unittest
from unittest.mock import patch

from byewords.theme import (
    ThemeIntrusionComparison,
    ThemeIntrusionTrialReport,
    ThemeRetrievalComparison,
    ThemeRetrievalMetricReport,
)
from byewords.theme_index_builder import (
    DEFAULT_EMBEDDING_LICENSE,
    DEFAULT_EMBEDDING_SOURCE,
    DEFAULT_EMBEDDING_URL,
    build_word_vector_payload,
    main,
    parse_args,
)


class _FakeEncoder:
    def encode(
        self,
        words: list[str],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> list[list[float]]:
        del batch_size
        self.convert_to_numpy = convert_to_numpy
        self.normalize_embeddings = normalize_embeddings
        self.show_progress_bar = show_progress_bar
        return [
            [1.0, 0.0, 0.0] if word == "beach" else
            [0.6, 0.8, 0.0] if word == "ocean" else
            [0.0, 1.0, 0.0]
            for word in words
        ]


class TestThemeIndexBuilder(unittest.TestCase):
    def test_parse_args_preserves_legacy_vector_invocation(self) -> None:
        args = parse_args(["--output", "custom.json", "--batch-size", "16", "--device", "cpu"])

        self.assertEqual(args.command, "vectors")
        self.assertEqual(args.output, Path("custom.json"))
        self.assertEqual(args.batch_size, 16)
        self.assertEqual(args.device, "cpu")

    def test_build_word_vector_payload_extracts_bge_vectors(self) -> None:
        encoder = _FakeEncoder()
        with patch("byewords.theme_index_builder._load_sentence_transformer", return_value=encoder):
            payload = build_word_vector_payload(("beach", "ocean", "music"), batch_size=8, device="cpu")

        self.assertEqual(payload["source"], DEFAULT_EMBEDDING_SOURCE)
        self.assertEqual(payload["dimensions"], 3)
        self.assertEqual(payload["source_url"], DEFAULT_EMBEDDING_URL)
        self.assertEqual(payload["license"], DEFAULT_EMBEDDING_LICENSE)
        self.assertEqual(payload["model_name"], "BAAI/bge-large-en-v1.5")
        attribution = payload["attribution"]
        vectors = payload["vectors"]
        self.assertIsInstance(attribution, str)
        self.assertIsInstance(vectors, dict)
        typed_vectors = cast(dict[str, list[int]], vectors)
        self.assertIn("bge-large-en-v1.5", cast(str, attribution))
        self.assertEqual(tuple(sorted(typed_vectors)), ("beach", "music", "ocean"))
        self.assertTrue(encoder.convert_to_numpy)
        self.assertTrue(encoder.normalize_embeddings)
        self.assertFalse(encoder.show_progress_bar)
        for vector in typed_vectors.values():
            self.assertEqual(len(vector), 3)
            self.assertTrue(any(component != 0 for component in vector))

    def test_build_word_vector_payload_rejects_dimension_mismatch(self) -> None:
        class BadEncoder:
            def encode(self, words: list[str], **_: object) -> list[list[float]]:
                return [[1.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.5]][:len(words)]

        with (
            patch("byewords.theme_index_builder._load_sentence_transformer", return_value=BadEncoder()),
            self.assertRaisesRegex(ValueError, "inconsistent vector dimensions"),
        ):
            build_word_vector_payload(("beach", "ocean", "music"))

    def test_parse_args_supports_cache_command(self) -> None:
        args = parse_args(["cache", "--candidates-per-seed", "3", "--top-clue-limit", "20"])

        self.assertEqual(args.command, "cache")
        self.assertEqual(args.candidates_per_seed, 3)
        self.assertEqual(args.top_clue_limit, 20)

    def test_main_cache_command_builds_offline_store(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "puzzles.json"
            stdout = StringIO()
            lexicon_words = ("beach", "ocean")
            clue_bank = {"beach": ("Sandy shore",)}
            sentinel_vectors = object()

            with (
                redirect_stdout(stdout),
                patch("byewords.theme_index_builder._load_bundled_inputs", return_value=(lexicon_words, clue_bank)),
                patch("byewords.theme_index_builder._load_validated_vectors", return_value=sentinel_vectors),
                patch(
                    "byewords.theme_index_builder.build_batch_puzzle_cache",
                    return_value=(output_path, 2, 1),
                ) as build_cache,
            ):
                exit_code = main(["cache", "--output", str(output_path), "--candidates-per-seed", "2"])

        self.assertEqual(exit_code, 0)
        build_cache.assert_called_once_with(
            lexicon_words,
            clue_bank,
            path=output_path,
            vectors=sentinel_vectors,
            candidates_per_seed=2,
            top_clue_limit=100,
        )
        self.assertIn("Cached 2 puzzles in", stdout.getvalue())

    def test_main_retrieval_review_json_prints_structured_report(self) -> None:
        stdout = StringIO()
        report = ThemeRetrievalComparison(
            seed="beach",
            cosine=ThemeRetrievalMetricReport(
                metric="cosine",
                top_words=("ocean", "waves"),
                expected_hits=("ocean",),
                unexpected_hits=(),
                expected_coverage=0.5,
                unexpected_intrusion_rate=0.0,
            ),
            rank_overlap=ThemeRetrievalMetricReport(
                metric="rank_overlap",
                top_words=("ocean", "wharf"),
                expected_hits=("ocean",),
                unexpected_hits=(),
                expected_coverage=0.5,
                unexpected_intrusion_rate=0.0,
            ),
        )

        with (
            redirect_stdout(stdout),
            patch("byewords.theme_index_builder._load_bundled_inputs", return_value=(("beach", "ocean"), {})),
            patch("byewords.theme_index_builder._load_validated_vectors", return_value=object()),
            patch("byewords.theme_index_builder.review_theme_retrieval", return_value=(report,)),
        ):
            exit_code = main(["retrieval-review", "--json"])

        self.assertEqual(exit_code, 0)
        output = stdout.getvalue()
        self.assertIn('"seed": "beach"', output)
        self.assertIn('"metric": "cosine"', output)

    def test_main_vectors_command_builds_vectors(self) -> None:
        stdout = StringIO()
        with (
            redirect_stdout(stdout),
            patch("byewords.theme_index_builder.write_word_vectors") as write_word_vectors,
        ):
            exit_code = main(["vectors", "--output", "custom.json", "--batch-size", "32", "--device", "cpu"])

        self.assertEqual(exit_code, 0)
        write_word_vectors.assert_called_once_with(Path("custom.json"), batch_size=32, device="cpu")
        self.assertIn("Wrote semantic vectors to custom.json", stdout.getvalue())

    def test_main_intrusion_review_prints_text_summary(self) -> None:
        stdout = StringIO()
        report = ThemeIntrusionComparison(
            seed="beach",
            expected_theme_words=("ocean", "waves", "wharf"),
            baseline_selected_words=("ocean", "waves", "wharf"),
            baseline_total=2.0,
            baseline_weakest_link=0.4,
            trials=(
                ThemeIntrusionTrialReport(
                    intruder="piano",
                    selected_words=("ocean", "waves", "wharf"),
                    intruder_selected=False,
                    total=1.8,
                    weakest_link=0.3,
                    total_delta=-0.2,
                    weakest_link_delta=-0.1,
                    passed=True,
                ),
            ),
            pass_rate=1.0,
        )

        with (
            redirect_stdout(stdout),
            patch("byewords.theme_index_builder._load_bundled_inputs", return_value=(("beach", "ocean"), {})),
            patch("byewords.theme_index_builder._load_validated_vectors", return_value=object()),
            patch("byewords.theme_index_builder.review_theme_intrusions", return_value=(report,)),
        ):
            exit_code = main(["intrusion-review"])

        self.assertEqual(exit_code, 0)
        self.assertIn("BEACH: pass_rate=1.00", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
