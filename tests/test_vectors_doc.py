from pathlib import Path
import unittest


VECTORS_DOC = Path(__file__).resolve().parents[1] / "docs" / "vectors.md"


class TestVectorsDoc(unittest.TestCase):
    def test_vectors_doc_records_plan_and_status(self) -> None:
        text = VECTORS_DOC.read_text(encoding="utf-8")

        self.assertIn("# Vector Model Migration", text)
        self.assertIn("BAAI/bge-small-en-v1.5", text)
        self.assertIn("## Performance Bottleneck", text)
        self.assertIn("## Best Candidates", text)
        self.assertIn("## Recommendation", text)
        self.assertIn("_cosine_similarity()", text)
        self.assertIn("## Plan", text)
        self.assertIn("## Completed", text)
        self.assertIn("## Not Completed Yet", text)
        self.assertIn("uv run ruff check .", text)
        self.assertIn("uv run ty check", text)
        self.assertIn("uv run python -m unittest discover -s tests", text)
        self.assertIn("uv run byewords", text)


if __name__ == "__main__":
    unittest.main()
