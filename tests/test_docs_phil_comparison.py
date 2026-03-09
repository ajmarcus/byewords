from pathlib import Path
import unittest


DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"
PLAN_MD = DOCS_DIR / "plan.md"
IMPLEMENTATION_MD = DOCS_DIR / "implementation.md"


class TestPhilComparisonDocs(unittest.TestCase):
    def test_plan_records_why_phil_does_not_replace_the_core_search(self) -> None:
        plan = PLAN_MD.read_text(encoding="utf-8")

        self.assertIn("Comparison: `Phil`", plan)
        self.assertIn("WebAssembly build of the Glucose SAT solver", plan)
        self.assertIn("row-by-row prefix pruning is the simpler and better fit", plan)
        self.assertIn("Steal from Phil at the edges, not at the core", plan)

    def test_implementation_records_the_actionable_lessons_from_phil(self) -> None:
        implementation = IMPLEMENTATION_MD.read_text(encoding="utf-8")

        self.assertIn("Comparison with `Phil`", implementation)
        self.assertIn("forced letters", implementation)
        self.assertIn("SearchDiagnostics", implementation)
        self.assertIn("define a separate solver interface", implementation)


if __name__ == "__main__":
    unittest.main()
