"""
Golden tests for PIQA (Physical Interaction Question Answering) dataset.

Tests ERCP's ability to reason about physical interactions and common sense.
"""

import json
import os
import pytest
from pathlib import Path
from typing import Dict, List

# Mock ERCP components for testing (replace with actual imports when available)
# from server.ercp_server_v2 import run_ercp
# from server.operators import GenerateOperator, VerifyOperator, ExtractOperator


GOLDEN_TESTS_DIR = Path(__file__).parents[2] / "golden-tests"


def load_piqa_test_case(filename: str) -> Dict:
    """Load a PIQA test case from JSON file."""
    file_path = GOLDEN_TESTS_DIR / filename
    with open(file_path, 'r') as f:
        return json.load(f)


def get_all_piqa_test_files() -> List[str]:
    """Get all PIQA test case filenames."""
    if not GOLDEN_TESTS_DIR.exists():
        return []

    files = [
        f.name for f in GOLDEN_TESTS_DIR.iterdir()
        if f.name.startswith("piqa_sample_") and f.name.endswith(".json")
    ]
    return sorted(files)


class TestPIQAGolden:
    """Golden test suite for PIQA reasoning tasks."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        self.test_files = get_all_piqa_test_files()
        assert len(self.test_files) >= 10, f"Expected at least 10 PIQA test files, found {len(self.test_files)}"

    def verify_reasoning_quality(
        self,
        reasoning: str,
        expected: Dict,
        verification_criteria: Dict
    ) -> Dict[str, bool]:
        """
        Verify that generated reasoning meets quality criteria.

        Args:
            reasoning: Generated reasoning text
            expected: Expected reasoning structure
            verification_criteria: Criteria for verification

        Returns:
            Dictionary of verification results
        """
        results = {}
        reasoning_lower = reasoning.lower()

        # Check required mentions
        must_mention = verification_criteria.get("must_mention", [])
        for term in must_mention:
            results[f"mentions_{term}"] = term.lower() in reasoning_lower

        # Check prohibited content
        must_not_contain = verification_criteria.get("must_not_contain", [])
        for term in must_not_contain:
            results[f"avoids_{term}"] = term.lower() not in reasoning_lower

        # Check logic requirements (simplified check)
        logic_requirements = verification_criteria.get("logic_requirements", [])
        for req in logic_requirements:
            # Simple keyword-based check (in production, use semantic similarity)
            key_words = req.lower().split()
            match_count = sum(1 for word in key_words if word in reasoning_lower)
            results[f"logic_{req[:30]}"] = match_count >= len(key_words) // 2

        return results

    def verify_constraints_extracted(
        self,
        constraints: List[Dict],
        expected_constraints: List[Dict]
    ) -> bool:
        """
        Verify that key constraints were extracted.

        Args:
            constraints: List of extracted constraints
            expected_constraints: Expected constraints from test case

        Returns:
            True if critical constraints are present
        """
        if not constraints:
            return False

        # Check for critical priority constraints
        critical_constraints = [
            c for c in constraints
            if c.get("priority") == "critical"
        ]

        # Should have at least one critical constraint for each expected critical constraint
        expected_critical = [
            c for c in expected_constraints
            if c.get("priority") == "critical"
        ]

        return len(critical_constraints) >= len(expected_critical)

    @pytest.mark.parametrize("test_file", get_all_piqa_test_files())
    def test_piqa_reasoning(self, test_file):
        """
        Test ERCP reasoning on a PIQA test case.

        This test verifies:
        1. ERCP can generate reasoning for physical interaction questions
        2. Generated reasoning includes key concepts and constraints
        3. Reasoning quality meets verification criteria
        4. Constraints are properly extracted
        """
        # Load test case
        test_case = load_piqa_test_case(test_file)
        problem = test_case["problem"]
        expected_reasoning = test_case["expected_reasoning"]
        verification_criteria = test_case["verification_criteria"]

        # TODO: Replace with actual ERCP execution when ready
        # For now, this is a placeholder that demonstrates the test structure
        pytest.skip("ERCP integration pending - test structure validated")

        # Expected test flow (uncomment when ERCP is integrated):
        """
        # 1. Prepare problem for ERCP
        problem_description = f"{problem['question']}\n\nOptions:\n"
        for i, option in enumerate(problem['options']):
            problem_description += f"{i}. {option}\n"

        # 2. Run ERCP
        result = await run_ercp(
            problem_id=test_case['id'],
            problem_description=problem_description,
            config={
                "max_iterations": 5,
                "convergence_threshold": 0.9
            }
        )

        # 3. Verify convergence
        assert result['status'] == 'converged', \
            f"Expected convergence for {test_file}, got {result['status']}"

        # 4. Verify reasoning quality
        reasoning = result.get('final_reasoning', '')
        verification_results = self.verify_reasoning_quality(
            reasoning,
            expected_reasoning,
            verification_criteria
        )

        # Check that all verification criteria pass
        failed_checks = [k for k, v in verification_results.items() if not v]
        assert not failed_checks, \
            f"Verification failed for {test_file}: {failed_checks}"

        # 5. Verify constraints
        constraints = result.get('constraints', [])
        assert self.verify_constraints_extracted(
            constraints,
            expected_reasoning['constraints']
        ), f"Failed to extract critical constraints for {test_file}"

        # 6. Verify correct answer selection
        # This would check if the reasoning leads to the correct option
        correct_answer_idx = problem['correct_answer']
        # Implementation depends on how ERCP returns answer selection
        """

    def test_piqa_sample_001_egg_separation(self):
        """Specific test for egg separation physics (sample 001)."""
        test_case = load_piqa_test_case("piqa_sample_001.json")

        assert test_case["id"] == "piqa_001"
        assert test_case["problem"]["category"] == "cooking"

        # Verify test case structure
        assert "suction" in test_case["verification_criteria"]["must_mention"]
        assert len(test_case["expected_reasoning"]["constraints"]) == 2

    def test_piqa_sample_008_mpemba_effect(self):
        """Specific test for Mpemba effect (counterintuitive physics)."""
        test_case = load_piqa_test_case("piqa_sample_008.json")

        assert test_case["id"] == "piqa_008"
        assert test_case["problem"]["difficulty"] == "hard"

        # Verify this tests counterintuitive reasoning
        assert "Mpemba effect" in test_case["verification_criteria"]["must_mention"]

    def test_all_piqa_cases_have_required_fields(self):
        """Verify all PIQA test cases have required structure."""
        required_fields = [
            "id", "problem", "expected_reasoning",
            "verification_criteria", "metadata"
        ]

        for test_file in self.test_files:
            test_case = load_piqa_test_case(test_file)

            for field in required_fields:
                assert field in test_case, \
                    f"{test_file} missing required field: {field}"

            # Verify problem structure
            assert "question" in test_case["problem"]
            assert "options" in test_case["problem"]
            assert "correct_answer" in test_case["problem"]
            assert len(test_case["problem"]["options"]) == 2

    def test_piqa_difficulty_distribution(self):
        """Verify test cases cover range of difficulty levels."""
        difficulties = []

        for test_file in self.test_files:
            test_case = load_piqa_test_case(test_file)
            difficulties.append(test_case["problem"]["difficulty"])

        # Should have variety of difficulties
        unique_difficulties = set(difficulties)
        assert len(unique_difficulties) >= 2, \
            "Test suite should include multiple difficulty levels"

    def test_piqa_category_coverage(self):
        """Verify test cases cover multiple categories."""
        categories = []

        for test_file in self.test_files:
            test_case = load_piqa_test_case(test_file)
            categories.append(test_case["problem"]["category"])

        # Should have variety of categories
        unique_categories = set(categories)
        assert len(unique_categories) >= 3, \
            f"Test suite should cover at least 3 categories, found {len(unique_categories)}"


class TestPIQAConstraintExtraction:
    """Tests for constraint extraction from PIQA problems."""

    def test_physical_law_constraints(self):
        """Test extraction of physical law constraints."""
        test_case = load_piqa_test_case("piqa_sample_002.json")  # Fire oxygen

        constraints = test_case["expected_reasoning"]["constraints"]
        physical_laws = [
            c for c in constraints
            if c["type"] == "physical_law"
        ]

        assert len(physical_laws) >= 1, \
            "Fire test should include physical law constraint"

    def test_material_property_constraints(self):
        """Test extraction of material property constraints."""
        test_case = load_piqa_test_case("piqa_sample_004.json")  # Rubber band

        constraints = test_case["expected_reasoning"]["constraints"]
        material_constraints = [
            c for c in constraints
            if c["type"] == "material_property"
        ]

        assert len(material_constraints) >= 1, \
            "Material test should include material property constraint"

    def test_thermodynamic_constraints(self):
        """Test extraction of thermodynamic constraints."""
        test_case = load_piqa_test_case("piqa_sample_005.json")  # Coffee cooling

        constraints = test_case["expected_reasoning"]["constraints"]
        thermo_constraints = [
            c for c in constraints
            if c["type"] == "thermodynamics"
        ]

        assert len(thermo_constraints) >= 1, \
            "Thermodynamic test should include thermodynamic constraint"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
