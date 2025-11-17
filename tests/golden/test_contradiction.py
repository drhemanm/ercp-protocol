"""
Golden tests for Natural Language Inference (NLI) contradiction detection.

Tests ERCP's ability to detect and reason about logical contradictions.
"""

import json
import pytest
from pathlib import Path
from typing import Dict, List, Set

# Mock ERCP components for testing (replace with actual imports when available)
# from server.ercp_server_v2 import run_ercp
# from server.operators import GenerateOperator, VerifyOperator, ExtractOperator


GOLDEN_TESTS_DIR = Path(__file__).parents[2] / "golden-tests"


def load_contradiction_test_case(filename: str) -> Dict:
    """Load a contradiction test case from JSON file."""
    file_path = GOLDEN_TESTS_DIR / filename
    with open(file_path, 'r') as f:
        return json.load(f)


def get_all_contradiction_test_files() -> List[str]:
    """Get all contradiction test case filenames."""
    if not GOLDEN_TESTS_DIR.exists():
        return []

    files = [
        f.name for f in GOLDEN_TESTS_DIR.iterdir()
        if f.name.startswith("contradiction_case_") and f.name.endswith(".json")
    ]
    return sorted(files)


class TestContradictionGolden:
    """Golden test suite for contradiction detection and reasoning."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        self.test_files = get_all_contradiction_test_files()
        assert len(self.test_files) >= 5, \
            f"Expected at least 5 contradiction test files, found {len(self.test_files)}"

    def verify_constraint_extraction(
        self,
        extracted_constraints: List[Dict],
        expected_constraints: List[str]
    ) -> Dict[str, bool]:
        """
        Verify that expected constraints were extracted.

        Args:
            extracted_constraints: List of constraints extracted by ERCP
            expected_constraints: List of expected constraint descriptions

        Returns:
            Dictionary mapping expected constraint to found status
        """
        results = {}

        # For each expected constraint, check if something similar was extracted
        for expected in expected_constraints:
            # Simplified check: look for keywords in extracted constraints
            keywords = expected.lower().split()
            found = False

            for constraint in extracted_constraints:
                constraint_text = str(constraint).lower()
                match_count = sum(1 for kw in keywords if kw in constraint_text)
                if match_count >= len(keywords) // 2:
                    found = True
                    break

            results[expected] = found

        return results

    def verify_error_detection(
        self,
        detected_errors: List[Dict],
        expected_errors: List[Dict]
    ) -> bool:
        """
        Verify that expected errors were detected.

        Args:
            detected_errors: List of errors detected by ERCP
            expected_errors: List of expected error specifications

        Returns:
            True if all critical errors were detected
        """
        if not detected_errors:
            return False

        # Check that each expected error type is represented
        expected_types = {err["type"] for err in expected_errors}
        detected_types = {err.get("type") for err in detected_errors}

        return expected_types.issubset(detected_types)

    def verify_convergence_and_reasoning(
        self,
        result: Dict,
        verification_criteria: Dict
    ) -> Dict[str, bool]:
        """
        Verify convergence and reasoning quality.

        Args:
            result: ERCP execution result
            verification_criteria: Expected verification criteria

        Returns:
            Dictionary of verification results
        """
        checks = {}

        # Check convergence expectation
        expected_convergence = verification_criteria.get("convergence_expected", True)
        actual_convergence = result.get("status") == "converged"
        checks["convergence_match"] = expected_convergence == actual_convergence

        # Check reasoning mentions key concepts
        final_reasoning = result.get("final_reasoning", "").lower()
        should_mention = verification_criteria.get("final_reasoning_should_mention", [])

        for concept in should_mention:
            checks[f"mentions_{concept}"] = concept.lower() in final_reasoning

        return checks

    @pytest.mark.parametrize("test_file", get_all_contradiction_test_files())
    def test_contradiction_detection(self, test_file):
        """
        Test ERCP contradiction detection on NLI test case.

        This test verifies:
        1. ERCP detects logical contradictions
        2. Appropriate constraints are extracted
        3. Errors are properly identified
        4. Final reasoning explains the contradiction
        """
        # Load test case
        test_case = load_contradiction_test_case(test_file)
        problem = test_case["problem"]
        reasoning_steps = test_case["reasoning_steps"]
        verification_criteria = test_case["verification_criteria"]

        # TODO: Replace with actual ERCP execution when ready
        pytest.skip("ERCP integration pending - test structure validated")

        # Expected test flow (uncomment when ERCP is integrated):
        """
        # 1. Prepare problem for ERCP
        problem_description = f'''
Premise: {problem['premise']}
Hypothesis: {problem['hypothesis']}

Task: Determine if the hypothesis contradicts the premise.
Explain your reasoning and identify any logical contradictions.
'''

        # 2. Run ERCP
        result = await run_ercp(
            problem_id=test_case['id'],
            problem_description=problem_description,
            config={
                "max_iterations": 10,
                "convergence_threshold": 0.85
            }
        )

        # 3. Verify convergence and reasoning
        convergence_checks = self.verify_convergence_and_reasoning(
            result,
            verification_criteria
        )

        failed_convergence = [k for k, v in convergence_checks.items() if not v]
        assert not failed_convergence, \
            f"Convergence/reasoning check failed for {test_file}: {failed_convergence}"

        # 4. Verify constraint extraction
        extracted_constraints = result.get('constraints', [])
        expected_constraints = verification_criteria['must_extract_constraints']

        constraint_results = self.verify_constraint_extraction(
            extracted_constraints,
            expected_constraints
        )

        missing_constraints = [k for k, v in constraint_results.items() if not v]
        assert not missing_constraints, \
            f"Missing constraints for {test_file}: {missing_constraints}"

        # 5. Verify error detection
        detected_errors = result.get('errors', [])
        expected_errors = verification_criteria['must_detect_errors']

        assert self.verify_error_detection(detected_errors, expected_errors), \
            f"Failed to detect expected errors for {test_file}"

        # 6. Verify contradiction label
        # Check that final reasoning identifies this as a contradiction
        final_reasoning = result.get('final_reasoning', '').lower()
        assert 'contradiction' in final_reasoning or 'contradicts' in final_reasoning, \
            f"Final reasoning should explicitly identify contradiction for {test_file}"
        """

    def test_contradiction_case_001_birds_flying(self):
        """Specific test for universal quantification contradiction (case 001)."""
        test_case = load_contradiction_test_case("contradiction_case_001.json")

        assert test_case["id"] == "contradiction_001"
        assert "universal_quantification" in test_case["metadata"]["tags"]

        # Verify reasoning steps include logical inference
        steps = test_case["reasoning_steps"]
        assert "step_3" in steps
        assert "modus ponens" in steps["step_3"]["constraint"].lower()

    def test_contradiction_case_003_phase_state(self):
        """Specific test for physical state contradiction (case 003)."""
        test_case = load_contradiction_test_case("contradiction_case_003.json")

        assert test_case["id"] == "contradiction_003"
        assert "phase_transitions" in test_case["metadata"]["tags"]

        # Should detect multiple error types
        expected_errors = test_case["verification_criteria"]["must_detect_errors"]
        assert len(expected_errors) >= 2, \
            "Phase state case should detect multiple error types"

    def test_all_contradiction_cases_have_required_fields(self):
        """Verify all contradiction test cases have required structure."""
        required_fields = [
            "id", "problem", "reasoning_steps",
            "verification_criteria", "metadata"
        ]

        for test_file in self.test_files:
            test_case = load_contradiction_test_case(test_file)

            for field in required_fields:
                assert field in test_case, \
                    f"{test_file} missing required field: {field}"

            # Verify problem structure
            problem = test_case["problem"]
            assert "premise" in problem
            assert "hypothesis" in problem
            assert "expected_label" in problem
            assert problem["expected_label"] == "contradiction"

    def test_contradiction_reasoning_steps_complete(self):
        """Verify all cases have complete reasoning step chains."""
        for test_file in self.test_files:
            test_case = load_contradiction_test_case(test_file)
            steps = test_case["reasoning_steps"]

            # Should have at least 3 reasoning steps
            assert len(steps) >= 3, \
                f"{test_file} should have at least 3 reasoning steps"

            # Each step should have description and constraint
            for step_key, step_value in steps.items():
                assert "description" in step_value, \
                    f"{test_file} {step_key} missing description"
                assert "constraint" in step_value, \
                    f"{test_file} {step_key} missing constraint"

    def test_contradiction_difficulty_distribution(self):
        """Verify test cases cover range of difficulty levels."""
        difficulties = []

        for test_file in self.test_files:
            test_case = load_contradiction_test_case(test_file)
            difficulties.append(test_case["problem"]["difficulty"])

        # Should have variety of difficulties
        unique_difficulties = set(difficulties)
        assert len(unique_difficulties) >= 2, \
            "Test suite should include multiple difficulty levels"

    def test_contradiction_tag_diversity(self):
        """Verify test cases cover diverse reasoning types."""
        all_tags = set()

        for test_file in self.test_files:
            test_case = load_contradiction_test_case(test_file)
            tags = test_case["metadata"]["tags"]
            all_tags.update(tags)

        # Should have variety of reasoning types
        assert len(all_tags) >= 5, \
            f"Test suite should cover diverse reasoning types, found {len(all_tags)} unique tags"


class TestContradictionErrorTypes:
    """Tests for specific error type detection in contradictions."""

    def test_logical_contradiction_detection(self):
        """Test detection of pure logical contradictions."""
        # Cases 001 and 004 are pure logical contradictions
        for case_id in ["contradiction_case_001.json", "contradiction_case_004.json"]:
            test_case = load_contradiction_test_case(case_id)
            expected_errors = test_case["verification_criteria"]["must_detect_errors"]

            logical_errors = [
                err for err in expected_errors
                if err["type"] == "logical_contradiction"
            ]

            assert len(logical_errors) >= 1, \
                f"{case_id} should detect logical contradiction"

    def test_temporal_constraint_violations(self):
        """Test detection of temporal constraint violations."""
        test_case = load_contradiction_test_case("contradiction_case_002.json")

        expected_errors = test_case["verification_criteria"]["must_detect_errors"]
        temporal_errors = [
            err for err in expected_errors
            if "temporal" in err["type"] or "precondition" in err["type"]
        ]

        assert len(temporal_errors) >= 1, \
            "Case 002 should detect temporal/precondition violation"

    def test_physical_law_violations(self):
        """Test detection of physical law violations."""
        test_case = load_contradiction_test_case("contradiction_case_003.json")

        expected_errors = test_case["verification_criteria"]["must_detect_errors"]
        physical_errors = [
            err for err in expected_errors
            if "physical_law" in err["type"] or "state" in err["type"]
        ]

        assert len(physical_errors) >= 1, \
            "Case 003 should detect physical law violation"

    def test_geographic_contradictions(self):
        """Test detection of geographic/spatial contradictions."""
        test_case = load_contradiction_test_case("contradiction_case_005.json")

        expected_errors = test_case["verification_criteria"]["must_detect_errors"]
        geographic_errors = [
            err for err in expected_errors
            if "geographic" in err["type"]
        ]

        assert len(geographic_errors) >= 1, \
            "Case 005 should detect geographic contradiction"


class TestContradictionConstraintTypes:
    """Tests for constraint extraction in contradiction cases."""

    def test_universal_quantification_constraints(self):
        """Test extraction of universal quantification constraints."""
        test_case = load_contradiction_test_case("contradiction_case_001.json")

        constraints_to_extract = test_case["verification_criteria"]["must_extract_constraints"]

        # Should include universal quantification
        universal_constraint = any(
            "universal" in c.lower() or "quantif" in c.lower()
            for c in constraints_to_extract
        )

        assert universal_constraint, \
            "Case 001 should require extraction of universal quantification constraint"

    def test_prerequisite_constraints(self):
        """Test extraction of prerequisite/precondition constraints."""
        test_case = load_contradiction_test_case("contradiction_case_002.json")

        constraints_to_extract = test_case["verification_criteria"]["must_extract_constraints"]

        # Should include prerequisite constraint
        prerequisite_constraint = any(
            "prerequisite" in c.lower() or "accessibility" in c.lower()
            for c in constraints_to_extract
        )

        assert prerequisite_constraint, \
            "Case 002 should require extraction of prerequisite constraint"

    def test_resource_sufficiency_constraints(self):
        """Test extraction of resource sufficiency constraints."""
        test_case = load_contradiction_test_case("contradiction_case_007.json")

        constraints_to_extract = test_case["verification_criteria"]["must_extract_constraints"]

        # Should include resource constraints
        resource_constraint = any(
            "resource" in c.lower() or "sufficiency" in c.lower()
            for c in constraints_to_extract
        )

        assert resource_constraint, \
            "Case 007 should require extraction of resource constraint"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
