"""
Golden tests for contradiction detection.

These tests validate that the ERCP system correctly detects contradictions
in reasoning chains and extracts appropriate constraints.
"""

import json
from pathlib import Path
from typing import Dict, Any, List

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from server.operators.generate import GenerateOperator
from server.operators.verify import VerifyOperator
from server.operators.extract import ExtractOperator
from server.validators.nli_validator import NLIValidator
from server.validators.rule_validator import RuleValidator


@pytest.fixture
def golden_tests_dir():
    """Path to golden test files."""
    return Path(__file__).parent.parent.parent / "golden-tests"


@pytest.fixture
def contradiction_test_cases(golden_tests_dir):
    """Load all contradiction test cases from golden-tests directory."""
    test_cases = []
    for file_path in sorted(golden_tests_dir.glob("contradiction_case_*.json")):
        with open(file_path, "r") as f:
            test_cases.append(json.load(f))
    return test_cases


class TestContradictionDetection:
    """Golden tests for contradiction detection."""

    @pytest.mark.asyncio
    async def test_contradiction_001_numeric(
        self,
        golden_tests_dir,
        mock_nli_model,
        test_db: AsyncSession
    ):
        """Test contradiction detection for numeric inconsistency (boiling point)."""
        # Load test case
        with open(golden_tests_dir / "contradiction_case_01.json", "r") as f:
            test_case = json.load(f)

        # Run verify operator on the erroneous reasoning
        ver_op = VerifyOperator()
        result = ver_op.execute(
            reasoning_text=test_case["reasoning_with_error"],
            reasoning_id=test_case["problem_id"],
            constraints=None,
            run_fact_check=False
        )

        # Should detect errors
        assert result["has_errors"], "Should detect contradiction in reasoning"
        assert result["error_count"] > 0, "Should have at least one error"

        # Validate error details
        errors = result["errors"]
        self._validate_expected_errors(errors, test_case["expected_errors"])

    @pytest.mark.asyncio
    async def test_contradiction_002_factual(
        self,
        golden_tests_dir,
        mock_nli_model,
        test_db: AsyncSession
    ):
        """Test contradiction detection for factual inconsistency (ice density)."""
        with open(golden_tests_dir / "contradiction_case_02.json", "r") as f:
            test_case = json.load(f)

        ver_op = VerifyOperator()
        result = ver_op.execute(
            reasoning_text=test_case["reasoning_with_error"],
            reasoning_id=test_case["problem_id"],
            constraints=None,
            run_fact_check=False
        )

        assert result["has_errors"], "Should detect contradiction in reasoning"
        assert result["error_count"] > 0, "Should have at least one error"

        errors = result["errors"]
        self._validate_expected_errors(errors, test_case["expected_errors"])

    @pytest.mark.asyncio
    async def test_all_contradictions_detected(
        self,
        contradiction_test_cases,
        mock_nli_model,
        test_db: AsyncSession
    ):
        """Test that all contradiction cases are detected."""
        ver_op = VerifyOperator()

        for test_case in contradiction_test_cases:
            result = ver_op.execute(
                reasoning_text=test_case["reasoning_with_error"],
                reasoning_id=test_case["problem_id"],
                constraints=None,
                run_fact_check=False
            )

            assert result["has_errors"], \
                f"Case {test_case['problem_id']} should detect contradiction"
            assert result["error_count"] > 0, \
                f"Case {test_case['problem_id']} should have errors"

    @pytest.mark.asyncio
    async def test_contradiction_constraint_extraction(
        self,
        golden_tests_dir,
        mock_generate_model,
        mock_nli_model,
        test_db: AsyncSession
    ):
        """Test that contradictions lead to constraint extraction."""
        with open(golden_tests_dir / "contradiction_case_01.json", "r") as f:
            test_case = json.load(f)

        # First, detect the error
        ver_op = VerifyOperator()
        ver_result = ver_op.execute(
            reasoning_text=test_case["reasoning_with_error"],
            reasoning_id=test_case["problem_id"],
            constraints=None,
            run_fact_check=False
        )

        errors = ver_result["errors"]
        assert len(errors) > 0, "Should detect errors"

        # Then, extract constraints
        ext_op = ExtractOperator()
        ext_result = ext_op.execute(
            errors=errors,
            reasoning_text=test_case["reasoning_with_error"],
            problem_description=test_case["description"],
            threshold=0.7
        )

        # Should extract at least one constraint
        all_constraints = ext_result["constraints"] + ext_result["candidate_constraints"]
        assert len(all_constraints) > 0, "Should extract constraints from errors"

        # Constraints should have required fields
        for constraint in all_constraints:
            assert "nl_text" in constraint, "Constraint should have natural language text"
            assert "priority" in constraint, "Constraint should have priority"
            assert "confidence" in constraint, "Constraint should have confidence"

    @pytest.mark.asyncio
    async def test_nli_validator_confidence_scores(
        self,
        contradiction_test_cases,
        mock_nli_model
    ):
        """Test that NLI validator produces appropriate confidence scores."""
        nli_validator = NLIValidator()

        for test_case in contradiction_test_cases:
            reasoning = test_case["reasoning_with_error"]
            sentences = reasoning.split(". ")

            errors = nli_validator.validate(
                reasoning_text=reasoning,
                sentences=sentences,
                constraints=None
            )

            # Should detect errors
            assert len(errors) > 0, \
                f"Case {test_case['problem_id']} should detect errors"

            # Check confidence scores meet minimum threshold
            for expected_error in test_case["expected_errors"]:
                if "confidence_min" in expected_error:
                    min_confidence = expected_error["confidence_min"]
                    # At least one error should meet the confidence threshold
                    max_detected_confidence = max(e["confidence"] for e in errors)
                    assert max_detected_confidence >= min_confidence, \
                        f"Error confidence {max_detected_confidence} below minimum {min_confidence}"

    @pytest.mark.asyncio
    async def test_rule_validator_numeric_contradictions(self, test_db: AsyncSession):
        """Test that rule validator catches numeric contradictions."""
        rule_validator = RuleValidator()

        # Example with numeric contradiction
        reasoning = "The temperature is 100 degrees. This is very hot. The temperature is 50 degrees."
        sentences = reasoning.split(". ")

        errors = rule_validator.validate(
            reasoning_text=reasoning,
            sentences=sentences,
            constraints=None
        )

        # Should detect numeric contradiction
        numeric_errors = [e for e in errors if e["error_type"] == "numeric_contradiction"]
        assert len(numeric_errors) > 0, "Should detect numeric contradiction"

        # Error should have proper structure
        for error in numeric_errors:
            assert "confidence" in error
            assert error["confidence"] > 0.5
            assert "excerpt" in error

    @pytest.mark.asyncio
    async def test_ercp_loop_fixes_contradictions(
        self,
        golden_tests_dir,
        mock_generate_model,
        mock_nli_model,
        mock_embedding_model,
        test_db: AsyncSession
    ):
        """Test that ERCP loop can detect and fix contradictions through iterations."""
        with open(golden_tests_dir / "contradiction_case_01.json", "r") as f:
            test_case = json.load(f)

        # Simulate ERCP loop
        gen_op = GenerateOperator()
        ver_op = VerifyOperator()
        ext_op = ExtractOperator()

        problem_desc = test_case["description"]
        constraints_accum = []

        # First, verify the bad reasoning detects errors
        ver_result_bad = ver_op.execute(
            reasoning_text=test_case["reasoning_with_error"],
            reasoning_id="test-bad",
            constraints=None,
            run_fact_check=False
        )
        assert ver_result_bad["has_errors"], "Should detect errors in bad reasoning"

        # Extract constraints from those errors
        if ver_result_bad["errors"]:
            ext_result = ext_op.execute(
                errors=ver_result_bad["errors"],
                reasoning_text=test_case["reasoning_with_error"],
                problem_description=problem_desc,
                threshold=0.7
            )
            constraints_accum.extend(ext_result["constraints"])

        # Now generate new reasoning with those constraints
        gen_result = gen_op.execute(
            problem_description=problem_desc,
            constraints=constraints_accum,
            max_tokens=256,
            temperature=0.0,
            seed=42
        )

        # Verify the new reasoning
        ver_result_new = ver_op.execute(
            reasoning_text=gen_result["reasoning_text"],
            reasoning_id=gen_result["reasoning_id"],
            constraints=constraints_accum,
            run_fact_check=False
        )

        # New reasoning should have fewer errors (ideally none)
        # Note: In real implementation, this depends on the model's ability to follow constraints
        assert ver_result_new["error_count"] <= ver_result_bad["error_count"], \
            "New reasoning should have same or fewer errors after constraint extraction"

    # Helper methods
    def _validate_expected_errors(
        self,
        detected_errors: List[Dict[str, Any]],
        expected_errors: List[Dict[str, Any]]
    ):
        """Validate that detected errors match expected errors."""
        assert len(detected_errors) > 0, "Should detect at least one error"

        for expected in expected_errors:
            # Check error type
            error_type = expected["type"]
            matching_errors = [
                e for e in detected_errors
                if e.get("error_type") == error_type or
                   e.get("error_type") == f"{error_type}_error" or
                   error_type in e.get("error_type", "")
            ]

            assert len(matching_errors) > 0, \
                f"Should detect error of type '{error_type}'"

            # Check confidence threshold
            if "confidence_min" in expected:
                min_confidence = expected["confidence_min"]
                max_confidence = max(e["confidence"] for e in matching_errors)
                assert max_confidence >= min_confidence, \
                    f"Error confidence {max_confidence} below minimum {min_confidence}"

            # Check span if specified
            if "span" in expected:
                expected_span = expected["span"]
                # At least one error should have a span overlapping the expected span
                has_overlap = any(
                    self._spans_overlap(e.get("span", [0, 0]), expected_span)
                    for e in matching_errors
                )
                assert has_overlap, \
                    f"No error found with span overlapping {expected_span}"

    def _spans_overlap(self, span1: List[int], span2: List[int]) -> bool:
        """Check if two spans overlap."""
        return not (span1[1] < span2[0] or span2[1] < span1[0])
