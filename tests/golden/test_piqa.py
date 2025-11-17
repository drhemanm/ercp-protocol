"""
Golden tests for PIQA (Physical Interaction QA) problems.

These tests validate that the ERCP system produces high-quality reasoning
for common-sense physical reasoning tasks.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from server.operators.generate import GenerateOperator
from server.operators.verify import VerifyOperator
from server.operators.extract import ExtractOperator
from server.operators.stabilize import StabilizeOperator


@pytest.fixture
def golden_tests_dir():
    """Path to golden test files."""
    return Path(__file__).parent.parent.parent / "golden-tests"


@pytest.fixture
def piqa_test_cases(golden_tests_dir):
    """Load all PIQA test cases from golden-tests directory."""
    test_cases = []
    for file_path in sorted(golden_tests_dir.glob("piqa_sample_*.json")):
        with open(file_path, "r") as f:
            test_cases.append(json.load(f))
    return test_cases


class TestPIQAGolden:
    """Golden tests for PIQA problems."""

    @pytest.mark.asyncio
    async def test_piqa_001_rust_removal(
        self,
        golden_tests_dir,
        mock_generate_model,
        mock_nli_model,
        mock_embedding_model,
        test_db: AsyncSession
    ):
        """Test PIQA-001: Rust removal with citric acid."""
        # Load test case
        with open(golden_tests_dir / "piqa_sample_01.json", "r") as f:
            test_case = json.load(f)

        # Run ERCP loop
        result = await self._run_ercp_on_problem(
            test_case["description"],
            max_iterations=3
        )

        # Validate expected reasoning keywords are present
        reasoning_lower = result["final_reasoning"].lower()
        for keyword in test_case["expected_reasoning_contains"]:
            assert keyword.lower() in reasoning_lower, \
                f"Expected keyword '{keyword}' not found in reasoning"

        # Validate answer is mentioned (A or B)
        expected = test_case["expected_answer"]
        assert expected in result["final_reasoning"], \
            f"Expected answer '{expected}' not found in reasoning"

        # Validate convergence
        assert result["status"] in ["converged", "max_iterations"], \
            "ERCP should converge or reach max iterations"

    @pytest.mark.asyncio
    async def test_piqa_002_ice_melting(
        self,
        golden_tests_dir,
        mock_generate_model,
        mock_nli_model,
        mock_embedding_model,
        test_db: AsyncSession
    ):
        """Test PIQA-002: Ice melting with salt."""
        with open(golden_tests_dir / "piqa_sample_02.json", "r") as f:
            test_case = json.load(f)

        result = await self._run_ercp_on_problem(
            test_case["description"],
            max_iterations=3
        )

        reasoning_lower = result["final_reasoning"].lower()
        for keyword in test_case["expected_reasoning_contains"]:
            assert keyword.lower() in reasoning_lower, \
                f"Expected keyword '{keyword}' not found in reasoning"

        expected = test_case["expected_answer"]
        assert expected in result["final_reasoning"], \
            f"Expected answer '{expected}' not found in reasoning"

    @pytest.mark.asyncio
    async def test_piqa_003_water_boiling_safety(
        self,
        golden_tests_dir,
        mock_generate_model,
        mock_nli_model,
        mock_embedding_model,
        test_db: AsyncSession
    ):
        """Test PIQA-003: Testing water boiling (safety)."""
        with open(golden_tests_dir / "piqa_sample_03.json", "r") as f:
            test_case = json.load(f)

        result = await self._run_ercp_on_problem(
            test_case["description"],
            max_iterations=3
        )

        reasoning_lower = result["final_reasoning"].lower()
        for keyword in test_case["expected_reasoning_contains"]:
            assert keyword.lower() in reasoning_lower, \
                f"Expected keyword '{keyword}' not found in reasoning"

        # Validate safety priority
        assert "safe" in reasoning_lower, "Safety should be mentioned"

        expected = test_case["expected_answer"]
        assert expected in result["final_reasoning"], \
            f"Expected answer '{expected}' not found in reasoning"

    @pytest.mark.asyncio
    async def test_all_piqa_cases_converge(
        self,
        piqa_test_cases,
        mock_generate_model,
        mock_nli_model,
        mock_embedding_model,
        test_db: AsyncSession
    ):
        """Test that all PIQA cases converge within reasonable iterations."""
        for test_case in piqa_test_cases:
            result = await self._run_ercp_on_problem(
                test_case["description"],
                max_iterations=5
            )

            # All should converge or reach max iterations
            assert result["status"] in ["converged", "max_iterations"], \
                f"Problem {test_case['problem_id']} failed with status {result['status']}"

            # All should produce non-empty reasoning
            assert len(result["final_reasoning"]) > 50, \
                f"Problem {test_case['problem_id']} produced too short reasoning"

    @pytest.mark.asyncio
    async def test_piqa_constraint_satisfaction(
        self,
        piqa_test_cases,
        mock_generate_model,
        mock_nli_model,
        mock_embedding_model,
        test_db: AsyncSession
    ):
        """Test that PIQA solutions satisfy expected constraints."""
        for test_case in piqa_test_cases:
            if "expected_constraints" not in test_case:
                continue

            result = await self._run_ercp_on_problem(
                test_case["description"],
                max_iterations=5
            )

            reasoning = result["final_reasoning"]

            # Check each constraint
            for constraint_desc in test_case["expected_constraints"]:
                if "Answer must be A or B" in constraint_desc:
                    assert "A)" in reasoning or "B)" in reasoning, \
                        f"Problem {test_case['problem_id']} should contain A or B"

                if "Must explain chemical mechanism" in constraint_desc:
                    # Should contain some explanation beyond just the answer
                    assert len(reasoning.split()) > 20, \
                        f"Problem {test_case['problem_id']} should explain mechanism"

                if "Must prioritize safety" in constraint_desc:
                    assert "safe" in reasoning.lower(), \
                        f"Problem {test_case['problem_id']} should mention safety"

    # Helper method
    async def _run_ercp_on_problem(
        self,
        problem_description: str,
        max_iterations: int = 5
    ) -> Dict[str, Any]:
        """
        Run a full ERCP loop on a problem.

        This mimics the /run endpoint behavior but in a test context.
        """
        gen_op = GenerateOperator()
        ver_op = VerifyOperator()
        ext_op = ExtractOperator()
        stab_op = StabilizeOperator()

        constraints_accum = []
        reasoning_prev = None
        status = "max_iterations"

        for iteration in range(max_iterations):
            # Generate
            gen_result = gen_op.execute(
                problem_description=problem_description,
                constraints=constraints_accum,
                max_tokens=512,
                temperature=0.0,
                seed=42
            )
            reasoning_curr = gen_result["reasoning_text"]

            # Verify
            ver_result = ver_op.execute(
                reasoning_text=reasoning_curr,
                reasoning_id=gen_result["reasoning_id"],
                constraints=constraints_accum,
                run_fact_check=False
            )
            errors = ver_result["errors"]

            # Extract (if errors found)
            if errors:
                ext_result = ext_op.execute(
                    errors=errors,
                    reasoning_text=reasoning_curr,
                    problem_description=problem_description,
                    threshold=0.7
                )
                constraints_accum.extend(ext_result["constraints"])

            # Stabilize
            stab_result = stab_op.execute(
                reasoning_curr=reasoning_curr,
                reasoning_prev=reasoning_prev,
                threshold=0.95,
                errors=errors
            )

            if stab_result["stable"]:
                status = "converged"
                break

            reasoning_prev = reasoning_curr

        return {
            "status": status,
            "final_reasoning": reasoning_curr,
            "iterations": iteration + 1,
            "final_errors": errors,
            "constraints_accumulated": len(constraints_accum)
        }
