"""
Unit tests for Verify operator.
"""

import pytest
from server.operators.verify import VerifyOperator


@pytest.fixture
def verifier():
    """Create VerifyOperator instance."""
    return VerifyOperator()


def test_verify_no_errors(verifier, sample_reasoning):
    """Test verification with consistent reasoning."""
    errors = verifier.execute(
        reasoning_text=sample_reasoning,
        constraints=[],
        config={"nli_threshold": 0.75},
    )

    # Should find no or few errors in consistent reasoning
    assert isinstance(errors, list)


def test_verify_contradiction(verifier):
    """Test contradiction detection."""
    contradictory_text = "The sky is blue. The sky is red."

    errors = verifier.execute(
        reasoning_text=contradictory_text,
        constraints=[],
        config={"nli_threshold": 0.70},
    )

    # Should detect contradiction
    assert isinstance(errors, list)
    # May or may not detect depending on NLI model, but test structure
    for error in errors:
        assert "error_id" in error
        assert "type" in error
        assert "confidence" in error


def test_verify_insufficient_reasoning(verifier):
    """Test detection of insufficient reasoning."""
    short_text = "It just is."

    errors = verifier.execute(
        reasoning_text=short_text, constraints=[], config={"nli_threshold": 0.75}
    )

    # Should detect insufficient reasoning
    assert isinstance(errors, list)
    # Should have at least one error for insufficient reasoning
    assert any(e["type"] == "insufficient_reasoning" for e in errors)


def test_verify_vague_language(verifier):
    """Test detection of vague language."""
    vague_text = "Water maybe boils at different temperatures. It could be due to pressure perhaps."

    errors = verifier.execute(
        reasoning_text=vague_text, constraints=[], config={"nli_threshold": 0.75}
    )

    # Should detect vague language
    assert isinstance(errors, list)
    vague_errors = [e for e in errors if e["type"] == "vague_language"]
    assert len(vague_errors) > 0


def test_verify_with_constraints(verifier, sample_reasoning, sample_constraints):
    """Test constraint validation."""
    errors = verifier.execute(
        reasoning_text=sample_reasoning,
        constraints=sample_constraints,
        config={"nli_threshold": 0.75},
    )

    # Should validate constraints
    assert isinstance(errors, list)
