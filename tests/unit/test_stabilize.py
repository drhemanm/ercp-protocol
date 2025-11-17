"""
Unit tests for Stabilize operator.
"""

import pytest
from server.operators.stabilize import StabilizeOperator


@pytest.fixture
def stabilizer():
    """Create StabilizeOperator instance."""
    return StabilizeOperator()


def test_stabilize_first_iteration(stabilizer, sample_reasoning):
    """Test stability check on first iteration."""
    result = stabilizer.execute(
        prev_reasoning=None,
        curr_reasoning=sample_reasoning,
        threshold=0.95,
        errors=[],
    )

    assert result["stable"] is False
    assert result["score"] == 0.0
    assert result["reason"] == "first_iteration"


def test_stabilize_high_similarity(stabilizer, sample_reasoning):
    """Test stability with high similarity."""
    # Slightly different but semantically similar text
    similar_text = (
        "Water boils at lower temperatures when at higher altitudes because of reduced atmospheric pressure. "
        "At sea level, the boiling point is around 100°C. "
        "Higher elevations have lower pressure, allowing vapor formation at lower temperatures."
    )

    result = stabilizer.execute(
        prev_reasoning=sample_reasoning,
        curr_reasoning=similar_text,
        threshold=0.85,
        errors=[],
    )

    # Should detect high similarity
    assert result["score"] >= 0.80  # Reasonable threshold
    assert result["stable"] is True
    assert result["error_count"] == 0


def test_stabilize_low_similarity(stabilizer, sample_reasoning):
    """Test instability with low similarity."""
    different_text = (
        "The chemical composition of water is H2O. "
        "It consists of two hydrogen atoms and one oxygen atom."
    )

    result = stabilizer.execute(
        prev_reasoning=sample_reasoning,
        curr_reasoning=different_text,
        threshold=0.90,
        errors=[],
    )

    # Should detect low similarity
    assert result["stable"] is False
    assert result["reason"] == "low_similarity"


def test_stabilize_with_errors(stabilizer, sample_reasoning):
    """Test instability when errors are present."""
    # Even with identical text, errors prevent stability
    mock_errors = [{"error_id": "e1", "type": "test_error"}]

    result = stabilizer.execute(
        prev_reasoning=sample_reasoning,
        curr_reasoning=sample_reasoning,
        threshold=0.95,
        errors=mock_errors,
    )

    assert result["stable"] is False
    assert result["reason"] == "has_errors"
    assert result["error_count"] == 1
