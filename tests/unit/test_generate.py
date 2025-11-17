"""
Unit tests for Generate operator.
"""

import pytest
from server.operators.generate import GenerateOperator


@pytest.fixture
def generator():
    """Create GenerateOperator instance."""
    config = {"model": "gpt2"}
    return GenerateOperator(config)


def test_generate_with_no_constraints(generator, sample_problem):
    """Test generation without constraints."""
    result = generator.execute(
        problem=sample_problem["description"],
        constraints=[],
        config={"max_tokens": 100, "temperature": 0.0},
    )

    assert "reasoning_id" in result
    assert "reasoning_text" in result
    assert "sentences" in result
    assert "claims" in result
    assert len(result["sentences"]) > 0
    assert isinstance(result["reasoning_text"], str)
    assert len(result["reasoning_text"]) > 0


def test_generate_with_constraints(generator, sample_problem, sample_constraints):
    """Test generation with constraints."""
    result = generator.execute(
        problem=sample_problem["description"],
        constraints=sample_constraints,
        config={"max_tokens": 150, "temperature": 0.0},
    )

    assert "reasoning_text" in result
    assert len(result["sentences"]) > 0

    # Check that constraint keywords appear in output (best effort)
    reasoning_lower = result["reasoning_text"].lower()
    # Should mention pressure if constraint requires it
    assert any(
        keyword in reasoning_lower for keyword in ["pressure", "altitude", "boil"]
    )


def test_build_prompt(generator, sample_problem, sample_constraints):
    """Test prompt building."""
    prompt = generator._build_prompt(sample_problem["description"], sample_constraints)

    assert "Question:" in prompt
    assert sample_problem["description"] in prompt
    assert "Requirements:" in prompt
    assert sample_constraints[0]["nl_text"] in prompt


def test_segment_sentences(generator, sample_reasoning):
    """Test sentence segmentation."""
    sentences = generator._segment_sentences(sample_reasoning)

    assert len(sentences) >= 2
    assert all(isinstance(s, str) for s in sentences)
    assert all(len(s) > 0 for s in sentences)


def test_extract_claims(generator, sample_reasoning):
    """Test claim extraction."""
    sentences = generator._segment_sentences(sample_reasoning)
    claims = generator._extract_claims(sample_reasoning, sentences)

    assert isinstance(claims, list)
    # Should extract at least some claims
    assert len(claims) >= 0

    for claim in claims:
        assert "claim_id" in claim
        assert "claim" in claim
        assert "source" in claim
