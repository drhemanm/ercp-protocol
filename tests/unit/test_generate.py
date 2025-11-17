"""
Unit Tests for Generate Operator
"""

import pytest
from unittest.mock import MagicMock, patch
from server.operators.generate import GenerateOperator


class TestGenerateOperator:
    """Test suite for GenerateOperator."""
    
    def test_generate_with_no_constraints(self, mock_generate_model):
        """Test generating reasoning without constraints."""
        operator = GenerateOperator(model_name="gpt2")
        
        result = operator.execute(
            problem_description="Test problem",
            constraints=[]
        )
        
        assert "reasoning_id" in result
        assert "reasoning_text" in result
        assert "sentences" in result
        assert "claims" in result
        assert len(result["sentences"]) > 0
    
    def test_generate_with_constraints(self, mock_generate_model):
        """Test generating reasoning with constraints."""
        operator = GenerateOperator(model_name="gpt2")
        
        constraints = [
            {"nl_text": "Answer must be accurate"},
            {"nl_text": "Use scientific terminology"}
        ]
        
        result = operator.execute(
            problem_description="Test problem",
            constraints=constraints
        )
        
        assert result is not None
        assert isinstance(result["sentences"], list)
        assert len(result["sentences"]) > 0
    
    def test_generate_deterministic(self, mock_generate_model):
        """Test that same input gives same output with temperature=0."""
        operator = GenerateOperator(model_name="gpt2")
        
        result1 = operator.execute(
            problem_description="Test problem",
            temperature=0.0,
            seed=42
        )
        
        result2 = operator.execute(
            problem_description="Test problem",
            temperature=0.0,
            seed=42
        )
        
        # With mocked model, results should be identical
        assert result1["reasoning_text"] == result2["reasoning_text"]
    
    def test_generate_handles_long_input(self, mock_generate_model):
        """Test handling of long problem descriptions."""
        operator = GenerateOperator(model_name="gpt2")
        
        long_problem = "This is a test problem. " * 100
        
        result = operator.execute(
            problem_description=long_problem,
            max_tokens=512
        )
        
        assert result is not None
        assert "reasoning_text" in result
    
    def test_build_constraint_prompt(self):
        """Test constraint prompt building."""
        operator = GenerateOperator(model_name="gpt2")
        
        constraints = [
            {"nl_text": "Constraint 1"},
            {"nl_text": "Constraint 2"}
        ]
        
        prompt = operator._build_constraint_prompt(constraints)
        
        assert "Constraint 1" in prompt
        assert "Constraint 2" in prompt
        assert "You must follow these constraints" in prompt
    
    def test_build_constraint_prompt_empty(self):
        """Test constraint prompt with no constraints."""
        operator = GenerateOperator(model_name="gpt2")
        
        prompt = operator._build_constraint_prompt([])
        
        assert prompt == ""
    
    def test_split_into_sentences(self):
        """Test sentence splitting."""
        operator = GenerateOperator(model_name="gpt2")
        
        text = "First sentence. Second sentence! Third sentence?"
        sentences = operator._split_into_sentences(text)
        
        assert len(sentences) == 3
        assert sentences[0] == "First sentence."
        assert sentences[1] == "Second sentence!"
        assert sentences[2] == "Third sentence?"
    
    def test_extract_claims(self):
        """Test claim extraction from sentences."""
        operator = GenerateOperator(model_name="gpt2")
        
        sentences = [
            "Water boils at 100°C.",
            "This is a declarative claim.",
            "Is this a question?",  # Should be skipped
            "Short."  # Too short, should be skipped
        ]
        
        claims = operator._extract_claims(sentences)
        
        assert len(claims) == 2  # Only 2 valid claims
        assert all("claim" in claim for claim in claims)
        assert all("source" in claim for claim in claims)
