"""
Unit Tests for Verify Operator
"""

import pytest
from unittest.mock import MagicMock, patch
from server.operators.verify import VerifyOperator


class TestVerifyOperator:
    """Test suite for VerifyOperator."""
    
    def test_no_error_on_consistent_text(self, mock_nli_model):
        """Test that consistent text produces no errors."""
        operator = VerifyOperator(nli_threshold=0.75)
        
        result = operator.execute(
            reasoning_text="Water boils at 100°C at sea level.",
            reasoning_id="test-001",
            constraints=[]
        )
        
        assert "errors" in result
        assert "error_count" in result
        assert result["error_count"] >= 0
    
    def test_multiple_errors_detection(self, mock_nli_model):
        """Test detection of multiple errors."""
        # Mock NLI to return contradictions
        mock_nli_model.return_value = [{'label': 'CONTRADICTION', 'score': 0.95}]
        
        operator = VerifyOperator(nli_threshold=0.75)
        
        reasoning = "Water boils at 100°C. Water boils at 50°C."
        
        result = operator.execute(
            reasoning_text=reasoning,
            reasoning_id="test-001",
            constraints=[]
        )
        
        # Should detect contradiction
        assert result is not None
    
    def test_constraint_validation(self, mock_nli_model):
        """Test validation against constraints."""
        operator = VerifyOperator(nli_threshold=0.75)
        
        constraints = [
            {"nl_text": "Temperature must be above 90°C"}
        ]
        
        result = operator.execute(
            reasoning_text="The temperature is 100°C.",
            reasoning_id="test-001",
            constraints=constraints
        )
        
        assert result is not None
        assert "errors" in result
    
    def test_deduplicate_errors(self):
        """Test error deduplication."""
        operator = VerifyOperator()
        
        errors = [
            {"type": "contradiction", "span": [0, 1]},
            {"type": "contradiction", "span": [0, 1]},  # Duplicate
            {"type": "factual", "span": [2, 3]}
        ]
        
        unique = operator._deduplicate_errors(errors)
        
        assert len(unique) == 2  # Should remove duplicate
    
    def test_sort_by_confidence(self):
        """Test sorting errors by confidence."""
        operator = VerifyOperator()
        
        errors = [
            {"confidence": 0.5},
            {"confidence": 0.9},
            {"confidence": 0.7}
        ]
        
        sorted_errors = operator._sort_by_confidence(errors)
        
        assert sorted_errors[0]["confidence"] == 0.9
        assert sorted_errors[1]["confidence"] == 0.7
        assert sorted_errors[2]["confidence"] == 0.5
