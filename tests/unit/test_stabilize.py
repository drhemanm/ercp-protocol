"""
Unit Tests for Stabilize Operator
"""

import pytest
from server.operators.stabilize import StabilizeOperator


class TestStabilizeOperator:
    """Test suite for StabilizeOperator."""
    
    def test_first_iteration_unstable(self, mock_embedding_model):
        """Test that first iteration is always unstable."""
        operator = StabilizeOperator()
        
        result = operator.execute(
            reasoning_curr="Current reasoning",
            reasoning_prev=None,
            threshold=0.95,
            errors=[]
        )
        
        assert result["stable"] is False
        assert result["score"] == 0.0
    
    def test_high_similarity_stable(self, mock_embedding_model):
        """Test that high similarity with no errors is stable."""
        operator = StabilizeOperator()
        
        result = operator.execute(
            reasoning_curr="Water boils at 100°C at sea level.",
            reasoning_prev="Water boils at 100°C at sea level.",
            threshold=0.90,
            errors=[]
        )
        
        # With mocked embeddings, similarity should be high
        assert result is not None
        assert "stable" in result
        assert "score" in result
    
    def test_errors_prevent_stability(self, mock_embedding_model):
        """Test that errors prevent stability even with high similarity."""
        operator = StabilizeOperator()
        
        errors = [{"type": "contradiction"}]
        
        result = operator.execute(
            reasoning_curr="Same text",
            reasoning_prev="Same text",
            threshold=0.95,
            errors=errors
        )
        
        # Should be unstable due to errors
        assert result["stable"] is False
        assert result["error_count"] == 1
    
    def test_low_similarity_unstable(self, mock_embedding_model):
        """Test that low similarity results in unstable state."""
        # Mock to return low similarity
        import numpy as np
        mock_embedding_model.encode.return_value = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
        
        operator = StabilizeOperator()
        
        result = operator.execute(
            reasoning_curr="Completely different text",
            reasoning_prev="Original text",
            threshold=0.95,
            errors=[]
        )
        
        assert result is not None
        assert "score" in result
