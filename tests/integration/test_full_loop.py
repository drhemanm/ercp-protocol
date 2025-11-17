"""
Integration Tests for Full ERCP Loop
"""

import pytest
from unittest.mock import patch, MagicMock


class TestFullERCPLoop:
    """Test the complete ERCP execution loop."""
    
    @pytest.mark.asyncio
    async def test_converges_on_simple_problem(
        self,
        async_test_client,
        sample_problem,
        sample_config,
        mock_generate_model,
        mock_nli_model,
        mock_embedding_model
    ):
        """Test that ERCP converges on a simple problem."""
        # Mock embeddings for stability
        import numpy as np
        mock_embedding_model.encode.return_value = np.array([
            [0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3]
        ])
        
        # Mock NLI to return no contradictions
        mock_nli_model.return_value = [{'label': 'ENTAILMENT', 'score': 0.95}]
        
        request_data = {
            "problem": sample_problem,
            "config": sample_config
        }
        
        response = await async_test_client.post(
            "/ercp/v1/run",
            json=request_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "trace_id" in data
        assert "status" in data
        assert "final_reasoning" in data
        assert "iteration_count" in data
    
    @pytest.mark.asyncio
    async def test_max_iterations_reached(
        self,
        async_test_client,
        sample_problem,
        mock_generate_model,
        mock_nli_model,
        mock_embedding_model
    ):
        """Test that max iterations limit is respected."""
        # Mock to never converge
        import numpy as np
        mock_embedding_model.encode.side_effect = lambda x: np.random.rand(len(x), 3)
        
        config = {
            "model": "gpt2",
            "max_iterations": 2,  # Low limit
            "similarity_threshold": 0.99  # High threshold
        }
        
        request_data = {
            "problem": sample_problem,
            "config": config
        }
        
        response = await async_test_client.post(
            "/ercp/v1/run",
            json=request_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["iteration_count"] <= 2
    
    @pytest.mark.asyncio
    async def test_constraint_accumulation(
        self,
        async_test_client,
        sample_problem,
        sample_config,
        mock_generate_model,
        mock_nli_model,
        mock_embedding_model
    ):
        """Test that constraints are accumulated over iterations."""
        # Mock NLI to detect contradictions
        mock_nli_model.return_value = [{'label': 'CONTRADICTION', 'score': 0.92}]
        
        request_data = {
            "problem": sample_problem,
            "config": sample_config
        }
        
        response = await async_test_client.post(
            "/ercp/v1/run",
            json=request_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Should have accumulated some constraints
        assert "constraints" in data
    
    @pytest.mark.asyncio
    async def test_trace_persistence(
        self,
        async_test_client,
        test_db,
        sample_problem,
        sample_config,
        mock_generate_model,
        mock_nli_model,
        mock_embedding_model
    ):
        """Test that trace is saved to database."""
        import numpy as np
        mock_embedding_model.encode.return_value = np.array([[0.1, 0.2], [0.1, 0.2]])
        
        request_data = {
            "problem": sample_problem,
            "config": sample_config
        }
        
        response = await async_test_client.post(
            "/ercp/v1/run",
            json=request_data
        )
        
        assert response.status_code == 200
        data = response.json()
        trace_id = data["trace_id"]
        
        # Retrieve trace from database
        trace_response = await async_test_client.get(f"/ercp/v1/trace/{trace_id}")
        
        assert trace_response.status_code == 200
        trace_data = trace_response.json()
        
        assert trace_data["trace_id"] == trace_id
        assert trace_data["problem_description"] == sample_problem["description"]
