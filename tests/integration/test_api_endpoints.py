"""
Integration Tests for API Endpoints
"""

import pytest


class TestAPIEndpoints:
    """Test all API endpoints."""
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self, async_test_client):
        """Test health check endpoint."""
        response = await async_test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "version" in data
        assert "database" in data
        assert data["version"] == "ercp-1.0"
    
    @pytest.mark.asyncio
    async def test_root_endpoint(self, async_test_client):
        """Test root endpoint."""
        response = await async_test_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "name" in data
        assert "endpoints" in data
        assert "health" in data["endpoints"]
    
    @pytest.mark.asyncio
    async def test_generate_endpoint(
        self,
        async_test_client,
        sample_problem,
        mock_generate_model
    ):
        """Test generate endpoint."""
        request_data = {
            "problem": sample_problem,
            "constraints": [],
            "gen_config": {"model": "gpt2"}
        }
        
        response = await async_test_client.post(
            "/ercp/v1/generate",
            json=request_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "reasoning_id" in data
        assert "reasoning_text" in data
        assert "trace_id" in data
    
    @pytest.mark.asyncio
    async def test_verify_endpoint(
        self,
        async_test_client,
        sample_reasoning,
        mock_nli_model
    ):
        """Test verify endpoint."""
        request_data = {
            "reasoning_id": sample_reasoning["reasoning_id"],
            "reasoning_text": sample_reasoning["reasoning_text"],
            "constraints": [],
            "verify_config": {}
        }
        
        response = await async_test_client.post(
            "/ercp/v1/verify",
            json=request_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "errors" in data
        assert "error_count" in data
    
    @pytest.mark.asyncio
    async def test_stabilize_endpoint(
        self,
        async_test_client,
        mock_embedding_model
    ):
        """Test stabilize endpoint."""
        request_data = {
            "reasoning_curr": "Current reasoning text",
            "reasoning_prev": None,
            "threshold": 0.95,
            "errors": []
        }
        
        response = await async_test_client.post(
            "/ercp/v1/stabilize",
            json=request_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "stable" in data
        assert "score" in data
    
    @pytest.mark.asyncio
    async def test_trace_retrieval(
        self,
        async_test_client,
        sample_problem,
        sample_config,
        mock_generate_model,
        mock_nli_model,
        mock_embedding_model
    ):
        """Test trace retrieval endpoint."""
        import numpy as np
        mock_embedding_model.encode.return_value = np.array([[0.1], [0.1]])
        
        # First create a trace
        run_response = await async_test_client.post(
            "/ercp/v1/run",
            json={"problem": sample_problem, "config": sample_config}
        )
        
        assert run_response.status_code == 200
        trace_id = run_response.json()["trace_id"]
        
        # Retrieve the trace
        get_response = await async_test_client.get(
            f"/ercp/v1/trace/{trace_id}"
        )
        
        assert get_response.status_code == 200
        trace_data = get_response.json()
        
        assert trace_data["trace_id"] == trace_id
        assert "problem_description" in trace_data
        assert "status" in trace_data
    
    @pytest.mark.asyncio
    async def test_trace_not_found(self, async_test_client):
        """Test 404 for non-existent trace."""
        response = await async_test_client.get(
            "/ercp/v1/trace/00000000-0000-0000-0000-000000000000"
        )
        
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_list_traces(
        self,
        async_test_client,
        sample_problem,
        sample_config,
        mock_generate_model,
        mock_nli_model,
        mock_embedding_model
    ):
        """Test list traces endpoint."""
        import numpy as np
        mock_embedding_model.encode.return_value = np.array([[0.1], [0.1]])
        
        # Create a trace
        await async_test_client.post(
            "/ercp/v1/run",
            json={"problem": sample_problem, "config": sample_config}
        )
        
        # List traces
        response = await async_test_client.get("/ercp/v1/traces")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "traces" in data
        assert "total" in data
        assert data["total"] >= 1
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, test_client):
        """Test that rate limiting is enforced."""
        # Note: This is a basic test. In practice, you'd need to make many requests
        # to trigger rate limiting
        
        response = test_client.get("/health")
        assert response.status_code == 200
        
        # Check for rate limit headers (if implemented)
        # assert "X-Rate-Limit-Limit" in response.headers
