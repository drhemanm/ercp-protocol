"""
Integration tests for ERCP API endpoints.
"""

import pytest
from fastapi.testclient import TestClient


def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_metrics_endpoint(client):
    """Test Prometheus metrics endpoint."""
    response = client.get("/metrics")
    # Metrics may be disabled in test, so accept 200 or 404
    assert response.status_code in [200, 404]


# Note: Full ERCP run tests would require ML models to be loaded
# For CI/CD, these can be marked with @pytest.mark.slow or skipped
@pytest.mark.skip(reason="Requires ML models to be loaded")
def test_full_ercp_run(client, sample_problem, sample_config):
    """Test complete ERCP run endpoint."""
    payload = {
        "problem": sample_problem,
        "config": sample_config,
    }

    response = client.post("/ercp/v1/run", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "trace_id" in data
    assert "status" in data
    assert "final_reasoning" in data
    assert "constraints" in data
    assert "trace_events" in data
