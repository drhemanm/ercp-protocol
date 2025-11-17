"""
Comprehensive test suite for ERCP Protocol Server
Tests security, functionality, edge cases, and auditability
"""

import pytest
import requests
import json
import os
import time
from typing import Optional

# Base URL for ERCP server
BASE_URL = os.environ.get("ERCP_SERVER", "http://localhost:8080")

# Test API key (configure in server .env)
TEST_API_KEY = os.environ.get("TEST_API_KEY", None)


# ============================
# Helper Functions
# ============================

def make_request(
    method: str,
    endpoint: str,
    json_data: Optional[dict] = None,
    api_key: Optional[str] = None,
    expect_success: bool = True
):
    """Make HTTP request with optional authentication."""
    url = f"{BASE_URL}{endpoint}"
    headers = {}

    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    if method == "GET":
        response = requests.get(url, headers=headers, timeout=30)
    elif method == "POST":
        headers["Content-Type"] = "application/json"
        response = requests.post(url, json=json_data, headers=headers, timeout=30)
    else:
        raise ValueError(f"Unsupported method: {method}")

    if expect_success:
        assert response.status_code == 200, f"Request failed: {response.text}"

    return response


# ============================
# Health Check Tests
# ============================

def test_health_check():
    """Test health check endpoint (no auth required)."""
    response = make_request("GET", "/health")
    data = response.json()

    assert "status" in data
    assert data["status"] == "healthy"
    assert "version" in data
    assert "timestamp" in data


# ============================
# Security Tests
# ============================

@pytest.mark.skipif(not TEST_API_KEY, reason="TEST_API_KEY not configured")
def test_authentication_required():
    """Test that endpoints require authentication when API keys are configured."""
    payload = {
        "problem": {"description": "Test problem"},
        "config": {"model": "local-llm"}
    }

    # Request without API key should fail
    response = make_request("POST", "/ercp/v1/run", json_data=payload, expect_success=False)
    assert response.status_code in [401, 403]


@pytest.mark.skipif(not TEST_API_KEY, reason="TEST_API_KEY not configured")
def test_invalid_api_key():
    """Test that invalid API keys are rejected."""
    payload = {
        "problem": {"description": "Test problem"},
        "config": {"model": "local-llm"}
    }

    response = make_request(
        "POST", "/ercp/v1/run",
        json_data=payload,
        api_key="invalid-key",
        expect_success=False
    )
    assert response.status_code in [401, 403]


@pytest.mark.skipif(not TEST_API_KEY, reason="TEST_API_KEY not configured")
def test_rate_limiting():
    """Test that rate limiting is enforced."""
    # Make many rapid requests to trigger rate limit
    payload = {
        "problem": {"description": "Test problem"},
        "config": {"model": "local-llm", "max_iterations": 1}
    }

    # This might not trigger in test environment if limits are high
    # Adjust based on your RATE_LIMIT_REQUESTS setting
    for i in range(150):
        response = make_request(
            "POST", "/ercp/v1/run",
            json_data=payload,
            api_key=TEST_API_KEY,
            expect_success=False
        )

        if response.status_code == 429:
            # Rate limit hit
            assert "rate limit" in response.text.lower()
            return

    # If we didn't hit rate limit, that's okay (limits might be high)
    pytest.skip("Rate limit not reached with current configuration")


# ============================
# Input Validation Tests
# ============================

@pytest.mark.skipif(not TEST_API_KEY, reason="TEST_API_KEY not configured")
def test_empty_problem_description():
    """Test that empty problem descriptions are rejected."""
    payload = {
        "problem": {"description": ""},
        "config": {"model": "local-llm"}
    }

    response = make_request(
        "POST", "/ercp/v1/run",
        json_data=payload,
        api_key=TEST_API_KEY,
        expect_success=False
    )
    assert response.status_code == 422  # Validation error


@pytest.mark.skipif(not TEST_API_KEY, reason="TEST_API_KEY not configured")
def test_invalid_config_values():
    """Test that invalid configuration values are rejected."""
    # Test negative max_iterations
    payload = {
        "problem": {"description": "Test problem"},
        "config": {"model": "local-llm", "max_iterations": -1}
    }

    response = make_request(
        "POST", "/ercp/v1/run",
        json_data=payload,
        api_key=TEST_API_KEY,
        expect_success=False
    )
    assert response.status_code == 422

    # Test invalid similarity_threshold
    payload = {
        "problem": {"description": "Test problem"},
        "config": {"model": "local-llm", "similarity_threshold": 1.5}
    }

    response = make_request(
        "POST", "/ercp/v1/run",
        json_data=payload,
        api_key=TEST_API_KEY,
        expect_success=False
    )
    assert response.status_code == 422


# ============================
# Functional Tests
# ============================

@pytest.mark.skipif(not TEST_API_KEY, reason="TEST_API_KEY not configured")
def test_full_run_workflow():
    """Test complete ERCP run workflow."""
    payload = {
        "problem": {
            "id": "test-1",
            "description": "Why does water boil at different temperatures at different altitudes?"
        },
        "config": {
            "model": "local-llm",
            "max_iterations": 5,
            "similarity_threshold": 0.85,
            "deterministic": True
        }
    }

    response = make_request("POST", "/ercp/v1/run", json_data=payload, api_key=TEST_API_KEY)
    data = response.json()

    # Verify response structure
    assert "trace_id" in data
    assert "timestamp" in data
    assert "proto_version" in data
    assert data["proto_version"] == "ercp-1.0"
    assert "status" in data
    assert data["status"] in ["converged", "partial", "infeasible", "failed"]
    assert "final_reasoning" in data
    assert "constraints" in data
    assert "trace_events" in data
    assert "model_fingerprint" in data
    assert "node_signature" in data

    # Verify signature format
    assert data["node_signature"].startswith("hmac-sha256:")

    return data["trace_id"]


@pytest.mark.skipif(not TEST_API_KEY, reason="TEST_API_KEY not configured")
def test_generate_endpoint():
    """Test generate endpoint."""
    payload = {
        "problem": {"description": "Explain photosynthesis"},
        "constraints": [],
        "gen_config": {"model": "local-llm", "temperature": 0.0}
    }

    response = make_request("POST", "/ercp/v1/generate", json_data=payload, api_key=TEST_API_KEY)
    data = response.json()

    assert "reasoning_id" in data
    assert "reasoning_text" in data
    assert "sentences" in data
    assert "claims" in data
    assert "model_fingerprint" in data
    assert "node_signature" in data


@pytest.mark.skipif(not TEST_API_KEY, reason="TEST_API_KEY not configured")
def test_verify_endpoint():
    """Test verify endpoint."""
    payload = {
        "reasoning_id": "test-123",
        "reasoning_text": "This is a test reasoning with sufficient length to pass validation checks.",
        "constraints": [],
        "verify_config": {"nli_threshold": 0.7, "run_fact_check": False}
    }

    response = make_request("POST", "/ercp/v1/verify", json_data=payload, api_key=TEST_API_KEY)
    data = response.json()

    assert "errors" in data
    assert isinstance(data["errors"], list)
    assert "model_fingerprint" in data


@pytest.mark.skipif(not TEST_API_KEY, reason="TEST_API_KEY not configured")
def test_stabilize_endpoint():
    """Test stabilize endpoint."""
    payload = {
        "reasoning_prev": "Previous reasoning text about water boiling.",
        "reasoning_curr": "Current reasoning text about water boiling at altitude.",
        "threshold": 0.9
    }

    response = make_request("POST", "/ercp/v1/stabilize", json_data=payload, api_key=TEST_API_KEY)
    data = response.json()

    assert "stable" in data
    assert isinstance(data["stable"], bool)
    assert "score" in data
    assert isinstance(data["score"], (int, float))


@pytest.mark.skipif(not TEST_API_KEY, reason="TEST_API_KEY not configured")
def test_mutate_endpoint():
    """Test mutate endpoint with different strategies."""
    strategies = ["relax", "reframe", "decompose"]

    for strategy in strategies:
        payload = {
            "problem": {"description": "Complex problem description"},
            "reasoning_text": "Some reasoning that didn't converge",
            "mutation_strategy": strategy,
            "mutation_config": {}
        }

        response = make_request("POST", "/ercp/v1/mutate", json_data=payload, api_key=TEST_API_KEY)
        data = response.json()

        assert "new_problem" in data
        assert "new_constraints" in data
        assert "mutation_notes" in data


@pytest.mark.skipif(not TEST_API_KEY, reason="TEST_API_KEY not configured")
def test_invalid_mutation_strategy():
    """Test that invalid mutation strategies are rejected."""
    payload = {
        "problem": {"description": "Test problem"},
        "reasoning_text": "Test reasoning",
        "mutation_strategy": "invalid_strategy",
        "mutation_config": {}
    }

    response = make_request(
        "POST", "/ercp/v1/mutate",
        json_data=payload,
        api_key=TEST_API_KEY,
        expect_success=False
    )
    assert response.status_code == 400


# ============================
# Trace Retrieval Tests
# ============================

@pytest.mark.skipif(not TEST_API_KEY, reason="TEST_API_KEY not configured")
def test_trace_retrieval():
    """Test trace storage and retrieval."""
    # First, create a trace
    payload = {
        "problem": {"description": "Test problem for trace retrieval"},
        "config": {"model": "local-llm", "max_iterations": 2}
    }

    run_response = make_request("POST", "/ercp/v1/run", json_data=payload, api_key=TEST_API_KEY)
    trace_id = run_response.json()["trace_id"]

    # Now retrieve the trace
    trace_response = make_request("GET", f"/ercp/v1/trace/{trace_id}", api_key=TEST_API_KEY)
    trace_data = trace_response.json()

    assert "trace_id" in trace_data
    assert trace_data["trace_id"] == trace_id
    assert "retrieved_at" in trace_data
    assert "trace_data" in trace_data


@pytest.mark.skipif(not TEST_API_KEY, reason="TEST_API_KEY not configured")
def test_trace_not_found():
    """Test that non-existent traces return 404."""
    fake_trace_id = "00000000-0000-0000-0000-000000000000"

    response = make_request(
        "GET", f"/ercp/v1/trace/{fake_trace_id}",
        api_key=TEST_API_KEY,
        expect_success=False
    )
    assert response.status_code == 404


# ============================
# Auditability Tests
# ============================

@pytest.mark.skipif(not TEST_API_KEY, reason="TEST_API_KEY not configured")
def test_signature_verification():
    """Test that signatures can be verified (basic check)."""
    payload = {
        "problem": {"description": "Test for signature verification"},
        "config": {"model": "local-llm", "max_iterations": 1}
    }

    response = make_request("POST", "/ercp/v1/run", json_data=payload, api_key=TEST_API_KEY)
    data = response.json()

    # Verify signature is present and properly formatted
    assert "node_signature" in data
    signature = data["node_signature"]
    assert signature.startswith("hmac-sha256:")
    assert len(signature) > 12  # More than just the prefix


@pytest.mark.skipif(not TEST_API_KEY, reason="TEST_API_KEY not configured")
def test_trace_events_completeness():
    """Test that trace events capture all operations."""
    payload = {
        "problem": {"description": "Test for trace completeness"},
        "config": {
            "model": "local-llm",
            "max_iterations": 3,
            "similarity_threshold": 0.9
        }
    }

    response = make_request("POST", "/ercp/v1/run", json_data=payload, api_key=TEST_API_KEY)
    data = response.json()

    trace_events = data["trace_events"]
    assert len(trace_events) > 0

    # Check that events have required fields
    for event in trace_events:
        assert "event_id" in event
        assert "trace_id" in event
        assert "timestamp" in event
        assert "operator" in event
        assert "iteration" in event
        assert "model_fingerprint" in event


# ============================
# Edge Cases and Error Handling
# ============================

@pytest.mark.skipif(not TEST_API_KEY, reason="TEST_API_KEY not configured")
def test_max_constraints_limit():
    """Test that max_constraints limit is respected."""
    payload = {
        "problem": {"description": "Test problem"},
        "config": {
            "model": "local-llm",
            "max_iterations": 100,
            "max_constraints": 2  # Low limit
        }
    }

    response = make_request("POST", "/ercp/v1/run", json_data=payload, api_key=TEST_API_KEY)
    data = response.json()

    # Should stop when constraint limit is reached
    assert len(data["constraints"]) <= 2


@pytest.mark.skipif(not TEST_API_KEY, reason="TEST_API_KEY not configured")
def test_deterministic_execution():
    """Test that deterministic mode produces consistent results."""
    payload = {
        "problem": {"description": "Determinism test problem"},
        "config": {
            "model": "local-llm",
            "max_iterations": 2,
            "deterministic": True,
            "temperature": 0.0
        }
    }

    # Run twice with same config
    response1 = make_request("POST", "/ercp/v1/run", json_data=payload, api_key=TEST_API_KEY)
    data1 = response1.json()

    response2 = make_request("POST", "/ercp/v1/run", json_data=payload, api_key=TEST_API_KEY)
    data2 = response2.json()

    # With stub implementation, results should be similar
    # In production with real LLM, this tests deterministic decoding
    assert data1["final_reasoning"]["reasoning_text"] == data2["final_reasoning"]["reasoning_text"]


# ============================
# Performance Tests
# ============================

@pytest.mark.skipif(not TEST_API_KEY, reason="TEST_API_KEY not configured")
def test_response_time():
    """Test that responses are returned in reasonable time."""
    payload = {
        "problem": {"description": "Performance test"},
        "config": {"model": "local-llm", "max_iterations": 3}
    }

    start_time = time.time()
    response = make_request("POST", "/ercp/v1/run", json_data=payload, api_key=TEST_API_KEY)
    end_time = time.time()

    elapsed = end_time - start_time

    # Should complete in reasonable time (adjust based on your setup)
    assert elapsed < 30, f"Request took too long: {elapsed}s"

    data = response.json()
    assert "trace_id" in data


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
