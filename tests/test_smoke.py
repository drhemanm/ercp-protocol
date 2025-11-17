# tests/test_smoke.py
import requests
import pytest
import os
import time

BASE = os.environ.get("ERCP_SERVER", "http://localhost:8080")

def server_up() -> bool:
    try:
        r = requests.get(f"{BASE}/ercp/v1/trace/nonexistent", timeout=2)
        # server may return 200 or 404; existence of response is enough
        return True
    except Exception:
        return False

@pytest.mark.skipif(not server_up(), reason="ERCP server not running on localhost:8080")
def test_run_smoke():
    payload = {
        "problem": {"id": "smoke1", "description": "Why does water boil at different temperatures?"},
        "config": {"model": "local-llm", "max_iterations": 2, "similarity_threshold": 0.8, "deterministic": True}
    }
    r = requests.post(f"{BASE}/ercp/v1/run", json=payload, timeout=30)
    assert r.status_code == 200
    data = r.json()
    assert "trace_id" in data
    assert "final_reasoning" in data
