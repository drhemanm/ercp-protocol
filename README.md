# ERCP Protocol

**Evo-Recursive Constraint Prompting (ERCP) — Protocol v1.0**

A model-agnostic, auditable runtime & protocol for provable, self-correcting LLM reasoning.
ERCP formalizes iterative constraint extraction, verification, and bi-level supervision to produce
stable, high-quality reasoning traces with convergence guarantees.

This repository contains:
- The ERCP Protocol specification (`ERCPSpec.md`)
- Production-ready server implementation (FastAPI) in `/server`
- SDKs (Python) in `/sdk`
- Comprehensive test suite in `/tests`
- Security and deployment documentation

---

## Quick Links

- **Spec**: `ERCPSpec.md` (protocol endpoints, JSON schemas, operators)
- **Server**: `server/ercp_server.py` (production-ready implementation)
- **Python SDK**: `sdk/python/ercp_client.py` (client library)
- **Tests**: `tests/` (comprehensive test suite)
- **Deployment**: `DEPLOYMENT.md` (deployment guide)
- **Security**: `SECURITY.md` (security best practices)

---

## Why ERCP?

ERCP introduces a rigorous, auditable approach to LLM reasoning:
- **Operator algebra**: `generate`, `verify`, `extract_constraints`, `stabilize`, `mutate`
- **Convergence guarantees**: provable termination under reasonable assumptions
- **Constraint-first**: explicit constraints expressed as both NL and machine predicates
- **Auditability**: signed traces, model fingerprints, and reproducible runs

ERCP is intended for researchers, engineers, and production systems that require
robust, interpretable, and auditable reasoning behavior from LLMs.

---

## Features

### Production-Ready Implementation

- ✅ **Secure**: API key authentication, rate limiting, HMAC signatures, input validation
- ✅ **Scalable**: Async operations, connection pooling, horizontal scaling support
- ✅ **Transparent**: Complete audit trails, trace storage and retrieval, signed events
- ✅ **Complete**: All ERCP operators implemented with proper error handling

### Core Capabilities

- **Operator Algebra**: Generate, Verify, Extract Constraints, Stabilize, Mutate
- **Convergence Guarantees**: Finite-state termination under deterministic decoding
- **Constraint Duality**: Natural language + machine-checkable predicates
- **Auditability**: ISO8601 timestamps, model fingerprints, HMAC signatures
- **Error Handling**: Comprehensive validation and error recovery
- **Rate Limiting**: Protection against abuse and DoS attacks
- **Trace Retrieval**: Full audit trail available via GET /trace endpoint

---

## Quickstart

### Development Setup

1. **Clone repository:**
   ```bash
   git clone https://github.com/your-org/ercp-protocol.git
   cd ercp-protocol
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment:**
   ```bash
   cp .env.example .env
   # Generate a secure secret key
   python -c "import secrets; print('ERCP_SECRET_KEY=' + secrets.token_hex(32))" >> .env
   ```

4. **Run the server:**
   ```bash
   cd server
   uvicorn ercp_server:app --reload --host 0.0.0.0 --port 8080
   ```

5. **Test the server:**
   ```bash
   curl http://localhost:8080/health
   ```

   Expected response:
   ```json
   {
     "status": "healthy",
     "version": "ercp-1.0",
     "timestamp": "2025-01-15T10:30:00Z"
   }
   ```

### Using the Python SDK

```python
from sdk.python.ercp_client import ERCPClient

# Initialize client (no auth for development)
client = ERCPClient(base_url="http://localhost:8080")

# Run ERCP reasoning loop
result = client.run(
    problem_description="Why does water boil at different temperatures at different altitudes?",
    config={
        "max_iterations": 10,
        "similarity_threshold": 0.95,
        "deterministic": True
    }
)

print(f"Status: {result['status']}")
print(f"Final reasoning: {result['final_reasoning']['reasoning_text']}")
print(f"Constraints extracted: {len(result['constraints'])}")

# Retrieve trace for auditability
trace = client.get_trace(result['trace_id'])
print(f"Total events: {len(trace['trace_data']['trace_events'])}")
```

### Example with Authentication

```python
from sdk.python.ercp_client import ERCPClient

# Initialize client with API key
client = ERCPClient(
    base_url="https://api.example.com",
    api_key="your-api-key-here"
)

try:
    result = client.run("Explain photosynthesis")
    print(f"Success: {result['status']}")
except ERCPAuthenticationError as e:
    print(f"Authentication failed: {e}")
except ERCPRateLimitError as e:
    print(f"Rate limit exceeded: {e}")
except ERCPError as e:
    print(f"Error: {e}")
```

---

## Testing

### Run Test Suite

```bash
# Install test dependencies
pip install pytest

# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_comprehensive.py::test_security -v
pytest tests/test_comprehensive.py::test_functional -v
pytest tests/test_comprehensive.py::test_auditability -v
```

### Run Tests with Authentication

```bash
git clone git@github.com/drhemanm/ercp-protocol.git
cd ercp-protocol
