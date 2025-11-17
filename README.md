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
# Set environment variables
export ERCP_SERVER=http://localhost:8080
export TEST_API_KEY=your-test-key

# Run tests
pytest tests/test_comprehensive.py -v
```

### Test Coverage

The test suite includes:
- ✅ Health check tests
- ✅ Security tests (authentication, rate limiting)
- ✅ Input validation tests
- ✅ Functional tests (all endpoints)
- ✅ Trace retrieval tests
- ✅ Auditability tests (signatures, events)
- ✅ Edge cases and error handling
- ✅ Performance tests

---

## Deployment

### Quick Deploy with Docker

```bash
# Build image
docker build -t ercp-server .

# Run container
docker run -d -p 8080:8080 --env-file .env ercp-server
```

### Production Deployment

See [`DEPLOYMENT.md`](DEPLOYMENT.md) for comprehensive deployment guide including:
- Docker and Docker Compose setup
- Nginx reverse proxy configuration
- Database integration (PostgreSQL)
- Caching (Redis)
- Horizontal scaling
- Monitoring and logging
- Performance optimization

---

## Security

### Security Features

- **Authentication**: Bearer token / API key authentication
- **Rate Limiting**: Configurable per-client limits
- **Input Validation**: Strict schema validation
- **Cryptographic Signatures**: HMAC-SHA256 for all responses
- **HTTPS/TLS**: Required in production
- **CORS Protection**: Configurable allowed origins
- **PII Redaction**: Basic sanitization (enhance for production)

### Security Best Practices

1. **Generate strong secret keys:**
   ```bash
   python -c "import secrets; print(secrets.token_hex(32))"
   ```

2. **Enable HTTPS/TLS in production**

3. **Configure API keys:**
   ```bash
   export ERCP_API_KEYS=key1,key2,key3
   ```

4. **Set up rate limiting:**
   ```bash
   export RATE_LIMIT_REQUESTS=100
   export RATE_LIMIT_WINDOW=60
   ```

See [`SECURITY.md`](SECURITY.md) for complete security guide including:
- Threat model
- Authentication & authorization
- Data protection
- Network security
- Compliance (GDPR, SOC 2, HIPAA)
- Incident response

---

## API Endpoints

### Core Endpoints

- `POST /ercp/v1/run` - Execute full ERCP reasoning loop
- `POST /ercp/v1/generate` - Generate reasoning with constraints
- `POST /ercp/v1/verify` - Verify reasoning for errors
- `POST /ercp/v1/extract_constraints` - Extract constraints from errors
- `POST /ercp/v1/stabilize` - Check semantic stability
- `POST /ercp/v1/mutate` - Mutate problem for reframing
- `GET /ercp/v1/trace/{trace_id}` - Retrieve audit trace

### Utility Endpoints

- `GET /health` - Health check (no auth required)

See [`ERCPSpec.md`](ERCPSpec.md) for complete API documentation.

---

## Architecture

```
┌─────────────┐
│   Client    │
│   (SDK)     │
└──────┬──────┘
       │
       │ HTTPS + Bearer Token
       │
┌──────▼──────────────────────────────────────┐
│         ERCP Server (FastAPI)               │
│                                             │
│  ┌─────────────────────────────────────┐   │
│  │  Security Layer                     │   │
│  │  - Authentication                   │   │
│  │  - Rate Limiting                    │   │
│  │  - Input Validation                 │   │
│  └────────────┬────────────────────────┘   │
│               │                             │
│  ┌────────────▼────────────────────────┐   │
│  │  ERCP Core Operators                │   │
│  │  - Generate (G)                     │   │
│  │  - Verify (V)                       │   │
│  │  - Extract Constraints (X)          │   │
│  │  - Stabilize (O_stab)               │   │
│  │  - Mutate (M)                       │   │
│  └────────────┬────────────────────────┘   │
│               │                             │
│  ┌────────────▼────────────────────────┐   │
│  │  Trace Storage & Audit              │   │
│  │  - Event logging                    │   │
│  │  - HMAC signatures                  │   │
│  │  - Model fingerprints               │   │
│  └─────────────────────────────────────┘   │
│                                             │
└─────────────────────────────────────────────┘
       │
       │ (Optional)
       │
┌──────▼──────────┐      ┌─────────────┐
│   PostgreSQL    │      │    Redis    │
│  (Trace Store)  │      │  (Caching)  │
└─────────────────┘      └─────────────┘
```

---

## Configuration

### Environment Variables

See `.env.example` for all configuration options:

**Security:**
- `ERCP_SECRET_KEY` - HMAC signing key (REQUIRED, min 32 bytes)
- `ERCP_API_KEYS` - Comma-separated API keys for authentication

**Rate Limiting:**
- `RATE_LIMIT_REQUESTS` - Max requests per window (default: 100)
- `RATE_LIMIT_WINDOW` - Time window in seconds (default: 60)

**Network:**
- `CORS_ORIGINS` - Allowed CORS origins (default: "*")
- `ALLOWED_HOSTS` - Trusted hosts for production
- `ENVIRONMENT` - development/production

**Limits:**
- `MAX_REQUEST_SIZE` - Max payload size in bytes (default: 1MB)

---

## Contributing

We welcome contributions! Please see [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines.

### Development Workflow

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run test suite
6. Submit pull request

### Code Quality

- Follow PEP 8 style guide
- Add type hints
- Write comprehensive tests
- Update documentation

---

## License

Apache-2.0 License - See LICENSE file for details

---

## Support

- **Issues**: [GitHub Issues](https://github.com/your-org/ercp-protocol/issues)
- **Documentation**: See `ERCPSpec.md`, `DEPLOYMENT.md`, `SECURITY.md`
- **Security**: Report vulnerabilities to security@example.com

---

## Roadmap

### Current (v1.0)
- ✅ Core ERCP operators
- ✅ Security & authentication
- ✅ Audit trails
- ✅ Python SDK

### Planned (v1.1)
- [ ] JavaScript/TypeScript SDK
- [ ] PostgreSQL integration
- [ ] Redis caching
- [ ] Advanced NLI verification
- [ ] Enhanced PII detection
- [ ] Metrics & monitoring dashboard

### Future (v2.0)
- [ ] Multi-model support
- [ ] Distributed execution
- [ ] GraphQL API
- [ ] WebSocket streaming
- [ ] Advanced constraint solvers (Z3, PySAT)

---

## Citation

If you use ERCP in your research, please cite:

```bibtex
@article{mohabeer2025ercp,
  title={ERCP: Evo-Recursive Constraint Prompting for Provable LLM Reasoning},
  author={Mohabeer, Heman},
  journal={EvoLogics AI Lab},
  year={2025}
}
```

---

**Version:** 1.0 (Production-Ready)
**Author:** Dr. Heman Mohabeer — EvoLogics AI Lab
**Last Updated:** 2025-01-15
