# ERCP Protocol — Error-Refinement Constraint Protocol

[![CI](https://github.com/drhemanm/ercp-protocol/actions/workflows/ci.yml/badge.svg)](https://github.com/drhemanm/ercp-protocol/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Production-Ready Implementation** of the Error-Refinement Constraint Protocol (ERCP) — a novel feedback loop for refining LLM reasoning using automated error detection and constraint extraction.

## 🎯 Overview

ERCP is a meta-reasoning protocol that iteratively improves LLM outputs through:
- **Automated error detection** using NLI models and rule-based validators
- **Constraint extraction** from detected errors
- **Iterative refinement** until semantic stability is achieved
- **Production-grade infrastructure** with real ML components, database persistence, and monitoring

## ✨ Features

### Real ML Components
- ✅ **Generate (G)**: Transformer-based text generation with constraint injection
- ✅ **Verify (V)**: NLI contradiction detection + rule-based validators
- ✅ **Extract (X)**: LLM-powered constraint synthesis from errors
- ✅ **Stabilize (O_stab)**: Sentence-transformer semantic similarity
- ✅ **Mutate (M)**: Problem decomposition and reframing

### Production Infrastructure
- 🔒 **Security**: JWT auth, rate limiting, input sanitization
- 💾 **Persistence**: PostgreSQL with async SQLAlchemy, Alembic migrations
- 📊 **Monitoring**: Structured logging, Prometheus metrics, health checks
- 🐳 **Deployment**: Docker, Kubernetes, CI/CD pipelines
- 🧪 **Testing**: Unit, integration, and golden test suites

## Server Implementations

This repository includes two server implementations:

### Production Server (`ercp_server_v2.py`)
- **Use this for production deployments**
- Full ML operator integration (real models)
- Database persistence with PostgreSQL
- Metrics, logging, and monitoring
- Security middleware and authentication
- **Start with:** `uvicorn server.ercp_server_v2:app`

### Reference Server (`ercp_server.py`)
- Lightweight reference implementation
- Stub operators for API contract testing
- No ML dependencies required
- Useful for development and documentation
- **Start with:** `uvicorn server.ercp_server:app`

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- PostgreSQL 15+ (optional for development)
- Docker & Docker Compose (recommended)

### Using Docker Compose (Recommended)

```bash
# Clone repository
git clone https://github.com/drhemanm/ercp-protocol.git
cd ercp-protocol

# Set up environment
cp .env.example .env
# Edit .env with your configuration

# Start all services (Postgres, Redis, ERCP server)
docker-compose up -d

# Check health
curl http://localhost:8080/health

# View logs
docker-compose logs -f ercp-server
```

### Manual Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Download ML models
python -m spacy download en_core_web_sm

# Set up database (optional for testing)
export DATABASE_URL="postgresql+asyncpg://user:pass@localhost:5432/ercp"
alembic upgrade head

# Run server
python -m uvicorn server.ercp_server_v2:app --host 0.0.0.0 --port 8080
```

## 📖 Usage

### Basic API Call

```python
import requests

# Define problem
payload = {
    "problem": {
        "id": "physics-1",
        "description": "Why does water boil at different temperatures at different altitudes?"
    },
    "config": {
        "model": "gpt2",
        "max_iterations": 10,
        "similarity_threshold": 0.95,
        "deterministic": true
    }
}

# Run ERCP
response = requests.post("http://localhost:8080/ercp/v1/run", json=payload)
result = response.json()

print(f"Status: {result['status']}")
print(f"Iterations: {result['iterations']}")
print(f"Final reasoning: {result['final_reasoning']['reasoning_text']}")
print(f"Constraints extracted: {len(result['constraints'])}")
```

### Using Python SDK

```python
from sdk.python.ercp_client import ERCPClient

client = ERCPClient(base_url="http://localhost:8080")

result = client.run(
    problem="Why does water boil at different temperatures?",
    max_iterations=10
)

print(result["final_reasoning"]["reasoning_text"])
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     ERCP Server                          │
├─────────────────────────────────────────────────────────┤
│  FastAPI App + Middleware (Auth, Rate Limit, CORS)      │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐            │
│  │ Generate  │→ │  Verify   │→ │  Extract  │            │
│  │    (G)    │  │    (V)    │  │    (X)    │            │
│  └───────────┘  └───────────┘  └───────────┘            │
│       ↓              ↓               ↓                    │
│  ┌───────────┐  ┌───────────┐                            │
│  │ Stabilize │  │  Mutate   │                            │
│  │  (O_stab) │  │    (M)    │                            │
│  └───────────┘  └───────────┘                            │
│                                                           │
├─────────────────────────────────────────────────────────┤
│  Model Registry (GPT-2, DeBERTa, SentenceTransformers)  │
├─────────────────────────────────────────────────────────┤
│  Database Layer (PostgreSQL + SQLAlchemy Async)         │
├─────────────────────────────────────────────────────────┤
│  Monitoring (Prometheus Metrics + Structured Logs)      │
└─────────────────────────────────────────────────────────┘
```

## 📊 Monitoring

### Prometheus Metrics

Access metrics at `http://localhost:8080/metrics`

**Key Metrics**:
- `ercp_runs_total{status}` - Total runs by status
- `ercp_iteration_count` - Iterations per run
- `ercp_duration_seconds` - Run duration
- `ercp_operator_duration_seconds{operator}` - Per-operator timing

### Structured Logging

JSON logs with contextual information:
```json
{
  "event": "ercp.run.start",
  "trace_id": "abc-123",
  "problem_id": "physics-1",
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "info"
}
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=server tests/

# Run specific test suite
pytest tests/unit/test_generate.py -v

# Run integration tests
pytest tests/integration/ -v
```

## 📚 Documentation

- [**Deployment Guide**](docs/DEPLOYMENT.md) - Production deployment instructions
- [**API Specification**](ERCPSpec.md) - Full protocol specification
- [**Contributing**](CONTRIBUTING.md) - Development guidelines

## 🛠️ Development

### Project Structure

```
ercp-protocol/
├── server/
│   ├── operators/          # ML operators (G, V, X, O_stab, M)
│   ├── models/             # Model registry and loading
│   ├── db/                 # Database models and repositories
│   ├── auth/               # Authentication (JWT)
│   ├── middleware/         # Rate limiting, CORS, sanitization
│   ├── logging/            # Structured logging
│   ├── metrics/            # Prometheus metrics
│   └── ercp_server_v2.py   # Main production server
├── tests/
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   └── conftest.py         # Pytest fixtures
├── k8s/                    # Kubernetes manifests
├── .github/workflows/      # CI/CD pipelines
├── Dockerfile              # Container image
├── docker-compose.yml      # Local development stack
└── requirements.txt        # Python dependencies
```

### Adding a New Operator

1. Create operator in `server/operators/`
2. Inherit from `BaseOperator`
3. Implement `execute()` method
4. Add to `server/operators/__init__.py`
5. Write tests in `tests/unit/`

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## 🔬 Research & Citation

If you use ERCP in your research, please cite:

```bibtex
@software{ercp2024,
  title = {ERCP: Error-Refinement Constraint Protocol},
  author = {Mohabeer, Heman and EvoLogics AI Lab},
  year = {2024},
  url = {https://github.com/drhemanm/ercp-protocol}
}
```

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- ML models from [HuggingFace Transformers](https://huggingface.co/transformers/)
- Semantic similarity via [Sentence Transformers](https://www.sbert.net/)

## 📧 Contact

**Dr. Heman Mohabeer**
EvoLogics AI Lab
Email: team@evologics.ai
GitHub: [@drhemanm](https://github.com/drhemanm)

---

**Status**: ✅ Production Ready | **Version**: 2.0 | **Last Updated**: 2024-01-15
