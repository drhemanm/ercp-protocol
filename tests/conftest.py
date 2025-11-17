"""
Pytest Configuration and Fixtures
Author: ERCP Protocol Implementation
License: Apache-2.0

Provides test fixtures for database, clients, and mocks.
"""

import pytest
import pytest_asyncio
import os
import sys
from typing import AsyncGenerator, Generator
from unittest.mock import Mock, MagicMock, patch

# Add server to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from fastapi.testclient import TestClient
from httpx import AsyncClient

from server.db.models import Base
from server.db.database import get_db
from server.ercp_server import app


# Test database URL (SQLite for testing)
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def test_engine():
    """Create a test database engine."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        echo=False,
        connect_args={"check_same_thread": False}
    )
    
    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # Drop all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()


@pytest_asyncio.fixture
async def test_db(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session."""
    async_session = async_sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    async with async_session() as session:
        yield session


@pytest.fixture
def test_client(test_db) -> Generator[TestClient, None, None]:
    """Create a test client with database override."""
    
    async def override_get_db():
        yield test_db
    
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as client:
        yield client
    
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def async_test_client(test_db) -> AsyncGenerator[AsyncClient, None]:
    """Create an async test client."""
    
    async def override_get_db():
        yield test_db
    
    app.dependency_overrides[get_db] = override_get_db
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client
    
    app.dependency_overrides.clear()


# Mock ML Models (to avoid loading large models in tests)

@pytest.fixture
def mock_generate_model():
    """Mock the generate model to avoid loading actual LLM."""
    with patch('server.operators.generate.get_model_registry') as mock_registry:
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        
        # Mock tokenizer
        mock_tokenizer.return_value = {
            'input_ids': MagicMock(),
            'attention_mask': MagicMock()
        }
        mock_tokenizer.eos_token_id = 0
        mock_tokenizer.pad_token = "[PAD]"
        mock_tokenizer.decode.return_value = "This is a test reasoning. Water boils at 100°C at sea level."
        
        # Mock model
        mock_model.generate.return_value = [[1, 2, 3, 4, 5]]
        
        # Mock registry
        mock_registry_instance = MagicMock()
        mock_registry_instance.get_generate_model.return_value = {
            'model': mock_model,
            'tokenizer': mock_tokenizer,
            'device': 'cpu'
        }
        mock_registry.return_value = mock_registry_instance
        
        yield mock_registry_instance


@pytest.fixture
def mock_nli_model():
    """Mock the NLI model to avoid loading actual model."""
    with patch('server.validators.nli_validator.get_model_registry') as mock_registry:
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{'label': 'ENTAILMENT', 'score': 0.95}]
        
        mock_registry_instance = MagicMock()
        mock_registry_instance.get_nli_model.return_value = mock_pipeline
        mock_registry.return_value = mock_registry_instance
        
        yield mock_pipeline


@pytest.fixture
def mock_embedding_model():
    """Mock the embedding model to avoid loading actual model."""
    with patch('server.operators.stabilize.get_model_registry') as mock_registry:
        import numpy as np
        
        mock_model = MagicMock()
        # Return similar embeddings for similarity testing
        mock_model.encode.return_value = np.array([
            [0.1, 0.2, 0.3, 0.4],
            [0.11, 0.21, 0.31, 0.41]
        ])
        
        mock_registry_instance = MagicMock()
        mock_registry_instance.get_embedding_model.return_value = mock_model
        mock_registry.return_value = mock_registry_instance
        
        yield mock_model


# Sample test data

@pytest.fixture
def sample_problem():
    """Sample problem for testing."""
    return {
        "id": "test-001",
        "description": "Explain why water boils at different temperatures at different altitudes.",
        "metadata": {}
    }


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "model": "gpt2",
        "max_iterations": 5,
        "max_constraints": 10,
        "similarity_threshold": 0.95,
        "temperature": 0.0,
        "deterministic": True,
        "verify_threshold": 0.75,
        "candidate_threshold": 0.60
    }


@pytest.fixture
def sample_reasoning():
    """Sample reasoning text for testing."""
    return {
        "reasoning_id": "test-reasoning-001",
        "reasoning_text": "Water boils at lower temperatures at higher altitudes due to decreased atmospheric pressure. At sea level, boiling occurs at around 100°C.",
        "sentences": [
            "Water boils at lower temperatures at higher altitudes due to decreased atmospheric pressure.",
            "At sea level, boiling occurs at around 100°C."
        ],
        "claims": [
            {"claim": "Boiling point decreases with altitude", "source": "llm"},
            {"claim": "Sea-level boiling is ~100°C", "source": "llm"}
        ]
    }


@pytest.fixture
def sample_constraint():
    """Sample constraint for testing."""
    return {
        "type": "contradiction",
        "priority": 80,
        "nl_text": "Avoid contradictory statements about boiling temperatures",
        "predicate": {"predicate_name": "no_contradiction", "args": {}},
        "confidence": 0.85,
        "immutable": False,
        "source": {"error_type": "contradiction"}
    }


@pytest.fixture
def sample_error():
    """Sample error for testing."""
    return {
        "type": "contradiction",
        "span": [0, 1],
        "excerpt": "Water boils at 100°C <-> Water boils at 50°C",
        "confidence": 0.92,
        "detected_by": "nli_validator",
        "evidence": {
            "nli_label": "contradiction",
            "nli_score": 0.92,
            "sentence_1": "Water boils at 100°C",
            "sentence_2": "Water boils at 50°C"
        }
    }


# Environment variable overrides for testing

@pytest.fixture(autouse=True)
def test_env_vars(monkeypatch):
    """Set test environment variables."""
    monkeypatch.setenv("DATABASE_URL", TEST_DATABASE_URL)
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-key-for-testing-only")
    monkeypatch.setenv("API_KEYS", "test-key-1,test-key-2")
    monkeypatch.setenv("LOG_LEVEL", "ERROR")  # Reduce noise in tests
    monkeypatch.setenv("WARM_UP_MODELS", "")  # Don't warm up models in tests
