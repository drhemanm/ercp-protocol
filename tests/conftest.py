"""
Pytest configuration and fixtures for ERCP Protocol tests.
"""

import pytest
import asyncio
from typing import AsyncGenerator
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Import app and database
from server.ercp_server import app
from server.db.database import Base, get_db

# Test database URL (use in-memory SQLite for tests)
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
async def test_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Create a fresh database for each test.
    """
    # Create async engine
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)

    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Create session
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    async with async_session() as session:
        yield session

    # Drop all tables after test
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def sample_problem():
    """Sample problem for testing."""
    return {
        "id": "test-problem-1",
        "description": "Why does water boil at different temperatures at different altitudes?",
        "metadata": {"source": "test"},
    }


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "model": "gpt2",
        "max_iterations": 5,
        "max_constraints": 10,
        "similarity_threshold": 0.90,
        "temperature": 0.0,
        "deterministic": True,
        "verify_threshold": 0.75,
        "candidate_threshold": 0.60,
    }


@pytest.fixture
def sample_reasoning():
    """Sample reasoning text for testing."""
    return (
        "Water boils at lower temperatures at higher altitudes due to decreased atmospheric pressure. "
        "At sea level, water boils at approximately 100°C (212°F). "
        "At higher elevations, the reduced air pressure allows water molecules to escape into vapor at lower temperatures."
    )


@pytest.fixture
def sample_constraints():
    """Sample constraints for testing."""
    return [
        {
            "constraint_id": "c1",
            "type": "requirement",
            "priority": "high",
            "nl_text": "Must explain the relationship between pressure and boiling point",
            "predicate": {"type": "requirement", "operator": "must", "condition": "explain pressure relationship"},
            "source": {"detected_by": ["manual"], "error_id": None},
            "confidence": 0.95,
            "immutable": False,
        }
    ]
