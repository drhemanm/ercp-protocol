"""
Database configuration and session management for ERCP Protocol.
Uses SQLAlchemy with async support and comprehensive timeout protection.
"""

import os
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool, QueuePool
from sqlalchemy import event, text

# Base class for declarative models
Base = declarative_base()

# Database URL from environment
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql+asyncpg://ercp_user:ercp_pass@localhost:5432/ercp"
)

# ============================================
# Timeout Configuration
# ============================================

# Connection pool timeout (seconds) - how long to wait for a connection from pool
DB_POOL_TIMEOUT = float(os.getenv("DB_POOL_TIMEOUT", "30"))

# Statement timeout (seconds) - how long a single SQL query can run
DB_STATEMENT_TIMEOUT = float(os.getenv("DB_STATEMENT_TIMEOUT", "60"))

# Query timeout (milliseconds) - PostgreSQL-specific statement timeout
# This is passed to PostgreSQL's statement_timeout parameter
DB_QUERY_TIMEOUT_MS = int(os.getenv("DB_QUERY_TIMEOUT_MS", "60000"))  # 60 seconds

# Pool configuration
DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "20"))
DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "10"))
DB_POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", "3600"))  # 1 hour

# Validate configuration
if DB_POOL_SIZE < 5 or DB_POOL_SIZE > 100:
    raise ValueError(f"DB_POOL_SIZE must be between 5 and 100, got {DB_POOL_SIZE}")

if DB_STATEMENT_TIMEOUT < 1 or DB_STATEMENT_TIMEOUT > 300:
    raise ValueError(f"DB_STATEMENT_TIMEOUT must be between 1 and 300 seconds, got {DB_STATEMENT_TIMEOUT}")

# ============================================
# Database-Specific Connect Args
# ============================================

def get_connect_args():
    """
    Get database-specific connection arguments with timeout settings.
    
    Returns:
        Dictionary of connection arguments
    """
    connect_args = {}
    
    if "postgresql" in DATABASE_URL:
        # PostgreSQL-specific settings
        connect_args = {
            "timeout": DB_POOL_TIMEOUT,  # Connection timeout
            "command_timeout": DB_STATEMENT_TIMEOUT,  # Query timeout
            "server_settings": {
                # Set statement timeout at PostgreSQL level
                "statement_timeout": str(DB_QUERY_TIMEOUT_MS),
                # Prevent idle in transaction timeout
                "idle_in_transaction_session_timeout": "300000",  # 5 minutes
            }
        }
    elif "mysql" in DATABASE_URL:
        # MySQL-specific settings
        connect_args = {
            "connect_timeout": int(DB_POOL_TIMEOUT),
            "read_timeout": int(DB_STATEMENT_TIMEOUT),
            "write_timeout": int(DB_STATEMENT_TIMEOUT),
        }
    elif "sqlite" in DATABASE_URL:
        # SQLite-specific settings
        connect_args = {
            "timeout": DB_POOL_TIMEOUT,
            "check_same_thread": False,
        }
    
    return connect_args


# ============================================
# Create Engine with Timeout Protection
# ============================================

# Determine pool class
if "sqlite" in DATABASE_URL:
    pool_class = NullPool  # SQLite doesn't support connection pooling well
else:
    pool_class = QueuePool

engine = create_async_engine(
    DATABASE_URL,
    # Logging
    echo=os.getenv("ENVIRONMENT", "development") == "development",
    echo_pool=os.getenv("ENVIRONMENT", "development") == "development",
    
    # Connection pool settings
    poolclass=pool_class,
    pool_size=DB_POOL_SIZE,
    max_overflow=DB_MAX_OVERFLOW,
    pool_timeout=DB_POOL_TIMEOUT,  # How long to wait for connection from pool
    pool_recycle=DB_POOL_RECYCLE,  # Recycle connections after this many seconds
    pool_pre_ping=True,  # Test connections before using
    
    # Database-specific connection arguments (timeouts)
    connect_args=get_connect_args(),
)


# ============================================
# Session Event Listeners for Additional Safety
# ============================================

@event.listens_for(engine.sync_engine, "connect")
def receive_connect(dbapi_conn, connection_record):
    """
    Set additional timeout parameters when a new connection is established.
    This provides defense-in-depth for timeout protection.
    """
    # Get the database dialect
    dialect_name = str(dbapi_conn.__class__.__module__)
    
    if "postgresql" in dialect_name or "asyncpg" in dialect_name:
        # For PostgreSQL, we already set this in connect_args
        # But we can double-check here
        pass
    
    elif "mysql" in dialect_name:
        # Set MySQL session variables
        cursor = dbapi_conn.cursor()
        cursor.execute(f"SET SESSION max_execution_time={DB_QUERY_TIMEOUT_MS}")
        cursor.close()
    
    elif "sqlite" in dialect_name:
        # SQLite timeout is already set in connect_args
        pass


@event.listens_for(engine.sync_engine, "begin")
def receive_begin(conn):
    """
    Called when a transaction begins.
    We can add transaction-level timeout logic here if needed.
    """
    pass


# ============================================
# Create Session Factory
# ============================================

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


# ============================================
# Session Dependency with Timeout Protection
# ============================================

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for getting async database sessions with timeout protection.

    Usage in FastAPI:
        @app.get("/endpoint")
        async def endpoint(db: AsyncSession = Depends(get_db)):
            # Use db session
    
    Features:
    - Automatic commit on success
    - Automatic rollback on error
    - Connection cleanup
    - Timeout protection at multiple levels
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ============================================
# Database Initialization
# ============================================

async def init_db():
    """
    Initialize database tables.
    
    Creates all tables defined in Base metadata.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_db():
    """
    Drop all database tables.
    
    WARNING: This will delete all data!
    Use with caution - typically only in tests.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


# ============================================
# Query Timeout Context Manager
# ============================================

class QueryTimeout:
    """
    Context manager for setting query-specific timeouts.
    
    Usage:
        async with QueryTimeout(session, timeout_seconds=30):
            result = await session.execute(long_running_query)
    """
    
    def __init__(self, session: AsyncSession, timeout_seconds: float):
        self.session = session
        self.timeout_ms = int(timeout_seconds * 1000)
        self.original_timeout = None
    
    async def __aenter__(self):
        """Set query timeout."""
        if "postgresql" in DATABASE_URL:
            # Save original timeout
            result = await self.session.execute(text("SHOW statement_timeout"))
            self.original_timeout = result.scalar()
            
            # Set new timeout
            await self.session.execute(
                text(f"SET LOCAL statement_timeout = {self.timeout_ms}")
            )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Restore original timeout."""
        if "postgresql" in DATABASE_URL and self.original_timeout:
            await self.session.execute(
                text(f"SET LOCAL statement_timeout = '{self.original_timeout}'")
            )
        return False


# ============================================
# Health Check Function
# ============================================

async def check_db_health() -> dict:
    """
    Check database health and connection status.
    
    Returns:
        Dictionary with health status and metrics
    """
    try:
        async with AsyncSessionLocal() as session:
            # Simple query to check connection
            result = await session.execute(text("SELECT 1"))
            result.scalar()
            
            # Get pool statistics if available
            pool_status = {
                "pool_size": engine.pool.size() if hasattr(engine.pool, 'size') else None,
                "checked_in": engine.pool.checkedin() if hasattr(engine.pool, 'checkedin') else None,
                "checked_out": engine.pool.checkedout() if hasattr(engine.pool, 'checkedout') else None,
                "overflow": engine.pool.overflow() if hasattr(engine.pool, 'overflow') else None,
            }
            
            return {
                "status": "healthy",
                "database": "connected",
                "pool": pool_status,
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e),
        }
