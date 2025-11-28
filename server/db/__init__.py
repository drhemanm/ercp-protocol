"""
Database package for ERCP Protocol.
Provides database models, session management, and repositories.

CORRECTED VERSION - Exports both get_db() and get_db_with_commit()
"""

from .database import (
    Base,
    engine,
    AsyncSessionLocal,
    get_db,
    get_db_with_commit,  # ← ADDED: Export both session functions
    init_db,
    drop_db,
)
from .models import Trace, TraceEvent, Constraint, Error, ModelCache

__all__ = [
    "Base",
    "engine",
    "AsyncSessionLocal",
    "get_db",
    "get_db_with_commit",  # ← ADDED: Make both available for import
    "init_db",
    "drop_db",
    "Trace",
    "TraceEvent",
    "Constraint",
    "Error",
    "ModelCache",
]
