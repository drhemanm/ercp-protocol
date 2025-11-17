"""
SQLAlchemy ORM models for ERCP Protocol.
Defines database schema for traces, events, constraints, and errors.
"""

import uuid
from datetime import datetime
from sqlalchemy import (
    Column,
    String,
    Float,
    Integer,
    JSON,
    DateTime,
    Boolean,
    ForeignKey,
    Text,
    ARRAY,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from .database import Base


def generate_uuid():
    """Generate UUID for primary keys."""
    return str(uuid.uuid4())


class Trace(Base):
    """Trace model - represents a full ERCP execution."""

    __tablename__ = "traces"

    trace_id = Column(String, primary_key=True, default=generate_uuid)
    problem_id = Column(String, nullable=False, index=True)
    problem_description = Column(Text, nullable=False)
    status = Column(
        String, nullable=False, index=True
    )  # converged, infeasible, partial, failed
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
    config = Column(JSON, nullable=False)
    final_reasoning = Column(JSON, nullable=True)
    utility_score = Column(Float, nullable=True)
    iteration_count = Column(Integer, default=0)

    # Relationships
    events = relationship(
        "TraceEvent", back_populates="trace", cascade="all, delete-orphan"
    )
    constraints = relationship(
        "Constraint", back_populates="trace", cascade="all, delete-orphan"
    )
    errors = relationship(
        "Error", back_populates="trace", cascade="all, delete-orphan"
    )


class TraceEvent(Base):
    """TraceEvent model - represents individual operator executions."""

    __tablename__ = "trace_events"

    event_id = Column(String, primary_key=True, default=generate_uuid)
    trace_id = Column(String, ForeignKey("traces.trace_id"), nullable=False, index=True)
    operator = Column(
        String, nullable=False, index=True
    )  # generate, verify, extract, stabilize, mutate
    iteration = Column(Integer, nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    input_summary = Column(JSON, nullable=True)
    output_summary = Column(JSON, nullable=True)
    model_fingerprint = Column(String, nullable=True)
    node_signature = Column(String, nullable=True)
    duration_ms = Column(Float, nullable=True)

    # Relationship
    trace = relationship("Trace", back_populates="events")


class Constraint(Base):
    """Constraint model - stores extracted constraints."""

    __tablename__ = "constraints"

    constraint_id = Column(String, primary_key=True, default=generate_uuid)
    trace_id = Column(String, ForeignKey("traces.trace_id"), nullable=False, index=True)
    type = Column(String, nullable=False, index=True)
    priority = Column(String, nullable=False)  # critical, high, medium, low
    nl_text = Column(Text, nullable=False)
    predicate = Column(JSON, nullable=False)
    source = Column(JSON, nullable=False)
    confidence = Column(Float, nullable=False)
    immutable = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationship
    trace = relationship("Trace", back_populates="constraints")


class Error(Base):
    """Error model - stores detected errors during verification."""

    __tablename__ = "errors"

    error_id = Column(String, primary_key=True, default=generate_uuid)
    trace_id = Column(String, ForeignKey("traces.trace_id"), nullable=False, index=True)
    event_id = Column(String, ForeignKey("trace_events.event_id"), nullable=True)
    type = Column(String, nullable=False, index=True)
    span = Column(JSON, nullable=True)  # Stored as JSON array
    excerpt = Column(Text, nullable=True)
    confidence = Column(Float, nullable=False)
    detected_by = Column(JSON, nullable=False)  # Stored as JSON array
    evidence = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationship
    trace = relationship("Trace", back_populates="errors")


class ModelCache(Base):
    """ModelCache model - caches LLM outputs for efficiency."""

    __tablename__ = "model_cache"

    cache_key = Column(String, primary_key=True)
    model_name = Column(String, nullable=False, index=True)
    prompt_hash = Column(String, nullable=False, index=True)
    output = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=True)
    hit_count = Column(Integer, default=0)
