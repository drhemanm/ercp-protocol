"""
Database Models - SQLAlchemy ORM Models
Author: ERCP Protocol Implementation
License: Apache-2.0

Defines all database tables for the ERCP protocol:
- Trace: Main trace records
- TraceEvent: Events within a trace (operator executions)
- Constraint: Constraints extracted during execution
- Error: Errors detected during verification
- ModelCache: Cache for LLM outputs
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Boolean, 
    ForeignKey, Text, Index, JSON
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func


Base = declarative_base()


class Trace(Base):
    """
    Main trace record for an ERCP execution run.
    
    Each trace represents a complete execution of the ERCP protocol
    for a specific problem.
    """
    __tablename__ = "traces"

    trace_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    problem_id = Column(String(255), nullable=True, index=True)
    problem_description = Column(Text, nullable=False)
    status = Column(
        String(50), 
        nullable=False, 
        default="running",
        index=True
    )  # running, converged, max_iterations, failed
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())
    
    # Configuration used for this trace
    config = Column(JSONB, nullable=True)
    
    # Final reasoning output
    final_reasoning = Column(Text, nullable=True)
    
    # Utility score (if computed)
    utility_score = Column(Float, nullable=True)
    
    # Metadata
    iteration_count = Column(Integer, default=0)
    constraint_count = Column(Integer, default=0)
    error_count = Column(Integer, default=0)
    
    # Relationships
    events = relationship("TraceEvent", back_populates="trace", cascade="all, delete-orphan")
    constraints = relationship("Constraint", back_populates="trace", cascade="all, delete-orphan")
    errors = relationship("Error", back_populates="trace", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Trace(trace_id={self.trace_id}, status={self.status})>"


class TraceEvent(Base):
    """
    Event record for operator executions within a trace.
    
    Each event represents a single operator execution (generate, verify, etc.)
    """
    __tablename__ = "trace_events"

    event_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trace_id = Column(UUID(as_uuid=True), ForeignKey("traces.trace_id", ondelete="CASCADE"), nullable=False)
    
    operator = Column(String(50), nullable=False, index=True)  # generate, verify, extract, stabilize, mutate
    iteration = Column(Integer, nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Summaries of inputs and outputs (to avoid storing huge payloads)
    input_summary = Column(JSONB, nullable=True)
    output_summary = Column(JSONB, nullable=True)
    
    # Model information
    model_fingerprint = Column(String(255), nullable=True)
    node_signature = Column(String(255), nullable=True)
    
    # Execution metadata
    duration_seconds = Column(Float, nullable=True)
    success = Column(Boolean, default=True)
    error_message = Column(Text, nullable=True)
    
    # Relationship
    trace = relationship("Trace", back_populates="events")

    def __repr__(self):
        return f"<TraceEvent(event_id={self.event_id}, operator={self.operator}, iteration={self.iteration})>"

    __table_args__ = (
        Index('idx_trace_iteration', 'trace_id', 'iteration'),
        Index('idx_trace_operator', 'trace_id', 'operator'),
    )


class Constraint(Base):
    """
    Constraint record extracted from errors.
    
    Constraints are used to guide future reasoning iterations.
    """
    __tablename__ = "constraints"

    constraint_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trace_id = Column(UUID(as_uuid=True), ForeignKey("traces.trace_id", ondelete="CASCADE"), nullable=False)
    
    type = Column(String(50), nullable=False, index=True)  # contradiction, factual_incorrect, etc.
    priority = Column(Integer, nullable=False, default=50, index=True)
    
    # Natural language representation
    nl_text = Column(Text, nullable=False)
    
    # Machine-readable predicate (JSONB)
    predicate = Column(JSONB, nullable=True)
    
    # Source information (which error it came from)
    source = Column(JSONB, nullable=True)
    
    confidence = Column(Float, nullable=False, default=0.0)
    immutable = Column(Boolean, default=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationship
    trace = relationship("Trace", back_populates="constraints")

    def __repr__(self):
        return f"<Constraint(constraint_id={self.constraint_id}, type={self.type}, priority={self.priority})>"

    __table_args__ = (
        Index('idx_trace_priority', 'trace_id', 'priority'),
        Index('idx_trace_type', 'trace_id', 'type'),
    )


class Error(Base):
    """
    Error record detected during verification.
    
    Errors represent issues found in reasoning (contradictions, factual errors, etc.)
    """
    __tablename__ = "errors"

    error_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trace_id = Column(UUID(as_uuid=True), ForeignKey("traces.trace_id", ondelete="CASCADE"), nullable=False)
    event_id = Column(UUID(as_uuid=True), ForeignKey("trace_events.event_id", ondelete="CASCADE"), nullable=True)
    
    type = Column(String(50), nullable=False, index=True)  # contradiction, factual_incorrect, etc.
    
    # Span indicates which sentences are involved (e.g., [0, 1] means sentences 0 and 1)
    span = Column(JSONB, nullable=True)
    
    # Text excerpt showing the error
    excerpt = Column(Text, nullable=True)
    
    confidence = Column(Float, nullable=False, default=0.0, index=True)
    
    # Which validator detected this error
    detected_by = Column(String(50), nullable=True)
    
    # Evidence supporting the error detection
    evidence = Column(JSONB, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationship
    trace = relationship("Trace", back_populates="errors")
    event = relationship("TraceEvent")

    def __repr__(self):
        return f"<Error(error_id={self.error_id}, type={self.type}, confidence={self.confidence})>"

    __table_args__ = (
        Index('idx_trace_confidence', 'trace_id', 'confidence'),
        Index('idx_trace_type_error', 'trace_id', 'type'),
    )


class ModelCache(Base):
    """
    Cache for LLM model outputs.
    
    Caches LLM responses by prompt hash to avoid redundant API calls.
    """
    __tablename__ = "model_cache"

    cache_key = Column(String(255), primary_key=True)  # hash of prompt + model_name
    model_name = Column(String(255), nullable=False, index=True)
    prompt_hash = Column(String(64), nullable=False, index=True)
    
    # Cached output (JSONB for structured data)
    output = Column(JSONB, nullable=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True, index=True)
    
    # Hit count for analytics
    hit_count = Column(Integer, default=0)
    last_accessed = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<ModelCache(cache_key={self.cache_key}, model_name={self.model_name})>"

    __table_args__ = (
        Index('idx_prompt_model', 'prompt_hash', 'model_name'),
    )
