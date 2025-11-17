"""
TraceEvent Repository - CRUD Operations for TraceEvent Model
Author: ERCP Protocol Implementation
License: Apache-2.0

Provides async CRUD operations for TraceEvent records.
"""

import logging
from typing import List, Optional, Dict, Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import TraceEvent


logger = logging.getLogger(__name__)


class TraceEventRepository:
    """Repository for TraceEvent CRUD operations."""

    @staticmethod
    async def create_event(
        db: AsyncSession,
        event_id: UUID,
        trace_id: UUID,
        operator: str,
        iteration: int,
        input_summary: Optional[Dict[str, Any]] = None,
        output_summary: Optional[Dict[str, Any]] = None,
        model_fingerprint: Optional[str] = None,
        node_signature: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> TraceEvent:
        """
        Create a new trace event record.

        Args:
            db: Database session
            event_id: Unique event ID
            trace_id: Parent trace ID
            operator: Operator name (generate, verify, etc.)
            iteration: Iteration number
            input_summary: Summary of inputs
            output_summary: Summary of outputs
            model_fingerprint: Model fingerprint
            node_signature: Node signature
            duration_seconds: Execution duration
            success: Whether execution was successful
            error_message: Error message if failed

        Returns:
            Created TraceEvent object
        """
        event = TraceEvent(
            event_id=event_id,
            trace_id=trace_id,
            operator=operator,
            iteration=iteration,
            input_summary=input_summary or {},
            output_summary=output_summary or {},
            model_fingerprint=model_fingerprint,
            node_signature=node_signature,
            duration_seconds=duration_seconds,
            success=success,
            error_message=error_message
        )

        db.add(event)
        await db.flush()
        await db.refresh(event)

        logger.debug(
            f"Created event: {event_id} (trace: {trace_id}, "
            f"operator: {operator}, iteration: {iteration})"
        )
        return event

    @staticmethod
    async def get_event(
        db: AsyncSession,
        event_id: UUID
    ) -> Optional[TraceEvent]:
        """
        Get a trace event by ID.

        Args:
            db: Database session
            event_id: Event ID to fetch

        Returns:
            TraceEvent object or None if not found
        """
        query = select(TraceEvent).where(TraceEvent.event_id == event_id)
        result = await db.execute(query)
        event = result.scalar_one_or_none()

        if event:
            logger.debug(f"Found event: {event_id}")
        else:
            logger.debug(f"Event not found: {event_id}")

        return event

    @staticmethod
    async def get_events_for_trace(
        db: AsyncSession,
        trace_id: UUID,
        operator: Optional[str] = None,
        iteration: Optional[int] = None
    ) -> List[TraceEvent]:
        """
        Get all events for a trace.

        Args:
            db: Database session
            trace_id: Trace ID
            operator: Optional filter by operator
            iteration: Optional filter by iteration

        Returns:
            List of TraceEvent objects
        """
        query = select(TraceEvent).where(TraceEvent.trace_id == trace_id)

        # Apply filters
        if operator:
            query = query.where(TraceEvent.operator == operator)
        if iteration is not None:
            query = query.where(TraceEvent.iteration == iteration)

        # Order by iteration, then timestamp
        query = query.order_by(TraceEvent.iteration, TraceEvent.timestamp)

        result = await db.execute(query)
        events = result.scalars().all()

        logger.debug(f"Found {len(events)} events for trace {trace_id}")
        return list(events)

    @staticmethod
    async def get_events_by_operator(
        db: AsyncSession,
        trace_id: UUID,
        operator: str
    ) -> List[TraceEvent]:
        """
        Get all events for a specific operator in a trace.

        Args:
            db: Database session
            trace_id: Trace ID
            operator: Operator name

        Returns:
            List of TraceEvent objects
        """
        return await TraceEventRepository.get_events_for_trace(
            db=db,
            trace_id=trace_id,
            operator=operator
        )

    @staticmethod
    async def get_events_by_iteration(
        db: AsyncSession,
        trace_id: UUID,
        iteration: int
    ) -> List[TraceEvent]:
        """
        Get all events for a specific iteration in a trace.

        Args:
            db: Database session
            trace_id: Trace ID
            iteration: Iteration number

        Returns:
            List of TraceEvent objects
        """
        return await TraceEventRepository.get_events_for_trace(
            db=db,
            trace_id=trace_id,
            iteration=iteration
        )

    @staticmethod
    async def get_latest_event_for_operator(
        db: AsyncSession,
        trace_id: UUID,
        operator: str
    ) -> Optional[TraceEvent]:
        """
        Get the most recent event for a specific operator.

        Args:
            db: Database session
            trace_id: Trace ID
            operator: Operator name

        Returns:
            TraceEvent object or None
        """
        query = (
            select(TraceEvent)
            .where(TraceEvent.trace_id == trace_id)
            .where(TraceEvent.operator == operator)
            .order_by(TraceEvent.timestamp.desc())
            .limit(1)
        )

        result = await db.execute(query)
        event = result.scalar_one_or_none()

        return event
