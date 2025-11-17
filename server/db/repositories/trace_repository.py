"""
Trace Repository - CRUD Operations for Trace Model
Author: ERCP Protocol Implementation
License: Apache-2.0

Provides async CRUD operations for Trace records.
"""

import logging
from typing import List, Optional, Dict, Any
from uuid import UUID

from sqlalchemy import select, update, delete, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..models import Trace, TraceEvent, Constraint, Error


logger = logging.getLogger(__name__)


class TraceRepository:
    """Repository for Trace CRUD operations."""

    @staticmethod
    async def create_trace(
        db: AsyncSession,
        trace_id: UUID,
        problem_description: str,
        problem_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        status: str = "running"
    ) -> Trace:
        """
        Create a new trace record.

        Args:
            db: Database session
            trace_id: Unique trace ID
            problem_description: The problem being solved
            problem_id: Optional problem identifier
            config: Configuration dict
            status: Initial status (default: "running")

        Returns:
            Created Trace object
        """
        trace = Trace(
            trace_id=trace_id,
            problem_id=problem_id,
            problem_description=problem_description,
            status=status,
            config=config or {}
        )

        db.add(trace)
        await db.flush()
        await db.refresh(trace)

        logger.info(f"Created trace: {trace_id}")
        return trace

    @staticmethod
    async def get_trace(
        db: AsyncSession,
        trace_id: UUID,
        include_events: bool = False,
        include_constraints: bool = False,
        include_errors: bool = False
    ) -> Optional[Trace]:
        """
        Get a trace by ID.

        Args:
            db: Database session
            trace_id: Trace ID to fetch
            include_events: Whether to load events
            include_constraints: Whether to load constraints
            include_errors: Whether to load errors

        Returns:
            Trace object or None if not found
        """
        query = select(Trace).where(Trace.trace_id == trace_id)

        # Eagerly load relationships if requested
        if include_events:
            query = query.options(selectinload(Trace.events))
        if include_constraints:
            query = query.options(selectinload(Trace.constraints))
        if include_errors:
            query = query.options(selectinload(Trace.errors))

        result = await db.execute(query)
        trace = result.scalar_one_or_none()

        if trace:
            logger.debug(f"Found trace: {trace_id}")
        else:
            logger.debug(f"Trace not found: {trace_id}")

        return trace

    @staticmethod
    async def update_trace(
        db: AsyncSession,
        trace_id: UUID,
        updates: Dict[str, Any]
    ) -> Optional[Trace]:
        """
        Update a trace record.

        Args:
            db: Database session
            trace_id: Trace ID to update
            updates: Dictionary of fields to update

        Returns:
            Updated Trace object or None if not found
        """
        stmt = (
            update(Trace)
            .where(Trace.trace_id == trace_id)
            .values(**updates)
            .returning(Trace)
        )

        result = await db.execute(stmt)
        trace = result.scalar_one_or_none()

        if trace:
            await db.flush()
            await db.refresh(trace)
            logger.info(f"Updated trace: {trace_id}")
        else:
            logger.warning(f"Trace not found for update: {trace_id}")

        return trace

    @staticmethod
    async def delete_trace(
        db: AsyncSession,
        trace_id: UUID
    ) -> bool:
        """
        Delete a trace record (CASCADE will delete related records).

        Args:
            db: Database session
            trace_id: Trace ID to delete

        Returns:
            True if deleted, False if not found
        """
        stmt = delete(Trace).where(Trace.trace_id == trace_id)
        result = await db.execute(stmt)

        deleted = result.rowcount > 0
        if deleted:
            logger.info(f"Deleted trace: {trace_id}")
        else:
            logger.warning(f"Trace not found for deletion: {trace_id}")

        return deleted

    @staticmethod
    async def list_traces(
        db: AsyncSession,
        limit: int = 100,
        offset: int = 0,
        status: Optional[str] = None,
        problem_id: Optional[str] = None
    ) -> List[Trace]:
        """
        List traces with optional filtering.

        Args:
            db: Database session
            limit: Maximum number of results
            offset: Number of results to skip
            status: Filter by status
            problem_id: Filter by problem ID

        Returns:
            List of Trace objects
        """
        query = select(Trace)

        # Apply filters
        if status:
            query = query.where(Trace.status == status)
        if problem_id:
            query = query.where(Trace.problem_id == problem_id)

        # Order by created_at desc
        query = query.order_by(Trace.created_at.desc())

        # Apply pagination
        query = query.limit(limit).offset(offset)

        result = await db.execute(query)
        traces = result.scalars().all()

        logger.debug(f"Listed {len(traces)} traces")
        return list(traces)

    @staticmethod
    async def count_traces(
        db: AsyncSession,
        status: Optional[str] = None,
        problem_id: Optional[str] = None
    ) -> int:
        """
        Count traces with optional filtering.

        Args:
            db: Database session
            status: Filter by status
            problem_id: Filter by problem ID

        Returns:
            Count of traces
        """
        query = select(func.count(Trace.trace_id))

        # Apply filters
        if status:
            query = query.where(Trace.status == status)
        if problem_id:
            query = query.where(Trace.problem_id == problem_id)

        result = await db.execute(query)
        count = result.scalar()

        return count or 0

    @staticmethod
    async def get_trace_with_all_data(
        db: AsyncSession,
        trace_id: UUID
    ) -> Optional[Trace]:
        """
        Get a trace with all related data (events, constraints, errors).

        Args:
            db: Database session
            trace_id: Trace ID to fetch

        Returns:
            Trace object with all relationships loaded, or None
        """
        return await TraceRepository.get_trace(
            db=db,
            trace_id=trace_id,
            include_events=True,
            include_constraints=True,
            include_errors=True
        )
