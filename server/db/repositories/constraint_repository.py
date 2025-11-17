"""
Constraint Repository - CRUD Operations for Constraint Model
Author: ERCP Protocol Implementation
License: Apache-2.0

Provides async CRUD operations for Constraint records.
"""

import logging
from typing import List, Optional, Dict, Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import Constraint


logger = logging.getLogger(__name__)


class ConstraintRepository:
    """Repository for Constraint CRUD operations."""

    @staticmethod
    async def create_constraint(
        db: AsyncSession,
        constraint_id: UUID,
        trace_id: UUID,
        type: str,
        nl_text: str,
        priority: int = 50,
        predicate: Optional[Dict[str, Any]] = None,
        source: Optional[Dict[str, Any]] = None,
        confidence: float = 0.0,
        immutable: bool = False
    ) -> Constraint:
        """
        Create a new constraint record.

        Args:
            db: Database session
            constraint_id: Unique constraint ID
            trace_id: Parent trace ID
            type: Constraint type
            nl_text: Natural language representation
            priority: Priority score
            predicate: Machine-readable predicate
            source: Source information
            confidence: Confidence score
            immutable: Whether constraint is immutable

        Returns:
            Created Constraint object
        """
        constraint = Constraint(
            constraint_id=constraint_id,
            trace_id=trace_id,
            type=type,
            priority=priority,
            nl_text=nl_text,
            predicate=predicate,
            source=source or {},
            confidence=confidence,
            immutable=immutable
        )

        db.add(constraint)
        await db.flush()
        await db.refresh(constraint)

        logger.debug(f"Created constraint: {constraint_id} (trace: {trace_id})")
        return constraint

    @staticmethod
    async def get_constraint(
        db: AsyncSession,
        constraint_id: UUID
    ) -> Optional[Constraint]:
        """
        Get a constraint by ID.

        Args:
            db: Database session
            constraint_id: Constraint ID to fetch

        Returns:
            Constraint object or None if not found
        """
        query = select(Constraint).where(Constraint.constraint_id == constraint_id)
        result = await db.execute(query)
        constraint = result.scalar_one_or_none()

        if constraint:
            logger.debug(f"Found constraint: {constraint_id}")
        else:
            logger.debug(f"Constraint not found: {constraint_id}")

        return constraint

    @staticmethod
    async def get_constraints_for_trace(
        db: AsyncSession,
        trace_id: UUID,
        type: Optional[str] = None,
        immutable_only: bool = False,
        min_priority: Optional[int] = None
    ) -> List[Constraint]:
        """
        Get all constraints for a trace.

        Args:
            db: Database session
            trace_id: Trace ID
            type: Optional filter by constraint type
            immutable_only: Only return immutable constraints
            min_priority: Minimum priority threshold

        Returns:
            List of Constraint objects
        """
        query = select(Constraint).where(Constraint.trace_id == trace_id)

        # Apply filters
        if type:
            query = query.where(Constraint.type == type)
        if immutable_only:
            query = query.where(Constraint.immutable == True)
        if min_priority is not None:
            query = query.where(Constraint.priority >= min_priority)

        # Order by priority (high to low), then created_at
        query = query.order_by(Constraint.priority.desc(), Constraint.created_at)

        result = await db.execute(query)
        constraints = result.scalars().all()

        logger.debug(f"Found {len(constraints)} constraints for trace {trace_id}")
        return list(constraints)

    @staticmethod
    async def create_constraints_bulk(
        db: AsyncSession,
        constraints: List[Dict[str, Any]]
    ) -> List[Constraint]:
        """
        Create multiple constraints in bulk.

        Args:
            db: Database session
            constraints: List of constraint dictionaries

        Returns:
            List of created Constraint objects
        """
        constraint_objects = []
        
        for constraint_data in constraints:
            constraint = Constraint(**constraint_data)
            db.add(constraint)
            constraint_objects.append(constraint)

        await db.flush()
        
        for constraint in constraint_objects:
            await db.refresh(constraint)

        logger.debug(f"Created {len(constraint_objects)} constraints in bulk")
        return constraint_objects

    @staticmethod
    async def get_high_priority_constraints(
        db: AsyncSession,
        trace_id: UUID,
        priority_threshold: int = 60
    ) -> List[Constraint]:
        """
        Get high-priority constraints for a trace.

        Args:
            db: Database session
            trace_id: Trace ID
            priority_threshold: Minimum priority (default: 60)

        Returns:
            List of high-priority Constraint objects
        """
        return await ConstraintRepository.get_constraints_for_trace(
            db=db,
            trace_id=trace_id,
            min_priority=priority_threshold
        )

    @staticmethod
    async def get_immutable_constraints(
        db: AsyncSession,
        trace_id: UUID
    ) -> List[Constraint]:
        """
        Get all immutable constraints for a trace.

        Args:
            db: Database session
            trace_id: Trace ID

        Returns:
            List of immutable Constraint objects
        """
        return await ConstraintRepository.get_constraints_for_trace(
            db=db,
            trace_id=trace_id,
            immutable_only=True
        )
