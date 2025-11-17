"""
Repository pattern implementation for Constraint model.

Provides CRUD operations and query methods for extracted constraints.
"""

from typing import List, Optional
from uuid import UUID
from datetime import datetime

from sqlalchemy import select, update, delete, func
from sqlalchemy.ext.asyncio import AsyncSession

from server.db.models import Constraint


class ConstraintRepository:
    """Repository for managing Constraint records."""

    def __init__(self, session: AsyncSession):
        """
        Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        self.session = session

    async def create(self, constraint: Constraint) -> Constraint:
        """
        Create a new constraint record.

        Args:
            constraint: Constraint instance to create

        Returns:
            Created constraint with generated ID
        """
        self.session.add(constraint)
        await self.session.flush()
        await self.session.refresh(constraint)
        return constraint

    async def create_batch(self, constraints: List[Constraint]) -> List[Constraint]:
        """
        Create multiple constraints in batch.

        Args:
            constraints: List of Constraint instances

        Returns:
            List of created constraints
        """
        self.session.add_all(constraints)
        await self.session.flush()
        for constraint in constraints:
            await self.session.refresh(constraint)
        return constraints

    async def get_by_id(self, constraint_id: UUID) -> Optional[Constraint]:
        """
        Get constraint by ID.

        Args:
            constraint_id: Constraint UUID

        Returns:
            Constraint instance or None if not found
        """
        result = await self.session.execute(
            select(Constraint).where(Constraint.constraint_id == constraint_id)
        )
        return result.scalar_one_or_none()

    async def list_by_trace_id(
        self,
        trace_id: UUID,
        include_immutable_only: bool = False
    ) -> List[Constraint]:
        """
        List all constraints for a specific trace.

        Args:
            trace_id: Trace UUID
            include_immutable_only: If True, only return immutable constraints

        Returns:
            List of constraints ordered by priority and creation time
        """
        query = select(Constraint).where(Constraint.trace_id == trace_id)

        if include_immutable_only:
            query = query.where(Constraint.immutable == True)

        # Order by priority: critical > high > medium > low
        query = query.order_by(
            Constraint.priority.desc(),
            Constraint.created_at.asc()
        )

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def list_by_type(
        self,
        trace_id: UUID,
        constraint_type: str
    ) -> List[Constraint]:
        """
        List constraints by type for a specific trace.

        Args:
            trace_id: Trace UUID
            constraint_type: Type of constraint to filter by

        Returns:
            List of constraints matching the type
        """
        result = await self.session.execute(
            select(Constraint)
            .where(
                Constraint.trace_id == trace_id,
                Constraint.type == constraint_type
            )
            .order_by(Constraint.created_at.asc())
        )
        return list(result.scalars().all())

    async def list_by_priority(
        self,
        trace_id: UUID,
        priority: str
    ) -> List[Constraint]:
        """
        List constraints by priority level.

        Args:
            trace_id: Trace UUID
            priority: Priority level (critical/high/medium/low)

        Returns:
            List of constraints with specified priority
        """
        result = await self.session.execute(
            select(Constraint)
            .where(
                Constraint.trace_id == trace_id,
                Constraint.priority == priority
            )
            .order_by(Constraint.created_at.asc())
        )
        return list(result.scalars().all())

    async def get_critical_constraints(self, trace_id: UUID) -> List[Constraint]:
        """
        Get all critical priority constraints for a trace.

        Args:
            trace_id: Trace UUID

        Returns:
            List of critical constraints
        """
        return await self.list_by_priority(trace_id, "critical")

    async def get_immutable_constraints(self, trace_id: UUID) -> List[Constraint]:
        """
        Get all immutable constraints for a trace.

        Args:
            trace_id: Trace UUID

        Returns:
            List of immutable constraints
        """
        result = await self.session.execute(
            select(Constraint)
            .where(
                Constraint.trace_id == trace_id,
                Constraint.immutable == True
            )
            .order_by(Constraint.priority.desc(), Constraint.created_at.asc())
        )
        return list(result.scalars().all())

    async def update_confidence(
        self,
        constraint_id: UUID,
        confidence: float
    ) -> bool:
        """
        Update constraint confidence score.

        Args:
            constraint_id: Constraint UUID
            confidence: New confidence value (0.0 to 1.0)

        Returns:
            True if updated, False if not found
        """
        result = await self.session.execute(
            update(Constraint)
            .where(Constraint.constraint_id == constraint_id)
            .values(confidence=confidence, updated_at=datetime.utcnow())
        )
        return result.rowcount > 0

    async def mark_immutable(self, constraint_id: UUID) -> bool:
        """
        Mark a constraint as immutable.

        Args:
            constraint_id: Constraint UUID

        Returns:
            True if updated, False if not found
        """
        result = await self.session.execute(
            update(Constraint)
            .where(Constraint.constraint_id == constraint_id)
            .values(immutable=True, updated_at=datetime.utcnow())
        )
        return result.rowcount > 0

    async def delete(self, constraint_id: UUID) -> bool:
        """
        Delete a constraint.

        Args:
            constraint_id: Constraint UUID

        Returns:
            True if deleted, False if not found
        """
        result = await self.session.execute(
            delete(Constraint).where(Constraint.constraint_id == constraint_id)
        )
        return result.rowcount > 0

    async def delete_by_trace_id(self, trace_id: UUID) -> int:
        """
        Delete all constraints for a trace.

        Args:
            trace_id: Trace UUID

        Returns:
            Number of constraints deleted
        """
        result = await self.session.execute(
            delete(Constraint).where(Constraint.trace_id == trace_id)
        )
        return result.rowcount

    async def count_by_trace_id(self, trace_id: UUID) -> int:
        """
        Count constraints for a trace.

        Args:
            trace_id: Trace UUID

        Returns:
            Total number of constraints
        """
        result = await self.session.execute(
            select(func.count(Constraint.constraint_id))
            .where(Constraint.trace_id == trace_id)
        )
        return result.scalar_one()

    async def get_constraint_stats(self, trace_id: UUID) -> dict:
        """
        Get constraint statistics for a trace.

        Args:
            trace_id: Trace UUID

        Returns:
            Dictionary with counts by priority and type
        """
        # Count by priority
        priority_result = await self.session.execute(
            select(Constraint.priority, func.count(Constraint.constraint_id))
            .where(Constraint.trace_id == trace_id)
            .group_by(Constraint.priority)
        )
        priority_counts = {priority: count for priority, count in priority_result.all()}

        # Count by type
        type_result = await self.session.execute(
            select(Constraint.type, func.count(Constraint.constraint_id))
            .where(Constraint.trace_id == trace_id)
            .group_by(Constraint.type)
        )
        type_counts = {ctype: count for ctype, count in type_result.all()}

        # Count immutable
        immutable_result = await self.session.execute(
            select(func.count(Constraint.constraint_id))
            .where(Constraint.trace_id == trace_id, Constraint.immutable == True)
        )
        immutable_count = immutable_result.scalar_one()

        return {
            "by_priority": priority_counts,
            "by_type": type_counts,
            "immutable_count": immutable_count,
            "total": sum(priority_counts.values())
        }

    async def search_by_text(
        self,
        trace_id: UUID,
        search_term: str
    ) -> List[Constraint]:
        """
        Search constraints by natural language text.

        Args:
            trace_id: Trace UUID
            search_term: Text to search for in nl_text field

        Returns:
            List of matching constraints
        """
        result = await self.session.execute(
            select(Constraint)
            .where(
                Constraint.trace_id == trace_id,
                Constraint.nl_text.ilike(f"%{search_term}%")
            )
            .order_by(Constraint.priority.desc())
        )
        return list(result.scalars().all())
