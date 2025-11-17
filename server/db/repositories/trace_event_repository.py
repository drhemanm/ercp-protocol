"""
Repository pattern implementation for TraceEvent model.

Provides CRUD operations and query methods for operator execution events.
"""

from typing import List, Optional
from uuid import UUID
from datetime import datetime

from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from server.db.models import TraceEvent


class TraceEventRepository:
    """Repository for managing TraceEvent records."""

    def __init__(self, session: AsyncSession):
        """
        Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        self.session = session

    async def create(self, event: TraceEvent) -> TraceEvent:
        """
        Create a new trace event record.

        Args:
            event: TraceEvent instance to create

        Returns:
            Created event with generated ID
        """
        self.session.add(event)
        await self.session.flush()
        await self.session.refresh(event)
        return event

    async def create_batch(self, events: List[TraceEvent]) -> List[TraceEvent]:
        """
        Create multiple events in batch.

        Args:
            events: List of TraceEvent instances

        Returns:
            List of created events
        """
        self.session.add_all(events)
        await self.session.flush()
        for event in events:
            await self.session.refresh(event)
        return events

    async def get_by_id(self, event_id: UUID) -> Optional[TraceEvent]:
        """
        Get event by ID.

        Args:
            event_id: Event UUID

        Returns:
            TraceEvent instance or None if not found
        """
        result = await self.session.execute(
            select(TraceEvent).where(TraceEvent.event_id == event_id)
        )
        return result.scalar_one_or_none()

    async def list_by_trace_id(
        self,
        trace_id: UUID,
        limit: Optional[int] = None
    ) -> List[TraceEvent]:
        """
        List all events for a specific trace.

        Args:
            trace_id: Trace UUID
            limit: Optional limit on number of events

        Returns:
            List of events ordered by iteration and creation time
        """
        query = (
            select(TraceEvent)
            .where(TraceEvent.trace_id == trace_id)
            .order_by(TraceEvent.iteration.asc(), TraceEvent.created_at.asc())
        )

        if limit:
            query = query.limit(limit)

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def list_by_operator(
        self,
        trace_id: UUID,
        operator: str
    ) -> List[TraceEvent]:
        """
        List events by operator type.

        Args:
            trace_id: Trace UUID
            operator: Operator name (e.g., 'generate', 'verify', 'extract')

        Returns:
            List of events for the specified operator
        """
        result = await self.session.execute(
            select(TraceEvent)
            .where(
                TraceEvent.trace_id == trace_id,
                TraceEvent.operator == operator
            )
            .order_by(TraceEvent.iteration.asc())
        )
        return list(result.scalars().all())

    async def list_by_iteration(
        self,
        trace_id: UUID,
        iteration: int
    ) -> List[TraceEvent]:
        """
        List all events for a specific iteration.

        Args:
            trace_id: Trace UUID
            iteration: Iteration number

        Returns:
            List of events in the iteration
        """
        result = await self.session.execute(
            select(TraceEvent)
            .where(
                TraceEvent.trace_id == trace_id,
                TraceEvent.iteration == iteration
            )
            .order_by(TraceEvent.created_at.asc())
        )
        return list(result.scalars().all())

    async def get_latest_by_operator(
        self,
        trace_id: UUID,
        operator: str
    ) -> Optional[TraceEvent]:
        """
        Get the most recent event for a specific operator.

        Args:
            trace_id: Trace UUID
            operator: Operator name

        Returns:
            Most recent event or None if not found
        """
        result = await self.session.execute(
            select(TraceEvent)
            .where(
                TraceEvent.trace_id == trace_id,
                TraceEvent.operator == operator
            )
            .order_by(TraceEvent.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def count_by_trace_id(self, trace_id: UUID) -> int:
        """
        Count events for a trace.

        Args:
            trace_id: Trace UUID

        Returns:
            Total number of events
        """
        result = await self.session.execute(
            select(func.count(TraceEvent.event_id))
            .where(TraceEvent.trace_id == trace_id)
        )
        return result.scalar_one()

    async def count_by_operator(self, trace_id: UUID) -> dict:
        """
        Get count of events grouped by operator.

        Args:
            trace_id: Trace UUID

        Returns:
            Dictionary mapping operator name to count
        """
        result = await self.session.execute(
            select(TraceEvent.operator, func.count(TraceEvent.event_id))
            .where(TraceEvent.trace_id == trace_id)
            .group_by(TraceEvent.operator)
        )
        return {operator: count for operator, count in result.all()}

    async def get_max_iteration(self, trace_id: UUID) -> int:
        """
        Get the maximum iteration number for a trace.

        Args:
            trace_id: Trace UUID

        Returns:
            Maximum iteration number or 0 if no events
        """
        result = await self.session.execute(
            select(func.max(TraceEvent.iteration))
            .where(TraceEvent.trace_id == trace_id)
        )
        max_iter = result.scalar_one()
        return max_iter if max_iter is not None else 0

    async def get_performance_stats(self, trace_id: UUID) -> dict:
        """
        Get performance statistics for all events in a trace.

        Args:
            trace_id: Trace UUID

        Returns:
            Dictionary with duration statistics by operator
        """
        result = await self.session.execute(
            select(
                TraceEvent.operator,
                func.count(TraceEvent.event_id).label("count"),
                func.avg(TraceEvent.duration_ms).label("avg_duration"),
                func.min(TraceEvent.duration_ms).label("min_duration"),
                func.max(TraceEvent.duration_ms).label("max_duration"),
                func.sum(TraceEvent.duration_ms).label("total_duration")
            )
            .where(TraceEvent.trace_id == trace_id)
            .group_by(TraceEvent.operator)
        )

        stats = {}
        for row in result.all():
            stats[row.operator] = {
                "count": row.count,
                "avg_duration_ms": float(row.avg_duration) if row.avg_duration else 0.0,
                "min_duration_ms": float(row.min_duration) if row.min_duration else 0.0,
                "max_duration_ms": float(row.max_duration) if row.max_duration else 0.0,
                "total_duration_ms": float(row.total_duration) if row.total_duration else 0.0
            }

        return stats

    async def get_model_fingerprints(self, trace_id: UUID) -> dict:
        """
        Get all unique model fingerprints used in a trace.

        Args:
            trace_id: Trace UUID

        Returns:
            Dictionary mapping operator to list of model fingerprints
        """
        result = await self.session.execute(
            select(TraceEvent.operator, TraceEvent.model_fingerprint)
            .where(TraceEvent.trace_id == trace_id)
            .distinct()
        )

        fingerprints = {}
        for operator, fingerprint in result.all():
            if operator not in fingerprints:
                fingerprints[operator] = []
            if fingerprint and fingerprint not in fingerprints[operator]:
                fingerprints[operator].append(fingerprint)

        return fingerprints

    async def search_by_output(
        self,
        trace_id: UUID,
        search_term: str
    ) -> List[TraceEvent]:
        """
        Search events by output summary.

        Args:
            trace_id: Trace UUID
            search_term: Text to search for in output_summary

        Returns:
            List of matching events
        """
        result = await self.session.execute(
            select(TraceEvent)
            .where(
                TraceEvent.trace_id == trace_id,
                TraceEvent.output_summary.ilike(f"%{search_term}%")
            )
            .order_by(TraceEvent.iteration.asc())
        )
        return list(result.scalars().all())

    async def get_operator_timeline(self, trace_id: UUID) -> List[dict]:
        """
        Get chronological timeline of operator executions.

        Args:
            trace_id: Trace UUID

        Returns:
            List of dictionaries with operator execution details
        """
        result = await self.session.execute(
            select(
                TraceEvent.iteration,
                TraceEvent.operator,
                TraceEvent.duration_ms,
                TraceEvent.created_at
            )
            .where(TraceEvent.trace_id == trace_id)
            .order_by(TraceEvent.created_at.asc())
        )

        timeline = []
        for row in result.all():
            timeline.append({
                "iteration": row.iteration,
                "operator": row.operator,
                "duration_ms": row.duration_ms,
                "timestamp": row.created_at.isoformat() if row.created_at else None
            })

        return timeline

    async def delete_by_trace_id(self, trace_id: UUID) -> int:
        """
        Delete all events for a trace.

        Args:
            trace_id: Trace UUID

        Returns:
            Number of events deleted
        """
        from sqlalchemy import delete

        result = await self.session.execute(
            delete(TraceEvent).where(TraceEvent.trace_id == trace_id)
        )
        return result.rowcount
