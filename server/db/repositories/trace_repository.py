"""
Repository pattern implementation for Trace model.

Provides CRUD operations and query methods for ERCP execution traces.
"""

from typing import List, Optional
from uuid import UUID
from datetime import datetime

from sqlalchemy import select, update, delete, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from server.db.models import Trace, TraceEvent, Constraint, Error


class TraceRepository:
    """Repository for managing Trace records."""

    def __init__(self, session: AsyncSession):
        """
        Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        self.session = session

    async def create(self, trace: Trace) -> Trace:
        """
        Create a new trace record.

        Args:
            trace: Trace instance to create

        Returns:
            Created trace with generated ID
        """
        self.session.add(trace)
        await self.session.flush()
        await self.session.refresh(trace)
        return trace

    async def get_by_id(self, trace_id: UUID) -> Optional[Trace]:
        """
        Get trace by ID.

        Args:
            trace_id: Trace UUID

        Returns:
            Trace instance or None if not found
        """
        result = await self.session.execute(
            select(Trace).where(Trace.trace_id == trace_id)
        )
        return result.scalar_one_or_none()

    async def get_with_relationships(self, trace_id: UUID) -> Optional[Trace]:
        """
        Get trace with all related events, constraints, and errors eagerly loaded.

        Args:
            trace_id: Trace UUID

        Returns:
            Trace with relationships or None if not found
        """
        result = await self.session.execute(
            select(Trace)
            .options(
                selectinload(Trace.events),
                selectinload(Trace.constraints),
                selectinload(Trace.errors)
            )
            .where(Trace.trace_id == trace_id)
        )
        return result.scalar_one_or_none()

    async def list_by_problem_id(
        self,
        problem_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Trace]:
        """
        List all traces for a specific problem.

        Args:
            problem_id: Problem identifier
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of traces ordered by creation time (newest first)
        """
        result = await self.session.execute(
            select(Trace)
            .where(Trace.problem_id == problem_id)
            .order_by(Trace.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def list_by_status(
        self,
        status: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Trace]:
        """
        List traces by status.

        Args:
            status: Status filter (converged/infeasible/partial/failed)
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of traces with specified status
        """
        result = await self.session.execute(
            select(Trace)
            .where(Trace.status == status)
            .order_by(Trace.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def update_status(
        self,
        trace_id: UUID,
        status: str,
        final_reasoning: Optional[str] = None
    ) -> bool:
        """
        Update trace status and optionally final reasoning.

        Args:
            trace_id: Trace UUID
            status: New status value
            final_reasoning: Optional final reasoning text

        Returns:
            True if trace was updated, False if not found
        """
        update_data = {"status": status, "updated_at": datetime.utcnow()}
        if final_reasoning is not None:
            update_data["final_reasoning"] = final_reasoning

        result = await self.session.execute(
            update(Trace)
            .where(Trace.trace_id == trace_id)
            .values(**update_data)
        )
        return result.rowcount > 0

    async def update_metrics(
        self,
        trace_id: UUID,
        utility_score: Optional[float] = None,
        iteration_count: Optional[int] = None
    ) -> bool:
        """
        Update trace metrics.

        Args:
            trace_id: Trace UUID
            utility_score: Utility score for the final reasoning
            iteration_count: Total number of iterations

        Returns:
            True if trace was updated, False if not found
        """
        update_data = {"updated_at": datetime.utcnow()}
        if utility_score is not None:
            update_data["utility_score"] = utility_score
        if iteration_count is not None:
            update_data["iteration_count"] = iteration_count

        result = await self.session.execute(
            update(Trace)
            .where(Trace.trace_id == trace_id)
            .values(**update_data)
        )
        return result.rowcount > 0

    async def delete(self, trace_id: UUID) -> bool:
        """
        Delete a trace (cascades to events, constraints, errors).

        Args:
            trace_id: Trace UUID

        Returns:
            True if trace was deleted, False if not found
        """
        result = await self.session.execute(
            delete(Trace).where(Trace.trace_id == trace_id)
        )
        return result.rowcount > 0

    async def count_by_status(self) -> dict:
        """
        Get count of traces grouped by status.

        Returns:
            Dictionary mapping status to count
        """
        result = await self.session.execute(
            select(Trace.status, func.count(Trace.trace_id))
            .group_by(Trace.status)
        )
        return {status: count for status, count in result.all()}

    async def get_recent_traces(
        self,
        limit: int = 20,
        include_relationships: bool = False
    ) -> List[Trace]:
        """
        Get most recent traces.

        Args:
            limit: Maximum number of traces to return
            include_relationships: Whether to eagerly load relationships

        Returns:
            List of recent traces ordered by creation time
        """
        query = select(Trace).order_by(Trace.created_at.desc()).limit(limit)

        if include_relationships:
            query = query.options(
                selectinload(Trace.events),
                selectinload(Trace.constraints),
                selectinload(Trace.errors)
            )

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def search_by_problem_description(
        self,
        search_term: str,
        limit: int = 50,
        use_fulltext: bool = True
    ) -> List[Trace]:
        """
        Search traces by problem description using PostgreSQL full-text search.

        Args:
            search_term: Text to search for in problem descriptions
            limit: Maximum number of results
            use_fulltext: Use full-text search (default) or fallback to ILIKE

        Returns:
            List of matching traces ordered by relevance (full-text) or creation time

        Note:
            Full-text search requires the migration 002_add_fulltext_search to be applied.
            If not applied, set use_fulltext=False to use ILIKE pattern matching.
        """
        if use_fulltext:
            # Use PostgreSQL full-text search with ranking
            # to_tsquery requires proper formatting, so we use plainto_tsquery for user input
            from sqlalchemy import text, func

            result = await self.session.execute(
                select(Trace)
                .where(
                    text("problem_description_tsv @@ plainto_tsquery('english', :search_term)")
                )
                .order_by(
                    # Order by relevance (ts_rank) first, then by creation time
                    text("ts_rank(problem_description_tsv, plainto_tsquery('english', :search_term)) DESC"),
                    Trace.created_at.desc()
                )
                .limit(limit)
                .params(search_term=search_term)
            )
            return list(result.scalars().all())
        else:
            # Fallback to ILIKE pattern matching (less efficient, but works without migration)
            result = await self.session.execute(
                select(Trace)
                .where(Trace.problem_description.ilike(f"%{search_term}%"))
                .order_by(Trace.created_at.desc())
                .limit(limit)
            )
            return list(result.scalars().all())

    async def search_by_problem_description_fuzzy(
        self,
        search_term: str,
        limit: int = 50,
        similarity_threshold: float = 0.3
    ) -> List[Trace]:
        """
        Fuzzy search traces by problem description using trigram similarity.

        Args:
            search_term: Text to search for in problem descriptions
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score (0.0 to 1.0)

        Returns:
            List of matching traces ordered by similarity

        Note:
            Requires migration 002_add_fulltext_search to be applied (enables pg_trgm).
        """
        from sqlalchemy import text, func

        result = await self.session.execute(
            select(Trace)
            .where(
                text("similarity(problem_description, :search_term) > :threshold")
            )
            .order_by(
                text("similarity(problem_description, :search_term) DESC"),
                Trace.created_at.desc()
            )
            .limit(limit)
            .params(search_term=search_term, threshold=similarity_threshold)
        )
        return list(result.scalars().all())
