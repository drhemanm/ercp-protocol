"""
Repository pattern implementation for ModelCache model.

Provides CRUD operations and query methods for LLM output caching.
"""

from typing import List, Optional
from uuid import UUID
from datetime import datetime, timedelta

from sqlalchemy import select, update, delete, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from server.db.models import ModelCache


class ModelCacheRepository:
    """Repository for managing ModelCache records."""

    def __init__(self, session: AsyncSession):
        """
        Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        self.session = session

    async def create(self, cache_entry: ModelCache) -> ModelCache:
        """
        Create a new cache entry.

        Args:
            cache_entry: ModelCache instance to create

        Returns:
            Created cache entry with generated ID
        """
        self.session.add(cache_entry)
        await self.session.flush()
        await self.session.refresh(cache_entry)
        return cache_entry

    async def get_by_cache_key(self, cache_key: str) -> Optional[ModelCache]:
        """
        Get cache entry by cache key.

        Args:
            cache_key: Cache key string

        Returns:
            ModelCache instance or None if not found or expired
        """
        result = await self.session.execute(
            select(ModelCache)
            .where(
                ModelCache.cache_key == cache_key,
                ModelCache.expires_at > datetime.utcnow()
            )
        )
        return result.scalar_one_or_none()

    async def get_by_prompt_hash(
        self,
        model_name: str,
        prompt_hash: str
    ) -> Optional[ModelCache]:
        """
        Get cache entry by model name and prompt hash.

        Args:
            model_name: Name of the model
            prompt_hash: Hash of the prompt

        Returns:
            ModelCache instance or None if not found or expired
        """
        result = await self.session.execute(
            select(ModelCache)
            .where(
                ModelCache.model_name == model_name,
                ModelCache.prompt_hash == prompt_hash,
                ModelCache.expires_at > datetime.utcnow()
            )
        )
        return result.scalar_one_or_none()

    async def increment_hit_count(self, cache_key: str) -> bool:
        """
        Increment the hit count for a cache entry.

        Args:
            cache_key: Cache key string

        Returns:
            True if updated, False if not found
        """
        result = await self.session.execute(
            update(ModelCache)
            .where(ModelCache.cache_key == cache_key)
            .values(
                hit_count=ModelCache.hit_count + 1,
                updated_at=datetime.utcnow()
            )
        )
        return result.rowcount > 0

    async def extend_expiration(
        self,
        cache_key: str,
        extension_hours: int = 24
    ) -> bool:
        """
        Extend the expiration time for a cache entry.

        Args:
            cache_key: Cache key string
            extension_hours: Number of hours to extend

        Returns:
            True if updated, False if not found
        """
        new_expiration = datetime.utcnow() + timedelta(hours=extension_hours)

        result = await self.session.execute(
            update(ModelCache)
            .where(ModelCache.cache_key == cache_key)
            .values(expires_at=new_expiration, updated_at=datetime.utcnow())
        )
        return result.rowcount > 0

    async def list_by_model_name(
        self,
        model_name: str,
        include_expired: bool = False,
        limit: int = 100
    ) -> List[ModelCache]:
        """
        List cache entries for a specific model.

        Args:
            model_name: Name of the model
            include_expired: Whether to include expired entries
            limit: Maximum number of results

        Returns:
            List of cache entries
        """
        query = select(ModelCache).where(ModelCache.model_name == model_name)

        if not include_expired:
            query = query.where(ModelCache.expires_at > datetime.utcnow())

        query = query.order_by(ModelCache.created_at.desc()).limit(limit)

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_popular_entries(
        self,
        min_hit_count: int = 5,
        limit: int = 50
    ) -> List[ModelCache]:
        """
        Get most popular cache entries by hit count.

        Args:
            min_hit_count: Minimum hit count to include
            limit: Maximum number of results

        Returns:
            List of popular cache entries
        """
        result = await self.session.execute(
            select(ModelCache)
            .where(
                ModelCache.hit_count >= min_hit_count,
                ModelCache.expires_at > datetime.utcnow()
            )
            .order_by(ModelCache.hit_count.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def delete_expired(self) -> int:
        """
        Delete all expired cache entries.

        Returns:
            Number of entries deleted
        """
        result = await self.session.execute(
            delete(ModelCache)
            .where(ModelCache.expires_at <= datetime.utcnow())
        )
        return result.rowcount

    async def delete_by_cache_key(self, cache_key: str) -> bool:
        """
        Delete a specific cache entry.

        Args:
            cache_key: Cache key string

        Returns:
            True if deleted, False if not found
        """
        result = await self.session.execute(
            delete(ModelCache).where(ModelCache.cache_key == cache_key)
        )
        return result.rowcount > 0

    async def delete_by_model_name(self, model_name: str) -> int:
        """
        Delete all cache entries for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Number of entries deleted
        """
        result = await self.session.execute(
            delete(ModelCache).where(ModelCache.model_name == model_name)
        )
        return result.rowcount

    async def get_cache_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        # Total entries
        total_result = await self.session.execute(
            select(func.count(ModelCache.cache_key))
        )
        total_entries = total_result.scalar_one()

        # Active (non-expired) entries
        active_result = await self.session.execute(
            select(func.count(ModelCache.cache_key))
            .where(ModelCache.expires_at > datetime.utcnow())
        )
        active_entries = active_result.scalar_one()

        # Total hits
        hits_result = await self.session.execute(
            select(func.sum(ModelCache.hit_count))
        )
        total_hits = hits_result.scalar_one() or 0

        # Entries by model
        model_result = await self.session.execute(
            select(
                ModelCache.model_name,
                func.count(ModelCache.cache_key)
            )
            .where(ModelCache.expires_at > datetime.utcnow())
            .group_by(ModelCache.model_name)
        )
        entries_by_model = {model: count for model, count in model_result.all()}

        # Hit rate (entries with at least one hit)
        hit_entries_result = await self.session.execute(
            select(func.count(ModelCache.cache_key))
            .where(
                ModelCache.hit_count > 0,
                ModelCache.expires_at > datetime.utcnow()
            )
        )
        entries_with_hits = hit_entries_result.scalar_one()
        hit_rate = (entries_with_hits / active_entries * 100) if active_entries > 0 else 0

        return {
            "total_entries": total_entries,
            "active_entries": active_entries,
            "expired_entries": total_entries - active_entries,
            "total_hits": total_hits,
            "entries_by_model": entries_by_model,
            "hit_rate_percent": round(hit_rate, 2),
            "avg_hits_per_entry": round(total_hits / active_entries, 2) if active_entries > 0 else 0
        }

    async def get_expiring_soon(
        self,
        hours: int = 24,
        limit: int = 100
    ) -> List[ModelCache]:
        """
        Get cache entries expiring within specified hours.

        Args:
            hours: Number of hours to look ahead
            limit: Maximum number of results

        Returns:
            List of cache entries expiring soon
        """
        expiration_threshold = datetime.utcnow() + timedelta(hours=hours)

        result = await self.session.execute(
            select(ModelCache)
            .where(
                and_(
                    ModelCache.expires_at > datetime.utcnow(),
                    ModelCache.expires_at <= expiration_threshold
                )
            )
            .order_by(ModelCache.expires_at.asc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def count_by_model_name(self) -> dict:
        """
        Get count of cache entries grouped by model name.

        Returns:
            Dictionary mapping model name to count
        """
        result = await self.session.execute(
            select(
                ModelCache.model_name,
                func.count(ModelCache.cache_key)
            )
            .where(ModelCache.expires_at > datetime.utcnow())
            .group_by(ModelCache.model_name)
        )
        return {model: count for model, count in result.all()}

    async def prune_low_hit_entries(
        self,
        max_hit_count: int = 2,
        days_old: int = 7
    ) -> int:
        """
        Delete old cache entries with low hit counts.

        Args:
            max_hit_count: Maximum hit count to consider for deletion
            days_old: Minimum age in days

        Returns:
            Number of entries deleted
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)

        result = await self.session.execute(
            delete(ModelCache)
            .where(
                and_(
                    ModelCache.hit_count <= max_hit_count,
                    ModelCache.created_at < cutoff_date
                )
            )
        )
        return result.rowcount
