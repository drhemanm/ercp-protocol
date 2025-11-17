"""
ModelCache Repository - CRUD Operations for ModelCache Model
Author: ERCP Protocol Implementation
License: Apache-2.0

Provides async CRUD operations for ModelCache records.
"""

import logging
import hashlib
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

from sqlalchemy import select, delete, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import func

from ..models import ModelCache


logger = logging.getLogger(__name__)


class ModelCacheRepository:
    """Repository for ModelCache CRUD operations."""

    @staticmethod
    def generate_cache_key(model_name: str, prompt: str) -> str:
        """
        Generate a cache key from model name and prompt.

        Args:
            model_name: Name of the model
            prompt: Input prompt

        Returns:
            Cache key (SHA256 hash)
        """
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        cache_key = f"{model_name}:{prompt_hash}"
        return cache_key

    @staticmethod
    def generate_prompt_hash(prompt: str) -> str:
        """
        Generate a hash of the prompt.

        Args:
            prompt: Input prompt

        Returns:
            SHA256 hash of prompt
        """
        return hashlib.sha256(prompt.encode()).hexdigest()

    @staticmethod
    async def get_cached_output(
        db: AsyncSession,
        model_name: str,
        prompt: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached output for a model/prompt combination.

        Args:
            db: Database session
            model_name: Name of the model
            prompt: Input prompt

        Returns:
            Cached output dict or None if not found or expired
        """
        prompt_hash = ModelCacheRepository.generate_prompt_hash(prompt)

        query = (
            select(ModelCache)
            .where(ModelCache.model_name == model_name)
            .where(ModelCache.prompt_hash == prompt_hash)
        )

        result = await db.execute(query)
        cache_entry = result.scalar_one_or_none()

        if not cache_entry:
            logger.debug(f"Cache miss: {model_name}:{prompt_hash[:16]}...")
            return None

        # Check if expired
        if cache_entry.expires_at and cache_entry.expires_at < datetime.utcnow():
            logger.debug(f"Cache expired: {model_name}:{prompt_hash[:16]}...")
            # Delete expired entry
            await ModelCacheRepository.delete_cache_entry(db, cache_entry.cache_key)
            return None

        # Update hit count and last_accessed
        await db.execute(
            update(ModelCache)
            .where(ModelCache.cache_key == cache_entry.cache_key)
            .values(
                hit_count=ModelCache.hit_count + 1,
                last_accessed=datetime.utcnow()
            )
        )

        logger.debug(f"Cache hit: {model_name}:{prompt_hash[:16]}...")
        return cache_entry.output

    @staticmethod
    async def set_cached_output(
        db: AsyncSession,
        model_name: str,
        prompt: str,
        output: Dict[str, Any],
        ttl_seconds: Optional[int] = None
    ) -> ModelCache:
        """
        Set cached output for a model/prompt combination.

        Args:
            db: Database session
            model_name: Name of the model
            prompt: Input prompt
            output: Output to cache
            ttl_seconds: Time-to-live in seconds (None = no expiry)

        Returns:
            Created ModelCache object
        """
        prompt_hash = ModelCacheRepository.generate_prompt_hash(prompt)
        cache_key = ModelCacheRepository.generate_cache_key(model_name, prompt)

        # Calculate expiration time
        expires_at = None
        if ttl_seconds:
            expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)

        # Check if entry already exists
        existing = await db.execute(
            select(ModelCache).where(ModelCache.cache_key == cache_key)
        )
        existing_entry = existing.scalar_one_or_none()

        if existing_entry:
            # Update existing entry
            await db.execute(
                update(ModelCache)
                .where(ModelCache.cache_key == cache_key)
                .values(
                    output=output,
                    expires_at=expires_at,
                    last_accessed=datetime.utcnow()
                )
            )
            await db.flush()
            cache_entry = existing_entry
            logger.debug(f"Updated cache: {cache_key[:32]}...")
        else:
            # Create new entry
            cache_entry = ModelCache(
                cache_key=cache_key,
                model_name=model_name,
                prompt_hash=prompt_hash,
                output=output,
                expires_at=expires_at,
                hit_count=0
            )
            db.add(cache_entry)
            await db.flush()
            await db.refresh(cache_entry)
            logger.debug(f"Created cache: {cache_key[:32]}...")

        return cache_entry

    @staticmethod
    async def delete_cache_entry(
        db: AsyncSession,
        cache_key: str
    ) -> bool:
        """
        Delete a cache entry.

        Args:
            db: Database session
            cache_key: Cache key to delete

        Returns:
            True if deleted, False if not found
        """
        stmt = delete(ModelCache).where(ModelCache.cache_key == cache_key)
        result = await db.execute(stmt)

        deleted = result.rowcount > 0
        if deleted:
            logger.debug(f"Deleted cache entry: {cache_key[:32]}...")
        
        return deleted

    @staticmethod
    async def clear_expired_cache(
        db: AsyncSession
    ) -> int:
        """
        Clear all expired cache entries.

        Args:
            db: Database session

        Returns:
            Number of entries deleted
        """
        stmt = delete(ModelCache).where(
            ModelCache.expires_at < datetime.utcnow()
        )
        result = await db.execute(stmt)

        count = result.rowcount
        logger.info(f"Cleared {count} expired cache entries")
        return count

    @staticmethod
    async def clear_all_cache(
        db: AsyncSession
    ) -> int:
        """
        Clear all cache entries.

        WARNING: This will delete all cached data!

        Args:
            db: Database session

        Returns:
            Number of entries deleted
        """
        stmt = delete(ModelCache)
        result = await db.execute(stmt)

        count = result.rowcount
        logger.warning(f"Cleared all cache: {count} entries deleted")
        return count

    @staticmethod
    async def get_cache_stats(
        db: AsyncSession
    ) -> Dict[str, Any]:
        """
        Get cache statistics.

        Args:
            db: Database session

        Returns:
            Dictionary with cache statistics
        """
        # Total entries
        total_count = await db.execute(select(func.count(ModelCache.cache_key)))
        total = total_count.scalar()

        # Expired entries
        expired_count = await db.execute(
            select(func.count(ModelCache.cache_key))
            .where(ModelCache.expires_at < datetime.utcnow())
        )
        expired = expired_count.scalar()

        # Total hits
        total_hits = await db.execute(select(func.sum(ModelCache.hit_count)))
        hits = total_hits.scalar() or 0

        return {
            "total_entries": total or 0,
            "expired_entries": expired or 0,
            "active_entries": (total or 0) - (expired or 0),
            "total_hits": hits
        }
