"""
Example usage of database timeout features.
"""

from server.db.database import get_db, QueryTimeout
from sqlalchemy import text


async def example_with_custom_timeout():
    """Example: Run a query with custom timeout."""
    async for db in get_db():
        # Use custom 30-second timeout for this specific query
        async with QueryTimeout(db, timeout_seconds=30):
            result = await db.execute(
                text("SELECT * FROM large_table WHERE complex_condition")
            )
            data = result.fetchall()
        
        return data


async def example_long_running_query():
    """Example: Handle potential timeout gracefully."""
    from sqlalchemy.exc import OperationalError
    
    async for db in get_db():
        try:
            # This query might timeout
            result = await db.execute(
                text("SELECT COUNT(*) FROM traces")
            )
            count = result.scalar()
            return {"count": count}
        
        except OperationalError as e:
            if "timeout" in str(e).lower():
                # Query timed out
                return {"error": "Query timeout", "message": "Operation took too long"}
            else:
                raise
