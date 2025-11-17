"""
Database timeout monitoring and alerting.
Tracks slow queries and timeout events.
"""

import time
from typing import Dict, List
from collections import deque
from datetime import datetime
from server.logging import logger


class TimeoutMonitor:
    """
    Monitor database query timeouts and slow queries.
    
    Tracks:
    - Query execution times
    - Timeout events
    - Slow query patterns
    """
    
    def __init__(self, slow_query_threshold: float = 5.0, max_history: int = 100):
        """
        Initialize timeout monitor.
        
        Args:
            slow_query_threshold: Seconds before query is considered slow
            max_history: Maximum number of events to track
        """
        self.slow_query_threshold = slow_query_threshold
        self.max_history = max_history
        
        self.slow_queries: deque = deque(maxlen=max_history)
        self.timeout_events: deque = deque(maxlen=max_history)
        self.query_stats: Dict[str, List[float]] = {}
    
    def record_query(self, query: str, duration: float, timed_out: bool = False):
        """
        Record a query execution.
        
        Args:
            query: SQL query string (truncated)
            duration: Execution time in seconds
            timed_out: Whether the query timed out
        """
        # Truncate query for logging
        query_short = query[:100] + "..." if len(query) > 100 else query
        
        if timed_out:
            self.timeout_events.append({
                "query": query_short,
                "duration": duration,
                "timestamp": datetime.utcnow().isoformat(),
            })
            logger.error(
                "db.query.timeout",
                query=query_short,
                duration=duration,
            )
        elif duration > self.slow_query_threshold:
            self.slow_queries.append({
                "query": query_short,
                "duration": duration,
                "timestamp": datetime.utcnow().isoformat(),
            })
            logger.warning(
                "db.query.slow",
                query=query_short,
                duration=duration,
            )
        
        # Track stats
        if query_short not in self.query_stats:
            self.query_stats[query_short] = []
        self.query_stats[query_short].append(duration)
        
        # Keep only recent stats
        if len(self.query_stats[query_short]) > 50:
            self.query_stats[query_short] = self.query_stats[query_short][-50:]
    
    def get_stats(self) -> dict:
        """
        Get timeout monitoring statistics.
        
        Returns:
            Dictionary with monitoring stats
        """
        return {
            "slow_queries_count": len(self.slow_queries),
            "timeout_events_count": len(self.timeout_events),
            "recent_slow_queries": list(self.slow_queries)[-10:],
            "recent_timeouts": list(self.timeout_events)[-10:],
        }


# Global timeout monitor instance
timeout_monitor = TimeoutMonitor()
