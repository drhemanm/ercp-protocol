"""
Rate Limiting Middleware
Author: ERCP Protocol Implementation
License: Apache-2.0

Implements rate limiting per IP/user using slowapi.
"""

import os
import logging

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded


logger = logging.getLogger(__name__)


# Rate limit configuration from environment
RATE_LIMIT_PER_MINUTE = os.getenv("RATE_LIMIT_PER_MINUTE", "10")
RATE_LIMIT_BURST = os.getenv("RATE_LIMIT_BURST", "20")


def get_identifier(request):
    """
    Get identifier for rate limiting.
    
    Uses authenticated user if available, otherwise falls back to IP address.
    
    Args:
        request: FastAPI request
        
    Returns:
        Identifier string for rate limiting
    """
    # Try to get user from JWT or API key
    # For now, use IP address (can be enhanced to use user_id from auth)
    return get_remote_address(request)


# Create limiter instance
limiter = Limiter(
    key_func=get_identifier,
    default_limits=[f"{RATE_LIMIT_PER_MINUTE}/minute"],
    storage_uri=os.getenv("REDIS_URL", "memory://"),  # Use Redis if available, otherwise in-memory
    strategy="fixed-window"  # or "moving-window" for more accurate limiting
)


logger.info(
    f"Rate limiter configured: {RATE_LIMIT_PER_MINUTE}/minute "
    f"(burst: {RATE_LIMIT_BURST})"
)
