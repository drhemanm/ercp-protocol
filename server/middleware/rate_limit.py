"""
Rate limiting middleware for ERCP Protocol.
Prevents abuse and ensures fair usage.
"""

import os
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request, Response
from fastapi.responses import JSONResponse

# Initialize limiter
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[
        f"{os.getenv('RATE_LIMIT_PER_MINUTE', '60')}/minute",
        f"{os.getenv('RATE_LIMIT_PER_HOUR', '1000')}/hour",
    ],
)


def custom_rate_limit_handler(request: Request, exc: RateLimitExceeded) -> Response:
    """
    Custom handler for rate limit exceeded errors.

    Args:
        request: FastAPI request
        exc: Rate limit exception

    Returns:
        JSON error response
    """
    return JSONResponse(
        status_code=429,
        content={
            "error": "rate_limit_exceeded",
            "message": "Too many requests. Please slow down.",
            "detail": str(exc.detail),
        },
    )


# Export for use in main app
__all__ = ["limiter", "custom_rate_limit_handler", "RateLimitExceeded"]
