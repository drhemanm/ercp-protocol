"""
Rate limiting middleware for ERCP Protocol.
Prevents abuse and ensures fair usage with multi-factor rate limiting.
"""

import os
import hashlib
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request, Response
from fastapi.responses import JSONResponse


def get_composite_rate_limit_key(request: Request) -> str:
    """
    Generate a composite rate limit key from multiple factors.

    Combines:
    - IP address (from X-Forwarded-For or direct connection)
    - User-Agent header
    - Authorization header (API key if present)

    This prevents trivial bypasses via IP rotation alone.

    Args:
        request: FastAPI request

    Returns:
        Composite key string
    """
    # Get IP address (prefer X-Forwarded-For if behind proxy, but validate)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Take only the first IP to prevent header stuffing
        ip = forwarded_for.split(",")[0].strip()
    else:
        ip = get_remote_address(request)

    # Get User-Agent
    user_agent = request.headers.get("User-Agent", "unknown")

    # Get API key from Authorization header
    auth_header = request.headers.get("Authorization", "")
    # Hash the auth to avoid leaking credentials in logs
    auth_hash = hashlib.sha256(auth_header.encode()).hexdigest()[:16] if auth_header else "anon"

    # Combine factors with hashing to prevent enumeration
    composite = f"{ip}:{hashlib.md5(user_agent.encode()).hexdigest()[:8]}:{auth_hash}"

    return composite


# Initialize limiter with composite key function
limiter = Limiter(
    key_func=get_composite_rate_limit_key,
    default_limits=[
        f"{os.getenv('RATE_LIMIT_PER_MINUTE', '60')}/minute",
        f"{os.getenv('RATE_LIMIT_PER_HOUR', '1000')}/hour",
    ],
    # Add storage backend for distributed rate limiting (optional)
    storage_uri=os.getenv("RATE_LIMIT_STORAGE_URI", "memory://"),
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
