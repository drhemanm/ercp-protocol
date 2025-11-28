"""
CORS middleware configuration for ERCP Protocol.
"""

import os
from fastapi.middleware.cors import CORSMiddleware


def get_cors_origins():
    """
    Get allowed CORS origins from environment.

    Returns:
        List of allowed origins
    """
    origins_env = os.getenv(
        "CORS_ORIGINS", "http://localhost:3000,http://localhost:8080"
    )
    origins = [origin.strip() for origin in origins_env.split(",")]
    return origins


def get_cors_headers():
    """
    Get allowed CORS headers from environment.

    Returns:
        List of allowed headers
    """
    default_headers = [
        "Authorization",
        "Content-Type",
        "X-Request-ID",
        "Accept",
        "Origin",
        "X-Requested-With",
    ]
    headers_env = os.getenv("CORS_ALLOWED_HEADERS", "")
    if headers_env:
        custom_headers = [h.strip() for h in headers_env.split(",") if h.strip()]
        return default_headers + custom_headers
    return default_headers


def add_cors_middleware(app):
    """
    Add CORS middleware to FastAPI app.

    Args:
        app: FastAPI application instance
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_cors_origins(),
        allow_credentials=os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower()
        == "true",
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=get_cors_headers(),
        expose_headers=["X-Total-Count", "X-Page-Number", "X-Request-ID"],
    )
