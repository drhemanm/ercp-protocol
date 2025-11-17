"""
CORS Configuration Middleware
Author: ERCP Protocol Implementation
License: Apache-2.0

Configures Cross-Origin Resource Sharing (CORS) for the ERCP server.
"""

import os
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


logger = logging.getLogger(__name__)


def setup_cors(app: FastAPI) -> None:
    """
    Configure CORS middleware for the FastAPI application.

    Args:
        app: FastAPI application instance
    """
    # Get allowed origins from environment
    cors_origins_env = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8000")
    allowed_origins = [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()]

    # Special case: allow all origins if * is specified (NOT recommended for production)
    if "*" in allowed_origins:
        logger.warning("CORS configured to allow ALL origins (*). This is insecure for production!")
        allowed_origins = ["*"]

    logger.info(f"CORS configured with allowed origins: {allowed_origins}")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,  # Allow cookies and authorization headers
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        allow_headers=[
            "Content-Type",
            "Authorization",
            "X-API-Key",
            "X-Requested-With",
            "Accept",
            "Origin"
        ],
        expose_headers=[
            "Content-Type",
            "X-Total-Count",  # For pagination
            "X-Rate-Limit-Limit",
            "X-Rate-Limit-Remaining",
            "X-Rate-Limit-Reset"
        ],
        max_age=3600,  # Cache preflight requests for 1 hour
    )

    logger.info("CORS middleware configured successfully")
