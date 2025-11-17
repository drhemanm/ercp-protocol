"""
Logging Middleware
Author: ERCP Protocol Implementation
License: Apache-2.0

Logs all HTTP requests and responses with timing information.
"""

import time
import logging
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log all HTTP requests and responses.
    
    Logs:
    - Request method, path, query params
    - Response status code
    - Request duration
    - User agent
    - Client IP
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and log details.

        Args:
            request: FastAPI request
            call_next: Next middleware/handler

        Returns:
            Response from next handler
        """
        # Start timer
        start_time = time.time()

        # Extract request info
        method = request.method
        path = request.url.path
        query_params = str(request.query_params) if request.query_params else ""
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")

        # Log request
        logger.info(
            f"Request started: {method} {path} "
            f"from {client_ip} "
            f"{f'?{query_params}' if query_params else ''}"
        )

        try:
            # Process request
            response = await call_next(request)

            # Calculate duration
            duration = time.time() - start_time
            duration_ms = int(duration * 1000)

            # Log response
            logger.info(
                f"Request completed: {method} {path} "
                f"status={response.status_code} "
                f"duration={duration_ms}ms"
            )

            # Add custom headers for observability
            response.headers["X-Process-Time"] = str(duration_ms)

            return response

        except Exception as e:
            # Calculate duration even for errors
            duration = time.time() - start_time
            duration_ms = int(duration * 1000)

            # Log error
            logger.error(
                f"Request failed: {method} {path} "
                f"error={str(e)} "
                f"duration={duration_ms}ms",
                exc_info=True
            )

            # Re-raise exception to be handled by FastAPI
            raise
