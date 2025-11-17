"""
Input sanitization middleware for ERCP Protocol.
Prevents injection attacks and malicious inputs.
"""

import re
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class SanitizationMiddleware(BaseHTTPMiddleware):
    """
    Middleware to sanitize and validate incoming requests.
    Prevents common injection attacks.
    """

    # Dangerous patterns to detect
    DANGEROUS_PATTERNS = [
        # Prompt injection attempts
        r"ignore\s+previous\s+instructions",
        r"disregard\s+all\s+prior",
        r"forget\s+everything",
        r"system\s*:",
        r"<\s*script\s*>",
        r"javascript\s*:",
        # Code injection
        r"eval\s*\(",
        r"exec\s*\(",
        r"__import__",
        r"subprocess",
        # SQL injection (basic)
        r"'\s*OR\s+'1'\s*=\s*'1",
        r";\s*DROP\s+TABLE",
        r"UNION\s+SELECT",
    ]

    async def dispatch(self, request: Request, call_next):
        """
        Process request and sanitize input.

        Args:
            request: Incoming request
            call_next: Next middleware/route handler

        Returns:
            Response or error
        """
        # Only check POST/PUT requests with body
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                # Read body
                body = await request.body()
                body_str = body.decode("utf-8")

                # Check for dangerous patterns
                for pattern in self.DANGEROUS_PATTERNS:
                    if re.search(pattern, body_str, re.IGNORECASE):
                        return JSONResponse(
                            status_code=400,
                            content={
                                "error": "invalid_input",
                                "message": "Potentially malicious input detected",
                            },
                        )

                # Check body size (prevent DoS)
                max_body_size = 10 * 1024 * 1024  # 10 MB
                if len(body) > max_body_size:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "error": "payload_too_large",
                            "message": f"Request body exceeds maximum size of {max_body_size} bytes",
                        },
                    )

            except Exception as e:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "invalid_request",
                        "message": "Failed to parse request body",
                    },
                )

        # Continue processing
        response = await call_next(request)
        return response
