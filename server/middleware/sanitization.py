"""
Input Sanitization Middleware
Author: ERCP Protocol Implementation
License: Apache-2.0

Prevents prompt injection, XSS, and other malicious input patterns.
"""

import re
import logging
from typing import Any, Dict

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse


logger = logging.getLogger(__name__)


# Dangerous patterns that indicate potential attacks
DANGEROUS_PATTERNS = [
    # Prompt injection attempts
    r"ignore\s+(previous|all|prior)\s+instructions",
    r"disregard\s+(all|everything|prior)",
    r"forget\s+(everything|all|previous)",
    r"you\s+are\s+now",
    r"new\s+instructions",
    r"system:\s*",
    r"assistant:\s*",
    r"<\s*system\s*>",
    r"<\s*assistant\s*>",
    r"\[INST\]",
    r"\[/INST\]",
    
    # Script injection
    r"<\s*script[^>]*>",
    r"<\s*/\s*script\s*>",
    r"javascript:",
    r"onerror\s*=",
    r"onload\s*=",
    
    # Code execution
    r"eval\s*\(",
    r"exec\s*\(",
    r"__import__",
    r"subprocess",
    
    # SQL injection (though ORM protects us)
    r";\s*(drop|delete|insert|update)\s+",
    r"union\s+select",
    r"or\s+1\s*=\s*1",
    
    # Path traversal
    r"\.\./",
    r"\.\.\\",
]

# Compile patterns for efficiency
COMPILED_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in DANGEROUS_PATTERNS]


def check_for_dangerous_patterns(text: str) -> tuple[bool, Optional[str]]:
    """
    Check text for dangerous patterns.

    Args:
        text: Text to check

    Returns:
        Tuple of (is_dangerous, pattern_matched)
    """
    if not isinstance(text, str):
        return False, None

    for pattern in COMPILED_PATTERNS:
        match = pattern.search(text)
        if match:
            return True, match.group(0)

    return False, None


def sanitize_dict(data: Dict[str, Any], path: str = "root") -> None:
    """
    Recursively check dictionary for dangerous patterns.

    Args:
        data: Dictionary to check
        path: Current path in dict (for error messages)

    Raises:
        HTTPException: If dangerous pattern detected
    """
    if not isinstance(data, dict):
        return

    for key, value in data.items():
        current_path = f"{path}.{key}"

        if isinstance(value, str):
            is_dangerous, pattern = check_for_dangerous_patterns(value)
            if is_dangerous:
                logger.warning(
                    f"Blocked dangerous pattern in {current_path}: {pattern}"
                )
                raise HTTPException(
                    status_code=400,
                    detail=f"Potentially malicious input detected in {key}. "
                           f"Pattern: {pattern[:50]}"
                )

        elif isinstance(value, dict):
            sanitize_dict(value, current_path)

        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, str):
                    is_dangerous, pattern = check_for_dangerous_patterns(item)
                    if is_dangerous:
                        logger.warning(
                            f"Blocked dangerous pattern in {current_path}[{i}]: {pattern}"
                        )
                        raise HTTPException(
                            status_code=400,
                            detail=f"Potentially malicious input detected in {key}[{i}]"
                        )
                elif isinstance(item, dict):
                    sanitize_dict(item, f"{current_path}[{i}]")


async def sanitize_input(request: Request, call_next):
    """
    Middleware to sanitize all incoming requests.

    Args:
        request: FastAPI request
        call_next: Next middleware/handler

    Returns:
        Response from next handler

    Raises:
        HTTPException: If malicious input detected
    """
    # Skip sanitization for health check and root endpoints
    if request.url.path in ["/health", "/"]:
        return await call_next(request)

    # Only check POST/PUT/PATCH requests with JSON body
    if request.method in ["POST", "PUT", "PATCH"]:
        try:
            # Get request body
            body = await request.body()

            if body:
                import json
                try:
                    data = json.loads(body)

                    # Check for dangerous patterns
                    sanitize_dict(data)

                    # Reconstruct request with checked body
                    async def receive():
                        return {"type": "http.request", "body": body}

                    request._receive = receive

                except json.JSONDecodeError:
                    # Not JSON, skip sanitization
                    pass

        except Exception as e:
            logger.error(f"Sanitization error: {str(e)}")
            # Don't block on sanitization errors, just log
            pass

    response = await call_next(request)
    return response


def sanitize_string(text: str, field_name: str = "input") -> str:
    """
    Utility function to sanitize a single string.

    Args:
        text: String to sanitize
        field_name: Name of field (for error messages)

    Returns:
        Original text if safe

    Raises:
        HTTPException: If dangerous pattern detected
    """
    is_dangerous, pattern = check_for_dangerous_patterns(text)

    if is_dangerous:
        logger.warning(f"Blocked dangerous pattern in {field_name}: {pattern}")
        raise HTTPException(
            status_code=400,
            detail=f"Potentially malicious input detected in {field_name}"
        )

    return text
