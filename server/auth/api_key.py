"""
API Key Authentication Module
Author: ERCP Protocol Implementation
License: Apache-2.0

Provides simple API key authentication for services and automation.
"""

import os
import logging
from typing import Optional

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader


logger = logging.getLogger(__name__)

# API Key header name
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Load API keys from environment
API_KEYS_ENV = os.getenv("API_KEYS", "")
VALID_API_KEYS = set([key.strip() for key in API_KEYS_ENV.split(",") if key.strip()])

if not VALID_API_KEYS:
    logger.warning("No API keys configured. API key authentication will reject all requests.")
else:
    logger.info(f"Loaded {len(VALID_API_KEYS)} valid API keys")


def validate_api_key(api_key: str) -> bool:
    """
    Validate an API key.

    Args:
        api_key: The API key to validate

    Returns:
        True if valid, False otherwise
    """
    if not api_key:
        return False

    is_valid = api_key in VALID_API_KEYS

    if is_valid:
        logger.debug(f"Valid API key: {api_key[:8]}...")
    else:
        logger.warning(f"Invalid API key attempt: {api_key[:8] if len(api_key) >= 8 else api_key}")

    return is_valid


def get_api_key(api_key: str = Security(api_key_header)) -> str:
    """
    FastAPI dependency to validate API key from request header.

    Args:
        api_key: API key from X-API-Key header

    Returns:
        The valid API key

    Raises:
        HTTPException: If API key is missing or invalid
    """
    if not api_key:
        logger.warning("API key missing from request")
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Include X-API-Key header."
        )

    if not validate_api_key(api_key):
        logger.warning(f"Invalid API key: {api_key[:8]}...")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )

    return api_key


def optional_api_key(api_key: Optional[str] = Security(api_key_header)) -> Optional[str]:
    """
    Optional API key dependency.
    Returns API key if provided and valid, None otherwise.

    Args:
        api_key: Optional API key from header

    Returns:
        API key if valid, None otherwise
    """
    if not api_key:
        return None

    if validate_api_key(api_key):
        return api_key

    return None
