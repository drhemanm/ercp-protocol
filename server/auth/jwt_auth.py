"""
JWT Authentication Module
Author: ERCP Protocol Implementation
License: Apache-2.0

Provides JWT token-based authentication for the ERCP server.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt


logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer()

# JWT Configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))

# Validate that JWT_SECRET_KEY is set
if not JWT_SECRET_KEY:
    logger.error("JWT_SECRET_KEY not set in environment variables!")
    raise ValueError(
        "JWT_SECRET_KEY must be set in environment variables. "
        "Generate one with: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
    )

if JWT_SECRET_KEY == "CHANGE_THIS_IN_PRODUCTION":
    logger.warning(
        "JWT_SECRET_KEY is set to default value! "
        "This is insecure for production use."
    )


def create_access_token(
    user_id: str,
    expires_delta: Optional[timedelta] = None,
    additional_claims: Optional[Dict[str, Any]] = None
) -> str:
    """
    Create a JWT access token.

    Args:
        user_id: Unique identifier for the user
        expires_delta: Optional custom expiration time
        additional_claims: Optional additional claims to include in token

    Returns:
        Encoded JWT token string
    """
    if expires_delta is None:
        expires_delta = timedelta(hours=JWT_EXPIRATION_HOURS)

    expire = datetime.utcnow() + expires_delta

    to_encode = {
        "sub": user_id,
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    }

    # Add additional claims if provided
    if additional_claims:
        to_encode.update(additional_claims)

    try:
        encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        logger.debug(f"Created JWT token for user: {user_id}")
        return encoded_jwt
    except Exception as e:
        logger.error(f"Failed to create JWT token: {str(e)}")
        raise


def decode_token(token: str) -> Dict[str, Any]:
    """
    Decode and validate a JWT token.

    Args:
        token: JWT token string

    Returns:
        Dictionary containing token payload

    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])

        # Verify token type
        if payload.get("type") != "access":
            raise HTTPException(
                status_code=401,
                detail="Invalid token type"
            )

        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=401,
                detail="Invalid token: missing user ID"
            )

        logger.debug(f"Successfully decoded token for user: {user_id}")
        return payload

    except JWTError as e:
        logger.warning(f"JWT validation failed: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail=f"Could not validate credentials: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error(f"Token decode error: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> Dict[str, Any]:
    """
    FastAPI dependency to get the current authenticated user from JWT token.

    Args:
        credentials: HTTP Bearer credentials from request header

    Returns:
        Dictionary containing user information from token payload

    Raises:
        HTTPException: If authentication fails
    """
    token = credentials.credentials
    payload = decode_token(token)
    return payload


def optional_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security)
) -> Optional[Dict[str, Any]]:
    """
    Optional authentication dependency.
    Returns user info if token is provided and valid, None otherwise.

    Args:
        credentials: Optional HTTP Bearer credentials

    Returns:
        User info dict if authenticated, None otherwise
    """
    if credentials is None:
        return None

    try:
        return decode_token(credentials.credentials)
    except HTTPException:
        # Invalid token, but don't raise error for optional auth
        return None
