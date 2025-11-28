"""
JWT Authentication for ERCP Protocol.
Provides token-based authentication for API endpoints.
"""

import os
from datetime import datetime, timedelta
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from pydantic import BaseModel

# JWT Configuration
# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError(
        "JWT_SECRET_KEY environment variable is required. "
        "Generate one with: openssl rand -hex 32"
    )

if len(SECRET_KEY) < 32:
    raise RuntimeError(
        f"JWT_SECRET_KEY must be at least 32 characters long. "
        f"Current length: {len(SECRET_KEY)}"
    )

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))  # 24 hours

security = HTTPBearer()


class TokenData(BaseModel):
    """Token payload data model."""

    user_id: Optional[str] = None
    exp: Optional[datetime] = None
    roles: list[str] = []  # User roles from JWT claims


def create_access_token(
    data: dict, expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT access token.

    Args:
        data: Data to encode in token
        expires_delta: Optional expiration time delta

    Returns:
        Encoded JWT token string
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    return encoded_jwt


def verify_token(token: str) -> TokenData:
    """
    Verify and decode a JWT token.

    Args:
        token: JWT token string

    Returns:
        TokenData with decoded payload

    Raises:
        HTTPException: If token is invalid or expired
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")

        if user_id is None:
            raise credentials_exception

        # Extract roles from JWT claims (default to empty list)
        roles = payload.get("roles", [])
        if not isinstance(roles, list):
            roles = []

        token_data = TokenData(
            user_id=user_id,
            exp=payload.get("exp"),
            roles=roles,
        )
        return token_data

    except JWTError:
        raise credentials_exception


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> TokenData:
    """
    Dependency to get current authenticated user.

    Args:
        credentials: Bearer token credentials

    Returns:
        TokenData with user info and roles

    Raises:
        HTTPException: If authentication fails
    """
    token = credentials.credentials
    token_data = verify_token(token)

    return token_data


def require_role(required_role: str):
    """
    Factory function to create role-checking dependencies.

    Args:
        required_role: Role name required for access

    Returns:
        Dependency function that checks for the required role
    """
    async def role_checker(
        token_data: TokenData = Depends(get_current_user),
    ) -> TokenData:
        if required_role not in token_data.roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required role: {required_role}",
            )
        return token_data

    return role_checker


# Admin-only dependency using proper role-based access control
async def get_current_admin_user(
    token_data: TokenData = Depends(get_current_user),
) -> TokenData:
    """
    Dependency for admin-only endpoints.

    Checks for 'admin' role in JWT claims.

    Args:
        token_data: Token data from authenticated user

    Returns:
        TokenData for admin user

    Raises:
        HTTPException: If user does not have admin role
    """
    if "admin" not in token_data.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required. User does not have admin role.",
        )

    return token_data
