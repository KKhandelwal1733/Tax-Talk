"""Authentication dependency helpers for API routes."""

from __future__ import annotations

from functools import lru_cache

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient

from tax_talk.core.config import settings

security = HTTPBearer(auto_error=False)
_credentials_dependency = Depends(security)


@lru_cache(maxsize=1)
def get_jwks_client() -> PyJWKClient:
    """Get cached JWKS client for Supabase JWT signature verification."""
    jwks_url = f"{settings.supabase_url.strip()}/auth/v1/.well-known/jwks.json"
    return PyJWKClient(jwks_url)


def verify_supabase_token(token: str) -> dict:
    """Verify and decode a Supabase JWT with full signature validation.

    Args:
        token: Bearer token string to verify.

    Returns:
        Decoded JWT payload as dict.

    Raises:
        HTTPException: On expired, malformed, or invalid signature.
    """
    token = token.strip()
    try:
        signing_key = get_jwks_client().get_signing_key_from_jwt(token)

        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["ES256"],
            audience=settings.supabase_jwt_audience,
            issuer=settings.supabase_jwt_issuer,
        )

        return payload

    except jwt.ExpiredSignatureError as err:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
        ) from err

    except jwt.InvalidTokenError as err:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        ) from err


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = _credentials_dependency,
) -> dict:
    """Extract and verify current user from Supabase JWT bearer token.

    Args:
        credentials: HTTPBearer credentials from request.

    Returns:
        User info dict with user_id, email, role, and full claims.

    Raises:
        HTTPException: 401 if token missing or invalid.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    payload = verify_supabase_token(credentials.credentials)

    return {
        "user_id": payload.get("sub"),
        "email": payload.get("email"),
        "role": payload.get("role"),
        "claims": payload,
    }


async def get_optional_user(
    credentials: HTTPAuthorizationCredentials | None = _credentials_dependency,
) -> dict | None:
    """Extract and verify current user, but return None if token is missing or invalid.

    Args:
        credentials: HTTPBearer credentials from request.

    Returns:
        User info dict, or None if no valid token.
    """
    if credentials is None:
        return None

    try:
        payload = verify_supabase_token(credentials.credentials)

        return {
            "user_id": payload.get("sub"),
            "email": payload.get("email"),
            "role": payload.get("role"),
            "claims": payload,
        }
    except HTTPException:
        return None
