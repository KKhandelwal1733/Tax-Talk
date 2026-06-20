"""Authentication dependency helpers for API routes."""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from tax_talk.core.config import settings

_scheme = HTTPBearer(auto_error=False)
_credentials_dependency = Depends(_scheme)


def _decode_jwt_payload(token: str) -> dict:
    """Decode JWT payload without signature validation for lightweight auth checks."""
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Invalid JWT format")

    payload_segment = parts[1]
    padding = "=" * (-len(payload_segment) % 4)
    raw = base64.urlsafe_b64decode(payload_segment + padding)
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JWT payload must be an object")
    return payload


def _validate_lightweight_claims(payload: dict) -> None:
    """Validate lightweight JWT claims configured for API auth checks."""
    exp = payload.get("exp")
    if exp is not None:
        if not isinstance(exp, int | float):
            raise ValueError("JWT exp claim must be numeric")
        if datetime.now(UTC).timestamp() >= float(exp):
            raise ValueError("JWT token is expired")

    expected_issuer = settings.supabase_jwt_issuer.strip()
    if expected_issuer:
        if payload.get("iss") != expected_issuer:
            raise ValueError("JWT issuer claim is invalid")

    expected_audience = settings.supabase_jwt_audience.strip()
    if expected_audience:
        audience = payload.get("aud")
        if isinstance(audience, str):
            valid_audience = audience == expected_audience
        elif isinstance(audience, list):
            valid_audience = expected_audience in audience
        else:
            valid_audience = False
        if not valid_audience:
            raise ValueError("JWT audience claim is invalid")


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = _credentials_dependency,
) -> dict:
    """Return minimal current user claims extracted from a Supabase JWT bearer token."""
    if credentials is None or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Missing bearer token")

    token = credentials.credentials
    try:
        payload = _decode_jwt_payload(token)
        _validate_lightweight_claims(payload)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="Invalid bearer token") from exc

    return {
        "sub": payload.get("sub", ""),
        "role": payload.get("role", ""),
        "raw_claims": payload,
    }
