"""Request-level observability middleware."""

from __future__ import annotations

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from langfuse import observe


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Wrap each request in a root trace span."""

    @observe(name="api-request", as_type="span", capture_input=False, capture_output=False)
    async def dispatch(self, request: Request, call_next) -> Response:
        return await call_next(request)
