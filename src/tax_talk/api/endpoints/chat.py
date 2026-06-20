"""Chat endpoints for synchronous and streaming responses."""

from __future__ import annotations

from collections.abc import AsyncIterable

from fastapi import APIRouter, Depends
from fastapi.sse import EventSourceResponse, ServerSentEvent

from tax_talk.api.dependencies.auth import get_current_user
from tax_talk.api.dependencies.services import get_chat_service
from tax_talk.api.schemas.chat import ChatRequest, ChatResponse
from tax_talk.api.schemas.errors import ErrorResponse
from tax_talk.api.services.chat_services import ChatService

router = APIRouter(tags=["chat"])
_chat_service_dependency = Depends(get_chat_service)
_current_user_dependency = Depends(get_current_user)


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={401: {"model": ErrorResponse}},
)
async def chat(
    request: ChatRequest,
    service: ChatService = _chat_service_dependency,
    current_user: dict = _current_user_dependency,
) -> ChatResponse:
    """Return a grounded chat answer."""
    return await service.answer(request, current_user=current_user)


@router.post(
    "/chat/stream",
    response_class=EventSourceResponse,
    responses={401: {"model": ErrorResponse}},
)
async def chat_stream(
    request: ChatRequest,
    service: ChatService = _chat_service_dependency,
    current_user: dict = _current_user_dependency,
) -> AsyncIterable[ServerSentEvent]:
    """Stream chat tokens as Server-Sent Events."""

    event_id = 0
    async for event in service.stream_answer(request, current_user=current_user):
        event_id += 1
        yield ServerSentEvent(
            id=str(event_id),
            event=event.event,
            data=event.model_dump(exclude={"id", "event"}, exclude_none=True),
        )
