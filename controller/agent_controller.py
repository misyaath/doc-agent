from fastapi import APIRouter, Depends, status
from fastapi.responses import StreamingResponse

from middleware.auth_middleware import get_current_user_id
from schemas.agent import AgentChatRequest, AgentChatResponse
from services.agent_service import AgentService, get_agent_service

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post(
    "/chat",
    response_model=AgentChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Agent chat request",
    description="Runs the RAG agent for the provided chat and prompt.",
)
async def agent_chat(
    payload: AgentChatRequest,
    service: AgentService = Depends(get_agent_service),
    user_id: int = Depends(get_current_user_id),
) -> AgentChatResponse:
    """
    Agent chat.

    Purpose:
        Implements agent_chat for the HTTP controller layer that validates incoming
            requests, delegates to services, and shapes API responses.
    Args:
        payload (AgentChatRequest): Validated request payload supplied by the API
            caller.
        service (AgentService): Injected service dependency that performs the business
            operation.
        user_id (int): Authenticated user identifier used to scope the operation.
    Returns:
        AgentChatResponse: API response model returned to the client.
    Why Added:
        Provides a documented entry point for this module-level behavior and keeps
            callers
        from needing to know lower-level implementation details.
    """
    return await service.handle_chat(payload=payload, user_id=user_id)


@router.post("/chat/stream")
async def chat_stream(
    payload: AgentChatRequest,
    user_id: int = Depends(get_current_user_id),
    service: AgentService = Depends(get_agent_service),
) -> StreamingResponse:
    """
    Chat stream.

    Purpose:
        Implements chat_stream for the HTTP controller layer that validates incoming
            requests, delegates to services, and shapes API responses.
    Args:
        payload (AgentChatRequest): Validated request payload supplied by the API
            caller.
        user_id (int): Authenticated user identifier used to scope the operation.
        service (AgentService): Injected service dependency that performs the business
            operation.
    Returns:
        StreamingResponse: Streaming response or iterator that yields incremental
            output.
    Why Added:
        Provides a documented entry point for this module-level behavior and keeps
            callers
        from needing to know lower-level implementation details.
    """
    return StreamingResponse(
        service.stream_chat(
            payload=payload,
            user_id=user_id,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
