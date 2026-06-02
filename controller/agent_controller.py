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
    return await service.handle_chat(payload=payload, user_id=user_id)


@router.post("/chat/stream")
async def chat_stream(
        payload: AgentChatRequest,
        user_id=Depends(get_current_user_id),
        service: AgentService = Depends(get_agent_service),
):
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
