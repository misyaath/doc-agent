from fastapi import APIRouter, Depends, Response, status

from middleware.auth_middleware import get_current_user_id
from schemas.agent import AgentChatRequest
from services.agent_service import AgentService, get_agent_service

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post(
    "/chat",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Agent chat request",
    description="Checks chat ownership and warms Redis cache with file title/summary data for the chat.",
)
async def agent_chat(
        payload: AgentChatRequest,
        service: AgentService = Depends(get_agent_service),
        user_id: int = Depends(get_current_user_id),
) -> Response:
    await service.handle_chat(payload=payload, user_id=user_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
