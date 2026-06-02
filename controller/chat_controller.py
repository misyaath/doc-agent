from fastapi import APIRouter, Depends, status

from middleware.auth_middleware import get_current_user_id
from schemas.chat import ChatCreateResponse
from services.chat_service import ChatService, get_chat_service

router = APIRouter(prefix="/chats", tags=["chats"])


@router.post(
    "",
    response_model=ChatCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create chat",
    description="Creates a new chat for the authenticated user and returns user and chat identifiers.",
)
async def create_chat(
    service: ChatService = Depends(get_chat_service),
    user_id: int = Depends(get_current_user_id),
) -> ChatCreateResponse:
    """
    Create chat.

    Purpose:
        Implements create_chat for the HTTP controller layer that validates incoming
            requests, delegates to services, and shapes API responses.
    Args:
        service (ChatService): Injected service dependency that performs the business
            operation.
        user_id (int): Authenticated user identifier used to scope the operation.
    Returns:
        ChatCreateResponse: API response model returned to the client.
    Why Added:
        Provides a documented entry point for this module-level behavior and keeps
            callers
        from needing to know lower-level implementation details.
    """
    return await service.create_chat(user_id=user_id)
