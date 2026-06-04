from fastapi import APIRouter, Depends, status

from middleware.auth_middleware import get_current_user_id
from schemas.chat import ChatCreateRequest, ChatCreateResponse, ChatListResponse
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
    payload: ChatCreateRequest | None = None,
    service: ChatService = Depends(get_chat_service),
    user_id: int = Depends(get_current_user_id),
) -> ChatCreateResponse:
    """
    Create chat.

    Purpose:
        Implements create_chat for the HTTP controller layer that validates incoming
            requests, delegates to services, and shapes API responses.
    Args:
        payload (ChatCreateRequest | None): Validated request payload supplied by the
            API caller.
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
    request = payload or ChatCreateRequest()
    return await service.create_chat(user_id=user_id, name=request.name)


@router.get(
    "",
    response_model=ChatListResponse,
    status_code=status.HTTP_200_OK,
    summary="List chats",
    description="Lists chats for the authenticated user with file metadata and processing status.",
)
async def list_chats(
    service: ChatService = Depends(get_chat_service),
    user_id: int = Depends(get_current_user_id),
) -> ChatListResponse:
    """
    List chats.

    Purpose:
        Implements list_chats for the HTTP controller layer that validates incoming
            requests, delegates to services, and shapes API responses.
    Args:
        service (ChatService): Injected service dependency that performs the business
            operation.
        user_id (int): Authenticated user identifier used to scope the operation.
    Returns:
        ChatListResponse: API response model returned to the client.
    Why Added:
        Provides a documented entry point for this module-level behavior and keeps
            callers
        from needing to know lower-level implementation details.
    """
    return await service.list_chats(user_id=user_id)
