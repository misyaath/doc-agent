import uuid

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from middleware.auth_middleware import get_current_user_id
from models.chat import Chat
from schemas.chat import ChatCreateResponse

router = APIRouter(prefix="/chats", tags=["chats"])


@router.post(
    "",
    response_model=ChatCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create chat",
    description="Creates a new chat for the authenticated user and returns user and chat identifiers.",
)
async def create_chat(
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
) -> ChatCreateResponse:
    chat_id = str(uuid.uuid4())
    chat = Chat(id=chat_id, user_id=user_id)
    db.add(chat)
    await db.commit()
    return ChatCreateResponse(user_id=user_id, chat_id=chat_id)
