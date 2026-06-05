from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from models.chat import Chat
from models.file import File as FileModel


class ChatRepository:
    """
    Chat Repository.

    Purpose:
        Defines ChatRepository in the repository layer that isolates database
            persistence from higher-level business logic.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def __init__(self, db: AsyncSession) -> None:
        """
        Initialize the object with its required dependencies.

        Purpose:
            Implements __init__ for the repository layer that isolates database
                persistence from higher-level business logic.
        Class:
            Belongs to ChatRepository; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            db (AsyncSession): Database session used to read or persist application
                records.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside ChatRepository so related code remains
                cohesive and testable.
        """
        self._db = db

    async def get_by_id_and_user_id(self, chat_id: str, user_id: int) -> Chat | None:
        """
        Get by id and user id.

        Purpose:
            Implements get_by_id_and_user_id for the repository layer that isolates
                database persistence from higher-level business logic.
        Class:
            Belongs to ChatRepository; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            chat_id (str): Chat/session identifier used to scope retrieval, tasks, or
                responses.
            user_id (int): Authenticated user identifier used to scope the operation.
        Returns:
            Chat | None: Domain or persistence object produced by the operation.
        Why Added:
            Centralizes this behavior inside ChatRepository so related code remains
                cohesive and testable.
        """
        return await self._db.scalar(select(Chat).where(Chat.id == chat_id, Chat.user_id == user_id))

    async def get_by_id_and_user_id_with_files(self, *, chat_id: str, user_id: int) -> Chat | None:
        """
        Get by id and user id with files.

        Purpose:
            Implements get_by_id_and_user_id_with_files for the repository layer that
                isolates database persistence from higher-level business logic.
        Class:
            Belongs to ChatRepository; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            chat_id (str): Chat/session identifier used to scope retrieval, tasks, or
                responses.
            user_id (int): Authenticated user identifier used to scope the operation.
        Returns:
            Chat | None: Domain or persistence object produced by the operation.
        Why Added:
            Centralizes this behavior inside ChatRepository so related code remains
                cohesive and testable.
        """
        return await self._db.scalar(
            select(Chat)
            .where(Chat.id == chat_id, Chat.user_id == user_id)
            .options(selectinload(Chat.files).selectinload(FileModel.process_stages))
        )

    async def create(self, *, chat_id: str, user_id: int, name: str) -> Chat:
        """
        Create.

        Purpose:
            Implements create for the repository layer that isolates database
                persistence from higher-level business logic.
        Class:
            Belongs to ChatRepository; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            chat_id (str): Chat/session identifier used to scope retrieval, tasks, or
                responses.
            user_id (int): Authenticated user identifier used to scope the operation.
            name (str): Human-readable chat name supplied by the user.
        Returns:
            Chat: Domain or persistence object produced by the operation.
        Why Added:
            Centralizes this behavior inside ChatRepository so related code remains
                cohesive and testable.
        """
        chat = Chat(id=chat_id, user_id=user_id, name=name)
        self._db.add(chat)
        await self._db.commit()
        await self._db.refresh(chat)
        return chat

    async def list_by_user_id_with_files(self, user_id: int) -> list[Chat]:
        """
        List by user id with files.

        Purpose:
            Implements list_by_user_id_with_files for the repository layer that isolates
                database persistence from higher-level business logic.
        Class:
            Belongs to ChatRepository; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            user_id (int): Authenticated user identifier used to scope the operation.
        Returns:
            list[Chat]: Domain or persistence objects produced by the operation.
        Why Added:
            Centralizes this behavior inside ChatRepository so related code remains
                cohesive and testable.
        """
        result = await self._db.execute(
            select(Chat)
            .where(Chat.user_id == user_id)
            .options(selectinload(Chat.files).selectinload(FileModel.process_stages))
            .order_by(Chat.created_at.desc())
        )
        return list(result.scalars().all())

    async def delete(self, chat: Chat) -> None:
        """
        Delete.

        Purpose:
            Implements delete for the repository layer that isolates database
                persistence from higher-level business logic.
        Class:
            Belongs to ChatRepository; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            chat (Chat): Chat ORM object selected for deletion.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside ChatRepository so related code remains
                cohesive and testable.
        """
        await self._db.delete(chat)
        await self._db.commit()
