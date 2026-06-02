from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.user import User


class UserRepository:
    """
    User Repository.

    Purpose:
        Defines UserRepository in the repository layer that isolates database
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
            Belongs to UserRepository; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            db (AsyncSession): Database session used to read or persist application
                records.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside UserRepository so related code remains
                cohesive and testable.
        """
        self._db = db

    async def get_by_email(self, email: str) -> User | None:
        """
        Get by email.

        Purpose:
            Implements get_by_email for the repository layer that isolates database
                persistence from higher-level business logic.
        Class:
            Belongs to UserRepository; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            email (str): Input value for the email parameter.
        Returns:
            User | None: Domain or persistence object produced by the operation.
        Why Added:
            Centralizes this behavior inside UserRepository so related code remains
                cohesive and testable.
        """
        return await self._db.scalar(select(User).where(User.email == email))

    async def create(
        self,
        *,
        full_name: str,
        email: str,
        password_hash: str,
        is_active: bool = True,
        is_verified: bool = False,
    ) -> User:
        """
        Create.

        Purpose:
            Implements create for the repository layer that isolates database
                persistence from higher-level business logic.
        Class:
            Belongs to UserRepository; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            full_name (str): Input value for the full name parameter.
            email (str): Input value for the email parameter.
            password_hash (str): Stored password hash used for credential verification.
            is_active (bool): Input value for the is active parameter.
            is_verified (bool): Input value for the is verified parameter.
        Returns:
            User: Domain or persistence object produced by the operation.
        Why Added:
            Centralizes this behavior inside UserRepository so related code remains
                cohesive and testable.
        """
        user = User(
            full_name=full_name,
            email=email,
            password_hash=password_hash,
            is_active=is_active,
            is_verified=is_verified,
        )
        self._db.add(user)
        await self._db.commit()
        await self._db.refresh(user)
        return user
