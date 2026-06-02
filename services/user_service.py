from __future__ import annotations

from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from repositories.user_repository import UserRepository
from schemas.user import TokenResponse, UserLoginRequest, UserRegisterRequest, UserRegisterResponse
from services.auth_service import create_access_token, hash_password, verify_password


class UserService:
    """
    User Service.

    Purpose:
        Defines UserService in the business-service layer that coordinates repositories,
            security helpers, RAG execution, and API workflows.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def __init__(self, user_repository: UserRepository) -> None:
        """
        Initialize the object with its required dependencies.

        Purpose:
            Implements __init__ for the business-service layer that coordinates
                repositories, security helpers, RAG execution, and API workflows.
        Class:
            Belongs to UserService; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            user_repository (UserRepository): Input value for the user repository
                parameter.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside UserService so related code remains
                cohesive and testable.
        """
        self._user_repository = user_repository

    async def register(self, payload: UserRegisterRequest) -> UserRegisterResponse:
        """
        Register.

        Purpose:
            Implements register for the business-service layer that coordinates
                repositories, security helpers, RAG execution, and API workflows.
        Class:
            Belongs to UserService; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            payload (UserRegisterRequest): Validated request payload supplied by the API
                caller.
        Returns:
            UserRegisterResponse: API response model returned to the client.
        Why Added:
            Centralizes this behavior inside UserService so related code remains
                cohesive and testable.
        """
        existing = await self._user_repository.get_by_email(payload.email)
        if existing:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")

        password_hash = hash_password(payload.password)
        user = await self._user_repository.create(
            full_name=payload.full_name.strip(),
            email=payload.email,
            password_hash=password_hash,
            is_active=True,
            is_verified=False,
        )
        return UserRegisterResponse(
            id=user.id,
            full_name=user.full_name,
            email=user.email,
            is_active=user.is_active,
            is_verified=user.is_verified,
        )

    async def login(self, payload: UserLoginRequest) -> TokenResponse:
        """
        Login.

        Purpose:
            Implements login for the business-service layer that coordinates
                repositories, security helpers, RAG execution, and API workflows.
        Class:
            Belongs to UserService; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            payload (UserLoginRequest): Validated request payload supplied by the API
                caller.
        Returns:
            TokenResponse: API response model returned to the client.
        Why Added:
            Centralizes this behavior inside UserService so related code remains
                cohesive and testable.
        """
        user = await self._user_repository.get_by_email(payload.email)
        if not user or not verify_password(payload.password, user.password_hash):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

        token, expires_in = create_access_token(str(user.id))
        return TokenResponse(access_token=token, expires_in=expires_in)


def get_user_service(db: AsyncSession = Depends(get_db)) -> UserService:
    """
    Get user service.

    Purpose:
        Implements get_user_service for the business-service layer that coordinates
            repositories, security helpers, RAG execution, and API workflows.
    Args:
        db (AsyncSession): Database session used to read or persist application records.
    Returns:
        UserService: Domain or persistence object produced by the operation.
    Why Added:
        Provides a documented entry point for this module-level behavior and keeps
            callers
        from needing to know lower-level implementation details.
    """
    return UserService(user_repository=UserRepository(db))
