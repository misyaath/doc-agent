from __future__ import annotations

from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from repositories.user_repository import UserRepository
from schemas.user import TokenResponse, UserLoginRequest, UserRegisterRequest, UserRegisterResponse
from services.auth_service import create_access_token, hash_password, verify_password


class UserService:
    def __init__(self, user_repository: UserRepository) -> None:
        self._user_repository = user_repository

    async def register(self, payload: UserRegisterRequest) -> UserRegisterResponse:
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
        user = await self._user_repository.get_by_email(payload.email)
        if not user or not verify_password(payload.password, user.password_hash):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

        token, expires_in = create_access_token(str(user.id))
        return TokenResponse(access_token=token, expires_in=expires_in)


def get_user_service(db: AsyncSession = Depends(get_db)) -> UserService:
    return UserService(user_repository=UserRepository(db))
