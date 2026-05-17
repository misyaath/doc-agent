from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.user import User


class UserRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def get_by_email(self, email: str) -> User | None:
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
