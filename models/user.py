from sqlalchemy import BigInteger, Boolean, DateTime, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base


class User(Base):
    """
    User.

    Purpose:
        Defines User in the SQLAlchemy model layer that maps application entities to
            database tables.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.

    Attributes:
        id (Mapped[int]): Declared data field for this class.
        full_name (Mapped[str]): Declared data field for this class.
        email (Mapped[str]): Declared data field for this class.
        password_hash (Mapped[str]): Declared data field for this class.
        is_active (Mapped[bool]): Declared data field for this class.
        is_verified (Mapped[bool]): Declared data field for this class.
        created_at (Mapped[DateTime]): Declared data field for this class.
        updated_at (Mapped[DateTime]): Declared data field for this class.
        chats (Any): Class-level value used by this class.
    """

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    full_name: Mapped[str] = mapped_column(String(120), nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default="true")
    is_verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
    chats = relationship("Chat", back_populates="user", cascade="all, delete-orphan")
