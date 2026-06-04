import uuid
from datetime import datetime

from sqlalchemy import BigInteger, DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base


class Chat(Base):
    """
    Chat.

    Purpose:
        Defines Chat in the SQLAlchemy model layer that maps application entities to
            database tables.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.

    Attributes:
        id (Mapped[str]): Declared data field for this class.
        name (Mapped[str]): Declared data field for this class.
        user_id (Mapped[int]): Declared data field for this class.
        created_at (Mapped[datetime]): Declared data field for this class.
        updated_at (Mapped[datetime | None]): Declared data field for this class.
        user (Any): Class-level value used by this class.
        files (Any): Class-level value used by this class.
    """

    __tablename__ = "chats"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(120), nullable=False, default="New Chat", server_default="New Chat")
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )

    user = relationship("User", back_populates="chats")
    files = relationship("File", back_populates="chat", cascade="all, delete-orphan")
