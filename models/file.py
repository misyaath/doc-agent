import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base


class File(Base):
    """
    File.

    Purpose:
        Defines File in the SQLAlchemy model layer that maps application entities to
            database tables.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.

    Attributes:
        file_id (Mapped[str]): Declared data field for this class.
        chat_id (Mapped[str]): Declared data field for this class.
        file_name (Mapped[str]): Declared data field for this class.
        unique_generated_name (Mapped[str]): Declared data field for this class.
        full_path (Mapped[str]): Declared data field for this class.
        title (Mapped[str | None]): Declared data field for this class.
        summary (Mapped[dict[str, Any] | None]): Declared data field for this class.
        created_at (Mapped[datetime]): Declared data field for this class.
        chat (Any): Class-level value used by this class.
        process_stages (Any): Class-level value used by this class.
    """

    __tablename__ = "files"

    file_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    chat_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("chats.id", ondelete="CASCADE"), nullable=False, index=True
    )
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    unique_generated_name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    full_path: Mapped[str] = mapped_column(String(1024), nullable=False)

    title: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
    )

    summary: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    chat = relationship("Chat", back_populates="files")
    process_stages = relationship("FileProcessStage", back_populates="file", cascade="all, delete-orphan")
