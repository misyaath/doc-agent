import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base


class File(Base):
    __tablename__ = "files"

    file_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
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
