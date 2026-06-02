import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base

ALLOWED_STAGES = {"uploaded", "extracted", "normalizer", "enriched", "embedding", "done"}
ALLOWED_STATUS = {"waiting", "started", "processing", "done", "failed"}


class FileProcessStage(Base):
    """
    File Process Stage.

    Purpose:
        Defines FileProcessStage in the SQLAlchemy model layer that maps application
            entities to database tables.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.

    Attributes:
        id (Mapped[str]): Declared data field for this class.
        file_id (Mapped[str]): Declared data field for this class.
        stage (Mapped[str]): Declared data field for this class.
        status (Mapped[str]): Declared data field for this class.
        created_at (Mapped[datetime]): Declared data field for this class.
        updated_at (Mapped[datetime]): Declared data field for this class.
        file (Any): Class-level value used by this class.
    """

    __tablename__ = "file_process_stages"
    __table_args__ = (
        CheckConstraint(
            "stage IN ('uploaded','extracted','normalizer','enriched','embedding','done')",
            name="ck_file_process_stages_stage",
        ),
        CheckConstraint(
            "status IN ('waiting','started','processing','done','failed')",
            name="ck_file_process_stages_status",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    file_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("files.file_id", ondelete="CASCADE"), nullable=False, index=True
    )
    stage: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    file = relationship("File", back_populates="process_stages")
