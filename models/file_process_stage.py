import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base

ALLOWED_STAGES = {"uploaded", "extracted", "storage", "enrrich", "embeeding", "done"}
ALLOWED_STATUS = {"waiting", "started", "processing", "Done", "failed"}


class FileProcessStage(Base):
    __tablename__ = "file_process_stages"
    __table_args__ = (
        CheckConstraint(
            "stage IN ('uploaded','extracted','storage','enrrich','embeeding','done')",
            name="ck_file_process_stages_stage",
        ),
        CheckConstraint(
            "status IN ('waiting','started','processing','Done','failed')",
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
