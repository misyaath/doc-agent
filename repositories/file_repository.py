from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.file import File as FileModel
from models.file_process_stage import FileProcessStage


class FileRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    def add_file_with_stage(
            self,
            *,
            file_id: str,
            chat_id: str,
            file_name: str,
            unique_generated_name: str,
            full_path: str,
            stage: str,
            status: str,
    ) -> FileModel:
        file_record = FileModel(
            file_id=file_id,
            chat_id=chat_id,
            file_name=file_name,
            unique_generated_name=unique_generated_name,
            full_path=full_path,
        )
        stage_record = FileProcessStage(
            file_id=file_id,
            stage=stage,
            status=status,
        )
        self._db.add(file_record)
        self._db.add(stage_record)
        return file_record

    async def commit(self) -> None:
        await self._db.commit()

    async def refresh(self, record: FileModel) -> None:
        await self._db.refresh(record)

    async def get_title_summaries_by_chat_id(self, chat_id: str) -> list[dict]:
        rows = await self._db.execute(
            select(FileModel.file_id, FileModel.title, FileModel.summary).where(FileModel.chat_id == chat_id)
        )
        return [
            {
                "file_id": file_id,
                "title": title,
                "summary": summary,
            }
            for file_id, title, summary in rows.all()
        ]
