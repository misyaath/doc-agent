from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.file import File as FileModel
from models.file_process_stage import FileProcessStage


class FileRepository:
    """
    File Repository.

    Purpose:
        Defines FileRepository in the repository layer that isolates database
            persistence from higher-level business logic.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def __init__(self, db: AsyncSession) -> None:
        """
        Initialize the object with its required dependencies.

        Purpose:
            Implements __init__ for the repository layer that isolates database
                persistence from higher-level business logic.
        Class:
            Belongs to FileRepository; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            db (AsyncSession): Database session used to read or persist application
                records.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside FileRepository so related code remains
                cohesive and testable.
        """
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
        """
        Add file with stage.

        Purpose:
            Implements add_file_with_stage for the repository layer that isolates
                database persistence from higher-level business logic.
        Class:
            Belongs to FileRepository; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            file_id (str): File identifier used to locate metadata, processing stages,
                or indexed chunks.
            chat_id (str): Chat/session identifier used to scope retrieval, tasks, or
                responses.
            file_name (str): Input value for the file name parameter.
            unique_generated_name (str): Input value for the unique generated name
                parameter.
            full_path (str): Input value for the full path parameter.
            stage (str): Input value for the stage parameter.
            status (str): Input value for the status parameter.
        Returns:
            FileModel: Domain or persistence object produced by the operation.
        Why Added:
            Centralizes this behavior inside FileRepository so related code remains
                cohesive and testable.
        """
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
        """
        Commit.

        Purpose:
            Implements commit for the repository layer that isolates database
                persistence from higher-level business logic.
        Class:
            Belongs to FileRepository; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside FileRepository so related code remains
                cohesive and testable.
        """
        await self._db.commit()

    async def refresh(self, record: FileModel) -> None:
        """
        Refresh.

        Purpose:
            Implements refresh for the repository layer that isolates database
                persistence from higher-level business logic.
        Class:
            Belongs to FileRepository; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            record (FileModel): Input value for the record parameter.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside FileRepository so related code remains
                cohesive and testable.
        """
        await self._db.refresh(record)

    async def get_title_summaries_by_chat_id(self, chat_id: str) -> list[dict]:
        """
        Get title summaries by chat id.

        Purpose:
            Implements get_title_summaries_by_chat_id for the repository layer that
                isolates database persistence from higher-level business logic.
        Class:
            Belongs to FileRepository; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            chat_id (str): Chat/session identifier used to scope retrieval, tasks, or
                responses.
        Returns:
            list[dict]: Structured data produced by the operation.
        Why Added:
            Centralizes this behavior inside FileRepository so related code remains
                cohesive and testable.
        """
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
