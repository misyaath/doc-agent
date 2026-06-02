from __future__ import annotations

import asyncio
from asyncio import AbstractEventLoop
from typing import Any
from uuid import UUID

from sqlalchemy import select

from core.logging import get_logger
from database import SessionLocal
from models.file import File

logger = get_logger(__name__)


class FileSummaryRepository:
    """
    File Summary Repository.

    Purpose:
        Defines FileSummaryRepository in the repository layer that isolates database
            persistence from higher-level business logic.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    async def update_file_summary(
        self,
        file_id: str | UUID,
        title: str,
        summary: Any,
    ) -> File:
        """
        Update file summary.

        Purpose:
            Implements update_file_summary for the repository layer that isolates
                database persistence from higher-level business logic.
        Class:
            Belongs to FileSummaryRepository; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            file_id (str | UUID): File identifier used to locate metadata, processing
                stages, or indexed chunks.
            title (str): Input value for the title parameter.
            summary (Any): Input value for the summary parameter.
        Returns:
            File: Domain or persistence object produced by the operation.
        Why Added:
            Centralizes this behavior inside FileSummaryRepository so related code
                remains cohesive and testable.
        """
        async with SessionLocal() as db:
            result = await db.execute(select(File).where(File.file_id == file_id))

            file = result.scalar_one_or_none()

            if file is None:
                raise ValueError(f"File not found: {file_id}")

            file.title = title
            file.summary = summary

            db.add(file)

            await db.commit()
            await db.refresh(file)

            return file

    def update_file_summary_sync(
        self,
        file_id: str | UUID,
        title: str,
        summary: Any,
        loop: AbstractEventLoop | None = None,
    ) -> File:
        """
        Update file summary sync.

        Purpose:
            Implements update_file_summary_sync for the repository layer that isolates
                database persistence from higher-level business logic.
        Class:
            Belongs to FileSummaryRepository; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            file_id (str | UUID): File identifier used to locate metadata, processing
                stages, or indexed chunks.
            title (str): Input value for the title parameter.
            summary (Any): Input value for the summary parameter.
            loop (AbstractEventLoop | None): Input value for the loop parameter.
        Returns:
            File: Domain or persistence object produced by the operation.
        Why Added:
            Centralizes this behavior inside FileSummaryRepository so related code
                remains cohesive and testable.
        """
        logger.info("Updating file summary", extra={"file_id": str(file_id)})
        coroutine = self.update_file_summary(
            file_id=file_id,
            title=title,
            summary=summary,
        )
        if loop is not None:
            return loop.run_until_complete(coroutine)
        return asyncio.run(coroutine)
