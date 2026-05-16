from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID

from sqlalchemy import select

from database import SessionLocal
from models.file import File


class FileSummaryRepository:
    async def update_file_summary(
            self,
            file_id: str | UUID,
            title: str,
            summary: Any,
    ) -> File:
        async with SessionLocal() as db:
            result = await db.execute(
                select(File).where(File.file_id == file_id)
            )

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
    ) -> File:
        """
        Use this from normal sync Celery tasks.
        """
        print("update_file_summary_sync")
        return asyncio.run(
            self.update_file_summary(
                file_id=file_id,
                title=title,
                summary=summary,
            )
        )
