from __future__ import annotations

from enum import StrEnum


class FileStage(StrEnum):
    UPLOADED = "uploaded"
    EXTRACTED = "extracted"
    NORMALIZER = "normalizer"
    ENRICHED = "enriched"
    EMBEDDING = "embedding"
    DONE = "done"


class FileStageStatus(StrEnum):
    WAITING = "waiting"
    STARTED = "started"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"
