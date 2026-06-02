from __future__ import annotations

from enum import StrEnum


class FileStage(StrEnum):
    """
    File Stage.

    Purpose:
        Defines FileStage in the domain constants and enums that describe file-
            processing state.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.

    Attributes:
        UPLOADED (Any): Class-level value used by this class.
        EXTRACTED (Any): Class-level value used by this class.
        NORMALIZER (Any): Class-level value used by this class.
        ENRICHED (Any): Class-level value used by this class.
        EMBEDDING (Any): Class-level value used by this class.
        DONE (Any): Class-level value used by this class.
    """

    UPLOADED = "uploaded"
    EXTRACTED = "extracted"
    NORMALIZER = "normalizer"
    ENRICHED = "enriched"
    EMBEDDING = "embedding"
    DONE = "done"


class FileStageStatus(StrEnum):
    """
    File Stage Status.

    Purpose:
        Defines FileStageStatus in the domain constants and enums that describe file-
            processing state.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.

    Attributes:
        WAITING (Any): Class-level value used by this class.
        STARTED (Any): Class-level value used by this class.
        PROCESSING (Any): Class-level value used by this class.
        DONE (Any): Class-level value used by this class.
        FAILED (Any): Class-level value used by this class.
    """

    WAITING = "waiting"
    STARTED = "started"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"
