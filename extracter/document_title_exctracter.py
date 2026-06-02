import re
from pathlib import Path


class DocumentTitleDetector:
    """
    Document Title Detector.

    Purpose:
        Defines DocumentTitleDetector in the document extraction pipeline that
            normalizes PDFs, enriches visual content, builds RAG units, and indexes
            data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.

    Attributes:
        HEADING_RE (Any): Class-level value used by this class.
    """

    HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

    def detect_from_markdown(
        self,
        markdown_path: str | Path,
        fallback_title: str | None = None,
    ) -> str:
        """
        Detect from markdown.

        Purpose:
            Implements detect_from_markdown for the document extraction pipeline that
                normalizes PDFs, enriches visual content, builds RAG units, and indexes
                data.
        Class:
            Belongs to DocumentTitleDetector; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            markdown_path (str | Path): Input value for the markdown path parameter.
            fallback_title (str | None): Input value for the fallback title parameter.
        Returns:
            str: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside DocumentTitleDetector so related code
                remains cohesive and testable.
        """
        path = Path(markdown_path)

        if not path.exists():
            return fallback_title or "Untitled Document"

        for line in path.read_text(encoding="utf-8").splitlines():
            match = self.HEADING_RE.match(line.strip())

            if not match:
                continue

            title = match.group(2).strip()

            if title:
                return title

        return fallback_title or "Untitled Document"
