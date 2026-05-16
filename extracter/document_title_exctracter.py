import re
from pathlib import Path


class DocumentTitleDetector:
    HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

    def detect_from_markdown(
            self,
            markdown_path: str | Path,
            fallback_title: str | None = None,
    ) -> str:
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
