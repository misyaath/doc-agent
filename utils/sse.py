from __future__ import annotations

import json
from typing import Any


def sse_event(event: str, data: Any) -> str:
    """
    Format data as a Server-Sent Event.
    Browser/EventSource expects:
      event: token
      data: {"text": "..."}

    followed by a blank line.
    """
    return (
        f"event: {event}\n"
        f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
    )
