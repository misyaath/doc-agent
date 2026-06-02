from __future__ import annotations

import json
from typing import Any


def sse_event(event: str, data: Any) -> str:
    """
    Sse event.

    Purpose:
        Implements sse_event for the shared utility layer used by API and service code.
    Args:
        event (str): Event name or payload value being formatted for streaming.
        data (Any): Input value for the data parameter.
    Returns:
        str: Result produced by the operation.
    Why Added:
        Provides a documented entry point for this module-level behavior and keeps
            callers
        from needing to know lower-level implementation details.
    """
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
