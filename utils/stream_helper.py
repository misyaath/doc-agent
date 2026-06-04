from typing import Any


def stream_step(
    key: str,
    label: str,
    status: str = "running",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "type": "step",
        "key": key,
        "label": label,
        "status": status,
        "metadata": metadata or {},
    }


def node_label(node_name: str | None) -> str:
    labels = {
        "query_parsing": "Understanding your question",
        "retrieve": "Searching relevant document sections",
        "generate_answer": "Generating answer",
    }

    return labels.get(node_name or "", "Working")


def node_done_label(node_name: str | None) -> str:
    labels = {
        "query_parsing": "Question understood",
        "retrieve": "Relevant sections found",
        "generate_answer": "Answer generated",
    }

    return labels.get(node_name or "", "Step completed")
