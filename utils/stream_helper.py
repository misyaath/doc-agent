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
        "query_planner": "Understanding your question",
        "rewrite_query": "Understanding your question",
        "retrieve": "Searching relevant document sections",
        "retriever": "Searching relevant document sections",
        "rerank": "Selecting the best evidence",
        "answer": "Generating answer",
        "generate_answer": "Generating answer",
    }

    return labels.get(node_name or "", "Working")


def node_done_label(node_name: str | None) -> str:
    labels = {
        "query_planner": "Question understood",
        "rewrite_query": "Question understood",
        "retrieve": "Relevant sections found",
        "retriever": "Relevant sections found",
        "rerank": "Best evidence selected",
        "answer": "Answer generated",
        "generate_answer": "Answer generated",
    }

    return labels.get(node_name or "", "Step completed")
