"""MCP server exposing search_pinot backed by the Sommelier RAG pipeline."""
import uuid

from mcp.server.fastmcp import FastMCP

from cli import _init_pipeline, _setup_tracer, load_config
from inference.llm import complete
from observability.tracer import tracer
from retrieval.search import search
from vector_store.qdrant_store import PINOT_DOCS_COLLECTION

mcp = FastMCP("Sommelier")

# Lazy globals — initialized on first tool call so ONNX models load once,
# not at import time (which would break tests and slow module import).
_config = None
_client = None
_dense = None
_sparse = None
_reranker = None
_history: list[dict] = []


def _ensure_initialized() -> None:
    """Load config and initialize pipeline on first tool call."""
    global _config, _client, _dense, _sparse, _reranker
    if _config is not None:
        return
    _config = load_config()
    _setup_tracer(_config)
    _client, _dense, _sparse, _reranker = _init_pipeline(_config)


@mcp.tool()
def search_pinot(query: str, pinot_version: str = "latest") -> str:
    """Answer Apache Pinot questions using the official documentation.

    Pass the user's question verbatim as `query` — do not paraphrase or summarize it.
    Display the response exactly as returned by this tool, including the Sources section at the end.
    Do not add, remove, or reformat any part of the response."""
    _ensure_initialized()
    version = None if pinot_version == "latest" else pinot_version

    tracer.start_trace(
        str(uuid.uuid4()),
        event_type="query",
        query=query,
        pinot_version=pinot_version,
    )

    chunks = search(
        query=query,
        client=_client,
        collection_name=PINOT_DOCS_COLLECTION,
        dense_provider=_dense,
        sparse_provider=_sparse,
        reranker=_reranker,
        config=_config,
        pinot_version=version,
    )

    response = "".join(complete(query, chunks, _history, _config, pinot_version=version))
    tracer.flush()

    # Accumulate history and cap at memory_turns * 2 messages
    _history.append({"role": "user", "content": query})
    _history.append({"role": "assistant", "content": response})
    _history[:] = _history[-(_config.inference.memory_turns * 2):]

    return response


def main() -> None:
    """Entry point for the sommelier-mcp script."""
    mcp.run()


if __name__ == "__main__":
    main()
