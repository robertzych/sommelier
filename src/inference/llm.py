"""LiteLLM wrapper: builds user message, streams LLM response, emits to tracer."""
import pathlib
import time
from collections.abc import Generator

import litellm
from qdrant_client.models import ScoredPoint

from observability.tracer import tracer

_REPO_ROOT = pathlib.Path(__file__).parent.parent.parent
_SYSTEM_PROMPT = (_REPO_ROOT / "prompts" / "system_prompt.md").read_text()


def build_user_message(
    chunks: list[ScoredPoint],
    query: str,
    pinot_version: str | None,
) -> str:
    """Build the user message from reranked chunks, query, and optional version."""
    parts = ["Context from Apache Pinot documentation:\n"]
    for i, chunk in enumerate(chunks, 1):
        payload = chunk.payload or {}
        file_path = payload.get("file_path", "")
        headers = [payload.get(h, "") for h in ("h1", "h2", "h3")]
        breadcrumb = " > ".join(h for h in headers if h)
        header_line = f"[{i}] {file_path}"
        if breadcrumb:
            header_line += f" — {breadcrumb}"
        parts.append(f"{header_line}\n{payload.get('text', '')}\n")

    parts.append(f"Pinot version: {pinot_version or 'latest'}")
    parts.append(f"Question: {query}")
    return "\n".join(parts)


def complete(
    query: str,
    chunks: list[ScoredPoint],
    history: list[dict],
    config,
    pinot_version: str | None = None,
) -> Generator[str, None, None]:
    """Stream an LLM response for a query with retrieved context.

    Yields token strings. Emits tokens, latency, and response to the tracer
    singleton when the stream is exhausted or the generator is closed.
    """
    memory_turns = getattr(config.inference, "memory_turns", 5)
    recent_history = history[-(memory_turns * 2):]

    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    messages.extend(recent_history)
    messages.append({"role": "user", "content": build_user_message(chunks, query, pinot_version)})

    api_key = getattr(config.inference, "api_key", "") or None

    t0 = time.monotonic()
    stream = litellm.completion(
        model=config.inference.model,
        messages=messages,
        temperature=config.inference.temperature,
        stream=True,
        stream_options={"include_usage": True},
        **({"api_key": api_key} if api_key else {}),
    )

    response_parts: list[str] = []
    usage: dict[str, int] = {"input": 0, "output": 0}
    try:
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                response_parts.append(delta)
                yield delta
            if getattr(chunk, "usage", None):
                usage = {
                    "input": chunk.usage.prompt_tokens,
                    "output": chunk.usage.completion_tokens,
                }
    finally:
        llm_latency_ms = int((time.monotonic() - t0) * 1000)
        stage_latency = dict(tracer._trace.get("stage_latency_ms", {}))
        stage_latency["llm"] = llm_latency_ms
        tracer.emit(
            "llm",
            {
                "tokens": usage,
                "stage_latency_ms": stage_latency,
                "response": "".join(response_parts),
            },
        )
