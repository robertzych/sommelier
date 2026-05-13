"""Unit and integration tests for inference/llm.py."""
import os
import types
import uuid

import pytest
from qdrant_client.models import ScoredPoint

from inference.llm import build_user_message, complete
from observability.tracer import tracer

_OPENAI_TEST_API_KEY = os.environ.get("SOMMELIER_TEST_OPENAI_API_KEY")


def _make_config(memory_turns: int = 5) -> types.SimpleNamespace:
    """Build a minimal config for inference tests."""
    return types.SimpleNamespace(
        inference=types.SimpleNamespace(
            model="gpt-4o-mini",
            temperature=0.1,
            memory_turns=memory_turns,
            api_key=_OPENAI_TEST_API_KEY or "",
        ),
    )


def _make_chunk(
    text: str,
    file_path: str = "docs/test.md",
    h1: str = "",
    h2: str = "",
    h3: str = "",
) -> ScoredPoint:
    """Build a ScoredPoint with the payload fields used by build_user_message."""
    return ScoredPoint(
        id=str(uuid.uuid4()),
        version=0,
        score=0.9,
        payload={"text": text, "file_path": file_path, "h1": h1, "h2": h2, "h3": h3},
        vector=None,
    )


# ── build_user_message unit tests ────────────────────────────────────────────


class TestBuildUserMessage:
    """Unit tests for build_user_message."""

    def test_empty_chunks_produces_header_version_and_question(self):
        """No chunks yields only the context header, version, and question lines."""
        msg = build_user_message([], "What is a broker?", "1.2")
        assert "Context from Apache Pinot documentation:" in msg
        assert "Pinot version: 1.2" in msg
        assert "Question: What is a broker?" in msg

    def test_none_version_renders_as_latest(self):
        """pinot_version=None renders 'Pinot version: latest'."""
        msg = build_user_message([], "query", None)
        assert "Pinot version: latest" in msg

    def test_chunk_index_is_one_based(self):
        """Two chunks are numbered [1] and [2]."""
        chunks = [_make_chunk("first"), _make_chunk("second")]
        msg = build_user_message(chunks, "q", None)
        assert "[1]" in msg
        assert "[2]" in msg

    def test_chunk_with_all_headers_builds_full_breadcrumb(self):
        """A chunk with h1, h2, h3 produces 'h1 > h2 > h3' in the header line."""
        chunk = _make_chunk("text", h1="Concepts", h2="Table", h3="Table Types")
        msg = build_user_message([chunk], "q", None)
        assert "Concepts > Table > Table Types" in msg

    def test_chunk_with_partial_headers_omits_empty(self):
        """A chunk with only h1 set produces 'h1' with no trailing ' > '."""
        chunk = _make_chunk("text", h1="Concepts", h2="", h3="")
        msg = build_user_message([chunk], "q", None)
        assert "Concepts" in msg
        assert "Concepts >" not in msg

    def test_chunk_text_and_file_path_appear_in_message(self):
        """The chunk's text and file_path are both included in the output."""
        chunk = _make_chunk(
            "The default broker port is 8099.",
            file_path="configuration/broker.md",
        )
        msg = build_user_message([chunk], "q", None)
        assert "The default broker port is 8099." in msg
        assert "configuration/broker.md" in msg


# ── complete() integration tests ─────────────────────────────────────────────


@pytest.mark.skipif(not _OPENAI_TEST_API_KEY, reason="SOMMELIER_TEST_OPENAI_API_KEY not set")
class TestComplete:
    """Integration tests for complete() using the live OpenAI API."""

    def test_streams_tokens_and_emits_full_tracer_output(self):
        """complete() yields string tokens and emits the full tracer payload.

        Pre-emits retrieval/reranker latencies to verify that complete() merges its
        llm latency into the existing stage_latency_ms dict rather than overwriting it.
        Uses memory_turns=1 with a 4-turn history to verify truncation doesn't break the call.
        Checks that the tracer response equals the concatenation of all yielded tokens.
        """
        config = _make_config(memory_turns=1)
        chunk = _make_chunk(
            "The default Pinot broker port is 8099.",
            file_path="configuration/broker.md",
            h1="Configuration",
        )
        history = [
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "recent question"},
            {"role": "assistant", "content": "recent answer"},
        ]

        tracer.start_trace("test-complete", event_type="query")
        tracer.emit("retrieval", {"stage_latency_ms": {"retrieval": 42, "reranker": 17}})
        tokens = list(complete("What is the broker port?", [chunk], history, config, "1.2"))

        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)
        assert tracer._trace["tokens"]["input"] > 0
        assert tracer._trace["tokens"]["output"] > 0
        assert "model" in tracer._trace
        assert tracer._trace["response"] == "".join(tokens)
        latencies = tracer._trace["stage_latency_ms"]
        assert latencies["retrieval"] == 42
        assert latencies["reranker"] == 17
        assert isinstance(latencies["llm"], int)

        tracer._trace = {}
