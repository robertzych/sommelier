"""Unit tests for mcp_server.py: search_pinot tool logic."""
import pathlib
from contextlib import ExitStack
from unittest import mock

import mcp_server
from cli import load_config

_FIXTURE_TOML = pathlib.Path(__file__).parent / "fixtures" / "sommelier_test.toml"


def _reset_globals():
    """Reset lazy pipeline globals and history between tests."""
    mcp_server._config = None
    mcp_server._client = None
    mcp_server._dense = None
    mcp_server._sparse = None
    mcp_server._reranker = None
    mcp_server._history.clear()


def _base_patches(config, fake_complete):
    """Return the standard patch list used by most tests."""
    return [
        mock.patch("mcp_server.load_config", return_value=config),
        mock.patch("mcp_server._setup_tracer"),
        mock.patch("mcp_server._init_pipeline", return_value=(None, None, None, None)),
        mock.patch("mcp_server.search", return_value=[]),
        mock.patch("mcp_server.complete", side_effect=fake_complete),
        mock.patch("observability.tracer.tracer.start_trace"),
        mock.patch("observability.tracer.tracer.flush"),
    ]


class TestSearchPinot:
    """Unit tests for the search_pinot MCP tool."""

    def setup_method(self):
        """Reset pipeline state before each test."""
        _reset_globals()

    def test_returns_joined_tokens(self):
        """search_pinot joins all tokens from complete into a single string."""
        config = load_config(str(_FIXTURE_TOML))

        def fake_complete(query, chunks, history, cfg, pinot_version=None):
            yield "hello"
            yield " world"

        with ExitStack() as stack:
            for p in _base_patches(config, fake_complete):
                stack.enter_context(p)
            result = mcp_server.search_pinot("What is the broker port?")

        assert result == "hello world"

    def test_latest_version_passes_none_to_search(self):
        """pinot_version='latest' is converted to None for the search call."""
        config = load_config(str(_FIXTURE_TOML))
        captured: dict = {}

        def fake_complete(query, chunks, history, cfg, pinot_version=None):
            yield "ok"

        patches = _base_patches(config, fake_complete)
        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            search_mock = stack.enter_context(mock.patch("mcp_server.search", return_value=[]))
            mcp_server.search_pinot("Some question", pinot_version="latest")
            captured["version"] = search_mock.call_args.kwargs["pinot_version"]

        assert captured["version"] is None

    def test_specific_version_passed_to_search(self):
        """A specific pinot_version string is forwarded to search unchanged."""
        config = load_config(str(_FIXTURE_TOML))
        captured: dict = {}

        def fake_complete(query, chunks, history, cfg, pinot_version=None):
            yield "ok"

        patches = _base_patches(config, fake_complete)
        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            search_mock = stack.enter_context(mock.patch("mcp_server.search", return_value=[]))
            mcp_server.search_pinot("Some question", pinot_version="1.2")
            captured["version"] = search_mock.call_args.kwargs["pinot_version"]

        assert captured["version"] == "1.2"

    def test_history_accumulates_across_calls(self):
        """search_pinot accumulates user+assistant turns in _history across consecutive calls."""
        config = load_config(str(_FIXTURE_TOML))
        history_snapshots: list[list[dict]] = []

        def fake_complete(query, chunks, history, cfg, pinot_version=None):
            history_snapshots.append(list(history))
            yield "answer"

        with ExitStack() as stack:
            for p in _base_patches(config, fake_complete):
                stack.enter_context(p)
            mcp_server.search_pinot("Q1")
            mcp_server.search_pinot("Q2")

        assert history_snapshots[0] == []
        assert history_snapshots[1] == [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "answer"},
        ]

    def test_history_capped_at_memory_turns(self):
        """_history is capped at memory_turns*2 messages (fixture: memory_turns=2 → 4 messages)."""
        config = load_config(str(_FIXTURE_TOML))

        def fake_complete(query, chunks, history, cfg, pinot_version=None):
            yield "ans"

        with ExitStack() as stack:
            for p in _base_patches(config, fake_complete):
                stack.enter_context(p)
            for i in range(5):
                mcp_server.search_pinot(f"Q{i}")

        # memory_turns=2 → cap at 4 messages (2 turns × 2 messages each)
        assert len(mcp_server._history) == 4

    def test_tracer_start_and_flush_called_per_request(self):
        """tracer.start_trace and tracer.flush are each called once per search_pinot call."""
        config = load_config(str(_FIXTURE_TOML))

        def fake_complete(query, chunks, history, cfg, pinot_version=None):
            yield "ok"

        with ExitStack() as stack:
            for p in _base_patches(config, fake_complete):
                stack.enter_context(p)
            start_mock = stack.enter_context(
                mock.patch("observability.tracer.tracer.start_trace")
            )
            flush_mock = stack.enter_context(
                mock.patch("observability.tracer.tracer.flush")
            )
            mcp_server.search_pinot("What is the broker port?")

        start_mock.assert_called_once()
        flush_mock.assert_called_once()
        assert start_mock.call_args.kwargs["event_type"] == "query"
        assert start_mock.call_args.kwargs["query"] == "What is the broker port?"
