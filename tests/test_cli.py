"""Unit tests for cli.py: cmd_chat history management."""
import argparse
import pathlib
from contextlib import ExitStack
from unittest import mock

from cli import cmd_chat, load_config

_FIXTURE_TOML = pathlib.Path(__file__).parent / "fixtures" / "sommelier_test.toml"


def _make_chat_args(version: str = "latest") -> argparse.Namespace:
    """Build a minimal argparse Namespace for cmd_chat."""
    return argparse.Namespace(version=version)


class TestCmdChatHistoryManagement:
    """Unit tests for cmd_chat conversation history accumulation and capping."""

    def test_history_accumulates_and_caps_across_turns(self):
        """cmd_chat passes empty history on turn 1, prior turn on turn 2, and caps at memory_turns*2.

        Uses memory_turns=2 from the fixture. Runs 3 turns and verifies:
        - turn 1 receives empty history
        - turn 2 receives turn 1's user+assistant messages
        - turn 3 receives exactly 4 messages (memory_turns*2 cap applied after turn 2)
        """
        config = load_config(str(_FIXTURE_TOML))
        complete_history_calls: list[list[dict]] = []

        def fake_complete(query, chunks, history, cfg, pinot_version=None):
            complete_history_calls.append(list(history))
            yield "ok"

        patches = [
            mock.patch("cli.load_config", return_value=config),
            mock.patch("cli._setup_tracer"),
            mock.patch("cli._init_pipeline", return_value=(None, None, None, None)),
            mock.patch("retrieval.search.search", return_value=[]),
            mock.patch("inference.llm.complete", side_effect=fake_complete),
            mock.patch("observability.tracer.tracer.start_trace"),
            mock.patch("observability.tracer.tracer.flush"),
            mock.patch("builtins.input", side_effect=["q1", "q2", "q3", EOFError()]),
            mock.patch("builtins.print"),
        ]
        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            cmd_chat(_make_chat_args())

        assert complete_history_calls[0] == []
        assert complete_history_calls[1] == [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "ok"},
        ]
        assert len(complete_history_calls[2]) == 4
