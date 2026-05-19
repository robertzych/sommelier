"""Unit tests for cli.py: cmd_chat history management and cmd_query followup support."""
import argparse
import json
import pathlib
import tempfile
from contextlib import ExitStack
from unittest import mock

from cli import _load_followup_history, cmd_chat, cmd_query, load_config

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


_FOLLOWUP_JSON = {
    "version": "1.0",
    "questions": [
        {
            "id": "q001_t2",
            "parent_id": "q001",
            "turn": 2,
            "query_type": "how-to",
            "question": "What are the limitations of upsert?",
            "expected_sources": ["build-with-pinot/ingestion/upsert-and-dedup/upsert.md"],
            "parent": {
                "question": "How do you configure upsert on a Pinot real-time table?",
                "sommelier_answer": "To enable upserts, add upsertConfig to your table config.",
            },
            "baseline_responses": {
                "sommelier (claude-haiku-4-5)": {
                    "answer": None,
                    "scores": {"accuracy": None, "completeness": None, "citations": None, "total": None},
                    "scored_by": None,
                    "scored_at": None,
                }
            },
        }
    ],
}


class TestLoadFollowupHistory:
    """Unit tests for _load_followup_history."""

    def test_returns_question_and_two_message_history(self, tmp_path):
        """_load_followup_history returns the follow-up question and a [user, assistant] history pair."""
        followup_file = tmp_path / "followup.json"
        followup_file.write_text(json.dumps(_FOLLOWUP_JSON))

        question, history = _load_followup_history(str(followup_file), "q001_t2")

        assert question == "What are the limitations of upsert?"
        assert history == [
            {"role": "user", "content": "How do you configure upsert on a Pinot real-time table?"},
            {"role": "assistant", "content": "To enable upserts, add upsertConfig to your table config."},
        ]

    def test_raises_on_unknown_id(self, tmp_path):
        """_load_followup_history raises SystemExit when the id is not found."""
        followup_file = tmp_path / "followup.json"
        followup_file.write_text(json.dumps(_FOLLOWUP_JSON))

        try:
            _load_followup_history(str(followup_file), "q999_t2")
            assert False, "expected SystemExit"
        except SystemExit as e:
            assert "q999_t2" in str(e)


class TestCmdQueryFollowup:
    """Unit tests for cmd_query with --followup-id."""

    def _make_query_args(self, question=None, followup_id=None, followup_file=None, version="latest"):
        """Build a minimal argparse Namespace for cmd_query."""
        return argparse.Namespace(
            question=question,
            version=version,
            followup_id=followup_id,
            followup_file=followup_file,
        )

    def test_followup_id_seeds_history_and_uses_file_question(self, tmp_path):
        """cmd_query with --followup-id passes parent Q&A as history and uses the file question."""
        followup_file = tmp_path / "followup.json"
        followup_file.write_text(json.dumps(_FOLLOWUP_JSON))

        config = load_config(str(_FIXTURE_TOML))
        captured: list[dict] = {}

        def fake_complete(query, chunks, history, cfg, pinot_version=None):
            captured["query"] = query
            captured["history"] = list(history)
            yield "answer"

        patches = [
            mock.patch("cli.load_config", return_value=config),
            mock.patch("cli._setup_tracer"),
            mock.patch("cli._init_pipeline", return_value=(None, None, None, None)),
            mock.patch("retrieval.search.search", return_value=[]),
            mock.patch("inference.llm.complete", side_effect=fake_complete),
            mock.patch("observability.tracer.tracer.start_trace"),
            mock.patch("observability.tracer.tracer.flush"),
            mock.patch("builtins.print"),
        ]
        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            cmd_query(self._make_query_args(followup_id="q001_t2", followup_file=str(followup_file)))

        assert captured["query"] == "What are the limitations of upsert?"
        assert captured["history"] == [
            {"role": "user", "content": "How do you configure upsert on a Pinot real-time table?"},
            {"role": "assistant", "content": "To enable upserts, add upsertConfig to your table config."},
        ]

    def test_explicit_question_overrides_file_question(self, tmp_path):
        """cmd_query with --followup-id uses the CLI question when provided, ignoring the file question."""
        followup_file = tmp_path / "followup.json"
        followup_file.write_text(json.dumps(_FOLLOWUP_JSON))

        config = load_config(str(_FIXTURE_TOML))
        captured: list[dict] = {}

        def fake_complete(query, chunks, history, cfg, pinot_version=None):
            captured["query"] = query
            yield "answer"

        patches = [
            mock.patch("cli.load_config", return_value=config),
            mock.patch("cli._setup_tracer"),
            mock.patch("cli._init_pipeline", return_value=(None, None, None, None)),
            mock.patch("retrieval.search.search", return_value=[]),
            mock.patch("inference.llm.complete", side_effect=fake_complete),
            mock.patch("observability.tracer.tracer.start_trace"),
            mock.patch("observability.tracer.tracer.flush"),
            mock.patch("builtins.print"),
        ]
        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            cmd_query(self._make_query_args(
                question="Custom override question?",
                followup_id="q001_t2",
                followup_file=str(followup_file),
            ))

        assert captured["query"] == "Custom override question?"

    def test_no_followup_id_passes_empty_history(self):
        """cmd_query without --followup-id passes empty history to complete."""
        config = load_config(str(_FIXTURE_TOML))
        captured: list[dict] = {}

        def fake_complete(query, chunks, history, cfg, pinot_version=None):
            captured["history"] = list(history)
            yield "answer"

        patches = [
            mock.patch("cli.load_config", return_value=config),
            mock.patch("cli._setup_tracer"),
            mock.patch("cli._init_pipeline", return_value=(None, None, None, None)),
            mock.patch("retrieval.search.search", return_value=[]),
            mock.patch("inference.llm.complete", side_effect=fake_complete),
            mock.patch("observability.tracer.tracer.start_trace"),
            mock.patch("observability.tracer.tracer.flush"),
            mock.patch("builtins.print"),
        ]
        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            cmd_query(self._make_query_args(question="What is the broker port?"))

        assert captured["history"] == []
