"""Tests for observability/exporters.py: LocalJSONExporter, LangfuseExporter, get_exporters."""
import json
import os
import types
from unittest import mock

import pytest

from observability.exporters import LocalJSONExporter, LangfuseExporter, get_exporters

_LANGFUSE_TEST_KEYS = (
    os.environ.get("SOMMELIER_TEST_LANGFUSE_PUBLIC_KEY"),
    os.environ.get("SOMMELIER_TEST_LANGFUSE_SECRET_KEY"),
)
_LANGFUSE_HOST = os.environ.get("SOMMELIER_TEST_LANGFUSE_HOST", "https://cloud.langfuse.com")


def _make_obs_config(tmp_path, exporter="local"):
    """Build a minimal config namespace with observability settings pointing to tmp_path."""
    return types.SimpleNamespace(
        observability=types.SimpleNamespace(
            log_path=str(tmp_path / "traces.jsonl"),
            exporter=exporter,
            max_log_mb=100,
            log_rotations_kept=3,
            langfuse_public_key=_LANGFUSE_TEST_KEYS[0] or "",
            langfuse_secret_key=_LANGFUSE_TEST_KEYS[1] or "",
            langfuse_host=_LANGFUSE_HOST,
        )
    )


class TestLocalJSONExporter:
    """Tests for LocalJSONExporter: write, append, rotation, and directory creation."""

    def test_export_writes_json_line(self, tmp_path):
        """export() creates the log file and writes the trace as a valid JSON line."""
        log_path = tmp_path / "traces.jsonl"
        exporter = LocalJSONExporter(str(log_path))
        exporter.export({"event_type": "query", "trace_id": "abc"})

        lines = log_path.read_text().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0]) == {"event_type": "query", "trace_id": "abc"}

    def test_export_appends_multiple_traces(self, tmp_path):
        """Calling export() multiple times appends each trace as a separate JSON line."""
        log_path = tmp_path / "traces.jsonl"
        exporter = LocalJSONExporter(str(log_path))
        exporter.export({"n": 1})
        exporter.export({"n": 2})

        lines = log_path.read_text().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"n": 1}
        assert json.loads(lines[1]) == {"n": 2}

    def test_creates_parent_directories(self, tmp_path):
        """export() creates any missing parent directories before writing."""
        log_path = tmp_path / "deep" / "nested" / "traces.jsonl"
        exporter = LocalJSONExporter(str(log_path))
        exporter.export({"event": "test"})

        assert log_path.exists()

    def test_rotates_when_file_exceeds_size_limit(self, tmp_path):
        """export() rotates the current log to .1 when file size exceeds max_log_mb."""
        log_path = tmp_path / "traces.jsonl"
        log_path.write_text('{"existing": "data"}\n')

        # max_log_mb=0 → max_bytes=0; any non-empty file exceeds the limit
        exporter = LocalJSONExporter(str(log_path), max_log_mb=0, log_rotations_kept=3)
        exporter.export({"new": "trace"})

        rotated = tmp_path / "traces.jsonl.1"
        assert rotated.exists()
        assert '{"existing": "data"}' in rotated.read_text()
        assert json.loads(log_path.read_text().strip()) == {"new": "trace"}

    def test_rotation_does_not_trigger_below_limit(self, tmp_path):
        """No rotation occurs when the log file is within the size limit."""
        log_path = tmp_path / "traces.jsonl"
        exporter = LocalJSONExporter(str(log_path), max_log_mb=100, log_rotations_kept=3)
        exporter.export({"n": 1})
        exporter.export({"n": 2})

        assert not (tmp_path / "traces.jsonl.1").exists()
        assert len(log_path.read_text().splitlines()) == 2

    def test_prunes_oldest_rotation_beyond_limit(self, tmp_path):
        """Rotation deletes the oldest file when log_rotations_kept would be exceeded."""
        log_path = tmp_path / "traces.jsonl"
        log_path.write_text('{"seed": "data"}\n')

        exporter = LocalJSONExporter(str(log_path), max_log_mb=0, log_rotations_kept=2)
        for i in range(4):
            exporter.export({"n": i})

        assert (tmp_path / "traces.jsonl").exists()
        assert (tmp_path / "traces.jsonl.1").exists()
        assert (tmp_path / "traces.jsonl.2").exists()
        assert not (tmp_path / "traces.jsonl.3").exists()

    def test_rotation_shifts_existing_numbered_files(self, tmp_path):
        """Each rotation increments existing numbered files (e.g. .1 → .2)."""
        log_path = tmp_path / "traces.jsonl"
        log_path.write_text('{"a": 1}\n')

        exporter = LocalJSONExporter(str(log_path), max_log_mb=0, log_rotations_kept=3)
        exporter.export({"a": 2})
        exporter.export({"a": 3})

        assert json.loads((tmp_path / "traces.jsonl.1").read_text().strip()) == {"a": 2}
        assert '{"a": 1}' in (tmp_path / "traces.jsonl.2").read_text()


@pytest.mark.skipif(
    not all(_LANGFUSE_TEST_KEYS),
    reason="requires SOMMELIER_TEST_LANGFUSE_PUBLIC_KEY and SOMMELIER_TEST_LANGFUSE_SECRET_KEY",
)
class TestLangfuseExporter:
    """Integration tests for LangfuseExporter against a real Langfuse instance."""

    def _make_exporter(self):
        """Build a LangfuseExporter using test credentials."""
        return LangfuseExporter(
            public_key=_LANGFUSE_TEST_KEYS[0],
            secret_key=_LANGFUSE_TEST_KEYS[1],
            host=_LANGFUSE_HOST,
        )

    def test_export_query_trace_without_error(self):
        """export() sends a query trace to Langfuse without raising an exception."""
        exporter = self._make_exporter()
        exporter.export({
            "event_type": "query",
            "trace_id": "test-query-trace-001",
            "query": "What is the default broker port?",
            "response": "The default Pinot broker port is 8099.",
            "total_latency_ms": 500,
        })

    def test_export_ingestion_trace_without_error(self):
        """export() sends an ingestion trace to Langfuse without raising an exception."""
        exporter = self._make_exporter()
        exporter.export({
            "event_type": "ingestion",
            "trace_id": "test-ingestion-trace-001",
            "file_path": "docs/config/broker.md",
            "inserted": 5,
            "deleted": 1,
            "skipped": 20,
        })


class TestGetExporters:
    """Tests for get_exporters factory."""

    def test_local_config_returns_single_local_exporter(self, tmp_path):
        """get_exporters returns a list with one LocalJSONExporter when exporter='local'."""
        config = _make_obs_config(tmp_path, exporter="local")
        exporters = get_exporters(config)

        assert len(exporters) == 1
        assert isinstance(exporters[0], LocalJSONExporter)

    def test_local_exporter_uses_config_values(self, tmp_path):
        """The LocalJSONExporter is configured with log_path, max_log_mb, and log_rotations_kept from config."""
        config = _make_obs_config(tmp_path, exporter="local")
        exporters = get_exporters(config)
        local = exporters[0]

        assert str(local._log_path) == str(tmp_path / "traces.jsonl")
        assert local._max_bytes == 100 * 1024 * 1024
        assert local._rotations_kept == 3

    @pytest.mark.skipif(
        not all(_LANGFUSE_TEST_KEYS),
        reason="requires SOMMELIER_TEST_LANGFUSE_PUBLIC_KEY and SOMMELIER_TEST_LANGFUSE_SECRET_KEY",
    )
    def test_langfuse_config_returns_both_exporters(self, tmp_path):
        """get_exporters returns LocalJSONExporter and LangfuseExporter when exporter='langfuse'."""
        config = _make_obs_config(tmp_path, exporter="langfuse")
        exporters = get_exporters(config)

        assert len(exporters) == 2
        assert isinstance(exporters[0], LocalJSONExporter)
        assert isinstance(exporters[1], LangfuseExporter)
