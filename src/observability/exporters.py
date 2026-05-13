import json
import pathlib
from typing import Any

from langfuse import Langfuse

from observability.tracer import Exporter


class LocalJSONExporter:
    """Appends traces as JSON lines; rotates the log file when it exceeds max_log_mb."""

    def __init__(self, log_path: str, max_log_mb: int = 100, log_rotations_kept: int = 3) -> None:
        self._log_path = pathlib.Path(log_path)
        self._max_bytes = max_log_mb * 1024 * 1024
        self._rotations_kept = log_rotations_kept

    def export(self, trace: dict[str, Any]) -> None:
        """Append trace as a JSON line, rotating the log file if it exceeds the size limit."""
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        if self._log_path.exists() and self._log_path.stat().st_size > self._max_bytes:
            self._rotate()
        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(trace) + "\n")

    def _rotate(self) -> None:
        oldest = self._log_path.parent / f"{self._log_path.name}.{self._rotations_kept}"
        if oldest.exists():
            oldest.unlink()
        for i in range(self._rotations_kept - 1, 0, -1):
            src = self._log_path.parent / f"{self._log_path.name}.{i}"
            dst = self._log_path.parent / f"{self._log_path.name}.{i + 1}"
            if src.exists():
                src.rename(dst)
        self._log_path.rename(self._log_path.parent / f"{self._log_path.name}.1")


class LangfuseExporter:
    """Sends traces to Langfuse in real time."""

    def __init__(self, public_key: str, secret_key: str, host: str) -> None:
        self._client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)

    def export(self, trace: dict[str, Any]) -> None:
        """Send a single trace to Langfuse."""
        import time
        from langfuse.types import TraceContext

        event_type = trace.get("event_type", "unknown")
        trace_id = trace.get("trace_id", "")
        total_latency_ms = trace.get("total_latency_ms", 0)

        if event_type in ("query", "chat_turn"):
            inp = {"query": trace.get("query", "")}
            out = {"response": trace.get("response", "")}
            tokens = trace.get("tokens", {})
            kwargs: dict[str, Any] = {
                "as_type": "generation",
                "model": trace.get("model", ""),
                "usage_details": {
                    "input": tokens.get("input", 0),
                    "output": tokens.get("output", 0),
                },
            }
        elif event_type == "ingestion":
            inp = {"file_path": trace.get("file_path", "")}
            out = {
                "inserted": trace.get("inserted", 0),
                "deleted": trace.get("deleted", 0),
                "skipped": trace.get("skipped", 0),
            }
            kwargs = {"as_type": "span"}
        else:
            inp = {}
            out = {}
            kwargs = {"as_type": "span"}

        # Langfuse v4 requires 32-char hex trace IDs; derive one from our UUID seed.
        langfuse_trace_id = Langfuse.create_trace_id(seed=trace_id)
        span = self._client.start_observation(
            trace_context=TraceContext(trace_id=langfuse_trace_id),
            name=event_type,
            input=inp,
            output=out,
            metadata=trace,
            **kwargs,
        )
        # Pass end_time as now + total_latency_ms (ns) so Langfuse shows the real latency.
        end_ns = int((time.time() + total_latency_ms / 1000) * 1e9)
        span.end(end_time=end_ns)
        self._client.flush()


def get_exporters(config) -> list[Exporter]:
    """Return exporters configured via config.observability."""
    obs = config.observability
    exporters: list[Exporter] = [
        LocalJSONExporter(
            log_path=obs.log_path,
            max_log_mb=obs.max_log_mb,
            log_rotations_kept=obs.log_rotations_kept,
        )
    ]
    if getattr(obs, "exporter", "local") == "langfuse":
        exporters.append(
            LangfuseExporter(
                public_key=getattr(obs, "langfuse_public_key", ""),
                secret_key=getattr(obs, "langfuse_secret_key", ""),
                host=getattr(obs, "langfuse_host", "https://cloud.langfuse.com"),
            )
        )
    return exporters
