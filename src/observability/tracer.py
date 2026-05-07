import pathlib
import subprocess
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

_REPO_ROOT = pathlib.Path(__file__).parent.parent.parent


def _get_prompt_version() -> str:
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", "prompts/system_prompt.md"],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
        )
        commit = result.stdout.strip()
        return commit[:7] if commit else "unknown"
    except Exception:
        return "unknown"


@runtime_checkable
class Exporter(Protocol):
    def export(self, trace: dict[str, Any]) -> None: ...


class Tracer:
    def __init__(self) -> None:
        self.prompt_version: str = _get_prompt_version()
        self._trace: dict[str, Any] = {}
        self._start_time: float = 0.0
        self._exporters: list[Exporter] = []

    def register_exporter(self, exporter: Exporter) -> None:
        self._exporters.append(exporter)

    def start_trace(self, trace_id: str, event_type: str, **kwargs: Any) -> None:
        self._start_time = time.monotonic()
        self._trace = {
            "event_type": event_type,
            "trace_id": trace_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            **kwargs,
        }
        if event_type == "query":
            self._trace["prompt_version"] = self.prompt_version

    def emit(self, stage: str, data: dict[str, Any]) -> None:
        # stage is reserved for future per-stage routing; data is merged flat into the trace
        self._trace.update(data)

    def flush(self) -> None:
        if not self._trace:
            return
        self._trace["total_latency_ms"] = int((time.monotonic() - self._start_time) * 1000)
        for exporter in self._exporters:
            exporter.export(dict(self._trace))
        self._trace = {}


tracer = Tracer()
