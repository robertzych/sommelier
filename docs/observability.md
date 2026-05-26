# Observability

Every query and ingestion run is traced to `sommelier_traces.jsonl`. Traces are JSON lines — one object per query — containing the full retrieval pipeline state:

```json
{
  "event_type": "query",
  "trace_id": "f3a2b1c0-...",
  "ts": "2026-05-19T22:15:03Z",
  "query": "What is the default broker port?",
  "candidates": [
    {"file_path": "reference/configuration-reference/broker.md", "rrf_score": 0.82},
    {"file_path": "configuration-reference/server.md", "rrf_score": 0.61}
  ],
  "reranked": [
    {"file_path": "reference/configuration-reference/broker.md", "cross_encoder_score": 0.91}
  ],
  "stage_latency_ms": {"retrieval": 450, "reranker": 2300, "llm": 2150},
  "total_latency_ms": 4900,
  "tokens": {"input": 2100, "output": 180}
}
```

Key fields to look at: `candidates[*].file_path` (what the hybrid search found), `reranked[*].file_path` (what survived reranking), and `stage_latency_ms` (where time is spent). If a question gets a wrong answer, the trace tells you whether it's a retrieval miss (expected source not in candidates) or a reranker miss (expected source in candidates but dropped).

For a browser-based UI, set `exporter = "langfuse"` in `sommelier.toml` and add your [Langfuse](https://langfuse.com/) API keys. See [configuration.md](configuration.md) for the full `[observability]` settings.
