# Configuration Reference

All configuration lives in `sommelier.toml`. Copy `sommelier.toml.example` as a starting point.

## `[inference]`

| Field | Default | Description |
|---|---|---|
| `model` | `anthropic/claude-haiku-4-5-20251001` | LiteLLM model string. Use `ollama/llama3.1:8b` for fully local (requires [Ollama](https://ollama.com)). |
| `api_key` | `""` | API key for the inference provider. Leave blank to use the provider's env var (e.g. `ANTHROPIC_API_KEY`). |
| `temperature` | `0.1` | LLM temperature. Low values produce more consistent, factual responses. |
| `memory_turns` | `5` | Number of prior conversation turns included in each LLM call. Each turn = 1 user + 1 assistant message. |

## `[retrieval]`

| Field | Default | Description |
|---|---|---|
| `retrieval_top_k` | `20` | Candidates retrieved by hybrid search (Stage 1). |
| `rerank_top_k` | `5` | Final chunks sent to the LLM after cross-encoder reranking (Stage 2). |
| `reranker` | `fastembed` | Reranker backend: `fastembed` (local, no API key) or `cohere` (higher quality, requires `COHERE_API_KEY`). |
| `fastembed_reranker_model` | `Xenova/ms-marco-MiniLM-L-6-v2` | FastEmbed cross-encoder model. |
| `cohere_api_key` | `""` | Required when `reranker = "cohere"`. |

## `[embeddings]`

| Field | Default | Description |
|---|---|---|
| `provider` | `fastembed` | Embedding provider: `fastembed` (local) or `openai`. |
| `dense_model` | `BAAI/bge-base-en-v1.5` | Dense embedding model. Changing this requires re-ingesting (different vector dimensions). |
| `openai_api_key` | `""` | Required when `provider = "openai"`. |

## `[vector_store]`

| Field | Default | Description |
|---|---|---|
| `backend` | `qdrant_local` | Storage backend: `qdrant_local` (embedded, no server) or `qdrant_docker`. |
| `path` | `./qdrant_storage` | Local storage directory (used when `backend = "qdrant_local"`). |
| `docker_url` | `http://localhost:6333` | Qdrant server URL (used when `backend = "qdrant_docker"`). |

## `[observability]`

| Field | Default | Description |
|---|---|---|
| `log_path` | `./sommelier_traces.jsonl` | Path for query and ingestion traces (JSON lines). |
| `exporter` | `local` | Exporter: `local` (JSON lines file) or `langfuse` (requires Langfuse API keys). |
| `max_log_mb` | `100` | Rotate the trace log when it exceeds this size. |
| `log_rotations_kept` | `3` | Number of rotated log files to retain. |
| `debug_retrieval` | `false` | When `true`, runs 3 Qdrant queries per request to populate `from_dense`/`from_sparse` flags on candidates (useful for debugging retrieval). |
| `langfuse_public_key` | `""` | Required when `exporter = "langfuse"`. |
| `langfuse_secret_key` | `""` | Required when `exporter = "langfuse"`. |
| `langfuse_host` | `https://cloud.langfuse.com` | Langfuse server URL. |
