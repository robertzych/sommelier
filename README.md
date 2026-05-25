# Sommelier

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![Built with uv](https://img.shields.io/badge/built%20with-uv-purple)](https://github.com/astral-sh/uv)

A RAG assistant for Apache Pinot available as an MCP server for Claude Desktop and Claude Code.

---

## Quick Start

**Requires**: git, [uv](https://github.com/astral-sh/uv), Python 3.11+, and an `ANTHROPIC_API_KEY`.

**First run only**: FastEmbed downloads the embedding (219 MB) and reranker (92 MB) ONNX models on the first query. Subsequent runs use the cached models in `~/.cache`.

```bash
git clone https://github.com/robertzych/sommelier.git && cd sommelier
uv sync
cp sommelier.toml.example sommelier.toml   # set inference.api_key or export ANTHROPIC_API_KEY
uv run sommelier query "What is the default broker port?"
```

Or wire it as an MCP server and ask directly inside Claude — see [Claude Desktop](#using-sommelier-with-claude-desktop) and [Claude Code](#using-sommelier-with-claude-code) below.

---

## Introduction

[Apache Pinot](https://pinot.apache.org/) is a distributed OLAP datastore built for real-time analytics at scale — used at LinkedIn, Uber, and Stripe to serve sub-second queries over billions of rows. Its configuration surface is dense: dozens of table config keys, multiple ingestion paths, pluggable components at every layer.

[docs.pinot.apache.org](https://docs.pinot.apache.org) has an LLM-powered search that handles single questions well, but it doesn't support follow-up questions. General-purpose LLMs like Claude also have a training cutoff — the latest features are simply outside their knowledge.

Sommelier addresses both: it retrieves the relevant doc sections in real time, delivers high quality answers (2.96/3 on a 25-question golden set vs 1.48/3 for raw Claude), supports multi-turn conversations so follow-up questions carry full context, and integrates into Claude as an MCP server so you can ask Pinot questions natively inside your existing tools.

---

## Design Overview

![Sommelier Architecture Diagram](Sommelier%20Architecture%20Diagram.png)

### Ingestion Pipeline

**Chunking** — Documents are sourced from the [pinot-contrib/pinot-docs](https://github.com/pinot-contrib/pinot-docs) GitHub repo (a standalone GitBook repo, separate from the main Pinot monorepo). Each `.md` file is cleaned first — GitBook template tags (`{% ... %}`) are stripped and excess blank lines collapsed — then split in two stages: `MarkdownHeaderTextSplitter` divides the document at H1/H2/H3 headers, carrying each header forward as metadata on its child chunks; `RecursiveCharacterTextSplitter` further bounds each section to 512 tokens with 50-token overlap (tiktoken-encoded). The 50-token overlap preserves sentence continuity at split boundaries without duplicating full paragraphs. Header metadata is then serialized into a section breadcrumb (`H1 > H2 > H3`) and prepended to each chunk's text before embedding, so BM25 and dense models see document structure — not just local content. Two retrieval problems emerged during evaluation and required additional transforms within this pipeline:

**Markdown table normalization** — Pinot's config reference docs store property names, default values, and descriptions in separate markdown columns; when chunked, the semantic link between a property and its default is lost. A query for the default broker port retrieved chunks from `broker.md` with empty default columns — the chunk containing the answer (8099) used different vocabulary ("deprecated", "legacy") and was never retrieved. Fix: tables whose second column header contains "default" are converted to prose — `"<property>: default <value>. <description>"` — co-locating all three in one retrievable string.

**LLM code annotation** — A code-heavy chunk has minimal prose for BM25 or dense embeddings to match against; for the star-tree index question, the file was retrieved via its prose intro chunks (MRR@5 0.20) but the Example chunk containing the complete `tableIndexConfig.starTreeIndexConfigs` JSON was not in the top-20 candidates at all, causing the LLM to respond with a placeholder instead of real config. Fix: for code-heavy chunks (≥50% code characters), the LLM generates a one-sentence `Description:` and one representative `Question:` inserted between a section breadcrumb and the chunk body — giving hybrid search the natural-language signal needed to retrieve it:
   ```
   Star-Tree Index > Configuration > Example
   Description: Star-tree index configuration specifying dimension split order and aggregation functions.
   Question: How do I configure a star-tree index with custom split order and sum aggregation?
   [JSON code block]
   ```

**Embedding** — each processed chunk is encoded into two representations before being stored in Qdrant: a dense vector (`BAAI/bge-base-en-v1.5`, 768 dims, FastEmbed) for semantic similarity and a sparse BM25 vector (`Qdrant/bm25`, FastEmbed) for keyword matching. The dense model is configurable (e.g. `all-MiniLM-L6-v2` for speed, `text-embedding-3-small` for higher quality); changing it requires re-ingesting since vector dimensions are fixed at collection creation.

**Incremental updates** (`src/ingestion/ingest.py`): Point IDs are SHA-256 content hashes of the chunk text, cast to UUIDs — deterministic and unique per chunk. When a doc file changes, modified chunks get new IDs; the old IDs become orphans. Per file: scroll existing point IDs, diff against new, delete orphans, insert new. Unchanged chunks (same ID already in Qdrant) are skipped entirely. No updates — only inserts and deletes.

```
File unchanged → skip (IDs already in Qdrant)
File modified  → old IDs deleted + new IDs inserted
File deleted   → old IDs deleted
New file       → all IDs inserted
```

**Why Qdrant** — Three constraints drove the choice: hybrid search (BM25 + semantic) was required from the start given Pinot's technical vocabulary; FastEmbed should work without API keys; and incremental updates via content hash need robust upsert semantics. ChromaDB was ruled out — it has no native hybrid search, so implementing it would have required a separate BM25 index, custom score fusion, and two indexes to keep in sync on every incremental update. Qdrant covers all three in one client: native sparse+dense vectors with Reciprocal Rank Fusion (no manual score normalization), built-in FastEmbed integration, and a local embedded mode (`QdrantClient(path="./qdrant_storage")`) that runs with no server or Docker — just the Python client — while using the same API as a self-hosted or cloud instance.

### Retrieval Pipeline

Apache Pinot's documentation is dense with technical vocabulary: exact config keys (`pinot.broker.client.queryPort`), class names (`RealtimeToOfflineSegmentsTask`), and numeric constants. Semantic-only search misses exact term matches; keyword-only search misses paraphrased queries. Both are required.

**Stage 1 — Hybrid retrieval (top 20 candidates)**

Each query is encoded into two representations simultaneously:

- **Dense vector** (`BAAI/bge-base-en-v1.5`, 768 dims, FastEmbed) — captures semantic meaning; finds "what port does the broker listen on?" even if the docs say "broker query port"
- **Sparse vector** (`Qdrant/bm25`, FastEmbed) — captures exact term overlap; critical for config keys and class names that semantic models may not distinguish

Qdrant's native Reciprocal Rank Fusion (RRF) merges the two ranked lists. Chunks that rank well in both lists rise to the top; chunks present in only one are demoted. No manual score normalization.

**Stage 2 — Cross-encoder reranking (top 20 → top 5)**

Bi-encoders (used in Stage 1) embed query and document independently — fast at scale, but imprecise. A cross-encoder (`Xenova/ms-marco-MiniLM-L-6-v2`) scores each (query, chunk) pair jointly, attending to interactions between the two. More accurate, but too slow to run against the full collection. The 20→5 funnel gets both: fast broad retrieval, then precise reranking on a small candidate set.

```
User query
    ├── dense embed ──→ top 20 by cosine ──┐
    └── BM25 encode  ──→ top 20 by BM25  ──┴── RRF → top 20 → cross-encoder → top 5 → LLM
```

**System prompt** — The system prompt (`prompts/system_prompt.md`, tracked in git) governs how the LLM uses retrieved chunks. Key constraints: answers are scoped strictly to Apache Pinot; config keys, class names, and port numbers are only cited if they appear verbatim in the retrieved context — no fabrication; if the retrieved context is incomplete, the response explicitly notes the coverage gap rather than filling it with training data. The user message pairs numbered context blocks (file path + section breadcrumb + chunk text) with the active Pinot version and the user's question.

Every trace logs a `prompt_version` field — a SHA-256 content hash of `system_prompt.md` — so eval score changes can be correlated to specific prompt edits without relying on git history.

### Evaluation: Golden Set, Baselines, and V1 Results

**Methodology**: 25 parent questions + 5 follow-up questions, constructed source-first — expected doc sections are chosen before writing the questions, ensuring deliberate coverage across ingestion, indexing, querying, operations, and configuration. Five query types: how-to (8), factual (5), conceptual (4), comparison (3), new-in-2026 (5). The new-in-2026 category specifically targets features introduced after LLM training cutoffs to expose knowledge gaps.

Three baselines scored on a 3-point rubric (accuracy 0/1, completeness 0/1, citations 0/1) per question:

**Quality results** (25 questions):

| System | Accuracy | Completeness | Citations | Total |
|---|---|---|---|---|
| Sommelier (`claude-haiku-4-5`) | 0.96 | 1.00 | 1.00 | **2.96 / 3** |
| docs.pinot.apache.org | 1.00 | 1.00 | 1.00 | 3.00 / 3 |
| Claude (`claude-sonnet-4-6`) | 0.56 | 0.92 | 0.00 | 1.48 / 3 |

Sommelier matches the official docs AI (2.96 vs 3.00) while integrating into Claude natively. The gap versus raw Claude is largest on new-in-2026 questions: Claude accuracy 0.00 vs Sommelier 1.00 — LLM training cutoffs are a real limitation for a fast-moving project like Pinot.

Both Sommelier and the Claude baseline use models with the same August 2025 training cutoff. Sommelier uses `claude-haiku-4-5` (~3× cheaper than `claude-sonnet-4-6`) yet scores 2× better overall — demonstrating that RAG impact outweighs model size for domain-specific Q&A.

**Retrieval results** (hit rate, recall, MRR, full recall at each pipeline stage):

| Stage | Hit Rate | Recall | MRR | Full Recall |
|---|---|---|---|---|
| Reranked @5 | 96% (24/25) | 96% | 0.77 | 96% (24/25) |
| Candidates @20 | 100% (25/25) | 100% | 0.78 | — |

The one remaining reranked@5 miss (q015) reaches candidates@20 at rank 1 but is dropped by the cross-encoder — a known reranker issue tracked in V2.

Full per-question breakdown: [evals/results.md](evals/results.md)

### MCP Server

**MCP server** (`src/mcp_server.py`): Sommelier runs as a persistent process, so the pipeline is initialized lazily on the first tool call rather than at import time — ONNX model loads happen once and are amortized across all subsequent queries. A server-side `_history` list accumulates user/assistant turns across MCP calls, capped at `memory_turns * 2`, giving follow-up questions full conversation context. The tool docstring doubles as a behavior contract: it instructs Claude to pass the user's question verbatim (no paraphrasing) and display the response exactly as returned.

---

## Roadmap

- **Latency** — End-to-end latency is ~4.9s (retrieval ~450ms, reranker ~2.3s, LLM ~2.2s). The FastEmbed cross-encoder is the primary bottleneck; candidates to explore: reduce `retrieval_top_k` from 20→10 (halves reranker inference time if hit rate holds), switch to Cohere Rerank (single HTTP call vs local ONNX), profile to separate ONNX model-load time from inference time (load is amortized in the MCP server's persistent process).
- **Retrieval quality** — q015 is a known reranker miss: the expected source reaches candidates@20 at rank 1 but is dropped by the cross-encoder. Candidates: `rerank_top_k=7` as a zero-cost mitigation, Cohere Rerank as a higher-quality alternative.
- **LLM judge** — Automate eval scoring with a LiteLLM judge (`--judge claude` mode in `eval.py`) to enable regression detection on every prompt or ingestion change. Validate automated scores against the manual baseline before trusting them.
- **Observability CLI** — `sommelier logs --review`: lists recent query traces with retrieved file paths and latency; `p` promotes a query to the golden set (pre-fills `expected_sources` from reranked results), `n` skips, `q` quits.
- **PyPI** — `uv publish` so users can `uv tool install sommelier` without cloning the repo.
- **Git LFS** — Migrate `qdrant_storage/` to Git LFS to avoid history bloat from binary re-indexing commits.

---

## Environment Setup

**Prerequisites**: [git](https://git-scm.com/), [uv](https://github.com/astral-sh/uv), Python 3.11+, and an `ANTHROPIC_API_KEY` (or another [LiteLLM-supported](https://docs.litellm.ai/docs/providers) provider — see [Configuration Reference](#configuration-reference)).

Install `uv` if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**1. Clone and install**

```bash
git clone https://github.com/robertzych/sommelier.git
cd sommelier
uv sync
```

`uv sync` downloads all Python dependencies into an isolated virtual environment. Pre-built Qdrant data is included in the repo — no separate download needed.

On the **first query**, FastEmbed downloads the embedding (~219 MB) and reranker (~92 MB) ONNX models and caches them in `~/.cache`. Subsequent runs skip this step.

**2. Configure**

```bash
cp sommelier.toml.example sommelier.toml
```

Edit `sommelier.toml` and set your API key:

```toml
[inference]
model = "anthropic/claude-haiku-4-5-20251001"
api_key = "sk-ant-..."   # or export ANTHROPIC_API_KEY in your shell
```

---

## Using Sommelier from the CLI

**One-shot query:**

```bash
uv run sommelier query "What is the default broker port?"
```

**Multi-turn REPL** (conversation memory across turns, Ctrl+C to exit):

```bash
uv run sommelier chat
```

---

## Using Sommelier with Claude Desktop

**1. Complete [Environment Setup](#environment-setup) (steps 1–3)**

**2. Edit Claude Desktop's config file**

Open `~/Library/Application Support/Claude/claude_desktop_config.json` (create it if it doesn't exist) and add the `sommelier` server:

```json
{
  "mcpServers": {
    "sommelier": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/where/you/cloned/sommelier",
        "sommelier-mcp"
      ],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-..."
      }
    }
  }
}
```

Update `--directory` to the absolute path of the cloned repo.

**3. Add a global instruction to Claude Desktop**

Go to **Settings > General > Instructions for Claude** and add:

> For any Apache Pinot question, always use the search_pinot tool.

Without this, Claude may answer well-known Pinot facts from training data rather than calling the tool. With it, no query prefix is needed — ask Pinot questions naturally and Sommelier is invoked automatically.

**4. Restart Claude Desktop and test**

Ask any Apache Pinot question. You should see a tool call to `search_pinot` and a response with a Sources section at the end.

---

## Using Sommelier with Claude Code

**1. Complete [Environment Setup](#environment-setup) (steps 1–2)**

**2. Create `.mcp.json` from the example**

```bash
cp .mcp.json.example .mcp.json
```

Edit `.mcp.json` and set `--directory` to the absolute path of the cloned repo:

```json
{
  "mcpServers": {
    "sommelier": {
      "command": "uv",
      "args": ["run", "--directory", "/where/you/cloned/sommelier", "sommelier-mcp"]
    }
  }
}
```

Claude Code inherits `ANTHROPIC_API_KEY` from your shell, so no `env` block is needed. Open a Claude Code session in the `sommelier/` directory and ask any Apache Pinot question — the tool is called automatically. Follow-up questions carry full conversation context.

**Using Sommelier in other projects**: copy `.mcp.json.example` and `.claude/settings.json` to that project's root directory, rename to `.mcp.json`, and update `--directory` to point to your Sommelier clone.

---

## Running Ingestion

First time users may skip this section as pre-built Qdrant data is included in the repo.

Run ingestion to build or rebuild the vector index from the official Pinot docs:

```bash
# Clone the Pinot docs repo (separate from the apache/pinot monorepo)
git clone https://github.com/pinot-contrib/pinot-docs

# Index everything
uv run sommelier ingest --docs-path /path/to/pinot-docs

# Incremental update after a docs pull
cd pinot-docs && git pull && cd ..
uv run sommelier ingest --docs-path /path/to/pinot-docs
```

On the first run, FastEmbed downloads the embedding (~219 MB) and BM25 models before indexing begins. After the downloads complete, a progress bar shows per-file progress. Full ingestion of the 628-file pinot-docs repo takes ~47 minutes on CPU (no local GPU required). ~29% of that time (~13 minutes) is LLM annotation calls for code-heavy chunks (552 out of 6,191 total chunks, ~1.5s per call).

Subsequent runs skip unchanged chunks — only modified or new content is re-embedded. After re-indexing, commit the updated `qdrant_storage/` to make the new data available to others.

---

## Configuration Reference

All configuration lives in `sommelier.toml`. Copy `sommelier.toml.example` as a starting point.

### `[inference]`

| Field | Default | Description |
|---|---|---|
| `model` | `anthropic/claude-haiku-4-5-20251001` | LiteLLM model string. Use `ollama/llama3.1:8b` for fully local (requires [Ollama](https://ollama.com)). |
| `api_key` | `""` | API key for the inference provider. Leave blank to use the provider's env var (e.g. `ANTHROPIC_API_KEY`). |
| `temperature` | `0.1` | LLM temperature. Low values produce more consistent, factual responses. |
| `memory_turns` | `5` | Number of prior conversation turns included in each LLM call. Each turn = 1 user + 1 assistant message. |

### `[retrieval]`

| Field | Default | Description |
|---|---|---|
| `retrieval_top_k` | `20` | Candidates retrieved by hybrid search (Stage 1). |
| `rerank_top_k` | `5` | Final chunks sent to the LLM after cross-encoder reranking (Stage 2). |
| `reranker` | `fastembed` | Reranker backend: `fastembed` (local, no API key) or `cohere` (higher quality, requires `COHERE_API_KEY`). |
| `fastembed_reranker_model` | `Xenova/ms-marco-MiniLM-L-6-v2` | FastEmbed cross-encoder model. |
| `cohere_api_key` | `""` | Required when `reranker = "cohere"`. |

### `[embeddings]`

| Field | Default | Description |
|---|---|---|
| `provider` | `fastembed` | Embedding provider: `fastembed` (local) or `openai`. |
| `dense_model` | `BAAI/bge-base-en-v1.5` | Dense embedding model. Changing this requires re-ingesting (different vector dimensions). |
| `openai_api_key` | `""` | Required when `provider = "openai"`. |

### `[vector_store]`

| Field | Default | Description |
|---|---|---|
| `backend` | `qdrant_local` | Storage backend: `qdrant_local` (embedded, no server) or `qdrant_docker`. |
| `path` | `./qdrant_storage` | Local storage directory (used when `backend = "qdrant_local"`). |
| `docker_url` | `http://localhost:6333` | Qdrant server URL (used when `backend = "qdrant_docker"`). |

### `[observability]`

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

---

## Observability

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

For a browser-based UI, set `exporter = "langfuse"` in `sommelier.toml` and add your [Langfuse](https://langfuse.com/) API keys.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
