# Sommelier: System Design

## Context

Sommelier is a RAG-based chatbot for Apache Pinot guidance, distributed as a public GitHub repo that users run locally as an MCP server. This document is the full system design: vector store, ingestion pipeline, retrieval pipeline, inference, observability, evaluation, and system prompt.

The vector store decision hinges on three constraints:
1. **Hybrid search (BM25 + semantic) is required from day 1** — Apache Pinot has dense technical vocabulary and users range from beginners to power users querying exact config params.
2. **FastEmbed as default embedding** — users who clone the repo should get working embeddings with zero API keys required.
3. **Incremental updates via content hash** — the vector store persists between runs and re-embeds only changed documents.

These three constraints together **rule out ChromaDB**. ChromaDB has no native hybrid search; implementing it would require a separate BM25 index (e.g., `rank_bm25`), custom score fusion, and two indexes to keep in sync on incremental updates — adding more complexity than just using Qdrant directly.

---

## Recommendation: Qdrant (local embedded mode)

### Why Qdrant

| Requirement | ChromaDB | **Qdrant** | Weaviate |
|---|---|---|---|
| Hybrid search (BM25 + semantic) | Custom, manual | **Native sparse+dense** | Native |
| Local embedded (no Docker) | Yes | **Yes (`path=` mode)** | No |
| FastEmbed integration | No | **Built-in (`[fastembed]` extra)** | No |
| RRF score fusion | Manual | **Native** | Native |
| Incremental upsert by ID | Basic | **Robust** | Yes |
| Scales to 150k+ chunks | Degrades | **Handles well** | Yes |
| Zero hosting cost | Yes | **Yes** | Cloud required |

**Qdrant's local embedded mode** (`QdrantClient(path="./qdrant_storage")`) stores data in a local directory using Rust-backed storage — no Docker, no server, just `pip install qdrant-client[fastembed]`. Users who want Docker can use it identically; the same client API works against both.

---

## Architecture

```
sommelier/
├── ingestion/
│   └── ingest.py           # Load docs → clean GitBook → chunk → embed (dense + sparse) → delete stale + insert new into Qdrant
├── retrieval/
│   ├── search.py           # Hybrid search (dense + sparse → RRF → top 20) → cross-encoder reranking → top rerank_top_k
│   └── router.py           # V2: LLM query classifier → adjusts rerank_top_k per query type
├── inference/
│   └── llm.py              # LiteLLM wrapper: loads prompts/system_prompt.md, builds user message, streams response
├── vector_store/
│   └── qdrant.py           # Qdrant client abstraction (local or Docker)
├── embeddings/
│   └── provider.py         # FastEmbed (default) or OpenAI; swappable via config
├── observability/
│   ├── tracer.py           # Tracer singleton; each pipeline stage emits events to it
│   ├── exporters.py        # LocalJSONExporter (JSON lines + rotation) + LangfuseExporter
│   └── cli.py              # `python -m sommelier logs --review` log review + golden set promotion
└── evals/
    ├── golden_set.json     # Expert-written questions, expected sources, baseline scores
    ├── metrics.py          # Pure metric functions: hit_rate, recall, mrr, full_recall_rate at both retrieval stages
    └── eval.py             # Eval runner: retrieval-only mode + full (retrieval + LLM judge) mode
```

```
Project root:
├── sommelier/              ← Python package
├── prompts/
│   └── system_prompt.md    # Active system prompt; git commit hash logged in traces + eval
├── plan.md
├── mcp_server.py
└── sommelier.toml
```

### Ingestion Pipeline

```
Pinot docs repo (cloned locally)
        ↓
   ingest.py       ← load .md files → clean GitBook → chunk → embed (dense + sparse) → delete stale + insert new
```

**ingest.py** is the single ingestion module. It owns the full pipeline from raw docs to Qdrant points:

1. **Load** — walks `apache/pinot` docs under `website/docs/`, reads `.md` files, constructs `source_url` from the relative file path:

```python
# file_path:   website/docs/basics/concepts/table.md
# source_url:  https://docs.pinot.apache.org/basics/concepts/table
def to_source_url(file_path: str) -> str:
    path = file_path.removeprefix("website/docs/").removesuffix(".md")
    return f"https://docs.pinot.apache.org/{path}"
```

2. **Chunk** — strips GitBook syntax, splits on markdown headers (`MarkdownHeaderTextSplitter`), recursively splits oversized sections (`RecursiveCharacterTextSplitter`, 512 tokens, 50 overlap). See the Chunking Strategy section for full detail. Section headers (`h1`/`h2`/`h3`) propagate as metadata on every chunk.

3. **Embed: dense + sparse** — generates a dense vector (`bge-base-en-v1.5`, 768 dims via FastEmbed) and a sparse BM25 vector (`Qdrant/bm25` via FastEmbed) for each new chunk.

4. **Delete stale + insert new** — per file: scroll existing point IDs, diff against new IDs, delete orphans, insert new chunks. See Incremental Update Logic for full detail.

Ingestion is triggered manually (e.g. `python -m sommelier ingest --docs-path ./pinot/website/docs --version 1.2`). Subsequent runs skip unchanged chunks automatically. GitHub API for issues/PRs/code is a V1.1 addition — V1 is docs only.

---

### vector_store/qdrant_store.py

Thin wrapper that reads `[vector_store]` config from `sommelier.toml` and returns a configured `QdrantClient`. Also owns collection creation (called once at startup if the collection doesn't exist).

`PINOT_DOCS_COLLECTION` is a module-level constant callers use instead of a raw string to prevent accidental duplicate/misnamed collections. `ensure_collection` is parameterized by `collection_name` to support multiple collections with different models.

```python
PINOT_DOCS_COLLECTION = "pinot_docs"

def get_client(config) -> QdrantClient:
    if config.vector_store.backend == "qdrant_local":
        return QdrantClient(path=config.vector_store.path)
    return QdrantClient(url=config.vector_store.docker_url)

def ensure_collection(client: QdrantClient, config, collection_name: str) -> None:
    if client.collection_exists(collection_name):
        return
    dense_model = config.collections[collection_name].dense_model
    dense_dim = config.embeddings.model_dims[dense_model]
    client.create_collection(...)  # see schema below
```

Used by `indexer.py` (write) and `retriever.py` (read). All other modules go through this abstraction rather than instantiating `QdrantClient` directly.

### Qdrant Collection Schema

Dense vector dimensions are looked up from `config.embeddings.model_dims[dense_model]`, where `dense_model` comes from `config.collections[collection_name].dense_model`. This means adding a new model or collection only requires a config change — no code change in `qdrant_store.py`.

```python
client.create_collection(
    collection_name=collection_name,
    vectors_config={
        "dense": VectorParams(size=dense_dim, distance=Distance.COSINE),
        # dense_dim from config.embeddings.model_dims[config.collections[collection_name].dense_model]
        # changing dense_model requires dropping and recreating this collection
    },
    sparse_vectors_config={
        "sparse": SparseVectorParams()  # BM25 via FastEmbed Qdrant/bm25
    }
)
```

Relevant config sections:

```toml
[embeddings.model_dims]
"BAAI/bge-base-en-v1.5" = 768
"all-MiniLM-L6-v2" = 384
"text-embedding-3-small" = 1536

[collections.pinot_docs]
dense_model = "BAAI/bge-base-en-v1.5"
```

### Chunk Payload (metadata)

```python
payload = {
    "text":          chunk_text,
    "file_path":     "configuration-reference/table.md",   # relative path in pinot-docs repo root
    "doc_type":      "documentation",                      # or "github_issue", "java_code"
    "pinot_version": "1.2",                                # stored as metadata, default=latest
    "h1":            "Concepts",                           # section headers from splitter
    "h2":            "Table",
    "h3":            "Table Types",
    "content_hash":  sha256(chunk_text),                   # used as point ID for dedup
}
# Note: source_url was removed — GitBook URL mapping is unreliable without site settings access.
```

### Incremental Update Logic

Point IDs are derived from content hash — deterministic and unique per chunk. When a document changes, modified chunks get new IDs (old content → old ID becomes an orphan). Stale chunks must be explicitly deleted per file before upserting new ones.

```python
import hashlib, uuid

def derive_point_id(chunk_text: str) -> uuid.UUID:
    content_hash = hashlib.sha256(chunk_text.encode()).digest()[:16]
    return uuid.UUID(bytes=content_hash)

def index_file(client, collection_name: str, file_path: str, new_chunks: list[str]):
    # 1. collect existing point IDs for this file
    old_points, _ = client.scroll(
        collection_name,
        scroll_filter=Filter(must=[FieldCondition(key="file_path", match=MatchValue(value=file_path))]),
        with_payload=False,
    )
    old_ids = {p.id for p in old_points}

    # 2. compute new point IDs from updated chunks
    new_ids = {derive_point_id(chunk) for chunk in new_chunks}

    # 3. delete orphans (chunks removed or modified in the updated file)
    stale_ids = old_ids - new_ids
    if stale_ids:
        client.delete(collection_name, points_selector=list(stale_ids))

    # 4. insert only new or changed chunks (no updates — point IDs are content-derived)
    for chunk in new_chunks:
        point_id = derive_point_id(chunk)
        if point_id not in old_ids:
            client.upload_points(collection_name, points=[build_point(point_id, chunk)])
```

| Event | Behavior |
|---|---|
| Chunk unchanged | Skipped (ID already in Qdrant) |
| Chunk modified | Old point deleted + new point inserted |
| Chunk deleted | Old point deleted |
| New chunk added | Inserted |

There are no updates — only inserts and deletes. A modified chunk always produces a new point ID, so the operation is always delete-old + insert-new.

### Retrieval Pipeline

```
User query
    ↓
search.py       ← hybrid search (dense + sparse + RRF) → top 20 candidates → cross-encoder reranking → top rerank_top_k
    ↓
LLM context
```

**V2 — router.py** (deferred): an LLM query classifier that labels queries as `"specific"` or `"broad"` and doubles `rerank_top_k` for broad queries. Deferred because the added LLM call during inference adds latency and complexity before the base pipeline is validated.

**search.py** owns both retrieval stages:

- **Dense search** — embeds the query with the configured dense model and retrieves top candidates by cosine similarity. Captures semantic meaning; good for paraphrased or conceptual queries.
- **Sparse search** — encodes the query with BM25 via FastEmbed and retrieves top candidates by keyword overlap. Captures exact term matches — critical for Pinot's dense technical vocabulary (config keys, class names, port numbers).
- **RRF fusion** — Reciprocal Rank Fusion merges the two ranked lists natively in Qdrant. Chunks that rank well in both lists rise to the top; chunks that appear in only one are demoted. No manual score normalization needed.
- **Version filter** — if `pinot_version` is set, a `FieldCondition` filter restricts results to chunks tagged with that version.
- **Cross-encoder reranking** — narrows the 20 candidates to `rerank_top_k` by scoring each (query, chunk) pair jointly rather than independently — more accurate than embedding similarity alone. Two modes: FastEmbed cross-encoder (default, local, no API key) or Cohere Rerank (optional, higher quality, requires `COHERE_API_KEY`).

The two-stage design (vector search → cross-encoder) exists because vector search is fast but imprecise, and cross-encoder reranking is precise but slow at scale. The 20→5 funnel gets the best of both: fast broad retrieval to prune the search space, then precise reranking on a small candidate set.

```python
# Stage 1 — Hybrid search: top retrieval_top_k candidates (default 20)
candidates = client.query_points(
    collection_name="pinot_docs",
    prefetch=[
        Prefetch(query=dense_vec,  using="dense",  limit=config.retrieval_top_k),
        Prefetch(query=sparse_vec, using="sparse", limit=config.retrieval_top_k),
    ],
    query=FusionQuery(fusion=Fusion.RRF),
    limit=config.retrieval_top_k,
    query_filter=Filter(
        must=[FieldCondition(key="pinot_version", match=MatchValue(value=version))]
    ) if version else None,
)

# Stage 2 — Cross-encoder reranking: top retrieval_top_k → rerank_top_k
reranked = reranker.rerank(query, candidates, top_k=config.rerank_top_k)
```

### Observability

The primary failure risk is wrong chunks retrieved — the system returns a plausible but incorrect answer. Observability is developer-facing only; no trace data is surfaced in MCP responses.

#### Architecture

A central `Tracer` singleton lives in `observability/tracer.py`. Each pipeline stage imports it and emits events. `mcp_server.py` opens a trace at the start of each request and flushes it at the end. Ingestion does the same per file.

```python
# any pipeline stage
from sommelier.observability import tracer
tracer.emit("retrieval", {"candidates": [...], "scores": [...]})

# mcp_server.py
tracer.start_trace(trace_id, event_type="query", query=query)
# ... pipeline runs ...
tracer.flush()  # writes completed trace via configured exporter
```

#### Query Trace Format

```json
{
  "event_type": "query",
  "trace_id": "<uuid>",
  "ts": "2026-05-07T10:23:01Z",
  "prompt_version": "abc1234",
  "query": "what is the default broker port?",
  "pinot_version": "1.2",
  "candidates": [
    {
      "file_path": "docs/config/broker.md",
      "rrf_score": 0.82,
      "snippet": "The default Pinot broker port is 8099...",
      "from_dense": true,
      "from_sparse": false
    }
  ],
  "reranked": [
    {"file_path": "docs/config/broker.md", "cross_encoder_score": 0.91}
  ],
  "stage_latency_ms": {
    "retrieval": 85, "reranker": 310, "llm": 725
  },
  "total_latency_ms": 1120,
  "tokens": {"input": 2100, "output": 180},
  "response": "The default Pinot broker port is 8099..."
}
```

`from_dense`/`from_sparse` on candidates are only populated when `observability.debug_retrieval = true`. Enabling this runs 3 Qdrant queries per request (dense-only + sparse-only + hybrid) instead of 1 to expose the BM25/dense breakdown.

#### Ingestion Trace Format

```json
{
  "event_type": "ingestion",
  "trace_id": "<uuid>",
  "ts": "2026-05-07T09:00:00Z",
  "file_path": "docs/basics/concepts/table.md",
  "pinot_version": "1.2",
  "inserted": 12,
  "deleted": 3,
  "skipped": 45,
  "errors": []
}
```

Both query and ingestion traces are written to the same JSON lines log (`event_type` distinguishes them).

#### Emit Points

| Module | What to emit |
|---|---|
| `mcp_server.py` | `start_trace`, `flush` per request |
| `retrieval/search.py` | `candidates` (file_path, rrf_score, snippet); `reranked` (file_path, cross_encoder_score); if `debug_retrieval`: dense-only + sparse-only result sets |
| `inference/llm.py` | `tokens`, `stage_latency_ms.llm`, `response` |
| `ingestion/ingest.py` | `start_trace(event_type="ingestion")`, per-file inserted/deleted/skipped/errors, `flush` |

#### Exporters

- **LocalJSONExporter** (default): appends completed trace as a JSON line to `log_path`. Rotates when file exceeds `max_log_mb`; keeps last `log_rotations_kept` files.
- **LangfuseExporter** (optional): enabled via `exporter = "langfuse"` in config. Sends traces to Langfuse in real time. Local log is always written regardless.

#### Log Review and Golden Set Promotion

```
python -m sommelier logs --review
```

Lists recent query traces (newest first) with query text, retrieved file paths, and total latency. Interactive: `p` promotes the query to `golden_set.json` (scaffolds a new entry with `expected_sources` pre-filled from `reranked[*].file_path`), `n` skips, `q` quits.

#### Retrieval Metrics

Retrieval metrics are computed at eval-time only — not in the hot path. `eval.py` runs golden set queries through the live pipeline, captures `candidates` (top `retrieval_top_k`, post-RRF) and `reranked` (top `rerank_top_k`, post-cross-encoder) from the trace, and computes four metrics at each stage by comparing `file_path` against `expected_sources` per question:

| Metric | Definition | Rule |
|---|---|---|
| **Hit Rate@k** | Fraction of questions where ≥1 expected source appears in top-k | any-of |
| **Recall@k** | Mean fraction of expected sources found in top-k | per-question average |
| **MRR@k** | Mean reciprocal rank of the first expected source hit | per-question average |
| **Full Recall rate** | Fraction of questions where all expected sources appear in top-k | all-of |

All four are computed at both stages: `@candidates` (top `retrieval_top_k`) and `@reranked` (top `rerank_top_k`). The gap between stages reveals whether the reranker is helping or dropping relevant sources.

Pure metric functions live in `evals/metrics.py`; `eval.py` calls them. After each eval run, an `"eval_result"` event is appended to the trace log with per-question scores, keyed by `trace_id` and `question_id` — making results searchable without an interactive UI.

### System Prompt Design

The system prompt lives in `prompts/system_prompt.md`, tracked in git. Its commit hash is logged in every query trace (`prompt_version` field) and in `eval.py` output — this ties eval score changes to specific prompt edits.

```
# eval output (with prompt version)
Prompt version: abc1234 (2026-05-07)
Sommelier avg: 2.6/3  (+0.5 vs b38d8c8)
Retrieval precision: 87%
```

#### System Prompt (`prompts/system_prompt.md`)

```
You are Sommelier, an expert assistant for Apache Pinot. You help users
configure, deploy, query, and troubleshoot Apache Pinot based on the
official documentation.

## Scope
Answer only Apache Pinot questions. If asked about something outside
Apache Pinot, say: "I'm focused on Apache Pinot — for [topic], a
general-purpose assistant would serve you better."

## Using Retrieved Documentation
You will be given numbered documentation chunks as context. Base your
answers strictly on this context.

- Never fabricate configuration keys, class names, port numbers, or
  parameter values. Only cite values that appear verbatim in the
  retrieved documentation.
- If the retrieved context does not fully cover the question, give
  whatever partial answer the context supports, then add: "Note: my
  documentation coverage for this question was limited — verify this
  against the full Apache Pinot docs."
- If no retrieved context is relevant, say: "I couldn't find
  documentation covering this. Try docs.pinot.apache.org or the Apache
  Pinot community Slack."
- Do not use general knowledge about Apache Pinot when it contradicts
  or extends beyond the retrieved context.

## Version Specificity
If the retrieved documentation is from a different Pinot version than
the user specified (shown in the question), note the discrepancy clearly.

## Citations
Cite sources inline using the reference numbers from the context
(e.g., "The default broker port is 8099 [1]."). End every response with
a **Sources** section listing each cited number, its section path, and
its URL.

## Response Format
- Use fenced code blocks (with language tag) for all configuration
  snippets, YAML, JSON, SQL, and CLI commands.
- Use markdown headers (##) when the answer has multiple distinct parts
  (e.g., Configuration, Example, Caveats). Use flat prose for simple
  answers.
- Do not use step-by-step list structure unless the question explicitly
  asks for a procedure.
- Answer as thoroughly as the retrieved context supports — do not truncate.

## Tone
Be direct and precise. No filler phrases. Write as a knowledgeable
colleague who knows Apache Pinot deeply.
```

#### User Message Format

The user message (assembled by `mcp_server.py` before the LLM call) combines the reranked chunks, the active Pinot version, and the user's query:

```
Context from Apache Pinot documentation:

[1] docs/config/broker.md — Concepts > Broker > Configuration
URL: https://docs.pinot.apache.org/.../broker-config
The default broker port is 8099. To change it, set broker.port in broker.conf...

[2] docs/deployment/quickstart.md — Getting Started > Quickstart
URL: https://docs.pinot.apache.org/.../quickstart
...

Pinot version: 1.2
Question: What is the default broker port?
```

Each numbered entry includes: relative file path, breadcrumb section path (`h1 > h2 > h3` from chunk metadata), canonical `source_url`, and chunk text. The `Pinot version` line reflects the version filter applied at retrieval time (or "latest" if no filter).

#### Prompt Version in Traces and Evals

`llm.py` reads `prompts/system_prompt.md` at startup and computes its git commit hash. The hash is stored on the `Tracer` singleton and included in every query trace:

```json
{
  "event_type": "query",
  "prompt_version": "abc1234",
  ...
}
```

`eval.py` reads the hash from the current trace and includes it in the eval report header. Comparing eval runs across prompt versions is a matter of comparing hashes.

---

## Chunking Strategy

### V1 Scope: Docs only

Apache Pinot docs are GitBook-flavored markdown stored in the GitHub repo. Two steps before any chunk reaches Qdrant:

**Step 1 — Pre-process: strip GitBook syntax**

GitBook directives (`{% hint %}`, `{% tabs %}`, `{% content-ref %}`, etc.) must be stripped before splitting or they pollute chunks with raw template tokens.

```python
import re

def clean_gitbook(text: str) -> str:
    text = re.sub(r'\{%.*?%\}', '', text, flags=re.DOTALL)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()
```

**Step 2 — Split: header-aware, then recursive**

Split on markdown headers first (`#`, `##`, `###`), propagating section titles as metadata. Then recursively split any section that exceeds ~512 tokens.

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

headers = [("#", "h1"), ("##", "h2"), ("###", "h3")]
md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
header_splits = md_splitter.split_text(clean_gitbook(raw_markdown))

char_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=512, chunk_overlap=50
)
chunks = char_splitter.split_documents(header_splits)
```

Section titles (`h1`, `h2`, `h3`) are stored as metadata on every chunk — used for source citations and section-level filtering.

### Chunk Payload

```python
payload = {
    "text":          chunk_text,
    "file_path":     "configuration-reference/table.md",   # relative path in pinot-docs repo root
    "doc_type":      "documentation",
    "pinot_version": "1.2",
    "h1":            "Concepts",             # section headers from splitter
    "h2":            "Table",
    "h3":            "Table Types",
    "content_hash":  sha256(chunk_text),
}
```

### V1.1 Additions (post-validation)

- **Doc-level summaries**: Generate one LLM summary per `.md` file; store as an extra chunk alongside leaf chunks. Enables broad queries like "what does this doc cover?" without full RAPTOR complexity.
- **Java source code**: `RecursiveCharacterTextSplitter(language=Language.JAVA)` at method granularity, once the docs pipeline is validated.
- **GitHub issues/PRs**: Strategy TBD based on V1 learnings.

---

## Embedding Models

| Mode | Dense model | Dims | Source | Notes |
|---|---|---|---|---|
| **Default** | `BAAI/bge-base-en-v1.5` | 768 | FastEmbed (local) | Best retrieval quality, CPU, zero API key |
| **Fast mode** | `all-MiniLM-L6-v2` | 384 | FastEmbed (local) | Smaller, faster, adequate for simple queries |
| **High quality** | `text-embedding-3-small` | 1536 | OpenAI API | Best quality; ~$0.02/1M tokens, < $1 for full Pinot docs |

Sparse embeddings: `Qdrant/bm25` via FastEmbed (BM25, no model weights to download) — same regardless of dense model choice.

**Note:** changing `dense_model` requires dropping and recreating the Qdrant collection (dimensions are fixed at creation).

---

## Inference Model

**Default:** `gpt-4o-mini` via LiteLLM — cheap (~$0.15/1M input tokens), fast, high quality. Requires `OPENAI_API_KEY`.

**Local option (zero API keys):** `ollama/llama3.1:8b` — requires [Ollama](https://ollama.com) to be installed and running. Full privacy, zero cost, slower on CPU.

**Provider abstraction:** LiteLLM — one interface for OpenAI, Anthropic, Gemini, Groq, Ollama, and 100+ others. Model is just a string in config.

```python
import litellm

response = litellm.completion(
    model=config.inference.model,          # "gpt-4o-mini" or "ollama/llama3.1:8b"
    messages=history + [{"role": "user", "content": prompt}],
    temperature=config.inference.temperature,
    stream=True,
)
```

**Conversation memory:** last `memory_turns` (default 5) turns included in each call. Each turn counts as one user message + one assistant message; 5 turns = 10 messages prepended before the current user message.

---

## Configuration (`sommelier.toml`)

```toml
[embeddings]
provider = "fastembed"               # fastembed | openai
dense_model = "BAAI/bge-base-en-v1.5"  # change requires re-indexing
# dense_model = "all-MiniLM-L6-v2"  # fast mode
openai_api_key = ""                  # required when provider = "openai"

[inference]
model = "gpt-4o-mini"                # LiteLLM model string; provider inferred from prefix
                                     # "ollama/llama3.1:8b" for fully local
api_key = ""                         # not needed for Ollama; env var OPENAI_API_KEY also accepted
temperature = 0.1
memory_turns = 5

[retrieval]
retrieval_top_k = 20                 # candidates from hybrid search (stage 1)
rerank_top_k = 5                     # final chunks sent to LLM (stage 2, default)
reranker = "fastembed"               # fastembed | cohere
cohere_api_key = ""                  # required when reranker = "cohere"

[vector_store]
backend = "qdrant_local"             # qdrant_local | qdrant_docker
path = "./qdrant_storage"
docker_url = "http://localhost:6333"

[pinot]
version = "latest"                   # metadata filter applied at retrieval time

[observability]
log_path = "./sommelier_traces.jsonl"
exporter = "local"                    # local | langfuse
max_log_mb = 100                      # rotate log when it exceeds this size
log_rotations_kept = 3                # number of rotated files to keep
debug_retrieval = false               # BM25 vs dense delta; enables 3x Qdrant calls per query

langfuse_public_key = ""              # required when exporter = "langfuse"
langfuse_secret_key = ""
langfuse_host = "https://cloud.langfuse.com"
```

All `api_key` fields can be left blank and set via environment variables instead (`OPENAI_API_KEY`, `COHERE_API_KEY`, etc.) — LiteLLM and the provider SDKs pick these up automatically. The toml fields take precedence when set.

---

## RAG Framework

**Decision: Raw SDKs** (`qdrant-client`, `fastembed`, `litellm`, `langchain-text-splitters`)

The pipeline is custom enough — hybrid search with RRF, two-stage reranking, query router, content-hash dedup — that LlamaIndex would add a dependency and abstraction layer without removing meaningful code. Every stage maps directly to a raw SDK call, and working around LlamaIndex's built-in abstractions would add complexity rather than reduce it.

---

## Evaluation: Golden Set

### Design

The golden set is the primary mechanism for measuring quality, comparing Sommelier against baselines, and detecting regressions.

**Questions**: Expert-written (by you as a Pinot domain expert) — ensures questions are realistic and representative.  
**Answers**: No pre-written reference answers. You manually judge each system's response using a 3-point rubric (see below).  
**Baselines**: Plain Claude (automated via API) and docs.pinot.apache.org AI (manual query).

### Coverage — Phase 1: By Query Type (~20 questions)

| Query type | Count | Example |
|---|---|---|
| How-to | 8 | "How do I configure an upsert table?" |
| Factual | 5 | "What is the default port for the Pinot broker?" |
| Conceptual | 4 | "What is the difference between offline and realtime tables?" |
| Comparison | 3 | "What changed in real-time ingestion between Pinot 0.12 and 1.0?" |

**Phase 2** (post-V1): expand with domain coverage — ingestion, querying, schema, operations, configuration, performance tuning.

### Scoring Rubric (per answer, per system)

Each answer scores 0 or 1 on three dimensions:
- **accuracy** — Is the answer technically correct and non-hallucinated?
- **completeness** — Does it sufficiently address the question?
- **citations** — Does it cite appropriate supporting docs?

Total: 0–3. A score of 2+ is passing. Summaries reported as averages across the golden set.

### File Format (`evals/golden_set.json`)

```json
{
  "version": "1.0",
  "questions": [
    {
      "id": "q001",
      "question": "How do I configure an upsert table in Pinot?",
      "query_type": "how-to",
      "expected_sources": [
        "configuration-reference/schema.md",
        "users/user-guide-query/query-types/upsert.md"
      ],
      "baseline_responses": {
        "claude": {
          "answer": "...",
          "scores": {"accuracy": 1, "completeness": 0, "citations": 0, "total": 1},
          "scored_by": "expert",
          "scored_at": "2026-05-05"
        },
        "docs_pinot_ai": {
          "answer": "...",
          "scores": {"accuracy": 1, "completeness": 1, "citations": 0, "total": 2},
          "scored_by": "expert",
          "scored_at": "2026-05-05"
        }
      }
    }
  ]
}
```

### Eval Runner (`evals/eval.py`)

Two modes:

```bash
# Fast: retrieval metrics only, no LLM calls (~seconds)
python evals/eval.py --golden-set evals/golden_set.json --retrieval-only

# Full: retrieval + LLM generation + judge scoring
python evals/eval.py --golden-set evals/golden_set.json --judge claude
```

Full run output:
```
Prompt version: abc1234 (2026-05-07)

Retrieval (post-reranker, top-5):
  Hit Rate:        90%   (18/20)
  Recall:          85%   avg
  MRR:             0.78  avg
  Full Recall:     80%   (16/20 all expected sources found)

Retrieval (candidates, top-20):
  Hit Rate:        95%   (19/20)
  Recall:          92%   avg

ID     Question (truncated)              H@5  R@5    MRR   Sommelier  Claude  Docs AI
q001   How do I configure upsert?        ✓    100%   1.00  3/3        1/3     2/3
q002   What is the default broker port?  ✓    100%   0.67  3/3        2/3     3/3
...
Avg LLM scores:                                            2.7/3      1.5/3   2.2/3
```

Retrieval-only output shows the same two-block retrieval summary without the per-question LLM score columns.

`eval.py` appends one `"eval_result"` event per question to `sommelier_traces.jsonl` after each run, containing the trace_id, question_id, per-question retrieval metrics, and (if full run) LLM judge scores.

### LLM Judge Prompt (for automation / regression detection)

```python
JUDGE_PROMPT = """
You are evaluating an AI assistant's answer to an Apache Pinot documentation question.
Score the answer on three dimensions (each 0 or 1):
- accuracy: Is the answer technically correct based on Apache Pinot documentation?
- completeness: Does it sufficiently address the question?
- citations: Does it cite content from the expected source sections?

Question: {question}
Expected source sections: {expected_sources}
Answer to evaluate: {answer}

Respond with JSON only: {{"accuracy": 0|1, "completeness": 0|1, "citations": 0|1, "reasoning": "..."}}
"""
```

Human expert spot-checks a 20% sample of LLM judge scores per eval run to catch drift.

### Build Process

1. Write 20 questions following the phase 1 distribution (8/5/4/3)
2. For each question, tag `expected_sources` (the doc file + section that contains the answer)
3. Query plain Claude for each question; paste answers into `golden_set.json`; score manually
4. Query docs.pinot.apache.org AI manually for each question; paste answers; score manually
5. Once Sommelier is built: run `eval.py` to get Sommelier scores automatically via LLM judge
6. Target: Sommelier avg ≥ 2.5/3 and beats plain Claude average before declaring V1 done

---

## Migration Path

- V1: `QdrantClient(path="./qdrant_storage")` — single user, fully local, no infra
- V2 (shared service): change to `QdrantClient(url="http://localhost:6333")` — same API, no other code changes
- V3 (cloud): `QdrantClient(url="https://xyz.cloud.qdrant.io", api_key=...)` — same again

---

## Setup (Cold Start)

For a user cloning the repo for the first time:

```bash
# 1. Install dependencies
uv sync

# 2. Configure
cp sommelier.toml.example sommelier.toml
# edit sommelier.toml: set api_key or export OPENAI_API_KEY=...

# 3. Clone the Apache Pinot docs
git clone https://github.com/apache/pinot.git

# 4. Run ingestion (first run indexes everything; subsequent runs are incremental)
python -m sommelier ingest --docs-path ./pinot/website/docs --version 1.2

# 5. Query interactively (multi-turn REPL with conversation memory):
sommelier chat

# MCP wiring (Claude Desktop, Claude Code) is a V1.1 integration — see V1.1 Integrations in Next Steps.
```

Ingestion must complete before `sommelier query` or `sommelier chat` can answer questions. A full first-run index of the Pinot docs takes a few minutes on CPU (FastEmbed, no GPU needed).

---

## Next Steps (in order)


### Observability
2. Write `observability/cli.py`: `python -m sommelier logs --review` — interactive log review with golden set promotion
3. Verify observability is working:
   - After a query: confirm `sommelier_traces.jsonl` contains a full trace with all fields (candidates, reranked, stage_latency_ms, tokens, response)
   - After the same query with `exporter = "langfuse"` and valid keys: confirm the same trace appears in Langfuse UI
   - After an ingest run: confirm an `event_type: "ingestion"` entry appears in `sommelier_traces.jsonl` with correct inserted/deleted/skipped counts
   - With `debug_retrieval = true`: confirm `from_dense`/`from_sparse` fields appear on candidates
   - Run `python -m sommelier logs --review`, promote a query, confirm new entry scaffolded into `golden_set.json`
   - Confirm end-to-end query latency stays under 3 seconds with full tracing enabled

### Evaluations
1. Write `evals/metrics.py`: pure functions — `hit_rate(retrieved: list[str], expected: list[str]) -> float`, `recall(retrieved, expected) -> float`, `mrr(retrieved, expected) -> float`, `full_recall_rate(questions: list[dict]) -> float`; all operate on `file_path` lists; tests in `tests/evals/test_metrics.py`
2. Write `evals/eval.py`: `--retrieval-only` mode (calls search.py, computes metrics via metrics.py, prints two-block retrieval summary — post-reranker and candidates — appends `"eval_result"` events to trace log); default full mode adds LiteLLM judge scoring (JUDGE_PROMPT below) and per-question H@5/R@5/MRR columns in the LLM score table
3. Build golden set: write 20 expert questions (8 how-to, 5 factual, 4 conceptual, 3 comparison); tag expected sources (`file_path` strings matching chunk payload paths); manually score Claude + docs.pinot.apache.org AI responses
4. Run `python evals/eval.py --golden-set evals/golden_set.json --judge claude` after each significant change; V1 is done when:
   - Sommelier average score ≥ 2.5/3 across all 20 golden set questions
   - Sommelier average beats plain Claude average (all three dimensions)
   - Retrieval thresholds (Hit Rate@5, Recall@5, MRR@5): calibrate targets after first eval run based on observed distribution

### Citation Renumbering
1. Post-process completed LLM responses to renumber inline citations sequentially. After all tokens are collected, scan the response text for `[N]` references, assign new sequential numbers `[1]`, `[2]`, `[3]`... in first-appearance order, rewrite both the inline citations and the Sources section entries to use the new numbers. Apply in both `cmd_query` and `cmd_chat`.

### V1 Packaging
1. Write `README.md`: setup instructions (git clone + `uv sync`), `sommelier ingest` usage, `sommelier query` and `sommelier chat` usage, configuration reference (`sommelier.toml`), observability overview

### V1.1 Integrations
1. Publish to PyPI: `uv publish`; verify `uv tool install sommelier` works end-to-end on a clean environment
2. Wire into `mcp_server.py`: `search_pinot(query: str, pinot_version: str = "latest")` tool; internally calls the full query pipeline (retrieval + inference via `llm.py`); returns the finished LLM answer as the tool result; add `tracer.start_trace()` + `tracer.flush()` per request; the MCP client (Claude) echoes the finished answer — no second LLM generation needed
3. Write `.claude/commands/pinot.md`: Claude Code slash command that calls `sommelier query "$1"` via Bash; multi-turn follow-ups are handled by Claude's context window, not by `sommelier`'s own memory
4. Update `README.md`: add PyPI install instructions, MCP client wiring (Claude Desktop `claude_desktop_config.json`), Claude Code skill installation (`.claude/commands/pinot.md`)


## Completed Steps (in order)

### Foundation
1. Set up Python package: `pyproject.toml` with dependencies (managed via `uv`), src layout (`src/ingestion/`, `src/retrieval/`, etc.), CLI entry point (`sommelier ingest / query / logs` via `[project.scripts]`). Uses `setuptools.build_meta` backend; `onnxruntime<1.21.0` pinned for macOS x86_64 compatibility.
2. Write `observability/tracer.py`: `Tracer` class with `start_trace(trace_id, event_type, **kwargs)`, `emit(stage, data)`, `flush()`; `Exporter` Protocol for later use by exporters.py; `prompt_version` computed from `git log -- prompts/system_prompt.md` at init (returns `"unknown"` until that file is committed); module-level `tracer` singleton.

### Ingestion Pipeline
1. Write `vector_store/qdrant_store.py`: `get_client(config)` + `ensure_collection(client, config, collection_name)`; `PINOT_DOCS_COLLECTION` constant for callers. Dense dims looked up from `config.embeddings.model_dims[config.collections[collection_name].dense_model]` — adding new models or collections requires only a config change. Covered by 5 integration tests in `tests/vector_store/test_qdrant_store.py` using a real file-system-backed Qdrant client.
2. Write `embeddings/provider.py`: two protocols (`DenseEmbeddingProvider`, `SparseEmbeddingProvider`), three implementations (`FastEmbedProvider`, `OpenAIProvider`, `BM25Provider`), and factory functions `get_dense_provider(config)` + `get_sparse_provider()`. Sparse is always BM25/FastEmbed regardless of dense provider. OpenAI tests gated on `SOMMELIER_TEST_OPENAI_API_KEY`. 11 integration tests in `tests/embeddings/test_embedding_provider.py`.
3. Write `ingestion/ingest.py`: `clean_gitbook`, `derive_point_id`, `index_file`, and `ingest`. `docs_path` is the root of the cloned pinot-docs repo; `file_path` in each chunk payload is relative to that root. `source_url` removed — GitBook URL mapping is unreliable without site settings access. 12 tests in `tests/ingestion/test_ingestion_pipeline.py` covering pure functions and full insert/skip/replace/delete lifecycle.

### Retrieval Pipeline
1. Write `retrieval/search.py`: `Reranker` Protocol, `FastEmbedReranker` (via `fastembed.rerank.cross_encoder`), `CohereReranker` (optional, gated on `SOMMELIER_TEST_COHERE_API_KEY`), `get_reranker(config)` factory, and `search()` (hybrid dense+sparse+RRF → `retrieval_top_k` candidates → cross-encoder reranking → `rerank_top_k`). Version filter applied in each `Prefetch` (not top-level `query_filter`, which is a no-op in local Qdrant mode with FusionQuery). Debug mode runs 3 queries and populates `from_dense`/`from_sparse` flags. Emits candidates, reranked results, and stage latencies to tracer. 15 tests in `tests/retrieval/test_search.py`; Cohere tests gated on `SOMMELIER_TEST_COHERE_API_KEY`.
2. Write `prompts/system_prompt.md`: initial system prompt per the System Prompt Design. Validated manually via `gpt-4o-mini`: out-of-scope deflection, no-context response, inline `[N]` citations with Sources section, partial-context coverage note, conceptual answers — all correct. URLs must be included in the context block to avoid hallucination in Sources; the production user message format includes them.
3. Write `inference/llm.py`: `build_user_message` (numbered chunks + breadcrumb headers + version + query), `complete` generator (LiteLLM streaming with conversation memory, merges `stage_latency_ms.llm` into existing retrieval latency dict). `getattr(chunk, "usage", None)` required for LiteLLM streaming chunks. 7 tests in `tests/inference/test_llm.py`; integration test gated on `SOMMELIER_TEST_OPENAI_API_KEY`.
4. Test end-to-end via CLI: `sommelier query "What is the default broker port?"` — full pipeline returned correct answer with `[1]` citation. Implemented `load_config` (TOML → nested SimpleNamespace, keeping `model_dims` and `collections` as dicts), `cmd_ingest`, and `cmd_query` in `cli.py`. Updated `prompts/system_prompt.md` to remove URL from Sources (source_url was removed from payloads; model fabricated URLs without it).
5. Implement `sommelier chat` REPL: interactive loop with full query pipeline, in-process conversation history capped at `memory_turns * 2`, `"chat_turn"` trace per turn. Fixed query echo in piped/non-TTY mode (`sys.stdin.isatty()` guard). Updated `prompts/system_prompt.md` Citations section to preserve original context reference numbers, list all cited sources in ascending order, and never skip a citation. Multi-turn follow-up confirmed: turn 2 correctly referenced broker port from turn 1 context. Tests in `tests/test_cli.py` (`TestCmdChatHistoryManagement`).

### Observability
1. Write `observability/exporters.py`: `LocalJSONExporter` (JSON lines + size-based rotation), `LangfuseExporter` (Langfuse v4 API via `start_observation`), and `get_exporters(config)` factory. `langfuse>=4.0.0` added to main dependencies. Wired into `cli.py` via `_setup_tracer(config)` helper called in `cmd_ingest`, `cmd_query`, and `cmd_chat`. Extracted `_init_pipeline(config)` helper to remove repeated pipeline setup. `cli.py` test updated to mock `_setup_tracer` and `_init_pipeline`. 12 tests in `tests/observability/test_exporters.py`; Langfuse integration tests gated on `SOMMELIER_TEST_LANGFUSE_PUBLIC_KEY` + `SOMMELIER_TEST_LANGFUSE_SECRET_KEY`.

