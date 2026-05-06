# Sommelier: Vector Database Decision

## Context

Sommelier is a RAG-based chatbot for Apache Pinot guidance, distributed as a public GitHub repo that users run locally as an MCP server. The design phase is ongoing with no code yet. This plan resolves the open checkbox: _"Choose a vector database, chunking strategy, and embedding/inference models"_ — specifically the vector DB and its supporting architecture.

The decision hinges on three constraints discovered during the interview:
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
│   ├── loader.py           # Load docs from cloned GitHub repo; GitHub API for issues/PRs/code
│   ├── chunker.py          # GitBook pre-process → header-aware split → recursive split to ~512 tokens
│   └── indexer.py          # Embed (dense + sparse) + delete stale + insert new into Qdrant
├── retrieval/
│   ├── retriever.py        # Hybrid search: dense + sparse → RRF → top 20 candidates
│   ├── reranker.py         # Cross-encoder reranking: top 20 → top 5 (FastEmbed or Cohere)
│   └── router.py           # LLM query classifier → adjusts rerank_top_k per query type
├── inference/
│   └── llm.py              # LiteLLM wrapper: GPT-4o-mini (default) or Ollama llama3.1:8b (local)
├── vector_store/
│   └── qdrant.py           # Qdrant client abstraction (local or Docker)
├── embeddings/
│   └── provider.py         # FastEmbed (default) or OpenAI; swappable via config
├── evals/
│   ├── golden_set.json     # Expert-written questions, expected sources, baseline scores
│   └── eval.py             # Eval runner: queries Sommelier, LLM judge scoring, comparison report
└── mcp_server.py           # MCP tool: search_pinot(query, pinot_version=None)
```

### Qdrant Collection Schema

```python
client.create_collection(
    collection_name="pinot_docs",
    vectors_config={
        "dense": VectorParams(size=768, distance=Distance.COSINE),
        # 768 = bge-base-en-v1.5 (default); 384 = all-MiniLM-L6-v2 (fast mode); 1536 = text-embedding-3-small
        # changing dense_model requires dropping and recreating this collection
    },
    sparse_vectors_config={
        "sparse": SparseVectorParams()  # BM25 via FastEmbed Qdrant/bm25
    }
)
```

### Chunk Payload (metadata)

```python
payload = {
    "text":          chunk_text,
    "source_url":    "https://docs.pinot.apache.org/...",
    "doc_type":      "documentation",       # or "github_issue", "java_code"
    "pinot_version": "1.2",                 # stored as metadata, default=latest
    "content_hash":  sha256(chunk_text),    # used as point ID for dedup
}
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

### Retrieval Pipeline (two-stage)

```python
# Stage 1 — LLM query classifier (fast, cheap)
query_type = router.classify(query)          # "specific" | "broad"
final_top_k = config.rerank_top_k if query_type == "specific" else config.rerank_top_k * 2

# Stage 2 — Hybrid search: top retrieval_top_k candidates (default 20)
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

# Stage 3 — Cross-encoder reranking: top retrieval_top_k → final_top_k
reranked = reranker.rerank(query, candidates, top_k=final_top_k)
```

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

char_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
chunks = char_splitter.split_documents(header_splits)
```

Section titles (`h1`, `h2`, `h3`) are stored as metadata on every chunk — used for source citations and section-level filtering.

### Updated Chunk Payload

```python
payload = {
    "text":          chunk_text,
    "source_url":    "https://docs.pinot.apache.org/...",  # canonical URL for citation
    "file_path":     "docs/basics/concepts/table.md",      # relative path in repo
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

**Conversation memory:** last `memory_turns` (default 5) turns included in each call.

**Response format:** markdown with source citations. System prompt instructs the model to cite section titles and URLs from retrieved chunk metadata.

---

## Configuration (`sommelier.toml`)

```toml
[embeddings]
provider = "fastembed"               # fastembed | openai
dense_model = "BAAI/bge-base-en-v1.5"  # change requires re-indexing
# dense_model = "all-MiniLM-L6-v2"  # fast mode
openai_api_key = ""                  # required when provider = "openai"

[inference]
provider = "openai"                  # openai | ollama | anthropic | gemini (LiteLLM strings)
model = "gpt-4o-mini"                # "ollama/llama3.1:8b" for fully local
api_key = ""                         # not needed for Ollama
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
```

---

## RAG Framework

**Recommendation: LlamaIndex** (but raw SDKs are a strong alternative)

- LlamaIndex has first-class Qdrant support (`QdrantVectorStore`), handles chunking pipelines, and is purpose-built for this use case. Less boilerplate than raw SDKs for the retrieval-augmented generation loop.
- Raw SDKs (openai + qdrant-client directly) would show deeper RAG understanding in a portfolio context, but require more code. Worth revisiting once the pipeline design is clearer.
- LangChain: fine, but heavier and less ergonomic for RAG-specific patterns.

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

```
python evals/eval.py --golden-set evals/golden_set.json --judge claude
```

Output:
```
ID     Question (truncated)                    Sommelier  Claude  Docs AI
q001   How do I configure an upsert table?    3/3        1/3     2/3
q002   What is the default broker port?       3/3        2/3     3/3
...
Avg                                            2.7/3      1.5/3   2.2/3
Retrieval precision (expected sources hit)    87%        n/a     n/a
```

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

## Next Steps (in order)

1. Validate in a notebook: `pip install qdrant-client[fastembed] litellm langchain-text-splitters` → test local Qdrant + hybrid search + bge-base-en-v1.5 + BM25 + FastEmbed cross-encoder reranking end-to-end
2. Write `embeddings/provider.py`: FastEmbed wrapper (bge-base default, all-MiniLM fast mode) + OpenAI provider; reads config from `sommelier.toml`
3. Write `ingestion/chunker.py`: strip GitBook syntax → `MarkdownHeaderTextSplitter` → `RecursiveCharacterTextSplitter` (512 tokens, 50 overlap); propagate `h1`/`h2`/`h3` as metadata
4. Write `ingestion/loader.py` + `ingestion/indexer.py`: read `.md` files from cloned Pinot docs repo → chunk → embed (dense + sparse) → per file: scroll existing IDs, delete stale IDs, insert new chunks by content-hash point ID
5. Write `retrieval/retriever.py`: hybrid search (dense + sparse + RRF), returns top `retrieval_top_k` candidates
6. Write `retrieval/reranker.py`: FastEmbed cross-encoder (default) or Cohere Rerank (optional); narrows to `rerank_top_k`
7. Write `retrieval/router.py`: LLM query classifier → "specific" | "broad" → adjusts `rerank_top_k`
8. Write `inference/llm.py`: LiteLLM wrapper with conversation memory (`memory_turns`), markdown+citations response format, streaming
9. Wire into `mcp_server.py`: `search_pinot(query: str, pinot_version: str = "latest")` tool
10. Build golden set: write 20 expert questions (8 how-to, 5 factual, 4 conceptual, 3 comparison); tag expected sources; manually score Claude + docs.pinot.apache.org AI responses
11. Run `eval.py` once Sommelier is wired up; target avg ≥ 2.5/3 and beat plain Claude average before V1 is done

## Verification

Run `python evals/eval.py --golden-set evals/golden_set.json --judge claude` after each significant change.

**V1 exit criteria:**
- Sommelier average score ≥ 2.5/3 across all 20 golden set questions
- Sommelier average beats plain Claude average (all three dimensions)
- Retrieval precision (expected sources hit) ≥ 80%

The 5 sample questions from earlier planning are good seeds for the factual and comparison categories in the golden set.
