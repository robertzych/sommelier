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
│   ├── crawler.py          # Fetch Pinot docs, GitHub issues/PRs, Java code
│   ├── chunker.py          # Per-source chunking strategies
│   └── indexer.py          # Embed + upsert into Qdrant
├── retrieval/
│   └── retriever.py        # Hybrid query: dense + sparse → RRF fusion
├── vector_store/
│   └── qdrant.py           # Qdrant client abstraction (local or Docker)
├── embeddings/
│   └── provider.py         # Swappable: FastEmbed (default) or OpenAI
└── mcp_server.py           # MCP tool: search_pinot(query, pinot_version=None)
```

### Qdrant Collection Schema

```python
client.create_collection(
    collection_name="pinot_docs",
    vectors_config={
        "dense": VectorParams(size=384, distance=Distance.COSINE),
        # size=384 for all-MiniLM-L6-v2; 1536 for text-embedding-3-small
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
    "chunk_index":   3,                     # position within source doc
}
```

### Incremental Update Logic

```python
# content hash is both the point ID and dedup key
point_id = uuid5(NAMESPACE_URL, content_hash)

# upsert: insert if new, overwrite if changed, skip if identical
# "skip if identical" = check existing IDs before upserting
existing_ids = client.retrieve(collection_name, ids=[point_id], with_payload=False)
if not existing_ids:
    client.upsert(collection_name, points=[new_point])
```

### Hybrid Query with RRF

```python
from qdrant_client.models import Prefetch, FusionQuery, Fusion

results = client.query_points(
    collection_name="pinot_docs",
    prefetch=[
        Prefetch(query=dense_vec,  using="dense",  limit=20),
        Prefetch(query=sparse_vec, using="sparse", limit=20),
    ],
    query=FusionQuery(fusion=Fusion.RRF),
    limit=5,
    query_filter=Filter(  # optional version filter
        must=[FieldCondition(key="pinot_version", match=MatchValue(value="1.2"))]
    ) if version else None
)
```

---

## Embedding: FastEmbed (default) + OpenAI (configurable)

**Default (zero API key):**
- Dense: `sentence-transformers/all-MiniLM-L6-v2` via FastEmbed (384 dims, CPU, ~22ms/doc)
- Sparse: `Qdrant/bm25` via FastEmbed (true BM25, no model weights to download)

**Optional (better quality):**
- Dense: `openai/text-embedding-3-small` via OpenAI API (1536 dims, ~$0.02/1M tokens, < $1 for full Pinot docs)

**Config in `sommelier.toml`:**
```toml
[embeddings]
provider = "fastembed"          # or "openai"
dense_model = "all-MiniLM-L6-v2"
openai_api_key = ""             # optional, overrides provider

[vector_store]
backend = "qdrant_local"        # or "qdrant_docker"
path = "./qdrant_storage"
docker_url = "http://localhost:6333"
```

---

## RAG Framework

**Recommendation: LlamaIndex** (but raw SDKs are a strong alternative)

- LlamaIndex has first-class Qdrant support (`QdrantVectorStore`), handles chunking pipelines, and is purpose-built for this use case. Less boilerplate than raw SDKs for the retrieval-augmented generation loop.
- Raw SDKs (openai + qdrant-client directly) would show deeper RAG understanding in a portfolio context, but require more code. Worth revisiting once the pipeline design is clearer.
- LangChain: fine, but heavier and less ergonomic for RAG-specific patterns.

---

## Migration Path

- V1: `QdrantClient(path="./qdrant_storage")` — single user, fully local, no infra
- V2 (shared service): change to `QdrantClient(url="http://localhost:6333")` — same API, no other code changes
- V3 (cloud): `QdrantClient(url="https://xyz.cloud.qdrant.io", api_key=...)` — same again

---

## Next Steps (in order)

1. `pip install qdrant-client[fastembed]` and validate local embedded mode + hybrid search in a notebook
2. Write `provider.py` embedding abstraction (FastEmbed default, OpenAI optional)
3. Write `indexer.py`: crawl Pinot docs → chunk (recursive strategy for V1) → embed → upsert with content hash
4. Write `retriever.py`: hybrid query with RRF, optional version metadata filter
5. Wire into MCP server with a single `search_pinot(query: str, pinot_version: str = "latest")` tool
6. Evaluate against baseline: same query to plain Claude vs. Sommelier, track which answers better

## Verification

- Run a 5-question eval suite against the indexed Pinot docs:
  - Beginner: "How do I start an Apache Pinot cluster in under 10 minutes?"
  - Exact term: "What is a StarTree index and when should I use it?"
  - Config lookup: "What JVM flags control Pinot server memory?"
  - Cross-doc: "What changed in real-time ingestion between Pinot 0.12 and 1.0?"
  - Code: "Show me a Java example of querying Pinot with JDBC"
- Compare Sommelier answers to raw Claude 3.5 Sonnet on the same questions
- All 5 should have Sommelier match or beat raw Claude before declaring V1 done
