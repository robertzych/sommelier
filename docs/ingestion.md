# Ingestion Pipeline

Documents are sourced from the [pinot-contrib/pinot-docs](https://github.com/pinot-contrib/pinot-docs) GitHub repo (a standalone GitBook repo, separate from the main Pinot monorepo).

## Chunking

Each `.md` file is cleaned first — GitBook template tags (`{% ... %}`) are stripped and excess blank lines collapsed — then split in two stages: `MarkdownHeaderTextSplitter` divides the document at H1/H2/H3 headers, carrying each header forward as metadata on its child chunks; `RecursiveCharacterTextSplitter` further bounds each section to 512 tokens with 50-token overlap (tiktoken-encoded). The 50-token overlap preserves sentence continuity at split boundaries without duplicating full paragraphs. Header metadata is then serialized into a section breadcrumb (`H1 > H2 > H3`) and prepended to each chunk's text before embedding, so BM25 and dense models see document structure — not just local content.

Two retrieval problems emerged during evaluation and required additional transforms within this pipeline:

## Markdown Table Normalization

Pinot's config reference docs store property names, default values, and descriptions in separate markdown columns; when chunked, the semantic link between a property and its default is lost. A query for the default broker port couldn't retrieve the ideal chunk from from the expected source document. The original chunk contained the following pipe-delimited text:

```
| pinot.broker.client.queryPort                                   | 8099                                                                  | **(Deprecated: use `pinot.broker.client.access.protocols.http.port` instead.)** Legacy port to query broker via http.
```

The fix detects tables whose second column header contains "default" and merges the row into one retrievable string:

```
pinot.broker.client.queryPort: default 8099. **(Deprecated: use `pinot.broker.client.access.protocols.http.port` instead.)** Legacy port to query broker via http.
```

## LLM Code Annotation

A code-heavy chunk has minimal prose for BM25 or dense embeddings to match against. For the star-tree index question, the file was retrieved via its prose intro chunks (MRR@5 0.20) but the Example chunk containing the complete `tableIndexConfig.starTreeIndexConfigs` JSON was not in the top-20 candidates at all, causing the LLM to respond with a placeholder instead of real config.

Fix: for code-heavy chunks (≥50% code characters), the LLM generates a one-sentence `Description:` and one representative `Question:` inserted between a section breadcrumb and the chunk body — giving hybrid search the natural-language signal needed to retrieve it:

```
Star-Tree Index > Configuration > Example
Description: Star-tree index configuration specifying dimension split order and aggregation functions.
Question: How do I configure a star-tree index with custom split order and sum aggregation?
[JSON code block]
```

## Embedding

Each processed chunk is encoded into two representations before being stored in Qdrant: a dense vector (`BAAI/bge-base-en-v1.5`, 768 dims, FastEmbed) for semantic similarity and a sparse BM25 vector (`Qdrant/bm25`, FastEmbed) for keyword matching. The dense model is configurable (e.g. `all-MiniLM-L6-v2` for speed, `text-embedding-3-small` for higher quality); changing it requires re-ingesting since vector dimensions are fixed at collection creation.

## Incremental Updates

(`src/ingestion/ingest.py`): Point IDs are SHA-256 content hashes of the chunk text, cast to UUIDs — deterministic and unique per chunk. When a doc file changes, modified chunks get new IDs; the old IDs become orphans. Per file: scroll existing point IDs, diff against new, delete orphans, insert new. Unchanged chunks (same ID already in Qdrant) are skipped entirely. No updates — only inserts and deletes.

```
File unchanged → skip (IDs already in Qdrant)
File modified  → old IDs deleted + new IDs inserted
File deleted   → old IDs deleted
New file       → all IDs inserted
```

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
