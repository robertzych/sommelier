"""Integration and unit tests for ingestion/ingest.py."""
import types
import uuid

import pytest

from embeddings.provider import BM25Provider, FastEmbedProvider
from ingestion.ingest import clean_gitbook, derive_point_id, index_file, ingest
from vector_store.qdrant_store import PINOT_DOCS_COLLECTION, ensure_collection, get_client

_DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_DENSE_DIMS = 384

_SAMPLE_MARKDOWN = """
# Concepts

## Table

### Table Types

Apache Pinot has two main table types: offline and realtime.

### Schema

The schema defines the structure of the data.

## Ingestion

Data ingestion in Apache Pinot supports multiple connectors.
"""

_MODIFIED_MARKDOWN = """
# Concepts

## Table

### Table Types

Apache Pinot has three main table types: offline, realtime, and hybrid.

### Schema

The schema defines the structure of the data.

## Ingestion

Data ingestion in Apache Pinot supports multiple connectors.
"""


# ── Pure function tests ───────────────────────────────────────────────────────


class TestCleanGitBook:
    """Unit tests for clean_gitbook: verify GitBook template directives are stripped."""

    def test_strips_hint_tags(self):
        """Hint blocks are removed; surrounding prose is preserved."""
        raw = "Text\n{% hint style='info' %}\nNote\n{% endhint %}\nMore"
        result = clean_gitbook(raw)
        assert "{% hint" not in result
        assert "Text" in result
        assert "More" in result

    def test_collapses_excess_blank_lines(self):
        """Three or more consecutive blank lines are collapsed to two."""
        raw = "Line one\n\n\n\nLine two"
        assert "\n\n\n" not in clean_gitbook(raw)

    def test_strips_content_ref(self):
        """content-ref blocks are removed; surrounding prose is preserved."""
        raw = "Before\n{% content-ref url='other.md' %}\n{% endcontent-ref %}\nAfter"
        result = clean_gitbook(raw)
        assert "content-ref" not in result
        assert "Before" in result
        assert "After" in result


class TestDerivePointId:
    """Unit tests for derive_point_id: verify content-hash-based deduplication."""

    def test_same_text_gives_same_id(self):
        """Identical text always produces the same UUID (stable dedup key)."""
        assert derive_point_id("hello world") == derive_point_id("hello world")

    def test_different_text_gives_different_id(self):
        """Different text produces different UUIDs."""
        assert derive_point_id("hello world") != derive_point_id("goodbye world")

    def test_returns_uuid(self):
        """Output is always a uuid.UUID instance."""
        assert isinstance(derive_point_id("some text"), uuid.UUID)


# ── Integration tests for index_file ─────────────────────────────────────────


def _make_config(tmp_path, collection_name=PINOT_DOCS_COLLECTION):
    """Build a minimal SimpleNamespace config pointing to a local Qdrant store in tmp_path."""
    return types.SimpleNamespace(
        vector_store=types.SimpleNamespace(
            backend="qdrant_local",
            path=str(tmp_path / "qdrant"),
            docker_url="http://localhost:6333",
        ),
        embeddings=types.SimpleNamespace(
            model_dims={_DENSE_MODEL: _DENSE_DIMS},
            provider="fastembed",
            dense_model=_DENSE_MODEL,
            openai_api_key="",
        ),
        collections={
            collection_name: types.SimpleNamespace(dense_model=_DENSE_MODEL),
        },
    )


@pytest.fixture
def collection_context(tmp_path):
    """Yield (client, dense_provider, sparse_provider) with a fresh collection."""
    config = _make_config(tmp_path)
    client = get_client(config)
    ensure_collection(client, config, PINOT_DOCS_COLLECTION)
    dense = FastEmbedProvider(dense_model=_DENSE_MODEL)
    sparse = BM25Provider()
    return client, dense, sparse


def _run_index(client, dense, sparse, raw_markdown=_SAMPLE_MARKDOWN, file_path="concepts/table.md"):
    """Helper: call index_file with common defaults."""
    return index_file(
        client=client,
        collection_name=PINOT_DOCS_COLLECTION,
        file_path=file_path,
        raw_markdown=raw_markdown,
        dense_provider=dense,
        sparse_provider=sparse,
        pinot_version="1.2",
    )


class TestIndexFile:
    """Integration tests for index_file covering insert, skip, replace, and delete behavior."""

    def test_inserts_chunks_on_first_run(self, collection_context):
        """First run on a new file inserts all chunks with no deletions or skips."""
        client, dense, sparse = collection_context
        stats = _run_index(client, dense, sparse)
        assert stats["inserted"] > 0
        assert stats["deleted"] == 0
        assert stats["skipped"] == 0
        assert stats["errors"] == []

    def test_skips_unchanged_chunks_on_second_run(self, collection_context):
        """Re-indexing the same content inserts nothing; all chunks are skipped."""
        client, dense, sparse = collection_context
        first = _run_index(client, dense, sparse)
        second = _run_index(client, dense, sparse)
        assert second["inserted"] == 0
        assert second["deleted"] == 0
        assert second["skipped"] == first["inserted"]

    def test_replaces_modified_chunks(self, collection_context):
        """Changing a chunk's text causes the old point to be deleted and a new one inserted."""
        client, dense, sparse = collection_context
        _run_index(client, dense, sparse, raw_markdown=_SAMPLE_MARKDOWN)
        second = _run_index(client, dense, sparse, raw_markdown=_MODIFIED_MARKDOWN)
        assert second["deleted"] > 0
        assert second["inserted"] > 0

    def test_chunk_payload_contains_required_fields(self, collection_context):
        """Each indexed point payload contains all required metadata fields."""
        client, dense, sparse = collection_context
        _run_index(client, dense, sparse)
        records, _ = client.scroll(PINOT_DOCS_COLLECTION, with_payload=True, limit=10)
        assert len(records) > 0
        for record in records:
            for field in ("text", "file_path", "doc_type", "pinot_version", "content_hash", "h1", "h2", "h3"):
                assert field in record.payload, f"missing field: {field}"
            assert record.payload["file_path"] == "concepts/table.md"
            assert record.payload["pinot_version"] == "1.2"
            assert record.payload["doc_type"] == "documentation"
        table_types_chunks = [r.payload for r in records if r.payload.get("h3") == "Table Types"]
        assert len(table_types_chunks) == 1
        assert table_types_chunks[0]["h1"] == "Concepts"
        assert table_types_chunks[0]["h2"] == "Table"
        schema_chunks = [r.payload for r in records if r.payload.get("h3") == "Schema"]
        assert len(schema_chunks) == 1
        assert schema_chunks[0]["h1"] == "Concepts"
        assert schema_chunks[0]["h2"] == "Table"
        ingestion_chunks = [r.payload for r in records if r.payload.get("h2") == "Ingestion"]
        assert len(ingestion_chunks) == 1
        assert ingestion_chunks[0]["h1"] == "Concepts"
        assert ingestion_chunks[0]["h3"] == ""

    def test_deletes_all_chunks_when_file_becomes_empty(self, collection_context):
        """Re-indexing an empty file removes all previously indexed chunks for that file."""
        client, dense, sparse = collection_context
        first = _run_index(client, dense, sparse)
        second = _run_index(client, dense, sparse, raw_markdown="")
        assert second["deleted"] == first["inserted"]
        assert second["inserted"] == 0


# ── Integration test for ingest() ────────────────────────────────────────────


class TestIngest:
    """Integration tests for the top-level ingest() orchestrator."""

    def test_walks_docs_root_and_indexes_all_md_files(self, tmp_path):
        """ingest() discovers all .md files under docs_path and indexes them."""
        docs_root = tmp_path / "pinot-docs"
        docs_root.mkdir()
        (docs_root / "concepts.md").write_text("# Concepts\n\nBasic Pinot concepts.")
        (docs_root / "quickstart.md").write_text("# Quickstart\n\nHow to get started.")

        config = _make_config(tmp_path)
        ingest(docs_path=str(docs_root), config=config, pinot_version="1.2")

        client = get_client(config)
        records, _ = client.scroll(PINOT_DOCS_COLLECTION, with_payload=True, limit=100)
        file_paths = {r.payload["file_path"] for r in records}
        assert "concepts.md" in file_paths
        assert "quickstart.md" in file_paths
