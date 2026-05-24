"""Integration and unit tests for ingestion/ingest.py."""
import types
import uuid

import pytest
from langchain_core.documents import Document

from embeddings.provider import BM25Provider, FastEmbedProvider
from ingestion.ingest import (
    LLMCodeAnnotator,
    _build_breadcrumb,
    _chunk_markdown,
    _insert_annotation,
    _is_code_heavy,
    _merge_code_chunks,
    clean_gitbook,
    derive_point_id,
    index_file,
    ingest,
    normalize_tables,
)
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


class TestNormalizeTables:
    """Unit tests for normalize_tables: verify config-reference tables are converted to prose."""

    _CONFIG_TABLE = (
        "| Property | Default | Description |\n"
        "| --- | --- | --- |\n"
        "| pinot.broker.timeoutMs | 10 seconds | Timeout for broker query. |\n"
        "| pinot.broker.client.queryPort | 8099 | Legacy broker HTTP port. |\n"
    )

    def test_config_table_converted_to_prose(self):
        """A Property/Default/Description table is serialized to one prose line per row."""
        result = normalize_tables(self._CONFIG_TABLE)
        assert "pinot.broker.timeoutMs: default 10 seconds." in result
        assert "Timeout for broker query." in result
        assert "pinot.broker.client.queryPort: default 8099." in result
        assert "Legacy broker HTTP port." in result

    def test_no_markdown_table_syntax_in_output(self):
        """Normalized output contains no pipe characters from the original table."""
        result = normalize_tables(self._CONFIG_TABLE)
        assert "|" not in result

    def test_empty_default_becomes_no_default(self):
        """A row with an empty Default cell produces 'no default' in the prose."""
        table = (
            "| Property | Default | Description |\n"
            "| --- | --- | --- |\n"
            "| pinot.broker.client.access.protocols.http.port |  | Port to query broker via http |\n"
        )
        result = normalize_tables(table)
        assert "no default" in result
        assert "Port to query broker via http" in result

    def test_non_default_table_left_unchanged(self):
        """A table whose second column is not 'Default' is left as-is."""
        table = (
            "| Feature | Status | Notes |\n"
            "| --- | --- | --- |\n"
            "| Upsert | GA | Requires LLC |\n"
        )
        result = normalize_tables(table)
        assert result == table

    def test_prose_outside_table_preserved(self):
        """Text before and after the table is not modified."""
        text = "Before the table.\n" + self._CONFIG_TABLE + "After the table.\n"
        result = normalize_tables(text)
        assert result.startswith("Before the table.")
        assert result.endswith("After the table.\n")

    def test_empty_description_omitted(self):
        """A row with an empty description produces only '<property>: default <value>.'"""
        table = (
            "| Property | Default | Description |\n"
            "| --- | --- | --- |\n"
            "| pinot.broker.enableTableLevelMetrics | true |  |\n"
        )
        result = normalize_tables(table)
        assert result.strip() == "pinot.broker.enableTableLevelMetrics: default true."

    def test_non_table_text_unchanged(self):
        """Plain prose with no table is returned unchanged."""
        text = "The default broker port is 8099.\n"
        assert normalize_tables(text) == text

    def test_multiple_tables_mixed(self):
        """Config tables are normalized while non-config tables are preserved."""
        config_table = (
            "| Property | Default | Description |\n"
            "| --- | --- | --- |\n"
            "| pinot.broker.timeoutMs | 10 seconds | Broker timeout. |\n"
        )
        other_table = (
            "| Table Type | Storage | Notes |\n"
            "| --- | --- | --- |\n"
            "| Offline | Deep store | Batch |\n"
        )
        text = config_table + "\n" + other_table
        result = normalize_tables(text)
        assert "pinot.broker.timeoutMs: default 10 seconds." in result
        assert "| Offline | Deep store | Batch |" in result


def _doc(content: str, **meta) -> Document:
    """Convenience constructor for test Documents."""
    return Document(page_content=content, metadata=meta)


_PURE_CODE = "```json\n{\"dimensionsSplitOrder\": [\"Country\", \"Browser\"]}\n```"
_PROSE_CONFIG = "The dimensionsSplitOrder field controls the split order of dimensions."
_PROSE_OTHER = "Unsupported predicates cannot be used with the star-tree index."


class TestIsCodeHeavy:
    """Unit tests for _is_code_heavy: detect chunks whose content is mostly fenced code."""

    def test_pure_code_block_is_heavy(self):
        """Text entirely inside a fenced block exceeds the 0.5 threshold."""
        assert _is_code_heavy(_PURE_CODE)

    def test_pure_prose_is_not_heavy(self):
        """Text with no fenced blocks is never code-heavy."""
        assert not _is_code_heavy(_PROSE_CONFIG)

    def test_empty_string_is_not_heavy(self):
        """Empty text returns False (no characters, no code)."""
        assert not _is_code_heavy("")

    def test_majority_code_is_heavy(self):
        """A short prose intro followed by a large code block crosses the threshold."""
        text = "Short intro.\n\n```json\n" + ("x" * 200) + "\n```"
        assert _is_code_heavy(text)

    def test_majority_prose_is_not_heavy(self):
        """A large prose block with a tiny code snippet stays below the threshold."""
        text = ("A" * 200) + "\n\n```json\n{}\n```"
        assert not _is_code_heavy(text)


class TestMergeCodeChunks:
    """Unit tests for _merge_code_chunks: semantic merging of code-heavy header splits."""

    def test_no_code_chunks_returns_unchanged(self):
        """A list with no code-heavy chunks is returned as-is."""
        docs = [_doc(_PROSE_CONFIG), _doc(_PROSE_OTHER)]
        result = _merge_code_chunks(docs)
        assert len(result) == 2
        assert result[0].page_content == _PROSE_CONFIG
        assert result[1].page_content == _PROSE_OTHER

    def test_empty_list_returned_unchanged(self):
        """An empty input returns an empty list."""
        assert _merge_code_chunks([]) == []

    def test_all_code_chunks_returned_unchanged(self):
        """When every chunk is code-heavy, nothing can be merged; return as-is."""
        docs = [_doc(_PURE_CODE), _doc(_PURE_CODE)]
        result = _merge_code_chunks(docs)
        assert len(result) == 2

    def test_code_chunk_merges_into_higher_overlap_prose(self):
        """Code chunk merges into the prose chunk that shares more keyword tokens.

        Simulates the q003 star-tree case: the Example JSON is adjacent to an
        unrelated prose chunk but should merge into the configuration prose chunk
        that shares identifiers with the code.
        """
        docs = [
            _doc(_PROSE_OTHER),   # unrelated prose — no shared tokens with code
            _doc(_PURE_CODE),     # code: dimensionsSplitOrder, Country, Browser
            _doc(_PROSE_CONFIG),  # config prose: dimensionsSplitOrder — high overlap
        ]
        result = _merge_code_chunks(docs)
        assert len(result) == 2
        # Code should be appended to the config prose, not the unrelated prose
        assert _PURE_CODE.rstrip() in result[1].page_content
        assert _PURE_CODE.rstrip() not in result[0].page_content

    def test_code_chunk_appended_to_previous_when_higher_overlap(self):
        """Code chunk merges backward into the preceding prose when it has more overlap.

        Simulates the q005 Kafka case: the JSON config follows the prose that
        introduces it, so the previous chunk has higher keyword overlap.
        """
        kafka_prose = "Save the streamConfigs block to configure stream.kafka settings."
        kafka_code = "```json\n{\"streamConfigs\": {\"stream.kafka.consumer.type\": \"lowlevel\"}}\n```"
        unrelated_prose = "Verify that the ingestion pipeline is running correctly."
        docs = [
            _doc(kafka_prose),    # high overlap with kafka_code
            _doc(kafka_code),     # code chunk
            _doc(unrelated_prose),  # low overlap
        ]
        result = _merge_code_chunks(docs)
        assert len(result) == 2
        assert kafka_code.rstrip() in result[0].page_content
        assert kafka_code.rstrip() not in result[1].page_content

    def test_multiple_code_chunks_each_merged_into_best_prose(self):
        """Two code chunks each merge into the prose chunk with the most overlap."""
        kafka_code = "```json\n{\"streamConfigs\": {\"stream.kafka.topic\": \"events\"}}\n```"
        star_tree_code = "```json\n{\"dimensionsSplitOrder\": [\"Country\"]}\n```"
        kafka_prose = "The streamConfigs block controls stream.kafka connection settings."
        star_tree_prose = "The dimensionsSplitOrder controls which dimensions are split."
        docs = [
            _doc(kafka_code),
            _doc(star_tree_code),
            _doc(kafka_prose),
            _doc(star_tree_prose),
        ]
        result = _merge_code_chunks(docs)
        assert len(result) == 2
        kafka_result = next(d for d in result if "streamConfigs" in d.page_content and "```" in d.page_content)
        star_result = next(d for d in result if "dimensionsSplitOrder" in d.page_content and "```" in d.page_content)
        assert "stream.kafka.topic" in kafka_result.page_content
        assert "Country" in star_result.page_content

    def test_plain_column_values_do_not_drive_merge_target(self):
        """Shared plain column values (Country, Browser) do not outscore config keys.

        Validates the technical-token restriction: data-domain words that appear in
        both a code block and an unrelated prose chunk (e.g. a tree-structure
        visualization) must not redirect the merge away from the prose chunk that
        shares actual config-key identifiers with the code.
        """
        plain_value_prose = "Country and Browser columns appear in the tree structure."
        config_prose = "The dimensionsSplitOrder field defines the dimension split order."
        # Code shares Country/Browser with plain_value_prose AND dimensionsSplitOrder
        # with config_prose; only dimensionsSplitOrder is a technical identifier.
        docs = [
            _doc(plain_value_prose),
            _doc(_PURE_CODE),
            _doc(config_prose),
        ]
        result = _merge_code_chunks(docs)
        assert len(result) == 2
        # Code must merge into config_prose (has camelCase overlap), not plain_value_prose
        assert "```json" not in result[0].page_content
        assert "```json" in result[1].page_content
        assert "dimensionsSplitOrder" in result[1].page_content

    def test_prose_chunk_order_preserved(self):
        """Prose chunks appear in their original document order after merging."""
        docs = [
            _doc("First prose chunk."),
            _doc(_PURE_CODE),
            _doc("Second prose chunk."),
        ]
        result = _merge_code_chunks(docs)
        assert len(result) == 2
        # Code is absorbed by one prose chunk; both prose chunks stay in original order
        assert "First prose chunk." in result[0].page_content
        assert "Second prose chunk." in result[1].page_content


class TestBuildBreadcrumb:
    """Unit tests for _build_breadcrumb: section context string from header metadata."""

    def test_all_three_headers(self):
        """h1, h2, h3 are joined with ' > '."""
        meta = {"h1": "Star-tree index", "h2": "Star-tree solution", "h3": "Index generation configuration"}
        assert _build_breadcrumb(meta) == "Star-tree index > Star-tree solution > Index generation configuration"

    def test_missing_h3(self):
        """Missing h3 is omitted; only h1 and h2 are joined."""
        meta = {"h1": "Star-tree index", "h2": "Star-tree solution"}
        assert _build_breadcrumb(meta) == "Star-tree index > Star-tree solution"

    def test_only_h1(self):
        """Only h1 present returns h1 without trailing separator."""
        meta = {"h1": "Concepts"}
        assert _build_breadcrumb(meta) == "Concepts"

    def test_empty_metadata(self):
        """Empty metadata returns an empty string."""
        assert _build_breadcrumb({}) == ""

    def test_empty_header_values_excluded(self):
        """Headers with empty string values are not included in the breadcrumb."""
        meta = {"h1": "Concepts", "h2": "", "h3": ""}
        assert _build_breadcrumb(meta) == "Concepts"


class TestInsertAnnotation:
    """Unit tests for _insert_annotation: annotation placed between breadcrumb and body."""

    def test_inserts_after_first_blank_line(self):
        """Annotation is placed between the breadcrumb and the rest of the content."""
        text = "Star-tree index > Configuration\n\nSome prose content."
        annotation = "Description: Shows config. Question: How do you configure it?"
        result = _insert_annotation(text, annotation)
        assert result == (
            "Star-tree index > Configuration\n\n"
            "Description: Shows config. Question: How do you configure it?\n\n"
            "Some prose content."
        )

    def test_no_blank_line_prepends_annotation(self):
        """When there is no blank line, annotation is prepended to the full text."""
        text = "No blank line here"
        annotation = "Description: X. Question: Y?"
        result = _insert_annotation(text, annotation)
        assert result == "Description: X. Question: Y?\n\nNo blank line here"

    def test_annotation_is_sandwiched_between_parts(self):
        """Original breadcrumb and original body are preserved on either side."""
        text = "Breadcrumb\n\nBody text."
        annotation = "Description: D. Question: Q?"
        result = _insert_annotation(text, annotation)
        assert result == "Breadcrumb\n\nDescription: D. Question: Q?\n\nBody text."


class TestLLMCodeAnnotatorTracking:
    """Unit tests for LLMCodeAnnotator call_count and total_latency_ms tracking."""

    def test_initial_counters_are_zero(self):
        """call_count and total_latency_ms start at zero before any calls."""
        ann = LLMCodeAnnotator(model="openai/gpt-4o-mini")
        assert ann.call_count == 0
        assert ann.total_latency_ms == 0.0

    def test_call_count_increments_on_each_call(self, monkeypatch):
        """call_count increments by 1 for each annotate() invocation."""
        ann = LLMCodeAnnotator(model="openai/gpt-4o-mini")

        class _FakeResponse:
            choices = [types.SimpleNamespace(message=types.SimpleNamespace(
                content="Description: X.\nQuestion: Y?"
            ))]

        def _fake_completion(**kwargs):
            return _FakeResponse()

        monkeypatch.setattr("litellm.completion", _fake_completion)
        ann.annotate("some code")
        ann.annotate("more code")
        assert ann.call_count == 2

    def test_total_latency_ms_accumulates(self, monkeypatch):
        """total_latency_ms is positive and grows after each call."""
        ann = LLMCodeAnnotator(model="openai/gpt-4o-mini")

        class _FakeResponse:
            choices = [types.SimpleNamespace(message=types.SimpleNamespace(
                content="Description: X.\nQuestion: Y?"
            ))]

        def _fake_completion(**kwargs):
            return _FakeResponse()

        monkeypatch.setattr("litellm.completion", _fake_completion)
        ann.annotate("chunk one")
        after_one = ann.total_latency_ms
        ann.annotate("chunk two")
        assert ann.total_latency_ms >= after_one > 0

    def test_call_count_increments_even_on_failure(self, monkeypatch):
        """call_count and total_latency_ms are updated even when the LLM call raises."""
        ann = LLMCodeAnnotator(model="openai/gpt-4o-mini")

        def _raise(**kwargs):
            raise RuntimeError("network error")

        monkeypatch.setattr("litellm.completion", _raise)
        result = ann.annotate("code")
        assert result == ""
        assert ann.call_count == 1
        assert ann.total_latency_ms >= 0


class _StubAnnotator:
    """Stub annotator that counts calls and returns a fixed annotation without calling any LLM."""

    ANNOTATION = "Description: Code example.\nQuestion: How do you configure this?"

    def __init__(self):
        """Initialise call counter and latency accumulator."""
        self.call_count = 0
        self.total_latency_ms = 0.0

    def annotate(self, chunk_text: str) -> str:
        """Return a fixed annotation and increment the call counter."""
        self.call_count += 1
        self.total_latency_ms += 10.0
        return self.ANNOTATION


class TestChunkMarkdownContextual:
    """Unit tests for contextual chunking: breadcrumb is prepended to each chunk's text."""

    def test_breadcrumb_prepended_to_chunks(self):
        """Each chunk's page_content starts with the section breadcrumb."""
        md = "# Star-tree index\n\n## Configuration\n\nSet dimensionsSplitOrder to define split order."
        chunks = _chunk_markdown(md)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.page_content.startswith("Star-tree index")

    def test_breadcrumb_contains_h1_and_h2(self):
        """Chunks from an h2 section have both h1 and h2 in the breadcrumb."""
        md = "# Star-tree index\n\n## Configuration\n\nSet dimensionsSplitOrder to define split order."
        chunks = _chunk_markdown(md)
        assert any(
            "Star-tree index > Configuration" in c.page_content for c in chunks
        )

    def test_original_content_preserved_after_breadcrumb(self):
        """The original chunk text follows the breadcrumb, separated by a blank line."""
        md = "# Concepts\n\n## Table\n\nApache Pinot has two main table types."
        chunks = _chunk_markdown(md)
        combined = " ".join(c.page_content for c in chunks)
        assert "Apache Pinot has two main table types." in combined

    def test_no_breadcrumb_for_empty_metadata(self):
        """Chunks with no header metadata are returned without a breadcrumb prefix."""
        # A markdown document with no headers produces chunks with empty metadata
        md = "Plain text with no headers at all."
        chunks = _chunk_markdown(md)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.page_content == "Plain text with no headers at all."


class TestDerivePointId:
    """Unit tests for derive_point_id: verify content-hash-based deduplication."""

    def test_same_text_and_path_gives_same_id(self):
        """Identical text and file_path always produces the same UUID (stable dedup key)."""
        assert derive_point_id("hello world", "a.md") == derive_point_id("hello world", "a.md")

    def test_different_text_gives_different_id(self):
        """Different text produces different UUIDs."""
        assert derive_point_id("hello world", "a.md") != derive_point_id("goodbye world", "a.md")

    def test_same_text_different_file_gives_different_id(self):
        """Identical chunk text in different files produces different UUIDs."""
        assert derive_point_id("hello world", "a.md") != derive_point_id("hello world", "b.md")

    def test_returns_uuid(self):
        """Output is always a uuid.UUID instance."""
        assert isinstance(derive_point_id("some text", "a.md"), uuid.UUID)


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

    def test_annotates_new_code_heavy_chunks(self, collection_context):
        """New code-heavy chunks are annotated; the stored text contains the annotation."""
        client, dense, sparse = collection_context
        md = "# Indexing\n\n## Star-tree\n\nIntro.\n\n```json\n" + ("x" * 300) + "\n```"
        annotator = _StubAnnotator()
        index_file(
            client=client,
            collection_name=PINOT_DOCS_COLLECTION,
            file_path="test.md",
            raw_markdown=md,
            dense_provider=dense,
            sparse_provider=sparse,
            pinot_version="latest",
            annotator=annotator,
        )
        assert annotator.call_count >= 1
        records, _ = client.scroll(PINOT_DOCS_COLLECTION, with_payload=True, limit=100)
        annotated = [r for r in records if _StubAnnotator.ANNOTATION in r.payload.get("text", "")]
        assert len(annotated) >= 1

    def test_skips_annotation_for_existing_code_heavy_chunks(self, collection_context):
        """Re-indexing unchanged content does not call the annotator again."""
        client, dense, sparse = collection_context
        md = "# Indexing\n\n## Star-tree\n\nIntro.\n\n```json\n" + ("x" * 300) + "\n```"
        annotator = _StubAnnotator()
        index_file(
            client=client,
            collection_name=PINOT_DOCS_COLLECTION,
            file_path="test.md",
            raw_markdown=md,
            dense_provider=dense,
            sparse_provider=sparse,
            pinot_version="latest",
            annotator=annotator,
        )
        calls_after_first_run = annotator.call_count
        index_file(
            client=client,
            collection_name=PINOT_DOCS_COLLECTION,
            file_path="test.md",
            raw_markdown=md,
            dense_provider=dense,
            sparse_provider=sparse,
            pinot_version="latest",
            annotator=annotator,
        )
        assert annotator.call_count == calls_after_first_run

    def test_prose_chunks_not_annotated(self, collection_context):
        """Prose-only chunks are never passed to the annotator."""
        client, dense, sparse = collection_context
        md = "# Concepts\n\n## Table\n\nApache Pinot has two main table types: offline and realtime."
        annotator = _StubAnnotator()
        index_file(
            client=client,
            collection_name=PINOT_DOCS_COLLECTION,
            file_path="test.md",
            raw_markdown=md,
            dense_provider=dense,
            sparse_provider=sparse,
            pinot_version="latest",
            annotator=annotator,
        )
        assert annotator.call_count == 0


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
