"""Integration and unit tests for retrieval/search.py."""
import os
import types
import uuid

import pytest
from qdrant_client.models import ScoredPoint

from embeddings.provider import BM25Provider, FastEmbedProvider
from ingestion.ingest import index_file
from retrieval.search import (
    CohereReranker,
    FastEmbedReranker,
    Reranker,
    get_reranker,
    search,
)
from vector_store.qdrant_store import PINOT_DOCS_COLLECTION, ensure_collection, get_client

_COHERE_TEST_API_KEY = os.environ.get("SOMMELIER_TEST_COHERE_API_KEY")


def _make_config(tmp_path, reranker="fastembed", debug_retrieval=False):
    """Build a minimal SimpleNamespace config for retrieval tests."""
    return types.SimpleNamespace(
        vector_store=types.SimpleNamespace(
            backend="qdrant_local",
            path=str(tmp_path / "qdrant"),
            docker_url="http://localhost:6333",
        ),
        embeddings=types.SimpleNamespace(
            model_dims={"sentence-transformers/all-MiniLM-L6-v2": 384},
            provider="fastembed",
            dense_model="sentence-transformers/all-MiniLM-L6-v2",
            openai_api_key="",
        ),
        collections={
            PINOT_DOCS_COLLECTION: types.SimpleNamespace(
                dense_model="sentence-transformers/all-MiniLM-L6-v2"
            ),
        },
        retrieval=types.SimpleNamespace(
            retrieval_top_k=10,
            rerank_top_k=3,
            reranker=reranker,
            fastembed_reranker_model="Xenova/ms-marco-MiniLM-L-6-v2",
            cohere_api_key=_COHERE_TEST_API_KEY or "",
            cohere_reranker_model="rerank-multilingual-v3.0",
        ),
        observability=types.SimpleNamespace(debug_retrieval=debug_retrieval),
    )


def _make_scored_point(text: str, score: float = 0.5) -> ScoredPoint:
    """Build a ScoredPoint with minimal payload for reranker unit tests."""
    return ScoredPoint(
        id=str(uuid.uuid4()),
        version=0,
        score=score,
        payload={"text": text, "file_path": "docs/test.md"},
        vector=None,
    )


@pytest.fixture
def indexed_collection(tmp_path):
    """Yield (client, dense_provider, sparse_provider, config) with pre-indexed documents."""
    config = _make_config(tmp_path)
    client = get_client(config)
    ensure_collection(client, config, PINOT_DOCS_COLLECTION)
    dense = FastEmbedProvider(dense_model=config.embeddings.dense_model)
    sparse = BM25Provider()

    docs = {
        "broker/config.md": (
            "# Broker\n\n## Configuration\n\nThe default Pinot broker port is 8099. "
            "To change it, set broker.port in broker.conf."
        ),
        "concepts/table.md": (
            "# Concepts\n\n## Table\n\nApache Pinot has offline and realtime tables. "
            "Offline tables load data from batch sources."
        ),
        "quickstart.md": (
            "# Quickstart\n\nInstall Apache Pinot and run the quick start example "
            "to verify your setup is working correctly."
        ),
    }
    for file_path, content in docs.items():
        index_file(
            client=client,
            collection_name=PINOT_DOCS_COLLECTION,
            file_path=file_path,
            raw_markdown=content,
            dense_provider=dense,
            sparse_provider=sparse,
            pinot_version="1.2",
        )

    return client, dense, sparse, config


# ── Reranker unit tests ───────────────────────────────────────────────────────


class TestFastEmbedReranker:
    """Unit tests for FastEmbedReranker using real ONNX model scoring."""

    def test_satisfies_reranker_protocol(self, tmp_path):
        """FastEmbedReranker implements the Reranker protocol."""
        config = _make_config(tmp_path)
        reranker = FastEmbedReranker(model_name=config.retrieval.fastembed_reranker_model)
        assert isinstance(reranker, Reranker)

    def test_rerank_returns_top_k(self, tmp_path):
        """rerank returns exactly top_k results when more candidates are provided."""
        config = _make_config(tmp_path)
        reranker = FastEmbedReranker(model_name=config.retrieval.fastembed_reranker_model)
        candidates = [
            _make_scored_point("The broker port is 8099."),
            _make_scored_point("Offline tables load batch data."),
            _make_scored_point("Quickstart installs Pinot."),
            _make_scored_point("Schema defines the data structure."),
        ]
        results = reranker.rerank("What is the broker port?", candidates, top_k=2)
        assert len(results) == 2

    def test_rerank_returns_scored_point_float_tuples(self, tmp_path):
        """rerank returns a list of (ScoredPoint, float) tuples."""
        config = _make_config(tmp_path)
        reranker = FastEmbedReranker(model_name=config.retrieval.fastembed_reranker_model)
        candidates = [_make_scored_point("Broker port is 8099.")]
        results = reranker.rerank("broker port", candidates, top_k=1)
        assert len(results) == 1
        point, score = results[0]
        assert isinstance(point, ScoredPoint)
        assert isinstance(score, float)

    def test_rerank_empty_candidates_returns_empty(self, tmp_path):
        """rerank returns an empty list when no candidates are provided."""
        config = _make_config(tmp_path)
        reranker = FastEmbedReranker(model_name=config.retrieval.fastembed_reranker_model)
        assert reranker.rerank("broker port", [], top_k=3) == []

    def test_rerank_ranks_relevant_chunk_highest(self, tmp_path):
        """The most query-relevant candidate receives the highest cross-encoder score."""
        config = _make_config(tmp_path)
        reranker = FastEmbedReranker(model_name=config.retrieval.fastembed_reranker_model)
        relevant = _make_scored_point("The default Pinot broker port is 8099.")
        irrelevant = _make_scored_point("Quickstart installs Apache Pinot on your machine.")
        results = reranker.rerank(
            "What is the default broker port?", [irrelevant, relevant], top_k=2
        )
        top_point, _ = results[0]
        assert top_point.id == relevant.id


# ── CohereReranker integration tests ─────────────────────────────────────────


@pytest.mark.skipif(not _COHERE_TEST_API_KEY, reason="SOMMELIER_TEST_COHERE_API_KEY not set")
class TestCohereReranker:
    """Integration tests for CohereReranker using the live Cohere Rerank API."""

    def test_satisfies_reranker_protocol(self, tmp_path):
        """CohereReranker implements the Reranker protocol."""
        config = _make_config(tmp_path, reranker="cohere")
        reranker = CohereReranker(
            api_key=config.retrieval.cohere_api_key,
            model_name=config.retrieval.cohere_reranker_model,
        )
        assert isinstance(reranker, Reranker)

    def test_rerank_returns_top_k_with_scores(self, tmp_path):
        """rerank returns exactly top_k (ScoredPoint, float) tuples via Cohere API."""
        config = _make_config(tmp_path, reranker="cohere")
        reranker = CohereReranker(
            api_key=config.retrieval.cohere_api_key,
            model_name=config.retrieval.cohere_reranker_model,
        )
        candidates = [
            _make_scored_point("The default Pinot broker port is 8099."),
            _make_scored_point("Quickstart installs Apache Pinot on your machine."),
            _make_scored_point("Offline tables load batch data from external sources."),
        ]
        results = reranker.rerank("What is the broker port?", candidates, top_k=2)
        assert len(results) == 2
        for point, score in results:
            assert isinstance(point, ScoredPoint)
            assert isinstance(score, float)

    def test_rerank_empty_candidates_returns_empty(self, tmp_path):
        """rerank returns an empty list when no candidates are provided."""
        config = _make_config(tmp_path, reranker="cohere")
        reranker = CohereReranker(
            api_key=config.retrieval.cohere_api_key,
            model_name=config.retrieval.cohere_reranker_model,
        )
        assert reranker.rerank("broker port", [], top_k=3) == []

    def test_rerank_ranks_relevant_chunk_highest(self, tmp_path):
        """The most query-relevant candidate receives the highest Cohere relevance score."""
        config = _make_config(tmp_path, reranker="cohere")
        reranker = CohereReranker(
            api_key=config.retrieval.cohere_api_key,
            model_name=config.retrieval.cohere_reranker_model,
        )
        relevant = _make_scored_point("The default Pinot broker port is 8099.")
        irrelevant = _make_scored_point("Quickstart installs Apache Pinot on your machine.")
        results = reranker.rerank(
            "What is the default broker port?", [irrelevant, relevant], top_k=2
        )
        top_point, _ = results[0]
        assert top_point.id == relevant.id


# ── get_reranker factory tests ────────────────────────────────────────────────


class TestGetReranker:
    """Unit tests for the get_reranker factory function."""

    def test_returns_fastembed_reranker_by_default(self, tmp_path):
        """get_reranker returns FastEmbedReranker when reranker = 'fastembed'."""
        config = _make_config(tmp_path, reranker="fastembed")
        reranker = get_reranker(config)
        assert isinstance(reranker, FastEmbedReranker)

    @pytest.mark.skipif(not _COHERE_TEST_API_KEY, reason="SOMMELIER_TEST_COHERE_API_KEY not set")
    def test_returns_cohere_reranker_when_configured(self, tmp_path):
        """get_reranker returns CohereReranker when reranker = 'cohere'."""
        config = _make_config(tmp_path, reranker="cohere")
        reranker = get_reranker(config)
        assert isinstance(reranker, CohereReranker)


# ── search() integration tests ────────────────────────────────────────────────


class TestSearch:
    """Integration tests for search() using real Qdrant, embeddings, and reranker."""

    def test_search_returns_rerank_top_k_results_with_required_payload_fields(
        self, indexed_collection
    ):
        """search returns exactly rerank_top_k ScoredPoints each with text and file_path."""
        client, dense, sparse, config = indexed_collection
        reranker = FastEmbedReranker(model_name=config.retrieval.fastembed_reranker_model)
        results = search(
            query="What is the default broker port?",
            client=client,
            collection_name=PINOT_DOCS_COLLECTION,
            dense_provider=dense,
            sparse_provider=sparse,
            reranker=reranker,
            config=config,
        )
        assert len(results) == config.retrieval.rerank_top_k
        assert all(isinstance(p, ScoredPoint) for p in results)
        for point in results:
            assert "text" in point.payload
            assert "file_path" in point.payload

    def test_search_with_version_filter_excludes_other_version_chunks(
        self, indexed_collection
    ):
        """search with pinot_version=1.2 excludes chunks that would appear unfiltered."""
        client, dense, sparse, config = indexed_collection

        # A unique term that only appears in the v0.12 doc, ensuring it would rank
        # in hybrid search results when queried without a version filter.
        unique_term = "xyzzy-legacy-connector-v012"
        index_file(
            client=client,
            collection_name=PINOT_DOCS_COLLECTION,
            file_path="v0.12/legacy.md",
            raw_markdown=f"# Legacy\n\nThe {unique_term} was deprecated in version 1.0.",
            dense_provider=dense,
            sparse_provider=sparse,
            pinot_version="0.12",
        )

        reranker = FastEmbedReranker(model_name=config.retrieval.fastembed_reranker_model)

        # Without a version filter, the v0.12 chunk appears in results.
        unfiltered = search(
            query=unique_term,
            client=client,
            collection_name=PINOT_DOCS_COLLECTION,
            dense_provider=dense,
            sparse_provider=sparse,
            reranker=reranker,
            config=config,
        )
        unfiltered_versions = {p.payload.get("pinot_version") for p in unfiltered}
        assert "0.12" in unfiltered_versions

        # With pinot_version="1.2", the v0.12 chunk is excluded.
        filtered = search(
            query=unique_term,
            client=client,
            collection_name=PINOT_DOCS_COLLECTION,
            dense_provider=dense,
            sparse_provider=sparse,
            reranker=reranker,
            config=config,
            pinot_version="1.2",
        )
        for point in filtered:
            assert point.payload.get("pinot_version") == "1.2"

    def test_search_emits_trace_candidates_and_reranked(self, indexed_collection):
        """search emits candidates and reranked lists to the tracer."""
        from observability.tracer import tracer

        client, dense, sparse, config = indexed_collection
        reranker = FastEmbedReranker(model_name=config.retrieval.fastembed_reranker_model)

        tracer.start_trace("test-trace", event_type="query")
        search(
            query="broker port",
            client=client,
            collection_name=PINOT_DOCS_COLLECTION,
            dense_provider=dense,
            sparse_provider=sparse,
            reranker=reranker,
            config=config,
        )

        assert "candidates" in tracer._trace
        assert "reranked" in tracer._trace
        assert "stage_latency_ms" in tracer._trace
        assert len(tracer._trace["candidates"]) > 0
        assert len(tracer._trace["reranked"]) > 0

        candidate = tracer._trace["candidates"][0]
        assert "file_path" in candidate
        assert "rrf_score" in candidate
        assert "snippet" in candidate
        assert "from_dense" not in candidate  # debug_retrieval is False

        reranked = tracer._trace["reranked"][0]
        assert "file_path" in reranked
        assert "cross_encoder_score" in reranked
        assert "snippet" in reranked

        tracer._trace = {}  # clean up without flushing

    def test_search_debug_retrieval_adds_from_dense_from_sparse(self, tmp_path):
        """With debug_retrieval=True, candidate records include from_dense and from_sparse flags."""
        from observability.tracer import tracer

        config = _make_config(tmp_path, debug_retrieval=True)
        client = get_client(config)
        ensure_collection(client, config, PINOT_DOCS_COLLECTION)
        dense = FastEmbedProvider(dense_model=config.embeddings.dense_model)
        sparse = BM25Provider()

        index_file(
            client=client,
            collection_name=PINOT_DOCS_COLLECTION,
            file_path="broker/config.md",
            raw_markdown="# Broker\n\nThe default Pinot broker port is 8099.",
            dense_provider=dense,
            sparse_provider=sparse,
            pinot_version="1.2",
        )

        reranker = FastEmbedReranker(model_name=config.retrieval.fastembed_reranker_model)
        tracer.start_trace("debug-trace", event_type="query")
        search(
            query="broker port",
            client=client,
            collection_name=PINOT_DOCS_COLLECTION,
            dense_provider=dense,
            sparse_provider=sparse,
            reranker=reranker,
            config=config,
        )

        assert len(tracer._trace["candidates"]) > 0
        candidate = tracer._trace["candidates"][0]
        assert "from_dense" in candidate
        assert "from_sparse" in candidate
        assert isinstance(candidate["from_dense"], bool)
        assert isinstance(candidate["from_sparse"], bool)

        tracer._trace = {}  # clean up without flushing
