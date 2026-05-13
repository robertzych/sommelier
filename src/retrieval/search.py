"""Hybrid search (dense + sparse + RRF) with cross-encoder reranking."""
import time
from typing import Protocol, runtime_checkable

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    Prefetch,
    ScoredPoint,
    SparseVector,
)

from embeddings.provider import DenseEmbeddingProvider, SparseEmbeddingProvider
from observability.tracer import tracer


@runtime_checkable
class Reranker(Protocol):
    """Protocol for cross-encoder rerankers that score (query, document) pairs."""

    def rerank(
        self, query: str, candidates: list[ScoredPoint], top_k: int
    ) -> list[tuple[ScoredPoint, float]]: ...


class FastEmbedReranker:
    """Cross-encoder reranker using FastEmbed ONNX models (no API key required)."""

    def __init__(self, model_name: str) -> None:
        from fastembed.rerank.cross_encoder.text_cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(model_name=model_name)

    def rerank(
        self, query: str, candidates: list[ScoredPoint], top_k: int
    ) -> list[tuple[ScoredPoint, float]]:
        """Score each candidate against the query and return the top_k by descending score."""
        if not candidates:
            return []
        texts = [c.payload["text"] for c in candidates]
        scores = list(self._model.rerank(query, texts))
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


class CohereReranker:
    """Cross-encoder reranker using Cohere Rerank API (requires COHERE_API_KEY)."""

    def __init__(self, api_key: str, model_name: str) -> None:
        import cohere

        self._client = cohere.Client(api_key)
        self._model = model_name

    def rerank(
        self, query: str, candidates: list[ScoredPoint], top_k: int
    ) -> list[tuple[ScoredPoint, float]]:
        """Rerank candidates via Cohere API and return top_k results."""
        if not candidates:
            return []
        documents = [c.payload["text"] for c in candidates]
        response = self._client.rerank(
            query=query, documents=documents, model=self._model, top_n=top_k
        )
        return [(candidates[r.index], r.relevance_score) for r in response.results]


def get_reranker(config) -> Reranker:
    """Return the configured reranker (fastembed or cohere)."""
    if config.retrieval.reranker == "cohere":
        return CohereReranker(
            api_key=config.retrieval.cohere_api_key,
            model_name=config.retrieval.cohere_reranker_model,
        )
    return FastEmbedReranker(model_name=config.retrieval.fastembed_reranker_model)


def _fetch_single_vector_ids(
    client: QdrantClient,
    collection_name: str,
    dense_vec: list[float],
    sparse_vec: SparseVector,
    retrieval_top_k: int,
    version_filter: Filter | None,
) -> tuple[set[str], set[str]]:
    """Run separate dense-only and sparse-only queries; return their result ID sets.

    Used only when observability.debug_retrieval is true to populate
    from_dense/from_sparse flags on candidates.
    """
    dense_result = client.query_points(
        collection_name=collection_name,
        query=dense_vec,
        using="dense",
        limit=retrieval_top_k,
        query_filter=version_filter,
        with_payload=False,
    )
    sparse_result = client.query_points(
        collection_name=collection_name,
        query=sparse_vec,
        using="sparse",
        limit=retrieval_top_k,
        query_filter=version_filter,
        with_payload=False,
    )
    return (
        {str(p.id) for p in dense_result.points},
        {str(p.id) for p in sparse_result.points},
    )


def _emit_retrieval_trace(
    candidates: list[ScoredPoint],
    reranked_with_scores: list[tuple[ScoredPoint, float]],
    retrieval_latency_ms: int,
    rerank_start: float,
    config,
    client: QdrantClient,
    collection_name: str,
    dense_vec: list[float],
    sparse_vec: SparseVector,
    version_filter: Filter | None,
) -> None:
    """Compute rerank latency, optionally fetch debug IDs, and emit the retrieval trace."""
    rerank_latency_ms = int((time.monotonic() - rerank_start) * 1000)

    debug_retrieval = getattr(getattr(config, "observability", None), "debug_retrieval", False)

    dense_ids: set[str] = set()
    sparse_ids: set[str] = set()
    if debug_retrieval:
        dense_ids, sparse_ids = _fetch_single_vector_ids(
            client, collection_name, dense_vec, sparse_vec,
            config.retrieval.retrieval_top_k, version_filter,
        )

    candidate_records = [
        {
            "file_path": c.payload.get("file_path", ""),
            "rrf_score": c.score,
            "snippet": c.payload.get("text", "")[:200],
            **(
                {"from_dense": str(c.id) in dense_ids, "from_sparse": str(c.id) in sparse_ids}
                if debug_retrieval
                else {}
            ),
        }
        for c in candidates
    ]

    reranked_records = [
        {
            "file_path": point.payload.get("file_path", ""),
            "cross_encoder_score": score,
            "snippet": point.payload.get("text", "")[:200],
        }
        for point, score in reranked_with_scores
    ]

    tracer.emit(
        "retrieval",
        {
            "candidates": candidate_records,
            "reranked": reranked_records,
            "stage_latency_ms": {
                "retrieval": retrieval_latency_ms,
                "reranker": rerank_latency_ms,
            },
        },
    )


def search(
    query: str,
    client: QdrantClient,
    collection_name: str,
    dense_provider: DenseEmbeddingProvider,
    sparse_provider: SparseEmbeddingProvider,
    reranker: Reranker,
    config,
    pinot_version: str | None = None,
) -> list[ScoredPoint]:
    """Hybrid search + cross-encoder reranking for a query.

    Stage 1: dense + sparse vectors fused via RRF → top retrieval_top_k candidates.
    Stage 2: cross-encoder reranking → top rerank_top_k results.
    Emits candidates and reranked results to the tracer singleton.
    """
    t0 = time.monotonic()

    dense_vec = dense_provider.embed([query])[0]
    sparse_vec = sparse_provider.embed([query])[0]

    version_filter = (
        Filter(must=[FieldCondition(key="pinot_version", match=MatchValue(value=pinot_version))])
        if pinot_version
        else None
    )

    result = client.query_points(
        collection_name=collection_name,
        prefetch=[
            Prefetch(
                query=dense_vec,
                using="dense",
                limit=config.retrieval.retrieval_top_k,
                filter=version_filter,
            ),
            Prefetch(
                query=sparse_vec,
                using="sparse",
                limit=config.retrieval.retrieval_top_k,
                filter=version_filter,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=config.retrieval.retrieval_top_k,
        with_payload=True,
    )
    candidates = result.points
    retrieval_latency_ms = int((time.monotonic() - t0) * 1000)

    rerank_start = time.monotonic()
    reranked_with_scores = reranker.rerank(query, candidates, top_k=config.retrieval.rerank_top_k)

    _emit_retrieval_trace(
        candidates, reranked_with_scores, retrieval_latency_ms,
        rerank_start, config, client, collection_name,
        dense_vec, sparse_vec, version_filter,
    )

    return [point for point, _ in reranked_with_scores]
