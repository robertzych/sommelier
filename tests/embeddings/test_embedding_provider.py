"""Integration tests for embeddings/provider.py using real FastEmbed models."""
import os
import types

import pytest
from qdrant_client.models import SparseVector

from embeddings.provider import (
    BM25Provider,
    DenseEmbeddingProvider,
    FastEmbedProvider,
    OpenAIProvider,
    SparseEmbeddingProvider,
    get_dense_provider,
    get_sparse_provider,
)

# Use a small 384-dim model to keep first-run downloads fast
_DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_DENSE_DIMS = 384

_OPENAI_TEST_API_KEY = os.environ.get("SOMMELIER_TEST_OPENAI_API_KEY")


def _make_config(provider="fastembed", dense_model=_DENSE_MODEL, openai_api_key=""):
    return types.SimpleNamespace(
        embeddings=types.SimpleNamespace(
            provider=provider,
            dense_model=dense_model,
            openai_api_key=openai_api_key,
        )
    )


class TestFastEmbedProvider:
    def test_satisfies_dense_protocol(self):
        provider = FastEmbedProvider(dense_model=_DENSE_MODEL)
        assert isinstance(provider, DenseEmbeddingProvider)

    def test_embed_returns_correct_dims(self):
        provider = FastEmbedProvider(dense_model=_DENSE_MODEL)
        vecs = provider.embed(["test text"])
        assert len(vecs) == 1
        assert len(vecs[0]) == _DENSE_DIMS

    def test_embed_batches_correctly(self):
        provider = FastEmbedProvider(dense_model=_DENSE_MODEL)
        texts = ["first text", "second text", "third text"]
        vecs = provider.embed(texts)
        assert len(vecs) == len(texts)
        assert all(len(v) == _DENSE_DIMS for v in vecs)


class TestBM25Provider:
    def test_satisfies_sparse_protocol(self):
        provider = BM25Provider()
        assert isinstance(provider, SparseEmbeddingProvider)

    def test_embed_returns_sparse_vectors(self):
        provider = BM25Provider()
        vecs = provider.embed(["apache pinot broker port"])
        assert len(vecs) == 1
        sv = vecs[0]
        assert isinstance(sv, SparseVector)
        assert len(sv.indices) > 0
        assert len(sv.values) > 0
        assert len(sv.indices) == len(sv.values)

    def test_embed_batches_correctly(self):
        provider = BM25Provider()
        texts = ["first query", "second query"]
        vecs = provider.embed(texts)
        assert len(vecs) == len(texts)
        assert all(isinstance(v, SparseVector) for v in vecs)


class TestGetDenseProvider:
    def test_returns_fastembed_by_default(self):
        config = _make_config(provider="fastembed")
        provider = get_dense_provider(config)
        assert isinstance(provider, FastEmbedProvider)

    def test_returns_openai_when_configured(self):
        config = _make_config(provider="openai", openai_api_key="sk-fake")
        provider = get_dense_provider(config)
        assert isinstance(provider, OpenAIProvider)


class TestGetSparseProvider:
    def test_returns_bm25_provider(self):
        provider = get_sparse_provider()
        assert isinstance(provider, BM25Provider)


@pytest.mark.skipif(not _OPENAI_TEST_API_KEY, reason="SOMMELIER_TEST_OPENAI_API_KEY not set")
class TestOpenAIProvider:
    def test_embed_returns_correct_dims(self):
        provider = OpenAIProvider(
            dense_model="text-embedding-3-small",
            api_key=_OPENAI_TEST_API_KEY,
        )
        vecs = provider.embed(["test text"])
        assert len(vecs) == 1
        assert len(vecs[0]) == 1536

    def test_embed_batches_correctly(self):
        provider = OpenAIProvider(
            dense_model="text-embedding-3-small",
            api_key=_OPENAI_TEST_API_KEY,
        )
        texts = ["first text", "second text"]
        vecs = provider.embed(texts)
        assert len(vecs) == len(texts)
        assert all(len(v) == 1536 for v in vecs)
