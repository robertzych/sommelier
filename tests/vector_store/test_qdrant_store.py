"""Integration tests for vector_store/qdrant_store.py using a real local Qdrant client."""
import types

import pytest
from qdrant_client.models import PointStruct, SparseVector

from vector_store.qdrant_store import PINOT_DOCS_COLLECTION, ensure_collection, get_client


def _make_config(tmp_path, collection_dense_model="BAAI/bge-base-en-v1.5"):
    return types.SimpleNamespace(
        vector_store=types.SimpleNamespace(
            backend="qdrant_local",
            path=str(tmp_path),
            docker_url="http://localhost:6333",
        ),
        embeddings=types.SimpleNamespace(
            model_dims={"BAAI/bge-base-en-v1.5": 768, "all-MiniLM-L6-v2": 384},
        ),
        collections={
            PINOT_DOCS_COLLECTION: types.SimpleNamespace(dense_model=collection_dense_model),
        },
    )


class TestGetClient:
    def test_local_backend_creates_client(self, tmp_path):
        config = _make_config(tmp_path)
        client = get_client(config)
        client.get_collections()

    def test_local_backend_persists_to_path(self, tmp_path):
        config = _make_config(tmp_path)
        client = get_client(config)
        client.get_collections()
        assert any(tmp_path.iterdir())


class TestEnsureCollection:
    def test_creates_collection_when_missing(self, tmp_path):
        config = _make_config(tmp_path)
        client = get_client(config)

        assert not client.collection_exists(PINOT_DOCS_COLLECTION)
        ensure_collection(client, config, PINOT_DOCS_COLLECTION)
        assert client.collection_exists(PINOT_DOCS_COLLECTION)

    def test_no_error_when_called_twice(self, tmp_path):
        config = _make_config(tmp_path)
        client = get_client(config)

        ensure_collection(client, config, PINOT_DOCS_COLLECTION)
        ensure_collection(client, config, PINOT_DOCS_COLLECTION)

    def test_collection_accepts_dense_and_sparse_points(self, tmp_path):
        config = _make_config(tmp_path)
        client = get_client(config)
        ensure_collection(client, config, PINOT_DOCS_COLLECTION)

        point = PointStruct(
            id=1,
            vector={
                "dense": [0.1] * 768,
                "sparse": SparseVector(indices=[0, 1], values=[0.5, 0.3]),
            },
            payload={"text": "test chunk"},
        )
        client.upsert(collection_name=PINOT_DOCS_COLLECTION, points=[point])

        results = client.retrieve(
            collection_name=PINOT_DOCS_COLLECTION,
            ids=[1],
            with_payload=True,
        )
        assert len(results) == 1
        assert results[0].payload["text"] == "test chunk"
