from qdrant_client import QdrantClient
from qdrant_client.models import Distance, SparseVectorParams, VectorParams

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
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(size=dense_dim, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(),
        },
    )
