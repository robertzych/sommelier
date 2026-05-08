from typing import Protocol, runtime_checkable

from qdrant_client.models import SparseVector

_BM25_MODEL = "Qdrant/bm25"


@runtime_checkable
class DenseEmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class SparseEmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[SparseVector]: ...


class FastEmbedProvider:
    def __init__(self, dense_model: str) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=dense_model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(texts)]


class OpenAIProvider:
    def __init__(self, dense_model: str, api_key: str) -> None:
        import openai

        self._client = openai.OpenAI(api_key=api_key or None)
        self._model = dense_model

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]


class BM25Provider:
    def __init__(self) -> None:
        from fastembed import SparseTextEmbedding

        self._model = SparseTextEmbedding(model_name=_BM25_MODEL)

    def embed(self, texts: list[str]) -> list[SparseVector]:
        return [
            SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist())
            for emb in self._model.embed(texts)
        ]


def get_dense_provider(config) -> DenseEmbeddingProvider:
    if config.embeddings.provider == "openai":
        return OpenAIProvider(
            dense_model=config.embeddings.dense_model,
            api_key=config.embeddings.openai_api_key,
        )
    return FastEmbedProvider(dense_model=config.embeddings.dense_model)


def get_sparse_provider() -> SparseEmbeddingProvider:
    return BM25Provider()
