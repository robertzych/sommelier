import argparse
import tomllib
import types
import uuid

from observability.tracer import tracer


def load_config(path: str = "sommelier.toml") -> types.SimpleNamespace:
    """Load sommelier.toml and return a typed config namespace."""
    with open(path, "rb") as f:
        data = tomllib.load(f)

    emb_data = data.get("embeddings", {})
    model_dims = emb_data.get("model_dims", {})
    emb_fields = {k: v for k, v in emb_data.items() if k != "model_dims"}

    collections_raw = data.get("collections", {})
    collections = {name: types.SimpleNamespace(**vals) for name, vals in collections_raw.items()}

    return types.SimpleNamespace(
        embeddings=types.SimpleNamespace(**emb_fields, model_dims=model_dims),
        inference=types.SimpleNamespace(**data.get("inference", {})),
        retrieval=types.SimpleNamespace(**data.get("retrieval", {})),
        vector_store=types.SimpleNamespace(**data.get("vector_store", {})),
        pinot=types.SimpleNamespace(**data.get("pinot", {"version": "latest"})),
        observability=types.SimpleNamespace(**data.get("observability", {})),
        collections=collections,
    )


def cmd_ingest(args):
    """Index all .md files in docs_path into Qdrant."""
    from ingestion.ingest import ingest

    config = load_config()
    ingest(docs_path=args.docs_path, config=config, pinot_version=args.version)


def cmd_query(args):
    """Run a single query through the full search → rerank → llm pipeline."""
    from embeddings.provider import get_dense_provider, get_sparse_provider
    from inference.llm import complete
    from retrieval.search import get_reranker, search
    from vector_store.qdrant_store import PINOT_DOCS_COLLECTION, ensure_collection, get_client

    config = load_config()
    pinot_version = None if args.version == "latest" else args.version

    tracer.start_trace(
        str(uuid.uuid4()),
        event_type="query",
        query=args.question,
        pinot_version=args.version,
    )

    client = get_client(config)
    ensure_collection(client, config, PINOT_DOCS_COLLECTION)
    dense = get_dense_provider(config)
    sparse = get_sparse_provider()
    reranker = get_reranker(config)

    chunks = search(
        query=args.question,
        client=client,
        collection_name=PINOT_DOCS_COLLECTION,
        dense_provider=dense,
        sparse_provider=sparse,
        reranker=reranker,
        config=config,
        pinot_version=pinot_version,
    )

    for token in complete(args.question, chunks, [], config, pinot_version=pinot_version):
        print(token, end="", flush=True)
    print()

    tracer.flush()


def cmd_logs(args):
    raise NotImplementedError("log review not yet implemented")


def main():
    parser = argparse.ArgumentParser(prog="sommelier")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Index Apache Pinot docs into Qdrant")
    ingest.add_argument("--docs-path", required=True, help="Path to the pinot-docs repo root")
    ingest.add_argument("--version", default="latest", help="Pinot version tag to store on chunks")
    ingest.set_defaults(func=cmd_ingest)

    query = sub.add_parser("query", help="Run a one-shot query through the full pipeline")
    query.add_argument("question", help="Question to ask Sommelier")
    query.add_argument("--version", default="latest", help="Pinot version filter")
    query.set_defaults(func=cmd_query)

    logs = sub.add_parser("logs", help="Review query traces")
    logs.add_argument("--review", action="store_true", help="Interactive review with golden set promotion")
    logs.set_defaults(func=cmd_logs)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
