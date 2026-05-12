import argparse
import sys
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


def cmd_chat(args):
    """Interactive multi-turn REPL with conversation memory."""
    from embeddings.provider import get_dense_provider, get_sparse_provider
    from inference.llm import complete
    from retrieval.search import get_reranker, search
    from vector_store.qdrant_store import PINOT_DOCS_COLLECTION, ensure_collection, get_client

    config = load_config()
    pinot_version = None if args.version == "latest" else args.version
    memory_turns = config.inference.memory_turns

    client = get_client(config)
    ensure_collection(client, config, PINOT_DOCS_COLLECTION)
    dense = get_dense_provider(config)
    sparse = get_sparse_provider()
    reranker = get_reranker(config)

    history: list[dict] = []
    turn = 0

    print("Sommelier — Apache Pinot assistant. Type 'exit' or Ctrl-D to quit.\n")

    while True:
        try:
            query = input("You: ").strip()
            if not sys.stdin.isatty():
                print(query)
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            break

        turn += 1
        tracer.start_trace(
            str(uuid.uuid4()),
            event_type="chat_turn",
            query=query,
            turn=turn,
            pinot_version=args.version,
        )

        chunks = search(
            query=query,
            client=client,
            collection_name=PINOT_DOCS_COLLECTION,
            dense_provider=dense,
            sparse_provider=sparse,
            reranker=reranker,
            config=config,
            pinot_version=pinot_version,
        )

        print("Sommelier: ", end="", flush=True)
        response_parts: list[str] = []
        for token in complete(query, chunks, history, config, pinot_version=pinot_version):
            print(token, end="", flush=True)
            response_parts.append(token)
        print("\n")

        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": "".join(response_parts)})
        history = history[-(memory_turns * 2):]

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

    chat = sub.add_parser("chat", help="Interactive multi-turn REPL with conversation memory")
    chat.add_argument("--version", default="latest", help="Pinot version filter")
    chat.set_defaults(func=cmd_chat)

    logs = sub.add_parser("logs", help="Review query traces")
    logs.add_argument("--review", action="store_true", help="Interactive review with golden set promotion")
    logs.set_defaults(func=cmd_logs)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
