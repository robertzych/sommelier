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


def _setup_tracer(config) -> None:
    """Register configured exporters on the tracer singleton."""
    from observability.exporters import get_exporters

    for exporter in get_exporters(config):
        tracer.register_exporter(exporter)


def _init_pipeline(config):
    """Initialize and return (client, dense_provider, sparse_provider, reranker)."""
    from embeddings.provider import get_dense_provider, get_sparse_provider
    from retrieval.search import get_reranker
    from vector_store.qdrant_store import PINOT_DOCS_COLLECTION, ensure_collection, get_client

    client = get_client(config)
    ensure_collection(client, config, PINOT_DOCS_COLLECTION)
    return (
        client,
        get_dense_provider(config),
        get_sparse_provider(),
        get_reranker(config),
    )


def cmd_ingest(args):
    """Index all .md files in docs_path into Qdrant."""
    from ingestion.ingest import ingest

    config = load_config()
    _setup_tracer(config)
    ingest(docs_path=args.docs_path, config=config, pinot_version=args.version)


def _load_followup_history(followup_file: str, followup_id: str) -> tuple[str, list[dict]]:
    """Return (question, history) for a follow-up entry in golden_set_followup.json.

    history is a two-message list: the parent question as the user turn and the
    parent sommelier answer as the assistant turn — matching cmd_chat's history format.
    """
    import json

    with open(followup_file) as f:
        data = json.load(f)

    entry = next((q for q in data["questions"] if q["id"] == followup_id), None)
    if entry is None:
        raise SystemExit(f"followup-id '{followup_id}' not found in {followup_file}")

    parent = entry.get("parent", {})
    history = [
        {"role": "user", "content": parent["question"]},
        {"role": "assistant", "content": parent["sommelier_answer"]},
    ]
    return entry["question"], history


def cmd_query(args):
    """Run a single query through the full search → rerank → llm pipeline."""
    from inference.llm import complete
    from retrieval.search import search
    from vector_store.qdrant_store import PINOT_DOCS_COLLECTION

    config = load_config()
    _setup_tracer(config)
    pinot_version = None if args.version == "latest" else args.version
    client, dense, sparse, reranker = _init_pipeline(config)

    history: list[dict] = []
    if args.followup_id:
        followup_file = args.followup_file or "evals/golden_set_followup.json"
        loaded_question, history = _load_followup_history(followup_file, args.followup_id)
        question = args.question or loaded_question
    else:
        if not args.question:
            raise SystemExit("error: question is required when --followup-id is not provided")
        question = args.question

    tracer.start_trace(
        str(uuid.uuid4()),
        event_type="query",
        query=question,
        pinot_version=args.version,
    )

    chunks = search(
        query=question,
        client=client,
        collection_name=PINOT_DOCS_COLLECTION,
        dense_provider=dense,
        sparse_provider=sparse,
        reranker=reranker,
        config=config,
        pinot_version=pinot_version,
    )

    for token in complete(question, chunks, history, config, pinot_version=pinot_version):
        print(token, end="", flush=True)
    print()

    tracer.flush()


def cmd_chat(args):
    """Interactive multi-turn REPL with conversation memory."""
    from inference.llm import complete
    from retrieval.search import search
    from vector_store.qdrant_store import PINOT_DOCS_COLLECTION

    config = load_config()
    _setup_tracer(config)
    pinot_version = None if args.version == "latest" else args.version
    memory_turns = config.inference.memory_turns
    client, dense, sparse, reranker = _init_pipeline(config)

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
    query.add_argument("question", nargs="?", help="Question to ask Sommelier (omit when using --followup-id)")
    query.add_argument("--version", default="latest", help="Pinot version filter")
    query.add_argument("--followup-id", default=None, help="ID of a follow-up entry in the followup golden set (e.g. q001_t2); seeds conversation history from the parent Q&A")
    query.add_argument("--followup-file", default=None, help="Path to followup golden set JSON (default: evals/golden_set_followup.json)")
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
