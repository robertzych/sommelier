"""Inspect Qdrant chunks for a given file_path.

Usage:
    uv run python evals/inspect_chunks.py <file_path>

Example:
    uv run python evals/inspect_chunks.py build-with-pinot/ingestion/stream-ingestion/README.md
"""
import sys

from cli import load_config
from qdrant_client.models import FieldCondition, Filter, MatchValue
from vector_store.qdrant_store import PINOT_DOCS_COLLECTION, get_client


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    file_path = sys.argv[1]
    config = load_config("sommelier.toml")
    client = get_client(config)

    results, _ = client.scroll(
        PINOT_DOCS_COLLECTION,
        scroll_filter=Filter(must=[FieldCondition(key="file_path", match=MatchValue(value=file_path))]),
        with_payload=True,
        limit=100,
    )

    print(f"file_path: {file_path}")
    print(f"Chunks found: {len(results)}")

    for i, point in enumerate(results, 1):
        p = point.payload
        breadcrumb = " > ".join(v for v in [p.get("h1"), p.get("h2"), p.get("h3")] if v)
        # text = p.get("text", "")[:400].replace("\n", " ")
        text = p.get("text", "").replace("\n", " ")
        print(f"\n[{i}] {breadcrumb}")
        print(f"    {text}")


if __name__ == "__main__":
    main()
