"""Experiment: annotate the star-tree Example code chunk with a description and question.

Finds the chunk in Qdrant that contains the star-tree tableIndexConfig example,
inserts 'Description: ... Question: ...' between its breadcrumb and body, re-embeds,
and upserts. Run the retrieval eval after this to check whether q003 improves.

Usage:
    uv run python -m evals.experiment_annotate_code_chunk
"""
import hashlib
import pathlib
import sys
import uuid

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import types

from embeddings.provider import get_dense_provider, get_sparse_provider
from ingestion.ingest import _chunk_markdown, derive_point_id
from qdrant_client.models import PointIdsList, PointStruct
from vector_store.qdrant_store import PINOT_DOCS_COLLECTION, get_client

_DOCS_ROOT = pathlib.Path("/Users/robertzych/dev/pinot-docs")
_TARGET_FILE = "build-with-pinot/indexing/star-tree-index.md"
_ANNOTATION = (
    "Description: Shows how to configure a star-tree index. "
    "Question: How do you configure a star-tree index in Pinot?"
)
# Unique signature that appears in the target code block
_CODE_SIGNATURE = '"starTreeIndexConfigs"'


def _load_config():
    import tomllib
    with open("sommelier.toml", "rb") as f:
        raw = tomllib.load(f)

    def _to_ns(d):
        if isinstance(d, dict):
            return types.SimpleNamespace(**{k: _to_ns(v) for k, v in d.items()})
        return d

    ns = _to_ns(raw)
    # Keep model_dims and collections as dicts for lookup
    ns.embeddings.model_dims = raw["embeddings"].get("model_dims", {})
    ns.collections = {k: _to_ns(v) for k, v in raw.get("collections", {}).items()}
    return ns


def _annotate_text(text: str) -> str:
    """Insert annotation between breadcrumb and body (split on first blank line)."""
    parts = text.split("\n\n", 1)
    if len(parts) == 2:
        return parts[0] + "\n\n" + _ANNOTATION + "\n\n" + parts[1]
    return _ANNOTATION + "\n\n" + text


def main():
    config = _load_config()
    client = get_client(config)
    dense_provider = get_dense_provider(config)
    sparse_provider = get_sparse_provider()

    raw_markdown = (_DOCS_ROOT / _TARGET_FILE).read_text(encoding="utf-8")
    chunks = _chunk_markdown(raw_markdown)

    target_chunks = [
        c for c in chunks
        if _CODE_SIGNATURE in c.page_content
    ]

    if not target_chunks:
        print(f"No chunks found containing {_CODE_SIGNATURE!r}.")
        return

    print(f"Found {len(target_chunks)} chunk(s) containing the target code signature.")

    for chunk in target_chunks:
        old_text = chunk.page_content
        old_id = str(derive_point_id(old_text))
        new_text = _annotate_text(old_text)
        new_id = str(derive_point_id(new_text))

        print(f"\nOld ID: {old_id}")
        print(f"New ID: {new_id}")
        print(f"Old text ({len(old_text)} chars), first 300 chars:")
        print(old_text[:300])
        print("---")
        print(f"New text ({len(new_text)} chars), first 400 chars:")
        print(new_text[:400])
        print("---")

        # Delete old point
        client.delete(
            PINOT_DOCS_COLLECTION,
            points_selector=PointIdsList(points=[old_id]),
        )
        print(f"Deleted old point {old_id}")

        # Re-embed and insert annotated chunk
        dense_vecs = dense_provider.embed([new_text])
        sparse_vecs = sparse_provider.embed([new_text])

        point = PointStruct(
            id=new_id,
            vector={"dense": dense_vecs[0], "sparse": sparse_vecs[0]},
            payload={
                "text": new_text,
                "file_path": _TARGET_FILE,
                "doc_type": "documentation",
                "pinot_version": "latest",
                "h1": chunk.metadata.get("h1", ""),
                "h2": chunk.metadata.get("h2", ""),
                "h3": chunk.metadata.get("h3", ""),
                "content_hash": hashlib.sha256(new_text.encode()).hexdigest(),
            },
        )
        client.upsert(PINOT_DOCS_COLLECTION, points=[point])
        print(f"Inserted annotated point {new_id}")

    print("\nDone. Run the retrieval eval to check q003.")


if __name__ == "__main__":
    main()
