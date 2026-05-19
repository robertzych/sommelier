"""Experiment: remove breadcrumbs from all chunks in Qdrant.

Fetches every point, strips the 'H1 > H2 > H3\\n\\n' prefix (derived from
h1/h2/h3 payload fields), re-embeds with dense+sparse providers, and upserts.
The LLM-generated Description/Question annotations are preserved.

Run retrieval eval after to measure breadcrumb contribution independently
of the LLM annotation.

Usage:
    uv run python -m evals.experiment_remove_breadcrumbs
"""
import hashlib
import pathlib
import sys
import types

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from embeddings.provider import get_dense_provider, get_sparse_provider
from ingestion.ingest import derive_point_id
from qdrant_client.models import PointIdsList, PointStruct
from vector_store.qdrant_store import PINOT_DOCS_COLLECTION, get_client

_BATCH_SIZE = 100


def _load_config():
    import tomllib
    with open("sommelier.toml", "rb") as f:
        raw = tomllib.load(f)

    def _to_ns(d):
        if isinstance(d, dict):
            return types.SimpleNamespace(**{k: _to_ns(v) for k, v in d.items()})
        return d

    ns = _to_ns(raw)
    ns.embeddings.model_dims = raw["embeddings"].get("model_dims", {})
    ns.collections = {k: _to_ns(v) for k, v in raw.get("collections", {}).items()}
    return ns


def _strip_breadcrumb(text: str, h1: str, h2: str, h3: str) -> str:
    """Remove the leading 'H1 > H2 > H3\\n\\n' breadcrumb if present."""
    parts = [h for h in (h1, h2, h3) if h]
    if not parts:
        return text
    breadcrumb = " > ".join(parts) + "\n\n"
    if text.startswith(breadcrumb):
        return text[len(breadcrumb):]
    return text


def main():
    config = _load_config()
    client = get_client(config)
    dense_provider = get_dense_provider(config)
    sparse_provider = get_sparse_provider()

    # Collect all points (paginate)
    all_records = []
    offset = None
    while True:
        records, offset = client.scroll(
            PINOT_DOCS_COLLECTION,
            with_payload=True,
            with_vectors=False,
            limit=1000,
            offset=offset,
        )
        all_records.extend(records)
        if offset is None:
            break

    print(f"Total points in collection: {len(all_records)}")

    # Find points whose text starts with a breadcrumb
    to_update: list[tuple] = []  # (old_id_str, new_text, payload)
    for record in all_records:
        payload = record.payload or {}
        old_text = payload.get("text", "")
        new_text = _strip_breadcrumb(
            old_text,
            payload.get("h1", ""),
            payload.get("h2", ""),
            payload.get("h3", ""),
        )
        if new_text != old_text:
            to_update.append((str(record.id), new_text, payload))

    print(f"Points with breadcrumb to strip: {len(to_update)}")
    if not to_update:
        print("Nothing to update.")
        return

    # Process in batches: delete old IDs, embed new text, insert new points
    updated = 0
    for batch_start in range(0, len(to_update), _BATCH_SIZE):
        batch = to_update[batch_start : batch_start + _BATCH_SIZE]
        old_ids = [item[0] for item in batch]
        new_texts = [item[1] for item in batch]
        payloads = [item[2] for item in batch]

        dense_vecs = dense_provider.embed(new_texts)
        sparse_vecs = sparse_provider.embed(new_texts)

        new_points = []
        for (old_id, new_text, payload), dense_vec, sparse_vec in zip(
            batch, dense_vecs, sparse_vecs
        ):
            new_payload = dict(payload)
            new_payload["text"] = new_text
            new_payload["content_hash"] = hashlib.sha256(new_text.encode()).hexdigest()
            new_points.append(
                PointStruct(
                    id=str(derive_point_id(new_text)),
                    vector={"dense": dense_vec, "sparse": sparse_vec},
                    payload=new_payload,
                )
            )

        client.delete(
            PINOT_DOCS_COLLECTION,
            points_selector=PointIdsList(points=old_ids),
        )
        client.upsert(PINOT_DOCS_COLLECTION, points=new_points)
        updated += len(batch)
        print(f"  {updated}/{len(to_update)} points updated")

    print(f"\nDone. Stripped breadcrumbs from {len(to_update)} points.")
    print("Run: uv run python -m evals.eval --golden-set evals/golden_set.json --retrieval-only")


if __name__ == "__main__":
    main()
