import hashlib
import pathlib
import re
import uuid

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointIdsList, PointStruct

from embeddings.provider import DenseEmbeddingProvider, SparseEmbeddingProvider
from observability.tracer import tracer
from vector_store.qdrant_store import PINOT_DOCS_COLLECTION

_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]
_GITBOOK_TAG = re.compile(r"\{%.*?%\}", re.DOTALL)
_EXCESS_BLANKS = re.compile(r"\n{3,}")
_TABLE_SEP_CELL = re.compile(r"^[-: ]+$")
_CODE_TOKEN_RE = re.compile(r"[^a-zA-Z0-9_.]")


def clean_gitbook(text: str) -> str:
    """Strip GitBook template tags and collapse excess blank lines."""
    text = _GITBOOK_TAG.sub("", text)
    text = _EXCESS_BLANKS.sub("\n\n", text)
    return text.strip()


def _parse_row(line: str) -> list[str]:
    """Split a markdown table row into stripped cell strings."""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def normalize_tables(text: str) -> str:
    """Convert config-reference markdown tables to prose for better retrieval.

    Tables whose second column header contains 'default' (case-insensitive) are
    converted from row-per-property format to one prose sentence per property:
    '<property>: default <value>. <description>'

    All other tables are left unchanged.
    """
    lines = text.splitlines(keepends=True)
    result: list[str] = []
    i = 0
    while i < len(lines):
        if not lines[i].rstrip("\n").startswith("|"):
            result.append(lines[i])
            i += 1
            continue

        # Collect contiguous table lines
        table_lines: list[str] = []
        while i < len(lines) and lines[i].rstrip("\n").startswith("|"):
            table_lines.append(lines[i].rstrip("\n"))
            i += 1

        # Need at least header + separator + one data row
        if len(table_lines) < 3:
            result.extend(l + "\n" for l in table_lines)
            continue

        header_cells = _parse_row(table_lines[0])
        sep_cells = _parse_row(table_lines[1])

        is_config_table = (
            len(header_cells) >= 2
            and bool(sep_cells)
            and all(_TABLE_SEP_CELL.match(c) for c in sep_cells)
            and "default" in header_cells[1].lower()
        )

        if not is_config_table:
            result.extend(l + "\n" for l in table_lines)
            continue

        for data_line in table_lines[2:]:
            cells = _parse_row(data_line)
            while len(cells) < 3:
                cells.append("")
            prop = cells[0]
            default_val = cells[1] if cells[1] else "no default"
            description = cells[2]
            sentence = f"{prop}: default {default_val}."
            if description:
                sentence = f"{sentence} {description}"
            result.append(sentence + "\n")

    return "".join(result)


def derive_point_id(chunk_text: str, file_path: str) -> uuid.UUID:
    """Return a deterministic UUID derived from the SHA-256 hash of file_path + chunk_text.

    Including file_path prevents ID collisions when two files contain identical chunk text.
    """
    digest = hashlib.sha256((file_path + "\n" + chunk_text).encode()).digest()[:16]
    return uuid.UUID(bytes=digest)


def _is_code_heavy(text: str, threshold: float = 0.5) -> bool:
    """Return True if the majority of text falls within fenced code blocks.

    Counts characters on lines inside ``` fences and divides by total length.
    Used to detect example/config sections that are almost entirely code and
    would score poorly in retrieval without surrounding prose context.
    """
    in_fence = False
    code_chars = 0
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            code_chars += len(line) + 1
    total = len(text)
    return total > 0 and code_chars / total >= threshold


def _code_tokens(text: str) -> set[str]:
    """Extract technical identifiers for overlap scoring.

    Restricts to camelCase, underscore, or dotted tokens (config keys, class names,
    property paths). Plain column values like 'Country' or 'Browser' are excluded so
    that shared data-domain vocabulary does not outscore config-key matches when
    choosing which prose chunk to merge a code block into.
    """
    return {
        t for t in _CODE_TOKEN_RE.split(text)
        if len(t) > 3 and ("_" in t or "." in t or any(c.isupper() for c in t[1:]))
    }


def _merge_code_chunks(docs: list) -> list:
    """Merge each code-heavy chunk into the prose chunk with the highest keyword overlap.

    Searches the full document — not just neighbors — to find the best prose target
    for each code-heavy chunk. This handles examples that appear multiple header
    sections away from the prose that introduces them (e.g., a standalone Example
    section whose JSON uses identifiers defined in a distant configuration section).
    Code content is appended to the target prose chunk in document order; prose chunk
    order is preserved. If all chunks are code-heavy, they are returned unchanged.
    """
    if not docs:
        return docs

    # Identify which chunks are code-heavy and which are prose
    code_set = {i for i, doc in enumerate(docs) if _is_code_heavy(doc.page_content)}
    if not code_set:
        return docs

    prose_indices = [i for i in range(len(docs)) if i not in code_set]
    if not prose_indices:
        return docs

    # Pre-compute token sets for all prose chunks before any mutation
    prose_tokens = {i: _code_tokens(docs[i].page_content) for i in prose_indices}

    # Assign each code chunk to the prose chunk with the most shared technical tokens.
    # `code_toks & prose_tokens[pi]` is the set intersection — technical identifiers
    # (camelCase, underscore, dotted) that appear in both chunks. `len(...)` converts
    # that to a count; max() selects the prose chunk with the largest overlap count.
    appended: dict[int, list[str]] = {}
    for ci in sorted(code_set):
        code_toks = _code_tokens(docs[ci].page_content)
        target = max(prose_indices, key=lambda pi: len(code_toks & prose_tokens[pi]))
        appended.setdefault(target, []).append(docs[ci].page_content)

    # Rebuild the chunk list: prose only, each with its assigned code appended
    result = []
    for i, doc in enumerate(docs):
        if i in code_set:
            continue
        if i in appended:
            doc.page_content = doc.page_content.rstrip() + "\n\n" + "\n\n".join(
                c.rstrip() for c in appended[i]
            )
        result.append(doc)

    return result


def _build_breadcrumb(metadata: dict) -> str:
    """Return a section breadcrumb string from header metadata, e.g. 'H1 > H2 > H3'."""
    parts = [metadata[k] for k in ("h1", "h2", "h3") if metadata.get(k)]
    return " > ".join(parts)


_ANNOTATION_PROMPT = """\
You are annotating a technical documentation chunk for search retrieval.

The chunk below is from Apache Pinot documentation and contains a code example.
Generate one sentence describing what this code demonstrates and one natural-language \
question that this code directly answers.

Chunk:
{chunk_text}

Respond with exactly two lines and no other text:
Description: <one sentence>
Question: <one question>"""


class LLMCodeAnnotator:
    """Generates a Description+Question annotation for code-heavy chunks via LLM.

    The annotation is inserted between the breadcrumb and the chunk body so the
    cross-encoder reranker immediately sees natural-language signal about the code's
    purpose, rather than having to infer it from dense table rows or code syntax.
    """

    def __init__(self, model: str, api_key: str = "") -> None:
        self._model = model
        self._api_key = api_key

    def annotate(self, chunk_text: str) -> str:
        """Return 'Description: ...\\nQuestion: ...' or empty string on failure."""
        import litellm

        prompt = _ANNOTATION_PROMPT.format(chunk_text=chunk_text[:3000])
        kwargs: dict = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 120,
        }
        if self._api_key:
            kwargs["api_key"] = self._api_key
        try:
            response = litellm.completion(**kwargs)
            text = response.choices[0].message.content.strip()
            if "Description:" in text and "Question:" in text:
                return text
        except Exception:
            pass
        return ""


def _insert_annotation(text: str, annotation: str) -> str:
    """Insert annotation between the breadcrumb line and the rest of the chunk body."""
    parts = text.split("\n\n", 1)
    if len(parts) == 2:
        return parts[0] + "\n\n" + annotation + "\n\n" + parts[1]
    return annotation + "\n\n" + text


def _chunk_markdown(raw_markdown: str):
    """Clean and split a markdown document into header-aware, size-bounded chunks."""
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=_HEADERS)
    cleaned = normalize_tables(clean_gitbook(raw_markdown))
    header_splits = md_splitter.split_text(cleaned)
    char_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512, chunk_overlap=50
    )
    char_splits = char_splitter.split_documents(header_splits)

    # Prepend section breadcrumb to each chunk so BM25 and dense embeddings see
    # document context that is otherwise only stored in metadata fields.
    for doc in char_splits:
        breadcrumb = _build_breadcrumb(doc.metadata)
        if breadcrumb:
            doc.page_content = breadcrumb + "\n\n" + doc.page_content

    return char_splits


def index_file(
    client: QdrantClient,
    collection_name: str,
    file_path: str,
    raw_markdown: str,
    dense_provider: DenseEmbeddingProvider,
    sparse_provider: SparseEmbeddingProvider,
    pinot_version: str,
    annotator: "LLMCodeAnnotator | None" = None,
) -> dict:
    """Incrementally index one markdown file into Qdrant.

    Chunks the document, diffs against existing points by content-hash ID,
    deletes stale points, and inserts only new or changed chunks.

    Returns a dict with keys: inserted, deleted, skipped, errors.
    """
    # Point IDs are derived from pre-annotation text so that unchanged chunks
    # retain the same ID across runs regardless of LLM annotation variability.
    chunks = _chunk_markdown(raw_markdown)
    chunk_texts = [c.page_content for c in chunks]

    new_points = {derive_point_id(t, file_path): (t, c) for t, c in zip(chunk_texts, chunks)}

    # collect existing point IDs for this file (paginate to handle large files)
    old_ids: set[uuid.UUID] = set()
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="file_path", match=MatchValue(value=file_path))]
            ),
            with_payload=False,
            limit=1000,
            offset=offset,
        )
        for r in records:
            old_ids.add(uuid.UUID(str(r.id)))
        if offset is None:
            break

    # delete stale points (removed or modified chunks)
    stale_ids = old_ids - set(new_points)
    if stale_ids:
        client.delete(
            collection_name,
            points_selector=PointIdsList(points=[str(sid) for sid in stale_ids]),
        )

    # embed and insert only chunks not already in the collection
    to_insert = {pid: val for pid, val in new_points.items() if pid not in old_ids}

    # annotate new code-heavy chunks only — chunks already in Qdrant are skipped
    # entirely so the LLM is never called for unchanged content
    if annotator is not None:
        for pid in list(to_insert):
            raw_text, doc = to_insert[pid]
            if _is_code_heavy(raw_text):
                annotation = annotator.annotate(raw_text)
                if annotation:
                    annotated_text = _insert_annotation(raw_text, annotation)
                    to_insert[pid] = (annotated_text, doc)

    if to_insert:
        pids = list(to_insert)
        texts = [to_insert[pid][0] for pid in pids]
        docs = [to_insert[pid][1] for pid in pids]
        dense_vecs = dense_provider.embed(texts)
        sparse_vecs = sparse_provider.embed(texts)

        points = [
            PointStruct(
                id=str(pid),
                vector={"dense": dense_vec, "sparse": sparse_vec},
                payload={
                    "text": text,
                    "file_path": file_path,
                    "doc_type": "documentation",
                    "pinot_version": pinot_version,
                    "h1": doc.metadata.get("h1", ""),
                    "h2": doc.metadata.get("h2", ""),
                    "h3": doc.metadata.get("h3", ""),
                    "content_hash": hashlib.sha256(text.encode()).hexdigest(),
                },
            )
            for pid, text, doc, dense_vec, sparse_vec in zip(
                pids, texts, docs, dense_vecs, sparse_vecs
            )
        ]
        client.upsert(collection_name, points=points)

    return {
        "inserted": len(to_insert),
        "deleted": len(stale_ids),
        "skipped": len(new_points) - len(to_insert),
        "errors": [],
    }


def ingest(
    docs_path: str,
    config,
    pinot_version: str,
    collection_name: str = PINOT_DOCS_COLLECTION,
) -> None:
    """Walk docs_path recursively, indexing every .md file into Qdrant.

    docs_path is the root of a cloned pinot-docs repo. The file_path stored
    in each chunk's payload is relative to that root (e.g. basics/concepts/table.md).

    Emits one ingestion trace per file via the tracer singleton.
    """
    from embeddings.provider import get_dense_provider, get_sparse_provider
    from vector_store.qdrant_store import ensure_collection, get_client

    client = get_client(config)
    ensure_collection(client, config, collection_name)
    dense_provider = get_dense_provider(config)
    sparse_provider = get_sparse_provider()

    inf = getattr(config, "inference", None)
    annotator = (
        LLMCodeAnnotator(
            model=inf.model,
            api_key=getattr(inf, "api_key", ""),
        )
        if inf is not None
        else None
    )

    from tqdm import tqdm

    docs_root = pathlib.Path(docs_path).resolve()
    md_files = sorted(docs_root.rglob("*.md"))

    totals = {"inserted": 0, "deleted": 0, "skipped": 0, "errors": 0}
    for md_file in tqdm(md_files, desc="Indexing", unit="file"):
        file_path = str(md_file.relative_to(docs_root))
        raw_markdown = md_file.read_text(encoding="utf-8")

        trace_id = str(uuid.uuid4())
        tracer.start_trace(
            trace_id,
            event_type="ingestion",
            file_path=file_path,
            pinot_version=pinot_version,
        )
        try:
            stats = index_file(
                client=client,
                collection_name=collection_name,
                file_path=file_path,
                raw_markdown=raw_markdown,
                dense_provider=dense_provider,
                sparse_provider=sparse_provider,
                pinot_version=pinot_version,
                annotator=annotator,
            )
            tracer.emit("ingestion", stats)
            for k in ("inserted", "deleted", "skipped"):
                totals[k] += stats[k]
            totals["errors"] += len(stats["errors"])
        except Exception as e:
            tracer.emit(
                "ingestion",
                {"inserted": 0, "deleted": 0, "skipped": 0, "errors": [str(e)]},
            )
            totals["errors"] += 1
        finally:
            tracer.flush()

    print(
        f"Done. {len(md_files)} files — "
        f"{totals['inserted']} inserted, "
        f"{totals['deleted']} deleted, "
        f"{totals['skipped']} skipped, "
        f"{totals['errors']} errors"
    )
