"""Inspect the most recent trace for a given query text.

Usage:
    uv run python evals/inspect_trace.py <query> [--filter-fp <file_path>]

Examples:
    uv run python evals/inspect_trace.py "How do you set up real-time ingestion from Kafka in Pinot?"
    uv run python evals/inspect_trace.py "How do you set up real-time ingestion from Kafka in Pinot?" --filter-fp basics/getting-started/first-stream-ingest.md
"""
import json
import sys


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    query = sys.argv[1]
    filter_fp = None
    if "--filter-fp" in sys.argv:
        idx = sys.argv.index("--filter-fp")
        filter_fp = sys.argv[idx + 1]
    last = None
    with open("sommelier_traces.jsonl") as f:
        for line in f:
            try:
                t = json.loads(line)
            except json.JSONDecodeError:
                continue
            if t.get("event_type") == "query" and t.get("query") == query:
                last = t

    if not last:
        print(f"No trace found for query: {query!r}")
        sys.exit(1)

    print(f"Query: {query}")
    print(f"Trace ID: {last.get('trace_id')}")
    print(f"Timestamp: {last.get('ts')}")

    candidates = last.get("candidates", [])
    snippet_by_fp: dict[str, list[str]] = {}
    for c in candidates:
        snippet_by_fp.setdefault(c["file_path"], []).append(c.get("snippet", ""))

    if filter_fp:
        print(f"\n=== CANDIDATES matching '{filter_fp}' ===")
        for i, c in enumerate(candidates, 1):
            if c["file_path"] == filter_fp:
                snippet = c.get("snippet", "").replace("\n", " ")
                print(f"\n[candidate {i:2}] rrf_score: {c.get('rrf_score', ''):.4f}")
                print(f"    {snippet}")
        return

    print("\n=== CANDIDATES (top 20) ===")
    for i, c in enumerate(candidates, 1):
        print(f"[{i:2}] {c['file_path']}")

    print("\n=== RERANKED (with snippets) ===")
    seen: dict[str, int] = {}
    for i, r in enumerate(last.get("reranked", []), 1):
        fp = r["file_path"]
        idx = seen.get(fp, 0)
        seen[fp] = idx + 1
        snippets = snippet_by_fp.get(fp, [])
        snippet = snippets[idx] if idx < len(snippets) else "(snippet not found)"
        score = r.get("cross_encoder_score", "")
        print(f"\n[{i}] {fp}  (score: {score:.3f})")
        print(f"    {snippet[:300].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
