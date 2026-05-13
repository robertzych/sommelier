"""Eval runner: retrieval metrics against the golden set.

Run with --retrieval-only (currently the only supported mode):
    uv run python -m evals.eval --golden-set evals/golden_set.json --retrieval-only

LLM judge scoring (--judge) is a future step; see plan.md.
"""
import argparse
import json
import uuid
from typing import Any

from cli import _init_pipeline, _setup_tracer, load_config
from evals.metrics import full_recall_rate, hit_rate, mrr, recall
from observability.tracer import tracer
from retrieval.search import search
from vector_store.qdrant_store import PINOT_DOCS_COLLECTION


def load_golden_set(path: str) -> list[dict]:
    """Load and return the questions list from a golden set JSON file."""
    with open(path) as f:
        return json.load(f)["questions"]


def _aggregate_retrieval_metrics(results: list[dict]) -> dict[str, Any]:
    """Compute aggregate retrieval metrics across all per-question result dicts.

    Each dict must have: hit_rate_reranked, recall_reranked, mrr_reranked,
    reranked_fps, hit_rate_candidates, recall_candidates, expected_sources.
    """
    n = len(results)
    if n == 0:
        return {}
    return {
        "n": n,
        "hr_reranked": sum(r["hit_rate_reranked"] for r in results) / n,
        "rc_reranked": sum(r["recall_reranked"] for r in results) / n,
        "mrr_reranked": sum(r["mrr_reranked"] for r in results) / n,
        "frr_reranked": full_recall_rate(
            [
                {"retrieved": r["reranked_fps"], "expected_sources": r["expected_sources"]}
                for r in results
            ]
        ),
        "hr_candidates": sum(r["hit_rate_candidates"] for r in results) / n,
        "rc_candidates": sum(r["recall_candidates"] for r in results) / n,
    }


def _print_retrieval_summary(agg: dict, rerank_k: int, retrieval_k: int) -> None:
    """Print the two-block retrieval summary (post-reranker + candidates) to stdout."""
    n = agg["n"]
    print(f"Retrieval (post-reranker, top-{rerank_k}):")
    print(f"  Hit Rate:    {agg['hr_reranked']:.0%}   ({int(agg['hr_reranked'] * n)}/{n})")
    print(f"  Recall:      {agg['rc_reranked']:.0%}   avg")
    print(f"  MRR:         {agg['mrr_reranked']:.2f}  avg")
    print(
        f"  Full Recall: {agg['frr_reranked']:.0%}   "
        f"({int(agg['frr_reranked'] * n)}/{n} all expected sources found)"
    )
    print()
    print(f"Retrieval (candidates, top-{retrieval_k}):")
    print(f"  Hit Rate:    {agg['hr_candidates']:.0%}   ({int(agg['hr_candidates'] * n)}/{n})")
    print(f"  Recall:      {agg['rc_candidates']:.0%}   avg")


def _run_search_for_question(
    question: dict,
    client,
    dense,
    sparse,
    reranker,
    config,
) -> tuple[list[str], list[str]]:
    """Search for one golden set question and return file path lists.

    Opens a tracer span (event_type='eval_result'); caller must call tracer.flush()
    to close and export it.

    Returns: (candidate_file_paths, reranked_file_paths)
    """
    trace_id = str(uuid.uuid4())
    tracer.start_trace(
        trace_id,
        event_type="eval_result",
        question_id=question["id"],
        query=question["question"],
    )
    search(
        query=question["question"],
        client=client,
        collection_name=PINOT_DOCS_COLLECTION,
        dense_provider=dense,
        sparse_provider=sparse,
        reranker=reranker,
        config=config,
    )
    candidate_fps = [c["file_path"] for c in tracer._trace.get("candidates", [])]
    reranked_fps = [r["file_path"] for r in tracer._trace.get("reranked", [])]
    return candidate_fps, reranked_fps


def main() -> None:
    """CLI entry point for the eval runner."""
    parser = argparse.ArgumentParser(
        prog="eval",
        description="Evaluate Sommelier retrieval quality against the golden set.",
    )
    parser.add_argument("--golden-set", required=True, help="Path to golden_set.json")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Retrieval metrics only, no LLM judge (currently the only supported mode)",
    )
    parser.add_argument(
        "--config",
        default="sommelier.toml",
        help="Path to sommelier.toml (default: sommelier.toml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    _setup_tracer(config)
    client, dense, sparse, reranker = _init_pipeline(config)
    questions = load_golden_set(args.golden_set)

    print(f"Prompt version: {tracer.prompt_version}")
    print(f"Eval run: {len(questions)} questions, retrieval-only")
    print()

    results: list[dict[str, Any]] = []

    for i, q in enumerate(questions, 1):
        preview = q["question"][:60] + ("..." if len(q["question"]) > 60 else "")
        print(f"  [{i}/{len(questions)}] {q['id']}: {preview}", end="", flush=True)

        candidate_fps, reranked_fps = _run_search_for_question(
            q, client, dense, sparse, reranker, config
        )
        expected = q.get("expected_sources", [])

        result: dict[str, Any] = {
            "question_id": q["id"],
            "question": q["question"],
            "expected_sources": expected,
            "candidate_fps": candidate_fps,
            "reranked_fps": reranked_fps,
            "hit_rate_candidates": hit_rate(candidate_fps, expected),
            "recall_candidates": recall(candidate_fps, expected),
            "mrr_candidates": mrr(candidate_fps, expected),
            "hit_rate_reranked": hit_rate(reranked_fps, expected),
            "recall_reranked": recall(reranked_fps, expected),
            "mrr_reranked": mrr(reranked_fps, expected),
        }

        tracer.emit(
            "eval_metrics",
            {
                "question_id": q["id"],
                "expected_sources": expected,
                "candidate_fps": candidate_fps,
                "reranked_fps": reranked_fps,
                "metrics": {
                    k: result[k]
                    for k in result
                    if k.startswith(("hit_rate", "recall", "mrr"))
                },
            },
        )
        tracer.flush()

        results.append(result)
        print(" done")

    print()
    agg = _aggregate_retrieval_metrics(results)
    if agg:
        _print_retrieval_summary(agg, config.retrieval.rerank_top_k, config.retrieval.retrieval_top_k)


if __name__ == "__main__":
    main()
