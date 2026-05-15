"""Eval report generator: quality tables + retrieval metrics + recommendations.

Reads evals/golden_set.json (manual quality scores) and sommelier_traces.jsonl
(retrieval results captured when sommelier answers were collected), then writes
evals/results.md.

Run:
    uv run python -m evals.report --golden-set evals/golden_set.json
"""
import argparse
import json
import pathlib
from datetime import datetime
from zoneinfo import ZoneInfo

from evals.eval import aggregate_retrieval_metrics, load_golden_set
from evals.metrics import hit_rate, mrr, recall

_PT = ZoneInfo("America/Los_Angeles")
_BASELINES = [
    "docs_pinot_ai",
    "claude (claude-sonnet-4-6)",
    "sommelier (claude-haiku-4-5)",
]
_SOMMELIER = "sommelier (claude-haiku-4-5)"
_CLAUDE = "claude (claude-sonnet-4-6)"
_DOCS = "docs_pinot_ai"


# ── Trace parsing ─────────────────────────────────────────────────────────────


def _parse_traces(jsonl_path: str) -> dict[str, dict]:
    """Return {query_text: trace_dict} — last occurrence per query for event_type='query'."""
    traces: dict[str, dict] = {}
    try:
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    trace = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if trace.get("event_type") != "query":
                    continue
                query = trace.get("query", "")
                if query:
                    traces[query] = trace
    except FileNotFoundError:
        pass
    return traces


# ── Per-question retrieval ────────────────────────────────────────────────────


def _compute_retrieval_per_question(
    questions: list[dict], traces: dict[str, dict]
) -> list[dict]:
    """Match each question to its trace and compute retrieval metrics.

    If no trace is found, all metrics are 0.0 and trace_missing=True.
    """
    results = []
    for q in questions:
        trace = traces.get(q["question"])
        expected = q.get("expected_sources", [])
        if trace is None:
            results.append(
                {
                    "question_id": q["id"],
                    "question": q["question"],
                    "expected_sources": expected,
                    "reranked_fps": [],
                    "candidate_fps": [],
                    "hit_rate_reranked": 0.0,
                    "recall_reranked": 0.0,
                    "mrr_reranked": 0.0,
                    "hit_rate_candidates": 0.0,
                    "recall_candidates": 0.0,
                    "mrr_candidates": 0.0,
                    "trace_missing": True,
                }
            )
        else:
            reranked_fps = [r["file_path"] for r in trace.get("reranked", [])]
            candidate_fps = [c["file_path"] for c in trace.get("candidates", [])]
            results.append(
                {
                    "question_id": q["id"],
                    "question": q["question"],
                    "expected_sources": expected,
                    "reranked_fps": reranked_fps,
                    "candidate_fps": candidate_fps,
                    "hit_rate_reranked": hit_rate(reranked_fps, expected),
                    "recall_reranked": recall(reranked_fps, expected),
                    "mrr_reranked": mrr(reranked_fps, expected),
                    "hit_rate_candidates": hit_rate(candidate_fps, expected),
                    "recall_candidates": recall(candidate_fps, expected),
                    "mrr_candidates": mrr(candidate_fps, expected),
                    "trace_missing": False,
                }
            )
    return results


# ── Quality aggregation ───────────────────────────────────────────────────────


def _aggregate_quality_by_system(questions: list[dict]) -> dict[str, dict]:
    """Return {system: {accuracy, completeness, citations, total}} averaged across all questions."""
    acc: dict[str, dict[str, list]] = {s: {"accuracy": [], "completeness": [], "citations": [], "total": []} for s in _BASELINES}
    for q in questions:
        for system in _BASELINES:
            scores = q["baseline_responses"][system]["scores"]
            for dim in ("accuracy", "completeness", "citations", "total"):
                if scores[dim] is not None:
                    acc[system][dim].append(scores[dim])
    return {
        system: {dim: sum(vals) / len(vals) if vals else 0.0 for dim, vals in dims.items()}
        for system, dims in acc.items()
    }


def _aggregate_quality_by_query_type(
    questions: list[dict],
) -> dict[str, dict[str, dict]]:
    """Return {query_type: {system: {accuracy, completeness, citations, total, n}}}."""
    raw: dict[str, dict[str, dict[str, list]]] = {}
    for q in questions:
        qt = q["query_type"]
        if qt not in raw:
            raw[qt] = {s: {"accuracy": [], "completeness": [], "citations": [], "total": []} for s in _BASELINES}
        for system in _BASELINES:
            scores = q["baseline_responses"][system]["scores"]
            for dim in ("accuracy", "completeness", "citations", "total"):
                if scores[dim] is not None:
                    raw[qt][system][dim].append(scores[dim])
    result: dict[str, dict[str, dict]] = {}
    for qt, systems in raw.items():
        result[qt] = {}
        for system, dims in systems.items():
            n = len(dims["total"])
            result[qt][system] = {dim: sum(vals) / len(vals) if vals else 0.0 for dim, vals in dims.items()}
            result[qt][system]["n"] = n
    return result


# ── Retrieval aggregation ─────────────────────────────────────────────────────


def _aggregate_retrieval_by_query_type(
    questions: list[dict], retrieval_results: list[dict]
) -> dict[str, dict]:
    """Return {query_type: {hr_reranked, rc_reranked, mrr_reranked, mrr_candidates, n}}."""
    by_qt: dict[str, list] = {}
    for q, r in zip(questions, retrieval_results):
        qt = q["query_type"]
        by_qt.setdefault(qt, []).append(r)
    result = {}
    for qt, rs in by_qt.items():
        n = len(rs)
        result[qt] = {
            "hr_reranked": sum(r["hit_rate_reranked"] for r in rs) / n,
            "rc_reranked": sum(r["recall_reranked"] for r in rs) / n,
            "mrr_reranked": sum(r["mrr_reranked"] for r in rs) / n,
            "n": n,
        }
    return result


# ── V1 gates ──────────────────────────────────────────────────────────────────


def _check_v1_gates(quality_by_system: dict[str, dict]) -> list[tuple[bool, str]]:
    """Return list of (passed, description) tuples for each V1 success criterion."""
    som = quality_by_system[_SOMMELIER]
    cld = quality_by_system[_CLAUDE]
    gates = []

    passed = som["total"] >= 2.5
    gates.append((passed, f"Sommelier avg total {som['total']:.2f}/3 {'≥' if passed else '<'} 2.5"))

    for dim in ("accuracy", "completeness", "citations"):
        passed = som[dim] > cld[dim]
        gates.append(
            (passed, f"Sommelier {dim} {som[dim]:.2f} {'>' if passed else '≤'} claude {cld[dim]:.2f}")
        )
    return gates


# ── Recommendations ───────────────────────────────────────────────────────────


def _generate_recommendations(
    questions: list[dict],
    retrieval_results: list[dict],
    quality_by_system: dict[str, dict],
    quality_by_qt: dict[str, dict[str, dict]],
) -> list[str]:
    """Return list of markdown bullet strings based on rule-based analysis."""
    recs: list[str] = []

    # Rule 1: retrieval miss where sommelier scores below baseline average
    for q, r in zip(questions, retrieval_results):
        if r["hit_rate_reranked"] == 0.0:
            s = q["baseline_responses"][_SOMMELIER]["scores"]["total"]
            d = q["baseline_responses"][_DOCS]["scores"]["total"]
            c = q["baseline_responses"][_CLAUDE]["scores"]["total"]
            if s is not None and d is not None and c is not None:
                baseline_avg = (d + c) / 2
                if s < baseline_avg:
                    srcs = ", ".join(f"`{src}`" for src in q["expected_sources"])
                    recs.append(
                        f"**{q['id']}** retrieval miss — sommelier scored {s}/3 vs baseline avg "
                        f"{baseline_avg:.1f}. Verify {srcs} is indexed."
                    )

    # Rule 2: query type where docs_pinot_ai outperforms sommelier by > 0.5 on total
    for qt, systems in quality_by_qt.items():
        dp = systems[_DOCS]["total"]
        som = systems[_SOMMELIER]["total"]
        if dp - som > 0.5:
            recs.append(
                f"**{qt}**: sommelier avg total {som:.1f} vs docs_pinot_ai {dp:.1f} — "
                f"investigate retrieval coverage or system prompt for this query type."
            )

    # Rule 3: expected sources never retrieved across ≥ 2 questions
    miss_counts: dict[str, int] = {}
    for q, r in zip(questions, retrieval_results):
        for src in q["expected_sources"]:
            if src not in r["reranked_fps"]:
                miss_counts[src] = miss_counts.get(src, 0) + 1
    for src, count in sorted(miss_counts.items(), key=lambda x: -x[1]):
        if count >= 2:
            recs.append(
                f"**`{src}`** not retrieved for {count} questions — "
                f"check ingestion or chunk granularity."
            )

    # Rule 4: sommelier dimension avg < 0.5
    som_agg = quality_by_system[_SOMMELIER]
    for dim in ("accuracy", "completeness", "citations"):
        if som_agg[dim] < 0.5:
            recs.append(
                f"**sommelier {dim}** avg {som_agg[dim]:.2f} — "
                f"consider system prompt tuning for {dim}."
            )

    # Rule 5: V1 gate failures
    if som_agg["total"] < 2.5:
        recs.append(
            f"**V1 gate**: sommelier avg total {som_agg['total']:.2f} < 2.5 — "
            f"focus improvement on query types with lowest scores."
        )
    cld_agg = quality_by_system[_CLAUDE]
    for dim in ("accuracy", "completeness", "citations"):
        if som_agg[dim] <= cld_agg[dim]:
            suggestion = {
                "accuracy": "improving retrieval recall so sommelier sees the right context",
                "completeness": "expanding system prompt to encourage fuller answers",
                "citations": "strengthening citation instructions in the system prompt",
            }[dim]
            recs.append(
                f"**V1 gate**: sommelier {dim} ({som_agg[dim]:.2f}) does not beat claude "
                f"({cld_agg[dim]:.2f}) — consider {suggestion}."
            )

    if not recs:
        recs.append("No significant issues detected. All V1 gates passed and retrieval looks healthy.")

    return recs


# ── Markdown rendering ────────────────────────────────────────────────────────


def _render_quality_table(questions: list[dict]) -> str:
    """Return a markdown table with one row per (question × system), 75 rows total."""
    lines = [
        "| Query | System | Accuracy | Completeness | Citations | Total |",
        "|-------|--------|----------|--------------|-----------|-------|",
    ]
    for q in questions:
        preview = q["question"][:50] + ("…" if len(q["question"]) > 50 else "")
        for i, system in enumerate(_BASELINES):
            scores = q["baseline_responses"][system]["scores"]
            cell_query = preview if i == 0 else ""
            lines.append(
                f"| {cell_query} | {system} | {scores['accuracy']} | "
                f"{scores['completeness']} | {scores['citations']} | {scores['total']} |"
            )
    return "\n".join(lines)


def _render_retrieval_table(questions: list[dict], retrieval_results: list[dict]) -> str:
    """Return a markdown table with one row per question, 25 rows total."""
    lines = [
        "| ID | Query | HR@5 | RC@5 | MRR@5 | Miss? | Expected Sources |",
        "|----|-------|------|------|-------|-------|------------------|",
    ]
    for q, r in zip(questions, retrieval_results):
        preview = q["question"][:50] + ("…" if len(q["question"]) > 50 else "")
        miss = "yes" if r["hit_rate_reranked"] == 0.0 else "no"
        flag = " ⚠️" if miss == "yes" else ""
        sources = ", ".join(f"`{s}`" for s in q["expected_sources"])
        lines.append(
            f"| {q['id']} | {preview} | {r['hit_rate_reranked']:.0%} | "
            f"{r['recall_reranked']:.0%} | {r['mrr_reranked']:.2f} | {miss}{flag} | {sources} |"
        )
    return "\n".join(lines)


def _render_markdown(
    questions: list[dict],
    retrieval_results: list[dict],
    quality_by_system: dict[str, dict],
    quality_by_qt: dict[str, dict[str, dict]],
    retrieval_agg: dict,
    retrieval_by_qt: dict[str, dict],
    v1_gates: list[tuple[bool, str]],
    recommendations: list[str],
    golden_set_path: str,
    traces_path: str,
    prompt_version: str,
) -> str:
    now = datetime.now(_PT).strftime("%Y-%m-%d %H:%M %Z")
    n = len(questions)

    # ── Section 1: Header ────────────────────────────────────────────────────
    header = (
        f"# Sommelier Eval Report\n\n"
        f"**Generated:** {now}  \n"
        f"**Prompt version:** {prompt_version}  \n"
        f"**Golden set:** {golden_set_path} ({n} questions)  \n"
        f"**Traces:** {traces_path}  "
    )

    # ── Section 2: V1 Success Criteria ───────────────────────────────────────
    gate_rows = "\n".join(
        f"| {'✅' if p else '❌'} | {msg} |" for p, msg in v1_gates
    )
    ra = retrieval_agg
    v1_section = (
        "## V1 Success Criteria\n\n"
        "| | Gate |\n"
        "|-|------|\n"
        f"{gate_rows}\n\n"
        "**Retrieval (observed — calibrating targets):**  \n"
        f"HR@5: {ra.get('hr_reranked', 0):.0%} &nbsp;|&nbsp; "
        f"RC@5: {ra.get('rc_reranked', 0):.0%} &nbsp;|&nbsp; "
        f"MRR@5: {ra.get('mrr_reranked', 0):.2f} &nbsp;|&nbsp; "
        f"FRR@5: {ra.get('frr_reranked', 0):.0%}"
    )

    # ── Section 3: Aggregate Quality ─────────────────────────────────────────
    overall_rows = "\n".join(
        f"| {s} | {v['accuracy']:.2f} | {v['completeness']:.2f} | {v['citations']:.2f} | {v['total']:.2f} |"
        for s, v in quality_by_system.items()
    )
    qt_order = ["how-to", "factual", "conceptual", "comparison", "new-in-2026"]
    qt_rows_lines = []
    for qt in qt_order:
        if qt not in quality_by_qt:
            continue
        for system in _BASELINES:
            sv = quality_by_qt[qt][system]
            qt_rows_lines.append(
                f"| {qt} | {system} | {sv['accuracy']:.2f} | {sv['completeness']:.2f} | "
                f"{sv['citations']:.2f} | {sv['total']:.2f} | {sv['n']} |"
            )
    quality_section = (
        "## Aggregate Quality\n\n"
        "### Overall by System\n\n"
        "| System | Avg Accuracy | Avg Completeness | Avg Citations | Avg Total |\n"
        "|--------|-------------|-----------------|--------------|----------|\n"
        f"{overall_rows}\n\n"
        "### By Query Type\n\n"
        "| Query Type | System | Avg Accuracy | Avg Completeness | Avg Citations | Avg Total | N |\n"
        "|------------|--------|-------------|-----------------|--------------|----------|---|\n"
        + "\n".join(qt_rows_lines)
    )

    # ── Section 4: Aggregate Retrieval ───────────────────────────────────────
    hr5 = ra.get("hr_reranked", 0)
    rc5 = ra.get("rc_reranked", 0)
    mrr5 = ra.get("mrr_reranked", 0)
    frr5 = ra.get("frr_reranked", 0)
    hr20 = ra.get("hr_candidates", 0)
    rc20 = ra.get("rc_candidates", 0)
    mrr20 = sum(r["mrr_candidates"] for r in retrieval_results) / n if n else 0
    qt_ret_rows = "\n".join(
        f"| {qt} | {retrieval_by_qt[qt]['hr_reranked']:.0%} | "
        f"{retrieval_by_qt[qt]['rc_reranked']:.0%} | "
        f"{retrieval_by_qt[qt]['mrr_reranked']:.2f} | {retrieval_by_qt[qt]['n']} |"
        for qt in qt_order if qt in retrieval_by_qt
    )
    retrieval_section = (
        "## Aggregate Retrieval Metrics\n\n"
        "### Overall\n\n"
        "| Metric | Reranked @5 | Candidates @20 |\n"
        "|--------|------------|---------------|\n"
        f"| Hit Rate | {hr5:.0%} ({int(hr5 * n)}/{n}) | {hr20:.0%} ({int(hr20 * n)}/{n}) |\n"
        f"| Recall | {rc5:.0%} avg | {rc20:.0%} avg |\n"
        f"| MRR | {mrr5:.2f} | {mrr20:.2f} |\n"
        f"| Full Recall Rate | {frr5:.0%} ({int(frr5 * n)}/{n}) | — |\n\n"
        "### By Query Type\n\n"
        "| Query Type | HR@5 | RC@5 | MRR@5 | N |\n"
        "|------------|------|------|-------|---|\n"
        f"{qt_ret_rows}"
    )

    # ── Section 5: Per-question quality table ────────────────────────────────
    quality_table_section = "## Per-Question Quality\n\n" + _render_quality_table(questions)

    # ── Section 6: Per-question retrieval table ──────────────────────────────
    retrieval_table_section = (
        "## Per-Question Retrieval\n\n" + _render_retrieval_table(questions, retrieval_results)
    )

    # ── Section 7: Recommendations ───────────────────────────────────────────
    rec_bullets = "\n".join(f"- {r}" for r in recommendations)
    rec_section = f"## Recommendations\n\n{rec_bullets}"

    return "\n\n---\n\n".join(
        [header, v1_section, quality_section, retrieval_section,
         quality_table_section, retrieval_table_section, rec_section]
    ) + "\n"


# ── CLI entry point ───────────────────────────────────────────────────────────


def main() -> None:
    """Generate evals/results.md from golden_set.json and sommelier_traces.jsonl."""
    parser = argparse.ArgumentParser(
        prog="report",
        description="Generate eval report from golden set quality scores and trace retrieval data.",
    )
    parser.add_argument("--golden-set", required=True, help="Path to golden_set.json")
    parser.add_argument(
        "--traces", default="sommelier_traces.jsonl", help="Path to traces JSONL file"
    )
    parser.add_argument(
        "--output", default="evals/results.md", help="Output markdown file path"
    )
    args = parser.parse_args()

    questions = load_golden_set(args.golden_set)
    traces = _parse_traces(args.traces)

    retrieval_results = _compute_retrieval_per_question(questions, traces)
    missing = [r["question_id"] for r in retrieval_results if r["trace_missing"]]
    if missing:
        print(f"Warning: no trace found for {len(missing)} question(s): {', '.join(missing)}")

    quality_by_system = _aggregate_quality_by_system(questions)
    quality_by_qt = _aggregate_quality_by_query_type(questions)
    retrieval_agg = aggregate_retrieval_metrics(retrieval_results)
    retrieval_by_qt = _aggregate_retrieval_by_query_type(questions, retrieval_results)
    v1_gates = _check_v1_gates(quality_by_system)
    recommendations = _generate_recommendations(
        questions, retrieval_results, quality_by_system, quality_by_qt
    )

    try:
        from observability.tracer import tracer
        prompt_version = tracer.prompt_version
    except Exception:
        prompt_version = "unknown"

    md = _render_markdown(
        questions=questions,
        retrieval_results=retrieval_results,
        quality_by_system=quality_by_system,
        quality_by_qt=quality_by_qt,
        retrieval_agg=retrieval_agg,
        retrieval_by_qt=retrieval_by_qt,
        v1_gates=v1_gates,
        recommendations=recommendations,
        golden_set_path=args.golden_set,
        traces_path=args.traces,
        prompt_version=prompt_version,
    )

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")

    print(f"Report written to {out}\n")
    for passed, msg in v1_gates:
        print(f"  {'✅' if passed else '❌'} {msg}")


if __name__ == "__main__":
    main()
