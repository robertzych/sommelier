"""Unit tests for evals/report.py: trace parsing, aggregation, recommendations, and rendering."""
import json
import pathlib

import pytest

from evals.report import (
    _aggregate_quality_by_query_type,
    _aggregate_quality_by_system,
    _aggregate_retrieval_by_query_type,
    _check_v1_gates,
    _compute_retrieval_per_question,
    _generate_recommendations,
    _parse_traces,
    _render_quality_table,
    _render_retrieval_table,
)

_BASELINES = [
    "docs_pinot_ai",
    "claude (claude-sonnet-4-6)",
    "sommelier (claude-haiku-4-5)",
]
_SOMMELIER = "sommelier (claude-haiku-4-5)"
_CLAUDE = "claude (claude-sonnet-4-6)"
_DOCS = "docs_pinot_ai"


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_question(
    qid: str = "q001",
    question: str = "How do you configure upsert?",
    query_type: str = "how-to",
    expected_sources: list[str] | None = None,
    scores: dict | None = None,
) -> dict:
    """Build a minimal golden set question dict."""
    if expected_sources is None:
        expected_sources = ["ingestion/upsert.md"]
    default_scores = {"accuracy": 1, "completeness": 1, "citations": 1, "total": 3}
    if scores:
        default_scores.update(scores)
    return {
        "id": qid,
        "question": question,
        "query_type": query_type,
        "expected_sources": expected_sources,
        "baseline_responses": {
            s: {"scores": dict(default_scores), "scored_by": "Robert Zych", "scored_on": "2026-05-14"}
            for s in _BASELINES
        },
    }


def _make_retrieval_result(
    question: dict,
    reranked_fps: list[str] | None = None,
    candidate_fps: list[str] | None = None,
    trace_missing: bool = False,
) -> dict:
    """Build a minimal retrieval result dict."""
    from evals.metrics import hit_rate, mrr, recall

    expected = question.get("expected_sources", [])
    reranked_fps = reranked_fps or []
    candidate_fps = candidate_fps or []
    return {
        "question_id": question["id"],
        "question": question["question"],
        "expected_sources": expected,
        "reranked_fps": reranked_fps,
        "candidate_fps": candidate_fps,
        "hit_rate_reranked": hit_rate(reranked_fps, expected),
        "recall_reranked": recall(reranked_fps, expected),
        "mrr_reranked": mrr(reranked_fps, expected),
        "hit_rate_candidates": hit_rate(candidate_fps, expected),
        "recall_candidates": recall(candidate_fps, expected),
        "mrr_candidates": mrr(candidate_fps, expected),
        "trace_missing": trace_missing,
    }


# ── TestParseTraces ───────────────────────────────────────────────────────────


class TestParseTraces:
    """Tests for _parse_traces: JSONL trace file parsing."""

    def test_returns_last_occurrence_per_query(self, tmp_path: pathlib.Path):
        """When two traces share a query text, the last one is returned."""
        lines = [
            json.dumps({"event_type": "query", "query": "What is the broker port?", "reranked": [{"file_path": "a.md"}]}),
            json.dumps({"event_type": "query", "query": "What is the broker port?", "reranked": [{"file_path": "b.md"}]}),
        ]
        f = tmp_path / "traces.jsonl"
        f.write_text("\n".join(lines) + "\n")

        result = _parse_traces(str(f))

        assert result["What is the broker port?"]["reranked"][0]["file_path"] == "b.md"

    def test_skips_non_query_event_types(self, tmp_path: pathlib.Path):
        """Traces with event_type != 'query' are not included."""
        lines = [
            json.dumps({"event_type": "ingestion", "query": "ignored"}),
            json.dumps({"event_type": "chat_turn", "query": "also ignored"}),
            json.dumps({"event_type": "query", "query": "included"}),
        ]
        f = tmp_path / "traces.jsonl"
        f.write_text("\n".join(lines) + "\n")

        result = _parse_traces(str(f))

        assert list(result.keys()) == ["included"]

    def test_empty_file_returns_empty_dict(self, tmp_path: pathlib.Path):
        """An empty JSONL file returns an empty dict."""
        f = tmp_path / "traces.jsonl"
        f.write_text("")

        assert _parse_traces(str(f)) == {}

    def test_missing_file_returns_empty_dict(self, tmp_path: pathlib.Path):
        """A missing file returns an empty dict without raising."""
        assert _parse_traces(str(tmp_path / "nonexistent.jsonl")) == {}


# ── TestComputeRetrievalPerQuestion ───────────────────────────────────────────


class TestComputeRetrievalPerQuestion:
    """Tests for _compute_retrieval_per_question: per-question metric computation."""

    def test_hit_rate_1_when_expected_source_in_reranked(self):
        """HR@5 = 1.0 when expected source appears in the trace's reranked list."""
        q = _make_question(expected_sources=["docs/upsert.md"])
        traces = {q["question"]: {"reranked": [{"file_path": "docs/upsert.md"}], "candidates": []}}

        results = _compute_retrieval_per_question([q], traces)

        assert results[0]["hit_rate_reranked"] == 1.0

    def test_hit_rate_0_when_no_expected_source_in_reranked(self):
        """HR@5 = 0.0 when no expected source appears in the reranked list."""
        q = _make_question(expected_sources=["docs/upsert.md"])
        traces = {q["question"]: {"reranked": [{"file_path": "docs/other.md"}], "candidates": []}}

        results = _compute_retrieval_per_question([q], traces)

        assert results[0]["hit_rate_reranked"] == 0.0

    def test_missing_trace_sets_all_metrics_zero(self):
        """When no trace matches a question, all metrics are 0.0 and trace_missing=True."""
        q = _make_question()
        results = _compute_retrieval_per_question([q], {})

        r = results[0]
        assert r["trace_missing"] is True
        assert r["hit_rate_reranked"] == 0.0
        assert r["recall_reranked"] == 0.0
        assert r["mrr_reranked"] == 0.0


# ── TestAggregateQualityBySystem ──────────────────────────────────────────────


class TestAggregateQualityBySystem:
    """Tests for _aggregate_quality_by_system: system-level quality averages."""

    def test_averages_across_all_questions(self):
        """Averages are computed correctly across two questions."""
        q1 = _make_question("q001", scores={"accuracy": 1, "completeness": 1, "citations": 1, "total": 3})
        q2 = _make_question("q002", scores={"accuracy": 0, "completeness": 0, "citations": 0, "total": 0})

        result = _aggregate_quality_by_system([q1, q2])

        for system in _BASELINES:
            assert result[system]["accuracy"] == pytest.approx(0.5)
            assert result[system]["total"] == pytest.approx(1.5)

    def test_empty_questions_returns_zeros(self):
        """Empty question list returns zero averages for all dimensions."""
        result = _aggregate_quality_by_system([])
        for system in _BASELINES:
            assert result[system]["total"] == 0.0


# ── TestAggregateQualityByQueryType ──────────────────────────────────────────


class TestAggregateQualityByQueryType:
    """Tests for _aggregate_quality_by_query_type: grouping by query_type."""

    def test_groups_correctly_by_query_type(self):
        """Questions are bucketed into their respective query types."""
        q1 = _make_question("q001", query_type="how-to", scores={"accuracy": 1, "completeness": 1, "citations": 1, "total": 3})
        q2 = _make_question("q002", query_type="factual", scores={"accuracy": 0, "completeness": 0, "citations": 0, "total": 0})

        result = _aggregate_quality_by_query_type([q1, q2])

        assert "how-to" in result
        assert "factual" in result
        assert result["how-to"][_DOCS]["total"] == pytest.approx(3.0)
        assert result["factual"][_DOCS]["total"] == pytest.approx(0.0)

    def test_n_count_correct_per_group(self):
        """N reflects the number of questions in each query_type group."""
        questions = [
            _make_question(f"q00{i}", query_type="how-to") for i in range(3)
        ] + [_make_question("q004", query_type="factual")]

        result = _aggregate_quality_by_query_type(questions)

        assert result["how-to"][_DOCS]["n"] == 3
        assert result["factual"][_DOCS]["n"] == 1


# ── TestAggregateRetrievalByQueryType ────────────────────────────────────────


class TestAggregateRetrievalByQueryType:
    """Tests for _aggregate_retrieval_by_query_type: retrieval metrics grouped by query_type."""

    def test_groups_retrieval_metrics_by_type(self):
        """HR@5 for each query_type is the average of its questions' HR@5 values."""
        q1 = _make_question("q001", query_type="how-to", expected_sources=["a.md"])
        q2 = _make_question("q002", query_type="how-to", expected_sources=["b.md"])
        q3 = _make_question("q003", query_type="factual", expected_sources=["c.md"])
        r1 = _make_retrieval_result(q1, reranked_fps=["a.md"])  # HR=1
        r2 = _make_retrieval_result(q2, reranked_fps=["x.md"])  # HR=0
        r3 = _make_retrieval_result(q3, reranked_fps=["c.md"])  # HR=1

        result = _aggregate_retrieval_by_query_type([q1, q2, q3], [r1, r2, r3])

        assert result["how-to"]["hr_reranked"] == pytest.approx(0.5)
        assert result["factual"]["hr_reranked"] == pytest.approx(1.0)
        assert result["how-to"]["n"] == 2
        assert result["factual"]["n"] == 1


# ── TestCheckV1Gates ──────────────────────────────────────────────────────────


class TestCheckV1Gates:
    """Tests for _check_v1_gates: V1 success criteria pass/fail."""

    def _make_quality(self, som_total, som_acc, som_cmp, som_cit, cld_acc, cld_cmp, cld_cit):
        return {
            _SOMMELIER: {"accuracy": som_acc, "completeness": som_cmp, "citations": som_cit, "total": som_total},
            _CLAUDE: {"accuracy": cld_acc, "completeness": cld_cmp, "citations": cld_cit, "total": 2.0},
            _DOCS: {"accuracy": 1.0, "completeness": 1.0, "citations": 1.0, "total": 3.0},
        }

    def test_all_gates_pass_when_sommelier_dominates(self):
        """All four gates pass when sommelier has total ≥ 2.5 and beats claude on all dims."""
        quality = self._make_quality(2.6, 0.9, 0.9, 0.9, 0.7, 0.7, 0.7)
        gates = _check_v1_gates(quality)
        assert all(p for p, _ in gates)

    def test_total_gate_fails_when_below_2_5(self):
        """Total gate fails when sommelier avg total < 2.5."""
        quality = self._make_quality(2.4, 0.9, 0.9, 0.9, 0.7, 0.7, 0.7)
        gates = _check_v1_gates(quality)
        assert gates[0][0] is False

    def test_dimension_gate_fails_when_below_claude(self):
        """A dimension gate fails when sommelier's score ≤ claude's score on that dim."""
        quality = self._make_quality(2.6, 0.6, 0.9, 0.9, 0.7, 0.7, 0.7)
        gates = _check_v1_gates(quality)
        passed = {msg: p for p, msg in gates}
        assert any(not p and "accuracy" in msg for p, msg in gates)


# ── TestGenerateRecommendations ───────────────────────────────────────────────


class TestGenerateRecommendations:
    """Tests for _generate_recommendations: rule-based recommendation generation."""

    def _run(self, questions, retrieval_results):
        qbs = _aggregate_quality_by_system(questions)
        qbt = _aggregate_quality_by_query_type(questions)
        return _generate_recommendations(questions, retrieval_results, qbs, qbt)

    def test_flags_retrieval_miss_with_score_drop(self):
        """Rule 1: flags question where HR@5=0 and sommelier scored below baseline average."""
        q = _make_question(expected_sources=["a.md"])
        q["baseline_responses"][_DOCS]["scores"] = {"accuracy": 1, "completeness": 1, "citations": 1, "total": 3}
        q["baseline_responses"][_CLAUDE]["scores"] = {"accuracy": 1, "completeness": 1, "citations": 1, "total": 3}
        q["baseline_responses"][_SOMMELIER]["scores"] = {"accuracy": 0, "completeness": 0, "citations": 0, "total": 0}
        r = _make_retrieval_result(q, reranked_fps=["other.md"])  # miss

        recs = self._run([q], [r])

        assert any("retrieval miss" in rec for rec in recs)

    def test_no_flag_when_retrieval_miss_but_score_ok(self):
        """Rule 1: no flag when HR@5=0 but sommelier matches baseline average."""
        q = _make_question(expected_sources=["a.md"])
        for system in _BASELINES:
            q["baseline_responses"][system]["scores"] = {"accuracy": 1, "completeness": 1, "citations": 1, "total": 3}
        r = _make_retrieval_result(q, reranked_fps=["other.md"])  # miss but scores equal

        recs = self._run([q], [r])

        assert not any("retrieval miss" in rec for rec in recs)

    def test_flags_underperforming_query_type(self):
        """Rule 2: flags a query_type where docs_pinot_ai avg total exceeds sommelier by > 0.5."""
        q = _make_question(query_type="factual")
        q["baseline_responses"][_DOCS]["scores"] = {"accuracy": 1, "completeness": 1, "citations": 1, "total": 3}
        q["baseline_responses"][_CLAUDE]["scores"] = {"accuracy": 1, "completeness": 1, "citations": 1, "total": 3}
        q["baseline_responses"][_SOMMELIER]["scores"] = {"accuracy": 0, "completeness": 0, "citations": 0, "total": 0}
        r = _make_retrieval_result(q, reranked_fps=["a.md"])

        recs = self._run([q], [r])

        assert any("factual" in rec for rec in recs)

    def test_flags_never_retrieved_source(self):
        """Rule 3: flags an expected source that is missed across ≥ 2 questions."""
        qs = [_make_question(f"q00{i}", expected_sources=["hard-to-find.md"]) for i in range(3)]
        rs = [_make_retrieval_result(q, reranked_fps=["other.md"]) for q in qs]

        recs = self._run(qs, rs)

        assert any("hard-to-find.md" in rec for rec in recs)

    def test_flags_weak_citation_dimension(self):
        """Rule 4: flags sommelier citations avg < 0.5."""
        q = _make_question()
        q["baseline_responses"][_SOMMELIER]["scores"] = {"accuracy": 1, "completeness": 1, "citations": 0, "total": 2}
        # Make many questions so avg stays low
        questions = [q] * 5
        retrieval = [_make_retrieval_result(q, reranked_fps=["ingestion/upsert.md"])] * 5

        recs = self._run(questions, retrieval)

        assert any("citations" in rec for rec in recs)

    def test_empty_list_when_all_pass(self):
        """No recommendations when sommelier dominates and retrieval is perfect."""
        questions = [
            _make_question(f"q00{i}", expected_sources=["a.md"]) for i in range(5)
        ]
        # Give sommelier high scores, beats claude
        for q in questions:
            q["baseline_responses"][_SOMMELIER]["scores"] = {"accuracy": 1, "completeness": 1, "citations": 1, "total": 3}
            q["baseline_responses"][_CLAUDE]["scores"] = {"accuracy": 0, "completeness": 0, "citations": 0, "total": 0}
            q["baseline_responses"][_DOCS]["scores"] = {"accuracy": 1, "completeness": 1, "citations": 1, "total": 3}
        retrieval = [_make_retrieval_result(q, reranked_fps=["a.md"]) for q in questions]

        recs = self._run(questions, retrieval)

        assert recs == ["No significant issues detected. All V1 gates passed and retrieval looks healthy."]


# ── TestRenderQualityTable ────────────────────────────────────────────────────


class TestRenderQualityTable:
    """Tests for _render_quality_table: markdown table structure."""

    def test_75_data_rows_for_25_questions(self):
        """75 data rows (25 questions × 3 systems) appear after the 2 header lines."""
        questions = [_make_question(f"q{i:03d}", question=f"Question {i}?") for i in range(25)]
        table = _render_quality_table(questions)
        data_rows = [line for line in table.split("\n") if line.startswith("|") and "---" not in line and "Query" not in line]
        assert len(data_rows) == 75

    def test_systems_in_correct_order(self):
        """docs_pinot_ai appears before claude before sommelier in each question block."""
        q = _make_question()
        table = _render_quality_table([q])
        rows = [line for line in table.split("\n") if line.startswith("|") and "---" not in line and "Query" not in line]
        assert "docs_pinot_ai" in rows[0]
        assert "claude" in rows[1]
        assert "sommelier" in rows[2]

    def test_query_truncated_to_50_chars(self):
        """A question longer than 50 chars is truncated with '…' in the first system row."""
        long_q = "A" * 60 + "?"
        q = _make_question(question=long_q)
        table = _render_quality_table([q])
        assert "A" * 50 + "…" in table


# ── TestRenderRetrievalTable ──────────────────────────────────────────────────


class TestRenderRetrievalTable:
    """Tests for _render_retrieval_table: markdown table structure."""

    def test_25_data_rows(self):
        """25 data rows (one per question) appear after the header lines."""
        questions = [_make_question(f"q{i:03d}", question=f"Question {i}?", expected_sources=["a.md"]) for i in range(25)]
        retrieval = [_make_retrieval_result(q, reranked_fps=["a.md"]) for q in questions]
        table = _render_retrieval_table(questions, retrieval)
        data_rows = [line for line in table.split("\n") if line.startswith("|") and "---" not in line and "ID" not in line]
        assert len(data_rows) == 25

    def test_miss_yes_when_hr_zero(self):
        """'yes' appears in a row when HR@5 = 0.0."""
        q = _make_question(expected_sources=["a.md"])
        r = _make_retrieval_result(q, reranked_fps=["other.md"])
        table = _render_retrieval_table([q], [r])
        assert "yes" in table

    def test_miss_no_when_hr_nonzero(self):
        """'no' appears in a row when HR@5 > 0."""
        q = _make_question(expected_sources=["a.md"])
        r = _make_retrieval_result(q, reranked_fps=["a.md"])
        table = _render_retrieval_table([q], [r])
        assert "no" in table
