"""Unit tests for evals/eval.py: aggregate metrics and retrieval summary output."""
import pytest

from evals.eval import _aggregate_retrieval_metrics, _print_retrieval_summary
from evals.metrics import hit_rate, mrr, recall


def _make_result(reranked_fps: list[str], candidate_fps: list[str], expected: list[str]) -> dict:
    """Build a minimal per-question result dict for aggregate metric tests."""
    return {
        "reranked_fps": reranked_fps,
        "candidate_fps": candidate_fps,
        "expected_sources": expected,
        "hit_rate_reranked": hit_rate(reranked_fps, expected),
        "recall_reranked": recall(reranked_fps, expected),
        "mrr_reranked": mrr(reranked_fps, expected),
        "hit_rate_candidates": hit_rate(candidate_fps, expected),
        "recall_candidates": recall(candidate_fps, expected),
    }


# ── _aggregate_retrieval_metrics ──────────────────────────────────────────────


class TestAggregateRetrievalMetrics:
    """Tests for _aggregate_retrieval_metrics: averages per-question metrics to dataset level."""

    def test_empty_results_returns_empty_dict(self):
        """Returns an empty dict when no results are provided."""
        assert _aggregate_retrieval_metrics([]) == {}

    def test_averages_mixed_hit_and_miss(self):
        """Averages are 0.5 when one question hits and one misses across all metrics."""
        results = [
            _make_result(["a.md"], ["a.md"], ["a.md"]),  # all metrics = 1.0
            _make_result(["z.md"], ["z.md"], ["b.md"]),  # all metrics = 0.0
        ]
        agg = _aggregate_retrieval_metrics(results)

        assert agg["n"] == 2
        assert agg["hr_reranked"] == pytest.approx(0.5)
        assert agg["rc_reranked"] == pytest.approx(0.5)
        assert agg["mrr_reranked"] == pytest.approx(0.5)
        assert agg["hr_candidates"] == pytest.approx(0.5)
        assert agg["rc_candidates"] == pytest.approx(0.5)

    def test_full_recall_rate_reflects_all_sources_found(self):
        """frr_reranked is 0.5 when one question has full recall and one does not."""
        results = [
            _make_result(["a.md", "b.md"], ["a.md", "b.md"], ["a.md", "b.md"]),  # full recall
            _make_result(["c.md"], ["c.md"], ["c.md", "d.md"]),  # partial recall
        ]
        agg = _aggregate_retrieval_metrics(results)

        assert agg["frr_reranked"] == pytest.approx(0.5)


# ── _print_retrieval_summary ──────────────────────────────────────────────────


class TestPrintRetrievalSummary:
    """Tests for _print_retrieval_summary: verifies the full formatted report."""

    def test_full_output_matches_expected(self, capsys):
        """Prints the exact two-block retrieval report for given aggregate metrics."""
        agg = {
            "n": 20,
            "hr_reranked": 0.9,
            "rc_reranked": 0.85,
            "mrr_reranked": 0.78,
            "frr_reranked": 0.8,
            "hr_candidates": 0.95,
            "rc_candidates": 0.92,
        }
        _print_retrieval_summary(agg, rerank_k=5, retrieval_k=20)
        out = capsys.readouterr().out

        expected = (
            "Retrieval (post-reranker, top-5):\n"
            "  Hit Rate:    90%   (18/20)\n"
            "  Recall:      85%   avg\n"
            "  MRR:         0.78  avg\n"
            "  Full Recall: 80%   (16/20 all expected sources found)\n"
            "\n"
            "Retrieval (candidates, top-20):\n"
            "  Hit Rate:    95%   (19/20)\n"
            "  Recall:      92%   avg\n"
        )
        assert out == expected
