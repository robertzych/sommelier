"""Unit tests for evals/metrics.py: hit_rate, recall, mrr, full_recall_rate."""
import pytest

from evals.metrics import full_recall_rate, hit_rate, mrr, recall


# ── hit_rate ──────────────────────────────────────────────────────────────────


class TestHitRate:
    """Tests for hit_rate: 1.0 if any expected source appears in retrieved."""

    def test_single_expected_found(self):
        """Returns 1.0 when the one expected source is in retrieved."""
        assert hit_rate(["a.md", "b.md"], ["a.md"]) == 1.0

    def test_multiple_expected_one_found(self):
        """Returns 1.0 when at least one of several expected sources is found."""
        assert hit_rate(["a.md", "b.md"], ["b.md", "c.md"]) == 1.0

    def test_no_expected_found(self):
        """Returns 0.0 when none of the expected sources appear in retrieved."""
        assert hit_rate(["a.md", "b.md"], ["c.md"]) == 0.0

    def test_empty_retrieved(self):
        """Returns 0.0 when retrieved list is empty."""
        assert hit_rate([], ["a.md"]) == 0.0

    def test_empty_expected(self):
        """Returns 0.0 when expected list is empty (no sources to find)."""
        assert hit_rate(["a.md"], []) == 0.0

    def test_both_empty(self):
        """Returns 0.0 when both lists are empty."""
        assert hit_rate([], []) == 0.0

    def test_all_expected_found(self):
        """Returns 1.0 when all expected sources are in retrieved."""
        assert hit_rate(["a.md", "b.md", "c.md"], ["a.md", "c.md"]) == 1.0

    def test_exact_match_single(self):
        """Returns 1.0 when retrieved contains exactly the one expected source."""
        assert hit_rate(["x.md"], ["x.md"]) == 1.0


# ── recall ────────────────────────────────────────────────────────────────────


class TestRecall:
    """Tests for recall: fraction of expected sources found in retrieved."""

    def test_all_expected_found(self):
        """Returns 1.0 when every expected source appears in retrieved."""
        assert recall(["a.md", "b.md", "c.md"], ["a.md", "b.md"]) == 1.0

    def test_none_found(self):
        """Returns 0.0 when no expected sources appear in retrieved."""
        assert recall(["a.md"], ["b.md", "c.md"]) == 0.0

    def test_half_found(self):
        """Returns 0.5 when exactly half of expected sources are found."""
        assert recall(["a.md", "c.md"], ["a.md", "b.md"]) == pytest.approx(0.5)

    def test_one_of_three_found(self):
        """Returns 1/3 when one of three expected sources is found."""
        assert recall(["a.md"], ["a.md", "b.md", "c.md"]) == pytest.approx(1 / 3)

    def test_empty_retrieved(self):
        """Returns 0.0 when retrieved is empty."""
        assert recall([], ["a.md"]) == 0.0

    def test_empty_expected(self):
        """Returns 0.0 when expected is empty."""
        assert recall(["a.md"], []) == 0.0

    def test_retrieved_superset_of_expected(self):
        """Returns 1.0 when retrieved is a superset of expected."""
        assert recall(["a.md", "b.md", "c.md", "d.md"], ["b.md", "c.md"]) == 1.0


# ── mrr ───────────────────────────────────────────────────────────────────────


class TestMRR:
    """Tests for mrr: reciprocal rank of the first expected source hit."""

    def test_first_position_match(self):
        """Returns 1.0 when the first retrieved item is an expected source."""
        assert mrr(["a.md", "b.md", "c.md"], ["a.md"]) == pytest.approx(1.0)

    def test_second_position_match(self):
        """Returns 0.5 when the second retrieved item is the first expected hit."""
        assert mrr(["x.md", "a.md", "b.md"], ["a.md"]) == pytest.approx(0.5)

    def test_third_position_match(self):
        """Returns 1/3 when the third retrieved item is the first expected hit."""
        assert mrr(["x.md", "y.md", "a.md"], ["a.md"]) == pytest.approx(1 / 3)

    def test_no_match(self):
        """Returns 0.0 when no expected source appears in retrieved."""
        assert mrr(["x.md", "y.md"], ["a.md"]) == pytest.approx(0.0)

    def test_empty_retrieved(self):
        """Returns 0.0 when retrieved is empty."""
        assert mrr([], ["a.md"]) == pytest.approx(0.0)

    def test_empty_expected(self):
        """Returns 0.0 when expected is empty (nothing to find)."""
        assert mrr(["a.md"], []) == pytest.approx(0.0)

    def test_multiple_expected_uses_first_hit(self):
        """Returns reciprocal rank of the first occurrence, ignoring later ones."""
        # "b.md" hits at rank 2; "a.md" hits at rank 3 — MRR = 1/2
        assert mrr(["x.md", "b.md", "a.md"], ["a.md", "b.md"]) == pytest.approx(0.5)

    def test_single_item_retrieved_match(self):
        """Returns 1.0 when the single retrieved item matches expected."""
        assert mrr(["a.md"], ["a.md", "b.md"]) == pytest.approx(1.0)


# ── full_recall_rate ──────────────────────────────────────────────────────────


class TestFullRecallRate:
    """Tests for full_recall_rate: fraction of questions with all expected sources found."""

    def test_all_questions_full_recall(self):
        """Returns 1.0 when every question has all expected sources in retrieved."""
        questions = [
            {"retrieved": ["a.md", "b.md"], "expected_sources": ["a.md"]},
            {"retrieved": ["c.md", "d.md"], "expected_sources": ["c.md", "d.md"]},
        ]
        assert full_recall_rate(questions) == pytest.approx(1.0)

    def test_no_questions_full_recall(self):
        """Returns 0.0 when no question has all expected sources in retrieved."""
        questions = [
            {"retrieved": ["a.md"], "expected_sources": ["a.md", "b.md"]},
            {"retrieved": ["c.md"], "expected_sources": ["c.md", "d.md"]},
        ]
        assert full_recall_rate(questions) == pytest.approx(0.0)

    def test_half_questions_full_recall(self):
        """Returns 0.5 when half of questions have full recall."""
        questions = [
            {"retrieved": ["a.md", "b.md"], "expected_sources": ["a.md", "b.md"]},
            {"retrieved": ["c.md"], "expected_sources": ["c.md", "d.md"]},
        ]
        assert full_recall_rate(questions) == pytest.approx(0.5)

    def test_empty_questions(self):
        """Returns 0.0 when questions list is empty."""
        assert full_recall_rate([]) == pytest.approx(0.0)

    def test_empty_expected_sources(self):
        """Returns 1.0 when expected_sources is empty (vacuously satisfied)."""
        questions = [{"retrieved": ["a.md"], "expected_sources": []}]
        assert full_recall_rate(questions) == pytest.approx(1.0)

    def test_single_question_full_recall(self):
        """Returns 1.0 for a single question with all expected sources found."""
        questions = [{"retrieved": ["a.md", "b.md", "c.md"], "expected_sources": ["b.md"]}]
        assert full_recall_rate(questions) == pytest.approx(1.0)

    def test_single_question_partial_recall(self):
        """Returns 0.0 for a single question missing one expected source."""
        questions = [{"retrieved": ["a.md"], "expected_sources": ["a.md", "b.md"]}]
        assert full_recall_rate(questions) == pytest.approx(0.0)

    def test_one_of_three_questions_full_recall(self):
        """Returns 1/3 when one of three questions has full recall."""
        questions = [
            {"retrieved": ["a.md", "b.md"], "expected_sources": ["a.md", "b.md"]},
            {"retrieved": ["c.md"], "expected_sources": ["c.md", "d.md"]},
            {"retrieved": ["e.md"], "expected_sources": ["f.md"]},
        ]
        assert full_recall_rate(questions) == pytest.approx(1 / 3)
