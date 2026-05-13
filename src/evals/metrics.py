"""Pure retrieval metric functions operating on file_path lists."""


def hit_rate(retrieved: list[str], expected: list[str]) -> float:
    """Return 1.0 if any expected source appears in retrieved, else 0.0."""
    if not expected:
        return 0.0
    return 1.0 if any(e in retrieved for e in expected) else 0.0


def recall(retrieved: list[str], expected: list[str]) -> float:
    """Return fraction of expected sources that appear in retrieved."""
    if not expected:
        return 0.0
    return sum(1 for e in expected if e in retrieved) / len(expected)


def mrr(retrieved: list[str], expected: list[str]) -> float:
    """Return reciprocal rank of the first expected source hit in retrieved."""
    for rank, item in enumerate(retrieved, start=1):
        if item in expected:
            return 1.0 / rank
    return 0.0


def full_recall_rate(questions: list[dict]) -> float:
    """Return fraction of questions where all expected sources appear in retrieved.

    Each dict must have 'retrieved' (list[str]) and 'expected_sources' (list[str]).
    """
    if not questions:
        return 0.0
    full_recalls = sum(
        1
        for q in questions
        if all(e in q["retrieved"] for e in q["expected_sources"])
    )
    return full_recalls / len(questions)
