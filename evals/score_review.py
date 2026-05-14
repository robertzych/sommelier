"""Interactive scorer: displays golden set Q&A and prompts for scores.

Skips questions where all three baselines are already fully scored.
Saves scores to golden_set.json after each baseline is completed.
Run from the repo root: uv run python evals/score_review.py
"""
import json
import os
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

_PT = ZoneInfo("America/Los_Angeles")

GOLDEN_SET_PATH = "evals/golden_set.json"
BASELINES = [
    "docs_pinot_ai",
    "claude (claude-sonnet-4-6)",
    "sommelier (claude-haiku-4-5)",
]
DIMENSIONS = [
    ("accuracy",     "Is the answer technically correct based on Apache Pinot documentation?"),
    ("completeness", "Does the answer cover all key aspects needed to fully answer the question?"),
    ("citations",    "Does the answer cite or reference relevant documentation sources?"),
]


def load() -> dict:
    with open(GOLDEN_SET_PATH) as f:
        return json.load(f)


def save(gs: dict) -> None:
    with open(GOLDEN_SET_PATH, "w") as f:
        json.dump(gs, f, indent=2)
        f.write("\n")


def is_scored(question: dict) -> bool:
    return all(
        question["baseline_responses"][b]["scores"]["total"] is not None
        for b in BASELINES
    )


def prompt_score(dimension: str, question_text: str) -> int:
    while True:
        raw = input(f"  {dimension} (0 or 1) — {question_text}\n  > ").strip()
        if raw.lower() == "q":
            print("Exiting.")
            sys.exit(0)
        if raw in ("0", "1"):
            return int(raw)
        print("  Enter 0 or 1. Type 'q' to quit.")


def divider(char: str = "─", width: int = 70) -> str:
    return char * width


def fmt(text) -> str:
    return text if text is not None else "(no answer)"


def review_question(q: dict, gs: dict, index: int, total_pending: int) -> None:
    print()
    done_by = (datetime.now(_PT) + timedelta(minutes=5)).strftime("%-I:%M %p %Z")
    print(divider("═"))
    print(f"  {q['id']} — {q['question']}")
    print(f"  type: {q['query_type']} | ({index}/{total_pending} pending) | done by {done_by}")
    print(divider("═"))

    for baseline in BASELINES:
        response = q["baseline_responses"][baseline]
        scores = response["scores"]

        if scores["total"] is not None:
            score_str = (
                f"accuracy={scores['accuracy']}  completeness={scores['completeness']}"
                f"  citations={scores['citations']}  total={scores['total']}"
            )
            print()
            print(divider())
            print(f"  {baseline}  [already scored: {score_str}]")
            print(divider())
            continue

        print()
        print(divider())
        print(f"  {baseline}")
        print(divider())
        print(fmt(response["answer"]))
        print()

        collected = {}
        for dim, _ in DIMENSIONS:
            collected[dim] = prompt_score(dim, q["question"])
            scores[dim] = collected[dim]
            save(gs)
            print()

        collected["total"] = sum(collected.values())
        scores["total"] = collected["total"]
        response["scored_by"] = "Robert Zych"
        response["scored_on"] = datetime.now(_PT).strftime("%Y-%m-%dT%H:%M:%S%z")
        save(gs)
        print(f"  total: {collected['total']}")


def main() -> None:
    gs = load()
    questions = gs["questions"]
    pending = [q for q in questions if not is_scored(q)]

    if not pending:
        print("All questions are scored.")
        return

    print(f"\nGolden set: {len(questions)} questions total, {len(pending)} pending.")
    print("Enter scores 0–1 for each dimension. Type 'q' at any prompt to quit.")

    for i, q in enumerate(pending, 1):
        review_question(q, gs, i, len(pending))

    print("\nAll pending questions scored.")


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    main()
