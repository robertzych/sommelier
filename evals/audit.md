# Sommelier Eval Pipeline Audit

Six subagents each ran one diagnostic area against real repo artifacts (`golden_set.json`, `score_review.py`, `retrieval_findings.md`, `plan.md`, `eval.py`/`metrics.py`/`report.py`, git history). Synthesized and de-duplicated below, ordered by impact.

## High Impact

### 1. Human review sees answers only, never the retrieval trace

**Status:** Problem exists
`evals/score_review.py` prints only `response["answer"]` (line 93) before asking for accuracy/completeness/citations scores. No retrieved chunks, reranked candidates, or intermediate trace are shown — a reviewer scoring "citations" can't check the citation against what was actually retrieved. Trace inspection exists (`inspect_trace.py`, `inspect_chunks.py`) but is a separate manual tool, not wired into scoring. Output is also raw, unformatted terminal text (markdown/code embedded in Pinot doc answers isn't rendered).
**Fix:** `build-review-interface` — build a browser-based reviewer that shows question + retrieved/reranked chunks + final answer together, with markdown rendering and code syntax highlighting.

### 2. Single rater, no inter-rater validation

**Status:** Problem exists
All 75 scored responses (25 questions × 3 baselines) have `scored_by: "Robert Zych"` — one person, no independence check. Already tracked in `plan.md:832` as an un-started "Multi-rater evaluation" step (Cohen's κ on completeness) — the gap is known but not executed.
**Fix:** Score a ~20% sample (15 responses) with a second independent rater; compute Cohen's κ on `completeness` (the most subjective dimension).

### 3. Error analysis scope is anomaly-driven, not systematic

**Status:** Problem exists
`retrieval_findings.md` investigated only the 6 questions flagged by retrieval-miss/score rules — the other 22 "passing" traces were never qualitatively reviewed. Failure modes invisible to the binary rubric (verbosity, over-broad citations, tone) could be hiding in traces marked "pass." Defensible at V1 scale, but a real blind spot.
**Fix:** Run `error-analysis` on a random sample of the passing traces, not just the flagged misses.

## Medium Impact

### 4. Golden set is too small for what comes next

**Status:** Problem exists
25 questions is below the skill's ~100-trace saturation target for error analysis. More critically for future judge work: Sommelier's own answers pass 24-25/25 on every dimension — pooled "fail" examples (63/12 accuracy, 71/4 completeness, 50/25 citations) come almost entirely from the deliberately-weak no-RAG Claude baseline, not from Sommelier itself. A judge validated on this pool wouldn't be calibrated to Sommelier's actual (rare) failure patterns.
**Fix:** Not urgent today (no judge exists yet) — but before judge validation starts, use `generate-synthetic-data` to construct harder/adversarial Sommelier-specific failure cases rather than reusing the current pool as-is.

### 5. Drafted future JUDGE_PROMPT is under-specified

**Status:** Problem exists (in planned, unbuilt code)
`plan.md`'s draft judge prompt asks three dimension-scoped questions but each is still loosely defined ("technically correct," "sufficiently address," undefined "cite") — no failure taxonomy, no few-shot examples.
**Fix:** Run `write-judge-prompt` before implementing `--judge claude`.

### 6. Planned judge-validation methodology uses spot-check/Cohen's κ, not TPR/TNR

**Status:** Problem exists (in planned, unbuilt code)
`plan.md:728` plans a "20% spot-check"; `plan.md:832` plans Cohen's κ. Neither is TPR/TNR, which the skill flags as the correct metric under class imbalance (and this pipeline is imbalanced — see #4).
**Fix:** Run `validate-evaluator` when judge work begins; require TPR/TNR on a held-out split, not raw agreement.

### 7. No CI enforcement of the change→re-eval discipline

**Status:** Problem exists (process gap, not a violation yet)
The actual discipline is strong in practice — commit history shows a real change→re-eval loop (chunking fixes → regrade, prompt format change → full 25-question regrade), and `prompt_version` (SHA-256 of `system_prompt.md`) is stamped on every trace for traceability. But it's entirely manual; no CI/pre-commit check exists to catch a future change made without re-running evals.
**Fix:** Add a lightweight check (CI or pre-commit) that flags when `system_prompt.md`/`ingest.py` changed but `results.md`'s embedded prompt_version is stale.

## Things Done Well (no action needed)

- Binary 0/1 scoring per dimension (no Likert-scale noise) — `score_review.py`.
- Retrieval metrics (`hit_rate`, `recall`, `mrr`) are pure code, zero LLM/similarity-metric involvement.
- No LLM judge deployed before a manual baseline was established — correctly sequenced per `eval.py`'s own docstring.
- Failure categories that were investigated are trace-grounded and application-specific (config table fragmentation, prose/example chunk splitting, reranker table bias), not generic ("hallucination," "coherence") labels.
- V1 gates check per-dimension pass/fail alongside the aggregate total, not just the scalar sum.

## Next Steps

Ready to act on now, independent of the deferred `--judge claude` work:

1. **Broaden error analysis** (#3) — run `error-analysis` over a random sample of the 22 currently-"passing" traces, not just the 6 flagged misses, to surface failure modes the binary rubric can't see.
2. **Build a real review interface** (#1) — run `build-review-interface` to replace `score_review.py`'s raw-text-only flow with a browser UI showing question + retrieved/reranked chunks + answer together, rendered as markdown.
3. **Run a second rater** (#2) — score a ~20% sample (15 of the 75 responses) independently and compute Cohen's κ on `completeness`; this is manual work for a human reviewer, not something a skill automates.
4. **Add a CI/pre-commit staleness check** (#7) — flag when `system_prompt.md` or `ingest.py` changes without a corresponding `results.md` regeneration.

Deferred until `--judge claude` work actually begins (running these now would be premature — no judge exists yet to validate):

5. `write-judge-prompt` (#5) — tighten the drafted `JUDGE_PROMPT` in `plan.md` with explicit pass/fail definitions and few-shot examples.
6. `validate-evaluator` (#6) — require TPR/TNR on a held-out split for judge calibration, not raw agreement/Cohen's κ.
7. `generate-synthetic-data` (#4) — construct harder/adversarial Sommelier-specific failure cases before judge validation, since the current golden set has too few genuine Sommelier failures to calibrate against.
