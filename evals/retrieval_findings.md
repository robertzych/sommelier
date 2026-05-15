# Retrieval Investigation Findings

Investigation of the 6 questions with retrieval misses or scoring issues identified
in the first eval run. Conducted 2026-05-15 against `sommelier_traces.jsonl`.

---

## q003 — How do you configure a star-tree index in Pinot?

**Scores:** accuracy=1, completeness=0, citations=1, total=2  
**Retrieval:** HR@5=100%, MRR@5=0.20 (expected source at reranked rank 5)

**Finding:** The expected source `build-with-pinot/indexing/star-tree-index.md` was
retrieved but only via intro/limitations chunks. The example section (chunk [31]) —
which contains the complete `tableIndexConfig.starTreeIndexConfigs` JSON — was not in
the top-20 candidates at all. The LLM received no concrete config example and used
`// Star-tree configuration details` as a placeholder.

**Root cause:** The chunker splits on markdown headers, so the "Example" section
becomes an isolated chunk of SQL + JSON with minimal prose context. The query
"how do you configure" matches the intro chunks (high prose overlap) but not the
example chunk (mostly JSON with specific column names like `Country`, `Browser`).

**Recommendation:** Keep example sections co-located with the prose section that
introduces them rather than splitting on every header. The "Index generation
configuration" prose + the "Example" JSON should be one chunk.

---

## q005 — How do you set up real-time ingestion from Kafka in Pinot?

**Scores:** accuracy=1, completeness=0, citations=1, total=2  
**Retrieval:** HR@5=0% (expected source `stream-ingestion/README.md` never retrieved)

**Finding — wrong expected source:** The correct authoritative document is
`basics/getting-started/first-stream-ingest.md` (a dedicated Kafka setup walkthrough),
not the general stream ingestion README. `first-stream-ingest.md` was retrieved at
reranked rank 1.

**Finding — config chunk not retrieved:** `first-stream-ingest.md` has 15 chunks. The
3 retrieved chunks were: frontmatter stub, conceptual intro, and the verify step. The
table config chunk (containing the full `streamConfigs` JSON with `stream.kafka.*`
keys) ranked 21+ in hybrid search — it is almost entirely JSON with no surrounding
prose, so both BM25 and dense similarity score it poorly.

**Root cause:** Two issues: (1) golden set expected source was wrong; (2) JSON-heavy
chunks without explanatory prose are retrieval-blind. The chunker splits the table
config JSON away from the "Save the realtime table config" prose section that precedes
it.

**Recommendation:** Update `expected_sources` to `basics/getting-started/first-stream-ingest.md`.
Fix chunking to keep the "Save the realtime table config" prose and its JSON example
in the same chunk.

---

## q006 — How do you reload Pinot segments after changing an index configuration?

**Scores (original):** accuracy=0, completeness=1, citations=1, total=2  
**Scores (corrected):** accuracy=1, completeness=1, citations=1, total=3  
**Retrieval:** HR@5=100%, MRR@5=0.20

**Finding:** Accuracy=0 was a scoring error. The answer cited the Swagger UI endpoint
`/help#!/Segment/reloadAllSegments`, which was read as "reload all tables". However,
the `segment-reload.md` docs confirm the endpoint is table-scoped
(`POST /segments/{tableName}/reload`). The Swagger method name is misleading but the
described behavior is correct.

**Root cause:** Scoring error, not a pipeline issue.

**Recommendation:** Score corrected to accuracy=1. No pipeline changes needed.

---

## q009 — What is the default HTTP port for the Pinot broker?

**Scores:** accuracy=0, completeness=1, citations=1, total=2  
**Retrieval:** HR@5=0% (expected source `reference/configuration-reference/broker.md`
had 2 candidates, both correctly dropped by reranker)

**Finding:** The 2 retrieved chunks from `broker.md` were config reference table rows
for `pinot.broker.client.access.protocols.http.port` and `https.port` — both with
empty default value columns. The reranker correctly dropped them (they don't answer
"what is the default"). The chunk with the answer (`queryPort = 8099`) was never
retrieved because it belongs to the deprecated property row, which uses vocabulary
("Deprecated", "Legacy") that diverges from the query's "default HTTP port".

**Root cause:** Markdown config reference tables store property name, default value,
and description in separate columns. When chunked, "8099" has no grammatical or
semantic tie to `access.protocols.http.port` in the same chunk. BM25 matches the
current property name but finds no default; the deprecated property chunk (which has
8099) uses mismatched vocabulary.

**Recommendation:** Normalize config reference table rows to prose during ingestion:
`"property X: default Y. description Z."` This co-locates the property name, default
value, and description in a single dense string that BM25 and dense embeddings can
match holistically.

---

## q013 — What is the default segment retention for a Pinot table?

**Scores:** accuracy=1, completeness=1, citations=1, total=3  
**Retrieval:** HR@5=0% (expected source `reference/configuration-reference/table.md`
never retrieved; fixed by updating expected source)

**Finding:** The expected source was wrong. `operate-pinot/troubleshooting/operations-faq.md`
was retrieved at reranked rank 1 and contains the correct answer: "By default there is
no retention set for a table in Apache Pinot." The `table.md` retention chunks
(`retentionTimeUnit`, `retentionTimeValue`) have the same table column fragmentation
problem as q009 — isolated config rows with no default value prose context.

**Root cause:** Wrong expected source + same config table fragmentation issue as q009.

**Recommendation:** Expected source updated to `operations-faq.md` (already applied).
The `table.md` table row normalization fix (same as q009) would also improve its
retrievability but is not blocking quality here.

---

## q015 — What is the difference between offline, real-time, and hybrid table types?

**Scores:** accuracy=1, completeness=1, citations=1, total=3  
**Retrieval:** HR@5=0% (expected source `basics/components/table/README.md` was
candidates rank 1 by RRF=0.6667 but dropped by reranker)

**Finding:** The expected source was the top RRF candidate — hybrid search worked
correctly. The reranker demoted both README.md chunks below rank 5, selecting prose
chunks from `architecture.md` and `time-boundary.md` instead. Quality was unaffected
(3/3) because the alternative chunks covered offline, real-time, and hybrid together.

The README.md chunk is a markdown comparison table (`| Offline | ... | Real-time |
... | Hybrid | ...`). The cross-encoder sees fragmented table rows rather than
coherent prose and scores them lower than narrative descriptions.

**Root cause:** FastEmbed cross-encoder underscores markdown comparison table chunks
relative to narrative prose for conceptual "difference between" queries. This is a
reranker model quality issue, not a chunking issue — the comparison table IS the right
format for this content and shouldn't be converted to prose.

**Recommendation:** Evaluate alternative rerankers (Cohere Rerank API, larger
cross-encoder models) to determine if a higher-quality model handles tabular content
better. Also evaluate increasing `rerank_top_k` from 5 to 7 as a low-risk mitigation:
the README chunk was candidates rank 1, so it would survive a slightly looser cutoff.

---

## Cross-Cutting Summary

| Question | Root Cause | Fix Category |
|----------|-----------|--------------|
| q003 | Example JSON split from intro prose | Chunking: keep prose+example together |
| q005 | Wrong expected source + JSON chunk retrieval-blind | Golden set label + chunking |
| q006 | Scoring error | Already corrected |
| q009 | Config table column fragmentation | Ingestion: normalize table rows to prose |
| q013 | Wrong expected source | Already corrected |
| q015 | Cross-encoder underscores table chunks | Reranker: evaluate alternatives |

### Three actionable fixes

**Normalize Tables Fix — Normalize config reference table rows to prose (q009, q013/table.md)**  
In `ingest.py`, detect markdown tables and serialize each row as:
`"<property>: default <value>. <description>"` before chunking. Affects all
config reference docs (`reference/configuration-reference/*.md`).

**Merge Chunks Fix — Keep prose+example sections in the same chunk (q003, q005)**  
In `ingest.py` chunking, avoid splitting between a prose intro section and its
immediately following code/JSON example. Apply a "no-split-before-code-block" rule:
when a section ends with a fenced code block, include it with the preceding prose
rather than as a standalone chunk.

**Alternative Rankers Fix — Evaluate alternative rerankers (q015)**  
Test Cohere Rerank API and/or a larger cross-encoder model against the golden set.
Also test `rerank_top_k=7` as a low-cost mitigation that costs no model quality.
Measure HR@5 and MRR@5 before/after.
