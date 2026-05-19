# Sommelier Eval Report

**Generated:** 2026-05-16 07:15 PDT  
**Prompt version:** 64b4182  
**Golden set:** evals/golden_set.json (25 questions)  
**Traces:** sommelier_traces.jsonl  

---

## V1 Success Criteria

| | Gate |
|-|------|
| ✅ | Sommelier avg total 2.88/3 ≥ 2.5 |
| ✅ | Sommelier accuracy 0.96 > claude 0.56 |
| ❌ | Sommelier completeness 0.92 ≤ claude 0.92 |
| ✅ | Sommelier citations 1.00 > claude 0.00 |

**Retrieval (observed — calibrating targets):**  
HR@5: 92% &nbsp;|&nbsp; RC@5: 92% &nbsp;|&nbsp; MRR@5: 0.65 &nbsp;|&nbsp; FRR@5: 92%

---

## Aggregate Quality

### Overall by System

| System | Avg Accuracy | Avg Completeness | Avg Citations | Avg Total |
|--------|-------------|-----------------|--------------|----------|
| docs_pinot_ai | 1.00 | 1.00 | 1.00 | 3.00 |
| claude (claude-sonnet-4-6) | 0.56 | 0.92 | 0.00 | 1.48 |
| sommelier (claude-haiku-4-5) | 0.96 | 0.92 | 1.00 | 2.88 |

### By Query Type

| Query Type | System | Avg Accuracy | Avg Completeness | Avg Citations | Avg Total | N |
|------------|--------|-------------|-----------------|--------------|----------|---|
| how-to | docs_pinot_ai | 1.00 | 1.00 | 1.00 | 3.00 | 8 |
| how-to | claude (claude-sonnet-4-6) | 0.38 | 0.88 | 0.00 | 1.25 | 8 |
| how-to | sommelier (claude-haiku-4-5) | 1.00 | 0.75 | 1.00 | 2.75 | 8 |
| factual | docs_pinot_ai | 1.00 | 1.00 | 1.00 | 3.00 | 5 |
| factual | claude (claude-sonnet-4-6) | 0.80 | 0.80 | 0.00 | 1.60 | 5 |
| factual | sommelier (claude-haiku-4-5) | 0.80 | 1.00 | 1.00 | 2.80 | 5 |
| conceptual | docs_pinot_ai | 1.00 | 1.00 | 1.00 | 3.00 | 4 |
| conceptual | claude (claude-sonnet-4-6) | 1.00 | 1.00 | 0.00 | 2.00 | 4 |
| conceptual | sommelier (claude-haiku-4-5) | 1.00 | 1.00 | 1.00 | 3.00 | 4 |
| comparison | docs_pinot_ai | 1.00 | 1.00 | 1.00 | 3.00 | 3 |
| comparison | claude (claude-sonnet-4-6) | 1.00 | 1.00 | 0.00 | 2.00 | 3 |
| comparison | sommelier (claude-haiku-4-5) | 1.00 | 1.00 | 1.00 | 3.00 | 3 |
| new-in-2026 | docs_pinot_ai | 1.00 | 1.00 | 1.00 | 3.00 | 5 |
| new-in-2026 | claude (claude-sonnet-4-6) | 0.00 | 1.00 | 0.00 | 1.00 | 5 |
| new-in-2026 | sommelier (claude-haiku-4-5) | 1.00 | 1.00 | 1.00 | 3.00 | 5 |

---

## Aggregate Retrieval Metrics

### Overall

| Metric | Reranked @5 | Candidates @20 |
|--------|------------|---------------|
| Hit Rate | 92% (23/25) | 100% (25/25) |
| Recall | 92% avg | 100% avg |
| MRR | 0.65 | 0.72 |
| Full Recall Rate | 92% (23/25) | — |

### By Query Type

| Query Type | HR@5 | RC@5 | MRR@5 | N |
|------------|------|------|-------|---|
| how-to | 100% | 100% | 0.65 | 8 |
| factual | 80% | 80% | 0.67 | 5 |
| conceptual | 75% | 75% | 0.56 | 4 |
| comparison | 100% | 100% | 0.33 | 3 |
| new-in-2026 | 100% | 100% | 0.90 | 5 |

---

## Per-Question Quality

| Query | System | Accuracy | Completeness | Citations | Total |
|-------|--------|----------|--------------|-----------|-------|
| How do you configure upsert on a Pinot real-time t… | docs_pinot_ai | 1 | 1 | 1 | 3 |
|  | claude (claude-sonnet-4-6) | 1 | 1 | 0 | 2 |
|  | sommelier (claude-haiku-4-5) | 1 | 1 | 1 | 3 |
| How do you enable upsert segment compaction on a P… | docs_pinot_ai | 1 | 1 | 1 | 3 |
|  | claude (claude-sonnet-4-6) | 0 | 0 | 0 | 0 |
|  | sommelier (claude-haiku-4-5) | 1 | 1 | 1 | 3 |
| How do you configure a star-tree index in Pinot? | docs_pinot_ai | 1 | 1 | 1 | 3 |
|  | claude (claude-sonnet-4-6) | 0 | 1 | 0 | 1 |
|  | sommelier (claude-haiku-4-5) | 1 | 0 | 1 | 2 |
| How do you enable an inverted index on a column in… | docs_pinot_ai | 1 | 1 | 1 | 3 |
|  | claude (claude-sonnet-4-6) | 0 | 1 | 0 | 1 |
|  | sommelier (claude-haiku-4-5) | 1 | 1 | 1 | 3 |
| How do you set up real-time ingestion from Kafka i… | docs_pinot_ai | 1 | 1 | 1 | 3 |
|  | claude (claude-sonnet-4-6) | 0 | 1 | 0 | 1 |
|  | sommelier (claude-haiku-4-5) | 1 | 0 | 1 | 2 |
| How do you reload Pinot segments after changing an… | docs_pinot_ai | 1 | 1 | 1 | 3 |
|  | claude (claude-sonnet-4-6) | 1 | 1 | 0 | 2 |
|  | sommelier (claude-haiku-4-5) | 1 | 1 | 1 | 3 |
| How do you configure a bloom filter on a Pinot tab… | docs_pinot_ai | 1 | 1 | 1 | 3 |
|  | claude (claude-sonnet-4-6) | 0 | 1 | 0 | 1 |
|  | sommelier (claude-haiku-4-5) | 1 | 1 | 1 | 3 |
| How do you apply a transformation to a column at i… | docs_pinot_ai | 1 | 1 | 1 | 3 |
|  | claude (claude-sonnet-4-6) | 1 | 1 | 0 | 2 |
|  | sommelier (claude-haiku-4-5) | 1 | 1 | 1 | 3 |
| What is the default HTTP port for the Pinot broker… | docs_pinot_ai | 1 | 1 | 1 | 3 |
|  | claude (claude-sonnet-4-6) | 1 | 0 | 0 | 1 |
|  | sommelier (claude-haiku-4-5) | 0 | 1 | 1 | 2 |
| How often does the Pinot retention manager run, an… | docs_pinot_ai | 1 | 1 | 1 | 3 |
|  | claude (claude-sonnet-4-6) | 0 | 1 | 0 | 1 |
|  | sommelier (claude-haiku-4-5) | 1 | 1 | 1 | 3 |
| What is the default port that Pinot servers listen… | docs_pinot_ai | 1 | 1 | 1 | 3 |
|  | claude (claude-sonnet-4-6) | 1 | 1 | 0 | 2 |
|  | sommelier (claude-haiku-4-5) | 1 | 1 | 1 | 3 |
| What periodic tasks does the Pinot controller run … | docs_pinot_ai | 1 | 1 | 1 | 3 |
|  | claude (claude-sonnet-4-6) | 1 | 1 | 0 | 2 |
|  | sommelier (claude-haiku-4-5) | 1 | 1 | 1 | 3 |
| What is the default segment retention for a Pinot … | docs_pinot_ai | 1 | 1 | 1 | 3 |
|  | claude (claude-sonnet-4-6) | 1 | 1 | 0 | 2 |
|  | sommelier (claude-haiku-4-5) | 1 | 1 | 1 | 3 |
| What are the roles of Apache Helix and ZooKeeper i… | docs_pinot_ai | 1 | 1 | 1 | 3 |
|  | claude (claude-sonnet-4-6) | 1 | 1 | 0 | 2 |
|  | sommelier (claude-haiku-4-5) | 1 | 1 | 1 | 3 |
| What is the difference between offline, real-time,… | docs_pinot_ai | 1 | 1 | 1 | 3 |
|  | claude (claude-sonnet-4-6) | 1 | 1 | 0 | 2 |
|  | sommelier (claude-haiku-4-5) | 1 | 1 | 1 | 3 |
| What is the difference between Pinot's single-stag… | docs_pinot_ai | 1 | 1 | 1 | 3 |
|  | claude (claude-sonnet-4-6) | 1 | 1 | 0 | 2 |
|  | sommelier (claude-haiku-4-5) | 1 | 1 | 1 | 3 |
| How does Pinot's storage model work — what are tab… | docs_pinot_ai | 1 | 1 | 1 | 3 |
|  | claude (claude-sonnet-4-6) | 1 | 1 | 0 | 2 |
|  | sommelier (claude-haiku-4-5) | 1 | 1 | 1 | 3 |
| Which Pinot index should you use for equality filt… | docs_pinot_ai | 1 | 1 | 1 | 3 |
|  | claude (claude-sonnet-4-6) | 1 | 1 | 0 | 2 |
|  | sommelier (claude-haiku-4-5) | 1 | 1 | 1 | 3 |
| What is the difference between resetting, reloadin… | docs_pinot_ai | 1 | 1 | 1 | 3 |
|  | claude (claude-sonnet-4-6) | 1 | 1 | 0 | 2 |
|  | sommelier (claude-haiku-4-5) | 1 | 1 | 1 | 3 |
| What is the difference between upsert and dedup in… | docs_pinot_ai | 1 | 1 | 1 | 3 |
|  | claude (claude-sonnet-4-6) | 1 | 1 | 0 | 2 |
|  | sommelier (claude-haiku-4-5) | 1 | 1 | 1 | 3 |
| What happened to the native text index in Pinot 1.… | docs_pinot_ai | 1 | 1 | 1 | 3 |
|  | claude (claude-sonnet-4-6) | 0 | 1 | 0 | 1 |
|  | sommelier (claude-haiku-4-5) | 1 | 1 | 1 | 3 |
| What is the availability status of Pinot's Time Se… | docs_pinot_ai | 1 | 1 | 1 | 3 |
|  | claude (claude-sonnet-4-6) | 0 | 1 | 0 | 1 |
|  | sommelier (claude-haiku-4-5) | 1 | 1 | 1 | 3 |
| What is MSE Lite Mode in Pinot, and how does its e… | docs_pinot_ai | 1 | 1 | 1 | 3 |
|  | claude (claude-sonnet-4-6) | 0 | 1 | 0 | 1 |
|  | sommelier (claude-haiku-4-5) | 1 | 1 | 1 | 3 |
| What is the minimum Java version required to build… | docs_pinot_ai | 1 | 1 | 1 | 3 |
|  | claude (claude-sonnet-4-6) | 0 | 1 | 0 | 1 |
|  | sommelier (claude-haiku-4-5) | 1 | 1 | 1 | 3 |
| How do you expand an array column into one row per… | docs_pinot_ai | 1 | 1 | 1 | 3 |
|  | claude (claude-sonnet-4-6) | 0 | 1 | 0 | 1 |
|  | sommelier (claude-haiku-4-5) | 1 | 1 | 1 | 3 |

---

## Per-Question Retrieval

| ID | Query | HR@5 | RC@5 | MRR@5 | Miss? | Expected Sources |
|----|-------|------|------|-------|-------|------------------|
| q001 | How do you configure upsert on a Pinot real-time t… | 100% | 100% | 0.33 | no | `build-with-pinot/ingestion/upsert-and-dedup/upsert.md` |
| q002 | How do you enable upsert segment compaction on a P… | 100% | 100% | 0.50 | no | `build-with-pinot/ingestion/upsert-and-dedup/segment-compaction-on-upserts.md` |
| q003 | How do you configure a star-tree index in Pinot? | 100% | 100% | 0.20 | no | `build-with-pinot/indexing/star-tree-index.md` |
| q004 | How do you enable an inverted index on a column in… | 100% | 100% | 1.00 | no | `build-with-pinot/indexing/inverted-index.md` |
| q005 | How do you set up real-time ingestion from Kafka i… | 100% | 100% | 1.00 | no | `basics/getting-started/first-stream-ingest.md` |
| q006 | How do you reload Pinot segments after changing an… | 100% | 100% | 0.20 | no | `operate-pinot/segment-reload.md` |
| q007 | How do you configure a bloom filter on a Pinot tab… | 100% | 100% | 1.00 | no | `build-with-pinot/indexing/bloom-filter.md` |
| q008 | How do you apply a transformation to a column at i… | 100% | 100% | 1.00 | no | `build-with-pinot/ingestion/ingestion-level-transformations.md` |
| q009 | What is the default HTTP port for the Pinot broker… | 0% | 0% | 0.00 | yes ⚠️ | `reference/configuration-reference/broker.md` |
| q010 | How often does the Pinot retention manager run, an… | 100% | 100% | 1.00 | no | `basics/concepts/segment-retention.md` |
| q011 | What is the default port that Pinot servers listen… | 100% | 100% | 0.33 | no | `reference/configuration-reference/server.md` |
| q012 | What periodic tasks does the Pinot controller run … | 100% | 100% | 1.00 | no | `basics/components/cluster/controller.md` |
| q013 | What is the default segment retention for a Pinot … | 100% | 100% | 1.00 | no | `operate-pinot/troubleshooting/operations-faq.md` |
| q014 | What are the roles of Apache Helix and ZooKeeper i… | 100% | 100% | 0.25 | no | `basics/architecture.md` |
| q015 | What is the difference between offline, real-time,… | 0% | 0% | 0.00 | yes ⚠️ | `basics/components/table/README.md` |
| q016 | What is the difference between Pinot's single-stag… | 100% | 100% | 1.00 | no | `build-with-pinot/querying-and-sql/sse-vs-mse.md` |
| q017 | How does Pinot's storage model work — what are tab… | 100% | 100% | 1.00 | no | `basics/concepts/pinot-storage-model.md` |
| q018 | Which Pinot index should you use for equality filt… | 100% | 100% | 0.25 | no | `build-with-pinot/indexing/choosing-indexes.md` |
| q019 | What is the difference between resetting, reloadin… | 100% | 100% | 0.50 | no | `operate-pinot/segment-lifecycle-and-repair.md` |
| q020 | What is the difference between upsert and dedup in… | 100% | 100% | 0.25 | no | `build-with-pinot/ingestion/upsert-and-dedup/dedup.md` |
| q021 | What happened to the native text index in Pinot 1.… | 100% | 100% | 1.00 | no | `build-with-pinot/indexing/native-text-index.md` |
| q022 | What is the availability status of Pinot's Time Se… | 100% | 100% | 1.00 | no | `build-with-pinot/querying-and-sql/time-series-queries.md` |
| q023 | What is MSE Lite Mode in Pinot, and how does its e… | 100% | 100% | 1.00 | no | `build-with-pinot/querying-and-sql/multi-stage-query/multistage-lite-mode.md` |
| q024 | What is the minimum Java version required to build… | 100% | 100% | 0.50 | no | `basics/getting-started/install/local.md` |
| q025 | How do you expand an array column into one row per… | 100% | 100% | 1.00 | no | `build-with-pinot/querying-and-sql/multi-stage-query/operator-types/unnest.md` |

---

## Recommendations

- **V1 gate**: sommelier completeness (0.92) does not beat claude (0.92) — consider expanding system prompt to encourage fuller answers.
