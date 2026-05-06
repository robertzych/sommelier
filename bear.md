# Sommelier: Chatbot for Pinot Guidance and Support
#ai/pinot
* Project Concept Google Doc
  * https://docs.google.com/document/d/1WWYmZzDlCIPQ3xVyJj6jjKD5AfTApjHHR8r5UG6_jaI/edit?usp=sharing
* Review notes from joseph.martin@deepatlas.ai on 5/1<!-- {"fold":true} -->
  * V1 requirements and initial questions
  - [x] Data ingestion from docs, changelogs, gh (code, issues and prs)
  - [x] Separate chunking strategies for each data source (may need sumarization for tutorials/guides)
    * Why consider different strategies?
      * Query optimization. Like different db indexes.
      * See the cheat sheet in the RAG handbook
        * Saved to: /Users/robertzych/dev/deep-atlas/deep_atlas_mli/agentic_systems/newsletter_expert/README.md
        * Recursive strategy for standard structured text like technical articles
          * Raptor/hierarchical style. doc level, paragraph level, then sentence level. create embedding at each level to capture the meaning at each of those levels
        * Documented-based strategy for code documentation or highly formatted manuals (python code classes)
          * Like recursive, but any strategy at the document level. Customize to the document format (python, java, etc)
          * See if LangChain has a “splitter” for java
            * [Splitting code text splitter integration guide - Docs by LangChain](https://docs.langchain.com/oss/python/integrations/splitters/code_splitter)
        * Semantic strategy for long-form prose or complex, shifting topics
        * Agentic/LLM for high-stakes documents where precision is critical
          * LLM decides the chunking strategy for different parts of the document. Good for docs that that have multiple types and styles of text (docs, slack threads, etc)
        * Hierarchical Strategy: Consider "Parent-Child" chunking, where small child chunks are used for retrieval, but the larger parent chunk is fed to the LLM for full context.
        * Structure Metadata: Always store source info (page numbers, section titles) as metadata to help the LLM cite its sources correctly.
    * Why consider summarization?<!-- {"fold":true} -->
      * Supports broader questions that spans chunks and documents 
  - [x] Choose an embedding model and a vector database
  - [x] Plan for how the users will interact with the chatbot. Will they use a custom site, Slack, MCP Server or CLI?
  - [x] Plan for what model will perform inference. Can users swap models?
  * For evals, consider scraping previous gh issues or the community Slack to build a handful of e2e scenarios
  - [x] Sommelier should be much better than just using Claude, ChatGPT or Google’s AI. Compare against alternatives and keep track of scoring results
  * After V1
    * Consider your curation strategy. How will you invalidate docs/chunks in the store?
    * Include more data sources (Slack channels, StarTree internal customer info)
    * Keep memory of individual users to inform conversations
* **Prioritize the main tasks**
  1. Design: Choose which questions you want to target in V1
  2. Design: Plan for how your users will interact with Sommelier. Will they use a custom site, Slack, MCP Server or CLI? Can they swap models?
  3. Design: Choose a vector database, chunking strategy, and embedding/inference models
  4. Build: Data ingestion from docs (save changelogs, code, issues and prs for later)
  5. Test: Sommelier should be much better than just using Claude, ChatGPT or Google’s AI. Compare against alternatives and keep track of scoring results
  6. Improve: Separate chunking strategies for each data source (may need sumarization for tutorials/guides)
* Design: Plan for how your users will interact with Sommelier. Will they use a custom site, Slack, MCP Server or CLI? Can they swap models?<!-- {"fold":true} -->
  * I envision V1 of Sommelier as a very low-cost tool. Users should be able to set it up the MCP Server and be able to configure the inference model
  * All other options (custom site, Slack, and MCP Server) treat Sommelier as a service and more expensive.
  * Since I want to minimize costs, I plan to run Sommelier locally and have it use a model hosted externally (OpenAI)
  * Should Sommelier be a CLI tool or a local MCP Server?
    * MCP Servers are designed to wrap external tools. Claude (or another coding assitant) would pass along the user’s query
    * A CLI tool may be a good first step before making the project public
- [ ] Design: Choose a vector database, chunking strategy, and embedding/inference models
  * Choose a vector database: Qdrant (see “Why Qdrant” in plan.md)<!-- {"fold":true} -->
    * Any reason to pick ChromaDB or Weaviate over Qdrant for this use case?
      * ChatGPT Prompt: I’m trying to decide which vector store I should use for my portfolio project Sommelier (RAG based Chatbot for the Apache Pinot guidance and support). I’m leaning towards Qdrant but also considering ChromaDB and Weaviate. What should I based this decision on? I want the lowest cost option that minimizes system requirements<!-- {"fold":true} -->
        * ChatGPT Response Review
          * don’t optimize for scale. optimize for simplicity, cost, and productivity 
          * an in-process/embedded option will be cheapest and simplest
          * don’t need enterprise features. just semantic search and basic metadata filtering (by doc type and source)
          * at 100k chunks performance will start to matter
          * ChromaDB is the easiest to get started with because it’s just a python import and has very low memory overhead
          * Qdrant is a self-hosted option that can handle production level workloads
          * Weaviate has built-in hybrid search and many other capabilities but is more complex and expensive
          * start with Chroma and add Qdrant if performance becomes an issue
          * design the system to allow you to swap out the vector store
      * Claude prompt: ultrathink review @bear.md help me decide which vector is appropriate for the project at this point.  Interview me in detail using AskUserQuestionTool about literally anything: technical implementation, concerns, tradeoffs, etc.
        * AskUserQuestionTool<!-- {"fold":true} -->
          1. Query types: When Pinot users query Sommelier, what kinds of questions do you expect most? This is critical — Apache Pinot has very specific technical vocabulary (StarTree indexes, minion tasks, upsert tables, broker queries) that pure semantic search can miss.
             * I want to be able support as many types of questions as possible including all of the above and more. But as a starter goal I'm looking to support beginner level questions such as "How can you start an Apache Pinot cluster in under ten minutes?"
          2. Deployment scope: What is the long-term deployment vision for Sommelier? This affects whether an embedded (in-process) or server-based vector store makes more sense.
             * I want to share it with the community but I don't want to pay for hosting. I'm planning to create a public github repo that would include instructions on how to run it as a local MCP server and how to use it within their coding assistants (Claude).
          3. Data volume: How much content do you realistically plan to index in V1? At ~100k chunks, ChromaDB performance starts to degrade noticeably.
             * I just want index the docs for now. But I want to add other sources too. For example, GitHub issues, PRs, and even the Java code from the main repo.
          4. Store persistence: How should the vector store be managed between runs? This determines how much operational complexity you're willing to take on.
             * Persist, incremental: Store embeddings permanently, only re-embed changed/new documents. Requires deduplication and deletion logic — more complex but efficient at scale.
          5. Setup complexity: What's the acceptable setup complexity for someone cloning the public repo and running Sommelier locally? This directly determines whether an embedded DB (ChromaDB, pure Python) or a server-based DB (Qdrant, requires Docker) is viable.
             * Either (pure python or docker) is fine: You don't have a strong preference — whatever makes the technical design cleanest.
          6. Embedding model: Where will embeddings be generated? This affects cost, offline capability, and quality — and is independent of which vector store you pick.
             * Let users configure it: Abstract the embedding provider so users can plug in OpenAI, Cohere, or a local model. More upfront design work but aligns with the ‘swappable models’ goal.
          7. Hybrid search: Hybrid search combines semantic similarity with keyword matching (BM25). It significantly improves recall for exact technical terms like 'StarTree index' or 'minion task heap size'. How do you want to handle this?
             * Build it in from day 1: Implement BM25 + semantic fusion from the start. Adds complexity but ensures Sommelier outperforms plain ChatGPT on precise technical queries.
          8. Change detection: When you re-run the ingestion pipeline to pick up new Pinot docs or GitHub issues, how should it detect what has changed and needs re-embedding?
             * Content hash: Hash each document’s content and skip re-embedding if the hash matches what’s already stored. Simple and reliable.
          9. RAG framework: Do you want to use LangChain or LlamaIndex to wire together the retriever, chunker, and LLM? Or build the RAG pipeline from raw SDKs (openai, chromadb/qdrant-client, etc.)?
             * LangChain (Recommended): Higher-level abstractions for chains, retrievers, and vector stores. Faster to build, lots of integrations, but adds a layer you don't fully control. Well-supported by Pinot community examples.
             * Not sure yet: Want to understand the tradeoffs before deciding
          10. Local embeddings: Qdrant's Python client ships with FastEmbed — a lightweight library that runs embedding models locally (CPU, no API key). This could be the default for users who don't want to configure OpenAI. Is this appealing?
              * Yes — great default (Recommended): Users who clone the repo get working embeddings with zero API keys. FastEmbed's all-MiniLM-L6-v2 is decent quality for docs. They can opt into OpenAI for better results.
          11. Pinot versioning: Apache Pinot has major versions (0.12, 1.0, 1.2). A user asking about 0.12 behavior might get wrong answers if the index mixes docs from different versions. How important is version-aware retrieval?
              * Track version as metadata. Store Pinot version as a metadata field on every chunk. During initial setup the user can re-configure to their version. But by default the configuration will be set to the latest version.
          12. Score fusion: Hybrid search merges a semantic similarity score with a keyword (BM25) score. How should Sommelier combine them?
              * Reciprocal Rank Fusion (Recommended): RRF: combine ranked result lists rather than raw scores. Robust, well-studied, and doesn’t require tuning weights. Works great out of the box.
        * Review Claude’s plan (`/Users/robertzych/.claude/plans/ultrathink-review-bear-md-help-vivid-wadler.md`)<!-- {"fold":true} -->
          * Hybrid search (BM25 + semantic) is required from day 1 — Apache Pinot has dense technical vocabulary and users range from beginners to power users querying exact config params.
          * FastEmbed as default embedding — users who clone the repo should get working embeddings with zero API keys required.
          * Incremental updates via content hash — the vector store persists between runs and re-embeds only changed documents.
          * These three constraints together rule out ChromaDB. ChromaDB has no native hybrid search; implementing it would require a separate BM25 index (e.g., rank_bm25), custom score fusion, and two indexes to keep in sync on incremental updates — adding more complexity than just using Qdrant directly.
          * Recommendation: Qdrant (local embedded mode). Qdrant's local embedded mode (QdrantClient(path="./qdrant_storage")) stores data in a local directory using Rust-backed storage — no Docker, no server, just pip install qdrant-client[fastembed]. Users who want Docker can use it identically; the same client API works against both.
          * crawler.py implies the docs will be source via a web crawler which isn’t required for docs because they are locally accessible through a github repo.
          * chunk payload contains `chunk_index` which implies a simple positional chunking strategy which is premature as the appropriate chunking strategies for each type of document hasn’t been decided yet
          * what does the `NAMESPACE_URL` parameter represent?
          * what are dense and sparse embeddings? when should each be used?
            * dense embeddings compress the space so that similar concepts end close together. they’re good for conceptual questions, paraphrased questions, and semantic similarity
            * sparse embeddings use non-zero values on when there’s a word match (BM25 is the classic example: it scores terms by frequency and rarity (TF-IDF style), producing a sparse vector where each dimension represents a vocabulary token.). they’re good for exact keyword matches (“StarTree index heap size”)
          * LlamaIndex has first-class Qdrant support (QdrantVectorStore), handles chunking pipelines, and is purpose-built for this use case. Less boilerplate than raw SDKs for the retrieval-augmented generation loop.
          * Raw SDKs (openai + qdrant-client directly) would show deeper RAG understanding in a portfolio context, but require more code. Worth revisiting once the pipeline design is clearer.
  * Choose a chunking strategy (header-aware, then recursive on any section that exceed 512 tokens)
    * Claude prompt: ultrathink review @bear.md and @plan.md help me decide which chunking strategies are appropriate for the project.  Interview me in detail using AskUserQuestionTool about literally anything: technical implementation, concerns, tradeoffs, etc.
      * AskUserQuestionTool<!-- {"fold":true} -->
        1. Docs format: What format are the Apache Pinot docs stored in within the GitHub repo? This determines whether a markdown-aware, RST-aware, or generic recursive splitter is the right starting point.
           * Markdown (GitBook): Standard .md files with headers (#, ##, ###), code blocks (```), tables, and possibly GitBook-specific directives like {% hint %} or {% tabs %}.
        2. Chunk hierarchy: Parent-child (hierarchical) chunking retrieves small, precise child chunks but passes the larger parent chunk to the LLM for full context. This improves both retrieval precision and answer quality, at the cost of more complex ingestion logic. Do you want this?
           * I’m not sure yet. I would to compare the recursive, raptor, and semantic chunking strategies more before making a decision on this.
        3. Java granularity: For the Apache Pinot Java source code, at what level should chunks be defined? This affects how precisely code-related queries can be answered.
           * I want to focus on the docs for now. But later I want to consider using LangChain’s [RecursiveCharacterTextSplitter](https://reference.langchain.com/python/langchain-text-splitters/character/RecursiveCharacterTextSplitter?_gl=1*l88me4*_gcl_au*MjAzMjYyNDUyOS4xNzc3MjQxNjcw*_ga*OTk5NzgwMDAwLjE3NzcyNDE2NzE.*_ga_47WX3HKKY2*czE3Nzc5MzY0ODAkbzUkZzAkdDE3Nzc5MzY0ODAkajYwJGwwJGgw) as it supports Java
        4. Issue structure: GitHub issues often have a body plus multiple comments — some comments are noise ('+1', 'me too') and some are the actual solution. How should issues and their comments be handled?
           * Skip issues in V1: Index only docs first. Add GitHub issues once the pipeline is working end-to-end.
        5. Query scope: RAPTOR builds a tree of LLM-generated summaries and excels at broad, multi-document queries. Recursive/semantic chunking is better for specific how-to lookups. What mix of query types do you expect in V1?
           * Mix of specific and broad queries: Some specific how-tos AND some broad questions like "What are all the ingestion methods in Pinot?" or "Summarize the differences between table types." RAPTOR shines here.
        6. Ingestion cost: RAPTOR requires LLM API calls to generate summaries at each tree level during ingestion — potentially hundreds of calls for the full Pinot docs corpus. Recursive and semantic chunking need only embeddings (much cheaper). Is LLM-based ingestion acceptable?
           * Yes — worth it for quality (Recommended): Willing to spend $1–5 in LLM API calls to build the index once if it meaningfully improves answer quality.
        7. GitBook syntax: GitBook markdown uses special directives that standard splitters don't understand: {% hint %}, {% tabs %}, {% content-ref %}, callout blocks, etc. How important is handling these correctly?
           * Important — strip or parse them: GitBook syntax should be stripped before chunking so it doesn't pollute chunks with raw {% %} tokens. Worth writing a pre-processing step.
        8. Chunk size: What chunk size (in tokens) should retrieval chunks target? Smaller chunks = more precise retrieval but less context per chunk. Larger chunks = richer context but noisier retrieval signal.
           * Medium — ~512 tokens (Recommended): Good balance. A few paragraphs per chunk. Works well flat (no parent-child needed) and aligns with most embedding model training distributions.
        9. RAPTOR depth: Full RAPTOR builds multiple levels of LLM summaries (leaf chunks → cluster summaries → summaries of summaries → root). A simpler variant generates just one summary per doc page, stored alongside leaf chunks. Both handle broad queries; full RAPTOR handles deeper cross-document synthesis.
           * Start recursive, add summaries later: Ship V1 with pure recursive chunking. Add the summary layer in V1.1 once you've validated the retrieval pipeline works.
        10. Markdown splitting: For GitBook markdown docs, two strategies exist for the base chunk split. Which fits better?
            * Header-aware splitting (Recommended): Split on markdown headers (#, ##, ###) first, then recursively split oversized sections to the 512-token target. Keeps each chunk within a single named section. Section title stored as metadata. LangChain's MarkdownHeaderTextSplitter or LlamaIndex's MarkdownNodeParser.
  - [ ] Design: Choose embedding/inference models
    - [ ] Claude prompt: 
  - [ ] Design: Choose which questions/answers you want to target in V1
    * Still to questions that can be answered only from the docs for now
    * How to get questions?
      * scrape generated questions/answers from docs.pinot.apache.org
      * have another LLM generate questions from the docs
      * scrape real questions from slack
      * write my own questions
    - [ ] How to get answers?
      * scrape generated answers from docs.pinot.apache.org
      * have another LLM generate answers 
      * write my own answers
    - [ ] How will I evaluate answer quality?
    - [ ] How many questions do I need? 20?
    - [ ] How will I categorize the questions?
    - [ ] 