# Sommelier: Chatbot for Pinot Guidance and Support
#ai/pinot
* Project Concept Google Doc<!-- {"fold":true} -->
  * https://docs.google.com/document/d/1WWYmZzDlCIPQ3xVyJj6jjKD5AfTApjHHR8r5UG6_jaI/edit?usp=sharing
* Review notes from joseph.martin@deepatlas.ai on 5/1<!-- {"fold":true} -->
  * V1 requirements and initial questions
  - [x] Data ingestion from docs, changelogs, gh (code, issues and prs)
  - [x] Separate chunking strategies for each data source (may need sumarization for tutorials/guides)
    * Why consider different strategies?<!-- {"fold":true} -->
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
* Prioritize the todos above
  1. Design: Plan for how your users will interact with Sommelier. Will they use a custom site, Slack, MCP Server or CLI? Can they swap models?
  2. Design: Choose a vector database, chunking strategy, and embedding/inference models
  3. Build: Data ingestion from docs (save changelogs, code, issues and prs for later)
  4. Test: Sommelier should be much better than just using Claude, ChatGPT or Google’s AI. Compare against alternatives and keep track of scoring results
  5. Improve: Separate chunking strategies for each data source (may need sumarization for tutorials/guides)
* Design: Plan for how your users will interact with Sommelier. Will they use a custom site, Slack, MCP Server or CLI? Can they swap models?<!-- {"fold":true} -->
  * I envision V1 of Sommelier as a very low-cost tool. Users should be able to set it up the MCP Server and be able to configure the inference model
  * All other options (custom site, Slack, and MCP Server) treat Sommelier as a service and more expensive.
  * Since I want to minimize costs, I plan to run Sommelier locally and have it use a model hosted externally (OpenAI)
  * Should Sommelier be a CLI tool or a local MCP Server?
    * MCP Servers are designed to wrap external tools. Claude (or another coding assitant) would pass along the user’s query
    * A CLI tool may be a good first step before making the project public
- [ ] Design: Choose a vector database, chunking strategy, and embedding/inference models
  * Choose a vector database: Chroma now (Qdrant maybe later)
    * Any reason to pick ChromaDB or Weaviate over Qdrant for this use case?
      * ChatGPT Prompt: I’m trying to decide which vector store I should use for my portfolio project Sommelier (RAG based Chatbot for the Apache Pinot guidance and support). I’m leaning towards Qdrant but also considering ChromaDB and Weaviate. What should I based this decision on? I want the lowest cost option that minimizes system requirements
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
  - [ ] Choose a chunking strategy
  - [ ] Choose embedding/inference models