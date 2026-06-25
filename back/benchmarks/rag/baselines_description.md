# RAG Benchmarking Baselines (B0 - B6 + CUSTOM)

This document provides a precise, detailed technical specification of the 7 benchmarking baselines (**B0** to **B6**) and the **CUSTOM** configuration implemented in the Science Graph RAG pipeline.

---

## 1. RAG Component Matrix

Below is a matrix showing which modules are enabled (**ON**) or disabled (**OFF**) for each baseline configuration:

| Component Name | B0 (Zero-Shot) | B1 (Lexical) | B2 (Dense) | B3 (HyDE) | B4 (Hybrid) | B5 (Static Graph) | B6 (Full Pipeline) | CUSTOM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`intent_classifier`** | OFF | OFF | OFF | OFF | OFF | OFF | OFF | OFF |
| **`graph_ontology_lookup`** | OFF | OFF | OFF | OFF | OFF | OFF | **ON** | **ON** |
| **`llm_query_expansion`** | OFF | OFF | OFF | OFF | OFF | OFF | **ON** | OFF |
| **`hyde`** | OFF | OFF | OFF | **ON** | OFF | OFF | OFF | OFF |
| **`lexical_search`** | OFF | **ON** | OFF | OFF | **ON** | **ON** | **ON** | **ON** |
| **`dense_search`** | OFF | OFF | **ON** | **ON** | **ON** | **ON** | **ON** | **ON** |
| **`dynamic_alpha_blending`**| OFF | OFF | OFF | OFF | **ON** | **ON** | **ON** | OFF |
| **`rrf`** (Reciprocal Rank Fusion) | OFF | OFF | OFF | OFF | **ON** | **ON** | **ON** | **ON** |
| **`graph_expansion`** | OFF | OFF | OFF | OFF | OFF | **ON (Static 1-Hop)** | **ON (Adaptive Crawl)** | OFF |
| **`reranker`** (Cross-Encoder) | OFF | OFF | OFF | OFF | OFF | OFF | **ON** | **ON** |
| **`score_blending`** | OFF | OFF | OFF | OFF | OFF | OFF | **ON** | OFF |
| **`context_trimming`** | OFF | OFF | OFF | OFF | OFF | **ON** | **ON** | **ON** |
| **`citation_repair`** | OFF | OFF | OFF | OFF | OFF | **ON** | **ON** | **ON** |

---

## 2. Baseline Descriptions

### B0: Zero-Shot (Base Knowledge Evaluation)
*   **Purpose**: Evaluation of the LLM's raw pre-trained knowledge without any external context.
*   **How it works**: Bypasses the retrieval stage entirely. The query is sent straight to the LLM with a instruction to answer based solely on its own general knowledge.
*   **Active Modules**: None.

### B1: Pure Lexical (Keyword Search)
*   **Purpose**: Simple keyword matching baseline.
*   **How it works**: Uses SQLite FTS5 (Full-Text Search) to retrieve the top text chunks matching the query. No semantic understanding.
*   **Active Modules**: `lexical_search`.

### B2: Pure Dense (Standard Vector RAG)
*   **Purpose**: Classic semantic search baseline.
*   **How it works**: Encodes the query into an embedding vector and retrieves the top-k chunks using cosine similarity from the vector database.
*   **Active Modules**: `dense_search`.

### B3: Dense + HyDE (Hypothetical Document Embeddings)
*   **Purpose**: Semantic search enhanced by hypothetical answers.
*   **How it works**: First, the LLM generates a hypothetical answer (HyDE) based on the user's query. This generated response is then embedded and used to query the vector database, bridging the vocabulary gap.
*   **Active Modules**: `dense_search`, `hyde` (with `hyde_enabled=True`).

### B4: Standard Hybrid (Lexical + Dense Fusion)
*   **Purpose**: Traditional non-graph retrieval baseline.
*   **How it works**: Runs both `dense_search` and `lexical_search` concurrently. Merges the retrieved chunks using Reciprocal Rank Fusion (RRF). Weighs FTS5 results dynamically through `dynamic_alpha_blending` based on keyword match strength.
*   **Active Modules**: `dense_search`, `lexical_search`, `rrf`, `dynamic_alpha_blending`.

### B5: Hybrid + Graph (Static Graph-RAG)
*   **Purpose**: Hybrid search with static 1-hop bibliographic context.
*   **How it works**:
    1.  Retrieves top text chunks using the **B4 Standard Hybrid** pipeline.
    2.  Identifies all `paper_id`s present in those chunks.
    3.  Queries the graph database for all **immediate neighbors (depth=1)** of those papers (e.g., authors, cited papers, mentioned concepts).
    4.  Formats these raw relationships as text lines and appends them statically to the prompt in the `### KNOWLEDGE GRAPH CONNECTIONS:` section.
    5.  Does **not** crawl deeper, does **not** fetch texts of neighboring papers, and does **not** filter relationships using an LLM.
*   **Active Modules**: `dense_search`, `lexical_search`, `rrf`, `dynamic_alpha_blending`, `graph_expansion` (Static neighbor mapping), `context_trimming`, `citation_repair`.
*   **Key configuration detail**: `self.expander` is set to `None`, forcing `RAGService.ask` to bypass the advanced graph crawl.

### B6: Full Pipeline (Advanced Graph-RAG)
*   **Purpose**: The maximum capability configuration of the Science Graph RAG pipeline.
*   **How it works**:
    1.  **Query Processing**: Bypasses the `intent_classifier` (disabled in code by default), but uses `llm_query_expansion` / `graph_ontology_lookup` to expand the query with synonyms.
    2.  **Hybrid Retrieval**: Retrieves candidates using dense and lexical search, scores them with `CrossEncoder` (`reranker`), and blends the scores using `score_blending`.
    3.  **Adaptive Graph Expansion**: Instantiates `ExperimentalGraphExpander` to perform an intelligent crawl:
        *   **Crawl with Geometric Decay**: Expands depth-first up to `limit` hops, stopping early if neighbors count decays ($K_n < 1.0$) to avoid combinatoric explosion.
        *   **Summary-First Evaluation**: Fetches abstract/summary cards of neighboring nodes and ranks them against the query using the Cross-Encoder.
        *   **Chunk Ingestion**: Ingests new text chunks for the highly relevant papers discovered during the crawl.
        *   **LLM Fact Filtering (Evidence List)**: Compiles all gathered chunks and graph connections, and prompts the LLM to filter out noise, leaving only *essential* facts (`is_essential: true`).
    4.  **Generation & Validation**: Feeds the unified `enrichment_block` to the generation prompt and cleans up citation indices using `citation_repair`.
*   **Active Modules**: All components enabled, except `hyde` and `intent_classifier` (which are disabled to isolate performance gains from the graph search and prevent intent classification overhead).
*   **Key configuration detail**: `self.expander` is instantiated as `ExperimentalGraphExpander`.

### CUSTOM: Custom Run Configuration
*   **Purpose**: User-customized pipeline configuration to optimize hyperparameters and evaluate alternative component selections.
*   **How it works**:
    1.  **Query Processing**: Skips LLM Query Expansion and Intent Classifier, but keeps `graph_ontology_lookup` enabled for short concepts.
    2.  **Hybrid Retrieval**: Retrieves candidate chunks using dense search and lexical search. Dynamic Alpha Blending is disabled, which locks the FTS5 weight at 1.0.
    3.  **Reranking**: Pulls the top 10 candidates from Stage 3 and rerank them with the Cross-Encoder. Bypasses Score Blending, ranking candidates strictly by the raw Cross-Encoder reranker score to select the top 5 chunks.
    4.  **Context Construction & Post-Processing**: Bypasses both Static and Adaptive Graph Expansion (graph crawl is completely deactivated due to `p_base=0.0` and `gamma=0.0` inside `GraphPreset`). Applies `context_trimming` and `citation_repair` on the top 5 chunks.
*   **Active Modules**: `graph_ontology_lookup`, `lexical_search`, `dense_search`, `rrf`, `reranker`, `context_trimming`, `citation_repair`.
*   **Key configuration details**:
    *   **Disabled Modules**: `intent_classifier=False`, `llm_query_expansion=False`, `dynamic_alpha_blending=False`, `graph_expansion=False`, `hyde=False`, `score_blending=False`.
    *   **Custom Hyperparameters**:
        *   `rag.score_blend_reranker_weight` = `0.75` (vs `0.7` default)
        *   `rag.score_blend_rrf_weight` = `0.25` (vs `0.3` default)
        *   `rag.dynamic_alpha_threshold_low` = `1.2` (vs `1.0` default)
        *   `rag.dynamic_alpha_val_low` = `1.0` (vs `0.2` default)
        *   `graph.p_base` = `0.0` (vs `0.75` default)
        *   `graph.gamma` = `0.0` (vs `0.5` default)
        *   `graph.semantic_score_threshold` = `0.35` (vs `0.4` default)
        *   Heuristic Graph edge weights set to 1.0 (e.g. `weight_authored=1.0`, `weight_cites=1.0`, `weight_mentions_concept=1.0`, `weight_default=1.0`).
