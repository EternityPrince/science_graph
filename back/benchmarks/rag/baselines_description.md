# RAG Baselines Description (B0 - B6 + CUSTOM)

This document provides a detailed technical description of the 7 baselines (**B0** – **B6**) and the custom configuration (**CUSTOM**) implemented in the Science Graph RAG pipeline.

---

## 1. Fixed and Dynamic Baselines

All baselines are categorized into two groups depending on how they are configured:
1. **Fixed Baselines (B0 - B5)**: Their set of enabled modules is hardcoded in [core/config.py](file:///Users/vladimirkasterin/python/science_graph/back/benchmarks/rag/core/config.py) and does not depend on settings in the user's configuration file.
2. **Dynamic Baselines (B6 and CUSTOM)**: They inherit component states directly from the application's configuration file [config.yaml](file:///Users/vladimirkasterin/.config/pdf-graph-analyzer/config.yaml) (except for the `hyde` module, which is forced to `false` for both, and B6 forces `reranker` and `graph_neighbors_in_rrf` to `true`).
   * *Note for CUSTOM*: Running the benchmark with the `--custom` flag overrides the CUSTOM configuration with a predefined set of parameters (preset) from [config_creator.py](file:///Users/vladimirkasterin/python/science_graph/back/benchmarks/rag/config_creator.py).

---

## 2. RAG Component Matrix (Theoretical vs Actual Default)

The table below compares the theoretical (intended default) state of components and their **actual** state for each baseline, based on the default configuration in `config.yaml`. 

> [!NOTE]
> By default, the configuration file `config.yaml` has all key graph and post-processing features **enabled**. 
> Thus, the active **B6 (Full Pipeline)** baseline runs in its full capacity as designed. If the user manually disables components in `config.yaml`, B6 will dynamically inherit those disabled states (except for `reranker` and `graph_neighbors_in_rrf`, which are always forced ON).

| Component Name | B0 | B1 | B2 | B3 | B4 | B5 | B6 (Theory & Default config.yaml) | CUSTOM (Default config.yaml)* |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`intent_classifier`** | OFF | OFF | OFF | OFF | OFF | OFF | OFF | **OFF** |
| **`graph_ontology_lookup`** | OFF | OFF | OFF | OFF | OFF | OFF | ON | **ON** |
| **`llm_query_expansion`** | OFF | OFF | OFF | OFF | OFF | OFF | ON | **ON** |
| **`hyde`** | OFF | OFF | OFF | ON | OFF | OFF | OFF *(Forced)* | **OFF** *(Forced)* |
| **`lexical_search`** | OFF | ON | OFF | OFF | ON | ON | ON | **ON** |
| **`dense_search`** | OFF | OFF | ON | ON | ON | ON | ON | **ON** |
| **`dynamic_alpha_blending`**| OFF | OFF | OFF | OFF | ON | ON | ON | **ON** |
| **`rrf`** | OFF | OFF | OFF | OFF | ON | ON | ON | **ON** |
| **`graph_neighbors_in_rrf`**| OFF | OFF | OFF | OFF | OFF | OFF | ON *(Forced)* | **OFF** |
| **`graph_expansion`** | OFF | OFF | OFF | OFF | OFF | ON (Static) | ON (Adaptive) | **ON** |
| **`reranker`** | OFF | OFF | OFF | OFF | ON | ON | ON *(Forced)* | **ON** |
| **`score_blending`** | OFF | OFF | OFF | OFF | OFF | OFF | ON | **ON** |
| **`context_trimming`** | OFF | OFF | OFF | OFF | OFF | ON | ON | **ON** |
| **`citation_repair`** | OFF | OFF | OFF | OFF | OFF | ON | ON | **ON** |

*\*Note: When running with the `--custom` flag in the CLI, fixed preset values are applied to CUSTOM (including `citation_repair: True` and disabling `graph_expansion` and `dynamic_alpha_blending`). Without this flag, CUSTOM completely matches the active config.yaml.*

---

## 3. B6 in the Current Setup

In the default state of the workspace (with no custom overrides in `config.yaml`), **B6** runs as a fully featured pipeline with all components enabled. 

Based on the active configuration (`config.yaml`), the components in the **B6** pipeline are configured as follows:

### Current Configured Pipeline Components for B6:
* **Active Components (🟢):**
  * `dense_search`: Semantic vector search.
  * `lexical_search`: Keyword search using SQLite FTS5.
  * `rrf`: Reciprocal Rank Fusion for candidate merging.
  * `reranker`: Cross-Encoder reranking (forced to `True` for B6).
  * `score_blending`: Combining Reranker and RRF scores.
  * `graph_neighbors_in_rrf`: Retrieval of neighboring node chunks (forced to `True` for B6).
  * `graph_ontology_lookup`: Synonyms/concept mapping using the graph ontology.
  * `context_trimming`: Fitting the final prompt into the context window.

* **Inactive Components (🔴):**
  * `graph_expansion`: Adaptive graph traversal via `ExperimentalGraphExpander` is **disabled**.
  * `dynamic_alpha_blending`: Dense/lexical weight balancing is **disabled**.
  * `citation_repair`: Citation validation and repair is **disabled**.
  * `hyde`: Hypothetical document embedding is **disabled** (forced to `False` for B6).
  * `llm_query_expansion`: LLM-based query expansion is **disabled**.
  * `intent_classifier`: Query intent routing is **disabled**.
  * `graph_bridge_retrieval`: Deterministic bridge retrieval is **disabled**.
  * `graph_concept_retrieval`: Deterministic concept retrieval is **disabled**.
  * `graph_selected_sources_card`: Response selected sources card is **disabled**.
  * `graph_retrieval_trace`: Tracing to JSONL is **disabled**.

### Active LLM and RAG Parameters:
* **LLM Provider**: `mlx` (Local execution)
* **LLM Model**: `/Users/vladimirkasterin/models/llm/OCC-RAG-1.7B`
* **Model Max Context / Synthesis Input Limit**: `12,000` / `9,500` tokens
* **Reranker Model**: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
* **NER Model**: `/Users/vladimirkasterin/models/ner/wikineural-multilingual-ner`
* **SpaCy Model**: `/Users/vladimirkasterin/models/lemmatization`

---

## 4. Detailed Description of Baselines

### B0: Zero-Shot (Pure Generation)
* **Purpose**: Evaluate the baseline knowledge of the LLM without providing context.
* **How it works**: Retrieval is completely bypassed. The query is sent directly to the LLM with instructions to answer based solely on pre-trained knowledge.

### B1: Pure Lexical (Keyword Search)
* **Purpose**: Simple keyword-based retrieval.
* **How it works**: Uses SQLite FTS5 to search for the most relevant text chunks. No semantic analysis is performed.

### B2: Pure Dense (Vector Search)
* **Purpose**: Classic semantic vector search.
* **How it works**: The query is encoded with an embedding model, and the top-k most similar chunks are retrieved from the vector database by cosine similarity.

### B3: Dense + HyDE
* **Purpose**: Semantic search enhanced by a hypothetical document.
* **How it works**: The LLM first generates a hypothetical response to the query. This text is embedded and used to query the vector database, which helps bridge the terminology gap.

### B4: Standard Hybrid + Reranker
* **Purpose**: Fusion of vector and keyword searches with subsequent Cross-Encoder reranking.
* **How it works**: Runs `dense_search` and `lexical_search` in parallel. Merges results using Reciprocal Rank Fusion (RRF). Lexical search weights are dynamically calibrated via `dynamic_alpha_blending` based on keyword match strength. Chunks are then reranked using a Cross-Encoder reranker (`reranker: True`) before constructing the context.

### B5: Hybrid + Graph + Reranker (Graph-RAG with Reranker)
* **Purpose**: Hybrid search with static first-order relationships and chunk reranking.
* **How it works**:
  1. Performs chunk retrieval, RRF merging, and reranking with a Cross-Encoder (same as **B4**).
  2. Extracts document IDs (`paper_id`) from the retrieved chunks.
  3. Queries the graph database for **only direct neighbors (depth = 1)** of these documents (authors, cited papers, mentioned concepts).
  4. Formats these relations as text and statically appends them to the prompt. No deep graph traversal or LLM-filtering of facts is performed.

### B6: Full Pipeline
* **Purpose**: The most complete RAG pipeline utilizing graph neighbor expansion, adaptive graph crawling, and multi-stage filtering.
* **How it works**:
  1. **Query Processing**: Synonym expansion (`llm_query_expansion` + `graph_ontology_lookup`).
  2. **Hybrid Retrieval + Graph Neighbors**: Retrieves candidates (`dense_search` + `lexical_search`), finds neighbor articles at depth `b6_graph_neighbors_order` (default 2), loads their chunks, computes cosine similarity (`graph_neighbors_in_rrf`), merges via RRF, reranks (`reranker`), and performs Score Blending.
  3. **Adaptive Graph Expansion**: Runs `ExperimentalGraphExpander` for intelligent graph crawling with geometric decay (crawling limit is `b6_graph_neighbors_order + 1`), Cross-Encoder evaluation of neighbor relevance, and fact-filtering via LLM.
  4. **Post-Processing**: Citation repair and context trimming.

### CUSTOM: Custom Run
* **Purpose**: Manual calibration of hyperparameters and testing the impact of individual components.
* **How it works (by default)**: Matches the active configuration file `config.yaml` (with `graph_neighbors_in_rrf` set to `False` by default).
* **How it works with `--custom` flag**:
  Forces `citation_repair: True`, disables `dynamic_alpha_blending` and `graph_expansion`. It also applies the following hyperparameter changes:
  * `score_blend_reranker_weight` = `0.75` (instead of `0.7`)
  * `score_blend_rrf_weight` = `0.25` (instead of `0.3`)
  * `graph.p_base` = `0.0` and `graph.gamma` = `0.0` (effectively blocks graph traversal).
  * Align graph edge heuristic weights to `1.0`.

---

## 5. Deterministic Graph Retrieval Extension

The pipeline includes a **Deterministic Graph Retrieval** extension designed to expand candidate generation using the structure of the knowledge graph without invoking LLM calls. It consists of the following components:

1. **Query Concept Extraction**:
   * Uses the graph ontology to map terms in the query to canonical concept IDs using exact names, aliases, or lemmatized forms (via spaCy).

2. **GraphConceptRetriever**:
   * Identifies candidate papers that mention the query concepts.
   * Ranks them using a deterministic sorting tuple: `(-matched_concepts_count, -concept_idf_sum, paper_id)`.
   * Limits candidates to `graph_retrieval_max_graph_candidate_papers`.

3. **GraphBridgeRetriever**:
   * Finds papers that form semantic bridges between seed papers (from base retrieval) and query concepts/other seed papers.
   * Leverages three distinct bridge conditions:
     * `seed_shared_query_concept`: Candidate shares a query concept with a seed paper.
     * `seed_citation_neighbor_with_query_concept`: Candidate is a citation neighbor of a seed paper and mentions a query concept.
     * `seed_shared_concept`: Candidate bridges two distinct seed papers through a shared concept.
   * Ranks them using a deterministic sorting tuple: `(-len(covered_query_concepts), -len(connected_seed_papers), min_graph_distance, -concept_idf_sum, paper_id)`.

4. **Scoped Chunk Retrieval**:
   * For the selected graph candidate papers, retrieves their best chunks based on similarity search matching the query (limited to `graph_retrieval_chunks_per_graph_paper` per paper).

5. **Deduplication and Metadata Merge**:
   * Merges duplicate chunks retrieved via dense/lexical/graph searches, preserving all source metadata in `retrieval_sources` and establishing a strict priority order.

6. **Graph Selected Sources Card**:
   * Post-selection explanation card appended to the LLM response showing links and citations between selected papers.

7. **Graph Retrieval Trace**:
   * Logs query diagnostics to `graph_retrieval_trace.jsonl` containing statistics like before/after rerank candidate counts and graph survival rate.

