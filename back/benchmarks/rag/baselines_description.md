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

### Active B6 Pipeline:
1. **Full Graph-RAG**: Adaptive graph traversal (`graph_expansion` via `ExperimentalGraphExpander` and LLM-filtering) is enabled.
2. **Neighbor Extraction**: It retrieves neighbor documents to depth `b6_graph_neighbors_order` (default 2) for all primary vector/lexical search hits (`graph_neighbors_in_rrf: True`). Chunks of neighbor articles are fetched, ranked via cosine similarity, and added to the candidate pool before RRF and Cross-Encoder reranking.
3. **Query Expansion**: Active synonym expansion (`llm_query_expansion` + `graph_ontology_lookup`).
4. **Post-Processing & Blending**: 
   * `dynamic_alpha_blending` is active (calibrates dense/lexical search balance dynamically).
   * `score_blending` is active (blends Reranker and RRF scores).
   * `citation_repair` is active (validates and repairs citation links in generated responses).

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
