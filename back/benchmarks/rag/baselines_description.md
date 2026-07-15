# RAG Baseline Pipelines & Component Architecture Guide

## Executive Summary

This document provides a comprehensive, actualized architectural reference and static analysis of the RAG baseline pipelines (**B0** through **B6** plus **CUSTOM**) in the Science Graph project. It details the precise component resolution logic in [core/config.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/config.py), the 19-stage pipeline execution inside [rag_service.py](file:///Users/vladimirkasterin/python/graph/back/src/services/rag_service.py), and the exact behavioral differences under the active user runtime configuration snapshot.

Key takeaways under the active runtime configuration:

* **Complete Baseline Suite (B0–B6)**: The evaluation framework defines 8 distinct pipeline configurations ranging from pure zero-shot generation (**B0**) to graph-enriched full-pipeline search (**B6**).
* **Isolation of Fixed Baselines (B0–B5)**: Baselines **B0** through **B5** strictly force their core retrieval, fusion, reranking, and graph flags in [core/config.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/config.py#L22-L48), isolating them from global configuration drift (except for `shannon_estimator_enabled`, which is inherited globally).
* **B6 Dynamic Inheritance**: Baseline **B6 (Full Pipeline)** forces `hyde = False`, `reranker = True`, and `graph_neighbors_in_rrf = True`. Unlike B4/B5, B6 dynamically inherits all remaining component toggles directly from the user's active `rag_components` snapshot.
* **Impact of Active User Config Snapshot**: With the user's current configuration (`graph_expansion: true`, `score_blending: true`, `graph_neighbors_in_rrf: true`, `dynamic_alpha_blending: false`, `citation_repair: false`):
  * **B4** runs standard Lexical + Dense RRF with Cross-Encoder reranking, with `graph_expansion` explicitly turned OFF (`context_graph` set to disabled).
  * **B5** enables `graph_expansion`, `context_trimming`, and `citation_repair` on top of B4, adding topological graph relations to prompt context.
  * **B6** combines **B5's prompt-level graph expansion** (`graph_expansion: true`) with **2-hop neighbor candidate expansion in RRF** (`graph_neighbors_in_rrf: true`) and **blended reranker scoring** (`score_blending: true`), achieving maximum candidate recall and context precision while retaining structural prompt context.

---

## 1. Pipeline Baseline Definitions (B0–B6 & CUSTOM)

Baselines are instantiated via `get_baseline_config()` in [core/config.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/config.py#L17-L70):

```
                       ┌───────────────────────────────────────┐
                       │   Global Snapshot / CLI Overrides     │
                       │        (config.rag_components)        │
                       └───────────────────┬───────────────────┘
                                           │
         ┌─────────────────────────────────┼─────────────────────────────────┐
         ▼                                 ▼                                 ▼
┌──────────────────┐             ┌──────────────────┐             ┌──────────────────┐
│  Fixed Baselines │             │   Baseline B6    │             │  CUSTOM Baseline │
│    (B0 - B5)     │             │ (Full Pipeline)  │             │ (Preset/CLI Run) │
├──────────────────┤             ├──────────────────┤             ├──────────────────┤
│ All component    │             │ Forces:          │             │ Loads preset     │
│ flags explicitly │             │ • hyde = False   │             │ overrides or user│
│ locked; ignores  │             │ • reranker = True│             │ CLI parameters;  │
│ user snapshot    │             │ • graph_neighbors│             │ restores defaults│
│ (except Shannon) │             │   _in_rrf = True │             │ after evaluation │
│                  │             │ Inherits remaining│             │                  │
│                  │             │ from snapshot    │             │                  │
└──────────────────┘             └──────────────────┘             └──────────────────┘
```

### Complete Baseline Reference

| Baseline ID | Name / Purpose | Core Architectural Description | Primary Code Location |
| :--- | :--- | :--- | :--- |
| **B0** | **Zero-Shot Generation** | Bypasses vector/FTS retrieval entirely. Feeds raw query into LLM prompt requesting an answer based on parametric memory. | [core/generation.py:91](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/generation.py#L91) |
| **B1** | **Pure Lexical (FTS5)** | Exact keyword search using SQLite FTS5 index. Evaluates lexical match bounds without vector search or reranking. | [core/config.py:27](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/config.py#L27) |
| **B2** | **Pure Dense** | Semantic vector search via dense chunk embeddings. Evaluates vector similarity without keyword search or reranking. | [core/config.py:29](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/config.py#L29) |
| **B3** | **Dense + HyDE** | Dense retrieval augmented with Hypothetical Document Embeddings generated by LLM prior to vector search. | [core/config.py:31](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/config.py#L31) |
| **B4** | **Standard Hybrid + Reranker** | Dual-path Lexical + Dense search fused via Reciprocal Rank Fusion (RRF) and dynamic alpha blending, reranked by Cross-Encoder. | [core/config.py:34](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/config.py#L34) |
| **B5** | **Hybrid + Graph + Reranker** | Extends B4 by turning ON `graph_expansion` (subgraph triples in prompt), `context_trimming`, and `citation_repair`. | [core/config.py:40](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/config.py#L40) |
| **B6** | **Full Pipeline** | Forces `reranker = True` and `graph_neighbors_in_rrf = True`, inheriting all other active snapshot components (`graph_expansion`, `score_blending`, etc.). | [core/config.py:49](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/config.py#L49) |
| **CUSTOM** | **Custom Preset / Interactive** | User-controlled configuration allowing custom hyperparameter and component toggles via CLI (`--graph-expansion`, `--reranker`, etc.) or YAML snapshot. | [config_creator.py:268](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/config_creator.py#L268) |

---

## 2. Active User Configuration Matrix

Below is the component matrix resolving the user's active `rag_components` snapshot across all baselines:

### Active Snapshot Settings Passed by User
```yaml
rag_components:
  citation_repair: false
  context_trimming: true
  dense_search: true
  dynamic_alpha_blending: false
  graph_bridge_retrieval: false
  graph_concept_retrieval: false
  graph_expansion: true
  graph_neighbors_in_rrf: true
  graph_ontology_lookup: true
  graph_retrieval_trace: true
  graph_selected_sources_card: true
  hyde: false
  intent_classifier: false
  lexical_search: true
  llm_query_expansion: false
  reranker: true
  rrf: true
  score_blending: true
  shannon_estimator_enabled: true
```

### Resolved Component State per Baseline

| Component | B0 | B1 | B2 | B3 | B4 | B5 | B6 (Active Snapshot) | Resolution Rule |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `lexical_search` | OFF | **ON** | OFF | OFF | **ON** | **ON** | **ON** | Forced per baseline; inherited in B6 |
| `dense_search` | OFF | OFF | **ON** | **ON** | **ON** | **ON** | **ON** | Forced per baseline; inherited in B6 |
| `hyde` | OFF | OFF | OFF | **ON** | OFF | OFF | OFF | Forced OFF in B4, B5, B6 |
| `rrf` | OFF | OFF | OFF | OFF | **ON** | **ON** | **ON** | Forced ON in B4, B5; inherited in B6 |
| `dynamic_alpha_blending` | OFF | OFF | OFF | OFF | **ON** | **ON** | OFF | Forced ON in B4, B5; inherited as `false` in B6 |
| `reranker` | OFF | OFF | OFF | OFF | **ON** | **ON** | **ON** | Forced ON in B4, B5, B6 |
| `score_blending` | OFF | OFF | OFF | OFF | OFF | OFF | **ON** | Inherited from snapshot in B6 (`true`) |
| `graph_expansion` | OFF | OFF | OFF | OFF | OFF | **ON** | **ON** | Forced OFF in B4, ON in B5; inherited in B6 (`true`) |
| `graph_neighbors_in_rrf` | OFF | OFF | OFF | OFF | OFF | OFF | **ON** | Forced ON only in B6 |
| `graph_concept_retrieval` | OFF | OFF | OFF | OFF | OFF | OFF | OFF | Inherited from snapshot in B6 (`false`) |
| `graph_bridge_retrieval` | OFF | OFF | OFF | OFF | OFF | OFF | OFF | Inherited from snapshot in B6 (`false`) |
| `graph_ontology_lookup` | OFF | OFF | OFF | OFF | OFF | OFF | **ON** | Inherited from snapshot in B6 (`true`) |
| `graph_selected_sources_card` | OFF | OFF | OFF | OFF | OFF | OFF | **ON** | Inherited from snapshot in B6 (`true`) |
| `graph_retrieval_trace` | OFF | OFF | OFF | OFF | OFF | OFF | **ON** | Inherited from snapshot in B6 (`true`) |
| `context_trimming` | OFF | OFF | OFF | OFF | OFF | **ON** | **ON** | Forced ON in B5; inherited in B6 (`true`) |
| `citation_repair` | OFF | OFF | OFF | OFF | OFF | **ON** | OFF | Forced ON in B5; inherited in B6 (`false`) |
| `intent_classifier` | OFF | OFF | OFF | OFF | OFF | OFF | OFF | Disabled globally unless explicitly enabled |
| `llm_query_expansion` | OFF | OFF | OFF | OFF | OFF | OFF | OFF | Inherited from snapshot in B6 (`false`) |
| `shannon_estimator_enabled` | **ON** | **ON** | **ON** | **ON** | **ON** | **ON** | **ON** | Inherited globally across ALL baselines |

---

## 3. RAG Pipeline Execution Architecture & 19 Component Reference

When an incoming query is processed by `RAGService` in [src/services/rag_service.py](file:///Users/vladimirkasterin/python/graph/back/src/services/rag_service.py), components are evaluated across 11 sequential execution stages:

```
[User Query]
  │
  ├─► Stage 1: Intent Classification & Filters (intent_classifier)
  ├─► Stage 2: Query Expansion & Concept Lookup (graph_ontology_lookup, llm_query_expansion)
  ├─► Stage 3: Candidate Retrieval (dense_search, hyde, lexical_search, graph_neighbors_in_rrf)
  ├─► Stage 4: Fusion & Blending (dynamic_alpha_blending, rrf)
  ├─► Stage 5: Graph Traversal (graph_concept_retrieval, graph_bridge_retrieval)
  ├─► Stage 6: Cross-Encoder Reranking (reranker, score_blending)
  ├─► Stage 7: Diagnostic Telemetry (graph_retrieval_trace)
  ├─► Stage 8: Prompt Construction (graph_expansion)
  ├─► Stage 9: Token Budget Trimming (context_trimming)
  ├─► Stage 10: LLM Generation & Uncertainty Estimation (shannon_estimator_enabled)
  └─► Stage 11: Post-Processing & Output Enrichment (citation_repair, graph_selected_sources_card)
```

### Detailed Breakdown of All 19 `rag_components`

#### 1. `intent_classifier`
* **Default State**: `False`
* **Execution Location**: [src/services/rag_service.py:553-605](file:///Users/vladimirkasterin/python/graph/back/src/services/rag_service.py#L553-L605)
* **Function**: Parses unstructured queries using LLM structured outputs into a clean query string and metadata filters (`year_start`, `year_end`, `author`, `venue`).

#### 2. `graph_ontology_lookup`
* **Default State**: `True`
* **Execution Location**: [src/services/rag_service.py:502-521](file:///Users/vladimirkasterin/python/graph/back/src/services/rag_service.py#L502-L521)
* **Function**: Queries knowledge graph concept nodes to inject canonical domain concept names and aliases into short queries ($\le 2$ words).

#### 3. `llm_query_expansion`
* **Default State**: `True`
* **Execution Location**: [src/services/rag_service.py:523-551](file:///Users/vladimirkasterin/python/graph/back/src/services/rag_service.py#L523-L551)
* **Function**: Asks LLM to generate domain search query rephrasings in English/Russian, triggering parallel dense searches across variants.

#### 4. `dense_search`
* **Default State**: `True`
* **Execution Location**: [src/services/rag_service.py:1324-1344](file:///Users/vladimirkasterin/python/graph/back/src/services/rag_service.py#L1324-L1344)
* **Function**: Computes cosine similarity vector search using embeddings model (`intfloat-multilingual-e5-base`).

#### 5. `hyde` (Hypothetical Document Embeddings)
* **Default State**: `True`
* **Execution Location**: [src/services/rag_service.py:1346-1394](file:///Users/vladimirkasterin/python/graph/back/src/services/rag_service.py#L1346-L1394)
* **Function**: Prompts LLM to generate a hypothetical answer snippet, embeds it, and retrieves candidates matching expected answer semantics.

#### 6. `lexical_search`
* **Default State**: `True`
* **Execution Location**: [src/services/rag_service.py:1396-1408](file:///Users/vladimirkasterin/python/graph/back/src/services/rag_service.py#L1396-L1408)
* **Function**: Executes Full-Text Keyword Match Search against SQLite FTS5 index.

#### 7. `graph_neighbors_in_rrf`
* **Default State**: `False`
* **Execution Location**: [src/services/rag_service.py:1425-1476](file:///Users/vladimirkasterin/python/graph/back/src/services/rag_service.py#L1425-L1476)
* **Function**: Gathers paper IDs from initial seed chunks, fetches chunks from 1st/2nd-hop citation graph neighbors, and injects them into the candidate pool *before* Reciprocal Rank Fusion.

#### 8. `dynamic_alpha_blending`
* **Default State**: `True`
* **Execution Location**: [src/services/rag_service.py:1490-1503](file:///Users/vladimirkasterin/python/graph/back/src/services/rag_service.py#L1490-L1503)
* **Function**: Dynamically adjusts lexical weight ($w_{\text{fts}}$) based on top BM25 match quality (low BM25 matches down-weight lexical candidates to $0.2$).

#### 9. `rrf` (Reciprocal Rank Fusion)
* **Default State**: `True`
* **Execution Location**: [src/services/rag_service.py:1505-1530](file:///Users/vladimirkasterin/python/graph/back/src/services/rag_service.py#L1505-L1530)
* **Function**: Fuses vector and FTS rank lists into a unified score:
  $$RRF(d) = \frac{w_{\text{dense}}}{k + \text{rank}_{\text{dense}}(d)} + \frac{w_{\text{fts}}}{k + \text{rank}_{\text{fts}}(d)}$$

#### 10. `graph_concept_retrieval`
* **Default State**: `False`
* **Execution Location**: [src/services/rag_service.py:1683-1724](file:///Users/vladimirkasterin/python/graph/back/src/services/rag_service.py#L1683-L1724)
* **Function**: Identifies concept nodes matching query concepts and retrieves candidate chunks from papers linked via `MENTIONS_CONCEPT` (Layer 2 expansion).

#### 11. `graph_bridge_retrieval`
* **Default State**: `False`
* **Execution Location**: [src/services/rag_service.py:1726-1865](file:///Users/vladimirkasterin/python/graph/back/src/services/rag_service.py#L1726-L1865)
* **Function**: Traverses 2-hop bridge pathways (`CITES`, `RELATED_TO`) connecting seed papers to expand multi-hop candidate coverage (Layers 3 & 4).

#### 12. `reranker` (Cross-Encoder Reranking)
* **Default State**: `True`
* **Execution Location**: [src/services/rag_service.py:2028-2067](file:///Users/vladimirkasterin/python/graph/back/src/services/rag_service.py#L2028-L2067)
* **Function**: Evaluates joint query-passage relevance scores using Transformer model `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`.

#### 13. `score_blending`
* **Default State**: `True`
* **Execution Location**: [src/services/rag_service.py:2047-2053](file:///Users/vladimirkasterin/python/graph/back/src/services/rag_service.py#L2047-L2053)
* **Function**: Linearly blends normalized Reranker score and RRF score:
  $$S_{\text{blended}} = 0.7 \cdot S_{\text{reranker}} + 0.3 \cdot S_{\text{rrf}}$$

#### 14. `graph_retrieval_trace`
* **Default State**: `False`
* **Execution Location**: [src/services/rag_service.py:1107-1277](file:///Users/vladimirkasterin/python/graph/back/src/services/rag_service.py#L1107-L1277)
* **Function**: Emits detailed diagnostic telemetry (candidate sizes, rank positions, survival rates) to `graph_retrieval_trace.jsonl`.

#### 15. `graph_expansion`
* **Default State**: `True`
* **Execution Location**: [src/services/rag_service.py:359-364](file:///Users/vladimirkasterin/python/graph/back/src/services/rag_service.py#L359-L364)
* **Function**: Formats topological graph relationship triples (`Paper A CITES Paper B`) into the `context_graph` block appended to the LLM prompt.

#### 16. `context_trimming`
* **Default State**: `True`
* **Execution Location**: [src/services/rag_service.py:367-484](file:///Users/vladimirkasterin/python/graph/back/src/services/rag_service.py#L367-L484)
* **Function**: Measures prompt tokens via `tiktoken` and prunes context graph triples and lower-ranked sentences if prompt length exceeds `model_max_context - 500`.

#### 17. `shannon_estimator_enabled`
* **Default State**: `True`
* **Execution Location**: [benchmarks/rag/core/generation.py:87](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/generation.py#L87)
* **Function**: Records output log probabilities during generation to compute Shannon entropy ($H$) and predictive uncertainty metrics during benchmarking.

#### 18. `citation_repair`
* **Default State**: `True`
* **Execution Location**: [src/services/rag_service.py:607-643](file:///Users/vladimirkasterin/python/graph/back/src/services/rag_service.py#L607-L643)
* **Function**: Validates numerical citation brackets (`[1]`, `[Block 2]`) in generated answers, removing hallucinated or out-of-bounds citations.

#### 19. `graph_selected_sources_card`
* **Default State**: `False`
* **Execution Location**: [src/services/rag_service.py:984-1105](file:///Users/vladimirkasterin/python/graph/back/src/services/rag_service.py#L984-L1105)
* **Function**: Appends a structured summary section to the final response detailing citation connections and shared concepts among cited papers.

---

## 4. Analysis of Pipeline Performance & Results Discrepancies

The observed metric variations across baselines **B4**, **B5**, and **B6** stem directly from the downstream stage interactions outlined below:

### Downstream Pipeline Comparison (B4 vs B5 vs B6)

| Pipeline Stage | Baseline B4 | Baseline B5 | Baseline B6 (Active Snapshot) | Impact on Performance & Safety |
| :--- | :--- | :--- | :--- | :--- |
| **Pre-RRF Candidates** | Dense + FTS5 | Dense + FTS5 | Dense + FTS5 + **2-hop Graph Neighbors** | **B6 Recall Gain**: Adding neighbor chunks expands retrieval coverage to facts missed by pure keyword/vector search. |
| **Rank Fusion** | Static RRF ($k=60$) | Static RRF ($k=60$) | Static RRF ($k=60$) | Standard reciprocal rank fusion across active candidate pools. |
| **Reranker Scoring** | Pure Reranker Score | Pure Reranker Score | **Blended Score** ($0.7 \cdot \text{Reranker} + 0.3 \cdot \text{RRF}$) | **B6 Precision Gain**: Score blending prevents the reranker from over-focusing on keyword overlap, maintaining structural rank ordering. |
| **Prompt Subgraph** | `Graph enrichment disabled.` | **Populated `context_graph`** | **Populated `context_graph`** | **B5 & B6 Safety Gain**: `graph_expansion: true` appends ~550 tokens of citation/concept triples. The LLM relies on these relations to correctly abstain on unanswerable queries (lowering Hallucination Rate). |
| **Post-Processing** | Raw output | Repairs Citations | Raw output (`citation_repair: false`) | **B5 Citation Cleanliness**: B5 removes out-of-bounds citations. In B6, `citation_repair` matches the user snapshot (`false`). |

---

## 5. Legitimate Behaviors vs Diagnostic Artifacts

### Legitimate System Behaviors
1. **B6 Candidate Expansion Advantage**: Higher Retrieval Recall and Context Precision in B6 are legitimate improvements driven by `graph_neighbors_in_rrf`, which seeds candidate lookup with citation graph neighbors.
2. **B5 & B6 Hallucination Reduction**: When `graph_expansion` is active (`true`), appending `context_graph` triples provides structural grounding. This allows the LLM to verify fact presence and abstain on unanswerable queries (converting False Positives to True Negatives).
3. **Shannon Entropy Tracking**: `shannon_estimator_enabled: true` runs evaluation-level uncertainty analysis across all baselines, recording logit entropy without affecting prompt retrieval text.

### Known Diagnostic Artifacts & Caveats
1. **Global Graph Retrieval Trigger Bug**: In [rag_service.py:1544](file:///Users/vladimirkasterin/python/graph/back/src/services/rag_service.py#L1544), `graph_retrieval_enabled` reads `concept_retrieval_enabled` and `bridge_retrieval_enabled` from global `config.data`. Because these settings default to `true` in snapshot files, graph candidate lookup steps (Layers 1–4) execute during benchmark runs for non-graph baselines (B1, B2, B4), logging trace files even though non-graph baselines discard graph candidates.
2. **Graph Candidate Survival Logging Bias**: If a chunk is retrieved by both base search (FTS/Dense) and graph candidate search, [rag_service.py:1925](file:///Users/vladimirkasterin/python/graph/back/src/services/rag_service.py#L1925) deduplicates the chunk and attributes it to base search. Consequently, the chunk is omitted from `graph_chunks_before_rerank`, artificially lowering the logged graph survival rate to `0.0%` even when graph-sourced content is present in the prompt.
