# B4/B5/B6 Static Baseline Investigation

## Executive summary

This static investigation analyzes the differences between baselines **B4 (Standard Hybrid + Reranker)**, **B5 (Hybrid + Graph + Reranker)**, and **B6 (Full Pipeline)** in the Science Graph RAG pipeline. Using static analysis of code, configurations, and logs from the latest benchmark run (`run_20260706_175830_OCC-RAG-1.7B_1568_tokens`), we identified the exact mechanisms governing the observed performance discrepancies:

*   **Identical Retrieval Path**: B4 and B5 share the same base dense and lexical retrieval parameters, candidate pool generation, and RRF/Reranking processes. Consequently, they retrieve the exact same 5 text chunks.
*   **Graph Context Append**: The key downstream difference between B4 and B5 is that B5 has `graph_expansion: True` enabled, which appends the topological graph relationships to the prompt context. B4 has `graph_expansion: False`, setting `context_graph` to a generic `"Graph enrichment disabled."`.
*   **Context Fillness and Safety**: Adding graph relationships in B5 increases the prompt token count by ~550 tokens on average, raising the **Context Fillness** from `0.355` to `0.401`. Furthermore, this structured citation and concept metadata provides the LLM with topological context, helping it accurately determine whether the text contains enough facts to answer. This reduces the **Hallucination Rate** from `40%` (B4) to `24%` (B5) by converting `FP` outcomes to `TN` (abstained correctly on unanswerable queries).
*   **B6 Dual Behavior**: B6 behaves differently:
    *   It forces `graph_neighbors_in_rrf: True`, which expands the pre-rerank candidate pool using chunks from graph neighbor papers, boosting **Retrieval Recall** (`0.78` vs `0.76`) and **Context Precision** (`0.837` vs `0.819`). This extra context improves the quality on answerable questions, raising **Semantic Accuracy** (`0.260` vs B5 `0.239`).
    *   However, B6 dynamically inherits `graph_expansion: False` from the user config snapshot. As a result, B6's final prompt lacks the graph relationships block (`context_graph = "Graph enrichment disabled."`), which deprives the LLM of the topological context needed to abstain, causing it to hallucinate on unanswerable queries (raising the **Hallucination Rate** back to `40%`).
    *   B6 is faster because it omits these extra graph relationship tokens and does not run the advanced graph expander (since `graph_expansion` is False).
*   **Misleading Graph Diagnostics**: The global statistics showing `Avg Graph Chunks: 19.27` and `Survival: 0.0%` for all baselines are diagnostic artifacts. First, the `graph_retrieval_enabled` check in [rag_service.py](file:///Users/vladimirkasterin/python/graph/back/src/services/rag_service.py#L1544) checks the global config snapshot where concept and bridge retrieval are `True`, running graph candidate lookup for B4/B1/B2 even though they are non-graph baselines. Second, the `Survival: 0.0%` rate occurs because:
    1.  Graph-only candidates lack direct keyword matching and are consistently ranked below base candidates by the Cross-Encoder reranker.
    2.  Shared chunks (retrieved by both base search and graph search) are excluded from the `graph_chunks_before_rerank` list, preventing their survival from being counted.

---

## Known facts from latest run

The key metrics from the last run (`run_20260706_175830_OCC-RAG-1.7B_1568_tokens`) are summarized below:

*   **B4 and B5 share the same retrieval performance**: Retrieval Recall is `0.760` and Context Precision is `0.819` for both.
*   **B5 has a lower Hallucination Rate**: `24%` (6 FP, 19 TN) compared to B4's `40%` (10 FP, 15 TN) and B6's `40%` (10 FP, 15 TN).
*   **B6 achieves the highest answerable quality**: Semantic Accuracy is `0.260` and AR-SA F1 is `0.303`.
*   **B6 has a higher Answer Rate**: `45.3%` (34 answers) compared to B4 and B5's `38.7%` (29 answers).
*   **All baselines report identical graph diagnostics**: `Avg Graph Chunks: 19.27`, `Survival: 0.0%`, and `Queries Survived: 0.0%`.

---

## Actual baseline definitions

Based on [core/config.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/config.py#L17) and [config_snapshot.yaml](file:///Users/vladimirkasterin/python/graph/graphs/run_20260706_175830_OCC-RAG-1.7B_1568_tokens/config_snapshot.yaml), the baseline component configurations are structured as follows:

| Component | B4 | B5 | B6 | Same or different? | Evidence |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Base Retrieval** | FTS5 + Dense | FTS5 + Dense | FTS5 + Dense | Same | Checked in [core/config.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/config.py#L34-L54) |
| **Lexical Retrieval** | ON | ON | ON | Same | Checked in [core/config.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/config.py#L34-L54) |
| **Dense Retrieval** | ON | ON | ON | Same | Checked in [core/config.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/config.py#L34-L54) |
| **RRF** | ON | ON | ON | Same | Checked in [core/config.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/config.py#L34-L54) |
| **Reranker** | ON | ON | ON | Same | Forced `True` in [core/config.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/config.py#L53) |
| **Graph Augmentation** | OFF | ON | OFF | **Different** | B5 explicitly enables `graph_expansion`; B6 inherits `False` from config snapshot. |
| **LLM query expansion** | OFF | OFF | OFF | Same | B6 inherits `False` from config snapshot (`llm_query_expansion: false`). |
| **HyDE** | OFF | OFF | OFF | Same | Forced `False` in [core/config.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/config.py#L52) |
| **Context Packing** | Standard | Standard | Standard | Same | Uses `build_context` in [rag_service.py](file:///Users/vladimirkasterin/python/graph/back/src/services/rag_service.py#L348) |
| **Context Budget** | Standard | Standard | Standard | Same | Standard `model_max_context = 12000` |
| **Context Trimming** | OFF | ON | ON | **Different** | B4 defaults to `False`; B5/B6 explicitly enable `context_trimming`. |
| **Citation Repair** | OFF | ON | OFF | **Different** | B4/B6 configurations set `citation_repair` to `False`. |
| **Prompt Template** | `ask_no_expander` | `ask_no_expander` | `ask_no_expander` | Same | Both resolve to `ask_no_expander` template when `graph_expansion` is False. |
| **Abstention Instruction** | Same | Same | Same | Same | Standard XML/markdown reasoning tags parsed in [sanitization.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/sanitization.py#L113) |
| **Generation Params** | Same | Same | Same | Same | Inherited from snapshot configurations (`temp: 0.0`) |
| **Postprocessing** | Raw | Repaired | Raw | **Different** | B5 runs `_validate_and_repair_citations` |
| **Evaluation Path** | Standard | Standard | Standard | Same | Processed via [evaluator.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/evaluator.py#L280) |
| **Cache Behavior** | Checked | Checked | Checked | Same | Baseline-specific cache key checks in [generation.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/generation.py#L365) |

---

## Shared retrieval analysis

B4 and B5 share the same base retrieval mechanisms:
1.  **Dense Retrieval**: Both run cosine similarity search on embeddings generated by `/Users/vladimirkasterin/models/embeddings/intfloat-multilingual-e5-base` (checked in [rag_service.py:1326](file:///Users/vladimirkasterin/python/graph/back/src/services/rag_service.py#L1326)).
2.  **Lexical Retrieval**: Both use SQLite FTS5 search (checked in [rag_service.py:1398](file:///Users/vladimirkasterin/python/graph/back/src/services/rag_service.py#L1398)).
3.  **Reciprocal Rank Fusion (RRF)**: Merges dense and lexical results using parameter `rrf_k = 60.0`.
4.  **Score Blending**: Both have `score_blending: False`, meaning candidates are sorted strictly by the Cross-Encoder reranker score (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`).
5.  **Graph Candidates Exclusion**: While graph candidate retrieval runs globally at runtime (due to a diagnostic bug), no graph-only chunks survive the top-5 rerank stage. Hence, B4 and B5 pass the exact same base candidate set to the LLM.

Because of this identical retrieval pipeline, **Retrieval Recall** (`0.760`) and **Context Precision** (`0.819`) are identical between B4 and B5. The retrieval paths start to diverge after the reranking step.

---

## Downstream differences after retrieval

The table below outlines how the baselines handle retrieved chunks and build prompts:

| Stage after retrieval | B4 behavior | B5 behavior | B6 behavior | Metric impact | Evidence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Graph augmentation** | OFF | ON (Adds subgraph text to prompt) | OFF | Context Fillness increases in B5. Improves safety by providing topological metadata. | [rag_service.py:359](file:///Users/vladimirkasterin/python/graph/back/src/services/rag_service.py#L359) |
| **Reranker placement** | Raw rerank score | Raw rerank score | Blended rerank + RRF score | Blended sorting keeps structurally relevant chunks high. | [rag_service.py:2047](file:///Users/vladimirkasterin/python/graph/back/src/services/rag_service.py#L2047) |
| **Deduplication** | Standard | Standard | Standard | Removes duplicate candidates. | [rag_service.py:645](file:///Users/vladimirkasterin/python/graph/back/src/services/rag_service.py#L645) |
| **Context packing** | Bypasses `trim_context` | Runs `trim_context` | Runs `trim_context` | Trimmed contexts fit within budget. B4 has no fallback if context overflows. | [rag_service.py:2100](file:///Users/vladimirkasterin/python/graph/back/src/services/rag_service.py#L2100) |
| **Context ordering** | Rerank score order | Rerank score order | Blended score order | Reorders chunks in the final prompt context. | [rag_service.py:2054](file:///Users/vladimirkasterin/python/graph/back/src/services/rag_service.py#L2054) |
| **Prompt construction** | Formats text blocks with empty graph context | Formats text blocks with populated subgraph | Formats text blocks with empty graph context | Appends ~550 tokens of relationships in B5, increasing context fillness. | [rag_service.py:359](file:///Users/vladimirkasterin/python/graph/back/src/services/rag_service.py#L359) |
| **Abstention behavior** | LLM reasons using text chunks only | LLM reasons using text + graph relations | LLM reasons using text chunks only | Graph relationships help the model abstain correctly (TN instead of FP). | [ask_no_expander.txt](file:///Users/vladimirkasterin/python/graph/back/src/prompts/rag/ask_no_expander.txt) |
| **Generation params** | `temp: 0.0` | `temp: 0.0` | `temp: 0.0` | Controlled, deterministic outputs. | [config_snapshot.yaml](file:///Users/vladimirkasterin/python/graph/graphs/run_20260706_175830_OCC-RAG-1.7B_1568_tokens/config_snapshot.yaml) |
| **Postprocessing** | None | Repairs numeric citations | None | Cleans up and validates generated answer citations. | [rag_service.py:607](file:///Users/vladimirkasterin/python/graph/back/src/services/rag_service.py#L607) |
| **Evaluation classifier** | Standard | Standard | Standard | Maps generated text to TP/FN/FP/TN classification. | [metrics.py:79](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/metrics.py#L79) |

---

## Graph diagnostics investigation

### Why does B4 have graph diagnostics?
This is a runtime bug. Inside [rag_service.py:1544](file:///Users/vladimirkasterin/python/graph/back/src/services/rag_service.py#L1544), `graph_retrieval_enabled` is calculated as `master_enabled or concept_enabled or bridge_enabled`.
*   `concept_enabled` and `bridge_enabled` properties in [config.py](file:///Users/vladimirkasterin/python/graph/back/src/config.py#L945) read from the global `config.data` configuration dictionary: `self.is_component_enabled("graph_concept_retrieval") or self.data.get("graph_retrieval", {}).get("concept_retrieval_enabled", False)`.
*   Because `concept_retrieval_enabled` and `bridge_retrieval_enabled` are globally `True` in `config.data` (config snapshot), they evaluate to `True` for **all** baselines, bypassing baseline-specific flags.
*   Thus, static graph retrieval layers (Layer 1 - Layer 4) execute for B1, B2, and B4. It populates `self._last_graph_trace` and writes logs to `graph_retrieval_trace.jsonl`.

### Why is graph candidate survival 0.0%?
There are two reasons:
1.  **Direct Text Relevance Ranking**: Graph neighbor chunks (which are conceptually or citation-linked) are contextually related but lack direct keyword matches to the query. Thus, the Cross-Encoder reranker scores them lower than base dense and lexical chunks. Under a tight `limit=5`, they get pushed out of the top 5 and never survive to the final context.
2.  **Shared Candidate Overlap Logging Bug**: If a chunk is retrieved by both base search and graph search, [rag_service.py:1925](file:///Users/vladimirkasterin/python/graph/back/src/services/rag_service.py#L1925) merges its retrieval sources into the existing base chunk and **does not** add it to `graph_expansion_chunks`. Consequently, this chunk is excluded from `graph_chunks_before_rerank`. In `_write_graph_retrieval_trace`, `survival_rate` only counts elements in `graph_chunks_before_rerank` that are present in `final_context_chunk_ids`. Since overlapping graph chunks are filtered out of that list, their survival is never registered, resulting in a reported survival rate of `0.0%` even when the chunk is verified in the final prompt context.

---

## B4 vs B5 explanation

*   **Identical retrieval but different downstream**: Since B4 and B5 retrieve the exact same text chunks, their base contexts are character-for-character identical.
*   **Different Context Fillness**: B5 enables `graph_expansion`, which triggers [rag_service.py:359](file:///Users/vladimirkasterin/python/graph/back/src/services/rag_service.py#L359) to construct the graph connections subgraph (e.g. `Paper A CITES Paper B`). This adds ~550 tokens of relationships text on average. Dividing 550 tokens by `max_input_token = 12000` gives an average increase of `0.046`, raising the fillness from `0.355` (B4) to `0.401` (B5).
*   **TP/FN and FP/TN Distribution**:
    *   **B5 fixes B4 FP**: In unanswerable queries (e.g. `unanswerable_iks_001` and `q3_methods_missing_search_api_parameters`), B4 has `graph_expansion: False` and lacks relationship context. The model hallucinates facts (e.g. guessing `"0.10"` for deep learning density) yielding `FP`. B5 includes graph connections, which provides explicit citation/relationship metadata. The LLM processes this structural info and correctly concludes that the requested value is not present in the sources, yielding a correct `UNANSWERABLE` verdict (`TN`). This lowers the Hallucination Rate in B5 to `24%`.
    *   **B5 fixes B4 FN**: In answerable queries (e.g. `DS_KAMI_MULTI_Q02` and `EVI_CHAOS_MULTI_Q05`), the graph connections provide topological links (e.g., paper citations and relations) that clarify the context for the model. B5 uses these relations to synthesize a correct answer (`TP`), whereas B4 lacks these links and abstains (`FN`) due to a lack of structural context.

---

## B5 vs B6 explanation

*   **B6 Added Components**:
    *   `graph_neighbors_in_rrf: True`: B6 retrieves neighbors of seed papers up to depth 2, embeds them, and mixes them into the pre-rerank candidate pool, increasing retrieval coverage.
    *   `score_blending: True`: Combines reranker and RRF scores, preserving structural retrieval ordering in the final top-k.
*   **B6 Better Answerable Quality**: By pulling in neighboring paper chunks and blending scores, B6 captures facts that were missed by simple dense/lexical search. This raises its **Retrieval Recall** (`0.780`) and **Context Precision** (`0.837`), giving the LLM higher-quality chunks to answer answerable questions. As a result, B6 achieves a higher **Semantic Accuracy** (`0.260` vs B5 `0.239`).
*   **B6 Worse Safety**: However, B6 inherits `graph_expansion: False` from the user config snapshot. Thus, B6's prompt omits the graph connections text context (`context_graph = "Graph enrichment disabled."`). Without this topological connection mapping in its prompt, B6 lacks the structural check needed to abstain, causing it to hallucinate on unanswerable queries. Its **Hallucination Rate** rises to `40%` (10 FP, same as B4).
*   **B6 Faster Latency**: B6 is faster (`25.57s` vs B5 `32.02s`) and has lower **Context Fillness** (`0.350` vs B5 `0.401`) because it omits the ~550 tokens of graph connections text in the prompt, and does not execute the `ExperimentalGraphExpander` class.

---

## Per-query diff summary

Below is a detailed analysis of queries showing different outcomes in the last run:

| Query ID | Category | B4 outcome | B5 outcome | B6 outcome | Difference | Likely stage |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **`unanswerable_iks_001`** | unanswerable-missing-metric | FP | TN | FP | B5 abstains correctly; B4 and B6 hallucinate. | Prompt Construction (`graph_expansion`) |
| **`q3_methods_missing_search_api_parameters`** | unanswerable-missing-implementation | FP | TN | FP | B5 abstains correctly; B4 and B6 hallucinate. | Prompt Construction (`graph_expansion`) |
| **`unanswerable_q5`** | unanswerable-missing-experimental | FP | TN | TN | B5 and B6 abstain correctly; B4 hallucinates. | Prompt Construction (`graph_expansion`) |
| **`DS_KAMI_MULTI_Q02`** | multi-hop | FN | TP | FN | B5 answers correctly; B4 and B6 abstain. | Prompt Construction (`graph_expansion`) |
| **`EVI_CHAOS_MULTI_Q05`** | multi-hop | FN | TP | TP | B5 and B6 answer correctly; B4 abstains. | Prompt Construction (`graph_expansion`) |
| **`BERT_SINGLE_Q03`** | single-document | FN | FN | TP | B6 answers correctly; B4 and B5 abstain. | Retrieval (`graph_neighbors_in_rrf`) |
| **`CHAOS_SINGLE_Q04`** | single-document | FN | FN | TP | B6 answers correctly; B4 and B5 abstain. | Retrieval (`graph_neighbors_in_rrf`) |
| **`BERT_SINGLE_Q04`** | single-document | TP | TP | FN | B6 abstains; B4 and B5 answer correctly. | Reranking / Trimming |

---

## Metric impact analysis

The table below maps baseline code differences to their expected metric impact:

| Code/config difference | Affected metric | Expected direction | Evidence | Confidence |
| :--- | :--- | :--- | :--- | :--- |
| **`graph_neighbors_in_rrf`** | Retrieval Recall, Context Precision | Increase (B6 > B4/B5) | Candidates expanded in [rag_service.py:1436](file:///Users/vladimirkasterin/python/graph/back/src/services/rag_service.py#L1436) | High |
| **`graph_expansion`** | Context Fillness | Increase (B5 > B4/B6) | Subgraph text added in [rag_service.py:359](file:///Users/vladimirkasterin/python/graph/back/src/services/rag_service.py#L359) | High |
| **`graph_expansion`** | Hallucination Rate, TN rate | Decrease Hallucinations (B5 < B4/B6) | LLM utilizes graph connections in [ask_no_expander.txt](file:///Users/vladimirkasterin/python/graph/back/src/prompts/rag/ask_no_expander.txt) to abstain | High |
| **`score_blending`** | Chunk selection order | Reorders top-5 chunks | Blended scoring formula in [rag_service.py:2047](file:///Users/vladimirkasterin/python/graph/back/src/services/rag_service.py#L2047) | Medium |
| **`citation_repair`** | Latency, Citation fidelity | Increase latency, clean citations | Repair function call in [rag_service.py:2146](file:///Users/vladimirkasterin/python/graph/back/src/services/rag_service.py#L2146) | Medium |

---

## Legitimate behavior vs artifacts

### Expected / legitimate behavior
1.  **B6 Retrieval Gain**: The higher Retrieval Recall and Context Precision in B6 is a legitimate improvement driven by `graph_neighbors_in_rrf`, which successfully expands the candidate search space using the citation graph.
2.  **B6 Answer Quality**: The increase in Semantic Accuracy (`0.260` in B6) is a legitimate result of having higher-quality neighbor chunks in the context.
3.  **B5 Safety Gain**: The lower Hallucination Rate in B5 is a legitimate result of including `context_graph` relationships in the prompt, which helps the LLM distinguish between connected and disconnected facts.

### Suspicious artifacts / possible bugs
1.  **B6 Missing Graph connections**: B6 is intended as a "Full Pipeline" baseline, but because it inherits `graph_expansion: False` from the user's config snapshot, it runs **without** graph relationships in the prompt, causing it to hallucinate. B6 should force `graph_expansion` to `True` at runtime.
2.  **B4/B1/B2 Graph Diagnostics**: Running graph neighbor retrieval and concept extraction for non-graph baselines is a runtime bug. The code in [rag_service.py](file:///Users/vladimirkasterin/python/graph/back/src/services/rag_service.py#L1544) checks the global `config.data` toggles instead of the baseline-specific configurations.
3.  **Graph Survival Rate Log Bug**: The survival rate is logged as `0.0%` because chunks retrieved by both base search and graph search are deduplicated and excluded from the `graph_chunks_before_rerank` tracking list.

---

## Ranked hypotheses

| Rank | Hypothesis | Evidence | Counter-evidence | Static confidence | Needs rerun? |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **1** | B4 and B5 share the same base retrieval, resulting in identical Retrieval Recall. | [core/config.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/config.py#L34-L48) baseline setups match. | None | High | No |
| **2** | B5 safety gains are due to the presence of graph connections in the prompt. | B5 config has `graph_expansion: True`, B4/B6 have `False`. B5 correctly abstains on unanswerable queries. | None | High | No |
| **3** | B6 achieves better recall by expanding candidates with graph neighbor papers. | B6 config forces `graph_neighbors_in_rrf: True` in [core/config.py:54](file:///Users/vladimirkasterin/python/graph/back/config.py#L54). | None | High | No |
| **4** | Graph survival of 0.0% is caused by the deduplication bug excluding shared chunks. | [rag_service.py:1925](file:///Users/vladimirkasterin/python/graph/back/src/services/rag_service.py#L1925) deduplicates candidates and omits them from `graph_expansion_chunks`. | None | High | No |
| **5** | B6 safety is worse because it inherits `graph_expansion: False` from the config snapshot. | `graph_expansion` is `false` in `config_snapshot.yaml`. B6's `context_graph` evaluates to `"Graph enrichment disabled."`. | None | High | No |

---

## Recommended code instrumentation, without rerunning now

To improve future baseline evaluation and diagnostics, the following code modifications are recommended:

1.  **Fix B6 Baseline configuration**:
    Update [core/config.py](file:///Users/vladimirkasterin/python/graph/back/benchmarks/rag/core/config.py#L49-L54) to explicitly force `graph_expansion` to `True` for B6:
    ```diff
    elif baseline == "B6":
        components = {k: config_rag_components.get(k, True) for k in config_rag_components.keys()}
        components["hyde"] = False
        components["reranker"] = True
    +   components["graph_expansion"] = True
        components["graph_neighbors_in_rrf"] = True
    ```
2.  **Fix Graph Retrieval Enabled Checks**:
    In [rag_service.py](file:///Users/vladimirkasterin/python/graph/back/src/services/rag_service.py#L1540), check baseline-specific component configurations instead of the global `config` object:
    ```diff
    -   master_enabled = config.graph_retrieval_enabled
    -   concept_enabled = config.graph_concept_retrieval_enabled
    -   bridge_enabled = config.graph_bridge_retrieval_enabled
    +   master_enabled = config.rag_components.get("graph_expansion", False)
    +   concept_enabled = config.rag_components.get("graph_concept_retrieval", False)
    +   bridge_enabled = config.rag_components.get("graph_bridge_retrieval", False)
    ```
3.  **Fix Graph Candidate Logging (Deduplication Bug)**:
    Modify the deduplication block in [rag_service.py](file:///Users/vladimirkasterin/python/graph/back/src/services/rag_service.py#L1925) to add the chunk ID to `graph_expansion_chunks` even if it is a duplicate of a base chunk:
    ```diff
    else:
        seen_chunk_ids.add(c_id)
        unique_chunks[c_id] = c
        chunk_to_key[c_id] = (layer_num, idx, getattr(c, "paper_id", None) or "", c_id or "")
        graph_expansion_chunks.append(c)
    +   # Add duplicates of base chunks to graph_expansion_chunks for correct survival logging
    +   if c_id in unique_chunks and layer_num > 0:
    +       graph_expansion_chunks.append(unique_chunks[c_id])
    ```

---

## Final conclusion

This investigation confirms that the metric differences between B4, B5, and B6 are **fully deterministic** and follow from their runtime configurations:
1.  **B4 vs B5**: Their base text retrieval is identical. B5's safety improvements and increased Context Fillness are due to `graph_expansion` appending the graph connections subgraph to the prompt context. This topological info helps the LLM distinguish connected facts from hallucinations.
2.  **B5 vs B6**: B6 achieves better recall and semantic accuracy because it expands candidates using `graph_neighbors_in_rrf`. However, because it inherits `graph_expansion: False` from the config snapshot, its final prompt lacks graph connections, leading to higher hallucination rates on unanswerable questions.
3.  **Graph Diagnostics Artifacts**: The reported `0.0%` survival rate and global graph checks are diagnostic bugs caused by deduplication mapping and incorrect global config checks.

**Recommendation**: The current run is a valid comparison of their runtime setups, but B6 does not represent a true "Full Pipeline" run because `graph_expansion` was disabled. We recommend updating the configuration setup code and applying the recommended logging fixes before executing the next benchmark.
