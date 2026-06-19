# RAG Custom Retrieval Benchmark Report

This report displays the retrieval stage performance of your custom configuration compared against active baselines.

## ⚙️ Custom Run Configuration Overrides

### Component Settings (vs B6 Full Pipeline)

| Component | Custom Value | B6 Default | Status |
| :--- | :---: | :---: | :---: |
| `citation_repair` | `True` | `True` | Unchanged |
| `context_trimming` | `True` | `True` | Unchanged |
| `dense_search` | `True` | `True` | Unchanged |
| `dynamic_alpha_blending` | `True` | `True` | Unchanged |
| `graph_expansion` | `True` | `True` | Unchanged |
| `graph_ontology_lookup` | `True` | `True` | Unchanged |
| `hyde` | `False` | `False` | Unchanged |
| `intent_classifier` | `False` | `False` | Unchanged |
| `lexical_search` | `True` | `True` | Unchanged |
| `llm_query_expansion` | `True` | `True` | Unchanged |
| `reranker` | `True` | `True` | Unchanged |
| `rrf` | `True` | `True` | Unchanged |
| `score_blending` | `True` | `True` | Unchanged |

### Hyperparameter Overrides (vs System Defaults)

| Parameter | Custom Value | Default Value | Status |
| :--- | :---: | :---: | :---: |
| `bm25.b` | `0.72` | `0.75` | ⚡ **Overridden** |
| `bm25.k1` | `1.6` | `1.5` | ⚡ **Overridden** |
| `graph.gamma` | `0.45` | `0.5` | ⚡ **Overridden** |
| `graph.p_base` | `0.82` | `0.75` | ⚡ **Overridden** |
| `graph.semantic_score_threshold` | `0.35` | `0.4` | ⚡ **Overridden** |
| `rag.dynamic_alpha_threshold_low` | `1.2` | `1.0` | ⚡ **Overridden** |
| `rag.dynamic_alpha_val_low` | `0.15` | `0.2` | ⚡ **Overridden** |
| `rag.rrf_k` | `55.0` | `60.0` | ⚡ **Overridden** |
| `rag.score_blend_reranker_weight` | `0.75` | `0.7` | ⚡ **Overridden** |
| `rag.score_blend_rrf_weight` | `0.25` | `0.3` | ⚡ **Overridden** |


## 📊 Retrieval Performance Summary

| Baseline | Success Rate | Mean Recall | Context Precision | Mean Latency |
| :--- | :---: | :---: | :---: | :---: |
| B4 | 100.0% | 1.0000 | 1.0000 | 0.001s |
| B6 | 100.0% | 1.0000 | 1.0000 | 1.837s |
| 🏆 **CUSTOM (Ours)** | 100.0% | 1.0000 | 1.0000 | 1.591s |


> [!NOTE]
> - **Retrieval Recall**: proportion of expected papers retrieved.
> - **Context Precision**: Mean Average Precision of the retrieved chunks/papers.
