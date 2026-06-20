# RAG Custom Retrieval Benchmark Report

This report displays the retrieval stage performance of your custom configuration compared against active baselines.

## ⚙️ Custom Run Configuration Overrides

### Component Settings (vs B6 Full Pipeline)

| Component | Custom Value | B6 Default | Status |
| :--- | :---: | :---: | :---: |
| `citation_repair` | `True` | `True` | Unchanged |
| `context_trimming` | `True` | `True` | Unchanged |
| `dense_search` | `True` | `True` | Unchanged |
| `dynamic_alpha_blending` | `False` | `True` | 🟢 **Modified** |
| `graph_expansion` | `False` | `True` | 🟢 **Modified** |
| `graph_ontology_lookup` | `True` | `True` | Unchanged |
| `hyde` | `False` | `False` | Unchanged |
| `intent_classifier` | `False` | `False` | Unchanged |
| `lexical_search` | `True` | `True` | Unchanged |
| `llm_query_expansion` | `False` | `True` | 🟢 **Modified** |
| `reranker` | `True` | `True` | Unchanged |
| `rrf` | `True` | `True` | Unchanged |
| `score_blending` | `False` | `True` | 🟢 **Modified** |

### Hyperparameter Overrides (vs System Defaults)

| Parameter | Custom Value | Default Value | Status |
| :--- | :---: | :---: | :---: |
| `graph.gamma` | `0.0` | `0.5` | ⚡ **Overridden** |
| `graph.p_base` | `0.0` | `0.75` | ⚡ **Overridden** |
| `graph.semantic_score_threshold` | `0.35` | `0.4` | ⚡ **Overridden** |
| `graph.weight_authored` | `1.0` | `0.8` | ⚡ **Overridden** |
| `graph.weight_cites` | `1.0` | `0.7` | ⚡ **Overridden** |
| `graph.weight_default` | `1.0` | `0.5` | ⚡ **Overridden** |
| `graph.weight_mentions_concept` | `1.0` | `0.6` | ⚡ **Overridden** |
| `rag.dynamic_alpha_threshold_low` | `1.2` | `1.0` | ⚡ **Overridden** |
| `rag.dynamic_alpha_val_low` | `1.0` | `0.2` | ⚡ **Overridden** |
| `rag.score_blend_reranker_weight` | `0.75` | `0.7` | ⚡ **Overridden** |
| `rag.score_blend_rrf_weight` | `0.25` | `0.3` | ⚡ **Overridden** |


## 📊 Retrieval Performance Summary

| Baseline | Success Rate | Mean Recall | Context Precision | Mean Latency |
| :--- | :---: | :---: | :---: | :---: |
| 🏆 **CUSTOM (Ours)** | 100.0% | 1.0000 | 0.9000 | 0.795s |


> [!NOTE]
> - **Retrieval Recall**: proportion of expected papers retrieved.
> - **Context Precision**: Mean Average Precision of the retrieved chunks/papers.
