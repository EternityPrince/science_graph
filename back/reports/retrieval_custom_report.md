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
| `hyde` | `True` | `False` | 🟢 **Modified** |
| `intent_classifier` | `False` | `False` | Unchanged |
| `lexical_search` | `True` | `True` | Unchanged |
| `llm_query_expansion` | `True` | `True` | Unchanged |
| `reranker` | `True` | `True` | Unchanged |
| `rrf` | `True` | `True` | Unchanged |
| `score_blending` | `True` | `True` | Unchanged |

### Hyperparameter Overrides (vs System Defaults)

| Parameter | Custom Value | Default Value | Status |
| :--- | :---: | :---: | :---: |
| *None* | | | | 


## 📊 Retrieval Performance Summary

| Baseline | Success Rate | Mean Recall | Context Precision | Mean Latency |
| :--- | :---: | :---: | :---: | :---: |
| B4 | 100.0% | 1.0000 | 0.8500 | 0.171s |
| B6 | 100.0% | 1.0000 | 0.8667 | 0.530s |
| 🏆 **CUSTOM (Ours)** | 100.0% | 1.0000 | 0.8667 | 0.827s |


> [!NOTE]
> - **Retrieval Recall**: proportion of expected papers retrieved.
> - **Context Precision**: Mean Average Precision of the retrieved chunks/papers.
