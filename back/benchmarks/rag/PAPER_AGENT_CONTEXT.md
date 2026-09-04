# Canonical Baselines

- **B1**: Pure Lexical (SQLite FTS5 keyword search only, zero dense retrieval, no reranker, no graph).
- **B2**: Pure Dense (Vector semantic search over chunk embeddings, zero lexical search, no reranker, no graph).
- **B3**: Standard Hybrid + Reranker (Dual-path Lexical + Dense fused via Reciprocal Rank Fusion [RRF $k=60$] with Dynamic Alpha Blending, reranked by Cross-Encoder `mmarco-mMiniLMv2-L12-H384-v1`, graph disabled).
- **B4**: Hybrid + Graph + Reranker (Standard Hybrid + Reranker baseline extended with static prompt-level graph augmentation `graph_expansion: true`, topological citation/concept triples injected into `context_graph`, context token trimming, and citation repair).
- **B5**: Full Pipeline (Comprehensive agentic pipeline combining prompt-level graph expansion with 2-hop neighbor candidate expansion before RRF `graph_neighbors_in_rrf: true`, blended reranker scoring $0.7 \cdot S_{\text{reranker}} + 0.3 \cdot S_{\text{rrf}}$, ontology lookup for short queries, and dynamic graph expander `ExperimentalGraphExpander`).

# Legacy Renumbering

- `B1` -> `B1` (unchanged)
- `B2` -> `B2` (unchanged)
- `legacy B4` -> `B3`
- `legacy B5` -> `B4`
- `legacy B6` -> `B5`

*Note: Legacy B3 (Dense + HyDE) was excluded from the final research suite due to lack of interest for the final comparative ablation.*

# Component Matrix

| Component | B1 | B2 | B3 | B4 | B5 | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Lexical retrieval (BM25 / FTS5) | ✓ | ✗ | ✓ | ✓ | ✓ | SQLite FTS5 exact keyword index |
| Dense retrieval (Vectors) | ✗ | ✓ | ✓ | ✓ | ✓ | Dense semantic embeddings (`e5-base` / `MiniLM`) |
| HyDE | ✗ | ✗ | ✗ | ✗ | ✗ | Forced OFF across all final baselines |
| Reciprocal Rank Fusion (RRF) | ✗ | ✗ | ✓ | ✓ | ✓ | $k = 60.0$ fusing dense and lexical ranks |
| Dynamic alpha blending | ✗ | ✗ | ✓ | ✓ | ✗ | BM25-score-dependent lexical weighting |
| Cross-encoder reranker | ✗ | ✗ | ✓ | ✓ | ✓ | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |
| Score blending | ✗ | ✗ | ✗ | ✗ | ✓ | Blended score: $0.7 \cdot S_{\text{reranker}} + 0.3 \cdot S_{\text{rrf}}$ |
| Graph prompt augmentation | ✗ | ✗ | ✗ | ✓ | ✓ | Citation and concept triples in prompt `context_graph` |
| Graph candidate expansion in RRF | ✗ | ✗ | ✗ | ✗ | ✓ | 2-hop citation graph neighbors added before RRF |
| Graph ontology lookup | ✗ | ✗ | ✗ | ✗ | ✓ | Injects canonical concepts for queries $\le 2$ words |
| Dynamic graph expander engine | ✗ | ✗ | ✗ | ✗ | ✓ | `ExperimentalGraphExpander` active |
| Graph bridge / concept traversal | ✗ | ✗ | ✗ | ✗ | ✗ | Inactive in default evaluated snapshot |
| Context trimming (pruning) | ✗ | ✗ | ✗ | ✓ | ✓ | Sentence / triple budget trimming via `tiktoken` |
| Citation repair | ✗ | ✗ | ✗ | ✓ | ✗ | Validates bracket citations; strips hallucinations |
| Graph selected sources card | ✗ | ✗ | ✗ | ✗ | ✓ | Appends structured citation relations card |
| Telemetry trace logging | ✗ | ✗ | ✗ | ✗ | ✓ | Logs candidate rank transitions to JSONL |
| Parent chunk deduplication | ✓ | ✓ | ✓ | ✓ | ✓ | Groups child chunks sharing parent |
| Shannon entropy estimator | ✓ | ✓ | ✓ | ✓ | ✓ | Telemetry for $H_{gen}$, $H_{rank}$, MSP uncertainty |
| LLM query expansion | ✗ | ✗ | ✗ | ✗ | ✗ | LLM query rephrasing disabled in active snapshot |
| Intent classifier | ✗ | ✗ | ✗ | ✗ | ✗ | Structured metadata filter disabled in active snapshot |

# Figures Updated

1. `generate_scientific_visualizations.py`:
   - Updated `load_run_summary()` to detect legacy runs (containing B6 or legacy B4/B5) and exclude legacy B3, mapping `B4->B3`, `B5->B4`, `B6->B5`.
   - Updated `_load_raw_logits_fast()` to stream `raw_logits.yaml` mapping legacy B4-B6 to B3-B5 while ignoring excluded legacy B3.
   - Regenerated all 10 publication figures for the active empirical run `run_20260729_211324_OCC-RAG-1.7B` and synced descriptions:
     - `fig01a_generation_entropy_paradox.png / .pdf`
     - `fig01b_rank_entropy_compression.png / .pdf`
     - `fig02_quality_metrics_radar.png / .html / .pdf`
     - `fig03_entropy_quality_tradeoff.png / .pdf`
     - `fig04_answerability_confusion_matrices.png / .pdf`
     - `fig05_graph_structural_diagnostics.png / .pdf`
     - `fig06_entropy_metrics_correlation_heatmap.png / .pdf`
     - `fig07_first_5_tokens_confidence_trajectory.png / .pdf`
     - `fig08_logit_margin_kde_distribution.png / .pdf`
     - `figures_description.md`
2. `scripts/plot_entropy_visualizations.py`:
   - Fixed `COLORS` palette and baseline ordering from legacy B1, B2, B4, B5, B6 to canonical B1, B2, B3, B4, B5.
   - Added automated legacy remapping and legacy B3 filtering when reading `metrics_details.csv`.
   - Regenerated entropy figures in `graphs/run_20260729_211324_OCC-RAG-1.7B/figures_entropy` and `graphs/run_20260725_194320_OCC-RAG-1.7B/figures_entropy`.
3. `core/connector.py`:
   - Updated `generate_statistical_plots()` so that legacy benchmark evaluations (having B1, B2, B4, B5, B6) automatically map to canonical B1–B5 on boxplot and Wilcoxon p-value heatmap axes without modifying underlying numerical records.
4. `core/visualization.py`:
   - Updated `COLOR_PALETTE` and `preferred_order` to canonical `['B1', 'B2', 'B3', 'B4', 'B5']`.

# Important Notes for Paper Agent

1. **Reranker Usage**:
   - B1 and B2 do **NOT** use a reranker.
   - B3, B4, and B5 all use the cross-encoder reranker (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`).
   - B5 additionally applies **Score Blending** ($0.7 \cdot \text{Reranker} + 0.3 \cdot \text{RRF}$), which prevents the reranker from completely overriding retrieval rank mass.

2. **Graph Integration Nuances**:
   - B1, B2, and B3 have **zero graph usage**.
   - B4 uses **prompt-level graph augmentation only** (`graph_expansion: true`), formatting 1st-degree citation and concept triples into `context_graph` in the LLM prompt.
   - B5 integrates the graph at multiple stages:
     - **Pre-RRF Retrieval**: expands candidates by adding 2-hop citation graph neighbors (`graph_neighbors_in_rrf: true`).
     - **Query Pre-processing**: ontology lookup injects canonical concept names for short queries ($\le 2$ words).
     - **Prompt Augmentation**: populates `context_graph`.
     - **Post-processing**: generates a structured cited sources card.
   - Dynamic multi-hop path crawling (`graph_bridge_retrieval`) and concept expansion (`graph_concept_retrieval`) were **disabled** in the active snapshot.

3. **Citation Handling**:
   - B4 uniquely runs `citation_repair: true` (which strips hallucinated or out-of-context citation brackets).
   - In B5, `citation_repair` was `false` in the evaluated snapshot, relying instead on the full prompt graph triples.

4. **The "Overconfidence Paradox" in B5**:
   - B5 exhibits the lowest generation entropy ($H_{gen} \approx 0.8909$ bits, indicating high token-level certainty) coupled with a sharp drop in semantic accuracy (0.236 vs 0.710 in B3).
   - Downstream reranker rank entropy ($H_{rank}$) sharply collapses from $3.32$ to $0.78$ bits across all reranker-enabled baselines (B3–B5).

5. **Sanity Check & Data Consistency**:
   - Underlying experimental YAML/CSV files (e.g. `raw_logits.yaml`, `metrics_details.csv`, `evaluation_results.yaml`) store keys under the legacy execution IDs (`B1, B2, B4, B5, B6`).
   - All presentation layers, plotting scripts, and statistical visualization tools now map them transparently to canonical `B1–B5`. No raw numerical values were altered.
