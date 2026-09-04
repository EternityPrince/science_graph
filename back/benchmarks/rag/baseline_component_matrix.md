# Baseline Component Comparison Matrix (Final B1–B5)

This matrix reflects the **actual executable configuration and implementation** of the final baselines B1 through B5 in the Science Graph benchmark codebase.

### Baseline ID Mapping
- **B1**: Pure Lexical (Code: `B1`)
- **B2**: Pure Dense (Code: `B2`)
- **B3**: Standard Hybrid + Reranker (Legacy Code: `B4`)
- **B4**: Hybrid + Graph + Reranker (Legacy Code: `B5`)
- **B5**: Full Pipeline (Legacy Code: `B6`)
*(Note: Legacy B3 [Dense + HyDE] was excluded from the final study).*

---

### Component Comparison Matrix

| Component | B1 (Pure Lexical) | B2 (Pure Dense) | B3 (Hybrid + Rerank) | B4 (Graph + Rerank) | B5 (Full Pipeline) | Notes / Code Reference |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Lexical Retrieval (BM25 / FTS5)** | ✓ | ✗ | ✓ | ✓ | ✓ | SQLite FTS5 exact keyword index (`lexical_search`) |
| **Dense Vector Retrieval** | ✗ | ✓ | ✓ | ✓ | ✓ | Chunk embeddings (`dense_search`, e5-base / MiniLM) |
| **HyDE (Hypothetical Doc Embeddings)** | ✗ | ✗ | ✗ | ✗ | ✗ | Forced OFF (`hyde = False`); legacy B3 excluded |
| **Reciprocal Rank Fusion (RRF)** | ✗ | ✗ | ✓ | ✓ | ✓ | Rank fusion ($k = 60.0$) fusing dense + lexical ranks (`rrf`) |
| **Dynamic Alpha Blending** | ✗ | ✗ | ✓ | ✓ | ✗ | BM25-score-dependent lexical weighting (`dynamic_alpha_blending`) |
| **Cross-Encoder Reranker** | ✗ | ✗ | ✓ | ✓ | ✓ | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (`reranker`) |
| **Score Blending** | ✗ | ✗ | ✗ | ✗ | ✓ | Blends $0.7 \cdot S_{\text{reranker}} + 0.3 \cdot S_{\text{rrf}}$ (`score_blending`) |
| **Graph Prompt Augmentation** | ✗ | ✗ | ✗ | ✓ | ✓ | Injects citation/concept triples into `context_graph` (`graph_expansion`) |
| **Graph Candidate Expansion (Pre-RRF)** | ✗ | ✗ | ✗ | ✗ | ✓ | 1st/2nd-hop citation neighbors added before RRF (`graph_neighbors_in_rrf`) |
| **Graph Ontology Lookup** | ✗ | ✗ | ✗ | ✗ | ✓ | Injects canonical concept aliases for queries $\le 2$ words (`graph_ontology_lookup`) |
| **Dynamic Graph Expander Engine** | ✗ | ✗ | ✗ | ✗ | ✓ | `ExperimentalGraphExpander` loaded in retrieval and generation pipeline |
| **Graph Multi-hop Bridge / Concepts** | ✗ | ✗ | ✗ | ✗ | ✗ | 2-hop bridge & concept traversal disabled in active snapshot |
| **Context Trimming (Pruning)** | ✗ | ✗ | ✗ | ✓ | ✓ | Prunes lower-ranked sentences/triples to fit budget (`context_trimming`) |
| **Citation Repair** | ✗ | ✗ | ✗ | ✓ | ✗ | Validates bracket citations; strips hallucinations (`citation_repair`) |
| **Graph Selected Sources Card** | ✗ | ✗ | ✗ | ✗ | ✓ | Appends structured summary of citation links (`graph_selected_sources_card`) |
| **Diagnostic Telemetry (Traces)** | ✗ | ✗ | ✗ | ✗ | ✓ | Emits candidate tracking to `graph_retrieval_trace.jsonl` |
| **Parent Chunk Deduplication** | ✓ | ✓ | ✓ | ✓ | ✓ | Groups child chunks by parent chunk (`deduplicate_parent_chunks`) |
| **Shannon Entropy Estimator** | ✓ | ✓ | ✓ | ✓ | ✓ | Computes $H_{gen}$, $H_{rank}$, MSP uncertainty (`shannon_estimator_enabled`) |
| **LLM Query Expansion** | ✗ | ✗ | ✗ | ✗ | ✗ | LLM domain rephrasing disabled in active snapshot (`llm_query_expansion`) |
| **Intent Classifier** | ✗ | ✗ | ✗ | ✗ | ✗ | Query metadata filter classification disabled in active snapshot |
