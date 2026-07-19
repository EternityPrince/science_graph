# Shannon Estimator Module

Information-theoretic diagnostics for evaluating RAG components.

**Primary developer guide** (metrics, pre/post semantics, how to use when building new RAG stages):

→ [`back/benchmarks/rag/docs/shannon_estimator.md`](../benchmarks/rag/docs/shannon_estimator.md)

## Implementation

| Module | Purpose |
|--------|---------|
| `benchmarks/rag/core/shannon_estimator.py` | Pure entropy math + `assemble_retrieval_shannon_fields` |
| `benchmarks/rag/core/retrieval.py` | Persists `pre_rerank_scores`, `context_text`, `graph_relations` |
| `benchmarks/rag/core/generation.py` / `pipelined.py` | Fills diagnostics during generation |
| `src/services/rag_service.py` | Sets `_last_pre_rerank_scores` and `_last_graph_relations` |

## Mathematical primitives

1. **Rank entropy** (`compute_rank_entropy`) — softmax / minmax / sum over candidate scores, bits.
2. **Lexical entropy** (`compute_lexical_entropy`) — unigram token entropy of context text.
3. **Graph entropy** (`compute_graph_entropy`) — relation-type and degree entropy over edge dicts.
4. **Generation / citation entropy** — from token logits and citation span alignment.
5. **ΔH_gen** (`compute_entropy_reduction`) — $H_{B0} - H_{RAG}$.

## Config

```yaml
rag_components:
  shannon_estimator_enabled: true
```

## Pre / post (summary)

- **H_rank pre/post**: scores before vs after reranker. Equality is honest when the reranker is off or pre scores were never captured.
- **H_lexical pre/post**: `context_text` vs `trimmed_text`. Equality is honest when trimming does not change text.
- **H_graph**: non-zero only when structured relations (or parseable graph text) exist — not hardcoded zeros when the graph is present.

## Performance notes

- Top-k logprob extraction on MLX reduces full-vocab sync cost during generation entropy.
- B0 $H_{\text{gen}}$ may be cached to avoid redundant zero-shot generation across baselines.
