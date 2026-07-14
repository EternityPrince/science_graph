# Architectural Decision Record: Shannon Estimator Module

## Overview
The `shannon_estimator.py` module introduces information-theoretic uncertainty metrics for evaluating RAG architectures.

## Mathematical Design & Principles

### 1. Score Normalization & Rank Entropy (`compute_rank_entropy`)
- Base units: Shannon entropy is calculated in **bits** ($\log_2$).
- Supports `softmax` (with temperature parameter $\tau$), `minmax`, and `sum` (L1) normalizations.
- Handles edge cases: Empty input or single candidate returns `0.0` bits.

### 2. Lexical Entropy (`compute_lexical_entropy`)
- Computes unigram frequency entropy in bits across tokenized words (`re.findall(r'\w+', text)`).
- Normalizes punctuation and case sensitivity. Returns `0.0` for empty or whitespace-only inputs.

### 3. Graph Topology Entropy (`compute_graph_entropy`)
- Measures graph structure uncertainty:
  - `relation_type_entropy`: Shannon entropy over edge relation type distribution.
  - `degree_entropy`: Shannon entropy over node total degree distribution $P(v) = d(v) / \sum_u d(u)$.

### 4. Citation Span Extraction (`find_citation_spans`)
- Uses regular expressions to detect citation markers, including:
  - `[sciq_paper_X]`
  - `[Block X]`
  - Numerical index citations e.g. `[1]`, `[1, 2]`, `[1-3]`
  - Generic paper/document references `[paper_1]`, `[doc_2]`
  - DOIs (`10.xxxx/...`) and arXiv identifiers (`arXiv:xxxx.xxxx`)
  - Author-year formats e.g. `(Smith et al., 2020)`
- Spans are merged to prevent character index overlap and returned as sorted `[start, end)` character bounds.

### 5. Generation & Citation Entropy (`compute_generation_entropy`, `compute_citation_entropy`)
- Extracts per-token entropy from probabilities, top logprobs, or surprisal metrics.
- Citation entropy maps token character bounds back to citation spans and calculates average token entropy restricted to citation tokens.

### 6. Entropy Reduction (`compute_entropy_reduction`)
- Computes baseline uncertainty reduction $\Delta H_{gen} = H_{b0} - H_{rag}$.
- Returns `0.0` when either metric is `None`.
