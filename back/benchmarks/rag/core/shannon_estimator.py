"""Shannon Estimator RAG Core Module.

Provides mathematical entropy estimation tools for evaluation of RAG architectures:
- Rank score entropy (softmax, minmax, sum)
- Lexical unigram entropy
- Graph topology entropy (relation type and degree distribution)
- Generation entropy over tokens
- Citation-specific token entropy and span extraction
- Entropy reduction delta measurement
"""

import math
import re
from typing import Any, Dict, List, Optional, Tuple


def compute_rank_entropy(
    scores: List[float], method: str = "softmax", tau: float = 1.0
) -> float:
    """Calculates Shannon entropy in bits over normalized candidate rank scores.

    Args:
        scores: List of candidate retrieval or similarity scores.
        method: Normalization method ("softmax", "minmax", "sum"). Default is "softmax".
        tau: Temperature scaling parameter for softmax (tau > 0).

    Returns:
        float: Shannon entropy in bits (log base 2). Returns 0.0 for empty or single item scores.
    """
    if not scores or len(scores) <= 1:
        return 0.0

    if method == "softmax":
        if tau <= 0:
            tau = 1e-6
        max_score = max(scores)
        exp_scores = [math.exp((s - max_score) / tau) for s in scores]
        sum_exp = sum(exp_scores)
        if sum_exp <= 0:
            return 0.0
        probs = [e / sum_exp for e in exp_scores]
    elif method == "minmax":
        min_s = min(scores)
        max_s = max(scores)
        range_s = max_s - min_s
        if range_s == 0:
            probs = [1.0 / len(scores)] * len(scores)
        else:
            norm = [(s - min_s) / range_s for s in scores]
            sum_norm = sum(norm)
            if sum_norm == 0:
                probs = [1.0 / len(scores)] * len(scores)
            else:
                probs = [n / sum_norm for n in norm]
    elif method in ("sum", "l1", "linear"):
        total = sum(scores)
        if total <= 0:
            return 0.0
        probs = [s / total for s in scores]
    else:
        max_score = max(scores)
        tau_val = tau if tau > 0 else 1.0
        exp_scores = [math.exp((s - max_score) / tau_val) for s in scores]
        sum_exp = sum(exp_scores)
        if sum_exp <= 0:
            return 0.0
        probs = [e / sum_exp for e in exp_scores]

    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * math.log2(p)
    return max(0.0, float(entropy))


def compute_lexical_entropy(text: str) -> float:
    """Calculates unigram frequency token entropy in bits.

    Args:
        text: Input string.

    Returns:
        float: Shannon entropy of unigram token distribution in bits. Returns 0.0 if empty.
    """
    if not text or not text.strip():
        return 0.0

    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        return 0.0

    total_tokens = len(tokens)
    counts: Dict[str, int] = {}
    for tok in tokens:
        counts[tok] = counts.get(tok, 0) + 1

    entropy = 0.0
    for count in counts.values():
        p = count / total_tokens
        if p > 0:
            entropy -= p * math.log2(p)

    return max(0.0, float(entropy))


def compute_graph_entropy(relations: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculates relation_type_entropy and degree_entropy for graph topology in bits.

    Args:
        relations: List of relation dicts representing graph edges.

    Returns:
        Dict with keys "relation_type_entropy" and "degree_entropy".
    """
    if not relations:
        return {"relation_type_entropy": 0.0, "degree_entropy": 0.0}

    type_counts: Dict[str, int] = {}
    node_degrees: Dict[str, int] = {}

    for rel in relations:
        r_type = (
            rel.get("type")
            or rel.get("relation")
            or rel.get("label")
            or rel.get("predicate")
            or "unknown"
        )
        type_counts[str(r_type)] = type_counts.get(str(r_type), 0) + 1

        src = (
            rel.get("source")
            or rel.get("head")
            or rel.get("subject")
            or rel.get("src")
            or rel.get("from")
        )
        dst = (
            rel.get("target")
            or rel.get("tail")
            or rel.get("object")
            or rel.get("dst")
            or rel.get("to")
        )

        if src is not None:
            src_str = str(src)
            node_degrees[src_str] = node_degrees.get(src_str, 0) + 1
        if dst is not None:
            dst_str = str(dst)
            node_degrees[dst_str] = node_degrees.get(dst_str, 0) + 1

    total_edges = len(relations)
    rel_entropy = 0.0
    if total_edges > 0:
        for count in type_counts.values():
            p = count / total_edges
            if p > 0:
                rel_entropy -= p * math.log2(p)

    total_degree = sum(node_degrees.values())
    degree_entropy = 0.0
    if total_degree > 0:
        for d in node_degrees.values():
            p = d / total_degree
            if p > 0:
                degree_entropy -= p * math.log2(p)

    return {
        "relation_type_entropy": max(0.0, float(rel_entropy)),
        "degree_entropy": max(0.0, float(degree_entropy)),
    }


def align_tokens_info(
    full_text: str, clean_text: str, tokens_info: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Aligns token character positions (char_start, char_end) when full_text
    undergoes stripping of thinking tokens or special formatting tags.

    Args:
        full_text: Raw accumulated generated output text.
        clean_text: Final cleaned string after strip_thinking_tokens.
        tokens_info: List of token metric dictionaries.

    Returns:
        List[Dict[str, Any]]: Aligned list of token metric dictionaries matching clean_text.
    """
    if not tokens_info or not full_text:
        return []

    if full_text == clean_text:
        return tokens_info

    # Case 1: clean_text is a contiguous substring of full_text
    offset = full_text.find(clean_text)
    if offset != -1:
        aligned: List[Dict[str, Any]] = []
        clean_len = len(clean_text)
        for t in tokens_info:
            cs = t.get("char_start", t.get("start", 0))
            ce = t.get("char_end", t.get("end", 0))
            if ce > offset and cs < offset + clean_len:
                t_copy = dict(t)
                t_copy["char_start"] = max(0, cs - offset)
                t_copy["char_end"] = min(clean_len, ce - offset)
                aligned.append(t_copy)
        return aligned

    # Case 2: General fallback using sequential matching against clean_text
    aligned = []
    current_pos = 0
    clean_len = len(clean_text)
    for t in tokens_info:
        tok_text = t.get("token_text") or t.get("text") or t.get("token") or ""
        if not tok_text:
            continue
        pos = clean_text.find(tok_text, current_pos)
        if pos != -1:
            t_copy = dict(t)
            t_copy["char_start"] = pos
            t_copy["char_end"] = pos + len(tok_text)
            current_pos = pos + len(tok_text)
            aligned.append(t_copy)
        elif current_pos < clean_len:
            t_copy = dict(t)
            aligned.append(t_copy)

    return aligned if aligned else tokens_info


def find_citation_spans(text: str) -> List[Tuple[int, int]]:
    """Finds character-level [start, end) spans for citation markers.

    Recognized patterns include:
    - [sciq_paper_X]
    - [Block X]
    - [1], [1, 2], [1-3]
    - [paper_1], [doc_2], [ref_1], [id_1], [source_1], [Source_1], [Источник: 1]
    - Paper DOIs: 10.xxxx/... or [10.xxxx/...]
    - Author-year citations: e.g. (Smith et al., 2020) or [Jones, 2019]
    - DOIs and arXiv IDs

    Args:
        text: Input text string.

    Returns:
        List[Tuple[int, int]]: Sorted list of non-overlapping [start, end) character spans.
    """
    if not text:
        return []

    patterns = [
        # [sciq_paper_X]
        r"\[sciq_paper_[^\]]+\]",
        # [Block X] or [Block X, Block Y]
        r"\[Block\s+[^\]]+\]",
        # [1], [1, 2], [1-3]
        r"\[\d+(?:[\s,–-]+\d+)*\]",
        # [paper_1], [doc_2], [ref_1], [id_1], [source_1], [Source_1], [Источник: 1]
        r"\[(?:paper|doc|ref|id|source|Source|Источник)[_\s:]*[^\]]+\]",
        # Paper DOIs in brackets
        r"\[10\.\d{4,9}/[^\]]+\]",
        # Standalone DOIs
        r"\b10\.\d{4,9}/[^\s,;()\]]+",
        # Bracketed author-year: [Jones, 2019] or [Smith et al., 2020]
        r"\[[A-Z][a-zA-Z\s.-]+(?:et\s+al\.)?,\s*\d{4}[a-z]?\]",
        # Parenthetical author-year: (Smith et al., 2020)
        r"\([A-Z][a-zA-Z\s.-]+(?:et\s+al\.)?,\s*\d{4}[a-z]?\)",
        # arXiv IDs
        r"\barXiv:\d{4}\.\d{4,5}(?:v\d+)?\b",
    ]

    combined_pattern = "|".join(f"(?:{p})" for p in patterns)
    raw_spans = []

    for m in re.finditer(combined_pattern, text):
        raw_spans.append((m.start(), m.end()))

    if not raw_spans:
        return []

    # Merge overlapping or touching citation spans
    raw_spans.sort(key=lambda x: (x[0], x[1]))
    merged_spans: List[Tuple[int, int]] = []
    current_start, current_end = raw_spans[0]

    for next_start, next_end in raw_spans[1:]:
        if next_start <= current_end:
            current_end = max(current_end, next_end)
        else:
            merged_spans.append((current_start, current_end))
            current_start, current_end = next_start, next_end
    merged_spans.append((current_start, current_end))

    return merged_spans


def _extract_single_token_entropy(t_info: Dict[str, Any]) -> float:
    """Helper function to calculate entropy in bits for a single token dictionary."""
    if "entropy" in t_info and t_info["entropy"] is not None:
        return max(0.0, float(t_info["entropy"]))

    if "probs" in t_info or "top_probs" in t_info:
        probs = t_info.get("probs") or t_info.get("top_probs")
        if isinstance(probs, dict):
            probs_list = list(probs.values())
        elif isinstance(probs, list):
            probs_list = probs
        else:
            probs_list = []
        if probs_list:
            entropy = 0.0
            for p in probs_list:
                if p > 0:
                    entropy -= p * math.log2(p)
            return max(0.0, float(entropy))

    if "top_logprobs" in t_info or "logprobs" in t_info:
        lp_data = t_info.get("top_logprobs") or t_info.get("logprobs")
        if isinstance(lp_data, dict):
            lps = list(lp_data.values())
        elif isinstance(lp_data, list):
            lps = [x.get("logprob", 0.0) if isinstance(x, dict) else x for x in lp_data]
        else:
            lps = []
        if lps:
            probs = [math.exp(lp) for lp in lps]
            sum_p = sum(probs)
            if sum_p > 0:
                probs = [p / sum_p for p in probs]
                entropy = 0.0
                for p in probs:
                    if p > 0:
                        entropy -= p * math.log2(p)
                return max(0.0, float(entropy))

    if "logprob" in t_info and t_info["logprob"] is not None:
        lp = float(t_info["logprob"])
        # Handle both true negative logprob (ln p <= 0) and positive surprisal (-ln p > 0)
        p = math.exp(-lp) if lp > 0 else math.exp(lp)
        if 0 < p <= 1:
            return float(-math.log2(p))

    if "prob" in t_info and t_info["prob"] is not None:
        p = t_info["prob"]
        if 0 < p <= 1:
            return float(-math.log2(p))

    return 0.0


def compute_generation_entropy(tokens_info: List[Dict[str, Any]]) -> float:
    """Calculates average per-token Shannon entropy in bits across generation tokens.

    Args:
        tokens_info: List of token metadata dictionaries.

    Returns:
        float: Average token entropy in bits. Returns 0.0 if list is empty.
    """
    if not tokens_info:
        return 0.0

    total_entropy = sum(_extract_single_token_entropy(t) for t in tokens_info)
    return max(0.0, float(total_entropy / len(tokens_info)))


def compute_citation_entropy(
    tokens_info: List[Dict[str, Any]], generated_text: str
) -> Tuple[float, int]:
    """Maps character citation spans back to token character position ranges,
    and calculates average entropy specifically for citation tokens.

    Args:
        tokens_info: List of token metric dicts.
        generated_text: The complete generated text string.

    Returns:
        Tuple[float, int]: (average citation entropy in bits, number of citation tokens).
    """
    if not tokens_info or not generated_text:
        return (0.0, 0)

    spans = find_citation_spans(generated_text)
    if not spans:
        return (0.0, 0)

    citation_tokens: List[Dict[str, Any]] = []

    has_explicit_range = any(
        ("char_start" in t or "start" in t) and ("char_end" in t or "end" in t)
        for t in tokens_info
    )

    if has_explicit_range:
        for t in tokens_info:
            c_start = t.get("char_start", t.get("start"))
            c_end = t.get("char_end", t.get("end"))
            if c_start is not None and c_end is not None:
                if c_start == c_end:
                    if any(s_start <= c_start < s_end for s_start, s_end in spans):
                        citation_tokens.append(t)
                else:
                    if any(max(c_start, s_start) < min(c_end, s_end) for s_start, s_end in spans):
                        citation_tokens.append(t)

    if not citation_tokens:
        current_idx = 0
        for t in tokens_info:
            tok_text = t.get("token_text") or t.get("text") or t.get("token") or t.get("token_str") or ""
            if not tok_text:
                continue
            pos = generated_text.find(tok_text, current_idx)
            if pos != -1:
                t_start = pos
                t_end = pos + len(tok_text)
                current_idx = t_end
                if any(max(t_start, s_start) < min(t_end, s_end) for s_start, s_end in spans):
                    citation_tokens.append(t)

    if not citation_tokens:
        return (0.0, 0)

    avg_entropy = compute_generation_entropy(citation_tokens)
    return (avg_entropy, len(citation_tokens))


def compute_entropy_reduction(
    h_b0: Optional[float], h_rag: Optional[float]
) -> float:
    """Calculates uncertainty reduction ΔH_gen = h_b0 - h_rag.

    Args:
        h_b0: Baseline model generation entropy.
        h_rag: RAG model generation entropy.

    Returns:
        float: Uncertainty reduction in bits. Returns 0.0 if either value is None.
    """
    if h_b0 is None or h_rag is None:
        return 0.0
    return float(h_b0 - h_rag)


_GRAPH_DISABLED_MARKERS = frozenset({
    "",
    "No direct graph relations found.",
    "Graph enrichment disabled.",
})

# Matches real Cypher-style lines produced by rag_service._get_scored_graph_lines, e.g.:
# - ('Long title with spaces':Paper)-[CITES {preview: "..."}]->(work:doi:10.1:ExternalWork)
# - (Author Name:Author)-[AUTHORED]->(Paper Title:Paper)
_GRAPH_LINE_RE = re.compile(
    r"\("
    r"(?:'([^']*)'|\"([^\"]*)\"|([^):\]]+))"  # src name (quoted or bare)
    r":"
    r"([^)]+)"  # src label
    r"\)-"
    r"\["
    r"(\w+)"  # relation type
    r"[^\]]*"
    r"\]->"
    r"\("
    r"(?:'([^']*)'|\"([^\"]*)\"|([^):\]]+))"  # tgt name
    r":"
    r"([^)]+)"  # tgt label
    r"\)"
)


def parse_graph_relations_from_text(graph_text: str) -> List[Dict[str, Any]]:
    """Parse formatted graph context lines into relation dicts for compute_graph_entropy.

    Accepts the Cypher-like lines emitted by RAGService._get_scored_graph_lines.
    Returns an empty list for missing/disabled graph text.
    """
    if not graph_text or graph_text.strip() in _GRAPH_DISABLED_MARKERS:
        return []

    relations: List[Dict[str, Any]] = []
    for match in _GRAPH_LINE_RE.finditer(graph_text):
        src_name = match.group(1) or match.group(2) or (match.group(3) or "").strip()
        src_label = (match.group(4) or "").strip()
        rel_type = match.group(5)
        tgt_name = match.group(6) or match.group(7) or (match.group(8) or "").strip()
        tgt_label = (match.group(9) or "").strip()
        source = f"{src_name}:{src_label}" if src_label else src_name
        target = f"{tgt_name}:{tgt_label}" if tgt_label else tgt_name
        if not source or not target or not rel_type:
            continue
        relations.append({
            "source": source,
            "target": target,
            "type": rel_type,
            "source_label": src_label,
            "target_label": tgt_label,
        })
    return relations


def _coerce_score_list(scores: Optional[List[Any]]) -> List[float]:
    if not scores:
        return []
    out: List[float] = []
    for s in scores:
        try:
            out.append(float(s))
        except (TypeError, ValueError):
            out.append(0.0)
    return out


def assemble_retrieval_shannon_fields(
    *,
    pre_scores: Optional[List[Any]] = None,
    post_scores: Optional[List[Any]] = None,
    pre_text: Optional[str] = None,
    post_text: Optional[str] = None,
    relations: Optional[List[Dict[str, Any]]] = None,
    graph_text: Optional[str] = None,
) -> Dict[str, float]:
    """Assemble rank / lexical / graph Shannon fields from staged inputs.

    Semantics:
    - When distinct pre_scores / pre_text are provided, pre metrics use them.
    - When pre is missing, post is used for both pre and post (honest fallback when
      the stage was off or pre boundary was not captured).
    - Graph entropy prefers structured ``relations``; falls back to parsing
      ``graph_text`` with parse_graph_relations_from_text.
    """
    post_score_list = _coerce_score_list(post_scores)
    pre_score_list = _coerce_score_list(pre_scores)
    if not pre_score_list:
        pre_score_list = list(post_score_list)

    post_text_val = post_text if post_text is not None else ""
    pre_text_val = pre_text if pre_text is not None else post_text_val

    rels: List[Dict[str, Any]] = list(relations) if relations else []
    if not rels and graph_text:
        rels = parse_graph_relations_from_text(graph_text)

    graph_ent = compute_graph_entropy(rels)

    return {
        "h_rank_pre_rerank": round(compute_rank_entropy(pre_score_list), 4),
        "h_rank_post_rerank": round(compute_rank_entropy(post_score_list), 4),
        "h_lexical_pre_trim": round(compute_lexical_entropy(pre_text_val), 4),
        "h_lexical_post_trim": round(compute_lexical_entropy(post_text_val), 4),
        "h_graph_relation_type": round(graph_ent["relation_type_entropy"], 4),
        "h_graph_degree": round(graph_ent["degree_entropy"], 4),
    }


def empty_retrieval_shannon_fields() -> Dict[str, float]:
    """Zero retrieval-stage Shannon fields (B0 / empty retrieval)."""
    return {
        "h_rank_pre_rerank": 0.0,
        "h_rank_post_rerank": 0.0,
        "h_lexical_pre_trim": 0.0,
        "h_lexical_post_trim": 0.0,
        "h_graph_relation_type": 0.0,
        "h_graph_degree": 0.0,
    }

