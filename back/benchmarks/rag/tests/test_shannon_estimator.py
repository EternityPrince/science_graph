"""Unit tests for the Shannon Estimator RAG core module."""

import math
import pytest
from core.shannon_estimator import (
    compute_rank_entropy,
    compute_lexical_entropy,
    compute_graph_entropy,
    find_citation_spans,
    compute_generation_entropy,
    compute_citation_entropy,
    compute_entropy_reduction,
)


def test_compute_rank_entropy_edge_cases():
    # Empty or single candidate scores return 0.0
    assert compute_rank_entropy([]) == 0.0
    assert compute_rank_entropy([0.8]) == 0.0

    # Equal scores with softmax should produce max entropy = log2(N)
    scores = [1.0, 1.0, 1.0, 1.0]
    expected_entropy = math.log2(4)  # 2.0 bits
    assert math.isclose(compute_rank_entropy(scores, method="softmax"), expected_entropy, rel_tol=1e-5)


def test_compute_rank_entropy_methods():
    scores = [10.0, 20.0, 30.0]

    # Softmax method
    h_softmax = compute_rank_entropy(scores, method="softmax", tau=1.0)
    assert h_softmax > 0.0

    # Minmax method
    h_minmax = compute_rank_entropy(scores, method="minmax")
    assert h_minmax > 0.0

    # Sum method
    h_sum = compute_rank_entropy(scores, method="sum")
    assert h_sum > 0.0

    # Invalid tau fallback
    h_invalid_tau = compute_rank_entropy(scores, method="softmax", tau=-1.0)
    assert h_invalid_tau >= 0.0


def test_compute_lexical_entropy():
    # Empty string
    assert compute_lexical_entropy("") == 0.0
    assert compute_lexical_entropy("   ") == 0.0

    # Single repeated token
    assert compute_lexical_entropy("test test test") == 0.0

    # Two tokens with equal count: 'a' and 'b' -> 50% each -> 1.0 bit
    text = "alpha beta alpha beta"
    assert math.isclose(compute_lexical_entropy(text), 1.0, rel_tol=1e-5)

    # Mixed case and punctuation
    text_punct = "Alpha, beta! Alpha; beta."
    assert math.isclose(compute_lexical_entropy(text_punct), 1.0, rel_tol=1e-5)


def test_compute_graph_entropy():
    # Empty graph
    res_empty = compute_graph_entropy([])
    assert res_empty == {"relation_type_entropy": 0.0, "degree_entropy": 0.0}

    # Graph with known topology
    relations = [
        {"source": "A", "target": "B", "type": "cites"},
        {"source": "B", "target": "C", "type": "cites"},
        {"source": "C", "target": "A", "type": "supports"},
        {"head": "A", "tail": "D", "relation": "supports"},
    ]
    res = compute_graph_entropy(relations)

    assert "relation_type_entropy" in res
    assert "degree_entropy" in res
    assert res["relation_type_entropy"] > 0.0
    assert res["degree_entropy"] > 0.0


def test_find_citation_spans():
    # Empty string
    assert find_citation_spans("") == []

    # Various citation formats
    text = (
        "According to [sciq_paper_42] and [Block 1], quantum gravity is studied in [1, 2]. "
        "See also 10.1038/s41586-020-2649-2 and arXiv:2106.01234 or (Smith et al., 2020)."
    )
    spans = find_citation_spans(text)
    assert len(spans) >= 6

    # Verify spans are sorted and non-overlapping
    for i in range(len(spans) - 1):
        assert spans[i][0] < spans[i][1]
        assert spans[i][1] <= spans[i + 1][0]

    # Verify text extracted from spans matches markers
    extracted = [text[start:end] for start, end in spans]
    assert "[sciq_paper_42]" in extracted
    assert "[Block 1]" in extracted
    assert "[1, 2]" in extracted
    assert "10.1038/s41586-020-2649-2" in extracted
    assert "arXiv:2106.01234" in extracted
    assert "(Smith et al., 2020)" in extracted


def test_find_citation_spans_overlapping():
    # Adjacent / overlapping markers
    text = "[Block 1][sciq_paper_99]"
    spans = find_citation_spans(text)
    assert len(spans) == 1
    assert spans[0] == (0, len(text))


def test_compute_generation_entropy():
    # Empty tokens info
    assert compute_generation_entropy([]) == 0.0

    # Tokens with direct entropy field
    tokens_entropy = [{"entropy": 1.5}, {"entropy": 0.5}]
    assert math.isclose(compute_generation_entropy(tokens_entropy), 1.0)

    # Tokens with top probabilities dict
    tokens_probs = [{"probs": {"a": 0.5, "b": 0.5}}]  # 1.0 bit
    assert math.isclose(compute_generation_entropy(tokens_probs), 1.0, rel_tol=1e-5)

    # Tokens with logprob field
    tokens_lp = [{"logprob": -math.log(0.25)}]  # -log2(0.25) = 2.0 bits
    assert math.isclose(compute_generation_entropy(tokens_lp), 2.0, rel_tol=1e-5)


def test_compute_citation_entropy():
    generated_text = "The result is shown in [Block 5] clearly."
    
    # Mock tokens with explicit character offsets
    tokens_info_explicit = [
        {"token": "The", "start": 0, "end": 3, "entropy": 0.2},
        {"token": " result", "start": 3, "end": 10, "entropy": 0.4},
        {"token": " in", "start": 18, "end": 21, "entropy": 0.1},
        {"token": " [Block 5]", "start": 21, "end": 32, "entropy": 1.8},
        {"token": " clearly.", "start": 32, "end": 41, "entropy": 0.3},
    ]

    h_cit, count = compute_citation_entropy(tokens_info_explicit, generated_text)
    assert count == 1
    assert math.isclose(h_cit, 1.8)

    # Mock tokens with sequential string reconstruction
    tokens_info_sequential = [
        {"token": "The", "entropy": 0.2},
        {"token": " result", "entropy": 0.4},
        {"token": " is", "entropy": 0.1},
        {"token": " shown", "entropy": 0.1},
        {"token": " in", "entropy": 0.1},
        {"token": " ", "entropy": 0.1},
        {"token": "[Block 5]", "entropy": 2.2},
        {"token": " clearly.", "entropy": 0.3},
    ]

    h_cit_seq, count_seq = compute_citation_entropy(tokens_info_sequential, generated_text)
    assert count_seq >= 1
    assert h_cit_seq > 0.0

    # Edge cases: no citations in text
    no_cit_text = "Plain answer with no citations."
    h_none, count_none = compute_citation_entropy(tokens_info_sequential, no_cit_text)
    assert h_none == 0.0
    assert count_none == 0


def test_compute_entropy_reduction():
    # None handling
    assert compute_entropy_reduction(None, 1.5) == 0.0
    assert compute_entropy_reduction(2.0, None) == 0.0
    assert compute_entropy_reduction(None, None) == 0.0

    # Normal delta calculation
    assert math.isclose(compute_entropy_reduction(3.5, 1.2), 2.3)
    assert math.isclose(compute_entropy_reduction(1.0, 2.5), -1.5)
