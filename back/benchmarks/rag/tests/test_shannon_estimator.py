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
    compute_citation_onset_entropy,
    compute_entropy_reduction,
    parse_graph_relations_from_text,
    assemble_retrieval_shannon_fields,
    empty_retrieval_shannon_fields,
    compute_softmax,
    compute_msp,
    compute_logit_margin,
    compute_first_token_metrics,
    compute_sequence_telemetry,
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


def test_compute_rank_entropy_sum_closed_form_scores_10_20_30():
    """Exact Shannon H for sum-normalized P = [10,20,30]/60 = [1/6, 1/3, 1/2]."""
    scores = [10.0, 20.0, 30.0]
    p = [10 / 60, 20 / 60, 30 / 60]  # [1/6, 1/3, 1/2]
    expected = -sum(pi * math.log2(pi) for pi in p if pi > 0)
    got = compute_rank_entropy(scores, method="sum")
    assert math.isclose(got, expected, rel_tol=1e-9, abs_tol=1e-12)
    # aliases
    assert math.isclose(compute_rank_entropy(scores, method="l1"), expected, rel_tol=1e-9)
    assert math.isclose(compute_rank_entropy(scores, method="linear"), expected, rel_tol=1e-9)


def test_compute_rank_entropy_minmax_closed_form_scores_10_20_30():
    """Minmax: n=[0, 0.5, 1], sum=1.5 → P=[0, 1/3, 2/3]; zero-mass term drops out."""
    scores = [10.0, 20.0, 30.0]
    # n_i = (s - min)/(max-min) → [0, 0.5, 1.0]; P = n/sum(n) = [0, 1/3, 2/3]
    p_nonzero = [1.0 / 3.0, 2.0 / 3.0]
    expected = -sum(pi * math.log2(pi) for pi in p_nonzero)
    got = compute_rank_entropy(scores, method="minmax")
    assert math.isclose(got, expected, rel_tol=1e-9, abs_tol=1e-12)
    # sanity: expected formula as written in the plan
    assert math.isclose(
        expected,
        -(1 / 3) * math.log2(1 / 3) - (2 / 3) * math.log2(2 / 3),
        rel_tol=1e-12,
    )


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


def test_compute_graph_entropy_degree_closed_form():
    """Star-like: A→B cites, A→C cites.

    Degrees: A=2, B=1, C=1; total_degree=4 → P=[1/2, 1/4, 1/4].
    H_degree = 0.5 + 0.5 = 1.5 bits. Both edges same type → relation_type H = 0.
    """
    relations = [
        {"source": "A", "target": "B", "type": "cites"},
        {"source": "A", "target": "C", "type": "cites"},
    ]
    res = compute_graph_entropy(relations)
    expected_degree = (
        -0.5 * math.log2(0.5) - 0.25 * math.log2(0.25) - 0.25 * math.log2(0.25)
    )
    assert math.isclose(res["degree_entropy"], expected_degree, rel_tol=1e-9)
    assert math.isclose(res["degree_entropy"], 1.5, rel_tol=1e-9)
    assert res["relation_type_entropy"] == 0.0


def test_parse_graph_relations_from_real_cypher_lines():
    """Real RAG graph lines use quoted titles with spaces — old \\w+ regex failed."""
    graph_text = (
        "- ('Increasing the speed of wavelet transforms':Paper)-[CITES]->"
        "(work:doi:10.1109/access.2019.2903125:ExternalWork)\n"
        "- (Alice Smith:Author)-[AUTHORED]->('Some Paper Title':Paper)\n"
        "- (p1:Paper)-[MENTIONS_CONCEPT]->(entropy:Concept)"
    )
    rels = parse_graph_relations_from_text(graph_text)
    assert len(rels) == 3
    types = {r["type"] for r in rels}
    assert types == {"CITES", "AUTHORED", "MENTIONS_CONCEPT"}
    ent = compute_graph_entropy(rels)
    assert ent["relation_type_entropy"] > 0.0
    assert ent["degree_entropy"] > 0.0


def test_parse_graph_relations_cypher_with_preview_attrs_and_fields():
    """Cypher-like line with relation attributes still yields typed source/target."""
    line = (
        "- ('Long title with spaces':Paper)-[CITES {preview: \"cite snippet\"}]->"
        "(work:doi:10.1/xyz:ExternalWork)"
    )
    rels = parse_graph_relations_from_text(line)
    assert len(rels) == 1
    r = rels[0]
    assert r["type"] == "CITES"
    assert "Long title with spaces" in r["source"]
    assert r["source_label"] == "Paper"
    assert "ExternalWork" in r["target"] or "work" in r["target"]


def test_parse_graph_relations_disabled_or_empty():
    assert parse_graph_relations_from_text("") == []
    assert parse_graph_relations_from_text("No direct graph relations found.") == []
    assert parse_graph_relations_from_text("Graph enrichment disabled.") == []


def test_assemble_retrieval_shannon_fields_distinct_pre_post():
    """Distinct pre vs post inputs must produce distinct rank/lexical metrics."""
    # Peaked post-rerank scores → lower entropy than flat pre-rerank
    pre_scores = [1.0, 1.0, 1.0, 1.0]  # flat → log2(4) = 2.0
    post_scores = [10.0, 0.1, 0.1, 0.1]  # peaked → lower H
    pre_text = "alpha beta gamma delta epsilon zeta eta theta"
    post_text = "alpha alpha alpha beta"  # lower unigram diversity

    fields = assemble_retrieval_shannon_fields(
        pre_scores=pre_scores,
        post_scores=post_scores,
        pre_text=pre_text,
        post_text=post_text,
        relations=[
            {"source": "A", "target": "B", "type": "CITES"},
            {"source": "B", "target": "C", "type": "AUTHORED"},
        ],
    )
    assert fields["h_rank_pre_rerank"] != fields["h_rank_post_rerank"]
    assert fields["h_rank_pre_rerank"] > fields["h_rank_post_rerank"]
    assert fields["h_lexical_pre_trim"] != fields["h_lexical_post_trim"]
    assert fields["h_graph_relation_type"] > 0.0
    assert fields["h_graph_degree"] > 0.0


def test_assemble_pre_flat_vs_post_peaked_h_rank_strictly_decreases():
    """Required case: pre=[1,1,1,1] post=[10,0.1,0.1,0.1] ⇒ h_rank_pre > h_rank_post."""
    fields = assemble_retrieval_shannon_fields(
        pre_scores=[1, 1, 1, 1],
        post_scores=[10, 0.1, 0.1, 0.1],
    )
    assert fields["h_rank_pre_rerank"] > fields["h_rank_post_rerank"]
    # flat softmax/default → log2(4)=2.0 bits (rounded to 4 dp in assemble)
    assert math.isclose(fields["h_rank_pre_rerank"], 2.0, abs_tol=1e-4)


def test_assemble_does_not_force_pre_equal_post_when_pre_exists():
    """Regression: pre must not be overwritten with post when pre scores/text exist."""
    fields = assemble_retrieval_shannon_fields(
        pre_scores=[0.5, 0.5, 0.5, 0.5],
        post_scores=[5.0, 0.01, 0.01, 0.01],
        pre_text="one two three four five six seven eight",
        post_text="one one one one",
    )
    # Hard failure if someone reintroduces pre := post copy
    assert fields["h_rank_pre_rerank"] != fields["h_rank_post_rerank"]
    assert fields["h_lexical_pre_trim"] != fields["h_lexical_post_trim"]


def test_assemble_fallback_when_pre_missing_uses_post():
    """When pre boundary is absent (stage off / not captured), pre equals post honestly."""
    post_scores = [1.0, 2.0, 3.0]
    post_text = "hello world hello"
    fields = assemble_retrieval_shannon_fields(
        pre_scores=None,
        post_scores=post_scores,
        pre_text=None,
        post_text=post_text,
    )
    assert fields["h_rank_pre_rerank"] == fields["h_rank_post_rerank"]
    assert fields["h_lexical_pre_trim"] == fields["h_lexical_post_trim"]


def test_assemble_graph_from_text_when_relations_missing():
    graph_text = (
        "- (p1:Paper)-[CITES]->(p2:Paper)\n"
        "- (p1:Paper)-[AUTHORED]->(a1:Author)"
    )
    fields = assemble_retrieval_shannon_fields(
        post_scores=[1.0, 2.0],
        post_text="context",
        relations=None,
        graph_text=graph_text,
    )
    assert fields["h_graph_relation_type"] > 0.0
    assert fields["h_graph_degree"] > 0.0


def test_empty_retrieval_shannon_fields():
    z = empty_retrieval_shannon_fields()
    assert all(v == 0.0 for v in z.values())
    assert set(z.keys()) == {
        "h_rank_pre_rerank",
        "h_rank_post_rerank",
        "h_lexical_pre_trim",
        "h_lexical_post_trim",
        "h_graph_relation_type",
        "h_graph_degree",
    }
    # every key is float 0.0 (not int / None)
    assert all(isinstance(v, float) and v == 0.0 for v in z.values())


def test_docs_pipeline_and_shannon_estimator_exist():
    """Structural smoke: package docs for pipeline + shannon estimator are present."""
    from pathlib import Path

    docs = Path(__file__).resolve().parents[1] / "docs"
    for name in ("pipeline_orchestration.md", "shannon_estimator.md"):
        path = docs / name
        assert path.is_file(), f"missing {path}"
        assert path.stat().st_size > 100, f"{path} is trivially empty"


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


def test_align_tokens_info():
    from core.shannon_estimator import align_tokens_info

    full_text = "<think>Reasoning step</think>Answer in [Block 1]."
    clean_text = "Answer in [Block 1]."
    offset = full_text.find(clean_text)

    raw_tokens = [
        {"token_text": "<think>Reasoning step</think>", "char_start": 0, "char_end": offset, "entropy": 0.5},
        {"token_text": "Answer ", "char_start": offset, "char_end": offset + 7, "entropy": 0.2},
        {"token_text": "in ", "char_start": offset + 7, "char_end": offset + 10, "entropy": 0.3},
        {"token_text": "[Block 1].", "char_start": offset + 10, "char_end": offset + 20, "entropy": 1.2},
    ]

    aligned = align_tokens_info(full_text, clean_text, raw_tokens)
    assert len(aligned) == 3
    assert aligned[0]["char_start"] == 0
    assert aligned[0]["char_end"] == 7
    assert aligned[2]["token_text"] == "[Block 1]."

    h_cit, count = compute_citation_entropy(aligned, clean_text)
    assert count == 1
    assert math.isclose(h_cit, 1.2)


def test_occ_rag_1_7b_integration_logits_shannon_entropy():
    """Integration test: load OCC-RAG-1.7B, generate response with logits, and verify non-zero H_gen."""
    import os
    from src.config import resolve_existing_model_path
    model_path = resolve_existing_model_path("/Users/vladimirkasterin/models/llm/OCC-RAG-1.7B")
    if not os.path.exists(model_path):
        pytest.skip("OCC-RAG-1.7B model path not found locally.")

    try:
        from src.llm_engine.mlx_impl import MlxLLMEngine
        engine = MlxLLMEngine(model_path=model_path)
    except Exception as e:
        pytest.skip(f"MlxLLMEngine initialization failed: {e}")

    prompt = "According to [Block 1], gravity acceleration on Earth is 9.8 m/s^2. State the key facts with citation."
    text, tokens_info = engine.generate_response_with_logits(prompt, max_tokens=40)

    assert len(text) > 0
    assert len(tokens_info) > 0

    h_gen = compute_generation_entropy(tokens_info)
    h_cit, n_cit = compute_citation_entropy(tokens_info, text)

    assert h_gen > 0.0, f"Expected non-zero H_gen, got {h_gen}"
    # Verify each token has non-negative valid entropy
    for t in tokens_info:
        assert "entropy" in t
        assert isinstance(t["entropy"], float)
        assert t["entropy"] >= 0.0

    if n_cit > 0:
        assert h_cit > 0.0, f"Expected non-zero H_citation when citation tokens present, got {h_cit}"


def test_build_token_char_spans_and_map_offset():
    from core.shannon_estimator import build_token_char_spans, map_char_offset_to_token_idx

    tokens_info = [
        {"token": "Hello", "logprob": -0.1},
        {"token": " World", "logprob": -0.2},
        {"token": "!", "logprob": -0.05},
    ]

    spans = build_token_char_spans(tokens_info)
    assert spans == [(0, 5), (5, 11), (11, 12)]
    assert tokens_info[0]["char_start"] == 0
    assert tokens_info[0]["char_end"] == 5

    # Boundary and in-token mappings
    assert map_char_offset_to_token_idx(0, spans) == 0
    assert map_char_offset_to_token_idx(3, spans) == 0
    assert map_char_offset_to_token_idx(5, spans) == 1
    assert map_char_offset_to_token_idx(10, spans) == 1
    assert map_char_offset_to_token_idx(11, spans) == 2
    assert map_char_offset_to_token_idx(100, spans) == 2
    assert map_char_offset_to_token_idx(-5, spans) == 0

    # Empty spans edge case
    assert map_char_offset_to_token_idx(5, []) == 0


def test_compute_log_likelihood_and_clr():
    from core.shannon_estimator import compute_log_likelihood, compute_clr

    rag_tokens = [
        {"token": "The", "logprob": -0.5},
        {"token": " answer", "logprob": -1.0},
        {"token": " is", "logprob": -0.2},
    ]
    base_tokens = [
        {"token": "The", "logprob": -1.5},
        {"token": " answer", "logprob": -2.0},
        {"token": " is", "logprob": -1.0},
    ]

    ll_rag = compute_log_likelihood(rag_tokens)
    ll_base = compute_log_likelihood(base_tokens)

    assert math.isclose(ll_rag, -1.7)
    assert math.isclose(ll_base, -4.5)

    clr = compute_clr(ll_rag, ll_base)
    assert math.isclose(clr, 2.8)

    # Empty tokens edge cases
    assert compute_log_likelihood([]) == 0.0
    assert compute_clr(0.0, 0.0) == 0.0


def test_compute_softmax_numerical_stability_and_normalization():
    # Basic logits
    logits = [1.0, 2.0, 3.0]
    probs = compute_softmax(logits)
    assert len(probs) == 3
    assert math.isclose(sum(probs), 1.0, abs_tol=1e-6)
    exp3 = math.exp(3 - 3)
    exp2 = math.exp(2 - 3)
    exp1 = math.exp(1 - 3)
    sum_exp = exp1 + exp2 + exp3
    assert math.isclose(probs[2], exp3 / sum_exp, rel_tol=1e-5)

    # Large values for numerical stability (no overflow)
    large_logits = [1000.0, 1002.0, 999.0]
    large_probs = compute_softmax(large_logits)
    assert len(large_probs) == 3
    assert math.isclose(sum(large_probs), 1.0, abs_tol=1e-6)
    assert not any(math.isnan(p) for p in large_probs)

    # Edge cases
    assert compute_softmax([]) == []
    assert compute_softmax([5.0]) == [1.0]
    assert compute_softmax({"a": 0.0, "b": 0.0}) == [0.5, 0.5]


def test_compute_msp_raw_logits_and_probs():
    # Raw logits
    logits = [1.0, 2.0, 3.0]
    msp_logits = compute_msp(logits)
    expected_msp = math.exp(0) / (math.exp(-2) + math.exp(-1) + math.exp(0))
    assert math.isclose(msp_logits, expected_msp, rel_tol=1e-5)

    # Pre-normalized probabilities
    probs = [0.1, 0.7, 0.2]
    assert math.isclose(compute_msp(probs), 0.7)

    # Dict and empty edge cases
    assert math.isclose(compute_msp({"tok1": 10.0, "tok2": 0.0}), math.exp(10) / (math.exp(10) + math.exp(0)), rel_tol=1e-5)
    assert compute_msp([]) == 0.0
    assert compute_msp(None) == 0.0


def test_compute_logit_margin_top1_top2():
    # Sorted & unsorted logits
    assert math.isclose(compute_logit_margin([10.0, 7.0, 3.0]), 3.0)
    assert math.isclose(compute_logit_margin([3.0, 10.0, 7.0]), 3.0)

    # Logprobs
    logprobs = [-0.2, -0.8, -2.5]
    assert math.isclose(compute_logit_margin(logprobs), 0.6)

    # Dict logits
    dict_logits = {"a": 15.5, "b": 12.0, "c": 8.0}
    assert math.isclose(compute_logit_margin(dict_logits), 3.5)

    # Edge cases
    assert compute_logit_margin([]) == 0.0
    assert compute_logit_margin([5.0]) == 0.0
    assert compute_logit_margin(None) == 0.0


def test_compute_first_token_metrics():
    tokens_info = [
        {"token": "First", "logits": [5.0, 2.0, 1.0]},
        {"token": "Second", "logits": [10.0, 9.0]},
    ]
    metrics = compute_first_token_metrics(tokens_info)
    assert "first_token_margin" in metrics
    assert "first_token_msp" in metrics
    assert math.isclose(metrics["first_token_margin"], 3.0)

    exp5 = math.exp(0)
    sum_exp = math.exp(0) + math.exp(-3) + math.exp(-4)
    assert math.isclose(metrics["first_token_msp"], exp5 / sum_exp, abs_tol=1e-3)

    # Empty list
    empty_metrics = compute_first_token_metrics([])
    assert empty_metrics == {"first_token_margin": 0.0, "first_token_msp": 0.0}


def test_compute_sequence_telemetry():
    tokens_info = [
        {"token": "A", "logits": [10.0, 7.0]},  # margin=3.0
        {"token": "B", "logits": [8.0, 6.0]},   # margin=2.0
    ]
    telemetry = compute_sequence_telemetry(tokens_info)
    assert "avg_msp" in telemetry
    assert "avg_logit_margin" in telemetry
    assert "first_token_margin" in telemetry
    assert "first_token_msp" in telemetry
    assert math.isclose(telemetry["avg_logit_margin"], 2.5)
    assert telemetry["first_token_margin"] == 3.0

    # Empty list
    empty_t = compute_sequence_telemetry([])
    assert empty_t["avg_msp"] == 0.0
    assert empty_t["avg_logit_margin"] == 0.0
    assert empty_t["first_token_margin"] == 0.0
    assert empty_t["first_token_msp"] == 0.0


def test_compute_citation_onset_entropy_regex_matching():
    text = "As stated in [sciq_paper_1], gravity exists. Doc 2 shows details."
    tokens_info = [
        {"token": "As stated in ", "char_start": 0, "char_end": 13, "entropy": 0.1},
        {"token": "[sciq_paper_1]", "char_start": 13, "char_end": 27, "entropy": 1.5},
        {"token": ", gravity exists. ", "char_start": 27, "char_end": 45, "entropy": 0.2},
        {"token": "Doc 2", "char_start": 45, "char_end": 50, "entropy": 2.5},
        {"token": " shows details.", "char_start": 50, "char_end": 65, "entropy": 0.3},
    ]

    h_onset, count = compute_citation_onset_entropy(tokens_info, text)
    assert count == 2
    assert math.isclose(h_onset, 2.0)

    # Empty/no match text
    h_empty, count_empty = compute_citation_onset_entropy(tokens_info, "Plain text without citations")
    assert h_empty == 0.0
    assert count_empty == 0


def test_metrics_reexport_of_shannon_math_functions():
    from core.metrics import (
        compute_softmax as m_softmax,
        compute_msp as m_msp,
        compute_logit_margin as m_margin,
        compute_clr as m_clr,
        compute_first_token_metrics as m_first,
        compute_sequence_telemetry as m_seq,
        compute_citation_onset_entropy as m_onset,
    )
    assert callable(m_softmax)
    assert callable(m_msp)
    assert callable(m_margin)
    assert callable(m_clr)
    assert callable(m_first)
    assert callable(m_seq)
    assert callable(m_onset)
    assert math.isclose(m_clr(5.0, 2.0), 3.0)



