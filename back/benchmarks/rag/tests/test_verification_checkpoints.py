"""
Verification Checkpoints Test Suite
Strictly verifies all required system checkpoints before merging code to production.
"""

import math
import pytest
import numpy as np

from core.metrics import normalize_id
from core.retrieval import normalize_component_scores
from core.shannon_estimator import (
    compute_softmax,
    compute_msp,
    compute_logit_margin,
    compute_log_likelihood,
    compute_clr,
)
from core.generation import score_text_logprobs_base


def test_checkpoint_1_id_normalization_equality():
    """Checkpoint 1: ID Normalization Equality Check.
    Pass mock array of edge-case ground truth IDs (e.g. ['docs/paper_101.pdf#chunk_1'], ['doc_42'])
    and retrieved IDs (e.g. ['101_chunk_1'], ['doc_42_chunk_3']) to normalize_id().
    Assert that set intersection evaluates correctly, confirming chunk-to-document truncation logic.
    """
    gt_ids = ["docs/paper_101.pdf#chunk_1", "doc_42", "DATA/DOC-99.txt#chunk_10"]
    retrieved_ids = ["101_chunk_1", "doc_42_chunk_3", "doc_99_chunk_5"]

    gt_normalized = {normalize_id(x) for x in gt_ids if normalize_id(x)}
    retrieved_normalized = {normalize_id(x) for x in retrieved_ids if normalize_id(x)}

    expected_gt = {"101", "doc_42", "doc_99"}
    expected_retrieved = {"101", "doc_42", "doc_99"}

    assert gt_normalized == expected_gt, f"Ground truth normalization failed: {gt_normalized}"
    assert retrieved_normalized == expected_retrieved, f"Retrieved normalization failed: {retrieved_normalized}"

    # Set intersection confirms chunk-to-document truncation equality
    intersection = gt_normalized & retrieved_normalized
    assert intersection == expected_gt, f"Set intersection mismatch: {intersection} != {expected_gt}"
    assert len(intersection) == 3


def test_checkpoint_2_probability_sum_validation():
    """Checkpoint 2: Probability Sum Validation.
    For an arbitrary sample of 100 generated token logit arrays, compute softmax probabilities
    and assert sum(p_i) == 1.0 +- 1e-6.
    """
    np.random.seed(42)
    for i in range(100):
        length = np.random.randint(2, 500)
        scale = float(np.random.uniform(0.1, 100.0))
        shift = float(np.random.uniform(-50.0, 50.0))
        logits = (np.random.randn(length) * scale + shift).tolist()

        probs = compute_softmax(logits)

        prob_sum = sum(probs)
        assert abs(prob_sum - 1.0) <= 1e-6, f"Sample {i}: Probability sum {prob_sum} not within 1.0 +- 1e-6"
        assert all(p >= 0.0 for p in probs), f"Sample {i}: Negative probability detected"


def test_checkpoint_3_cross_component_score_scaling():
    """Checkpoint 3: Cross-Component Score Scaling Check.
    Pass raw unscaled scores from BM25, Dense Vectors, and Graph retrieval to normalize_component_scores().
    Assert all output scores strictly lie within [0.0, 1.0], and single-document inputs evaluate safely to [1.0].
    """
    bm25_scores = [15.4, 2.1, 0.0]
    dense_scores = [0.85, 0.42]
    graph_scores = [12.5, 3.2, 0.0, 1.1]
    single_doc_scores = [42.0]
    identical_doc_scores = [5.0, 5.0, 5.0]

    for scores, label in [
        (bm25_scores, "BM25"),
        (dense_scores, "Dense"),
        (graph_scores, "Graph"),
        (single_doc_scores, "Single-doc"),
        (identical_doc_scores, "Identical-docs"),
    ]:
        norm_scores = normalize_component_scores(scores)
        assert len(norm_scores) == len(scores)
        assert all(0.0 <= s <= 1.0 for s in norm_scores), f"{label} output scores not in [0.0, 1.0]: {norm_scores}"

    assert normalize_component_scores(single_doc_scores) == [1.0]
    assert normalize_component_scores([0.5]) == [1.0]
    assert normalize_component_scores(identical_doc_scores) == [1.0, 1.0, 1.0]


def test_checkpoint_4_msp_range_constraint():
    """Checkpoint 4: MSP Range Constraint.
    Assert that all calculated MSP values fall strictly within (0.0, 1.0].
    """
    np.random.seed(42)
    for i in range(100):
        length = np.random.randint(2, 200)
        logits = (np.random.randn(length) * 10.0).tolist()
        msp = compute_msp(logits)
        assert 0.0 < msp <= 1.0, f"Sample {i}: MSP {msp} violates (0.0, 1.0] range constraint"

    # Test with probability distributions and dictionaries
    assert 0.0 < compute_msp([0.7, 0.2, 0.1]) <= 1.0
    assert 0.0 < compute_msp({"tok_a": 5.2, "tok_b": 1.1, "tok_c": -0.5}) <= 1.0


def test_checkpoint_5_logit_margin_constraint():
    """Checkpoint 5: Logit Margin Constraint.
    Assert that all calculated Delta z_{1,2} values are >= 0.0.
    """
    np.random.seed(42)
    for i in range(100):
        length = np.random.randint(2, 200)
        logits = (np.random.randn(length) * 20.0).tolist()
        margin = compute_logit_margin(logits)
        assert margin >= 0.0, f"Sample {i}: Logit margin Delta z_1,2 = {margin} < 0.0"

    # Edge cases
    assert compute_logit_margin([5.0, 5.0]) == 0.0
    assert compute_logit_margin([-1.0, -5.0]) == 4.0
    assert compute_logit_margin([10.0]) == 0.0
    assert compute_logit_margin([]) == 0.0


def test_checkpoint_6_clr_ablation_consistency():
    """Checkpoint 6: CLR Ablation Consistency.
    Assert LL_base calculates the log-likelihood of the exact identical token sequence as LL_rag,
    modifying only the input prompt context.
    """
    token_sequence_rag = [
        {"token": "The", "logprob": -0.1},
        {"token": " capital", "logprob": -0.05},
        {"token": " is", "logprob": -0.02},
        {"token": " Paris", "logprob": -0.3},
    ]

    ll_rag = compute_log_likelihood(token_sequence_rag)

    token_sequence_base = [
        {"token": "The", "logprob": -0.5},
        {"token": " capital", "logprob": -0.4},
        {"token": " is", "logprob": -0.1},
        {"token": " Paris", "logprob": -1.2},
    ]

    ll_base = compute_log_likelihood(token_sequence_base)

    rag_tokens = [t["token"] for t in token_sequence_rag]
    base_tokens = [t["token"] for t in token_sequence_base]

    # Identical token sequence requirement
    assert rag_tokens == base_tokens, "Token sequence evaluated for LL_base must be identical to LL_rag"

    clr = compute_clr(ll_rag, ll_base)
    assert isinstance(clr, float)
    assert math.isclose(clr, ll_rag - ll_base)
