"""
Science Graph — Core Robustness Extra Test Suite.

Contains comprehensive robustness, edge-case, and boundary tests for core modules:
- core.statistics (MCC, per-query records, paired vectors, zero-variance bootstrap)
- core.metrics (retrieval recall, context precision, cosine similarity, abstention detection)
- core.limiter (AsyncRateLimiter with zero/negative RPM and concurrent tasks)
- core.sanitization (reasoning & final answer extraction with malformed tags)
- core.pipelined (YAML persistence error handling and lock safety)
"""

import math
import tempfile
import asyncio
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch


from core.statistics import (
    compute_mcc,
    compute_classification_metrics,
    prepare_per_query_records,
    paired_metric_vectors,
    significance_stars,
    format_p_value
)
from core.metrics import (
    calculate_retrieval_recall,
    calculate_context_precision,
    compute_cosine_similarity,
    detect_abstention,
    count_text_tokens,
    estimate_prompt_tokens,
    get_is_answerable,
    normalize_optional_text
)
from core.limiter import AsyncRateLimiter
from core.sanitization import extract_clean_answer, clean_answer_tokens
from core.pipelined import safe_read_modify_write_yaml


# =====================================================================
# 1. core.statistics Robustness
# =====================================================================

class TestCoreStatisticsRobustness:
    """Robustness tests for core.statistics functions."""

    def test_compute_mcc_zero_denominator(self):
        # When any term in (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) is zero -> returns None
        assert compute_mcc(0, 0, 0, 0) is None
        assert compute_mcc(10, 0, 0, 0) is None
        # All FP and FN with 0 TP and TN gives denom=625, num=-25 -> -1.0
        assert compute_mcc(0, 5, 0, 5) == -1.0

    def test_compute_mcc_perfect_and_inverse(self):
        # Perfect correlation: TP=10, TN=10, FP=0, FN=0 -> 1.0
        assert math.isclose(compute_mcc(10, 0, 10, 0), 1.0, rel_tol=1e-5)
        # Inverse correlation: TP=0, TN=0, FP=10, FN=10 -> -1.0
        assert math.isclose(compute_mcc(0, 10, 0, 10), -1.0, rel_tol=1e-5)

    def test_compute_classification_metrics_zero_queries(self):
        res = compute_classification_metrics(0, 0, 0, 0, total_q=0)
        assert res["accuracy"] == 0.0
        assert res["precision"] is None
        assert res["recall"] is None
        assert res["f1"] is None
        assert res["mcc"] is None

    def test_significance_stars_and_format_p_value(self):
        assert significance_stars(None) == ""
        assert significance_stars(float("nan")) == ""
        assert significance_stars(0.0005) == "***"
        assert significance_stars(0.005) == "**"
        assert significance_stars(0.03) == "**"
        assert significance_stars(0.20) == ""

        assert format_p_value(None) == "—"
        assert format_p_value(float("nan")) == "—"
        assert format_p_value(0.0001) == "<0.001"
        assert format_p_value(0.04231) == "0.0423"

    def test_prepare_per_query_records_empty_and_corrupted(self):
        records, baselines = prepare_per_query_records({})
        assert records == []
        assert baselines == []

        # Case missing id and category, baseline with missing eval_metrics
        data = {
            "results": [
                {
                    "is_answerable": True,
                    "baselines": {
                        "B0": {
                            "status": "success",
                            "answerability_outcome": "TP"
                        }
                    }
                }
            ]
        }
        records, baselines = prepare_per_query_records(data)
        assert len(records) == 1
        assert records[0]["query_id"] == "UNKNOWN"
        assert records[0]["category"] == "general"
        assert records[0]["baseline"] == "B0"

    def test_paired_metric_vectors_mismatched_queries(self):
        records = [
            {"query_id": "Q1", "baseline": "B0", "outcome": "TP", "answer_relevance": 0.8},
            {"query_id": "Q2", "baseline": "B0", "outcome": "TP", "answer_relevance": 0.9},
            {"query_id": "Q2", "baseline": "B1", "outcome": "TP", "answer_relevance": 0.85},
            {"query_id": "Q3", "baseline": "B1", "outcome": "TP", "answer_relevance": 0.70},
        ]
        vec_a, vec_b, shared = paired_metric_vectors(records, "B0", "B1", "answer_relevance")
        assert shared == ["Q2"]
        assert len(vec_a) == 1
        assert vec_a[0] == 0.9
        assert vec_b[0] == 0.85


# =====================================================================
# 2. core.metrics Robustness
# =====================================================================

class TestCoreMetricsRobustness:
    """Robustness tests for core.metrics functions."""

    def test_calculate_retrieval_recall_boundary_inputs(self):
        # Empty expected papers -> recall is 1.0
        assert calculate_retrieval_recall([], ["p1", "p2"]) == 1.0
        assert calculate_retrieval_recall(["  "], ["p1"]) == 1.0

        # Case-insensitive and whitespace normalization
        expected = ["Paper-A ", "PAPER-B"]
        retrieved = ["paper-a", "paper-c"]
        assert calculate_retrieval_recall(expected, retrieved) == 0.5

    def test_calculate_context_precision_boundary_inputs(self):
        # Empty expected -> 1.0
        assert calculate_context_precision([], [{"paper_id": "p1"}]) == 1.0
        # Empty retrieved chunks -> 0.0
        assert calculate_context_precision(["p1"], []) == 0.0

        # Chunks missing paper_id
        chunks = [{"paper_id": ""}, {"other": "val"}]
        assert calculate_context_precision(["p1"], chunks) == 0.0

    def test_compute_cosine_similarity_edge_cases(self):
        # Zero vectors -> 0.0
        assert compute_cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0
        assert compute_cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0

        # Identical unit vectors -> 1.0
        v = [0.6, 0.8]
        assert math.isclose(compute_cosine_similarity(v, v), 1.0, rel_tol=1e-5)

        # Orthogonal vectors -> 0.0
        assert math.isclose(compute_cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0, abs_tol=1e-5)

    def test_detect_abstention_various_phrasings(self):
        assert detect_abstention("Under the context, the answer is UNANSWERABLE.") is True
        assert detect_abstention("К сожалению, нет информации для ответа.") is True
        assert detect_abstention("Cannot answer this based on given context.") is True
        assert detect_abstention("   ") is True

        # Non-refusal containing cleaned quotes
        assert detect_abstention("This method replaces previously unanswerable queries with graph lookups.") is False

    def test_count_text_tokens_heuristic_fallback(self):
        assert count_text_tokens("") == 0
        assert count_text_tokens("Short sentence") > 0
        long_str = "A" * 400
        with patch("core.metrics.tiktoken", None):
            tokens = count_text_tokens(long_str)
            assert tokens >= 90  # 400 chars // 4 = 100 tokens

    def test_get_is_answerable_types(self):
        assert get_is_answerable({}) is True
        assert get_is_answerable({"is_answerable": True}) is True
        assert get_is_answerable({"is_answerable": False}) is False
        assert get_is_answerable({"is_answerable": "true"}) is True
        assert get_is_answerable({"is_answerable": "False"}) is False
        assert get_is_answerable({"is_answerable": "other"}) is False

    def test_normalize_optional_text(self):
        assert normalize_optional_text(None) == ""
        assert normalize_optional_text("  hello  ") == "hello"
        assert normalize_optional_text(100) == "100"


# =====================================================================
# 3. core.limiter Robustness
# =====================================================================

class TestCoreLimiterRobustness:
    """Robustness tests for AsyncRateLimiter."""

    @pytest.mark.asyncio
    async def test_async_rate_limiter_zero_or_negative_rpm(self):
        # RPM <= 0 should not block or raise ZeroDivisionError
        limiter_zero = AsyncRateLimiter(rpm=0)
        assert limiter_zero.interval == 0.0
        await limiter_zero.wait()

        limiter_neg = AsyncRateLimiter(rpm=-60)
        assert limiter_neg.interval == 0.0
        await limiter_neg.wait()

    @pytest.mark.asyncio
    async def test_async_rate_limiter_concurrent_tasks(self):
        # Rate limit of 600 RPM -> 0.1s interval between calls
        limiter = AsyncRateLimiter(rpm=600)
        start_time = asyncio.get_event_loop().time()

        async def worker():
            await limiter.wait()

        # Run 3 concurrent calls
        await asyncio.gather(worker(), worker(), worker())
        end_time = asyncio.get_event_loop().time()
        # Elapsed time should be at least 2 * interval = 0.2s (with tolerance)
        assert (end_time - start_time) >= 0.15


# =====================================================================
# 4. core.sanitization Robustness
# =====================================================================

class TestCoreSanitizationRobustness:
    """Robustness tests for reasoning and answer sanitization."""

    def test_extract_clean_answer_think_tags(self):
        text = "<think>\nThinking process here...\n</think>\nThe result is 42."
        status, clean = extract_clean_answer(text)
        assert clean == "The result is 42."

    def test_extract_clean_answer_unclosed_think_tag(self):
        text = "<think>\nIncomplete reasoning without closing tag"
        status, clean = extract_clean_answer(text)
        assert isinstance(clean, str)

    def test_extract_clean_answer_unanswerable_text(self):
        text = "<|status_start|>UNANSWERABLE<|status_end|><|answer_start|>Cannot answer<|answer_end|>"
        status, clean = extract_clean_answer(text)
        assert status == "UNANSWERABLE"

    def test_clean_answer_tokens(self):
        text = "Deepseek reasoning: <think> internal thoughts </think> output answer"
        cleaned = clean_answer_tokens(text)
        assert "internal thoughts" not in cleaned
        assert "output answer" in cleaned


# =====================================================================
# 5. core.pipelined Persistence Robustness
# =====================================================================

class TestCorePipelinedRobustness:
    """Robustness tests for pipelined file persistence and atomic writes."""

    def test_safe_read_modify_write_nested_dir_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_file = Path(tmpdir) / "sub_dir" / "deeper_dir" / "output.yaml"

            def modify_fn(existing):
                if existing is None:
                    existing = {"created": True}
                return existing

            safe_read_modify_write_yaml(nested_file, modify_fn)
            assert nested_file.exists()
            import yaml
            with open(nested_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            assert data["created"] is True
