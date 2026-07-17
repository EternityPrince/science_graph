"""
Science Graph — Benchmark Robustness Test Suite.

Tests system behavior under edge cases, malformed LLM outputs, corrupted inputs,
extreme score distributions, asymmetric datasets, concurrent file access, and boundary conditions.
"""

import os
import json
import yaml
import math
import tempfile
import asyncio
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.evaluator import CloudEvaluator, evaluate_baseline_case, build_context_string
from core.analytics import analyze_metrics, QUALITY_METRICS
from core.shannon_estimator import compute_rank_entropy, compute_lexical_entropy
from core.pipelined import safe_read_modify_write_yaml, save_evaluation_baseline_result
from base_sweeper import BaseHyperparameterSweeper


# =====================================================================
# 1. Cloud Evaluator & JSON Parsing Robustness
# =====================================================================

class TestEvaluatorParsingRobustness:
    """Tests LLM response cleaning and JSON parsing under noisy and invalid outputs."""

    @pytest.fixture
    def mock_evaluator(self):
        with patch("openai.AsyncOpenAI"):
            evaluator = CloudEvaluator(
                api_key="mock_key",
                base_url="http://mock.url",
                model_name="mock-model",
                concurrency=2,
                rpm=60
            )
            return evaluator

    def test_clean_and_parse_json_valid_markdown(self, mock_evaluator):
        text = "Here is the result:\n```json\n{\"answer_relevance\": {\"score\": 0.95}}\n```"
        parsed = mock_evaluator.clean_and_parse_json(text)
        assert parsed == {"answer_relevance": {"score": 0.95}}

    def test_clean_and_parse_json_raw_json_with_surrounding_text(self, mock_evaluator):
        text = "Prefix text {\"answer_relevance\": {\"score\": 0.8}, \"semantic_accuracy\": {\"score\": 0.7}} Suffix text"
        parsed = mock_evaluator.clean_and_parse_json(text)
        assert parsed["answer_relevance"]["score"] == 0.8
        assert parsed["semantic_accuracy"]["score"] == 0.7

    def test_clean_and_parse_json_malformed_raises_json_decode_error(self, mock_evaluator):
        text = "This is not json at all"
        with pytest.raises(json.JSONDecodeError):
            mock_evaluator.clean_and_parse_json(text)

    def test_clean_and_parse_json_nested_codeblocks(self, mock_evaluator):
        text = "```\n{\n  \"faithfulness\": {\"score\": 1.0}\n}\n```"
        parsed = mock_evaluator.clean_and_parse_json(text)
        assert parsed["faithfulness"]["score"] == 1.0

    @pytest.mark.asyncio
    async def test_evaluate_all_metrics_missing_keys_fallback(self, mock_evaluator):
        # LLM returns valid JSON but missing required keys
        incomplete_json = json.dumps({"answer_relevance": {"score": 0.9}})
        mock_evaluator.call_llm = AsyncMock(return_value=incomplete_json)

        result = await mock_evaluator.evaluate_all_metrics(
            evaluator_config={"system_prompt": "sys", "user_prompt_template": "user {query}"},
            has_context=True,
            query="Q"
        )
        # Should return fallback dict with 0.0 scores for all required keys
        assert result["answer_relevance"]["score"] == 0.0
        assert "error" in result["answer_relevance"]
        assert result["faithfulness"]["score"] == 0.0
        assert result["citation_fidelity"]["score"] == 0.0

    @pytest.mark.asyncio
    async def test_evaluate_all_metrics_numeric_type_coercion(self, mock_evaluator):
        # LLM returns plain numbers instead of metric dicts
        plain_number_json = json.dumps({
            "answer_relevance": 0.85,
            "semantic_accuracy": 0.9,
            "faithfulness": 0.75,
            "citation_fidelity": 0.8
        })
        mock_evaluator.call_llm = AsyncMock(return_value=plain_number_json)

        result = await mock_evaluator.evaluate_all_metrics(
            evaluator_config={"system_prompt": "sys", "user_prompt_template": "user {query}"},
            has_context=True,
            query="Q"
        )
        assert result["answer_relevance"] == {"score": 0.85}
        assert result["semantic_accuracy"] == {"score": 0.9}
        assert result["faithfulness"] == {"score": 0.75}

    @pytest.mark.asyncio
    async def test_evaluate_all_metrics_invalid_data_type(self, mock_evaluator):
        # LLM returns string instead of dict or number
        invalid_type_json = json.dumps({
            "answer_relevance": "invalid_string",
            "semantic_accuracy": 0.9,
            "faithfulness": 0.75,
            "citation_fidelity": 0.8
        })
        mock_evaluator.call_llm = AsyncMock(return_value=invalid_type_json)

        result = await mock_evaluator.evaluate_all_metrics(
            evaluator_config={"system_prompt": "sys", "user_prompt_template": "user {query}"},
            has_context=True,
            query="Q"
        )
        assert result["answer_relevance"]["score"] == 0.0
        assert "error" in result["answer_relevance"]


# =====================================================================
# 2. Case Evaluation & Context Building Robustness
# =====================================================================

class TestEvaluatorCaseRobustness:
    """Tests evaluation runner resilience against empty or malformed inputs."""

    def test_build_context_string_empty_and_corrupted_chunks(self):
        # Empty list
        assert build_context_string([]) == "Context is empty."

        # Chunks missing optional fields
        chunks = [
            {"text_content": "Chunk text only"},
            {"paper_id": "P10", "page_number": 3},
            {"paper_id": "P20", "page_number": 5, "text_content": "  Valid text  "}
        ]
        ctx = build_context_string(chunks)
        assert "Paper: Unknown" in ctx
        assert "Page: Unknown" in ctx
        assert "Paper: P20, Page: 5" in ctx
        assert "Valid text" in ctx

    @pytest.mark.asyncio
    async def test_evaluate_baseline_case_failed_status(self):
        evaluator = AsyncMock()
        prompts = {"unified_with_context_evaluator": {"system_prompt": "sys", "user_prompt_template": "user"}}

        with tempfile.TemporaryDirectory() as tmpdir:
            chk = Path(tmpdir) / "checkpoint.json"
            res = await evaluate_baseline_case(
                evaluator=evaluator,
                prompts=prompts,
                case_id="C01",
                query="Query",
                golden_answer="Golden",
                expected_papers=["P1"],
                baseline_name="B0",
                baseline_data={
                    "status": "failed",
                    "error": "Timeout",
                    "generated_answer": "",
                    "retrieved_chunks": []
                },
                checkpoint_data={},
                checkpoint_path=chk,
                is_answerable=True
            )
            # Evaluator shouldn't be called for failed baseline
            assert evaluator.evaluate_all_metrics.called is False
            assert res["answer_relevance"] == 0.0
            assert res["retrieval_recall"] == 0.0
            assert res["status"] == "failed"

    @pytest.mark.asyncio
    async def test_evaluate_baseline_case_unanswerable_skips_judge(self):
        evaluator = AsyncMock()
        prompts = {"unified_with_context_evaluator": {"system_prompt": "sys", "user_prompt_template": "user"}}

        with tempfile.TemporaryDirectory() as tmpdir:
            chk = Path(tmpdir) / "checkpoint.json"
            res = await evaluate_baseline_case(
                evaluator=evaluator,
                prompts=prompts,
                case_id="C02",
                query="Query",
                golden_answer="",
                expected_papers=[],
                baseline_name="B0",
                baseline_data={
                    "status": "success",
                    "generated_answer": "Not enough info.",
                    "retrieved_chunks": []
                },
                checkpoint_data={},
                checkpoint_path=chk,
                is_answerable=False
            )
            # LLM judge should be skipped for unanswerable question
            assert evaluator.evaluate_all_metrics.called is False
            assert res["answerability_outcome"] == "TN"  # Unanswerable correctly abstained


# =====================================================================
# 3. Analytics & Statistical Aggregation Robustness
# =====================================================================

class TestAnalyticsRobustness:
    """Tests metrics calculation and reporting analytics under zero or edge inputs."""

    def test_analyze_metrics_empty_report(self):
        empty_data = {"metadata": {}, "results": []}
        with pytest.raises(ValueError, match="No results found"):
            analyze_metrics(empty_data)

    def test_analyze_metrics_single_case(self):
        data = {
            "metadata": {},
            "results": [
                {
                    "id": "Q1",
                    "query": "What is X?",
                    "is_answerable": True,
                    "baselines": {
                        "B0": {
                            "status": "success",
                            "answer_relevance": 0.8,
                            "semantic_accuracy": 0.9,
                            "retrieval_recall": 1.0,
                            "latency_sec": 1.2,
                            "answerability_outcome": "TP"
                        }
                    }
                }
            ]
        }
        summary = analyze_metrics(data)
        assert summary["total_queries"] == 1
        assert "B0" in summary["summary"]
        b0_stats = summary["summary"]["B0"]
        assert b0_stats["answer_relevance"]["mean"] == 0.8

    def test_analyze_metrics_none_and_nan_resilience(self):
        data = {
            "metadata": {},
            "results": [
                {
                    "id": "Q1",
                    "query": "Q1",
                    "is_answerable": True,
                    "baselines": {
                        "B1": {
                            "status": "success",
                            "answer_relevance": None,
                            "semantic_accuracy": 0.8,
                            "retrieval_recall": float("nan"),
                            "answerability_outcome": "TP"
                        }
                    }
                },
                {
                    "id": "Q2",
                    "query": "Q2",
                    "is_answerable": True,
                    "baselines": {
                        "B1": {
                            "status": "success",
                            "answer_relevance": 0.6,
                            "semantic_accuracy": None,
                            "retrieval_recall": 0.5,
                            "answerability_outcome": "TP"
                        }
                    }
                }
            ]
        }
        summary = analyze_metrics(data)
        b1_stats = summary["summary"]["B1"]
        assert b1_stats["answer_relevance"]["mean"] == 0.6
        assert b1_stats["semantic_accuracy"]["mean"] == 0.8
        assert b1_stats["retrieval_recall"]["mean"] == 0.5

    def test_analyze_metrics_confusion_matrix_zero_division(self):
        # All queries are unanswerable, and system answered all (all FP)
        data = {
            "metadata": {},
            "results": [
                {
                    "id": "Q1",
                    "query": "Q1",
                    "is_answerable": False,
                    "baselines": {"B0": {"status": "success", "answerability_outcome": "FP"}}
                },
                {
                    "id": "Q2",
                    "query": "Q2",
                    "is_answerable": False,
                    "baselines": {"B0": {"status": "success", "answerability_outcome": "FP"}}
                }
            ]
        }
        summary = analyze_metrics(data)
        b0 = summary["summary"]["B0"]["classification"]
        assert b0["FP"] == 2
        assert b0["TP"] == 0
        assert b0["FN"] == 0
        assert b0["TN"] == 0
        assert b0["precision"] == 0.0
        assert b0["recall"] is None
        assert b0["f1"] == 0.0


# =====================================================================
# 4. Shannon Estimator Robustness
# =====================================================================

class TestShannonEstimatorRobustness:
    """Tests mathematical entropy calculations under boundary conditions."""

    def test_compute_rank_entropy_empty_and_single(self):
        assert compute_rank_entropy([]) == 0.0
        assert compute_rank_entropy([0.5]) == 0.0

    def test_compute_rank_entropy_negative_and_zero_temperatures(self):
        scores = [0.1, 0.5, 0.9]
        # Temperature tau <= 0 should be safely clamped to positive small value
        entropy_tau_zero = compute_rank_entropy(scores, method="softmax", tau=0.0)
        assert isinstance(entropy_tau_zero, float)
        assert entropy_tau_zero >= 0.0

        entropy_tau_neg = compute_rank_entropy(scores, method="softmax", tau=-1.0)
        assert isinstance(entropy_tau_neg, float)
        assert entropy_tau_neg >= 0.0

    def test_compute_rank_entropy_identical_scores(self):
        # Uniform distribution should produce log2(N) bits of entropy
        scores = [1.0, 1.0, 1.0, 1.0]
        entropy = compute_rank_entropy(scores, method="softmax", tau=1.0)
        # log2(4) = 2.0 bits
        assert math.isclose(entropy, 2.0, rel_tol=1e-4)

    def test_compute_rank_entropy_minmax_zero_range(self):
        scores = [5.0, 5.0, 5.0]
        entropy = compute_rank_entropy(scores, method="minmax")
        # All equal in minmax should default to uniform -> log2(3) approx 1.5849
        assert math.isclose(entropy, math.log2(3), rel_tol=1e-4)

    def test_compute_lexical_entropy_edge_cases(self):
        assert compute_lexical_entropy("") == 0.0
        assert compute_lexical_entropy("   \n\t  ") == 0.0
        assert compute_lexical_entropy("!!! ??? ***") == 0.0
        
        # Single repeating word -> 0 entropy
        assert compute_lexical_entropy("test test test test") == 0.0
        
        # Two unique words equal count -> 1.0 bit
        assert compute_lexical_entropy("alpha beta alpha beta") == 1.0


# =====================================================================
# 5. Concurrent File Persistence & Atomic Writes Robustness
# =====================================================================

class TestPersistenceRobustness:
    """Tests file locking and read-modify-write safety under concurrent writes."""

    def test_concurrent_yaml_writes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target_file = Path(tmpdir) / "concurrent_results.yaml"
            num_threads = 10
            writes_per_thread = 5

            def worker(thread_idx):
                for i in range(writes_per_thread):
                    case_id = f"Q_{thread_idx}_{i}"
                    save_evaluation_baseline_result(
                        file_path=target_file,
                        case_id=case_id,
                        case_info={"category": "test", "query": f"Q {case_id}"},
                        baseline_name="B0",
                        baseline_data={"status": "success"},
                        eval_metrics_raw={"answer_relevance": 0.9},
                        metadata={"run_id": "test_run"}
                    )

            threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert target_file.exists()
            with open(target_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            assert len(data["results"]) == num_threads * writes_per_thread

    def test_safe_read_modify_write_corrupted_file_recovery(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target_file = Path(tmpdir) / "corrupted.yaml"
            # Write invalid non-YAML syntax with unclosed bracket
            with open(target_file, "w", encoding="utf-8") as f:
                f.write("invalid: [unclosed list")

            def modify_fn(existing):
                if not isinstance(existing, dict):
                    existing = {"recovered": True}
                existing["count"] = 1
                return existing

            safe_read_modify_write_yaml(target_file, modify_fn)
            with open(target_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            assert data["recovered"] is True
            assert data["count"] == 1


# =====================================================================
# 6. Hyperparameter Sweeper Robustness
# =====================================================================

class DummySweeper(BaseHyperparameterSweeper):
    def __init__(self, runs_list, **kwargs):
        super().__init__(**kwargs)
        self._runs = runs_list

    def get_runs(self):
        return self._runs


class TestSweeperRobustness:
    """Tests hyperparameter sweeper configuration handling."""

    def test_sweeper_empty_runs(self):
        sweeper = DummySweeper(runs_list=[])
        results = sweeper.run_sweep()
        assert results == []

    def test_sweeper_run_with_extreme_hyperparameters(self):
        runs = [
            {
                "name": "extreme_run",
                "components": {"reranker": True},
                "hyperparameters": {
                    "rag": {
                        "score_blend_reranker_weight": -1.0,
                        "score_blend_rrf_weight": 999.0
                    }
                }
            }
        ]
        sweeper = DummySweeper(runs_list=runs)
        configs = sweeper.get_runs()
        assert len(configs) == 1
        assert configs[0]["hyperparameters"]["rag"]["score_blend_reranker_weight"] == -1.0
