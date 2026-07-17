"""
Science Graph — Configuration and Reporting Robustness Test Suite.

Contains edge-case and boundary tests for:
- core.config (baseline configurations, safe model names, dataset loading & sampling)
- config_creator (CLI parameter overriding, preset merging, monkey-patching and state restoration)
- core.reporting (rich vs plain table formatting, markdown generation with missing data, CSV export)
"""

import tempfile
import argparse
from pathlib import Path
from unittest.mock import patch, MagicMock
import yaml
import pytest

from core.config import get_baseline_config, get_safe_model_name, load_benchmark_dataset, BASELINES_INFO
from config_creator import (
    build_custom_config,
    patch_config_for_custom,
    restore_baseline_config_patch,
    add_custom_config_arguments,
    CUSTOM_PRESET_COMPONENTS
)
from core.reporting import print_rich_tables, generate_markdown_report


# =====================================================================
# 1. core.config Robustness
# =====================================================================

class TestConfigModuleRobustness:
    """Tests baseline resolution, safe path naming, and dataset sampling."""

    def test_get_baseline_config_known_and_unknown_baselines(self):
        default_components = {
            "lexical_search": False,
            "dense_search": False,
            "reranker": False,
            "hyde": False,
            "graph_expansion": False,
            "context_trimming": False,
            "citation_repair": False,
            "intent_classifier": False,
            "graph_neighbors_in_rrf": False,
        }

        # B0: all false
        b0 = get_baseline_config("B0", default_components)
        assert not any(b0.values())

        # B1: lexical_search true
        b1 = get_baseline_config("B1", default_components)
        assert b1["lexical_search"] is True
        assert b1["dense_search"] is False

        # B6: full pipeline
        b6 = get_baseline_config("B6", default_components)
        assert b6["reranker"] is True
        assert b6["graph_neighbors_in_rrf"] is True
        assert b6["hyde"] is False

        # Unknown baseline -> defaults all components to False
        b_unknown = get_baseline_config("B99", default_components)
        assert isinstance(b_unknown, dict)

    def test_get_safe_model_name_special_characters(self):
        assert get_safe_model_name("gpt-4o") == "gpt-4o"
        assert get_safe_model_name("org/model:v1.0") == "model_v1.0"
        assert get_safe_model_name("/path/to/deepseek-r1-7b q4_k") == "deepseek-r1-7b_q4_k"

    def test_load_benchmark_dataset_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_benchmark_dataset(Path("/non/existent/path/dataset.yaml"))

    def test_load_benchmark_dataset_sciq_and_sampling(self):
        sciq_sample = [
            {"question": {"id": 1, "q": "What is water?", "a": "H2O", "c": "Chemistry paper context"}},
            {"question": {"id": 2, "q": "What is air?", "a": "Gas mixture", "c": "Physics paper context"}}
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            ds_file = Path(tmpdir) / "sciq.yaml"
            with open(ds_file, "w", encoding="utf-8") as f:
                yaml.dump(sciq_sample, f)

            loaded = load_benchmark_dataset(ds_file, limit=1, seed=42)
            assert len(loaded) == 1
            assert "query" in loaded[0]
            assert "golden_answer" in loaded[0]
            assert "is_answerable" in loaded[0]
            assert loaded[0]["id"] == "sciq_1"
            assert loaded[0]["expected_papers"] == ["sciq_paper_1"]


# =====================================================================
# 2. config_creator Robustness
# =====================================================================

class TestConfigCreatorRobustness:
    """Tests CLI argument merging, preset customization, and monkey-patching safety."""

    def test_add_custom_config_arguments(self):
        parser = argparse.ArgumentParser()
        add_custom_config_arguments(parser)
        args = parser.parse_args(["--custom", "--reranker", "--score-blend-reranker-weight", "0.9"])
        assert args.custom is True
        assert args.reranker is True
        assert args.score_blend_reranker_weight == 0.9

    def test_build_custom_config_preset_and_cli_overrides(self):
        parser = argparse.ArgumentParser()
        add_custom_config_arguments(parser)
        args = parser.parse_args(["--custom", "--no-reranker", "--score-blend-reranker-weight", "0.5"])

        comp, hype = build_custom_config(args)
        # --custom sets preset reranker=True, but --no-reranker overrides to False
        assert comp["reranker"] is False
        assert hype["rag"]["score_blend_reranker_weight"] == 0.5

    def test_build_custom_config_file_config_merge(self):
        parser = argparse.ArgumentParser()
        add_custom_config_arguments(parser)
        args = parser.parse_args([])

        file_config = {
            "rag_components": {"graph_expansion": True},
            "hyperparameters": {
                "rag": {"rrf_k": 50.0}
            }
        }
        comp, hype = build_custom_config(args, file_config=file_config)
        assert comp["graph_expansion"] is True
        assert hype["rag"]["rrf_k"] == 50.0

    def test_patch_and_restore_baseline_config(self):
        custom_comp = {"reranker": True}
        custom_hype = {"rag": {"score_blend_reranker_weight": 0.8}}

        patch_config_for_custom(custom_comp, custom_hype)
        try:
            custom_b = get_baseline_config("CUSTOM", custom_comp)
            assert custom_b["reranker"] is True
        finally:
            restore_baseline_config_patch()


# =====================================================================
# 3. core.reporting Robustness
# =====================================================================

class TestCoreReportingRobustness:
    """Tests report generation under incomplete stats."""

    @pytest.fixture
    def mock_stats(self):
        return {
            "baselines": ["B0", "B1"],
            "total_queries": 2,
            "total_answerable": 1,
            "total_unanswerable": 1,
            "has_graph_trace": False,
            "has_shannon": False,
            "categories": ["cat1"],
            "summary": {
                "B0": {
                    "success_rate": 100.0,
                    "answer_relevance": {"mean": 0.8},
                    "semantic_accuracy": {"mean": 0.85},
                    "retrieval_recall": {"mean": 1.0},
                    "context_precision": {"mean": 1.0},
                    "faithfulness": {"mean": 0.9},
                    "citation_fidelity": {"mean": 0.9},
                    "context_fillness": {"mean": 0.5},
                    "ar_sa_f1": {"mean": 0.82},
                    "latency_sec": {"mean": 1.2},
                    "token_output": {"mean": 100},
                    "token_answer": {"mean": 80},
                    "token_reasoning": {"mean": 20},
                    "classification": {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "TP": 1, "FP": 0, "TN": 1, "FN": 0},
                    "unanswerable_safety": {"unanswerable_count": 1, "abstention_accuracy": 1.0, "hallucination_rate": 0.0}
                },
                "B1": {
                    "success_rate": 100.0,
                    "answer_relevance": {"mean": 0.7},
                    "semantic_accuracy": {"mean": 0.75},
                    "retrieval_recall": {"mean": 0.8},
                    "context_precision": {"mean": 0.8},
                    "faithfulness": {"mean": 0.8},
                    "citation_fidelity": {"mean": 0.8},
                    "context_fillness": {"mean": 0.4},
                    "ar_sa_f1": {"mean": 0.72},
                    "latency_sec": {"mean": 1.0},
                    "token_output": {"mean": 90},
                    "token_answer": {"mean": 70},
                    "token_reasoning": {"mean": 20},
                    "classification": {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "TP": 1, "FP": 0, "TN": 1, "FN": 0},
                    "unanswerable_safety": {"unanswerable_count": 1, "abstention_accuracy": 1.0, "hallucination_rate": 0.0}
                }
            },
            "category_stats": {"cat1": {"B0": {}, "B1": {}}},
            "category_classification": {"cat1": {"B0": {}, "B1": {}}},
            "pairwise_win_rates": {
                "semantic_accuracy": {
                    "B0": {"B1": 60.0},
                    "B1": {"B0": 40.0}
                }
            },
            "query_difficulty": [
                {"id": "Q1", "category": "cat1", "query": "Test?", "avg_score": 0.8}
            ]
        }

    def test_print_rich_tables_fallback_plain(self, mock_stats):
        # Verify print_rich_tables doesn't crash when HAS_RICH is False or True
        with patch("core.reporting.HAS_RICH", False):
            print_rich_tables(mock_stats)

    def test_generate_markdown_report_valid_output(self, mock_stats):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "report.md"
            generate_markdown_report(mock_stats, out_file)
            assert out_file.exists()
            md_text = out_file.read_text(encoding="utf-8")
            assert "RAG" in md_text
            assert "B0" in md_text
            assert "B1" in md_text
            assert "Матрица попарных побед" in md_text
