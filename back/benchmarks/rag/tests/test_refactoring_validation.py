"""
Tests validating the four core evaluation pipeline refactoring fixes:
1. Graph metric contamination & leakage isolation for non-graph baselines.
2. State drift elimination between global and subcategory classification metrics.
3. Friedman test sample size preservation without list-wise query deletion.
4. Precision string formatting and standard rounding synchronization.
"""

import math
import pytest
from core.analytics import analyze_metrics, GRAPH_ENABLED_BASELINES
from core.statistics import friedman_omnibus_test
from parse_metrics import format_val
from core.reporting import format_pct, format_avg


def test_graph_metric_leakage_isolation():
    """Verify that non-graph baselines (e.g., B1, B2, B4) have isolated graph diagnostic fields."""
    mock_data = {
        "results": [
            {
                "id": "Q1",
                "query": "What is quantum mechanics?",
                "is_answerable": True,
                "category": "physics",
                "baselines": {
                    "B1": {"status": "success", "latency_sec": 0.5, "eval_metrics": {"semantic_accuracy": 0.8}},
                    "B5": {"status": "success", "latency_sec": 1.2, "eval_metrics": {"semantic_accuracy": 0.9}},
                    "B6": {"status": "success", "latency_sec": 1.5, "eval_metrics": {"semantic_accuracy": 0.95}},
                }
            }
        ]
    }
    # Trace map containing entries for all baselines (simulating potential leakage source)
    trace_map = {
        ("B1", "Q1"): {"graph_chunk_candidates_count": 10, "graph_retrieval_enabled": True},
        ("B5", "Q1"): {"graph_chunk_candidates_count": 5, "graph_retrieval_enabled": True},
        ("B6", "Q1"): {"graph_chunk_candidates_count": 8, "graph_retrieval_enabled": True},
    }

    stats = analyze_metrics(mock_data, trace_map=trace_map)
    b1_graph = stats["summary"]["B1"]["graph_diagnostics"]
    b5_graph = stats["summary"]["B5"]["graph_diagnostics"]

    # Non-graph baseline B1 must have 0 graph candidates despite trace_map entry
    assert b1_graph["avg_graph_chunk_candidates"] == 0.0
    assert b1_graph["enabled_rate"] == 0.0

    # Graph baseline B5 retains trace properties
    assert b5_graph["avg_graph_chunk_candidates"] == 5.0
    assert b5_graph["enabled_rate"] == 1.0


def test_subcategory_vs_global_classification_consistency():
    """Verify that subcategory and global classification metrics use identical outcome logic."""
    mock_data = {
        "results": [
            {
                "id": "Q1",
                "query": "Unanswerable query 1",
                "is_answerable": False,
                "category": "cat_a",
                "baselines": {
                    "B1": {
                        "status": "success",
                        "generated_answer": "В предоставленном контексте нет информации.",
                        "predicted_abstained": True,
                        "answerability_outcome": "TN"
                    }
                }
            },
            {
                "id": "Q2",
                "query": "Unanswerable query 2",
                "is_answerable": False,
                "category": "cat_a",
                "baselines": {
                    "B1": {
                        "status": "success",
                        "generated_answer": "This answer is hallucinated.",
                        "predicted_abstained": False,
                        "answerability_outcome": "FP"
                    }
                }
            }
        ]
    }

    stats = analyze_metrics(mock_data)
    global_safety = stats["summary"]["B1"]["unanswerable_safety"]
    cat_class = stats["category_classification"]["cat_a"]["B1"]

    assert global_safety["TN"] == cat_class["TN"] == 1
    assert global_safety["FP"] == cat_class["FP"] == 1
    assert global_safety["hallucination_rate"] == cat_class["hallucination_rate"] == 0.5
    assert global_safety["abstention_accuracy"] == cat_class["abstention_accuracy"] == 0.5


def test_friedman_omnibus_sample_size_preservation():
    """Verify that Friedman test does not drop sample size when quality metrics have missing/abstained values."""
    records = []
    baselines = ["B1", "B2", "B6"]
    # 50 answerable queries, where B1 abstains on 20 queries (giving None for semantic_accuracy)
    for i in range(50):
        qid = f"Q{i}"
        for b in baselines:
            is_ans = True
            rec = {
                "query_id": qid,
                "baseline": b,
                "is_answerable": is_ans,
                "outcome": "TN" if (b == "B1" and i < 20) else "TP",
                "semantic_accuracy": None if (b == "B1" and i < 20) else 0.85
            }
            records.append(rec)

    res = friedman_omnibus_test(records, baselines, metric="semantic_accuracy")
    # Sample size n should be 50 (all answerable queries), not 30
    assert res["n"] == 50
    assert res["statistic"] is not None


def test_precision_standard_rounding():
    """Verify that format_val and format_pct use standard round-to-nearest convention."""
    assert format_val(0.67666, is_pct=True) == "67.7%"
    assert format_val(0.67644, is_pct=True) == "67.6%"
    assert format_pct(0.67666) == "67.7%"
    assert format_pct(0.67644) == "67.6%"
    assert format_avg(3.14159) == "3.14"
