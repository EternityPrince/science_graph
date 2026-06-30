import pytest
from unittest.mock import patch
from core.analytics import analyze_metrics
from core.models import ReportOutput, TestCaseOutput, BaselineOutput

def test_analyze_metrics_empty_results():
    data = {"results": []}
    with pytest.raises(ValueError, match="No results found in the benchmarking data"):
        analyze_metrics(data)

def test_analyze_metrics_report_output_instance():
    report = ReportOutput(
        metadata={"version": "1.0"},
        summary={"total_cases": 1},
        results=[
            TestCaseOutput(
                id="q1",
                query="Query 1",
                category="test",
                baselines={
                    "B1": BaselineOutput(
                        status="success",
                        latency_sec=1.5,
                        eval_metrics={"retrieval_recall": 1.0, "semantic_accuracy": 0.8}
                    )
                }
            )
        ]
    )
    res = analyze_metrics(report)
    assert res["total_queries"] == 1
    assert "B1" in res["summary"]

def test_analyze_metrics_baselines_in_subsequent_results():
    data = {
        "results": [
            {
                "id": "q1",
                "query": "Q1",
                "category": "cat1",
                "baselines": {}
            },
            {
                "id": "q2",
                "query": "Q2",
                "category": "cat1",
                "baselines": {
                    "B1": {
                        "status": "success",
                        "latency_sec": 1.0,
                        "eval_metrics": {"retrieval_recall": 0.8, "semantic_accuracy": 0.9}
                    }
                }
            }
        ]
    }
    res = analyze_metrics(data)
    assert res["baselines"] == ["B1"]

def test_analyze_metrics_missing_baseline_data():
    data = {
        "results": [
            {
                "id": "q1",
                "query": "Q1",
                "category": "cat1",
                "baselines": {
                    "B1": {
                        "status": "success",
                        "latency_sec": 1.0,
                        "eval_metrics": {"retrieval_recall": 0.8}
                    },
                    "B2": {
                        "status": "success",
                        "latency_sec": 1.0,
                        "eval_metrics": {"retrieval_recall": 0.8}
                    }
                }
            },
            {
                "id": "q2",
                "query": "Q2",
                "category": "cat1",
                "baselines": {
                    # B1 is missing entirely for q2, triggering continue in loop
                    "B2": {
                        "status": "success",
                        "latency_sec": 1.0,
                        "eval_metrics": {"retrieval_recall": 0.8}
                    }
                }
            }
        ]
    }
    res = analyze_metrics(data)
    assert "B2" in res["summary"]
    assert "B1" in res["summary"]

@patch("core.analytics.calculate_semantic_accuracy", return_value=[0.75])
def test_analyze_metrics_missing_semantic_accuracy(mock_calc_accuracy):
    data = {
        "results": [
            {
                "id": "q1",
                "query": "Q1",
                "golden_answer": "Expected Gold Answer",
                "category": "cat1",
                "baselines": {
                    "B1": {
                        "status": "success",
                        "generated_answer": "Generated: Expected Gold Answer",
                        "latency_sec": 1.0,
                        # semantic_accuracy missing
                    }
                }
            }
        ]
    }
    res = analyze_metrics(data)
    assert res["summary"]["B1"]["semantic_accuracy"]["mean"] == 0.75
    mock_calc_accuracy.assert_called_once()

def test_analyze_metrics_max_input_token_fallback():
    # Make config import fail to trigger fallback to 4096
    orig_import = __import__
    def mock_import(name, *args, **kwargs):
        if name.startswith("src"):
            raise ImportError("mock import error")
        return orig_import(name, *args, **kwargs)
        
    with patch("builtins.__import__", side_effect=mock_import):
        data = {
            "results": [
                {
                    "id": "q1",
                    "query": "Q1",
                    "category": "cat1",
                    "baselines": {
                        "B1": {
                            "status": "success",
                            "latency_sec": 1.0,
                            "eval_metrics": {}
                        }
                    }
                }
            ]
        }
        res = analyze_metrics(data)
        # Should fallback to 4096 context. context_fillness should be calculated.
        assert "B1" in res["summary"]

def test_analyze_metrics_in_place_dict_update():
    data = {
        "metadata": {"version": "1.0"},
        "results": [
            {
                "id": "q1",
                "query": "Q1",
                "category": "cat1",
                "baselines": {
                    "B1": {
                        "status": "success",
                        "latency_sec": 1.0,
                        "eval_metrics": {}
                    }
                }
            }
        ]
    }
    res = analyze_metrics(data)
    # Check that in-place updates were applied
    assert "eval_metrics" in data["results"][0]["baselines"]["B1"]
    assert "retrieval_recall" in data["results"][0]["baselines"]["B1"]["eval_metrics"]

def test_analyze_metrics_pairwise_win_rates():
    # B1 beats B2 in retrieval_recall, but B2 beats B1 in latency_sec
    data = {
        "results": [
            {
                "id": "q1",
                "query": "Q1",
                "category": "cat1",
                "baselines": {
                    "B1": {
                        "status": "success",
                        "latency_sec": 2.0,
                        "eval_metrics": {"retrieval_recall": 1.0, "semantic_accuracy": 0.8}
                    },
                    "B2": {
                        "status": "success",
                        "latency_sec": 1.0,
                        "eval_metrics": {"retrieval_recall": 0.5, "semantic_accuracy": 0.8}
                    }
                }
            }
        ]
    }
    res = analyze_metrics(data)
    # Win rates check
    # B1 retrieval_recall win rate against B2 is 100%
    assert res["pairwise_win_rates"]["retrieval_recall"]["B1"]["B2"] == 100.0
    # B2 latency_sec win rate against B1 is 100%
    assert res["pairwise_win_rates"]["latency_sec"]["B2"]["B1"] == 100.0
