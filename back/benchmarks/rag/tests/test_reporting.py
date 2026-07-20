import pytest
import csv
import yaml
from core.reporting import (
    print_rich_tables,
    print_plain_tables,
    generate_markdown_report,
    export_wide_csv,
    export_detailed_csv,
    save_judge_report,
    save_individual_judge_reports
)

@pytest.fixture
def dummy_stats():
    return {
        "total_queries": 2,
        "baselines": ["B0", "B1"],
        "summary": {
            "B0": {
                "success_rate": 100.0,
                "latency_sec": {"mean": 2.5, "min": 2.0, "max": 3.0, "median": 2.5, "stdev": 0.5, "count": 2},
                "retrieval_recall": {"mean": 0.0, "min": 0.0, "max": 0.0, "median": 0.0, "stdev": 0.0, "count": 2},
                "context_precision": {"mean": 0.0, "min": 0.0, "max": 0.0, "median": 0.0, "stdev": 0.0, "count": 2},
                "faithfulness": {"mean": 0.0, "min": 0.0, "max": 0.0, "median": 0.0, "stdev": 0.0, "count": 2},
                "answer_relevance": {"mean": 0.8, "min": 0.7, "max": 0.9, "median": 0.8, "stdev": 0.1, "count": 2},
                "citation_fidelity": {"mean": 0.0, "min": 0.0, "max": 0.0, "median": 0.0, "stdev": 0.0, "count": 2},
                "semantic_accuracy": {"mean": 0.75, "min": 0.7, "max": 0.8, "median": 0.75, "stdev": 0.05, "count": 2},
                "context_fillness": {"mean": 0.1, "min": 0.1, "max": 0.1, "median": 0.1, "stdev": 0.0, "count": 2},
                "token_output": {"mean": 100.0, "min": 90, "max": 110, "median": 100.0, "stdev": 10.0, "count": 2},
                "token_answer": {"mean": 80.0, "min": 75, "max": 85, "median": 80.0, "stdev": 5.0, "count": 2},
                "token_reasoning": {"mean": 20.0, "min": 15, "max": 25, "median": 20.0, "stdev": 5.0, "count": 2},
            },
            "B1": {
                "success_rate": 100.0,
                "latency_sec": {"mean": 3.5, "min": 3.0, "max": 4.0, "median": 3.5, "stdev": 0.5, "count": 2},
                "retrieval_recall": {"mean": 0.9, "min": 0.8, "max": 1.0, "median": 0.9, "stdev": 0.1, "count": 2},
                "context_precision": {"mean": 0.85, "min": 0.8, "max": 0.9, "median": 0.85, "stdev": 0.05, "count": 2},
                "faithfulness": {"mean": 0.95, "min": 0.9, "max": 1.0, "median": 0.95, "stdev": 0.05, "count": 2},
                "answer_relevance": {"mean": 0.85, "min": 0.8, "max": 0.9, "median": 0.85, "stdev": 0.05, "count": 2},
                "citation_fidelity": {"mean": 0.9, "min": 0.8, "max": 1.0, "median": 0.9, "stdev": 0.1, "count": 2},
                "semantic_accuracy": {"mean": 0.8, "min": 0.75, "max": 0.85, "median": 0.8, "stdev": 0.05, "count": 2},
                "context_fillness": {"mean": 0.15, "min": 0.12, "max": 0.18, "median": 0.15, "stdev": 0.03, "count": 2},
                "token_output": {"mean": 120.0, "min": 110, "max": 130, "median": 120.0, "stdev": 10.0, "count": 2},
                "token_answer": {"mean": 95.0, "min": 90, "max": 100, "median": 95.0, "stdev": 5.0, "count": 2},
                "token_reasoning": {"mean": 25.0, "min": 20, "max": 30, "median": 25.0, "stdev": 5.0, "count": 2},
            }
        },
        "categories": ["cat1", "cat2"],
        "category_stats": {
            "cat1": {
                "B0": {"semantic_accuracy": 0.7},
                "B1": {"semantic_accuracy": 0.78}
            },
            "cat2": {
                "B0": {"semantic_accuracy": 0.8},
                "B1": {"semantic_accuracy": 0.82}
            }
        },
        "pairwise_win_rates": {
            "semantic_accuracy": {
                "B0": {"B0": 0.0, "B1": 10.0},
                "B1": {"B0": 90.0, "B1": 0.0}
            }
        },
        "query_difficulty": [
            {"id": "q1", "category": "cat1", "query": "Query 1", "avg_score": 0.5},
            {"id": "q2", "category": "cat2", "query": "Query 2", "avg_score": 0.8}
        ]
    }

@pytest.fixture
def dummy_report_data():
    return {
        "results": [
            {
                "id": "q1",
                "category": "cat1",
                "query": "Query 1",
                "golden_answer": "Golden 1",
                "baselines": {
                    "B0": {
                        "status": "success",
                        "latency_sec": 2.2,
                        "generated_answer": "### Final Answer: Hello 0",
                        "eval_metrics": {
                            "retrieval_recall": 0.0,
                            "context_precision": 0.0,
                            "faithfulness": 0.0,
                            "answer_relevance": 0.75,
                            "citation_fidelity": 0.0,
                            "semantic_accuracy": 0.7,
                            "token_output": 100,
                            "token_answer": 80,
                            "token_reasoning": 20
                        }
                    },
                    "B1": {
                        "status": "success",
                        "latency_sec": 3.2,
                        "generated_answer": "### Final Answer: Hello 1",
                        "eval_metrics": {
                            "retrieval_recall": 0.85,
                            "context_precision": 0.8,
                            "faithfulness": 0.9,
                            "answer_relevance": 0.82,
                            "citation_fidelity": 0.85,
                            "semantic_accuracy": 0.78,
                            "token_output": 115,
                            "token_answer": 92,
                            "token_reasoning": 23
                        }
                    }
                }
            }
        ]
    }

def test_print_tables(dummy_stats):
    # Test printing with Rich (mocked/not mocked) and plain text
    print_rich_tables(dummy_stats)
    print_plain_tables(dummy_stats)

def test_generate_markdown_report(dummy_stats, tmp_path):
    report_path = tmp_path / "test_report.md"
    generate_markdown_report(dummy_stats, report_path)
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "RAG Benchmarking Report" in content
    assert "B0" in content
    assert "B1" in content
    assert "Token Output" in content

def test_export_wide_csv(dummy_stats, tmp_path):
    csv_path = tmp_path / "summary.csv"
    export_wide_csv(dummy_stats, csv_path)
    assert csv_path.exists()
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
    assert reader[0][0] == "Baseline"
    assert "Token Output" in reader[0]

def test_export_detailed_csv(dummy_report_data, dummy_stats, tmp_path):
    csv_path = tmp_path / "detailed.csv"
    export_detailed_csv(dummy_report_data, dummy_stats, csv_path)
    assert csv_path.exists()
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
    assert reader[0][0] == "query_id"
    assert "token_output" in reader[0]

def test_save_judge_reports(dummy_report_data, tmp_path):
    judge_path = tmp_path / "judge.yaml"
    save_judge_report(dummy_report_data, judge_path)
    assert judge_path.exists()
    with open(judge_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert len(data["results"]) == 1
    assert "generated_answer" in data["results"][0]["baselines"]["B0"]
    
    save_individual_judge_reports(dummy_report_data, tmp_path, "prefix", ".yaml")
    indiv_b0 = tmp_path / "baselines" / "prefix_judge_b0.yaml"
    assert indiv_b0.exists()
