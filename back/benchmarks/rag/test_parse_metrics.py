import pytest
from pathlib import Path
from pydantic import ValidationError

from core.models import parse_report, ReportOutput, RetrievedChunk, BaselineOutput
from core.analytics import analyze_metrics


def test_retrieval_report_parsing():
    # Example format matching retrieved_contexts.yaml
    raw_data = [
        {
            "id": "Q1",
            "query": "What is Winograd?",
            "category": "single-document",
            "golden_answer": "Winograd is a method.",
            "expected_papers": ["paper_1"],
            "baselines": {
                "B1": {
                    "status": "success",
                    "latency_sec": 0.05,
                    "retrieved_papers": ["paper_1"],
                    "retrieved_chunks": [
                        {
                            "id": "paper_1#1",
                            "paper_id": "paper_1",
                            "page_number": 1,
                            "text_content": "Winograd adaptation text",
                            "score": 0.95
                        }
                    ]
                }
            }
        }
    ]
    
    report = parse_report(raw_data)
    assert isinstance(report, ReportOutput)
    assert len(report.results) == 1
    
    case = report.results[0]
    assert case.id == "Q1"
    assert case.query == "What is Winograd?"
    assert "B1" in case.baselines
    
    baseline = case.baselines["B1"]
    assert baseline.status == "success"
    assert baseline.latency_sec == 0.05
    assert len(baseline.retrieved_chunks) == 1
    assert baseline.retrieved_chunks[0].id == "paper_1#1"
    assert baseline.retrieved_chunks[0].score == 0.95


def test_full_evaluation_report_parsing():
    # Example format matching result_metrics.yaml
    raw_data = {
        "metadata": {
            "date": "2026-06-18 12:00:00",
            "original_metadata": {
                "llm": {"max_tokens": 8000}
            }
        },
        "results": [
            {
                "id": "Q2",
                "query": "Is BERT larger than GPT?",
                "category": "multi-hop",
                "golden_answer": "No, GPT is larger.",
                "expected_papers": ["bert_paper"],
                "baselines": {
                    "B1": {
                        "status": "success",
                        "latency_sec": 12.34,
                        "retrieved_papers": ["bert_paper"],
                        "generated_answer": "GPT is indeed larger.",
                        "eval_metrics": {
                            "retrieval_recall": 1.0,
                            "context_precision": 0.8,
                            "faithfulness": 0.9,
                            "answer_relevance": 0.95,
                            "semantic_accuracy": 0.85
                        }
                    }
                }
            }
        ]
    }
    
    report = parse_report(raw_data)
    assert isinstance(report, ReportOutput)
    assert report.metadata["date"] == "2026-06-18 12:00:00"
    assert len(report.results) == 1
    
    case = report.results[0]
    assert case.id == "Q2"
    assert case.baselines["B1"].generated_answer == "GPT is indeed larger."
    assert case.baselines["B1"].eval_metrics["semantic_accuracy"] == 0.85


def test_invalid_data_validation():
    # Score is missing and type is incorrect
    invalid_data = [
        {
            "id": "Q1",
            "query": "Invalid query",
            "baselines": {
                "B1": {
                    "status": "success",
                    "retrieved_chunks": [
                        {
                            "id": "chunk_1",
                            "paper_id": "paper_1",
                            "page_number": "not_an_int",  # invalid page number
                            "text_content": "some text"
                            # score is missing
                        }
                    ]
                }
            }
        }
    ]
    
    with pytest.raises(ValidationError):
        parse_report(invalid_data)


def test_analyze_metrics_with_retrieval_only():
    raw_data = [
        {
            "id": "Q1",
            "query": "Retrieval test",
            "category": "single-document",
            "expected_papers": ["paper_1"],
            "baselines": {
                "B1": {
                    "status": "success",
                    "latency_sec": 0.1,
                    "retrieved_papers": ["paper_1"],
                    "retrieved_chunks": [
                        {
                            "id": "paper_1#1",
                            "paper_id": "paper_1",
                            "page_number": 1,
                            "text_content": "Text",
                            "score": 1.0
                        }
                    ]
                }
            }
        }
    ]
    
    stats = analyze_metrics(raw_data)
    assert "B1" in stats["baselines"]
    
    # Retrieval metrics should be calculated automatically
    b1_stats = stats["summary"]["B1"]
    assert b1_stats["retrieval_recall"]["mean"] == 1.0
    assert b1_stats["context_precision"]["mean"] == 1.0
    
    # Generative metrics should be 0.0 or not set
    assert b1_stats["semantic_accuracy"]["mean"] == 0.0


def test_analyze_metrics_with_full_evaluation():
    raw_data = {
        "metadata": {
            "date": "2026-06-18 12:00:00"
        },
        "results": [
            {
                "id": "Q1",
                "query": "Eval test",
                "category": "single-document",
                "expected_papers": ["paper_1"],
                "baselines": {
                    "B1": {
                        "status": "success",
                        "latency_sec": 1.0,
                        "retrieved_papers": ["paper_1"],
                        "generated_answer": "Answer",
                        "eval_metrics": {
                            "retrieval_recall": 1.0,
                            "context_precision": 1.0,
                            "semantic_accuracy": 0.95
                        }
                    }
                }
            }
        ]
    }
    
    stats = analyze_metrics(raw_data)
    b1_stats = stats["summary"]["B1"]
    assert b1_stats["retrieval_recall"]["mean"] == 1.0
    assert b1_stats["semantic_accuracy"]["mean"] == 0.95


def test_analyze_metrics_with_context_fillness():
    raw_data = {
        "metadata": {
            "date": "2026-06-18 12:00:00",
            "original_metadata": {
                "llm": {
                    "max_tokens": 1000,
                    "model_max_context": 8000
                }
            }
        },
        "results": [
            {
                "id": "Q1",
                "query": "Context test",
                "category": "single-document",
                "expected_papers": ["paper_1"],
                "baselines": {
                    "B1": {
                        "status": "success",
                        "latency_sec": 1.0,
                        "retrieved_papers": ["paper_1"],
                        "context_token": 1600,
                    }
                }
            }
        ]
    }
    
    stats = analyze_metrics(raw_data)
    b1_stats = stats["summary"]["B1"]
    assert b1_stats["context_fillness"]["mean"] == 0.20
