import os
import json
import csv
import pytest
from pathlib import Path
import sys

# Add python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.rag.parse_metrics import main as parse_metrics_main


def test_parse_metrics_reads_graph_trace_from_graphs_run_dir(tmp_path, monkeypatch):
    # Setup temporary graphs/run_test directory structure
    run_dir = tmp_path / "graphs" / "run_test"
    traces_dir = run_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Create a dummy graph retrieval trace file
    graph_trace_file = traces_dir / "graph_retrieval_trace.jsonl"
    dummy_trace = {
        "query_id": "Q01",
        "baseline": "B6",
        "category": "multi-hop",
        "query": "What is self attention?",
        "graph_retrieval_enabled": True,
        "graph_retrieval_skip_reason": None,
        "base_candidates_count": 5,
        "graph_neighbor_paper_ids_count": 10,
        "graph_chunk_candidates_count": 3,
        "merged_candidates_count_before_reranker": 8,
        "reranker_input_count_before_limit": 8,
        "reranker_input_count_after_limit": 5,
        "candidate_count_after_reranker": 5,
        "graph_chunks_survived_final_context_count": 1,
        "graph_chunks_survived_final_context": ["p1#1"],
        "graph_survival_rate": 0.33,
        "distinct_papers_in_final_context": 2,
        "base_candidate_paper_ids": ["P1", "P2"],
        "graph_chunk_candidate_paper_ids": ["P3"],
        "final_context_paper_ids": ["P1", "P3"]
    }
    with open(graph_trace_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(dummy_trace) + "\n")
        
    # 2. Create a dummy metrics details file
    details_file = run_dir / "metrics_details.csv"
    with open(details_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "category", "baseline", "status", "latency_sec", "semantic_accuracy"])
        writer.writerow(["Q01", "multi-hop", "B6", "success", 1.5, 0.8])

    # Mock CLI arguments
    monkeypatch.setattr(sys, "argv", ["parse_metrics.py", str(run_dir), "--traces-only"])
    
    # Run the main parser function
    parse_metrics_main()
    
    # Assert parsed graph retrieval trace CSV was created
    parsed_csv = run_dir / "parsed" / "graph_retrieval_trace.parsed.csv"
    assert parsed_csv.exists()
    
    # Verify content
    with open(parsed_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["query_id"] == "Q01"
        assert rows[0]["baseline"] == "B6"
        assert float(rows[0]["graph_survival_rate"]) == 0.33


def test_parse_metrics_creates_per_query_joined(tmp_path, monkeypatch):
    run_dir = tmp_path / "graphs" / "run_test"
    traces_dir = run_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    
    # Trace file
    graph_trace_file = traces_dir / "graph_retrieval_trace.jsonl"
    dummy_trace = {
        "query_id": "Q01",
        "baseline": "B6",
        "category": "multi-hop",
        "query": "What is self attention?",
        "graph_retrieval_enabled": True,
        "graph_survival_rate": 0.5,
        "graph_chunks_survived_final_context_count": 2
    }
    with open(graph_trace_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(dummy_trace) + "\n")
        
    # Details CSV
    details_file = run_dir / "metrics_details.csv"
    with open(details_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "category", "baseline", "status", "latency_sec", "semantic_accuracy"])
        writer.writerow(["Q01", "multi-hop", "B6", "success", 1.5, 0.8])

    monkeypatch.setattr(sys, "argv", ["parse_metrics.py", str(run_dir), "--traces-only"])
    parse_metrics_main()
    
    joined_csv = run_dir / "parsed" / "per_query_joined.csv"
    assert joined_csv.exists()
    
    with open(joined_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["query_id"] == "Q01"
        assert float(rows[0]["semantic_accuracy"]) == 0.8
        assert float(rows[0]["graph_survival_rate"]) == 0.5


def test_parse_metrics_missing_trace_is_non_fatal(tmp_path, monkeypatch):
    run_dir = tmp_path / "graphs" / "run_test"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Details CSV
    details_file = run_dir / "metrics_details.csv"
    with open(details_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "category", "baseline", "status", "latency_sec", "semantic_accuracy"])
        writer.writerow(["Q01", "multi-hop", "B6", "success", 1.5, 0.8])

    monkeypatch.setattr(sys, "argv", ["parse_metrics.py", str(run_dir), "--traces-only"])
    
    # Should not raise exception even if traces folder or trace files are missing
    try:
        parse_metrics_main()
    except Exception as e:
        pytest.fail(f"parse_metrics.py failed on missing trace files: {e}")
        
    parsed_details = run_dir / "parsed" / "metrics_details.parsed.csv"
    assert parsed_details.exists()


def test_trace_outputs_not_written_to_project_root(tmp_path, monkeypatch):
    # Verify that RAGService writes trace relative to trace_dir inside run directory
    from src.services.rag_service import RAGService
    from unittest.mock import MagicMock
    
    # Clean up pre-existing project root trace file if present
    project_root_trace = "graph_retrieval_trace.jsonl"
    if os.path.exists(project_root_trace):
        try:
            os.remove(project_root_trace)
        except Exception:
            pass
    
    service = RAGService(MagicMock(), MagicMock(), MagicMock(), MagicMock())
    run_dir = tmp_path / "run_123"
    trace_dir = run_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    service.trace_dir = trace_dir
    
    service._last_graph_trace = {"query_concepts": []}
    service.current_trace = {"query_id": "Q01", "baseline": "B6"}
    
    # Write a trace
    service._write_graph_retrieval_trace("What is self attention?", [], [])
    
    # Ensure it's in the run_dir / traces, and NOT in project root (current working directory)
    assert (trace_dir / "graph_retrieval_trace.jsonl").exists()
    assert not os.path.exists("graph_retrieval_trace.jsonl")


def test_run_summary_contains_graph_retrieval_section(tmp_path, monkeypatch):
    run_dir = tmp_path / "graphs" / "run_test"
    traces_dir = run_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    
    # Trace file
    graph_trace_file = traces_dir / "graph_retrieval_trace.jsonl"
    dummy_trace = {
        "query_id": "Q01",
        "baseline": "B6",
        "category": "multi-hop",
        "query": "What is self-attention?",
        "graph_retrieval_enabled": True,
        "graph_neighbor_paper_ids_count": 5,
        "graph_chunk_candidates_count": 2,
        "graph_chunks_survived_final_context_count": 1,
        "graph_survival_rate": 0.5
    }
    with open(graph_trace_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(dummy_trace) + "\n")
        
    # Details CSV
    details_file = run_dir / "metrics_details.csv"
    with open(details_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "category", "baseline", "status", "latency_sec", "semantic_accuracy"])
        writer.writerow(["Q01", "multi-hop", "B6", "success", 1.5, 0.8])

    monkeypatch.setattr(sys, "argv", ["parse_metrics.py", str(run_dir), "--traces-only"])
    parse_metrics_main()
    
    summary_json_path = run_dir / "parsed" / "run_summary.json"
    assert summary_json_path.exists()
    
    with open(summary_json_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)
        assert "graph_retrieval" in summary_data
        assert "B6" in summary_data["graph_retrieval"]
        assert summary_data["graph_retrieval"]["B6"]["queries_with_graph_survival"] == 1


def test_parse_old_reports_directory_still_supported(tmp_path, monkeypatch):
    # Setup legacy reports/run_test directory structure
    legacy_dir = tmp_path / "reports" / "run_test"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    
    # Details CSV
    details_file = legacy_dir / "metrics_details.csv"
    with open(details_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "category", "baseline", "status", "latency_sec", "semantic_accuracy"])
        writer.writerow(["Q01", "multi-hop", "B6", "success", 1.5, 0.8])

    monkeypatch.setattr(sys, "argv", ["parse_metrics.py", str(legacy_dir), "--traces-only"])
    parse_metrics_main()
    
    parsed_details = legacy_dir / "parsed" / "metrics_details.parsed.csv"
    assert parsed_details.exists()
<<<<<<< HEAD


def test_load_graph_retrieval_trace(tmp_path):
    from benchmarks.rag.parse_metrics import load_graph_retrieval_trace
    trace_file = tmp_path / "graph_retrieval_trace.jsonl"
    content = """
    {"query_id": "Q01", "baseline": "B6", "category": "multi-hop", "query": "q1", "graph_retrieval_enabled": true}
    
    {"query": "q2", "baseline": "B6", "category": "single-document", "graph_retrieval_enabled": false}
    {"query_id": "Q03", "baseline": "B4", "category": "multi-hop", "graph_retrieval_enabled": true}
    invalid json here
    {"query_id": "Q01", "baseline": "B6", "category": "multi-hop", "query": "q1_updated", "graph_retrieval_enabled": false}
    """
    with open(trace_file, "w", encoding="utf-8") as f:
        f.write(content)
        
    trace_map = load_graph_retrieval_trace(trace_file)
    
    assert ("B6", "Q01") in trace_map
    assert trace_map[("B6", "Q01")]["graph_retrieval_enabled"] is False
    assert ("B6", "q2") in trace_map
    assert ("B4", "Q03") in trace_map


def test_details_merge_and_summary_aggregation():
    from benchmarks.rag.core.analytics import analyze_metrics
    data = {
        "results": [
            {
                "id": "Q01",
                "category": "multi-hop",
                "query": "q1",
                "baselines": {
                    "B6": {"status": "success", "latency_sec": 1.0, "eval_metrics": {"semantic_accuracy": 0.8}}
                }
            },
            {
                "id": "Q02",
                "category": "single-document",
                "query": "q2",
                "baselines": {
                    "B6": {"status": "success", "latency_sec": 1.5, "eval_metrics": {"semantic_accuracy": 0.9}}
                }
            }
        ]
    }
    
    trace_map = {
        ("B6", "Q01"): {
            "query_id": "Q01",
            "baseline": "B6",
            "category": "multi-hop",
            "graph_retrieval_enabled": True,
            "query_concepts_all": ["a", "b"],
            "query_concepts_strong": ["a"],
            "query_concepts_dropped": [{"concept": "b", "reason": "test"}],
            "graph_neighbor_nodes_total": 10,
            "graph_chunks_before_rerank_count": 5,
            "graph_chunks_survived_final_context_count": 2,
            "best_graph_candidate_rank_after_rerank": 3
        }
    }
    
    stats = analyze_metrics(data, trace_map)
    
    results = data["results"]
    b6_q01 = results[0]["baselines"]["B6"]
    assert b6_q01["graph_retrieval_enabled"] is True
    assert b6_q01["query_concepts_all_count"] == 2
    assert b6_q01["query_concepts_strong_count"] == 1
    assert b6_q01["query_concepts_dropped_count"] == 1
    assert b6_q01["graph_neighbor_nodes_total"] == 10
    
    b6_q02 = results[1]["baselines"]["B6"]
    assert b6_q02["graph_retrieval_enabled"] is False
    assert b6_q02["query_concepts_all_count"] == 0
    assert b6_q02["graph_neighbor_nodes_total"] == 0
    assert b6_q02["graph_survival_rate"] == 0.0
    
    gd = stats["summary"]["B6"]["graph_diagnostics"]
    assert gd["enabled_rate"] == 0.5
    assert gd["skipped_rate"] == 0.0
    assert gd["survival_rate"] == 0.4
    assert gd["queries_with_chunks"] == 0.5
    assert gd["avg_best_rank"] == 3.0
    
    assert "category_graph_stats" in stats
    cat_stats = stats["category_graph_stats"]
    assert cat_stats["multi-hop"]["B6"]["graph_survival_rate"] == 0.4
    assert cat_stats["single-document"]["B6"]["graph_survival_rate"] == 0.0


def test_markdown_rendering_and_failure_examples(tmp_path):
    from benchmarks.rag.core.reporting import generate_markdown_report
    stats = {
        "baselines": ["B6"],
        "summary": {
            "B6": {
                "success_rate": 100.0,
                "semantic_accuracy": {"mean": 0.8, "min": 0.8, "max": 0.8, "median": 0.8, "stdev": 0.0, "count": 1},
                "latency_sec": {"mean": 1.2, "min": 1.2, "max": 1.2, "median": 1.2, "stdev": 0.0, "count": 1},
                "retrieval_recall": {"mean": 0.9, "min": 0.9, "max": 0.9, "median": 0.9, "stdev": 0.0, "count": 1},
                "context_precision": {"mean": 0.8, "min": 0.8, "max": 0.8, "median": 0.8, "stdev": 0.0, "count": 1},
                "faithfulness": {"mean": 0.9, "min": 0.9, "max": 0.9, "median": 0.9, "stdev": 0.0, "count": 1},
                "answer_relevance": {"mean": 0.9, "min": 0.9, "max": 0.9, "median": 0.9, "stdev": 0.0, "count": 1},
                "citation_fidelity": {"mean": 0.9, "min": 0.9, "max": 0.9, "median": 0.9, "stdev": 0.0, "count": 1},
                "context_fillness": {"mean": 0.5, "min": 0.5, "max": 0.5, "median": 0.5, "stdev": 0.0, "count": 1},
                "token_output": {"mean": 100.0, "min": 100.0, "max": 100.0, "median": 100.0, "stdev": 0.0, "count": 1},
                "token_answer": {"mean": 80.0, "min": 80.0, "max": 80.0, "median": 80.0, "stdev": 0.0, "count": 1},
                "token_reasoning": {"mean": 20.0, "min": 20.0, "max": 20.0, "median": 20.0, "stdev": 0.0, "count": 1},
                "graph_diagnostics": {
                    "enabled_rate": 1.0,
                    "skipped_rate": 0.0,
                    "avg_concepts": 3.0,
                    "avg_strong": 2.0,
                    "avg_dropped": 1.0,
                    "avg_neighbor_nodes": 50.0,
                    "avg_neighbor_local_papers": 10.0,
                    "avg_neighbor_papers_with_chunks": 5.0,
                    "avg_graph_chunk_candidates": 5.0,
                    "survival_rate": 0.2,
                    "queries_with_chunks_survived": 1.0,
                    "avg_best_rank": 2.0,
                    "avg_neighbor_cand": 1.0,
                    "avg_concept_cand": 1.0,
                    "avg_bridge_cand": 3.0
                }
            }
        },
        "categories": ["multi-hop"],
        "category_stats": {
            "multi-hop": {
                "B6": {"semantic_accuracy": 0.8}
            }
        },
        "query_difficulty": [
            {"id": "Q01", "category": "multi-hop", "query": "What is self attention?", "avg_score": 0.8}
        ],
        "pairwise_win_rates": {
            "semantic_accuracy": {"B6": {"B6": 0.0}}
        },
        "total_queries": 1,
        "has_graph_trace": True,
        "category_graph_stats": {
            "multi-hop": {
                "B6": {
                    "avg_graph_chunk_candidates": 5.0,
                    "graph_survival_rate": 0.2,
                    "queries_with_graph_chunks_survived": 1.0,
                    "avg_strong_query_concepts": 2.0,
                    "avg_distinct_papers_in_final_context": 3.0
                }
            }
        },
        "top_failures": [
            {
                "query_id": "Q01",
                "query": "What is self attention?",
                "baseline": "B6",
                "category": "multi-hop",
                "skip_reason": "",
                "neighbor_nodes": 50,
                "papers_with_chunks": 0,
                "chunks_before": 5,
                "chunks_survived": 0
            }
        ]
    }
    
    md_path = tmp_path / "report_with_trace.md"
    generate_markdown_report(stats, md_path)
    
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()
        assert "## 📊 Graph Retrieval Diagnostics" in md_content
        assert "Graph Candidate Source Breakdown" in md_content
        assert "Query Concept Filtering" in md_content
        assert "Graph Retrieval Failure Examples" in md_content
        assert "Graph Retrieval by Category" in md_content
        assert "Q01" in md_content
        
    stats_no_trace = stats.copy()
    stats_no_trace["has_graph_trace"] = False
    
    md_path_no_trace = tmp_path / "report_no_trace.md"
    generate_markdown_report(stats_no_trace, md_path_no_trace)
    
    with open(md_path_no_trace, "r", encoding="utf-8") as f:
        md_content_no_trace = f.read()
        assert "Graph retrieval trace was not found for this run." in md_content_no_trace
        assert "Graph Candidate Source Breakdown" not in md_content_no_trace
=======
>>>>>>> 7c7b22f (add trace into parse_metrica)
