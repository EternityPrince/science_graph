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
