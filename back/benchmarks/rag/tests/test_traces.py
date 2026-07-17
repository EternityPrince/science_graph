import json
import pytest
from pathlib import Path

from core.traces import (
    load_graph_retrieval_trace,
    parse_graph_retrieval_trace,
    parse_eval_trace,
    parse_all_traces,
)


def test_load_graph_retrieval_trace(tmp_path):
    trace_file = tmp_path / "graph_retrieval_trace.jsonl"
    record1 = {"baseline": "B6", "query_id": "Q1", "query": "test query 1"}
    record2 = {"baseline": "B5", "query": "test query 2"}  # missing query_id, uses query
    record3 = "invalid json line\n"

    with open(trace_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(record1) + "\n")
        f.write(json.dumps(record2) + "\n")
        f.write(record3)

    trace_map = load_graph_retrieval_trace(trace_file)
    assert len(trace_map) == 2
    assert ("B6", "Q1") in trace_map
    assert ("B5", "test query 2") in trace_map


def test_parse_graph_retrieval_trace(tmp_path):
    traces_dir = tmp_path / "traces"
    parsed_dir = tmp_path / "parsed"
    traces_dir.mkdir()

    trace_file = traces_dir / "graph_retrieval_trace.jsonl"
    record = {
        "query_id": "Q1",
        "baseline": "B6",
        "category": "multi-hop",
        "query": "What is Graph RAG?",
        "graph_retrieval_enabled": True,
        "base_candidates_count": 5,
        "graph_neighbor_paper_ids_count": 3,
        "graph_chunk_candidates_count": 4,
        "merged_candidates_count_before_reranker": 8,
        "graph_chunks_survived_final_context_count": 2,
        "graph_survival_rate": 0.5,
        "distinct_papers_in_final_context": 3,
        "base_candidate_paper_ids": ["paper_1"],
        "graph_chunk_candidate_paper_ids": ["paper_2"],
        "final_context_paper_ids": ["paper_1", "paper_2"],
    }

    with open(trace_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    csv_rows, summary = parse_graph_retrieval_trace(traces_dir, parsed_dir)
    assert csv_rows is not None
    assert len(csv_rows) == 1
    assert summary["total_queries"] == 1
    assert (parsed_dir / "graph_retrieval_trace.parsed.csv").exists()
    assert (parsed_dir / "graph_retrieval_trace.summary.json").exists()


def test_parse_eval_trace(tmp_path):
    traces_dir = tmp_path / "traces"
    parsed_dir = tmp_path / "parsed"
    traces_dir.mkdir()

    trace_file = traces_dir / "eval_trace.jsonl"
    record = {
        "query_id": "Q1",
        "baseline": "B6",
        "category": "single-document",
        "judge_model": "gpt-4",
        "latency_sec": 1.23,
        "retrieval_recall": 1.0,
        "context_precision": 0.8,
        "faithfulness": 0.9,
        "answer_relevance": 0.95,
        "citation_fidelity": 1.0,
        "semantic_accuracy": 0.9,
        "context_fillness": 0.85,
        "is_answerable": True,
    }

    with open(trace_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    csv_rows, summary = parse_eval_trace(traces_dir, parsed_dir)
    assert csv_rows is not None
    assert len(csv_rows) == 1
    assert summary["total_eval_rows"] == 1
    assert (parsed_dir / "eval_trace.parsed.csv").exists()
    assert (parsed_dir / "eval_trace.summary.json").exists()


def test_parse_all_traces(tmp_path):
    traces_dir = tmp_path / "traces"
    parsed_dir = tmp_path / "parsed"
    traces_dir.mkdir()

    g_summary, e_summary = parse_all_traces(traces_dir, parsed_dir)
    assert g_summary is None
    assert e_summary is None
