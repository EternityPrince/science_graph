import os
import json
import pytest
from unittest.mock import MagicMock
from src.services.rag_service import RAGService
from src.config import config
from src.models import Chunk
from tests.graph_test_utils import FakeGraphRepository

class DummyChunk:
    def __init__(self, id, paper_id, text_content):
        self.id = id
        self.paper_id = paper_id
        self.text_content = text_content
        self.retrieval_sources = []

@pytest.fixture
def clean_trace_file():
    """Fixture to ensure the trace file is deleted before and after the test."""
    trace_path = "graph_retrieval_trace.jsonl"
    if os.path.exists(trace_path):
        os.remove(trace_path)
    yield trace_path
    if os.path.exists(trace_path):
        os.remove(trace_path)

def test_trace_disabled_does_not_write(clean_trace_file):
    """Verify trace is not written when trace is disabled in config."""
    config.data["graph_retrieval"]["trace_enabled"] = False
    
    service = RAGService(FakeGraphRepository(), MagicMock(), MagicMock(), MagicMock())
    service._last_graph_trace = {"query_concepts": []}
    
    service._write_graph_retrieval_trace("query", [], [])
    assert not os.path.exists(clean_trace_file)

def test_trace_enabled_writes_correct_fields(clean_trace_file):
    """Verify trace logs all expected JSON keys when enabled."""
    config.data["graph_retrieval"]["trace_enabled"] = True
    
    service = RAGService(FakeGraphRepository(), MagicMock(), MagicMock(), MagicMock())
    service.current_trace = {"query_id": "test_q_123"}
    
    # 2 graph chunks before rerank: p3#1 and p4#1
    service._last_graph_trace = {
        "query_concepts": ["c_memory"],
        "seed_paper_ids": ["P1"],
        "graph_concept_candidate_papers": ["P3"],
        "graph_bridge_candidate_papers": ["P4"],
        "graph_chunks_before_rerank": ["p3#1", "p4#1"],
        "graph_chunks_before_rerank_count": 2,
        "graph_candidate_source_breakdown": {
            "graph_concept_retrieval": 1,
            "graph_bridge_retrieval": 1
        }
    }
    
    # After rerank and context trimming:
    c4 = DummyChunk("p4#1", "P4", "t4")
    c4.retrieval_sources = [{"source": "graph_bridge_retrieval"}]
    c3 = DummyChunk("p3#1", "P3", "t3")
    c3.retrieval_sources = [{"source": "graph_concept_retrieval"}]
    c1 = DummyChunk("p1#0", "P1", "t1")
    c1.retrieval_sources = [{"source": "dense"}]
    
    final_chunks = [(c4, 0.9), (c3, 0.8), (c1, 0.7)]
    trimmed_chunks = [(c4, 0.9), (c1, 0.7)]
    
    service._write_graph_retrieval_trace("Explain memory", final_chunks, trimmed_chunks)
    
    assert os.path.exists(clean_trace_file)
    with open(clean_trace_file, "r") as f:
        lines = f.readlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        
        # Verify required keys
        assert entry["query_id"] == "test_q_123"
        assert entry["query"] == "Explain memory"
        assert entry["query_concepts"] == ["c_memory"]
        assert entry["seed_paper_ids"] == ["P1"]
        assert entry["graph_concept_candidate_papers"] == ["P3"]
        assert entry["graph_bridge_candidate_papers"] == ["P4"]
        assert entry["graph_chunks_before_rerank"] == ["p3#1", "p4#1"]
        assert entry["graph_chunks_before_rerank_count"] == 2
        
        # Survival rate calculations: before = 2, survived = 1 (p4#1 survived)
        assert entry["graph_chunks_survived_final_context"] == ["p4#1"]
        assert entry["graph_chunks_survived_final_context_count"] == 1
        assert entry["graph_survival_rate"] == 0.5
        
        # Best rank calculation: p4#1 is at rank 1, p3#1 is at rank 2. Best is 1.
        assert entry["best_graph_candidate_rank_after_rerank"] == 1
        
        # Breakdown of candidate sources
        assert entry["graph_candidate_source_breakdown"] == {"graph_concept_retrieval": 1, "graph_bridge_retrieval": 1}
        
        # final context lists
        assert entry["final_context_paper_ids"] == ["P1", "P4"]
        assert entry["final_context_chunk_ids"] == ["p1#0", "p4#1"]
        assert entry["distinct_papers_in_final_context"] == 2
        
        # rerank positions checks
        positions = entry["graph_candidate_rerank_positions"]
        assert len(positions) == 2
        p4_pos = next(p for p in positions if p["chunk_id"] == "p4#1")
        assert p4_pos["rank_after_rerank"] == 1
        assert p4_pos["survived_final_context"] is True
        
        p3_pos = next(p for p in positions if p["chunk_id"] == "p3#1")
        assert p3_pos["rank_after_rerank"] == 2
        assert p3_pos["survived_final_context"] is False

def test_trace_append_behavior(clean_trace_file):
    """Verify trace file appends new entries without overwriting existing ones."""
    config.data["graph_retrieval"]["trace_enabled"] = True
    
    service = RAGService(FakeGraphRepository(), MagicMock(), MagicMock(), MagicMock())
    service._last_graph_trace = {"query_concepts": []}
    
    # Write first trace
    service._write_graph_retrieval_trace("first query", [], [])
    assert os.path.exists(clean_trace_file)
    
    # Write second trace
    service._write_graph_retrieval_trace("second query", [], [])
    
    with open(clean_trace_file, "r") as f:
        lines = f.readlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["query"] == "first query"
        assert json.loads(lines[1])["query"] == "second query"

def test_trace_in_ask_no_double_write(clean_trace_file):
    """Verify retrieve_relevant_chunks does not write trace when called inside ask."""
    config.data["graph_retrieval"]["trace_enabled"] = True
    
    # Mock repositories
    graph_repo = FakeGraphRepository()
    vector_repo = MagicMock()
    emb_engine = MagicMock()
    emb_engine.get_embedding.return_value = [0.1] * 384
    
    service = RAGService(graph_repo, vector_repo, emb_engine, MagicMock())
    # Set _in_ask = True
    service._in_ask = True
    service._last_graph_trace = {"query_concepts": []}
    
    # Mock base search
    c1 = Chunk(id="p1#0", paper_id="P1", text_content="Paper One text", page_number=1)
    vector_repo.search_similar_chunks.return_value = [(c1, 0.8)]
    vector_repo.search_text_fts5.return_value = []
    
    # Calling retrieve_relevant_chunks should skip writing trace to file
    service.retrieve_relevant_chunks("Explain memory", limit=5)
    assert not os.path.exists(clean_trace_file)
