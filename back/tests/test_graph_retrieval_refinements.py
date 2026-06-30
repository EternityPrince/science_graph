import os
import json
import logging
import pytest
from unittest.mock import MagicMock, patch
from src.services.rag_service import RAGService
from src.config import Config, config
from src.models import Chunk, Paper

class DummyChunk:
    def __init__(self, id, paper_id, text_content, chunk_index=0):
        self.id = id
        self.paper_id = paper_id
        self.text_content = text_content
        self.chunk_index = chunk_index
        self.retrieval_sources = []

def test_config_budget_mode_validation(caplog):
    """Verify that an invalid candidate_budget_mode falls back to mirror_base and logs a warning."""
    cfg = Config()
    cfg.data["graph_retrieval"] = {
        "candidate_budget_mode": "invalid_mode_xyz"
    }
    with caplog.at_level(logging.WARNING):
        mode = cfg.graph_retrieval_candidate_budget_mode
        assert mode == "mirror_base"
        assert any("Invalid candidate_budget_mode" in record.message for record in caplog.records)

def test_config_defaults():
    """Verify default graph_retrieval.enabled is False to preserve backward compatibility."""
    cfg = Config()
    if "graph_retrieval" in cfg.data:
        del cfg.data["graph_retrieval"]
    assert cfg.graph_retrieval_enabled is False

def test_dedup_metadata_and_attributes_preservation():
    """Verify that deduplication preserves all sources and copies metadata attributes."""
    service = RAGService(MagicMock(), MagicMock(), MagicMock(), MagicMock())
    
    c1 = DummyChunk("chunk_1", "paper_1", "text one")
    c1.retrieval_sources.append({"source": "dense"})
    
    c2 = DummyChunk("chunk_1", "paper_1", "text one")
    c2.retrieval_sources.append({"source": "graph_neighbor"})
    c2.candidate_source = "graph_neighbor"
    c2.graph_distance = 2
    c2.graph_path_reason = "path reason"
    
    dedupped = service._deduplicate_candidates([c1, c2])
    
    assert len(dedupped) == 1
    final_chunk = dedupped[0]
    
    # Check that both sources are present in retrieval_sources list
    sources = [s["source"] for s in final_chunk.retrieval_sources]
    assert "dense" in sources
    assert "graph_neighbor" in sources
    
    # Check that attributes are copied to the first chunk
    assert final_chunk.candidate_source == "graph_neighbor"
    assert final_chunk.graph_distance == 2
    assert final_chunk.graph_path_reason == "path reason"

def test_embedding_fallback_sqlite_repo(caplog):
    """Verify that search_chunks_within_papers handles missing/empty query embedding by logging a warning and falling back to zero embedding."""
    from src.repository.sqlite_impl import SQLiteGraphRepository
    
    # Mock SQLite DB call
    repo = SQLiteGraphRepository(":memory:")
    # Create nodes and tables if necessary or just verify warning
    with patch.object(repo, "_get_connection") as mock_conn:
        mock_conn.return_value.__enter__.return_value.execute.return_value.fetchall.return_value = []
        
        with caplog.at_level(logging.WARNING):
            res = repo.search_chunks_within_papers(query_embedding=[], paper_ids=["P1"])
            assert res == []
            assert any("Query embedding is unavailable" in record.message for record in caplog.records)

def test_graph_retrieval_skip_reasons_in_trace(tmp_path, monkeypatch):
    """Verify that skip reasons are correctly calculated and logged in the trace."""
    monkeypatch.chdir(tmp_path)
    
    # Case 1: graph_retrieval is disabled
    monkeypatch.setattr(Config, "graph_retrieval_enabled", property(lambda self: False))
    
    service = RAGService(MagicMock(), MagicMock(), MagicMock(), MagicMock())
    service.trace_dir = tmp_path
    
    # Mock base search candidates
    c1 = DummyChunk("c1", "P1", "text 1")
    vector_repo = MagicMock()
    vector_repo.search_similar_chunks.return_value = [(c1, 0.9)]
    vector_repo.search_text_fts5.return_value = []
    service.vector_repo = vector_repo
    
    service.retrieve_relevant_chunks("query", limit=5)
    
    trace_file = tmp_path / "graph_retrieval_trace.jsonl"
    assert trace_file.exists()
    with open(trace_file, "r") as f:
        entry = json.loads(f.read().strip())
        assert entry["graph_retrieval_enabled"] is False
        assert entry["graph_retrieval_skip_reason"] == "disabled"
        
    # Clean up trace file
    os.remove(trace_file)
    
    # Case 2: graph_retrieval is enabled but graph_neighbors_in_rrf is disabled
    monkeypatch.setattr(Config, "graph_retrieval_enabled", property(lambda self: True))
    monkeypatch.setitem(config.data["rag_components"], "graph_neighbors_in_rrf", False)
    
    service = RAGService(MagicMock(), MagicMock(), MagicMock(), MagicMock())
    service.trace_dir = tmp_path
    
    vector_repo = MagicMock()
    vector_repo.search_similar_chunks.return_value = [(c1, 0.9)]
    vector_repo.search_text_fts5.return_value = []
    service.vector_repo = vector_repo
    
    service.retrieve_relevant_chunks("query", limit=5)
    
    assert trace_file.exists()
    with open(trace_file, "r") as f:
        entry = json.loads(f.read().strip())
        assert entry["graph_retrieval_enabled"] is True
        assert entry["graph_retrieval_skip_reason"] == "graph_neighbors_in_rrf_disabled"
