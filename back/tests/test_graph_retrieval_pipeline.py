import pytest
from unittest.mock import MagicMock, patch
from src.services.rag_service import RAGService
from src.models import Chunk
from src.config import config
from tests.graph_test_utils import FakeGraphRepository, FakeReranker

class DummyChunk:
    def __init__(self, id, paper_id, text_content, chunk_index=0):
        self.id = id
        self.paper_id = paper_id
        self.text_content = text_content
        self.chunk_index = chunk_index
        self.retrieval_sources = []

# =========================================================================
# Scoped Chunk Retrieval & Dedup Tests
# =========================================================================

def test_deduplicate_candidates_by_id():
    """Verify deduplication prioritizes chunk ID and merges retrieval sources."""
    service = RAGService(FakeGraphRepository(), MagicMock(), MagicMock(), MagicMock())
    
    c1 = DummyChunk("c1", "P1", "text 1")
    c1.retrieval_sources = [{"source": "dense"}]
    
    c2 = DummyChunk("c1", "P1", "text 1")
    c2.retrieval_sources = [{"source": "graph_concept_retrieval"}]
    
    dedupped = service._deduplicate_candidates([c1, c2])
    assert len(dedupped) == 1
    sources = [s["source"] for s in dedupped[0].retrieval_sources]
    assert "dense" in sources
    assert "graph_concept_retrieval" in sources

def test_deduplicate_candidates_fallback_indices():
    """Verify deduplication fallback by (paper_id, chunk_index) and content hash."""
    service = RAGService(FakeGraphRepository(), MagicMock(), MagicMock(), MagicMock())
    
    # 1. Fallback by (paper_id, chunk_index)
    c1 = DummyChunk(None, "P1", "text 1", chunk_index=0)
    c1.retrieval_sources = [{"source": "dense"}]
    c2 = DummyChunk(None, "P1", "text 1", chunk_index=0)
    c2.retrieval_sources = [{"source": "lexical"}]
    
    dedupped_idx = service._deduplicate_candidates([c1, c2])
    assert len(dedupped_idx) == 1
    assert "lexical" in [s["source"] for s in dedupped_idx[0].retrieval_sources]

    # 2. Fallback by content hash
    c3 = DummyChunk(None, None, "unique content text hash")
    c3.retrieval_sources = [{"source": "dense"}]
    c4 = DummyChunk(None, None, "unique content text hash")
    c4.retrieval_sources = [{"source": "graph_bridge_retrieval"}]
    
    dedupped_hash = service._deduplicate_candidates([c3, c4])
    assert len(dedupped_hash) == 1
    assert "graph_bridge_retrieval" in [s["source"] for s in dedupped_hash[0].retrieval_sources]

def test_deduplicate_source_priority():
    """Verify deduplication sorts retrieval sources stably: dense, lexical, graph_concept, graph_bridge."""
    service = RAGService(FakeGraphRepository(), MagicMock(), MagicMock(), MagicMock())
    
    c1 = DummyChunk("c1", "P1", "text")
    # Add sources out of order
    c1.retrieval_sources = [
        {"source": "graph_bridge_retrieval"},
        {"source": "lexical"},
        {"source": "graph_concept_retrieval"},
        {"source": "dense"},
        {"source": "unknown_source_type"}
    ]
    
    dedupped = service._deduplicate_candidates([c1])
    sources = [s["source"] for s in dedupped[0].retrieval_sources]
    assert sources == [
        "dense",
        "lexical",
        "graph_concept_retrieval",
        "graph_bridge_retrieval",
        "unknown_source_type"
    ]

# =========================================================================
# Pipeline Integration Tests
# =========================================================================

@pytest.fixture
def base_service_setup():
    """Returns a RAGService setup with a fake graph repository and mocked components."""
    graph_repo = FakeGraphRepository()
    vector_repo = MagicMock()
    emb_engine = MagicMock()
    llm_engine = MagicMock()
    
    # Return 384 dimensional embedding
    emb_engine.get_embedding.return_value = [0.1] * 384
    
    service = RAGService(graph_repo, vector_repo, emb_engine, llm_engine)
    # Mock _get_reranker to return FakeReranker
    service._get_reranker = MagicMock(return_value=FakeReranker())
    
    return service, graph_repo, vector_repo, emb_engine

def test_pipeline_disabled_flags_no_op(base_service_setup):
    """Verify that when graph flags are disabled, pipeline returns base retrieval and does not call graph retrievers."""
    service, graph_repo, vector_repo, emb_engine = base_service_setup
    
    # Configure base flags disabled
    config.data["rag_components"]["graph_concept_retrieval"] = False
    config.data["rag_components"]["graph_bridge_retrieval"] = False
    
    # Mock base dense/lexical search
    c1 = Chunk(id="p1#0", paper_id="P1", text_content="Paper One text", page_number=1)
    vector_repo.search_similar_chunks.return_value = [(c1, 0.8)]
    vector_repo.search_text_fts5.return_value = []
    
    with patch('src.services.graph_retrievers.GraphConceptRetriever.retrieve') as mock_concept:
        with patch('src.services.graph_retrievers.GraphBridgeRetriever.retrieve') as mock_bridge:
            res = service.retrieve_relevant_chunks("Explain memory", limit=5)
            
            # Verify retrievers were not called
            mock_concept.assert_not_called()
            mock_bridge.assert_not_called()
            
            # Verify candidates returned match base search
            assert len(res) == 1
            assert res[0][0].id == "p1#0"

def test_pipeline_concept_retrieval_enabled(base_service_setup):
    """Verify concept retrieval adds candidates, which flow through reranker and preserve metadata."""
    service, graph_repo, vector_repo, emb_engine = base_service_setup
    
    config.data["rag_components"]["graph_concept_retrieval"] = True
    config.data["rag_components"]["graph_bridge_retrieval"] = False
    config.data["graph_retrieval"]["max_graph_candidate_papers"] = 2
    config.data["graph_retrieval"]["chunks_per_graph_paper"] = 1
    # Enable reranker with score blending false for simplicity
    config.data["rag_components"]["reranker"] = True
    config.data["rag_components"]["score_blending"] = False
    
    # Mock base search
    c1 = Chunk(id="p1#0", paper_id="P1", text_content="Paper One text", page_number=1)
    vector_repo.search_similar_chunks.return_value = [(c1, 0.8)]
    vector_repo.search_text_fts5.return_value = []
    
    # Call pipeline. Concept is extracted: "memory" -> c_memory.
    # Candidates mentioning c_memory: P3, P4.
    # Scoped chunk retrieval will return: p3#1 and p4#1 (best chunks)
    res = service.retrieve_relevant_chunks("Explain memory", limit=5)
    
    # Verify we returned graph candidates
    # Reranker scores: p4 (0.9), p3 (0.8), p1 (0.7).
    # Since limit is 5, and we have 3 candidates total, all should be returned sorted by reranker score
    assert len(res) == 3
    assert res[0][0].id == "p4#1" # scored highest by FakeReranker and is the best chunk for P4
    assert res[1][0].id == "p3#1"
    assert res[2][0].id == "p1#0"
    
    # Check source metadata preserved
    p4_chunk = res[0][0]
    assert hasattr(p4_chunk, "retrieval_sources")
    assert any(s["source"] == "graph_concept_retrieval" and s["paper_id"] == "P4" for s in p4_chunk.retrieval_sources)

def test_pipeline_bridge_retrieval_enabled(base_service_setup):
    """Verify bridge retrieval adds candidates connecting seeds and query concepts."""
    service, graph_repo, vector_repo, emb_engine = base_service_setup
    
    config.data["rag_components"]["graph_concept_retrieval"] = False
    config.data["rag_components"]["graph_bridge_retrieval"] = True
    config.data["graph_retrieval"]["max_graph_candidate_papers"] = 2
    config.data["graph_retrieval"]["chunks_per_graph_paper"] = 1
    config.data["rag_components"]["reranker"] = True
    config.data["rag_components"]["score_blending"] = False
    
    # Seed paper is P1 (from search)
    c1 = Chunk(id="p1#0", paper_id="P1", text_content="Paper One text", page_number=1)
    vector_repo.search_similar_chunks.return_value = [(c1, 0.8)]
    vector_repo.search_text_fts5.return_value = []
    
    # Query mentions "memory" (c_memory)
    # Bridge candidates from P1 seed via c_memory: P3, P4
    res = service.retrieve_relevant_chunks("Explain memory", limit=5)
    
    assert len(res) == 3
    pids = {c[0].paper_id for c in res}
    assert "P3" in pids
    assert "P4" in pids
    
    # Bridge reason metadata check
    p3_chunk = next(c[0] for c in res if c[0].paper_id == "P3")
    assert any(s["source"] == "graph_bridge_retrieval" and "connected_seed_papers" in s for s in p3_chunk.retrieval_sources)

def test_pipeline_both_retrievers_enabled_dedup(base_service_setup):
    """Verify that both retrievers enabled does not duplicate chunks and merges metadata correctly."""
    service, graph_repo, vector_repo, emb_engine = base_service_setup
    
    config.data["rag_components"]["graph_concept_retrieval"] = True
    config.data["rag_components"]["graph_bridge_retrieval"] = True
    config.data["graph_retrieval"]["max_graph_candidate_papers"] = 2
    config.data["graph_retrieval"]["chunks_per_graph_paper"] = 1
    config.data["rag_components"]["reranker"] = False # RRF fallback
    
    # Seed is P1
    c1 = Chunk(id="p1#0", paper_id="P1", text_content="Paper One text", page_number=1)
    vector_repo.search_similar_chunks.return_value = [(c1, 0.8)]
    vector_repo.search_text_fts5.return_value = []
    
    # Query mentions "memory" (c_memory).
    # Concept retrieval adds P3, P4. Bridge retrieval adds P3, P4.
    # Deduplication should yield 3 candidates total (P1, P3, P4)
    res = service.retrieve_relevant_chunks("Explain memory", limit=5)
    
    assert len(res) == 3
    pids = [c[0].paper_id for c in res]
    assert sorted(pids) == ["P1", "P3", "P4"]
    
    # P3 and P4 chunks should contain both concept and bridge retrieval sources
    p4_chunk = next(c[0] for c in res if c[0].paper_id == "P4")
    sources = [s["source"] for s in p4_chunk.retrieval_sources]
    assert "graph_concept_retrieval" in sources
    assert "graph_bridge_retrieval" in sources

def test_pipeline_empty_query_concepts(base_service_setup):
    """Verify that if query has no concepts, no graph retrieval runs but pipeline returns base results."""
    service, graph_repo, vector_repo, emb_engine = base_service_setup
    
    config.data["rag_components"]["graph_concept_retrieval"] = True
    config.data["rag_components"]["graph_bridge_retrieval"] = True
    
    c1 = Chunk(id="p1#0", paper_id="P1", text_content="Paper One text", page_number=1)
    vector_repo.search_similar_chunks.return_value = [(c1, 0.8)]
    vector_repo.search_text_fts5.return_value = []
    
    # Query with no concept matches
    res = service.retrieve_relevant_chunks("Hello world, nothing here", limit=5)
    
    # Returns only base results
    assert len(res) == 1
    assert res[0][0].id == "p1#0"
