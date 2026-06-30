from unittest.mock import MagicMock
from src.services.rag_service import RAGService
from tests.graph_test_utils import FakeGraphRepository

class DummyChunk:
    def __init__(self, id, paper_id, text_content):
        self.id = id
        self.paper_id = paper_id
        self.text_content = text_content
        self.retrieval_sources = []

def test_empty_query_returns_empty():
    """Verify that an empty or whitespace-only query string returns empty lists or handles gracefully without crash."""
    service = RAGService(FakeGraphRepository(), MagicMock(), MagicMock(), MagicMock())
    
    # 1. Concept extraction on empty string
    concepts = service._extract_query_concepts("")
    assert concepts == []
    
    concepts_spaces = service._extract_query_concepts("   ")
    assert concepts_spaces == []
    
    # 2. Retrieve chunks on empty string
    # Should not invoke graph retrievers since no concepts can be extracted
    vector_repo = MagicMock()
    vector_repo.search_similar_chunks.return_value = []
    vector_repo.search_text_fts5.return_value = []
    service.vector_repo = vector_repo
    
    res = service.retrieve_relevant_chunks("", limit=5)
    assert res == []

def test_search_chunks_empty_papers():
    """Verify search_chunks_within_papers returns empty list when paper list is empty."""
    repo = FakeGraphRepository()
    res = repo.search_chunks_within_papers([0.1]*384, [])
    assert res == []

def test_missing_node_properties_handled():
    """Verify that missing titles or names in graph nodes do not crash selected sources card builder."""
    service = RAGService(FakeGraphRepository(), MagicMock(), MagicMock(), MagicMock())
    
    # Mock graph repo to return node structures with missing properties
    service.graph_repo = MagicMock()
    service.graph_repo.get_total_paper_count.return_value = 10
    
    # Concept with no name/aliases properties, Paper with no title
    service.graph_repo.get_concepts_for_papers.return_value = [
        ("P1", "c_memory", None),  # None name
    ]
    service.graph_repo.get_citation_neighbors.return_value = [
        ("P1", "P2", "seed_cites_candidate", None)  # None title
    ]
    
    trimmed_chunks = [
        (DummyChunk("p1#0", "P1", "t1"), 0.9),
        (DummyChunk("p2#0", "P2", "t2"), 0.8)
    ]
    
    # Should run and return empty card or not crash
    card = service._build_selected_sources_card(trimmed_chunks, ["c_memory"])
    if card is not None:
        assert isinstance(card, str)

def test_circular_or_multiple_concept_aliases():
    """Verify that concept alias mapping handles multiple mappings or circular structures gracefully."""
    service = RAGService(FakeGraphRepository(), MagicMock(), MagicMock(), MagicMock())
    
    # Mock the database calls that populate aliases
    service.graph_repo = MagicMock()
    service.graph_repo.get_concept_aliases.return_value = {
        "memory": "c_memory",
        "ltm": "c_memory",
        "agent": "c_agents",
        "agents": "c_agents",
        "loop": "loop",
    }
    service.graph_repo.get_nodes_by_label.return_value = [
        ("c_memory", {"name": "memory", "aliases": ["ltm", "working memory"]}),
        ("c_agents", {"name": "agents", "aliases": ["agentic"]}),
        ("loop", {"name": "loop", "aliases": []})
    ]
    
    concepts = service._extract_query_concepts("memory agents loop")
    assert "c_memory" in concepts
    assert "c_agents" in concepts
    assert "loop" in concepts

def test_nan_or_zero_dimension_embeddings():
    """Verify that NaN or zero-dimension embeddings in search query do not raise exceptions."""
    repo = FakeGraphRepository()
    
    # 1. Zero-dimension query embedding: should return chunks with similarity 0.0
    res_zero = repo.search_chunks_within_papers([], ["P1"])
    assert len(res_zero) == 1
    assert res_zero[0][1] == 0.0
    
    # 2. NaN/Inf query embedding
    nan_emb = [float('nan')] * 384
    res_nan = repo.search_chunks_within_papers(nan_emb, ["P1"])
    # Cosine similarity with nan will return nan or 0.0, shouldn't crash
    assert isinstance(res_nan, list)
