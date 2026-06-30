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

def test_build_selected_sources_card_basic():
    """Verify standard selected sources card contains correct facts and references."""
    service = RAGService(FakeGraphRepository(), MagicMock(), MagicMock(), MagicMock())
    
    # Trimmed chunks correspond to P1, P3, P4
    # Their indices in the final context (1-based) will be:
    # 1: P1, 2: P3, 3: P4
    trimmed_chunks = [
        (DummyChunk("p1#0", "P1", "text 1"), 0.9),
        (DummyChunk("p3#0", "P3", "text 2"), 0.8),
        (DummyChunk("p4#0", "P4", "text 3"), 0.7)
    ]
    
    # Query mentions c_memory
    card = service._build_selected_sources_card(trimmed_chunks, ["c_memory"])
    
    assert card is not None
    assert "Graph links among selected sources:" in card
    # [1] (P1) and [2] (P3) both mention concept "memory" (query concept)
    assert "[1] and [2] both mention concept \"memory\"" in card
    # [1] cites [2] (P1 cites P3)
    assert "[1] cites [2]" in card
    # [2] and [3] both mention concept "memory"
    assert "[2] and [3] both mention concept \"memory\"" in card
    # [1] and [3] both mention concept "memory"
    assert "[1] and [3] both mention concept \"memory\"" in card

def test_build_selected_sources_card_no_links():
    """Verify that card is skipped (None) when no relations exist among selected sources."""
    service = RAGService(FakeGraphRepository(), MagicMock(), MagicMock(), MagicMock())
    
    # P1 and P5 (unrelated)
    trimmed_chunks = [
        (DummyChunk("p1#0", "P1", "text 1"), 0.9),
        (DummyChunk("p5#0", "P5", "text 2"), 0.8)
    ]
    
    card = service._build_selected_sources_card(trimmed_chunks, ["c_general"])
    assert card is None

def test_build_selected_sources_card_prioritization_and_limit():
    """Verify prioritization: query concepts first, citation second, shared concepts third, max 5 facts."""
    service = RAGService(FakeGraphRepository(), MagicMock(), MagicMock(), MagicMock())
    
    # We will mock repo results to return multiple facts:
    # Let's say we have 6 facts in total.
    # Query concepts: c_memory
    # 3 query concept facts, 2 citation facts, 2 shared concept facts.
    service.graph_repo = MagicMock()
    service.graph_repo.get_total_paper_count.return_value = 100
    
    service.graph_repo.get_concepts_for_papers.return_value = [
        ("P1", "c_memory", "memory"),
        ("P2", "c_memory", "memory"),
        ("P3", "c_memory", "memory"),
        ("P1", "c_other1", "other one"),
        ("P2", "c_other1", "other one"),
        ("P1", "c_other2", "other two"),
        ("P3", "c_other2", "other two"),
        ("P2", "c_other3", "other three"),
        ("P3", "c_other3", "other three")
    ]
    service.graph_repo.get_citation_neighbors.return_value = [
        ("P1", "P2", "seed_cites_candidate", "Paper Two"),
        ("P2", "P3", "seed_cites_candidate", "Paper Three")
    ]
    # Doc frequencies for other concepts
    service.graph_repo.get_concept_document_frequencies.return_value = {
        "c_other1": 5,
        "c_other2": 10,
        "c_other3": 15
    }
    
    trimmed_chunks = [
        (DummyChunk("p1#0", "P1", "t1"), 0.9),
        (DummyChunk("p2#0", "P2", "t2"), 0.8),
        (DummyChunk("p3#0", "P3", "t3"), 0.7)
    ]
    
    card = service._build_selected_sources_card(trimmed_chunks, ["c_memory"])
    
    assert card is not None
    lines = [line.strip() for line in card.split("\n") if line.strip()]
    # First line is header, so max 6 lines total (1 header + 5 facts)
    assert len(lines) <= 6
    
    # Check that query concepts are listed first
    assert any("both mention concept \"memory\"" in l for l in lines[1:4])
    
    # Check that citations are next
    assert "- [1] cites [2]." in lines[4]
    assert "- [2] cites [3]." in lines[5]
    
    # Verify that the 6th/7th facts (shared concepts with other1/other2/other3) are cut off
    assert not any("other one" in l for l in lines)
    assert not any("other two" in l for l in lines)

def test_build_selected_sources_card_citation_direction():
    """Verify citation direction formats correctly (e.g., [1] cites [2] and not reversed)."""
    service = RAGService(FakeGraphRepository(), MagicMock(), MagicMock(), MagicMock())
    
    # P1 cites P3.
    # final context: 1: P3, 2: P1
    trimmed_chunks = [
        (DummyChunk("p3#0", "P3", "text 2"), 0.8),
        (DummyChunk("p1#0", "P1", "text 1"), 0.9)
    ]
    
    card = service._build_selected_sources_card(trimmed_chunks, [])
    assert card is not None
    # [2] cites [1] because P1 cites P3, and P1 is at index 2, P3 is at index 1
    assert "[2] cites [1]" in card
    assert "[1] cites [2]" not in card

def test_pipeline_card_integration(monkeypatch):
    """Verify card is appended to final answer only when graph_selected_sources_card_enabled is True."""
    # Mock LLM and retrieval
    graph_repo = FakeGraphRepository()
    vector_repo = MagicMock()
    emb_engine = MagicMock()
    emb_engine.get_embedding.return_value = [0.1] * 384
    llm_engine = MagicMock()
    llm_engine.generate_response.return_value = "This is the answer."
    
    service = RAGService(graph_repo, vector_repo, emb_engine, llm_engine)
    
    # Mock base search
    c1 = Chunk(id="p1#0", paper_id="P1", text_content="Paper One text", page_number=1)
    vector_repo.search_similar_chunks.return_value = [(c1, 0.8)]
    vector_repo.search_text_fts5.return_value = []
    
    # Mock trim_context to return P1 and P3 chunks
    c3 = Chunk(id="p3#0", paper_id="P3", text_content="Paper Three text", page_number=1)
    service.trim_context = MagicMock(return_value=(
        "P1 and P3 text",
        "graph text",
        [(c1, 0.9), (c3, 0.8)]
    ))
    
    # 1. Card disabled
    config.data["rag_components"]["graph_selected_sources_card"] = False
    ans_disabled = service.ask("Explain memory")
    assert "Graph links among selected sources:" not in ans_disabled
    
    # 2. Card enabled
    config.data["rag_components"]["graph_selected_sources_card"] = True
    ans_enabled = service.ask("Explain memory")
    assert "Graph links among selected sources:" in ans_enabled
    assert "[1] and [2] both mention concept \"memory\"" in ans_enabled
