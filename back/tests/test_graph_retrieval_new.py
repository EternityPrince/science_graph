from unittest.mock import MagicMock
from src.services.graph_retrievers import GraphConceptRetriever, GraphBridgeRetriever
from src.services.rag_service import RAGService
from src.config import Config

class DummyChunk:
    def __init__(self, id, paper_id, text_content, chunk_index=0):
        self.id = id
        self.paper_id = paper_id
        self.text_content = text_content
        self.chunk_index = chunk_index
        self.retrieval_sources = []

def test_extract_query_concepts():
    mock_graph_repo = MagicMock()
    mock_vector_repo = MagicMock()
    
    mock_graph_repo.get_nodes_by_label.return_value = [
        ("long_term_memory", {"name": "Long-Term Memory", "aliases": ["LTM", "long term memory"]}),
        ("neural_networks", {"name": "Neural Networks", "aliases": []})
    ]
    mock_graph_repo.get_concept_aliases.return_value = {
        "nn": "neural_networks"
    }
    
    service = RAGService(mock_graph_repo, mock_vector_repo, MagicMock(), MagicMock())
    
    c1 = service._extract_query_concepts("What is LTM?")
    assert "long_term_memory" in c1
    
    c2 = service._extract_query_concepts("Tell me about Neural Networks and nn")
    assert "neural_networks" in c2

def test_graph_concept_retriever():
    mock_graph_repo = MagicMock()
    mock_graph_repo.get_total_paper_count.return_value = 10
    mock_graph_repo.get_concept_document_frequencies.return_value = {
        "concept_a": 2,
        "concept_b": 5
    }
    
    def get_papers(concepts):
        res = []
        if "concept_a" in concepts:
            res.extend([("paper_1", "Title 1"), ("paper_2", "Title 2")])
        if "concept_b" in concepts:
            res.extend([("paper_2", "Title 2"), ("paper_3", "Title 3")])
        return list(set(res))
    mock_graph_repo.get_papers_mentioning_concepts.side_effect = get_papers
    
    mock_graph_repo.get_concepts_for_papers.return_value = [
        ("paper_1", "concept_a", "Concept A"),
        ("paper_2", "concept_a", "Concept A"),
        ("paper_2", "concept_b", "Concept B"),
        ("paper_3", "concept_b", "Concept B")
    ]
    
    retriever = GraphConceptRetriever(mock_graph_repo)
    
    candidates = retriever.retrieve(
        query="test query",
        query_concepts=["concept_a", "concept_b"],
        exclude_paper_ids=["paper_3"],
        max_candidate_papers=2
    )
    
    assert len(candidates) == 2
    assert candidates[0]["paper_id"] == "paper_2"
    assert candidates[1]["paper_id"] == "paper_1"

def test_graph_bridge_retriever():
    mock_graph_repo = MagicMock()
    mock_graph_repo.get_total_paper_count.return_value = 100
    mock_graph_repo.get_concept_document_frequencies.return_value = {
        "concept_a": 10,
        "concept_b": 20
    }
    
    mock_graph_repo.get_citation_neighbors.return_value = [
        ("paper_1", "paper_2", "seed_cites_candidate", "Title 2"),
        ("paper_1", "paper_3", "candidate_cites_seed", "Title 3")
    ]
    
    mock_graph_repo.get_concepts_for_papers.return_value = [
        ("paper_2", "concept_a", "Concept A"),
        ("paper_3", "concept_b", "Concept B")
    ]
    
    retriever = GraphBridgeRetriever(mock_graph_repo)
    candidates = retriever.retrieve(
        query="test query",
        seed_paper_ids=["paper_1"],
        query_concepts=["concept_a", "concept_b"],
        exclude_paper_ids=["paper_1"],
        max_candidate_papers=2
    )
    
    assert len(candidates) == 2
    assert candidates[0]["paper_id"] == "paper_2"
    assert candidates[1]["paper_id"] == "paper_3"
    assert candidates[0]["min_graph_distance"] == 1
    assert len(candidates[0]["paths"]) > 0

def test_deduplicate_candidates():
    service = RAGService(MagicMock(), MagicMock(), MagicMock(), MagicMock())
    
    c1 = DummyChunk("chunk_1", "paper_1", "text one")
    c1.retrieval_sources.append({"source": "dense"})
    
    c2 = DummyChunk("chunk_1", "paper_1", "text one")
    c2.retrieval_sources.append({"source": "graph_concept_retrieval"})
    
    c3 = DummyChunk("chunk_2", "paper_2", "text two")
    c3.retrieval_sources.append({"source": "lexical"})
    
    dedupped = service._deduplicate_candidates([c1, c2, c3])
    
    assert len(dedupped) == 2
    chunk1 = next(c for c in dedupped if c.id == "chunk_1")
    sources = [s["source"] for s in chunk1.retrieval_sources]
    assert "dense" in sources
    assert "graph_concept_retrieval" in sources

def test_build_selected_sources_card():
    service = RAGService(MagicMock(), MagicMock(), MagicMock(), MagicMock())
    
    service.graph_repo = MagicMock()
    service.graph_repo.get_concepts_for_papers.return_value = [
        ("paper_1", "concept_ltm", "long-term memory"),
        ("paper_2", "concept_ltm", "long-term memory"),
        ("paper_1", "concept_eval", "agent evaluation"),
        ("paper_3", "concept_eval", "agent evaluation")
    ]
    service.graph_repo.get_citation_neighbors.return_value = [
        ("paper_2", "paper_3", "seed_cites_candidate", "Title 3")
    ]
    service.graph_repo.get_total_paper_count.return_value = 100
    service.graph_repo.get_concept_document_frequencies.return_value = {
        "concept_ltm": 5,
        "concept_eval": 10
    }
    
    trimmed_chunks = [
        (DummyChunk("c1", "paper_1", "text 1"), 0.9),
        (DummyChunk("c2", "paper_2", "text 2"), 0.8),
        (DummyChunk("c3", "paper_3", "text 3"), 0.7)
    ]
    
    card = service._build_selected_sources_card(trimmed_chunks, ["concept_ltm"])
    
    assert card is not None
    assert "Graph links among selected sources:" in card
    assert "[1] and [2] both mention concept \"long-term memory\"" in card
    assert "[2] cites [3]" in card
    assert "[1] and [3] are connected through concept \"agent evaluation\"" in card

def test_trace_logging(tmp_path, monkeypatch):
    import json
    
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Config, "graph_retrieval_trace_enabled", True)
    
    service = RAGService(MagicMock(), MagicMock(), MagicMock(), MagicMock())
    service.current_trace = {"query_id": "test-query-id"}
    
    service._last_graph_trace = {
        "query_concepts": ["concept_a"],
        "seed_paper_ids": ["paper_1"],
        "graph_concept_candidate_papers": ["paper_2"],
        "graph_bridge_candidate_papers": [],
        "graph_chunks_before_rerank": ["chunk_2"],
        "graph_chunks_before_rerank_count": 1,
        "graph_candidate_source_breakdown": {"graph_concept_retrieval": 1}
    }
    
    final_chunks = [
        (DummyChunk("chunk_1", "paper_1", "t1"), 0.9),
        (DummyChunk("chunk_2", "paper_2", "t2"), 0.85)
    ]
    final_chunks[1][0].retrieval_sources.append({"source": "graph_concept_retrieval"})
    
    trimmed_chunks = [
        (DummyChunk("chunk_2", "paper_2", "t2"), 0.85)
    ]
    
    service._write_graph_retrieval_trace("test query", final_chunks, trimmed_chunks)
    
    trace_file = tmp_path / "graph_retrieval_trace.jsonl"
    assert trace_file.exists()
    
    with open(trace_file, "r") as f:
        lines = f.readlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["query_id"] == "test-query-id"
        assert data["graph_chunks_survived_final_context"] == ["chunk_2"]
        assert data["graph_survival_rate"] == 1.0
        assert data["best_graph_candidate_rank_after_rerank"] == 2


def test_get_neighbor_papers():
    from tests.graph_test_utils import FakeGraphRepository
    repo = FakeGraphRepository()
    neighbors = repo.get_neighbor_papers(["P1"], order=2)
    assert sorted(neighbors) == ["P3", "P4"]
