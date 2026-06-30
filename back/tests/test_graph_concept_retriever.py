import math
from unittest.mock import MagicMock
from src.services.graph_retrievers import GraphConceptRetriever
from src.services.rag_service import RAGService
from tests.graph_test_utils import FakeGraphRepository

def test_extract_query_concepts_canonical():
    """Verify that exact canonical label is matched in query concept extraction."""
    repo = FakeGraphRepository()
    service = RAGService(repo, MagicMock(), MagicMock(), MagicMock())
    
    # Query mentions "memory" which is the canonical name for c_memory
    concepts = service._extract_query_concepts("Explain memory systems")
    assert "c_memory" in concepts

def test_extract_query_concepts_alias():
    """Verify that alias is matched to canonical concept ID in query concept extraction."""
    repo = FakeGraphRepository()
    service = RAGService(repo, MagicMock(), MagicMock(), MagicMock())
    
    # Query mentions "ltm" which is an alias for c_memory
    concepts = service._extract_query_concepts("Explain LTM systems")
    assert "c_memory" in concepts

def test_extract_query_concepts_lemmatized():
    """Verify that lemmatized forms of aliases/names match successfully."""
    repo = FakeGraphRepository()
    service = RAGService(repo, MagicMock(), MagicMock(), MagicMock())
    
    # Query mentions "agents" or "agentic" - lemma should match
    concepts = service._extract_query_concepts("Explain agentic behavior")
    assert "c_agents" in concepts

def test_extract_query_concepts_word_boundaries():
    """Verify that concept words do not match inside larger words (e.g. 'agent' in 'reagent')."""
    repo = FakeGraphRepository()
    service = RAGService(repo, MagicMock(), MagicMock(), MagicMock())
    
    # Query mentions "reagent", which has "agent" as a substring but shouldn't match c_agents
    concepts = service._extract_query_concepts("We need a chemical reagent here")
    assert "c_agents" not in concepts

def test_extract_query_concepts_stable_dedup():
    """Verify that matching multiple aliases of the same concept deduplicates and keeps stable sorted order."""
    repo = FakeGraphRepository()
    service = RAGService(repo, MagicMock(), MagicMock(), MagicMock())
    
    # Query mentions "memory" and "LTM" both matching c_memory, and "agents" matching c_agents
    concepts = service._extract_query_concepts("Discuss working memory and ltm and agentic models")
    assert concepts == ["c_agents", "c_memory"]

def test_extract_query_concepts_empty_and_no_match():
    """Verify empty query or no-match inputs return empty list."""
    repo = FakeGraphRepository()
    service = RAGService(repo, MagicMock(), MagicMock(), MagicMock())
    
    assert service._extract_query_concepts("") == []
    assert service._extract_query_concepts("   ") == []
    assert service._extract_query_concepts(None) == []
    assert service._extract_query_concepts("hello world how are you") == []

def test_extract_query_concepts_high_frequency_not_filtered():
    """Verify that high-frequency concepts (like c_general) are not filtered out at extraction time."""
    repo = FakeGraphRepository()
    service = RAGService(repo, MagicMock(), MagicMock(), MagicMock())
    
    concepts = service._extract_query_concepts("Discuss common and general things")
    assert "c_general" in concepts


def test_graph_concept_retriever_basic():
    """Verify retriever finds papers by query concept."""
    repo = FakeGraphRepository()
    retriever = GraphConceptRetriever(repo)
    
    candidates = retriever.retrieve(
        query="test",
        query_concepts=["c_memory"],
        exclude_paper_ids=[],
        max_candidate_papers=5
    )
    # P1, P3, P4 mention c_memory
    pids = [c["paper_id"] for c in candidates]
    assert "P3" in pids
    assert "P4" in pids
    assert "P1" in pids

def test_graph_concept_retriever_exclude_seed():
    """Verify candidate list excludes seed papers."""
    repo = FakeGraphRepository()
    retriever = GraphConceptRetriever(repo)
    
    candidates = retriever.retrieve(
        query="test",
        query_concepts=["c_memory"],
        exclude_paper_ids=["P1"],
        max_candidate_papers=5
    )
    pids = [c["paper_id"] for c in candidates]
    assert "P1" not in pids
    assert "P3" in pids
    assert "P4" in pids

def test_graph_concept_retriever_idf_math():
    """Verify IDF formula exactly matches log((1 + total_papers) / (1 + df))."""
    repo = FakeGraphRepository()
    retriever = GraphConceptRetriever(repo)
    
    # query_concepts = [c_memory, c_agents]
    # total papers = 6
    # df(c_memory) = 3 (P1, P3, P4)
    # df(c_agents) = 2 (P2, P4)
    # idf(c_memory) = log((1+6) / (1+3)) = log(7/4) = log(1.75) ~ 0.5596
    # idf(c_agents) = log((1+6) / (1+2)) = log(7/3) = log(2.333) ~ 0.8473
    candidates = retriever.retrieve(
        query="test",
        query_concepts=["c_memory", "c_agents"],
        exclude_paper_ids=["P1", "P2"],
        max_candidate_papers=5
    )
    
    p3_cand = next(c for c in candidates if c["paper_id"] == "P3")
    p4_cand = next(c for c in candidates if c["paper_id"] == "P4")
    
    expected_p3_idf = math.log(7.0 / 4.0)
    assert abs(p3_cand["concept_idf_sum"] - expected_p3_idf) < 1e-6
    
    expected_p4_idf = math.log(7.0 / 4.0) + math.log(7.0 / 3.0)
    assert abs(p4_cand["concept_idf_sum"] - expected_p4_idf) < 1e-6

def test_graph_concept_retriever_sorting():
    """Verify sorting order: matched concept count desc, then idf sum desc, then paper_id asc."""
    repo = FakeGraphRepository()
    retriever = GraphConceptRetriever(repo)
    
    # For concepts c_memory and c_agents:
    # P4: mentions 2 query concepts (c_memory, c_agents) -> count=2, idf_sum=1.4069
    # P2: mentions 1 query concept (c_agents) -> count=1, idf_sum=0.8473
    # P1: mentions 1 query concept (c_memory) -> count=1, idf_sum=0.5596
    # P3: mentions 1 query concept (c_memory) -> count=1, idf_sum=0.5596
    candidates = retriever.retrieve(
        query="test",
        query_concepts=["c_memory", "c_agents"],
        exclude_paper_ids=[],
        max_candidate_papers=10
    )
    
    # Expected: P4, P2, P1, P3
    assert candidates[0]["paper_id"] == "P4"
    assert candidates[1]["paper_id"] == "P2"
    assert candidates[2]["paper_id"] == "P1"
    assert candidates[3]["paper_id"] == "P3"

def test_graph_concept_retriever_limits_and_empty():
    """Verify limits, empty query_concepts and max_candidate_papers=0 behaviors."""
    repo = FakeGraphRepository()
    retriever = GraphConceptRetriever(repo)
    
    cands_limit = retriever.retrieve(
        query="test",
        query_concepts=["c_memory"],
        exclude_paper_ids=[],
        max_candidate_papers=2
    )
    assert len(cands_limit) == 2
    
    assert retriever.retrieve("test", ["c_memory"], [], 0) == []
    assert retriever.retrieve("test", [], [], 5) == []

def test_graph_concept_retriever_reason_metadata():
    """Verify reason metadata fields are fully populated and correct."""
    repo = FakeGraphRepository()
    retriever = GraphConceptRetriever(repo)
    
    candidates = retriever.retrieve(
        query="test",
        query_concepts=["c_memory"],
        exclude_paper_ids=[],
        max_candidate_papers=1
    )
    assert len(candidates) == 1
    c = candidates[0]
    assert c["source"] == "graph_concept_retrieval"
    assert c["paper_id"] in ("P1", "P3", "P4")
    assert c["matched_concepts"] == ["c_memory"]
    assert c["reason"] == "paper_mentions_query_concept"
    assert "concept_idf_sum" in c
