import math
import pytest
from src.services.graph_retrievers import GraphBridgeRetriever
from tests.graph_test_utils import FakeGraphRepository

def test_bridge_retriever_shared_query_concept():
    """Verify that candidate is found when sharing a query concept with a seed paper."""
    repo = FakeGraphRepository()
    retriever = GraphBridgeRetriever(repo)
    
    # Seed is P2 (mentions c_agents). Query concept is c_agents.
    # P4 mentions c_agents. So P4 is a bridge candidate.
    candidates = retriever.retrieve(
        query="test",
        seed_paper_ids=["P2"],
        query_concepts=["c_agents"],
        exclude_paper_ids=["P2"],
        max_candidate_papers=5
    )
    pids = [c["paper_id"] for c in candidates]
    assert "P4" in pids
    
    p4_cand = next(c for c in candidates if c["paper_id"] == "P4")
    # Verify path metadata
    assert any(p["type"] == "seed_shared_query_concept" and p["seed_paper_id"] == "P2" and p["concept_id"] == "c_agents" for p in p4_cand["paths"])

def test_bridge_retriever_citation_neighbor():
    """Verify that candidate is found when it is a citation neighbor of a seed paper AND mentions a query concept."""
    repo = FakeGraphRepository()
    retriever = GraphBridgeRetriever(repo)
    
    # Seed is P1. P1 cites P3. P3 mentions c_memory (query concept).
    candidates = retriever.retrieve(
        query="test",
        seed_paper_ids=["P1"],
        query_concepts=["c_memory"],
        exclude_paper_ids=["P1"],
        max_candidate_papers=5
    )
    pids = [c["paper_id"] for c in candidates]
    assert "P3" in pids
    
    p3_cand = next(c for c in candidates if c["paper_id"] == "P3")
    assert p3_cand["min_graph_distance"] == 1
    assert any(p["type"] == "seed_citation_neighbor_with_query_concept" and p["seed_paper_id"] == "P1" and p["direction"] == "seed_cites_candidate" for p in p3_cand["paths"])

def test_bridge_retriever_reverse_citation_neighbor():
    """Verify that candidate is found when it cites a seed paper AND mentions a query concept."""
    repo = FakeGraphRepository()
    # Add a reverse citation edge: P6 cites P2, and P6 mentions query concept c_agents
    repo.edges.append(("P6", "P2", "CITES"))
    repo.edges.append(("P6", "c_agents", "MENTIONS_CONCEPT"))
    repo.chunks["P6"] = []
    
    retriever = GraphBridgeRetriever(repo)
    candidates = retriever.retrieve(
        query="test",
        seed_paper_ids=["P2"],
        query_concepts=["c_agents"],
        exclude_paper_ids=["P2"],
        max_candidate_papers=5
    )
    pids = [c["paper_id"] for c in candidates]
    assert "P6" in pids
    
    p6_cand = next(c for c in candidates if c["paper_id"] == "P6")
    assert any(p["type"] == "seed_citation_neighbor_with_query_concept" and p["seed_paper_id"] == "P2" and p["direction"] == "candidate_cites_seed" for p in p6_cand["paths"])

def test_bridge_retriever_bridge_between_two_seeds():
    """Verify that candidate is found when it bridges two distinct seed papers through a shared concept."""
    repo = FakeGraphRepository()
    # Seed papers: P1 and P2.
    # Add a concept c_shared that is mentioned by P1, P2 and candidate P6.
    repo.concepts["c_shared"] = {"id": "c_shared", "label": "Concept", "title": "shared", "properties": {"name": "shared", "aliases": []}}
    repo.edges.append(("P1", "c_shared", "MENTIONS_CONCEPT"))
    repo.edges.append(("P2", "c_shared", "MENTIONS_CONCEPT"))
    repo.edges.append(("P6", "c_shared", "MENTIONS_CONCEPT"))
    
    retriever = GraphBridgeRetriever(repo)
    # Query concepts doesn't have c_shared, but P6 mentions it.
    # Candidate P6 bridges seeds P1 and P2 via c_shared.
    candidates = retriever.retrieve(
        query="test",
        seed_paper_ids=["P1", "P2"],
        query_concepts=["c_general"],  # P6 mentions c_general (so it has query concept match)
        exclude_paper_ids=["P1", "P2"],
        max_candidate_papers=5
    )
    pids = [c["paper_id"] for c in candidates]
    assert "P6" in pids
    
    p6_cand = next(c for c in candidates if c["paper_id"] == "P6")
    bridge_paths = [p for p in p6_cand["paths"] if p["type"] == "seed_shared_concept" and p["concept_id"] == "c_shared"]
    assert len(bridge_paths) >= 2
    seeds_connected = {p["seed_paper_id"] for p in bridge_paths}
    assert seeds_connected == {"P1", "P2"}

def test_bridge_retriever_excludes_unrelated_and_concept_only():
    """Verify that unrelated papers and papers with query concepts but no seed connections are excluded."""
    repo = FakeGraphRepository()
    retriever = GraphBridgeRetriever(repo)
    
    # Query concepts includes c_general (P6 mentions it) and c_unrelated (P5 mentions it)
    # Seed is P1.
    # P5 mentions c_unrelated but has no connection to P1.
    # P6 mentions c_general but has no connection to P1.
    candidates = retriever.retrieve(
        query="test",
        seed_paper_ids=["P1"],
        query_concepts=["c_general", "c_unrelated"],
        exclude_paper_ids=["P1"],
        max_candidate_papers=5
    )
    pids = [c["paper_id"] for c in candidates]
    assert "P5" not in pids
    assert "P6" not in pids

def test_bridge_retriever_exclude_seeds():
    """Verify that seed papers themselves are excluded from candidates."""
    repo = FakeGraphRepository()
    retriever = GraphBridgeRetriever(repo)
    
    # P1 seed mentions c_memory.
    # If not excluded, it could look like a bridge to itself.
    candidates = retriever.retrieve(
        query="test",
        seed_paper_ids=["P1"],
        query_concepts=["c_memory"],
        exclude_paper_ids=["P1"],
        max_candidate_papers=5
    )
    pids = [c["paper_id"] for c in candidates]
    assert "P1" not in pids

def test_bridge_retriever_sorting():
    """Verify bridge retriever sorting: query concepts desc, connected seeds desc, distance asc, idf desc, paper_id asc."""
    repo = FakeGraphRepository()
    retriever = GraphBridgeRetriever(repo)
    
    # Seed papers: P1, P2. Query concepts: c_memory, c_agents.
    # Candidates:
    # P4: mentions c_memory and c_agents (2 query concepts), connected seeds: P1, P2 (2 seeds), min_dist = 2
    # P3: mentions c_memory (1 query concept), connected seeds: P1 (1 seed), min_dist = 1 (cites neighbor)
    candidates = retriever.retrieve(
        query="test",
        seed_paper_ids=["P1", "P2"],
        query_concepts=["c_memory", "c_agents"],
        exclude_paper_ids=["P1", "P2"],
        max_candidate_papers=10
    )
    
    assert candidates[0]["paper_id"] == "P4" # 2 concepts, 2 seeds
    assert candidates[1]["paper_id"] == "P3" # 1 concept, 1 seed, dist = 1

def test_bridge_retriever_empty_inputs():
    """Verify retriever returns empty on empty query concepts or empty seed papers."""
    repo = FakeGraphRepository()
    retriever = GraphBridgeRetriever(repo)
    
    assert retriever.retrieve("test", [], ["c_memory"], ["P1"], 5) == []
    assert retriever.retrieve("test", ["P1"], [], ["P1"], 5) == []
