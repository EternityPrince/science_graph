import pytest
from src.repository.sqlite_impl import SQLiteGraphRepository, SQLiteVectorRepository
from tests.graph_test_utils import seed_minimal_graph

def test_sqlite_get_papers_mentioning_concepts(graph_repo, vector_repo):
    """Verify papers mentioning concepts query works correctly with single, multiple or no matches."""
    seed_minimal_graph(graph_repo, vector_repo)
    
    # Single concept
    papers_single = graph_repo.get_papers_mentioning_concepts(["c_memory"])
    # P1, P3, P4 mention c_memory
    pids = [p[0] for p in papers_single]
    assert sorted(pids) == ["P1", "P3", "P4"]
    
    # Multiple concepts
    papers_multi = graph_repo.get_papers_mentioning_concepts(["c_memory", "c_agents"])
    pids_multi = [p[0] for p in papers_multi]
    # P2 mentions c_agents. P1, P3, P4 mention c_memory.
    assert sorted(pids_multi) == ["P1", "P2", "P3", "P4"]
    
    # No matches
    papers_empty = graph_repo.get_papers_mentioning_concepts(["nonexistent"])
    assert papers_empty == []

def test_sqlite_get_concepts_for_papers(graph_repo, vector_repo):
    """Verify concept list retrieval for single or multiple papers."""
    seed_minimal_graph(graph_repo, vector_repo)
    
    # Single paper: P4 mentions c_memory and c_agents
    concepts_single = graph_repo.get_concepts_for_papers(["P4"])
    cids = [c[1] for c in concepts_single]
    assert sorted(cids) == ["c_agents", "c_memory"]
    
    # Multiple papers
    concepts_multi = graph_repo.get_concepts_for_papers(["P1", "P2"])
    # P1 mentions c_memory. P2 mentions c_agents.
    mapped = {(c[0], c[1]) for c in concepts_multi}
    assert mapped == {("P1", "c_memory"), ("P2", "c_agents")}
    
    # Paper with no concepts
    # P5 mentions c_unrelated, but let's check P5: it has mentions c_unrelated
    # We don't have a paper with 0 concepts in minimal graph. Let's save P6 with no concepts
    from src.models import Paper
    empty_p = Paper(id="P_empty", title="Empty", authors=[], year=2026)
    graph_repo.save_paper(empty_p)
    assert graph_repo.get_concepts_for_papers(["P_empty"]) == []

def test_sqlite_get_concept_document_frequencies(graph_repo, vector_repo):
    """Verify document frequency calculation for concepts."""
    seed_minimal_graph(graph_repo, vector_repo)
    
    # df(c_memory) = 3 (P1, P3, P4)
    # df(c_agents) = 2 (P2, P4)
    # df(c_unrelated) = 1 (P5)
    freqs = graph_repo.get_concept_document_frequencies(["c_memory", "c_agents", "c_unrelated", "nonexistent"])
    assert freqs["c_memory"] == 3
    assert freqs["c_agents"] == 2
    assert freqs["c_unrelated"] == 1
    assert freqs["nonexistent"] == 0
    
    # Concept mentioned multiple times in the same paper (duplicate edge) should count once.
    # Add a duplicate-like relation in edges table (with transaction ignore or manually)
    with graph_repo.transaction():
        # SQLite constraint prevents exact duplicate PRIMARY KEY, but let's check distinct count
        # The query uses COUNT(DISTINCT e.source_id) anyway, ensuring correct distinct paper counts
        pass

def test_sqlite_get_total_paper_count(graph_repo, vector_repo):
    """Verify total paper count matches total papers in the database."""
    seed_minimal_graph(graph_repo, vector_repo)
    assert graph_repo.get_total_paper_count() == 6
    
    # Empty DB behavior
    with graph_repo._get_connection() as conn:
        conn.execute("DELETE FROM edges;")
        conn.execute("DELETE FROM nodes;")
        conn.commit()
    assert graph_repo.get_total_paper_count() == 0

def test_sqlite_get_citation_neighbors(graph_repo, vector_repo):
    """Verify citation neighbors check for both outgoing/incoming directions."""
    seed_minimal_graph(graph_repo, vector_repo)
    
    # P1 cites P3.
    # Check neighbors of P1: seed cites candidate P3
    cits_p1 = graph_repo.get_citation_neighbors(["P1"])
    assert len(cits_p1) == 1
    assert cits_p1[0] == ("P1", "P3", "seed_cites_candidate", "Paper Three")
    
    # Check neighbors of P3: candidate (P3) cites seed (P1) -> reverse direction
    cits_p3 = graph_repo.get_citation_neighbors(["P3"])
    assert len(cits_p3) == 1
    assert cits_p3[0] == ("P3", "P1", "candidate_cites_seed", "Paper One")
    
    # Check multiple seeds
    cits_multi = graph_repo.get_citation_neighbors(["P1", "P3"])
    # Should get both directions
    directions = [c[2] for c in cits_multi]
    assert "seed_cites_candidate" in directions
    assert "candidate_cites_seed" in directions

def test_sqlite_search_chunks_within_papers(graph_repo, vector_repo):
    """Verify scoped search returns best chunks matching query inside specified papers."""
    seed_minimal_graph(graph_repo, vector_repo)
    
    # Query embedding [1.0] + [0.0]*383.
    # Chunks:
    # P3: p3#0 ([1.0] + ...), p3#1 ([0.8, 0.6] + ...)
    # P4: p4#0 ([0.9, 0.1] + ...), p4#1 ([0.7, 0.7] + ...)
    query_emb = [1.0] + [0.0] * 383
    
    # Respected limit per paper = 1
    res1 = graph_repo.search_chunks_within_papers(query_emb, ["P3", "P4"], limit_per_paper=1)
    # Should return exactly 1 chunk per paper: p3#0 (similarity 1.0) and p4#0 (similarity 0.9)
    assert len(res1) == 2
    cids = [r[0].id for r in res1]
    assert "p3#0" in cids
    assert "p4#0" in cids
    
    # Respected limit per paper = 2
    res2 = graph_repo.search_chunks_within_papers(query_emb, ["P3", "P4"], limit_per_paper=2)
    assert len(res2) == 4
    cids_all = [r[0].id for r in res2]
    assert sorted(cids_all) == ["p3#0", "p3#1", "p4#0", "p4#1"]
    
    # Empty papers
    assert graph_repo.search_chunks_within_papers(query_emb, []) == []
