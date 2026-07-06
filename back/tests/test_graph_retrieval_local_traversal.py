import pytest
from unittest.mock import MagicMock
from src.models import Paper, Chunk, Edge, Concept
from tests.factories import create_paper, create_chunk
from src.services.rag_service import RAGService
from src.config import config

@pytest.fixture
def rag_service(graph_repo, vector_repo, mock_embedding_engine, mock_llm_engine):
    config.data["graph_retrieval"] = {
        "enabled": True,
        "allowed_retrieval_edge_types": ["CITES", "CITED_BY", "MENTIONS_CONCEPT", "RELATED_TO", "HAS_TAG"],
        "chunks_per_graph_paper": 2
    }
    config.data["rag_components"] = {
        "graph_neighbors_in_rrf": True,
        "graph_neighbors_order": 2,
        "reranker": False
    }
    return RAGService(
        graph_repo=graph_repo,
        vector_repo=vector_repo,
        embedding_engine=mock_embedding_engine,
        llm_engine=mock_llm_engine,
        warmup=False
    )

def test_external_paper_is_never_a_candidate(rag_service, graph_repo, vector_repo):
    # Create local paper A with chunks
    paper_a = create_paper(id="paper_a", title="Local Paper A")
    graph_repo.save_paper(paper_a)
    chunk_a = create_chunk(id="chunk_a", paper_id="paper_a", text_content="Content A", embedding=[0.1]*384)
    vector_repo.save_chunks([chunk_a])

    # Create external paper X with no chunks
    paper_x = create_paper(id="paper_x", title="External Paper X")
    graph_repo.save_paper(paper_x)

    # Add edge A cites X
    graph_repo.add_edge("paper_a", "paper_x", "CITES", {})

    # Mock search_similar_chunks to return chunk_a
    orig_search = vector_repo.search_similar_chunks
    vector_repo.search_similar_chunks = lambda *a, **k: [(chunk_a, 0.9)]

    # Run graph retrieval
    results = rag_service.retrieve_relevant_chunks(query="test", limit=5)
    
    # Restore
    vector_repo.search_similar_chunks = orig_search

    # Assert X is not returned
    chunk_ids = [c[0].id for c in results]
    assert "chunk_x" not in chunk_ids
    assert not any(c[0].paper_id == "paper_x" for c in results)


def test_external_paper_can_bridge_to_local_paper(rag_service, graph_repo, vector_repo):
    # Create local paper A with chunks
    paper_a = create_paper(id="paper_a", title="Local Paper A")
    graph_repo.save_paper(paper_a)
    chunk_a = create_chunk(id="chunk_a", paper_id="paper_a", text_content="Content A", embedding=[0.1]*384)
    vector_repo.save_chunks([chunk_a])

    # Create external paper X with no chunks
    paper_x = create_paper(id="paper_x", title="External Paper X")
    graph_repo.save_paper(paper_x)

    # Create local paper B with chunks
    paper_b = create_paper(id="paper_b", title="Local Paper B")
    graph_repo.save_paper(paper_b)
    chunk_b = create_chunk(id="chunk_b", paper_id="paper_b", text_content="Content B", embedding=[0.1]*384)
    vector_repo.save_chunks([chunk_b])

    # Add A cites X and B cites X
    graph_repo.add_edge("paper_a", "paper_x", "CITES", {})
    graph_repo.add_edge("paper_b", "paper_x", "CITES", {})

    # Mock search_similar_chunks to return chunk_a
    orig_search = vector_repo.search_similar_chunks
    vector_repo.search_similar_chunks = lambda *a, **k: [(chunk_a, 0.9)]

    # Run graph retrieval
    results = rag_service.retrieve_relevant_chunks(query="test", limit=5)

    # Restore
    vector_repo.search_similar_chunks = orig_search

    # Assert B is returned, X is not
    paper_ids = [c[0].paper_id for c in results]
    assert "paper_b" in paper_ids
    assert "paper_x" not in paper_ids


def test_placeholder_with_chunks_count_zero_is_rejected(rag_service, graph_repo, vector_repo):
    # Create local paper A with chunks
    paper_a = create_paper(id="paper_a", title="Local Paper A")
    graph_repo.save_paper(paper_a)
    chunk_a = create_chunk(id="chunk_a", paper_id="paper_a", text_content="Content A", embedding=[0.1]*384)
    vector_repo.save_chunks([chunk_a])

    # Create placeholder paper P with chunks_count = 0
    paper_p = create_paper(id="paper_p", title="Placeholder P", properties={"is_placeholder": 1})
    graph_repo.save_paper(paper_p)

    # Add P as neighbor of A
    graph_repo.add_edge("paper_a", "paper_p", "CITES", {})

    # Mock search_similar_chunks to return chunk_a
    orig_search = vector_repo.search_similar_chunks
    vector_repo.search_similar_chunks = lambda *a, **k: [(chunk_a, 0.9)]

    # Run retrieval
    results = rag_service.retrieve_relevant_chunks(query="test", limit=5)
    
    # Restore
    vector_repo.search_similar_chunks = orig_search

    # Assert P is not in candidates
    paper_ids = [c[0].paper_id for c in results]
    assert "paper_p" not in paper_ids


def test_intra_paper_expansion_works_without_query_category(rag_service, graph_repo, vector_repo):
    # Create local paper A with chunks
    paper_a = create_paper(id="paper_a", title="Local Paper A")
    graph_repo.save_paper(paper_a)
    # Save multiple chunks for paper A
    chunk_a1 = create_chunk(id="paper_a#0", paper_id="paper_a", text_content="Content A1", embedding=[0.1]*384)
    chunk_a2 = create_chunk(id="paper_a#1", paper_id="paper_a", text_content="Content A2", embedding=[0.1]*384)
    vector_repo.save_chunks([chunk_a1, chunk_a2])

    # Mock search_similar_chunks to return chunk_a1 only
    orig_search = vector_repo.search_similar_chunks
    vector_repo.search_similar_chunks = lambda *a, **k: [(chunk_a1, 0.9)]

    # Run retrieval
    results = rag_service.retrieve_relevant_chunks(query="test", limit=5)

    # Restore search
    vector_repo.search_similar_chunks = orig_search

    # Assert same-paper graph chunks are returned (both chunk_a1 and chunk_a2)
    chunk_ids = [c[0].id for c in results]
    assert "paper_a#0" in chunk_ids
    assert "paper_a#1" in chunk_ids


def test_bridge_to_local_works_without_query_category(rag_service, graph_repo, vector_repo):
    # Create local paper A with chunks
    paper_a = create_paper(id="paper_a", title="Local Paper A")
    graph_repo.save_paper(paper_a)
    chunk_a = create_chunk(id="chunk_a", paper_id="paper_a", text_content="Content A", embedding=[0.1]*384)
    vector_repo.save_chunks([chunk_a])

    # Create external paper X with no chunks
    paper_x = create_paper(id="paper_x", title="External Paper X")
    graph_repo.save_paper(paper_x)

    # Create local paper B with chunks
    paper_b = create_paper(id="paper_b", title="Local Paper B")
    graph_repo.save_paper(paper_b)
    chunk_b = create_chunk(id="chunk_b", paper_id="paper_b", text_content="Content B", embedding=[0.1]*384)
    vector_repo.save_chunks([chunk_b])

    # Add A cites X and B cites X
    graph_repo.add_edge("paper_a", "paper_x", "CITES", {})
    graph_repo.add_edge("paper_b", "paper_x", "CITES", {})

    # Mock search_similar_chunks to return chunk_a
    orig_search = vector_repo.search_similar_chunks
    vector_repo.search_similar_chunks = lambda *a, **k: [(chunk_a, 0.9)]

    # Run retrieval
    results = rag_service.retrieve_relevant_chunks(query="test", limit=5)

    # Restore search
    vector_repo.search_similar_chunks = orig_search

    # Assert B is eligible and returned
    paper_ids = [c[0].paper_id for c in results]
    assert "paper_b" in paper_ids


def test_no_graph_candidate_without_chunks_reaches_reranker(rag_service, graph_repo, vector_repo):
    # Create local paper A with chunks
    paper_a = create_paper(id="paper_a", title="Local Paper A")
    graph_repo.save_paper(paper_a)
    chunk_a = create_chunk(id="chunk_a", paper_id="paper_a", text_content="Content A", embedding=[0.1]*384)
    vector_repo.save_chunks([chunk_a])

    # Create external paper X with no chunks
    paper_x = create_paper(id="paper_x", title="External Paper X")
    graph_repo.save_paper(paper_x)

    # Add A cites X
    graph_repo.add_edge("paper_a", "paper_x", "CITES", {})

    # Mock search_similar_chunks to return chunk_a
    orig_search = vector_repo.search_similar_chunks
    vector_repo.search_similar_chunks = lambda *a, **k: [(chunk_a, 0.9)]

    # Run retrieval
    results = rag_service.retrieve_relevant_chunks(query="test", limit=5)

    # Restore search
    vector_repo.search_similar_chunks = orig_search

    # Assert all candidates returned have chunks and no paper_x is returned
    for chunk, score in results:
        cnt = graph_repo.chunks_count(chunk.paper_id)
        assert cnt > 0
        assert chunk.paper_id != "paper_x"


def test_empty_concepts_do_not_break_graph_traversal(rag_service, graph_repo, vector_repo):
    # Create local paper A with chunks
    paper_a = create_paper(id="paper_a", title="Local Paper A")
    graph_repo.save_paper(paper_a)
    chunk_a1 = create_chunk(id="paper_a#0", paper_id="paper_a", text_content="Content A1", embedding=[0.1]*384)
    chunk_a2 = create_chunk(id="paper_a#1", paper_id="paper_a", text_content="Content A2", embedding=[0.1]*384)
    vector_repo.save_chunks([chunk_a1, chunk_a2])

    # Mock search_similar_chunks to return chunk_a1
    orig_search = vector_repo.search_similar_chunks
    vector_repo.search_similar_chunks = lambda *a, **k: [(chunk_a1, 0.9)]

    # Run query with no concepts (mocking extraction to return empty)
    orig_extract = rag_service._extract_query_concepts
    rag_service._extract_query_concepts = lambda q: []

    results = rag_service.retrieve_relevant_chunks(query="test", limit=5)

    # Restore mocks
    vector_repo.search_similar_chunks = orig_search
    rag_service._extract_query_concepts = orig_extract

    # Assert same-paper graph chunks are still returned (Layer 1 still works)
    chunk_ids = [c[0].id for c in results]
    assert "paper_a#0" in chunk_ids
    assert "paper_a#1" in chunk_ids


def test_technical_single_token_concepts_not_dropped(rag_service, graph_repo):
    # Add concepts to DB
    concepts = [
        ("decimation", "decimation"),
        ("quantization", "quantization"),
        ("latency", "latency"),
        ("ablation", "ablation"),
        ("db2", "db2"),
        ("db3", "db3")
    ]
    for cid, name in concepts:
        graph_repo.save_concept(Concept(id=cid, name=name))

    # Test classification function directly
    strong, dropped = rag_service._classify_query_concepts(
        "test query", 
        ["decimation", "quantization", "latency", "ablation", "db2", "db3"]
    )
    
    # Assert none of them are dropped
    assert "decimation" in strong
    assert "quantization" in strong
    assert "latency" in strong
    assert "ablation" in strong
    assert "db2" in strong
    assert "db3" in strong
