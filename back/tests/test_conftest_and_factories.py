import os
from tests.factories import create_paper, create_author, create_chunk

def test_temp_db_and_teardown(temp_db):
    # Ensure database path is generated
    assert temp_db.endswith(".db")
    
    # Touch database and verify it gets created
    with open(temp_db, "w") as f:
        f.write("dummy db contents")
    assert os.path.exists(temp_db)
    
    # Touch corresponding .usearch file and verify it gets created
    usearch_path = temp_db.replace(".db", ".usearch")
    with open(usearch_path, "w") as f:
        f.write("dummy usearch contents")
    assert os.path.exists(usearch_path)

def test_repositories_with_factories(graph_repo, vector_repo):
    # 1. Test create_paper and graph_repo
    paper = create_paper(id="factory_paper", title="Factory Paper Title", authors=["Author One", "Author Two"])
    graph_repo.save_paper(paper)
    
    retrieved_paper = graph_repo.get_paper("factory_paper")
    assert retrieved_paper is not None
    assert retrieved_paper.title == "Factory Paper Title"
    assert retrieved_paper.authors == ["Author One", "Author Two"]
    
    # 2. Test create_author and graph_repo
    author = create_author(name="Author One")
    graph_repo.save_author(author)
    graph_repo.add_edge(author.id, paper.id, "AUTHORED")
    
    neighbors = graph_repo.get_neighbors(paper.id, max_depth=1)
    assert len(neighbors) >= 1
    edge_types = [n[2] for n in neighbors]
    assert "AUTHORED" in edge_types
    
    # 3. Test create_chunk and vector_repo
    chunk = create_chunk(id="factory_chunk_id", paper_id=paper.id, text_content="Interesting text for vector similarity search", embedding=[0.2] * 384)
    vector_repo.save_chunks([chunk])
    
    results = vector_repo.search_similar_chunks([0.2] * 384, limit=1)
    assert len(results) == 1
    best_chunk, score = results[0]
    assert best_chunk.id == "factory_chunk_id"
    assert "Interesting text" in best_chunk.text_content

def test_indexer_fixture(indexer):
    assert indexer.graph_repo is not None
    assert indexer.vector_repo is not None
    assert indexer.emb_engine is not None
    assert indexer.llm_engine is not None
    
    # Verify mock embedding engine works
    embs = indexer.emb_engine.get_embeddings(["test text"])
    assert len(embs) == 1
    assert len(embs[0]) == 384
