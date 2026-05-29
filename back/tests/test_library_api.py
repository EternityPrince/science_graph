import pytest
from fastapi.testclient import TestClient
from src.web_app import app, get_graph_repo
from src.models import Concept
from tests.factories import create_paper, create_author

@pytest.fixture
def client(graph_repo):
    # Override dependency to use our test repository
    app.dependency_overrides[get_graph_repo] = lambda: graph_repo
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()

def test_get_documents_default_list(client, graph_repo):
    p1 = create_paper(id="paper_1", title="Deep Learning Basics", year=2024)
    p2 = create_paper(id="paper_2", title="Attention Is All You Need", year=2017)
    p3 = create_paper(id="paper_3", title="Introduction to AI", year=2026)

    graph_repo.save_paper(p1)
    graph_repo.save_paper(p2)
    graph_repo.save_paper(p3)

    response = client.get("/api/documents?page=1&limit=10")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 3
    assert len(data["results"]) == 3
    
    titles = [p["title"] for p in data["results"]]
    assert "Deep Learning Basics" in titles
    assert "Attention Is All You Need" in titles
    assert "Introduction to AI" in titles

def test_get_documents_filter_by_source_type(client, graph_repo):
    p1 = create_paper(id="p1", title="Paper Title", properties={"source_type": "paper"})
    p2 = create_paper(id="p2", title="Video Title", properties={"source_type": "video"})
    p3 = create_paper(id="p3", title="Note Title", properties={"source_type": "note"})

    graph_repo.save_paper(p1)
    graph_repo.save_paper(p2)
    graph_repo.save_paper(p3)

    r = client.get("/api/documents?source_type=paper")
    assert r.status_code == 200
    d = r.json()
    assert d["total"] == 1
    assert d["results"][0]["id"] == "p1"

    r = client.get("/api/documents?source_type=video&source_type=note")
    assert r.status_code == 200
    d = r.json()
    assert d["total"] == 2
    ids = [item["id"] for item in d["results"]]
    assert "p2" in ids
    assert "p3" in ids

def test_get_documents_filter_by_date_range(client, graph_repo):
    p1 = create_paper(id="p1", title="P1")
    p1.created_at = "2026-05-10T10:00:00Z"
    p2 = create_paper(id="p2", title="P2")
    p2.created_at = "2026-05-15T12:00:00Z"
    p3 = create_paper(id="p3", title="P3")
    p3.created_at = "2026-05-20T14:00:00Z"

    graph_repo.save_paper(p1)
    graph_repo.save_paper(p2)
    graph_repo.save_paper(p3)

    # Filter by from_date only
    r = client.get("/api/documents?from_date=2026-05-15")
    assert r.json()["total"] == 2
    ids = [item["id"] for item in r.json()["results"]]
    assert "p2" in ids
    assert "p3" in ids

    # Filter by to_date only
    r = client.get("/api/documents?to_date=2026-05-15")
    assert r.json()["total"] == 2
    ids = [item["id"] for item in r.json()["results"]]
    assert "p1" in ids
    assert "p2" in ids

    # Filter by date range
    r = client.get("/api/documents?from_date=2026-05-11&to_date=2026-05-16")
    assert r.json()["total"] == 1
    assert r.json()["results"][0]["id"] == "p2"

def test_get_documents_filter_by_author(client, graph_repo):
    p1 = create_paper(id="p1", title="Paper 1")
    p2 = create_paper(id="p2", title="Paper 2")

    graph_repo.save_paper(p1)
    graph_repo.save_paper(p2)

    a1 = create_author(id="author_1", name="Yann LeCun")
    a2 = create_author(id="author_2", name="Yoshua Bengio")
    graph_repo.save_author(a1)
    graph_repo.save_author(a2)

    graph_repo.add_edge("author_1", "p1", "AUTHORED")
    graph_repo.add_edge("author_2", "p2", "AUTHORED")

    # Filter by author name
    r = client.get("/api/documents?author=Yann LeCun")
    assert r.json()["total"] == 1
    assert r.json()["results"][0]["id"] == "p1"

    # Filter by author ID
    r = client.get("/api/documents?author=author_2")
    assert r.json()["total"] == 1
    assert r.json()["results"][0]["id"] == "p2"

def test_get_documents_filter_by_concept(client, graph_repo):
    p1 = create_paper(id="p1", title="Paper 1")
    p2 = create_paper(id="p2", title="Paper 2")

    graph_repo.save_paper(p1)
    graph_repo.save_paper(p2)

    c1 = Concept(id="deep_learning", name="Deep Learning")
    c2 = Concept(id="transformer", name="Transformer")
    graph_repo.save_concept(c1)
    graph_repo.save_concept(c2)

    graph_repo.add_edge("p1", "deep_learning", "MENTIONS_CONCEPT")
    graph_repo.add_edge("p2", "transformer", "MENTIONS_CONCEPT")

    # Filter by concept name
    r = client.get("/api/documents?concept=Deep Learning")
    assert r.json()["total"] == 1
    assert r.json()["results"][0]["id"] == "p1"

    # Filter by concept ID
    r = client.get("/api/documents?concept=transformer")
    assert r.json()["total"] == 1
    assert r.json()["results"][0]["id"] == "p2"

def test_get_documents_filter_by_tag(client, graph_repo):
    p1 = create_paper(id="p1", title="Paper 1")
    p2 = create_paper(id="p2", title="Paper 2")

    graph_repo.save_paper(p1)
    graph_repo.save_paper(p2)

    t1 = Concept(id="tag_nlp", name="NLP", properties={"is_tag": True})
    t2 = Concept(id="tag_cv", name="Computer Vision", properties={"is_tag": True})
    graph_repo.save_concept(t1)
    graph_repo.save_concept(t2)

    graph_repo.add_edge("p1", "tag_nlp", "MENTIONS_CONCEPT")
    graph_repo.add_edge("p2", "tag_cv", "MENTIONS_CONCEPT")

    # Filter by tag name
    r = client.get("/api/documents?tag=NLP")
    assert r.json()["total"] == 1
    assert r.json()["results"][0]["id"] == "p1"

    # Filter by tag ID
    r = client.get("/api/documents?tag=tag_cv")
    assert r.json()["total"] == 1
    assert r.json()["results"][0]["id"] == "p2"

def test_get_documents_search_query_q(client, graph_repo, vector_repo):
    from tests.factories import create_chunk
    # Seed papers
    p1 = create_paper(id="p1", title="Quantum Mechanics")
    p2 = create_paper(id="p2", title="Relativity Theory")
    graph_repo.save_paper(p1)
    graph_repo.save_paper(p2)

    # Save a chunk for relativity
    chunk = create_chunk(id="c1", paper_id="p2", text_content="einstein space time gravity", embedding=[0.1] * 384)
    vector_repo.save_chunks([chunk])

    # Search title property
    r = client.get("/api/documents?q=Mechanics")
    assert r.json()["total"] == 1
    assert r.json()["results"][0]["id"] == "p1"

    # Search text content in chunks
    r = client.get("/api/documents?q=gravity")
    assert r.json()["total"] == 1
    assert r.json()["results"][0]["id"] == "p2"

def test_get_documents_only_indexed(client, graph_repo):
    p1 = create_paper(id="p1", title="Indexed Paper", properties={"is_placeholder": False})
    p2 = create_paper(id="p2", title="Placeholder Paper", properties={"is_placeholder": True})

    graph_repo.save_paper(p1)
    graph_repo.save_paper(p2)

    # default (only_indexed=False) -> both
    r = client.get("/api/documents")
    assert r.json()["total"] == 2

    # only_indexed=true -> only p1
    r = client.get("/api/documents?only_indexed=true")
    assert r.json()["total"] == 1
    assert r.json()["results"][0]["id"] == "p1"
