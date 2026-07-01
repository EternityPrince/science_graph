import json
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from src.web_app import (
    app,
    get_graph_repo,
    get_vector_repo,
    get_embedding_engine,
    get_llm_engine,
    get_default_llm_engine,
    get_rag_service,
    get_note_service
)

class TestWebAppEndpoints:
    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        # Create mocks
        self.mock_graph_repo = MagicMock()
        self.mock_vector_repo = MagicMock()
        self.mock_embedding_engine = MagicMock()
        self.mock_llm_engine = MagicMock()
        self.mock_rag_service = MagicMock()
        self.mock_note_service = MagicMock()

        # Apply dependency overrides
        app.dependency_overrides[get_graph_repo] = lambda: self.mock_graph_repo
        app.dependency_overrides[get_vector_repo] = lambda: self.mock_vector_repo
        app.dependency_overrides[get_embedding_engine] = lambda: self.mock_embedding_engine
        app.dependency_overrides[get_llm_engine] = lambda: self.mock_llm_engine
        app.dependency_overrides[get_default_llm_engine] = lambda: self.mock_llm_engine
        app.dependency_overrides[get_rag_service] = lambda: self.mock_rag_service
        app.dependency_overrides[get_note_service] = lambda: self.mock_note_service

        self.client = TestClient(app)
        
        yield
        
        # Clean up overrides
        app.dependency_overrides.clear()

    def test_get_stats(self):
        self.mock_graph_repo.get_stats.return_value = {
            "papers": 42,
            "authors": 10,
            "concepts": 15,
            "edges": 120
        }
        
        mock_storage_stats = {
            "total_size": 1000,
            "storage_dir": "/tmp",
            "extensions": [],
            "sources": []
        }
        with patch("src.config.config.get_storage_stats", return_value=mock_storage_stats):
            response = self.client.get("/api/stats")
            assert response.status_code == 200
            data = response.json()
            assert data["papers"] == 42
            assert data["storage"]["total_size"] == 1000

    def test_get_models(self):
        response = self.client.get("/api/models")
        assert response.status_code == 200
        data = response.json()
        assert "llm_local" in data
        assert "embedding" in data

    def test_get_graph(self):
        self.mock_graph_repo.get_all_nodes.return_value = [
            ("paper_1", "Paper", '{"title": "Paper One", "source_type": "paper"}'),
            ("author_1", "Author", '{"name": "Alice Smith"}')
        ]
        self.mock_graph_repo.get_all_edges.return_value = [
            ("author_1", "paper_1", "AUTHORED", "{}")
        ]

        response = self.client.get("/api/graph?show_references=true")
        assert response.status_code == 200
        data = response.json()
        assert len(data["nodes"]) == 2
        assert len(data["edges"]) == 1
        assert data["nodes"][0]["id"] == "paper_1"
        assert data["nodes"][0]["label"] == "Paper One"
        assert data["nodes"][1]["label"] == "Alice Smith"

    def test_open_file_not_found(self):
        response = self.client.post("/api/open-file", json={"file_path": "/nonexistent/file.pdf"})
        assert response.status_code == 404
        assert "File not found" in response.json()["detail"]

    @patch("os.path.exists", return_value=True)
    @patch("subprocess.run")
    @patch("sys.platform", "darwin")
    def test_open_file_success_macos(self, mock_run, mock_exists):
        response = self.client.post("/api/open-file", json={"file_path": "/valid/file.pdf"})
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        mock_run.assert_called_once_with(["open", "/valid/file.pdf"], check=True)

    def test_get_notes(self):
        self.mock_note_service.get_notes.return_value = [
            {"id": "note_1", "title": "My Note", "summary": "Note summary", "abstract": "abc", "created_at": "2026-05-25", "properties": {}}
        ]
        response = self.client.get("/api/notes")
        assert response.status_code == 200
        assert len(response.json()) == 1
        assert response.json()[0]["title"] == "My Note"

    @patch("src.web_app.get_rag_service")
    def test_query_rag_stream(self, mock_get_rag_service):
        async def dummy_stream(question, limit, **kwargs):
            yield {"token": "Hello"}
            yield {"token": " World"}
            yield {"status": "done"}

        mock_rag = MagicMock()
        mock_rag.generate_stream = dummy_stream
        mock_get_rag_service.return_value = mock_rag

        response = self.client.post("/api/query", json={"question": "Test question?", "limit": 5, "cloud": False})
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        
        # Read the stream lines
        events = [line for line in response.iter_lines() if line]
        assert len(events) == 3
        # Fast-API EventSourceResponse prefixes with "data: "
        assert json.loads(events[0].replace("data: ", "")) == {"token": "Hello"}
        assert json.loads(events[1].replace("data: ", "")) == {"token": " World"}
        assert json.loads(events[2].replace("data: ", "")) == {"status": "done"}

    @patch("src.indexer.Indexer.index_url")
    def test_index_url_route_multiple_urls(self, mock_index_url):
        mock_index_url.side_effect = ["paper_1", "paper_2"]
        
        from src.models import Paper
        paper1 = Paper(id="paper_1", title="Title One")
        paper2 = Paper(id="paper_2", title="Title Two")
        self.mock_graph_repo.get_paper.side_effect = lambda pid: {"paper_1": paper1, "paper_2": paper2}.get(pid)
        
        response = self.client.post("/api/index-url", json={"url": "https://a.com, https://b.com"})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["id"] == "paper_1, paper_2"
        assert data["title"] == "Title One, Title Two"
        
        assert mock_index_url.call_count == 2
        calls = [args[0][0] for args in mock_index_url.call_args_list]
        assert calls == ["https://a.com", "https://b.com"]

    def test_get_paper_text(self):
        from src.models import Chunk
        self.mock_vector_repo.get_chunks_for_paper.return_value = [
            Chunk(id="c2", paper_id="p1", text_content="chunk 2 content", page_number=2, embedding=[]),
            Chunk(id="c1", paper_id="p1", text_content="chunk 1 content", page_number=1, embedding=[]),
        ]
        
        response = self.client.get("/api/paper-text?paper_id=p1")
        assert response.status_code == 200
        data = response.json()
        assert data["paper_id"] == "p1"
        assert len(data["chunks"]) == 2
        # Check sorting: page 1 first
        assert data["chunks"][0]["id"] == "c1"
        assert data["chunks"][0]["text_content"] == "chunk 1 content"
        assert data["chunks"][1]["id"] == "c2"
        assert data["chunks"][1]["text_content"] == "chunk 2 content"

    def test_get_paper_pdf(self):
        import tempfile
        from src.models import Paper
        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
            paper = Paper(id="p1", title="Title One", file_path=tmp.name)
            self.mock_graph_repo.get_paper.return_value = paper
            
            response = self.client.get("/api/paper-pdf/p1")
            assert response.status_code == 200
            assert response.headers["content-type"] == "application/pdf"
            assert "inline" in response.headers["content-disposition"]

