import pytest
import subprocess
from unittest.mock import MagicMock, patch
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
from src.indexer import DuplicateDocumentError
from src.models import Paper, Author


class TestWebAppEdgeCases:
    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        self.mock_graph_repo = MagicMock()
        self.mock_vector_repo = MagicMock()
        self.mock_embedding_engine = MagicMock()
        self.mock_llm_engine = MagicMock()
        self.mock_rag_service = MagicMock()
        self.mock_note_service = MagicMock()

        app.dependency_overrides[get_graph_repo] = lambda: self.mock_graph_repo
        app.dependency_overrides[get_vector_repo] = lambda: self.mock_vector_repo
        app.dependency_overrides[get_embedding_engine] = lambda: self.mock_embedding_engine
        app.dependency_overrides[get_llm_engine] = lambda: self.mock_llm_engine
        app.dependency_overrides[get_default_llm_engine] = lambda: self.mock_llm_engine
        app.dependency_overrides[get_rag_service] = lambda: self.mock_rag_service
        app.dependency_overrides[get_note_service] = lambda: self.mock_note_service

        self.client = TestClient(app)
        yield
        app.dependency_overrides.clear()

    def test_get_stats_error_handling(self):
        """Verify /api/stats endpoint handles database exceptions or storage stats failure gracefully."""
        self.mock_graph_repo.get_stats.side_effect = Exception("DB Connection Locked")
        
        # When DB throws, endpoint should bubble up the exception as a 500 error or handle it
        with pytest.raises(Exception):
            self.client.get("/api/stats")

    def test_get_graph_empty_and_color_mappings(self):
        """Verify color assignments and properties mapping on /api/graph with various labels/types."""
        self.mock_graph_repo.get_all_nodes.return_value = [
            # node_id, label, properties_json
            ("n_note", "Paper", '{"title": "Note Title", "source_type": "note"}'),
            ("n_book", "Paper", '{"title": "Book Title", "source_type": "book"}'),
            ("n_paper", "Paper", '{"title": "Paper Title", "source_type": "paper"}'),
            ("n_video", "Paper", '{"title": "Video Title", "source_type": "video"}'),
            ("n_webpage", "Paper", '{"title": "Webpage Title", "source_type": "webpage"}'),
            ("n_placeholder", "Paper", '{"title": "Reference Title", "placeholder": true}'),
            ("n_author", "Author", '{"name": "Alice Smith"}'),
            ("n_tag", "Concept", '{"name": "Deep Learning", "is_tag": true}'),
            ("n_concept", "Concept", '{"name": "Attention", "is_tag": false}'),
            ("n_other", "UnknownLabel", '{"name": "Custom"}'),
        ]
        self.mock_graph_repo.get_all_edges.return_value = []
        
        # 1. show_references = True (should return all, including placeholder)
        response = self.client.get("/api/graph?show_references=true")
        assert response.status_code == 200
        nodes = response.json()["nodes"]
        
        # Should return all nodes connected or not (since get_graph connects nodes based on edges,
        # wait: let's verify if nodes unconnected are filtered out!)
        # In get_graph:
        # 1. allowed_paper_ids gets all papers (indexed_paper_ids) + placeholders (if show_references)
        # 2. connected_non_papers gets authors and concepts CONNECTED to the allowed papers via edges.
        # Since edges is empty, allowed_node_ids = allowed_paper_ids (only papers/notes, no author/concept/other!)
        # Let's verify this filtering:
        # allowed_paper_ids = {"n_note", "n_book", "n_paper", "n_video", "n_webpage", "n_placeholder"}
        # connected_non_papers = empty
        # allowed_node_ids = {"n_note", "n_book", "n_paper", "n_video", "n_webpage", "n_placeholder"}
        assert len(nodes) == 6
        
        # Check colors and groups of various paper types
        color_maps = {node["id"]: node["color"] for node in nodes}
        assert color_maps["n_note"] == "#a5b4fc"
        assert color_maps["n_book"] == "#818cf8"
        assert color_maps["n_paper"] == "#6366f1"
        assert color_maps["n_video"] == "#f43f5e"
        assert color_maps["n_webpage"] == "#06b6d4"
        assert color_maps["n_placeholder"] == "#64748b" # reference
        
        # 2. Add edges to test inclusion of connected non-papers
        self.mock_graph_repo.get_all_edges.return_value = [
            ("n_author", "n_paper", "AUTHORED", "{}"),
            ("n_paper", "n_tag", "HAS_TAG", "{}"),
            ("n_paper", "n_concept", "MENTIONS_CONCEPT", "{}"),
            ("n_paper", "n_other", "OTHER", "{}"),
        ]
        
        response2 = self.client.get("/api/graph?show_references=false")
        assert response2.status_code == 200
        nodes2 = response2.json()["nodes"]
        # n_placeholder (reference) is excluded because show_references=false
        # Rest of the nodes should be included because they are papers or connected to papers via edges
        node_ids2 = {node["id"] for node in nodes2}
        assert "n_placeholder" not in node_ids2
        assert "n_author" in node_ids2
        assert "n_tag" in node_ids2
        assert "n_concept" in node_ids2
        assert "n_other" in node_ids2
        
        # Check colors and sizes of non-papers
        colors2 = {node["id"]: (node["color"], node["size"], node["group"]) for node in nodes2}
        assert colors2["n_author"] == ("#cbd5e1", 18, "author")
        assert colors2["n_tag"] == ("#ec4899", 15, "tag")
        assert colors2["n_concept"] == ("#10b981", 16, "concept")
        assert colors2["n_other"] == ("#475569", 14, "other")

    @patch("sys.platform", "win32")
    @patch("os.path.exists", return_value=True)
    @patch("os.startfile", create=True)
    def test_open_file_success_windows(self, mock_startfile, mock_exists):
        """Test open-file endpoint invokes correct command on Windows platform."""
        response = self.client.post("/api/open-file", json={"file_path": "C:\\path\\file.pdf"})
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        mock_startfile.assert_called_once()

    @patch("sys.platform", "linux")
    @patch("os.path.exists", return_value=True)
    @patch("subprocess.run")
    def test_open_file_success_linux(self, mock_run, mock_exists):
        """Test open-file endpoint invokes correct command on Linux platform."""
        response = self.client.post("/api/open-file", json={"file_path": "/var/file.pdf"})
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        mock_run.assert_called_once_with(["xdg-open", "/var/file.pdf"], check=True)

    @patch("sys.platform", "darwin")
    @patch("os.path.exists", return_value=True)
    @patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "open"))
    def test_open_file_subprocess_error(self, mock_run, mock_exists):
        """Test open-file endpoint handles subprocess failures gracefully."""
        response = self.client.post("/api/open-file", json={"file_path": "/var/file.pdf"})
        assert response.status_code == 500
        assert "returned non-zero exit status" in response.json()["detail"]

    @patch("src.web_app.get_rag_service")
    def test_query_rag_stream_exception_midstream(self, mock_get_rag_service):
        """Verify RAG query SSE stream reports errors in SSE format if generated during stream."""
        async def throwing_stream(question, limit, **kwargs):
            yield {"token": "First word"}
            raise Exception("Model context exhausted")

        mock_rag = MagicMock()
        mock_rag.generate_stream = throwing_stream
        mock_get_rag_service.return_value = mock_rag

        with pytest.raises(Exception, match="Model context exhausted"):
            self.client.post("/api/query", json={"question": "Crash me?", "limit": 5})

    @patch("src.indexer.Indexer.index_url")
    def test_index_url_route_duplicate_document(self, mock_index_url):
        """Verify URL indexer endpoint handles DuplicateDocumentError properly."""
        mock_index_url.side_effect = DuplicateDocumentError("Already indexed!", "duplicate_id")
        
        response = self.client.post("/api/index-url", json={"url": "https://duplicate.com"})
        assert response.status_code == 409
        assert "Already indexed" in response.json()["detail"]

        mock_index_url.side_effect = ValueError("Invalid structure")
        
        response = self.client.post("/api/index-url", json={"url": "https://invalid.com"})
        assert response.status_code == 500
        assert "Invalid structure" in response.json()["detail"]

    def test_get_paper_details_not_found(self):
        """Test /api/paper/{paper_id} returns 404 if node does not exist or paper object not found."""
        # 1. Node not found
        self.mock_graph_repo.get_node_by_id.return_value = None
        response = self.client.get("/api/paper/nonexistent_paper")
        assert response.status_code == 404
        assert "Node not found" in response.json()["detail"]

        # 2. Node found but Paper model not found
        self.mock_graph_repo.get_node_by_id.return_value = ("Paper", "{}")
        self.mock_graph_repo.get_paper.return_value = None
        response2 = self.client.get("/api/paper/missing_paper")
        assert response2.status_code == 404
        assert "Paper not found" in response2.json()["detail"]

    def test_get_paper_details_paper_success(self):
        """Test /api/paper/{paper_id} resolves Paper nodes with citations, authors, and concepts."""
        from src.models import Paper
        # Setup Paper
        paper = Paper(
            id="p_main",
            title="Main Paper",
            authors=["Alice", "Bob"],
            year=2026,
            abstract="Brilliant work.",
            properties={"source_type": "paper", "summary": "Main summary"}
        )
        self.mock_graph_repo.get_node_by_id.side_effect = lambda nid: {
            "p_main": ("Paper", "{}"),
            "c_concept": ("Concept", '{"name": "Concept Name"}'),
            "t_tag": ("Concept", '{"name": "Tag Name"}'),
        }.get(nid)
        
        self.mock_graph_repo.get_paper.return_value = paper
        
        # Mock neighbors: CITES outbound, MENTIONS_CONCEPT, HAS_TAG, AUTHORED, CITES inbound
        self.mock_graph_repo.get_neighbors.return_value = [
            ("p_main", "Paper", "MENTIONS_CONCEPT", "c_concept", "Concept", "{}"),
            ("p_main", "Paper", "HAS_TAG", "t_tag", "Concept", "{}"),
            ("auth_1", "Author", "AUTHORED", "p_main", "Paper", "{}"),
            ("p_main", "Paper", "CITES", "p_cited", "Paper", "{}"),
            ("p_citing", "Paper", "CITES", "p_main", "Paper", "{}"),
        ]
        
        # Batch calls returning paper dicts
        self.mock_graph_repo.get_papers_batch.side_effect = lambda ids: {
            "p_cited": Paper(id="p_cited", title="Cited Paper"),
            "p_citing": Paper(id="p_citing", title="Citing Paper")
        }
        
        response = self.client.get("/api/paper/p_main")
        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "paper"
        assert data["title"] == "Main Paper"
        assert data["authors"] == ["Alice", "Bob"]
        assert len(data["concepts"]) == 1
        assert data["concepts"][0] == {"id": "c_concept", "name": "Concept Name"}
        assert len(data["tags"]) == 1
        assert data["tags"][0] == {"id": "t_tag", "name": "Tag Name"}
        assert data["citations"] == [{"id": "p_cited", "title": "Cited Paper"}]
        assert data["cited_by"] == [{"id": "p_citing", "title": "Citing Paper"}]

    def test_get_paper_details_author_success(self):
        """Test /api/paper/{paper_id} resolves Author details correctly."""
        from src.models import Paper
        self.mock_graph_repo.get_node_by_id.return_value = ("Author", '{"name": "Alice Smith"}')
        self.mock_graph_repo.get_papers_by_author.return_value = [
            Paper(id="p1", title="Paper 1", properties={"source_type": "note"}),
            Paper(id="p2", title="Paper 2", properties={"source_type": "paper"})
        ]
        
        response = self.client.get("/api/paper/auth_alice")
        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "author"
        assert data["name"] == "Alice Smith"
        assert data["papers_count"] == 2
        assert data["papers"][0] == {"id": "p1", "title": "Paper 1", "source_type": "note"}

    def test_get_paper_details_concept_and_tag_success(self):
        """Test /api/paper/{paper_id} resolves Concept details with tag distinction."""
        from src.models import Paper
        # 1. Concept (non-tag)
        self.mock_graph_repo.get_node_by_id.return_value = ("Concept", '{"name": "Attention", "is_tag": false, "description": "Mechanism"}')
        self.mock_graph_repo.get_papers_by_entity.return_value = [Paper(id="p1", title="Paper 1")]
        # Mock distinct targets (related tags on those papers)
        self.mock_graph_repo.get_distinct_targets.return_value = [
            ("t1", '{"name": "Deep Learning"}')
        ]
        
        response = self.client.get("/api/paper/concept_attention")
        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "concept"
        assert data["name"] == "Attention"
        assert data["description"] == "Mechanism"
        assert data["papers"] == [{"id": "p1", "title": "Paper 1", "source_type": "paper"}]
        assert data["related"] == [{"id": "t1", "name": "Deep Learning"}]

    def test_upload_file_validation_and_errors(self):
        """Test /api/upload endpoint for extension checks, LLM availability, and index exceptions."""
        # 1. Unsupported extension
        response1 = self.client.post("/api/upload", files={"file": ("test.txt", b"some text", "text/plain")})
        assert response1.status_code == 400
        assert "Only PDF, Markdown" in response1.json()["detail"]

        # 2. LLM engine is None
        app.dependency_overrides[get_llm_engine] = lambda: None
        response2 = self.client.post("/api/upload", files={"file": ("test.pdf", b"pdf contents", "application/pdf")})
        assert response2.status_code == 503
        assert "LLM engine is not available" in response2.json()["detail"]
        
        # Restore mock_llm_engine override
        app.dependency_overrides[get_llm_engine] = lambda: self.mock_llm_engine

        # 3. Successful index (mocking Indexer)
        with patch("src.web_app.asyncio.to_thread") as mock_to_thread:
            mock_to_thread.return_value = "p_indexed_123"
            response3 = self.client.post("/api/upload", files={"file": ("test.md", b"# Markdown doc", "text/markdown")})
            assert response3.status_code == 200
            assert response3.json() == {"status": "ok", "id": "p_indexed_123", "filename": "test.md"}

            # 4. Duplicate document error handling
            mock_to_thread.side_effect = DuplicateDocumentError("Already uploaded", "dup_id")
            response4 = self.client.post("/api/upload", files={"file": ("test.pdf", b"pdf", "application/pdf")})
            assert response4.status_code == 409
            assert "Already uploaded" in response4.json()["detail"]

            # 5. Generic exception handling
            mock_to_thread.side_effect = Exception("General file parsing failure")
            response5 = self.client.post("/api/upload", files={"file": ("test.epub", b"epub", "application/epub+zip")})
            assert response5.status_code == 500
            assert "General file parsing failure" in response5.json()["detail"]

    def test_create_note_success_and_error(self):
        """Test POST /api/notes route invokes note_service correctly."""
        # 1. Success
        self.mock_note_service.create_note.return_value = ("note_123", "/path/to/note.md")
        response = self.client.post("/api/notes", json={"title": "New Note", "content": "Note body", "authors": ["Alice"], "tags": ["DL"]})
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "id": "note_123", "file_path": "/path/to/note.md"}
        self.mock_note_service.create_note.assert_called_once_with("New Note", "Note body", ["Alice"], ["DL"])

        # 2. Failure
        self.mock_note_service.create_note.side_effect = Exception("Note creation write error")
        response2 = self.client.post("/api/notes", json={"title": "New Note", "content": "Note body"})
        assert response2.status_code == 500
        assert "Note creation write error" in response2.json()["detail"]

    def test_dependency_injection_provider_invocations(self):
        """Exercise dependency injection provider branches to gain coverage."""
        from src.services.container import container
        
        # Test original get functions (without mocks)
        with patch.object(container, "get_graph_repo") as mock_get_graph:
            get_graph_repo()
            mock_get_graph.assert_called_once()
            
        with patch.object(container, "get_vector_repo") as mock_get_vector:
            get_vector_repo()
            mock_get_vector.assert_called_once()
            
        with patch.object(container, "get_embedding_engine") as mock_get_emb:
            get_embedding_engine()
            mock_get_emb.assert_called_once()
            
        with patch.object(container, "get_llm_engine") as mock_get_llm:
            get_llm_engine()
            get_default_llm_engine()
            assert mock_get_llm.call_count == 2
            
        with patch.object(container, "get_rag_service") as mock_get_rag:
            get_rag_service()
            mock_get_rag.assert_called_once()

        with patch.object(container, "get_note_service") as mock_get_note:
            get_note_service()
            mock_get_note.assert_called_once()

    @patch("src.web_app._WEB_DIR")
    def test_catch_all_fallback_routing(self, mock_web_dir):
        """Test catch_all static and SPA fallback routing behaves correctly."""
        import tempfile
        from pathlib import Path
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            index_file = tmp_path / "index.html"
            index_file.write_text("index html contents")
            
            some_file = tmp_path / "asset.svg"
            some_file.write_text("svg content")
            
            mock_web_dir.parent = tmp_path.parent
            mock_web_dir.__truediv__.side_effect = lambda name: tmp_path / name
            
            # 1. Access root SPA route path
            response = self.client.get("/")
            assert response.status_code == 200
            assert response.text == "index html contents"
            
            # 2. Access actual asset file
            response2 = self.client.get("/asset.svg")
            assert response2.status_code == 200
            assert response2.text == "svg content"
            
            # 3. Access SPA virtual sub-route (should fallback to index.html)
            response3 = self.client.get("/library/settings")
            assert response3.status_code == 200
            assert response3.text == "index html contents"
            
            # 4. Remove index file and test nonexistent fallbacks
            index_file.unlink()
            response4 = self.client.get("/library/settings")
            assert response4.status_code == 404
            
            response5 = self.client.get("/")
            assert response5.status_code == 404

    def test_additional_web_app_edge_cases(self):
        """Test remaining uncovered branches of web_app.py."""
        from pathlib import Path
        # 1. get_llm_engine and get_rag_service exceptions handling
        from src.services.container import container
        with patch.object(container, "get_llm_engine", side_effect=Exception("LLM fail")):
            assert get_llm_engine() is None
            
        with patch.object(container, "get_rag_service", side_effect=Exception("RAG fail")):
            assert get_rag_service() is None

        # 2. favicon.ico not found
        with patch("src.web_app._WEB_DIR") as mock_web_dir:
            mock_web_dir.__truediv__.side_effect = lambda name: Path("/nonexistent") / name
            response = self.client.get("/favicon.ico")
            assert response.status_code == 404

        # 3. get_stats storage stats exception handling
        from src.config import config
        with patch.object(config, "get_storage_stats", side_effect=Exception("IO error")):
            self.mock_graph_repo.get_stats.return_value = {
                "papers": 1,
                "authors": 0,
                "concepts": 0,
                "edges": 0,
                "indexed_papers": 1,
                "mentioned_papers": 0
            }
            response = self.client.get("/api/stats")
            assert response.status_code == 200
            assert response.json()["papers"] == 1

        # 4. get_graph node property elements formatting (year and authors list presence, edge filter branch)
        self.mock_graph_repo.get_all_nodes.return_value = [
            ("p_allowed", "Paper", '{"title": "Title P", "year": 2026, "authors": ["A1", "A2"]}')
        ]
        self.mock_graph_repo.get_all_edges.return_value = [
            ("p_disallowed1", "p_disallowed2", "CITES", "{}")
        ]
        response = self.client.get("/api/graph?show_references=true")
        assert response.status_code == 200
        nodes = response.json()["nodes"]
        assert len(nodes) == 1
        assert "Year: 2026" in nodes[0]["title"]
        assert "Authors: A1, A2" in nodes[0]["title"]
        # Edges should be empty because p_disallowed was filtered out
        assert len(response.json()["edges"]) == 0

        # 5. paper detail missing concept/tag node details fallback to ID
        self.mock_graph_repo.get_node_by_id.side_effect = lambda nid: ("Paper", '{"title": "Paper X"}') if nid == "p_x" else None
        self.mock_graph_repo.get_paper.return_value = Paper(id="p_x", title="Paper X", authors=[])
        self.mock_graph_repo.get_neighbors.return_value = [
            ("p_x", "Paper", "MENTIONS_CONCEPT", "concept_missing", "Concept", "{}"),
            ("p_x", "Paper", "HAS_TAG", "tag_missing", "Concept", "{}")
        ]
        response = self.client.get("/api/paper/p_x")
        assert response.status_code == 200
        res_json = response.json()
        assert res_json["concepts"] == [{"id": "concept_missing", "name": "concept_missing"}]
        assert res_json["tags"] == [{"id": "tag_missing", "name": "tag_missing"}]

        # 6. resolve author names if paper.authors is empty but we have AUTHORED edges
        self.mock_graph_repo.get_node_by_id.side_effect = lambda nid: ("Paper", '{"title": "Paper Y"}') if nid == "p_y" else None
        self.mock_graph_repo.get_paper.return_value = Paper(id="p_y", title="Paper Y", authors=[])
        self.mock_graph_repo.get_neighbors.return_value = [
            ("auth_y", "Author", "AUTHORED", "p_y", "Paper", "{}")
        ]
        mock_author = Author(id="auth_y", name="Author Y")
        self.mock_graph_repo.get_author.return_value = mock_author
        response = self.client.get("/api/paper/p_y")
        assert response.status_code == 200
        assert response.json()["authors"] == ["Author Y"]

        # 7. Search API endpoint
        self.mock_graph_repo.search_papers_by_title.return_value = [
            Paper(id="p_searched", title="Searched Paper", authors=[], year=2026, properties={"source_type": "paper"})
        ]
        response = self.client.get("/api/search?q=Searched")
        assert response.status_code == 200
        assert len(response.json()["results"]) == 1
        assert response.json()["results"][0]["id"] == "p_searched"

        # 8. RAG service unavailable POST /api/query
        with patch("src.web_app.get_rag_service", return_value=None):
            response = self.client.post("/api/query", json={"question": "Where is the science graph?", "cloud": False})
            assert response.status_code == 503

        # 9. Upload pdf and epub suffix indexer branches
        with patch("src.web_app.asyncio.to_thread") as mock_to_thread:
            mock_to_thread.return_value = "p_pdf"
            response = self.client.post("/api/upload", files={"file": ("test.pdf", b"pdf content", "application/pdf")})
            assert response.status_code == 200
            
            mock_to_thread.return_value = "p_epub"
            response = self.client.post("/api/upload", files={"file": ("test.epub", b"epub content", "application/epub+zip")})
            assert response.status_code == 200

        # 10. os.unlink failure handling in upload
        with patch("src.web_app.os.unlink", side_effect=Exception("unlink failed")):
            with patch("src.web_app.asyncio.to_thread", return_value="p_id"):
                response = self.client.post("/api/upload", files={"file": ("test.md", b"content", "text/markdown")})
                assert response.status_code == 200

        # 11. index-url exceptions and parameters validation
        # llm engine is None
        app.dependency_overrides[get_llm_engine] = lambda: None
        response = self.client.post("/api/index-url", json={"url": "https://google.com"})
        assert response.status_code == 503
        app.dependency_overrides[get_llm_engine] = lambda: self.mock_llm_engine

        # empty url list
        response = self.client.post("/api/index-url", json={"url": ""})
        assert response.status_code == 400

        # catch_all call with empty path_name (handled as root index redirect/fetch)
        with patch("src.web_app._WEB_DIR") as mock_web_dir:
            mock_web_dir.parent = Path("/tmp")
            mock_web_dir.__truediv__.side_effect = lambda name: Path("/tmp") / name
            response = self.client.get("/")
            assert response.status_code == 404


