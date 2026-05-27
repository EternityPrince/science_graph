import unittest
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner
from src.cli import app

runner = CliRunner()

class TestCLI(unittest.TestCase):
    @patch("src.services.indexing_orchestrator.run_batch_index")
    def test_index_url(self, mock_run_batch_index):
        mock_run_batch_index.return_value = [{"name": "example.com", "success": True}]
        result = runner.invoke(app, ["index", "https://example.com"])
        self.assertEqual(result.exit_code, 0)
        mock_run_batch_index.assert_called_once_with(
            "https://example.com", True, False, False, 1
        )

    @patch("src.services.indexing_orchestrator.run_batch_index")
    def test_index_multiple_urls(self, mock_run_batch_index):
        mock_run_batch_index.return_value = [{"name": "example.com", "success": True}]
        result = runner.invoke(app, ["index", "https://example.com, https://google.com;https://github.com"])
        self.assertEqual(result.exit_code, 0)
        mock_run_batch_index.assert_called_once_with(
            "https://example.com, https://google.com;https://github.com",
            True, False, False, 1
        )

    @patch("src.services.storage_tui.click.getchar")
    @patch("src.cli.get_services")
    def test_storage_documents(self, mock_get_services, mock_getchar):
        """Storage shows document table by default and quits on 'q'."""
        mock_graph_repo = MagicMock()
        mock_get_services.return_value = (mock_graph_repo, MagicMock(), MagicMock(), MagicMock())
        mock_getchar.return_value = 'q'

        mock_graph_repo.get_browse_count.return_value = 1
        mock_graph_repo.get_browse_rows.return_value = [
            {"id": "test_id", "properties": '{"title": "Test Paper", "source_type": "paper", "authors": ["John Doe"]}'}
        ]

        result = runner.invoke(app, ["storage", "--limit", "10"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Test Paper", result.stdout)

    @patch("src.services.storage_tui.click.getchar")
    @patch("src.cli.get_services")
    def test_storage_tab_switch(self, mock_get_services, mock_getchar):
        """Tab key switches between tables."""
        mock_graph_repo = MagicMock()
        mock_get_services.return_value = (mock_graph_repo, MagicMock(), MagicMock(), MagicMock())
        # Tab → switch to Authors, then q to quit
        mock_getchar.side_effect = ['\t', 'q']

        mock_graph_repo.get_browse_count.return_value = 1
        mock_graph_repo.get_browse_rows.return_value = [
            {"id": "john_doe", "properties": '{"name": "John Doe"}', "papers_count": 1}
        ]

        result = runner.invoke(app, ["storage", "--limit", "10"])
        self.assertEqual(result.exit_code, 0)

    @patch("src.cli.get_services")
    def test_reindex_no_filter(self, mock_get_services):
        """Reindex requires at least one filter flag."""
        mock_get_services.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
        result = runner.invoke(app, ["reindex", "meta"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Please specify a filter", result.stdout)

    @patch("src.cli.Indexer")
    @patch("src.cli.get_services")
    def test_reindex_with_filter(self, mock_get_services, mock_indexer_cls):
        """Reindex with --missing-authors filters papers and calls reindex_metadata."""
        from src.models import Paper
        from src.indexer import Indexer
        
        mock_graph_repo = MagicMock()
        mock_get_services.return_value = (mock_graph_repo, MagicMock(), MagicMock(), MagicMock())
        
        real_indexer = Indexer(mock_graph_repo, MagicMock(), MagicMock(), MagicMock())
        real_indexer.reindex_metadata = MagicMock(return_value=True)
        mock_indexer_cls.return_value = real_indexer
        
        mock_graph_repo.get_non_placeholder_paper_ids.return_value = ["p1", "p2", "p3"]
        
        p1 = Paper(id="p1", title="Paper 1", authors=["John Doe"])
        p2 = Paper(id="p2", title="Paper 2", authors=[])
        p3 = None
        mock_graph_repo.get_paper.side_effect = lambda pid: {"p1": p1, "p2": p2, "p3": p3}.get(pid)
        
        result = runner.invoke(app, ["reindex", "meta", "--missing-authors"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Re-indexed 1/1 papers successfully.", result.stdout)
        real_indexer.reindex_metadata.assert_called_once_with("p2", use_llm=False)

    @patch("src.services.storage_tui.click.getchar")
    @patch("src.cli.get_services")
    def test_storage_multi_digit_selection(self, mock_get_services, mock_getchar):
        """Typing a multi-digit number in storage selects that row index."""
        mock_graph_repo = MagicMock()
        mock_get_services.return_value = (mock_graph_repo, MagicMock(), MagicMock(), MagicMock())
        
        # Simulating keystroke sequence: '1', then '2', then '\r' (Enter), then 'q' (Quit)
        mock_getchar.side_effect = ['1', '2', '\r', 'q']

        mock_graph_repo.get_browse_count.return_value = 15
        
        rows = []
        for idx in range(1, 16):
            rows.append({
                "id": f"p{idx}",
                "properties": f'{{"title": "Paper {idx}", "source_type": "paper", "authors": ["Author {idx}"]}}'
            })
        
        mock_graph_repo.get_browse_rows.return_value = rows

        result = runner.invoke(app, ["storage"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Row 12 selected", result.stdout)

    @patch("typer.confirm")
    @patch("src.cli.config")
    def test_reset_command(self, mock_config, mock_confirm):
        """Test reset CLI command with double confirmation prompts."""
        import tempfile
        from pathlib import Path
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            db_file = tmp_path / "test.db"
            db_file.write_text("db content", encoding="utf-8")
            
            usearch_file = tmp_path / "test.usearch"
            usearch_file.write_text("usearch content", encoding="utf-8")
            
            archive_dir = tmp_path / "archive"
            archive_dir.mkdir()
            (archive_dir / "paper.pdf").write_text("pdf content", encoding="utf-8")
            
            mock_config.db_path = str(db_file)
            mock_config.archive_dir = str(archive_dir)
            
            # Case 1: First confirmation declined
            mock_confirm.return_value = False
            result = runner.invoke(app, ["reset"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Reset cancelled", result.stdout)
            self.assertTrue(db_file.exists())
            self.assertTrue(usearch_file.exists())
            self.assertTrue((archive_dir / "paper.pdf").exists())
            
            # Case 2: First confirmation accepted, second declined
            mock_confirm.side_effect = [True, False]
            result = runner.invoke(app, ["reset"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Reset cancelled", result.stdout)
            self.assertTrue(db_file.exists())
            self.assertTrue(usearch_file.exists())
            self.assertTrue((archive_dir / "paper.pdf").exists())
            
            # Case 3: Both confirmations accepted
            mock_confirm.side_effect = None
            mock_confirm.return_value = True
            result = runner.invoke(app, ["reset"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Database and environment successfully reset", result.stdout)
            self.assertFalse(db_file.exists())
            self.assertFalse(usearch_file.exists())
            self.assertFalse((archive_dir / "paper.pdf").exists())
            self.assertTrue(archive_dir.exists())  # the directory itself should remain

    @patch("src.cli.config")
    def test_init_command(self, mock_config):
        """Test config init command."""
        result = runner.invoke(app, ["init"])
        self.assertEqual(result.exit_code, 0)
        mock_config.init_config.assert_called_once()
        self.assertIn("has been successfully updated", result.stdout)

    @patch("src.cli.get_services")
    def test_export_db_command(self, mock_get_services):
        """Test export-db command outputs correct JSON/YAML."""
        mock_graph_repo = MagicMock()
        mock_vector_repo = MagicMock()
        mock_get_services.return_value = (mock_graph_repo, mock_vector_repo, MagicMock(), MagicMock())
        
        mock_graph_repo.get_all_nodes.return_value = [("n1", "Paper", '{"title": "Test Paper"}')]
        mock_graph_repo.get_all_edges.return_value = [("n1", "n2", "AUTHORED", '{}')]
        
        # Test JSON format
        result = runner.invoke(app, ["export-db", "--format", "json", "--no-chunks"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('"id": "n1"', result.stdout)
        self.assertIn('"label": "Paper"', result.stdout)
        
        # Test YAML format
        result = runner.invoke(app, ["export-db", "--format", "yaml", "--no-chunks"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("label: Paper", result.stdout)

        # Test export with chunks (no --no-chunks option)
        mock_conn = MagicMock()
        mock_vector_repo._get_connection.return_value.__enter__.return_value = mock_conn
        mock_conn.execute.return_value.fetchall.return_value = [
            {"id": "paper1#42", "paper_id": "paper1", "text_content": "some text", "page_number": 3}
        ]
        
        result = runner.invoke(app, ["export-db", "--format", "json"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('"idx": 42', result.stdout)
        self.assertIn('"text_content": "some text"', result.stdout)
        self.assertIn('"page_number": 3', result.stdout)

    @patch("src.cli.get_services")
    @patch("src.cli.RAGPipeline")
    def test_query_command_with_cloud(self, mock_rag_pipeline_cls, mock_get_services):
        """Test query command with --cloud flag sets env var."""
        import os
        mock_get_services.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
        mock_pipeline_instance = MagicMock()
        mock_rag_pipeline_cls.return_value = mock_pipeline_instance
        mock_pipeline_instance.ask.return_value = "Cloud Answer"
        
        if "SCIENCE_GRAPH_USE_CLOUD" in os.environ:
            del os.environ["SCIENCE_GRAPH_USE_CLOUD"]
            
        result = runner.invoke(app, ["query", "test question", "--cloud"])
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(os.environ.get("SCIENCE_GRAPH_USE_CLOUD"), "1")

    @patch("src.cli.get_services")
    @patch("src.cli.config")
    @patch("src.cli.Path")
    def test_stats_command(self, mock_path, mock_config, mock_get_services):
        mock_graph_repo = MagicMock()
        mock_get_services.return_value = (mock_graph_repo, MagicMock(), MagicMock(), MagicMock())
        mock_graph_repo.get_stats.return_value = {
            "papers": 10,
            "authors": 20,
            "concepts": 30,
            "edges": 40
        }
        mock_config.db_path = "/mock/db.db"
        mock_config.get_storage_stats.return_value = {
            "storage_dir": "/mock/storage",
            "total_size": 1024 * 1024 * 15,
            "extensions": [
                {"extension": ".pdf", "count": 5, "size": 1024 * 1024 * 10},
                {"extension": ".epub", "count": 2, "size": 1024 * 1024 * 5}
            ],
            "sources": [
                {"source": "paper", "count": 5, "size": 1024 * 1024 * 10},
                {"source": "book", "count": 2, "size": 1024 * 1024 * 5}
            ]
        }
        
        mock_db_file = MagicMock()
        mock_db_file.exists.return_value = True
        mock_db_file.stat.return_value.st_size = 1024 * 1024 * 2
        mock_path.return_value = mock_db_file
        
        result = runner.invoke(app, ["stats"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Knowledge Base Statistics", result.stdout)
        self.assertIn("Papers / Books / Notes", result.stdout)
        self.assertIn("10", result.stdout)
        self.assertIn("2.0 MB", result.stdout)
        self.assertIn("15.00 MB", result.stdout)
        self.assertIn("10.00 MB", result.stdout)

    @patch("src.cli.config")
    @patch("src.cli.Path")
    def test_config_command(self, mock_path, mock_config):
        mock_config.db_path = "/mock/db.db"
        mock_config.archive_dir = "/mock/archive"
        mock_config.llm_provider = "openai"
        mock_config.llm_local_model_path = "/mock/gemma"
        mock_config.llm_cloud_model_name = "gpt-4"
        mock_config.llm_cloud_base_url = "https://api.openai.com/v1"
        mock_config.llm_cloud_api_key = "sk-key"
        mock_config.llm_max_tokens = 512
        mock_config.llm_temp = 0.5
        mock_config.embedding_model_name = "all-mpnet"
        mock_config.chunk_size = 1000
        mock_config.chunk_overlap = 200
        mock_config.spacy_model_name = "en_core_web_sm"
        mock_config.ner_model_name = "ner-model"
        mock_config.data = {"llm": {"cloud": {"provider": "openai"}}}
        
        mock_file = MagicMock()
        mock_file.exists.return_value = True
        mock_path.return_value = mock_file
        
        import os
        with patch.dict("os.environ", {"HF_TOKEN": "mock-token", "HF_HUB_VERBOSITY": "debug", "TOKENIZERS_PARALLELISM": "true"}):
            result = runner.invoke(app, ["config"])
            
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Science Graph", result.stdout)
        self.assertIn("Paths", result.stdout)
        self.assertIn("LLM Model", result.stdout)
        self.assertIn("Embedding Model", result.stdout)

    @patch("src.cli.get_services")
    @patch("src.services.visualizer.generate_html_graph")
    @patch("src.cli.webbrowser.open")
    @patch("src.cli.con.warning")
    def test_visualize_command(self, mock_con_warning, mock_browser_open, mock_generate_html_graph, mock_get_services):
        mock_get_services.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
        
        # Success path
        result = runner.invoke(app, ["visualize", "-o", "/mock/graph.html"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Graph saved to", result.stdout)
        mock_generate_html_graph.assert_called_once()
        mock_browser_open.assert_called_once()
        
        # Error path: generate_html_graph raises ValueError
        mock_generate_html_graph.reset_mock()
        mock_browser_open.reset_mock()
        mock_generate_html_graph.side_effect = ValueError("No nodes in graph")
        
        result = runner.invoke(app, ["visualize", "-o", "/mock/graph.html"])
        self.assertEqual(result.exit_code, 0)
        mock_con_warning.assert_called_once_with("No nodes in graph")
        mock_generate_html_graph.assert_called_once()
        mock_browser_open.assert_not_called()
        
        # Browser open throws an exception
        mock_generate_html_graph.reset_mock()
        mock_generate_html_graph.side_effect = None
        mock_browser_open.side_effect = Exception("Browser error")
        mock_con_warning.reset_mock()
        
        result = runner.invoke(app, ["visualize", "-o", "/mock/graph.html"])
        self.assertEqual(result.exit_code, 0)
        mock_con_warning.assert_called_once_with("Could not open browser: Browser error")

    @patch("src.cli.get_services")
    @patch("src.cli.RAGPipeline")
    @patch("src.tui.run_tui_chat")
    @patch("src.cli.con.error")
    def test_chat_command(self, mock_con_error, mock_run_tui_chat, mock_rag_pipeline_cls, mock_get_services):
        # Case 1: LLM Engine not available
        mock_get_services.return_value = (MagicMock(), MagicMock(), MagicMock(), None)
        
        result = runner.invoke(app, ["chat"])
        self.assertEqual(result.exit_code, 1)
        mock_con_error.assert_called_once_with("LLM engine is not available. Run: graph config")
        
        # Case 2: LLM Engine available, starts TUI
        mock_llm_engine = MagicMock()
        mock_get_services.return_value = (MagicMock(), MagicMock(), MagicMock(), mock_llm_engine)
        mock_pipeline_instance = MagicMock()
        mock_rag_pipeline_cls.return_value = mock_pipeline_instance
        
        result = runner.invoke(app, ["chat", "--cloud"])
        self.assertEqual(result.exit_code, 0)
        mock_run_tui_chat.assert_called_once_with(mock_pipeline_instance)

    @patch("src.cli.get_services")
    @patch("src.review_agent.ReviewAgent")
    @patch("src.cli.con.error")
    def test_review_command(self, mock_con_error, mock_review_agent_cls, mock_get_services):
        # Case 1: LLM Engine not available
        mock_get_services.return_value = (MagicMock(), MagicMock(), MagicMock(), None)
        
        result = runner.invoke(app, ["review", "Deep Learning"])
        self.assertEqual(result.exit_code, 1)
        mock_con_error.assert_called_once_with("LLM engine is required for review generation. Run: graph config")
        
        # Case 2: LLM Engine available, generates review
        mock_llm_engine = MagicMock()
        mock_get_services.return_value = (MagicMock(), MagicMock(), MagicMock(), mock_llm_engine)
        
        mock_agent_instance = MagicMock()
        mock_review_agent_cls.return_value = mock_agent_instance
        mock_agent_instance.run.return_value = "This is a detailed markdown literature review of Deep Learning."
        
        result = runner.invoke(app, ["review", "Deep Learning", "--fast", "--cloud"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Preview", result.stdout)
        self.assertIn("This is a detailed markdown literature review", result.stdout)
        mock_agent_instance.run.assert_called_once()

    @patch("src.cli.get_services")
    @patch("src.cli.con.error")
    def test_serve_command_missing_uvicorn(self, mock_con_error, mock_get_services):
        with patch.dict("sys.modules", {"uvicorn": None}):
            result = runner.invoke(app, ["serve"])
            self.assertEqual(result.exit_code, 1)
            mock_con_error.assert_called_once_with("uvicorn is not installed. Run: uv add uvicorn")

    @patch("src.cli.get_services")
    @patch("uvicorn.run")
    @patch("src.cli.webbrowser.open")
    @patch("threading.Thread")
    @patch("time.sleep")
    def test_serve_command_success(self, mock_sleep, mock_thread, mock_browser_open, mock_uvicorn_run, mock_get_services):
        mock_thread.side_effect = lambda target, daemon: MagicMock(start=target)
        
        result = runner.invoke(app, ["serve", "--open"])
        self.assertEqual(result.exit_code, 0)
        mock_uvicorn_run.assert_called_once_with(
            "src.web_app:app",
            host="127.0.0.1",
            port=8000,
            reload=False,
            log_level="warning",
        )
        mock_browser_open.assert_called_once_with("http://127.0.0.1:8000")
        mock_sleep.assert_called_once_with(1.5)

    @patch("src.mcp_server.mcp")
    def test_serve_mcp_command(self, mock_mcp):
        # Case 1: stdio transport
        result = runner.invoke(app, ["serve-mcp"])
        self.assertEqual(result.exit_code, 0)
        mock_mcp.run.assert_called_once_with(transport="stdio")
        
        # Case 2: SSE transport
        mock_mcp.run.reset_mock()
        result = runner.invoke(app, ["serve-mcp", "--sse", "--host", "0.0.0.0", "-p", "9000"])
        self.assertEqual(result.exit_code, 0)
        mock_mcp.run.assert_called_once_with(transport="sse", host="0.0.0.0", port=9000)

    @patch("src.cli.LLMEngine")
    @patch("src.services.extraction_service.ExtractionService")
    @patch("src.cli.Path")
    def test_extract_file_command(self, mock_path, mock_extraction_service_cls, mock_llm_engine_cls):
        # Mock file content
        mock_file = MagicMock()
        mock_file.exists.return_value = True
        mock_file.read_text.return_value = "# Test Title\n\nThis is abstract.\n\nThis is body text."
        mock_path.return_value = mock_file
        
        # Mock LLMEngine and ExtractionService
        mock_llm_instance = MagicMock()
        mock_llm_engine_cls.return_value = mock_llm_instance
        
        mock_extractor_instance = MagicMock()
        mock_extraction_service_cls.return_value = mock_extractor_instance
        
        mock_extraction_result = MagicMock()
        mock_extraction_result.authors = ["John Doe"]
        mock_extraction_result.concepts = [{"name": "ML", "description": "machine learning"}]
        mock_extraction_result.tags = ["tag1"]
        mock_extractor_instance.extract.return_value = mock_extraction_result
        
        # Run command
        result = runner.invoke(app, ["extract-file", "test.txt", "--no-llm"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("John Doe", result.stdout)
        
        # Extract title from path.stem when no markdown headers
        mock_file.read_text.return_value = "No headers here.\n\nAbstract here."
        mock_file.stem = "test_stem"
        result = runner.invoke(app, ["extract-file", "test.txt"])
        self.assertEqual(result.exit_code, 0)
        
        # Case 3: File not found
        mock_file.exists.return_value = False
        result = runner.invoke(app, ["extract-file", "nonexistent.txt"])
        self.assertEqual(result.exit_code, 1)
        
        # Case 4: Read error
        mock_file.exists.return_value = True
        mock_file.read_text.side_effect = Exception("Read failed")
        result = runner.invoke(app, ["extract-file", "test.txt"])
        self.assertEqual(result.exit_code, 1)
        
        # Case 5: Extraction fails
        mock_file.read_text.side_effect = None
        mock_file.read_text.return_value = "content"
        mock_extractor_instance.extract.side_effect = Exception("Extraction error")
        result = runner.invoke(app, ["extract-file", "test.txt"])
        self.assertEqual(result.exit_code, 1)




