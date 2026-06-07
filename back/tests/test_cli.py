import os
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

    def test_cli_import_warnings(self):
        import importlib
        import os
        
        with patch.dict(os.environ, {"PYTHONWARNINGS": "ignore:some_other_warning"}):
            import src.cli
            importlib.reload(src.cli)
            self.assertIn("ignore:resource_tracker:UserWarning", os.environ["PYTHONWARNINGS"])

    @patch("src.cli.container")
    @patch("src.cli.con.error")
    def test_get_services(self, mock_con_error, mock_container):
        from src.cli import get_services
        
        mock_graph = MagicMock()
        mock_vector = MagicMock()
        mock_emb = MagicMock()
        mock_llm = MagicMock()
        mock_container.get_graph_repo.return_value = mock_graph
        mock_container.get_vector_repo.return_value = mock_vector
        mock_container.get_embedding_engine.return_value = mock_emb
        mock_container.get_llm_engine.return_value = mock_llm
        
        r_graph, r_vector, r_emb, r_llm = get_services(load_llm=True, load_embeddings=True, use_cloud=True)
        self.assertEqual(r_graph, mock_graph)
        self.assertEqual(r_vector, mock_vector)
        self.assertEqual(r_emb, mock_emb)
        self.assertEqual(r_llm, mock_llm)
        mock_container.get_llm_engine.assert_called_with(use_cloud=True)
        
        r_graph, r_vector, r_emb, r_llm = get_services(load_llm=False, load_embeddings=False)
        self.assertIsNone(r_emb)
        self.assertIsNone(r_llm)
        
        mock_container.get_llm_engine.side_effect = Exception("LLM failure")
        r_graph, r_vector, r_emb, r_llm = get_services(load_llm=True, load_embeddings=False)
        self.assertIsNone(r_llm)
        mock_con_error.assert_called_once_with("Could not load LLM engine: LLM failure")

    @patch("src.cli.con.console.print")
    def test_print_trace_table_custom_stages(self, mock_print):
        from src.cli import print_trace_table
        
        trace_info = {
            "stages": {
                "Custom Stage": 1.23,
            },
            "tokens": {
                "Custom Stage": 100,
                "Token Only Stage": 500,
            }
        }
        print_trace_table("Test Doc", trace_info)
        mock_print.assert_called()

    @patch("src.cli.con.console.print")
    def test_print_session_summary_table_with_sizes(self, mock_print):
        from src.cli import print_session_summary_table
        
        session_traces = [
            {
                "success": True,
                "original_size": 1024 * 1024 * 10,
                "compressed_size": 1024 * 1024 * 7,
                "authors_count": 2,
                "concepts_count": 5,
                "tags_count": 3,
                "references_count": 4,
                "stages": {"stage1": 1.0},
                "tokens": {"stage1": 150}
            }
        ]
        print_session_summary_table(session_traces)
        mock_print.assert_called()

    @patch("src.services.indexing_orchestrator.run_batch_index")
    @patch("src.cli.con.error")
    def test_index_orchestrator_exception(self, mock_con_error, mock_run_batch_index):
        mock_run_batch_index.side_effect = Exception("Batch indexing failed")
        result = runner.invoke(app, ["index", "https://example.com"])
        self.assertEqual(result.exit_code, 1)
        mock_con_error.assert_called_once_with("Failed during batch indexing: Batch indexing failed")

    @patch("src.cli.get_services")
    @patch("src.cli.Indexer")
    def test_reindex_meta_cloud(self, mock_indexer_cls, mock_get_services):
        mock_get_services.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
        with patch.dict(os.environ, {}):
            if "SCIENCE_GRAPH_USE_CLOUD" in os.environ:
                del os.environ["SCIENCE_GRAPH_USE_CLOUD"]
            result = runner.invoke(app, ["reindex", "meta", "--all-metadata", "--cloud"])
            self.assertEqual(result.exit_code, 0)
            self.assertEqual(os.environ.get("SCIENCE_GRAPH_USE_CLOUD"), "1")

    @patch("src.cli.get_services")
    @patch("src.cli.Indexer")
    @patch("src.cli.con.warning")
    def test_reindex_meta_no_llm_engine(self, mock_con_warning, mock_indexer_cls, mock_get_services):
        mock_get_services.return_value = (MagicMock(), MagicMock(), MagicMock(), None)
        result = runner.invoke(app, ["reindex", "meta", "--all-metadata", "--use-llm"])
        self.assertEqual(result.exit_code, 0)
        mock_con_warning.assert_called_once_with("Proceeding with regex fallback extraction because LLM engine failed to load.")

    @patch("src.cli.get_services")
    @patch("src.cli.Indexer")
    def test_reindex_full_cloud(self, mock_indexer_cls, mock_get_services):
        mock_get_services.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
        with patch.dict(os.environ, {}):
            if "SCIENCE_GRAPH_USE_CLOUD" in os.environ:
                del os.environ["SCIENCE_GRAPH_USE_CLOUD"]
            result = runner.invoke(app, ["reindex", "full", "--all", "--cloud"])
            self.assertEqual(result.exit_code, 0)
            self.assertEqual(os.environ.get("SCIENCE_GRAPH_USE_CLOUD"), "1")

    @patch("src.cli.con.warning")
    def test_reindex_full_no_args(self, mock_con_warning):
        result = runner.invoke(app, ["reindex", "full"])
        self.assertEqual(result.exit_code, 0)
        mock_con_warning.assert_called_once_with("Please specify either --all or --id <paper_id>")

    @patch("src.cli.get_services")
    @patch("src.cli.Indexer")
    @patch("src.cli.con.warning")
    def test_reindex_full_no_llm_engine(self, mock_con_warning, mock_indexer_cls, mock_get_services):
        mock_get_services.return_value = (MagicMock(), MagicMock(), MagicMock(), None)
        result = runner.invoke(app, ["reindex", "full", "--all", "--use-llm"])
        self.assertEqual(result.exit_code, 0)
        mock_con_warning.assert_called_once_with("Proceeding with regex fallback extraction because LLM engine failed to load.")

    @patch("src.cli.get_services")
    @patch("src.cli.Indexer")
    @patch("src.cli.con.error")
    def test_reindex_full_value_error(self, mock_con_error, mock_indexer_cls, mock_get_services):
        mock_get_services.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
        mock_indexer = MagicMock()
        mock_indexer.reindex_full_batch.side_effect = ValueError("Invalid paper ID")
        mock_indexer_cls.return_value = mock_indexer
        
        result = runner.invoke(app, ["reindex", "full", "--id", "invalid_id"])
        self.assertEqual(result.exit_code, 1)
        mock_con_error.assert_called_once_with("Invalid paper ID")

    @patch("src.cli.get_services")
    @patch("src.cli.con.error")
    def test_query_no_llm_engine(self, mock_con_error, mock_get_services):
        mock_get_services.return_value = (MagicMock(), MagicMock(), MagicMock(), None)
        result = runner.invoke(app, ["query", "test question"])
        self.assertEqual(result.exit_code, 1)
        mock_con_error.assert_called_once_with("LLM engine is required for query. Check your model path with: graph config")

    @patch("src.cli.get_services")
    @patch("src.cli.RAGPipeline")
    def test_query_with_trace(self, mock_rag_pipeline_cls, mock_get_services):
        mock_get_services.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
        mock_pipeline = MagicMock()
        mock_rag_pipeline_cls.return_value = mock_pipeline
        mock_pipeline.ask.return_value = "Answer"
        
        import src.cli
        result = runner.invoke(app, ["query", "test question", "--trace"])
        self.assertEqual(result.exit_code, 0)
        self.assertTrue(src.cli.con.SHOW_TIME)
        mock_pipeline._get_reranker.assert_called_once()

    @patch("src.cli.get_services")
    @patch("src.cli.config")
    @patch("src.cli.Path.exists", return_value=True)
    @patch("src.cli.Path.stat")
    def test_stats_command_sizes_breakdown(self, mock_stat, mock_exists, mock_config, mock_get_services):
        mock_graph_repo = MagicMock()
        mock_get_services.return_value = (mock_graph_repo, MagicMock(), MagicMock(), MagicMock())
        mock_graph_repo.get_stats.return_value = {
            "papers": 1, "authors": 1, "concepts": 1, "edges": 1
        }
        mock_config.db_path = "/mock/db.db"
        mock_config.get_storage_stats.return_value = {
            "storage_dir": "/mock/storage",
            "total_size": 2 * 1024 * 1024 * 1024,
            "extensions": [
                {"extension": ".pdf", "count": 1, "size": 500 * 1024},
                {"extension": ".md", "count": 1, "size": 100}
            ],
            "sources": [
                {"source": "paper", "count": 1, "size": 500 * 1024},
                {"source": "note", "count": 1, "size": 100}
            ]
        }
        
        mock_stat.return_value.st_size = 100
        
        result = runner.invoke(app, ["stats"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("2.00 GB", result.stdout)
        self.assertIn("500.00 KB", result.stdout)
        self.assertIn("100 B", result.stdout)

    @patch("src.cli.get_services")
    @patch("src.cli.container")
    @patch("src.services.storage_tui.run_storage_tui")
    def test_storage_cloud(self, mock_run_storage_tui, mock_container, mock_get_services):
        mock_get_services.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
        with patch.dict(os.environ, {}):
            if "SCIENCE_GRAPH_USE_CLOUD" in os.environ:
                del os.environ["SCIENCE_GRAPH_USE_CLOUD"]
            result = runner.invoke(app, ["storage", "--cloud"])
            self.assertEqual(result.exit_code, 0)
            self.assertEqual(os.environ.get("SCIENCE_GRAPH_USE_CLOUD"), "1")

    @patch("src.cli.config")
    @patch("src.cli.Path.exists", return_value=True)
    def test_config_command_models_info(self, mock_exists, mock_config):
        mock_config.db_path = "/mock/db.db"
        mock_config.archive_dir = "/mock/archive"
        mock_config.llm_provider = "mlx"
        mock_config.llm_cloud_model_name = "gpt-4"
        mock_config.llm_cloud_base_url = "https://api.openai.com/v1"
        mock_config.llm_cloud_api_key = "sk-key"
        mock_config.llm_max_tokens = 512
        mock_config.llm_temp = 0.5
        mock_config.chunk_size = 1000
        mock_config.chunk_overlap = 200
        mock_config.spacy_model_name = "en_core_web_sm"
        mock_config.ner_model_name = "ner-model"
        mock_config.embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        mock_config.data = {"llm": {"cloud": {"provider": "openai"}}}
        
        mock_config.llm_local_model_path = "/mock/gemma-2b"
        result = runner.invoke(app, ["config"])
        self.assertIn("Google Gemma (MLX)", result.stdout)
        
        mock_config.llm_local_model_path = "/mock/qwen-1.8b"
        result = runner.invoke(app, ["config"])
        self.assertIn("Alibaba Qwen (MLX)", result.stdout)
        
        mock_config.llm_local_model_path = "/mock/llama-3"
        result = runner.invoke(app, ["config"])
        self.assertIn("Meta LLaMA (MLX)", result.stdout)
        
        mock_config.llm_local_model_path = "/mock/mistral-7b"
        result = runner.invoke(app, ["config"])
        self.assertIn("Mistral (MLX)", result.stdout)
        
        mock_config.embedding_model_name = "/mock/minilm-model"
        result = runner.invoke(app, ["config"])
        self.assertIn("MiniLM sentence embedder", result.stdout)
        
        mock_config.embedding_model_name = "/mock/all-mpnet-base"
        result = runner.invoke(app, ["config"])
        self.assertIn("MPNet sentence embedder", result.stdout)
        
        mock_config.embedding_model_name = "/mock/mxbai-rerank"
        result = runner.invoke(app, ["config"])
        self.assertIn("MixedBread cross-encoder", result.stdout)
        
        mock_config.embedding_model_name = "/mock/unknown-model"
        result = runner.invoke(app, ["config"])
        self.assertIn("Unknown", result.stdout)

    @patch("src.cli.get_services")
    @patch("uvicorn.run")
    def test_serve_cloud(self, mock_uvicorn_run, mock_get_services):
        with patch.dict(os.environ, {}):
            if "SCIENCE_GRAPH_USE_CLOUD" in os.environ:
                del os.environ["SCIENCE_GRAPH_USE_CLOUD"]
            result = runner.invoke(app, ["serve", "--cloud", "--no-open"])
            self.assertEqual(result.exit_code, 0)
            self.assertEqual(os.environ.get("SCIENCE_GRAPH_USE_CLOUD"), "1")

    @patch("src.cli.Path.exists", return_value=True)
    @patch("src.cli.Path.read_text", return_value="content")
    @patch("src.cli.LLMEngine")
    @patch("src.services.extraction_service.ExtractionService")
    def test_extract_file_cloud(self, mock_extraction_service_cls, mock_llm_engine_cls, mock_read_text, mock_exists):
        mock_extractor_instance = MagicMock()
        mock_extraction_result = MagicMock()
        mock_extraction_result.authors = ["John Doe"]
        mock_extraction_result.concepts = [{"name": "ML", "description": "machine learning"}]
        mock_extraction_result.tags = ["tag1"]
        mock_extractor_instance.extract.return_value = mock_extraction_result
        mock_extraction_service_cls.return_value = mock_extractor_instance

        with patch.dict(os.environ, {}):
            if "SCIENCE_GRAPH_USE_CLOUD" in os.environ:
                del os.environ["SCIENCE_GRAPH_USE_CLOUD"]
            result = runner.invoke(app, ["extract-file", "test.txt", "--cloud"])
            self.assertEqual(result.exit_code, 0)
            self.assertEqual(os.environ.get("SCIENCE_GRAPH_USE_CLOUD"), "1")

    @patch("src.cli.Path.exists", return_value=True)
    @patch("src.cli.Path.read_text", return_value="content")
    @patch("src.cli.LLMEngine")
    @patch("src.services.extraction_service.ExtractionService")
    @patch("src.cli.con.warning")
    def test_extract_file_llm_engine_failure(self, mock_con_warning, mock_extraction_service_cls, mock_llm_engine_cls, mock_read_text, mock_exists):
        mock_llm_engine_cls.side_effect = Exception("Engine initialization failed")
        
        mock_extractor_instance = MagicMock()
        mock_extraction_result = MagicMock()
        mock_extraction_result.authors = ["John Doe"]
        mock_extraction_result.concepts = [{"name": "ML", "description": "machine learning"}]
        mock_extraction_result.tags = ["tag1"]
        mock_extractor_instance.extract.return_value = mock_extraction_result
        mock_extraction_service_cls.return_value = mock_extractor_instance

        result = runner.invoke(app, ["extract-file", "test.txt"])
        self.assertEqual(result.exit_code, 0)
        mock_con_warning.assert_called_once_with(
            "Could not load LLM engine: Engine initialization failed. Falling back to regex extraction."
        )

    @patch("src.cli.get_services")
    @patch("src.cli.con.info")
    def test_cleanup_command_no_orphans(self, mock_con_info, mock_get_services):
        mock_graph_repo = MagicMock()
        mock_get_services.return_value = (mock_graph_repo, MagicMock(), MagicMock(), MagicMock())
        mock_graph_repo.cleanup_orphaned_concepts.return_value = 0
        
        result = runner.invoke(app, ["cleanup"])
        self.assertEqual(result.exit_code, 0)
        mock_con_info.assert_called_once_with("No orphaned concept nodes found in the database.")

    @patch("typer.confirm")
    @patch("src.cli.config")
    @patch("src.cli.con.error")
    @patch("src.cli.Path")
    @patch("shutil.rmtree")
    def test_reset_command_errors_and_subdirs(self, mock_rmtree, mock_path_cls, mock_con_error, mock_config, mock_confirm):
        mock_confirm.return_value = True
        
        mock_db_path = MagicMock()
        mock_db_path.exists.return_value = True
        mock_db_path.unlink.side_effect = Exception("Db unlink error")
        mock_db_path.__str__.return_value = "/mock/db.db"
        
        mock_side_file = MagicMock()
        mock_side_file.exists.return_value = True
        mock_side_file.unlink.side_effect = Exception("Side unlink error")
        
        mock_usearch_path = MagicMock()
        mock_usearch_path.exists.return_value = True
        mock_usearch_path.unlink.side_effect = Exception("Usearch unlink error")
        
        mock_archive_dir = MagicMock()
        mock_archive_dir.exists.return_value = True
        mock_child_dir = MagicMock()
        mock_child_dir.is_file.return_value = False
        mock_child_dir.is_symlink.return_value = False
        mock_child_dir.is_dir.return_value = True
        mock_archive_dir.iterdir.return_value = [mock_child_dir]
        
        mock_archive_dir.iterdir.side_effect = Exception("Archive iterdir error")
        
        mock_config.db_path = "/mock/db.db"
        mock_config.archive_dir = "/mock/archive"
        
        def path_side_effect(arg):
            arg_str = str(arg)
            if arg_str == "/mock/db.db":
                return mock_db_path
            elif arg_str.startswith("/mock/db.db-"):
                return mock_side_file
            elif arg_str == "/mock/db.usearch":
                return mock_usearch_path
            elif arg_str == "/mock/archive":
                return mock_archive_dir
            return MagicMock()
            
        mock_path_cls.side_effect = path_side_effect
        
        result = runner.invoke(app, ["reset"])
        self.assertEqual(result.exit_code, 0)
        
        mock_con_error.assert_any_call("Failed to delete SQLite database: Db unlink error")
        mock_con_error.assert_any_call("Failed to delete vector index: Usearch unlink error")
        mock_con_error.assert_any_call("Failed to clear archive directory: Archive iterdir error")
        
        mock_archive_dir.iterdir.side_effect = None
        mock_archive_dir.iterdir.return_value = [mock_child_dir]
        mock_con_error.reset_mock()
        
        result = runner.invoke(app, ["reset"])
        self.assertEqual(result.exit_code, 0)
        mock_rmtree.assert_called_once_with(mock_child_dir)

    @patch("src.cli.get_services")
    @patch("src.services.doctor_service.DoctorService")
    @patch("src.cli.con.success")
    def test_doctor_command_no_anomalies(self, mock_con_success, mock_doctor_service_cls, mock_get_services):
        mock_get_services.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
        mock_doctor = MagicMock()
        mock_doctor.run_diagnostics.return_value = {
            "stats": {
                "papers_checked": 0, "papers_fixed": 0,
                "authors_checked": 0, "authors_fixed": 0, "authors_migrated": 0, "authors_merged": 0,
                "concepts_checked": 0, "concepts_fixed": 0, "concepts_migrated": 0, "concepts_merged": 0,
                "chunks_checked": 0, "chunks_fixed": 0
            },
            "anomalies": {
                "papers": [],
                "authors": [],
                "concepts": [],
                "chunks": []
            }
        }
        mock_doctor_service_cls.return_value = mock_doctor
        
        result = runner.invoke(app, ["doctor"])
        self.assertEqual(result.exit_code, 0)
        mock_con_success.assert_called_once_with("🎉 No anomalies found! Database texts are completely sanitized and formatted.")

    @patch("src.cli.get_services")
    @patch("src.services.doctor_service.DoctorService")
    def test_doctor_command_with_anomalies(self, mock_doctor_service_cls, mock_get_services):
        mock_get_services.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
        mock_doctor = MagicMock()
        
        mock_doctor.run_diagnostics.return_value = {
            "stats": {
                "papers_checked": 5, "papers_fixed": 1,
                "authors_checked": 5, "authors_fixed": 1, "authors_migrated": 1, "authors_merged": 1,
                "concepts_checked": 5, "concepts_fixed": 1, "concepts_migrated": 1, "concepts_merged": 1,
                "chunks_checked": 5, "chunks_fixed": 1
            },
            "anomalies": {
                "papers": [
                    {
                        "id": "paper1",
                        "old_title": "old", "new_title": "new",
                        "old_abstract": "old_abs", "new_abstract": "new_abs",
                        "old_authors": "old_aut", "new_authors": "new_aut",
                        "missing_abstract": True, "generated_abstract": True,
                        "missing_summary": True, "generated_summary": False
                    }
                ],
                "authors": [
                    {
                        "id": "author1",
                        "old_name": "old_n", "new_name": "new_n",
                        "action": "merge"
                    }
                ],
                "concepts": [],
                "chunks": ["chunk1"]
            }
        }
        mock_doctor_service_cls.return_value = mock_doctor
        
        result = runner.invoke(app, ["doctor", "--fix"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("paper1", result.stdout)
        self.assertIn("Title: \"old\" -> \"new\"", result.stdout)
        self.assertIn("Abstract updated", result.stdout)
        self.assertIn("Authors: old_aut -> new_aut", result.stdout)
        self.assertIn("Abstract: generated", result.stdout)
        self.assertIn("Summary: missing", result.stdout)
        self.assertIn("author1", result.stdout)
        self.assertIn("Action: merge", result.stdout)
        self.assertIn("Found 1 chunk text content anomalies", result.stdout)

    @patch("src.cli.config")
    @patch("src.cli.con.error")
    def test_init_command_exception(self, mock_con_error, mock_config):
        mock_config.init_config.side_effect = Exception("Init error")
        result = runner.invoke(app, ["init"])
        self.assertEqual(result.exit_code, 1)
        mock_con_error.assert_called_once_with("Failed to initialize configuration: Init error")

    @patch("src.cli.get_services")
    def test_export_db_command_exceptions(self, mock_get_services):
        mock_graph_repo = MagicMock()
        mock_vector_repo = MagicMock()
        mock_get_services.return_value = (mock_graph_repo, mock_vector_repo, MagicMock(), MagicMock())
        
        mock_graph_repo.get_all_nodes.return_value = [("n1", "Paper", "invalid json")]
        mock_graph_repo.get_all_edges.return_value = [("n1", "n2", "AUTHORED", "invalid json")]
        
        mock_conn = MagicMock()
        mock_vector_repo._get_connection.return_value.__enter__.return_value = mock_conn
        mock_conn.execute.return_value.fetchall.return_value = [
            {"id": "paper1#abc", "paper_id": "paper1", "text_content": "some text", "page_number": 3}
        ]
        
        result = runner.invoke(app, ["export-db", "--format", "json"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('"properties": {}', result.stdout)
        self.assertIn('"idx": 0', result.stdout)

    @patch("typer.Typer.__call__")
    def test_cli_main_entrypoint(self, mock_typer_call):
        import runpy
        import pathlib
        
        cli_path = pathlib.Path(__file__).parent.parent / "src" / "cli.py"
        with patch("sys.argv", ["graph"]):
            runpy.run_path(str(cli_path), run_name="__main__")
            
        mock_typer_call.assert_called_once()






