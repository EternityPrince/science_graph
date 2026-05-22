import unittest
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner
from src.cli import app

runner = CliRunner()

class TestCLI(unittest.TestCase):
    @patch("src.cli.Indexer")
    @patch("src.cli.get_services")
    def test_index_url(self, mock_get_services, mock_indexer_cls):
        mock_indexer_instance = MagicMock()
        mock_indexer_cls.return_value = mock_indexer_instance
        mock_get_services.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
        result = runner.invoke(app, ["index", "https://example.com"])
        self.assertEqual(result.exit_code, 0)
        mock_indexer_instance.index_url.assert_called_once_with("https://example.com")
    @patch("click.getchar")
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

    @patch("click.getchar")
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
        mock_indexer_instance = MagicMock()
        mock_indexer_cls.return_value = mock_indexer_instance
        
        mock_graph_repo = MagicMock()
        mock_get_services.return_value = (mock_graph_repo, MagicMock(), MagicMock(), MagicMock())
        
        mock_graph_repo.get_non_placeholder_paper_ids.return_value = ["p1", "p2", "p3"]
        
        p1 = Paper(id="p1", title="Paper 1", authors=["John Doe"])
        p2 = Paper(id="p2", title="Paper 2", authors=[])
        p3 = None
        mock_graph_repo.get_paper.side_effect = lambda pid: {"p1": p1, "p2": p2, "p3": p3}.get(pid)
        
        mock_indexer_instance.reindex_metadata.return_value = True
        
        result = runner.invoke(app, ["reindex", "meta", "--missing-authors"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Re-indexed 1/1 papers successfully.", result.stdout)
        mock_indexer_instance.reindex_metadata.assert_called_once_with("p2", use_llm=False)

    @patch("click.getchar")
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




