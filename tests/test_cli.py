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
    @patch("src.cli.sqlite3.connect")
    @patch("src.cli.get_services")
    def test_storage(self, mock_get_services, mock_connect, mock_getchar):
        mock_get_services.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        mock_getchar.return_value = 'q'
        
        mock_conn.execute.return_value.fetchall.side_effect = [
            [{"id": "test_id", "properties": '{"title": "Test Paper", "source_type": "paper", "authors": ["John Doe"]}'}],
            [], # authors
            [{"id": "test_concept", "properties": '{"name": "Test Concept"}', "degree": 5}]
        ]
        
        mock_conn.execute.return_value.fetchone.side_effect = [
            (1,), (0,), (1,) # total_docs, total_authors, total_concepts
        ]
        
        result = runner.invoke(app, ["storage", "--limit", "10"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Test Paper", result.stdout)
        self.assertIn("Test Concept", result.stdout)
