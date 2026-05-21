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
    def test_storage_documents(self, mock_get_services, mock_connect, mock_getchar):
        """Storage shows document table by default and quits on 'q'."""
        mock_get_services.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        mock_getchar.return_value = 'q'

        # _count() calls fetchone, _get_rows() calls fetchall
        mock_conn.execute.return_value.fetchone.return_value = (1,)
        paper_row = MagicMock()
        paper_row.__getitem__ = lambda s, k: {"id": "test_id", "properties": '{"title": "Test Paper", "source_type": "paper", "authors": ["John Doe"]}'}[k]
        mock_conn.execute.return_value.fetchall.return_value = [paper_row]

        result = runner.invoke(app, ["storage", "--limit", "10"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Test Paper", result.stdout)

    @patch("click.getchar")
    @patch("src.cli.sqlite3.connect")
    @patch("src.cli.get_services")
    def test_storage_tab_switch(self, mock_get_services, mock_connect, mock_getchar):
        """Tab key switches between tables."""
        mock_get_services.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        # Tab → switch to Authors, then q to quit
        mock_getchar.side_effect = ['\t', 'q']

        mock_conn.execute.return_value.fetchone.return_value = (1,)
        author_row = MagicMock()
        author_row.__getitem__ = lambda s, k: {
            "id": "john_doe", "properties": '{"name": "John Doe"}', "papers_count": 1
        }[k]
        mock_conn.execute.return_value.fetchall.return_value = [author_row]

        result = runner.invoke(app, ["storage", "--limit", "10"])
        self.assertEqual(result.exit_code, 0)

