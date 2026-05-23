import unittest
from unittest.mock import MagicMock, patch
from src.tui import TUIHistory, run_tui_chat, handle_command

class TestTUI(unittest.TestCase):
    def test_tui_history_accumulation(self):
        history = TUIHistory(max_turns=2)
        history.add_turn("Q1", "A1")
        history.add_turn("Q2", "A2")
        self.assertEqual(len(history.turns), 2)
        history.add_turn("Q3", "A3")
        self.assertEqual(history.turns[0][0], "Q2")

    @patch("src.tui.PromptSession.prompt")
    @patch("rich.console.Console.print")
    def test_run_tui_chat_exit(self, mock_print, mock_prompt):
        mock_prompt.side_effect = ["exit"]
        rag_mock = MagicMock()
        rag_mock.graph_repo.get_stats.return_value = {"papers": 1}
        
        run_tui_chat(rag_mock)
        mock_prompt.assert_called_once()
        mock_print.assert_any_call("[bold red]Ending chat session. Goodbye![/bold red]")

    @patch("src.tui.Indexer")
    @patch("src.cli.get_services")
    @patch("rich.console.Console.print")
    def test_handle_command_index_url(self, mock_print, mock_get_services, mock_indexer_cls):
        rag_mock = MagicMock()
        mock_indexer_instance = MagicMock()
        mock_indexer_cls.return_value = mock_indexer_instance
        
        console_mock = MagicMock()
        handle_command("/index https://example.com", rag_mock, console_mock)
        mock_indexer_instance.index_url.assert_called_once_with("https://example.com")
