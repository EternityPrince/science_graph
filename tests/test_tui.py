import unittest
from unittest.mock import MagicMock, patch
from src.tui import TUIHistory, run_tui_chat

class TestTUI(unittest.TestCase):
    def test_tui_history_accumulation(self):
        history = TUIHistory(max_turns=2)
        
        # Add turns
        history.add_turn("What is Transformer?", "A neural network architecture.")
        history.add_turn("What is Self-Attention?", "A mechanism relating positions.")
        
        # Verify turns
        self.assertEqual(len(history.turns), 2)
        
        # Add third turn (should pop the first due to max_turns=2)
        history.add_turn("What is RAG?", "Retrieval-Augmented Generation.")
        self.assertEqual(len(history.turns), 2)
        self.assertEqual(history.turns[0][0], "What is Self-Attention?")
        self.assertEqual(history.turns[1][0], "What is RAG?")
        
        # Verify formatting
        formatted = history.format_for_llm()
        self.assertIn("What is Self-Attention?", formatted)
        self.assertIn("Retrieval-Augmented Generation.", formatted)
        self.assertNotIn("What is Transformer?", formatted)

    @patch("rich.prompt.Prompt.ask")
    @patch("rich.console.Console.print")
    def test_run_tui_chat_exit(self, mock_print, mock_ask):
        # Setup mock inputs: immediately exit
        mock_ask.side_effect = ["exit"]
        
        rag_mock = MagicMock()
        # Mock graph stats to prevent crashes during welcome print
        rag_mock.graph_repo.get_stats.return_value = {"papers": 1, "authors": 1, "concepts": 1, "edges": 1}
        
        run_tui_chat(rag_mock)
        
        # Verify loop ends and goodbye is printed
        mock_ask.assert_called_once()
        mock_print.assert_any_call("[bold red]Ending chat session. Goodbye![/bold red]")
