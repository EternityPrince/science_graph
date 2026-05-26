import pytest
from unittest.mock import patch, MagicMock
from src.mcp_server import get_stats, search_papers, query_rag, get_paper_details, get_notes

def test_mcp_get_stats():
    with patch("src.mcp_server.get_graph_repo") as mock_get_graph:
        mock_repo = MagicMock()
        mock_repo.get_stats.return_value = {"papers": 10, "authors": 5}
        mock_get_graph.return_value = mock_repo
        
        stats = get_stats()
        assert stats["papers"] == 10
        assert stats["authors"] == 5

def test_mcp_search_papers():
    with patch("src.mcp_server.get_graph_repo") as mock_get_graph:
        mock_repo = MagicMock()
        paper_mock = MagicMock()
        paper_mock.id = "paper_1"
        paper_mock.title = "Test Paper"
        paper_mock.year = 2025
        paper_mock.properties = {"source_type": "paper"}
        mock_repo.search_papers_by_title.return_value = [paper_mock]
        mock_get_graph.return_value = mock_repo
        
        results = search_papers("test")
        assert len(results) == 1
        assert results[0]["id"] == "paper_1"
        assert results[0]["title"] == "Test Paper"

def test_mcp_query_rag():
    with patch("src.mcp_server.get_rag_service") as mock_get_rag:
        mock_service = MagicMock()
        mock_service.ask.return_value = "Answer context"
        mock_get_rag.return_value = mock_service
        
        res = query_rag("question")
        assert res == "Answer context"
        mock_service.ask.assert_called_once_with("question", limit=5)
