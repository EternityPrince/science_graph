import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.review_agent import ReviewAgent
from tests.factories import create_paper, create_chunk

class TestReviewAgent:
    @pytest.fixture
    def mock_graph_repo(self):
        return MagicMock()

    @pytest.fixture
    def mock_vector_repo(self):
        return MagicMock()

    @pytest.fixture
    def mock_embedding_engine(self):
        engine = MagicMock()
        engine.get_embedding.return_value = [0.1] * 384
        return engine

    @pytest.fixture
    def mock_llm_engine(self):
        engine = MagicMock()
        engine.synthesize_section.return_value = "Synthesized Section Summary content"
        engine.cluster_chunks_by_topic.return_value = {
            "Introduction to Topic": ["chunk_1", "chunk_2"],
            "Methodology": ["chunk_3"]
        }
        return engine

    @pytest.fixture
    def review_agent(self, mock_graph_repo, mock_vector_repo, mock_embedding_engine, mock_llm_engine):
        return ReviewAgent(
            graph_repo=mock_graph_repo,
            vector_repo=mock_vector_repo,
            embedding_engine=mock_embedding_engine,
            llm_engine=mock_llm_engine
        )

    def test_run_no_chunks_found(self, review_agent, mock_vector_repo):
        mock_vector_repo.search_similar_chunks.return_value = []
        mock_vector_repo.search_text_fts5.return_value = []

        report = review_agent.run(topic="Quantum Mechanics", limit=5)

        assert "# Review: Quantum Mechanics" in report
        assert "*No relevant documents found in the index.*" in report

    def test_run_fast_mode(self, review_agent, mock_vector_repo, mock_graph_repo, mock_llm_engine):
        # Setup mock chunks
        chunk1 = create_chunk(id="chunk_1", paper_id="paper_1", text_content="Content of chunk 1")
        chunk2 = create_chunk(id="chunk_2", paper_id="paper_2", text_content="Content of chunk 2")
        mock_vector_repo.search_similar_chunks.return_value = [(chunk1, 0.9), (chunk2, 0.8)]
        mock_vector_repo.search_text_fts5.return_value = [(chunk2, 0.85)]

        # Setup mock papers
        paper1 = create_paper(id="paper_1", title="Paper One", authors=["Alice Smith", "Bob Jones"], year=2024, doi="10.1000/1")
        paper2 = create_paper(id="paper_2", title="Paper Two", authors=["Charlie Brown"], year=2025, doi="10.1000/2")
        mock_graph_repo.get_papers_batch.return_value = {
            "paper_1": paper1,
            "paper_2": paper2
        }

        # Mock neighbors for comparison table (no concepts for simplification)
        mock_graph_repo.get_neighbors.return_value = []

        report = review_agent.run(topic="Fast AI", limit=5, fast=True)

        assert "Literature Review: Fast AI" in report
        assert "## Overview" in report
        assert "Synthesized Section Summary content" in report
        assert "Paper One" in report
        assert "Charlie Brown" in report
        mock_llm_engine.synthesize_section.assert_called_once()
        assert not mock_llm_engine.cluster_chunks_by_topic.called

    def test_run_clustering_mode(self, review_agent, mock_vector_repo, mock_graph_repo, mock_llm_engine):
        # Setup mock chunks
        chunk1 = create_chunk(id="chunk_1", paper_id="paper_1", text_content="Content 1")
        chunk2 = create_chunk(id="chunk_2", paper_id="paper_1", text_content="Content 2")
        chunk3 = create_chunk(id="chunk_3", paper_id="paper_2", text_content="Content 3")
        mock_vector_repo.search_similar_chunks.return_value = [(chunk1, 0.9), (chunk2, 0.8), (chunk3, 0.7)]
        mock_vector_repo.search_text_fts5.return_value = []

        # Setup mock papers
        paper1 = create_paper(id="paper_1", title="Paper One", authors=["Alice Smith"], year=2024)
        paper2 = create_paper(id="paper_2", title="Paper Two", authors=["Bob Jones"], year=2025)
        mock_graph_repo.get_papers_batch.return_value = {
            "paper_1": paper1,
            "paper_2": paper2
        }

        # Return mock concepts for comparison table
        mock_graph_repo.get_neighbors.side_effect = lambda pid, max_depth: [
            ("paper_1", "Paper", "MENTIONS_CONCEPT", "concept_1", "Concept", {})
        ] if pid == "paper_1" else []
        mock_concept = MagicMock()
        mock_concept.name = "Deep Learning"
        mock_graph_repo.get_concept.return_value = mock_concept

        # Run
        report = review_agent.run(topic="Clustering AI", limit=5, fast=False)

        assert "## Introduction to Topic" in report
        assert "## Methodology" in report
        assert "Deep Learning" in report
        assert "Alice Smith" in report
        assert "Bob Jones" in report
        assert mock_llm_engine.synthesize_section.call_count == 2
        mock_llm_engine.cluster_chunks_by_topic.assert_called_once()

    def test_run_clustering_fallback(self, review_agent, mock_vector_repo, mock_graph_repo, mock_llm_engine):
        # Setup mock chunks
        chunk1 = create_chunk(id="chunk_1", paper_id="paper_1", text_content="Content 1")
        mock_vector_repo.search_similar_chunks.return_value = [(chunk1, 0.9)]
        mock_vector_repo.search_text_fts5.return_value = []

        mock_graph_repo.get_papers_batch.return_value = {"paper_1": create_paper(id="paper_1")}

        # Test fallback when clustering return None
        mock_llm_engine.cluster_chunks_by_topic.return_value = None

        report = review_agent.run(topic="Fallback Topic", limit=5, fast=False)
        assert "## Overview" in report
        assert "Synthesized Section Summary content" in report

        # Test fallback when clustering raises Exception or returns invalid type
        mock_llm_engine.cluster_chunks_by_topic.return_value = "invalid response type"
        report = review_agent.run(topic="Fallback Topic 2", limit=5, fast=False)
        assert "## Overview" in report

    def test_run_with_reranker(self, review_agent, mock_vector_repo, mock_graph_repo):
        # Setup mock chunks
        chunk1 = create_chunk(id="chunk_1", paper_id="paper_1", text_content="Content 1")
        chunk2 = create_chunk(id="chunk_2", paper_id="paper_2", text_content="Content 2")
        mock_vector_repo.search_similar_chunks.return_value = [(chunk1, 0.9), (chunk2, 0.8)]
        mock_vector_repo.search_text_fts5.return_value = []

        mock_graph_repo.get_papers_batch.return_value = {
            "paper_1": create_paper(id="paper_1"),
            "paper_2": create_paper(id="paper_2")
        }

        # Mock the reranker in _rag
        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = [0.95, 0.75]
        
        with patch.object(review_agent._rag, "_get_reranker", return_value=mock_reranker):
            report = review_agent.run(topic="Reranked Topic", limit=2, fast=True)
            assert "## Overview" in report
            mock_reranker.predict.assert_called_once_with([
                ("Reranked Topic", "Content 1"),
                ("Reranked Topic", "Content 2")
            ])

    def test_run_saving_to_file(self, review_agent, mock_vector_repo, mock_graph_repo):
        chunk1 = create_chunk(id="chunk_1", paper_id="paper_1", text_content="Content 1")
        mock_vector_repo.search_similar_chunks.return_value = [(chunk1, 0.9)]
        mock_vector_repo.search_text_fts5.return_value = []
        mock_graph_repo.get_papers_batch.return_value = {"paper_1": create_paper(id="paper_1")}

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "subfolder" / "report.md"
            report = review_agent.run(topic="Save Topic", limit=1, fast=True, output_path=out_file)
            
            assert out_file.exists()
            assert out_file.read_text(encoding="utf-8") == report
