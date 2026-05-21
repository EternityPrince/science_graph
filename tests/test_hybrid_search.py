import unittest
from unittest.mock import MagicMock, patch
from src.rag import RAGPipeline
from src.models import Chunk, Paper

class TestHybridSearch(unittest.TestCase):
    def setUp(self):
        self.graph_repo = MagicMock()
        self.vector_repo = MagicMock()
        self.emb_engine = MagicMock()
        self.llm_engine = MagicMock()
        
        # Build RAG Pipeline
        self.rag = RAGPipeline(
            graph_repo=self.graph_repo,
            vector_repo=self.vector_repo,
            embedding_engine=self.emb_engine,
            llm_engine=self.llm_engine
        )

    @patch("sentence_transformers.CrossEncoder")
    def test_hybrid_search_scoring_and_reranking(self, mock_cross_encoder_class):
        # 1. Setup mock chunks
        c1 = Chunk(id="chunk_1", paper_id="paper_1", text_content="Attention mechanisms are widely used in transformer neural networks.", page_number=1)
        c2 = Chunk(id="chunk_2", paper_id="paper_1", text_content="Reinforcement learning relies on reward functions to train policy networks.", page_number=2)
        c3 = Chunk(id="chunk_3", paper_id="paper_2", text_content="Contrastive learning is a self-supervised learning technique.", page_number=1)
        
        # 2. Setup mock dense and FTS5 search results
        self.vector_repo.search_similar_chunks.return_value = [
            (c2, 0.9),
            (c1, 0.7)
        ]
        self.vector_repo.search_text_fts5.return_value = [
            (c1, 1.2),
            (c2, 0.4)
        ]
        
        # 3. Setup mock Cross-Encoder
        mock_reranker = MagicMock()
        def mock_predict(pairs):
            return [1.5 if "Attention" in doc else 0.5 for query, doc in pairs]
        mock_reranker.predict.side_effect = mock_predict
        self.rag.service._reranker = mock_reranker
        
        # 4. Mock embedding
        self.emb_engine.get_embedding.return_value = [0.1] * 384
        
        # 5. Mock build_context
        self.rag.service.build_context = MagicMock(return_value=("context_text", "context_graph"))
        
        # 6. Call ask
        self.rag.ask("attention transformer reinforcement", limit=2)
        
        # 7. Assertions
        # Reranker predict should be called with query and candidates
        mock_reranker.predict.assert_called_once()
        self.rag.service.build_context.assert_called_once()
        
        # Verify the chunks passed to build_context are ranked by Cross-Encoder
        # Candidates were c1 (BM25 top and dense) and c2 (dense).
        # Cross encoder score for c1 was 1.5, for c2 was 0.5.
        # So final chunks should have c1 first, then c2.
        called_args = self.rag.service.build_context.call_args[0][0]
        self.assertEqual(called_args[0][0].id, "chunk_1")
        self.assertEqual(called_args[1][0].id, "chunk_2")
