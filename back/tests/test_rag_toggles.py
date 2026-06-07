import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import os
import json

from src.models import Chunk, Paper, Author, Concept
from src.services.rag_service import RAGService
from src.config import config

class TestRAGToggles(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.graph_repo = MagicMock()
        self.vector_repo = MagicMock()
        self.emb_engine = MagicMock()
        self.llm_engine = MagicMock()
        self.expander = MagicMock()
        
        self.service = RAGService(
            self.graph_repo,
            self.vector_repo,
            self.emb_engine,
            self.llm_engine,
            self.expander
        )
        
        # Reset environment overrides
        for key in list(os.environ.keys()):
            if key.startswith("RAG_"):
                del os.environ[key]

    def test_environment_override(self):
        # Default behavior: components should be enabled
        self.assertTrue(config.is_component_enabled("hyde"))
        
        # Override with env var
        os.environ["RAG_HYDE"] = "false"
        self.assertFalse(config.is_component_enabled("hyde"))
        
        os.environ["RAG_HYDE"] = "0"
        self.assertFalse(config.is_component_enabled("hyde"))
        
        os.environ["RAG_HYDE"] = "true"
        self.assertTrue(config.is_component_enabled("hyde"))
        
        os.environ["RAG_HYDE"] = "1"
        self.assertTrue(config.is_component_enabled("hyde"))

    @patch.dict(os.environ, {"RAG_INTENT_CLASSIFIER": "false"})
    def test_intent_classifier_disabled(self):
        self.vector_repo.search_similar_chunks.return_value = []
        self.vector_repo.search_text_fts5.return_value = []
        
        # Calling retrieve_relevant_chunks should not invoke _classify_intent_and_extract_filters
        with patch.object(self.service, "_classify_intent_and_extract_filters") as mock_classify:
            self.service.retrieve_relevant_chunks("test query")
            mock_classify.assert_not_called()

    @patch.dict(os.environ, {"RAG_LEXICAL_SEARCH": "false"})
    def test_lexical_search_disabled(self):
        self.vector_repo.search_similar_chunks.return_value = []
        self.vector_repo.search_text_fts5.return_value = [
            (MagicMock(spec=Chunk, id="c1"), 0.9)
        ]
        
        # When lexical search is disabled, fts5 results should be ignored
        res = self.service.retrieve_relevant_chunks("test query")
        self.assertEqual(res, [])
        self.vector_repo.search_text_fts5.assert_not_called()

    @patch.dict(os.environ, {"RAG_DENSE_SEARCH": "false"})
    def test_dense_search_disabled(self):
        chunk = Chunk(id="c2", paper_id="p2", text_content="content", page_number=1)
        self.vector_repo.search_similar_chunks.return_value = [
            (chunk, 0.9)
        ]
        self.vector_repo.search_text_fts5.return_value = []
        
        # When dense search is disabled, similar chunks should not be searched
        res = self.service.retrieve_relevant_chunks("test query")
        self.assertEqual(res, [])
        self.vector_repo.search_similar_chunks.assert_not_called()

    @patch.dict(os.environ, {"RAG_HYDE": "false"})
    @patch("src.config.Config.hyde_enabled", new_callable=PropertyMock)
    def test_hyde_disabled(self, mock_hyde_enabled):
        mock_hyde_enabled.return_value = True
        self.vector_repo.search_similar_chunks.return_value = []
        self.vector_repo.search_text_fts5.return_value = []
        
        # Calling retrieve_relevant_chunks should not invoke LLM generate_response for HyDE
        self.service.retrieve_relevant_chunks("test query")
        self.llm_engine.generate_response.assert_not_called()

    @patch.dict(os.environ, {"RAG_RERANKER": "false"})
    def test_reranker_disabled(self):
        chunk = Chunk(id="c1", paper_id="p1", text_content="content", page_number=1)
        self.vector_repo.search_similar_chunks.return_value = [(chunk, 0.8)]
        self.vector_repo.search_text_fts5.return_value = []
        
        with patch.object(self.service, "_get_reranker") as mock_get_reranker:
            res = self.service.retrieve_relevant_chunks("test query", limit=1)
            mock_get_reranker.assert_not_called()
            self.assertEqual(len(res), 1)
            self.assertEqual(res[0][0].id, "c1")

    @patch.dict(os.environ, {"RAG_GRAPH_EXPANSION": "false"})
    def test_graph_expansion_disabled(self):
        chunk = Chunk(id="c1", paper_id="p1", text_content="content", page_number=1)
        self.graph_repo.get_papers_batch.return_value = {"p1": MagicMock()}
        
        # build_context should return "Graph enrichment disabled."
        context_text, context_graph = self.service.build_context([(chunk, 0.8)])
        self.assertEqual(context_graph, "Graph enrichment disabled.")

    @patch.dict(os.environ, {"RAG_CITATION_REPAIR": "false"})
    def test_citation_repair_disabled(self):
        chunk = Chunk(id="c1", paper_id="p1", text_content="content", page_number=1)
        self.vector_repo.search_similar_chunks.return_value = [(chunk, 0.8)]
        self.vector_repo.search_text_fts5.return_value = []
        
        self.graph_repo.get_papers_batch.return_value = {"p1": MagicMock()}
        self.llm_engine.generate_response.return_value = "This is a response [99]." # Hallucinated citation [99]
        
        # When citation repair is disabled, hallucinated citation [99] should NOT be stripped/repaired
        with patch.object(self.service, "_validate_and_repair_citations") as mock_repair:
            res = self.service.ask("test query")
            mock_repair.assert_not_called()
            self.assertEqual(res, "This is a response [99].")
