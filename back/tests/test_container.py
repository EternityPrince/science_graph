"""
Unit tests for ServiceContainer.
"""

import unittest
from unittest.mock import MagicMock, patch
from src.services.container import ServiceContainer

class TestServiceContainer(unittest.TestCase):
    @patch("src.services.container.SQLiteGraphRepository")
    @patch("src.services.container.SQLiteVectorRepository")
    @patch("src.services.container.EmbeddingEngine")
    @patch("src.services.container.LLMEngine")
    @patch("src.services.container.RAGService")
    @patch("src.services.container.NoteService")
    @patch("src.services.container.config")
    def test_container_lazy_loading_and_caching(
        self,
        mock_config,
        mock_note_service_cls,
        mock_rag_service_cls,
        mock_llm_engine_cls,
        mock_embedding_engine_cls,
        mock_sqlite_vector_repo_cls,
        mock_sqlite_graph_repo_cls
    ):
        mock_config.db_path = "dummy_db_path"
        mock_llm_engine_cls.side_effect = lambda *args, **kwargs: MagicMock()
        mock_rag_service_cls.side_effect = lambda *args, **kwargs: MagicMock()
        
        container = ServiceContainer()
        
        # 1. Test get_graph_repo
        repo1 = container.get_graph_repo()
        repo2 = container.get_graph_repo()
        mock_sqlite_graph_repo_cls.assert_called_once_with("dummy_db_path")
        self.assertIs(repo1, repo2)
        
        # 2. Test get_vector_repo
        vec1 = container.get_vector_repo()
        vec2 = container.get_vector_repo()
        mock_sqlite_vector_repo_cls.assert_called_once_with("dummy_db_path")
        self.assertIs(vec1, vec2)
        
        # 3. Test get_embedding_engine
        emb1 = container.get_embedding_engine()
        emb2 = container.get_embedding_engine()
        mock_embedding_engine_cls.assert_called_once()
        self.assertIs(emb1, emb2)
        
        # 4. Test get_llm_engine (local vs cloud)
        llm_local1 = container.get_llm_engine(use_cloud=False)
        llm_local2 = container.get_llm_engine(use_cloud=False)
        llm_cloud1 = container.get_llm_engine(use_cloud=True)
        llm_cloud2 = container.get_llm_engine(use_cloud=True)
        
        mock_llm_engine_cls.assert_any_call(use_cloud=False)
        mock_llm_engine_cls.assert_any_call(use_cloud=True)
        self.assertIs(llm_local1, llm_local2)
        self.assertIs(llm_cloud1, llm_cloud2)
        self.assertIsNot(llm_local1, llm_cloud1)
        
        # 5. Test get_rag_service (local vs cloud)
        rag_local1 = container.get_rag_service(use_cloud=False)
        rag_local2 = container.get_rag_service(use_cloud=False)
        rag_cloud1 = container.get_rag_service(use_cloud=True)
        rag_cloud2 = container.get_rag_service(use_cloud=True)
        
        self.assertIs(rag_local1, rag_local2)
        self.assertIs(rag_cloud1, rag_cloud2)
        self.assertIsNot(rag_local1, rag_cloud1)
        
        # 6. Test get_note_service
        note1 = container.get_note_service()
        note2 = container.get_note_service()
        mock_note_service_cls.assert_called_once()
        self.assertIs(note1, note2)
