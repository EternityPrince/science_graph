"""
Unit tests for Indexing Orchestrator.
"""

import unittest
from unittest.mock import MagicMock, patch
import os
from src.services.indexing_orchestrator import run_batch_index

class TestIndexingOrchestrator(unittest.TestCase):
    def setUp(self):
        self.original_use_cloud = os.environ.get("SCIENCE_GRAPH_USE_CLOUD")
        
    def tearDown(self):
        if self.original_use_cloud is not None:
            os.environ["SCIENCE_GRAPH_USE_CLOUD"] = self.original_use_cloud
        elif "SCIENCE_GRAPH_USE_CLOUD" in os.environ:
            del os.environ["SCIENCE_GRAPH_USE_CLOUD"]

    @patch("src.services.indexing_orchestrator.container")
    @patch("src.services.indexing_orchestrator.Indexer")
    def test_run_batch_index_basic(self, mock_indexer_class, mock_container):
        # Set up container mocks
        mock_graph_repo = MagicMock()
        mock_vector_repo = MagicMock()
        mock_embedding_engine = MagicMock()
        
        mock_container.get_graph_repo.return_value = mock_graph_repo
        mock_container.get_vector_repo.return_value = mock_vector_repo
        mock_container.get_embedding_engine.return_value = mock_embedding_engine
        
        # Set up Indexer mock
        mock_indexer_instance = MagicMock()
        mock_indexer_instance.index_batch.return_value = [{"name": "target1", "success": True}]
        mock_indexer_class.return_value = mock_indexer_instance
        
        # Call the orchestrator
        results = run_batch_index(
            target="target1, target2; target3",
            use_llm=False,
            trace=False,
            cloud=False,
            chunk_pool_size=10
        )
        
        # Verify container calls
        mock_container.get_graph_repo.assert_called_once()
        mock_container.get_vector_repo.assert_called_once()
        mock_container.get_embedding_engine.assert_called_once()
        mock_container.get_llm_engine.assert_not_called()
        
        # Verify Indexer construction
        mock_indexer_class.assert_called_once_with(
            mock_graph_repo, mock_vector_repo, mock_embedding_engine, None
        )
        
        # Verify index_batch call
        mock_indexer_instance.index_batch.assert_called_once_with(
            targets=["target1", "target2", "target3"],
            use_llm=False,
            trace=False,
            chunk_pool_size=10
        )
        
        self.assertEqual(results, [{"name": "target1", "success": True}])

    @patch("src.services.indexing_orchestrator.container")
    @patch("src.services.indexing_orchestrator.Indexer")
    def test_run_batch_index_with_llm_success(self, mock_indexer_class, mock_container):
        mock_llm_engine = MagicMock()
        mock_container.get_llm_engine.return_value = mock_llm_engine
        
        mock_indexer_instance = MagicMock()
        mock_indexer_class.return_value = mock_indexer_instance
        
        run_batch_index(
            target="target1",
            use_llm=True,
            trace=False,
            cloud=True,
            chunk_pool_size=None
        )
        
        mock_container.get_llm_engine.assert_called_once_with(use_cloud=True)
        mock_indexer_class.assert_called_once_with(
            mock_container.get_graph_repo(),
            mock_container.get_vector_repo(),
            mock_container.get_embedding_engine(),
            mock_llm_engine
        )
        mock_indexer_instance.index_batch.assert_called_once_with(
            targets=["target1"],
            use_llm=True,
            trace=False,
            chunk_pool_size=None
        )
        self.assertEqual(os.environ.get("SCIENCE_GRAPH_USE_CLOUD"), "1")

    @patch("src.services.indexing_orchestrator.con")
    @patch("src.services.indexing_orchestrator.container")
    @patch("src.services.indexing_orchestrator.Indexer")
    def test_run_batch_index_with_llm_failure(self, mock_indexer_class, mock_container, mock_con):
        mock_container.get_llm_engine.side_effect = Exception("LLM Load Error")
        
        mock_indexer_instance = MagicMock()
        mock_indexer_class.return_value = mock_indexer_instance
        
        run_batch_index(
            target="target1",
            use_llm=True,
            trace=False,
            cloud=False
        )
        
        # Should call container get_llm_engine and fail
        mock_container.get_llm_engine.assert_called_once_with(use_cloud=False)
        # Should warn in con
        mock_con.warning.assert_any_call("Could not load LLM engine: LLM Load Error")
        mock_con.warning.assert_any_call("Proceeding with regex fallback extraction because LLM engine failed to load.")
        # Indexer should be created with None for llm_engine
        mock_indexer_class.assert_called_once_with(
            mock_container.get_graph_repo(),
            mock_container.get_vector_repo(),
            mock_container.get_embedding_engine(),
            None
        )

    @patch("src.services.indexing_orchestrator.con")
    @patch("src.services.indexing_orchestrator.container")
    @patch("src.services.indexing_orchestrator.Indexer")
    def test_run_batch_index_trace(self, mock_indexer_class, mock_container, mock_con):
        mock_con.SHOW_TIME = False
        run_batch_index(
            target="target1",
            use_llm=False,
            trace=True,
            cloud=False
        )
        self.assertTrue(mock_con.SHOW_TIME)

    def test_run_batch_index_empty_target(self):
        with self.assertRaises(ValueError) as context:
            run_batch_index(
                target=" , ; ",
                use_llm=False,
                trace=False,
                cloud=False
            )
        self.assertEqual(str(context.exception), "No targets provided to index.")
