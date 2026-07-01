import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock marker module and its submodules to prevent loading heavy dependencies
sys.modules["marker"] = MagicMock()
sys.modules["marker.convert"] = MagicMock()
sys.modules["marker.models"] = MagicMock()

from src.services.container import ServiceContainer
from src.llm_engine.factory import LLMEngine
from src.parsers.marker_parser import marker_session, shutdown_marker
import src.parsers.marker_parser as marker_module


class TestNewSplitModelsAndSessions(unittest.TestCase):
    def setUp(self):
        # Reset marker session state
        marker_module._marker_session_depth = 0
        marker_module._marker_models = None

    def test_marker_session_reference_counting(self):
        """Test that marker_session manages reference counting and calls shutdown_marker exactly when depth reaches 0."""
        # Setup mock models to simulate they are loaded
        marker_module._marker_models = ["mock_model"]

        with patch("src.parsers.marker_parser.shutdown_marker") as mock_shutdown:
            # Outermost session
            with marker_session():
                self.assertEqual(marker_module._marker_session_depth, 1)
                mock_shutdown.assert_not_called()

                # Nested session
                with marker_session():
                    self.assertEqual(marker_module._marker_session_depth, 2)
                    mock_shutdown.assert_not_called()

                # Exit nested session
                self.assertEqual(marker_module._marker_session_depth, 1)
                mock_shutdown.assert_not_called()

            # Exit outermost session
            self.assertEqual(marker_module._marker_session_depth, 0)
            mock_shutdown.assert_called_once()

    def test_shutdown_marker_unloads_and_clears_cache(self):
        """Test that shutdown_marker unloads models, runs garbage collection, and clears GPU cache."""
        marker_module._marker_models = ["mock_model"]

        # Patch gc and torch cache empty functions
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.mps.is_available.return_value = True
        sys.modules["torch"] = mock_torch

        with patch("gc.collect") as mock_gc:
            shutdown_marker()
            self.assertIsNone(marker_module._marker_models)
            mock_gc.assert_called_once()
            mock_torch.cuda.empty_cache.assert_called_once()
            mock_torch.mps.empty_cache.assert_called_once()

    @patch("src.config.Config.llm_local_model_path", "indexing_local_model_path")
    @patch("src.config.Config.llm_local_rag_model_path", "rag_local_model_path")
    @patch("src.config.Config.llm_cloud_model_name", "indexing_cloud_model")
    @patch("src.config.Config.llm_cloud_rag_model_name", "rag_cloud_model")
    @patch("src.llm_engine.factory.config")
    def test_split_models_loading_factory(self, mock_cfg):
        # Configure providers/paths on mock config
        mock_cfg.llm_provider = "mlx"
        mock_cfg.llm_local_model_path = "indexing_local_model_path"
        mock_cfg.llm_local_rag_model_path = "rag_local_model_path"
        mock_cfg.llm_cloud_model_name = "indexing_cloud_model"
        mock_cfg.llm_cloud_rag_model_name = "rag_cloud_model"

        # Mock the actual implementation classes
        import src.llm_engine
        src.llm_engine._local_engine_singleton = None
        src.llm_engine._cloud_engine_singleton = None
        src.llm_engine._local_rag_engine_singleton = None
        src.llm_engine._cloud_rag_engine_singleton = None

        with patch("src.llm_engine.MlxLLMEngine") as mock_mlx, \
             patch("src.llm_engine.OpenAILLMEngine") as mock_openai:
            
            # 1. Test local index model
            LLMEngine(use_cloud=False, purpose="index")
            mock_mlx.assert_any_call(model_path="indexing_local_model_path")

            # 2. Test local RAG model
            LLMEngine(use_cloud=False, purpose="rag")
            mock_mlx.assert_any_call(model_path="rag_local_model_path")

            # 3. Test cloud index model
            LLMEngine(use_cloud=True, purpose="index")
            mock_openai.assert_any_call(model_name="indexing_cloud_model")

            # 4. Test cloud RAG model
            LLMEngine(use_cloud=True, purpose="rag")
            mock_openai.assert_any_call(model_name="rag_cloud_model")

    def test_container_exposes_correct_engines(self):
        """Test that ServiceContainer returns the separate models for indexing and RAG."""
        container = ServiceContainer()

        mock_index_engine = MagicMock()
        mock_rag_engine = MagicMock()

        def mock_llm_factory(use_cloud=False, purpose="index", *args, **kwargs):
            if purpose == "rag":
                return mock_rag_engine
            return mock_index_engine

        with patch("src.services.container.LLMEngine", side_effect=mock_llm_factory):
            # Test getting index LLM engine
            index_engine = container.get_llm_engine(use_cloud=False, purpose="index")
            self.assertIs(index_engine, mock_index_engine)

            # Test getting RAG LLM engine
            rag_engine = container.get_llm_engine(use_cloud=False, purpose="rag")
            self.assertIs(rag_engine, mock_rag_engine)
            self.assertIsNot(index_engine, rag_engine)
