import os
from unittest.mock import patch
from src.config import config
from src.llm_engine import LLMEngine

def test_config_flat_fallback():
    """Test that Config class correctly falls back to flat configuration values if nested sections are absent."""
    original_data = config.data
    try:
        # Simulate flat config
        config.data = {
            "llm": {
                "provider": "mlx",
                "model_path": "/flat/path/to/model",
                "api_key": "flat_api_key",
                "base_url": "flat_base_url",
            }
        }
        
        # Verify fallbacks
        assert config.llm_local_model_path == "/flat/path/to/model"
        assert config.llm_cloud_model_name == "/flat/path/to/model"
        assert config.llm_cloud_api_key == "flat_api_key"
        assert config.llm_cloud_base_url == "flat_base_url"
        
        # Aliases
        assert config.llm_model_path == "/flat/path/to/model"
        assert config.llm_api_key == "flat_api_key"
        assert config.llm_base_url == "flat_base_url"
    finally:
        config.data = original_data

def test_config_nested_values():
    """Test that Config class correctly resolves to nested values when they are present, ignoring the flat values."""
    original_data = config.data
    try:
        # Simulate nested config
        config.data = {
            "llm": {
                "provider": "openai",
                "model_path": "/flat/path/to/model",  # Old flat path
                "api_key": "flat_api_key",
                "base_url": "flat_base_url",
                "local": {
                    "model_path": "/nested/local/path",
                },
                "cloud": {
                    "provider": "openai",
                    "model_name": "nested/cloud/model",
                    "api_key": "nested_api_key",
                    "base_url": "nested_base_url",
                }
            }
        }
        
        # Verify nested parameters are prioritized
        assert config.llm_local_model_path == "/nested/local/path"
        assert config.llm_cloud_model_name == "nested/cloud/model"
        assert config.llm_cloud_api_key == "nested_api_key"
        assert config.llm_cloud_base_url == "nested_base_url"
        
        # Aliases
        assert config.llm_model_path == "/nested/local/path"
        assert config.llm_api_key == "nested_api_key"
        assert config.llm_base_url == "nested_base_url"
    finally:
        config.data = original_data

@patch("src.llm_engine.OpenAILLMEngine")
@patch("src.llm_engine.MlxLLMEngine")
def test_llm_engine_factory_selection(mock_mlx_cls, mock_openai_cls):
    """Test that LLMEngine factory selects the correct engine based on provider / parameters."""
    original_data = config.data
    original_env = os.environ.get("SCIENCE_GRAPH_USE_CLOUD")
    
    try:
        # Clear singletons first to force instantiation
        import src.llm_engine
        src.llm_engine._local_engine_singleton = None
        src.llm_engine._cloud_engine_singleton = None
        
        # Scenario 1: provider is mlx -> Local MLX Engine
        config.data = {"llm": {"provider": "mlx", "local": {"model_path": "/some/path"}}}
        if "SCIENCE_GRAPH_USE_CLOUD" in os.environ:
            del os.environ["SCIENCE_GRAPH_USE_CLOUD"]
            
        with patch("os.path.isdir", return_value=True):
            engine = LLMEngine(use_cloud=False)
            mock_mlx_cls.assert_called_once()
            mock_openai_cls.assert_not_called()
            
        # Reset singletons
        src.llm_engine._local_engine_singleton = None
        src.llm_engine._cloud_engine_singleton = None
        mock_mlx_cls.reset_mock()
        mock_openai_cls.reset_mock()
        
        # Scenario 2: provider is openai -> Cloud OpenAI Engine
        config.data = {
            "llm": {
                "provider": "openai",
                "cloud": {
                    "provider": "openai",
                    "model_name": "gpt-4",
                    "api_key": "test-key",
                    "base_url": "test-url"
                }
            }
        }
        
        engine = LLMEngine()
        mock_openai_cls.assert_called_once()
        mock_mlx_cls.assert_not_called()
        
        # Reset singletons
        src.llm_engine._local_engine_singleton = None
        src.llm_engine._cloud_engine_singleton = None
        mock_mlx_cls.reset_mock()
        mock_openai_cls.reset_mock()
        
        # Scenario 3: provider is mlx, but use_cloud=True is passed -> Cloud OpenAI Engine
        config.data = {
            "llm": {
                "provider": "mlx",
                "cloud": {
                    "provider": "openai",
                    "model_name": "gpt-4",
                    "api_key": "test-key",
                    "base_url": "test-url"
                }
            }
        }
        
        engine = LLMEngine(use_cloud=True)
        mock_openai_cls.assert_called_once()
        mock_mlx_cls.assert_not_called()
        
    finally:
        config.data = original_data
        if original_env is not None:
            os.environ["SCIENCE_GRAPH_USE_CLOUD"] = original_env
        elif "SCIENCE_GRAPH_USE_CLOUD" in os.environ:
            del os.environ["SCIENCE_GRAPH_USE_CLOUD"]
