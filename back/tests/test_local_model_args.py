import os
import unittest
from src.config import config

class TestLocalModelArgsLoad(unittest.TestCase):
    def test_local_model_load_without_model_args_error(self):
        """
        Runs only if the configured local model path actually exists on this device.
        Verifies that MlxLLMEngine can load the model without encountering the ModelArgs.__init__() TypeError.
        """
        # Reload the actual user configuration from disk, bypassing the reset_config fixture
        from src.config import config, Config
        real_config = Config()
        model_path = real_config.llm_local_model_path
        
        if not model_path or not os.path.isdir(model_path):
            self.skipTest(f"Configured local model path '{model_path}' does not exist on this device. Skipping validation.")
        
        if os.path.exists(os.path.join(model_path, "optiq_metadata.json")):
            self.skipTest(f"Configured model path '{model_path}' is an OptiQ model, which is not supported by standard MLX engine. Skipping.")
            
        orig_data = config.data
        config.data = real_config.data
        try:
            from src.llm_engine.mlx_impl import MlxLLMEngine
            engine = MlxLLMEngine(model_path=model_path)
            
            # Attempt to load the model
            engine._ensure_model_loaded()
            
            # Verify successful load
            self.assertIsNotNone(engine.model, "Model should be successfully loaded")
            self.assertIsNotNone(engine.tokenizer, "Tokenizer should be successfully loaded")
        except ImportError as e:
            if "MLX is not installed" in str(e):
                self.skipTest("MLX is not installed in the current environment.")
            raise e
        except TypeError as e:
            err_str = str(e)
            if "ModelArgs.__init__() missing" in err_str:
                self.fail(f"Model failed to load due to ModelArgs configuration mismatch: {e}")
            raise e
        finally:
            config.data = orig_data
