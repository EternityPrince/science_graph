import unittest
from unittest.mock import MagicMock, patch
import src.llm_engine
from src.config import config
from src.llm_engine.openai_impl import OpenAILLMEngine
from src.llm_engine.factory import LLMEngine

class TestOpenAIMTPIntegration(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.orig_singletons = (
            src.llm_engine._local_engine_singleton,
            src.llm_engine._cloud_engine_singleton,
            src.llm_engine._local_rag_engine_singleton,
            src.llm_engine._cloud_rag_engine_singleton,
        )
        src.llm_engine._local_engine_singleton = None
        src.llm_engine._cloud_engine_singleton = None
        src.llm_engine._local_rag_engine_singleton = None
        src.llm_engine._cloud_rag_engine_singleton = None

        self.orig_data = config.data
        config.data = {
            "llm": {
                "provider": "openai-compatible",
                "base_url": "http://localhost:8000/v1",
                "model_name": "Qwen3.5-9B-OptiQ-4bit",
                "model_path": "/Users/vladimirkasterin/models/llm/Qwen3.5-9B-OptiQ-4bit",
                "enable_mtp": True,
                "auto_disable_mtp_if_missing_files": True,
                "max_tokens": 1000,
                "temp": 0.1,
                "request_delay": 0.0,
                "retry_backoff": 0.01,
            }
        }

    def tearDown(self):
        src.llm_engine._local_engine_singleton = self.orig_singletons[0]
        src.llm_engine._cloud_engine_singleton = self.orig_singletons[1]
        src.llm_engine._local_rag_engine_singleton = self.orig_singletons[2]
        src.llm_engine._cloud_rag_engine_singleton = self.orig_singletons[3]
        config.data = self.orig_data

    @patch("os.path.exists")
    @patch("openai.OpenAI")
    @patch("src.llm_engine.openai_impl.con")
    def test_mtp_enabled(self, mock_con, mock_openai, mock_exists):
        # 1. enable_mtp = true + mtp.safetensors exists -> MTP enabled.
        mock_exists.side_effect = lambda path: path.endswith("mtp.safetensors")
        
        self.assertTrue(config.llm_enable_mtp)
        self.assertTrue(config.llm_mtp_file_found)
        self.assertTrue(config.llm_effective_mtp_mode)
        
        # Verify launch command
        cmd = config.llm_expected_launch_command
        self.assertEqual(cmd, "optiq serve --model /Users/vladimirkasterin/models/llm/Qwen3.5-9B-OptiQ-4bit --mtp")

        # Initialize engine and check logs
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            OpenAILLMEngine()
            
            # Check warning was NOT logged
            mock_logger.warning.assert_not_called()
            mock_con.warning.assert_not_called()
            
            # Check info logging
            info_calls = [c[0][0] for c in mock_logger.info.call_args_list]
            self.assertTrue(any("MTP requested: true" in call for call in info_calls))
            self.assertTrue(any("MTP file found: true" in call for call in info_calls))
            self.assertTrue(any("MTP effective mode: enabled" in call for call in info_calls))

    @patch("os.listdir")
    @patch("os.path.isdir")
    @patch("os.path.exists")
    @patch("openai.OpenAI")
    @patch("src.llm_engine.openai_impl.con")
    def test_mtp_disabled_missing_file(self, mock_con, mock_openai, mock_exists, mock_isdir, mock_listdir):
        # 2. enable_mtp = true + mtp.safetensors is missing -> MTP disabled + warning logged.
        mock_exists.return_value = False
        mock_isdir.return_value = False
        mock_listdir.return_value = []
        
        self.assertTrue(config.llm_enable_mtp)
        self.assertFalse(config.llm_mtp_file_found)
        self.assertFalse(config.llm_effective_mtp_mode)
        
        # Verify launch command doesn't have --mtp
        cmd = config.llm_expected_launch_command
        self.assertEqual(cmd, "optiq serve --model /Users/vladimirkasterin/models/llm/Qwen3.5-9B-OptiQ-4bit")

        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            OpenAILLMEngine()
            
            # Check warning was logged
            mock_logger.warning.assert_called_once()
            mock_con.warning.assert_called_once()
            
            # Check info logs show disabled
            info_calls = [c[0][0] for c in mock_logger.info.call_args_list]
            self.assertTrue(any("MTP requested: true" in call for call in info_calls))
            self.assertTrue(any("MTP file found: false" in call for call in info_calls))
            self.assertTrue(any("MTP effective mode: disabled" in call for call in info_calls))

    @patch("os.path.exists")
    @patch("openai.OpenAI")
    @patch("src.llm_engine.openai_impl.con")
    def test_mtp_disabled_requested_false(self, mock_con, mock_openai, mock_exists):
        # 3. enable_mtp = false + mtp.safetensors exists -> MTP disabled.
        config.data["llm"]["enable_mtp"] = False
        mock_exists.side_effect = lambda path: path.endswith("mtp.safetensors")
        
        self.assertFalse(config.llm_enable_mtp)
        self.assertTrue(config.llm_mtp_file_found)
        self.assertFalse(config.llm_effective_mtp_mode)
        
        # Verify launch command doesn't have --mtp
        cmd = config.llm_expected_launch_command
        self.assertEqual(cmd, "optiq serve --model /Users/vladimirkasterin/models/llm/Qwen3.5-9B-OptiQ-4bit")

        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            OpenAILLMEngine()
            
            mock_logger.warning.assert_not_called()
            mock_con.warning.assert_not_called()

            info_calls = [c[0][0] for c in mock_logger.info.call_args_list]
            self.assertTrue(any("MTP requested: false" in call for call in info_calls))
            self.assertTrue(any("MTP file found: true" in call for call in info_calls))
            self.assertTrue(any("MTP effective mode: disabled" in call for call in info_calls))

    @patch("openai.OpenAI")
    def test_openai_request_payload_unchanged(self, mock_openai):
        # 6. OpenAI request payload not changing due to MTP, and 7. One-shot request is sent
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "Result response"
        mock_client.chat.completions.create.return_value = mock_completion

        engine = OpenAILLMEngine()
        res = engine.generate_response("hello prompt")
        
        self.assertEqual(res, "Result response")
        # Ensure we didn't inject anything extra/special into payload
        mock_client.chat.completions.create.assert_called_once_with(
            model="Qwen3.5-9B-OptiQ-4bit",
            messages=[{"role": "user", "content": "hello prompt"}],
            max_tokens=1000,
            temperature=0.1
        )

    @patch("openai.OpenAI")
    async def test_openai_request_payload_unchanged_async(self, mock_openai):
        # Async version
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "Result response async"
        mock_client.chat.completions.create.return_value = mock_completion

        engine = OpenAILLMEngine()
        res = await engine.generate_response_async("hello prompt async")
        
        self.assertEqual(res, "Result response async")
        mock_client.chat.completions.create.assert_called_once_with(
            model="Qwen3.5-9B-OptiQ-4bit",
            messages=[{"role": "user", "content": "hello prompt async"}],
            max_tokens=1000,
            temperature=0.1
        )

    @patch("openai.OpenAI")
    def test_factory_returns_openai_engine_for_compat_provider(self, mock_openai):
        # Ensure factory routes 'openai-compatible' to OpenAILLMEngine
        engine = LLMEngine()
        self.assertEqual(engine.__class__.__name__, "OpenAILLMEngine")

    @patch("openai.OpenAI")
    def test_token_limits_and_estimation_used(self, mock_openai):
        # 8. Token limits and tiktoken-based estimation continue to be used
        mock_tiktoken = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tiktoken.encoding_for_model.return_value = mock_tokenizer
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "hello"

        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "res"
        mock_client.chat.completions.create.return_value = mock_completion

        with patch.dict("sys.modules", {"tiktoken": mock_tiktoken}):
            engine = OpenAILLMEngine()
            engine._truncate_to_context("hello prompt", max_input_tokens=2)
            mock_tokenizer.encode.assert_called_once_with("hello prompt")
            mock_tokenizer.decode.assert_called_once_with([1, 2])
            
            engine.generate_response("prompt")
            mock_client.chat.completions.create.assert_called_with(
                model="Qwen3.5-9B-OptiQ-4bit",
                messages=[{"role": "user", "content": "prompt"}],
                max_tokens=config.llm_max_tokens,
                temperature=config.llm_temp
            )

    @patch("openai.OpenAI")
    def test_no_batch_or_concurrent_generation(self, mock_openai):
        # 9. MTP integration does not add batch or concurrent generation loops
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "response"
        mock_client.chat.completions.create.return_value = mock_completion

        engine = OpenAILLMEngine()
        engine.generate_response("hello")
        
        # Exactly 1 call is made - no batching, no speculative loops on client side
        mock_client.chat.completions.create.assert_called_once()

    @patch("os.path.exists")
    @patch("openai.OpenAI")
    @patch("src.services.extraction_service.con")
    def test_extraction_service_concurrency_with_mtp(self, mock_con, mock_openai, mock_exists):
        # Ensure MTP mode is effective
        mock_exists.side_effect = lambda path: path.endswith("mtp.safetensors")
        self.assertTrue(config.llm_effective_mtp_mode)

        from src.services.extraction_service import ExtractionService
        
        # 1. With default/unspecified chunk pool size, limit should be 1
        llm = OpenAILLMEngine()
        llm.use_cloud = False
        
        service = ExtractionService(llm_engine=llm)
        self.assertEqual(service.semaphore._value, 1)

        # 2. Even if chunk_pool_size is explicitly set to > 1, MTP mode should override it to 1 and print a warning
        service_custom = ExtractionService(llm_engine=llm, chunk_pool_size=5)
        self.assertEqual(service_custom.semaphore._value, 1)
        mock_con.warning.assert_called_once_with(
            "Speculative decoding (MTP) is active on local server. Overriding chunk pool concurrency to 1 to run sequentially."
        )
