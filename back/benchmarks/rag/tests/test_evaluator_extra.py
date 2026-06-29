import os
import sys
import yaml
import json
import pytest
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from core.evaluator import (
    CloudEvaluator,
    build_context_string,
    get_cloud_credentials,
    evaluate_baseline_case,
    load_checkpoint,
    save_checkpoint,
    run_evaluation,
    get_clean_judge_answer,
    _fallback_parse_reasoning_response,
    _fallback_clean_reasoning_text
)

def test_evaluator_import_no_rich():
    import importlib
    import core.evaluator
    with patch.dict(sys.modules, {"rich": None, "rich.table": None}):
        importlib.reload(core.evaluator)
        assert core.evaluator.HAS_RICH is False
    importlib.reload(core.evaluator)

def test_evaluator_openai_missing():
    with patch.dict(sys.modules, {"openai": None}):
        with pytest.raises(ImportError, match="Please install the 'openai' python package"):
            CloudEvaluator(api_key="key", base_url="url", model_name="model", concurrency=1, rpm=0, max_retries=1)

@pytest.mark.asyncio
async def test_call_llm_choice_message_content_none():
    with patch("openai.AsyncOpenAI") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        evaluator = CloudEvaluator(api_key="key", base_url="url", model_name="model", concurrency=1, rpm=0, max_retries=1)
        evaluator.client = mock_client
        
        # message content is None
        response_mock = MagicMock()
        choice_mock = MagicMock()
        choice_mock.message.content = None
        response_mock.choices = [choice_mock]
        mock_client.chat.completions.create = AsyncMock(return_value=response_mock)
        
        with pytest.raises(ValueError, match="Invalid API response: message content is None"):
            await evaluator.call_llm("sys", "user")

@pytest.mark.asyncio
@patch("core.evaluator.asyncio.sleep")
async def test_call_llm_rate_limit_value_error(mock_sleep):
    with patch("openai.AsyncOpenAI") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        evaluator = CloudEvaluator(api_key="key", base_url="url", model_name="model", concurrency=1, rpm=0, max_retries=2)
        evaluator.client = mock_client
        
        # Raise RateLimitError with invalid headers
        import openai
        response_mock = MagicMock()
        response_mock.headers = {"retry-after": "invalid_number", "x-ratelimit-reset": "invalid_number"}
        err = openai.RateLimitError("429 error", response=response_mock, body=None)
        
        mock_client.chat.completions.create = AsyncMock(side_effect=[err, MagicMock()])
        
        # Call should succeed on retry
        await evaluator.call_llm("sys", "user")
        mock_sleep.assert_called_once()

@pytest.mark.asyncio
@patch("core.evaluator.asyncio.sleep")
async def test_call_llm_rate_limit_header_millis(mock_sleep):
    with patch("openai.AsyncOpenAI") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        evaluator = CloudEvaluator(api_key="key", base_url="url", model_name="model", concurrency=1, rpm=0, max_retries=2)
        evaluator.client = mock_client
        
        # Reset is a millisecond timestamp
        import openai
        response_mock = MagicMock()
        response_mock.headers = {"x-ratelimit-reset": "150000000000000"} # far in future
        err = openai.RateLimitError("429 error", response=response_mock, body=None)
        
        mock_client.chat.completions.create = AsyncMock(side_effect=[err, MagicMock()])
        await evaluator.call_llm("sys", "user")
        mock_sleep.assert_called_once()

@pytest.mark.asyncio
async def test_evaluate_metric_json_parse_errors():
    with patch("openai.AsyncOpenAI") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        evaluator = CloudEvaluator(api_key="key", base_url="url", model_name="model", concurrency=1, rpm=0, max_retries=1)
        evaluator.client = mock_client
        
        # raw response does not contain score key
        evaluator.call_llm = AsyncMock(return_value='{"not_score": 1.0}')
        res = await evaluator.evaluate_metric({"system_prompt": "sys", "user_prompt_template": "user"}, "metric")
        assert res["score"] == 0.0
        assert "error" in res

@pytest.mark.asyncio
@patch("core.evaluator.asyncio.sleep")
async def test_evaluate_all_metrics_structure(mock_sleep):
    with patch("openai.AsyncOpenAI") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        evaluator = CloudEvaluator(api_key="key", base_url="url", model_name="model", concurrency=1, rpm=0, max_retries=1)
        evaluator.client = mock_client
        
        # Test case 1: missing required keys
        evaluator.call_llm = AsyncMock(return_value='{"answer_relevance": 1.0}') # missing semantic_accuracy
        res = await evaluator.evaluate_all_metrics({"system_prompt": "sys", "user_prompt_template": "user"}, has_context=False)
        assert res["answer_relevance"]["score"] == 0.0 # triggers fallback
        
        # Test case 2: metric_data is float/number (should convert to {"score": float})
        evaluator.call_llm = AsyncMock(return_value='{"answer_relevance": 0.8, "semantic_accuracy": 0.9}')
        res = await evaluator.evaluate_all_metrics({"system_prompt": "sys", "user_prompt_template": "user"}, has_context=False)
        assert res["answer_relevance"] == {"score": 0.8}
        assert res["semantic_accuracy"] == {"score": 0.9}
        
        # Test case 3: dict without 'score' key
        evaluator.call_llm = AsyncMock(return_value='{"answer_relevance": {"not_score": 1.0}, "semantic_accuracy": 1.0}')
        res = await evaluator.evaluate_all_metrics({"system_prompt": "sys", "user_prompt_template": "user"}, has_context=False)
        assert res["answer_relevance"]["score"] == 0.0
        
        # Test case 4: invalid datatype
        evaluator.call_llm = AsyncMock(return_value='{"answer_relevance": [], "semantic_accuracy": 1.0}')
        res = await evaluator.evaluate_all_metrics({"system_prompt": "sys", "user_prompt_template": "user"}, has_context=False)
        assert res["answer_relevance"]["score"] == 0.0

def test_clean_and_parse_json_markdown():
    res = CloudEvaluator.clean_and_parse_json(None, "some text ```json\n{\n  \"score\": 1.0\n}\n``` other text")
    assert res == {"score": 1.0}

def test_get_cloud_credentials_env():
    config = MagicMock()
    config.llm_cloud_api_key = None
    config.llm_cloud_base_url = None
    config.llm_cloud_model_name = None
    
    # Test exit when missing
    with patch.dict(os.environ, {}, clear=True), pytest.raises(SystemExit):
        get_cloud_credentials(config)
        
    # Test env override and base_url fallback for non-openrouter
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-123"}):
        key, url, model = get_cloud_credentials(config)
        assert key == "sk-123"
        assert url == "https://api.openai.com/v1"
        assert model == "google/gemini-2.5-flash"
        
    # Test openrouter url fallback
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "openrouter-456"}):
        key, url, model = get_cloud_credentials(config)
        assert key == "openrouter-456"
        assert url == "https://openrouter.ai/api/v1"

@pytest.mark.asyncio
async def test_evaluate_baseline_case_status_error(tmp_path):
    checkpoint_file = tmp_path / "checkpoint.json"
    evaluator = MagicMock()
    
    res = await evaluate_baseline_case(
        evaluator=evaluator,
        prompts={},
        case_id="c1",
        query="Q1",
        golden_answer="G1",
        expected_papers=[],
        baseline_name="B1",
        baseline_data={"status": "error"},
        checkpoint_data={},
        checkpoint_path=checkpoint_file
    )
    assert res["retrieval_recall"] == 0.0
    assert checkpoint_file.exists()

@pytest.mark.asyncio
async def test_evaluate_baseline_case_cached_values(tmp_path):
    checkpoint_file = tmp_path / "checkpoint.json"
    evaluator = MagicMock()
    
    # Pre-cached values in checkpoint
    checkpoint_data = {
        "c1_B1_135d0c58d6d7": {
            "metrics": {
                "retrieval_recall": 1.0,
                "context_precision": 0.5,
                "context_fillness": 0.2,
                "answer_relevance": 0.9,
                "semantic_accuracy": 0.8,
                "faithfulness": 0.7,
                "citation_fidelity": 0.6,
                "token_output": 100,
                "token_answer": 80,
                "token_reasoning": 20
            },
            "details": {}
        }
    }
    
    res = await evaluate_baseline_case(
        evaluator=evaluator,
        prompts={},
        case_id="c1",
        query="Q1",
        golden_answer="G1",
        expected_papers=[],
        baseline_name="B1",
        baseline_data={
            "generated_answer": "ans",
            "retrieved_chunks": [{"paper_id": "p1"}]
        },
        checkpoint_data=checkpoint_data,
        checkpoint_path=checkpoint_file
    )
    assert res["retrieval_recall"] == 1.0
    assert res["context_precision"] == 0.5
    assert res["context_fillness"] == 0.2

@pytest.mark.asyncio
async def test_evaluate_baseline_case_invalid_types_for_scores(tmp_path):
    checkpoint_file = tmp_path / "checkpoint.json"
    evaluator = MagicMock()
    
    # evaluator returning string/invalid scores
    evaluator.evaluate_all_metrics = AsyncMock(return_value={
        "answer_relevance": {"score": "not_a_float"},
        "semantic_accuracy": "invalid_val"
    })
    
    res = await evaluate_baseline_case(
        evaluator=evaluator,
        prompts={"unified_without_context_evaluator": {"system_prompt": "sys", "user_prompt_template": "user"}},
        case_id="c1",
        query="Q1",
        golden_answer="G1",
        expected_papers=[],
        baseline_name="B0", # no context
        baseline_data={"generated_answer": "ans"},
        checkpoint_data={},
        checkpoint_path=checkpoint_file
    )
    assert res["answer_relevance"] == 0.0
    assert res["semantic_accuracy"] == 0.0

def test_checkpoint_load_save_errors(tmp_path):
    # Load from invalid checkpoint json should return {}
    invalid_file = tmp_path / "invalid.json"
    invalid_file.write_text("invalid json")
    assert load_checkpoint(invalid_file) == {}
    
    # Save checkpoint to a directory instead of file path to trigger OS / permission errors
    save_checkpoint(tmp_path, {"test": 1}) # should not raise error

@pytest.mark.asyncio
async def test_run_evaluation_failures(tmp_path):
    args = MagicMock()
    args.input = str(tmp_path / "nonexistent.yaml")
    args.output = str(tmp_path / "report.yaml")
    
    # File not found
    with pytest.raises(SystemExit):
        await run_evaluation(args, MagicMock(), MagicMock())
        
    # Invalid benchmark structure
    input_file = tmp_path / "input.yaml"
    input_file.write_text("not_results: []")
    args.input = str(input_file)
    with pytest.raises(SystemExit):
        await run_evaluation(args, MagicMock(), MagicMock())

@pytest.mark.asyncio
async def test_run_evaluation_options(tmp_path):
    # Set up input and prompt YAMLs
    input_file = tmp_path / "input.yaml"
    input_data = {
        "metadata": {"version": "1.0"},
        "results": [
            {
                "id": "q1",
                "query": "Q1",
                "golden_answer": "G1",
                "expected_papers": ["p1"],
                "baselines": {
                    "B0": {
                        "status": "success",
                        "generated_answer": "ans0",
                        "latency_sec": 1.0
                    },
                    "B1": {
                        "status": "success",
                        "generated_answer": "ans1",
                        "latency_sec": 1.5,
                        "retrieved_chunks": [{"paper_id": "p1", "page_number": 1, "text_content": "text"}]
                    }
                }
            }
        ]
    }
    with open(input_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(input_data, f)
        
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    prompts_file = prompts_dir / "judge_prompts.yaml"
    prompts_data = {
        "unified_without_context_evaluator": {"system_prompt": "sys", "user_prompt_template": "user {answer}"},
        "unified_with_context_evaluator": {"system_prompt": "sys", "user_prompt_template": "user {answer}"}
    }
    with open(prompts_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(prompts_data, f)
        
    args = MagicMock()
    args.input = str(input_file)
    args.output = str(tmp_path / "reports" / "report.yaml")
    args.baselines = "B0,B1"
    args.clear_checkpoint = True
    args.concurrency = 1
    args.rpm = 0
    args.limit = 1
    
    config = MagicMock()
    config.llm_cloud_api_key = "key"
    config.llm_cloud_base_url = "url"
    config.llm_cloud_model_name = "model"
    config.llm_model_max_context = 4096
    
    mock_evaluator = MagicMock()
    mock_evaluator.evaluate_all_metrics = AsyncMock(return_value={
        "answer_relevance": 1.0,
        "semantic_accuracy": 0.8,
        "faithfulness": 0.9,
        "citation_fidelity": 0.7
    })
    
    # Mock Path.resolve to return a path where parents[1] is tmp_path
    mock_resolve_path = MagicMock()
    mock_resolve_path.parents = [None, tmp_path]
    with patch("core.evaluator.Path.resolve", return_value=mock_resolve_path), \
         patch("core.evaluator.CloudEvaluator", return_value=mock_evaluator):
        await run_evaluation(args, config, MagicMock())
        
    # Check that report was generated
    report_file = tmp_path / "reports" / "report.yaml"
    assert report_file.exists()

def test_fallback_parse_reasoning_response():
    # 1. Unclosed status
    st, ans = _fallback_parse_reasoning_response("<|status_start|>ANSWERABLE")
    assert st == "ANSWERABLE"
    
    # 2. Text fallback keywords
    st, ans = _fallback_parse_reasoning_response("4. _status... NOT ANSWERABLE")
    assert st == "UNANSWERABLE"
    
    # 3. Unclosed answer
    st, ans = _fallback_parse_reasoning_response("<|answer_start|>my final answer text")
    assert ans == "my final answer text"
    
    # 4. Empty/none input
    st, ans = _fallback_parse_reasoning_response(None)
    assert st == "UNKNOWN"
    assert "Error:" in ans

def test_fallback_clean_reasoning_text():
    # 1. Empty text
    assert _fallback_clean_reasoning_text("") == ""
    
    # 2. Section clean up
    text = "1. _analysis... 5. _answer... My Actual Answer"
    assert _fallback_clean_reasoning_text(text) == "My Actual Answer"


@pytest.mark.asyncio
@patch("core.evaluator.asyncio.sleep")
async def test_call_llm_rate_limit_relative_reset(mock_sleep):
    with patch("openai.AsyncOpenAI") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        evaluator = CloudEvaluator(api_key="key", base_url="url", model_name="model", concurrency=1, rpm=0, max_retries=2)
        evaluator.client = mock_client
        
        # Reset is a small relative value (e.g. 5 seconds)
        import openai
        response_mock = MagicMock()
        response_mock.headers = {"x-ratelimit-reset": "5"}
        err = openai.RateLimitError("429 error", response=response_mock, body=None)
        
        mock_client.chat.completions.create = AsyncMock(side_effect=[err, MagicMock()])
        await evaluator.call_llm("sys", "user")
        mock_sleep.assert_called_once()
        # wait_time should be sleep_seconds + 1.0 = 5 + 1.0 = 6.0
        # wait_time is the arg passed to sleep
        mock_sleep.assert_called_with(6.0)


@pytest.mark.asyncio
@patch("core.evaluator.asyncio.sleep")
async def test_call_llm_rate_limit_fallback_parse_regex(mock_sleep):
    with patch("openai.AsyncOpenAI") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        evaluator = CloudEvaluator(api_key="key", base_url="url", model_name="model", concurrency=1, rpm=0, max_retries=2)
        evaluator.client = mock_client
        
        # Raise generic exception containing rate limit (429) and reset info in its message
        err = Exception("429 error: X-RateLimit-Reset: '15'")
        
        mock_client.chat.completions.create = AsyncMock(side_effect=[err, MagicMock()])
        await evaluator.call_llm("sys", "user")
        mock_sleep.assert_called_once()
        # wait_time should be 15 + 1.0 = 16.0
        mock_sleep.assert_called_with(16.0)



@pytest.mark.asyncio
@patch("core.evaluator.asyncio.sleep")
async def test_evaluate_all_metrics_fallback_with_context(mock_sleep):
    with patch("openai.AsyncOpenAI") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        evaluator = CloudEvaluator(api_key="key", base_url="url", model_name="model", concurrency=1, rpm=0, max_retries=1)
        evaluator.client = mock_client
        
        # Make call_llm return invalid JSON to force JSON parse error and fallback
        evaluator.call_llm = AsyncMock(return_value="invalid json")
        
        # With has_context=True, it should include faithfulness and citation_fidelity in fallback
        res = await evaluator.evaluate_all_metrics(
            {"system_prompt": "sys", "user_prompt_template": "user"},
            has_context=True
        )
        assert "faithfulness" in res
        assert "citation_fidelity" in res
        assert res["faithfulness"]["score"] == 0.0
        assert res["citation_fidelity"]["score"] == 0.0
        assert res["answer_relevance"]["score"] == 0.0
        assert res["semantic_accuracy"]["score"] == 0.0

