import pytest
import time
import re
from unittest.mock import MagicMock, AsyncMock, patch
from rich.progress import Task
from rich.text import Text

from run_pipeline import IterationSpeedColumn
from core.evaluator import CloudEvaluator
import src.services.container


# =========================================================================
# 1. Test IterationSpeedColumn
# =========================================================================

def test_iteration_speed_column_render_zero_or_none():
    col = IterationSpeedColumn()
    task_mock = MagicMock(spec=Task)
    task_mock.finished_speed = None
    task_mock.speed = None
    
    res = col.render(task_mock)
    assert isinstance(res, Text)
    assert res.plain == "- sec/it"
    
    task_mock.speed = 0
    res = col.render(task_mock)
    assert res.plain == "- sec/it"


def test_iteration_speed_column_render_valid():
    col = IterationSpeedColumn()
    task_mock = MagicMock(spec=Task)
    task_mock.finished_speed = None
    
    # 2.0 iterations per second -> 0.50 seconds per iteration
    task_mock.speed = 2.0
    res = col.render(task_mock)
    assert res.plain == "0.50 sec/it"
    
    # 0.5 iterations per second -> 2.00 seconds per iteration
    task_mock.speed = 0.5
    res = col.render(task_mock)
    assert res.plain == "2.00 sec/it"


# =========================================================================
# 2. Test Step Output Parsing in run_command_with_progress
# =========================================================================

@patch("rich.progress.Progress")
@patch("subprocess.Popen")
def test_run_command_with_progress_retrieval(mock_popen, mock_progress_class):
    from run_pipeline import run_command_with_progress
    
    # Mock progress
    mock_progress = MagicMock()
    mock_progress_class.return_value.__enter__.return_value = mock_progress
    mock_task = MagicMock()
    mock_progress.add_task.return_value = mock_task
    
    # Mock subprocess output lines
    mock_proc = MagicMock()
    mock_popen.return_value = mock_proc
    mock_proc.stdout.readline.side_effect = [
        "Initializing retrieval...\n",
        "[Q01] Query: 'What is deep learning?' (B1)\n",
        "Loaded some index\n",
        "Query: 'Is this working?'\n",
        ""
    ]
    mock_proc.wait.return_value = 0
    
    run_command_with_progress(["python", "dummy.py"], "Title", 10, "retrieval")
    
    # Check that progress was advanced twice
    assert mock_progress.advance.call_count == 2
    mock_progress.advance.assert_called_with(mock_task, 1)


@patch("rich.progress.Progress")
@patch("subprocess.Popen")
def test_run_command_with_progress_generation(mock_popen, mock_progress_class):
    from run_pipeline import run_command_with_progress
    
    mock_progress = MagicMock()
    mock_progress_class.return_value.__enter__.return_value = mock_progress
    mock_task = MagicMock()
    mock_progress.add_task.return_value = mock_task
    
    mock_proc = MagicMock()
    mock_popen.return_value = mock_proc
    mock_proc.stdout.readline.side_effect = [
        "Running command: python main.py\n",
        "Running CUSTOM: Custom baseline details\n",
        "Loaded model successfully\n",
        "Running B1: baseline B1 details\n",
        ""
    ]
    mock_proc.wait.return_value = 0
    
    run_command_with_progress(["python", "dummy.py"], "Title", 10, "generation")
    
    # Should advance on "Running CUSTOM:" and "Running B1:", but NOT on "Running command:"
    assert mock_progress.advance.call_count == 2
    mock_progress.advance.assert_called_with(mock_task, 1)


@patch("rich.progress.Progress")
@patch("subprocess.Popen")
def test_run_command_with_progress_evaluation(mock_popen, mock_progress_class):
    from run_pipeline import run_command_with_progress
    
    mock_progress = MagicMock()
    mock_progress_class.return_value.__enter__.return_value = mock_progress
    mock_task = MagicMock()
    mock_progress.add_task.return_value = mock_task
    
    mock_proc = MagicMock()
    mock_popen.return_value = mock_proc
    mock_proc.stdout.readline.side_effect = [
        "Evaluated case 1/10\n",
        "Evaluated case 2/10\n",
        "Evaluated case 10/10\n",
        ""
    ]
    mock_proc.wait.return_value = 0
    
    run_command_with_progress(["python", "dummy.py"], "Title", 10, "evaluation")
    
    # Should call progress.update for completed counts
    assert mock_progress.update.call_count == 3
    mock_progress.update.assert_any_call(mock_task, completed=1)
    mock_progress.update.assert_any_call(mock_task, completed=2)
    mock_progress.update.assert_any_call(mock_task, completed=10)


# =========================================================================
# 3. Test Rate Limit & Safety Handling in CloudEvaluator (core/evaluator)
# =========================================================================

@pytest.mark.asyncio
async def test_cloud_evaluator_safety_handling_valid_response():
    evaluator = CloudEvaluator(
        api_key="test_key",
        base_url="https://api.test.com/v1",
        model_name="test-model",
        concurrency=1,
        rpm=0,
        max_retries=2
    )
    
    # Mock client and response structure
    mock_client = AsyncMock()
    evaluator.client = mock_client
    
    mock_choice = MagicMock()
    mock_choice.message.content = "{\"score\": 4.5}"
    
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    
    mock_client.chat.completions.create.return_value = mock_response
    
    res = await evaluator.call_llm("sys", "user")
    assert res == "{\"score\": 4.5}"
    mock_client.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_cloud_evaluator_safety_handling_missing_choices():
    evaluator = CloudEvaluator(
        api_key="test_key",
        base_url="https://api.test.com/v1",
        model_name="test-model",
        concurrency=1,
        rpm=0,
        max_retries=2
    )
    
    mock_client = AsyncMock()
    evaluator.client = mock_client
    
    # Return response with choices missing/None (OpenRouter sometimes wraps errors this way)
    mock_response = MagicMock()
    mock_response.choices = None
    mock_client.chat.completions.create.return_value = mock_response
    
    with pytest.raises(Exception) as exc_info:
        await evaluator.call_llm("sys", "user")
    
    assert "choices list is missing or empty" in str(exc_info.value)
    assert mock_client.chat.completions.create.call_count == 2  # Retried max_retries = 2 times


@pytest.mark.asyncio
@patch("asyncio.sleep")
async def test_cloud_evaluator_rate_limit_reset_header_parsing(mock_sleep):
    import openai
    evaluator = CloudEvaluator(
        api_key="test_key",
        base_url="https://api.test.com/v1",
        model_name="test-model",
        concurrency=1,
        rpm=0,
        max_retries=2
    )
    
    # Mock client to raise RateLimitError
    mock_client = AsyncMock()
    evaluator.client = mock_client
    
    # Mock HTTP response with rate limit headers
    mock_http_response = MagicMock()
    mock_http_response.headers = {
        "x-ratelimit-reset": str(int(time.time() + 15))  # reset in 15 seconds
    }
    
    rate_limit_error = openai.RateLimitError(
        message="Rate limit exceeded",
        response=mock_http_response,
        body=None
    )
    
    # First attempt raises rate limit, second attempt succeeds
    mock_choice = MagicMock()
    mock_choice.message.content = "Success response"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    
    mock_client.chat.completions.create.side_effect = [rate_limit_error, mock_response]
    
    res = await evaluator.call_llm("sys", "user")
    assert res == "Success response"
    
    # The sleep time should be around 15 seconds + 1.0 buffer = 16 seconds
    # Let's check the argument of the sleep call
    mock_sleep.assert_called_once()
    sleep_call_arg = mock_sleep.call_args[0][0]
    assert 13.0 <= sleep_call_arg <= 17.0


@pytest.mark.asyncio
@patch("asyncio.sleep")
async def test_cloud_evaluator_rate_limit_reset_string_fallback_parsing(mock_sleep):
    evaluator = CloudEvaluator(
        api_key="test_key",
        base_url="https://api.test.com/v1",
        model_name="test-model",
        concurrency=1,
        rpm=0,
        max_retries=2
    )
    
    mock_client = AsyncMock()
    evaluator.client = mock_client
    
    # Simulate an error with rate limit reset timestamp in the exception string message
    reset_timestamp_ms = int((time.time() + 25) * 1000)
    error_msg = f"Rate limit exceeded. Reset time: 'X-RateLimit-Reset': '{reset_timestamp_ms}'"
    rate_limit_error = Exception(error_msg)
    
    mock_choice = MagicMock()
    mock_choice.message.content = "Success response"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    
    mock_client.chat.completions.create.side_effect = [rate_limit_error, mock_response]
    
    res = await evaluator.call_llm("sys", "user")
    assert res == "Success response"
    
    # Should parse the timestamp in ms, convert to seconds, subtract current time, and sleep.
    # Expected sleep is around 25 seconds + 1.0 buffer = 26 seconds.
    mock_sleep.assert_called_once()
    sleep_call_arg = mock_sleep.call_args[0][0]
    assert 23.0 <= sleep_call_arg <= 27.0


# =========================================================================
# 4. Test Concurrency & Pipelined Execution
# =========================================================================

@pytest.mark.asyncio
async def test_concurrent_yaml_writes(tmp_path):
    import asyncio
    import yaml
    from core.pipelined import safe_read_modify_write_yaml
    
    file_path = tmp_path / "concurrent_test.yaml"
    
    def append_to_list(val):
        def modify_fn(existing_data):
            if not existing_data or not isinstance(existing_data, dict):
                existing_data = {"items": []}
            existing_data["items"].append(val)
            return existing_data
        return modify_fn
        
    async def writer_task(i):
        # run in thread pool to simulate thread concurrency
        await asyncio.to_thread(safe_read_modify_write_yaml, file_path, append_to_list(i))
        
    # Launch 30 concurrent writer tasks
    tasks = [writer_task(i) for i in range(30)]
    await asyncio.gather(*tasks)
    
    # Read and verify
    assert file_path.exists()
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        
    assert data is not None
    items = data.get("items", [])
    assert len(items) == 30
    assert sorted(items) == list(range(30))


@pytest.mark.asyncio
@patch("core.evaluator.CloudEvaluator.call_llm")
@patch("src.services.container.container.get_rag_service")
async def test_pipelined_stage_execution(mock_get_rag_service, mock_call_llm, tmp_path):
    import yaml
    import json
    from core.pipelined import run_pipelined_stage_async
    from src.config import config
    
    # Mock call_llm to return simulated judge JSON scores
    async def mock_call_llm_fn(system_prompt, user_prompt):
        # check if it needs context or not
        if "faithfulness" in user_prompt or "faithfulness" in system_prompt or "citation_fidelity" in user_prompt:
            return json.dumps({
                "answer_relevance": {"score": 0.9},
                "semantic_accuracy": {"score": 0.8},
                "faithfulness": {"score": 0.95},
                "citation_fidelity": {"score": 0.85}
            })
        else:
            return json.dumps({
                "answer_relevance": {"score": 0.9},
                "semantic_accuracy": {"score": 0.8}
            })
    mock_call_llm.side_effect = mock_call_llm_fn
    
    # Mock RAG Service
    rag_service = MagicMock()
    rag_service.llm_engine.generate_response.return_value = "generated pipelined response"
    rag_service.llm_engine.count_tokens.return_value = 10
    rag_service.llm_engine.unload_model = MagicMock()
    
    # Mock retrieved chunks
    class MockChunk:
        def __init__(self, cid, pid, text, page):
            self.id = cid
            self.paper_id = pid
            self.text_content = text
            self.page_number = page
            
    mock_chunk = MockChunk("c1", "p1", "some scientific context text", 1)
    rag_service.retrieve_relevant_chunks.return_value = [(mock_chunk, 0.95)]
    rag_service.ask.return_value = "ask response"
    rag_service.last_raw_response = "raw response"
    
    mock_get_rag_service.return_value = rag_service
    
    # Set up config mockup
    mock_config = MagicMock()
    mock_config.data = {
        "llm": {
            "provider": "local",
            "local": {"model_path": "test_model"},
            "temp": 0.1,
            "max_tokens": 100
        },
        "embedding": {
            "model_name": "test_emb"
        },
        "rag_components": {
            "reranker": False,
            "citation_repair": False
        }
    }
    mock_config.rag_components = mock_config.data["rag_components"]
    mock_config.llm_model_max_context = 4096
    mock_config.reranker_model_name = "disabled"
    mock_config.llm_evaluation_concurrency = 2
    mock_config.llm_evaluation_rpm = 100
    mock_config.llm_evaluation_retries = 3
    mock_config.llm_cloud_api_key = "dummy_key"
    mock_config.llm_cloud_base_url = "https://api.openai.com/v1"
    mock_config.llm_cloud_model_name = "google/gemini-2.5-flash"
    
    # Dataset path setup
    dataset_path = tmp_path / "dataset.yaml"
    dataset_data = [
        {"id": "Q1", "query": "What is deep learning?", "expected_papers": ["paper_1"]},
        {"id": "Q2", "query": "What is reinforcement learning?", "expected_papers": ["paper_2"]}
    ]
    with open(dataset_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dataset_data, f)
        
    eval_results = tmp_path / "evaluation_results.yaml"
    metrics_results = tmp_path / "result_metrics.yaml"
    
    # Set up args mock
    args = MagicMock()
    args.limit = 2
    args.consume_contexts = None
    args.cloud = False
    args.concurrency = 2
    args.rpm = 100
    args.retries = 2
    args.clear_checkpoint = True
    
    # Run pipelined async function
    await run_pipelined_stage_async(
        args=args,
        config=mock_config,
        run_dir=tmp_path,
        dataset_path=dataset_path,
        baselines_to_run=["B0", "B1"],
        eval_results=eval_results,
        metrics_results=metrics_results,
        retrieved_contexts_file=None,
        total_steps=4
    )
    
    # Assert eval_results exists and contains the generated answers
    assert eval_results.exists()
    with open(eval_results, "r", encoding="utf-8") as f:
        eval_data = yaml.safe_load(f)
    assert len(eval_data["results"]) == 2
    assert "B0" in eval_data["results"][0]["baselines"]
    assert "B1" in eval_data["results"][0]["baselines"]
    
    # Assert metrics_results exists and contains scores
    assert metrics_results.exists()
    with open(metrics_results, "r", encoding="utf-8") as f:
        metrics_data = yaml.safe_load(f)
    assert len(metrics_data["results"]) == 2
    assert "eval_metrics" in metrics_data["results"][0]["baselines"]["B0"]
    assert "eval_metrics" in metrics_data["results"][0]["baselines"]["B1"]
    
    # B0 should have relevance and semantic, but B1 (with context) should also have faithfulness and citation fidelity
    b0_metrics = metrics_data["results"][0]["baselines"]["B0"]["eval_metrics"]
    b1_metrics = metrics_data["results"][0]["baselines"]["B1"]["eval_metrics"]
    assert b0_metrics["answer_relevance"] == 0.9
    assert b0_metrics["semantic_accuracy"] == 0.8
    assert b1_metrics["faithfulness"] == 0.95
    assert b1_metrics["citation_fidelity"] == 0.85
    
    # Check that summary average is calculated
    assert "summary" in metrics_data
    assert "B0" in metrics_data["summary"]
    assert "avg_answer_relevance" in metrics_data["summary"]["B0"]
    assert metrics_data["summary"]["B0"]["avg_answer_relevance"] == 0.9

