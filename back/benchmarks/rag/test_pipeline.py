import pytest
import time
import re
from unittest.mock import MagicMock, AsyncMock, patch
from rich.progress import Task
from rich.text import Text

from run_pipeline import IterationSpeedColumn
from core.evaluator import CloudEvaluator

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
