import pytest
import time
import yaml
from unittest.mock import MagicMock, AsyncMock, patch
from rich.progress import Task
from rich.text import Text

from core.subprocess_runner import IterationSpeedColumn
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
    from core.subprocess_runner import run_command_with_progress
    
    # Mock progress
    mock_progress = MagicMock()
    mock_progress_class.return_value.__enter__.return_value = mock_progress
    mock_task = MagicMock()
    mock_progress.add_task.return_value = mock_task
    
    # Mock subprocess output lines — Stage-3 Query logs then Stage-5 PROGRESS
    mock_proc = MagicMock()
    mock_popen.return_value = mock_proc
    mock_proc.stdout.readline.side_effect = [
        "Initializing retrieval...\n",
        "[Q01] Query: 'What is deep learning?' (B1)\n",
        "Loaded some index\n",
        "Query: 'Is this working?'\n",
        "PROGRESS retrieval 1/10\n",
        "PROGRESS retrieval 2/10\n",
        ""
    ]
    mock_proc.wait.return_value = 0
    
    run_command_with_progress(["python", "dummy.py"], "Title", 10, "retrieval")
    
    # Query lines do not advance; only PROGRESS set 1,2 then success reconciles to 10
    completed_values = [
        c.kwargs.get("completed")
        for c in mock_progress.update.call_args_list
        if "completed" in c.kwargs
    ]
    assert completed_values[0] == 1
    assert 2 in completed_values
    assert completed_values[-1] == 10


@patch("rich.progress.Progress")
@patch("subprocess.Popen")
def test_run_command_with_progress_generation(mock_popen, mock_progress_class):
    from core.subprocess_runner import run_command_with_progress
    
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
        "Reusing previously generated answer for B2 from checkpoint.\n",
        ""
    ]
    mock_proc.wait.return_value = 0
    
    run_command_with_progress(["python", "dummy.py"], "Title", 10, "generation")
    
    # Running CUSTOM, Running B1, Reusing B2 -> completed 1,2,3 then reconcile 10
    completed_values = [
        c.kwargs.get("completed")
        for c in mock_progress.update.call_args_list
        if "completed" in c.kwargs
    ]
    assert 1 in completed_values
    assert 2 in completed_values
    assert 3 in completed_values
    assert completed_values[-1] == 10


@patch("rich.progress.Progress")
@patch("subprocess.Popen")
def test_run_command_with_progress_evaluation(mock_popen, mock_progress_class):
    from core.subprocess_runner import run_command_with_progress
    
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


@pytest.mark.asyncio
@patch("core.evaluator.CloudEvaluator.call_llm")
@patch("src.services.container.container.get_rag_service")
async def test_pipelined_stage_rate_limit_capacity_dispatcher(mock_get_rag_service, mock_call_llm, tmp_path):
    import yaml
    import json
    from core.pipelined import run_pipelined_stage_async
    from core.evaluator import CloudEvaluator

    async def mock_call_llm_fn(system_prompt, user_prompt):
        return json.dumps({
            "answer_relevance": {"score": 0.95},
            "semantic_accuracy": {"score": 0.90}
        })
    mock_call_llm.side_effect = mock_call_llm_fn

    rag_service = MagicMock()
    rag_service.llm_engine.generate_response.return_value = "generated response"
    rag_service.llm_engine.count_tokens.return_value = 10
    rag_service.llm_engine.unload_model = MagicMock()
    rag_service.retrieve_relevant_chunks.return_value = []
    rag_service.ask.return_value = "ask response"
    rag_service.last_raw_response = "raw response"
    mock_get_rag_service.return_value = rag_service

    mock_config = MagicMock()
    mock_config.data = {
        "llm": {
            "provider": "local",
            "local": {"model_path": "test_model"},
            "temp": 0.1,
            "max_tokens": 100
        },
        "embedding": {"model_name": "test_emb"},
        "rag_components": {"reranker": False, "citation_repair": False}
    }
    mock_config.rag_components = mock_config.data["rag_components"]
    mock_config.llm_model_max_context = 4096
    mock_config.reranker_model_name = "disabled"
    mock_config.llm_evaluation_concurrency = 2
    mock_config.llm_evaluation_rpm = 60
    mock_config.llm_evaluation_retries = 2
    mock_config.llm_cloud_api_key = "dummy_key"
    mock_config.llm_cloud_base_url = "https://api.openai.com/v1"
    mock_config.llm_cloud_model_name = "google/gemini-2.5-flash"

    dataset_path = tmp_path / "dataset.yaml"
    dataset_data = [{"id": "Q1", "query": "Test query?", "expected_papers": []}]
    with open(dataset_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dataset_data, f)

    eval_results = tmp_path / "evaluation_results.yaml"
    metrics_results = tmp_path / "result_metrics.yaml"

    args = MagicMock()
    args.limit = 1
    args.consume_contexts = None
    args.cloud = False
    args.concurrency = 2
    args.rpm = 60
    args.retries = 2
    args.clear_checkpoint = True

    # Spy on CloudEvaluator.has_capacity
    original_has_capacity = CloudEvaluator.has_capacity
    has_capacity_calls = 0

    def spy_has_capacity(self_eval):
        nonlocal has_capacity_calls
        has_capacity_calls += 1
        return original_has_capacity(self_eval)

    with patch.object(CloudEvaluator, "has_capacity", side_effect=spy_has_capacity, autospec=True):
        await run_pipelined_stage_async(
            args=args,
            config=mock_config,
            run_dir=tmp_path,
            dataset_path=dataset_path,
            baselines_to_run=["B0"],
            eval_results=eval_results,
            metrics_results=metrics_results,
            retrieved_contexts_file=None,
            total_steps=1
        )

    assert has_capacity_calls >= 1
    assert metrics_results.exists()


@pytest.mark.asyncio
async def test_run_evaluation_checkpoint_resumption_and_judge_reports(tmp_path):
    import json
    import yaml
    import hashlib
    from unittest.mock import MagicMock, patch, AsyncMock
    from core.evaluator import run_evaluation
    from src.config import Config

    # 1. Prepare temp paths
    input_yaml = tmp_path / "evaluation_results.yaml"
    output_yaml = tmp_path / "result_metrics.yaml"
    judge_yaml = tmp_path / "result_metrics_judge.yaml"

    # 2. Create dummy evaluation_results.yaml (input_data)
    input_data = {
        "metadata": {
            "llm": {"model_max_context": 4096}
        },
        "results": [
            {
                "id": "Q01",
                "category": "general",
                "query": "Query 1",
                "golden_answer": "Golden 1",
                "expected_papers": ["paper1"],
                "baselines": {
                    "B1": {
                        "status": "success",
                        "latency_sec": 1.5,
                        "generated_answer": "Answer 1",
                        "retrieved_chunks": [
                            {"id": "c1", "paper_id": "paper1", "page_number": 1, "text_content": "Content 1"}
                        ]
                    }
                }
            },
            {
                "id": "Q02",
                "category": "general",
                "query": "Query 2",
                "golden_answer": "Golden 2",
                "expected_papers": ["paper2"],
                "baselines": {
                    "B1": {
                        "status": "success",
                        "latency_sec": 2.5,
                        "generated_answer": "Answer 2",
                        "retrieved_chunks": [
                            {"id": "c2", "paper_id": "paper2", "page_number": 2, "text_content": "Content 2"}
                        ]
                    }
                }
            }
        ]
    }
    with open(input_yaml, "w", encoding="utf-8") as f:
        yaml.dump(input_data, f)

    # 3. Generate the payload hash for Q01 to place it in the checkpoint
    hash_payload = {
        "generated_answer": "Answer 1",
        "retrieved_chunks": [
            {"id": "c1", "paper_id": "paper1", "page_number": 1, "text_content": "Content 1"}
        ],
        "golden_answer": "Golden 1",
        "query": "Query 1"
    }
    payload_str = json.dumps(hash_payload, sort_keys=True, ensure_ascii=False)
    payload_hash = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()[:12]
    q01_checkpoint_key = f"Q01_B1_{payload_hash}"

    # Write checkpoint file
    checkpoint_path = tmp_path / ".eval_checkpoint.json"
    checkpoint_data = {
        q01_checkpoint_key: {
            "metrics": {
                "retrieval_recall": 1.0,
                "context_precision": 1.0,
                "faithfulness": 0.99,
                "answer_relevance": 0.98,
                "citation_fidelity": 0.97,
                "semantic_accuracy": 0.96,
                "context_fillness": 0.05,
                "token_output": 100,
                "token_answer": 80,
                "token_reasoning": 20
            },
            "details": {
                "faithfulness": {"reason": "Perfect alignment"},
                "answer_relevance": {"reason": "Highly relevant"}
            }
        }
    }
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f)

    # 4. Run run_evaluation with mock evaluator
    mock_evaluator = MagicMock()

    class DummyArgs:
        def __init__(self):
            self.input = str(input_yaml)
            self.output = str(output_yaml)
            self.baselines = "all"
            self.limit = None
            self.concurrency = 2
            self.rpm = 60
            self.retries = 3
            self.clear_checkpoint = False

    args = DummyArgs()

    mock_config = MagicMock(spec=Config)
    mock_config.llm_evaluation_concurrency = 2
    mock_config.llm_evaluation_rpm = 60
    mock_config.llm_evaluation_retries = 3
    mock_config.data = {
        "llm": {
            "cloud": {
                "api_key": "dummy",
                "base_url": "dummy",
                "model_name": "dummy"
            }
        }
    }

    mock_con = MagicMock()

    with patch("core.evaluator.get_cloud_credentials", return_value=("key", "url", "model")), \
         patch("core.evaluator.CloudEvaluator", return_value=mock_evaluator):

        # Mock evaluate_all_metrics on mock_evaluator
        mock_evaluator.evaluate_all_metrics = AsyncMock(return_value={
            "faithfulness": {"score": 0.9, "reason": "Good"},
            "answer_relevance": {"score": 0.8, "reason": "Ok"},
            "citation_fidelity": {"score": 0.7, "reason": "Reasonable"},
            "semantic_accuracy": {"score": 0.6, "reason": "Acceptable"}
        })

        await run_evaluation(args, mock_config, mock_con)

        # Verify that mock_evaluator was called ONLY once (for Q02), since Q01 was loaded from checkpoint
        assert mock_evaluator.evaluate_all_metrics.call_count == 1

    # Verify outputs
    assert output_yaml.exists()
    with open(output_yaml, "r", encoding="utf-8") as f:
        metrics_data = yaml.safe_load(f)

    # The checkpoint should be unlinked/deleted upon completion
    assert not checkpoint_path.exists()

    # Check results in metrics_data
    results = metrics_data["results"]
    assert len(results) == 2

    # Q01 should have cached metrics
    q01_res = next(r for r in results if r["id"] == "Q01")
    q01_metrics = q01_res["baselines"]["B1"]["eval_metrics"]
    q01_details = q01_res["baselines"]["B1"]["eval_details"]
    assert q01_metrics["faithfulness"] == 0.99
    assert q01_metrics["answer_relevance"] == 0.98
    assert q01_details["faithfulness"] == {"reason": "Perfect alignment"}

    # Q02 should have evaluated metrics
    q02_res = next(r for r in results if r["id"] == "Q02")
    q02_metrics = q02_res["baselines"]["B1"]["eval_metrics"]
    q02_details = q02_res["baselines"]["B1"]["eval_details"]
    assert q02_metrics["faithfulness"] == 0.9
    assert q02_metrics["answer_relevance"] == 0.8
    assert q02_details["faithfulness"] == {"score": 0.9, "reason": "Good"}

    # Check judge_yaml results
    assert judge_yaml.exists()
    with open(judge_yaml, "r", encoding="utf-8") as f:
        judge_data = yaml.safe_load(f)

    judge_results = judge_data["results"]
    assert len(judge_results) == 2

    q01_judge = next(r for r in judge_results if r["id"] == "Q01")
    assert q01_judge["baselines"]["B1"]["eval_metrics"]["faithfulness"] == 0.99
    assert q01_judge["baselines"]["B1"]["eval_details"]["faithfulness"] == {"reason": "Perfect alignment"}


def test_safe_read_modify_write_yaml_no_fcntl(tmp_path):
    from core.pipelined import safe_read_modify_write_yaml
    file_path = tmp_path / "test.yaml"
    
    def modify_fn(existing):
        if existing is None:
            return {"key": "val"}
        return existing
        
    with patch("core.pipelined.HAS_FCNTL", False):
        safe_read_modify_write_yaml(file_path, modify_fn)
        
    assert file_path.exists()
    with open(file_path, "r") as f:
        data = yaml.safe_load(f)
    assert data == {"key": "val"}

def test_safe_read_modify_write_yaml_read_error(tmp_path):
    from core.pipelined import safe_read_modify_write_yaml
    file_path = tmp_path / "test.yaml"
    file_path.write_text("some corrupted content")
    
    def modify_fn(existing):
        # existing should be None because of read error simulation
        assert existing is None
        return {"key": "recovered"}
        
    # Mock open to raise exception on read
    orig_open = open
    def mock_open(file, mode="r", *args, **kwargs):
        if str(file) == str(file_path) and "r" in mode:
            raise OSError("mock read error")
        return orig_open(file, mode, *args, **kwargs)
        
    with patch("builtins.open", side_effect=mock_open):
        safe_read_modify_write_yaml(file_path, modify_fn)
        
    with open(file_path, "r") as f:
        data = yaml.safe_load(f)
    assert data == {"key": "recovered"}

def test_save_evaluation_baseline_result_filtering(tmp_path):
    from core.pipelined import save_evaluation_baseline_result
    file_path = tmp_path / "test.yaml"
    
    case_info = {"category": "test", "query": "Q", "golden_answer": "G", "expected_papers": []}
    baseline_data = {"status": "success", "latency_sec": 1.2, "retrieved_chunks": []}
    eval_metrics_raw = {"faithfulness": 0.9, "citation_fidelity": 0.85, "context_precision": 0.95, "answer_relevance": 0.9}
    
    # Save a B1 baseline without retrieved chunks. RAG metrics (faithfulness, citation fidelity, context precision)
    # should be skipped from summary averages due to empty retrieved_chunks (line 145).
    save_evaluation_baseline_result(file_path, "Q1", case_info, "B1", baseline_data, eval_metrics_raw, {})
    
    with open(file_path, "r") as f:
        data = yaml.safe_load(f)
        
    summary = data["summary"]["B1"]
    assert "avg_faithfulness" not in summary or summary["avg_faithfulness"] == 0.0
    assert "avg_citation_fidelity" not in summary or summary["avg_citation_fidelity"] == 0.0
    assert summary["avg_answer_relevance"] == 0.9

def test_generate_baseline_case_pre_contexts_error():
    from core.pipelined import generate_baseline_case
    
    rag_service = MagicMock()
    config = MagicMock()
    config.llm_model_max_context = 4096
    
    # Baseline not found in pre_contexts -> status="error"
    pre_contexts = {"Q1": {"baselines": {}}}
    
    res = generate_baseline_case(rag_service, config, MagicMock(), "Q1", {"query": "Q"}, "B1", MagicMock(), pre_contexts)
    assert res["status"] == "error"
    assert "Error: No pre-retrieved context" in res["generated_answer"]
    assert res["retrieval_recall"] == 0.0
    assert res["context_fillness"] == 0.0

def test_generate_baseline_case_pre_contexts_success_and_fallbacks():
    from core.pipelined import generate_baseline_case
    
    rag_service = MagicMock()
    rag_service.llm_engine.generate_response.return_value = "response"
    # count_tokens raises exception to cover line 230-231
    rag_service.llm_engine.count_tokens.side_effect = Exception("token count error")
    # citation repair warning exception to cover line 247-249
    rag_service._validate_and_repair_citations.side_effect = Exception("repair error")
    
    config = MagicMock()
    config.rag_components = {"citation_repair": True}
    config.llm_model_max_context = 4096
    
    pre_contexts = {
        "Q1": {
            "baselines": {
                "B3": {
                    "status": "success",
                    "latency_sec": 0.5,
                    "retrieved_chunks": [{"id": "c1", "paper_id": "p1", "text_content": "text", "page_number": 1}],
                    "trimmed_text": "text",
                    "trimmed_graph": "graph",
                    "enrichment_block": None,
                    "metrics": {"components": {}}
                }
            }
        }
    }
    
    prompts = MagicMock()
    prompts.get_prompt.return_value = "prompt"
    
    args = MagicMock()
    
    # Test B3 with no enrichment block and citation repair (runs ask_no_expander)
    res = generate_baseline_case(rag_service, config, prompts, "Q1", {"query": "Q"}, "B3", args, pre_contexts)
    assert res["status"] == "success"
    assert res["generated_answer"] == "response"
    assert res["context_token"] == len("prompt") // 4
    
    # Test B3 with enrichment block and citation repair (runs ask_expander)
    pre_contexts["Q1"]["baselines"]["B3"]["enrichment_block"] = "essential graph info"
    res2 = generate_baseline_case(rag_service, config, prompts, "Q1", {"query": "Q"}, "B3", args, pre_contexts)
    assert res2["status"] == "success"
    assert res2["generated_answer"] == "response"
    assert res2["context_token"] == len("prompt") // 4
    
    # Test B0 prompt path (line 216-217) and trace token check non-int fallback (line 273-276)
    pre_contexts["Q1"]["baselines"]["B0"] = {
        "status": "success",
        "latency_sec": 0.1,
        "metrics": {"components": {}},
        "trace": {"answer_token_count": 0}
    }
    # Reset count_tokens mock for B0 to return an integer for prompts and non-int for answers
    def count_tokens_side_effect(arg):
        if "Question" in arg or "prompt" in arg:
            return 5
        return "non-int-string"
    rag_service.llm_engine.count_tokens.side_effect = count_tokens_side_effect
    
    res = generate_baseline_case(rag_service, config, prompts, "Q1", {"query": "Q"}, "B0", args, pre_contexts)
    assert res["status"] == "success"
    assert res["trace"]["answer_token_count"] == len("response") // 4

def test_generate_baseline_case_execution_error():
    from core.pipelined import generate_baseline_case
    rag_service = MagicMock()
    # Trigger exception in run_query_on_baseline (line 312-328)
    rag_service.retrieve_relevant_chunks.side_effect = Exception("execution error")
    
    config = MagicMock()
    config.rag_components = {"reranker": False}
    config.llm_model_max_context = 4096
    
    res = generate_baseline_case(rag_service, config, MagicMock(), "Q1", {"query": "Q"}, "B1", MagicMock(), None)
    assert res["status"] == "error"
    assert "Error occurred during generation" in res["generated_answer"]

@pytest.mark.asyncio
async def test_run_pipelined_stage_async_missing_prompts(tmp_path):
    from core.pipelined import run_pipelined_stage_async
    
    args = MagicMock()
    args.limit = 1
    args.cloud = False
    
    config = MagicMock()
    
    # Create fake directory structure without judge_prompts.yaml to trigger line 403-404
    with patch("core.pipelined.Path.resolve") as mock_resolve, \
         patch("src.services.container.container.get_rag_service"), \
         patch("core.evaluator.get_cloud_credentials", return_value=("key", "https://api.openai.com/v1", "model")), \
         patch("core.evaluator.CloudEvaluator"):
        
        mock_path = MagicMock()
        mock_path.parents = [None, tmp_path] # parents[1] is tmp_path
        mock_resolve.return_value = mock_path
        
        with pytest.raises(SystemExit):
            await run_pipelined_stage_async(
                args, config, tmp_path, tmp_path / "dataset.yaml", ["B0"],
                tmp_path / "eval.yaml", tmp_path / "metrics.yaml", None, 1
            )

@pytest.mark.asyncio
async def test_run_pipelined_stage_async_contexts_load_fail(tmp_path):
    from core.pipelined import run_pipelined_stage_async
    
    args = MagicMock()
    args.limit = 1
    args.cloud = False
    args.concurrency = 1
    args.rpm = 60
    args.retries = 2
    args.clear_checkpoint = True
    
    config = MagicMock()
    config.data = {
        "llm": {
            "provider": "local",
            "local": {"model_path": "local_model"} # local model metadata setup (line 431-432)
        },
        "embedding": {"model_name": "emb"},
        "rag_components": {"reranker": False}
    }
    config.rag_components = config.data["rag_components"]
    config.llm_model_max_context = 4096
    config.reranker_model_name = "disabled"
    
    # Corrupt pre-retrieved contexts yaml (triggers line 422-423 warning)
    contexts_file = tmp_path / "contexts.yaml"
    contexts_file.write_text("invalid: yaml: :")
    
    # Dataset
    dataset_path = tmp_path / "dataset.yaml"
    yaml.dump([{"id": "Q1", "query": "Q"}], open(dataset_path, "w"))
    
    # Create dummy judge_prompts
    (tmp_path / "prompts").mkdir(exist_ok=True)
    judge_prompts_file = tmp_path / "prompts" / "judge_prompts.yaml"
    yaml.dump({"prompt": "text"}, open(judge_prompts_file, "w"))
    
    rag_service = MagicMock()
    rag_service.llm_engine.generate_response.return_value = "ans"
    rag_service.llm_engine.count_tokens.return_value = 5
    
    mock_evaluator = MagicMock()
    mock_evaluator.evaluate_all_metrics = AsyncMock(return_value={})
    
    with patch("core.pipelined.Path.resolve") as mock_resolve, \
         patch("src.services.container.container.get_rag_service", return_value=rag_service), \
         patch("core.evaluator.get_cloud_credentials", return_value=("key", "https://api.openai.com/v1", "model")), \
         patch("core.evaluator.CloudEvaluator", return_value=mock_evaluator):
         
        mock_path = MagicMock()
        mock_path.parents = [None, tmp_path]
        mock_resolve.return_value = mock_path
        
        await run_pipelined_stage_async(
            args, config, tmp_path, dataset_path, ["B0"],
            tmp_path / "eval.yaml", tmp_path / "metrics.yaml", contexts_file, 1
        )
        
    assert (tmp_path / "eval.yaml").exists()

@pytest.mark.asyncio
async def test_run_pipelined_stage_async_unlink_error_and_reused_generation(tmp_path):
    from core.pipelined import run_pipelined_stage_async
    from pathlib import Path
    
    args = MagicMock()
    args.limit = 1
    args.cloud = True
    args.concurrency = 1
    args.rpm = 60
    args.retries = 2
    args.clear_checkpoint = True # triggers checkpoint unlink
    
    config = MagicMock()
    config.data = {
        "llm": {
            "provider": "cloud",
            "cloud": {"model_name": "cloud_model"}
        },
        "embedding": {"model_name": "emb"},
        "rag_components": {"reranker": False}
    }
    config.rag_components = config.data["rag_components"]
    config.llm_model_max_context = 4096
    config.reranker_model_name = "disabled"
    
    # Dataset
    dataset_path = tmp_path / "dataset.yaml"
    yaml.dump([{"id": "Q1", "query": "Q"}], open(dataset_path, "w"))
    
    # Pre-existing generation file to cover lines 467-474
    eval_results = tmp_path / "eval.yaml"
    yaml.dump({"results": [{"id": "Q1", "baselines": {"B0": {"status": "success", "generated_answer": "old answer"}}}]}, open(eval_results, "w"))
    
    # Create dummy judge_prompts
    (tmp_path / "prompts").mkdir(exist_ok=True)
    judge_prompts_file = tmp_path / "prompts" / "judge_prompts.yaml"
    yaml.dump({"prompt": "text"}, open(judge_prompts_file, "w"))
    
    # Create a dummy checkpoint file
    checkpoint_path = tmp_path / ".eval_checkpoint.json"
    checkpoint_path.write_text("{}")
    
    rag_service = MagicMock()
    rag_service.llm_engine.generate_response.return_value = "ans"
    rag_service.llm_engine.count_tokens.return_value = 5
    
    mock_evaluator = MagicMock()
    mock_evaluator.evaluate_all_metrics = AsyncMock(return_value={})
    
    # Modify args.clear_checkpoint to False to trigger reuse of old generation (lines 529-534)
    args.clear_checkpoint = False
    
    with patch("core.pipelined.Path.resolve") as mock_resolve, \
         patch("src.services.container.container.get_rag_service", return_value=rag_service), \
         patch("core.evaluator.get_cloud_credentials", return_value=("key", "https://api.openai.com/v1", "model")), \
         patch("core.evaluator.CloudEvaluator", return_value=mock_evaluator), \
         patch.object(Path, "unlink", side_effect=OSError("mock unlink fail")):
         
        mock_path = MagicMock()
        mock_path.parents = [None, tmp_path]
        mock_resolve.return_value = mock_path
        
        await run_pipelined_stage_async(
            args, config, tmp_path, dataset_path, ["B0"],
            eval_results, tmp_path / "metrics.yaml", None, 1
        )
        
    assert eval_results.exists()
    # verify answer reused
    with open(eval_results, "r") as f:
        res = yaml.safe_load(f)
    assert res["results"][0]["baselines"]["B0"]["generated_answer"] == "old answer"

@pytest.mark.asyncio
async def test_run_pipelined_stage_async_errors(tmp_path):
    from core.pipelined import run_pipelined_stage_async
    
    args = MagicMock()
    args.limit = 1
    args.cloud = True
    args.concurrency = 1
    args.rpm = 60
    args.retries = 2
    args.clear_checkpoint = True
    
    config = MagicMock()
    config.data = {
        "llm": {
            "provider": "cloud",
            "cloud": {"model_name": "cloud_model"}
        },
        "embedding": {"model_name": "emb"},
        "rag_components": {"reranker": False}
    }
    config.rag_components = config.data["rag_components"]
    config.llm_model_max_context = 4096
    config.reranker_model_name = "disabled"
    
    # Dataset
    dataset_path = tmp_path / "dataset.yaml"
    yaml.dump([{"id": "Q1", "query": "Q"}], open(dataset_path, "w"))
    
    # Create dummy judge_prompts
    (tmp_path / "prompts").mkdir(exist_ok=True)
    judge_prompts_file = tmp_path / "prompts" / "judge_prompts.yaml"
    yaml.dump({"prompt": "text"}, open(judge_prompts_file, "w"))
    
    rag_service = MagicMock()
    rag_service.llm_engine.generate_response.return_value = "ans"
    rag_service.llm_engine.count_tokens.return_value = 5
    # Model unload error (line 632-633)
    rag_service.llm_engine.unload_model.side_effect = Exception("unload error")
    
    mock_evaluator = MagicMock()
    # Evaluator worker raises error (line 608-609)
    mock_evaluator.evaluate_all_metrics.side_effect = ValueError("eval error")
    
    with patch("core.pipelined.Path.resolve") as mock_resolve, \
         patch("src.services.container.container.get_rag_service", return_value=rag_service), \
         patch("core.evaluator.get_cloud_credentials", return_value=("key", "https://api.openai.com/v1", "model")), \
         patch("core.evaluator.CloudEvaluator", return_value=mock_evaluator), \
         patch("core.reporting.save_judge_report", side_effect=Exception("save judge report failed")): # triggers line 627-628
         
        mock_path = MagicMock()
        mock_path.parents = [None, tmp_path]
        mock_resolve.return_value = mock_path
        
        await run_pipelined_stage_async(
            args, config, tmp_path, dataset_path, ["B0"],
            tmp_path / "eval.yaml", tmp_path / "metrics.yaml", None, 1
        )
        
    assert (tmp_path / "eval.yaml").exists()


def test_pipelined_import_no_fcntl():
    # Force reload core.pipelined with fcntl mocked out to raise ImportError (line 12-13)
    import sys
    import importlib
    with patch.dict("sys.modules", {"fcntl": None}):
        if "core.pipelined" in sys.modules:
            importlib.reload(sys.modules["core.pipelined"])
    # Reload again to restore standard imports
    if "core.pipelined" in sys.modules:
        importlib.reload(sys.modules["core.pipelined"])

def test_generate_baseline_case_more_fallbacks():
    from core.pipelined import generate_baseline_case
    
    rag_service = MagicMock()
    # Trigger exception on load_model (or generate_response) to cover lines 279-284
    rag_service.llm_engine._ensure_model_loaded.side_effect = Exception("failed loaded model")
    
    config = MagicMock()
    config.rag_components = {"citation_repair": True}
    config.llm_model_max_context = 4096
    
    pre_contexts = {
        "Q1": {
            "baselines": {
                "B3": {
                    "status": "success",
                    "latency_sec": 0.5,
                    "retrieved_chunks": [],
                    "trimmed_text": "text",
                    "trimmed_graph": "graph",
                    "metrics": {} # components key is missing -> covers line 257
                }
            }
        }
    }
    
    res = generate_baseline_case(rag_service, config, MagicMock(), "Q1", {"query": "Q"}, "B3", MagicMock(), pre_contexts)
    assert res["status"] == "error"
    assert "Error occurred during generation" in res["generated_answer"]


@pytest.mark.asyncio
@patch("core.evaluator.CloudEvaluator.call_llm")
@patch("src.services.container.container.get_rag_service")
async def test_pipelined_stage_e2e_small_limit(mock_get_rag_service, mock_call_llm, tmp_path):
    import yaml
    import json
    from core.pipelined import run_pipelined_stage_async

    async def mock_call_llm_fn(system_prompt, user_prompt):
        return json.dumps({
            "answer_relevance": {"score": 0.9},
            "semantic_accuracy": {"score": 0.85}
        })
    mock_call_llm.side_effect = mock_call_llm_fn

    rag_service = MagicMock()
    rag_service.llm_engine.generate_response.return_value = "pipelined answer"
    rag_service.llm_engine.count_tokens.return_value = 12
    rag_service.llm_engine.unload_model = MagicMock()
    rag_service.retrieve_relevant_chunks.return_value = []
    rag_service.ask.return_value = "ask answer"
    rag_service.last_raw_response = "raw answer"
    mock_get_rag_service.return_value = rag_service

    mock_config = MagicMock()
    mock_config.data = {
        "llm": {
            "provider": "local",
            "local": {"model_path": "test_model"},
            "temp": 0.1,
            "max_tokens": 100
        },
        "embedding": {"model_name": "test_emb"},
        "rag_components": {"reranker": False, "citation_repair": False}
    }
    mock_config.rag_components = mock_config.data["rag_components"]
    mock_config.llm_model_max_context = 4096
    mock_config.reranker_model_name = "disabled"
    mock_config.llm_evaluation_concurrency = 2
    mock_config.llm_evaluation_rpm = 60
    mock_config.llm_evaluation_retries = 2
    mock_config.llm_cloud_api_key = "dummy_key"
    mock_config.llm_cloud_base_url = "https://api.openai.com/v1"
    mock_config.llm_cloud_model_name = "google/gemini-2.5-flash"

    dataset_path = tmp_path / "dataset.yaml"
    dataset_data = [
        {"id": "Q1", "query": "First query?", "expected_papers": []},
        {"id": "Q2", "query": "Second query?", "expected_papers": []}
    ]
    with open(dataset_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dataset_data, f)

    eval_results = tmp_path / "evaluation_results.yaml"
    metrics_results = tmp_path / "result_metrics.yaml"

    args = MagicMock()
    args.limit = 2
    args.consume_contexts = None
    args.cloud = False
    args.concurrency = 2
    args.rpm = 60
    args.retries = 2
    args.clear_checkpoint = True

    await run_pipelined_stage_async(
        args=args,
        config=mock_config,
        run_dir=tmp_path,
        dataset_path=dataset_path,
        baselines_to_run=["B0"],
        eval_results=eval_results,
        metrics_results=metrics_results,
        retrieved_contexts_file=None,
        total_steps=2
    )

    assert eval_results.exists()
    assert metrics_results.exists()

    with open(eval_results, "r", encoding="utf-8") as f:
        eval_data = yaml.safe_load(f)
    assert len(eval_data["results"]) == 2

    with open(metrics_results, "r", encoding="utf-8") as f:
        metrics_data = yaml.safe_load(f)
    assert len(metrics_data["results"]) == 2
    assert metrics_data["results"][0]["baselines"]["B0"]["eval_metrics"]["answer_relevance"] == 0.9


def test_run_pipeline_unanswerable_limit_arg_forwarding(tmp_path):
    import sys
    from unittest.mock import patch, MagicMock

    dataset_file = tmp_path / "test_ds.yaml"
    with open(dataset_file, "w", encoding="utf-8") as f:
        yaml.safe_dump([
            {"id": "Q1", "query": "q1", "is_answerable": True},
            {"id": "Q2", "query": "q2", "is_answerable": False}
        ], f)

    test_args = [
        "run_pipeline.py",
        "--dataset", str(dataset_file),
        "--limit", "1",
        "--unanswerable-limit", "1",
        "--skip-eval",
        "--no-unique-dir",
        "--output-dir", str(tmp_path)
    ]

    with patch.object(sys, "argv", test_args), \
         patch("core.subprocess_runner.run_command_with_progress", return_value=0.1) as mock_run_cmd:
        from run_pipeline import main
        try:
            main()
        except SystemExit:
            pass

        assert mock_run_cmd.called
        for call_args in mock_run_cmd.call_args_list:
            cmd = call_args[0][0]
            assert "--unanswerable-limit" in cmd
            idx = cmd.index("--unanswerable-limit")
            assert cmd[idx + 1] == "1"


def test_custom_baseline_gating_default_all(tmp_path):
    import sys
    from unittest.mock import patch
    from core.config import STANDARD_BASELINES

    dataset_file = tmp_path / "test_ds.yaml"
    with open(dataset_file, "w", encoding="utf-8") as f:
        yaml.safe_dump([{"id": "Q1", "query": "q1", "is_answerable": True}], f)

    test_args = [
        "run_pipeline.py",
        "--dataset", str(dataset_file),
        "--baselines", "all",
        "--skip-eval",
        "--no-unique-dir",
        "--output-dir", str(tmp_path)
    ]

    with patch.object(sys, "argv", test_args), \
         patch("run_pipeline.run_command_with_progress", return_value=0.1) as mock_run_cmd:
        from run_pipeline import main
        try:
            main()
        except SystemExit:
            pass

        assert mock_run_cmd.called
        # Check generation command baselines arg
        gen_call = [c for c in mock_run_cmd.call_args_list if c[0][3] == "generation"][0]
        cmd = gen_call[0][0]
        idx = cmd.index("--baselines")
        baselines_passed = cmd[idx + 1].split(",")
        assert baselines_passed == STANDARD_BASELINES
        assert "CUSTOM" not in baselines_passed


def test_pipeline_flag_alias_and_custom_inclusion(tmp_path):
    import sys
    from unittest.mock import patch

    dataset_file = tmp_path / "test_ds.yaml"
    with open(dataset_file, "w", encoding="utf-8") as f:
        yaml.safe_dump([{"id": "Q1", "query": "q1", "is_answerable": True}], f)

    test_args = [
        "run_pipeline.py",
        "--dataset", str(dataset_file),
        "--pipeline", "CUSTOM,B4,B6",
        "--skip-eval",
        "--no-unique-dir",
        "--output-dir", str(tmp_path)
    ]

    with patch.object(sys, "argv", test_args), \
         patch("run_pipeline.run_command_with_progress", return_value=0.1) as mock_run_cmd:
        from run_pipeline import main
        try:
            main()
        except SystemExit:
            pass

        assert mock_run_cmd.called
        gen_call = [c for c in mock_run_cmd.call_args_list if c[0][3] == "generation"][0]
        cmd = gen_call[0][0]
        idx = cmd.index("--baselines")
        baselines_passed = cmd[idx + 1].split(",")
        assert "CUSTOM" in baselines_passed
        assert "B4" in baselines_passed
        assert "B6" in baselines_passed






