import pytest
import re
import yaml
import json
import hashlib
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

from core.evaluator import (
    evaluate_baseline_case,
    get_clean_judge_answer,
    run_evaluation
)
from core.reporting import save_judge_report, save_individual_judge_reports
from core.generation import run_benchmarking
from core.sanitization import extract_clean_answer

FORBIDDEN_CLEAN_ANSWER_PATTERNS = [
    r"###\s*1\.\s*_analysis",
    r"###\s*2\.\s*_start",
    r"###\s*3\.\s*_reasoning",
    r"###\s*4\.\s*_status",
    r"###\s*5\.\s*_answer",
    r"###\s*Final\s+Answer\s*:?",
    r"<think\b",
    r"</think>",
    r"<\|query_analysis",
    r"<\|source_analysis",
    r"<\|reasoning",
    r"<\|status_",
    r"<\|answer_",
    r"<\|im_",
    r"<\|eot_id\|>",
    r"\[INST\]",
    r"\[/INST\]",
]

def assert_no_reasoning_leak(text: str) -> None:
    for pattern in FORBIDDEN_CLEAN_ANSWER_PATTERNS:
        assert not re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL), f"Forbidden pattern '{pattern}' found in text: {text}"

# Raw answer containing reasoning sections
RAW_ANSWER_WITH_REASONING = """### 1. _analysis

The user asks for wavelet filtering details.

### 2. _start

The sources contain wavelet adaptation.

### 3. _reasoning

- Source 1 Winograd method adaptation.
- Source 2 area-delay product optimization.

### 4. _status

The question asks for configuration.

### 5. _answer

The text describes adaptation of Winograd method but not specific configurations.

### Final Answer:

The text describes the adaptation of the Winograd method for wavelet filtering
with decimation to lower computational delay and area-delay product. It mentions
general performance metrics (34-63% and 39-66%) and the best efficiency ratio
(K(3,4,2) for 4th order, K(5,6,2) for 6th order), but it does not provide
the specific K configurations that ensured the best efficiency or the exact
percentage reduction for those configurations. Therefore, the specific details
requested are not present in the provided sources."""

# =========================================================================
# Test 1: evaluator sends only clean Final Answer to LLM judge
# =========================================================================

@pytest.mark.asyncio
async def test_evaluator_sends_only_clean_answer(tmp_path):
    mock_evaluator = MagicMock()
    eval_calls = []
    
    async def mock_evaluate_all_metrics(evaluator_config, has_context, **kwargs):
        eval_calls.append(("all", kwargs))
        res = {
            "answer_relevance": {"score": 0.9},
            "semantic_accuracy": {"score": 0.9}
        }
        if has_context:
            res["faithfulness"] = {"score": 0.9}
            res["citation_fidelity"] = {"score": 0.9}
        return res
        
    mock_evaluator.evaluate_all_metrics = mock_evaluate_all_metrics
    
    prompts = {
        "unified_with_context_evaluator": {"system_prompt": "sys", "user_prompt_template": "user {answer}"},
        "unified_without_context_evaluator": {"system_prompt": "sys", "user_prompt_template": "user {answer}"}
    }
    
    baseline_data = {
        "generated_answer": RAW_ANSWER_WITH_REASONING,
        "retrieved_papers": ["paper1"],
        "retrieved_chunks": [
            {
                "paper_id": "paper1",
                "page_number": 1,
                "text_content": "Some text content here"
            }
        ]
    }
    
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_data = {}
    
    res = await evaluate_baseline_case(
        evaluator=mock_evaluator,
        prompts=prompts,
        case_id="Q01",
        query="What is the adaptation?",
        golden_answer="A golden answer",
        expected_papers=["paper1"],
        baseline_name="B1",
        baseline_data=baseline_data,
        checkpoint_data=checkpoint_data,
        checkpoint_path=checkpoint_path,
        max_input_token=10000
    )
    
    assert len(eval_calls) == 1
    _, kwargs = eval_calls[0]
    assert "answer" in kwargs
    ans_text = kwargs["answer"]
    assert_no_reasoning_leak(ans_text)
    assert "The text describes the adaptation of the Winograd method" in ans_text
        
    assert res["answer_relevance"] == 0.9
    assert res["semantic_accuracy"] == 0.9
    assert res["faithfulness"] == 0.9
    assert res["citation_fidelity"] == 0.9

# =========================================================================
# Test 2: raw generated_answer remains preserved in benchmark output
# =========================================================================

def test_raw_generated_answer_preserved(tmp_path):
    dataset_yaml = tmp_path / "test_smoke.yaml"
    test_dataset = [
        {
            "id": "Q01",
            "category": "general",
            "query": "Is Winograd method adapted?",
            "golden_answer": "Yes",
            "expected_papers": ["paper1"]
        }
    ]
    with open(dataset_yaml, "w", encoding="utf-8") as f:
        yaml.dump(test_dataset, f)
        
    output_yaml = tmp_path / "evaluation_results.yaml"
    
    args = MagicMock()
    args.dataset = str(dataset_yaml)
    args.output = str(output_yaml)
    args.baselines = "B0"
    args.clear_checkpoint = True
    args.no_unique_dir = True
    args.consume_contexts = None
    args.cloud = False
    args.limit = None
    
    class MockConfig:
        def __init__(self):
            self.data = {
                "llm": {
                    "provider": "local",
                    "local": {"model_path": "mock_local_model"},
                    "temp": 0.1,
                    "max_tokens": 1000
                },
                "embedding": {
                    "model_name": "mock_emb_model"
                },
                "rag_components": {
                    "reranker": False
                }
            }
            self.llm_model_max_context = 4096
            self.rag_components = {
                "lexical_search": False,
                "dense_search": False,
                "graph_expansion": False,
                "reranker": False,
                "hyde": False,
                "intent_classifier": False,
                "citation_repair": False
            }
            self.reranker_model_name = "mock_reranker"
        def is_component_enabled(self, name):
            return False
            
    config_mock = MockConfig()
    prompts_mock = MagicMock()
    container_mock = MagicMock()
    mock_rag = MagicMock()
    container_mock.get_rag_service.return_value = mock_rag
    
    mock_llm = MagicMock()
    mock_llm.generate_response.return_value = RAW_ANSWER_WITH_REASONING
    mock_llm.count_tokens.return_value = 10
    mock_rag.llm_engine = mock_llm
    
    con_mock = MagicMock()
    
    run_benchmarking(args, config_mock, prompts_mock, container_mock, con_mock)
    
    assert output_yaml.exists()
    with open(output_yaml, "r", encoding="utf-8") as f:
        output_data = yaml.safe_load(f)
        
    baseline_res = output_data["results"][0]["baselines"]["B0"]
    assert baseline_res["status"] == "success"
    
    gen_ans = baseline_res["generated_answer"]
    assert "### 1. _analysis" in gen_ans
    assert "### Final Answer:" in gen_ans
    assert "wavelet filtering" in gen_ans
    
    assert "retrieval_recall" in baseline_res
    assert "context_precision" in baseline_res

# =========================================================================
# Test 3: citation repair does not overwrite raw generated_answer
# =========================================================================

def test_citation_repair_does_not_overwrite_raw_generated_answer(tmp_path):
    dataset_yaml = tmp_path / "test_smoke.yaml"
    test_dataset = [
        {
            "id": "Q01",
            "category": "general",
            "query": "Is Winograd method adapted?",
            "golden_answer": "Yes",
            "expected_papers": ["paper1"]
        }
    ]
    with open(dataset_yaml, "w", encoding="utf-8") as f:
        yaml.dump(test_dataset, f)
        
    pre_contexts_yaml = tmp_path / "pre_contexts.yaml"
    pre_contexts_data = [
        {
            "id": "Q01",
            "baselines": {
                "B1": {
                    "status": "success",
                    "retrieved_papers": ["paper1"],
                    "retrieved_chunks": [
                        {
                            "id": "ch1",
                            "paper_id": "paper1",
                            "text_content": "Wavelet Winograd method details.",
                            "page_number": 1
                        }
                    ],
                    "trimmed_text": "trimmed text content",
                    "trimmed_graph": "trimmed graph content",
                    "enrichment_block": "",
                    "metrics": {
                        "components": {}
                    },
                    "latency_sec": 1.2
                }
            }
        }
    ]
    with open(pre_contexts_yaml, "w", encoding="utf-8") as f:
        yaml.dump(pre_contexts_data, f)
        
    output_yaml = tmp_path / "evaluation_results.yaml"
    
    args = MagicMock()
    args.dataset = str(dataset_yaml)
    args.output = str(output_yaml)
    args.baselines = "B1"
    args.clear_checkpoint = True
    args.no_unique_dir = True
    args.consume_contexts = str(pre_contexts_yaml)
    args.cloud = False
    args.limit = None
    
    class MockConfig:
        def __init__(self):
            self.data = {
                "llm": {
                    "provider": "local",
                    "local": {"model_path": "mock_local_model"},
                    "temp": 0.1,
                    "max_tokens": 1000
                },
                "embedding": {
                    "model_name": "mock_emb_model"
                },
                "rag_components": {
                    "reranker": False,
                    "citation_repair": True
                }
            }
            self.llm_model_max_context = 4096
            self.rag_components = {
                "lexical_search": False,
                "dense_search": False,
                "graph_expansion": False,
                "reranker": False,
                "hyde": False,
                "intent_classifier": False,
                "citation_repair": True
            }
            self.reranker_model_name = "mock_reranker"
        def is_component_enabled(self, name):
            return True if name == "citation_repair" else False
            
    config_mock = MockConfig()
    prompts_mock = MagicMock()
    container_mock = MagicMock()
    mock_rag = MagicMock()
    container_mock.get_rag_service.return_value = mock_rag
    
    mock_llm = MagicMock()
    mock_llm.generate_response.return_value = RAW_ANSWER_WITH_REASONING
    mock_llm.count_tokens.return_value = 10
    mock_rag.llm_engine = mock_llm
    
    # Mock citation repair to return modified text
    mock_rag._validate_and_repair_citations.return_value = "REPAIRED CLEAN ANSWER"
    
    con_mock = MagicMock()
    
    run_benchmarking(args, config_mock, prompts_mock, container_mock, con_mock)
    
    assert output_yaml.exists()
    with open(output_yaml, "r", encoding="utf-8") as f:
        output_data = yaml.safe_load(f)
        
    baseline_res = output_data["results"][0]["baselines"]["B1"]
    assert baseline_res["status"] == "success"
    
    gen_ans = baseline_res["generated_answer"]
    # It must contain the raw response, not the repaired clean answer
    assert "### 1. _analysis" in gen_ans
    assert "REPAIRED CLEAN ANSWER" not in gen_ans

# =========================================================================
# Test 4: simplified judge reports do not leak reasoning sections
# =========================================================================

def test_simplified_judge_reports_do_not_leak_reasoning(tmp_path):
    human_data = {
        "results": [
            {
                "id": "Q01",
                "query": "Is Winograd method adapted?",
                "golden_answer": "Yes",
                "baselines": {
                    "B0": {
                        "generated_answer": RAW_ANSWER_WITH_REASONING
                    }
                }
            }
        ]
    }
    
    judge_output_path = tmp_path / "evaluation_results_judge.yaml"
    
    save_judge_report(human_data, judge_output_path)
    save_individual_judge_reports(human_data, tmp_path, "evaluation_results", ".yaml")
    
    # Check single judge report
    assert judge_output_path.exists()
    with open(judge_output_path, "r", encoding="utf-8") as f:
        judge_data = yaml.safe_load(f)
        
    judge_ans = judge_data["results"][0]["baselines"]["B0"]["generated_answer"]
    assert_no_reasoning_leak(judge_ans)
    assert "The text describes the adaptation of the Winograd method" in judge_ans
    
    # Check individual reports
    indiv_path = tmp_path / "baselines" / "evaluation_results_judge_b0.yaml"
    assert indiv_path.exists()
    with open(indiv_path, "r", encoding="utf-8") as f:
        indiv_data = yaml.safe_load(f)
        
    indiv_ans = indiv_data["results"][0]["baselines"]["B0"]["generated_answer"]
    assert_no_reasoning_leak(indiv_ans)
    assert "The text describes the adaptation of the Winograd method" in indiv_ans
    
    # The original human_data dict should not be mutated
    assert human_data["results"][0]["baselines"]["B0"]["generated_answer"] == RAW_ANSWER_WITH_REASONING

# =========================================================================
# Test 5: fallback parser handles real markdown reasoning format
# =========================================================================

def test_fallback_parser_handles_real_markdown_reasoning_format():
    # Case 1: Full markdown format
    status, ans = extract_clean_answer(RAW_ANSWER_WITH_REASONING)
    assert_no_reasoning_leak(ans)
    assert "The text describes the adaptation of the Winograd method" in ans
    
    # Case 2: Markdown format without ### Final Answer:, but with ### 5. _answer...
    no_final_answer = """### 1. _analysis
    Analysis content.
    ### 5. _answer
    This is the fallback answer content.
    """
    status, ans = extract_clean_answer(no_final_answer)
    assert_no_reasoning_leak(ans)
    assert "This is the fallback answer content." in ans
    
    # Case 3: XML/special format
    xml_format = """<|status_start|>ANSWERABLE<|status_end|>
    <|answer_start|>clean answer content here<|answer_end|>
    """
    status, ans = extract_clean_answer(xml_format)
    assert status == "ANSWERABLE"
    assert ans == "clean answer content here"
    
    # Case 4: Mixed noisy format
    mixed_noisy = """<think>some private thought</think>
    [INST]
    <|im_start|>
    ### 1. _analysis
    Analysis logic.
    ### Final Answer: clean answer with inst
    [/INST]
    <|im_end|>
    """
    status, ans = extract_clean_answer(mixed_noisy)
    assert_no_reasoning_leak(ans)
    assert "clean answer with inst" in ans

# =========================================================================
# Test 6: evaluator skips LLM judge calls when clean answer is empty
# =========================================================================

@pytest.mark.asyncio
async def test_evaluator_skips_llm_judge_calls_when_clean_answer_is_empty(tmp_path):
    mock_evaluator = MagicMock()
    eval_calls = []
    
    async def mock_evaluate_all_metrics(evaluator_config, has_context, **kwargs):
        eval_calls.append(("all", kwargs))
        res = {
            "answer_relevance": {"score": 0.9},
            "semantic_accuracy": {"score": 0.9}
        }
        if has_context:
            res["faithfulness"] = {"score": 0.9}
            res["citation_fidelity"] = {"score": 0.9}
        return res
        
    mock_evaluator.evaluate_all_metrics = mock_evaluate_all_metrics
    
    prompts = {
        "unified_with_context_evaluator": {"system_prompt": "sys", "user_prompt_template": "user {answer}"},
        "unified_without_context_evaluator": {"system_prompt": "sys", "user_prompt_template": "user {answer}"}
    }
    
    # Answer contains reasoning, but clean final answer will be empty
    generated_answer_empty = """### 1. _analysis
    Some analysis.
    ### 3. _reasoning
    Some reasoning.
    """
    
    baseline_data = {
        "generated_answer": generated_answer_empty,
        "retrieved_papers": ["paper1"],
        "retrieved_chunks": [
            {
                "paper_id": "paper1",
                "page_number": 1,
                "text_content": "Some text content here"
            }
        ]
    }
    
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_data = {}
    
    res = await evaluate_baseline_case(
        evaluator=mock_evaluator,
        prompts=prompts,
        case_id="Q01",
        query="What is the adaptation?",
        golden_answer="A golden answer",
        expected_papers=["paper1"],
        baseline_name="B1",
        baseline_data=baseline_data,
        checkpoint_data=checkpoint_data,
        checkpoint_path=checkpoint_path,
        max_input_token=10000
    )
    
    # LLM judge calls are skipped entirely
    assert len(eval_calls) == 0
    # Deterministic metrics are still returned
    assert res["retrieval_recall"] == 1.0
    assert checkpoint_path.exists()

# =========================================================================
# Test 7: result_metrics.yaml preserves retrieved_chunks and token metadata
# =========================================================================

@pytest.mark.asyncio
@patch("core.evaluator.get_cloud_credentials")
@patch("core.evaluator.CloudEvaluator")
async def test_result_metrics_yaml_preserves_retrieved_chunks(mock_eval_class, mock_get_creds, tmp_path):
    mock_get_creds.return_value = ("api_key", "base_url", "model_name")
    
    mock_evaluator = MagicMock()
    mock_eval_class.return_value = mock_evaluator
    
    # Mock evaluate_all_metrics to return deterministic dicts
    async def mock_evaluate_all_metrics(evaluator_config, has_context, **kwargs):
        res = {
            "answer_relevance": {"score": 0.9},
            "semantic_accuracy": {"score": 0.9}
        }
        if has_context:
            res["faithfulness"] = {"score": 0.9}
            res["citation_fidelity"] = {"score": 0.9}
        return res
    mock_evaluator.evaluate_all_metrics = mock_evaluate_all_metrics
    
    input_yaml = tmp_path / "evaluation_results.yaml"
    output_yaml = tmp_path / "result_metrics.yaml"
    
    # Mock input data including retrieved_chunks and token metadata
    chunks_data = [
        {
            "id": "chunk_1",
            "paper_id": "paper1",
            "page_number": 1,
            "text_content": "Wavelet Winograd method details.",
            "score": 0.95
        }
    ]
    input_data = {
        "metadata": {
            "llm": {"model_max_context": 4096}
        },
        "results": [
            {
                "id": "Q01",
                "query": "Is Winograd method adapted?",
                "golden_answer": "Yes",
                "expected_papers": ["paper1"],
                "baselines": {
                    "B1": {
                        "status": "success",
                        "latency_sec": 1.5,
                        "retrieved_papers": ["paper1"],
                        "generated_answer": "Yes, Winograd method is adapted.",
                        "retrieved_chunks": chunks_data,
                        "context_token": 1500,
                        "max_input_token": 4096,
                        "context_fillness": 0.366
                    }
                }
            }
        ]
    }
    with open(input_yaml, "w", encoding="utf-8") as f:
        yaml.dump(input_data, f)
        
    args = MagicMock()
    args.input = str(input_yaml)
    args.output = str(output_yaml)
    args.baselines = "all"
    args.concurrency = 1
    args.rpm = 60
    args.clear_checkpoint = True
    args.limit = None
    
    config = MagicMock()
    con = MagicMock()
    
    await run_evaluation(args, config, con)
    
    assert output_yaml.exists()
    with open(output_yaml, "r", encoding="utf-8") as f:
        output_report = yaml.safe_load(f)
        
    b1_output = output_report["results"][0]["baselines"]["B1"]
    
    # Assert retrieved_chunks and token metadata are preserved in the final output file
    assert "retrieved_chunks" in b1_output
    assert b1_output["retrieved_chunks"] == chunks_data
    assert b1_output["context_token"] == 1500
    assert b1_output["max_input_token"] == 4096
    assert b1_output["context_fillness"] == 0.366

# =========================================================================
# Test 8: checkpoint invalidates when generated_answer changes
# =========================================================================

@pytest.mark.asyncio
async def test_checkpoint_invalidates_when_generated_answer_changes(tmp_path):
    mock_evaluator = MagicMock()
    eval_calls = []
    
    async def mock_evaluate_all_metrics(evaluator_config, has_context, **kwargs):
        eval_calls.append("all")
        res = {
            "answer_relevance": {"score": 0.9},
            "semantic_accuracy": {"score": 0.9}
        }
        if has_context:
            res["faithfulness"] = {"score": 0.9}
            res["citation_fidelity"] = {"score": 0.9}
        return res
        
    mock_evaluator.evaluate_all_metrics = mock_evaluate_all_metrics
    
    prompts = {
        "unified_with_context_evaluator": {"system_prompt": "sys", "user_prompt_template": "user {answer}"},
        "unified_without_context_evaluator": {"system_prompt": "sys", "user_prompt_template": "user {answer}"}
    }
    
    baseline_data_1 = {
        "generated_answer": "First raw answer text.",
        "retrieved_papers": ["paper1"],
        "retrieved_chunks": []
    }
    
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_data = {}
    
    # First run with answer A
    await evaluate_baseline_case(
        evaluator=mock_evaluator,
        prompts=prompts,
        case_id="Q01",
        query="query",
        golden_answer="golden",
        expected_papers=["paper1"],
        baseline_name="B1",
        baseline_data=baseline_data_1,
        checkpoint_data=checkpoint_data,
        checkpoint_path=checkpoint_path,
        max_input_token=10000
    )
    
    assert len(eval_calls) == 1  # single call to evaluate all metrics
    eval_calls.clear()
    
    # Second run with different answer B (same case_id, same baseline_name)
    baseline_data_2 = {
        "generated_answer": "Second completely different answer text.",
        "retrieved_papers": ["paper1"],
        "retrieved_chunks": []
    }
    
    await evaluate_baseline_case(
        evaluator=mock_evaluator,
        prompts=prompts,
        case_id="Q01",
        query="query",
        golden_answer="golden",
        expected_papers=["paper1"],
        baseline_name="B1",
        baseline_data=baseline_data_2,
        checkpoint_data=checkpoint_data,
        checkpoint_path=checkpoint_path,
        max_input_token=10000
    )
    
    # Because generated_answer changed, payload hash changed, key invalidated.
    # LLM judge should be called again!
    assert len(eval_calls) == 1
