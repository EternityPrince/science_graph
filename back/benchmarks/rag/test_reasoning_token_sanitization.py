import pytest
import re
import yaml
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.evaluator import evaluate_baseline_case, get_clean_judge_answer, _fallback_parse_reasoning_response
from core.reporting import save_judge_report, save_individual_judge_reports
from core.generation import run_benchmarking

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

def assert_no_reasoning_leak(text: str):
    FORBIDDEN_PATTERNS = [
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
    for pattern in FORBIDDEN_PATTERNS:
        assert not re.search(pattern, text, re.IGNORECASE), f"Leaked pattern: {pattern} in text: {text}"

# =========================================================================
# Test 1: evaluator sends only Final Answer to LLM judge
# =========================================================================

@pytest.mark.asyncio
async def test_evaluator_sends_only_final_answer_to_llm_judge(tmp_path):
    # Fake evaluator
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
        
    # Metrics are returned as number and saved in checkpoint
    assert res["answer_relevance"] == 0.9
    assert res["semantic_accuracy"] == 0.9
    assert res["faithfulness"] == 0.9
    assert res["citation_fidelity"] == 0.9
    
    assert checkpoint_path.exists()

# =========================================================================
# Test 2: raw generated_answer remains preserved in generation output
# =========================================================================

def test_raw_generated_answer_remains_preserved_in_generation_output(tmp_path):
    # Setup test smoke dataset with 1 case
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
    
    # Mock CLI args
    args = MagicMock()
    args.dataset = str(dataset_yaml)
    args.output = str(output_yaml)
    args.baselines = "B0"
    args.clear_checkpoint = True
    args.no_unique_dir = True
    args.consume_contexts = None
    args.cloud = False
    args.limit = None
    
    # Mock config
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
    
    # Mock prompts
    prompts_mock = MagicMock()
    
    # Mock container and RAG Service
    container_mock = MagicMock()
    mock_rag = MagicMock()
    container_mock.get_rag_service.return_value = mock_rag
    
    # Mock llm_engine behavior
    mock_llm = MagicMock()
    mock_llm.generate_response.return_value = RAW_ANSWER_WITH_REASONING
    mock_llm.count_tokens.return_value = 10
    mock_rag.llm_engine = mock_llm
    
    # Mock console con
    con_mock = MagicMock()
    
    # Run the benchmarking path
    run_benchmarking(args, config_mock, prompts_mock, container_mock, con_mock)
    
    # Verify main report output contains raw answer
    assert output_yaml.exists()
    with open(output_yaml, "r", encoding="utf-8") as f:
        output_data = yaml.safe_load(f)
        
    baseline_res = output_data["results"][0]["baselines"]["B0"]
    assert baseline_res["status"] == "success"
    
    # Check that generated_answer is raw and preserves reasoning
    gen_ans = baseline_res["generated_answer"]
    assert "### 1. _analysis" in gen_ans
    assert "### Final Answer:" in gen_ans
    assert "wavelet filtering" in gen_ans
    
    # Check that non-LLM metrics did not fail
    assert "retrieval_recall" in baseline_res
    assert "context_precision" in baseline_res

# =========================================================================
# Test 3: simplified judge reports must not leak reasoning sections
# =========================================================================

def test_simplified_judge_reports_must_not_leak_reasoning_sections(tmp_path):
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
    
    # Save the reports
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
    
    # In the original human_data, the raw generated answer should remain unchanged
    assert human_data["results"][0]["baselines"]["B0"]["generated_answer"] == RAW_ANSWER_WITH_REASONING

# =========================================================================
# Test 4: fallback parser handles real markdown reasoning format
# =========================================================================

def test_fallback_parser_handles_various_reasoning_formats():
    # Case 1: Full markdown format
    status, ans = _fallback_parse_reasoning_response(RAW_ANSWER_WITH_REASONING)
    assert_no_reasoning_leak(ans)
    assert "The text describes the adaptation of the Winograd method" in ans
    
    # Case 2: Markdown format without ### Final Answer:, but with ### 5. _answer...
    no_final_answer = """### 1. _analysis
    Analysis content.
    ### 5. _answer
    This is the fallback answer content.
    """
    status, ans = _fallback_parse_reasoning_response(no_final_answer)
    assert_no_reasoning_leak(ans)
    assert "This is the fallback answer content." in ans
    
    # Case 3: XML/special format
    xml_format = """<|status_start|>ANSWERABLE<|status_end|>
    <|answer_start|>clean answer content here<|answer_end|>
    """
    status, ans = _fallback_parse_reasoning_response(xml_format)
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
    status, ans = _fallback_parse_reasoning_response(mixed_noisy)
    assert_no_reasoning_leak(ans)
    assert "clean answer with inst" in ans
