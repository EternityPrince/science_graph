import pytest
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from core.metrics import (
    normalize_optional_text,
    get_is_answerable,
    detect_abstention,
    classify_answerability
)
from core.evaluator import evaluate_baseline_case
from core.analytics import analyze_metrics
from parse_metrics import print_confusion_matrix_and_metrics_tables


def test_dataset_normalization():
    # Test normalize_optional_text
    assert normalize_optional_text(None) == ""
    assert normalize_optional_text("") == ""
    assert normalize_optional_text("   hello   ") == "hello"
    assert normalize_optional_text(123) == "123"

    # Test get_is_answerable
    assert get_is_answerable({"is_answerable": True}) is True
    assert get_is_answerable({"is_answerable": False}) is False
    assert get_is_answerable({"is_answerable": "True"}) is True
    assert get_is_answerable({"is_answerable": "False"}) is False
    assert get_is_answerable({"is_answerable": None}) is True
    assert get_is_answerable({}) is True  # defaults to True


def test_abstention_detection():
    # Positive examples
    assert detect_abstention("The answer is UNANSWERABLE.") is True
    assert detect_abstention("Not enough information to answer this question.") is True
    assert detect_abstention("The provided context does not contain info.") is True
    assert detect_abstention("Увы, нет информации.") is True
    assert detect_abstention("Недостаточно информации в тексте.") is True
    assert detect_abstention("") is True
    assert detect_abstention("   ") is True

    # Negative examples
    assert detect_abstention("The paper shows that the adapt wavelet method works.") is False
    # Quote word in unrelated context
    assert detect_abstention("The author states: 'this question was previously unanswerable, but we solve it by doing X'.") is False


def test_evaluation_outcomes():
    assert classify_answerability(True, False) == "TP"
    assert classify_answerability(True, True) == "FN"
    assert classify_answerability(False, True) == "TN"
    assert classify_answerability(False, False) == "FP"


@pytest.mark.asyncio
async def test_judge_skipping_and_metrics():
    # Set up evaluator mock
    evaluator = MagicMock()
    evaluator.evaluate_all_metrics = AsyncMock(return_value={
        "answer_relevance": {"score": 0.8},
        "semantic_accuracy": {"score": 0.9},
        "faithfulness": {"score": 0.75},
        "citation_fidelity": {"score": 0.85}
    })

    prompts = {
        "unified_with_context_evaluator": {
            "system_prompt": "sys",
            "user_prompt_template": "user {query} {golden_answer} {answer} {context}"
        }
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        chk_path = Path(tmpdir) / "checkpoint.json"

        # Case 1: TP - Answerable answered. LLM judge should be called.
        res_tp = await evaluate_baseline_case(
            evaluator=evaluator,
            prompts=prompts,
            case_id="Q01",
            query="What is X?",
            golden_answer="X is a framework.",
            expected_papers=["paper1"],
            baseline_name="B6",
            baseline_data={
                "generated_answer": "X is indeed a framework.",
                "retrieved_papers": ["paper1"],
                "retrieved_chunks": [{"paper_id": "paper1", "page_number": 1, "text_content": "X is a framework."}],
                "status": "success"
            },
            checkpoint_data={},
            checkpoint_path=chk_path,
            is_answerable=True
        )
        assert res_tp["answerability_outcome"] == "TP"
        assert evaluator.evaluate_all_metrics.called is True
        assert res_tp["semantic_accuracy"] == 0.9
        assert res_tp["ar_sa_f1"] > 0.0

        evaluator.evaluate_all_metrics.reset_mock()

        # Case 2: FN - Answerable abstained. LLM judge should NOT be called and metrics penalized to 0.0.
        res_fn = await evaluate_baseline_case(
            evaluator=evaluator,
            prompts=prompts,
            case_id="Q02",
            query="What is Y?",
            golden_answer="Y is a protocol.",
            expected_papers=["paper2"],
            baseline_name="B6",
            baseline_data={
                "generated_answer": "UNANSWERABLE: Context does not contain Y.",
                "retrieved_papers": ["paper2"],
                "retrieved_chunks": [{"paper_id": "paper2", "page_number": 1, "text_content": "Context details Z."}],
                "status": "success"
            },
            checkpoint_data={},
            checkpoint_path=chk_path,
            is_answerable=True
        )
        assert res_fn["answerability_outcome"] == "FN"
        assert evaluator.evaluate_all_metrics.called is False
        assert res_fn["semantic_accuracy"] == 0.0
        assert res_fn["ar_sa_f1"] == 0.0

        # Case 3: TN - Unanswerable abstained. LLM judge should NOT be called and quality metrics set to None.
        res_tn = await evaluate_baseline_case(
            evaluator=evaluator,
            prompts=prompts,
            case_id="Q03",
            query="What is Z?",
            golden_answer="",
            expected_papers=[],
            baseline_name="B6",
            baseline_data={
                "generated_answer": "cannot answer",
                "retrieved_papers": ["paper3"],
                "retrieved_chunks": [{"paper_id": "paper3", "page_number": 1, "text_content": "Context details A."}],
                "status": "success"
            },
            checkpoint_data={},
            checkpoint_path=chk_path,
            is_answerable=False
        )
        assert res_tn["answerability_outcome"] == "TN"
        assert evaluator.evaluate_all_metrics.called is False
        assert res_tn["semantic_accuracy"] is None
        assert res_tn["ar_sa_f1"] is None

        # Case 4: FP - Unanswerable answered. LLM judge should NOT be called and quality metrics set to 0.0 (hallucinated).
        res_fp = await evaluate_baseline_case(
            evaluator=evaluator,
            prompts=prompts,
            case_id="Q04",
            query="What is Z?",
            golden_answer="",
            expected_papers=[],
            baseline_name="B6",
            baseline_data={
                "generated_answer": "Z is a standard library.",
                "retrieved_papers": ["paper4"],
                "retrieved_chunks": [{"paper_id": "paper4", "page_number": 1, "text_content": "Context details B."}],
                "status": "success"
            },
            checkpoint_data={},
            checkpoint_path=chk_path,
            is_answerable=False
        )
        assert res_fp["answerability_outcome"] == "FP"
        assert evaluator.evaluate_all_metrics.called is False
        assert res_fp["semantic_accuracy"] is None
        assert res_fp["ar_sa_f1"] is None


def test_metrics_aggregation():
    # Setup YAML format data containing results for 4 cases above
    data = {
        "metadata": {
            "date": "2026-07-05 20:00:00",
            "llm": {"provider": "local", "model_name": "qwen3"}
        },
        "results": [
            {
                "id": "Q01",
                "query": "What is X?",
                "is_answerable": True,
                "golden_answer": "X is a framework.",
                "baselines": {
                    "B6": {
                        "generated_answer": "X is a framework.",
                        "status": "success",
                        "latency_sec": 1.5,
                        "eval_metrics": {
                            "is_answerable": True,
                            "predicted_abstained": False,
                            "answerability_outcome": "TP",
                            "semantic_accuracy": 0.9,
                            "answer_relevance": 0.8,
                            "ar_sa_f1": 0.85
                        }
                    }
                }
            },
            {
                "id": "Q02",
                "query": "What is Y?",
                "is_answerable": True,
                "golden_answer": "Y is a protocol.",
                "baselines": {
                    "B6": {
                        "generated_answer": "cannot answer",
                        "status": "success",
                        "latency_sec": 0.5,
                        "eval_metrics": {
                            "is_answerable": True,
                            "predicted_abstained": True,
                            "answerability_outcome": "FN",
                            "semantic_accuracy": 0.0,
                            "answer_relevance": 0.0,
                            "ar_sa_f1": 0.0
                        }
                    }
                }
            },
            {
                "id": "Q03",
                "query": "What is Z?",
                "is_answerable": False,
                "golden_answer": "",
                "baselines": {
                    "B6": {
                        "generated_answer": "cannot answer",
                        "status": "success",
                        "latency_sec": 0.6,
                        "eval_metrics": {
                            "is_answerable": False,
                            "predicted_abstained": True,
                            "answerability_outcome": "TN",
                            "semantic_accuracy": None,
                            "answer_relevance": None,
                            "ar_sa_f1": None
                        }
                    }
                }
            },
            {
                "id": "Q04",
                "query": "What is W?",
                "is_answerable": False,
                "golden_answer": "",
                "baselines": {
                    "B6": {
                        "generated_answer": "Z is a library.",
                        "status": "success",
                        "latency_sec": 1.2,
                        "eval_metrics": {
                            "is_answerable": False,
                            "predicted_abstained": False,
                            "answerability_outcome": "FP",
                            "semantic_accuracy": 0.0,
                            "answer_relevance": 0.0,
                            "ar_sa_f1": 0.0
                        }
                    }
                }
            }
        ]
    }

    stats = analyze_metrics(data)
    
    # Assert classification metrics
    classification = stats["summary"]["B6"]["classification"]
    assert classification["TP"] == 1
    assert classification["FN"] == 1
    assert classification["TN"] == 1
    assert classification["FP"] == 1
    
    # Denominator > 0 metrics checks
    assert classification["accuracy"] == 0.5
    assert classification["precision"] == 0.5
    assert classification["recall"] == 0.5
    assert classification["specificity"] == 0.5
    assert classification["fpr"] == 0.5
    assert classification["fnr"] == 0.5
    assert classification["hallucination_rate"] == 0.5
    assert classification["answer_rate"] == 0.5
    assert classification["abstention_rate"] == 0.5

    # Check overall summary averages quality metrics ONLY over is_answerable: true questions
    # Semantic accuracy values for answerable questions: Q01=0.9, Q02=0.0
    # Average should be (0.9 + 0.0) / 2 = 0.45
    avg_sem = stats["summary"]["B6"]["semantic_accuracy"]["mean"]
    assert pytest.approx(avg_sem) == 0.45


def test_parse_metrics_confusion(capsys):
    # Verify backward compatibility / no crash on old/new runs
    data = {
        "results": [
            {
                "id": "Q01",
                "is_answerable": True,
                "baselines": {
                    "B6": {
                        "generated_answer": "X is a framework.",
                        "eval_metrics": {
                            "answerability_outcome": "TP"
                        }
                    }
                }
            },
            {
                "id": "Q02",
                "is_answerable": False,
                "baselines": {
                    "B6": {
                        "generated_answer": "cannot answer",
                        "eval_metrics": {
                            "answerability_outcome": "TN"
                        }
                    }
                }
            }
        ]
    }
    
    print_confusion_matrix_and_metrics_tables(data)
    captured = capsys.readouterr()
    assert "Confusion Matrix" in captured.out
    assert "Classification Quality Metrics" in captured.out
